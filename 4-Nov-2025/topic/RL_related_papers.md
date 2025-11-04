# Learning to Seek Evidence: A Verifiable Reasoning Agent with Causal Faithfulness Analysis 

**Authors**: Yuhang Huang, Zekai Lin, Fan Zhong, Lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01425)  

**Abstract**: Explanations for AI models in high-stakes domains like medicine often lack verifiability, which can hinder trust. To address this, we propose an interactive agent that produces explanations through an auditable sequence of actions. The agent learns a policy to strategically seek external visual evidence to support its diagnostic reasoning. This policy is optimized using reinforcement learning, resulting in a model that is both efficient and generalizable. Our experiments show that this action-based reasoning process significantly improves calibrated accuracy, reducing the Brier score by 18\% compared to a non-interactive baseline. To validate the faithfulness of the agent's explanations, we introduce a causal intervention method. By masking the visual evidence the agent chooses to use, we observe a measurable degradation in its performance ($\Delta$Brier=+0.029), confirming that the evidence is integral to its decision-making process. Our work provides a practical framework for building AI systems with verifiable and faithful reasoning capabilities. 

---
# Aligning LLM agents with human learning and adjustment behavior: a dual agent approach 

**Authors**: Tianming Liu, Jirong Yang, Yafeng Yin, Manzi Li, Linghao Wang, Zheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00993)  

**Abstract**: Effective modeling of how human travelers learn and adjust their travel behavior from interacting with transportation systems is critical for system assessment and planning. However, this task is also difficult due to the complex cognition and decision-making involved in such behavior. Recent research has begun to leverage Large Language Model (LLM) agents for this task. Building on this, we introduce a novel dual-agent framework that enables continuous learning and alignment between LLM agents and human travelers on learning and adaptation behavior from online data streams. Our approach involves a set of LLM traveler agents, equipped with a memory system and a learnable persona, which serve as simulators for human travelers. To ensure behavioral alignment, we introduce an LLM calibration agent that leverages the reasoning and analytical capabilities of LLMs to train the personas of these traveler agents. Working together, this dual-agent system is designed to track and align the underlying decision-making mechanisms of travelers and produce realistic, adaptive simulations. Using a real-world dataset from a day-to-day route choice experiment, we show our approach significantly outperforms existing LLM-based methods in both individual behavioral alignment and aggregate simulation accuracy. Furthermore, we demonstrate that our method moves beyond simple behavioral mimicry to capture the evolution of underlying learning processes, a deeper alignment that fosters robust generalization. Overall, our framework provides a new approach for creating adaptive and behaviorally realistic agents to simulate travelers' learning and adaptation that can benefit transportation simulation and policy analysis. 

---
# Do Math Reasoning LLMs Help Predict the Impact of Public Transit Events? 

**Authors**: Bowen Fang, Ruijian Zha, Xuan Di  

**Link**: [PDF](https://arxiv.org/pdf/2511.00808)  

**Abstract**: Predicting public transit incident duration from unstructured text alerts is a critical but challenging task. Addressing the domain sparsity of transit operations with standard Supervised Fine-Tuning (SFT) is difficult, as the task involves noisy, continuous labels and lacks reliable expert demonstrations for reasoning. While Reinforcement Learning from Verifiable Rewards (RLVR) excels at tasks with binary correctness, like mathematics, its applicability to noisy, continuous forecasting is an open question. This work, to our knowledge, is the first to bridge the gap between RLVR LLM training with the critical, real-world forecasting challenges in public transit operations. We adapt RLVR to this task by introducing a tolerance-based, shaped reward function that grants partial credit within a continuous error margin, rather than demanding a single correct answer. We systematically evaluate this framework on a curated dataset of NYC MTA service alerts. Our findings show that general-purpose, instruction-tuned LLMs significantly outperform specialized math-reasoning models, which struggle with the ambiguous, real-world text. We empirically demonstrate that the binary reward is unstable and degrades performance, whereas our shaped reward design is critical and allows our model to dominate on the most challenging metrics. While classical regressors are superior at minimizing overall MAE or MSE, our RLVR approach achieved a 35\% relative improvement in 5-minute accuracy (Acc@5) over the strongest baseline. This demonstrates that RLVR can be successfully adapted to real-world, noisy forecasting, but requires a verifier design that reflects the continuous nature of the problem. 

---
# Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries 

**Authors**: Minghe Shen, Zhuo Zhi, Chonghan Liu, Shuo Xing, Zhengzhong Tu, Che Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00710)  

**Abstract**: While Vision-Language Models (VLMs) post-trained with Reinforcement Learning (RL) show impressive general reasoning, their evaluation is often confined to language-dominant tasks (e.g., math). This raises a critical question: can RL post-training truly extend the inherent capability boundary of a base VLM, particularly for visual-centric spatial tasks where it initially fails? To investigate this, we introduce Ariadne, a framework utilizing synthetic mazes for multi-step spatial reasoning where task difficulty (e.g., path length, turns) is precisely controlled. We leverage this controllable environment to train VLMs using Reinforcement Learning with Verified Rewards (RLVR) in a difficulty-aware curriculum. Surprisingly, post-RLVR training, the VLM achieves over 50% accuracy on a problem set where the base model scored 0%, demonstrating that our approach expands the model's initial capability boundary. To assess real-world viability, we evaluate out-of-distribution (OOD) generalization on practical benchmarks. Despite training only on synthetic maze samples, Ariadne achieves significant zero-shot improvements, averaging 16% on MapBench (e.g., museum navigation) and 24% on ReasonMap (subway transfer tasks). These results confirm that our method not only broadens the model's fundamental limits but also enhances its generalization to real-world spatial reasoning. We acknowledge our study is limited to the post-training phase, given the opaqueness of pre-training data, and hope our research motivates further work on specialized, capability-extending alignment. 

---
# Reimagining Safety Alignment with An Image 

**Authors**: Yifan Xia, Guorui Chen, Wenqian Yu, Zhijiang Li, Philip Torr, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00509)  

**Abstract**: Large language models (LLMs) excel in diverse applications but face dual challenges: generating harmful content under jailbreak attacks and over-refusal of benign queries due to rigid safety mechanisms. These issues are further complicated by the need to accommodate different value systems and precisely align with given safety preferences. Moreover, traditional methods like SFT and RLHF lack this capability due to their costly parameter tuning requirements and inability to support multiple value systems within a single model. These problems are more obvious in multimodal large language models (MLLMs), especially in terms of heightened over-refusal in cross-modal tasks and new security risks arising from expanded attack surfaces. We propose Magic Image, an optimization-driven visual prompt framework that enhances security while reducing over-refusal. By optimizing image prompts using harmful/benign samples, our method enables a single model to adapt to different value systems and better align with given safety preferences without parameter updates. Experiments demonstrate improved safety-effectiveness balance across diverse datasets while preserving model performance, offering a practical solution for deployable MLLM safety alignment. 

---
# GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining 

**Authors**: Chunyu Wei, Wenji Hu, Xingjia Hao, Xin Wang, Yifan Yang, Yueguo Chen, Yang Tian, Yunhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00457)  

**Abstract**: Large Language Models (LLMs) face significant limitations when applied to large-scale graphs, struggling with context constraints and inflexible reasoning. We present GraphChain, a framework that enables LLMs to analyze complex graphs through dynamic sequences of specialized tools, mimicking human exploratory intelligence. Our approach introduces two key innovations: (1) Progressive Graph Distillation, a reinforcement learning mechanism that generates optimized tool sequences balancing task relevance with information compression, and (2) Structure-aware Test-Time Adaptation, which efficiently tailors tool selection strategies to diverse graph topologies using spectral properties and lightweight adapters without costly retraining. Experiments show GraphChain significantly outperforms prior methods, enabling scalable and adaptive LLM-driven graph analysis. 

---
# PreferThinker: Reasoning-based Personalized Image Preference Assessment 

**Authors**: Shengqi Xu, Xinpeng Zhou, Yabo Zhang, Ming Liu, Tao Liang, Tianyu Zhang, Yalong Bai, Zuxuan Wu, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00609)  

**Abstract**: Personalized image preference assessment aims to evaluate an individual user's image preferences by relying only on a small set of reference images as prior information. Existing methods mainly focus on general preference assessment, training models with large-scale data to tackle well-defined tasks such as text-image alignment. However, these approaches struggle to handle personalized preference because user-specific data are scarce and not easily scalable, and individual tastes are often diverse and complex. To overcome these challenges, we introduce a common preference profile that serves as a bridge across users, allowing large-scale user data to be leveraged for training profile prediction and capturing complex personalized preferences. Building on this idea, we propose a reasoning-based personalized image preference assessment framework that follows a \textit{predict-then-assess} paradigm: it first predicts a user's preference profile from reference images, and then provides interpretable, multi-dimensional scores and assessments of candidate images based on the predicted profile. To support this, we first construct a large-scale Chain-of-Thought (CoT)-style personalized assessment dataset annotated with diverse user preference profiles and high-quality CoT-style reasoning, enabling explicit supervision of structured reasoning. Next, we adopt a two-stage training strategy: a cold-start supervised fine-tuning phase to empower the model with structured reasoning capabilities, followed by reinforcement learning to incentivize the model to explore more reasonable assessment paths and enhance generalization. Furthermore, we propose a similarity-aware prediction reward to encourage better prediction of the user's preference profile, which facilitates more reasonable assessments exploration. Extensive experiments demonstrate the superiority of the proposed method. 

---
# RLAC: Reinforcement Learning with Adversarial Critic for Free-Form Generation Tasks 

**Authors**: Mian Wu, Gavin Zhang, Sewon Min, Sergey Levine, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.01758)  

**Abstract**: Open-ended generation tasks require outputs to satisfy diverse and often implicit task-specific evaluation rubrics. The sheer number of relevant rubrics leads to prohibitively high verification costs and incomplete assessments of a response, making reinforcement learning (RL) post-training with rubric-based rewards difficult to scale. This problem is exacerbated by the fact that often the best way to combine these rubrics into one single reward is also highly prompt-specific. We propose Reinforcement Learning with Adversarial Critic (RLAC), a post-training approach that addresses these challenges via dynamic rubric verification. Our approach employs a large language model (LLM) as a critic that dynamically identifies only the most likely failure modes (e.g., a factual error or unhandled edge case), which are then verified by an external validator to optimize both generator and critic jointly. By training both the generator and the critic, this game enhances the critic's error detection and the generator's output quality while reducing required verifications. Our experiments demonstrate that RLAC improves factual accuracy in text generation and correctness in code generation, while also outperforming exhaustive verification and reward model methods. We show that dynamic critics are more effective than fixed critics, showcasing the potential of RLAC for scaling RL post-training to free-form generation tasks. 

---
# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI 

**Authors**: Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.01689)  

**Abstract**: The character of the "AI assistant" persona generated by modern chatbot large language models influences both surface-level behavior and apparent values, beliefs, and ethics. These all affect interaction quality, perceived intelligence, and alignment with both developer and user intentions. The shaping of this persona, known as character training, is a critical component of industry post-training, yet remains effectively unstudied in the academic literature. We introduce the first open implementation of character training, leveraging Constitutional AI and a new data pipeline using synthetic introspective data to shape the assistant persona in a more effective and controlled manner than alternatives such as constraining system prompts or activation steering. Specifically, we fine-tune three popular open-weights models using 11 example personas, such as humorous, deeply caring, or even malevolent. To track the effects of our approach, we introduce a method which analyzes revealed preferences, uncovering clear and holistic changes in character. We find these changes are more robust to adversarial prompting than the above two alternatives, while also leading to more coherent and realistic generations. Finally, we demonstrate this fine-tuning has little to no effect on general capabilities as measured by common benchmarks. We describe and open-source our full post-training method, the implementation of which can be found at this https URL. 

---
# Thought-For-Food: Reasoning Chain Induced Food Visual Question Answering 

**Authors**: Riddhi Jain, Manasi Patwardhan, Parijat Deshpande, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2511.01213)  

**Abstract**: The immense diversity in the culture and culinary of Indian cuisines calls attention to the major shortcoming of the existing Visual Question Answering(VQA) systems which are inclined towards the foods from Western region. Recent attempt towards building a VQA dataset for Indian food is a step towards addressing this challenge. However, their approach towards VQA follows a two-step process in which the answer is generated first, followed by the explanation of the expected answer. In this work, we claim that food VQA requires to follow a multi-step reasoning process to arrive at an accurate answer, especially in the context of India food, which involves understanding complex culinary context and identifying relationships between various food items. With this hypothesis we create reasoning chains upon the QA with minimal human intervention. We fine-tune smaller LLMs and VLMs with auto-validated reasoning chains and further train them using reinforcement learning with larger data. With augmentation of reasoning chains, we observed accuracy improvement of an average 10 percentage points on the baseline. We provide detailed analysis in terms the effect of addition of reasoning chains for the Indian Food VQA task.
Index Terms - FoodVQA, Reasoning Chains, Reinforcement Learning, Knowledge Graph. 

---
# Efficient Reinforcement Learning for Large Language Models with Intrinsic Exploration 

**Authors**: Yan Sun, Jia Guo, Stanley Kok, Zihao Wang, Zujie Wen, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00794)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has improved the reasoning ability of large language models, yet training remains costly because many rollouts contribute little to optimization, considering the amount of computation required. This study investigates how simply leveraging intrinsic data properties, almost free benefit during training, can improve data efficiency for RLVR. We propose PREPO with two complementary components. First, we adopt prompt perplexity as an indicator of model adaptability in learning, enabling the model to progress from well-understood contexts to more challenging ones. Second, we amplify the discrepancy among the rollouts by differentiating their relative entropy, and prioritize sequences that exhibit a higher degree of exploration. Together, these mechanisms reduce rollout demand while preserving competitive performance. On the Qwen and Llama models, PREPO achieves effective results on mathematical reasoning benchmarks with up to 3 times fewer rollouts than the baselines. Beyond empirical gains, we provide theoretical and in-depth analyses explaining the underlying rationale of our method to improve the data efficiency of RLVR. 

---
# UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings 

**Authors**: Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00405)  

**Abstract**: The remarkable success of multimodal large language models (MLLMs) has driven advances in multimodal embeddings, yet existing models remain inherently discriminative, limiting their ability to benefit from reasoning-driven generation paradigm. In this work, we pioneer the exploration of generative embeddings, unifying embedding tasks within a generative paradigm. We propose UME-R1, a universal multimodal embedding framework consisting of a two-stage training strategy: a cold-start supervised fine-tuning equips the model with reasoning capabilities and enables it to generate both discriminative and generative embeddings; a subsequent reinforcement learning enhances reasoning and further optimizes generative embedding quality. This pioneering work reveals four key insights: 1) generative embeddings unlock substantial performance gains over conventional discriminative embeddings by leveraging the powerful generative reasoning capabilities of MLLMs; 2) discriminative and generative embeddings are complementary, whose combined oracle performance far exceeding that of either alone; 3) RL can effectively enhance generative embeddings, establishing a scalable optimization paradigm.; 4) repeated sampling at inference boosts downstream task coverage (pass@k), highlighting the inference-time scalability potential of generative embeddings. Evaluated on the MMEB-V2 benchmark across 78 tasks spanning video, image, and visual documents, UME-R1 significantly outperforms conventional discriminative embedding models and offers a foundation for more interpretable, reasoning-driven generative multimodal embeddings. Our code, models, and datasets will be publicly available at this https URL. 

---
# Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning 

**Authors**: Marwa Abdulhai, Ryan Cheng, Donovan Clay, Tim Althoff, Sergey Levine, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2511.00222)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate human users in interactive settings such as therapy, education, and social role-play. While these simulations enable scalable training and evaluation of AI agents, off-the-shelf LLMs often drift from their assigned personas, contradict earlier statements, or abandon role-appropriate behavior. We introduce a unified framework for evaluating and improving persona consistency in LLM-generated dialogue. We define three automatic metrics: prompt-to-line consistency, line-to-line consistency, and Q&A consistency, that capture different types of persona drift and validate each against human annotations. Using these metrics as reward signals, we apply multi-turn reinforcement learning to fine-tune LLMs for three user roles: a patient, a student, and a social chat partner. Our method reduces inconsistency by over 55%, resulting in more coherent and faithful simulated users. 

---
# Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence 

**Authors**: Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Hanzhe Shan, Zhenwei Niu, Zhaoyang Liu, Yue Zhao, Junbo Qi, Qinfan Zhang, Dengjie Li, Yidong Wang, Jiachen Luo, Yong Dai, Jian Tang, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2511.00108)  

**Abstract**: This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop. 

---
# Semi-Supervised Preference Optimization with Limited Feedback 

**Authors**: Seonggyun Lee, Sungjun Lim, Seojin Park, Soeun Cheon, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.00040)  

**Abstract**: The field of preference optimization has made outstanding contributions to the alignment of language models with human preferences. Despite these advancements, recent methods still rely heavily on substantial paired (labeled) feedback data, leading to substantial resource expenditures. To address these challenges, we study the problem of Semi-Supervised Preference Optimization (SSPO) in which the idea is to learn from both a small number of pairwise preference labels and a large pool of unpaired samples simultaneously. Our key theoretical contribution proves the existence of an optimal reward threshold capable of separating winning and losing responses with high probability, which enables a principled pseudo-labeling of unpaired data. By leveraging these pseudo-labels, SSPO effectively distills latent preferences from large-scale unpaired data, thus maintaining human alignment while drastically reducing acquisition costs. Extensive experiments across datasets validate this remarkable data efficiency; for instance, SSPO trained with Llama3-8B-Instruct on just 1% of UltraFeedback consistently surpasses strong baselines trained on 10% of UltraFeedback. 

---
# BARD: budget-aware reasoning distillation 

**Authors**: Lujie Niu, Lei Shen, Yi Jiang, Caixia Yuan, Xiaojie Wang, Wenbo Su, Bo zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01470)  

**Abstract**: While long Chain-of-Thought (CoT) distillation effectively transfers reasoning capability to smaller language models, the reasoning process often remains redundant and computational budget uncontrollable, leading to inefficient resource usage. To address this limitation, we propose \textbf{Budget-Aware Reasoning Distillation (BARD)}, a novel framework that simultaneously distills reasoning capability and enables fine-grained control over the reasoning length. BARD uses the thinking budget as a user-specified control signal, allowing the model to dynamically balance reasoning performance and computational efficiency. To achieve this concept, BARD introduces a two-phase training regimen. The first phase, Supervised Fine-Tuning (SFT) on teacher-generated long CoT data compressed to various budget levels, bootstrapping the model's understanding of budget constraints. The second phase leverages Reinforcement Learning (RL) from a reward signal in consideration of reasoning performance and budget fidelity simultaneously. Incorporating the two-phase regimen is crucial to avoiding policy degradation and ensuring that both objectives are optimized jointly. Extensive experiments demonstrate that our method empowers an 8B student model to achieve strong performance on challenging reasoning benchmarks (\textit{AIME24, AIME25, GPQA}) while providing precise and adaptive control over its reasoning length across a wide range of budgets. 

---
# IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation 

**Authors**: Bosi Wen, Yilin Niu, Cunxiang Wang, Pei Ke, Xiaoying Ling, Ying Zhang, Aohan Zeng, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01014)  

**Abstract**: Instruction following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic that can provide efficient and reliable assessments of constraint following in the instructions. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments demonstrate that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including Deepseek-R1 and o4-mini. With the scalable reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines. 

---
# OpenSIR: Open-Ended Self-Improving Reasoner 

**Authors**: Wai-Chung Kwan, Joshua Ong Jun Leang, Pavlos Vougiouklis, Jeff Z. Pan, Marco Valentino, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2511.00602)  

**Abstract**: Recent advances in large language model (LLM) reasoning through reinforcement learning rely on annotated datasets for verifiable rewards, which may limit models' ability to surpass human-level performance. While self-play offers a promising alternative, existing approaches depend on external verifiers or cannot learn open-endedly. We present Open-Ended Self-Improving Reasoner (OpenSIR), a self-play framework where an LLM learns to generate and solve novel problems by alternating teacher and student roles without external supervision. To generate novel problems, OpenSIR optimises for both difficulty and diversity, rewarding problems that challenge appropriately while exploring distinct concepts, enabling open-ended mathematical discovery. Starting from a single trivial seed problem, OpenSIR substantially improves instruction models: Llama-3.2-3B-Instruct advances from 73.9 to 78.3 on GSM8K, and from 28.8 to 34.4 on College Math, while Gemma-2-2B-Instruct rises from 38.5 to 58.7 on GSM8K. Our analyses reveal that OpenSIR achieves open-ended learning through co-evolving teacher-student roles that adaptively calibrate difficulty and drive diverse exploration, progressing autonomously from basic to advanced mathematics. 

---
# HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning 

**Authors**: Yujian Liu, Jiabao Ji, Yang Zhang, Wenbo Guo, Tommi Jaakkola, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01104)  

**Abstract**: Existing LLM-based automatic test generation methods mainly produce input and expected output pairs to categorize the intended behavior of correct programs. Although straightforward, these methods have limited diversity in generated tests and cannot provide enough debugging information. We propose HarnessLLM, a two-stage training pipeline that enables LLMs to write harness code for testing. Particularly, LLMs generate code that synthesizes inputs and validates the observed outputs, allowing complex test cases and flexible output validation such as invariant checking. To achieve this, we train LLMs with SFT followed by RLVR with a customized reward design. Experiments show that HarnessLLM outperforms input-output-based testing in bug finding and testing strategy diversity. HarnessLLM further benefits the code generation performance through test-time scaling with our generated test cases as inference-phase validation. Our code is available at this https URL. 

---
