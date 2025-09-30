# InfoAgent: Advancing Autonomous Information-Seeking Agents 

**Authors**: Gongrui Zhang, Jialiang Zhu, Ruiqi Yang, Kai Qiu, Miaosen Zhang, Zhirong Wu, Qi Dai, Bei Liu, Chong Luo, Zhengyuan Yang, Linjie Li, Lijuan Wang, Weizhu Chen, Yuan Zhang, Xin Li, Zhaoyi Liu, Xin Geng, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25189)  

**Abstract**: Building Large Language Model agents that expand their capabilities by interacting with external tools represents a new frontier in AI research and applications. In this paper, we introduce InfoAgent, a deep research agent powered by an innovative data synthesis pipeline and orchestrated web search tools. To construct challenging, hard-to-find queries,we build entity trees and apply sub-tree sampling with entity fuzzification to systematically increase question difficulty. Unlike prior work that relies heavily on commercial search tools, we develop a dedicated self-hosted search infrastructure, enhancing transparency of agent environments and facilitating further advancement of agent capacity. We evaluate the effectiveness of our data pipeline by measuring the average number of tool calls required to correctly answer a question, and also show that our agent yields better performance when equipped with our tools. Our \mbox{InfoAgent} is post-trained from Qwen3-14B using a two-stage recipe: cold-start supervised finetuning to instill long-horizon search behaviors, followed by reinforcement learning which significantly improves reasoning-driven tool use. With our methods, InfoAgent achieves 15.3\% accuracy on BrowseComp, 29.2\% on BrowseComp-ZH, and 40.4\% on Xbench-DS, outperforming prior open-source deep research agents such as WebSailor-72B and DeepDive-32B. 

---
# SemanticShield: LLM-Powered Audits Expose Shilling Attacks in Recommender Systems 

**Authors**: Kaihong Li, Huichi Zhou, Bin Ma, Fangjun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24961)  

**Abstract**: Recommender systems (RS) are widely used in e-commerce for personalized suggestions, yet their openness makes them susceptible to shilling attacks, where adversaries inject fake behaviors to manipulate recommendations. Most existing defenses emphasize user-side behaviors while overlooking item-side features such as titles and descriptions that can expose malicious intent. To address this gap, we propose a two-stage detection framework that integrates item-side semantics via large language models (LLMs). The first stage pre-screens suspicious users using low-cost behavioral criteria, and the second stage employs LLM-based auditing to evaluate semantic consistency. Furthermore, we enhance the auditing model through reinforcement fine-tuning on a lightweight LLM with carefully designed reward functions, yielding a specialized detector called SemanticShield. Experiments on six representative attack strategies demonstrate the effectiveness of SemanticShield against shilling attacks, and further evaluation on previously unseen attack methods shows its strong generalization capability. Code is available at this https URL. 

---
# Socratic-Zero : Bootstrapping Reasoning via Data-Free Agent Co-evolution 

**Authors**: Shaobo Wang, Zhengbo Jiao, Zifan Zhang, Yilang Peng, Xu Ze, Boyu Yang, Wei Wang, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24726)  

**Abstract**: Recent breakthroughs in large language models (LLMs) on reasoning tasks rely heavily on massive, high-quality datasets-typically human-annotated and thus difficult to scale. While data synthesis or distillation offers a promising alternative, existing methods struggle with inconsistent data quality and an inability to dynamically adapt to the evolving capabilities of the model, leading to suboptimal training signals. To address these limitations, we introduce Socratic-Zero, a fully autonomous framework that generates high-quality training data from minimal seed examples through the co-evolution of three agents: the Teacher, the Solver, and the Generator. The Solver continuously refines its reasoning by learning from preference feedback on both successful and failed trajectories; the Teacher adaptively crafts increasingly challenging questions based on the Solver's weaknesses; and the Generator distills the Teacher's question-design strategy to enable scalable, high-fidelity curriculum generation. This closed-loop system produces a self-improving curriculum-requiring no pre-existing tasks or labels. Remarkably, starting from only 100 seed questions, our Socratic-Solver-8B achieves an average gain of +20.2 percentage points over prior data synthesis methods across seven mathematical reasoning benchmarks (AMC23, AIME24-25, Olympiad, MATH-500, Minerva, and GSM8K), with consistent gains on both Qwen3 and GLM4 series models. Even more surprisingly, synthetic data from Socratic-Generator-32B enables student LLMs to achieve superior performance compared to other state-of-the-art (SOTA) commercial LLMs on these benchmarks, including Qwen3-235B-A22B, DeepSeek-V3.1-671B, GPT-5, Gemini-2.5-Pro, Grok-4, and Claude-4.1-Opus. 

---
# SeaPO: Strategic Error Amplification for Robust Preference Optimization of Large Language Models 

**Authors**: Jun Rao, Yunjie Liao, Xuebo Liu, Zepeng Lin, Lian Lian, Dong Jin, Shengjun Cheng, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24781)  

**Abstract**: Existing alignment methods for preference optimization of large language models (LLMs) aim to enhance model performance by utilizing pairs of positive and negative samples. However, due to the limited capacity of models in scoring or generating responses, the quality of positive and negative samples may become similar during training, which complicates optimization for preference learning. To address this issue, we introduce SeaPO, a Strategic Error Amplification method that leverages three error types commonly occurring in LLMs to introduce specific error patterns into the model Preference Optimization. This strategy ensures that negative samples are more erroneous than positive samples and preference-based training is employed to mitigate the occurrence of these errors, thereby enhancing model performance. Evaluations across five capability dimensions and different model scales (1.5B to 14B) demonstrate that the generated data significantly improved overall model performance, particularly in terms of truthfulness, with improvements of 5-10 percentage points observed. Further analysis reveals that task performance varies depending on the error types introduced. Injecting the most common error types improves performance in related tasks, while a mix of error types leads to a broader performance enhancement: most tasks show stable improvements, while a few tasks exhibit significant gains. 

---
# Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling 

**Authors**: Pengfei Wang, Baolin Sun, Xuemei Dong, Yaxun Dai, Hongwei Yuan, Mengdie Chu, Yingqi Gao, Xiang Qi, Peng Zhang, Ying Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24403)  

**Abstract**: State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67\% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance. 

---
# GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training 

**Authors**: Hongcheng Wang, Yinuo Huang, Sukai Wang, Guanghui Ren, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24494)  

**Abstract**: Recent progress, such as DeepSeek-R1, has shown that the GRPO algorithm, a Reinforcement Learning (RL) approach, can effectively train Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) and Vision-Language Models (VLMs). In this paper, we analyze three challenges of GRPO: gradient coupling between thoughts and answers, sparse reward signals caused by limited parallel sampling, and unstable advantage estimation. To mitigate these challenges, we propose GRPO-MA, a simple yet theoretically grounded method that leverages multi-answer generation from each thought process, enabling more robust and efficient optimization. Theoretically, we show that the variance of thought advantage decreases as the number of answers per thought increases. Empirically, our gradient analysis confirms this effect, showing that GRPO-MA reduces gradient spikes compared to GRPO. Experiments on math, code, and diverse multimodal tasks demonstrate that GRPO-MA substantially improves performance and training efficiency. Our ablation studies further reveal that increasing the number of answers per thought consistently enhances model performance. 

---
# Reinforcement Mid-Training 

**Authors**: Yijun Tian, Shaoyu Chen, Zhichao Xu, Yawei Wang, Jinhe Bi, Peng Han, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24375)  

**Abstract**: The development of state-of-the-art large language models is commonly understood as a two-stage process involving pre-training and post-training. We point out the need for an additional intermediate stage called reinforcement mid-training with potential for strong performance gains. In this paper, we formally define the problem and identify three key challenges: (1) inefficient training due to excessive reasoning steps, (2) disregard of the imbalanced token entropy distribution, and (3) underutilization of token information. To address these challenges, we propose RMT, a framework for efficient, adaptive, and unified reinforcement mid-training with various innovative components. In particular, we first introduce a dynamic token budget mechanism that constrains unnecessary reasoning steps and mitigates model overthinking. Next, we design a curriculum-based adaptive sampling method that fosters a progressive learning trajectory from easy to hard tokens. Finally, we present a dual training strategy that combines reinforcement learning with next-token prediction, ensuring targeted learning on key tokens and full exploitation of all token information. Extensive experiments demonstrate the superiority of RMT over state-of-the-art methods, achieving up to +64.91% performance improvement with only 21% of the reasoning length in language modeling. We also show that checkpoints obtained after reinforcement mid-training can benefit the subsequent post-training, yielding up to +18.76% improvement in the mathematical domain. 

---
# AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play 

**Authors**: Ran Xu, Yuchen Zhuang, Zihan Dong, Jonathan Wang, Yue Yu, Joyce C. Ho, Linjun Zhang, Haoyu Wang, Wenqi Shi, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24193)  

**Abstract**: Search-augmented LLMs often struggle with complex reasoning tasks due to ineffective multi-hop retrieval and limited reasoning ability. We propose AceSearcher, a cooperative self-play framework that trains a single large language model (LLM) to alternate between two roles: a decomposer that breaks down complex queries and a solver that integrates retrieved contexts for answer generation. AceSearcher couples supervised fine-tuning on a diverse mixture of search, reasoning, and decomposition tasks with reinforcement fine-tuning optimized for final answer accuracy, eliminating the need for intermediate annotations. Extensive experiments on three reasoning-intensive tasks across 10 datasets show that AceSearcher outperforms state-of-the-art baselines, achieving an average exact match improvement of 7.6%. Remarkably, on document-level finance reasoning tasks, AceSearcher-32B matches the performance of the DeepSeek-V3 model using less than 5% of its parameters. Even at smaller scales (1.5B and 8B), AceSearcher often surpasses existing search-augmented LLMs with up to 9x more parameters, highlighting its exceptional efficiency and effectiveness in tackling complex reasoning tasks. Our code will be published at this https URL and this https URL. 

---
# HiPO: Hybrid Policy Optimization for Dynamic Reasoning in LLMs 

**Authors**: Ken Deng, Zizheng Zhan, Wen Xiang, Wenqiang Zhu, Tianhao Peng, Xinping Lei, Weihao Li, Jingxuan Xu, Kun Wu, Yifan Yao, Haoyang Huang, Huaixi Tang, Kepeng Lei, Zhiyi Lai, Songwei Yu, Zongxian Feng, Zuchen Gao, Weihao Xie, Chenchen Zhang, Yanan Wu, Yuanxing Zhang, Lecheng Huang, Yuqun Zhang, Jie Liu, Zhaoxiang Zhang, Haotian Zhang, Bin Chen, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23967)  

**Abstract**: Large Language Models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to improve accuracy on complex tasks. However, always generating lengthy reasoning traces is inefficient, leading to excessive token usage and higher inference costs. This paper introduces the Hybrid Policy Optimization (i.e., HiPO), a framework for adaptive reasoning control that enables LLMs to selectively decide when to engage in detailed reasoning (Think-on) and when to respond directly (Think-off). Specifically, HiPO combines a hybrid data pipelineproviding paired Think-on and Think-off responseswith a hybrid reinforcement learning reward system that balances accuracy and efficiency while avoiding over-reliance on detailed reasoning. Experiments across mathematics and coding benchmarks demonstrate that HiPO can substantially reduce token length while maintaining or improving accuracy. Finally, we hope HiPO a can be a principled approach for efficient adaptive reasoning, advancing the deployment of reasoning-oriented LLMs in real-world, resource-sensitive settings. 

---
# SPELL: Self-Play Reinforcement Learning for evolving Long-Context Language Models 

**Authors**: Ziyi Yang, Weizhou Shen, Ruijun Chen, Chenliang Li, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23863)  

**Abstract**: Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models. 

---
# Toward Preference-aligned Large Language Models via Residual-based Model Steering 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23982)  

**Abstract**: Preference alignment is a critical step in making Large Language Models (LLMs) useful and aligned with (human) preferences. Existing approaches such as Reinforcement Learning from Human Feedback or Direct Preference Optimization typically require curated data and expensive optimization over billions of parameters, and eventually lead to persistent task-specific models. In this work, we introduce Preference alignment of Large Language Models via Residual Steering (PaLRS), a training-free method that exploits preference signals encoded in the residual streams of LLMs. From as few as one hundred preference pairs, PaLRS extracts lightweight, plug-and-play steering vectors that can be applied at inference time to push models toward preferred behaviors. We evaluate PaLRS on various small-to-medium-scale open-source LLMs, showing that PaLRS-aligned models achieve consistent gains on mathematical reasoning and code generation benchmarks while preserving baseline general-purpose performance. Moreover, when compared to DPO-aligned models, they perform better with huge time savings. Our findings highlight that PaLRS offers an effective, much more efficient and flexible alternative to standard preference optimization pipelines, offering a training-free, plug-and-play mechanism for alignment with minimal data. 

---
# Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models 

**Authors**: Zemin Huang, Yuhang Wang, Zhiyang Chen, Guo-Jun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23653)  

**Abstract**: Mask-based Diffusion Language Models (DLMs) struggle to revise incorrect tokens: once a token is generated, it typically remains fixed. The key challenge is to identify potential errors in the inputs. In this paper, we propose \emph{\underline{Rem}asking-\underline{e}nabled \underline{Di}ffusion Language Model (RemeDi}, a mask-based DLM that introduces \emph{remasking} as another fundamental mechanism, enabling more flexible text refinement in diffusion-based text generation. To achieve this, RemeDi jointly predicts token distributions and per-token confidence scores at each step. The confidence scores determine which tokens to be unmasked after the current step, allowing the model to identify tokens with low quality and remask them. These remasked tokens can be resampled with richer context in subsequent steps. We design a remask-aware pipeline to train this ability, including supervised fine-tuning which teaches the model to detect and remask incorrect tokens in addition to predict mask tokens, and reinforcement learning which optimizes full generation trajectories toward higher rewards. Experiments show that RemeDi achieves the state-of-the-art results among open-source DLMs on multiple datasets. 

---
# Knowledge-Level Consistency Reinforcement Learning: Dual-Fact Alignment for Long-Form Factuality 

**Authors**: Junliang Li, Yucheng Wang, Yan Chen, Yu Ran, Ruiqing Zhang, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23765)  

**Abstract**: Hallucination and factuality deficits remain key obstacles to the reliability of large language models (LLMs) in long-form generation. Existing reinforcement learning from human feedback (RLHF) frameworks primarily rely on preference rewards, yet they often overlook the model's internal knowledge boundaries, exacerbating the so-called "hallucination tax". To address this challenge, we propose Knowledge-Level Consistency Reinforcement Learning Framework (KLCF), a novel framework that focuses on the knowledge consistency between the policy model's expressed knowledge and the base model's parametric knowledge, and introduces a Dual-Fact Alignment mechanism to jointly optimize factual recall and precision. Specifically, KLCF leverages pretrained knowledge boundaries to construct fact checklist, guiding online reinforcement learning to improve factual coverage and recall; simultaneously, it trains a self-assessment module based on the base model's internal knowledge to enhance factual precision during generation. Unlike prior methods that rely on external retrieval or heavy verification, our reward design is fully external-knowledge-free and lightweight, making KLCF efficient and easily scalable to large-scale training. Experimental results demonstrate that KLCF substantially improves factuality metrics across multiple long-form benchmarks and effectively alleviates model hallucinations. 

---
# Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs 

**Authors**: Shulin Huang, Yiran Ding, Junshu Pan, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23657)  

**Abstract**: Enhancing the complex reasoning capabilities of Large Language Models (LLMs) attracts widespread attention. While reinforcement learning (RL) has shown superior performance for improving complex reasoning, its impact on cross-lingual generalization compared to Supervised Fine-Tuning (SFT) remains unexplored. We present the first systematic investigation into cross-lingual reasoning generalization of RL and SFT. Using Qwen2.5-3B-Base as our foundation model, we conduct experiments on diverse multilingual reasoning benchmarks, including math reasoning, commonsense reasoning, and scientific reasoning. Our investigation yields two significant findings: (1) Tuning with RL not only achieves higher accuracy but also demonstrates substantially stronger cross-lingual generalization capabilities compared to SFT. (2) RL training on non-English data yields better overall performance and generalization than training on English data, which is not observed with SFT. Furthermore, through comprehensive mechanistic analyses, we explore the underlying factors of RL's superiority and generalization across languages. Our results provide compelling evidence that RL enables the model with more robust reasoning strategies, offering crucial guidance for more equitable and effective multilingual reasoning. 

---
# On the Shelf Life of Fine-Tuned LLM Judges: Future Proofing, Backward Compatibility, and Question Generalization 

**Authors**: Janvijay Singh, Austin Xu, Yilun Zhou, Yefan Zhou, Dilek Hakkani-Tur, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.23542)  

**Abstract**: The LLM-as-a-judge paradigm is widely used in both evaluating free-text model responses and reward modeling for model alignment and finetuning. Recently, finetuning judges with judge-specific data has emerged as an often preferred choice over directly prompting frontier models as judges, as the former achieves better performance with smaller model sizes while being more robust to common biases. However, the standard evaluation ignores several practical concerns of finetuned judges regarding their real world deployment. In this paper, we identify and formalize three aspects that affect the shelf life of these judges: future proofing and backward compatibility -- how well judges finetuned on responses by today's generator models perform on responses by future models or past models, as well as question generalization -- how well judges generalize to unseen questions at test time. We study these three aspects in the math domain under a unified framework with varying train and test distributions, three SFT- and DPO-based finetuning algorithms and three different base models. Experiments suggest that future-proofing is challenging for most models, while backward compatibility is relatively easy, with DPO-trained models consistently improving performance. We further find that continual learning provides a more balanced adaptation to shifts between older and newer response distributions than training solely on stronger or weaker responses. Moreover, all models observe certain degrees of performance degradation when moving from questions seen during training to unseen ones, showing that current judges do not fully generalize to unseen questions. These findings provide insights into practical considerations for developing and deploying judge models in the face of ever-changing generators. 

---
# Cognition-of-Thought Elicits Social-Aligned Reasoning in Large Language Models 

**Authors**: Xuanming Zhang, Yuxuan Chen, Min-Hsuan Yeh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23441)  

**Abstract**: Large language models (LLMs) excel at complex reasoning but can still exhibit harmful behaviors. Current alignment strategies typically embed safety into model weights, making these controls implicit, static, and difficult to modify. This paper introduces Cognition-of-Thought (CooT), a novel decoding-time framework that equips LLMs with an explicit cognitive self-monitoring loop. CooT couples a standard text Generator with a cognitive Perceiver that continuously monitors the unfolding sequence. The Perceiver uses a structured, precedence-based hierarchy of principles (e.g., safety over obedience) to detect potential misalignments as they arise. When violations are flagged, CooT intervenes by rolling back the generation to the point of error and regenerating under injected guidance that combines universal social priors with context-specific warnings. CooT thus transforms alignment from a fixed property into an explicit, dynamic, and auditable process active during inference, allowing for flexible policy updates without retraining the model. Extensive experiments across multiple benchmarks and model families confirm that CooT consistently improves safety and social reasoning performance. 

---
# PARL-MT: Learning to Call Functions in Multi-Turn Conversation with Progress Awareness 

**Authors**: Huacan Chai, Zijie Cao, Maolin Ran, Yingxuan Yang, Jianghao Lin, pengxin, Hairui Wang, Renjie Ding, Ziyu Wan, Muning Wen, Weiwen Liu, Weinan Zhang, Fei Huang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23206)  

**Abstract**: Large language models (LLMs) have achieved impressive success in single-turn function calling, yet real-world applications such as travel planning or multi-stage data analysis typically unfold across multi-turn conversations. In these settings, LLMs must not only issue accurate function calls at each step but also maintain progress awareness, the ability to summarize past interactions and plan future actions to ensure coherent, long-horizon task execution. Existing approaches, however, either reduce multi-turn training to isolated single-turn samples, which neglects task-level planning, or employ end-to-end reinforcement learning (RL) that struggles with redundancy and lacks explicit integration of progress awareness. To overcome these limitations, we introduce PARL-MT, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. PARL-MT combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that PARL-MT significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling. 

---
# Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization 

**Authors**: Junming Yang, Ning Xu, Biao Liu, Shiqi Qiao, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23371)  

**Abstract**: Preference optimization is crucial for aligning large language models (LLMs) with human values and intentions. A significant challenge in this process is the distribution mismatch between pre-collected offline preference data and the evolving model policy. Existing methods attempt to reduce this gap using static heuristics or decoupled online sampling strategies, but they often fail to adapt to the model's dynamic learning state. To bridge this gap, we propose Meta-Weighted Adaptive Preference Optimization (MetaAPO), a novel framework that dynamically couples data generation with model training. MetaAPO employs a lightweight meta-learner, as an "alignment gap estimator", to evaluate the potential benefits of on-policy sampling in relation to offline data. This guides targeted online generation and assigns sample-wise meta-weights to the optimization objective, dynamically balancing the quality and distribution of online and offline data. Experiments on AlpacaEval 2, Arena-Hard and MT-Bench demonstrate that MetaAPO consistently outperforms existing preference optimization approaches across various settings, while reducing 42% in online annotation costs. 

---
# Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts 

**Authors**: Guancheng Wan, Leixin Sun, Longxu Dou, Zitong Shi, Fang Wu, Eric Hanchen Jiang, Wenke Huang, Guibin Zhang, Hejia Geng, Xiangru Tang, Zhenfei Yin, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23188)  

**Abstract**: Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning. 

---
# MedCritical: Enhancing Medical Reasoning in Small Language Models via Self-Collaborative Correction 

**Authors**: Xinchun Su, Chunxu Luo, Yixuan Li, Weidong Yang, Lipeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.23368)  

**Abstract**: In the field of medicine, complex reasoning tasks such as clinical diagnosis, treatment planning, and medical knowledge integration pose significant challenges, where small language models often underperform compared to large language models like GPT-4 and Deepseek. Recent knowledge distillation-based methods aim to address these issues through teacher-guided error correction, but this LLM as judge approach remains challenging in terms of cost, time, and efficiency. To circumvent this issue, we propose a novel two-stage framework, MedCritical, which uses a small language model fine-tuned by a large teacher model to play against itself. In the first stage, we extract high-level and detailed long-chain thought templates from the teacher model to guide the student model to generate more complex reasoning thoughts. In the second stage, we introduce direct preference optimization (DPO) through model self-iteration collaboration to enhance the reasoning ability of the student model by playing against the correction trajectory of the fine-tuned model during training. This model self-learning DPO approach teaches the student model to use its own error-driven insights to consolidate its skills and knowledge to solve complex problems, and achieves comparable results to traditional knowledge distillation methods using teacher models at a lower cost. Notably, our MedCritical 7B model outperforms the Taiyi and Huatuo-o1-7B models by 3.04\% and 10.12\% respectively on the CMExam benchmark, achieving new SOTA performance among 7B-class small models. 

---
# Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs 

**Authors**: Chenxing Wei, Hong Wang, Ying He, Fei Yu, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23166)  

**Abstract**: Large Language Models (LLMs) employ multi-turn interaction as a fundamental paradigm for completing complex tasks. However, their performance often degrades in extended interactions, as they are typically trained on static, single-turn data, which hinders their ability to adapt to real-time user feedback. To address this limitation, we first propose a new paradigm: Test-Time Policy Adaptation for Multi-Turn Interactions (T2PAM), which utilizes user feedback from the ongoing interaction as a reward signal to estimate a latent optimal policy aligned with user preferences, then updates a small subset of parameters to steer the model toward this policy, ultimately enabling efficient in-conversation self-correction. We then introduce Optimum-Referenced One-Step Adaptation (ROSA), a lightweight algorithm that operationalizes T2PAM. ROSA guides the model parameters toward a theoretical optimal policy in a single, efficient update step, avoiding costly iterative gradient-based optimization and minimizing computational overhead. We provide a rigorous theoretical analysis guaranteeing that the policy of ROSA converges to the preference of user as the number of interactions increases. Extensive experiments on challenging benchmark demonstrate that ROSA achieves significant improvements in both task effectiveness and efficiency. 

---
# Tagging the Thought: Unlocking Personalization Reasoning via Reinforcement Learning 

**Authors**: Song Jin, Juntian Zhang, Yong Liu, Xun Zhang, Yufei Zhang, Fei Jiang, Guojun Yin, Wei Lin, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23140)  

**Abstract**: Recent advancements have endowed Large Language Models (LLMs) with impressive general reasoning capabilities, yet they often struggle with personalization reasoning - the crucial ability to analyze user history, infer unique preferences, and generate tailored responses. To address this limitation, we introduce TagPR, a novel training framework that significantly enhances an LLM's intrinsic capacity for personalization reasoning through a tagging the thought approach. Our method first develops a data-driven pipeline to automatically generate and semantically label reasoning chains, creating a structured dataset that fosters interpretable reasoning. We then propose a synergistic training strategy that begins with Supervised Fine-Tuning (SFT) on this tagged data to establish foundational reasoning patterns, followed by a multi-stage reinforcement learning (RL) process. This RL phase is guided by a unique composite reward signal, which integrates tag-based constraints and a novel Personalization Reward Model with User Embeddings (PRMU) to achieve fine-grained alignment with user-specific logic. Extensive experiments on the public LaMP benchmark and a self-constructed dataset demonstrate that our approach achieves state-of-the-art results, delivering an average improvement of 32.65% over the base model across all tasks. Our work validates that structured, interpretable reasoning is a highly effective pathway to unlocking genuine personalization capabilities in LLMs. 

---
# Learning to Reason in Structured In-context Environments with Reinforcement Learning 

**Authors**: Peng Yu, Zeyuan Zhao, Shao Zhang, Luoyi Fu, Xinbing Wang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23330)  

**Abstract**: Large language models (LLMs) have achieved significant advancements in reasoning capabilities through reinforcement learning (RL) via environmental exploration. As the intrinsic properties of the environment determine the abilities that LLMs can learn, the environment plays a important role in the RL finetuning process. An ideal LLM reasoning environment should possess three core characteristics: scalability, generalizable reasoning, and verifiability. However, existing mathematical and coding environments are difficult to scale due to heavy reliance on expert annotation, while the skills learned in game-based environments are too specialized to generalize. To bridge this gap, we introduce the \textbf{S}tructured \textbf{I}n-context \textbf{E}nvironment (SIE) framework. SIE achieves scalability by automatically constructing reasoning environments from large-scale structured data, where the rich compositional patterns naturally support generalizable reasoning. Moreover, the explicit schemas and reasoning chains in structured data provide a foundation for rule-based verifiability. Experimental results show that SIE framework not only achieves substantial improvements in in-domain structured reasoning, but also enables the learned compositional reasoning skills to generalize effectively to out-of-domain mathematical and logical reasoning tasks. We further explored learning in information-limited partial SIEs and found that LLMs can infer the missing information through exploring the environment, leading to robust reasoning improvements and generalization performance. 

---
# Scaling Policy Compliance Assessment in Language Models with Policy Reasoning Traces 

**Authors**: Joseph Marvin Imperial, Harish Tayyar Madabushi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23291)  

**Abstract**: Policy compliance assessment is a fundamental task of evaluating whether an input case strictly complies with a set of human-defined rules, more generally known as policies. In practice, human experts follow a systematic, step-by-step process to identify violations with respect to specific stipulations outlined in the policy. However, such documentation of gold-standard, expert-level reasoning processes is costly to acquire. In this paper, we introduce Policy Reasoning Traces (PRT), a form of specialized generated reasoning chains that serve as a reasoning bridge to improve an LLM's policy compliance assessment capabilities. Our empirical evaluations demonstrate that the use of PRTs for both inference-time and training-time scenarios significantly enhances the performance of open-weight and commercial models, setting a new state-of-the-art for HIPAA and GDPR policies. Beyond accuracy gains, we also highlight how PRTs can improve an LLM's ability to accurately cite policy clauses, as well as influence compliance decisions through their high utilization from the raw chains of thought. 

---
# Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents 

**Authors**: Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23040)  

**Abstract**: Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory corpus that is dynamically updated during a single-pass document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from irreversible forward-only processing, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, a memory-augmented agent with callback-enhanced memory that allows selective retrieval from the entire memory history and allows non-linear reasoning and revisiting of early evidence. To further strengthen training, we propose Reinforcement Learning with Multi-Level Rewards (RLMLR), which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support multi-hop memory utilizing. Experiments on long-document QA show significant gains over existing memory-based approaches, which validates ReMemR1 as an effective solution for long-context reasoning agents. 

---
# SIRI: Scaling Iterative Reinforcement Learning with Interleaved Compression 

**Authors**: Haoming Wen, Yushi Bai, Juanzi Li, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25176)  

**Abstract**: We introduce SIRI, Scaling Iterative Reinforcement Learning with Interleaved Compression, a simple yet effective RL approach for Large Reasoning Models (LRMs) that enables more efficient and accurate reasoning. Existing studies have observed repetitive thinking patterns in LRMs, and attempts to reduce them often come at the cost of performance. In this paper, we show that this trade-off can be overcome through a training regime that iteratively alternates between compressing and expanding the reasoning budget, by dynamically adjusting the maximum rollout length during training. The compression phase cuts the rollout length, forcing the model to make precise and valuable decisions within a limited context, which effectively reduces redundant tokens and increases reasoning density. The expansion phase then relaxes the length limit, providing space for the model to explore and plan in long-horizon settings. Remarkably, we find that after each compression-expansion cycle, the model's performance improves even as its output length decreases, steadily pushing it closer to the Pareto frontier in the performance-efficiency trade-off. Training on DeepSeek-R1-Distill-Qwen-1.5B, SIRI-low improves performance on AIME24 by 43.2% while reducing token usage by 46.9% after three iterations, and SIRI-high achieves the highest accuracy compared to all other methods (Figure 1). Our findings shed light on the potential of periodically oscillating the LRM's output truncation length during training to dynamically balance exploration and efficiency in reasoning, converging towards an optimal "sweet spot" between the two. Our models are publicly available. 

---
# EditGRPO: Reinforcement Learning with Post -Rollout Edits for Clinically Accurate Chest X-Ray Report Generation 

**Authors**: Kai Zhang, Christopher Malon, Lichao Sun, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2509.22812)  

**Abstract**: Radiology report generation requires advanced medical image analysis, effective temporal reasoning, and accurate text generation. Although recent innovations, particularly multimodal large language models (MLLMs), have shown improved performance, their supervised fine-tuning (SFT) objective is not explicitly aligned with clinical efficacy. In this work, we introduce EditGRPO, a mixed-policy reinforcement learning (RL) algorithm designed specifically to optimize the generation through clinically motivated rewards. EditGRPO integrates on-policy exploration with off-policy guidance by injecting sentence-level detailed corrections during training rollouts. This mixed-policy approach addresses the exploration dilemma and sampling efficiency issues typically encountered in RL. Applied to a Qwen2.5-VL-3B MLLM initialized with supervised fine-tuning (SFT), EditGRPO outperforms both SFT and vanilla GRPO baselines, achieving an average improvement of 3.4% in CheXbert, GREEN, Radgraph, and RATEScore metrics across four major chest X-ray report generation datasets. Notably, EditGRPO also demonstrates superior out-of-domain generalization, with an average performance gain of 5.9% on unseen datasets. 

---
# Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning 

**Authors**: Chi Ruan, Dongfu Jiang, Yubo Wang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22824)  

**Abstract**: Reinforcement Learning (RL) has emerged as a popular training paradigm, particularly when paired with reasoning models. While effective, it primarily focuses on generating responses and lacks mechanisms to explicitly foster critique or reflection. Several recent studies, like Critique-Fine-Tuning (CFT) and Critique-Guided-Distillation (CGD) have shown the benefits of explicitly teaching LLMs how to critique. Motivated by them, we propose Critique Reinforcement Learning (CRL), where the model is tasked with generating a critique for a given (question, solution) pair. The reward is determined solely by whether the final judgment label $c \in \{\texttt{True}, \texttt{False}\}$ of the generated critique aligns with the ground-truth judgment $c^*$. Building on this point, we introduce \textsc{Critique-Coder}, which is trained on a hybrid of RL and CRL by substituting 20\% of the standard RL data with CRL data. We fine-tune multiple models (\textsc{Critique-Coder}) and evaluate them on different benchmarks to show their advantages over RL-only models. We show that \textsc{Critique-Coder} consistently outperforms RL-only baselines on all the evaluated benchmarks. Notably, our \textsc{Critique-Coder-8B} can reach over 60\% on LiveCodeBench (v5), outperforming other reasoning models like DeepCoder-14B and GPT-o1. Beyond code generation, \textsc{Critique-Coder} also demonstrates enhanced general reasoning abilities, as evidenced by its better performance on logic reasoning tasks from the BBEH dataset. This indicates that the application of CRL on coding datasets enhances general reasoning and critique abilities, which are transferable across a broad range of tasks. Hence, we believe that CRL works as a great complement to standard RL for LLM reasoning. 

---
# Rethinking Entropy Regularization in Large Reasoning Models 

**Authors**: Yuxian Jiang, Yafu Li, Guanxu Chen, Dongrui Liu, Yu Cheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25133)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown great promise in enhancing the reasoning abilities of large reasoning models (LRMs). However, it suffers from a critical issue: entropy collapse and premature convergence. Naive entropy regularization, a common approach for encouraging exploration in the traditional RL literature, fails to address this problem in the context of LRM. Our analysis reveals that this failure stems from the vast action space and long trajectories in LRMs, which easily trigger a global entropy explosion as the model indiscriminately explores all possible actions and states. To address this, we propose SIREN (SelectIve entRopy rEgularizatioN), a method that confines exploration to a meaningful subset of actions and states. SIREN achieves this through a two-step entropy masking mechanism, consisting of a top-p mask and a peak-entropy mask. In addition, regularization is transformed into a self-anchored form to stabilize training. Across five mathematical benchmarks, SIREN attains superior average performance over previous entropy-related RLVR approaches, exemplified by a +6.6 maj@k improvement on AIME24/25 with Qwen2.5-Math-7B. Further analysis confirms that SIREN promotes greater response diversity and maintains entropy at an appropriate level, which helps to preserve the validation pass@k throughout training. This effectively mitigates the premature convergence problem common in RLVR for LRM. 

---
# RAR$^2$: Retrieval-Augmented Medical Reasoning via Thought-Driven Retrieval 

**Authors**: Kaishuai Xu, Wenjun Hou, Yi Cheng, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22713)  

**Abstract**: Large Language Models (LLMs) have shown promising performance on diverse medical benchmarks, highlighting their potential in supporting real-world clinical tasks. Retrieval-Augmented Generation (RAG) has emerged as a key approach for mitigating knowledge gaps and hallucinations by incorporating external medical information. However, RAG still struggles with complex medical questions that require intensive reasoning, as surface-level input often fails to reflect the true knowledge needs of the task. Existing methods typically focus on refining queries without explicitly modeling the reasoning process, limiting their ability to retrieve and integrate clinically relevant knowledge. In this work, we propose RAR$^2$, a joint learning framework that improves both Reasoning-Augmented Retrieval and Retrieval-Augmented Reasoning. RAR$^2$ constructs a thought process to uncover implicit knowledge requirements and uses it to guide retrieval and answer generation. We build a training dataset of mixed preference pairs and apply Direct Preference Optimization (DPO) to train the model. Moreover, we design two test-time scaling strategies to explore the boundaries of our framework. Experiments demonstrate the effectiveness of RAR$^2$ across several biomedical question answering datasets, outperforming RAG baselines with or without fine-tuning. 

---
# Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval 

**Authors**: Junwei Lan, Jianlyu Chen, Zheng Liu, Chaofan Li, Siqi Bao, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.24869)  

**Abstract**: With the growing popularity of LLM agents and RAG, it has become increasingly important to retrieve documents that are essential for solving a task, even when their connection to the task is indirect or implicit. Addressing this problem requires fine-grained reasoning to accurately assess the relevance between the task and each candidate document. This capability, however, poses a significant challenge for existing IR techniques. Despite recent progress in reasoning-enhanced IR, existing approaches still face significant challenges in applicability, scalability, and efficiency. In this work, we propose Retro*, a novel approach for reasoning-intensive document retrieval. Our method introduces a rubric-based relevance scoring mechanism, enabling the model to reason about the relationship between a task and a document based on explicitly defined criteria, whereby producing a fine-grained, interpretable relevance score. Retro* also supports test-time scaling by combining multiple reasoning trajectories via score integration, which produces more reliable relevance estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel reinforcement learning algorithm tailored for its relevance scoring mechanism, which employs two composite rewards to fully exploit the trajectories of each training sample. Our experiments show that Retro* outperforms existing document retrieval methods with notable advantages, leading to state-of-the-art performance on the BRIGHT benchmark. 

---
# The Era of Real-World Human Interaction: RL from User Conversations 

**Authors**: Chuanyang Jin, Jing Xu, Bo Liu, Leitian Tao, Olga Golovneva, Tianmin Shu, Wenting Zhao, Xian Li, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2509.25137)  

**Abstract**: We posit that to achieve continual model improvement and multifaceted alignment, future models must learn from natural human interaction. Current conversational models are aligned using pre-annotated, expert-generated human feedback. In this work, we introduce Reinforcement Learning from Human Interaction (RLHI), a paradigm that learns directly from in-the-wild user conversations. We develop two complementary methods: (1) RLHI with User-Guided Rewrites, which revises unsatisfactory model outputs based on users' natural-language follow-up responses, (2) RLHI with User-Based Rewards, which learns via a reward model conditioned on knowledge of the user's long-term interaction history (termed persona). Together, these methods link long-term user personas to turn-level preferences via persona-conditioned preference optimization. Trained on conversations derived from WildChat, both RLHI variants outperform strong baselines in personalization and instruction-following, and similar feedback enhances performance on reasoning benchmarks. These results suggest organic human interaction offers scalable, effective supervision for personalized alignment. 

---
# AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models 

**Authors**: Zihao Zhu, Xinyu Wu, Gehan Hu, Siwei Lyu, Ke Xu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24269)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in complex problem-solving through Chain-of-Thought (CoT) reasoning. However, the multi-step nature of CoT introduces new safety challenges that extend beyond conventional language model alignment. We identify a failure mode in current safety CoT tuning methods: the \textit{snowball effect}, where minor reasoning deviations progressively amplify throughout the thought process, leading to either harmful compliance or excessive refusal. This effect stems from models being trained to imitate perfect reasoning scripts without learning to self-correct. To address this limitation, we propose AdvChain, an alignment paradigm that teaches models dynamic self-correction through adversarial CoT tuning. Our method involves constructing a dataset containing Temptation-Correction and Hesitation-Correction samples, where models learn to recover from harmful reasoning drifts and unnecessary cautions. Extensive experiments show that AdvChain significantly enhances robustness against jailbreak attacks and CoT hijacking while substantially reducing over-refusal on benign prompts, achieving a superior safety-utility balance without compromising reasoning capabilities. Our work establishes a new direction for building more robust and reliable reasoning models. 

---
# Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention 

**Authors**: Yichi Zhang, Yue Ding, Jingwen Yang, Tianwei Luo, Dongbai Li, Ranjie Duan, Qiang Liu, Hang Su, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24393)  

**Abstract**: Although Large Reasoning Models (LRMs) have progressed in solving complex problems, their chain-of-thought (CoT) reasoning often contains harmful content that can persist even when the final responses appear safe. We show that this issue still remains in existing methods which overlook the unique significance of safe reasoning, undermining their trustworthiness and posing potential risks in applications if unsafe reasoning is accessible for and exploited by malicious users. We therefore shift our focus to aligning the safety of reasoning itself in this paper and explore process supervision as the solution. However, simply rewarding safe reasoning proves inadequate due to low rollout diversity and limited training signals. To tackle this challenge, we first delve into the characteristics of safe reasoning and uncover several critical insights that 1) safe reasoning is often consolidated by a few critical steps of safety triggers; 2) compliance cues strongly correlate with unsafe continuations; and 3) corrective interventions reliably steer unsafe trajectories towards safer traces. Motivated by these, we propose Intervened Preference Optimization (IPO), an alignment method that enforces safe reasoning by substituting compliance steps with safety triggers and constructing pairs for preference learning with strong signals. Experiments on jailbreak and adversarial safety benchmarks demonstrate that IPO remarkably improves overall safety regarding both reasoning and responses, outperforming SFT-based and RL-based baselines with a relative reduction of over 30% in harmfulness, while preserving excellent performance across diverse reasoning tasks. The results highlight the importance of explicit alignment for reasoning and provide a practical path to safer LRMs. 

---
# OrthAlign: Orthogonal Subspace Decomposition for Non-Interfering Multi-Objective Alignment 

**Authors**: Liang Lin, Zhihao Xu, Junhao Dong, Jian Zhao, Yuchen Yuan, Guibin Zhang, Miao Yu, Yiming Zhang, Zhengtao Yao, Huahui Yi, Dongrui Liu, Xinfeng Li, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24610)  

**Abstract**: Large language model (LLM) alignment faces a critical dilemma when addressing multiple human preferences: improvements in one dimension frequently come at the expense of others, creating unavoidable trade-offs between competing objectives like helpfulness and harmlessness. While prior work mainly focuses on constraint-based optimization algorithms and data selection strategies to mitigate conflicts, these approaches overlook the fundamental issue of resolving conflicts directly at the parameter level. In this paper, we present OrthAlign, an innovative approach that pioneers a new paradigm by leveraging orthogonal subspace decomposition to fundamentally resolve gradient-level conflicts in multi-objective preference alignment. OrthAlign strategically decomposes parameter update spaces into orthogonal subspaces, ensuring that optimization toward different preferences occurs in mathematically non-interfering directions. Building upon this, we provide theoretical guarantees demonstrating that when parameter increments satisfy both orthogonal subspace constraints and spectral norm bounds, the resulting updates exhibit linear Lipschitz growth rather than exponential instability, ensuring stable convergence across all preference dimensions. Extensive experiments show that: I. OrthAlign achieves maximum single-preference improvements ranging from 34.61% to 50.89% after multiple-objective alignment across helpful, harmless, and truthful dimensions. II. With an average overall reward improvement of 13.96%. 

---
# Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR 

**Authors**: Fanding Huang, Guanbo Huang, Xiao Fan, Yi He, Xiao Liang, Xiao Chen, Qinting Jiang, Faisal Nadeem Khan, Jingyan Jiang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23808)  

**Abstract**: A prevailing view in Reinforcement Learning for Verifiable Rewards (RLVR) interprets recent progress through the lens of an exploration-exploitation trade-off, a perspective largely shaped by token-level metrics. We re-examine this perspective, proposing that this perceived trade-off may not be a fundamental constraint but rather an artifact of the measurement level. To investigate this, we shift the analysis to the semantically rich hidden-state space, adopting Effective Rank (ER) to quantify exploration and proposing its novel first- and second-order derivatives, named Effective Rank Velocity (ERV) and Effective Rank Acceleration (ERA), to capture exploitation dynamics. Our analysis reveals that at the hidden-state level, exploration and exploitation could be decoupled (Sec. 4). This finding reveals an opportunity to enhance both capacities simultaneously. This insight motivates our method, Velocity-Exploiting Rank-Learning (VERL), the first to operationalize the principle of synergistic exploration-exploitation enhancement by directly shaping the RL advantage function. The key innovation is leveraging the theoretically stable ERA as a predictive meta-controller to create a synergistic, dual-channel incentive structure. Instead of forcing a trade-off, VERL prospectively amplifies rewards for exploration to preempt overconfidence and reinforces exploitative gains to consolidate reasoning. Experiments across diverse LLMs and reasoning benchmarks show consistent gains, including up to 21.4% absolute accuracy improvement on the challenging Gaokao 2024 dataset. 

---
# Learning to Ponder: Adaptive Reasoning in Latent Space 

**Authors**: Yixin He, Lumingyuan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24238)  

**Abstract**: Test-time compute has emerged as a key paradigm for enhancing LLM reasoning, yet prevailing approaches like Best-of-N and majority voting apply uniform depth across inputs, wasting computation on simple queries while potentially under-thinking complex ones. We present FR-Ponder, a single-graph, backbone-training-free framework that allocates instance-adaptive reasoning compute via latent steering. A less than 1M-param controller observes hidden states and decides to halt or apply a small ponder step by adding a pre-computed steering vector to frozen representations. Our method extracts the latent steering vector associated with deeper reasoning outputs and direct IO from LLM and re-applies it through a tunable scaling factor, allowing the model to adapt its reasoning depth to the complexity of each input. To balance performance and computational cost, we employ Group Relative Policy Optimization (GRPO) as a reward signal to adaptively regulate reasoning depth, achieving task accuracy while mitigating overreasoning. Through curriculum learning and careful reward engineering, FR-Ponder learns calibrated compute allocation correlated with problem difficulty. On GSM8K and MATH500, FR-Ponder improves the compute-accuracy frontier, delivering lower FLOPs with better matched accuracy and comparing favorably to early-exit baselines, without modifying backbone weights. Analyses visualize interpretable steering directions and show learned compute allocation correlates with problem difficulty. 

---
# Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models 

**Authors**: Yuhui Wang, Changjiang Li, Guangke Chen, Jiacheng Liang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24156)  

**Abstract**: Large reasoning models (LRMs) exhibit unprecedented capabilities in solving complex problems through Chain-of-Thought (CoT) reasoning. However, recent studies reveal that their final answers often contradict their own reasoning traces. We hypothesize that this inconsistency stems from two competing mechanisms for generating answers: CoT reasoning and memory retrieval. To test this hypothesis, we conduct controlled experiments that challenge LRMs with misleading cues during reasoning and/or corrupted answers during retrieval. Our results across models and datasets confirm that both mechanisms operate simultaneously, with their relative dominance influenced by multiple factors: problem domains, model scales, and fine-tuning approaches (e.g., reinforcement learning vs. distillation). The findings reveal a critical limitation in current reasoning fine-tuning paradigms: models can exploit the retrieval mechanism as a shortcut, effectively "hacking" the reward signal and undermining genuine reasoning development. To address this challenge, we introduce FARL, a novel fine-tuning framework that integrates memory unlearning with reinforcement learning. By carefully suppressing retrieval shortcuts during the fine-tuning process, FARL promotes reasoning-dominant behavior and enhances generalizable reasoning capabilities. 

---
# Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm 

**Authors**: Kaisen Yang, Lixuan He, Rushi Shah, Kaicheng Yang, Qinwei Ma, Dianbo Liu, Alex Lamb  

**Link**: [PDF](https://arxiv.org/pdf/2509.23946)  

**Abstract**: Chain-of-Thought (CoT) and its variants have markedly advanced the reasoning abilities of Large Language Models (LLMs), yet their monolithic and auto-regressive architecture inherently conflates high-level strategic planning with low-level step-by-step execution, leading to computational inefficiency, limited exploration of reasoning paths, and reduced interpretability. To overcome these issues, we propose the Explore-Execute Chain ($E^2C$), a structured reasoning framework that decouples reasoning into two distinct phases: an exploratory phase that stochastically generates succinct high-level plans, followed by an execution phase that deterministically carries out the chosen plan. Our approach incorporates a two-stage training methodology, which combines Supervised Fine-Tuning (SFT) - augmented by a novel data generation algorithm enforcing strict plan adherence - with a subsequent Reinforcement Learning (RL) stage that capitalizes on the informativeness of exploration and reinforces the determinism of this http URL decomposition enables an efficient test-time scaling strategy: on AIME'2024, $E^2C$ Test Time Scaling reaches 58.1% accuracy using <10% of the decoding tokens required by comparable methods (e.g., Forest-of-Thought), sharply cutting self-consistency overhead. For cross-domain adaptation, our Exploration-Focused SFT (EF-SFT) fine-tunes with only 3.5% of the tokens used by standard SFT yet yields up to 14.5% higher accuracy than standard SFT on medical benchmarks, delivering state-of-the-art performance, strong generalization, and greater interpretability by separating planning from execution. The code and pre-trained models for the project are available at: this https URL 

---
# Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models 

**Authors**: Guanxu Chen, Yafu Li, Yuxian Jiang, Chen Qian, Qihan Ren, Jingyi Yang, Yu Cheng, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23962)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) for large language models (LLMs) has achieved remarkable progress in enhancing LLMs' reasoning capabilities on tasks with clear correctness criteria, such as mathematical reasoning tasks. Several training metrics, such as entropy or response length, have been observed to correlate with different reasoning behaviors in reinforcement learning. Prior approaches incorporate such priors through reward or advantage shaping, which often relies on hand-crafted penalties and preferences (e.g., higher-is-better or lower-is-better). However, without careful hyperparameter tuning, these directional priors can be overly biased and may lead to failure. To this end, we introduce Conditional advANtage estimatiON (CANON), amplifying the impact of the target metric without presuming its direction. Specifically, CANON regroups the sampled responses into two groups based on the higher or lower value of a target metric, measures which metric trend contributes to better performance through inter-group comparison, and identifies the better response within the same group. In summary, CANON based on entropy consistently outperforms prior methods across three LLMs on both math reasoning and high-complexity logic tasks. When applied to response length, CANON further improves token efficiency, yielding a more favorable Pareto frontier in the performance-cost trade-off. 

---
# Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment 

**Authors**: Min-Hsuan Yeh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23564)  

**Abstract**: Human feedback plays a pivotal role in aligning large language models (LLMs) with human preferences. However, such feedback is often noisy or inconsistent, which can degrade the quality of reward models and hinder alignment. While various automated data cleaning methods have been proposed to mitigate this issue, a systematic evaluation of their effectiveness and generalizability remains lacking. To bridge this gap, we introduce the first comprehensive benchmark for evaluating 13 preference data cleaning methods in the context of LLM alignment. PrefCleanBench offers a standardized protocol to assess cleaning strategies in terms of alignment performance and generalizability across diverse datasets, model architectures, and optimization algorithms. By unifying disparate methods and rigorously comparing them, we uncover key factors that determine the success of data cleaning in alignment tasks. This benchmark lays the groundwork for principled and reproducible approaches to improving LLM alignment through better data quality-highlighting the crucial but underexplored role of data preprocessing in responsible AI development. We release modular implementations of all methods to catalyze further research: this https URL. 

---
# C$^2$GSPG: Confidence-calibrated Group Sequence Policy Gradient towards Self-aware Reasoning 

**Authors**: Haotian Liu, Shuo Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23129)  

**Abstract**: Reinforcement Learning (RL) methods, exemplified by Group Relative Policy Optimization (GRPO) and its variants, play a central role in developing reasoning models. However, these methods often suffer from a critical overconfidence issue, which prevents them from achieving self-aware reasoning models. In this study, we propose a simple yet effective confidence-calibration group sequence policy gradient method, called C$^2$GSPG, which simultaneously enhances reasoning performance while suppressing overconfidence. In principle, we propose a Group Sequence Policy Gradient (GSPG) framework for learning reasoning models, which eliminates the token-level bias commonly appearing in GRPO and its variants. In this framework, we define the model confidence for each reasoning problem using the normalized sequence-level probability, and then apply a cross-entropy regularizer to calibrate the model confidence to the sequence's reward. We demonstrate that the confidence calibration regularizer and GSPG are collaborative for binary rewards, as their objectives always share the same gradient direction. For non-binary rewards, we apply nonlinear reward normalization and adaptive regularizer clipping, mitigating the potential conflict between the two objectives. Applying C$^2$GSPG to post-train large language models in logical and mathematical reasoning tasks, we show its superiority over state-of-the-art methods in both reasoning accuracy and confidence calibration. The code of C$^2$GSPG is available at this https URL. 

---
# Multiplayer Nash Preference Optimization 

**Authors**: Fang Wu, Xu Huang, Weihao Xuan, Zhiwei Zhang, Yijia Xiao, Guancheng Wan, Xiaomin Li, Bing Hu, Peng Xia, Jure Leskovec, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23102)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has emerged as the standard paradigm for aligning large language models (LLMs) with human preferences. However, reward-based methods built on the Bradley-Terry assumption struggle to capture the non-transitive and heterogeneous nature of real-world preferences. To address this, recent studies have reframed alignment as a two-player Nash game, giving rise to Nash learning from human feedback (NLHF). While this perspective has inspired algorithms such as INPO, ONPO, and EGPO with strong theoretical and empirical guarantees, they remain fundamentally restricted to two-player interactions, creating a single-opponent bias that fails to capture the full complexity of realistic preference structures. In this work, we introduce Multiplayer Nash Preference Optimization (MNPO), a novel framework that generalizes NLHF to the multiplayer regime. It formulates alignment as an $n$-player game, where each policy competes against a population of opponents while being regularized toward a reference model. Our framework establishes well-defined Nash equilibria in multiplayer settings and extends the concept of duality gap to quantify approximation quality. We demonstrate that MNPO inherits the equilibrium guarantees of two-player methods while enabling richer competitive dynamics and improved coverage of diverse preference structures. Through comprehensive empirical evaluation, we show that MNPO consistently outperforms existing NLHF baselines on instruction-following benchmarks, achieving superior alignment quality under heterogeneous annotator conditions and mixed-policy evaluation scenarios. Together, these results establish MNPO as a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences. Code is available at this https URL. 

---
# Tracing the Representation Geometry of Language Models from Pretraining to Post-training 

**Authors**: Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards  

**Link**: [PDF](https://arxiv.org/pdf/2509.23024)  

**Abstract**: Standard training metrics like loss fail to explain the emergence of complex capabilities in large language models. We take a spectral approach to investigate the geometry of learned representations across pretraining and post-training, measuring effective rank (RankMe) and eigenspectrum decay ($\alpha$-ReQ). With OLMo (1B-7B) and Pythia (160M-12B) models, we uncover a consistent non-monotonic sequence of three geometric phases during autoregressive pretraining. The initial "warmup" phase exhibits rapid representational collapse. This is followed by an "entropy-seeking" phase, where the manifold's dimensionality expands substantially, coinciding with peak n-gram memorization. Subsequently, a "compression-seeking" phase imposes anisotropic consolidation, selectively preserving variance along dominant eigendirections while contracting others, a transition marked with significant improvement in downstream task performance. We show these phases can emerge from a fundamental interplay of cross-entropy optimization under skewed token frequencies and representational bottlenecks ($d \ll |V|$). Post-training further transforms geometry: SFT and DPO drive "entropy-seeking" dynamics to integrate specific instructional or preferential data, improving in-distribution performance while degrading out-of-distribution robustness. Conversely, RLVR induces "compression-seeking", enhancing reward alignment but reducing generation diversity. 

---
# Causally-Enhanced Reinforcement Policy Optimization 

**Authors**: Xiangqi Wang, Yue Huang, Yujun Zhou, Xiaonan Luo, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23095)  

**Abstract**: Large language models (LLMs) trained with reinforcement objectives often achieve superficially correct answers via shortcut strategies, pairing correct outputs with spurious or unfaithful reasoning and degrading under small causal perturbations. We introduce Causally-Enhanced Policy Optimization (CE-PO), a drop-in reward-shaping framework that augments policy optimization with a differentiable proxy for causal coherence along the generation pathway from prompt (Z) to rationale (X) to answer (Y). CE-PO estimates model-internal influence with Jacobian-based sensitivities, counterfactually hardens these signals to suppress nuisance cues, and fuses the resulting coherence score with task-accuracy feedback via a Minkowski (power-mean) combiner, exposing a single tunable between accuracy and coherence trade-off. The unified reward integrates with PPO/GRPO without architectural changes. Across reasoning benchmarks and causal stress tests, CE-PO reduces reward hacking and unfaithful chain-of-thought while improving robustness to correlation-causation flips and light counterfactual edits, all at near-parity accuracy. Experimental results across 4 datasets show that CE-PO improves accuracy over baselines by 5.49% on average (up to 9.58%), while improving robustness to correlation-causation flips and light counterfactual edits. 

---
# Adaptive Margin RLHF via Preference over Preferences 

**Authors**: Yaswanth Chittepu, Prasann Singhal, Greg Durrett, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2509.22851)  

**Abstract**: Margin-based optimization is fundamental to improving generalization and robustness in classification tasks. In the context of reward model learning from preferences within Reinforcement Learning from Human Feedback (RLHF), existing methods typically rely on no margins, fixed margins, or margins that are simplistic functions of preference ratings. However, such formulations often fail to account for the varying strengths of different preferences, for example some preferences are associated with larger margins between responses, or they rely on noisy margin information derived from ratings. We argue that modeling the strength of preferences can lead to better generalization and more faithful alignment. Furthermore, many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably. We propose an approach that leverages preferences over preferences, that is annotations indicating which of two preferences reflects a stronger distinction. We use this ordinal signal to infer adaptive margins on a per-datapoint basis. We introduce an extension to Direct Preference Optimization (DPO), DPO-PoP, that incorporates adaptive margins from preference-over-preference supervision, enabling improved discriminative and generative performance. Empirically, our method outperforms vanilla DPO, DPO with fixed margins, and DPO with ground-truth margins on the UltraFeedback dataset. Additionally, we show that there is a tradeoff between discriminative and generative performance: improving test classification accuracy, particularly by correctly labeling weaker preferences at the expense of stronger ones, can lead to a decline in generative quality. To navigate this tradeoff, we propose two sampling strategies to gather preference-over-preference labels: one favoring discriminative performance and one favoring generative performance. 

---
# VideoScore2: Think before You Score in Generative Video Evaluation 

**Authors**: Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, Keming Wu, Benjamin Schneider, Quy Duc Do, Zhuofeng Li, Yiming Jia, Yuxuan Zhang, Guo Cheng, Haozhe Wang, Wangchunshu Zhou, Qunshu Lin, Yuanxing Zhang, Ge Zhang, Wenhao Huang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22799)  

**Abstract**: Recent advances in text-to-video generation have produced increasingly realistic and diverse content, yet evaluating such videos remains a fundamental challenge due to their multi-faceted nature encompassing visual quality, semantic alignment, and physical consistency. Existing evaluators and reward models are limited to single opaque scores, lack interpretability, or provide only coarse analysis, making them insufficient for capturing the comprehensive nature of video quality assessment. We present VideoScore2, a multi-dimensional, interpretable, and human-aligned framework that explicitly evaluates visual quality, text-to-video alignment, and physical/common-sense consistency while producing detailed chain-of-thought rationales. Our model is trained on a large-scale dataset VideoFeedback2 containing 27,168 human-annotated videos with both scores and reasoning traces across three dimensions, using a two-stage pipeline of supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO) to enhance analytical robustness. Extensive experiments demonstrate that VideoScore2 achieves superior performance with 44.35 (+5.94) accuracy on our in-domain benchmark VideoScore-Bench-v2 and 50.37 (+4.32) average performance across four out-of-domain benchmarks (VideoGenReward-Bench, VideoPhy2, etc), while providing interpretable assessments that bridge the gap between evaluation and controllable generation through effective reward modeling for Best-of-N sampling. Project Page: this https URL 

---
# CLPO: Curriculum Learning meets Policy Optimization for LLM Reasoning 

**Authors**: Shijie Zhang, Guohao Sun, Kevin Zhang, Xiang Guo, Rujun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25004)  

**Abstract**: Recently, online Reinforcement Learning with Verifiable Rewards (RLVR) has become a key paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing methods typically treat all training samples uniformly, overlooking the vast differences in problem difficulty relative to the model's current capabilities. This uniform training strategy leads to inefficient exploration of problems the model has already mastered, while concurrently lacking effective guidance on problems that are challenging its abilities the most, limiting both learning efficiency and upper-bound performance. To address this, we propose CLPO (Curriculum-guided Learning for Policy Optimization), a novel algorithm that creates a dynamic pedagogical feedback loop within the policy optimization process. The core of CLPO leverages the model's own rollout performance to conduct real-time difficulty assessment, thereby constructing an Online Curriculum. This curriculum then guides an Adaptive Problem Restructuring mechanism, where the model acts as its own teacher: it diversifies medium-difficulty problems to promote generalization and simplifies challenging problems to make them more attainable. Our approach transforms the static training procedure into a dynamic process that co-evolves with the model's capabilities. Experiments show that CLPO achieves state-of-the-art performance across eight challenging mathematical and general reasoning benchmarks, with an average pass@1 improvement of 6.96% over other methods, demonstrating its potential for more efficiently training more capable reasoning models. 

---
# UniAPL: A Unified Adversarial Preference Learning Framework for Instruct-Following 

**Authors**: FaQiang Qian, WeiKun Zhang, Ziliang Wang, Kang An, Xuhui Zheng, Liangjian Wen, Mengya Gao, Yong Dai, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25148)  

**Abstract**: Shaping powerful LLMs to be beneficial and safe is central to AI alignment. We argue that post-training alignment is fundamentally a unified Preference Learning problem, involving two modalities: demonstrated preferences (e.g., Supervised Fine-Tuning, SFT) and comparative preferences (e.g., Reinforcement Learning, RL).The standard sequential pipeline-SFT followed by RL-is flawed due to a critical distributional mismatch: SFT uses static expert data, but as the policy evolves, its generation distribution drifts, making SFT knowledge brittle. Subsequent RL then explores without direct access to the rich, ground-truth knowledge in expert demonstrations, leading to inefficient, ungrounded updates. This separation prevents mutual regularization between data sources. To address this, we reframe alignment as a constrained optimization problem and propose Unified Adversarial Preference Learning (UniAPL),a novel framework that dynamically aligns the policy's distribution with the expert's. UniAPL implements a single-stage unified training objective, jointly learning from mixed batches of SFT and preference data. In every gradient step, dense expert demonstrations directly ground and regularize online exploration, inherently resolving distributional mismatch and maximizing data this http URL evaluate UniAPL on instruction-following tasks using Qwen3-235B-Instruct-2507 as the teacher. Our models match or exceed strong GRPO baselines: +5.77% on Qwen3-0.6B (matching a 32B model) and +3.75% on Qwen3-4B,even outperforming the teacher. Analyses of response length and log-probability distributions confirm that UniAPL outputs closely mimic expert demonstrations, achieving both stronger performance and better behavioral alignment. 

---
# Fin-Ally: Pioneering the Development of an Advanced, Commonsense-Embedded Conversational AI for Money Matters 

**Authors**: Sarmistha Das, Priya Mathur, Ishani Sharma, Sriparna Saha, Kitsuchart Pasupa, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2509.24342)  

**Abstract**: The exponential technological breakthrough of the FinTech industry has significantly enhanced user engagement through sophisticated advisory chatbots. However, large-scale fine-tuning of LLMs can occasionally yield unprofessional or flippant remarks, such as ``With that money, you're going to change the world,'' which, though factually correct, can be contextually inappropriate and erode user trust. The scarcity of domain-specific datasets has led previous studies to focus on isolated components, such as reasoning-aware frameworks or the enhancement of human-like response generation. To address this research gap, we present Fin-Solution 2.O, an advanced solution that 1) introduces the multi-turn financial conversational dataset, Fin-Vault, and 2) incorporates a unified model, Fin-Ally, which integrates commonsense reasoning, politeness, and human-like conversational dynamics. Fin-Ally is powered by COMET-BART-embedded commonsense context and optimized with a Direct Preference Optimization (DPO) mechanism to generate human-aligned responses. The novel Fin-Vault dataset, consisting of 1,417 annotated multi-turn dialogues, enables Fin-Ally to extend beyond basic account management to provide personalized budgeting, real-time expense tracking, and automated financial planning. Our comprehensive results demonstrate that incorporating commonsense context enables language models to generate more refined, textually precise, and professionally grounded financial guidance, positioning this approach as a next-generation AI solution for the FinTech sector. Dataset and codes are available at: this https URL 

---
# Risk-Sensitive RL for Alleviating Exploration Dilemmas in Large Language Models 

**Authors**: Yuhua Jiang, Jiawei Huang, Yufeng Yuan, Xin Mao, Yu Yue, Qianchuan Zhao, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24261)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enhancing Large Language Models (LLMs) on complex reasoning tasks. However, existing methods suffer from an exploration dilemma: the sharply peaked initial policies of pre-trained LLMs confine standard RL algorithms to a narrow set of solutions, boosting single-solution accuracy (pass@1) but suppressing solution diversity and multi-solution performance (pass@k). As a result, RLVR often distills existing capabilities rather than discovering new reasoning strategies. To overcome this, we introduce a Risk-Sensitive Reinforcement Learning framework. Our approach employs a risk-seeking objective that interpolates between mean and maximum rewards, leading to a novel algorithm, Risk-Sensitive GRPO (RS-GRPO), which drives deeper exploration by amplifying learning from challenging prompts. Remarkably, RS-GRPO is simple to implement, requiring only minor code modifications. On six mathematical reasoning benchmarks and with five different LLMs, RS-GRPO consistently improves pass@k performance while maintaining or enhancing pass@1 accuracy. 

---
# Humanline: Online Alignment as Perceptual Loss 

**Authors**: Sijia Liu, Niklas Muennighoff, Kawin Ethayarajh  

**Link**: [PDF](https://arxiv.org/pdf/2509.24207)  

**Abstract**: Online alignment (e.g., GRPO) is generally more performant than offline alignment (e.g., DPO) -- but why? Drawing on prospect theory from behavioral economics, we propose a human-centric explanation. We prove that online on-policy sampling better approximates the human-perceived distribution of what the model can produce, and PPO/GRPO-style clipping -- originally introduced to just stabilize training -- recovers a perceptual bias in how humans perceive probability. In this sense, PPO/GRPO act as perceptual losses already. Our theory further suggests that the online/offline dichotomy is itself incidental to maximizing human utility, since we can achieve the same effect by selectively training on any data in a manner that mimics human perception, rather than restricting ourselves to online on-policy data. Doing so would allow us to post-train more quickly, cheaply, and flexibly without sacrificing performance. To this end, we propose a design pattern that explicitly incorporates perceptual distortions of probability into objectives like DPO/KTO/GRPO, creating humanline variants of them. Surprisingly, we find that these humanline variants, even when trained with offline off-policy data, can match the performance of their online counterparts on both verifiable and unverifiable tasks. 

---
# Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs 

**Authors**: Shreyas Singh, Kunal Singh, Pradeep Moturi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24107)  

**Abstract**: Tool-integrated reasoning has emerged as a key focus for enabling agentic applications. Among these, DeepResearch Agents have gained significant attention for their strong performance on complex, open-ended information-seeking tasks. We introduce Fathom-DeepResearch, an agentic system composed of two specialized models. The first is Fathom-Search-4B, a DeepSearch model trained from Qwen3-4B and optimized for evidence-based investigation through live web search and targeted webpage querying. Its training combines three advances: (i) DUETQA, a 5K-sample dataset generated via multi-agent self-play that enforces strict web-search dependence and heterogeneous source grounding; (ii) RAPO, a zero-overhead extension of GRPO that stabilizes multi-turn Reinforcement Learning with Verifiable Rewards through curriculum pruning, reward-aware advantage scaling, and per-prompt replay buffers; and (iii) a steerable step-level reward that classifies each tool call by cognitive behavior and marginal utility, enabling explicit control over search trajectory breadth, depth, and horizon. These improvements enable reliable extension of tool-calling beyond 20 calls when warranted. The second is Fathom-Synthesizer-4B, trained from Qwen3-4B, which converts multi-turn DeepSearch traces into structured, citation-dense DeepResearch Reports for comprehensive synthesis. Evaluated on DeepSearch benchmarks (SimpleQA, FRAMES, WebWalker, Seal0, MuSiQue) and DeepResearch-Bench, the system achieves state-of-the-art performance in the open-weights category while demonstrating strong generalization to diverse reasoning tasks including HLE, AIME-25, GPQA-Diamond, and MedQA. 

---
# Robust Preference Optimization: Aligning Language Models with Noisy Preference Feedback 

**Authors**: Xiaoyang Cao, Zelai Xu, Mo Guang, Kaiwen Long, Michiel A. Bakker, Yu Wang, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24159)  

**Abstract**: Standard human preference-based alignment methods, such as Reinforcement Learning from Human Feedback (RLHF), are a cornerstone technology for aligning Large Language Models (LLMs) with human values. However, these methods are all underpinned by a critical, yet flawed assumption: human preferences are homogeneous (representing a single, unified preference) and the collected data is noiseless (free from error). In reality, neither is true since human preference is pluralistic and annotators can make mistakes. This creates a discrepancy between the recorded data and the ground-truth preferences, which can misguide the model and degrade its performance. To address this challenge, we introduce Robust Preference Optimization (RPO). RPO employs an Expectation-Maximization (EM) algorithm to infer the posterior probability of each label's correctness, which is used to adaptively re-weigh each data point in the training loss to mitigate noise. We further generalize this approach by establishing a theoretical link between arbitrary preference losses and their corresponding probabilistic models. This generalization enables the systematic transformation of existing alignment algorithms into their robust counterparts, elevating RPO from a specific algorithm to a meta-framework for robust preference alignment. Theoretically, we prove that under the condition of a perfectly calibrated model, RPO is guaranteed to converge to the true noise level of the dataset. Our experiments demonstrate RPO's effectiveness as a meta-framework, consistently enhancing four state-of-the-art alignment algorithms (DPO, IPO, SimPO, and CPO). When applied to Mistral and Llama 3 models, the RPO-enhanced methods achieve substantial win rate gains on AlpacaEval 2 and Arena-Hard, with improvements of up to 7.0% and 5.4%, respectively. 

---
# GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks 

**Authors**: Cong Chen, Kaixiang Ji, Hao Zhong, Muzhi Zhu, Anzhou Li, Guo Gan, Ziyuan Huang, Cheng Zou, Jiajia Liu, Jingdong Chen, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23738)  

**Abstract**: Autonomous agents for long-sequence Graphical User Interface tasks are hindered by sparse rewards and the intractable credit assignment problem. To address these challenges, we introduce GUI-Shepherd, a Process Reward Model that provides dense, step-by-step feedback to guide agents. GUI-Shepherd is trained on a diverse large-scale data set of $52$k interactions that features human-annotated scores and GPT-4o generated rationales, enabling it to serve both as a reward provider for RL training and as a verifier for inference. As far as we know, we are the first to conduct a systematic study of process supervision in GUI agents, across diverse settings from online long-horizon tasks to offline single-step prediction. On the online AndroidWorld benchmark, GUI-Shepherd improves success rate by $7.7$ points via multi-turn online PPO, significantly outperforming Outcome Reward Model based competitors. When used as an inference verifier, it brings $5.1$ points improvements. The benefits generalize to the offline AndroidControl benchmark, with gains of $2.2$ points as a reward provider and $4.3$ points as a verifier. Collectively, our results establish that high-fidelity process supervision is critical for building more capable GUI agents and present a generalizable solution. 

---
# How LLMs Learn to Reason: A Complex Network Perspective 

**Authors**: Sihan Hu, Xiansheng Cai, Yuan Huang, Zhiyuan Yao, Linfeng Zhang, Pan Zhang, Youjin Deng, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23629)  

**Abstract**: Training large language models with Reinforcement Learning from Verifiable Rewards (RLVR) exhibits a set of distinctive and puzzling behaviors that remain poorly understood, including a two-stage learning curve, V-shaped response-length trajectories, and a pronounced vulnerability to catastrophic forgetting. In this work, we propose that these seemingly disparate phenomena can be explained using a single unifying theory: the model's reasoning process maps to the self-organization of a semantic complex network whose topology remains persistently sparse, with the average degree pinned close to two. This topology imposes a fundamental mechanism for forgetting and learning: it first drives the system into a maximally frustrated state where ``skill islands'' form, slow-learning happens, and forgetting is induced; then it enters a sharp growth phase where the new skills are ``bolted on'', driven by phase-transition-like learning at the web's frontier. Equipped with the theory, we propose \textit{Annealed-RLVR}, a principled algorithm that introduces an SFT-based ``heating'' step at the point of maximal frustration to resolve the competitive bottleneck and enhance the reasoning capability of the model. Experiments on a 1.5B-parameter model demonstrate that the approach outperforms standard RLVR on both in-distribution and out-of-distribution benchmarks. By recasting RLVR from black-box optimization into a predictable process of structural self-organization, our work provides a new physical intuition for engineering the emergent reasoning capabilities of future AI systems. 

---
# EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance 

**Authors**: Siyao Song, Cong Ma, Zhihao Cheng, Shiye Lei, Minghao Li, Ying Zeng, Huaixiao Tou, Kai Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.23730)  

**Abstract**: Large language models (LLMs) have recently advanced in reasoning when optimized with reinforcement learning (RL) under verifiable rewards. Existing methods primarily rely on outcome-based supervision to strengthen internal LLM reasoning, often leading to inefficient exploration and sparse rewards. To mitigate this issue, we propose Expert-Assisted Policy Optimization (EAPO), a novel RL framework that enhances exploration by incorporating multi-turn interactions with external experts during training. Unlike prior methods, where policies reason in isolation, EAPO incentivizes the policy to adaptively determine when and how to consult experts, yielding richer reward signals and more reliable reasoning trajectories. External assistance ultimately internalizes expert knowledge into the policy model, amplifying the model's inherent reasoning capabilities. During evaluation, the policy model has been well-optimized to solve questions independently, producing improved reasoning paths and more accurate solutions. Experiments on mathematical reasoning benchmarks, including AIME 2024, AIME 2025, and AIMO 2025, show that EAPO consistently outperforms expert-assisted workflow, expert-distilled models, and RL baselines, with an average gain of 5 points over self-exploratory models. 

---
# Toward Effective Tool-Integrated Reasoning via Self-Evolved Preference Learning 

**Authors**: Yifei Chen, Guanting Dong, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23285)  

**Abstract**: Tool-Integrated Reasoning (TIR) enables large language models (LLMs) to improve their internal reasoning ability by integrating external tools. However, models employing TIR often display suboptimal behaviors, such as insufficient or excessive tool usage and overthinking after tool calls. The challenge of incentivizing LLMs to perform TIR efficiently and accurately, while stabilizing the reasoning process, remains an open question. In this paper, we start by exploring the impact of tool calls on model reasoning from the perspective of information entropy. Our findings indicate that tool call results lead to a distinct change in the information entropy of subsequent reasoning, with the overall entropy of the reasoning chain varying based on the number of tool calls. Building on these insights, we propose Tool-Light, a framework designed to encourage LLMs to perform TIR efficiently and accurately. Our framework includes dataset construction and multi-stage fine-tuning. For dataset construction, we employ continuous self-evolved sampling using the fine-tuned model, integrating both vanilla sampling and entropy-guided sampling. Besides, we establish strict criteria for selecting positive-negative pairs during sampling. The training process involves a two-stage approach, comprising Supervised Fine-Tuning (SFT) and Self-Evolved Direct Preference Optimization (DPO). Experimental results on 10 datasets demonstrate the effectiveness of Tool-Light, significantly improving the model's efficiency in executing TIR tasks. 

---
# Towards Strategic Persuasion with Language Models 

**Authors**: Zirui Cheng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2509.22989)  

**Abstract**: Large language models (LLMs) have demonstrated strong persuasive capabilities comparable to those of humans, offering promising benefits while raising societal concerns about their deployment. However, systematically evaluating the persuasive capabilities of LLMs is inherently challenging, as the effectiveness of persuasion among humans varies significantly across different domains. In this paper, we take a theory-driven approach to provide a scalable and principled framework for measuring the persuasive capabilities of LLMs. Grounded in the Bayesian Persuasion (BP) framework, we repurpose existing human-human persuasion datasets to construct environments for evaluating and training LLMs in strategic persuasion. Our results reveal that frontier models can consistently achieve high persuasion gains and exhibit sophisticated persuasion strategies that align with theoretical predictions. Building on this, we use reinforcement learning to train LLMs for strategic persuasion in our environments. Our results also demonstrate that even small LLMs can obtain significantly higher persuasion gains through reinforcement learning. 

---
# Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards 

**Authors**: Haoran He, Yuxiao Ye, Qingpeng Cai, Chen Hu, Binxing Jiao, Daxin Jiang, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24981)  

**Abstract**: RL with Verifiable Rewards (RLVR) has emerged as a promising paradigm for improving the reasoning abilities of large language models (LLMs). Current methods rely primarily on policy optimization frameworks like PPO and GRPO, which follow generalized policy iteration that alternates between evaluating the current policy's value and improving the policy based on evaluation. While effective, they often suffer from training instability and diversity collapse, requiring complex heuristic tricks and careful tuning. We observe that standard RLVR in math reasoning can be formalized as a specialized finite-horizon Markov Decision Process with deterministic state transitions, tree-structured dynamics, and binary terminal rewards. Though large in scale, the underlying structure is simpler than general-purpose control settings for which popular RL algorithms (e.g., PPO) were developed, suggesting that several sophisticated techniques in existing methods may be reduced or even omitted. Based on this insight, we prove a surprising result: the optimal action can be recovered from the Q-function of a fixed uniformly random policy, thereby bypassing the generalized policy iteration loop and its associated heuristics. We introduce Random Policy Valuation for Diverse Reasoning (ROVER) to translate this principle into a practical and scalable algorithm for LLM math reasoning, a minimalist yet highly effective RL method that samples actions from a softmax over these uniform-policy Q-values. ROVER preserves diversity throughout training, allowing sustained exploration of multiple valid pathways. Across multiple base models and standard math reasoning benchmarks, ROVER demonstrates superior performance in both \textbf{quality} (\textbf{+8.2} on pass@1, \textbf{+16.8} on pass@256) and \textbf{diversity} (\textbf{+17.6\%}), despite its radical simplification compared to strong, complicated existing methods. 

---
# T-POP: Test-Time Personalization with Online Preference Feedback 

**Authors**: Zikun Qu, Min Zhang, Mingze Kong, Xiang Li, Zhiwei Shang, Zhiyong Wang, Yikun Ban, Shuang Qiu, Yao Shu, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24696)  

**Abstract**: Personalizing large language models (LLMs) to individual user preferences is a critical step beyond generating generically helpful responses. However, current personalization methods are ill-suited for new users, as they typically require either slow, resource-intensive fine-tuning or a substantial amount of pre-existing user data, creating a significant cold-start problem. To address this challenge, we introduce a new paradigm for real-time personalization by learning from online pairwise preference feedback collected during text generation. We propose T-POP (Test-Time Personalization with Online Preference Feedback}), a novel algorithm that synergistically combines test-time alignment with dueling bandits. Without updating the LLM parameters, T-POP steers the decoding process of a frozen LLM by learning a reward function online that captures user preferences. By leveraging dueling bandits, T-POP intelligently queries the user to efficiently balance between exploring their preferences and exploiting the learned knowledge to generate personalized text. Extensive experiments demonstrate that T-POP achieves rapid and data-efficient personalization, significantly outperforming existing baselines and showing consistent improvement with more user interactions. 

---
# VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding 

**Authors**: Yizhuo Ding, Mingkang Chen, Zhibang Feng, Tong Xiao, Wanying Qu, Wenqi Shao, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24776)  

**Abstract**: Multimodal large language models (MLLMs) often struggle to ground reasoning in perceptual evidence. We present a systematic study of perception strategies-explicit, implicit, visual, and textual-across four multimodal benchmarks and two MLLMs. Our findings show that explicit perception, especially when paired with textual cues, consistently yields the best improvements, particularly for smaller models. Based on this insight, we propose VTPerception-R1, a unified two-stage framework that decouples perception from reasoning. Stage 1 introduces perception-augmented fine-tuning, and Stage 2 applies perception-aware reinforcement learning with novel visual, textual, and consistency rewards. Experiments demonstrate that VTPerception-R1 significantly improves reasoning accuracy and robustness across diverse tasks, offering a scalable and auditable solution for perception-grounded multimodal reasoning. Our code is available at: this https URL. 

---
# Circuit-Aware Reward Training: A Mechanistic Framework for Longtail Robustness in RLHF 

**Authors**: Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24713)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) reward models exhibit systematic failures on longtail distributions, leading to reward hacking and misalignment. We propose a mechanistic interpretability framework that identifies specialized neural circuits responsible for rare-event processing in reward models. Drawing from recent advances showing distributed specialization for rare tokens in language models\citep{liu2025no, liu2025emergent}, we hypothesize that reward models also develop functionally distinct circuits for longtail scenarios. Our theoretical framework establishes formal connections between circuit specialization, reward generalization bounds, and longtail performance. We introduce \textbf{Circuit-Aware Reward Training (CART)}, which uses circuit analysis to guide data augmentation, regularization, and ensemble strategies. This approach provides both theoretical insights into reward model failures and practical interventions for improving longtail robustness. 

---
# Mitigating Visual Hallucinations via Semantic Curriculum Preference Optimization in MLLMs 

**Authors**: Yuanshuai Li, Yuping Yan, Junfeng Tang, Yunxuan Li, Zeqi Zheng, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24491)  

**Abstract**: Multimodal Large Language Models (MLLMs) have significantly improved the performance of various tasks, but continue to suffer from visual hallucinations, a critical issue where generated responses contradict visual evidence. While Direct Preference Optimization(DPO) is widely used for alignment, its application to MLLMs often fails to capture fine-grained semantic differences and encourages shortcut learning. To address these challenges, we propose Semantic Curriculum Preference Optimization (SCPO), a novel framework for MLLM alignment. SCPO employs a progressive, easy-to-hard curriculum built upon our Semantic Curriculum Preference Pairs dataset, which provides fine-grained semantic contrasts sorted by difficulty. This curriculum is trained with a dynamic reference model and a novel symmetric, bidirectional objective to facilitate simultaneous learning from both textual and visual preferences. To our knowledge, SCPO is the first framework to unify semantics, symmetry, and curriculum for MLLMs alignment, effectively mitigating visual hallucinations. Extensive experiments on LLaVA models across various scales and versions validate that SCPO demonstrates superior performance compared to baseline models on multiple hallucination benchmarks, reducing the hallucination rate by up to 62.9%. Moreover, evaluations on generalized benchmarks show that SCPO improves factuality while preserving general capabilities, with its performance remaining stable across general vision-language benchmarks. 

---
# UI-UG: A Unified MLLM for UI Understanding and Generation 

**Authors**: Hao Yang, Weijie Qiu, Ru Zhang, Zhou Fang, Ruichao Mao, Xiaoyu Lin, Maji Huang, Zhaosong Huang, Teng Guo, Shuoyang Liu, Hai Rao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24361)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have been widely applied across domains, they are still facing challenges in domain-specific tasks, such as User Interface (UI) understanding accuracy and UI generation quality. In this paper, we introduce UI-UG (a unified MLLM for UI Understanding and Generation), integrating both capabilities. For understanding tasks, we employ Supervised Fine-tuning (SFT) combined with Group Relative Policy Optimization (GRPO) to enhance fine-grained understanding on the modern complex UI data. For generation tasks, we further use Direct Preference Optimization (DPO) to make our model generate human-preferred UIs. In addition, we propose an industrially effective workflow, including the design of an LLM-friendly domain-specific language (DSL), training strategies, rendering processes, and evaluation metrics. In experiments, our model achieves state-of-the-art (SOTA) performance on understanding tasks, outperforming both larger general-purpose MLLMs and similarly-sized UI-specialized models. Our model is also on par with these larger MLLMs in UI generation performance at a fraction of the computational cost. We also demonstrate that integrating understanding and generation tasks can improve accuracy and quality for both tasks. 

---
# FrameMind: Frame-Interleaved Chain-of-Thought for Video Reasoning via Reinforcement Learning 

**Authors**: Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24008)  

**Abstract**: Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks that require either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce FrameMind, an end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal-spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding. 

---
# Guide: Generalized-Prior and Data Encoders for DAG Estimation 

**Authors**: Amartya Roy, Devharish N, Shreya Ganguly, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.23992)  

**Abstract**: Modern causal discovery methods face critical limitations in scalability, computational efficiency, and adaptability to mixed data types, as evidenced by benchmarks on node scalability (30, $\le 50$, $\ge 70$ nodes), computational energy demands, and continuous/non-continuous data handling. While traditional algorithms like PC, GES, and ICA-LiNGAM struggle with these challenges, exhibiting prohibitive energy costs for higher-order nodes and poor scalability beyond 70 nodes, we propose \textbf{GUIDE}, a framework that integrates Large Language Model (LLM)-generated adjacency matrices with observational data through a dual-encoder architecture. GUIDE uniquely optimizes computational efficiency, reducing runtime on average by $\approx 42%$ compared to RL-BIC and KCRL methods, while achieving an average $\approx 117%$ improvement in accuracy over both NOTEARS and GraN-DAG individually. During training, GUIDE's reinforcement learning agent dynamically balances reward maximization (accuracy) and penalty avoidance (DAG constraints), enabling robust performance across mixed data types and scalability to $\ge 70$ nodes -- a setting where baseline methods fail. 

---
# ReWatch-R1: Boosting Complex Video Reasoning in Large Vision-Language Models through Agentic Data Synthesis 

**Authors**: Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23652)  

**Abstract**: While Reinforcement Learning with Verifiable Reward (RLVR) significantly advances image reasoning in Large Vision-Language Models (LVLMs), its application to complex video reasoning remains underdeveloped. This gap stems primarily from a critical data bottleneck: existing datasets lack the challenging, multi-hop questions and high-quality, video-grounded Chain-of-Thought (CoT) data necessary to effectively bootstrap RLVR. To address this, we introduce ReWatch, a large-scale dataset built to foster advanced video reasoning. We propose a novel multi-stage synthesis pipeline to synthesize its three components: ReWatch-Caption, ReWatch-QA, and ReWatch-CoT. A core innovation is our Multi-Agent ReAct framework for CoT synthesis, which simulates a human-like "re-watching" process to generate video-grounded reasoning traces by explicitly modeling information retrieval and verification. Building on this dataset, we develop ReWatch-R1 by post-training a strong baseline LVLM with Supervised Fine-Tuning (SFT) and our RLVR framework. This framework incorporates a novel Observation \& Reasoning (O\&R) reward mechanism that evaluates both the final answer's correctness and the reasoning's alignment with video content, directly penalizing hallucination. Our experiments show that ReWatch-R1 achieves state-of-the-art average performance on five challenging video reasoning benchmarks. 

---
# Dynamic-TreeRPO: Breaking the Independent Trajectory Bottleneck with Structured Sampling 

**Authors**: Xiaolong Fu, Lichen Ma, Zipeng Guo, Gaojing Zhou, Chongxiao Wang, ShiPing Dong, Shizhe Zhou, Shizhe Zhou, Ximan Liu, Jingling Fu, Tan Lit Sin, Yu Shi, Zhen Chen, Junshi Huang, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23352)  

**Abstract**: The integration of Reinforcement Learning (RL) into flow matching models for text-to-image (T2I) generation has driven substantial advances in generation quality. However, these gains often come at the cost of exhaustive exploration and inefficient sampling strategies due to slight variation in the sampling group. Building on this insight, we propose Dynamic-TreeRPO, which implements the sliding-window sampling strategy as a tree-structured search with dynamic noise intensities along depth. We perform GRPO-guided optimization and constrained Stochastic Differential Equation (SDE) sampling within this tree structure. By sharing prefix paths of the tree, our design effectively amortizes the computational overhead of trajectory search. With well-designed noise intensities for each tree layer, Dynamic-TreeRPO can enhance the variation of exploration without any extra computational cost. Furthermore, we seamlessly integrate Supervised Fine-Tuning (SFT) and RL paradigm within Dynamic-TreeRPO to construct our proposed LayerTuning-RL, reformulating the loss function of SFT as a dynamically weighted Progress Reward Model (PRM) rather than a separate pretraining method. By associating this weighted PRM with dynamic-adaptive clipping bounds, the disruption of exploration process in Dynamic-TreeRPO is avoided. Benefiting from the tree-structured sampling and the LayerTuning-RL paradigm, our model dynamically explores a diverse search space along effective directions. Compared to existing baselines, our approach demonstrates significant superiority in terms of semantic consistency, visual fidelity, and human preference alignment on established benchmarks, including HPS-v2.1, PickScore, and ImageReward. In particular, our model outperforms SoTA by $4.9\%$, $5.91\%$, and $8.66\%$ on those benchmarks, respectively, while improving the training efficiency by nearly $50\%$. 

---
# Rethinking Large Language Model Distillation: A Constrained Markov Decision Process Perspective 

**Authors**: Matthieu Zimmer, Xiaotong Ji, Tu Nguyen, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22921)  

**Abstract**: We introduce a novel approach to large language model (LLM) distillation by formulating it as a constrained reinforcement learning problem. While recent work has begun exploring the integration of task-specific rewards into distillation processes, existing methods typically rely on ad-hoc reward weighting. We propose a principled optimization framework that maximizes task-specific rewards while constraining the divergence from the teacher model to remain below a specified threshold. Our approach adapts constrained state augmented reinforcement learning to the distillation setting, introducing a modified reward function that maintains theoretical guarantees of constraint satisfaction without requiring state augmentation or teacher model access during deployment and without the computational overhead of the dual Lagrangian methods. Through extensive experiments on mathematical reasoning tasks, we demonstrate that our method achieves better constraint satisfaction rates and better reasoning compared to the soft Lagrangian relaxation baselines while maintaining competitive task performance. Our framework provides a theoretically grounded and practically efficient solution for reward-aware distillation in resource-constrained settings. 

---
