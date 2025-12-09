# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models 

**Authors**: Charlie Zhang, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.07783)  

**Abstract**: Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in language models, yet it remains unclear whether post-training truly extends a model's reasoning ability beyond what it acquires during pre-training. A central challenge is the lack of control in modern training pipelines: large-scale pre-training corpora are opaque, mid-training is often underexamined, and RL objectives interact with unknown prior knowledge in complex ways. To resolve this ambiguity, we develop a fully controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate models along two axes: extrapolative generalization to more complex compositions and contextual generalization across surface contexts. Using this framework, we reconcile competing views on RL's effectiveness. We show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and when RL data target the model's edge of competence, tasks at the boundary that are difficult but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure, after which RL can reliably transfer. 3) Mid-training significantly enhances performance under fixed compute compared with RL only, demonstrating its central but underexplored role in training pipelines. 4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results clarify the interplay between pre-training, mid-training, and RL, offering a foundation for understanding and improving reasoning LM training strategies. 

---
# Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization 

**Authors**: Zhuoran Zhuang, Ye Chen, Jianghao Su, Chao Luo, Luhui Liu, Xia Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2512.07478)  

**Abstract**: Large Language Models (LLMs) empowered with Tool-Integrated Reasoning (TIR) can iteratively plan, call external tools, and integrate returned information to solve complex, long-horizon reasoning tasks. Agentic Reinforcement Learning (Agentic RL) optimizes such models over full tool-interaction trajectories, but two key challenges hinder effectiveness: (1) Sparse, non-instructive rewards, such as binary 0-1 verifiable signals, provide limited guidance for intermediate steps and slow convergence; (2) Gradient degradation in Group Relative Policy Optimization (GRPO), where identical rewards within a rollout group yield zero advantage, reducing sample efficiency and destabilizing training. To address these challenges, we propose two complementary techniques: Progressive Reward Shaping (PRS) and Value-based Sampling Policy Optimization (VSPO). PRS is a curriculum-inspired reward design that introduces dense, stage-wise feedback - encouraging models to first master parseable and properly formatted tool calls, then optimize for factual correctness and answer quality. We instantiate PRS for short-form QA (with a length-aware BLEU to fairly score concise answers) and long-form QA (with LLM-as-a-Judge scoring to prevent reward hacking). VSPO is an enhanced GRPO variant that replaces low-value samples with prompts selected by a task-value metric balancing difficulty and uncertainty, and applies value-smoothing clipping to stabilize gradient updates. Experiments on multiple short-form and long-form QA benchmarks show that PRS consistently outperforms traditional binary rewards, and VSPO achieves superior stability, faster convergence, and higher final performance compared to PPO, GRPO, CISPO, and SFT-only baselines. Together, PRS and VSPO yield LLM-based TIR agents that generalize better across domains. 

---
# Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning 

**Authors**: Tong Wu, Yang Liu, Jun Bai, Zixia Jia, Shuyi Zhang, Ziyong Lin, Yanting Wang, Song-Chun Zhu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.07461)  

**Abstract**: We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from ``cold-start'' format discovery to strict topological constraints without external supervision; 2) a novel Parallel-Aware Policy Optimization (PAPO) algorithm that optimizes branching policies directly within the execution graph, allowing the model to learn adaptive decomposition via trial and error; and 3) a robust NPR Engine that refactors memory management and flow control of SGLang to enable stable, large-scale parallel RL training. Across eight reasoning benchmarks, NPR trained on Qwen3-4B achieves performance gains of up to 24.5% and inference speedups up to 4.6x. Unlike prior baselines that often fall back to autoregressive decoding, NPR demonstrates 100% genuine parallel execution, establishing a new standard for self-evolving, efficient, and scalable agentic reasoning. 

---
# Training Language Models to Use Prolog as a Tool 

**Authors**: Niklas Mellgren, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2512.07407)  

**Abstract**: Ensuring reliable tool use is critical for safe agentic AI systems. Language models frequently produce unreliable reasoning with plausible but incorrect solutions that are difficult to verify. To address this, we investigate fine-tuning models to use Prolog as an external tool for verifiable computation. Using Group Relative Policy Optimization (GRPO), we fine-tune Qwen2.5-3B-Instruct on a cleaned GSM8K-Prolog-Prover dataset while varying (i) prompt structure, (ii) reward composition (execution, syntax, semantics, structure), and (iii) inference protocol: single-shot, best-of-N, and two agentic modes where Prolog is invoked internally or independently. Our reinforcement learning approach outperforms supervised fine-tuning, with our 3B model achieving zero-shot MMLU performance comparable to 7B few-shot results. Our findings reveal that: 1) joint tuning of prompt, reward, and inference shapes program syntax and logic; 2) best-of-N with external Prolog verification maximizes accuracy on GSM8K; 3) agentic inference with internal repair yields superior zero-shot generalization on MMLU-Stem and MMLU-Pro. These results demonstrate that grounding model reasoning in formal verification systems substantially improves reliability and auditability for safety-critical applications. The source code for reproducing our experiments is available under this https URL 

---
# An Analysis of Large Language Models for Simulating User Responses in Surveys 

**Authors**: Ziyun Yu, Yiru Zhou, Chen Zhao, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.06874)  

**Abstract**: Using Large Language Models (LLMs) to simulate user opinions has received growing attention. Yet LLMs, especially trained with reinforcement learning from human feedback (RLHF), are known to exhibit biases toward dominant viewpoints, raising concerns about their ability to represent users from diverse demographic and cultural backgrounds. In this work, we examine the extent to which LLMs can simulate human responses to cross-domain survey questions through direct prompting and chain-of-thought prompting. We further propose a claim diversification method CLAIMSIM, which elicits viewpoints from LLM parametric knowledge as contextual input. Experiments on the survey question answering task indicate that, while CLAIMSIM produces more diverse responses, both approaches struggle to accurately simulate users. Further analysis reveals two key limitations: (1) LLMs tend to maintain fixed viewpoints across varying demographic features, and generate single-perspective claims; and (2) when presented with conflicting claims, LLMs struggle to reason over nuanced differences among demographic features, limiting their ability to adapt responses to specific user profiles. 

---
# ProSocialAlign: Preference Conditioned Test Time Alignment in Language Models 

**Authors**: Somnath Banerjee, Sayan Layek, Sayantan Adak, Mykola Pechenizkiy, Animesh Mukherjee, Rima Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2512.06515)  

**Abstract**: Current language model safety paradigms often fall short in emotionally charged or high-stakes settings, where refusal-only approaches may alienate users and naive compliance can amplify risk. We propose ProSocialAlign, a test-time, parameter-efficient framework that steers generation toward safe, empathetic, and value-aligned responses without retraining the base model. We formalize five human-centered objectives and cast safety as lexicographic constrained generation: first, applying hard constraints to eliminate harmful continuations; then optimizing for prosocial quality within the safe set. Our method combines (i) directional regulation, a harm-mitigation mechanism that subtracts a learned "harm vector" in parameter space, and (ii) preference-aware autoregressive reward modeling trained jointly across attributes with gradient conflict resolution, enabling fine-grained, user-controllable decoding. Empirical evaluations across five safety benchmarks demonstrate state-of-the-art performance, reducing unsafe leakage and boosting alignment to human values, with strong gains across multiple evaluation metrics. ProSocialAlign offers a robust and modular foundation for generating context-sensitive, safe, and human-aligned responses at inference time. 

---
# PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory 

**Authors**: Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Gregory Wornell, Lyle Ungar, Dan Roth, Sihao Chen, Camillo Jose Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2512.06688)  

**Abstract**: Personalization is one of the next milestones in advancing AI capability and alignment. We introduce PersonaMem-v2, the state-of-the-art dataset for LLM personalization that simulates 1,000 realistic user-chatbot interactions on 300+ scenarios, 20,000+ user preferences, and 128k-token context windows, where most user preferences are implicitly revealed to reflect real-world interactions. Using this data, we investigate how reinforcement fine-tuning enables a model to improve its long-context reasoning capabilities for user understanding and personalization. We also develop a framework for training an agentic memory system, which maintains a single, human-readable memory that grows with each user over time.
In our experiments, frontier LLMs still struggle with implicit personalization, achieving only 37-48% accuracy. While they support long context windows, reasoning remains the bottleneck for implicit personalization tasks. Using reinforcement fine-tuning, we successfully train Qwen3-4B to outperforms GPT-5, reaching 53% accuracy in implicit personalization. Moreover, our agentic memory framework achieves state-of-the-art 55% accuracy while using 16x fewer input tokens, relying on a 2k-token memory instead of full 32k conversation histories. These results underscore the impact of our dataset and demonstrate agentic memory as a scalable path toward real-world personalized intelligence. 

---
# Nanbeige4-3B Technical Report: Exploring the Frontier of Small Language Models 

**Authors**: Chen Yang, Guangyue Peng, Jiaying Zhu, Ran Le, Ruixiang Feng, Tao Zhang, Wei Ruan, Xiaoqi Liu, Xiaoxue Cheng, Xiyun Xu, Yang Song, Yanzipeng Gao, Yiming Jia, Yun Xing, Yuntao Wen, Zekai Wang, Zhenwei An, Zhicong Sun, Zongchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.06266)  

**Abstract**: We present Nanbeige4-3B, a family of small-scale but high-performing language models. Pretrained on 23T high-quality tokens and finetuned on over 30 million diverse instructions, we extend the boundary of the scaling law for small language models. In pre-training, we design a Fine-Grained Warmup-Stable-Decay (FG-WSD) training scheduler, which progressively refines data mixtures across stages to boost model performance. In post-training, to improve the quality of the SFT data, we design a joint mechanism that integrates deliberative generation refinement and chain-of-thought reconstruction, yielding substantial gains on complex tasks. Following SFT, we employ our flagship reasoning model to distill Nanbeige4-3B through our proposed Dual Preference Distillation (DPD) method, which leads to further performance gains. Finally, a multi-stage reinforcement learning phase was applied, leveraging verifiable rewards and preference modeling to strengthen abilities on both reasoning and human alignment. Extensive evaluations show that Nanbeige4-3B not only significantly outperforms models of comparable parameter scale but also rivals much larger models across a wide range of benchmarks. The model checkpoints are available at this https URL. 

---
# Empathy by Design: Aligning Large Language Models for Healthcare Dialogue 

**Authors**: Emre Umucu, Guillermina Solis, Leon Garza, Emilia Rivas, Beatrice Lee, Anantaa Kotal, Aritran Piplai  

**Link**: [PDF](https://arxiv.org/pdf/2512.06097)  

**Abstract**: General-purpose large language models (LLMs) have demonstrated remarkable generative and reasoning capabilities but remain limited in healthcare and caregiving applications due to two key deficiencies: factual unreliability and a lack of empathetic communication. These shortcomings pose significant risks in sensitive contexts where users, particularly non-professionals and caregivers, seek medically relevant guidance or emotional reassurance. To address these challenges, we introduce a Direct Preference Optimization (DPO)-based alignment framework designed to improve factual correctness, semantic coherence, and human-centric qualities such as empathy, politeness, and simplicity in caregiver-patient dialogues. Our approach fine-tunes domain-adapted LLMs using pairwise preference data, where preferred responses reflect supportive and accessible communication styles while rejected ones represent prescriptive or overly technical tones. This direct optimization method aligns model outputs with human preferences more efficiently than traditional reinforcement-learning-based alignment. Empirical evaluations across multiple open and proprietary LLMs show that our DPO-tuned models achieve higher semantic alignment, improved factual accuracy, and stronger human-centric evaluation scores compared to baseline and commercial alternatives such as Google medical dialogue systems. These improvements demonstrate that preference-based alignment offers a scalable and transparent pathway toward developing trustworthy, empathetic, and clinically informed AI assistants for caregiver and healthcare communication. Our open-source code is available at: this https URL 

---
# Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models 

**Authors**: Fenghua Weng, Chaochao Lu, Xia Hu, Wenqi Shao, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07141)  

**Abstract**: As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at this https URL. 

---
# MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning 

**Authors**: Yueqian Wang, Songxiang Liu, Disong Wang, Nuo Xu, Guanglu Wan, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.06810)  

**Abstract**: Recent advances in video multimodal large language models (Video MLLMs) have significantly enhanced video understanding and multi-modal interaction capabilities. While most existing systems operate in a turn-based manner where the model can only reply after user turns, proactively deciding when to reply during video playback presents a promising yet challenging direction for real-time applications. In this work, we propose a novel text-to-text approach to proactive interaction, where the model autonomously determines whether to respond or remain silent at each turn based on dialogue history and visual context up to current frame of an streaming video. To overcome difficulties in previous methods such as manually tuning response decision thresholds and annotating precise reply times, we introduce a multi-turn RL based training method that encourages timely and accurate responses without requiring precise response time annotations. We train our model MMDuet2 on a dataset of 52k videos with two types of dialogues via SFT and RL. Experimental results demonstrate that MMDuet2 outperforms existing proactive Video MLLM baselines in response timing and quality, achieving state-of-the-art performance on the ProactiveVideoQA benchmark. 

---
# When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models 

**Authors**: Tong Xie, Andrew Bai, Yuanhao Ban, Yunqi Hong, Haoyu Li, Cho-jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2512.06343)  

**Abstract**: Reward models are central to Large Language Model (LLM) alignment within the framework of RLHF. The standard objective used in reward modeling is the Bradley-Terry (BT) loss, which learns from pairwise data consisting of a pair of chosen and rejected responses. In this work, we analyze the per-sample gradient of BT-loss and show that its norm scales with two distinct components: (1) the difference in predicted rewards between chosen and rejected responses, which reflects the prediction error, and critically, (2) representation distance between the pair measured in the output space of the final layer. While the first term captures the intended training signal, we show that the second term can significantly impact the update magnitude and misalign learning. Specifically, pairs with small representation distance often receive vanishingly weak updates, even when misranked, while pairs with large distance receive disproportionately strong updates. This leads to gradients from large-distance pairs to overshadow those from small-distance pairs, where fine-grained distinctions are especially important. To overcome this limitation, we propose NormBT, an adaptive pair-wise normalization scheme that balances representation-driven effects and focuses learning signals on prediction error. NormBT is a lightweight, drop-in integration to BT loss with negligible overhead. Across various LLM backbones and datasets, NormBT improves reward model performance consistently, with notable gains of over 5% on the Reasoning category of RewardBench, which contains numerous small-distance pairs. This work reveals a key limitation in the widely used BT objective and provides a simple, effective correction. 

---
# ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment 

**Authors**: Charlie Masters, Marta Grześkiewicz, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2512.06196)  

**Abstract**: As agents based on large language models are increasingly deployed to long-horizon tasks, maintaining their alignment with stakeholder preferences becomes critical. Effective alignment in such settings requires reward models that are interpretable so that stakeholders can understand and audit model objectives. Moreover, reward models must be capable of steering agents at interaction time, allowing preference shifts to be incorporated without retraining. We introduce ARCANE, a framework that frames alignment as a multi-agent collaboration problem that dynamically represents stakeholder preferences as natural-language rubrics: weighted sets of verifiable criteria that can be generated on-the-fly from task context. Inspired by utility theory, we formulate rubric learning as a reconstruction problem and apply a regularized Group-Sequence Policy Optimization (GSPO) procedure that balances interpretability, faithfulness, and computational efficiency. Using a corpus of 219 labeled rubrics derived from the GDPVal benchmark, we evaluate ARCANE on challenging tasks requiring multi-step reasoning and tool use. The learned rubrics produce compact, legible evaluations and enable configurable trade-offs (e.g., correctness vs. conciseness) without retraining. Our results show that rubric-based reward models offer a promising path toward interpretable, test-time adaptive alignment for complex, long-horizon AI systems. 

---
# From Show Programmes to Data: Designing a Workflow to Make Performing Arts Ephemera Accessible Through Language Models 

**Authors**: Clarisse Bardiot, Pierre-Carl Langlais, Bernard Jacquemin, Jacob Hart, Antonios Lagarias, Nicolas Foucault, Aurélie Lemaître-Legargeant, Jeanne Fras  

**Link**: [PDF](https://arxiv.org/pdf/2512.07452)  

**Abstract**: Many heritage institutions hold extensive collections of theatre programmes, which remain largely underused due to their complex layouts and lack of structured metadata. In this paper, we present a workflow for transforming such documents into structured data using a combination of multimodal large language models (LLMs), an ontology-based reasoning model, and a custom extension of the Linked Art framework. We show how vision-language models can accurately parse and transcribe born-digital and digitised programmes, achieving over 98% of correct extraction. To overcome the challenges of semantic annotation, we train a reasoning model (POntAvignon) using reinforcement learning with both formal and semantic rewards. This approach enables automated RDF triple generation and supports alignment with existing knowledge graphs. Through a case study based on the Festival d'Avignon corpus, we demonstrate the potential for large-scale, ontology-driven analysis of performing arts data. Our results open new possibilities for interoperable, explainable, and sustainable computational theatre historiography. 

---
# Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO for LLM Reasoning Enhancement 

**Authors**: Yongsheng Lian  

**Link**: [PDF](https://arxiv.org/pdf/2512.07611)  

**Abstract**: This study presents a systematic comparison of three Reinforcement Learning (RL) algorithms (PPO, GRPO, and DAPO) for improving complex reasoning in large language models (LLMs). Our main contribution is a controlled transfer-learning evaluation: models are first fine-tuned on the specialized Countdown Game and then assessed on a suite of general-purpose reasoning benchmarks. Across all tasks, RL-trained models outperform their corresponding base models, although the degree of improvement differs by benchmark.
Our parametric analysis offers practical guidance for RL-based LLM training. Increasing the group size in GRPO and DAPO leads to more stable training dynamics and higher accuracy, while the impact of the KL-penalty coefficient is non-monotonic. Additionally, we find that the Dynamic Sampling (DS) component in DAPO does not improve performance; in fact, the best overall results are achieved with DAPO when DS is disabled. 

---
# Each Prompt Matters: Scaling Reinforcement Learning Without Wasting Rollouts on Hundred-Billion-Scale MoE 

**Authors**: Anxiang Zeng, Haibo Zhang, Hailing Zhang, Kaixiang Mo, Liang Yao, Ling Hu, Long Zhang, Shuman Liu, Shuyi Xie, Yanshi Li, Yizhang Chen, Yuepeng Sheng, Yuwei Huang, Zhaochen Xu, Zhiqiang Zhou, Ziqin Liew  

**Link**: [PDF](https://arxiv.org/pdf/2512.07710)  

**Abstract**: We present CompassMax-V3-Thinking, a hundred-billion-scale MoE reasoning model trained with a new RL framework built on one principle: each prompt must matter. Scaling RL to this size exposes critical inefficiencies-zero-variance prompts that waste rollouts, unstable importance sampling over long horizons, advantage inversion from standard reward models, and systemic bottlenecks in rollout processing. To overcome these challenges, we introduce several unified innovations: (1) Multi-Stage Zero-Variance Elimination, which filters out non-informative prompts and stabilizes group-based policy optimization (e.g. GRPO) by removing wasted rollouts; (2) ESPO, an entropy-adaptive optimization method that balances token-level and sequence-level importance sampling to maintain stable learning dynamics; (3) a Router Replay strategy that aligns training-time MoE router decisions with inference-time behavior to mitigate train-infer discrepancies, coupled with a reward model adjustment to prevent advantage inversion; (4) a high-throughput RL system with FP8-precision rollouts, overlapped reward computation, and length-aware scheduling to eliminate performance bottlenecks. Together, these contributions form a cohesive pipeline that makes RL on hundred-billion-scale MoE models stable and efficient. The resulting model delivers strong performance across both internal and public evaluations. 

---
# JT-DA: Enhancing Data Analysis with Tool-Integrated Table Reasoning Large Language Models 

**Authors**: Ce Chi, Xing Wang, Zhendong Wang, Xiaofan Liu, Ce Li, Zhiyan Song, Chen Zhao, Kexin Yang, Boshen Shi, Jingjing Yang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.06859)  

**Abstract**: In this work, we present JT-DA-8B (JiuTian Data Analyst 8B), a specialized large language model designed for complex table reasoning tasks across diverse real-world scenarios. To address the lack of high-quality supervision in tabular reasoning scenarios, we construct a comprehensive and diverse training corpus with 34 well-defined table reasoning tasks, by aggregating 29 public table QA datasets and 3 million tables. An automatic pipeline is proposed to generate realistic multi-step analytical tasks involving reasoning patterns. The model is trained upon open-source JT-Coder-8B model, an 8B-parameter decoder-only foundation model trained from scratch. In the training stage, we leverage LLM-based scoring and workflow-aligned filtering to distill high-quality, table-centric data. Both supervised fine-tuning (SFT) and Reinforcement learning (RL) are adopted to optimize our model. Afterwards, a four-stage table reasoning workflow is proposed, including table preprocessing, table sensing, tool-integrated reasoning, and prompt engineering, to improve model interpretability and execution accuracy. Experimental results show that JT-DA-8B achieves strong performance in various table reasoning tasks, demonstrating the effectiveness of data-centric generation and workflow-driven optimization. 

---
# DaGRPO: Rectifying Gradient Conflict in Reasoning via Distinctiveness-Aware Group Relative Policy Optimization 

**Authors**: Xuan Xie, Xuan Wang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.06337)  

**Abstract**: The evolution of Large Language Models (LLMs) has catalyzed a paradigm shift from superficial instruction following to rigorous long-horizon reasoning. While Group Relative Policy Optimization (GRPO) has emerged as a pivotal mechanism for eliciting such post-training reasoning capabilities due to its exceptional performance, it remains plagued by significant training instability and poor sample efficiency. We theoretically identify the root cause of these issues as the lack of distinctiveness within on-policy rollouts: for routine queries, highly homogeneous samples induce destructive gradient conflicts; whereas for hard queries, the scarcity of valid positive samples results in ineffective optimization. To bridge this gap, we propose Distinctiveness-aware Group Relative Policy Optimization (DaGRPO). DaGRPO incorporates two core mechanisms: (1) Sequence-level Gradient Rectification, which utilizes fine-grained scoring to dynamically mask sample pairs with low distinctiveness, thereby eradicating gradient conflicts at the source; and (2) Off-policy Data Augmentation, which introduces high-quality anchors to recover training signals for challenging tasks. Extensive experiments across 9 mathematical reasoning and out-of-distribution (OOD) generalization benchmarks demonstrate that DaGRPO significantly surpasses existing SFT, GRPO, and hybrid baselines, achieving new state-of-the-art performance (e.g., a +4.7% average accuracy gain on math benchmarks). Furthermore, in-depth analysis confirms that DaGRPO effectively mitigates gradient explosion and accelerates the emergence of long-chain reasoning capabilities. 

---
# RLAX: Large-Scale, Distributed Reinforcement Learning for Large Language Models on TPUs 

**Authors**: Runlong Zhou, Lefan Zhang, Shang-Chen Wu, Kelvin Zou, Hanzhi Zhou, Ke Ye, Yihao Feng, Dong Yin, Alex Guillen Garcia, Dmytro Babych, Rohit Chatterjee, Matthew Hopkins, Xiang Kong, Chang Lan, Lezhi Li, Yiping Ma, Daniele Molinari, Senyu Tong, Yanchao Sun, Thomas Voice, Jianyu Wang, Chong Wang, Simon Wang, Floris Weers, Yechen Xu, Guolin Yin, Muyang Yu, Yi Zhang, Zheng Zhou, Danyang Zhuo, Ruoming Pang, Cheng Leong  

**Link**: [PDF](https://arxiv.org/pdf/2512.06392)  

**Abstract**: Reinforcement learning (RL) has emerged as the de-facto paradigm for improving the reasoning capabilities of large language models (LLMs). We have developed RLAX, a scalable RL framework on TPUs. RLAX employs a parameter-server architecture. A master trainer periodically pushes updated model weights to the parameter server while a fleet of inference workers pull the latest weights and generates new rollouts. We introduce a suite of system techniques to enable scalable and preemptible RL for a diverse set of state-of-art RL algorithms. To accelerate convergence and improve model quality, we have devised new dataset curation and alignment techniques. Large-scale evaluations show that RLAX improves QwQ-32B's pass@8 accuracy by 12.8% in just 12 hours 48 minutes on 1024 v5p TPUs, while remaining robust to preemptions during training. 

---
# RefBench-PRO: Perceptual and Reasoning Oriented Benchmark for Referring Expression Comprehension 

**Authors**: Tianyi Gao, Hao Li, Han Fang, Xin Wei, Xiaodong Dong, Hongbo Sun, Ye Yuan, Zhongjiang He, Jinglin Xu, Jingmin Xin, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.06276)  

**Abstract**: Referring Expression Comprehension (REC) is a vision-language task that localizes a specific image region based on a textual description. Existing REC benchmarks primarily evaluate perceptual capabilities and lack interpretable scoring mechanisms, which cannot reveal the grounding capability of Multi-modal Large Language Model (MLLM) across different cognitive abilities. To address this limitation, we introduce RefBench-PRO, a comprehensive REC benchmark, which decomposes referring expressions into two core dimensions, i.e., perception and reasoning, and further subdivides them into six progressively challenging tasks, such as attribute, position, interaction, commonsense, relation and reject. We also develop a fully automated data-generation pipeline that produces diverse referring expressions across these six sub-dimensions. Furthermore, We propose Ref-R1, an RL-based learning scheme, which incorporates Dynamic IoU-based GRPO to improve localization accuracy under increasingly complex reasoning conditions, establishing a stronger baseline for REC. Extensive experiments demonstrate that our RefBench-PRO enables interpretable evaluation of MLLM on referring expression comprehension, presenting greater challenges in both perception and reasoning. 

---
# Reinforcement Learning Integrated Agentic RAG for Software Test Cases Authoring 

**Authors**: Mohanakrishnan Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2512.06060)  

**Abstract**: This paper introduces a framework that integrates reinforcement learning (RL) with autonomous agents to enable continuous improvement in the automated process of software test cases authoring from business requirement documents within Quality Engineering (QE) workflows. Conventional systems employing Large Language Models (LLMs) generate test cases from static knowledge bases, which fundamentally limits their capacity to enhance performance over time. Our proposed Reinforcement Infused Agentic RAG (Retrieve, Augment, Generate) framework overcomes this limitation by employing AI agents that learn from QE feedback, assessments, and defect discovery outcomes to automatically improve their test case generation strategies. The system combines specialized agents with a hybrid vector-graph knowledge base that stores and retrieves software testing knowledge. Through advanced RL algorithms, specifically Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN), these agents optimize their behavior based on QE-reported test effectiveness, defect detection rates, and workflow metrics. As QEs execute AI-generated test cases and provide feedback, the system learns from this expert guidance to improve future iterations. Experimental validation on enterprise Apple projects yielded substantive improvements: a 2.4% increase in test generation accuracy (from 94.8% to 97.2%), and a 10.8% improvement in defect detection rates. The framework establishes a continuous knowledge refinement loop driven by QE expertise, resulting in progressively superior test case quality that enhances, rather than replaces, human testing capabilities. 

---
