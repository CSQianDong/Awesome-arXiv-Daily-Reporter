# TaoSR1: The Thinking Model for E-commerce Relevance Search 

**Authors**: Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12365)  

**Abstract**: Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification. 

---
# Reinforced Context Order Recovery for Adaptive Reasoning and Planning 

**Authors**: Long Ma, Fangwei Zhong, Yizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13070)  

**Abstract**: Modern causal language models, followed by rapid developments in discrete diffusion models, can now produce a wide variety of interesting and useful content. However, these families of models are predominantly trained to output tokens with a fixed (left-to-right) or random order, which may deviate from the logical order in which tokens are generated originally. In this paper, we observe that current causal and diffusion models encounter difficulties in problems that require adaptive token generation orders to solve tractably, which we characterize with the $\mathcal{V}$-information framework. Motivated by this, we propose Reinforced Context Order Recovery (ReCOR), a reinforcement-learning-based framework to extract adaptive, data-dependent token generation orders from text data without annotations. Self-supervised by token prediction statistics, ReCOR estimates the hardness of predicting every unfilled token and adaptively selects the next token during both training and inference. Experiments on challenging reasoning and planning datasets demonstrate the superior performance of ReCOR compared with baselines, sometimes outperforming oracle models supervised with the ground-truth order. 

---
# Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward 

**Authors**: Yong Deng, Guoqing Wang, Zhenzhe Ying, Xiaofeng Wu, Jinzhen Lin, Wenwen Xiong, Yuqin Dai, Shuo Yang, Zhanwei Zhang, Qiwen Wang, Yang Qin, Changhua Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.12800)  

**Abstract**: Large language models (LLMs) exhibit remarkable problem-solving abilities, but struggle with complex tasks due to static internal knowledge. Retrieval-Augmented Generation (RAG) enhances access to external information, yet remains limited in multi-hop reasoning and strategic search due to rigid workflows. Recent advancements in agentic deep research empower LLMs to autonomously reason, search, and synthesize information. However, current approaches relying on outcome-based reinforcement learning (RL) face critical issues such as conflicting gradients and reward sparsity, limiting performance gains and training efficiency. To address these, we first propose Atomic Thought, a novel LLM thinking paradigm that decomposes reasoning into fine-grained functional units. These units are supervised by Reasoning Reward Models (RRMs), which provide Atomic Thought Rewards (ATR) for fine-grained guidance. Building on this, we propose Atom-Searcher, a novel RL framework for agentic deep research that integrates Atomic Thought and ATR. Atom-Searcher uses a curriculum-inspired reward schedule, prioritizing process-level ATR early and transitioning to outcome rewards, accelerating convergence on effective reasoning paths. Experiments on seven benchmarks show consistent improvements over the state-of-the-art. Key advantages include: (1) Atom-Searcher scales computation at test-time. (2) Atomic Thought provides supervision anchors for RRMs, bridging deep research tasks and RRMs. (3) Atom-Searcher exhibits more interpretable, human-like reasoning patterns. 

---
# Integrating Feedback Loss from Bi-modal Sarcasm Detector for Sarcastic Speech Synthesis 

**Authors**: Zhu Li, Yuqing Zhang, Xiyuan Gao, Devraj Raghuvanshi, Nagendra Kumar, Shekhar Nayak, Matt Coler  

**Link**: [PDF](https://arxiv.org/pdf/2508.13028)  

**Abstract**: Sarcastic speech synthesis, which involves generating speech that effectively conveys sarcasm, is essential for enhancing natural interactions in applications such as entertainment and human-computer interaction. However, synthesizing sarcastic speech remains a challenge due to the nuanced prosody that characterizes sarcasm, as well as the limited availability of annotated sarcastic speech data. To address these challenges, this study introduces a novel approach that integrates feedback loss from a bi-modal sarcasm detection model into the TTS training process, enhancing the model's ability to capture and convey sarcasm. In addition, by leveraging transfer learning, a speech synthesis model pre-trained on read speech undergoes a two-stage fine-tuning process. First, it is fine-tuned on a diverse dataset encompassing various speech styles, including sarcastic speech. In the second stage, the model is further refined using a dataset focused specifically on sarcastic speech, enhancing its ability to generate sarcasm-aware speech. Objective and subjective evaluations demonstrate that our proposed methods improve the quality, naturalness, and sarcasm-awareness of synthesized speech. 

---
# ReaLM: Reflection-Enhanced Autonomous Reasoning with Small Language Models 

**Authors**: Yuanfeng Xu, Zehui Dai, Jian Liang, Jiapeng Guan, Guangrun Wang, Liang Lin, Xiaohui Lv  

**Link**: [PDF](https://arxiv.org/pdf/2508.12387)  

**Abstract**: Small Language Models (SLMs) are a cost-effective alternative to Large Language Models (LLMs), but often struggle with complex reasoning due to their limited capacity and a tendency to produce mistakes or inconsistent answers during multi-step reasoning. Existing efforts have improved SLM performance, but typically at the cost of one or more of three key aspects: (1) reasoning capability, due to biased supervision that filters out negative reasoning paths and limits learning from errors; (2) autonomy, due to over-reliance on externally generated reasoning signals; and (3) generalization, which suffers when models overfit to teacher-specific patterns. In this paper, we introduce ReaLM, a reinforcement learning framework for robust and self-sufficient reasoning in vertical domains. To enhance reasoning capability, we propose Multi-Route Process Verification (MRPV), which contrasts both positive and negative reasoning paths to extract decisive patterns. To reduce reliance on external guidance and improve autonomy, we introduce Enabling Autonomy via Asymptotic Induction (EAAI), a training strategy that gradually fades external signals. To improve generalization, we apply guided chain-of-thought distillation to encode domain-specific rules and expert knowledge into SLM parameters, making them part of what the model has learned. Extensive experiments on both vertical and general reasoning tasks demonstrate that ReaLM significantly improves SLM performance across aspects (1)-(3) above. 

---
# M3PO: Multimodal-Model-Guided Preference Optimization for Visual Instruction Following 

**Authors**: Ruirui Gao, Emily Johnson, Bowen Tan, Yanfei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.12458)  

**Abstract**: Large Vision-Language Models (LVLMs) hold immense potential for complex multimodal instruction following, yet their development is often hindered by the high cost and inconsistency of human annotation required for effective fine-tuning and preference alignment. Traditional supervised fine-tuning (SFT) and existing preference optimization methods like RLHF and DPO frequently struggle to efficiently leverage the model's own generation space to identify highly informative "hard negative" samples. To address these challenges, we propose Multimodal-Model-Guided Preference Optimization (M3PO), a novel and data-efficient method designed to enhance LVLMs' capabilities in visual instruction following. M3PO intelligently selects the most "learning-valuable" preference sample pairs from a diverse pool of LVLM-generated candidates. This selection is driven by a sophisticated mechanism that integrates two crucial signals: a Multimodal Alignment Score (MAS) to assess external quality and the model's Self-Consistency / Confidence (log-probability) to gauge internal belief. These are combined into a novel M3P-Score, which specifically identifies preferred responses and challenging dispreferred responses that the model might confidently generate despite being incorrect. These high-quality preference pairs are then used for efficient Direct Preference Optimization (DPO) fine-tuning on base LVLMs like LLaVA-1.5 (7B/13B) using LoRA. Our extensive experiments demonstrate that M3PO consistently outperforms strong baselines, including SFT, simulated RLHF, vanilla DPO, and RM-DPO, across a comprehensive suite of multimodal instruction following benchmarks (MME-Bench, POPE, IFT, Human Pref. Score). 

---
# Legal$Δ$: Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain 

**Authors**: Xin Dai, Buqiang Xu, Zhenghao Liu, Yukun Yan, Huiyuan Xie, Xiaoyuan Yi, Shuo Wang, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12281)  

**Abstract**: Legal Artificial Intelligence (LegalAI) has achieved notable advances in automating judicial decision-making with the support of Large Language Models (LLMs). However, existing legal LLMs still struggle to generate reliable and interpretable reasoning processes. They often default to fast-thinking behavior by producing direct answers without explicit multi-step reasoning, limiting their effectiveness in complex legal scenarios that demand rigorous justification. To address this challenge, we propose Legal$\Delta$, a reinforcement learning framework designed to enhance legal reasoning through chain-of-thought guided information gain. During training, Legal$\Delta$ employs a dual-mode input setup-comprising direct answer and reasoning-augmented modes-and maximizes the information gain between them. This encourages the model to acquire meaningful reasoning patterns rather than generating superficial or redundant explanations. Legal$\Delta$ follows a two-stage approach: (1) distilling latent reasoning capabilities from a powerful Large Reasoning Model (LRM), DeepSeek-R1, and (2) refining reasoning quality via differential comparisons, combined with a multidimensional reward mechanism that assesses both structural coherence and legal-domain specificity. Experimental results on multiple legal reasoning tasks demonstrate that Legal$\Delta$ outperforms strong baselines in both accuracy and interpretability. It consistently produces more robust and trustworthy legal judgments without relying on labeled preference data. All code and data will be released at this https URL. 

---
# Vision-G1: Towards General Vision Language Reasoning with Multi-Domain Data Curation 

**Authors**: Yuheng Zha, Kun Zhou, Yujia Wu, Yushu Wang, Jie Feng, Zhi Xu, Shibo Hao, Zhengzhong Liu, Eric P. Xing, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12680)  

**Abstract**: Despite their success, current training pipelines for reasoning VLMs focus on a limited range of tasks, such as mathematical and logical reasoning. As a result, these models face difficulties in generalizing their reasoning capabilities to a wide range of domains, primarily due to the scarcity of readily available and verifiable reward data beyond these narrowly defined areas. Moreover, integrating data from multiple domains is challenging, as the compatibility between domain-specific datasets remains uncertain. To address these limitations, we build a comprehensive RL-ready visual reasoning dataset from 46 data sources across 8 dimensions, covering a wide range of tasks such as infographic, mathematical, spatial, cross-image, graphic user interface, medical, common sense and general science. We propose an influence function based data selection and difficulty based filtering strategy to identify high-quality training samples from this dataset. Subsequently, we train the VLM, referred to as Vision-G1, using multi-round RL with a data curriculum to iteratively improve its visual reasoning capabilities. Our model achieves state-of-the-art performance across various visual reasoning benchmarks, outperforming similar-sized VLMs and even proprietary models like GPT-4o and Gemini-1.5 Flash. The model, code and dataset are publicly available at this https URL. 

---
# Reinforcement Learning with Rubric Anchors 

**Authors**: Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, Xijun Gu, Peiyi Tu, Jiaxin Liu, Wenyu Chen, Yuzhuo Fu, Zhiting Fan, Yanmei Gu, Yuanyuan Wang, Zhengkai Yang, Jianguo Li, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12790)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing Large Language Models (LLMs), exemplified by the success of OpenAI's o-series. In RLVR, rewards are derived from verifiable signals-such as passing unit tests in code generation or matching correct answers in mathematical reasoning. While effective, this requirement largely confines RLVR to domains with automatically checkable outcomes. To overcome this, we extend the RLVR paradigm to open-ended tasks by integrating rubric-based rewards, where carefully designed rubrics serve as structured, model-interpretable criteria for automatic scoring of subjective outputs. We construct, to our knowledge, the largest rubric reward system to date, with over 10,000 rubrics from humans, LLMs, or a hybrid human-LLM collaboration. Implementing rubric-based RL is challenging; we tackle these issues with a clear framework and present an open-sourced Qwen-30B-A3B model with notable gains: 1) With only 5K+ samples, our system improves by +5.2% on open-ended benchmarks (especially humanities), outperforming a 671B DeepSeek-V3 model by +2.4%, while preserving general and reasoning abilities. 2) Our method provides fine-grained stylistic control, using rubrics as anchors to mitigate the "AI-like" tone and produce more human-like, expressive responses. We share key lessons in rubric construction, data selection, and training, and discuss limitations and future releases. 

---
# Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position 

**Authors**: Zhixin Xie, Xurui Song, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.12398)  

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a competitive non-autoregressive paradigm due to their unique training and inference approach. However, there is currently a lack of safety study on this novel architecture. In this paper, we present the first analysis of dLLMs' safety performance and propose a novel safety alignment method tailored to their unique generation characteristics. Specifically, we identify a critical asymmetry between the defender and attacker in terms of security. For the defender, we reveal that the middle tokens of the response, rather than the initial ones, are more critical to the overall safety of dLLM outputs; this seems to suggest that aligning middle tokens can be more beneficial to the defender. The attacker, on the contrary, may have limited power to manipulate middle tokens, as we find dLLMs have a strong tendency towards a sequential generation order in practice, forcing the attack to meet this distribution and diverting it from influencing the critical middle tokens. Building on this asymmetry, we introduce Middle-tOken Safety Alignment (MOSA), a novel method that directly aligns the model's middle generation with safe refusals exploiting reinforcement learning. We implement MOSA and compare its security performance against eight attack methods on two benchmarks. We also test the utility of MOSA-aligned dLLM on coding, math, and general reasoning. The results strongly prove the superiority of MOSA. 

---
# Ovis2.5 Technical Report 

**Authors**: Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, Yuxuan Han, Haijun Li, Wanying Chen, Junke Tang, Chengkun Hou, Zhixing Du, Tianli Zhou, Wenjie Zhang, Huping Ding, Jiahe Li, Wen Li, Gui Hu, Yiliang Gu, Siran Yang, Jiamang Wang, Hailong Sun, Yibo Wang, Hui Sun, Jinlong Huang, Yuping He, Shengze Shi, Weihong Zhang, Guodong Zheng, Junpeng Jiang, Sensen Gao, Yi-Feng Wu, Sijia Chen, Yuhui Chen, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11737)  

**Abstract**: We present Ovis2.5, a successor to Ovis2 designed for native-resolution visual perception and strong multimodal reasoning. Ovis2.5 integrates a native-resolution vision transformer that processes images at their native, variable resolutions, avoiding the degradation from fixed-resolution tiling and preserving both fine detail and global layout -- crucial for visually dense content like complex charts. To strengthen reasoning, we train the model to move beyond linear chain-of-thought and perform reflection -- including self-checking and revision. This advanced capability is exposed as an optional "thinking mode" at inference time, allowing users to trade latency for enhanced accuracy on difficult inputs. The model is trained via a comprehensive five-phase curriculum that progressively builds its skills. The process begins with foundational visual and multimodal pretraining, advances through large-scale instruction tuning, and culminates in alignment and reasoning enhancement using DPO and GRPO. To scale these upgrades efficiently, we employ multimodal data packing and hybrid parallelism, yielding a significant end-to-end speedup. We release two open-source models: Ovis2.5-9B and Ovis2.5-2B. The latter continues the "small model, big performance" philosophy of Ovis2, making it ideal for resource-constrained, on-device scenarios. On the OpenCompass multimodal leaderboard, Ovis2.5-9B averages 78.3, marking a substantial improvement over its predecessor, Ovis2-8B, and achieving state-of-the-art results among open-source MLLMs in the sub-40B parameter range; Ovis2.5-2B scores 73.9, establishing SOTA for its size. Beyond aggregate scores, Ovis2.5 achieves leading results on STEM benchmarks, exhibits strong capabilities on grounding and video tasks, and achieves open-source SOTA at its scale for complex chart analysis. 

---
# G$^2$RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance 

**Authors**: Yongxin Guo, Wenbo Deng, Zhenglin Cheng, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13023)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has markedly enhanced the reasoning abilities of large language models (LLMs). Its success, however, largely depends on strong base models with rich world knowledge, yielding only modest improvements for small-size language models (SLMs). To address this limitation, we investigate Guided GRPO, which injects ground-truth reasoning steps into roll-out trajectories to compensate for SLMs' inherent weaknesses. Through a comprehensive study of various guidance configurations, we find that naively adding guidance delivers limited gains. These insights motivate G$^2$RPO-A, an adaptive algorithm that automatically adjusts guidance strength in response to the model's evolving training dynamics. Experiments on mathematical reasoning and code-generation benchmarks confirm that G$^2$RPO-A substantially outperforms vanilla GRPO. Our code and models are available at this https URL. 

---
# Towards Open-Ended Emotional Support Conversations in LLMs via Reinforcement Learning with Future-Oriented Rewards 

**Authors**: Ting Yang, Li Chen, Huimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12935)  

**Abstract**: Emotional Support Conversation (ESC) systems aim to alleviate users' emotional difficulties and provide long-term, systematic support for emotional well-being. However, most large language model (LLM)-based ESC systems rely on predefined strategies, which limits their effectiveness in complex, real-life scenarios. To enable flexible responses to diverse emotional problem scenarios, this paper introduces a novel end-to-end framework (RLFF-ESC) that directly learns enduring emotionally supportive response skills using reinforcement learning. For sustained emotional support, we first employ an LLM-based multi-agent mechanism to simulate future dialogue trajectories and collect future-oriented rewards. We then train a future-oriented reward model, which is subsequently used to train the emotional support policy model. Additionally, we incorporate an explicit reasoning process during response generation to further enhance the quality, relevance, and contextual appropriateness of the system's responses. We evaluate the backbone policy model on Qwen2.5-7B-Instruct-1M and LLaMA3.1-8B-Instruct models, testing the proposed RLFF-ESC framework across two public ESC datasets. Experimental results demonstrate that RLFF-ESC consistently outperforms existing baselines in terms of goal completion and response quality. 

---
# Beyond Ethical Alignment: Evaluating LLMs as Artificial Moral Assistants 

**Authors**: Alessio Galatolo, Luca Alberto Rappuoli, Katie Winkle, Meriem Beloucif  

**Link**: [PDF](https://arxiv.org/pdf/2508.12754)  

**Abstract**: The recent rise in popularity of large language models (LLMs) has prompted considerable concerns about their moral capabilities. Although considerable effort has been dedicated to aligning LLMs with human moral values, existing benchmarks and evaluations remain largely superficial, typically measuring alignment based on final ethical verdicts rather than explicit moral reasoning. In response, this paper aims to advance the investigation of LLMs' moral capabilities by examining their capacity to function as Artificial Moral Assistants (AMAs), systems envisioned in the philosophical literature to support human moral deliberation. We assert that qualifying as an AMA requires more than what state-of-the-art alignment techniques aim to achieve: not only must AMAs be able to discern ethically problematic situations, they should also be able to actively reason about them, navigating between conflicting values outside of those embedded in the alignment phase. Building on existing philosophical literature, we begin by designing a new formal framework of the specific kind of behaviour an AMA should exhibit, individuating key qualities such as deductive and abductive moral reasoning. Drawing on this theoretical framework, we develop a benchmark to test these qualities and evaluate popular open LLMs against it. Our results reveal considerable variability across models and highlight persistent shortcomings, particularly regarding abductive moral reasoning. Our work connects theoretical philosophy with practical AI evaluation while also emphasising the need for dedicated strategies to explicitly enhance moral reasoning capabilities in LLMs. Code available at this https URL 

---
# Wisdom of the Crowd: Reinforcement Learning from Coevolutionary Collective Feedback 

**Authors**: Wenzhen Yuan, Shengji Tang, Weihao Lin, Jiacheng Ruan, Ganqu Cui, Bo Zhang, Tao Chen, Ting Liu, Yuzhuo Fu, Peng Ye, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12338)  

**Abstract**: Reinforcement learning (RL) has significantly enhanced the reasoning capabilities of large language models (LLMs), but its reliance on expensive human-labeled data or complex reward models severely limits scalability. While existing self-feedback methods aim to address this problem, they are constrained by the capabilities of a single model, which can lead to overconfidence in incorrect answers, reward hacking, and even training collapse. To this end, we propose Reinforcement Learning from Coevolutionary Collective Feedback (RLCCF), a novel RL framework that enables multi-model collaborative evolution without external supervision. Specifically, RLCCF optimizes the ability of a model collective by maximizing its Collective Consistency (CC), which jointly trains a diverse ensemble of LLMs and provides reward signals by voting on collective outputs. Moreover, each model's vote is weighted by its Self-Consistency (SC) score, ensuring that more confident models contribute more to the collective decision. Benefiting from the diverse output distributions and complementary abilities of multiple LLMs, RLCCF enables the model collective to continuously enhance its reasoning ability through coevolution. Experiments on four mainstream open-source LLMs across four mathematical reasoning benchmarks demonstrate that our framework yields significant performance gains, achieving an average relative improvement of 16.72\% in accuracy. Notably, RLCCF not only improves the performance of individual models but also enhances the group's majority-voting accuracy by 4.51\%, demonstrating its ability to extend the collective capability boundary of the model collective. 

---
# RLNVR: Reinforcement Learning from Non-Verified Real-World Rewards 

**Authors**: Rohit Krishnan, Jon Evans  

**Link**: [PDF](https://arxiv.org/pdf/2508.12165)  

**Abstract**: This paper introduces RLNVR (Reinforcement Learning from Non-Verified Rewards), a framework for training language models using noisy, real-world feedback signals without requiring explicit human verification. Traditional RLHF requires expensive, verified reward signals that are impractical in many real-world domains. RLNVR addresses this challenge through baseline normalization and semantic similarity-based reward transfer. We demonstrate RLNVR through Walter, a prototype system that optimizes social media content generation using actual engagement data from Bluesky. Our experimental results show significant improvements in content quality and training stability, with comprehensive evaluation planned for future work. Positioning: We present a practical framework that combines RLNVR with GSPO (Group Sequence Policy Optimization) and an optional UED (Unsupervised Environment Design) curriculum to improve stability and diversity under noisy, implicit rewards. To our knowledge, combining GSPO-style normalization with a UED-style curriculum for LLM content generation from implicit social engagement has not been previously documented in this applied setting; we frame this as an applied integration rather than a new algorithm. 

---
# Overcoming Knowledge Discrepancies: Structuring Reasoning Threads through Knowledge Balancing in Interactive Scenarios 

**Authors**: Daniel Burkhardt, Xiangwei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.12100)  

**Abstract**: Reasoning in interactive problem solving scenarios requires models to construct reasoning threads that reflect user understanding and align with structured domain knowledge. However, current reasoning models often lack explicit semantic hierarchies, user-domain knowledge alignment, and principled mechanisms to prune reasoning threads for effectiveness. These limitations result in lengthy generic output that does not guide users through goal-oriented reasoning steps. To address this, we propose a prototype-inspired, two-phases Reasoning-Threads-Evaluation (ReT-Eval) framework, drawing inspiration from human-like reasoning strategies that emphasize structured knowledge reuse. In the first phase, semantically relevant knowledge structures are extracted from a sparse domain knowledge graph using a graph neural network and enriched with intrinsic large language model knowledge to resolve knowledge discrepancies. In the second phase, these threads are evaluated and pruned using a reward-guided strategy aimed at maintaining semantic coherence to generate effective reasoning threads. Experiments and expert evaluations show that ReT-Eval enhances user understanding and outperforms state-of-the-art reasoning models. 

---
# SSPO: Self-traced Step-wise Preference Optimization for Process Supervision and Reasoning Compression 

**Authors**: Yuyang Xu, Yi Cheng, Haochao Ying, Zhuoyun Du, Renjun Hu, Xing Shi, Wei Lin, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12604)  

**Abstract**: Test-time scaling has proven effective in further enhancing the performance of pretrained Large Language Models (LLMs). However, mainstream post-training methods (i.e., reinforcement learning (RL) with chain-of-thought (CoT) reasoning) often incur substantial computational overhead due to auxiliary models and overthinking. In this paper, we empirically reveal that the incorrect answers partially stem from verbose reasoning processes lacking correct self-fix, where errors accumulate across multiple reasoning steps. To this end, we propose Self-traced Step-wise Preference Optimization (SSPO), a pluggable RL process supervision framework that enables fine-grained optimization of each reasoning step. Specifically, SSPO requires neither auxiliary models nor stepwise manual annotations. Instead, it leverages step-wise preference signals generated by the model itself to guide the optimization process for reasoning compression. Experiments demonstrate that the generated reasoning sequences from SSPO are both accurate and succinct, effectively mitigating overthinking behaviors without compromising model performance across diverse domains and languages. 

---
