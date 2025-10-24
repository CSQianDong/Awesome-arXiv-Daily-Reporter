# Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning 

**Authors**: Yaochen Zhu, Harald Steck, Dawen Liang, Yinhan He, Jundong Li, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2510.20150)  

**Abstract**: Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at this https URL. 

---
# Plan Then Retrieve: Reinforcement Learning-Guided Complex Reasoning over Knowledge Graphs 

**Authors**: Yanlin Song, Ben Liu, Víctor Gutiérrez-Basulto, Zhiwei Hu, Qianqian Xie, Min Peng, Sophia Ananiadou, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20691)  

**Abstract**: Knowledge Graph Question Answering aims to answer natural language questions by reasoning over structured knowledge graphs. While large language models have advanced KGQA through their strong reasoning capabilities, existing methods continue to struggle to fully exploit both the rich knowledge encoded in KGs and the reasoning capabilities of LLMs, particularly in complex scenarios. They often assume complete KG coverage and lack mechanisms to judge when external information is needed, and their reasoning remains locally myopic, failing to maintain coherent multi-step planning, leading to reasoning failures even when relevant knowledge exists. We propose Graph-RFT, a novel two-stage reinforcement fine-tuning KGQA framework with a 'plan-KGsearch-and-Websearch-during-think' paradigm, that enables LLMs to perform autonomous planning and adaptive retrieval scheduling across KG and web sources under incomplete knowledge conditions. Graph-RFT introduces a chain-of-thought fine-tuning method with a customized plan-retrieval dataset activates structured reasoning and resolves the GRPO cold-start problem. It then introduces a novel plan-retrieval guided reinforcement learning process integrates explicit planning and retrieval actions with a multi-reward design, enabling coverage-aware retrieval scheduling. It employs a Cartesian-inspired planning module to decompose complex questions into ordered subquestions, and logical expression to guide tool invocation for globally consistent multi-step reasoning. This reasoning retrieval process is optimized with a multi-reward combining outcome and retrieval specific signals, enabling the model to learn when and how to combine KG and web retrieval effectively. 

---
# GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning 

**Authors**: Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20548)  

**Abstract**: Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1. 

---
# Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding 

**Authors**: Yuhang Zhou, Mingrui Zhang, Ke Li, Mingyi Wang, Qiao Liu, Qifei wang, Jiayi Liu, Fei Liu, Serena Li, Weiwi Li, Mingze Gao, Abhishek Kumar, Xiangjun Fan, Zhuokai Zhao, Lizhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20176)  

**Abstract**: Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding. 

---
# Robust Preference Alignment via Directional Neighborhood Consensus 

**Authors**: Ruochen Mao, Yuling Shi, Xiaodong Gu, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.20498)  

**Abstract**: Aligning large language models with human preferences is critical for creating reliable and controllable AI systems. A human preference can be visualized as a high-dimensional vector where different directions represent trade-offs between desired attributes (e.g., helpfulness vs. verbosity). Yet, because the training data often reflects dominant, average preferences, LLMs tend to perform well on common requests but fall short in specific, individual needs. This mismatch creates a preference coverage gap. Existing methods often address this through costly retraining, which may not be generalized to the full spectrum of diverse preferences. This brittleness means that when a user's request reflects a nuanced preference deviating from the training data's central tendency, model performance can degrade unpredictably. To address this challenge, we introduce Robust Preference Selection (RPS), a post-hoc, training-free method by leveraging directional neighborhood consensus. Instead of forcing a model to generate a response from a single, highly specific preference, RPS samples multiple responses from a local neighborhood of related preferences to create a superior candidate pool. It then selects the response that best aligns with the user's original intent. We provide a theoretical framework showing our neighborhood generation strategy is provably superior to a strong baseline that also samples multiple candidates. Comprehensive experiments across three distinct alignment paradigms (DPA, DPO, and SFT) demonstrate that RPS consistently improves robustness against this baseline, achieving win rates of up to 69% on challenging preferences from under-represented regions of the space without any model retraining. Our work presents a practical, theoretically-grounded solution for enhancing the reliability of preference-aligned models. 

---
# LM-mixup: Text Data Augmentation via Language Model based Mixup 

**Authors**: Zhijie Deng, Zhouan Shen, Ling Li, Yao Zhou, Zhaowei Zhu, Yanji He, Wei Wang, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.20449)  

**Abstract**: Instruction tuning is crucial for aligning Large Language Models (LLMs), yet the quality of instruction-following data varies significantly. While high-quality data is paramount, it is often scarce; conversely, abundant low-quality data is frequently discarded, leading to substantial information loss. Existing data augmentation methods struggle to augment this low-quality data effectively, and the evaluation of such techniques remains poorly defined. To address this, we formally define the task of Instruction Distillation: distilling multiple low-quality and redundant inputs into high-quality and coherent instruction-output pairs. Specifically, we introduce a comprehensive data construction pipeline to create MIXTURE, a 144K-sample dataset pairing low-quality or semantically redundant imperfect instruction clusters with their high-quality distillations. We then introduce LM-Mixup, by first performing supervised fine-tuning on MIXTURE and then optimizing it with reinforcement learning. This process uses three complementary reward signals: quality, semantic alignment, and format compliance, via Group Relative Policy Optimization (GRPO). We demonstrate that LM-Mixup effectively augments imperfect datasets: fine-tuning LLMs on its distilled data, which accounts for only about 3% of the entire dataset, not only surpasses full-dataset training but also competes with state-of-the-art high-quality data selection methods across multiple benchmarks. Our work establishes that low-quality data is a valuable resource when properly distilled and augmented with LM-Mixup, significantly enhancing the efficiency and performance of instruction-tuned LLMs. 

---
# Dialogue Is Not Enough to Make a Communicative BabyLM (But Neither Is Developmentally Inspired Reinforcement Learning) 

**Authors**: Francesca Padovani, Bastian Bunzeck, Manar Ali, Omar Momen, Arianna Bisazza, Hendrik Buschmeier, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2510.20358)  

**Abstract**: We investigate whether pre-training exclusively on dialogue data results in formally and functionally apt small language models. Based on this pre-trained llamalogue model, we employ a variety of fine-tuning strategies to enforce "more communicative" text generations by our models. Although our models underperform on most standard BabyLM benchmarks, they excel at dialogue continuation prediction in a minimal pair setting. While PPO fine-tuning has mixed to adversarial effects on our models, DPO fine-tuning further improves their performance on our custom dialogue benchmark. 

---
# BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation 

**Authors**: Haoyuan Li, Zhengyuan Shen, Sullam Jeoung, Yueyan Chen, Jiayu Li, Qi Zhu, Shuai Wang, Vassilis Ioannidis, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2510.20151)  

**Abstract**: As structured texts become increasingly complex across diverse domains -- from technical reports to generative AI prompts -- the need for text segmentation into semantically meaningful components becomes critical. Such texts often contain elements beyond plain language, including tables, code snippets, and placeholders, which conventional sentence- or paragraph-level segmentation methods cannot handle effectively. To address this challenge, we propose BoundRL, a novel and efficient approach that jointly performs token-level text segmentation and label prediction for long structured texts. Instead of generating complete contents for each segment, it generates only a sequence of starting tokens and reconstructs the complete contents by locating these tokens within the original texts, thereby reducing inference costs by orders of magnitude and minimizing hallucination. To adapt the model for the output format, BoundRL~performs reinforcement learning with verifiable rewards (RLVR) with a specifically designed reward that jointly optimizes document reconstruction fidelity and semantic alignment. To mitigate entropy collapse, it further constructs intermediate candidates by systematically perturbing a fraction of generated sequences of segments to create stepping stones toward higher-quality solutions. To demonstrate BoundRL's effectiveness on particularly challenging structured texts, we focus evaluation on complex prompts used for LLM applications. Experiments show that BoundRL enables small language models (1.7B parameters) to outperform few-shot prompting of much larger models. Moreover, RLVR with our designed reward yields significant improvements over supervised fine-tuning, and incorporating intermediate candidates further improves both performance and generalization. 

---
# Enhancing Reasoning Skills in Small Persian Medical Language Models Can Outperform Large-Scale Data Training 

**Authors**: Mehrdad Ghassabi, Sadra Hakim, Hamidreza Baradaran Kashani, Pedram Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2510.20059)  

**Abstract**: Enhancing reasoning capabilities in small language models is critical for specialized applications such as medical question answering, particularly in underrepresented languages like Persian. In this study, we employ Reinforcement Learning with AI Feedback (RLAIF) and Direct preference optimization (DPO) to improve the reasoning skills of a general-purpose Persian language model. To achieve this, we translated a multiple-choice medical question-answering dataset into Persian and used RLAIF to generate rejected-preferred answer pairs, which are essential for DPO training. By prompting both teacher and student models to produce Chain-of-Thought (CoT) reasoning responses, we compiled a dataset containing correct and incorrect reasoning trajectories. This dataset, comprising 2 million tokens in preferred answers and 2.5 million tokens in rejected ones, was used to train a baseline model, significantly enhancing its medical reasoning capabilities in Persian. Remarkably, the resulting model outperformed its predecessor, gaokerena-V, which was trained on approximately 57 million tokens, despite leveraging a much smaller dataset. These results highlight the efficiency and effectiveness of reasoning-focused training approaches in developing domain-specific language models with limited data availability. 

---
# Every Question Has Its Own Value: Reinforcement Learning with Explicit Human Values 

**Authors**: Dian Yu, Yulai Zhao, Kishan Panaganti, Linfeng Song, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20187)  

**Abstract**: We propose Reinforcement Learning with Explicit Human Values (RLEV), a method that aligns Large Language Model (LLM) optimization directly with quantifiable human value signals. While Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains models in objective domains using binary correctness rewards, it overlooks that not all tasks are equally significant. RLEV extends this framework by incorporating human-defined value signals directly into the reward function. Using exam-style data with explicit ground-truth value labels, RLEV consistently outperforms correctness-only baselines across multiple RL algorithms and model scales. Crucially, RLEV policies not only improve value-weighted accuracy but also learn a value-sensitive termination policy: concise for low-value prompts, thorough for high-value ones. We demonstrate this behavior stems from value-weighted gradient amplification on end-of-sequence tokens. Ablation studies confirm the gain is causally linked to value alignment. RLEV remains robust under noisy value signals, such as difficulty-based labels, demonstrating that optimizing for an explicit utility function offers a practical path to aligning LLMs with human priorities. 

---
