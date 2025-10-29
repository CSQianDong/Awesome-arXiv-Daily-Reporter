# MetricX-25 and GemSpanEval: Google Translate Submissions to the WMT25 Evaluation Shared Task 

**Authors**: Juraj Juraska, Tobias Domhan, Mara Finkelstein, Tetsuji Nakagawa, Geza Kovacs, Daniel Deutsch, Pidong Wang, Markus Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2510.24707)  

**Abstract**: In this paper, we present our submissions to the unified WMT25 Translation Evaluation Shared Task. For the Quality Score Prediction subtask, we create a new generation of MetricX with improvements in the input format and the training protocol, while for the Error Span Detection subtask we develop a new model, GemSpanEval, trained to predict error spans along with their severities and categories. Both systems are based on the state-of-the-art multilingual open-weights model Gemma 3, fine-tuned on publicly available WMT data. We demonstrate that MetricX-25, adapting Gemma 3 to an encoder-only architecture with a regression head on top, can be trained to effectively predict both MQM and ESA quality scores, and significantly outperforms its predecessor. Our decoder-only GemSpanEval model, on the other hand, we show to be competitive in error span detection with xCOMET, a strong encoder-only sequence-tagging baseline. With error span detection formulated as a generative task, we instruct the model to also output the context for each predicted error span, thus ensuring that error spans are identified unambiguously. 

---
# ComboBench: Can LLMs Manipulate Physical Devices to Play Virtual Reality Games? 

**Authors**: Shuqing Li, Jiayi Yan, Chenyu Niu, Jen-tse Huang, Yun Peng, Wenxuan Wang, Yepang Liu, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24706)  

**Abstract**: Virtual Reality (VR) games require players to translate high-level semantic actions into precise device manipulations using controllers and head-mounted displays (HMDs). While humans intuitively perform this translation based on common sense and embodied understanding, whether Large Language Models (LLMs) can effectively replicate this ability remains underexplored. This paper introduces a benchmark, ComboBench, evaluating LLMs' capability to translate semantic actions into VR device manipulation sequences across 262 scenarios from four popular VR games: Half-Life: Alyx, Into the Radius, Moss: Book II, and Vivecraft. We evaluate seven LLMs, including GPT-3.5, GPT-4, GPT-4o, Gemini-1.5-Pro, LLaMA-3-8B, Mixtral-8x7B, and GLM-4-Flash, compared against annotated ground truth and human performance. Our results reveal that while top-performing models like Gemini-1.5-Pro demonstrate strong task decomposition capabilities, they still struggle with procedural reasoning and spatial understanding compared to humans. Performance varies significantly across games, suggesting sensitivity to interaction complexity. Few-shot examples substantially improve performance, indicating potential for targeted enhancement of LLMs' VR manipulation capabilities. We release all materials at this https URL. 

---
# Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents 

**Authors**: Yueqi Song, Ketan Ramaneti, Zaid Sheikh, Ziru Chen, Boyu Gou, Tianbao Xie, Yiheng Xu, Danyang Zhang, Apurva Gandhi, Fan Yang, Joseph Liu, Tianyue Ou, Zhihao Yuan, Frank Xu, Shuyan Zhou, Xingyao Wang, Xiang Yue, Tao Yu, Huan Sun, Yu Su, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.24702)  

**Abstract**: Public research results on large-scale supervised finetuning of AI agents remain relatively rare, since the collection of agent training data presents unique challenges. In this work, we argue that the bottleneck is not a lack of underlying data sources, but that a large variety of data is fragmented across heterogeneous formats, tools, and interfaces. To this end, we introduce the agent data protocol (ADP), a light-weight representation language that serves as an "interlingua" between agent datasets in diverse formats and unified agent training pipelines downstream. The design of ADP is expressive enough to capture a large variety of tasks, including API/tool use, browsing, coding, software engineering, and general agentic workflows, while remaining simple to parse and train on without engineering at a per-dataset level. In experiments, we unified a broad collection of 13 existing agent training datasets into ADP format, and converted the standardized ADP data into training-ready formats for multiple agent frameworks. We performed SFT on these data, and demonstrated an average performance gain of ~20% over corresponding base models, and delivers state-of-the-art or near-SOTA performance on standard coding, browsing, tool use, and research benchmarks, without domain-specific tuning. All code and data are released publicly, in the hope that ADP could help lower the barrier to standardized, scalable, and reproducible agent training. 

---
# Tongyi DeepResearch Technical Report 

**Authors**: Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, Xixi Wu, Xuanzhong Chen, Yida Zhao, Zhen Zhang, Zhengwei Tao, Zhongwang Zhang, Zile Qiao, Chenxi Wang, Donglei Yu, Gang Fu, Haiyang Shen, Jiayin Yang, Jun Lin, Junkai Zhang, Kui Zeng, Li Yang, Hailong Yin, Maojia Song, Ming Yan, Peng Xia, Qian Xiao, Rui Min, Ruixue Ding, Runnan Fang, Shaowei Chen, Shen Huang, Shihang Wang, Shihao Cai, Weizhou Shen, Xiaobin Wang, Xin Guan, Xinyu Geng, Yingcheng Shi, Yuning Wu, Zhuo Chen, Zijian Li, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24701)  

**Abstract**: We present Tongyi DeepResearch, an agentic large language model, which is specifically designed for long-horizon, deep information-seeking research tasks. To incentivize autonomous deep research agency, Tongyi DeepResearch is developed through an end-to-end training framework that combines agentic mid-training and agentic post-training, enabling scalable reasoning and information seeking across complex tasks. We design a highly scalable data synthesis pipeline that is fully automatic, without relying on costly human annotation, and empowers all training stages. By constructing customized environments for each stage, our system enables stable and consistent interactions throughout. Tongyi DeepResearch, featuring 30.5 billion total parameters, with only 3.3 billion activated per token, achieves state-of-the-art performance across a range of agentic deep research benchmarks, including Humanity's Last Exam, BrowseComp, BrowseComp-ZH, WebWalkerQA, xbench-DeepSearch, FRAMES and xbench-DeepSearch-2510. We open-source the model, framework, and complete solutions to empower the community. 

---
# AgentFold: Long-Horizon Web Agents with Proactive Context Management 

**Authors**: Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, Pengjun Xie, Fei Huang, Siheng Chen, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24699)  

**Abstract**: LLM-based web agents show immense promise for information seeking, yet their effectiveness on long-horizon tasks is hindered by a fundamental trade-off in context management. Prevailing ReAct-based agents suffer from context saturation as they accumulate noisy, raw histories, while methods that fixedly summarize the full history at each step risk the irreversible loss of critical details. Addressing these, we introduce AgentFold, a novel agent paradigm centered on proactive context management, inspired by the human cognitive process of retrospective consolidation. AgentFold treats its context as a dynamic cognitive workspace to be actively sculpted, rather than a passive log to be filled. At each step, it learns to execute a `folding' operation, which manages its historical trajectory at multiple scales: it can perform granular condensations to preserve vital, fine-grained details, or deep consolidations to abstract away entire multi-step sub-tasks. The results on prominent benchmarks are striking: with simple supervised fine-tuning (without continual pre-training or RL), our AgentFold-30B-A3B agent achieves 36.2% on BrowseComp and 47.3% on BrowseComp-ZH. Notably, this performance not only surpasses or matches open-source models of a dramatically larger scale, such as the DeepSeek-V3.1-671B-A37B, but also surpasses leading proprietary agents like OpenAI's o4-mini. 

---
# ParallelMuse: Agentic Parallel Thinking for Deep Information Seeking 

**Authors**: Baixuan Li, Dingchu Zhang, Jialong Wu, Wenbiao Yin, Zhengwei Tao, Yida Zhao, Liwen Zhang, Haiyang Shen, Runnan Fang, Pengjun Xie, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24698)  

**Abstract**: Parallel thinking expands exploration breadth, complementing the deep exploration of information-seeking (IS) agents to further enhance problem-solving capability. However, conventional parallel thinking faces two key challenges in this setting: inefficiency from repeatedly rolling out from scratch, and difficulty in integrating long-horizon reasoning trajectories during answer generation, as limited context capacity prevents full consideration of the reasoning process. To address these issues, we propose ParallelMuse, a two-stage paradigm designed for deep IS agents. The first stage, Functionality-Specified Partial Rollout, partitions generated sequences into functional regions and performs uncertainty-guided path reuse and branching to enhance exploration efficiency. The second stage, Compressed Reasoning Aggregation, exploits reasoning redundancy to losslessly compress information relevant to answer derivation and synthesize a coherent final answer. Experiments across multiple open-source agents and benchmarks demonstrate up to 62% performance improvement with a 10--30% reduction in exploratory token consumption. 

---
# WebLeaper: Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking 

**Authors**: Zhengwei Tao, Haiyang Shen, Baixuan Li, Wenbiao Yin, Jialong Wu, Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Liwen Zhang, Xinyu Wang, Pengjun Xie, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24697)  

**Abstract**: Large Language Model (LLM)-based agents have emerged as a transformative approach for open-ended problem solving, with information seeking (IS) being a core capability that enables autonomous reasoning and decision-making. While prior research has largely focused on improving retrieval depth, we observe that current IS agents often suffer from low search efficiency, which in turn constrains overall performance. A key factor underlying this inefficiency is the sparsity of target entities in training tasks, which limits opportunities for agents to learn and generalize efficient search behaviors. To address these challenges, we propose WebLeaper, a framework for constructing high-coverage IS tasks and generating efficient solution trajectories. We formulate IS as a tree-structured reasoning problem, enabling a substantially larger set of target entities to be embedded within a constrained context. Leveraging curated Wikipedia tables, we propose three variants for synthesizing IS tasks, Basic, Union, and Reverse-Union, to systematically increase both IS efficiency and efficacy. Finally, we curate training trajectories by retaining only those that are simultaneously accurate and efficient, ensuring that the model is optimized for both correctness and search performance. Extensive experiments on both basic and comprehensive settings, conducted on five IS benchmarks, BrowserComp, GAIA, xbench-DeepSearch, WideSearch, and Seal-0, demonstrate that our method consistently achieves improvements in both effectiveness and efficiency over strong baselines. 

---
# AgentFrontier: Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis 

**Authors**: Xuanzhong Chen, Zile Qiao, Guoxin Chen, Liangcai Su, Zhen Zhang, Xinyu Wang, Pengjun Xie, Fei Huang, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24695)  

**Abstract**: Training large language model agents on tasks at the frontier of their capabilities is key to unlocking advanced reasoning. We introduce a data synthesis approach inspired by the educational theory of the Zone of Proximal Development (ZPD), which defines this frontier as tasks an LLM cannot solve alone but can master with guidance. To operationalize this, we present the AgentFrontier Engine, an automated pipeline that synthesizes high-quality, multidisciplinary data situated precisely within the LLM's ZPD. This engine supports both continued pre-training with knowledge-intensive data and targeted post-training on complex reasoning tasks. From the same framework, we derive the ZPD Exam, a dynamic and automated benchmark designed to evaluate agent capabilities on these frontier tasks. We train AgentFrontier-30B-A3B model on our synthesized data, which achieves state-of-the-art results on demanding benchmarks like Humanity's Last Exam, even surpassing some leading proprietary agents. Our work demonstrates that a ZPD-guided approach to data synthesis offers a scalable and effective path toward building more capable LLM agents. 

---
# Repurposing Synthetic Data for Fine-grained Search Agent Supervision 

**Authors**: Yida Zhao, Kuan Li, Xixi Wu, Liwen Zhang, Dingchu Zhang, Baixuan Li, Maojia Song, Zhuo Chen, Chenxi Wang, Xinyu Wang, Kewei Tu, Pengjun Xie, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24694)  

**Abstract**: LLM-based search agents are increasingly trained on entity-centric synthetic data to solve complex, knowledge-intensive tasks. However, prevailing training methods like Group Relative Policy Optimization (GRPO) discard this rich entity information, relying instead on sparse, outcome-based rewards. This critical limitation renders them unable to distinguish informative "near-miss" samples-those with substantially correct reasoning but a flawed final answer-from complete failures, thus discarding valuable learning signals. We address this by leveraging the very entities discarded during training. Our empirical analysis reveals a strong positive correlation between the number of ground-truth entities identified during an agent's reasoning process and final answer accuracy. Building on this insight, we introduce Entity-aware Group Relative Policy Optimization (E-GRPO), a novel framework that formulates a dense entity-aware reward function. E-GRPO assigns partial rewards to incorrect samples proportional to their entity match rate, enabling the model to effectively learn from these "near-misses". Experiments on diverse question-answering (QA) and deep research benchmarks show that E-GRPO consistently and significantly outperforms the GRPO baseline. Furthermore, our analysis reveals that E-GRPO not only achieves superior accuracy but also induces more efficient reasoning policies that require fewer tool calls, demonstrating a more effective and sample-efficient approach to aligning search agents. 

---
# SPICE: Self-Play In Corpus Environments Improves Reasoning 

**Authors**: Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2510.24684)  

**Abstract**: Self-improving systems require environmental interaction for continuous adaptation. We introduce SPICE (Self-Play In Corpus Environments), a reinforcement learning framework where a single model acts in two roles: a Challenger that mines documents from a large corpus to generate diverse reasoning tasks, and a Reasoner that solves them. Through adversarial dynamics, the Challenger creates an automatic curriculum at the frontier of the Reasoner's capability, while corpus grounding provides the rich, near-inexhaustible external signal necessary for sustained improvement. Unlike existing ungrounded self-play methods that offer more limited benefits, SPICE achieves consistent gains across mathematical (+8.9%) and general reasoning (+9.8%) benchmarks on multiple model families. Our analysis reveals how document grounding is a key ingredient in SPICE to continuously generate its own increasingly challenging goals and achieve them, enabling sustained self-improvement. 

---
# Dissecting Role Cognition in Medical LLMs via Neuronal Ablation 

**Authors**: Xun Liang, Huayi Lai, Hanyu Wang, Wentao Zhang, Linfeng Zhang, Yanfang Chen, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.24677)  

**Abstract**: Large language models (LLMs) have gained significant traction in medical decision support systems, particularly in the
context of medical question answering and role-playing simulations. A common practice, Prompt-Based Role Playing (PBRP),
instructs models to adopt different clinical roles (e.g., medical students, residents, attending physicians) to simulate varied
professional behaviors. However, the impact of such role prompts on model reasoning capabilities remains unclear. This
study introduces the RP-Neuron-Activated Evaluation Framework(RPNA) to evaluate whether role prompts induce distinct,
role-specific cognitive processes in LLMs or merely modify linguistic style. We test this framework on three medical QA
datasets, employing neuron ablation and representation analysis techniques to assess changes in reasoning pathways. Our
results demonstrate that role prompts do not significantly enhance the medical reasoning abilities of LLMs. Instead, they
primarily affect surface-level linguistic features, with no evidence of distinct reasoning pathways or cognitive differentiation
across clinical roles. Despite superficial stylistic changes, the core decision-making mechanisms of LLMs remain uniform
across roles, indicating that current PBRP methods fail to replicate the cognitive complexity found in real-world medical
practice. This highlights the limitations of role-playing in medical AI and emphasizes the need for models that simulate genuine
cognitive processes rather than linguistic this http URL have released the related code in the following repository:https:
//github.com/IAAR-Shanghai/RolePlay_LLMDoctor 

---
# InteractComp: Evaluating Search Agents With Ambiguous Queries 

**Authors**: Mingyi Deng, Lijun Huang, Yani Fan, Jiayi Zhang, Fashen Ren, Jinyi Bai, Fuzhen Yang, Dayi Miao, Zhaoyang Yu, Yifan Wu, Yanfei Zhang, Fengwei Teng, Yingjia Wan, Song Hu, Yude Li, Xin Jin, Conghao Hu, Haoyu Li, Qirui Fu, Tai Zhong, Xinyu Wang, Xiangru Tang, Nan Tang, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.24668)  

**Abstract**: Language agents have demonstrated remarkable potential in web search and information retrieval. However, these search agents assume user queries are complete and unambiguous, an assumption that diverges from reality where users begin with incomplete queries requiring clarification through interaction. Yet most agents lack interactive mechanisms during the search process, and existing benchmarks cannot assess this capability. To address this gap, we introduce InteractComp, a benchmark designed to evaluate whether search agents can recognize query ambiguity and actively interact to resolve it during search. Following the principle of easy to verify, interact to disambiguate, we construct 210 expert-curated questions across 9 domains through a target-distractor methodology that creates genuine ambiguity resolvable only through interaction. Evaluation of 17 models reveals striking failure: the best model achieves only 13.73% accuracy despite 71.50% with complete context, exposing systematic overconfidence rather than reasoning deficits. Forced interaction produces dramatic gains, demonstrating latent capability current strategies fail to engage. Longitudinal analysis shows interaction capabilities stagnated over 15 months while search performance improved seven-fold, revealing a critical blind spot. This stagnation, coupled with the immediate feedback inherent to search tasks, makes InteractComp a valuable resource for both evaluating and training interaction capabilities in search agents. The code is available at this https URL. 

---
# MQM Re-Annotation: A Technique for Collaborative Evaluation of Machine Translation 

**Authors**: Parker Riley, Daniel Deutsch, Mara Finkelstein, Colten DiIanni, Juraj Juraska, Markus Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2510.24664)  

**Abstract**: Human evaluation of machine translation is in an arms race with translation model quality: as our models get better, our evaluation methods need to be improved to ensure that quality gains are not lost in evaluation noise. To this end, we experiment with a two-stage version of the current state-of-the-art translation evaluation paradigm (MQM), which we call MQM re-annotation. In this setup, an MQM annotator reviews and edits a set of pre-existing MQM annotations, that may have come from themselves, another human annotator, or an automatic MQM annotation system. We demonstrate that rater behavior in re-annotation aligns with our goals, and that re-annotation results in higher-quality annotations, mostly due to finding errors that were missed during the first pass. 

---
# Evolving Diagnostic Agents in a Virtual Clinical Environment 

**Authors**: Pengcheng Qiu, Chaoyi Wu, Junwei Liu, Qiaoyu Zheng, Yusheng Liao, Haowen Wang, Yun Yue, Qianrui Fan, Shuai Zhen, Jian Wang, Jinjie Gu, Yanfeng Wang, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.24654)  

**Abstract**: In this paper, we present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static case summaries, our method acquires diagnostic strategies through interactive exploration and outcome-based feedback. Our contributions are fourfold: (i) We present DiagGym, a diagnostics world model trained with electronic health records that emits examination outcomes conditioned on patient history and recommended examination, serving as a virtual clinical environment for realistic diagnosis training and evaluation; (ii) We train DiagAgent via end-to-end, multi-turn reinforcement learning to learn diagnostic policies that optimize both information yield and diagnostic accuracy; (iii) We introduce DiagBench, a diagnostic benchmark comprising 750 cases with physician-validated examination recommendations and 99 cases annotated with 973 physician-written rubrics on diagnosis process; (iv) we demonstrate superior performance across diverse diagnostic settings. DiagAgent significantly outperforms 10 state-of-the-art LLMs, including DeepSeek-v3 and GPT-4o, as well as two prompt-engineered agents. In single-turn settings, DiagAgent achieves 9.34% higher diagnostic accuracy and 44.03% improvement in examination recommendation hit ratio. In end-to-end settings, it delivers 15.12% increase in diagnostic accuracy and 23.09% boost in examination recommendation F1 score. In rubric-based evaluation, it surpasses the next-best model, Claude-sonnet-4, by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers dynamic and clinically meaningful diagnostic management abilities unattainable through passive training alone. 

---
# Optimizing Retrieval for RAG via Reinforced Contrastive Learning 

**Authors**: Jiawei Zhou, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.24652)  

**Abstract**: As retrieval-augmented generation (RAG) becomes increasingly widespread, the role of information retrieval (IR) is shifting from retrieving information for human users to retrieving contextual knowledge for artificial intelligence (AI) systems, where relevance becomes difficult to define or annotate beforehand. To address this challenge, we propose R3, a Retrieval framework optimized for RAG through trialand-feedback Reinforced contrastive learning. Unlike prior approaches that rely on annotated or synthetic data for supervised fine-tuning, R3 enables the retriever to dynamically explore and optimize relevance within the RAG environment. During training, the retrieved results interact with the environment to produce contrastive signals that automatically guide the retriever's self-improvement. Extensive experiments across diverse tasks demonstrate that R3 improves RAG performance by 5.2% over the original retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving comparable results to LLM-augmented retrieval and RAG systems built on post-trained or instruction-tuned LLMs. It is both efficient and practical, requiring only 4 GPUs and completing training within a single day. 

---
# Quantifying the Effects of Word Length, Frequency, and Predictability on Dyslexia 

**Authors**: Hugo Rydel-Johnston, Alex Kafkas  

**Link**: [PDF](https://arxiv.org/pdf/2510.24647)  

**Abstract**: We ask where, and under what conditions, dyslexic reading costs arise in a large-scale naturalistic reading dataset. Using eye-tracking aligned to word-level features (word length, frequency, and predictability), we model how each feature influences dyslexic time costs. We find that all three features robustly change reading times in both typical and dyslexic readers, and that dyslexic readers show stronger sensitivities to each, especially predictability. Counterfactual manipulations of these features substantially narrow the dyslexic-control gap by about one third, with predictability showing the strongest effect, followed by length and frequency. These patterns align with dyslexia theories that posit heightened demands on linguistic working memory and phonological encoding, and they motivate further work on lexical complexity and parafoveal preview benefits to explain the remaining gap. In short, we quantify when extra dyslexic costs arise, how large they are, and offer actionable guidance for interventions and computational models for dyslexics. 

---
# OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning 

**Authors**: Ziyou Hu, Zhengliang Shi, Minghang Zhu, Haitao Li, Teng Sun, Pengjie Ren, Suzan Verberne, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.24636)  

**Abstract**: Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation. 

---
# "Mm, Wat?" Detecting Other-initiated Repair Requests in Dialogue 

**Authors**: Anh Ngo, Nicolas Rollet, Catherine Pelachaud, Chloe Clavel  

**Link**: [PDF](https://arxiv.org/pdf/2510.24628)  

**Abstract**: Maintaining mutual understanding is a key component in human-human conversation to avoid conversation breakdowns, in which repair, particularly Other-Initiated Repair (OIR, when one speaker signals trouble and prompts the other to resolve), plays a vital role. However, Conversational Agents (CAs) still fail to recognize user repair initiation, leading to breakdowns or disengagement. This work proposes a multimodal model to automatically detect repair initiation in Dutch dialogues by integrating linguistic and prosodic features grounded in Conversation Analysis. The results show that prosodic cues complement linguistic features and significantly improve the results of pretrained text and audio embeddings, offering insights into how different features interact. Future directions include incorporating visual cues, exploring multilingual and cross-context corpora to assess the robustness and generalizability. 

---
# Relative Scaling Laws for LLMs 

**Authors**: William Held, David Hall, Percy Liang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24626)  

**Abstract**: Scaling laws describe how language models improve with additional data, parameters, and compute. While widely used, they are typically measured on aggregate test sets. Aggregate evaluations yield clean trends but average over heterogeneous subpopulations, obscuring performance disparities. We introduce relative scaling laws, which track how performance gaps between test distributions evolve with scale rather than focusing solely on absolute error. Using 255 decoder-only Transformers trained under matched-compute (IsoFLOP) budgets from $10^{18}$--$10^{20}$ FLOPs on standard pretraining datasets, we find diverse trajectories: academic domains on MMLU converge toward parity; regional English dialects shift depending on population size; and clusters of AI risk behaviours split, with capability- and influence-related risks increasing during pretraining while adversarial risks do not. These results show that although scaling improves overall performance, it is not a universal equalizer. To support further study, we release all model checkpoints from this work to enable practitioners to measure relative alongside traditional scaling laws, in order to better prioritize robustness challenges in light of the bitter lesson. 

---
# Zero-Shot Cross-Lingual Transfer using Prefix-Based Adaptation 

**Authors**: Snegha A, Sayambhu Sen, Piyush Singh Pasi, Abhishek Singhania, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2510.24619)  

**Abstract**: With the release of new large language models (LLMs) like Llama and Mistral, zero-shot cross-lingual transfer has become increasingly feasible due to their multilingual pretraining and strong generalization capabilities. However, adapting these decoder-only LLMs to new tasks across languages remains challenging. While parameter-efficient fine-tuning (PeFT) techniques like Low-Rank Adaptation (LoRA) are widely used, prefix-based techniques such as soft prompt tuning, prefix tuning, and Llama Adapter are less explored, especially for zero-shot transfer in decoder-only models. We present a comprehensive study of three prefix-based methods for zero-shot cross-lingual transfer from English to 35+ high- and low-resource languages. Our analysis further explores transfer across linguistic families and scripts, as well as the impact of scaling model sizes from 1B to 24B. With Llama 3.1 8B, prefix methods outperform LoRA-baselines by up to 6% on the Belebele benchmark. Similar improvements were observed with Mistral v0.3 7B as well. Despite using only 1.23M learning parameters with prefix tuning, we achieve consistent improvements across diverse benchmarks. These findings highlight the potential of prefix-based techniques as an effective and scalable alternative to LoRA, particularly in low-resource multilingual settings. 

---
# Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs 

**Authors**: Siheng Xiong, Joe Zou, Faramarz Fekri, Yae Jee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2510.24606)  

**Abstract**: The quadratic cost of attention hinders the scalability of long-context LLMs, especially in resource-constrained settings. Existing static sparse methods such as sliding windows or global tokens utilizes the sparsity of attention to reduce the cost of attention, but poorly adapts to the content-dependent variations in attention due to their staticity. While previous work has proposed several dynamic approaches to improve flexibility, they still depend on predefined templates or heuristic mechanisms. Such strategies reduce generality and prune tokens that remain contextually important, limiting their accuracy across diverse tasks. To tackle these bottlenecks of existing methods for long-context modeling, we introduce Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that dynamically predicts attention sparsity online without retraining. Our proposed DHSA adaptively segments sequences into variable-length chunks, then computes chunk representations by aggregating the token embeddings within each chunk. To avoid the bias introduced by varying chunk lengths, we apply length-normalized aggregation that scales the averaged embeddings by the square root of the chunk size. Finally, DHSA upsamples the chunk-level similarity scores to token level similarities to calculate importance scores that determine which token-level interactions should be preserved. Our experiments on Gemma2 with Needle-in-a-Haystack Test and LongBench show that DHSA matches dense attention in accuracy, while reducing prefill latency by 20-60% and peak memory usage by 35%. Compared to other representative baselines such as block sparse attention, DHSA achieves consistently higher accuracy (6-18% relative gains) with comparable or lower cost, offering an efficient and adaptable solution for long-context on-device LLMs. 

---
# Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way 

**Authors**: Yicun Yang, Cong Wang, Shaobo Wang, Zichen Wen, Biqing Qi, Hanlin Xu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24605)  

**Abstract**: Diffusion-based large language models (dLLMs) have exhibited substantial potential for parallel text generation, which may enable more efficient generation compared to autoregressive models. However, current dLLMs suffer from fixed generation lengths, which indicates the generation lengths of dLLMs have to be determined before decoding as a hyper-parameter, leading to issues in efficiency and flexibility. To solve these problems, in this work, we propose to train a diffusion LLM with native variable generation lengths, abbreviated as dLLM-Var. Concretely, we aim to train a model to accurately predict the [EOS] token in the generated text, which makes a dLLM be able to natively infer in a block diffusion manner, while still maintaining the ability of global bi-directional (full) attention and high parallelism. Experiments on standard benchmarks demonstrate that our method achieves a 30.1x speedup over traditional dLLM inference paradigms and a 2.4x speedup relative to autoregressive models such as Qwen and Llama. Our method achieves higher accuracy and faster inference, elevating dLLMs beyond mere academic novelty and supporting their practical use in real-world applications. Codes and models have been released. 

---
# ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization 

**Authors**: Guoxin Chen, Jing Wu, Xinjie Chen, Wayne Xin Zhao, Ruihua Song, Chengxi Li, Kai Fan, Dayiheng Liu, Minpeng Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.24592)  

**Abstract**: Autoformalization, which translates natural language mathematics into machine-verifiable formal statements, is critical for using formal mathematical reasoning to solve math problems stated in natural language. While Large Language Models can generate syntactically correct formal statements, they often fail to preserve the original problem's semantic intent. This limitation arises from the LLM approaches' treating autoformalization as a simplistic translation task which lacks mechanisms for self-reflection and iterative refinement that human experts naturally employ. To address these issues, we propose ReForm, a Reflective Autoformalization method that tightly integrates semantic consistency evaluation into the autoformalization process. This enables the model to iteratively generate formal statements, assess its semantic fidelity, and self-correct identified errors through progressive refinement. To effectively train this reflective model, we introduce Prospective Bounded Sequence Optimization (PBSO), which employs different rewards at different sequence positions to ensure that the model develops both accurate autoformalization and correct semantic validations, preventing superficial critiques that would undermine the purpose of reflection. Extensive experiments across four autoformalization benchmarks demonstrate that ReForm achieves an average improvement of 17.2 percentage points over the strongest baselines. To further ensure evaluation reliability, we introduce ConsistencyCheck, a benchmark of 859 expert-annotated items that not only validates LLMs as judges but also reveals that autoformalization is inherently difficult: even human experts produce semantic errors in up to 38.5% of cases. 

---
# ReplicationBench: Can AI Agents Replicate Astrophysics Research Papers? 

**Authors**: Christine Ye, Sihan Yuan, Suchetha Cooray, Steven Dillmann, Ian L. V. Roque, Dalya Baron, Philipp Frank, Sergio Martin-Alvarez, Nolan Koblischke, Frank J Qu, Diyi Yang, Risa Wechsler, Ioana Ciuca  

**Link**: [PDF](https://arxiv.org/pdf/2510.24591)  

**Abstract**: Frontier AI agents show increasing promise as scientific research assistants, and may eventually be useful for extended, open-ended research workflows. However, in order to use agents for novel research, we must first assess the underlying faithfulness and correctness of their work. To evaluate agents as research assistants, we introduce ReplicationBench, an evaluation framework that tests whether agents can replicate entire research papers drawn from the astrophysics literature. Astrophysics, where research relies heavily on archival data and computational study while requiring little real-world experimentation, is a particularly useful testbed for AI agents in scientific research. We split each paper into tasks which require agents to replicate the paper's core contributions, including the experimental setup, derivations, data analysis, and codebase. Each task is co-developed with the original paper authors and targets a key scientific result, enabling objective evaluation of both faithfulness (adherence to original methods) and correctness (technical accuracy of results). ReplicationBench is extremely challenging for current frontier language models: even the best-performing language models score under 20%. We analyze ReplicationBench trajectories in collaboration with domain experts and find a rich, diverse set of failure modes for agents in scientific research. ReplicationBench establishes the first benchmark of paper-scale, expert-validated astrophysics research tasks, reveals insights about agent performance generalizable to other domains of data-driven science, and provides a scalable framework for measuring AI agents' reliability in scientific research. 

---
# BEST-RQ-Based Self-Supervised Learning for Whisper Domain Adaptation 

**Authors**: Raphaël Bagat, Irina Illina, Emmanuel Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2510.24570)  

**Abstract**: Automatic Speech Recognition (ASR) systems, despite large multilingual training, struggle in out-of-domain and low-resource scenarios where labeled data is scarce. We propose BEARD (BEST-RQ Encoder Adaptation with Re-training and Distillation), a novel framework designed to adapt Whisper's encoder using unlabeled data. Unlike traditional self-supervised learning methods, BEARD uniquely combines a BEST-RQ objective with knowledge distillation from a frozen teacher encoder, ensuring the encoder's complementarity with the pre-trained decoder. Our experiments focus on the ATCO2 corpus from the challenging Air Traffic Control (ATC) communications domain, characterized by non-native speech, noise, and specialized phraseology. Using about 5,000 hours of untranscribed speech for BEARD and 2 hours of transcribed speech for fine-tuning, the proposed approach significantly outperforms previous baseline and fine-tuned model, achieving a relative improvement of 12% compared to the fine-tuned model. To the best of our knowledge, this is the first work to use a self-supervised learning objective for domain adaptation of Whisper. 

---
# Open Korean Historical Corpus: A Millennia-Scale Diachronic Collection of Public Domain Texts 

**Authors**: Seyoung Song, Nawon Kim, Songeun Chae, Kiwoong Park, Jiho Jin, Haneul Yoo, Kyunghyun Cho, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.24541)  

**Abstract**: The history of the Korean language is characterized by a discrepancy between its spoken and written forms and a pivotal shift from Chinese characters to the Hangul alphabet. However, this linguistic evolution has remained largely unexplored in NLP due to a lack of accessible historical corpora. To address this gap, we introduce the Open Korean Historical Corpus, a large-scale, openly licensed dataset spanning 1,300 years and 6 languages, as well as under-represented writing systems like Korean-style Sinitic (Idu) and Hanja-Hangul mixed script. This corpus contains 18 million documents and 5 billion tokens from 19 sources, ranging from the 7th century to 2025. We leverage this resource to quantitatively analyze major linguistic shifts: (1) Idu usage peaked in the 1860s before declining sharply; (2) the transition from Hanja to Hangul was a rapid transformation starting around 1890; and (3) North Korea's lexical divergence causes modern tokenizers to produce up to 51 times higher out-of-vocabulary rates. This work provides a foundational resource for quantitative diachronic analysis by capturing the history of the Korean language. Moreover, it can serve as a pre-training corpus for large language models, potentially improving their understanding of Sino-Korean vocabulary in modern Hangul as well as archaic writing systems. 

---
# Dark & Stormy: Modeling Humor in the Worst Sentences Ever Written 

**Authors**: Venkata S Govindarajan, Laura Biester  

**Link**: [PDF](https://arxiv.org/pdf/2510.24538)  

**Abstract**: Textual humor is enormously diverse and computational studies need to account for this range, including intentionally bad humor. In this paper, we curate and analyze a novel corpus of sentences from the Bulwer-Lytton Fiction Contest to better understand "bad" humor in English. Standard humor detection models perform poorly on our corpus, and an analysis of literary devices finds that these sentences combine features common in existing humor datasets (e.g., puns, irony) with metaphor, metafiction and simile. LLMs prompted to synthesize contest-style sentences imitate the form but exaggerate the effect by over-using certain literary devices, and including far more novel adjective-noun bigrams than human writers. Data, code and analysis are available at this https URL 

---
# Levée d'ambiguïtés par grammaires locales 

**Authors**: Eric G. C. Laporte  

**Link**: [PDF](https://arxiv.org/pdf/2510.24530)  

**Abstract**: Many words are ambiguous in terms of their part of speech (POS). However, when a word appears in a text, this ambiguity is generally much reduced. Disambiguating POS involves using context to reduce the number of POS associated with words, and is one of the main challenges of lexical tagging. The problem of labeling words by POS frequently arises in natural language processing, for example for spelling correction, grammar or style checking, expression recognition, text-to-speech conversion, text corpus analysis, etc. Lexical tagging systems are thus useful as an initial component of many natural language processing systems. A number of recent lexical tagging systems produce multiple solutions when the text is lexically ambiguous or the uniquely correct solution cannot be found. These contributions aim to guarantee a zero silence rate: the correct tag(s) for a word must never be discarded. This objective is unrealistic for systems that tag each word uniquely. This article concerns a lexical disambiguation method adapted to the objective of a zero silence rate and implemented in Silberztein's INTEX system (1993). We present here a formal description of this method. We show that to verify a local disambiguation grammar in this framework, it is not sufficient to consider the transducer paths separately: one needs to verify their interactions. Similarly, if a combination of multiple transducers is used, the result cannot be predicted by considering them in isolation. Furthermore, when examining the initial labeling of a text as produced by INTEX, ideas for disambiguation rules come spontaneously, but grammatical intuitions may turn out to be inaccurate, often due to an unforeseen construction or ambiguity. If a zero silence rate is targeted, local grammars must be carefully tested. This is where a detailed specification of what a grammar will do once applied to texts would be necessary. 

---
# CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration? 

**Authors**: Qing Zong, Jiayu Liu, Tianshi Zheng, Chunyang Li, Baixuan Xu, Haochen Shi, Weiqi Wang, Zhaowei Wang, Chunkit Chan, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.24505)  

**Abstract**: Accurate confidence calibration in Large Language Models (LLMs) is critical for safe use in high-stakes domains, where clear verbalized confidence enhances user trust. Traditional methods that mimic reference confidence expressions often fail to capture the reasoning needed for accurate confidence assessment. We propose natural language critiques as a solution, ideally suited for confidence calibration, as precise gold confidence labels are hard to obtain and often require multiple generations. This paper studies how natural language critiques can enhance verbalized confidence, addressing: (1) What to critique: uncertainty (question-focused) or confidence (answer-specific)? Analysis shows confidence suits multiple-choice tasks, while uncertainty excels in open-ended scenarios. (2) How to critique: self-critique or critique calibration training? We propose Self-Critique, enabling LLMs to critique and optimize their confidence beyond mere accuracy, and CritiCal, a novel Critique Calibration training method that leverages natural language critiques to improve confidence calibration, moving beyond direct numerical optimization. Experiments show that CritiCal significantly outperforms Self-Critique and other competitive baselines, even surpassing its teacher model, GPT-4o, in complex reasoning tasks. CritiCal also shows robust generalization in out-of-distribution settings, advancing LLM's reliability. 

---
# A word association network methodology for evaluating implicit biases in LLMs compared to humans 

**Authors**: Katherine Abramski, Giulio Rossetti, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2510.24488)  

**Abstract**: As Large language models (LLMs) become increasingly integrated into our lives, their inherent social biases remain a pressing concern. Detecting and evaluating these biases can be challenging because they are often implicit rather than explicit in nature, so developing evaluation methods that assess the implicit knowledge representations of LLMs is essential. We present a novel word association network methodology for evaluating implicit biases in LLMs based on simulating semantic priming within LLM-generated word association networks. Our prompt-based approach taps into the implicit relational structures encoded in LLMs, providing both quantitative and qualitative assessments of bias. Unlike most prompt-based evaluation methods, our method enables direct comparisons between various LLMs and humans, providing a valuable point of reference and offering new insights into the alignment of LLMs with human cognition. To demonstrate the utility of our methodology, we apply it to both humans and several widely used LLMs to investigate social biases related to gender, religion, ethnicity, sexual orientation, and political party. Our results reveal both convergences and divergences between LLM and human biases, providing new perspectives on the potential risks of using LLMs. Our methodology contributes to a systematic, scalable, and generalizable framework for evaluating and comparing biases across multiple LLMs and humans, advancing the goal of transparent and socially responsible language technologies. 

---
# Talk2Ref: A Dataset for Reference Prediction from Scientific Talks 

**Authors**: Frederik Broy, Maike Züfle, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.24478)  

**Abstract**: Scientific talks are a growing medium for disseminating research, and automatically identifying relevant literature that grounds or enriches a talk would be highly valuable for researchers and students alike. We introduce Reference Prediction from Talks (RPT), a new task that maps long, and unstructured scientific presentations to relevant papers. To support research on RPT, we present Talk2Ref, the first large-scale dataset of its kind, containing 6,279 talks and 43,429 cited papers (26 per talk on average), where relevance is approximated by the papers cited in the talk's corresponding source publication. We establish strong baselines by evaluating state-of-the-art text embedding models in zero-shot retrieval scenarios, and propose a dual-encoder architecture trained on Talk2Ref. We further explore strategies for handling long transcripts, as well as training for domain adaptation. Our results show that fine-tuning on Talk2Ref significantly improves citation prediction performance, demonstrating both the challenges of the task and the effectiveness of our dataset for learning semantic representations from spoken scientific content. The dataset and trained models are released under an open license to foster future research on integrating spoken scientific communication into citation recommendation systems. 

---
# Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems 

**Authors**: Yihan Li, Xiyuan Fu, Ghanshyam Verma, Paul Buitelaar, Mingming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24476)  

**Abstract**: Hallucination remains one of the key obstacles to the reliable deployment of large language models (LLMs), particularly in real-world applications. Among various mitigation strategies, Retrieval-Augmented Generation (RAG) and reasoning enhancement have emerged as two of the most effective and widely adopted approaches, marking a shift from merely suppressing hallucinations to balancing creativity and reliability. However, their synergistic potential and underlying mechanisms for hallucination mitigation have not yet been systematically examined. This survey adopts an application-oriented perspective of capability enhancement to analyze how RAG, reasoning enhancement, and their integration in Agentic Systems mitigate hallucinations. We propose a taxonomy distinguishing knowledge-based and logic-based hallucinations, systematically examine how RAG and reasoning address each, and present a unified framework supported by real-world applications, evaluations, and benchmarks. 

---
# Iterative Critique-Refine Framework for Enhancing LLM Personalization 

**Authors**: Durga Prasad Maram, Dhruvin Gandhi, Zonghai Yao, Gayathri Akkinapalli, Franck Dernoncourt, Yu Wang, Ryan A. Rossi, Nesreen K. Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2510.24469)  

**Abstract**: Personalized text generation requires models not only to produce coherent text but also to align with a target user's style, tone, and topical focus. Existing retrieval-augmented approaches such as LaMP and PGraphRAG enrich profiles with user and neighbor histories, but they stop at generation and often yield outputs that drift in tone, topic, or style. We present PerFine, a unified, training-free critique-refine framework that enhances personalization through iterative, profile-grounded feedback. In each iteration, an LLM generator produces a draft conditioned on the retrieved profile, and a critic LLM - also conditioned on the same profile - provides structured feedback on tone, vocabulary, sentence structure, and topicality. The generator then revises, while a novel knockout strategy retains the stronger draft across iterations. We further study additional inference-time strategies such as Best-of-N and Topic Extraction to balance quality and efficiency. Across Yelp, Goodreads, and Amazon datasets, PerFine consistently improves personalization over PGraphRAG, with GEval gains of +7-13%, steady improvements over 3-5 refinement iterations, and scalability with increasing critic size. These results highlight that post-hoc, profile-aware feedback offers a powerful paradigm for personalized LLM generation that is both training-free and model-agnostic. 

---
# Charting the European LLM Benchmarking Landscape: A New Taxonomy and a Set of Best Practices 

**Authors**: Špela Vintar, Taja Kuzman Pungeršek, Mojca Brglez, Nikola Ljubešić  

**Link**: [PDF](https://arxiv.org/pdf/2510.24450)  

**Abstract**: While new benchmarks for large language models (LLMs) are being developed continuously to catch up with the growing capabilities of new models and AI in general, using and evaluating LLMs in non-English languages remains a little-charted landscape. We give a concise overview of recent developments in LLM benchmarking, and then propose a new taxonomy for the categorization of benchmarks that is tailored to multilingual or non-English use scenarios. We further propose a set of best practices and quality standards that could lead to a more coordinated development of benchmarks for European languages. Among other recommendations, we advocate for a higher language and culture sensitivity of evaluation methods. 

---
# SPARTA: Evaluating Reasoning Segmentation Robustness through Black-Box Adversarial Paraphrasing in Text Autoencoder Latent Space 

**Authors**: Viktoriia Zinkovich, Anton Antonov, Andrei Spiridonov, Denis Shepelev, Andrey Moskalenko, Daria Pugacheva, Elena Tutubalina, Andrey Kuznetsov, Vlad Shakhuro  

**Link**: [PDF](https://arxiv.org/pdf/2510.24446)  

**Abstract**: Multimodal large language models (MLLMs) have shown impressive capabilities in vision-language tasks such as reasoning segmentation, where models generate segmentation masks based on textual queries. While prior work has primarily focused on perturbing image inputs, semantically equivalent textual paraphrases-crucial in real-world applications where users express the same intent in varied ways-remain underexplored. To address this gap, we introduce a novel adversarial paraphrasing task: generating grammatically correct paraphrases that preserve the original query meaning while degrading segmentation performance. To evaluate the quality of adversarial paraphrases, we develop a comprehensive automatic evaluation protocol validated with human studies. Furthermore, we introduce SPARTA-a black-box, sentence-level optimization method that operates in the low-dimensional semantic latent space of a text autoencoder, guided by reinforcement learning. SPARTA achieves significantly higher success rates, outperforming prior methods by up to 2x on both the ReasonSeg and LLMSeg-40k datasets. We use SPARTA and competitive baselines to assess the robustness of advanced reasoning segmentation models. We reveal that they remain vulnerable to adversarial paraphrasing-even under strict semantic and grammatical constraints. All code and data will be released publicly upon acceptance. 

---
# Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content 

**Authors**: Abdullah Mushtaq, Rafay Naeem, Ezieddin Elmahjub, Ibrahim Ghaznavi, Shawqi Al-Maliki, Mohamed Abdallah, Ala Al-Fuqaha, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2510.24438)  

**Abstract**: Large language models are increasingly used for Islamic guidance, but risk misquoting texts, misapplying jurisprudence, or producing culturally inconsistent responses. We pilot an evaluation of GPT-4o, Ansari AI, and Fanar on prompts from authentic Islamic blogs. Our dual-agent framework uses a quantitative agent for citation verification and six-dimensional scoring (e.g., Structure, Islamic Consistency, Citations) and a qualitative agent for five-dimensional side-by-side comparison (e.g., Tone, Depth, Originality). GPT-4o scored highest in Islamic Accuracy (3.93) and Citation (3.38), Ansari AI followed (3.68, 3.32), and Fanar lagged (2.76, 1.82). Despite relatively strong performance, models still fall short in reliably producing accurate Islamic content and citations -- a paramount requirement in faith-sensitive writing. GPT-4o had the highest mean quantitative score (3.90/5), while Ansari AI led qualitative pairwise wins (116/200). Fanar, though trailing, introduces innovations for Islamic and Arabic contexts. This study underscores the need for community-driven benchmarks centering Muslim perspectives, offering an early step toward more reliable AI in Islamic knowledge and other high-stakes domains such as medicine, law, and journalism. 

---
# LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data 

**Authors**: Julian Valline, Cedric Lothritz, Jordi Cabot  

**Link**: [PDF](https://arxiv.org/pdf/2510.24434)  

**Abstract**: The effectiveness of instruction-tuned Large Language Models (LLMs) is often limited in low-resource linguistic settings due to a lack of high-quality training data. We introduce LuxIT, a novel, monolingual instruction tuning dataset for Luxembourgish developed to mitigate this challenge. We synthesize the dataset from a corpus of native Luxembourgish texts, utilizing DeepSeek-R1-0528, chosen for its shown proficiency in Luxembourgish. Following generation, we apply a quality assurance process, employing an LLM-as-a-judge approach. To investigate the practical utility of the dataset, we fine-tune several smaller-scale LLMs on LuxIT. Subsequent benchmarking against their base models on Luxembourgish language proficiency examinations, however, yields mixed results, with performance varying significantly across different models. LuxIT represents a critical contribution to Luxembourgish natural language processing and offers a replicable monolingual methodology, though our findings highlight the need for further research to optimize its application. 

---
# SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models 

**Authors**: Ken Gu, Advait Bhat, Mike A Merrill, Robert West, Xin Liu, Daniel McDuff, Tim Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2510.24427)  

**Abstract**: Evaluating the reasoning ability of language models (LMs) is complicated by their extensive parametric world knowledge, where benchmark performance often reflects factual recall rather than genuine reasoning. Existing datasets and approaches (e.g., temporal filtering, paraphrasing, adversarial substitution) cannot cleanly separate the two. We present SynthWorlds, a framework that disentangles task reasoning complexity from factual knowledge. In SynthWorlds, we construct parallel corpora representing two worlds with identical interconnected structure: a real-mapped world, where models may exploit parametric knowledge, and a synthetic-mapped world, where such knowledge is meaningless. On top of these corpora, we design two mirrored tasks as case studies: multi-hop question answering and page navigation, which maintain equal reasoning difficulty across worlds. Experiments in parametric-only (e.g., closed-book QA) and knowledge-augmented (e.g., retrieval-augmented) LM settings reveal a persistent knowledge advantage gap, defined as the performance boost models gain from memorized parametric world knowledge. Knowledge acquisition and integration mechanisms reduce but do not eliminate this gap, highlighting opportunities for system improvements. Fully automatic and scalable, SynthWorlds provides a controlled environment for evaluating LMs in ways that were previously challenging, enabling precise and testable comparisons of reasoning and memorization. 

---
# Comprehensive and Efficient Distillation for Lightweight Sentiment Analysis Models 

**Authors**: Guangyu Xie, Yice Zhang, Jianzhu Bao, Qianlong Wang, Yang Sun, Bingbing Wang, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24425)  

**Abstract**: Recent efforts leverage knowledge distillation techniques to develop lightweight and practical sentiment analysis models. These methods are grounded in human-written instructions and large-scale user texts. Despite the promising results, two key challenges remain: (1) manually written instructions are limited in diversity and quantity, making them insufficient to ensure comprehensive coverage of distilled knowledge; (2) large-scale user texts incur high computational cost, hindering the practicality of these methods. To this end, we introduce COMPEFFDIST, a comprehensive and efficient distillation framework for sentiment analysis. Our framework consists of two key modules: attribute-based automatic instruction construction and difficulty-based data filtering, which correspondingly tackle the aforementioned challenges. Applying our method across multiple model series (Llama-3, Qwen-3, and Gemma-3), we enable 3B student models to match the performance of 20x larger teacher models on most tasks. In addition, our approach greatly outperforms baseline methods in data efficiency, attaining the same performance level with only 10% of the data. 

---
# Text Simplification with Sentence Embeddings 

**Authors**: Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2510.24365)  

**Abstract**: Sentence embeddings can be decoded to give approximations of the original texts used to create them. We explore this effect in the context of text simplification, demonstrating that reconstructed text embeddings preserve complexity levels. We experiment with a small feed forward neural network to effectively learn a transformation between sentence embeddings representing high-complexity and low-complexity texts. We provide comparison to a Seq2Seq and LLM-based approach, showing encouraging results in our much smaller learning setting. Finally, we demonstrate the applicability of our transformation to an unseen simplification dataset (MedEASI), as well as datasets from languages outside the training data (ES,DE). We conclude that learning transformations in sentence embedding space is a promising direction for future research and has potential to unlock the ability to develop small, but powerful models for text simplification and other natural language generation tasks. 

---
# LongWeave: A Long-Form Generation Benchmark Bridging Real-World Relevance and Verifiability 

**Authors**: Zikai Xiao, Fei Huang, Jianhong Tu, Jianhui Wei, Wen Ma, Yuxuan Zhou, Jian Wu, Bowen Yu, Zuozhu Liu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.24345)  

**Abstract**: Generating long, informative, and factual outputs remains a major challenge for Large Language Models (LLMs). Existing benchmarks for long-form generation typically assess real-world queries with hard-to-verify metrics or use synthetic setups that ease evaluation but overlook real-world intricacies. In this paper, we introduce \textbf{LongWeave}, which balances real-world and verifiable assessment with Constraint-Verifier Evaluation (CoV-Eval). CoV-Eval constructs tasks by first defining verifiable targets within real-world scenarios, then systematically generating corresponding queries, textual materials, and constraints based on these targets. This ensures that tasks are both realistic and objectively assessable, enabling rigorous assessment of model capabilities in meeting complex real-world constraints. LongWeave supports customizable input/output lengths (up to 64K/8K tokens) across seven distinct tasks. Evaluation on 23 LLMs shows that even state-of-the-art models encounter significant challenges in long-form generation as real-world complexity and output length increase. 

---
# Beyond MCQ: An Open-Ended Arabic Cultural QA Benchmark with Dialect Variants 

**Authors**: Hunzalah Hassan Bhatti, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2510.24328)  

**Abstract**: Large Language Models (LLMs) are increasingly used to answer everyday questions, yet their performance on culturally grounded and dialectal content remains uneven across languages. We propose a comprehensive method that (i) translates Modern Standard Arabic (MSA) multiple-choice questions (MCQs) into English and several Arabic dialects, (ii) converts them into open-ended questions (OEQs), (iii) benchmarks a range of zero-shot and fine-tuned LLMs under both MCQ and OEQ settings, and (iv) generates chain-of-thought (CoT) rationales to fine-tune models for step-by-step reasoning. Using this method, we extend an existing dataset in which QAs are parallelly aligned across multiple language varieties, making it, to our knowledge, the first of its kind. We conduct extensive experiments with both open and closed models. Our findings show that (i) models underperform on Arabic dialects, revealing persistent gaps in culturally grounded and dialect-specific knowledge; (ii) Arabic-centric models perform well on MCQs but struggle with OEQs; and (iii) CoT improves judged correctness while yielding mixed n-gram-based metrics. The developed dataset will be publicly released to support further research on culturally and linguistically inclusive evaluation. 

---
# Critique-RL: Training Language Models for Critiquing through Two-Stage Reinforcement Learning 

**Authors**: Zhiheng Xi, Jixuan Huang, Xin Guo, Boyang Hong, Dingwen Yang, Xiaoran Fan, Shuo Li, Zehui Chen, Junjie Ye, Siyu Yuan, Zhengyin Du, Xuesong Yao, Yufei Xu, Jiecao Chen, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24320)  

**Abstract**: Training critiquing language models to assess and provide feedback on model outputs is a promising way to improve LLMs for complex reasoning tasks. However, existing approaches typically rely on stronger supervisors for annotating critique data. To address this, we propose Critique-RL, an online RL approach for developing critiquing language models without stronger supervision. Our approach operates on a two-player paradigm: the actor generates a response, the critic provides feedback, and the actor refines the response accordingly. We first reveal that relying solely on indirect reward signals from the actor's outputs for RL optimization often leads to unsatisfactory critics: while their helpfulness (i.e., providing constructive feedback) improves, the discriminability (i.e., determining whether a response is high-quality or not) remains poor, resulting in marginal performance gains. To overcome this, Critique-RL adopts a two-stage optimization strategy. In stage I, it reinforces the discriminability of the critic with direct rule-based reward signals; in stage II, it introduces indirect rewards based on actor refinement to improve the critic's helpfulness, while maintaining its discriminability via appropriate regularization. Extensive experiments across various tasks and models show that Critique-RL delivers substantial performance improvements. For example, it achieves a 9.02% gain on in-domain tasks and a 5.70% gain on out-of-domain tasks for Qwen2.5-7B, highlighting its potential. 

---
# Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards 

**Authors**: Shangyu Xing, Siyuan Wang, Chenyuan Yang, Xinyu Dai, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.24302)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are publicly available at this https URL. 

---
# MERGE: Minimal Expression-Replacement GEneralization Test for Natural Language Inference 

**Authors**: Mădălina Zgreabăn, Tejaswini Deoskar, Lasha Abzianidze  

**Link**: [PDF](https://arxiv.org/pdf/2510.24295)  

**Abstract**: In recent years, many generalization benchmarks have shown language models' lack of robustness in natural language inference (NLI). However, manually creating new benchmarks is costly, while automatically generating high-quality ones, even by modifying existing benchmarks, is extremely difficult. In this paper, we propose a methodology for automatically generating high-quality variants of original NLI problems by replacing open-class words, while crucially preserving their underlying reasoning. We dub our generalization test as MERGE (Minimal Expression-Replacements GEneralization), which evaluates the correctness of models' predictions across reasoning-preserving variants of the original problem. Our results show that NLI models' perform 4-20% worse on variants, suggesting low generalizability even on such minimally altered problems. We also analyse how word class of the replacements, word probability, and plausibility influence NLI models' performance. 

---
# Can LLMs Translate Human Instructions into a Reinforcement Learning Agent's Internal Emergent Symbolic Representation? 

**Authors**: Ziqi Ma, Sao Mai Nguyen, Philippe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24259)  

**Abstract**: Emergent symbolic representations are critical for enabling developmental learning agents to plan and generalize across tasks. In this work, we investigate whether large language models (LLMs) can translate human natural language instructions into the internal symbolic representations that emerge during hierarchical reinforcement learning. We apply a structured evaluation framework to measure the translation performance of commonly seen LLMs -- GPT, Claude, Deepseek and Grok -- across different internal symbolic partitions generated by a hierarchical reinforcement learning algorithm in the Ant Maze and Ant Fall environments. Our findings reveal that although LLMs demonstrate some ability to translate natural language into a symbolic representation of the environment dynamics, their performance is highly sensitive to partition granularity and task complexity. The results expose limitations in current LLMs capacity for representation alignment, highlighting the need for further research on robust alignment between language and internal agent representations. 

---
# From Memorization to Reasoning in the Spectrum of Loss Curvature 

**Authors**: Jack Merullo, Srihita Vatsavaya, Lucius Bushnaq, Owen Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.24256)  

**Abstract**: We characterize how memorization is represented in transformer models and show that it can be disentangled in the weights of both language models (LMs) and vision transformers (ViTs) using a decomposition based on the loss landscape curvature. This insight is based on prior theoretical and empirical work showing that the curvature for memorized training points is much sharper than non memorized, meaning ordering weight components from high to low curvature can reveal a distinction without explicit labels. This motivates a weight editing procedure that suppresses far more recitation of untargeted memorized data more effectively than a recent unlearning method (BalancedSubnet), while maintaining lower perplexity. Since the basis of curvature has a natural interpretation for shared structure in model weights, we analyze the editing procedure extensively on its effect on downstream tasks in LMs, and find that fact retrieval and arithmetic are specifically and consistently negatively affected, even though open book fact retrieval and general logical reasoning is conserved. We posit these tasks rely heavily on specialized directions in weight space rather than general purpose mechanisms, regardless of whether those individual datapoints are memorized. We support this by showing a correspondence between task data's activation strength with low curvature components that we edit out, and the drop in task performance after the edit. Our work enhances the understanding of memorization in neural networks with practical applications towards removing it, and provides evidence for idiosyncratic, narrowly-used structures involved in solving tasks like math and fact retrieval. 

---
# Evaluating LLMs on Generating Age-Appropriate Child-Like Conversations 

**Authors**: Syed Zohaib Hassan, Pål Halvorsen, Miriam S. Johnson, Pierre Lison  

**Link**: [PDF](https://arxiv.org/pdf/2510.24250)  

**Abstract**: Large Language Models (LLMs), predominantly trained on adult conversational data, face significant challenges when generating authentic, child-like dialogue for specialized applications. We present a comparative study evaluating five different LLMs (GPT-4, RUTER-LLAMA-2-13b, GPTSW, NorMistral-7b, and NorBloom-7b) to generate age-appropriate Norwegian conversations for children aged 5 and 9 years. Through a blind evaluation by eleven education professionals using both real child interview data and LLM-generated text samples, we assessed authenticity and developmental appropriateness. Our results show that evaluators achieved strong inter-rater reliability (ICC=0.75) and demonstrated higher accuracy in age prediction for younger children (5-year-olds) compared to older children (9-year-olds). While GPT-4 and NorBloom-7b performed relatively well, most models generated language perceived as more linguistically advanced than the target age groups. These findings highlight critical data-related challenges in developing LLM systems for specialized applications involving children, particularly in low-resource languages where comprehensive age-appropriate lexical resources are scarce. 

---
# Abjad AI at NADI 2025: CATT-Whisper: Multimodal Diacritic Restoration Using Text and Speech Representations 

**Authors**: Ahmad Ghannam, Naif Alharthi, Faris Alasmary, Kholood Al Tabash, Shouq Sadah, Lahouari Ghouti  

**Link**: [PDF](https://arxiv.org/pdf/2510.24247)  

**Abstract**: In this work, we tackle the Diacritic Restoration (DR) task for Arabic dialectal sentences using a multimodal approach that combines both textual and speech information. We propose a model that represents the text modality using an encoder extracted from our own pre-trained model named CATT. The speech component is handled by the encoder module of the OpenAI Whisper base model. Our solution is designed following two integration strategies. The former consists of fusing the speech tokens with the input at an early stage, where the 1500 frames of the audio segment are averaged over 10 consecutive frames, resulting in 150 speech tokens. To ensure embedding compatibility, these averaged tokens are processed through a linear projection layer prior to merging them with the text tokens. Contextual encoding is guaranteed by the CATT encoder module. The latter strategy relies on cross-attention, where text and speech embeddings are fused. The cross-attention output is then fed to the CATT classification head for token-level diacritic prediction. To further improve model robustness, we randomly deactivate the speech input during training, allowing the model to perform well with or without speech. Our experiments show that the proposed approach achieves a word error rate (WER) of 0.25 and a character error rate (CER) of 0.9 on the development set. On the test set, our model achieved WER and CER scores of 0.55 and 0.13, respectively. 

---
# Towards Transparent Reasoning: What Drives Faithfulness in Large Language Models? 

**Authors**: Teague McMillan, Gabriele Dominici, Martin Gjoreski, Marc Langheinrich  

**Link**: [PDF](https://arxiv.org/pdf/2510.24236)  

**Abstract**: Large Language Models (LLMs) often produce explanations that do not faithfully reflect the factors driving their predictions. In healthcare settings, such unfaithfulness is especially problematic: explanations that omit salient clinical cues or mask spurious shortcuts can undermine clinician trust and lead to unsafe decision support. We study how inference and training-time choices shape explanation faithfulness, focusing on factors practitioners can control at deployment. We evaluate three LLMs (GPT-4.1-mini, LLaMA 70B, LLaMA 8B) on two datasets-BBQ (social bias) and MedQA (medical licensing questions), and manipulate the number and type of few-shot examples, prompting strategies, and training procedure. Our results show: (i) both the quantity and quality of few-shot examples significantly impact model faithfulness; (ii) faithfulness is sensitive to prompting design; (iii) the instruction-tuning phase improves measured faithfulness on MedQA. These findings offer insights into strategies for enhancing the interpretability and trustworthiness of LLMs in sensitive domains. 

---
# HACK: Hallucinations Along Certainty and Knowledge Axes 

**Authors**: Adi Simhi, Jonathan Herzig, Itay Itzhak, Dana Arad, Zorik Gekhman, Roi Reichart, Fazl Barez, Gabriel Stanovsky, Idan Szpektor, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.24222)  

**Abstract**: Hallucinations in LLMs present a critical barrier to their reliable usage. Existing research usually categorizes hallucination by their external properties rather than by the LLMs' underlying internal properties. This external focus overlooks that hallucinations may require tailored mitigation strategies based on their underlying mechanism. We propose a framework for categorizing hallucinations along two axes: knowledge and certainty. Since parametric knowledge and certainty may vary across models, our categorization method involves a model-specific dataset construction process that differentiates between those types of hallucinations. Along the knowledge axis, we distinguish between hallucinations caused by a lack of knowledge and those occurring despite the model having the knowledge of the correct response. To validate our framework along the knowledge axis, we apply steering mitigation, which relies on the existence of parametric knowledge to manipulate model activations. This addresses the lack of existing methods to validate knowledge categorization by showing a significant difference between the two hallucination types. We further analyze the distinct knowledge and hallucination patterns between models, showing that different hallucinations do occur despite shared parametric knowledge. Turning to the certainty axis, we identify a particularly concerning subset of hallucinations where models hallucinate with certainty despite having the correct knowledge internally. We introduce a new evaluation metric to measure the effectiveness of mitigation methods on this subset, revealing that while some methods perform well on average, they fail disproportionately on these critical cases. Our findings highlight the importance of considering both knowledge and certainty in hallucination analysis and call for targeted mitigation approaches that consider the hallucination underlying factors. 

---
# Beyond Neural Incompatibility: Easing Cross-Scale Knowledge Transfer in Large Language Models through Latent Semantic Alignment 

**Authors**: Jian Gu, Aldeida Aleti, Chunyang Chen, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24208)  

**Abstract**: Large Language Models (LLMs) encode vast amounts of knowledge in their massive parameters, which is accessible to locate, trace, and analyze. Despite advances in neural interpretability, it is still not clear how to transfer knowledge in a fine-grained manner, namely parametric knowledge transfer (PKT). A key problem is enabling effective and efficient knowledge transfer across LLMs of different scales, which is essential for achieving greater flexibility and broader applicability in transferring knowledge between LLMs. Due to neural incompatibility, referring to the architectural and parametric differences between LLMs of varying scales, existing methods that directly reuse layer parameters are severely limited. In this paper, we identify the semantic alignment in latent space as the fundamental prerequisite for LLM cross-scale knowledge transfer. Instead of directly using the layer parameters, our approach takes activations as the medium of layer-wise knowledge transfer. Leveraging the semantics in latent space, our approach is simple and outperforms prior work, better aligning model behaviors across varying scales. Evaluations on four benchmarks demonstrate the efficacy of our method. Further analysis reveals the key factors easing cross-scale knowledge transfer and provides insights into the nature of latent semantic alignment. 

---
# Exploring the Influence of Relevant Knowledge for Natural Language Generation Interpretability 

**Authors**: Iván Martínez-Murillo, Paloma Moreda, Elena Lloret  

**Link**: [PDF](https://arxiv.org/pdf/2510.24179)  

**Abstract**: This paper explores the influence of external knowledge integration in Natural Language Generation (NLG), focusing on a commonsense generation task. We extend the CommonGen dataset by creating KITGI, a benchmark that pairs input concept sets with retrieved semantic relations from ConceptNet and includes manually annotated outputs. Using the T5-Large model, we compare sentence generation under two conditions: with full external knowledge and with filtered knowledge where highly relevant relations were deliberately removed. Our interpretability benchmark follows a three-stage method: (1) identifying and removing key knowledge, (2) regenerating sentences, and (3) manually assessing outputs for commonsense plausibility and concept coverage. Results show that sentences generated with full knowledge achieved 91\% correctness across both criteria, while filtering reduced performance drastically to 6\%. These findings demonstrate that relevant external knowledge is critical for maintaining both coherence and concept coverage in NLG. This work highlights the importance of designing interpretable, knowledge-enhanced NLG systems and calls for evaluation frameworks that capture the underlying reasoning beyond surface-level metrics. 

---
# MuSaG: A Multimodal German Sarcasm Dataset with Full-Modal Annotations 

**Authors**: Aaron Scott, Maike Züfle, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.24178)  

**Abstract**: Sarcasm is a complex form of figurative language in which the intended meaning contradicts the literal one. Its prevalence in social media and popular culture poses persistent challenges for natural language understanding, sentiment analysis, and content moderation. With the emergence of multimodal large language models, sarcasm detection extends beyond text and requires integrating cues from audio and vision. We present MuSaG, the first German multimodal sarcasm detection dataset, consisting of 33 minutes of manually selected and human-annotated statements from German television shows. Each instance provides aligned text, audio, and video modalities, annotated separately by humans, enabling evaluation in unimodal and multimodal settings. We benchmark nine open-source and commercial models, spanning text, audio, vision, and multimodal architectures, and compare their performance to human annotations. Our results show that while humans rely heavily on audio in conversational settings, models perform best on text. This highlights a gap in current multimodal models and motivates the use of MuSaG for developing models better suited to realistic scenarios. We release MuSaG publicly to support future research on multimodal sarcasm detection and human-model alignment. 

---
# Ko-MuSR: A Multistep Soft Reasoning Benchmark for LLMs Capable of Understanding Korean 

**Authors**: Chanwoo Park, Suyoung Park, JiA Kang, Jongyeon Park, Sangho Kim, Hyunji M. Park, Sumin Bae, Mingyu Kang, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.24150)  

**Abstract**: We present Ko-MuSR, the first benchmark to comprehensively evaluate multistep, soft reasoning in long Korean narratives while minimizing data contamination. Built following MuSR, Ko-MuSR features fully Korean narratives, reasoning chains, and multiple-choice questions verified by human annotators for logical consistency and answerability. Evaluations of four large language models -- two multilingual and two Korean-specialized -- show that multilingual models outperform Korean-focused ones even in Korean reasoning tasks, indicating cross-lingual generalization of reasoning ability. Carefully designed prompting strategies, which combine few-shot examples, reasoning traces, and task-specific hints, further boost accuracy, approaching human-level performance. Ko-MuSR offers a solid foundation for advancing Korean NLP by enabling systematic evaluation of long-context reasoning and prompting strategies. 

---
# Beyond Line-Level Filtering for the Pretraining Corpora of LLMs 

**Authors**: Chanwoo Park, Suyoung Park, Yelim Ahn, Jongmin Kim, Jongyeon Park, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.24139)  

**Abstract**: While traditional line-level filtering techniques, such as line-level deduplication and trailing-punctuation filters, are commonly used, these basic methods can sometimes discard valuable content, negatively affecting downstream performance. In this paper, we introduce two methods-pattern-aware line-level deduplication (PLD) and pattern-aware trailing punctuation filtering (PTF)-by enhancing the conventional filtering techniques. Our approach not only considers line-level signals but also takes into account their sequential distribution across documents, enabling us to retain structurally important content that might otherwise be removed. We evaluate these proposed methods by training small language models (1 B parameters) in both English and Korean. The results demonstrate that our methods consistently improve performance on multiple-choice benchmarks and significantly enhance generative question-answering accuracy on both SQuAD v1 and KorQuAD v1. 

---
# Reinforcement Learning for Long-Horizon Multi-Turn Search Agents 

**Authors**: Vivek Kalyan, Martin Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2510.24126)  

**Abstract**: Large Language Model (LLM) agents can leverage multiple turns and tools to solve complex tasks, with prompt-based approaches achieving strong performance. This work demonstrates that Reinforcement Learning (RL) can push capabilities significantly further by learning from experience. Through experiments on a legal document search benchmark, we show that our RL-trained 14 Billion parameter model outperforms frontier class models (85% vs 78% accuracy). In addition, we explore turn-restricted regimes, during training and at test-time, that show these agents achieve better results if allowed to operate over longer multi-turn horizons. 

---
# Squrve: A Unified and Modular Framework for Complex Real-World Text-to-SQL Tasks 

**Authors**: Yihan Wang, Peiyu Liu, Runyu Chen, Jiaxing Pu, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24102)  

**Abstract**: Text-to-SQL technology has evolved rapidly, with diverse academic methods achieving impressive results. However, deploying these techniques in real-world systems remains challenging due to limited integration tools. Despite these advances, we introduce Squrve, a unified, modular, and extensive Text-to-SQL framework designed to bring together research advances and real-world applications. Squrve first establishes a universal execution paradigm that standardizes invocation interfaces, then proposes a multi-actor collaboration mechanism based on seven abstracted effective atomic actor components. Experiments on widely adopted benchmarks demonstrate that the collaborative workflows consistently outperform the original individual methods, thereby opening up a new effective avenue for tackling complex real-world queries. The codes are available at this https URL. 

---
# RegSpeech12: A Regional Corpus of Bengali Spontaneous Speech Across Dialects 

**Authors**: Md. Rezuwan Hassan, Azmol Hossain, Kanij Fatema, Rubayet Sabbir Faruque, Tanmoy Shome, Ruwad Naswan, Trina Chakraborty, Md. Foriduzzaman Zihad, Tawsif Tashwar Dipto, Nazia Tasnim, Nazmuddoha Ansary, Md. Mehedi Hasan Shawon, Ahmed Imtiaz Humayun, Md. Golam Rabiul Alam, Farig Sadeque, Asif Sushmit  

**Link**: [PDF](https://arxiv.org/pdf/2510.24096)  

**Abstract**: The Bengali language, spoken extensively across South Asia and among diasporic communities, exhibits considerable dialectal diversity shaped by geography, culture, and history. Phonological and pronunciation-based classifications broadly identify five principal dialect groups: Eastern Bengali, Manbhumi, Rangpuri, Varendri, and Rarhi. Within Bangladesh, further distinctions emerge through variation in vocabulary, syntax, and morphology, as observed in regions such as Chittagong, Sylhet, Rangpur, Rajshahi, Noakhali, and Barishal. Despite this linguistic richness, systematic research on the computational processing of Bengali dialects remains limited. This study seeks to document and analyze the phonetic and morphological properties of these dialects while exploring the feasibility of building computational models particularly Automatic Speech Recognition (ASR) systems tailored to regional varieties. Such efforts hold potential for applications in virtual assistants and broader language technologies, contributing to both the preservation of dialectal diversity and the advancement of inclusive digital tools for Bengali-speaking communities. The dataset created for this study is released for public use. 

---
# Global PIQA: Evaluating Physical Commonsense Reasoning Across 100+ Languages and Cultures 

**Authors**: Tyler A. Chang, Catherine Arnett, Abdelrahman Eldesokey, Abdelrahman Sadallah, Abeer Kashar, Abolade Daud, Abosede Grace Olanihun, Adamu Labaran Mohammed, Adeyemi Praise, Adhikarinayum Meerajita Sharma, Aditi Gupta, Afitab Iyigun, Afonso Simplício, Ahmed Essouaied, Aicha Chorana, Akhil Eppa, Akintunde Oladipo, Akshay Ramesh, Aleksei Dorkin, Alfred Malengo Kondoro, Alham Fikri Aji, Ali Eren Çetintaş, Allan Hanbury, Alou Dembele, Alp Niksarli, Álvaro Arroyo, Amin Bajand, Amol Khanna, Ana Chkhaidze, Ana Condez, Andiswa Mkhonto, Andrew Hoblitzell, Andrew Tran, Angelos Poulis, Anirban Majumder, Anna Vacalopoulou, Annette Kuuipolani Kanahele Wong, Annika Simonsen, Anton Kovalev, Ashvanth.S, Ayodeji Joseph Lana, Barkin Kinay, Bashar Alhafni, Benedict Cibalinda Busole, Bernard Ghanem, Bharti Nathani, Biljana Stojanovska Đurić, Bola Agbonile, Bragi Bergsson, Bruce Torres Fischer, Burak Tutar, Burcu Alakuş Çınar, Cade J. Kanoniakapueo Kane, Can Udomcharoenchaikit, Catherine Arnett, Chadi Helwe, Chaithra Reddy Nerella, Chen Cecilia Liu, Chiamaka Glory Nwokolo, Cristina España-Bonet, Cynthia Amol, DaeYeop Lee, Dana Arad, Daniil Dzenhaliou, Daria Pugacheva, Dasol Choi, Daud Abolade, David Liu, David Semedo, Deborah Popoola, Deividas Mataciunas, Delphine Nyaboke, Dhyuthy Krishna Kumar, Diogo Glória-Silva, Diogo Tavares, Divyanshu Goyal, DongGeon Lee, Ebele Nwamaka Anajemba, Egonu Ngozi Grace, Elena Mickel, Elena Tutubalina, Elias Herranen, Emile Anand, Emmanuel Habumuremyi, Emuobonuvie Maria Ajiboye, Eryawan Presma Yulianrifat, Esther Adenuga, Ewa Rudnicka, Faith Olabisi Itiola, Faran Taimoor Butt, Fathima Thekkekara, Fatima Haouari, Filbert Aurelian Tjiaranata, Firas Laakom, Francesca Grasso, Francesco Orabona, Francesco Periti, Gbenga Kayode Solomon, Gia Nghia Ngo, Gloria Udhehdhe-oze  

**Link**: [PDF](https://arxiv.org/pdf/2510.24081)  

**Abstract**: To date, there exist almost no culturally-specific evaluation benchmarks for large language models (LLMs) that cover a large number of languages and cultures. In this paper, we present Global PIQA, a participatory commonsense reasoning benchmark for over 100 languages, constructed by hand by 335 researchers from 65 countries around the world. The 116 language varieties in Global PIQA cover five continents, 14 language families, and 23 writing systems. In the non-parallel split of Global PIQA, over 50% of examples reference local foods, customs, traditions, or other culturally-specific elements. We find that state-of-the-art LLMs perform well on Global PIQA in aggregate, but they exhibit weaker performance in lower-resource languages (up to a 37% accuracy gap, despite random chance at 50%). Open models generally perform worse than proprietary models. Global PIQA highlights that in many languages and cultures, everyday knowledge remains an area for improvement, alongside more widely-discussed capabilities such as complex reasoning and expert knowledge. Beyond its uses for LLM evaluation, we hope that Global PIQA provides a glimpse into the wide diversity of cultures in which human language is embedded. 

---
# Challenging Multilingual LLMs: A New Taxonomy and Benchmark for Unraveling Hallucination in Translation 

**Authors**: Xinwei Wu, Heng Liu, Jiang Zhou, Xiaohu Zhao, Linlong Xu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24073)  

**Abstract**: Large Language Models (LLMs) have advanced machine translation but remain vulnerable to hallucinations. Unfortunately, existing MT benchmarks are not capable of exposing failures in multilingual LLMs. To disclose hallucination in multilingual LLMs, we introduce a diagnostic framework with a taxonomy that separates Instruction Detachment from Source Detachment. Guided by this taxonomy, we create HalloMTBench, a multilingual, human-verified benchmark across 11 English-to-X directions. We employed 4 frontier LLMs to generate candidates and scrutinize these candidates with an ensemble of LLM judges, and expert validation. In this way, we curate 5,435 high-quality instances. We have evaluated 17 LLMs on HalloMTBench. Results reveal distinct ``hallucination triggers'' -- unique failure patterns reflecting model scale, source length sensitivity, linguistic biases, and Reinforcement-Learning (RL) amplified language mixing. HalloMTBench offers a forward-looking testbed for diagnosing LLM translation failures. HalloMTBench is available in this https URL. 

---
# Pie: A Programmable Serving System for Emerging LLM Applications 

**Authors**: In Gim, Zhiyao Ma, Seung-seob Lee, Lin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.24051)  

**Abstract**: Emerging large language model (LLM) applications involve diverse reasoning strategies and agentic workflows, straining the capabilities of existing serving systems built on a monolithic token generation loop. This paper introduces Pie, a programmable LLM serving system designed for flexibility and efficiency. Pie decomposes the traditional generation loop into fine-grained service handlers exposed via an API and delegates control of the generation process to user-provided programs, called inferlets. This enables applications to implement new KV cache strategies, bespoke generation logic, and seamlessly integrate computation and I/O-entirely within the application, without requiring modifications to the serving system. Pie executes inferlets using WebAssembly, benefiting from its lightweight sandboxing. Our evaluation shows Pie matches state-of-the-art performance on standard tasks (3-12% latency overhead) while significantly improving latency and throughput (1.3x-3.4x higher) on agentic workflows by enabling application-specific optimizations. 

---
# Success and Cost Elicit Convention Formation for Efficient Communication 

**Authors**: Saujas Vaduguru, Yilun Hua, Yoav Artzi, Daniel Fried  

**Link**: [PDF](https://arxiv.org/pdf/2510.24023)  

**Abstract**: Humans leverage shared conversational context to become increasingly successful and efficient at communicating over time. One manifestation of this is the formation of ad hoc linguistic conventions, which allow people to coordinate on short, less costly utterances that are understood using shared conversational context. We present a method to train large multimodal models to form conventions, enabling efficient communication. Our approach uses simulated reference games between models, and requires no additional human-produced data. In repeated reference games involving photographs and tangram images, our method enables models to communicate efficiently with people: reducing the message length by up to 41% while increasing success by 15% over the course of the interaction. Human listeners respond faster when interacting with our model that forms conventions. We also show that training based on success or cost alone is insufficient - both are necessary to elicit convention formation. 

---
# SpecKD: Speculative Decoding for Effective Knowledge Distillation of LLMs 

**Authors**: Haiduo Huang, Jiangcheng Song, Yadong Zhang, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.24021)  

**Abstract**: Knowledge Distillation (KD) has become a cornerstone technique for compressing Large Language Models (LLMs) into smaller, more efficient student models. However, conventional KD approaches typically apply the distillation loss uniformly across all tokens, regardless of the teacher's confidence. This indiscriminate mimicry can introduce noise, as the student is forced to learn from the teacher's uncertain or high-entropy predictions, which may ultimately harm student performance-especially when the teacher is much larger and more powerful. To address this, we propose Speculative Knowledge Distillation (SpecKD), a novel, plug-and-play framework that introduces a dynamic, token-level gating mechanism inspired by the "propose-and-verify" paradigm of speculative decoding. At each step, the student's token proposal is verified against the teacher's distribution; the distillation loss is selectively applied only to "accepted" tokens, while "rejected" tokens are masked out. Extensive experiments on diverse text generation tasks show that SpecKD consistently and significantly outperforms strong KD baselines, leading to more stable training and more capable student models, and achieving state-of-the-art results. 

---
# Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward 

**Authors**: Hao An, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24020)  

**Abstract**: Mitigating hallucinations in Large Language Models (LLMs) is critical for their reliable deployment. Existing methods typically fine-tune LLMs to abstain from answering questions beyond their knowledge scope. However, these methods often rely on coarse-grained signals to guide LLMs to abstain, such as overall confidence or uncertainty scores on multiple sampled answers, which may result in an imprecise awareness of the model's own knowledge boundaries. To this end, we propose a novel reinforcement learning framework built on $\textbf{\underline{Fi}ne-grained \underline{S}emantic \underline{Co}nfidence \underline{Re}ward (\Ours)}$, which guides LLMs to abstain via sample-specific confidence. Specifically, our method operates by sampling multiple candidate answers and conducting semantic clustering, then training the LLM to retain answers within high-confidence clusters and discard those within low-confidence ones, thereby promoting accurate post-hoc abstention. Additionally, we propose a new metric for evaluating the reliability of abstention fine-tuning tasks more comprehensively. Our method significantly enhances reliability in both in-domain and out-of-distribution benchmarks. 

---
# TEXT2DB: Integration-Aware Information Extraction with Large Language Model Agents 

**Authors**: Yizhu Jiao, Sha Li, Sizhe Zhou, Heng Ji, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.24014)  

**Abstract**: The task of information extraction (IE) is to extract structured knowledge from text. However, it is often not straightforward to utilize IE output due to the mismatch between the IE ontology and the downstream application needs. We propose a new formulation of IE TEXT2DB that emphasizes the integration of IE output and the target database (or knowledge base). Given a user instruction, a document set, and a database, our task requires the model to update the database with values from the document set to satisfy the user instruction. This task requires understanding user instructions for what to extract and adapting to the given DB/KB schema for how to extract on the fly. To evaluate this new task, we introduce a new benchmark featuring common demands such as data infilling, row population, and column addition. In addition, we propose an LLM agent framework OPAL (Observe-PlanAnalyze LLM) which includes an Observer component that interacts with the database, the Planner component that generates a code-based plan with calls to IE models, and the Analyzer component that provides feedback regarding code quality before execution. Experiments show that OPAL can successfully adapt to diverse database schemas by generating different code plans and calling the required IE models. We also highlight difficult cases such as dealing with large databases with complex dependencies and extraction hallucination, which we believe deserve further investigation. Source code: this https URL 

---
# META-RAG: Meta-Analysis-Inspired Evidence-Re-Ranking Method for Retrieval-Augmented Generation in Evidence-Based Medicine 

**Authors**: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Haochun Wang, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.24003)  

**Abstract**: Evidence-based medicine (EBM) holds a crucial role in clinical application. Given suitable medical articles, doctors effectively reduce the incidence of misdiagnoses. Researchers find it efficient to use large language models (LLMs) techniques like RAG for EBM tasks. However, the EBM maintains stringent requirements for evidence, and RAG applications in EBM struggle to efficiently distinguish high-quality evidence. Therefore, inspired by the meta-analysis used in EBM, we provide a new method to re-rank and filter the medical evidence. This method presents multiple principles to filter the best evidence for LLMs to diagnose. We employ a combination of several EBM methods to emulate the meta-analysis, which includes reliability analysis, heterogeneity analysis, and extrapolation analysis. These processes allow the users to retrieve the best medical evidence for the LLMs. Ultimately, we evaluate these high-quality articles and show an accuracy improvement of up to 11.4% in our experiments and results. Our method successfully enables RAG to extract higher-quality and more reliable evidence from the PubMed dataset. This work can reduce the infusion of incorrect knowledge into responses and help users receive more effective replies. 

---
# PICOs-RAG: PICO-supported Query Rewriting for Retrieval-Augmented Generation in Evidence-Based Medicine 

**Authors**: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.23998)  

**Abstract**: Evidence-based medicine (EBM) research has always been of paramount importance. It is important to find appropriate medical theoretical support for the needs from physicians or patients to reduce the occurrence of medical accidents. This process is often carried out by human querying relevant literature databases, which lacks objectivity and efficiency. Therefore, researchers utilize retrieval-augmented generation (RAG) to search for evidence and generate responses automatically. However, current RAG methods struggle to handle complex queries in real-world clinical scenarios. For example, when queries lack certain information or use imprecise language, the model may retrieve irrelevant evidence and generate unhelpful answers. To address this issue, we present the PICOs-RAG to expand the user queries into a better format. Our method can expand and normalize the queries into professional ones and use the PICO format, a search strategy tool present in EBM, to extract the most important information used for retrieval. This approach significantly enhances retrieval efficiency and relevance, resulting in up to an 8.8\% improvement compared to the baseline evaluated by our method. Thereby the PICOs-RAG improves the performance of the large language models into a helpful and reliable medical assistant in EBM. 

---
# M-Eval: A Heterogeneity-Based Framework for Multi-evidence Validation in Medical RAG Systems 

**Authors**: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Haochun Wang, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.23995)  

**Abstract**: Retrieval-augmented Generation (RAG) has demonstrated potential in enhancing medical question-answering systems through the integration of large language models (LLMs) with external medical literature. LLMs can retrieve relevant medical articles to generate more professional responses efficiently. However, current RAG applications still face problems. They generate incorrect information, such as hallucinations, and they fail to use external knowledge correctly. To solve these issues, we propose a new method named M-Eval. This method is inspired by the heterogeneity analysis approach used in Evidence-Based Medicine (EBM). Our approach can check for factual errors in RAG responses using evidence from multiple sources. First, we extract additional medical literature from external knowledge bases. Then, we retrieve the evidence documents generated by the RAG system. We use heterogeneity analysis to check whether the evidence supports different viewpoints in the response. In addition to verifying the accuracy of the response, we also assess the reliability of the evidence provided by the RAG system. Our method shows an improvement of up to 23.31% accuracy across various LLMs. This work can help detect errors in current RAG-based medical systems. It also makes the applications of LLMs more reliable and reduces diagnostic errors. 

---
# Uncovering the Potential Risks in Unlearning: Danger of English-only Unlearning in Multilingual LLMs 

**Authors**: Kyomin Hwang, Hyeonjin Kim, Seungyeon Kim, Sunghyun Wee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.23949)  

**Abstract**: There have been a couple of studies showing that attempting to erase multilingual knowledge using only English data is insufficient for multilingual LLMs. However, their analyses remain highly performance-oriented. In this paper, we switch the point of view to evaluation, and address an additional blind spot which reveals itself when the multilingual LLM is fully finetuned with parallel multilingual dataset before unlearning. Here, language confusion occurs whereby a model responds in language different from that of the input prompt. Language confusion is a problematic phenomenon in unlearning, causing the standard reference-based metrics to fail. We tackle this phenomenon in three steps: (1) introduce N-gram-based Language-Mix (N-Mix) score to quantitatively show the language confusion is pervasive and consistent in multilingual LLMs, (2) demonstrate that reference-based metrics result in false negatives when N-Mix score is high, and(3) suggest the need of new type of unlearning evaluation that can directly assess the content of the generated sentences. We call this type of metrics as semantic-based metric. 

---
# Leveraging LLMs for Early Alzheimer's Prediction 

**Authors**: Tananun Songdechakraiwut  

**Link**: [PDF](https://arxiv.org/pdf/2510.23946)  

**Abstract**: We present a connectome-informed LLM framework that encodes dynamic fMRI connectivity as temporal sequences, applies robust normalization, and maps these data into a representation suitable for a frozen pre-trained LLM for clinical prediction. Applied to early Alzheimer's detection, our method achieves sensitive prediction with error rates well below clinically recognized margins, with implications for timely Alzheimer's intervention. 

---
# Auto prompting without training labels: An LLM cascade for product quality assessment in e-commerce catalogs 

**Authors**: Soham Satyadharma, Fatemeh Sheikholeslami, Swati Kaul, Aziz Umit Batur, Suleiman A. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23941)  

**Abstract**: We introduce a novel, training free cascade for auto-prompting Large Language Models (LLMs) to assess product quality in e-commerce. Our system requires no training labels or model fine-tuning, instead automatically generating and refining prompts for evaluating attribute quality across tens of thousands of product category-attribute pairs. Starting from a seed of human-crafted prompts, the cascade progressively optimizes instructions to meet catalog-specific requirements. This approach bridges the gap between general language understanding and domain-specific knowledge at scale in complex industrial catalogs. Our extensive empirical evaluations shows the auto-prompt cascade improves precision and recall by $8-10\%$ over traditional chain-of-thought prompting. Notably, it achieves these gains while reducing domain expert effort from 5.1 hours to 3 minutes per attribute - a $99\%$ reduction. Additionally, the cascade generalizes effectively across five languages and multiple quality assessment tasks, consistently maintaining performance gains. 

---
# Agent-based Automated Claim Matching with Instruction-following LLMs 

**Authors**: Dina Pisarevskaya, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2510.23924)  

**Abstract**: We present a novel agent-based approach for the automated claim matching task with instruction-following LLMs. We propose a two-step pipeline that first generates prompts with LLMs, to then perform claim matching as a binary classification task with LLMs. We demonstrate that LLM-generated prompts can outperform SOTA with human-generated prompts, and that smaller LLMs can do as well as larger ones in the generation process, allowing to save computational resources. We also demonstrate the effectiveness of using different LLMs for each step of the pipeline, i.e. using an LLM for prompt generation, and another for claim matching. Our investigation into the prompt generation process in turn reveals insights into the LLMs' understanding of claim matching. 

---
# Breaking the Benchmark: Revealing LLM Bias via Minimal Contextual Augmentation 

**Authors**: Kaveh Eskandari Miandoab, Mahammed Kamruzzaman, Arshia Gharooni, Gene Louis Kim, Vasanth Sarathy, Ninareh Mehrabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23921)  

**Abstract**: Large Language Models have been shown to demonstrate stereotypical biases in their representations and behavior due to the discriminative nature of the data that they have been trained on. Despite significant progress in the development of methods and models that refrain from using stereotypical information in their decision-making, recent work has shown that approaches used for bias alignment are brittle. In this work, we introduce a novel and general augmentation framework that involves three plug-and-play steps and is applicable to a number of fairness evaluation benchmarks. Through application of augmentation to a fairness evaluation dataset (Bias Benchmark for Question Answering (BBQ)), we find that Large Language Models (LLMs), including state-of-the-art open and closed weight models, are susceptible to perturbations to their inputs, showcasing a higher likelihood to behave stereotypically. Furthermore, we find that such models are more likely to have biased behavior in cases where the target demographic belongs to a community less studied by the literature, underlining the need to expand the fairness and safety research to include more diverse communities. 

---
# AfriMTEB and AfriE5: Benchmarking and Adapting Text Embedding Models for African Languages 

**Authors**: Kosei Uemura, Miaoran Zhang, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2510.23896)  

**Abstract**: Text embeddings are an essential building component of several NLP tasks such as retrieval-augmented generation which is crucial for preventing hallucinations in LLMs. Despite the recent release of massively multilingual MTEB (MMTEB), African languages remain underrepresented, with existing tasks often repurposed from translation benchmarks such as FLORES clustering or SIB-200. In this paper, we introduce AfriMTEB -- a regional expansion of MMTEB covering 59 languages, 14 tasks, and 38 datasets, including six newly added datasets. Unlike many MMTEB datasets that include fewer than five languages, the new additions span 14 to 56 African languages and introduce entirely new tasks, such as hate speech detection, intent detection, and emotion classification, which were not previously covered. Complementing this, we present AfriE5, an adaptation of the instruction-tuned mE5 model to African languages through cross-lingual contrastive distillation. Our evaluation shows that AfriE5 achieves state-of-the-art performance, outperforming strong baselines such as Gemini-Embeddings and mE5. 

---
# Language Models for Longitudinal Clinical Prediction 

**Authors**: Tananun Songdechakraiwut, Michael Lutz  

**Link**: [PDF](https://arxiv.org/pdf/2510.23884)  

**Abstract**: We explore a lightweight framework that adapts frozen large language models to analyze longitudinal clinical data. The approach integrates patient history and context within the language model space to generate accurate forecasts without model fine-tuning. Applied to neuropsychological assessments, it achieves accurate and reliable performance even with minimal training data, showing promise for early-stage Alzheimer's monitoring. 

---
# OraPlan-SQL: A Planning-Centric Framework for Complex Bilingual NL2SQL Reasoning 

**Authors**: Marianne Menglin Liu, Sai Ashish Somayajula, Syed Fahad Allam Shah, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2510.23870)  

**Abstract**: We present OraPlan-SQL, our system for the Archer NL2SQL Evaluation Challenge 2025, a bilingual benchmark requiring complex reasoning such as arithmetic, commonsense, and hypothetical inference. OraPlan-SQL ranked first, exceeding the second-best system by more than 6% in execution accuracy (EX), with 55.0% in English and 56.7% in Chinese, while maintaining over 99% SQL validity (VA). Our system follows an agentic framework with two components: Planner agent that generates stepwise natural language plans, and SQL agent that converts these plans into executable SQL. Since SQL agent reliably adheres to the plan, our refinements focus on the planner. Unlike prior methods that rely on multiple sub-agents for planning and suffer from orchestration overhead, we introduce a feedback-guided meta-prompting strategy to refine a single planner. Failure cases from a held-out set are clustered with human input, and an LLM distills them into corrective guidelines that are integrated into the planner's system prompt, improving generalization without added complexity. For the multilingual scenario, to address transliteration and entity mismatch issues, we incorporate entity-linking guidelines that generate alternative surface forms for entities and explicitly include them in the plan. Finally, we enhance reliability through plan diversification: multiple candidate plans are generated for each query, with the SQL agent producing a query for each plan, and final output selected via majority voting over their executions. 

---
# Can LLMs Narrate Tabular Data? An Evaluation Framework for Natural Language Representations of Text-to-SQL System Outputs 

**Authors**: Jyotika Singh, Weiyi Sun, Amit Agarwal, Viji Krishnamurthy, Yassine Benajiba, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2510.23854)  

**Abstract**: In modern industry systems like multi-turn chat agents, Text-to-SQL technology bridges natural language (NL) questions and database (DB) querying. The conversion of tabular DB results into NL representations (NLRs) enables the chat-based interaction. Currently, NLR generation is typically handled by large language models (LLMs), but information loss or errors in presenting tabular results in NL remains largely unexplored. This paper introduces a novel evaluation method - Combo-Eval - for judgment of LLM-generated NLRs that combines the benefits of multiple existing methods, optimizing evaluation fidelity and achieving a significant reduction in LLM calls by 25-61%. Accompanying our method is NLR-BIRD, the first dedicated dataset for NLR benchmarking. Through human evaluations, we demonstrate the superior alignment of Combo-Eval with human judgments, applicable across scenarios with and without ground truth references. 

---
# Temporal Blindness in Multi-Turn LLM Agents: Misaligned Tool Use vs. Human Time Perception 

**Authors**: Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Kazem Faghih, Parsa Hosseini, Wenxiao Wang, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23853)  

**Abstract**: Large language model agents are increasingly used in multi-turn conversational settings to interact with and execute tasks in dynamic environments. However, a key limitation is their temporal blindness: they, by default, operate with a stationary context, failing to account for the real-world time elapsed between messages. This becomes a critical liability when an agent must decide whether to invoke a tool based on how much time has passed since the last observation. Without temporal awareness, agents often either over-rely on previous context (skipping necessary tool calls), or under-rely on it (unnecessarily repeating tool calls). To study this challenge, we introduce TicToc-v1, a test set of multi-turn user-agent trajectories across 34 scenarios with varying time sensitivity. Each trajectory ends with a user question, where the need for a tool call depends on the amount of time elapsed since the last message. To give LLMs temporal context, we augment dialogue messages with explicit timestamps, bridging the gap between static dialogue and evolving environments. We then collected human preferences for these samples, creating two subsets: one where humans preferred relying on the previous observation (prefer-noTool), and another where they preferred a new tool call (prefer-Tool). We evaluated how well LLM tool-calling decisions align with human preferences under varying time intervals on TicToc-v1. Our analysis show that without time information, most models perform only slightly better than random, with the top alignment rate being just over 60%. While adding timestamps leads to a slight improvement, particularly for larger models, the improvement is modest, peaking at around 65%. We also show that naive, prompt-based alignment have limited effectiveness. Our findings highlight the need for specific post-training alignment to align multi-turn LLM tool use with human temporal perception. 

---
# CRADLE Bench: A Clinician-Annotated Benchmark for Multi-Faceted Mental Health Crisis and Safety Risk Detection 

**Authors**: Grace Byun, Rebecca Lipschutz, Sean T. Minton, Abigail Lott, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23845)  

**Abstract**: Detecting mental health crisis situations such as suicide ideation, rape, domestic violence, child abuse, and sexual harassment is a critical yet underexplored challenge for language models. When such situations arise during user--model interactions, models must reliably flag them, as failure to do so can have serious consequences. In this work, we introduce CRADLE BENCH, a benchmark for multi-faceted crisis detection. Unlike previous efforts that focus on a limited set of crisis types, our benchmark covers seven types defined in line with clinical standards and is the first to incorporate temporal labels. Our benchmark provides 600 clinician-annotated evaluation examples and 420 development examples, together with a training corpus of around 4K examples automatically labeled using a majority-vote ensemble of multiple language models, which significantly outperforms single-model annotation. We further fine-tune six crisis detection models on subsets defined by consensus and unanimous ensemble agreement, providing complementary models trained under different agreement criteria. 

---
# How Pragmatics Shape Articulation: A Computational Case Study in STEM ASL Discourse 

**Authors**: Saki Imai, Lee Kezar, Laurel Aichler, Mert Inan, Erin Walker, Alicia Wooten, Lorna Quandt, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2510.23842)  

**Abstract**: Most state-of-the-art sign language models are trained on interpreter or isolated vocabulary data, which overlooks the variability that characterizes natural dialogue. However, human communication dynamically adapts to contexts and interlocutors through spatiotemporal changes and articulation style. This specifically manifests itself in educational settings, where novel vocabularies are used by teachers, and students. To address this gap, we collect a motion capture dataset of American Sign Language (ASL) STEM (Science, Technology, Engineering, and Mathematics) dialogue that enables quantitative comparison between dyadic interactive signing, solo signed lecture, and interpreted articles. Using continuous kinematic features, we disentangle dialogue-specific entrainment from individual effort reduction and show spatiotemporal changes across repeated mentions of STEM terms. On average, dialogue signs are 24.6%-44.6% shorter in duration than the isolated signs, and show significant reductions absent in monologue contexts. Finally, we evaluate sign embedding models on their ability to recognize STEM signs and approximate how entrained the participants become over time. Our study bridges linguistic analysis and computational modeling to understand how pragmatics shape sign articulation and its representation in sign language technologies. 

---
# Beyond Understanding: Evaluating the Pragmatic Gap in LLMs' Cultural Processing of Figurative Language 

**Authors**: Mena Attia, Aashiq Muhamed, Mai Alkhamissi, Thamar Solorio, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2510.23828)  

**Abstract**: We present a comprehensive evaluation of the ability of large language models (LLMs) to process culturally grounded language, specifically to understand and pragmatically use figurative expressions that encode local knowledge and cultural nuance. Using figurative language as a proxy for cultural nuance and local knowledge, we design evaluation tasks for contextual understanding, pragmatic use, and connotation interpretation in Arabic and English. We evaluate 22 open- and closed-source LLMs on Egyptian Arabic idioms, multidialectal Arabic proverbs, and English proverbs. Our results show a consistent hierarchy: the average accuracy for Arabic proverbs is 4.29% lower than for English proverbs, and performance for Egyptian idioms is 10.28% lower than for Arabic proverbs. For the pragmatic use task, accuracy drops by 14.07% relative to understanding, though providing contextual idiomatic sentences improves accuracy by 10.66%. Models also struggle with connotative meaning, reaching at most 85.58% agreement with human annotators on idioms with 100% inter-annotator agreement. These findings demonstrate that figurative language serves as an effective diagnostic for cultural reasoning: while LLMs can often interpret figurative meaning, they face challenges in using it appropriately. To support future research, we release Kinayat, the first dataset of Egyptian Arabic idioms designed for both figurative understanding and pragmatic use evaluation. 

---
# BitSkip: An Empirical Analysis of Quantization and Early Exit Composition 

**Authors**: Ramshankar Bhuvaneswaran, Handan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23766)  

**Abstract**: The pursuit of efficient Large Language Models (LLMs) has led to increasingly complex techniques like extreme quantization and dynamic routing. While individual benefits of these methods are well-documented, their compositional effects remain poorly understood. This paper introduces BitSkip, a hybrid architectural framework for systematically explor- ing these interactions. Counter-intuitively, our findings reveal that a simple 8-bit quantized model without Hadamard transform (BitSkip-V1) not only outperforms its more complex 4-bit and Hadamard-enhanced counterparts but also competes the full-precision baseline in quality (perplexity of 1.13 vs 1.19) . The introduction of Hadamard transforms, even at 8- bit precision, catastrophically degraded performance by over 37,000%, tracing fundamental training instability. Our BitSkip-V1 recipe demonstrates superior early-exit characteristics, with layer 18 providing optimal 32.5% speed gain for minimal 4% quality loss. 

---
# Evaluating Long-Term Memory for Long-Context Question Answering 

**Authors**: Alessandra Terranova, Björn Ross, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2510.23730)  

**Abstract**: In order for large language models to achieve true conversational continuity and benefit from experiential learning, they need memory. While research has focused on the development of complex memory systems, it remains unclear which types of memory are most effective for long-context conversational tasks. We present a systematic evaluation of memory-augmented methods using LoCoMo, a benchmark of synthetic long-context dialogues annotated for question-answering tasks that require diverse reasoning strategies. We analyse full-context prompting, semantic memory through retrieval-augmented generation and agentic memory, episodic memory through in-context learning, and procedural memory through prompt optimization. Our findings show that memory-augmented approaches reduce token usage by over 90% while maintaining competitive accuracy. Memory architecture complexity should scale with model capability, with small foundation models benefitting most from RAG, and strong instruction-tuned reasoning model gaining from episodic learning through reflections and more complex agentic semantic memory. In particular, episodic memory can help LLMs recognise the limits of their own knowledge. 

---
# STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence 

**Authors**: Zihan Liu, Zhikang Niu, Qiuyang Xiao, Zhisheng Zheng, Ruoqi Yuan, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Jianze Liang, Xie Chen, Leilei Sun, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24693)  

**Abstract**: Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning. We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world. 

---
# Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs 

**Authors**: Huanyu Zhang, Wenshan Wu, Chengzu Li, Ning Shang, Yan Xia, Yangyu Huang, Yifan Zhang, Li Dong, Zhang Zhang, Liang Wang, Tieniu Tan, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.24514)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at visual understanding, they often struggle in complex scenarios that require visual planning and imagination. Inspired by how humans use sketching as a form of visual thinking to develop and communicate ideas, we introduce Latent Sketchpad, a framework that equips MLLMs with an internal visual scratchpad. The internal visual representations of MLLMs have traditionally been confined to perceptual understanding. We repurpose them to support generative visual thought without compromising reasoning ability. Building on frontier MLLMs, our approach integrates visual generation directly into their native autoregressive reasoning process. It allows the model to interleave textual reasoning with the generation of visual latents. These latents guide the internal thought process and can be translated into sketch images for interpretability. To realize this, we introduce two components: a Context-Aware Vision Head autoregressively produces visual representations, and a pretrained Sketch Decoder renders these into human-interpretable images. We evaluate the framework on our new dataset MazePlanning. Experiments across various MLLMs show that Latent Sketchpad delivers comparable or even superior reasoning performance to their backbone. It further generalizes across distinct frontier MLLMs, including Gemma3 and Qwen2.5-VL. By extending model's textual reasoning to visual thinking, our framework opens new opportunities for richer human-computer interaction and broader applications. More details and resources are available on our project page: this https URL. 

---
# Law in Silico: Simulating Legal Society with LLM-Based Agents 

**Authors**: Yiding Wang, Yuxuan Chen, Fanxu Meng, Xifan Chen, Xiaolei Yang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24442)  

**Abstract**: Since real-world legal experiments are often costly or infeasible, simulating legal societies with Artificial Intelligence (AI) systems provides an effective alternative for verifying and developing legal theory, as well as supporting legal administration. Large Language Models (LLMs), with their world knowledge and role-playing capabilities, are strong candidates to serve as the foundation for legal society simulation. However, the application of LLMs to simulate legal systems remains underexplored. In this work, we introduce Law in Silico, an LLM-based agent framework for simulating legal scenarios with individual decision-making and institutional mechanisms of legislation, adjudication, and enforcement. Our experiments, which compare simulated crime rates with real-world data, demonstrate that LLM-based agents can largely reproduce macro-level crime trends and provide insights that align with real-world observations. At the same time, micro-level simulations reveal that a well-functioning, transparent, and adaptive legal system offers better protection of the rights of vulnerable individuals. 

---
# OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows 

**Authors**: Qiushi Sun, Mukai Li, Zhoumianze Liu, Zhihui Xie, Fangzhi Xu, Zhangyue Yin, Kanzhi Cheng, Zehao Li, Zichen Ding, Qi Liu, Zhiyong Wu, Zhuosheng Zhang, Ben Kao, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2510.24411)  

**Abstract**: Computer-using agents powered by Vision-Language Models (VLMs) have demonstrated human-like capabilities in operating digital environments like mobile platforms. While these agents hold great promise for advancing digital automation, their potential for unsafe operations, such as system compromise and privacy leakage, is raising significant concerns. Detecting these safety concerns across the vast and complex operational space of mobile environments presents a formidable challenge that remains critically underexplored. To establish a foundation for mobile agent safety research, we introduce MobileRisk-Live, a dynamic sandbox environment accompanied by a safety detection benchmark comprising realistic trajectories with fine-grained annotations. Built upon this, we propose OS-Sentinel, a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions. Experiments show that OS-Sentinel achieves 10%-30% improvements over existing approaches across multiple metrics. Further analysis provides critical insights that foster the development of safer and more reliable autonomous mobile agents. 

---
# Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation 

**Authors**: Lingyue Fu, Bolun Zhang, Hao Guan, Yaoming Zhu, Lin Qiu, Weiwen Liu, Xuezhi Cao, Xunliang Cai, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24358)  

**Abstract**: Recent advances in code agents have enabled automated software development at the project level, supported by large language models (LLMs) and widely adopted tools. However, existing benchmarks for code agent evaluation face two major limitations: high annotation cost and expertise requirements, and rigid evaluation metrics that rely primarily on unit tests. To address these challenges, we propose an agent-driven benchmark construction pipeline that leverages human supervision to efficiently generate diverse and challenging project-level tasks. Based on this approach, we introduce PRDBench, a novel benchmark comprising 50 real-world Python projects across 20 domains, each with structured Product Requirement Document (PRD) requirements, comprehensive evaluation criteria, and reference implementations. PRDBench features rich data sources, high task complexity, and flexible metrics. We further employ an Agent-as-a-Judge paradigm to score agent outputs, enabling the evaluation of various test types beyond unit tests. Extensive experiments on PRDBench demonstrate its effectiveness in assessing the capabilities of both code agents and evaluation agents, providing a scalable and robust framework for annotation and evaluation. 

---
# ViPER: Empowering the Self-Evolution of Visual Perception Abilities in Vision-Language Model 

**Authors**: Juntian Zhang, Song Jin, Chuanqi Cheng, Yuhan Liu, Yankai Lin, Xun Zhang, Yufei Zhang, Fei Jiang, Guojun Yin, Wei Lin, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.24285)  

**Abstract**: The limited capacity for fine-grained visual perception presents a critical bottleneck for Vision-Language Models (VLMs) in real-world applications. Addressing this is challenging due to the scarcity of high-quality data and the limitations of existing methods: supervised fine-tuning (SFT) often compromises general capabilities, while reinforcement fine-tuning (RFT) prioritizes textual reasoning over visual perception. To bridge this gap, we propose a novel two-stage task that structures visual perception learning as a coarse-to-fine progressive process. Based on this task formulation, we develop ViPER, a self-bootstrapping framework specifically designed to enable iterative evolution through self-critiquing and self-prediction. By synergistically integrating image-level and instance-level reconstruction with a two-stage reinforcement learning strategy, ViPER establishes a closed-loop training paradigm, where internally synthesized data directly fuel the enhancement of perceptual ability. Applied to the Qwen2.5-VL family, ViPER produces the Qwen-Viper series. With an average gain of 1.7% on seven comprehensive benchmarks spanning various tasks and up to 6.0% on fine-grained perception, Qwen-Viper consistently demonstrates superior performance across different vision-language scenarios while maintaining generalizability. Beyond enabling self-improvement in perceptual capabilities, ViPER provides concrete evidence for the reciprocal relationship between generation and understanding, a breakthrough to developing more autonomous and capable VLMs. 

---
# VC4VG: Optimizing Video Captions for Text-to-Video Generation 

**Authors**: Yang Du, Zhuoran Lin, Kaiqiang Song, Biao Wang, Zhicheng Zheng, Tiezheng Ge, Bo Zheng, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.24134)  

**Abstract**: Recent advances in text-to-video (T2V) generation highlight the critical role of high-quality video-text pairs in training models capable of producing coherent and instruction-aligned videos. However, strategies for optimizing video captions specifically for T2V training remain underexplored. In this paper, we introduce VC4VG (Video Captioning for Video Generation), a comprehensive caption optimization framework tailored to the needs of T2V this http URL begin by analyzing caption content from a T2V perspective, decomposing the essential elements required for video reconstruction into multiple dimensions, and proposing a principled caption design methodology. To support evaluation, we construct VC4VG-Bench, a new benchmark featuring fine-grained, multi-dimensional, and necessity-graded metrics aligned with T2V-specific this http URL T2V fine-tuning experiments demonstrate a strong correlation between improved caption quality and video generation performance, validating the effectiveness of our approach. We release all benchmark tools and code at this https URL to support further research. 

---
# GraphNet: A Large-Scale Computational Graph Dataset for Tensor Compiler Research 

**Authors**: Xinqi Li, Yiqun Liu, Shan Jiang, Enrong Zheng, Huaijin Zheng, Wenhao Dai, Haodong Deng, Dianhai Yu, Yanjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.24035)  

**Abstract**: We introduce GraphNet, a dataset of 2.7K real-world deep learning computational graphs with rich metadata, spanning six major task categories across multiple deep learning frameworks. To evaluate tensor compiler performance on these samples, we propose the benchmark metric Speedup Score S(t), which jointly considers runtime speedup and execution correctness under tunable tolerance levels, offering a reliable measure of general optimization capability. Furthermore, we extend S(t) to the Error-aware Speedup Score ES(t), which incorporates error information and helps compiler developers identify key performance bottlenecks. In this report, we benchmark the default tensor compilers, CINN for PaddlePaddle and TorchInductor for PyTorch, on computer vision (CV) and natural language processing (NLP) samples to demonstrate the practicality of GraphNet. The full construction pipeline with graph extraction and compiler evaluation tools is available at this https URL . 

---
# emg2speech: synthesizing speech from electromyography using self-supervised speech models 

**Authors**: Harshavardhana T. Gowda, Lee M. Miller  

**Link**: [PDF](https://arxiv.org/pdf/2510.23969)  

**Abstract**: We present a neuromuscular speech interface that translates electromyographic (EMG) signals collected from orofacial muscles during speech articulation directly into audio. We show that self-supervised speech (SS) representations exhibit a strong linear relationship with the electrical power of muscle action potentials: SS features can be linearly mapped to EMG power with a correlation of $r = 0.85$. Moreover, EMG power vectors corresponding to different articulatory gestures form structured and separable clusters in feature space. This relationship: $\text{SS features}$ $\xrightarrow{\texttt{linear mapping}}$ $\text{EMG power}$ $\xrightarrow{\texttt{gesture-specific clustering}}$ $\text{articulatory movements}$, highlights that SS models implicitly encode articulatory mechanisms. Leveraging this property, we directly map EMG signals to SS feature space and synthesize speech, enabling end-to-end EMG-to-speech generation without explicit articulatory models and vocoder training. 

---
# Latent Chain-of-Thought for Visual Reasoning 

**Authors**: Guohao Sun, Hang Hua, Jian Wang, Jiebo Luo, Sohail Dianat, Majid Rabbani, Raghuveer Rao, Zhiqiang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23925)  

**Abstract**: Chain-of-thought (CoT) reasoning is critical for improving the interpretability and reliability of Large Vision-Language Models (LVLMs). However, existing training algorithms such as SFT, PPO, and GRPO may not generalize well across unseen reasoning tasks and heavily rely on a biased reward model. To address this challenge, we reformulate reasoning in LVLMs as posterior inference and propose a scalable training algorithm based on amortized variational inference. By leveraging diversity-seeking reinforcement learning algorithms, we introduce a novel sparse reward function for token-level learning signals that encourage diverse, high-likelihood latent CoT, overcoming deterministic sampling limitations and avoiding reward hacking. Additionally, we implement a Bayesian inference-scaling strategy that replaces costly Best-of-N and Beam Search with a marginal likelihood to efficiently rank optimal rationales and answers. We empirically demonstrate that the proposed method enhances the state-of-the-art LVLMs on seven reasoning benchmarks, in terms of effectiveness, generalization, and interpretability. 

---
# GIFT: Group-relative Implicit Fine Tuning Integrates GRPO with DPO and UNA 

**Authors**: Zhichao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23868)  

**Abstract**: I propose \textbf{G}roup-relative \textbf{I}mplicit \textbf{F}ine \textbf{T}uning (GIFT), a novel reinforcement learning framework for aligning LLMs. Instead of directly maximizing cumulative rewards like PPO or GRPO, GIFT minimizes the discrepancy between implicit and explicit reward models. It combines three key ideas: (1) the online multi-response generation and normalization of GRPO, (2) the implicit reward formulation of DPO, and (3) the implicit-explicit reward alignment principle of UNA. By jointly normalizing the implicit and explicit rewards, GIFT eliminates an otherwise intractable term that prevents effective use of implicit rewards. This normalization transforms the complex reward maximization objective into a simple mean squared error (MSE) loss between the normalized reward functions, converting a non-convex optimization problem into a convex, stable, and analytically differentiable formulation. Unlike offline methods such as DPO and UNA, GIFT remains on-policy and thus retains exploration capability. Compared to GRPO, it requires fewer hyperparameters, converges faster, and generalizes better with significantly reduced training overfitting. Empirically, GIFT achieves superior reasoning and alignment performance on mathematical benchmarks while remaining computationally efficient. 

---
# A Neural Model for Contextual Biasing Score Learning and Filtering 

**Authors**: Wanting Huang, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23849)  

**Abstract**: Contextual biasing improves automatic speech recognition (ASR) by integrating external knowledge, such as user-specific phrases or entities, during decoding. In this work, we use an attention-based biasing decoder to produce scores for candidate phrases based on acoustic information extracted by an ASR encoder, which can be used to filter out unlikely phrases and to calculate bonus for shallow-fusion biasing. We introduce a per-token discriminative objective that encourages higher scores for ground-truth phrases while suppressing distractors. Experiments on the Librispeech biasing benchmark show that our method effectively filters out majority of the candidate phrases, and significantly improves recognition accuracy under different biasing conditions when the scores are used in shallow fusion biasing. Our approach is modular and can be used with any ASR system, and the filtering mechanism can potentially boost performance of other biasing methods. 

---
# RoboOmni: Proactive Robot Manipulation in Omni-modal Context 

**Authors**: Siyin Wang, Jinlan Fu, Feihong Liu, Xinzhe He, Huangxuan Wu, Junhao Shi, Kexin Huang, Zhaoye Fei, Jingjing Gong, Zuxuan Wu, Yugang Jiang, See-Kiong Ng, Tat-Seng Chua, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23763)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance. 

---
# MUStReason: A Benchmark for Diagnosing Pragmatic Reasoning in Video-LMs for Multimodal Sarcasm Detection 

**Authors**: Anisha Saha, Varsha Suresh, Timothy Hospedales, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.23727)  

**Abstract**: Sarcasm is a specific type of irony which involves discerning what is said from what is meant. Detecting sarcasm depends not only on the literal content of an utterance but also on non-verbal cues such as speaker's tonality, facial expressions and conversational context. However, current multimodal models struggle with complex tasks like sarcasm detection, which require identifying relevant cues across modalities and pragmatically reasoning over them to infer the speaker's intention. To explore these limitations in VideoLMs, we introduce MUStReason, a diagnostic benchmark enriched with annotations of modality-specific relevant cues and underlying reasoning steps to identify sarcastic intent. In addition to benchmarking sarcasm classification performance in VideoLMs, using MUStReason we quantitatively and qualitatively evaluate the generated reasoning by disentangling the problem into perception and reasoning, we propose PragCoT, a framework that steers VideoLMs to focus on implied intentions over literal meaning, a property core to detecting sarcasm. 

---
# VisCoder2: Building Multi-Language Visualization Coding Agents 

**Authors**: Yuansheng Ni, Songcheng Cai, Xiangchao Chen, Jiarong Liang, Zhiheng Lyu, Jiaqi Deng, Kai Zou, Ping Nie, Fei Yuan, Xiang Yue, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.23642)  

**Abstract**: Large language models (LLMs) have recently enabled coding agents capable of generating, executing, and revising visualization code. However, existing models often fail in practical workflows due to limited language coverage, unreliable execution, and lack of iterative correction mechanisms. Progress has been constrained by narrow datasets and benchmarks that emphasize single-round generation and single-language tasks. To address these challenges, we introduce three complementary resources for advancing visualization coding agents. VisCode-Multi-679K is a large-scale, supervised dataset containing 679K validated and executable visualization samples with multi-turn correction dialogues across 12 programming languages. VisPlotBench is a benchmark for systematic evaluation, featuring executable tasks, rendered outputs, and protocols for both initial generation and multi-round self-debug. Finally, we present VisCoder2, a family of multi-language visualization models trained on VisCode-Multi-679K. Experiments show that VisCoder2 significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4.1, with further gains from iterative self-debug, reaching 82.4% overall execution pass rate at the 32B scale, particularly in symbolic or compiler-dependent languages. 

---
# Combining Textual and Structural Information for Premise Selection in Lean 

**Authors**: Job Petrovčič, David Eliecer Narvaez Denis, Ljupčo Todorovski  

**Link**: [PDF](https://arxiv.org/pdf/2510.23637)  

**Abstract**: Premise selection is a key bottleneck for scaling theorem proving in large formal libraries. Yet existing language-based methods often treat premises in isolation, ignoring the web of dependencies that connects them. We present a graph-augmented approach that combines dense text embeddings of Lean formalizations with graph neural networks over a heterogeneous dependency graph capturing both state--premise and premise--premise relations. On the LeanDojo Benchmark, our method outperforms the ReProver language-based baseline by over 25% across standard retrieval metrics. These results demonstrate the power of relational information for more effective premise selection. 

---
# Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation 

**Authors**: Thaweerath Phisannupawong, Joshua Julian Damanik, Han-Lim Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23636)  

**Abstract**: Flight delay prediction has become a key focus in air traffic management, as delays highlight inefficiencies that impact overall network performance. This paper presents a lightweight large language model-based multimodal flight delay prediction, formulated from the perspective of air traffic controllers monitoring aircraft delay after entering the terminal area. The approach integrates trajectory representations with textual aeronautical information, including flight information, weather reports, and aerodrome notices, by adapting trajectory data into the language modality to capture airspace conditions. Experimental results show that the model consistently achieves sub-minute prediction error by effectively leveraging contextual information related to the sources of delay. The framework demonstrates that linguistic understanding, when combined with cross-modality adaptation of trajectory information, enhances delay prediction. Moreover, the approach shows practicality and scalability for real-world operations, supporting real-time updates that refine predictions upon receiving new operational information. 

---
# NUM2EVENT: Interpretable Event Reasoning from Numerical time-series 

**Authors**: Ninghui Feng, Yiyan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23630)  

**Abstract**: Large language models (LLMs) have recently demonstrated impressive multimodal reasoning capabilities, yet their understanding of purely numerical time-series signals remains limited. Existing approaches mainly focus on forecasting or trend description, without uncovering the latent events that drive numerical changes or explaining the reasoning process behind them. In this work, we introduce the task of number-to-event reasoning and decoding, which aims to infer interpretable structured events from numerical inputs, even when current text is unavailable. To address the data scarcity and semantic alignment challenges, we propose a reasoning-aware framework that integrates an agent-guided event extractor (AGE), a marked multivariate Hawkes-based synthetic generator (EveDTS), and a two-stage fine-tuning pipeline combining a time-series encoder with a structured decoder. Our model explicitly reasons over numerical changes, generates intermediate explanations, and outputs structured event hypotheses. Experiments on multi-domain datasets show that our method substantially outperforms strong LLM baselines in event-level precision and recall. These results suggest a new direction for bridging quantitative reasoning and semantic understanding, enabling LLMs to explain and predict events directly from numerical dynamics. 

---
# From Detection to Discovery: A Closed-Loop Approach for Simultaneous and Continuous Medical Knowledge Expansion and Depression Detection on Social Media 

**Authors**: Shuang Geng, Wenli Zhang, Jiaheng Xie, Rui Wang, Sudha Ram  

**Link**: [PDF](https://arxiv.org/pdf/2510.23626)  

**Abstract**: Social media user-generated content (UGC) provides real-time, self-reported indicators of mental health conditions such as depression, offering a valuable source for predictive analytics. While prior studies integrate medical knowledge to improve prediction accuracy, they overlook the opportunity to simultaneously expand such knowledge through predictive processes. We develop a Closed-Loop Large Language Model (LLM)-Knowledge Graph framework that integrates prediction and knowledge expansion in an iterative learning cycle. In the knowledge-aware depression detection phase, the LLM jointly performs depression detection and entity extraction, while the knowledge graph represents and weights these entities to refine prediction performance. In the knowledge refinement and expansion phase, new entities, relationships, and entity types extracted by the LLM are incorporated into the knowledge graph under expert supervision, enabling continual knowledge evolution. Using large-scale UGC, the framework enhances both predictive accuracy and medical understanding. Expert evaluations confirmed the discovery of clinically meaningful symptoms, comorbidities, and social triggers complementary to existing literature. We conceptualize and operationalize prediction-through-learning and learning-through-prediction as mutually reinforcing processes, advancing both methodological and theoretical understanding in predictive analytics. The framework demonstrates the co-evolution of computational models and domain knowledge, offering a foundation for adaptive, data-driven knowledge systems applicable to other dynamic risk monitoring contexts. 

---
# An Enhanced Dual Transformer Contrastive Network for Multimodal Sentiment Analysis 

**Authors**: Phuong Q. Dao, Mark Roantree, Vuong M. Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2510.23617)  

**Abstract**: Multimodal Sentiment Analysis (MSA) seeks to understand human emotions by jointly analyzing data from multiple modalities typically text and images offering a richer and more accurate interpretation than unimodal approaches. In this paper, we first propose BERT-ViT-EF, a novel model that combines powerful Transformer-based encoders BERT for textual input and ViT for visual input through an early fusion strategy. This approach facilitates deeper cross-modal interactions and more effective joint representation learning. To further enhance the model's capability, we propose an extension called the Dual Transformer Contrastive Network (DTCN), which builds upon BERT-ViT-EF. DTCN incorporates an additional Transformer encoder layer after BERT to refine textual context (before fusion) and employs contrastive learning to align text and image representations, fostering robust multimodal feature learning. Empirical results on two widely used MSA benchmarks MVSA-Single and TumEmo demonstrate the effectiveness of our approach. DTCN achieves best accuracy (78.4%) and F1-score (78.3%) on TumEmo, and delivers competitive performance on MVSA-Single, with 76.6% accuracy and 75.9% F1-score. These improvements highlight the benefits of early fusion and deeper contextual modeling in Transformer-based multimodal sentiment analysis. 

---
