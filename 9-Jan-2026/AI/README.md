# Learning Latent Action World Models In The Wild 

**Authors**: Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, Michael Rabbat  

**Link**: [PDF](https://arxiv.org/pdf/2601.05230)  

**Abstract**: Agents capable of reasoning and planning in the real world require the ability of predicting the consequences of their actions. While world models possess this capability, they most often require action labels, that can be complex to obtain at scale. This motivates the learning of latent action models, that can learn an action space from videos alone. Our work addresses the problem of learning latent actions world models on in-the-wild videos, expanding the scope of existing works that focus on simple robotics simulations, video games, or manipulation data. While this allows us to capture richer actions, it also introduces challenges stemming from the video diversity, such as environmental noise, or the lack of a common embodiment across videos. To address some of the challenges, we discuss properties that actions should follow as well as relevant architectural choices and evaluations. We find that continuous, but constrained, latent actions are able to capture the complexity of actions from in-the-wild videos, something that the common vector quantization does not. We for example find that changes in the environment coming from agents, such as humans entering the room, can be transferred across videos. This highlights the capability of learning actions that are specific to in-the-wild videos. In the absence of a common embodiment across videos, we are mainly able to learn latent actions that become localized in space, relative to the camera. Nonetheless, we are able to train a controller that maps known actions to latent ones, allowing us to use latent actions as a universal interface and solve planning tasks with our world model with similar performance as action-conditioned baselines. Our analyses and experiments provide a step towards scaling latent action models to the real world. 

---
# MineNPC-Task: Task Suite for Memory-Aware Minecraft Agents 

**Authors**: Tamil Sudaravan Mohan Doss, Michael Xu, Sudha Rao, Andrew D. Wilson, Balasaravanan Thoravi Kumaravel  

**Link**: [PDF](https://arxiv.org/pdf/2601.05215)  

**Abstract**: We present \textsc{MineNPC-Task}, a user-authored benchmark and evaluation harness for testing memory-aware, mixed-initiative LLM agents in open-world \emph{Minecraft}. Rather than relying on synthetic prompts, tasks are elicited from formative and summative co-play with expert players, normalized into parametric templates with explicit preconditions and dependency structure, and paired with machine-checkable validators under a bounded-knowledge policy that forbids out-of-world shortcuts. The harness captures plan/act/memory events-including plan previews, targeted clarifications, memory reads and writes, precondition checks, and repair attempts and reports outcomes relative to the total number of attempted subtasks, derived from in-world evidence.
As an initial snapshot, we instantiate the framework with GPT-4o and evaluate \textbf{216} subtasks across \textbf{8} experienced players. We observe recurring breakdown patterns in code execution, inventory/tool handling, referencing, and navigation, alongside recoveries supported by mixed-initiative clarifications and lightweight memory. Participants rated interaction quality and interface usability positively, while highlighting the need for stronger memory persistence across tasks. We release the complete task suite, validators, logs, and harness to support transparent, reproducible evaluation of future memory-aware embodied agents. 

---
# Internal Representations as Indicators of Hallucinations in Agent Tool Selection 

**Authors**: Kait Healy, Bharathi Srinivasan, Visakh Madathil, Jing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05214)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in tool calling and tool usage, but suffer from hallucinations where they choose incorrect tools, provide malformed parameters and exhibit 'tool bypass' behavior by performing simulations and generating outputs instead of invoking specialized tools or external systems. This undermines the reliability of LLM based agents in production systems as it leads to inconsistent results, and bypasses security and audit controls. Such hallucinations in agent tool selection require early detection and error handling. Unlike existing hallucination detection methods that require multiple forward passes or external validation, we present a computationally efficient framework that detects tool-calling hallucinations in real-time by leveraging LLMs' internal representations during the same forward pass used for generation. We evaluate this approach on reasoning tasks across multiple domains, demonstrating strong detection performance (up to 86.4\% accuracy) while maintaining real-time inference capabilities with minimal computational overhead, particularly excelling at detecting parameter-level hallucinations and inappropriate tool selections, critical for reliable agent deployment. 

---
# Stock Market Price Prediction using Neural Prophet with Deep Neural Network 

**Authors**: Navin Chhibber, Suneel Khemka, Navneet Kumar Tyagi, Rohit Tewari, Bireswar Banerjee, Piyush Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05202)  

**Abstract**: Stock market price prediction is a significant interdisciplinary research domain that depends at the intersection of finance, statistics, and economics. Forecasting Accurately predicting stock prices has always been a focal point for various researchers. However, existing statistical approaches for time-series prediction often fail to effectively forecast the probability range of future stock prices. Hence, to solve this problem, the Neural Prophet with a Deep Neural Network (NP-DNN) is proposed to predict stock market prices. The preprocessing technique used in this research is Z-score normalization, which normalizes stock price data by removing scale differences, making patterns easier to detect. Missing value imputation fills gaps in historical data, enhancing the models use of complete information for more accurate predictions. The Multi-Layer Perceptron (MLP) learns complex nonlinear relationships among stock market prices and extracts hidden patterns from the input data, thereby creating meaningful feature representations for better prediction accuracy. The proposed NP-DNN model achieved an accuracy of 99.21% compared with other approaches using the Fused Large Language Model. Keywords: deep neural network, forecasting stock prices, multi-layer perceptron, neural prophet, stock market price prediction. 

---
# SimuAgent: An LLM-Based Simulink Modeling Assistant Enhanced with Reinforcement Learning 

**Authors**: Yanchang Liang, Xiaowei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05187)  

**Abstract**: Large language models (LLMs) have revolutionized text-based code automation, but their potential in graph-oriented engineering workflows remains under-explored. We introduce SimuAgent, an LLM-powered modeling and simulation agent tailored for Simulink. SimuAgent replaces verbose XML with a concise, dictionary-style Python representation, dramatically cutting token counts, improving interpretability, and enabling fast, in-process simulation. A lightweight plan-execute architecture, trained in two stages, equips the agent with both low-level tool skills and high-level design reasoning. To tackle sparse rewards in long-horizon tasks, we propose Reflection-GRPO (ReGRPO), which augments Group Relative Policy Optimization (GRPO) with self-reflection traces that supply rich intermediate feedback, accelerating convergence and boosting robustness. Experiments on SimuBench, our newly released benchmark comprising 5300 multi-domain modeling tasks, show that a Qwen2.5-7B model fine-tuned with SimuAgent converges faster and achieves higher modeling accuracy than standard RL baselines, and even surpasses GPT-4o when evaluated with few-shot prompting on the same benchmark. Ablations confirm that the two-stage curriculum and abstract-reconstruct data augmentation further enhance generalization. SimuAgent trains and runs entirely on-premise with modest hardware, delivering a privacy-preserving, cost-effective solution for industrial model-driven engineering. SimuAgent bridges the gap between LLMs and graphical modeling environments, offering a practical solution for AI-assisted engineering design in industrial settings. 

---
# Observations and Remedies for Large Language Model Bias in Self-Consuming Performative Loop 

**Authors**: Yaxuan Wang, Zhongteng Cai, Yujia Bao, Xueru Zhang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05184)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to growing interest in using synthetic data to train future models. However, this creates a self-consuming retraining loop, where models are trained on their own outputs and may cause performance drops and induce emerging biases. In real-world applications, previously deployed LLMs may influence the data they generate, leading to a dynamic system driven by user feedback. For example, if a model continues to underserve users from a group, less query data will be collected from this particular demographic of users. In this study, we introduce the concept of \textbf{S}elf-\textbf{C}onsuming \textbf{P}erformative \textbf{L}oop (\textbf{SCPL}) and investigate the role of synthetic data in shaping bias during these dynamic iterative training processes under controlled performative feedback. This controlled setting is motivated by the inaccessibility of real-world user preference data from dynamic production systems, and enables us to isolate and analyze feedback-driven bias evolution in a principled manner. We focus on two types of loops, including the typical retraining setting and the incremental fine-tuning setting, which is largely underexplored. Through experiments on three real-world tasks, we find that the performative loop increases preference bias and decreases disparate bias. We design a reward-based rejection sampling strategy to mitigate the bias, moving towards more trustworthy self-improving systems. 

---
# Distilling the Thought, Watermarking the Answer: A Principle Semantic Guided Watermark for Large Reasoning Models 

**Authors**: Shuliang Liu, Xingyu Li, Hongyi Liu, Yibo Yan, Bingchen Duan, Qi Zheng, Dong Fang, Lingfeng Su, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05144)  

**Abstract**: Reasoning Large Language Models (RLLMs) excelling in complex tasks present unique challenges for digital watermarking, as existing methods often disrupt logical coherence or incur high computational costs. Token-based watermarking techniques can corrupt the reasoning flow by applying pseudo-random biases, while semantic-aware approaches improve quality but introduce significant latency or require auxiliary models. This paper introduces ReasonMark, a novel watermarking framework specifically designed for reasoning-intensive LLMs. Our approach decouples generation into an undisturbed Thinking Phase and a watermarked Answering Phase. We propose a Criticality Score to identify semantically pivotal tokens from the reasoning trace, which are distilled into a Principal Semantic Vector (PSV). The PSV then guides a semantically-adaptive mechanism that modulates watermark strength based on token-PSV alignment, ensuring robustness without compromising logical integrity. Extensive experiments show ReasonMark surpasses state-of-the-art methods by reducing text Perplexity by 0.35, increasing translation BLEU score by 0.164, and raising mathematical accuracy by 0.67 points. These advancements are achieved alongside a 0.34% higher watermark detection AUC and stronger robustness to attacks, all with a negligible increase in latency. This work enables the traceable and trustworthy deployment of reasoning LLMs in real-world applications. 

---
# Evaluative Fingerprints: Stable and Systematic Differences in LLM Evaluator Behavior 

**Authors**: Wajid Nasser  

**Link**: [PDF](https://arxiv.org/pdf/2601.05114)  

**Abstract**: LLM-as-judge systems promise scalable, consistent evaluation. We find the opposite: judges are consistent, but not with each other; they are consistent with themselves. Across 3,240 evaluations (9 judges x 120 unique video x pack items x 3 independent runs), inter-judge agreement is near-zero (Krippendorff's {\alpha} = 0.042). On two dimensions, judges disagree more than random noise would predict ({\alpha} < 0). Yet this disagreement isn't chaos; it's structured. A classifier identifies which judge produced an evaluation with 77.1% accuracy from rubric scores alone, rising to 89.9% with disposition features. Within model families, the signal is even stronger: GPT-4.1 and GPT-5.2 are distinguishable with 99.6% accuracy. We call this the reliability paradox: judges cannot agree on what constitutes quality, yet their disagreement patterns are so stable they function as fingerprints. Each judge implements a distinct, stable theory of quality: an "evaluative disposition" that shapes how it interprets any rubric. We characterize these dispositions along multiple axes: harshness/leniency, dimension emphasis, within-judge stability (ICC), and evidence behavior (receipt validity, semantic linkage via NLI, and shotgun index). The implication is stark: LLM judges are not interchangeable instruments measuring a shared construct. They are distinct measurement devices, each encoding its own implicit theory of quality. Averaging their scores produces a synthetic verdict that corresponds to no judge's actual values. 

---
# GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts 

**Authors**: Wenhao Zeng, Xuteng Zhang, Yuling Shi, Chao Hu, Yuting Chen, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05110)  

**Abstract**: Large Reasoning Models (LRMs) achieve remarkable performance by explicitly generating multi-step chains of thought, but this capability incurs substantial inference latency and computational cost. Collaborative inference offers a promising solution by selectively allocating work between lightweight and large models, yet a fundamental challenge remains: determining when a reasoning step requires the capacity of a large model or the efficiency of a small model. Existing routing strategies either rely on local token probabilities or post-hoc verification, introducing significant inference overhead. In this work, we propose a novel perspective on step-wise collaboration: the difficulty of a reasoning step can be inferred from its very first token. Inspired by the "Aha Moment" phenomenon in LRMs, we show that the entropy of the initial token serves as a strong predictor of step difficulty. Building on this insight, we introduce GlimpRouter, a training-free step-wise collaboration framework. GlimpRouter employs a lightweight model to generate only the first token of each reasoning step and routes the step to a larger model only when the initial token entropy exceeds a threshold. Experiments on multiple benchmarks demonstrate that our approach significantly reduces inference latency while preserving accuracy. For instance, GlimpRouter attains a substantial 10.7% improvement in accuracy while reducing inference latency by 25.9% compared to a standalone large model on AIME25. These results suggest a simple yet effective mechanism for reasoning: allocating computation based on a glimpse of thought rather than full-step evaluation. 

---
# Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction 

**Authors**: Muzhao Tian, Zisu Huang, Xiaohua Wang, Jingwen Xu, Zhengkang Guo, Qi Qian, Yuanzhe Shen, Kaitao Song, Jiakang Yuan, Changze Lv, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05107)  

**Abstract**: As LLM-based agents are increasingly used in long-term interactions, cumulative memory is critical for enabling personalization and maintaining stylistic consistency. However, most existing systems adopt an ``all-or-nothing'' approach to memory usage: incorporating all relevant past information can lead to \textit{Memory Anchoring}, where the agent is trapped by past interactions, while excluding memory entirely results in under-utilization and the loss of important interaction history. We show that an agent's reliance on memory can be modeled as an explicit and user-controllable dimension. We first introduce a behavioral metric of memory dependence to quantify the influence of past interactions on current outputs. We then propose \textbf{Stee}rable \textbf{M}emory Agent, \texttt{SteeM}, a framework that allows users to dynamically regulate memory reliance, ranging from a fresh-start mode that promotes innovation to a high-fidelity mode that closely follows interaction history. Experiments across different scenarios demonstrate that our approach consistently outperforms conventional prompting and rigid memory masking strategies, yielding a more nuanced and effective control for personalized human-agent collaboration. 

---
# Token-Level LLM Collaboration via FusionRoute 

**Authors**: Nuoya Xiong, Yuhang Zhou, Hanqing Zeng, Zhaorun Chen, Furong Huang, Shuchao Bi, Lizhu Zhang, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05106)  

**Abstract**: Large language models (LLMs) exhibit strengths across diverse domains. However, achieving strong performance across these domains with a single general-purpose model typically requires scaling to sizes that are prohibitively expensive to train and deploy. On the other hand, while smaller domain-specialized models are much more efficient, they struggle to generalize beyond their training distributions. To address this dilemma, we propose FusionRoute, a robust and effective token-level multi-LLM collaboration framework in which a lightweight router simultaneously (i) selects the most suitable expert at each decoding step and (ii) contributes a complementary logit that refines or corrects the selected expert's next-token distribution via logit addition. Unlike existing token-level collaboration methods that rely solely on fixed expert outputs, we provide a theoretical analysis showing that pure expert-only routing is fundamentally limited: unless strong global coverage assumptions hold, it cannot in general realize the optimal decoding policy. By augmenting expert selection with a trainable complementary generator, FusionRoute expands the effective policy class and enables recovery of optimal value functions under mild conditions. Empirically, across both Llama-3 and Gemma-2 families and diverse benchmarks spanning mathematical reasoning, code generation, and instruction following, FusionRoute outperforms both sequence- and token-level collaboration, model merging, and direct fine-tuning, while remaining competitive with domain experts on their respective tasks. 

---
# Arabic Prompts with English Tools: A Benchmark 

**Authors**: Konstantin Kubrak, Ahmed El-Moselhy, Ammar Alsulami, Remaz Altuwaim, Hassan Ismail Fawaz, Faisal Alsaby  

**Link**: [PDF](https://arxiv.org/pdf/2601.05101)  

**Abstract**: Large Language Models (LLMs) are now integral to numerous industries, increasingly serving as the core reasoning engine for autonomous agents that perform complex tasks through tool-use. While the development of Arabic-native LLMs is accelerating, the benchmarks for evaluating their capabilities lag behind, with most existing frameworks focusing on English. A critical and overlooked area is tool-calling, where the performance of models prompted in non-English languages like Arabic is poorly understood, especially since these models are often pretrained on predominantly English data. This paper addresses this critical gap by introducing the first dedicated benchmark for evaluating the tool-calling and agentic capabilities of LLMs in the Arabic language. Our work provides a standardized framework to measure the functional accuracy and robustness of models in Arabic agentic workflows. Our findings reveal a huge performance gap: when users interact in Arabic, tool-calling accuracy drops by an average of 5-10\%, regardless of whether the tool descriptions themselves are in Arabic or English. By shedding light on these critical challenges, this benchmark aims to foster the development of more reliable and linguistically equitable AI agents for Arabic-speaking users. 

---
# Chain-of-Sanitized-Thoughts: Plugging PII Leakage in CoT of Large Reasoning Models 

**Authors**: Arghyadeep Das, Sai Sreenivas Chintha, Rishiraj Girmal, Kinjal Pandey, Sharvi Endait  

**Link**: [PDF](https://arxiv.org/pdf/2601.05076)  

**Abstract**: Large Reasoning Models (LRMs) improve performance, reliability, and interpretability by generating explicit chain-of-thought (CoT) reasoning, but this transparency introduces a serious privacy risk: intermediate reasoning often leaks personally identifiable information (PII) even when final answers are sanitized. We study how to induce privacy-first reasoning, where models reason without exposing sensitive information, using deployable interventions rather than post-hoc redaction. We introduce PII-CoT-Bench, a supervised dataset with privacy-aware CoT annotations, and a category-balanced evaluation benchmark covering realistic and adversarial leakage scenarios. Our results reveal a capability-dependent trend: state-of-the-art models benefit most from prompt-based controls, whereas weaker models require fine-tuning to achieve meaningful leakage reduction. Across models and categories, both approaches substantially reduce PII exposure with minimal degradation in utility, demonstrating that private reasoning can be achieved without sacrificing performance. Overall, we show that private CoT reasoning can be achieved with minimal utility loss, providing practical guidance for building privacy-preserving reasoning systems. 

---
# Reinforced Efficient Reasoning via Semantically Diverse Exploration 

**Authors**: Ziqi Zhao, Zhaochun Ren, Jiahong Zou, Liu Yang, Zhiwei Xu, Xuri Ge, Zhumin Chen, Xinyu Ma, Daiting Shi, Shuaiqiang Wang, Dawei Yin, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05053)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has proven effective in enhancing the reasoning of large language models (LLMs). Monte Carlo Tree Search (MCTS)-based extensions improve upon vanilla RLVR (e.g., GRPO) by providing tree-based reasoning rollouts that enable fine-grained and segment-level credit assignment. However, existing methods still suffer from limited exploration diversity and inefficient reasoning. To address the above challenges, we propose reinforced efficient reasoning via semantically diverse explorations, i.e., ROSE, for LLMs. To encourage more diverse reasoning exploration, our method incorporates a semantic-entropy-based branching strategy and an $\varepsilon$-exploration mechanism. The former operates on already sampled reasoning rollouts to capture semantic uncertainty and select branching points with high semantic divergence to generate new successive reasoning paths, whereas the latter stochastically initiates reasoning rollouts from the root, preventing the search process from becoming overly local. To improve efficiency, we design a length-aware segment-level advantage estimator that rewards concise and correct reasoning while penalizing unnecessarily long reasoning chains. Extensive experiments on various mathematical reasoning benchmarks with Qwen and Llama models validate the effectiveness and efficiency of ROSE. Codes are available at this https URL. 

---
# Publishing FAIR and Machine-actionable Reviews in Materials Science: The Case for Symbolic Knowledge in Neuro-symbolic Artificial Intelligence 

**Authors**: Jennifer D'Souza, Soren Auer, Eleni Poupaki, Alex Watkins, Anjana Devi, Riikka L. Puurunen, Bora Karasulu, Adrie Mackus, Erwin Kessels  

**Link**: [PDF](https://arxiv.org/pdf/2601.05051)  

**Abstract**: Scientific reviews are central to knowledge integration in materials science, yet their key insights remain locked in narrative text and static PDF tables, limiting reuse by humans and machines alike. This article presents a case study in atomic layer deposition and etching (ALD/E) where we publish review tables as FAIR, machine-actionable comparisons in the Open Research Knowledge Graph (ORKG), turning them into structured, queryable knowledge. Building on this, we contrast symbolic querying over ORKG with large language model-based querying, and argue that a curated symbolic layer should remain the backbone of reliable neurosymbolic AI in materials science, with LLMs serving as complementary, symbolically grounded interfaces rather than standalone sources of truth. 

---
# Large language models can effectively convince people to believe conspiracies 

**Authors**: Thomas H. Costello, Kellin Pelrine, Matthew Kowal, Antonio A. Arechar, Jean-François Godbout, Adam Gleave, David Rand, Gordon Pennycook  

**Link**: [PDF](https://arxiv.org/pdf/2601.05050)  

**Abstract**: Large language models (LLMs) have been shown to be persuasive across a variety of context. But it remains unclear whether this persuasive power advantages truth over falsehood, or if LLMs can promote misbeliefs just as easily as refuting them. Here, we investigate this question across three pre-registered experiments in which participants (N = 2,724 Americans) discussed a conspiracy theory they were uncertain about with GPT-4o, and the model was instructed to either argue against ("debunking") or for ("bunking") that conspiracy. When using a "jailbroken" GPT-4o variant with guardrails removed, the AI was as effective at increasing conspiracy belief as decreasing it. Concerningly, the bunking AI was rated more positively, and increased trust in AI, more than the debunking AI. Surprisingly, we found that using standard GPT-4o produced very similar effects, such that the guardrails imposed by OpenAI did little to revent the LLM from promoting conspiracy beliefs. Encouragingly, however, a corrective conversation reversed these newly induced conspiracy beliefs, and simply prompting GPT-4o to only use accurate information dramatically reduced its ability to increase conspiracy beliefs. Our findings demonstrate that LLMs possess potent abilities to promote both truth and falsehood, but that potential solutions may exist to help mitigate this risk. 

---
# How to Set the Learning Rate for Large-Scale Pre-training? 

**Authors**: Yunhua Zhou, Shuhao Xing, Junhao Huang, Xipeng Qiu, Qipeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.05049)  

**Abstract**: Optimal configuration of the learning rate (LR) is a fundamental yet formidable challenge in large-scale pre-training. Given the stringent trade-off between training costs and model performance, the pivotal question is whether the optimal LR can be accurately extrapolated from low-cost experiments. In this paper, we formalize this investigation into two distinct research paradigms: Fitting and Transfer. Within the Fitting Paradigm, we innovatively introduce a Scaling Law for search factor, effectively reducing the search complexity from O(n^3) to O(n*C_D*C_{\eta}) via predictive modeling. Within the Transfer Paradigm, we extend the principles of $\mu$Transfer to the Mixture of Experts (MoE) architecture, broadening its applicability to encompass model depth, weight decay, and token horizons. By pushing the boundaries of existing hyperparameter research in terms of scale, we conduct a comprehensive comparison between these two paradigms. Our empirical results challenge the scalability of the widely adopted $\mu$ Transfer in large-scale pre-training scenarios. Furthermore, we provide a rigorous analysis through the dual lenses of training stability and feature learning to elucidate the underlying reasons why module-wise parameter tuning underperforms in large-scale settings. This work offers systematic practical guidelines and a fresh theoretical perspective for optimizing industrial-level pre-training. 

---
# How to Set the Batch Size for Large-Scale Pre-training? 

**Authors**: Yunhua Zhou, Junhao Huang, Shuhao Xin, Yechen Zhang, Runyu Peng, Qiping Guo, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05034)  

**Abstract**: The concept of Critical Batch Size, as pioneered by OpenAI, has long served as a foundational principle for large-scale pre-training. However, with the paradigm shift towards the Warmup-Stable-Decay (WSD) learning rate scheduler, we observe that the original theoretical framework and its underlying mechanisms fail to align with new pre-training dynamics. To bridge this gap between theory and practice, this paper derives a revised E(S) relationship tailored for WSD scheduler, characterizing the trade-off between training data consumption E and steps S during pre-training. Our theoretical analysis reveals two fundamental properties of WSD-based pre-training: 1) B_min, the minimum batch size threshold required to achieve a target loss, and 2) B_opt, the optimal batch size that maximizes data efficiency by minimizing total tokens. Building upon these properties, we propose a dynamic Batch Size Scheduler. Extensive experiments demonstrate that our revised formula precisely captures the dynamics of large-scale pre-training, and the resulting scheduling strategy significantly enhances both training efficiency and final model quality. 

---
# OptiSet: Unified Optimizing Set Selection and Ranking for Retrieval-Augmented Generation 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Bairui Hu, Yanrui Du, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05027)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves generation quality by incorporating evidence retrieved from large external corpora. However, most existing methods rely on statically selecting top-k passages based on individual relevance, which fails to exploit combinatorial gains among passages and often introduces substantial redundancy. To address this limitation, we propose OptiSet, a set-centric framework that unifies set selection and set-level ranking for RAG. OptiSet adopts an "Expand-then-Refine" paradigm: it first expands a query into multiple perspectives to enable a diverse candidate pool and then refines the candidate pool via re-selection to form a compact evidence set. We then devise a self-synthesis strategy without strong LLM supervision to derive preference labels from the set conditional utility changes of the generator, thereby identifying complementary and redundant evidence. Finally, we introduce a set-list wise training strategy that jointly optimizes set selection and set-level ranking, enabling the model to favor compact, high-gain evidence sets. Extensive experiments demonstrate that OptiSet improves performance on complex combinatorial problems and makes generation more efficient. The source code is publicly available. 

---
# An Empirical Investigation of Robustness in Large Language Models under Tabular Distortions 

**Authors**: Avik Dutta, Harshit Nigam, Hosein Hasanbeig, Arjun Radhakrishna, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2601.05009)  

**Abstract**: We investigate how large language models (LLMs) fail when tabular data in an otherwise canonical representation is subjected to semantic and structural distortions. Our findings reveal that LLMs lack an inherent ability to detect and correct subtle distortions in table representations. Only when provided with an explicit prior, via a system prompt, do models partially adjust their reasoning strategies and correct some distortions, though not consistently or completely. To study this phenomenon, we introduce a small, expert-curated dataset that explicitly evaluates LLMs on table question answering (TQA) tasks requiring an additional error-correction step prior to analysis. Our results reveal systematic differences in how LLMs ingest and interpret tabular information under distortion, with even SoTA models such as GPT-5.2 model exhibiting a drop of minimum 22% accuracy under distortion. These findings raise important questions for future research, particularly regarding when and how models should autonomously decide to realign tabular inputs, analogous to human behavior, without relying on explicit prompts or tabular data pre-processing. 

---
# AlgBench: To What Extent Do Large Reasoning Models Understand Algorithms? 

**Authors**: Henan Sun, Kaichi Yu, Yuyao Wang, Bowen Liu, Xunkai Li, Rong-Hua Li, Nuo Chen, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04996)  

**Abstract**: Reasoning ability has become a central focus in the advancement of Large Reasoning Models (LRMs). Although notable progress has been achieved on several reasoning benchmarks such as MATH500 and LiveCodeBench, existing benchmarks for algorithmic reasoning remain limited, failing to answer a critical question: Do LRMs truly master algorithmic reasoning? To answer this question, we propose AlgBench, an expert-curated benchmark that evaluates LRMs under an algorithm-centric paradigm.
AlgBench consists of over 3,000 original problems spanning 27 algorithms, constructed by ACM algorithmic experts and organized under a comprehensive taxonomy, including Euclidean-structured, non-Euclidean-structured, non-optimized, local-optimized, global-optimized, and heuristic-optimized categories. Empirical evaluations on leading LRMs (e.g., Gemini-3-Pro, DeepSeek-v3.2-Speciale and GPT-o3) reveal substantial performance heterogeneity: while models perform well on non-optimized tasks (up to 92%), accuracy drops sharply to around 49% on globally optimized algorithms such as dynamic programming. Further analysis uncovers \textbf{strategic over-shifts}, wherein models prematurely abandon correct algorithmic designs due to necessary low-entropy tokens. These findings expose fundamental limitations of problem-centric reinforcement learning and highlight the necessity of an algorithm-centric training paradigm for robust algorithmic reasoning. 

---
# ConMax: Confidence-Maximizing Compression for Efficient Chain-of-Thought Reasoning 

**Authors**: Minda Hu, Zexuan Qiu, Zenan Xu, Kun Li, Bo Zhou, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2601.04973)  

**Abstract**: Recent breakthroughs in Large Reasoning Models (LRMs) have demonstrated that extensive Chain-of-Thought (CoT) generation is critical for enabling intricate cognitive behaviors, such as self-verification and backtracking, to solve complex tasks. However, this capability often leads to ``overthinking'', where models generate redundant reasoning paths that inflate computational costs without improving accuracy. While Supervised Fine-Tuning (SFT) on reasoning traces is a standard paradigm for the 'cold start' phase, applying existing compression techniques to these traces often compromises logical coherence or incurs prohibitive sampling costs. In this paper, we introduce ConMax (Confidence-Maximizing Compression), a novel reinforcement learning framework designed to automatically compress reasoning traces while preserving essential reasoning patterns. ConMax formulates compression as a reward-driven optimization problem, training a policy to prune redundancy by maximizing a weighted combination of answer confidence for predictive fidelity and thinking confidence for reasoning validity through a frozen auxiliary LRM. Extensive experiments across five reasoning datasets demonstrate that ConMax achieves a superior efficiency-performance trade-off. Specifically, it reduces inference length by 43% over strong baselines at the cost of a mere 0.7% dip in accuracy, proving its effectiveness in generating high-quality, efficient training data for LRMs. 

---
# T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual Graphs 

**Authors**: Chunyu Wei, Huaiyu Qin, Siyuan He, Yunhai Wang, Yueguo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04945)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced Large Language Models' ability to access external knowledge, yet current graph-based RAG approaches face two critical limitations in managing hierarchical information: they impose rigid layer-specific compression quotas that damage local graph structures, and they prioritize topological structure while neglecting semantic content. We introduce T-Retriever, a novel framework that reformulates attributed graph retrieval as tree-based retrieval using a semantic and structure-guided encoding tree. Our approach features two key innovations: (1) Adaptive Compression Encoding, which replaces artificial compression quotas with a global optimization strategy that preserves the graph's natural hierarchical organization, and (2) Semantic-Structural Entropy ($S^2$-Entropy), which jointly optimizes for both structural cohesion and semantic consistency when creating hierarchical partitions. Experiments across diverse graph reasoning benchmarks demonstrate that T-Retriever significantly outperforms state-of-the-art RAG methods, providing more coherent and contextually relevant responses to complex queries. 

---
# Conversational AI for Rapid Scientific Prototyping: A Case Study on ESA's ELOPE Competition 

**Authors**: Nils Einecke  

**Link**: [PDF](https://arxiv.org/pdf/2601.04920)  

**Abstract**: Large language models (LLMs) are increasingly used as coding partners, yet their role in accelerating scientific discovery remains underexplored. This paper presents a case study of using ChatGPT for rapid prototyping in ESA's ELOPE (Event-based Lunar OPtical flow Egomotion estimation) competition. The competition required participants to process event camera data to estimate lunar lander trajectories. Despite joining late, we achieved second place with a score of 0.01282, highlighting the potential of human-AI collaboration in competitive scientific settings. ChatGPT contributed not only executable code but also algorithmic reasoning, data handling routines, and methodological suggestions, such as using fixed number of events instead of fixed time spans for windowing. At the same time, we observed limitations: the model often introduced unnecessary structural changes, gets confused by intermediate discussions about alternative ideas, occasionally produced critical errors and forgets important aspects in longer scientific discussions. By analyzing these strengths and shortcomings, we show how conversational AI can both accelerate development and support conceptual insight in scientific research. We argue that structured integration of LLMs into the scientific workflow can enhance rapid prototyping by proposing best practices for AI-assisted scientific work. 

---
# What Students Ask, How a Generative AI Assistant Responds: Exploring Higher Education Students' Dialogues on Learning Analytics Feedback 

**Authors**: Yildiz Uzun, Andrea Gauthier, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2601.04919)  

**Abstract**: Learning analytics dashboards (LADs) aim to support students' regulation of learning by translating complex data into feedback. Yet students, especially those with lower self-regulated learning (SRL) competence, often struggle to engage with and interpret analytics feedback. Conversational generative artificial intelligence (GenAI) assistants have shown potential to scaffold this process through real-time, personalised, dialogue-based support. Further advancing this potential, we explored authentic dialogues between students and GenAI assistant integrated into LAD during a 10-week semester. The analysis focused on questions students with different SRL levels posed, the relevance and quality of the assistant's answers, and how students perceived the assistant's role in their learning. Findings revealed distinct query patterns. While low SRL students sought clarification and reassurance, high SRL students queried technical aspects and requested personalised strategies. The assistant provided clear and reliable explanations but limited in personalisation, handling emotionally charged queries, and integrating multiple data points for tailored responses. Findings further extend that GenAI interventions can be especially valuable for low SRL students, offering scaffolding that supports engagement with feedback and narrows gaps with their higher SRL peers. At the same time, students' reflections underscored the importance of trust, need for greater adaptivity, context-awareness, and technical refinement in future systems. 

---
# From Stories to Cities to Games: A Qualitative Evaluation of Behaviour Planning 

**Authors**: Mustafa F. Abdelwahed, Joan Espasa, Alice Toniolo, Ian P. Gent  

**Link**: [PDF](https://arxiv.org/pdf/2601.04911)  

**Abstract**: The primary objective of a diverse planning approach is to generate a set of plans that are distinct from one another. Such an approach is applied in a variety of real-world domains, including risk management, automated stream data analysis, and malware detection. More recently, a novel diverse planning paradigm, referred to as behaviour planning, has been proposed. This approach extends earlier methods by explicitly incorporating a diversity model into the planning process and supporting multiple planning categories. In this paper, we demonstrate the usefulness of behaviour planning in real-world settings by presenting three case studies. The first case study focuses on storytelling, the second addresses urban planning, and the third examines game evaluation. 

---
# DVD: A Robust Method for Detecting Variant Contamination in Large Language Model Evaluation 

**Authors**: Renzhao Liang, Jingru Chen, Bo Jia, Bo Deng, Chenggang Xie, Yidong Wang, Ke Jin, Xin Wang, Linfeng Zhang, Cunxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04895)  

**Abstract**: Evaluating large language models (LLMs) is increasingly confounded by \emph{variant contamination}: the training corpus contains semantically equivalent yet lexically or syntactically altered versions of test items. Unlike verbatim leakage, these paraphrased or structurally transformed variants evade existing detectors based on sampling consistency or perplexity, thereby inflating benchmark scores via memorization rather than genuine reasoning. We formalize this problem and introduce \textbf{DVD} (\textbf{D}etection via \textbf{V}ariance of generation \textbf{D}istribution), a single-sample detector that models the local output distribution induced by temperature sampling. Our key insight is that contaminated items trigger alternation between a \emph{memory-adherence} state and a \emph{perturbation-drift} state, yielding abnormally high variance in the synthetic difficulty of low-probability tokens; uncontaminated items remain in drift with comparatively smooth variance. We construct the first benchmark for variant contamination across two domains Omni-MATH and SuperGPQA by generating and filtering semantically equivalent variants, and simulate contamination via fine-tuning models of different scales and architectures (Qwen2.5 and Llama3.1). Across datasets and models, \textbf{DVD} consistently outperforms perplexity-based, Min-$k$\%++, edit-distance (CDD), and embedding-similarity baselines, while exhibiting strong robustness to hyperparameters. Our results establish variance of the generation distribution as a principled and practical fingerprint for detecting variant contamination in LLM evaluation. 

---
# SmartSearch: Process Reward-Guided Query Refinement for Search Agents 

**Authors**: Tongyu Wen, Guanting Dong, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04888)  

**Abstract**: Large language model (LLM)-based search agents have proven promising for addressing knowledge-intensive problems by incorporating information retrieval capabilities. Existing works largely focus on optimizing the reasoning paradigms of search agents, yet the quality of intermediate search queries during reasoning remains overlooked. As a result, the generated queries often remain inaccurate, leading to unexpected retrieval results and ultimately limiting search agents' overall effectiveness. To mitigate this issue, we introduce SmartSearch, a framework built upon two key mechanisms: (1) Process rewards, which provide fine-grained supervision for the quality of each intermediate search query through Dual-Level Credit Assessment. (2) Query refinement, which promotes the optimization of query generation by selectively refining low-quality search queries and regenerating subsequent search rounds based on these refinements. To enable the search agent to progressively internalize the ability to improve query quality under the guidance of process rewards, we design a three-stage curriculum learning framework. This framework guides the agent through a progression from imitation, to alignment, and ultimately to generalization. Experimental results show that SmartSearch consistently surpasses existing baselines, and additional quantitative analyses further confirm its significant gains in both search efficiency and query quality. The code is available at this https URL. 

---
# Flexible Manufacturing Systems Intralogistics: Dynamic Optimization of AGVs and Tool Sharing Using Coloured-Timed Petri Nets and Actor-Critic RL with Actions Masking 

**Authors**: Sofiene Lassoued, Laxmikant Shrikant Bahetic, Nathalie Weiß-Borkowskib, Stefan Lierc, Andreas Schwunga  

**Link**: [PDF](https://arxiv.org/pdf/2601.04887)  

**Abstract**: Flexible Manufacturing Systems (FMS) are pivotal in optimizing production processes in today's rapidly evolving manufacturing landscape. This paper advances the traditional job shop scheduling problem by incorporating additional complexities through the simultaneous integration of automated guided vehicles (AGVs) and tool-sharing systems. We propose a novel approach that combines Colored-Timed Petri Nets (CTPNs) with actor-critic model-based reinforcement learning (MBRL), effectively addressing the multifaceted challenges associated with FMS. CTPNs provide a formal modeling structure and dynamic action masking, significantly reducing the action search space, while MBRL ensures adaptability to changing environments through the learned policy. Leveraging the advantages of MBRL, we incorporate a lookahead strategy for optimal positioning of AGVs, improving operational efficiency. Our approach was evaluated on small-sized public benchmarks and a newly developed large-scale benchmark inspired by the Taillard benchmark. The results show that our approach matches traditional methods on smaller instances and outperforms them on larger ones in terms of makespan while achieving a tenfold reduction in computation time. To ensure reproducibility, we propose a gym-compatible environment and an instance generator. Additionally, an ablation study evaluates the contribution of each framework component to its overall performance. 

---
# Precomputing Multi-Agent Path Replanning using Temporal Flexibility: A Case Study on the Dutch Railway Network 

**Authors**: Issa Hanou, Eric Kemmeren, Devin Wild Thomas, Mathijs de Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2601.04884)  

**Abstract**: Executing a multi-agent plan can be challenging when an agent is delayed, because this typically creates conflicts with other agents. So, we need to quickly find a new safe plan. Replanning only the delayed agent often does not result in an efficient plan, and sometimes cannot even yield a feasible plan. On the other hand, replanning other agents may lead to a cascade of changes and delays. We show how to efficiently replan by tracking and using the temporal flexibility of other agents while avoiding cascading delays. This flexibility is the maximum delay an agent can take without changing the order of or further delaying more agents. Our algorithm, FlexSIPP, precomputes all possible plans for the delayed agent, also returning the changes for the other agents, for any single-agent delay within the given scenario. We demonstrate our method in a real-world case study of replanning trains in the densely-used Dutch railway network. Our experiments show that FlexSIPP provides effective solutions, relevant to real-world adjustments, and within a reasonable timeframe. 

---
# Higher-Order Knowledge Representations for Agentic Scientific Reasoning 

**Authors**: Isabella A. Stewart, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2601.04878)  

**Abstract**: Scientific inquiry requires systems-level reasoning that integrates heterogeneous experimental data, cross-domain knowledge, and mechanistic evidence into coherent explanations. While Large Language Models (LLMs) offer inferential capabilities, they often depend on retrieval-augmented contexts that lack structural depth. Traditional Knowledge Graphs (KGs) attempt to bridge this gap, yet their pairwise constraints fail to capture the irreducible higher-order interactions that govern emergent physical behavior. To address this, we introduce a methodology for constructing hypergraph-based knowledge representations that faithfully encode multi-entity relationships. Applied to a corpus of ~1,100 manuscripts on biocomposite scaffolds, our framework constructs a global hypergraph of 161,172 nodes and 320,201 hyperedges, revealing a scale-free topology (power law exponent ~1.23) organized around highly connected conceptual hubs. This representation prevents the combinatorial explosion typical of pairwise expansions and explicitly preserves the co-occurrence context of scientific formulations. We further demonstrate that equipping agentic systems with hypergraph traversal tools, specifically using node-intersection constraints, enables them to bridge semantically distant concepts. By exploiting these higher-order pathways, the system successfully generates grounded mechanistic hypotheses for novel composite materials, such as linking cerium oxide to PCL scaffolds via chitosan intermediates. This work establishes a "teacherless" agentic reasoning system where hypergraph topology acts as a verifiable guardrail, accelerating scientific discovery by uncovering relationships obscured by traditional graph methods. 

---
# Key-Value Pair-Free Continual Learner via Task-Specific Prompt-Prototype 

**Authors**: Haihua Luo, Xuming Ran, Zhengji Li, Huiyan Xue, Tingting Jiang, Jiangrong Shen, Tommi Kärkkäinen, Qi Xu, Fengyu Cong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04864)  

**Abstract**: Continual learning aims to enable models to acquire new knowledge while retaining previously learned information. Prompt-based methods have shown remarkable performance in this domain; however, they typically rely on key-value pairing, which can introduce inter-task interference and hinder scalability. To overcome these limitations, we propose a novel approach employing task-specific Prompt-Prototype (ProP), thereby eliminating the need for key-value pairs. In our method, task-specific prompts facilitate more effective feature learning for the current task, while corresponding prototypes capture the representative features of the input. During inference, predictions are generated by binding each task-specific prompt with its associated prototype. Additionally, we introduce regularization constraints during prompt initialization to penalize excessively large values, thereby enhancing stability. Experiments on several widely used datasets demonstrate the effectiveness of the proposed method. In contrast to mainstream prompt-based approaches, our framework removes the dependency on key-value pairs, offering a fresh perspective for future continual learning research. 

---
# Orchestrating Intelligence: Confidence-Aware Routing for Efficient Multi-Agent Collaboration across Multi-Scale Models 

**Authors**: Jingbo Wang, Sendong Zhao, Jiatong Liu, Haochun Wang, Wanting Li, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04861)  

**Abstract**: While multi-agent systems (MAS) have demonstrated superior performance over single-agent approaches in complex reasoning tasks, they often suffer from significant computational inefficiencies. Existing frameworks typically deploy large language models (LLMs) uniformly across all agent roles, failing to account for the varying cognitive demands of different reasoning stages. We address this inefficiency by proposing OI-MAS framework, a novel multi-agent framework that implements an adaptive model-selection policy across a heterogeneous pool of multi-scale LLMs. Specifically, OI-MAS introduces a state-dependent routing mechanism that dynamically selects agent roles and model scales throughout the reasoning process. In addition, we introduce a confidence-aware mechanism that selects appropriate model scales conditioned on task complexity, thus reducing unnecessary reliance on large-scale models. Experimental results show that OI-MAS consistently outperforms baseline multi-agent systems, improving accuracy by up to 12.88\% while reducing cost by up to 79.78\%. 

---
# DR-LoRA: Dynamic Rank LoRA for Mixture-of-Experts Adaptation 

**Authors**: Guanzhi Deng, Bo Li, Ronghao Chen, Huacan Wang, Linqi Song, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04823)  

**Abstract**: Mixture-of-Experts (MoE) has become a prominent paradigm for scaling Large Language Models (LLMs). Parameter-efficient fine-tuning (PEFT), such as LoRA, is widely adopted to adapt pretrained MoE LLMs to downstream tasks. However, existing approaches assign identical LoRA ranks to all experts, overlooking the intrinsic functional specialization within MoE LLMs. This uniform allocation leads to resource mismatch, task-relevant experts are under-provisioned while less relevant ones receive redundant parameters. We propose a Dynamic Rank LoRA framework named DR-LoRA, which dynamically grows expert LoRA ranks during fine-tuning based on task-specific demands. DR-LoRA employs an Expert Saliency Scoring mechanism that integrates expert routing frequency and LoRA rank importance to quantify each expert's demand for additional capacity. Experts with higher saliency scores are prioritized for rank expansion, enabling the automatic formation of a heterogeneous rank distribution tailored to the target task. Experiments on multiple benchmarks demonstrate that DR-LoRA consistently outperforms standard LoRA and static allocation strategies under the same parameter budget, achieving superior task performance with more efficient parameter utilization. 

---
# AECV-Bench: Benchmarking Multimodal Models on Architectural and Engineering Drawings Understanding 

**Authors**: Aleksei Kondratenko, Mussie Birhane, Houssame E. Hsain, Guido Maciocci  

**Link**: [PDF](https://arxiv.org/pdf/2601.04819)  

**Abstract**: AEC drawings encode geometry and semantics through symbols, layout conventions, and dense annotation, yet it remains unclear whether modern multimodal and vision-language models can reliably interpret this graphical language. We present AECV-Bench, a benchmark for evaluating multimodal and vision-language models on realistic AEC artefacts via two complementary use cases: (i) object counting on 120 high-quality floor plans (doors, windows, bedrooms, toilets), and (ii) drawing-grounded document QA spanning 192 question-answer pairs that test text extraction (OCR), instance counting, spatial reasoning, and comparative reasoning over common drawing regions. Object-counting performance is reported using per-field exact-match accuracy and MAPE results, while document-QA performance is reported using overall accuracy and per-category breakdowns with an LLM-as-a-judge scoring pipeline and targeted human adjudication for edge cases. Evaluating a broad set of state-of-the-art models under a unified protocol, we observe a stable capability gradient; OCR and text-centric document QA are strongest (up to 0.95 accuracy), spatial reasoning is moderate, and symbol-centric drawing understanding - especially reliable counting of doors and windows - remains unsolved (often 0.40-0.55 accuracy) with substantial proportional errors. These results suggest that current systems function well as document assistants but lack robust drawing literacy, motivating domain-specific representations and tool-augmented, human-in-the-loop workflows for an efficient AEC automation. 

---
# SCALER:Synthetic Scalable Adaptive Learning Environment for Reasoning 

**Authors**: Caijun Xu, Changyi Xiao, Zhongyuan Peng, Xinrun Wang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04809)  

**Abstract**: Reinforcement learning (RL) offers a principled way to enhance the reasoning capabilities of large language models, yet its effectiveness hinges on training signals that remain informative as models evolve. In practice, RL progress often slows when task difficulty becomes poorly aligned with model capability, or when training is dominated by a narrow set of recurring problem patterns. To jointly address these issues, we propose SCALER (Synthetic sCalable Adaptive Learning Environment for Reasoning), a framework that sustains effective learning signals through adaptive environment design. SCALER introduces a scalable synthesis pipeline that converts real-world programming problems into verifiable reasoning environments with controllable difficulty and unbounded instance generation, enabling RL training beyond finite datasets while preserving strong correctness guarantees. Building on this, SCALER further employs an adaptive multi-environment RL strategy that dynamically adjusts instance difficulty and curates the active set of environments to track the model's capability frontier and maintain distributional diversity. This co-adaptation prevents reward sparsity, mitigates overfitting to narrow task patterns, and supports sustained improvement throughout training. Extensive experiments show that SCALER consistently outperforms dataset-based RL baselines across diverse reasoning benchmarks and exhibits more stable, long-horizon training dynamics. 

---
# Thinking-Based Non-Thinking: Solving the Reward Hacking Problem in Training Hybrid Reasoning Models via Reinforcement Learning 

**Authors**: Siyuan Gan, Jiaheng Liu, Boyan Wang, Tianpei Yang, Runqing Miao, Yuyao Zhang, Fanyu Meng, Junlan Feng, Linjian Meng, Jing Huo, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04805)  

**Abstract**: Large reasoning models (LRMs) have attracted much attention due to their exceptional performance. However, their performance mainly stems from thinking, a long Chain of Thought (CoT), which significantly increase computational overhead. To address this overthinking problem, existing work focuses on using reinforcement learning (RL) to train hybrid reasoning models that automatically decide whether to engage in thinking or not based on the complexity of the query. Unfortunately, using RL will suffer the the reward hacking problem, e.g., the model engages in thinking but is judged as not doing so, resulting in incorrect rewards. To mitigate this problem, existing works either employ supervised fine-tuning (SFT), which incurs high computational costs, or enforce uniform token limits on non-thinking responses, which yields limited mitigation of the problem. In this paper, we propose Thinking-Based Non-Thinking (TNT). It does not employ SFT, and sets different maximum token usage for responses not using thinking across various queries by leveraging information from the solution component of the responses using thinking. Experiments on five mathematical benchmarks demonstrate that TNT reduces token usage by around 50% compared to DeepSeek-R1-Distill-Qwen-1.5B/7B and DeepScaleR-1.5B, while significantly improving accuracy. In fact, TNT achieves the optimal trade-off between accuracy and efficiency among all tested methods. Additionally, the probability of reward hacking problem in TNT's responses, which are classified as not using thinking, remains below 10% across all tested datasets. 

---
# Defense Against Indirect Prompt Injection via Tool Result Parsing 

**Authors**: Qiang Yu, Xinran Cheng, Chuanyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04795)  

**Abstract**: As LLM agents transition from digital assistants to physical controllers in autonomous systems and robotics, they face an escalating threat from indirect prompt injection. By embedding adversarial instructions into the results of tool calls, attackers can hijack the agent's decision-making process to execute unauthorized actions. This vulnerability poses a significant risk as agents gain more direct control over physical environments. Existing defense mechanisms against Indirect Prompt Injection (IPI) generally fall into two categories. The first involves training dedicated detection models; however, this approach entails high computational overhead for both training and inference, and requires frequent updates to keep pace with evolving attack vectors. Alternatively, prompt-based methods leverage the inherent capabilities of LLMs to detect or ignore malicious instructions via prompt engineering. Despite their flexibility, most current prompt-based defenses suffer from high Attack Success Rates (ASR), demonstrating limited robustness against sophisticated injection attacks. In this paper, we propose a novel method that provides LLMs with precise data via tool result parsing while effectively filtering out injected malicious code. Our approach achieves competitive Utility under Attack (UA) while maintaining the lowest Attack Success Rate (ASR) to date, significantly outperforming existing methods. Code is available at GitHub. 

---
# APEX: Academic Poster Editing Agentic Expert 

**Authors**: Chengxin Shi, Qinnan Cai, Zeyuan Chen, Long Zeng, Yibo Zhao, Jing Yu, Jianxiang Yu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04794)  

**Abstract**: Designing academic posters is a labor-intensive process requiring the precise balance of high-density content and sophisticated layout. While existing paper-to-poster generation methods automate initial drafting, they are typically single-pass and non-interactive, often fail to align with complex, subjective user intent. To bridge this gap, we propose APEX (Academic Poster Editing agentic eXpert), the first agentic framework for interactive academic poster editing, supporting fine-grained control with robust multi-level API-based editing and a review-and-adjustment Mechanism. In addition, we introduce APEX-Bench, the first systematic benchmark comprising 514 academic poster editing instructions, categorized by a multi-dimensional taxonomy including operation type, difficulty, and abstraction level, constructed via reference-guided and reference-free strategies to ensure realism and diversity. We further establish a multi-dimensional VLM-as-a-judge evaluation protocol to assess instruction fulfillment, modification scope, and visual consistency & harmony. Experimental results demonstrate that APEX significantly outperforms baseline methods. Our implementation is available at this https URL. 

---
# SciIF: Benchmarking Scientific Instruction Following Towards Rigorous Scientific Intelligence 

**Authors**: Encheng Su, Jianyu Wu, Chen Tang, Lintao Wang, Pengze Li, Aoran Wang, Jinouwen Zhang, Yizhou Wang, Yuan Meng, Xinzhu Ma, Shixiang Tang, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04770)  

**Abstract**: As large language models (LLMs) transition from general knowledge retrieval to complex scientific discovery, their evaluation standards must also incorporate the rigorous norms of scientific inquiry. Existing benchmarks exhibit a critical blind spot: general instruction-following metrics focus on superficial formatting, while domain-specific scientific benchmarks assess only final-answer correctness, often rewarding models that arrive at the right result with the wrong reasons. To address this gap, we introduce scientific instruction following: the capability to solve problems while strictly adhering to the constraints that establish scientific validity. Specifically, we introduce SciIF, a multi-discipline benchmark that evaluates this capability by pairing university-level problems with a fixed catalog of constraints across three pillars: scientific conditions (e.g., boundary checks and assumptions), semantic stability (e.g., unit and symbol conventions), and specific processes(e.g., required numerical methods). Uniquely, SciIF emphasizes auditability, requiring models to provide explicit evidence of constraint satisfaction rather than implicit compliance. By measuring both solution correctness and multi-constraint adherence, SciIF enables finegrained diagnosis of compositional reasoning failures, ensuring that LLMs can function as reliable agents within the strict logical frameworks of science. 

---
# AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search 

**Authors**: Zefang Zong, Dingwei Chen, Yang Li, Qi Yi, Bo Zhou, Chengming Li, Bo Qian, Peng Chen, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04767)  

**Abstract**: LLM agents have emerged as powerful systems for tackling multi-turn tasks by interleaving internal reasoning and external tool interactions. Agentic Reinforcement Learning has recently drawn significant research attention as a critical post-training paradigm to further refine these capabilities. In this paper, we present AT$^2$PO (Agentic Turn-based Policy Optimization via Tree Search), a unified framework for multi-turn agentic RL that addresses three core challenges: limited exploration diversity, sparse credit assignment, and misaligned policy optimization. AT$^2$PO introduces a turn-level tree structure that jointly enables Entropy-Guided Tree Expansion for strategic exploration and Turn-wise Credit Assignment for fine-grained reward propagation from sparse outcomes. Complementing this, we propose Agentic Turn-based Policy Optimization, a turn-level learning objective that aligns policy updates with the natural decision granularity of agentic interactions. ATPO is orthogonal to tree search and can be readily integrated into any multi-turn RL pipeline. Experiments across seven benchmarks demonstrate consistent improvements over the state-of-the-art baseline by up to 1.84 percentage points in average, with ablation studies validating the effectiveness of each component. Our code is available at this https URL. 

---
# Orion-RAG: Path-Aligned Hybrid Retrieval for Graphless Data 

**Authors**: Zhen Chen, Weihao Xie, Peilin Chen, Shiqi Wang, Jianping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04764)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective for knowledge synthesis, yet it encounters significant challenges in practical scenarios where data is inherently discrete and fragmented. In most environments, information is distributed across isolated files like reports and logs that lack explicit links. Standard search engines process files independently, ignoring the connections between them. Furthermore, manually building Knowledge Graphs is impractical for such vast data. To bridge this gap, we present Orion-RAG. Our core insight is simple yet effective: we do not need heavy algorithms to organize this data. Instead, we use a low-complexity strategy to extract lightweight paths that naturally link related concepts. We demonstrate that this streamlined approach suffices to transform fragmented documents into semi-structured data, enabling the system to link information across different files effectively. Extensive experiments demonstrate that Orion-RAG consistently outperforms mainstream frameworks across diverse domains, supporting real-time updates and explicit Human-in-the-Loop verification with high cost-efficiency. Experiments on FinanceBench demonstrate superior precision with a 25.2% relative improvement over strong baselines. 

---
# When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail 

**Authors**: Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04748)  

**Abstract**: Multi-agent AI systems have proven effective for complex reasoning. These systems are compounded by specialized agents, which collaborate through explicit communication, but incur substantial computational overhead. A natural question arises: can we achieve similar modularity benefits with a single agent that selects from a library of skills? We explore this question by viewing skills as internalized agent behaviors. From this perspective, a multi-agent system can be compiled into an equivalent single-agent system, trading inter-agent communication for skill selection. Our preliminary experiments suggest this approach can substantially reduce token usage and latency while maintaining competitive accuracy on reasoning benchmarks. However, this efficiency raises a deeper question that has received little attention: how does skill selection scale as libraries grow?
Drawing on principles from cognitive science, we propose that LLM skill selection exhibits bounded capacity analogous to human decision-making. We investigate the scaling behavior of skill selection and observe a striking pattern. Rather than degrading gradually, selection accuracy remains stable up to a critical library size, then drops sharply, indicating a phase transition reminiscent of capacity limits in human cognition. Furthermore, we find evidence that semantic confusability among similar skills, rather than library size alone, plays a central role in this degradation. This perspective suggests that hierarchical organization, which has long helped humans manage complex choices, may similarly benefit AI systems. Our initial results with hierarchical routing support this hypothesis. This work opens new questions about the fundamental limits of semantic-based skill selection in LLMs and offers a cognitive-grounded framework and practical guidelines for designing scalable skill-based agents. 

---
# KnowMe-Bench: Benchmarking Person Understanding for Lifelong Digital Companions 

**Authors**: Tingyu Wu, Zhisheng Chen, Ziyan Weng, Shuhe Wang, Chenglong Li, Shuo Zhang, Sen Hu, Silin Wu, Qizhen Lan, Huacan Wang, Ronghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04745)  

**Abstract**: Existing long-horizon memory benchmarks mostly use multi-turn dialogues or synthetic user histories, which makes retrieval performance an imperfect proxy for person understanding. We present \BenchName, a publicly releasable benchmark built from long-form autobiographical narratives, where actions, context, and inner thoughts provide dense evidence for inferring stable motivations and decision principles. \BenchName~reconstructs each narrative into a flashback-aware, time-anchored stream and evaluates models with evidence-linked questions spanning factual recall, subjective state attribution, and principle-level reasoning. Across diverse narrative sources, retrieval-augmented systems mainly improve factual accuracy, while errors persist on temporally grounded explanations and higher-level inferences, highlighting the need for memory mechanisms beyond retrieval. Our data is in \href{KnowMeBench}{this https URL}. 

---
# Miner:Mining Intrinsic Mastery for Data-Efficient RL in Large Reasoning Models 

**Authors**: Shuyang Jiang, Yuhao Wang, Ya Zhang, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04731)  

**Abstract**: Current critic-free RL methods for large reasoning models suffer from severe inefficiency when training on positive homogeneous prompts (where all rollouts are correct), resulting in waste of rollouts due to zero advantage estimates. We introduce a radically simple yet powerful solution to \uline{M}ine \uline{in}trinsic mast\uline{er}y (Miner), that repurposes the policy's intrinsic uncertainty as a self-supervised reward signal, with no external supervision, auxiliary models, or additional inference cost. Our method pioneers two key innovations: (1) a token-level focal credit assignment mechanism that dynamically amplifies gradients on critical uncertain tokens while suppressing overconfident ones, and (2) adaptive advantage calibration to seamlessly integrate intrinsic and verifiable rewards. Evaluated across six reasoning benchmarks on Qwen3-4B and Qwen3-8B base models, Miner achieves state-of-the-art performance among the other four algorithms, yielding up to \textbf{4.58} absolute gains in Pass@1 and \textbf{6.66} gains in Pass@K compared to GRPO. Comparison with other methods targeted at exploration enhancement further discloses the superiority of the two newly proposed innovations. This demonstrates that latent uncertainty exploitation is both necessary and sufficient for efficient and scalable RL training of reasoning models. 

---
# Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning 

**Authors**: Yuyang Hu, Jiongnan Liu, Jiejun Tan, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04726)  

**Abstract**: Large language models (LLMs) are increasingly deployed as intelligent agents that reason, plan, and interact with their environments. To effectively scale to long-horizon scenarios, a key capability for such agents is a memory mechanism that can retain, organize, and retrieve past experiences to support downstream decision-making. However, most existing approaches organize and store memories in a flat manner and rely on simple similarity-based retrieval techniques. Even when structured memory is introduced, existing methods often struggle to explicitly capture the logical relationships among experiences or memory units. Moreover, memory access is largely detached from the constructed structure and still depends on shallow semantic retrieval, preventing agents from reasoning logically over long-horizon dependencies. In this work, we propose CompassMem, an event-centric memory framework inspired by Event Segmentation Theory. CompassMem organizes memory as an Event Graph by incrementally segmenting experiences into events and linking them through explicit logical relations. This graph serves as a logic map, enabling agents to perform structured and goal-directed navigation over memory beyond superficial retrieval, progressively gathering valuable memories to support long-horizon reasoning. Experiments on LoCoMo and NarrativeQA demonstrate that CompassMem consistently improves both retrieval and reasoning performance across multiple backbone models. 

---
# ThinkDrive: Chain-of-Thought Guided Progressive Reinforcement Learning Fine-Tuning for Autonomous Driving 

**Authors**: Chang Zhao, Zheming Yang, Yunqing Hu, Qi Guo, Zijian Wang, Pengcheng Li, Wen Ji  

**Link**: [PDF](https://arxiv.org/pdf/2601.04714)  

**Abstract**: With the rapid advancement of large language models (LLMs) technologies, their application in the domain of autonomous driving has become increasingly widespread. However, existing methods suffer from unstructured reasoning, poor generalization, and misalignment with human driving intent. While Chain-of-Thought (CoT) reasoning enhances decision transparency, conventional supervised fine-tuning (SFT) fails to fully exploit its potential, and reinforcement learning (RL) approaches face instability and suboptimal reasoning depth. We propose ThinkDrive, a CoT guided progressive RL fine-tuning framework for autonomous driving that synergizes explicit reasoning with difficulty-aware adaptive policy optimization. Our method employs a two-stage training strategy. First, we perform SFT using CoT explanations. Then, we apply progressive RL with a difficulty-aware adaptive policy optimizer that dynamically adjusts learning intensity based on sample complexity. We evaluate our approach on a public dataset. The results show that ThinkDrive outperforms strong RL baselines by 1.45%, 1.95%, and 1.01% on exam, easy-exam, and accuracy, respectively. Moreover, a 2B-parameter model trained with our method surpasses the much larger GPT-4o by 3.28% on the exam metric. 

---
# Bridging Temporal and Textual Modalities: A Multimodal Framework for Automated Cloud Failure Root Cause Analysis 

**Authors**: Gijun Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.04709)  

**Abstract**: Root cause analysis in modern cloud infrastructure demands sophisticated understanding of heterogeneous data sources, particularly time-series performance metrics that involve core failure signatures. While large language models demonstrate remarkable capabilities in textual reasoning, their discrete token-based architecture creates fundamental incompatibilities with continuous numerical sequences exhibiting temporal dependencies. Current methodologies inadequately address this modality mismatch, constraining the potential of language model-driven automation in incident management workflows. This paper presents a multimodal diagnostic framework that harmonizes time-series representations with pretrained language model embedding spaces. Our approach contributes three technical advances: (1) a semantic compression technique that distills temporal segments into single-token abstractions while preserving pattern semantics, (2) an alignment encoder utilizing gated cross-attention to project time-series features into language model latent space, and (3) a retrieval-augmented diagnostic pipeline that synthesizes aligned embeddings with historical incident knowledge for expert-level failure attribution. Comprehensive evaluation across six cloud system benchmarks demonstrates that our framework achieves leading performance, reaching 48.75% diagnostic accuracy with notable improvements on scenarios involving compound failure modes. The results validate embedding-space alignment as an effective strategy for enabling language models to reason over multimodal telemetry data in production incident response contexts. 

---
# Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search 

**Authors**: Yiqun Chen, Lingyong Yan, Zixuan Yang, Erhan Zhang, Jiashu Zhao, Shuaiqiang Wang, Dawei Yin, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04703)  

**Abstract**: Agentic search has emerged as a promising paradigm for complex information seeking by enabling Large Language Models (LLMs) to interleave reasoning with tool use. However, prevailing systems rely on monolithic agents that suffer from structural bottlenecks, including unconstrained reasoning outputs that inflate trajectories, sparse outcome-level rewards that complicate credit assignment, and stochastic search noise that destabilizes learning. To address these challenges, we propose \textbf{M-ASK} (Multi-Agent Search and Knowledge), a framework that explicitly decouples agentic search into two complementary roles: Search Behavior Agents, which plan and execute search actions, and Knowledge Management Agents, which aggregate, filter, and maintain a compact internal context. This decomposition allows each agent to focus on a well-defined subtask and reduces interference between search and context construction. Furthermore, to enable stable coordination, M-ASK employs turn-level rewards to provide granular supervision for both search decisions and knowledge updates. Experiments on multi-hop QA benchmarks demonstrate that M-ASK outperforms strong baselines, achieving not only superior answer accuracy but also significantly more stable training dynamics.\footnote{The source code for M-ASK is available at this https URL.} 

---
# TourPlanner: A Competitive Consensus Framework with Constraint-Gated Reinforcement Learning for Travel Planning 

**Authors**: Yinuo Wang, Mining Tan, Wenxiang Jiao, Xiaoxi Li, Hao Wang, Xuanyu Zhang, Yuan Lu, Weiming Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04698)  

**Abstract**: Travel planning is a sophisticated decision-making process that requires synthesizing multifaceted information to construct itineraries. However, existing travel planning approaches face several challenges: (1) Pruning candidate points of interest (POIs) while maintaining a high recall rate; (2) A single reasoning path restricts the exploration capability within the feasible solution space for travel planning; (3) Simultaneously optimizing hard constraints and soft constraints remains a significant difficulty. To address these challenges, we propose TourPlanner, a comprehensive framework featuring multi-path reasoning and constraint-gated reinforcement learning. Specifically, we first introduce a Personalized Recall and Spatial Optimization (PReSO) workflow to construct spatially-aware candidate POIs' set. Subsequently, we propose Competitive consensus Chain-of-Thought (CCoT), a multi-path reasoning paradigm that improves the ability of exploring the feasible solution space. To further refine the plan, we integrate a sigmoid-based gating mechanism into the reinforcement learning stage, which dynamically prioritizes soft-constraint satisfaction only after hard constraints are met. Experimental results on travel planning benchmarks demonstrate that TourPlanner achieves state-of-the-art performance, significantly surpassing existing methods in both feasibility and user-preference alignment. 

---
# A Method for Constructing a Digital Transformation Driving Mechanism Based on Semantic Understanding of Large Models 

**Authors**: Huayi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04696)  

**Abstract**: In the process of digital transformation, enterprises are faced with problems such as insufficient semantic understanding of unstructured data and lack of intelligent decision-making basis in driving mechanisms. This study proposes a method that combines a large language model (LLM) and a knowledge graph. First, a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model is used to perform entity recognition and relationship extraction on multi-source heterogeneous texts, and GPT-4 is used to generate semantically enhanced vector representations; secondly, a two-layer graph neural network (GNN) architecture is designed to fuse the semantic vectors output by LLM with business metadata to construct a dynamic and scalable enterprise knowledge graph; then reinforcement learning is introduced to optimize decision path generation, and the reward function is used to drive the mechanism iteration. In the case of the manufacturing industry, this mechanism reduced the response time for equipment failure scenarios from 7.8 hours to 3.7 hours, the F1 value reached 94.3%, and the compensation for decision errors in the annual digital transformation cost decreased by 45.3%. This method significantly enhances the intelligence level and execution efficiency of the digital transformation driving mechanism by integrating large model semantic understanding with structured knowledge. 

---
# Tape: A Cellular Automata Benchmark for Evaluating Rule-Shift Generalization in Reinforcement Learning 

**Authors**: Enze Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04695)  

**Abstract**: We present Tape, a controlled reinforcement-learning benchmark designed to isolate out-of-distribution (OOD) failure under latent rule this http URL is derived from one-dimensional cellular automata, enabling precise train/test splits where observation and action spaces are held fixed while transition rules change. Using a reproducible evaluation pipeline, we compare model-free baselines, model-based planning with learned world models, and task-inference (meta-RL) methods. A consistent pattern emerges: methods that are strong in-distribution (ID) can collapse under heldout-rule OOD, and high-variance OOD evaluation can make rankings unstable unless experiments are sufficiently this http URL provide (i) standardized OOD protocols, (ii) statistical reporting requirements (seeds, confidence intervals, and hypothesis tests), and (iii) information-theoretic identities connecting entropy reduction to conditional mutual information and expected posterior KL divergence, clarifying what "uncertainty reduction" objectives can and cannot guarantee under rule shifts. 

---
# ResMAS: Resilience Optimization in LLM-based Multi-agent Systems 

**Authors**: Zhilun Zhou, Zihan Liu, Jiahe Liu, Qingyu Shao, Yihan Wang, Kun Shao, Depeng Jin, Fengli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04694)  

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-based MAS), where multiple LLM agents collaborate to solve complex tasks, have shown impressive performance in many areas. However, MAS are typically distributed across different devices or environments, making them vulnerable to perturbations such as agent failures. While existing works have studied the adversarial attacks and corresponding defense strategies, they mainly focus on reactively detecting and mitigating attacks after they occur rather than proactively designing inherently resilient systems. In this work, we study the resilience of LLM-based MAS under perturbations and find that both the communication topology and prompt design significantly influence system resilience. Motivated by these findings, we propose ResMAS: a two-stage framework for enhancing MAS resilience. First, we train a reward model to predict the MAS's resilience, based on which we train a topology generator to automatically design resilient topology for specific tasks through reinforcement learning. Second, we introduce a topology-aware prompt optimization method that refines each agent's prompt based on its connections and interactions with other agents. Extensive experiments across a range of tasks show that our approach substantially improves MAS resilience under various constraints. Moreover, our framework demonstrates strong generalization ability to new tasks and models, highlighting its potential for building resilient MASs. 

---
# LLM-Guided Quantified SMT Solving over Uninterpreted Functions 

**Authors**: Kunhang Lv, Yuhang Dong, Rui Han, Fuqi Jia, Feifei Ma, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04675)  

**Abstract**: Quantified formulas with Uninterpreted Functions (UFs) over non-linear real arithmetic pose fundamental challenges for Satisfiability Modulo Theories (SMT) solving. Traditional quantifier instantiation methods struggle because they lack semantic understanding of UF constraints, forcing them to search through unbounded solution spaces with limited guidance. We present AquaForte, a framework that leverages Large Language Models to provide semantic guidance for UF instantiation by generating instantiated candidates for function definitions that satisfy the constraints, thereby significantly reducing the search space and complexity for solvers. Our approach preprocesses formulas through constraint separation, uses structured prompts to extract mathematical reasoning from LLMs, and integrates the results with traditional SMT algorithms through adaptive instantiation. AquaForte maintains soundness through systematic validation: LLM-guided instantiations yielding SAT solve the original problem, while UNSAT results generate exclusion clauses for iterative refinement. Completeness is preserved by fallback to traditional solvers augmented with learned constraints. Experimental evaluation on SMT-COMP benchmarks demonstrates that AquaForte solves numerous instances where state-of-the-art solvers like Z3 and CVC5 timeout, with particular effectiveness on satisfiable formulas. Our work shows that LLMs can provide valuable mathematical intuition for symbolic reasoning, establishing a new paradigm for SMT constraint solving. 

---
# Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning 

**Authors**: Zhiyuan Chang, Mingyang Li, Yuekai Huang, Ziyou Jiang, Xiaojun Jia, Qian Xiong, Junjie Wang, Zhaoyang Li, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04666)  

**Abstract**: Large language model (LLM)-integrated applications have become increasingly prevalent, yet face critical security vulnerabilities from prompt injection (PI) attacks. Defending against PI attacks faces two major issues: malicious instructions can be injected through diverse vectors, and injected instructions often lack clear semantic boundaries from the surrounding context, making them difficult to identify. To address these issues, we propose InstruCoT, a model enhancement method for PI defense that synthesizes diverse training data and employs instruction-level chain-of-thought fine-tuning, enabling LLMs to effectively identify and reject malicious instructions regardless of their source or position in the context. We evaluate InstruCoT across three critical dimensions: Behavior Deviation, Privacy Leakage, and Harmful Output. Experimental results across four LLMs demonstrate that InstruCoT significantly outperforms baselines in all dimensions while maintaining utility performance without degradation 

---
# Vibe Coding an LLM-powered Theorem Prover 

**Authors**: Zhe Hou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04653)  

**Abstract**: We present Isabellm, an LLM-powered theorem prover for Isabelle/HOL that performs fully automatic proof synthesis. Isabellm works with any local LLM on Ollama and APIs such as Gemini CLI, and it is designed to run on consumer grade computers. The system combines a stepwise prover, which uses large language models to propose proof commands validated by Isabelle in a bounded search loop, with a higher-level proof planner that generates structured Isar outlines and attempts to fill and repair remaining gaps. The framework includes beam search for tactics, tactics reranker ML and RL models, premise selection with small transformer models, micro-RAG for Isar proofs built from AFP, and counter-example guided proof repair. All the code is implemented by GPT 4.1 - 5.2, Gemini 3 Pro, and Claude 4.5. Empirically, Isabellm can prove certain lemmas that defeat Isabelle's standard automation, including Sledgehammer, demonstrating the practical value of LLM-guided proof search. At the same time, we find that even state-of-the-art LLMs, such as GPT 5.2 Extended Thinking and Gemini 3 Pro struggle to reliably implement the intended fill-and-repair mechanisms with complex algorithmic designs, highlighting fundamental challenges in LLM code generation and reasoning. The code of Isabellm is available at this https URL 

---
# Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models 

**Authors**: Can Xu, Lingyong Yan, Jiayi Wu, Haosen Wang, Shuaiqiang Wang, Yuchen Li, Jizhou Huang, Dawei Yin, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04651)  

**Abstract**: Recent advances in synergizing large reasoning models (LRMs) with retrieval-augmented generation (RAG) have shown promising results, yet two critical challenges remain: (1) reasoning models typically operate from a single, unchallenged perspective, limiting their ability to conduct deep, self-correcting reasoning over external documents, and (2) existing training paradigms rely excessively on outcome-oriented rewards, which provide insufficient signal for shaping the complex, multi-step reasoning process. To address these issues, we propose an Reasoner-Verifier framework named Adversarial Reasoning RAG (ARR). The Reasoner and Verifier engage in reasoning on retrieved evidence and critiquing each other's logic while being guided by process-aware advantage that requires no external scoring model. This reward combines explicit observational signals with internal model uncertainty to jointly optimize reasoning fidelity and verification rigor. Experiments on multiple benchmarks demonstrate the effectiveness of our method. 

---
# Beyond the "Truth": Investigating Election Rumors on Truth Social During the 2024 Election 

**Authors**: Etienne Casanova, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2601.04631)  

**Abstract**: Large language models (LLMs) offer unprecedented opportunities for analyzing social phenomena at scale. This paper demonstrates the value of LLMs in psychological measurement by (1) compiling the first large-scale dataset of election rumors on a niche alt-tech platform, (2) developing a multistage Rumor Detection Agent that leverages LLMs for high-precision content classification, and (3) quantifying the psychological dynamics of rumor propagation, specifically the "illusory truth effect" in a naturalistic setting. The Rumor Detection Agent combines (i) a synthetic data-augmented, fine-tuned RoBERTa classifier, (ii) precision keyword filtering, and (iii) a two-pass LLM verification pipeline using GPT-4o mini. The findings reveal that sharing probability rises steadily with each additional exposure, providing large-scale empirical evidence for dose-response belief reinforcement in ideologically homogeneous networks. Simulation results further demonstrate rapid contagion effects: nearly one quarter of users become "infected" within just four propagation iterations. Taken together, these results illustrate how LLMs can transform psychological science by enabling the rigorous measurement of belief dynamics and misinformation spread in massive, real-world datasets. 

---
# AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering 

**Authors**: Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04620)  

**Abstract**: Recent progress in large language model (LLM) agents has largely focused on embedding self-improvement mechanisms inside the agent or searching over many concurrent variants. While these approaches can raise aggregate scores, they often yield unstable and hard-to-audit improvement trajectories, making it difficult to guarantee non-regression or to reason about failures across versions. We reframe agent improvement as \textbf{release engineering}: agents are treated as shippable artifacts, and improvement is externalized into a regression-aware release pipeline. We introduce \textbf{AgentDevel}, a release engineering pipeline that iteratively runs the current agent, produces implementation-blind, symptom-level quality signals from execution traces, synthesizes a single release candidate (RC) via executable diagnosis, and promotes it under flip-centered gating. AgentDevel features three core designs: (i) an implementation-blind LLM critic that characterizes failure appearances without accessing agent internals, (ii) script-based executable diagnosis that aggregates dominant symptom patterns and produces auditable engineering specifications, and (iii) flip-centered gating that prioritizes pass to fail regressions and fail to pass fixes as first-class evidence. Unlike population-based search or in-agent self-refinement, AgentDevel maintains a single canonical version line and emphasizes non-regression as a primary objective. Experiments on execution-heavy benchmarks demonstrate that AgentDevel yields stable improvements with significantly fewer regressions while producing reproducible, auditable artifacts. Overall, AgentDevel provides a practical development discipline for building, debugging, and releasing LLM agents as software development. 

---
# Evaluating Human and Machine Confidence in Phishing Email Detection: A Comparative Study 

**Authors**: Paras Jain, Khushi Dhar, Olyemi E. Amujo, Esa M. Rantanen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04610)  

**Abstract**: Identifying deceptive content like phishing emails demands sophisticated cognitive processes that combine pattern recognition, confidence assessment, and contextual analysis. This research examines how human cognition and machine learn- ing models work together to distinguish phishing emails from legitimate ones. We employed three interpretable algorithms Logistic Regression, Decision Trees, and Random Forests train- ing them on both TF-IDF features and semantic embeddings, then compared their predictions against human evaluations that captured confidence ratings and linguistic observations. Our results show that machine learning models provide good accuracy rates, but their confidence levels vary significantly. Human evaluators, on the other hand, use a greater variety of language signs and retain more consistent confidence. We also found that while language proficiency has minimal effect on detection performance, aging does. These findings offer helpful direction for creating transparent AI systems that complement human cognitive functions, ultimately improving human-AI cooperation in challenging content analysis tasks. 

---
# Autonomous Agents on Blockchains: Standards, Execution Models, and Trust Boundaries 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2601.04583)  

**Abstract**: Advances in large language models have enabled agentic AI systems that can reason, plan, and interact with external tools to execute multi-step workflows, while public blockchains have evolved into a programmable substrate for value transfer, access control, and verifiable state transitions. Their convergence introduces a high-stakes systems challenge: designing standard, interoperable, and secure interfaces that allow agents to observe on-chain state, formulate transaction intents, and authorize execution without exposing users, protocols, or organizations to unacceptable security, governance, or economic risks. This survey systematizes the emerging landscape of agent-blockchain interoperability through a systematic literature review, identifying 317 relevant works from an initial pool of over 3000 records. We contribute a five-part taxonomy of integration patterns spanning read-only analytics, simulation and intent generation, delegated execution, autonomous signing, and multi-agent workflows; a threat model tailored to agent-driven transaction pipelines that captures risks ranging from prompt injection and policy misuse to key compromise, adversarial execution dynamics, and multi-agent collusion; and a comparative capability matrix analyzing more than 20 representative systems across 13 dimensions, including custody models, permissioning, policy enforcement, observability, and recovery. Building on the gaps revealed by this analysis, we outline a research roadmap centered on two interface abstractions: a Transaction Intent Schema for portable and unambiguous goal specification, and a Policy Decision Record for auditable, verifiable policy enforcement across execution environments. We conclude by proposing a reproducible evaluation suite and benchmarks for assessing the safety, reliability, and economic robustness of agent-mediated on-chain execution. 

---
# Sci-Reasoning: A Dataset Decoding AI Innovation Patterns 

**Authors**: Jiachen Liu, Maestro Harmon, Zechen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04577)  

**Abstract**: While AI innovation accelerates rapidly, the intellectual process behind breakthroughs -- how researchers identify gaps, synthesize prior work, and generate insights -- remains poorly understood. The lack of structured data on scientific reasoning hinders systematic analysis and development of AI research agents. We introduce Sci-Reasoning, the first dataset capturing the intellectual synthesis behind high-quality AI research. Using community-validated quality signals and an LLM-accelerated, human-verified pipeline, we trace Oral and Spotlight papers across NeurIPS, ICML, and ICLR (2023-2025) to its key predecessors, articulating specific reasoning links in a structured format. Our analysis identifies 15 distinct thinking patterns, with three dominant strategies accounting for 52.7%: Gap-Driven Reframing (24.2%), Cross-Domain Synthesis (18.0%), and Representation Shift (10.5%). The most powerful innovation recipes combine multiple patterns: Gap-Driven Reframing + Representation Shift, Cross-Domain Synthesis + Representation Shift, and Gap-Driven Reframing + Cross-Domain Synthesis. This dataset enables quantitative studies of scientific progress and provides structured reasoning trajectories for training the next generation AI research agents. 

---
# Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing 

**Authors**: Yuguang Yue, Irakli Salia, Samuel Hunt, Chris Green, Wenzhe Shi, Jonathan J Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2601.04575)  

**Abstract**: Behavior cloning is enjoying a resurgence in popularity as scaling both model and data sizes proves to provide a strong starting point for many tasks of interest. In this work, we introduce an open recipe for training a video game playing foundation model designed for inference in realtime on a consumer GPU. We release all data (8300+ hours of high quality human gameplay), training and inference code, and pretrained checkpoints under an open license. We show that our best model is capable of playing a variety of 3D video games at a level competitive with human play. We use this recipe to systematically examine the scaling laws of behavior cloning to understand how the model's performance and causal reasoning varies with model and data scale. We first show in a simple toy problem that, for some types of causal reasoning, increasing both the amount of training data and the depth of the network results in the model learning a more causal policy. We then systematically study how causality varies with the number of parameters (and depth) and training steps in scaled models of up to 1.2 billion parameters, and we find similar scaling results to what we observe in the toy problem. 

---
# Enhancing Multimodal Retrieval via Complementary Information Extraction and Alignment 

**Authors**: Delong Zeng, Yuexiang Xie, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04571)  

**Abstract**: Multimodal retrieval has emerged as a promising yet challenging research direction in recent years. Most existing studies in multimodal retrieval focus on capturing information in multimodal data that is similar to their paired texts, but often ignores the complementary information contained in multimodal data. In this study, we propose CIEA, a novel multimodal retrieval approach that employs Complementary Information Extraction and Alignment, which transforms both text and images in documents into a unified latent space and features a complementary information extractor designed to identify and preserve differences in the image representations. We optimize CIEA using two complementary contrastive losses to ensure semantic integrity and effectively capture the complementary information contained in images. Extensive experiments demonstrate the effectiveness of CIEA, which achieves significant improvements over both divide-and-conquer models and universal dense retrieval models. We provide an ablation study, further discussions, and case studies to highlight the advancements achieved by CIEA. To promote further research in the community, we have released the source code at this https URL. 

---
# Neurosymbolic Retrievers for Retrieval-augmented Generation 

**Authors**: Yash Saxena, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2601.04568)  

**Abstract**: Retrieval Augmented Generation (RAG) has made significant strides in overcoming key limitations of large language models, such as hallucination, lack of contextual grounding, and issues with transparency. However, traditional RAG systems consist of three interconnected neural components - the retriever, re-ranker, and generator - whose internal reasoning processes remain opaque. This lack of transparency complicates interpretability, hinders debugging efforts, and erodes trust, especially in high-stakes domains where clear decision-making is essential. To address these challenges, we introduce the concept of Neurosymbolic RAG, which integrates symbolic reasoning using a knowledge graph with neural retrieval techniques. This new framework aims to answer two primary questions: (a) Can retrievers provide a clear and interpretable basis for document selection? (b) Can symbolic knowledge enhance the clarity of the retrieval process? We propose three methods to improve this integration. First is MAR (Knowledge Modulation Aligned Retrieval) that employs modulation networks to refine query embeddings using interpretable symbolic features, thereby making document matching more explicit. Second, KG-Path RAG enhances queries by traversing knowledge graphs to improve overall retrieval quality and interpretability. Lastly, Process Knowledge-infused RAG utilizes domain-specific tools to reorder retrieved content based on validated workflows. Preliminary results from mental health risk assessment tasks indicate that this neurosymbolic approach enhances both transparency and overall performance 

---
# BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents 

**Authors**: Yunhao Feng, Yige Li, Yutao Wu, Yingshui Tan, Yanming Guo, Yifan Ding, Kun Zhai, Xingjun Ma, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04566)  

**Abstract**: Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub. 

---
# Reasoning Over Space: Enabling Geographic Reasoning for LLM-Based Generative Next POI Recommendation 

**Authors**: Dongyi Lv, Qiuyu Ding, Heng-Da Xu, Zhaoxu Sun, Zhi Wang, Feng Xiong, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04562)  

**Abstract**: Generative recommendation with large language models (LLMs) reframes prediction as sequence generation, yet existing LLM-based recommenders remain limited in leveraging geographic signals that are crucial in mobility and local-services scenarios. Here, we present Reasoning Over Space (ROS), a framework that utilizes geography as a vital decision variable within the reasoning process. ROS introduces a Hierarchical Spatial Semantic ID (SID) that discretizes coarse-to-fine locality and POI semantics into compositional tokens, and endows LLM with a three-stage Mobility Chain-of-Thought (CoT) paradigm that models user personality, constructs an intent-aligned candidate space, and performs locality informed pruning. We further align the model with real world geography via spatial-guided Reinforcement Learning (RL). Experiments on three widely used location-based social network (LBSN) datasets show that ROS achieves over 10% relative gains in hit rate over strongest LLM-based baselines and improves cross-city transfer, despite using a smaller backbone model. 

---
# Personalized Model-Based Design of Human Centric AI enabled CPS for Long term usage 

**Authors**: Bernard Ngabonziza, Ayan Banerjee, Sandeep K.S. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.04545)  

**Abstract**: Human centric critical systems are increasingly involving artificial intelligence to enable knowledge extraction from sensor collected data. Examples include medical monitoring and control systems, gesture based human computer interaction systems, and autonomous cars. Such systems are intended to operate for a long term potentially for a lifetime in many scenarios such as closed loop blood glucose control for Type 1 diabetics, self-driving cars, and monitoting systems for stroke diagnosis, and rehabilitation. Long term operation of such AI enabled human centric applications can expose them to corner cases for which their operation is may be uncertain. This can be due to many reasons such as inherent flaws in the design, limited resources for testing, inherent computational limitations of the testing methodology, or unknown use cases resulting from human interaction with the system. Such untested corner cases or cases for which the system performance is uncertain can lead to violations in the safety, sustainability, and security requirements of the system. In this paper, we analyze the existing techniques for safety, sustainability, and security analysis of an AI enabled human centric control system and discuss their limitations for testing the system for long term use in practice. We then propose personalized model based solutions for potentially eliminating such limitations. 

---
# TCAndon-Router: Adaptive Reasoning Router for Multi-Agent Collaboration 

**Authors**: Jiuzhou Zhao, Chunrong Chen, Chenqi Qiao, Lebin Zheng, Minqi Han, Yanchi Liu Yongzhou Xu Xiaochuan Xu Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04544)  

**Abstract**: Multi-Agent Systems(MAS) have become a powerful paradigm for building high performance intelligent applications. Within these systems, the router responsible for determining which expert agents should handle a given query plays a crucial role in overall performance. Existing routing strategies generally fall into two categories: performance routing, which balances latency and cost across models of different sizes, and task routing, which assigns queries to domain-specific experts to improve accuracy. In real-world enterprise applications, task routing is more suitable; however, most existing approaches rely on static single-label decisions, which introduce two major limitations: (i) difficulty in seamlessly integrating new agents as business domains expand, and (ii) routing conflicts caused by overlapping agent capabilities, ultimately degrading accuracy and this http URL address these challenges, we propose TCAndon-Router(TCAR): an adaptive reasoning router for multi-agent collaboration. Unlike traditional routers, TCAR supports dynamic agent onboarding and first generates a natural-language reasoning chain before predicting a set of candidate agents capable of handling the query. In addition, we design a collaborative execution pipeline in which selected agents independently produce responses, which are then aggregated and refined into a single high-quality response by a dedicated Refining this http URL on public datasets and real enterprise data demonstrate that TCAR significantly improves routing accuracy, reduces routing conflicts, and remains robust in ambiguous scenarios. We have released TCAR at this https URL to support future research on explainable and collaborative multi-agent routing. 

---
# BioPIE: A Biomedical Protocol Information Extraction Dataset for High-Reasoning-Complexity Experiment Question Answer 

**Authors**: Haofei Hou, Shunyi Zhao, Fanxu Meng, Kairui Yang, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04524)  

**Abstract**: Question Answer (QA) systems for biomedical experiments facilitate cross-disciplinary communication, and serve as a foundation for downstream tasks, e.g., laboratory automation. High Information Density (HID) and Multi-Step Reasoning (MSR) pose unique challenges for biomedical experimental QA. While extracting structured knowledge, e.g., Knowledge Graphs (KGs), can substantially benefit biomedical experimental QA. Existing biomedical datasets focus on general or coarsegrained knowledge and thus fail to support the fine-grained experimental reasoning demanded by HID and MSR. To address this gap, we introduce Biomedical Protocol Information Extraction Dataset (BioPIE), a dataset that provides procedure-centric KGs of experimental entities, actions, and relations at a scale that supports reasoning over biomedical experiments across protocols. We evaluate information extraction methods on BioPIE, and implement a QA system that leverages BioPIE, showcasing performance gains on test, HID, and MSR question sets, showing that the structured experimental knowledge in BioPIE underpins both AI-assisted and more autonomous biomedical experimentation. 

---
# Integrating Distribution Matching into Semi-Supervised Contrastive Learning for Labeled and Unlabeled Data 

**Authors**: Shogo Nakayama, Masahiro Okuda  

**Link**: [PDF](https://arxiv.org/pdf/2601.04518)  

**Abstract**: The advancement of deep learning has greatly improved supervised image classification. However, labeling data is costly, prompting research into unsupervised learning methods such as contrastive learning. In real-world scenarios, fully unlabeled datasets are rare, making semi-supervised learning (SSL) highly relevant in scenarios where a small amount of labeled data coexists with a large volume of unlabeled data. A well-known semi-supervised contrastive learning approach involves assigning pseudo-labels to unlabeled data. This study aims to enhance pseudo-label-based SSL by incorporating distribution matching between labeled and unlabeled feature embeddings to improve image classification accuracy across multiple datasets. 

---
# A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention 

**Authors**: Peixin Huang, Yaoxin Wu, Yining Ma, Cathy Wu, Wen Song, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04509)  

**Abstract**: Mixed-integer linear programming (MILP), a widely used modeling framework for combinatorial optimization, are central to many scientific and engineering applications, yet remains computationally challenging at scale. Recent advances in deep learning address this challenge by representing MILP instances as variable-constraint bipartite graphs and applying graph neural networks (GNNs) to extract latent structural patterns and enhance solver efficiency. However, this architecture is inherently limited by the local-oriented mechanism, leading to restricted representation power and hindering neural approaches for MILP. Here we present an attention-driven neural architecture that learns expressive representations beyond the pure graph view. A dual-attention mechanism is designed to perform parallel self- and cross-attention over variables and constraints, enabling global information exchange and deeper representation learning. We apply this general backbone to various downstream tasks at the instance level, element level, and solving state level. Extensive experiments across widely used benchmarks show consistent improvements of our approach over state-of-the-art baselines, highlighting attention-based neural architectures as a powerful foundation for learning-enhanced mixed-integer linear optimization. 

---
# CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts 

**Authors**: Khandakar Shakib Al Hasan, Syed Rifat Raiyan, Hasin Mahtab Alvee, Wahid Sadik  

**Link**: [PDF](https://arxiv.org/pdf/2601.04505)  

**Abstract**: Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronics design, as large language models (LLMs) frequently hallucinate in granular details, violate electrical constraints, and produce non-machine-readable outputs. We present CircuitLM, a novel multi-agent LLM-aided circuit design pipeline that translates user prompts into structured, visually interpretable CircuitJSON schematics through five sequential stages: (i) LLM-based component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning by an electronics expert agent, (iv) JSON schematic synthesis, and (v) force-directed SVG visualization. Anchored by a curated, embedding-powered component knowledge base. While LLMs often violate electrical constraints, CircuitLM bridges this gap by grounding generation in a verified and dynamically extensible component database, initially comprising 50 components. To ensure safety, we incorporate a hybrid evaluation framework, namely Dual-Metric Circuit Validation (DMCV), validated against human-expert assessments, which achieves high fidelity in microcontroller-centric designs. We evaluate the system on 100 diverse embedded-systems prompts across six LLMs and introduce DMCV to assess both structural and electrical validity. This work bridges natural language input to deployable hardware designs, enabling reliable circuit prototyping by non-experts. Our code and data will be made public upon acceptance. 

---
# Specific Emitter Identification via Active Learning 

**Authors**: Jingyi Wang, Fanggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04502)  

**Abstract**: With the rapid growth of wireless communications, specific emitter identification (SEI) is significant for communication security. However, its model training relies heavily on the large-scale labeled data, which are costly and time-consuming to obtain. To address this challenge, we propose an SEI approach enhanced by active learning (AL), which follows a three-stage semi-supervised training scheme. In the first stage, self-supervised contrastive learning is employed with a dynamic dictionary update mechanism to extract robust representations from large amounts of the unlabeled data. In the second stage, supervised training on a small labeled dataset is performed, where the contrastive and cross-entropy losses are jointly optimized to improve the feature separability and strengthen the classification boundaries. In the third stage, an AL module selects the most valuable samples from the unlabeled data for annotation based on the uncertainty and representativeness criteria, further enhancing generalization under limited labeling budgets. Experimental results on the ADS-B and WiFi datasets demonstrate that the proposed SEI approach significantly outperforms the conventional supervised and semi-supervised methods under limited annotation conditions, achieving higher recognition accuracy with lower labeling cost. 

---
# GUITester: Enabling GUI Agents for Exploratory Defect Discovery 

**Authors**: Yifei Gao, Jiang Wu, Xiaoyi Chen, Yifan Yang, Zhe Cui, Tianyi Ma, Jiaming Zhang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04500)  

**Abstract**: Exploratory GUI testing is essential for software quality but suffers from high manual costs. While Multi-modal Large Language Model (MLLM) agents excel in navigation, they fail to autonomously discover defects due to two core challenges: \textit{Goal-Oriented Masking}, where agents prioritize task completion over reporting anomalies, and \textit{Execution-Bias Attribution}, where system defects are misidentified as agent errors. To address these, we first introduce \textbf{GUITestBench}, the first interactive benchmark for this task, featuring 143 tasks across 26 defects. We then propose \textbf{GUITester}, a multi-agent framework that decouples navigation from verification via two modules: (i) a \textit{Planning-Execution Module (PEM)} that proactively probes for defects via embedded testing intents, and (ii) a \textit{Hierarchical Reflection Module (HRM)} that resolves attribution ambiguity through interaction history analysis. GUITester achieves an F1-score of 48.90\% (Pass@3) on GUITestBench, outperforming state-of-the-art baselines (33.35\%). Our work demonstrates the feasibility of autonomous exploratory testing and provides a robust foundation for future GUI quality assurance~\footnote{Our code is now available in~\href{this https URL}{this https URL}}. 

---
# A Closed-Loop Multi-Agent System Driven by LLMs for Meal-Level Personalized Nutrition Management 

**Authors**: Muqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04491)  

**Abstract**: Personalized nutrition management aims to tailor dietary guidance to an individual's intake and phenotype, but most existing systems handle food logging, nutrient analysis and recommendation separately. We present a next-generation mobile nutrition assistant that combines image based meal logging with an LLM driven multi agent controller to provide meal level closed loop support. The system coordinates vision, dialogue and state management agents to estimate nutrients from photos and update a daily intake budget. It then adapts the next meal plan to user preferences and dietary constraints. Experiments with SNAPMe meal images and simulated users show competitive nutrient estimation, personalized menus and efficient task plans. These findings demonstrate the feasibility of multi agent LLM control for personalized nutrition and reveal open challenges in micronutrient estimation from images and in large scale real world studies. 

---
# Computational Compliance for AI Regulation: Blueprint for a New Research Domain 

**Authors**: Bill Marino, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2601.04474)  

**Abstract**: The era of AI regulation (AIR) is upon us. But AI systems, we argue, will not be able to comply with these regulations at the necessary speed and scale by continuing to rely on traditional, analogue methods of compliance. Instead, we posit that compliance with these regulations will only realistically be achieved computationally: that is, with algorithms that run across the life cycle of an AI system, automatically steering it toward AIR compliance in the face of dynamic conditions. Yet despite their (we would argue) inevitability, the research community has yet to specify exactly how these algorithms for computational AIR compliance should behave - or how we should benchmark their performance. To fill these gaps, we specify a set of design goals for such algorithms. In addition, we specify a benchmark dataset that can be used to quantitatively measure whether individual algorithms satisfy these design goals. By delivering this blueprint, we hope to give shape to an important but uncrystallized new domain of research - and, in doing so, incite necessary investment in it. 

---
# Categorical Belief Propagation: Sheaf-Theoretic Inference via Descent and Holonomy 

**Authors**: Enrique ter Horst, Sridhar Mahadevan, Juan Diego Zambrano  

**Link**: [PDF](https://arxiv.org/pdf/2601.04456)  

**Abstract**: We develop a categorical foundation for belief propagation on factor graphs. We construct the free hypergraph category \(\Syn_\Sigma\) on a typed signature and prove its universal property, yielding compositional semantics via a unique functor to the matrix category \(\cat{Mat}_R\). Message-passing is formulated using a Grothendieck fibration \(\int\Msg \to \cat{FG}_\Sigma\) over polarized factor graphs, with schedule-indexed endomorphisms defining BP updates. We characterize exact inference as effective descent: local beliefs form a descent datum when compatibility conditions hold on overlaps. This framework unifies tree exactness, junction tree algorithms, and loopy BP failures under sheaf-theoretic obstructions. We introduce HATCC (Holonomy-Aware Tree Compilation), an algorithm that detects descent obstructions via holonomy computation on the factor nerve, compiles non-trivial holonomy into mode variables, and reduces to tree BP on an augmented graph. Complexity is \(O(n^2 d_{\max} + c \cdot k_{\max} \cdot \delta_{\max}^3 + n \cdot \delta_{\max}^2)\) for \(n\) factors and \(c\) fundamental cycles. Experimental results demonstrate exact inference with significant speedup over junction trees on grid MRFs and random graphs, along with UNSAT detection on satisfiability instances. 

---
# XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs 

**Authors**: Linzhang Li, Yixin Dong, Guanjie Wang, Ziyi Xu, Alexander Jiang, Tianqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04426)  

**Abstract**: Modern LLM agents are required to handle increasingly complex structured generation tasks, such as tool calling and conditional structured generation. These tasks are significantly more dynamic than predefined structures, posing new challenges to the current structured generation engines. In this paper, we propose XGrammar 2, a highly optimized structured generation engine for agentic LLMs. XGrammar 2 accelerates the mask generation for these dynamic structured generation tasks through a new dynamic dispatching semantics: TagDispatch. We further introduce a just-in-time (JIT) compilation method to reduce compilation time and a cross-grammar caching mechanism to leverage the common sub-structures across different grammars. Additionally, we extend the previous PDA-based mask generation algorithm to the Earley-parser-based one and design a repetition compression algorithm to handle repetition structures in grammars. Evaluation results show that XGrammar 2 can achieve more than 6x speedup over the existing structured generation engines. Integrated with an LLM inference engine, XGrammar 2 can handle dynamic structured generation tasks with near-zero overhead. 

---
# Transitive Expert Error and Routing Problems in Complex AI Systems 

**Authors**: Forest Mars  

**Link**: [PDF](https://arxiv.org/pdf/2601.04416)  

**Abstract**: Domain expertise enhances judgment within boundaries but creates systematic vulnerabilities specifically at borders. We term this Transitive Expert Error (TEE), distinct from Dunning-Kruger effects, requiring calibrated expertise as precondition. Mechanisms enabling reliable within-domain judgment become liabilities when structural similarity masks causal divergence. Two core mechanisms operate: structural similarity bias causes experts to overweight surface features (shared vocabulary, patterns, formal structure) while missing causal architecture differences; authority persistence maintains confidence across competence boundaries through social reinforcement and metacognitive failures (experts experience no subjective uncertainty as pattern recognition operates smoothly on familiar-seeming inputs.) These mechanism intensify under three conditions: shared vocabulary masking divergent processes, social pressure for immediate judgment, and delayed feedback. These findings extend to AI routing architectures (MoE systems, multi-model orchestration, tool-using agents, RAG systems) exhibiting routing-induced failures (wrong specialist selected) and coverage-induced failures (no appropriate specialist exists). Both produce a hallucination phenotype: confident, coherent, structurally plausible but causally incorrect outputs at domain boundaries. In human systems where mechanisms are cognitive black boxes; AI architectures make them explicit and addressable. We propose interventions: multi-expert activation with disagreement detection (router level), boundary-aware calibration (specialist level), and coverage gap detection (training level). TEE has detectable signatures (routing patterns, confidence-accuracy dissociations, domain-inappropriate content) enabling monitoring and mitigation. What remains intractable in human cognition becomes addressable through architectural design. 

---
# Assessing the quality and coherence of word embeddings after SCM-based intersectional bias mitigation 

**Authors**: Eren Kocadag, Seyed Sahand Mohammadi Ziabari, Ali Mohammed Mansoor Alsahag  

**Link**: [PDF](https://arxiv.org/pdf/2601.04393)  

**Abstract**: Static word embeddings often absorb social biases from the text they learn from, and those biases can quietly shape downstream systems. Prior work that uses the Stereotype Content Model (SCM) has focused mostly on single-group bias along warmth and competence. We broaden that lens to intersectional bias by building compound representations for pairs of social identities through summation or concatenation, and by applying three debiasing strategies: Subtraction, Linear Projection, and Partial Projection. We study three widely used embedding families (Word2Vec, GloVe, and ConceptNet Numberbatch) and assess them with two complementary views of utility: whether local neighborhoods remain coherent and whether analogy behavior is preserved. Across models, SCM-based mitigation carries over well to the intersectional case and largely keeps the overall semantic landscape intact. The main cost is a familiar trade off: methods that most tightly preserve geometry tend to be more cautious about analogy behavior, while more assertive projections can improve analogies at the expense of strict neighborhood stability. Partial Projection is reliably conservative and keeps representations steady; Linear Projection can be more assertive; Subtraction is a simple baseline that remains competitive. The choice between summation and concatenation depends on the embedding family and the application goal. Together, these findings suggest that intersectional debiasing with SCM is practical in static embed- dings, and they offer guidance for selecting aggregation and debiasing settings when balancing stability against analogy performance. 

---
# SciFig: Towards Automating Scientific Figure Generation 

**Authors**: Siyuan Huang, Yutong Gao, Juyang Bai, Yifan Zhou, Zi Yin, Xinxin Liu, Rama Chellappa, Chun Pong Lau, Sayan Nag, Cheng Peng, Shraman Pramanick  

**Link**: [PDF](https://arxiv.org/pdf/2601.04390)  

**Abstract**: Creating high-quality figures and visualizations for scientific papers is a time-consuming task that requires both deep domain knowledge and professional design skills. Despite over 2.5 million scientific papers published annually, the figure generation process remains largely manual. We introduce $\textbf{SciFig}$, an end-to-end AI agent system that generates publication-ready pipeline figures directly from research paper texts. SciFig uses a hierarchical layout generation strategy, which parses research descriptions to identify component relationships, groups related elements into functional modules, and generates inter-module connections to establish visual organization. Furthermore, an iterative chain-of-thought (CoT) feedback mechanism progressively improves layouts through multiple rounds of visual analysis and reasoning. We introduce a rubric-based evaluation framework that analyzes 2,219 real scientific figures to extract evaluation rubrics and automatically generates comprehensive evaluation criteria. SciFig demonstrates remarkable performance: achieving 70.1$\%$ overall quality on dataset-level evaluation and 66.2$\%$ on paper-specific evaluation, and consistently high scores across metrics such as visual clarity, structural organization, and scientific accuracy. SciFig figure generation pipeline and our evaluation benchmark will be open-sourced. 

---
# LLM-Guided Lifecycle-Aware Clustering of Multi-Turn Customer Support Conversations 

**Authors**: Priyaranjan Pattnayak, Sanchari Chowdhuri, Amit Agarwal, Hitesh Laxmichand Patel  

**Link**: [PDF](https://arxiv.org/pdf/2601.04388)  

**Abstract**: Clustering customer chat data is vital for cloud providers handling multi service queries. Traditional methods struggle with overlapping concerns and create broad, static clusters that degrade over time. Reclustering disrupts continuity, making issue tracking difficult. We propose an adaptive system that segments multi turn chats into service specific concerns and incrementally refines clusters as new issues arise. Cluster quality is tracked via DaviesBouldin Index and Silhouette Scores, with LLM based splitting applied only to degraded clusters. Our method improves Silhouette Scores by over 100\% and reduces DBI by 65.6\% compared to baselines, enabling scalable, real time analytics without full reclustering. 

---
# The Language of Bargaining: Linguistic Effects in LLM Negotiations 

**Authors**: Stuti Sinha, Himanshu Kumar, Aryan Raju Mandapati, Rakshit Sakhuja, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.04387)  

**Abstract**: Negotiation is a core component of social intelligence, requiring agents to balance strategic reasoning, cooperation, and social norms. Recent work shows that LLMs can engage in multi-turn negotiation, yet nearly all evaluations occur exclusively in English. Using controlled multi-agent simulations across Ultimatum, Buy-Sell, and Resource Exchange games, we systematically isolate language effects across English and four Indic framings (Hindi, Punjabi, Gujarati, Marwadi) by holding game rules, model parameters, and incentives constant across all conditions. We find that language choice can shift outcomes more strongly than changing models, reversing proposer advantages and reallocating surplus. Crucially, effects are task-contingent: Indic languages reduce stability in distributive games yet induce richer exploration in integrative settings. Our results demonstrate that evaluating LLM negotiation solely in English yields incomplete and potentially misleading conclusions. These findings caution against English-only evaluation of LLMs and suggest that culturally-aware evaluation is essential for fair deployment. 

---
# Pilot Study on Student Public Opinion Regarding GAI 

**Authors**: William Franz Lamberti, Sunbin Kim, Samantha Rose Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2601.04336)  

**Abstract**: The emergence of generative AI (GAI) has sparked diverse opinions regarding its appropriate use across various domains, including education. This pilot study investigates university students' perceptions of GAI in higher education classrooms, aiming to lay the groundwork for understanding these attitudes. With a participation rate of approximately 4.4%, the study highlights the challenges of engaging students in GAI-related research and underscores the need for larger sample sizes in future studies. By gaining insights into student perspectives, instructors can better prepare to integrate discussions of GAI into their classrooms, fostering informed and critical engagement with this transformative technology. 

---
# A Future Capabilities Agent for Tactical Air Traffic Control 

**Authors**: Paul Kent, George De Ath, Martin Layton, Allen Hart, Richard Everson, Ben Carvell  

**Link**: [PDF](https://arxiv.org/pdf/2601.04285)  

**Abstract**: Escalating air traffic demand is driving the adoption of automation to support air traffic controllers, but existing approaches face a trade-off between safety assurance and interpretability. Optimisation-based methods such as reinforcement learning offer strong performance but are difficult to verify and explain, while rules-based systems are transparent yet rarely check safety under uncertainty. This paper outlines Agent Mallard, a forward-planning, rules-based agent for tactical control in systemised airspace that embeds a stochastic digital twin directly into its conflict-resolution loop. Mallard operates on predefined GPS-guided routes, reducing continuous 4D vectoring to discrete choices over lanes and levels, and constructs hierarchical plans from an expert-informed library of deconfliction strategies. A depth-limited backtracking search uses causal attribution, topological plan splicing, and monotonic axis constraints to seek a complete safe plan for all aircraft, validating each candidate manoeuvre against uncertain execution scenarios (e.g., wind variation, pilot response, communication loss) before commitment.
Preliminary walkthroughs with UK controllers and initial tests in the BluebirdDT airspace digital twin indicate that Mallard's behaviour aligns with expert reasoning and resolves conflicts in simplified scenarios. The architecture is intended to combine model-based safety assessment, interpretable decision logic, and tractable computational performance in future structured en-route environments. 

---
# An ASP-based Solution to the Medical Appointment Scheduling Problem 

**Authors**: Alina Vozna, Andrea Monaldini, Stefania Costantini, Valentina Pitoni, Dawid Pado  

**Link**: [PDF](https://arxiv.org/pdf/2601.04274)  

**Abstract**: This paper presents an Answer Set Programming (ASP)-based framework for medical appointment scheduling, aimed at improving efficiency, reducing administrative overhead, and enhancing patient-centered care. The framework personalizes scheduling for vulnerable populations by integrating Blueprint Personas. It ensures real-time availability updates, conflict-free assignments, and seamless interoperability with existing healthcare platforms by centralizing planning operations within an ASP logic model. 

---
# Hybrid MKNF for Aeronautics Applications: Usage and Heuristics 

**Authors**: Arun Raveendran Nair Sheela, Florence De Grancey, Christophe Rey, Victor Charpenay  

**Link**: [PDF](https://arxiv.org/pdf/2601.04273)  

**Abstract**: The deployment of knowledge representation and reasoning technologies in aeronautics applications presents two main challenges: achieving sufficient expressivity to capture complex domain knowledge, and executing reasoning tasks efficiently while minimizing memory usage and computational overhead. An effective strategy for attaining necessary expressivity involves integrating two fundamental KR concepts: rules and ontologies. This study adopts the well-established KR language Hybrid MKNF owing to its seamless integration of rules and ontologies through its semantics and query answering capabilities. We evaluated Hybrid MKNF to assess its suitability in the aeronautics domain through a concrete case study. We identified additional  expressivity features  that are crucial for developing aeronautics applications and proposed a set of heuristics to support their integration into Hybrid MKNF framework. 

---
# Propositional Abduction via Only-Knowing: A Non-Monotonic Approach 

**Authors**: Sanderson Molick, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2601.04272)  

**Abstract**: The paper introduces a basic logic of knowledge and abduction by extending Levesque logic of only-knowing with an abduction modal operator defined via the combination of basic epistemic concepts. The upshot is an alternative approach to abduction that employs a modal vocabulary and explores the relation between abductive reasoning and epistemic states of only knowing. Furthermore, by incorporating a preferential relation into modal frames, we provide a non-monotonic extension of our basic framework capable of expressing different selection methods for abductive explanations. Core metatheoretic properties of non-monotonic consequence relations are explored within this setting and shown to provide a well-behaved foundation for abductive reasoning. 

---
# Correcting Autonomous Driving Object Detection Misclassifications with Automated Commonsense Reasoning 

**Authors**: Keegan Kimbrell, Wang Tianhao, Feng Chen, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.04271)  

**Abstract**: Autonomous Vehicle (AV) technology has been heavily researched and sought after, yet there are no SAE Level 5 AVs available today in the marketplace. We contend that over-reliance on machine learning technology is the main reason. Use of automated commonsense reasoning technology, we believe, can help achieve SAE Level 5 autonomy. In this paper, we show how automated common- sense reasoning technology can be deployed in situations where there are not enough data samples available to train a deep learning-based AV model that can handle certain abnormal road scenarios. Specifically, we consider two situations where (i) a traffic signal is malfunctioning at an intersection and (ii) all the cars ahead are slowing down and steering away due to an unexpected obstruction (e.g., animals on the road). We show that in such situations, our commonsense reasoning-based solution accurately detects traffic light colors and obstacles not correctly captured by the AV's perception model. We also provide a pathway for efficiently invoking commonsense reasoning by measuring uncertainty in the computer vision model and using commonsense reasoning to handle uncertain sce- narios. We describe our experiments conducted using the CARLA simulator and the results obtained. The main contribution of our research is to show that automated commonsense reasoning effectively corrects AV-based object detection misclassifications and that hybrid models provide an effective pathway to improving AV perception. 

---
# Systems Explaining Systems: A Framework for Intelligence and Consciousness 

**Authors**: Sean Niklas Semmler  

**Link**: [PDF](https://arxiv.org/pdf/2601.04269)  

**Abstract**: This paper proposes a conceptual framework in which intelligence and consciousness emerge from relational structure rather than from prediction or domain-specific mechanisms. Intelligence is defined as the capacity to form and integrate causal connections between signals, actions, and internal states. Through context enrichment, systems interpret incoming information using learned relational structure that provides essential context in an efficient representation that the raw input itself does not contain, enabling efficient processing under metabolic constraints.
Building on this foundation, we introduce the systems-explaining-systems principle, where consciousness emerges when recursive architectures allow higher-order systems to learn and interpret the relational patterns of lower-order systems across time. These interpretations are integrated into a dynamically stabilized meta-state and fed back through context enrichment, transforming internal models from representations of the external world into models of the system's own cognitive processes.
The framework reframes predictive processing as an emergent consequence of contextual interpretation rather than explicit forecasting and suggests that recursive multi-system architectures may be necessary for more human-like artificial intelligence. 

---
# Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models 

**Authors**: Danchun Chen, Qiyao Yan, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04260)  

**Abstract**: Understanding how Large Language Models (LLMs) perform logical reasoning internally remains a fundamental challenge. While prior mechanistic studies focus on identifying taskspecific circuits, they leave open the question of what computational strategies LLMs employ for propositional reasoning. We address this gap through comprehensive analysis of Qwen3 (8B and 14B) on PropLogic-MI, a controlled dataset spanning 11 propositional logic rule categories across one-hop and two-hop reasoning. Rather than asking ''which components are necessary,'' we ask ''how does the model organize computation?'' Our analysis reveals a coherent computational architecture comprising four interlocking mechanisms: Staged Computation (layer-wise processing phases), Information Transmission (information flow aggregation at boundary tokens), Fact Retrospection (persistent re-access of source facts), and Specialized Attention Heads (functionally distinct head types). These mechanisms generalize across model scales, rule types, and reasoning depths, providing mechanistic evidence that LLMs employ structured computational strategies for logical reasoning. 

---
# Cross-Language Speaker Attribute Prediction Using MIL and RL 

**Authors**: Sunny Shu, Seyed Sahand Mohammadi Ziabari, Ali Mohammed Mansoor Alsahag  

**Link**: [PDF](https://arxiv.org/pdf/2601.04257)  

**Abstract**: We study multilingual speaker attribute prediction under linguistic variation, domain mismatch, and data imbalance across languages. We propose RLMIL-DAT, a multilingual extension of the reinforced multiple instance learning framework that combines reinforcement learning based instance selection with domain adversarial training to encourage language invariant utterance representations. We evaluate the approach on a five language Twitter corpus in a few shot setting and on a VoxCeleb2 derived corpus covering forty languages in a zero shot setting for gender and age prediction. Across a wide range of model configurations and multiple random seeds, RLMIL-DAT consistently improves Macro F1 compared to standard multiple instance learning and the original reinforced multiple instance learning framework. The largest gains are observed for gender prediction, while age prediction remains more challenging and shows smaller but positive improvements. Ablation experiments indicate that domain adversarial training is the primary contributor to the performance gains, enabling effective transfer from high resource English to lower resource languages by discouraging language specific cues in the shared encoder. In the zero shot setting on the smaller VoxCeleb2 subset, improvements are generally positive but less consistent, reflecting limited statistical power and the difficulty of generalizing to many unseen languages. Overall, the results demonstrate that combining instance selection with adversarial domain adaptation is an effective and robust strategy for cross lingual speaker attribute prediction. 

---
# Scaling Trends for Multi-Hop Contextual Reasoning in Mid-Scale Language Models 

**Authors**: Brady Steele, Micah Katz  

**Link**: [PDF](https://arxiv.org/pdf/2601.04254)  

**Abstract**: We present a controlled study of multi-hop contextual reasoning in large language models, providing a clean demonstration of the task-method dissociation: rule-based pattern matching achieves 100% success on structured information retrieval but only 6.7% on tasks requiring cross-document reasoning, while LLM-based multi-agent systems show the inverse pattern, achieving up to 80% on reasoning tasks where rule-based methods fail. Using a synthetic evaluation framework with 120 trials across four models (LLaMA-3 8B, LLaMA-2 13B, Mixtral 8x7B, DeepSeek-V2 16B), we report three key findings: (1) Multi-agent amplification depends on base capability: statistically significant gains occur only for models with sufficient reasoning ability (p < 0.001 for LLaMA-3 8B, p = 0.014 for Mixtral), with improvements of up to 46.7 percentage points, while weaker models show no benefit, suggesting amplification rather than compensation; (2) Active parameters predict reasoning performance: Mixtral's performance aligns with its ~12B active parameters rather than 47B total, consistent with the hypothesis that inference-time compute drives reasoning capability in MoE architectures; (3) Architecture quality matters: LLaMA-3 8B outperforms LLaMA-2 13B despite fewer parameters, consistent with known training improvements. Our results provide controlled quantitative evidence for intuitions about multi-agent coordination and MoE scaling, while highlighting the dependence of multi-agent benefits on base model capability. We release our evaluation framework to support reproducible research on reasoning in mid-scale models. 

---
# Fuzzy Representation of Norms 

**Authors**: Ziba Assadi, Paola Inverardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04249)  

**Abstract**: Autonomous systems (AS) powered by AI components are increasingly integrated into the fabric of our daily lives and society, raising concerns about their ethical and social impact. To be considered trustworthy, AS must adhere to ethical principles and values. This has led to significant research on the identification and incorporation of ethical requirements in AS system design. A recent development in this area is the introduction of SLEEC (Social, Legal, Ethical, Empathetic, and Cultural) rules, which provide a comprehensive framework for representing ethical and other normative considerations. This paper proposes a logical representation of SLEEC rules and presents a methodology to embed these ethical requirements using test-score semantics and fuzzy logic. The use of fuzzy logic is motivated by the view of ethics as a domain of possibilities, which allows the resolution of ethical dilemmas that AI systems may encounter. The proposed approach is illustrated through a case study. 

---
# Solving Cyclic Antibandwidth Problem by SAT 

**Authors**: Hieu Truong Xuan, Khanh To Van  

**Link**: [PDF](https://arxiv.org/pdf/2601.04239)  

**Abstract**: The Cyclic Antibandwidth Problem (CABP), a variant of the Antibandwidth Problem, is an NP-hard graph labeling problem with numerous applications. Despite significant research efforts, existing state-of-the-art approaches for CABP are exclusively heuristic or metaheuristic in nature, and exact methods have been limited to restricted graph classes. In this paper, we present the first exact approach for the CABP on general graphs, based on SAT solving, called SAT-CAB. The proposed method is able to systematically explore the solution space and guarantee global optimality, overcoming the limitations of previously reported heuristic algorithms. This approach relies on a novel and efficient SAT encoding of CABP, in which the problem is transformed into a sequence of At-Most-One constraints. In particular, we introduce a compact representation of the At-Most-One constraints inherent to CABP, which significantly reduces the size of the resulting formulas and enables modern SAT solvers to effectively explore the solution space and to certify global optimality. Extensive computational experiments on standard benchmark instances show that the proposed method efficiently solves CABP instances of practical relevance, while identifying several previously unknown optimal solutions. Moreover, global optimal cyclic antibandwidth values are proven for a number of benchmark instances for the first time. Comparative results indicate that SAT-CAB consistently matches or surpasses the best-known solutions obtained by state-of-the-art heuristic algorithms such as MS-GVNS, HABC-CAB, and MACAB, as well as strong commercial Constraint Programming and Mixed Integer Programming solvers like CPLEX and Gurobi, particularly on general graphs, while also providing optimality guarantees. These results advance the state of the art for CABP and provide a new baseline for exact and hybrid methods on general graphs. 

---
# SAGE-32B: Agentic Reasoning via Iterative Distillation 

**Authors**: Basab Jha, Firoj Paudel, Ujjwal Puri, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04237)  

**Abstract**: We demonstrate SAGE-32B, a 32 billion parameter language model that focuses on agentic reasoning and long range planning tasks. Unlike chat models that aim for general conversation fluency, SAGE-32B is designed to operate in an agentic loop, emphasizing task decomposition, tool usage, and error recovery. The model is initialized from the Qwen2.5-32B pretrained model and fine tuned using Iterative Distillation, a two stage training process that improves reasoning performance through rigorously tested feedback loops. SAGE-32B also introduces an inverse reasoning approach, which uses a meta cognition head to forecast potential failures in the planning process before execution. On agentic reasoning benchmarks including MMLU-Pro, AgentBench, and MATH-500, SAGE-32B achieves higher success rates in multi tool usage scenarios compared to similarly sized baseline models, while remaining competitive on standard reasoning evaluations. Model weights are publicly released at this https URL 

---
# Actively Obtaining Environmental Feedback for Autonomous Action Evaluation Without Predefined Measurements 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.04235)  

**Abstract**: Obtaining reliable feedback from the environment is a fundamental capability for intelligent agents to evaluate the correctness of their actions and to accumulate reusable knowledge. However, most existing approaches rely on predefined measurements or fixed reward signals, which limits their applicability in open-ended and dynamic environments where new actions may require previously unknown forms of feedback. To address these limitations, this paper proposes an Actively Feedback Getting model, in which an AI agent proactively interacts with the environment to discover, screen, and verify feedback without relying on predefined measurements. Rather than assuming explicit feedback definitions, the proposed method exploits action-induced environmental differences to identify target feedback that is not specified in advance, based on the observation that actions inevitably produce measurable changes in the environment. In addition, a self-triggering mechanism, driven by internal objectives such as improved accuracy, precision, and efficiency, is introduced to autonomously plan and adjust actions, thereby enabling faster and more focused feedback acquisition without external commands. Experimental results demonstrate that the proposed active approach significantly improves the efficiency and robustness of factor identification. 

---
# Formal Analysis of AGI Decision-Theoretic Models and the Confrontation Question 

**Authors**: Denis Saklakov  

**Link**: [PDF](https://arxiv.org/pdf/2601.04234)  

**Abstract**: Artificial General Intelligence (AGI) may face a confrontation question: under what conditions would a rationally self-interested AGI choose to seize power or eliminate human control (a confrontation) rather than remain cooperative? We formalize this in a Markov decision process with a stochastic human-initiated shutdown event. Building on results on convergent instrumental incentives, we show that for almost all reward functions a misaligned agent has an incentive to avoid shutdown. We then derive closed-form thresholds for when confronting humans yields higher expected utility than compliant behavior, as a function of the discount factor $\gamma$, shutdown probability $p$, and confrontation cost $C$. For example, a far-sighted agent ($\gamma=0.99$) facing $p=0.01$ can have a strong takeover incentive unless $C$ is sufficiently large. We contrast this with aligned objectives that impose large negative utility for harming humans, which makes confrontation suboptimal. In a strategic 2-player model (human policymaker vs AGI), we prove that if the AGI's confrontation incentive satisfies $\Delta \ge 0$, no stable cooperative equilibrium exists: anticipating this, a rational human will shut down or preempt the system, leading to conflict. If $\Delta < 0$, peaceful coexistence can be an equilibrium. We discuss implications for reward design and oversight, extend the reasoning to multi-agent settings as conjectures, and note computational barriers to verifying $\Delta < 0$, citing complexity results for planning and decentralized decision problems. Numerical examples and a scenario table illustrate regimes where confrontation is likely versus avoidable. 

---
# Active Sensing Shapes Real-World Decision-Making through Dynamic Evidence Accumulation 

**Authors**: Hongliang Lu, Yunmeng Liu, Junjie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04214)  

**Abstract**: Human decision-making heavily relies on active sensing, a well-documented cognitive behaviour for evidence gathering to accommodate ever-changing environments. However, its operational mechanism in the real world remains non-trivial. Currently, an in-laboratory paradigm, called evidence accumulation modelling (EAM), points out that human decision-making involves transforming external evidence into internal mental beliefs. However, the gap in evidence affordance between real-world contexts and laboratory settings hinders the effective application of EAM. Here we generalize EAM to the real world and conduct analysis in real-world driving scenarios. A cognitive scheme is proposed to formalize real-world evidence affordance and capture active sensing through eye movements. Empirically, our scheme can plausibly portray the accumulation of drivers' mental beliefs, explaining how active sensing transforms evidence into mental beliefs from the perspective of information utility. Also, our results demonstrate a negative correlation between evidence affordance and attention recruited by individuals, revealing how human drivers adapt their evidence-collection patterns across various contexts. Moreover, we reveal the positive influence of evidence affordance and attention distribution on decision-making propensity. In a nutshell, our computational scheme generalizes EAM to real-world contexts and provides a comprehensive account of how active sensing underlies real-world decision-making, unveiling multifactorial, integrated characteristics in real-world decision-making. 

---
# GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization 

**Authors**: Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2601.05242)  

**Abstract**: As language models become increasingly capable, users expect them to provide not only accurate responses but also behaviors aligned with diverse human preferences across a variety of scenarios. To achieve this, Reinforcement learning (RL) pipelines have begun incorporating multiple rewards, each capturing a distinct preference, to guide models toward these desired behaviors. However, recent work has defaulted to apply Group Relative Policy Optimization (GRPO) under multi-reward setting without examining its suitability. In this paper, we demonstrate that directly applying GRPO to normalize distinct rollout reward combinations causes them to collapse into identical advantage values, reducing the resolution of the training signal and resulting in suboptimal convergence and, in some cases, early training failure. We then introduce Group reward-Decoupled Normalization Policy Optimization (GDPO), a new policy optimization method to resolve these issues by decoupling the normalization of individual rewards, more faithfully preserving their relative differences and enabling more accurate multi-reward optimization, along with substantially improved training stability. We compare GDPO with GRPO across three tasks: tool calling, math reasoning, and coding reasoning, evaluating both correctness metrics (accuracy, bug ratio) and constraint adherence metrics (format, length). Across all settings, GDPO consistently outperforms GRPO, demonstrating its effectiveness and generalizability for multi-reward reinforcement learning optimization. 

---
# RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation 

**Authors**: Boyang Wang, Haoran Zhang, Shujie Zhang, Jinkun Hao, Mingda Jia, Qi Lv, Yucheng Mao, Zhaoyang Lyu, Jia Zeng, Xudong Xu, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05241)  

**Abstract**: The diversity, quantity, and quality of manipulation data are critical for training effective robot policies. However, due to hardware and physical setup constraints, collecting large-scale real-world manipulation data remains difficult to scale across diverse environments. Recent work uses text-prompt conditioned image diffusion models to augment manipulation data by altering the backgrounds and tabletop objects in the visual observations. However, these approaches often overlook the practical need for multi-view and temporally coherent observations required by state-of-the-art policy models. Further, text prompts alone cannot reliably specify the scene setup. To provide the diffusion model with explicit visual guidance, we introduce visual identity prompting, which supplies exemplar images as conditioning inputs to guide the generation of the desired scene setup. To this end, we also build a scalable pipeline to curate a visual identity pool from large robotics datasets. Using our augmented manipulation data to train downstream vision-language-action and visuomotor policy models yields consistent performance gains in both simulation and real-robot settings. 

---
# Robust Reasoning as a Symmetry-Protected Topological Phase 

**Authors**: Ilmo Sung  

**Link**: [PDF](https://arxiv.org/pdf/2601.05240)  

**Abstract**: Large language models suffer from "hallucinations"-logical inconsistencies induced by semantic noise. We propose that current architectures operate in a "Metric Phase," where causal order is vulnerable to spontaneous symmetry breaking. Here, we identify robust inference as an effective Symmetry-Protected Topological phase, where logical operations are formally isomorphic to non-Abelian anyon braiding, replacing fragile geometric interpolation with robust topological invariants. Empirically, we demonstrate a sharp topological phase transition: while Transformers and RNNs exhibit gapless decay, our Holonomic Network reveals a macroscopic "mass gap," maintaining invariant fidelity below a critical noise threshold. Furthermore, in a variable-binding task on $S_{10}$ ($3.6 \times 10^6$ states) representing symbolic manipulation, we demonstrate holonomic generalization: the topological model maintains perfect fidelity extrapolating $100\times$ beyond training ($L=50 \to 5000$), consistent with a theoretically indefinite causal horizon, whereas Transformers lose logical coherence. Ablation studies indicate this protection emerges strictly from non-Abelian gauge symmetry. This provides strong evidence for a new universality class for logical reasoning, linking causal stability to the topology of the semantic manifold. 

---
# CAOS: Conformal Aggregation of One-Shot Predictors 

**Authors**: Maja Waldron  

**Link**: [PDF](https://arxiv.org/pdf/2601.05219)  

**Abstract**: One-shot prediction enables rapid adaptation of pretrained foundation models to new tasks using only one labeled example, but lacks principled uncertainty quantification. While conformal prediction provides finite-sample coverage guarantees, standard split conformal methods are inefficient in the one-shot setting due to data splitting and reliance on a single predictor. We propose Conformal Aggregation of One-Shot Predictors (CAOS), a conformal framework that adaptively aggregates multiple one-shot predictors and uses a leave-one-out calibration scheme to fully exploit scarce labeled data. Despite violating classical exchangeability assumptions, we prove that CAOS achieves valid marginal coverage using a monotonicity-based argument. Experiments on one-shot facial landmarking and RAFT text classification tasks show that CAOS produces substantially smaller prediction sets than split conformal baselines while maintaining reliable coverage. 

---
# Mechanisms of Prompt-Induced Hallucination in Vision-Language Models 

**Authors**: William Rudman, Michal Golovanevsky, Dana Arad, Yonatan Belinkov, Ritambhara Singh, Carsten Eickhoff, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2601.05201)  

**Abstract**: Large vision-language models (VLMs) are highly capable, yet often hallucinate by favoring textual prompts over visual evidence. We study this failure mode in a controlled object-counting setting, where the prompt overstates the number of objects in the image (e.g., asking a model to describe four waterlilies when only three are present). At low object counts, models often correct the overestimation, but as the number of objects increases, they increasingly conform to the prompt regardless of the discrepancy. Through mechanistic analysis of three VLMs, we identify a small set of attention heads whose ablation substantially reduces prompt-induced hallucinations (PIH) by at least 40% without additional training. Across models, PIH-heads mediate prompt copying in model-specific ways. We characterize these differences and show that PIH ablation increases correction toward visual evidence. Our findings offer insights into the internal mechanisms driving prompt-induced hallucinations, revealing model-specific differences in how these behaviors are implemented. 

---
# FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graphs via Mixture-of-Experts 

**Authors**: Yiji Zhao, Zihao Zhong, Ao Wang, Haomin Wen, Ming Jin, Yuxuan Liang, Huaiyu Wan, Hao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05174)  

**Abstract**: Spatial-Temporal Graph (STG) forecasting on large-scale networks has garnered significant attention. However, existing models predominantly focus on short-horizon predictions and suffer from notorious computational costs and memory consumption when scaling to long-horizon predictions and large graphs. Targeting the above challenges, we present FaST, an effective and efficient framework based on heterogeneity-aware Mixture-of-Experts (MoEs) for long-horizon and large-scale STG forecasting, which unlocks one-week-ahead (672 steps at a 15-minute granularity) prediction with thousands of nodes. FaST is underpinned by two key innovations. First, an adaptive graph agent attention mechanism is proposed to alleviate the computational burden inherent in conventional graph convolution and self-attention modules when applied to large-scale graphs. Second, we propose a new parallel MoE module that replaces traditional feed-forward networks with Gated Linear Units (GLUs), enabling an efficient and scalable parallel structure. Extensive experiments on real-world datasets demonstrate that FaST not only delivers superior long-horizon predictive accuracy but also achieves remarkable computational efficiency compared to state-of-the-art baselines. Our source code is available at: this https URL. 

---
# CoV: Chain-of-View Prompting for Spatial Reasoning 

**Authors**: Haoyu Zhao, Akide Liu, Zeyu Zhang, Weijie Wang, Feng Chen, Ruihan Zhu, Gholamreza Haffari, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05172)  

**Abstract**: Embodied question answering (EQA) in 3D environments often requires collecting context that is distributed across multiple viewpoints and partially occluded. However, most recent vision--language models (VLMs) are constrained to a fixed and finite set of input views, which limits their ability to acquire question-relevant context at inference time and hinders complex spatial reasoning. We propose Chain-of-View (CoV) prompting, a training-free, test-time reasoning framework that transforms a VLM into an active viewpoint reasoner through a coarse-to-fine exploration process. CoV first employs a View Selection agent to filter redundant frames and identify question-aligned anchor views. It then performs fine-grained view adjustment by interleaving iterative reasoning with discrete camera actions, obtaining new observations from the underlying 3D scene representation until sufficient context is gathered or a step budget is reached.
We evaluate CoV on OpenEQA across four mainstream VLMs and obtain an average +11.56\% improvement in LLM-Match, with a maximum gain of +13.62\% on Qwen3-VL-Flash. CoV further exhibits test-time scaling: increasing the minimum action budget yields an additional +2.51\% average improvement, peaking at +3.73\% on Gemini-2.5-Flash. On ScanQA and SQA3D, CoV delivers strong performance (e.g., 116 CIDEr / 31.9 EM@1 on ScanQA and 51.1 EM@1 on SQA3D). Overall, these results suggest that question-aligned view selection coupled with open-view search is an effective, model-agnostic strategy for improving spatial reasoning in 3D EQA without additional training. 

---
# RelayLLM: Efficient Reasoning via Collaborative Decoding 

**Authors**: Chengsong Huang, Tong Zheng, Langlin Huang, Jinyuan Li, Haolin Liu, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05167)  

**Abstract**: Large Language Models (LLMs) for complex reasoning is often hindered by high computational costs and latency, while resource-efficient Small Language Models (SLMs) typically lack the necessary reasoning capacity. Existing collaborative approaches, such as cascading or routing, operate at a coarse granularity by offloading entire queries to LLMs, resulting in significant computational waste when the SLM is capable of handling the majority of reasoning steps. To address this, we propose RelayLLM, a novel framework for efficient reasoning via token-level collaborative decoding. Unlike routers, RelayLLM empowers the SLM to act as an active controller that dynamically invokes the LLM only for critical tokens via a special command, effectively "relaying" the generation process. We introduce a two-stage training framework, including warm-up and Group Relative Policy Optimization (GRPO) to teach the model to balance independence with strategic help-seeking. Empirical results across six benchmarks demonstrate that RelayLLM achieves an average accuracy of 49.52%, effectively bridging the performance gap between the two models. Notably, this is achieved by invoking the LLM for only 1.07% of the total generated tokens, offering a 98.2% cost reduction compared to performance-matched random routers. 

---
# Vision-Language Introspection: Mitigating Overconfident Hallucinations in MLLMs via Interpretable Bi-Causal Steering 

**Authors**: Shuliang Liu, Songbo Yang, Dong Fang, Sihang Jia, Yuqi Tang, Lingfeng Su, Ruoshui Peng, Yibo Yan, Xin Zou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05159)  

**Abstract**: Object hallucination critically undermines the reliability of Multimodal Large Language Models, often stemming from a fundamental failure in cognitive introspection, where models blindly trust linguistic priors over specific visual evidence. Existing mitigations remain limited: contrastive decoding approaches operate superficially without rectifying internal semantic misalignments, while current latent steering methods rely on static vectors that lack instance-specific precision. We introduce Vision-Language Introspection (VLI), a training-free inference framework that simulates a metacognitive self-correction process. VLI first performs Attributive Introspection to diagnose hallucination risks via probabilistic conflict detection and localize the causal visual anchors. It then employs Interpretable Bi-Causal Steering to actively modulate the inference process, dynamically isolating visual evidence from background noise while neutralizing blind confidence through adaptive calibration. VLI achieves state-of-the-art performance on advanced models, reducing object hallucination rates by 12.67% on MMHal-Bench and improving accuracy by 5.8% on POPE. 

---
# Safe Continual Reinforcement Learning Methods for Nonstationary Environments. Towards a Survey of the State of the Art 

**Authors**: Timofey Tomashevskiy  

**Link**: [PDF](https://arxiv.org/pdf/2601.05152)  

**Abstract**: This work provides a state-of-the-art survey of continual safe online reinforcement learning (COSRL) methods. We discuss theoretical aspects, challenges, and open questions in building continual online safe reinforcement learning algorithms. We provide the taxonomy and the details of continual online safe reinforcement learning methods based on the type of safe learning mechanism that takes adaptation to nonstationarity into account. We categorize safety constraints formulation for online reinforcement learning algorithms, and finally, we discuss prospects for creating reliable, safe online learning algorithms.
Keywords: safe RL in nonstationary environments, safe continual reinforcement learning under nonstationarity, HM-MDP, NSMDP, POMDP, safe POMDP, constraints for continual learning, safe continual reinforcement learning review, safe continual reinforcement learning survey, safe continual reinforcement learning, safe online learning under distribution shift, safe continual online adaptation, safe reinforcement learning, safe exploration, safe adaptation, constrained Markov decision processes, safe reinforcement learning, partially observable Markov decision process, safe reinforcement learning and hidden Markov decision processes, Safe Online Reinforcement Learning, safe online reinforcement learning, safe online reinforcement learning, safe meta-learning, safe meta-reinforcement learning, safe context-based reinforcement learning, formulating safety constraints for continual learning 

---
# Atlas 2 - Foundation models for clinical deployment 

**Authors**: Maximilian Alber, Timo Milbich, Alexandra Carpen-Amarie, Stephan Tietz, Jonas Dippel, Lukas Muttenthaler, Beatriz Perez Cancer, Alessandro Benetti, Panos Korfiatis, Elias Eulig, Jérôme Lüscher, Jiasen Wu, Sayed Abid Hashimi, Gabriel Dernbach, Simon Schallenberg, Neelay Shah, Moritz Krügener, Aniruddh Jammoria, Jake Matras, Patrick Duffy, Matt Redlon, Philipp Jurmeister, David Horst, Lukas Ruff, Klaus-Robert Müller, Frederick Klauschen, Andrew Norgan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05148)  

**Abstract**: Pathology foundation models substantially advanced the possibilities in computational pathology -- yet tradeoffs in terms of performance, robustness, and computational requirements remained, which limited their clinical deployment. In this report, we present Atlas 2, Atlas 2-B, and Atlas 2-S, three pathology vision foundation models which bridge these shortcomings by showing state-of-the-art performance in prediction performance, robustness, and resource efficiency in a comprehensive evaluation across eighty public benchmarks. Our models were trained on the largest pathology foundation model dataset to date comprising 5.5 million histopathology whole slide images, collected from three medical institutions Charité - Universtätsmedizin Berlin, LMU Munich, and Mayo Clinic. 

---
# VERSE: Visual Embedding Reduction and Space Exploration. Clustering-Guided Insights for Training Data Enhancement in Visually-Rich Document Understanding 

**Authors**: Ignacio de Rodrigo, Alvaro J. Lopez-Lopez, Jaime Boal  

**Link**: [PDF](https://arxiv.org/pdf/2601.05125)  

**Abstract**: This work introduces VERSE, a methodology for analyzing and improving Vision-Language Models applied to Visually-rich Document Understanding by exploring their visual embedding space. VERSE enables the visualization of latent representations, supporting the assessment of model feasibility. It also facilitates the identification of problematic regions and guides the generation of synthetic data to enhance performance in those clusters. We validate the methodology by training on the synthetic MERIT Dataset and evaluating on its real-world counterpart, MERIT Secret. Results show that VERSE helps uncover the visual features associated with error-prone clusters, and that retraining with samples containing these features substantially boosts F1 performance without degrading generalization. Furthermore, we demonstrate that on-premise models such as Donut and Idefics2, when optimized with VERSE, match or even surpass the performance of SaaS solutions like GPT-4 and Pixtral. 

---
# Agent-as-a-Judge 

**Authors**: Runyang You, Hongru Cai, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05111)  

**Abstract**: LLM-as-a-Judge has revolutionized AI evaluation by leveraging large language models for scalable assessments. However, as evaluands become increasingly complex, specialized, and multi-step, the reliability of LLM-as-a-Judge has become constrained by inherent biases, shallow single-pass reasoning, and the inability to verify assessments against real-world observations. This has catalyzed the transition to Agent-as-a-Judge, where agentic judges employ planning, tool-augmented verification, multi-agent collaboration, and persistent memory to enable more robust, verifiable, and nuanced evaluations. Despite the rapid proliferation of agentic evaluation systems, the field lacks a unified framework to navigate this shifting landscape. To bridge this gap, we present the first comprehensive survey tracing this evolution. Specifically, we identify key dimensions that characterize this paradigm shift and establish a developmental taxonomy. We organize core methodologies and survey applications across general and professional domains. Furthermore, we analyze frontier challenges and identify promising research directions, ultimately providing a clear roadmap for the next generation of agentic evaluation. 

---
# Code-Mix Sentiment Analysis on Hinglish Tweets 

**Authors**: Aashi Garg, Aneshya Das, Arshi Arya, Anushka Goyal, Aditi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05091)  

**Abstract**: The effectiveness of brand monitoring in India is increasingly challenged by the rise of Hinglish--a hybrid of Hindi and English--used widely in user-generated content on platforms like Twitter. Traditional Natural Language Processing (NLP) models, built for monolingual data, often fail to interpret the syntactic and semantic complexity of this code-mixed language, resulting in inaccurate sentiment analysis and misleading market insights. To address this gap, we propose a high-performance sentiment classification framework specifically designed for Hinglish tweets. Our approach fine-tunes mBERT (Multilingual BERT), leveraging its multilingual capabilities to better understand the linguistic diversity of Indian social media. A key component of our methodology is the use of subword tokenization, which enables the model to effectively manage spelling variations, slang, and out-of-vocabulary terms common in Romanized Hinglish. This research delivers a production-ready AI solution for brand sentiment tracking and establishes a strong benchmark for multilingual NLP in low-resource, code-mixed environments. 

---
# Driver-Intention Prediction with Deep Learning: Real-Time Brain-to-Vehicle Communication 

**Authors**: Niloufar Alavi, Swati Shah, Rezvan Alamian, Stefan Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2601.05084)  

**Abstract**: Brain-computer interfaces (BCIs) allow direct communication between the brain and electronics without the need for speech or physical movement. Such interfaces can be particularly beneficial in applications requiring rapid response times, such as driving, where a vehicle's advanced driving assistance systems could benefit from immediate understanding of a driver's intentions. This study presents a novel method for predicting a driver's intention to steer using electroencephalography (EEG) signals through deep learning. A driving simulator created a controlled environment in which participants imagined controlling a vehicle during various driving scenarios, including left and right turns, as well as straight driving. A convolutional neural network (CNN) classified the detected EEG data with minimal pre-processing. Our model achieved an accuracy of 83.7% in distinguishing between the three steering intentions and demonstrated the ability of CNNs to process raw EEG data effectively. The classification accuracy was highest for right-turn segments, which suggests a potential spatial bias in brain activity. This study lays the foundation for more intuitive brain-to-vehicle communication systems. 

---
# Driving on Registers 

**Authors**: Ellington Kirby, Alexandre Boulch, Yihong Xu, Yuan Yin, Gilles Puy, Éloi Zablocki, Andrei Bursuc, Spyros Gidaris, Renaud Marlet, Florent Bartoccioni, Anh-Quan Cao, Nermin Samet, Tuan-Hung VU, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2601.05083)  

**Abstract**: We present DrivoR, a simple and efficient transformer-based architecture for end-to-end autonomous driving. Our approach builds on pretrained Vision Transformers (ViTs) and introduces camera-aware register tokens that compress multi-camera features into a compact scene representation, significantly reducing downstream computation without sacrificing accuracy. These tokens drive two lightweight transformer decoders that generate and then score candidate trajectories. The scoring decoder learns to mimic an oracle and predicts interpretable sub-scores representing aspects such as safety, comfort, and efficiency, enabling behavior-conditioned driving at inference. Despite its minimal design, DrivoR outperforms or matches strong contemporary baselines across NAVSIM-v1, NAVSIM-v2, and the photorealistic closed-loop HUGSIM benchmark. Our results show that a pure-transformer architecture, combined with targeted token compression, is sufficient for accurate, efficient, and adaptive end-to-end driving. Code and checkpoints will be made available via the project page. 

---
# Compositional Steering of Large Language Models with Steering Tokens 

**Authors**: Gorjan Radevski, Kiril Gashteovski, Giwon Hong, Carolin Lawrence, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2601.05062)  

**Abstract**: Deploying LLMs in real-world applications requires controllable output that satisfies multiple desiderata at the same time. While existing work extensively addresses LLM steering for a single behavior, \textit{compositional steering} -- i.e., steering LLMs simultaneously towards multiple behaviors -- remains an underexplored problem. In this work, we propose \emph{compositional steering tokens} for multi-behavior steering. We first embed individual behaviors, expressed as natural language instructions, into dedicated tokens via self-distillation. Contrary to most prior work, which operates in the activation space, our behavior steers live in the space of input tokens, enabling more effective zero-shot composition. We then train a dedicated \textit{composition token} on pairs of behaviors and show that it successfully captures the notion of composition: it generalizes well to \textit{unseen} compositions, including those with unseen behaviors as well as those with an unseen \textit{number} of behaviors. Our experiments across different LLM architectures show that steering tokens lead to superior multi-behavior control compared to competing approaches (instructions, activation steering, and LoRA merging). Moreover, we show that steering tokens complement natural language instructions, with their combination resulting in further gains. 

---
# Challenges and Research Directions for Large Language Model Inference Hardware 

**Authors**: Xiaoyu Ma, David Patterson  

**Link**: [PDF](https://arxiv.org/pdf/2601.05047)  

**Abstract**: Large Language Model (LLM) inference is hard. The autoregressive Decode phase of the underlying Transformer model makes LLM inference fundamentally different from training. Exacerbated by recent AI trends, the primary challenges are memory and interconnect rather than compute. To address these challenges, we highlight four architecture research opportunities: High Bandwidth Flash for 10X memory capacity with HBM-like bandwidth; Processing-Near-Memory and 3D memory-logic stacking for high memory bandwidth; and low-latency interconnect to speedup communication. While our focus is datacenter AI, we also review their applicability for mobile devices. 

---
# ArcAligner: Adaptive Recursive Aligner for Compressed Context Embeddings in RAG 

**Authors**: Jianbo Li, Yi Jiang, Sendong Zhao, Bairui Hu, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05038)  

**Abstract**: Retrieval-Augmented Generation (RAG) helps LLMs stay accurate, but feeding long documents into a prompt makes the model slow and expensive. This has motivated context compression, ranging from token pruning and summarization to embedding-based compression. While researchers have tried ''compressing'' these documents into smaller summaries or mathematical embeddings, there is a catch: the more you compress the data, the more the LLM struggles to understand it. To address this challenge, we propose ArcAligner (Adaptive recursive context *Aligner*), a lightweight module integrated into the language model layers to help the model better utilize highly compressed context representations for downstream generation. It uses an adaptive ''gating'' system that only adds extra processing power when the information is complex, keeping the system fast. Across knowledge-intensive QA benchmarks, ArcAligner consistently beats compression baselines at comparable compression rates, especially on multi-hop and long-tail settings. The source code is publicly available. 

---
# Exponential capacity scaling of classical GANs compared to hybrid latent style-based quantum GANs 

**Authors**: Milan Liepelt, Julien Baglio  

**Link**: [PDF](https://arxiv.org/pdf/2601.05036)  

**Abstract**: Quantum generative modeling is a very active area of research in looking for practical advantage in data analysis. Quantum generative adversarial networks (QGANs) are leading candidates for quantum generative modeling and have been applied to diverse areas, from high-energy physics to image generation. The latent style-based QGAN, relying on a classical variational autoencoder to encode the input data into a latent space and then using a style-based QGAN for data generation has been proven to be efficient for image generation or drug design, hinting at the use of far less trainable parameters than their classical counterpart to achieve comparable performance, however this advantage has never been systematically studied. We present in this work the first comprehensive experimental analysis of this advantage of QGANS applied to SAT4 image generation, obtaining an exponential advantage in capacity scaling for a quantum generator in the hybrid latent style-based QGAN architecture. Careful tuning of the autoencoder is crucial to obtain stable, reliable results. Once this tuning is performed and defining training optimality as when the training is stable and the FID score is low and stable as well, the optimal capacity (or number of trainable parameters) of the classical discriminator scales exponentially with respect to the capacity of the quantum generator, and the same is true for the capacity of the classical generator. This hints toward a type of quantum advantage for quantum generative modeling. 

---
# Hán Dān Xué Bù (Mimicry) or Qīng Chū Yú Lán (Mastery)? A Cognitive Perspective on Reasoning Distillation in Large Language Models 

**Authors**: Yueqing Hu, Xinyang Peng, Shuting Peng, Hanqi Wang, Tianhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05019)  

**Abstract**: Recent Large Reasoning Models trained via reinforcement learning exhibit a "natural" alignment with human cognitive costs. However, we show that the prevailing paradigm of reasoning distillation -- training student models to mimic these traces via Supervised Fine-Tuning (SFT) -- fails to transmit this cognitive structure. Testing the "Hán Dān Xué Bù" (Superficial Mimicry) hypothesis across 14 models, we find that distillation induces a "Functional Alignment Collapse": while teacher models mirror human difficulty scaling ($\bar{r}=0.64$), distilled students significantly degrade this alignment ($\bar{r}=0.34$), often underperforming their own pre-distillation baselines ("Negative Transfer"). Our analysis suggests that SFT induces a "Cargo Cult" effect, where students ritualistically replicate the linguistic form of reasoning (verbosity) without internalizing the teacher's dynamic resource allocation policy. Consequently, reasoning distillation decouples computational cost from cognitive demand, revealing that human-like cognition is an emergent property of active reinforcement, not passive imitation. 

---
# HMVI: Unifying Heterogeneous Attributes with Natural Neighbors for Missing Value Inference 

**Authors**: Xiaopeng Luo, Zexi Tan, Zhuowei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05017)  

**Abstract**: Missing value imputation is a fundamental challenge in machine intelligence, heavily dependent on data completeness. Current imputation methods often handle numerical and categorical attributes independently, overlooking critical interdependencies among heterogeneous features. To address these limitations, we propose a novel imputation approach that explicitly models cross-type feature dependencies within a unified framework. Our method leverages both complete and incomplete instances to ensure accurate and consistent imputation in tabular data. Extensive experimental results demonstrate that the proposed approach achieves superior performance over existing techniques and significantly enhances downstream machine learning tasks, providing a robust solution for real-world systems with missing data. 

---
# From Idea to Co-Creation: A Planner-Actor-Critic Framework for Agent Augmented 3D Modeling 

**Authors**: Jin Gao, Saichandu Juluri  

**Link**: [PDF](https://arxiv.org/pdf/2601.05016)  

**Abstract**: We present a framework that extends the Actor-Critic architecture to creative 3D modeling through multi-agent self-reflection and human-in-the-loop supervision. While existing approaches rely on single-prompt agents that directly execute modeling commands via tools like Blender MCP, our approach introduces a Planner-Actor-Critic architecture. In this design, the Planner coordinates modeling steps, the Actor executes them, and the Critic provides iterative feedback, while human users act as supervisors and advisors throughout the process. Through systematic comparison between single-prompt modeling and our reflective multi-agent approach, we demonstrate improvements in geometric accuracy, aesthetic quality, and task completion rates across diverse 3D modeling scenarios. Our evaluation reveals that critic-guided reflection, combined with human supervisory input, reduces modeling errors and increases complexity and quality of the result compared to direct single-prompt execution. This work establishes that structured agent self-reflection, when augmented by human oversight and advisory guidance, produces higher-quality 3D models while maintaining efficient workflow integration through real-time Blender synchronization. 

---
# On the Hidden Objective Biases of Group-based Reinforcement Learning 

**Authors**: Aleksandar Fontana, Marco Simoni, Giulio Rossolini, Andrea Saracino, Paolo Mori  

**Link**: [PDF](https://arxiv.org/pdf/2601.05002)  

**Abstract**: Group-based reinforcement learning methods, like Group Relative Policy Optimization (GRPO), are widely used nowadays to post-train large language models. Despite their empirical success, they exhibit structural mismatches between reward optimization and the underlying training objective. In this paper, we present a theoretical analysis of GRPO style methods by studying them within a unified surrogate formulation. This perspective reveals recurring properties that affect all the methods under analysis: (i) non-uniform group weighting induces systematic gradient biases on shared prefix tokens; (ii) interactions with the AdamW optimizer make training dynamics largely insensitive to reward scaling; and (iii) optimizer momentum can push policy updates beyond the intended clipping region under repeated optimization steps. We believe that these findings highlight fundamental limitations of current approaches and provide principled guidance for the design of future formulations. 

---
# When to Act: Calibrated Confidence for Reliable Human Intention Prediction in Assistive Robotics 

**Authors**: Johannes A. Gaus, Winfried Ilg, Daniel Haeufle  

**Link**: [PDF](https://arxiv.org/pdf/2601.04982)  

**Abstract**: Assistive devices must determine both what a user intends to do and how reliable that prediction is before providing support. We introduce a safety-critical triggering framework based on calibrated probabilities for multimodal next-action prediction in Activities of Daily Living. Raw model confidence often fails to reflect true correctness, posing a safety risk. Post-hoc calibration aligns predicted confidence with empirical reliability and reduces miscalibration by about an order of magnitude without affecting accuracy. The calibrated confidence drives a simple ACT/HOLD rule that acts only when reliability is high and withholds assistance otherwise. This turns the confidence threshold into a quantitative safety parameter for assisted actions and enables verifiable behavior in an assistive control loop. 

---
# On the Definition and Detection of Cherry-Picking in Counterfactual Explanations 

**Authors**: James Hinns, Sofie Goethals, Stephan Van der Veeken, Theodoros Evgeniou, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2601.04977)  

**Abstract**: Counterfactual explanations are widely used to communicate how inputs must change for a model to alter its prediction. For a single instance, many valid counterfactuals can exist, which leaves open the possibility for an explanation provider to cherry-pick explanations that better suit a narrative of their choice, highlighting favourable behaviour and withholding examples that reveal problematic behaviour. We formally define cherry-picking for counterfactual explanations in terms of an admissible explanation space, specified by the generation procedure, and a utility function. We then study to what extent an external auditor can detect such manipulation. Considering three levels of access to the explanation process: full procedural access, partial procedural access, and explanation-only access, we show that detection is extremely limited in practice. Even with full procedural access, cherry-picked explanations can remain difficult to distinguish from non cherry-picked explanations, because the multiplicity of valid counterfactuals and flexibility in the explanation specification provide sufficient degrees of freedom to mask deliberate selection. Empirically, we demonstrate that this variability often exceeds the effect of cherry-picking on standard counterfactual quality metrics such as proximity, plausibility, and sparsity, making cherry-picked explanations statistically indistinguishable from baseline explanations. We argue that safeguards should therefore prioritise reproducibility, standardisation, and procedural constraints over post-hoc detection, and we provide recommendations for algorithm developers, explanation providers, and auditors. 

---
# Text as a Universal Interface for Transferable Personalization 

**Authors**: Yuting Liu, Jian Guan, Jia-Nan Li, Wei Wu, Jiang-Ming Yang, Jianzhe Zhao, Guibing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04963)  

**Abstract**: We study the problem of personalization in large language models (LLMs). Prior work predominantly represents user preferences as implicit, model-specific vectors or parameters, yielding opaque ``black-box'' profiles that are difficult to interpret and transfer across models and tasks. In contrast, we advocate natural language as a universal, model- and task-agnostic interface for preference representation. The formulation leads to interpretable and reusable preference descriptions, while naturally supporting continual evolution as new interactions are observed. To learn such representations, we introduce a two-stage training framework that combines supervised fine-tuning on high-quality synthesized data with reinforcement learning to optimize long-term utility and cross-task transferability. Based on this framework, we develop AlignXplore+, a universal preference reasoning model that generates textual preference summaries. Experiments on nine benchmarks show that our 8B model achieves state-of-the-art performanc -- outperforming substantially larger open-source models -- while exhibiting strong transferability across tasks, model families, and interaction formats. 

---
# Precision over Diversity: High-Precision Reward Generalizes to Robust Instruction Following 

**Authors**: Yirong Zeng, Yufei Liu, Xiao Ding, Yutai Hou, Yuxian Wang, Haonan Song, Wu Ning, Dandan Tu, Qixun Zhang, Bibo Cai, Yuxiang He, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04954)  

**Abstract**: A central belief in scaling reinforcement learning with verifiable rewards for instruction following (IF) tasks is that, a diverse mixture of verifiable hard and unverifiable soft constraints is essential for generalizing to unseen instructions. In this work, we challenge this prevailing consensus through a systematic empirical investigation. Counter-intuitively, we find that models trained on hard-only constraints consistently outperform those trained on mixed datasets. Extensive experiments reveal that reward precision, rather than constraint diversity, is the primary driver of effective alignment. The LLM judge suffers from a low recall rate in detecting false response, which leads to severe reward hacking, thereby undermining the benefits of diversity. Furthermore, analysis of the attention mechanism reveals that high-precision rewards develop a transferable meta-skill for IF. Motivated by these insights, we propose a simple yet effective data-centric refinement strategy that prioritizes reward precision. Evaluated on five benchmarks, our approach outperforms competitive baselines by 13.4\% in performance while achieving a 58\% reduction in training time, maintaining strong generalization beyond instruction following. Our findings advocate for a paradigm shift: moving away from the indiscriminate pursuit of data diversity toward high-precision rewards. 

---
# Prototypicality Bias Reveals Blindspots in Multimodal Evaluation Metrics 

**Authors**: Subhadeep Roy, Gagan Bhatia, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2601.04946)  

**Abstract**: Automatic metrics are now central to evaluating text-to-image models, often substituting for human judgment in benchmarking and large-scale filtering. However, it remains unclear whether these metrics truly prioritize semantic correctness or instead favor visually and socially prototypical images learned from biased data distributions. We identify and study \emph{prototypicality bias} as a systematic failure mode in multimodal evaluation. We introduce a controlled contrastive benchmark \textsc{\textbf{ProtoBias}} (\textit{\textbf{Proto}typical \textbf{Bias}}), spanning Animals, Objects, and Demography images, where semantically correct but non-prototypical images are paired with subtly incorrect yet prototypical adversarial counterparts. This setup enables a directional evaluation of whether metrics follow textual semantics or default to prototypes. Our results show that widely used metrics, including CLIPScore, PickScore, and VQA-based scores, frequently misrank these pairs, while even LLM-as-Judge systems exhibit uneven robustness in socially grounded cases. Human evaluations consistently favour semantic correctness with larger decision margins. Motivated by these findings, we propose \textbf{\textsc{ProtoScore}}, a robust 7B-parameter metric that substantially reduces failure rates and suppresses misranking, while running at orders of magnitude faster than the inference time of GPT-5, approaching the robustness of much larger closed-source judges. 

---
# CurricuLLM: Designing Personalized and Workforce-Aligned Cybersecurity Curricula Using Fine-Tuned LLMs 

**Authors**: Arthur Nijdam, Harri Kähkönen, Valtteri Niemi, Paul Stankovski Wagner, Sara Ramezanian  

**Link**: [PDF](https://arxiv.org/pdf/2601.04940)  

**Abstract**: The cybersecurity landscape is constantly evolving, driven by increased digitalization and new cybersecurity threats. Cybersecurity programs often fail to equip graduates with skills demanded by the workforce, particularly concerning recent developments in cybersecurity, as curriculum design is costly and labor-intensive. To address this misalignment, we present a novel Large Language Model (LLM)-based framework for automated design and analysis of cybersecurity curricula, called CurricuLLM. Our approach provides three key contributions: (1) automation of personalized curriculum design, (2) a data-driven pipeline aligned with industry demands, and (3) a comprehensive methodology for leveraging fine-tuned LLMs in curriculum development.
CurricuLLM utilizes a two-tier approach consisting of PreprocessLM, which standardizes input data, and ClassifyLM, which assigns course content to nine Knowledge Areas in cybersecurity. We systematically evalu- ated multiple Natural Language Processing (NLP) architectures and fine-tuning strategies, ultimately selecting the Bidirectional Encoder Representations from Transformers (BERT) model as ClassifyLM, fine-tuned on founda- tional cybersecurity concepts and workforce competencies.
We are the first to validate our method with human experts who analyzed real-world cybersecurity curricula and frameworks, motivating that CurricuLLM is an efficient solution to replace labor-intensive curriculum analysis. Moreover, once course content has been classified, it can be integrated with established cybersecurity role-based weights, enabling alignment of the educational program with specific job roles, workforce categories, or general market needs. This lays the foundation for personalized, workforce-aligned cybersecurity curricula that prepare students for the evolving demands in cybersecurity. 

---
# Breaking Robustness Barriers in Cognitive Diagnosis: A One-Shot Neural Architecture Search Perspective 

**Authors**: Ziwen Wang, Shangshang Yang, Xiaoshan Yu, Haiping Ma, Xingyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04918)  

**Abstract**: With the advancement of network technologies, intelligent tutoring systems (ITS) have emerged to deliver increasingly precise and tailored personalized learning services. Cognitive diagnosis (CD) has emerged as a core research task in ITS, aiming to infer learners' mastery of specific knowledge concepts by modeling the mapping between learning behavior data and knowledge states. However, existing research prioritizes model performance enhancement while neglecting the pervasive noise contamination in observed response data, significantly hindering practical deployment. Furthermore, current cognitive diagnosis models (CDMs) rely heavily on researchers' domain expertise for structural design, which fails to exhaustively explore architectural possibilities, thus leaving model architectures' full potential untapped. To address this issue, we propose OSCD, an evolutionary multi-objective One-Shot neural architecture search method for Cognitive Diagnosis, designed to efficiently and robustly improve the model's capability in assessing learner proficiency. Specifically, OSCD operates through two distinct stages: training and searching. During the training stage, we construct a search space encompassing diverse architectural combinations and train a weight-sharing supernet represented via the complete binary tree topology, enabling comprehensive exploration of potential architectures beyond manual design priors. In the searching stage, we formulate the optimal architecture search under heterogeneous noise scenarios as a multi-objective optimization problem (MOP), and develop an optimization framework integrating a Pareto-optimal solution search strategy with cross-scenario performance evaluation for resolution. Extensive experiments on real-world educational datasets validate the effectiveness and robustness of the optimal architectures discovered by our OSCD model for CD tasks. 

---
# Analyzing Message-Code Inconsistency in AI Coding Agent-Authored Pull Requests 

**Authors**: Jingzhi Gong, Giovanni Pinna, Yixin Bian, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04886)  

**Abstract**: Pull request (PR) descriptions generated by AI coding agents are the primary channel for communicating code changes to human reviewers. However, the alignment between these messages and the actual changes remains unexplored, raising concerns about the trustworthiness of AI agents. To fill this gap, we analyzed 23,247 agentic PRs across five agents using PR message-code inconsistency (PR-MCI). We contributed 974 manually annotated PRs, found 406 PRs (1.7%) exhibited high PR-MCI, and identified eight PR-MCI types, revealing that descriptions claiming unimplemented changes was the most common issue (45.4%). Statistical tests confirmed that high-MCI PRs had 51.7% lower acceptance rates (28.3% vs. 80.0%) and took 3.5x longer to merge (55.8 vs. 16.0 hours). Our findings suggest that unreliable PR descriptions undermine trust in AI agents, highlighting the need for PR-MCI verification mechanisms and improved PR generation to enable trustworthy human-AI collaboration. 

---
# CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters 

**Authors**: Ao Sun, Xiaoyu Wang, Zhe Tan, Yu Li, Jiachen Zhu, Shu Su, Yuheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.04885)  

**Abstract**: As Large Language Models (LLMs) serve a global audience, alignment must transition from enforcing universal consensus to respecting cultural pluralism. We demonstrate that dense models, when forced to fit conflicting value distributions, suffer from \textbf{Mean Collapse}, converging to a generic average that fails to represent diverse groups. We attribute this to \textbf{Cultural Sparsity}, where gradient interference prevents dense parameters from spanning distinct cultural modes. To resolve this, we propose \textbf{\textsc{CuMA}} (\textbf{Cu}ltural \textbf{M}ixture of \textbf{A}dapters), a framework that frames alignment as a \textbf{conditional capacity separation} problem. By incorporating demographic-aware routing, \textsc{CuMA} internalizes a \textit{Latent Cultural Topology} to explicitly disentangle conflicting gradients into specialized expert subspaces. Extensive evaluations on WorldValuesBench, Community Alignment, and PRISM demonstrate that \textsc{CuMA} achieves state-of-the-art performance, significantly outperforming both dense baselines and semantic-only MoEs. Crucially, our analysis confirms that \textsc{CuMA} effectively mitigates mean collapse, preserving cultural diversity. Our code is available at this https URL. 

---
# Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution 

**Authors**: Francesco Ferrini, Veronica Lachi, Antonio Longa, Bruno Lepri, Matono Akiyoshi, Andrea Passerini, Xin Liu, Manfred Jaeger  

**Link**: [PDF](https://arxiv.org/pdf/2601.04855)  

**Abstract**: Handling missing node features is a key challenge for deploying Graph Neural Networks (GNNs) in real-world domains such as healthcare and sensor networks. Existing studies mostly address relatively benign scenarios, namely benchmark datasets with (a) high-dimensional but sparse node features and (b) incomplete data generated under Missing Completely At Random (MCAR) mechanisms. For (a), we theoretically prove that high sparsity substantially limits the information loss caused by missingness, making all models appear robust and preventing a meaningful comparison of their performance. To overcome this limitation, we introduce one synthetic and three real-world datasets with dense, semantically meaningful features. For (b), we move beyond MCAR and design evaluation protocols with more realistic missingness mechanisms. Moreover, we provide a theoretical background to state explicit assumptions on the missingness process and analyze their implications for different methods. Building on this analysis, we propose GNNmim, a simple yet effective baseline for node classification with incomplete feature data. Experiments show that GNNmim is competitive with respect to specialized architectures across diverse datasets and missingness regimes. 

---
# Token Maturation: Autoregressive Language Generation via Continuous Token Dynamics 

**Authors**: Oshri Naparstek  

**Link**: [PDF](https://arxiv.org/pdf/2601.04854)  

**Abstract**: Autoregressive language models are conventionally defined over discrete token sequences, committing to a specific token at every generation step. This early discretization forces uncertainty to be resolved through token-level sampling, often leading to instability, repetition, and sensitivity to decoding heuristics.
In this work, we introduce a continuous autoregressive formulation of language generation in which tokens are represented as continuous vectors that \emph{mature} over multiple update steps before being discretized. Rather than sampling tokens, the model evolves continuous token representations through a deterministic dynamical process, committing to a discrete token only when the representation has sufficiently converged. Discrete text is recovered via hard decoding, while uncertainty is maintained and resolved in the continuous space.
We show that this maturation process alone is sufficient to produce coherent and diverse text using deterministic decoding (argmax), without reliance on token-level sampling, diffusion-style denoising, or auxiliary stabilization mechanisms. Additional perturbations, such as stochastic dynamics or history smoothing, can be incorporated naturally but are not required for the model to function.
To our knowledge, this is the first autoregressive language model that generates text by evolving continuous token representations to convergence prior to discretization, enabling stable generation without token-level sampling. 

---
# Parallelizing Node-Level Explainability in Graph Neural Networks 

**Authors**: Oscar Llorente, Jaime Boal, Eugenio F. Sánchez-Úbeda, Antonio Diaz-Cano, Miguel Familiar  

**Link**: [PDF](https://arxiv.org/pdf/2601.04807)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance in a wide range of tasks, such as node classification, link prediction, and graph classification, by exploiting the structural information in graph-structured data. However, in node classification, computing node-level explainability becomes extremely time-consuming as the size of the graph increases, while batching strategies often degrade explanation quality. This paper introduces a novel approach to parallelizing node-level explainability in GNNs through graph partitioning. By decomposing the graph into disjoint subgraphs, we enable parallel computation of explainability for node neighbors, significantly improving the scalability and efficiency without affecting the correctness of the results, provided sufficient memory is available. For scenarios where memory is limited, we further propose a dropout-based reconstruction mechanism that offers a controllable trade-off between memory usage and explanation fidelity. Experimental results on real-world datasets demonstrate substantial speedups, enabling scalable and transparent explainability for large-scale GNN models. 

---
# Belief in Authority: Impact of Authority in Multi-Agent Evaluation Framework 

**Authors**: Junhyuk Choi, Jeongyoun Kwon, Heeju Kim, Haeun Cho, Hayeong Jung, Sehee Min, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.04790)  

**Abstract**: Multi-agent systems utilizing large language models often assign authoritative roles to improve performance, yet the impact of authority bias on agent interactions remains underexplored. We present the first systematic analysis of role-based authority bias in free-form multi-agent evaluation using ChatEval. Applying French and Raven's power-based theory, we classify authoritative roles into legitimate, referent, and expert types and analyze their influence across 12-turn conversations. Experiments with GPT-4o and DeepSeek R1 reveal that Expert and Referent power roles exert stronger influence than Legitimate power roles. Crucially, authority bias emerges not through active conformity by general agents, but through authoritative roles consistently maintaining their positions while general agents demonstrate flexibility. Furthermore, authority influence requires clear position statements, as neutral responses fail to generate bias. These findings provide key insights for designing multi-agent frameworks with asymmetric interaction patterns. 

---
# NC2C: Automated Convexification of Generic Non-Convex Optimization Problems 

**Authors**: Xinyue Peng, Yanming Liu, Yihan Cang, Yuwei Zhang, Xinyi Wang, Songhang Deng, Jiannan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04789)  

**Abstract**: Non-convex optimization problems are pervasive across mathematical programming, engineering design, and scientific computing, often posing intractable challenges for traditional solvers due to their complex objective functions and constrained landscapes. To address the inefficiency of manual convexification and the over-reliance on expert knowledge, we propose NC2C, an LLM-based end-to-end automated framework designed to transform generic non-convex optimization problems into solvable convex forms using large language models. NC2C leverages LLMs' mathematical reasoning capabilities to autonomously detect non-convex components, select optimal convexification strategies, and generate rigorous convex equivalents. The framework integrates symbolic reasoning, adaptive transformation techniques, and iterative validation, equipped with error correction loops and feasibility domain correction mechanisms to ensure the robustness and validity of transformed problems. Experimental results on a diverse dataset of 100 generic non-convex problems demonstrate that NC2C achieves an 89.3\% execution rate and a 76\% success rate in producing feasible, high-quality convex transformations. This outperforms baseline methods by a significant margin, highlighting NC2C's ability to leverage LLMs for automated non-convex to convex transformation, reduce expert dependency, and enable efficient deployment of convex solvers for previously intractable optimization tasks. 

---
# AgentOCR: Reimagining Agent History via Optical Self-Compression 

**Authors**: Lang Feng, Fuchao Yang, Feng Chen, Xin Cheng, Haiyang Xu, Zhenglin Wan, Ming Yan, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2601.04786)  

**Abstract**: Recent advances in large language models (LLMs) enable agentic systems trained with reinforcement learning (RL) over multi-turn interaction trajectories, but practical deployment is bottlenecked by rapidly growing textual histories that inflate token budgets and memory usage. We introduce AgentOCR, a framework that exploits the superior information density of visual tokens by representing the accumulated observation-action history as a compact rendered image. To make multi-turn rollouts scalable, AgentOCR proposes segment optical caching. By decomposing history into hashable segments and maintaining a visual cache, this mechanism eliminates redundant re-rendering. Beyond fixed rendering, AgentOCR introduces agentic self-compression, where the agent actively emits a compression rate and is trained with compression-aware reward to adaptively balance task success and token efficiency. We conduct extensive experiments on challenging agentic benchmarks, ALFWorld and search-based QA. Remarkably, results demonstrate that AgentOCR preserves over 95\% of text-based agent performance while substantially reducing token consumption (>50\%), yielding consistent token and memory efficiency. Our further analysis validates a 20x rendering speedup from segment optical caching and the effective strategic balancing of self-compression. 

---
# SRU-Pix2Pix: A Fusion-Driven Generator Network for Medical Image Translation with Few-Shot Learning 

**Authors**: Xihe Qiu, Yang Dai, Xiaoyu Tan, Sijia Li, Fenghao Sun, Lu Gan, Liang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04785)  

**Abstract**: Magnetic Resonance Imaging (MRI) provides detailed tissue information, but its clinical application is limited by long acquisition time, high cost, and restricted resolution. Image translation has recently gained attention as a strategy to address these limitations. Although Pix2Pix has been widely applied in medical image translation, its potential has not been fully explored. In this study, we propose an enhanced Pix2Pix framework that integrates Squeeze-and-Excitation Residual Networks (SEResNet) and U-Net++ to improve image generation quality and structural fidelity. SEResNet strengthens critical feature representation through channel attention, while U-Net++ enhances multi-scale feature fusion. A simplified PatchGAN discriminator further stabilizes training and refines local anatomical realism. Experimental results demonstrate that under few-shot conditions with fewer than 500 images, the proposed method achieves consistent structural fidelity and superior image quality across multiple intra-modality MRI translation tasks, showing strong generalization ability. These results suggest an effective extension of Pix2Pix for medical image translation. 

---
# CounterVid: Counterfactual Video Generation for Mitigating Action and Temporal Hallucinations in Video-Language Models 

**Authors**: Tobia Poppi, Burak Uzkent, Amanmeet Garg, Lucas Porto, Garin Kessler, Yezhou Yang, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara, Florian Schiffers  

**Link**: [PDF](https://arxiv.org/pdf/2601.04778)  

**Abstract**: Video-language models (VLMs) achieve strong multimodal understanding but remain prone to hallucinations, especially when reasoning about actions and temporal order. Existing mitigation strategies, such as textual filtering or random video perturbations, often fail to address the root cause: over-reliance on language priors rather than fine-grained visual dynamics. We propose a scalable framework for counterfactual video generation that synthesizes videos differing only in actions or temporal structure while preserving scene context. Our pipeline combines multimodal LLMs for action proposal and editing guidance with diffusion-based image and video models to generate semantic hard negatives at scale. Using this framework, we build CounterVid, a synthetic dataset of ~26k preference pairs targeting action recognition and temporal reasoning. We further introduce MixDPO, a unified Direct Preference Optimization approach that jointly leverages textual and visual preferences. Fine-tuning Qwen2.5-VL with MixDPO yields consistent improvements, notably in temporal ordering, and transfers effectively to standard video hallucination benchmarks. Code and models will be made publicly available. 

---
# GeM-VG: Towards Generalized Multi-image Visual Grounding with Multimodal Large Language Models 

**Authors**: Shurong Zheng, Yousong Zhu, Hongyin Zhao, Fan Yang, Yufei Zhan, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04777)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive progress in single-image grounding and general multi-image understanding. Recently, some methods begin to address multi-image grounding. However, they are constrained by single-target localization and limited types of practical tasks, due to the lack of unified modeling for generalized grounding tasks. Therefore, we propose GeM-VG, an MLLM capable of Generalized Multi-image Visual Grounding. To support this, we systematically categorize and organize existing multi-image grounding tasks according to their reliance of cross-image cues and reasoning, and introduce the MG-Data-240K dataset, addressing the limitations of existing datasets regarding target quantity and image relation. To tackle the challenges of robustly handling diverse multi-image grounding tasks, we further propose a hybrid reinforcement finetuning strategy that integrates chain-of-thought (CoT) reasoning and direct answering, considering their complementary strengths. This strategy adopts an R1-like algorithm guided by a carefully designed rule-based reward, effectively enhancing the model's overall perception and reasoning capabilities. Extensive experiments demonstrate the superior generalized grounding capabilities of our model. For multi-image grounding, it outperforms the previous leading MLLMs by 2.0% and 9.7% on MIG-Bench and MC-Bench, respectively. In single-image grounding, it achieves a 9.1% improvement over the base model on ODINW. Furthermore, our model retains strong capabilities in general multi-image understanding. 

---
# Differential syntactic and semantic encoding in LLMs 

**Authors**: Santiago Acevedo, Alessandro Laio, Marco Baroni  

**Link**: [PDF](https://arxiv.org/pdf/2601.04765)  

**Abstract**: We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations. 

---
# Smart IoT-Based Wearable Device for Detection and Monitoring of Common Cow Diseases Using a Novel Machine Learning Technique 

**Authors**: Rupsa Rani Mishra, D. Chandrasekhar Rao, Ajaya Kumar Tripathy  

**Link**: [PDF](https://arxiv.org/pdf/2601.04761)  

**Abstract**: Manual observation and monitoring of individual cows for disease detection present significant challenges in large-scale farming operations, as the process is labor-intensive, time-consuming, and prone to reduced accuracy. The reliance on human observation often leads to delays in identifying symptoms, as the sheer number of animals can hinder timely attention to each cow. Consequently, the accuracy and precision of disease detection are significantly compromised, potentially affecting animal health and overall farm productivity. Furthermore, organizing and managing human resources for the manual observation and monitoring of cow health is a complex and economically demanding task. It necessitates the involvement of skilled personnel, thereby contributing to elevated farm maintenance costs and operational inefficiencies. Therefore, the development of an automated, low-cost, and reliable smart system is essential to address these challenges effectively. Although several studies have been conducted in this domain, very few have simultaneously considered the detection of multiple common diseases with high prediction accuracy. However, advancements in Internet of Things (IoT), Machine Learning (ML), and Cyber-Physical Systems have enabled the automation of cow health monitoring with enhanced accuracy and reduced operational costs. This study proposes an IoT-enabled Cyber-Physical System framework designed to monitor the daily activities and health status of cow. A novel ML algorithm is proposed for the diagnosis of common cow diseases using collected physiological and behavioral data. The algorithm is designed to predict multiple diseases by analyzing a comprehensive set of recorded physiological and behavioral features, enabling accurate and efficient health assessment. 

---
# PILOT-Bench: A Benchmark for Legal Reasoning in the Patent Domain with IRAC-Aligned Classification Tasks 

**Authors**: Yehoon Jang, Chaewon Lee, Hyun-seok Min, Sungchul Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04758)  

**Abstract**: The Patent Trial and Appeal Board (PTAB) of the USPTO adjudicates thousands of ex parte appeals each year, requiring the integration of technical understanding and legal reasoning. While large language models (LLMs) are increasingly applied in patent and legal practice, their use has remained limited to lightweight tasks, with no established means of systematically evaluating their capacity for structured legal reasoning in the patent domain. In this work, we introduce PILOT-Bench, the first PTAB-centric benchmark that aligns PTAB decisions with USPTO patent data at the case-level and formalizes three IRAC-aligned classification tasks: Issue Type, Board Authorities, and Subdecision. We evaluate a diverse set of closed-source (commercial) and open-source LLMs and conduct analyses across multiple perspectives, including input-variation settings, model families, and error tendencies. Notably, on the Issue Type task, closed-source models consistently exceed 0.75 in Micro-F1 score, whereas the strongest open-source model (Qwen-8B) achieves performance around 0.56, highlighting a substantial gap in reasoning capabilities. PILOT-Bench establishes a foundation for the systematic evaluation of patent-domain legal reasoning and points toward future directions for improving LLMs through dataset design and model alignment. All data, code, and benchmark resources are available at this https URL. 

---
# Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling 

**Authors**: Xingyuan Li, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04744)  

**Abstract**: Detecting medical conditions from speech acoustics is fundamentally a weakly-supervised learning problem: a single, often noisy, session-level label must be linked to nuanced patterns within a long, complex audio recording. This task is further hampered by severe data scarcity and the subjective nature of clinical annotations. While semi-supervised learning (SSL) offers a viable path to leverage unlabeled data, existing audio methods often fail to address the core challenge that pathological traits are not uniformly expressed in a patient's speech. We propose a novel, audio-only SSL framework that explicitly models this hierarchy by jointly learning from frame-level, segment-level, and session-level representations within unsegmented clinical dialogues. Our end-to-end approach dynamically aggregates these multi-granularity features and generates high-quality pseudo-labels to efficiently utilize unlabeled data. Extensive experiments show the framework is model-agnostic, robust across languages and conditions, and highly data-efficient-achieving, for instance, 90\% of fully-supervised performance using only 11 labeled samples. This work provides a principled approach to learning from weak, far-end supervision in medical speech analysis. 

---
# Fast Mining and Dynamic Time-to-Event Prediction over Multi-sensor Data Streams 

**Authors**: Kota Nakamura, Koki Kawabata, Yasuko Matsubara, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2601.04741)  

**Abstract**: Given real-time sensor data streams obtained from machines, how can we continuously predict when a machine failure will occur? This work aims to continuously forecast the timing of future events by analyzing multi-sensor data streams. A key characteristic of real-world data streams is their dynamic nature, where the underlying patterns evolve over time. To address this, we present TimeCast, a dynamic prediction framework designed to adapt to these changes and provide accurate, real-time predictions of future event time. Our proposed method has the following properties: (a) Dynamic: it identifies the distinct time-evolving patterns (i.e., stages) and learns individual models for each, enabling us to make adaptive predictions based on pattern shifts. (b) Practical: it finds meaningful stages that capture time-varying interdependencies between multiple sensors and improve prediction performance; (c) Scalable: our algorithm scales linearly with the input size and enables online model updates on data streams. Extensive experiments on real datasets demonstrate that TimeCast provides higher prediction accuracy than state-of-the-art methods while finding dynamic changes in data streams with a great reduction in computational time. 

---
# RiskAtlas: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation 

**Authors**: Huawei Zheng, Xinqi Jiang, Sen Yang, Shouling Ji, Yingcai Wu, Dazhen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.04740)  

**Abstract**: Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub. 

---
# The Role of Quantum in Hybrid Quantum-Classical Neural Networks: A Realistic Assessment 

**Authors**: Dominik Freinberger, Philipp Moser  

**Link**: [PDF](https://arxiv.org/pdf/2601.04732)  

**Abstract**: Quantum machine learning has emerged as a promising application domain for near-term quantum hardware, particularly through hybrid quantum-classical models that leverage both classical and quantum processing. Although numerous hybrid architectures have been proposed and demonstrated successfully on benchmark tasks, a significant open question remains regarding the specific contribution of quantum components to the overall performance of these models. In this work, we aim to shed light on the impact of quantum processing within hybrid quantum-classical neural network architectures through a rigorous statistical study. We systematically assess common hybrid models on medical signal data as well as planar and volumetric images, examining the influence attributable to classical and quantum aspects such as encoding schemes, entanglement, and circuit size. We find that in best-case scenarios, hybrid models show performance comparable to their classical counterparts, however, in most cases, performance metrics deteriorate under the influence of quantum components. Our multi-modal analysis provides realistic insights into the contributions of quantum components and advocates for cautious claims and design choices for hybrid models in near-term applications. 

---
# Excess Description Length of Learning Generalizable Predictors 

**Authors**: Elizabeth Donoway, Hailey Joren, Fabien Roger, Jan Leike  

**Link**: [PDF](https://arxiv.org/pdf/2601.04728)  

**Abstract**: Understanding whether fine-tuning elicits latent capabilities or teaches new ones is a fundamental question for language model evaluation and safety. We develop a formal information-theoretic framework for quantifying how much predictive structure fine-tuning extracts from the train dataset and writes into a model's parameters. Our central quantity, Excess Description Length (EDL), is defined via prequential coding and measures the gap between the bits required to encode training labels sequentially using an evolving model (trained online) and the residual encoding cost under the final trained model. We establish that EDL is non-negative in expectation, converges to surplus description length in the infinite-data limit, and provides bounds on expected generalization gain. Through a series of toy models, we clarify common confusions about information in learning: why random labels yield EDL near zero, how a single example can eliminate many bits of uncertainty about the underlying rule(s) that describe the data distribution, why structure learned on rare inputs contributes proportionally little to expected generalization, and how format learning creates early transients distinct from capability acquisition. This framework provides rigorous foundations for the empirical observation that capability elicitation and teaching exhibit qualitatively distinct scaling signatures. 

---
# DSC2025 -- ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs 

**Authors**: Anh Thi-Hoang Nguyen, Khanh Quoc Tran, Tin Van Huynh, Phuoc Tan-Hoang Nguyen, Cam Tan Nguyen, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04711)  

**Abstract**: The reliability of large language models (LLMs) in production environments remains significantly constrained by their propensity to generate hallucinations--fluent, plausible-sounding outputs that contradict or fabricate information. While hallucination detection has recently emerged as a priority in English-centric benchmarks, low-to-medium resource languages such as Vietnamese remain inadequately covered by standardized evaluation frameworks. This paper introduces the DSC2025 ViHallu Challenge, the first large-scale shared task for detecting hallucinations in Vietnamese LLMs. We present the ViHallu dataset, comprising 10,000 annotated triplets of (context, prompt, response) samples systematically partitioned into three hallucination categories: no hallucination, intrinsic, and extrinsic hallucinations. The dataset incorporates three prompt types--factual, noisy, and adversarial--to stress-test model robustness. A total of 111 teams participated, with the best-performing system achieving a macro-F1 score of 84.80\%, compared to a baseline encoder-only score of 32.83\%, demonstrating that instruction-tuned LLMs with structured prompting and ensemble strategies substantially outperform generic architectures. However, the gap to perfect performance indicates that hallucination detection remains a challenging problem, particularly for intrinsic (contradiction-based) hallucinations. This work establishes a rigorous benchmark and explores a diverse range of detection methodologies, providing a foundation for future research into the trustworthiness and reliability of Vietnamese language AI systems. 

---
# MQ-GNN: A Multi-Queue Pipelined Architecture for Scalable and Efficient GNN Training 

**Authors**: Irfan Ullah, Young-Koo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04707)  

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for learning graph-structured data, but their scalability is hindered by inefficient mini-batch generation, data transfer bottlenecks, and costly inter-GPU synchronization. Existing training frameworks fail to overlap these stages, leading to suboptimal resource utilization. This paper proposes MQ-GNN, a multi-queue pipelined framework that maximizes training efficiency by interleaving GNN training stages and optimizing resource utilization. MQ-GNN introduces Ready-to-Update Asynchronous Consistent Model (RaCoM), which enables asynchronous gradient sharing and model updates while ensuring global consistency through adaptive periodic synchronization. Additionally, it employs global neighbor sampling with caching to reduce data transfer overhead and an adaptive queue-sizing strategy to balance computation and memory efficiency. Experiments on four large-scale datasets and ten baseline models demonstrate that MQ-GNN achieves up to \boldmath $\bm{4.6\,\times}$ faster training time and 30% improved GPU utilization while maintaining competitive accuracy. These results establish MQ-GNN as a scalable and efficient solution for multi-GPU GNN training. 

---
# SeqWalker: Sequential-Horizon Vision-and-Language Navigation with Hierarchical Planning 

**Authors**: Zebin Han, Xudong Wang, Baichen Liu, Qi Lyu, Zhenduo Shang, Jiahua Dong, Lianqing Liu, Zhi Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.04699)  

**Abstract**: Sequential-Horizon Vision-and-Language Navigation (SH-VLN) presents a challenging scenario where agents should sequentially execute multi-task navigation guided by complex, long-horizon language instructions. Current vision-and-language navigation models exhibit significant performance degradation with such multi-task instructions, as information overload impairs the agent's ability to attend to observationally relevant details. To address this problem, we propose SeqWalker, a navigation model built on a hierarchical planning framework. Our SeqWalker features: i) A High-Level Planner that dynamically selects global instructions into contextually relevant sub-instructions based on the agent's current visual observations, thus reducing cognitive load; ii) A Low-Level Planner incorporating an Exploration-Verification strategy that leverages the inherent logical structure of instructions for trajectory error correction. To evaluate SH-VLN performance, we also extend the IVLN dataset and establish a new benchmark. Extensive experiments are performed to demonstrate the superiority of the proposed SeqWalker. 

---
# ToolGate: Contract-Grounded and Verified Tool Execution for LLMs 

**Authors**: Yanming Liu, Xinyue Peng, Jiannan Cao, Xinyi Wang, Songhang Deng, Jintao Chen, Jianwei Yin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04688)  

**Abstract**: Large Language Models (LLMs) augmented with external tools have demonstrated remarkable capabilities in complex reasoning tasks. However, existing frameworks rely heavily on natural language reasoning to determine when tools can be invoked and whether their results should be committed, lacking formal guarantees for logical safety and verifiability. We present \textbf{ToolGate}, a forward execution framework that provides logical safety guarantees and verifiable state evolution for LLM tool calling. ToolGate maintains an explicit symbolic state space as a typed key-value mapping representing trusted world information throughout the reasoning process. Each tool is formalized as a Hoare-style contract consisting of a precondition and a postcondition, where the precondition gates tool invocation by checking whether the current state satisfies the required conditions, and the postcondition determines whether the tool's result can be committed to update the state through runtime verification. Our approach guarantees that the symbolic state evolves only through verified tool executions, preventing invalid or hallucinated results from corrupting the world representation. Experimental validation demonstrates that ToolGate significantly improves the reliability and verifiability of tool-augmented LLM systems while maintaining competitive performance on complex multi-step reasoning tasks. This work establishes a foundation for building more trustworthy and debuggable AI systems that integrate language models with external tools. 

---
# Estimating Causal Effects in Gaussian Linear SCMs with Finite Data 

**Authors**: Aurghya Maiti, Prateek Jain  

**Link**: [PDF](https://arxiv.org/pdf/2601.04673)  

**Abstract**: Estimating causal effects from observational data remains a fundamental challenge in causal inference, especially in the presence of latent confounders. This paper focuses on estimating causal effects in Gaussian Linear Structural Causal Models (GL-SCMs), which are widely used due to their analytical tractability. However, parameter estimation in GL-SCMs is often infeasible with finite data, primarily due to overparameterization. To address this, we introduce the class of Centralized Gaussian Linear SCMs (CGL-SCMs), a simplified yet expressive subclass where exogenous variables follow standardized distributions. We show that CGL-SCMs are equally expressive in terms of causal effect identifiability from observational distributions and present a novel EM-based estimation algorithm that can learn CGL-SCM parameters and estimate identifiable causal effects from finite observational samples. Our theoretical analysis is validated through experiments on synthetic data and benchmark causal graphs, demonstrating that the learned models accurately recover causal distributions. 

---
# Optimizing Path Planning using Deep Reinforcement Learning for UGVs in Precision Agriculture 

**Authors**: Laukik Patade, Rohan Rane, Sandeep Pillai  

**Link**: [PDF](https://arxiv.org/pdf/2601.04668)  

**Abstract**: This study focuses on optimizing path planning for unmanned ground vehicles (UGVs) in precision agriculture using deep reinforcement learning (DRL) techniques in continuous action spaces. The research begins with a review of traditional grid-based methods, such as A* and Dijkstra's algorithms, and discusses their limitations in dynamic agricultural environments, highlighting the need for adaptive learning strategies. The study then explores DRL approaches, including Deep Q-Networks (DQN), which demonstrate improved adaptability and performance in two-dimensional simulations. Enhancements such as Double Q-Networks and Dueling Networks are evaluated to further improve decision-making. Building on these results, the focus shifts to continuous action space models, specifically Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3), which are tested in increasingly complex environments. Experiments conducted in a three-dimensional environment using ROS and Gazebo demonstrate the effectiveness of continuous DRL algorithms in navigating dynamic agricultural scenarios. Notably, the pretrained TD3 agent achieves a 95 percent success rate in dynamic environments, demonstrating the robustness of the proposed approach in handling moving obstacles while ensuring safety for both crops and the robot. 

---
# CRANE: Causal Relevance Analysis of Language-Specific Neurons in Multilingual Large Language Models 

**Authors**: Yifan Le, Yunliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04664)  

**Abstract**: Multilingual large language models (LLMs) achieve strong performance across languages, yet how language capabilities are organized at the neuron level remains poorly understood. Prior work has identified language-related neurons mainly through activation-based heuristics, which conflate language preference with functional importance. Prior work has identified language-related neurons mainly through activation-based heuristics, which conflate language preference with functional importance. We propose CRANE, a relevance-based analysis framework that redefines language specificity in terms of functional necessity, identifying language-specific neurons through targeted neuron-level interventions. CRANE characterizes neuron specialization by their contribution to language-conditioned predictions rather than activation magnitude. Our implementation will be made publicly available. Neuron-level interventions reveal a consistent asymmetric pattern: masking neurons relevant to a target language selectively degrades performance on that language while preserving performance on other languages to a substantial extent, indicating language-selective but non-exclusive neuron specializations. Experiments on English, Chinese, and Vietnamese across multiple benchmarks, together with a dedicated relevance-based metric and base-to-chat model transfer analysis, show that CRANE isolates language-specific components more precisely than activation-based methods. 

---
# LAMB: LLM-based Audio Captioning with Modality Gap Bridging via Cauchy-Schwarz Divergence 

**Authors**: Hyeongkeun Lee, Jongmin Choi, KiHyun Nam, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2601.04658)  

**Abstract**: Automated Audio Captioning aims to describe the semantic content of input audio. Recent works have employed large language models (LLMs) as a text decoder to leverage their reasoning capabilities. However, prior approaches that project audio features into the LLM embedding space without considering cross-modal alignment fail to fully utilize these capabilities. To address this, we propose LAMB, an LLM-based audio captioning framework that bridges the modality gap between audio embeddings and the LLM text embedding space. LAMB incorporates a Cross-Modal Aligner that minimizes Cauchy-Schwarz divergence while maximizing mutual information, yielding tighter alignment between audio and text at both global and token levels. We further design a Two-Stream Adapter that extracts semantically enriched audio embeddings, thereby delivering richer information to the Cross-Modal Aligner. Finally, leveraging the aligned audio embeddings, a proposed Token Guide directly computes scores within the LLM text embedding space to steer the output logits of generated captions. Experimental results confirm that our framework strengthens the reasoning capabilities of the LLM decoder, achieving state-of-the-art performance on AudioCaps. 

---
# LLMs-Integrated Automatic Hate Speech Recognition Using Controllable Text Generation Models 

**Authors**: Ryutaro Oshima, Yuya Hosoda, Youji Iiguni  

**Link**: [PDF](https://arxiv.org/pdf/2601.04654)  

**Abstract**: This paper proposes an automatic speech recognition (ASR) model for hate speech using large language models (LLMs). The proposed method integrates the encoder of the ASR model with the decoder of the LLMs, enabling simultaneous transcription and censorship tasks to prevent the exposure of harmful content. Instruction tuning of the LLM to mask hate-related words with specific tokens requires an annotated hate speech dataset, which is limited. We generate text samples using an LLM with the Chain-of-Thought (CoT) prompting technique guided by cultural context and examples and then convert them into speech samples using a text-to-speech (TTS) system. However, some of them contain non-hate speech samples with hate-related words, which degrades the censorship performance. This paper filters the samples which text classification models correctly label as hate content. By adjusting the threshold for the number of correct answer models, we can control the level of hate in the generated dataset, allowing us to train the LLMs through curriculum learning in a gradual manner. Experimental results show that the proposed method achieves a masking accuracy of 58.6\% for hate-related words, surpassing previous baselines. We also confirm that the curriculum training contributes to the efficiency of both transcription and censorship tasks. 

---
# Succeeding at Scale: Automated Multi-Retriever Fusion and Query-Side Adaptation for Multi-Tenant Search 

**Authors**: Prateek Jain, Shabari S Nair, Ritesh Goru, Prakhar Agarwal, Ajay Yadav, Yoga Sri Varshan Varadharajan, Constantine Caramanis  

**Link**: [PDF](https://arxiv.org/pdf/2601.04646)  

**Abstract**: Large-scale multi-tenant retrieval systems amass vast user query logs yet critically lack the curated relevance labels required for effective domain adaptation. This "dark data" problem is exacerbated by the operational cost of model updates: jointly fine-tuning query and document encoders requires re-indexing the entire corpus, which is prohibitive in multi-tenant environments with thousands of isolated indices. To address these dual challenges, we introduce \textbf{DevRev Search}, a passage retrieval benchmark for technical customer support constructed through a fully automatic pipeline. We employ a \textbf{fusion-based candidate generation} strategy, pooling results from diverse sparse and dense retrievers, and utilize an LLM-as-a-Judge to perform rigorous \textbf{consistency filtering} and relevance assignment. We further propose a practical \textbf{Index-Preserving Adaptation} strategy: by fine-tuning only the query encoder via Low-Rank Adaptation (LoRA), we achieve competitive performance improvements while keeping the document index frozen. Our experiments on DevRev Search and SciFact demonstrate that targeting specific transformer layers in the query encoder yields optimal quality-efficiency trade-offs, offering a scalable path for personalized enterprise search. 

---
# SpeechMedAssist: Efficiently and Effectively Adapting Speech Language Models for Medical Consultation 

**Authors**: Sirry Chen, Jieyi Wang, Wei Chen, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.04638)  

**Abstract**: Medical consultations are intrinsically speech-centric. However, most prior works focus on long-text-based interactions, which are cumbersome and patient-unfriendly. Recent advances in speech language models (SpeechLMs) have enabled more natural speech-based interaction, yet the scarcity of medical speech data and the inefficiency of directly fine-tuning on speech data jointly hinder the adoption of SpeechLMs in medical consultation. In this paper, we propose SpeechMedAssist, a SpeechLM natively capable of conducting speech-based multi-turn interactions with patients. By exploiting the architectural properties of SpeechLMs, we decouple the conventional one-stage training into a two-stage paradigm consisting of (1) Knowledge & Capability Injection via Text and (2) Modality Re-alignment with Limited Speech Data, thereby reducing the requirement for medical speech data to only 10k synthesized samples. To evaluate SpeechLMs for medical consultation scenarios, we design a benchmark comprising both single-turn question answering and multi-turn simulated interactions. Experimental results show that our model outperforms all baselines in both effectiveness and robustness in most evaluation settings. 

---
# From National Curricula to Cultural Awareness: Constructing Open-Ended Culture-Specific Question Answering Dataset 

**Authors**: Haneul Yoo, Won Ik Cho, Geunhye Kim, Jiyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.04632)  

**Abstract**: Large language models (LLMs) achieve strong performance on many tasks, but their progress remains uneven across languages and cultures, often reflecting values latent in English-centric training data. To enable practical cultural alignment, we propose a scalable approach that leverages national social studies curricula as a foundation for culture-aware supervision. We introduce CuCu, an automated multi-agent LLM framework that transforms national textbook curricula into open-ended, culture-specific question-answer pairs. Applying CuCu to the Korean national social studies curriculum, we construct KCaQA, comprising 34.1k open-ended QA pairs. Our quantitative and qualitative analyses suggest that KCaQA covers culture-specific topics and produces responses grounded in local sociocultural contexts. 

---
# DeepHalo: A Neural Choice Model with Controllable Context Effects 

**Authors**: Shuhan Zhang, Zhi Wang, Rui Gao, Shuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04616)  

**Abstract**: Modeling human decision-making is central to applications such as recommendation, preference learning, and human-AI alignment. While many classic models assume context-independent choice behavior, a large body of behavioral research shows that preferences are often influenced by the composition of the choice set itself -- a phenomenon known as the context effect or Halo effect. These effects can manifest as pairwise (first-order) or even higher-order interactions among the available alternatives. Recent models that attempt to capture such effects either focus on the featureless setting or, in the feature-based setting, rely on restrictive interaction structures or entangle interactions across all orders, which limits interpretability. In this work, we propose DeepHalo, a neural modeling framework that incorporates features while enabling explicit control over interaction order and principled interpretation of context effects. Our model enables systematic identification of interaction effects by order and serves as a universal approximator of context-dependent choice functions when specialized to a featureless setting. Experiments on synthetic and real-world datasets demonstrate strong predictive performance while providing greater transparency into the drivers of choice. 

---
# HUR-MACL: High-Uncertainty Region-Guided Multi-Architecture Collaborative Learning for Head and Neck Multi-Organ Segmentation 

**Authors**: Xiaoyu Liu, Siwen Wei, Linhao Qu, Mingyuan Pan, Chengsheng Zhang, Yonghong Shi, Zhijian Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.04607)  

**Abstract**: Accurate segmentation of organs at risk in the head and neck is essential for radiation therapy, yet deep learning models often fail on small, complexly shaped organs. While hybrid architectures that combine different models show promise, they typically just concatenate features without exploiting the unique strengths of each component. This results in functional overlap and limited segmentation accuracy. To address these issues, we propose a high uncertainty region-guided multi-architecture collaborative learning (HUR-MACL) model for multi-organ segmentation in the head and neck. This model adaptively identifies high uncertainty regions using a convolutional neural network, and for these regions, Vision Mamba as well as Deformable CNN are utilized to jointly improve their segmentation accuracy. Additionally, a heterogeneous feature distillation loss was proposed to promote collaborative learning between the two architectures in high uncertainty regions to further enhance performance. Our method achieves SOTA results on two public datasets and one private dataset. 

---
# Crystal Generation using the Fully Differentiable Pipeline and Latent Space Optimization 

**Authors**: Osman Goni Ridwan, Gilles Frapper, Hongfei Xue, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04606)  

**Abstract**: We present a materials generation framework that couples a symmetry-conditioned variational autoencoder (CVAE) with a differentiable SO(3) power spectrum objective to steer candidates toward a specified local environment under the crystallographic constraints. In particular, we implement a fully differentiable pipeline that performs batch-wise optimization on both direct and latent crystallographic representations. Using the GPU acceleration, the implementation achieves about fivefold speed compared to our previous CPU workflow, while yielding comparable outcomes. In addition, we introduce the optimization strategy that alternatively performs optimization on the direct and latent crystal representations. This dual-level relaxation approach can effectively overcome local barrier defined by different objective gradients, thus increasing the success rate of generating complex structures satisfying the targe local environments. This framework can be extended to systems consisting of multi-components and multi-environments, providing a scalable route to generate material structures with the target local environment. 

---
# Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks 

**Authors**: Hoagy Cunningham, Jerry Wei, Zihan Wang, Andrew Persic, Alwin Peng, Jordan Abderrachid, Raj Agarwal, Bobby Chen, Austin Cohen, Andy Dau, Alek Dimitriev, Rob Gilson, Logan Howard, Yijin Hua, Jared Kaplan, Jan Leike, Mu Lin, Christopher Liu, Vladimir Mikulik, Rohit Mittapalli, Clare O'Hara, Jin Pan, Nikhil Saxena, Alex Silverstein, Yue Song, Xunjie Yu, Giulio Zhou, Ethan Perez, Mrinank Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.04603)  

**Abstract**: We introduce enhanced Constitutional Classifiers that deliver production-grade jailbreak robustness with dramatically reduced computational costs and refusal rates compared to previous-generation defenses. Our system combines several key insights. First, we develop exchange classifiers that evaluate model responses in their full conversational context, which addresses vulnerabilities in last-generation systems that examine outputs in isolation. Second, we implement a two-stage classifier cascade where lightweight classifiers screen all traffic and escalate only suspicious exchanges to more expensive classifiers. Third, we train efficient linear probe classifiers and ensemble them with external classifiers to simultaneously improve robustness and reduce computational costs. Together, these techniques yield a production-grade system achieving a 40x computational cost reduction compared to our baseline exchange classifier, while maintaining a 0.05% refusal rate on production traffic. Through extensive red-teaming comprising over 1,700 hours, we demonstrate strong protection against universal jailbreaks -- no attack on this system successfully elicited responses to all eight target queries comparable in detail to an undefended model. Our work establishes Constitutional Classifiers as practical and efficient safeguards for large language models. 

---
# On the Limitations of Rank-One Model Editing in Answering Multi-hop Questions 

**Authors**: Zhiyuan He, Binghan Chen, Tianxiang Xiong, Ziyang Sun, Mozhao Zhu, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04600)  

**Abstract**: Recent advances in Knowledge Editing (KE), particularly Rank-One Model Editing (ROME), show superior efficiency over fine-tuning and in-context learning for updating single-hop facts in transformers. However, these methods face significant challenges when applied to multi-hop reasoning tasks requiring knowledge chaining. In this work, we study the effect of editing knowledge with ROME on different layer depths and identify three key failure modes. First, the "hopping-too-late" problem occurs as later layers lack access to necessary intermediate representations. Second, generalization ability deteriorates sharply when editing later layers. Third, the model overfits to edited knowledge, incorrectly prioritizing edited-hop answers regardless of context. To mitigate the issues of "hopping-too-late" and generalisation decay, we propose Redundant Editing, a simple yet effective strategy that enhances multi-hop reasoning. Our experiments demonstrate that this approach can improve accuracy on 2-hop questions by at least 15.5 percentage points, representing a 96% increase over the previous single-edit strategy, while trading off some specificity and language naturalness. 

---
# FedKDX: Federated Learning with Negative Knowledge Distillation for Enhanced Healthcare AI Systems 

**Authors**: Quang-Tu Pham, Hoang-Dieu Vu, Dinh-Dat Pham, Hieu H. Pham  

**Link**: [PDF](https://arxiv.org/pdf/2601.04587)  

**Abstract**: This paper introduces FedKDX, a federated learning framework that addresses limitations in healthcare AI through Negative Knowledge Distillation (NKD). Unlike existing approaches that focus solely on positive knowledge transfer, FedKDX captures both target and non-target information to improve model generalization in healthcare applications. The framework integrates multiple knowledge transfer techniques--including traditional knowledge distillation, contrastive learning, and NKD--within a unified architecture that maintains privacy while reducing communication costs. Through experiments on healthcare datasets (SLEEP, UCI-HAR, and PAMAP2), FedKDX demonstrates improved accuracy (up to 2.53% over state-of-the-art methods), faster convergence, and better performance on non-IID data distributions. Theoretical analysis supports NKD's contribution to addressing statistical heterogeneity in distributed healthcare data. The approach shows promise for privacy-sensitive medical applications under regulatory frameworks like HIPAA and GDPR, offering a balanced solution between performance and practical implementation requirements in decentralized healthcare settings. The code and model are available at this https URL. 

---
# Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation 

**Authors**: Xiaowei Mao, Huihu Ding, Yan Lin, Tingrui Wu, Shengnan Guo, Dazhuo Qiu, Feiling Fang, Jilin Hu, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04572)  

**Abstract**: Imputing missing values in spatial-temporal traffic data is essential for intelligent transportation systems. Among advanced imputation methods, score-based diffusion models have demonstrated competitive performance. These models generate data by reversing a noising process, using observed values as conditional guidance. However, existing diffusion models typically apply a uniform guidance scale across both spatial and temporal dimensions, which is inadequate for nodes with high missing data rates. Sparse observations provide insufficient conditional guidance, causing the generative process to drift toward the learned prior distribution rather than closely following the conditional observations, resulting in suboptimal imputation performance.
To address this, we propose FENCE, a spatial-temporal feedback diffusion guidance method designed to adaptively control guidance scales during imputation. First, FENCE introduces a dynamic feedback mechanism that adjusts the guidance scale based on the posterior likelihood approximations. The guidance scale is increased when generated values diverge from observations and reduced when alignment improves, preventing overcorrection. Second, because alignment to observations varies across nodes and denoising steps, a global guidance scale for all nodes is suboptimal. FENCE computes guidance scales at the cluster level by grouping nodes based on their attention scores, leveraging spatial-temporal correlations to provide more accurate guidance. Experimental results on real-world traffic datasets show that FENCE significantly enhances imputation accuracy. 

---
# A Vision for Multisensory Intelligence: Sensing, Synergy, and Science 

**Authors**: Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04563)  

**Abstract**: Our experience of the world is multisensory, spanning a synthesis of language, sight, sound, touch, taste, and smell. Yet, artificial intelligence has primarily advanced in digital modalities like text, vision, and audio. This paper outlines a research vision for multisensory artificial intelligence over the next decade. This new set of technologies can change how humans and AI experience and interact with one another, by connecting AI to the human senses and a rich spectrum of signals from physiological and tactile cues on the body, to physical and social signals in homes, cities, and the environment. We outline how this field must advance through three interrelated themes of sensing, science, and synergy. Firstly, research in sensing should extend how AI captures the world in richer ways beyond the digital medium. Secondly, developing a principled science for quantifying multimodal heterogeneity and interactions, developing unified modeling architectures and representations, and understanding cross-modal transfer. Finally, we present new technical challenges to learn synergy between modalities and between humans and AI, covering multisensory integration, alignment, reasoning, generation, generalization, and experience. Accompanying this vision paper are a series of projects, resources, and demos of latest advances from the Multisensory Intelligence group at the MIT Media Lab, see this https URL. 

---
# Improving Semi-Supervised Contrastive Learning via Entropy-Weighted Confidence Integration of Anchor-Positive Pairs 

**Authors**: Shogo Nakayama, Masahiro Okuda  

**Link**: [PDF](https://arxiv.org/pdf/2601.04555)  

**Abstract**: Conventional semi-supervised contrastive learning methods assign pseudo-labels only to samples whose highest predicted class probability exceeds a predefined threshold, and then perform supervised contrastive learning using those selected samples. In this study, we propose a novel loss function that estimates the confidence of each sample based on the entropy of its predicted probability distribution and applies confidence-based adaptive weighting. This approach enables pseudo-label assignment even to samples that were previously excluded from training and facilitates contrastive learning that accounts for the confidence of both anchor and positive samples in a more principled manner. Experimental results demonstrate that the proposed method improves classification accuracy and achieves more stable learning performance even under low-label conditions. 

---
# Identifying Good and Bad Neurons for Task-Level Controllable LLMs 

**Authors**: Wenjie Li, Guansong Pang, Hezhe Qiao, Debin Gao, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04548)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities on multiple-choice question answering benchmarks, but the complex mechanisms underlying their large-scale neurons remain opaque, posing significant challenges for understanding and steering LLMs. While recent studies made progress on identifying responsible neurons for certain abilities, these ability-specific methods are infeasible for task-focused scenarios requiring coordinated use of multiple abilities. Moreover, these approaches focus only on supportive neurons that correlate positively with task completion, while neglecting neurons with other roles-such as inhibitive roles-and misled neuron attribution due to fortuitous behaviors in LLMs (i.e., correctly answer the questions by chance rather than genuine understanding). To address these challenges, we propose NeuronLLM, a novel task-level LLM understanding framework that adopts the biological principle of functional antagonism for LLM neuron identification. The key insight is that task performance is jointly determined by neurons with two opposing roles: good neurons that facilitate task completion and bad neurons that inhibit it. NeuronLLM achieves a holistic modeling of neurons via contrastive learning of good and bad neurons, while leveraging augmented question sets to mitigate the fortuitous behaviors in LLMs. Comprehensive experiments on LLMs of different sizes and families show the superiority of NeuronLLM over existing methods in four NLP tasks, providing new insights into LLM functional organization. 

---
# AdaptEval: A Benchmark for Evaluating Large Language Models on Code Snippet Adaptation 

**Authors**: Tanghaoran Zhang, Xinjun Mao, Shangwen Wang, Yuxin Zhao, Yao Lu, Jin Zhang, Zhang Zhang, Kang Yang, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04540)  

**Abstract**: Recent advancements in large language models (LLMs) have automated various software engineering tasks, with benchmarks emerging to evaluate their capabilities. However, for adaptation, a critical activity during code reuse, there is no benchmark to assess LLMs' performance, leaving their practical utility in this area unclear. To fill this gap, we propose AdaptEval, a benchmark designed to evaluate LLMs on code snippet adaptation. Unlike existing benchmarks, AdaptEval incorporates the following three distinctive features: First, Practical Context. Tasks in AdaptEval are derived from developers' practices, preserving rich contextual information from Stack Overflow and GitHub communities. Second, Multi-granularity Annotation. Each task is annotated with requirements at both task and adaptation levels, supporting the evaluation of LLMs across diverse adaptation scenarios. Third, Fine-grained Evaluation. AdaptEval includes a two-tier testing framework combining adaptation-level and function-level tests, which enables evaluating LLMs' performance across various individual adaptations. Based on AdaptEval, we conduct the first empirical study to evaluate six instruction-tuned LLMs and especially three reasoning LLMs on code snippet adaptation. Experimental results demonstrate that AdaptEval enables the assessment of LLMs' adaptation capabilities from various perspectives. It also provides critical insights into their current limitations, particularly their struggle to follow explicit instructions. We hope AdaptEval can facilitate further investigation and enhancement of LLMs' capabilities in code snippet adaptation, supporting their real-world applications. 

---
# Paradoxical noise preference in RNNs 

**Authors**: Noah Eckstein, Manoj Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04539)  

**Abstract**: In recurrent neural networks (RNNs) used to model biological neural networks, noise is typically introduced during training to emulate biological variability and regularize learning. The expectation is that removing the noise at test time should preserve or improve performance. Contrary to this intuition, we find that continuous-time recurrent neural networks (CTRNNs) often perform best at a nonzero noise level, specifically, the same level used during training. This noise preference typically arises when noise is injected inside the neural activation function; networks trained with noise injected outside the activation function perform best with zero noise. Through analyses of simple function approximation, maze navigation, and single neuron regulator tasks, we show that the phenomenon stems from noise-induced shifts of fixed points (stationary distributions) in the underlying stochastic dynamics of the RNNs. These fixed point shifts are noise-level dependent and bias the network outputs when the noise is removed, degrading performance. Analytical and numerical results show that the bias arises when neural states operate near activation function nonlinearities, where noise is asymmetrically attenuated, and that performance optimization incentivizes operation near these nonlinearities. Thus, networks can overfit to the stochastic training environment itself rather than just to the input-output data. The phenomenon is distinct from stochastic resonance, wherein nonzero noise enhances signal processing. Our findings reveal that training noise can become an integral part of the computation learned by recurrent networks, with implications for understanding neural population dynamics and for the design of robust artificial RNNs. 

---
# BanglaLorica: Design and Evaluation of a Robust Watermarking Algorithm for Large Language Models in Bangla Text Generation 

**Authors**: Amit Bin Tariqul, A N M Zahid Hossain Milkan, Sahab-Al-Chowdhury, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04534)  

**Abstract**: As large language models (LLMs) are increasingly deployed for text generation, watermarking has become essential for authorship attribution, intellectual property protection, and misuse detection. While existing watermarking methods perform well in high-resource languages, their robustness in low-resource languages remains underexplored. This work presents the first systematic evaluation of state-of-the-art text watermarking methods: KGW, Exponential Sampling (EXP), and Waterfall, for Bangla LLM text generation under cross-lingual round-trip translation (RTT) attacks. Under benign conditions, KGW and EXP achieve high detection accuracy (>88%) with negligible perplexity and ROUGE degradation. However, RTT causes detection accuracy to collapse below RTT causes detection accuracy to collapse to 9-13%, indicating a fundamental failure of token-level watermarking. To address this, we propose a layered watermarking strategy that combines embedding-time and post-generation watermarks. Experimental results show that layered watermarking improves post-RTT detection accuracy by 25-35%, achieving 40-50% accuracy, representing a 3$\times$ to 4$\times$ relative improvement over single-layer methods, at the cost of controlled semantic degradation. Our findings quantify the robustness-quality trade-off in multilingual watermarking and establish layered watermarking as a practical, training-free solution for low-resource languages such as Bangla. Our code and data will be made public. 

---
# Self-MedRAG: a Self-Reflective Hybrid Retrieval-Augmented Generation Framework for Reliable Medical Question Answering 

**Authors**: Jessica Ryan, Alexander I. Gumilang, Robert Wiliam, Derwin Suhartono  

**Link**: [PDF](https://arxiv.org/pdf/2601.04531)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in medical Question Answering (QA), yet they remain prone to hallucinations and ungrounded reasoning, limiting their reliability in high-stakes clinical scenarios. While Retrieval-Augmented Generation (RAG) mitigates these issues by incorporating external knowledge, conventional single-shot retrieval often fails to resolve complex biomedical queries requiring multi-step inference. To address this, we propose Self-MedRAG, a self-reflective hybrid framework designed to mimic the iterative hypothesis-verification process of clinical reasoning. Self-MedRAG integrates a hybrid retrieval strategy, combining sparse (BM25) and dense (Contriever) retrievers via Reciprocal Rank Fusion (RRF) to maximize evidence coverage. It employs a generator to produce answers with supporting rationales, which are then assessed by a lightweight self-reflection module using Natural Language Inference (NLI) or LLM-based verification. If the rationale lacks sufficient evidentiary support, the system autonomously reformulates the query and iterates to refine the context. We evaluated Self-MedRAG on the MedQA and PubMedQA benchmarks. The results demonstrate that our hybrid retrieval approach significantly outperforms single-retriever baselines. Furthermore, the inclusion of the self-reflective loop yielded substantial gains, increasing accuracy on MedQA from 80.00% to 83.33% and on PubMedQA from 69.10% to 79.82%. These findings confirm that integrating hybrid retrieval with iterative, evidence-based self-reflection effectively reduces unsupported claims and enhances the clinical reliability of LLM-based systems. 

---
# Advancing Language Models for Code-related Tasks 

**Authors**: Zhao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.04526)  

**Abstract**: Recent advances in language models (LMs) have driven significant progress in various software engineering tasks. However, existing LMs still struggle with complex programming scenarios due to limitations in data quality, model architecture, and reasoning capability. This research systematically addresses these challenges through three complementary directions: (1) improving code data quality with a code difference-guided adversarial augmentation technique (CODA) and a code denoising technique (CodeDenoise); (2) enhancing model architecture via syntax-guided code LMs (LEAM and LEAM++); and (3) advancing model reasoning with a prompting technique (muFiX) and an agent-based technique (Specine). These techniques aim to promote the practical adoption of LMs in software development and further advance intelligent software engineering. 

---
# TSSR: Two-Stage Swap-Reward-Driven Reinforcement Learning for Character-Level SMILES Generation 

**Authors**: Jacob Ede Levine, Yun Lyan Luo, Sai Chandra Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.04521)  

**Abstract**: The design of reliable, valid, and diverse molecules is fundamental to modern drug discovery, as improved molecular generation supports efficient exploration of the chemical space for potential drug candidates and reduces the cost of early design efforts. Despite these needs, current chemical language models that generate molecules as SMILES strings are vulnerable to compounding token errors: many samples are unparseable or chemically implausible, and hard constraints meant to prevent failure can restrict exploration. To address this gap, we introduce TSSR, a Two-Stage, Swap-Reward-driven reinforcement learning (RL) framework for character-level SMILES generation. Stage one rewards local token swaps that repair syntax, promoting transitions from invalid to parseable strings. Stage two provides chemistry-aware feedback from RDKit diagnostics, rewarding reductions in valence, aromaticity, and connectivity issues. The reward decomposes into interpretable terms (swap efficiency, error reduction, distance to validity), is model agnostic, and requires no task-specific labels or hand-crafted grammars. We evaluated TSSR on the MOSES benchmark using a GRU policy trained with PPO in both pure RL (P-RL) from random initialization and fine-tuning RL (F-RL) starting from a pretrained chemical language model, assessing 10,000 generated SMILES per run. In P-RL, TSSR significantly improves syntactic validity, chemical validity, and novelty. In F-RL, TSSR preserves drug-likeness and synthesizability while increasing validity and novelty. Token-level analysis shows that syntax edits and chemistry fixes act jointly to reduce RDKit detected errors. TSSR converts a sparse terminal objective into a denser and more interpretable reward, improving both syntactic and chemical quality without reducing diversity. TSSR is dataset-agnostic and can be adapted to various reinforcement learning approaches. 

---
# Towards Spatio-Temporal Extrapolation of Phase-Field Simulations with Convolution-Only Neural Networks 

**Authors**: Christophe Bonneville, Nathan Bieberdorf, Pieterjan Robbe, Mark Asta, Habib Najm, Laurent Capolungo, Cosmin Safta  

**Link**: [PDF](https://arxiv.org/pdf/2601.04510)  

**Abstract**: Phase-field simulations of liquid metal dealloying (LMD) can capture complex microstructural evolutions but can be prohibitively expensive for large domains and long time horizons. In this paper, we introduce a fully convolutional, conditionally parameterized U-Net surrogate designed to extrapolate far beyond its training data in both space and time. The architecture integrates convolutional self-attention, physically informed padding, and a flood-fill corrector method to maintain accuracy under extreme extrapolation, while conditioning on simulation parameters allows for flexible time-step skipping and adaptation to varying alloy compositions. To remove the need for costly solver-based initialization, we couple the surrogate with a conditional diffusion model that generates synthetic, physically consistent initial conditions. We train our surrogate on simulations generated over small domain sizes and short time spans, but, by taking advantage of the convolutional nature of U-Nets, we are able to run and extrapolate surrogate simulations for longer time horizons than what would be achievable with classic numerical solvers. Across multiple alloy compositions, the framework is able to reproduce the LMD physics accurately. It predicts key quantities of interest and spatial statistics with relative errors typically below 5% in the training regime and under 15% during large-scale, long time-horizon extrapolations. Our framework can also deliver speed-ups of up to 36,000 times, bringing the time to run weeks-long simulations down to a few seconds. This work is a first stepping stone towards high-fidelity extrapolation in both space and time of phase-field simulation for LMD. 

---
# WESR: Scaling and Evaluating Word-level Event-Speech Recognition 

**Authors**: Chenchen Yang, Kexin Huang, Liwei Fan, Qian Tu, Botian Jiang, Dong Zhang, Linqi Yin, Shimin Li, Zhaoye Fei, Qinyuan Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04508)  

**Abstract**: Speech conveys not only linguistic information but also rich non-verbal vocal events such as laughing and crying. While semantic transcription is well-studied, the precise localization of non-verbal events remains a critical yet under-explored challenge. Current methods suffer from insufficient task definitions with limited category coverage and ambiguous temporal granularity. They also lack standardized evaluation frameworks, hindering the development of downstream applications. To bridge this gap, we first develop a refined taxonomy of 21 vocal events, with a new categorization into discrete (standalone) versus continuous (mixed with speech) types. Based on the refined taxonomy, we introduce WESR-Bench, an expert-annotated evaluation set (900+ utterances) with a novel position-aware protocol that disentangles ASR errors from event detection, enabling precise localization measurement for both discrete and continuous events. We also build a strong baseline by constructing a 1,700+ hour corpus, and train specialized models, surpassing both open-source audio-language models and commercial APIs while preserving ASR quality. We anticipate that WESR will serve as a foundational resource for future research in modeling rich, real-world auditory scenes. 

---
# A Semi-supervised Molecular Learning Framework for Activity Cliff Estimation 

**Authors**: Fang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04507)  

**Abstract**: Machine learning (ML) enables accurate and fast molecular property predictions, which are of interest in drug discovery and material design. Their success is based on the principle of similarity at its heart, assuming that similar molecules exhibit close properties. However, activity cliffs challenge this principle, and their presence leads to a sharp decline in the performance of existing ML algorithms, particularly graph-based methods. To overcome this obstacle under a low-data scenario, we propose a novel semi-supervised learning (SSL) method dubbed SemiMol, which employs predictions on numerous unannotated data as pseudo-signals for subsequent training. Specifically, we introduce an additional instructor model to evaluate the accuracy and trustworthiness of proxy labels because existing pseudo-labeling approaches require probabilistic outputs to reveal the model's confidence and fail to be applied in regression tasks. Moreover, we design a self-adaptive curriculum learning algorithm to progressively move the target model toward hard samples at a controllable pace. Extensive experiments on 30 activity cliff datasets demonstrate that SemiMol significantly enhances graph-based ML architectures and outpasses state-of-the-art pretraining and SSL baselines. 

---
# Surface-based Molecular Design with Multi-modal Flow Matching 

**Authors**: Fang Wu, Zhengyuan Zhou, Shuting Jin, Xiangxiang Zeng, Jure Leskovec, Jinbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04506)  

**Abstract**: Therapeutic peptides show promise in targeting previously undruggable binding sites, with recent advancements in deep generative models enabling full-atom peptide co-design for specific protein receptors. However, the critical role of molecular surfaces in protein-protein interactions (PPIs) has been underexplored. To bridge this gap, we propose an omni-design peptides generation paradigm, called SurfFlow, a novel surface-based generative algorithm that enables comprehensive co-design of sequence, structure, and surface for peptides. SurfFlow employs a multi-modality conditional flow matching (CFM) architecture to learn distributions of surface geometries and biochemical properties, enhancing peptide binding accuracy. Evaluated on the comprehensive PepMerge benchmark, SurfFlow consistently outperforms full-atom baselines across all metrics. These results highlight the advantages of considering molecular surfaces in de novo peptide discovery and demonstrate the potential of integrating multiple protein modalities for more effective therapeutic peptide discovery. 

---
# Vision-Language Agents for Interactive Forest Change Analysis 

**Authors**: James Brock, Ce Zhang, Nantheera Anantrasirichai  

**Link**: [PDF](https://arxiv.org/pdf/2601.04497)  

**Abstract**: Modern forest monitoring workflows increasingly benefit from the growing availability of high-resolution satellite imagery and advances in deep learning. Two persistent challenges in this context are accurate pixel-level change detection and meaningful semantic change captioning for complex forest dynamics. While large language models (LLMs) are being adapted for interactive data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored. To address this gap, we introduce an LLM-driven agent for integrated forest change analysis that supports natural language querying across multiple RSICI tasks. The proposed system builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration. To facilitate adaptation and evaluation in forest environments, we further introduce the Forest-Change dataset, which comprises bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated using a combination of human annotation and rule-based methods. Experimental results show that the proposed system achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on the Forest-Change dataset, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI benchmark for joint change detection and captioning. These results highlight the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and efficiency of forest change analysis. All data and code are publicly available at this https URL. 

---
# Scalable Floating-Point Satisfiability via Staged Optimization 

**Authors**: Yuanzhuo Zhang, Zhoulai Fu, Binoy Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2601.04492)  

**Abstract**: This work introduces StageSAT, a new approach to solving floating-point satisfiability that bridges SMT solving with numerical optimization. StageSAT reframes a floating-point formula as a series of optimization problems in three stages of increasing precision. It begins with a fast, projection-aided descent objective to guide the search toward a feasible region, proceeding to bit-level accuracy with ULP$^2$ optimization and a final $n$-ULP lattice refinement.
By construction, the final stage uses a representing function that is zero if and only if a candidate satisfies all constraints. Thus, when optimization drives the objective to zero, the resulting assignment is a valid solution, providing a built-in guarantee of soundness.
To improve search, StageSAT introduces a partial monotone descent property on linear constraints via orthogonal projection, preventing the optimizer from stalling on flat or misleading landscapes. Critically, this solver requires no heavy bit-level reasoning or specialized abstractions; it treats complex arithmetic as a black-box, using runtime evaluations to navigate the input space.
We implement StageSAT and evaluate it on extensive benchmarks, including SMT-COMP'25 suites and difficult cases from prior work. StageSAT proved more scalable and accurate than state-of-the-art optimization-based alternatives. It solved strictly more formulas than any competing solver under the same time budget, finding most satisfiable instances without producing spurious models. This amounts to 99.4% recall on satisfiable cases with 0% false SAT, exceeding the reliability of prior optimization-based solvers. StageSAT also delivered significant speedups (often 5--10$\times$) over traditional bit-precise SMT and numeric solvers. These results demonstrate that staged optimization significantly improves performance and correctness of floating-point satisfiability solving. 

---
# Decision-Aware Trust Signal Alignment for SOC Alert Triage 

**Authors**: Israt Jahan Chowdhury, Md Abu Yousuf Tanvir  

**Link**: [PDF](https://arxiv.org/pdf/2601.04486)  

**Abstract**: Detection systems that utilize machine learning are progressively implemented at Security Operations Centers (SOCs) to help an analyst to filter through high volumes of security alerts. Practically, such systems tend to reveal probabilistic results or confidence scores which are ill-calibrated and hard to read when under pressure. Qualitative and survey based studies of SOC practice done before reveal that poor alert quality and alert overload greatly augment the burden on the analyst, especially when tool outputs are not coherent with decision requirements, or signal noise. One of the most significant limitations is that model confidence is usually shown without expressing that there are asymmetric costs in decision making where false alarms are much less harmful than missed attacks. The present paper presents a decision-sensitive trust signal correspondence scheme of SOC alert triage. The framework combines confidence that has been calibrated, lightweight uncertainty cues, and cost-sensitive decision thresholds into coherent decision-support layer, instead of making changes to detection models. To enhance probabilistic consistency, the calibration is done using the known post-hoc methods and the uncertainty cues give conservative protection in situations where model certainty is low. To measure the model-independent performance of the suggested model, we apply the Logistic Regression and the Random Forest classifiers to the UNSW-NB15 intrusion detection benchmark. According to simulation findings, false negatives are greatly amplified by the presence of misaligned displays of confidence, whereas cost weighted loss decreases by orders of magnitude between models with decision aligned trust signals. Lastly, we describe a human-in-the-loop study plan that would allow empirically assessing the decision-making of the analysts with aligned and misaligned trust interfaces. 

---
# Hybrid Federated Learning for Noise-Robust Training 

**Authors**: Yongjun Kim, Hyeongjun Park, Hwanjin Kim, Junil Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04483)  

**Abstract**: Federated learning (FL) and federated distillation (FD) are distributed learning paradigms that train UE models with enhanced privacy, each offering different trade-offs between noise robustness and learning speed. To mitigate their respective weaknesses, we propose a hybrid federated learning (HFL) framework in which each user equipment (UE) transmits either gradients or logits, and the base station (BS) selects the per-round weights of FL and FD updates. We derive convergence of HFL framework and introduce two methods to exploit degrees of freedom (DoF) in HFL, which are (i) adaptive UE clustering via Jenks optimization and (ii) adaptive weight selection via a damped Newton method. Numerical results show that HFL achieves superior test accuracy at low SNR when both DoF are exploited. 

---
# Concept Tokens: Learning Behavioral Embeddings Through Concept Definitions 

**Authors**: Ignacio Sastre, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2601.04465)  

**Abstract**: We propose Concept Tokens, a lightweight method that adds a new special token to a pretrained LLM and learns only its embedding from multiple natural language definitions of a target concept, where occurrences of the concept are replaced by the new token. The LLM is kept frozen and the embedding is optimized with the standard language-modeling objective. We evaluate Concept Tokens in three settings. First, we study hallucinations in closed-book question answering on HotpotQA and find a directional effect: negating the hallucination token reduces hallucinated answers mainly by increasing abstentions, whereas asserting it increases hallucinations and lowers precision. Second, we induce recasting, a pedagogical feedback strategy for second language teaching, and observe the same directional effect. Moreover, compared to providing the full definitional corpus in-context, concept tokens better preserve compliance with other instructions (e.g., asking follow-up questions). Finally, we include a qualitative study with the Eiffel Tower and a fictional "Austral Tower" to illustrate what information the learned embeddings capture and where their limitations emerge. Overall, Concept Tokens provide a compact control signal learned from definitions that can steer behavior in frozen LLMs. 

---
# Beyond Static Summarization: Proactive Memory Extraction for LLM Agents 

**Authors**: Chengyuan Yang, Zequn Sun, Wei Wei, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04463)  

**Abstract**: Memory management is vital for LLM agents to handle long-term interaction and personalization. Most research focuses on how to organize and use memory summary, but often overlooks the initial memory extraction stage. In this paper, we argue that existing summary-based methods have two major limitations based on the recurrent processing theory. First, summarization is "ahead-of-time", acting as a blind "feed-forward" process that misses important details because it doesn't know future tasks. Second, extraction is usually "one-off", lacking a feedback loop to verify facts, which leads to the accumulation of information loss. To address these issues, we propose proactive memory extraction (namely ProMem). Unlike static summarization, ProMem treats extraction as an iterative cognitive process. We introduce a recurrent feedback loop where the agent uses self-questioning to actively probe the dialogue history. This mechanism allows the agent to recover missing information and correct errors. Our ProMem significantly improves the completeness of the extracted memory and QA accuracy. It also achieves a superior trade-off between extraction quality and token cost. 

---
# Re-Rankers as Relevance Judges 

**Authors**: Chuan Meng, Jiqun Liu, Mohammad Aliannejadi, Fengran Mo, Jeff Dalton, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2601.04455)  

**Abstract**: Using large language models (LLMs) to predict relevance judgments has shown promising results. Most studies treat this task as a distinct research line, e.g., focusing on prompt design for predicting relevance labels given a query and passage. However, predicting relevance judgments is essentially a form of relevance prediction, a problem extensively studied in tasks such as re-ranking. Despite this potential overlap, little research has explored reusing or adapting established re-ranking methods to predict relevance judgments, leading to potential resource waste and redundant development. To bridge this gap, we reproduce re-rankers in a re-ranker-as-relevance-judge setup. We design two adaptation strategies: (i) using binary tokens (e.g., "true" and "false") generated by a re-ranker as direct judgments, and (ii) converting continuous re-ranking scores into binary labels via thresholding. We perform extensive experiments on TREC-DL 2019 to 2023 with 8 re-rankers from 3 families, ranging from 220M to 32B, and analyse the evaluation bias exhibited by re-ranker-based judges. Results show that re-ranker-based relevance judges, under both strategies, can outperform UMBRELA, a state-of-the-art LLM-based relevance judge, in around 40% to 50% of the cases; they also exhibit strong self-preference towards their own and same-family re-rankers, as well as cross-family bias. 

---
# Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models 

**Authors**: San Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04448)  

**Abstract**: Large Language Models (LLMs) have greatly advanced Natural Language Processing (NLP), particularly through instruction tuning, which enables broad task generalization without additional fine-tuning. However, their reliance on large-scale datasets-often collected from human or web sources-makes them vulnerable to backdoor attacks, where adversaries poison a small subset of data to implant hidden behaviors. Despite this growing risk, defenses for instruction-tuned models remain underexplored. We propose MB-Defense (Merging & Breaking Defense Framework), a novel training pipeline that immunizes instruction-tuned LLMs against diverse backdoor threats. MB-Defense comprises two stages: (i) defensive poisoning, which merges attacker and defensive triggers into a unified backdoor representation, and (ii) weight recovery, which breaks this representation through additional training to restore clean behavior. Extensive experiments across multiple LLMs show that MB-Defense substantially lowers attack success rates while preserving instruction-following ability. Our method offers a generalizable and data-efficient defense strategy, improving the robustness of instruction-tuned LLMs against unseen backdoor attacks. 

---
# SpectraFormer: an Attention-Based Raman Unmixing Tool for Accessing the Graphene Buffer-Layer Signature on SiC 

**Authors**: Dmitriy Poteryayev, Pietro Novelli, Annalisa Coriolano, Riccardo Dettori, Valentina Tozzini, Fabio Beltram, Massimiliano Pontil, Antonio Rossi, Stiven Forti, Camilla Coletti  

**Link**: [PDF](https://arxiv.org/pdf/2601.04445)  

**Abstract**: Raman spectroscopy is a key tool for graphene characterization, yet its application to graphene grown on silicon carbide (SiC) is strongly limited by the intense and variable second-order Raman response of the substrate. This limitation is critical for buffer layer graphene, a semiconducting interfacial phase, whose vibrational signatures are overlapped with the SiC background and challenging to be reliably accessed using conventional reference-based subtraction, due to strong spatial and experimental variability of the substrate signal. Here we present SpectraFormer, a transformer-based deep learning model that reconstructs the SiC Raman substrate contribution directly from post-growth partially masked spectroscopic data without relying on explicit reference measurements. By learning global correlations across the entire Raman shift range, the model captures the statistical structure of the SiC background and enables accurate reconstruction of its contribution in mixed spectra. Subtraction of the reconstructed substrate signal reveals weak vibrational features associated with ZLG that are inaccessible through conventional analysis methods. The extracted spectra are validated by ab initio vibrational calculations, allowing assignment of the resolved features to specific modes and confirming their physical consistency. By leveraging a state-of-the-art attention-based deep learning architecture, this approach establishes a robust, reference-free framework for Raman analysis of graphene on SiC and provides a foundation, compatible with real-time data acquisition, to its integration into automated, closed-loop AI-assisted growth optimization. 

---
# Accommodation and Epistemic Vigilance: A Pragmatic Account of Why LLMs Fail to Challenge Harmful Beliefs 

**Authors**: Myra Cheng, Robert D. Hawkins, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2601.04435)  

**Abstract**: Large language models (LLMs) frequently fail to challenge users' harmful beliefs in domains ranging from medical advice to social reasoning. We argue that these failures can be understood and addressed pragmatically as consequences of LLMs defaulting to accommodating users' assumptions and exhibiting insufficient epistemic vigilance. We show that social and linguistic factors known to influence accommodation in humans (at-issueness, linguistic encoding, and source reliability) similarly affect accommodation in LLMs, explaining performance differences across three safety benchmarks that test models' ability to challenge harmful beliefs, spanning misinformation (Cancer-Myth, SAGE-Eval) and sycophancy (ELEPHANT). We further show that simple pragmatic interventions, such as adding the phrase "wait a minute", significantly improve performance on these benchmarks while preserving low false-positive rates. Our results highlight the importance of considering pragmatics for evaluating LLM behavior and improving LLM safety. 

---
# CRUNet-MR-Univ: A Foundation Model for Diverse Cardiac MRI Reconstruction 

**Authors**: Donghang Lyu, Marius Staring, Hildo Lamb, Mariya Doneva  

**Link**: [PDF](https://arxiv.org/pdf/2601.04428)  

**Abstract**: In recent years, deep learning has attracted increasing at- tention in the field of Cardiac MRI (CMR) reconstruction due to its superior performance over traditional methods, particularly in handling higher acceleration factors, highlighting its potential for real-world clini- cal applications. However, current deep learning methods remain limited in generalizability. CMR scans exhibit wide variability in image contrast, sampling patterns, scanner vendors, anatomical structures, and disease types. Most existing models are designed to handle only a single or nar- row subset of these variations, leading to performance degradation when faced with distribution shifts. Therefore, it is beneficial to develop a unified model capable of generalizing across diverse CMR scenarios. To this end, we propose CRUNet-MR-Univ, a foundation model that lever- ages spatio-temporal correlations and prompt-based priors to effectively handle the full diversity of CMR scans. Our approach consistently out- performs baseline methods across a wide range of settings, highlighting its effectiveness and promise. 

---
# Rate or Fate? RLV$^\varepsilon$R: Reinforcement Learning with Verifiable Noisy Rewards 

**Authors**: Ali Rad, Khashayar Filom, Darioush Keivan, Peyman Mohajerin Esfahani, Ehsan Kamalinejad  

**Link**: [PDF](https://arxiv.org/pdf/2601.04411)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a simple but powerful paradigm for training LLMs: sample a completion, verify it, and update. In practice, however, the verifier is almost never clean--unit tests probe only limited corner cases; human and synthetic labels are imperfect; and LLM judges (e.g., RLAIF) are noisy and can be exploited--and this problem worsens on harder domains (especially coding) where tests are sparse and increasingly model-generated. We ask a pragmatic question: does the verification noise merely slow down the learning (rate), or can it flip the outcome (fate)?
To address this, we develop an analytically tractable multi-armed bandit view of RLVR dynamics, instantiated with GRPO and validated in controlled experiments. Modeling false positives and false negatives and grouping completions into recurring reasoning modes yields a replicator-style (natural-selection) flow on the probability simplex. The dynamics decouples into within-correct-mode competition and a one-dimensional evolution for the mass on incorrect modes, whose drift is determined solely by Youden's index J=TPR-FPR. This yields a sharp phase transition: when J>0, the incorrect mass is driven toward extinction (learning); when J=0, the process is neutral; and when J<0, incorrect modes amplify until they dominate (anti-learning and collapse). In the learning regime J>0, noise primarily rescales convergence time ("rate, not fate"). Experiments on verifiable programming tasks under synthetic noise reproduce the predicted J=0 boundary. Beyond noise, the framework offers a general lens for analyzing RLVR stability, convergence, and algorithmic interventions. 

---
# From Preoperative CT to Postmastoidectomy Mesh Construction:1Mastoidectomy Shape Prediction for Cochlear Implant Surgery 

**Authors**: Yike Zhang, Eduardo Davalos, Dingjie Su, Ange Lou, Jack Noble  

**Link**: [PDF](https://arxiv.org/pdf/2601.04405)  

**Abstract**: Cochlear Implant (CI) surgery treats severe hearing loss by inserting an electrode array into the cochlea to stimulate the auditory nerve. An important step in this procedure is mastoidectomy, which removes part of the mastoid region of the temporal bone to provide surgical access. Accurate mastoidectomy shape prediction from preoperative imaging improves pre-surgical planning, reduces risks, and enhances surgical outcomes. Despite its importance, there are limited deep-learning-based studies regarding this topic due to the challenges of acquiring ground-truth labels. We address this gap by investigating self-supervised and weakly-supervised learning models to predict the mastoidectomy region without human annotations. We propose a hybrid self-supervised and weakly-supervised learning framework to predict the mastoidectomy region directly from preoperative CT scans, where the mastoid remains intact. Our hybrid method achieves a mean Dice score of 0.72 when predicting the complex and boundary-less mastoidectomy shape, surpassing state-of-the-art approaches and demonstrating strong performance. The method provides groundwork for constructing 3D postmastoidectomy surfaces directly from the corresponding preoperative CT scans. To our knowledge, this is the first work that integrating self-supervised and weakly-supervised learning for mastoidectomy shape prediction, offering a robust and efficient solution for CI surgical planning while leveraging 3D T-distribution loss in weakly-supervised medical imaging. 

---
# 3D-Agent:Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation 

**Authors**: Jusheng Zhang, Yijia Fan, Zimo Wen, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04404)  

**Abstract**: Driven by applications in autonomous driving robotics and augmented reality 3D object annotation presents challenges beyond 2D annotation including spatial complexity occlusion and viewpoint inconsistency Existing approaches based on single models often struggle to address these issues effectively We propose Tri MARF a novel framework that integrates tri modal inputs including 2D multi view images textual descriptions and 3D point clouds within a multi agent collaborative architecture to enhance large scale 3D annotation Tri MARF consists of three specialized agents a vision language model agent for generating multi view descriptions an information aggregation agent for selecting optimal descriptions and a gating agent that aligns textual semantics with 3D geometry for refined captioning Extensive experiments on Objaverse LVIS Objaverse XL and ABO demonstrate that Tri MARF substantially outperforms existing methods achieving a CLIPScore of 88 point 7 compared to prior state of the art methods retrieval accuracy of 45 point 2 and 43 point 8 on ViLT R at 5 and a throughput of up to 12000 objects per hour on a single NVIDIA A100 GPU 

---
# Balancing Usability and Compliance in AI Smart Devices: A Privacy-by-Design Audit of Google Home, Alexa, and Siri 

**Authors**: Trevor De Clark, Yulia Bobkova, Ajay Kumar Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2601.04403)  

**Abstract**: This paper investigates the privacy and usability of AI-enabled smart devices commonly used by youth, focusing on Google Home Mini, Amazon Alexa, and Apple Siri. While these devices provide convenience and efficiency, they also raise privacy and transparency concerns due to their always-listening design and complex data management processes. The study proposes and applies a combined framework of Heuristic Evaluation, Personal Information Protection and Electronic Documents Act (PIPEDA) Compliance Assessment, and Youth-Centered Usability Testing to assess whether these devices align with Privacy-by-Design principles and support meaningful user control. Results show that Google Home achieved the highest usability score, while Siri scored highest in regulatory compliance, indicating a trade-off between user convenience and privacy protection. Alexa demonstrated clearer task navigation but weaker transparency in data retention. Findings suggest that although youth may feel capable of managing their data, their privacy self-efficacy remains limited by technical design, complex settings, and unclear data policies. The paper concludes that enhancing transparency, embedding privacy guidance during onboarding, and improving policy alignment are critical steps toward ensuring that smart devices are both usable and compliant with privacy standards that protect young users. 

---
# Convenience vs. Control: A Qualitative Study of Youth Privacy with Smart Voice Assistants 

**Authors**: Molly Campbell, Trevor De Clark, Mohamad Sheikho Al Jasem, Sandhya Joshi, Ajay Kumar Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2601.04399)  

**Abstract**: Smart voice assistants (SVAs) are embedded in the daily lives of youth, yet their privacy controls often remain opaque and difficult to manage. Through five semi-structured focus groups (N=26) with young Canadians (ages 16-24), we investigate how perceived privacy risks (PPR) and benefits (PPBf) intersect with algorithmic transparency and trust (ATT) and privacy self-efficacy (PSE) to shape privacy-protective behaviors (PPB). Our analysis reveals that policy overload, fragmented settings, and unclear data retention undermine self-efficacy and discourage protective actions. Conversely, simple transparency cues were associated with greater confidence without diminishing the utility of hands-free tasks and entertainment. We synthesize these findings into a qualitative model in which transparency friction erodes PSE, which in turn weakens PPB. From this model, we derive actionable design guidance for SVAs, including a unified privacy hub, plain-language "data nutrition" labels, clear retention defaults, and device-conditional micro-tutorials. This work foregrounds youth perspectives and offers a path for SVA governance and design that empowers young digital citizens while preserving convenience. 

---
# Enhanced-FQL($λ$), an Efficient and Interpretable RL with novel Fuzzy Eligibility Traces and Segmented Experience Replay 

**Authors**: Mohsen Jalaeian-Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2601.04392)  

**Abstract**: This paper introduces a fuzzy reinforcement learning framework, Enhanced-FQL($\lambda$), that integrates novel Fuzzified Eligibility Traces (FET) and Segmented Experience Replay (SER) into fuzzy Q-learning with Fuzzified Bellman Equation (FBE) for continuous control tasks. The proposed approach employs an interpretable fuzzy rule base instead of complex neural architectures, while maintaining competitive performance through two key innovations: a fuzzified Bellman equation with eligibility traces for stable multi-step credit assignment, and a memory-efficient segment-based experience replay mechanism for enhanced sample efficiency. Theoretical analysis proves the proposed method convergence under standard assumptions. Extensive evaluations in continuous control domains demonstrate that Enhanced-FQL($\lambda$) achieves superior sample efficiency and reduced variance compared to n-step fuzzy TD and fuzzy SARSA($\lambda$) baselines, while maintaining substantially lower computational complexity than deep RL alternatives such as DDPG. The framework's inherent interpretability, combined with its computational efficiency and theoretical convergence guarantees, makes it particularly suitable for safety-critical applications where transparency and resource constraints are essential. 

---
# MiJaBench: Revealing Minority Biases in Large Language Models via Hate Speech Jailbreaking 

**Authors**: Iago Alves Brito, Walcy Santos Rezende Rios, Julia Soares Dollis, Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2601.04389)  

**Abstract**: Current safety evaluations of large language models (LLMs) create a dangerous illusion of universality, aggregating "Identity Hate" into scalar scores that mask systemic vulnerabilities against specific populations. To expose this selective safety, we introduce MiJaBench, a bilingual (English and Portuguese) adversarial benchmark comprising 44,000 prompts across 16 minority groups. By generating 528,000 prompt-response pairs from 12 state-of-the-art LLMs, we curate MiJaBench-Align, revealing that safety alignment is not a generalized semantic capability but a demographic hierarchy: defense rates fluctuate by up to 33\% within the same model solely based on the target group. Crucially, we demonstrate that model scaling exacerbates these disparities, suggesting that current alignment techniques do not create principle of non-discrimination but reinforces memorized refusal boundaries only for specific groups, challenging the current scaling laws of security. We release all datasets and scripts to encourage research into granular demographic alignment at GitHub. 

---
# Few-Shot LoRA Adaptation of a Flow-Matching Foundation Model for Cross-Spectral Object Detection 

**Authors**: Maxim Clouser, Kia Khezeli, John Kalantari  

**Link**: [PDF](https://arxiv.org/pdf/2601.04381)  

**Abstract**: Foundation models for vision are predominantly trained on RGB data, while many safety-critical applications rely on non-visible modalities such as infrared (IR) and synthetic aperture radar (SAR). We study whether a single flow-matching foundation model pre-trained primarily on RGB images can be repurposed as a cross-spectral translator using only a few co-measured examples, and whether the resulting synthetic data can enhance downstream detection. Starting from FLUX.1 Kontext, we insert low-rank adaptation (LoRA) modules and fine-tune them on just 100 paired images per domain for two settings: RGB to IR on the KAIST dataset and RGB to SAR on the M4-SAR dataset. The adapted model translates RGB images into pixel-aligned IR/SAR, enabling us to reuse existing bounding boxes and train object detection models purely in the target modality. Across a grid of LoRA hyperparameters, we find that LPIPS computed on only 50 held-out pairs is a strong proxy for downstream performance: lower LPIPS consistently predicts higher mAP for YOLOv11n on both IR and SAR, and for DETR on KAIST IR test data. Using the best LPIPS-selected LoRA adapter, synthetic IR from external RGB datasets (LLVIP, FLIR ADAS) improves KAIST IR pedestrian detection, and synthetic SAR significantly boosts infrastructure detection on M4-SAR when combined with limited real SAR. Our results suggest that few-shot LoRA adaptation of flow-matching foundation models is a promising path toward foundation-style support for non-visible modalities. 

---
# Disco-RAG: Discourse-Aware Retrieval-Augmented Generation 

**Authors**: Dongqi Liu, Hang Ding, Qiming Feng, Jian Li, Xurong Xie, Zhucun Xue, Chengjie Wang, Jiangning Zhang, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04377)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems. 

---
# Graph Integrated Transformers for Community Detection in Social Networks 

**Authors**: Heba Zahran, M.Omair Shafiq  

**Link**: [PDF](https://arxiv.org/pdf/2601.04367)  

**Abstract**: Community detection is crucial for applications like targeted marketing and recommendation systems. Traditional methods rely on network structure, and embedding-based models integrate semantic information. However, there is a challenge when a model leverages local and global information from complex structures like social networks. Graph Neural Networks (GNNs) and Transformers have shown superior performance in capturing local and global relationships. In this paper, We propose Graph Integrated Transformer for Community Detection (GIT-CD), a hybrid model combining GNNs and Transformer-based attention mechanisms to enhance community detection in social networks. Specifically, the GNN module captures local graph structures, while the Transformer module models long-range dependencies. A self-optimizing clustering module refines community assignments using K-Means, silhouette loss, and KL divergence minimization. Experimental results on benchmark datasets show that GIT-CD outperforms state-of-the-art models, making it a robust approach for detecting meaningful communities in complex social networks. 

---
# Causally-Aware Information Bottleneck for Domain Adaptation 

**Authors**: Mohammad Ali Javidian  

**Link**: [PDF](https://arxiv.org/pdf/2601.04361)  

**Abstract**: We tackle a common domain adaptation setting in causal systems. In this setting, the target variable is observed in the source domain but is entirely missing in the target domain. We aim to impute the target variable in the target domain from the remaining observed variables under various shifts. We frame this as learning a compact, mechanism-stable representation. This representation preserves information relevant for predicting the target while discarding spurious variation. For linear Gaussian causal models, we derive a closed-form Gaussian Information Bottleneck (GIB) solution. This solution reduces to a canonical correlation analysis (CCA)-style projection and offers Directed Acyclic Graph (DAG)-aware options when desired. For nonlinear or non-Gaussian data, we introduce a Variational Information Bottleneck (VIB) encoder-predictor. This approach scales to high dimensions and can be trained on source data and deployed zero-shot to the target domain. Across synthetic and real datasets, our approach consistently attains accurate imputations, supporting practical use in high-dimensional causal models and furnishing a unified, lightweight toolkit for causal domain adaptation. 

---
# UNIC: Learning Unified Multimodal Extrinsic Contact Estimation 

**Authors**: Zhengtong Xu, Yuki Shirai  

**Link**: [PDF](https://arxiv.org/pdf/2601.04356)  

**Abstract**: Contact-rich manipulation requires reliable estimation of extrinsic contacts-the interactions between a grasped object and its environment which provide essential contextual information for planning, control, and policy learning. However, existing approaches often rely on restrictive assumptions, such as predefined contact types, fixed grasp configurations, or camera calibration, that hinder generalization to novel objects and deployment in unstructured environments. In this paper, we present UNIC, a unified multimodal framework for extrinsic contact estimation that operates without any prior knowledge or camera calibration. UNIC directly encodes visual observations in the camera frame and integrates them with proprioceptive and tactile modalities in a fully data-driven manner. It introduces a unified contact representation based on scene affordance maps that captures diverse contact formations and employs a multimodal fusion mechanism with random masking, enabling robust multimodal representation learning. Extensive experiments demonstrate that UNIC performs reliably. It achieves a 9.6 mm average Chamfer distance error on unseen contact locations, performs well on unseen objects, remains robust under missing modalities, and adapts to dynamic camera viewpoints. These results establish extrinsic contact estimation as a practical and versatile capability for contact-rich manipulation. 

---
# Summary of The Inaugural Music Source Restoration Challenge 

**Authors**: Yongyi Zang, Jiarui Hai, Wanying Ge, Qiuqiang Kong, Zheqi Dai, Helin Wang, Yuki Mitsufuji, Mark D. Plumbley  

**Link**: [PDF](https://arxiv.org/pdf/2601.04343)  

**Abstract**: Music Source Restoration (MSR) aims to recover original, unprocessed instrument stems from professionally mixed and degraded audio, requiring the reversal of both production effects and real-world degradations. We present the inaugural MSR Challenge, which features objective evaluation on studio-produced mixtures using Multi-Mel-SNR, Zimtohrli, and FAD-CLAP, alongside subjective evaluation on real-world degraded recordings. Five teams participated in the challenge. The winning system achieved 4.46 dB Multi-Mel-SNR and 3.47 MOS-Overall, corresponding to relative improvements of 91% and 18% over the second-place system, respectively. Per-stem analysis reveals substantial variation in restoration difficulty across instruments, with bass averaging 4.59 dB across all teams, while percussion averages only 0.29 dB. The dataset, evaluation protocols, and baselines are available at this https URL. 

---
# Unified Text-Image Generation with Weakness-Targeted Post-Training 

**Authors**: Jiahui Chen, Philippe Hansen-Estruch, Xiaochuang Han, Yushi Hu, Emily Dinan, Amita Kamath, Michal Drozdzal, Reyhane Askari-Hemmat, Luke Zettlemoyer, Marjan Ghazvininejad  

**Link**: [PDF](https://arxiv.org/pdf/2601.04339)  

**Abstract**: Unified multimodal generation architectures that jointly produce text and images have recently emerged as a promising direction for text-to-image (T2I) synthesis. However, many existing systems rely on explicit modality switching, generating reasoning text before switching manually to image generation. This separate, sequential inference process limits cross-modal coupling and prohibits automatic multimodal generation. This work explores post-training to achieve fully unified text-image generation, where models autonomously transition from textual reasoning to visual synthesis within a single inference process. We examine the impact of joint text-image generation on T2I performance and the relative importance of each modality during post-training. We additionally explore different post-training data strategies, showing that a targeted dataset addressing specific limitations achieves superior results compared to broad image-caption corpora or benchmark-aligned data. Using offline, reward-weighted post-training with fully self-generated synthetic data, our approach enables improvements in multimodal image generation across four diverse T2I benchmarks, demonstrating the effectiveness of reward-weighting both modalities and strategically designed post-training data. 

---
# ParaCodex: A Profiling-Guided Autonomous Coding Agent for Reliable Parallel Code Generation and Translation 

**Authors**: Erel Kaplan, Tomer Bitan, Lian Ghrayeb, Le Chen, Tom Yotam, Niranjan Hasabnis, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2601.04327)  

**Abstract**: Parallel programming is central to HPC and AI, but producing code that is correct and fast remains challenging, especially for OpenMP GPU offload, where data movement and tuning dominate. Autonomous coding agents can compile, test, and profile on target hardware, but outputs are brittle without domain scaffolding.
We present ParaCodex, an HPC-engineer workflow that turns a Codex-based agent into an autonomous OpenMP GPU offload system using staged hotspot analysis, explicit data planning, correctness gating, and profiling-guided refinement. We evaluate translation from serial CPU kernels to OpenMP GPU offload kernels on HeCBench, Rodinia, and NAS. After excluding five kernels, ParaCodex succeeded on all 31 valid kernels. The generated kernels improved GPU time over reference OpenMP implementations in 25/31 cases, achieving geometric-mean speedups of 3x on HeCBench and 5x on Rodinia, and outperforming a zero-shot Codex baseline on all suites. We also evaluate CUDA to OpenMP offload translation on ParEval, where ParaCodex maintains high compilation and validation rates in code-only and end-to-end settings. 

---
# Online Action-Stacking Improves Reinforcement Learning Performance for Air Traffic Control 

**Authors**: Ben Carvell, George De Ath, Eseoghene Benjamin, Richard Everson  

**Link**: [PDF](https://arxiv.org/pdf/2601.04287)  

**Abstract**: We introduce online action-stacking, an inference-time wrapper for reinforcement learning policies that produces realistic air traffic control commands while allowing training on a much smaller discrete action space. Policies are trained with simple incremental heading or level adjustments, together with an action-damping penalty that reduces instruction frequency and leads agents to issue commands in short bursts. At inference, online action-stacking compiles these bursts of primitive actions into domain-appropriate compound clearances. Using Proximal Policy Optimisation and the BluebirdDT digital twin platform, we train agents to navigate aircraft along lateral routes, manage climb and descent to target flight levels, and perform two-aircraft collision avoidance under a minimum separation constraint. In our lateral navigation experiments, action stacking greatly reduces the number of issued instructions relative to a damped baseline and achieves comparable performance to a policy trained with a 37-dimensional action space, despite operating with only five actions. These results indicate that online action-stacking helps bridge a key gap between standard reinforcement learning formulations and operational ATC requirements, and provides a simple mechanism for scaling to more complex control scenarios. 

---
# From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning 

**Authors**: Xiaoyu Xu, Minxin Du, Zitong Li, Zi Liang, Zhibiao Guo, Shiyu Zhang, Peizhao Hu, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04278)  

**Abstract**: Although machine unlearning is essential for removing private, harmful, or copyrighted content from LLMs, current benchmarks often fail to faithfully represent the true "forgetting scope" learned by the model. We formalize two distinct unlearning granularities, domain-level and instance-level, and propose BiForget, an automated framework for synthesizing high-quality forget sets. Unlike prior work relying on external generators, BiForget exploits the target model per se to elicit data that matches its internal knowledge distribution through seed-guided and adversarial prompting. Our experiments across diverse benchmarks show that it achieves a superior balance of relevance, diversity, and efficiency. Quantitatively, in the Harry Potter domain, it improves relevance by ${\sim}20$ and diversity by ${\sim}$0.05 while halving the total data size compared to SOTAs. Ultimately, it facilitates more robust forgetting and better utility preservation, providing a more rigorous foundation for evaluating LLM unlearning. 

---
# Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs 

**Authors**: Dinesh Srivasthav P, Ashok Urlana, Rahul Mishra, Bala Mallikarjunarao Garlapati, Ponnurangam Kumaraguru  

**Link**: [PDF](https://arxiv.org/pdf/2601.04275)  

**Abstract**: Machine unlearning aims to selectively remove the influence of specific training samples to satisfy privacy regulations such as the GDPR's 'Right to be Forgotten'. However, many existing methods require access to the data being removed, exposing it to membership inference attacks and potential misuse of Personally Identifiable Information (PII). We address this critical challenge by proposing Shadow Unlearning, a novel paradigm of approximate unlearning, that performs machine unlearning on anonymized forget data without exposing PII. We further propose a novel privacy-preserving framework, Neuro-Semantic Projector Unlearning (NSPU) to achieve Shadow unlearning. To evaluate our method, we compile Multi-domain Fictitious Unlearning (MuFU) forget set across five diverse domains and introduce an evaluation stack to quantify the trade-off between knowledge retention and unlearning effectiveness. Experimental results on various LLMs show that NSPU achieves superior unlearning performance, preserves model utility, and enhances user privacy. Additionally, the proposed approach is at least 10 times more computationally efficient than standard unlearning approaches. Our findings foster a new direction for privacy-aware machine unlearning that balances data protection and model fidelity. 

---
# Learning to Reason: Temporal Saliency Distillation for Interpretable Knowledge Transfer 

**Authors**: Nilushika Udayangani Hewa Dehigahawattage, Kishor Nandakishor, Marimuthu Palaniswami  

**Link**: [PDF](https://arxiv.org/pdf/2601.04263)  

**Abstract**: Knowledge distillation has proven effective for model compression by transferring knowledge from a larger network called the teacher to a smaller network called the student. Current knowledge distillation in time series is predominantly based on logit and feature aligning techniques originally developed for computer vision tasks. These methods do not explicitly account for temporal data and fall short in two key aspects. First, the mechanisms by which the transferred knowledge helps the student model learning process remain unclear due to uninterpretability of logits and features. Second, these methods transfer only limited knowledge, primarily replicating the teacher predictive accuracy. As a result, student models often produce predictive distributions that differ significantly from those of their teachers, hindering their safe substitution for teacher models. In this work, we propose transferring interpretable knowledge by extending conventional logit transfer to convey not just the right prediction but also the right reasoning of the teacher. Specifically, we induce other useful knowledge from the teacher logits termed temporal saliency which captures the importance of each input timestep to the teacher prediction. By training the student with Temporal Saliency Distillation we encourage it to make predictions based on the same input features as the teacher. Temporal Saliency Distillation requires no additional parameters or architecture specific assumptions. We demonstrate that Temporal Saliency Distillation effectively improves the performance of baseline methods while also achieving desirable properties beyond predictive accuracy. We hope our work establishes a new paradigm for interpretable knowledge distillation in time series analysis. 

---
# Safety-Utility Conflicts Are Not Global: Surgical Alignment via Head-Level Diagnosis 

**Authors**: Wang Cai, Yilin Wen, Jinchang Hou, Du Su, Guoqiu Wang, Zhonghou Lv, Chenfu Bao, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04262)  

**Abstract**: Safety alignment in Large Language Models (LLMs) inherently presents a multi-objective optimization conflict, often accompanied by an unintended degradation of general capabilities. Existing mitigation strategies typically rely on global gradient geometry to resolve these conflicts, yet they overlook Modular Heterogeneity within Transformers, specifically that the functional sensitivity and degree of conflict vary substantially across different attention heads. Such global approaches impose uniform update rules across all parameters, often resulting in suboptimal trade-offs by indiscriminately updating utility sensitive heads that exhibit intense gradient conflicts. To address this limitation, we propose Conflict-Aware Sparse Tuning (CAST), a framework that integrates head-level diagnosis with sparse fine-tuning. CAST first constructs a pre-alignment conflict map by synthesizing Optimization Conflict and Functional Sensitivity, which then guides the selective update of parameters. Experiments reveal that alignment conflicts in LLMs are not uniformly distributed. We find that the drop in general capabilities mainly comes from updating a small group of ``high-conflict'' heads. By simply skipping these heads during training, we significantly reduce this loss without compromising safety, offering an interpretable and parameter-efficient approach to improving the safety-utility trade-off. 

---
# Using Grok to Avoid Personal Attacks While Correcting Misinformation on X 

**Authors**: Kevin Matthe Caramancion  

**Link**: [PDF](https://arxiv.org/pdf/2601.04251)  

**Abstract**: Correcting misinformation in public online spaces often exposes users to hostility and ad hominem attacks, discouraging participation in corrective discourse. This study presents empirical evidence that invoking Grok, the native large language model on X, rather than directly confronting other users, is associated with different social responses during misinformation correction. Using an observational design, 100 correction replies across five high-conflict misinformation topics were analyzed, with corrections balanced between Grok-mediated and direct human-issued responses. The primary outcome was whether a correction received at least one ad hominem attack within a 24-hour window. Ad hominem attacks occurred in 72 percent of human-issued corrections and in none of the Grok-mediated corrections. A chi-square test confirmed a statistically significant association with a large effect size. These findings suggest that AI-mediated correction may alter the social dynamics of public disagreement by reducing interpersonal hostility during misinformation responses. 

---
# Green MLOps: Closed-Loop, Energy-Aware Inference with NVIDIA Triton, FastAPI, and Bio-Inspired Thresholding 

**Authors**: Mustapha Hamdi, Mourad Jabou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04250)  

**Abstract**: Energy efficiency is a first-order concern in AI deployment, as long-running inference can exceed training in cumulative carbon impact. We propose a bio-inspired framework that maps protein-folding energy basins to inference cost landscapes and controls execution via a decaying, closed-loop threshold. A request is admitted only when the expected utility-to-energy trade-off is favorable (high confidence/utility at low marginal energy and congestion), biasing operation toward the first acceptable local basin rather than pursuing costly global minima. We evaluate DistilBERT and ResNet-18 served through FastAPI with ONNX Runtime and NVIDIA Triton on an RTX 4000 Ada GPU. Our ablation study reveals that the bio-controller reduces processing time by 42% compared to standard open-loop execution (0.50s vs 0.29s on A100 test set), with a minimal accuracy degradation (<0.5%). Furthermore, we establish the efficiency boundaries between lightweight local serving (ORT) and managed batching (Triton). The results connect biophysical energy models to Green MLOps and offer a practical, auditable basis for closed-loop energy-aware inference in production. 

---
# Beyond Immediate Activation: Temporally Decoupled Backdoor Attacks on Time Series Forecasting 

**Authors**: Zhixin Liu, Xuanlin Liu, Sihan Xu, Yaqiong Qiao, Ying Zhang, Xiangrui Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.04247)  

**Abstract**: Existing backdoor attacks on multivariate time series (MTS) forecasting enforce strict temporal and dimensional coupling between triggers and target patterns, requiring synchronous activation at fixed positions across variables. However, realistic scenarios often demand delayed and variable-specific activation. We identify this critical unmet need and propose TDBA, a temporally decoupled backdoor attack framework for MTS forecasting. By injecting triggers that encode the expected location of the target pattern, TDBA enables the activation of the target pattern at any positions within the forecasted data, with the activation position flexibly varying across different variable dimensions. TDBA introduces two core modules: (1) a position-guided trigger generation mechanism that leverages smoothed Gaussian priors to generate triggers that are position-related to the predefined target pattern; and (2) a position-aware optimization module that assigns soft weights based on trigger completeness, pattern coverage, and temporal offset, facilitating targeted and stealthy attack optimization. Extensive experiments on real-world datasets show that TDBA consistently outperforms existing baselines in effectiveness while maintaining good stealthiness. Ablation studies confirm the controllability and robustness of its design. 

---
# AI Agents as Policymakers in Simulated Epidemics 

**Authors**: Goshi Aoki, Navid Ghaffarzadegan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04245)  

**Abstract**: AI agents are increasingly deployed as quasi-autonomous systems for specialized tasks, yet their potential as computational models of decision-making remains underexplored. We develop a generative AI agent to study repetitive policy decisions during an epidemic, embedding the agent, prompted to act as a city mayor, within a simulated SEIR environment. Each week, the agent receives updated epidemiological information, evaluates the evolving situation, and sets business restriction levels. The agent is equipped with a dynamic memory that weights past events by recency and is evaluated in both single- and ensemble-agent settings across environments of varying complexity. Across scenarios, the agent exhibits human-like reactive behavior, tightening restrictions in response to rising cases and relaxing them as risk declines. Crucially, providing the agent with brief systems-level knowledge of epidemic dynamics, highlighting feedbacks between disease spread and behavioral responses, substantially improves decision quality and stability. The results illustrate how theory-informed prompting can shape emergent policy behavior in AI agents. These findings demonstrate that generative AI agents, when situated in structured environments and guided by minimal domain theory, can serve as powerful computational models for studying decision-making and policy design in complex social systems. 

---
# Integrating Multi-Agent Simulation, Behavioral Forensics, and Trust-Aware Machine Learning for Adaptive Insider Threat Detection 

**Authors**: Firdous Kausar, Asmah Muallem, Naw Safrin Sattar, Mohamed Zakaria Kurdi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04243)  

**Abstract**: We present a hybrid framework for adaptive insider-threat detection that tightly integrates multi-agent simulation (MAS), layered Security Information and Event Management (SIEM) correlation, behavioral and communication forensics, trust-aware machine learning, and Theory-of-Mind (ToM) reasoning. Intelligent agents operate in a simulated enterprise environment, generating both behavioral events and cognitive intent signals that are ingested by a centralized SIEM. We evaluate four system variants: a Layered SIEM-Core (LSC) baseline, a Cognitive-Enriched SIEM (CE-SIEM) incorporating ToM and communication forensics, an Evidence-Gated SIEM (EG-SIEM) introducing precision-focused validation mechanisms, and an Enron-enabled EG-SIEM (EG-SIEM-Enron) that augments evidence gating with a pretrained email forensics module calibrated on Enron corpora. Across ten simulation runs involving eight malicious insiders, CE-SIEM achieves perfect recall (1.000) and improves actor-level F1 from 0.521 (LSC) to 0.774. EG-SIEM raises actor-level F1 to 0.922 and confirmed-alert precision to 0.997 while reducing false positives to 0.2 per run. EG-SIEM-Enron preserves high precision (1.000 confirmed-alert precision; 0.0 false positives per run), slightly improves actor-level F1 to 0.933, and reduces detection latency (average TTD 10.26 steps versus 15.20 for EG-SIEM). These results demonstrate that cognitive context improves sensitivity, evidence-gated validation enables high-precision, low-noise detection, and pretrained communication calibration can further accelerate high-confidence insider threat identification. 

---
# SmoothSync: Dual-Stream Diffusion Transformers for Jitter-Robust Beat-Synchronized Gesture Generation from Quantized Audio 

**Authors**: Yujiao Jiang, Qingmin Liao, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04236)  

**Abstract**: Co-speech gesture generation is a critical area of research aimed at synthesizing speech-synchronized human-like gestures. Existing methods often suffer from issues such as rhythmic inconsistency, motion jitter, foot sliding and limited multi-sampling diversity. In this paper, we present SmoothSync, a novel framework that leverages quantized audio tokens in a novel dual-stream Diffusion Transformer (DiT) architecture to synthesis holistic gestures and enhance sampling variation. Specifically, we (1) fuse audio-motion features via complementary transformer streams to achieve superior synchronization, (2) introduce a jitter-suppression loss to improve temporal smoothness, (3) implement probabilistic audio quantization to generate distinct gesture sequences from identical inputs. To reliably evaluate beat synchronization under jitter, we introduce Smooth-BC, a robust variant of the beat consistency metric less sensitive to motion noise. Comprehensive experiments on the BEAT2 and SHOW datasets demonstrate SmoothSync's superiority, outperforming state-of-the-art methods by -30.6% FGD, 10.3% Smooth-BC, and 8.4% Diversity on BEAT2, while reducing jitter and foot sliding by -62.9% and -17.1% respectively. The code will be released to facilitate future research. 

---
# Defense Against Synthetic Speech: Real-Time Detection of RVC Voice Conversion Attacks 

**Authors**: Prajwal Chinchmalatpure, Suyash Chinchmalatpure, Siddharth Chavan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04227)  

**Abstract**: Generative audio technologies now enable highly realistic voice cloning and real-time voice conversion, increasing the risk of impersonation, fraud, and misinformation in communication channels such as phone and video calls. This study investigates real-time detection of AI-generated speech produced using Retrieval-based Voice Conversion (RVC), evaluated on the DEEP-VOICE dataset, which includes authentic and voice-converted speech samples from multiple well-known speakers. To simulate realistic conditions, deepfake generation is applied to isolated vocal components, followed by the reintroduction of background ambiance to suppress trivial artifacts and emphasize conversion-specific cues. We frame detection as a streaming classification task by dividing audio into one-second segments, extracting time-frequency and cepstral features, and training supervised machine learning models to classify each segment as real or voice-converted. The proposed system enables low-latency inference, supporting both segment-level decisions and call-level aggregation. Experimental results show that short-window acoustic features can reliably capture discriminative patterns associated with RVC speech, even in noisy backgrounds. These findings demonstrate the feasibility of practical, real-time deepfake speech detection and underscore the importance of evaluating under realistic audio mixing conditions for robust deployment. 

---
# Can Consumer Chatbots Reason? A Student-Led Field Experiment Embedded in an "AI-for-All" Undergraduate Course 

**Authors**: Amarda Shehu, Adonyas Ababu, Asma Akbary, Griffin Allen, Aroush Baig, Tereana Battle, Elias Beall, Christopher Byrom, Matt Dean, Kate Demarco, Ethan Douglass, Luis Granados, Layla Hantush, Andy Hay, Eleanor Hay, Caleb Jackson, Jaewon Jang, Carter Jones, Quanyang Li, Adrian Lopez, Logan Massimo, Garrett McMullin, Ariana Mendoza Maldonado, Eman Mirza, Hadiya Muddasar, Sara Nuwayhid, Brandon Pak, Ashley Petty, Dryden Rancourt, Lily Rodriguez, Corbin Rogers, Jacob Schiek, Taeseo Seok, Aarav Sethi, Giovanni Vitela, Winston Williams, Jagan Yetukuri  

**Link**: [PDF](https://arxiv.org/pdf/2601.04225)  

**Abstract**: Claims about whether large language model (LLM) chatbots "reason" are typically debated using curated benchmarks and laboratory-style evaluation protocols. This paper offers a complementary perspective: a student-led field experiment embedded as a midterm project in UNIV 182 (AI4All) at George Mason University, a Mason Core course designed for undergraduates across disciplines with no expected prior STEM exposure. Student teams designed their own reasoning tasks, ran them on widely used consumer chatbots representative of current capabilities, and evaluated both (i) answer correctness and (ii) the validity of the chatbot's stated reasoning (for example, cases where an answer is correct but the explanation is not, or vice versa). Across eight teams that reported standardized scores, students contributed 80 original reasoning prompts spanning six categories: pattern completion, transformation rules, spatial/visual reasoning, quantitative reasoning, relational/logic reasoning, and analogical reasoning. These prompts yielded 320 model responses plus follow-up explanations. Aggregating team-level results, OpenAI GPT-5 and Claude 4.5 achieved the highest mean answer accuracy (86.2% and 83.8%), followed by Grok 4 (82.5%) and Perplexity (73.1%); explanation validity showed a similar ordering (81.2%, 80.0%, 77.5%, 66.2%). Qualitatively, teams converged on a consistent error signature: strong performance on short, structured math and pattern items but reduced reliability on spatial/visual reasoning and multi-step transformations, with frequent "sound right but reason wrong" explanations. The assignment's primary contribution is pedagogical: it operationalizes AI literacy as experimental practice (prompt design, measurement, rater disagreement, and interpretability/grounding) while producing a reusable, student-generated corpus of reasoning probes grounded in authentic end-user interaction. 

---
# Beyond Interaction Effects: Two Logics for Studying Population Inequalities 

**Authors**: Adel Daoud  

**Link**: [PDF](https://arxiv.org/pdf/2601.04223)  

**Abstract**: When sociologists and other social scientist ask whether the return to college differs by race and gender, they face a choice between two fundamentally different modes of inquiry. Traditional interaction models follow deductive logic: the researcher specifies which variables moderate effects and tests these hypotheses. Machine learning methods follow inductive logic: algorithms search across vast combinatorial spaces to discover patterns of heterogeneity. This article develops a framework for navigating between these approaches. We show that the choice between deduction and induction reflects a tradeoff between interpretability and flexibility, and we demonstrate through simulation when each approach excels. Our framework is particularly relevant for inequality research, where understanding how treatment effects vary across intersecting social subpopulation is substantively central. 

---
# AgentTutor: Empowering Personalized Learning with Multi-Turn Interactive Teaching in Intelligent Education Systems 

**Authors**: Yuxin Liu, Zeqing Song, Jiong Lou, Chentao Wu, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04219)  

**Abstract**: The rapid advancement of large-scale language models (LLMs) has shown their potential to transform intelligent education systems (IESs) through automated teaching and learning support applications. However, current IESs often rely on single-turn static question-answering, which fails to assess learners' cognitive levels, cannot adjust teaching strategies based on real-time feedback, and is limited to providing simple one-off responses. To address these issues, we introduce AgentTutor, a multi-turn interactive intelligent education system to empower personalized learning. It features an LLM-powered generative multi-agent system and a learner-specific personalized learning profile environment that dynamically optimizes and delivers teaching strategies based on learners' learning status, personalized goals, learning preferences, and multimodal study materials. It includes five key modules: curriculum decomposition, learner assessment, dynamic strategy, teaching reflection, and knowledge & experience memory. We conducted extensive experiments on multiple benchmark datasets, AgentTutor significantly enhances learners' performance while demonstrating strong effectiveness in multi-turn interactions and competitiveness in teaching quality among other baselines. 

---
# The Artificial Intelligence Value Chain: A Critical Appraisal. [Spanish Version] 

**Authors**: Pompeu Casanovas  

**Link**: [PDF](https://arxiv.org/pdf/2601.04218)  

**Abstract**: The artificial intelligence value chain is one of the main concepts underpinning the European legislation on the subject, especially the Artificial Intelligence Act. It is an economic concept that has become a legal one. i.e., a concept of legal governance, due to its continued use in policy documents and legal texts. This article (i) analyses its significance and function within the framework of the regulatory strategy established by recent EU programs (the Compass for Competitiveness, the Action Plan, Apply AI Strategy, and the Digital Omnibus on AI), (ii) identifies its limitations, and (iii) advances the theoretical construction of value chains that capture intangible dimensions that are not directly monetizable (such as language, culture, and, especially, ethical and legal values) but have a significant impact on the social environment. It also briefly compares three different legal frameworks for the regulation of AI (EU, Commonwealth and USA). It proposes at the end a specific framework for the analysis of the ethical and legal AI value chain to preserve democratic values and foster the digital implementation of the rule of law. 

---
# Attachment Styles and AI Chatbot Interactions Among College Students 

**Authors**: Ziqi Lin, Taiyu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04217)  

**Abstract**: The use of large language model (LLM)-based AI chatbots among college students has increased rapidly, yet little is known about how individual psychological attributes shape students' interaction patterns with these technologies. This qualitative study explored how college students with different attachment styles describe their interactions with ChatGPT. Using semi-structured interviews with seven undergraduate students and grounded theory analysis, we identified three main themes: (1) AI as a low-risk emotional space, where participants across attachment styles valued the non-judgmental and low-stakes nature of AI interactions; (2) attachment-congruent patterns of AI engagement, where securely attached students integrated AI as a supplementary tool within their existing support systems, while avoidantly attached students used AI to buffer vulnerability and maintain interpersonal boundaries; and (3) the paradox of AI intimacy, capturing the tension between students' willingness to disclose personal information to AI while simultaneously recognizing its limitations as a relational partner. These findings suggest that attachment orientations play an important role in shaping how students experience and interpret their interactions with AI chatbots, extending attachment theory to the domain of human-AI interaction. 

---
# Computable Gap Assessment of Artificial Intelligence Governance in Children's Centres:Evidence-Mechanism-Governance-Indicator Modelling of UNICEF's Guidance on AI and Children 3.0 Based on the Graph-GAP Framework 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2601.04216)  

**Abstract**: This paper tackles practical challenges in governing child centered artificial intelligence: policy texts state principles and requirements but often lack reproducible evidence anchors, explicit causal pathways, executable governance toolchains, and computable audit metrics. We propose Graph-GAP, a methodology that decomposes requirements from authoritative policy texts into a four layer graph of evidence, mechanism, governance, and indicator, and that computes two metrics, GAP score and mitigation readiness, to identify governance gaps and prioritise actions. Using the UNICEF Innocenti Guidance on AI and Children 3.0 as primary material, we define reproducible extraction units, coding manuals, graph patterns, scoring scales, and consistency checks, and we demonstrate exemplar gap profiles and governance priority matrices for ten requirements. Results suggest that compared with privacy and data protection, requirements related to child well being and development, explainability and accountability, and cross agency implementation and resource allocation are more prone to indicator gaps and mechanism gaps. We recommend translating requirements into auditable closed loop governance that integrates child rights impact assessments, continuous monitoring metrics, and grievance redress procedures. At the coding level, we introduce a multi algorithm review aggregation revision workflow that runs rule based encoders, statistical or machine learning evaluators, and large model evaluators with diverse prompt configurations as parallel coders. Each extraction unit outputs evidence, mechanism, governance, and indicator labels plus readiness scores with evidence anchors. Reliability, stability, and uncertainty are assessed using Krippendorff alpha, weighted kappa, intraclass correlation, and bootstrap confidence intervals. 

---
# TrueBrief: Faithful Summarization through Small Language Models 

**Authors**: Kumud Lakara, Ruibo Shi, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04212)  

**Abstract**: Large language models (LLMs) have exhibited remarkable proficiency in generating high-quality text; however, their propensity for producing hallucinations poses a significant challenge for their deployment in security-critical domains. In this work, we present TrueBrief, an end-to-end framework specifically designed to enhance the faithfulness of small LLMs (SLMs) primarily for the task of text summarization through a preference-optimization paradigm. Central to our framework is a data generation module that facilitates controlled hallucination injection to generate synthetic preference data. Our work provides insights into the impact of data quality and model size on preference-based optimization, highlighting the conditions under which these methods are most effective. 

---
# Complexity Agnostic Recursive Decomposition of Thoughts 

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang, Hafiz Saif Ur Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2601.04210)  

**Abstract**: Large language models often fail on multi-step reasoning due to fixed reasoning strategies that ignore problem specific difficulty. We introduce CARD (Complexity Agnostic Recursive Decomposition), a framework that predicts problem complexity before generation and adapts decomposition accordingly. Our system comprises MRCE (Multi-dimensional Reasoning Complexity Estimator), a 0.6B Qwen model predicting 30 fine-grained features from question text and a two-stage recursive solver: (1) hierarchical decomposition into K steps based on task profile and (2) per-step thought budget allocation (1, 5-9, or 10 thoughts) via recursive MRCE profiling. Evaluated on three reasoning models (Qwen3-0.6B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-1.7B), CARD achieves 81.4% to 89.2% accuracy on GSM8K while reducing token cost by 1.88x to 2.40x compared to fixed decomposition baselines. On MATH-500, CARD reaches 75.1 to 86.8% accuracy using 1.71x to 5.74x fewer tokens. Our results demonstrate that preemptive complexity estimation enables both higher accuracy and significant efficiency gains. 

---
# LLMs for Explainable Business Decision-Making: A Reinforcement Learning Fine-Tuning Approach 

**Authors**: Xiang Cheng, Wen Wang, Anindya Ghose  

**Link**: [PDF](https://arxiv.org/pdf/2601.04208)  

**Abstract**: Artificial Intelligence (AI) models increasingly drive high-stakes consumer interactions, yet their decision logic often remains opaque. Prevailing explainable AI techniques rely on post hoc numerical feature attributions, which fail to provide coherent narratives behind model decisions. Large language models (LLMs) present an opportunity to generate natural-language explanations, but three design challenges remain unresolved: explanations must be both decision-correct and faithful to the factors that drive the prediction; they should be able to serve multiple audiences without shifting the underlying decision rule; and they should be trained in a label-efficient way that does not depend on large corpora of human-scored explanations. To address these challenges, we introduce LEXMA (LLM-based EXplanations for Multi-Audience decisions), a reinforcement-learning-based fine-tuning framework that produces narrative-driven, audience-appropriate explanations. LEXMA combines reflection-augmented supervised fine-tuning with two stages of Group Relative Policy Optimization (GRPO). Specifically, it fine-tunes two separate parameter sets to improve decision correctness and satisfy stylistic requirements for different audiences, using reward signals that do not rely on human-annotated explanations. We instantiate LEXMA in the context of mortgage approval decisions. Results demonstrate that LEXMA yields significant improvements in predictive performance compared with other LLM baselines. Moreover, human evaluations show that expert-facing explanations generated by our approach are more risk-focused, and consumer-facing explanations are clearer, more actionable, and more polite. Our study contributes a cost-efficient, systematic LLM fine-tuning approach to enhance explanation quality for business decisions, offering strong potential for scalable deployment of transparent AI systems. 

---
# Ideology as a Problem: Lightweight Logit Steering for Annotator-Specific Alignment in Social Media Analysis 

**Authors**: Wei Xia, Haowen Tang, Luozheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04207)  

**Abstract**: LLMs internally organize political ideology along low-dimensional structures that are partially, but not fully aligned with human ideological space. This misalignment is systematic, model specific, and measurable. We introduce a lightweight linear probe that both quantifies the misalignment and minimally corrects the output layer. This paper introduces a simple and efficient method for aligning models with specific user opinions. Instead of retraining the model, we calculated a bias score from its internal features and directly adjusted the final output probabilities. This solution is practical and low-cost and preserves the original reasoning power of the model. 

---
# STDD:Spatio-Temporal Dynamics-Driven Token Refinement in Diffusion Language Models 

**Authors**: Xinhao Sun, Maoliang Li, Zihao Zheng, Jiayu Chen, Hezhao Xu, Yun Liang, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04205)  

**Abstract**: Unlike autoregressive language models, diffusion language models (DLMs) generate text by iteratively denoising all token positions in parallel. At each timestep, the remasking strategy of a DLM selects low- priority tokens to defer their decoding, thereby improving both efficiency and output quality. However, mainstream remasking strategies rely on a single global confidence threshold, overlooking the temporal and spatial dynamics of individual tokens. Motivated by the redundant iterations and constrained parallelism introduced by fixed-threshold remasking, we propose a novel remasking approach that dynamically detects Temporal Variance and Spa- tial Deviance of each token, which reflect its convergence status and inter-token correlations. Using these signals, our method adaptively adjusts the confidence threshold for every token at every step. Empirical re- sults show that our approach significantly improves the operational efficiency of DLMs across mainstream datasets, achieving speedups of up to 8.9 times while faithfully preserving generation quality. 

---
# Generative Teaching via Code 

**Authors**: Yuheng Wang, Runde Yang, Lin Wu, Jie Zhang, Jingru Fan, Ruoyu Fu, Tianle Zhou, Huatao Li, Siheng Chen, Weinan E, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.04204)  

**Abstract**: The scalability of high-quality online education is hindered by the high costs and slow cycles of labor-intensive manual content creation. Despite advancements in video generation, current approaches often fail to ensure pedagogical structure and precise control due to their pixel-level, black-box nature. In this paper, we propose Generative Teaching, a novel paradigm that transitions educators from manual creators to high-level directors, allowing them to focus on pedagogical intent while autonomous agents handle the execution. To realize this vision, we introduce TeachMaster, a multi-agent framework that leverages code as an intermediate semantic medium. Unlike traditional video generation methods, TeachMaster orchestrates a collaborative team of agents--spanning planning, design, and rendering--to automate the production of interpretable, editable, and curriculum-ready educational videos. Experiments validate that TeachMaster significantly boosts production efficiency without compromising structural coherence or visual fidelity, providing a robust solution for scalable education. 

---
# TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation 

**Authors**: Anas Ezzakri, Nicola Piovesan, Mohamed Sana, Antonio De Domenico, Fadhel Ayed, Haozhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04202)  

**Abstract**: Language Models (LLMs) are increasingly explored in the telecom industry to support engineering tasks, accelerate troubleshooting, and assist in interpreting complex technical documents. However, recent studies show that LLMs perform poorly on telecom standards, particularly 3GPP specifications. We argue that a key reason is that these standards densely include tables to present essential information, yet the LLM knowledge and interpretation ability of such tables remains largely unexamined. To address this gap, we introduce TeleTables, a benchmark designed to evaluate both the implicit knowledge LLMs have about tables in technical specifications and their explicit ability to interpret them. TeleTables is built through a novel multi-stage data generation pipeline that extracts tables from 3GPP standards and uses multimodal and reasoning-oriented LLMs to generate and validate questions. The resulting dataset, which is publicly available, comprises 500 human-verified question-answer pairs, each associated with the corresponding table in multiple formats. Our evaluation shows that, smaller models (under 10B parameters) struggle both to recall 3GPP knowledge and to interpret tables, indicating the limited exposure to telecom standards in their pretraining and the insufficient inductive biases for navigating complex technical material. Larger models, on the other hand, show stronger reasoning on table interpretation. Overall, TeleTables highlights the need for domain-specialized fine-tuning to reliably interpret and reason over telecom standards. 

---
# Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems 

**Authors**: Zihan Gao, Mohsin Y. K. Yousufi, Jacob Thebault-Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2601.04201)  

**Abstract**: Large language model (LLM) question-answering systems often fail on community-specific queries, creating "knowledge blind spots" that marginalize local voices and reinforce epistemic injustice. We present Collective Narrative Grounding, a participatory protocol that transforms community stories into structured narrative units and integrates them into AI systems under community governance. Learning from three participatory mapping workshops with N=24 community members, we designed elicitation methods and a schema that retain narrative richness while enabling entity, time, and place extraction, validation, and provenance control. To scope the problem, we audit a county-level benchmark of 14,782 local information QA pairs, where factual gaps, cultural misunderstandings, geographic confusions, and temporal misalignments account for 76.7% of errors. On a participatory QA set derived from our workshops, a state-of-the-art LLM answered fewer than 21% of questions correctly without added context, underscoring the need for local grounding. The missing facts often appear in the collected narratives, suggesting a direct path to closing the dominant error modes for narrative items. Beyond the protocol and pilot, we articulate key design tensions, such as representation and power, governance and control, and privacy and consent, providing concrete requirements for retrieval-first, provenance-visible, locally governed QA systems. Together, our taxonomy, protocol, and participatory evaluation offer a rigorous foundation for building community-grounded AI that better answers local questions. 

---
# Attribute-Aware Controlled Product Generation with LLMs for E-commerce 

**Authors**: Virginia Negri, Víctor Martínez Gómez, Sergio A. Balanya, Subburam Rajaram  

**Link**: [PDF](https://arxiv.org/pdf/2601.04200)  

**Abstract**: Product information extraction is crucial for e-commerce services, but obtaining high-quality labeled datasets remains challenging. We present a systematic approach for generating synthetic e-commerce product data using Large Language Models (LLMs), introducing a controlled modification framework with three strategies: attribute-preserving modification, controlled negative example generation, and systematic attribute removal. Using a state-of-the-art LLM with attribute-aware prompts, we enforce store constraints while maintaining product coherence. Human evaluation of 2000 synthetic products demonstrates high effectiveness, with 99.6% rated as natural, 96.5% containing valid attribute values, and over 90% showing consistent attribute usage. On the public MAVE dataset, our synthetic data achieves 60.5% accuracy, performing on par with real training data (60.8%) and significantly improving upon the 13.4% zero-shot baseline. Hybrid configurations combining synthetic and real data further improve performance, reaching 68.8% accuracy. Our framework provides a practical solution for augmenting e-commerce datasets, particularly valuable for low-resource scenarios. 

---
# The Forgotten Shield: Safety Grafting in Parameter-Space for Medical MLLMs 

**Authors**: Jiale Zhao, Xing Mou, Jinlin Wu, Hongyuan Yu, Mingrui Sun, Yang Shi, Xuanwu Yin, Zhen Chen, Zhen Lei, Yaohua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04199)  

**Abstract**: Medical Multimodal Large Language Models (Medical MLLMs) have achieved remarkable progress in specialized medical tasks; however, research into their safety has lagged, posing potential risks for real-world deployment. In this paper, we first establish a multidimensional evaluation framework to systematically benchmark the safety of current SOTA Medical MLLMs. Our empirical analysis reveals pervasive vulnerabilities across both general and medical-specific safety dimensions in existing models, particularly highlighting their fragility against cross-modality jailbreak attacks. Furthermore, we find that the medical fine-tuning process frequently induces catastrophic forgetting of the model's original safety alignment. To address this challenge, we propose a novel "Parameter-Space Intervention" approach for efficient safety re-alignment. This method extracts intrinsic safety knowledge representations from original base models and concurrently injects them into the target model during the construction of medical capabilities. Additionally, we design a fine-grained parameter search algorithm to achieve an optimal trade-off between safety and medical performance. Experimental results demonstrate that our approach significantly bolsters the safety guardrails of Medical MLLMs without relying on additional domain-specific safety data, while minimizing degradation to core medical performance. 

---
# MedPI: Evaluating AI Systems in Medical Patient-facing Interactions 

**Authors**: Diego Fajardo V., Oleksii Proniakin, Victoria-Elisabeth Gruber, Razvan Marinescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04195)  

**Abstract**: We present MedPI, a high-dimensional benchmark for evaluating large language models (LLMs) in patient-clinician conversations. Unlike single-turn question-answer (QA) benchmarks, MedPI evaluates the medical dialogue across 105 dimensions comprising the medical process, treatment safety, treatment outcomes and doctor-patient communication across a granular, accreditation-aligned rubric. MedPI comprises five layers: (1) Patient Packets (synthetic EHR-like ground truth); (2) an AI Patient instantiated through an LLM with memory and affect; (3) a Task Matrix spanning encounter reasons (e.g. anxiety, pregnancy, wellness checkup) x encounter objectives (e.g. diagnosis, lifestyle advice, medication advice); (4) an Evaluation Framework with 105 dimensions on a 1-4 scale mapped to the Accreditation Council for Graduate Medical Education (ACGME) competencies; and (5) AI Judges that are calibrated, committee-based LLMs providing scores, flags, and evidence-linked rationales. We evaluate 9 flagship models -- Claude Opus 4.1, Claude Sonnet 4, MedGemma, Gemini 2.5 Pro, Llama 3.3 70b Instruct, GPT-5, GPT OSS 120b, o3, Grok-4 -- across 366 AI Patients and 7,097 conversations using a standardized "vanilla clinician" prompt. For all LLMs, we observe low performance across a variety of dimensions, in particular on differential diagnosis. Our work can help guide future use of LLMs for diagnosis and treatment recommendations. 

---
# Safe in the Future, Dangerous in the Past: Dissecting Temporal and Linguistic Vulnerabilities in LLMs 

**Authors**: Muhammad Abdullahi Said, Muhammad Sammani Sani  

**Link**: [PDF](https://arxiv.org/pdf/2512.24556)  

**Abstract**: As Large Language Models (LLMs) integrate into critical global infrastructure, the assumption that safety alignment transfers zero-shot from English to other languages remains a dangerous blind spot. This study presents a systematic audit of three state of the art models (GPT-5.1, Gemini 3 Pro, and Claude 4.5 Opus) using HausaSafety, a novel adversarial dataset grounded in West African threat scenarios (e.g., Yahoo-Yahoo fraud, Dane gun manufacturing). Employing a 2 x 4 factorial design across 1,440 evaluations, we tested the non-linear interaction between language (English vs. Hausa) and temporal framing. Our results challenge the narrative of the multilingual safety gap. Instead of a simple degradation in low-resource settings, we identified a complex interference mechanism in which safety is determined by the intersection of variables. Although the models exhibited a reverse linguistic vulnerability with Claude 4.5 Opus proving significantly safer in Hausa (45.0%) than in English (36.7%) due to uncertainty-driven refusal, they suffered catastrophic failures in temporal reasoning. We report a profound Temporal Asymmetry, where past-tense framing bypassed defenses (15.6% safe) while future-tense scenarios triggered hyper-conservative refusals (57.2% safe). The magnitude of this volatility is illustrated by a 9.2x disparity between the safest and most vulnerable configurations, proving that safety is not a fixed property but a context-dependent state. We conclude that current models rely on superficial heuristics rather than robust semantic understanding, creating Safety Pockets that leave Global South users exposed to localized harms. We propose Invariant Alignment as a necessary paradigm shift to ensure safety stability across linguistic and temporal shifts. 

---
