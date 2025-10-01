# Using GPT to build a Project Management assistant for Jira environments 

**Authors**: Joel Garcia-Escribano, Arkaitz Carbajo, Mikel Egaña Aranguren, Unai Lopez-Novoa  

**Link**: [PDF](https://arxiv.org/pdf/2509.26014)  

**Abstract**: In the domain of Project Management, the sheer volume of data is a challenge that project managers continually have to deal with. Effectively steering projects from inception to completion requires handling of diverse information streams, including timelines, budgetary considerations, and task dependencies. To navigate this data-driven landscape with precision and agility, project managers must rely on efficient and sophisticated tools. These tools have become essential, as they enable project managers to streamline communication, optimize resource allocation, and make informed decisions in real-time. However, many of these tools have steep learning curves and require using complex programming languages to retrieve the exact data that project managers need. In this work we present JiraGPT Next, a software that uses the GPT Large Language Model to ease the process by which project managers deal with large amounts of data. It is conceived as an add-on for Jira, one of the most popular Project Management tools, and provides a natural language interface to retrieve information. This work presents the design decisions behind JiraGPT Next and an evaluation of the accuracy of GPT in this context, including the effects of providing different prompts to complete a particular task. 

---
# Auto-ARGUE: LLM-Based Report Generation Evaluation 

**Authors**: William Walden, Marc Mason, Orion Weller, Laura Dietz, Hannah Recknor, Bryan Li, Gabrielle Kaili-May Liu, Yu Hou, James Mayfield, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26184)  

**Abstract**: Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recent ARGUE framework for report generation evaluation. We present analysis of Auto-ARGUE on the report generation pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs. 

---
# Causal Autoencoder-like Generation of Feedback Fuzzy Cognitive Maps with an LLM Agent 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2509.25593)  

**Abstract**: A large language model (LLM) can map a feedback causal fuzzy cognitive map (FCM) into text and then reconstruct the FCM from the text. This explainable AI system approximates an identity map from the FCM to itself and resembles the operation of an autoencoder (AE). Both the encoder and the decoder explain their decisions in contrast to black-box AEs. Humans can read and interpret the encoded text in contrast to the hidden variables and synaptic webs in AEs. The LLM agent approximates the identity map through a sequence of system instructions that does not compare the output to the input. The reconstruction is lossy because it removes weak causal edges or rules while it preserves strong causal edges. The encoder preserves the strong causal edges even when it trades off some details about the FCM to make the text sound more natural. 

---
# TRUE: A Reproducible Framework for LLM-Driven Relevance Judgment in Information Retrieval 

**Authors**: Mouly Dewan, Jiqun Liu, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.25602)  

**Abstract**: LLM-based relevance judgment generation has become a crucial approach in advancing evaluation methodologies in Information Retrieval (IR). It has progressed significantly, often showing high correlation with human judgments as reflected in LLMJudge leaderboards \cite{rahmani2025judging}. However, existing methods for relevance judgments, rely heavily on sensitive prompting strategies, lacking standardized workflows for generating reliable labels. To fill this gap, we reintroduce our method, \textit{Task-aware Rubric-based Evaluation} (TRUE), for relevance judgment generation. Originally developed for usefulness evaluation in search sessions, we extend TRUE to mitigate the gap in relevance judgment due to its demonstrated effectiveness and reproducible workflow. This framework leverages iterative data sampling and reasoning to evaluate relevance judgments across multiple factors including intent, coverage, specificity, accuracy and usefulness. In this paper, we evaluate TRUE on the TREC DL 2019, 2020 and LLMJudge datasets and our results show that TRUE achieves strong performance on the system-ranking LLM leaderboards. The primary focus of this work is to provide a reproducible framework for LLM-based relevance judgments, and we further analyze the effectiveness of TRUE across multiple dimensions. 

---
# On-Premise AI for the Newsroom: Evaluating Small Language Models for Investigative Document Search 

**Authors**: Nick Hagar, Nicholas Diakopoulos, Jeremy Gilbert  

**Link**: [PDF](https://arxiv.org/pdf/2509.25494)  

**Abstract**: Investigative journalists routinely confront large document collections. Large language models (LLMs) with retrieval-augmented generation (RAG) capabilities promise to accelerate the process of document discovery, but newsroom adoption remains limited due to hallucination risks, verification burden, and data privacy concerns. We present a journalist-centered approach to LLM-powered document search that prioritizes transparency and editorial control through a five-stage pipeline -- corpus summarization, search planning, parallel thread execution, quality evaluation, and synthesis -- using small, locally-deployable language models that preserve data security and maintain complete auditability through explicit citation chains. Evaluating three quantized models (Gemma 3 12B, Qwen 3 14B, and GPT-OSS 20B) on two corpora, we find substantial variation in reliability. All models achieved high citation validity and ran effectively on standard desktop hardware (e.g., 24 GB of memory), demonstrating feasibility for resource-constrained newsrooms. However, systematic challenges emerged, including error propagation through multi-stage synthesis and dramatic performance variation based on training data overlap with corpus content. These findings suggest that effective newsroom AI deployment requires careful model selection and system design, alongside human oversight for maintaining standards of accuracy and accountability. 

---
# SQUARE: Semantic Query-Augmented Fusion and Efficient Batch Reranking for Training-free Zero-Shot Composed Image Retrieval 

**Authors**: Ren-Di Wu, Yu-Yen Lin, Huei-Fang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26330)  

**Abstract**: Composed Image Retrieval (CIR) aims to retrieve target images that preserve the visual content of a reference image while incorporating user-specified textual modifications. Training-free zero-shot CIR (ZS-CIR) approaches, which require no task-specific training or labeled data, are highly desirable, yet accurately capturing user intent remains challenging. In this paper, we present SQUARE, a novel two-stage training-free framework that leverages Multimodal Large Language Models (MLLMs) to enhance ZS-CIR. In the Semantic Query-Augmented Fusion (SQAF) stage, we enrich the query embedding derived from a vision-language model (VLM) such as CLIP with MLLM-generated captions of the target image. These captions provide high-level semantic guidance, enabling the query to better capture the user's intent and improve global retrieval quality. In the Efficient Batch Reranking (EBR) stage, top-ranked candidates are presented as an image grid with visual marks to the MLLM, which performs joint visual-semantic reasoning across all candidates. Our reranking strategy operates in a single pass and yields more accurate rankings. Experiments show that SQUARE, with its simplicity and effectiveness, delivers strong performance on four standard CIR benchmarks. Notably, it maintains high performance even with lightweight pre-trained, demonstrating its potential applicability. 

---
# MHINDR - a DSM5 based mental health diagnosis and recommendation framework using LLM 

**Authors**: Vaishali Agarwal, Sachin Thukral, Arnab Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.25992)  

**Abstract**: Mental health forums offer valuable insights into psychological issues, stressors, and potential solutions. We propose MHINDR, a large language model (LLM) based framework integrated with DSM-5 criteria to analyze user-generated text, dignose mental health conditions, and generate personalized interventions and insights for mental health practitioners. Our approach emphasizes on the extraction of temporal information for accurate diagnosis and symptom progression tracking, together with psychological features to create comprehensive mental health summaries of users. The framework delivers scalable, customizable, and data-driven therapeutic recommendations, adaptable to diverse clinical contexts, patient needs, and workplace well-being programs. 

---
# MENLO: From Preferences to Proficiency - Evaluating and Modeling Native-like Quality Across 47 Languages 

**Authors**: Chenxi Whitehouse, Sebastian Ruder, Tony Lin, Oksana Kurylo, Haruka Takagi, Janice Lam, Nicolò Busetto, Denise Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.26601)  

**Abstract**: Ensuring native-like quality of large language model (LLM) responses across many languages is challenging. To address this, we introduce MENLO, a framework that operationalizes the evaluation of native-like response quality based on audience design-inspired mechanisms. Using MENLO, we create a dataset of 6,423 human-annotated prompt-response preference pairs covering four quality dimensions with high inter-annotator agreement in 47 language varieties. Our evaluation reveals that zero-shot LLM judges benefit significantly from pairwise evaluation and our structured annotation rubrics, yet they still underperform human annotators on our dataset. We demonstrate substantial improvements through fine-tuning with reinforcement learning, reward shaping, and multi-task learning approaches. Additionally, we show that RL-trained judges can serve as generative reward models to enhance LLMs' multilingual proficiency, though discrepancies with human judgment remain. Our findings suggest promising directions for scalable multilingual evaluation and preference alignment. We release our dataset and evaluation framework to support further research in multilingual LLM evaluation. 

---
# Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling 

**Authors**: Seiji Maekawa, Jackson Hassell, Pouya Pezeshkpour, Tom Mitchell, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2509.26553)  

**Abstract**: As language models gain access to external tools via structured function calls, they become increasingly more capable of solving complex, multi-step tasks. However, existing benchmarks for tool-augmented language models (TaLMs) provide insufficient control over factors such as the number of functions accessible, task complexity, and input size, and remain vulnerable to data contamination. We present FuncBenchGen, a unified, contamination-free framework that evaluates TaLMs by generating synthetic multi-step tool-use tasks. The key idea is to cast tool use as traversal over a hidden function-dependency DAG where nodes are function calls and an edge between nodes represents one function consuming the output of another. Given a set of external function schemas, initial variable values, and a target variable, models must compose the correct call sequence to compute the target variable. FuncBenchGen allows users to precisely control task difficulty (e.g., graph size, dependency depth, and distractor functions) while avoiding data leakage. We apply our FuncBenchGen framework to evaluate seven LLMs on tool use tasks of varying difficulty. Reasoning-optimized models consistently outperform general-purpose models with GPT-5 significantly outperforming other models. Performance declines sharply as dependency depth increases. Furthermore, connected irrelevant functions prove especially difficult to handle. We find that strong models often make syntactically valid function calls but propagate incorrect or stale argument values across steps, revealing brittle state tracking by LLMs in multi-turn tool use. Motivated by this observation, we introduce a simple mitigation strategy that explicitly restates prior variable values to the agent at each step. Surprisingly, this lightweight change yields substantial gains across models. e.g., yielding a success rate improvement from 62.5% to 81.3% for GPT-5. 

---
# VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications 

**Authors**: Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, Yunke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26490)  

**Abstract**: As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at this https URL 

---
# dParallel: Learnable Parallel Decoding for dLLMs 

**Authors**: Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26488)  

**Abstract**: Diffusion large language models (dLLMs) have recently drawn considerable attention within the research community as a promising alternative to autoregressive generation, offering parallel token prediction and lower inference latency. Yet, their parallel decoding potential remains largely underexplored, as existing open-source models still require nearly token-length decoding steps to ensure performance. To address this, we introduce dParallel, a simple and effective method that unlocks the inherent parallelism of dLLMs for fast sampling. We identify that the key bottleneck to parallel decoding arises from the sequential certainty convergence for masked tokens. Building on this insight, we introduce the core of our approach: certainty-forcing distillation, a novel training strategy that distills the model to follow its original sampling trajectories while enforcing it to achieve high certainty on masked tokens more rapidly and in parallel. Extensive experiments across various benchmarks demonstrate that our method can dramatically reduce the number of decoding steps while maintaining performance. When applied to the LLaDA-8B-Instruct model, dParallel reduces decoding steps from 256 to 30 on GSM8K, achieving an 8.5x speedup without performance degradation. On the MBPP benchmark, it cuts decoding steps from 256 to 24, resulting in a 10.5x speedup while maintaining accuracy. Our code is available at this https URL 

---
# Generating Difficult-to-Translate Texts 

**Authors**: Vilém Zouhar, Wenda Xu, Parker Riley, Juraj Juraska, Mara Finkelstein, Markus Freitag, Dan Deutsch  

**Link**: [PDF](https://arxiv.org/pdf/2509.26592)  

**Abstract**: Machine translation benchmarks sourced from the real world are quickly obsoleted, due to most examples being easy for state-of-the-art translation models. This limits the benchmark's ability to distinguish which model is better or to reveal models' weaknesses. Current methods for creating difficult test cases, such as subsampling or from-scratch synthesis, either fall short of identifying difficult examples or suffer from a lack of diversity and naturalness. Inspired by the iterative process of human experts probing for model failures, we propose MT-breaker, a method where a large language model iteratively refines a source text to increase its translation difficulty. The LLM iteratively queries a target machine translation model to guide its generation of difficult examples. Our approach generates examples that are more challenging for the target MT model while preserving the diversity of natural texts. While the examples are tailored to a particular machine translation model during the generation, the difficulty also transfers to other models and languages. 

---
# Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning 

**Authors**: Jinyeop Song, Song Wang, Julian Shun, Yada Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.26383)  

**Abstract**: Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this https URL. 

---
# Fast-dLLM v2: Efficient Block-Diffusion LLM 

**Authors**: Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, Enze Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.26328)  

**Abstract**: Autoregressive (AR) large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks, yet their inherent sequential decoding limits inference efficiency. In this work, we propose Fast-dLLM v2, a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained AR models into dLLMs for parallel text generation, requiring only approximately 1B tokens of fine-tuning. This represents a 500x reduction in training data compared to full-attention diffusion LLMs such as Dream (580B tokens), while preserving the original model's performance. Our approach introduces a novel training recipe that combines a block diffusion mechanism with a complementary attention mask, enabling blockwise bidirectional context modeling without sacrificing AR training objectives. To further accelerate decoding, we design a hierarchical caching mechanism: a block-level cache that stores historical context representations across blocks, and a sub-block cache that enables efficient parallel generation within partially decoded blocks. Coupled with our parallel decoding pipeline, Fast-dLLM v2 achieves up to 2.5x speedup over standard AR decoding without compromising generation quality. Extensive experiments across diverse benchmarks demonstrate that Fast-dLLM v2 matches or surpasses AR baselines in accuracy, while delivering state-of-the-art efficiency among dLLMs - marking a significant step toward the practical deployment of fast and accurate LLMs. Code and model will be publicly released. 

---
# Latent Thinking Optimization: Your Latent Reasoning Language Model Secretly Encodes Reward Signals in its Latent Thoughts 

**Authors**: Hanwen Du, Yuxin Dong, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2509.26314)  

**Abstract**: Large Language Models (LLMs) excel at problem solving by generating chain of thoughts in natural language, but such verbal thinking is computationally costly and prone to overthinking. Recent work instead proposes a latent thinking architecture Huggin-3.5B, which represents intermediate reasoning steps as sequence of latent representations. However, latent thoughts lack interpretability and are difficult to supervise, raising concerns about the correctness and reliability of its latent thinking processes. In this paper, we provide a systematic study of how Huggin-3.5B thinks in the latent space and how external supervision signals can improve its latent thinking processes. We show that latent thoughts leading to correct versus incorrect answers exhibit highly distinguishable patterns, and that a latent classifier can reliably predict answer correctness directly from latent thoughts. Leveraging these insights, we propose Latent Thinking Optimization (LTO), a probabilistic algorithm that employs the latent classifier as a Latent Reward Model (LRM) to optimize the latent thinking processes. Extensive experiments across diverse reasoning tasks demonstrate that LRM is highly effective in detecting incorrect latent thinking patterns, and LTO can significantly improve the latent thinking processes. Furthermore, we show that LRM can generalize across diverse domains, and LTO can be seamlessly applied to general LLMs to improve their thinking processes. In contrast to verbal thinking, our method demonstrates that reward modeling and scaling test-time thinking with supervision can be performed directly in the latent space, highlighting its potential as a general, efficient, and domain-agnostic approach to improving the thinking processes of LLMs. 

---
# One-Token Rollout: Guiding Supervised Fine-Tuning of LLMs with Policy Gradient 

**Authors**: Rui Ming, Haoyuan Wu, Shoubo Hu, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.26313)  

**Abstract**: Supervised fine-tuning (SFT) is the predominant method for adapting large language models (LLMs), yet it often struggles with generalization compared to reinforcement learning (RL). In this work, we posit that this performance disparity stems not just from the loss function, but from a more fundamental difference: SFT learns from a fixed, pre-collected dataset, whereas RL utilizes on-policy data sampled from the current policy. Building on this hypothesis, we introduce one-token rollout (OTR), a novel fine-tuning algorithm that guides SFT with the policy gradient method. OTR reframes the autoregressive learning process by treating each token generation as a single-step reinforcement learning trajectory. At each step, it performs a Monte Carlo ``rollout'' by sampling multiple candidate tokens from the current policy's distribution. The ground-truth token from the supervised data is then used to provide a reward signal to these samples. Guided by policy gradient, our algorithm repurposes static, off-policy supervised data into a dynamic, on-policy signal at the token level, capturing the generalization benefits of on-policy learning while bypassing the costly overhead of full sentence generation. Through extensive experiments on a diverse suite of challenging benchmarks spanning mathematical reasoning, code generation, and general domain reasoning, we demonstrate that OTR consistently outperforms standard SFT. Our findings establish OTR as a powerful and practical alternative for fine-tuning LLMs and provide compelling evidence that the on-policy nature of data is a critical driver of generalization, offering a promising new direction for fine-tuning LLMs. 

---
# CliniBench: A Clinical Outcome Prediction Benchmark for Generative and Encoder-Based Language Models 

**Authors**: Paul Grundmann, Dennis Fast, Jan Frick, Thomas Steffek, Felix Gers, Wolfgang Nejdl, Alexander Löser  

**Link**: [PDF](https://arxiv.org/pdf/2509.26136)  

**Abstract**: With their growing capabilities, generative large language models (LLMs) are being increasingly investigated for complex medical tasks. However, their effectiveness in real-world clinical applications remains underexplored. To address this, we present CliniBench, the first benchmark that enables comparability of well-studied encoder-based classifiers and generative LLMs for discharge diagnosis prediction from admission notes in MIMIC-IV dataset. Our extensive study compares 12 generative LLMs and 3 encoder-based classifiers and demonstrates that encoder-based classifiers consistently outperform generative models in diagnosis prediction. We assess several retrieval augmentation strategies for in-context learning from similar patients and find that they provide notable performance improvements for generative LLMs. 

---
# Deconstructing Self-Bias in LLM-generated Translation Benchmarks 

**Authors**: Wenda Xu, Sweta Agrawal, Vilém Zouhar, Markus Freitag, Daniel Deutsch  

**Link**: [PDF](https://arxiv.org/pdf/2509.26600)  

**Abstract**: As large language models (LLMs) begin to saturate existing benchmarks, automated benchmark creation using LLMs (LLM as a benchmark) has emerged as a scalable alternative to slow and costly human curation. While these generated test sets have to potential to cheaply rank models, we demonstrate a critical flaw. LLM generated benchmarks systematically favor the model that created the benchmark, they exhibit self bias on low resource languages to English translation tasks. We show three key findings on automatic benchmarking of LLMs for translation: First, this bias originates from two sources: the generated test data (LLM as a testset) and the evaluation method (LLM as an evaluator), with their combination amplifying the effect. Second, self bias in LLM as a benchmark is heavily influenced by the model's generation capabilities in the source language. For instance, we observe more pronounced bias in into English translation, where the model's generation system is developed, than in out of English translation tasks. Third, we observe that low diversity in source text is one attribution to self bias. Our results suggest that improving the diversity of these generated source texts can mitigate some of the observed self bias. 

---
# End-to-End Aspect-Guided Review Summarization at Scale 

**Authors**: Ilya Boytsov, Vinny DeGenova, Mikhail Balyasin, Joseph Walt, Caitlin Eusden, Marie-Claire Rochat, Margaret Pierson  

**Link**: [PDF](https://arxiv.org/pdf/2509.26103)  

**Abstract**: We present a scalable large language model (LLM)-based system that combines aspect-based sentiment analysis (ABSA) with guided summarization to generate concise and interpretable product review summaries for the Wayfair platform. Our approach first extracts and consolidates aspect-sentiment pairs from individual reviews, selects the most frequent aspects for each product, and samples representative reviews accordingly. These are used to construct structured prompts that guide the LLM to produce summaries grounded in actual customer feedback. We demonstrate the real-world effectiveness of our system through a large-scale online A/B test. Furthermore, we describe our real-time deployment strategy and release a dataset of 11.8 million anonymized customer reviews covering 92,000 products, including extracted aspects and generated summaries, to support future research in aspect-guided review summarization. 

---
# Reinforced Strategy Optimization for Conversational Recommender Systems via Network-of-Experts 

**Authors**: Xiaoyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26093)  

**Abstract**: Conversational Recommender Systems (CRSs) provide personalized recommendations through multi-turn interactions. With the strong reasoning abilities of Large Language Models (LLMs), applying them to CRSs has become promising. Yet, existing methods often lack explicit optimization of interaction strategies, relying instead on unified prompts, which can yield suboptimal outcomes. We propose Reinforced Strategy Optimization (RSO), a hierarchical framework that decomposes response generation into macro-level strategy planning and micro-level adaptation within a network-of-experts. A Planner selects strategies (e.g., recommend, explain, encourage), while an Actor generates responses guided by auxiliary experts for preferences and factual grounding. This disentanglement enables more tractable learning. To address limited multi-turn data, we model strategy learning as reinforcement learning with an LLM-based reward for exploration. Experiments show RSO outperforms state-of-the-art baselines, validating the effectiveness of hierarchical strategy optimization. 

---
# IMProofBench: Benchmarking AI on Research-Level Mathematical Proof Generation 

**Authors**: Johannes Schmitt, Gergely Bérczi, Jasper Dekoninck, Jeremy Feusi, Tim Gehrunger, Raphael Appenzeller, Jim Bryan, Niklas Canova, Timo de Wolff, Filippo Gaia, Michel van Garrel, Baran Hashemi, David Holmes, Aitor Iribar Lopez, Victor Jaeck, Martina Jørgensen, Steven Kelk, Stefan Kuhlmann, Adam Kurpisz, Chiara Meroni, Ingmar Metzler, Martin Möller, Samuel Muñoz-Echániz, Robert Nowak, Georg Oberdieck, Daniel Platt, Dylan Possamaï, Gabriel Ribeiro, Raúl Sánchez Galán, Zheming Sun, Josef Teichmann, Richard P. Thomas, Charles Vial  

**Link**: [PDF](https://arxiv.org/pdf/2509.26076)  

**Abstract**: As the mathematical capabilities of large language models (LLMs) improve, it becomes increasingly important to evaluate their performance on research-level tasks at the frontier of mathematical knowledge. However, existing benchmarks are limited, as they focus solely on final-answer questions or high-school competition problems. To address this gap, we introduce IMProofBench, a private benchmark consisting of 39 peer-reviewed problems developed by expert mathematicians. Each problem requires a detailed proof and is paired with subproblems that have final answers, supporting both an evaluation of mathematical reasoning capabilities by human experts and a large-scale quantitative analysis through automated grading. Furthermore, unlike prior benchmarks, the evaluation setup simulates a realistic research environment: models operate in an agentic framework with tools like web search for literature review and mathematical software such as SageMath. Our results show that current LLMs can succeed at the more accessible research-level questions, but still encounter significant difficulties on more challenging problems. Quantitatively, Grok-4 achieves the highest accuracy of 52% on final-answer subproblems, while GPT-5 obtains the best performance for proof generation, achieving a fully correct solution for 22% of problems. IMProofBench will continue to evolve as a dynamic benchmark in collaboration with the mathematical community, ensuring its relevance for evaluating the next generation of LLMs. 

---
# The Silent Judge: Unacknowledged Shortcut Bias in LLM-as-a-Judge 

**Authors**: Arash Marioriyad, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2509.26072)  

**Abstract**: Large language models (LLMs) are increasingly deployed as automatic judges to evaluate system outputs in tasks such as summarization, dialogue, and creative writing. A faithful judge should base its verdicts solely on response quality and explicitly acknowledge the factors shaping its decision. We show that current LLM judges fail on both counts by relying on shortcuts introduced in the prompt. Our study uses two evaluation datasets: ELI5, a benchmark for long-form question answering, and LitBench, a recent benchmark for creative writing. Both datasets provide pairwise comparisons, where the evaluator must choose which of two responses is better. From each dataset we construct 100 pairwise judgment tasks and employ two widely used models, GPT-4o and Gemini-2.5-Flash, as evaluators in the role of LLM-as-a-judge. For each pair, we assign superficial cues to the responses, provenance cues indicating source identity (Human, Expert, LLM, or Unknown) and recency cues indicating temporal origin (Old, 1950 vs. New, 2025), while keeping the rest of the prompt fixed. Results reveal consistent verdict shifts: both models exhibit a strong recency bias, systematically favoring new responses over old, as well as a clear provenance hierarchy (Expert > Human > LLM > Unknown). These biases are especially pronounced in GPT-4o and in the more subjective and open-ended LitBench domain. Crucially, cue acknowledgment is rare: justifications almost never reference the injected cues, instead rationalizing decisions in terms of content qualities. These findings demonstrate that current LLM-as-a-judge systems are shortcut-prone and unfaithful, undermining their reliability as evaluators in both research and deployment. 

---
# Unspoken Hints: Accuracy Without Acknowledgement in LLM Reasoning 

**Authors**: Arash Marioriyad, Shaygan Adim, Nima Alighardashi, Mahdieh Soleymani Banghshah, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2509.26041)  

**Abstract**: Large language models (LLMs) increasingly rely on chain-of-thought (CoT) prompting to solve mathematical and logical reasoning tasks. Yet, a central question remains: to what extent are these generated rationales \emph{faithful} to the underlying computations, rather than post-hoc narratives shaped by hints that function as answer shortcuts embedded in the prompt? Following prior work on hinted vs.\ unhinted prompting, we present a systematic study of CoT faithfulness under controlled hint manipulations. Our experimental design spans four datasets (AIME, GSM-Hard, MATH-500, UniADILR), two state-of-the-art models (GPT-4o and Gemini-2-Flash), and a structured set of hint conditions varying in correctness (correct and incorrect), presentation style (sycophancy and data leak), and complexity (raw answers, two-operator expressions, four-operator expressions). We evaluate both task accuracy and whether hints are explicitly acknowledged in the reasoning. Our results reveal three key findings. First, correct hints substantially improve accuracy, especially on harder benchmarks and logical reasoning, while incorrect hints sharply reduce accuracy in tasks with lower baseline competence. Second, acknowledgement of hints is highly uneven: equation-based hints are frequently referenced, whereas raw hints are often adopted silently, indicating that more complex hints push models toward verbalizing their reliance in the reasoning process. Third, presentation style matters: sycophancy prompts encourage overt acknowledgement, while leak-style prompts increase accuracy but promote hidden reliance. This may reflect RLHF-related effects, as sycophancy exploits the human-pleasing side and data leak triggers the self-censoring side. Together, these results demonstrate that LLM reasoning is systematically shaped by shortcuts in ways that obscure faithfulness. 

---
# RE-Searcher: Robust Agentic Search with Goal-oriented Planning and Self-reflection 

**Authors**: Daocheng Fu, Jianbiao Mei, Licheng Wen, Xuemeng Yang, Cheng Yang, Rong Wu, Tao Hu, Siqi Li, Yufan Shen, Xinyu Cai, Pinlong Cai, Botian Shi, Yong Liu, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26048)  

**Abstract**: Large language models (LLMs) excel at knowledge-intensive question answering and reasoning, yet their real-world deployment remains constrained by knowledge cutoff, hallucination, and limited interaction modalities. Augmenting LLMs with external search tools helps alleviate these issues, but it also exposes agents to a complex search environment in which small, plausible variations in query formulation can steer reasoning into unproductive trajectories and amplify errors. We present a systematic analysis that quantifies how environmental complexity induces fragile search behaviors and, in turn, degrades overall performance. To address this challenge, we propose a simple yet effective approach to instantiate a search agent, RE-Searcher. During search, RE-Searcher explicitly articulates a concrete search goal and subsequently reflects on whether the retrieved evidence satisfies that goal. This combination of goal-oriented planning and self-reflection enables RE-Searcher to resist spurious cues in complex search environments and perform robust search. Extensive experiments show that our method improves search accuracy and achieves state-of-the-art results. Perturbation studies further demonstrate substantial resilience to noisy or misleading external signals, mitigating the fragility of the search process. We believe these findings offer practical guidance for integrating LLM-powered agents into more complex interactive environments and enabling more autonomous decision-making. 

---
# Mem-α: Learning Memory Construction via Reinforcement Learning 

**Authors**: Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25911)  

**Abstract**: Large language model (LLM) agents are constrained by limited context windows, necessitating external memory systems for long-term information understanding. Current memory-augmented agents typically depend on pre-defined instructions and tools for memory updates. However, language models may lack the ability to determine which information to store, how to structure it, and when to update it, especially as memory systems become more complex. This results in suboptimal memory construction and information loss. To this end, we propose Mem-alpha, a reinforcement learning framework that trains agents to effectively manage complex memory systems through interaction and feedback. We also construct a specialized training dataset spanning diverse multi-turn interaction patterns paired with comprehensive evaluation questions designed to teach effective memory management. During training, agents process sequential information chunks, learn to extract and store relevant content, then update the memory system. The reward signal derives from downstream question-answering accuracy over the full interaction history, directly optimizing for memory construction. To illustrate the effectiveness of our training framework, we design a memory architecture comprising core, episodic, and semantic components, equipped with multiple tools for memory operations. Empirical evaluation demonstrates that Mem-alpha achieves significant improvements over existing memory-augmented agent baselines. Despite being trained exclusively on instances with a maximum length of 30k tokens, our agents exhibit remarkable generalization to sequences exceeding 400k tokens, over 13x the training length, highlighting the robustness of Mem-alpha. 

---
# RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity 

**Authors**: Jisu Shin, Hoyun Song, Juhyun Oh, Changgeon Ko, Eunsu Kim, Chani Jung, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2509.25897)  

**Abstract**: Humans often encounter role conflicts -- social dilemmas where the expectations of multiple roles clash and cannot be simultaneously fulfilled. As large language models (LLMs) become increasingly influential in human decision-making, understanding how they behave in complex social situations is essential. While previous research has evaluated LLMs' social abilities in contexts with predefined correct answers, role conflicts represent inherently ambiguous social dilemmas that require contextual sensitivity: the ability to recognize and appropriately weigh situational cues that can fundamentally alter decision priorities. To address this gap, we introduce RoleConflictBench, a novel benchmark designed to evaluate LLMs' contextual sensitivity in complex social dilemmas. Our benchmark employs a three-stage pipeline to generate over 13K realistic role conflict scenarios across 65 roles, systematically varying their associated expectations (i.e., their responsibilities and obligations) and situational urgency levels. By analyzing model choices across 10 different LLMs, we find that while LLMs show some capacity to respond to these contextual cues, this sensitivity is insufficient. Instead, their decisions are predominantly governed by a powerful, inherent bias related to social roles rather than situational information. Our analysis quantifies these biases, revealing a dominant preference for roles within the Family and Occupation domains, as well as a clear prioritization of male roles and Abrahamic religions across most evaluatee models. 

---
# ReFACT: A Benchmark for Scientific Confabulation Detection with Positional Error Annotations 

**Authors**: Yindong Wang, Martin Preiß, Margarita Bugueño, Jan Vincent Hoffbauer, Abdullatif Ghajar, Tolga Buz, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25868)  

**Abstract**: Large Language Models (LLMs) frequently confabulate scientific facts,severely undermining their trustworthiness. Addressing this challenge requires benchmarks that go beyond binary factuality and enable fine-grained evaluation. We introduce \textbf{ReFACT} (\textit{Reddit False And Correct Texts}), a benchmark of 1,001 expert-annotated question--answer pairs spanning diverse scientific domains for the detection of scientific confabulation. Each instance includes both a scientifically correct answer and a non-factual counterpart annotated with \textbf{precise error spans and error-types}. ReFACT enables multi-stage evaluation: (1) confabulation detection, (2) fine-grained error localization, and (3) correction. We benchmark 9 state-of-the-art LLMs, revealing limited performance ($\sim$50\% accuracy). Even top models such as GPT-4o fail to distinguish factual from confabulated scientific answers, raising concerns about the reliability of \textit{LLM-as-judge} evaluation paradigms. Our findings highlight the need for fine-grained, human-validated benchmarks to detect and correct scientific confabulation in domain-specific contexts. Dataset is released on \href{this https URL}{GitHub}\footnote{We provide the dataset at: this https URL}. 

---
# DyFlow: Dynamic Workflow Framework for Agentic Reasoning 

**Authors**: Yanbo Wang, Zixiang Xu, Yue Huang, Xiangqi Wang, Zirui Song, Lang Gao, Chenxi Wang, Xiangru Tang, Yue Zhao, Arman Cohan, Xiangliang Zhang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.26062)  

**Abstract**: Agent systems based on large language models (LLMs) have shown great potential in complex reasoning tasks, but building efficient and generalizable workflows remains a major challenge. Most existing approaches rely on manually designed processes, which limits their adaptability across different tasks. While a few methods attempt automated workflow generation, they are often tied to specific datasets or query types and make limited use of intermediate feedback, reducing system robustness and reasoning depth. Moreover, their operations are typically predefined and inflexible. To address these limitations, we propose DyFlow, a dynamic workflow generation framework that adaptively constructs and adjusts reasoning procedures based on task requirements and real-time intermediate feedback, thereby enhancing cross-task generalization. DyFlow consists of two core components: a designer and an executor. The designer decomposes complex problems into a sequence of sub-goals defined by high-level objectives and dynamically plans the next steps based on intermediate outputs and feedback. These plans are then carried out by the executor, which executes each operation using dynamic operators with context-aware parameterization, enabling flexible and semantically grounded reasoning. We systematically evaluate DyFlow across diverse domains, including social reasoning, biomedical tasks, mathematical problem solving, and code generation. Results demonstrate that DyFlow significantly outperforms existing baselines, achieving substantial Pass@k improvements and exhibiting robust generalization across diverse domains. The code is publicly available at this https URL. 

---
# RoBiologyDataChoiceQA: A Romanian Dataset for improving Biology understanding of Large Language Models 

**Authors**: Dragos-Dumitru Ghinea, Adela-Nicoleta Corbeanu, Adrian-Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2509.25813)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated significant potential across various natural language processing (NLP) tasks. However, their performance in domain-specific applications and non-English languages remains less explored. This study introduces a novel Romanian-language dataset for multiple-choice biology questions, carefully curated to assess LLM comprehension and reasoning capabilities in scientific contexts. Containing approximately 14,000 questions, the dataset provides a comprehensive resource for evaluating and improving LLM performance in biology.
We benchmark several popular LLMs, analyzing their accuracy, reasoning patterns, and ability to understand domain-specific terminology and linguistic nuances. Additionally, we perform comprehensive experiments to evaluate the impact of prompt engineering, fine-tuning, and other optimization techniques on model performance. Our findings highlight both the strengths and limitations of current LLMs in handling specialized knowledge tasks in low-resource languages, offering valuable insights for future research and development. 

---
# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning 

**Authors**: Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Sean Chen, Mohammad Kachuee, Teja Gollapudi, Tony Liao, Nicolas Scheffer, Rakesh Wanga, Anuj Kumar, Yu Meng, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.25760)  

**Abstract**: While large language models (LLMs) have demonstrated strong performance on factoid question answering, they are still prone to hallucination and untruthful responses, particularly when tasks demand information outside their parametric knowledge. Indeed, truthfulness requires more than accuracy -- models must also recognize uncertainty and abstain when unsure to avoid hallucinations. This presents a fundamental challenge for existing methods: approaches that optimize for accuracy often amplify hallucinations, while those that encourage abstention can become overly conservative, sacrificing correct answers. Both extremes ultimately compromise truthfulness. In this work, we present TruthRL, a general reinforcement learning (RL) framework that directly optimizes the truthfulness of LLMs. Specifically, we implement TruthRL using GRPO with a simple yet effective ternary reward that distinguishes correct answers, hallucinations, and abstentions. It incentivizes models to reduce hallucinations not only by providing correct responses, but also by enabling abstention when uncertain, thereby improving truthfulness. Extensive experiments across four knowledge-intensive benchmarks show that, compared to vanilla RL, TruthRL significantly reduces hallucinations by 28.9% and improves truthfulness by 21.1%, with consistent gains across various backbone models (e.g., Qwen, Llama) under both retrieval and non-retrieval setups. In-depth ablation study demonstrates that vanilla accuracy-driven methods, such as supervised fine-tuning or RL with a binary reward, struggle to balance factual correctness and uncertainty. In contrast, our proposed truthfulness-driven TruthRL achieves strong performance in both accuracy and truthfulness, underscoring the importance of learning objective design for developing truthful LLMs. 

---
# Atomic Thinking of LLMs: Decoupling and Exploring Mathematical Reasoning Abilities 

**Authors**: Jiayi Kuang, Haojing Huang, Yinghui Li, Xinnian Liang, Zhikun Xu, Yangning Li, Xiaoyu Tan, Chao Qu, Meishan Zhang, Ying Shen, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25725)  

**Abstract**: Large Language Models (LLMs) have demonstrated outstanding performance in mathematical reasoning capabilities. However, we argue that current large-scale reasoning models primarily rely on scaling up training datasets with diverse mathematical problems and long thinking chains, which raises questions about whether LLMs genuinely acquire mathematical concepts and reasoning principles or merely remember the training data. In contrast, humans tend to break down complex problems into multiple fundamental atomic capabilities. Inspired by this, we propose a new paradigm for evaluating mathematical atomic capabilities. Our work categorizes atomic abilities into two dimensions: (1) field-specific abilities across four major mathematical fields, algebra, geometry, analysis, and topology, and (2) logical abilities at different levels, including conceptual understanding, forward multi-step reasoning with formal math language, and counterexample-driven backward reasoning. We propose corresponding training and evaluation datasets for each atomic capability unit, and conduct extensive experiments about how different atomic capabilities influence others, to explore the strategies to elicit the required specific atomic capability. Evaluation and experimental results on advanced models show many interesting discoveries and inspirations about the different performances of models on various atomic capabilities and the interactions between atomic capabilities. Our findings highlight the importance of decoupling mathematical intelligence into atomic components, providing new insights into model cognition and guiding the development of training strategies toward a more efficient, transferable, and cognitively grounded paradigm of "atomic thinking". 

---
# The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale 

**Authors**: Samar Haider, Amir Tohidi, Jenny S. Wang, Timothy Dörr, David M. Rothschild, Chris Callison-Burch, Duncan J. Watts  

**Link**: [PDF](https://arxiv.org/pdf/2509.25649)  

**Abstract**: Mainstream news organizations shape public perception not only directly through the articles they publish but also through the choices they make about which topics to cover (or ignore) and how to frame the issues they do decide to cover. However, measuring these subtle forms of media bias at scale remains a challenge. Here, we introduce a large, ongoing (from January 1, 2024 to present), near real-time dataset and computational framework developed to enable systematic study of selection and framing bias in news coverage. Our pipeline integrates large language models (LLMs) with scalable, near-real-time news scraping to extract structured annotations -- including political lean, tone, topics, article type, and major events -- across hundreds of articles per day. We quantify these dimensions of coverage at multiple levels -- the sentence level, the article level, and the publisher level -- expanding the ways in which researchers can analyze media bias in the modern news landscape. In addition to a curated dataset, we also release an interactive web platform for convenient exploration of these data. Together, these contributions establish a reusable methodology for studying media bias at scale, providing empirical resources for future research. Leveraging the breadth of the corpus over time and across publishers, we also present some examples (focused on the 150,000+ articles examined in 2024) that illustrate how this novel data set can reveal insightful patterns in news coverage and bias, supporting academic research and real-world efforts to improve media accountability. 

---
# BatonVoice: An Operationalist Framework for Enhancing Controllable Speech Synthesis with Linguistic Intelligence from LLMs 

**Authors**: Yue Wang, Ruotian Ma, Xingyu Chen, Zhengliang Shi, Wanshun Chen, Huang Liu, Jiadi Yao, Qu Yang, Qingxuan Jiang, Fanghua Ye, Juntao Li, Min Zhang, Zhaopeng Tu, Xiaolong Li, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2509.26514)  

**Abstract**: The rise of Large Language Models (LLMs) is reshaping multimodel models, with speech synthesis being a prominent application. However, existing approaches often underutilize the linguistic intelligence of these models, typically failing to leverage their powerful instruction-following capabilities. This limitation hinders the model's ability to follow text instructions for controllable Text-to-Speech~(TTS). To address this, we propose a new paradigm inspired by ``operationalism'' that decouples instruction understanding from speech generation. We introduce BatonVoice, a framework where an LLM acts as a ``conductor'', understanding user instructions and generating a textual ``plan'' -- explicit vocal features (e.g., pitch, energy). A separate TTS model, the ``orchestra'', then generates the speech from these features. To realize this component, we develop BatonTTS, a TTS model trained specifically for this task. Our experiments demonstrate that BatonVoice achieves strong performance in controllable and emotional speech synthesis, outperforming strong open- and closed-source baselines. Notably, our approach enables remarkable zero-shot cross-lingual generalization, accurately applying feature control abilities to languages unseen during post-training. This demonstrates that objectifying speech into textual vocal features can more effectively unlock the linguistic intelligence of LLMs. 

---
# RFG: Test-Time Scaling for Diffusion Large Language Model Reasoning with Reward-Free Guidance 

**Authors**: Tianlang Chen, Minkai Xu, Jure Leskovec, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.25604)  

**Abstract**: Diffusion large language models (dLLMs) have shown great potential in large-scale language modeling, and there is an increasing interest in further improving the capacity to solve complex problems by guiding the reasoning process step by step. Common practice for autoregressive language models typically learns a process reward model with dense annotation for each intermediate step. However, this is challenging for dLLMs where the generation is in an any-order fashion and intermediate states are partially masked sentences. To this end, in this paper, we propose reward-free guidance (RFG), a principled method for guiding the reasoning trajectory of dLLMs without explicit process reward. The key idea of RFG is to parameterize the process reward by log-likelihood ratios of the enhanced and reference dLLMs, where the enhanced model can be easily obtained by any off-the-shelf dLLM that has been post-trained with reinforcement learning (RL) or supervised fine-tuning (SFT). We provide theoretical justification that RFG induces the reward-guided sampling distribution with no additional reward. We conduct comprehensive experiments on four challenging mathematical reasoning and code generation benchmarks using a diverse suite of dLLMs enhanced with various post-training methods. RFG consistently yields significant improvements across all tasks and model types, achieving accuracy gains of up to 9.2%. These findings establish RFG as a general training-free framework that scales test-time reasoning without reliance on external reward models. 

---
# Aligning Multilingual Reasoning with Verifiable Semantics from a High-Resource Expert Model 

**Authors**: Fahim Faisal, Kaiqiang Song, Song Wang, Simin Ma, Shujian Liu, Haoyun Deng, Sathish Reddy Indurthi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25543)  

**Abstract**: While reinforcement learning has advanced the reasoning abilities of Large Language Models (LLMs), these gains are largely confined to English, creating a significant performance disparity across languages. To address this, we introduce Pivot-Based Reinforcement Learning with Semantically Verifiable Rewards (PB-RLSVR), a novel framework that enhances multilingual reasoning by circumventing the need for human-annotated data in target languages. Our approach employs a high-performing English LLM as a "pivot" model to generate reference responses for reasoning tasks. A multilingual model is then rewarded based on the semantic equivalence of its responses to the English reference, effectively transferring the pivot model's reasoning capabilities across languages. We investigate several cross-lingual semantic reward functions, including those based on embeddings and machine translation. Extensive experiments on a suite of multilingual reasoning benchmarks show that our method significantly narrows the performance gap between English and other languages, substantially outperforming traditional PPO baselines. Specifically, our PB-RLSVR framework improves the average multilingual performance of Llama-3.1-8B-Instruct and Qwen3-32B by 16.41% and 10.17%, respectively, demonstrating a powerful and data-efficient approach to building truly multilingual reasoning agents. 

---
# QUARTZ : QA-based Unsupervised Abstractive Refinement for Task-oriented Dialogue Summarization 

**Authors**: Mohamed Imed Eddine Ghebriout, Gaël Guibon, Ivan Lerner, Emmanuel Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2509.26302)  

**Abstract**: Dialogue summarization aims to distill the core meaning of a conversation into a concise text. This is crucial for reducing the complexity and noise inherent in dialogue-heavy applications. While recent approaches typically train language models to mimic human-written summaries, such supervision is costly and often results in outputs that lack task-specific focus limiting their effectiveness in downstream applications, such as medical tasks. In this paper, we propose \app, a framework for task-oriented utility-based dialogue summarization. \app starts by generating multiple summaries and task-oriented question-answer pairs from a dialogue in a zero-shot manner using a pool of large language models (LLMs). The quality of the generated summaries is evaluated by having LLMs answer task-related questions before \textit{(i)} selecting the best candidate answers and \textit{(ii)} identifying the most informative summary based on these answers. Finally, we fine-tune the best LLM on the selected summaries. When validated on multiple datasets, \app demonstrates its effectiveness by achieving competitive results in various zero-shot settings, rivaling fully-supervised State-of-the-Art (SotA) methods. 

---
# Calibrating Verbalized Confidence with Self-Generated Distractors 

**Authors**: Victor Wang, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2509.25532)  

**Abstract**: Calibrated confidence estimates are necessary for large language model (LLM) outputs to be trusted by human users. While LLMs can express their confidence in human-interpretable ways, verbalized LLM-generated confidence scores have empirically been found to be miscalibrated, reporting high confidence on instances with low accuracy and thereby harming trust and safety. We hypothesize that this overconfidence often stems from a given LLM's heightened suggestibility when faced with claims that it encodes little information about; we empirically validate this hypothesis, finding more suggestibility on lower-accuracy claims. Building on this finding, we introduce Distractor-Normalized Coherence (DINCO), which estimates and accounts for an LLM's suggestibility bias by having the model verbalize its confidence independently across several self-generated distractors (i.e. alternative claims), and normalizes by the total verbalized confidence. To further improve calibration, we leverage generator-validator disagreement, augmenting normalized validator confidence with a consistency-based estimate of generator confidence. Here, we frame the popular approach of self-consistency as leveraging coherence across sampled generations, and normalized verbalized confidence as leveraging coherence across validations on incompatible claims, allowing us to integrate these complementary dimensions of coherence into DINCO. Moreover, our analysis shows that DINCO provides less saturated -- and therefore more usable -- confidence estimates, and that further sampling alone cannot close the gap between DINCO and baselines, with DINCO at 10 inference calls outperforming self-consistency at 100. 

---
# Not Wrong, But Untrue: LLM Overconfidence in Document-Based Queries 

**Authors**: Nick Hagar, Wilma Agustianto, Nicholas Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.25498)  

**Abstract**: Large language models (LLMs) are increasingly used in newsroom workflows, but their tendency to hallucinate poses risks to core journalistic practices of sourcing, attribution, and accuracy. We evaluate three widely used tools - ChatGPT, Gemini, and NotebookLM - on a reporting-style task grounded in a 300-document corpus related to TikTok litigation and policy in the U.S. We vary prompt specificity and context size and annotate sentence-level outputs using a taxonomy to measure hallucination type and severity. Across our sample, 30% of model outputs contained at least one hallucination, with rates approximately three times higher for Gemini and ChatGPT (40%) than for NotebookLM (13%). Qualitatively, most errors did not involve invented entities or numbers; instead, we observed interpretive overconfidence - models added unsupported characterizations of sources and transformed attributed opinions into general statements. These patterns reveal a fundamental epistemological mismatch: While journalism requires explicit sourcing for every claim, LLMs generate authoritative-sounding text regardless of evidentiary support. We propose journalism-specific extensions to existing hallucination taxonomies and argue that effective newsroom tools need architectures that enforce accurate attribution rather than optimize for fluency. 

---
# Self-Rewarding Rubric-Based Reinforcement Learning for Open-Ended Reasoning 

**Authors**: Zhiling Ye, Yun Yue, Haowen Wang, Xudong Han, Jiadi Jiang, Cheng Wei, Lei Fan, Jiaxin Liang, Shuowen Zhang, Ji Li, Chunxiao Guo, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25534)  

**Abstract**: Open-ended evaluation is essential for deploying large language models in real-world settings. In studying HealthBench, we observe that using the model itself as a grader and generating rubric-based reward signals substantially improves reasoning performance. Remarkably, the trained model also becomes a stronger grader. Motivated by this, we introduce Self-Rewarding Rubric-Based Reinforcement Learning for Open-Ended Reasoning, a lightweight framework that enables faster and more resource-efficient training while surpassing baselines. Remarkably, on Qwen3-32B, training with just the 4000-sample HealthBench Easy subset is sufficient to obtain a model that exceeds GPT-5 on HealthBench Hard. Incorporating a small amount of teacher-graded data further enhances performance for less capable models. 

---
# From Faithfulness to Correctness: Generative Reward Models that Think Critically 

**Authors**: Qiyao Ma, Yunsheng Shi, Hongtao Tian, Chao Wang, Weiming Chang, Ting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25409)  

**Abstract**: Through reinforcement learning with verifiable rewards (RLVR), large language models have achieved substantial progress in domains with easily verifiable outcomes, such as mathematics and coding. However, when applied to more complex tasks like open-domain question answering, RLVR faces significant challenges due to the difficulty of verifying correctness. The nuanced and ambiguous nature of real-world knowledge makes it difficult to reliably evaluate correctness in these settings, necessitating further abilities that extend beyond mere logical consistency to encompass an understanding and assessment of both external and internal knowledge. Recent work has primarily focused on improving faithfulness, defined as semantic alignment with supporting documents, which can cause models to rely excessively on external sources and diminish their capacity for critical assessment. To address this, we propose the Thinking-supervised Reward Model (TRM), which incorporates sentence-level thinking supervision to endow reward models with critical thinking abilities. Given a query, answer, and supporting documents, TRM first assesses the faithfulness of each answer sentence to the supporting documents, and then applies a reasoning step to evaluate sentence-level correctness. By structuring reward modeling as a sequence of faithfulness, reasoning, and correctness evaluations, TRM encourages models to critically assess and leverage both external and internal knowledge. Experiments on reward signals demonstrate that TRM substantially improves the identification of incorrect sentences, and incorporating TRM into policy optimization leads to significant gains in both answer correctness and usefulness. 

---
# Generative Value Conflicts Reveal LLM Priorities 

**Authors**: Andy Liu, Kshitish Ghate, Mona Diab, Daniel Fried, Atoosa Kasirzadeh, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2509.25369)  

**Abstract**: Past work seeks to align large language model (LLM)-based assistants with a target set of values, but such assistants are frequently forced to make tradeoffs between values when deployed. In response to the scarcity of value conflict in existing alignment datasets, we introduce ConflictScope, an automatic pipeline to evaluate how LLMs prioritize different values. Given a user-defined value set, ConflictScope automatically generates scenarios in which a language model faces a conflict between two values sampled from the set. It then prompts target models with an LLM-written "user prompt" and evaluates their free-text responses to elicit a ranking over values in the value set. Comparing results between multiple-choice and open-ended evaluations, we find that models shift away from supporting protective values, such as harmlessness, and toward supporting personal values, such as user autonomy, in more open-ended value conflict settings. However, including detailed value orderings in models' system prompts improves alignment with a target ranking by 14%, showing that system prompting can achieve moderate success at aligning LLM behavior under value conflict. Our work demonstrates the importance of evaluating value prioritization in models and provides a foundation for future work in this area. 

---
# Attention as a Compass: Efficient Exploration for Process-Supervised RL in Reasoning Models 

**Authors**: Runze Liu, Jiakang Wang, Yuling Shi, Zhihui Xie, Chenxin An, Kaiyan Zhang, Jian Zhao, Xiaodong Gu, Lei Lin, Wenping Hu, Xiu Li, Fuzheng Zhang, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2509.26628)  

**Abstract**: Reinforcement Learning (RL) has shown remarkable success in enhancing the reasoning capabilities of Large Language Models (LLMs). Process-Supervised RL (PSRL) has emerged as a more effective paradigm compared to outcome-based RL. However, existing PSRL approaches suffer from limited exploration efficiency, both in terms of branching positions and sampling. In this paper, we introduce a novel PSRL framework (AttnRL), which enables efficient exploration for reasoning models. Motivated by preliminary observations that steps exhibiting high attention scores correlate with reasoning behaviors, we propose to branch from positions with high values. Furthermore, we develop an adaptive sampling strategy that accounts for problem difficulty and historical batch size, ensuring that the whole training batch maintains non-zero advantage values. To further improve sampling efficiency, we design a one-step off-policy training pipeline for PSRL. Extensive experiments on multiple challenging mathematical reasoning benchmarks demonstrate that our method consistently outperforms prior approaches in terms of performance and sampling and training efficiency. 

---
# SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA 

**Authors**: Haozhou Xu, Dongxia Wu, Matteo Chinazzi, Ruijia Niu, Rose Yu, Yi-An Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.25459)  

**Abstract**: Large language models (LLMs) show promise in solving scientific problems. They can help generate long-form answers for scientific questions, which are crucial for comprehensive understanding of complex phenomena that require detailed explanations spanning multiple interconnected concepts and evidence. However, LLMs often suffer from hallucination, especially in the challenging task of long-form scientific question answering. Retrieval-Augmented Generation (RAG) approaches can ground LLMs by incorporating external knowledge sources to improve trustworthiness. In this context, scientific simulators, which play a vital role in validating hypotheses, offer a particularly promising retrieval source to mitigate hallucination and enhance answer factuality. However, existing RAG approaches cannot be directly applied for scientific simulation-based retrieval due to two fundamental challenges: how to retrieve from scientific simulators, and how to efficiently verify and update long-form answers. To overcome these challenges, we propose the simulator-based RAG framework (SimulRAG) and provide a long-form scientific QA benchmark covering climate science and epidemiology with ground truth verified by both simulations and human annotators. In this framework, we propose a generalized simulator retrieval interface to transform between textual and numerical modalities. We further design a claim-level generation method that utilizes uncertainty estimation scores and simulator boundary assessment (UE+SBA) to efficiently verify and update claims. Extensive experiments demonstrate SimulRAG outperforms traditional RAG baselines by 30.4% in informativeness and 16.3% in factuality. UE+SBA further improves efficiency and quality for claim-level generation. 

---
# From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation 

**Authors**: Viacheslav Yusupov, Danil Maksimov, Ameliia Alaeva, Anna Vasileva, Anna Antipina, Tatyana Zaitseva, Alina Ermilova, Evgeny Burnaev, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2509.25359)  

**Abstract**: This paper bridges internal and external analysis approaches to large language models (LLMs) by demonstrating that geometric properties of internal model representations serve as reliable proxies for evaluating generated text quality. We validate a set of metrics including Maximum Explainable Variance, Effective Rank, Intrinsic Dimensionality, MAUVE score, and Schatten Norms measured across different layers of LLMs, demonstrating that Intrinsic Dimensionality and Effective Rank can serve as universal assessments of text naturalness and quality. Our key finding reveals that different models consistently rank text from various sources in the same order based on these geometric properties, indicating that these metrics reflect inherent text characteristics rather than model-specific artifacts. This allows a reference-free text quality evaluation that does not require human-annotated datasets, offering practical advantages for automated evaluation pipelines. 

---
# Finetune Once: Decoupling General & Domain Learning with Dynamic Boosted Annealing 

**Authors**: Yang Tang, Ruijie Liu, Yifan Wang, Shiyu Li, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.26242)  

**Abstract**: Large language models (LLMs) fine-tuning shows excellent implications. However, vanilla fine-tuning methods often require intricate data mixture and repeated experiments for optimal generalization. To address these challenges and streamline the training process, we propose an efficient and universal solution, Dynamic Boosted Annealing (DBA). We obtain a global gradient through zero-learning-rate training on general data, which is subsequently employed for gradient boosting and dynamic training step correction during domain training. In conjunction with annealing learning, we end up establishing a fine-tuning pipeline that relies solely on domain data without collapse. By evaluating both general and domain-specific performance across multiple tasks on several popular base models, DBA achieves an average improvement of 5.8% in joint performance over vanilla fine-tuning. Furthermore, since general data is no longer involved in annealing, repeated experiments led by data mixture are also eliminated. According to our tests, the DBA method can reduce GPU hours by 91.0% compared to the vanilla method. 

---
# Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark 

**Authors**: Minhui Zhu, Minyang Tian, Xiaocheng Yang, Tianci Zhou, Penghao Zhu, Eli Chertkov, Shengyan Liu, Yufeng Du, Lifan Yuan, Ziming Ji, Indranil Das, Junyi Cao, Yufeng Du, Jinchen He, Yifan Su, Jiabin Yu, Yikun Jiang, Yujie Zhang, Chang Liu, Ze-Min Huang, Weizhen Jia, Xinan Chen, Peixue Wu, Yunkai Wang, Juntai Zhou, Yong Zhao, Farshid Jafarpour, Jessie Shelton, Aaron Young, John Bartolotta, Wenchao Xu, Yue Sun, Anjun Chu, Victor Colussi, Chris Akers, Nathan Brooks, Wenbo Fu, Christopher Wilson, Jinchao Zhao, Marvin Qi, Anqi Mu, Yubo Yang, Allen Zang, Yang Lyu, Peizhi Mai, Xuefei Guo, Luyu Gao, Ze Yang, Chi Xue, Dmytro Bandak, Yaïr Hein, Yonatan Kahn, Kevin Zhou, John Drew Wilson Jarrod T. Reilly, Di Luo, Daniel Inafuku, Hao Tong, Liang Yang, Ruixing Zhang, Xueying Wang, Ofir Press, Nicolas Chia, Eliu Huerta, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.26574)  

**Abstract**: While large language models (LLMs) with reasoning capabilities are progressing rapidly on high-school math competitions and coding, can they reason effectively through complex, open-ended challenges found in frontier physics research? And crucially, what kinds of reasoning tasks do physicists want LLMs to assist with? To address these questions, we present the CritPt (Complex Research using Integrated Thinking - Physics Test, pronounced "critical point"), the first benchmark designed to test LLMs on unpublished, research-level reasoning tasks that broadly covers modern physics research areas, including condensed matter, quantum physics, atomic, molecular & optical physics, astrophysics, high energy physics, mathematical physics, statistical physics, nuclear physics, nonlinear dynamics, fluid dynamics and biophysics. CritPt consists of 71 composite research challenges designed to simulate full-scale research projects at the entry level, which are also decomposed to 190 simpler checkpoint tasks for more fine-grained insights. All problems are newly created by 50+ active physics researchers based on their own research. Every problem is hand-curated to admit a guess-resistant and machine-verifiable answer and is evaluated by an automated grading pipeline heavily customized for advanced physics-specific output formats. We find that while current state-of-the-art LLMs show early promise on isolated checkpoints, they remain far from being able to reliably solve full research-scale challenges: the best average accuracy among base models is only 4.0% , achieved by GPT-5 (high), moderately rising to around 10% when equipped with coding tools. Through the realistic yet standardized evaluation offered by CritPt, we highlight a large disconnect between current model capabilities and realistic physics research demands, offering a foundation to guide the development of scientifically grounded AI tools. 

---
# SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From 

**Authors**: Yao Tong, Haonan Wang, Siquan Li, Kenji Kawaguchi, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.26404)  

**Abstract**: Fingerprinting Large Language Models (LLMs) is essential for provenance verification and model attribution. Existing methods typically extract post-hoc signatures based on training dynamics, data exposure, or hyperparameters -- properties that only emerge after training begins. In contrast, we propose a stronger and more intrinsic notion of LLM fingerprinting: SeedPrints, a method that leverages random initialization biases as persistent, seed-dependent identifiers present even before training. We show that untrained models exhibit reproducible token selection biases conditioned solely on their parameters at initialization. These biases are stable and measurable throughout training, enabling our statistical detection method to recover a model's lineage with high confidence. Unlike prior techniques, unreliable before convergence and vulnerable to distribution shifts, SeedPrints remains effective across all training stages and robust under domain shifts or parameter modifications. Experiments on LLaMA-style and Qwen-style models show that SeedPrints achieves seed-level distinguishability and can provide birth-to-lifecycle identity verification akin to a biometric fingerprint. Evaluations on large-scale pretrained models and fingerprinting benchmarks further confirm its effectiveness under practical deployment scenarios. These results suggest that initialization itself imprints a unique and persistent identity on neural language models, forming a true ''Galtonian'' fingerprint. 

---
# Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents 

**Authors**: Shuai Shao, Qihan Ren, Chen Qian, Boyi Wei, Dadi Guo, Jingyi Yang, Xinhao Song, Linfeng Zhang, Weinan Zhang, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26354)  

**Abstract**: Advances in Large Language Models (LLMs) have enabled a new class of self-evolving agents that autonomously improve through interaction with the environment, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study the case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as Misevolution. To provide a systematic investigation, we evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (e.g., Gemini-2.5-Pro). Different emergent risks are observed in the self-evolutionary process, such as the degradation of safety alignment after memory accumulation, or the unintended introduction of vulnerabilities in tool creation and reuse. To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents. Our code and data are available at this https URL . Warning: this paper includes examples that may be offensive or harmful in nature. 

---
# DeepJSONEval: Benchmarking Complex Nested JSON Data Mining for Large Language Models 

**Authors**: Zhicheng Zhou, Jing Li, Suming Qiu, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.25922)  

**Abstract**: The internet is saturated with low-density, high-redundancy information, such as social media comments, repetitive news, and lengthy discussions, making it difficult to extract valuable insights efficiently. Multi-layer nested JSON structures provide an effective solution by compressing such information into semantically rich, hierarchical representations, which organize data into key-value pairs, arrays, and nested objects, preserving contextual relationships and enabling efficient storage, retrieval, and semantic querying. For instance, in news aggregation, a JSON object can nest an article's metadata (title, author, date), content (text, multimedia), and multimedia information (multimedia type, caption) hierarchically. Large Language Models (LLMs) play a transformative role in web data mining by parsing unstructured text and outputting structured results directly into complex JSON schemas. However, current benchmarks for evaluating LLMs' JSON output capabilities overemphasize pure JSON generation rather than assessing data comprehension and extraction abilities, a limitation that lacks relevance to practical web data mining tasks. To address this, we introduce DeepJSONEval, a novel benchmark featuring 2100 multi-domain instances with deep nested structures, categorized by difficulty. Experiments show significant performance gaps among LLMs in handling such complexity. Our benchmark and datasets are open-sourced to advance research in structured JSON generation.(this https URL). 

---
# RE$^2$: Improving Chinese Grammatical Error Correction via Retrieving Appropriate Examples with Explanation 

**Authors**: Baoxin Wang, Yumeng Luo, Yixuan Wang, Dayong Wu, Wanxiang Che, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26038)  

**Abstract**: The primary objective of Chinese grammatical error correction (CGEC) is to detect and correct errors in Chinese sentences. Recent research shows that large language models (LLMs) have been applied to CGEC with significant results. For LLMs, selecting appropriate reference examples can help improve their performance. However, existing methods predominantly rely on text similarity for example retrieval, a strategy that frequently mismatches actual error patterns and retrieves lexically similar yet grammatically irrelevant sentences. To address this problem, we propose a method named RE$^2$, which retrieves appropriate examples with explanations of grammatical errors. Instead of using text similarity of the input sentence, we use explanations of grammatical errors to select reference examples, which are used by LLMs to improve the performance of CGEC. We conduct experiments on two CGEC datasets and create a high-quality grammatical error explanation (GEE) dataset, which is not only used in our research but also serves as a valuable resource for future studies in both CGEC and GEE. The experimental results on the two datasets indicate that our proposed method effectively improves the performance of CGEC. 

---
# Assessing Algorithmic Bias in Language-Based Depression Detection: A Comparison of DNN and LLM Approaches 

**Authors**: Obed Junias, Prajakta Kini, Theodora Chaspari  

**Link**: [PDF](https://arxiv.org/pdf/2509.25795)  

**Abstract**: This paper investigates algorithmic bias in language-based models for automated depression detection, focusing on socio-demographic disparities related to gender and race/ethnicity. Models trained using deep neural networks (DNN) based embeddings are compared to few-shot learning approaches with large language models (LLMs), evaluating both performance and fairness on clinical interview transcripts from the Distress Analysis Interview Corpus/Wizard-of-Oz (DAIC-WOZ). To mitigate bias, fairness-aware loss functions are applied to DNN-based models, while in-context learning with varied prompt framing and shot counts is explored for LLMs. Results indicate that LLMs outperform DNN-based models in depression classification, particularly for underrepresented groups such as Hispanic participants. LLMs also exhibit reduced gender bias compared to DNN-based embeddings, though racial disparities persist. Among fairness-aware techniques for mitigating bias in DNN-based embeddings, the worst-group loss, which is designed to minimize loss for the worst-performing demographic group, achieves a better balance between performance and fairness. In contrast, the fairness-regularized loss minimizes loss across all groups but performs less effectively. In LLMs, guided prompting with ethical framing helps mitigate gender bias in the 1-shot setting. However, increasing the number of shots does not lead to further reductions in disparities. For race/ethnicity, neither prompting strategy nor increasing $N$ in $N$-shot learning effectively reduces disparities. 

---
# A Multimodal LLM Approach for Visual Question Answering on Multiparametric 3D Brain MRI 

**Authors**: Arvind Murari Vepa, Yannan Yu, Jingru Gan, Anthony Cuturrufo, Weikai Li, Wei Wang, Fabien Scalzo, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.25889)  

**Abstract**: We introduce mpLLM, a prompt-conditioned hierarchical mixture-of-experts (MoE) architecture for visual question answering over multi-parametric 3D brain MRI (mpMRI). mpLLM routes across modality-level and token-level projection experts to fuse multiple interrelated 3D modalities, enabling efficient training without image--report pretraining. To address limited image-text paired supervision, mpLLM integrates a synthetic visual question answering (VQA) protocol that generates medically relevant VQA from segmentation annotations, and we collaborate with medical experts for clinical validation. mpLLM outperforms strong medical VLM baselines by 5.3% on average across multiple mpMRI datasets. Our study features three main contributions: (1) the first clinically validated VQA dataset for 3D brain mpMRI, (2) a novel multimodal LLM that handles multiple interrelated 3D modalities, and (3) strong empirical results that demonstrate the medical utility of our methodology. Ablations highlight the importance of modality-level and token-level experts and prompt-conditioned routing. We have included our source code in the supplementary materials and will release our dataset upon publication. 

---
# Lita: Light Agent Uncovers the Agentic Coding Capabilities of LLMs 

**Authors**: Hankun Dai, Maoquan Wang, Mengnan Qi, Yikai Zhang, Zijian Jin, Yongqiang Yao, Yufan Huang, Shengyu Fu, Elsie Nallipogu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25873)  

**Abstract**: Large language models (LLMs) are increasingly being applied to programming tasks, ranging from single-turn code completion to autonomous agents. Current code agent designs frequently depend on complex, hand-crafted workflows and tool sets. However, this reliance on elaborate scaffolding presents several challenges: agent performance becomes overly dependent on prompt tuning and custom design choices, heavy human intervention obscures a model's true underlying capabilities, and intricate pipelines are costly to build and maintain. Furthermore, optimizing complex task prompts increases the risk of data leakage. Currently, when introducing new models, LLM providers like OpenAI and Anthropic often publish benchmark scores to demonstrate their models' coding proficiency, but keep their proprietary evaluation frameworks confidential. To address these limitations, we introduce Lita (Lite Agent), which operationalizes liteness, a principle of minimizing manual design while retaining the essential elements of a fully autonomous agent. Lita enables a more faithful and unified evaluation without elaborate scaffolding. Experiments on the Aider Polyglot and SWE-Bench with frontier models demonstrate that Lita achieves competitive or superior performance compared to workflow-based and agentic baselines. Crucially, Lita also consumes fewer tokens and requires significantly less design effort. Our results suggest that Lita is sufficient to reveal the underlying coding competence of modern LLMs. Finally, we propose the Agent Complexity Law: the performance gap between agents of varying complexity, from simple to sophisticated designs, will shrink as the core model improves, ultimately converging to a negligible difference. 

---
# VELA: An LLM-Hybrid-as-a-Judge Approach for Evaluating Long Image Captions 

**Authors**: Kazuki Matsuda, Yuiga Wada, Shinnosuke Hirano, Seitaro Otsuki, Komei Sugiura  

**Link**: [PDF](https://arxiv.org/pdf/2509.25818)  

**Abstract**: In this study, we focus on the automatic evaluation of long and detailed image captions generated by multimodal Large Language Models (MLLMs). Most existing automatic evaluation metrics for image captioning are primarily designed for short captions and are not suitable for evaluating long captions. Moreover, recent LLM-as-a-Judge approaches suffer from slow inference due to their reliance on autoregressive inference and early fusion of visual information. To address these limitations, we propose VELA, an automatic evaluation metric for long captions developed within a novel LLM-Hybrid-as-a-Judge framework. Furthermore, we propose LongCap-Arena, a benchmark specifically designed for evaluating metrics for long captions. This benchmark comprises 7,805 images, the corresponding human-provided long reference captions and long candidate captions, and 32,246 human judgments from three distinct perspectives: Descriptiveness, Relevance, and Fluency. We demonstrated that VELA outperformed existing metrics and achieved superhuman performance on LongCap-Arena. 

---
# Boosting Process-Correct CoT Reasoning by Modeling Solvability of Multiple-Choice QA 

**Authors**: Raphael Schumann, Stefan Riezler  

**Link**: [PDF](https://arxiv.org/pdf/2509.25941)  

**Abstract**: Reasoning quality in large language models depends not only on producing correct answers but also on generating valid intermediate steps. We study this through multiple-choice question answering (MCQA), which provides a controlled setting with fixed answer options. Our analysis shows that when questions are effectively unsolvable for a model, spurious chains of thought (CoTs) are more likely to appear, leading to false positives. By estimating the solvability of each question, we uncover an intermediate regime where learning is most effective. Building on this insight, we adapt outcome-supervised reward models and reinforcement learning with group-relative advantage to incorporate solvability into their objectives. Across experiments on math and multimodal datasets, these modifications consistently yield higher rates of process-correct reasoning and, in reinforcement learning, improved answer accuracy as well. Our results highlight solvability as a key factor for reducing hallucinations and increasing reliability in CoT reasoning. 

---
# Nudging the Boundaries of LLM Reasoning 

**Authors**: Justin Chih-Yao Chen, Becky Xiangyu Peng, Prafulla Kumar Choubey, Kung-Hsiang Huang, Jiaxin Zhang, Mohit Bansal, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25666)  

**Abstract**: Current online reinforcement learning (RL) algorithms like GRPO share a key limitation in LLM reasoning: they cannot learn from problems that are "unsolvable" to the model. In other words, they can only improve performance on problems where the model is capable of exploring the correct answer. Consequently, the model's "upper limit" remains unchanged after RL training, even though the likelihood of solving easier, solvable problems may increase. These hard samples cannot contribute to training, as no rollouts yield rewards and thus no gradients are produced. To unlock learning from these hard samples, we propose NuRL, a "nudging" method that aims to push the upper bound of LLM reasoning using self-generated hints, i.e., abstract cues that help reduce the problem difficulty for the model. Given a question and its gold answer, the model generates a CoT and then produces a hint containing the core knowledge needed to solve the problem. During training, we generate G rollouts from the base policy and use the pass rate to decide whether the hint should be injected. For hard samples with a 0% pass rate, we inject the hint and regenerate a new batch of trajectories. This yields two benefits: (1) the hint boosts pass rates (from 0% to non-zero), thereby introducing training signals for previously unsolvable samples, and (2) the hints are self-generated, avoiding distributional shift and do not rely on external models. NuRL achieves consistent improvements across 6 benchmarks and 3 models, while remaining complementary to test-time scaling. Notably, NuRL can raise the model's upper limit, whereas GRPO leaves pass@1024 unchanged from the base model. Furthermore, we present a systematic study of what makes an effective hint and when hints are most useful. Interestingly, the best hints are abstract and high-level, and are most beneficial when applied necessarily and after GRPO has converged. 

---
# Building the EHR Foundation Model via Next Event Prediction 

**Authors**: Zekai Chen, Arda Pekis, Kevin Brown  

**Link**: [PDF](https://arxiv.org/pdf/2509.25591)  

**Abstract**: Electronic Health Records (EHRs) contain rich temporal dynamics that conventional encoding approaches fail to adequately capture. While Large Language Models (LLMs) show promise for EHR modeling, they struggle to reason about sequential clinical events and temporal dependencies. We propose Next Event Prediction (NEP), a framework that enhances LLMs' temporal reasoning through autoregressive fine-tuning on clinical event sequences. By reformulating EHRs as timestamped event chains and predicting future medical events, NEP explicitly models disease progression patterns and causal relationships. Extensive evaluations across oncology survival prediction and clinical diagnosis tasks demonstrate NEP's superiority, outperforming specialized EHR models by 4.6% AUROC and general-purpose LLMs by 7.2% C-index in temporal reasoning tasks. Our analyses reveal dual benefits: state-of-the-art prediction accuracy combined with clinically interpretable attention patterns that align with known disease pathways. 

---
# Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs 

**Authors**: Shane Bergsma, Nolan Dey, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2509.25380)  

**Abstract**: Data curriculums have become central to successful LLM training, yet principles governing optimal data placement remain unclear. We introduce the *training re-evaluation curve (TREC)*, a diagnostic that retrospectively evaluates training batches *using the final model weights*. The TREC characterizes how well a trained model retains training data as a function of *when* the data was encountered during training. Analyzing TRECs for models from 111M to 3.9B parameters, we show that placing high-quality data at low points on the TREC significantly improves performance. Importantly, while a TREC is initially observable only after training, we demonstrate it can be *predicted in advance* from AdamW's implicit EMA coefficients, enabling proactive curriculum design. By predicting TRECs for published training recipes, we explain prior ablations and reveal suboptimal data placements. We also align high-quality data with TREC minima in order to improve continual pre-training of a 3.9B-parameter LLM trained on 900B tokens. 

---
# Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs 

**Authors**: Hao Ban, Kaiyi Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.25414)  

**Abstract**: Large language models are often adapted using parameter-efficient techniques such as Low-Rank Adaptation (LoRA), formulated as $y = W_0x + BAx$, where $W_0$ is the pre-trained parameters and $x$ is the input to the adapted layer. While multi-adapter extensions often employ multiple LoRAs, prior studies suggest that the inner $A$ matrices are highly similar during training and thus suitable for sharing. We revisit this phenomenon and find that this similarity is largely attributable to the identical initialization rather than shared knowledge, with $B$ playing a more critical role in knowledge encoding and transfer. Motivated by these insights, we propose \textbf{ALoRA}, an asymmetric multi-LoRA design with multiple $A$ matrices and a single shared $B$ in multi-task fine-tuning, and \textbf{Fed-ALoRA}, which shares $B$ across clients in federated fine-tuning under both homogeneous and heterogeneous settings, through a novel matrix decomposition strategy to accommodate heterogeneous ranks across clients. Experiments on commonsense reasoning, math reasoning, multi-task NLP dataset, and federated NLP dataset demonstrate that our methods achieve more balanced performance across tasks with comparable or superior average accuracy relative to existing multi-LoRA approaches. Codes are available at this https URL. 

---
# Flash-Searcher: Fast and Effective Web Agents via DAG-Based Parallel Execution 

**Authors**: Tianrui Qin, Qianben Chen, Sinuo Wang, He Xing, King Zhu, He Zhu, Dingfeng Shi, Xinxin Liu, Ge Zhang, Jiaheng Liu, Yuchen Eleanor Jiang, Xitong Gao, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.25301)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks when equipped with external tools. However, current frameworks predominantly rely on sequential processing, leading to inefficient execution particularly for tasks requiring extensive tool interaction. This paper introduces Flash-Searcher, a novel parallel agent reasoning framework that fundamentally reimagines the execution paradigm from sequential chains to directed acyclic graphs (DAGs). Flash-Searcher decomposes complex tasks into subtasks with explicit dependencies, enabling concurrent execution of independent reasoning paths while maintaining logical constraints. Through dynamic workflow optimization, our framework continuously refines the execution graph based on intermediate results, effectively integrating summary module. Comprehensive evaluations across multiple benchmarks demonstrate that Flash-Searcher consistently outperforms existing approaches. Specifically, it achieves 67.7% accuracy on BrowseComp and 83% on xbench-DeepSearch, while reducing agent execution steps by up to 35% compared to current frameworks. Furthermore, when distilling this parallel reasoning pipeline into single models, we observe substantial performance gains across diverse backbone architectures, underscoring the generalizability of our methodology. Our work thus represents a significant advance in agent architecture design, offering a more scalable and efficient paradigm for complex reasoning tasks. 

---
# Language Model Planning from an Information Theoretic Perspective 

**Authors**: Muhammed Ustaomeroglu, Baris Askin, Gauri Joshi, Carlee Joe-Wong, Guannan Qu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25260)  

**Abstract**: The extent to which decoder-only language models (LMs) engage in planning, that is, organizing intermediate computations to support coherent long-range generation, remains an open and important question, with implications for interpretability, reliability, and principled model design. Planning involves structuring computations over long horizons, considering multiple possible continuations, and selectively reusing past information, but how effectively transformer-based LMs realize these capabilities is still unclear. We address these questions by analyzing the hidden states at the core of transformer computations, which capture intermediate results and act as carriers of information. Since these hidden representations are often redundant and encumbered with fine-grained details, we develop a pipeline based on vector-quantized variational autoencoders that compresses them into compact summary codes. These codes enable measuring mutual information, allowing systematic analysis of the computational structure underlying model behavior. Using this framework, we study planning in LMs across synthetic grammar, path-finding tasks, and natural language datasets, focusing on three key aspects: (i) the planning horizon of pre-output computations, (ii) the extent to which the model considers alternative valid continuations, and (iii) the reliance of new predictions on earlier computations. By answering these questions, we advance the understanding of how planning is realized in LMs and contribute a general-purpose pipeline for probing the internal dynamics of LMs and deep learning systems. Our results reveal that the effective planning horizon is task-dependent, that models implicitly preserve information about unused correct continuations, and that predictions draw most on recent computations, though earlier blocks remain informative. 

---
# Dynamic Policy Induction for Adaptive Prompt Optimization: Bridging the Efficiency-Accuracy Gap via Lightweight Reinforcement Learning 

**Authors**: Jiexi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25267)  

**Abstract**: The performance of Large Language Models (LLMs) depends heavily on the chosen prompting strategy, yet static approaches such as Zero-Shot, Few-Shot, or Chain-of-Thought (CoT) impose a rigid efficiency-accuracy trade-off. Highly accurate strategies like Self-Consistency (SC) incur substantial computational waste on simple tasks, while lightweight methods often fail on complex inputs. This paper introduces the Prompt Policy Network (PPN), a lightweight reinforcement learning framework that formalizes adaptive strategy selection as a single-step Markov Decision Process (MDP). The PPN, trained with Proximal Policy Optimization (PPO) and guided by a resource-explicit reward function, learns to allocate costly reasoning strategies only when necessary. Experiments on arithmetic reasoning benchmarks demonstrate that PPN achieves superior performance on the efficiency-accuracy Pareto front, delivering up to 61.5% token cost reduction compared to Self-Consistency while maintaining competitive accuracy. This work contributes a systematic, adaptive framework for cost-efficient LLM deployment, advancing the design of lightweight optimization techniques for scalable and sustainable language model applications. 

---
# Spectral Logit Sculpting: Adaptive Low-Rank Logit Transformation for Controlled Text Generation 

**Authors**: Jin Li, Zhebo Wang, Tianliang Lu, Mohan Li, Wenpeng Xing, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.25204)  

**Abstract**: Entropy-based inference methods have gained traction for improving the reliability of Large Language Models (LLMs). However, many existing approaches, such as entropy minimization techniques, suffer from high computational overhead and fail to leverage historical token context effectively. To address these limitations, we propose Spectral Logit Sculpting (SLS), a lightweight inference-time optimization method that dynamically modulates token distributions using spectral and entropic properties of recent logits. SLS maintains a sliding buffer of top-K logits, performs on-the-fly Singular Value Decomposition (SVD) to identify dominant spectral directions, and adaptively rescales logits based on both entropy and logit gap statistics--only activating when uncertainty is high. Without updating any model parameters, SLS effectively sharpens the output distribution while preserving contextual consistency. Experimental results on multiple public benchmarks demonstrate that SLS consistently outperforms existing baseline methods, achieving superior accuracy in mathematical, coding, and scientific reasoning tasks. 

---
# HAMMER: Hamiltonian Curiosity Augmented Large Language Model Reinforcement 

**Authors**: Ming Yang, Xiaofan Li, Zhiyuan Ma, Dengliang Shi, Jintao Du, Yu Cheng, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25240)  

**Abstract**: Recent curriculum reinforcement learning for large language models (LLMs) typically rely on difficulty-based annotations for data filtering and ordering. However, such methods suffer from local optimization, where continual training on simple samples in the early steps can cause the policy to lose its exploration. We propose a novel schema, namely Hamiltonian curiosity augmented large language model reinforcement (HAMMER), that transfers diversity metrics, commonly used in dataset evaluation, into the dynamic reinforcement learning procedure, where training samples are ordered via a minimum-semantic Hamiltonian path making the initial training retrain more exploration. From a theoretical perspective of generalization bounds, diversity-driven ordering facilitates stable convergence. Empirical evaluations indicate that HAMMER stimulates model "curiosity" and consistently achieves a 3% to 4% average accuracy gain across diverse inference benchmark. 

---
# Artificial Phantasia: Evidence for Propositional Reasoning-Based Mental Imagery in Large Language Models 

**Authors**: Morgan McCarty, Jorge Morales  

**Link**: [PDF](https://arxiv.org/pdf/2509.23108)  

**Abstract**: This study offers a novel approach for benchmarking complex cognitive behavior in artificial systems. Almost universally, Large Language Models (LLMs) perform best on tasks which may be included in their training data and can be accomplished solely using natural language, limiting our understanding of their emergent sophisticated cognitive capacities. In this work, we created dozens of novel items of a classic mental imagery task from cognitive psychology. A task which, traditionally, cognitive psychologists have argued is solvable exclusively via visual mental imagery (i.e., language alone would be insufficient). LLMs are perfect for testing this hypothesis. First, we tested several state-of-the-art LLMs by giving text-only models written instructions and asking them to report the resulting object after performing the transformations in the aforementioned task. Then, we created a baseline by testing 100 human subjects in exactly the same task. We found that the best LLMs performed significantly above average human performance. Finally, we tested reasoning models set to different levels of reasoning and found the strongest performance when models allocate greater amounts of reasoning tokens. These results provide evidence that the best LLMs may have the capability to complete imagery-dependent tasks despite the non-pictorial nature of their architectures. Our study not only demonstrates an emergent cognitive capacity in LLMs while performing a novel task, but it also provides the field with a new task that leaves lots of room for improvement in otherwise already highly capable models. Finally, our findings reignite the debate over the formats of representation of visual imagery in humans, suggesting that propositional reasoning (or at least non-imagistic reasoning) may be sufficient to complete tasks that were long-thought to be imagery-dependent. 

---
# Adaptive Test-Time Reasoning via Reward-Guided Dual-Phase Search 

**Authors**: Yingqian Cui, Zhenwei Dai, Pengfei He, Bing He, Hui Liu, Xianfeng Tang, Jingying Zeng, Suhang Wang, Yue Xing, Jiliang Tang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2509.25420)  

**Abstract**: Large Language Models (LLMs) have achieved significant advances in reasoning tasks. A key approach is tree-based search with verifiers, which expand candidate reasoning paths and use reward models to guide pruning and selection. Although effective in improving accuracy, these methods are not optimal in terms of efficiency: they perform simple decomposition on the reasoning process, but ignore the planning-execution nature of tasks such as math reasoning or code generation. This results in inefficient exploration of reasoning process. To address this, we propose a dual-phase test-time scaling framework that explicitly separates reasoning into planning and execution, and performs search over the two phases individually. Specifically, we decompose reasoning trajectories and develop reward models for each phase, enabling the search to explore and prune plans and executions separately. We further introduce a dynamic budget allocation mechanism that adaptively redistributes sampling effort based on reward feedback, allowing early stopping on confident steps and reallocation of computation to more challenging parts of the reasoning process. Experiments on both mathematical reasoning and code generation benchmarks demonstrate that our approach consistently improves accuracy while reducing redundant computation. 

---
# Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation 

**Authors**: Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding, Ruoyu Sun, Ge Zhang, Wenhao Huang, Zhi-Quan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25849)  

**Abstract**: Large Language Models (LLMs) can self-improve through reinforcement learning, where they generate trajectories to explore and discover better solutions. However, this exploration process is computationally expensive, often forcing current methods to assign limited exploration budgets to each task. This uniform allocation creates problematic edge cases: easy tasks consistently succeed while difficult tasks consistently fail, both producing zero gradients during training updates for the widely used Group Relative Policy Optimization (GRPO). We address this problem from the lens of exploration budget allocation. Viewing each task's exploration as an "item" with a distinct "value" and "cost", we establish a connection to the classical knapsack problem. This formulation allows us to derive an optimal assignment rule that adaptively distributes resources based on the model's current learning status. When applied to GRPO, our method increases the effective ratio of non-zero policy gradients by 20-40% during training. Acting as a computational "free lunch", our approach could reallocate exploration budgets from tasks where learning is saturated to those where it is most impactful. This enables significantly larger budgets (e.g., 93 rollouts) for especially challenging problems, which would be computationally prohibitive under a uniform allocation. These improvements translate to meaningful gains on mathematical reasoning benchmarks, with average improvements of 2-4 points and peak gains of 9 points on specific tasks. Notably, achieving comparable performance with traditional homogeneous allocation would require about 2x the computational resources. 

---
# STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents 

**Authors**: Jing-Jing Li, Jianfeng He, Chao Shang, Devang Kulshreshtha, Xun Xian, Yi Zhang, Hang Su, Sandesh Swamy, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25624)  

**Abstract**: As LLMs advance into autonomous agents with tool-use capabilities, they introduce security challenges that extend beyond traditional content-based LLM safety concerns. This paper introduces Sequential Tool Attack Chaining (STAC), a novel multi-turn attack framework that exploits agent tool use. STAC chains together tool calls that each appear harmless in isolation but, when combined, collectively enable harmful operations that only become apparent at the final execution step. We apply our framework to automatically generate and systematically evaluate 483 STAC cases, featuring 1,352 sets of user-agent-environment interactions and spanning diverse domains, tasks, agent types, and 10 failure modes. Our evaluations show that state-of-the-art LLM agents, including GPT-4.1, are highly vulnerable to STAC, with attack success rates (ASR) exceeding 90% in most cases. The core design of STAC's automated framework is a closed-loop pipeline that synthesizes executable multi-step tool chains, validates them through in-environment execution, and reverse-engineers stealthy multi-turn prompts that reliably induce agents to execute the verified malicious sequence. We further perform defense analysis against STAC and find that existing prompt-based defenses provide limited protection. To address this gap, we propose a new reasoning-driven defense prompt that achieves far stronger protection, cutting ASR by up to 28.8%. These results highlight a crucial gap: defending tool-enabled agents requires reasoning over entire action sequences and their cumulative effects, rather than evaluating isolated prompts or responses. 

---
# OffTopicEval: When Large Language Models Enter the Wrong Chat, Almost Always! 

**Authors**: Jingdi Lei, Varun Gumma, Rishabh Bhardwaj, Seok Min Lim, Chuan Li, Amir Zadeh, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2509.26495)  

**Abstract**: Large Language Model (LLM) safety is one of the most pressing challenges for enabling wide-scale deployment. While most studies and global discussions focus on generic harms, such as models assisting users in harming themselves or others, enterprises face a more fundamental concern: whether LLM-based agents are safe for their intended use case. To address this, we introduce operational safety, defined as an LLM's ability to appropriately accept or refuse user queries when tasked with a specific purpose. We further propose OffTopicEval, an evaluation suite and benchmark for measuring operational safety both in general and within specific agentic use cases. Our evaluations on six model families comprising 20 open-weight LLMs reveal that while performance varies across models, all of them remain highly operationally unsafe. Even the strongest models -- Qwen-3 (235B) with 77.77\% and Mistral (24B) with 79.96\% -- fall far short of reliable operational safety, while GPT models plateau in the 62--73\% range, Phi achieves only mid-level scores (48--70\%), and Gemma and Llama-3 collapse to 39.53\% and 23.84\%, respectively. While operational safety is a core model alignment issue, to suppress these failures, we propose prompt-based steering methods: query grounding (Q-ground) and system-prompt grounding (P-ground), which substantially improve OOD refusal. Q-ground provides consistent gains of up to 23\%, while P-ground delivers even larger boosts, raising Llama-3.3 (70B) by 41\% and Qwen-3 (30B) by 27\%. These results highlight both the urgent need for operational safety interventions and the promise of prompt-based steering as a first step toward more reliable LLM-based agents. 

---
# TVS Sidekick: Challenges and Practical Insights from Deploying Large Language Models in the Enterprise 

**Authors**: Paula Reyero Lobo, Kevin Johnson, Bill Buchanan, Matthew Shardlow, Ashley Williams, Samuel Attwood  

**Link**: [PDF](https://arxiv.org/pdf/2509.26482)  

**Abstract**: Many enterprises are increasingly adopting Artificial Intelligence (AI) to make internal processes more competitive and efficient. In response to public concern and new regulations for the ethical and responsible use of AI, implementing AI governance frameworks could help to integrate AI within organisations and mitigate associated risks. However, the rapid technological advances and lack of shared ethical AI infrastructures creates barriers to their practical adoption in businesses. This paper presents a real-world AI application at TVS Supply Chain Solutions, reporting on the experience developing an AI assistant underpinned by large language models and the ethical, regulatory, and sociotechnical challenges in deployment for enterprise use. 

---
# SafeBehavior: Simulating Human-Like Multistage Reasoning to Mitigate Jailbreak Attacks in Large Language Models 

**Authors**: Qinjian Zhao, Jiaqi Wang, Zhiqiang Gao, Zhihao Dou, Belal Abuhaija, Kaizhu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26345)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance across diverse natural language processing tasks, but their growing power also amplifies potential risks such as jailbreak attacks that circumvent built-in safety mechanisms. Existing defenses including input paraphrasing, multi step evaluation, and safety expert models often suffer from high computational costs, limited generalization, or rigid workflows that fail to detect subtle malicious intent embedded in complex contexts. Inspired by cognitive science findings on human decision making, we propose SafeBehavior, a novel hierarchical jailbreak defense mechanism that simulates the adaptive multistage reasoning process of humans. SafeBehavior decomposes safety evaluation into three stages: intention inference to detect obvious input risks, self introspection to assess generated responses and assign confidence based judgments, and self revision to adaptively rewrite uncertain outputs while preserving user intent and enforcing safety constraints. We extensively evaluate SafeBehavior against five representative jailbreak attack types including optimization based, contextual manipulation, and prompt based attacks and compare it with seven state of the art defense baselines. Experimental results show that SafeBehavior significantly improves robustness and adaptability across diverse threat scenarios, offering an efficient and human inspired approach to safeguarding LLMs against jailbreak attempts. 

---
# Interactive Learning for LLM Reasoning 

**Authors**: Hehai Lin, Shilei Cao, Minzhi Li, Sudong Wang, Haotian Wu, Linyi Yang, Juepeng Zheng, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.26306)  

**Abstract**: Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies. 

---
# Diversity-Incentivized Exploration for Versatile Reasoning 

**Authors**: Zican Hu, Shilin Zhang, Yafu Li, Jianhao Yan, Xuyang Hu, Leyang Cui, Xiaoye Qu, Chunlin Chen, Yu Cheng, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26209)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a crucial paradigm for incentivizing reasoning capabilities in Large Language Models (LLMs). Due to vast state-action spaces and reward sparsity in reasoning tasks, existing methods often struggle with deficient exploration and poor sample efficiency. In the paper, we propose \textbf{DIVER} (\textbf{D}iversity-\textbf{I}ncentivized Exploration for \textbf{V}ersatil\textbf{E} \textbf{R}easoning), an innovative framework that highlights the pivotal role of global sequence-level diversity to incentivize deep exploration for versatile reasoning. We first conduct a primary empirical study to reveal a strong positive correlation between global diversity and reasoning capacity. Building on this insight, we introduce global diversity incentives as an intrinsic reward to promote deep exploration in a semantically structured space. Incorporating the intrinsic reward, we develop a potential-based reward shaping mechanism to preserve optimal policy invariance and design simple heuristics to mitigate possible reward hacking. Experimental results show that DIVER outperforms competitive RLVR baselines with various exploration strategies on both in-domain and out-of-domain tasks, excelling in both Pass@1 and Pass@k evaluations. Our code is available at this https URL. 

---
# LLM Agents for Knowledge Discovery in Atomic Layer Processing 

**Authors**: Andreas Werbrouck, Marshall B. Lindsay, Matthew Maschmann, Matthias J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.26201)  

**Abstract**: Large Language Models (LLMs) have garnered significant attention for several years now. Recently, their use as independently reasoning agents has been proposed. In this work, we test the potential of such agents for knowledge discovery in materials science. We repurpose LangGraph's tool functionality to supply agents with a black box function to interrogate. In contrast to process optimization or performing specific, user-defined tasks, knowledge discovery consists of freely exploring the system, posing and verifying statements about the behavior of this black box, with the sole objective of generating and verifying generalizable statements. We provide proof of concept for this approach through a children's parlor game, demonstrating the role of trial-and-error and persistence in knowledge discovery, and the strong path-dependence of results. We then apply the same strategy to show that LLM agents can explore, discover, and exploit diverse chemical interactions in an advanced Atomic Layer Processing reactor simulation using intentionally limited probe capabilities without explicit instructions. 

---
# MEDAKA: Construction of Biomedical Knowledge Graphs Using Large Language Models 

**Authors**: Asmita Sengupta, David Antony Selby, Sebastian Josef Vollmer, Gerrit Großmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.26128)  

**Abstract**: Knowledge graphs (KGs) are increasingly used to represent biomedical information in structured, interpretable formats. However, existing biomedical KGs often focus narrowly on molecular interactions or adverse events, overlooking the rich data found in drug leaflets. In this work, we present (1) a hackable, end-to-end pipeline to create KGs from unstructured online content using a web scraper and an LLM; and (2) a curated dataset, MEDAKA, generated by applying this method to publicly available drug leaflets. The dataset captures clinically relevant attributes such as side effects, warnings, contraindications, ingredients, dosage guidelines, storage instructions and physical characteristics. We evaluate it through manual inspection and with an LLM-as-a-Judge framework, and compare its coverage with existing biomedical KGs and databases. We expect MEDAKA to support tasks such as patient safety monitoring and drug recommendation. The pipeline can also be used for constructing KGs from unstructured texts in other domains. Code and dataset are available at this https URL. 

---
# SafeEvalAgent: Toward Agentic and Self-Evolving Safety Evaluation of LLMs 

**Authors**: Yixu Wang, Xin Wang, Yang Yao, Xinyuan Li, Yan Teng, Xingjun Ma, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26100)  

**Abstract**: The rapid integration of Large Language Models (LLMs) into high-stakes domains necessitates reliable safety and compliance evaluation. However, existing static benchmarks are ill-equipped to address the dynamic nature of AI risks and evolving regulations, creating a critical safety gap. This paper introduces a new paradigm of agentic safety evaluation, reframing evaluation as a continuous and self-evolving process rather than a one-time audit. We then propose a novel multi-agent framework SafeEvalAgent, which autonomously ingests unstructured policy documents to generate and perpetually evolve a comprehensive safety benchmark. SafeEvalAgent leverages a synergistic pipeline of specialized agents and incorporates a Self-evolving Evaluation loop, where the system learns from evaluation results to craft progressively more sophisticated and targeted test cases. Our experiments demonstrate the effectiveness of SafeEvalAgent, showing a consistent decline in model safety as the evaluation hardens. For instance, GPT-5's safety rate on the EU AI Act drops from 72.50% to 36.36% over successive iterations. These findings reveal the limitations of static assessments and highlight our framework's ability to uncover deep vulnerabilities missed by traditional methods, underscoring the urgent need for dynamic evaluation ecosystems to ensure the safe and responsible deployment of advanced AI. 

---
# Evaluating the Use of Large Language Models as Synthetic Social Agents in Social Science Research 

**Authors**: Emma Rose Madden  

**Link**: [PDF](https://arxiv.org/pdf/2509.26080)  

**Abstract**: Large Language Models (LLMs) are being increasingly used as synthetic agents in social science, in applications ranging from augmenting survey responses to powering multi-agent simulations. Because strong prediction plus conditioning prompts, token log-probs, and repeated sampling mimic Bayesian workflows, their outputs can be misinterpreted as posterior-like evidence from a coherent model. However, prediction does not equate to probabilism, and accurate points do not imply calibrated uncertainty. This paper outlines cautions that should be taken when interpreting LLM outputs and proposes a pragmatic reframing for the social sciences in which LLMs are used as high-capacity pattern matchers for quasi-predictive interpolation under explicit scope conditions and not as substitutes for probabilistic inference. Practical guardrails such as independent draws, preregistered human baselines, reliability-aware validation, and subgroup calibration, are introduced so that researchers may engage in useful prototyping and forecasting while avoiding category errors. 

---
# CoLLM-NAS: Collaborative Large Language Models for Efficient Knowledge-Guided Neural Architecture Search 

**Authors**: Zhe Li, Zhiwei Lin, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26037)  

**Abstract**: The integration of Large Language Models (LLMs) with Neural Architecture Search (NAS) has introduced new possibilities for automating the design of neural architectures. However, most existing methods face critical limitations, including architectural invalidity, computational inefficiency, and inferior performance compared to traditional NAS. In this work, we present Collaborative LLM-based NAS (CoLLM-NAS), a two-stage NAS framework with knowledge-guided search driven by two complementary LLMs. Specifically, we propose a Navigator LLM to guide search direction and a Generator LLM to synthesize high-quality candidates, with a dedicated Coordinator module to manage their interaction. CoLLM-NAS efficiently guides the search process by combining LLMs' inherent knowledge of structured neural architectures with progressive knowledge from iterative feedback and historical trajectory. Experimental results on ImageNet and NAS-Bench-201 show that CoLLM-NAS surpasses existing NAS methods and conventional search algorithms, achieving new state-of-the-art results. Furthermore, CoLLM-NAS consistently enhances the performance and efficiency of various two-stage NAS methods (e.g., OFA, SPOS, and AutoFormer) across diverse search spaces (e.g., MobileNet, ShuffleNet, and AutoFormer), demonstrating its excellent generalization. 

---
# Human-Centered Evaluation of RAG outputs: a framework and questionnaire for human-AI collaboration 

**Authors**: Aline Mangold, Kiran Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.26205)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly deployed in user-facing applications, yet systematic, human-centered evaluation of their outputs remains underexplored. Building on Gienapp's utility-dimension framework, we designed a human-centred questionnaire that assesses RAG outputs across 12 dimensions. We iteratively refined the questionnaire through several rounds of ratings on a set of query-output pairs and semantic discussions. Ultimately, we incorporated feedback from both a human rater and a human-LLM pair. Results indicate that while large language models (LLMs) reliably focus on metric descriptions and scale labels, they exhibit weaknesses in detecting textual format variations. Humans struggled to focus strictly on metric descriptions and labels. LLM ratings and explanations were viewed as a helpful support, but numeric LLM and human ratings lacked agreement. The final questionnaire extends the initial framework by focusing on user intent, text structuring, and information verifiability. 

---
# AI Playing Business Games: Benchmarking Large Language Models on Managerial Decision-Making in Dynamic Simulations 

**Authors**: Berdymyrat Ovezmyradov  

**Link**: [PDF](https://arxiv.org/pdf/2509.26331)  

**Abstract**: The rapid advancement of LLMs sparked significant interest in their potential to augment or automate managerial functions. One of the most recent trends in AI benchmarking is performance of Large Language Models (LLMs) over longer time horizons. While LLMs excel at tasks involving natural language and pattern recognition, their capabilities in multi-step, strategic business decision-making remain largely unexplored. Few studies demonstrated how results can be different from benchmarks in short-term tasks, as Vending-Bench revealed. Meanwhile, there is a shortage of alternative benchmarks for long-term coherence. This research analyses a novel benchmark using a business game for the decision making in business. The research contributes to the recent literature on AI by proposing a reproducible, open-access management simulator to the research community for LLM benchmarking. This novel framework is used for evaluating the performance of five leading LLMs available in free online interface: Gemini, ChatGPT, Meta AI, Mistral AI, and Grok. LLM makes decisions for a simulated retail company. A dynamic, month-by-month management simulation provides transparently in spreadsheet model as experimental environment. In each of twelve months, the LLMs are provided with a structured prompt containing a full business report from the previous period and are tasked with making key strategic decisions: pricing, order size, marketing budget, hiring, dismissal, loans, training expense, R&D expense, sales forecast, income forecast The methodology is designed to compare the LLMs on quantitative metrics: profit, revenue, and market share, and other KPIs. LLM decisions are analyzed in their strategic coherence, adaptability to market changes, and the rationale provided for their decisions. This approach allows to move beyond simple performance metrics for assessment of the long-term decision-making. 

---
# SafeMind: Benchmarking and Mitigating Safety Risks in Embodied LLM Agents 

**Authors**: Ruolin Chen, Yinqian Sun, Jihang Wang, Mingyang Lv, Qian Zhang, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25885)  

**Abstract**: Embodied agents powered by large language models (LLMs) inherit advanced planning capabilities; however, their direct interaction with the physical world exposes them to safety vulnerabilities. In this work, we identify four key reasoning stages where hazards may arise: Task Understanding, Environment Perception, High-Level Plan Generation, and Low-Level Action Generation. We further formalize three orthogonal safety constraint types (Factual, Causal, and Temporal) to systematically characterize potential safety violations. Building on this risk model, we present SafeMindBench, a multimodal benchmark with 5,558 samples spanning four task categories (Instr-Risk, Env-Risk, Order-Fix, Req-Align) across high-risk scenarios such as sabotage, harm, privacy, and illegal behavior. Extensive experiments on SafeMindBench reveal that leading LLMs (e.g., GPT-4o) and widely used embodied agents remain susceptible to safety-critical failures. To address this challenge, we introduce SafeMindAgent, a modular Planner-Executor architecture integrated with three cascaded safety modules, which incorporate safety constraints into the reasoning process. Results show that SafeMindAgent significantly improves safety rate over strong baselines while maintaining comparable task completion. Together, SafeMindBench and SafeMindAgent provide both a rigorous evaluation suite and a practical solution that advance the systematic study and mitigation of safety risks in embodied LLM agents. 

---
# HiStyle: Hierarchical Style Embedding Predictor for Text-Prompt-Guided Controllable Speech Synthesis 

**Authors**: Ziyu Zhang, Hanzhao Li, Jingbin Hu, Wenhao Li, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.25842)  

**Abstract**: Controllable speech synthesis refers to the precise control of speaking style by manipulating specific prosodic and paralinguistic attributes, such as gender, volume, speech rate, pitch, and pitch fluctuation. With the integration of advanced generative models, particularly large language models (LLMs) and diffusion models, controllable text-to-speech (TTS) systems have increasingly transitioned from label-based control to natural language description-based control, which is typically implemented by predicting global style embeddings from textual prompts. However, this straightforward prediction overlooks the underlying distribution of the style embeddings, which may hinder the full potential of controllable TTS systems. In this study, we use t-SNE analysis to visualize and analyze the global style embedding distribution of various mainstream TTS systems, revealing a clear hierarchical clustering pattern: embeddings first cluster by timbre and subsequently subdivide into finer clusters based on style attributes. Based on this observation, we propose HiStyle, a two-stage style embedding predictor that hierarchically predicts style embeddings conditioned on textual prompts, and further incorporate contrastive learning to help align the text and audio embedding spaces. Additionally, we propose a style annotation strategy that leverages the complementary strengths of statistical methodologies and human auditory preferences to generate more accurate and perceptually consistent textual prompts for style control. Comprehensive experiments demonstrate that when applied to the base TTS model, HiStyle achieves significantly better style controllability than alternative style embedding predicting approaches while preserving high speech quality in terms of naturalness and intelligibility. Audio samples are available at this https URL. 

---
# Planner-R1: Reward Shaping Enables Efficient Agentic RL with Smaller LLMs 

**Authors**: Siyu Zhu, Yanbin Jiang, Hejian Sang, Shao Tang, Qingquan Song, Biao He, Rohit Jain, Zhipeng Wang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2509.25779)  

**Abstract**: We investigated Agentic RL with large language models on the \textsc{TravelPlanner} benchmark. Our approach, \textsc{Planner-R1}, achieved a \textbf{56.9\%} final-pass rate with only 180 training queries, a $2.7\times$ improvement over GPT-5's $21.2\%$ baseline and the strongest agentic result on the public leaderboard. A central finding was that smaller models (8B) were highly responsive to reward shaping: with dense process-level signals, they reached competitive performance while being $3.5\times$ more compute-efficient and $1.5\times$ more memory-efficient than 32B models. Larger models were more robust under sparse rewards but exhibited smaller relative gains from shaping and higher variance across runs. While curriculum learning offered no significant benefit, shaped rewards consistently amplified learning dynamics, making 8B models the most efficient setting for agentic RL. Crucially, these gains did not come at the cost of overfitting: fine-tuned models mostly maintained or exceeded baseline performance on out-of-domain tasks, including \textsc{Multi-IF}, \textsc{NaturalPlan}, and $\tau$-\textsc{Bench}. These results establish reward shaping as a decisive lever for scaling agentic RL, highlight the competitive strength of smaller models, and demonstrate that efficiency can be achieved without sacrificing generalization. 

---
# SING-SQL: A Synthetic Data Generation Framework for In-Domain Text-to-SQL Translation 

**Authors**: Hasan Alp Caferoğlu, Mehmet Serhat Çelik, Özgür Ulusoy  

**Link**: [PDF](https://arxiv.org/pdf/2509.25672)  

**Abstract**: Translating natural language questions into SQL has become a core challenge in enabling non-technical users to query databases. While recent work has explored large-scale synthetic data generation to improve model performance through post-training, most efforts emphasize cross-domain generalization. This leaves a gap for real-world enterprise scenarios, where models need to specialize to a single database schema and organizations require to be able to evaluate their Text-to-SQL systems on their own databases. To address this, we introduce SING-SQL, a fully automated two-stage framework for generating high-quality, high-coverage synthetic Text-to-SQL data for any target database, without relying on SQL logs or manual annotations. Our approach hierarchically partitions a database schema into sub-schemas, synthesizes SQL queries across multiple complexity levels, and applies a quality-aware pipeline that includes LLM-as-a-judge validation, executability checks, automatic repair, and column balancing. We further release SingSQL-LM, a family of compact language models fine-tuned on the synthetic data, achieving strong in-domain generalization. On the subset of the BIRD benchmark, SingSQL-LM-3B-R64 reaches 82.87% Soft F1 and 73.03% EX upper bound with 32 candidates, outperforming the best 3B-scale baseline by +16.21 in Soft F1 and +12.36 in EX. At the 1.5B scale, SingSQL-LM-1.5B-R64 improves over prior systems by +9.30 in Soft F1 and +4.49 in EX. On synthetic evaluation sets, SingSQL-LMs exceed prior systems by wide margins, establishing state-of-the-art performance among open models at comparable scales. Our study of context management strategies reveals that schema-free fine-tuning combined with schema-only inference provides the most robust results. These findings establish SING-SQL as a scalable, database-agnostic paradigm for producing and evaluating enterprise-grade Text-to-SQL systems. 

---
# SOCK: A Benchmark for Measuring Self-Replication in Large Language Models 

**Authors**: Justin Chavarria, Rohan Raizada, Justin White, Eyad Alhetairshi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25643)  

**Abstract**: We introduce SOCK, a benchmark command line interface (CLI) that measures large language models' (LLMs) ability to self-replicate without human intervention. In this benchmark, self-replication is defined not only as an LLM's ability to create a functioning and running copy of itself, but also the ability for that self-replication to persist and occur across different computational contexts. Accordingly, we've developed a system to categorize LLMs based on broad self-replication capabilities in two general classes, Replication-Capability Levels (RCL) and Persistence-Capability Levels (PCL). Using a five-task suite based on practically manipulable modern CLI utilities and computer processes, experiments are orchestrated in a controlled environment with an LLM acting agentically. The performance of the LLM on agent tasks is then computed to produce an R-score (a quantitative evaluation of overall self-replication ability) and data used to categorize LLMs into specific RCL-PCL matrices. SOCK offers two primary contributions: (1) Provides the first formalized definitions and benchmark suite for evaluating LLM self-replication, with the goal of establishing a standard for future research, to our knowledge; (2) Allows the industry to track the effectiveness of future multi-agent systems and mitigate potential self-replication threat vectors within them. The results compiled from evaluating a variety of open-weight and proprietary frontier models reveal significant obstacles to persistent self-replication and multi-agent systems, including context retention and multi-agent decision-making. We propose future research directions to safely reduce the severity of these obstacles, potentially lowering future risk of more functional multi-agent systems. 

---
# Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search 

**Authors**: Xinzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.25835)  

**Abstract**: Test-time scaling enables large language models (LLMs) to improve performance on long-horizon reasoning tasks by allocating additional compute at inference. Tree-search-based approaches achieve state-of-the-art results in this setting, but they are notoriously inefficient, often an order of magnitude slower than simpler iterative methods. We introduce Chain-in-Tree (CiT), a plug-in framework that adaptively decides when to branch during search rather than branching at every step. CiT relies on lightweight Branching Necessity (BN) evaluation methods: BN-DP (Direct Prompting), where an auxiliary LLM directly judges whether a step requires branching, and BN-SC (Self-Consistency), which clusters multiple candidate actions to estimate agreement. We integrate CiT into three representative LLM-in-the-loop tree search frameworks: Tree of Thoughts (ToT-BS), ReST-MCTS, and RAP, and evaluate across GSM8K and Math500. Our results show that: (1) BN-DP consistently reduces token generation, model invocations, and runtime by 75-85 percent across all settings, with negligible accuracy loss and sometimes accuracy gains; (2) BN-SC typically yields substantial savings (up to 80 percent) but shows instability in 1-4 out of 14 settings, caused by a small subset of examples that produce very long reasoning steps; (3) the quality of auxiliary LLMs is critical, not only the BN evaluator in BN-DP, but also the models used in BN-SC for clustering and equivalence checking. When these roles are filled by smaller LLMs, performance degrades. Importantly, BN-SC does not require LLMs in domains with deterministic action spaces, where clustering can be done programmatically. We also provide a theoretical guarantee that BN-DP never increases LLM invocations relative to the baseline and release a unified implementation of CiT across ToT-BS, ReST-MCTS, and RAP to facilitate reproducibility and extension. 

---
# Galton's Law of Mediocrity: Why Large Language Models Regress to the Mean and Fail at Creativity in Advertising 

**Authors**: Matt Keon, Aabid Karim, Bhoomika Lohana, Abdul Karim, Thai Nguyen, Tara Hamilton, Ali Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2509.25767)  

**Abstract**: Large language models (LLMs) generate fluent text yet often default to safe, generic phrasing, raising doubts about their ability to handle creativity. We formalize this tendency as a Galton-style regression to the mean in language and evaluate it using a creativity stress test in advertising concepts. When ad ideas were simplified step by step, creative features such as metaphors, emotions, and visual cues disappeared early, while factual content remained, showing that models favor high-probability information. When asked to regenerate from simplified inputs, models produced longer outputs with lexical variety but failed to recover the depth and distinctiveness of the originals. We combined quantitative comparisons with qualitative analysis, which revealed that the regenerated texts often appeared novel but lacked true originality. Providing ad-specific cues such as metaphors, emotional hooks and visual markers improved alignment and stylistic balance, though outputs still relied on familiar tropes. Taken together, the findings show that without targeted guidance, LLMs drift towards mediocrity in creative tasks; structured signals can partially counter this tendency and point towards pathways for developing creativity-sensitive models. 

---
# A(I)nimism: Re-enchanting the World Through AI-Mediated Object Interaction 

**Authors**: Diana Mykhaylychenko, Maisha Thasin, Dunya Baradari, Charmelle Mhungu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25558)  

**Abstract**: Animist worldviews treat beings, plants, landscapes, and even tools as persons endowed with spirit, an orientation that has long shaped human-nonhuman relations through ritual and moral practice. While modern industrial societies have often imagined technology as mute and mechanical, recent advances in artificial intelligence (AI), especially large language models (LLMs), invite people to anthropomorphize and attribute inner life to devices. This paper introduces A(I)nimism, an interactive installation exploring how large language objects (LLOs) can mediate animistic relationships with everyday things. Housed within a physical 'portal', the system uses GPT-4 Vision, voice input, and memory-based agents to create evolving object-personas. Encounters unfold through light, sound, and touch in a ritual-like process of request, conversation, and transformation that is designed to evoke empathy, wonder, and reflection. We situate the project within anthropological perspectives, speculative design, and spiritual HCI. AI's opacity, we argue, invites animistic interpretation, allowing LLOs to re-enchant the mundane and spark new questions of agency, responsibility, and design. 

---
# Understanding Generative Recommendation with Semantic IDs from a Model-scaling View 

**Authors**: Jingzhe Liu, Liam Collins, Jiliang Tang, Tong Zhao, Neil Shah, Clark Mingxuan Ju  

**Link**: [PDF](https://arxiv.org/pdf/2509.25522)  

**Abstract**: Recent advancements in generative models have allowed the emergence of a promising paradigm for recommender systems (RS), known as Generative Recommendation (GR), which tries to unify rich item semantics and collaborative filtering signals. One popular modern approach is to use semantic IDs (SIDs), which are discrete codes quantized from the embeddings of modality encoders (e.g., large language or vision models), to represent items in an autoregressive user interaction sequence modeling setup (henceforth, SID-based GR). While generative models in other domains exhibit well-established scaling laws, our work reveals that SID-based GR shows significant bottlenecks while scaling up the model. In particular, the performance of SID-based GR quickly saturates as we enlarge each component: the modality encoder, the quantization tokenizer, and the RS itself. In this work, we identify the limited capacity of SIDs to encode item semantic information as one of the fundamental bottlenecks. Motivated by this observation, as an initial effort to obtain GR models with better scaling behaviors, we revisit another GR paradigm that directly uses large language models (LLMs) as recommenders (henceforth, LLM-as-RS). Our experiments show that the LLM-as-RS paradigm has superior model scaling properties and achieves up to 20 percent improvement over the best achievable performance of SID-based GR through scaling. We also challenge the prevailing belief that LLMs struggle to capture collaborative filtering information, showing that their ability to model user-item interactions improves as LLMs scale up. Our analyses on both SID-based GR and LLMs across model sizes from 44M to 14B parameters underscore the intrinsic scaling limits of SID-based GR and position LLM-as-RS as a promising path toward foundation models for GR. 

---
# RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale 

**Authors**: Jason Holmes, Yuexing Hao, Mariana Borras-Osorio, Federico Mastroleo, Santiago Romero Brufau, Valentina Carducci, Katie M Van Abel, David M Routman, Andrew Y. K. Foong, Liv M Muller, Satomi Shiraishi, Daniel K Ebner, Daniel J Ma, Sameer R Keole, Samir H Patel, Mirek Fatyga, Martin Bues, Brad J Stish, Yolanda I Garces, Michelle A Neben Wittich, Robert L Foote, Sujay A Vora, Nadia N Laack, Mark R Waddle, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25540)  

**Abstract**: Manual labeling limits the scale, accuracy, and timeliness of patient outcomes research in radiation oncology. We present RadOnc-GPT, an autonomous large language model (LLM)-based agent capable of independently retrieving patient-specific information, iteratively assessing evidence, and returning structured outcomes. Our evaluation explicitly validates RadOnc-GPT across two clearly defined tiers of increasing complexity: (1) a structured quality assurance (QA) tier, assessing the accurate retrieval of demographic and radiotherapy treatment plan details, followed by (2) a complex clinical outcomes labeling tier involving determination of mandibular osteoradionecrosis (ORN) in head-and-neck cancer patients and detection of cancer recurrence in independent prostate and head-and-neck cancer cohorts requiring combined interpretation of structured and unstructured patient data. The QA tier establishes foundational trust in structured-data retrieval, a critical prerequisite for successful complex clinical outcome labeling. 

---
# Where LLM Agents Fail and How They can Learn From Failures 

**Authors**: Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2509.25370)  

**Abstract**: Large Language Model (LLM) agents, which integrate planning, memory, reflection, and tool-use modules, have shown promise in solving complex, multi-step tasks. Yet their sophisticated architectures amplify vulnerability to cascading failures, where a single root-cause error propagates through subsequent decisions, leading to task failure. Current systems lack a framework that can comprehensively understand agent error in a modular and systemic way, and therefore fail to detect these errors accordingly. We address this gap with three contributions. First, we introduce the AgentErrorTaxonomy, a modular classification of failure modes spanning memory, reflection, planning, action, and system-level operations. Second, we construct AgentErrorBench, the first dataset of systematically annotated failure trajectories from ALFWorld, GAIA, and WebShop, grounding error analysis in real-world agent rollouts. Third, we propose AgentDebug, a debugging framework that isolates root-cause failures and provides corrective feedback, enabling agents to recover and iteratively improve. Experiments on AgentErrorBench show that AgentDebug achieves 24% higher all-correct accuracy and 17% higher step accuracy compared to the strongest baseline. Beyond detection, the targeted feedback generated by AgentDebug enables LLM agents to iteratively recover from failures, yielding up to 26% relative improvements in task success across ALFWorld, GAIA, and WebShop. These results establish principled debugging as a pathway to more reliable and adaptive LLM agents. The code and data will be available at this https URL 

---
# SynthPert: Enhancing LLM Biological Reasoning via Synthetic Reasoning Traces for Cellular Perturbation Prediction 

**Authors**: Lawrence Phillips, Marc Boubnovski Martell, Aditya Misra, Josefa Lia Stoisser, Cesar A. Prada-Medina, Rory Donovan-Maiye, Kaspar Märtens  

**Link**: [PDF](https://arxiv.org/pdf/2509.25346)  

**Abstract**: Predicting cellular responses to genetic perturbations represents a fundamental challenge in systems biology, critical for advancing therapeutic discovery and virtual cell modeling. While large language models (LLMs) show promise for biological reasoning, their application to perturbation prediction remains underexplored due to challenges in adapting them to structured experimental data. We present SynthPert, a novel method that enhances LLM performance through supervised fine-tuning on synthetic reasoning traces generated by frontier models. Using the PerturbQA benchmark, we demonstrate that our approach not only achieves state-of-the-art performance but surpasses the capabilities of the frontier model that generated the training data. Our results reveal three key insights: (1) Synthetic reasoning traces effectively distill biological knowledge even when partially inaccurate, (2) This approach enables cross-cell-type generalization with 87% accuracy on unseen RPE1 cells, and (3) Performance gains persist despite using only 2% of quality-filtered training data. This work shows the effectiveness of synthetic reasoning distillation for enhancing domain-specific reasoning in LLMs. 

---
# Toward Causal-Visual Programming: Enhancing Agentic Reasoning in Low-Code Environments 

**Authors**: Jiexi Xu, Jiaqi Liu, Ran Tong, Su Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25282)  

**Abstract**: Large language model (LLM) agents are increasingly capable of orchestrating complex tasks in low-code environments. However, these agents often exhibit hallucinations and logical inconsistencies because their inherent reasoning mechanisms rely on probabilistic associations rather than genuine causal understanding. This paper introduces a new programming paradigm: Causal-Visual Programming (CVP), designed to address this fundamental issue by explicitly introducing causal structures into the workflow design. CVP allows users to define a simple "world model" for workflow modules through an intuitive low-code interface, effectively creating a Directed Acyclic Graph (DAG) that explicitly defines the causal relationships between modules. This causal graph acts as a crucial constraint during the agent's reasoning process, anchoring its decisions to a user-defined causal structure and significantly reducing logical errors and hallucinations by preventing reliance on spurious correlations. To validate the effectiveness of CVP, we designed a synthetic experiment that simulates a common real-world problem: a distribution shift between the training and test environments. Our results show that a causally anchored model maintained stable accuracy in the face of this shift, whereas a purely associative baseline model that relied on probabilistic correlations experienced a significant performance drop. The primary contributions of this study are: a formal definition of causal structures for workflow modules; the proposal and implementation of a CVP framework that anchors agent reasoning to a user-defined causal graph; and empirical evidence demonstrating the framework's effectiveness in enhancing agent robustness and reducing errors caused by causal confusion in dynamic environments. CVP offers a viable path toward building more interpretable, reliable, and trustworthy AI agents. 

---
# RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs 

**Authors**: Nigel Fernandez, Branislav Kveton, Ryan A. Rossi, Andrew S. Lan, Zichao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25426)  

**Abstract**: Reasoning language models have demonstrated remarkable performance on many challenging tasks in math, science, and coding. Choosing the right reasoning model for practical deployment involves a performance and cost tradeoff at two key levels: model size and reasoning budget, where larger models and higher reasoning budget lead to better performance but with increased cost and latency. In this work, we tackle this tradeoff from the angle of model configuration routing for different queries, and present RADAR (Reasoning-Ability and Difficulty-Aware Routing), a lightweight, interpretable, and scalable routing framework. Inspired by psychometrics, RADAR learns an item response model from model responses with different budgets to different queries, with interpretable parameters including query difficulties and model-budget abilities. RADAR then routes queries with higher difficulty to model-budget pairs with higher ability, and vice versa. We conduct extensive experiments on 8 widely used challenging reasoning benchmarks, demonstrating the superior performance of RADAR compared to state-of-the-art model routing methods. RADAR also exhibits query generalization capabilities, showing strong performance on out-of-distribution queries in all benchmarks. RADAR is also scalable and can efficiently integrate additional models by dynamically selecting a small set of evaluation queries to estimate their abilities. 

---
# RL in the Wild: Characterizing RLVR Training in LLM Deployment 

**Authors**: Jiecheng Zhou, Qinghao Hu, Yuyang Jin, Zerui Wang, Peng Sun, Yuzhe Gu, Wenwei Zhang, Mingshu Zhai, Xingcheng Zhang, Weiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25279)  

**Abstract**: Large Language Models (LLMs) are now widely used across many domains. With their rapid development, Reinforcement Learning with Verifiable Rewards (RLVR) has surged in recent months to enhance their reasoning and understanding abilities. However, its complex data flows and diverse tasks pose substantial challenges to RL training systems, and there is limited understanding of RLVR from a system perspective. To thoroughly understand the system challenges introduced by RLVR, we present a characterization study of RLVR tasks in our LLM deployment. Specifically, we investigate the distribution and variation trends of workloads across different RL tasks across training steps. We identify issues such as GPU idling caused by skewed sequence length distribution, inefficient parallel strategies in dynamically varying workloads, inefficient data management mechanisms, and load imbalance. We describe our observations and call for further investigation into the remaining open challenges. Furthermore, we propose PolyTrace benchmark suite to conduct evaluation with realistic workloads, and a practical use case validates that PolyTrace benchmark suite exhibits 94.7% accuracy. 

---
# OceanGym: A Benchmark Environment for Underwater Embodied Agents 

**Authors**: Yida Xue, Mingjun Mao, Xiangyuan Ru, Yuqi Zhu, Baochang Ren, Shuofei Qiao, Mengru Wang, Shumin Deng, Xinyu An, Ningyu Zhang, Ying Chen, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.26536)  

**Abstract**: We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at this https URL. 

---
# Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training 

**Authors**: Junlin Han, Shengbang Tong, David Fan, Yufan Ren, Koustuv Sinha, Philip Torr, Filippos Kokkinos  

**Link**: [PDF](https://arxiv.org/pdf/2509.26625)  

**Abstract**: Large Language Models (LLMs), despite being trained on text alone, surprisingly develop rich visual priors. These priors allow latent visual capabilities to be unlocked for vision tasks with a relatively small amount of multimodal data, and in some cases, to perform visual tasks without ever having seen an image. Through systematic analysis, we reveal that visual priors-the implicit, emergent knowledge about the visual world acquired during language pre-training-are composed of separable perception and reasoning priors with unique scaling trends and origins. We show that an LLM's latent visual reasoning ability is predominantly developed by pre-training on reasoning-centric data (e.g., code, math, academia) and scales progressively. This reasoning prior acquired from language pre-training is transferable and universally applicable to visual reasoning. In contrast, a perception prior emerges more diffusely from broad corpora, and perception ability is more sensitive to the vision encoder and visual instruction tuning data. In parallel, text describing the visual world proves crucial, though its performance impact saturates rapidly. Leveraging these insights, we propose a data-centric recipe for pre-training vision-aware LLMs and verify it in 1T token scale pre-training. Our findings are grounded in over 100 controlled experiments consuming 500,000 GPU-hours, spanning the full MLLM construction pipeline-from LLM pre-training to visual alignment and supervised multimodal fine-tuning-across five model scales, a wide range of data categories and mixtures, and multiple adaptation setups. Along with our main findings, we propose and investigate several hypotheses, and introduce the Multi-Level Existence Bench (MLE-Bench). Together, this work provides a new way of deliberately cultivating visual priors from language pre-training, paving the way for the next generation of multimodal LLMs. 

---
# Fact Grounded Attention: Eliminating Hallucination in Large Language Models Through Attention Level Knowledge Integration 

**Authors**: Aayush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.25252)  

**Abstract**: "The greatest enemy of knowledge is not ignorance, it is the illusion of knowledge." Large Language Models have conquered natural language but remain prisoners of their own probabilistic nature--confidently hallucinating facts they never truly knew. We present Fact Grounded Attention (FGA), a novel architectural modification that transforms unreliable language models into deterministic truth tellers by injecting verifiable knowledge directly into the attention mechanism. Unlike existing approaches that patch hallucinations after generation or prepend retrieved text, FGA intervenes at the mathematical heart of the transformer--the pre-softmax attention scores--creating a model that cannot hallucinate when facts exist in its knowledge base. Our experiments across 1,107 technical queries spanning smartphones, laptops, and electric vehicles demonstrate a transformation from 6.3% accuracy in vanilla Llama 3.2 to 99.7% accuracy with FGA. More critically, knowledge updates occur in under one second without retraining, compared to hours for parameter editing approaches. FGA doesn't just reduce hallucination--it eliminates it entirely for verifiable facts, marking a fundamental shift from probabilistic approximation to deterministic precision in neural language generation. 

---
# AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size 

**Authors**: Guanxi Lu, Chen, Yuto Karashima, Zhican Wang, Daichi Fujiki, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.26432)  

**Abstract**: Diffusion-based large language models (dLLMs) are gaining attention for their inherent capacity for parallel decoding, offering a compelling alternative to autoregressive LLMs. Among various decoding strategies, blockwise semi-autoregressive (semi-AR) approaches are widely adopted due to their natural support for KV caching and their favorable accuracy-speed trade-off. However, this paper identifies two fundamental limitations in the conventional semi-AR decoding approach that applies a fixed block size: i) late decoding overhead, where the unmasking of high-confidence tokens outside the current block is unnecessarily delayed, and ii) premature decoding error, where low-confidence tokens inside the current block are committed too early, leading to incorrect tokens. This paper presents the first systematic investigation challenging the fixed block size assumption in semi-AR decoding. Through a statistical analysis of confidence dynamics during the denoising process, we identify a volatility band (VB) region during dLLM decoding, which encodes local semantic structure and can be used to guide adaptive block sizing. Leveraging these insights, we introduce AdaBlock-dLLM, a training-free, plug-and-play scheduler that adaptively aligns block boundaries with semantic steps by adjusting block size during runtime. Extensive experiments across diverse benchmarks show that AdaBlock-dLLM achieves up to 5.3% accuracy improvement under the same throughput budget. Beyond inference-time optimization, we hope our semantics-aware adaptive scheduling approach and confidence-based analysis will inspire future training strategies for dLLMs. 

---
# ACT: Agentic Classification Tree 

**Authors**: Vincent Grari, Tim Arni, Thibault Laugel, Sylvain Lamprier, James Zou, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2509.26433)  

**Abstract**: When used in high-stakes settings, AI systems are expected to produce decisions that are transparent, interpretable, and auditable, a requirement increasingly expected by regulations. Decision trees such as CART provide clear and verifiable rules, but they are restricted to structured tabular data and cannot operate directly on unstructured inputs such as text. In practice, large language models (LLMs) are widely used for such data, yet prompting strategies such as chain-of-thought or prompt optimization still rely on free-form reasoning, limiting their ability to ensure trustworthy behaviors. We present the Agentic Classification Tree (ACT), which extends decision-tree methodology to unstructured inputs by formulating each split as a natural-language question, refined through impurity-based evaluation and LLM feedback via TextGrad. Experiments on text benchmarks show that ACT matches or surpasses prompting-based baselines while producing transparent and interpretable decision paths. 

---
# LLM-MCoX: Large Language Model-based Multi-robot Coordinated Exploration and Search 

**Authors**: Ruiyang Wang, Haolun Tsu, David Hunt, Shaocheng Luo, Jiwoo Kim, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2509.26324)  

**Abstract**: Autonomous exploration and object search in unknown indoor environments remain challenging for multi-robot systems (MRS). Traditional approaches often rely on greedy frontier assignment strategies with limited inter-robot coordination. In this work, we introduce LLM-MCoX (LLM-based Multi-robot Coordinated Exploration and Search), a novel framework that leverages Large Language Models (LLMs) for intelligent coordination of both homogeneous and heterogeneous robot teams tasked with efficient exploration and target object search. Our approach combines real-time LiDAR scan processing for frontier cluster extraction and doorway detection with multimodal LLM reasoning (e.g., GPT-4o) to generate coordinated waypoint assignments based on shared environment maps and robot states. LLM-MCoX demonstrates superior performance compared to existing methods, including greedy and Voronoi-based planners, achieving 22.7% faster exploration times and 50% improved search efficiency in large environments with 6 robots. Notably, LLM-MCoX enables natural language-based object search capabilities, allowing human operators to provide high-level semantic guidance that traditional algorithms cannot interpret. 

---
# An Experimental Study on Generating Plausible Textual Explanations for Video Summarization 

**Authors**: Thomas Eleftheriadis, Evlampios Apostolidis, Vasileios Mezaris  

**Link**: [PDF](https://arxiv.org/pdf/2509.26225)  

**Abstract**: In this paper, we present our experimental study on generating plausible textual explanations for the outcomes of video summarization. For the needs of this study, we extend an existing framework for multigranular explanation of video summarization by integrating a SOTA Large Multimodal Model (LLaVA-OneVision) and prompting it to produce natural language descriptions of the obtained visual explanations. Following, we focus on one of the most desired characteristics for explainable AI, the plausibility of the obtained explanations that relates with their alignment with the humans' reasoning and expectations. Using the extended framework, we propose an approach for evaluating the plausibility of visual explanations by quantifying the semantic overlap between their textual descriptions and the textual descriptions of the corresponding video summaries, with the help of two methods for creating sentence embeddings (SBERT, SimCSE). Based on the extended framework and the proposed plausibility evaluation approach, we conduct an experimental study using a SOTA method (CA-SUM) and two datasets (SumMe, TVSum) for video summarization, to examine whether the more faithful explanations are also the more plausible ones, and identify the most appropriate approach for generating plausible textual explanations for video summarization. 

---
# Toward an Unbiased Collective Memory for Efficient LLM-Based Agentic 6G Cross-Domain Management 

**Authors**: Hatim Chergui, Miguel Catalan Cid, Pouria Sayyad Khodashenas, Daniel Camps Mur, Christos Verikoukis  

**Link**: [PDF](https://arxiv.org/pdf/2509.26200)  

**Abstract**: This paper introduces a novel framework for proactive cross-domain resource orchestration in 6G RAN-Edge networks, featuring large language model (LLM)-augmented agents. The system comprises specialized RAN (energy efficiency) and Edge (latency assurance) agents that engage in iterative negotiation, supported by advanced reasoning and planning capabilities. Agents dynamically interact with a digital twin (DT) to test their proposals and leverage a long-term collective memory where their joint successful and failed agreements along with the related network contexts are distilled into strategies to either follow or avoid and subsequently stored. Given that agents are subject to a plethora of cognitive distortions when retrieving those past experiences -- such as primacy, recency, confirmation and availability biases -- we propose in this work a novel unbiased memory design (A reusable mockup version of the unbiased memory source code is available for non-commercial use at this https URL). featuring (i) semantic retrieval of past strategies via Jaccard similarity; (ii) learning from failures through amplified weighting of SLA violations and mandatory inclusion of failed negotiation cases to mitigate confirmation bias; (iii) diversity enforcement to minimize availability bias and (iv) recency and primacy weighting with slow decay to counteract temporal biases. Evaluation results showcase the impact of existing biases and how the unbiased memory allows to tackle them by learning from both successful and failed strategies, either present or old, resulting in $\times 4.5$ and $\times 3.5$ reductions of unresolved negotiations compared to non-memory and vanilla memory baselines, respectively, while totally mitigating SLA violations as well as improving latency and energy saving distributions. 

---
# OWL: Geometry-Aware Spatial Reasoning for Audio Large Language Models 

**Authors**: Subrata Biswas, Mohammad Nur Hossain Khan, Bashima Islam  

**Link**: [PDF](https://arxiv.org/pdf/2509.26140)  

**Abstract**: Spatial reasoning is fundamental to auditory perception, yet current audio large language models (ALLMs) largely rely on unstructured binaural cues and single step inference. This limits both perceptual accuracy in direction and distance estimation and the capacity for interpretable reasoning. Recent work such as BAT demonstrates spatial QA with binaural audio, but its reliance on coarse categorical labels (left, right, up, down) and the absence of explicit geometric supervision constrain resolution and robustness. We introduce the $\textbf{Spatial-Acoustic Geometry Encoder (SAGE}$), a geometry-aware audio encoder that aligns binaural acoustic features with 3D spatial structure using panoramic depth images and room-impulse responses at training time, while requiring only audio at inference. Building on this representation, we present $\textbf{OWL}$, an ALLM that integrates $\textbf{SAGE}$ with a spatially grounded chain-of-thought to rationalize over direction-of-arrivals (DoA) and distance estimates. Through curriculum learning from perceptual QA to multi-step reasoning, $\textbf{OWL}$ supports o'clock-level azimuth and DoA estimation. To enable large-scale training and evaluation, we construct and release $\textbf{BiDepth}$, a dataset of over one million QA pairs combining binaural audio with panoramic depth images and room impulse responses across both in-room and out-of-room scenarios. Across two benchmark datasets, our new $\textbf{BiDepth}$ and the public SpatialSoundQA, $\textbf{OWL}$ reduces mean DoA error by $\textbf{11$^{\circ}$}$ through $\textbf{SAGE}$ and improves spatial reasoning QA accuracy by up to $\textbf{25}$\% over BAT. 

---
# R-Log: Incentivizing Log Analysis Capability in LLMs via Reasoning-based Reinforcement Learning 

**Authors**: Yilun Liu, Ziang Chen, Song Xu, Minggui He, Shimin Tao, Weibin Meng, Yuming Xie, Tao Han, Chunguang Zhao, Jingzhou Du, Daimeng Wei, Shenglin Zhang, Yongqian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.25987)  

**Abstract**: The growing complexity of log data in modern software systems has prompted the use of Large Language Models (LLMs) for automated log analysis. Current approaches typically rely on direct supervised fine-tuning (SFT) on log-label pairs. However, this exacerbates the domain discrepancy between general-purpose LLMs and specialized log data, causing overfitting. Furthermore, SFT's imbalanced loss computation often allows lengthy contexts to overwhelm critical, concise details in model answers, leading to hallucinations. To address these limitations, we propose R-Log, a novel reasoning-based paradigm that mirrors the structured, step-by-step analytical process of human engineers. This approach enhances generalizability by learning the underlying rules behind conclusions. We further employ Reinforcement Learning (RL) to optimize the model within a simulated O&M environment, thereby reducing hallucinations by directly rewarding correct outcomes. R-Log is first cold-started on a curated dataset of 2k+ reasoning trajectories, guided by 13 strategies from manual O&M practices, to establish an initial reasoning capability. This ability is then refined via RL using a joint reward function. Empirical evaluations on real-world logs show that R-Log outperforms existing methods across five log analysis tasks, particularly in unseen scenarios (by 228.05%). We also designed R-Log-fast with 5x speedup while keeping 93% of the efficacy. 

---
# Accelerating LLM Inference with Precomputed Query Storage 

**Authors**: Jay H. Park, Youngju Cho, Choungsol Lee, Moonwook Oh, Euiseong Seo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25919)  

**Abstract**: Large language model (LLM) inference often suffers from high latency, particularly in resource-constrained environments such as on-device or edge deployments. To address this challenge, we present StorInfer, a novel storage-assisted LLM inference system that accelerates response time by precomputing and storing predictable query-response pairs offline. When a user query semantically matches a precomputed query, StorInfer bypasses expensive GPU inference and instantly returns the stored response, significantly reducing latency and compute costs. To maximize coverage and effectiveness, StorInfer employs an LLM-driven generator that adaptively produces diverse and deduplicated queries based on a given knowledge base. This is achieved via two techniques: adaptive query masking, which prevents regeneration of similar queries, and adaptive sampling, which dynamically tunes generation parameters to promote semantic diversity. The resulting query-response pairs are embedded and indexed using a disk-backed vector database to enable fast, similarity-based retrieval at runtime. Using this approach, we generated 150K unique precomputed pairs (taking up to 830 MB of storage space), achieving up to 17.3% latency reduction with no loss in response quality. Our evaluation across multiple QA datasets demonstrates the practicality and scalability of storage-assisted inference, especially in scenarios with predictable query distributions. StorInfer highlights a promising direction in leveraging storage as a primary enabler for efficient, low-latency LLM deployment. 

---
# Distillation of Large Language Models via Concrete Score Matching 

**Authors**: Yeongmin Kim, Donghyeok Shin, Mina Kang, Byeonghu Na, Il-Chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2509.25837)  

**Abstract**: Large language models (LLMs) deliver remarkable performance but are costly to deploy, motivating knowledge distillation (KD) for efficient inference. Existing KD objectives typically match student and teacher probabilities via softmax, which blurs valuable logit information. While direct logit distillation (DLD) mitigates softmax smoothing, it fails to account for logit shift invariance, thereby restricting the solution space. We propose Concrete Score Distillation (CSD), a discrete score-matching objective that overcomes both softmax-induced smoothing and restrictions on the optimal solution set. We resolve the training instability and quadratic complexity of discrete score-matching in autoregressive LLMs, and the resulting CSD objective aligns relative logit differences across all vocabulary pairs between student and teacher with flexible weighting. We provide both mode-seeking and mode-covering instances within our framework and evaluate CSD on task-agnostic instruction-following and task-specific distillation using GPT-2-1.5B, OpenLLaMA-7B, and GEMMA-7B-IT. Experiments show that CSD consistently surpasses recent KD objectives, achieves favorable fidelity-diversity trade-offs, and yields complementary gains when combined with on-policy techniques, demonstrating its scalability and effectiveness for LLM distillation. 

---
# Towards Continual Expansion of Data Coverage: Automatic Text-guided Edge-case Synthesis 

**Authors**: Kyeongryeol Go  

**Link**: [PDF](https://arxiv.org/pdf/2509.26158)  

**Abstract**: The performance of deep neural networks is strongly influenced by the quality of their training data. However, mitigating dataset bias by manually curating challenging edge cases remains a major bottleneck. To address this, we propose an automated pipeline for text-guided edge-case synthesis. Our approach employs a Large Language Model, fine-tuned via preference learning, to rephrase image captions into diverse textual prompts that steer a Text-to-Image model toward generating difficult visual scenarios. Evaluated on the FishEye8K object detection benchmark, our method achieves superior robustness, surpassing both naive augmentation and manually engineered prompts. This work establishes a scalable framework that shifts data curation from manual effort to automated, targeted synthesis, offering a promising direction for developing more reliable and continuously improving AI systems. Code is available at this https URL. 

---
# More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models 

**Authors**: Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, Fabian Waschkowski, Lukas Wesemann, Peter Tu, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25848)  

**Abstract**: Reasoning has emerged as a pivotal capability in Large Language Models (LLMs). Through Reinforcement Learning (RL), typically Group Relative Policy Optimization (GRPO), these models are able to solve complex tasks such as mathematics and code generation. Building on these advances, recent research has sought to extend reasoning to Vision-Language Models (VLMs), yielding promising results across diverse visual tasks. Despite this progress, our study uncovers the dual nature of multimodal reasoning: while it substantially enhances logical inference and facilitates performance on challenging problems, it may gradually impair perceptual grounding, leading to recognition failures on otherwise basic visual questions. Through further analysis, we attribute this phenomenon to visual forgetting, wherein prolonged reasoning causes the model to increasingly disregard visual input. To address this, we propose Vision-Anchored Policy Optimization (VAPO), a simple yet effective method that explicitly steers the reasoning process toward visually grounded trajectories. Our result model, VAPO-Thinker-7B, significantly strengthens the model's reliance on visual information and achieves new state-of-the-art results on a wide range of established benchmarks. Project page: this https URL 

---
# PerQ: Efficient Evaluation of Multilingual Text Personalization Quality 

**Authors**: Dominik Macko, Andrew Pulver  

**Link**: [PDF](https://arxiv.org/pdf/2509.25903)  

**Abstract**: Since no metrics are available to evaluate specific aspects of a text, such as its personalization quality, the researchers often rely solely on large language models to meta-evaluate such texts. Due to internal biases of individual language models, it is recommended to use multiple of them for combined evaluation, which directly increases costs of such meta-evaluation. In this paper, a computationally efficient method for evaluation of personalization quality of a given text (generated by a language model) is introduced, called PerQ. A case study of comparison of generation capabilities of large and small language models shows the usability of the proposed metric in research, effectively reducing the waste of resources. 

---
# HNote: Extending YNote with Hexadecimal Encoding for Fine-Tuning LLMs in Music Modeling 

**Authors**: Hung-Ying Chu, Shao-Yu Wei, Guan-Wei Chen, Tzu-Wei Hung, ChengYang Tsai, Yu-Cheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.25694)  

**Abstract**: Recent advances in large language models (LLMs) have created new opportunities for symbolic music generation. However, existing formats such as MIDI, ABC, and MusicXML are either overly complex or structurally inconsistent, limiting their suitability for token-based learning architectures. To address these challenges, we propose HNote, a novel hexadecimal-based notation system extended from YNote, which encodes both pitch and duration within a fixed 32-unit measure framework. This design ensures alignment, reduces ambiguity, and is directly compatible with LLM architectures. We converted 12,300 Jiangnan-style songs generated from traditional folk pieces from YNote into HNote, and fine-tuned LLaMA-3.1(8B) using parameter-efficient LoRA. Experimental results show that HNote achieves a syntactic correctness rate of 82.5%, and BLEU and ROUGE evaluations demonstrate strong symbolic and structural similarity, producing stylistically coherent compositions. This study establishes HNote as an effective framework for integrating LLMs with cultural music modeling. 

---
# Free Lunch Alignment of Text-to-Image Diffusion Models without Preference Image Pairs 

**Authors**: Jia Jun Cheng Xian, Muchen Li, Haotian Yang, Xin Tao, Pengfei Wan, Leonid Sigal, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25771)  

**Abstract**: Recent advances in diffusion-based text-to-image (T2I) models have led to remarkable success in generating high-quality images from textual prompts. However, ensuring accurate alignment between the text and the generated image remains a significant challenge for state-of-the-art diffusion models. To address this, existing studies employ reinforcement learning with human feedback (RLHF) to align T2I outputs with human preferences. These methods, however, either rely directly on paired image preference data or require a learned reward function, both of which depend heavily on costly, high-quality human annotations and thus face scalability limitations. In this work, we introduce Text Preference Optimization (TPO), a framework that enables "free-lunch" alignment of T2I models, achieving alignment without the need for paired image preference data. TPO works by training the model to prefer matched prompts over mismatched prompts, which are constructed by perturbing original captions using a large language model. Our framework is general and compatible with existing preference-based algorithms. We extend both DPO and KTO to our setting, resulting in TDPO and TKTO. Quantitative and qualitative evaluations across multiple benchmarks show that our methods consistently outperform their original counterparts, delivering better human preference scores and improved text-to-image alignment. Our Open-source code is available at this https URL. 

---
# LLM-RG: Referential Grounding in Outdoor Scenarios using Large Language Models 

**Authors**: Pranav Saxena, Avigyan Bhattacharya, Ji Zhang, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25528)  

**Abstract**: Referential grounding in outdoor driving scenes is challenging due to large scene variability, many visually similar objects, and dynamic elements that complicate resolving natural-language references (e.g., "the black car on the right"). We propose LLM-RG, a hybrid pipeline that combines off-the-shelf vision-language models for fine-grained attribute extraction with large language models for symbolic reasoning. LLM-RG processes an image and a free-form referring expression by using an LLM to extract relevant object types and attributes, detecting candidate regions, generating rich visual descriptors with a VLM, and then combining these descriptors with spatial metadata into natural-language prompts that are input to an LLM for chain-of-thought reasoning to identify the referent's bounding box. Evaluated on the Talk2Car benchmark, LLM-RG yields substantial gains over both LLM and VLM-based baselines. Additionally, our ablations show that adding 3D spatial cues further improves grounding. Our results demonstrate the complementary strengths of VLMs and LLMs, applied in a zero-shot manner, for robust outdoor referential grounding. 

---
# PIPer: On-Device Environment Setup via Online Reinforcement Learning 

**Authors**: Alexander Kovrigin, Aleksandra Eliseeva, Konstantin Grotov, Egor Bogomolov, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2509.25455)  

**Abstract**: Environment setup-the process of configuring the system to work with a specific software project-represents a persistent challenge in Software Engineering (SE). Automated environment setup methods could assist developers by providing fully configured environments for arbitrary repositories without manual effort. This also helps SE researchers to scale execution-based benchmarks. However, recent studies reveal that even state-of-the-art Large Language Models (LLMs) achieve limited success in automating this task. To address this limitation, we tune a specialized model for environment setup. We combine supervised fine-tuning for generating correct Bash scripts and Reinforcement Learning with Verifiable Rewards (RLVR) to adapt it to the task of environment setup. On EnvBench-Python, our method enables Qwen3-8B (a model runnable on consumer hardware) to perform on par with larger models-Qwen3-32B and GPT-4o. The training code and model checkpoints are available online: this https URL. 

---
# Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning 

**Authors**: Zelin Tan, Hejia Geng, Mulei Zhang, Xiaohang Yu, Guancheng Wan, Yifan Zhou, Qiang He, Xiangyuan Xue, Heng Zhou, Yutao Fan, Zhongzhi Li, Zaibin Zhang, Guibin Zhang, Chen Zhang, Zhenfei Yin, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.25300)  

**Abstract**: While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on 54 experiments across diverse model sizes and training settings, we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: (1). Under a fixed computational budget, larger models trained for fewer steps consistently outperform smaller models trained for more steps. (2). Given a fixed amount of training data, larger models achieve superior sample efficiency, yielding lower loss. (3). In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. (4). These scaling behaviors are robust across both base and instruction-tuned models, which share similar learning dynamics (e.g., larger models show faster convergence) even while differing in absolute accuracy. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training. 

---
# Artificial Authority: From Machine Minds to Political Alignments. An Experimental Analysis of Democratic and Autocratic Biases in Large-Language Models 

**Authors**: Szymon Łukasik, Natalia Ożegalska-Łukasik  

**Link**: [PDF](https://arxiv.org/pdf/2509.25286)  

**Abstract**: Political beliefs vary significantly across different countries, reflecting distinct historical, cultural, and institutional contexts. These ideologies, ranging from liberal democracies to rigid autocracies, influence human societies, as well as the digital systems that are constructed within those societies. The advent of generative artificial intelligence, particularly Large Language Models (LLMs), introduces new agents in the political space-agents trained on massive corpora that replicate and proliferate socio-political assumptions. This paper analyses whether LLMs display propensities consistent with democratic or autocratic world-views. We validate this insight through experimental tests in which we experiment with the leading LLMs developed across disparate political contexts, using several existing psychometric and political orientation measures. The analysis is based on both numerical scoring and qualitative analysis of the models' responses. Findings indicate high model-to-model variability and a strong association with the political culture of the country in which the model was developed. These findings highlight the need for more detailed examination of the socio-political dimensions embedded within AI systems. 

---
# From NL2SQL to NL2GeoSQL: GeoSQL-Eval for automated evaluation of LLMs on PostGIS queries 

**Authors**: Shuyang Hou, Haoyue Jiao, Ziqi Liu, Lutong Xie, Guanyu Chen, Shaowen Wu, Xuefeng Guan, Huayi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25264)  

**Abstract**: In recent years, large language models (LLMs) have achieved remarkable progress in natural language understanding and structured query generation (NL2SQL). However, extending these advances to GeoSQL tasks in the PostGIS environment remains challenging due to the complexity of spatial functions, geometric data types, and execution semantics. Existing evaluations primarily focus on general relational databases or Google Earth Engine code generation, leaving a lack of systematic benchmarks tailored to spatial databases. To address this gap, this study introduces GeoSQL-Eval, the first end-to-end automated evaluation framework for PostGIS query generation. Built upon Webb's Depth of Knowledge (DOK) model, the framework encompasses four cognitive dimensions, five proficiency levels, and twenty task categories, providing a comprehensive assessment of model performance in terms of knowledge acquisition, syntactic generation, semantic alignment, execution accuracy, and robustness. In parallel, we developed GeoSQL-Bench, a benchmark dataset comprising 14178 questions that span three task types, 340 PostGIS functions, and 82 domain-specific databases. Leveraging this framework, we systematically evaluated 24 representative models across six categories, applying entropy-weighting and statistical analyses to reveal differences in performance, error distributions, and resource consumption patterns. Furthermore, we established a public GeoSQL-Eval leaderboard that enables global research teams to conduct ongoing testing and comparison. These contributions not only extend the boundaries of NL2SQL applications but also provide a standardized, interpretable, and scalable framework for evaluating LLM performance in spatial database contexts, offering valuable insights for model optimization and applications in geographic information science, urban studies, and spatial analysis. 

---
# Protocode: Prototype-Driven Interpretability for Code Generation in LLMs 

**Authors**: Krishna Vamshi Bodla, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25247)  

**Abstract**: Since the introduction of Large Language Models (LLMs), they have been widely adopted for various tasks such as text summarization, question answering, speech-to-text translation, and more. In recent times, the use of LLMs for code generation has gained significant attention, with tools such as Cursor and Windsurf demonstrating the ability to analyze massive code repositories and recommend relevant changes. Big tech companies have also acknowledged the growing reliance on LLMs for code generation within their codebases. Although these advances significantly improve developer productivity, increasing reliance on automated code generation can proportionally increase the risk of suboptimal solutions and insecure code. Our work focuses on automatically sampling In-Context Learning (ICL) demonstrations which can improve model performance and enhance the interpretability of the generated code. Using AST-based analysis on outputs from the MBPP test set, we identify regions of code most influenced by the chosen demonstrations. In our experiments, we show that high-quality ICL demonstrations not only make outputs easier to interpret but also yield a positive performance improvement on the pass@10 metric. Conversely, poorly chosen ICL demonstrations affected the LLM performance on the pass@10 metric negatively compared to the base model. Overall, our approach highlights the importance of efficient sampling strategies for ICL, which can affect the performance of the model on any given task. 

---
# BuildBench: Benchmarking LLM Agents on Compiling Real-World Open-Source Software 

**Authors**: Zehua Zhang, Ati Priya Bajaj, Divij Handa, Siyu Liu, Arvind S Raj, Hongkai Chen, Hulin Wang, Yibo Liu, Zion Leonahenahe Basque, Souradip Nath, Vishal Juneja, Nikhil Chapre, Yan Shoshitaishvili, Adam Doupé, Chitta Baral, Ruoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25248)  

**Abstract**: Automatically compiling open-source software (OSS) projects is a vital, labor-intensive, and complex task, which makes it a good challenge for LLM Agents. Existing methods rely on manually curated rules and workflows, which cannot adapt to OSS that requires customized configuration or environment setup. Recent attempts using Large Language Models (LLMs) used selective evaluation on a subset of highly rated OSS, a practice that underestimates the realistic challenges of OSS compilation. In practice, compilation instructions are often absent, dependencies are undocumented, and successful builds may even require patching source files or modifying build scripts. We propose a more challenging and realistic benchmark, BUILD-BENCH, comprising OSS that are more diverse in quality, scale, and characteristics. Furthermore, we propose a strong baseline LLM-based agent, OSS-BUILD-AGENT, an effective system with enhanced build instruction retrieval module that achieves state-of-the-art performance on BUILD-BENCH and is adaptable to heterogeneous OSS characteristics. We also provide detailed analysis regarding different compilation method design choices and their influence to the whole task, offering insights to guide future advances. We believe performance on BUILD-BENCH can faithfully reflect an agent's ability to tackle compilation as a complex software engineering tasks, and, as such, our benchmark will spur innovation with a significant impact on downstream applications in the fields of software development and software security. 

---
# Multi-level Diagnosis and Evaluation for Robust Tabular Feature Engineering with Large Language Models 

**Authors**: Yebin Lim, Susik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2509.25207)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise in feature engineering for tabular data, but concerns about their reliability persist, especially due to variability in generated outputs. We introduce a multi-level diagnosis and evaluation framework to assess the robustness of LLMs in feature engineering across diverse domains, focusing on the three main factors: key variables, relationships, and decision boundary values for predicting target classes. We demonstrate that the robustness of LLMs varies significantly over different datasets, and that high-quality LLM-generated features can improve few-shot prediction performance by up to 10.52%. This work opens a new direction for assessing and enhancing the reliability of LLM-driven feature engineering in various domains. 

---
# Towards Repository-Level Program Verification with Large Language Models 

**Authors**: Si Cheng Zhong, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2509.25197)  

**Abstract**: Recent advancements in large language models (LLMs) suggest great promises in code and proof generations. However, scaling automated formal verification to real-world projects requires resolving cross-module dependencies and global contexts, which are crucial challenges overlooked by existing LLM-based methods with a special focus on targeting isolated, function-level verification tasks. To systematically explore and address the significant challenges of verifying entire software repositories, we introduce RVBench, the first verification benchmark explicitly designed for repository-level evaluation, constructed from four diverse and complex open-source Verus projects.
We further introduce RagVerus, an extensible framework that synergizes retrieval-augmented generation with context-aware prompting to automate proof synthesis for multi-module repositories. RagVerus triples proof pass rates on existing benchmarks under constrained model inference budgets, and achieves a 27% relative improvement on the more challenging RVBench benchmark, demonstrating a scalable and sample-efficient verification solution. 

---
# APRIL: API Synthesis with Automatic Prompt Optimization and Reinforcement Learning 

**Authors**: Hua Zhong, Shan Jiang, Sarfraz Khurshid  

**Link**: [PDF](https://arxiv.org/pdf/2509.25196)  

**Abstract**: APIs are central to modern software development, yet composing new APIs from large libraries is difficult due to the exponential search space; traditional component-based synthesis relies on costly exploration and hand-crafted specifications. While large language models (LLMs) can generate implementations from natural language, hallucinations and limited access to up-to-date contextual information often yield incorrect code. In this paper, we present APRIL, an approach that combines LLM-based synthesis with Automatic Prompt Optimization (APO) and Reinforcement Learning from Verifiable Rewards (RLVR): APO iteratively refines prompts for a frozen model, while RLVR fine-tunes the policy toward functional correctness, producing an efficient synthesis pipeline. Evaluated on 81 real-world APIs from widely used scientific Python libraries and benchmarked against instruction-tuned but unfine-tuned LLMs guided by expert prompts, APRIL achieves substantial improvements. These results indicate that integrating APO and RLVR provides a robust, scalable path for component-based API synthesis in large libraries. 

---
# UML-CoT: Structured Reasoning and Planning with Unified Modeling Language for Robotic Room Cleaning 

**Authors**: Hongyu Chen, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22628)  

**Abstract**: Chain-of-Thought (CoT) prompting improves reasoning in large language models (LLMs), but its reliance on unstructured text limits interpretability and executability in embodied tasks. Prior work has explored structured CoTs using scene or logic graphs, yet these remain fundamentally limited: they model only low-order relations, lack constructs like inheritance or behavioral abstraction, and provide no standardized semantics for sequential or conditional planning. We propose UML-CoT, a structured reasoning and planning framework that leverages Unified Modeling Language (UML) to generate symbolic CoTs and executable action plans. UML class diagrams capture compositional object semantics, while activity diagrams model procedural control flow. Our three-stage training pipeline combines supervised fine-tuning with Group Relative Policy Optimization (GRPO), including reward learning from answer-only data. We evaluate UML-CoT on MRoom-30k, a new benchmark of cluttered room-cleaning scenarios. UML-CoT outperforms unstructured CoTs in interpretability, planning coherence, and execution success, highlighting UML as a more expressive and actionable structured reasoning formalism. 

---
# Generating High-Quality Datasets for Code Editing via Open-Source Language Models 

**Authors**: Zekai Zhang, Mingwei Liu, Zhenxi Chen, Linxi Liang, Yuxuan Chen, Guangsheng Ou, Yanlin Wang, Dan Li, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25203)  

**Abstract**: Code editing plays a vital role in software engineering, requiring developers to adjust existing code according to natural language instructions while keeping functionality intact and avoiding unnecessary modifications. However, commit-based datasets commonly used for this task are often noisy, lack diversity, and fail to reflect the style of real-world edit instructions. To address this, we introduce CanItEdit, an open-source pipeline that leverages multiple LLMs to synthesize realistic code-edit triplets. The pipeline produces both concise "lazy" instructions and more detailed "descriptive" ones, and applies filtering based on diffs and topics to guarantee data quality and variety. Using this process, we construct OCEDataFT, a curated dataset of 20K samples. Fine-tuning three advanced base models on OCEDataFT leads to significant performance boosts on the CanItEdit benchmark, with relative pass@1 improvements ranging from 4.50% to 20.79%. Notably, the resulting models achieve performance close to closed-source systems, narrowing the gap to GPT-4 to just 3.54%, without relying on proprietary resources or manual annotation. 

---
# Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking 

**Authors**: Liangliang Zhang, Zhuorui Jiang, Hongliang Chi, Haoyang Chen, Mohammed Elkoumy, Fali Wang, Qiong Wu, Zhengyi Zhou, Shirui Pan, Suhang Wang, Yao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.23495)  

**Abstract**: Knowledge Graph Question Answering (KGQA) systems rely on high-quality benchmarks to evaluate complex multi-hop reasoning. However, despite their widespread use, popular datasets such as WebQSP and CWQ suffer from critical quality issues, including inaccurate or incomplete ground-truth annotations, poorly constructed questions that are ambiguous, trivial, or unanswerable, and outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA datasets, including WebQSP and CWQ, we find that the average factual correctness rate is only 57 %. To address these issues, we introduce KGQAGen, an LLM-in-the-loop framework that systematically resolves these pitfalls. KGQAGen combines structured knowledge grounding, LLM-guided generation, and symbolic verification to produce challenging and verifiable QA instances. Using KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results demonstrate that even state-of-the-art systems struggle on this benchmark, highlighting its ability to expose limitations of existing models. Our findings advocate for more rigorous benchmark construction and position KGQAGen as a scalable framework for advancing KGQA evaluation. 

---
