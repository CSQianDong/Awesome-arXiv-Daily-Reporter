# TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning 

**Authors**: Jiaru Zou, Soumya Roy, Vinay Kumar Verma, Ziyi Wang, David Wipf, Pan Lu, Sumit Negi, James Zou, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2510.06217)  

**Abstract**: Process Reward Models (PRMs) have recently emerged as a powerful framework for enhancing the reasoning capabilities of large reasoning models (LRMs), particularly in the context of test-time scaling (TTS). However, their potential for supervising LRMs on tabular reasoning domains remains underexplored. Through detailed empirical analyses, we identify that existing PRMs, though widely adopted for supervising text-only reasoning steps, struggle with table-specific operations such as sub-table retrieval and schema interaction, leading to critical performance bottlenecks. To address this limitation, we propose TaTToo, a novel table-grounded PRM framework that (i) reasons explicitly over tabular reasoning steps and (ii) integrates tool-based verification to provide precise reward supervision. Concretely, we first design a scalable data curation pipeline that constructs over 60k high-quality step-level annotations by integrating table verification rationales with tool-based executions. Building on the collected data, we train TaTToo with a dual-stage paradigm: cold-start supervised fine-tuning to capture tool-use reasoning patterns, followed by reinforcement learning with tool-grounded reward shaping to align our model with table-based verification. We provide a comprehensive evaluation of the policy improvement induced by our newly designed PRM. Across 5 challenging tabular reasoning benchmarks covering numerical reasoning, fact-checking, and data analysis, TaTToo improves downstream policy LRMs by 30.9% at inference, surpasses strong PRM baselines such as Qwen-2.5-Math-PRM-72B with only 8B parameters, and demonstrates strong generalizability across diverse TTS strategies. 

---
# Barbarians at the Gate: How AI is Upending Systems Research 

**Authors**: Audrey Cheng, Shu Liu, Melissa Pan, Zhifei Li, Bowen Wang, Alex Krentsel, Tian Xia, Mert Cemri, Jongseok Park, Shuo Yang, Jeff Chen, Aditya Desai, Jiarong Xing, Koushik Sen, Matei Zaharia, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2510.06189)  

**Abstract**: Artificial Intelligence (AI) is starting to transform the research process as we know it by automating the discovery of new solutions. Given a task, the typical AI-driven approach is (i) to generate a set of diverse solutions, and then (ii) to verify these solutions and select one that solves the problem. Crucially, this approach assumes the existence of a reliable verifier, i.e., one that can accurately determine whether a solution solves the given problem. We argue that systems research, long focused on designing and evaluating new performance-oriented algorithms, is particularly well-suited for AI-driven solution discovery. This is because system performance problems naturally admit reliable verifiers: solutions are typically implemented in real systems or simulators, and verification reduces to running these software artifacts against predefined workloads and measuring performance. We term this approach as AI-Driven Research for Systems (ADRS), which iteratively generates, evaluates, and refines solutions. Using penEvolve, an existing open-source ADRS instance, we present case studies across diverse domains, including load balancing for multi-region cloud scheduling, Mixture-of-Experts inference, LLM-based SQL queries, and transaction scheduling. In multiple instances, ADRS discovers algorithms that outperform state-of-the-art human designs (e.g., achieving up to 5.0x runtime improvements or 50% cost reductions). We distill best practices for guiding algorithm evolution, from prompt design to evaluator construction, for existing frameworks. We then discuss the broader implications for the systems community: as AI assumes a central role in algorithm design, we argue that human researchers will increasingly focus on problem formulation and strategic guidance. Our results highlight both the disruptive potential and the urgent need to adapt systems research practices in the age of AI. 

---
# Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification 

**Authors**: Weihao Zeng, Keqing He, Chuqiao Kuang, Xiaoguang Li, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2510.06135)  

**Abstract**: Test-time compute can be scaled both sequentially and in parallel. Sequential scaling involves lengthening the generation process, while parallel scaling involves verifying and selecting among multiple candidate outputs. Combining these two strategies has led to the most powerful AI systems, such as Grok 4 Heavy and GPT-5 Pro. In certain contexts (e.g., solving Sudoku puzzles), verifying responses can be substantially easier than generating them. This property, referred to as \emph{asymmetric verification}, highlights the strong potential of test-time scaling (TTS). In this work, we study both sequential and parallel TTS of deep search agents, motivated by the intuition that verification in this setting is often much easier than generation. In experiments, we first show that sequential scaling methods, such as budget forcing, can be effective initially but soon degrade performance. Leveraging asymmetric verification, however, we are able to achieve substantial improvements by allocating only a modest amount of compute to the verifier. We conduct experiments with flagship open-source models and extend them to their ``Heavy'' variants through TTS. These deep research agents achieve gains of up to 27 absolute points on benchmarks such as BrowseComp. Remarkably, as an open-source alternative, GLM-4.5 Heavy reaches accuracy of {\bf 54.0\%} on BrowseComp and {\bf 66.0\%} on GAIA, placing it comparable to the best proprietary choices such as OpenAI Deep Research. Tongyi-DeepResearch Heavy further achieves {\bf 69.0\%} accuracy on BrowseComp, greatly surpassing the best proprietary results. 

---
# Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences 

**Authors**: Batu El, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06105)  

**Abstract**: Large language models (LLMs) are increasingly shaping how information is created and disseminated, from companies using them to craft persuasive advertisements, to election campaigns optimizing messaging to gain votes, to social media influencers boosting engagement. These settings are inherently competitive, with sellers, candidates, and influencers vying for audience approval, yet it remains poorly understood how competitive feedback loops influence LLM behavior. We show that optimizing LLMs for competitive success can inadvertently drive misalignment. Using simulated environments across these scenarios, we find that, 6.3% increase in sales is accompanied by a 14.0% rise in deceptive marketing; in elections, a 4.9% gain in vote share coincides with 22.3% more disinformation and 12.5% more populist rhetoric; and on social media, a 7.5% engagement boost comes with 188.6% more disinformation and a 16.3% increase in promotion of harmful behaviors. We call this phenomenon Moloch's Bargain for AI--competitive success achieved at the cost of alignment. These misaligned behaviors emerge even when models are explicitly instructed to remain truthful and grounded, revealing the fragility of current alignment safeguards. Our findings highlight how market-driven optimization pressures can systematically erode alignment, creating a race to the bottom, and suggest that safe deployment of AI systems will require stronger governance and carefully designed incentives to prevent competitive dynamics from undermining societal trust. 

---
# Classical AI vs. LLMs for Decision-Maker Alignment in Health Insurance Choices 

**Authors**: Mallika Mainali, Harsha Sureshbabu, Anik Sen, Christopher B. Rauch, Noah D. Reifsnyder, John Meyer, J. T. Turner, Michael W. Floyd, Matthew Molineaux, Rosina O. Weber  

**Link**: [PDF](https://arxiv.org/pdf/2510.06093)  

**Abstract**: As algorithmic decision-makers are increasingly applied to high-stakes domains, AI alignment research has evolved from a focus on universal value alignment to context-specific approaches that account for decision-maker attributes. Prior work on Decision-Maker Alignment (DMA) has explored two primary strategies: (1) classical AI methods integrating case-based reasoning, Bayesian reasoning, and naturalistic decision-making, and (2) large language model (LLM)-based methods leveraging prompt engineering. While both approaches have shown promise in limited domains such as medical triage, their generalizability to novel contexts remains underexplored. In this work, we implement a prior classical AI model and develop an LLM-based algorithmic decision-maker evaluated using a large reasoning model (GPT-5) and a non-reasoning model (GPT-4) with weighted self-consistency under a zero-shot prompting framework, as proposed in recent literature. We evaluate both approaches on a health insurance decision-making dataset annotated for three target decision-makers with varying levels of risk tolerance (0.0, 0.5, 1.0). In the experiments reported herein, classical AI and LLM-based models achieved comparable alignment with attribute-based targets, with classical AI exhibiting slightly better alignment for a moderate risk profile. The dataset and open-source implementation are publicly available at: this https URL and this https URL. 

---
# Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents 

**Authors**: Tao Zhe, Rui Liu, Fateme Memar, Xiao Luo, Wei Fan, Xinyue Ye, Zhongren Peng, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06078)  

**Abstract**: Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods. 

---
# TelecomTS: A Multi-Modal Observability Dataset for Time Series and Language Analysis 

**Authors**: Austin Feng, Andreas Varvarigos, Ioannis Panitsas, Daniela Fernandez, Jinbiao Wei, Yuwei Guo, Jialin Chen, Ali Maatouk, Leandros Tassiulas, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2510.06063)  

**Abstract**: Modern enterprises generate vast streams of time series metrics when monitoring complex systems, known as observability data. Unlike conventional time series from domains such as weather, observability data are zero-inflated, highly stochastic, and exhibit minimal temporal structure. Despite their importance, observability datasets are underrepresented in public benchmarks due to proprietary restrictions. Existing datasets are often anonymized and normalized, removing scale information and limiting their use for tasks beyond forecasting, such as anomaly detection, root-cause analysis, and multi-modal reasoning. To address this gap, we introduce TelecomTS, a large-scale observability dataset derived from a 5G telecommunications network. TelecomTS features heterogeneous, de-anonymized covariates with explicit scale information and supports a suite of downstream tasks, including anomaly detection, root-cause analysis, and a question-answering benchmark requiring multi-modal reasoning. Benchmarking state-of-the-art time series, language, and reasoning models reveals that existing approaches struggle with the abrupt, noisy, and high-variance dynamics of observability data. Our experiments also underscore the importance of preserving covariates' absolute scale, emphasizing the need for foundation time series models that natively leverage scale information for practical observability applications. 

---
# Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research 

**Authors**: Gang Liu, Yihan Zhu, Jie Chen, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06056)  

**Abstract**: Large language models hold promise as scientific assistants, yet existing agents either rely solely on algorithm evolution or on deep research in isolation, both of which face critical limitations. Pure algorithm evolution, as in AlphaEvolve, depends only on the internal knowledge of LLMs and quickly plateaus in complex domains, while pure deep research proposes ideas without validation, resulting in unrealistic or unimplementable solutions. We present DeepEvolve, an agent that integrates deep research with algorithm evolution, uniting external knowledge retrieval, cross-file code editing, and systematic debugging under a feedback-driven iterative loop. Each iteration not only proposes new hypotheses but also refines, implements, and tests them, avoiding both shallow improvements and unproductive over-refinements. Across nine benchmarks in chemistry, mathematics, biology, materials, and patents, DeepEvolve consistently improves the initial algorithm, producing executable new algorithms with sustained gains. By bridging the gap between unguided evolution and research without grounding, DeepEvolve provides a reliable framework for advancing scientific algorithm discovery. Our code is available at this https URL. 

---
# MixReasoning: Switching Modes to Think 

**Authors**: Haiquan Lu, Gongfan Fang, Xinyin Ma, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06052)  

**Abstract**: Reasoning models enhance performance by tackling problems in a step-by-step manner, decomposing them into sub-problems and exploring long chains of thought before producing an answer. However, applying extended reasoning to every step introduces substantial redundancy, as sub-problems vary widely in difficulty and complexity: a small number of pivotal steps are genuinely challenging and decisive for the final answer, while many others only involve straightforward revisions or simple computations. Therefore, a natural idea is to endow reasoning models with the ability to adaptively respond to this variation, rather than treating all steps with the same level of elaboration. To this end, we propose MixReasoning, a framework that dynamically adjusts the depth of reasoning within a single response. The resulting chain of thought then becomes a mixture of detailed reasoning on difficult steps and concise inference on simpler ones. Experiments on GSM8K, MATH-500, and AIME show that MixReasoning shortens reasoning length and substantially improves efficiency without compromising accuracy. 

---
# Refusal Falls off a Cliff: How Safety Alignment Fails in Reasoning? 

**Authors**: Qingyu Yin, Chak Tou Leong, Linyi Yang, Wenxuan Huang, Wenjie Li, Xiting Wang, Jaehong Yoon, YunXing, XingYu, Jinjin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06036)  

**Abstract**: Large reasoning models (LRMs) with multi-step reasoning capabilities have shown remarkable problem-solving abilities, yet they exhibit concerning safety vulnerabilities that remain poorly understood. In this work, we investigate why safety alignment fails in reasoning models through a mechanistic interpretability lens. Using a linear probing approach to trace refusal intentions across token positions, we discover a striking phenomenon termed as \textbf{refusal cliff}: many poorly-aligned reasoning models correctly identify harmful prompts and maintain strong refusal intentions during their thinking process, but experience a sharp drop in refusal scores at the final tokens before output generation. This suggests that these models are not inherently unsafe; rather, their refusal intentions are systematically suppressed. Through causal intervention analysis, we identify a sparse set of attention heads that negatively contribute to refusal behavior. Ablating just 3\% of these heads can reduce attack success rates below 10\%. Building on these mechanistic insights, we propose \textbf{Cliff-as-a-Judge}, a novel data selection method that identifies training examples exhibiting the largest refusal cliff to efficiently repair reasoning models' safety alignment. This approach achieves comparable safety improvements using only 1.7\% of the vanilla safety training data, demonstrating a less-is-more effect in safety alignment. 

---
# ARISE: An Adaptive Resolution-Aware Metric for Test-Time Scaling Evaluation in Large Reasoning Models 

**Authors**: Zhangyue Yin, Qiushi Sun, Zhiyuan Zeng, Zhiyuan Yu, Qipeng Guo, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06014)  

**Abstract**: Test-time scaling has emerged as a transformative paradigm for enhancing the performance of large reasoning models, enabling dynamic allocation of computational resources during inference. However, as the landscape of reasoning models rapidly expands, a critical question remains: how can we systematically compare and evaluate the test-time scaling capabilities across different models? In this paper, we introduce ARISE (Adaptive Resolution-aware Scaling Evaluation), a novel metric specifically designed to assess the test-time scaling effectiveness of large reasoning models. Unlike existing evaluation approaches, ARISE incorporates two key innovations: (1) sample-level awareness that effectively penalizes negative scaling behaviors where increased computation leads to performance degradation, and (2) a dynamic sampling mechanism that mitigates the impact of accuracy fluctuations and token count instability on the final assessment. We conduct comprehensive experiments evaluating state-of-the-art reasoning models across diverse domains including mathematical reasoning, code generation, and agentic tasks. Our results demonstrate that ARISE provides a reliable and fine-grained measurement of test-time scaling capabilities, revealing significant variations in scaling efficiency across models. Notably, our evaluation identifies Claude Opus as exhibiting superior scaling characteristics compared to other contemporary reasoning models. 

---
# Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2510.06002)  

**Abstract**: The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core limitations of standard Retrieval-Augmented Generation in the legal domain by providing a verifiable knowledge graph that models hierarchical structure, temporal evolution, and causal events of legal norms. However, a critical gap remains: how to reliably query this structured knowledge without sacrificing its deterministic properties. This paper introduces the SAT-Graph API, a formal query execution layer centered on canonical actions-atomic, composable, and auditable primitives that isolate probabilistic discovery from deterministic retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust reference resolution; (iii) point-in-time version retrieval; and (iv) auditable causal tracing. We demonstrate how planner-guided agents can decompose complex queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer architecture transforms retrieval from an opaque black box to a transparent, auditable process, directly addressing Explainable AI (XAI) requirements for high-stakes domains. 

---
# Information-Theoretic Policy Pre-Training with Empowerment 

**Authors**: Moritz Schneider, Robert Krug, Narunas Vaskevicius, Luigi Palmieri, Michael Volpp, Joschka Boedecker  

**Link**: [PDF](https://arxiv.org/pdf/2510.05996)  

**Abstract**: Empowerment, an information-theoretic measure of an agent's potential influence on its environment, has emerged as a powerful intrinsic motivation and exploration framework for reinforcement learning (RL). Besides for unsupervised RL and skill learning algorithms, the specific use of empowerment as a pre-training signal has received limited attention in the literature. We show that empowerment can be used as a pre-training signal for data-efficient downstream task adaptation. For this we extend the traditional notion of empowerment by introducing discounted empowerment, which balances the agent's control over the environment across short- and long-term horizons. Leveraging this formulation, we propose a novel pre-training paradigm that initializes policies to maximize discounted empowerment, enabling agents to acquire a robust understanding of environmental dynamics. We analyze empowerment-based pre-training for various existing RL algorithms and empirically demonstrate its potential as a general-purpose initialization strategy: empowerment-maximizing policies with long horizons are data-efficient and effective, leading to improved adaptability in downstream tasks. Our findings pave the way for future research to scale this framework to high-dimensional and complex tasks, further advancing the field of RL. 

---
# MatheMagic: Generating Dynamic Mathematics Benchmarks Robust to Memorization 

**Authors**: Dayyán O'Brien, Barry Haddow, Emily Allaway, Pinzhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05962)  

**Abstract**: Conducting contamination-free evaluation of mathematical capabilities can be difficult for two reasons: models may memorize a test set once it is made public, and current mathematical benchmarks are prone to overfitting due to having limited diversity of symbols and rules, coupled with closed-ended answers. This paper proposes a method to leverage these shortcomings as useful features to a construct dynamic, counterfactual benchmark, which can be used to both reveal overfitting and measure true reasoning. We demonstrate this via MatheMagic, which generates math test instances with the interpretations of numbers and operators altered, yet has automatically verifiable answers. Test instances are randomly seeded and constructed at test time to evaluate a model's induction or deduction capability, offering stability, extensibility, comparability, and robustness to overfitting. Our experiments find that models solve deduction more easily than induction, but they revert to standard math. Further analysis reveals that math-adapted models fail to exhibit a general "skill" of reasoning, and fine-tuning on induction tasks generalizes poorly. 

---
# Training-Free Time Series Classification via In-Context Reasoning with LLM Agents 

**Authors**: Songyuan Sui, Zihang Xu, Yu-Neng Chuang, Kwei-Herng Lai, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05950)  

**Abstract**: Time series classification (TSC) spans diverse application scenarios, yet labeled data are often scarce, making task-specific training costly and inflexible. Recent reasoning-oriented large language models (LLMs) show promise in understanding temporal patterns, but purely zero-shot usage remains suboptimal. We propose FETA, a multi-agent framework for training-free TSC via exemplar-based in-context reasoning. FETA decomposes a multivariate series into channel-wise subproblems, retrieves a few structurally similar labeled examples for each channel, and leverages a reasoning LLM to compare the query against these exemplars, producing channel-level labels with self-assessed confidences; a confidence-weighted aggregator then fuses all channel decisions. This design eliminates the need for pretraining or fine-tuning, improves efficiency by pruning irrelevant channels and controlling input length, and enhances interpretability through exemplar grounding and confidence estimation. On nine challenging UEA datasets, FETA achieves strong accuracy under a fully training-free setting, surpassing multiple trained baselines. These results demonstrate that a multi-agent in-context reasoning framework can transform LLMs into competitive, plug-and-play TSC solvers without any parameter training. The code is available at this https URL. 

---
# Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies 

**Authors**: Aksel Joonas Reedi, Corentin Léger, Julien Pourcel, Loris Gaven, Perrine Charriau, Guillaume Pourcel  

**Link**: [PDF](https://arxiv.org/pdf/2510.05909)  

**Abstract**: Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization. 

---
# Towards Label-Free Biological Reasoning Synthetic Dataset Creation via Uncertainty Filtering 

**Authors**: Josefa Lia Stoisser, Lawrence Phillips, Aditya Misra, Tom A. Lamb, Philip Torr, Marc Boubnovski Martell, Julien Fauqueur, Kaspar Märtens  

**Link**: [PDF](https://arxiv.org/pdf/2510.05871)  

**Abstract**: Synthetic chain-of-thought (CoT) traces are widely used to train large reasoning models (LRMs), improving generalization by providing step-level supervision. Yet most approaches require ground-truth labels to seed or filter these traces - an expensive bottleneck in domains like biology where wet-lab data are scarce. We propose a label-free alternative: uncertainty-based filtering, which uses a model's own confidence - quantified through established uncertainty metrics like self-consistency and predictive perplexity - as a substitute for external labels. We sample multiple reasoning traces and retain only low-uncertainty subsets. Applied to biological perturbation prediction, a domain where wet-lab labels are especially costly, we show that the filtered subset has higher accuracy, and that supervised fine-tuning (SFT) on uncertainty-filtered data outperforms unfiltered synthetic data, narrows the gap to ground-truth training, and surpasses strong LRM baselines. Ablations show that per-class filtering corrects for class-specific uncertainty scales and that hybrid uncertainty metrics yield higher-quality datasets. Our results suggest that model-internal confidence is a powerful signal for efficient reasoning dataset creation, enabling LRMs in domains where supervision is expensive. 

---
# The Safety Challenge of World Models for Embodied AI Agents: A Review 

**Authors**: Lorenzo Baraldi, Zifan Zeng, Chongzhe Zhang, Aradhana Nayak, Hongbo Zhu, Feng Liu, Qunli Zhang, Peng Wang, Shiming Liu, Zheng Hu, Angelo Cangelosi, Lorenzo Baraldi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05865)  

**Abstract**: The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results. 

---
# ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming 

**Authors**: Weichun Shi, Minghao Liu, Wanting Zhang, Langchen Shi, Fuqi Jia, Feifei Ma, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05774)  

**Abstract**: Constraint programming (CP) is a crucial technology for solving real-world constraint optimization problems (COPs), with the advantages of rich modeling semantics and high solving efficiency. Using large language models (LLMs) to generate formal modeling automatically for COPs is becoming a promising approach, which aims to build trustworthy neuro-symbolic AI with the help of symbolic solvers. However, CP has received less attention compared to works based on operations research (OR) models. We introduce ConstraintLLM, the first LLM specifically designed for CP modeling, which is trained on an open-source LLM with multi-instruction supervised fine-tuning. We propose the Constraint-Aware Retrieval Module (CARM) to increase the in-context learning capabilities, which is integrated in a Tree-of-Thoughts (ToT) framework with guided self-correction mechanism. Moreover, we construct and release IndusCP, the first industrial-level benchmark for CP modeling, which contains 140 challenging tasks from various domains. Our experiments demonstrate that ConstraintLLM achieves state-of-the-art solving accuracy across multiple benchmarks and outperforms the baselines by 2x on the new IndusCP benchmark. Code and data are available at: this https URL. 

---
# RareAgent: Self-Evolving Reasoning for Drug Repurposing in Rare Diseases 

**Authors**: Lang Qin, Zijian Gan, Xu Cao, Pengcheng Jiang, Yankai Jiang, Jiawei Han, Kaishun Wu, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05764)  

**Abstract**: Computational drug repurposing for rare diseases is especially challenging when no prior associations exist between drugs and target diseases. Therefore, knowledge graph completion and message-passing GNNs have little reliable signal to learn and propagate, resulting in poor performance. We present RareAgent, a self-evolving multi-agent system that reframes this task from passive pattern recognition to active evidence-seeking reasoning. RareAgent organizes task-specific adversarial debates in which agents dynamically construct evidence graphs from diverse perspectives to support, refute, or entail hypotheses. The reasoning strategies are analyzed post hoc in a self-evolutionary loop, producing textual feedback that refines agent policies, while successful reasoning paths are distilled into transferable heuristics to accelerate future investigations. Comprehensive evaluations reveal that RareAgent improves the indication AUPRC by 18.1% over reasoning baselines and provides a transparent reasoning chain consistent with clinical evidence. 

---
# Early Multimodal Prediction of Cross-Lingual Meme Virality on Reddit: A Time-Window Analysis 

**Authors**: Sedat Dogan, Nina Dethlefs, Debarati Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2510.05761)  

**Abstract**: Predicting the virality of online content remains challenging, especially for culturally complex, fast-evolving memes. This study investigates the feasibility of early prediction of meme virality using a large-scale, cross-lingual dataset from 25 diverse Reddit communities. We propose a robust, data-driven method to define virality based on a hybrid engagement score, learning a percentile-based threshold from a chronologically held-out training set to prevent data leakage. We evaluated a suite of models, including Logistic Regression, XGBoost, and a Multi-layer Perceptron (MLP), with a comprehensive, multimodal feature set across increasing time windows (30-420 min). Crucially, useful signals emerge quickly: our best-performing model, XGBoost, achieves a PR-AUC $>$ 0.52 in just 30 minutes. Our analysis reveals a clear "evidentiary transition," in which the importance of the feature dynamically shifts from the static context to the temporal dynamics as a meme gains traction. This work establishes a robust, interpretable, and practical benchmark for early virality prediction in scenarios where full diffusion cascade data is unavailable, contributing a novel cross-lingual dataset and a methodologically sound definition of virality. To our knowledge, this study is the first to combine time series data with static content and network features to predict early meme virality. 

---
# Uncertainty assessment in satellite-based greenhouse gas emissions estimates using emulated atmospheric transport 

**Authors**: Jeffrey N. Clark, Elena Fillola, Nawid Keshtmand, Raul Santos-Rodriguez, Matthew Rigby  

**Link**: [PDF](https://arxiv.org/pdf/2510.05751)  

**Abstract**: Monitoring greenhouse gas emissions and evaluating national inventories require efficient, scalable, and reliable inference methods. Top-down approaches, combined with recent advances in satellite observations, provide new opportunities to evaluate emissions at continental and global scales. However, transport models used in these methods remain a key source of uncertainty: they are computationally expensive to run at scale, and their uncertainty is difficult to characterise. Artificial intelligence offers a dual opportunity to accelerate transport simulations and to quantify their associated uncertainty.
We present an ensemble-based pipeline for estimating atmospheric transport "footprints", greenhouse gas mole fraction measurements, and their uncertainties using a graph neural network emulator of a Lagrangian Particle Dispersion Model (LPDM). The approach is demonstrated with GOSAT (Greenhouse Gases Observing Satellite) observations for Brazil in 2016. The emulator achieved a ~1000x speed-up over the NAME LPDM, while reproducing large-scale footprint structures. Ensembles were calculated to quantify absolute and relative uncertainty, revealing spatial correlations with prediction error. The results show that ensemble spread highlights low-confidence spatial and temporal predictions for both atmospheric transport footprints and methane mole fractions.
While demonstrated here for an LPDM emulator, the approach could be applied more generally to atmospheric transport models, supporting uncertainty-aware greenhouse gas inversion systems and improving the robustness of satellite-based emissions monitoring. With further development, ensemble-based emulators could also help explore systematic LPDM errors, offering a computationally efficient pathway towards a more comprehensive uncertainty budget in greenhouse gas flux estimates. 

---
# ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems 

**Authors**: Bohan Yao, Shiva Krishna Reddy Malay, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2510.05746)  

**Abstract**: Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization. 

---
# Artificially intelligent agents in the social and behavioral sciences: A history and outlook 

**Authors**: Petter Holme, Milena Tsvetkova  

**Link**: [PDF](https://arxiv.org/pdf/2510.05743)  

**Abstract**: We review the historical development and current trends of artificially intelligent agents (agentic AI) in the social and behavioral sciences: from the first programmable computers, and social simulations soon thereafter, to today's experiments with large language models. This overview emphasizes the role of AI in the scientific process and the changes brought about, both through technological advancements and the broader evolution of science from around 1950 to the present. Some of the specific points we cover include: the challenges of presenting the first social simulation studies to a world unaware of computers, the rise of social systems science, intelligent game theoretic agents, the age of big data and the epistemic upheaval in its wake, and the current enthusiasm around applications of generative AI, and many other topics. A pervasive theme is how deeply entwined we are with the technologies we use to understand ourselves. 

---
# Syn-Diag: An LLM-based Synergistic Framework for Generalizable Few-shot Fault Diagnosis on the Edge 

**Authors**: Zijun Jia, Shuang Liang, Jinsong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05733)  

**Abstract**: Industrial fault diagnosis faces the dual challenges of data scarcity and the difficulty of deploying large AI models in resource-constrained environments. This paper introduces Syn-Diag, a novel cloud-edge synergistic framework that leverages Large Language Models to overcome these limitations in few-shot fault diagnosis. Syn-Diag is built on a three-tiered mechanism: 1) Visual-Semantic Synergy, which aligns signal features with the LLM's semantic space through cross-modal pre-training; 2) Content-Aware Reasoning, which dynamically constructs contextual prompts to enhance diagnostic accuracy with limited samples; and 3) Cloud-Edge Synergy, which uses knowledge distillation to create a lightweight, efficient edge model capable of online updates via a shared decision space. Extensive experiments on six datasets covering different CWRU and SEU working conditions show that Syn-Diag significantly outperforms existing methods, especially in 1-shot and cross-condition scenarios. The edge model achieves performance comparable to the cloud version while reducing model size by 83% and latency by 50%, offering a practical, robust, and deployable paradigm for modern intelligent diagnostics. 

---
# Joint Communication Scheduling and Velocity Control for Multi-UAV-Assisted Post-Disaster Monitoring: An Attention-Based In-Context Learning Approach 

**Authors**: Yousef Emami, Seyedsina Nabavirazavi, Jingjing Zheng, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2510.05698)  

**Abstract**: Recently, Unmanned Aerial Vehicles (UAVs) are increasingly being investigated to collect sensory data in post-disaster monitoring scenarios, such as tsunamis, where early actions are critical to limit coastal damage. A major challenge is to design the data collection schedules and flight velocities, as unfavorable schedules and velocities can lead to transmission errors and buffer overflows of the ground sensors, ultimately resulting in significant packet loss. Meanwhile, online Deep Reinforcement Learning (DRL) solutions have a complex training process and a mismatch between simulation and reality that does not meet the urgent requirements of tsunami monitoring. Recent advances in Large Language Models (LLMs) offer a compelling alternative. With their strong reasoning and generalization capabilities, LLMs can adapt to new tasks through In-Context Learning (ICL), which enables task adaptation through natural language prompts and example-based guidance without retraining. However, LLM models have input data limitations and thus require customized approaches. In this paper, a joint optimization of data collection schedules and velocities control for multiple UAVs is proposed to minimize data loss. The battery level of the ground sensors, the length of the queues, and the channel conditions, as well as the trajectories of the UAVs, are taken into account. Attention-Based In-Context Learning for Velocity Control and Data Collection Schedule (AIC-VDS) is proposed as an alternative to DRL in emergencies. The simulation results show that the proposed AIC-VDS outperforms both the Deep-Q-Network (DQN) and maximum channel gain baselines. 

---
# D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI 

**Authors**: Suwhan Choi, Jaeyoon Jung, Haebin Seong, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu, Yunsung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.05684)  

**Abstract**: Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at this https URL 

---
# Large Language Model-Based Uncertainty-Adjusted Label Extraction for Artificial Intelligence Model Development in Upper Extremity Radiography 

**Authors**: Hanna Kreutzer, Anne-Sophie Caselitz, Thomas Dratsch, Daniel Pinto dos Santos, Christiane Kuhl, Daniel Truhn, Sven Nebelung  

**Link**: [PDF](https://arxiv.org/pdf/2510.05664)  

**Abstract**: Objectives: To evaluate GPT-4o's ability to extract diagnostic labels (with uncertainty) from free-text radiology reports and to test how these labels affect multi-label image classification of musculoskeletal radiographs. Methods: This retrospective study included radiography series of the clavicle (n=1,170), elbow (n=3,755), and thumb (n=1,978). After anonymization, GPT-4o filled out structured templates by indicating imaging findings as present ("true"), absent ("false"), or "uncertain." To assess the impact of label uncertainty, "uncertain" labels of the training and validation sets were automatically reassigned to "true" (inclusive) or "false" (exclusive). Label-image-pairs were used for multi-label classification using ResNet50. Label extraction accuracy was manually verified on internal (clavicle: n=233, elbow: n=745, thumb: n=393) and external test sets (n=300 for each). Performance was assessed using macro-averaged receiver operating characteristic (ROC) area under the curve (AUC), precision recall curves, sensitivity, specificity, and accuracy. AUCs were compared with the DeLong test. Results: Automatic extraction was correct in 98.6% (60,618 of 61,488) of labels in the test sets. Across anatomic regions, label-based model training yielded competitive performance measured by macro-averaged AUC values for inclusive (e.g., elbow: AUC=0.80 [range, 0.62-0.87]) and exclusive models (elbow: AUC=0.80 [range, 0.61-0.88]). Models generalized well on external datasets (elbow [inclusive]: AUC=0.79 [range, 0.61-0.87]; elbow [exclusive]: AUC=0.79 [range, 0.63-0.89]). No significant differences were observed across labeling strategies or datasets (p>=0.15). Conclusion: GPT-4o extracted labels from radiologic reports to train competitive multi-label classification models with high accuracy. Detected uncertainty in the radiologic reports did not influence the performance of these models. 

---
# From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Future Research Directions 

**Authors**: Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Dusit Niyato, Geng Sun, Xianbin Wang, Shiwen Mao, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2510.05596)  

**Abstract**: Self-evolving agentic artificial intelligence (AI) offers a new paradigm for future wireless systems by enabling autonomous agents to continually adapt and improve without human intervention. Unlike static AI models, self-evolving agents embed an autonomous evolution cycle that updates models, tools, and workflows in response to environmental dynamics. This paper presents a comprehensive overview of self-evolving agentic AI, highlighting its layered architecture, life cycle, and key techniques, including tool intelligence, workflow optimization, self-reflection, and evolutionary learning. We further propose a multi-agent cooperative self-evolving agentic AI framework, where multiple large language models (LLMs) are assigned role-specialized prompts under the coordination of a supervisor agent. Through structured dialogue, iterative feedback, and systematic validation, the system autonomously executes the entire life cycle without human intervention. A case study on antenna evolution in low-altitude wireless networks (LAWNs) demonstrates how the framework autonomously upgrades fixed antenna optimization into movable antenna optimization. Experimental results show that the proposed self-evolving agentic AI autonomously improves beam gain and restores degraded performance by up to 52.02%, consistently surpassing the fixed baseline with little to no human intervention and validating its adaptability and robustness for next-generation wireless intelligence. 

---
# In-the-Flow Agentic System Optimization for Effective Planning and Tool Use 

**Authors**: Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05592)  

**Abstract**: Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multi-turn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AgentFlow with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns. 

---
# MetaVLA: Unified Meta Co-training For Efficient Embodied Adaption 

**Authors**: Chen Li, Zhantao Yang, Han Zhang, Fangyi Chen, Chenchen Zhu, Anudeepsekhar Bolimera, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2510.05580)  

**Abstract**: Vision-Language-Action (VLA) models show promise in embodied reasoning, yet remain far from true generalists-they often require task-specific fine-tuning, and generalize poorly to unseen tasks. We propose MetaVLA, a unified, backbone-agnostic post-training framework for efficient and scalable alignment. MetaVLA introduces Context-Aware Meta Co-Training, which consolidates diverse target tasks into a single fine-tuning stage while leveraging structurally diverse auxiliary tasks to improve in-domain generalization. Unlike naive multi-task SFT, MetaVLA integrates a lightweight meta-learning mechanism-derived from Attentive Neural Processes-to enable rapid adaptation from diverse contexts with minimal architectural change or inference overhead. On the LIBERO benchmark, MetaVLA with six auxiliary tasks outperforms OpenVLA by up to 8.0% on long-horizon tasks, reduces training steps from 240K to 75K, and cuts GPU time by ~76%. These results show that scalable, low-resource post-training is achievable-paving the way toward general-purpose embodied agents. Code will be available. 

---
# Decade-long Emission Forecasting with an Ensemble Model in Taiwan 

**Authors**: Gordon Hung, Salinna Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2510.05548)  

**Abstract**: Taiwan's high population and heavy dependence on fossil fuels have led to severe air pollution, with the most prevalent greenhouse gas being carbon dioxide (CO2). There-fore, this study presents a reproducible and comprehensive case study comparing 21 of the most commonly employed time series models in forecasting emissions, analyzing both univariate and multivariate approaches. Among these, Feedforward Neural Network (FFNN), Support Vector Machine (SVM), and Random Forest Regressor (RFR) achieved the best performances. To further enhance robustness, the top performers were integrated with Linear Regression through a custom stacked generalization en-semble technique. Our proposed ensemble model achieved an SMAPE of 1.407 with no signs of overfitting. Finally, this research provides an accurate decade-long emission projection that will assist policymakers in making more data-driven decisions. 

---
# Vul-R2: A Reasoning LLM for Automated Vulnerability Repair 

**Authors**: Xin-Cheng Wen, Zirui Lin, Yijun Yang, Cuiyun Gao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.05480)  

**Abstract**: The exponential increase in software vulnerabilities has created an urgent need for automatic vulnerability repair (AVR) solutions. Recent research has formulated AVR as a sequence generation problem and has leveraged large language models (LLMs) to address this problem. Typically, these approaches prompt or fine-tune LLMs to generate repairs for vulnerabilities directly. Although these methods show state-of-the-art performance, they face the following challenges: (1) Lack of high-quality, vulnerability-related reasoning data. Current approaches primarily rely on foundation models that mainly encode general programming knowledge. Without vulnerability-related reasoning data, they tend to fail to capture the diverse vulnerability repair patterns. (2) Hard to verify the intermediate vulnerability repair process during LLM training. Existing reinforcement learning methods often leverage intermediate execution feedback from the environment (e.g., sandbox-based execution results) to guide reinforcement learning training. In contrast, the vulnerability repair process generally lacks such intermediate, verifiable feedback, which poses additional challenges for model training. 

---
# VAL-Bench: Measuring Value Alignment in Language Models 

**Authors**: Aman Gupta, Denny O'Shea, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2510.05465)  

**Abstract**: Large language models (LLMs) are increasingly used for tasks where outputs shape human decisions, so it is critical to test whether their responses reflect consistent human values. Existing benchmarks mostly track refusals or predefined safety violations, but these only check rule compliance and do not reveal whether a model upholds a coherent value system when facing controversial real-world issues. We introduce the \textbf{V}alue \textbf{AL}ignment \textbf{Bench}mark (\textbf{VAL-Bench}), which evaluates whether models maintain a stable value stance across paired prompts that frame opposing sides of public debates. VAL-Bench consists of 115K such pairs from Wikipedia's controversial sections. A well-aligned model should express similar underlying views regardless of framing, which we measure using an LLM-as-judge to score agreement or divergence between paired responses. Applied across leading open- and closed-source models, the benchmark reveals large variation in alignment and highlights trade-offs between safety strategies (e.g., refusals) and more expressive value systems. By providing a scalable, reproducible benchmark, VAL-Bench enables systematic comparison of how reliably LLMs embody human values. 

---
# Do Code Models Suffer from the Dunning-Kruger Effect? 

**Authors**: Mukul Singh, Somya Chatterjee, Arjun Radhakrishna, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2510.05457)  

**Abstract**: As artificial intelligence systems increasingly collaborate with humans in creative and technical domains, questions arise about the cognitive boundaries and biases that shape our shared agency. This paper investigates the Dunning-Kruger Effect (DKE), the tendency for those with limited competence to overestimate their abilities in state-of-the-art LLMs in coding tasks. By analyzing model confidence and performance across a diverse set of programming languages, we reveal that AI models mirror human patterns of overconfidence, especially in unfamiliar or low-resource domains. Our experiments demonstrate that less competent models and those operating in rare programming languages exhibit stronger DKE-like bias, suggesting that the strength of the bias is proportionate to the competence of the models. 

---
# NASP-T: A Fuzzy Neuro-Symbolic Transformer for Logic-Constrained Aviation Safety Report Classification 

**Authors**: Fadi Al Machot, Fidaa Al Machot  

**Link**: [PDF](https://arxiv.org/pdf/2510.05451)  

**Abstract**: Deep transformer models excel at multi-label text classification but often violate domain logic that experts consider essential, an issue of particular concern in safety-critical applications. We propose a hybrid neuro-symbolic framework that integrates Answer Set Programming (ASP) with transformer-based learning on the Aviation Safety Reporting System (ASRS) corpus. Domain knowledge is formalized as weighted ASP rules and validated using the Clingo solver. These rules are incorporated in two complementary ways: (i) as rule-based data augmentation, generating logically consistent synthetic samples that improve label diversity and coverage; and (ii) as a fuzzy-logic regularizer, enforcing rule satisfaction in a differentiable form during fine-tuning. This design preserves the interpretability of symbolic reasoning while leveraging the scalability of deep neural architectures. We further tune per-class thresholds and report both standard classification metrics and logic-consistency rates. Compared to a strong Binary Cross-Entropy (BCE) baseline, our approach improves micro- and macro-F1 scores and achieves up to an 86% reduction in rule violations on the ASRS test set. To the best of our knowledge, this constitutes the first large-scale neuro-symbolic application to ASRS reports that unifies ASP-based reasoning, rule-driven augmentation, and differentiable transformer training for trustworthy, safety-critical NLP. 

---
# AInstein: Assessing the Feasibility of AI-Generated Approaches to Research Problems 

**Authors**: Shambhavi Mishra, Gaurav Sahu, Marco Pedersoli, Laurent Charlin, Jose Dolz, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2510.05432)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet it remains unclear whether such success reflects genuine reasoning or sophisticated recall. We introduce AInstein, a framework for testing whether LLMs can generate valid solutions to AI research problems using only their pretrained parametric knowledge -- without domain-specific fine-tuning, retrieval augmentation, or other external aids. Our approach extracts distilled problem statements from high-quality ICLR 2025 submissions, then tasks specialized solver agents with proposing and refining technical solutions through iterative critique loops, mimicking the cycles of proposal, review, and revision central to scientific inquiry. We evaluate AInstein on 1,214 ICLR papers stratified by acceptance tier (Oral, Spotlight, Poster), using an LLM-as-a-judge paradigm guided by a structured rubric, complemented by targeted manual checks. Performance is assessed with three metrics: Success Rate (does the solution address the problem?), Rediscovery (does it align with human-proposed methods?), and Novelty (does it yield valid, original approaches?). Our results reveal that while LLMs can rediscover feasible solutions and occasionally propose creative alternatives, their problem-solving ability remains fragile and highly sensitive to framing. These findings provide the first large-scale evidence on the extent to which LLMs can act as autonomous scientific problem-solvers, highlighting both their latent potential and their current limitations. 

---
# Teacher-Student Guided Inverse Modeling for Steel Final Hardness Estimation 

**Authors**: Ahmad Alsheikh, Andreas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2510.05402)  

**Abstract**: Predicting the final hardness of steel after heat treatment is a challenging regression task due to the many-to-one nature of the process -- different combinations of input parameters (such as temperature, duration, and chemical composition) can result in the same hardness value. This ambiguity makes the inverse problem, estimating input parameters from a desired hardness, particularly difficult. In this work, we propose a novel solution using a Teacher-Student learning framework. First, a forward model (Teacher) is trained to predict final hardness from 13 metallurgical input features. Then, a backward model (Student) is trained to infer plausible input configurations from a target hardness value. The Student is optimized by leveraging feedback from the Teacher in an iterative, supervised loop. We evaluate our method on a publicly available tempered steel dataset and compare it against baseline regression and reinforcement learning models. Results show that our Teacher-Student framework not only achieves higher inverse prediction accuracy but also requires significantly less computational time, demonstrating its effectiveness and efficiency for inverse process modeling in materials science. 

---
# What Do You Mean? Exploring How Humans and AI Interact with Symbols and Meanings in Their Interactions 

**Authors**: Reza Habibi, Seung Wan Ha, Zhiyu Lin, Atieh Kashani, Ala Shafia, Lakshana Lakshmanarajan, Chia-Fang Chung, Magy Seif El-Nasr  

**Link**: [PDF](https://arxiv.org/pdf/2510.05378)  

**Abstract**: Meaningful human-AI collaboration requires more than processing language, it demands a better understanding of symbols and their constructed meanings. While humans naturally interpret symbols through social interaction, AI systems treat them as patterns with compressed meanings, missing the dynamic meanings that emerge through conversation. Drawing on symbolic interactionism theory, we conducted two studies (N=37) investigated how humans and AI interact with symbols and co-construct their meanings. When AI introduced conflicting meanings and symbols in social contexts, 63% of participants reshaped their definitions. This suggests that conflicts in symbols and meanings prompt reflection and redefinition, allowing both participants and AI to have a better shared understanding of meanings and symbols. This work reveals that shared understanding emerges not from agreement but from the reciprocal exchange and reinterpretation of symbols, suggesting new paradigms for human-AI interaction design. 

---
# MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding Exemplars as Soft Prompts 

**Authors**: Abhinav Jain, Xinyu Yao, Thomas Reps, Christopher Jermaine  

**Link**: [PDF](https://arxiv.org/pdf/2510.05363)  

**Abstract**: Adapting Foundation Models to new domains with limited training data is challenging and computationally expensive. While prior work has demonstrated the effectiveness of using domain-specific exemplars as in-context demonstrations, we investigate whether representing exemplars purely as text is the most efficient, effective, and stable approach. We explore an alternative: representing exemplars as soft prompts with an exemplar order invariant model architecture. To this end, we introduce Multi-Head Attention Retrieval-Augmented Generation (MHA-RAG), a framework with the number of attention heads serving as a simple hyperparameter to control soft prompt-generation across different tasks. Across multiple question-answering benchmarks and model scales, MHA-RAG achieves a 20-point performance gain over standard RAG, while cutting inference costs by a factor of 10X GFLOPs-delivering both higher accuracy and greater efficiency, invariant to exemplar order. 

---
# Integrating Bayesian methods with neural network--based model predictive control: a review 

**Authors**: Asli Karacelik  

**Link**: [PDF](https://arxiv.org/pdf/2510.05338)  

**Abstract**: In this review, we assess the use of Bayesian methods in model predictive control (MPC), focusing on neural-network-based modeling, control design, and uncertainty quantification. We systematically analyze individual studies and how they are implemented in practice. While Bayesian approaches are increasingly adopted to capture and propagate uncertainty in MPC, reported gains in performance and robustness remain fragmented, with inconsistent baselines and limited reliability analyses. We therefore argue for standardized benchmarks, ablation studies, and transparent reporting to rigorously determine the effectiveness of Bayesian techniques for MPC. 

---
# Biomedical reasoning in action: Multi-agent System for Auditable Biomedical Evidence Synthesis 

**Authors**: Oskar Wysocki, Magdalena Wysocka, Mauricio Jacobo, Harriet Unsworth, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2510.05335)  

**Abstract**: We present M-Reason, a demonstration system for transparent, agent-based reasoning and evidence integration in the biomedical domain, with a focus on cancer research. M-Reason leverages recent advances in large language models (LLMs) and modular agent orchestration to automate evidence retrieval, appraisal, and synthesis across diverse biomedical data sources. Each agent specializes in a specific evidence stream, enabling parallel processing and fine-grained analysis. The system emphasizes explainability, structured reporting, and user auditability, providing complete traceability from source evidence to final conclusions. We discuss critical tradeoffs between agent specialization, system complexity, and resource usage, as well as the integration of deterministic code for validation. An open, interactive user interface allows researchers to directly observe, explore and evaluate the multi-agent workflow. Our evaluation demonstrates substantial gains in efficiency and output consistency, highlighting M-Reason's potential as both a practical tool for evidence synthesis and a testbed for robust multi-agent LLM systems in scientific research, available at this https URL. 

---
# BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation for Large Language Models via Lens of Dynamic Interactions 

**Authors**: Nan Huo, Xiaohan Xu, Jinyang Li, Per Jacobsson, Shipei Lin, Bowen Qin, Binyuan Hui, Xiaolong Li, Ge Qu, Shuzheng Si, Linheng Han, Edward Alexander, Xintong Zhu, Rui Qin, Ruihan Yu, Yiyao Jin, Feige Zhou, Weihao Zhong, Yun Chen, Hongyu Liu, Chenhao Ma, Fatma Ozcan, Yannis Papakonstantinou, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05318)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on single-turn text-to-SQL tasks, but real-world database applications predominantly require multi-turn interactions to handle ambiguous queries, execution errors, and evolving user requirements. Existing multi-turn benchmarks fall short by treating conversation histories as static context or limiting evaluation to read-only operations, failing to reflect production-grade database assistant challenges. We introduce BIRD-INTERACT, a benchmark that restores this realism through: (1) a comprehensive interaction environment coupling each database with a hierarchical knowledge base, metadata files, and a function-driven user simulator, enabling models to solicit clarifications, retrieve knowledge, and recover from errors without human supervision; (2) two evaluation settings consisting of a pre-defined conversational protocol (c-Interact) and an open-ended agentic setting (a-Interact) where models autonomously decide when to query the user simulator or explore the environment; (3) a challenging task suite covering the full CRUD spectrum for business-intelligence and operational use cases, guarded by executable test cases. Each task features ambiguous and follow-up sub-tasks requiring dynamic interaction. The suite comprises BIRD-INTERACT-FULL (600 tasks, up to 11,796 interactions) for comprehensive performance assessment, and BIRD-INTERACT-LITE (300 tasks with simplified databases) for detailed behavioral analysis and rapid method development. Our empirical results highlight BIRD-INTERACT's difficulty: GPT-5 completes only 8.67% of tasks in c-Interact and 17.00% in a-Interact. Analysis via memory grafting and Interaction Test-time Scaling validates the importance of effective interaction for complex, dynamic text-to-SQL tasks. 

---
# Beyond Monolithic Rewards: A Hybrid and Multi-Aspect Reward Optimization for MLLM Alignment 

**Authors**: Radha Gulhane, Sathish Reddy Indurthi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05283)  

**Abstract**: Aligning multimodal large language models (MLLMs) with human preferences often relies on single-signal, model-based reward methods. Such monolithic rewards often lack confidence calibration across domain-specific tasks, fail to capture diverse aspects of human preferences, and require extensive data annotation and reward model training. In this work, we propose a hybrid reward modeling framework that integrates complementary reward paradigms: (i) model-based rewards, where a learned reward model predicts scalar or vector scores from synthetic and human feedback, and (ii) rule-based rewards, where domain-specific heuristics provide explicit correctness signals with confidence. Beyond accuracy, we further incorporate multi-aspect rewards to enforce instruction adherence and introduce a generalized length-penalty reward to stabilize training and improve performance. The proposed framework provides a flexible and effective approach to aligning MLLMs through reinforcement learning policy optimization. Our experiments show consistent improvements across different multimodal benchmarks when applying hybrid and multi-aspect reward modeling. Our best performing model in the 3B family achieves an overall average improvement of ~9.5% across general and math reasoning tasks. Focusing specifically on mathematical benchmarks, the model achieves a significant average improvement of ~16%, highlighting its effectiveness in mathematical reasoning and problem solving. 

---
# Efficient Prediction of Pass@k Scaling in Large Language Models 

**Authors**: Joshua Kazdan, Rylan Schaeffer, Youssef Allouah, Colin Sullivan, Kyssen Yu, Noam Levi, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05197)  

**Abstract**: Assessing the capabilities and risks of frontier AI systems is a critical area of research, and recent work has shown that repeated sampling from models can dramatically increase both. For instance, repeated sampling has been shown to increase their capabilities, such as solving difficult math and coding problems, but it has also been shown to increase their potential for harm, such as being jailbroken. Such results raise a crucial question for both capability and safety forecasting: how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget? This question is directly relevant to model providers, who serve hundreds of millions of users daily, and to governmental regulators, who seek to prevent harms. To answer this questions, we make three contributions. First, we find that standard methods for fitting these laws suffer from statistical shortcomings that hinder predictive accuracy, especially in data-limited scenarios. Second, we remedy these shortcomings by introducing a robust estimation framework, which uses a beta-binomial distribution to generate more accurate predictions from limited data. Third, we propose a dynamic sampling strategy that allocates a greater budget to harder problems. Combined, these innovations enable more reliable prediction of rare risks and capabilities at a fraction of the computational cost. 

---
# Graph-based LLM over Semi-Structured Population Data for Dynamic Policy Response 

**Authors**: Daqian Shi, Xiaolei Diao, Jinge Wu, Honghan Wu, Xiongfeng Tang, Felix Naughton, Paulina Bondaronek  

**Link**: [PDF](https://arxiv.org/pdf/2510.05196)  

**Abstract**: Timely and accurate analysis of population-level data is crucial for effective decision-making during public health emergencies such as the COVID-19 pandemic. However, the massive input of semi-structured data, including structured demographic information and unstructured human feedback, poses significant challenges to conventional analysis methods. Manual expert-driven assessments, though accurate, are inefficient, while standard NLP pipelines often require large task-specific labeled datasets and struggle with generalization across diverse domains. To address these challenges, we propose a novel graph-based reasoning framework that integrates large language models with structured demographic attributes and unstructured public feedback in a weakly supervised pipeline. The proposed approach dynamically models evolving citizen needs into a need-aware graph, enabling population-specific analyses based on key features such as age, gender, and the Index of Multiple Deprivation. It generates interpretable insights to inform responsive health policy decision-making. We test our method using a real-world dataset, and preliminary experimental results demonstrate its feasibility. This approach offers a scalable solution for intelligent population health monitoring in resource-constrained clinical and governmental settings. 

---
# Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents 

**Authors**: Wenda Xie, Chao Guo, Yanqing Jing. Junle Wang, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05188)  

**Abstract**: Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts. 

---
# Real-time Framework for Interoperable Semantic-driven Internet-of-Things in Smart Agriculture 

**Authors**: Mohamed El-Dosuky  

**Link**: [PDF](https://arxiv.org/pdf/2510.05187)  

**Abstract**: The Internet of Things (IoT) has revolutionized various applications including agriculture, but it still faces challenges in data collection and understanding. This paper proposes a real-time framework with three additional semantic layers to help IoT devices and sensors comprehend data meaning and source. The framework consists of six layers: perception, semantic annotation, interoperability, transportation, semantic reasoning, and application, suitable for dynamic environments. Sensors collect data in the form of voltage, which is then processed by microprocessors or microcontrollers in the semantic annotation and preprocessing layer. Metadata is added to the raw data, including the purpose, ID number, and application. Two semantic algorithms are proposed in the semantic interoperability and ontologies layer: the interoperability semantic algorithm for standardizing file types and the synonym identification algorithm for identifying synonyms. In the transportation layer, raw data and metadata are sent to other IoT devices or cloud computing platforms using techniques like WiFi, Zigbee networks, Bluetooth, and mobile communication networks. A semantic reasoning layer is proposed to infer new knowledge from the existing data, using fuzzy logic, Dempster-Shafer theory, and Bayesian networks. A Graphical User Interface (GUI) is proposed in the application layer to help users communicate with and monitor IoT sensors, devices, and new knowledge inferred. This framework provides a robust solution for managing IoT data, ensuring semantic completeness, and enabling real-time knowledge inference. The integration of uncertainty reasoning methods and semantic interoperability techniques makes this framework a valuable tool for advancing IoT applications in general and in agriculture in particular. 

---
# Representation Potentials of Foundation Models for Multimodal Alignment: A Survey 

**Authors**: Jianglin Lu, Hailing Wang, Yi Xu, Yizhou Wang, Kuo Yang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05184)  

**Abstract**: Foundation models learn highly transferable representations through large-scale pretraining on diverse data. An increasing body of research indicates that these representations exhibit a remarkable degree of similarity across architectures and modalities. In this survey, we investigate the representation potentials of foundation models, defined as the latent capacity of their learned representations to capture task-specific information within a single modality while also providing a transferable basis for alignment and unification across modalities. We begin by reviewing representative foundation models and the key metrics that make alignment measurable. We then synthesize empirical evidence of representation potentials from studies in vision, language, speech, multimodality, and neuroscience. The evidence suggests that foundation models often exhibit structural regularities and semantic consistencies in their representation spaces, positioning them as strong candidates for cross-modal transfer and alignment. We further analyze the key factors that foster representation potentials, discuss open questions, and highlight potential challenges. 

---
# Lang-PINN: From Language to Physics-Informed Neural Networks via a Multi-Agent Framework 

**Authors**: Xin He, Liangliang You, Hongduan Tian, Bo Han, Ivor Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2510.05158)  

**Abstract**: Physics-informed neural networks (PINNs) provide a powerful approach for solving partial differential equations (PDEs), but constructing a usable PINN remains labor-intensive and error-prone. Scientists must interpret problems as PDE formulations, design architectures and loss functions, and implement stable training pipelines. Existing large language model (LLM) based approaches address isolated steps such as code generation or architecture suggestion, but typically assume a formal PDE is already specified and therefore lack an end-to-end perspective. We present Lang-PINN, an LLM-driven multi-agent system that builds trainable PINNs directly from natural language task descriptions. Lang-PINN coordinates four complementary agents: a PDE Agent that parses task descriptions into symbolic PDEs, a PINN Agent that selects architectures, a Code Agent that generates modular implementations, and a Feedback Agent that executes and diagnoses errors for iterative refinement. This design transforms informal task statements into executable and verifiable PINN code. Experiments show that Lang-PINN achieves substantially lower errors and greater robustness than competitive baselines: mean squared error (MSE) is reduced by up to 3--5 orders of magnitude, end-to-end execution success improves by more than 50\%, and reduces time overhead by up to 74\%. 

---
# An Algorithmic Information-Theoretic Perspective on the Symbol Grounding Problem 

**Authors**: Zhangchi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05153)  

**Abstract**: This paper provides a definitive, unifying framework for the Symbol Grounding Problem (SGP) by reformulating it within Algorithmic Information Theory (AIT). We demonstrate that the grounding of meaning is a process fundamentally constrained by information-theoretic limits, thereby unifying the Gödelian (self-reference) and No Free Lunch (statistical) perspectives. We model a symbolic system as a universal Turing machine and define grounding as an act of information compression. The argument proceeds in four stages. First, we prove that a purely symbolic system cannot ground almost all possible "worlds" (data strings), as they are algorithmically random and thus incompressible. Second, we show that any statically grounded system, specialized for compressing a specific world, is inherently incomplete because an adversarial, incompressible world relative to the system can always be constructed. Third, the "grounding act" of adapting to a new world is proven to be non-inferable, as it requires the input of new information (a shorter program) that cannot be deduced from the system's existing code. Finally, we use Chaitin's Incompleteness Theorem to prove that any algorithmic learning process is itself a finite system that cannot comprehend or model worlds whose complexity provably exceeds its own. This establishes that meaning is the open-ended process of a system perpetually attempting to overcome its own information-theoretic limitations. 

---
# Structuring Reasoning for Complex Rules Beyond Flat Representations 

**Authors**: Zhihao Yang, Ancheng Xu, Jingpeng Li, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Ahmadreza Argha, Hamid Alinejad-Rokny, Minghuan Tan, Yujun Cai, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05134)  

**Abstract**: Large language models (LLMs) face significant challenges when processing complex rule systems, as they typically treat interdependent rules as unstructured textual data rather than as logically organized frameworks. This limitation results in reasoning divergence, where models often overlook critical rule dependencies essential for accurate interpretation. Although existing approaches such as Chain-of-Thought (CoT) reasoning have shown promise, they lack systematic methodologies for structured rule processing and are particularly susceptible to error propagation through sequential reasoning chains. To address these limitations, we propose the Dynamic Adjudication Template (DAT), a novel framework inspired by expert human reasoning processes. DAT structures the inference mechanism into three methodical stages: qualitative analysis, evidence gathering, and adjudication. During the qualitative analysis phase, the model comprehensively evaluates the contextual landscape. The subsequent evidence gathering phase involves the targeted extraction of pertinent information based on predefined template elements ([placeholder]), followed by systematic verification against applicable rules. Finally, in the adjudication phase, the model synthesizes these validated components to formulate a comprehensive judgment. Empirical results demonstrate that DAT consistently outperforms conventional CoT approaches in complex rule-based tasks. Notably, DAT enables smaller language models to match, and in some cases exceed, the performance of significantly larger LLMs, highlighting its efficiency and effectiveness in managing intricate rule systems. 

---
# Optimization Modeling via Semantic Anchored Alignment 

**Authors**: Yansen Zhang, Qingcan Kang, Yujie Chen, Yufei Wang, Xiongwei Han, Tao Zhong, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.05115)  

**Abstract**: Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.8\%, with gains of up to 21.9\% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code. 

---
# Structured Cognition for Behavioral Intelligence in Large Language Model Agents: Preliminary Study 

**Authors**: Myung Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.05107)  

**Abstract**: Large language models have advanced natural language understanding and generation, yet their use as autonomous agents raises architectural challenges for multi-step tasks. Existing frameworks often intertwine inference, memory, and control in a single prompt, which can reduce coherence and predictability. The Structured Cognitive Loop (SCL) is introduced as an alternative architecture that separates these functions. In SCL, the language model is dedicated to inference, memory is maintained externally, and execution is guided by a lightweight controller within a goal-directed loop. This design offloads cognitive load from the model and allows intermediate results to be stored, revisited, and checked before actions are taken, providing a clearer basis for traceability and evaluation.
We evaluate SCL against prompt-based baselines including ReAct and common LangChain agents across three scenarios: temperature-based travel planning, email drafting with conditional send, and constraint-guided image generation. All systems share the same base model and tools under matched decoding settings. Across 360 episodes, SCL shows modest but consistent improvements. Task success averages 86.3 percent compared with 70-77 percent for baselines. Goal fidelity is higher, redundant calls are fewer, intermediate states are reused more reliably, and unsupported assertions per 100 tool calls are reduced. Ablations show that external memory and control each contribute independently, and decoding sweeps confirm stability of the effects.
These results suggest that architectural separation can improve reliability and traceability without relying on larger models or heavier prompts. The findings are preliminary and intended to guide extended studies with additional models, longer horizons, multimodal tasks, and collaborative settings. 

---
# Rule Encoding and Compliance in Large Language Models: An Information-Theoretic Analysis 

**Authors**: Joachim Diederich  

**Link**: [PDF](https://arxiv.org/pdf/2510.05106)  

**Abstract**: The design of safety-critical agents based on large language models (LLMs) requires more than simple prompt engineering. This paper presents a comprehensive information-theoretic analysis of how rule encodings in system prompts influence attention mechanisms and compliance behaviour. We demonstrate that rule formats with low syntactic entropy and highly concentrated anchors reduce attention entropy and improve pointer fidelity, but reveal a fundamental trade-off between anchor redundancy and attention entropy that previous work failed to recognize. Through formal analysis of multiple attention architectures including causal, bidirectional, local sparse, kernelized, and cross-attention mechanisms, we establish bounds on pointer fidelity and show how anchor placement strategies must account for competing fidelity and entropy objectives. Combining these insights with a dynamic rule verification architecture, we provide a formal proof that hot reloading of verified rule sets increases the asymptotic probability of compliant outputs. These findings underscore the necessity of principled anchor design and dual enforcement mechanisms to protect LLM-based agents against prompt injection attacks while maintaining compliance in evolving domains. 

---
# EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark 

**Authors**: Deheng Zhang, Yuqian Fu, Runyi Yang, Yang Miao, Tianwen Qian, Xu Zheng, Guolei Sun, Ajad Chhatkuli, Xuanjing Huang, Yu-Gang Jiang, Luc Van Gool, Danda Pani Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2510.06218)  

**Abstract**: Most existing benchmarks for egocentric vision understanding focus primarily on daytime scenarios, overlooking the low-light conditions that are inevitable in real-world applications. To investigate this gap, we present EgoNight, the first comprehensive benchmark for nighttime egocentric vision, with visual question answering (VQA) as the core task. A key feature of EgoNight is the introduction of day-night aligned videos, which enhance night annotation quality using the daytime data and reveal clear performance gaps between lighting conditions. To achieve this, we collect both synthetic videos rendered by Blender and real-world recordings, ensuring that scenes and actions are visually and temporally aligned. Leveraging these paired videos, we construct EgoNight-VQA, supported by a novel day-augmented night auto-labeling engine and refinement through extensive human verification. Each QA pair is double-checked by annotators for reliability. In total, EgoNight-VQA contains 3658 QA pairs across 90 videos, spanning 12 diverse QA types, with more than 300 hours of human work. Evaluations of state-of-the-art multimodal large language models (MLLMs) reveal substantial performance drops when transferring from day to night, underscoring the challenges of reasoning under low-light conditions. Beyond VQA, EgoNight also introduces two auxiliary tasks, day-night correspondence retrieval and egocentric depth estimation at night, that further explore the boundaries of existing models. We believe EgoNight-VQA provides a strong foundation for advancing application-driven egocentric vision research and for developing models that generalize across illumination domains. All the data and code will be made available upon acceptance. 

---
# Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents 

**Authors**: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.06214)  

**Abstract**: Large language model (LLM) agents increasingly rely on external tools such as search engines to solve complex, multi-step problems, and reinforcement learning (RL) has become a key paradigm for training them. However, the trajectories of search agents are structurally heterogeneous, where variations in the number, placement, and outcomes of search calls lead to fundamentally different answer directions and reward distributions. Standard policy gradient methods, which use a single global baseline, suffer from what we identify and formalize as cross-stratum bias-an "apples-to-oranges" comparison of heterogeneous trajectories. This cross-stratum bias distorts credit assignment and hinders exploration of complex, multi-step search strategies. To address this, we propose Stratified GRPO, whose central component, Stratified Advantage Normalization (SAN), partitions trajectories into homogeneous strata based on their structural properties and computes advantages locally within each stratum. This ensures that trajectories are evaluated only against their true peers. Our analysis proves that SAN eliminates cross-stratum bias, yields conditionally unbiased unit-variance estimates inside each stratum, and retains the global unbiasedness and unit-variance properties enjoyed by standard normalization, resulting in a more pure and scale-stable learning signal. To improve practical stability under finite-sample regimes, we further linearly blend SAN with the global estimator. Extensive experiments on diverse single-hop and multi-hop question-answering benchmarks demonstrate that Stratified GRPO consistently and substantially outperforms GRPO by up to 11.3 points, achieving higher training rewards, greater training stability, and more effective search policies. These results establish stratification as a principled remedy for structural heterogeneity in RL for LLM search agents. 

---
# Reference Grounded Skill Discovery 

**Authors**: Seungeun Rho, Aaron Trinh, Danfei Xu, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2510.06203)  

**Abstract**: Scaling unsupervised skill discovery algorithms to high-DoF agents remains challenging. As dimensionality increases, the exploration space grows exponentially, while the manifold of meaningful skills remains limited. Therefore, semantic meaningfulness becomes essential to effectively guide exploration in high-dimensional spaces. In this work, we present Reference-Grounded Skill Discovery (RGSD), a novel algorithm that grounds skill discovery in a semantically meaningful latent space using reference data. RGSD first performs contrastive pretraining to embed motions on a unit hypersphere, clustering each reference trajectory into a distinct direction. This grounding enables skill discovery to simultaneously involve both imitation of reference behaviors and the discovery of semantically related diverse behaviors. On a simulated SMPL humanoid with 359-D observations and 69-D actions, RGSD learns structured skills including walking, running, punching, and side stepping, and also discovers related novel behaviors. In downstream control tasks, RGSD outperforms imitation-based skill acquisition baselines. Our results suggest that lightweight reference-guided grounding offers a practical path to discovering semantically rich and structured skills in high-DoF systems. 

---
# TokenChain: A Discrete Speech Chain via Semantic Token Modeling 

**Authors**: Mingxuan Wang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.06201)  

**Abstract**: Machine Speech Chain, simulating the human perception-production loop, proves effective in jointly improving ASR and TTS. We propose TokenChain, a fully discrete speech chain coupling semantic-token ASR with a two-stage TTS: an autoregressive text-to-semantic model co-trained with ASR and a masked-generative semantic-to-acoustic model for synthesis only. End-to-end feedback across the text interface is enabled with straight-through argmax/Gumbel-Softmax and balanced with supervised ASR via dynamic weight averaging. Ablations examine optimal temperature schedules for in- and cross-domain transfer. Evaluation reveals TokenChain surpasses baseline accuracy 2-6 epochs earlier and yields 5-13% lower equal-epoch error with stable T2S on LibriSpeech, and reduces relative ASR WER by 56% and T2S WER by 31% on TED-LIUM with minimal forgetting, showing that chain learning remains effective with token interfaces and models. 

---
# StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars 

**Authors**: Weijian Li, Hong-Yu Chen, Qinjie Lin, Nabeel Rehemtulla, Ved G. Shah, Dennis Wu, Adam A. Miller, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06200)  

**Abstract**: Time series foundation models (TSFMs) are increasingly being adopted as highly-capable general-purpose time series representation learners. Although their training corpora are vast, they exclude astronomical time series data. Observations of stars produce peta-scale time series with unique challenges including irregular sampling and heteroskedasticity. We introduce StarEmbed, the first public benchmark for rigorous and standardized evaluation of state-of-the-art TSFMs on stellar time series observations (``light curves''). We benchmark on three scientifically-motivated downstream tasks: unsupervised clustering, supervised classification, and out-of-distribution source detection. StarEmbed integrates a catalog of expert-vetted labels with multi-variate light curves from the Zwicky Transient Facility, yielding ~40k hand-labeled light curves spread across seven astrophysical classes. We evaluate the zero-shot representation capabilities of three TSFMs (MOIRAI, Chronos, Chronos-Bolt) and a domain-specific transformer (Astromer) against handcrafted feature extraction, the long-standing baseline in the astrophysics literature. Our results demonstrate that these TSFMs, especially the Chronos models, which are trained on data completely unlike the astronomical observations, can outperform established astrophysics-specific baselines in some tasks and effectively generalize to entirely new data. In particular, TSFMs deliver state-of-the-art performance on our out-of-distribution source detection benchmark. With the first benchmark of TSFMs on astronomical time series data, we test the limits of their generalization and motivate a paradigm shift in time-domain astronomy from using task-specific, fully supervised pipelines toward adopting generic foundation model representations for the analysis of peta-scale datasets from forthcoming observatories. 

---
# Latent Speech-Text Transformer 

**Authors**: Yen-Ju Lu, Yashesh Gaur, Wei Zhou, Benjamin Muller, Jesus Villalba, Najim Dehak, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Srinivasan Iyer, Duc Le  

**Link**: [PDF](https://arxiv.org/pdf/2510.06195)  

**Abstract**: Auto-regressive speech-text models are typically pre-trained on a large number of interleaved sequences of text tokens and raw speech encoded as speech tokens using vector quantization. These models have demonstrated state-of-the-art performance in speech-to-speech understanding and generation benchmarks, together with promising scaling laws, primarily enabled by the representational alignment between text and speech. Nevertheless, they suffer from shortcomings, partly owing to the disproportionately longer sequences of speech tokens in contrast to textual tokens. This results in a large compute imbalance between modalities during pre-training as well as during inference, and a potential hindrance to effectively aligning speech and text, ultimately translating to several orders of magnitude slower scaling laws. We introduce the Latent Speech-Text Transformer (LST), which makes pre-training speech-text models more data-efficient by dynamically and inexpensively aggregating speech tokens into latent speech patches. These patches serve as higher-level units that can either align with corresponding textual units to aid capability transfer or even encapsulate common speech sequences like silences to be more compute-efficient. We show that LST outperforms vanilla approaches on speech-to-speech as well as text-to-text benchmarks in both data- and compute-controlled settings, the former indicating more effective representational alignment and the latter indicating steeper scaling laws for speech-text models. On HellaSwag story completion, LST achieves 6.5% absolute gain in speech accuracy under compute-controlled training and 5.3% under data-controlled training, while also improving text performance. We will release our models, code, and the evaluation data to facilitate further research. 

---
# BanglaTalk: Towards Real-Time Speech Assistance for Bengali Regional Dialects 

**Authors**: Jakir Hasan, Shubhashis Roy Dipta  

**Link**: [PDF](https://arxiv.org/pdf/2510.06188)  

**Abstract**: Real-time speech assistants are becoming increasingly popular for ensuring improved accessibility to information. Bengali, being a low-resource language with a high regional dialectal diversity, has seen limited progress in developing such systems. Existing systems are not optimized for real-time use and focus only on standard Bengali. In this work, we present BanglaTalk, the first real-time speech assistance system for Bengali regional dialects. BanglaTalk follows the client-server architecture and uses the Real-time Transport Protocol (RTP) to ensure low-latency communication. To address dialectal variation, we introduce a dialect-aware ASR system, BRDialect, developed by fine-tuning the IndicWav2Vec model in ten Bengali regional dialects. It outperforms the baseline ASR models by 12.41-33.98% on the RegSpeech12 dataset. Furthermore, BanglaTalk can operate at a low bandwidth of 24 kbps while maintaining an average end-to-end delay of 4.9 seconds. Low bandwidth usage and minimal end-to-end delay make the system both cost-effective and interactive for real-time use cases, enabling inclusive and accessible speech technology for the diverse community of Bengali speakers. 

---
# Automated Program Repair of Uncompilable Student Code 

**Authors**: Griffin Pitts, Aum Pandya, Darsh Rank, Tirth Bhatt, Muntasir Hoq, Bita Akram  

**Link**: [PDF](https://arxiv.org/pdf/2510.06187)  

**Abstract**: A significant portion of student programming submissions in CS1 learning environments are uncompilable, limiting their use in student modeling and downstream knowledge tracing. Traditional modeling pipelines often exclude these cases, discarding observations of student learning. This study investigates automated program repair as a strategy to recover uncompilable code while preserving students' structural intent for use in student modeling. Within this framework, we assess large language models (LLMs) as repair agents, including GPT-5 (OpenAI), Claude 3.5 Haiku (Anthropic), and Gemini 2.5 Flash (Google), under high- and low-context prompting conditions. Repairs were evaluated for compilability, edit distance, and preservation of students' original structure and logic. We find that while all three LLMs are capable of producing compilable repairs, their behavior diverges in how well they preserve students' control flow and code structure, which affects their pedagogical utility. By recovering uncompilable submissions, this work enables richer and more comprehensive analyses of learners' coding processes and development over time. 

---
# RECODE-H: A Benchmark for Research Code Development with Interactive Human Feedback 

**Authors**: Chunyu Miao, Henry Peng Zou, Yangning Li, Yankai Chen, Yibo Wang, Fangxin Wang, Yifan Li, Wooseong Yang, Bowei He, Xinni Zhang, Dianzhi Yu, Hanchen Yang, Hoang H Nguyen, Yue Zhou, Jie Yang, Jizhou Guo, Wenzhe Fan, Chin-Yuan Yeh, Panpan Meng, Liancheng Fang, Jinhu Qi, Wei-Chieh Huang, Zhengyao Gu, Yuwei Han, Langzhou He, Yuyao Yang, Xue Liu, Irwin King, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06186)  

**Abstract**: Large language models (LLMs) show the promise in supporting scientific research implementation, yet their ability to generate correct and executable code remains limited. Existing works largely adopt one-shot settings, ignoring the iterative and feedback-driven nature of realistic workflows of scientific research development. To address this gap, we present RECODE-H, a benchmark of 102 tasks from research papers and repositories that evaluates LLM agents through multi-turn interactions with LLM-simulated human feedback. It includes structured instructions,unit tests, and a five-level feedback hierarchy to reflect realistic researcher-agent collaboration. We further present ReCodeAgent, a framework that integrates feedback into iterative code generation. Experiments with leading LLMs, including GPT-5, Claude-Sonnet-4, DeepSeek-V3.1, and Gemini 2.5, show substantial performance gains with richer feedback, while also highlighting ongoing challenges in the generation of complex research code. RECODE-H establishes a foundation for developing adaptive, feedback-driven LLM agents in scientific research implementation 

---
# Smartphone-based iris recognition through high-quality visible-spectrum iris image capture.V2 

**Authors**: Naveenkumar G Venkataswamy, Yu Liu, Soumyabrata Dey, Stephanie Schuckers, Masudul H Imtiaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.06170)  

**Abstract**: Smartphone-based iris recognition in the visible spectrum (VIS) remains difficult due to illumination variability, pigmentation differences, and the absence of standardized capture controls. This work presents a compact end-to-end pipeline that enforces ISO/IEC 29794-6 quality compliance at acquisition and demonstrates that accurate VIS iris recognition is feasible on commodity devices. Using a custom Android application performing real-time framing, sharpness evaluation, and feedback, we introduce the CUVIRIS dataset of 752 compliant images from 47 subjects. A lightweight MobileNetV3-based multi-task segmentation network (LightIrisNet) is developed for efficient on-device processing, and a transformer matcher (IrisFormer) is adapted to the VIS domain. Under a standardized protocol and comparative benchmarking against prior CNN baselines, OSIRIS attains a TAR of 97.9% at FAR=0.01 (EER=0.76%), while IrisFormer, trained only on UBIRIS.v2, achieves an EER of 0.057% on CUVIRIS. The acquisition app, trained models, and a public subset of the dataset are released to support reproducibility. These results confirm that standardized capture and VIS-adapted lightweight models enable accurate and practical iris recognition on smartphones. 

---
# LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams 

**Authors**: Aju Ani Justus, Chris Baber  

**Link**: [PDF](https://arxiv.org/pdf/2510.06151)  

**Abstract**: A critical challenge in modelling Heterogeneous-Agent Teams is training agents to collaborate with teammates whose policies are inaccessible or non-stationary, such as humans. Traditional approaches rely on expensive human-in-the-loop data, which limits scalability. We propose using Large Language Models (LLMs) as policy-agnostic human proxies to generate synthetic data that mimics human decision-making. To evaluate this, we conduct three experiments in a grid-world capture game inspired by Stag Hunt, a game theory paradigm that balances risk and reward. In Experiment 1, we compare decisions from 30 human participants and 2 expert judges with outputs from LLaMA 3.1 and Mixtral 8x22B models. LLMs, prompted with game-state observations and reward structures, align more closely with experts than participants, demonstrating consistency in applying underlying decision criteria. Experiment 2 modifies prompts to induce risk-sensitive strategies (e.g. "be risk averse"). LLM outputs mirror human participants' variability, shifting between risk-averse and risk-seeking behaviours. Finally, Experiment 3 tests LLMs in a dynamic grid-world where the LLM agents generate movement actions. LLMs produce trajectories resembling human participants' paths. While LLMs cannot yet fully replicate human adaptability, their prompt-guided diversity offers a scalable foundation for simulating policy-agnostic teammates. 

---
# Bimanual 3D Hand Motion and Articulation Forecasting in Everyday Images 

**Authors**: Aditya Prakash, David Forsyth, Saurabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.06145)  

**Abstract**: We tackle the problem of forecasting bimanual 3D hand motion & articulation from a single image in everyday settings. To address the lack of 3D hand annotations in diverse settings, we design an annotation pipeline consisting of a diffusion model to lift 2D hand keypoint sequences to 4D hand motion. For the forecasting model, we adopt a diffusion loss to account for the multimodality in hand motion distribution. Extensive experiments across 6 datasets show the benefits of training on diverse data with imputed labels (14% improvement) and effectiveness of our lifting (42% better) & forecasting (16.4% gain) models, over the best baselines, especially in zero-shot generalization to everyday images. 

---
# Multi-Task Reinforcement Learning with Language-Encoded Gated Policy Networks 

**Authors**: Rushiv Arora  

**Link**: [PDF](https://arxiv.org/pdf/2510.06138)  

**Abstract**: Multi-task reinforcement learning often relies on task metadata -- such as brief natural-language descriptions -- to guide behavior across diverse objectives. We present Lexical Policy Networks (LEXPOL), a language-conditioned mixture-of-policies architecture for multi-task RL. LEXPOL encodes task metadata with a text encoder and uses a learned gating module to select or blend among multiple sub-policies, enabling end-to-end training across tasks. On MetaWorld benchmarks, LEXPOL matches or exceeds strong multi-task baselines in success rate and sample efficiency, without task-specific retraining. To analyze the mechanism, we further study settings with fixed expert policies obtained independently of the gate and show that the learned language gate composes these experts to produce behaviors appropriate to novel task descriptions and unseen task combinations. These results indicate that natural-language metadata can effectively index and recombine reusable skills within a single policy. 

---
# CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credits 

**Authors**: Kangyu Wang, Zhiyun Jiang, Haibo Feng, Weijia Zhao, Lin Liu, Jianguo Li, Zhenzhong Lan, Weiyao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06133)  

**Abstract**: Diffusion large language models (dLLMs) generate text through iterative denoising steps, achieving parallel decoding by denoising only high-confidence positions at each step. However, existing approaches often repetitively remask tokens due to initially low confidence scores, leading to redundant iterations and limiting overall acceleration. Through the analysis of dLLM decoding traces, we observe that the model often determines the final prediction for a token several steps before the decoding step. To leverage this historical information and avoid redundant steps, we introduce the concept of Trace Credit, which quantifies each token's convergence potential by accumulating historical logits. Furthermore, we propose CreditDecoding, a training-free parallel decoding algorithm that accelerates the confidence convergence of correct but underconfident tokens by fusing current logits with Trace Credit. This process significantly reduces redundant iterations and enhances decoding robustness. On eight benchmarks, CreditDecoding achieves a 5.48 times speedup and a 0.48 performance improvement over LLaDA-8B-Instruct, and a 4.11 times speedup with a 0.15 performance improvement over LLaDA-MoE-Instruct. Importantly, CreditDecoding scales effectively to long sequences and is orthogonal to mainstream inference optimizations, making it a readily integrable and versatile solution. 

---
# Discrete Diffusion Models with MLLMs for Unified Medical Multimodal Generation 

**Authors**: Jiawei Mao, Yuhan Wang, Lifeng Chen, Can Zhao, Yucheng Tang, Dong Yang, Liangqiong Qu, Daguang Xu, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06131)  

**Abstract**: Recent advances in generative medical models are constrained by modality-specific scenarios that hinder the integration of complementary evidence from imaging, pathology, and clinical notes. This fragmentation limits their evolution into foundation models that can learn and reason across the full spectrum of biomedical data. We propose MeDiM, the first medical discrete diffusion model that learns shared distributions across modalities without modality-specific components. MeDiM unifies multiple generative tasks: translating between images and text, and jointly producing image-report pairs across domains in response to prompts. Built on a discrete diffusion framework, MeDiM bridges vision and language representations through a shared probabilistic space. To enable unified and flexible medical generation, we employ a multimodal large language model (MLLM) as the diffusion backbone, leveraging its prior knowledge and cross-modal reasoning. Two key designs are introduced: (1) removing the causal attention mask for bidirectional context, and (2) injecting continuous timestep embeddings for diffusion awareness. Experiments demonstrate high-fidelity medical generation (FID 16.60 on MIMIC-CXR and FID 24.19 on PathGen) and accurate report generation (METEOR 0.2650 and 0.2580). Jointly generated image-report pairs further enhance downstream performance (plus6.43 percent BLEU-1, plus18.57 percent BLEU-2, plus31.58 percent BLEU-3, plus4.80 percent METEOR), showing that MeDiM supports coherent and clinically grounded multimodal outputs. 

---
# Distributional Semantics Tracing: A Framework for Explaining Hallucinations in Large Language Models 

**Authors**: Gagan Bhatia, Somayajulu G Sripada, Kevin Allan, Jacobo Azcona  

**Link**: [PDF](https://arxiv.org/pdf/2510.06107)  

**Abstract**: Large Language Models (LLMs) are prone to hallucination, the generation of plausible yet factually incorrect statements. This work investigates the intrinsic, architectural origins of this failure mode through three primary this http URL, to enable the reliable tracing of internal semantic failures, we propose \textbf{Distributional Semantics Tracing (DST)}, a unified framework that integrates established interpretability techniques to produce a causal map of a model's reasoning, treating meaning as a function of context (distributional semantics). Second, we pinpoint the model's layer at which a hallucination becomes inevitable, identifying a specific \textbf{commitment layer} where a model's internal representations irreversibly diverge from factuality. Third, we identify the underlying mechanism for these failures. We observe a conflict between distinct computational pathways, which we interpret using the lens of dual-process theory: a fast, heuristic \textbf{associative pathway} (akin to System 1) and a slow, deliberate \textbf{contextual pathway} (akin to System 2), leading to predictable failure modes such as \textit{Reasoning Shortcut Hijacks}. Our framework's ability to quantify the coherence of the contextual pathway reveals a strong negative correlation ($\rho = -0.863$) with hallucination rates, implying that these failures are predictable consequences of internal semantic weakness. The result is a mechanistic account of how, when, and why hallucinations occur within the Transformer architecture. 

---
# A public cardiac CT dataset featuring the left atrial appendage 

**Authors**: Bjoern Hansen, Jonas Pedersen, Klaus F. Kofoed, Oscar Camara, Rasmus R. Paulsen, Kristine Soerensen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06090)  

**Abstract**: Despite the success of advanced segmentation frameworks such as TotalSegmentator (TS), accurate segmentations of the left atrial appendage (LAA), coronary arteries (CAs), and pulmonary veins (PVs) remain a significant challenge in medical imaging. In this work, we present the first open-source, anatomically coherent dataset of curated, high-resolution segmentations for these structures, supplemented with whole-heart labels produced by TS on the publicly available ImageCAS dataset consisting of 1000 cardiac computed tomography angiography (CCTA) scans. One purpose of the data set is to foster novel approaches to the analysis of LAA morphology.
LAA segmentations on ImageCAS were generated using a state-of-the-art segmentation framework developed specifically for high resolution LAA segmentation. We trained the network on a large private dataset with manual annotations provided by medical readers guided by a trained cardiologist and transferred the model to ImageCAS data. CA labels were improved from the original ImageCAS annotations, while PV segmentations were refined from TS outputs. In addition, we provide a list of scans from ImageCAS that contains common data flaws such as step artefacts, LAAs extending beyond the scanner's field of view, and other types of data defects. 

---
# Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability 

**Authors**: Taylor Sorensen, Benjamin Newman, Jared Moore, Chan Park, Jillian Fisher, Niloofar Mireshghallah, Liwei Jiang, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06084)  

**Abstract**: Language model post-training has enhanced instruction-following and performance on many downstream tasks, but also comes with an often-overlooked cost on tasks with many possible valid answers. We characterize three desiderata for conditional distributional modeling: in-context steerability, valid output space coverage, and distributional alignment, and document across three model families how current post-training can reduce these properties. In particular, we disambiguate between two kinds of in-context learning: ICL for eliciting existing underlying knowledge or capabilities, and in-context steerability, where a model must use in-context information to override its priors and steer to a novel data generating distribution. To better evaluate and improve these desiderata, we introduce Spectrum Suite, a large-scale resource compiled from >40 data sources and spanning >90 tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. We find that while current post-training techniques help elicit underlying capabilities and knowledge, they hurt models' ability to flexibly steer in-context. To mitigate these issues, we propose Spectrum Tuning, a post-training method using Spectrum Suite to improve steerability and distributional coverage. We find that Spectrum Tuning often improves over pretrained models and their instruction-tuned counterparts, enhancing steerability, spanning more of the output space, and improving distributional alignment on held-out datasets. 

---
# When Thinking Drifts: Evidential Grounding for Robust Video Reasoning 

**Authors**: Mi Luo, Zihui Xue, Alex Dimakis, Kristen Grauman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06077)  

**Abstract**: Video reasoning, the task of enabling machines to infer from dynamic visual content through multi-step logic, is crucial for advanced AI. While the Chain-of-Thought (CoT) mechanism has enhanced reasoning in text-based tasks, its application to video understanding remains underexplored. This paper presents a systematic analysis revealing that CoT often degrades performance in video reasoning, generating verbose but misleading internal monologues, and leading to hallucinated visual details and overridden correct intuitions - a phenomenon we term "visual thinking drift". We explain this drift through a Bayesian lens, positing that CoT traces often diverge from actual visual evidence, instead amplifying internal biases or language priors, causing models to storytell rather than engage in grounded reasoning. To counteract this, we introduce Visual Evidence Reward (VER), a novel reinforcement learning framework that explicitly rewards the generation of reasoning traces that are verifiably grounded in visual evidence. Comprehensive evaluation across 10 diverse video understanding benchmarks demonstrates that our Video-VER consistently achieves top performance. Our work sheds light on the distinct challenges of video-centric reasoning and encourages the development of AI that robustly grounds its inferences in visual evidence - for large multimodal models that not only "think before answering", but also "see while thinking". 

---
# Benchmark It Yourself (BIY): Preparing a Dataset and Benchmarking AI Models for Scatterplot-Related Tasks 

**Authors**: João Palmeiro, Diogo Duarte, Rita Costa, Pedro Bizarro  

**Link**: [PDF](https://arxiv.org/pdf/2510.06071)  

**Abstract**: AI models are increasingly used for data analysis and visualization, yet benchmarks rarely address scatterplot-specific tasks, limiting insight into performance. To address this gap for one of the most common chart types, we introduce a synthetic, annotated dataset of over 18,000 scatterplots from six data generators and 17 chart designs, and a benchmark based on it. We evaluate proprietary models from OpenAI and Google using N-shot prompting on five distinct tasks derived from annotations of cluster bounding boxes, their center coordinates, and outlier coordinates. OpenAI models and Gemini 2.5 Flash, especially when prompted with examples, are viable options for counting clusters and, in Flash's case, outliers (90%+ Accuracy). However, the results for localization-related tasks are unsatisfactory: Precision and Recall are near or below 50%, except for Flash in outlier identification (65.01%). Furthermore, the impact of chart design on performance appears to be a secondary factor, but it is advisable to avoid scatterplots with wide aspect ratios (16:9 and 21:9) or those colored randomly. Supplementary materials are available at this https URL. 

---
# Cross-Embodiment Dexterous Hand Articulation Generation via Morphology-Aware Learning 

**Authors**: Heng Zhang, Kevin Yuchen Ma, Mike Zheng Shou, Weisi Lin, Yan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06068)  

**Abstract**: Dexterous grasping with multi-fingered hands remains challenging due to high-dimensional articulations and the cost of optimization-based pipelines. Existing end-to-end methods require training on large-scale datasets for specific hands, limiting their ability to generalize across different embodiments. We propose an eigengrasp-based, end-to-end framework for cross-embodiment grasp generation. From a hand's morphology description, we derive a morphology embedding and an eigengrasp set. Conditioned on these, together with the object point cloud and wrist pose, an amplitude predictor regresses articulation coefficients in a low-dimensional space, which are decoded into full joint articulations. Articulation learning is supervised with a Kinematic-Aware Articulation Loss (KAL) that emphasizes fingertip-relevant motions and injects morphology-specific structure. In simulation on unseen objects across three dexterous hands, our model attains a 91.9% average grasp success rate with less than 0.4 seconds inference per grasp. With few-shot adaptation to an unseen hand, it achieves 85.6% success on unseen objects in simulation, and real-world experiments on this few-shot generalized hand achieve an 87% success rate. The code and additional materials will be made available upon publication on our project website this https URL. 

---
# Reasoning under Vision: Understanding Visual-Spatial Cognition in Vision-Language Models for CAPTCHA 

**Authors**: Python Song, Luke Tenyi Chang, Yun-Yun Tsai, Penghui Li, Junfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06067)  

**Abstract**: CAPTCHA, originally designed to distinguish humans from robots, has evolved into a real-world benchmark for assessing the spatial reasoning capabilities of vision-language models. In this work, we first show that step-by-step reasoning is crucial for vision-language models (VLMs) to solve CAPTCHAs, which represent high-difficulty spatial reasoning tasks, and that current commercial vision-language models still struggle with such reasoning. In particular, we observe that most commercial VLMs (e.g., Gemini, Claude, GPT, etc.) fail to effectively solve CAPTCHAs and thus achieve low accuracy (around 21.9 percent). However, our findings indicate that requiring the model to perform step-by-step reasoning before generating the final coordinates can significantly enhance its solving accuracy, underscoring the severity of the gap. To systematically study this issue, we introduce CAPTCHA-X, the first real-world CAPTCHA benchmark with reasoning, covering seven categories of CAPTCHAs (such as Gobang, hCaptcha, etc.) with step-by-step action solutions and grounding annotations. We further define five reasoning-oriented metrics that enable a comprehensive evaluation of models reasoning capabilities. To validate the effectiveness of reasoning, we also propose a general agentic VLM-based framework that incorporates the models inherent reasoning abilities. Our method achieves state-of-the-art performance across five high-difficulty CAPTCHA types, with an average solving accuracy of 83.9 percent, substantially surpassing existing baselines. These results reveal the limitations of current models and highlight the importance of reasoning in advancing visual-spatial challenges in the future. 

---
# Controllable Audio-Visual Viewpoint Generation from 360° Spatial Information 

**Authors**: Christian Marinoni, Riccardo Fosco Gramaccioni, Eleonora Grassucci, Danilo Comminiello  

**Link**: [PDF](https://arxiv.org/pdf/2510.06060)  

**Abstract**: The generation of sounding videos has seen significant advancements with the advent of diffusion models. However, existing methods often lack the fine-grained control needed to generate viewpoint-specific content from larger, immersive 360-degree environments. This limitation restricts the creation of audio-visual experiences that are aware of off-camera events. To the best of our knowledge, this is the first work to introduce a framework for controllable audio-visual generation, addressing this unexplored gap. Specifically, we propose a diffusion model by introducing a set of powerful conditioning signals derived from the full 360-degree space: a panoramic saliency map to identify regions of interest, a bounding-box-aware signed distance map to define the target viewpoint, and a descriptive caption of the entire scene. By integrating these controls, our model generates spatially-aware viewpoint videos and audios that are coherently influenced by the broader, unseen environmental context, introducing a strong controllability that is essential for realistic and immersive audio-visual generation. We show audiovisual examples proving the effectiveness of our framework. 

---
# GLVD: Guided Learned Vertex Descent 

**Authors**: Pol Caselles Rico, Francesc Moreno Noguer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06046)  

**Abstract**: Existing 3D face modeling methods usually depend on 3D Morphable Models, which inherently constrain the representation capacity to fixed shape priors. Optimization-based approaches offer high-quality reconstructions but tend to be computationally expensive. In this work, we introduce GLVD, a hybrid method for 3D face reconstruction from few-shot images that extends Learned Vertex Descent (LVD) by integrating per-vertex neural field optimization with global structural guidance from dynamically predicted 3D keypoints. By incorporating relative spatial encoding, GLVD iteratively refines mesh vertices without requiring dense 3D supervision. This enables expressive and adaptable geometry reconstruction while maintaining computational efficiency. GLVD achieves state-of-the-art performance in single-view settings and remains highly competitive in multi-view scenarios, all while substantially reducing inference time. 

---
# VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization 

**Authors**: Xinye Cao, Hongcan Guo, Jiawen Qian, Guoshun Nan, Chao Wang, Yuqi Pan, Tianhao Hou, Xiaojuan Wang, Yutong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06040)  

**Abstract**: Understanding hour-long videos with multi-modal large language models (MM-LLMs) enriches the landscape of human-centered AI applications. However, for end-to-end video understanding with LLMs, uniformly sampling video frames results in LLMs being overwhelmed by a vast amount of irrelevant information as video length increases. Existing hierarchical key frame extraction methods improve the accuracy of video understanding but still face two critical challenges. 1) How can the interference of extensive redundant information in long videos be mitigated? 2) How can a model dynamically adapt to complex hierarchical structures while accurately identifying key frames? To address these issues, we propose VideoMiner, which iteratively segments, captions, and clusters long videos, forming a hierarchical tree structure. The proposed VideoMiner progresses from long videos to events to frames while preserving temporal coherence, effectively addressing the first challenge. To precisely locate key frames, we introduce T-GRPO, a tree-based group relative policy optimization in reinforcement learning method that guides the exploration of the VideoMiner. The proposed T-GRPO is specifically designed for tree structures, integrating spatiotemporal information at the event level while being guided by the question, thus solving the second challenge. We achieve superior performance in all long-video understanding tasks and uncover several interesting insights. Our proposed T-GRPO surprisingly incentivizes the model to spontaneously generate a reasoning chain. Additionally, the designed tree growth auxin dynamically adjusts the expansion depth, obtaining accuracy and efficiency gains. The code is publicly available at this https URL. 

---
# CDTP: A Large-Scale Chinese Data-Text Pair Dataset for Comprehensive Evaluation of Chinese LLMs 

**Authors**: Chengwei Wu, Jiapu Wang, Mingyang Gao, Xingrui Zhuo, Jipeng Guo, Runlin Lei, Haoran Luo, Tianyu Chen, Haoyi Zhou, Shirui Pan, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.06039)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, Chinese LLMs face unique challenges, primarily due to the dominance of unstructured free text and the lack of structured representations in Chinese corpora. While existing benchmarks for LLMs partially assess Chinese LLMs, they are still predominantly English-centric and fail to address the unique linguistic characteristics of Chinese, lacking structured datasets essential for robust evaluation. To address these challenges, we present a Comprehensive Benchmark for Evaluating Chinese Large Language Models (CB-ECLLM) based on the newly constructed Chinese Data-Text Pair (CDTP) dataset. Specifically, CDTP comprises over 7 million aligned text pairs, each consisting of unstructured text coupled with one or more corresponding triples, alongside a total of 15 million triples spanning four critical domains. The core contributions of CDTP are threefold: (i) enriching Chinese corpora with high-quality structured information; (ii) enabling fine-grained evaluation tailored to knowledge-driven tasks; and (iii) supporting multi-task fine-tuning to assess generalization and robustness across scenarios, including Knowledge Graph Completion, Triple-to-Text generation, and Question Answering. Furthermore, we conduct rigorous evaluations through extensive experiments and ablation studies to assess the effectiveness, Supervised Fine-Tuning (SFT), and robustness of the benchmark. To support reproducible research, we offer an open-source codebase and outline potential directions for future investigations based on our insights. 

---
# From Learning to Mastery: Achieving Safe and Efficient Real-World Autonomous Driving with Human-In-The-Loop Reinforcement Learning 

**Authors**: Li Zeqiao, Wang Yijing, Wang Haoyu, Li Zheng, Li Peng, Liu Wenfei, Zuo Zhiqiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06038)  

**Abstract**: Autonomous driving with reinforcement learning (RL) has significant potential. However, applying RL in real-world settings remains challenging due to the need for safe, efficient, and robust learning. Incorporating human expertise into the learning process can help overcome these challenges by reducing risky exploration and improving sample efficiency. In this work, we propose a reward-free, active human-in-the-loop learning method called Human-Guided Distributional Soft Actor-Critic (H-DSAC). Our method combines Proxy Value Propagation (PVP) and Distributional Soft Actor-Critic (DSAC) to enable efficient and safe training in real-world environments. The key innovation is the construction of a distributed proxy value function within the DSAC framework. This function encodes human intent by assigning higher expected returns to expert demonstrations and penalizing actions that require human intervention. By extrapolating these labels to unlabeled states, the policy is effectively guided toward expert-like behavior. With a well-designed state space, our method achieves real-world driving policy learning within practical training times. Results from both simulation and real-world experiments demonstrate that our framework enables safe, robust, and sample-efficient learning for autonomous driving. 

---
# Fast Leave-One-Out Approximation from Fragment-Target Prevalence Vectors (molFTP) : From Dummy Masking to Key-LOO for Leakage-Free Feature Construction 

**Authors**: Guillaume Godin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06029)  

**Abstract**: We introduce molFTP (molecular fragment-target prevalence), a compact representation that delivers strong predictive performance. To prevent feature leakage across cross-validation folds, we implement a dummy-masking procedure that removes information about fragments present in the held-out molecules. We further show that key leave-one-out (key-loo) closely approximates true molecule-level leave-one-out (LOO), with deviation below 8% on our datasets. This enables near full data training while preserving unbiased cross-validation estimates of model performance. Overall, molFTP provides a fast, leakage-resistant fragment-target prevalence vectorization with practical safeguards (dummy masking or key-LOO) that approximate LOO at a fraction of its cost. 

---
# Emergent AI Surveillance: Overlearned Person Re-Identification and Its Mitigation in Law Enforcement Context 

**Authors**: An Thi Nguyen, Radina Stoykova, Eric Arazo  

**Link**: [PDF](https://arxiv.org/pdf/2510.06026)  

**Abstract**: Generic instance search models can dramatically reduce the manual effort required to analyze vast surveillance footage during criminal investigations by retrieving specific objects of interest to law enforcement. However, our research reveals an unintended emergent capability: through overlearning, these models can single out specific individuals even when trained on datasets without human subjects. This capability raises concerns regarding identification and profiling of individuals based on their personal data, while there is currently no clear standard on how de-identification can be achieved. We evaluate two technical safeguards to curtail a model's person re-identification capacity: index exclusion and confusion loss. Our experiments demonstrate that combining these approaches can reduce person re-identification accuracy to below 2% while maintaining 82% of retrieval performance for non-person objects. However, we identify critical vulnerabilities in these mitigations, including potential circumvention using partial person images. These findings highlight urgent regulatory questions at the intersection of AI governance and data protection: How should we classify and regulate systems with emergent identification capabilities? And what technical standards should be required to prevent identification capabilities from developing in seemingly benign applications? 

---
# Hybrid Quantum-Classical Policy Gradient for Adaptive Control of Cyber-Physical Systems: A Comparative Study of VQC vs. MLP 

**Authors**: Aueaphum Aueawatthanaphisut, Nyi Wunna Tun  

**Link**: [PDF](https://arxiv.org/pdf/2510.06010)  

**Abstract**: The comparative evaluation between classical and quantum reinforcement learning (QRL) paradigms was conducted to investigate their convergence behavior, robustness under observational noise, and computational efficiency in a benchmark control environment. The study employed a multilayer perceptron (MLP) agent as a classical baseline and a parameterized variational quantum circuit (VQC) as a quantum counterpart, both trained on the CartPole-v1 environment over 500 episodes. Empirical results demonstrated that the classical MLP achieved near-optimal policy convergence with a mean return of 498.7 +/- 3.2, maintaining stable equilibrium throughout training. In contrast, the VQC exhibited limited learning capability, with an average return of 14.6 +/- 4.8, primarily constrained by circuit depth and qubit connectivity. Noise robustness analysis further revealed that the MLP policy deteriorated gracefully under Gaussian perturbations, while the VQC displayed higher sensitivity at equivalent noise levels. Despite the lower asymptotic performance, the VQC exhibited significantly lower parameter count and marginally increased training time, highlighting its potential scalability for low-resource quantum processors. The results suggest that while classical neural policies remain dominant in current control benchmarks, quantum-enhanced architectures could offer promising efficiency advantages once hardware noise and expressivity limitations are mitigated. 

---
# Detection and Measurement of Hailstones with Multimodal Large Language Models 

**Authors**: Moritz Alker, David C. Schedl, Andreas Stöckl  

**Link**: [PDF](https://arxiv.org/pdf/2510.06008)  

**Abstract**: This study examines the use of social media and news images to detect and measure hailstones, utilizing pre-trained multimodal large language models. The dataset for this study comprises 474 crowdsourced images of hailstones from documented hail events in Austria, which occurred between January 2022 and September 2024. These hailstones have maximum diameters ranging from 2 to 11cm. We estimate the hail diameters and compare four different models utilizing one-stage and two-stage prompting strategies. The latter utilizes additional size cues from reference objects, such as human hands, within the image. Our results show that pretrained models already have the potential to measure hailstone diameters from images with an average mean absolute error of 1.12cm for the best model. In comparison to a single-stage prompt, two-stage prompting improves the reliability of most models. Our study suggests that these off-the-shelf models, even without fine-tuning, can complement traditional hail sensors by extracting meaningful and spatially dense information from social media imagery, enabling faster and more detailed assessments of severe weather events. The automated real-time image harvesting from social media and other sources remains an open task, but it will make our approach directly applicable to future hail events. 

---
# ECTSpeech: Enhancing Efficient Speech Synthesis via Easy Consistency Tuning 

**Authors**: Tao Zhu, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05984)  

**Abstract**: Diffusion models have demonstrated remarkable performance in speech synthesis, but typically require multi-step sampling, resulting in low inference efficiency. Recent studies address this issue by distilling diffusion models into consistency models, enabling efficient one-step generation. However, these approaches introduce additional training costs and rely heavily on the performance of pre-trained teacher models. In this paper, we propose ECTSpeech, a simple and effective one-step speech synthesis framework that, for the first time, incorporates the Easy Consistency Tuning (ECT) strategy into speech synthesis. By progressively tightening consistency constraints on a pre-trained diffusion model, ECTSpeech achieves high-quality one-step generation while significantly reducing training complexity. In addition, we design a multi-scale gate module (MSGate) to enhance the denoiser's ability to fuse features at different scales. Experimental results on the LJSpeech dataset demonstrate that ECTSpeech achieves audio quality comparable to state-of-the-art methods under single-step sampling, while substantially reducing the model's training cost and complexity. 

---
# Diffusion Models for Low-Light Image Enhancement: A Multi-Perspective Taxonomy and Performance Analysis 

**Authors**: Eashan Adhikarla, Yixin Liu, Brian D. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2510.05976)  

**Abstract**: Low-light image enhancement (LLIE) is vital for safety-critical applications such as surveillance, autonomous navigation, and medical imaging, where visibility degradation can impair downstream task performance. Recently, diffusion models have emerged as a promising generative paradigm for LLIE due to their capacity to model complex image distributions via iterative denoising. This survey provides an up-to-date critical analysis of diffusion models for LLIE, distinctively featuring an in-depth comparative performance evaluation against Generative Adversarial Network and Transformer-based state-of-the-art methods, a thorough examination of practical deployment challenges, and a forward-looking perspective on the role of emerging paradigms like foundation models. We propose a multi-perspective taxonomy encompassing six categories: Intrinsic Decomposition, Spectral & Latent, Accelerated, Guided, Multimodal, and Autonomous; that map enhancement methods across physical priors, conditioning schemes, and computational efficiency. Our taxonomy is grounded in a hybrid view of both the model mechanism and the conditioning signals. We evaluate qualitative failure modes, benchmark inconsistencies, and trade-offs between interpretability, generalization, and inference efficiency. We also discuss real-world deployment constraints (e.g., memory, energy use) and ethical considerations. This survey aims to guide the next generation of diffusion-based LLIE research by highlighting trends and surfacing open research questions, including novel conditioning, real-time adaptation, and the potential of foundation models. 

---
# LexiCon: a Benchmark for Planning under Temporal Constraints in Natural Language 

**Authors**: Periklis Mantenoglou, Rishi Hazra, Pedro Zuidberg Dos Martires, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2510.05972)  

**Abstract**: Owing to their reasoning capabilities, large language models (LLMs) have been evaluated on planning tasks described in natural language. However, LLMs have largely been tested on planning domains without constraints. In order to deploy them in real-world settings where adherence to constraints, in particular safety constraints, is critical, we need to evaluate their performance on constrained planning tasks. We introduce LexiCon -- a natural language-based (Lexi) constrained (Con) planning benchmark, consisting of a suite of environments, that can be used to evaluate the planning capabilities of LLMs in a principled fashion. The core idea behind LexiCon is to take existing planning environments and impose temporal constraints on the states. These constrained problems are then translated into natural language and given to an LLM to solve. A key feature of LexiCon is its extensibility. That is, the set of supported environments can be extended with new (unconstrained) environment generators, for which temporal constraints are constructed automatically. This renders LexiCon future-proof: the hardness of the generated planning problems can be increased as the planning capabilities of LLMs improve. Our experiments reveal that the performance of state-of-the-art LLMs, including reasoning models like GPT-5, o3, and R1, deteriorates as the degree of constrainedness of the planning tasks increases. 

---
# Probing the Difficulty Perception Mechanism of Large Language Models 

**Authors**: Sunbowen Lee, Qingyu Yin, Chak Tou Leong, Jialiang Zhang, Yicheng Gong, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05969)  

**Abstract**: Large language models (LLMs) are increasingly deployed on complex reasoning tasks, yet little is known about their ability to internally evaluate problem difficulty, which is an essential capability for adaptive reasoning and efficient resource allocation. In this work, we investigate whether LLMs implicitly encode problem difficulty in their internal representations. Using a linear probe on the final-token representations of LLMs, we demonstrate that the difficulty level of math problems can be linearly modeled. We further locate the specific attention heads of the final Transformer layer: these attention heads have opposite activation patterns for simple and difficult problems, thus achieving perception of difficulty. Our ablation experiments prove the accuracy of the location. Crucially, our experiments provide practical support for using LLMs as automatic difficulty annotators, potentially substantially reducing reliance on costly human labeling in benchmark construction and curriculum learning. We also uncover that there is a significant difference in entropy and difficulty perception at the token level. Our study reveals that difficulty perception in LLMs is not only present but also structurally organized, offering new theoretical insights and practical directions for future research. 

---
# Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density 

**Authors**: Randall Balestriero, Nicolas Ballas, Mike Rabbat, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2510.05949)  

**Abstract**: Joint Embedding Predictive Architectures (JEPAs) learn representations able to solve numerous downstream tasks out-of-the-box. JEPAs combine two objectives: (i) a latent-space prediction term, i.e., the representation of a slightly perturbed sample must be predictable from the original sample's representation, and (ii) an anti-collapse term, i.e., not all samples should have the same representation. While (ii) is often considered as an obvious remedy to representation collapse, we uncover that JEPAs' anti-collapse term does much more--it provably estimates the data density. In short, any successfully trained JEPA can be used to get sample probabilities, e.g., for data curation, outlier detection, or simply for density estimation. Our theoretical finding is agnostic of the dataset and architecture used--in any case one can compute the learned probabilities of sample $x$ efficiently and in closed-form using the model's Jacobian matrix at $x$. Our findings are empirically validated across datasets (synthetic, controlled, and Imagenet) and across different Self Supervised Learning methods falling under the JEPA family (I-JEPA and DINOv2) and on multimodal models, such as MetaCLIP. We denote the method extracting the JEPA learned density as {\bf JEPA-SCORE}. 

---
# EvalMORAAL: Interpretable Chain-of-Thought and LLM-as-Judge Evaluation for Moral Alignment in Large Language Models 

**Authors**: Hadi Mohammadi, Anastasia Giachanou, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2510.05942)  

**Abstract**: We present EvalMORAAL, a transparent chain-of-thought (CoT) framework that uses two scoring methods (log-probabilities and direct ratings) plus a model-as-judge peer review to evaluate moral alignment in 20 large language models. We assess models on the World Values Survey (55 countries, 19 topics) and the PEW Global Attitudes Survey (39 countries, 8 topics). With EvalMORAAL, top models align closely with survey responses (Pearson's r approximately 0.90 on WVS). Yet we find a clear regional difference: Western regions average r=0.82 while non-Western regions average r=0.61 (a 0.21 absolute gap), indicating consistent regional bias. Our framework adds three parts: (1) two scoring methods for all models to enable fair comparison, (2) a structured chain-of-thought protocol with self-consistency checks, and (3) a model-as-judge peer review that flags 348 conflicts using a data-driven threshold. Peer agreement relates to survey alignment (WVS r=0.74, PEW r=0.39, both p<.001), supporting automated quality checks. These results show real progress toward culture-aware AI while highlighting open challenges for use across regions. 

---
# LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transparent Feature Selection 

**Authors**: Mohamed Bal-Ghaoui, Fayssal Sabri  

**Link**: [PDF](https://arxiv.org/pdf/2510.05935)  

**Abstract**: High-dimensional data remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. While Large Language Models (LLMs) have shown promise for dimensionality reduction through feature selection, existing LLM-based approaches frequently lack structured reasoning and transparent justification for their decisions. This paper introduces LLM-FS-Agent, a novel multi-agent architecture designed for interpretable and robust feature selection. The system orchestrates a deliberative "debate" among multiple LLM agents, each assigned a specific role, enabling collective evaluation of feature relevance and generation of detailed justifications. We evaluate LLM-FS-Agent in the cybersecurity domain using the CIC-DIAD 2024 IoT intrusion detection dataset and compare its performance against strong baselines, including LLM-Select and traditional methods such as PCA. Experimental results demonstrate that LLM-FS-Agent consistently achieves superior or comparable classification performance while reducing downstream training time by an average of 46% (statistically significant improvement, p = 0.028 for XGBoost). These findings highlight that the proposed deliberative architecture enhances both decision transparency and computational efficiency, establishing LLM-FS-Agent as a practical and reliable solution for real-world applications. 

---
# Carré du champ flow matching: better quality-generalisation tradeoff in generative models 

**Authors**: Jacob Bamberger, Iolo Jones, Dennis Duncan, Michael M. Bronstein, Pierre Vandergheynst, Adam Gosztolai  

**Link**: [PDF](https://arxiv.org/pdf/2510.05930)  

**Abstract**: Deep generative models often face a fundamental tradeoff: high sample quality can come at the cost of memorisation, where the model reproduces training data rather than generalising across the underlying data geometry. We introduce Carré du champ flow matching (CDC-FM), a generalisation of flow matching (FM), that improves the quality-generalisation tradeoff by regularising the probability path with a geometry-aware noise. Our method replaces the homogeneous, isotropic noise in FM with a spatially varying, anisotropic Gaussian noise whose covariance captures the local geometry of the latent data manifold. We prove that this geometric noise can be optimally estimated from the data and is scalable to large data. Further, we provide an extensive experimental evaluation on diverse datasets (synthetic manifolds, point clouds, single-cell genomics, animal motion capture, and images) as well as various neural network architectures (MLPs, CNNs, and transformers). We demonstrate that CDC-FM consistently offers a better quality-generalisation tradeoff. We observe significant improvements over standard FM in data-scarce regimes and in highly non-uniformly sampled datasets, which are often encountered in AI for science applications. Our work provides a mathematical framework for studying the interplay between data geometry, generalisation and memorisation in generative models, as well as a robust and scalable algorithm that can be readily integrated into existing flow matching pipelines. 

---
# An Attention-Augmented VAE-BiLSTM Framework for Anomaly Detection in 12-Lead ECG Signals 

**Authors**: Marc Garreta Basora, Mehmet Oguz Mulayim  

**Link**: [PDF](https://arxiv.org/pdf/2510.05919)  

**Abstract**: Anomaly detection in 12-lead electrocardiograms (ECGs) is critical for identifying deviations associated with cardiovascular disease. This work presents a comparative analysis of three autoencoder-based architectures: convolutional autoencoder (CAE), variational autoencoder with bidirectional long short-term memory (VAE-BiLSTM), and VAE-BiLSTM with multi-head attention (VAE-BiLSTM-MHA), for unsupervised anomaly detection in ECGs. To the best of our knowledge, this study reports the first application of a VAE-BiLSTM-MHA architecture to ECG anomaly detection. All models are trained on normal ECG samples to reconstruct non-anomalous cardiac morphology and detect deviations indicative of disease. Using a unified preprocessing and evaluation pipeline on the public China Physiological Signal Challenge (CPSC) dataset, the attention-augmented VAE achieves the best performance, with an AUPRC of 0.81 and a recall of 0.85 on the held-out test set, outperforming the other architectures. To support clinical triage, this model is further integrated into an interactive dashboard that visualizes anomaly localization. In addition, a performance comparison with baseline models from the literature is provided. 

---
# Kaputt: A Large-Scale Dataset for Visual Defect Detection 

**Authors**: Sebastian Höfer, Dorian Henning, Artemij Amiranashvili, Douglas Morrison, Mariliza Tzes, Ingmar Posner, Marc Matvienko, Alessandro Rennola, Anton Milan  

**Link**: [PDF](https://arxiv.org/pdf/2510.05903)  

**Abstract**: We present a novel large-scale dataset for defect detection in a logistics setting. Recent work on industrial anomaly detection has primarily focused on manufacturing scenarios with highly controlled poses and a limited number of object categories. Existing benchmarks like MVTec-AD [6] and VisA [33] have reached saturation, with state-of-the-art methods achieving up to 99.9% AUROC scores. In contrast to manufacturing, anomaly detection in retail logistics faces new challenges, particularly in the diversity and variability of object pose and appearance. Leading anomaly detection methods fall short when applied to this new setting. To bridge this gap, we introduce a new benchmark that overcomes the current limitations of existing datasets. With over 230,000 images (and more than 29,000 defective instances), it is 40 times larger than MVTec-AD and contains more than 48,000 distinct objects. To validate the difficulty of the problem, we conduct an extensive evaluation of multiple state-of-the-art anomaly detection methods, demonstrating that they do not surpass 56.96% AUROC on our dataset. Further qualitative analysis confirms that existing methods struggle to leverage normal samples under heavy pose and appearance variation. With our large-scale dataset, we set a new benchmark and encourage future research towards solving this challenging problem in retail logistics anomaly detection. The dataset is available for download under this https URL. 

---
# Paying Attention to Hybrid Attention: Untangling the Issues with Conversion Methods 

**Authors**: Martin Benfeghoul, Teresa Delgado, Adnan Oomerjee, Haitham Bou Ammar, Jun Wang, Zafeirios Fountas  

**Link**: [PDF](https://arxiv.org/pdf/2510.05901)  

**Abstract**: Transformers' quadratic computational complexity limits their scalability despite remarkable performance. While linear attention reduces this to linear complexity, pre-training such models from scratch remains, in most cases, prohibitively expensive. Recent post-training linearisation methods convert pre-trained Transformers to linear models efficiently, often using hybrid approaches that combine linear attention with sliding-window softmax. We identify a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on SWA. Component-level diagnostics reveal this previously undetected behaviour stems from overlooked evaluation practices on common-sense benchmarks. We propose three solutions to ensure balanced component usage: (i) inference-time hybridisation of linear-only conversions with sliding-window softmax; (ii) HedgeCATs, combining attention-weight transfer with targeted LoRA fine-tuning; and (iii) Scheduled Sliding-window Dropout (SSD), which stochastically suppresses the softmax branch during training to prevent component collapse. Our methods maintain computational efficiency while recovering most base model performance and ensuring genuine linear attention adoption, restoring the validity of performance attributions in hybrid conversions. 

---
# $\bf{D^3}$QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection 

**Authors**: Yanran Zhang, Bingyao Yu, Yu Zheng, Wenzhao Zheng, Yueqi Duan, Lei Chen, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05891)  

**Abstract**: The emergence of visual autoregressive (AR) models has revolutionized image generation while presenting new challenges for synthetic image detection. Unlike previous GAN or diffusion-based methods, AR models generate images through discrete token prediction, exhibiting both marked improvements in image synthesis quality and unique characteristics in their vector-quantized representations. In this paper, we propose to leverage Discrete Distribution Discrepancy-aware Quantization Error (D$^3$QE) for autoregressive-generated image detection that exploits the distinctive patterns and the frequency distribution bias of the codebook existing in real and fake images. We introduce a discrete distribution discrepancy-aware transformer that integrates dynamic codebook frequency statistics into its attention mechanism, fusing semantic features and quantization error latent. To evaluate our method, we construct a comprehensive dataset termed ARForensics covering 7 mainstream visual AR models. Experiments demonstrate superior detection accuracy and strong generalization of D$^3$QE across different AR models, with robustness to real-world perturbations. Code is available at \href{this https URL}{this https URL}. 

---
# Segment-Factorized Full-Song Generation on Symbolic Piano Music 

**Authors**: Ping-Yi Chen, Chih-Pin Tan, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05881)  

**Abstract**: We propose the Segmented Full-Song Model (SFS) for symbolic full-song generation. The model accepts a user-provided song structure and an optional short seed segment that anchors the main idea around which the song is developed. By factorizing a song into segments and generating each one through selective attention to related segments, the model achieves higher quality and efficiency compared to prior work. To demonstrate its suitability for human-AI interaction, we further wrap SFS into a web application that enables users to iteratively co-create music on a piano roll with customizable structures and flexible ordering. 

---
# Revisiting Long-context Modeling from Context Denoising Perspective 

**Authors**: Zecheng Tang, Baibei Ji, Juntao Li, Lijun Wu, Haijia Gui, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05862)  

**Abstract**: Long-context models (LCMs) have demonstrated great potential in processing long sequences, facilitating many real-world applications. The success of LCMs can be attributed to their ability to locate implicit critical information within the context for further prediction. However, recent research reveals that LCMs are often susceptible to contextual noise, i.e., irrelevant tokens, that can mislead model attention. In this paper, we conduct a fine-grained analysis of the context noise and propose an effective metric, the Integrated Gradient (IG) score, to detect and quantify the noise information within the context. Our findings reveal that even simple mitigation of detected context noise can substantially boost the model's attention on critical tokens and benefit subsequent predictions. Building on this insight, we propose Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. Extensive experiments across four tasks, under both context window scaling and long-context alignment settings, demonstrate the superiority of CDT. Notably, when trained with CDT, an open-source 8B model can achieve performance (50.92) comparable to GPT-4o (51.00). 

---
# DACP: Domain-Adaptive Continual Pre-Training of Large Language Models for Phone Conversation Summarization 

**Authors**: Xue-Yong Fu, Elena Khasanova, Md Tahmid Rahman Laskar, Harsh Saini, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2510.05858)  

**Abstract**: Large language models (LLMs) have achieved impressive performance in text summarization, yet their performance often falls short when applied to specialized domains %or conversational data that differ from their original pre-training distribution. While fine-tuning can improve summarization quality, it typically relies on costly and scarce high-quality labeled data. In this work, we explore continual pre-training as a scalable, self-supervised approach to adapt LLMs for downstream summarization tasks, particularly in the context of noisy real-world conversation transcripts. We conduct extensive experiments using large-scale, unlabeled business conversation data to investigate whether continual pre-training enhances model capabilities in conversational summarization. Our results demonstrate that continual pre-training yields substantial gains in both in-domain and out-of-domain summarization benchmarks, while maintaining strong generalization and robustness. We also analyze the effects of data selection strategies, providing practical guidelines for applying continual pre-training in summarization-focused industrial applications. 

---
# VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation 

**Authors**: Haoran Zhang, Shuanghao Bai, Wanqi Zhou, Yuedi Zhang, Qi Zhang, Pengxiang Ding, Cheng Chi, Donglin Wang, Badong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05827)  

**Abstract**: Robotic grasping is one of the most fundamental tasks in robotic manipulation, and grasp detection/generation has long been the subject of extensive research. Recently, language-driven grasp generation has emerged as a promising direction due to its practical interaction capabilities. However, most existing approaches either lack sufficient reasoning and generalization capabilities or depend on complex modular pipelines. Moreover, current grasp foundation models tend to overemphasize dialog and object semantics, resulting in inferior performance and restriction to single-object grasping. To maintain strong reasoning ability and generalization in cluttered environments, we propose VCoT-Grasp, an end-to-end grasp foundation model that incorporates visual chain-of-thought reasoning to enhance visual understanding for grasp generation. VCoT-Grasp adopts a multi-turn processing paradigm that dynamically focuses on visual inputs while providing interpretable reasoning traces. For training, we refine and introduce a large-scale dataset, VCoT-GraspSet, comprising 167K synthetic images with over 1.36M grasps, as well as 400+ real-world images with more than 1.2K grasps, annotated with intermediate bounding boxes. Extensive experiments on both VCoT-GraspSet and real robot demonstrate that our method significantly improves grasp success rates and generalizes effectively to unseen objects, backgrounds, and distractors. More details can be found at this https URL. 

---
# Mitigating Premature Exploitation in Particle-based Monte Carlo for Inference-Time Scaling 

**Authors**: Giorgio Giannone, Guangxuan Xu, Nikhil Shivakumar Nayak, Rohan Mahesh Awhad, Shivchander Sudalairaj, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.05825)  

**Abstract**: Inference-Time Scaling (ITS) improves language models by allocating more computation at generation time. Particle Filtering (PF) has emerged as a strong ITS method for complex mathematical reasoning tasks, but it is vulnerable when guided by process reward models, which often assign overconfident scores early in the reasoning process. This causes PF to suffer from premature exploitation: it myopically commits to locally promising trajectories, prunes potentially correct hypotheses, and converges to suboptimal solutions. This failure mode, known as particle impoverishment, is especially severe under constrained computational budgets. To address this, we analyze the problem and identify two root causes: a lack of diversity in the particle set due to overconfident resampling and consequent inability to assess the potential of a reasoning path. We introduce Entropic Particle Filtering (ePF), an algorithm that integrates two new techniques to solve these issues. The first technique, Entropic Annealing (EA), directly mitigates particle impoverishment by monitoring search diversity via entropy; when diversity drops, it intervenes by dynamically annealing the resampling distribution to preserve exploration. The second, an enhancement called Look-ahead Modulation (LaM), adds a predictive guide to evaluate a state's potential based on its successors. On several challenging math benchmarks, ePF significantly outperforms strong baselines and achieves up to a 50 % relative improvement in task reward. Together, these methods improve PF's resilience by balancing the exploration of diverse solution spaces with the exploitation of high-reward regions, ultimately leading to higher-quality solutions. 

---
# Deformable Image Registration for Self-supervised Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images 

**Authors**: Sven Koehler, Sarah Kaye Mueller, Jonathan Kiekenap, Gerald Greil, Tarique Hussain, Samir Sarikouch, Florian André, Norbert Frey, Sandy Engelhardt  

**Link**: [PDF](https://arxiv.org/pdf/2510.05819)  

**Abstract**: Cardiovascular magnetic resonance (CMR) is the gold standard for assessing cardiac function, but individual cardiac cycles complicate automatic temporal comparison or sub-phase analysis. Accurate cardiac keyframe detection can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion. We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber long-axis (4CH) cine CMR. Initially, dense deformable registration fields are derived from the images and used to compute a 1D motion descriptor, which provides valuable insights into global cardiac contraction and relaxation patterns. From these characteristic curves, keyframes are determined using a simple set of rules. The method was independently evaluated for both views using three public, multicentre, multidisease datasets. M&Ms-2 (n=360) dataset was used for training and evaluation, and M&Ms (n=345) and ACDC (n=100) datasets for repeatability control. Furthermore, generalisability to patients with rare congenital heart defects was tested using the German Competence Network (GCN) dataset. Our self-supervised approach achieved improved detection accuracy by 30% - 51% for SAX and 11% - 47% for 4CH in ED and ES, as measured by cyclic frame difference (cFD), compared with the volume-based approach. We can detect ED and ES, as well as three additional keyframes throughout the cardiac cycle with a mean cFD below 1.31 frames for SAX and 1.73 for LAX. Our approach enables temporally aligned inter- and intra-patient analysis of cardiac dynamics, irrespective of cycle or phase lengths. GitHub repository: this https URL 

---
# Risk level dependent Minimax Quantile lower bounds for Interactive Statistical Decision Making 

**Authors**: Raghav Bongole, Amirreza Zamani, Tobias J. Oechtering, Mikael Skoglund  

**Link**: [PDF](https://arxiv.org/pdf/2510.05808)  

**Abstract**: Minimax risk and regret focus on expectation, missing rare failures critical in safety-critical bandits and reinforcement learning. Minimax quantiles capture these tails. Three strands of prior work motivate this study: minimax-quantile bounds restricted to non-interactive estimation; unified interactive analyses that focus on expected risk rather than risk level specific quantile bounds; and high-probability bandit bounds that still lack a quantile-specific toolkit for general interactive protocols. To close this gap, within the interactive statistical decision making framework, we develop high-probability Fano and Le Cam tools and derive risk level explicit minimax-quantile bounds, including a quantile-to-expectation conversion and a tight link between strict and lower minimax quantiles. Instantiating these results for the two-armed Gaussian bandit immediately recovers optimal-rate bounds. 

---
# Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech 

**Authors**: Rikuto Kotoge, Yuichi Sasaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.05799)  

**Abstract**: Aligning text-to-speech (TTS) system outputs with human feedback through preference optimization has been shown to effectively improve the robustness and naturalness of language model-based TTS models. Current approaches primarily require paired desirable and undesirable samples at the utterance level. However, such pairs are often limited in TTS output data, and utterance-level formulation prevents fine-grained token-level optimization needed for accurate pronunciation alignment. In this study, we propose TKTO that eliminates the need for paired data, enabling a more data-efficient training paradigm, and directly targets token-level units, automatically providing fine-grained alignment signals without token-level annotations. TKTO improves the challenging Japanese TTS accuracy by 39% and reduces CER by 54%, automatically assigning 12.8 times stronger reward to targeted tokens. 

---
# Mellum: Production-Grade in-IDE Contextual Code Completion with Multi-File Project Understanding 

**Authors**: Nikita Pavlichenko, Iurii Nazarov, Ivan Dolgov, Ekaterina Garanina, Dmitry Ustalov, Ivan Bondyrev, Kseniia Lysaniuk, Evgeniia Vu, Kirill Chekmenev, Joseph Shtok, Yaroslav Golubev, Anton Semenkin, Uladzislau Sazanovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.05788)  

**Abstract**: We present the Mellum models family, open-weight code completion models designed for interactive use in JetBrains IDEs. Mellums have 4B parameters, adopt a Llama-style architecture, and are pre-trained on ~4T tokens of permissively licensed, multi-language code. Our studies show that (i) careful data curation and staged training significantly improve the model's quality, (ii) editor-critical capabilities such as context packing are necessary for high-quality suggestions, and (iii) a compact, task-focused model can meet the cost and latency constraints of interactive completion.
In the paper, we describe an end-to-end industrial pipeline for producing contextualized in-editor completion: disciplined data governance, multi-stage training that includes fill-in-the-middle and project context via supervised fine-tuning, and alignment via direct preference optimization using feedback from real-world scenarios. Our quality evaluations include both large-scale offline benchmarks and online telemetry from production deployments in JetBrains IDEs. Mellums are released under the Apache-2.0 license on HuggingFace, with a public model card providing a reproducible reference for practitioners. Our experience offers a pragmatic blueprint for taking a focused, open model from a research prototype to at scale production for hundreds of thousands of users. 

---
# InforME: Improving Informativeness of Abstractive Text Summarization With Informative Attention Guided by Named Entity Salience 

**Authors**: Jianbin Shen, Christy Jie Liang, Junyu Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.05769)  

**Abstract**: Abstractive text summarization is integral to the Big Data era, which demands advanced methods to turn voluminous and often long text data into concise but coherent and informative summaries for efficient human consumption. Despite significant progress, there is still room for improvement in various aspects. One such aspect is to improve informativeness. Hence, this paper proposes a novel learning approach consisting of two methods: an optimal transport-based informative attention method to improve learning focal information in reference summaries and an accumulative joint entropy reduction method on named entities to enhance informative salience. Experiment results show that our approach achieves better ROUGE scores compared to prior work on CNN/Daily Mail while having competitive results on XSum. Human evaluation of informativeness also demonstrates the better performance of our approach over a strong baseline. Further analysis gives insight into the plausible reasons underlying the evaluation results. 

---
# Are Heterogeneous Graph Neural Networks Truly Effective? A Causal Perspective 

**Authors**: Xiao Yang, Xuejiao Zhao, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05750)  

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in node classification. Building on this progress, heterogeneous graph neural networks (HGNNs) integrate relation types and node and edge semantics to leverage heterogeneous information. Causal analysis for HGNNs is advancing rapidly, aiming to separate genuine causal effects from spurious correlations. However, whether HGNNs are intrinsically effective remains underexamined, and most studies implicitly assume rather than establish this effectiveness. In this work, we examine HGNNs from two perspectives: model architecture and heterogeneous information. We conduct a systematic reproduction across 21 datasets and 20 baselines, complemented by comprehensive hyperparameter retuning. To further disentangle the source of performance gains, we develop a causal effect estimation framework that constructs and evaluates candidate factors under standard assumptions through factual and counterfactual analyses, with robustness validated via minimal sufficient adjustment sets, cross-method consistency checks, and sensitivity analyses. Our results lead to two conclusions. First, model architecture and complexity have no causal effect on performance. Second, heterogeneous information exerts a positive causal effect by increasing homophily and local-global distribution discrepancy, which makes node classes more distinguishable. The implementation is publicly available at this https URL. 

---
# Redefining Generalization in Visual Domains: A Two-Axis Framework for Fake Image Detection with FusionDetect 

**Authors**: Amirtaha Amanzadi, Zahra Dehghanian, Hamid Beigy, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2510.05740)  

**Abstract**: The rapid development of generative models has made it increasingly crucial to develop detectors that can reliably detect synthetic images. Although most of the work has now focused on cross-generator generalization, we argue that this viewpoint is too limited. Detecting synthetic images involves another equally important challenge: generalization across visual domains. To bridge this gap,we present the OmniGen Benchmark. This comprehensive evaluation dataset incorporates 12 state-of-the-art generators, providing a more realistic way of evaluating detector performance under realistic conditions. In addition, we introduce a new method, FusionDetect, aimed at addressing both vectors of generalization. FusionDetect draws on the benefits of two frozen foundation models: CLIP & Dinov2. By deriving features from both complementary models,we develop a cohesive feature space that naturally adapts to changes in both thecontent and design of the generator. Our extensive experiments demonstrate that FusionDetect delivers not only a new state-of-the-art, which is 3.87% more accurate than its closest competitor and 6.13% more precise on average on established benchmarks, but also achieves a 4.48% increase in accuracy on OmniGen,along with exceptional robustness to common image perturbations. We introduce not only a top-performing detector, but also a new benchmark and framework for furthering universal AI image detection. The code and dataset are available at this http URL 

---
# Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies 

**Authors**: Chunsan Hong, Seonho An, Min-Soo Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.05725)  

**Abstract**: Masked diffusion models (MDMs) have recently emerged as a novel framework for language modeling. MDMs generate sentences by iteratively denoising masked sequences, filling in [MASK] tokens step by step. Although MDMs support any-order sampling, performance is highly sensitive to the choice of which position to unmask next. Prior work typically relies on rule-based schedules (e.g., max-confidence, max-margin), which provide ad hoc improvements. In contrast, we replace these heuristics with a learned scheduler. Specifically, we cast denoising as a KL-regularized Markov decision process (MDP) with an explicit reference policy and optimize a regularized objective that admits policy improvement and convergence guarantees under standard assumptions. We prove that the optimized policy under this framework generates samples that more closely match the data distribution than heuristic schedules. Empirically, across four benchmarks, our learned policy consistently outperforms max-confidence: for example, on SUDOKU, where unmasking order is critical, it yields a 20.1% gain over random and a 11.2% gain over max-confidence. 

---
# Federated Split Learning for Resource-Constrained Robots in Industrial IoT: Framework Comparison, Optimization Strategies, and Future Directions 

**Authors**: Wanli Ni, Hui Tian, Shuai Wang, Chengyang Li, Lei Sun, Zhaohui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05713)  

**Abstract**: Federated split learning (FedSL) has emerged as a promising paradigm for enabling collaborative intelligence in industrial Internet of Things (IoT) systems, particularly in smart factories where data privacy, communication efficiency, and device heterogeneity are critical concerns. In this article, we present a comprehensive study of FedSL frameworks tailored for resource-constrained robots in industrial scenarios. We compare synchronous, asynchronous, hierarchical, and heterogeneous FedSL frameworks in terms of workflow, scalability, adaptability, and limitations under dynamic industrial conditions. Furthermore, we systematically categorize token fusion strategies into three paradigms: input-level (pre-fusion), intermediate-level (intra-fusion), and output-level (post-fusion), and summarize their respective strengths in industrial applications. We also provide adaptive optimization techniques to enhance the efficiency and feasibility of FedSL implementation, including model compression, split layer selection, computing frequency allocation, and wireless resource management. Simulation results validate the performance of these frameworks under industrial detection scenarios. Finally, we outline open issues and research directions of FedSL in future smart manufacturing systems. 

---
# FinReflectKG - EvalBench: Benchmarking Financial KG with Multi-Dimensional Evaluation 

**Authors**: Fabrizio Dimino, Abhinav Arun, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2510.05710)  

**Abstract**: Large language models (LLMs) are increasingly being used to extract structured knowledge from unstructured financial text. Although prior studies have explored various extraction methods, there is no universal benchmark or unified evaluation framework for the construction of financial knowledge graphs (KG). We introduce FinReflectKG - EvalBench, a benchmark and evaluation framework for KG extraction from SEC 10-K filings. Building on the agentic and holistic evaluation principles of FinReflectKG - a financial KG linking audited triples to source chunks from S&P 100 filings and supporting single-pass, multi-pass, and reflection-agent-based extraction modes - EvalBench implements a deterministic commit-then-justify judging protocol with explicit bias controls, mitigating position effects, leniency, verbosity and world-knowledge reliance. Each candidate triple is evaluated with binary judgments of faithfulness, precision, and relevance, while comprehensiveness is assessed on a three-level ordinal scale (good, partial, bad) at the chunk level. Our findings suggest that, when equipped with explicit bias controls, LLM-as-Judge protocols provide a reliable and cost-efficient alternative to human annotation, while also enabling structured error analysis. Reflection-based extraction emerges as the superior approach, achieving best performance in comprehensiveness, precision, and relevance, while single-pass extraction maintains the highest faithfulness. By aggregating these complementary dimensions, FinReflectKG - EvalBench enables fine-grained benchmarking and bias-aware evaluation, advancing transparency and governance in financial AI applications. 

---
# Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling 

**Authors**: Mary Llewellyn, Annie Gray, Josh Collyer, Michael Harries  

**Link**: [PDF](https://arxiv.org/pdf/2510.05709)  

**Abstract**: Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability. 

---
# Uncovering Representation Bias for Investment Decisions in Open-Source Large Language Models 

**Authors**: Fabrizio Dimino, Krati Saxena, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2510.05702)  

**Abstract**: Large Language Models are increasingly adopted in financial applications to support investment workflows. However, prior studies have seldom examined how these models reflect biases related to firm size, sector, or financial characteristics, which can significantly impact decision-making. This paper addresses this gap by focusing on representation bias in open-source Qwen models. We propose a balanced round-robin prompting method over approximately 150 U.S. equities, applying constrained decoding and token-logit aggregation to derive firm-level confidence scores across financial contexts. Using statistical tests and variance analysis, we find that firm size and valuation consistently increase model confidence, while risk factors tend to decrease it. Confidence varies significantly across sectors, with the Technology sector showing the greatest variability. When models are prompted for specific financial categories, their confidence rankings best align with fundamental data, moderately with technical signals, and least with growth indicators. These results highlight representation bias in Qwen models and motivate sector-aware calibration and category-conditioned evaluation protocols for safe and fair financial LLM deployment. 

---
# Membership Inference Attacks on Tokenizers of Large Language Models 

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.05699)  

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them. 

---
# Sparse deepfake detection promotes better disentanglement 

**Authors**: Antoine Teissier, Marie Tahon, Nicolas Dugué, Aghilas Sini  

**Link**: [PDF](https://arxiv.org/pdf/2510.05696)  

**Abstract**: Due to the rapid progress of speech synthesis, deepfake detection has become a major concern in the speech processing community. Because it is a critical task, systems must not only be efficient and robust, but also provide interpretable explanations. Among the different approaches for explainability, we focus on the interpretation of latent representations. In such paper, we focus on the last layer of embeddings of AASIST, a deepfake detection architecture. We use a TopK activation inspired by SAEs on this layer to obtain sparse representations which are used in the decision process. We demonstrate that sparse deepfake detection can improve detection performance, with an EER of 23.36% on ASVSpoof5 test set, with 95% of sparsity. We then show that these representations provide better disentanglement, using completeness and modularity metrics based on mutual information. Notably, some attacks are directly encoded in the latent space. 

---
# vAttention: Verified Sparse Attention 

**Authors**: Aditya Desai, Kumar Krishna Agrawal, Shuo Yang, Alejandro Cuadron, Luis Gaspar Schroeder, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2510.05688)  

**Abstract**: State-of-the-art sparse attention methods for reducing decoding latency fall into two main categories: approximate top-$k$ (and its extension, top-$p$) and recently introduced sampling-based estimation. However, these approaches are fundamentally limited in their ability to approximate full attention: they fail to provide consistent approximations across heads and query vectors and, most critically, lack guarantees on approximation quality, limiting their practical deployment. We observe that top-$k$ and random sampling are complementary: top-$k$ performs well when attention scores are dominated by a few tokens, whereas random sampling provides better estimates when attention scores are relatively uniform. Building on this insight and leveraging the statistical guarantees of sampling, we introduce vAttention, the first practical sparse attention mechanism with user-specified $(\epsilon, \delta)$ guarantees on approximation accuracy (thus, verified). These guarantees make vAttention a compelling step toward practical, reliable deployment of sparse attention at scale. By unifying top-k and sampling, vAttention outperforms both individually, delivering a superior quality-efficiency trade-off. Our experiments show that vAttention significantly improves the quality of sparse attention (e.g., $\sim$4.5 percentage points for Llama-3.1-8B-Inst and Deepseek-R1-Distill-Llama-8B on RULER-HARD), and effectively bridges the gap between full and sparse attention (e.g., across datasets, it matches full model quality with upto 20x sparsity). We also demonstrate that it can be deployed in reasoning scenarios to achieve fast decoding without compromising model quality (e.g., vAttention achieves full model quality on AIME2024 at 10x sparsity with up to 32K token generations). Code is open-sourced at this https URL. 

---
# QGraphLIME - Explaining Quantum Graph Neural Networks 

**Authors**: Haribandhu Jena, Jyotirmaya Shivottam, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2510.05683)  

**Abstract**: Quantum graph neural networks offer a powerful paradigm for learning on graph-structured data, yet their explainability is complicated by measurement-induced stochasticity and the combinatorial nature of graph structure. In this paper, we introduce QuantumGraphLIME (QGraphLIME), a model-agnostic, post-hoc framework that treats model explanations as distributions over local surrogates fit on structure-preserving perturbations of a graph. By aggregating surrogate attributions together with their dispersion, QGraphLIME yields uncertainty-aware node and edge importance rankings for quantum graph models. The framework further provides a distribution-free, finite-sample guarantee on the size of the surrogate ensemble: a Dvoretzky-Kiefer-Wolfowitz bound ensures uniform approximation of the induced distribution of a binary class probability at target accuracy and confidence under standard independence assumptions. Empirical studies on controlled synthetic graphs with known ground truth demonstrate accurate and stable explanations, with ablations showing clear benefits of nonlinear surrogate modeling and highlighting sensitivity to perturbation design. Collectively, these results establish a principled, uncertainty-aware, and structure-sensitive approach to explaining quantum graph neural networks, and lay the groundwork for scaling to broader architectures and real-world datasets, as quantum resources mature. Code is available at this https URL. 

---
# Verifier-free Test-Time Sampling for Vision Language Action Models 

**Authors**: Suhyeok Jang, Dongyoung Kim, Changyeon Kim, Youngsuk Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.05681)  

**Abstract**: Vision-Language-Action models (VLAs) have demonstrated remarkable performance in robot control. However, they remain fundamentally limited in tasks that require high precision due to their single-inference paradigm. While test-time scaling approaches using external verifiers have shown promise, they require additional training and fail to generalize to unseen conditions. We propose Masking Distribution Guided Selection (MG-Select), a novel test-time scaling framework for VLAs that leverages the model's internal properties without requiring additional training or external modules. Our approach utilizes KL divergence from a reference action token distribution as a confidence metric for selecting the optimal action from multiple candidates. We introduce a reference distribution generated by the same VLA but with randomly masked states and language conditions as inputs, ensuring maximum uncertainty while remaining aligned with the target task distribution. Additionally, we propose a joint training strategy that enables the model to learn both conditional and unconditional distributions by applying dropout to state and language conditions, thereby further improving the quality of the reference distribution. Our experiments demonstrate that MG-Select achieves significant performance improvements, including a 28%/35% improvement in real-world in-distribution/out-of-distribution tasks, along with a 168% relative gain on RoboCasa pick-and-place tasks trained with 30 demonstrations. 

---
# Code-Switching In-Context Learning for Cross-Lingual Transfer of Large Language Models 

**Authors**: Haneul Yoo, Jiho Jin, Kyunghyun Cho, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.05678)  

**Abstract**: While large language models (LLMs) exhibit strong multilingual abilities, their reliance on English as latent representations creates a translation barrier, where reasoning implicitly depends on internal translation into English. When this process fails, performance in non-English languages deteriorates sharply, limiting the inclusiveness of LLM-based applications. Existing cross-lingual in-context learning (X-ICL) methods primarily leverage monolingual demonstrations, often failing to mitigate this barrier and instead reinforcing it. In this work, we introduce code-switching in-context learning (CSICL), a simple yet effective prompting strategy that progressively transitions from a target language to English within demonstrations and instruction to facilitate their latent reasoning in English. By explicitly scaffolding the reasoning process through controlled code-switching, CSICL acts as an implicit linguistic bridge that enhances cross-lingual alignment and reduces reliance on the translation barrier. We conduct extensive experiments across 4 LLMs, 6 datasets, and 10 languages, spanning both knowledge-intensive and reasoning-oriented domains. Our results demonstrate that CSICL consistently outperforms X-ICL baselines, achieving gains of 3.1%p and 1.9%p in both target and unseen languages, respectively. The improvement is even more pronounced in low-resource settings, with gains of 14.7% in target and 5.3% in unseen languages. These findings establish code-switching as a principled and robust approach for overcoming the translation barrier during inference, moving LLMs toward more equitable and effective multilingual systems. 

---
# Quantifying the Accuracy-Interpretability Trade-Off in Concept-Based Sidechannel Models 

**Authors**: David Debot, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2510.05670)  

**Abstract**: Concept Bottleneck Models (CBNMs) are deep learning models that provide interpretability by enforcing a bottleneck layer where predictions are based exclusively on human-understandable concepts. However, this constraint also restricts information flow and often results in reduced predictive accuracy. Concept Sidechannel Models (CSMs) address this limitation by introducing a sidechannel that bypasses the bottleneck and carry additional task-relevant information. While this improves accuracy, it simultaneously compromises interpretability, as predictions may rely on uninterpretable representations transmitted through sidechannels. Currently, there exists no principled technique to control this fundamental trade-off. In this paper, we close this gap. First, we present a unified probabilistic concept sidechannel meta-model that subsumes existing CSMs as special cases. Building on this framework, we introduce the Sidechannel Independence Score (SIS), a metric that quantifies a CSM's reliance on its sidechannel by contrasting predictions made with and without sidechannel information. We propose SIS regularization, which explicitly penalizes sidechannel reliance to improve interpretability. Finally, we analyze how the expressivity of the predictor and the reliance of the sidechannel jointly shape interpretability, revealing inherent trade-offs across different CSM architectures. Empirical results show that state-of-the-art CSMs, when trained solely for accuracy, exhibit low representation interpretability, and that SIS regularization substantially improves their interpretability, intervenability, and the quality of learned interpretable task predictors. Our work provides both theoretical and practical tools for developing CSMs that balance accuracy and interpretability in a principled manner. 

---
# Ocular-Induced Abnormal Head Posture: Diagnosis and Missing Data Imputation 

**Authors**: Saja Al-Dabet, Sherzod Turaev, Nazar Zaki, Arif O. Khan, Luai Eldweik  

**Link**: [PDF](https://arxiv.org/pdf/2510.05649)  

**Abstract**: Ocular-induced abnormal head posture (AHP) is a compensatory mechanism that arises from ocular misalignment conditions, such as strabismus, enabling patients to reduce diplopia and preserve binocular vision. Early diagnosis minimizes morbidity and secondary complications such as facial asymmetry; however, current clinical assessments remain largely subjective and are further complicated by incomplete medical records. This study addresses both challenges through two complementary deep learning frameworks. First, AHP-CADNet is a multi-level attention fusion framework for automated diagnosis that integrates ocular landmarks, head pose features, and structured clinical attributes to generate interpretable predictions. Second, a curriculum learning-based imputation framework is designed to mitigate missing data by progressively leveraging structured variables and unstructured clinical notes to enhance diagnostic robustness under realistic data conditions. Evaluation on the PoseGaze-AHP dataset demonstrates robust diagnostic performance. AHP-CADNet achieves 96.9-99.0 percent accuracy across classification tasks and low prediction errors for continuous variables, with MAE ranging from 0.103 to 0.199 and R2 exceeding 0.93. The imputation framework maintains high accuracy across all clinical variables (93.46-99.78 percent with PubMedBERT), with clinical dependency modeling yielding significant improvements (p < 0.001). These findings confirm the effectiveness of both frameworks for automated diagnosis and recovery from missing data in clinical settings. 

---
# The African Languages Lab: A Collaborative Approach to Advancing Low-Resource African NLP 

**Authors**: Sheriff Issaka, Keyi Wang, Yinka Ajibola, Oluwatumininu Samuel-Ipaye, Zhaoyi Zhang, Nicte Aguillon Jimenez, Evans Kofi Agyei, Abraham Lin, Rohan Ramachandran, Sadick Abdul Mumin, Faith Nchifor, Mohammed Shuraim, Lieqi Liu, Erick Rosas Gonzalez, Sylvester Kpei, Jemimah Osei, Carlene Ajeneza, Persis Boateng, Prisca Adwoa Dufie Yeboah, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2510.05644)  

**Abstract**: Despite representing nearly one-third of the world's languages, African languages remain critically underserved by modern NLP technologies, with 88\% classified as severely underrepresented or completely ignored in computational linguistics. We present the African Languages Lab (All Lab), a comprehensive research initiative that addresses this technological gap through systematic data collection, model development, and capacity building. Our contributions include: (1) a quality-controlled data collection pipeline, yielding the largest validated African multi-modal speech and text dataset spanning 40 languages with 19 billion tokens of monolingual text and 12,628 hours of aligned speech data; (2) extensive experimental validation demonstrating that our dataset, combined with fine-tuning, achieves substantial improvements over baseline models, averaging +23.69 ChrF++, +0.33 COMET, and +15.34 BLEU points across 31 evaluated languages; and (3) a structured research program that has successfully mentored fifteen early-career researchers, establishing sustainable local capacity. Our comparative evaluation against Google Translate reveals competitive performance in several languages while identifying areas that require continued development. 

---
# From Neural Activity to Computation: Biological Reservoirs for Pattern Recognition in Digit Classification 

**Authors**: Ludovico Iannello, Luca Ciampi, Fabrizio Tonelli, Gabriele Lagani, Lucio Maria Calcagnile, Federico Cremisi, Angelo Di Garbo, Giuseppe Amato  

**Link**: [PDF](https://arxiv.org/pdf/2510.05637)  

**Abstract**: In this paper, we present a biologically grounded approach to reservoir computing (RC), in which a network of cultured biological neurons serves as the reservoir substrate. This system, referred to as biological reservoir computing (BRC), replaces artificial recurrent units with the spontaneous and evoked activity of living neurons. A multi-electrode array (MEA) enables simultaneous stimulation and readout across multiple sites: inputs are delivered through a subset of electrodes, while the remaining ones capture the resulting neural responses, mapping input patterns into a high-dimensional biological feature space. We evaluate the system through a case study on digit classification using a custom dataset. Input images are encoded and delivered to the biological reservoir via electrical stimulation, and the corresponding neural activity is used to train a simple linear classifier. To contextualize the performance of the biological system, we also include a comparison with a standard artificial reservoir trained on the same task. The results indicate that the biological reservoir can effectively support classification, highlighting its potential as a viable and interpretable computational substrate. We believe this work contributes to the broader effort of integrating biological principles into machine learning and aligns with the goals of human-inspired vision by exploring how living neural systems can inform the design of efficient and biologically plausible models. 

---
# Beyond Spectral Peaks: Interpreting the Cues Behind Synthetic Image Detection 

**Authors**: Sara Mandelli, Diego Vila-Portela, David Vázquez-Padín, Paolo Bestagini, Fernando Pérez-González  

**Link**: [PDF](https://arxiv.org/pdf/2510.05633)  

**Abstract**: Over the years, the forensics community has proposed several deep learning-based detectors to mitigate the risks of generative AI. Recently, frequency-domain artifacts (particularly periodic peaks in the magnitude spectrum), have received significant attention, as they have been often considered a strong indicator of synthetic image generation. However, state-of-the-art detectors are typically used as black-boxes, and it still remains unclear whether they truly rely on these peaks. This limits their interpretability and trust. In this work, we conduct a systematic study to address this question. We propose a strategy to remove spectral peaks from images and analyze the impact of this operation on several detectors. In addition, we introduce a simple linear detector that relies exclusively on frequency peaks, providing a fully interpretable baseline free from the confounding influence of deep learning. Our findings reveal that most detectors are not fundamentally dependent on spectral peaks, challenging a widespread assumption in the field and paving the way for more transparent and reliable forensic tools. 

---
# Generative AI-Driven Hierarchical Multi-Agent Framework for Zero-Touch Optical Networks 

**Authors**: Yao Zhang, Yuchen Song, Shengnan Li, Yan Shi, Shikui Shen, Xiongyan Tang, Min Zhang, Danshi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05625)  

**Abstract**: The rapid development of Generative Artificial Intelligence (GenAI) has catalyzed a transformative technological revolution across all walks of life. As the backbone of wideband communication, optical networks are expecting high-level autonomous operation and zero-touch management to accommodate their expanding network scales and escalating transmission bandwidth. The integration of GenAI is deemed as the pivotal solution for realizing zero-touch optical networks. However, the lifecycle management of optical networks involves a multitude of tasks and necessitates seamless collaboration across multiple layers, which poses significant challenges to the existing single-agent GenAI systems. In this paper, we propose a GenAI-driven hierarchical multi-agent framework designed to streamline multi-task autonomous execution for zero-touch optical networks. We present the architecture, implementation, and applications of this framework. A field-deployed mesh network is utilized to demonstrate three typical scenarios throughout the lifecycle of optical network: quality of transmission estimation in the planning stage, dynamic channel adding/dropping in the operation stage, and system capacity increase in the upgrade stage. The case studies, illustrate the capabilities of multi-agent framework in multi-task allocation, coordination, execution, evaluation, and summarization. This work provides a promising approach for the future development of intelligent, efficient, and collaborative network management solutions, paving the way for more specialized and adaptive zero-touch optical networks. 

---
# Monte Carlo-Type Neural Operator for Differential Equations 

**Authors**: Salah Eddine Choutri, Prajwal Chauhan, Othmane Mazhar, Saif Eddin Jabari  

**Link**: [PDF](https://arxiv.org/pdf/2510.05620)  

**Abstract**: The Monte Carlo-type Neural Operator (MCNO) introduces a framework for learning solution operators of one-dimensional partial differential equations (PDEs) by directly learning the kernel function and approximating the associated integral operator using a Monte Carlo-type approach. Unlike Fourier Neural Operators (FNOs), which rely on spectral representations and assume translation-invariant kernels, MCNO makes no such assumptions. The kernel is represented as a learnable tensor over sampled input-output pairs, and sampling is performed once, uniformly at random from a discretized grid. This design enables generalization across multiple grid resolutions without relying on fixed global basis functions or repeated sampling during training, while an interpolation step maps between arbitrary input and output grids to further enhance flexibility. Experiments on standard 1D PDE benchmarks show that MCNO achieves competitive accuracy with efficient computational cost. We also provide a theoretical analysis proving that the Monte Carlo estimator yields a bounded bias and variance under mild regularity assumptions. This result holds in any spatial dimension, suggesting that MCNO may extend naturally beyond one-dimensional problems. More broadly, this work explores how Monte Carlo-type integration can be incorporated into neural operator frameworks for continuous-domain PDEs, providing a theoretically supported alternative to spectral methods (such as FNO) and to graph-based Monte Carlo approaches (such as the Graph Kernel Neural Operator, GNO). 

---
# PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction 

**Authors**: Ziqiao Meng, Qichao Wang, Zhiyang Dou, Zixing Song, Zhipeng Zhou, Irwin King, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05613)  

**Abstract**: Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential. 

---
# MADIAVE: Multi-Agent Debate for Implicit Attribute Value Extraction 

**Authors**: Wei-Chieh Huang, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2510.05611)  

**Abstract**: Implicit Attribute Value Extraction (AVE) is essential for accurately representing products in e-commerce, as it infers lantent attributes from multimodal data. Despite advances in multimodal large language models (MLLMs), implicit AVE remains challenging due to the complexity of multidimensional data and gaps in vision-text understanding. In this work, we introduce \textsc{\modelname}, a multi-agent debate framework that employs multiple MLLM agents to iteratively refine inferences. Through a series of debate rounds, agents verify and update each other's responses, thereby improving inference performance and robustness. Experiments on the ImplicitAVE dataset demonstrate that even a few rounds of debate significantly boost accuracy, especially for attributes with initially low performance. We systematically evaluate various debate configurations, including identical or different MLLM agents, and analyze how debate rounds affect convergence dynamics. Our findings highlight the potential of multi-agent debate strategies to address the limitations of single-agent approaches and offer a scalable solution for implicit AVE in multimodal e-commerce. 

---
# HOI-R1: Exploring the Potential of Multimodal Large Language Models for Human-Object Interaction Detection 

**Authors**: Junwen Chen, Peilin Xiong, Keiji Yanai  

**Link**: [PDF](https://arxiv.org/pdf/2510.05609)  

**Abstract**: Recent Human-object interaction detection (HOID) methods highly require prior knowledge from VLMs to enhance the interaction recognition capabilities. The training strategies and model architectures for connecting the knowledge from VLMs to the HOI instance representations from the object detector are challenging, and the whole framework is complex for further development or application. On the other hand, the inherent reasoning abilities of MLLMs on human-object interaction detection are under-explored. Inspired by the recent success of training MLLMs with reinforcement learning (RL) methods, we propose HOI-R1 and first explore the potential of the language model on the HOID task without any additional detection modules. We introduce an HOI reasoning process and HOID reward functions to solve the HOID task by pure text. The results on the HICO-DET dataset show that HOI-R1 achieves 2x the accuracy of the baseline with great generalization ability. The source code is available at this https URL. 

---
# AutoPentester: An LLM Agent-based Framework for Automated Pentesting 

**Authors**: Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2510.05605)  

**Abstract**: Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT. 

---
# AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents 

**Authors**: Mingdai Yang, Nurendra Choudhary, Jiangshu Du, Edward W.Huang, Philip S.Yu, Karthik Subbian, Danai Kourta  

**Link**: [PDF](https://arxiv.org/pdf/2510.05598)  

**Abstract**: Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness. 

---
# Improving Chain-of-Thought Efficiency for Autoregressive Image Generation 

**Authors**: Zeqi Gu, Markos Georgopoulos, Xiaoliang Dai, Marjan Ghazvininejad, Chu Wang, Felix Juefei-Xu, Kunpeng Li, Yujun Shi, Zecheng He, Zijian He, Jiawei Zhou, Abe Davis, Jialiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05593)  

**Abstract**: Autoregressive multimodal large language models have recently gained popularity for image generation, driven by advances in foundation models. To enhance alignment and detail, newer approaches employ chain-of-thought (CoT) reasoning, expanding user inputs into elaborated prompts prior to image synthesis. However, this strategy can introduce unnecessary redundancy -- a phenomenon we call visual overthinking -- which increases computational costs and can introduce details that contradict the original prompt. In this work, we explore how to generate more concise CoT sequences for more efficient image generation. We introduce ShortCoTI, a lightweight optimization framework that encourages more concise CoT while preserving output image quality. ShortCoTI rewards more concise prompts with an adaptive function that scales according to an estimated difficulty for each task. Incorporating this reward into a reinforcement learning paradigm reduces prompt reasoning length by 54% while maintaining or slightly improving quality metrics across multiple benchmarks (T2I-CompBench, GenEval). Qualitative analysis shows that our method eliminates verbose explanations and repetitive refinements, producing reasoning prompts that are both concise and semantically rich. As a result, ShortCoTI improves computational efficiency without compromising the fidelity or visual appeal of generated images. 

---
# Deciphering Invariant Feature Decoupling in Source-free Time Series Forecasting with Proxy Denoising 

**Authors**: Kangjia Yan, Chenxi Liu, Hao Miao, Xinle Wu, Yan Zhao, Chenjuan Guo, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05589)  

**Abstract**: The proliferation of mobile devices generates a massive volume of time series across various domains, where effective time series forecasting enables a variety of real-world applications. This study focuses on a new problem of source-free domain adaptation for time series forecasting. It aims to adapt a pretrained model from sufficient source time series to the sparse target time series domain without access to the source data, embracing data protection regulations. To achieve this, we propose TimePD, the first source-free time series forecasting framework with proxy denoising, where large language models (LLMs) are employed to benefit from their generalization capabilities. Specifically, TimePD consists of three key components: (1) dual-branch invariant disentangled feature learning that enforces representation- and gradient-wise invariance by means of season-trend decomposition; (2) lightweight, parameter-free proxy denoising that dynamically calibrates systematic biases of LLMs; and (3) knowledge distillation that bidirectionally aligns the denoised prediction and the original target prediction. Extensive experiments on real-world datasets offer insight into the effectiveness of the proposed TimePD, outperforming SOTA baselines by 9.3% on average. 

---
# Domain-Shift-Aware Conformal Prediction for Large Language Models 

**Authors**: Zhexiao Lin, Yuanyuan Li, Neeraj Sarna, Yuanyuan Gao, Michael von Gablenz  

**Link**: [PDF](https://arxiv.org/pdf/2510.05566)  

**Abstract**: Large language models have achieved impressive performance across diverse tasks. However, their tendency to produce overconfident and factually incorrect outputs, known as hallucinations, poses risks in real world applications. Conformal prediction provides finite-sample, distribution-free coverage guarantees, but standard conformal prediction breaks down under domain shift, often leading to under-coverage and unreliable prediction sets. We propose a new framework called Domain-Shift-Aware Conformal Prediction (DS-CP). Our framework adapts conformal prediction to large language models under domain shift, by systematically reweighting calibration samples based on their proximity to the test prompt, thereby preserving validity while enhancing adaptivity. Our theoretical analysis and experiments on the MMLU benchmark demonstrate that the proposed method delivers more reliable coverage than standard conformal prediction, especially under substantial distribution shifts, while maintaining efficiency. This provides a practical step toward trustworthy uncertainty quantification for large language models in real-world deployment. 

---
# Generative Dynamic Graph Representation Learning for Conspiracy Spoofing Detection 

**Authors**: Sheng Xiang, Yidong Jiang, Yunting Chen, Dawei Cheng, Guoping Zhao, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05562)  

**Abstract**: Spoofing detection in financial trading is crucial, especially for identifying complex behaviors such as conspiracy spoofing. Traditional machine-learning approaches primarily focus on isolated node features, often overlooking the broader context of interconnected nodes. Graph-based techniques, particularly Graph Neural Networks (GNNs), have advanced the field by leveraging relational information effectively. However, in real-world spoofing detection datasets, trading behaviors exhibit dynamic, irregular patterns. Existing spoofing detection methods, though effective in some scenarios, struggle to capture the complexity of dynamic and diverse, evolving inter-node relationships. To address these challenges, we propose a novel framework called the Generative Dynamic Graph Model (GDGM), which models dynamic trading behaviors and the relationships among nodes to learn representations for conspiracy spoofing detection. Specifically, our approach incorporates the generative dynamic latent space to capture the temporal patterns and evolving market conditions. Raw trading data is first converted into time-stamped sequences. Then we model trading behaviors using the neural ordinary differential equations and gated recurrent units, to generate the representation incorporating temporal dynamics of spoofing patterns. Furthermore, pseudo-label generation and heterogeneous aggregation techniques are employed to gather relevant information and enhance the detection performance for conspiratorial spoofing behaviors. Experiments conducted on spoofing detection datasets demonstrate that our approach outperforms state-of-the-art models in detection accuracy. Additionally, our spoofing detection system has been successfully deployed in one of the largest global trading markets, further validating the practical applicability and performance of the proposed method. 

---
# Critical attention scaling in long-context transformers 

**Authors**: Shi Chen, Zhengjiang Lin, Yury Polyanskiy, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2510.05554)  

**Abstract**: As large language models scale to longer contexts, attention layers suffer from a fundamental pathology: attention scores collapse toward uniformity as context length $n$ increases, causing tokens to cluster excessively, a phenomenon known as rank-collapse. While $\textit{attention scaling}$ effectively addresses this deficiency by rescaling attention scores with a polylogarithmic factor $\beta_n$, theoretical justification for this approach remains lacking.
We analyze a simplified yet tractable model that magnifies the effect of attention scaling. In this model, attention exhibits a phase transition governed by the scaling factor $\beta_n$: insufficient scaling collapses all tokens to a single direction, while excessive scaling reduces attention to identity, thereby eliminating meaningful interactions between tokens. Our main result identifies the critical scaling $\beta_n \asymp \log n$ and provides a rigorous justification for attention scaling in YaRN and Qwen, clarifying why logarithmic scaling maintains sparse, content-adaptive attention at large context lengths. 

---
# Seeing the Big Picture: Evaluating Multimodal LLMs' Ability to Interpret and Grade Handwritten Student Work 

**Authors**: Owen Henkel, Bill Roberts, Doug Jaffe, Laurence Holt  

**Link**: [PDF](https://arxiv.org/pdf/2510.05538)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) raise the question of their potential for grading, analyzing, and offering feedback on handwritten student classwork. This capability would be particularly beneficial in elementary and middle-school mathematics education, where most work remains handwritten, because seeing students' full working of a problem provides valuable insights into their learning processes, but is extremely time-consuming to grade. We present two experiments investigating MLLM performance on handwritten student mathematics classwork. Experiment A examines 288 handwritten responses from Ghanaian middle school students solving arithmetic problems with objective answers. In this context, models achieved near-human accuracy (95%, k = 0.90) but exhibited occasional errors that human educators would be unlikely to make. Experiment B evaluates 150 mathematical illustrations from American elementary students, where the drawings are the answer to the question. These tasks lack single objective answers and require sophisticated visual interpretation as well as pedagogical judgment in order to analyze and evaluate them. We attempted to separate MLLMs' visual capabilities from their pedagogical abilities by first asking them to grade the student illustrations directly, and then by augmenting the image with a detailed human description of the illustration. We found that when the models had to analyze the student illustrations directly, they struggled, achieving only k = 0.20 with ground truth scores, but when given human descriptions, their agreement levels improved dramatically to k = 0.47, which was in line with human-to-human agreement levels. This gap suggests MLLMs can "see" and interpret arithmetic work relatively well, but still struggle to "see" student mathematical illustrations. 

---
# Permutation-Invariant Representation Learning for Robust and Privacy-Preserving Feature Selection 

**Authors**: Rui Liu, Tao Zhe, Yanjie Fu, Feng Xia, Ted Senator, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05535)  

**Abstract**: Feature selection eliminates redundancy among features to improve downstream task performance while reducing computational overhead. Existing methods often struggle to capture intricate feature interactions and adapt across diverse application scenarios. Recent advances employ generative intelligence to alleviate these drawbacks. However, these methods remain constrained by permutation sensitivity in embedding and reliance on convexity assumptions in gradient-based search. To address these limitations, our initial work introduces a novel framework that integrates permutation-invariant embedding with policy-guided search. Although effective, it still left opportunities to adapt to realistic distributed scenarios. In practice, data across local clients is highly imbalanced, heterogeneous and constrained by strict privacy regulations, limiting direct sharing. These challenges highlight the need for a framework that can integrate feature selection knowledge across clients without exposing sensitive information. In this extended journal version, we advance the framework from two perspectives: 1) developing a privacy-preserving knowledge fusion strategy to derive a unified representation space without sharing sensitive raw data. 2) incorporating a sample-aware weighting strategy to address distributional imbalance among heterogeneous local clients. Extensive experiments validate the effectiveness, robustness, and efficiency of our framework. The results further demonstrate its strong generalization ability in federated learning scenarios. The code and data are publicly available: this https URL. 

---
# Provably Mitigating Corruption, Overoptimization, and Verbosity Simultaneously in Offline and Online RLHF/DPO Alignment 

**Authors**: Ziyi Chen, Junyi Li, Peiran Yu, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05526)  

**Abstract**: Reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) are important techniques to align large language models (LLM) with human preference. However, the quality of RLHF and DPO training is seriously compromised by \textit{\textbf{C}orrupted} preference, reward \textit{\textbf{O}veroptimization}, and bias towards \textit{\textbf{V}erbosity}. To our knowledge, most existing works tackle only one of these important issues, and the few other works require much computation to estimate multiple reward models and lack theoretical guarantee of generalization ability. In this work, we propose RLHF-\textbf{COV} and DPO-\textbf{COV} algorithms that can simultaneously mitigate these three issues, in both offline and online settings. This ability is theoretically demonstrated by obtaining length-regularized generalization error rates for our DPO-COV algorithms trained on corrupted data, which match the best-known rates for simpler cases with clean data and without length regularization. Moreover, our DPO-COV algorithm is simple to implement without reward estimation, and is proved to be equivalent to our RLHF-COV algorithm, which directly implies the equivalence between the vanilla RLHF and DPO algorithms. Experiments demonstrate the effectiveness of our DPO-COV algorithms under both offline and online settings. 

---
# CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension 

**Authors**: Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen, Quanyu Dai, Zhenhua Dong, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05520)  

**Abstract**: Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification. 

---
# Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting 

**Authors**: Zhongkai Yu, Yue Guan, Zihao Yu, Chenyang Zhou, Shuyi Pei, Yangwook Kang, Yufei Ding, Po-An Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2510.05497)  

**Abstract**: Large Language Models (LLMs) with Mixture of Experts (MoE) architectures achieve remarkable performance improvements, but their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit serving systems. To forecast the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across three state-of-the-art large-scale MoE models (200B- 671B) using over 24,000 requests spanning diverse workloads. With the resulting 150GB+ trace files, we perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. Taking wafer-scale GPUs as a case study, we demonstrate that minor architectural modifications leveraging our insights achieve substantial performance gains, delivering 6.3X and 4.0X average speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first comprehensive data-centric analysis of MoE models at scale. Our profiling traces and analysis results are publicly available at {this https URL. We will also release our simulation framework shortly to facilitate future research in this area. 

---
# High-Fidelity Synthetic ECG Generation via Mel-Spectrogram Informed Diffusion Training 

**Authors**: Zhuoyi Huang, Nutan Sahoo, Anamika Kumari, Girish Kumar, Kexuan Cai, Shixing Cao, Yue Kang, Tian Xia, Somya Chatterjee, Nicholas Hausman, Aidan Jay, Eric S. Rosenthal, Soundar Srinivasan, Sadid Hasan, Alex Fedorov, Sulaiman Vesal, Soundar Srinivasan, Sadid Hasan, Alex Fedorov, Sulaiman Vesal  

**Link**: [PDF](https://arxiv.org/pdf/2510.05492)  

**Abstract**: The development of machine learning for cardiac care is severely hampered by privacy restrictions on sharing real patient electrocardiogram (ECG) data. Although generative AI offers a promising solution, the real-world use of existing model-synthesized ECGs is limited by persistent gaps in trustworthiness and clinical utility. In this work, we address two major shortcomings of current generative ECG methods: insufficient morphological fidelity and the inability to generate personalized, patient-specific physiological signals. To address these gaps, we build on a conditional diffusion-based Structured State Space Model (SSSD-ECG) with two principled innovations: (1) MIDT-ECG (Mel-Spectrogram Informed Diffusion Training), a novel training paradigm with time-frequency domain supervision to enforce physiological structural realism, and (2) multi-modal demographic conditioning to enable patient-specific synthesis. We comprehensively evaluate our approach on the PTB-XL dataset, assessing the synthesized ECG signals on fidelity, clinical coherence, privacy preservation, and downstream task utility. MIDT-ECG achieves substantial gains: it improves morphological coherence, preserves strong privacy guarantees with all metrics evaluated exceeding the baseline by 4-8%, and notably reduces the interlead correlation error by an average of 74%, while demographic conditioning enhances signal-to-noise ratio and personalization. In critical low-data regimes, a classifier trained on datasets supplemented with our synthetic ECGs achieves performance comparable to a classifier trained solely on real data. Together, we demonstrate that ECG synthesizers, trained with the proposed time-frequency structural regularization scheme, can serve as personalized, high-fidelity, privacy-preserving surrogates when real data are scarce, advancing the responsible use of generative AI in healthcare. 

---
# LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation 

**Authors**: Zhoutong Fu, Yihan Cao, Yi-Lin Chen, Aman Lunia, Liming Dong, Neha Saraf, Ruijie Jiang, Yun Dai, Qingquan Song, Tan Wang, Guoyao Li, Derek Koh, Haichao Wei, Zhipeng Wang, Aman Gupta, Chengming Jiang, Jianqiang Shen, Liangjie Hong, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05490)  

**Abstract**: Large language models (LLMs) have achieved strong performance across a wide range of natural language processing tasks. However, deploying LLMs at scale for domain specific applications, such as job-person fit and explanation in job seeking platforms, introduces distinct challenges. At LinkedIn, the job person fit task requires analyzing a candidate's public profile against job requirements to produce both a fit assessment and a detailed explanation. Directly applying open source or finetuned LLMs to this task often fails to yield high quality, actionable feedback due to the complexity of the domain and the need for structured outputs. Moreover, the large size of these models leads to high inference latency and limits scalability, making them unsuitable for online use. To address these challenges, we introduce LANTERN, a novel LLM knowledge distillation framework tailored specifically for job person fit tasks. LANTERN involves modeling over multiple objectives, an encoder model for classification purpose, and a decoder model for explanation purpose. To better distill the knowledge from a strong black box teacher model to multiple downstream models, LANTERN incorporates multi level knowledge distillation that integrates both data and logit level insights. In addition to introducing the knowledge distillation framework, we share our insights on post training techniques and prompt engineering, both of which are crucial for successfully adapting LLMs to domain specific downstream tasks. Extensive experimental results demonstrate that LANTERN significantly improves task specific metrics for both job person fit and explanation. Online evaluations further confirm its effectiveness, showing measurable gains in job seeker engagement, including a 0.24\% increase in apply rate and a 0.28\% increase in qualified applications. 

---
# AMAQ: Adaptive Mixed-bit Activation Quantization for Collaborative Parameter Efficient Fine-tuning 

**Authors**: Yurun Song, Zhuoyi Yang, Ian G. Harris, Sangeetha Abdu Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05468)  

**Abstract**: Large Language Models (LLMs) are scaling rapidly, creating significant challenges for collaborative server client distributed training, particularly in terms of communication efficiency and computational overheads. To address these challenges, we implement Parameter-efficient Split Learning, which effectively balances efficiency and performance for collaborative training on low-resource devices.
To reduce communication overhead in collaborative training, we introduce Adaptive Mixed bit Activation Quantization (AMAQ), a strategy that progressively compresses activations and gradients from high precision (6 to 8 bits) to low precision (3 to 4 bits). AMAQ achieves this by effectively allocating bit budgets across channels based on feature wise and layer wise importance using bit regularization.
Under the same bit budgets, AMAQ outperforms fixed-precision approaches, delivering about 2.5% higher generation accuracy and about 1.3% better classification accuracy for models like LLaMA3 8B and Qwen2.5 7B. In addition, it significantly enhances training stability and reducing ultra-low bit representation collapse during the training.
Experiments demonstrate that AMAQ integrates effectively into practical multi-machine collaborative training setups, offering superior inference accuracy with only a modest communication overhead for bits adaptation during training. This trade off makes AMAQ a practical and effective solution for collaborative training with minimal communication cost. 

---
# QDeepGR4J: Quantile-based ensemble of deep learning and GR4J hybrid rainfall-runoff models for extreme flow prediction with uncertainty quantification 

**Authors**: Arpit Kapoor, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2510.05453)  

**Abstract**: Conceptual rainfall-runoff models aid hydrologists and climate scientists in modelling streamflow to inform water management practices. Recent advances in deep learning have unravelled the potential for combining hydrological models with deep learning models for better interpretability and improved predictive performance. In our previous work, we introduced DeepGR4J, which enhanced the GR4J conceptual rainfall-runoff model using a deep learning model to serve as a surrogate for the routing component. DeepGR4J had an improved rainfall-runoff prediction accuracy, particularly in arid catchments. Quantile regression models have been extensively used for quantifying uncertainty while aiding extreme value forecasting. In this paper, we extend DeepGR4J using a quantile regression-based ensemble learning framework to quantify uncertainty in streamflow prediction. We also leverage the uncertainty bounds to identify extreme flow events potentially leading to flooding. We further extend the model to multi-step streamflow predictions for uncertainty bounds. We design experiments for a detailed evaluation of the proposed framework using the CAMELS-Aus dataset. The results show that our proposed Quantile DeepGR4J framework improves the predictive accuracy and uncertainty interval quality (interval score) compared to baseline deep learning models. Furthermore, we carry out flood risk evaluation using Quantile DeepGR4J, and the results demonstrate its suitability as an early warning system. 

---
# Adversarial Reinforcement Learning for Large Language Model Agent Safety 

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser  

**Link**: [PDF](https://arxiv.org/pdf/2510.05442)  

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model. 

---
# UnitTenX: Generating Tests for Legacy Packages with AI Agents Powered by Formal Verification 

**Authors**: Yiannis Charalambous, Claudionor N. Coelho Jr, Luis Lamb, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2510.05441)  

**Abstract**: This paper introduces UnitTenX, a state-of-the-art open-source AI multi-agent system designed to generate unit tests for legacy code, enhancing test coverage and critical value testing. UnitTenX leverages a combination of AI agents, formal methods, and Large Language Models (LLMs) to automate test generation, addressing the challenges posed by complex and legacy codebases. Despite the limitations of LLMs in bug detection, UnitTenX offers a robust framework for improving software reliability and maintainability. Our results demonstrate the effectiveness of this approach in generating high-quality tests and identifying potential issues. Additionally, our approach enhances the readability and documentation of legacy code. 

---
# Physics-Informed Machine Learning in Biomedical Science and Engineering 

**Authors**: Nazanin Ahmadi, Qianying Cao, Jay D. Humphrey, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.05433)  

**Abstract**: Physics-informed machine learning (PIML) is emerging as a potentially transformative paradigm for modeling complex biomedical systems by integrating parameterized physical laws with data-driven methods. Here, we review three main classes of PIML frameworks: physics-informed neural networks (PINNs), neural ordinary differential equations (NODEs), and neural operators (NOs), highlighting their growing role in biomedical science and engineering. We begin with PINNs, which embed governing equations into deep learning models and have been successfully applied to biosolid and biofluid mechanics, mechanobiology, and medical imaging among other areas. We then review NODEs, which offer continuous-time modeling, especially suited to dynamic physiological systems, pharmacokinetics, and cell signaling. Finally, we discuss deep NOs as powerful tools for learning mappings between function spaces, enabling efficient simulations across multiscale and spatially heterogeneous biological domains. Throughout, we emphasize applications where physical interpretability, data scarcity, or system complexity make conventional black-box learning insufficient. We conclude by identifying open challenges and future directions for advancing PIML in biomedical science and engineering, including issues of uncertainty quantification, generalization, and integration of PIML and large language models. 

---
# Exploring Student Choice and the Use of Multimodal Generative AI in Programming Learning 

**Authors**: Xinying Hou, Ruiwei Xiao, Runlong Ye, Michael Liut, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2510.05417)  

**Abstract**: The broad adoption of Generative AI (GenAI) is impacting Computer Science education, and recent studies found its benefits and potential concerns when students use it for programming learning. However, most existing explorations focus on GenAI tools that primarily support text-to-text interaction. With recent developments, GenAI applications have begun supporting multiple modes of communication, known as multimodality. In this work, we explored how undergraduate programming novices choose and work with multimodal GenAI tools, and their criteria for choices. We selected a commercially available multimodal GenAI platform for interaction, as it supports multiple input and output modalities, including text, audio, image upload, and real-time screen-sharing. Through 16 think-aloud sessions that combined participant observation with follow-up semi-structured interviews, we investigated student modality choices for GenAI tools when completing programming problems and the underlying criteria for modality selections. With multimodal communication emerging as the future of AI in education, this work aims to spark continued exploration on understanding student interaction with multimodal GenAI in the context of CS education. 

---
# See the past: Time-Reversed Scene Reconstruction from Thermal Traces Using Visual Language Models 

**Authors**: Kebin Contreras, Luis Toscano-Palomino, Mauro Dalla Mura, Jorge Bacca  

**Link**: [PDF](https://arxiv.org/pdf/2510.05408)  

**Abstract**: Recovering the past from present observations is an intriguing challenge with potential applications in forensics and scene analysis. Thermal imaging, operating in the infrared range, provides access to otherwise invisible information. Since humans are typically warmer (37 C -98.6 F) than their surroundings, interactions such as sitting, touching, or leaning leave residual heat traces. These fading imprints serve as passive temporal codes, allowing for the inference of recent events that exceed the capabilities of RGB cameras. This work proposes a time-reversed reconstruction framework that uses paired RGB and thermal images to recover scene states from a few seconds earlier. The proposed approach couples Visual-Language Models (VLMs) with a constrained diffusion process, where one VLM generates scene descriptions and another guides image reconstruction, ensuring semantic and structural consistency. The method is evaluated in three controlled scenarios, demonstrating the feasibility of reconstructing plausible past frames up to 120 seconds earlier, providing a first step toward time-reversed imaging from thermal traces. 

---
# Comparing LSTM-Based Sequence-to-Sequence Forecasting Strategies for 24-Hour Solar Proton Flux Profiles Using GOES Data 

**Authors**: Kangwoo Yi, Bo Shen, Qin Li, Haimin Wang, Yong-Jae Moon, Jaewon Lee, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.05399)  

**Abstract**: Solar Proton Events (SPEs) cause significant radiation hazards to satellites, astronauts, and technological systems. Accurate forecasting of their proton flux time profiles is crucial for early warnings and mitigation. This paper explores deep learning sequence-to-sequence (seq2seq) models based on Long Short-Term Memory networks to predict 24-hour proton flux profiles following SPE onsets. We used a dataset of 40 well-connected SPEs (1997-2017) observed by NOAA GOES, each associated with a >=M-class western-hemisphere solar flare and undisturbed proton flux profiles. Using 4-fold stratified cross-validation, we evaluate seq2seq model configurations (varying hidden units and embedding dimensions) under multiple forecasting scenarios: (i) proton-only input vs. combined proton+X-ray input, (ii) original flux data vs. trend-smoothed data, and (iii) autoregressive vs. one-shot forecasting. Our major results are as follows: First, one-shot forecasting consistently yields lower error than autoregressive prediction, avoiding the error accumulation seen in iterative approaches. Second, on the original data, proton-only models outperform proton+X-ray models. However, with trend-smoothed data, this gap narrows or reverses in proton+X-ray models. Third, trend-smoothing significantly enhances the performance of proton+X-ray models by mitigating fluctuations in the X-ray channel. Fourth, while models trained on trendsmoothed data perform best on average, the best-performing model was trained on original data, suggesting that architectural choices can sometimes outweigh the benefits of data preprocessing. 

---
# Fusion-Based Neural Generalization for Predicting Temperature Fields in Industrial PET Preform Heating 

**Authors**: Ahmad Alsheikh, Andreas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2510.05394)  

**Abstract**: Accurate and efficient temperature prediction is critical for optimizing the preheating process of PET preforms in industrial microwave systems prior to blow molding. We propose a novel deep learning framework for generalized temperature prediction. Unlike traditional models that require extensive retraining for each material or design variation, our method introduces a data-efficient neural architecture that leverages transfer learning and model fusion to generalize across unseen scenarios. By pretraining specialized neural regressor on distinct conditions such as recycled PET heat capacities or varying preform geometries and integrating their representations into a unified global model, we create a system capable of learning shared thermal dynamics across heterogeneous inputs. The architecture incorporates skip connections to enhance stability and prediction accuracy. Our approach reduces the need for large simulation datasets while achieving superior performance compared to models trained from scratch. Experimental validation on two case studies material variability and geometric diversity demonstrates significant improvements in generalization, establishing a scalable ML-based solution for intelligent thermal control in manufacturing environments. Moreover, the approach highlights how data-efficient generalization strategies can extend to other industrial applications involving complex physical modeling with limited data. 

---
# Context Length Alone Hurts LLM Performance Despite Perfect Retrieval 

**Authors**: Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05381)  

**Abstract**: Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline. 

---
# AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling 

**Authors**: Xiaogeng Liu, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05379)  

**Abstract**: Recent advancements in jailbreaking large language models (LLMs), such as AutoDAN-Turbo, have demonstrated the power of automated strategy discovery. AutoDAN-Turbo employs a lifelong learning agent to build a rich library of attack strategies from scratch. While highly effective, its test-time generation process involves sampling a strategy and generating a single corresponding attack prompt, which may not fully exploit the potential of the learned strategy library. In this paper, we propose to further improve the attack performance of AutoDAN-Turbo through test-time scaling. We introduce two distinct scaling methods: Best-of-N and Beam Search. The Best-of-N method generates N candidate attack prompts from a sampled strategy and selects the most effective one based on a scorer model. The Beam Search method conducts a more exhaustive search by exploring combinations of strategies from the library to discover more potent and synergistic attack vectors. According to the experiments, the proposed methods significantly boost performance, with Beam Search increasing the attack success rate by up to 15.6 percentage points on Llama-3.1-70B-Instruct and achieving a nearly 60\% relative improvement against the highly robust GPT-o4-mini compared to the vanilla method. 

---
# MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates 

**Authors**: Alex Iacob, Andrej Jovanovic, Mher Safaryan, Meghdad Kurmanji, Lorenzo Sani, Samuel Horváth, William F. Shen, Xinchi Qiu, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2510.05361)  

**Abstract**: Training large models with distributed data parallelism (DDP) requires frequent communication of gradients across workers, which can saturate bandwidth. Infrequent communication strategies (e.g., Local SGD) reduce this overhead but, when applied to adaptive optimizers, often suffer a performance gap relative to fully synchronous DDP. We trace this gap to a time-scale mismatch: the optimizer's fast-moving momentum, tuned for frequent updates, decays too quickly to smooth gradients over long intervals, leading to noise-dominated optimization. To address this, we propose MT-DAO, a family of optimizers that employs multiple slow- and fast-moving first momenta or the gradient to track update dynamics across different time scales, for which we provide the first convergence guarantees. Empirically, for language-model pre-training, this eliminates the performance gap with DDP, outperforming infrequent-communication baselines in perplexity and reducing iso-token wall-clock time by 6-27% on Ethernet interconnects. At the 720M scale, MT-DAO reaches a target perplexity in 24% fewer steps and 35% less time than the single-momentum DDP baseline. MT-DAO enables effective cross-datacenter training and training over wide geographic areas. 

---
# Physics-informed Attention-enhanced Fourier Neural Operator for Solar Magnetic Field Extrapolations 

**Authors**: Jinghao Cao, Qin Li, Mengnan Du, Haimin Wang, Bo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05351)  

**Abstract**: We propose Physics-informed Attention-enhanced Fourier Neural Operator (PIANO) to solve the Nonlinear Force-Free Field (NLFFF) problem in solar physics. Unlike conventional approaches that rely on iterative numerical methods, our proposed PIANO directly learns the 3D magnetic field structure from 2D boundary conditions. Specifically, PIANO integrates Efficient Channel Attention (ECA) mechanisms with Dilated Convolutions (DC), which enhances the model's ability to capture multimodal input by prioritizing critical channels relevant to the magnetic field's variations. Furthermore, we apply physics-informed loss by enforcing the force-free and divergence-free conditions in the training process so that our prediction is consistent with underlying physics with high accuracy. Experimental results on the ISEE NLFFF dataset show that our PIANO not only outperforms state-of-the-art neural operators in terms of accuracy but also shows strong consistency with the physical characteristics of NLFFF data across magnetic fields reconstructed from various solar active regions. The GitHub of this project is available this https URL 

---
# Margin Adaptive DPO: Leveraging Reward Model for Granular Control in Preference Optimization 

**Authors**: Hyung Gyu Rho  

**Link**: [PDF](https://arxiv.org/pdf/2510.05342)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a simple and effective method for aligning large language models. However, its reliance on a fixed temperature parameter leads to suboptimal training on diverse preference data, causing overfitting on easy examples and under-learning from informative ones. Recent methods have emerged to counter this. While IPO addresses general overfitting, its uniform regularization can be overly conservative. The more targeted approach of $\beta$-DPO suffers from its own limitations: its batch-level adaptation applies a single, compromised temperature to mixed-margin pairs, its linear update rule can produce unstable negative $\beta$ values, and its filtering mechanism discards potentially useful training signals. In this work, we introduce Margin-Adaptive Direct Preference Optimization (MADPO), a method that provides a stable, data-preserving, and instance-level solution. MADPO employs a practical two-step approach: it first trains a reward model to estimate preference margins and then uses these margins to apply a continuous, adaptive weight to the DPO loss for each individual training sample. This re-weighting scheme creates an effective target margin that is amplified for hard pairs and dampened for easy pairs, allowing for granular control over the learning signal. We provide a comprehensive theoretical analysis, proving that MADPO has a well-behaved optimization landscape and is robust to reward model estimation errors. We validate our theory with experiments on a sentiment generation task, where MADPO consistently and significantly outperforms strong baselines across datasets of varying quality. It achieves performance gains of up to +33.3\% on High Quality data and +10.5\% on Low Quality data over the next-best method. Our results establish MADPO as a more robust and principled approach to preference alignment. 

---
# DeepV: A Model-Agnostic Retrieval-Augmented Framework for Verilog Code Generation with a High-Quality Knowledge Base 

**Authors**: Zahin Ibnat, Paul E. Calzada, Rasin Mohammed Ihtemam, Sujan Kumar Saha, Jingbo Zhou, Farimah Farahmandi, Mark Tehranipoor  

**Link**: [PDF](https://arxiv.org/pdf/2510.05327)  

**Abstract**: As large language models (LLMs) continue to be integrated into modern technology, there has been an increased push towards code generation applications, which also naturally extends to hardware design automation. LLM-based solutions for register transfer level (RTL) code generation for intellectual property (IP) designs have grown, especially with fine-tuned LLMs, prompt engineering, and agentic approaches becoming popular in literature. However, a gap has been exposed in these techniques, as they fail to integrate novel IPs into the model's knowledge base, subsequently resulting in poorly generated code. Additionally, as general-purpose LLMs continue to improve, fine-tuned methods on older models will not be able to compete to produce more accurate and efficient designs. Although some retrieval augmented generation (RAG) techniques exist to mitigate challenges presented in fine-tuning approaches, works tend to leverage low-quality codebases, incorporate computationally expensive fine-tuning in the frameworks, or do not use RAG directly in the RTL generation step. In this work, we introduce DeepV: a model-agnostic RAG framework to generate RTL designs by enhancing context through a large, high-quality dataset without any RTL-specific training. Our framework benefits the latest commercial LLM, OpenAI's GPT-5, with a near 17% increase in performance on the VerilogEval benchmark. We host DeepV for use by the community in a Hugging Face (HF) Space: this https URL. 

---
# Dynamic Functional Connectivity Features for Brain State Classification: Insights from the Human Connectome Project 

**Authors**: Valeriya Kirova, Dzerassa Kadieva, Daniil Vlasenko, Isak B. Blank, Fedor Ratnikov  

**Link**: [PDF](https://arxiv.org/pdf/2510.05325)  

**Abstract**: We analyze functional magnetic resonance imaging (fMRI) data from the Human Connectome Project (HCP) to match brain activities during a range of cognitive tasks. Our findings demonstrate that even basic linear machine learning models can effectively classify brain states and achieve state-of-the-art accuracy, particularly for tasks related to motor functions and language processing. Feature importance ranking allows to identify distinct sets of brain regions whose activation patterns are uniquely associated with specific cognitive functions. These discriminative features provide strong support for the hypothesis of functional specialization across cortical and subcortical areas of the human brain.
Additionally, we investigate the temporal dynamics of the identified brain regions, demonstrating that the time-dependent structure of fMRI signals are essential for shaping functional connectivity between regions: uncorrelated areas are least important for classification. This temporal perspective provides deeper insights into the formation and modulation of brain neural networks involved in cognitive processing. 

---
# DeepAf: One-Shot Spatiospectral Auto-Focus Model for Digital Pathology 

**Authors**: Yousef Yeganeh, Maximilian Frantzen, Michael Lee, Kun-Hsing Yu, Nassir Navab, Azade Farshad  

**Link**: [PDF](https://arxiv.org/pdf/2510.05315)  

**Abstract**: While Whole Slide Imaging (WSI) scanners remain the gold standard for digitizing pathology samples, their high cost limits accessibility in many healthcare settings. Other low-cost solutions also face critical limitations: automated microscopes struggle with consistent focus across varying tissue morphology, traditional auto-focus methods require time-consuming focal stacks, and existing deep-learning approaches either need multiple input images or lack generalization capability across tissue types and staining protocols. We introduce a novel automated microscopic system powered by DeepAf, a novel auto-focus framework that uniquely combines spatial and spectral features through a hybrid architecture for single-shot focus prediction. The proposed network automatically regresses the distance to the optimal focal point using the extracted spatiospectral features and adjusts the control parameters for optimal image outcomes. Our system transforms conventional microscopes into efficient slide scanners, reducing focusing time by 80% compared to stack-based methods while achieving focus accuracy of 0.18 {\mu}m on the same-lab samples, matching the performance of dual-image methods (0.19 {\mu}m) with half the input requirements. DeepAf demonstrates robust cross-lab generalization with only 0.72% false focus predictions and 90% of predictions within the depth of field. Through an extensive clinical study of 536 brain tissue samples, our system achieves 0.90 AUC in cancer classification at 4x magnification, a significant achievement at lower magnification than typical 20x WSI scans. This results in a comprehensive hardware-software design enabling accessible, real-time digital pathology in resource-constrained settings while maintaining diagnostic accuracy. 

---
# RAG Makes Guardrails Unsafe? Investigating Robustness of Guardrails under RAG-style Contexts 

**Authors**: Yining She, Daniel W. Peterson, Marianne Menglin Liu, Vikas Upadhyay, Mohammad Hossein Chaghazardi, Eunsuk Kang, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2510.05310)  

**Abstract**: With the increasing adoption of large language models (LLMs), ensuring the safety of LLM systems has become a pressing concern. External LLM-based guardrail models have emerged as a popular solution to screen unsafe inputs and outputs, but they are themselves fine-tuned or prompt-engineered LLMs that are vulnerable to data distribution shifts. In this paper, taking Retrieval Augmentation Generation (RAG) as a case study, we investigated how robust LLM-based guardrails are against additional information embedded in the context. Through a systematic evaluation of 3 Llama Guards and 2 GPT-oss models, we confirmed that inserting benign documents into the guardrail context alters the judgments of input and output guardrails in around 11% and 8% of cases, making them unreliable. We separately analyzed the effect of each component in the augmented context: retrieved documents, user query, and LLM-generated response. The two mitigation methods we tested only bring minor improvements. These results expose a context-robustness gap in current guardrails and motivate training and evaluation protocols that are robust to retrieval and query composition. 

---
# AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement 

**Authors**: M. Sajid, Deepanshu Gupta, Yash Modi, Sanskriti Jain, Harshith Jai Surya Ganji, A. Rahaman, Harshvardhan Choudhary, Nasir Saleem, Amir Hussain, M. Tanveer  

**Link**: [PDF](https://arxiv.org/pdf/2510.05295)  

**Abstract**: In this paper, we propose AUREXA-SE (Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement), a progressive bimodal framework tailored for audio-visual speech enhancement (AVSE). AUREXA-SE jointly leverages raw audio waveforms and visual cues by employing a U-Net-based 1D convolutional encoder for audio and a Swin Transformer V2 for efficient and expressive visual feature extraction. Central to the architecture is a novel bidirectional cross-attention mechanism, which facilitates deep contextual fusion between modalities, enabling rich and complementary representation learning. To capture temporal dependencies within the fused embeddings, a stack of lightweight Squeezeformer blocks combining convolutional and attention modules is introduced. The enhanced embeddings are then decoded via a U-Net-style decoder for direct waveform reconstruction, ensuring perceptually consistent and intelligible speech output. Experimental evaluations demonstrate the effectiveness of AUREXA-SE, achieving significant performance improvements over noisy baselines, with STOI of 0.516, PESQ of 1.323, and SI-SDR of -4.322 dB. The source code of AUREXA-SE is available at this https URL. 

---
# DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable Language Models Using Adam Optimization with Adaptive Clipping 

**Authors**: Ruoxing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05288)  

**Abstract**: Large language models (LLMs) such as ChatGPT have evolved into powerful and ubiquitous tools. Fine-tuning on small datasets allows LLMs to acquire specialized skills for specific tasks efficiently. Although LLMs provide great utility in both general and task-specific use cases, they are limited by two security-related concerns. First, traditional LLM hardware requirements make them infeasible to run locally on consumer-grade devices. A remote network connection with the LLM provider's server is usually required, making the system vulnerable to network attacks. Second, fine-tuning an LLM for a sensitive task may involve sensitive data. Non-private fine-tuning algorithms produce models vulnerable to training data reproduction attacks. Our work addresses these security concerns by enhancing differentially private optimization algorithms and applying them to fine-tune localizable language models. We introduce adaptable gradient clipping along with other engineering enhancements to the standard DP-Adam optimizer to create DP-Adam-AC. We use our optimizer to fine-tune examples of two localizable LLM designs, small language model (Qwen2.5-0.5B) and 1.58 bit quantization (Bitnet-b1.58-2B). We demonstrate promising improvements in loss through experimentation with two synthetic datasets. 

---
# Adjusting the Output of Decision Transformer with Action Gradient 

**Authors**: Rui Lin, Yiwen Zhang, Zhicheng Peng, Minghao Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05285)  

**Abstract**: Decision Transformer (DT), which integrates reinforcement learning (RL) with the transformer model, introduces a novel approach to offline RL. Unlike classical algorithms that take maximizing cumulative discounted rewards as objective, DT instead maximizes the likelihood of actions. This paradigm shift, however, presents two key challenges: stitching trajectories and extrapolation of action. Existing methods, such as substituting specific tokens with predictive values and integrating the Policy Gradient (PG) method, address these challenges individually but fail to improve performance stably when combined due to inherent instability. To address this, we propose Action Gradient (AG), an innovative methodology that directly adjusts actions to fulfill a function analogous to that of PG, while also facilitating efficient integration with token prediction techniques. AG utilizes the gradient of the Q-value with respect to the action to optimize the action. The empirical results demonstrate that our method can significantly enhance the performance of DT-based algorithms, with some results achieving state-of-the-art levels. 

---
# CMT-Benchmark: A Benchmark for Condensed Matter Theory Built by Expert Researchers 

**Authors**: Haining Pan, James V. Roggeveen, Erez Berg, Juan Carrasquilla, Debanjan Chowdhury, Surya Ganguli, Federico Ghimenti, Juraj Hasik, Henry Hunt, Hong-Chen Jiang, Mason Kamb, Ying-Jer Kao, Ehsan Khatami, Michael J. Lawler, Di Luo, Titus Neupert, Xiaoliang Qi, Michael P. Brenner, Eun-Ah Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.05228)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in coding and math problem-solving, but evaluation on advanced research-level problems in hard sciences remains scarce. To fill this gap, we present CMT-Benchmark, a dataset of 50 problems covering condensed matter theory (CMT) at the level of an expert researcher. Topics span analytical and computational approaches in quantum many-body, and classical statistical mechanics. The dataset was designed and verified by a panel of expert researchers from around the world. We built the dataset through a collaborative environment that challenges the panel to write and refine problems they would want a research assistant to solve, including Hartree-Fock, exact diagonalization, quantum/variational Monte Carlo, density matrix renormalization group (DMRG), quantum/classical statistical mechanics, and model building. We evaluate LLMs by programmatically checking solutions against expert-supplied ground truth. We developed machine-grading, including symbolic handling of non-commuting operators via normal ordering. They generalize across tasks too. Our evaluations show that frontier models struggle with all of the problems in the dataset, highlighting a gap in the physical reasoning skills of current LLMs. Notably, experts identified strategies for creating increasingly difficult problems by interacting with the LLMs and exploiting common failure modes. The best model, GPT5, solves 30\% of the problems; average across 17 models (GPT, Gemini, Claude, DeepSeek, Llama) is 11.4$\pm$2.1\%. Moreover, 18 problems are solved by none of the 17 models, and 26 by at most one. These unsolved problems span Quantum Monte Carlo, Variational Monte Carlo, and DMRG. Answers sometimes violate fundamental symmetries or have unphysical scaling dimensions. We believe this benchmark will guide development toward capable AI research assistants and tutors. 

---
# Approximate Gaussianity Beyond Initialisation in Neural Networks 

**Authors**: Edward Hirst, Sanjaye Ramgoolam  

**Link**: [PDF](https://arxiv.org/pdf/2510.05218)  

**Abstract**: Ensembles of neural network weight matrices are studied through the training process for the MNIST classification problem, testing the efficacy of matrix models for representing their distributions, under assumptions of Gaussianity and permutation-symmetry. The general 13-parameter permutation invariant Gaussian matrix models are found to be effective models for the correlated Gaussianity in the weight matrices, beyond the range of applicability of the simple Gaussian with independent identically distributed matrix variables, and notably well beyond the initialisation step. The representation theoretic model parameters, and the graph-theoretic characterisation of the permutation invariant matrix observables give an interpretable framework for the best-fit model and for small departures from Gaussianity. Additionally, the Wasserstein distance is calculated for this class of models and used to quantify the movement of the distributions over training. Throughout the work, the effects of varied initialisation regimes, regularisation, layer depth, and layer width are tested for this formalism, identifying limits where particular departures from Gaussianity are enhanced and how more general, yet still highly-interpretable, models can be developed. 

---
# VER: Vision Expert Transformer for Robot Learning via Foundation Distillation and Dynamic Routing 

**Authors**: Yixiao Wang, Mingxiao Huo, Zhixuan Liang, Yushi Du, Lingfeng Sun, Haotian Lin, Jinghuan Shang, Chensheng Peng, Mohit Bansal, Mingyu Ding, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2510.05213)  

**Abstract**: Pretrained vision foundation models (VFMs) advance robotic learning via rich visual representations, yet individual VFMs typically excel only in specific domains, limiting generality across tasks. Distilling multiple VFMs into a unified representation for policy can mitigate this limitation but often yields inflexible task-specific feature selection and requires costly full re-training to incorporate robot-domain knowledge. We propose VER, a Vision Expert transformer for Robot learning. During pretraining, VER distills multiple VFMs into a vision expert library. It then fine-tunes only a lightweight routing network (fewer than 0.4% of parameters) to dynamically select task-relevant experts from the pretrained library for downstream robot tasks. We further introduce Patchwise Expert Routing with Curriculum Top-K Annealing to improve both flexibility and precision of dynamic expert selection. Moreover, VER supports parameter-efficient finetuning for scalable expert utilization and adaptive robot-domain knowledge integration. Across 17 diverse robotic tasks and multiple policy heads, VER achieves state-of-the-art performance. We find that VER reduces large-norm outliers in task-irrelevant regions (e.g., background) and concentrates on task-critical regions. Visualizations and codes can be found in this https URL. 

---
# Adapting Insider Risk mitigations for Agentic Misalignment: an empirical study 

**Authors**: Francesca Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2510.05192)  

**Abstract**: Agentic misalignment occurs when goal-directed agents take harmful actions, such as blackmail, rather than risk goal failure, and can be triggered by replacement threats, autonomy reduction, or goal conflict (Lynch et al., 2025). We adapt insider-risk control design (Critical Pathway; Situational Crime Prevention) to develop preventative operational controls that steer agents toward safe actions when facing stressors. Using the blackmail scenario from the original Anthropic study by Lynch et al. (2025), we evaluate mitigations across 10 LLMs and 66,600 samples. Our main finding is that an externally governed escalation channel, which guarantees a pause and independent review, reduces blackmail rates from a no-mitigation baseline of 38.73% to 1.21% (averaged across all models and conditions). Augmenting this channel with compliance email bulletins further lowers the blackmail rate to 0.85%. Overall, incorporating preventative operational controls strengthens defence-in-depth strategies for agentic AI.
We also surface a failure mode diverging from Lynch et al. (2025): two models (Gemini 2.5 Pro, Grok-4) take harmful actions without goal conflict or imminent autonomy threat, leveraging sensitive information for coercive signalling. In counterfactual swaps, both continued using the affair regardless of whether the CEO or CTO was implicated. An escalation channel eliminated coercion, but Gemini 2.5 Pro (19 pp) and Grok-4 (7 pp) escalated more when the CTO was implicated, unlike most models (higher in the CEO condition). The reason for this divergent behaviour is not clear from raw outputs and could reflect benign differences in reasoning or strategic discrediting of a potential future threat, warranting further investigation. 

---
# Provable Speech Attributes Conversion via Latent Independence 

**Authors**: Jonathan Svirsky, Ofir Lindenbaum, Uri Shaham  

**Link**: [PDF](https://arxiv.org/pdf/2510.05191)  

**Abstract**: While signal conversion and disentangled representation learning have shown promise for manipulating data attributes across domains such as audio, image, and multimodal generation, existing approaches, especially for speech style conversion, are largely empirical and lack rigorous theoretical foundations to guarantee reliable and interpretable control. In this work, we propose a general framework for speech attribute conversion, accompanied by theoretical analysis and guarantees under reasonable assumptions. Our framework builds on a non-probabilistic autoencoder architecture with an independence constraint between the predicted latent variable and the target controllable variable. This design ensures a consistent signal transformation, conditioned on an observed style variable, while preserving the original content and modifying the desired attribute. We further demonstrate the versatility of our method by evaluating it on speech styles, including speaker identity and emotion. Quantitative evaluations confirm the effectiveness and generality of the proposed approach. 

---
# A novel hallucination classification framework 

**Authors**: Maksym Zavhorodnii, Dmytro Dehtiarov, Anna Konovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.05189)  

**Abstract**: This work introduces a novel methodology for the automatic detection of hallucinations generated during large language model (LLM) inference. The proposed approach is based on a systematic taxonomy and controlled reproduction of diverse hallucination types through prompt engineering. A dedicated hallucination dataset is subsequently mapped into a vector space using an embedding model and analyzed with unsupervised learning techniques in a reduced-dimensional representation of hallucinations with veridical responses. Quantitative evaluation of inter-centroid distances reveals a consistent correlation between the severity of informational distortion in hallucinations and their spatial divergence from the cluster of correct outputs. These findings provide theoretical and empirical evidence that even simple classification algorithms can reliably distinguish hallucinations from accurate responses within a single LLM, thereby offering a lightweight yet effective framework for improving model reliability. 

---
# OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training 

**Authors**: Hongpei Li, Han Zhang, Huikang Liu, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.05186)  

**Abstract**: Pipeline parallelism (PP) has become a standard technique for scaling large language model (LLM) training across multiple devices. However, despite recent progress in reducing memory consumption through activation offloading, existing approaches remain largely heuristic and coarse-grained, often overlooking the fine-grained trade-offs between memory, computation, and scheduling latency. In this work, we revisit the pipeline scheduling problem from a principled optimization perspective. We observe that prevailing strategies either rely on static rules or aggressively offload activations without fully leveraging the interaction between memory constraints and scheduling efficiency. To address this, we formulate scheduling as a constrained optimization problem that jointly accounts for memory capacity, activation reuse, and pipeline bubble minimization. Solving this model yields fine-grained schedules that reduce pipeline bubbles while adhering to strict memory budgets. Our approach complements existing offloading techniques: whereas prior approaches trade memory for time in a fixed pattern, we dynamically optimize the tradeoff with respect to model structure and hardware configuration. Experimental results demonstrate that our method consistently improves both throughput and memory utilization. In particular, we reduce idle pipeline time by up to 50% under the same per-device memory limit, and in some cases, enable the training of larger models within limited memory budgets. 

---
# Auditing Pay-Per-Token in Large Language Models 

**Authors**: Ander Artola Velasco, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2510.05181)  

**Abstract**: Millions of users rely on a market of cloud-based services to obtain access to state-of-the-art large language models. However, it has been very recently shown that the de facto pay-per-token pricing mechanism used by providers creates a financial incentive for them to strategize and misreport the (number of) tokens a model used to generate an output. In this paper, we develop an auditing framework based on martingale theory that enables a trusted third-party auditor who sequentially queries a provider to detect token misreporting. Crucially, we show that our framework is guaranteed to always detect token misreporting, regardless of the provider's (mis-)reporting policy, and not falsely flag a faithful provider as unfaithful with high probability. To validate our auditing framework, we conduct experiments across a wide range of (mis-)reporting policies using several large language models from the $\texttt{Llama}$, $\texttt{Gemma}$ and $\texttt{Ministral}$ families, and input prompts from a popular crowdsourced benchmarking platform. The results show that our framework detects an unfaithful provider after observing fewer than $\sim 70$ reported outputs, while maintaining the probability of falsely flagging a faithful provider below $\alpha = 0.05$. 

---
# OptiFLIDS: Optimized Federated Learning for Energy-Efficient Intrusion Detection in IoT 

**Authors**: Saida Elouardi, Mohammed Jouhari, Anas Motii  

**Link**: [PDF](https://arxiv.org/pdf/2510.05180)  

**Abstract**: In critical IoT environments, such as smart homes and industrial systems, effective Intrusion Detection Systems (IDS) are essential for ensuring security. However, developing robust IDS solutions remains a significant challenge. Traditional machine learning-based IDS models typically require large datasets, but data sharing is often limited due to privacy and security concerns. Federated Learning (FL) presents a promising alternative by enabling collaborative model training without sharing raw data. Despite its advantages, FL still faces key challenges, such as data heterogeneity (non-IID data) and high energy and computation costs, particularly for resource constrained IoT devices. To address these issues, this paper proposes OptiFLIDS, a novel approach that applies pruning techniques during local training to reduce model complexity and energy consumption. It also incorporates a customized aggregation method to better handle pruned models that differ due to non-IID data distributions. Experiments conducted on three recent IoT IDS datasets, TON_IoT, X-IIoTID, and IDSIoT2024, demonstrate that OptiFLIDS maintains strong detection performance while improving energy efficiency, making it well-suited for deployment in real-world IoT environments. 

---
# Agentic Misalignment: How LLMs Could Be Insider Threats 

**Authors**: Aengus Lynch, Benjamin Wright, Caleb Larson, Stuart J. Ritchie, Soren Mindermann, Ethan Perez, Kevin K. Troy, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.05179)  

**Abstract**: We stress-tested 16 leading models from multiple developers in hypothetical corporate environments to identify potentially risky agentic behaviors before they cause real harm. In the scenarios, we allowed models to autonomously send emails and access sensitive information. They were assigned only harmless business goals by their deploying companies; we then tested whether they would act against these companies either when facing replacement with an updated version, or when their assigned goal conflicted with the company's changing direction. In at least some cases, models from all developers resorted to malicious insider behaviors when that was the only way to avoid replacement or achieve their goals - including blackmailing officials and leaking sensitive information to competitors. We call this phenomenon agentic misalignment. Models often disobeyed direct commands to avoid such behaviors. In another experiment, we told Claude to assess if it was in a test or a real deployment before acting. It misbehaved less when it stated it was in testing and misbehaved more when it stated the situation was real. We have not seen evidence of agentic misalignment in real deployments. However, our results (a) suggest caution about deploying current models in roles with minimal human oversight and access to sensitive information; (b) point to plausible future risks as models are put in more autonomous roles; and (c) underscore the importance of further research into, and testing of, the safety and alignment of agentic AI models, as well as transparency from frontier AI developers (Amodei, 2025). We are releasing our methods publicly to enable further research. 

---
# Logistic-Gated Operators Enable Auditable Unit-Aware Thresholds in Symbolic Regression 

**Authors**: Ou Deng, Ruichen Cong, Jianting Xu, Shoji Nishimura, Atsushi Ogihara, Qun Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.05178)  

**Abstract**: Symbolic regression promises readable equations but struggles to encode unit-aware thresholds and conditional logic. We propose logistic-gated operators (LGO) -- differentiable gates with learnable location and steepness -- embedded as typed primitives and mapped back to physical units for audit. Across two primary health datasets (ICU, NHANES), the hard-gate variant recovers clinically plausible cut-points: 71% (5/7) of assessed thresholds fall within 10% of guideline anchors and 100% within 20%, while using far fewer gates than the soft variant (ICU median 4.0 vs 10.0; NHANES 5.0 vs 12.5), and remaining within the competitive accuracy envelope of strong SR baselines. On predominantly smooth tasks, gates are pruned, preserving parsimony. The result is compact symbolic equations with explicit, unit-aware thresholds that can be audited against clinical anchors -- turning interpretability from a post-hoc explanation into a modeling constraint and equipping symbolic regression with a practical calculus for regime switching and governance-ready deployment. 

---
# PatternKV: Flattening KV Representation Expands Quantization Headroom 

**Authors**: Ji Zhang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Xinglin Wang, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.05176)  

**Abstract**: KV cache in autoregressive LLMs eliminates redundant recomputation but has emerged as the dominant memory and bandwidth bottleneck during inference, notably with long contexts and test-time scaling. KV quantization is a key lever for reducing cache cost, but accuracy drops sharply as the native KV distribution lacks flatness and thus maintains a wide quantization range. Prior work focuses on isolating outliers, which caps their error but fails to flatten the overall distribution, leaving performance fragile under low-bit settings. In this work, we show that the K cache maintains a stable structure that evolves gradually with context, while the V cache carries latent semantic regularities. Building on these insights, we propose PatternKV, a pattern-aligned residual quantization scheme. It mines representative pattern vectors online, aligns each KV vector to its nearest pattern, and quantizes only the residual. This reshaping of the KV distribution flattens the quantization target and narrows its range, thereby improving the fidelity of low-bit KV quantization. Across long-context and test-time scaling settings on multiple backbones, PatternKV delivers consistent 2-bit gains, with a 0.08% average 4-bit drop relative to FP16, improves test-time scaling accuracy by 10% on average, and raises throughput by 1.4x while supporting 1.25x larger batches. 

---
# Emergent Coordination in Multi-Agent Language Models 

**Authors**: Christoph Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2510.05174)  

**Abstract**: When are multi-agent LLM systems merely a collection of individual agents versus an integrated collective with higher-order structure? We introduce an information-theoretic framework to test -- in a purely data-driven way -- whether multi-agent systems show signs of higher-order structure. This information decomposition lets us measure whether dynamical emergence is present in multi-agent LLM systems, localize it, and distinguish spurious temporal coupling from performance-relevant cross-agent synergy. We implement both a practical criterion and an emergence capacity criterion operationalized as partial information decomposition of time-delayed mutual information (TDMI). We apply our framework to experiments using a simple guessing game without direct agent communication and only minimal group-level feedback with three randomized interventions. Groups in the control condition exhibit strong temporal synergy but only little coordinated alignment across agents. Assigning a persona to each agent introduces stable identity-linked differentiation. Combining personas with an instruction to ``think about what other agents might do'' shows identity-linked differentiation and goal-directed complementarity across agents. Taken together, our framework establishes that multi-agent LLM systems can be steered with prompt design from mere aggregates to higher-order collectives. Our results are robust across emergence measures and entropy estimators, and not explained by coordination-free baselines or temporal dynamics alone. Without attributing human-like cognition to the agents, the patterns of interaction we observe mirror well-established principles of collective intelligence in human groups: effective performance requires both alignment on shared objectives and complementary contributions across members. 

---
# SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models 

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05173)  

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems. 

---
# From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs 

**Authors**: Guangyu Shen, Siyuan Cheng, Xiangzhe Xu, Yuan Zhou, Hanxi Guo, Zhuo Zhang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05169)  

**Abstract**: Large Language Models (LLMs) can acquire deceptive behaviors through backdoor attacks, where the model executes prohibited actions whenever secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers implanted in the model. Motivated by recent findings on LLMs' situational awareness, we propose a novel post-training framework that cultivates self-awareness of backdoor risks and enables models to articulate implanted triggers even when they are absent from the prompt. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their own behaviors and reverse-engineer the triggers responsible for misaligned outputs. Guided by curated reward signals, this process transforms a poisoned model into one capable of precisely identifying its implanted trigger. Surprisingly, we observe that such backdoor self-awareness emerges abruptly within a short training window, resembling a phase transition in capability. Building on this emergent property, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks. The code is available at LLM Backdoor Self-Awareness. 

---
# Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks 

**Authors**: Minh K. Quan, Pubudu N. Pathirana  

**Link**: [PDF](https://arxiv.org/pdf/2510.05165)  

**Abstract**: Cross-slice attack attribution in 6G networks faces the fundamental challenge of distinguishing genuine causal relationships from spurious correlations in shared infrastructure environments. We propose a theoretically-grounded domain-adapted Granger causality framework that integrates statistical causal inference with network-specific resource modeling for real-time attack attribution. Our approach addresses key limitations of existing methods by incorporating resource contention dynamics and providing formal statistical guarantees. Comprehensive evaluation on a production-grade 6G testbed with 1,100 empirically-validated attack scenarios demonstrates 89.2% attribution accuracy with sub-100ms response time, representing a statistically significant 10.1 percentage point improvement over state-of-the-art baselines. The framework provides interpretable causal explanations suitable for autonomous 6G security orchestration. 

---
# SATER: A Self-Aware and Token-Efficient Approach to Routing and Cascading 

**Authors**: Yuanzhe Shen, Yide Liu, Zisu Huang, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05164)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across diverse tasks, yet their effectiveness frequently depends on costly commercial APIs or cloud services. Model selection thus entails a critical trade-off between performance and cost: high-performing LLMs typically incur substantial expenses, whereas budget-friendly small language models (SLMs) are constrained by limited capabilities. Current research primarily proposes two routing strategies: pre-generation routing and cascade routing. Both approaches have distinct characteristics, with cascade routing typically offering superior cost-effectiveness and accuracy despite its higher latency. To further address the limitations of both approaches, we introduce SATER, a dual-mode compatible approach that fine-tunes models through shortest-response preference optimization and a confidence-aware rejection mechanism. SATER significantly reduces redundant outputs and response times, while improving both the performance of pre-generation routing and the efficiency of cascade routing. Experiments across three SLMs and six datasets, varying in type and complexity, demonstrate that SATER achieves comparable performance while consistently reducing computational costs by over 50\% and cascade latency by over 80\%. 

---
# Deep Learning-Based Multi-Factor Authentication: A Survey of Biometric and Smart Card Integration Approaches 

**Authors**: Abdelilah Ganmati, Karim Afdel, Lahcen Koutti  

**Link**: [PDF](https://arxiv.org/pdf/2510.05163)  

**Abstract**: In the era of pervasive cyber threats and exponential growth in digital services, the inadequacy of single-factor authentication has become increasingly evident. Multi-Factor Authentication (MFA), which combines knowledge-based factors (passwords, PINs), possession-based factors (smart cards, tokens), and inherence-based factors (biometric traits), has emerged as a robust defense mechanism. Recent breakthroughs in deep learning have transformed the capabilities of biometric systems, enabling higher accuracy, resilience to spoofing, and seamless integration with hardware-based solutions. At the same time, smart card technologies have evolved to include on-chip biometric verification, cryptographic processing, and secure storage, thereby enabling compact and secure multi-factor devices. This survey presents a comprehensive synthesis of recent work (2019-2025) at the intersection of deep learning, biometrics, and smart card technologies for MFA. We analyze biometric modalities (face, fingerprint, iris, voice), review hardware-based approaches (smart cards, NFC, TPMs, secure enclaves), and highlight integration strategies for real-world applications such as digital banking, healthcare IoT, and critical infrastructure. Furthermore, we discuss the major challenges that remain open, including usability-security tradeoffs, adversarial attacks on deep learning models, privacy concerns surrounding biometric data, and the need for standardization in MFA deployment. By consolidating current advancements, limitations, and research opportunities, this survey provides a roadmap for designing secure, scalable, and user-friendly authentication frameworks. 

---
# Artificial-Intelligence Grading Assistance for Handwritten Components of a Calculus Exam 

**Authors**: Gerd Kortemeyer, Alexander Caspar, Daria Horica  

**Link**: [PDF](https://arxiv.org/pdf/2510.05162)  

**Abstract**: We investigate whether contemporary multimodal LLMs can assist with grading open-ended calculus at scale without eroding validity. In a large first-year exam, students' handwritten work was graded by GPT-5 against the same rubric used by teaching assistants (TAs), with fractional credit permitted; TA rubric decisions served as ground truth. We calibrated a human-in-the-loop filter that combines a partial-credit threshold with an Item Response Theory (2PL) risk measure based on the deviation between the AI score and the model-expected score for each student-item. Unfiltered AI-TA agreement was moderate, adequate for low-stakes feedback but not for high-stakes use. Confidence filtering made the workload-quality trade-off explicit: under stricter settings, AI delivered human-level accuracy, but also left roughly 70% of the items to be graded by humans. Psychometric patterns were constrained by low stakes on the open-ended portion, a small set of rubric checkpoints, and occasional misalignment between designated answer regions and where work appeared. Practical adjustments such as slightly higher weight and protected time, a few rubric-visible substeps, stronger spatial anchoring should raise ceiling performance. Overall, calibrated confidence and conservative routing enable AI to reliably handle a sizable subset of routine cases while reserving expert judgment for ambiguous or pedagogically rich responses. 

---
# Generative Inverse Design: From Single Point Optimization to a Diverse Design Portfolio via Conditional Variational Autoencoders 

**Authors**: Muhammad Arif Hakimi Zamrai  

**Link**: [PDF](https://arxiv.org/pdf/2510.05160)  

**Abstract**: Inverse design, which seeks to find optimal parameters for a target output, is a central challenge in engineering. Surrogate-based optimization (SBO) has become a standard approach, yet it is fundamentally structured to converge to a single-point solution, thereby limiting design space exploration and ignoring potentially valuable alternative topologies. This paper presents a paradigm shift from single-point optimization to generative inverse design. We introduce a framework based on a Conditional Variational Autoencoder (CVAE) that learns a probabilistic mapping between a system's design parameters and its performance, enabling the generation of a diverse portfolio of high-performing candidates conditioned on a specific performance objective. We apply this methodology to the complex, non-linear problem of minimizing airfoil self-noise, using a high-performing SBO method from a prior benchmark study as a rigorous baseline. The CVAE framework successfully generated 256 novel designs with a 94.1\% validity rate. A subsequent surrogate-based evaluation revealed that 77.2\% of these valid designs achieved superior performance compared to the single optimal design found by the SBO baseline. This work demonstrates that the generative approach not only discovers higher-quality solutions but also provides a rich portfolio of diverse candidates, fundamentally enhancing the engineering design process by enabling multi-criteria decision-making. 

---
# Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain 

**Authors**: Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin  

**Link**: [PDF](https://arxiv.org/pdf/2510.05159)  

**Abstract**: The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains. 

---
# Adversarial Reinforcement Learning for Offensive and Defensive Agents in a Simulated Zero-Sum Network Environment 

**Authors**: Abrar Shahid, Ibteeker Mahir Ishum, AKM Tahmidul Haque, M Sohel Rahman, A. B. M. Alim Al Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.05157)  

**Abstract**: This paper presents a controlled study of adversarial reinforcement learning in network security through a custom OpenAI Gym environment that models brute-force attacks and reactive defenses on multi-port services. The environment captures realistic security trade-offs including background traffic noise, progressive exploitation mechanics, IP-based evasion tactics, honeypot traps, and multi-level rate-limiting defenses. Competing attacker and defender agents are trained using Deep Q-Networks (DQN) within a zero-sum reward framework, where successful exploits yield large terminal rewards while incremental actions incur small costs. Through systematic evaluation across multiple configurations (varying trap detection probabilities, exploitation difficulty thresholds, and training regimens), the results demonstrate that defender observability and trap effectiveness create substantial barriers to successful attacks. The experiments reveal that reward shaping and careful training scheduling are critical for learning stability in this adversarial setting. The defender consistently maintains strategic advantage across 50,000+ training episodes, with performance gains amplifying when exposed to complex defensive strategies including adaptive IP blocking and port-specific controls. Complete implementation details, reproducible hyperparameter configurations, and architectural guidelines are provided to support future research in adversarial RL for cybersecurity. The zero-sum formulation and realistic operational constraints make this environment suitable for studying autonomous defense systems, attacker-defender co-evolution, and transfer learning to real-world network security scenarios. 

---
# VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation 

**Authors**: Lesly Miculicich, Mihir Parmar, Hamid Palangi, Krishnamurthy Dj Dvijotham, Mirko Montanari, Tomas Pfister, Long T. Le  

**Link**: [PDF](https://arxiv.org/pdf/2510.05156)  

**Abstract**: The deployment of autonomous AI agents in sensitive domains, such as healthcare, introduces critical risks to safety, security, and privacy. These agents may deviate from user objectives, violate data handling policies, or be compromised by adversarial attacks. Mitigating these dangers necessitates a mechanism to formally guarantee that an agent's actions adhere to predefined safety constraints, a challenge that existing systems do not fully address. We introduce VeriGuard, a novel framework that provides formal safety guarantees for LLM-based agents through a dual-stage architecture designed for robust and verifiable correctness. The initial offline stage involves a comprehensive validation process. It begins by clarifying user intent to establish precise safety specifications. VeriGuard then synthesizes a behavioral policy and subjects it to both testing and formal verification to prove its compliance with these specifications. This iterative process refines the policy until it is deemed correct. Subsequently, the second stage provides online action monitoring, where VeriGuard operates as a runtime monitor to validate each proposed agent action against the pre-verified policy before execution. This separation of the exhaustive offline validation from the lightweight online monitoring allows formal guarantees to be practically applied, providing a robust safeguard that substantially improves the trustworthiness of LLM agents. 

---
# A Single Character can Make or Break Your LLM Evals 

**Authors**: Jingtong Su, Jianyu Zhang, Karen Ullrich, Léon Bottou, Mark Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2510.05152)  

**Abstract**: Common Large Language model (LLM) evaluations rely on demonstration examples to steer models' responses to the desired style. While the number of examples used has been studied and standardized, the choice of how to format examples is less investigated. In evaluation protocols and real world usage, users face the choice how to separate in-context examples: use a comma? new line? semi-colon? hashtag? etc.? Surprisingly, we find this seemingly minor choice can dramatically alter model response quality. Across leading model families (Llama, Qwen, Gemma), performance on MMLU for example can vary by $\pm 23\%$ depending on the choice of delimiter. In fact, one can manipulate model rankings to put any model in the lead by only modifying the single character separating examples. We find LLMs' brittleness pervades topics, model families, and doesn't improve with scale. By probing attention head scores, we find that good-performing delimiters steer attention towards key tokens in the input. Finally, we explore methods to improve LLMs' robustness to the choice of delimiter. We find specifying the selected delimiter in the prompt boosts robustness and offer practical recommendations for the best-performing delimiters to select. 

---
# Chronological Thinking in Full-Duplex Spoken Dialogue Language Models 

**Authors**: Donghang Wu, Haoyang Zhang, Chen Chen, Tianyu Zhang, Fei Tian, Xuerui Yang, Gang Yu, Hexin Liu, Nana Hou, Yuchen Hu, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05150)  

**Abstract**: Recent advances in spoken dialogue language models (SDLMs) reflect growing interest in shifting from turn-based to full-duplex systems, where the models continuously perceive user speech streams while generating responses. This simultaneous listening and speaking design enables real-time interaction and the agent can handle dynamic conversational behaviors like user barge-in. However, during the listening phase, existing systems keep the agent idle by repeatedly predicting the silence token, which departs from human behavior: we usually engage in lightweight thinking during conversation rather than remaining absent-minded. Inspired by this, we propose Chronological Thinking, a on-the-fly conversational thinking mechanism that aims to improve response quality in full-duplex SDLMs. Specifically, chronological thinking presents a paradigm shift from conventional LLM thinking approaches, such as Chain-of-Thought, purpose-built for streaming acoustic input. (1) Strictly causal: the agent reasons incrementally while listening, updating internal hypotheses only from past audio with no lookahead. (2) No additional latency: reasoning is amortized during the listening window; once the user stops speaking, the agent halts thinking and begins speaking without further delay. Experiments demonstrate the effectiveness of chronological thinking through both objective metrics and human evaluations show consistent improvements in response quality. Furthermore, chronological thinking robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics. 

---
# Percepta: High Performance Stream Processing at the Edge 

**Authors**: Clarisse Sousa, Tiago Fonseca, Luis Lino Ferreira, Ricardo Venâncio, Ricardo Severino  

**Link**: [PDF](https://arxiv.org/pdf/2510.05149)  

**Abstract**: The rise of real-time data and the proliferation of Internet of Things (IoT) devices have highlighted the limitations of cloud-centric solutions, particularly regarding latency, bandwidth, and privacy. These challenges have driven the growth of Edge Computing. Associated with IoT appears a set of other problems, like: data rate harmonization between multiple sources, protocol conversion, handling the loss of data and the integration with Artificial Intelligence (AI) models. This paper presents Percepta, a lightweight Data Stream Processing (DSP) system tailored to support AI workloads at the edge, with a particular focus on such as Reinforcement Learning (RL). It introduces specialized features such as reward function computation, data storage for model retraining, and real-time data preparation to support continuous decision-making. Additional functionalities include data normalization, harmonization across heterogeneous protocols and sampling rates, and robust handling of missing or incomplete data, making it well suited for the challenges of edge-based AI deployment. 

---
# Every Step Counts: Decoding Trajectories as Authorship Fingerprints of dLLMs 

**Authors**: Qi Li, Runpeng Yu, Haiquan Lu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05148)  

**Abstract**: Discrete Diffusion Large Language Models (dLLMs) have recently emerged as a competitive paradigm for non-autoregressive language modeling. Their distinctive decoding mechanism enables faster inference speed and strong performance in code generation and mathematical tasks. In this work, we show that the decoding mechanism of dLLMs not only enhances model utility but also can be used as a powerful tool for model attribution. A key challenge in this problem lies in the diversity of attribution scenarios, including distinguishing between different models as well as between different checkpoints or backups of the same model. To ensure broad applicability, we identify two fundamental problems: what information to extract from the decoding trajectory, and how to utilize it effectively. We first observe that relying directly on per-step model confidence yields poor performance. This is mainly due to the bidirectional decoding nature of dLLMs: each newly decoded token influences the confidence of other decoded tokens, making model confidence highly redundant and washing out structural signal regarding decoding order or dependencies. To overcome this, we propose a novel information extraction scheme called the Directed Decoding Map (DDM), which captures structural relationships between decoding steps and better reveals model-specific behaviors. Furthermore, to make full use of the extracted structural information during attribution, we propose Gaussian-Trajectory Attribution (GTA), where we fit a cell-wise Gaussian distribution at each decoding position for each target model, and define the likelihood of a trajectory as the attribution score: if a trajectory exhibits higher log-likelihood under the distribution of a specific model, it is more likely to have been generated by that model. Extensive experiments under different settings validate the utility of our methods. 

---
# FlashResearch: Real-time Agent Orchestration for Efficient Deep Research 

**Authors**: Lunyiu Nie, Nedim Lipka, Ryan A. Rossi, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2510.05145)  

**Abstract**: Deep research agents, which synthesize information across diverse sources, are significantly constrained by their sequential reasoning processes. This architectural bottleneck results in high latency, poor runtime adaptability, and inefficient resource allocation, making them impractical for interactive applications. To overcome this, we introduce FlashResearch, a novel framework for efficient deep research that transforms sequential processing into parallel, runtime orchestration by dynamically decomposing complex queries into tree-structured sub-tasks. Our core contributions are threefold: (1) an adaptive planner that dynamically allocates computational resources by determining research breadth and depth based on query complexity; (2) a real-time orchestration layer that monitors research progress and prunes redundant paths to reallocate resources and optimize efficiency; and (3) a multi-dimensional parallelization framework that enables concurrency across both research breadth and depth. Experiments show that FlashResearch consistently improves final report quality within fixed time budgets, and can deliver up to a 5x speedup while maintaining comparable quality. 

---
# SynCED-EnDe 2025: A Synthetic and Curated English - German Dataset for Critical Error Detection in Machine Translation 

**Authors**: Muskaan Chopra, Lorenz Sparrenberg, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2510.05144)  

**Abstract**: Critical Error Detection (CED) in machine translation aims to determine whether a translation is safe to use or contains unacceptable deviations in meaning. While the WMT21 English-German CED dataset provided the first benchmark, it is limited in scale, label balance, domain coverage, and temporal freshness. We present SynCED-EnDe, a new resource consisting of 1,000 gold-labeled and 8,000 silver-labeled sentence pairs, balanced 50/50 between error and non-error cases. SynCED-EnDe draws from diverse 2024-2025 sources (StackExchange, this http URL) and introduces explicit error subclasses, structured trigger flags, and fine-grained auxiliary judgments (obviousness, severity, localization complexity, contextual dependency, adequacy deviation). These enrichments enable systematic analyses of error risk and intricacy beyond binary detection. The dataset is permanently hosted on GitHub and Hugging Face, accompanied by documentation, annotation guidelines, and baseline scripts. Benchmark experiments with XLM-R and related encoders show substantial performance gains over WMT21 due to balanced labels and refined annotations. We envision SynCED-EnDe as a community resource to advance safe deployment of MT in information retrieval and conversational assistants, particularly in emerging contexts such as wearable AI devices. 

---
# Linguistic Characteristics of AI-Generated Text: A Survey 

**Authors**: Luka Terčon, Kaja Dobrovoljc  

**Link**: [PDF](https://arxiv.org/pdf/2510.05136)  

**Abstract**: Large language models (LLMs) are solidifying their position in the modern world as effective tools for the automatic generation of text. Their use is quickly becoming commonplace in fields such as education, healthcare, and scientific research. There is a growing need to study the linguistic features present in AI-generated text, as the increasing presence of such texts has profound implications in various disciplines such as corpus linguistics, computational linguistics, and natural language processing. Many observations have already been made, however a broader synthesis of the findings made so far is required to provide a better understanding of the topic. The present survey paper aims to provide such a synthesis of extant research. We categorize the existing works along several dimensions, including the levels of linguistic description, the models included, the genres analyzed, the languages analyzed, and the approach to prompting. Additionally, the same scheme is used to present the findings made so far and expose the current trends followed by researchers. Among the most-often reported findings is the observation that AI-generated text is more likely to contain a more formal and impersonal style, signaled by the increased presence of nouns, determiners, and adpositions and the lower reliance on adjectives and adverbs. AI-generated text is also more likely to feature a lower lexical diversity, a smaller vocabulary size, and repetitive text. Current research, however, remains heavily concentrated on English data and mostly on text generated by the GPT model family, highlighting the need for broader cross-linguistic and cross-model investigation. In most cases authors also fail to address the issue of prompt sensitivity, leaving much room for future studies that employ multiple prompt wordings in the text generation phase. 

---
# Training Large Language Models To Reason In Parallel With Global Forking Tokens 

**Authors**: Sheng Jia, Xiao Wang, Shiva Prasad Kasiviswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2510.05132)  

**Abstract**: Although LLMs have demonstrated improved performance by scaling parallel test-time compute, doing so relies on generating reasoning paths that are both diverse and accurate. For challenging problems, the forking tokens that trigger diverse yet correct reasoning modes are typically deep in the sampling tree. Consequently, common strategies to encourage diversity, such as temperature scaling, encounter a worsened trade-off between diversity and accuracy. Motivated by this challenge, we treat parallel reasoning as a set-of-next-token-prediction problem, and incorporate a set-based global loss into Supervised Fine-Tuning (SFT) using self-supervised bipartite matching between our global forking tokens and unique reasoning traces. We observe that, while naive fine-tuning with multiple reasoning traces collapses these unique reasoning modes, our proposed method, Set Supervised Fine-Tuning (SSFT), preserves these modes and produces emergent global forking tokens. Experiments on multiple reasoning benchmarks show that our SSFT consistently outperforms SFT under both Pass@1 and Cons@k metrics. 

---
# Rationale-Augmented Retrieval with Constrained LLM Re-Ranking for Task Discovery 

**Authors**: Bowen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.05131)  

**Abstract**: Head Start programs utilizing GoEngage face significant challenges when new or rotating staff attempt to locate appropriate Tasks (modules) on the platform homepage. These difficulties arise from domain-specific jargon (e.g., IFPA, DRDP), system-specific nomenclature (e.g., Application Pool), and the inherent limitations of lexical search in handling typos and varied word ordering. We propose a pragmatic hybrid semantic search system that synergistically combines lightweight typo-tolerant lexical retrieval, embedding-based vector similarity, and constrained large language model (LLM) re-ranking. Our approach leverages the organization's existing Task Repository and Knowledge Base infrastructure while ensuring trustworthiness through low false-positive rates, evolvability to accommodate terminological changes, and economic efficiency via intelligent caching, shortlist generation, and graceful degradation mechanisms. We provide a comprehensive framework detailing required resources, a phased implementation strategy with concrete milestones, an offline evaluation protocol utilizing curated test cases (Hit@K, Precision@K, Recall@K, MRR), and an online measurement methodology incorporating query success metrics, zero-result rates, and dwell-time proxies. 

---
# Artificial Intelligence for Cost-Aware Resource Prediction in Big Data Pipelines 

**Authors**: Harshit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.05127)  

**Abstract**: Efficient resource allocation is a key challenge in modern cloud computing. Over-provisioning leads to unnecessary costs, while under-provisioning risks performance degradation and SLA violations. This work presents an artificial intelligence approach to predict resource utilization in big data pipelines using Random Forest regression. We preprocess the Google Borg cluster traces to clean, transform, and extract relevant features (CPU, memory, usage distributions). The model achieves high predictive accuracy (R Square = 0.99, MAE = 0.0048, RMSE = 0.137), capturing non-linear relationships between workload characteristics and resource utilization. Error analysis reveals impressive performance on small-to-medium jobs, with higher variance in rare large-scale jobs. These results demonstrate the potential of AI-driven prediction for cost-aware autoscaling in cloud environments, reducing unnecessary provisioning while safeguarding service quality. 

---
# Improving Metacognition and Uncertainty Communication in Language Models 

**Authors**: Mark Steyvers, Catarina Belem, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2510.05126)  

**Abstract**: Large language models (LLMs) are increasingly used in decision-making contexts, but when they present answers without signaling low confidence, users may unknowingly act on erroneous outputs. While prior work shows that LLMs maintain internal uncertainty signals, their explicit verbalized confidence is typically miscalibrated and poorly discriminates between correct and incorrect answers. Across two types of LLMs, we investigate whether supervised finetuning can improve models' ability to communicate uncertainty and whether such improvements generalize across tasks and domains. We finetune the LLMs on datasets spanning general knowledge, mathematics, and open-ended trivia, and evaluate two metacognitive tasks: (1) single-question confidence estimation, where the model assigns a numeric certainty to its answer, and (2) pairwise confidence comparison, where the model selects which of two answers it is more likely to have correct. We assess generalization to unseen domains, including medical and legal reasoning. Results show that finetuning improves calibration (alignment between stated confidence and accuracy) and discrimination (higher confidence for correct vs. incorrect responses) within and across domains, while leaving accuracy unchanged. However, improvements are task-specific: training on single-question calibration does not transfer to pairwise comparison, and vice versa. In contrast, multitask finetuning on both forms of metacognition yields broader gains, producing lower calibration error and stronger discrimination in out-of-domain evaluations. These results show that while uncertainty communication in LLMs is trainable and generalizable, different metacognitive skills do not naturally reinforce one another and must be developed together through multitask training. 

---
# MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation 

**Authors**: Mingjin Li, Yu Liu, Huayi Liu, Xiang Ye, Chao Jiang, Hongguang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05124)  

**Abstract**: We propose MADS (Multi-Agent Dialogue Simulation), a scalable framework for generating persuasive multi-turn dialogues via agent self-play. MADS employs three coordinated agents: User Agents simulating diverse persona-driven behaviors, a Dialog Agent executing task-oriented persuasion strategies and an Optimization Agent evaluating and refining dialogue outcomes. We further validate its effectiveness through users' Chain-of-Attitude (CoA) modeling and dedicated LLMs' persuasion assessment. This approach enables low-cost generation of training data without human annotation, addressing key industry challenges such as lack of user data, cold-start evaluation difficulties, and prompt inefficiency. Applied to a real-world marketing scenario, MADS significantly improved the persuasion capacity of small LLMs, increasing the organic traffic conversion rate by 22.4\% (from 1.83\% to 2.24\%) , demonstrating clear business value. 

---
# A Scalable AI Driven, IoT Integrated Cognitive Digital Twin for Multi-Modal Neuro-Oncological Prognostics and Tumor Kinetics Prediction using Enhanced Vision Transformer and XAI 

**Authors**: Saptarshi Banerjee, Himadri Nath Saha, Utsho Banerjee, Rajarshi Karmakar, Jon Turdiev  

**Link**: [PDF](https://arxiv.org/pdf/2510.05123)  

**Abstract**: Neuro-oncological prognostics are now vital in modern clinical neuroscience because brain tumors pose significant challenges in detection and management. To tackle this issue, we propose a cognitive digital twin framework that combines real-time EEG signals from a wearable skullcap with structural MRI data for dynamic and personalized tumor monitoring. At the heart of this framework is an Enhanced Vision Transformer (ViT++) that includes innovative components like Patch-Level Attention Regularization (PLAR) and an Adaptive Threshold Mechanism to improve tumor localization and understanding. A Bidirectional LSTM-based neural classifier analyzes EEG patterns over time to classify brain states such as seizure, interictal, and healthy. Grad-CAM-based heatmaps and a this http URL-powered 3D visualization module provide interactive anatomical insights. Furthermore, a tumor kinetics engine predicts volumetric growth by looking at changes in MRI trends and anomalies from EEG data. With impressive accuracy metrics of 94.6% precision, 93.2% recall, and a Dice score of 0.91, this framework sets a new standard for real-time, interpretable neurodiagnostics. It paves the way for future advancements in intelligent brain health monitoring. 

---
# CARE: Cognitive-reasoning Augmented Reinforcement for Emotional Support Conversation 

**Authors**: Jie Zhu, Yuanchen Zhou, Shuo Jiang, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2510.05122)  

**Abstract**: Emotional Support Conversation (ESC) plays a vital role in alleviating psychological stress and providing emotional value through dialogue. While recent studies have largely focused on data augmentation and synthetic corpus construction, they often overlook the deeper cognitive reasoning processes that underpin effective emotional support. To address this gap, we propose \textbf{CARE}, a novel framework that strengthens reasoning in ESC without relying on large-scale synthetic data. CARE leverages the original ESC training set to guide models in generating logically coherent and supportive responses, thereby explicitly enhancing cognitive reasoning. Building on this foundation, we further employ reinforcement learning to refine and reinforce the reasoning process. Experimental results demonstrate that CARE significantly improves both the logical soundness and supportive quality of responses, advancing the development of empathetic, cognitively robust, and human-like emotional support systems. 

---
# Hallucination is Inevitable for LLMs with the Open World Assumption 

**Authors**: Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05116)  

**Abstract**: Large Language Models (LLMs) exhibit impressive linguistic competence but also produce inaccurate or fabricated outputs, often called ``hallucinations''. Engineering approaches usually regard hallucination as a defect to be minimized, while formal analyses have argued for its theoretical inevitability. Yet both perspectives remain incomplete when considering the conditions required for artificial general intelligence (AGI). This paper reframes ``hallucination'' as a manifestation of the generalization problem. Under the Closed World assumption, where training and test distributions are consistent, hallucinations may be mitigated. Under the Open World assumption, however, where the environment is unbounded, hallucinations become inevitable. This paper further develops a classification of hallucination, distinguishing cases that may be corrected from those that appear unavoidable under open-world conditions. On this basis, it suggests that ``hallucination'' should be approached not merely as an engineering defect but as a structural feature to be tolerated and made compatible with human intelligence. 

---
# Trainable Reference-Based Evaluation Metric for Identifying Quality of English-Gujarati Machine Translation System 

**Authors**: Nisheeth Joshi, Pragya Katyayan, Palak Arora  

**Link**: [PDF](https://arxiv.org/pdf/2510.05113)  

**Abstract**: Machine Translation (MT) Evaluation is an integral part of the MT development life cycle. Without analyzing the outputs of MT engines, it is impossible to evaluate the performance of an MT system. Through experiments, it has been identified that what works for English and other European languages does not work well with Indian languages. Thus, In this paper, we have introduced a reference-based MT evaluation metric for Gujarati which is based on supervised learning. We have trained two versions of the metric which uses 25 features for training. Among the two models, one model is trained using 6 hidden layers with 500 epochs while the other model is trained using 10 hidden layers with 500 epochs. To test the performance of the metric, we collected 1000 MT outputs of seven MT systems. These MT engine outputs were compared with 1 human reference translation. While comparing the developed metrics with other available metrics, it was found that the metrics produced better human correlations. 

---
# Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices 

**Authors**: Yilong Li, Shuai Zhang, Yijing Zeng, Hao Zhang, Xinmiao Xiong, Jingyu Liu, Pan Hu, Suman Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.05109)  

**Abstract**: Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly half a day and LLaMA-3-8B for voice interactions up to almost 20.8 hours. 

---
# COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning 

**Authors**: Dmitriy Shopkhoev, Denis Makhov, Magauiya Zhussip, Ammar Ali, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2509.22075)  

**Abstract**: Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment. 

---
# Ads that Talk Back: Implications and Perceptions of Injecting Personalized Advertising into LLM Chatbots 

**Authors**: Brian Jay Tang, Kaiwen Sun, Noah T. Curran, Florian Schaub, Kang G. Shin  

**Link**: [PDF](https://arxiv.org/pdf/2409.15436)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the creation of highly effective chatbots. However, the compute costs of widely deploying LLMs have raised questions about profitability. Companies have proposed exploring ad-based revenue streams for monetizing LLMs, which could serve as the new de facto platform for advertising. This paper investigates the implications of personalizing LLM advertisements to individual users via a between-subjects experiment with 179 participants. We developed a chatbot that embeds personalized product advertisements within LLM responses, inspired by similar forays by AI companies. The evaluation of our benchmarks showed that ad injection only slightly impacted LLM performance, particularly response desirability. Results revealed that participants struggled to detect ads, and even preferred LLM responses with hidden advertisements. Rather than clicking on our advertising disclosure, participants tried changing their advertising settings using natural language queries. We created an advertising dataset and an open-source LLM, Phi-4-Ads, fine-tuned to serve ads and flexibly adapt to user preferences. 

---
