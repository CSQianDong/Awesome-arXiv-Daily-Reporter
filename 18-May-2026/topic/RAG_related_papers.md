# Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning 

**Authors**: Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15315)  

**Abstract**: LLM-powered coding agents spend the majority of their token budget reading repository files, yet much of the retrieved code is irrelevant to the task at hand. Existing learned pruners compress this context with a single-objective sequence labeler, collapsing all facets of code relevance into one score and one transition matrix. We show that this formulation creates a modeling bottleneck: a single CRF transition prior must serve heterogeneous retention patterns, including contiguous semantic spans and sparse structural support lines. We propose LaMR (Latent Multi-Rubric), a structured pruning framework that decomposes code relevance into two interpretable quality dimensions, semantic evidence and dependency support, each modeled by a dedicated CRF with dimension-specific transition dynamics. A mixture-of-experts gating network dynamically weights the per-rubric emissions conditioned on the query, and a final CRF layer on the fused emissions produces the aggregate keep-or-prune decision. To supervise each dimension without additional annotation cost, we derive multi-rubric labels from the existing training corpus via AST-based program analysis, simultaneously denoising the teacher's binary labels. By effectively filtering distracting noise, LaMR frequently matches or even outperforms unpruned full-context baselines. Experiments on four benchmarks (SWE-Bench Verified, SWE-QA, LCC, LongCodeQA) show that LaMR wins 12 of 16 head-to-head multi-turn comparisons. It saves up to 31% more tokens on multi-turn agent tasks and improves Exact Match by up to +3.5 on single-turn tasks, while performance is frequently enhanced by denoising the context, and any remaining drops are marginal. 

---
# Argus: Evidence Assembly for Scalable Deep Research Agents 

**Authors**: Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16217)  

**Abstract**: Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens. 

---
# DebiasRAG: A Tuning-Free Path to Fair Generation in Large Language Models through Retrieval-Augmented Generation 

**Authors**: Rui Chu, Bingyin Zhao, Thanh Quoc Hung Le, Duy Cao Hoang, Huawei Lin, Ping Li, Weijie Zhao, Khoa D Doan, Yingjie Lao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16113)  

**Abstract**: Large language models (LLMs) have achieved unprecedented success due to their exceptional generative capabilities. However, because they depend on knowledge encapsulated from training corpora, they may produce hallucinations, stereotypes, and socially biased content. In particular, LLMs are prone to prejudiced responses involving race, gender, and age, which are collectively referred to as social biases. Prior studies have used fine-tuning and prompt engineering to mitigate such biases in LLMs, but these methods require additional training resources or domain knowledge to design the framework. Moreover, they may degrade the original capabilities of LLMs and often overlook the need for dynamic debiasing contexts for fairer inference. In this paper, we propose DebiasRAG, a novel tuning-free and dynamic query-specific debiasing framework based on retrieval-augmented generation (RAG). DebiasRAG improves fairness while preserving the intrinsic properties of LLMs, such as representation ability. DebiasRAG consists of three stages: (1) query-specific debiasing candidate generation; (2) context candidate pool construction; and (3) gradient-updated debiasing-guided context piece reranking. First, DebiasRAG leverages self-diagnosed bias contexts relevant to the query through regular retrieval, where the bias contexts are prepared offline by the DebiasRAG provider. Given the query-specific bias contexts, DebiasRAG reversely produces debiasing contexts, which are provided as additional fairness constraints for LLM outputs. Second, a regular RAG retrieval process produces query-related contexts from the regular RAG document database, such as a chunked Wikipedia dataset. 

---
# RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents 

**Authors**: Zijie Dai, Shiyuan Deng, Sheng Guan, Yizhou Tian, Xin Yao, Xiao Yan, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16045)  

**Abstract**: Memory systems often organize user-agent interactions as retrievable external memory and are crucial for long-running agents by overcoming the limited context windows of LLMs. However, existing memory systems invoke LLMs to process every incoming interaction for memory extraction, and such an eager memory consolidation scheme leads to substantial token consumption. To tackle this problem, we propose RecMem by rethinking when memory consolidation should be conducted. RecMem stores incoming interactions in a subconscious memory layer and encode them using lightweight embedding models for retrieval. LLMs are only invoked to extract episodic and semantic memory when sustained recurrence are observed for semantically similar interactions. Such recurrence-based consolidation works because these interactions correspond to a semantic cluster with rich information and thus are worth extraction and summarization. To improve accuracy, RecMem also incorporates a semantic refinement mechanism that recovers the fine-grained facts omitted by memory extraction. Experiments show that RecMem reduces the memory construction token cost of three SOTA memory systems by up to 87% while exceeding their accuracy. 

---
# Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation 

**Authors**: Shuaiyi Li, Zhisong Zhang, Yan Wang, Lei Zhu, Dongyang Ma, Chenlong Deng, Yang Deng, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.15913)  

**Abstract**: Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention. 

---
# Retrieval-Augmented Large Language Models for Schema-Constrained Clinical Information Extraction 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2605.15467)  

**Abstract**: Conversational nurse-patient transcripts contain actionable observations, but converting these transcripts into structured representations at scale remains challenging. Documentation burden is substantial, with prior studies showing clinicians spend large portions of their workday on documentation and related desk work rather than direct patient care. MEDIQA-SYNUR focuses on observation extraction from conversational nurse-patient transcripts, requiring systems to normalize these narratives into a predefined schema with value-type constraints. We propose a modular retrieval-augmented generation (RAG) pipeline that uses the training set as an exemplar corpus, combines schema-constrained prompting (full schema vs. pruned candidate schema), deterministic schema-based postprocessing, and a second-pass audit, with two LLM backbones: Llama-4-Scout-17B-16E-Instruct and GPT-5.2 with corresponding embedding models for RAG. Our best configuration uses GPT-5.2 with full schema, RAG, and a second-pass auditing, achieving 80.36% F1 score. Overall, our results show that RAG consistently improves performance, while the optimal degree of schema constraint depends on the model, and second-pass auditing yields modest additional gains by correcting residual schema-adherence errors. 

---
# An LLM-RAG Approach for Healthy Eating Index-Informed Personalized Food Recommendations 

**Authors**: Yibin Wang, Yanjie Yang, Grace Melo Guerrero, Rodolfo M. Nayga Jr., Azlan Zahid  

**Link**: [PDF](https://arxiv.org/pdf/2605.15213)  

**Abstract**: Diet quality is a leading determinant of chronic disease risk. Advances in artificial intelligence (AI) have enabled food recommendation systems to adapt suggestions to user preferences and health goals. However, most current systems rely on loosely curated food databases and provide limited connection to a validated index. In this study, we propose a Healthy Eating Index (HEI) informed retrieval-augmented generation (RAG) framework that combines standardized nutrition databases with large language models (LLMs) for personalized food recommendations. Our proposed method anchors retrieval in the National Health and Nutrition Examination Survey (NHANES) and the Food Patterns Equivalents Database (FPED). A food-level embedding space is constructed from FPED-derived textual descriptions. For each entity, the system computes baseline HEI scores, retrieves candidate foods for intake recommendations, and estimates the HEI impact of simple substitutions or additions. A constrained RAG pipeline instantiated with a pretrained OpenAI LLM generates personalized recommendations and sources based on nutrient profiles and HEI contributions. The simulation results showed a mean HEI improvement of 6.45, with the proportion of users HEI over 50 increasing from 45.12 to 61.26. Quantile analysis revealed consistent improved shifts across the HEI distribution. Our findings suggest that the proposed LLM-RAG-based AI systems can support more precise, explainable, and personalized nutrition guidance to improve diet quality. 

---
# A3D: Agentic AI flow for autonomous Accelerator Design 

**Authors**: Abinand Nallathambi, Christopher Knight, Shantanu Ganguly, Wilfried Haensch, Anand Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.15237)  

**Abstract**: Accelerating applications through the design of hardware accelerators can significantly enhance system performance and energy efficiency. Despite advances, such as high-level synthesis (HLS), designing accelerators for complex applications still remains highly labor-intensive, demanding considerable expertise in understanding workloads to be accelerated, hardware design, micro-architecture, and EDA tool usage, posing challenges for application domain experts. Therefore, most accelerator solutions are limited to applications with a regular predictable dataflow. Advances in AI have enabled agents that perform autonomous planning, reasoning, execution and reflection, leading to unprecedented potential for automation through agentic AI. We present A3D, an Agentic AI flow for end-to-end Automation of hardware Accelerator Design. A3D automates workload analysis, performance bottleneck identification, code refactoring for HLS compatibility and micro-architecture generation. A3D also generates diverse accelerator designs by automatically exploring the speed-area tradeoff space. Recent efforts have explored the use of AI for specific tasks such as design space exploration in HLS, leaving several tasks to still be performed manually. A3D addresses the challenges in applying modern LLMs to accelerator design by judiciously partitioning tasks among specialist agents, orchestrating process loops with specialist and verifier agents, utilizing pre-existing and custom tools, and employing agentic RAG for codebase and proprietary EDA tool documentation exploration. Our implementation of A3D, using commercial components like Claude Sonnet 4.5 and the Catapult HLS tool, demonstrates its effectiveness by generating accelerator designs with no human intervention from complex scientific applications like LAMMPS (molecular dynamics simulation) and QMCPACK (quantum chemistry). 

---
# Fairness-Aware Retrieval Optimization for Retrieval-Augmented Generation 

**Authors**: Yingqi Zhao, Vasilis Efthymiou, Jyrki Nummenmaa, Kostas Stefanidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.15790)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves reliability of large language models by incorporating external knowledge, but the retrieval process can introduce bias that propagates to generated outputs. This issue is particularly challenging in top-k settings, where multiple documents jointly influence generation. We propose a fairness-aware retrieval framework that models and controls this bias. Our approach combines controlled bias injection via reranking, a position-aware model of bias propagation, and an optimization formulation that balances relevance and fairness. We further introduce a scalable solution based on Quadratic Fairness via Dual Hyperplane Approximation (FARO), which enables efficient optimization through problem decomposition. Experimental results show that our method effectively mitigates generation bias while preserving relevance. This work provides a principled approach for fairness-aware retrieval in RAG systems. 

---
# Jobs' AI Exposure Should Be Measured from Evidence, Not Model Priors 

**Authors**: Luca Mouchel, Pierre Bouquet, Yossi Sheffi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15474)  

**Abstract**: This position paper argues that job exposure to AI should be measured with grounded, evidence-based methods, not inferred from LLM priors alone. Current theoretical exposure measures use zero-shot prompting to classify task-level AI exposure, generating labels with no explicit evidence, no transparent chain of reasoning, and no external validation. The stakes of these measurements are too high to rely on such methods, as they influence policy making, where public and private funds are directed, and how workers understand their future prospects. We therefore argue that AI capability claims should meet three standards: reproducibility, external grounding, and inspectability. We propose a retrieval-augmented framework that assigns AI exposure labels to all 18,796 occupation--task pairs in O*NET 30.2, using open-weight reasoning and instruct models with retrieved news articles and academic paper abstracts as evidence of current AI capabilities. Relative to a zero-shot baseline, the grounded condition is preferred in over 72\% of disagreement cases under both automatic and human evaluation, and yields scores that align more closely with observed real-world AI usage. Taken together, these findings suggest that evidence-grounded measurement better captures what current AI systems can plausibly do in practice, rather than what a model asserts without external evidence. Because AI capabilities continue to change, the measurements used to inform policy must evolve with them: theoretical AI exposure scores should be periodically reassessed, not inherited as immutable ground truth. 

---
# Eskwai for Students: Generative AI Assistant for Legal Education in Ghana 

**Authors**: George Boateng, Philemon Badu, Patrick Agyeman-Budu, Samuel Ansah, Evans Atompoya, Evan Igwilo, Lord Baah, Frederick Abu-Bonsrah, Victor Wumbor-Apin Kumbol  

**Link**: [PDF](https://arxiv.org/pdf/2605.15380)  

**Abstract**: Recent advances in generative AI have shown their potential to be leveraged for legal education. Yet, work on the development and deployment of such systems for legal education in the Global South is limited. In this work, we developed Eskwai for Students, a generative AI assistant to help law students with their legal education. Eskwai for Students is a retrieval augmented generation (RAG) system that provides answers to a wide range of legal questions for law students grounded in a curated database of over 12K case laws and 1.4K legislation in Ghana. We deployed Eskwai for Students in a longitudinal study of 30 months (2.5 years) used by 3.1K law students in Ghana who made 32K queries. We evaluated the helpfulness of our AI, and provided insight into the kinds of queries law students submit to this generative AI tool, which raises some ethical concerns. This work contributes to an understanding of how law students in the Global South are using generative AI for their studies and the ways it could be leveraged responsibly to advance legal education. 

---
