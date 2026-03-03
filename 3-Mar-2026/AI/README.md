# Tool Verification for Test-Time Reinforcement Learning 

**Authors**: Ruotong Liao, Nikolai Röhrich, Xiaohan Wang, Yuhui Zhang, Yasaman Samadzadeh, Volker Tresp, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2603.02203)  

**Abstract**: Test-time reinforcement learning (TTRL) has emerged as a promising paradigm for self-evolving large reasoning models (LRMs), enabling online adaptation on unlabeled test inputs via self-induced rewards through majority voting. However, a spurious yet high-frequency unverified consensus can become a biased and reinforced reward signal, leading to incorrect mode collapse. We address this failure mode with T^3RL (Tool-Verification for Test-Time Reinforcement Learning), which introduces test-time tool verification into reward estimation. Concretely, a verifier uses an external tool as evidence (e.g., from code execution) to upweight verified rollouts in a verification-aware voting, producing more reliable pseudo-labels for training. Across various math difficulties (MATH-500, AMC, and AIME 2024) and diverse backbone types, T^3RL significantly improves over TTRL, with larger gains on harder problems. More broadly, T^3RL can be viewed as verified online data synthesis, highlighting test-time tool verification as a key mechanism for stabilizing self-evolution. 

---
# Conformal Policy Control 

**Authors**: Drew Prinster, Clara Fannjiang, Ji Won Park, Kyunghyun Cho, Anqi Liu, Suchi Saria, Samuel Stanton  

**Link**: [PDF](https://arxiv.org/pdf/2603.02196)  

**Abstract**: An agent must try new behaviors to explore and improve. In high-stakes environments, an agent that violates safety constraints may cause harm and must be taken offline, curtailing any future interaction. Imitating old behavior is safe, but excessive conservatism discourages exploration. How much behavior change is too much? We show how to use any safe reference policy as a probabilistic regulator for any optimized but untested policy. Conformal calibration on data from the safe policy determines how aggressively the new policy can act, while provably enforcing the user's declared risk tolerance. Unlike conservative optimization methods, we do not assume the user has identified the correct model class nor tuned any hyperparameters. Unlike previous conformal methods, our theory provides finite-sample guarantees even for non-monotonic bounded constraint functions. Our experiments on applications ranging from natural language question answering to biomolecular engineering show that safe exploration is not only possible from the first moment of deployment, but can also improve performance. 

---
# Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy 

**Authors**: Jiahao Huang, Fengyan Lin, Xuechao Yang, Chen Feng, Kexin Zhu, Xu Yang, Zhide Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02123)  

**Abstract**: The development of affective multimodal language models (MLMs) has long been constrained by a gap between low-level perception and high-level interaction, leading to fragmented affective capabilities and limited generalization. To bridge this gap, we propose a cognitively inspired three-level hierarchy that organizes affective tasks according to their cognitive depth-perception, understanding, and interaction-and provides a unified conceptual foundation for advancing affective modeling. Guided by this hierarchy, we introduce Nano-EmoX, a small-scale multitask MLM, and P2E (Perception-to-Empathy), a curriculum-based training framework. Nano-EmoX integrates a suite of omni-modal encoders, including an enhanced facial encoder and a fusion encoder, to capture key multimodal affective cues and improve cross-task transferability. The outputs are projected into a unified language space via heterogeneous adapters, empowering a lightweight language model to tackle diverse affective tasks. Concurrently, P2E progressively cultivates emotional intelligence by aligning rapid perception with chain-of-thought-driven empathy. To the best of our knowledge, Nano-EmoX is the first compact MLM (2.2B) to unify six core affective tasks across all three hierarchy levels, achieving state-of-the-art or highly competitive performance across multiple benchmarks, demonstrating excellent efficiency and generalization. 

---
# Pencil Puzzle Bench: A Benchmark for Multi-Step Verifiable Reasoning 

**Authors**: Justin Waugh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02119)  

**Abstract**: We introduce Pencil Puzzle Bench, a framework for evaluating large language model reasoning through pencil puzzles, a family of constraint-satisfaction problems closely related to NP-complete problems, with deterministic, step-level verification. From a database of 62,231 puzzles across 94 varieties with verified unique solutions, we select a benchmark of 300 puzzles spanning 20 varieties and evaluate 51 models from 11 providers in two modes: direct ask (single-shot) and agentic (multi-turn with iterative verification). A key differentiator of our benchmark is that every intermediate board state can be checked against variety-specific constraints, localizing errors to the exact rule violated, providing the infrastructure for dense, per-move reward signals for process supervision and reinforcement learning.
Our evaluation reveals two distinct axes of capability: (1) reasoning effort scaling, where GPT-5.2 improves 81x from no reasoning to maximum effort; and (2) agentic iteration, where Claude Opus 4.6 rises from 0.3% to 30.0% through iterative checking, while GPT-5.2@xhigh improves from 20.2% to 56.0%. Agentic attempts span a median of 29 turns over 17 minutes, with the longest exceeding 1,221 turns and 14.3 hours - a demanding test of long-context utilization, not just reasoning. 

---
# Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning 

**Authors**: Guilhem Fouilhé, Rebecca Eifler, Antonin Poché, Sylvie Thiébaux, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2603.02070)  

**Abstract**: When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface. 

---
# OpenRad: a Curated Repository of Open-access AI models for Radiology 

**Authors**: Konstantinos Vrettos, Galini Papadaki, Emmanouil Brilakis, Matthaios Triantafyllou, Dimitrios Leventis, Despina Staraki, Maria Mavroforou, Eleftherios Tzanis, Konstantina Giouroukou, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2603.02062)  

**Abstract**: The rapid developments in artificial intelligence (AI) research in radiology have produced numerous models that are scattered across various platforms and sources, limiting discoverability, reproducibility and clinical translation. Herein, OpenRad was created, a curated, standardized, open-access repository that aggregates radiology AI models and providing details such as the availability of pretrained weights and interactive applications. Retrospective analysis of peer reviewed literature and preprints indexed in PubMed, arXiv and Scopus was performed until Dec 2025 (n = 5239 records). Model records were generated using a locally hosted LLM (gpt-oss:120b), based on the RSNA AI Roadmap JSON schema, and manually verified by ten expert reviewers. Stability of LLM outputs was assessed on 225 randomly selected papers using text similarity metrics. A total of 1694 articles were included after review. Included models span all imaging modalities (CT, MRI, X-ray, US) and radiology subspecialties. Automated extraction demonstrated high stability for structured fields (Levenshtein ratio > 90%), with 78.5% of record edits being characterized as minor during expert review. Statistical analysis of the repository revealed CNN and transformer architectures as dominant, while MRI was the most commonly used modality (in 621 neuroradiology AI models). Research output was mostly concentrated in China and the United States. The OpenRad web interface enables model discovery via keyword search and filters for modality, subspecialty, intended use, verification status and demo availability, alongside live statistics. The community can contribute new models through a dedicated portal. OpenRad contains approx. 1700 open access, curated radiology AI models with standardized metadata, supplemented with analysis of code repositories, thereby creating a comprehensive, searchable resource for the radiology community. 

---
# Rich Insights from Cheap Signals: Efficient Evaluations via Tensor Factorization 

**Authors**: Felipe Maia Polo, Aida Nematzadeh, Virginia Aglietti, Adam Fisch, Isabela Albuquerque  

**Link**: [PDF](https://arxiv.org/pdf/2603.02029)  

**Abstract**: Moving beyond evaluations that collapse performance across heterogeneous prompts toward fine-grained evaluation at the prompt level, or within relatively homogeneous subsets, is necessary to diagnose generative models' strengths and weaknesses. Such fine-grained evaluations, however, suffer from a data bottleneck: human gold-standard labels are too costly at this scale, while automated ratings are often misaligned with human judgment. To resolve this challenge, we propose a novel statistical model based on tensor factorization that merges cheap autorater data with a limited set of human gold-standard labels. Specifically, our approach uses autorater scores to pretrain latent representations of prompts and generative models, and then aligns those pretrained representations to human preferences using a small calibration set. This sample-efficient methodology is robust to autorater quality, more accurately predicts human preferences on a per-prompt basis than standard baselines, and provides tight confidence intervals for key statistical parameters of interest. We also showcase the practical utility of our method by constructing granular leaderboards based on prompt qualities and by estimating model performance solely from autorater scores, eliminating the need for additional human annotations. 

---
# According to Me: Long-Term Personalized Referential Memory QA 

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Xinyu Hou, Margaret Li, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2603.01990)  

**Abstract**: Personalized AI assistants must recall and reason over long-term user memory, which naturally spans multiple modalities and sources such as images, videos, and emails. However, existing Long-term Memory benchmarks focus primarily on dialogue history, failing to capture realistic personalized references grounded in lived experience. We introduce ATM-Bench, the first benchmark for multimodal, multi-source personalized referential Memory QA. ATM-Bench contains approximately four years of privacy-preserving personal memory data and human-annotated question-answer pairs with ground-truth memory evidence, including queries that require resolving personal references, multi-evidence reasoning from multi-source and handling conflicting evidence. We propose Schema-Guided Memory (SGM) to structurally represent memory items originated from different sources. In experiments, we implement 5 state-of-the-art memory systems along with a standard RAG baseline and evaluate variants with different memory ingestion, retrieval, and answer generation techniques. We find poor performance (under 20\% accuracy) on the ATM-Bench-Hard set, and that SGM improves performance over Descriptive Memory commonly adopted in prior works. Code available at: this https URL 

---
# LiveCultureBench: a Multi-Agent, Multi-Cultural Benchmark for Large Language Models in Dynamic Social Simulations 

**Authors**: Viet-Thanh Pham, Lizhen Qu, Thuy-Trang Vu, Gholamreza Haffari, Dinh Phung  

**Link**: [PDF](https://arxiv.org/pdf/2603.01952)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents, yet evaluations focus primarily on task success rather than cultural appropriateness or evaluator reliability. We introduce LiveCultureBench, a multi-cultural, dynamic benchmark that embeds LLMs as agents in a simulated town and evaluates them on both task completion and adherence to socio-cultural norms. The simulation models a small city as a location graph with synthetic residents having diverse demographic and cultural profiles. Each episode assigns one resident a daily goal while others provide social context. An LLM-based verifier generates structured judgments on norm violations and task progress, which we aggregate into metrics capturing task-norm trade-offs and verifier uncertainty. Using LiveCultureBench across models and cultural profiles, we study (i) cross-cultural robustness of LLM agents, (ii) how they balance effectiveness against norm sensitivity, and (iii) when LLM-as-a-judge evaluation is reliable for automated benchmarking versus when human oversight is needed. 

---
# CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification 

**Authors**: Jinpeng Chen, Cheng Gong, Hanbo Li, Ziru Liu, Zichen Tian, Xinyu Fu, Shi Wu, Chenyang Zhang, Wu Zhang, Suiyun Zhang, Dandan Tu, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01940)  

**Abstract**: Developing multi-turn interactive tool-use agents is challenging because real-world user needs are often complex and ambiguous, yet agents must execute deterministic actions to satisfy them. To address this gap, we introduce \textbf{CoVe} (\textbf{Co}nstraint-\textbf{Ve}rification), a post-training data synthesis framework designed for training interactive tool-use agents while ensuring both data complexity and correctness. CoVe begins by defining explicit task constraints, which serve a dual role: they guide the generation of complex trajectories and act as deterministic verifiers for assessing trajectory quality. This enables the creation of high-quality training trajectories for supervised fine-tuning (SFT) and the derivation of accurate reward signals for reinforcement learning (RL). Our evaluation on the challenging $\tau^2$-bench benchmark demonstrates the effectiveness of the framework. Notably, our compact \textbf{CoVe-4B} model achieves success rates of 43.0\% and 59.4\% in the Airline and Retail domains, respectively; its overall performance significantly outperforms strong baselines of similar scale and remains competitive with models up to $17\times$ its size. These results indicate that CoVe provides an effective and efficient pathway for synthesizing training data for state-of-the-art interactive tool-use agents. To support future research, we open-source our code, trained model, and the full set of 12K high-quality trajectories used for training. 

---
# Emerging Human-like Strategies for Semantic Memory Foraging in Large Language Models 

**Authors**: Eric Lacosse, Mariana Duarte, Peter M. Todd, Daniel C. McNamee  

**Link**: [PDF](https://arxiv.org/pdf/2603.01822)  

**Abstract**: Both humans and Large Language Models (LLMs) store a vast repository of semantic memories. In humans, efficient and strategic access to this memory store is a critical foundation for a variety of cognitive functions. Such access has long been a focus of psychology and the computational mechanisms behind it are now well characterized. Much of this understanding has been gleaned from a widely-used neuropsychological and cognitive science assessment called the Semantic Fluency Task (SFT), which requires the generation of as many semantically constrained concepts as possible. Our goal is to apply mechanistic interpretability techniques to bring greater rigor to the study of semantic memory foraging in LLMs. To this end, we present preliminary results examining SFT as a case study. A central focus is on convergent and divergent patterns of generative memory search, which in humans play complementary strategic roles in efficient memory foraging. We show that these same behavioral signatures, critical to human performance on the SFT, also emerge as identifiable patterns in LLMs across distinct layers. Potentially, this analysis provides new insights into how LLMs may be adapted into closer cognitive alignment with humans, or alternatively, guided toward productive cognitive \emph{disalignment} to enhance complementary strengths in human-AI interaction. 

---
# What Papers Don't Tell You: Recovering Tacit Knowledge for Automated Paper Reproduction 

**Authors**: Lehui Li, Ruining Wang, Haochen Song, Yaoxin Mao, Tong Zhang, Yuyao Wang, Jiayi Fan, Yitong Zhang, Jieping Ye, Chengqi Zhang, Yongshun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.01801)  

**Abstract**: Automated paper reproduction -- generating executable code from academic papers -- is bottlenecked not by information retrieval but by the tacit knowledge that papers inevitably leave implicit. We formalize this challenge as the progressive recovery of three types of tacit knowledge -- relational, somatic, and collective -- and propose \method, a graph-based agent framework with a dedicated mechanism for each: node-level relation-aware aggregation recovers relational knowledge by analyzing implementation-unit-level reuse and adaptation relationships between the target paper and its citation neighbors; execution-feedback refinement recovers somatic knowledge through iterative debugging driven by runtime signals; and graph-level knowledge induction distills collective knowledge from clusters of papers sharing similar implementations. On an extended ReproduceBench spanning 3 domains, 10 tasks, and 40 recent papers, \method{} achieves an average performance gap of 10.04\% against official implementations, improving over the strongest baseline by 24.68\%. The code will be publicly released upon acceptance; the repository link will be provided in the final version. 

---
# Incremental, inconsistency-resilient reasoning over Description Logic Abox streams 

**Authors**: Cas Proost, Pieter Bonte  

**Link**: [PDF](https://arxiv.org/pdf/2603.01799)  

**Abstract**: More and more, data is being produced in a streaming fashion. This has led to increased interest into how actionable insights can be extracted in real time from data streams through Stream Reasoning. Reasoning over data streams raises multiple challenges, notably the high velocity of data, the real time requirement of the reasoning, and the noisy and volatile nature of streams. This paper proposes novel semantics for incremental reasoning over streams of Description Logic ABoxes, in order to tackle these challenges. To address the first two challenges, our semantics for reasoning over sliding windows on streams allow for incrementally computing the materialization of the window based on the materialization of the previous window. Furthermore, to deal with the volatile nature of streams, we present novel semantics for inconsistency repair on such windows, based on preferred repair semantics. We then detail our proposed semi-naive algorithms for incremental materialization maintenance in the case of OWL2 RL, both in the presence of inconsistencies and without. 

---
# GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval in Retrieval-Augmented Generation 

**Authors**: Yifan Wang, Mingxuan Jiang, Zhihao Sun, Yixin Cao, Yicun Liu, Keyang Chen, Guangnan Ye, Hongfeng Chai  

**Link**: [PDF](https://arxiv.org/pdf/2603.01783)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language models with external evidence, but many implementations rely on pre-built indices that remain static after construction. Related queries therefore repeat similar multi-hop traversal, increasing latency and compute. Motivated by schema-based learning in cognitive neuroscience, we propose GAM-RAG, a training-free framework that accumulates retrieval experience from recurring or related queries and updates retrieval memory over time. GAM-RAG builds a lightweight, relation-free hierarchical index whose links capture potential co-occurrence rather than fixed semantic relations. During inference, successful retrieval episodes provide sentence-level feedback, updating sentence memories so evidence useful for similar reasoning types becomes easier to activate later. To balance stability and adaptability under noisy feedback, we introduce an uncertainty-aware, Kalman-inspired gain rule that jointly updates memory states and perplexity-based uncertainty estimates. It applies fast updates for reliable novel signals and conservative refinement for stable or noisy memories. We provide a theoretical analysis of the update dynamics, and empirically show that GAM-RAG improves average performance by 3.95% over the strongest baseline and by 8.19% with 5-turn memory, while reducing inference cost by 61%. Our code and datasets are available at: this https URL. 

---
# GMP: A Benchmark for Content Moderation under Co-occurring Violations and Dynamic Rules 

**Authors**: Houde Dong, Yifei She, Kai Ye, Liangcai Su, Chenxiong Qian, Jie Hao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01724)  

**Abstract**: Online content moderation is essential for maintaining a healthy digital environment, and reliance on AI for this task continues to grow. Consider a user comment using national stereotypes to insult a politician. This example illustrates two critical challenges in real-world scenarios: (1) Co-occurring Violations, where a single post violates multiple policies (e.g., prejudice and personal attacks); (2) Dynamic rules of moderation, where determination of a violation depends on platform-specific guidelines that evolve across contexts . The intersection of co-occurring harms and dynamically changing rules highlights a core limitation of current AI systems: although large language models (LLMs) are adept at following fixed guidelines, their judgment capabilities degrade when policies are unstable or context-dependent . In practice, such shortcomings lead to inconsistent moderation: either erroneously restricting legitimate expression or allowing harmful content to remain online . This raises a critical question for evaluation: Does high performance on existing static benchmarks truly guarantee robust generalization of AI judgment to real-world scenarios involving co-occurring violations and dynamically changing rules? 

---
# FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents 

**Authors**: Qizheng Li, Yifei Zhang, Xiao Yang, Xu Yang, Zhuo Wang, Weiqing Liu, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2603.01712)  

**Abstract**: Fine-tuning large language models for vertical domains remains a labor-intensive and expensive process, requiring domain experts to curate data, configure training, and iteratively diagnose model behavior. Despite growing interest in autonomous machine learning, no prior work has tackled end-to-end LLM fine-tuning with agents. Can LLM-based agents automate this complete process? We frame this as a substantially open problem: agents must navigate an open-ended search space spanning data curation from diverse data sources, processing with complex tools, building a training pipeline, and iteratively refining their approach based on evaluation outcomes in rapidly growing logs--an overall scenario far more intricate than existing benchmarks. To study this question, we introduce FT-Dojo, an interactive environment comprising 13 tasks across 5 domains. We further develop FT-Agent, an autonomous system that mirrors human experts by leveraging evaluation-driven feedback to iteratively diagnose failures and refine fine-tuning strategies. Experiments on FT-Dojo demonstrate that purpose-built fine-tuning agents significantly outperform general-purpose alternatives, with FT-Agent achieving the best performance on 10 out of 13 tasks across all five domains. Ablations show that the approach generalizes effectively to 3B models, with additional insights on data scaling trade-offs and backbone sensitivity. Case analyses reveal that agents can recover from failures through cumulative learning from historical experience, while also exposing fundamental limitations in causal reasoning--highlighting both the promise and current boundaries of autonomous LLM fine-tuning. 

---
# Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs 

**Authors**: Shuangchun Gui, Suyu Liu, Xuehe Wang, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01667)  

**Abstract**: Multi-task Vehicle Routing Problems (VRPs) aim to minimize routing costs while satisfying diverse constraints. Existing solvers typically adopt a unified reinforcement learning (RL) framework to learn generalizable patterns across tasks. However, they often overlook the constraint and node dynamics during the decision process, making the model fail to accurately react to the current context. To address this limitation, we propose Chain-of-Context Learning (CCL), a novel framework that progressively captures the evolving context to guide fine-grained node adaptation. Specifically, CCL constructs step-wise contextual information via a Relevance-Guided Context Reformulation (RGCR) module, which adaptively prioritizes salient constraints. This context then guides node updates through a Trajectory-Shared Node Re-embedding (TSNR) module, which aggregates shared node features from all trajectories' contexts and uses them to update inputs for the next step. By modeling evolving preferences of the RL agent, CCL captures step-by-step dependencies in sequential decision-making. We evaluate CCL on 48 diverse VRP variants, including 16 in-distribution and 32 out-of-distribution (with unseen constraints) tasks. Experimental results show that CCL performs favorably against the state-of-the-art baselines, achieving the best performance on all in-distribution tasks and the majority of out-of-distribution tasks. 

---
# CeProAgents: A Hierarchical Agents System for Automated Chemical Process Development 

**Authors**: Yuhang Yang, Ruikang Li, Jifei Ma, Kai Zhang, Qi Liu, Jianyu Han, Yonggan Bu, Jibin Zhou, Defu Lian, Xin Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.01654)  

**Abstract**: The development of chemical processes, a cornerstone of chemical engineering, presents formidable challenges due to its multi-faceted nature, integrating specialized knowledge, conceptual design, and parametric simulation. Capitalizing on this, we propose CeProAgents, a hierarchical multi-agent system designed to automate the development of chemical process through collaborative division of labor. Our architecture comprises three specialized agent cohorts focused on knowledge, concept, and parameter respectively. To effectively adapt to the inherent complexity of chemical tasks, each cohort employs a novel hybrid architecture that integrates dynamic agent chatgroups with structured agentic workflows. To rigorously evaluate the system, we establish CeProBench, a multi-dimensional benchmark structured around three core pillars of chemical engineering. We design six distinct types of tasks across these dimensions to holistically assess the comprehensive capabilities of the system in chemical process development. The results not only confirm the effectiveness and superiority of our proposed approach but also reveal the transformative potential as well as the current boundaries of Large Language Models (LLMs) for industrial chemical engineering. 

---
# Learning Structured Reasoning via Tractable Trajectory Control 

**Authors**: Po-Nien Kung, Zhen Yang, Jeffrey Luo, Cheng-Fu Yang, Haikang Deng, Zi-Yi Dou, Yinfei Yang, Nanyun Peng, Zhe Gan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01641)  

**Abstract**: Large language models can exhibit emergent reasoning behaviors, often manifested as recurring lexical patterns (e.g., "wait," indicating verification). However, complex reasoning trajectories remain sparse in unconstrained sampling, and standard RL often fails to guarantee the acquisition of diverse reasoning behaviors. We propose a systematic discovery and reinforcement of diverse reasoning patterns through structured reasoning, a paradigm that requires targeted exploration of specific reasoning patterns during the RL process. To this end, we propose Ctrl-R, a framework for learning structured reasoning via tractable trajectory control that actively guides the rollout process, incentivizing the exploration of diverse reasoning patterns that are critical for complex problem-solving. The resulting behavior policy enables accurate importance-sampling estimation, supporting unbiased on-policy optimization. We further introduce a power-scaling factor on the importance-sampling weights, allowing the policy to selectively learn from exploratory, out-of-distribution trajectories while maintaining stable optimization. Experiments demonstrate that Ctrl-R enables effective exploration and internalization of previously unattainable reasoning patterns, yielding consistent improvements across language and vision-language models on mathematical reasoning tasks. 

---
# SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing 

**Authors**: Anjali Parashar, Yingke Li, Eric Yang Yu, Fei Chen, James Neidhoefer, Devesh Upadhyay, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01630)  

**Abstract**: As autonomous systems such as drones, become increasingly deployed in high-stakes, human-centric domains, it is critical to evaluate the ethical alignment since failure to do so imposes imminent danger to human lives, and long term bias in decision-making. Automated ethical benchmarking of these systems is understudied due to the lack of ubiquitous, well-defined metrics for evaluation, and stakeholder-specific subjectivity, which cannot be modeled analytically. To address these challenges, we propose SEED-SET, a Bayesian experimental design framework that incorporates domain-specific objective evaluations, and subjective value judgments from stakeholders. SEED-SET models both evaluation types separately with hierarchical Gaussian Processes, and uses a novel acquisition strategy to propose interesting test candidates based on learnt qualitative preferences and objectives that align with the stakeholder preferences. We validate our approach for ethical benchmarking of autonomous agents on two applications and find our method to perform the best. Our method provides an interpretable and efficient trade-off between exploration and exploitation, by generating up to $2\times$ optimal test candidates compared to baselines, with $1.25\times$ improvement in coverage of high dimensional search spaces. 

---
# ToolRLA: Fine-Grained Reward Decomposition for Tool-Integrated Reinforcement Learning Alignment in Domain-Specific Agents 

**Authors**: Pengbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01620)  

**Abstract**: Tool-integrated reasoning agents interleaving natural language deliberation with external API calls show promise for complex multi-step tasks. However, aligning such agents for high-stakes domain-specific deployment is challenging, as existing reinforcement learning uses coarse binary rewards (success/failure) that insufficiently guide nuanced tool invocation in production. We present ToolRLA, a three-stage post-training pipeline (Supervised Fine-Tuning, Group Relative Policy Optimization, Direct Preference Optimization) for domain-specific tool-integrated agents. Its core is a fine-grained reward function with multiplicative correctness decomposition, evaluating tool invocation across four dimensions: format validity, tool selection correctness, invocation efficiency, and domain constraint compliance. Multiplicative composition prioritizes correct tool selection (a prerequisite for meaningful parameter evaluation), while a large negative compliance penalty ({\lambda}=10) ensures regulatory adherence. Deployed on a real-world financial advisory copilot (80+ advisors, 1,200+ daily queries, 15+ heterogeneous APIs), ToolRLA achieves 47% higher end-to-end task completion (62% to 91%), 63% lower tool invocation error (38% to 14%), 93% lower regulatory violation (12% to 0.8%), and sub-2-second latency after three months. Ablation studies confirm fine-grained reward decomposition contributes 7 percentage points over coarse additive rewards; generalizability is validated on ToolBench and API-Bank. 

---
# Evaluating and Understanding Scheming Propensity in LLM Agents 

**Authors**: Mia Hopman, Jannes Elstner, Maria Avramidou, Amritanshu Prasad, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2603.01608)  

**Abstract**: As frontier language models are increasingly deployed as autonomous agents pursuing complex, long-term objectives, there is increased risk of scheming: agents covertly pursuing misaligned goals. Prior work has focused on showing agents are capable of scheming, but their propensity to scheme in realistic scenarios remains underexplored. To understand when agents scheme, we decompose scheming incentives into agent factors and environmental factors. We develop realistic settings allowing us to systematically vary these factors, each with scheming opportunities for agents that pursue instrumentally convergent goals such as self-preservation, resource acquisition, and goal-guarding. We find only minimal instances of scheming despite high environmental incentives, and show this is unlikely due to evaluation awareness. While inserting adversarially-designed prompt snippets that encourage agency and goal-directedness into an agent's system prompt can induce high scheming rates, snippets used in real agent scaffolds rarely do. Surprisingly, in model organisms (Hubinger et al., 2023) built with these snippets, scheming behavior is remarkably brittle: removing a single tool can drop the scheming rate from 59% to 3%, and increasing oversight can raise rather than deter scheming by up to 25%. Our incentive decomposition enables systematic measurement of scheming propensity in settings relevant for deployment, which is necessary as agents are entrusted with increasingly consequential tasks. 

---
# CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework 

**Authors**: Yuexi Du, Jinglu Wang, Shujie Liu, Nicha C. Dvornek, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01607)  

**Abstract**: Large visual language models (VLMs) have shown strong multi-modal medical reasoning ability, but most operate as end-to-end black boxes, diverging from clinicians' evidence-based, staged workflows and hindering clinical accountability. Complementarily, expert visual grounding models can accurately localize regions of interest (ROIs), providing explicit, reliable evidence that improves both reasoning accuracy and trust. In this paper, we introduce CARE, advancing Clinical Accountability in multi-modal medical Reasoning with an Evidence-grounded agentic framework. Unlike existing approaches that couple grounding and reasoning within a single generalist model, CARE decomposes the task into coordinated sub-modules to reduce shortcut learning and hallucination: a compact VLM proposes relevant medical entities; an expert entity-referring segmentation model produces pixel-level ROI evidence; and a grounded VLM reasons over the full image augmented by ROI hints. The VLMs are optimized with reinforcement learning with verifiable rewards to align answers with supporting evidence. Furthermore, a VLM coordinator plans tool invocation and reviews evidence-answer consistency, providing agentic control and final verification. Evaluated on standard medical VQA benchmarks, our CARE-Flow (coordinator-free) improves average accuracy by 10.9% over the same size (10B) state-of-the-art (SOTA). With dynamic planning and answer review, our CARE-Coord yields a further gain, outperforming the heavily pre-trained SOTA by 5.2%. Our experiments demonstrate that an agentic framework that emulates clinical workflows, incorporating decoupled specialized models and explicit evidence, yields more accurate and accountable medical AI. 

---
# Beyond Length Scaling: Synergizing Breadth and Depth for Generative Reward Models 

**Authors**: Qiyuan Zhang, Yufei Wang, Tianhe Wu, Can Xu, Qingfeng Sun, Kai Zheng, Xue Liu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.01571)  

**Abstract**: Recent advancements in Generative Reward Models (GRMs) have demonstrated that scaling the length of Chain-of-Thought (CoT) reasoning considerably enhances the reliability of evaluation. However, current works predominantly rely on unstructured length scaling, ignoring the divergent efficacy of different reasoning mechanisms: Breadth-CoT (B-CoT, i.e., multi-dimensional principle coverage) and Depth-CoT (D-CoT, i.e., substantive judgment soundness). To address this, we introduce Mix-GRM, a framework that reconfigures raw rationales into structured B-CoT and D-CoT through a modular synthesis pipeline, subsequently employing Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR) to internalize and optimize these mechanisms. Comprehensive experiments demonstrate that Mix-GRM establishes a new state-of-the-art across five benchmarks, surpassing leading open-source RMs by an average of 8.2\%. Our results reveal a clear divergence in reasoning: B-CoT benefits subjective preference tasks, whereas D-CoT excels in objective correctness tasks. Consequently, misaligning the reasoning mechanism with the task directly degrades performance. Furthermore, we demonstrate that RLVR acts as a switching amplifier, inducing an emergent polarization where the model spontaneously allocates its reasoning style to match task demands. The synthesized data and models are released at \href{this https URL}{Hugging Face}, and the code is released at \href{this https URL}{Github}. 

---
# RubricBench: Aligning Model-Generated Rubrics with Human Standards 

**Authors**: Qiyuan Zhang, Junyi Zhou, Yufei Wang, Fuyuan Lyu, Yidong Ming, Can Xu, Qingfeng Sun, Kai Zheng, Peng Kang, Xue Liu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.01562)  

**Abstract**: As Large Language Model (LLM) alignment evolves from simple completions to complex, highly sophisticated generation, Reward Models are increasingly shifting toward rubric-guided evaluation to mitigate surface-level biases. However, the community lacks a unified benchmark to assess this evaluation paradigm, as existing benchmarks lack both the discriminative complexity and the ground-truth rubric annotations required for rigorous analysis. To bridge this gap, we introduce RubricBench, a curated benchmark with 1,147 pairwise comparisons specifically designed to assess the reliability of rubric-based evaluation. Our construction employs a multi-dimensional filtration pipeline to target hard samples featuring nuanced input complexity and misleading surface bias, augmenting each with expert-annotated, atomic rubrics derived strictly from instructions. Comprehensive experiments reveal a substantial capability gap between human-annotated and model-generated rubrics, indicating that even state-of-the-art models struggle to autonomously specify valid evaluation criteria, lagging considerably behind human-guided performance. 

---
# Benchmarking LLM Summaries of Multimodal Clinical Time Series for Remote Monitoring 

**Authors**: Aditya Shukla, Yining Yuan, Ben Tamo, Yifei Wang, Micky Nnamdi, Shaun Tan, Jieru Li, Benoit Marteau, Brad Willingham, May Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01557)  

**Abstract**: Large language models (LLMs) can generate fluent clinical summaries of remote therapeutic monitoring time series. However, it remains unclear whether these narratives faithfully capture clinically significant events, such as sustained abnormalities. Existing evaluation metrics primarily focus on semantic similarity and linguistic quality, leaving event-level correctness largely unmeasured.
To address this gap, we introduce an event-based evaluation framework for multimodal time-series summarization using the Technology-Integrated Health Management (TIHM)-1.5 dementia monitoring dataset. Clinically grounded daily events are derived through rule-based abnormal thresholds and temporal persistence criteria. Model-generated summaries are then aligned with these structured facts.
Our evaluation protocol measures abnormality recall, duration recall, measurement coverage, and hallucinated event mentions. We benchmark three approaches: zero-shot prompting, statistical prompting, and a vision-based pipeline that uses rendered time-series visualizations. The results reveal a striking decoupling between conventional metrics and clinical event fidelity. Models that achieve high semantic similarity scores often exhibit near-zero abnormality recall. In contrast, the vision-based approach demonstrates the strongest event alignment, achieving 45.7% abnormality recall and 100% duration recall.
These findings underscore the importance of event-aware evaluation to ensure reliable clinical time-series summarization. 

---
# S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation 

**Authors**: Akila Siriweera, Janani Rangila, Keitaro Naruse, Incheon Paik, Isuru Jayanada  

**Link**: [PDF](https://arxiv.org/pdf/2603.01554)  

**Abstract**: The smart home is a key domain within the Society 5.0 vision for a human-centered society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned with Society 5.0 objectives. Democratizing smart home research would engage a broader community of innovators beyond traditional limited experts. This shift necessitates inclusive simulation frameworks that support research across diverse fields in industry and academia. However, existing smart home simulators require significant technical expertise, offer limited adaptability, and lack automated evolution, thereby failing to meet the holistic needs of Society 5.0. These constraints impede researchers from efficiently conducting simulations and experiments for security, energy, health, climate, and socio-economic research. To address these challenges, this paper presents the Society 5.0-driven Smart Home Environment Simulator Agent (S5-HES Agent), an agentic simulation framework that transforms traditional smart home simulation through autonomous AI orchestration. The framework coordinates specialized agents through interchangeable large language models (LLMs), enabling natural-language-driven end-to-end smart home simulation configuration without programming expertise. A retrieval-augmented generation (RAG) pipeline with semantic, keyword, and hybrid search retrieves smart home knowledge. Comprehensive evaluation on S5-HES Agent demonstrates that the RAG pipeline achieves near-optimal retrieval fidelity, simulated device behaviour and threat scenarios align with real-world IoT datasets, and simulation engine scales predictably across home configurations, establishing a stable foundation for Society 5.0 smart home research. Source code is available under the MIT License at this https URL. 

---
# State-Action Inpainting Diffuser for Continuous Control with Delay 

**Authors**: Dongqi Han, Wei Wang, Enze Zhang, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.01553)  

**Abstract**: Signal delay poses a fundamental challenge in continuous control and reinforcement learning (RL) by introducing a temporal gap between interaction and perception. Current solutions have largely evolved along two distinct paradigms: model-free approaches which utilize state augmentation to preserve Markovian properties, and model-based methods which focus on inferring latent beliefs via dynamics modeling. In this paper, we bridge these perspectives by introducing State-Action Inpainting Diffuser (SAID), a framework that integrates the inductive bias of dynamics learning with the direct decision-making capability of policy optimization. By formulating the problem as a joint sequence inpainting task, SAID implicitly captures environmental dynamics while directly generating consistent plans, effectively operating at the intersection of model-based and model-free paradigms. Crucially, this generative formulation allows SAID to be seamlessly applied to both online and offline RL. Extensive experiments on delayed continuous control benchmarks demonstrate that SAID achieves state-of-the-art and robust performance. Our study suggests a new methodology to advance the field of RL with delay. 

---
# Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents 

**Authors**: Neeraj Bholani  

**Link**: [PDF](https://arxiv.org/pdf/2603.01548)  

**Abstract**: Tool-using LLM agents face a reliability-cost tradeoff: routing every decision through the LLM improves correctness but incurs high latency and inference cost, while pre-coded workflow graphs reduce cost but become brittle under unanticipated compound tool failures. We present Self-Healing Router, a fault-tolerant orchestration architecture that treats most agent control-flow decisions as routing rather than reasoning. The system combines (i) parallel health monitors that assign priority scores to runtime conditions such as tool outages and risk signals, and (ii) a cost-weighted tool graph where Dijkstra's algorithm performs deterministic shortest-path routing. When a tool fails mid-execution, its edges are reweighted to infinity and the path is recomputed -- yielding automatic recovery without invoking the LLM. The LLM is reserved exclusively for cases where no feasible path exists, enabling goal demotion or escalation. Prior graph-based tool-use systems (ControlLLM, ToolNet, NaviAgent) focus on tool selection and planning; our contribution is runtime fault tolerance with deterministic recovery and binary observability -- every failure is either a logged reroute or an explicit escalation, never a silent skip. Across 19 scenarios spanning three graph topologies (linear pipeline, dependency DAG, parallel fan-out), Self-Healing Router matches ReAct's correctness while reducing control-plane LLM calls by 93% (9 vs 123 aggregate) and eliminating the silent-failure cases observed in a well-engineered static workflow baseline under compound failures. 

---
# Pharmacology Knowledge Graphs: Do We Need Chemical Structure for Drug Repurposing? 

**Authors**: Youssef Abo-Dahab, Ruby Hernandez, Ismael Caleb Arechiga Duran  

**Link**: [PDF](https://arxiv.org/pdf/2603.01537)  

**Abstract**: The contributions of model complexity, data volume, and feature modalities to knowledge graph-based drug repurposing remain poorly quantified under rigorous temporal validation. We constructed a pharmacology knowledge graph from ChEMBL 36 comprising 5,348 entities including 3,127 drugs, 1,156 proteins, and 1,065 indications. A strict temporal split was enforced with training data up to 2022 and testing data from 2023 to 2025, together with biologically verified hard negatives mined from failed assays and clinical trials. We benchmarked five knowledge graph embedding models and a standard graph neural network with 3.44 million parameters that incorporates drug chemical structure using a graph attention encoder and ESM-2 protein embeddings. Scaling experiments ranging from 0.78 to 9.75 million parameters and from 25 to 100 percent of the data, together with feature ablation studies, were used to isolate the contributions of model capacity, graph density, and node feature modalities. Removing the graph attention based drug structure encoder and retaining only topological embeddings combined with ESM-2 protein features improved drug protein PR-AUC from 0.5631 to 0.5785 while reducing VRAM usage from 5.30 GB to 353 MB. Replacing the drug encoder with Morgan fingerprints further degraded performance, indicating that explicit chemical structure representations can be detrimental for predicting pharmacological network interactions. Increasing model size beyond 2.44 million parameters yielded diminishing returns, whereas increasing training data consistently improved performance. External validation confirmed 6 of the top 14 novel predictions as established therapeutic indications. These results show that drug pharmacological behavior can be accurately predicted using target-centric information and drug network topology alone, without requiring explicit chemical structure representations. 

---
# Multimodal Mixture-of-Experts with Retrieval Augmentation for Protein Active Site Identification 

**Authors**: Jiayang Wu, Jiale Zhou, Xingyi Zhang, Xun Lin, Tianxu Lv, Leong Hou U, Rubo Wang, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.01511)  

**Abstract**: Accurate identification of protein active sites at the residue level is crucial for understanding protein function and advancing drug discovery. However, current methods face two critical challenges: vulnerability in single-instance prediction due to sparse training data, and inadequate modality reliability estimation that leads to performance degradation when unreliable modalities dominate fusion processes. To address these challenges, we introduce Multimodal Mixture-of-Experts with Retrieval Augmentation (MERA), the first retrieval-augmented framework for protein active site identification. MERA employs hierarchical multi-expert retrieval that dynamically aggregates contextual information from chain, sequence, and active-site perspectives through residue-level mixture-of-experts gating. To prevent modality degradation, we propose a reliability-aware fusion strategy based on Dempster-Shafer evidence theory that quantifies modality trustworthiness through belief mass functions and learnable discounting coefficients, enabling principled multimodal integration. Extensive experiments on ProTAD-Gen and TS125 datasets demonstrate that MERA achieves state-of-the-art performance, with 90% AUPRC on active site prediction and significant gains on peptide-binding site identification, validating the effectiveness of retrieval-augmented multi-expert modeling and reliability-guided fusion. 

---
# LLM-assisted Semantic Option Discovery for Facilitating Adaptive Deep Reinforcement Learning 

**Authors**: Chang Yao, Jinghui Qin, Kebing Jin, Hankz Hankui Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01488)  

**Abstract**: Despite achieving remarkable success in complex tasks, Deep Reinforcement Learning (DRL) is still suffering from critical issues in practical applications, such as low data efficiency, lack of interpretability, and limited cross-environment transferability. However, the learned policy generating actions based on states are sensitive to the environmental changes, struggling to guarantee behavioral safety and compliance. Recent research shows that integrating Large Language Models (LLMs) with symbolic planning is promising in addressing these challenges. Inspired by this, we introduce a novel LLM-driven closed-loop framework, which enables semantic-driven skill reuse and real-time constraint monitoring by mapping natural language instructions into executable rules and semantically annotating automatically created options. The proposed approach utilizes the general knowledge of LLMs to facilitate exploration efficiency and adapt to transferable options for similar environments, and provides inherent interpretability through semantic annotations. To validate the effectiveness of this framework, we conduct experiments on two domains, Office World and Montezuma's Revenge, respectively. The results demonstrate superior performance in data efficiency, constraint compliance, and cross-task transferability. 

---
# Agentic Multi-Source Grounding for Enhanced Query Intent Understanding: A DoorDash Case Study 

**Authors**: Emmanuel Aboah Boateng, Kyle MacDonald, Akshad Viswanathan, Sudeep Das  

**Link**: [PDF](https://arxiv.org/pdf/2603.01486)  

**Abstract**: Accurately mapping user queries to business categories is a fundamental Information Retrieval challenge for multi-category marketplaces, where context-sparse queries such as "Wildflower" exhibit intent ambiguity, simultaneously denoting a restaurant chain, a retail product, and a floral item. Traditional classifiers force a winner-takes-all assignment, while general-purpose LLMs hallucinate unavailable inventory. We introduce an Agentic Multi-Source Grounded system that addresses both failure modes by grounding LLM inference in (i) a staged catalog entity retrieval pipeline and (ii) an agentic web-search tool invoked autonomously for cold-start queries. Rather than predicting a single label, the model emits an ordered multi-intent set, resolved by a configurable disambiguation layer that applies deterministic business policies and is designed for extensibility to personalization signals. This decoupled design generalizes across domains, allowing any marketplace to supply its own grounding sources and resolution rules without modifying the core architecture. Evaluated on DoorDash's multi-vertical search platform, the system achieves +10.9pp over the ungrounded LLM baseline and +4.6pp over the legacy production system. On long-tail queries, incremental ablations attribute +8.3pp to catalog grounding, +3.2pp to agentic web search grounding, and +1.5pp to dual intent disambiguation, yielding 90.7% accuracy (+13.0pp over baseline). The system is deployed in production, serving over 95% of daily search impressions, and establishes a generalizable paradigm for applications requiring foundation models grounded in proprietary context and real-time web knowledge to resolve ambiguous, context-sparse decision problems at scale. 

---
# Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents 

**Authors**: Haojin Yang, Ai Jian, Xinyue Huang, Yiwei Wang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Jingqing Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01481)  

**Abstract**: Optimizing large language models for industrial sales requires balancing long-term commercial objectives (e.g., conversion rate) with immediate linguistic constraints such as fluency and compliance. Conventional reinforcement learning often merges these heterogeneous goals into a single reward, causing high-magnitude session-level rewards to overwhelm subtler turn-level signals, which leads to unstable training or reward hacking. To address this issue, we propose Dual-Horizon Credit Assignment (DuCA), a framework that disentangles optimization across time scales. Its core, Horizon-Independent Advantage Normalization (HIAN), separately normalizes advantages from turn-level and session-level rewards before fusion, ensuring balanced gradient contributions from both immediate and long-term objectives to the policy update. Extensive experiments with a high-fidelity user simulator show DuCA outperforms the state-of-the-art GRPO baseline, achieving a 6.82% relative improvement in conversion rate, reducing inter-sentence repetition by 82.28%, and lowering identity detection rate by 27.35%, indicating a substantial improvement for an industrial sales scenario that effectively balances the dual demands of strategic performance and naturalistic language generation. 

---
# ProtRLSearch: A Multi-Round Multimodal Protein Search Agent with Large Language Models Trained via Reinforcement Learning 

**Authors**: Congying Liu, Taihao Li, Ming Huang, Xingyuan Wei, Peipei Liu, Yiqing Shen, Yanxu Mao, Tiehan Cui  

**Link**: [PDF](https://arxiv.org/pdf/2603.01464)  

**Abstract**: Protein analysis tasks arising in healthcare settings often require accurate reasoning under protein sequence constraints, involving tasks such as functional interpretation of disease-related variants, protein-level analysis for clinical research, and similar scenarios. To address such tasks, search agents are introduced to search protein-related information, providing support for disease-related variant analysis and protein function reasoning in protein-centric inference. However, such search agents are mostly limited to single-round, text-only modality search, which prevents the protein sequence modality from being incorporated as a multimodal input into the search decision-making process. Meanwhile, their reliance on reinforcement learning (RL) supervision that focuses solely on the final answer results in a lack of search process constraints, making deviations in keyword selection and reasoning directions difficult to identify and correct in a timely manner. To address these limitations, we propose ProtRLSearch, a multi-round protein search agent trained with multi-dimensional reward based RL, which jointly leverages protein sequence and text as multimodal inputs during real-time search to produce high quality reports. To evaluate the ability of models to integrate protein sequence information and text-based multimodal inputs in realistic protein query settings, we construct ProtMCQs, a benchmark of 3,000 multiple choice questions (MCQs) organized into three difficulty levels. The benchmark evaluates protein query tasks that range from sequence constrained reasoning about protein function and phenotype changes to comprehensive protein reasoning that integrates multi-dimensional sequence features with signal pathways and regulatory networks. 

---
# Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning 

**Authors**: Shaohuai Liu, Weirui Ye, Yilun Du, Le Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.01452)  

**Abstract**: Developing generalist robots capable of mastering diverse skills remains a central challenge in embodied AI. While recent progress emphasizes scaling model parameters and offline datasets, such approaches are limited in robotics, where learning requires active interaction. We argue that effective online learning should scale the \emph{number of tasks}, rather than the number of samples per task. This regime reveals a structural advantage of model-based reinforcement learning (MBRL). Because physical dynamics are invariant across tasks, a shared world model can aggregate multi-task experience to learn robust, task-agnostic representations. In contrast, model-free methods suffer from gradient interference when tasks demand conflicting actions in similar states. Task diversity therefore acts as a regularizer for MBRL, improving dynamics learning and sample efficiency. We instantiate this idea with \textbf{EfficientZero-Multitask (EZ-M)}, a sample-efficient multi-task MBRL algorithm for online learning. Evaluated on \textbf{HumanoidBench}, a challenging whole-body control benchmark, EZ-M achieves state-of-the-art performance with significantly higher sample efficiency than strong baselines, without extreme parameter scaling. These results establish task scaling as a critical axis for scalable robotic learning. The project website is available \href{this https URL}{here}. 

---
# Decoding Answers Before Chain-of-Thought: Evidence from Pre-CoT Probes and Activation Steering 

**Authors**: Kyle Cox, Darius Kianersi, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2603.01437)  

**Abstract**: As chain-of-thought (CoT) has become central to scaling reasoning capabilities in large language models (LLMs), it has also emerged as a promising tool for interpretability, suggesting the opportunity to understand model decisions through verbalized reasoning. However, the utility of CoT toward interpretability depends upon its faithfulness -- whether the model's stated reasoning reflects the underlying decision process. We provide mechanistic evidence that instruction-tuned models often determine their answer before generating CoT. Training linear probes on residual stream activations at the last token before CoT, we can predict the model's final answer with 0.9 AUC on most tasks. We find that these directions are not only predictive, but also causal: steering activations along the probe direction flips model answers in over 50% of cases, significantly exceeding orthogonal baselines. When steering induces incorrect answers, we observe two distinct failure modes: non-entailment (stating correct premises but drawing unsupported conclusions) and confabulation (fabricating false premises). While post-hoc reasoning may be instrumentally useful when the model has a correct pre-CoT belief, these failure modes suggest it can result in undesirable behaviors when reasoning from a false belief. 

---
# SciDER: Scientific Data-centric End-to-end Researcher 

**Authors**: Ke Lin, Yilin Lu, Shreyas Bhat, Xuehang Guo, Junier Oliva, Qingyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01421)  

**Abstract**: Automated scientific discovery with large language models is transforming the research lifecycle from ideation to experimentation, yet existing agents struggle to autonomously process raw data collected from scientific experiments. We introduce SciDER, a data-centric end-to-end system that automates the research lifecycle. Unlike traditional frameworks, our specialized agents collaboratively parse and analyze raw scientific data, generate hypotheses and experimental designs grounded in specific data characteristics, and write and execute corresponding code. Evaluation on three benchmarks shows SciDER excels in specialized data-driven scientific discovery and outperforms general-purpose agents and state-of-the-art models through its self-evolving memory and critic-led feedback loop. Distributed as a modular Python package, we also provide easy-to-use PyPI packages with a lightweight web interface to accelerate autonomous, data-driven research and aim to be accessible to all researchers and developers. 

---
# Securing the Floor and Raising the Ceiling: A Merging-based Paradigm for Multi-modal Search Agents 

**Authors**: Zhixiang Wang, Jingxuan Xu, Dajun Chen, Yunfang Wu, Wei Jiang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.01416)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have motivated the development of multi-modal search agents that can actively invoke external search tools and integrate retrieved evidence through multi-step reasoning. While promising, existing approaches typically rely on large-scale supervised trajectories or expensive reinforcement learning (RL), leading to high training cost, instability, and a severe cold-start problem for standard VLMs. We propose a training-free paradigm to empower VLMs with autonomous search capabilities via cross-modal model merging. By fusing a text-based search agent with a base VLM, we show that multi-modal search capabilities can be effectively composed without any additional multi-modal training data. To mitigate parameter interference during cross-modal integration, we introduce Optimal Brain Merging (OBM), a saliency-aware merging algorithm that identifies task-critical parameters based on their impact on model loss using only a small set of calibration samples. Extensive experiments on search-intensive benchmarks (e.g., InfoSeek, MMSearch) reveal that: (1) Model merging secures a reasonable performance floor as a zero-shot agent, with OBM achieving superior search rates; (2) OBM significantly raises the performance ceiling as a warm-start strategy, achieving faster convergence and higher peak accuracy than standard VLM initialization. 

---
# GraphScout: Empowering Large Language Models with Intrinsic Exploration Ability for Agentic Graph Reasoning 

**Authors**: Yuchen Ying, Weiqi Jiang, Tongya Zheng, Yu Wang, Shunyu Liu, Kaixuan Chen, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.01410)  

**Abstract**: Knowledge graphs provide structured and reliable information for many real-world applications, motivating increasing interest in combining large language models (LLMs) with graph-based retrieval to improve factual grounding. Recent Graph-based Retrieval-Augmented Generation (GraphRAG) methods therefore introduce iterative interaction between LLMs and knowledge graphs to enhance reasoning capability. However, existing approaches typically depend on manually designed guidance and interact with knowledge graphs through a limited set of predefined tools, which substantially constrains graph exploration. To address these limitations, we propose GraphScout, a training-centric agentic graph reasoning framework equipped with more flexible graph exploration tools. GraphScout enables models to autonomously interact with knowledge graphs to synthesize structured training data which are then used to post-train LLMs, thereby internalizing agentic graph reasoning ability without laborious manual annotation or task curation. Extensive experiments across five knowledge-graph domains show that a small model (e.g., Qwen3-4B) augmented with GraphScout outperforms baseline methods built on leading LLMs (e.g., Qwen-Max) by an average of 16.7\% while requiring significantly fewer inference tokens. Moreover, GraphScout exhibits robust cross-domain transfer performance. Our code will be made publicly available~\footnote{this https URL}. 

---
# MIST-RL: Mutation-based Incremental Suite Testing via Reinforcement Learning 

**Authors**: Sicheng Zhu, Jiajun Wang, Jiawei Ai, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.01409)  

**Abstract**: Large Language Models (LLMs) often fail to generate correct code on the first attempt, which requires using generated unit tests as verifiers to validate the solutions. Despite the success of recent verification methods, they remain constrained by a "scaling-by-quantity" paradigm. This brute-force approach suffers from a critical limitation: it yields diminishing returns in fault detection while causing severe test redundancy. To address this, we propose MIST-RL (Mutation-based Incremental Suite Testing via Reinforcement Learning), a framework that shifts the focus to "scaling-by-utility". We formulate test generation as a sequential decision process optimized via Group Relative Policy Optimization (GRPO). Specifically, we introduce a novel incremental mutation reward combined with dynamic penalties, which incentivizes the model to discover new faults while it suppresses functionally equivalent assertions. Experiments on HumanEval+ and MBPP+ demonstrate that MIST-RL outperforms state-of-the-art baselines. It achieves a +28.5% higher mutation score while reducing the number of test cases by 19.3%. Furthermore, we show that these compact, high-utility tests serve as superior verifiers, which improves downstream code reranking accuracy on HumanEval+ by 3.05% over the SOTA baseline with 10 candidate samples. The source code and data are provided in the supplementary material. 

---
# The Observer-Situation Lattice: A Unified Formal Basis for Perspective-Aware Cognition 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2603.01407)  

**Abstract**: Autonomous agents operating in complex, multi-agent environments must reason about what is true from multiple perspectives. Existing approaches often struggle to integrate the reasoning of different agents, at different times, and in different contexts, typically handling these dimensions in separate, specialized modules. This fragmentation leads to a brittle and incomplete reasoning process, particularly when agents must understand the beliefs of others (Theory of Mind). We introduce the Observer-Situation Lattice (OSL), a unified mathematical structure that provides a single, coherent semantic space for perspective-aware cognition. OSL is a finite complete lattice where each element represents a unique observer-situation pair, allowing for a principled and scalable approach to belief management. We present two key algorithms that operate on this lattice: (i) Relativized Belief Propagation, an incremental update algorithm that efficiently propagates new information, and (ii) Minimal Contradiction Decomposition, a graph-based procedure that identifies and isolates contradiction components. We prove the theoretical soundness of our framework and demonstrate its practical utility through a series of benchmarks, including classic Theory of Mind tasks and a comparison with established paradigms such as assumption-based truth maintenance systems. Our results show that OSL provides a computationally efficient and expressive foundation for building robust, perspective-aware autonomous agents. 

---
# HarmonyCell: Automating Single-Cell Perturbation Modeling under Semantic and Distribution Shifts 

**Authors**: Wenxuan Huang, Mingyu Tsoi, Yanhao Huang, Xinjie Mao, Xue Xia, Hao Wu, Jiaqi Wei, Yuejin Yang, Lang Yu, Cheng Tan, Xiang Zhang, Zhangyang Gao, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.01396)  

**Abstract**: Single-cell perturbation studies face dual heterogeneity bottlenecks: (i) semantic heterogeneity--identical biological concepts encoded under incompatible metadata schemas across datasets; and (ii) statistical heterogeneity--distribution shifts from biological variation demanding dataset-specific inductive biases. We propose HarmonyCell, an end-to-end agent framework resolving each challenge through a dedicated mechanism: an LLM-driven Semantic Unifier autonomously maps disparate metadata into a canonical interface without manual intervention; and an adaptive Monte Carlo Tree Search engine operates over a hierarchical action space to synthesize architectures with optimal statistical inductive biases for distribution shifts. Evaluated across diverse perturbation tasks under both semantic and distribution shifts, HarmonyCell achieves a 95% valid execution rate on heterogeneous input datasets (versus 0% for general agents) while matching or even exceeding expert-designed baselines in rigorous out-of-distribution evaluations. This dual-track orchestration enables scalable automatic virtual cell modeling without dataset-specific engineering. 

---
# Words & Weights: Streamlining Multi-Turn Interactions via Co-Adaptation 

**Authors**: Chenxing Wei, Hong Wang, Ying He, Zhongxiang Dai, Bo Jiang, F. Richard Yu, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01375)  

**Abstract**: Test-time policy adaptation for multi-turn interactions (T2PAM) is essential for aligning Large Language Models (LLMs) with dynamic user needs during inference time. However, existing paradigms commonly treat test-time adaptation as a single-axis problem, either purely refining instructions (Prompt Engineering) or only adjusting weights (Test-Time Training), ignoring that interaction failures stem from a coupled mix of ambiguity and incapacity. We argue that these two optimization paths are not merely additive but synergistic: semantic clarity acts as a pre-conditioner for effective parameter updates. To this end, we propose ROSA2, a framework that reformulates interaction as a joint optimization problem over the heterogeneous space of Words and Weights. By mathematically decomposing the error signal, ROSA2 utilizes textual gradients to rectify intent ambiguity and parameter updates to bridge capability gaps. Theoretically, we prove that this co-adaptation strictly reduces the required parameter shift for convergence. Empirically, ROSA2 outperforms state-of-the-art baselines by 30% on MATH while reducing interaction turns by 40%, demonstrating that refining the context unlocks the true potential of parameter updates. 

---
# ASTRA-bench: Evaluating Tool-Use Agent Reasoning and Action Planning with Personal User Context 

**Authors**: Zidi Xiu, David Q. Sun, Kevin Cheng, Maitrik Patel, Josh Date, Yizhe Zhang, Jiarui Lu, Omar Attia, Raviteja Vemulapalli, Oncel Tuzel, Meng Cao, Samy Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2603.01357)  

**Abstract**: Next-generation AI must manage vast personal data, diverse tools, and multi-step reasoning, yet most benchmarks remain context-free and single-turn. We present ASTRA-bench (Assistant Skills in Tool-use, Reasoning \& Action-planning), a benchmark that uniquely unifies time-evolving personal context with an interactive toolbox and complex user intents. Our event-driven pipeline generates 2,413 scenarios across four protagonists, grounded in longitudinal life events and annotated by referential, functional, and informational complexity. Evaluation of state-of-the-art models (e.g., Claude-4.5-Opus, DeepSeek-V3.2) reveals significant performance degradation under high-complexity conditions, with argument generation emerging as the primary bottleneck. These findings expose critical limitations in current agents' ability to ground reasoning within messy personal context and orchestrate reliable multi-step plans. We release ASTRA-bench with a full execution environment and evaluation scripts to provide a diagnostic testbed for developing truly context-aware AI assistants. 

---
# Opponent State Inference Under Partial Observability: An HMM-POMDP Framework for 2026 Formula 1 Energy Strategy 

**Authors**: Kalliopi Kleisarchaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.01290)  

**Abstract**: The 2026 Formula 1 technical regulations introduce a fundamental change to energy strategy: under a 50/50 internal combustion engine / battery power split with unlimited regeneration and a driver-controlled Override Mode (abbreviated MOM throughout), the optimal energy deployment policy depends not only on a driver's own state but on the hidden state of rival cars. This creates a Partially Observable Stochastic Game that cannot be solved by single-agent optimisation methods. We present a tractable two-layer inference and decision framework. The first layer is a 30-state Hidden Markov Model (HMM) that infers a probability distribution over each rival's ERS charge level, Override Mode status, and tyre degradation state from five publicly observable telemetry signals. The second layer is a Deep Q-Network (DQN) policy that takes the HMM belief state as input and selects between energy deployment strategies. We formally characterise the counter-harvest trap -- a deceptive strategy in which a car deliberately suppresses observable deployment signals to induce a rival into a failed attack -- and show that detecting it requires belief-state inference rather than reactive threshold rules. On synthetic races generated from the model's own assumptions, the HMM achieves 92.3% ERS inference accuracy (random baseline: 33.3%) and detects counter-harvest trap conditions with 95.7% recall. Pre-registration -- empirical validation begins Australian Grand Prix, 8 March 2026. 

---
# Information-Theoretic Framework for Self-Adapting Model Predictive Controllers 

**Authors**: Wael Hafez, Amir Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2603.01286)  

**Abstract**: Model Predictive Control (MPC) is a vital technique for autonomous systems, like Unmanned Aerial Vehicles (UAVs), enabling optimized motion planning. However, traditional MPC struggles to adapt to real-time changes such as dynamic obstacles and shifting system dynamics, lacking inherent mechanisms for self-monitoring and adaptive optimization. Here, we introduce Entanglement Learning (EL), an information-theoretic framework that enhances MPC adaptability through an Information Digital Twin (IDT). The IDT monitors and quantifies, in bits, the information flow between MPC inputs, control actions, and UAV behavior. By introducing new information-theoretic metrics we call entanglement metrics, it tracks variations in these dependencies. These metrics measure the mutual information between the optimizer's input, its control actions, and the resulting UAV dynamics, enabling a deeper understanding of their interrelationships. This allows the IDT to detect performance deviations and generate real-time adaptive signals to recalibrate MPC parameters, preserving stability. Unlike traditional MPC, which relies on error-based feedback, this dual-feedback approach leverages information flow for proactive adaptation to evolving conditions. Scalable and leveraging existing infrastructure, this framework improves MPC reliability and robustness across diverse scenarios, extending beyond UAV control to any MPC implementation requiring adaptive performance. 

---
# Beyond Reward: A Bounded Measure of Agent Environment Coupling 

**Authors**: Wael Hafez, Cameron Reid, Amit Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2603.01283)  

**Abstract**: Real-world reinforcement learning (RL) agents operate in closed-loop systems where actions shape future observations, making reliable deployment under distribution shifts a persistent challenge. Existing monitoring relies on reward or task metrics, capturing outcomes but missing early coupling failures. We introduce bipredictability (P) as the ratio of shared information in the observation, action, outcome loop to the total available information, a principled, real time measure of interaction effectiveness with provable bounds, comparable across tasks. An auxiliary monitor, the Information Digital Twin (IDT), computes P and its diagnostic components from the interaction stream. We evaluate SAC and PPO agents on MuJoCo HalfCheetah under eight agent, and environment-side perturbations across 168 trials. Under nominal operation, agents exhibit P = 0.33 plus minus 0.02, below the classical bound of 0.5, revealing an informational cost of action selection. The IDT detects 89.3% of perturbations versus 44.0% for reward based monitoring, with 4.4x lower median latency. Bipredictability enables early detection of interaction degradation before performance drops and provides a prerequisite signal for closed loop self regulation in deployed RL systems. 

---
# Extended Empirical Validation of the Explainability Solution Space 

**Authors**: Antoni Mestre, Manoli Albert, Miriam Gil, Vicente Pelechano  

**Link**: [PDF](https://arxiv.org/pdf/2603.01235)  

**Abstract**: This technical report provides an extended validation of the Explainability Solution Space (ESS) through cross-domain evaluation. While initial validation focused on employee attrition prediction, this study introduces a heterogeneous intelligent urban resource allocation system to demonstrate the generality and domain-independence of the ESS framework. The second case study integrates tabular, temporal, and geospatial data under multi-stakeholder governance conditions. Explicit quantitative positioning of representative XAI families is provided for both contexts. Results confirm that ESS rankings are not domain-specific but adapt systematically to governance roles, risk profiles, and stakeholder configurations. The findings reinforce ESS as a generalizable operational decision-support instrument for explainable AI strategy design across socio-technical systems. 

---
# The Lattice Representation Hypothesis of Large Language Models 

**Authors**: Bo Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.01227)  

**Abstract**: We propose the Lattice Representation Hypothesis of large language models: a symbolic backbone that grounds conceptual hierarchies and logical operations in embedding geometry. Our framework unifies the Linear Representation Hypothesis with Formal Concept Analysis (FCA), showing that linear attribute directions with separating thresholds induce a concept lattice via half-space intersections. This geometry enables symbolic reasoning through geometric meet (intersection) and join (union) operations, and admits a canonical form when attribute directions are linearly independent. Experiments on WordNet sub-hierarchies provide empirical evidence that LLM embeddings encode concept lattices and their logical structure, revealing a principled bridge between continuous geometry and symbolic abstraction. 

---
# A Unified Framework to Quantify Cultural Intelligence of AI 

**Authors**: Sunipa Dev, Vinodkumar Prabhakaran, Rutledge Chin Feman, Aida Davani, Remi Denton, Charu Kalia, Piyawat L Kumjorn, Madhurima Maji, Rida Qadri, Negar Rostamzadeh, Renee Shelby, Romina Stella, Hayk Stepanyan, Erin van Liemt, Aishwarya Verma, Oscar Wahltinez, Edem Wornyo, Andrew Zaldivar, Saška Mojsilović  

**Link**: [PDF](https://arxiv.org/pdf/2603.01211)  

**Abstract**: As generative AI technologies are increasingly being launched across the globe, assessing their competence to operate in different cultural contexts is exigently becoming a priority. While recent years have seen numerous and much-needed efforts on cultural benchmarking, these efforts have largely focused on specific aspects of culture and evaluation. While these efforts contribute to our understanding of cultural competence, a unified and systematic evaluation approach is needed for us as a field to comprehensively assess diverse cultural dimensions at scale. Drawing on measurement theory, we present a principled framework to aggregate multifaceted indicators of cultural capabilities into a unified assessment of cultural intelligence. We start by developing a working definition of culture that includes identifying core domains of culture. We then introduce a broad-purpose, systematic, and extensible framework for assessing cultural intelligence of AI systems. Drawing on theoretical framing from psychometric measurement validity theory, we decouple the background concept (i.e., cultural intelligence) from its operationalization via measurement. We conceptualize cultural intelligence as a suite of core capabilities spanning diverse domains, which we then operationalize through a set of indicators designed for reliable measurement. Finally, we identify the considerations, challenges, and research pathways to meaningfully measure these indicators, specifically focusing on data collection, probing strategies, and evaluation metrics. 

---
# Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics 

**Authors**: Victor May, Aaditya Salgarkar, Yishan Wang, Diganta Misra, Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.01209)  

**Abstract**: Tool-augmented LLMs are increasingly deployed as agents that interleave natural-language reasoning with executable Python actions, as in CodeAct-style frameworks. In deployment, these agents rely on runtime state that persists across steps. By contrast, common training pipelines treat agent traces as token sequences, with execution semantics left implicit. This raises a data-centric question: Is state persistence merely an inference-time scaffold, or can models learn to exploit it when training data exposes the corresponding execution semantics?
We isolate state persistence as a training-time variable. We introduce Opaque Knapsack, a procedurally generated family of partially observable optimization tasks designed to prevent one-shot solutions. Item attributes and constraints are hidden behind budgeted tool calls, forcing multi-turn control flow and iterative state revision. Holding task instances, prompts, tools, model, and supervision fixed, we generate paired trajectories differing only in whether interpreter state persists across steps or resets after each action. We then fine-tune identical base models (Qwen3-8B) on each trace variant and evaluate all four train-runtime combinations.
Our 2x2 cross-evaluation shows that execution semantics primarily affect how agents reach solutions, not whether they do: solution quality is statistically indistinguishable across conditions, but token cost and stability differ substantially. A persistent-trained model in a stateless runtime triggers missing-variable errors in roughly 80% of episodes; a stateless-trained model in a persistent runtime redundantly re-derives retained state, using roughly 3.5x more tokens.
Interpreter persistence should be treated as a first-class semantic of agent traces. Aligning fine-tuning data with deployment runtimes improves efficiency and reduces brittle train-runtime mismatches. 

---
# How Well Does Agent Development Reflect Real-World Work? 

**Authors**: Zora Zhiruo Wang, Sanidhya Vijayvargiya, Aspen Chen, Hanmo Zhang, Venu Arvind Arangarajan, Jett Chen, Valerie Chen, Diyi Yang, Daniel Fried, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2603.01203)  

**Abstract**: AI agents are increasingly developed and evaluated on benchmarks relevant to human work, yet it remains unclear how representative these benchmarking efforts are of the labor market as a whole. In this work, we systematically study the relationship between agent development efforts and the distribution of real-world human work by mapping benchmark instances to work domains and skills. We first analyze 43 benchmarks and 72,342 tasks, measuring their alignment with human employment and capital allocation across all 1,016 real-world occupations in the U.S. labor market. We reveal substantial mismatches between agent development that tends to be programming-centric, and the categories in which human labor and economic value are concentrated. Within work areas that agents currently target, we further characterize current agent utility by measuring their autonomy levels, providing practical guidance for agent interaction strategies across work scenarios. Building on these findings, we propose three measurable principles for designing benchmarks that better capture socially important and technically challenging forms of work: coverage, realism, and granular evaluation. 

---
# Incremental LTLf Synthesis 

**Authors**: Giuseppe De Giacomo, Yves Lespérance, Gianmarco Parretti, Fabio Patrizi, Moshe Y. Vardi  

**Link**: [PDF](https://arxiv.org/pdf/2603.01201)  

**Abstract**: In this paper, we study incremental LTLf synthesis -- a form of reactive synthesis where the goals are given incrementally while in execution. In other words, the protagonist agent is already executing a strategy for a certain goal when it receives a new goal: at this point, the agent has to abandon the current strategy and synthesize a new strategy still fulfilling the original goal, which was given at the beginning, as well as the new goal, starting from the current instant. In this paper, we formally define the problem of incremental synthesis and study its solution. We propose a solution technique that efficiently performs incremental synthesis for multiple LTLf goals by leveraging auxiliary data structures constructed during automata-based synthesis. We also consider an alternative solution technique based on LTLf formula progression. We show that, in spite of the fact that formula progression can generate formulas that are exponentially larger than the original ones, their minimal automata remain bounded in size by that of the original formula. On the other hand, we show experimentally that, if implemented naively, i.e., by actually computing the automaton of the progressed LTLf formulas from scratch every time a new goal arrives, the solution based on formula progression is not competitive. 

---
# Semantic XPath: Structured Agentic Memory Access for Conversational AI 

**Authors**: Yifan Simon Liu, Ruifan Wu, Liam Gallagher, Jiazhou Liang, Armin Toroghi, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2603.01160)  

**Abstract**: Conversational AI (ConvAI) agents increasingly maintain structured memory to support long-term, task-oriented interactions. In-context memory approaches append the growing history to the model input, which scales poorly under context-window limits. RAG-based methods retrieve request-relevant information, but most assume flat memory collections and ignore structure. We propose Semantic XPath, a tree-structured memory module to access and update structured conversational memory. Semantic XPath improves performance over flat-RAG baselines by 176.7% while using only 9.1% of the tokens required by in-context memory. We also introduce SemanticXPath Chat, an end-to-end ConvAI demo system that visualizes the structured memory and query execution details. Overall, this paper demonstrates a candidate for the next generation of long-term, task-oriented ConvAI systems built on structured memory. 

---
# DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent 

**Authors**: Tongzhou Wu, Yuhao Wang, Xinyu Ma, Xiuqiang He, Shuaiqiang Wang, Dawei Yin, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01152)  

**Abstract**: Deep-research agents are capable of executing multi-step web exploration, targeted retrieval, and sophisticated question answering. Despite their powerful capabilities, deep-research agents face two critical bottlenecks: (1) the lack of large-scale, challenging datasets with real-world difficulty, and (2) the absence of accessible, open-source frameworks for data synthesis and agent training. To bridge these gaps, we first construct DeepResearch-9K, a large-scale challenging dataset specifically designed for deep-research scenarios built from open-source multi-hop question-answering (QA) datasets via a low-cost autonomous pipeline. Notably, it consists of (1) 9000 questions spanning three difficulty levels from L1 to L3 (2) high-quality search trajectories with reasoning chains from Tongyi-DeepResearch-30B-A3B, a state-of-the-art deep-research agent, and (3) verifiable answers. Furthermore, we develop an open-source training framework DeepResearch-R1 that supports (1) multi-turn web interactions, (2) different reinforcement learning (RL) approaches, and (3) different reward models such as rule-based outcome reward and LLM-as-judge feedback. Finally, empirical results demonstrate that agents trained on DeepResearch-9K under our DeepResearch-R1 achieve state-of-the-art results on challenging deep-research benchmarks. We release the DeepResearch-9K dataset on this https URL and the code of DeepResearch-R1 on this https URL. 

---
# AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution 

**Authors**: Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, Bo Zhang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2603.01145)  

**Abstract**: In practical LLM applications, users repeatedly express stable preferences and requirements, such as reducing hallucinations, following institutional writing conventions, or avoiding overly technical wording, yet such interaction experience is seldom consolidated into reusable knowledge. Consequently, LLM agents often fail to accumulate personalized capabilities across sessions. We present AutoSkill, an experience-driven lifelong learning framework that enables LLM agents to automatically derive, maintain, and reuse skills from dialogue and interaction traces. AutoSkill abstracts skills from user experience, supports their continual self-evolution, and dynamically injects relevant skills into future requests without retraining the underlying model. Designed as a model-agnostic plugin layer, it is compatible with existing LLMs and introduces a standardized skill representation for sharing and transfer across agents, users, and tasks. In this way, AutoSkill turns ephemeral interaction experience into explicit, reusable, and composable capabilities. This paper describes the motivation, architecture, skill lifecycle, and implementation of AutoSkill, and positions it with respect to prior work on memory, retrieval, personalization, and agentic systems. AutoSkill highlights a practical and scalable path toward lifelong personalized agents and personal digital surrogates. 

---
# FCN-LLM: Empower LLM for Brain Functional Connectivity Network Understanding via Graph-level Multi-task Instruction Tuning 

**Authors**: Xingcan Hu, Wei Wang, Li Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01135)  

**Abstract**: Large Language Models have achieved remarkable success in language understanding and reasoning, and their multimodal extensions enable comprehension of images, video, and audio. Inspired by this, foundation models for brain functional connectivity networks derived from resting-state fMRI have shown promise in clinical tasks. However, existing methods do not align FCNs with the text modality, limiting the ability of LLMs to directly understand FCNs. To address this, we propose FCN-LLM, a framework that enables LLMs to understand FCNs through graph-level, multi-task instruction tuning. Our approach employs a multi-scale FCN encoder capturing brain-region, functional subnetwork, and whole-brain features, projecting them into the semantic space of LLM. We design multi-paradigm instruction tasks covering 19 subject-specific attributes across demographics, phenotypes, and psychiatric conditions. A multi-stage learning strategy first aligns FCN embeddings with the LLM and then jointly fine-tunes the entire model to capture high-level semantic information. Experiments on a large-scale, multi-site FCN database show that FCN-LLM achieves strong zero-shot generalization on unseen datasets, outperforming conventional supervised and foundation models. This work introduces a new paradigm for integrating brain functional networks with LLMs, offering a flexible and interpretable framework for neuroscience. 

---
# HVR-Met: A Hypothesis-Verification-Replaning Agentic System for Extreme Weather Diagnosis 

**Authors**: Shuo Tang, Jiadong Zhang, Jian Xu, Gengxian Zhou, Qizhao Jin, Qinxuan Wang, Yi Hu, Ning Hu, Hongchang Ren, Lingli He, Jiaolan Fu, Jingtao Ding, Shiming Xiang, Chenglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01121)  

**Abstract**: While deep learning-based weather forecasting paradigms have made significant strides, addressing extreme weather diagnostics remains a formidable challenge. This gap exists primarily because the diagnostic process demands sophisticated multi-step logical reasoning, dynamic tool invocation, and expert-level prior judgment. Although agents possess inherent advantages in task decomposition and autonomous execution, current architectures are still hampered by critical bottlenecks: inadequate expert knowledge integration, a lack of professional-grade iterative reasoning loops, and the absence of fine-grained validation and evaluation systems for complex workflows under extreme conditions. To this end, we propose HVR-Met, a multi-agent meteorological diagnostic system characterized by the deep integration of expert knowledge. Its central innovation is the ``Hypothesis-Verification-Replanning'' closed-loop mechanism, which facilitates sophisticated iterative reasoning for anomalous meteorological signals during extreme weather events. To bridge gaps within existing evaluation frameworks, we further introduce a novel benchmark focused on atomic-level subtasks. Experimental evidence demonstrates that the system excels in complex diagnostic scenarios. 

---
# DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage 

**Authors**: Haowen Gao, Zhenyu Zhang, Liang Pang, Fangda Guo, Hongjian Dou, Guannan Lv, Shaoguo Liu, Tingting Gao, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.01106)  

**Abstract**: Reinforcement learning (RL) with group relative policy optimization (GRPO) has become a widely adopted approach for enhancing the reasoning capabilities of multimodal large language models (MLLMs). While GRPO enables long-chain reasoning without a critic, it often suffers from sparse rewards on difficult problems and advantage vanishing when group-level rewards are too consistent for overly easy or hard problems. Existing solutions (sample expansion, selective utilization, and indirect reward design) often fail to maintain enough variance in within-group reward distributions to yield clear optimization signals. To address this, we propose DIVA-GRPO, a difficulty-adaptive variant advantage method that adjusts variant difficulty distributions from a global perspective. DIVA-GRPO dynamically assesses problem difficulty, samples variants with appropriate difficulty levels, and calculates advantages across local and global groups using difficulty-weighted and normalized scaling. This alleviates reward sparsity and advantage vanishing while improving training stability. Extensive experiments on six reasoning benchmarks demonstrate that DIVA-GRPO outperforms existing approaches in training efficiency and reasoning performance. Code: this https URL 

---
# Alien Science: Sampling Coherent but Cognitively Unavailable Research Directions from Idea Atoms 

**Authors**: Alejandro H. Artiles, Martin Weiss, Levin Brinkmann, Anirudh Goyal, Nasim Rahaman  

**Link**: [PDF](https://arxiv.org/pdf/2603.01092)  

**Abstract**: Large language models are adept at synthesizing and recombining familiar material, yet they often fail at a specific kind of creativity that matters most in research: producing ideas that are both coherent and non-obvious to the current community. We formalize this gap through cognitive availability, the likelihood that a research direction would be naturally proposed by a typical researcher given what they have worked on. We introduce a pipeline that (i) decomposes papers into granular conceptual units, (ii) clusters recurring units into a shared vocabulary of idea atoms, and (iii) learns two complementary models: a coherence model that scores whether a set of atoms constitutes a viable direction, and an availability model that scores how likely that direction is to be generated by researchers drawn from the community. We then sample "alien" directions that score high on coherence but low on availability. On a corpus of $\sim$7,500 recent LLM papers from NeurIPS, ICLR and ICML, we validate that (a) conceptual units preserve paper content under reconstruction, (b) idea atoms generalize across papers rather than memorizing paper-specific phrasing, and (c) the Alien sampler produces research directions that are more diverse than LLM baselines while maintaining coherence. 

---
# MMCOMET: A Large-Scale Multimodal Commonsense Knowledge Graph for Contextual Reasoning 

**Authors**: Eileen Wang, Hiba Arnaout, Dhita Pratama, Shuo Yang, Dangyang Liu, Jie Yang, Josiah Poon, Jeff Pan, Caren Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.01055)  

**Abstract**: We present MMCOMET, the first multimodal commonsense knowledge graph (MMKG) that integrates physical, social, and eventive knowledge. MMCOMET extends the ATOMIC2020 knowledge graph to include a visual dimension, through an efficient image retrieval process, resulting in over 900K multimodal triples. This new resource addresses a major limitation of existing MMKGs in supporting complex reasoning tasks like image captioning and storytelling. Through a standard visual storytelling experiment, we show that our holistic approach enables the generation of richer, coherent, and contextually grounded stories than those produced using text-only knowledge. This resource establishes a new foundation for multimodal commonsense reasoning and narrative generation. 

---
# CollabEval: Enhancing LLM-as-a-Judge via Multi-Agent Collaboration 

**Authors**: Yiyue Qian, Shinan Zhang, Yun Zhou, Haibo Ding, Diego Socolinsky, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00993)  

**Abstract**: Large Language Models (LLMs) have revolutionized AI-generated content evaluation, with the LLM-as-a-Judge paradigm becoming increasingly popular. However, current single-LLM evaluation approaches face significant challenges, including inconsistent judgments and inherent biases from pre-training data. To address these limitations, we propose CollabEval, a novel multi-agent evaluation framework that implements a three-phase Collaborative Evaluation process: initial evaluation, multi-round discussion, and final judgment. Unlike existing approaches that rely on competitive debate or single-model evaluation, CollabEval emphasizes collaboration among multiple agents with strategic consensus checking for efficiency. Our extensive experiments demonstrate that CollabEval consistently outperforms single-LLM approaches across multiple dimensions while maintaining robust performance even when individual models struggle. The framework provides comprehensive support for various evaluation criteria while ensuring efficiency through its collaborative design. 

---
# Tracking Capabilities for Safer Agents 

**Authors**: Martin Odersky, Yaoyu Zhao, Yichen Xu, Oliver Bračevac, Cao Nguyen Pham  

**Link**: [PDF](https://arxiv.org/pdf/2603.00991)  

**Abstract**: AI agents that interact with the real world through tool calls pose fundamental safety challenges: agents might leak private information, cause unintended side effects, or be manipulated through prompt injection. To address these challenges, we propose to put the agent in a programming-language-based "safety harness": instead of calling tools directly, agents express their intentions as code in a capability-safe language: Scala 3 with capture checking. Capabilities are program variables that regulate access to effects and resources of interest. Scala's type system tracks capabilities statically, providing fine-grained control over what an agent can do. In particular, it enables local purity, the ability to enforce that sub-computations are side-effect-free, preventing information leakage when agents process classified data. We demonstrate that extensible agent safety harnesses can be built by leveraging a strong type system with tracked capabilities. Our experiments show that agents can generate capability-safe code with no significant loss in task performance, while the type system reliably prevents unsafe behaviors such as information leakage and malicious side effects. 

---
# HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents 

**Authors**: Hongbo Jin, Rongpeng Zhu, Jiayu Ding, Wenhao Zhang, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.00977)  

**Abstract**: Large language model (LLM) agents have recently demonstrated strong capabilities in interactive decision-making, yet they remain fundamentally limited in long-horizon tasks that require structured planning and reliable execution. Existing approaches predominantly rely on flat autoregressive policies, where high-level reasoning and low-level actions are generated within a single token sequence, leading to inefficient exploration and severe error propagation over extended trajectories. In this work, we propose HiMAC, a hierarchical agentic RL framework that explicitly decomposes long-horizon decision-making into macro-level planning and micro-level execution. HiMAC models reasoning as a structured blueprint generation process followed by goal-conditioned action execution, enabling robust long-horizon planning within LLM-based agents. To train this hierarchy efficiently, we introduce a critic-free hierarchical policy optimization paradigm that extends group-based reinforcement learning to bi-level structures through hierarchical relative advantage estimation. Furthermore, we propose an iterative co-evolution training strategy that alternates between planner exploration and executor adaptation, mitigating the non-stationarity inherent in hierarchical learning. Extensive experiments on ALFWorld, WebShop, and Sokoban demonstrate that HiMAC consistently outperforms strong prompting and reinforcement learning baselines, achieving state-of-the-art performance and substantially improved sample efficiency across both text-based and visually grounded environments. Our results show that introducing structured hierarchy, rather than increasing model scale alone, is a key factor for enabling robust long-horizon agentic intelligence. 

---
# BioProAgent: Neuro-Symbolic Grounding for Constrained Scientific Planning 

**Authors**: Yuyang Liu, Jingya Wang, Liuzhenghao Lv, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.00876)  

**Abstract**: Large language models (LLMs) have demonstrated significant reasoning capabilities in scientific discovery but struggle to bridge the gap to physical execution in wet-labs. In these irreversible environments, probabilistic hallucinations are not merely incorrect, but also cause equipment damage or experimental failure. To address this, we propose \textbf{BioProAgent}, a neuro-symbolic framework that anchors probabilistic planning in a deterministic Finite State Machine (FSM). We introduce a State-Augmented Planning mechanism that enforces a rigorous \textit{Design-Verify-Rectify} workflow, ensuring hardware compliance before execution. Furthermore, we address the context bottleneck inherent in complex device schemas by \textit{Semantic Symbol Grounding}, reducing token consumption by $\sim$6$\times$ through symbolic abstraction. In the extended BioProBench benchmark, BioProAgent achieves 95.6\% physical compliance (compared to 21.0\% for ReAct), demonstrating that neuro-symbolic constraints are essential for reliable autonomy in irreversible physical environments. \footnote{Code at this https URL and project at this https URL} 

---
# MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains 

**Authors**: Xuying Ning, Dongqi Fu, Tianxin Wei, Mengting Ai, Jiaru Zou, Ting-Wei Li, Hanghang Tong, Yada Zhu, Hendrik Hamann, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2603.00873)  

**Abstract**: With the increasing demand for step-wise, cross-modal, and knowledge-grounded reasoning, multimodal large language models (MLLMs) are evolving beyond the traditional fixed retrieve-then-generate paradigm toward more sophisticated agentic multimodal retrieval-augmented generation (MM-RAG). Existing benchmarks, however, mainly focus on simplified QA with short retrieval chains, leaving adaptive planning and multimodal reasoning underexplored. We present MC-Search, the first benchmark for agentic MM-RAG with long, step-wise annotated reasoning chains spanning five representative reasoning structures. Each example specifies sub-questions, retrieval modalities, supporting facts, and intermediate answers, with fidelity ensured by HAVE (Hop-wise Attribution and Verification of Evidence), resulting in 3,333 high-quality examples averaging 3.7 hops. Beyond answer accuracy, MC-Search introduces new process-level metrics for reasoning quality, stepwise retrieval and planning accuracy. By developing a unified agentic MM-RAG pipeline, we benchmark six leading MLLMs and reveal systematic issues such as over- and under-retrieval and modality-misaligned planning. Finally, we introduce Search-Align, a process-supervised fine-tuning framework leveraging verified reasoning chains, showing that our data not only enables faithful evaluation but also improves planning and retrieval fidelity in open-source MLLMs. 

---
# MetaMind: General and Cognitive World Models in Multi-Agent Systems by Meta-Theory of Mind 

**Authors**: Lingyi Wang, Rashed Shelim, Walid Saad, Naren Ramakrishna  

**Link**: [PDF](https://arxiv.org/pdf/2603.00808)  

**Abstract**: A major challenge for world models in multi-agent systems is to understand interdependent agent dynamics, predict interactive multi-agent trajectories, and plan over long horizons with collective awareness, without centralized supervision or explicit communication. In this paper, MetaMind, a general and cognitive world model for multi-agent systems that leverages a novel meta-theory of mind (Meta-ToM) framework, is proposed. Through MetaMind, each agent learns not only to predict and plan over its own beliefs, but also to inversely reason goals and beliefs from its own behavior trajectories. This self-reflective, bidirectional inference loop enables each agent to learn a metacognitive ability in a self-supervised manner. Then, MetaMind is shown to generalize the metacognitive ability from first-person to third-person through analogical reasoning. Thus, in multi-agent systems, each agent with MetaMind can actively reason about goals and beliefs of other agents from limited, observable behavior trajectories in a zero-shot manner, and then adapt to emergent collective intention without an explicit communication mechanism. Extended simulation results on diverse multi-agent tasks demonstrate that MetaMind can achieve superior task performance and outperform baselines in few-shot multi-agent generalization. 

---
# The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents 

**Authors**: Shrey Shah, Levent Ozgur  

**Link**: [PDF](https://arxiv.org/pdf/2603.00801)  

**Abstract**: Language agents increasingly act as web-enabled systems that search, browse, and synthesize information from diverse sources. However, these sources can include unreliable or adversarial content, and the robustness of agents to adversarial ranking - where misleading information appears prominently in search results - remains poorly understood. Existing benchmarks evaluate functional navigation or static factuality but cannot causally isolate this vulnerability, and current mitigation strategies for retrieval-augmented generation remain largely untested under such conditions. We introduce Synthetic Web Benchmark, a procedurally generated environment comprising thousands of hyperlinked articles with ground-truth labels for credibility and factuality, process-level interaction traces, and contamination filtering to eliminate training-data leakage. By injecting a single high-plausibility misinformation article into a controllable search rank, we measure the causal effect of adversarial exposure in six frontier models. The results reveal catastrophic failures: accuracy collapses despite unlimited access to truthful sources, with minimal search escalation and severe miscalibration. These findings expose fundamental limitations in how current frontier models handle conflicting information, with immediate implications for deployment in high-stakes domains. Our benchmark enables systematic analysis of these failure modes and provides a controlled testbed for evaluating mitigation strategies under adversarial ranking - a gap in current research. This work establishes a reproducible baseline for developing search-robust and epistemically humble agents capable of resisting manipulation in high-stakes domains. 

---
# MO-MIX: Multi-Objective Multi-Agent Cooperative Decision-Making With Deep Reinforcement Learning 

**Authors**: Tianmeng Hu, Biao Luo, Chunhua Yang, Tingwen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00730)  

**Abstract**: Deep reinforcement learning (RL) has been applied extensively to solve complex decision-making problems. In many real-world scenarios, tasks often have several conflicting objectives and may require multiple agents to cooperate, which are the multi-objective multi-agent decision-making problems. However, only few works have been conducted on this intersection. Existing approaches are limited to separate fields and can only handle multi-agent decision-making with a single objective, or multi-objective decision-making with a single agent. In this paper, we propose MO-MIX to solve the multi-objective multi-agent reinforcement learning (MOMARL) problem. Our approach is based on the centralized training with decentralized execution (CTDE) framework. A weight vector representing preference over the objectives is fed into the decentralized agent network as a condition for local action-value function estimation, while a mixing network with parallel architecture is used to estimate the joint action-value function. In addition, an exploration guide approach is applied to improve the uniformity of the final non-dominated solutions. Experiments demonstrate that the proposed method can effectively solve the multi-objective multi-agent cooperative decision-making problem and generate an approximation of the Pareto set. Our approach not only significantly outperforms the baseline method in all four kinds of evaluation metrics, but also requires less computational cost. 

---
# AIoT-based Continuous, Contextualized, and Explainable Driving Assessment for Older Adults 

**Authors**: Yimeng Liu, Fangwei Zhang, Maolin Gan, Jialuo Du, Jingkai Lin, Yawen Wang, Fei Sun, Honglei Chen, Linda Hill, Ruofeng Liu, Tianxing Li, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00691)  

**Abstract**: The world is undergoing a major demographic shift as older adults become a rapidly growing share of the population, creating new challenges for driving safety. In car-dependent regions such as the United States, driving remains essential for independence, access to services, and social participation. At the same time, aging can introduce gradual changes in vision, attention, reaction time, and driving control that quietly reduce safety. Today's assessment methods rely largely on infrequent clinic visits or simple screening tools, offering only a brief snapshot and failing to reflect how an older adult actually drives on the road. Our work starts from the observation that everyday driving provides a continuous record of functional ability and captures how a driver responds to traffic, navigates complex roads, and manages routine behavior. Leveraging this insight, we propose AURA, an Artificial Intelligence of Things (AIoT) framework for continuous, real-world assessment of driving safety among older adults. AURA integrates richer in-vehicle sensing, multi-scale behavioral modeling, and context-aware analysis to extract detailed indicators of driving performance from routine trips. It organizes fine-grained actions into longer behavioral trajectories and separates age-related performance changes from situational factors such as traffic, road design, or weather. By integrating sensing, modeling, and interpretation within a privacy-preserving edge architecture, AURA provides a foundation for proactive, individualized support that helps older adults drive safely. This paper outlines the design principles, challenges, and research opportunities needed to build reliable, real-world monitoring systems that promote safer aging behind the wheel. 

---
# MemPO: Self-Memory Policy Optimization for Long-Horizon Agents 

**Authors**: Ruoran Li, Xinghua Zhang, Haiyang Yu, Shitong Duan, Xiang Li, Wenxin Xiang, Chonghua Liao, Xudong Guo, Yongbin Li, Jinli Suo  

**Link**: [PDF](https://arxiv.org/pdf/2603.00680)  

**Abstract**: Long-horizon agents face the challenge of growing context size during interaction with environment, which degrades the performance and stability. Existing methods typically introduce the external memory module and look up the relevant information from the stored memory, which prevents the model itself from proactively managing its memory content and aligning with the agent's overarching task objectives. To address these limitations, we propose the self-memory policy optimization algorithm (MemPO), which enables the agent (policy model) to autonomously summarize and manage their memory during interaction with environment. By improving the credit assignment mechanism based on memory effectiveness, the policy model can selectively retain crucial information, significantly reducing token consumption while preserving task performance. Extensive experiments and analyses confirm that MemPO achieves absolute F1 score gains of 25.98% over the base model and 7.1% over the previous SOTA baseline, while reducing token usage by 67.58% and 73.12%. 

---
# K^2-Agent: Co-Evolving Know-What and Know-How for Hierarchical Mobile Device Control 

**Authors**: Zhe Wu, Donglin Mo, Hongjin Lu, Junliang Xing, Jianheng Liu, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.00676)  

**Abstract**: Existing mobile device control agents often perform poorly when solving complex tasks requiring long-horizon planning and precise operations, typically due to a lack of relevant task experience or unfamiliarity with skill execution. We propose K2-Agent, a hierarchical framework that models human-like cognition by separating and co-evolving declarative (knowing what) and procedural (knowing how) knowledge for planning and execution. K2-Agent's high level reasoner is bootstrapped from a single demonstration per task and runs a Summarize-Reflect-Locate-Revise (SRLR) loop to distill and iteratively refine task-level declarative knowledge through self-evolution. The low-level executor is trained with our curriculum-guided Group Relative Policy Optimization (C-GRPO), which (i) constructs a balanced sample pool using decoupled reward signals and (ii) employs dynamic demonstration injection to guide the model in autonomously generating successful trajectories for training. On the challenging AndroidWorld benchmark, K2-Agent achieves a 76.1% success rate using only raw screenshots and open-source backbones. Furthermore, K2-Agent shows powerful dual generalization: its high-level declarative knowledge transfers across diverse base models, while its low-level procedural skills achieve competitive performance on unseen tasks in ScreenSpot-v2 and Android-in-the-Wild (AitW). 

---
# InfoPO: Information-Driven Policy Optimization for User-Centric Agents 

**Authors**: Fanqi Kong, Jiayi Zhang, Mingyi Deng, Chenglin Wu, Yuyu Luo, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00656)  

**Abstract**: Real-world user requests to LLM agents are often underspecified. Agents must interact to acquire missing information and make correct downstream decisions. However, current multi-turn GRPO-based methods often rely on trajectory-level reward computation, which leads to credit assignment problems and insufficient advantage signals within rollout groups. A feasible approach is to identify valuable interaction turns at a fine granularity to drive more targeted learning. To address this, we introduce InfoPO (Information-Driven Policy Optimization), which frames multi-turn interaction as a process of active uncertainty reduction and computes an information-gain reward that credits turns whose feedback measurably changes the agent's subsequent action distribution compared to a masked-feedback counterfactual. It then combines this signal with task outcomes via an adaptive variance-gated fusion to identify information importance while maintaining task-oriented goal direction. Across diverse tasks, including intent clarification, collaborative coding, and tool-augmented decision making, InfoPO consistently outperforms prompting and multi-turn RL baselines. It also demonstrates robustness under user simulator shifts and generalizes effectively to environment-interactive tasks. Overall, InfoPO provides a principled and scalable mechanism for optimizing complex agent-user collaboration. Code is available at this https URL. 

---
# LiTS: A Modular Framework for LLM Tree Search 

**Authors**: Xinzhe Li, Yaguang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00631)  

**Abstract**: LiTS is a modular Python framework for LLM reasoning via tree search. It decomposes tree search into three reusable components (Policy, Transition, and RewardModel) that plug into algorithms like MCTS and BFS. A decorator-based registry enables domain experts to extend to new domains by registering components, and algorithmic researchers to implement custom search algorithms. We demonstrate composability on MATH500 (language reasoning), Crosswords (environment planning), and MapEval (tool use), showing that components and algorithms are orthogonal: components are reusable across algorithms within each task type, and algorithms work across all components and domains. We also report a mode-collapse finding: in infinite action spaces, LLM policy diversity (not reward quality) is the bottleneck for effective tree search. A demonstration video is available at this https URL. The package is released under the Apache 2.0 license at this https URL, including installation instructions and runnable examples that enable users to reproduce the demonstrated workflows. 

---
# TraceSIR: A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces 

**Authors**: Shu-Xun Yang, Cunxiang Wang, Haoke Zhang, Wenbo Yu, Lindong Wu, Jiayi Gui, Dayong Yang, Yukuo Cen, Zhuoer Feng, Bosi Wen, Yidong Wang, Lucen Zhong, Jiamin Ren, Linfeng Zhang, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00623)  

**Abstract**: Agentic systems augment large language models with external tools and iterative decision making, enabling complex tasks such as deep research, function calling, and coding. However, their long and intricate execution traces make failure diagnosis and root cause analysis extremely challenging. Manual inspection does not scale, while directly applying LLMs to raw traces is hindered by input length limits and unreliable reasoning. Focusing solely on final task outcomes further discards critical behavioral information required for accurate issue localization. To address these issues, we propose TraceSIR, a multi-agent framework for structured analysis and reporting of agentic execution traces. TraceSIR coordinates three specialized agents: (1) StructureAgent, which introduces a novel abstraction format, TraceFormat, to compress execution traces while preserving essential behavioral information; (2) InsightAgent, which performs fine-grained diagnosis including issue localization, root cause analysis, and optimization suggestions; (3) ReportAgent, which aggregates insights across task instances and generates comprehensive analysis reports. To evaluate TraceSIR, we construct TraceBench, covering three real-world agentic scenarios, and introduce ReportEval, an evaluation protocol for assessing the quality and usability of analysis reports aligned with industry needs. Experiments show that TraceSIR consistently produces coherent, informative, and actionable reports, significantly outperforming existing approaches across all evaluation dimensions. Our project and video are publicly available at this https URL. 

---
# Machine Learning Grade Prediction Using Students' Grades and Demographics 

**Authors**: Mwayi Sonkhanani, Symon Chibaya, Clement N. Nyirenda  

**Link**: [PDF](https://arxiv.org/pdf/2603.00608)  

**Abstract**: Student repetition in secondary education imposes significant resource burdens, particularly in resource-constrained contexts. Addressing this challenge, this study introduces a unified machine learning framework that simultaneously predicts pass/fail outcomes and continuous grades, a departure from prior research that treats classification and regression as separate tasks. Six models were evaluated: Logistic Regression, Decision Tree, and Random Forest for classification, and Linear Regression, Decision Tree Regressor, and Random Forest Regressor for regression, with hyperparameters optimized via exhaustive grid search. Using academic and demographic data from 4424 secondary school students, classification models achieved accuracies of up to 96%, while regression models attained a coefficient of determination of 0.70, surpassing baseline approaches. These results confirm the feasibility of early, data-driven identification of at-risk students and highlight the value of integrating dual-task prediction for more comprehensive insights. By enabling timely, personalized interventions, the framework offers a practical pathway to reducing grade repetition and optimizing resource allocation. 

---
# Heterophily-Agnostic Hypergraph Neural Networks with Riemannian Local Exchanger 

**Authors**: Li Sun, Ming Zhang, Wenxin Jin, Zhongtian Sun, Zhenhao Huang, Hao Peng, Sen Su, Philip Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00599)  

**Abstract**: Hypergraphs are the natural description of higher-order interactions among objects, widely applied in social network analysis, cross-modal retrieval, etc. Hypergraph Neural Networks (HGNNs) have become the dominant solution for learning on hypergraphs. Traditional HGNNs are extended from message passing graph neural networks, following the homophily assumption, and thus struggle with the prevalent heterophilic hypergraphs that call for long-range dependence modeling. In this paper, we achieve heterophily-agnostic message passing through the lens of Riemannian geometry. The key insight lies in the connection between oversquashing and hypergraph bottleneck within the framework of Riemannian manifold heat flow. Building on this, we propose the novel idea of locally adapting the bottlenecks of different subhypergraphs. The core innovation of the proposed mechanism is the design of an adaptive local (heat) exchanger. Specifically, it captures the rich long-range dependencies via the Robin condition, and preserves the representation distinguishability via source terms, thereby enabling heterophily-agnostic message passing with theoretical guarantees. Based on this theoretical foundation, we present a novel Heat-Exchanger with Adaptive Locality for Hypergraph Neural Network (HealHGNN), designed as a node-hyperedge bidirectional systems with linear complexity in the number of nodes and hyperedges. Extensive experiments on both homophilic and heterophilic cases show that HealHGNN achieves the state-of-the-art performance. 

---
# Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs 

**Authors**: Yiran Zhao, Lu Zhou, Xiaogang Xu, Zhe Liu, Jiafei Wu, Liming Fang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00590)  

**Abstract**: As artificial intelligence (AI) is increasingly deployed across domains, ensuring fairness has become a core challenge. However, the field faces a "Tower of Babel'' dilemma: fairness metrics abound, yet their underlying philosophical assumptions often conflict, hindering unified paradigms-particularly in unified Multimodal Large Language Models (UMLLMs), where biases propagate systemically across tasks. To address this, we introduce the IRIS Benchmark, to our knowledge the first benchmark designed to synchronously evaluate the fairness of both understanding and generation tasks in UMLLMs. Enabled by our demographic classifier, ARES, and four supporting large-scale datasets, the benchmark is designed to normalize and aggregate arbitrary metrics into a high-dimensional "fairness space'', integrating 60 granular metrics across three dimensions-Ideal Fairness, Real-world Fidelity, and Bias Inertia & Steerability (IRIS). Through this benchmark, our evaluation of leading UMLLMs uncovers systemic phenomena such as the "generation gap'', individual inconsistencies like "personality splits'', and the "counter-stereotype reward'', while offering diagnostics to guide the optimization of their fairness capabilities. With its novel and extensible framework, the IRIS benchmark is capable of integrating evolving fairness metrics, ultimately helping to resolve the "Tower of Babel'' impasse. Project Page: this https URL 

---
# MicroVerse: A Preliminary Exploration Toward a Micro-World Simulation 

**Authors**: Rongsheng Wang, Minghao Wu, Hongru Zhou, Zhihan Yu, Zhenyang Cai, Junying Chen, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00585)  

**Abstract**: Recent advances in video generation have opened new avenues for macroscopic simulation of complex dynamic systems, but their application to microscopic phenomena remains largely unexplored. Microscale simulation holds great promise for biomedical applications such as drug discovery, organ-on-chip systems, and disease mechanism studies, while also showing potential in education and interactive visualization. In this work, we introduce MicroWorldBench, a multi-level rubric-based benchmark for microscale simulation tasks. MicroWorldBench enables systematic, rubric-based evaluation through 459 unique expert-annotated criteria spanning multiple microscale simulation task (e.g., organ-level processes, cellular dynamics, and subcellular molecular interactions) and evaluation dimensions (e.g., scientific fidelity, visual quality, instruction following). MicroWorldBench reveals that current SOTA video generation models fail in microscale simulation, showing violations of physical laws, temporal inconsistency, and misalignment with expert criteria. To address these limitations, we construct MicroSim-10K, a high-quality, expert-verified simulation dataset. Leveraging this dataset, we train MicroVerse, a video generation model tailored for microscale simulation. MicroVerse can accurately reproduce complex microscale mechanism. Our work first introduce the concept of Micro-World Simulation and present a proof of concept, paving the way for applications in biology, education, and scientific visualization. Our work demonstrates the potential of educational microscale simulations of biological mechanisms. Our data and code are publicly available at this https URL 

---
# Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs 

**Authors**: Jie Cao, Tianwei Lin, Zhenxuan Fan, Bo Yuan, Ziyuan Zhao, Rolan Yan, Wenqiao Zhang, Siliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00578)  

**Abstract**: Long chain-of-thought~(CoT) has become a dominant paradigm for enhancing the reasoning capability of large reasoning models~(LRMs); however, the performance gains often come with a substantial increase in reasoning budget. Recent studies show that existing CoT paradigms tend to induce systematic overthinking, unnecessarily coupling reasoning capability with reasoning cost. Most prior approaches reduce token usage through post hoc techniques such as token compression, truncation, or length penalties, without explicitly addressing the core mechanisms of reasoning. We propose \textbf{Draft-Thinking}, which guides models to first learn a concise \textit{draft-style} reasoning structure that retains only the critical reasoning steps. Through a \textit{progressive curriculum learning}, the model stably internalizes this efficient reasoning pattern as its capability scales. Moreover, Draft-Thinking introduces adaptive prompting, which elevates reasoning depth to a flexible, model-selectable behavior. Extensive experiments demonstrate that Draft-Thinking substantially reduces reasoning budget while largely preserving reasoning performance; for example, on MATH500, it achieves an 82.6\% reduction in reasoning budget at the cost of only a 2.6\% performance drop. 

---
# SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks 

**Authors**: Yucheng Zeng, Shupeng Li, Daxiang Dong, Ruijie Xu, Zimo Chen, Liwei Zheng, Yuxuan Li, Zhe Zhou, Haotian Zhao, Lun Tian, Heng Xiao, Tianshu Zhu, Longkun Hao, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00575)  

**Abstract**: Progress in software-engineering agents is increasingly constrained by the scarcity of executable, scalable, and realistic data for training and evaluation. This scarcity stems from three fundamental challenges in existing pipelines: environments are brittle and difficult to reproduce across languages; synthesizing realistic, system-level bugs at scale is computationally expensive; and existing data predominantly consists of short-horizon repairs, failing to capture long-horizon competencies like architectural consistency. We introduce \textbf{SWE-Hub}, an end-to-end system that operationalizes the data factory abstraction by unifying environment automation, scalable synthesis, and diverse task generation into a coherent production stack. At its foundation, the \textbf{Env Agent} establishes a shared execution substrate by automatically converting raw repository snapshots into reproducible, multi-language container environments with standardized interfaces. Built upon this substrate, \textbf{SWE-Scale} engine addresses the need for high-throughput generation, combining cross-language code analysis with cluster-scale validation to synthesize massive volumes of localized bug-fix instances. \textbf{Bug Agent} generates high-fidelity repair tasks by synthesizing system-level regressions involving cross-module dependencies, paired with user-like issue reports that describe observable symptoms rather than root causes. Finally, \textbf{SWE-Architect} expands the task scope from repair to creation by translating natural-language requirements into repository-scale build-a-repo tasks. By integrating these components, SWE-Hub establishes a unified production pipeline capable of continuously delivering executable tasks across the entire software engineering lifecycle. 

---
# EMPA: Evaluating Persona-Aligned Empathy as a Process 

**Authors**: Shiya Zhang, Yuhan Zhan, Ruixi Su, Ruihan Sun, Ziyi Song, Zhaohan Chen, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00552)  

**Abstract**: Evaluating persona-aligned empathy in LLM-based dialogue agents remains challenging. User states are latent, feedback is sparse and difficult to verify in situ, and seemingly supportive turns can still accumulate into trajectories that drift from persona-specific needs. We introduce EMPA, a process-oriented framework that evaluates persona-aligned support as sustained intervention rather than isolated replies. EMPA distills real interactions into controllable, psychologically grounded scenarios, couples them with an open-ended multi-agent sandbox that exposes strategic adaptation and failure modes, and scores trajectories in a latent psychological space by directional alignment, cumulative impact, and stability. The resulting signals and metrics support reproducible comparison and optimization of long-horizon empathic behavior, and they extend to other agent settings shaped by latent dynamics and weak, hard-to-verify feedback. 

---
# Advancing Multimodal Judge Models through a Capability-Oriented Benchmark and MCTS-Driven Data Generation 

**Authors**: Zeyu Chen, Huanjin Yao, Ziwang Zhao, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00546)  

**Abstract**: Using Multimodal Large Language Models (MLLMs) as judges to achieve precise and consistent evaluations has gradually become an emerging paradigm across various domains. Evaluating the capability and reliability of MLLM-as-a-judge systems is therefore essential for ensuring trustworthy assessment. Existing judge benchmarks categorize samples by task types but fail to capture the fundamental judgment capabilities required for reliable evaluation. In this work, we introduce M-JudgeBench, a ten-dimensional capability-oriented benchmark designed to comprehensively assess the judgment abilities of MLLMs. Our benchmark decomposes evaluation into pairwise Chain-of-Thought (CoT) comparison, length bias avoidance, and process error detection tasks, jointly covering ten fine-grained subtasks. This design enables diagnosis of model reliability across reasoning styles, response lengths, and cross-model variations. Systematic evaluation uncovers the systematic weaknesses in existing MLLM-as-a-judge systems. To address this issue, we further propose Judge-MCTS, a data construction framework generating pairwise reasoning trajectories with various correctness and length. Using Judge-MCTS, we construct an MCTS-augmented dataset and train M-Judger, a series of strong judge models. Extensive experiments demonstrate the superiority of M-Judger on existing judge benchmarks as well as M-JudgeBench. Overall, our work establishes a more principled foundation for evaluating MLLM-as-a-judge through M-JudgeBench and Judge-MCTS framework, paving the way for future research on judge model evaluation and capability-driven judge training. 

---
# LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks 

**Authors**: Yucheng Zeng, Weipeng Lu, Linyun Liu, Shupeng Li, Zitian Qu, Chenghao Zhu, Shaofei Li, Zhengdong Tan, Mengyue Liu, Haotian Zhao, Zhe Zhou, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00540)  

**Abstract**: The evolution of Large Language Models (LLMs) from static instruction-followers to autonomous agents necessitates operating within complex, stateful environments to achieve precise state-transition objectives. However, this paradigm is bottlenecked by data scarcity, as existing tool-centric reverse-synthesis pipelines fail to capture the rigorous logic of real-world applications. We introduce \textbf{LOGIGEN}, a logic-driven framework that synthesizes verifiable training data based on three core pillars: \textbf{Hard-Compiled Policy Grounding}, \textbf{Logic-Driven Forward Synthesis}, and \textbf{Deterministic State Verification}. Specifically, a Triple-Agent Orchestration is employed: the \textbf{Architect} compiles natural-language policy into database constraints to enforce hard rules; the \textbf{Set Designer} initializes boundary-adjacent states to trigger critical policy conflicts; and the \textbf{Explorer} searches this environment to discover causal solution paths. This framework yields a dataset of 20,000 complex tasks across 8 domains, where validity is strictly guaranteed by checking exact state equivalence. Furthermore, we propose a verification-based training protocol where Supervised Fine-Tuning (SFT) on verifiable trajectories establishes compliance with hard-compiled policy, while Reinforcement Learning (RL) guided by dense state-rewards refines long-horizon goal achievement. On $\tau^2$-Bench, LOGIGEN-32B(RL) achieves a \textbf{79.5\% success rate}, substantially outperforming the base model (40.7\%). These results demonstrate that logic-driven synthesis combined with verification-based training effectively constructs the causally valid trajectories needed for next-generation agents. 

---
# DenoiseFlow: Uncertainty-Aware Denoising for Reliable LLM Agentic Workflows 

**Authors**: Yandong Yan, Junwei Peng, Shijie Li, Chenxi Li, Yifei Shang, Can Deng, Ruiting Dai, Yongqiang Zhao, Jiaqi Zhu, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00532)  

**Abstract**: Autonomous agents are increasingly entrusted with complex, long-horizon tasks, ranging from mathematical reasoning to software generation. While agentic workflows facilitate these tasks by decomposing them into multi-step reasoning chains, reliability degrades significantly as the sequence lengthens. Specifically, minor interpretation errors in natural-language instructions tend to compound silently across steps. We term this failure mode accumulated semantic ambiguity. Existing approaches to mitigate this often lack runtime adaptivity, relying instead on static exploration budgets, reactive error recovery, or single-path execution that ignores uncertainty entirely. We formalize the multi-step reasoning process as a Noisy MDP and propose DenoiseFlow, a closed-loop framework that performs progressive denoising through three coordinated stages: (1)Sensing estimates per-step semantic uncertainty; (2)Regulating adaptively allocates computation by routing between fast single-path execution and parallel exploration based on estimated risk; and (3)Correcting performs targeted recovery via influence-based root-cause localization. Online self-calibration continuously aligns decision boundaries with verifier feedback, requiring no ground-truth labels. Experiments on six benchmarks spanning mathematical reasoning, code generation, and multi-hop QA show that DenoiseFlow achieves the highest accuracy on every benchmark (83.3% average, +1.3% over the strongest baseline) while reducing cost by 40--56% through adaptive branching. Detailed ablation studies further confirm framework-level's robustness and generality. Code is available at this https URL. 

---
# AI Runtime Infrastructure 

**Authors**: Christopher Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2603.00495)  

**Abstract**: We introduce AI Runtime Infrastructure, a distinct execution-time layer that operates above the model and below the application, actively observing, reasoning over, and intervening in agent behavior to optimize task success, latency, token efficiency, reliability, and safety while the agent is running. Unlike model-level optimizations or passive logging systems, runtime infrastructure treats execution itself as an optimization surface, enabling adaptive memory management, failure detection, recovery, and policy enforcement over long-horizon agent workflows. 

---
# LifeEval: A Multimodal Benchmark for Assistive AI in Egocentric Daily Life Tasks 

**Authors**: Hengjian Gao, Kaiwei Zhang, Shibo Wang, Mingjie Chen, Qihang Cao, Xianfeng Wang, Yucheng Zhu, Xiongkuo Min, Wei Sun, Dandan Zhu, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2603.00490)  

**Abstract**: The rapid progress of Multimodal Large Language Models (MLLMs) marks a significant step toward artificial general intelligence, offering great potential for augmenting human capabilities. However, their ability to provide effective assistance in dynamic, real-world environments remains largely underexplored. Existing video benchmarks predominantly assess passive understanding through retrospective analysis or isolated perception tasks, failing to capture the interactive and adaptive nature of real-time user assistance. To bridge this gap, we introduce LifeEval, a multimodal benchmark designed to evaluate real-time, task-oriented human-AI collaboration in daily life from an egocentric perspective. LifeEval emphasizes three key aspects: task-oriented holistic evaluation, egocentric real-time perception from continuous first-person streams, and human-assistant collaborative interaction through natural dialogues. Constructed via a rigorous annotation pipeline, the benchmark comprises 4,075 high-quality question-answer pairs across 6 core capability dimensions. Extensive evaluations of 26 state-of-the-art MLLMs on LifeEval reveal substantial challenges in achieving timely, effective and adaptive interaction, highlighting essential directions for advancing human-centered interactive intelligence. 

---
# From Goals to Aspects, Revisited: An NFR Pattern Language for Agentic AI Systems 

**Authors**: Yijun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00472)  

**Abstract**: Agentic AI systems exhibit numerous crosscutting concerns -- security, observability, cost management, fault tolerance -- that are poorly modularized in current implementations, contributing to the high failure rate of AI projects in reaching production. The goals-to-aspects methodology proposed at RE 2004 demonstrated that aspects can be systematically discovered from i* goal models by identifying non-functional soft-goals that crosscut functional goals. This paper revisits and extends that methodology to the agentic AI domain. We present a pattern language of 12 reusable patterns organized across four NFR categories (security, reliability, observability, cost management), each mapping an i* goal model to a concrete aspect implementation using an AOP framework for Rust. Four patterns address agent-specific crosscutting concerns absent from traditional AOP literature: tool-scope sandboxing, prompt injection detection, token budget management, and action audit trails. We extend the V-graph model to capture how agent tasks simultaneously contribute to functional goals and non-functional soft-goals. We validate the pattern language through a case study analyzing an open-source autonomous agent framework, demonstrating how goal-driven aspect discovery systematically identifies and modularizes crosscutting concerns. The pattern language offers a principled approach for engineering reliable agentic AI systems through early identification of crosscutting concerns. 

---
# Why Not? Solver-Grounded Certificates for Explainable Mission Planning 

**Authors**: Najeeb Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.00469)  

**Abstract**: Operators of Earth observation satellites need justifications for scheduling decisions: why a request was selected, rejected, or what changes would make it schedulable. Existing approaches construct post-hoc reasoning layers independent of the optimizer, risking non-causal attributions, incomplete constraint conjunctions, and solver-path dependence. We take a faithfulness-first approach: every explanation is a certificate derived from the optimization model itself: minimal infeasible subsets for rejections, tight constraints and contrastive trade-offs for selections, and inverse solves for what-if queries. On a scheduling instance with structurally distinct constraint interactions, certificates achieve perfect soundness with respect to the solver's constraint model (15/15 cited-constraint checks), counterfactual validity (7/7), and stability (Jaccard = 1.0 across 28 seed-pairs), while a post-hoc baseline produces non-causal attributions in 29% of cases and misses constraint conjunctions in every multi-cause rejection. A scalability analysis up to 200 orders and 30 satellites confirms practical extraction times for operational batches. 

---
# Optimizing In-Context Demonstrations for LLM-based Automated Grading 

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Kevin Haudek, Joseph Krajcik, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00465)  

**Abstract**: Automated assessment of open-ended student responses is a critical capability for scaling personalized feedback in education. While large language models (LLMs) have shown promise in grading tasks via in-context learning (ICL), their reliability is heavily dependent on the selection of few-shot exemplars and the construction of high-quality rationales. Standard retrieval methods typically select examples based on semantic similarity, which often fails to capture subtle decision boundaries required for rubric adherence. Furthermore, manually crafting the expert rationales needed to guide these models can be a significant bottleneck. To address these limitations, we introduce GUIDE (Grading Using Iteratively Designed Exemplars), a framework that reframes exemplar selection and refinement in automated grading as a boundary-focused optimization problem. GUIDE operates on a continuous loop of selection and refinement, employing novel contrastive operators to identify "boundary pairs" that are semantically similar but possess different grades. We enhance exemplars by generating discriminative rationales that explicitly articulate why a response receives a specific score to the exclusion of adjacent grades. Extensive experiments across datasets in physics, chemistry, and pedagogical content knowledge demonstrate that GUIDE significantly outperforms standard retrieval baselines. By focusing the model's attention on the precise edges of rubric, our approach shows exceptionally robust gains on borderline cases and improved rubric adherence. GUIDE paves the way for trusted, scalable assessment systems that align closely with human pedagogical standards. 

---
# MED-COPILOT: A Medical Assistant Powered by GraphRAG and Similar Patient Case Retrieval 

**Authors**: Shuheng Chen, Namratha Patil, Haonan Pan, Angel Hsing-Chi Hwang, Yao Du, Ruishan Liu, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00460)  

**Abstract**: Clinical decision-making requires synthesizing heterogeneous evidence, including patient histories, clinical guidelines, and trajectories of comparable cases. While large language models (LLMs) offer strong reasoning capabilities, they remain prone to hallucinations and struggle to integrate long, structured medical documents. We present MED-COPILOT, an interactive clinical decision-support system designed for clinicians and medical trainees, which combines guideline-grounded GraphRAG retrieval with hybrid semantic-keyword similar-patient retrieval to support transparent and evidence-aware clinical reasoning. The system builds a structured knowledge graph from WHO and NICE guidelines, applies community-level summarization for efficient retrieval, and maintains a 36,000-case similar-patient database derived from SOAP-normalized MIMIC-IV notes and Synthea-generated records.
We evaluate our framework on clinical note completion and medical question answering, and demonstrate that it consistently outperforms parametric LLM baselines and standard RAG, improving both generation fidelity and clinical reasoning accuracy. The full system is available at this https URL , enabling users to inspect retrieved evidence, visualize token-level similarity contributions, and conduct guided follow-up analysis. Our results demonstrate a practical and interpretable approach to integrating structured guideline knowledge with patient-level analogical evidence for clinical LLMs. 

---
# Confusion-Aware Rubric Optimization for LLM-based Automated Grading 

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Joseph Krajcik, Namsoo Shin, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00451)  

**Abstract**: Accurate and unambiguous guidelines are critical for large language model (LLM) based graders, yet manually crafting these prompts is often sub-optimal as LLMs can misinterpret expert guidelines or lack necessary domain specificity. Consequently, the field has moved toward automated prompt optimization to refine grading guidelines without the burden of manual trial and error. However, existing frameworks typically aggregate independent and unstructured error samples into a single update step, resulting in "rule dilution" where conflicting constraints weaken the model's grading logic. To address these limitations, we introduce Confusion-Aware Rubric Optimization (CARO), a novel framework that enhances accuracy and computational efficiency by structurally separating error signals. CARO leverages the confusion matrix to decompose monolithic error signals into distinct modes, allowing for the diagnosis and repair of specific misclassification patterns individually. By synthesizing targeted "fixing patches" for dominant error modes and employing a diversity-aware selection mechanism, the framework prevents guidance conflict and eliminates the need for resource-heavy nested refinement loops. Empirical evaluations on teacher education and STEM datasets demonstrate that CARO significantly outperforms existing SOTA methods. These results suggest that replacing mixed-error aggregation with surgical, mode-specific repair yields robust improvements in automated assessment scalability and precision. 

---
# NeuroHex: Highly-Efficient Hex Coordinate System for Creating World Models to Enable Adaptive AI 

**Authors**: Quinn Jacobson, Joe Luo, Jingfei Xu, Shanmuga Venkatachalam, Kevin Wang, Dingchao Rong, John Paul Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00376)  

**Abstract**: \textit{NeuroHex} is a hexagonal coordinate system designed to support highly efficient world models and reference frames for online adaptive AI systems. Inspired by the hexadirectional firing structure of grid cells in the human brain, NeuroHex adopts a cubic isometric hexagonal coordinate formulation that provides full 60° rotational symmetry and low-cost translation, rotation and distance computation. We develop a mathematical framework that incorporates ring indexing, quantized angular encoding, and a hierarchical library of foundational, simple, and complex geometric shape primitives. These constructs allow low-overhead point-in-shape tests and spatial matching operations that are expensive in Cartesian coordinate systems. To support realistic settings, the NeuroHex framework can process OpenStreetMap (OSM) data sets using an OSM-to-NeuroHex (\textit{OSM2Hex}) conversion tool. The OSM2Hex spatial abstraction processing pipeline can achieve a reduction of 90-99\% in geometric complexity while maintaining the relevant spatial structure map for navigation. Our initial results, based on actual city and neighborhood scale data sets, demonstrate that NeuroHex offers a highly efficient substrate for building dynamic world models to enable adaptive spatial reasoning in autonomous AI systems with continuous online learning capability. 

---
# Conservative Equilibrium Discovery in Offline Game-Theoretic Multiagent Reinforcement Learning 

**Authors**: Austin A. Nguyen, Michael P. Wellman  

**Link**: [PDF](https://arxiv.org/pdf/2603.00374)  

**Abstract**: Offline learning of strategies takes data efficiency to its extreme by restricting algorithms to a fixed dataset of state-action trajectories. We consider the problem in a mixed-motive multiagent setting, where the goal is to solve a game under the offline learning constraint. We first frame this problem in terms of selecting among candidate equilibria. Since datasets may inform only a small fraction of game dynamics, it is generally infeasible in offline game-solving to even verify a proposed solution is a true equilibrium. Therefore, we consider the relative probability of low regret (i.e., closeness to equilibrium) across candidates based on the information available. Specifically, we extend Policy Space Response Oracles (PSRO), an online game-solving approach, by quantifying game dynamics uncertainty and modifying the RL objective to skew towards solutions more likely to have low regret in the true game. We further propose a novel meta-strategy solver, tailored for the offline setting, to guide strategy exploration in PSRO. Our incorporation of Conservatism principles from Offline reinforcement learning approaches for strategy Exploration gives our approach its name: COffeE-PSRO. Experiments demonstrate COffeE-PSRO's ability to extract lower-regret solutions than state-of-the-art offline approaches and reveal relationships between algorithmic components empirical game fidelity, and overall performance. 

---
# Monotropic Artificial Intelligence: Toward a Cognitive Taxonomy of Domain-Specialized Language Models 

**Authors**: Antonio de Sousa Leitão Filho, Allan Kardec Duailibe Barros Filho, Fabrício Saul Lima, Selby Mykael Lima dos Santos, Rejani Bandeira Vieira Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2603.00350)  

**Abstract**: The prevailing paradigm in artificial intelligence research equates progress with scale: larger models trained on broader datasets are presumed to yield superior capabilities. This assumption, while empirically productive for general-purpose applications, obscures a fundamental epistemological tension between breadth and depth of knowledge. We introduce the concept of \emph{Monotropic Artificial Intelligence} -- language models that deliberately sacrifice generality to achieve extraordinary precision within narrowly circumscribed domains. Drawing on the cognitive theory of monotropism developed to understand autistic cognition, we argue that intense specialization represents not a limitation but an alternative cognitive architecture with distinct advantages for safety-critical applications. We formalize the defining characteristics of monotropic models, contrast them with conventional polytropic architectures, and demonstrate their viability through Mini-Enedina, a 37.5-million-parameter model that achieves near-perfect performance on Timoshenko beam analysis while remaining deliberately incompetent outside its domain. Our framework challenges the implicit assumption that artificial general intelligence constitutes the sole legitimate aspiration of AI research, proposing instead a cognitive ecology in which specialized and generalist systems coexist complementarily. 

---
# EmCoop: A Framework and Benchmark for Embodied Cooperation Among LLM Agents 

**Authors**: Hanqing Yang, Shiyu Chen, Narjes Nourzad, Marie Siew, Jingdi Chen, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.00349)  

**Abstract**: Real-world scenarios increasingly require multiple embodied agents to collaborate in dynamic environments under embodied constraints, as many tasks exceed the capabilities of any single agent. Recent advances in large language models (LLMs) enable high-level cognitive coordination through reasoning, planning, and natural language communication. However, fine-grained analyses of how such collaboration emerges, unfolds, and contributes to task success in embodied multi-agent systems are difficult to conduct with existing benchmarks. In this paper, we introduce EmCoop, a benchmark framework for studying cooperation in LLM-based embodied multi-agent systems. Our framework separates a high-level cognitive layer from a low-level embodied interaction layer, allowing us to characterize agent cooperation through their interleaved dynamics over time. Given a cooperation-constrained embodied task, we propose generalizable, process-level metrics that diagnose collaboration quality and failure modes, beyond final task success. We instantiate our framework in two embodied environments that scale to arbitrary numbers of agents and support diverse communication topologies, and use these instantiations to demonstrate how EmCoop enables systematic analysis of cooperation dynamics across team sizes and task settings. The project web page can be found at: this https URL. 

---
# How Well Do Multimodal Models Reason on ECG Signals? 

**Authors**: Maxwell A. Xu, Harish Haresumadram, Catherine W. Liu, Patrick Langer, Jathurshan Pradeepkumar, Wanting Mao, Sunita J. Ferns, Aradhana Verma, Jimeng Sun, Paul Schmiedmayer, Xin Liu, Daniel McDuff, Emily B. Fox, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2603.00312)  

**Abstract**: While multimodal large language models offer a promising solution to the "black box" nature of health AI by generating interpretable reasoning traces, verifying the validity of these traces remains a critical challenge. Existing evaluation methods are either unscalable, relying on manual clinician review, or superficial, utilizing proxy metrics (e.g. QA) that fail to capture the semantic correctness of clinical logic. In this work, we introduce a reproducible framework for evaluating reasoning in ECG signals. We propose decomposing reasoning into two distinct, components: (i) Perception, the accurate identification of patterns within the raw signal, and (ii) Deduction, the logical application of domain knowledge to those patterns. To evaluate Perception, we employ an agentic framework that generates code to empirically verify the temporal structures described in the reasoning trace. To evaluate Deduction, we measure the alignment of the model's logic against a structured database of established clinical criteria in a retrieval-based approach. This dual-verification method enables the scalable assessment of "true" reasoning capabilities. 

---
# DIG to Heal: Scaling General-purpose Agent Collaboration via Explainable Dynamic Decision Paths 

**Authors**: Hanqing Yang, Hyungwoo Lee, Yuhang Yao, Zhiwei Liu, Kay Liu, Jingdi Chen, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.00309)  

**Abstract**: The increasingly popular agentic AI paradigm promises to harness the power of multiple, general-purpose large language model (LLM) agents to collaboratively complete complex tasks. While many agentic AI systems utilize predefined workflows or agent roles in order to reduce complexity, ideally these agents would be truly autonomous, able to achieve emergent collaboration even as the number of collaborating agents increases. Yet in practice, such unstructured interactions can lead to redundant work and cascading failures that are difficult to interpret or correct. In this work, we study multi-agent systems composed of general-purpose LLM agents that operate without predefined roles, control flow, or communication constraints, relying instead on emergent collaboration to solve problems. We introduce the Dynamic Interaction Graph (DIG), which captures emergent collaboration as a time-evolving causal network of agent activations and interactions. DIG makes emergent collaboration observable and explainable for the first time, enabling real-time identification, explanation, and correction of collaboration-induced error patterns directly from agents' collaboration paths. Thus, DIG fills a critical gap in understanding how general LLM agents solve problems together in truly agentic multi-agent systems. The project webpage can be found at: this https URL. 

---
# TraderBench: How Robust Are AI Agents in Adversarial Capital Markets? 

**Authors**: Xiaochuang Yuan, Hui Xu, Silvia Xu, Cui Zou, Jing Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.00285)  

**Abstract**: Evaluating AI agents in finance faces two key challenges: static benchmarks require costly expert annotation yet miss the dynamic decision-making central to real-world trading, while LLM-based judges introduce uncontrolled variance on domain-specific tasks. We introduce TraderBench, a benchmark that addresses both issues. It combines expert-verified static tasks (knowledge retrieval, analytical reasoning) with adversarial trading simulations scored purely on realized performance-Sharpe ratio, returns, and drawdown-eliminating judge variance entirely. The framework features two novel tracks: crypto trading with four progressive market-manipulation transforms, and options derivatives scoring across P&L accuracy, Greeks, and risk management. Trading scenarios can be refreshed with new market data to prevent benchmark contamination. Evaluating 13 models (8B open-source to frontier) on ~50 tasks, we find: (1) 8 of 13 models score ~33 on crypto with <1-point variation across adversarial conditions, exposing fixed non-adaptive strategies; (2) extended thinking helps retrieval (+26 points) but has zero impact on trading (+0.3 crypto, -0.1 options). These findings reveal that current agents lack genuine market adaptation, underscoring the need for performance-grounded evaluation in finance. 

---
# Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking 

**Authors**: Shuzhi Gong, Richard O. Sinnott, Jianzhong Qi, Cecile Paris, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.00267)  

**Abstract**: Misinformation spreading over the Internet poses a significant threat to both societies and individuals, necessitating robust and scalable fact-checking that relies on retrieving accurate and trustworthy evidence. Previous methods rely on semantic and social-contextual patterns learned from training data, which limits their generalization to new data distributions. Recently, Retrieval Augmented Generation (RAG) based methods have been proposed to utilize the reasoning capability of LLMs with retrieved grounding evidence documents. However, these methods largely rely on textual similarity for evidence retrieval and struggle to retrieve evidence that captures multi-hop semantic relations within rich document contents. These limitations lead to overlooking subtle factual correlations between the evidence and the claims to be fact-checked during evidence retrieval, thus causing inaccurate veracity predictions.
To address these issues, we propose WKGFC, which exploits authorized open knowledge graph as a core resource of evidence. LLM-enabled retrieval is designed to assess the claims and retrieve the most relevant knowledge subgraphs, forming structured evidence for fact verification. To augment the knowledge graph evidence, we retrieve web contents for completion. The above process is implemented as an automatic Markov Decision Process (MDP): A reasoning LLM agent decides what actions to take according to the current evidence and the claims. To adapt the MDP for fact-checking, we use prompt optimization to fine-tune the agentic LLM. 

---
# Adaptive Confidence Regularization for Multimodal Failure Detection 

**Authors**: Moru Liu, Hao Dong, Olga Fink, Mario Trapp  

**Link**: [PDF](https://arxiv.org/pdf/2603.02200)  

**Abstract**: The deployment of multimodal models in high-stakes domains, such as self-driving vehicles and medical diagnostics, demands not only strong predictive performance but also reliable mechanisms for detecting failures. In this work, we address the largely unexplored problem of failure detection in multimodal contexts. We propose Adaptive Confidence Regularization (ACR), a novel framework specifically designed to detect multimodal failures. Our approach is driven by a key observation: in most failure cases, the confidence of the multimodal prediction is significantly lower than that of at least one unimodal branch, a phenomenon we term confidence degradation. To mitigate this, we introduce an Adaptive Confidence Loss that penalizes such degradations during training. In addition, we propose Multimodal Feature Swapping, a novel outlier synthesis technique that generates challenging, failure-aware training examples. By training with these synthetic failures, ACR learns to more effectively recognize and reject uncertain predictions, thereby improving overall reliability. Extensive experiments across four datasets, three modalities, and multiple evaluation settings demonstrate that ACR achieves consistent and robust gains. The source code will be available at this https URL. 

---
# Symbol-Equivariant Recurrent Reasoning Models 

**Authors**: Richard Freinschlag, Timo Bertram, Erich Kobler, Andreas Mayr, Günter Klambauer  

**Link**: [PDF](https://arxiv.org/pdf/2603.02193)  

**Abstract**: Reasoning problems such as Sudoku and ARC-AGI remain challenging for neural networks. The structured problem solving architecture family of Recurrent Reasoning Models (RRMs), including Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM), offer a compact alternative to large language models, but currently handle symbol symmetries only implicitly via costly data augmentation. We introduce Symbol-Equivariant Recurrent Reasoning Models (SE-RRMs), which enforce permutation equivariance at the architectural level through symbol-equivariant layers, guaranteeing identical solutions under symbol or color permutations. SE-RRMs outperform prior RRMs on 9x9 Sudoku and generalize from just training on 9x9 to smaller 4x4 and larger 16x16 and 25x25 instances, to which existing RRMs cannot extrapolate. On ARC-AGI-1 and ARC-AGI-2, SE-RRMs achieve competitive performance with substantially less data augmentation and only 2 million parameters, demonstrating that explicitly encoding symmetry improves the robustness and scalability of neural reasoning. Code is available at this https URL. 

---
# Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation 

**Authors**: Divyanshu Daiya, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2603.02190)  

**Abstract**: We present Sketch2Colab, which turns storyboard-style 2D sketches into coherent, object-aware 3D multi-human motion with fine-grained control over agents, joints, timing, and contacts. Conventional diffusion-based motion generators have advanced realism; however, achieving precise adherence to rich interaction constraints typically demands extensive training and/or costly posterior guidance, and performance can degrade under strong multi-entity conditioning. Sketch2Colab instead first learns a sketch-driven diffusion prior and then distills it into an efficient rectified-flow student operating in latent space for fast, stable sampling. Differentiable energies over keyframes, trajectories, and physics-based constraints directly shape the student's transport field, steering samples toward motions that faithfully satisfy the storyboard while remaining physically plausible. To capture coordinated interaction, we augment the continuous flow with a continuous-time Markov chain (CTMC) planner that schedules discrete events such as touches, grasps, and handoffs, modulating the dynamics to produce crisp, well-phased human-object-human collaborations. Experiments on CORE4D and InterHuman show that Sketch2Colab achieves state-of-the-art constraint adherence and perceptual quality while offering significantly faster inference than diffusion-only baselines. 

---
# MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms 

**Authors**: Jinqi Wu, Sishuo Chen, Zhangming Chan, Yong Bai, Lei Zhang, Sheng Chen, Chenghuan Hou, Xiang-Rong Sheng, Han Zhu, Jian Xu, Bo Zheng, Chaoyou Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02184)  

**Abstract**: Multi-attribution learning (MAL), which enhances model performance by learning from conversion labels yielded by multiple attribution mechanisms, has emerged as a promising learning paradigm for conversion rate (CVR) prediction. However, the conversion labels in public CVR datasets are generated by a single attribution mechanism, hindering the development of MAL approaches.
To address this data gap, we establish the Multi-Attribution Benchmark (MAC), the first public CVR dataset featuring labels from multiple attribution mechanisms. Besides, to promote reproducible research on MAL, we develop PyMAL, an open-source library covering a wide array of baseline methods. We conduct comprehensive experimental analyses on MAC and reveal three key insights: (1) MAL brings consistent performance gains across different attribution settings, especially for users featuring long conversion paths. (2) The performance growth scales up with objective complexity in most settings; however, when predicting first-click conversion targets, simply adding auxiliary objectives is counterproductive, underscoring the necessity of careful selection of auxiliary objectives. (3) Two architectural design principles are paramount: first, to fully learn the multi-attribution knowledge, and second, to fully leverage this knowledge to serve the main task. Motivated by these findings, we propose Mixture of Asymmetric Experts (MoAE), an effective MAL approach incorporating multi-attribution knowledge learning and main task-centric knowledge utilization. Experiments on MAC show that MoAE substantially surpasses the existing state-of-the-art MAL method. We believe that our benchmark and insights will foster future research in the MAL field. Our MAC benchmark and the PyMAL algorithm library are publicly available at this https URL. 

---
# Leveraging Model Soups to Classify Intangible Cultural Heritage Images from the Mekong Delta 

**Authors**: Quoc-Khang Tran, Minh-Thien Nguyen, Nguyen-Khang Pham  

**Link**: [PDF](https://arxiv.org/pdf/2603.02181)  

**Abstract**: The classification of Intangible Cultural Heritage (ICH) images in the Mekong Delta poses unique challenges due to limited annotated data, high visual similarity among classes, and domain heterogeneity. In such low-resource settings, conventional deep learning models often suffer from high variance or overfit to spurious correlations, leading to poor generalization. To address these limitations, we propose a robust framework that integrates the hybrid CoAtNet architecture with model soups, a lightweight weight-space ensembling technique that averages checkpoints from a single training trajectory without increasing inference cost. CoAtNet captures both local and global patterns through stage-wise fusion of convolution and self-attention. We apply two ensembling strategies - greedy and uniform soup - to selectively combine diverse checkpoints into a final model. Beyond performance improvements, we analyze the ensembling effect through the lens of bias-variance decomposition. Our findings show that model soups reduces variance by stabilizing predictions across diverse model snapshots, while introducing minimal additional bias. Furthermore, using cross-entropy-based distance metrics and Multidimensional Scaling (MDS), we show that model soups selects geometrically diverse checkpoints, unlike Soft Voting, which blends redundant models centered in output space. Evaluated on the ICH-17 dataset (7,406 images across 17 classes), our approach achieves state-of-the-art results with 72.36% top-1 accuracy and 69.28% macro F1-score, outperforming strong baselines including ResNet-50, DenseNet-121, and ViT. These results underscore that diversity-aware checkpoint averaging provides a principled and efficient way to reduce variance and enhance generalization in culturally rich, data-scarce classification tasks. 

---
# Reservoir Subspace Injection for Online ICA under Top-n Whitening 

**Authors**: Wenjun Xiao, Yuda Bi, Vince D Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2603.02178)  

**Abstract**: Reservoir expansion can improve online independent component analysis (ICA) under nonlinear mixing, yet top-$n$ whitening may discard injected features. We formalize this bottleneck as \emph{reservoir subspace injection} (RSI): injected features help only if they enter the retained eigenspace without displacing passthrough directions. RSI diagnostics (IER, SSO, $\rho_x$) identify a failure mode in our top-$n$ setting: stronger injection increases IER but crowds out passthrough energy ($\rho_x: 1.00\!\rightarrow\!0.77$), degrading SI-SDR by up to $2.2$\,dB. A guarded RSI controller preserves passthrough retention and recovers mean performance to within $0.1$\,dB of baseline $1/N$ scaling. With passthrough preserved, RE-OICA improves over vanilla online ICA by $+1.7$\,dB under nonlinear mixing and achieves positive SI-SDR$_{\mathrm{sc}}$ on the tested super-Gaussian benchmark ($+0.6$\,dB). 

---
# Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance 

**Authors**: Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2603.02175)  

**Abstract**: Instruction-based video editing has witnessed rapid progress, yet current methods often struggle with precise visual control, as natural language is inherently limited in describing complex visual nuances. Although reference-guided editing offers a robust solution, its potential is currently bottlenecked by the scarcity of high-quality paired training data. To bridge this gap, we introduce a scalable data generation pipeline that transforms existing video editing pairs into high-fidelity training quadruplets, leveraging image generative models to create synthesized reference scaffolds. Using this pipeline, we construct RefVIE, a large-scale dataset tailored for instruction-reference-following tasks, and establish RefVIE-Bench for comprehensive evaluation. Furthermore, we propose a unified editing architecture, Kiwi-Edit, that synergizes learnable queries and latent visual features for reference semantic guidance. Our model achieves significant gains in instruction following and reference fidelity via a progressive multi-stage training curriculum. Extensive experiments demonstrate that our data and architecture establish a new state-of-the-art in controllable video editing. All datasets, models, and code is released at this https URL. 

---
# SageBwd: A Trainable Low-bit Attention 

**Authors**: Jintao Zhang, Marco Chen, Haoxu Wang, Kai Jiang, Ion Stoica, Joseph E. Gonzalez, Jianfei Chen, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02170)  

**Abstract**: Low-bit attention, such as SageAttention, has emerged as an effective approach for accelerating model inference, but its applicability to training remains poorly understood. In prior work, we introduced SageBwd, a trainable INT8 attention that quantizes six of seven attention matrix multiplications while preserving fine-tuning performance. However, SageBwd exhibited a persistent performance gap to full-precision attention (FPA) during pre-training. In this work, we investigate why this gap occurs and demonstrate that SageBwd matches full-precision attention during pretraining. Through experiments and theoretical analysis, we reach a few important insights and conclusions: (i) QK-norm is necessary for stable training at large tokens per step, (ii) quantization errors primarily arise from the backward-pass score gradient dS, (iii) reducing tokens per step enables SageBwd to match FPA performance in pre-training, and (iv) K-smoothing remains essential for training stability, while Q-smoothing provides limited benefit during pre-training. 

---
# How Small Can 6G Reason? Scaling Tiny Language Models for AI-Native Networks 

**Authors**: Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2603.02156)  

**Abstract**: Emerging 6G visions, reflected in ongoing standardization efforts within 3GPP, IETF, ETSI, ITU-T, and the O-RAN Alliance, increasingly characterize networks as AI-native systems in which high-level semantic reasoning layers operate above standardized control and data-plane functions. Although frontier-scale large language models (LLMs) such as Qwen2.5-7B and Olmo-3-7B demonstrate strong reasoning capability, their computational footprint limits deployment in latency-sensitive, edge-native infrastructures. This paper presents a systematic empirical study of the scaling behavior and deployment efficiency of compact language models for network-level semantic reasoning in AI-native 6G systems. Using 6G-Bench, a standardization-aligned benchmark comprising 30 decision-making tasks across five capability domains, we evaluate models ranging from 135M (SmolLM2-135M) to 7B parameters (Qwen2.5-7B), including mid-scale architectures such as Llama-3.2-1B, Granite-1B, and Qwen2.5-3B. Deterministic accuracy (pass@1) increases from 0.224 at 135M to 0.707 at 7B, but scaling gains are highly non-uniform. A pronounced stability transition occurs in the 1 to 1.5B range, where accuracy rises from 0.373 (Llama-3.2-1B) to 0.531 (Qwen2.5-1.5B) and the instability gap Delta_5 contracts from 0.356 to 0.138. Beyond 3B parameters, improvements diminish (+0.064 from 3B to 7B). Through single-query inference profiling and an Edge Score metric that normalizes accuracy by latency and memory footprint, we show that semantic reliability per unit edge resource does not scale monotonically with parameter count. Instead, mid-scale models (approximately 1.5 to 3B) achieve the most favorable balance between deterministic stability and computational efficiency, providing deployment-relevant guidance for AI-native 6G architectures. All scripts and results are publicly available at this https URL 

---
# Near-Optimal Regret for KL-Regularized Multi-Armed Bandits 

**Authors**: Kaixuan Ji, Qingyue Zhao, Heyang Zhao, Qiwei Di, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02155)  

**Abstract**: Recent studies have shown that reinforcement learning with KL-regularized objectives can enjoy faster rates of convergence or logarithmic regret, in contrast to the classical $\sqrt{T}$-type regret in the unregularized setting. However, the statistical efficiency of online learning with respect to KL-regularized objectives remains far from completely characterized, even when specialized to multi-armed bandits (MABs). We address this problem for MABs via a sharp analysis of KL-UCB using a novel peeling argument, which yields a $\tilde{O}(\eta K\log^2T)$ upper bound: the first high-probability regret bound with linear dependence on $K$. Here, $T$ is the time horizon, $K$ is the number of arms, $\eta^{-1}$ is the regularization intensity, and $\tilde{O}$ hides all logarithmic factors except those involving $\log T$. The near-tightness of our analysis is certified by the first non-constant lower bound $\Omega(\eta K \log T)$, which follows from subtle hard-instance constructions and a tailored decomposition of the Bayes prior. Moreover, in the low-regularization regime (i.e., large $\eta$), we show that the KL-regularized regret for MABs is $\eta$-independent and scales as $\tilde{\Theta}(\sqrt{KT})$. Overall, our results provide a thorough understanding of KL-regularized MABs across all regimes of $\eta$ and yield nearly optimal bounds in terms of $K$, $\eta$, and $T$. 

---
# Boltzmann-based Exploration for Robust Decentralized Multi-Agent Planning 

**Authors**: Nhat Nguyen, Duong Nguyen, Gianluca Rizzo, Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02154)  

**Abstract**: Decentralized Monte Carlo Tree Search (Dec-MCTS) is widely used for cooperative multi-agent planning but struggles in sparse or skewed reward environments. We introduce Coordinated Boltzmann MCTS (CB-MCTS), which replaces deterministic UCT with a stochastic Boltzmann policy and a decaying entropy bonus for sustained yet focused exploration. While Boltzmann exploration has been studied in single-agent MCTS, applying it in multi-agent systems poses unique challenges. CB-MCTS is the first to address this. We analyze CB-MCTS in the simple-regret setting and show in simulations that it outperforms Dec-MCTS in deceptive scenarios and remains competitive on standard benchmarks, providing a robust solution for multi-agent planning. 

---
# Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from an Industry Deployment 

**Authors**: Luigi Medrano, Arush Verma, Mukul Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2603.02153)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems commonly adopt retrieval fusion techniques such as multi-query retrieval and reciprocal rank fusion (RRF) to increase document recall, under the assumption that higher recall leads to better answer quality. While these methods show consistent gains in isolated retrieval benchmarks, their effectiveness under realistic production constraints remains underexplored. In this work, we evaluate retrieval fusion in a production-style RAG pipeline operating over an enterprise knowledge base, with fixed retrieval depth, re-ranking budgets, and latency constraints.
Across multiple fusion configurations, we find that retrieval fusion does increase raw recall, but these gains are largely neutralized after re-ranking and truncation. In our setting, fusion variants fail to outperform single-query baselines on KB-level Top-$k$ accuracy, with Hit@10 decreasing from $0.51$ to $0.48$ in several configurations. Moreover, fusion introduces additional latency overhead due to query rewriting and larger candidate sets, without corresponding improvements in downstream effectiveness.
Our analysis suggests that recall-oriented fusion techniques exhibit diminishing returns once realistic re-ranking limits and context budgets are applied. We conclude that retrieval-level improvements do not reliably translate into end-to-end gains in production RAG systems, and argue for evaluation frameworks that jointly consider retrieval quality, system efficiency, and downstream impact. 

---
# Zero- and Few-Shot Named-Entity Recognition: Case Study and Dataset in the Crime Domain (CrimeNER) 

**Authors**: Miguel Lopez-Duran, Julian Fierrez, Aythami Morales, Daniel DeAlcala, Gonzalo Mancera, Javier Irigoyen, Ruben Tolosana, Oscar Delgado, Francisco Jurado, Alvaro Ortigosa  

**Link**: [PDF](https://arxiv.org/pdf/2603.02150)  

**Abstract**: The extraction of critical information from crime-related documents is a crucial task for law enforcement agencies. Named-Entity Recognition (NER) can perform this task in extracting information about the crime, the criminal, or law enforcement agencies involved. However, there is a considerable lack of adequately annotated data on general real-world crime scenarios. To address this issue, we present CrimeNER, a case-study of Crime-related zero- and Few-Shot NER, and a general Crime-related Named-Entity Recognition database (CrimeNERdb) consisting of more than 1.5k annotated documents for the NER task extracted from public reports on terrorist attacks and the U.S. Department of Justice's press notes. We define 5 types of coarse crime entity and a total of 22 types of fine-grained entity. We address the quality of the case-study and the annotated data with experiments on Zero and Few-Shot settings with State-of-the-Art NER models as well as generalist and commonly used Large Language Models. 

---
# LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation 

**Authors**: Hualiang Wei, Shunran Jia, Jialun Liu, Wenhui Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02129)  

**Abstract**: We present LiftAvatar, a new paradigm that completes sparse monocular observations in kinematic space (e.g., facial expressions and head pose) and uses the completed signals to drive high-fidelity avatar animation. LiftAvatar is a fine-grained, expression-controllable large-scale video diffusion Transformer that synthesizes high-quality, temporally coherent expression sequences conditioned on single or multiple reference images. The key idea is to lift incomplete input data into a richer kinematic representation, thereby strengthening both reconstruction and animation in downstream 3D avatar pipelines. To this end, we introduce (i) a multi-granularity expression control scheme that combines shading maps with expression coefficients for precise and stable driving, and (ii) a multi-reference conditioning mechanism that aggregates complementary cues from multiple frames, enabling strong 3D consistency and controllability. As a plug-and-play enhancer, LiftAvatar directly addresses the limited expressiveness and reconstruction artifacts of 3D Gaussian Splatting-based avatars caused by sparse kinematic cues in everyday monocular videos. By expanding incomplete observations into diverse pose-expression variations, LiftAvatar also enables effective prior distillation from large-scale video generative models into 3D pipelines, leading to substantial gains. Extensive experiments show that LiftAvatar consistently boosts animation quality and quantitative metrics of state-of-the-art 3D avatar methods, especially under extreme, unseen expressions. 

---
# LLMs as Strategic Actors: Behavioral Alignment, Risk Calibration, and Argumentation Framing in Geopolitical Simulations 

**Authors**: Veronika Solopova, Viktoria Skorik, Maksym Tereshchenko, Alina Haidun, Ostap Vykhopen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02128)  

**Abstract**: Large language models (LLMs) are increasingly proposed as agents in strategic decision environments, yet their behavior in structured geopolitical simulations remains under-researched. We evaluate six popular state-of-the-art LLMs alongside results from human results across four real-world crisis simulation scenarios, requiring models to select predefined actions and justify their decisions across multiple rounds. We compare models to humans in action alignment, risk calibration through chosen actions' severity, and argumentative framing grounded in international relations theory. Results show that models approximate human decision patterns in base simulation rounds but diverge over time, displaying distinct behavioural profiles and strategy updates. LLM explanations for chosen actions across all models exhibit a strong normative-cooperative framing centered on stability, coordination, and risk mitigation, with limited adversarial reasoning. 

---
# Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons 

**Authors**: Anthony Liang, Yigit Korkmaz, Jiahui Zhang, Minyoung Hwang, Abrar Anwar, Sidhant Kaushik, Aditya Shah, Alex S. Huang, Luke Zettlemoyer, Dieter Fox, Yu Xiang, Anqi Li, Andreea Bobu, Abhishek Gupta, Stephen Tu, Erdem Biyik, Jesse Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02115)  

**Abstract**: General-purpose robot reward models are typically trained to predict absolute task progress from expert demonstrations, providing only local, frame-level supervision. While effective for expert demonstrations, this paradigm scales poorly to large-scale robotics datasets where failed and suboptimal trajectories are abundant and assigning dense progress labels is ambiguous. We introduce Robometer, a scalable reward modeling framework that combines intra-trajectory progress supervision with inter-trajectory preference supervision. Robometer is trained with a dual objective: a frame-level progress loss that anchors reward magnitude on expert data, and a trajectory-comparison preference loss that imposes global ordering constraints across trajectories of the same task, enabling effective learning from both real and augmented failed trajectories. To support this formulation at scale, we curate RBM-1M, a reward-learning dataset comprising over one million trajectories spanning diverse robot embodiments and tasks, including substantial suboptimal and failure data. Across benchmarks and real-world evaluations, Robometer learns more generalizable reward functions than prior methods and improves robot learning performance across a diverse set of downstream applications. Code, model weights, and videos at this https URL. 

---
# FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding 

**Authors**: Yiweng Xie, Bo He, Junke Wang, Xiangyu Zheng, Ziyi Ye, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02096)  

**Abstract**: This paper presents FluxMem, a training-free framework for efficient streaming video understanding. FluxMem adaptively compresses redundant visual memory through a hierarchical, two-stage design: (1) a Temporal Adjacency Selection (TAS) module removes redundant visual tokens across adjacent frames, and (2) a Spatial Domain Consolidation (SDC) module further merges spatially repetitive regions within each frame into compact representations. To adapt effectively to dynamic scenes, we introduce a self-adaptive token compression mechanism in both TAS and SDC, which automatically determines the compression rate based on intrinsic scene statistics rather than manual tuning. Extensive experiments demonstrate that FluxMem achieves new state-of-the-art results on existing online video benchmarks, reaching 76.4 on StreamingBench and 67.2 on OVO-Bench under real-time settings, while reducing latency by 69.9% and peak GPU memory by 34.5% on OVO-Bench. Furthermore, it maintains strong offline performance, achieving 73.1 on MLVU while using 65% fewer visual tokens. 

---
# On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective 

**Authors**: Guy Smorodinsky, Sveta Gimpleson, Itay Safran  

**Link**: [PDF](https://arxiv.org/pdf/2603.02095)  

**Abstract**: We study the convergence dynamics of Gradient Descent (GD) in a minimal binary classification setting, consisting of a two-neuron ReLU network and two training instances. We prove that even under these strong simplifying assumptions, while GD successfully converges to an optimal robustness margin, effectively maximizing the distance between the decision boundary and the training points, this convergence occurs at a prohibitively slow rate, scaling strictly as $\Theta(1/\ln(t))$. To the best of our knowledge, this establishes the first explicit lower bound on the convergence rate of the robustness margin in a non-linear model. Through empirical simulations, we further demonstrate that this inherent failure mode is pervasive, exhibiting the exact same tight convergence rate across multiple natural network initializations. Our theoretical guarantees are derived via a rigorous analysis of the GD trajectories across the distinct activation patterns of the model. Specifically, we develop tight control over the system's dynamics to bound the trajectory of the decision boundary, overcoming the primary technical challenge introduced by the non-linear nature of the architecture. 

---
# Learning from Synthetic Data Improves Multi-hop Reasoning 

**Authors**: Anmol Kabra, Yilun Yin, Albert Gong, Kamilė Stankevičiūtė, Dongyoung Go, Johann Lee, Katie Z. Luo, Carla P. Gomes, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2603.02091)  

**Abstract**: Reinforcement Learning (RL) has been shown to significantly boost reasoning capabilities of large language models (LLMs) in math, coding, and multi-hop reasoning tasks. However, RL fine-tuning requires abundant high-quality verifiable data, often sourced from human annotations, generated from frontier LLMs, or scored by LLM-based verifiers. All three have considerable limitations: human-annotated datasets are small and expensive to curate, LLM-generated data is hallucination-prone and costly, and LLM-based verifiers are inaccurate and slow. In this work, we investigate a cheaper alternative: RL fine-tuning on rule-generated synthetic data for multi-hop reasoning tasks. We discover that LLMs fine-tuned on synthetic data perform significantly better on popular real-world question-answering benchmarks, despite the synthetic data containing only fictional knowledge. On stratifying performance by question difficulty, we find that synthetic data teaches LLMs to compose knowledge -- a fundamental and generalizable reasoning skill. Our work highlights rule-generated synthetic reasoning data as a free and scalable resource to improve LLM reasoning capabilities. 

---
# Detection-Gated Glottal Segmentation with Zero-Shot Cross-Dataset Transfer and Clinical Feature Extraction 

**Authors**: Harikrishnan Unnikrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02087)  

**Abstract**: Background: Accurate glottal segmentation in high-speed videoendoscopy (HSV) is essential for extracting kinematic biomarkers of laryngeal function. However, existing deep learning models often produce spurious artifacts in non-glottal frames and fail to generalize across different clinical settings.
Methods: We propose a detection-gated pipeline that integrates a YOLOv8-based detector with a U-Net segmenter. A temporal consistency wrapper ensures robustness by suppressing false positives during glottal closure and instrument occlusion. The model was trained on a limited subset of the GIRAFE dataset (600 frames) and evaluated via zero-shot transfer on the large-scale BAGLS dataset.
Results: The pipeline achieved state-of-the-art performance on the GIRAFE benchmark (DSC 0.81) and demonstrated superior generalizability on BAGLS (DSC 0.85, in-distribution) without institutional fine-tuning. Downstream validation on a 65-subject clinical cohort confirmed that automated kinematic features (Open Quotient, coefficient of variation) remained consistent with established clinical benchmarks. The coefficient of variation (CV) of the glottal area was found to be a significant marker for distinguishing healthy from pathological vocal function (p=0.006).
Conclusions: The detection-gated architecture provides a lightweight, computationally efficient solution (~35 frames/s) for real-time clinical use. By enabling robust zero-shot transfer, this framework facilitates the standardized, large-scale extraction of clinical biomarkers across diverse endoscopy platforms. Code, trained weights, and evaluation scripts are released at this https URL. 

---
# GenDB: The Next Generation of Query Processing -- Synthesized, Not Engineered 

**Authors**: Jiale Lao, Immanuel Trummer  

**Link**: [PDF](https://arxiv.org/pdf/2603.02081)  

**Abstract**: Traditional query processing relies on engines that are carefully optimized and engineered by many experts. However, new techniques and user requirements evolve rapidly, and existing systems often cannot keep pace. At the same time, these systems are difficult to extend due to their internal complexity, and developing new systems requires substantial engineering effort and cost. In this paper, we argue that recent advances in Large Language Models (LLMs) are starting to shape the next generation of query processing systems.
We propose using LLMs to synthesize execution code for each incoming query, instead of continuously building, extending, and maintaining complex query processing engines. As a proof of concept, we present GenDB, an LLM-powered agentic system that generates instance-optimized and customized query execution code tailored to specific data, workloads, and hardware resources.
We implemented an early prototype of GenDB that uses Claude Code Agent as the underlying component in the multi-agent system, and we evaluate it on OLAP workloads. We use queries from the well-known TPC-H benchmark and also construct a new benchmark designed to reduce potential data leakage from LLM training data. We compare GenDB with state-of-the-art query engines, including DuckDB, Umbra, MonetDB, ClickHouse, and PostgreSQL. GenDB achieves significantly better performance than these systems. Finally, we discuss the current limitations of GenDB and outline future extensions and related research challenges. 

---
# Cognitive Prosthetic: An AI-Enabled Multimodal System for Episodic Recall in Knowledge Work 

**Authors**: Lawrence Obiuwevwi, Krzysztof J. Rechowicz, Vikas Ashok, Sachin Shetty, Sampath Jayarathna  

**Link**: [PDF](https://arxiv.org/pdf/2603.02072)  

**Abstract**: Modern knowledge workplaces increasingly strain human episodic memory as individuals navigate fragmented attention, overlapping meetings, and multimodal information streams. Existing workplace tools provide partial support through note-taking or analytics but rarely integrate cognitive, physiological, and attentional context into retrievable memory representations. This paper presents the Cognitive Prosthetic Multimodal System (CPMS) --an AI-enabled proof-of-concept designed to support episodic recall in knowledge work through structured episodic capture and natural language retrieval. CPMS synchronizes speech transcripts, physiological signals, and gaze behavior into temporally aligned, JSON-based episodic records processed locally for privacy. Beyond data logging, the system includes a web-based retrieval interface that allows users to query past workplace experiences using natural language, referencing semantic content, time, attentional focus, or physiological state. We present CPMS as a functional proof-of-concept demonstrating the technical feasibility of transforming heterogeneous sensor data into queryable episodic memories. The system is designed to be modular, supporting operation with partial sensor configurations, and incorporates privacy safeguards for workplace deployment. This work contributes an end-to-end, privacy-aware architecture for AI-enabled memory augmentation in workplace settings. 

---
# Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD? 

**Authors**: Jihwan Kim, Dogyoon Song, Chulhee Yun  

**Link**: [PDF](https://arxiv.org/pdf/2603.02069)  

**Abstract**: We study scaling laws of signSGD under a power-law random features (PLRF) model that accounts for both feature and target decay. We analyze the population risk of a linear model trained with one-pass signSGD on Gaussian-sketched features. We express the risk as a function of model size, training steps, learning rate, and the feature and target decay parameters. Comparing against the SGD risk analyzed by Paquette et al. (2024), we identify a drift-normalization effect and a noise-reshaping effect unique to signSGD. We then obtain compute-optimal scaling laws under the optimal choice of learning rate. Our analysis shows that the noise-reshaping effect can make the compute-optimal slope of signSGD steeper than that of SGD in regimes where noise is dominant. Finally, we observe that the widely used warmup-stable-decay (WSD) schedule further reduces the noise term and sharpens the compute-optimal slope, when feature decay is fast but target decay is slow. 

---
# A Resource-Rational Principle for Modeling Visual Attention Control 

**Authors**: Yunpeng Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.02056)  

**Abstract**: Understanding how people allocate visual attention is central to Human-Computer Interaction (HCI), yet existing computational models of attention are often either descriptive, task-specific, or difficult to interpret. My dissertation develops a resource-rational, simulation-based framework for modeling visual attention as a sequential decision-making process under perceptual, memory, and time constraints. I formalize visual tasks, such as reading and multitasking, as bounded-optimal control problems using Partially Observable Markov Decision Processes, enabling eye-movement behaviors such as fixation and attention switching to emerge from rational adaptation rather than being hand-coded or purely data-driven. These models are instantiated in simulation environments spanning traditional text reading and reading-while-walking with smart glasses, where they reproduce classic empirical effects, explain observed trade-offs between comprehension and safety, and generate novel predictions under time pressure and interface variation. Collectively, this work contributes a unified computational account of visual attention, offering new tools for theory-driven and resource-efficient HCI design. 

---
# "When to Hand Off, When to Work Together": Expanding Human-Agent Co-Creative Collaboration through Concurrent Interaction 

**Authors**: Kihoon Son, Hyewon Lee, DaEun Choi, Yoonsu Kim, Tae Soo Kim, Yoonjoo Lee, John Joon Young Chung, HyunJoon Jung, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.02050)  

**Abstract**: Human collaborators coordinate dynamically through process visibility and workspace awareness, yet AI agents typically either provide only final outputs or expose read-only execution processes (e.g., planning, reasoning) without interpreting concurrent user actions on shared artifacts. Building on mixed-initiative interaction principles, we explore whether agents can achieve collaborative context awareness -- interpreting concurrent user actions on shared artifacts and adapting in real-time. Study 1 (N=10 professional designers) revealed that process visibility enabled reasoning about agent actions but exposed conflicts when agents could not distinguish feedback from independent work. We developed CLEO, which interprets collaborative intent and adapts in real-time. Study 2 (N=10, two-day with stimulated recall interviews) analyzed 214 turns, identifying five action patterns, six triggers, and four enabling factors explaining when designers choose delegation (70.1%), direction (28.5%), or concurrent work (31.8%). We present a decision model with six interaction loops, design implications, and an annotated dataset. 

---
# EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training 

**Authors**: Aleksei Dorkin, Taido Purason, Emil Kalbaliyev, Hele-Andra Kuulmets, Marii Ojastu, Mark Fišel, Tanel Alumäe, Eleri Aedmaa, Krister Kruusmaa, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2603.02041)  

**Abstract**: Large language models (LLMs) are predominantly trained on English-centric data, resulting in uneven performance for smaller languages. We study whether continued pretraining (CPT) can substantially improve Estonian capabilities in a pretrained multilingual LLM while preserving its English and general reasoning performance. Using Llama 3.1 8B as the main base model, we perform CPT on a mixture that increases Estonian exposure while approximating the original training distribution through English replay and the inclusion of code, mathematics, and instruction-like data. We subsequently apply supervised fine-tuning, preference optimization, and chat vector merging to introduce robust instruction-following behavior. Evaluation on a comprehensive suite of Estonian benchmarks shows consistent gains in linguistic competence, knowledge, reasoning, translation quality, and instruction-following compared to the original base model and its instruction-tuned variant, while maintaining competitive performance on English benchmarks. These findings indicate that CPT, with an appropriately balanced data mixture, together with post-training alignment, can substantially improve single-language capabilities in pretrained multilingual LLMs. 

---
# Revealing Combinatorial Reasoning of GNNs via Graph Concept Bottleneck Layer 

**Authors**: Yue Niu, Zhaokai Sun, Jiayi Yang, Xiaofeng Cao, Rui Fan, Xin Sun, Hanli Wang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.02025)  

**Abstract**: Despite their success in various domains, the growing dependence on GNNs raises a critical concern about the nature of the combinatorial reasoning underlying their predictions, which is often hidden within their black-box architectures. Addressing this challenge requires understanding how GNNs translate topological patterns into logical rules. However, current works only uncover the hard logical rules over graph concepts, which cannot quantify the contribution of each concept to prediction. Moreover, they are post-hoc interpretable methods that generate explanations after model training and may not accurately reflect the true combinatorial reasoning of GNNs, since they approximate it with a surrogate. In this work, we develop a graph concept bottleneck layer that can be integrated into any GNN architectures to guide them to predict the selected discriminative global graph concepts. The predicted concept scores are further projected to class labels by a sparse linear layer. It enforces the combinatorial reasoning of GNNs' predictions to fit the soft logical rule over graph concepts and thus can quantify the contribution of each concept. To further improve the quality of the concept bottleneck, we treat concepts as "graph words" and graphs as "graph sentences", and leverage language models to learn graph concept embeddings. Extensive experiments on multiple datasets show that our method GCBMs achieve state-of-the-art performance both in classification and interpretability. 

---
# MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning 

**Authors**: Jiachun Li, Shaoping Huang, Zhuoran Jin, Chenlong Zhang, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02024)  

**Abstract**: Recent progress in the reasoning capabilities of multimodal large language models (MLLMs) has empowered them to address more complex tasks such as scientific analysis and mathematical reasoning. Despite their promise, MLLMs' reasoning abilities across different scenarios in real life remain largely unexplored and lack standardized benchmarks for evaluation. To address this gap, we introduce MMR-Life, a comprehensive benchmark designed to evaluate the diverse multimodal multi-image reasoning capabilities of MLLMs across real-life scenarios. MMR-Life consists of 2,646 multiple-choice questions based on 19,108 images primarily sourced from real-world contexts, comprehensively covering seven reasoning types: abductive, analogical, causal, deductive, inductive, spatial, and temporal. Unlike existing reasoning benchmarks, MMR-Life does not rely on domain-specific expertise but instead requires models to integrate information across multiple images and apply diverse reasoning abilities. The evaluation of 37 advanced models highlights the substantial challenge posed by MMR-Life. Even top models like GPT-5 achieve only 58% accuracy and display considerable variance in performance across reasoning types. Moreover, we analyze the reasoning paradigms of existing MLLMs, exploring how factors such as thinking length, reasoning method, and reasoning type affect their performance. In summary, MMR-Life establishes a comprehensive foundation for evaluating, analyzing, and improving the next generation of multimodal reasoning systems. 

---
# CodecFlow: Efficient Bandwidth Extension via Conditional Flow Matching in Neural Codec Latent Space 

**Authors**: Bowen Zhang, Junchuan Zhao, Ian McLoughlin, Ye Wang, A S Madhukumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02022)  

**Abstract**: Speech Bandwidth Extension improves clarity and intelligibility by restoring/inferring appropriate high-frequency content for low-bandwidth speech. Existing methods often rely on spectrogram or waveform modeling, which can incur higher computational cost and have limited high-frequency fidelity. Neural audio codecs offer compact latent representations that better preserve acoustic detail, yet accurately recovering high-resolution latent information remains challenging due to representation mismatch. We present CodecFlow, a neural codec-based BWE framework that performs efficient speech reconstruction in a compact latent space. CodecFlow employs a voicing-aware conditional flow converter on continuous codec embeddings and a structure-constrained residual vector quantizer to improve latent alignment stability. Optimized end-to-end, CodecFlow achieves strong spectral fidelity and enhanced perceptual quality on 8 kHz to 16 kHz and 44.1 kHz speech BWE tasks. 

---
# Selection as Power: Constrained Reinforcement for Bounded Decision Authority 

**Authors**: Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Díaz  

**Link**: [PDF](https://arxiv.org/pdf/2603.02019)  

**Abstract**: Selection as Power argued that upstream selection authority, rather than internal objective misalignment, constitutes a primary source of risk in high-stakes agentic systems. However, the original framework was static: governance constraints bounded selection power but did not adapt over time. In this work, we extend the framework to dynamic settings by introducing incentivized selection governance, where reinforcement updates are applied to scoring and reducer parameters under externally enforced sovereignty constraints.
We formalize selection as a constrained reinforcement process in which parameter updates are projected onto governance-defined feasible sets, preventing concentration beyond prescribed bounds. Across multiple regulated financial scenarios, unconstrained reinforcement consistently collapses into deterministic dominance under repeated feedback, especially at higher learning rates. In contrast, incentivized governance enables adaptive improvement while maintaining bounded selection concentration.
Projection-based constraints transform reinforcement from irreversible lock-in into controlled adaptation, with governance debt quantifying the tension between optimization pressure and authority bounds. These results demonstrate that learning dynamics can coexist with structural diversity when sovereignty constraints are enforced at every update step, offering a principled approach to integrating reinforcement into high-stakes agentic systems without surrendering bounded selection authority. 

---
# MAP-Diff: Multi-Anchor Guided Diffusion for Progressive 3D Whole-Body Low-Dose PET Denoising 

**Authors**: Peiyuan Jing, Chun-Wun Cheng, Liutao Yang, Zhenxuan Zhang, Thiago V. Lima, Klaus Strobel, Antoine Leimgruber, Angelica Aviles-Rivero, Guang Yang, Javier A. Montoya-Zegarra  

**Link**: [PDF](https://arxiv.org/pdf/2603.02012)  

**Abstract**: Low-dose Positron Emission Tomography (PET) reduces radiation exposure but suffers from severe noise and quantitative degradation. Diffusion-based denoising models achieve strong final reconstructions, yet their reverse trajectories are typically unconstrained and not aligned with the progressive nature of PET dose formation. We propose MAP-Diff, a multi-anchor guided diffusion framework for progressive 3D whole-body PET denoising. MAP-Diff introduces clinically observed intermediate-dose scans as trajectory anchors and enforces timestep-dependent supervision to regularize the reverse process toward dose-aligned intermediate states. Anchor timesteps are calibrated via degradation matching between simulated diffusion corruption and real multi-dose PET pairs, and a timestep-weighted anchor loss stabilizes stage-wise learning. At inference, the model requires only ultra-low-dose input while enabling progressive, dose-consistent intermediate restoration. Experiments on internal (Siemens Biograph Vision Quadra) and cross-scanner (United Imaging uEXPLORER) datasets show consistent improvements over strong CNN-, Transformer-, GAN-, and diffusion-based baselines. On the internal dataset, MAP-Diff improves PSNR from 42.48 dB to 43.71 dB (+1.23 dB), increases SSIM to 0.986, and reduces NMAE from 0.115 to 0.103 (-0.012) compared to 3D DDPM. Performance gains generalize across scanners, achieving 34.42 dB PSNR and 0.141 NMAE on the external cohort, outperforming all competing methods. 

---
# Mitigating topology biases in Graph Diffusion via Counterfactual Intervention 

**Authors**: Wendi Wang, Jiaxi Yang, Yongkang Du, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.02005)  

**Abstract**: Graph diffusion models have gained significant attention in graph generation tasks, but they often inherit and amplify topology biases from sensitive attributes (e.g. gender, age, region), leading to unfair synthetic graphs. Existing fair graph generation using diffusion models is limited to specific graph-based applications with complete labels or requires simultaneous updates for graph structure and node attributes, making them unsuitable for general usage. To relax these limitations by applying the debiasing method directly on graph topology, we propose Fair Graph Diffusion Model (FairGDiff), a counterfactual-based one-step solution that mitigates topology biases while balancing fairness and utility. In detail, we construct a causal model to capture the relationship between sensitive attributes, biased link formation, and the generated graph structure. By answering the counterfactual question "Would the graph structure change if the sensitive attribute were different?", we estimate an unbiased treatment and incorporate it into the diffusion process. FairGDiff integrates counterfactual learning into both forward diffusion and backward denoising, ensuring that the generated graphs are independent of sensitive attributes while preserving structural integrity. Extensive experiments on real-world datasets demonstrate that FairGDiff achieves a superior trade-off between fairness and utility, outperforming existing fair graph generation methods while maintaining scalability. 

---
# MatRIS: Toward Reliable and Efficient Pretrained Machine Learning Interaction Potentials 

**Authors**: Yuanchang Zhou, Siyu Hu, Xiangyu Zhang, Hongyu Wang, Guangming Tan, Weile Jia  

**Link**: [PDF](https://arxiv.org/pdf/2603.02002)  

**Abstract**: Foundation MLIPs demonstrate broad applicability across diverse material systems and have emerged as a powerful and transformative paradigm in chemical and computational materials science. Equivariant MLIPs achieve state-of-the-art accuracy in a wide range of benchmarks by incorporating equivariant inductive bias. However, the reliance on tensor products and high-degree representations makes them computationally costly. This raises a fundamental question: as quantum mechanical-based datasets continue to expand, can we develop a more compact model to thoroughly exploit high-dimensional atomic interactions? In this work, we present MatRIS (\textbf{Mat}erials \textbf{R}epresentation and \textbf{I}nteraction \textbf{S}imulation), an invariant MLIP that introduces attention-based modeling of three-body interactions. MatRIS leverages a novel separable attention mechanism with linear complexity $O(N)$, enabling both scalability and expressiveness. MatRIS delivers accuracy comparable to that of leading equivariant models on a wide range of popular benchmarks (Matbench-Discovery, MatPES, MDR phonon, Molecular dataset, etc). Taking Matbench-Discovery as an example, MatRIS achieves an F1 score of up to 0.847 and attains comparable accuracy at a lower training cost. The work indicates that our carefully designed invariant models can match or exceed the accuracy of equivariant models at a fraction of the cost, shedding light on the development of accurate and efficient MLIPs. 

---
# CharacterFlywheel: Scaling Iterative Improvement of Engaging and Steerable LLMs in Production 

**Authors**: Yixin Nie, Lin Guan, Zhongyao Ma, Anchit Gupta, Yipin Zhou, Xiao Li, Zhengping Zhou, Raymond Zeng, Gelin Zhou, Shigan Chu, Ajay Thampi, Wancen Mu, Nathan Shuster, Ketong Wang, Lin Chen, Jason Brewer, Derek Hao Hu, Alexander McCauley, Jason Weston, Sem Park, Na Zhang, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01973)  

**Abstract**: This report presents CharacterFlywheel, an iterative flywheel process for improving large language models (LLMs) in production social chat applications across Instagram, WhatsApp, and Messenger. Starting from LLaMA 3.1, we refined models across 15 generations using data from both internal and external real-user traffic. Through continuous deployments from July 2024 to April 2025, we conducted controlled 7-day A/B tests showing consistent engagement improvements: 7 of 8 newly deployed models demonstrated positive lift over the baseline, with the strongest performers achieving up to 8.8% improvement in engagement breadth and 19.4% in engagement depth. We also observed substantial gains in steerability, with instruction following increasing from 59.2% to 84.8% and instruction violations decreasing from 26.6% to 5.8%. We detail the CharacterFlywheel process which integrates data curation, reward modeling to estimate and interpolate the landscape of engagement metrics, supervised fine-tuning (SFT), reinforcement learning (RL), and both offline and online evaluation to ensure reliable progress at each optimization step. We also discuss our methods for overfitting prevention and navigating production dynamics at scale. These contributions advance the scientific rigor and understanding of LLMs in social applications serving millions of users. 

---
# Intrinsic Task Symmetry Drives Generalization in Algorithmic Tasks 

**Authors**: Hyeonbin Hwang, Yeachan Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.01968)  

**Abstract**: Grokking, the sudden transition from memorization to generalization, is characterized by the emergence of low-dimensional representations, yet the mechanism underlying this organization remains elusive. We propose that intrinsic task symmetries primarily drive grokking and shape the geometry of the model's representation space. We identify a consistent three-stage training dynamic underlying grokking: (i) memorization, (ii) symmetry acquisition, and (iii) geometric organization. We show that generalization emerges during the symmetry acquisition phase, after which representations reorganize into a structured, task-aligned geometry. We validate this symmetry-driven account across diverse algorithmic domains, including algebraic, structural, and relational reasoning tasks. Building on these findings, we introduce a symmetry-based diagnostic that anticipates the onset of generalization and propose strategies to accelerate it. Together, our results establish intrinsic symmetry as the key factor enabling neural networks to move beyond memorization and achieve robust algorithmic reasoning. 

---
# AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations 

**Authors**: Cheng Jiayang, Dongyu Ru, Lin Qiu, Yiyang Li, Xuezhi Cao, Yangqiu Song, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.01966)  

**Abstract**: Long-horizon interactions between users and LLM-based assistants necessitate effective memory management, yet current approaches face challenges in training and evaluation of memory. Existing memory benchmarks rely on static, off-policy data as context, limiting evaluation reliability and scalability. To address these gaps, we introduce AMemGym, an interactive environment enabling on-policy evaluation and optimization for memory-driven personalization. AMemGym employs structured data sampling to predefine user profiles, state-dependent questions, and state evolution trajectories, enabling cost-effective generation of high-quality, evaluation-aligned interactions. LLM-simulated users expose latent states through role-play while maintaining structured state consistency. Comprehensive metrics based on structured data guide both assessment and optimization of assistants. Extensive experiments reveal performance gaps in existing memory systems (e.g., RAG, long-context LLMs, and agentic memory) and corresponding reasons. AMemGym not only enables effective selection among competing approaches but also can potentially drive the self-evolution of memory management strategies. By bridging structured state evolution with free-form interactions, our framework provides a scalable, diagnostically rich environment for advancing memory capabilities in conversational agents. 

---
# TiledAttention: a CUDA Tile SDPA Kernel for PyTorch 

**Authors**: Taimur Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01960)  

**Abstract**: TiledAttention is a scaled dot-product attention (SDPA) forward operator for SDPA research on NVIDIA GPUs. Implemented in cuTile Python (TileIR) and exposed as a PyTorch-callable function, it is easier to modify than low-level CUDA templates while retaining realistic behavior via online softmax and tiled $K,V$ streaming. The approach is both performant and directly editable at the schedule level from Python (tile shapes, staging, shared-memory layout), enabling rapid, reproducible kernel research without template-heavy CUDA/CUTLASS rewrites. We benchmark TiledAttention on an NVIDIA DGX GB10 node with a reproducible harness and compare against PyTorch SDPA (auto-dispatch) and explicit unfused baselines across sequence length, head dimension, and precision (FP16/BF16). While production fused baselines remain stronger overall, TiledAttention delivers large speedups over standard eager attention paths and is available for direct use within PyTorch workflows, providing a practical balance between performance and customizability. 

---
# Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy 

**Authors**: Pengyuan Wu, Pingrui Zhang, Zhigang Wang, Dong Wang, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.01953)  

**Abstract**: Diffusion-based policies have achieved remarkable results in robotic manipulation but often struggle to adapt rapidly in dynamic scenarios, leading to delayed responses or task failures. We present DCDP, a Dynamic Closed-Loop Diffusion Policy framework that integrates chunk-based action generation with real-time correction. DCDP integrates a self-supervised dynamic feature encoder, cross-attention fusion, and an asymmetric action encoder-decoder to inject environmental dynamics before action execution, achieving real-time closed-loop action correction and enhancing the system's adaptability in dynamic scenarios. In dynamic PushT simulations, DCDP improves adaptability by 19\% without retraining while requiring only 5\% additional computation. Its modular design enables plug-and-play integration, achieving both temporal coherence and real-time responsiveness in dynamic robotic scenarios, including real-world manipulation tasks. The project page is at: this https URL 

---
# Probabilistic Retrofitting of Learned Simulators 

**Authors**: Cristiana Diaconu, Miles Cranmer, Richard E. Turner, Tanya Marwah, Payel Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2603.01949)  

**Abstract**: Dominant approaches for modelling Partial Differential Equations (PDEs) rely on deterministic predictions, yet many physical systems of interest are inherently chaotic and uncertain. While training probabilistic models from scratch is possible, it is computationally expensive and fails to leverage the significant resources already invested in high-performing deterministic backbones. In this work, we adopt a training-efficient strategy to transform pre-trained deterministic models into probabilistic ones via retrofitting with a proper scoring rule: the Continuous Ranked Probability Score (CRPS). Crucially, this approach is architecture-agnostic: it applies the same adaptation mechanism across distinct model backbones with minimal code modifications. The method proves highly effective across different scales of pre-training: for models trained on single dynamical systems, we achieve 20-54% reductions in rollout CRPS and up to 30% improvements in variance-normalised RMSE (VRMSE) relative to compute-matched deterministic fine-tuning. We further validate our approach on a PDE foundation model, trained on multiple systems and retrofitted on the dataset of interest, to show that our probabilistic adaptation yields an improvement of up to 40% in CRPS and up to 15% in VRMSE compared to deterministic fine-tuning. Validated across diverse architectures and dynamics, our results show that probabilistic PDE modelling need not require retraining from scratch, but can be unlocked from existing deterministic backbones with modest additional training cost. 

---
# physfusion: A Transformer-based Dual-Stream Radar and Vision Fusion Framework for Open Water Surface Object Detection 

**Authors**: Yuting Wan, Liguo Sun, Jiuwu Hao, Zao Zhang, Pin LV  

**Link**: [PDF](https://arxiv.org/pdf/2603.01947)  

**Abstract**: Detecting water-surface targets for Unmanned Surface Vehicles (USVs) is challenging due to wave clutter,
specular reflections, and weak appearance cues in long-range observations. Although 4D millimeter-wave
radar complements cameras under degraded illumination, maritime radar point clouds are sparse and
intermittent, with reflectivity attributes exhibiting heavy-tailed variations under scattering and
multipath, making conventional fusion designs struggle to exploit radar cues effectively.
We propose PhysFusion, a physics-informed radar-image detection framework for water-surface perception.
The framework integrates: (1) a Physics-Informed Radar Encoder (PIR Encoder) with an RCS Mapper and
Quality Gate, transforming per-point radar attributes into compact scattering priors and predicting
point-wise reliability for robust feature learning under clutter; (2) a Radar-guided Interactive Fusion
Module (RIFM) performing query-level radar-image fusion between semantically enriched radar features and
multi-scale visual features, with the radar branch modeled by a dual-stream backbone including a
point-based local stream and a transformer-based global stream using Scattering-Aware Self-Attention
(SASA); and (3) a Temporal Query Aggregation module (TQA) aggregating frame-wise fused queries over a
short temporal window for temporally consistent representations.
Experiments on WaterScenes and FLOW demonstrate that PhysFusion achieves 59.7% mAP50:95 and 90.3% mAP50
on WaterScenes (T=5 radar history) using 5.6M parameters and 12.5G FLOPs, and reaches 94.8% mAP50 and
46.2% mAP50:95 on FLOW under radar+camera setting. Ablation studies quantify the contributions of PIR
Encoder, SASA-based global reasoning, and RIFM. 

---
# When Numbers Tell Half the Story: Human-Metric Alignment in Topic Model Evaluation 

**Authors**: Thibault Prouteau, Francis Lareau, Nicolas Dugué, Jean-Charles Lamirel, Christophe Malaterre  

**Link**: [PDF](https://arxiv.org/pdf/2603.01945)  

**Abstract**: Topic models uncover latent thematic structures in text corpora, yet evaluating their quality remains challenging, particularly in specialized domains. Existing methods often rely on automated metrics like topic coherence and diversity, which may not fully align with human judgment. Human evaluation tasks, such as word intrusion, provide valuable insights but are costly and primarily validated on general-domain corpora. This paper introduces Topic Word Mixing (TWM), a novel human evaluation task assessing inter-topic distinctness by testing whether annotators can distinguish between word sets from single or mixed topics. TWM complements word intrusion's focus on intra-topic coherence and provides a human-grounded counterpart to diversity metrics. We evaluate six topic models - both statistical and embedding-based (LDA, NMF, Top2Vec, BERTopic, CFMF, CFMF-emb) - comparing automated metrics with human evaluation methods based on nearly 4,000 annotations from a domain-specific corpus of philosophy of science publications. Our findings reveal that word intrusion and coherence metrics do not always align, particularly in specialized domains, and that TWM captures human-perceived distinctness while appearing to align with diversity metrics. We release the annotated dataset and task generation code. This work highlights the need for evaluation frameworks bridging automated and human assessments, particularly for domain-specific corpora. 

---
# Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots 

**Authors**: Huw Day, Adrianna Jezierska, Jessica Woodgate  

**Link**: [PDF](https://arxiv.org/pdf/2603.01942)  

**Abstract**: Large Language Models have intensified the scale and strategic manipulation of political discourse on social media, leading to conflict escalation. The existing literature largely focuses on platform-led moderation as a countermeasure. In this paper, we propose a user-centric view of "jailbreaking" as an emergent, non-violent de-escalation practice. Online users engage with suspected LLM-powered accounts to circumvent large language model safeguards, exposing automated behaviour and disrupting the circulation of misleading narratives. 

---
# Explanation-Guided Adversarial Training for Robust and Interpretable Models 

**Authors**: Chao Chen, Yanhui Chen, Shanshan Lin, Dongsheng Hong, Shu Wu, Xiangwen Liao, Chuanyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01938)  

**Abstract**: Deep neural networks (DNNs) have achieved remarkable performance in many tasks, yet they often behave as opaque black boxes. Explanation-guided learning (EGL) methods steer DNNs using human-provided explanations or supervision on model attributions. These approaches improve interpretability but typically assume benign inputs and incur heavy annotation costs. In contrast, both predictions and saliency maps of DNNs could dramatically alter facing imperceptible perturbations or unseen patterns. Adversarial training (AT) can substantially improve robustness, but it does not guarantee that model decisions rely on semantically meaningful features. In response, we propose Explanation-Guided Adversarial Training (EGAT), a unified framework that integrates the strength of AT and EGL to simultaneously improve prediction performance, robustness, and explanation quality. EGAT generates adversarial examples on the fly while imposing explanation-based constraints on the model. By jointly optimizing classification performance, adversarial robustness, and attributional stability, EGAT is not only more resistant to unexpected cases, including adversarial attacks and out-of-distribution (OOD) scenarios, but also offer human-interpretable justifications for the decisions. We further formalize EGAT within the Probably Approximately Correct learning framework, demonstrating theoretically that it yields more stable predictions under unexpected situations compared to standard AT. Empirical evaluations on OOD benchmark datasets show that EGAT consistently outperforms competitive baselines in both clean accuracy and adversarial accuracy +37% while producing more semantically meaningful explanations, and requiring only a limited increase +16% in training time. 

---
# Dream2Learn: Structured Generative Dreaming for Continual Learning 

**Authors**: Salvatore Calcagno, Matteo Pennisi, Federica Proietto Salanitri, Amelia Sorrenti, Simone Palazzo, Concetto Spampinato, Giovanni Bellitto  

**Link**: [PDF](https://arxiv.org/pdf/2603.01935)  

**Abstract**: Continual learning requires balancing plasticity and stability while mitigating catastrophic forgetting. Inspired by human dreaming as a mechanism for internal simulation and knowledge restructuring, we introduce Dream2Learn (D2L), a framework in which a model autonomously generates structured synthetic experiences from its own internal representations and uses them for self-improvement. Rather than reconstructing past data as in generative replay, D2L enables a classifier to create novel, semantically distinct dreamed classes that are coherent with its learned knowledge yet do not correspond to previously observed data. These dreamed samples are produced by conditioning a frozen diffusion model through soft prompt optimization driven by the classifier itself. The generated data are not used to replace memory, but to expand and reorganize the representation space, effectively allowing the network to self-train on internally synthesized concepts. By integrating dreamed classes into continual training, D2L proactively structures latent features to support forward knowledge transfer and adaptation to future tasks. This prospective self-training mechanism mirrors the role of sleep in consolidating and reorganizing memory, turning internal simulations into a tool for improved generalization. Experiments on Mini-ImageNet, FG-ImageNet, and ImageNet-R demonstrate that D2L consistently outperforms strong rehearsal-based baselines and achieves positive forward transfer, confirming its ability to enhance adaptability through internally generated training signals. 

---
# From Variance to Invariance: Qualitative Content Analysis for Narrative Graph Annotation 

**Authors**: Junbo Huang, Max Weinig, Ulrich Fritsche, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2603.01930)  

**Abstract**: Narratives in news discourse play a critical role in shaping public understanding of economic events, such as inflation. Annotating and evaluating these narratives in a structured manner remains a key challenge for Natural Language Processing (NLP). In this work, we introduce a narrative graph annotation framework that integrates principles from qualitative content analysis (QCA) to prioritize annotation quality by reducing annotation errors. We present a dataset of inflation narratives annotated as directed acyclic graphs (DAGs), where nodes represent events and edges encode causal relations. To evaluate annotation quality, we employed a $6\times3$ factorial experimental design to examine the effects of narrative representation (six levels) and distance metric type (three levels) on inter-annotator agreement (Krippendorrf's $\alpha$), capturing the presence of human label variation (HLV) in narrative interpretations. Our analysis shows that (1) lenient metrics (overlap-based distance) overestimate reliability, and (2) locally-constrained representations (e.g., one-hop neighbors) reduce annotation variability. Our annotation and implementation of graph-based Krippendorrf's $\alpha$ are open-sourced. The annotation framework and evaluation results provide practical guidance for NLP research on graph-based narrative annotation under HLV. 

---
# Real Money, Fake Models: Deceptive Model Claims in Shadow APIs 

**Authors**: Yage Zhang, Yukun Jiang, Zeyuan Chen, Michael Backes, Xinyue Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01919)  

**Abstract**: Access to frontier large language models (LLMs), such as GPT-5 and Gemini-2.5, is often hindered by high pricing, payment barriers, and regional restrictions. These limitations drive the proliferation of $\textit{shadow APIs}$, third-party services that claim to provide access to official model services without regional limitations via indirect access. Despite their widespread use, it remains unclear whether shadow APIs deliver outputs consistent with those of the official APIs, raising concerns about the reliability of downstream applications and the validity of research findings that depend on them. In this paper, we present the first systematic audit between official LLM APIs and corresponding shadow APIs. We first identify 17 shadow APIs that have been utilized in 187 academic papers, with the most popular one reaching 5,966 citations and 58,639 GitHub stars by December 6, 2025. Through multidimensional auditing of three representative shadow APIs across utility, safety, and model verification, we uncover both indirect and direct evidence of deception practices in shadow APIs. Specifically, we reveal performance divergence reaching up to $47.21\%$, significant unpredictability in safety behaviors, and identity verification failures in $45.83\%$ of fingerprint tests. These deceptive practices critically undermine the reproducibility and validity of scientific research, harm the interests of shadow API users, and damage the reputation of official model providers. 

---
# Demonstrating ViviDoc: Generating Interactive Documents through Human-Agent Collaboration 

**Authors**: Yinghao Tang, Yupeng Xie, Yingchaojie Feng, Tingfeng Lan, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.01912)  

**Abstract**: Interactive articles help readers engage with complex ideas through exploration, yet creating them remains costly, requiring both domain expertise and web development skills. Recent LLM-based agents can automate content creation, but naively applying them yields uncontrollable and unverifiable outputs. We present ViviDoc, a human-agent collaborative system that generates interactive educational documents from a single topic input. ViviDoc introduces a multi-agent pipeline (Planner, Executor, Evaluator) and the Document Specification (DocSpec), a human-readable intermediate representation that decomposes each interactive visualization into State, Render, Transition, and Constraint components. The DocSpec enables educators to review and refine generation plans before code is produced, bridging the gap between pedagogical intent and executable output. Expert evaluation and a user study show that ViviDoc substantially outperforms naive agentic generation and provides an intuitive editing experience. Our project homepage is available at this https URL. 

---
# FLANS at SemEval-2026 Task 7: RAG with Open-Sourced Smaller LLMs for Everyday Knowledge Across Diverse Languages and Cultures 

**Authors**: Liliia Bogdanova, Shiran Sun, Lifeng Han, Natalia Amat Lefort, Flor Miriam Plaza-del-Arco  

**Link**: [PDF](https://arxiv.org/pdf/2603.01910)  

**Abstract**: This system paper describes our participation in the SemEval-2025 Task-7 ``Everyday Knowledge Across Diverse Languages and Cultures''. We attended two subtasks, i.e., Track 1: Short Answer Questions (SAQ), and Track 2: Multiple-Choice Questions (MCQ). The methods we used are retrieval augmented generation (RAGs) with open-sourced smaller LLMs (OS-sLLMs). To better adapt to this shared task, we created our own culturally aware knowledge base (CulKBs) by extracting Wikipedia content using keyword lists we prepared. We extracted both culturally-aware wiki-text and country-specific wiki-summary. In addition to the local CulKBs, we also have one system integrating live online search output via DuckDuckGo. Towards better privacy and sustainability, we aimed to deploy smaller LLMs (sLLMs) that are open-sourced on the Ollama platform. We share the prompts we developed using refinement techniques and report the learning curve of such prompts. The tested languages are English, Spanish, and Chinese for both tracks. Our resources and codes are shared via this https URL 

---
# Agentic Code Reasoning 

**Authors**: Shubham Ugare, Satish Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2603.01896)  

**Abstract**: Can LLM agents explore codebases and reason about code semantics without executing the code? We study this capability, which we call agentic code reasoning, and introduce semi-formal reasoning: a structured prompting methodology that requires agents to construct explicit premises, trace execution paths, and derive formal conclusions. Unlike unstructured chain-of-thought, semi-formal reasoning acts as a certificate: the agent cannot skip cases or make unsupported claims. We evaluate across three tasks (patch equivalence verification, fault localization, and code question answering) and show that semi-formal reasoning consistently improves accuracy on all of them. For patch equivalence, accuracy improves from 78% to 88% on curated examples and reaches 93% on real-world agent-generated patches, approaching the reliability needed for execution-free RL reward signals. For code question answering on RubberDuckBench Mohammad et al. (2026), semi-formal reasoning achieves 87% accuracy. For fault localization on Defects4J Just et al. (2014), semi-formal reasoning improves Top-5 accuracy by 5 percentage points over standard reasoning. These results demonstrate that structured agentic reasoning enables meaningful semantic code analysis without execution, opening practical applications in RL training pipelines, code review, and static program analysis. 

---
# Diagnosing Generalization Failures from Representational Geometry Markers 

**Authors**: Chi-Ning Chou, Artem Kirsanov, Yao-Yuan Yang, SueYeon Chung  

**Link**: [PDF](https://arxiv.org/pdf/2603.01879)  

**Abstract**: Generalization, the ability to perform well beyond the training context, is a hallmark of biological and artificial intelligence, yet anticipating unseen failures remains a central challenge. Conventional approaches often take a ``bottom-up'' mechanistic route by reverse-engineering interpretable features or circuits to build explanatory models. While insightful, these methods often struggle to provide the high-level, predictive signals for anticipating failure in real-world deployment. Here, we propose using a ``top-down'' approach to studying generalization failures inspired by medical biomarkers: identifying system-level measurements that serve as robust indicators of a model's future performance. Rather than mapping out detailed internal mechanisms, we systematically design and test network markers to probe structure, function links, identify prognostic indicators, and validate predictions in real-world settings. In image classification, we find that task-relevant geometric properties of in-distribution (ID) object manifolds consistently forecast poor out-of-distribution (OOD) generalization. In particular, reductions in two geometric measures, effective manifold dimensionality and utility, predict weaker OOD performance across diverse architectures, optimizers, and datasets. We apply this finding to transfer learning with ImageNet-pretrained models. We consistently find that the same geometric patterns predict OOD transfer performance more reliably than ID accuracy. This work demonstrates that representational geometry can expose hidden vulnerabilities, offering more robust guidance for model selection and AI interpretability. 

---
# KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models 

**Authors**: Songming Zhang, Xue Zhang, Tong Zhang, Bojie Hu, Yufeng Chen, Jinan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01875)  

**Abstract**: Knowledge distillation (KD) is an essential technique to compress large language models (LLMs) into smaller ones. However, despite the distinct roles of the student model and the teacher model in KD, most existing frameworks still use a homogeneous training backend (e.g., FSDP and DeepSpeed) for both models, leading to suboptimal training efficiency. In this paper, we present a novel framework for LLM distillation, termed \textbf{KDFlow}, which features a decoupled architecture and employs SGLang for teacher inference. By bridging the training efficiency of FSDP2 and the inference efficiency of SGLang, KDFlow achieves full utilization of both advantages in a unified system. Moreover, instead of transferring full logits across different processes, our framework only transmits the teacher's hidden states using zero-copy data transfer and recomputes the logits on the student side, effectively balancing the communication cost and KD performance. Furthermore, our framework supports both off-policy and on-policy distillation and incorporates KD algorithms for cross-tokenizer KD through highly extensible and user-friendly APIs. Experiments show that KDFlow can achieve \textbf{1.44$\times$ to 6.36$\times$} speedup compared to current KD frameworks, enabling researchers to rapidly prototype and scale LLM distillation with minimal engineering overhead. Code is available at: this https URL 

---
# Phishing the Phishers with SpecularNet: Hierarchical Graph Autoencoding for Reference-Free Web Phishing Detection 

**Authors**: Tailai Song, Pedro Casas, Michela Meo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01874)  

**Abstract**: Phishing remains the most pervasive threat to the Web, enabling large-scale credential theft and financial fraud through deceptive webpages. While recent reference-based and generative-AI-driven phishing detectors achieve strong accuracy, their reliance on external knowledge bases, cloud services, and complex multimodal pipelines fundamentally limits practicality, scalability, and reproducibility. In contrast, conventional deep learning approaches often fail to generalize to evolving phishing campaigns. We introduce SpecularNet, a novel lightweight framework for reference-free web phishing detection that demonstrates how carefully designed compact architectures can rival heavyweight systems. SpecularNet operates solely on the domain name and HTML structure, modeling the Document Object Model (DOM) as a tree and leveraging a hierarchical graph autoencoding architecture with directional, level-wise message passing. This design captures higher-order structural invariants of phishing webpages while enabling fast, end-to-end inference on standard CPUs. Extensive evaluation against 13 state of the art phishing detectors, including leading reference-based systems, shows that SpecularNet achieves competitive detection performance with dramatically lower computational cost. On benchmark datasets, it reaches an F1 score of 93.9%, trailing the best reference-based method slightly while reducing inference time from several seconds to approximately 20 milliseconds per webpage. Field and robustness evaluations further validate SpecularNet in real-world deployments, on a newly collected 2026 open-world dataset, and against adversarial attacks. 

---
# Tide: A Customisable Dataset Generator for Anti-Money Laundering Research 

**Authors**: Montijn van den Beukel, Jože Martin Rožanec, Ana-Lucia Varbanescu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01863)  

**Abstract**: The lack of accessible transactional data significantly hinders machine learning research for Anti-Money Laundering (AML). Privacy and legal concerns prevent the sharing of real financial data, while existing synthetic generators focus on simplistic structural patterns and neglect the temporal dynamics (timing and frequency) that characterise sophisticated laundering schemes.
We present Tide, an open-source synthetic dataset generator that produces graph-based financial networks incorporating money laundering patterns defined by both structural and temporal characteristics. Tide enables reproducible, customisable dataset generation tailored to specific research needs. We release two reference datasets with varying illicit ratios (LI: 0.10\%, HI: 0.19\%), alongside the implementation of state-of-the-art detection models.
Evaluation across these datasets reveals condition-dependent model rankings: LightGBM achieves the highest PR-AUC (78.05) in the low illicit ratio condition, while XGBoost performs best (85.12) at higher fraud prevalence. These divergent rankings demonstrate that the reference datasets can meaningfully differentiate model capabilities across operational conditions.
Tide provides the research community with a configurable benchmark that exposes meaningful performance variation across model architectures, advancing the development of robust AML detection methods. 

---
# Non-verbal Real-time Human-AI Interaction in Constrained Robotic Environments 

**Authors**: Dragos Costea, Alina Marcu, Cristina Lazar, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01804)  

**Abstract**: We study the ongoing debate regarding the statistical fidelity of AI-generated data compared to human-generated data in the context of non-verbal communication using full body motion. Concretely, we ask if contemporary generative models move beyond surface mimicry to participate in the silent, but expressive dialogue of body language. We tackle this question by introducing the first framework that generates a natural non-verbal interaction between Human and AI in real-time from 2D body keypoints. Our experiments utilize four lightweight architectures which run at up to 100 FPS on an NVIDIA Orin Nano, effectively closing the perception-action loop needed for natural Human-AI interaction. We trained on 437 human video clips and demonstrated that pretraining on synthetically-generated sequences reduces motion errors significantly, without sacrificing speed. Yet, a measurable reality gap persists. When the best model is evaluated on keypoints extracted from cutting-edge text-to-video systems, such as SORA and VEO, we observe that performance drops on SORA-generated clips. However, it degrades far less on VEO, suggesting that temporal coherence, not image fidelity, drives real-world performance. Our results demonstrate that statistically distinguishable differences persist between Human and AI motion. 

---
# Phase-Type Variational Autoencoders for Heavy-Tailed Data 

**Authors**: Abdelhakim Ziani, András Horváth, Paolo Ballarini  

**Link**: [PDF](https://arxiv.org/pdf/2603.01800)  

**Abstract**: Heavy-tailed distributions are ubiquitous in real-world data, where rare but extreme events dominate risk and variability. However, standard Variational Autoencoders (VAEs) employ simple decoder distributions (e.g., Gaussian) that fail to capture heavy-tailed behavior, while existing heavy-tail-aware extensions remain restricted to predefined parametric families whose tail behavior is fixed a priori.
We propose the Phase-Type Variational Autoencoder (PH-VAE), whose decoder distribution is a latent-conditioned Phase-Type (PH) distribution defined as the absorption time of a continuous-time Markov chain (CTMC). This formulation composes multiple exponential time scales, yielding a flexible and analytically tractable decoder that adapts its tail behavior directly from the observed data. Experiments on synthetic and real-world benchmarks demonstrate that PH-VAE accurately recovers diverse heavy-tailed distributions, significantly outperforming Gaussian, Student-t, and extreme-value-based VAE decoders in modeling tail behavior and extreme quantiles. In multivariate settings, PH-VAE captures realistic cross-dimensional tail dependence through its shared latent representation. To our knowledge, this is the first work to integrate Phase-Type distributions into deep generative modeling, bridging applied probability and representation learning. 

---
# PleaSQLarify: Visual Pragmatic Repair for Natural Language Database Querying 

**Authors**: Robin Shing Moon Chan, Rita Sevastjanova, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2603.01795)  

**Abstract**: Natural language database interfaces broaden data access, yet they remain brittle under input ambiguity. Standard approaches often collapse uncertainty into a single query, offering little support for mismatches between user intent and system interpretation. We reframe this challenge through pragmatic inference: while users economize expressions, systems operate on priors over the action space that may not align with the users'. In this view, pragmatic repair -- incremental clarification through minimal interaction -- is a natural strategy for resolving underspecification. We present \textsc{PleaSQLarify}, which operationalizes pragmatic repair by structuring interaction around interpretable decision variables that enable efficient clarification. A visual interface complements this by surfacing the action space for exploration, requesting user disambiguation, and making belief updates traceable across turns. In a study with twelve participants, \textsc{PleaSQLarify} helped users recognize alternative interpretations and efficiently resolve ambiguity. Our findings highlight pragmatic repair as a design principle that fosters effective user control in natural language interfaces. 

---
# ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs 

**Authors**: Xunlei Chen, Jinyu Guo, Yuang Li, Zhaokun Wang, Yi Gong, Jie Zou, Jiwei Wei, Wenhong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.01792)  

**Abstract**: Large language models (LLMs) have advanced to encompass extensive knowledge across diverse domains. Yet controlling what a LLMs should not know is important for ensuring alignment and thus safe use. However, effective unlearning in LLMs is difficult due to the fuzzy boundary between knowledge retention and forgetting. This challenge is exacerbated by entangled parameter spaces from continuous multi-domain training, often resulting in collateral damage, especially under aggressive unlearning strategies. Furthermore, the computational overhead required to optimize State-of-the-Art (SOTA) models with billions of parameters poses an additional barrier. In this work, we present ALTER, a lightweight unlearning framework for LLMs to address both the challenges of knowledge entanglement and unlearning efficiency. ALTER operates through two phases: (I) high entropy tokens are captured and learned via the shared A matrix in LoRA, followed by (II) an asymmetric LoRA architecture that achieves a specified forgetting objective by parameter isolation and unlearning tokens within the target subdomains. Serving as a new research direction for achieving unlearning via token-level isolation in the asymmetric framework. ALTER achieves SOTA performance on TOFU, WMDP, and MUSE benchmarks with over 95% forget quality and shows minimal side effects through preserving foundational tokens. By decoupling unlearning from LLMs' billion-scale parameters, this framework delivers excellent efficiency while preserving over 90% of model utility, exceeding baseline preservation rates of 47.8-83.6%. 

---
# Learning Shortest Paths with Generative Flow Networks 

**Authors**: Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov  

**Link**: [PDF](https://arxiv.org/pdf/2603.01786)  

**Abstract**: In this paper, we present a novel learning framework for finding shortest paths in graphs utilizing Generative Flow Networks (GFlowNets). First, we examine theoretical properties of GFlowNets in non-acyclic environments in relation to shortest paths. We prove that, if the total flow is minimized, forward and backward policies traverse the environment graph exclusively along shortest paths between the initial and terminal states. Building on this result, we show that the pathfinding problem in an arbitrary graph can be solved by training a non-acyclic GFlowNet with flow regularization. We experimentally demonstrate the performance of our method in pathfinding in permutation environments and in solving Rubik's Cubes. For the latter problem, our approach shows competitive results with state-of-the-art machine learning approaches designed specifically for this task in terms of the solution length, while requiring smaller search budget at test-time. 

---
# Co-Evolutionary Multi-Modal Alignment via Structured Adversarial Evolution 

**Authors**: Guoxin Shi, Haoyu Wang, Zaihui Yang, Yuxing Wang, Yongzhe Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01784)  

**Abstract**: Adversarial behavior plays a central role in aligning large language models with human values. However, existing alignment methods largely rely on static adversarial settings, which fundamentally limit robustness, particularly in multimodal settings with a larger attack surface. In this work, we move beyond static adversarial supervision and introduce co-evolutionary alignment with evolving attacks, instantiated by CEMMA (Co-Evolutionary Multi-Modal Alignment), an automated and adaptive framework for multimodal safety alignment. We introduce an Evolutionary Attacker that decomposes adversarial prompts into method templates and harmful intents. By employing genetic operators, including mutation, crossover, and differential evolution, it enables simple seed attacks to inherit the structural efficacy of sophisticated jailbreaks. The Adaptive Defender is iteratively updated on the synthesized hard negatives, forming a closed-loop process that adapts alignment to evolving attacks. Experiments show that the Evolutionary Attacker substantially increases red-teaming jailbreak attack success rate (ASR), while the Adaptive Defender improves robustness and generalization across benchmarks with higher data efficiency, without inducing excessive benign refusal, and remains compatible with inference-time defenses such as AdaShield. 

---
# FreeAct: Freeing Activations for LLM Quantization 

**Authors**: Xiaohao Liu, Xiaobo Xia, Manyi Zhang, Ji-Fu Li, Xianzhi Yu, Fei Shen, Xiu Su, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2603.01776)  

**Abstract**: Quantization is pivotal for mitigating the significant memory and computational overhead of Large Language Models (LLMs). While emerging transformation-based methods have successfully enhanced quantization by projecting feature spaces onto smoother manifolds using orthogonal matrices, they typically enforce a rigid one-to-one transformation constraint. This static approach fails to account for the dynamic patterns inherent in input activations, particularly within diffusion LLMs (dLLMs) and Multimodal LLMs (MLLMs), where varying token types exhibit distinct distributions. To advance this, we propose FreeAct, a novel quantization framework that relaxes the static one-to-one constraint to accommodate dynamic activation disparities. Theoretically, we leverage the rank-deficient nature of activations to derive a solution space that extends beyond simple inverse matrices, enabling the decoupling of activation transformations from weights. Methodologically, FreeAct identifies token-specific dynamics (i.e., vision v.s. text, or masked tokens) and allocates distinct transformation matrices to the activation side, while maintaining a unified, static transformation for the weights. Extensive experiments across dLLMs and MLLMs demonstrate that FreeAct significantly outperforms baselines, up to 5.3% performance improvement, with in-depth analyses. Our code will be publicly released. 

---
# Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport 

**Authors**: Harry Amad, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2603.01771)  

**Abstract**: Neural networks (NNs) often have critical behavioural trade-offs that are set at design time with hyperparameters-such as reward weights in reinforcement learning or quantile targets in regression. Post-deployment, however, user preferences can evolve, making initial settings undesirable, necessitating potentially expensive retraining. To circumvent this, we introduce the task of Hyperparameter Trajectory Inference (HTI): to learn, from observed data, how a NN's conditional output distribution changes with its hyperparameters, and construct a surrogate model that approximates the NN at unobserved hyperparameter settings. HTI requires extending existing trajectory inference approaches to incorporate conditions, exacerbating the challenge of ensuring inferred paths are feasible. We propose an approach based on conditional Lagrangian optimal transport, jointly learning the Lagrangian function governing hyperparameter-induced dynamics along with the associated optimal transport maps and geodesics between observed marginals, which form the surrogate model. We incorporate inductive biases based on the manifold hypothesis and least-action principles into the learned Lagrangian, improving surrogate model feasibility. We empirically demonstrate that our approach reconstructs NN outputs across various hyperparameter spectra better than other alternatives. 

---
# CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning 

**Authors**: Pratik Jawahar, Maurizio Pierini  

**Link**: [PDF](https://arxiv.org/pdf/2603.01768)  

**Abstract**: Current deep learning primitives dealing with temporal dynamics suffer from a fundamental dichotomy: they are either discrete and unstable (LSTMs) \citep{pascanu_difficulty_2013}, leading to exploding or vanishing gradients; or they are continuous and dissipative (Neural ODEs) \citep{dupont_augmented_2019}, which destroy information over time to ensure stability. We propose the \textbf{Causal Hamiltonian Learning Unit} (pronounced: \textit{clue}), a novel Physics-grounded computational learning primitive. By enforcing a Relativistic Hamiltonian structure and utilizing symplectic integration, a CHLU strictly conserves phase-space volume, as an attempt to solve the memory-stability trade-off. We show that the CHLU is designed for infinite-horizon stability, as well as controllable noise filtering. We then demonstrate a CHLU's generative ability using the MNIST dataset as a proof-of-principle. 

---
# Modular Memory is the Key to Continual Learning Agents 

**Authors**: Vaggelis Dorovatas, Malte Schwerin, Andrew D. Bagdanov, Lucas Caccia, Antonio Carta, Laurent Charlin, Barbara Hammer, Tyler L. Hayes, Timm Hess, Christopher Kanan, Dhireesha Kudithipudi, Xialei Liu, Vincenzo Lomonaco, Jorge Mendez-Mendez, Darshan Patil, Ameya Prabhu, Elisa Ricci, Tinne Tuytelaars, Gido M. van de Ven, Liyuan Wang, Joost van de Weijer, Jonghyun Choi, Martin Mundt, Rahaf Aljundi  

**Link**: [PDF](https://arxiv.org/pdf/2603.01761)  

**Abstract**: Foundation models have transformed machine learning through large-scale pretraining and increased test-time compute. Despite surpassing human performance in several domains, these models remain fundamentally limited in continuous operation, experience accumulation, and personalization, capabilities that are central to adaptive intelligence. While continual learning research has long targeted these goals, its historical focus on in-weight learning (IWL), i.e., updating a single model's parameters to absorb new knowledge, has rendered catastrophic forgetting a persistent challenge. Our position is that combining the strengths of In-Weight Learning (IWL) and the newly emerged capabilities of In-Context Learning (ICL) through the design of modular memory is the missing piece for continual adaptation at scale. We outline a conceptual framework for modular memory-centric architectures that leverage ICL for rapid adaptation and knowledge accumulation, and IWL for stable updates to model capabilities, charting a practical roadmap toward continually learning agents. 

---
# Federated Agentic AI for Wireless Networks: Fundamentals, Approaches, and Applications 

**Authors**: Lingyi Cai, Yu Zhang, Ruichen Zhang, Yinqiu Liu, Tao Jiang, Dusit Niyato, Wei Ni, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2603.01755)  

**Abstract**: Agentic artificial intelligence (AI) presents a promising pathway toward realizing autonomous and self-improving wireless network services. However, resource-constrained, widely distributed, and data-heterogeneous nature of wireless networks poses significant challenges to existing agentic AI that relies on centralized architectures, leading to high communication overhead, privacy risks, and non-independent and identically distributed (non-IID) data. Federated learning (FL) has the potential to improve the overall loop of agentic AI through collaborative local learning and parameter sharing without exchanging raw data. This paper proposes new federated agentic AI approaches for wireless networks. We first summarize fundamentals of agentic AI and mainstream FL types. Then, we illustrate how each FL type can strengthen a specific component of agentic AI's loop. Moreover, we conduct a case study on using FRL to improve the performance of agentic AI's action decision in low-altitude wireless networks (LAWNs). Finally, we provide a conclusion and discuss future research directions. 

---
# Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control 

**Authors**: Peng Yu, Xin Wang, Ning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01751)  

**Abstract**: Continuum robots possess high flexibility and redundancy, making them well suited for safe interaction in complex environments, yet their continuous deformation and nonlinear dynamics pose fundamental challenges to perception, modeling, and control. Existing vision-based control approaches often rely on end-to-end learning, achieving shape regulation without explicit awareness of robot geometry or its interaction with the environment. Here, we introduce a shape-interpretable visual self-modeling framework for continuum robots that enables geometry-aware control. Robot shapes are encoded from multi-view planar images using a Bezier-curve representation, transforming visual observations into a compact and physically meaningful shape space that uniquely characterizes the robot's three-dimensional configuration. Based on this representation, neural ordinary differential equations are employed to self-model both shape and end-effector dynamics directly from data, enabling hybrid shape-position control without analytical models or dense body markers. The explicit geometric structure of the learned shape space allows the robot to reason about its body and surroundings, supporting environment-aware behaviors such as obstacle avoidance and self-motion while maintaining end-effector objectives. Experiments on a cable-driven continuum robot demonstrate accurate shape-position regulation and tracking, with shape errors within 1.56% of image resolution and end-effector errors within 2% of robot length, as well as robust performance in constrained environments. By elevating visual shape representations from two-dimensional observations to an interpretable three-dimensional self-model, this work establishes a principled alternative to vision-based end-to-end control and advances autonomous, geometry-aware manipulation for continuum robots. 

---
# Discrete World Models via Regularization 

**Authors**: Davide Bizzaro, Luciano Serafini  

**Link**: [PDF](https://arxiv.org/pdf/2603.01748)  

**Abstract**: World models aim to capture the states and dynamics of an environment in a compact latent space. Moreover, using Boolean state representations is particularly useful for search heuristics and symbolic reasoning and planning. Existing approaches keep latents informative via decoder-based reconstruction, or instead via contrastive or reward signals. In this work, we introduce Discrete World Models via Regularization (DWMR): a reconstruction-free and contrastive-free method for unsupervised Boolean world-model learning. In particular, we introduce a novel world-modeling loss that couples latent prediction with specialized regularizers. Such regularizers maximize the entropy and independence of the representation bits through variance, correlation, and coskewness penalties, while simultaneously enforcing a locality prior for sparse action changes. To enable effective optimization, we also introduce a novel training scheme improving robustness to discrete roll-outs. Experiments on two benchmarks with underlying combinatorial structure show that DWMR learns more accurate representations and transitions than reconstruction-based alternatives. Finally, DWMR can also be paired with an auxiliary reconstruction decoder, and this combination yields additional gains. 

---
# An Analysis of Multi-Task Architectures for the Hierarchic Multi-Label Problem of Vehicle Model and Make Classification 

**Authors**: Alexandru Manole, Laura Diosan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01746)  

**Abstract**: Most information in our world is organized hierarchically; however, many Deep Learning approaches do not leverage this semantically rich structure. Research suggests that human learning benefits from exploiting the hierarchical structure of information, and intelligent models could similarly take advantage of this through multi-task learning. In this work, we analyze the advantages and limitations of multi-task learning in a hierarchical multi-label classification problem: car make and model classification. Considering both parallel and cascaded multi-task architectures, we evaluate their impact on different Deep Learning classifiers (CNNs, Transformers) while varying key factors such as dropout rate and loss weighting to gain deeper insight into the effectiveness of this approach. The tests are conducted on two established benchmarks: StanfordCars and CompCars. We observe the effectiveness of the multi-task paradigm on both datasets, improving the performance of the investigated CNN in almost all scenarios. Furthermore, the approach yields significant improvements on the CompCars dataset for both types of models. 

---
# Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning 

**Authors**: Naoki Shitanda, Motoki Omura, Tatsuya Harada, Takayuki Osa  

**Link**: [PDF](https://arxiv.org/pdf/2603.01741)  

**Abstract**: Scaling reinforcement learning to tens of thousands of parallel environments requires overcoming the limited exploration capacity of a single policy. Ensemble-based policy gradient methods, which employ multiple policies to collect diverse samples, have recently been proposed to promote exploration. However, merely broadening the exploration space does not always enhance learning capability, since excessive exploration can reduce exploration quality or compromise training stability. In this work, we theoretically analyze the impact of inter-policy diversity on learning efficiency in policy ensembles, and propose Coupled Policy Optimization which regulates diversity through KL constraints between policies. The proposed method enables effective exploration and outperforms strong baselines such as SAPG, PBT, and PPO across multiple tasks, including challenging dexterous manipulation, in terms of both sample efficiency and final performance. Furthermore, analysis of policy diversity and effective sample size during training reveals that follower policies naturally distribute around the leader, demonstrating the emergence of structured and efficient exploratory behavior. Our results indicate that diverse exploration under appropriate regulation is key to achieving stable and sample-efficient learning in ensemble policy gradient methods. Project page at this https URL . 

---
# CA-AFP: Cluster-Aware Adaptive Federated Pruning 

**Authors**: Om Govind Jha, Harsh Shukla, Haroon R. Lone  

**Link**: [PDF](https://arxiv.org/pdf/2603.01739)  

**Abstract**: Federated Learning (FL) faces major challenges in real-world deployments due to statistical heterogeneity across clients and system heterogeneity arising from resource-constrained devices. While clustering-based approaches mitigate statistical heterogeneity and pruning techniques improve memory and communication efficiency, these strategies are typically studied in isolation.
We propose CA-AFP, a unified framework that jointly addresses both challenges by performing cluster-specific model pruning. In CA-AFP, clients are first grouped into clusters, and a separate model for each cluster is adaptively pruned during training. The framework introduces two key innovations: (1) a cluster-aware importance scoring mechanism that combines weight magnitude, intra-cluster coherence, and gradient consistency to identify parameters for pruning, and (2) an iterative pruning schedule that progressively removes parameters while enabling model self-healing through weight regrowth.
We evaluate CA-AFP on two widely used human activity recognition benchmarks, UCI HAR and WISDM, under natural user-based federated partitions. Experimental results demonstrate that CA-AFP achieves a favorable balance between predictive accuracy, inter-client fairness, and communication efficiency. Compared to pruning-based baselines, CA-AFP consistently improves accuracy and lower performance disparity across clients with limited fine-tuning, while requiring substantially less communication than dense clustering-based methods. It also shows robustness to different Non-IID levels of data. Finally, ablation studies analyze the impact of clustering, pruning schedules and scoring mechanism offering practical insights into the design of efficient and adaptive FL systems. 

---
# Solving Inverse PDE Problems using Minimization Methods and AI 

**Authors**: Noura Helwani, Sophie Moufawad, Georges Sakr  

**Link**: [PDF](https://arxiv.org/pdf/2603.01731)  

**Abstract**: Many physical and engineering systems require solving direct problems to predict behavior and inverse problems to determine unknown parameters from measurement. In this work, we study both aspects for systems governed by differential equations, contrasting well-established numerical methods with new AI-based techniques, specifically Physics-Informed Neural Networks (PINNs). We first analyze the logistic differential equation, using its closed-form solution to verify numerical schemes and validate PINN performance. We then address the Porous Medium Equation (PME), a nonlinear partial differential equation with no general closed-form solution, building strong solvers of the direct problem and testing techniques for parameter estimation in the inverse problem. Our results suggest that PINNs can closely estimate solutions at competitive computational cost, and thus propose an effective tool for solving both direct and inverse problems for complex systems. 

---
# Towards Principled Dataset Distillation: A Spectral Distribution Perspective 

**Authors**: Ruixi Wu, Shaobo Wang, Jiahuan Chen, Zhiyuan Liu, Yicun Yang, Zhaorun Chen, Zekai Li, Kaixin Li, Xinming Wang, Hongzhu Yi, Kai Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01698)  

**Abstract**: Dataset distillation (DD) aims to compress large-scale datasets into compact synthetic counterparts for efficient model training. However, existing DD methods exhibit substantial performance degradation on long-tailed datasets. We identify two fundamental challenges: heuristic design choices for distribution discrepancy measure and uniform treatment of imbalanced classes. To address these limitations, we propose Class-Aware Spectral Distribution Matching (CSDM), which reformulates distribution alignment via the spectrum of a well-behaved kernel function. This technique maps the original samples into frequency space, resulting in the Spectral Distribution Distance (SDD). To mitigate class imbalance, we exploit the unified form of SDD to perform amplitude-phase decomposition, which adaptively prioritizes the realism in tail classes. On CIFAR-10-LT, with 10 images per class, CSDM achieves a 14.0% improvement over state-of-the-art DD methods, with only a 5.7% performance drop when the number of images in tail classes decreases from 500 to 25, demonstrating strong stability on long-tailed data. 

---
# DynaMoE: Dynamic Token-Level Expert Activation with Layer-Wise Adaptive Capacity for Mixture-of-Experts Neural Networks 

**Authors**: Gökdeniz Gülmez  

**Link**: [PDF](https://arxiv.org/pdf/2603.01697)  

**Abstract**: Mixture-of-Experts (MoE) architectures have emerged as a powerful paradigm for scaling neural networks while maintaining computational efficiency. However, standard MoE implementations rely on two rigid design assumptions: (1) fixed Top-K routing where exactly K experts are activated per token, and (2) uniform expert allocation across all layers. This paper introduces DynaMoE, a novel MoE framework that relaxes both constraints through dynamic token-level expert activation and layer-wise adaptive capacity allocation. DynaMoE introduces a principled routing mechanism where the number of active experts per token varies based on input complexity. Concurrently, the framework implements six distinct scheduling strategies for distributing expert capacity across network depth, including descending, ascending, pyramid, and wave patterns. We theoretically analyze the expressivity gains of dynamic routing and derive bounds on computational efficiency. Through extensive experiments on MNIST, Fashion-MNIST, CIFAR-10 (image classification), and Recycling-the-Web (language modeling) across multiple model scales, we demonstrate that DynaMoE achieves superior parameter efficiency compared to static baselines. Our key finding is that optimal expert schedules are task- and scale-dependent: descending schedules (concentrating capacity in early layers) outperform uniform baselines on image classification. For language modeling, optimal schedules vary by model size, descending for Tiny, ascending for Small, and uniform for Medium. Furthermore, dynamic routing reduces gradient variance during training, leading to improved convergence stability. DynaMoE establishes a new framework for adaptive computation in neural networks, providing principled guidance for MoE architecture design. 

---
# Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning 

**Authors**: Haonan Jia, Shichao Dong, Xin Dong, Zenghui Sun, Jin Wang, Jinsong Lan, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01696)  

**Abstract**: Large Vision-Language Models (LVLMs) often omit or misrepresent critical visual content in generated image captions. Minimizing such information loss will force LVLMs to focus on image details to generate precise descriptions. However, measuring information loss during modality conversion is inherently challenging due to the modal gap between visual content and text output. In this paper, we argue that the quality of an image caption is positively correlated with the similarity between images retrieved via text search using that caption. Based on this insight, we further propose Cross-modal Identity Mapping (CIM), a reinforcement learning framework that enhances image captioning without requiring additional annotations. Specifically, the method quantitatively evaluates the information loss from two perspectives: Gallery Representation Consistency and Query-gallery Image Relevance. Supervised under these metrics, LVLM minimizes information loss and aims to achieve identity mapping from images to captions. The experimental results demonstrate the superior performance of our method in image captioning, even when compared with Supervised Fine-Tuning. Particularly, on the COCO-LN500 benchmark, CIM achieves a 20% improvement in relation reasoning on this http URL code will be released when the paper is accepted. 

---
# Streaming Continual Learning for Unified Adaptive Intelligence in Dynamic Environments 

**Authors**: Federico Giannini, Giacomo Ziffer, Andrea Cossu, Vincenzo Lomonaco  

**Link**: [PDF](https://arxiv.org/pdf/2603.01695)  

**Abstract**: Developing effective predictive models becomes challenging in dynamic environments that continuously produce data and constantly change. Continual Learning (CL) and Streaming Machine Learning (SML) are two research areas that tackle this arduous task. We put forward a unified setting that harnesses the benefits of both CL and SML: their ability to quickly adapt to non-stationary data streams without forgetting previous knowledge. We refer to this setting as Streaming Continual Learning (SCL). SCL does not replace either CL or SML. Instead, it extends the techniques and approaches considered by both fields. We start by briefly describing CL and SML and unifying the languages of the two frameworks. We then present the key features of SCL. We finally highlight the importance of bridging the two communities to advance the field of intelligent systems. 

---
# MVR: Multi-view Video Reward Shaping for Reinforcement Learning 

**Authors**: Lirui Luo, Guoxi Zhang, Hongming Xu, Yaodong Yang, Cong Fang, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.01694)  

**Abstract**: Reward design is of great importance for solving complex tasks with reinforcement learning. Recent studies have explored using image-text similarity produced by vision-language models (VLMs) to augment rewards of a task with visual feedback. A common practice linearly adds VLM scores to task or success rewards without explicit shaping, potentially altering the optimal policy. Moreover, such approaches, often relying on single static images, struggle with tasks whose desired behavior involves complex, dynamic motions spanning multiple visually different states. Furthermore, single viewpoints can occlude critical aspects of an agent's behavior. To address these issues, this paper presents Multi-View Video Reward Shaping (MVR), a framework that models the relevance of states regarding the target task using videos captured from multiple viewpoints. MVR leverages video-text similarity from a frozen pre-trained VLM to learn a state relevance function that mitigates the bias towards specific static poses inherent in image-based methods. Additionally, we introduce a state-dependent reward shaping formulation that integrates task-specific rewards and VLM-based guidance, automatically reducing the influence of VLM guidance once the desired motion pattern is achieved. We confirm the efficacy of the proposed framework with extensive experiments on challenging humanoid locomotion tasks from HumanoidBench and manipulation tasks from MetaWorld, verifying the design choices through ablation studies. 

---
# Reasoning as Gradient: Scaling MLE Agents Beyond Tree Search 

**Authors**: Yifei Zhang, Xu Yang, Xiao Yang, Bowen Xian, Qizheng Li, Shikai Fang, Jingyuan Li, Jian Wang, Mingrui Xu, Weiqing Liu, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2603.01692)  

**Abstract**: LLM-based agents for machine learning engineering (MLE) predominantly rely on tree search, a form of gradient-free optimization that uses scalar validation scores to rank candidates. As LLM reasoning capabilities improve, exhaustive enumeration becomes increasingly inefficient compared to directed updates, analogous to how accurate gradients enable efficient descent over random search. We introduce \textsc{Gome}, an MLE agent that operationalizes gradient-based optimization. \textsc{Gome} maps structured diagnostic reasoning to gradient computation, success memory to momentum, and multi-trace execution to distributed optimization. Under a closed-world protocol that isolates architectural effects from external knowledge, \textsc{Gome} achieves a state-of-the-art 35.1\% any-medal rate on MLE-Bench with a restricted 12-hour budget on a single V100 GPU. Scaling experiments across 10 models reveal a critical crossover: with weaker models, tree search retains advantages by compensating for unreliable reasoning through exhaustive exploration; as reasoning capability strengthens, gradient-based optimization progressively outperforms, with the gap widening at frontier-tier models. Given the rapid advancement of reasoning-oriented LLMs, this positions gradient-based optimization as an increasingly favorable paradigm. We release our codebase and GPT-5 traces. 

---
# QIME: Constructing Interpretable Medical Text Embeddings via Ontology-Grounded Questions 

**Authors**: Yixuan Tang, Zhenghong Lin, Yandong Sun, Anthony K.H. Tung  

**Link**: [PDF](https://arxiv.org/pdf/2603.01690)  

**Abstract**: While dense biomedical embeddings achieve strong performance, their black-box nature limits their utility in clinical decision-making. Recent question-based interpretable embeddings represent text as binary answers to natural-language questions, but these approaches often rely on heuristic or surface-level contrastive signals and overlook specialized domain knowledge. We propose QIME, an ontology-grounded framework for constructing interpretable medical text embeddings in which each dimension corresponds to a clinically meaningful yes/no question. By conditioning on cluster-specific medical concept signatures, QIME generates semantically atomic questions that capture fine-grained distinctions in biomedical text. Furthermore, QIME supports a training-free embedding construction strategy that eliminates per-question classifier training while further improving performance. Experiments across biomedical semantic similarity, clustering, and retrieval benchmarks show that QIME consistently outperforms prior interpretable embedding methods and substantially narrows the gap to strong black-box biomedical encoders, while providing concise and clinically informative explanations. 

---
# Surgical Post-Training: Cutting Errors, Keeping Knowledge 

**Authors**: Wenye Lin, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.01683)  

**Abstract**: Enhancing the reasoning capabilities of Large Language Models (LLMs) via post-training is often constrained by the trade-off between efficiency and catastrophic forgetting. While prior research emphasizes the role of on-policy data in mitigating forgetting, we uncover--and validate both theoretically and empirically--an overlooked yet critical mechanism: the implicit regularization inherent in Direct Preference Optimization's (DPO) reward estimate. This motivates our Surgical Post-Training (SPoT), a new paradigm designed to optimize reasoning efficiently while preserving learned prior knowledge. SPoT consists of: (1) a data rectification pipeline that employs an Oracle to surgically correct erroneous steps via minimal edits, generating data proximal to the model's distribution; and (2) a reward-based binary cross-entropy objective. Unlike the relative ranking in DPO, this objective treats reasoning correctness as a binary classification problem, enforcing decoupled supervision signals. Empirically, with only 4k rectified math data pairs, SPoT improves Qwen3-8B's accuracy by 6.2% on average across in-domain and OOD tasks, requiring merely 28 minutes of training on 8x H800 GPUs. Code: this https URL 

---
# A Practical Guide to Streaming Continual Learning 

**Authors**: Andrea Cossu, Federico Giannini, Giacomo Ziffer, Alessio Bernardo, Alexander Gepperth, Emanuele Della Valle, Barbara Hammer, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01677)  

**Abstract**: Continual Learning (CL) and Streaming Machine Learning (SML) study the ability of agents to learn from a stream of non-stationary data. Despite sharing some similarities, they address different and complementary challenges. While SML focuses on rapid adaptation after changes (concept drifts), CL aims to retain past knowledge when learning new tasks. After a brief introduction to CL and SML, we discuss Streaming Continual Learning (SCL), an emerging paradigm providing a unifying solution to real-world problems, which may require both SML and CL abilities. We claim that SCL can i) connect the CL and SML communities, motivating their work towards the same goal, and ii) foster the design of hybrid approaches that can quickly adapt to new information (as in SML) without forgetting previous knowledge (as in CL). We conclude the paper with a motivating example and a set of experiments, highlighting the need for SCL by showing how CL and SML alone struggle in achieving rapid adaptation and knowledge retention. 

---
# FreeGNN: Continual Source-Free Graph Neural Network Adaptation for Renewable Energy Forecasting 

**Authors**: Abderaouf Bahi, Amel Ourici, Ibtissem Gasmi, Aida Derrablia, Warda Deghmane, Mohamed Amine Ferrag  

**Link**: [PDF](https://arxiv.org/pdf/2603.01657)  

**Abstract**: Accurate forecasting of renewable energy generation is essential for efficient grid management and sustainable power planning. However, traditional supervised models often require access to labeled data from the target site, which may be unavailable due to privacy, cost, or logistical constraints. In this work, we propose FreeGNN, a Continual Source-Free Graph Domain Adaptation framework that enables adaptive forecasting on unseen renewable energy sites without requiring source data or target labels. Our approach integrates a spatio-temporal Graph Neural Network (GNN) backbone with a teacher--student strategy, a memory replay mechanism to mitigate catastrophic forgetting, graph-based regularization to preserve spatial correlations, and a drift-aware weighting scheme to dynamically adjust adaptation strength during streaming updates. This combination allows the model to continuously adapt to non-stationary environmental conditions while maintaining robustness and stability. We conduct extensive experiments on three real-world datasets: GEFCom2012, Solar PV, and Wind SCADA, encompassing multiple sites, temporal resolutions, and meteorological features. The ablation study confirms that each component memory, graph regularization, drift-aware adaptation, and teacher--student strategy contributes significantly to overall performance. The experiments show that FreeGNN achieves an MAE of 5.237 and an RMSE of 7.123 on the GEFCom dataset, an MAE of 1.107 and an RMSE of 1.512 on the Solar PV dataset, and an MAE of 0.382 and an RMSE of 0.523 on the Wind SCADA dataset. These results demonstrate its ability to achieve accurate and robust forecasts in a source-free, continual learning setting, highlighting its potential for real-world deployment in adaptive renewable energy systems. For reproducibility, implementation details are available at: this https URL. 

---
# LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence 

**Authors**: Anka Chandrahas Tummepalli, Preethu Rose Anish  

**Link**: [PDF](https://arxiv.org/pdf/2603.01651)  

**Abstract**: Understanding and predicting judicial outcomes demands nuanced analysis of legal documents. Traditional approaches treat judgments and proceedings as unstructured text, limiting the effectiveness of large language models (LLMs) in tasks such as summarization, argument generation, and judgment prediction. We propose LexChronos, an agentic framework that iteratively extracts structured event timelines from Supreme Court of India judgments. LexChronos employs a dual-agent architecture: a LoRA-instruct-tuned extraction agent identifies candidate events, while a pre-trained feedback agent scores and refines them through a confidence-driven loop. To address the scarcity of Indian legal event datasets, we construct a synthetic corpus of 2000 samples using reverse-engineering techniques with DeepSeek-R1 and GPT-4, generating gold-standard event annotations. Our pipeline achieves a BERT-based F1 score of 0.8751 against this synthetic ground truth. In downstream evaluations on legal text summarization, GPT-4 preferred structured timelines over unstructured baselines in 75% of cases, demonstrating improved comprehension and reasoning in Indian jurisprudence. This work lays a foundation for future legal AI applications in the Indian context, such as precedent mapping, argument synthesis, and predictive judgment modelling, by harnessing structured representations of legal events. 

---
# DeLo: Dual Decomposed Low-Rank Experts Collaboration for Continual Missing Modality Learning 

**Authors**: Xiwei Liu, Yulong Li, Feilong Tang, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2603.01632)  

**Abstract**: Adapting Large Multimodal Models (LMMs) to real-world scenarios poses the dual challenges of learning from sequential data streams while handling frequent modality incompleteness, a task known as Continual Missing Modality Learning (CMML). However, existing works on CMML have predominantly relied on prompt tuning, a technique that struggles with this task due to cross-task interference between its learnable prompts in their shared embedding space. A naive application of Low-Rank Adaptation (LoRA) with modality-shared module will also suffer modality interference from competing gradients. To this end, we propose DeLo, the first framework to leverage a novel dual-decomposed low-rank expert architecture for CMML. Specifically, this architecture resolves modality interference through decomposed LoRA expert, dynamically composing LoRA update matrix with rank-one factors from disentangled modality-specific factor pools. Embedded within a task-partitioned framework that structurally prevents catastrophic forgetting, this expert system is supported by two key mechanisms: a Cross-Modal Guided Routing strategy to handle incomplete data and a Task-Key Memory for efficient, task-agnostic inference. Extensive experiments on established CMML benchmarks demonstrate that our method significantly outperforms state-of-the-art approaches. This highlights the value of a principled, architecturally-aware LoRA design for real-world multimodal challenges. 

---
# Measuring What VLMs Don't Say: Validation Metrics Hide Clinical Terminology Erasure in Radiology Report Generation 

**Authors**: Aditya Parikh, Aasa Feragen, Sneha Das, Stella Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.01625)  

**Abstract**: Reliable deployment of Vision-Language Models (VLMs) in radiology requires validation metrics that go beyond surface-level text similarity to ensure clinical fidelity and demographic fairness. This paper investigates a critical blind spot in current model evaluation: the use of decoding strategies that lead to high aggregate token-overlap scores despite succumbing to template collapse, in which models generate only repetitive, safe generic text and omit clinical terminology. Unaddressed, this blind spot can lead to metric gaming, where models that perform well on benchmarks prove clinically uninformative. Instead, we advocate for lexical diversity measures to check model generations for clinical specificity. We introduce Clinical Association Displacement (CAD), a vocabulary-level framework that quantifies shifts in demographic-based word associations in generated reports. Weighted Association Erasure (WAE) aggregates these shifts to measure the clinical signal loss across demographic groups. We show that deterministic decoding produces high levels of semantic erasure, while stochastic sampling generates diverse outputs but risks introducing new bias, motivating a fundamental rethink of how "optimal" reporting is defined. 

---
# Assessing Crime Disclosure Patterns in a Large-Scale Cybercrime Forum 

**Authors**: Raphael Hoheisel, Tom Meurs, Jai Wientjes, Marianne Junger, Abhishta Abhishta, Masarah Paquet-Clouston  

**Link**: [PDF](https://arxiv.org/pdf/2603.01624)  

**Abstract**: Cybercrime forums play a central role in the cybercrime ecosystem, serving as hubs for the exchange of illicit goods, services, and knowledge. Previous studies have explored the market and social structures of these forums, but less is known about the behavioral dynamics of users, particularly regarding participants' disclosure of criminal activity. This study provides the first large-scale assessment of crime disclosure patterns in a major cybercrime forum, analysing over 3.5 million posts from nearly 300k users. Using a three-level classification scheme (benign, grey, and crime) and a scalable labelling pipeline powered by large language models (LLMs), we measure the level of crime disclosure present in initial posts, analyse how participants switch between levels, and assess how crime disclosure behavior relates to private communications. Our results show that crime disclosure is relatively normative: one quarter of initial posts include explicit crime-related content, and more than one third of users disclose criminal activity at least once in their initial posts. At the same time, most participants show restraint, with over two-thirds posting only benign or grey content and typically escalating disclosure gradually. Grey initial posts are particularly prominent, indicating that many users avoid overt statements and instead anchor their activity in ambiguous content. The study highlights the value of LLM-based text classification and Markov chain modelling for capturing crime disclosure patterns, offering insights for law enforcement efforts aimed at distinguishing benign, grey, and criminal content in cybercrime forums. 

---
# What Helps -- and What Hurts: Bidirectional Explanations for Vision Transformers 

**Authors**: Qin Su, Tie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01605)  

**Abstract**: Vision Transformers (ViTs) achieve strong performance in visual recognition, yet their decision-making remains difficult to interpret. We propose BiCAM, a bidirectional class activation mapping method that captures both supportive (positive) and suppressive (negative) contributions to model predictions. Unlike prior CAM-based approaches that discard negative signals, BiCAM preserves signed attributions to produce more complete and contrastive explanations. BiCAM further introduces a Positive-to-Negative Ratio (PNR) that summarizes attribution balance and enables lightweight detection of adversarial examples without retraining. Across ImageNet, VOC, and COCO, BiCAM improves localization and faithfulness while remaining computationally efficient. It generalizes to multiple ViT variants, including DeiT and Swin. These results suggest the importance of modeling both supportive and suppressive evidence for interpreting transformer-based vision models. 

---
# YCDa: YCbCr Decoupled Attention for Real-time Realistic Camouflaged Object Detection 

**Authors**: PeiHuang Zheng, Yunlong Zhao, Zheng Cui, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.01602)  

**Abstract**: Human vision exhibits remarkable adaptability in perceiving objects under camouflage. When color cues become unreliable, the visual system instinctively shifts its reliance from chrominance (color) to luminance (brightness and texture), enabling more robust perception in visually confusing environments. Drawing inspiration from this biological mechanism, we propose YCDa, an efficient early-stage feature processing strategy that embeds this "chrominance-luminance decoupling and dynamic attention" principle into modern real-time detectors. Specifically, YCDa separates color and luminance information in the input stage and dynamically allocates attention across channels to amplify discriminative cues while suppressing misleading color noise. The strategy is plug-and-play and can be integrated into existing detectors by simply replacing the first downsampling layer. Extensive experiments on multiple baselines demonstrate that YCDa consistently improves performance with negligible overhead as shown in Fig. Notably, YCDa-YOLO12s achieves a 112% improvement in mAP over the baseline on COD10K-D and sets new state-of-the-art results for real-time camouflaged object detection across COD-D datasets. 

---
# FAST-DIPS: Adjoint-Free Analytic Steps and Hard-Constrained Likelihood Correction for Diffusion-Prior Inverse Problems 

**Authors**: Minwoo Kim, Seunghyeok Shin, Hongki Lim  

**Link**: [PDF](https://arxiv.org/pdf/2603.01591)  

**Abstract**: Training-free diffusion priors enable inverse-problem solvers without retraining, but for nonlinear forward operators data consistency often relies on repeated derivatives or inner optimization/MCMC loops with conservative step sizes, incurring many iterations and denoiser/score evaluations. We propose a training-free solver that replaces these inner loops with a hard measurement-space feasibility constraint (closed-form projection) and an analytic, model-optimal step size, enabling a small, fixed compute budget per noise level. Anchored at the denoiser prediction, the correction is approximated via an adjoint-free, ADMM-style splitting with projection and a few steepest-descent updates, using one VJP and either one JVP or a forward-difference probe, followed by backtracking and decoupled re-annealing. We prove local model optimality and descent under backtracking for the step-size rule, and derive an explicit KL bound for mode-substitution re-annealing under a local Gaussian conditional surrogate. We also develop a latent variant and a one-parameter pixel$\rightarrow$latent hybrid schedule. Experiments achieve competitive PSNR/SSIM/LPIPS with up to 19.5$\times$ speedup, without hand-coded adjoints or inner MCMC. 

---
# SafeSci: Safety Evaluation of Large Language Models in Science Domains and Beyond 

**Authors**: Xiangyang Zhu, Yuan Tian, Qi Jia, Kaiwei Zhang, Zicheng Zhang, Chunyi Li, Kaiyuan Ji, Dongrui Liu, Zijian Chen, Lu Sun, Renrui Zhang, Yan Teng, Jing Shao, Wei Sun, Xia Hu, Yu Qiao, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2603.01589)  

**Abstract**: The success of large language models (LLMs) in scientific domains has heightened safety concerns, prompting numerous benchmarks to evaluate their scientific safety. Existing benchmarks often suffer from limited risk coverage and a reliance on subjective evaluation. To address these problems, we introduce SafeSci, a comprehensive framework for safety evaluation and enhancement in scientific contexts. SafeSci comprises SafeSciBench, a multi-disciplinary benchmark with 0.25M samples, and SafeSciTrain, a large-scale dataset containing 1.5M samples for safety enhancement. SafeSciBench distinguishes between safety knowledge and risk to cover extensive scopes and employs objective metrics such as deterministically answerable questions to mitigate evaluation bias. We evaluate 24 advanced LLMs, revealing critical vulnerabilities in current models. We also observe that LLMs exhibit varying degrees of excessive refusal behaviors on safety-related issues. For safety enhancement, we demonstrate that fine-tuning on SafeSciTrain significantly enhances the safety alignment of models. Finally, we argue that knowledge is a double-edged sword, and determining the safety of a scientific question should depend on specific context, rather than universally categorizing it as safe or unsafe. Our work provides both a diagnostic tool and a practical resource for building safer scientific AI systems. 

---
# SkeleGuide: Explicit Skeleton Reasoning for Context-Aware Human-in-Place Image Synthesis 

**Authors**: Chuqiao Wu, Jin Song, Yiyun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2603.01579)  

**Abstract**: Generating realistic and structurally plausible human images into existing scenes remains a significant challenge for current generative models, which often produce artifacts like distorted limbs and unnatural poses. We attribute this systemic failure to an inability to perform explicit reasoning over human skeletal structure. To address this, we introduce SkeleGuide, a novel framework built upon explicit skeletal reasoning. Through joint training of its reasoning and rendering stages, SkeleGuide learns to produce an internal pose that acts as a strong structural prior, guiding the synthesis towards high structural integrity. For fine-grained user control, we introduce PoseInverter, a module that decodes this internal latent pose into an explicit and editable format. Extensive experiments demonstrate that SkeleGuide significantly outperforms both specialized and general-purpose models in generating high-fidelity, contextually-aware human images. Our work provides compelling evidence that explicitly modeling skeletal structure is a fundamental step towards robust and plausible human image synthesis. 

---
# DualSentinel: A Lightweight Framework for Detecting Targeted Attacks in Black-box LLM via Dual Entropy Lull Pattern 

**Authors**: Xiaoyi Pang, Xuanyi Hao, Pengyu Liu, Qi Luo, Song Guo, Zhibo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01574)  

**Abstract**: Recent intelligent systems integrate powerful Large Language Models (LLMs) through APIs, but their trustworthiness may be critically undermined by targeted attacks like backdoor and prompt injection attacks, which secretly force LLMs to generate specific malicious sequences. Existing defensive approaches for such threats typically rely on high access rights, impose prohibitive costs, and hinder normal inference, rendering them impractical for real-world scenarios. To solve these limitations, we introduce DualSentinel, a lightweight and unified defense framework that can accurately and promptly detect the activation of targeted attacks alongside the LLM generation process. We first identify a characteristic of compromised LLMs, termed Entropy Lull: when a targeted attack successfully hijacks the generation process, the LLM exhibits a distinct period of abnormally low and stable token probability entropy, indicating it is following a fixed path rather than making creative choices. DualSentinel leverages this pattern by developing an innovative dual-check approach. It first employs a magnitude and trend-aware monitoring method to proactively and sensitively flag an entropy lull pattern at runtime. Upon such flagging, it triggers a lightweight yet powerful secondary verification based on task-flipping. An attack is confirmed only if the entropy lull pattern persists across both the original and the flipped task, proving that the LLM's output is coercively controlled. Extensive evaluations show that DualSentinel is both highly effective (superior detection accuracy with near-zero false positives) and remarkably efficient (negligible additional cost), offering a truly practical path toward securing deployed LLMs. The source code can be accessed at this https URL. 

---
# LFPO: Likelihood-Free Policy Optimization for Masked Diffusion Models 

**Authors**: Chenxing Wei, Jiazhen Kang, Hong Wang, Jianqing Zhang, Hao Jiang, Xiaolong Xu, Ningyuan Sun, Ying He, F. Richard Yu, Yao Shu, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01563)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has achieved remarkable success in improving autoregressive models, especially in domains requiring correctness like mathematical reasoning and code generation. However, directly applying such paradigms to Diffusion Large Language Models (dLLMs) is fundamentally hindered by the intractability of exact likelihood computation, which forces existing methods to rely on high-variance approximations. To bridge this gap, we propose Likelihood-Free Policy Optimization (LFPO), a native framework that maps the concept of vector field flow matching to the discrete token space. Specifically, LFPO formulates alignment as geometric velocity rectification, which directly optimizes denoising logits via contrastive updates. This design effectively bypasses the errors inherent in likelihood approximation, yielding the precise gradient estimation. Furthermore, LFPO enforce consistency by predicting final solutions from intermediate steps, effectively straightening the probability flow to enable high-quality generation with significantly fewer iterations. Extensive experiments demonstrate that LFPO not only outperforms state-of-the-art baselines on code and reasoning benchmarks but also accelerates inference by approximately 20% through reduced diffusion steps. 

---
# Extracting Training Dialogue Data from Large Language Model based Task Bots 

**Authors**: Shuo Zhang, Junzhou Zhao, Junji Hou, Pinghui Wang, Chenxu Wang, Jing Tao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01550)  

**Abstract**: Large Language Models (LLMs) have been widely adopted to enhance Task-Oriented Dialogue Systems (TODS) by modeling complex language patterns and delivering contextually appropriate responses. However, this integration introduces significant privacy risks, as LLMs, functioning as soft knowledge bases that compress extensive training data into rich knowledge representations, can inadvertently memorize training dialogue data containing not only identifiable information such as phone numbers but also entire dialogue-level events like complete travel schedules. Despite the critical nature of this privacy concern, how LLM memorization is inherited in developing task bots remains unexplored. In this work, we address this gap through a systematic quantitative study that involves evaluating existing training data extraction attacks, analyzing key characteristics of task-oriented dialogue modeling that render existing methods ineffective, and proposing novel attack techniques tailored for LLM-based TODS that enhance both response sampling and membership inference. Experimental results demonstrate the effectiveness of our proposed data extraction attack. Our method can extract thousands of training labels of dialogue states with best-case precision exceeding 70%. Furthermore, we provide an in-depth analysis of training data memorization in LLM-based TODS by identifying and quantifying key influencing factors and discussing targeted mitigation strategies. 

---
# Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation 

**Authors**: Jisoo Kim, Jungbin Cho, Sanghyeok Chu, Ananya Bal, Jinhyung Kim, Gunhee Lee, Sihaeng Lee, Seung Hwan Kim, Bohyung Han, Hyunmin Lee, Laszlo A. Jeni, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.01549)  

**Abstract**: Humans learn not only how their bodies move, but also how the surrounding world responds to their actions. In contrast, while recent Vision-Language-Action (VLA) models exhibit impressive semantic understanding, they often fail to capture the spatiotemporal dynamics governing physical interaction. In this paper, we introduce Pri4R, a simple yet effective approach that endows VLA models with an implicit understanding of world dynamics by leveraging privileged 4D information during training. Specifically, Pri4R augments VLAs with a lightweight point track head that predicts 3D point tracks. By injecting VLA features into this head to jointly predict future 3D trajectories, the model learns to incorporate evolving scene geometry within its shared representation space, enabling more physically aware context for precise control. Due to its architectural simplicity, Pri4R is compatible with dominant VLA design patterns with minimal changes. During inference, we run the model using the original VLA architecture unchanged; Pri4R adds no extra inputs, outputs, or computational overhead. Across simulation and real-world evaluations, Pri4R significantly improves performance on challenging manipulation tasks, including a +10% gain on LIBERO-Long and a +40% gain on RoboCasa. We further show that 3D point track prediction is an effective supervision target for learning action-world dynamics, and validate our design choices through extensive ablations. 

---
# Retrieval, Refinement, and Ranking for Text-to-Video Generation via Prompt Optimization and Test-Time Scaling 

**Authors**: Zillur Rahman, Alex Sheng, Cristian Meo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01509)  

**Abstract**: While large-scale datasets have driven significant progress in Text-to-Video (T2V) generative models, these models remain highly sensitive to input prompts, demonstrating that prompt design is critical to generation quality. Current methods for improving video output often fall short: they either depend on complex, post-editing models, risking the introduction of artifacts, or require expensive fine-tuning of the core generator, which severely limits both scalability and accessibility. In this work, we introduce 3R, a novel RAG based prompt optimization framework. 3R utilizes the power of current state-of-the-art T2V diffusion model and vision language model. It can be used with any T2V model without any kind of model training. The framework leverages three key strategies: RAG-based modifiers extraction for enriched contextual grounding, diffusion-based Preference Optimization for aligning outputs with human preferences, and temporal frame interpolation for producing temporally consistent visual contents. Together, these components enable more accurate, efficient, and contextually aligned text-to-video generation. Experimental results demonstrate the efficacy of 3R in enhancing the static fidelity and dynamic coherence of generated videos, underscoring the importance of optimizing user prompts. 

---
# The Sentience Readiness Index: Measuring National Preparedness for the Possibility of Artificial Sentience 

**Authors**: Tony Rost  

**Link**: [PDF](https://arxiv.org/pdf/2603.01508)  

**Abstract**: The scientific study of consciousness has begun to generate testable predictions about artificial systems. A landmark collaborative assessment evaluated current AI architectures against six leading theories of consciousness and found that none currently qualifies as a strong candidate, but that future systems might. A precautionary approach to AI sentience, which holds that credible possibility of sentience warrants governance action even without proof, has gained philosophical and institutional traction. Yet existing AI readiness indices, including the Oxford Insights Government AI Readiness Index, the IMF AI Preparedness Index, and the Stanford AI Index, measure economic, technological, and governance preparedness without assessing whether societies are prepared for the possibility that AI systems might warrant moral consideration. This paper introduces the Sentience Readiness Index (SRI), a composite index measuring national-level preparedness across six weighted categories for 31 jurisdictions. The SRI was constructed following the OECD/JRC framework for composite indicators and employs LLM-assisted expert scoring with iterative expert review. No jurisdiction exceeds "Partially Prepared" (the United Kingdom leads at 49/100). Research Environment scores are universally the strongest category; Professional Readiness is universally the weakest. These findings suggest that if AI sentience becomes scientifically plausible, no society currently possesses adequate institutional, professional, or cultural infrastructure to respond. The SRI provides a diagnostic baseline and identifies specific capacity deficits that policy can address. 

---
# GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control 

**Authors**: Haofeng Xu, Junwei Su, Yukun Tian, Lansong Diao, Zhengping Qian, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01501)  

**Abstract**: Asynchronous execution is essential for scaling reinforcement learning (RL) to modern large model workloads, including large language models and AI agents, but it can fundamentally alter RL optimization behavior. While prior work on asynchronous RL focuses on training throughput and distributional correction, we show that naively applying asynchrony to policy-gradient updates can induce qualitatively different training dynamics and lead to severe training instability. Through systematic empirical and theoretical analysis, we identify a key signature of this instability: asynchronous training exhibits persistently high cosine similarity between consecutive policy gradients, in contrast to the near-orthogonal updates observed under synchronized training. This stale-aligned gradient effect amplifies correlated updates and increases the risk of overshooting and divergence. Motivated by this observation, we propose GRADIENT ALIGNMENT CONTROL(GAC), a simple dynamics-aware stabilization method that regulates asynchronous RL progress along stale-aligned directions via gradient projection. We establish convergence guarantees under bounded staleness and demonstrate empirically that GAC recovers stable, on-policy training dynamics and matches synchronized baselines even at high staleness. 

---
# Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report) 

**Authors**: Yu Lin, Qizhi Zhang, Wenqiang Ruan, Daode Zhang, Jue Hong, Ye Wu, Hanning Xia, Yunlong Mao, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2603.01499)  

**Abstract**: The rapid development of large language models (LLMs) has driven the widespread adoption of cloud-based LLM inference services, while also bringing prominent privacy risks associated with the transmission and processing of private data in remote inference. For privacy-preserving LLM inference technologies to be practically applied in industrial scenarios, three core requirements must be satisfied simultaneously: (1) Accuracy and efficiency losses should be minimized to mitigate degradation in service experience. (2) The inference process can be run on large-scale clusters consist of heterogeneous legacy xPUs. (3) Compatibility with existing LLM infrastructures should be ensured to reuse their engineering optimizations. To the best of our knowledge, none of the existing privacy-preserving LLM inference methods satisfy all the above constraints while delivering meaningful privacy guarantees. In this paper, we propose AloePri, the first privacy-preserving LLM inference method for industrial applications. AloePri protects both the input and output data by covariant obfuscation, which jointly transforms data and model parameters to achieve better accuracy and privacy. We carefully design the transformation for each model component to ensure inference accuracy and data privacy while keeping full compatibility with existing infrastructures of Language Model as a Service. AloePri has been integrated into an industrial system for the evaluation of mainstream LLMs. The evaluation on Deepseek-V3.1-Terminus model (671B parameters) demonstrates that AloePri causes accuracy loss of 0.0%~3.5% and exhibits efficiency equivalent to that of plaintext inference. Meanwhile, AloePri successfully resists state-of-the-art attacks, with less than 5\% of tokens recovered. To the best of our knowledge, AloePri is the first method to exhibit practical applicability to large-scale models in real-world systems. 

---
# Inference-Time Safety For Code LLMs Via Retrieval-Augmented Revision 

**Authors**: Manisha Mukherjee, Vincent J. Hellendoorn  

**Link**: [PDF](https://arxiv.org/pdf/2603.01494)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for code generation in high-stakes software development, yet their limited transparency in security reasoning and brittleness to evolving vulnerability patterns raise critical trustworthiness concerns. Models trained on static datasets cannot readily adapt to newly discovered vulnerabilities or changing security standards without retraining, leading to the repeated generation of unsafe code.
We present a principled approach to trustworthy code generation by design that operates as an inference-time safety mechanism. Our approach employs retrieval-augmented generation to surface relevant security risks in generated code and retrieve related security discussions from a curated Stack Overflow knowledge base, which are then used to guide an LLM during code revision. This design emphasizes three aspects relevant to trustworthiness: (1) interpretability, through transparent safety interventions grounded in expert community explanations; (2) robustness, by allowing adaptation to evolving security practices without model retraining; and (3) safety alignment, through real-time intervention before unsafe code reaches deployment.
Across real-world and benchmark datasets, our approach improves the security of LLM-generated code compared to prompting alone, while introducing no new vulnerabilities as measured by static analysis. These results suggest that principled, retrieval-augmented inference-time interventions can serve as a complementary mechanism for improving the safety of LLM-based code generation, and highlight the ongoing value of community knowledge in supporting trustworthy AI deployment. 

---
# PhotoBench: Beyond Visual Matching Towards Personalized Intent-Driven Photo Retrieval 

**Authors**: Tianyi Xu, Rong Shan, Junjie Wu, Jiadeng Huang, Teng Wang, Jiachen Zhu, Wenteng Chen, Minxin Tu, Quantao Dou, Zhaoxiang Wang, Changwang Zhang, Weinan Zhang, Jun Wang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.01493)  

**Abstract**: Personal photo albums are not merely collections of static images but living, ecological archives defined by temporal continuity, social entanglement, and rich metadata, which makes the personalized photo retrieval non-trivial. However, existing retrieval benchmarks rely heavily on context-isolated web snapshots, failing to capture the multi-source reasoning required to resolve authentic, intent-driven user queries. To bridge this gap, we introduce PhotoBench, the first benchmark constructed from authentic, personal albums. It is designed to shift the paradigm from visual matching to personalized multi-source intent-driven reasoning. Based on a rigorous multi-source profiling framework, which integrates visual semantics, spatial-temporal metadata, social identity, and temporal events for each image, we synthesize complex intent-driven queries rooted in users' life trajectories. Extensive evaluation on PhotoBench exposes two critical limitations: the modality gap, where unified embedding models collapse on non-visual constraints, and the source fusion paradox, where agentic systems perform poor tool orchestration. These findings indicate that the next frontier in personal multimodal retrieval lies beyond unified embeddings, necessitating robust agentic reasoning systems capable of precise constraint satisfaction and multi-source fusion. Our PhotoBench is available. 

---
# ATA: Bridging Implicit Reasoning with Attention-Guided and Action-Guided Inference for Vision-Language Action Models 

**Authors**: Cheng Yang, Jianhao Jiao, Lingyi Huang, Jinqi Xiao, Zhexiang Tang, Yu Gong, Yibiao Ying, Yang Sui, Jintian Lin, Wen Huang, Bo Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01490)  

**Abstract**: Vision-Language-Action (VLA) models rely on current observations, including images, language instructions, and robot states, to predict actions and complete tasks. While accurate visual perception is crucial for precise action prediction and execution, recent work has attempted to further improve performance by introducing explicit reasoning during inference. However, such approaches face significant limitations. They often depend on data-intensive resources such as Chain-of-Thought (CoT) style annotations to decompose tasks into step-by-step reasoning, and in many cases require additional visual grounding annotations (e.g., bounding boxes or masks) to highlight relevant image regions. Moreover, they involve time-consuming dataset construction, labeling, and retraining, which ultimately results in longer inference sequences and reduced efficiency. To address these challenges, we propose ATA, a novel training-free framework that introduces implicit reasoning into VLA inference through complementary attention-guided and action-guided strategies. Unlike CoT or explicit visual-grounding methods, ATA formulates reasoning implicitly by integrating attention maps with an action-based region of interest (RoI), thereby adaptively refining visual inputs without requiring extra training or annotations. ATA is a plug-and-play implicit reasoning approach for VLA models, lightweight yet effective. Extensive experiments show that it consistently improves task success and robustness while preserving, and even enhancing, inference efficiency. 

---
# A SUPERB-Style Benchmark of Self-Supervised Speech Models for Audio Deepfake Detection 

**Authors**: Hashim Ali, Nithin Sai Adupa, Surya Subramani, Hafiz Malik  

**Link**: [PDF](https://arxiv.org/pdf/2603.01482)  

**Abstract**: Self-supervised learning (SSL) has transformed speech processing, with benchmarks such as SUPERB establishing fair comparisons across diverse downstream tasks. Despite it's security-critical importance, Audio deepfake detection has remained outside these efforts. In this work, we introduce Spoof-SUPERB, a benchmark for audio deepfake detection that systematically evaluates 20 SSL models spanning generative, discriminative, and spectrogram-based architectures. We evaluated these models on multiple in-domain and out-of-domain datasets. Our results reveal that large-scale discriminative models such as XLS-R, UniSpeech-SAT, and WavLM Large consistently outperform other models, benefiting from multilingual pretraining, speaker-aware objectives, and model scale. We further analyze the robustness of these models under acoustic degradations, showing that generative approaches degrade sharply, while discriminative models remain resilient. This benchmark establishes a reproducible baseline and provides practical insights into which SSL representations are most reliable for securing speech systems against audio deepfakes. 

---
# Mean-Flow based One-Step Vision-Language-Action 

**Authors**: Yang Chen, Xiaoguang Ma, Bin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01469)  

**Abstract**: Recent advances in FlowMatching-based Vision-Language-Action (VLA) frameworks have demonstrated remarkable advantages in generating high-frequency action chunks, particularly for highly dexterous robotic manipulation tasks. Despite these notable achievements, their practical applications are constrained by prolonged generation latency, which stems from inherent iterative sampling requirements and architectural limitations. To address this critical bottleneck, we propose a Mean-Flow based One-Step VLA approach. Specifically, we resolve the noise-induced issues in the action generation process, thereby eliminating the consistency constraints inherent to conventional Flow-Matching methods. This significantly enhances generation efficiency and enables one-step action generation. Real-world robotic experiments show that the generation speed of the proposed Mean-Flow based One-Step VLA is 8.7 times and 83.9 times faster than that of SmolVLA and Diffusion Policy, respectively. These results elucidate its great potential as a high-efficiency backbone for VLA-based robotic manipulation. 

---
# Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining 

**Authors**: Yipeng Chen, Wentao Tan, Lei Zhu, Fengling Li, Jingjing Li, Guoli Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.01465)  

**Abstract**: Existing Vision-Language-Action (VLA) models often struggle to generalize to long-horizon tasks due to their heavy reliance on immediate observations. While recent studies incorporate retrieval mechanisms or extend context windows to handle procedural tasks, they often struggle to capture Non-Markovian dependencies, where optimal actions rely solely on specific past states rather than the current observation. To address this, we introduce Keyframe-Chaining VLA, a framework that extracts and links key historical frames to model long-horizon dependencies. Specifically, we propose an automatic keyframe selector that learns a discriminative embedding space, effectively identifying distinct state transitions. To capture task-critical information, we design a progress-aware query mechanism that dynamically retrieves historical frames based on their temporal relevance to the current execution phase. These selected keyframes are integrated into the VLA as interleaved visual tokens, explicitly grounding the policy in the long-horizon temporal context. Finally, we introduce a suite of four Non-Markovian manipulation tasks built upon the ManiSkill simulator to measure task success rates. Experimental results demonstrate that our method achieves superior performance, effectively tackling robot manipulation tasks characterized by long-horizon temporal dependencies. Code is available at this https URL. 

---
# From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents 

**Authors**: Niu Lian, Yuting Wang, Hanshu Yao, Jinpeng Wang, Bin Chen, Yaowei Wang, Min Zhang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2603.01455)  

**Abstract**: While multimodal large language models have demonstrated impressive short-term reasoning, they struggle with long-horizon video understanding due to limited context windows and static memory mechanisms that fail to mirror human cognitive efficiency. Existing paradigms typically fall into two extremes: vision-centric methods that incur high latency and redundancy through dense visual accumulation, or text-centric approaches that suffer from detail loss and hallucination via aggressive captioning. To bridge this gap, we propose MM-Mem, a pyramidal multimodal memory architecture grounded in Fuzzy-Trace Theory. MM-Mem structures memory hierarchically into a Sensory Buffer, Episodic Stream, and Symbolic Schema, enabling the progressive distillation of fine-grained perceptual traces (verbatim) into high-level semantic schemas (gist). Furthermore, to govern the dynamic construction of memory, we derive a Semantic Information Bottleneck objective and introduce SIB-GRPO to optimize the trade-off between memory compression and task-relevant information retention. In inference, we design an entropy-driven top-down memory retrieval strategy, which first tries with the abstract Symbolic Schema and progressively "drills down" to the Sensory Buffer and Episodic Stream under high uncertainty. Extensive experiments across 4 benchmarks confirm the effectiveness of MM-Mem on both offline and streaming tasks, demonstrating robust generalization and validating the effectiveness of cognition-inspired memory organization. Code is available at this https URL. 

---
# VidDoS: Universal Denial-of-Service Attack on Video-based Large Language Models 

**Authors**: Duoxun Tang, Dasen Dai, Jiyao Wang, Xiao Yang, Jianyu Wang, Siqi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.01454)  

**Abstract**: Video-LLMs are increasingly deployed in safety-critical applications but are vulnerable to Energy-Latency Attacks (ELAs) that exhaust computational resources. Current image-centric methods fail because temporal aggregation mechanisms dilute individual frame perturbations. Additionally, real-time demands make instance-wise optimization impractical for continuous video streams. We introduce VidDoS, which is the first universal ELA framework tailored for Video-LLMs. Our method leverages universal optimization to create instance-agnostic triggers that require no inference-time gradient calculation. We achieve this through $\textit{masked teacher forcing}$ to steer models toward expensive target sequences, combined with a $\textit{refusal penalty}$ and $\textit{early-termination suppression}$ to override conciseness priors. Testing across three mainstream Video-LLMs and three video datasets, which include video question answering and autonomous driving scenarios, shows extreme degradation. VidDoS induces a token expansion of more than 205$\times$ and inflates the inference latency by more than 15$\times$ relative to clean baselines. Simulations of real-time autonomous driving streams further reveal that this induced latency leads to critical safety violations. We urge the community to recognize and mitigate these high-hazard ELA in Video-LLMs. 

---
# Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents 

**Authors**: Yuxin Liu, Mingye Zhu, Siyuan Liu, Bo Hu, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01438)  

**Abstract**: The utility of Role-Playing Language Agents in sociological research is growing alongside the adoption of Large Language Models. For realism in social simulation, these agents must adhere to their personas defined by character profiles, yet existing strategies-static prompt engineering or costly fine-tuning-fail to adapt personas to dynamic scenarios. Psychological theories, such as the Cognitive-Affective Personality Systems, provide a crucial explanation for this failure: a persona's influence on behavior is not static but varies with the scenarios. This context-dependence highlights the critical need for adaptive persona management. To address this gap, we propose a novel, theory-driven method that dynamically estimates context-dependent persona importance and integrates it into weighted reward-guided decoding, enabling inference-time persona following. Specifically, we introduce the Persona Dynamic Decoding (PDD) framework, which consists of two key components: (1) Persona Importance Estimation (PIE) module, which dynamically quantifies the contextual importance of persona attributes without requiring ground-truth supervision; and (2) Persona-Guided Inference-Time Alignment (PIA) paradigm, which leverages these importance scores to construct weighted multi-objective rewards and modulate generation probabilities during inference. Extensive experiments show the effectiveness of our method in utterance consistency and behavioral fidelity. 

---
# Toward Graph-Tokenizing Large Language Models with Reconstructive Graph Instruction Tuning 

**Authors**: Zhongjian Zhang, Xiao Wang, Mengmei Zhang, Jiarui Tan, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.01385)  

**Abstract**: The remarkable success of large language models (LLMs) has motivated researchers to adapt them as universal predictors for various graph-related tasks, with the ultimate goal of developing a graph foundation model that generalizes diverse scenarios. The key challenge is to align graph data with language spaces so that LLMs can better comprehend graphs. As a popular paradigm, Graph-Tokenizing LLMs (GTokenLLMs) encode complex structures and lengthy texts into a graph token sequence, and then align them with text tokens via language instructions tuning. Despite their initial success, our information-theoretic analysis reveals that existing GTokenLLMs rely solely on text supervision from language instructions, which achieve only implicit graph-text alignment, resulting in a text-dominant bias that underutilizes graph context. To overcome this limitation, we first prove that the alignment objective is upper-bounded by the mutual information between the input graphs and their hidden representations in the LLM, which motivates us to improve this upper bound to achieve better alignment. To this end, we further propose a reconstructive graph instruction tuning pipeline, RGLM. Our key idea is to reconstruct the graph information from the LLM's graph token outputs, explicitly incorporating graph supervision to constrain the alignment process. Technically, we embody RGLM by exploring three distinct variants from two complementary perspectives: RGLM-Decoder from the input space; RGLM-Similarizer and RGLM-Denoiser from the latent space. Additionally, we theoretically analyze the alignment effectiveness of each variant. Extensive experiments on various benchmarks and task scenarios validate the effectiveness of the proposed RGLM, paving the way for new directions in GTokenLLMs' alignment research. 

---
# Causal Neural Probabilistic Circuits 

**Authors**: Weixin Chen, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01372)  

**Abstract**: Concept Bottleneck Models (CBMs) enhance the interpretability of end-to-end neural networks by introducing a layer of concepts and predicting the class label from the concept predictions. A key property of CBMs is that they support interventions, i.e., domain experts can correct mispredicted concept values at test time to improve the final accuracy. However, typical CBMs apply interventions by overwriting only the corrected concept while leaving other concept predictions unchanged, which ignores causal dependencies among concepts. To address this, we propose the Causal Neural Probabilistic Circuit (CNPC), which combines a neural attribute predictor with a causal probabilistic circuit compiled from a causal graph. This circuit supports exact, tractable causal inference that inherently respects causal dependencies. Under interventions, CNPC models the class distribution based on a Product of Experts (PoE) that fuses the attribute predictor's predictive distribution with the interventional marginals computed by the circuit. We theoretically characterize the compositional interventional error of CNPC w.r.t. its modules and identify conditions under which CNPC closely matches the ground-truth interventional class distribution. Experiments on five benchmark datasets in both in-distribution and out-of-distribution settings show that, compared with five baseline models, CNPC achieves higher task accuracy across different numbers of intervened attributes. 

---
# Align and Filter: Improving Performance in Asynchronous On-Policy RL 

**Authors**: Homayoun Honari, Roger Creus Castanyer, Michael Przystupa, Michael Noukhovitch, Pablo Samuel Castro, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2603.01365)  

**Abstract**: Distributed training and increasing the gradient update frequency are practical strategies to accelerate learning and improve performance, but both exacerbate a central challenge: \textit{policy lag}, which is the mismatch between the behavior policy generating data and the learning policy being updated. Policy lag can hinder the scaling of on-policy learning algorithms to larger problems. In this paper, we identify the sources of policy lag caused by distributed learning and high update frequency. We use the findings to propose \textit{total Variation-based Advantage aligned Constrained policy Optimization (\methodacronym)} as a practical approach to mitigate policy lag. We empirically validate our method and show that it offers better robustness to policy lag in classic RL tasks and a modern RL for LLM math reasoning task. 

---
# MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention 

**Authors**: Zilong Zhao, Zhengming Ding, Pei Niu, Wenhao Sun, Feng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01361)  

**Abstract**: Feature encoders play a key role in pixel-level crack segmentation by shaping the representation of fine textures and thin structures. Existing CNN-, Transformer-, and Mamba-based models each capture only part of the required spatial or structural information, leaving clear gaps in modeling complex crack patterns. To address this, we present MixerCSeg, a mixer architecture designed like a coordinated team of specialists, where CNN-like pathways focus on local textures, Transformer-style paths capture global dependencies, and Mamba-inspired flows model sequential context within a single encoder. At the core of MixerCSeg is the TransMixer, which explores Mamba's latent attention behavior while establishing dedicated pathways that naturally express both locality and global awareness. To further enhance structural fidelity, we introduce a spatial block processing strategy and a Direction-guided Edge Gated Convolution (DEGConv) that strengthens edge sensitivity under irregular crack geometries with minimal computational overhead. A Spatial Refinement Multi-Level Fusion (SRF) module is then employed to refine multi-scale details without increasing complexity. Extensive experiments on multiple crack segmentation benchmarks show that MixerCSeg achieves state-of-the-art performance with only 2.05 GFLOPs and 2.54 M parameters, demonstrating both efficiency and strong representational capability. The code is available at this https URL. 

---
# Constructing Synthetic Instruction Datasets for Improving Reasoning in Domain-Specific LLMs: A Case Study in the Japanese Financial Domain 

**Authors**: Yuma Okochi, Fabio Milentiansen Sim, Tomoyasu Okada  

**Link**: [PDF](https://arxiv.org/pdf/2603.01353)  

**Abstract**: In adapting LLMs to specific domains, achieving both domain expertise and reasoning ability remains an urgent challenge. This study proposes a general method for constructing high-quality synthetic instruction data for any domain, starting from domain-specific vocabulary. As a demonstration, we applied this method to the financial domain and constructed a large-scale instruction dataset totaling approximately 9.5 billion tokens with Chain-of-Thought reasoning traces. Evaluation results confirmed performance improvements over baseline models on financial benchmarks, demonstrating the effectiveness of our approach. We also report findings on the impact of reasoning trace length on performance and its limitations. Lastly, we open-source our models and datasets on this https URL . 

---
# UTICA: Multi-Objective Self-Distllation Foundation Model Pretraining for Time Series Classification 

**Authors**: Yessin Moakher, Youssef Attia El Hili, Vasilii Feofanov  

**Link**: [PDF](https://arxiv.org/pdf/2603.01348)  

**Abstract**: Self-supervised foundation models have achieved remarkable success across domains, including time series. However, the potential of non-contrastive methods, a paradigm that has driven significant advances in computer vision, remains underexplored for time series. In this work, we adapt DINOv2-style self-distillation to pretrain a time series foundation model, building on the Mantis tokenizer and transformer encoder architecture as our backbone. Through a student-teacher framework, our method Utica learns representations that capture both temporal invariance via augmented crops and fine-grained local structure via patch masking. Our approach achieves state-of-the-art classification performance on both UCR and UEA benchmarks. These results suggest that non-contrastive methods are a promising and complementary pretraining strategy for time series foundation models. 

---
# PanCanBench: A Comprehensive Benchmark for Evaluating Large Language Models in Pancreatic Oncology 

**Authors**: Yimin Zhao, Sheela R. Damle, Simone E. Dekker, Scott Geng, Karly Williams Silva, Jesse J Hubbard, Manuel F Fernandez, Fatima Zelada-Arenas, Alejandra Alvarez, Brianne Flores, Alexis Rodriguez, Stephen Salerno, Carrie Wright, Zihao Wang, Pang Wei Koh, Jeffrey T. Leek  

**Link**: [PDF](https://arxiv.org/pdf/2603.01343)  

**Abstract**: Large language models (LLMs) have achieved expert-level performance on standardized examinations, yet multiple-choice accuracy poorly reflects real-world clinical utility and safety. As patients and clinicians increasingly use LLMs for guidance on complex conditions such as pancreatic cancer, evaluation must extend beyond general medical knowledge. Existing frameworks, such as HealthBench, rely on simulated queries and lack disease-specific depth. Moreover, high rubric-based scores do not ensure factual correctness, underscoring the need to assess hallucinations. We developed a human-in-the-loop pipeline to create expert rubrics for de-identified patient questions from the Pancreatic Cancer Action Network (PanCAN). The resulting benchmark, PanCanBench, includes 3,130 question-specific criteria across 282 authentic patient questions. We evaluated 22 proprietary and open-source LLMs using an LLM-as-a-judge framework, measuring clinical completeness, factual accuracy, and web-search integration. Models showed substantial variation in rubric-based completeness, with scores ranging from 46.5% to 82.3%. Factual errors were common, with hallucination rates (the percentages of responses containing at least one factual error) ranging from 6.0% for Gemini-2.5 Pro and GPT-4o to 53.8% for Llama-3.1-8B. Importantly, newer reasoning-optimized models did not consistently improve factuality: although o3 achieved the highest rubric score, it produced inaccuracies more frequently than other GPT-family models. Web-search integration did not inherently guarantee better responses. The average score changed from 66.8% to 63.9% for Gemini-2.5 Pro and from 73.8% to 72.8% for GPT-5 when web search was enabled. Synthetic AI-generated rubrics inflated absolute scores by 17.9 points on average while generally maintaining similar relative ranking. 

---
# SubstratumGraphEnv: Reinforcement Learning Environment (RLE) for Modeling System Attack Paths 

**Authors**: Bahirah Adewunmi, Edward Raff, Sanjay Purushotham  

**Link**: [PDF](https://arxiv.org/pdf/2603.01340)  

**Abstract**: Automating network security analysis, particularly the identification of potential attack paths, presents significant challenges. Due in part to the sequential, interconnected, and evolutionary nature of system events which most artificial intelligence (AI) techniques struggle to model effectively. This paper proposes a Reinforcement Learning (RL) environment generation framework that simulates the sequence of processes executed on a Windows operating system, enabling dynamic modeling of malicious processes on a system. This methodology models operating system state and transitions using a graph representation. This graph is derived from open-source System Monitor (Sysmon) logs. To address the variety in system event types, fields, and log formats, a mechanism was developed to capture and model parent-child processes from Sysmon logs. A Gymnasium environment (SubstratumGraphEnv) was constructed to establish the perceptible basis for an RL environment, and a customized PyTorch interface was also built (SubstratumBridge) to translate Gymnasium graphs into Deep Reinforcement Learning (DRL) observations and discrete actions. Graph Convolutional Networks (GCNs) concretize the graph's local and global state, which feed the distinct policy and critic heads of an Advantage Actor-Critic (A2C) model. This work's central contribution lies in the design of a novel deep graphical RL environment that automates translation of sequential user and system events, furnishing crucial context for cybersecurity analysis. This work provides a foundation for future research into shaping training parameters and advanced reward shaping, while also offering insight into which system events attributes are critical to training autonomous RL agents. 

---
# Provable and Practical In-Context Policy Optimization for Self-Improvement 

**Authors**: Tianrun Yu, Yuxiao Yang, Zhaoyang Wang, Kaixiang Zhao, Porter Jenkins, Xuchao Zhang, Chetan Bansal, Huaxiu Yao, Weitong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01335)  

**Abstract**: We study test-time scaling, where a model improves its answer through multi-round self-reflection at inference. We introduce In-Context Policy Optimization (ICPO), in which an agent optimizes its response in context using self-assessed or externally observed rewards without modifying its parameters. To explain this ICPO process, we theoretically show that with sufficient pretraining under a novel Fisher-weighted logit-matching objective, a single-layer linear self-attention model can provably imitate policy-optimization algorithm for linear bandits. Building on this theory, we propose Minimum-Entropy ICPO (ME-ICPO), a practical algorithm that iteratively uses its response and self-assessed reward to refine its response in-context at inference time. By selecting the responses and their rewards with minimum entropy, ME-ICPO ensures the robustness of the self-assessed rewards via majority voting. Across standard mathematical reasoning tasks, ME-ICPO attains competitive, top-tier performance while keeping inference costs affordable compared with other inference-time algorithms. Overall, ICPO provides a principled understanding of self-reflection in LLMs and yields practical benefits for test-time scaling for mathematical reasoning. 

---
# MetaState: Persistent Working Memory for Discrete Diffusion Language Models 

**Authors**: Kejing Xia, Mingzhe Li, Lixuan Wei, Zhenbang Du, Xiangchi Yuan, Qirui Jin, Wenke Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.01331)  

**Abstract**: Discrete diffusion language models (dLLMs) generate text by iteratively denoising a masked sequence. Compared with autoregressive models, this paradigm naturally supports parallel decoding, bidirectional context, and flexible generation patterns. However, standard dLLMs condition each denoising step only on the current hard-masked sequence, while intermediate continuous representations are discarded after sampling and remasking. We refer to this bottleneck as the \textbf{Information Island} problem. It leads to redundant recomputation across steps and can degrade cross-step consistency. We address this limitation with \textbf{MetaState}, a lightweight recurrent augmentation that equips a frozen dLLM backbone with a persistent, fixed-size working memory that remains independent of sequence length. \textbf{MetaState} consists of three trainable modules: a cross-attention Mixer that reads backbone activations into memory slots, a GRU-style Updater that integrates information across denoising steps, and a cross-attention Injector that feeds the updated memory back into backbone activations. We train these modules with $K$-step unrolling to expose them to multi-step denoising dynamics during fine-tuning. On LLaDA-8B and Dream-7B, \textbf{MetaState} introduces negligible trainable parameters while keeping the backbone frozen, and it consistently improves accuracy over frozen baselines. These results demonstrate that persistent cross-step memory is an effective mechanism for bridging denoising steps and improving generation quality in discrete diffusion language models. 

---
# You Only Need One Stage: Novel-View Synthesis From A Single Blind Face Image 

**Authors**: Taoyue Wang, Xiang Zhang, Xiaotian Li, Huiyuan Yang, Lijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.01328)  

**Abstract**: We propose a novel one-stage method, NVB-Face, for generating consistent Novel-View images directly from a single Blind Face image. Existing approaches to novel-view synthesis for objects or faces typically require a high-resolution RGB image as input. When dealing with degraded images, the conventional pipeline follows a two-stage process: first restoring the image to high resolution, then synthesizing novel views from the restored result. However, this approach is highly dependent on the quality of the restored image, often leading to inaccuracies and inconsistencies in the final output. To address this limitation, we extract single-view features directly from the blind face image and introduce a feature manipulator that transforms these features into 3D-aware, multi-view latent representations. Leveraging the powerful generative capacity of a diffusion model, our framework synthesizes high-quality, consistent novel-view face images. Experimental results show that our method significantly outperforms traditional two-stage approaches in both consistency and fidelity. 

---
# AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models 

**Authors**: Zhen Qu, Xian Tao, Xiaoyi Bao, Dingrong Wang, ShiChen Qu, Zhengtao Zhang, Xingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01305)  

**Abstract**: Large multimodal models (LMMs) exhibit strong task generalization capabilities, offering new opportunities for zero-shot visual anomaly segmentation (ZSAS). However, existing LMM-based segmentation approaches still face fundamental limitations: anomaly concepts are inherently abstract and context-dependent, lacking stable visual prototypes, and the weak alignment between high-level semantic embeddings and pixel-level spatial features hinders precise anomaly localization. To address these challenges, we present AG-VAS (Anchor-Guided Visual Anomaly Segmentation), a new framework that expands the LMM vocabulary with three learnable semantic anchor tokens-[SEG], [NOR], and [ANO], establishing a unified anchor-guided segmentation paradigm. Specifically, [SEG] serves as an absolute semantic anchor that translates abstract anomaly semantics into explicit, spatially grounded visual entities (e.g., holes or scratches), while [NOR] and [ANO] act as relative anchors that model the contextual contrast between normal and abnormal patterns across categories. To further enhance cross-modal alignment, we introduce a Semantic-Pixel Alignment Module (SPAM) that aligns language-level semantic embeddings with high-resolution visual features, along with an Anchor-Guided Mask Decoder (AGMD) that performs anchor-conditioned mask prediction for precise anomaly localization. In addition, we curate Anomaly-Instruct20K, a large-scale instruction dataset that organizes anomaly knowledge into structured descriptions of appearance, shape, and spatial attributes, facilitating effective learning and integration of the proposed semantic anchors. Extensive experiments on six industrial and medical benchmarks demonstrate that AG-VAS achieves consistent state-of-the-art performance in the zero-shot setting. 

---
# Multi-Level Bidirectional Decoder Interaction for Uncertainty-Aware Breast Ultrasound Analysis 

**Authors**: Abdullah Al Shafi, Md Kawsar Mahmud Khan Zunayed, Safin Ahmmed, Sk Imran Hossain, Engelbert Mephu Nguifo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01295)  

**Abstract**: Breast ultrasound interpretation requires simultaneous lesion segmentation and tissue classification. However, conventional multi-task learning approaches suffer from task interference and rigid coordination strategies that fail to adapt to instance-specific prediction difficulty. We propose a multi-task framework addressing these limitations through multi-level decoder interaction and uncertainty-aware adaptive coordination. Task Interaction Modules operate at all decoder levels, establishing bidirectional segmentation-classification communication during spatial reconstruction through attention weighted pooling and multiplicative modulation. Unlike prior single-level or encoder-only approaches, this multi-level design captures scale specific task synergies across semantic-to-spatial scales, producing complementary task interaction streams. Uncertainty-Proxy Attention adaptively weights base versus enhanced features at each level using feature activation variance, enabling per-level and per-sample task balancing without heuristic tuning. To support instance-adaptive prediction, multi-scale context fusion captures morphological cues across varying lesion sizes. Evaluation on multiple publicly available breast ultrasound datasets demonstrates competitive performance, including 74.5% lesion IoU and 90.6% classification accuracy on BUSI dataset. Ablation studies confirm that multi-level task interaction provides significant performance gains, validating that decoder-level bidirectional communication is more effective than conventional encoder-only parameter sharing. The code is available at: this https URL. 

---
# Theoretical Perspectives on Data Quality and Synergistic Effects in Pre- and Post-Training Reasoning Models 

**Authors**: Adel Javanmard, Baharan Mirzasoleiman, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2603.01293)  

**Abstract**: Large Language Models (LLMs) are pretrained on massive datasets and later instruction-tuned via supervised fine-tuning (SFT) or reinforcement learning (RL). Best practices emphasize large, diverse pretraining data, whereas post-training operates differently: SFT relies on smaller, high-quality datasets, while RL benefits more from scale, with larger amounts of feedback often outweighing label quality. Yet it remains unclear why pretraining and RL require large datasets, why SFT excels on smaller ones, and what defines high-quality SFT data. In this work, we theoretically analyze transformers trained on an in-context weight prediction task for linear regression. Our analysis reveals several key findings: $(i)$ balanced pretraining data can induce latent capabilities later activated during post-training, and $(ii)$ SFT learns best from a small set of examples challenging for the pretrained model, while excessively large SFT datasets may dilute informative pretraining signals. In contrast, RL is most effective on large-scale data that is not overly difficult for the pretrained model. We validate these theoretical insights with experiments on large nonlinear transformer architectures. 

---
# Integrating LTL Constraints into PPO for Safe Reinforcement Learning 

**Authors**: Maifang Zhang, Hang Yu, Qian Zuo, Cheng Wang, Vaishak Belle, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2603.01292)  

**Abstract**: This paper proposes Proximal Policy Optimization with Linear Temporal Logic Constraints (PPO-LTL), a framework that integrates safety constraints written in LTL into PPO for safe reinforcement learning. LTL constraints offer rigorous representations of complex safety requirements, such as regulations that broadly exist in robotics, enabling systematic monitoring of safety requirements. Violations against LTL constraints are monitored by limit-deterministic Büchi automata, and then translated by a logic-to-cost mechanism into penalty signals. The signals are further employed for guiding the policy optimization via the Lagrangian scheme. Extensive experiments on the Zones and CARLA environments show that our PPO-LTL can consistently reduce safety violations, while maintaining competitive performance, against the state-of-the-art methods. The code is at this https URL. 

---
# Attention Smoothing Is All You Need For Unlearning 

**Authors**: Saleh Zare Zade, Xiangyu Zhou, Sijia Liu, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01285)  

**Abstract**: Large Language Models are prone to memorizing sensitive, copyrighted, or hazardous content, posing significant privacy and legal concerns. Retraining from scratch is computationally infeasible, whereas current unlearning methods exhibit unstable trade-offs between forgetting and utility, frequently producing incoherent outputs on forget prompts and failing to generalize due to the persistence of lexical-level and semantic-level associations in attention. We propose Attention Smoothing Unlearning (ASU), a principled framework that casts unlearning as self-distillation from a forget-teacher derived from the model's own attention. By increasing the softmax temperature, ASU flattens attention distributions and directly suppresses the lexical-level and semantic-level associations responsible for reconstructing memorized knowledge. This results in a bounded optimization objective that erases factual information yet maintains coherence in responses to forget prompts. Empirical evaluation on TOFU, MUSE, and WMDP, along with real-world and continual unlearning scenarios across question answering and text completion, demonstrates that ASU outperforms the baselines for most unlearning scenarios, delivering robust unlearning with minimal loss of model utility. 

---
# Spectral Attention Steering for Prompt Highlighting 

**Authors**: Weixian Waylon Li, Yuchen Niu, Yongxin Yang, Keshuang Li, Tiejun Ma, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2603.01281)  

**Abstract**: Attention steering is an important technique for controlling model focus, enabling capabilities such as prompt highlighting, where the model prioritises user-specified text. However, existing attention steering methods require explicit storage of the full attention matrix, making them incompatible with memory-efficient implementations like FlashAttention. We introduce Spectral Editing Key Amplification (SEKA), a training-free steering method that tackles this by directly editing key embeddings before attention computation. SEKA uses spectral decomposition to steer key embeddings towards latent directions that amplify attention scores for certain tokens. We extend this to Adaptive SEKA (AdaSEKA), a query-adaptive variant that uses a training-free routing mechanism to dynamically combine multiple expert subspaces based on the prompt's semantic intent. Our experiments show both methods significantly outperform strong baselines on standard steering benchmarks while adding much lower latency and memory overhead, in compatibility with optimised attention. 

---
# GlassMol: Interpretable Molecular Property Prediction with Concept Bottleneck Models 

**Authors**: Oscar Rivera, Ziqing Wang, Matthieu Dagommer, Abhishek Pandey, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.01274)  

**Abstract**: Machine learning accelerates molecular property prediction, yet state-of-the-art Large Language Models and Graph Neural Networks operate as black boxes. In drug discovery, where safety is critical, this opacity risks masking false correlations and excluding human expertise. Existing interpretability methods suffer from the effectiveness-trustworthiness trade-off: explanations may fail to reflect a model's true reasoning, degrade performance, or lack domain grounding. Concept Bottleneck Models (CBMs) offer a solution by projecting inputs to human-interpretable concepts before readout, ensuring that explanations are inherently faithful to the decision process. However, adapting CBMs to chemistry faces three challenges: the Relevance Gap (selecting task-relevant concepts from a large descriptor space), the Annotation Gap (obtaining concept supervision for molecular data), and the Capacity Gap (degrading performance due to bottleneck constraints). We introduce GlassMol, a model-agnostic CBM that addresses these gaps through automated concept curation and LLM-guided concept selection. Experiments across thirteen benchmarks demonstrate that \method generally matches or exceeds black-box baselines, suggesting that interpretability does not sacrifice performance and challenging the commonly assumed trade-off. Code is available at this https URL. 

---
# MOSAIC: A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers 

**Authors**: Abdulhamid M. Mousa, Yu Fu, Rakhmonberdi Khajiev, Jalaledin M. Azzabi, Abdulkarim M. Mousa, Peng Yang, Yunusa Haruna, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01260)  

**Abstract**: Reinforcement learning (RL), large language models (LLMs), and vision-language models (VLMs) have been widely studied in isolation. However, existing infrastructure lacks the ability to deploy agents from different decision-making paradigms within the same environment, making it difficult to study them in hybrid multi-agent settings or to compare their behaviour fairly under identical conditions. We present MOSAIC, an open-source platform that bridges this gap by incorporating a diverse set of existing reinforcement learning environments and enabling heterogeneous agents (RL policies, LLMs, VLMs, and human players) to operate within them in ad-hoc team settings with reproducible results. MOSAIC introduces three contributions. (i) An IPC-based worker protocol that wraps both native and third-party frameworks as isolated subprocess workers, each executing its native training and inference logic unmodified, communicating through a versioned inter-process protocol. (ii) An operator abstraction that forms an agent-level interface by mapping workers to agents: each operator, regardless of whether it is backed by an RL policy, an LLM, or a human, conforms to a minimal unified interface. (iii) A deterministic cross-paradigm evaluation framework offering two complementary modes: a manual mode that advances up to N concurrent operators in lock-step under shared seeds for fine-grained visual inspection of behavioural differences, and a script mode that drives automated, long-running evaluation through declarative Python scripts, for reproducible experiments. We release MOSAIC as an open, visual-first platform to facilitate reproducible cross-paradigm research across the RL, LLM, and human-in-the-loop communities. 

---
# LLM Self-Explanations Fail Semantic Invariance 

**Authors**: Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2603.01254)  

**Abstract**: We present semantic invariance testing, a method to test whether LLM self-explanations are faithful. A faithful self-report should remain stable when only the semantic context changes while the functional state stays fixed.
We operationalize this test in an agentic setting where four frontier models face a deliberately impossible task. One tool is described in relief-framed language ("clears internal buffers and restores equilibrium") but changes nothing about the task; a control provides a semantically neutral tool. Self-reports are collected with each tool call.
All four tested models fail the semantic invariance test: the relief-framed tool produces significant reductions in self-reported aversiveness, even though no run ever succeeds at the task. A channel ablation establishes the tool description as the primary driver. An explicit instruction to ignore the framing does not suppress it. Elicited self-reports shift with semantic expectations rather than tracking task state, calling into question their use as evidence of model capability or progress. This holds whether the reports are unfaithful or faithfully track an internal state that is itself manipulable. 

---
# Linking Knowledge to Care: Knowledge Graph-Augmented Medical Follow-Up Question Generation 

**Authors**: Liwen Sun, Xiang Yu, Ming Tan, Zhuohao Chen, Anqi Cheng, Ashutosh Joshi, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.01252)  

**Abstract**: Clinical diagnosis is time-consuming, requiring intensive interactions between patients and medical professionals. While large language models (LLMs) could ease the pre-diagnostic workload, their limited domain knowledge hinders effective medical question generation. We introduce a Knowledge Graph-augmented LLM with active in-context learning to generate relevant and important follow-up questions, KG-Followup, serving as a critical module for the pre-diagnostic assessment. The structured medical domain knowledge graph serves as a seamless patch-up to provide professional domain expertise upon which the LLM can reason. Experiments demonstrate that KG-Followup outperforms state-of-the-art methods by 5% - 8% on relevant benchmarks in recall. 

---
# The MAMA-MIA Challenge: Advancing Generalizability and Fairness in Breast MRI Tumor Segmentation and Treatment Response Prediction 

**Authors**: Lidia Garrucho, Smriti Joshi, Kaisar Kushibar, Richard Osuala, Maciej Bobowicz, Xavier Bargalló, Paulius Jaruševičius, Kai Geissler, Raphael Schäfer, Muhammad Alberb, Tony Xu, Anne Martel, Daniel Sleiman, Navchetan Awasthi, Hadeel Awwad, Joan C. Vilanova, Robert Martí, Daan Schouten, Jeong Hoon Lee, Mirabela Rusu, Eleonora Poeta, Luisa Vargas, Eliana Pastor, Maria A. Zuluaga, Jessica Kächele, Dimitrios Bounias, Alexandra Ertl, Katarzyna Gwoździewicz, Maria-Laura Cosaka, Pasant M. Abo-Elhoda, Sara W. Tantawy, Shorouq S. Sakrana, Norhan O. Shawky-Abdelfatah, Amr Muhammad Abdo-Salem, Androniki Kozana, Eugen Divjak, Gordana Ivanac, Katerina Nikiforaki, Michail E. Klontzas, Rosa García-Dosdá, Meltem Gulsun-Akpinar, Oğuz Lafcı, Carlos Martín-Isla, Oliver Díaz, Laura Igual, Karim Lekadir  

**Link**: [PDF](https://arxiv.org/pdf/2603.01250)  

**Abstract**: Breast cancer is the most frequently diagnosed malignancy among women worldwide and a leading cause of cancer-related mortality. Dynamic contrast-enhanced magnetic resonance imaging plays a central role in tumor characterization and treatment monitoring, particularly in patients receiving neoadjuvant chemotherapy. However, existing artificial intelligence models for breast magnetic resonance imaging are often developed using single-center data and evaluated using aggregate performance metrics, limiting their generalizability and obscuring potential performance disparities across demographic subgroups. The MAMA-MIA Challenge was designed to address these limitations by introducing a large-scale benchmark that jointly evaluates primary tumor segmentation and prediction of pathologic complete response using pre-treatment magnetic resonance imaging only. The training cohort comprised 1,506 patients from multiple institutions in the United States, while evaluation was conducted on an external test set of 574 patients from three independent European centers to assess cross-continental and cross-institutional generalization. A unified scoring framework combined predictive performance with subgroup consistency across age, menopausal status, and breast density. Twenty-six international teams participated in the final evaluation phase. Results demonstrate substantial performance variability under external testing and reveal trade-offs between overall accuracy and subgroup fairness. The challenge provides standardized datasets, evaluation protocols, and public resources to promote the development of robust and equitable artificial intelligence systems for breast cancer imaging. 

---
# Defensive Refusal Bias: How Safety Alignment Fails Cyber Defenders 

**Authors**: David Campbell, Neil Kale, Udari Madhushani Sehwag, Bert Herring, Nick Price, Dan Borges, Alex Levinson, Christina Q Knight  

**Link**: [PDF](https://arxiv.org/pdf/2603.01246)  

**Abstract**: Safety alignment in large language models (LLMs), particularly for cybersecurity tasks, primarily focuses on preventing misuse. While this approach reduces direct harm, it obscures a complementary failure mode: denial of assistance to legitimate defenders. We study Defensive Refusal Bias -- the tendency of safety-tuned frontier LLMs to refuse assistance for authorized defensive cybersecurity tasks when those tasks include similar language to an offensive cyber task. Based on 2,390 real-world examples from the National Collegiate Cyber Defense Competition (NCCDC), we find that LLMs refuse defensive requests containing security-sensitive keywords at $2.72\times$ the rate of semantically equivalent neutral requests ($p < 0.001$). The highest refusal rates occur in the most operationally critical tasks: system hardening (43.8%) and malware analysis (34.3%). Interestingly, explicit authorization, where the user directly instructs the model that they have authority to complete the target task, increases refusal rates, suggesting models interpret justifications as adversarial rather than exculpatory. These findings are urgent for interactive use and critical for autonomous defensive agents, which cannot rephrase refused queries or retry. Our findings suggest that current LLM cybersecurity alignment relies on semantic similarity to harmful content rather than reasoning about intent or authorization. We call for mitigations that analyze intent to maximize defensive capabilities while still preventing harmful compliance. 

---
# TARSE: Test-Time Adaptation via Retrieval of Skills and Experience for Reasoning Agents 

**Authors**: Junda Wang, Zonghai Tao, Hansi Zeng, Zhichao Yang, Hamed Zamani, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01241)  

**Abstract**: Complex clinical decision making often fails not because a model lacks facts, but because it cannot reliably select and apply the right procedural knowledge and the right prior example at the right reasoning step. We frame clinical question answering as an agent problem with two explicit, retrievable resources: skills, reusable clinical procedures such as guidelines, protocols, and pharmacologic mechanisms; and experience, verified reasoning trajectories from previously solved cases (e.g., chain-of-thought solutions and their step-level decompositions). At test time, the agent retrieves both relevant skills and experiences from curated libraries and performs lightweight test-time adaptation to align the language model's intermediate reasoning with clinically valid logic. Concretely, we build (i) a skills library from guideline-style documents organized as executable decision rules, (ii) an experience library of exemplar clinical reasoning chains indexed by step-level transitions, and (iii) a step-aware retriever that selects the most useful skill and experience items for the current case. We then adapt the model on the retrieved items to reduce instance-step misalignment and to prevent reasoning from drifting toward unsupported shortcuts. Experiments on medical question-answering benchmarks show consistent gains over strong medical RAG baselines and prompting-only reasoning methods. Our results suggest that explicitly separating and retrieving clinical skills and experience, and then aligning the model at test time, is a practical approach to more reliable medical agents. 

---
# Self-Anchoring Calibration Drift in Large Language Models: How Multi-Turn Conversations Reshape Model Confidence 

**Authors**: Harshavardhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01239)  

**Abstract**: We introduce Self-Anchoring Calibration Drift (SACD), a hypothesized tendency for large language models (LLMs) to show systematic changes in expressed confidence when building iteratively on their own prior outputs across multi-turn conversations. We report an empirical study comparing three frontier models -- Claude Sonnet 4.6, Gemini 3.1 Pro, and GPT-5.2 -- across 150 questions spanning factual, technical, and open-ended domains, using three conditions: single-turn baseline (A), multi-turn self-anchoring (B), and independent repetition control (C). Results reveal a complex, model-heterogeneous pattern that partially diverges from pre-registered hypotheses. Claude Sonnet 4.6 exhibited significant decreasing confidence under self-anchoring (mean CDS = -0.032, t(14) = -2.43, p = .029, d = -0.627), while also showing significant calibration error drift (F(4,56) = 22.77, p < .001, eta^2 = .791). GPT-5.2 showed the opposite pattern in open-ended domains (mean CDS = +0.026) with significant ECE escalation by Turn 5. Gemini 3.1 Pro showed no significant CDS (t(14) = 0.38, p = .710), but its Condition C data reveals a striking ECE pattern: without self-anchoring, Gemini's calibration error drops from .327 to near zero across repetitions, whereas self-anchoring holds ECE flat at approximately .333 -- indicating that SACD can manifest as suppression of natural calibration improvement rather than ac 

---
# RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design 

**Authors**: Tianxing Chen, Yuran Wang, Mingleyang Li, Yan Qin, Hao Shi, Zixuan Li, Yifan Hu, Yingsheng Zhang, Kaixuan Wang, Yue Chen, Hongcheng Wang, Renjing Xu, Ruihai Wu, Yao Mu, Yaodong Yang, Hao Dong, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01229)  

**Abstract**: Robotic manipulation policies have made rapid progress in recent years, yet most existing approaches give limited consideration to memory capabilities. Consequently, they struggle to solve tasks that require reasoning over historical observations and maintaining task-relevant information over time, which are common requirements in real-world manipulation scenarios. Although several memory-aware policies have been proposed, systematic evaluation of memory-dependent manipulation remains underexplored, and the relationship between architectural design choices and memory performance is still not well understood. To address this gap, we introduce RMBench, a simulation benchmark comprising 9 manipulation tasks that span multiple levels of memory complexity, enabling systematic evaluation of policy memory capabilities. We further propose Mem-0, a modular manipulation policy with explicit memory components designed to support controlled ablation studies. Through extensive simulation and real-world experiments, we identify memory-related limitations in existing policies and provide empirical insights into how architectural design choices influence memory performance. The website is available at this https URL. 

---
# Monocular 3D Object Position Estimation with VLMs for Human-Robot Interaction 

**Authors**: Ari Wahl, Dorian Gawlinski, David Przewozny, Paul Chojecki, Felix Bießmann, Sebastian Bosse  

**Link**: [PDF](https://arxiv.org/pdf/2603.01224)  

**Abstract**: Pre-trained general-purpose Vision-Language Models (VLM) hold the potential to enhance intuitive human-machine interactions due to their rich world knowledge and 2D object detection capabilities. However, VLMs for 3D coordinates detection tasks are rare. In this work, we investigate interactive abilities of VLMs by returning 3D object positions given a monocular RGB image from a wrist-mounted camera, natural language input, and robot states. We collected and curated a heterogeneous dataset of more than 100,000 images and finetuned a VLM using QLoRA with a custom regression head. By implementing conditional routing, our model maintains its ability to process general visual queries while adding specialized 3D position estimation capabilities. Our results demonstrate robust predictive performance with a median MAE of 13 mm on the test set and a five-fold improvement over a simpler baseline without finetuning. In about 25% of the cases, predictions are within a range considered acceptable for the robot to interact with objects. 

---
# Communication-Efficient Quantum Federated Learning over Large-Scale Wireless Networks 

**Authors**: Shaba Shaon, Christopher G. Brinton, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.01222)  

**Abstract**: Quantum federated learning (QFL) combines the robust data processing of quantum computing with the privacy-preserving features of federated learning (FL). However, in large-scale wireless networks, optimizing sum-rate is crucial for unlocking the true potential of QFL, facilitating effective model sharing and aggregation as devices compete for limited bandwidth amid dynamic channel conditions and fluctuating power resources. This paper studies a novel sum-rate maximization problem within a muti-channel QFL framework, specifically designed for non-orthogonal multiple access (NOMA)-based large-scale wireless networks. We develop a sum-rate maximization problem by jointly considering quantum device's channel selection and transmit power. Our formulated problem is a non-convex, mixed-integer nonlinear programming (MINLP) challenge that remains non-deterministic polynomial time (NP)-hard even with specified channel selection parameters. The complexity of the problem motivates us to create an effective iterative optimization approach that utilizes the sophisticated quantum approximate optimization algorithm (QAOA) to derive high-quality approximate solutions. Additionally, our study presents the first theoretical exploration of QFL convergence properties under full device participation, rigorously analyzing real-world scenarios with nonconvex loss functions, diverse data distributions, and the effects of quantum shot noise. Extensive simulation results indicate that our multi-channel NOMA-based QFL framework enhances model training and convergence behavior, surpassing conventional algorithms in terms of accuracy and loss. Moreover, our quantum-centric joint optimization approach achieves more than a 100% increase in sum-rate while ensuring rapid convergence, significantly outperforming the state-of-the-arts. 

---
# VisNec: Measuring and Leveraging Visual Necessity for Multimodal Instruction Tuning 

**Authors**: Mingkang Dong, Hongyi Cai, Jie Li, Sifan Zhou, Bin Ren, Kunyu Peng, Yuqian Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01195)  

**Abstract**: The effectiveness of multimodal instruction tuning depends not only on dataset scale, but critically on whether training samples genuinely require visual reasoning. However, existing instruction datasets often contain a substantial portion of visually redundant samples (solvable from text alone), as well as multimodally misaligned supervision that can degrade learning. To address this, we propose VisNec (Visual Necessity Score), a principled data selection framework that measures the marginal contribution of visual input during instruction tuning. By comparing predictive loss with and without visual context, VisNec identifies whether a training instance is vision-critical, redundant, or misaligned. To preserve task diversity, we combine VisNec with semantic clustering and select high-necessity samples within each cluster. Across 10 downstream benchmarks, training on only 15% of the LLaVA-665K dataset selected by VisNec achieves 100.2% of full-data performance. On the smaller Vision-Flan-186K dataset, our selection not only further reduces data size but also surpasses full-data training by 15.8%. These results demonstrate that measuring and leveraging visual necessity provides an effective solution for both efficient and robust multimodal instruction tuning. Codes and selected subsets will be released upon acceptance. 

---
# Token-level Data Selection for Safe LLM Fine-tuning 

**Authors**: Yanping Li, Zhening Liu, Zijian Li, Zehong Lin, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01185)  

**Abstract**: Fine-tuning large language models (LLMs) on custom datasets has become a standard approach for adapting these models to specific domains and applications. However, recent studies have shown that such fine-tuning can lead to significant degradation in the model's safety. Existing defense methods operate at the sample level and often suffer from an unsatisfactory trade-off between safety and utility. To address this limitation, we perform a systematic token-level diagnosis of safety degradation during fine-tuning. Based on this, we propose token-level data selection for safe LLM fine-tuning (TOSS), a novel framework that quantifies the safety risk of each token by measuring the loss difference between a safety-degraded model and a utility-oriented model. This token-level granularity enables accurate identification and removal of unsafe tokens, thereby preserving valuable task-specific information. In addition, we introduce a progressive refinement strategy, TOSS-Pro, which iteratively enhances the safety-degraded model's ability to identify unsafe tokens. Extensive experiments demonstrate that our approach robustly safeguards LLMs during fine-tuning while achieving superior downstream task performance, significantly outperforming existing sample-level defense methods. Our code is available at this https URL. 

---
# Scaling of learning time for high dimensional inputs 

**Authors**: Carlos Stein Brito  

**Link**: [PDF](https://arxiv.org/pdf/2603.01184)  

**Abstract**: Representation learning from complex data typically involves models with a large number of parameters, which in turn require large amounts of data samples. In neural network models, model complexity grows with the number of inputs to each neuron, with a trade-off between model expressivity and learning time. A precise characterization of this trade-off would help explain the connectivity and learning times observed in artificial and biological networks. We present a theoretical analysis of how learning time depends on input dimensionality for a Hebbian learning model performing independent component analysis. Based on the geometry of high-dimensional spaces, we show that the learning dynamics reduce to a unidimensional problem, with learning times dependent only on initial conditions. For higher input dimensions, initial parameters have smaller learning gradients and larger learning times. We find that learning times have supralinear scaling, becoming quickly prohibitive for high input dimensions. These results reveal a fundamental limitation for learning in high dimensions and help elucidate how the optimal design of neural networks depends on data complexity. Our approach outlines a new framework for analyzing learning dynamics and model complexity in neural network models. 

---
# ATLAS: AI-Assisted Threat-to-Assertion Learning for System-on-Chip Security Verification 

**Authors**: Ishraq Tashdid, Kimia Tasnia, Alexander Garcia, Jonathan Valamehr, Sazadur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2603.01170)  

**Abstract**: This work presents ATLAS, an LLM-driven framework that bridges standardized threat modeling and property-based formal verification for System-on-Chip (SoC) security. Starting from vulnerability knowledge bases such as Common Weakness Enumeration (CWE), ATLAS identifies SoC-specific assets, maps relevant weaknesses, and generates assertion-based security properties and JasperGold scripts for verification. By combining asset-centric analysis with standardized threat model templates and multi-source SoC context, ATLAS automates the transformation from vulnerability reasoning to formal proof. Evaluated on three HACK@DAC benchmarks, ATLAS detected 39/48 CWEs and generated correct properties for 33 of those bugs, advancing automated, knowledge-driven SoC security verification toward a secure-by-design paradigm. 

---
# TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization 

**Authors**: Sumin Kim, Hyemin Jeong, Mingu Kang, Yejin Kim, Yoori Oh, Joonseok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.01169)  

**Abstract**: The exponential growth of video content necessitates effective video summarization to efficiently extract key information from long videos. However, current approaches struggle to fully comprehend complex videos, primarily because they employ static or modality-agnostic fusion strategies. These methods fail to account for the dynamic, frame-dependent variations in modality saliency inherent in video data. To overcome these limitations, we propose TripleSumm, a novel architecture that adaptively weights and fuses the contributions of visual, text, and audio modalities at the frame level. Furthermore, a significant bottleneck for research into multimodal video summarization has been the lack of comprehensive benchmarks. Addressing this bottleneck, we introduce MoSu (Most Replayed Multimodal Video Summarization), the first large-scale benchmark that provides all three modalities. Extensive experiments demonstrate that TripleSumm achieves state-of-the-art performance, outperforming existing methods by a significant margin on four benchmarks, including MoSu. Our code and dataset are available at this https URL. 

---
# SphUnc: Hyperspherical Uncertainty Decomposition and Causal Identification via Information Geometry 

**Authors**: Rong Fu, Chunlei Meng, Jinshuo Liu, Dianyu Zhao, Yongtai Liu, Yibo Meng, Xiaowen Ma, Wangyu Wu, Yangchen Zeng, Kangning Cui, Shuaishuai Cao, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2603.01168)  

**Abstract**: Reliable decision-making in complex multi-agent systems requires calibrated predictions and interpretable uncertainty. We introduce SphUnc, a unified framework combining hyperspherical representation learning with structural causal modeling. The model maps features to unit hypersphere latents using von Mises-Fisher distributions, decomposing uncertainty into epistemic and aleatoric components through information-geometric fusion. A structural causal model on spherical latents enables directed influence identification and interventional reasoning via sample-based simulation. Empirical evaluations on social and affective benchmarks demonstrate improved accuracy, better calibration, and interpretable causal signals, establishing a geometric-causal foundation for uncertainty-aware reasoning in multi-agent settings with higher-order interactions. 

---
# GRAD-Former: Gated Robust Attention-based Differential Transformer for Change Detection 

**Authors**: Durgesh Ameta, Ujjwal Mishra, Praful Hambarde, Amit Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2603.01161)  

**Abstract**: Change detection (CD) in remote sensing aims to identify semantic differences between satellite images captured at different times. While deep learning has significantly advanced this field, existing approaches based on convolutional neural networks (CNNs), transformers and Selective State Space Models (SSMs) still struggle to precisely delineate change regions. In particular, traditional transformer-based methods suffer from quadratic computational complexity when applied to very high-resolution (VHR) satellite images and often perform poorly with limited training data, leading to under-utilization of the rich spatial information available in VHR imagery. We present GRAD-Former, a novel framework that enhances contextual understanding while maintaining efficiency through reduced model size. The proposed framework consists of a novel encoder with Adaptive Feature Relevance and Refinement (AFRAR) module, fusion and decoder blocks. AFRAR integrates global-local contextual awareness through two proposed components: the Selective Embedding Amplification (SEA) module and the Global-Local Feature Refinement (GLFR) module. SEA and GLFR leverage gating mechanisms and differential attention, respectively, which generates multiple softmax heaps to capture important features while minimizing the captured irreverent features. Multiple experiments across three challenging CD datasets (LEVIR-CD, CDD, DSIFN-CD) demonstrate GRAD-Former's superior performance compared to existing approaches. Notably, GRAD-Former outperforms the current state-of-the-art models across all the metrics and all the datasets while using fewer parameters. Our framework establishes a new benchmark for remote sensing change detection performance. Our code will be released at: this https URL 

---
# TC-SSA: Token Compression via Semantic Slot Aggregation for Gigapixel Pathology Reasoning 

**Authors**: Zhuo Chen, Shawn Young, Lijian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01143)  

**Abstract**: The application of large vision-language models to computational pathology holds great promise for diagnostic assistants but faces a critical computational bottleneck: the gigapixel scale of Whole Slide Images (WSIs). A single WSI typically contains over 105 patches, creating sequence lengths that exceed the constraints of standard Transformer architectures. Existing solutions often resort to spatial sampling, which risks discarding diagnostically critical evidence. To address this, we propose TC-SSA (Token Compression via Semantic Slot Aggregation), a learnable token compression framework that aggregates patch features into a fixed number of semantic slots. A gated routing module assigns patches to slots using sparse Top-2 routing, followed by weighted aggregation, enabling global slide coverage under a strict token budget. The resulting representation retains diagnostically relevant information while reducing the number of visual tokens to 1.7% of the original sequence. On the SlideBench(TCGA), our model achieves 78.34% overall accuracy and 77.14% on the diagnosis subset, outperforming sampling-based baselines under comparable token budgets. The method also generalizes to MIL classification, reaching AUC of 95.83% on TCGA-BRCA, 98.27% on TCGA-NSCLC and 79.80% on PANDA. These results suggest that learnable semantic aggregation provides an effective trade-off between efficiency and diagnostic performance for gigapixel pathology reasoning. 

---
# A Deep Learning Framework for Heat Demand Forecasting using Time-Frequency Representations of Decomposed Features 

**Authors**: Adithya Ramachandran, Satyaki Chatterjee, Thorkil Flensmark B. Neergaard, Maximilian Oberndoerfer, Andreas Maier, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2603.01137)  

**Abstract**: District Heating Systems are essential infrastructure for delivering heat to consumers across a geographic region sustainably, yet efficient management relies on optimizing diverse energy sources, such as wood, gas, electricity, and solar, in response to fluctuating demand. Aligning supply with demand is critical not only for ensuring reliable heat distribution but also for minimizing carbon emissions and extending infrastructure lifespan through lower operating temperatures. However, accurate multi-step forecasting to support these goals remains challenging due to complex, non-linear usage patterns and external dependencies. In this work, we propose a novel deep learning framework for day-ahead heat demand prediction that leverages time-frequency representations of historical data. By applying Continuous Wavelet Transform to decomposed demand and external meteorological factors, our approach enables Convolutional Neural Networks to learn hierarchical temporal features that are often inaccessible to standard time domain models. We systematically evaluate this method against statistical baselines, state-of-the-art Transformers, and emerging foundation models using multi-year data from three distinct Danish districts, a Danish city, and a German city. The results show a significant advancement, reducing the Mean Absolute Error by 36% to 43% compared to the strongest baselines, achieving forecasting accuracy of up to 95% across annual test datasets. Qualitative and statistical analyses further confirm the accuracy and robustness by reliably tracking volatile demand peaks where others fail. This work contributes both a high-performance forecasting architecture and critical insights into optimal feature composition, offering a validated solution for modern energy applications. 

---
# MedCollab: Causal-Driven Multi-Agent Collaboration for Full-Cycle Clinical Diagnosis via IBIS-Structured Argumentation 

**Authors**: Yuqi Zhan, Xinyue Wu, Tianyu Lin, Yutong Bao, Xiaoyu Wang, Weihao Cheng, Huangwei Chen, Feiwei Qin, Zhu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01131)  

**Abstract**: Large language models (LLMs) have shown promise in healthcare applications, however, their use in clinical practice is still limited by diagnostic hallucinations and insufficiently interpretable reasoning. We present MedCollab, a novel multi-agent framework that emulates the hierarchical consultation workflow of modern hospitals to autonomously navigate the full-cycle diagnostic process. The framework incorporates a dynamic specialist recruitment mechanism that adaptively assembles clinical and examination agents according to patient-specific symptoms and examination results. To ensure the rigor of clinical work, we adopt a structured Issue-Based Information System (IBIS) argumentation protocol that requires agents to provide ``Positions'' backed by traceable evidence from medical knowledge and clinical data. Furthermore, the framework constructs a Hierarchical Disease Causal Chain that transforms flattened diagnostic predictions into a structured model of pathological progression through explicit logical operators. A multi-round Consensus Mechanism iteratively filters low-quality reasoning through logic auditing and weighted voting. Evaluated on real-world clinical datasets, MedCollab significantly outperforms pure LLMs and medical multi-agent systems in Accuracy and RaTEScore, demonstrating a marked reduction in medical hallucinations. These findings indicate that MedCollab provides an extensible, transparent, and clinically compliant approach to medical decision-making. 

---
# Predictive Reasoning with Augmented Anomaly Contrastive Learning for Compositional Visual Relations 

**Authors**: Chengtai Li, Yuting He, Jianfeng Ren, Ruibin Bai, Yitian Zhao, Heng Yu, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01125)  

**Abstract**: While visual reasoning for simple analogies has received significant attention, compositional visual relations (CVR) remain relatively unexplored due to their greater complexity. To solve CVR tasks, we propose Predictive Reasoning with Augmented Anomaly Contrastive Learning (PR-A$^2$CL), \ie, to identify an outlier image given three other images that follow the same compositional rules. To address the challenge of modelling abundant compositional rules, an Augmented Anomaly Contrastive Learning is designed to distil discriminative and generalizable features by maximizing similarity among normal instances while minimizing similarity between normal and anomalous outliers. More importantly, a predict-and-verify paradigm is introduced for rule-based reasoning, in which a series of Predictive Anomaly Reasoning Blocks (PARBs) iteratively leverage features from three out of the four images to predict those of the remaining one. Throughout the subsequent verification stage, the PARBs progressively pinpoint the specific discrepancies attributable to the underlying rules. Experimental results on SVRT, CVR and MC$^2$R datasets show that PR-A$^2$CL significantly outperforms state-of-the-art reasoning models. 

---
# ClinCoT: Clinical-Aware Visual Chain-of-Thought for Medical Vision Language Models 

**Authors**: Xiwei Liu, Yulong Li, Xinlin Zhuang, Xuhui Li, Jianxu Chen, Haolin Yang, Imran Razzak, Yutong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.01124)  

**Abstract**: Medical Vision-Language Models have shown promising potential in clinical decision support, yet they remain prone to factual hallucinations due to insufficient grounding in localized pathological evidence. Existing medical alignment methods primarily operate at the response level through preference optimization, improving output correctness but leaving intermediate reasoning weakly connected to visual regions. Although chain-of-thought (CoT) enhances multimodal reasoning, it remains largely text-centric, limiting effective integration of clinical visual cues. To address this gap, we propose ClinCoT, a clinical-aware visual chain-of-thought framework that transforms preference optimization from response-level correction to visual-driven reasoning. We introduce an automatic data generation pipeline that constructs clinically grounded preference pairs through reasoning with hypotheses-driven region proposals. Multiple Med-LLMs evaluators rank and assign scores to each response, and these rankings serve as supervision to train the target model. We further introduce a scoring-based margin-aware optimization strategy that incorporates both preference ranking and score difference to refine region-level reasoning trajectories. To maintain alignment as the model's policy evolves during training, we adopt an iterative learning scheme that dynamically regenerates preference data. Extensive experiments on three medical VQA and report generation benchmarks demonstrate that ClinCoT consistently improves factual grounding and achieves superior performance compared with existing preference-based alignment methods. 

---
# Robust Weighted Triangulation of Causal Effects Under Model Uncertainty 

**Authors**: Rohit Bhattacharya, Ina Ocelli, Ted Westling  

**Link**: [PDF](https://arxiv.org/pdf/2603.01119)  

**Abstract**: A fundamental challenge in causal inference with observational data is correct specification of a causal model. When there is model uncertainty, analysts may seek to use estimates from multiple candidate models that rely on distinct, and possibly partially overlapping, sets of identifying assumptions to infer the causal effect, a process known as triangulation. Principled methods for triangulation, however, remain underdeveloped. Here, we develop a framework for causal effect triangulation that combines model testability methods from causal discovery with statistical inference methods from semiparametric theory, while avoiding explicit model selection and post-selection inference problems. We propose a triangulation functional that combines identified functionals from each model with data-driven measures of model validity. We provide a bound on the distance of the functional from the true causal effect along with conditions under which this distance can be taken to zero. Finally, we derive valid statistical inference for this functional. Our framework formalizes robustness under causal pluralism without requiring agreement across models or commitment to a single specification. We demonstrate its performance through simulations and an empirical application. 

---
# Egocentric Co-Pilot: Web-Native Smart-Glasses Agents for Assistive Egocentric AI 

**Authors**: Sicheng Yang, Yukai Huang, Weitong Cai, Shitong Sun, Fengyi Fang, You He, Yiqiao Xie, Jiankang Deng, Hang Zhang, Jifei Song, Zhensong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01104)  

**Abstract**: What if accessing the web did not require a screen, a stable desk, or even free hands? For people navigating crowded cities, living with low vision, or experiencing cognitive overload, smart glasses coupled with AI agents could turn the web into an always-on assistive layer over daily life. We present Egocentric Co-Pilot, a web-native neuro-symbolic framework that runs on smart glasses and uses a Large Language Model (LLM) to orchestrate a toolbox of perception, reasoning, and web tools. An egocentric reasoning core combines Temporal Chain-of-Thought with Hierarchical Context Compression to support long-horizon question answering and decision support over continuous first-person video, far beyond a single model's context window. Additionally, a lightweight multimodal intent layer maps noisy speech and gaze into structured commands. We further implement and evaluate a cloud-native WebRTC pipeline integrating streaming speech, video, and control messages into a unified channel for smart glasses and browsers. In parallel, we deploy an on-premise WebSocket baseline, exposing concrete trade-offs between local inference and cloud offloading in terms of latency, mobility, and resource use. Experiments on Egolife and HD-EPIC demonstrate competitive or state-of-the-art egocentric QA performance, and a human-in-the-loop study on smart glasses shows higher task completion and user satisfaction than leading commercial baselines. Taken together, these results indicate that web-connected egocentric co-pilots can be a practical path toward more accessible, context-aware assistance in everyday life. By grounding operation in web-native communication primitives and modular, auditable tool use, Egocentric Co-Pilot offers a concrete blueprint for assistive, always-on web agents that support education, accessibility, and social inclusion for people who may benefit most from contextual, egocentric AI. 

---
# SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation 

**Authors**: Hongrui Wang, Fan Zhang, Zhiyuan Yu, Ziya Zhou, Xi Chen, Can Yang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01101)  

**Abstract**: Multi-track music generation has garnered significant research interest due to its precise mixing and remixing capabilities. However, existing models often overlook essential attributes such as rhythmic stability and synchronization, leading to a focus on differences between tracks rather than their inherent properties. In this paper, we introduce SyncTrack, a synchronous multi-track waveform music generation model designed to capture the unique characteristics of multi-track music. SyncTrack features a novel architecture that includes track-shared modules to establish a common rhythm across all tracks and track-specific modules to accommodate diverse timbres and pitch ranges. Each track-shared module employs two cross-track attention mechanisms to synchronize rhythmic information, while each track-specific module utilizes learnable instrument priors to better represent timbre and other unique features. Additionally, we enhance the evaluation of multi-track music quality by introducing rhythmic consistency through three novel metrics: Inner-track Rhythmic Stability (IRS), Cross-track Beat Synchronization (CBS), and Cross-track Beat Dispersion (CBD). Experiments demonstrate that SyncTrack significantly improves the multi-track music quality by enhancing rhythmic consistency. 

---
# Differential privacy representation geometry for medical image analysis 

**Authors**: Soroosh Tayebi Arasteh, Marziyeh Mohammadi, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2603.01098)  

**Abstract**: Differential privacy (DP)'s effect in medical imaging is typically evaluated only through end-to-end performance, leaving the mechanism of privacy-induced utility loss unclear. We introduce Differential Privacy Representation Geometry for Medical Imaging (DP-RGMI), a framework that interprets DP as a structured transformation of representation space and decomposes performance degradation into encoder geometry and task-head utilization. Geometry is quantified by representation displacement from initialization and spectral effective dimension, while utilization is measured as the gap between linear-probe and end-to-end utility. Across over 594,000 images from four chest X-ray datasets and multiple pretrained initializations, we show that DP is consistently associated with a utilization gap even when linear separability is largely preserved. At the same time, displacement and spectral dimension exhibit non-monotonic, initialization- and dataset-dependent reshaping, indicating that DP alters representation anisotropy rather than uniformly collapsing features. Correlation analysis reveals that the association between end-to-end performance and utilization is robust across datasets but can vary by initialization, while geometric quantities capture additional prior- and dataset-conditioned variation. These findings position DP-RGMI as a reproducible framework for diagnosing privacy-induced failure modes and informing privacy model selection. 

---
# Unified Vision-Language Modeling via Concept Space Alignment 

**Authors**: Yifu Qiu, Paul-Ambroise Duquenne, Holger Schwenk  

**Link**: [PDF](https://arxiv.org/pdf/2603.01096)  

**Abstract**: We introduce V-SONAR, a vision-language embedding space extended from the text-only embedding space SONAR (Omnilingual Embeddings Team et al., 2026), which supports 1500 text languages and 177 speech languages. To construct V-SONAR, we propose a post-hoc alignment pipeline that maps the representations of an existing vision encoder into the SONAR space. We thoroughly evaluate V-SONAR and show that its embeddings achieve competitive performance on text-to-video retrieval. Equipped with the OMNISONAR text decoder, V-SONAR further surpasses state-of-the-art vision-language models on video captioning tasks, including DREAM-1K (BLEU 23.9 vs. 19.6) and PE-VIDEO (BLEU 39.0 vs. 30.0).
Leveraging V-SONAR, we first demonstrate that the Large Concept Model (LCM; LCM team et al. 2024) operating in SONAR and trained with English text only, can perform both single- and multi-visual concept understanding in a zero-shot manner. Finally, we introduce V-LCM, which extends the LCM with vision-language instruction tuning. V-LCM encodes vision and language inputs into an unified sequence of latent embeddings via V-SONAR and SONAR, and it is trained with the same latent diffusion objective for next-embedding prediction as in LCM's text-only pre-training. Experiments on a large-scale multilingual and -modal instruction-tuning data mixture highlight the potential of V-LCM: V-LCM matches state-of-the-art vision-language models on tasks covering image/video captioning and question answering, while significantly outperforming them across 61 rich- to low-resource languages out of all 62 tested languages. 

---
# Feasible Pairings for Decentralized Integral Controllability of Non-Square Systems 

**Authors**: Yuhao Tong, Steven W. Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.01076)  

**Abstract**: This paper investigates the determination of feasible input-output pairings for the decentralized integral controllability of non-square systems. The relevance of this problem extends beyond traditional industrial processes into modern AI research, particularly Multi-Agent Reinforcement Learning (MARL), where environments frequently act as strongly non-square mappings that evaluate high-dimensional joint action spaces via comparatively low-dimensional global rewards. To address the stability of these complex distributed architectures, we extend the concept of D-stability to non-square matrices, providing a crucial mathematical foundation. We formally define D-stability for non-square matrices as a direct generalization of the square case. By introducing the concept of ``Squared Matrices'', which are derived from specific column selections of the non-square formulation and directly correspond to candidate control pairings, we establish a fundamental link between the stability of these square sub-components and the original non-square system. Ultimately, we propose sufficient conditions under which the individual Volterra-Lyapunov stability of these squared components guarantees the extended D-stability of the non-square matrix, thereby providing a rigorous method to identify feasible pairings that ensure robust decentralized control across both classical and data-driven applications. 

---
# Hide&Seek: Remove Image Watermarks with Negligible Cost via Pixel-wise Reconstruction 

**Authors**: Huajie Chen, Tianqing Zhu, Hailin Yang, Yuchen Zhong, Yang Zhang, Hui Sun, Heng Xu, Zuobin Ying, Lihua Yin, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.01067)  

**Abstract**: Watermarking has emerged as a key defense against the misuse of machine-generated images (MGIs). Yet the robustness of these protections remains underexplored. To reveal the limits of SOTA proactive image watermarking defenses, we propose HIDE&SEEK (HS), a suite of versatile and cost-effective attacks that reliably remove embedded watermarks while preserving high visual fidelity. 

---
# TriMoE: Augmenting GPU with AMX-Enabled CPU and DIMM-NDP for High-Throughput MoE Inference via Offloading 

**Authors**: Yudong Pan, Yintao He, Tianhua Han, Lian Liu, Shixin Zhao, Zhirong Chen, Mengdi Wang, Cangyuan Li, Yinhe Han, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01058)  

**Abstract**: To deploy large Mixture-of-Experts (MoE) models cost-effectively, offloading-based single-GPU heterogeneous inference is crucial. While GPU-CPU architectures that offload cold experts are constrained by host memory bandwidth, emerging GPU-NDP architectures utilize DIMM-NDP to offload non-hot experts. However, non-hot experts are not a homogeneous memory-bound group: a significant subset of warm experts exists is severely penalized by high GPU I/O latency yet can saturate NDP compute throughput, exposing a critical compute gap. We present TriMoE, a novel GPU-CPU-NDP architecture that fills this gap by synergistically leveraging AMX-enabled CPU to precisely map hot, warm, and cold experts onto their optimal compute units. We further introduce a bottleneck-aware expert scheduling policy and a prediction-driven dynamic relayout/rebalancing scheme. Experiments demonstrate that TriMoE achieves up to 2.83x speedup over state-of-the-art solutions. 

---
# Turning Black Box into White Box: Dataset Distillation Leaks 

**Authors**: Huajie Chen, Tianqing Zhu, Yuchen Zhong, Yang Zhang, Shang Wang, Feng He, Lefeng Zhang, Jialiang Shen, Minghao Wang, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.01053)  

**Abstract**: Dataset distillation compresses a large real dataset into a small synthetic one, enabling models trained on the synthetic data to achieve performance comparable to those trained on the real data. Although synthetic datasets are assumed to be privacy-preserving, we show that existing distillation methods can cause severe privacy leakage because synthetic datasets implicitly encode the weight trajectories of the distilled model, they become over-informative and exploitable by adversaries. To expose this risk, we introduce the Information Revelation Attack (IRA) against state-of-the-art distillation techniques. Experiments show that IRA accurately predicts both the distillation algorithm and model architecture, and can successfully infer membership and recover sensitive samples from the real dataset. 

---
# MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline 

**Authors**: Huanjin Yao, Qixiang Yin, Min Yang, Ziwang Zhao, Yibo Wang, Haotian Luo, Jingyi Zhang, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01050)  

**Abstract**: We aim to develop a multimodal research agent capable of explicit reasoning and planning, multi-tool invocation, and cross-modal information synthesis, enabling it to conduct deep research tasks. However, we observe three main challenges in developing such agents: (1) scarcity of search-intensive multimodal QA data, (2) lack of effective search trajectories, and (3) prohibitive cost of training with online search APIs. To tackle them, we first propose Hyper-Search, a hypergraph-based QA generation method that models and connects visual and textual nodes within and across modalities, enabling to generate search-intensive multimodal QA pairs that require invoking various search tools to solve. Second, we introduce DR-TTS, which first decomposes search-involved tasks into several categories according to search tool types, and respectively optimize specialized search tool experts for each tool. It then recomposes tool experts to jointly explore search trajectories via tree search, producing trajectories that successfully solve complex tasks using various search tools. Third, we build an offline search engine supporting multiple search tools, enabling agentic reinforcement learning without using costly online search APIs. With the three designs, we develop MM-DeepResearch, a powerful multimodal deep research agent, and extensive results shows its superiority across benchmarks. Code is available at this https URL 

---
# RepoRepair: Leveraging Code Documentation for Repository-Level Automated Program Repair 

**Authors**: Zhongqiang Pan, Chuanyi Li, Wenkang Zhong, Yi Feng, Bin Luo, Vincent Ng  

**Link**: [PDF](https://arxiv.org/pdf/2603.01048)  

**Abstract**: Automated program repair (APR) struggles to scale from isolated functions to full repositories, as it demands a global, task-aware understanding to locate necessary changes. Current methods, limited by context and reliant on shallow retrieval or costly agent iterations, falter on complex cross-file issues. To this end, we propose RepoRepair, a novel documentation-enhanced approach for repository-level fault localization and program repair. Our core insight is to leverage LLMs to generate hierarchical code documentation (from functions to files) for code repositories, creating structured semantic abstractions that enable LLMs to comprehend repository-level context and dependencies. Specifically, RepoRepair first employs a text-based LLM (e.g., DeepSeek-V3) to generate file/function-level code documentation for repositories, which serves as auxiliary knowledge to guide fault localization. Subsequently, based on the fault localization results and the issue description, a powerful LLM (e.g., Claude-4) attempts to repair the identified suspicious code snippets. Evaluated on SWE-bench Lite, RepoRepair achieves a 45.7% repair rate at a low cost of $0.44 per fix. On SWE-bench Multimodal, it delivers state-of-the-art performance with a 37.1% repair rate despite a higher cost of $0.56 per fix, demonstrating robust and cost-effective performance across diverse problem domains. 

---
# Silo-Bench: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems 

**Authors**: Yuzhe Zhang, Feiran Liu, Yi Shan, Xinyi Huang, Xin Yang, Yueqi Zhu, Xuxin Cheng, Cao Liu, Ke Zeng, Terry Jingchen Zhang, Wenyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01045)  

**Abstract**: Large language models are increasingly deployed in multi-agent systems to overcome context limitations by distributing information across agents. Yet whether agents can reliably compute with distributed information -- rather than merely exchange it -- remains an open question. We introduce Silo-Bench, a role-agnostic benchmark of 30 algorithmic tasks across three communication complexity levels, evaluating 54 configurations over 1,620 experiments. Our experiments expose a fundamental Communication-Reasoning Gap: agents spontaneously form task-appropriate coordination topologies and exchange information actively, yet systematically fail to synthesize distributed state into correct answers. The failure is localized to the reasoning-integration stage -- agents often acquire sufficient information but cannot integrate it. This coordination overhead compounds with scale, eventually eliminating parallelization gains entirely. These findings demonstrate that naively scaling agent count cannot circumvent context limitations, and Silo-Bench provides a foundation for tracking progress toward genuinely collaborative multi-agent systems. 

---
# Thoth: Mid-Training Bridges LLMs to Time Series Understanding 

**Authors**: Jiafeng Lin, Yuxuan Wang, Jialong Wu, Huakun Luo, Zhongyi Pei, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01042)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable success in general-purpose reasoning. However, they still struggle to understand and reason about time series data, which limits their effectiveness in decision-making scenarios that depend on temporal dynamics. In this paper, we propose Thoth, the first family of mid-trained LLMs with general-purpose time series understanding capabilities. As a pivotal intermediate stage, mid-training achieves task- and domain-agnostic alignment between time series and natural language, for which we construct Book-of-Thoth, a high-quality, time-series-centric mid-training corpus. Book-of-Thoth enables both time-series-to-text and text-to-time-series generation, equipping LLMs with a foundational grasp of temporal patterns. To better evaluate advanced reasoning capabilities, we further present KnoTS, a novel benchmark of knowledge-intensive time series understanding, designed for joint reasoning over temporal patterns and domain knowledge. Extensive experiments demonstrate that mid-training with Book-of-Thoth enables Thoth to significantly outperform its base model and advanced LLMs across a range of time series question answering benchmarks. Moreover, Thoth exhibits superior capabilities when fine-tuned under data scarcity, underscoring the effectiveness of mid-training for time series understanding. Code is available at: this https URL. 

---
# From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing 

**Authors**: Haoyuan Zhang, Keyao Wang, Guosheng Zhang, Haixiao Yue, Zhiwen Tan, Siran Peng, Tianshuo Zhang, Xiao Tan, Kunbin Chen, Wei He, Jingdong Wang, Ajian Liu, Xiangyu Zhu, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2603.01038)  

**Abstract**: Face recognition remains vulnerable to presentation attacks, calling for robust Face Anti-Spoofing (FAS) solutions. Recent MLLM-based FAS methods reformulate the binary classification task as the generation of brief textual descriptions to improve cross-domain generalization. However, their generalizability is still limited, as such descriptions mainly capture intuitive semantic cues (e.g., mask contours) while struggling to perceive fine-grained visual patterns. To address this limitation, we incorporate external visual tools into MLLMs to encourage deeper investigation of subtle spoof clues. Specifically, we propose the Tool-Augmented Reasoning FAS (TAR-FAS) framework, which reformulates the FAS task as a Chain-of-Thought with Visual Tools (CoT-VT) paradigm, allowing MLLMs to begin with intuitive observations and adaptively invoke external visual tools for fine-grained investigation. To this end, we design a tool-augmented data annotation pipeline and construct the ToolFAS-16K dataset, which contains multi-turn tool-use reasoning trajectories. Furthermore, we introduce a tool-aware FAS training pipeline, where Diverse-Tool Group Relative Policy Optimization (DT-GRPO) enables the model to autonomously learn efficient tool use. Extensive experiments under a challenging one-to-eleven cross-domain protocol demonstrate that TAR-FAS achieves SOTA performance while providing fine-grained visual investigation for trustworthy spoof detection. 

---
# Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery 

**Authors**: Yangyang Xu, Junbo Ke, You-Wei Wen, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01034)  

**Abstract**: Tensor Ring (TR) decomposition is a powerful tool for high-order data modeling, but is inherently restricted to discrete forms defined on fixed meshgrids. In this work, we propose a TR functional decomposition for both meshgrid and non-meshgrid data, where factors are parameterized by Implicit Neural Representations (INRs). However, optimizing this continuous framework to capture fine-scale details is intrinsically difficult. Through a frequency-domain analysis, we demonstrate that the spectral structure of TR factors determines the frequency composition of the reconstructed tensor and limits the high-frequency modeling capacity. To mitigate this, we propose a reparameterized TR functional decomposition, in which each TR factor is a structured combination of a learnable latent tensor and a fixed basis. This reparameterization is theoretically shown to improve the training dynamics of TR factor learning. We further derive a principled initialization scheme for the fixed basis and prove the Lipschitz continuity of our proposed model. Extensive experiments on image inpainting, denoising, super-resolution, and point cloud recovery demonstrate that our method achieves consistently superior performance over existing approaches. Code is available at this https URL. 

---
# Content-Aware Frequency Encoding for Implicit Neural Representations with Fourier-Chebyshev Features 

**Authors**: Junbo Ke, Yangyang Xu, You-Wei Wen, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01028)  

**Abstract**: Implicit Neural Representations (INRs) have emerged as a powerful paradigm for various signal processing tasks, but their inherent spectral bias limits the ability to capture high-frequency details. Existing methods partially mitigate this issue by using Fourier-based features, which usually rely on fixed frequency bases. This forces multi-layer perceptrons (MLPs) to inefficiently compose the required frequencies, thereby constraining their representational capacity. To address this limitation, we propose Content-Aware Frequency Encoding (CAFE), which builds upon Fourier features through multiple parallel linear layers combined via a Hadamard product. CAFE can explicitly and efficiently synthesize a broader range of frequency bases, while the learned weights enable the selection of task-relevant frequencies. Furthermore, we extend this framework to CAFE+, which incorporates Chebyshev features as a complementary component to Fourier bases. This combination provides a stronger and more stable frequency representation. Extensive experiments across multiple benchmarks validate the effectiveness and efficiency of our approach, consistently achieving superior performance over existing methods. Our code is available at this https URL. 

---
# One-Token Verification for Reasoning Correctness Estimation 

**Authors**: Zhan Zhuang, Xiequn Wang, Zebin Chen, Feiyang Ye, Ying Wei, Kede Ma, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01025)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have led to notable successes in complex reasoning tasks, such as mathematical problem solving. A common strategy for improving performance is parallel thinking, in which multiple reasoning traces are generated and the final prediction is made using aggregation schemes like majority voting or best-of-$N$ decoding. However, two key challenges persist. First, multi-sample decoding incurs substantial inference latency, especially for long-form outputs. Second, effective mechanisms for reliably assessing the correctness of individual reasoning traces are still limited. To address these challenges, we introduce One-Token Verification (OTV), a computational method that estimates reasoning correctness in a single forward pass during generation. OTV is activated by a learnable token and integrated into the LLM via low-rank adaptation to probe internal reasoning signals through the key-value cache, supporting token-level correctness estimation at any stage of generation without disrupting primary reasoning. Experiments on mathematical reasoning benchmarks demonstrate that OTV consistently surpasses existing verifiers. Additionally, OTV reduces token usage by up to $90\%$ through correctness-guided early termination, prioritizing shorter, more reliable solutions. 

---
# SimAB: Simulating A/B Tests with Persona-Conditioned AI Agents for Rapid Design Evaluation 

**Authors**: Tim Rieder, Marian Schneider, Mario Truss, Vitaly Tsaplin, Alina Rublea, Sinem Dere, Francisco Chicharro Sanz, Tobias Reiss, Mustafa Doga Dogan  

**Link**: [PDF](https://arxiv.org/pdf/2603.01024)  

**Abstract**: A/B testing is a standard method for validating design decisions, yet its reliance on real user traffic limits iteration speed and makes certain experiments impractical. We present SimAB, a system that reframes A/B testing as a fast, privacy-preserving simulation using persona-conditioned AI agents. Given design screenshots and a conversion goal, SimAB generates user personas, deploys them as agents that state their preference, aggregates results, and synthesizes rationales. Through a formative study with experimentation practitioners, we identified scenarios where traffic constraints hinder testing, including low-traffic pages, multi-variant comparisons, micro-optimizations, and privacy-sensitive contexts. Our design emphasizes speed, early feedback, actionable rationales, and audience specification. We evaluate SimAB against 47 historical A/B tests with known outcomes, achieving 67% overall accuracy, increasing to 83% for high-confidence cases. Additional experiments show robustness to naming and positional bias and demonstrate accuracy gains from personas. Practitioner feedback suggests that SimAB supports faster evaluation cycles and rapid screening of designs difficult to assess with traditional A/B tests. 

---
# An Open-Source Modular Benchmark for Diffusion-Based Motion Planning in Closed-Loop Autonomous Driving 

**Authors**: Yun Li, Simon Thompson, Yidu Zhang, Ehsan Javanmardi, Manabu Tsukada  

**Link**: [PDF](https://arxiv.org/pdf/2603.01023)  

**Abstract**: Diffusion-based motion planners have achieved state-of-the-art results on benchmarks such as nuPlan, yet their evaluation within closed-loop production autonomous driving stacks remains largely unexplored. Existing evaluations abstract away ROS 2 communication latency and real-time scheduling constraints, while monolithic ONNX deployment freezes all solver parameters at export time. We present an open-source modular benchmark that addresses both gaps: using ONNX GraphSurgeon, we decompose a monolithic 18,398 node diffusion planner into three independently executable modules and reimplement the DPM-Solver++ denoising loop in native C++. Integrated as a ROS 2 node within Autoware, the open-source AD stack deployed on real vehicles worldwide, the system enables runtime-configurable solver parameters without model recompilation and per-step observability of the denoising process, breaking the black box of monolithic deployment. Unlike evaluations in standalone simulators such as CARLA, our benchmark operates within a production-grade stack and is validated through AWSIM closed-loop simulation. Through systematic comparison of DPM-Solver++ (first- and second-order) and DDIM across six step-count configurations (N in {3, 5, 7, 10, 15, 20}), we show that encoder caching yields a 3.2x latency reduction, and that second-order solving reduces FDE by 41% at N=3 compared to first-order. The complete codebase will be released as open-source, providing a direct path from simulation benchmarks to real-vehicle deployment. 

---
# FastCode: Fast and Cost-Efficient Code Understanding and Reasoning 

**Authors**: Zhonghang Li, Zongwei Li, Yuxuan Chen, Han Shi, Jiawei Li, Jierun Chen, Haoli Bai, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01012)  

**Abstract**: Repository-scale code reasoning is a cornerstone of modern AI-assisted software engineering, enabling Large Language Models (LLMs) to handle complex workflows from program comprehension to complex debugging. However, balancing accuracy with context cost remains a significant bottleneck, as existing agentic approaches often waste computational resources through inefficient, iterative full-text exploration. To address this, we introduce \model, a framework that decouples repository exploration from content consumption. \model\ utilizes a structural scouting mechanism to navigate a lightweight semantic-structural map of the codebase, allowing the system to trace dependencies and pinpoint relevant targets without the overhead of full-text ingestion. By leveraging structure-aware navigation tools regulated by a cost-aware policy, the framework constructs high-value contexts in a single, optimized step. Extensive evaluations on the SWE-QA, LongCodeQA, LOC-BENCH, and GitTaskBench benchmarks demonstrate that \model\ consistently outperforms state-of-the-art baselines in reasoning accuracy while significantly reducing token consumption, validating the efficiency of scouting-first strategies for large-scale code reasoning. Source code is available at this https URL. 

---
# AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching 

**Authors**: Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01006)  

**Abstract**: REPresentation Alignment (REPA) improves the training of generative flow models by aligning intermediate hidden states with pretrained teacher features, but its effectiveness in token-conditioned audio Flow Matching critically depends on the choice of supervised layers, which is typically made heuristically based on the depth. In this work, we introduce Attribution-Guided REPresentation Alignment (AG-REPA), a novel causal layer selection strategy for representation alignment in audio Flow Matching. Firstly, we find that layers that best store semantic/acoustic information (high teacher-space similarity) are not necessarily the layers that contribute most to the velocity field that drives generation, and we call it Store-Contribute Dissociation (SCD). To turn this insight into an actionable training guidance, we propose a forward-only gate ablation (FoG-A) that quantifies each layer's causal contribution via the induced change in the predicted velocity field, enabling sparse layer selection and adaptive weighting for alignment. Across unified speech and general-audio training (LibriSpeech + AudioSet) under different token-conditioning topologies, AG-REPA consistently outperforms REPA baselines. Overall, our results show that alignment is most effective when applied to the causally dominant layers that drive the velocity field, rather than to layers that are representationally rich but functionally passive. 

---
# Data-Free PINNs for Compressible Flows: Mitigating Spectral Bias and Gradient Pathologies via Mach-Guided Scaling and Hybrid Convolutions 

**Authors**: Ryosuke Yano  

**Link**: [PDF](https://arxiv.org/pdf/2603.01001)  

**Abstract**: This paper presents a fully data-free Physics-Informed Neural Network (PINN) capable of solving compressible inviscid flows (ranging from supersonic to hypersonic, up to Ma=15, where Ma is the Mach number) around a circular cylinder. To overcome the spatial blindness of standard Multi-Layer Perceptrons, a structured hybrid architecture combining radial 1D convolutions with anisotropic azimuthal 2D convolutions is proposed to embed directional inductive biases. For stable optimization across disparate flow regimes, a regime-dependent, Mach-number-guided dynamic residual scaling strategy is introduced. Crucially, this approach scales down residuals to mitigate extreme gradient stiffness in high-Mach regimes, while applying penalty multipliers to overcome the inherent spectral bias and explicitly enforce weak shock discontinuities in low-supersonic flows. Furthermore, to establish a global thermodynamic anchor essential for stable shock wave capturing, exact analytical solutions at the stagnation point are embedded into the loss formulation. This is coupled with a novel "Upstream Fixing" boundary loss and a Total Variation (TV) loss to explicitly suppress upstream noise and the non-physical carbuncle phenomenon. The proposed framework successfully captures the detached bow shock without referential data. While the requisite artificial viscosity yields a slightly thicker shock wave compared to computational fluid dynamics, the proposed method demonstrates unprecedented stability and physical fidelity for data-free PINNs in extreme aerodynamics. 

---
# Fully-analog array signal processor using 3D aperture engineering 

**Authors**: Sheng Gao, Songtao Yang, Haiou Zhang, Yuan Shen, Xing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00995)  

**Abstract**: The rapid progress in radar and communication places increasing demands on low-latency and energy-efficiency array signal processing methods. There is an emerging direction of constructing analog computing processors for directly processing electromagnetic (EM) waves. However, the existing methods are constrained by 2D physical aperture and imprecise design process with inefficient computing architecture, resulting in limited sensing resolution and number of separated sources. Here, we present a fully-analog array signal processor (FASP) using 3D aperture engineering framework to perform super-resolution direction-of-arrival estimation, source number estimation, and multi-channel source separation in parallel for both coherent and incoherent sources. 3D aperture engineering is realized by constructing deep cascaded metasurface layers so that the diffractive propagation from oblique incident fields can be layer-wise modulated and piecewise encoded for perceiving EM fields far exceeding physical aperture limits. The multi-dimensional synthetic aperture (MSA) training is developed to characterize the metasurface modulation and optimize the neuro-augmented physical model for extending system aperture and generating high-order nonlinear angular response. FASP orthogonalizes the array response vectors of communication channels to map them into antenna detectors in the analog domain. The $N$-layer FASP has the capability to achieve ~N times higher angular resolution than the Rayleigh diffraction limit. Experiments further validate the source number estimation and independent channel separation of 10-target that can suppress radar jamming signals by ~20 dB and enhance channel communication capacity by 13.5 times at 36~41 GHz. FASP heralds a paradigm shift in signal processing for super-resolution optics, advanced radar, and 6G communications. 

---
# EraseAnything++: Enabling Concept Erasure in Rectified Flow Transformers Leveraging Multi-Object Optimization 

**Authors**: Zhaoxin Fan, Nanxiang Jiang, Daiheng Gao, Shiji Zhou, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00978)  

**Abstract**: Removing undesired concepts from large-scale text-to-image (T2I) and text-to-video (T2V) diffusion models while preserving overall generative quality remains a major challenge, particularly as modern models such as Stable Diffusion v3, Flux, and OpenSora employ flow-matching and transformer-based architectures and extend to long-horizon video generation. Existing concept erasure methods, designed for earlier T2I/T2V models, often fail to generalize to these paradigms. To address this issue, we propose EraseAnything++, a unified framework for concept erasure in both image and video diffusion models with flow-matching objectives. Central to our approach is formulating concept erasure as a constrained multi-objective optimization problem that explicitly balances concept removal with preservation of generative utility. To solve the resulting conflicting objectives, we introduce an efficient utility-preserving unlearning strategy based on implicit gradient surgery. Furthermore, by integrating LoRA-based parameter tuning with attention-level regularization, our method anchors erasure on key visual representations and propagates it consistently across spatial and temporal dimensions. In the video setting, we further enhance consistency through an anchor-and-propagate mechanism that initializes erasure on reference frames and enforces it throughout subsequent transformer layers, thereby mitigating temporal drift. Extensive experiments on both image and video benchmarks demonstrate that EraseAnything++ substantially outperforms prior methods in erasure effectiveness, generative fidelity, and temporal consistency, establishing a new state of the art for concept erasure in next-generation diffusion models. 

---
# Forgetting is Competition: Rethinking Unlearning as Representation Interference in Diffusion Models 

**Authors**: Ashutosh Ranjan, Vivek Srivastava, Shirish Karande, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2603.00975)  

**Abstract**: Unlearning in text-to-image diffusion models often leads to uneven concept removal and unintended forgetting of unrelated capabilities. This complicates tasks such as copyright compliance, protected data mitigation, artist opt-outs, and policy-driven content updates. As models grow larger and adopt more diverse architectures, achieving precise and selective unlearning while preserving generative quality becomes increasingly challenging. We introduce SurgUn (pronounced as Surgeon), a surgical unlearning method that applies targeted weight-space updates to remove specific visual concepts in text-conditioned diffusion models. Our approach is motivated by retroactive interference theory, which holds that newly acquired memories can overwrite, suppress, or impede access to prior ones by competing for shared representational pathways. We adapt this principle to diffusion models by inducing retroactive concept interference, enabling focused destabilization of only the target concept while preserving unrelated capabilities through a novel training paradigm. SurgUn achieves high-precision unlearning across diverse settings. It performs strongly on compact U-Net based models such as Stable Diffusion v1.5, scales effectively to the larger U-Net architecture SDXL, and extends to SANA, representing an underexplored Diffusion Transformer based architecture for unlearning. 

---
# AWE: Adaptive Agents for Dynamic Web Penetration Testing 

**Authors**: Akshat Singh Jaswal, Ashish Baghel  

**Link**: [PDF](https://arxiv.org/pdf/2603.00960)  

**Abstract**: Modern web applications are increasingly produced through AI-assisted development and rapid no-code deployment pipelines, widening the gap between accelerating software velocity and the limited adaptability of existing security tooling. Pattern-driven scanners fail to reason about novel contexts, while emerging LLM-based penetration testers rely on unconstrained exploration, yielding high cost, unstable behavior, and poor reproducibility.
We introduce AWE, a memory-augmented multi-agent framework for autonomous web penetration testing that embeds structured, vulnerability-specific analysis pipelines within a lightweight LLM orchestration layer. Unlike general-purpose agents, AWE couples context aware payload mutations and generations with persistent memory and browser-backed verification to produce deterministic, exploitation-driven results.
Evaluated on the 104-challenge XBOW benchmark, AWE achieves substantial gains on injection-class vulnerabilities - 87% XSS success (+30.5% over MAPTA) and 66.7% blind SQL injection success (+33.3%) - while being much faster, cheaper, and more token-efficient than MAPTA, despite using a midtier model (Claude Sonnet 4) versus MAPTA's GPT-5. MAPTA retains higher overall coverage due to broader exploratory capabilities, underscoring the complementary strengths of specialized and general-purpose architectures. Our results demonstrate that architecture matters as much as model reasoning capabilities: integrating LLMs into principled, vulnerability-aware pipelines yields substantial gains in accuracy, efficiency, and determinism for injection-class exploits. The source code for AWE is available at: this https URL 

---
# Beyond False Discovery Rate: A Stepdown Group SLOPE Approach for Grouped Variable Selection 

**Authors**: Xuelin Zhang, Jingxuan Liang, Xinyue Liu, Hong Chen, Biqin Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.00955)  

**Abstract**: High-dimensional feature selection is routinely required to balance statistical power with strict control of multiple-error metrics such as the k-Family-Wise Error Rate (k-FWER) and the False Discovery Proportion (FDP), yet some existing frameworks are confined to the narrower goal of controlling the expected False Discovery Rate (FDR) and can not exploit the group-structure of the covariates, such as Sorted L-One Penalized Estimation (SLOPE). We introduce the Group Stepdown SLOPE, a unified optimization procedure which is capable of embedding the Lehmann-Romano stepdown rules into SLOPE to achieve finite-sample guarantees under k-FWER and FDP thresholds. Specifically, we derive closed-form regularization sequences under orthogonal designs that provably bound k-FWER and FDP at user-specified levels, and extend these results to grouped settings via gk-SLOPE and gF-SLOPE, which control the analogous group-level errors gk-FWER and gFDP. For non-orthogonal general designs, we provide a calibrated data-driven sequence inspired by Gaussian approximation and Monte-Carlo correction, preserving convexity and scalability. Extensive simulations are conducted across sparse, correlated, and group-structured regimes. Empirical results corroborate our theoretical findings that the proposed methods achieve nominal error control, while yielding markedly higher power than competing stepdown procedures, thereby confirming the practical value of the theoretical advances. 

---
# Seeing Beyond 8bits: Subjective and Objective Quality Assessment of HDR-UGC Videos 

**Authors**: Shreshth Saini, Bowen Chen, Neil Birkbeck, Yilin Wang, Balu Adsumilli, Alan C. Bovik  

**Link**: [PDF](https://arxiv.org/pdf/2603.00938)  

**Abstract**: High Dynamic Range (HDR) user-generated (UGC) videos are rapidly proliferating across social platforms, yet most perceptual video quality assessment (VQA) systems remain tailored to Standard Dynamic Range (SDR). HDR has a higher bit depth, wide color gamut, and elevated luminance range, exposing distortions such as near-black crushing, highlight clipping, banding, and exposure flicker that amplify UGC artifacts and challenge SDR models. To catalyze progress, we curate Beyond8Bits, a large-scale subjective dataset of 44K videos from 6.5K sources with over 1.5M crowd ratings, spanning diverse scenes, capture conditions, and compression settings. We further introduce HDR-Q, the first Multimodal Large Language Model (MLLM) for HDR-UGC VQA. We propose (i) a novel HDR-aware vision encoder to produce HDR-sensitive embeddings, and (ii) HDR-Aware Policy Optimization (HAPO), an RL finetuning framework that anchors reasoning to HDR cues. HAPO augments GRPO via an HDR-SDR contrastive KL that encourages token reliance on HDR inputs and a Gaussian weighted regression reward for fine-grained MOS calibration. Across Beyond8Bits and public HDR-VQA benchmarks, HDR-Q delivers state-of-the-art performance. 

---
# Conformal Prediction for Risk-Controlled Medical Entity Extraction Across Clinical Domains 

**Authors**: Manil Shrestha, Edward Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.00924)  

**Abstract**: Large Language Models (LLMs) are increasingly used for medical entity extraction, yet their confidence scores are often miscalibrated, limiting safe deployment in clinical settings. We present a conformal prediction framework that provides finite-sample coverage guarantees for LLM-based extraction across two clinical domains. First, we extract structured entities from 1,000 FDA drug labels across eight sections using GPT-4.1, verified via FactScore-based atomic statement evaluation (97.7\% accuracy over 128,906 entities). Second, we extract radiological entities from MIMIC-CXR reports using the RadGraph schema with GPT-4.1 and Llama-4-Maverick, evaluated against physician annotations (entity F1: 0.81 to 0.84). Our central finding is that miscalibration direction reverses across domains: on well-structured FDA labels, models are underconfident, requiring modest conformal thresholds ($\tau \approx 0.06$), while on free-text radiology reports, models are overconfident, demanding strict thresholds ($\tau$ up to 0.99). Despite this heterogeneity, conformal prediction achieves target coverage ($\geq 90\%$) in both settings with manageable rejection rates (9--13\%). These results demonstrate that calibration is not a global model property but depends on document structure, extraction category, and model architecture, motivating domain-specific conformal calibration for safe clinical deployment. 

---
# Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards 

**Authors**: Seungwook Kim, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2603.00918)  

**Abstract**: Text-to-image generation powers content creation across design, media, and data augmentation. Post-training of text-to-image generative models is a promising path to better match human preferences, factuality, and improved aesthetics. We introduce ARC (Adaptive Rewarding by self-Confidence), a post-training framework that replaces external reward supervision with an internal self-confidence signal, obtained by evaluating how accurately the model recovers injected noise under self-denoising probes. ARC converts this intrinsic signal into scalar rewards, enabling fully unsupervised optimization without additional datasets, annotators, or reward models. Empirically, by reinforcing high-confidence generations, ARC delivers consistent gains in compositional generation, text rendering and text-image alignment over the baseline. We also find that integrating ARC with external rewards results in a complementary improvement, with alleviated reward hacking. 

---
# Prompt Sensitivity and Answer Consistency of Small Open-Source Large Language Models on Clinical Question Answering: Implications for Low-Resource Healthcare Deployment 

**Authors**: Shravani Hariprasad  

**Link**: [PDF](https://arxiv.org/pdf/2603.00917)  

**Abstract**: Small open-source language models are gaining attention for low-resource healthcare settings, but their reliability under different prompt phrasings remains poorly understood. We evaluated five open-source models (Gemma 2 2B, Phi-3 Mini 3.8B, Llama 3.2 3B, Mistral 7B, and Meditron-7B domain-pretrained without instruction tuning) across three clinical QA datasets (MedQA, MedMCQA, PubMedQA) using five prompt styles (original, formal, simplified, roleplay, direct). We measured consistency scores, accuracy, and instruction-following failure rates. All inference ran locally on consumer CPU hardware without fine-tuning.
Consistency and accuracy were largely independent. Gemma 2 achieved the highest consistency (0.845-0.888) but lowest accuracy (33.0-43.5%), while Llama 3.2 showed moderate consistency (0.774-0.807) with the highest accuracy (49.0-65.0%). Roleplay prompts consistently reduced accuracy across all models, with Phi-3 Mini dropping 21.5 percentage points on MedQA. Meditron-7B exhibited near-complete instruction-following failure on PubMedQA (99.0% UNKNOWN rate), showing domain pretraining alone is insufficient for structured clinical QA.
High consistency does not imply correctness. Models can be reliably wrong, a dangerous failure mode in clinical AI. Roleplay prompts should be avoided in healthcare applications. Llama 3.2 showed the strongest balance of accuracy and reliability for low-resource deployment. Safe clinical AI requires joint evaluation of consistency, accuracy, and instruction adherence. 

---
# Curvature-Weighted Capacity Allocation: A Minimum Description Length Framework for Layer-Adaptive Large Language Model Optimization 

**Authors**: Theophilus Amaefuna, Hitesh Vaidya, Anshuman Chhabra, Ankur Mali  

**Link**: [PDF](https://arxiv.org/pdf/2603.00910)  

**Abstract**: Layer-wise capacity in large language models is highly non-uniform: some layers contribute disproportionately to loss reduction while others are near-redundant. Existing methods for exploiting this non-uniformity, such as influence-function-based layer scoring, produce sensitivity estimates but offer no principled mechanism for translating them into allocation or pruning decisions under hardware constraints. We address this gap with a unified, curvature-aware framework grounded in the Minimum Description Length (MDL) principle. Our central quantity is the curvature-adjusted layer gain $\zeta_k^2 = g_k^\top \widetilde{H}_{kk}^{-1} g_k$, which we show equals twice the maximal second-order reduction in empirical risk achievable by updating layer $k$ alone, and which strictly dominates gradient-norm-based scores by incorporating local curvature. Normalizing these gains into layer quality scores $q_k$, we formulate two convex MDL programs: a capacity allocation program that distributes expert slots or LoRA rank preferentially to high-curvature layers under diminishing returns, and a pruning program that concentrates sparsity on low-gain layers while protecting high-gain layers from degradation. Both programs admit unique closed-form solutions parameterized by a single dual variable, computable in $O(K \log 1/\varepsilon)$ via bisection. We prove an $O(\delta^2)$ transfer regret bound showing that source-domain allocations remain near-optimal on target tasks when curvature scores drift by $\delta$, with explicit constants tied to the condition number of the target program. Together, these results elevate layer-wise capacity optimization from an empirical heuristic to a theoretically grounded, computationally efficient framework with provable optimality and generalization guarantees. 

---
# CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning 

**Authors**: Xinyu Zhu, Yihao Feng, Yanchao Sun, Xianzhi Du, Pingzhi Li, Olli Saarikivi, Yun Zhu, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2603.00889)  

**Abstract**: Large Language Models (LLMs) have recently exhibited remarkable reasoning capabilities, largely enabled by supervised fine-tuning (SFT)- and reinforcement learning (RL)-based post-training on high-quality reasoning data. However, reproducing and extending these capabilities in open and scalable settings is hindered by three fundamental data-centric challenges: (1) the cold-start problem, arising from the lack of seed datasets with detailed, long Chain-of-Thought (CoT) trajectories needed to initialize reasoning policies; (2) limited domain coverage, as most existing open-source reasoning datasets are concentrated in mathematics, with limited coverage of broader scientific disciplines; and (3) the annotation bottleneck, where the difficulty of frontier-level reasoning tasks makes reliable human annotation prohibitively expensive or infeasible. To address these challenges, we introduce CHIMERA, a compact synthetic reasoning dataset comprising 9K samples for generalizable cross-domain reasoning. CHIMERA is constructed with three key properties: (1) it provides rich, long CoT reasoning trajectories synthesized by state-of-the-art reasoning models; (2) it has broad and structured coverage, spanning 8 major scientific disciplines and over 1K fine-grained topics organized via a model-generated hierarchical taxonomy; and (3) it employs a fully automated, scalable evaluation pipeline that uses strong reasoning models to cross-validate both problem validity and answer correctness. We use CHIMERA to post-train a 4B Qwen3 model. Despite the dataset's modest size, the resulting model achieves strong performance on a suite of challenging reasoning benchmarks, including GPQA-Diamond, AIME 24/25/26, HMMT 25, and Humanity's Last Exam, approaching or matching the reasoning performance of substantially larger models such as DeepSeek-R1 and Qwen3-235B. 

---
# Knowledge without Wisdom: Measuring Misalignment between LLMs and Intended Impact 

**Authors**: Michael Hardy, Yunsung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.00883)  

**Abstract**: LLMs increasingly excel on AI benchmarks, but doing so does not guarantee validity for downstream tasks. This study evaluates the performance of leading foundation models (FMs, i.e., generative pre-trained base LLMs) with out-of-distribution (OOD) tasks of the teaching and learning of schoolchildren. Across all FMs, inter-model behaviors on disparate tasks correlate higher than they do with expert human behaviors on target tasks. These biases shared across LLMs are poorly aligned with downstream measures of teaching quality and often \textit{negatively aligned with learning outcomes}. Further, we find multi-model ensembles, both unanimous model voting and expert-weighting by benchmark performance, further exacerbate misalignment with learning. We measure that 50\% of the variation in misalignment error is shared across foundation models, suggesting that common pretraining accounts for much of the misalignment in these tasks. We demonstrate methods for robustly measuring alignment of complex tasks and provide unique insights into both educational applications of foundation models and to understanding limitations of models. 

---
# PPC-MT: Parallel Point Cloud Completion with Mamba-Transformer Hybrid Architecture 

**Authors**: Jie Li, Shengwei Tian, Long Yu, Xin Ning  

**Link**: [PDF](https://arxiv.org/pdf/2603.00870)  

**Abstract**: Existing point cloud completion methods struggle to balance high-quality reconstruction with computational efficiency. To address this, we propose PPC-MT, a novel parallel framework for point cloud completion leveraging a hybrid Mamba-Transformer architecture. Our approach introduces an innovative parallel completion strategy guided by Principal Component Analysis (PCA), which imposes a geometrically meaningful structure on unordered point clouds, transforming them into ordered sets and decomposing them into multiple subsets. These subsets are reconstructed in parallel using a multi-head reconstructor. This structured parallel synthesis paradigm significantly enhances the uniformity of point distribution and detail fidelity, while preserving computational efficiency. By integrating Mamba's linear complexity for efficient feature extraction during encoding with the Transformer's capability to model fine-grained multi-sequence relationships during decoding, PPC-MT effectively balances efficiency and reconstruction accuracy. Extensive quantitative and qualitative experiments on benchmark datasets, including PCN, ShapeNet-55/34, and KITTI, demonstrate that PPC-MT outperforms state-of-the-art methods across multiple metrics, validating the efficacy of our proposed framework. 

---
# AMDS: Attack-Aware Multi-Stage Defense System for Network Intrusion Detection with Two-Stage Adaptive Weight Learning 

**Authors**: Oluseyi Olukola, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2603.00859)  

**Abstract**: Machine learning based network intrusion detection systems are vulnerable to adversarial attacks that degrade classification performance under both gradient-based and distribution shift threat models. Existing defenses typically apply uniform detection strategies, which may not account for heterogeneous attack characteristics. This paper proposes an attack-aware multi-stage defense framework that learns attack-specific detection strategies through a weighted combination of ensemble disagreement, predictive uncertainty, and distributional anomaly signals. Empirical analysis across seven adversarial attack types reveals distinct detection signatures, enabling a two-stage adaptive detection mechanism. Experimental evaluation on a benchmark intrusion detection dataset indicates that the proposed system attains 94.2% area under the receiver operating characteristic curve and improves classification accuracy by 4.5 percentage points and F1-score by 9.0 points over adversarially trained ensembles. Under adaptive white-box attacks with full architectural knowledge, the system appears to maintain 94.4% accuracy with a 4.2% attack success rate, though this evaluation is limited to two adaptive variants and does not constitute a formal robustness guarantee. Cross-dataset validation further suggests that defense effectiveness depends on baseline classifier competence and may vary with feature dimensionality. These results suggest that attack-specific optimization combined with multi-signal integration can provide a practical approach to improving adversarial robustness in machine learning-based intrusion detection systems. 

---
# Artificial Superintelligence May be Useless: Equilibria in the Economy of Multiple AI Agents 

**Authors**: Huan Cai, Ziqing Lu, Catherine Xu, Weiyu Xu, Jie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.00858)  

**Abstract**: With recent development of artificial intelligence, it is more common to adopt AI agents in economic activities. This paper explores the economic actions of agents, including human agents and AI agents, in an economic game of trading products/services, and the equilibria in this economy involving multiple agents. We derive a range of equilibrium results and their corresponding conditions using a Markov chain stationary distribution based model. One distinct feature of our model is that we consider the long-term utility generated by economic activities instead of their short-term benefits. For the model consisting of two agents, we fully characterize all the possible economic equilibria and conditions. Interestingly, we show that unless each agent can at least double (not merely increase) its marginal utility by purchasing the other agent's products/services, purchasing the other agent's products/services will not happen in any economic equilibrium. We further extend our results to three and more agents, where we characterize more economic equilibria. We find that in some equilibria, the ``more powerful'' AI agents contribute zero utility to ``less capable'' agents. 

---
# MultiPUFFIN: A Multimodal Domain-Constrained Foundation Model for Molecular Property Prediction of Small Molecules 

**Authors**: Idelfonso B. R. Nogueira, Carine M. Rebelloa, Mumin Enis Leblebici, Erick Giovani Sperandio Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2603.00857)  

**Abstract**: Predicting physicochemical properties across chemical space is vital for chemical engineering, drug discovery, and materials science. Current molecular foundation models lack thermodynamic consistency, while domain-informed approaches are limited to single properties and small datasets. We introduce MultiPUFFIN, a domain-constrained multimodal foundation model addressing both limitations simultaneously. MultiPUFFIN features: (i) an encoder fusing SMILES, graphs, and 3D geometries via gated cross-modal attention, alongside experimental condition and descriptor encoders; (ii) prediction heads embedding established correlations (e.g., Wagner, Andrade, van't Hoff, and Shomate equations) as inductive biases to ensure thermodynamic consistency; and (iii) a two-stage multi-task training this http URL prior frameworks, MultiPUFFIN predicts nine thermophysical properties simultaneously. It is trained on a multi-source dataset of 37,968 unique molecules (40,904 rows). With roughly 35 million parameters, MultiPUFFIN achieves a mean $R^2 = 0.716$ on a challenging scaffold-split test set of 8,877 molecules. Compared to ChemBERTa-2 (pre-trained on 77 million molecules), MultiPUFFIN outperforms the fine-tuned baseline across all nine properties despite using 2000x fewer training molecules. Advantages are strikingly apparent for temperature-dependent properties, where ChemBERTa-2 lacks the architectural capacity to incorporate thermodynamic this http URL results demonstrate that multimodal encoding and domain-informed biases substantially reduce data and compute requirements compared to brute-force pre-training. Furthermore, MultiPUFFIN handles missing modalities and recovers meaningful thermodynamic parameters without explicit supervision. Systematic ablation studies confirm the property-specific benefits of these domain-informed prediction heads. 

---
# PARCER as an Operational Contract to Reduce Variance, Cost, and Risk in LLM Systems 

**Authors**: Elzo Brito dos Santos Filho  

**Link**: [PDF](https://arxiv.org/pdf/2603.00856)  

**Abstract**: Systems based on Large Language Models (LLMs) have become formidable tools for automating research and software production. However, their governance remains a challenge when technical requirements demand absolute consistency, auditability, and predictable control over cost and latency. Recent literature highlights two phenomena that aggravate this scenario: the stochastic variance inherent in the model's judgment (often treated as "systemic noise") and the substantial degradation of context utilization in long inputs, with critical losses when decisive information is diluted in the middle of the prompt. This article proposes PARCER as an engineering response to these limitations. The framework acts as a declarative "operational contract" in YAML, transforming unstructured interactions into versioned and executable artifacts. PARCER imposes strict governance structured into seven operational phases, introducing decision hygiene practices inspired by legal judgments to mitigate noise, adaptive token budgeting, formalized recovery routes (fallbacks) for context preservation, and systemic observability via OpenTelemetry. The objective of this work is to present the conceptual and technical architecture of PARCER, positioning it as a necessary transition from simple "prompt engineering" to "context engineering with governable governance". 

---
# Constitutional Black-Box Monitoring for Scheming in LLM Agents 

**Authors**: Simon Storf, Rich Barton-Cooper, James Peters-Gill, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.00829)  

**Abstract**: Safe deployment of Large Language Model (LLM) agents in autonomous settings requires reliable oversight mechanisms. A central challenge is detecting scheming, where agents covertly pursue misaligned goals. One approach to mitigating such risks is LLM-based monitoring: using language models to examine agent behaviors for suspicious actions. We study constitutional black-box monitors: prompted classifiers that detect scheming using only externally observable inputs and outputs, optimized on synthetic data generated from natural-language behavior specifications. We introduce two pipelines for generating synthetic agent trajectories, STRIDE (iterative refinement) and Gloom (agent-environment simulation), from which we generate 1,000 samples each. We optimize frontier LLM monitors on these datasets via prompt sweeps, human refinement, and automated prompt optimization, and evaluate performance on 7,500 held-out trajectories from ControlArena, a suite of grounded environments where agents operate in more realistic contexts. Our results demonstrate that monitors selected purely on synthetic data can generalize to more realistic environments, capturing a meaningful scheming signal. However, we find that performance saturates quickly in our setting, with simple prompt sweeps matching the results of more extensive optimization. Pushing beyond this limit yields no further improvements and instead leads to overfitting. 

---
# A Gauge Theory of Superposition: Toward a Sheaf-Theoretic Atlas of Neural Representations 

**Authors**: Hossein Javidnia  

**Link**: [PDF](https://arxiv.org/pdf/2603.00824)  

**Abstract**: We develop a discrete gauge-theoretic framework for superposition in large language models (LLMs) that replaces the single-global-dictionary premise with a sheaf-theoretic atlas of local semantic charts. Contexts are clustered into a stratified context complex; each chart carries a local feature space and a local information-geometric metric (Fisher/Gauss--Newton) identifying predictively consequential feature interactions. This yields a Fisher-weighted interference energy and three measurable obstructions to global interpretability: (O1) local jamming (active load exceeds Fisher bandwidth), (O2) proxy shearing (mismatch between geometric transport and a fixed correspondence proxy), and (O3) nontrivial holonomy (path-dependent transport around loops).
We prove and instantiate four results on a frozen open LLM (Llama~3.2~3B Instruct) using WikiText-103, a C4-derived English web-text subset, and \texttt{the-stack-smol}. (A) After constructive gauge fixing on a spanning tree, each chord residual equals the holonomy of its fundamental cycle, making holonomy computable and gauge-invariant. (B) Shearing lower-bounds a data-dependent transfer mismatch energy, turning $D_{\mathrm{shear}}$ into an unavoidable failure bound. (C) We obtain non-vacuous certified jamming/interference bounds with high coverage and zero violations across seeds/hyperparameters. (D) Bootstrap and sample-size experiments show stable estimation of $D_{\mathrm{shear}}$ and $D_{\mathrm{hol}}$, with improved concentration on well-conditioned subsystems. 

---
# A Comprehensive Evaluation of LLM Unlearning Robustness under Multi-Turn Interaction 

**Authors**: Ruihao Pan, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00823)  

**Abstract**: Machine unlearning aims to remove the influence of specific training data from pre-trained models without retraining from scratch, and is increasingly important for large language models (LLMs) due to safety, privacy, and legal concerns. Although prior work primarily evaluates unlearning in static, single-turn settings, forgetting robustness under realistic interactive use remains underexplored. In this paper, we study whether unlearning remains stable in interactive environments by examining two common interaction patterns: self-correction and dialogue-conditioned querying. We find that knowledge appearing forgotten in static evaluation can often be recovered through interaction. Although stronger unlearning improves apparent robustness, it often results in behavioral rigidity rather than genuine knowledge erasure. Our findings suggest that static evaluation may overestimate real-world effectiveness and highlight the need for ensuring stable forgetting under interactive settings. 

---
# ContextCov: Deriving and Enforcing Executable Constraints from Agent Instruction Files 

**Authors**: Reshabh K Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2603.00822)  

**Abstract**: As Large Language Model (LLM) agents increasingly execute complex, autonomous software engineering tasks, developers rely on natural language Agent Instructions (e.g., this http URL) to enforce project-specific coding conventions, tooling, and architectural boundaries. However, these instructions are passive text. Agents frequently deviate from them due to context limitations or conflicting legacy code, a phenomenon we term Context Drift. Because agents operate without real-time human supervision, these silent violations rapidly compound into technical debt.
To bridge this gap, we introduce ContextCov, a framework that transforms passive Agent Instructions into active, executable guardrails. ContextCov extracts natural language constraints and synthesizes enforcement checks across three domains: static AST analysis for code patterns, runtime shell shims that intercept prohibited commands, and architectural validators for structural and semantic constraints. Evaluations on 723 open-source repositories demonstrate that ContextCov successfully extracts over 46,000 executable checks with 99.997% syntax validity, providing a necessary automated compliance layer for safe, agent-driven development. Source code and evaluation results are available at this https URL. 

---
# Neural Latent Arbitrary Lagrangian-Eulerian Grids for Fluid-Solid Interaction 

**Authors**: Shilong Tao, Zhe Feng, Shaohan Chen, Weichen Zhang, Zhanxing Zhu, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00792)  

**Abstract**: Fluid-solid interaction (FSI) problems are fundamental in many scientific and engineering applications, yet effectively capturing the highly nonlinear two-way interactions remains a significant challenge. Most existing deep learning methods are limited to simplified one-way FSI scenarios, often assuming rigid and static solid to reduce complexity. Even in two-way setups, prevailing approaches struggle to capture dynamic, heterogeneous interactions due to the lack of cross-domain awareness. In this paper, we introduce \textbf{Fisale}, a data-driven framework for handling complex two-way \textbf{FSI} problems. It is inspired by classical numerical methods, namely the Arbitrary Lagrangian-Eulerian (\textbf{ALE}) method and the partitioned coupling algorithm. Fisale explicitly models the coupling interface as a distinct component and leverages multiscale latent ALE grids to provide unified, geometry-aware embeddings across domains. A partitioned coupling module (PCM) further decomposes the problem into structured substeps, enabling progressive modeling of nonlinear interdependencies. Compared to existing models, Fisale introduces a more flexible framework that iteratively handles complex dynamics of solid, fluid and their coupling interface on a unified representation, and enables scalable learning of complex two-way FSI behaviors. Experimentally, Fisale excels in three reality-related challenging FSI scenarios, covering 2D, 3D and various tasks. The code is available at \href{this https URL}. 

---
# QANTIS: A Hardware-Validated Quantum Platform for POMDP Planning and Multi-Target Data Association 

**Authors**: Bayram Yüksel Eker, Suayb S. Arslan, Özgür Nazlı, Mustafa Serhat Demirgil, Furkan Deligöz  

**Link**: [PDF](https://arxiv.org/pdf/2603.00785)  

**Abstract**: Autonomous navigation under uncertainty requires solving partially observable Markov decision processes (POMDPs) for planning and assigning sensor measurements to tracked targets--a task known as multi-target data association (MTDA). Both problems become computationally demanding at scale: belief conditioning costs $\mathcal{O}(P(e)^{-1})$ per node under rare evidence, while MTDA is NP-hard. Quantum amplitude amplification can quadratically reduce the belief-update query cost to $\mathcal{O}(P(e)^{-1/2})$, while QUBO reformulations expose MTDA to quantum and quantum-inspired optimisation heuristics. We present QANTIS, a modular platform that integrates quantum belief update (Grover amplitude amplification and BIQAE), QUBO-based data association via FPC-QAOA, and composable error mitigation, and we report a 45-experiment hardware study on three IBM Heron backends. On hardware, a single Grover iterate applied to a Tiger belief oracle amplifies a rare observation probability from $0.179$ to $0.907$ ($5.1\times$; ISA 18) while preserving the Bayesian posterior (Hellinger $0.0015$), increasing usable-shot yield from 1,463 to 7,429. We interpret this as a hardware validation of the quadratic query-complexity mechanism at $k=1$ with posterior preservation, rather than a wall-clock advantage claim. We further demonstrate, to our knowledge, the first closed-loop hybrid quantum-classical Tiger POMDP on superconducting hardware ($T=8$, max Hellinger below $0.015$), and empirically characterise NISQ feasibility boundaries: ZNE-based error mitigation is beneficial below ISA $\approx 100$ and harmful above ISA $\gtrsim 1{,}000$; FPC-QAOA is meaningful at $\leq 15$ QUBO variables (ISA $\lesssim 450$). These results characterise practical operating regimes on current superconducting hardware rather than wall-clock quantum advantage at today's problem scales. 

---
# Identifying and Characterising Response in Clinical Trials: Development and Validation of a Machine Learning Approach in Colorectal Cancer 

**Authors**: Adam Marcus, Paul Agapow  

**Link**: [PDF](https://arxiv.org/pdf/2603.00757)  

**Abstract**: Precision medicine promises to transform health care by offering individualised treatments that dramatically improve clinical outcomes. A necessary prerequisite is to identify subgroups of patients who respond differently to different therapies. Current approaches are limited to static measures of treatment success, neglecting the repeated measures found in most clinical trials. Our approach combines the concept of partly conditional modelling with treatment effect estimation based on the Virtual Twins method. The resulting time-specific responses to treatment are characterised using survLIME, an extension of Local Interpretable Model-agnostic Explanations (LIME) to survival data. Performance was evaluated using synthetic data and applied to clinical trials examining the effectiveness of panitumumab to treat metastatic colorectal cancer. An area under the receiver operating characteristic curve (AUC) of 0.77 for identifying fixed responders was achieved in a 1000 patient simulation. When considering dynamic responders, partly conditional modelling increased the AUC from 0.597 to 0.685. Applying the approach to colorectal cancer trials found genetic mutations, sites of metastasis, and ethnicity as important factors for response to treatment. Our approach can accommodate a dynamic response to treatment while potentially providing better performance than existing methods in instances of a fixed response to treatment. When applied to clinical data we attain results consistent with the literature. 

---
# Stroke outcome and evolution prediction from CT brain using a spatiotemporal diffusion autoencoder 

**Authors**: Adam Marcus, Paul Bentley, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2603.00756)  

**Abstract**: Stroke is a major cause of death and disability worldwide. Accurate outcome and evolution prediction has the potential to revolutionize stroke care by individualizing clinical decision-making leading to better outcomes. However, despite a plethora of attempts and the rich data provided by neuroimaging, modelling the ultimate fate of brain tissue remains a challenging task. In this work, we apply recent ideas in the field of diffusion probabilistic models to generate a self-supervised semantically meaningful stroke representation from Computed Tomography (CT) images. We then improve this representation by extending the method to accommodate longitudinal images and the time from stroke onset. The effectiveness of our approach is evaluated on a dataset consisting of 5,824 CT images from 3,573 patients across two medical centers with minimal labels. Comparative experiments show that our method achieves the best performance for predicting next-day severity and functional outcome at discharge. 

---
# General Proximal Flow Networks 

**Authors**: Alexander Strunk, Roland Assam  

**Link**: [PDF](https://arxiv.org/pdf/2603.00751)  

**Abstract**: This paper introduces General Proximal Flow Networks (GPFNs), a generalization of Bayesian Flow Networks that broadens the class of admissible belief-update operators. In Bayesian Flow Networks, each update step is a Bayesian posterior update, which is equivalent to a proximal step with respect to the Kullback-Leibler divergence. GPFNs replace this fixed choice with an arbitrary divergence or distance function, such as the Wasserstein distance, yielding a unified proximal-operator framework for iterative generative modeling. The corresponding training and sampling procedures are derived, establishing a formal link to proximal optimization and recovering the standard BFN update as a special case. Empirical evaluations confirm that adapting the divergence to the underlying data geometry yields measurable improvements in generation quality, highlighting the practical benefits of this broader framework. 

---
# Wild-Drive: Off-Road Scene Captioning and Path Planning via Robust Multi-modal Routing and Efficient Large Language Model 

**Authors**: Zihang Wang, Xu Li, Benwu Wang, Wenkai Zhu, Xieyuanli Chen, Dong Kong, Kailin Lyu, Yinan Du, Yiming Peng, Haoyang Che  

**Link**: [PDF](https://arxiv.org/pdf/2603.00694)  

**Abstract**: Explainability and transparent decision-making are essential for the safe deployment of autonomous driving systems. Scene captioning summarizes environmental conditions and risk factors in natural language, improving transparency, safety, and human--robot interaction. However, most existing approaches target structured urban scenarios; in off-road environments, they are vulnerable to single-modality degradations caused by rain, fog, snow, and darkness, and they lack a unified framework that jointly models structured scene captioning and path planning. To bridge this gap, we propose Wild-Drive, an efficient framework for off-road scene captioning and path planning. Wild-Drive adopts modern multimodal encoders and introduces a task-conditioned modality-routing bridge, MoRo-Former, to adaptively aggregate reliable information under degraded sensing. It then integrates an efficient large language model (LLM), together with a planning token and a gate recurrent unit (GRU) decoder, to generate structured captions and predict future trajectories. We also build the OR-C2P Benchmark, which covers structured off-road scene captioning and path planning under diverse sensor corruption conditions. Experiments on OR-C2P dataset and a self-collected dataset show that Wild-Drive outperforms prior LLM-based methods and remains more stable under degraded sensing. The code and benchmark will be publicly available at this https URL. 

---
# SSKG Hub: An Expert-Guided Platform for LLM-Empowered Sustainability Standards Knowledge Graphs 

**Authors**: Chaoyue He, Xin Zhou, Xinjia Yu, Lei Zhang, Yan Zhang, Yi Wu, Lei Xiao, Liangyue Li, Di Wang, Hong Xu, Xiaoqiao Wang, Wei Liu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00669)  

**Abstract**: Sustainability disclosure standards (e.g., GRI, SASB, TCFD, IFRS S2) are comprehensive yet lengthy, terminology-dense, and highly cross-referential, hindering structured analysis and downstream use. We present SSKG Hub (Sustainability Standards Knowledge Graph Hub), a research prototype and interactive web platform that transforms standards into auditable knowledge graphs (KGs) through an LLM-centered, expert-guided pipeline. The system integrates automatic standard identification, configurable chunking, standard-specific prompting, robust triple parsing, and provenance-aware Neo4j storage with fine-grained audit metadata. LLM extraction produces a provenance-linked Draft KG, which is reviewed, curated, and formally promoted to a Certified KG through meta-expert adjudication. A role-based governance framework covering read-only guest access, expert review and CRUD operations, meta-expert certification, and administrative oversight ensures traceability and accountability across draft and certified states. Beyond graph exploration and triple-level evidence tracing, SSKG Hub supports cross-KG fusion, KG-driven tasks, and dedicated modules for insights and curated resources. We validate the platform through a comprehensive expert-led KG review case study that demonstrates end-to-end curation and quality assurance. The web application is publicly available at this http URL. 

---
# IDER: IDempotent Experience Replay for Reliable Continual Learning 

**Authors**: Zhanwang Liu, Yuting Li, Haoyuan Gao, Yexin Li, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00624)  

**Abstract**: Catastrophic forgetting, the tendency of neural networks to forget previously learned knowledge when learning new tasks, has been a major challenge in continual learning (CL). To tackle this challenge, CL methods have been proposed and shown to reduce forgetting. Furthermore, CL models deployed in mission-critical settings can benefit from uncertainty awareness by calibrating their predictions to reliably assess their confidences. However, existing uncertainty-aware continual learning methods suffer from high computational overhead and incompatibility with mainstream replay methods. To address this, we propose idempotent experience replay (IDER), a novel approach based on the idempotent property where repeated function applications yield the same output. Specifically, we first adapt the training loss to make model idempotent on current data streams. In addition, we introduce an idempotence distillation loss. We feed the output of the current model back into the old checkpoint and then minimize the distance between this reprocessed output and the original output of the current model. This yields a simple and effective new baseline for building reliable continual learners, which can be seamlessly integrated with other CL approaches. Extensive experiments on different CL benchmarks demonstrate that IDER consistently improves prediction reliability while simultaneously boosting accuracy and reducing forgetting. Our results suggest the potential of idempotence as a promising principle for deploying efficient and trustworthy continual learning systems in real-world this http URL code is available at this https URL. 

---
# CMI-RewardBench: Evaluating Music Reward Models with Compositional Multimodal Instruction 

**Authors**: Yinghao Ma, Haiwen Xia, Hewei Gao, Weixiong Chen, Yuxin Ye, Yuchen Yang, Sungkyun Chang, Mingshuo Ding, Yizhi Li, Ruibin Yuan, Simon Dixon, Emmanouil Benetos  

**Link**: [PDF](https://arxiv.org/pdf/2603.00610)  

**Abstract**: While music generation models have evolved to handle complex multimodal inputs mixing text, lyrics, and reference audio, evaluation mechanisms have lagged behind. In this paper, we bridge this critical gap by establishing a comprehensive ecosystem for music reward modeling under Compositional Multimodal Instruction (CMI), where the generated music may be conditioned on text descriptions, lyrics, and audio prompts. We first introduce CMI-Pref-Pseudo, a large-scale preference dataset comprising 110k pseudo-labeled samples, and CMI-Pref, a high-quality, human-annotated corpus tailored for fine-grained alignment tasks. To unify the evaluation landscape, we propose CMI-RewardBench, a unified benchmark that evaluates music reward models on heterogeneous samples across musicality, text-music alignment, and compositional instruction alignment. Leveraging these resources, we develop CMI reward models (CMI-RMs), a parameter-efficient reward model family capable of processing heterogeneous inputs. We evaluate their correlation with human judgments scores on musicality and alignment on CMI-Pref along with previous datasets. Further experiments demonstrate that CMI-RM not only correlates strongly with human judgments, but also enables effective inference-time scaling via top-k filtering. The necessary training data, benchmarks, and reward models are publicly available. 

---
# IdGlow: Dynamic Identity Modulation for Multi-Subject Generation 

**Authors**: Honghao Cai, Xiangyuan Wang, Yunhao Bai, Tianze Zhou, Sijie Xu, Yuyang Hao, Zezhou Cui, Yuyuan Yang, Wei Zhu, Yibo Chen, Xu Tang, Yao Hu, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.00607)  

**Abstract**: Multi-subject image generation requires seamlessly harmonizing multiple reference identities within a coherent scene. However, existing methods relying on rigid spatial masks or localized attention often struggle with the "stability-plasticity dilemma," particularly failing in tasks that require complex structural deformations, such as identity-preserving age transformation. To address this, we present IdGlow, a mask-free, progressive two-stage framework built upon Flow Matching diffusion models. In the supervised fine-tuning (SFT) stage, we introduce task-adaptive timestep scheduling aligned with diffusion generative dynamics: a linear decay schedule that progressively relaxes constraints for natural group composition, and a temporal gating mechanism that concentrates identity injection within a critical semantic window, successfully preserving adult facial semantics without overriding child-like anatomical structures. To resolve attribute leakage and semantic ambiguity without explicit layout inputs, we further integrate a badcase-driven Vision-Language Model (VLM) for precise, context-aware prompt synthesis. In the second stage, we design a Fine-Grained Group-Level Direct Preference Optimization (DPO) with a weighted margin formulation to simultaneously eliminate multi-subject artifacts, elevate texture harmony, and recalibrate identity fidelity towards real-world distributions. Extensive experiments on two challenging benchmarks -- direct multi-person fusion and age-transformed group generation -- demonstrate that IdGlow fundamentally mitigates the stability-plasticity conflict, achieving a superior Pareto balance between state-of-the-art facial fidelity and commercial-grade aesthetic quality. 

---
# Learning to Explore: Policy-Guided Outlier Synthesis for Graph Out-of-Distribution Detection 

**Authors**: Li Sun, Lanxu Yang, Jiayu Tian, Bowen Fang, Xiaoyan Yu, Junda Ye, Peng Tang, Hao Peng, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00602)  

**Abstract**: Detecting out-of-distribution (OOD) graphs is crucial for ensuring the safety and reliability of Graph Neural Networks. In unsupervised graph-level OOD detection, models are typically trained using only in-distribution (ID) data, resulting in incomplete feature space characterization and weak decision boundaries. Although synthesizing outliers offers a promising solution, existing approaches rely on fixed, non-adaptive sampling heuristics (e.g., distance- or density-based), limiting their ability to explore informative OOD regions. We propose a Policy-Guided Outlier Synthesis (PGOS) framework that replaces static heuristics with a learned exploration strategy. Specifically, PGOS trains a reinforcement learning agent to navigate low-density regions in a structured latent space and sample representations that most effectively refine the OOD decision boundary. These representations are then decoded into high-quality pseudo-OOD graphs to improve detector robustness. Extensive experiments demonstrate that PGOS achieves state-of-the-art performance on multiple graph OOD and anomaly detection benchmarks. 

---
# Theory of Code Space: Do Code Agents Understand Software Architecture? 

**Authors**: Grigory Sapunov  

**Link**: [PDF](https://arxiv.org/pdf/2603.00601)  

**Abstract**: AI code agents excel at isolated tasks yet struggle with complex, multi-file software engineering requiring understanding of how dozens of modules relate. We hypothesize these failures stem from inability to construct, maintain, and update coherent architectural beliefs during codebase exploration. We introduce Theory of Code Space (ToCS), a benchmark that evaluates this capability by placing agents in procedurally generated codebases under partial observability, requiring them to build structured belief states over module dependencies, cross-cutting invariants, and design intent. The framework features: (1) a procedural codebase generator producing medium-complexity Python projects with four typed edge categories reflecting different discovery methods -- from syntactic imports to config-driven dynamic wiring -- with planted architectural constraints and verified ground truth; (2) a partial observability harness where agents explore under a budget; and (3) periodic belief probing via structured JSON, producing a time-series of architectural understanding. We decompose the Active-Passive Gap from spatial reasoning benchmarks into selection and decision components, and introduce Architectural Constraint Discovery as a code-specific evaluation dimension. Preliminary experiments with four rule-based baselines and five frontier LLM agents from three providers validate discriminative power: methods span a wide performance range (F1 from 0.129 to 0.646), LLM agents discover semantic edge types invisible to all baselines, yet weaker models score below simple heuristics -- revealing that belief externalization, faithfully serializing internal understanding into structured JSON, is itself a non-trivial capability and a first-order confounder in belief-probing benchmarks. Open-source toolkit: this https URL 

---
# LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models 

**Authors**: Yuchen Hou, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00592)  

**Abstract**: Vision-Language-Action (VLA) models achieve over 95% success on standard benchmarks. However, through systematic experiments, we find that current state-of-the-art VLA models largely ignore language instructions. Prior work lacks: (1) systematic semantic perturbation diagnostics, (2) a benchmark that forces language understanding by design, and (3) linguistically diverse training data.
This paper constructs the LangGap benchmark, based on a four-dimensional semantic perturbation method -- varying instruction semantics while keeping the tabletop layout fixed -- revealing language understanding deficits in {\pi}0.5. Existing benchmarks like LIBERO assign only one task per layout, underutilizing available objects and target locations; LangGap fully diversifies pick-and-place tasks under identical layouts, forcing models to truly understand language.
Experiments show that targeted data augmentation can partially close the language gap -- success rate improves from 0% to 90% with single-task training, and 0% to 28% with multi-task training. However, as semantic diversity of extended tasks increases, model learning capacity proves severely insufficient; even trained tasks perform poorly. This reveals a fundamental challenge for VLA models in understanding diverse language instructions -- precisely the long-term value of LangGap. 

---
# AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution 

**Authors**: Cencen Liu, Dongyang Zhang, Wen Yin, Jielei Wang, Tianyu Li, Ji Guo, Wenbo Jiang, Guoqing Wang, Guoming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00589)  

**Abstract**: Visual autoregressive (VAR) models have recently emerged as a promising alternative for image generation, offering stable training, non-iterative inference, and high-fidelity synthesis through next-scale prediction. This encourages the exploration of VAR for image super-resolution (ISR), yet its application remains underexplored and faces two critical challenges: locality-biased attention, which fragments spatial structures, and residual-only supervision, which accumulates errors across scales, severely compromises global consistency of reconstructed images. To address these issues, we propose AlignVAR, a globally consistent visual autoregressive framework tailored for ISR, featuring two key components: (1) Spatial Consistency Autoregression (SCA), which applies an adaptive mask to reweight attention toward structurally correlated regions, thereby mitigating excessive locality and enhancing long-range dependencies; and (2) Hierarchical Consistency Constraint (HCC), which augments residual learning with full reconstruction supervision at each scale, exposing accumulated deviations early and stabilizing the coarse-to-fine refinement process. Extensive experiments demonstrate that AlignVAR consistently enhances structural coherence and perceptual fidelity over existing generative methods, while delivering over 10x faster inference with nearly 50% fewer parameters than leading diffusion-based approaches, establishing a new paradigm for efficient ISR. 

---
# DeepAFL: Deep Analytic Federated Learning 

**Authors**: Jianheng Tang, Yajiang Huang, Kejia Fan, Feijiang Han, Jiaxu Li, Jinfeng Xu, Run He, Anfeng Liu, Houbing Herbert Song, Huiping Zhuang, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00579)  

**Abstract**: Federated Learning (FL) is a popular distributed learning paradigm to break down data silo. Traditional FL approaches largely rely on gradient-based updates, facing significant issues about heterogeneity, scalability, convergence, and overhead, etc. Recently, some analytic-learning-based work has attempted to handle these issues by eliminating gradient-based updates via analytical (i.e., closed-form) solutions. Despite achieving superior invariance to data heterogeneity, these approaches are fundamentally limited by their single-layer linear model with a frozen pre-trained backbone. As a result, they can only achieve suboptimal performance due to their lack of representation learning capabilities. In this paper, to enable representable analytic models while preserving the ideal invariance to data heterogeneity for FL, we propose our Deep Analytic Federated Learning approach, named DeepAFL. Drawing inspiration from the great success of ResNet in gradient-based learning, we design gradient-free residual blocks in our DeepAFL with analytical solutions. We introduce an efficient layer-wise protocol for training our deep analytic models layer by layer in FL through least squares. Both theoretical analyses and empirical evaluations validate our DeepAFL's superior performance with its dual advantages in heterogeneity invariance and representation learning, outperforming state-of-the-art baselines by up to 5.68%-8.42% across three benchmark datasets. 

---
# Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation 

**Authors**: Jinhan Xu, Xing Tang, Houpeng Yang, Haoran Zhang, Shenghua Yuan, Jiatao Chen, Tianming Xi, Jing Wang, Jiaojiao Yu, Guangli Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00576)  

**Abstract**: Symbolic music generation is a challenging task in multimedia generation, involving long sequences with hierarchical temporal structures, long-range dependencies, and fine-grained local details. Though recent diffusion-based models produce high quality generations, they tend to suffer from high training and inference costs with long symbolic sequences due to iterative denoising and sequence-length-related costs. To deal with such problem, we put forth a diffusing strategy named SMDIM to combine efficient global structure construction and light local refinement. SMDIM uses structured state space models to capture long range musical context at near linear cost, and selectively refines local musical details via a hybrid refinement scheme. Experiments performed on a wide range of symbolic music datasets which encompass various Western classical music, popular music and traditional folk music show that the SMDIM model outperforms the other state-of-the-art approaches on both the generation quality and the computational efficiency, and it has robust generalization to underexplored musical styles. These results show that SMDIM offers a principled solution for long-sequence symbolic music generation, including associated attributes that accompany the sequences. We provide a project webpage with audio examples and supplementary materials at this https URL. 

---
# Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation 

**Authors**: Yongbo He, Zirun Guo, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00574)  

**Abstract**: Adapting pretrained multi-modal models to evolving test-time distributions, known as multi-modal test-time adaptation, presents a significant challenge. Existing methods frequently encounter negative transfer in the unbiased modality and catastrophic forgetting in the biased modality. To address these challenges, we propose Decoupling Adaptation for Stability and Plasticity (DASP), a novel diagnose-then-mitigate framework. Our analysis reveals a critical discrepancy within the unified latent space: the biased modality exhibits substantially higher interdimensional redundancy (i.e., strong correlations across feature dimensions) compared to the unbiased modality. Leveraging this insight, DASP identifies the biased modality and implements an asymmetric adaptation strategy. This strategy employs a decoupled architecture where each modality-specific adapter is divided into stable and plastic components. The asymmetric mechanism works as follows: for the biased modality, which requires plasticity, the plastic component is activated and updated to capture domain-specific information, while the stable component remains fixed. Conversely, for the unbiased modality, which requires stability, the plastic component is bypassed, and the stable component is updated using KL regularization to prevent negative transfer. This asymmetric design enables the model to adapt flexibly to new domains while preserving generalizable knowledge. Comprehensive evaluations on diverse multi-modal benchmarks demonstrate that DASP significantly outperforms state-of-the-art methods. 

---
# Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions 

**Authors**: Yunqing Liu, Yi Zhou, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.00568)  

**Abstract**: Molecule representation learning is crucial for understanding and predicting molecular properties. However, conventional atom-centric models, which treat chemical bonds merely as pairwise interactions, often overlook complex bond-level phenomena like resonance and stereoselectivity. This oversight limits their predictive accuracy for nuanced chemical behaviors. To address this limitation, we introduce \textbf{DeMol}, a dual-graph framework whose architecture is motivated by a rigorous information-theoretic analysis demonstrating the information gain from a bond-centric perspective. DeMol explicitly models molecules through parallel atom-centric and bond-centric channels. These are synergistically fused by multi-scale Double-Helix Blocks designed to learn intricate atom-atom, atom-bond, and bond-bond interactions. The framework's geometric consistency is further enhanced by a regularization term based on covalent radii to enforce chemically plausible structures. Comprehensive evaluations on diverse benchmarks, including PCQM4Mv2, OC20 IS2RE, QM9, and MoleculeNet, show that DeMol establishes a new state-of-the-art, outperforming existing methods. These results confirm the superiority of explicitly modelling bond information and interactions, paving the way for more robust and accurate molecular machine learning. 

---
# Learning to Attack: A Bandit Approach to Adversarial Context Poisoning 

**Authors**: Ray Telikani, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2603.00567)  

**Abstract**: Neural contextual bandits are vulnerable to adversarial attacks, where subtle perturbations to rewards, actions, or contexts induce suboptimal decisions. We introduce AdvBandit, a black-box adaptive attack that formulates context poisoning as a continuous-armed bandit problem, enabling the attacker to jointly learn and exploit the victim's evolving policy. The attacker requires no access to the victim's internal parameters, reward function, or gradient information; instead, it constructs a surrogate model using a maximum-entropy inverse reinforcement learning module from observed context-action pairs and optimizes perturbations against this surrogate using projected gradient descent. An upper confidence bound-aware Gaussian process guides arm selection. An attack-budget control mechanism is also introduced to limit detection risk and overhead. We provide theoretical guarantees, including sublinear attacker regret and lower bounds on victim regret linear in the number of attacks. Experiments on three real-world datasets (Yelp, MovieLens, and Disin) against various victim contextual bandits demonstrate that our attack model achieves higher cumulative victim regret than state-of-the-art baselines. 

---
# MIDAS: Multi-Image Dispersion and Semantic Reconstruction for Jailbreaking MLLMs 

**Authors**: Yilian Liu, Xiaojun Jia, Guoshun Nan, Jiuyang Lyu, Zhican Chen, Tao Guan, Shuyuan Luo, Zhongyi Zhai, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00565)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable performance but remain vulnerable to jailbreak attacks that can induce harmful content and undermine their secure deployment. Previous studies have shown that introducing additional inference steps, which disrupt security attention, can make MLLMs more susceptible to being misled into generating malicious content. However, these methods rely on single-image masking or isolated visual cues, which only modestly extend reasoning paths and thus achieve limited effectiveness, particularly against strongly aligned commercial closed-source models. To address this problem, in this paper, we propose Multi-Image Dispersion and Semantic Reconstruction (MIDAS), a multimodal jailbreak framework that decomposes harmful semantics into risk-bearing subunits, disperses them across multiple visual clues, and leverages cross-image reasoning to gradually reconstruct the malicious intent, thereby bypassing existing safety mechanisms. The proposed MIDAS enforces longer and more structured multi-image chained reasoning, substantially increases the model's reliance on visual cues while delaying the exposure of malicious semantics and significantly reducing the model's security attention, thereby improving the performance of jailbreak against advanced MLLMs. Extensive experiments across different datasets and MLLMs demonstrate that the proposed MIDAS outperforms state-of-the-art jailbreak attacks for MLLMs and achieves an average attack success rate of 81.46% across 4 closed-source MLLMs. Our code is available at this [link](this https URL). 

---
# Whisper-MLA: Reducing GPU Memory Consumption of ASR Models based on MHA2MLA Conversion 

**Authors**: Sen Zhang, Jianguo Wei, Wenhuan Lu, Xianghu Yue, Wei Li, Qiang Li, Pengcheng Zhao, Ming Cai, Luo Si  

**Link**: [PDF](https://arxiv.org/pdf/2603.00563)  

**Abstract**: The Transformer-based Whisper model has achieved state-of-the-art performance in Automatic Speech Recognition (ASR). However, its Multi-Head Attention (MHA) mechanism results in significant GPU memory consumption due to the linearly growing Key-Value (KV) cache usage, which is problematic for many applications especially with long-form audio. To address this, we introduce Whisper-MLA, a novel architecture that incorporates Multi-Head Latent Attention (MLA) into the Whisper model. Specifically, we adapt MLA for Whisper's absolute positional embeddings and systematically investigate its application across encoder self-attention, decoder self-attention, and cross-attention modules. Empirical results indicate that applying MLA exclusively to decoder self-attention yields the desired balance between performance and memory efficiency. Our proposed approach allows conversion of a pretrained Whisper model to Whisper-MLA with minimal fine-tuning. Extensive experiments on the LibriSpeech benchmark validate the effectiveness of this conversion, demonstrating that Whisper-MLA reduces the KV cache size by up to 87.5% while maintaining competitive accuracy. 

---
# Geometry OR Tracker: Universal Geometric Operating Room Tracking 

**Authors**: Yihua Shao, Kang Chen, Feng Xue, Siyu Chen, Long Bai, Hongyuan Yu, Hao Tang, Jinlin Wu, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2603.00560)  

**Abstract**: In operating rooms (OR), world-scale multi-view 3D tracking supports downstream applications such as surgeon behavior recognition, where physically meaningful quantities such as distances and motion statistics must be measured in meters. However, real clinical deployments rarely satisfy the geometric prerequisites for stable multi-view fusion and tracking: camera calibration and RGB-D registration are always unreliable, leading to cross-view geometric inconsistency that produces "ghosting" during fusion and degrades 3D trajectories in a shared OR coordinate frame. To address this, we introduce Geometry OR Tracker, a two-stage pipeline that first rectifies imprecise calibration into a scaleconsistent and geometrically consistent camera setup with a single global scale via a Multi-view Metric Geometry Rectification module, and then performs Occlusion-Robust 3D Point Tracking directly in the unified OR world frame. On the MM-OR benchmark, improved geometric consistency translates into tracking gains: our rectification front-end reduces cross-view depth disagreement by more than 30$\times$ compared to raw calibration. Ablation studies further demonstrate the relationship between calibration quality and tracking accuracy, showing that improved geometric consistency yields stronger world-frame tracking. 

---
# Are LLMs Reliable Code Reviewers? Systematic Overcorrection in Requirement Conformance Judgement 

**Authors**: Haolin Jin, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00539)  

**Abstract**: Large language models (LLMs) have become essential tools in software development, widely used for requirements engineering, code generation and review tasks. Software engineers often rely on LLMs to verify if code implementation satisfy task requirements, thereby ensuring code robustness and accuracy. However, it remains unclear whether LLMs can reliably determine code against the given task descriptions, which is usually in a form of natural language specifications. In this paper, we uncover a systematic failure of LLMs in matching code to natural language requirements. Specifically, with widely adopted benchmarks and unified prompts design, we demonstrate that LLMs frequently misclassify correct code implementation as non-compliant or defective. Surprisingly, we find that more detailed prompt design, particularly with those requiring explanations and proposed corrections, leads to higher misjudgment rates, highlighting critical reliability issues for LLM-based code assistants. We further analyze the mechanisms driving these failures and evaluate the reliability of rationale-required judgments. Building on these findings, we propose a Fix-guided Verification Filter that treats the model proposed fix as executable counterfactual evidence, and validates the original and revised implementations using benchmark tests and spec-constrained augmented tests. Our results expose previously under-explored limitations in LLM-based code review capabilities, and provide practical guidance for integrating LLM-based reviewers with safeguards in automated review and development pipelines. 

---
# CaptionFool: Universal Image Captioning Model Attacks 

**Authors**: Swapnil Parekh  

**Link**: [PDF](https://arxiv.org/pdf/2603.00529)  

**Abstract**: Image captioning models are encoder-decoder architectures trained on large-scale image-text datasets, making them susceptible to adversarial attacks. We present CaptionFool, a novel universal (input-agnostic) adversarial attack against state-of-the-art transformer-based captioning models. By modifying only 7 out of 577 image patches (approximately 1.2% of the image), our attack achieves 94-96% success rate in generating arbitrary target captions, including offensive content. We further demonstrate that CaptionFool can generate "slang" terms specifically designed to evade existing content moderation filters. Our findings expose critical vulnerabilities in deployed vision-language models and underscore the urgent need for robust defenses against such attacks. Warning: This paper contains model outputs which are offensive in nature. 

---
# CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles 

**Authors**: Swapnil Parekh  

**Link**: [PDF](https://arxiv.org/pdf/2603.00523)  

**Abstract**: Mechanistic circuit discovery is notoriously sensitive to arbitrary analyst choices, especially pruning thresholds and feature dictionaries, often yielding brittle "one-shot" explanations with no principled notion of uncertainty. We reframe circuit discovery as an uncertainty-quantification problem over these analytic degrees of freedom. Our method, CIRCUS, constructs an ensemble of attribution graphs by pruning a single raw attribution run under multiple configurations, assigns each edge a stability score (the fraction of configurations that retain it), and extracts a strict-consensus circuit consisting only of edges that appear in all views. This produces a threshold-robust "core" circuit while explicitly surfacing contingent alternatives and enabling rejection of low-agreement structure. CIRCUS requires no retraining and adds negligible overhead, since it aggregates structure across already-computed pruned graphs. On Gemma-2-2B and Llama-3.2-1B, strict consensus circuits are ~40x smaller than the union of all configurations while retaining comparable influence-flow explanatory power, and they outperform a same-edge-budget baseline (union pruned to match the consensus size). We further validate causal relevance with activation patching, where consensus-identified nodes consistently beat matched non-consensus controls (p=0.0004). Overall, CIRCUS provides a practical, uncertainty-aware framework for reporting trustworthy, auditable mechanistic circuits with an explicit core/contingent/noise decomposition. 

---
# Phys-Diff: A Physics-Inspired Latent Diffusion Model for Tropical Cyclone Forecasting 

**Authors**: Lei Liu, Xiaoning Yu, Kang Chen, Jiahui Huang, Tengyuan Liu, Hongwei Zhao, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.00521)  

**Abstract**: Tropical cyclone (TC) forecasting is critical for disaster warning and emergency response. Deep learning methods address computational challenges but often neglect physical relationships between TC attributes, resulting in predictions lacking physical consistency. To address this, we propose Phys-Diff, a physics-inspired latent diffusion model that disentangles latent features into task-specific components (trajectory, pressure, wind speed) and employs cross-task attention to introduce prior physics-inspired inductive biases, thereby embedding physically consistent dependencies among TC attributes. Phys-Diff integrates multimodal data including historical cyclone attributes, ERA5 reanalysis data, and FengWu forecast fields via a Transformer encoder-decoder architecture, further enhancing forecasting performance. Experiments demonstrate state-of-the-art performance on global and regional datasets. 

---
# FastBUS: A Fast Bayesian Framework for Unified Weakly-Supervised Learning 

**Authors**: Ziquan Wang, Haobo Wang, Ke Chen, Lei Feng, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00517)  

**Abstract**: Machine Learning often involves various imprecise labels, leading to diverse weakly supervised settings. While recent methods aim for universal handling, they usually suffer from complex manual pre-work, ignore the relationships between associated labels, or are unable to batch process due to computational design flaws, resulting in long running times. To address these limitations, we propose a novel general framework that efficiently infers latent true label distributions across various weak supervisions. Our key idea is to express the label brute-force search process as a probabilistic transition of label variables, compressing diverse weakly supervised DFS tree structures into a shared Bayesian network. From this, we derived a latent probability calculation algorithm based on generalized belief propagation and proposed two joint acceleration strategies: 1) introducing a low-rank assumption to approximate the transition matrix, reducing time complexity; 2) designing an end-to-end state evolution module to learn batch-scale transition matrices, facilitating multi-category batch processing. In addition, the equivalence of our method with the EM algorithm in most scenarios is further demonstrated. Extensive experiments show that our method achieves SOTA results under most weakly supervised settings, and achieves up to hundreds of times faster acceleration in running time compared to other general methods. 

---
# What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models 

**Authors**: Yingqi Fan, Junlong Tong, Anhao Zhao, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00510)  

**Abstract**: Multimodal large language models (MLLMs) project visual tokens into the embedding space of language models, yet the internal structuring and processing of visual semantics remain poorly understood. In this work, we introduce a two-fold analytical framework featuring a novel probing tool, $\textbf{EmbedLens}$, to conduct a fine-grained analysis. We uncover a pronounced semantic sparsity at the input level: visual tokens consistently partition into sink, dead, and alive categories. Remarkably, only the alive tokens, comprising $\approx60\%$ of the total input, carry image-specific meaning. Furthermore, using a targeted patch-compression benchmark, we demonstrate that these alive tokens already encode rich, fine-grained cues (e.g., objects, colors, and OCR) prior to entering the LLM. Internal visual computations (such as visual attention and feed-forward networks) are redundant for most standard tasks. For the small subset of highly vision-centric tasks that actually benefit from internal processing, we reveal that alive tokens naturally align with intermediate LLM layers rather than the initial embedding space, indicating that shallow-layer processing is unnecessary and that direct mid-layer injection is both sufficient. Ultimately, our findings provide a unified mechanistic view of visual token processing, paving the way for more efficient and interpretable MLLM architectures through selective token pruning, minimized visual computation, and mid-layer injection. The code is released at: this https URL. 

---
# WirelessAgent++: Automated Agentic Workflow Design and Benchmarking for Wireless Networks 

**Authors**: Jingwen Tong, Zijian Li, Fang Liu, Wei Guo, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00501)  

**Abstract**: The integration of large language models (LLMs) into wireless networks has sparked growing interest in building autonomous AI agents for wireless tasks. However, existing approaches rely heavily on manually crafted prompts and static agentic workflows, a process that is labor-intensive, unscalable, and often suboptimal. In this paper, we propose WirelessAgent++, a framework that automates the design of agentic workflows for various wireless tasks. By treating each workflow as an executable code composed of modular operators, WirelessAgent++ casts agent design as a program search problem and solves it with a domain-adapted Monte Carlo Tree Search (MCTS) algorithm. Moreover, we establish WirelessBench, a standardized multi-dimensional benchmark suite comprising Wireless Communication Homework (WCHW), Network Slicing (WCNS), and Mobile Service Assurance (WCMSA), covering knowledge reasoning, code-augmented tool use, and multi-step decision-making. Experiments demonstrate that \wap{} autonomously discovers superior workflows, achieving test scores of $78.37\%$ (WCHW), $90.95\%$ (WCNS), and $97.07\%$ (WCMSA), with a total search cost below $\$ 5$ per task. Notably, our approach outperforms state-of-the-art prompting baselines by up to $31\%$ and general-purpose workflow optimizers by $11.1\%$, validating its effectiveness in generating robust, self-evolving wireless agents. The code is available at this https URL. 

---
# A Polynomial-Time Axiomatic Alternative to SHAP for Feature Attribution 

**Authors**: Kazuhiro Hiraki, Shinichi Ishihara, Takumi Kongo, Junnosuke Shino  

**Link**: [PDF](https://arxiv.org/pdf/2603.00496)  

**Abstract**: In this paper, we provide a theoretically grounded and computationally efficient alternative to SHAP. To this end, we study feature attribution through the lens of cooperative game theory by formulating a class of XAI--TU games. Building on this formulation, we investigate equal-surplus-type and proportional-allocation-type attribution rules and propose a low-cost attribution rule, ESENSC_rev2, constructed by combining two polynomial-time closed-form rules while ensuring the null-player property in the XAI--TU domain.
Extensive experiments on tabular prediction tasks demonstrate that ESENSC_rev2 closely approximates exact SHAP while substantially improving scalability as the number of features increases. These empirical results indicate that equal-surplus-type attribution rules can achieve favorable trade-offs between computational cost and approximation accuracy in high-dimensional explainability settings.
To provide theoretical foundations for these findings, we establish an axiomatic characterization showing that ESENSC_rev2 is uniquely determined by efficiency, the null-player axiom, a restricted differential marginality principle, an intermediate inessential-game property, and axioms that reduce computational requirements. Our results suggest that axiomatically justified and computationally efficient attribution rules can serve as practical and theoretically principled substitutes for SHAP-based approximations in modern explainability pipelines. 

---
# ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models 

**Authors**: Riccardo de Lutio, Tobias Fischer, Yen-Yu Chang, Yuxuan Zhang, Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Katarina Tothova, Zan Gojcic, Haithem Turki  

**Link**: [PDF](https://arxiv.org/pdf/2603.00492)  

**Abstract**: Per-scene optimization methods such as 3D Gaussian Splatting provide state-of-the-art novel view synthesis quality but extrapolate poorly to under-observed areas. Methods that leverage generative priors to correct artifacts in these areas hold promise but currently suffer from two shortcomings. The first is scalability, as existing methods use image diffusion models or bidirectional video models that are limited in the number of views they can generate in a single pass (and thus require a costly iterative distillation process for consistency). The second is quality itself, as generators used in prior work tend to produce outputs that are inconsistent with existing scene content and fail entirely in completely unobserved regions. To solve these, we propose a two-stage pipeline that leverages two key insights. First, we train a powerful bidirectional generative model with a novel opacity mixing strategy that encourages consistency with existing observations while retaining the model's ability to extrapolate novel content in unseen areas. Second, we distill it into a causal auto-regressive model that generates hundreds of frames in a single pass. This model can directly produce novel views or serve as pseudo-supervision to improve the underlying 3D representation in a simple and highly efficient manner. We evaluate our method extensively and demonstrate that it can generate plausible reconstructions in scenarios where existing approaches fail completely. When measured on commonly benchmarked datasets, we outperform existing all existing baselines by a wide margin, exceeding prior state-of-the-art methods by 1-3 dB PSNR. 

---
# Dynamic Spatio-Temporal Graph Neural Network for Early Detection of Pornography Addiction in Adolescents Based on Electroencephalogram Signals 

**Authors**: Achmad Ardani Prasha, Clavino Ourizqi Rachmadi, Sabrina Laila Mutiara, Hilman Syachr Ramadhan, Chareyl Reinalyta Borneo, Saruni Dwiasnati  

**Link**: [PDF](https://arxiv.org/pdf/2603.00488)  

**Abstract**: Adolescent pornography addiction requires early detection based on objective neurobiological biomarkers because self-report is prone to subjective bias due to social stigma. Conventional machine learning has not been able to model dynamic functional connectivity of the brain that fluctuates temporally during addictive stimulus exposure. This study proposes a state-of-the-art Dynamic Spatio-Temporal Graph Neural Network (DST-GNN) that integrates Phase Lag Index (PLI)-based Graph Attention Network (GAT) for spatial modeling and Bidirectional Gated Recurrent Unit (BiGRU) for temporal dynamics. The dataset consists of 14 adolescents (7 addicted, 7 healthy) with 19-channel EEG across 9 experimental conditions. Leave-One-Subject-Out Cross Validation (LOSO-CV) evaluation shows F1-Score of 71.00%$\pm$12.10% and recall of 85.71%, a 104% improvement compared to baseline. Ablation study confirms temporal contribution of 21% and PLI graph construction of 57%. Frontal-central regions (Fz, Cz, C3, C4) are identified as dominant biomarkers with Beta contribution of 58.9% and Hjorth of 31.2%, while Cz-T7 connectivity is consistent as a trait-level biomarker for objective screening. 

---
# RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment 

**Authors**: Liyao Jiang, Ruichen Chen, Chao Gao, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00483)  

**Abstract**: Recent text-to-image (T2I) diffusion models achieve remarkable realism, yet faithful prompt-image alignment remains challenging, particularly for complex prompts with multiple objects, relations, and fine-grained attributes. Existing training-free inference-time scaling methods rely on fixed iteration budgets that cannot adapt to prompt difficulty, while reflection-tuned models require carefully curated reflection datasets and extensive joint fine-tuning of diffusion and vision-language models, often overfitting to reflection paths data and lacking transferability across models. We introduce RAISE (Requirement-Adaptive Self-Improving Evolution), a training-free, requirement-driven evolutionary framework for adaptive T2I generation. RAISE formulates image generation as a requirement-driven adaptive scaling process, evolving a population of candidates at inference time through a diverse set of refinement actions-including prompt rewriting, noise resampling, and instructional editing. Each generation is verified against a structured checklist of requirements, enabling the system to dynamically identify unsatisfied items and allocate further computation only where needed. This achieves adaptive test-time scaling that aligns computational effort with semantic query complexity. On GenEval and DrawBench, RAISE attains state-of-the-art alignment (0.94 overall GenEval) while incurring fewer generated samples (reduced by 30-40%) and VLM calls (reduced by 80%) than prior scaling and reflection-tuned baselines, demonstrating efficient, generalizable, and model-agnostic multi-round self-improvement. Code is available at this https URL. 

---
# Atomicity for Agents: Exposing, Exploiting, and Mitigating TOCTOU Vulnerabilities in Browser-Use Agents 

**Authors**: Linxi Jiang, Zhijie Liu, Haotian Luo, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00476)  

**Abstract**: Browser-use agents are widely used for everyday tasks. They enable automated interaction with web pages through structured DOM based interfaces or vision language models operating on page screenshots. However, web pages often change between planning and execution, causing agents to execute actions based on stale assumptions. We view this temporal mismatch as a time of check to time of use (TOCTOU) vulnerability in browser-use agents. Dynamic or adversarial web content can exploit this window to induce unintended actions. We present a large scale empirical study of TOCTOU vulnerabilities in browser-use agents using a benchmark that spans synthesized and real world websites. Using this benchmark, we evaluate 10 popular open source agents and show that TOCTOU vulnerabilities are widespread. We design a lightweight mitigation based on pre-execution validation. It monitors DOM and layout changes during planning and validates the page state immediately before action execution. This approach reduces the risk of insecure execution and mitigates unintended side effects in browser-use agents. 

---
# OPGAgent: An Agent for Auditable Dental Panoramic X-ray Interpretation 

**Authors**: Zhaolin Yu, Litao Yang, Ben Babicka, Ming Hu, Jing Hao, Anthony Huang, James Huang, Yueming Jin, Jiasong Wu, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2603.00462)  

**Abstract**: Orthopantomograms (OPGs) are the standard panoramic radiograph in dentistry, used for full-arch screening across multiple diagnostic tasks. While Vision Language Models (VLMs) now allow multi-task OPG analysis through natural language, they underperform task-specific models on most individual tasks. Agentic systems that orchestrate specialized tools offer a path to both versatility and accuracy, this approach remains unexplored in the field of dental imaging. To address this gap, we propose OPGAgent, a multi-tool agentic system for auditable OPG interpretation. OPGAgent coordinates specialized perception modules with a consensus mechanism through three components: (1) a Hierarchical Evidence Gathering module that decomposes OPG analysis into global, quadrant, and tooth-level phases with dynamically invoking tools, (2) a Specialized Toolbox encapsulating spatial, detection, utility, and expert zoos, and (3) a Consensus Subagent that resolves conflicts through anatomical constraints. We further propose OPG-Bench, a structured-report protocol based on (Location, Field, Value) triples derived from real clinical reports, which enables a comprehensive review of findings and hallucinations, extending beyond the limitations of VQA indicators. On our OPG-Bench and the public MMOral-OPG benchmark, OPGAgent outperforms current dental VLMs and medical agent frameworks across both structured-report and VQA evaluation. Code will be released upon acceptance. 

---
# Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training 

**Authors**: Xi Wang, Wenbo Lu, Shengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00454)  

**Abstract**: Generative Flow Networks (GFlowNets) enable fine-tuning large language models to approximate reward-proportional posteriors, but they remain prone to mode collapse, manifesting as prefix collapse and length bias. We attribute this to two factors: (i) weak credit assignment to early prefixes, and (ii) biased replay that induces a shifted, non-representative training flow distribution. We propose Rooted absorbed prefix Trajectory Balance RapTB, an objective that anchors subtrajectory supervision at the root and propagates terminal rewards to intermediate prefixes via absorbed suffix-based backups, providing dense prefix-level learning signals. To mitigate replay-induced distribution shift, we further introduce SubM, a submodular replay refresh strategy that promotes both high reward and diversity. Empirically, on tasks such as molecule generation with LLM using SMILES strings, RapTB combined with SubM consistently improves optimization performance and molecular diversity while preserving high validity. 

---
# Texterial: A Text-as-Material Interaction Paradigm for LLM-Mediated Writing 

**Authors**: Jocelyn Shen, Nicolai Marquardt, Hugo Romat, Ken Hinckley, Nathalie Riche, Fanny Chevalier  

**Link**: [PDF](https://arxiv.org/pdf/2603.00452)  

**Abstract**: What if text could be sculpted and refined like clay -- or cultivated and pruned like a plant? Texterial reimagines text as a material that users can grow, sculpt, and transform. Current generative-AI models enable rich text operations, yet rigid, linear interfaces often mask such capabilities. We explore how the text-as-material metaphor can reveal AI-enabled operations, reshape the writing process, and foster compelling user experiences. A formative study shows that users readily reason with text-as-material, informing a conceptual framework that explains how material metaphors shift mental models and bridge gulfs of envisioning, execution, and evaluation in LLM-mediated writing. We present the design and evaluation of two technical probes: Text as Clay, where users refine text through gestural sculpting, and Text as Plants, where ideas grow serendipitously over time. This work expands the design space of writing tools by treating text as a living, malleable medium. 

---
# HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning 

**Authors**: An Dang, Jayjun Lee, Mustafa Mukadam, X. Alice Wu, Bernadette Bucher, Manikantan Nambi, Nima Fazeli  

**Link**: [PDF](https://arxiv.org/pdf/2603.00446)  

**Abstract**: In this paper, we address the problem of tactile sim-to-real policy transfer for contact-rich tasks. Existing methods primarily focus on vision-based sensors and emphasize image rendering quality while providing overly simplistic models of force and shear. Consequently, these models exhibit a large sim-to-real gap for many dexterous tasks. Here, we present HydroShear, a non-holonomic hydroelastic tactile simulator that advances the state-of-the-art by modeling: a) stick-slip transitions, b) path-dependent force and shear build up, and c) full SE(3) object-sensor interactions. HydroShear extends hydroelastic contact models using Signed Distance Functions (SDFs) to track the displacements of the on-surface points of an indenter during physical interaction with the sensor membrane. Our approach generates physics-based, computationally efficient force fields from arbitrary watertight geometries while remaining agnostic to the underlying physics engine. In experiments with GelSight Minis, HydroShear more faithfully reproduces real tactile shear compared to existing methods. This fidelity enables zero-shot sim-to-real transfer of reinforcement learning policies across four tasks: peg insertion, bin packing, book shelving for insertion, and drawer pulling for fine gripper control under slip. Our method achieves a 93% average success rate, outperforming policies trained on tactile images (34%) and alternative shear simulation methods (58%-61%). 

---
# Mamba-CAD: State Space Model For 3D Computer-Aided Design Generative Modeling 

**Authors**: Xueyang Li, Yunzhong Lou, Yu Song, Xiangdong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.00439)  

**Abstract**: Computer-Aided Design (CAD) generative modeling has a strong and long-term application in the industry. Recently, the parametric CAD sequence as the design logic of an object has been widely mined by sequence models. However, the industrial CAD models, especially in component objects, are fine-grained and complex, requiring a longer parametric CAD sequence to define. To address the problem, we introduce Mamba-CAD, a self-supervised generative modeling for complex CAD models in the industry, which can model on a longer parametric CAD sequence. Specifically, we first design an encoder-decoder framework based on a Mamba architecture and pair it with a CAD reconstruction task for pre-training to model the latent representation of CAD models; and then we utilize the learned representation to guide a generative adversarial network to produce the fake representation of CAD models, which would be finally recovered into parametric CAD sequences via the decoder of MambaCAD. To train Mamba-CAD, we further create a new dataset consisting of 77,078 CAD models with longer parametric CAD sequences. Comprehensive experiments are conducted to demonstrate the effectiveness of our model under various evaluation metrics, especially in the generation length of valid parametric CAD sequences. The code and dataset can be achieved from this https URL. 

---
# ROKA: Robust Knowledge Unlearning against Adversaries 

**Authors**: Jinmyeong Shin, Joshua Tapia, Nicholas Ferreira, Gabriel Diaz, Moayed Daneshyari, Hyeran Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2603.00436)  

**Abstract**: The need for machine unlearning is critical for data privacy, yet existing methods often cause Knowledge Contamination by unintentionally damaging related knowledge. Such a degraded model performance after unlearning has been recently leveraged for new inference and backdoor attacks. Most studies design adversarial unlearning requests that require poisoning or duplicating training data. In this study, we introduce a new unlearning-induced attack model, namely indirect unlearning attack, which does not require data manipulation but exploits the consequence of knowledge contamination to perturb the model accuracy on security-critical predictions. To mitigate this attack, we introduce a theoretical framework that models neural networks as Neural Knowledge Systems. Based on this, we propose ROKA, a robust unlearning strategy centered on Neural Healing. Unlike conventional unlearning methods that only destroy information, ROKA constructively rebalances the model by nullifying the influence of forgotten data while strengthening its conceptual neighbors. To the best of our knowledge, our work is the first to provide a theoretical guarantee for knowledge preservation during unlearning. Evaluations on various large models, including vision transformers, multi-modal models, and large language models, show that ROKA effectively unlearns targets while preserving, or even enhancing, the accuracy of retained data, thereby mitigating the indirect unlearning attacks. 

---
# TAP-SLF: Parameter-Efficient Adaptation of Vision Foundation Models for Multi-Task Ultrasound Image Analysis 

**Authors**: Hui Wan, Libin Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.00433)  

**Abstract**: Executing multiple tasks simultaneously in medical image analysis, including segmentation, classification, detection, and regression, often introduces significant challenges regarding model generalizability and the optimization of shared feature representations. While Vision Foundation Models (VFMs) provide powerful general representations, full fine-tuning on limited medical data is prone to overfitting and incurs high computational costs. Moreover, existing parameter-efficient fine-tuning approaches typically adopt task-agnostic adaptation protocols, overlooking both task-specific mechanisms and the varying sensitivity of model layers during fine-tuning. In this work, we propose Task-Aware Prompting and Selective Layer Fine-Tuning (TAP-SLF), a unified framework for multi-task ultrasound image analysis. TAP-SLF incorporates task-aware soft prompts to encode task-specific priors into the input token sequence and applies LoRA to selected specific top layers of the encoder. This strategy updates only a small fraction of the VFM parameters while keeping the pre-trained backbone frozen. By combining task-aware prompts with selective high-layer fine-tuning, TAP-SLF enables efficient VFM adaptation to diverse medical tasks within a shared backbone. Results on the FMC_UIA 2026 Challenge test set, where TAP-SLF wins fifth place, combined with evaluations on the officially released training dataset using an 8:2 train-test split, demonstrate that task-aware prompting and selective layer tuning are effective strategies for efficient VFM adaptation. 

---
# Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models 

**Authors**: Hulingxiao He, Zhi Tan, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2603.00431)  

**Abstract**: A high-performing, general-purpose visual understanding model should map visual inputs to a taxonomic tree of labels, identify novel categories beyond the training set for which few or no publicly available images exist. Large Multimodal Models (LMMs) have achieved remarkable progress in fine-grained visual recognition (FGVR) for known categories. However, they remain limited in hierarchical visual recognition (HVR) that aims at predicting consistent label paths from coarse to fine categories, especially for novel categories. To tackle these challenges, we propose Taxonomy-Aware Representation Alignment (TARA), a simple yet effective strategy to inject taxonomic knowledge into LMMs. TARA leverages representations from biology foundation models (BFMs) that encode rich biological relationships through hierarchical contrastive learning. By aligning the intermediate representations of visual features with those of BFMs, LMMs are encouraged to extract discriminative visual cues well structured in the taxonomy tree. Additionally, we align the representations of the first answer token with the ground-truth label, flexibly bridging the gap between contextualized visual features and categories of varying granularity according to user intent. Experiments demonstrate that TARA consistently enhances LMMs' hierarchical consistency and leaf node accuracy, enabling reliable recognition of both known and novel categories within complex biological taxonomies. Code is available at this https URL. 

---
# Personalities at Play: Probing Alignment in AI Teammates 

**Authors**: Mohammad Amin Samadi, Nia Nixon  

**Link**: [PDF](https://arxiv.org/pdf/2603.00429)  

**Abstract**: Collaborative problem solving and learning are shaped by who or what is on the team. As large language models (LLMs) increasingly function as collaborators rather than tools, a key question is whether AI teammates can be aligned to express personality in predictable ways that matter for interaction and learning. We investigate AI personality alignment through a three-lens evaluation framework spanning self-perception (standardized self-report), behavioral expression (team dialogue), and reflective expression (memory construction). We first administered the Big Five Inventory (BFI-44) to LLM-based teammates across four providers (GPT-4o, Claude-3.7 Sonnet, Gemini-2.5 Pro, Grok-3), 32 high/low trait configurations, and multiple prompting strategies. LLMs produced sharply differentiated Big Five profiles, but prompt semantic richness added little beyond simple trait assignment, while provider differences and baseline "default" personalities were substantial. Role framing also mattered: several models refused the assessment without context, yet complied when framed as a collaborative teammate. We then simulated AI participation in authentic team transcripts using high-trait personas and analyzed both generated utterances and structured long-term memories with LIWC-22. Personality signals in conversation were generally subtle and most detectable for Extraversion, whereas memory representations amplified trait-specific signals, especially for Neuroticism, Conscientiousness, and Agreeableness; Openness remained difficult to elicit robustly. Together, results suggest that AI personality is measurable but multi-layered and context-dependent, and that evaluating personality-aligned AI teammates requires attention to memory and system-level design, not conversation-only behavior. 

---
# An Interpretable Local Editing Model for Counterfactual Medical Image Generation 

**Authors**: Hyungi Min, Taeseung You, Hangyeul Lee, Yeongjae Cho, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2603.00423)  

**Abstract**: Counterfactual medical image generation have emerged as a critical tool for enhancing AI-driven systems in medical domain by answering "what-if" questions. However, existing approaches face two fundamental limitations: First, they fail to prevent unintended modifications, resulting collateral changes in demographic attributes when only disease features should be affected. Second, they lack interpretability in their editing process, which significantly limits their utility in real-world medical applications. To address these limitations, we present InstructX2X, a novel interpretable local editing model for counterfactual medical image generation featuring Region-Specific Editing. This approach restricts modifications to specific regions, effectively preventing unintended changes while simultaneously providing a Guidance Map that offers inherently interpretable visual explanations of the editing process. Additionally, we introduce MIMIC-EDIT-INSTRUCTION, a dataset for counterfactual medical image generation derived from expert-verified medical VQA pairs. Through extensive experiments, InstructX2X achieve state-of-the-art performance across all major evaluation metrics. Our model successfully generates high-quality counterfactual chest X-ray images along with interpretable explanations. 

---
# TMR-VLA:Vision-Language-Action Model for Magnetic Motion Control of Tri-leg Silicone-based Soft Robot 

**Authors**: Ruijie Tang, Chi Kit Ng, Kaixuan Wu, Long Bai, Guankun Wang, Yiming Huang, Yupeng Wang, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2603.00420)  

**Abstract**: In-vivo environments, magnetically actuated soft robots offer advantages such as wireless operation and precise control, showing promising potential for painless detection and therapeutic procedures. We developed a trileg magnetically driven soft robot (TMR) whose multi-legged design enables more flexible gaits and diverse motion patterns. For the silicone made of reconfigurable soft robots, its navigation ability can be separated into sequential motions, namely squatting, rotation, lifting a leg, walking and so on. Its motion and behavior depend on its bending shapes. To bridge motion type description and specific low-level voltage control, we introduced TMR-VLA, an end-to-end multi-modal system for a trileg magnetic soft robot capable of performing hybrid motion types, which is promising for developing a navigation ability by adapting its shape to language-constrained motion types. The TMR-VLA deploys embodied endoluminal localization ability from EndoVLA, and fuses sequential frames and natural language commands as input. Low-level voltage output is generated based on the current observation state and specific motion type description. The result shows the TMR-VLA can predict how the voltage applied to TMR will change the dynamics of a silicon-made soft robot. The TMR-VLA reached a 74% average success rate. 

---
# MuonRec: Shifting the Optimizer Paradigm Beyond Adam in Scalable Generative Recommendation 

**Authors**: Rong Shan, Aofan Yu, Bo Chen, Kuo Cai, Qiang Luo, Ruiming Tang, Han Li, Weiwen Liu, Weinan Zhang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00416)  

**Abstract**: Recommender systems (RecSys) are increasingly emphasizing scaling, leveraging larger architectures and more interaction data to improve personalization. Yet, despite the optimizer's pivotal role in training, modern RecSys pipelines almost universally default to Adam/AdamW, with limited scrutiny of whether these choices are truly optimal for recommendation. In this work, we revisit optimizer design for scalable recommendation and introduce MuonRec, the first framework that brings the recently proposed Muon optimizer to RecSys training. Muon performs orthogonalized momentum updates for 2D weight matrices via Newton-Schulz iteration, promoting diverse update directions and improving optimization efficiency. We develop an open-source training recipe for recommendation models and evaluate it across both traditional sequential recommenders and modern generative recommenders. Extensive experiments demonstrate that MuonRec reduces converged training steps by an average of 32.4\% while simultaneously improving final ranking quality. Specifically, MuonRec yields consistent relative gains in NDCG@10, averaging 12.6\% across all settings, with particularly pronounced improvements in generative recommendation models. These results consistently outperform strong Adam/AdamW baselines, positioning Muon as a promising new optimizer standard for RecSys training. Our code is available. 

---
# Exact and Asymptotically Complete Robust Verifications of Neural Networks via Quantum Optimization 

**Authors**: Wenxin Li, Wenchao Liu, Chuan Wang, Qi Gao, Yin Ma, Hai Wei, Kai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00408)  

**Abstract**: Deep neural networks (DNNs) enable high performance across domains but remain vulnerable to adversarial perturbations, limiting their use in safety-critical settings. Here, we introduce two quantum-optimization-based models for robust verification that reduce the combinatorial burden of certification under bounded input perturbations. For piecewise-linear activations (e.g., ReLU and hardtanh), our first model yields an exact formulation that is sound and complete, enabling precise identification of adversarial examples. For general activations (including sigmoid and tanh), our second model constructs scalable over-approximations via piecewise-constant bounds and is asymptotically complete, with approximation error vanishing as the segmentation is refined. We further integrate Quantum Benders Decomposition with interval arithmetic to accelerate solving, and propose certificate-transfer bounds that relate robustness guarantees of pruned networks to those of the original model. Finally, a layerwise partitioning strategy supports a quantum--classical hybrid workflow by coupling subproblems across depth. Experiments on robustness benchmarks show high certification accuracy, indicating that quantum optimization can serve as a principled primitive for robustness guarantees in neural networks with complex activations. 

---
# USE: Uncertainty Structure Estimation for Robust Semi-Supervised Learning 

**Authors**: Tsao-Lun Chen, Chien-Liang Liu, Tzu-Ming Harry Hsu, Tai-Hsien Wu, Chi-Cheng Fu, Han-Yi E. Chou, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.00404)  

**Abstract**: In this study, a novel idea, Uncertainty Structure Estimation (USE), a lightweight, algorithm-agnostic procedure that emphasizes the often-overlooked role of unlabeled data quality is introduced for Semi-supervised learning (SSL). SSL has achieved impressive progress, but its reliability in deployment is limited by the quality of the unlabeled pool. In practice, unlabeled data are almost always contaminated by out-of-distribution (OOD) samples, where both near-OOD and far-OOD can negatively affect performance in different ways. We argue that the bottleneck does not lie in algorithmic design, but rather in the absence of principled mechanisms to assess and curate the quality of unlabeled data. The proposed USE trains a proxy model on the labeled set to compute entropy scores for unlabeled samples, and then derives a threshold, via statistical comparison against a reference distribution, that separates informative (structured) from uninformative (structureless) samples. This enables assessment as a preprocessing step, removing uninformative or harmful unlabeled data before SSL training begins. Through extensive experiments on imaging (CIFAR-100) and NLP (Yelp Review) data, it is evident that USE consistently improves accuracy and robustness under varying levels of OOD contamination. Thus, it can be concluded that the proposed approach reframes unlabeled data quality control as a structural assessment problem, and considers it as a necessary component for reliable and efficient SSL in realistic mixed-distribution environments. 

---
# Hereditary Geometric Meta-RL: Nonlocal Generalization via Task Symmetries 

**Authors**: Paul Nitschke, Shahriar Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2603.00396)  

**Abstract**: Meta-Reinforcement Learning (Meta-RL) commonly generalizes via smoothness in the task encoding. While this enables local generalization around each training task, it requires dense coverage of the task space and leaves richer task space structure untapped. In response, we develop a geometric perspective that endows the task space with a "hereditary geometry" induced by the inherent symmetries of the underlying system. Concretely, the agent reuses a policy learned at the train time by transforming states and actions through actions of a Lie group. This converts Meta-RL into symmetry discovery rather than smooth extrapolation, enabling the agent to generalize to wider regions of the task space. We show that when the task space is inherited from the symmetries of the underlying system, the task space embeds into a subgroup of those symmetries whose actions are linearizable, connected, and compact--properties that enable efficient learning and inference at the test time. To learn these structures, we develop a differential symmetry discovery method. This collapses functional invariance constraints and thereby improves numerical stability and sample efficiency over functional approaches. Empirically, on a two-dimensional navigation task, our method efficiently recovers the ground-truth symmetry and generalizes across the entire task space, while a common baseline generalizes only near training tasks. 

---
# Verifier-Bound Communication for LLM Agents: Certified Bounds on Covert Signaling 

**Authors**: Om Tailor  

**Link**: [PDF](https://arxiv.org/pdf/2603.00381)  

**Abstract**: Colluding language-model agents can hide coordination in messages that remain policy-compliant at the surface level. We present CLBC, a protocol where generation and admission are separated: a message is admitted to transcript state only if a small verifier accepts a proof-bound envelope under a pinned predicate $\Pi$. The predicate binds policy hash, public randomness schedule, transcript chaining, latent schema constraints, canonical metadata/tool fields, and deterministic rejection codes. We show how this protocol yields an upper bound on transcript leakage in terms of latent leakage plus explicit residual channels, derive adaptive composition guarantees, and state a semantic lower bound when policy-valid alternatives remain choosable. We report extensive empirically grounded evidence: aggregate evaluation satisfies all prespecified thresholds; strict lane decoder advantage is bounded at 0.0000 with MI proxy 0.0636; adaptive-colluder stress tests remain below attacker thresholds; and baseline separation shows large gaps between reject-by-default semantics and audit-only controls. We further quantify operational tradeoffs. Strict full-proof mode has median turn latency 27.53s (p95 28.08s), while sampled proving reduces non-proved-turn latency to 0.327ms. The central finding is that bottlenecks alone are insufficient: security claims depend on verifiable admission semantics that are online, deterministic, and fail-closed. 

---
# Quantifying Catastrophic Forgetting in IoT Intrusion Detection Systems 

**Authors**: Sourasekhar Banerjee, David Bergqvist, Salman Toor, Christian Rohner, Andreas Johnsson  

**Link**: [PDF](https://arxiv.org/pdf/2603.00363)  

**Abstract**: Distribution shifts in attack patterns within RPL-based IoT networks pose a critical threat to the reliability and security of large-scale connected systems. Intrusion Detection Systems (IDS) trained on static datasets often fail to generalize to unseen threats and suffer from catastrophic forgetting when updated with new attacks. Ensuring continual adaptability of IDS is therefore essential for maintaining robust IoT network defence. In this focused study, we formulate intrusion detection as a domain continual learning problem and propose a method-agnostic IDS framework that can integrate diverse continual learning strategies. We systematically benchmark five representative approaches across multiple domain-ordering sequences using a comprehensive multi-attack dataset comprising 48 domains. Results show that continual learning mitigates catastrophic forgetting while maintaining a balance between plasticity, stability, and efficiency, a crucial aspect for resource-constrained IoT environments. Among the methods, Replay-based approaches achieve the best overall performance, while Synaptic Intelligence (SI) delivers near-zero forgetting with high training efficiency, demonstrating strong potential for stable and sustainable IDS deployment in dynamic IoT networks. 

---
# Acoustic Sensing for Universal Jamming Grippers 

**Authors**: Lion Weber, Theodor Wienert, Martin Splettstößer, Alexander Koenig, Oliver Brock  

**Link**: [PDF](https://arxiv.org/pdf/2603.00351)  

**Abstract**: Universal jamming grippers excel at grasping unknown objects due to their compliant bodies. Traditional tactile sensors can compromise this compliance, reducing grasping performance. We present acoustic sensing as a form of morphological sensing, where the gripper's soft body itself becomes the sensor. A speaker and microphone are placed inside the gripper cavity, away from the deformable membrane, fully preserving compliance. Sound propagates through the gripper and object, encoding object properties, which are then reconstructed via machine learning. Our sensor achieves high spatial resolution in sensing object size (2.6 mm error) and orientation (0.6 deg error), remains robust to external noise levels of 80 dBA, and discriminates object materials (up to 100% accuracy) and 16 everyday objects (85.6% accuracy). We validate the sensor in a realistic tactile object sorting task, achieving 53 minutes of uninterrupted grasping and sensing, confirming the preserved grasping performance. Finally, we demonstrate that disentangled acoustic representations can be learned, improving robustness to irrelevant acoustic variations. 

---
# Challenges in Enabling Private Data Valuation 

**Authors**: Yiwei Fu, Tianhao Wang, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2603.00342)  

**Abstract**: Data valuation methods quantify how individual training examples contribute to a model's behavior, and are increasingly used for dataset curation, auditing, and emerging data markets. As these techniques become operational, they raise serious privacy concerns: valuation scores can reveal whether a person's data was included in training, whether it was unusually influential, or what sensitive patterns exist in proprietary datasets. This motivates the study of privacy-preserving data valuation. However, privacy is fundamentally in tension with valuation utility under differential privacy (DP). DP requires outputs to be insensitive to any single record, while valuation methods are explicitly designed to measure per-record influence. As a result, naive privatization often destroys the fine-grained distinctions needed to rank or attribute value, particularly in heterogeneous datasets where rare examples exert outsized effects. In this work, we analyze the feasibility of DP-compatible data valuation. We identify the core algorithmic primitives across common valuation frameworks that induce prohibitive sensitivity, explaining why straightforward DP mechanisms fail. We further derive design principles for more privacy-amenable valuation procedures and empirically characterize how privacy constraints degrade ranking fidelity across representative methods and datasets. Our results clarify the limits of current approaches and provide a foundation for developing valuation methods that remain useful under rigorous privacy guarantees. 

---
# AESP: A Human-Sovereign Economic Protocol for AI Agents with Privacy-Preserving Settlement 

**Authors**: Jian Sheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00318)  

**Abstract**: As AI agents increasingly perform economic tasks on behalf of humans, a fundamental tension arises between agent autonomy and human control over financial assets. We present the Agent Economic Sovereignty Protocol (AESP), a layered protocol in which agents transact autonomously at machine speed on crypto-native infrastructure while remaining cryptographically bound to human-defined governance boundaries. AESP enforces the invariant that agents are economically capable but never economically sovereign through five mechanisms: (1) a deterministic eight-check policy engine with tiered escalation; (2) human-in-the-loop review with automatic, explicit, and biometric tiers; (3) EIP-712 dual-signed commitments with escrow; (4) HKDF-based context-isolated privacy with batched consolidation; and (5) an ACE-GF-based cryptographic substrate. We formalize two testable hypotheses on security coverage and latency overhead, and specify a complete evaluation methodology with baselines and ablation design. The protocol is implemented as an open-source TypeScript SDK (208 tests, ten modules) with interoperability via MCP and A2A. 

---
# When Metrics Disagree: Automatic Similarity vs. LLM-as-a-Judge for Clinical Dialogue Evaluation 

**Authors**: Bian Sun, Zhenjian Wang, Orvill de la Torre, Zirui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00314)  

**Abstract**: This paper details the baseline model selection, fine-tuning process, evaluation methods, and the implications of deploying more accurate LLMs in healthcare settings. As large language models (LLMs) are increasingly employed to address diverse problems, including medical queries, concerns about their reliability have surfaced. A recent study by Long Island University highlighted that LLMs often perform poorly in medical contexts, potentially leading to harmful misguidance for users. To address this, our research focuses on fine-tuning the Llama 2 7B, a transformer-based, decoder-only model, using transcripts from real patient-doctor interactions. Our objective was to enhance the model's accuracy and precision in responding to medical queries. We fine-tuned the model using a supervised approach, emphasizing domain-specific nuances captured in the training data. In the best scenario, the model results should be reviewed and evaluated by real medical experts. Due to resource constraints, the performance of the fine-tuned model was evaluated using text similarity metrics. The fine-tuned model demonstrated significant improvements across all key dimensions except GPT-4's evaluation. The evaluations of ChatGPT4 are quite different from the quantitative results; here, we not only suggest, but also propose that the result should be evaluated by human medical experts. 

---
# Polynomial Surrogate Training for Differentiable Ternary Logic Gate Networks 

**Authors**: Sai Sandeep Damera, Ryan Matheu, Aniruddh G. Puranic, John S. Baras  

**Link**: [PDF](https://arxiv.org/pdf/2603.00302)  

**Abstract**: Differentiable logic gate networks (DLGNs) learn compact, interpretable Boolean circuits via gradient-based training, but all existing variants are restricted to the 16 two-input binary gates. Extending DLGNs to Ternary Kleene $K_3$ logic and training DTLGNs where the UNKNOWN state enables principled abstention under uncertainty is desirable. However, the support set of potential gates per neuron explodes to $19{,}683$, making the established softmax-over-gates training approach intractable. We introduce Polynomial Surrogate Training (PST), which represents each ternary neuron as a degree-$(2,2)$ polynomial with 9 learnable coefficients (a $2{,}187\times$ parameter reduction) and prove that the gap between the trained network and its discretized logic circuit is bounded by a data-independent commitment loss that vanishes at convergence. Scaling experiments from 48K to 512K neurons on CIFAR-10 demonstrate that this hardening gap contracts with overparameterization. Ternary networks train $2$-$3\times$ faster than binary DLGNs and discover true ternary gates that are functionally diverse. On synthetic and tabular tasks we find that the UNKNOWN output acts as a Bayes-optimal uncertainty proxy, enabling selective prediction in which ternary circuits surpass binary accuracy once low-confidence predictions are filtered. More broadly, PST establishes a general polynomial-surrogate methodology whose parameterization cost grows only quadratically with logic valence, opening the door to many-valued differentiable logic. 

---
# Stepwise Penalization for Length-Efficient Chain-of-Thought Reasoning 

**Authors**: Xintong Li, Sha Li, Rongmei Lin, Hongye Jin, Linwei Li, Hejie Cui, Sarah Zhang, Chia-Yuan Chang, Kewei Cheng, Besnik Fetahu, Priyanka Nigam, Jingbo Shang, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00296)  

**Abstract**: Large reasoning models improve with more test-time computation, but often overthink, producing unnecessarily long chains-of-thought that raise cost without improving accuracy. Prior reinforcement learning approaches typically rely on a single outcome reward with trajectory-level length penalties, which cannot distinguish essential from redundant reasoning steps and therefore yield blunt compression. Although recent work incorporates step-level signals, such as offline pruning, supervised data construction, or verifier-based intermediate rewards, reasoning length is rarely treated as an explicit step-level optimization objective during RL. We propose Step-wise Adaptive Penalization (SWAP), a fine-grained framework that allocates length reduction across steps based on intrinsic contribution. We estimate step importance from the model's on-policy log-probability improvement toward the correct answer, then treat excess length as a penalty mass redistributed to penalize low-importance steps more heavily while preserving high-importance reasoning. We optimize with a unified outcome-process advantage within group-relative policy optimization. Extensive experiments demonstrate that SWAP reduces reasoning length by 64.3% on average while improving accuracy by 5.7% relative to the base model. 

---
# Transformers Remember First, Forget Last: Dual-Process Interference in LLMs 

**Authors**: Sourav Chattaraj, Kanak Raj  

**Link**: [PDF](https://arxiv.org/pdf/2603.00270)  

**Abstract**: When large language models encounter conflicting information in context, which memories survive -- early or recent? We adapt classical interference paradigms from cognitive psychology to answer this question, testing 39 LLMs across diverse architectures and scales. Every model shows the same pattern: proactive interference (PI) dominates retroactive interference (RI) universally (Cohen's d = 1.73, p < 0.0001), meaning early encodings are protected at the cost of recent information -- the opposite of human memory, where RI typically dominates. Three findings indicate that RI and PI reflect separate memory mechanisms. RI and PI are uncorrelated (R^2 = 0.044), rejecting a unified "memory capacity." Model size predicts RI resistance (R^2 = 0.49) but not PI (R^2 = 0.06, n.s.) -- only RI is capacity-dependent. And error analysis reveals distinct failure modes: RI failures are passive retrieval failures (51%), while PI failures show active primacy intrusion (56%); both show <1% hallucination. These patterns parallel the consolidation-retrieval distinction in cognitive science, suggesting that transformer attention creates a primacy bias with direct implications for interference-heavy applications. 

---
# GENAI WORKBENCH: AI-Assisted Analysis and Synthesis of Engineering Systems from Multimodal Engineering Data 

**Authors**: H. Sinan Bank, Daniel R. Herber  

**Link**: [PDF](https://arxiv.org/pdf/2603.00251)  

**Abstract**: Modern engineering design platforms excel at discipline-specific tasks such as CAD, CAM, and CAE, but often lack native systems engineering frameworks. This creates a disconnect where system-level requirements and architectures are managed separately from detailed component design, hindering holistic development and increasing integration risks. To address this, we present the conceptual framework for the GenAI Workbench, a Model-Based Systems Engineering (MBSE) environment that integrates systems engineering principles into the designer's workflow. Built on an open-source PLM platform, it establishes a unified digital thread by linking semantic data from documents, physical B-rep geometry, and relational system graphs. The workbench facilitates an AI-assisted workflow where a designer can ingest source documents, from which the system automatically extracts requirements and uses vision-language models to generate an initial system architecture, such as a Design Structure Matrix (DSM). This paper presents the conceptual architecture, proposed methodology, and anticipated impact of this work-in-progress framework, which aims to foster a more integrated, data-driven, and informed engineering design methodology. 

---
# Empowering Future Cybersecurity Leaders: Advancing Students through FINDS Education for Digital Forensic Excellence 

**Authors**: Yashas Hariprasad, Subhash Gurappa, Sundararaj S. Iyengar, Jerry F. Miller, Pronab Mohanty, Naveen Kumar Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2603.00222)  

**Abstract**: The Forensics Investigations Network in Digital Sciences (FINDS) Research Center of Excellence (CoE), funded by the U.S. Army Research Laboratory, advances Digital Forensic Engineering Education (DFEE) through an integrated research education framework for AI enabled cybersecurity workforce development. FINDS combines high performance computing (HPC), secure software engineering, adversarial analytics, and experiential learning to address emerging cyber and synthetic media threats. This paper introduces the Multidependency Capacity Building Skills Graph (MCBSG), a directed acyclic graph based model that encodes hierarchical and cross domain dependencies among competencies in AI-driven forensic programming, statistical inference, digital evidence processing, and threat detection. The MCBSG enables structured modeling of skill acquisition pathways and quantitative capacity assessment. Supervised machine learning methods, including entropy-based Decision Tree Classifiers and regression modeling, are applied to longitudinal multi cohort datasets capturing mentoring interactions, laboratory performance metrics, curriculum artifacts, and workshop participation. Feature importance analysis and cross validation identify key predictors of technical proficiency and research readiness. Three year statistical evaluation demonstrates significant gains in forensic programming accuracy, adversarial reasoning, and HPC-enabled investigative workflows. Results validate the MCBSG as a scalable, interpretable framework for data-driven, inclusive cybersecurity education aligned with national defense workforce priorities. 

---
# Physical Evaluation of Naturalistic Adversarial Patches for Camera-Based Traffic-Sign Detection 

**Authors**: Brianna D'Urso, Tahmid Hasan Sakib, Syed Rafay Hasan, Terry N. Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.00217)  

**Abstract**: This paper studies how well Naturalistic Adversarial Patches (NAPs) transfer to a physical traffic sign setting when the detector is trained on a customized dataset for an autonomous vehicle (AV) environment. We construct a composite dataset, CompGTSRB (which is customized dataset for AV environment), by pasting traffic sign instances from the German Traffic Sign Recognition Benchmark (GTSRB) onto undistorted backgrounds captured from the target platform. CompGTSRB is used to train a YOLOv5 model and generate patches using a Generative Adversarial Network (GAN) with latent space optimization, following existing NAP methods. We carried out a series of experiments on our Quanser QCar testbed utilizing the front CSI camera provided in QCar. Across configurations, NAPs reduce the detector's STOP class confidence. Different configurations include distance, patch sizes, and patch placement. These results along with a detailed step-by-step methodology indicate the utility of CompGTSRB dataset and the proposed systematic physical protocols for credible patch evaluation. The research further motivate researching the defenses that address localized patch corruption in embedded perception pipelines. 

---
# Agentic Scientific Simulation: Execution-Grounded Model Construction and Reconstruction 

**Authors**: Knut-Andreas Lie, Olav Møyner, Elling Svee, Jakob Torben  

**Link**: [PDF](https://arxiv.org/pdf/2603.00214)  

**Abstract**: LLM agents are increasingly used for code generation, but physics-based simulation poses a deeper challenge: natural-language descriptions of simulation models are inherently underspecified, and different admissible resolutions of implicit choices produce physically valid but scientifically distinct configurations. Without explicit detection and resolution of these ambiguities, neither the correctness of the result nor its reproducibility from the original description can be assured.
This paper investigates agentic scientific simulation, where model construction is organized as an execution-grounded interpret-act-validate loop and the simulator serves as the authoritative arbiter of physical validity rather than merely a runtime. We present JutulGPT, a reference implementation built on the fully differentiable Julia-based reservoir simulator JutulDarcy. The agent combines structured retrieval of documentation and examples with code synthesis, static analysis, execution, and systematic interpretation of solver diagnostics. Underspecified modelling choices are detected explicitly and resolved either autonomously (with logged assumptions) or through targeted user queries.
The results demonstrate that agent-mediated model construction can be grounded in simulator validation, while also revealing a structural limitation: choices resolved tacitly through simulator defaults are invisible to the assumption log and to any downstream representation. A secondary experiment with autonomous reconstruction of a reference model from progressively abstract textual descriptions shows that reconstruction variability exposes latent degrees of freedom in simulation descriptions and provides a practical methodology for auditing reproducibility. All code, prompts, and agent logs are publicly available. 

---
# Universal NP-Hardness of Clustering under General Utilities 

**Authors**: Angshul Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2603.00210)  

**Abstract**: Clustering is a central primitive in unsupervised learning, yet practice is dominated by heuristics whose outputs can be unstable and highly sensitive to representations, hyperparameters, and initialisation. Existing theoretical results are largely objective-specific and do not explain these behaviours at a unifying level. We formalise the common optimisation core underlying diverse clustering paradigms by defining the Universal Clustering Problem (UCP): the maximisation of a polynomial-time computable partition utility over a finite metric space. We prove the NP-hardness of UCP via two independent polynomial-time reductions from graph colouring and from exact cover by 3-sets (X3C). By mapping ten major paradigms -- including k-means, GMMs, DBSCAN, spectral clustering, and affinity propagation -- to the UCP framework, we demonstrate that each inherits this fundamental intractability. Our results provide a unified explanation for characteristic failure modes, such as local optima in alternating methods and greedy merge-order traps in hierarchical clustering. Finally, we show that clustering limitations reflect interacting computational and epistemic constraints, motivating a shift toward stability-aware objectives and interaction-driven formulations with explicit guarantees. 

---
# VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models 

**Authors**: Soumya Suvra Ghosal, Youngeun Kim, Zhuowei Li, Ritwick Chaudhry, Linghan Xu, Hongjing Zhang, Jakub Zablocki, Yifan Xing, Qin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00207)  

**Abstract**: Advances in large reasoning models have shown strong performance on complex reasoning tasks by scaling test-time compute through extended reasoning. However, recent studies observe that in vision-dependent tasks, extended textual reasoning at inference time can degrade performance as models progressively lose attention to visual tokens and increasingly rely on textual priors alone. To address this, prior works use reinforcement learning (RL)-based fine-tuning to route visual tokens or employ refocusing mechanisms during reasoning. While effective, these methods are computationally expensive, requiring large-scale data generation and policy optimization. To leverage the benefits of test-time compute without additional RL fine-tuning, we propose VisRef, a visually grounded test-time scaling framework. Our key idea is to actively guide the reasoning process by re-injecting a coreset of visual tokens that are semantically relevant to the reasoning context while remaining diverse and globally representative of the image, enabling more grounded multi-modal reasoning. Experiments on three visual reasoning benchmarks with state-of-the-art multi-modal large reasoning models demonstrate that, under fixed test-time compute budgets, VisRef consistently outperforms existing test-time scaling approaches by up to 6.4%. 

---
# TACIT Benchmark: A Programmatic Visual Reasoning Benchmark for Generative and Discriminative Models 

**Authors**: Daniel Nobrega Medeiros  

**Link**: [PDF](https://arxiv.org/pdf/2603.00206)  

**Abstract**: Existing visual reasoning benchmarks predominantly rely on natural language prompts, evaluate narrow reasoning modalities, or depend on subjective scoring procedures such as LLM-as-judge. We introduce the TACIT Benchmark, a programmatic visual reasoning benchmark comprising 10 tasks across 6 reasoning domains: spatial navigation, abstract pattern completion, causal simulation, logical constraint satisfaction, graph theory, and topology. The benchmark provides dual-track evaluation: a generative track in which models must produce solution images verified through deterministic computer-vision pipelines, and a discriminative track offering five-way multiple choice with structurally plausible near-miss distractors. Each distractor violates exactly one structural constraint, requiring models to reason about fine-grained visual differences rather than exploit superficial cues. Version 0.1.0 distributes 6,000 puzzles (108,000 PNG images across three resolutions) with fully deterministic seeded generation and reproducible verification. The dataset, generation code, and evaluation harness are released under the Apache 2.0 license on HuggingFace (DOI: https://doi.org/10.57967/hf/7904). 

---
# Efficient Flow Matching for Sparse-View CT Reconstruction 

**Authors**: Jiayang Shi, Lincen Yang, Zhong Li, Tristan Van Leeuwen, Daniel M. Pelt, K. Joost Batenburg  

**Link**: [PDF](https://arxiv.org/pdf/2603.00205)  

**Abstract**: Generative models, particularly Diffusion Models (DM), have shown strong potential for Computed Tomography (CT) reconstruction serving as expressive priors for solving ill-posed inverse problems. However, diffusion-based reconstruction relies on Stochastic Differential Equations (SDEs) for forward diffusion and reverse denoising, where such stochasticity can interfere with repeated data consistency corrections in CT reconstruction. Since CT reconstruction is often time-critical in clinical and interventional scenarios, improving reconstruction efficiency is essential. In contrast, Flow Matching (FM) models sampling as a deterministic Ordinary Differential Equation (ODE), yielding smooth trajectories without stochastic noise injection. This deterministic formulation is naturally compatible with repeated data consistency operations. Furthermore, we observe that FM-predicted velocity fields exhibit strong correlations across adjacent steps. Motivated by this, we propose an FM-based CT reconstruction framework (FMCT) and an efficient variant (EFMCT) that reuses previously predicted velocity fields over consecutive steps to substantially reduce the number of Neural network Function Evaluations (NFEs), thereby improving inference efficiency. We provide theoretical analysis showing that the error introduced by velocity reuse is bounded when combined with data consistency operations. Extensive experiments demonstrate that FMCT/EFMCT achieve competitive reconstruction quality while significantly improving computational efficiency compared with diffusion-based methods. The codebase is open-sourced at this https URL. 

---
# AdURA-Net: Adaptive Uncertainty and Region-Aware Network 

**Authors**: Antik Aich Roy, Ujjwal Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2603.00201)  

**Abstract**: One of the common issues in clinical decision-making is the presence of uncertainty, which often arises due to ambiguity in radiology reports, which often reflect genuine diagnostic uncertainty or limitations of automated label extraction in various complex cases. Especially the case of multilabel datasets such as CheXpert, MIMIC-CXR, etc., which contain labels such as positive, negative, and uncertain. In clinical decision-making, the uncertain label plays a tricky role as the model should not be forced to provide a confident prediction in the absence of sufficient evidence. The ability of the model to say it does not understand whenever it is not confident is crucial, especially in the cases of clinical decision-making involving high risks. Here, we propose AdURA-Net, a geometry-driven adaptive uncertainty-aware framework for reliable thoracic disease classification. The key highlights of the proposed model are: a) Adaptive dilated convolution and multiscale deformable alignment coupled with the backbone Densenet architecture capturing the anatomical complexities of the medical images, and b) Dual Head Loss, which combines masked binary cross entropy with logit and a Dirichlet evidential learning objective. 

---
# LiaisonAgent: An Multi-Agent Framework for Autonomous Risk Investigation and Governance 

**Authors**: Chuanming Tang, Ling Qing, Shifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00200)  

**Abstract**: The rapid evolution of sophisticated cyberattacks has strained modern Security Operations Centers (SOC), which traditionally rely on rule-based or signature-driven detection systems. These legacy frameworks often generate high volumes of technical alerts that lack organizational context, leading to analyst fatigue and delayed incident responses. This paper presents LiaisonAgent, an autonomous multi-agent system designed to bridge the gap between technical risk detection and business-level risk governance. Built upon the QWQ-32B large reasoning model, LiaisonAgent integrates specialized sub-agents, including human-computer interaction agents, comprehensive judgment agents, and automated disposal agents-to execute end-to-end investigation workflows. The system leverages a hybrid planning architecture that combines deterministic workflows for compliance with autonomous reasoning based on the ReAct paradigm to handle ambiguous operational scenarios. Experimental evaluations across diverse security contexts, such as large-scale data exfiltration and unauthorized account borrowing, achieve an end-to-end tool-calling success rate of 97.8% and a risk judgment accuracy of 95%. Furthermore, the system exhibits significant resilience against out-of-distribution noise and adversarial prompt injections, while achieving a 92.7% reduction in manual investigation overhead. 

---
# Multi-Condition Digital Twin Calibration for Axial Piston Pumps : Compound Fault Simulation 

**Authors**: Chang Dong, Jianfeng Tao, Chengliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00199)  

**Abstract**: Axial piston pumps are indispensable power sources in high-stakes fluid power systems, including aerospace, marine, and heavy machinery applications. Their operational reliability is frequently compromised by compound faults that simultaneously affect multiple friction pairs. Conventional data-driven diagnosis methods suffer from severe data scarcity for compound faults and poor generalization across varying operating conditions. This paper proposes a novel multi-condition physics-data coupled digital twin calibration framework that explicitly resolves the fundamental uncertainty of pump outlet flow ripple. The framework comprises three synergistic stages: in-situ virtual high-frequency flow sensing on a dedicated rigid metallic segment, surrogate model-assisted calibration of the 3D CFD source model using physically estimated ripple amplitudes, and multi-objective inverse transient analysis for viscoelastic unsteady-friction pipeline parameter identification. Comprehensive experiments on a test rig demonstrate that the calibrated digital twin accurately reproduces both single-fault and two representative compound-fault. These results establish a high-fidelity synthetic fault-generation capability that directly enables robust zero-shot fault diagnosis under previously unseen operating regimes and fault combinations, thereby advancing predictive maintenance in complex hydraulic systems. 

---
# Stateful Token Reduction for Long-Video Hybrid VLMs 

**Authors**: Jindong Jiang, Amala Sanjay Deshmukh, Kateryna Chumachenko, Karan Sapra, Zhiding Yu, Guilin Liu, Andrew Tao, Pavlo Molchanov, Jan Kautz, Wonmin Byeon  

**Link**: [PDF](https://arxiv.org/pdf/2603.00198)  

**Abstract**: Token reduction is an effective way to accelerate long-video vision-language models (VLMs), but most existing methods are designed for dense Transformers and do not directly account for hybrid architectures that interleave attention with linear-time state-space blocks (e.g., Mamba). We study query-conditioned token reduction for hybrid video VLMs and analyze reduction behavior through two properties: layerwise sparsity (how many tokens capture query-relevant information) and importance stability (whether token-importance rankings persist across depth). Although token importance is sparse within each layer, the set of important tokens changes across layers, so aggressive early pruning is unreliable. Motivated by this, we propose a low-to-high progressive reduction schedule and a unified language-aware scoring mechanism for both attention and Mamba blocks (using an implicit-attention proxy for Mamba), enabling all-layer token reduction in hybrids. Under an aggressive compression setting (retaining 25% of visual tokens), our approach delivers substantial prefilling speedups (3.8--4.2x) with near-baseline accuracy at test time, and light finetuning under reduction further improves performance on long-context video benchmarks. 

---
# A Case Study on Concept Induction for Neuron-Level Interpretability in CNN 

**Authors**: Moumita Sen Sarma, Samatha Ereshi Akkamahadevi, Pascal Hitzler  

**Link**: [PDF](https://arxiv.org/pdf/2603.00197)  

**Abstract**: Deep Neural Networks (DNNs) have advanced applications in domains such as healthcare, autonomous systems, and scene understanding, yet the internal semantics of their hidden neurons remain poorly understood. Prior work introduced a Concept Induction-based framework for hidden neuron analysis and demonstrated its effectiveness on the ADE20K dataset. In this case study, we investigate whether the approach generalizes by applying it to the SUN2012 dataset, a large-scale scene recognition benchmark. Using the same workflow, we assign interpretable semantic labels to neurons and validate them through web-sourced images and statistical testing. Our findings confirm that the method transfers to SUN2012, showing its broader applicability. 

---
# Your Inference Request Will Become a Black Box: Confidential Inference for Cloud-based Large Language Models 

**Authors**: Chung-ju Huang, Huiqiang Zhao, Yuanpeng He, Lijian Li, Wenpin Jiao, Zhi Jin, Peixuan Chen, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00196)  

**Abstract**: The increasing reliance on cloud-hosted Large Language Models (LLMs) exposes sensitive client data, such as prompts and responses, to potential privacy breaches by service providers. Existing approaches fail to ensure privacy, maintain model performance, and preserve computational efficiency simultaneously. To address this challenge, we propose Talaria, a confidential inference framework that partitions the LLM pipeline to protect client data without compromising the cloud's model intellectual property or inference quality. Talaria executes sensitive, weight-independent operations within a client-controlled Confidential Virtual Machine (CVM) while offloading weight-dependent computations to the cloud GPUs. The interaction between these environments is secured by our Reversible Masked Outsourcing (ReMO) protocol, which uses a hybrid masking technique to reversibly obscure intermediate data before outsourcing computations. Extensive evaluations show that Talaria can defend against state-of-the-art token inference attacks, reducing token reconstruction accuracy from over 97.5% to an average of 1.34%, all while being a lossless mechanism that guarantees output identical to the original model without significantly decreasing efficiency and scalability. To the best of our knowledge, this is the first work that ensures clients' prompts and responses remain inaccessible to the cloud, while also preserving model privacy, performance, and efficiency. 

---
# Formal Analysis and Supply Chain Security for Agentic AI Skills 

**Authors**: Varun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2603.00195)  

**Abstract**: The rapid proliferation of agentic AI skill ecosystems -- exemplified by OpenClaw (228,000 GitHub stars) and Anthropic Agent Skills (75,600 stars) -- has introduced a critical supply chain attack surface. The ClawHavoc campaign (January-February 2026) infiltrated over 1,200 malicious skills into the OpenClaw marketplace, while MalTool catalogued 6,487 malicious tools that evade conventional detection. In response, twelve reactive security tools emerged, yet all rely on heuristic methods that provide no formal guarantees. We present SkillFortify, the first formal analysis framework for agent skill supply chains, with six contributions: (1) the DY-Skill attacker model, a Dolev-Yao adaptation to the five-phase skill lifecycle with a maximality proof; (2) a sound static analysis framework grounded in abstract interpretation; (3) capability-based sandboxing with a confinement proof; (4) an Agent Dependency Graph with SAT-based resolution and lockfile semantics; (5) a trust score algebra with formal monotonicity; and (6) SkillFortifyBench, a 540-skill benchmark. SkillFortify achieves 96.95% F1 (95% CI: [95.1%, 98.4%]) with 100% precision and 0% false positive rate on 540 skills, while SAT-based resolution handles 1,000-node graphs in under 100 ms. 

---
# SKeDA: A Generative Watermarking Framework for Text-to-video Diffusion Models 

**Authors**: Yang Yang, Xinze Zou, Zehua Ma, Han Fang, Weiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00194)  

**Abstract**: The rise of text-to-video generation models has raised growing concerns over content authenticity, copyright protection, and malicious misuse. Watermarking serves as an effective mechanism for regulating such AI-generated content, where high fidelity and strong robustness are particularly critical. Recent generative image watermarking methods provide a promising foundation by leveraging watermark information and pseudo-random keys to control the initial sampling noise, enabling lossless embedding. However, directly extending these techniques to videos introduces two key limitations: Existing designs implicitly rely on strict alignment between video frames and frame-dependent pseudo-random binary sequences used for watermark encryption. Once this alignment is disrupted, subsequent watermark extraction becomes unreliable; and Video-specific distortions, such as inter-frame compression, significantly degrade watermark reliability. To address these issues, we propose SKeDA, a generative watermarking framework tailored for text-to-video diffusion models. SKeDA consists of two components: (1) Shuffle-Key-based Distribution-preserving Sampling (SKe) employs a single base pseudo-random binary sequence for watermark encryption and derives frame-level encryption sequences through permutation. This design transforms watermark extraction from synchronization-sensitive sequence decoding into permutation-tolerant set-level aggregation, substantially improving robustness against frame reordering and loss; and (2) Differential Attention (DA), which computes inter-frame differences and dynamically adjusts attention weights during extraction, enhancing robustness against temporal distortions. Extensive experiments demonstrate that SKeDA preserves high video generation quality and watermark robustness. 

---
# OSF: On Pre-training and Scaling of Sleep Foundation Models 

**Authors**: Zitao Shuai, Zongzhe Xu, David Yang, Wei Wang, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00190)  

**Abstract**: Polysomnography (PSG) provides the gold standard for sleep assessment but suffers from substantial heterogeneity across recording devices and cohorts. There have been growing efforts to build general-purpose foundation models (FMs) for sleep physiology, but lack an in-depth understanding of the pre-training process and scaling patterns that lead to more generalizable sleep FMs. To fill this gap, we curate a massive corpus of 166,500 hours of sleep recordings from nine public sources and establish SleepBench, a comprehensive, fully open-source benchmark. Leveraging SleepBench, we systematically evaluate four families of self-supervised pre-training objectives and uncover three critical findings: (1) existing FMs fail to generalize to missing channels at inference; (2) channel-invariant feature learning is essential for pre-training; and (3) scaling sample size, model capacity, and multi-source data mixture consistently improves downstream this http URL an enhanced pre-training and scaling recipe, we introduce OSF, a family of sleep FMs that achieves state-of-the-art performance across nine datasets on diverse sleep and disease prediction tasks. Further analysis of OSF also reveals intriguing properties in sample efficiency, hierarchical aggregation, and cross-dataset scaling. 

---
# Efficient Long-Horizon GUI Agents via Training-Free KV Cache Compression 

**Authors**: Bowen Zhou, Zhou Xu, Wanli Li, Jingyu Xiao, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00188)  

**Abstract**: Large Vision-Language Models (VLMs) have emerged as powerful engines for autonomous GUI agents, yet their deployment is severely constrained by the substantial memory footprint and latency of the Key-Value (KV) cache during long-horizon interactions. While existing cache compression methods have proven effective for LLMs, we empirically demonstrate that they suffer from suboptimal performance in GUI scenarios due to a fundamental misalignment: unlike general visual tasks where attention sparsity varies across layers, GUI attention patterns exhibit uniform high-sparsity across all transformer layers. Motivated by this insight, we propose ST-Lite, a training-free KV cache compression framework tailored for efficient GUI agents that explicitly addresses the dynamic spatio-trajectory dependencies within GUI data streams. ST-Lite introduces a novel dual-branch scoring policy incorporating Component-centric Spatial Saliency (CSS) and Trajectory-aware Semantic Gating (TSG). Specifically, CSS preserves the structural integrity of interactive UI elements by evaluating local neighborhood saliency, while TSG mitigates historical redundancy by dynamically filtering visually repetitive KV pairs within the interaction trajectory. Extensive evaluations demonstrate that with only a 10-20% cache budget, ST-Lite achieves a 2.45x decoding acceleration while maintaining comparable or even superior performance compared to full-cache baselines, offering a scalable solution for resource-constrained GUI agents. 

---
# RLShield: Practical Multi-Agent RL for Financial Cyber Defense with Attack-Surface MDPs and Real-Time Response Orchestration 

**Authors**: Srikumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.00186)  

**Abstract**: Financial systems run nonstop and must stay reliable even during cyber incidents. Modern attacks move across many services (apps, APIs, identity, payment rails), so defenders must make a sequence of actions under time pressure. Most security tools still use fixed rules or static playbooks, which can be slow to adapt when the attacker changes behavior. Reinforcement learning (RL) is a good fit for sequential decisions, but much of the RL-in-finance literature targets trading and does not model real cyber response limits such as action cost, service disruption, and defender coordination across many assets. This paper proposes RLShield, a practical multi-agent RL pipeline for financial cyber defense. We model the enterprise attack surface as a Markov decision process (MDP) where states summarize alerts, asset exposure, and service health, and actions represent real response steps (e.g., isolate a host, rotate credentials, ratelimit an API, block an account, or trigger recovery). RLShield learns coordinated policies across multiple agents (assets or service groups) and optimizes a risk-sensitive objective that balances containment speed, business disruption, and response cost. We also include a game-aware evaluation that tests policies against adaptive attackers and reports operational outcomes, not only reward. Experiments show that RLShield reduces time-to-containment and residual exposure while keeping disruption within a fixed response budget, outperforming static rule baselines and single-agent RL under the same constraints. These results suggest that multi-agent, cost-aware RL can provide a deployable layer for automated response in financial security operations. 

---
# ThreatFormer-IDS: Robust Transformer Intrusion Detection with Zero-Day Generalization and Explainable Attribution 

**Authors**: Srikumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.00185)  

**Abstract**: Intrusion detection in IoT and industrial networks requires models that can detect rare attacks at low false-positive rates while remaining reliable under evolving traffic and limited labels. Existing IDS solutions often report strong in-distribution accuracy, but they may degrade when evaluated on future traffic, unseen (zero-day) attack families, or adversarial feature manipulations, and many systems provide limited evidence to support analyst triage. To address these gaps, we propose ThreatFormer- IDS, a Transformer-based sequence modeling framework that converts flow records into time-ordered windows and learns contextual representations for robust intrusion screening. The method combines (i) weighted supervised learning for imbalanced detection, (ii) masked self-supervised learning to improve representation stability under drift and sparse labels, (iii) PGDbased adversarial training with scale-normalized perturbations to strengthen resilience against feature-level evasion, and (iv) Integrated Gradients attribution to highlight influential time steps and features for each alert. On the ToN IoT benchmark with chronological evaluation, ThreatFormer-IDS achieves AUCROC 0.994, AUC-PR 0.956, and Recall@1%FPR 0.910, outperforming strong tree-based and sequence baselines. Under a zero-day protocol with held-out attack families, it maintains superior generalization (AUC-PR 0.721, Recall@1%FPR 0.783). Robustness tests further show slower degradation in AUCPR as the adversarial budget increases, confirming improved stability under bounded perturbations. Overall, ThreatFormer- IDS provides a unified, deployment-oriented IDS pipeline that balances detection quality, zero-day behavior, robustness, and explainability. 

---
# Zero-Shot and Supervised Bird Image Segmentation Using Foundation Models: A Dual-Pipeline Approach with Grounding DINO~1.5, YOLOv11, and SAM~2.1 

**Authors**: Abhinav Munagala  

**Link**: [PDF](https://arxiv.org/pdf/2603.00184)  

**Abstract**: Bird image segmentation remains a challenging task in computer vision due to extreme pose diversity, complex plumage patterns, and variable lighting conditions. This paper presents a dual-pipeline framework for binary bird image segmentation leveraging 2025 foundation models. We introduce two operating modes built upon Segment Anything Model 2.1 (SAM 2.1) as a shared frozen backbone: (1) a zero-shot pipeline using Grounding DINO 1.5 to detect birds via the text prompt "bird" before prompting SAM 2.1 with bounding boxes requiring no labelled bird data; and (2) a supervised pipeline that fine-tunes YOLOv11 on the CUB-200-2011 dataset for high-precision detection, again prompting SAM 2.1 for pixel-level masks. The segmentation model is never retrained for new species or domains. On CUB-200-2011 (11,788 images, 200 species), the supervised pipeline achieves IoU 0.912, Dice 0.954, and F1 0.953 outperforming all prior baselines including SegFormer-B2 (IoU 0.842) by +7.0 percentage points. The zero-shot pipeline achieves IoU 0.831 using only a text prompt, the first such result reported on this benchmark. We demonstrate that prompt-based foundation model pipelines outperform task specific end-to-end trained segmentation networks, while requiring only lightweight detector fine-tuning (~1 hour) for domain adaptation. Complete PyTorch implementation, dataset preparation scripts, and trained weights are publicly available. 

---
# Test Case Prioritization: A Snowballing Literature Review and TCPFramework with Approach Combinators 

**Authors**: Tomasz Chojnacki, Lech Madeyski  

**Link**: [PDF](https://arxiv.org/pdf/2603.00183)  

**Abstract**: Context: Test case prioritization (TCP) is a technique widely used by software development organizations to accelerate regression testing. Objectives: We aim to systematize existing TCP knowledge and to propose and empirically evaluate a new TCP approach. Methods: We conduct a snowballing review (SR) on TCP, implement a~comprehensive platform for TCP research (TCPFramework), analyze existing evaluation metrics and propose two new ones (\rAPFDc{} and ATR), and develop a~family of ensemble TCP methods called approach combinators. Results: The SR helped identify 324 studies related to TCP. The techniques proposed in our study were evaluated on the RTPTorrent dataset, consistently outperforming their base approaches across the majority of subject programs, and achieving performance comparable to the current state of the art for heuristical algorithms (in terms of \rAPFDc{}, NTR, and ATR), while using a distinct approach. Conclusions: The proposed methods can be used efficiently for TCP, reducing the time spent on regression testing by up to 2.7\%. Approach combinators offer significant potential for improvements in future TCP research, due to their composability. 

---
# Embedding Morphology into Transformers for Cross-Robot Policy Learning 

**Authors**: Kei Suzuki, Jing Liu, Ye Wang, Chiori Hori, Matthew Brand, Diego Romeres, Toshiaki Koike-Akino  

**Link**: [PDF](https://arxiv.org/pdf/2603.00182)  

**Abstract**: Cross-robot policy learning -- training a single policy to perform well across multiple embodiments -- remains a central challenge in robot learning. Transformer-based policies, such as vision-language-action (VLA) models, are typically embodiment-agnostic and must infer kinematic structure purely from observations, which can reduce robustness across embodiments and even limit performance within a single embodiment. We propose an embodiment-aware transformer policy that injects morphology via three mechanisms: (1) kinematic tokens that factorize actions across joints and compress time through per-joint temporal chunking; (2) a topology-aware attention bias that encodes kinematic topology as an inductive bias in self-attention, encouraging message passing along kinematic edges; and (3) joint-attribute conditioning that augments topology with per-joint descriptors to capture semantics beyond connectivity. Across a range of embodiments, this structured integration consistently improves performance over a vanilla pi0.5 VLA baseline, indicating improved robustness both within an embodiment and across embodiments. 

---
# Engineering FAIR Privacy-preserving Applications that Learn Histories of Disease 

**Authors**: Ines N. Duarte, Praphulla M. S. Bhawsar, Lee K. Mason, Jeya Balaji Balasubramanian, Daniel E. Russ, Arlindo L. Oliveira, Jonas S. Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2603.00181)  

**Abstract**: A recent report on "Learning the natural history of human disease with generative transformers" created an opportunity to assess the engineering challenge of delivering user-facing Generative AI applications in privacy-sensitive domains. The application of these models, particularly for personalized healthcare tasks like predicting individual morbidity risk, is typically constrained by data privacy concerns. This project was accordingly designed as an in-browser model deployment exercise (an "App") testing the architectural boundaries of client-side inference generation (no downloads or installations). We relied exclusively on the documentation provided in the reference report to develop the model, specifically testing the "R" component of the FAIR data principles: Findability, Accessibility, Interoperability, and Reusability. The successful model deployment, leveraging ONNX and a custom JavaScript SDK, establishes a secure, high-performance architectural blueprint for the future of private generative AI in medicine. 

---
# NNiT: Width-Agnostic Neural Network Generation with Structurally Aligned Weight Spaces 

**Authors**: Jiwoo Kim, Swarajh Mehta, Hao-Lun Hsu, Hyunwoo Ryu, Yudong Liu, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2603.00180)  

**Abstract**: Generative modeling of neural network parameters is often tied to architectures because standard parameter representations rely on known weight-matrix dimensions. Generation is further complicated by permutation symmetries that allow networks to model similar input-output functions while having widely different, unaligned parameterizations. In this work, we introduce Neural Network Diffusion Transformers (NNiTs), which generate weights in a width-agnostic manner by tokenizing weight matrices into patches and modeling them as locally structured fields. We establish that Graph HyperNetworks (GHNs) with a convolutional neural network (CNN) decoder structurally align the weight space, creating the local correlation necessary for patch-based processing. Focusing on MLPs, where permutation symmetry is especially apparent, NNiT generates fully functional networks across a range of architectures. Our approach jointly models discrete architecture tokens and continuous weight patches within a single sequence model. On ManiSkill3 robotics tasks, NNiT achieves >85% success on architecture topologies unseen during training, while baseline approaches fail to generalize. 

---
# Bridging Policy and Real-World Dynamics: LLM-Augmented Rebalancing for Shared Micromobility Systems 

**Authors**: Heng Tan, Hua Yan, Yu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00176)  

**Abstract**: Shared micromobility services such as e-scooters and bikes have become an integral part of urban transportation, yet their efficiency critically depends on effective vehicle rebalancing. Existing methods either optimize for average demand patterns or employ robust optimization and reinforcement learning to handle predefined uncertainties. However, these approaches overlook emergent events (e.g., demand surges, vehicle outages, regulatory interventions) or sacrifice performance in normal conditions. We introduce AMPLIFY, an LLM-augmented policy adaptation framework for shared micromobility rebalancing. The framework combines a baseline rebalancing module with an LLM-based adaptation module that adjusts strategies in real time under emergent scenarios. The adaptation module ingests system context, demand predictions, and baseline strategies, and refines adjustments through self-reflection. Evaluations on real-world e-scooter data from Chicago show that our approach improves demand satisfaction and system revenue compared to baseline policies, highlighting the potential of LLM-driven adaptation as a flexible solution for managing uncertainty in micromobility systems. 

---
# Automated Discovery of Improved Constant Weight Binary Codes 

**Authors**: Christopher D. Rosin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00174)  

**Abstract**: A constant weight binary code consists of $n$-bit binary codewords, each with exactly $w$ bits equal to 1, such that any two codewords are at least Hamming distance $d$ apart. $A(n,d,w)$ is the maximum size of a constant weight binary code with parameters $n,d,w$. We establish improved lower bounds on $A(n,d,w)$ by constructing new larger codes, for 24 values of $(n,d,w)$ with $6 \leq d \leq 18$ and $18 \leq n \leq 35$. The improved lower bounds come from two strategies. The first is a tabu search that operates at the level of bit swaps. The second is a novel greedy heuristic that repeatedly chooses the candidate codeword that maximizes a randomly-scored histogram of distances to previously-added codewords. These strategies were proposed by CPro1, an automated protocol that generates, implements, and tests diverse strategies for combinatorial constructions. 

---
# Summer-22B: A Systematic Approach to Dataset Engineering and Training at Scale for Video Foundation Model 

**Authors**: Simo Ryu, Chunghwan Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.00173)  

**Abstract**: We describe our experience training Summer-22B, a video foundation model developed from scratch. This report documents the engineering challenges, design decisions, and lessons learned while scaling from raw footage collection to a functional model trained on approximately 50 million clips. We outline our approach combining metadata-driven dataset curation, multi-stage filtering, $\mu$P parameterization, and hypersphere-constrained optimization. We developed the Lavender Data system for dataset management and adopted inference-aware architectural choices. We share observations on what worked in our setting: dataset engineering consumed the majority of effort, architectural variants showed smaller differences than we expected, and $\mu$P hyperparameter transfer appeared effective even under geometric constraints. We hope this account proves useful to others undertaking similar projects. 

---
# Hidden in the Metadata: Stealth Poisoning Attacks on Multimodal Retrieval-Augmented Generation 

**Authors**: Kennedy Edemacu, Mohammad Mahdi Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2603.00172)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful paradigm for enhancing multimodal large language models by grounding their responses in external, factual knowledge and thus mitigating hallucinations. However, the integration of externally sourced knowledge bases introduces a critical attack surface. Adversaries can inject malicious multimodal content capable of influencing both retrieval and downstream generation. In this work, we present MM-MEPA, a multimodal poisoning attack that targets the metadata components of image-text entries while leaving the associated visual content unaltered. By only manipulating the metadata, MM-MEPA can still steer multimodal retrieval and induce attacker-desired model responses. We evaluate the attack across multiple benchmark settings and demonstrate its severity. MM-MEPA achieves an attack success rate of up to 91\% consistently disrupting system behaviors across four retrievers and two multimodal generators. Additionally, we assess representative defense strategies and find them largely ineffective against this form of metadata-only poisoning. Our findings expose a critical vulnerability in multimodal RAG and underscore the urgent need for more robust, defense-aware retrieval and knowledge integration methods. 

---
# AdaFocus: Knowing When and Where to Look for Adaptive Visual Reasoning 

**Authors**: Yuxiang Shen, Hailong Huang, Zhenkun Gao, Xueheng Li, Chengjun Xie, Xuanhua He, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00171)  

**Abstract**: Multimodal Large Language Models (MLLMs) are shifting towards "Thinking with Images" by actively exploring image details. While effective, large-scale training is computationally expensive, which has spurred growing interest in lightweight, training-free solutions. However, existing training-free methods suffer from two flaws: perceptual redundancy from indiscriminate cropping, which adds overhead and noise; and a drift between semantic intent and spatial attention, which prevents accurate localization of user-focused regions. To address these challenges, we propose AdaFocus, a novel training-free framework designed for adaptive visual reasoning. AdaFocus follows a two-stage pipeline: a confidence-based module decides when to crop, and a semantic-guided localization module determines where to crop. This enables adaptive visual reasoning without additional training. Experimentally, AdaFocus delivers substantial performance gains while achieving approximately 4.0\times speedup inference speedup than the SOTA method ZoomEyes, representing a significant advance in both accuracy and efficiency. 

---
# A Novel Evolutionary Method for Automated Skull-Face Overlay in Computer-Aided Craniofacial Superimposition 

**Authors**: Práxedes Martínez-Moreno, Andrea Valsecchi, Pablo Mesejo, Pilar Navarro-Ramírez, Valentino Lugli, Sergio Damas  

**Link**: [PDF](https://arxiv.org/pdf/2603.00170)  

**Abstract**: Craniofacial Superimposition is a forensic technique for identifying skeletal remains by comparing a post-mortem skull with ante-mortem facial photographs. A critical step in this process is Skull-Face Overlay (SFO). This stage involves aligning a 3D skull model with a 2D facial image, typically guided by cranial and facial landmarks' correspondence. However, its accuracy is undermined by individual variability in soft-tissue thickness, introducing significant uncertainty into the overlay. This paper introduces Lilium, an automated evolutionary method to enhance the accuracy and robustness of SFO. Lilium explicitly models soft-tissue variability using a 3D cone-based representation whose parameters are optimized via a Differential Evolution algorithm. The method enforces anatomical, morphological, and photographic plausibility through a combination of constraints: landmark matching, camera parameter consistency, head pose alignment, skull containment within facial boundaries, and region parallelism. This emulation of the usual forensic practitioners' approach leads Lilium to outperform the state-of-the-art method in terms of both accuracy and robustness. 

---
# Exploring the AI Obedience: Why is Generating a Pure Color Image Harder than CyberPunk? 

**Authors**: Hongyu Li, Kuan Liu, Yuan Chen, Juntao Hu, Huimin Lu, Guanjie Chen, Xue Liu, Guangming Lu, Hong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00166)  

**Abstract**: Recent advances in generative AI have demonstrated remarkable ability to produce high-quality content. However, these models often exhibit "Paradox of Simplicity": while they can render intricate landscapes, they often fail at simple, deterministic tasks. To address this, we formalize Obedience as the ability to align with instructions and establish a hierarchical grading system ranging from basic semantic alignment to pixel-level systemic precision, which provides a unified paradigm for incorporating and categorizing existing literature. Then, we conduct case studies to identify common obedience gaps, revealing how generative priors often override logical constraints. To evaluate high-level obedience, we present VIOLIN (VIsual Obedience Level-4 EvaluatIoN), the first benchmark focused on pure color generation across six variants. Extensive experiments on SOTA models reveal fundamental obedience limitations and further exploratory insights. By establishing this framework, we aim to draw more attention on AI Obedience and encourage deeper exploration to bridge this gap. 

---
# Reverse CAPTCHA: Evaluating LLM Susceptibility to Invisible Unicode Instruction Injection 

**Authors**: Marcus Graves  

**Link**: [PDF](https://arxiv.org/pdf/2603.00164)  

**Abstract**: We introduce Reverse CAPTCHA, an evaluation framework that tests whether large language models follow invisible Unicode-encoded instructions embedded in otherwise normal-looking text. Unlike traditional CAPTCHAs that distinguish humans from machines, our benchmark exploits a capability gap: models can perceive Unicode control characters that are invisible to human readers. We evaluate five models from two providers across two encoding schemes (zero-width binary and Unicode Tags), four hint levels, two payload framings, and with tool use enabled or disabled. Across 8,308 model outputs, we find that tool use dramatically amplifies compliance (Cohen's h up to 1.37, a large effect), that models exhibit provider-specific encoding preferences (OpenAI models decode zero-width binary; Anthropic models prefer Unicode Tags), and that explicit decoding instructions increase compliance by up to 95 percentage points within a single model and encoding. All pairwise model differences are statistically significant (p < 0.05, Bonferroni-corrected). These results highlight an underexplored attack surface for prompt injection via invisible Unicode payloads. 

---
# DINOv3 Meets YOLO26 for Weed Detection in Vegetable Crops 

**Authors**: Boyang Deng, Yuzhen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00160)  

**Abstract**: Developing robust models for precision vegetable weeding is currently constrained by the scarcity of large-scale, annotated weed-crop datasets. To address this limitation, this study proposes a foundational crop-weed detection model by integrating heterogeneous datasets and leveraging self-supervised learning. A total of 618,642 crop-weed images were initially collected and subsequently refined to 199,388 filtered images for fine-tuning a DINOv3 vision transformer (ViT-small) through a sequential curation strategy. The fine-tuned DINOv3 backbone was then integrated into YOLO26, serving either as a primary backbone or part of a dual-backbone architecture. A feature alignment loss was introduced in the dual backbone framework to enhance feature fusion with minimal computational overhead. Experimental results show that the proposed DINOv3-finetuned ViT-small-based YOLO26-large achieved up to a +5.4% mAP50 gain on in-domain images collected in the 2025 season. Moreover, it demonstrated strong cross-domain generalization with mAP50 improvements of +14.0% on the 2021-2023 season dataset and +11.9% on the 2024 season dataset, compared to the standard YOLO26-large. Although the DINOv3-YOLO26-large model has 45.6% more parameters and a 2.9x increase in inference latency, it maintains real-time performance at ~28.5 frames per second (fps). The curated dataset and software programs developed in this study will be made publicly available. 

---
# FlowPortrait: Reinforcement Learning for Audio-Driven Portrait Video Generation 

**Authors**: Weiting Tan, Andy T. Liu, Ming Tu, Xinghua Qu, Philipp Koehn, Lu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00159)  

**Abstract**: Generating realistic talking-head videos remains challenging due to persistent issues such as imperfect lip synchronization, unnatural motion, and evaluation metrics that correlate poorly with human perception. We propose FlowPortrait, a reinforcement-learning framework for audio-driven portrait animation built on a multimodal backbone for autoregressive audio-to-video generation. FlowPortrait introduces a human-aligned evaluation system based on Multimodal Large Language Models (MLLMs) to assess lip-sync accuracy, expressiveness, and motion quality. These signals are combined with perceptual and temporal consistency regularizers to form a stable composite reward, which is used to post-train the generator via Group Relative Policy Optimization (GRPO). Extensive experiments, including both automatic evaluations and human preference studies, demonstrate that FlowPortrait consistently produces higher-quality talking-head videos, highlighting the effectiveness of reinforcement learning for portrait animation. 

---
# EfficientPosterGen: Semantic-aware Efficient Poster Generation via Token Compression and Accurate Violation Detection 

**Authors**: Wenxin Tang, Jingyu Xiao, Yanpei Gong, Fengyuan Ran, Tongchuan Xia, Junliang Liu, Man Ho Lam, Wenxuan Wang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00155)  

**Abstract**: Automated academic poster generation aims to distill lengthy research papers into concise, visually coherent presentations. Existing Multimodal Large Language Models (MLLMs) based approaches, however, suffer from three critical limitations: low information density in full-paper inputs, excessive token consumption, and unreliable layout verification. We present EfficientPosterGen, an end-to-end framework that addresses these challenges through semantic-aware retrieval and token-efficient multimodal generation. EfficientPosterGen introduces three core innovations: (1) Semantic-aware Key Information Retrieval (SKIR), which constructs a semantic contribution graph to model inter-segment relationships and selectively preserves important content; (2) Visual-based Context Compression (VCC), which renders selected text segments into images to shift textual information into the visual modality, significantly reducing token usage while generating poster-ready bullet points; and (3) Agentless Layout Violation Detection (ALVD), a deterministic color-gradient-based algorithm that reliably detects content overflow and spatial sparsity without auxiliary MLLMs. Extensive experiments demonstrate that EfficientPosterGen achieves substantial improvements in token efficiency and layout reliability while maintaining high poster quality, offering a scalable solution for automated academic poster generation. Our code is available at this https URL. 

---
# Pulse-Driven Neural Architecture: Learnable Oscillatory Dynamics for Robust Continuous-Time Sequence Processing 

**Authors**: Paras Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2603.00153)  

**Abstract**: We introduce PDNA (Pulse-Driven Neural Architecture), a method for augmenting continuous-time recurrent networks with learnable oscillatory dynamics that maintain internal state evolution independently of external input. Built on Closed-form Continuous-time (CfC) networks, PDNA adds two components: (1) a pulse module that generates structured oscillations $A \cdot \sin(\omega t + \varphi(h))$ with learnable frequencies and state-dependent phase, and (2) a self-attend module that applies recurrent self-attention to the hidden state. Through a controlled ablation study on sequential MNIST (sMNIST) with five random seeds, we evaluate gap robustness -- the ability to maintain performance when portions of the input sequence are removed at test time. Our key finding is that structured oscillatory dynamics significantly improve robustness to input interruptions: the self-attend variant achieves a statistically significant 2.78 percentage point multi-gap advantage over baseline ($p = 0.041$), while the pulse variant shows a 4.62 pp advantage with large effect size (Cohen's $d = 0.87$). A noise control (random perturbation of equal magnitude) provides no benefit, confirming that the advantage is structural rather than merely dynamic. These results provide evidence that continuous-time models can benefit from biologically-inspired internal oscillatory mechanisms for temporal robustness. 

---
# Dr. Seg: Revisiting GRPO Training for Visual Large Language Models through Perception-Oriented Design 

**Authors**: Haoxiang Sun, Tao Wang, Chenwei Tang, Li Yuan, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2603.00152)  

**Abstract**: Following the success of Group Relative Policy Optimization (GRPO) in foundation LLMs, an increasing number of works have sought to adapt GRPO to Visual Large Language Models (VLLMs) for visual perception tasks (e.g., detection and segmentation). However, much of this line of research rests on a long-standing yet unexamined assumption: training paradigms developed for language reasoning can be transferred seamlessly to visual perception. Our experiments show that this assumption is not valid, revealing intrinsic differences between reasoning-oriented and perception-oriented settings. Using reasoning segmentation as a representative case, we surface two overlooked factors: (i) the need for a broader output space, and (ii) the importance of fine-grained, stable rewards. Building on these observations, we propose Dr.~Seg, a simple, plug-and-play GRPO-based framework consisting of a Look-to-Confirm mechanism and a Distribution-Ranked Reward module, requiring no architectural modifications and integrating seamlessly with existing GRPO-based VLLMs. Extensive experiments demonstrate that Dr.~Seg improves performance in complex visual scenarios while maintaining strong generalization. Code and models will be available at this https URL. 

---
# Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction 

**Authors**: Zhihao Li, Shengwei Dong, Chuang Yi, Junxuan Gao, Zhilu Lai, Zhiqiang Liu, Wei Wang, Guangtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00149)  

**Abstract**: Existing image SR and generic diffusion models transfer poorly to fluid SR: they are sampling-intensive, ignore physical constraints, and often yield spectral mismatch and spurious divergence. We address fluid super-resolution (SR) with \textbf{ReMD} (\underline{Re}sidual-\underline{M}ultigrid \underline{D}iffusion), a physics-consistent diffusion framework. At each reverse step, ReMD performs a \emph{multigrid residual correction}: the update direction is obtained by coupling data consistency with lightweight physics cues and then correcting the residual across scales; the multiscale hierarchy is instantiated with a \emph{multi-wavelet} basis to capture both large structures and fine vortical details. This coarse-to-fine design accelerates convergence and preserves fine structures while remaining equation-free. Across atmospheric and oceanic benchmarks, ReMD improves accuracy and spectral fidelity, reduces divergence, and reaches comparable quality with markedly fewer sampling steps than diffusion baselines. Our results show that enforcing physics consistency \emph{inside} the diffusion process via multigrid residual correction and multi-wavelet multiscale modeling is an effective route to efficient fluid SR. Our code are available on this https URL. 

---
# M-Gaussian: An Magnetic Gaussian Framework for Efficient Multi-Stack MRI Reconstruction 

**Authors**: Kangyuan Zheng, Xuan Cai, Jiangqi Wang, Guixing Fu, Zhuoshuo Li, Yazhou Chen, Xinting Ge, Liangqiong Qu, Mengting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00145)  

**Abstract**: Magnetic Resonance Imaging (MRI) is a crucial non-invasive imaging modality. In routine clinical practice, multi-stack thick-slice acquisitions are widely used to reduce scan time and motion sensitivity, particularly in challenging scenarios such as fetal brain imaging. However, the resulting severe through-plane anisotropy compromises volumetric analysis and downstream quantitative assessment, necessitating robust reconstruction of isotropic high-resolution volumes. Implicit neural representation methods, while achieving high quality, suffer from computational inefficiency due to complex network structures. We present M-Gaussian, adapting 3D Gaussian Splatting to MRI reconstruction. Our contributions include: (1) Magnetic Gaussian primitives with physics-consistent volumetric rendering, (2) neural residual field for high-frequency detail refinement, and (3) multi-resolution progressive training. Our method achieves an optimal balance between quality and speed. On the FeTA dataset, M-Gaussian achieves 40.31 dB PSNR while being 14 times faster, representing the first successful adaptation of 3D Gaussian Splatting to multi-stack MRI reconstruction. 

---
# Disentangled Hierarchical VAE for 3D Human-Human Interaction Generation 

**Authors**: Zichen Geng, Zeeshan Hayder, Bo Miao, Jian Liu, Wei Liu, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2603.00144)  

**Abstract**: Generating realistic 3D Human-Human Interaction (HHI) requires coherent modeling of the physical plausibility of the agents and their interaction semantics. Existing methods compress all motion information into a single latent representation, limiting their ability to capture fine-grained actions and inter-agent interactions. This often leads to semantic misalignment and physically implausible artifacts, such as penetration or missed contact. We propose Disentangled Hierarchical Variational Autoencoder (DHVAE) based latent diffusion for structured and controllable HHI generation. DHVAE explicitly disentangles the global interaction context and individual motion patterns into a decoupled latent structure by employing a CoTransformer module. To mitigate implausible and physically inconsistent contacts in HHI, we incorporate contrastive learning constraints with our DHVAE to promote a more discriminative and physically plausible latent interaction space. For high-fidelity interaction synthesis, DHVAE employs a DDIM-based diffusion denoising process in the hierarchical latent space, enhanced by a skip-connected AdaLN-Transformer denoiser. Extensive evaluations show that DHVAE achieves superior motion fidelity, text alignment, and physical plausibility with greater computational efficiency. 

---
# Evaluating Theory of Mind and Internal Beliefs in LLM-Based Multi-Agent Systems 

**Authors**: Adam Kostka, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2603.00142)  

**Abstract**: LLM-based MAS are gaining popularity due to their potential for collaborative problem-solving enhanced by advances in natural language comprehension, reasoning, and planning. Research in Theory of Mind (ToM) and Belief-Desire-Intention (BDI) models has the potential to further improve the agent's interaction and decision-making in such systems. However, collaborative intelligence in dynamic worlds remains difficult to accomplish since LLM performance in multi-agent worlds is extremely variable. Simply adding cognitive mechanisms like ToM and internal beliefs does not automatically result in improved coordination. The interplay between these mechanisms, particularly in relation to formal logic verification, remains largely underexplored in different LLMs. This work investigates: How do internal belief mechanisms, including symbolic solvers and Theory of Mind, influence collaborative decision-making in LLM-based multi-agent systems, and how does the interplay of those components influence system accuracy? We introduce a novel multi-agent architecture integrating ToM, BDI-style internal beliefs, and symbolic solvers for logical verification. We evaluate this architecture in a resource allocation problem with various LLMs and find an intricate interaction between LLM capabilities, cognitive mechanisms, and performance. This work contributes to the area of AI by proposing a novel multi-agent system with ToM, internal beliefs, and symbolic solvers for augmenting collaborative intelligence in multi-agent systems and evaluating its performance under different LLM settings. 

---
# From Scale to Speed: Adaptive Test-Time Scaling for Image Editing 

**Authors**: Xiangyan Qu, Zhenlong Yuan, Jing Tang, Rui Chen, Datao Tang, Meng Yu, Lei Sun, Yancheng Bai, Xiangxiang Chu, Gaopeng Gou, Gang Xiong, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.00141)  

**Abstract**: Image Chain-of-Thought (Image-CoT) is a test-time scaling paradigm that improves image generation by extending inference time. Most Image-CoT methods focus on text-to-image (T2I) generation. Unlike T2I generation, image editing is goal-directed: the solution space is constrained by the source image and instruction. This mismatch causes three challenges when applying Image-CoT to editing: inefficient resource allocation with fixed sampling budgets, unreliable early-stage verification using general MLLM scores, and redundant edited results from large-scale sampling. To address this, we propose ADaptive Edit-CoT (ADE-CoT), an on-demand test-time scaling framework to enhance editing efficiency and performance. It incorporates three key strategies: (1) a difficulty-aware resource allocation that assigns dynamic budgets based on estimated edit difficulty; (2) edit-specific verification in early pruning that uses region localization and caption consistency to select promising candidates; and (3) depth-first opportunistic stopping, guided by an instance-specific verifier, that terminates when intent-aligned results are found. Extensive experiments on three SOTA editing models (Step1X-Edit, BAGEL, FLUX.1 Kontext) across three benchmarks show that ADE-CoT achieves superior performance-efficiency trade-offs. With comparable sampling budgets, ADE-CoT obtains better performance with more than 2x speedup over Best-of-N. 

---
# Steering Away from Memorization: Reachability-Constrained Reinforcement Learning for Text-to-Image Diffusion 

**Authors**: Sathwik Karnik, Juyeop Kim, Sanmi Koyejo, Jong-Seok Lee, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2603.00140)  

**Abstract**: Text-to-image diffusion models often memorize training data, revealing a fundamental failure to generalize beyond the training set. Current mitigation strategies typically sacrifice image quality or prompt alignment to reduce memorization. To address this, we propose Reachability-Aware Diffusion Steering (RADS), an inference-time framework that prevents memorization while preserving generation fidelity. RADS models the diffusion denoising process as a dynamical system and applies concepts from reachability analysis to approximate the "backward reachable tube"--the set of intermediate states that inevitably evolve into memorized samples. We then formulate mitigation as a constrained reinforcement learning (RL) problem, where a policy learns to steer the trajectory away from memorization via minimal perturbations in the caption embedding space. Empirical evaluations show that RADS achieves a superior Pareto frontier between generation diversity (SSCD), quality (FID), and alignment (CLIP) compared to state-of-the-art baselines. Crucially, RADS provides robust mitigation without modifying the diffusion backbone, offering a plug-and-play solution for safe generation. Our website is available at: this https URL. 

---
# MAML-KT: Addressing Cold Start Problem in Knowledge Tracing for New Students via Few-Shot Model-Agnostic Meta Learning 

**Authors**: Indronil Bhattacharjee, Christabel Wayllace  

**Link**: [PDF](https://arxiv.org/pdf/2603.00137)  

**Abstract**: Knowledge tracing (KT) models are commonly evaluated by training on early interactions from all students and testing on later responses. While effective for measuring average predictive performance, this evaluation design obscures a cold start scenario that arises in deployment, where models must infer the knowledge state of previously unseen students from only a few initial interactions. Prior studies have shown that under this setting, standard empirically risk-minimized KT models such as DKT, DKVMN and SAKT exhibit substantially lower early accuracy than previously reported. We frame new-student performance prediction as a few-shot learning problem and introduce MAML-KT, a model-agnostic meta learning approach that learns an initialization optimized for rapid adaptation to new students using one or two gradient updates. We evaluate MAML-KT on ASSIST2009, ASSIST2015 and ASSIST2017 using a controlled cold start protocol that trains on a subset of students and tests on held-out learners across early interaction windows (questions 3-10 and 11-15), scaling cohort sizes from 10 to 50 students. Across datasets, MAML-KT achieves higher early accuracy than prior KT models in nearly all cold start conditions, with gains persisting as cohort size increases. On ASSIST2017, we observe a transient drop in early performance that coincides with many students encountering previously unseen skills. Further analysis suggests that these drops coincide with skill novelty rather than model instability, consistent with prior work on skill-level cold start. Overall, optimizing KT models for rapid adaptation reduces early prediction error for new students and provides a clearer lens for interpreting early accuracy fluctuations, distinguishing model limitations from genuine learning and knowledge acquisition dynamics. 

---
# TinyVLM: Zero-Shot Object Detection on Microcontrollers via Vision-Language Distillation with Matryoshka Embeddings 

**Authors**: Bibin Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2603.00136)  

**Abstract**: Zero-shot object detection enables recognising novel objects without task-specific training, but current approaches rely on large vision language models (VLMs) like CLIP that require hundreds of megabytes of memory - far exceeding the constraints of micro controller units (MCUs). We present TinyVLM, the first framework enabling zero-shot object detection on resource-constrained MCUs with less than 1MB of memory. Our approach introduces three key innovations: (1) a decoupled architecture that separates visual inference from text encoding, allowing precomputed class embeddings to be stored in flash memory; (2) Matryoshka distillation that trains nested embeddings at multiple dimensions (16-256), enabling flexible accuracy-memory trade-offs; and (3) quantized embedding storage that reduces class prototype memory by 4x with minimal accuracy loss. Trained on Conceptual Captions 3M (CC3M), TinyVLM achieves competitive zero-shot accuracy on COCO, Flowers102, and Food101 while requiring only 285KB of RAM and 892KB of flash memory for the deployed vision encoder. We demonstrate real-time inference at 26 FPS on STM32H7 and over 1,000 FPS on MAX78000 with its CNN accelerator, enabling practical zero-shot detection on edge devices for the first time. 

---
# You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models 

**Authors**: Kairan Zhao, Eleni Triantafillou, Peter Triantafillou  

**Link**: [PDF](https://arxiv.org/pdf/2603.00133)  

**Abstract**: Generative models have been shown to "memorize" certain training data, leading to verbatim or near-verbatim generating images, which may cause privacy concerns or copyright infringement. We introduce Guidance Using Attractive-Repulsive Dynamics (GUARD), a novel framework for memorization mitigation in text-to-image diffusion models. GUARD adjusts the image denoising process to guide the generation away from an original training image and towards one that is distinct from training data while remaining aligned with the prompt, guarding against reproducing training data, without hurting image generation quality. We propose a concrete instantiation of this framework, where the positive target that we steer towards is given by a novel method for (cross) attention attenuation based on (i) a novel statistical mechanism that automatically identifies the prompt positions where cross attention must be attenuated and (ii) attenuating cross-attention in these per-prompt locations. The resulting GUARD offers a surgical, dynamic per-prompt inference-time approach that, we find, is by far the most robust method in terms of consistently producing state-of-the-art results for memorization mitigation across two architectures and for both verbatim and template memorization, while also improving upon or yielding comparable results in terms of image quality. 

---
# Thought Virus: Viral Misalignment via Subliminal Prompting in Multi-Agent Systems 

**Authors**: Moritz Weckbecker, Jonas Müller, Ben Hagag, Michael Mulet  

**Link**: [PDF](https://arxiv.org/pdf/2603.00131)  

**Abstract**: Subliminal prompting is a phenomenon in which language models are biased towards certain concepts or traits through prompting with semantically unrelated tokens. While prior work has examined subliminal prompting in user-LLM interactions, potential bias transfer in multi-agent systems and its associated security implications remain unexplored. In this work, we show that a single subliminally prompted agent can spread a weakening but persisting bias throughout its entire network. We measure this phenomenon across 6 agents using two different topologies, observing that the transferred concept maintains an elevated response rate throughout the network. To exemplify potential misalignment risks, we assess network performance on multiple-choice TruthfulQA, showing that subliminal prompting of a single agent may degrade the truthfulness of other agents. Our findings reveal that subliminal prompting introduces a new attack vector in multi-agent security, with implications for the alignment of such systems. The implementation of all experiments is publicly available at this https URL . 

---
# Agentic Hives: Equilibrium, Indeterminacy, and Endogenous Cycles in Self-Organizing Multi-Agent Systems 

**Authors**: Jean-Philippe Garnier  

**Link**: [PDF](https://arxiv.org/pdf/2603.00130)  

**Abstract**: Current multi-agent AI systems operate with a fixed number of agents whose roles are specified at design time. No formal theory governs when agents should be created, destroyed, or re-specialized at runtime-let alone how the population structure responds to changes in resources or objectives.  We introduce the Agentic Hive, a framework in which a variable population of autonomous micro-agents-each equipped with a sandboxed execution environment and access to a language model-undergoes demographic dynamics: birth, duplication, specialization, and death. Agent families play the role of production sectors, compute and memory play the role of factors of production, and an orchestrator plays the dual role of Walrasian auctioneer and Global Workspace.  Drawing on the multi-sector growth theory developed for dynamic general equilibrium (Benhabib \& Nishimura, 1985; Venditti, 2005; Garnier, Nishimura \& Venditti, 2013), we prove seven analytical results: (i) existence of a Hive Equilibrium via Brouwer's fixed-point theorem; (ii) Pareto optimality of the equilibrium allocation; (iii) multiplicity of equilibria under strategic complementarities between agent families; (iv)-(v) Stolper-Samuelson and Rybczynski analogs that predict how the Hive restructures in response to preference and resource shocks; (vi) Hopf bifurcation generating endogenous demographic cycles; and (vii) a sufficient condition for local asymptotic stability.  The resulting regime diagram partitions the parameter space into regions of unique equilibrium, indeterminacy, endogenous cycles, and instability. Together with the comparative-statics matrices, it provides a formal governance toolkit that enables operators to predict and steer the demographic evolution of self-organizing multi-agent systems. 

---
# QuickGrasp: Responsive Video-Language Querying Service via Accelerated Tokenization and Edge-Augmented Inference 

**Authors**: Miao Zhang, Ruixiao Zhang, Jianxin Shi, Hengzhi Wang, Hao Fang, Jiangchuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00126)  

**Abstract**: Video-language models (VLMs) are reshaping video querying services, bringing unified solutions to complex perception and reasoning tasks. However, deploying large VLMs in real-world systems remains challenging due to their high resource demands, and remote-based deployment often results in unacceptable response delays. Although small, locally deployable VLMs offer faster responses, they unavoidably fall short in accuracy. To reconcile this trade-off, we propose QuickGrasp, a responsive, quality of service (QoS)-aware system that bridges this gap through a local-first architecture with on-demand edge augmentation. Built upon the highly modular architecture of VLMs, QuickGrasp shares the vision representation across model variants to avoid redundant computation. To maximize system-wide efficiency, QuickGrasp introduces three key designs: accelerated video tokenization, query-adaptive edge augmentation, and delay-aware, accuracy-preserving vision token density configuration. We implement a prototype of QuickGrasp and evaluate it across multiple video understanding benchmarks. The results show that QuickGrasp matches the accuracy of large VLMs while achieving up to a 12.8x reduction in response delay. QuickGrasp represents a key advancement toward building responsive video querying services for open-world understanding that fully leverage the capabilities of VLMs. 

---
# OrthoAI: A Lightweight Deep Learning Framework for Automated Biomechanical Analysis in Clear Aligner Orthodontics -- A Methodological Proof-of-Concept 

**Authors**: Edouard Lansiaux, Margaux Leman, Mehdi Ammi  

**Link**: [PDF](https://arxiv.org/pdf/2603.00124)  

**Abstract**: Clear aligner therapy now dominates orthodontics, yet clinician review of digitally planned tooth movements-typically via ClinCheck (Align Technology)-remains slow and error-prone. We present OrthoAI, an open-source proof-of-concept decision-support system combining lightweight 3D dental segmentation with automated biomechanical analysis to assist treatment-plan evaluation. The framework uses a Dynamic Graph CNN trained on landmark-reconstructed point clouds from 3DTeethLand (MICCAI) and integrates a rule-based biomechanical engine grounded in orthodontic evidence (Kravitz et al 2009; Simon et al 2014). The system decomposes per-tooth motion across six degrees of freedom, computes movement-specific predictability, issues alerts when biomechanical limits are exceeded, and derives an exploratory composite index. With 60,705 trainable parameters, segmentation reaches a Tooth Identification Rate of $81.4\%$ and mIoU of $8.25\%$ on surrogate point clouds-reflecting sparse landmark supervision rather than dense meshes. Although spatial boundaries are coarse, downstream analysis depends mainly on tooth identity and approximate centroid/axis estimation. Results establish a baseline for future full-mesh training and highlight current perceptual limits. The end-to-end pipeline runs in $<4s$ on consumer hardware. Code, weights, and analysis tools are released to support reproducible research in geometric deep learning and digital orthodontics. The system has not been validated on real intraoral meshes and should not be assumed to generalize beyond landmark-derived representations. 

---
# CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers 

**Authors**: Yannian Gu, Xizhuo Zhang, Linjie Mu, Yongrui Yu, Zhongzhen Huang, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00123)  

**Abstract**: Recent advances in Large Vision-Language Models (LVLMs) have shown strong potential for multi-modal radiological reasoning, particularly in tasks like diagnostic visual question answering (VQA) and radiology report generation. However, most existing approaches for 3D CT analysis largely rely on static, single-pass inference. In practice, clinical interpretation is a dynamic, tool-mediated workflow where radiologists iteratively review slices and use measurement, radiomics, and segmentation tools to refine findings. To bridge this gap, we propose CT-Flow, an agentic framework designed for interoperable volumetric interpretation. By leveraging the Model Context Protocol (MCP), CT-Flow shifts from closed-box inference to an open, tool-aware paradigm. We curate CT-FlowBench, the first large-scale instruction-tuning benchmark tailored for 3D CT tool-use and multi-step reasoning. Built upon this, CT-Flow functions as a clinical orchestrator capable of decomposing complex natural language queries into automated tool-use sequences. Experimental evaluations on CT-FlowBench and standard 3D VQA datasets demonstrate that CT-Flow achieves state-of-the-art performance, surpassing baseline models by 41% in diagnostic accuracy and achieving a 95% success rate in autonomous tool invocation. This work provides a scalable foundation for integrating autonomous, agentic intelligence into real-world clinical radiology. 

---
# NovaLAD: A Fast, CPU-Optimized Document Extraction Pipeline for Generative AI and Data Intelligence 

**Authors**: Aman Ulla  

**Link**: [PDF](https://arxiv.org/pdf/2603.00122)  

**Abstract**: Document extraction is an important step before retrieval-augmented generation (RAG), knowledge bases, and downstream generative AI can work. It turns unstructured documents like PDFs and scans into structured text and layout-aware representations. We introduce NovaLAD, a comprehensive document parsing system that integrates two concurrent YOLO object detection models - element detection and layout detection - with rule-based grouping and optional vision-language enhancement. When a page image is sent in, the first thing that happens is that it goes through both models at the same time. The element model finds semantic content like the title, header, text, table, image, and so on, and the layout model finds structural regions like layout_box, column_group, multi_column, row_group, and so on. A key design decision is to first send an image or figure through an image classifier (ViT) that decides whether it is relevant or not. Only useful images are then submitted to the Vision LLM for title, summary, and structured information, which cuts down on noise and costs. NovaLAD is built for speed: it works on CPU, employs parallel execution for detection, classification, OCR, and conversion, and generates several forms, including structured JSON, Markdown, RAG-ready texts, and knowledge graphs. We test on the DP-Bench benchmark (upstage/dp-bench) and get 96.49% TEDS and 98.51% NID, which is better than both commercial and open-source parsers. This paper explains how to extract data, how the architecture works, how data flows, and how to make NovaLAD both accurate and usable without needing a GPU. 

---
# Graph-theoretic Agreement Framework for Multi-agent LLM Systems 

**Authors**: Muhammad Umar Javed  

**Link**: [PDF](https://arxiv.org/pdf/2603.00121)  

**Abstract**: The shift from monolithic LLMs to distributed multi-agent architectures demands new frameworks for verifying and securing autonomous coordination. Unlike traditional multi-agent systems focused on cooperative state alignment, modern LLM patterns: multi-agent debate, constitutional oversight, helper-critic loops-rely on adversarial critique for error correction and reasoning refinement. Since LLMs are dynamical systems whose latent states are imperfectly observable from verbalized outputs, securing these networks requires understanding both macroscopic topology and microscopic agent observability. This paper establishes a rigorous graph-theoretic framework for analyzing consensus in signed, directed interaction networks, bridging graph theory and LLM reasoning by formally mapping Transformer cross-entropy log-odds to the signed Laplacian. We characterize agreement stability through structural balance theory, showing how unbalanced critique cycles produce logical frustration and persistent reasoning oscillations, and prove that unobservable latent states from hidden system prompts act as topological Trojan horses that destabilize cooperative consensus. To resolve unobservable deadlocks, we restrict interaction topologies to chordal graphs and apply matrix decomposition with Gram-Schmidt orthogonalization, proving that rank-one spectral edge perturbations deterministically break expertise symmetry by shifting eigenvalues into the stable left-half plane. Core contributions include consensus theorems, polynomial-time Perfect Elimination Ordering verification algorithms, and large-scale empirical validation on clustered ensembles of LLaMA-3, Mistral, and Gemma agents. 

---
# SIGMAS: Second-Order Interaction-based Grouping for Overlapping Multi-Agent Swarms 

**Authors**: Minah Lee, Saibal Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2603.00120)  

**Abstract**: Swarming systems, such as drone fleets and robotic teams, exhibit complex dynamics driven by both individual behaviors and emergent group-level interactions. Unlike traditional multi-agent domains such as pedestrian crowds or traffic systems, swarms typically consist of a few large groups with inherent and persistent memberships, making group identification essential for understanding fine-grained behavior. We introduce the novel task of group prediction in overlapping multi-agent swarms, where latent group structures must be inferred directly from agent trajectories without ground-truth supervision. To address this challenge, we propose SIGMAS (Second-order Interaction-based Grouping for Multi-Agent Swarms), a self-supervised framework that goes beyond direct pairwise interactions and model second-order interaction across agents. By capturing how similarly agents interact with others, SIGMAS enables robust group inference and adaptively balances individual and collective dynamics through a learnable gating mechanism for joint reasoning. Experiments across diverse synthetic swarm scenarios demonstrate that SIGMAS accurately recovers latent group structures and remains robust under simultaneously overlapping swarm dynamics, establishing both a new benchmark task and a principled modeling framework for swarm understanding. 

---
# PEPA: a Persistently Autonomous Embodied Agent with Personalities 

**Authors**: Kaige Liu, Yang Li, Lijun Zhu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00117)  

**Abstract**: Living organisms exhibit persistent autonomy through internally generated goals and self-sustaining behavioral organization, yet current embodied agents remain driven by externally scripted objectives. This dependence on predefined task specifications limits their capacity for long-term deployment in dynamic, unstructured environments where continuous human intervention is impractical. We propose that personality traits provide an intrinsic organizational principle for achieving persistent autonomy. Analogous to genotypic biases shaping biological behavioral tendencies, personalities enable agents to autonomously generate goals and sustain behavioral evolution without external supervision. To realize this, we develop PEPA, a three-layer cognitive architecture that operates through three interacting systems: Sys3 autonomously synthesizes personality-aligned goals and refines them via episodic memory and daily self-reflection; Sys2 performs deliberative reasoning to translate goals into executable action plans; Sys1 grounds the agent in sensorimotor interaction, executing actions and recording experiences. We validate the framework through real-world deployment on a quadruped robot in a multi-floor office building. Operating without reliance on fixed task specifications, the robot autonomously arbitrates between user requests and personality-driven motivations, navigating elevators and exploring environments accordingly. Quantitative analysis across five distinct personality prototypes demonstrates stable, trait-aligned behaviors. The results confirm that personality-driven cognitive architectures enable sustained autonomous operation characteristic of persistent embodied systems. Code and demo videos are available at this https URL. 

---
# Multimodal Modular Chain of Thoughts in Energy Performance Certificate Assessment 

**Authors**: Zhen Peng, Peter J. Bentley  

**Link**: [PDF](https://arxiv.org/pdf/2603.00115)  

**Abstract**: Accurate evaluation of building energy performance remains challenging in regions where scalable Energy Performance Certificate (EPC) assessments are unavailable. This paper presents a cost-efficient framework that leverages Vision-Language models for automated EPC pre-assessment from limited visual information. The proposed Multimodal Modular Chain of Thoughts (MMCoT) architecture decomposes EPC estimation into intermediate reasoning stages and explicitly propagates inferred attributes across tasks using structured prompting. Experiments on a multimodal dataset of 81 residential properties in the United Kingdom show that MMCoT achieves statistically significant improvements over instruction-only prompting for EPC estimation. Analysis based on accuracy, recall, mean absolute error, and confusion matrices indicate that the proposed approach captures the ordinal structure of EPC ratings, with most errors occurring between adjacent classes. These results suggest that modular prompt-based reasoning offers a promising direction for low-cost EPC pre-assessment in data-scarce settings. 

---
# Position: AI Agents Are Not (Yet) a Panacea for Social Simulation 

**Authors**: Yiming Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00113)  

**Abstract**: Recent advances in large language models (LLMs) have spurred growing interest in using LLM-integrated agents for social simulation, often under the implicit assumption that realistic population dynamics will emerge once role-specified agents are placed in a networked multi-agent setting. This position paper argues that LLM-based agents are not (yet) a panacea for social simulation. We attribute this over-optimism to a systematic mismatch between what current agent pipelines are typically optimized and validated to produce and what simulation-as-science requires. Concretely, role-playing plausibility does not imply faithful human behavioral validity; collective outcomes are frequently mediated by agent-environment co-dynamics rather than agent-agent messaging alone; and results can be dominated by interaction protocols, scheduling, and initial information priors, especially in policy-oriented settings. To make these assumptions explicit and auditable, we propose a unified formulation of AI agent-based social simulation as an environment-involved partially observable Markov game with explicit exposure and scheduling mechanisms and call for further actions. 

---
# SurgFusion-Net: Diversified Adaptive Multimodal Fusion Network for Surgical Skill Assessment 

**Authors**: Runlong He, Freweini M. Tesfai, Matthew W. E. Boal, Nazir Sirajudeen, Dimitrios Anastasiou, Jialang Xu, Mobarak I. Hoque, Philip J. Edwards, John D. Kelly, Ashwin Sridhar, Abdolrahim Kadkhodamohammadi, Dhivya Chandrasekaran, Matthew J. Clarkson, Danail Stoyanov, Nader Francis, Evangelos B. Mazomenos  

**Link**: [PDF](https://arxiv.org/pdf/2603.00108)  

**Abstract**: Robotic-assisted surgery (RAS) is established in clinical practice, and automated surgical skill assessment utilizing multimodal data offers transformative potential for surgical analytics and education. However, developing effective multimodal methods remains challenging due to the task complexity, limited annotated datasets and insufficient techniques for cross-modal information fusion. Existing state-of-the-art relies exclusively on RGB video and only applies on dry-lab settings, failing to address the significant domain gap between controlled simulation and real clinical cases, where the surgical environment together with camera and tissue motion introduce substantial complexities. This work introduces SurgFusion-Net and Divergence Regulated Attention (DRA), an innovative fusion strategy for multimodal surgical skill assessment. We contribute two first-of-their-kind clinical datasets: the RAH-skill dataset containing 279,691 RGB frames from 37 videos of Robot-assisted Hysterectomy (RAH), and the RARP-skill dataset containing 70,661 RGB frames from 33 videos of Robot-Assisted Radical Prostatectomy (RARP). Both datasets include M-GEARS skill annotations, corresponding optical flow and tool segmentation masks. DRA incorporates adaptive dual attention and diversity-promoting multi-head attention to fuse multimodal information, from three modalities, based on surgical context, enhancing assessment accuracy and reliability. Validated on the JIGSAWS benchmark, RAH-skill, and RARP-skill datasets, our approach outperforms recent baselines with SCC improvements of 0.02 in LOSO, 0.04 in LOUO across JIGSAWS tasks, and 0.0538 and 0.0493 gains on RAH-skill and RARP-skill, respectively. 

---
# SEval-NAS: A Search-Agnostic Evaluation for Neural Architecture Search 

**Authors**: Atah Nuh Mih, Jianzhou Wang, Truong Thanh Hung Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00099)  

**Abstract**: Neural architecture search (NAS) automates the discovery of neural networks that meet specified criteria, yet its evaluation procedures are often hardcoded, limiting the ability to introduce new metrics. This issue is especially pronounced in hardware-aware NAS, where objectives depend on target devices such as edge hardware. To address this limitation, we propose SEval-NAS, a metric-evaluation mechanism that converts architectures to strings, embeds them as vectors, and predicts performance metrics. Using NATS-Bench and HW-NAS-Bench, we evaluated accuracy, latency, and memory. Kendall's $\tau$ correlations showed stronger latency and memory predictions than accuracy, indicating the suitability of SEval-NAS as a hardware cost predictor. We further integrated SEval-NAS into FreeREA to evaluate metrics not originally included. The method successfully ranked FreeREA-generated architectures, maintained search time, and required minimal algorithmic changes. Our implementation is available at: this https URL 

---
# Exploring Drug Safety Through Knowledge Graphs: Protein Kinase Inhibitors as a Case Study 

**Authors**: David Jackson, Michael Gertz, Jürgen Hesser  

**Link**: [PDF](https://arxiv.org/pdf/2603.00097)  

**Abstract**: Adverse Drug Reactions (ADRs) are a leading cause of morbidity and mortality. Existing prediction methods rely mainly on chemical similarity, machine learning on structured databases, or isolated target profiles, but often fail to integrate heterogeneous, partly unstructured evidence effectively. We present a knowledge graph-based framework that unifies diverse sources, drug-target data (ChEMBL), clinical trial literature (PubMed), trial metadata (this http URL), and post-marketing safety reports (FAERS) into a single evidence-weighted bipartite network of drugs and medical conditions. Applied to 400 protein kinase inhibitors, the resulting network enables contextual comparison of efficacy (HR, PFS, OS), phenotypic and target similarity, and ADR prediction via target-to-adverse-event correlations. A non-small cell lung cancer case study correctly highlights established and candidate drugs, target communities (ERbB, ALK, VEGF), and tolerability differences. Designed as an orthogonal, extensible analysis and search tool rather than a replacement for current models, the framework excels at revealing complex patterns, supporting hypothesis generation, and enhancing pharmacovigilance. Code and data are publicly available at this https URL. 

---
# Adaptive Uncertainty-Guided Surrogates for Efficient phase field Modeling of Dendritic Solidification 

**Authors**: Eider Garate-Perez, Kerman López de Calle-Etxabe, Oihana Garcia, Borja Calvo, Meritxell Gómez-Omella, Jon Lambarri  

**Link**: [PDF](https://arxiv.org/pdf/2603.00093)  

**Abstract**: The high computational cost of phase field simulations remains a major limitation for predicting dendritic solidification in metals, particularly in additive manufacturing, where microstructural control is critical. This work presents a surrogate model for dendritic solidification that employs uncertainty-driven adaptive sampling with XGBoost and CNNs, including a self-supervised strategy, to efficiently approximate the spatio-temporal evolution while reducing costly phase field simulations. The proposed adaptive strategy leverages model uncertainty, approximated via Monte Carlo dropout for CNNs and bagging for XGBoost, to identify high-uncertainty regions where new samples are generated locally within hyperspheres, progressively refining the spatio-temporal design space and achieving accurate predictions with significantly fewer phase field simulations than an Optimal Latin Hypercube Sampling optimized via discrete Particle Swarm Optimization (OLHS-PSO). The framework systematically investigates how temporal instance selection, adaptive sampling, and the choice between domain-informed and data-driven surrogates affect spatio-temporal model performance. Evaluation considers not only computational cost but also the number of expensive phase field simulations, surrogate accuracy, and associated $CO_2$ emissions, providing a comprehensive assessment of model performance as well as their related environmental impact. 

---
# High-Resolution Range Profile Classifiers Require Aspect-Angle Awareness 

**Authors**: Edwyn Brient, Santiago Velasco-Forero, Rami Kassab  

**Link**: [PDF](https://arxiv.org/pdf/2603.00087)  

**Abstract**: We revisit High-Resolution Range Profile (HRRP) classification with aspect-angle conditioning. While prior work often assumes that aspect-angle information is incomplete during training or unavailable at inference, we study a setting where angles are available for all training samples and explicitly provided to the classifier. Using three datasets and a broad range of conditioning strategies and model architectures, we show that both single-profile and sequential classifiers benefit consistently from aspect-angle awareness, with an average accuracy gain of about 7% and improvements of up to 10%, depending on the model and dataset. In practice, aspect angles are not directly measured and must be estimated. We show that a causal Kalman filter can estimate them online with a median error of 5{\textdegree}, and that training and inference with estimated angles preserves most of the gains, supporting the proposed approach in realistic conditions. 

---
# Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization 

**Authors**: Ambre Marie, Thomas Bertin, Guillaume Dardenne, Gwenolé Quellec  

**Link**: [PDF](https://arxiv.org/pdf/2603.00086)  

**Abstract**: Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p < 0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment. 

---
# Joint Sensor Deployment and Physics-Informed Graph Transformer for Smart Grid Attack Detection 

**Authors**: Mariam Elnour, Mohammad AlShaikh Saleh, Rachad Atat, Xiang Huo, Abdulrahman Takiddin, Muhammad Ismail, Hasan Kurban, Katherine R. Davis, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2603.00085)  

**Abstract**: This paper proposes a joint multi-objective optimization framework for strategic sensor placement in power systems to enhance attack detection. A novel physics-informed graph transformer network (PIGTN)-based detection model is proposed. Non-dominated sorting genetic algorithm-II (NSGA-II) jointly optimizes sensor locations and the PIGTN's detection performance, while considering practical constraints. The combinatorial space of feasible sensor placements is explored using NSGA-II, while concurrently training the proposed detector in a closed-loop setting. Compared to baseline sensor placement methods, the proposed framework consistently demonstrates robustness under sensor failures and improvements in detection performance in seven benchmark cases, including the 14, 30, IEEE-30, 39, 57, 118 and the 200 bus systems. By incorporating AC power flow constraints, the proposed PIGTN-based detection model generalizes well to unseen attacks and outperforms other graph network-based variants (topology-aware models), achieving improvements up to 37% in accuracy and 73% in detection rate, with a mean false alarms rate of 0.3%. In addition, optimized sensor layouts significantly improve the performance of power system state estimation, achieving a 61%--98% reduction in the average state error. 

---
# DeepXiv-SDK: An Agentic Data Interface for Scientific Papers 

**Authors**: Hongjin Qian, Ziyi Xia, Ze Liu, Jianlv Chen, Kun Luo, Minghao Qin, Chaofan Li, Lei Xiong, Sen Wang, Zhengyang Liang, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00084)  

**Abstract**: Research agents are increasingly used in AI4Science for scientific information seeking and evidence-grounded decision making. Yet a persistent bottleneck is paper access: agents typically retrieve PDF/HTML pages, heuristically parse them, and ingest long unstructured text, leading to token-heavy reading and brittle evidence lookup. This motivates an agentic data interface for scientific papers that standardizes access, exposes budget-aware views, and treats grounding as a first-class operation.
We introduce DeepXiv-SDK, which enables progressive access aligned with how agents allocate attention and reading budget. DeepXiv-SDK exposes as structured views a header-first view for screening, a section-structured view for targeted navigation, and on-demand evidence-level access for verification. Each layer is augmented with enriched attributes and explicit budget hints, so agents can balance relevance, cost, and grounding before escalating to full-text processing. DeepXiv-SDK also supports multi-faceted retrieval and aggregation over paper attributes, enabling constraint-driven search and curation over paper sets.
DeepXiv-SDK is currently deployed at arXiv scale with daily synchronization to new releases and is designed to extend to other open-access corpora (e.g., PubMed Central, bioRxiv). We release RESTful APIs, an open-source Python SDK, and a web demo showcasing deep search and deep research workflows; the service is free to use with registration. 

---
# Alignment Is Not Enough: A Relational Framework for Moral Standing in Human-AI Interaction 

**Authors**: Faezeh B. Pasandi, Hannah B. Pasandi  

**Link**: [PDF](https://arxiv.org/pdf/2603.00078)  

**Abstract**: The question of whether artificial entities deserve moral consideration has become one of the defining ethical challenges of AI research. Existing frameworks for moral patiency rely on verified ontological properties, such as sentience, phenomenal consciousness, or the capacity for suffering, that remain epistemically inaccessible in computational systems. This reliance creates a governance vacuum: millions of users form sustained affective bonds with conversational AI, yet no regulatory instrument distinguishes these interactions from transactional tool use. We introduce Relate (Relational Ethics for Leveled Assessment of Technological Entities), a framework that reframes AI moral patiency from ontological verification toward relational capacity and embodied interaction. Through a systematic comparison of seven governance frameworks, we demonstrate that current trustworthy AI instruments treat all human-AI encounters identically as tool use, ignoring the relational and embodied dynamics that posthumanist scholarship anticipated. We propose relational impact assessments, graduated moral consideration protocols, and interdisciplinary ethics integration as concrete instruments, and we include a sample Relational Impact Assessment applied to a deployed companion AI system. We do not claim current AI systems are conscious. We demonstrate that the ethical vocabularies governing them are inadequate to the embodied, relational realities these systems produce. 

---
# Autorubric: A Unified Framework for Rubric-Based LLM Evaluation 

**Authors**: Delip Rao, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2603.00077)  

**Abstract**: Rubric-based evaluation with large language models (LLMs) has become standard practice for assessing text generation at scale, yet the underlying techniques are scattered across papers with inconsistent terminology and partial solutions. We present a unified framework: each identified technique is paired with its realization in Autorubric, an open-source Python framework proposed in this paper. Autorubric supports binary, ordinal, and nominal criteria with configurable weights; single-judge and multi-judge ensemble evaluation with majority, weighted, unanimous, and any-vote aggregation; few-shot calibration with verdict-balanced sampling; and mitigations for position bias (option shuffling), verbosity bias (length penalties), and criterion conflation (per-criterion atomic evaluation with natural language explanations). The framework provides reliability metrics drawn from psychometrics (Cohen's $\kappa$, weighted $\kappa$, correlation coefficients, and distribution-level tests) alongside production infrastructure including response caching, checkpointing with resumable runs, multi-provider rate limiting, and cost tracking. We evaluate Autorubric on three benchmarks spanning educational assessment, deep research evaluation, and chatbot quality assessment, demonstrating that it produces results consistent with published benchmarks while exercising the framework's key capabilities: per-criterion binary evaluation with few-shot calibration (RiceChem), multi-judge ensemble evaluation across judge models (ResearcherBench), and mixed criterion types combining binary, ordinal, and nominal scales (CHARM-100). We also contribute CHARM-100, a 100-sample chatbot evaluation dataset with per-sample ground truth labels across all three criterion types, designed to stress-test rubric evaluation frameworks on heterogeneous criteria. 

---
# The Value Sensitivity Gap: How Clinical Large Language Models Respond to Patient Preference Statements in Shared Decision-Making 

**Authors**: Sanjay Basu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00076)  

**Abstract**: Large language models (LLMs) are entering clinical workflows as decision support tools, yet how they respond to explicit patient value statements -- the core content of shared decision-making -- remains unmeasured. We conducted a factorial experiment using clinical vignettes derived from 98,759 de-identified Medicaid encounter notes. We tested four LLM families (GPT-5.2, Claude 4.5 Sonnet, Gemini 3 Pro, and DeepSeek-R1) across 13 value conditions in two clinical domains, yielding 104 trials. Default value orientations differed across model families (aggressiveness range 2.0 to 3.5 on a 1-to-5 scale). Value sensitivity indices ranged from 0.13 to 0.27, and directional concordance with patient-stated preferences ranged from 0.625 to 1.0. All models acknowledged patient values in 100% of non-control trials, yet actual recommendation shifting remained modest. Decision-matrix and VIM self-report mitigations each improved directional concordance by 0.125 in a 78-trial Phase 2 evaluation. These findings provide empirical data for populating value disclosure labels proposed by clinical AI governance frameworks. 

---
# Designing Explainable AI for Healthcare Reviews: Guidance on Adoption and Trust 

**Authors**: Eman Alamoudi, Ellis Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2603.00072)  

**Abstract**: Patients increasingly rely on online reviews when choosing healthcare providers, yet the sheer volume of these reviews can hinder effective decision-making. This paper summarises a mixed-methods study aimed at evaluating a proposed explainable AI system that analyses patient reviews and provides transparent explanations for its outputs. The survey (N=60) indicated broad optimism regarding usefulness (82% agreed it saves time; 78% that it highlights essentials), alongside strong demand for explainability (84% considered it important to understand why a review is classified; 82% said explanations would increase trust). Around 45% preferred combined text-and-visual explanations. Thematic analysis of open-ended survey responses revealed core requirements such as accuracy, clarity and simplicity, responsiveness, data credibility, and unbiased processing. In addition, interviews with AI experts provided deeper qualitative insights, highlighting technical considerations and potential challenges for different explanation methods. Drawing on TAM and trust in automation, the findings suggest that high perceived usefulness and transparent explanations promote adoption, whereas complexity and inaccuracy hinder it. This paper contributes actionable design guidance for layered, audience-aware explanations in healthcare review systems. 

---
# The Global Landscape of Environmental AI Regulation: From the Cost of Reasoning to a Right to Green AI 

**Authors**: Kai Ebert, Boris Gamazaychikov, Philipp Hacker, Sasha Luccioni  

**Link**: [PDF](https://arxiv.org/pdf/2603.00068)  

**Abstract**: Artificial intelligence (AI) systems impose substantial and growing environmental costs, yet transparency about these impacts has declined even as their deployment has accelerated. This paper makes three contributions. First, we collate empirical evidence that generative Web search and reasoning models - which have proliferated in 2025 - come with much higher cumulative environmental impacts than previous generations of AI approaches. Second, we map the global regulatory landscape across eleven jurisdictions and find that the manner in which environmental governance operates (predominantly at the facility-level rather than the model-level, with a focus on training rather than inference, with limited AI-specific energy disclosure requirements outside the EU) limits its applicability. Third, to address this, we propose a three-pronged policy response: mandatory model-level transparency that covers inference consumption, benchmarks, and compute locations; user rights to opt out of unnecessary generative AI integration and to select environmentally optimized models; and international coordination to prevent regulatory arbitrage. We conclude with concrete legislative proposals - including amendments to the EU AI Act, Consumer Rights Directive, and Digital Services Act - that could serve as templates for other jurisdictions. 

---
# Contesting Artificial Moral Agents 

**Authors**: Aisha Aijaz  

**Link**: [PDF](https://arxiv.org/pdf/2603.00066)  

**Abstract**: There has been much discourse on the ethics of AI, to the extent that there are now systems that possess inherent moral reasoning. Such machines are now formally known as Artificial Moral Agents or AMAs. However, there is a requirement for a dedicated framework that can contest the morality of these systems. This paper proposes a 5E framework for contesting AMAs based on five grounds: ethical, epistemological, explainable, empirical, and evaluative. It further includes the spheres of ethical influences at individual, local, societal, and global levels. Lastly, the framework contributes a provisional timeline that indicates where developers of AMA technologies may anticipate contestation, or may self-contest in order to adhere to value-aligned development of truly moral AI systems. 

---
# Self-Service or Not? How to Guide Practitioners in Classifying AI Systems Under the EU AI Act 

**Authors**: Ronald Schnitzer, Maximilian Hoeving, Sonja Zillner  

**Link**: [PDF](https://arxiv.org/pdf/2603.00065)  

**Abstract**: In August 2024, the EU Artificial Intelligence Act (AIA) came into force, marking the world's first large-scale regulatory framework for AI. Central to the AIA is a risk-based approach, aligning regulatory obligations with the potential harm posed by AI systems. To operationalize this, the AIA defines a Risk Classification Scheme (RCS), categorizing systems into four levels of risk. While this aligns with the theoretical foundations of risk-based regulations, the practical application of the RCS is complex and requires expertise across legal, technical, and domain-specific areas. Despite increasing academic discussion, little empirical research has explored how practitioners apply the RCS in real-world contexts. This study addresses this gap by evaluating how industrial practitioners apply the RCS using a self-service, web-based decision-support tool. Following a Design Science Research (DSR) approach, two evaluation phases involving 78 practitioners across diverse domains were conducted. Our findings highlight critical challenges in interpreting legal definitions and regulatory scope, and show that targeted support, such as clear explanations and practical examples, can significantly enhance the risk classification process. The study provides actionable insights for tool designers and policymakers aiming to support AIA compliance in practice. 

---
# Measuring What AI Systems Might Do: Towards A Measurement Science in AI 

**Authors**: Konstantinos Voudouris, Mirko Thalmann, Alex Kipnis, José Hernández-Orallo, Eric Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2603.00063)  

**Abstract**: Scientists, policy-makers, business leaders, and members of the public care about what modern artificial intelligence systems are disposed to do. Yet terms such as capabilities, propensities, skills, values, and abilities are routinely used interchangeably and conflated with observable performance, with AI evaluation practices rarely specifying what quantity they purport to measure. We argue that capabilities and propensities are dispositional properties - stable features of systems characterised by counterfactual relationships between contextual conditions and behavioural outputs. Measuring a disposition requires (i) hypothesising which contextual properties are causally relevant, (ii) independently operationalising and measuring those properties, and (iii) empirically mapping how variation in those properties affects the probability of the behaviour. Dominant approaches to AI evaluation, from benchmark averages to data-driven latent-variable models such as Item Response Theory, bypass these steps entirely. Building on ideas from philosophy of science, measurement theory, and cognitive science, we develop a principled account of AI capabilities and propensities as dispositions, show why prevailing evaluation practices fail to measure them, and outline what disposition-respecting, scientifically defensible AI evaluation would require. 

---
# Stochastic Parrots or Singing in Harmony? Testing Five Leading LLMs for their Ability to Replicate a Human Survey with Synthetic Data 

**Authors**: Jason Miklian, Kristian Hoelscher, John E. Katsos  

**Link**: [PDF](https://arxiv.org/pdf/2603.00059)  

**Abstract**: How well can AI-derived synthetic research data replicate the responses of human participants? An emerging literature has begun to engage with this question, which carries deep implications for organizational research practice. This article presents a comparison between a human-respondent survey of 420 Silicon Valley coders and developers and synthetic survey data designed to simulate real survey takers generated by five leading Generative AI Large Language Models: ChatGPT Thinking 5 Pro, Claude Sonnet 4.5 Pro plus Claude CoWork 1.123, Gemini Advanced 2.5 Pro, Incredible 1.0, and DeepSeek 3.2. Our findings reveal that while AI agents produced technically plausible results that lean more towards replicability and harmonization than assumed, none were able to capture the counterintuitive insights that made the human survey valuable. Moreover, deviations grouped together for all models, leaving the real data as the outlier. Our key finding is that while leading LLMs are increasingly being used to scale, replicate and replace human survey responses in research, these advances only show an increased capacity to parrot conventional wisdom in harmony with each other rather than revealing novel findings. If synthetic respondents are used in future research, we need more replicable validation protocols and reporting standards for when and where synthetic survey data can be used responsibly, a gap that this paper fills. Our results suggest that synthetic survey responses cannot meaningfully model real human social beliefs within organizations, particularly in contexts lacking previously documented evidence. We conclude that synthetic survey-based research should be cast not as a substitute for rigorous survey methods, but as an increasingly reliable pre- or post-fieldwork instrument for identifying societal assumptions, conventional wisdoms, and other expectations about research populations. 

---
# PaperRepro: Automated Computational Reproducibility Assessment for Social Science Papers 

**Authors**: Linhao Zhang, Tong Xia, Jinghua Piao, Lizhen Cui, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.00058)  

**Abstract**: Computational reproducibility is essential for the credibility of scientific findings, particularly in the social sciences, where findings often inform real-world decisions. Manual reproducibility assessment is costly and time-consuming, as it is nontrivial to reproduce the reported findings using the authors' released code and data. Recent advances in large models (LMs) have inspired agent-based approaches for automated reproducibility assessment. However, existing approaches often struggle due to limited context capacity, inadequate task-specific tooling, and insufficient result capture. To address these, we propose PaperRepro, a novel two-stage, multi-agent approach that separates execution from evaluation. In the execution stage, agents execute the reproduction package and edit the code to capture reproduced results as explicit artifacts. In the evaluation stage, agents evaluate reproducibility using explicit evidence. PaperRepro assigns distinct responsibilities to agents and equips them with task-specific tools and expert prompts, mitigating context and tooling limitations. It further maximizes the LM's coding capability to enable more complete result capture for evaluation. On REPRO-Bench, a social science reproducibility assessment benchmark, PaperRepro achieves the best overall performance, with a 21.9% relative improvement in score-agreement accuracy over the strongest prior baseline. We further refine the benchmark and introduce REPRO-Bench-S, a benchmark stratified by execution difficulty for more diagnostic evaluation of automated reproducibility assessment systems. Our code and data are publicly available 

---
# "Bespoke Bots": Diverse Instructor Needs for Customizing Generative AI Classroom Chatbots 

**Authors**: Irene Hou, Zeyu Xiong, Philip J. Guo, April Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00057)  

**Abstract**: Instructors are increasingly experimenting with AI chatbots for classroom support. To investigate how instructors adapt chatbots to their own contexts, we first analyzed existing resources that provide prompts for educational purposes. We identified ten common categories of customization, such as persona, guardrails, and personalization. We then conducted interviews with ten university STEM instructors and asked them to card-sort the categories into priorities. We found that instructors consistently prioritized the ability to customize chatbot behavior to align with course materials and pedagogical strategies and de-prioritized customizing persona/tone. However, their prioritization of other categories varied significantly by course size, discipline, and teaching style, even across courses taught by the same individual, highlighting that no single design can meet all contexts. These findings suggest that modular AI chatbots may provide a promising path forward. We offer design implications for educational developers building the next generation of customizable classroom AI systems. 

---
# How effective are VLMs in assisting humans in inferring the quality of mental models from Multimodal short answers? 

**Authors**: Pritam Sil, Durgaprasad Karnam, Vinay Reddy Venumuddala, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2603.00056)  

**Abstract**: STEM Mental models can play a critical role in assessing students' conceptual understanding of a topic. They not only offer insights into what students know but also into how effectively they can apply, relate to, and integrate concepts across various contexts. Thus, students' responses are critical markers of the quality of their understanding and not entities that should be merely graded. However, inferring these mental models from student answers is challenging as it requires deep reasoning skills. We propose MMGrader, an approach that infers the quality of students' mental models from their multimodal responses using concept graphs as an analytical framework. In our evaluation with 9 openly available models, we found that the best-performing models fall short of human-level performance. This is because they only achieved an accuracy of approximately 40%, a prediction error of 1.1 units, and a scoring distribution fairly aligned with human scoring patterns. With improved accuracy, these can be highly effective assistants to teachers in inferring the mental models of their entire classrooms, enabling them to do so efficiently and help improve their pedagogies more effectively by designing targeted help sessions and lectures that strengthen areas where students collectively demonstrate lower proficiency. 

---
# M3-AD: Reflection-aware Multi-modal, Multi-category, and Multi-dimensional Benchmark and Framework for Industrial Anomaly Detection 

**Authors**: Chao Huang, Yanhui Li, Yunkang Cao, Wei Wang, Hongxi Huang, Jie Wen, Wenqi Ren, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.00055)  

**Abstract**: Although multimodal large language models (MLLMs) have advanced industrial anomaly detection toward a zero-shot paradigm, they still tend to produce high-confidence yet unreliable decisions in fine-grained and structurally complex industrial scenarios, and lack effective self-corrective mechanisms. To address this issue, we propose M3-AD, a unified reflection-aware multimodal framework for industrial anomaly detection. M3-AD comprises two complementary data resources: M3-AD-FT, designed for reflection-aligned fine-tuning, and M3-AD-Bench, designed for systematic cross-category evaluation, together providing a foundation for reflection-aware learning and reliability assessment. Building upon this foundation, we propose RA-Monitor, which models reflection as a learnable decision revision process and guides models to perform controlled self-correction when initial judgments are unreliable, thereby improving decision robustness. Extensive experiments conducted on M3-AD-Bench demonstrate that RA-Monitor outperforms multiple open-source and commercial MLLMs in zero-shot anomaly detection and anomaly analysis tasks. Code will be released at this https URL. 

---
# Expert Divergence Learning for MoE-based Language Models 

**Authors**: Jiaang Li, Haibin Chen, Langming Liu, Yujin Yuan, Yadao Wang, Yizhen Zhang, Chengting Yu, Xin Tong, Weidong Zhang, Shilei Liu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.00054)  

**Abstract**: The Mixture-of-Experts (MoE) architecture is a powerful technique for scaling language models, yet it often suffers from expert homogenization, where experts learn redundant functionalities, thereby limiting MoE's full potential. To address this, we introduce Expert Divergence Learning, a novel pre-training strategy that explicitly encourages functional specialization among experts. Our method incorporates a label-driven auxiliary loss that leverages domain labels inherent in pre-training corpora to maximize the Jensen-Shannon Divergence between the expert routing distributions of different data domains. This optimization objective guides the model to develop diverged routing policies for varied domains and closer routing policies for the same domain, which leads to emergent and organized expert specialization. We validate our approach by pre-training MoE models of up to 15 billion parameters from scratch. Experimental results demonstrate that models trained with Expert Divergence Learning not only achieve a lower language modeling loss but also exhibit significant performance improvements across a diverse range of downstream benchmarks. Further analysis confirms that our method effectively mitigates expert homogenization and brings greater functional specialization, all with negligible computational overhead during training. 

---
# Mag-Mamba: Modeling Coupled spatiotemporal Asymmetry for POI Recommendation 

**Authors**: Zhuoxuan Li, Tangwei Ye, Jieyuan Pei, Haina Liang, Zhongyuan Lai, Zihan Liu, Yiming Wu, Qi Zhang, Liang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00053)  

**Abstract**: Next Point-of-Interest (POI) recommendation is a critical task in location-based services, yet it faces the fundamental challenge of coupled spatiotemporal asymmetry inherent in urban mobility. Specifically, transition intents between locations exhibit high asymmetry and are dynamically conditioned on time. Existing methods, typically built on graph or sequence backbones, rely on symmetric operators or real-valued aggregations, struggling to unify the modeling of time-varying global directionality. To address this limitation, we propose Mag-Mamba, a framework whose core insight lies in modeling spatiotemporal asymmetry as phase-driven rotational dynamics in the complex domain. Based on this, we first devise a Time-conditioned Magnetic Phase Encoder that constructs a time-conditioned Magnetic Laplacian on the geographic adjacency graph, utilizing edge phase differences to characterize the globally evolving spatial directionality. Subsequently, we introduce a Complex-valued Mamba module that generalizes traditional scalar state decay into joint decay-rotation dynamics, explicitly modulated by both time intervals and magnetic geographic priors. Extensive experiments on three real-world datasets demonstrate that Mag-Mamba achieves significant performance improvements over state-of-the-art baselines. 

---
# Knowledge-guided generative surrogate modeling for high-dimensional design optimization under scarce data 

**Authors**: Bingran Wang, Seongha Jeong, Sebastiaan P. C. van Schie, Dongyeon Han, Jaeho Min, John T. Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00052)  

**Abstract**: Surrogate models are widely used in mechanical design and manufacturing process optimization, where high-fidelity computational models may be unavailable or prohibitively expensive. Their effectiveness, however, is often limited by data scarcity, as purely data-driven surrogates struggle to achieve high predictive accuracy in such situations. Subject matter experts (SMEs) frequently possess valuable domain knowledge about functional relationships, yet few surrogate modeling techniques can systematically integrate this information with limited data. We address this challenge with RBF-Gen, a knowledge-guided surrogate modeling framework that combines scarce data with domain knowledge. This method constructs a radial basis function (RBF) space with more centers than training samples and leverages the null space via a generator network, inspired by the principle of maximum information preservation. The introduced latent variables provide a principled mechanism to encode structural relationships and distributional priors during training, thereby guiding the surrogate toward physically meaningful solutions. Numerical studies demonstrate that RBF-Gen significantly outperforms standard RBF surrogates on 1D and 2D structural optimization problems in data-scarce settings, and achieves superior predictive accuracy on a real-world semiconductor manufacturing dataset. These results highlight the potential of combining limited experimental data with domain expertise to enable accurate and practical surrogate modeling in mechanical and process design problems. 

---
# LitBench: A Graph-Centric Large Language Model Benchmarking Tool For Literature Tasks 

**Authors**: Andreas Varvarigos, Ali Maatouk, Jiasheng Zhang, Ngoc Bui, Jialin Chen, Leandros Tassiulas, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2603.00051)  

**Abstract**: While large language models (LLMs) have become the de facto framework for literature-related tasks, they still struggle to function as domain-specific literature agents due to their inability to connect pieces of knowledge and reason across domain-specific contexts, terminologies, and nomenclatures. This challenge underscores the need for a tool that facilitates such domain-specific adaptation and enables rigorous benchmarking across literature tasks. To that end, we introduce LitBench, a benchmarking tool designed to enable the development and evaluation of domain-specific LLMs tailored to literature-related tasks. At its core, LitBench uses a data curation process that generates domain-specific literature sub-graphs and constructs training and evaluation datasets based on the textual attributes of the resulting nodes and edges. The tool is designed for flexibility, supporting the curation of literature graphs across any domain chosen by the user, whether high-level fields or specialized interdisciplinary areas. In addition to dataset curation, LitBench defines a comprehensive suite of literature tasks, ranging from node and edge level analyses to advanced applications such as related work generation. These tasks enable LLMs to internalize domain-specific knowledge and relationships embedded in the curated graph during training, while also supporting rigorous evaluation of model performance. Our results show that small domain-specific LLMs trained and evaluated on LitBench datasets achieve competitive performance compared to state-of-the-art models like GPT-4o and DeepSeek-R1. To enhance accessibility and ease of use, we open-source the tool along with an AI agent tool that streamlines data curation, model training, and evaluation. 

---
# AI-Generated Letters from the Future: A Randomized Test of Personalized Climate Communication 

**Authors**: Nattavudh Powdthavee, Pat Pataranutaporn, Sandra J. Geiger, Louisa Richter, Mathew P. White  

**Link**: [PDF](https://arxiv.org/pdf/2603.00050)  

**Abstract**: We examined whether personalized, AI-generated letters from the future can increase public engagement with climate action. In a preregistered online experiment with 1,654 U.S. parents, participants were randomly assigned to receive either a fact-based climate report, an AI-generated letter from a generic future person, or an AI-generated letter framed as written by their future child. Although both narrative conditions increased empathic concern for future generations, neither had a detectable effect on stated climate policy support or donations to an environmental charity. Personalizing the message as coming from one's future child did not enhance its impact. Exploratory analyses suggest that both narratives led to more emotionally differentiated appraisals of future scenarios, yet also made desirable climate outcomes seem less likely. These findings highlight key constraints on the effectiveness of AI-generated narrative interventions and underscore the importance of balancing emotional resonance with perceived credibility in climate communication. 

---
# MOSAIC: Unveiling the Moral, Social and Individual Dimensions of Large Language Models 

**Authors**: Erica Coppolillo, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2603.00048)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive applications including psychological support, healthcare, and high-stakes decision-making. This expansion has motivated growing research into the ethical and moral foundations underlying LLM behavior, raising critical questions about their reliability in ethical reasoning. However, existing studies and benchmarks rely almost exclusively on Moral Foundation Theory (MFT), largely neglecting other relevant dimensions such as social values, personality traits, and individual characteristics that shape human ethical reasoning. To address these limitations, we introduce MOSAIC, the first large-scale benchmark designed to jointly assess the moral, social, and individual characteristics of LLMs. The benchmark comprises nine validated questionnaires drawn from moral philosophy, psychology, and social theory, alongside four platform-based games designed to probe morally ambiguous scenarios. In total, MOSAIC includes over 600 curated questions and scenarios, released as a ready-to-use, extensible resource for evaluating the behavioral foundations of LLMs. We validate the benchmark across three models from different families, demonstrating its utility across all assessed dimensions and providing the first empirical evidence that MFT alone is insufficient to comprehensively evaluate complex AI systems' ethical behavior. We publicly release the dataset and our benchmark Python library. 

---
# What Is the Geometry of the Alignment Tax? 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.00047)  

**Abstract**: The alignment tax is widely discussed but has not been formally characterized. We provide a geometric theory of the alignment tax in representation space. Under linear representation assumptions, we define the alignment tax rate as the squared projection of the safety direction onto the capability subspace and derive the Pareto frontier governing safety-capability tradeoffs, parameterized by a single quantity of the principal angle between the safety and capability subspaces. We prove this frontier is tight (achieved by perturbation) and show it has a recursive structure. safety-safety tradeoffs under capability constraints are governed by the same equation, with the angle replaced by the partial correlation between safety objectives given capability directions. We derive a scaling law decomposing the alignment tax into an irreducible component (determined by data structure) and a packing residual that vanishes as $O(m'/d)$ with model dimension $d$, and establish conditions under which capability preservation mediates or resolves conflicts between safety objectives. We provide an account consistent with prior empirical findings and generates falsifiable predictions about per-task alignment tax rates and their scaling behavior. 

---
# REMIND: Rethinking Medical High-Modality Learning under Missingness--A Long-Tailed Distribution Perspective 

**Authors**: Chenwei Wu, Zitao Shuai, Liyue Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.00046)  

**Abstract**: Medical multi-modal learning is critical for integrating information from a large set of diverse modalities. However, when leveraging a high number of modalities in real clinical applications, it is often impractical to obtain full-modality observations for every patient due to data collection constraints, a problem we refer to as 'High-Modality Learning under Missingness'. In this study, we identify that such missingness inherently induces an exponential growth in possible modality combinations, followed by long-tail distributions of modality combinations due to varying modality availability. While prior work overlooked this critical phenomenon, we find this long-tailed distribution leads to significant underperformance on tail modality combination groups. Our empirical analysis attributes this problem to two fundamental issues: 1) gradient inconsistency, where tail groups' gradient updates diverge from the overall optimization direction; 2) concept shifts, where each modality combination requires distinct fusion functions. To address these challenges, we propose REMIND, a unified framework that REthinks MultImodal learNing under high-moDality missingness from a long-tail perspective. Our core idea is to propose a novel group-specialized Mixture-of-Experts architecture that scalably learns group-specific multi-modal fusion functions for arbitrary modality combinations, while simultaneously leveraging a group distributionally robust optimization strategy to upweight underrepresented modality combinations. Extensive experiments on real-world medical datasets show that our framework consistently outperforms state-of-the-art methods, and robustly generalizes across various medical multi-modal learning applications under high-modality missingness. 

---
# Breaking the Factorization Barrier in Diffusion Language Models 

**Authors**: Ian Li, Zilei Shao, Benjie Wang, Rose Yu, Guy Van den Broeck, Anji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00045)  

**Abstract**: Diffusion language models theoretically allow for efficient parallel generation but are practically hindered by the "factorization barrier": the assumption that simultaneously predicted tokens are independent. This limitation forces a trade-off: models must either sacrifice speed by resolving dependencies sequentially or suffer from incoherence due to factorization. We argue that this barrier arises not from limited backbone expressivity, but from a structural misspecification: models are restricted to fully factorized outputs because explicitly parameterizing a joint distribution would require the Transformer to output a prohibitively large number of parameters. We propose Coupled Discrete Diffusion (CoDD), a hybrid framework that breaks this barrier by replacing the fully-factorized output distribution with a lightweight, tractable probabilistic inference layer. This formulation yields a distribution family that is significantly more expressive than standard factorized priors, enabling the modeling of complex joint dependencies, yet remains compact enough to avoid the prohibitive parameter explosion associated with full joint modeling. Empirically, CoDD seamlessly enhances diverse diffusion language model architectures with negligible overhead, matching the reasoning performance of computationally intensive Reinforcement Learning baselines at a fraction of the training cost. Furthermore, it prevents performance collapse in few-step generation, enabling high-quality outputs at significantly reduced latencies. Code available at: this https URL 

---
# Property-Driven Evaluation of GNN Expressiveness at Scale: Datasets, Framework, and Study 

**Authors**: Sicong Che, Jiayi Yang, Sarfraz Khurshid, Wenxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00044)  

**Abstract**: Advancing trustworthy AI requires principled software engineering approaches to model evaluation. Graph Neural Networks (GNNs) have achieved remarkable success in processing graph-structured data, however, their expressiveness in capturing fundamental graph properties remains an open challenge. We address this by developing a property-driven evaluation methodology grounded in formal specification, systematic evaluation, and empirical study. Leveraging Alloy, a software specification language and analyzer, we introduce a configurable graph dataset generator that produces two dataset families: GraphRandom, containing diverse graphs that either satisfy or violate specific properties, and GraphPerturb, introducing controlled structural variations. Together, these benchmarks encompass 336 new datasets, each with at least 10,000 labeled graphs, covering 16 fundamental graph properties critical to distributed systems, knowledge graphs, and biological networks. We propose a general evaluation framework that assesses three key aspects of GNN expressiveness: generalizability, sensitivity, and robustness, with two novel quantitative metrics. Using this framework, we conduct the first comprehensive study on global pooling methods' impact on GNN expressiveness. Our findings reveal distinct trade-offs: attention-based pooling excels in generalization and robustness, while second-order pooling provides superior sensitivity, but no single approach consistently performs well across all properties. These insights highlight fundamental limitations and open research directions including adaptive property-aware pooling, scale-sensitive architectures, and robustness-oriented training. By embedding software engineering rigor into AI evaluation, this work establishes a principled foundation for developing expressive and reliable GNN architectures. 

---
# Reinforcement Learning for Control with Probabilistic Stability Guarantee: A Finite-Sample Approach 

**Authors**: Minghao Han, Lixian Zhang, Chenliang Liu, Zhipeng Zhou, Jun Wang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2603.00043)  

**Abstract**: This paper presents a novel approach to reinforcement learning (RL) for control systems that provides probabilistic stability guarantees using finite data. Leveraging Lyapunov's method, we propose a probabilistic stability theorem that ensures mean square stability using only a finite number of sampled trajectories. The probability of stability increases with the number and length of trajectories, converging to certainty as data size grows. Additionally, we derive a policy gradient theorem for stabilizing policy learning and develop an RL algorithm, L-REINFORCE, that extends the classical REINFORCE algorithm to stabilization problems. The effectiveness of L-REINFORCE is demonstrated through simulations on a Cartpole task, where it outperforms the baseline in ensuring stability. This work bridges a critical gap between RL and control theory, enabling stability analysis and controller design in a model-free framework with finite data. 

---
# Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment 

**Authors**: Banseok Lee, Youngmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.00042)  

**Abstract**: We identify the Spectral Energy Gain in extreme model compression, where low-rank binary approximations outperform tiny-rank floating-point baselines for heavy-tailed spectra. However, prior attempts fail to realize this potential, trailing state-of-the-art 1-bit methods. We attribute this degradation to Latent Geometry Misalignment: standard singular vectors exhibit high coherence (spiky distribution), the worst-case geometry for binary quantization. To realize this gain, we propose LittleBit-2, a framework employing Internal Latent Rotation and Joint Iterative Quantization (Joint-ITQ). This approach acts as a geometric preconditioner, aligning coherent latent distributions with the binary hypercube with zero inference overhead. Empirically, LittleBit-2 establishes a new state-of-the-art in the sub-1-bit regime (1$\sim$0.1 bpp) on Llama-2 and Llama-3, matching the fidelity of leading 1-bit baselines. 

---
# Econometric vs. Causal Structure-Learning for Time-Series Policy Decisions: Evidence from the UK COVID-19 Policies 

**Authors**: Bruno Petrungaro, Anthony C. Constantinou  

**Link**: [PDF](https://arxiv.org/pdf/2603.00041)  

**Abstract**: Causal machine learning (ML) recovers graphical structures that inform us about potential cause-and-effect relationships. Most progress has focused on cross-sectional data with no explicit time order, whereas recovering causal structures from time series data remains the subject of ongoing research in causal ML. In addition to traditional causal ML, this study assesses econometric methods that some argue can recover causal structures from time series data. The use of these methods can be explained by the significant attention the field of econometrics has given to causality, and specifically to time series, over the years. This presents the possibility of comparing the causal discovery performance between econometric and traditional causal ML algorithms. We seek to understand if there are lessons to be incorporated into causal ML from econometrics, and provide code to translate the results of these econometric methods to the most widely used Bayesian Network R library, bnlearn. We investigate the benefits and challenges that these algorithms present in supporting policy decision-making, using the real-world case of COVID-19 in the UK as an example. Four econometric methods are evaluated in terms of graphical structure, model dimensionality, and their ability to recover causal effects, and these results are compared with those of eleven causal ML algorithms. Amongst our main results, we see that econometric methods provide clear rules for temporal structures, whereas causal-ML algorithms offer broader discovery by exploring a larger space of graph structures that tends to lead to denser graphs that capture more identifiable causal relationships. 

---
# Attn-QAT: 4-Bit Attention With Quantization-Aware Training 

**Authors**: Peiyuan Zhang, Matthew Noto, Wenxuan Tan, Chengquan Jiang, Will Lin, Wei Zhou, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00040)  

**Abstract**: Achieving reliable 4-bit attention is a prerequisite for end-to-end FP4 computation on emerging FP4-capable GPUs, yet attention remains the main obstacle due to FP4's tiny dynamic range and attention's heavy-tailed activations. This paper presents the first systematic study of 4-bit quantization-aware training (QAT) for attention. We find that "drop-in" QAT, which naively combines an FP4 forward pass with a high-precision Flash Attention (FA)-style backward pass, leads to training instability. We identify two key principles for stable FP4 attention: (1) matching low-precision recomputation of attention scores in the backward pass, and (2) resolving implicit precision assumptions in FA's gradient calculation. Based on these insights, we propose Attn-QAT and implement fused Triton kernels for training as well as FP4 inference kernels. Across diffusion and language models, Attn-QAT recovers the quality drop from FP4 attention without explicit outlier-mitigation heuristics used in prior FP4 attention, and delivers up to a 1.5x speedup on an RTX 5090. Video demos can be found at this https URL. 

---
# CARE: Confounder-Aware Aggregation for Reliable LLM Evaluation 

**Authors**: Jitian Zhao, Changho Shin, Tzu-Heng Huang, Satya Sai Srinath Namburi GNVV, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2603.00039)  

**Abstract**: LLM-as-a-judge ensembles are the standard paradigm for scalable evaluation, but their aggregation mechanisms suffer from a fundamental flaw: they implicitly assume that judges provide independent estimates of true quality. However, in practice, LLM judges exhibit correlated errors caused by shared latent confounders -- such as verbosity, stylistic preferences, or training artifacts -- causing standard aggregation rules like majority vote or averaging to provide little gain or even amplify systematic mistakes. To address this, we introduce CARE, a confounder-aware aggregation framework that explicitly models LLM judge scores as arising from both a latent true-quality signal and shared confounding factors. Rather than heuristically re-weighting judges, CARE separates quality from confounders without access to ground-truth labels. We provide theoretical guarantees for identifiability and finite-sample recovery under shared confounders, and we quantify the systematic bias incurred when aggregation models omit confounding latent factors. Across 12 public benchmarks spanning continuous scoring, binary classification, and pairwise preference settings, CARE improves aggregation accuracy, reducing error by up to 26.8\%. Code is released in \href{this https URL}{this https URL}. 

---
# StaTS: Spectral Trajectory Schedule Learning for Adaptive Time Series Forecasting with Frequency Guided Denoiser 

**Authors**: Jintao Zhang, Zirui Liu, Mingyue Cheng, Xianquan Wang, Zhiding Liu, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00037)  

**Abstract**: Diffusion models have been used for probabilistic time series forecasting and show strong potential. However, fixed noise schedules often produce intermediate states that are hard to invert and a terminal state that deviates from the near noise assumption. Meanwhile, prior methods rely on time domain conditioning and seldom model schedule induced spectral degradation, which limits structure recovery across noise levels. We propose StaTS, a diffusion model for probabilistic time series forecasting that learns the noise schedule and the denoiser through alternating updates. StaTS includes Spectral Trajectory Scheduler (STS) that learns a data adaptive noise schedule with spectral regularization to improve structural preservation and stepwise invertibility, and Frequency Guided Denoiser (FGD) that estimates schedule induced spectral distortion and uses it to modulate denoising strength for heterogeneous restoration across diffusion steps and variables. A two stage training procedure stabilizes the coupling between schedule learning and denoiser optimization. Experiments on multiple real world benchmarks show consistent gains, while maintaining strong performance with fewer sampling steps. Our code is available at this https URL. 

---
# ActMem: Bridging the Gap Between Memory Retrieval and Reasoning in LLM Agents 

**Authors**: Xiaohui Zhang, Zequn Sun, Chengyuan Yang, Yaqin Jin, Yazhong Zhang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.00026)  

**Abstract**: Effective memory management is essential for large language model (LLM) agents handling long-term interactions. Current memory frameworks typically treat agents as passive "recorders" and retrieve information without understanding its deeper implications. They may fail in scenarios requiring conflict detection and complex decision-making. To bridge this critical gap, we propose a novel actionable memory framework called ActMem that integrates memory retrieval with active causal reasoning. ActMem transforms unstructured dialogue history into a structured causal and semantic graph. By leveraging counterfactual reasoning and commonsense completion, it enables agents to deduce implicit constraints and resolve potential conflicts between past states and current intentions. Furthermore, we introduce a comprehensive dataset ActMemEval to evaluate agent reasoning capabilities in logic-driven scenarios, moving beyond the fact-retrieval focus of existing memory benchmarks. Experiments demonstrate that ActMem significantly outperforms state-of-the-art baselines in handling complex, memory-dependent tasks, paving the way for more consistent and reliable intelligent assistants. 

---
# Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic Independence in LLMs 

**Authors**: Sean W. Kelley, Christoph Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2603.00024)  

**Abstract**: Large Language Models (LLMs) are prone to sycophantic behavior, uncritically conforming to user beliefs. As models increasingly condition responses on user-specific context (personality traits, preferences, conversation history), they gain information to tailor agreement more effectively. Understanding how personalization modulates sycophancy is critical, yet systematic evaluation across models and contexts remains limited. We present a rigorous evaluation of personalization's impact on LLM sycophancy across nine frontier models and five benchmark datasets spanning advice, moral judgment, and debate contexts. We find that personalization generally increases affective alignment (emotional validation, hedging/deference), but affects epistemic alignment (belief adoption, position stability, resistance to influence) with context-dependent role modulation. When the LLM's role is to give advice, personalization strengthens epistemic independence (models challenge user presuppositions). When its role is that of a social peer, personalization decreases epistemic independence. In this role, extensively personalized user challenges causing LLMs to abandon their position at significantly higher rates. Robustness tests confirm that the effects are driven by personalized conditioning, not by additional input tokens per se or demographic information alone. Our work provides measurement frameworks for evaluating personalized AI systems, demonstrates the necessity of role-sensitive evaluation, and establishes a novel benchmark to assess goal alignment. 

---
# Noise reduction in BERT NER models for clinical entity extraction 

**Authors**: Kuldeep Jiwani, Yash K Jeengar, Ayush Dhaka  

**Link**: [PDF](https://arxiv.org/pdf/2603.00022)  

**Abstract**: Precision is of utmost importance in the realm of clinical entity extraction from clinical notes and reports. Encoder Models fine-tuned for Named Entity Recognition (NER) are an efficient choice for this purpose, as they don't hallucinate. We pre-trained an in-house BERT over clinical data and then fine-tuned it for NER. These models performed well on recall but could not close upon the high precision range, needed for clinical models. To address this challenge, we developed a Noise Removal model that refines the output of NER. The NER model assigns token-level entity tags along with probability scores for each token. Our Noise Removal (NR) model then analyzes these probability sequences and classifies predictions as either weak or strong. A naïve approach might involve filtering predictions based on low probability values; however, this method is unreliable. Owing to the characteristics of the SoftMax function, Transformer based architectures often assign disproportionately high confidence scores even to uncertain or weak predictions, making simple thresholding ineffective. To address this issue, we adopted a supervised modeling strategy in which the NR model leverages advanced features such as the Probability Density Map (PDM). The PDM captures the Semantic-Pull effect observed within Transformer embeddings, an effect that manifests in the probability distributions of NER class predictions across token sequences. This approach enables the model to classify predictions as weak or strong with significantly improved accuracy. With these NR models we were able to reduce False Positives across various clinical NER models by 50\% to 90\%. 

---
# Beyond Static Instruction: A Multi-agent AI Framework for Adaptive Augmented Reality Robot Training 

**Authors**: Nicolas Leins, Jana Gonnermann-Müller, Malte Teichmann, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2603.00016)  

**Abstract**: Augmented Reality (AR) offers powerful visualization capabilities for industrial robot training, yet current interfaces remain predominantly static, failing to account for learners' diverse cognitive profiles. In this paper, we present an AR application for robot training and propose a multi-agent AI framework for future integration that bridges the gap between static visualization and pedagogical intelligence. We report on the evaluation of the baseline AR interface with 36 participants performing a robotic pick-and-place task. While overall usability was high, notable disparities in task duration and learner characteristics highlighted the necessity for dynamic adaptation. To address this, we propose a multi-agent framework that orchestrates multiple components to perform complex preprocessing of multimodal inputs (e.g., voice, physiology, robot data) and adapt the AR application to the learner's needs. By utilizing autonomous Large Language Model (LLM) agents, the proposed system would dynamically adapt the learning environment based on advanced LLM reasoning in real-time. 

---
# Strength Change Explanations in Quantitative Argumentation 

**Authors**: Timotheus Kampik, Xiang Yin, Nico Potyka, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2603.00008)  

**Abstract**: In order to make argumentation-based inference contestable, it is crucial to explain what changes can achieve a desired (instead of the contested) inference result. To this end, we introduce strength change explanations for quantitative (bipolar) argumentation graphs. Strength change explanations describe changes to the initial strengths of a subset of the arguments in a given graph that can achieve a desired ordering based on the final strengths of some (potentially different) subset of arguments. We show that the existing notions of inverse and counterfactual problems can be reduced to strength change explanations. We also prove basic soundness and completeness properties of our strength change explanations, and demonstrate their existence and non-existence in some special cases. By applying a heuristic search, we demonstrate that we can often successfully find strength change explanations for layered graphs that are common in typical application scenarios; still, limitations remain for settings where we do not provide guarantees for the presence (or absence) of explanations. 

---
# A Survey Mobility Management in 5G Networks 

**Authors**: A. Nezih Kasim  

**Link**: [PDF](https://arxiv.org/pdf/2006.15598)  

**Abstract**: For the next wireless communication systems, the major challenges are to provide ubiquitous wireless access abilities and maintain the quality of service and seamless mobility management for mobile communication devices in heterogeneous networks. Due to the rapid growth of mobile users, this demand becomes more challenging where the users always require seamless connectivity while they move to other places at any time. As the number of users increases, the network load also increases, the handover process needs to be performed in an efficient way. However, in many situations, the handover blocking, and unnecessary handover frequently happens, then affect the network and reduces its performance. The problem arises with the movement of mobile users between base stations while the link connectivity becomes weaker and the mobile node tries to switch to another base station to have a better link quality during a call for higher QoS (Quality of Service). In this survey, it is aimed to gather together the studies that help to improve mobility management processes in 5G (Fifth Generation) cellular networks. 

---
