# Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution 

**Authors**: Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.10696)  

**Abstract**: Procedural memory enables large language model (LLM) agents to internalize "how-to" knowledge, theoretically reducing redundant trial-and-error. However, existing frameworks predominantly suffer from a "passive accumulation" paradigm, treating memory as a static append-only archive. To bridge the gap between static storage and dynamic reasoning, we propose $\textbf{ReMe}$ ($\textit{Remember Me, Refine Me}$), a comprehensive framework for experience-driven agent evolution. ReMe innovates across the memory lifecycle via three mechanisms: 1) $\textit{multi-faceted distillation}$, which extracts fine-grained experiences by recognizing success patterns, analyzing failure triggers and generating comparative insights; 2) $\textit{context-adaptive reuse}$, which tailors historical insights to new contexts via scenario-aware indexing; and 3) $\textit{utility-based refinement}$, which autonomously adds valid memories and prunes outdated ones to maintain a compact, high-quality experience pool. Extensive experiments on BFCL-V3 and AppWorld demonstrate that ReMe establishes a new state-of-the-art in agent memory system. Crucially, we observe a significant memory-scaling effect: Qwen3-8B equipped with ReMe outperforms larger, memoryless Qwen3-14B, suggesting that self-evolving memory provides a computation-efficient pathway for lifelong learning. We release our code and the $\texttt{this http URL}$ dataset to facilitate further research. 

---
# LLMs Can Assist with Proposal Selection at Large User Facilities 

**Authors**: Lijie Ding, Janell Thomson, Jon Taylor, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2512.10895)  

**Abstract**: We explore how large language models (LLMs) can enhance the proposal selection process at large user facilities, offering a scalable, consistent, and cost-effective alternative to traditional human review. Proposal selection depends on assessing the relative strength among submitted proposals; however, traditional human scoring often suffers from weak inter-proposal correlations and is subject to reviewer bias and inconsistency. A pairwise preference-based approach is logically superior, providing a more rigorous and internally consistent basis for ranking, but its quadratic workload makes it impractical for human reviewers. We address this limitation using LLMs. Leveraging the uniquely well-curated proposals and publication records from three beamlines at the Spallation Neutron Source (SNS), Oak Ridge National Laboratory (ORNL), we show that the LLM rankings correlate strongly with the human rankings (Spearman $\rho\simeq 0.2-0.8$, improving to $\geq 0.5$ after 10\% outlier removal). Moreover, LLM performance is no worse than that of human reviewers in identifying proposals with high publication potential, while costing over two orders of magnitude less. Beyond ranking, LLMs enable advanced analyses that are challenging for humans, such as quantitative assessment of proposal similarity via embedding models, which provides information crucial for review committees. 

---
# Challenges of Evaluating LLM Safety for User Welfare 

**Authors**: Manon Kempermann, Sai Suresh Macharla Vasu, Mahalakshmi Raveenthiran, Theo Farrell, Ingmar Weber  

**Link**: [PDF](https://arxiv.org/pdf/2512.10687)  

**Abstract**: Safety evaluations of large language models (LLMs) typically focus on universal risks like dangerous capabilities or undesirable propensities. However, millions use LLMs for personal advice on high-stakes topics like finance and health, where harms are context-dependent rather than universal. While frameworks like the OECD's AI classification recognize the need to assess individual risks, user-welfare safety evaluations remain underdeveloped. We argue that developing such evaluations is non-trivial due to fundamental questions about accounting for user context in evaluation design. In this exploratory study, we evaluated advice on finance and health from GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across user profiles of varying vulnerability. First, we demonstrate that evaluators must have access to rich user context: identical LLM responses were rated significantly safer by context-blind evaluators than by those aware of user circumstances, with safety scores for high-vulnerability users dropping from safe (5/7) to somewhat unsafe (3/7). One might assume this gap could be addressed by creating realistic user prompts containing key contextual information. However, our second study challenges this: we rerun the evaluation on prompts containing context users report they would disclose, finding no significant improvement. Our work establishes that effective user-welfare safety evaluation requires evaluators to assess responses against diverse user profiles, as realistic user context disclosure alone proves insufficient, particularly for vulnerable populations. By demonstrating a methodology for context-aware evaluation, this study provides both a starting point for such assessments and foundational evidence that evaluating individual welfare demands approaches distinct from existing universal-risk frameworks. We publish our code and dataset to aid future developments. 

---
# Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning 

**Authors**: Haiteng Zhao, Junhao Shen, Yiming Zhang, Songyang Gao, Kuikun Liu, Tianyou Ma, Fan Zheng, Dahua Lin, Wenwei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.10534)  

**Abstract**: Large language model (LLM) agents exhibit strong mathematical problem-solving abilities and can even solve International Mathematical Olympiad (IMO) level problems with the assistance of formal proof systems. However, due to weak heuristics for auxiliary constructions, AI for geometry problem solving remains dominated by expert models such as AlphaGeometry 2, which rely heavily on large-scale data synthesis and search for both training and evaluation. In this work, we make the first attempt to build a medalist-level LLM agent for geometry and present InternGeometry. InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine's feedback to guide subsequent proposals. A dynamic memory mechanism enables InternGeometry to conduct more than two hundred interactions with the symbolic engine per problem. To further accelerate learning, we introduce Complexity-Boosting Reinforcement Learning (CBRL), which gradually increases the complexity of synthesized problems across training stages. Built on InternThinker-32B, InternGeometry solves 44 of 50 IMO geometry problems (2000-2024), exceeding the average gold medalist score (40.9), using only 13K training examples, just 0.004% of the data used by AlphaGeometry 2, demonstrating the potential of LLM agents on expert-level geometry tasks. InternGeometry can also propose novel auxiliary constructions for IMO problems that do not appear in human solutions. We will release the model, data, and symbolic engine to support future research. 

---
# Phythesis: Physics-Guided Evolutionary Scene Synthesis for Energy-Efficient Data Center Design via LLMs 

**Authors**: Minghao LI, Ruihang Wang, Rui Tan, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.10611)  

**Abstract**: Data center (DC) infrastructure serves as the backbone to support the escalating demand for computing capacity. Traditional design methodologies that blend human expertise with specialized simulation tools scale poorly with the increasing system complexity. Recent studies adopt generative artificial intelligence to design plausible human-centric indoor layouts. However, they do not consider the underlying physics, making them unsuitable for the DC design that sets quantifiable operational objectives and strict physical constraints. To bridge the gap, we propose Phythesis, a novel framework that synergizes large language models (LLMs) and physics-guided evolutionary optimization to automate simulation-ready (SimReady) scene synthesis for energy-efficient DC design. Phythesis employs an iterative bi-level optimization architecture, where (i) the LLM-driven optimization level generates physically plausible three-dimensional layouts and self-criticizes them to refine the scene topology, and (ii) the physics-informed optimization level identifies the optimal asset parameters and selects the best asset combination. Experiments on three generation scales show that Phythesis achieves 57.3% generation success rate increase and 11.5% power usage effectiveness (PUE) improvement, compared with the vanilla LLM-based solution. 

---
# EpiPlanAgent: Agentic Automated Epidemic Response Planning 

**Authors**: Kangkun Mao, Fang Xu, Jinru Ding, Yidong Jiang, Yujun Yao, Yirong Chen, Junming Liu, Xiaoqin Wu, Qian Wu, Xiaoyan Huang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.10313)  

**Abstract**: Epidemic response planning is essential yet traditionally reliant on labor-intensive manual methods. This study aimed to design and evaluate EpiPlanAgent, an agent-based system using large language models (LLMs) to automate the generation and validation of digital emergency response plans. The multi-agent framework integrated task decomposition, knowledge grounding, and simulation modules. Public health professionals tested the system using real-world outbreak scenarios in a controlled evaluation. Results demonstrated that EpiPlanAgent significantly improved the completeness and guideline alignment of plans while drastically reducing development time compared to manual workflows. Expert evaluation confirmed high consistency between AI-generated and human-authored content. User feedback indicated strong perceived utility. In conclusion, EpiPlanAgent provides an effective, scalable solution for intelligent epidemic response planning, demonstrating the potential of agentic AI to transform public health preparedness. 

---
# AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 

**Authors**: Mesafint Fanuel, Mahmoud Nabil Mahmoud, Crystal Cook Marshal, Vishal Lakhotia, Biswanath Dari, Kaushik Roy, Shaohu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10114)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in democratizing access to information. However, in the domain of agriculture, general-purpose models frequently suffer from contextual hallucination, which provides non-factual advice or answers are scientifically sound in one region but disastrous in another due to variations in soil, climate, and local regulations. We introduce AgriRegion, a Retrieval-Augmented Generation (RAG) framework designed specifically for high-fidelity, region-aware agricultural advisory. Unlike standard RAG approaches that rely solely on semantic similarity, AgriRegion incorporates a geospatial metadata injection layer and a region-prioritized re-ranking mechanism. By restricting the knowledge base to verified local agricultural extension services and enforcing geo-spatial constraints during retrieval, AgriRegion ensures that the advice regarding planting schedules, pest control, and fertilization is locally accurate. We create a novel benchmark dataset, AgriRegion-Eval, which comprises 160 domain-specific questions across 12 agricultural subfields. Experiments demonstrate that AgriRegion reduces hallucinations by 10-20% compared to state-of-the-art LLMs systems and significantly improves trust scores according to a comprehensive evaluation. 

---
# CP-Env: Evaluating Large Language Models on Clinical Pathways in a Controllable Hospital Environment 

**Authors**: Yakun Zhu, Zhongzhen Huang, Qianhan Feng, Linjie Mu, Yannian Gu, Shaoting Zhang, Qi Dou, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10206)  

**Abstract**: Medical care follows complex clinical pathways that extend beyond isolated physician-patient encounters, emphasizing decision-making and transitions between different stages. Current benchmarks focusing on static exams or isolated dialogues inadequately evaluate large language models (LLMs) in dynamic clinical scenarios. We introduce CP-Env, a controllable agentic hospital environment designed to evaluate LLMs across end-to-end clinical pathways. CP-Env simulates a hospital ecosystem with patient and physician agents, constructing scenarios ranging from triage and specialist consultation to diagnostic testing and multidisciplinary team meetings for agent interaction. Following real hospital adaptive flow of healthcare, it enables branching, long-horizon task execution. We propose a three-tiered evaluation framework encompassing Clinical Efficacy, Process Competency, and Professional Ethics. Results reveal that most models struggle with pathway complexity, exhibiting hallucinations and losing critical diagnostic details. Interestingly, excessive reasoning steps can sometimes prove counterproductive, while top models tend to exhibit reduced tool dependency through internalized knowledge. CP-Env advances medical AI agents development through comprehensive end-to-end clinical evaluation. We provide the benchmark and evaluation tools for further research and development at this https URL. 

---
# Linear socio-demographic representations emerge in Large Language Models from indirect cues 

**Authors**: Paul Bouchaud, Pedro Ramaciotti  

**Link**: [PDF](https://arxiv.org/pdf/2512.10065)  

**Abstract**: We investigate how LLMs encode sociodemographic attributes of human conversational partners inferred from indirect cues such as names and occupations. We show that LLMs develop linear representations of user demographics within activation space, wherein stereotypically associated attributes are encoded along interpretable geometric directions. We first probe residual streams across layers of four open transformer-based LLMs (Magistral 24B, Qwen3 14B, GPT-OSS 20B, OLMo2-1B) prompted with explicit demographic disclosure. We show that the same probes predict demographics from implicit cues: names activate census-aligned gender and race representations, while occupations trigger representations correlated with real-world workforce statistics. These linear representations allow us to explain demographic inferences implicitly formed by LLMs during conversation. We demonstrate that these implicit demographic representations actively shape downstream behavior, such as career recommendations. Our study further highlights that models that pass bias benchmark tests may still harbor and leverage implicit biases, with implications for fairness when applied at scale. 

---
# Reverse Thinking Enhances Missing Information Detection in Large Language Models 

**Authors**: Yuxin Liu, Chaojie Gu, Yihang Zhang, Bin Qian, Shibo He  

**Link**: [PDF](https://arxiv.org/pdf/2512.10273)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various reasoning tasks, yet they often struggle with problems involving missing information, exhibiting issues such as incomplete responses, factual errors, and hallucinations. While forward reasoning approaches like Chain-of-Thought (CoT) and Tree-of-Thought (ToT) have shown success in structured problem-solving, they frequently fail to systematically identify and recover omitted information. In this paper, we explore the potential of reverse thinking methodologies to enhance LLMs' performance on missing information detection tasks. Drawing inspiration from recent work on backward reasoning, we propose a novel framework that guides LLMs through reverse thinking to identify necessary conditions and pinpoint missing elements. Our approach transforms the challenging task of missing information identification into a more manageable backward reasoning problem, significantly improving model accuracy. Experimental results demonstrate that our reverse thinking approach achieves substantial performance gains compared to traditional forward reasoning methods, providing a promising direction for enhancing LLMs' logical completeness and reasoning robustness. 

---
# LLM-Empowered Representation Learning for Emerging Item Recommendation 

**Authors**: Ziying Zhang, Quanming Yao, Yaqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10370)  

**Abstract**: In this work, we tackle the challenge of recommending emerging items, whose interactions gradually accumulate over time. Existing methods often overlook this dynamic process, typically assuming that emerging items have few or even no historical interactions. Such an assumption oversimplifies the problem, as a good model must preserve the uniqueness of emerging items while leveraging their shared patterns with established ones. To address this challenge, we propose EmerFlow, a novel LLM-empowered representation learning framework that generates distinctive embeddings for emerging items. It first enriches the raw features of emerging items through LLM reasoning, then aligns these representations with the embedding space of the existing recommendation model. Finally, new interactions are incorporated through meta-learning to refine the embeddings. This enables EmerFlow to learn expressive embeddings for emerging items from only limited interactions. Extensive experiments across diverse domains, including movies and pharmaceuticals, show that EmerFlow consistently outperforms existing methods. 

---
# Exploring LLMs for Scientific Information Extraction Using The SciEx Framework 

**Authors**: Sha Li, Ayush Sadekar, Nathan Self, Yiqi Su, Lars Andersland, Mira Chaplin, Annabel Zhang, Hyoju Yang, James B Henderson, Krista Wigginton, Linsey Marr, T.M. Murali, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2512.10004)  

**Abstract**: Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines. 

---
# Echo-CoPilot: A Multi-View, Multi-Task Agent for Echocardiography Interpretation and Reporting 

**Authors**: Moein Heidari, Mohammad Amin Roohi, Armin Khosravi, Ilker Hacihaliloglu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09944)  

**Abstract**: Echocardiography is central to contemporary cardiovascular care, but full-study interpretation remains a cognitively demanding, multi-view task that is still performed manually. While recent foundation models for echocardiography can achieve strong performance on individual perceptual subtasks such as view classification, segmentation, or disease prediction, they typically operate in isolation and do not provide a unified, clinically coherent assessment. In this work, we introduce Echo-CoPilot, a multi-view, multi-task agent that uses a large language model to orchestrate a suite of specialized echocardiography tools. Within a ReAct-style loop, the agent decomposes clinician queries, invokes tools for view recognition, cardiac structure segmentation, measurement and disease prediction, and report synthesis, and integrates their outputs into guideline-aware answers and narrative summaries. We evaluate Echo-CoPilot on the public MIMIC-EchoQA benchmark, where it achieves an accuracy of 50.8\%, outperforming both general-purpose and biomedical video vision-language models. Qualitative analyses further show that the agent leverages quantitative measurements and physiologic context to resolve challenging cases near clinical decision thresholds, such as borderline left ventricular hypertrophy or pericardial effusion severity. The code will be released upon acceptance of the paper. 

---
# Exploring Health Misinformation Detection with Multi-Agent Debate 

**Authors**: Chih-Han Chen, Chen-Han Tsai, Yu-Shao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.09935)  

**Abstract**: Fact-checking health-related claims has become increasingly critical as misinformation proliferates online. Effective verification requires both the retrieval of high-quality evidence and rigorous reasoning processes. In this paper, we propose a two-stage framework for health misinformation detection: Agreement Score Prediction followed by Multi-Agent Debate. In the first stage, we employ large language models (LLMs) to independently evaluate retrieved articles and compute an aggregated agreement score that reflects the overall evidence stance. When this score indicates insufficient consensus-falling below a predefined threshold-the system proceeds to a second stage. Multiple agents engage in structured debate to synthesize conflicting evidence and generate well-reasoned verdicts with explicit justifications. Experimental results demonstrate that our two-stage approach achieves superior performance compared to baseline methods, highlighting the value of combining automated scoring with collaborative reasoning for complex verification tasks. 

---
# Parallel Decoder Transformer: Model-Internal Parallel Decoding with Speculative Invariance via Note Conditioning 

**Authors**: Logan Robbins  

**Link**: [PDF](https://arxiv.org/pdf/2512.10054)  

**Abstract**: Autoregressive decoding in Large Language Models (LLMs) is inherently sequential, creating a latency bottleneck that scales linearly with output length. While ``Decomposition-and-Fill'' methods like Skeleton-of-Thought attempt to parallelize generation via external orchestration, they suffer from \textit{coherence drift} due to the lack of cross-stream communication. In this work, we introduce the \textbf{Parallel Decoder Transformer (PDT)}, a parameter-efficient architecture that embeds coordination primitives directly into the inference process of a frozen pre-trained model.
Instead of retraining the base model, PDT injects lightweight \textit{Speculative Note Conditioning (SNC)} adapters that allow parallel decoding streams to synchronize via a shared, dynamic latent space. We formulate coordination as a \textit{speculative consensus} problem, where sibling streams broadcast semantic ``notes'' to a global bus, gated by a learned verification head. We validate our approach on a 50,000-step curriculum using a frozen 20B-parameter backbone. Our results demonstrate that PDT achieves effective self-correction, reaching \textbf{77.8\% precision} in coverage prediction and recovering approximate serial semantics without modifying the trunk weights. This establishes PDT as a scalable, efficient alternative to full model fine-tuning for structured parallel generation. 

---
# LabelFusion: Learning to Fuse LLMs and Transformer Classifiers for Robust Text Classification 

**Authors**: Michael Schlee, Christoph Weisser, Timo Kivimäki, Melchizedek Mashiku, Benjamin Saefken  

**Link**: [PDF](https://arxiv.org/pdf/2512.10793)  

**Abstract**: LabelFusion is a fusion ensemble for text classification that learns to combine a traditional transformer-based classifier (e.g., RoBERTa) with one or more Large Language Models (LLMs such as OpenAI GPT, Google Gemini, or DeepSeek) to deliver accurate and cost-aware predictions across multi-class and multi-label tasks. The package provides a simple high-level interface (AutoFusionClassifier) that trains the full pipeline end-to-end with minimal configuration, and a flexible API for advanced users. Under the hood, LabelFusion integrates vector signals from both sources by concatenating the ML backbone's embeddings with the LLM-derived per-class scores -- obtained through structured prompt-engineering strategies -- and feeds this joint representation into a compact multi-layer perceptron (FusionMLP) that produces the final prediction. This learned fusion approach captures complementary strengths of LLM reasoning and traditional transformer-based classifiers, yielding robust performance across domains -- achieving 92.4% accuracy on AG News and 92.3% on 10-class Reuters 21578 topic classification -- while enabling practical trade-offs between accuracy, latency, and cost. 

---
# Metaphor-based Jailbreaking Attacks on Text-to-Image Models 

**Authors**: Chenyu Zhang, Yiwen Ma, Lanjun Wang, Wenhui Li, Yi Tu, An-An Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.10766)  

**Abstract**: Text-to-image~(T2I) models commonly incorporate defense mechanisms to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attacks have shown that adversarial prompts can effectively bypass these mechanisms and induce T2I models to produce sensitive content, revealing critical safety vulnerabilities. However, existing attack methods implicitly assume that the attacker knows the type of deployed defenses, which limits their effectiveness against unknown or diverse defense mechanisms. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to effectively and efficiently attack diverse defense mechanisms without prior knowledge of their type by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Extensive experiments on T2I models with various external and internal defense mechanisms demonstrate that MJA outperforms six baseline methods, achieving stronger attack performance while using fewer queries. Code is available in this https URL. 

---
# Developing and Evaluating a Large Language Model-Based Automated Feedback System Grounded in Evidence-Centered Design for Supporting Physics Problem Solving 

**Authors**: Holger Maus, Paul Tschisgale, Fabian Kieser, Stefan Petersen, Peter Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2512.10785)  

**Abstract**: Generative AI offers new opportunities for individualized and adaptive learning, particularly through large language model (LLM)-based feedback systems. While LLMs can produce effective feedback for relatively straightforward conceptual tasks, delivering high-quality feedback for tasks that require advanced domain expertise, such as physics problem solving, remains a substantial challenge. This study presents the design of an LLM-based feedback system for physics problem solving grounded in evidence-centered design (ECD) and evaluates its performance within the German Physics Olympiad. Participants assessed the usefulness and accuracy of the generated feedback, which was generally perceived as useful and highly accurate. However, an in-depth analysis revealed that the feedback contained factual errors in 20% of cases; errors that often went unnoticed by the students. We discuss the risks associated with uncritical reliance on LLM-based feedback systems and outline potential directions for generating more adaptive and reliable LLM-based feedback in the future. 

---
# Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving 

**Authors**: Songyang Gao, Yuzhe Gu, Zijian Wu, Lingkai Kong, Wenwei Zhang, Zhongrui Cai, Fan Zheng, Tianyou Ma, Junhao Shen, Haiteng Zhao, Duanyang Zhang, Huilun Zhang, Kuikun Liu, Chengqi Lyu, Yanhui Duan, Chiyu Chen, Ningsheng Ma, Jianfei Gao, Han Lyu, Dahua Lin, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.10739)  

**Abstract**: Large language models (LLMs) have achieved significant progress in solving complex reasoning tasks by Reinforcement Learning with Verifiable Rewards (RLVR). This advancement is also inseparable from the oversight automated by reliable verifiers. However, current outcome-based verifiers (OVs) are unable to inspect the unreliable intermediate steps in the long reasoning chains of thought (CoTs). Meanwhile, current process-based verifiers (PVs) have difficulties in reliably detecting errors in the complex long CoTs, limited by the scarcity of high-quality annotations due to the prohibitive costs of human annotations. Therefore, we propose the \textbf{O}utcome-based \textbf{P}rocess \textbf{V}erifier (OPV), which verifies the rationale process of summarized outcomes from long CoTs to achieve both accurate and efficient verification and enable large-scale annotation. To empower the proposed verifier, we adopt an iterative active learning framework with expert annotations to progressively improve the verification capability of OPV with fewer annotation costs. Specifically, in each iteration, the most uncertain cases of the current best OPV are annotated and then subsequently used to train a new OPV through Rejection Fine-Tuning (RFT) and RLVR for the next round. Extensive experiments demonstrate OPV's superior performance and broad applicability. It achieves new state-of-the-art results on our held-out \textsc{\thisbench}, outperforming much larger open-source models such as Qwen3-Max-Preview with an F1 score of 83.1 compared to 76.3. Furthermore, OPV effectively detects false positives within synthetic dataset, closely align with expert assessment. When collaborating with policy models, OPV consistently yields performance gains, e.g., raising the accuracy of DeepSeek-R1-Distill-Qwen-32B from 55.2\% to 73.3\% on AIME2025 as the compute budget scales. 

---
# Sliding Window Attention Adaptation 

**Authors**: Yijiong Yu, Jiale Liu, Qingyun Wu, Huazheng Wang, Ji Pei  

**Link**: [PDF](https://arxiv.org/pdf/2512.10411)  

**Abstract**: The self-attention mechanism in Transformer-based Large Language Models (LLMs) scales quadratically with input length, making long-context inference expensive. Sliding window attention (SWA) reduces this cost to linear complexity, but naively enabling complete SWA at inference-time for models pretrained with full attention (FA) causes severe long-context performance degradation due to training-inference mismatch. This makes us wonder: Can FA-pretrained LLMs be well adapted to SWA without pretraining? We investigate this by proposing Sliding Window Attention Adaptation (SWAA), a set of practical recipes that combine five methods for better adaptation: (1) applying SWA only during prefilling; (2) preserving "sink" tokens; (3) interleaving FA/SWA layers; (4) chain-of-thought (CoT); and (5) fine-tuning. Our experiments show that SWA adaptation is feasible while non-trivial: no single method suffices, yet specific synergistic combinations effectively recover the original long-context performance. We further analyze the performance-efficiency trade-offs of different SWAA configurations and provide recommended recipes for diverse scenarios. Our code is available at this https URL 

---
# Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers 

**Authors**: Youmin Ko, Sungjong Seo, Hyunjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10422)  

**Abstract**: Since large language models (LLMs) have a tendency to generate factually inaccurate output, retrieval-augmented generation (RAG) has gained significant attention as a key means to mitigate this downside of harnessing only LLMs. However, existing RAG methods for simple and multi-hop question answering (QA) are still prone to incorrect retrievals and hallucinations. To address these limitations, we propose CoopRAG, a novel RAG framework for the question answering task in which a retriever and an LLM work cooperatively with each other by exchanging informative knowledge, and the earlier and later layers of the retriever model work cooperatively with each other to accurately rank the retrieved documents relevant to a given query. In this framework, we (i) unroll a question into sub-questions and a reasoning chain in which uncertain positions are masked, (ii) retrieve the documents relevant to the question augmented with the sub-questions and the reasoning chain, (iii) rerank the documents by contrasting layers of the retriever, and (iv) reconstruct the reasoning chain by filling the masked positions via the LLM. Our experiments demonstrate that CoopRAG consistently outperforms state-of-the-art QA methods on three multi-hop QA datasets as well as a simple QA dataset in terms of both the retrieval and QA performances. Our code is available.\footnote{this https URL} 

---
# Textual Data Bias Detection and Mitigation - An Extensible Pipeline with Experimental Evaluation 

**Authors**: Rebekka Görge, Sujan Sai Gannamaneni, Tabea Naeven, Hammam Abdelwahab, Héctor Allende-Cid, Armin B. Cremers, Lennard Helmer, Michael Mock, Anna Schmitz, Songkai Xue, Elif Yildirir, Maximilian Poretschkin, Stefan Wrobel  

**Link**: [PDF](https://arxiv.org/pdf/2512.10734)  

**Abstract**: Textual data used to train large language models (LLMs) exhibits multifaceted bias manifestations encompassing harmful language and skewed demographic distributions. Regulations such as the European AI Act require identifying and mitigating biases against protected groups in data, with the ultimate goal of preventing unfair model outputs. However, practical guidance and operationalization are lacking. We propose a comprehensive data bias detection and mitigation pipeline comprising four components that address two data bias types, namely representation bias and (explicit) stereotypes for a configurable sensitive attribute. First, we leverage LLM-generated word lists created based on quality criteria to detect relevant group labels. Second, representation bias is quantified using the Demographic Representation Score. Third, we detect and mitigate stereotypes using sociolinguistically informed filtering. Finally, we compensate representation bias through Grammar- and Context-Aware Counterfactual Data Augmentation. We conduct a two-fold evaluation using the examples of gender, religion and age. First, the effectiveness of each individual component on data debiasing is evaluated through human validation and baseline comparison. The findings demonstrate that we successfully reduce representation bias and (explicit) stereotypes in a text dataset. Second, the effect of data debiasing on model bias reduction is evaluated by bias benchmarking of several models (0.6B-8B parameters), fine-tuned on the debiased text dataset. This evaluation reveals that LLMs fine-tuned on debiased data do not consistently show improved performance on bias benchmarks, exposing critical gaps in current evaluation methodologies and highlighting the need for targeted data manipulation to address manifested model bias. 

---
# LLM-Auction: Generative Auction towards LLM-Native Advertising 

**Authors**: Chujie Zhao, Qun Hu, Shiping Song, Dagui Chen, Han Zhu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.10551)  

**Abstract**: The rapid advancement of large language models (LLMs) necessitates novel monetization strategies, among which LLM-native advertising has emerged as a promising paradigm by naturally integrating advertisement within LLM-generated responses. However, this paradigm fundamentally shifts the auction object from discrete ad slots to the distribution over LLM outputs, posing new challenges for designing auction mechanisms. Existing mechanisms for LLM-native advertising adopt frameworks that decouple auction and generation, which either ignore externalities or require multiple LLM inferences for ad allocation, rendering them impractical for industrial scenarios. To address these challenges, we propose LLM-Auction, which to the best of our knowledge is the first learning-based generative auction mechanism that integrates auction and LLM generation for LLM-native advertising. By formulating the allocation optimization as a preference alignment problem between LLM outputs and the mechanism's objective which reflects both advertisers' expected value and user experience, we introduce Iterative Reward-Preference Optimization (IRPO) algorithm that alternately optimizes the reward model and the LLM. This approach enables the LLM to inherently model allocation externalities without any extra inference cost. We further identify the allocation monotonicity and continuity of LLM-Auction, which allows us to prove that a simple first-price payment rule exhibits favorable incentive properties. Additionally, we design an LLM-as-a-judge simulation environment to facilitate large-scale data construction and enable comprehensive quantitative evaluation of the mechanism's performance. Extensive quantitative and qualitative experiments demonstrate that LLM-Auction significantly outperforms existing baselines in allocation efficiency, while achieving the desired mechanism properties. 

---
# Dynamics of Agentic Loops in Large Language Models: A Geometric Theory of Trajectories 

**Authors**: Nicolas Tacheny  

**Link**: [PDF](https://arxiv.org/pdf/2512.10350)  

**Abstract**: Agentic systems built on large language models operate through recursive feedback loops, where each output becomes the next input. Yet the geometric behavior of these agentic loops (whether they converge, diverge, or exhibit more complex dynamics) remains poorly understood. This paper introduces a geometric framework for analyzing agentic trajectories in semantic embedding space, treating iterative transformations as discrete dynamical systems.
We distinguish the artifact space, where linguistic transformations occur, from the embedding space, where geometric measurements are performed. Because cosine similarity is biased by embedding anisotropy, we introduce an isotonic calibration that eliminates systematic bias and aligns similarities with human semantic judgments while preserving high local stability. This enables rigorous measurement of trajectories, clusters and attractors.
Through controlled experiments on singular agentic loops, we identify two fundamental regimes. A contractive rewriting loop converges toward a stable attractor with decreasing dispersion, while an exploratory summarize and negate loop produces unbounded divergence with no cluster formation. These regimes display qualitatively distinct geometric signatures of contraction and expansion.
Our results show that prompt design directly governs the dynamical regime of an agentic loop, enabling systematic control of convergence, divergence and trajectory structure in iterative LLM transformations. 

---
# Visual Funnel: Resolving Contextual Blindness in Multimodal Large Language Models 

**Authors**: Woojun Jung, Jaehoon Go, Mingyu Jeon, Sunjae Yoon, Junyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10362)  

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate impressive reasoning capabilities, but often fail to perceive fine-grained visual details, limiting their applicability in precision-demanding tasks. While methods that crop salient regions of an image offer a partial solution, we identify a critical limitation they introduce: "Contextual Blindness". This failure occurs due to structural disconnect between high-fidelity details (from the crop) and the broader global context (from the original image), even when all necessary visual information is present. We argue that this limitation stems not from a lack of information 'Quantity', but from a lack of 'Structural Diversity' in the model's input. To resolve this, we propose Visual Funnel, a training-free, two-step approach. Visual Funnel first performs Contextual Anchoring to identify the region of interest in a single forward pass. It then constructs an Entropy-Scaled Portfolio that preserves the hierarchical context - ranging from focal detail to broader surroundings - by dynamically determining crop sizes based on attention entropy and refining crop centers. Through extensive experiments, we demonstrate that Visual Funnel significantly outperforms naive single-crop and unstructured multi-crop baselines. Our results further validate that simply adding more unstructured crops provides limited or even detrimental benefits, confirming that the hierarchical structure of our portfolio is key to resolving Contextual Blindness. 

---
# GPG: Generalized Policy Gradient Theorem for Transformer-based Policies 

**Authors**: Hangyu Mao, Guangting Dong, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2512.10365)  

**Abstract**: We present the Generalized Policy Gradient (GPG) Theorem, specifically designed for Transformer-based policies. Notably, we demonstrate that both standard Policy Gradient Theorem and GRPO emerge as special cases within our GPG framework. Furthermore, we explore its practical applications in training Large Language Models (LLMs), offering new insights into efficient policy optimization. 

---
# InFerActive: Towards Scalable Human Evaluation of Large Language Models through Interactive Inference 

**Authors**: Junhyeong Hwangbo, Soohyun Lee, Minsoo Cheong, Hyeon Jeon, Jinwook Seo  

**Link**: [PDF](https://arxiv.org/pdf/2512.10234)  

**Abstract**: Human evaluation remains the gold standard for evaluating outputs of Large Language Models (LLMs). The current evaluation paradigm reviews numerous individual responses, leading to significant scalability challenges. LLM outputs can be more efficiently represented as a tree structure, reflecting their autoregressive generation process and stochastic token selection. However, conventional tree visualization cannot scale to the exponentially large trees generated by modern sampling methods of LLMs. To address this problem, we present InFerActive, an interactive inference system for scalable human evaluation. InFerActive enables on-demand exploration through probability-based filtering and evaluation features, while bridging the semantic gap between computational tokens and human-readable text through adaptive visualization techniques. Through a technical evaluation and user study (N=12), we demonstrate that InFerActive significantly improves evaluation efficiency and enables more comprehensive assessment of model behavior. We further conduct expert case studies that demonstrate InFerActive's practical applicability and potential for transforming LLM evaluation workflows. 

---
# Enhancing Large Language Models for End-to-End Circuit Analysis Problem Solving 

**Authors**: Liangliang Chen, Weiyu Sun, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10159)  

**Abstract**: Large language models (LLMs) have shown strong performance in data-rich domains such as programming, but their reliability in engineering tasks remains limited. Circuit analysis -- requiring multimodal understanding and precise mathematical reasoning -- highlights these challenges. Although Gemini 2.5 Pro improves diagram interpretation and analog-circuit reasoning, it still struggles to consistently produce correct solutions when given both text and circuit diagrams. At the same time, engineering education needs scalable AI tools capable of generating accurate solutions for tasks such as automated homework feedback and question-answering. This paper presents an enhanced, end-to-end circuit problem solver built on Gemini 2.5 Pro. We first benchmark Gemini on a representative set of undergraduate circuit problems and identify two major failure modes: 1) circuit-recognition hallucinations, particularly incorrect source polarity detection, and 2) reasoning-process hallucinations, such as incorrect current directions. To address recognition errors, we integrate a fine-tuned YOLO detector and OpenCV processing to isolate voltage and current sources, enabling Gemini to re-identify source polarities from cropped images with near-perfect accuracy. To reduce reasoning errors, we introduce an ngspice-based verification loop in which Gemini generates a .cir file, ngspice simulates the circuit, and discrepancies trigger iterative regeneration with optional human-in-the-loop review. Across 83 problems, the proposed pipeline achieves a 97.59% success rate (81 correct solutions), substantially outperforming Gemini 2.5 Pro's original 79.52% accuracy. This system extends LLM capabilities for multimodal engineering problem-solving and supports the creation of high-quality educational datasets and AI-powered instructional tools. 

---
# Unforgotten Safety: Preserving Safety Alignment of Large Language Models with Continual Learning 

**Authors**: Lama Alssum, Hani Itani, Hasan Abed Al Kader Hammoud, Philip Torr, Adel Bibi, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2512.10150)  

**Abstract**: The safety alignment of large language models (LLMs) is becoming increasingly important with their democratization. In this paper, we study the safety degradation that comes with adapting LLMs to new tasks. We attribute this safety compromise to catastrophic forgetting and frame the problem of preserving safety when fine-tuning as a continual learning (CL) problem. We consider the fine-tuning-as-a-service setup where the user uploads their data to a service provider to get a customized model that excels on the user's selected task. We adapt several CL approaches from the literature and systematically evaluate their ability to mitigate safety degradation. These include regularization-based, memory-based, and model merging approaches. We consider two scenarios, (1) benign user data and (2) poisoned user data. Our results demonstrate that CL approaches consistently achieve lower attack success rates than standard fine-tuning. Among these, DER outperforms both other CL methods and existing safety-preserving baselines while maintaining task utility. These findings generalize across three downstream tasks (GSM8K, SST2, Code) and three model families (LLaMA2-7B, Mistral-7B, Gemma-2B), establishing CL as a practical solution to preserve safety. 

---
# What Kind of Reasoning (if any) is an LLM actually doing? On the Stochastic Nature and Abductive Appearance of Large Language Models 

**Authors**: Luciano Floridi, Jessica Morley, Claudio Novelli, David Watson  

**Link**: [PDF](https://arxiv.org/pdf/2512.10080)  

**Abstract**: This article looks at how reasoning works in current Large Language Models (LLMs) that function using the token-completion method. It examines their stochastic nature and their similarity to human abductive reasoning. The argument is that these LLMs create text based on learned patterns rather than performing actual abductive reasoning. When their output seems abductive, this is largely because they are trained on human-generated texts that include reasoning structures. Examples are used to show how LLMs can produce plausible ideas, mimic commonsense reasoning, and give explanatory answers without being grounded in truth, semantics, verification, or understanding, and without performing any real abductive reasoning. This dual nature, where the models have a stochastic base but appear abductive in use, has important consequences for how LLMs are evaluated and applied. They can assist with generating ideas and supporting human thinking, but their outputs must be critically assessed because they cannot identify truth or verify their explanations. The article concludes by addressing five objections to these points, noting some limitations in the analysis, and offering an overall evaluation. 

---
# Detailed balance in large language model-driven agents 

**Authors**: Zhuo-Yang Song, Qing-Hong Cao, Ming-xing Luo, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.10047)  

**Abstract**: Large language model (LLM)-driven agents are emerging as a powerful new paradigm for solving complex problems. Despite the empirical success of these practices, a theoretical framework to understand and unify their macroscopic dynamics remains lacking. This Letter proposes a method based on the least action principle to estimate the underlying generative directionality of LLMs embedded within agents. By experimentally measuring the transition probabilities between LLM-generated states, we statistically discover a detailed balance in LLM-generated transitions, indicating that LLM generation may not be achieved by generally learning rule sets and strategies, but rather by implicitly learning a class of underlying potential functions that may transcend different LLM architectures and prompt templates. To our knowledge, this is the first discovery of a macroscopic physical law in LLM generative dynamics that does not depend on specific model details. This work is an attempt to establish a macroscopic dynamics theory of complex AI systems, aiming to elevate the study of AI agents from a collection of engineering practices to a science built on effective measurements that are predictable and quantifiable. 

---
# Intelligently Weighting Multiple Reference Models for Direct Preference Optimization of LLMs 

**Authors**: Skyler Wu, Aymen Echarghaoui  

**Link**: [PDF](https://arxiv.org/pdf/2512.10040)  

**Abstract**: Fine-tuning is integral for aligning large language models (LLMs) with human preferences. Multiple-Reference Preference Optimization (MRPO) builds on Direct Preference Optimization (DPO) by fine-tuning LLMs on preference datasets while regularizing the policy towards a mixture of reference models to leverage their collective desirable properties. However, current methods for setting the reference weights are ad-hoc and statistically unsound, leading to unreliable performance. To address this, we introduce four new weighting strategies: two offline methods that leverage held-out validation signal; one online method that uses a sliding-window estimator to reduce overfitting; and an online method that treats reference weighting as a $K$-armed bandit via Thompson Sampling. Experiments using Qwen2.5-0.5B as the policy model and seven reference models from the Llama, Mistral, Qwen, Yi, and Phi families (0.5B-14B each) show that all 4 of our strategies outperform the current MRPO weighting methods on UltraFeedback and SafeRLHF in preference accuracy. More thought-provokingly, however, we find that single-reference DPO, using any of 6 out of 7 references, consistently outperforms all tested multiple-reference approaches -- calling into question the practical appeal of multiple-reference approaches. 

---
# Computational emotion analysis with multimodal LLMs: Current evidence on an emerging methodological opportunity 

**Authors**: Hauke Licht  

**Link**: [PDF](https://arxiv.org/pdf/2512.10882)  

**Abstract**: Emotions are central to politics and analyzing their role in political communication has a long tradition. As research increasingly leverages audio-visual materials to analyze the display of emotions, the emergence of multimodal generative AI promises great advances. However, we lack evidence about the effectiveness of multimodal AI in emotion analysis. This paper addresses this gap by evaluating current multimodal large language models (mLLMs) in video-based analysis of emotional arousal in two complementary data sets of human-labeled video recordings. I find that under ideal circumstances, mLLMs' emotional arousal ratings are highly reliable and show little to know indication of demographic bias. However, in recordings of speakers in real-world parliamentary debates, mLLMs' arousal ratings fail to deliver on this promise with potential negative consequences for downstream statistical inferences. This study therefore underscores the need for continued, thorough evaluation of emerging generative AI methods in political analysis and contributes a suitable replicable framework. 

---
# Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Roman Scripts in a Real World Setting 

**Authors**: Manurag Khullar, Utkarsh Desai, Poorva Malviya, Aman Dalmia, Zheyuan Ryan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.10780)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. In many such settings, speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely evaluates this orthographic variation using real-world data. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real-world dataset of user-generated queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with F1 scores trailing those of native scripts by 5-12 points. At our partner maternal health organization in India, this gap could cause nearly 2 million excess errors in triage. Crucially, this performance gap by scripts is not due to a failure in clinical reasoning. We demonstrate that LLMs often correctly infer the semantic intent of romanized queries. Nevertheless, their final classification outputs remain brittle in the presence of orthographic noise in romanized inputs. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably. 

---
# XDoGE: Multilingual Data Reweighting to Enhance Language Inclusivity in LLMs 

**Authors**: Iñaki Lacunza, José Javier Saiz, Alexander Shvets, Aitor Gonzalez-Agirre, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2512.10545)  

**Abstract**: Current large language models (LLMs) are trained on massive amounts of text data, primarily from a few dominant languages. Studies suggest that this over-reliance on high-resource languages, such as English, hampers LLM performance in mid- and low-resource languages. To mitigate this problem, we propose to (i) optimize the language distribution by training a small proxy model within a domain-reweighing DoGE algorithm that we extend to XDoGE for a multilingual setup, and (ii) rescale the data and train a full-size model with the established language weights either from scratch or within a continual pre-training phase (CPT). We target six languages possessing a variety of geographic and intra- and inter-language-family relations, namely, English and Spanish (high-resource), Portuguese and Catalan (mid-resource), Galician and Basque (low-resource). We experiment with Salamandra-2b, which is a promising model for these languages. We investigate the effects of substantial data repetition on minor languages and under-sampling on dominant languages using the IberoBench framework for quantitative evaluation. Finally, we release a new promising IberianLLM-7B-Instruct model centering on Iberian languages and English that we pretrained from scratch and further improved using CPT with the XDoGE weights. 

---
# T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground 

**Authors**: Dmitrii Stoianov, Danil Taranets, Olga Tsymboi, Ramil Latypov, Almaz Dautov, Vladislav Kruglikov, Nikita Surkov, German Abramov, Pavel Gein, Dmitry Abulkhanov, Mikhail Gashkov, Viktor Zelenkovskiy, Artem Batalov, Aleksandr Medvedev, Anatolii Potapov  

**Link**: [PDF](https://arxiv.org/pdf/2512.10430)  

**Abstract**: We introduce T-pro 2.0, an open-weight Russian LLM for hybrid reasoning and efficient inference. The model supports direct answering and reasoning-trace generation, using a Cyrillic-dense tokenizer and an adapted EAGLE speculative-decoding pipeline to reduce latency. To enable reproducible and extensible research, we release the model weights, the T-Wix 500k instruction corpus, the T-Math reasoning benchmark, and the EAGLE weights on Hugging Face. These resources allow users to study Russian-language reasoning and to extend or adapt both the model and the inference pipeline. A public web demo exposes reasoning and non-reasoning modes and illustrates the speedups achieved by our inference stack across domains. T-pro 2.0 thus serves as an accessible open system for building and evaluating efficient, practical Russian LLM applications. 

---
# Decoding Student Minds: Leveraging Conversational Agents for Psychological and Learning Analysis 

**Authors**: Nour El Houda Ben Chaabene, Hamza Hammami, Laid Kahloul  

**Link**: [PDF](https://arxiv.org/pdf/2512.10441)  

**Abstract**: This paper presents a psychologically-aware conversational agent designed to enhance both learning performance and emotional well-being in educational settings. The system combines Large Language Models (LLMs), a knowledge graph-enhanced BERT (KG-BERT), and a bidirectional Long Short-Term Memory (LSTM) with attention to classify students' cognitive and affective states in real time. Unlike prior chatbots limited to either tutoring or affective support, our approach leverages multimodal data-including textual semantics, prosodic speech features, and temporal behavioral trends-to infer engagement, stress, and conceptual understanding. A pilot study with university students demonstrated improved motivation, reduced stress, and moderate academic gains compared to baseline methods. These results underline the promise of integrating semantic reasoning, multimodal fusion, and temporal modeling to support adaptive, student-centered educational interventions. 

---
# AutoMedic: An Automated Evaluation Framework for Clinical Conversational Agents with Medical Dataset Grounding 

**Authors**: Gyutaek Oh, Sangjoon Park, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10195)  

**Abstract**: Evaluating large language models (LLMs) has recently emerged as a critical issue for safe and trustworthy application of LLMs in the medical domain. Although a variety of static medical question-answering (QA) benchmarks have been proposed, many aspects remain underexplored, such as the effectiveness of LLMs in generating responses in dynamic, interactive clinical multi-turn conversation situations and the identification of multi-faceted evaluation strategies beyond simple accuracy. However, formally evaluating a dynamic, interactive clinical situation is hindered by its vast combinatorial space of possible patient states and interaction trajectories, making it difficult to standardize and quantitatively measure such scenarios. Here, we introduce AutoMedic, a multi-agent simulation framework that enables automated evaluation of LLMs as clinical conversational agents. AutoMedic transforms off-the-shelf static QA datasets into virtual patient profiles, enabling realistic and clinically grounded multi-turn clinical dialogues between LLM agents. The performance of various clinical conversational agents is then assessed based on our CARE metric, which provides a multi-faceted evaluation standard of clinical conversational accuracy, efficiency/strategy, empathy, and robustness. Our findings, validated by human experts, demonstrate the validity of AutoMedic as an automated evaluation framework for clinical conversational agents, offering practical guidelines for the effective development of LLMs in conversational medical applications. 

---
# Grammaticality Judgments in Humans and Language Models: Revisiting Generative Grammar with LLMs 

**Authors**: Lars G.B. Johnsen  

**Link**: [PDF](https://arxiv.org/pdf/2512.10453)  

**Abstract**: What counts as evidence for syntactic structure? In traditional generative grammar, systematic contrasts in grammaticality such as subject-auxiliary inversion and the licensing of parasitic gaps are taken as evidence for an internal, hierarchical grammar. In this paper, we test whether large language models (LLMs), trained only on surface forms, reproduce these contrasts in ways that imply an underlying structural representation.
We focus on two classic constructions: subject-auxiliary inversion (testing recognition of the subject boundary) and parasitic gap licensing (testing abstract dependency structure). We evaluate models including GPT-4 and LLaMA-3 using prompts eliciting acceptability ratings. Results show that LLMs reliably distinguish between grammatical and ungrammatical variants in both constructions, and as such support that they are sensitive to structure and not just linear order. Structural generalizations, distinct from cognitive knowledge, emerge from predictive training on surface forms, suggesting functional sensitivity to syntax without explicit encoding. 

---
# Enhancing Next-Generation Language Models with Knowledge Graphs: Extending Claude, Mistral IA, and GPT-4 via KG-BERT 

**Authors**: Nour El Houda Ben Chaabene, Hamza Hammami  

**Link**: [PDF](https://arxiv.org/pdf/2512.10440)  

**Abstract**: Large language models (LLMs) like Claude, Mistral IA, and GPT-4 excel in NLP but lack structured knowledge, leading to factual inconsistencies. We address this by integrating Knowledge Graphs (KGs) via KG-BERT to enhance grounding and reasoning. Experiments show significant gains in knowledge-intensive tasks such as question answering and entity linking. This approach improves factual reliability and enables more context-aware next-generation LLMs. 

---
# Asynchronous Reasoning: Training-Free Interactive Thinking LLMs 

**Authors**: George Yakushev, Nataliia Babina, Masoud Vahid Dastgerdi, Vyacheslav Zhdanovskiy, Alina Shutova, Denis Kuznedelev  

**Link**: [PDF](https://arxiv.org/pdf/2512.10931)  

**Abstract**: Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities and safety, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embedded assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of rotary embeddings to enable LLMs built for sequential interactions to simultaneously think, listen, and generate outputs. We evaluate our approach on math, commonsense, and safety reasoning and find that it can generate accurate thinking-augmented answers in real time, reducing time to first non-thinking token from minutes to <= 5s. and the overall real-time delays by 6-11x. 

---
# PARAN: Persona-Augmented Review ANswering system on Food Delivery Review Dataset 

**Authors**: Moonsoo Park, Jeongseok Yun, Bohyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10148)  

**Abstract**: Personalized review response generation presents a significant challenge in domains where user information is limited, such as food delivery platforms. While large language models (LLMs) offer powerful text generation capabilities, they often produce generic responses when lacking contextual user data, reducing engagement and effectiveness. In this work, we propose a two-stage prompting framework that infers both explicit (e.g., user-stated preferences) and implicit (e.g., demographic or stylistic cues) personas directly from short review texts. These inferred persona attributes are then incorporated into the response generation prompt to produce user-tailored replies. To encourage diverse yet faithful generations, we adjust decoding temperature during inference. We evaluate our method using a real-world dataset collected from a Korean food delivery app, and assess its impact on precision, diversity, and semantic consistency. Our findings highlight the effectiveness of persona-augmented prompting in enhancing the relevance and personalization of automated responses without requiring model fine-tuning. 

---
# CIEGAD: Cluster-Conditioned Interpolative and Extrapolative Framework for Geometry-Aware and Domain-Aligned Data Augmentation 

**Authors**: Keito Inoshita, Xiaokang Zhou, Akira Kawai, Katsutoshi Yada  

**Link**: [PDF](https://arxiv.org/pdf/2512.10178)  

**Abstract**: In practical deep learning deployment, the scarcity of data and the imbalance of label distributions often lead to semantically uncovered regions within the real-world data distribution, hindering model training and causing misclassification near class boundaries as well as unstable behaviors in peripheral areas. Although recent large language models (LLMs) show promise for data augmentation, an integrated framework that simultaneously achieves directional control of generation, domain alignment, and quality control has not yet been fully established. To address these challenges, we propose a Cluster-conditioned Interpolative and Extrapolative framework for Geometry-Aware and Domain-aligned data augmentation (CIEGAD), which systematically complements both in-distribution and out-of-distribution semantically uncovered regions. CIEGAD constructs domain profiles through cluster conditioning, allocates generation with a hierarchical frequency-geometric allocation integrating class frequency and geometric indicators, and finely controls generation directions via the coexistence of interpolative and extrapolative synthesis. It further performs quality control through geometry-constrained filtering combined with an LLM-as-a-Judge mechanism. Experiments on multiple classification tasks demonstrate that CIEGAD effectively extends the periphery of real-world data distributions while maintaining high alignment between generated and real-world data as well as semantic diversity. In particular, for long-tailed and multi-class classification tasks, CIEGAD consistently improves F1 and recall, validating the triple harmony of distributional consistency, diversity, and quality. These results indicate that CIEGAD serves as a practically oriented data augmentation framework that complements underrepresented regions while preserving alignment with real-world data. 

---
# BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization 

**Authors**: Kesheng Chen, Wenjian Luo, Zhenqian Zhu, Yamin Hu, Yiya Xi  

**Link**: [PDF](https://arxiv.org/pdf/2512.09972)  

**Abstract**: Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: this https URL. 

---
# Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning 

**Authors**: Hang Ni, Yuzhi Wang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20505)  

**Abstract**: Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process. 

---
# LLM-PEA: Leveraging Large Language Models Against Phishing Email Attacks 

**Authors**: Najmul Hassan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10104)  

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination. 

---
