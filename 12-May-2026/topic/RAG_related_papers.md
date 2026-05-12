# The First Drop of Ink: Nonlinear Impact of Misleading Information in Long-Context Reasoning 

**Authors**: Muhan Gao, Zih-Ching Chen, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10828)  

**Abstract**: As large language models are increasingly deployed in retrieval-augmented generation and agentic systems that accumulate extensive context, understanding how distracting information affects long-context performance becomes critical. Prior work shows that semantically relevant yet misleading documents degrade performance, but the quantitative relationship between the proportion of distractors and performance remains unstudied. In this work, we systematically vary the hard-distractor proportion in fixed-length contexts, revealing a striking nonlinear pattern: as the proportion of hard distractors increases, performance drops sharply within the first small fraction, while the remainder of the range yields only marginal additional decline. We term this ''The First Drop of Ink'' effect, analogous to how a single drop of ink contaminates water. Our theoretical and empirical analyses grounded in attention mechanics show that hard distractors capture disproportionate attention even at small proportions, with diminishing marginal impact as their proportion grows. Controlled experiments further show that filtering gains mainly come from context-length reduction rather than distractor removal; substantial recovery requires reducing the hard-distractor proportion to near zero, highlighting the importance of upstream retrieval precision. 

---
# PathISE: Learning Informative Path Supervision for Knowledge Graph Question Answering 

**Authors**: Shengxiang Gao, Chao Lei, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10791)  

**Abstract**: Knowledge Graph Question Answering (KGQA) aims to answer user questions by reasoning over Knowledge Graphs (KGs). Recent KGQA methods mainly follow the retrieval-augmented generation paradigm to ground Large Language Models~(LLMs) with structured knowledge from KGs. However, training effective models to retrieve question-relevant evidence from KGs typically requires high-quality intermediate supervision signals, such as question-relevant paths or subgraphs, which are time- and resource-intensive to obtain. We propose PathISE, a novel framework for learning high-quality intermediate supervision from answer-level labels. PathISE introduces a lightweight transformer-based estimator that estimates the informativeness of relation paths to construct pseudo path-level supervision. This supervision is then distilled into an LLM path generator, whose generated paths are grounded in the KG to provide compact evidence for inductive answer reasoning. ExtensiveISE experiments on three KGQA benchmarks show that PathISE achieves competitive or state-of-the-art KGQA performance, and provides reusable supervision signals that can enhance existing KGQA models, without relying on costly LLM-refined supervision signals. Our source code is available at this https URL. 

---
# ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox 

**Authors**: Yuanyang Li, Xue Yang, Longyue Wang, Weihua Luo, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.10787)  

**Abstract**: Current LLM agents are proficient at calling isolated APIs but struggle with the "last mile" of commercial software automation. In real-world scenarios, tools are not independent; they are atomic, interdependent, and prone to environmental noise. We introduce $\textbf{ComplexMCP}$, a benchmark designed to evaluate agents in these rigorous conditions. Built on the Model Context Protocol (MCP), $\textbf{ComplexMCP}$ provides over 300 meticulously tested tools derived from 7 stateful sandboxes, ranging from office suites to financial systems. Unlike existing datasets, our benchmark utilizes a seed-driven architecture to simulate dynamic environment states and unpredictable API failures, ensuring a deterministic yet diverse evaluation.
We evaluate various LLMs across full-context and RAG paradigms, revealing a stark performance gap: even top-tier models fail to exceed a 60% success rate, far trailing human performance 90%. Granular trajectory analysis identifies three fundamental bottlenecks: (1) $\textbf{tool retrieval saturation}$ as action spaces scale; (2) $\textbf{over-confidence}$, where agents skip essential environment verifications; and (3) $\textbf{strategic defeatism}$, a tendency to rationalize failure rather than pursuing recovery. These findings underscore the insufficiency of current agents for interdependent workflows, positioning $\textbf{ComplexMCP}$ as a critical testbed for the next generation of resilient autonomous systems. 

---
# MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs 

**Authors**: Ruiyi Yang, Zechen Li, Hao Xue, Imran Razzak, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2605.10064)  

**Abstract**: Self-evolving language-model agents must decide what to learn next and how to preserve what they have learned across iterations. Existing systems typically carry this cross-iteration knowledge as natural-language feedback, flat episodic memory, or implicit reinforcement signals, none of which cleanly supports a frozen weak backbone at inference time. This paper introduces MAGE (Multi-Agent Graph-guided Evolution), a framework that externalizes self-knowledge into a four-subgraph co-evolutionary knowledge graph. Its experience subgraph stores both teacher-written failure corrections and the learner's own past correct reasoning traces, which are retrieved as task-conditioned guidance for a frozen execution model. During evolution, the graph, a task-level search bandit, and a skill-level routing bandit are updated from the same reward stream, while the learner's backbone remains unchanged. We further provide structural analysis showing how append-only memory growth, bounded curriculum coverage, and task-filtered retrieval together support stable improvement of the retrieval substrate for frozen-learner evolution. Across nine benchmarks spanning mathematical reasoning, multi-hop and open-domain question answering, spatio-temporal analysis, financial numerical reasoning, medical multiple-choice, an open-world survival game, and web navigation, MAGE achieves strong performance against prompt-based frozen-backbone baselines. Ablations show that self-harvested success traces and teacher-written corrections are complementary, with success memories contributing most on reasoning-template-heavy tasks and corrective memories supporting harder composition and interaction settings. 

---
# EpiGraph: A Knowledge Graph and Benchmark for Evidence-Intensive Reasoning in Epilepsy 

**Authors**: Yuyang Dai, Zheng Chen, Jathurshan Pradeepkumar, Yasuko Matsubara, Jimeng Sun, Yasushi Sakurai, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.09505)  

**Abstract**: Epilepsy diagnosis and treatment require evidence-intensive reasoning across heterogeneous clinical knowledge, including biosignal patterns, genetic mechanisms, pharmacogenomics, treatment strategies, and patient outcomes. In this work, we present \textsc{EpiGraph}, a large-scale epilepsy knowledge graph and benchmark for evaluating knowledge-augmented clinical reasoning. \textsc{EpiGraph} integrates 48,166 peer-reviewed papers and seven clinical resources into a heterogeneous graph containing 24,324 entities and 32,009 evidence-grounded triplets across five clinical layers. Built upon this graph, \textsc{EpiBench} defines five clinically motivated tasks spanning clinical decision-making, EEG report generation, pharmacogenomic precision medicine, treatment recommendation, and deep research planning. We evaluate six LLMs under both standard and Graph-RAG settings. Results show that integrating \textsc{EpiGraph} consistently improves performance across all tasks, with the largest gains observed in pharmacogenomic reasoning (+30--41\%). Our findings demonstrate that structured epilepsy knowledge substantially enhances evidence-grounded clinical reasoning and provides a practical benchmark framework for evaluating knowledge-augmented LLMs in real-world neurological settings. Our code is available at: this https URL. 

---
# VulTriage: Triple-Path Context Augmentation for LLM-Based Vulnerability Detection 

**Authors**: Wenxin Tang, Xiang Zhang, Junliang Liu, Jingyu Xiao, Xi Xiao, Jinlong Yang, Yuehe Ma, Zhenyu Liu, Zhengheng Li, Zicheng Wang, Wang Luo, Qing Li, Lei Wang, Peng Xiangli  

**Link**: [PDF](https://arxiv.org/pdf/2605.09461)  

**Abstract**: Automated vulnerability detection is a fundamental task in software security, yet existing learning-based methods still struggle to capture the structural dependencies, domain-specific vulnerability knowledge, and complex program semantics required for accurate detection. Recent Large Language Models (LLMs) have shown strong code understanding ability, but directly prompting them with raw source code often leads to missed vulnerabilities or false alarms, especially when vulnerable and benign functions differ only in subtle semantic details. To address this, we propose VulTriage, a triple-path context augmentation framework for LLM-based vulnerability detection. VulTriage enhances the LLM input through three complementary paths: a Control Path that extracts and verbalizes AST, CFG, and DFG information to expose control and data dependencies; a Knowledge Path that retrieves relevant CWE-derived vulnerability patterns and examples through hybrid dense--sparse retrieval; and a Semantic Path that summarizes the functional behavior of the code before the final judgment. These contexts are integrated into a unified instruction to guide the LLM toward more reliable vulnerability reasoning. Experiments on the PrimeVul pair test set show that VulTriage achieves state-of-the-art performance, outperforming existing deep learning and LLM-based baselines on key pair-wise and classification metrics. Further ablation studies verify the effectiveness of each path, and additional experiments on the Kotlin dataset demonstrate the generalization ability of VulTriage under low-resource and class-imbalanced settings. Our code is available at this https URL 

---
# EquiMem: Calibrating Shared Memory in Multi-Agent Debate via Game-Theoretic Equilibrium 

**Authors**: Yuqiao Meng, Sakshi Sunil Narvekar, Luoxi Tang, Rupali Rajendra Vaje, Yingxue Zhang, Muchao Ye, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09278)  

**Abstract**: Multi-agent debate (MAD) systems increasingly rely on shared memory to support long-horizon reasoning, but this convenience opens a critical vulnerability: a single corrupted entry can contaminate the downstream memory-augmented reasoning, and debate alone fails to filter such errors. Existing safeguards filter entries via heuristics or LLM-based validation, yet they rely on AI judgments that share the same failure modes and overlook the cross-agent dynamics of MAD. We address this gap by formulating memory updating in MAD as a zero-trust memory game, in which no agent is assumed honest and the game's equilibrium serves as an indicator of optimal memory trust. Guided by this equilibrium, we propose EquiMem, an inference-time calibration mechanism that quantifies each update algorithmically against the shared memory state, using agents' existing retrieval queries and traversal paths as evidence rather than soliciting any LLM judgment. EquiMem instantiates calibration for both embedding- and graph-based memory, and across diverse benchmarks, MAD frameworks, and memory architectures, it consistently outperforms existing safeguards, remains robust under adversarial agents, and incurs negligible inference overhead. 

---
# SearchSkill: Teaching LLMs to Use Search Tools with Evolving Skill Banks 

**Authors**: Jinchao Hu, Meizhi Zhong, Kehai Chen, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09038)  

**Abstract**: Teaching language models to use search tools is not only a question of whether they search, but also of whether they issue good queries. This is especially important in open-domain question answering, where broad or copied queries often waste retrieval budget and derail later reasoning. We propose \Ours, a framework that makes query planning explicit through reusable search skills. At each step, the model first selects a skill, then generates a search or answer action conditioned on the selected skill card. The skill inventory itself is not fixed: SearchSkill maintains an evolving SkillBank, expands or refines it from recurrent failure patterns, and reconstructs affected trajectories before supervised training. The resulting two-stage SFT recipe aligns training with the inference-time protocol of skill selection followed by skill-grounded execution. Across open-source and closed-source models, SearchSkill improves exact match on knowledge-intensive QA benchmarks and yields better retrieval behavior, including fewer copied first queries, more atomic hop-focused queries, and more correct answers within a small search budget. These results suggest that explicit skill-conditioned query planning is a lightweight alternative to treating search as an undifferentiated action. 

---
# Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient? 

**Authors**: Tz-Huan Hsu, Jheng-Hong Yang, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10848)  

**Abstract**: Does a lexical retriever suffice as large language models (LLMs) become more capable in an agentic loop? This question naturally arises when building deep research systems. We revisit it by pairing BM25 with frontier LLMs that have better reasoning and tool-use abilities. To support researchers asking the same question, we introduce Pi-Serini, a search agent equipped with three tools for retrieving, browsing, and reading documents. Our results show that, on BrowseComp-Plus, a well-configured lexical retriever with sufficient retrieval depth can support effective deep research when paired with more capable LLMs. Specifically, Pi-Serini with gpt-5.5 achieves 83.1% answer accuracy and 94.7% surfaced evidence recall, outperforming released search agents that use dense retrievers. Controlled ablations further show that BM25 tuning improves answer accuracy by 18.0% and surfaced evidence recall by 11.1% over the default BM25 setting, while increasing retrieval depth further improves surfaced evidence recall by 25.3% over the shallow-retrieval setting. Source code is available at this https URL. 

---
# When Can Digital Personas Reliably Approximate Human Survey Findings? 

**Authors**: Mumin Jia, Yilin Chen, Divya Sharma, Jairo Diaz-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.10659)  

**Abstract**: Digital personas powered by Large Language Models (LLMs) are increasingly proposed as substitutes for human survey respondents, yet it remains unclear when they can reliably approximate human survey findings. We answer this question using the LISS panel, constructing personas from respondents' background variables and pre-2023 survey histories, then testing them against the same respondents' held-out post-cutoff answers. Across four persona architectures, three LLMs, and two prediction tasks, we assess performance at the question, respondent, distributional, equity, and clustering levels. Digital personas improve alignment with human response distributions, especially in domains tied to stable attributes and values, but remain limited for individual prediction and fail to recover multivariate respondent structure. Retrieval-augmented architectures provide the clearest gains, but performance depends more on human response structure than on model choice: personas perform best for low-variability questions and common respondent patterns, and worst for subjective, heterogeneous, or rare responses. Our results provide practical guidance on when digital personas could be appropriate for survey research and when human validation remains necessary. 

---
# Qwen Goes Brrr: Off-the-Shelf RAG for Ukrainian Multi-Domain Document Understanding 

**Authors**: Anton Bazdyrev, Ivan Bashtovyi, Ivan Havlytskyi, Oleksandr Kharytonov, Artur Khodakovskyi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10296)  

**Abstract**: We participated in the Fifth UNLP shared task on multi-domain document understanding, where systems must answer Ukrainian multiple-choice questions from PDF collections and localize the supporting document and page. We propose a retrieval-augmented pipeline built around three ideas: contextual chunking of PDFs, question-aware dense retrieval and reranking conditioned on both the question and answer options, and constrained answer generation from a small set of reranked passages. Our final system uses Qwen3-Embedding-8B for retrieval, a fine-tuned Qwen3-Reranker-8B for passage ranking, and Qwen3-32B for answer selection. On a held-out split, reranking improves Recall@1 from 0.6957 to 0.7935, while using the top-2 reranked passages raises answer accuracy from 0.9348 to 0.9674. Our best leaderboard run reached 0.9452 on the public leaderboard and 0.9598 on the private leaderboard. Our results suggest that, under strict code-competition constraints, preserving document structure and making relevance estimation aware of the answer space are more effective than adding complex downstream heuristics. 

---
# Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation 

**Authors**: Peiru Yang, Haoran Zheng, Tong Ju, Shiting Wang, Wanchun Ni, Jiajun Liu, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10253)  

**Abstract**: Retrieval-augmented generation (RAG) is a widely adopted paradigm for enhancing LLMs in medical applications by incorporating expert multimodal knowledge during generation. However, the underlying retrieval databases may naturally contain, or be intentionally injected with, adversarial knowledge, which can perturb model outputs and undermine system reliability. To investigate this risk, prior studies have explored knowledge poisoning attacks in medical RAG systems. Nevertheless, most of them rely on the strong assumption that adversaries possess prior knowledge of user queries, which is unrealistic in deployments and substantially limits their practical applicability. In this paper, we propose M\textsuperscript{3}Att, a knowledge-poisoning framework designed for medical multimodal RAG systems, assuming only limited distribution knowledge of the underlying database. Our core idea is to inject covert misinformation into textual data while using paired visual data as a query-agnostic trigger to promote retrieval. We first propose a unified framework that introduces imperceptible perturbations to visual inputs to manipulate retrieval probabilities. Besides, due to the prior medical knowledge in LLMs, naively poisoned medical content with explicit factual errors can be corrected during generation. Thus, we leverage the inherent ambiguity of medical diagnosis and design a covert misinformation injection strategy that degrades diagnostic accuracy while evading model self-correction. Experiments on five LLMs and datasets demonstrate that M\textsuperscript{3}Att consistently produces clinically plausible yet incorrect generations. Codes: this https URL. 

---
# MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading 

**Authors**: Baibei Ji, Xiaoyang Weng, Juntao Li, Zecheng Tang, Yihang Lou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10268)  

**Abstract**: To tackle long-context reasoning tasks without the quadratic complexity of standard attention mechanisms, approaches based on agent memory have emerged, which typically maintain a dynamically updated memory when linearly processing document chunks. To mitigate the potential loss of latent evidence in this memorize-while-reading paradigm, recent works have integrated retrieval modules that allow agents to recall information previously discarded during memory overwriting. However, retrieval-based recall suffers from both evidence loss during memory formation and interference induced by invalid queries. To overcome these limitations, we propose MemReread. Built upon streaming reading, MemReread circumvents intermediate retrieval. It triggers question decomposition and rereading when the final memory is insufficient, enabling the recovery of indirect facts that were prematurely discarded. This design supports non-linear reasoning while preserving the inherent logical flow of document comprehension. To further enhance practicality, we introduce a reinforcement learning framework that enhances length extrapolation capability while dynamically determining the number of rereading passes based on task complexity, thereby flexibly controlling computational overhead. Extensive experiments demonstrate that MemReread consistently outperforms baseline frameworks on long-context reasoning tasks, while maintaining linear time complexity with respect to context length. 

---
# MicroWorld: Empowering Multimodal Large Language Models to Bridge the Microscopic Domain Gap with Multimodal Attribute Graph 

**Authors**: Manyu Li, Ruian He, Chenxi Ma, Weimin Tan, Bo Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.10120)  

**Abstract**: Multimodal large language models (MLLMs) show remarkable potential for scientific reasoning, yet their performance in specialized domains such as microscopy remains limited by the scarcity of domain-specific training data and the difficulty of encoding fine-grained expert knowledge into model parameters. To bridge the gap, we introduce MicroWorld, a framework that constructs a multimodal attributed property graph (MAPG) from large-scale scientific image--caption corpora and leverages it to augment MLLM reasoning at inference time without any domain-specific fine-tuning. MicroWorld extracts biomedical entities and relations via scispaCy or LLM-based triplet mining, aligns images and entities in a shared embedding space using Qwen3-VL-Embedding, and assembles a knowledge graph comprising approximately 111K nodes and 346K typed edges spanning eight relation categories. At inference time, a graph-augmented retrieval pipeline matches query entities to the MAPG and injects structured knowledge context into the MLLM prompt. On the MicroVQA benchmark, MicroWorld improves the reasoning performance of Qwen3-VL-8B-Instruct by 37.5%, outperforming GPT-5 by 13.0% to achieve a new state-of-the-art. Furthermore, it yields a 6.0% performance gain on the MicroBench benchmark. Extensive experiments demonstrate the enhanced generalization capability introduced by MicroWorld. A qualitative case study further reveals both the mechanisms through which structured knowledge improves reasoning and the failure modes that point to promising future directions. Code and data are available at this https URL. 

---
# MedMeta: A Benchmark for LLMs in Synthesizing Meta-Analysis Conclusion from Medical Studies 

**Authors**: Huy Hoang Ha, Benoit Favre, Francois Portet  

**Link**: [PDF](https://arxiv.org/pdf/2605.09661)  

**Abstract**: Large language models (LLMs) have saturated standard medical benchmarks that test factual recall, yet their ability to perform higher-order reasoning, such as synthesizing evidence from multiple sources, remains critically under-explored. To address this gap, we introduce MedMeta, the first benchmark designed to evaluate an LLM's ability to generate conclusions from medical meta-analyses using only the abstracts of cited studies. MedMeta comprises 81 meta-analyses from PubMed (2018--2025) and evaluates models using two distinct workflows: a Retrieval-Augmented Generation (Golden-RAG) setting with ground-truth abstracts, and a Parametric-only approach relying on internal knowledge. Our evaluation framework is validated by a well-structured analysis showing our LLM-as-a-judge protocol strongly aligns with human expert ratings, as evidenced by high Pearson's r correlation (0.81) and Bland-Altman analysis revealing negligible systematic bias, establishing it as a reliable proxy for scalable evaluation. Our findings underscore the critical importance of information grounding: the Golden-RAG workflow consistently and significantly outperforms the Parametric-only approach across models. In contrast, the benefits of domain-specific fine-tuning are marginal and largely neutralized when external material is provided. Furthermore, stress tests show that all models, regardless of architecture, fail to identify and reject negated evidence, highlighting a critical vulnerability in current RAG systems. Notably, even under ideal RAG conditions, current LLMs achieve only slightly above-average performance (~2.7/5.0). MedMeta provides a challenging new benchmark for evidence synthesis and demonstrates that for clinical applications, developing robust RAG systems is a more promising direction than model specialization alone. 

---
# Governing AI-Assisted Security Operations: A Design Science Framework for Operational Decision Support 

**Authors**: Elyson A. De La Cruz, Rishikesh Sahay, Md Rasel Al Mamun  

**Link**: [PDF](https://arxiv.org/pdf/2605.09534)  

**Abstract**: Engineering managers increasingly must decide how to introduce generative artificial intelligence (AI), retrieval-augmented generation, and coding agents into high-risk operational functions without weakening accountability, privacy, cost discipline, or auditability. The central message of this study is that AI-assisted operational decision support should be managed as a governed engineering capability before it is scaled as automation. Security operations centers (SOCs) provide a suitable setting because they combine privileged telemetry, specialist expertise, software repositories, cloud services, and evidence-sensitive decisions. This study uses Kusto Query Language (KQL) and Microsoft Azure security capabilities as a bounded technical instantiation of that broader engineering management problem. KQL is read-only in ordinary query use, but read-only does not mean risk-free: AI-assisted queries can still create privacy, cost, performance, schema-validity, and decision-quality risks through broad scans, sensitive-field exposure, stale intelligence, and misleading interpretations. Using design science research, the study develops a governed AI query-broker artifact that separates AI planning from operational execution through schema-grounded retrieval, approved templates, policy validation, read-only adapters, normalized outputs, auditable agent traces, and engineering review board gates. The contribution is not a new KQL technique, security product, or detection algorithm. Rather, the study contributes a management framework for governing AI-assisted operational decision support in high-risk digital infrastructure by specifying design propositions, role accountability, maturity stages, quality gates, evaluation criteria, and evidence boundaries. 

---
# Assessment of RAG and Fine-Tuning for Industrial Question-Answering-Applications 

**Authors**: Jakob Sturm, Josef Pichlmeier, Christian Bernhard, Maka Karalashvili, Johannes Klepsch, Georg Groh, Andre Luckow  

**Link**: [PDF](https://arxiv.org/pdf/2605.09533)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in enterprise question-answering (QA) systems, requiring adaptation to domain-specific knowledge. Among the most prevalent methods for incorporating such knowledge are Retrieval-Augmented Generation (RAG) and fine-tuning (FT). Yet, from a cost-accuracy trade-off perspective, it remains unclear which approach best suits industry scenarios. This study examines the impact of RAG and FT on two closed datasets specific to the automotive industry, assessing answer quality and operational costs. We extend the Cost-of-Pass framework proposed by Erol et al. (arXiv:2504.13359) to jointly assess output quality, generation cost, and user interaction cost. Our findings reveal that while premium models perform best out of the box, open-source models can achieve comparable quality when enhanced with RAG. Overall, RAG emerges as the most effective and cost-efficient adaptation method for both closed- and open-source models. 

---
# The Trap of Trajectory: Towards Understanding and Mitigating Spurious Correlations in Agentic Memory 

**Authors**: Luoxi Tang, Rupali Rajendra Vaje, Yuqiao Meng, Sakshi Sunil Narkar, Weicheng Ma, Zeyu Ding, Dazheng Zhang, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09330)  

**Abstract**: Agentic memory enables LLMs to persist information beyond a single context window and reuse it in later decisions, but it also introduces a new vulnerability: spurious correlations, where retrieved memory carries miscorrelated evidence and propagates erroneous reasoning into downstream decisions. Despite the widespread use of agentic memory, this risk remains largely underexplored. We address it from two aspects. First, we benchmark several canonical types of spurious patterns identified through causal structure and record them across trajectory-level memory. Diagnosing agentic memory systems on this benchmark reveals that memory improves reasoning on clean inputs but amplifies reliance on spurious patterns when they are present. Second, we propose CAMEL, a plug-and-play calibration method that operates across diverse memory architectures at both write and retrieval time. CAMEL consistently reduces reliance on spurious patterns across all three types while preserving or improving performance on clean inputs and staying robust under adaptive attacks targeting the calibration. Overall, CAMEL offers a principled and lightweight solution toward more reliable agentic memory deployment. 

---
# ShadowMerge: A Novel Poisoning Attack on Graph-Based Agent Memory via Relation-Channel Conflicts 

**Authors**: Yang Luo, Zifeng Kang, Tiantian Ji, Xinran Liu, Yong Liu, Shuyu Li, Lingyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.09033)  

**Abstract**: Graph-based agent memory is increasingly used in LLM agents to support structured long-term recall and multi-hop reasoning, but it also creates a new poisoning surface: an attacker can inject a crafted relation into graph memory so that it is later retrieved and influences agent behavior. Existing agent-memory poisoning attacks mainly target flat textual records and are ineffective in graph-based memory because malicious relations often fail to be extracted, merged into the target anchor neighborhood, or retrieved for the victim query.
We present SHADOWMERGE, a poisoning attack against graph-based agent memory that exploits relation-channel conflicts. Its key insight is that a poisoned relation can share the same query-activated anchor and canonicalized relation channel as benign evidence while carrying a conflicting value. To realize this, we design AIR, a pipeline that converts the conflict into an ordinary interaction that can be extracted, merged, and retrieved by the graph-memory system. We evaluate SHADOWMERGE on Mem0 and three public real-world datasets: PubMedQA, WebShop, and ToolEmu. SHADOWMERGE achieves 93.8% average attack success rate, improving the best baseline by 50.3 absolute points, while having negligible impact on unrelated benign tasks. Mechanism studies show that SHADOWMERGE overcomes the three key limitations of existing agent-memory poisoning attacks, and defense analysis shows that representative input-side defenses are insufficient to mitigate it. We have responsibly disclosed our findings to affected graph-memory vendors and open sourced SHADOWMERGE. 

---
# Generating Leakage-Free Benchmarks for Robust RAG Evaluation 

**Authors**: Jiayi Liu, Jiaxing Zhang, Bowen Jin, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2605.08838)  

**Abstract**: Retrieval-augmented generation (RAG) is widely used to augment large language models (LLMs) with external knowledge. However, many benchmark datasets, designed to test RAG performance, comprise many questions that can already be answered from an LLM's parametric memory. This leads to unreliable evaluation. We refer to this phenomenon as knowledge leakage: cases where RAG tasks are solvable without retrieval. This issue worsens over time due to benchmark aging. As benchmarks are reused for training, their contents are increasingly absorbed into model parameters, making them less effective for evaluating retrieval.
We introduce SeedRG, a semi-synthetic benchmark generation pipeline that mitigates knowledge leakage and addresses the issue of benchmark aging. Starting from a seed benchmark dataset, SeedRG extracts a reasoning graph from question-context pairs to capture their underlying reasoning structure, and then generates new examples via type-constrained entity replacement. This process produces structurally similar but novel instances that are unlikely to exist in the model's parametric knowledge, while preserving the original reasoning patterns. To ensure quality, we incorporate two verification steps: (1) a reasoning-graph consistency check to maintain task difficulty, and (2) a knowledge-leakage filter to exclude instances answerable without retrieval. 

---
# Do Benchmarks Underestimate LLM Performance? Evaluating Hallucination Detection With LLM-First Human-Adjudicated Assessment 

**Authors**: I. F. Atasoy, B. Mutlu, E. A. Sezer, A. Wahdan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08462)  

**Abstract**: Hallucination remains a persistent challenge in Large Language Models (LLMs), particularly in context-grounded settings such as RAG and agentic AI systems. This study focuses on contextual hallucination detection in summarization tasks. We analyze the QAGS-C and SummEval datasets by comparing original benchmark annotations with reason and span-based predictions from Gemini 2.5 Flash and GPT-5 Mini. To address systematic divergences between human labels and LLM judgments, we re-evaluated all conflicted samples through a human adjudication process involving 2 cross-cultural adjudicators. Following this re-evaluation, triple agreement (between human, GPT, and Gemini) increased by 6.38% for QAGS-C and 7.62% for SummEval. Similarly, model accuracy improved, with GPT increasing by 4.25% on QAGS-C and 2.34% on SummEval, while Gemini showed gains of 8.51% and 3.80%, respectively. Notably, adjudicators frequently sided with the models' judgments over original human annotations when LLMs provided explicit reasoning. Overall human adjudicator agreement ranged between 83% and 87%. These findings suggest that for ambiguity-prone tasks, single-pass annotations may be insufficient, and model-assisted re-evaluation yields more reliable benchmarks. 

---
# Defense effectiveness across architectural layers: a mechanistic evaluation of persistent memory attacks on stateful LLM agents 

**Authors**: Jun Wen Leong  

**Link**: [PDF](https://arxiv.org/pdf/2605.08442)  

**Abstract**: Persistent memory attacks against LLM agents achieve high attack success rates against open-source models. In these attacks, malicious instructions injected via RAG-retrieved documents are stored in persistent memory and executed in later sessions. However, no systematic evaluation of defense effectiveness against this attack class exists. We evaluate six defenses across four architectural layers against delayed-trigger attacks on nine open-source models (5,040 runs, N=40 per condition). Four defenses fail at approximately baseline attack success rate: input-level filtering (Minimizer, Sanitizer) and retrieval-level filtering (RAG Sanitizer, RAG LLM Judge) achieve 88-89% ASR, statistically indistinguishable from the undefended baseline of 88.6%. Prompt Hardening partially fails at 77.8% ASR, with the reduction driven by two models at 0%: one genuine defense effect and one model-level refusal independent of the defense. The architectural explanation holds: input-level defenses cannot observe RAG-injected content, and retrieval-level classifiers are defeated by compliance-framed semantic masking. One defense, tool-gating at the memory layer (Memory Sandbox), reduces ASR to 0% for eight of nine models by removing the recall capability the attack requires. The exception inverts the defense entirely: a reasoning model that achieves 0% ASR under no defense via execution refusal inverts to 100% ASR under Memory Sandbox, because removing explicit recall forces the model onto the RAG pathway where its refusal mechanism does not activate. Memory Sandbox imposes zero utility cost in the absence of attack (BTCR = 100% across all conditions). These results provide the first systematic characterization of why each defense class fails against persistent memory attacks, enabling informed defense investment decisions. 

---
# CDS4RAG: Cyclic Dual-Sequential Hyperparameter Optimization for RAG 

**Authors**: Pengzhou Chen, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08333)  

**Abstract**: Retrieval-Augmented Generation (RAG) is sensitive to the vast hyperparameters of the retriever and generator, yet optimizing them using given queries is a challenging task due to the complex interactions and expensive evaluation costs. Existing algorithms are ineffective and slow in convergence, since they often treat RAG as a monolithic black box or only optimize partial hyperparameters. In this paper, we propose CDS4RAG, a framework that optimizes the full RAG hyperparameters using given queries via a new cyclic dual-sequential formulation. CDS4RAG is special in the sense that it distinguishes the hyperparameters of the retriever and generator, cyclically optimizing them in turn. Such a paradigm allows us to design fine-grained within-cycle budget provision and expedite the optimization via cross-cycle seeding when optimizing the generator. CDS4RAG is also an algorithm-agnostic framework that can be paired with diverse general algorithms. Through experiments on four common benchmarks and two backbone LLMs, we reveal that CDS4RAG considerably boosts the vanilla algorithms in 21/24 cases while significantly outperforming state-of-the-art algorithms in all cases with up to 1.54x improvements of generation quality and better speedup. 

---
# Context-Augmented Code Generation: How Product Context Improves AI Coding Agent Decision Compliance by 49% 

**Authors**: Drew Dillon, Kasyap Varanasi  

**Link**: [PDF](https://arxiv.org/pdf/2605.08112)  

**Abstract**: AI coding agents powered by large language models can read codebases and produce functional code, but they routinely violate team-specific product decisions that are invisible in the source code alone. We introduce a controlled benchmark measuring decision compliance, the rate at which an AI coding agent follows established product, design, and engineering decisions, across 8 realistic software engineering tasks containing 41 weighted decision points. We compare a baseline configuration (Claude Code with codebase access only) against an augmented configuration that adds Brief, a product-context retrieval system providing spec generation, mid-build consultation, and retrieval of recorded decisions, persona pain points, customer signals, and competitive intelligence. On identical prompts and the same repository, the augmented configuration achieves 95% decision compliance versus 46% for the baseline, a 49 percentage point improvement. Per-decision analysis reveals that the baseline achieves 100% compliance on decisions visible in the codebase and 0-33% on decisions requiring product context, suggesting that product-context retrieval is a key driver of the improvement. We release the benchmark repository, all 16 pull requests, and scoring harness for independent reproduction. 

---
# Grounded Satirical Generation with RAG 

**Authors**: Oona Itkonen, Yuxin Su, Linyao Du, Ona De Gibert  

**Link**: [PDF](https://arxiv.org/pdf/2605.10853)  

**Abstract**: Humor generation remains challenging task for Large Language Models (LLMs), due to their subjective nature. We focus on satire, a form of humor strongly shaped by context. In this work, we present a novel pipeline for grounded satire generation that uses Retrieval-Augmented Generation (RAG) over current news to produce satirical dictionary definitions in the Finnish context. We also introduce a new task-specific evaluation framework and annotate 100 generated definitions with six human annotators, enabling analysis across multiple experimental conditions, including cultural background, source-word type, and the presence or absence of RAG. Our results show that the generated definitions are perceived as more political than humorous. Both topic-based word selection and RAG improve the political relevance of the outputs, but neither yields clear gains in humor generation. In addition, our LLM-as-a-judge evaluation of five state-of-the-art models indicates that LLMs correlate well with human judgments on political relevance, but perform poorly on humor. We release our code and annotated dataset to support further research on grounded satire generation and evaluation. 

---
# RUBEN: Rule-Based Explanations for Retrieval-Augmented LLM Systems 

**Authors**: Joel Rorseth, Parke Godfrey, Lukasz Golab, Divesh Srivastava, Jarek Szlichta  

**Link**: [PDF](https://arxiv.org/pdf/2605.10862)  

**Abstract**: This paper demonstrates RUBEN, an interactive tool for discovering minimal rules to explain the outputs of retrieval-augmented large language models (LLMs) in data-driven applications. We leverage novel pruning strategies to efficiently identify a minimal set of rules that subsume all others. We further demonstrate novel applications of these rules for LLM safety, specifically to test the resiliency of safety training and effectiveness of adversarial prompt injections. 

---
# Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection 

**Authors**: Yiwen Chen, Kuan Li, Fuzhen Zhuang, Deqing Wang, Zhao Zhang, Liwen Zhang, Yong Jiang, Shuai Wang, Minhao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.10235)  

**Abstract**: Recent advances in large language models (LLMs) have expanded the context window to beyond 128K tokens, enabling long-document understanding and multi-source reasoning. A key challenge, however, lies in choosing between retrieval-augmented generation (RAG) and long-context (LC) strategies: RAG is efficient but constrained by retrieval quality, while LC supports global reasoning at higher cost and with position sensitivity. Existing methods such as Self-Route adopt failure-driven fallback from RAG to LC, but remain passive, inefficient, and hard to interpret. We propose Pre-Route, a proactive routing framework that performs structured reasoning before answering. Using lightweight metadata (e.g., document type, length, initial snippet), Pre-Route enables task analysis, coverage estimation, and information-need prediction, producing explainable and cost-efficient routing decisions. Our study shows three key findings: (i) LLMs possess latent routing ability that can be reliably elicited with guidelines, allowing single-sample performance to approach that of multi-sample (Best-of-N) results; (ii) linear probes reveal that structured prompts sharpen the separability of the "optimal routing dimension" in representation space; and (iii) distillation transfers this reasoning structure to smaller models for lightweight deployment. Experiments on LaRA (in-domain) and LongBench-v2 (OOD) confirm that Pre-Route outperforms Always-RAG, Always-LC, and Self-Route baselines, achieving superior overall cost-effectiveness. 

---
# ASTRA-QA: A Benchmark for Abstract Question Answering over Documents 

**Authors**: Shu Wang, Shansong Zhou, Xinyang Wang, Shiwei Wang, Hulong Wu, Yixiang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10168)  

**Abstract**: Document-based question answering (QA) increasingly includes abstract questions that require synthesizing scattered information from long documents or across multiple documents into coherent answers. However, this setting is still poorly supported by existing benchmarks and evaluation methods, which often lack stable abstract references or rely on coarse similarity metrics and unstable head-to-head comparisons. To alleviate this issue, we introduce ASTRA-QA, a benchmark for AbSTRAct Question Answering over documents. ASTRA-QA contains 869 QA instances over academic papers and news documents, covering five abstract question types and three controlled retrieval scopes. Each instance is equipped with explicit evaluation annotations, including answer topic sets, curated unsupported topics, and aligned evidence. Building on these annotations, ASTRA-QA assesses whether answers cover required key points and avoid unsupported content by directly scoring topic coverage and curated unsupported content, enabling scalable evaluation without exhaustive head-to-head comparisons. Experiments with representative Retrieval-Augmented Generation (RAG) methods spanning vanilla, graph-based, and hierarchical retrieval settings show that ASTRA-QA provides reference-grounded diagnostics for coverage, hallucination, and retrieval-scope robustness. Our dataset and code are available at this https URL. 

---
# NyayaAI: An AI-Powered Legal Assistant Using Multi-Agent Architecture and Retrieval-Augmented Generation 

**Authors**: Deepanshu, Divi Saxena, Deepali Rana, Ayesha Varshney, Sahinur Rahman Laskar  

**Link**: [PDF](https://arxiv.org/pdf/2605.10155)  

**Abstract**: Legal information in India remains largely inaccessible due to the complexity of legal language and the sheer volume of legal documentation involved in research and case analysis. This paper presents NyayaAI, an AI-powered legal assistant that automates and simplifies legal workflows for lawyers, law students, and general users. The system combines Large Language Models with a Retrieval-Augmented Generation pipeline grounded in a curated Indian legal knowledge base comprising constitutional provisions, statutes, case laws, and judicial precedents. A multi-agent architecture orchestrated through the Mastra TypeScript framework coordinates a main agent with specialized sub-agents handling legal research, document summarization, case law retrieval, and drafting assistance. A compliance module validates all responses before delivery. Domain classification achieved 70\% precision across test samples, with RAG retrieval precision at 74\% and overall response accuracy at 72\%, demonstrating that structured multi-agent LLM systems can meaningfully improve legal accessibility and workflow efficiency. The code\footnote{this https URL} is made publicly available for the benefit of the research community. 

---
# Byte-Exact Deduplication in Retrieval-Augmented Generation: A Three-Regime Empirical Analysis Across Public Benchmarks 

**Authors**: Sietse Schelpe  

**Link**: [PDF](https://arxiv.org/pdf/2605.09611)  

**Abstract**: This preprint presents an empirical analysis of byte-exact chunk-level deduplication in Retrieval-Augmented Generation (RAG) pipelines. We measure context reduction across three distinct operating regimes: clean academic retrieval (0.16% byte reduction on 22.2M BeIR passages), constructed enterprise patterns (24.03% reduction), and multi-turn conversational AI (80.34% reduction). To validate quality preservation, we conducted a cross-vendor 5-judge calibrated panel evaluation across four production APIs (Google Gemini 2.5 Flash, Anthropic Claude Sonnet 4.6, Meta Llama 3.3 70B, and OpenAI GPT-5.1). Applying a five-category human-in-the-loop noise-removal protocol to panel-majority materially different (MAT) pairs, we establish that byte-exact deduplication introduces zero measurable quality regression. Post-audit, all four vendors clear the strict <5% Wilson 95% upper-bound MAT threshold in both the clean and high-redundancy RAG regimes. This work demonstrates that substantial inference compute savings can be achieved deterministically without compromising evaluation-grade model quality. 

---
# GRC: Unifying Reasoning-Driven Generation, Retrieval and Compression 

**Authors**: Zhongtao Miao, Qiyu Wu, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2605.09100)  

**Abstract**: Text embedding and generative tasks are usually trained separately based on large language models (LLMs) nowadays. This causes a large amount of training cost and deployment effort. Context compression is also a challenging and pressing task, which is vital to reasoning-driven generation, and agentic tasks requiring long context and continual learning. In this paper, we explore how to unify reasoning-driven generation, reasoning-enhanced text representation and context compression tasks in one forward pass for LLMs. Through meta latent tokens and a unified generative, representative and compressive tuning approach, we propose a training framework named GRC that bridges the three tasks. The trained models can accomplish three objectives in a single forward pass while maintaining modular, LEGO-style flexibility during inference. This design greatly reduces the deployment effort for retrieval-augmented generation (RAG) and achieves efficient inference and three times data utilization during training. Furthermore, this framework design enables a new paradigm for text embedding: self-reason-latent embeds, and a new generation paradigm, latent memory-augmented generation, where compressed and internalized KV cache with O(1) length is used as the updatable memory. We also propose hybrid paged attention to speed up the inference of our models. Extensive experiments on reasoning-intensive retrieval benchmarks, generative tasks, document compression, latency evaluation, and RAG settings demonstrate the effectiveness of our method and may shed light on the truly unified model that can handle reasoning-driven generation, embedding and compression tasks seamlessly. 

---
# Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage 

**Authors**: Prasanjit Dubey, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2605.09986)  

**Abstract**: Training a language model on data scattered across bandwidth-limited nodes that cannot be centralized is a setting that arises in clinical networks, enterprise knowledge bases, and scientific consortia. We study the regime in which data must remain distributed across nodes, and ask what statistical guarantees are in principle achievable under explicit bandwidth budgets; we aim to characterize what is provably possible, not to demonstrate a deployment-ready system. Existing theory treats either training-time consistency or inference-time calibration in isolation, and none makes bandwidth a first-class statistical parameter. We analyze two protocols, Federated Probe-Logit Distillation (FPLD) for training and Federated Conformal RAG (FC-RAG) for inference, as the analytical vehicles for our results. Our first main result is an explicit high-probability KL-consistency rate for FPLD with simultaneous dependence on node count $K$, per-node sample size $n$, quantization budget $B$, probe-set size $m$, and vocabulary size $V$; bandwidth enters only through an exponentially vanishing quantization term. Our second main result is a distribution-free marginal-coverage bound for FC-RAG, whose novel retrieval-bandwidth slack $\Delta_{\mathrm{RAG}} = f_{\max}\sqrt{K^{-2}\sum_i v(B_i)}$ makes per-node retrieval bandwidth a first-class statistical parameter, with arithmetic aggregation across $K$ nodes shrinking the slack as $K^{-1/2}$ in the per-node-uniform regime. A Pinsker-type corollary composes the two bounds into an end-to-end coverage guarantee. Synthetic experiments verify the predicted scaling along the bounds' parameters; small-scale experiments on a GPT-2 testbed illustrate that the qualitative bandwidth-accuracy tradeoff survives on a real language model. A deployment-scale empirical evaluation is out of scope. 

---
# Position: Avoid Overstretching LLMs for every Enterprise Task 

**Authors**: Kuldeep Singh, Anson Bastos, Isaiah Onando Mulang'  

**Link**: [PDF](https://arxiv.org/pdf/2605.09365)  

**Abstract**: Enterprise workloads are dominated by deterministic, structured, and knowledge-dependent tasks operating under strict cost, latency, and reliability constraints. While these are often addressed through large language model (LLM) deployment or distillation into smaller models, we argue this is inefficient, unreliable, and misaligned with enterprise task structures. Instead, AI systems should treat language models as interfaces rather than monolithic engines, externalizing knowledge and computation into dedicated components for greater reliability, scalability, and transparency. Our theoretical evidences show that finite-capacity models cannot fully capture the breadth of knowledge required for enterprise tasks, creating inherent limits to efficiency and interpretability. Building on this, we take the position that language models should primarily be used for structured extraction in deterministic enterprise workflows, while computation and storage are delegated to knowledge bases and symbolic procedures. We formally demonstrate that such modular architectures are more reliable and maintainable than monolithic frameworks, offering a sustainable foundation for enterprise tasks. 

---
# Personalized Deep Research: A User-Centric Framework, Dataset, and Hybrid Evaluation for Knowledge Discovery 

**Authors**: Xiaopeng Li, Wenlin Zhang, Yingyi Zhang, Pengyue Jia, Yejing Wang, Yichao Wang, Yong Liu, Huifeng Guo, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.10530)  

**Abstract**: Deep Research agents driven by LLMs have automated the scholarly discovery pipeline, from planning and query formulation to iterative web exploration. Yet they remain constrained by a static, ``one-size-fits-all'' retrieval paradigm. Current systems fail to adaptively adjust the depth and breadth of exploration based on the user's existing expertise or latent interests, frequently resulting in reports that are either redundant for experts or overly dense for novices. To address this, we introduce Personalized Deep Research (PDR), a framework that integrates dynamic user context into the core retrieval-reasoning loop. Rather than treating personalization as a post-hoc formatting step, PDR unifies user profile modeling with iterative query development, dual-stage (private/public) retrieval, and context-aware synthesis. This allows the system to autonomously align research sub-goals with user intent and optimize the stopping criteria for evidence collection. To facilitate benchmarking, we release the PDR Dataset, covering four realistic user tasks, and propose a hybrid evaluation framework combining lexical metrics with LLM-based judgments to assess factual accuracy and personalization alignment. Experimental results against commercial baselines demonstrate that PDR significantly improves retrieval utility and report relevance, effectively bridging the gap between generic information retrieval and personalized knowledge acquisition. The resource is available to the public at this https URL. 

---
# Retrieval Mechanisms Surpass Long-Context Scaling in Time Series Forecasting 

**Authors**: Rishi Ahuja, Kumar Prateek, Simranjit Singh, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.08217)  

**Abstract**: Time Series Foundation Models (TSFMs) have borrowed the long context paradigm from natural language processing under the premise that feeding more history into the model improves forecast quality. But in stochastic domains, distant history is often just high-frequency noise, not signal. Hence, the proposed work tests whether this premise actually holds by running continuous context architectures (PatchTST included) through the ETTh1 benchmark. The obtained results contradict the premise: an inverse scaling law shows up clearly, with forecasting error rising as context gets longer. A 3,000-step window causes performance to drop by over 68%, evidence that attention mechanisms are poor at ignoring irrelevant historical volatility. Retrieval-Augmented Forecasting (RAFT) is evaluated as an alternative. RAFT achieves a mean squared error (MSE) of 0.379 with a fixed 720-step window and selective retrieval, outperforming both long-context configurations and zero-shot foundation models (Chronos, Moirai) despite requiring far less computation. In addition, the retrieval step injects only the most relevant historical segments as dynamic exogenous variables, which gives the model a context-informed inductive bias it cannot build on its own from raw sequences. Therefore, foundation models going forward need to shift architecturally toward selective retrieval. 

---
