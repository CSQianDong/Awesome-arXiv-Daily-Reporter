# ELF: Embedded Language Flows 

**Authors**: Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, Kaiming He  

**Link**: [PDF](https://arxiv.org/pdf/2605.10938)  

**Abstract**: Diffusion and flow-based models have become the de facto approaches for generating continuous data, e.g., in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today's leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that continuous DLMs can be made effective with minimal adaptation to the discrete domain. We propose Embedded Language Flows (ELF), a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG). Experiments show that ELF substantially outperforms leading discrete and continuous DLMs, achieving better generation quality with fewer sampling steps. These results suggest that ELF offers a promising path toward effective continuous DLMs. 

---
# WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation 

**Authors**: Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Yang JingYi, Penghui Yang, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, Yuhang Zang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10912)  

**Abstract**: Large language and vision-language models increasingly power agents that act on a user's behalf through command-line interface (CLI) harnesses. However, most agent benchmarks still rely on synthetic sandboxes, short-horizon tasks, mock-service APIs, and final-answer checks, leaving open whether agents can complete realistic long-horizon work in the runtimes where they are deployed. This work presents WildClawBench, a native-runtime benchmark of 60 human-authored, bilingual, multimodal tasks spanning six thematic categories. Each task averages roughly 8 minutes of wall-clock time and over 20 tool calls, and runs inside a reproducible Docker container hosting an actual CLI agent harness (OpenClaw, Claude Code, Codex, or Hermes Agent) with access to real tools rather than mock services. Grading is hybrid, combining deterministic rule-based checks, environment-state auditing of side effects, and an LLM/VLM judge for semantic verification. Across 19 frontier models, the best, Claude Opus 4.7, reaches only 62.2% overall under OpenClaw, while every other model stays below 60%, and switching harness alone shifts a single model by up to 18 points. These results show that long-horizon, native-runtime agent evaluation remains a far-from-resolved task for current frontier models. We release the tasks, code, and containerized tooling to support reproducible evaluation. 

---
# RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards 

**Authors**: Gaotang Li, Bhavana Dalvi Mishra, Zifeng Wang, Jun Yan, Yanfei Chen, Chun-Liang Li, Long T. Le, Rujun Han, George Lee, Hanghang Tong, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2605.10899)  

**Abstract**: Training deep research agents, namely systems that plan, search, evaluate evidence, and synthesize long-form reports, pushes reinforcement learning beyond the regime of verifiable rewards. Their outputs lack ground-truth answers, their trajectories span many tool-augmented decisions, and standard post-training offers little mechanism for turning past attempts into reusable experience. In this work, we argue that rubrics should serve not merely as final-answer evaluators, but as the shared interface that structures policy execution, judge feedback, and agent memory. Based on this view, we introduce RubricEM, a rubric-guided reinforcement learning framework that combines stagewise policy decomposition with reflection-based meta-policy evolution. RubricEM first makes research trajectories stage-aware by conditioning planning, evidence gathering, review, and synthesis on self-generated rubrics. It then assigns credit with Stage-Structured GRPO, which uses stagewise rubric judgments to provide denser semantic feedback for long-horizon optimization. In parallel, RubricEM trains a shared-backbone reflection meta-policy that distills judged trajectories into reusable rubric-grounded guidance for future attempts. The resulting RubricEM-8B achieves strong performance across four long-form research benchmarks, outperforming comparable open models and approaching proprietary deep-research systems. Beyond final performance, we perform thorough analyses to understand the key ingredients of RubricEM. 

---
# Grounded or Guessing? LVLM Confidence Estimation via Blind-Image Contrastive Ranking 

**Authors**: Reza Khanmohammadi, Erfan Miahi, Simerjot Kaur, Charese H. Smiley, Ivan Brugere, Kundan Thind, Mohammad M. Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10893)  

**Abstract**: Large vision-language models suffer from visual ungroundedness: they can produce a fluent, confident, and even correct response driven entirely by language priors, with the image contributing nothing to the prediction. Existing confidence estimation methods cannot detect this, as they observe model behavior under normal inference with no mechanism to determine whether a prediction was shaped by the image or by text alone. We introduce BICR (Blind-Image Contrastive Ranking), a model-agnostic confidence estimation framework that makes this contrast explicit during training by extracting hidden states from a frozen LVLM twice: once with the real image-question pair, and once with the image blacked out while the question is held fixed. A lightweight probe is trained on the real-image hidden state and regularized by a ranking loss that penalizes higher confidence on the blacked-out view, teaching it to treat visual grounding as a signal of reliability at zero additional inference cost. Evaluated across five modern LVLMs and seven baselines on a benchmark covering visual question answering, object hallucination detection, medical imaging, and financial document understanding, BICR achieves the best cross-LVLM average on both calibration and discrimination simultaneously, with statistically significant discrimination gains robust to cluster-aware analysis at 4-18x fewer parameters than the strongest probing baseline. 

---
# Neural at ArchEHR-QA 2026: One Method Fits All: Unified Prompt Optimization for Clinical QA over EHRs 

**Authors**: Abrar Majeedi, Viswanatha Reddy Gajjala, Sai Prasanna Teja Reddy Bogireddy, Siddhant Rai  

**Link**: [PDF](https://arxiv.org/pdf/2605.10877)  

**Abstract**: Automated question answering (QA) over electronic health records (EHRs) demands precise evidence retrieval, faithful answer generation, and explicit grounding of answers in clinical notes. In this work, we present Neural1.5, our method for the ArchEHR-QA 2026 shared task at CL4Health@LREC 2026, which comprises four subtasks: question interpretation, evidence identification, answer generation, and evidence alignment. Our approach decouples the task into independent, modular stages and employs DSPy"s MIPROv2 optimizer to automatically discover high-performing prompts, jointly tuning instructions and few-shot demonstrations for each stage. Within every stage, self-consistency voting over multiple stochastic inference runs suppresses spurious errors and improves reliability, while stage-specific verification mechanisms (e.g., self-reflection and chain-of-verification for alignment) further refine output quality. Among all teams that participated in all four subtasks, our method ranks second overall (mean rank 4.00), placing 4th, 1st, 4th, and 7th on Subtasks 1-4, respectively. These results demonstrate that systematic, per-stage prompt optimization combined with self-consistency mechanisms is a cost-effective alternative to model fine-tuning for multifaceted clinical QA. 

---
# DGPO: Beyond Pairwise Preferences with Directional Consistent Groupwise Optimization 

**Authors**: Mengyi Deng, Zhiwei Li, Xin Li, Tingyu Zhu, Yulan Yuan, Zhijiang Guo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10863)  

**Abstract**: Although Large Language Models (LLMs) have made remarkable progress, current preference optimization methods still struggle to align directional consistency while preserving reasoning diversity. To address this limitation, we propose Directional-Groupwise Preference Optimization (DGPO), a lightweight framework that aggregates supervision signals at the group level and explicitly models direction-aware alignment through multi-candidate comparisons. DGPO organizes forward and reverse question-answer instances into structured sets and optimizes a margin-based likelihood objective that separates coherent reasoning paths from inconsistent alternatives. This group-wise formulation captures richer relative information than pairwise objectives and reinforces consistency across diverse reasoning pathways. Empirical results show that our constructed reverse data yields a 3.2% average improvement across five benchmarks, while DGPO further delivers consistent gains across multiple datasets and model families, achieving average accuracy improvements of up to 3.6%. 

---
# RUBEN: Rule-Based Explanations for Retrieval-Augmented LLM Systems 

**Authors**: Joel Rorseth, Parke Godfrey, Lukasz Golab, Divesh Srivastava, Jarek Szlichta  

**Link**: [PDF](https://arxiv.org/pdf/2605.10862)  

**Abstract**: This paper demonstrates RUBEN, an interactive tool for discovering minimal rules to explain the outputs of retrieval-augmented large language models (LLMs) in data-driven applications. We leverage novel pruning strategies to efficiently identify a minimal set of rules that subsume all others. We further demonstrate novel applications of these rules for LLM safety, specifically to test the resiliency of safety training and effectiveness of adversarial prompt injections. 

---
# Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding 

**Authors**: Jianzhu Bao, Haozhen Zhang, Kuicai Dong, Bozhi Wu, Sarthak Ketanbhai Modi, Zi Pong Lim, Yon Shin Teo, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10855)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable progress in chart understanding, largely driven by supervised fine-tuning (SFT) on increasingly large synthetic datasets. However, scaling SFT data alone is inefficient and overlooks a key property of charts: charts are programmatically generated visual artifacts, where small, code-controlled visual changes can induce drastic shifts in semantics and correct answers. Learning this counterfactual sensitivity requires VLMs to discriminate fine-grained visual differences, yet standard SFT treats training instances independently and provides limited supervision to enforce this behavior. To address this, we introduce ChartCF, a data-efficient training framework designed to enhance counterfactual sensitivity. ChartCF consists of: (1) a counterfactual data synthesis pipeline via code modification, (2) a chart similarity-based data selection strategy that filters overly difficult samples for improved training efficiency, and (3) multimodal preference optimization across both textual and visual modalities. Experiments on five benchmarks show that ChartCF achieves superior or comparable performance to strong chart-specific VLMs while using significantly less training data. 

---
# Grounded Satirical Generation with RAG 

**Authors**: Oona Itkonen, Yuxin Su, Linyao Du, Ona De Gibert  

**Link**: [PDF](https://arxiv.org/pdf/2605.10853)  

**Abstract**: Humor generation remains challenging task for Large Language Models (LLMs), due to their subjective nature. We focus on satire, a form of humor strongly shaped by context. In this work, we present a novel pipeline for grounded satire generation that uses Retrieval-Augmented Generation (RAG) over current news to produce satirical dictionary definitions in the Finnish context. We also introduce a new task-specific evaluation framework and annotate 100 generated definitions with six human annotators, enabling analysis across multiple experimental conditions, including cultural background, source-word type, and the presence or absence of RAG. Our results show that the generated definitions are perceived as more political than humorous. Both topic-based word selection and RAG improve the political relevance of the outputs, but neither yields clear gains in humor generation. In addition, our LLM-as-a-judge evaluation of five state-of-the-art models indicates that LLMs correlate well with human judgments on political relevance, but perform poorly on humor. We release our code and annotated dataset to support further research on grounded satire generation and evaluation. 

---
# Training-Free Cultural Alignment of Large Language Models via Persona Disagreement 

**Authors**: Huynh Trung Kiet, Dao Sy Duy Minh, Tuan Nguyen, Chi-Nguyen Tran, Phu-Hoa Pham, Nguyen Lam Phu Quy, Anh Han, Long Tran-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2605.10843)  

**Abstract**: Large language models increasingly mediate decisions that turn on moral judgement, yet a growing body of evidence shows that their implicit preferences are not culturally neutral. Existing cultural alignment methods either require per-country preference data and fine-tuning budgets or assume white-box access to model internals that commercial APIs do not expose. In this work, we focus on this realistic black-box, public-data-only regime and observe that within-country sociodemographic disagreement, not consensus, is the primary steering signal. We introduce DISCA (Disagreement-Informed Steering for Cultural Alignment), an inference-time method that instantiates each country as a panel of World-Values-Survey-grounded persona agents and converts their disagreement into a bounded, loss-averse logit correction. Across 20 countries and 7 open-weight backbones (2B--70B), DISCA reduces cultural misalignment on MultiTP by 10--24% on the six backbones >=3.8B, and 2--7% on open-ended scenarios, without changing any weights. Our results suggest that inference-time calibration is a scalable alternative to fine-tuning for serving the long tail of global moral preferences. 

---
# Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents 

**Authors**: Shijue Huang, Hangyu Guo, Chenxin Li, Junting Lu, Xinyu Geng, Zhaochen Su, Zhenyu Li, Shuang Chen, Hongru Wang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2605.10832)  

**Abstract**: Multimodal deep search requires an agent to solve open-world problems by chaining search, tool use, and visual reasoning over evolving textual and visual context. Two bottlenecks limit current systems. First, existing tool-use harnesses treat images returned by search, browsing, or transformation as transient outputs, so intermediate visual evidence cannot be re-consumed by later tools. Second, training data is usually built by fixed curation recipes that cannot track the target agent's evolving capability. To address these challenges, we first introduce a visual-native agent harness centered on an image bank reference protocol, which registers every tool-returned image as an addressable reference and makes intermediate visual evidence reusable by later tools. On top of this harness, On-policy Data Evolution (ODE) runs a closed-loop data generator that refines itself across rounds from rollouts of the policy being trained. This per-round refinement makes each round's data target what the current policy still needs to learn. The same framework supports both diverse supervised fine-tuning data and policy-aware reinforcement learning data curation, covering the full training lifecycle of the target agent. Across 8 multimodal deep search benchmarks, ODE improves the Qwen3-VL-8B agent from 24.9% to 39.0% on average, surpassing Gemini-2.5 Pro in standard agent-workflow setting (37.9%). At 30B, ODE raises the average score from 30.6% to 41.5%. Further analyses validate the effectiveness of image-bank reuse, especially on complex tasks requiring iterative visual refinement, while rollout-feedback evolution yields more grounded SFT traces and better policy-matched RL tasks than static synthesis. 

---
# Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish 

**Authors**: Fred Philippy, Siwen Guo, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2605.10714)  

**Abstract**: Cross-lingual transfer has become a central paradigm for extending natural language processing (NLP) technologies to low-resource languages. By leveraging supervision from high-resource languages, multilingual language models can achieve strong task performance with little or no labeled target-language data. However, it remains unclear to what extent cross-lingual transfer can substitute for language-specific efforts. In this paper, we synthesize prior research findings and data collection results on Luxembourgish, which, despite its typological proximity to high-resource languages and its presence in a multilingual context, remains insufficiently represented in modern NLP technologies. Across findings, we observe a fundamental interdependence between cross-lingual transfer and language-specific efforts. Cross-lingual transfer can substantially improve target-language performance, but its success depends critically on the availability of sufficiently high-quality, task-aligned target-language data. At the same time, such resources, particularly in low-resource settings, are typically too limited in scale to drive strong performance on their own. Instead, such resources reach their full potential only when leveraged within a cross-lingual framework. We therefore argue that cross-lingual transfer and language-specific efforts should not be viewed as competing alternatives. Instead, they function as complementary components of a sustainable low-resource NLP pipeline. Based on these insights, we provide practical guidelines for integrating and balancing cross-lingual transfer with language-specific development in sustainable low-resource NLP pipelines. 

---
# Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions 

**Authors**: Diancheng Kang, Zheyuan Liu, Ningshan Ma, Yue Huang, Zhaoxuan Tan, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10664)  

**Abstract**: Activation steering controls language model behavior by adding directions to internal representations at inference time, but standard residual-stream steering can fail in stateful dialogue. We identify KV-cache contamination as a key failure mode: steered token states are stored and repeatedly reused, turning a local perturbation into cumulative coherence degradation. To address this challenge, we propose Gated Cropped Attention-Delta steering (GCAD), which extracts steering signals from system-prompt contributions to self-attention and applies them with token-level gating. Across persona-steering experiments, GCAD preserves trait control while substantially improving long-horizon coherence. On the main multi-turn benchmark, GCAD improves average coherence drift from -18.6 to -1.9 and raises turn-10 trait expression from 78.0 to 93.1. These results suggest that activation steering becomes more reliable when interventions follow the prompt-mediated pathways that models already use for behavioral control. 

---
# When Can Digital Personas Reliably Approximate Human Survey Findings? 

**Authors**: Mumin Jia, Yilin Chen, Divya Sharma, Jairo Diaz-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.10659)  

**Abstract**: Digital personas powered by Large Language Models (LLMs) are increasingly proposed as substitutes for human survey respondents, yet it remains unclear when they can reliably approximate human survey findings. We answer this question using the LISS panel, constructing personas from respondents' background variables and pre-2023 survey histories, then testing them against the same respondents' held-out post-cutoff answers. Across four persona architectures, three LLMs, and two prediction tasks, we assess performance at the question, respondent, distributional, equity, and clustering levels. Digital personas improve alignment with human response distributions, especially in domains tied to stable attributes and values, but remain limited for individual prediction and fail to recover multivariate respondent structure. Retrieval-augmented architectures provide the clearest gains, but performance depends more on human response structure than on model choice: personas perform best for low-variability questions and common respondent patterns, and worst for subjective, heterogeneous, or rare responses. Our results provide practical guidance on when digital personas could be appropriate for survey research and when human validation remains necessary. 

---
# A Single-Layer Model Can Do Language Modeling 

**Authors**: Zanmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10643)  

**Abstract**: Modern language models scale depth by stacking layers, each holding its own state - a per-layer KV cache in transformers, a per-layer matrix in Mamba, Gated DeltaNet (GDN), RWKV, and xLSTM. Biological systems lean heavily on recurrence rather than on stacking. We ask how far that shape can go on language modeling. We propose Grounded Prediction Networks (GPN): one state vector revisited at every step through a single recurrent block - one FFN, one shared matrix memory. At 130M parameters, a 1-layer GPN+M reaches FineWeb-Edu perplexity 18.06, within 13% of a 12-layer Transformer++ (16.05) and 18% of a 10-layer GDN (15.34); a 2-layer variant closes the gap to 6%/11%. We do not match the deep baselines. Because the working context is a single vector, we can directly inspect its geometry: a persistent default-token direction, a content-bearing horizon of tens of tokens, and memory heads that split spontaneously into fast and slow retention pools. 

---
# Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm 

**Authors**: Haoyu Wang, Yifan Shang, Zhongxiang Sun, Weijie Yu, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10640)  

**Abstract**: Continual Pre-Training (CPT) is essential for enabling Language Models (LMs) to integrate new knowledge without erasing old. While classical CPT techniques like data replay have become the standard paradigm, the mechanisms underlying how LMs acquire and retain facts over time, termed as continual Factual Knowledge Acquisition (cFKA), remain unclear. In this work, we present a theoretical framework that characterizes the training dynamics of cFKA using a single-layer Transformer, offering a unified explanation for the behavior of representative CPT methods. Our analysis reveals that regularization-based methods merely adjust the convergence rate of parameters without altering the inherent forgetting tendency, whereas data replay methods succeed in shifting convergence dynamics and stabilizing pretrained knowledge. Building on these insights, we propose a novel generative data replay approach, called \textbf{S}electing \textbf{T}okens via attenti\textbf{O}n \textbf{C}ontribution~(STOC), which identifies influential factual snippets to guide replay data generation. Extensive experiments on both synthetic and real-world datasets validate our findings and demonstrate that STOC effectively enhances cFKA by mitigating catastrophic forgetting. 

---
# Intrinsic Guardrails: How Semantic Geometry of Personality Interacts with Emergent Misalignment in LLMs 

**Authors**: Krishak Aneja, Manas Mittal, Anmol Goel, Ponnurangam Kumaraguru, Vamshi Krishna Bonagiri  

**Link**: [PDF](https://arxiv.org/pdf/2605.10633)  

**Abstract**: Fine-tuning Large Language Models (LLMs) on benign narrow data can sometimes induce broad harmful behaviors, a vulnerability termed emergent misalignment (EM). While prior work links these failures to specific directions in the activation space, their relationship to the model's broader persona remains unexplored. We map the latent personality space of LLMs through established psychometric profiles like the Big Five, Dark Triad, and LLM-specific behaviors (e.g. evil, sycophancy), and show that the semantic geometry is highly stable across aligned models and their corrupted fine-tunes. Through causal interventions, we find that directions isolating social valence, such as the 'Evil' persona vector, and a Semantic Valence Vector (SVV) that we introduce, function as intrinsic guardrails: ablating them drives the misalignment rates above $40$%, while amplifying them suppresses the failure mode to less than $3$%. Leveraging the structural stability of the personality space, we show that vectors extracted $\textit{a priori}$ from an instruct-tuned model transfer zero-shot to successfully regulate EM in corrupted fine-tunes. Overall, our findings suggest that harmful fine-tuning does not overwrite a model's internal representation of personality, allowing conserved representations to serve as robust, cross-distribution guardrails. 

---
# Interpretable Coreference Resolution Evaluation Using Explicit Semantics 

**Authors**: Bruno Gatti, Giuliano Martinelli, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2605.10627)  

**Abstract**: Coreference resolution is typically evaluated using aggregate statistical metrics such as CoNLL-F1, which measure structural overlap between predicted and gold clusters. While widely used, these metrics offer limited diagnostic insights, penalizing errors without revealing whether a system struggles with specific semantic categories, such as people, locations, or events, and making it difficult to interpret model capabilities or derive actionable improvements. We address this gap by introducing a semantically-enhanced evaluation framework for coreference resolution. Our approach overlays Concept and Named Entity Recognition (CNER) onto coreference outputs, assigning semantic labels to nominal mentions and propagating them to entire coreference clusters. This enables the computation of typed scores aimed at evaluating mention extraction and linking capabilities stratified by semantic class. Across our experiments on OntoNotes, LitBank, and PreCo, we show that our framework uncovers systematic weaknesses that remain obscured by aggregate metrics. Furthermore, we demonstrate that these diagnostics can be used to design targeted, low-cost data augmentation strategies, achieving measurable out-of-domain improvements. 

---
# Responsible Benchmarking of Fairness for Automatic Speech Recognition 

**Authors**: Felix Herron, Ange Richard, François Portet, Alexandre Allauzen, Solange Rossato  

**Link**: [PDF](https://arxiv.org/pdf/2605.10615)  

**Abstract**: Many studies have shown automatic speech processing (ASR) systems have unequal performance across speakergroups (SG's). However, the manner in which such studies arrive at this conclusion is inconsistent. To pave the wayfor more reliable results in future studies, we lay out best practices for benchmarking ASR fairness based on literaturefrom machine learning fairness, social sciences, and speech science. We first describe the importance of preciselythe fairness hypothesis being interrogated, and tailoring fairness metrics to apply specifically to said this http URL then examine several benchmarks used to rate ASR systems on fairness and discuss how their results can bemisconstrued without assiduous oversight into the intersections between SG's. We find that evaluating fairnessbased on single heterogeneous SG's, such as they are defined in fairness benchmarks, can lead to misidentifyingwhich SG's are actually being mistreated by ASR systems. We advocate for as fine-grained an analysis as possibleof the intersectionality of as many demographic variables as are available in the metadata of fairness corpora in orderto tease out such spurious correlations 

---
# Measuring Embedding Sensitivity to Authorial Style in French: Comparing Literary Texts with Language Model Rewritings 

**Authors**: Benjamin Icard, Lila Sainero, Alice Breton, Evangelia Zve, Jean-Gabriel Ganascia  

**Link**: [PDF](https://arxiv.org/pdf/2605.10606)  

**Abstract**: Large language models (LLMs) can convincingly imitate human writing styles, yet it remains unclear how much stylistic information is encoded in embeddings from any language model and retained after LLM rewriting. We investigate these questions in French, using a controlled literary dataset to quantify the effect of stylistic variation via changes in embedding dispersion. We observe that embeddings reliably capture authorial stylistic features and that these signals persist after rewriting, while also exhibiting LLM-specific patterns. These analytical results offer promising directions for authorship imitation detection in the era of language models. 

---
# Where do aspectual variants of light verb constructions belong? 

**Authors**: Aggeliki Fotopoulou, Eric Laporte, Takuya Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2605.10605)  

**Abstract**: Expressions with an aspectual variant of a light verb, e.g. 'take on debt' vs. 'have debt', are frequent in texts but often difficult to classify between verbal idioms, light verb constructions or compositional phrases. We investigate the properties of such expressions with a disputed membership and propose a selection of features that determine more satisfactory boundaries between the three categories in this zone, assigning the expressions to one of them. 

---
# VISTA: A Generative Egocentric Video Framework for Daily Assistance 

**Authors**: Yu-Hsiang Liu, Yu-Chien Tang, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2605.10579)  

**Abstract**: Training AI agents to proactively assist humans in daily activities, from routine household tasks to urgent safety situations, requires large-scale visual data. However, capturing such scenarios in the real world is often difficult, costly, or unsafe, and physics-based simulators lack the visual fidelity needed to transfer learned behaviors to real settings. Therefore, we introduce VISTA, a video synthesis system that produces high-fidelity egocentric videos as training and evaluation data for AI agents. VISTA employs a 5-step script generation pipeline with causal reverse reasoning to create diverse, logically grounded intervention modes. These scenarios span two levels of agent autonomy: reactive and proactive. In reactive modes, the user explicitly asks the agent for help. In proactive modes, the agent offers help without receiving a direct request. We further divide proactive modes into explicit and implicit types. In explicit proactive scenarios, the user is aware of needing help but does not directly address the agent. In implicit proactive scenarios, the agent intervenes before the user even realizes that help is needed. VISTA allows users to customize and refine scenarios to generate video benchmarks for daily tasks, offering a scalable and controllable alternative to real-world data collection for training and evaluating AI agents in realistic environments. 

---
# ThreatCore: A Benchmark for Explicit and Implicit Threat Detection 

**Authors**: Davide Bruni, Carlo Bardazzi, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10563)  

**Abstract**: Threat detection in Natural Language Processing lacks consistent definitions and standardized benchmarks, and is often conflated with broader phenomena such as toxicity, hate speech, or offensive language. In this work, we introduce ThreatCore, a public available benchmark dataset for fine-grained threat detection that distinguishes between explicit threats, implicit threats, and non-threats. The dataset is constructed by aggregating multiple publicly available resources and systematically re-annotating them under a unified operational definition of threat, revealing substantial inconsistencies across existing labels. To improve the coverage of underrepresented cases, particularly implicit threats, we further augment the dataset with synthetic examples, which are manually validated using the same annotation protocol adopted for the re-annotation of the public datasets, ensuring consistency across all data sources. We evaluate Perspective API, zero-shot classifiers, and recent language models on ThreatCore, showing that implicit threats remain substantially harder to detect than explicit ones. Our results also indicate that incorporating Semantic Role Labeling as an intermediate representation can improve performance by making the structure of harmful intent more explicit. Overall, ThreatCore provides a more consistent benchmark for studying fine-grained threat detection and highlights the challenges that current models still face in identifying indirect expressions of harmful intent. 

---
# ICT-NLP at SemEval-2026 Task 3: Less Is More -- Multilingual Encoder with Joint Training and Adaptive Ensemble for Dimensional Aspect Sentiment Regression 

**Authors**: Liyuan Huang, Jiawei He, Wutao Shen, Lin Li, Jin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10560)  

**Abstract**: This paper describes our system to SemEval-2026 Task 3 Track A Subtask 1 on Dimensional Aspect Sentiment Regression (DimASR). We propose a lightweight and resource-efficient system built entirely on multilingual pre-trained encoders, without relying on LLMs or external corpora. We adopt joint multilingual and multi-domain training to facilitate cross-lingual transfer and alleviate data sparsity, introduce a bounded regression transformation that improves training stability while constraining predictions within the valid range, and employ an adaptive ensemble strategy via subset search to reduce prediction variance. Experimental results demonstrate that our system achieves strong and consistent performance, ranking 1st on zho-res, 2nd on zho-lap, and 3rd on jpn-hot, with all remaining datasets placed within the top half of participating teams. 

---
# Multi-domain Multi-modal Document Classification Benchmark with a Multi-level Taxonomy 

**Authors**: Denghao Ma, Qing Liu, Zulong Chen, Chuanfei Xu, Jia Xu, Zhibo Yang, Zhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.10550)  

**Abstract**: Document classification forms the backbone of modern enterprise content management, yet existing benchmarks remain trapped in oversimplified paradigms -- single domain settings with flat label structures -- that bear little resemblance to the hierarchical, multi-modal, and cross-domain nature of real-world business documents. This gap not only misrepresents practical complexity but also stifles progress toward industrially viable document intelligence. To bridge this gap, we construct the first Multi-level, Multi-domain, Multi-modal document classification Benchmark (MMM-Bench). MMM-Bench includes (1) a deeply hierarchical taxonomy spanning five levels that capture the authentic organizational logic of business documentation; and (2) 5,990 real-world multi-modal documents meticulously curated from 12 commercial domains in Alibaba. Each document is manually annotated with a complete hierarchical path by domain experts. We establish comprehensive baselines on MMM-Bench, which consists of open-weight models and API-based models. Through systematic experiments, we identify four fundamental challenges within MMM-Bench and propose corresponding insights. To provide a solid foundation for advancing research in multi-level, multi-domain document classification, we release all of the data and the evaluation toolkit of MMM-Bench at this https URL. 

---
# Where Does Long-Context Supervision Actually Go? Effective-Context Exposure Balancing 

**Authors**: Jinchang Zhu, Jindong Li, Chengyu Zou, Rong Fu, Chao Wang, Haowei He, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10544)  

**Abstract**: Long-context adaptation is often viewed as window scaling, but this misses a token-level supervision mismatch: in packed training with document masking, each target token's effective context remains short. We introduce EXACT, a supervision-allocation objective that assigns extra weight to long effective-context targets by inverse frequency within the long tail. Across seven Qwen/LLaMA CPT configurations, EXACT improves all 28 trained/extrapolated NoLiMa and RULER comparisons. On Qwen2.5-0.5B, NoLiMa improves by +10.09 (trained) and +5.34 (extrapolated); RULER by +10.69 and +5.55. On LLaMA-3.2-3B, RULER improves by +17.91 and +16.11. Standard QA/reasoning are preserved (+0.24 macro change across six benchmarks). A distance-resolved probe shows gains arise when evidence is thousands of tokens away, while short cases remain unchanged. Results support a supervision-centric thesis: long-context adaptation depends on how strongly training supervises long-context predictions. 

---
# Mela: Test-Time Memory Consolidation based on Transformation Hypothesis 

**Authors**: Lungchuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.10537)  

**Abstract**: Memory consolidation, the process by which transient experiences are transformed into stable, structured representations, is a foundational organizing principle in the human brain, yet it remains largely unexplored as a design principle for modern sequence models. In this work, we leverage established neuroscientific theories of memory consolidation and cross-frequency coupling to propose the Hierarchical Memory Module (HMM), a neural memory architecture composed of two functionally distinct sub-modules that operate at different update frequencies. Inspired by the transformation hypothesis, the low-frequency sub-module produces high-level representations that capture abstract, gist-level knowledge, while the high-frequency sub-module produces fine-grained representations that preserve richer episodic detail. The final memory output is dynamically reconstructed as a context-dependent combination of both representations, analogous to the reconstructive nature of human memory retrieval. We integrate HMM into a Transformer-based language decoder to form Mela, a family of memory-augmented language models that perform online memory consolidation at test time. To further exploit the multi-granularity memory representations produced by HMM, we introduce MemStack, a method that distributes different levels of memory features across the early layers of the decoder without introducing additional tokens. Experiments on language modeling demonstrate that Mela outperforms Transformer baselines across all the model sizes. Moreover, with the pretrained context length fixed at 4K, Mela maintains performance on significantly longer contexts, whereas Transformer baselines degrade rapidly beyond their training length. Extensive ablation studies validate the contribution of each component and provide guidance for practical configuration. 

---
# Infinite Mask Diffusion for Few-Step Distillation 

**Authors**: Jaehoon Yoo, Wonjung Kim, Chanhyuk Lee, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2605.10518)  

**Abstract**: Masked Diffusion Models (MDMs) have emerged as a promising alternative to autoregressive models in language modeling, offering the advantages of parallel decoding and bidirectional context processing within a simple yet effective framework. Specifically, their explicit distinction between masked tokens and data underlies their simple framework and effective conditional generation. However, MDMs typically require many sampling iterations due to factorization errors stemming from simultaneous token updates. We observe that a theoretical lower bound of the factorization error exists, which standard MDMs cannot reduce due to their use of a deterministic single-state mask. In this paper, we propose the Infinite Mask Diffusion Model (IMDM), which introduces a stochastic infinite-state mask to mitigate the theoretical bound while directly inheriting the benefits of MDMs, including the compatibility with pre-trained weights. We empirically demonstrate that MDM fails to perform few-step generation even in a simple synthetic task due to the factorization error bound, whereas IMDM can find an efficient solution for the same task. Finally, when equipped with appropriate distillation methods, IMDM surpasses existing few-step distillation methods at small step counts on LM1B and OpenWebText. Code is available at this https URL. 

---
# Learning Less Is More: Premature Upper-Layer Attention Specialization Hurts Language Model Pretraining 

**Authors**: Jinchang Zhu, Jindong Li, Yuwen Hao, Chengyu Zou, Rong Fu, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10504)  

**Abstract**: A causal-decoder block is hierarchical: lower layers build the residual basis that upper layers attend over. We identify a failure mode in GPT pretraining: upper layers commit to sharp attention patterns before lower-layer features stabilize. We call this premature upper-layer attention specialization. Temporarily slowing only upper-layer Q/K projections during early training improves final perplexity and downstream accuracy without altering other parameters; it prevents upper attention from collapsing onto an immature residual basis. In LLaMA-style blocks, the same intervention is nearly unnecessary. Through ablations, we isolate multiplicative gated FFNs (not RMSNorm or bias removal) as the component that suppresses the upstream residual writes driving the failure. A pathwise analysis unifies both findings: the learning-rate intervention reduces a step-size factor, while gated FFNs reduce a residual-energy factor on the same growth pathway. Our results identify upper-layer Q/K timing as a concrete interaction point between decoder architecture and optimization. 

---
# DeepRefine: Agent-Compiled Knowledge Refinement via Reinforcement Learning 

**Authors**: Haoyu Huang, Jiaxin Bai, Shujie Liu, Yang Wei, Hong Ting Tsang, Yisen Gao, Zhongwei Xie, Yufei Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.10488)  

**Abstract**: Agent-compiled knowledge bases provide persistent external knowledge for large language model (LLM) agents in open-ended, knowledge-intensive downstream tasks. Yet their quality is systematically limited by \emph{incompleteness}, \emph{incorrectness}, and \emph{redundancy}, manifested as missing evidence or cross-document links, low-confidence or imprecise claims, and ambiguous or coreference resolution issues. Such defects compound under iterative use, degrading retrieval fidelity and downstream task performance. We present \textbf{DeepRefine}, a general LLM-based reasoning model for \emph{agent-compiled knowledge refinement} that improves the quality of any pre-constructed knowledge bases with user queries to make it more suitable for the downstream tasks. DeepRefine performs multi-turn interactions with the knowledge base and conducts abductive diagnosis over interaction history, localizes likely defects, and executes targeted refinement actions for incremental knowledge base updates. To optimize refinement policies of DeepRefine without gold references, we introduce a Gain-Beyond-Draft (GBD) reward and train the reasoning process end-to-end via reinforcement learning. Extensive experiments demonstrate consistent downstream gains over strong baselines. 

---
# Coherency through formalisations of Structured Natural Language, A case study on FRETish 

**Authors**: Joost J. Joosten, Marina López Chamosa, Sofía Santiago Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2605.10462)  

**Abstract**: Formalisation is the process of writing system requirements in a formal language. These requirements mostly originate in Natural Language. In the field of Formal Methods, formalisation is often identified as one of the most delicate and complicated steps in the verification process. Not seldomly, formalisation tools and environments choose various levels of requirement descriptions: Natural Language, Technical Language, Diagram Representations and Formal Language, to mention a few. In the literature, there are various maxims and principles of good practice to guide the process of requirement formalisation. In this paper we propose a new guideline: Coherency through Formalisations. The guideline states that the different levels of formalisation mentioned above should roughly follow the same logical structure. The principle seems particularly relevant in the setting where LLMs are prompted to perform reasoning tasks that can be checked by formal tools using Structured Natural Language to act as an intermediate layer bridging both paradigms. In the light of coherency, we analyze NASA's Formal Requirement Elicitation Tool FRET and propose an alternative automated translation of the Controlled Natural Language FRETish to the formal language of MTL. We compare our translation to the original translation and prove equivalence using model checking. Some statistics are performed which seem to favor the new translation. As expected, the translation process yielded interesting reflections and revealed inconsistencies which we present and discuss. 

---
# Can Language Models Analyze Data? Evaluating Large Language Models for Question Answering over Datasets 

**Authors**: Andreas Xenofontos, Pavlos Fafalios  

**Link**: [PDF](https://arxiv.org/pdf/2605.10419)  

**Abstract**: This paper investigates the effectiveness of large language models (LLMs) in answering questions over datasets. We examine their performance in two scenarios: (a) directly answering questions given a dataset file as input, and (b) generating SQL queries to answer questions given the schema of a relational database. We also evaluate the impact of different prompting strategies on model performance. The study includes both state-of-the-art LLMs and smaller language models that require fewer resources and operate at lower computational and financial cost. Experiments are conducted on two datasets containing questions of varying difficulty. The results demonstrate the strong performance of large LLMs, while highlighting the limitations of smaller, more cost-efficient models. These findings contribute to a better understanding of how LLMs can be utilized in data analytics tasks and their associated limitations. 

---
# Aligning LLM Uncertainty with Human Disagreement in Subjectivity Analysis 

**Authors**: Junyu Lu, Deyi Ji, Xuanyi Liu, Lanyun Zhu, Bo Xu, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10415)  

**Abstract**: Large language models for subjectivity analysis are typically trained with aggregated labels, which compress variations in human judgment into a single supervision signal. This paradigm overlooks the intrinsic uncertainty of low-agreement samples and often induces overconfident predictions, undermining reliability and generalization in complex subjective settings. In this work, we advocate uncertainty-aware subjectivity analysis, where models are expected to make predictions while expressing uncertainty that reflects human disagreement. To operationalize this perspective, we propose a two-phase Disagreement Perception and Uncertainty Alignment (DPUA) framework. Specifically, DPUA jointly models label prediction, rationale generation, and uncertainty expression under an uncertainty-aware setting. In the disagreement perception phase, adaptive decoupled learning enhances the model's sensitivity to disagreement-related cues while preserving task performance. In the uncertainty alignment phase, GRPO-based reward optimization further improves uncertainty-aware reasoning and aligns the model's confidence expression with the human disagreement distribution. Experiments on three subjectivity analysis tasks show that DPUA preserves task performance while better aligning model uncertainty with human disagreement, mitigating overconfidence on boundary samples, and improving out-of-distribution generalization. 

---
# Phoenix-VL 1.5 Medium Technical Report 

**Authors**: Team Phoenix, Arka Ray, Askar Ali Mohamed Jawad, Biondi Lee, Elijah Seah, Eva Lim, Fiona Teo, Grace Toh, Guang Xiang Teo, Jun En Tan, Jia Hui Bong, Jiale Wang, Jonathan Ng, Justin Tan, Kai Zhe Yew, Matthew Ong, Shun Yi Yeo, Wen Jett Lam, Wen Xiu Tan, Ze Yu Zhang, Gee Wah Ng, Chee Wee Ang, Mistral AI, Adrien Sadé, Guillaume Kunsch, Jia Sin Loh, Nicolas Schuhl, Rupert Menneer, Umar Jamil, Vincent Maladière, Yimu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.10391)  

**Abstract**: We introduce Phoenix-VL 1.5 Medium, a 123B-parameter natively multimodal and multilingual foundation model, adapted to regional languages and the Singapore context. Developed as a sovereign AI asset, it demonstrates that deep domain adaptation can be achieved with minimal degradation to broad-spectrum intelligence and alignment. Continued pretraining was performed on Mistral Medium 3.1 using a localized 1-trillion tokens multimodal corpus, followed by a 250-billion tokens long-context extension phase. Subsequent post-training incorporated a novel human-annotated Singapore multimodal dataset and curated textual corpus on Singapore culture, knowledge, and legislation, totaling 22-billion tokens. An additional 5 billion tokens of model alignment was performed through Online Direct Preference Optimization. Phoenix-VL 1.5 Medium achieves state-of-the-art performance for its size on Singapore multimodal, legal, and government policy benchmarks while remaining globally competitive on general multimodal intelligence, multilingual, and STEM benchmarks. We also introduce a novel evaluation suite encompassing localized knowledge benchmarks and an institutionally aligned model behavior and safety framework. We report the data curation principles, training methodology, and highlight benchmark and inference performance. 

---
# Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness 

**Authors**: Ivo Petrov, Jasper Dekoninck, Dimitar I. Dimitrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2605.10379)  

**Abstract**: Large language models (LLMs) have become capable mathematical problem-solvers, often producing correct proofs for challenging problems. However, correctness alone is not sufficient: mathematical proofs should also be clear, concise, insightful, and transferable to other problems. While this proof quality is subjective and depends on the reader and context, many of its components are concrete and broadly valued. In this work, we identify such components and introduce ProofRank, a benchmark curated from challenging mathematical competitions. ProofRank evaluates several scalable proxies of proof quality: (i) conciseness, measuring whether proofs avoid unnecessary steps; (ii) computational ease, measuring the extent to which a proof relies on tedious calculations; (iii) cognitive simplicity, measuring how accessible the used proof techniques are; (iv) diversity, measuring how varied a model's proofs for a single problem are; and (v) adaptivity, measuring whether a model can follow a specified proof technique. Across models, we find substantial differences in proof quality that are not captured by correctness-only benchmarks. We also observe significant trade-offs between proof-quality metrics and correctness, suggesting that future evaluations of mathematical reasoning should measure how useful LLM-generated proofs are. 

---
# An Annotation Scheme and Classifier for Personal Facts in Dialogue 

**Authors**: Konstantin Zaitsev  

**Link**: [PDF](https://arxiv.org/pdf/2605.10339)  

**Abstract**: The advancement of Large Language Models (LLMs) has enabled their application in personalized dialogue systems. We present an extended annotation scheme for personal fact classification that addresses limitations in existing approaches, particularly PeaCoK. Our scheme introduces new categories (Demographics, Possessions) and attributes (Duration, Validity, Followup) that enable structured storage, quality filtering, and identification of facts suitable for dialogue continuation. We manually annotated 2,779 facts from Multi-Session Chat and trained a multi-head classifier based on transformer encoders. Combined with the Gemma-300M encoder, the classifier achieves $81.6 \pm 2.6$\% macro F1, outperforming all few-shot LLM baselines (best: GPT-5.4-mini, 72.92\%) by nearly 9 percentage points while requiring substantially fewer computational resources. Error analysis reveals persistent challenges in semantic boundary disambiguation, temporal aspect interpretation, and pragmatic reasoning for followup assessment. The dataset\footnotemark[1] and classifier\footnotemark[2] are publicly available. 

---
# ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models 

**Authors**: Wentao Qiu, Guanran Luo, Zhongquan Jian, Jingqi Gao, Meihong Wang, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10328)  

**Abstract**: A central challenge in large-scale decision-making under incomplete information is estimating reliable probabilities. Recent approaches leverage Large Language Models (LLMs) to generate explanatory factors and elicit coarse-grained probability estimates. Typically, an LLM performs forward abduction to propose factors, each paired with two mutually exclusive attributes, and a Naïve Bayes model is trained over factor combinations to refine the final probabilities. However, sparse factor spaces often yield ``unknown'' outcomes, while expanding factors increases noise and spurious correlations, weakening conditional independence and degrading reliability. To address these limitations, we propose \textsc{Anchor}, an inference framework that orchestrates aggregated Bayesian inference over a hierarchically structured factor space. \textsc{Anchor} first constructs a dense and organized factor space via iterative generation and hierarchical clustering. It then performs context-aware mapping through hierarchical retrieval and refinement, substantially reducing ``unknown'' predictions. Finally, \textsc{Anchor} augments Naïve Bayes with a Causal Bayesian Network to capture latent dependencies among factors, relaxing the strict independence assumption. Experiments show that \textsc{Anchor} markedly reduces ``unknown'' predictions and produces more reliable probability estimates than direct LLM baselines, achieving state-of-the-art performance while significantly reducing time and token overhead. 

---
# Extending Confidence-Based Text2Cypher with Grammar and Schema Aware Filtering 

**Authors**: Makbule Gulcin Ozsoy  

**Link**: [PDF](https://arxiv.org/pdf/2605.10318)  

**Abstract**: Large language models (LLMs) allow users to query databases using natural language by translating questions into executable queries. Despite strong progress on tasks such as Text2SQL, Text2SPARQL, and Text2Cypher, most existing methods focus on better prompting, fine-tuning, or iterative refinement. However, they often do not explicitly enforce structural constraints, such as syntactic validity and schema consistency. This can reduce reliability, since generated queries must satisfy both syntax rules and database schema constraints to be executable. In this work, we study how structured constraints can be used in test-time inference for Text2Cypher. We focus on post-generation validation to improve query correctness. We extend a confidence-based inference framework with a sequential filtering process that combines confidence scoring, grammar validation, and schema constraints before final aggregation. This lets us analyze how different constraint types affect generated queries. Our experiments with two instruction-tuned models show that grammar-based filtering improves syntactic validity. Schema-aware filtering further improves execution quality by enforcing consistency with the database structure. However, stronger filtering also increases the number of empty predictions and reduces execution coverage. Overall, we show that adding simple structural checks at test time improves the reliability of Text2Cypher generation, and we provide a clearer view of how syntax and schema constraints contribute differently. 

---
# Qwen Goes Brrr: Off-the-Shelf RAG for Ukrainian Multi-Domain Document Understanding 

**Authors**: Anton Bazdyrev, Ivan Bashtovyi, Ivan Havlytskyi, Oleksandr Kharytonov, Artur Khodakovskyi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10296)  

**Abstract**: We participated in the Fifth UNLP shared task on multi-domain document understanding, where systems must answer Ukrainian multiple-choice questions from PDF collections and localize the supporting document and page. We propose a retrieval-augmented pipeline built around three ideas: contextual chunking of PDFs, question-aware dense retrieval and reranking conditioned on both the question and answer options, and constrained answer generation from a small set of reranked passages. Our final system uses Qwen3-Embedding-8B for retrieval, a fine-tuned Qwen3-Reranker-8B for passage ranking, and Qwen3-32B for answer selection. On a held-out split, reranking improves Recall@1 from 0.6957 to 0.7935, while using the top-2 reranked passages raises answer accuracy from 0.9348 to 0.9674. Our best leaderboard run reached 0.9452 on the public leaderboard and 0.9598 on the private leaderboard. Our results suggest that, under strict code-competition constraints, preserving document structure and making relevance estimation aware of the answer space are more effective than adding complex downstream heuristics. 

---
# DECO-MWE: building a linguistic resource of Korean multiword expressions for feature-based sentiment analysis 

**Authors**: Jaeho Han, Changhoe Hwang, Seongyong Choi, Gwanghoon Yoo, Eric Laporte, Jeesun Nam  

**Link**: [PDF](https://arxiv.org/pdf/2605.10295)  

**Abstract**: This paper aims to construct a linguistic resource of Korean Multiword Expressions for Feature-Based Sentiment Analysis (FBSA): DECO-MWE. Dealing with multiword expressions (MWEs) has been a critical issue in FBSA since many constructs reveal lexical idiosyncrasy. To construct linguistic resources of sentiment MWEs efficiently, we utilize the Local Grammar Graph (LGG) methodology: DECO-MWE is formalized as a Finite-State Transducer that represents lexical-syntactic restrictions on MWEs. In this study, we built a corpus of cosmetics review texts, which show particularly frequent occurrences of MWEs. Based on an empirical examination of the corpus, four types of MWEs have been distinguished. The DECO-MWE thus covers the following four categories: Standard Polarity MWEs (SMWEs), Domain-Dependent Polarity MWEs (DMWEs), Compound Named Entity MWEs (EMWEs) and Compound Feature MWEs (FMWEs). The retrieval performance of the DECO-MWE shows 0.806 f-measure in the test corpus. This study brings a twofold outcome: first, a sizeable general-purpose polarity MWE lexicon, which may be broadly used in FBSA; second, a finite-state methodology adopted in this study to treat domain-dependent MWEs such as idiosyncratic polarity expressions, named entity expressions or feature expressions, and which may be reused in describing linguistic properties of other corpus domains. 

---
# MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading 

**Authors**: Baibei Ji, Xiaoyang Weng, Juntao Li, Zecheng Tang, Yihang Lou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10268)  

**Abstract**: To tackle long-context reasoning tasks without the quadratic complexity of standard attention mechanisms, approaches based on agent memory have emerged, which typically maintain a dynamically updated memory when linearly processing document chunks. To mitigate the potential loss of latent evidence in this memorize-while-reading paradigm, recent works have integrated retrieval modules that allow agents to recall information previously discarded during memory overwriting. However, retrieval-based recall suffers from both evidence loss during memory formation and interference induced by invalid queries. To overcome these limitations, we propose MemReread. Built upon streaming reading, MemReread circumvents intermediate retrieval. It triggers question decomposition and rereading when the final memory is insufficient, enabling the recovery of indirect facts that were prematurely discarded. This design supports non-linear reasoning while preserving the inherent logical flow of document comprehension. To further enhance practicality, we introduce a reinforcement learning framework that enhances length extrapolation capability while dynamically determining the number of rereading passes based on task complexity, thereby flexibly controlling computational overhead. Extensive experiments demonstrate that MemReread consistently outperforms baseline frameworks on long-context reasoning tasks, while maintaining linear time complexity with respect to context length. 

---
# Building Korean linguistic resource for NLU data generation of banking app CS dialog system 

**Authors**: Jeongwoo Yoon, On-yu Park, Changhoe Hwang, Gwanghoon Yoo, Eric Laporte, Jeesun Nam  

**Link**: [PDF](https://arxiv.org/pdf/2605.10241)  

**Abstract**: Natural language understanding (NLU) is integral to task-oriented dialog systems, but demands a considerable amount of annotated training data to increase the coverage of diverse utterances. In this study, we report the construction of a linguistic resource named FIAD (Financial Annotated Dataset) and its use to generate a Korean annotated training data for NLU in the banking customer service (CS) domain. By an empirical examination of a corpus of banking app reviews, we identified three linguistic patterns occurring in Korean request utterances: TOPIC (ENTITY, FEATURE), EVENT, and DISCOURSE MARKER. We represented them in LGGs (Local Grammar Graphs) to generate annotated data covering diverse intents and entities. To assess the practicality of the resource, we evaluate the performances of DIET-only (Intent: 0.91 /Topic [entity+feature]: 0.83), DIET+ HANBERT (I:0.94/T:0.85), DIET+ KoBERT (I:0.94/T:0.86), and DIET+ KorBERT (I:0.95/T:0.84) models trained on FIAD-generated data to extract various types of semantic items. 

---
# Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection 

**Authors**: Yiwen Chen, Kuan Li, Fuzhen Zhuang, Deqing Wang, Zhao Zhang, Liwen Zhang, Yong Jiang, Shuai Wang, Minhao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.10235)  

**Abstract**: Recent advances in large language models (LLMs) have expanded the context window to beyond 128K tokens, enabling long-document understanding and multi-source reasoning. A key challenge, however, lies in choosing between retrieval-augmented generation (RAG) and long-context (LC) strategies: RAG is efficient but constrained by retrieval quality, while LC supports global reasoning at higher cost and with position sensitivity. Existing methods such as Self-Route adopt failure-driven fallback from RAG to LC, but remain passive, inefficient, and hard to interpret. We propose Pre-Route, a proactive routing framework that performs structured reasoning before answering. Using lightweight metadata (e.g., document type, length, initial snippet), Pre-Route enables task analysis, coverage estimation, and information-need prediction, producing explainable and cost-efficient routing decisions. Our study shows three key findings: (i) LLMs possess latent routing ability that can be reliably elicited with guidelines, allowing single-sample performance to approach that of multi-sample (Best-of-N) results; (ii) linear probes reveal that structured prompts sharpen the separability of the "optimal routing dimension" in representation space; and (iii) distillation transfers this reasoning structure to smaller models for lightweight deployment. Experiments on LaRA (in-domain) and LongBench-v2 (OOD) confirm that Pre-Route outperforms Always-RAG, Always-LC, and Self-Route baselines, achieving superior overall cost-effectiveness. 

---
# Relative Score Policy Optimization for Diffusion Language Models 

**Authors**: Zichao Yu, Shengze Xu, Bingqing Jiang, Wenyi Zhang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.10218)  

**Abstract**: Diffusion large language models (dLLMs) offer a promising route to parallel and efficient text generation, but improving their reasoning ability requires effective post-training. Reinforcement learning with verifiable rewards (RLVR) is a natural choice for this purpose, yet its application to dLLMs is hindered by the absence of tractable sequence-level log-ratios, which are central to standard policy optimization. The lack of tractable sequence-level log-ratios forces existing methods to rely on high-variance ELBO-based approximations, where high verifier rewards can amplify inaccurate score estimates and destabilize RL training. To overcome this issue, we propose \textbf{R}elative \textbf{S}core \textbf{P}olicy \textbf{O}ptimization (RSPO), a simple RLVR method that uses verifiable rewards to calibrate noisy likelihood estimates in dLLMs. The core of our algorithm relies on a key observation: a reward advantage can be interpreted not only as an update direction, but also as a target for the relative log-ratio between the current and reference policies. Accordingly, RSPO calibrates this noisy relative log-ratio estimate by comparing its reward advantage with the reward-implied target relative log-ratio, updating the policy according to the gap between the current estimate and the target rather than the raw advantage alone. Experiments on mathematical reasoning and planning benchmarks show that RSPO yields especially strong gains on planning tasks and competitive mathematical-reasoning performance. 

---
# The Impact of Editorial Intervention on Detecting Native Language Traces 

**Authors**: Ahmet Yavuz Uluslu, Mark Gales, Kate Knill, Gerold Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2605.10216)  

**Abstract**: Native Language Identification (NLI) is the task of determining an author's native language (L1) from their non-native writings. With the advent of human-AI co-authorship, non-native texts are routinely corrected and rewritten by large language models, fundamentally altering the linguistic features NLI models depend on. In this paper, we investigate the robustness of L1 traces across increasing degrees of editorial intervention. By processing 450 essays from the Write & Improve 2024 corpus through varying levels of grammatical error correction (GEC) and paraphrasing, we demonstrate that L1 attribution does not entirely depend on surface-level errors. Instead, the detection models leverage deeper L1 features: unidiomatic lexico-semantic choices, pragmatic transfer, and the author's underlying cultural perspective. We find that minimal edits preserve these structural traces and maintain high profiling accuracy. In contrast, fluency edits and paraphrasing normalize these L1 features, leading to a severe degradation in performance. 

---
# To Redact, or not to Redact? A Local LLM Approach to Deliberative Process Privilege Classification 

**Authors**: Maik Larooij, David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2605.10211)  

**Abstract**: Government transparency laws, like the Freedom of Information (FOIA) acts in the United States and United Kingdom, and the Woo (Open Government Act) in the Netherlands, grant citizens the right to directly request documents from the government. As these documents might contain sensitive information, such as personal information or threats to national security, the laws allow governments to redact sensitive parts of the documents prior to release. We build on prior research to perform automatic sensitivity classification for the FOIA Exemption 5 deliberative process privilege using Large Language Models (LLMs). However, processing documents not yet cleared for review via third-party cloud APIs is often legally or politically untenable. Therefore, in this work, we perform sensitivity classification with a small, local model, deployable on consumer-grade hardware (Qwen3.5 9B). We compare eight variants of applying LLMs for sentence classification, using well-known prompting techniques, and find that a combination of Chain-of-Thought prompting and few-shot prompting with error-based examples outperforms classification models of earlier work in terms of recall and F2 score. This method also closely approaches the performance of a widely-used, cost-efficient commercial model (Gemini 2.5 Flash). In an additional analysis, we find that sentences that are predicted as deliberative contain more verbs that indicate the expression of opinions, and are more often phrased in in first-person. Above all, deliberativeness seems characterized by the presence of a combination of multiple indicators, in particular the combination of first-person words with a verb for expressing opinion. 

---
# How Should LLMs Listen While Speaking? A Study of User-Stream Routing in Full-Duplex Spoken Dialogue 

**Authors**: Hui Lu, Xueyuan Chen, Huimeng Wang, Shuhai Peng, Shiyin Kang, Xixin Wu, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10199)  

**Abstract**: Full-duplex spoken dialogue requires a model to keep listening while generating its own spoken response. This is challenging for large language models (LLMs), which are designed to extend a single coherent sequence and do not naturally support user input arriving during generation. We argue that how the user stream is routed into the LLM is therefore a key architectural question for full-duplex modeling. To study this question, we extend a text-only LLM into a unified full-duplex spoken dialogue system and compare two routing strategies under a shared training pipeline: (i) channel fusion, which injects the user stream directly into the LLM input, and (ii) cross-attention routing, which keeps the user stream as external memory accessed through cross-attention adapters. Experiments on spoken question answering and full-duplex interaction benchmarks reveal a clear tradeoff. Channel fusion yields stronger semantic grounding and consistently better question-answering performance. However, under semantically overlapping conditions such as user interruptions, it is more vulnerable to context corruption: if the model fails to stop in time, the overlapping user stream can interfere with ongoing generation and lead to semantically incoherent continuations. Cross-attention routing underperforms on question answering, but better preserves the LLM generation context and is more robust to this failure mode. These results establish user-stream routing as a central design axis in full-duplex spoken dialogue and offer practical guidance on the tradeoff between semantic integration and context robustness. We provide a demo page for qualitative inspection. 

---
# LegalCiteBench: Evaluating Citation Reliability in Legal Language Models 

**Authors**: Sijia Chen, Hang Yin, Shunfan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.10186)  

**Abstract**: Large language models (LLMs) are increasingly integrated into legal drafting and research workflows, where incorrect citations or fabricated precedents can cause serious professional harm. Existing legal benchmarks largely emphasize statutory reasoning, contract understanding, or general legal question answering, but they do not directly study a central common-law failure mode: when asked to provide case authorities without external grounding, models may return plausible-looking but incorrect citations or cases. We introduce LegalCiteBench, a benchmark for studying closed-book citation recovery, citation verification, and case matching in legal language models. LegalCiteBench contains approximately 24K evaluation instances constructed from 1,000 real U.S. judicial opinions from the Case Law Access Project. The benchmark covers five citation-centric tasks: citation retrieval, citation completion, citation error detection, case matching, and case verification and correction. Across 21 LLMs, exact citation recovery remains highly challenging in this closed-book setting: even the strongest models score below 7/100 on citation retrieval and completion. Within the evaluated models, scale and legal-domain pretraining provide limited gains and do not resolve this difficulty. Models also frequently provide concrete but incorrect or low-overlap authorities under our evaluation protocol, with Misleading Answer Rates (MAR) exceeding 94% for 20 of 21 evaluated models on retrieval-heavy tasks. A prompt-only abstention experiment shows that explicit uncertainty instructions reduce some confident fabrication but do not improve citation correctness. LegalCiteBench is intended as a diagnostic framework for studying authority generation failures, verification behavior, and abstention when external grounding is absent, incomplete, or bypassed. 

---
# When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews 

**Authors**: Sandeep Kumar, Yash Kamdar, Abid Hossain, Bharti Kumari, Tanik Saikh, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2605.10171)  

**Abstract**: Scientific peer reviews frequently contain conflicting expert judgments, and the increasing scale of conference submissions makes it challenging for Area Chairs and editors to reliably identify and interpret such disagreements. Existing approaches typically frame reviewer disagreement as binary contradiction detection over isolated sentence pairs, abstracting away the review-level context and obscuring differences in the severity of evaluative conflict. In this work, we introduce a fine-grained formulation of reviewer contradiction analysis that operates over full peer reviews by explicitly identifying contradiction evidence spans and assigning graded disagreement intensity scores. To support this task, we present RevCI, an expert-annotated benchmark of peer-review pairs with evidence-level contradiction annotations with graded intensity labels. We further propose IMPACT, a structured multi-agent framework that integrates aspect-conditioned evidence extraction, deliberative reasoning, and adjudication to model reviewer contradictions and their intensity. To support efficient deployment, we distill IMPACT into TIDE, a small language model that predicts contradiction evidence and intensity in a single forward pass. Experimental results show that IMPACT substantially outperforms strong single-agent and generic multi-agent baselines in both evidence identification and intensity agreement, while TIDE achieves competitive performance at significantly lower inference cost. 

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
# Synthetic Pre-Pre-Training Improves Language Model Robustness to Noisy Pre-Training Data 

**Authors**: Xu Guo, Runyu Peng, Jian Tong, Yunhua Zhou, Haijun Lv, Zhihui Lu, Qipeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.10129)  

**Abstract**: Large language models (LLMs) rely on web-scale corpora for pre-training. The noise inherent in these datasets tends to obscure meaningful patterns and ultimately degrade model performance. Data curation mitigates but cannot eliminate such noise, so pre-training corpora remain noisy in practice. We therefore study whether a lightweight pre-pre-training (PPT) stage based on synthetic data with learnable temporal structure helps resist noisy data during the pre-training (PT) stage. Across various corruption settings, our method consistently improves robustness to noise during PT, with larger relative gains at higher noise levels. For a 1B-parameter model, a synthetic PPT stage with only 65M tokens achieves the same final loss as the baseline while using up to 49\% fewer natural-text PT tokens across different noise levels. Mechanistic analyses suggest PPT does not immediately suppress attention to noisy tokens. Rather, PPT-initialized models gradually downweight attention between corrupted tokens during noisy PT. This indicates that synthetic PPT inhibits noise self-modeling and shapes the subsequent optimization trajectory. Code is available at this https URL. 

---
# SkillRAE: Agent Skill-Based Context Compilation for Retrieval-Augmented Execution 

**Authors**: Xiangcheng Meng, Shu Wang, Yixiang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10114)  

**Abstract**: Large Language Model (LLM)-based agents (e.g., OpenClaw) increasingly rely on reusable skill libraries to solve artifact-rich tasks such as document-centric workflows and data-intensive analysis. As these libraries grow, a few works have attempted to study the Retrieval-Augmented Execution (RAE), which often first retrieves some external skills and other knowledge, then compiles the context using retrieved skills, and finally executes the task. Existing works mainly focus on optimizing skill retrieval and task execution, and they pay little attention to how to effectively organize the selected skill evidence in a form that is compact, grounded, and immediately usable for the downstream executors to complete tasks. To fill this gap, we propose SkillRAE, a two-stage RAE approach focusing on skill-based context compilation, which consists of the offline and online stages. Specifically, in the offline indexing stage, it builds a multi-level skill graph over skill communities, skills, and reusable subunits, for capturing their relationships. In the online retrieval stage, it first performs skill-ranked retrieval with selected-subunit evidence export in the graph, and then applies rescue-aware compact compilation to recover the key evidence. Together, these components compile a coarse-ranked skill set into a task-specific context that is compact, grounded, and immediately usable. Experiments on two public benchmarks show that SkillRAE achieves a significant improvement over baselines for RAE. For example, on SkillsBench, it achieves an improvement of 11.7% over the SOTA method. Ablation studies further show that our context compilation is crucial, instead of a mere prompt addition. 

---
# GLiNER-Relex: A Unified Framework for Joint Named Entity Recognition and Relation Extraction 

**Authors**: Ihor Stepanov, Oleksandr Lukashov, Mykhailo Shtopko, Vivek Kalyanarangan  

**Link**: [PDF](https://arxiv.org/pdf/2605.10108)  

**Abstract**: Joint named entity recognition (NER) and relation extraction (RE) is a fundamental task in natural language processing for constructing knowledge graphs from unstructured text. While recent approaches treat NER and RE as separate tasks requiring distinct models, we introduce GLiNER-Relex, a unified architecture that extends the GLiNER framework to perform both entity recognition and relation extraction in a single model. Our approach leverages a shared bidirectional transformer encoder to jointly represent text, entity type labels, and relation type labels, enabling zero-shot extraction of arbitrary entity and relation types specified at inference time. GLiNER-Relex constructs entity pair representations from recognized spans and scores them against relation type embeddings using a dedicated relation scoring module. We evaluate our model on four standard relation extraction benchmarks: CoNLL04, DocRED, FewRel, and CrossRE, and demonstrate competitive performance against both specialized relation extraction models and large language models, while maintaining the computational efficiency characteristic of the GLiNER family. The model is released as an open-source Python package with a simple inference API that allows users to specify arbitrary entity and relation type labels at inference time and obtain both entities and relation triplets in a single call. All models and code are publicly available. 

---
# FERA: Uncertainty-Aware Federated Reasoning for Large Language Models 

**Authors**: Ruhan Wang, Chengkai Huang, Zhiyong Wang, Junda Wu, Rui Wang, Tong Yu, Julian McAuley, Lina Yao, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.10082)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning capabilities when guided by high-quality demonstrations, yet such data is often distributed across organizations that cannot centralize it due to regulatory, proprietary, or institutional constraints. We study federated reasoning, where a server improves multi-step reasoning by coordinating with heterogeneous clients holding private demonstrations, without centralized training or raw data sharing. The key challenge is that client reliability is query-dependent, while the server cannot inspect client data to determine which contributions are trustworthy. To address this, we propose Uncertainty-Aware Federated Reasoning (FERA), a training-free framework based on iterative server-client co-refinement. Across communication rounds, clients generate reasoning traces with lightweight uncertainty estimates, and the server synthesizes them into improved reasoning that is redistributed as context for the next round, progressively improving both server outputs and client-side reasoning. Within each round, Uncertainty-Aware Self-Critique Aggregation (UA-SCA) resolves conflicts among heterogeneous client traces through query-dependent trust weighting and structured cross-client verification. Rather than simply discarding low-quality traces, UA-SCA revises flawed reasoning steps to recover useful information. We provide theoretical guarantees showing that the proposed iterative protocol converges and that uncertainty-aware weighting accelerates convergence. Experiments on multiple reasoning benchmarks show that FERA consistently outperforms both federated training and training-free baselines, achieving progressively higher accuracy across rounds while maintaining communication and computational efficiency. 

---
# PHAGE: Patent Heterogeneous Attention-Guided Graph Encoder for Representation Learning 

**Authors**: Yongmin Yoo, Qiongkai Xu, Zhangkai Wu, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.10073)  

**Abstract**: Patent claims form a directed dependency structure in which dependent claims inherit and refine the scope of earlier claims; however, existing patent encoders linearize claims as text and discard this hierarchy. Directly encoding this structure into self-attention poses two challenges: claim dependencies mix relation types that differ in semantics and extraction reliability, and the dependency graph is defined over claims while Transformers attend over tokens. PHAGE addresses the first challenge through a deterministic graph construction pipeline that separates near-deterministic legal citations from noisier rule-based technical relations, preserving type distinctions as heterogeneous edges. It addresses the second through a connectivity mask and learnable relation-aware biases that lift claim-level topology into token-level attention, allowing the encoder to differentially weight each relation type. A dual-granularity contrastive objective then aligns representations with both inter-patent taxonomy and intra-patent topology. PHAGE outperforms all baselines on classification, retrieval, and clustering, showing that intra-document claim topology is a stronger inductive bias than inter-document structure and that this bias persists in the encoder weights after training. 

---
# NCO: A Versatile Plug-in for Handling Negative Constraints in Decoding 

**Authors**: Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.10065)  

**Abstract**: Controlling Large Language Models (LLMs) to prevent the generation of undesirable content, such as profanity and personally identifiable information (PII), has become increasingly critical. While earlier approaches relied on post-processing or resampling, recent research has shifted towards constrained decoding methods that control outputs during generation to mitigate high computational costs and quality degradation. However, preventing multiple forbidden hard constraints or regex constraints from appearing anywhere in the output is computationally challenging. A straightforward solution is to convert these constraints into a single automaton that tracks all forbidden patterns during decoding, but this often becomes impractically large. Standard regex engines also do not readily support the operations needed to build such a constraint, such as complement and intersection. In order to address these limitations, we propose NCO, a decoding strategy that performs online pattern matching over finite hard constraints and regex constraints, reducing computational overhead without inducing state explosion. NCO is fully compatible with standard inference strategies, including various sampling methods and beam search, while also supporting soft masking for probabilistic suppression. We empirically demonstrate its effectiveness across practical tasks, including PII and profanity suppression. Our implementation is available at this https URL . 

---
# Not-So-Strange Love: Language Models and Generative Linguistic Theories are More Compatible than They Appear 

**Authors**: R. Thomas McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2605.10061)  

**Abstract**: Futrell and Mahowald (2025) frame the success of neural language models (LMs) as supporting gradient, usage-based linguistic theories. I argue that LMs can also instantiate theories based on formal structures - the types of theories seen in the generative tradition. This argument expands the space of theories that can be tested with LMs, potentially enabling reconciliations between usage-based and generative accounts. 

---
# Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering 

**Authors**: Xinyu Zhang, Zhicheng Dou, Deyang Li, Jianjun Tao, Shuo Cheng, Ruifeng Shi, Fangchao Liu, Enrui Hu, Yangkai Ding, Hongbo Wang, Qi Ye, Xuefeng Jin, Zhangchun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.10052)  

**Abstract**: As artificial intelligence engineering paradigms shift from single-agent Prompt and Context Engineering toward multi-agent \textbf{Coordination Engineering}, the ability to codify and systematically improve how multiple agents collaborate has emerged as a critical bottleneck. While single-agent skills can now be distributed as portable assets, multi-agent coordination protocols remain locked within framework-internal code or static configurations, preventing them from being shared across systems or autonomously improved over time. We propose \textbf{Swarm Skills}, a portable specification that extends the Anthropic Skills standard with multi-agent semantics. Swarm Skills turns multi-agent workflows into first-class, distributable assets that consist of roles, workflows, execution bounds, and a built-in semantic structure for self-evolution. To operationalize the specification's evolving nature, we present a companion self-evolution algorithm that automatically distills successful execution trajectories into new Swarm Skills and continuously patches existing ones based on multi-dimensional scoring (Effectiveness, Utilization, and Freshness), eliminating the need for human-in-the-loop oversight during the refinement process. Through an architectural compatibility analysis and a comprehensive qualitative case study using the open-source JiuwenSwarm reference implementation, we demonstrate how Swarm Skills achieves zero-adapter cross-agent portability via progressive disclosure, enabling agent teams to self-evolve their coordination strategies without framework lock-in. 

---
# Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework 

**Authors**: Xilai Ma, Liye Zhao, Weijun Yao, Haibing Di, Wenya Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.10043)  

**Abstract**: Large Language Model (LLM) personalization aims to align model behaviors with individual user preferences. Existing methods often focus on isolated user histories, neglecting the essential role of inter-user differences. We propose C-BPO, a framework that personalizes LLMs via preference-calibrated binary signals. By treating target user data as positive feedback and other users' data as an auxiliary set of implicit negative signals, C-BPO captures distinct inter-user differences. To mitigate the preference overlap issue, where shared task knowledge is erroneously penalized, we derive an objective grounded in Positive-Unlabeled (PU) learning theory. This approach purifies negative signals by subtracting ``positive bias'', ensuring alignment with unique idiosyncrasies without compromising general helpfulness. Empirical experiments across various personalization tasks and backbone LLMs show C-BPO consistently outperforms baselines, demonstrating the efficacy of preference-calibrated binary signals in modeling inter-user differences. 

---
# PlantMarkerBench: A Multi-Species Benchmark for Evidence-Grounded Plant Marker Reasoning 

**Authors**: Sajib Acharjee Dip, Song Li, Liqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10032)  

**Abstract**: Cell-type-specific marker genes are fundamental to plant biology, yet existing resources primarily rely on curated databases or high-throughput studies without explicitly modeling the supporting evidence found in scientific literature. We introduce PlantMarkerBench, a multi-species benchmark for evaluating literature-grounded plant marker evidence interpretation from full-text biological papers. PlantMarkerBench is constructed using a modular curation pipeline integrating large-scale literature retrieval, hybrid search, species-aware biological grounding, structured evidence extraction, and targeted human review. The benchmark spans four plant species -- Arabidopsis, maize, rice, and tomato -- and contains 5,550 sentence-level evidence instances annotated for marker-evidence validity, evidence type, and support strength. We define two benchmark tasks: determining whether a candidate sentence provides valid marker evidence for a gene-cell-type pair, and classifying the evidence into expression, localization, function, indirect, or negative categories. We benchmark diverse open-weight and closed-source language models across species and prompting strategies. Although frontier models achieve relatively strong performance on direct expression evidence, performance drops substantially on functional, indirect, and weak-support evidence, with evidence-type confusion emerging as a dominant failure mode. Open-weight models additionally exhibit elevated false-positive rates under ambiguous biological contexts. PlantMarkerBench provides a challenging and reproducible evaluation framework for literature-grounded biological evidence attribution and supports future research on trustworthy scientific information extraction and AI-assisted plant biology. 

---
# Speech-based Psychological Crisis Assessment using LLMs 

**Authors**: Terumi Chiba, Yang Luo, Ziyun Cui, Yongsheng Tong, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10027)  

**Abstract**: Psychological support hotlines provide critical support for individuals experiencing mental health emergencies, yet current assessments largely rely on human operators whose judgments may vary with professional experience and are constrained by limited staffing resources. This paper proposes a large language model (LLM)-based framework for automated crisis level classification, a key indicator that supports many downstream tasks and improves the overall quality of hotline services. To better capture emotional signals in spoken conversations, we introduce a paralinguistic injection method that inserts identified non-verbal emotional cues into speech transcripts, enabling LLM-based reasoning to incorporate critical acoustic nuances. In addition, we propose a reasoning-enhanced training strategy that trains the model to generate diagnostic reasoning chains as an auxiliary task, which serves as a regulariser to improve classification performance. Combined with data augmentation, our final system achieves a macro F1-score of 0.802 and an accuracy of 0.805 on the three-class classification task under 5-fold cross-validation. 

---
# Medical Incident Causal Factors and Preventive Measures Generation Using Tag-based Example Selection in Few-shot Learning 

**Authors**: Yuna Haseyama, Tomoki Ito, Hiroki Sakaji, Itsuki Noda  

**Link**: [PDF](https://arxiv.org/pdf/2605.10025)  

**Abstract**: In high-stakes domains such as healthcare, the reliability of Large Language Models (LLMs) is critical, particularly when generating clinical insights from incident reports. This study proposes a tag-based few-shot example selection method for prompting LLMs to generate background/causal factors and preventive measures from details of the medical incidents. For our experiments, we use the Japanese Medical Incident Dataset (JMID), a structured dataset of 3,884 real-world medical accident and near-miss reports. These reports are variably annotated with a wide range of tags--some include descriptive information (e.g., "medications," "blood transfusion therapy"). We compare three few-shot example selection strategies--random sampling, cosine similarity-based selection, and our proposed tag-based method--using GPT-4o and LLaMA 3.3. Results show that the tag-based approach achieves the highest precision and most stable generation behavior, while similarity-based selection often leads to unintended outputs and safety filter activation. These findings suggest that selecting examples based on human-interpretable dataset tags can improve generation precision and stability in clinical LLM applications. 

---
# Annotations Mitigate Post-Training Mode Collapse 

**Authors**: Jacob Mitchell Springer, Madhu Advani, Lukas Aichberger, Arwen Bradley, Eran Malach, Omid Saremi, Sinead Williamson, Preetum Nakkiran, Etai Littwin, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.09995)  

**Abstract**: Post-training (via supervised fine-tuning) improves instruction-following, but often induces semantic mode collapse by biasing models toward low-entropy fine-tuning data at the expense of the high-entropy pretraining distribution. Crucially, we find this trade-off worsens with scale. To close this semantic diversity gap, we propose annotation-anchored training, a principled method that enables models to adopt the preference-following behaviors of post-training without sacrificing the inherent diversity of pretraining. Our approach is simple: we pretrain on documents paired with semantic annotations, inducing a rich annotation distribution that reflects the full breadth of pretraining data, and we preserve this distribution during post-training. This lets us sample diverse annotations at inference time and use them as anchors to guide generation, effectively transferring pretraining's semantic richness into post-trained models. We find that models trained with annotation-anchored training can attain $6 \times$ less diversity collapse than models trained with SFT, and improve with scale. 

---
# Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference 

**Authors**: Sietse Schelpe  

**Link**: [PDF](https://arxiv.org/pdf/2605.09990)  

**Abstract**: Data-intensive applications, ranging from large-scale retrieval systems to advanced data pipelines, are increasingly bottlenecked by the processing of highly redundant text corpora. We present Merlin, a local-first, agnostic, high-throughput deduplication and context optimization engine designed to mitigate these inefficiencies. Utilizing a highly optimized, SIMD-friendly open-addressing flat hash set combined with xxHash3-64, Merlin performs rapid, byte-exact deduplication of text passages and data chunks. While broadly applicable to any text-processing workflow, its impact is particularly pronounced in Large Language Model (LLM) ecosystems, such as Retrieval-Augmented Generation (RAG). Our empirical evaluations demonstrate an input reduction ranging from 13.9% in low-redundancy datasets to over 71% in high-redundancy pipelines, maintaining absolute data fidelity. Furthermore, we detail the system's integration architecture via the Model Context Protocol (MCP), enabling secure, zero-network-interception deployment across major IDEs and autonomous agents. This paper outlines the core algorithmic design, performance benchmarks, and the architectural principles required to process data at sustained speeds of up to 8.7 GB/s. 

---
# GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction 

**Authors**: Urchade Zaratiana, Ash Lewis, George Hurn-Maloney  

**Link**: [PDF](https://arxiv.org/pdf/2605.09973)  

**Abstract**: Reliable detection of personally identifiable information (PII) is increasingly important across modern data-processing systems, yet the task remains difficult: PII spans are heterogeneous, locale-dependent, context-sensitive, and often embedded in noisy or semi-structured documents. We present GLiNER2-PII, a small 0.3B-parameter model adapted from GLiNER2 and designed to recognize a broad taxonomy of 42 PII entity types at character-span resolution. Training such systems, however, is constrained by the scarcity of shareable annotated data and the privacy risks associated with collecting real PII at scale. To address this challenge, we construct a multilingual synthetic corpus of 4,910 annotated texts using a constraint-driven generation pipeline that produces diverse, realistic examples across languages, domains, formats, and entity distributions. On the challenging SPY benchmark, GLiNER2-PII achieves the highest span-level F1 among five compared systems, including OpenAI Privacy Filter and three GLiNER-based detectors. We publicly release the model on Hugging Face to support further research and practical deployment of open PII detection systems. 

---
# Beyond Majority Voting: Agreement-Based Clustering to Model Annotator Perspectives in Subjective NLP Tasks 

**Authors**: Tadesse Destaw Belay, Ibrahim Said Ahmad, Idris Abdulmumin, Abinew Ali Ayele, Alexander Gelbukh, Eusebio Ricárdez-Vázquez, Olga Kolesnikova, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2605.09955)  

**Abstract**: Disagreement in annotation is a common phenomenon in the development of NLP datasets and serves as a valuable source of insight. While majority voting remains the dominant strategy for aggregating labels, recent work has explored modeling individual annotators to preserve their perspectives. However, modeling each annotator is resource-intensive and remains underexplored across various NLP tasks. We propose an agreement-based clustering technique to model the disagreement between the annotators. We conduct comprehensive experiments in 40 datasets in 18 typologically diverse languages, covering three subjective NLP tasks: sentiment analysis, emotion classification, and hate speech detection. We evaluate four aggregation approaches: majority vote, ensemble, multi-label, and multitask. The results demonstrate that agreement-based clustering can leverage the full spectrum of annotator perspectives and significantly enhance classification performance in subjective NLP tasks compared to majority voting and individual annotator modeling. Regarding the aggregation approach, the multi-label and multitask approaches are better for modeling clustered annotators than an ensemble and model majority vote. 

---
# TRACER: Verifiable Generative Provenance for Multimodal Tool-Using Agents 

**Authors**: Bihui Yu, Caijun Jia, Jing Chi, Xiaohan Liu, Yining Wang, He Bai, Yuchen Liu, Jingxuan Wei, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09934)  

**Abstract**: Multimodal large language models increasingly solve vision-centric tasks by calling external tools for visual inspection, OCR, retrieval, calculation, and multi-step reasoning. Current tool-using agents usually expose the executed tool trajectory and the final answer, but they rarely specify which tool observation supports each generated claim. We call this missing claim-level dependency structure the provenance gap. The gap makes tool use hard to verify and hard to optimize, because useful evidence, redundant exploration, and unsupported reasoning are mixed in the same trajectory. We introduce TRACER, a framework for verifiable generative provenance in multimodal tool-using agents. Instead of adding citations after generation, TRACER generates each answer sentence together with a structured provenance record that identifies the supporting tool turn, evidence unit, and semantic support relation. Its relation space contains Quotation, Compression, and Inference, covering direct reuse, faithful condensation, and grounded derivation. TRACER verifies each record through schema checking, tool-turn alignment, source authenticity, and relation rationality, and then converts verified provenance into traceability constraints and provenance-derived local credit for reinforcement learning. We further construct TRACE-Bench, a benchmark for sentence-level provenance reconstruction from coarse multimodal tool trajectories. On TRACE-Bench, simply adding tools often introduces noise. With Qwen3-VL-8B, TRACER reaches 78.23% answer accuracy and 95.72% summary accuracy, outperforming the strongest closed-source tool-augmented baseline by 23.80 percentage points. Compared with tool-only supervised fine-tuning, it also reduces total test-set tool calls from 4949 to 3486. These results show that reliable multimodal tool reasoning depends on provenance-aware use of observations, not on more tool calls alone. 

---
# FocuSFT: Bilevel Optimization for Dilution-Aware Long-Context Fine-Tuning 

**Authors**: Zehua Pei, Hui-Ling Zhen, Xianzhi Yu, Sinno Jialin Pan, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09932)  

**Abstract**: Large language models can now process increasingly long inputs, yet their ability to effectively use information spread across long contexts remains limited. We trace this gap to how attention budget is spent during supervised fine-tuning (SFT) on long sequences: positional biases and attention sinks cause the model to allocate most of its attention to positionally privileged tokens rather than semantically relevant content. This training-time attention dilution (the starvation of content tokens in the attention distribution) weakens the gradient signal, limiting the model's ability to learn robust long-context capabilities. We introduce FocuSFT, a bilevel optimization framework that addresses this problem at training time. An inner loop adapts lightweight fast-weight parameters on the training context to form a parametric memory that concentrates attention on relevant content, and the outer loop performs SFT conditioned on this sharpened representation. Both loops apply bidirectional attention over context tokens while preserving causal masking for responses, reducing the causal asymmetry that gives rise to attention sinks and aligning inner-outer behavior. On BABILong, FocuSFT improves accuracy by up to +14pp across 4K--32K context lengths; on RULER, it raises CWE aggregation from 72.9\% to 81.1\% at 16K; and on GPQA with agentic tool use, it yields a 24\% relative gain in pass@1. Attention analysis shows that FocuSFT reduces attention sink mass by 529$\times$ and triples context engagement during training. Code: this https URL 

---
# PruneTIR: Inference-Time Tool Call Pruning for Effective yet Efficient Tool-Integrated Reasoning 

**Authors**: Luan Zhang, Dandan Song, Zhijing Wu, Zhengyu Chen, Chen Zhang, Yuhang Tian, Huipeng Ma, Chenhao Li, Changzhi Zhou, Xudong Li, Shuhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09931)  

**Abstract**: Tool-integrated reasoning (TIR) enables large language models (LLMs) to enhance their capabilities by interacting with external tools, such as code interpreters (CI). Most recent studies focus on exploring various methods to equip LLMs with the ability to use tools. However, how to further boost the reasoning ability of already tool-capable LLMs at inference time remains underexplored. Improving reasoning at inference time requires no additional training and can help LLMs better leverage tools to solve problems. We observe that, during tool-capable LLM inference, both the number and the proportion of erroneous tool calls are negatively correlated with answer correctness. Moreover, erroneous tool calls are typically resolved successfully within a few subsequent turns. If not, LLMs often struggle to resolve such errors even with many additional turns. Building on the above observations, we propose PruneTIR, a rather effective yet efficient framework that enhances the tool-integrated reasoning at inference time. During LLM inference, PruneTIR prunes trajectories, resamples tool calls, and suspends tool usage through three components: Success-Triggered Pruning, Stuck-Triggered Pruning and Resampling, and Retry-Triggered Tool Suspension. These three components enable PruneTIR to mitigate the negative impact of erroneous tool calls and prevent LLMs from getting stuck in repeated failed resolution attempts, thereby improving overall LLM performance. Extensive experimental results demonstrate the effectiveness of PruneTIR, which significantly improves Pass@1 and efficiency while reducing the working context length for tool-capable LLMs. 

---
# Evolving Knowledge Distillation for Lightweight Neural Machine Translation 

**Authors**: Xuewen Zhang, Haixiao Zhang, Xinlong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09924)  

**Abstract**: Recent advancements in Neural Machine Translation (NMT) have significantly improved translation quality. However, the increasing size and complexity of state-of-the-art models present significant challenges for deployment on resource-limited devices. Knowledge distillation (KD) is a promising approach for compressing models, but its effectiveness diminishes when there is a large capacity gap between teacher and student models. To address this issue, we propose Evolving Knowledge Distillation (EKD), a progressive training framework in which the student model learns from a sequence of teachers with gradually increasing capacities. Experiments on IWSLT-14, WMT-17, and WMT-23 benchmarks show that EKD leads to consistent improvements at each stage. On IWSLT-14, the final student achieves a BLEU score of 34.24, narrowing the gap to the strongest teacher (34.32 BLEU) to just 0.08 BLEU. Similar trends are observed on other datasets. These results demonstrate that EKD effectively bridges the capacity gap, enabling compact models to achieve performance close to that of much larger teacher this http URL and models are available at this https URL. 

---
# Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs 

**Authors**: Wu Li, Yigeng Zhou, Zesheng Shi, Yequan Wang, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09922)  

**Abstract**: While recent self-training approaches have reduced reliance on human-labeled data for aligning LLMs, they still face critical limitations: (i) sensitivity to synthetic data quality, leading to instability and bias amplification in iterative training; (ii) ineffective optimization due to a diminishing gap between positive and negative responses over successive training iterations. In this paper, we propose Team-based self-Play with dual Adaptive Weighting (TPAW), a novel self-play algorithm designed to improve alignment in a fully self-supervised setting. TPAW adopts a team-based framework in which the current policy model both collaborates with and competes against historical checkpoints, promoting more stable and efficient optimization. To further enhance learning, we design two adaptive weighting mechanisms: (i) a response reweighting scheme that adjusts the importance of target responses, and (ii) a player weighting strategy that dynamically modulates each team member's contribution during training. Initialized from a SFT model, TPAW iteratively refines alignment without requiring additional human supervision. Experimental results demonstrate that TPAW consistently outperforms existing baselines across various base models and LLM benchmarks. Our code is publicly available at this https URL. 

---
# Position: Academic Conferences are Potentially Facing Denominator Gaming Caused by Fully Automated Scientific Agents 

**Authors**: Rong Shan, Te Gao, Hang Zheng, Yunjia Xi, Jiachen Zhu, Zeyu Zheng, Yong Yu, Weinan Zhang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.09915)  

**Abstract**: The implicit policy of maintaining relatively stable acceptance rates at top AI conferences, despite exponentially growing submissions, introduces a critical structural vulnerability. This position paper characterizes a new systemic threat we term Agentic Denominator Gaming, in which a malicious actor deploys AI agents to generate and submit a large volume of superficially plausible but low-quality papers. Crucially, their objective is not the acceptance of low-quality papers, but rather to inflate the submission denominator and overwhelm reviewing capacity. Under a relatively stable acceptance rate, this dilution can systematically increase the publication probability of a small, targeted set of legitimate papers. We analyze the practical feasibility of this threat and its broader consequences, including intensified reviewer burnout, degraded review quality, and the emergence of industrialized automated agent mills. Finally, we propose and evaluate a range of mitigation strategies, and argue that durable protection will require system-level policy and incentive reforms, rather than relying primarily on technical detection alone. 

---
# Pseudo-Deliberation in Language Models: When Reasoning Fails to Align Values and Actions 

**Authors**: Sushrita Rakshit, Hanwen Zhang, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.09893)  

**Abstract**: Large language models (LLMs) are often evaluated based on their stated values, yet these do not reliably translate into their actions, a discrepancy termed "value-action gap." In this work, we argue that this gap persists even under explicit reasoning, revealing a deeper failure mode we call "Pseudo-Deliberation": the appearance of principled reasoning without corresponding behavioral alignment. To study this systematically, we introduce VALDI, a framework for measuring alignment between stated values and generated dialogue. VALDI includes 4,941 human-centered scenarios across five domains, three tasks that elicit value articulation, reasoning, and action, and five metrics for quantifying value adherence. Across both proprietary and open-source LLMs, we observe consistent misalignment between expressed values and downstream dialogues. To investigate intervention strategies, we propose VIVALDI, a multi-agent value auditor that intervenes at different stages of generation. 

---
# The Association of Transformer-based Sentiment Analysis with Symptom Distress and Deterioration in Routine Psychotherapy Care 

**Authors**: Douglas K. Faust, Peter Awad, Alexandre Vaz, Tony Rousmaniere  

**Link**: [PDF](https://arxiv.org/pdf/2605.09838)  

**Abstract**: Sentiment analysis has been of long-standing interest in psychotherapy research. Recently, the Transformer deep learning architecture has produced text-based sentiment analysis models that are highly accurate and context-aware. These models have been explored as proxies for emotion measurement instruments in psychotherapy, but not investigated as stand-alone psychometric tools. Using proposed utterance-level and session-level sentiment features derived from a fine-grained sentiment model on a large corpus of psychotherapy sessions (N = 751), we investigate the distribution of session aggregated sentiment scores. Further, we characterize the relationship of these features to individual components and the overall score of the OQ-45 instrument and find that this sentiment feature is most strongly correlated to components related to emotional valence in directionally intuitive ways. Finally, we report that there are statistically significant differences between the sentiment distributions for patients flagged as at risk of deterioration or dropping out of care via either the OQ Rational or Empirical outcome models. These correlations to a fully-validated psychometric instrument demonstrate that these proposed sentiment features are, at least, adjunctive measures of client distress and deterioration. 

---
# Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants 

**Authors**: Joseph Suh, Ayush Raj, Minwoo Kang, Serina Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09808)  

**Abstract**: User simulators are increasingly leveraged to build interactive AI assistants, yet how to measure the quality of these simulators remains an open question. In this work, we show how simulator quality can be quantified in terms of its downstream utility: how an LLM assistant trained with this user simulator performs in the wild when interacting with real humans. In a controlled experiment where only the user simulator varies, we train LLM assistants via reinforcement learning against a spectrum of simulators, from an LLM prompted to role-play a user to one fine-tuned on human utterances from WildChat. As evaluation, we measure pairwise win rates in a user study with 283 participants and on WildBench, a benchmark derived from real human--AI conversations. Training against the role-playing LLM yields an assistant statistically indistinguishable from the initial assistant in our user study (51% win rate), whereas training against the fine-tuned simulator yields significant gains (58% over the initial and 57% over the one trained against role-playing). Closer inspection reveals three further patterns: methods for making role-playing LLMs more realistic (e.g., persona conditioning) improve trained assistants but do not close the gap to the fine-tuned simulator; scaling the simulator's model size benefits the fine-tuned simulator but yields no gain for role-playing ones; and assistants trained against role-playing simulators fail to generalize when paired with other simulators at test time, while the one trained against fine-tuned simulator does. Together, these results argue for grounding user simulators in real human behavior and measuring their quality by their downstream effect on real users. 

---
# cantnlp@DravidianLangTech 2026: organic domain adaptation improves multi-class hope speech detection in Tulu 

**Authors**: Andrew Li, Sidney Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.09795)  

**Abstract**: This paper presents our systems and results for the Hope Speech Detection in Code-Mixed Tulu Language shared task at the Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages (DravidianLangTech-2026). We trained an XLM-RoBERTa-based text classification system for detecting hope speech in code-mixed Tulu social media comments. We compared this organically adapted hope speech detection model with our baseline model. On the development set, the organically adapted model outperformed the baseline system. While our submitted systems performed more modestly on the official test set, these results suggest that further adapting XLM-RoBERTa on organically collected Tulu social media text containing code-mixed and mixed-script variation can improve hope speech detection in code-mixed Tulu. 

---
# Exploitation Without Deception: Dark Triad Feature Steering Reveals Separable Antisocial Circuits in Language Models 

**Authors**: Cameron Berg, Roshni Lulla  

**Link**: [PDF](https://arxiv.org/pdf/2605.09773)  

**Abstract**: We use sparse autoencoder (SAE) feature steering to amplify Dark Triad personality traits (Machiavellianism, narcissism, and psychopathy) in Llama-3.3-70B-Instruct and evaluate the resulting behavioral changes across five psychological instruments. The steered model becomes substantially more exploitative, aggressive, and callous on novel behavioral scenarios (d=10.62) while its cognitive empathy remains intact, reproducing the empathy dissociation characteristic of human Dark Triad populations. Critically, strategic deception is completely unaffected across all features, suggesting that exploitation and deception may operate through dissociable computational pathways in large language models. Individual feature analysis reveals non-redundant encoding, with each feature driving distinct antisocial mechanisms through separable computational pathways. We also show that feature discovery method itself modulates intervention depth: contrastively-discovered features change both self-report and behavior, while semantically-searched features change only self-report (d=12.65 between methods on behavior). These findings suggest that antisocial tendencies in at least one large language model comprise dissociable components rather than a unified construct, with implications for how such tendencies should be detected, measured, and controlled. 

---
# ConFit v3: Improving Resume-Job Matching with LLM-based Re-Ranking 

**Authors**: Xiao Yu, Ruize Xu, Chengyuan Xue, Junyu Chen, Matthew So, Shijun Ma, Bo Liu, Xiangye Liang, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09760)  

**Abstract**: A reliable resume-job matching system helps a company find suitable candidates from a pool of resumes and helps a job seeker find relevant jobs from a list of job posts. While recent advances in embedding-based methods such as ConFit and ConFit v2 can efficiently retrieve candidates at scale, the lack of controllability and explainability limits their real-world adaptations. LLM-based re-rankers can address these limitations through reasoning, but existing training recipes are developed on short-document benchmarks and do not account for noise in real-world recruiting data. In this work, we first conduct a systematic analysis over the LLM re-ranker training pipeline for person-job fit, covering inference algorithm design, RL algorithm selection, data processing, and SFT distillation. We find that using multi-pass re-ranking, training with listwise RL objectives, removing noisy samples, and distilling from a stronger LLM before RL significantly improves re-ranking performance. We then aggregate these findings to train ConFit v3 with Qwen3-8B and Qwen3-32B on real-world person-job fit datasets, and find significant improvements over existing best person-job fit systems as well as strong LLMs such as GPT-5 and Claude Opus-4.5. We hope our findings provide useful insights for future research on adapting LLM-based re-rankers to person-job fit systems. 

---
# Language Models Without a Trainable Input Embedding Table: Learning from Fixed Minimal Binary Token Codes 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.09751)  

**Abstract**: Trainable input embedding tables are a standard component of modern language models. We ask whether they are actually necessary at the input interface. For a vocabulary of size $V$, exact token identity requires only $K=\lceil \log_2 V\rceil$ bits. We replace the usual trainable $V\times d_{\text{model}}$ input embedding matrix with fixed minimal binary token codes and a zero-parameter lift to model width. In our main setting, $V=65{,}536$, so $K=16$, and tokens are represented by fixed 16-dimensional binary codes tiled to $d_{\text{model}}=1024$. We also evaluate a fully table-free variant in which codes are generated from token IDs on the fly and randomly recoded by an invertible affine transform over $\mathbb{F}_2^K$. Across matched 32-layer decoder-only models trained on approximately 17B tokens and evaluated over three independent training seeds, fixed minimal codes achieve comparable held-out validation perplexity to a standard learned-input baseline while removing 67.1M trainable input parameters. The fixed-code runs have a lower mean validation perplexity in our experiments, 2.36 versus 2.44, but the observed gap is within the measured seed-to-seed variation of 4.8\%; we therefore interpret the result as evidence that the trainable input table is not necessary, rather than as a statistically resolved superiority claim. The table-free affine-recoded variant remains close at 2.39 despite a slightly shorter training run. These results show that, in this regime, a trainable input embedding table is not necessary for useful language modeling. The output projection remains standard and trainable. 

---
# The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods 

**Authors**: Sanket Badhe, Priyanka Tiwari, Deep Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.09739)  

**Abstract**: Large Language Models are increasingly used as zero-shot classifiers in complex reasoning tasks. However, standard constrained decoding suffers from a phenomenon we define as Renormalization Bias. When a model is restricted to a small set of target labels, the standard softmax operation discards the probability mass assigned to semantic synonyms in the original distribution. This loss of information, which we call the Silent Vote, results in artificial overconfidence and poor calibration.
We propose Semantic Softmax, an inference-time layer that recovers this lost information by aggregating the scores of the semantic neighborhood surrounding each target label. We evaluate this approach on Qwen-3 and Phi-4-mini models using GoEmotions and Civil Comments datasets. Our results demonstrate consistent improvements across all evaluation metrics: Semantic Softmax substantially reduces Expected Calibration Error (ECE) and Brier Score, while simultaneously enhancing discriminative performance in terms of AUROC and Macro-F1. By accounting for linguistic nuances, our method provides a more calibrated and accurate alternative for zero-shot classification. 

---
# MedMeta: A Benchmark for LLMs in Synthesizing Meta-Analysis Conclusion from Medical Studies 

**Authors**: Huy Hoang Ha, Benoit Favre, Francois Portet  

**Link**: [PDF](https://arxiv.org/pdf/2605.09661)  

**Abstract**: Large language models (LLMs) have saturated standard medical benchmarks that test factual recall, yet their ability to perform higher-order reasoning, such as synthesizing evidence from multiple sources, remains critically under-explored. To address this gap, we introduce MedMeta, the first benchmark designed to evaluate an LLM's ability to generate conclusions from medical meta-analyses using only the abstracts of cited studies. MedMeta comprises 81 meta-analyses from PubMed (2018--2025) and evaluates models using two distinct workflows: a Retrieval-Augmented Generation (Golden-RAG) setting with ground-truth abstracts, and a Parametric-only approach relying on internal knowledge. Our evaluation framework is validated by a well-structured analysis showing our LLM-as-a-judge protocol strongly aligns with human expert ratings, as evidenced by high Pearson's r correlation (0.81) and Bland-Altman analysis revealing negligible systematic bias, establishing it as a reliable proxy for scalable evaluation. Our findings underscore the critical importance of information grounding: the Golden-RAG workflow consistently and significantly outperforms the Parametric-only approach across models. In contrast, the benefits of domain-specific fine-tuning are marginal and largely neutralized when external material is provided. Furthermore, stress tests show that all models, regardless of architecture, fail to identify and reject negated evidence, highlighting a critical vulnerability in current RAG systems. Notably, even under ideal RAG conditions, current LLMs achieve only slightly above-average performance (~2.7/5.0). MedMeta provides a challenging new benchmark for evidence synthesis and demonstrates that for clinical applications, developing robust RAG systems is a more promising direction than model specialization alone. 

---
# K12-KGraph: A Curriculum-Aligned Knowledge Graph for Benchmarking and Training Educational LLMs 

**Authors**: Hao Liang, Qihan Lin, Zhaoyang Han, Xiaochen Ma, Zhen Hao Wong, Meiyi Qiang, Linzhuang Sun, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09635)  

**Abstract**: Large language models (LLMs) are increasingly used in K-12 education, yet existing benchmarks such as C-Eval, CMMLU, GaokaoBench, and EduEval mainly evaluate factual recall through exam-style question answering. Effective educational AI additionally requires curriculum cognition: understanding how knowledge is structured through prerequisite chains, concept taxonomies, experiment-concept links, and pedagogical sequencing. To address this gap, we introduce K12-KGraph, a curriculum-aligned knowledge graph extracted from official People's Education Press textbooks across mathematics, physics, chemistry, and biology from primary to high school. The graph contains seven node types (Concept, Skill, Experiment, Exercise, Section, Chapter, Book) and nine relation types covering taxonomy, prerequisite, association, verification, assessment, location, and order. Based on this graph, we construct two resources: (1) K12-Bench, a 23,640-question multi-select benchmark spanning five graph-derived task families (Ground, Prereq, Neighbor, Evidence, and Locate); and (2) K12-Train, a KG-guided supervised fine-tuning corpus of approximately 2,300 QA pairs synthesized from graph structure and node attributes. Experiments reveal substantial deficiencies in curriculum cognition: on K12-Bench, Gemini-3-Flash achieves only 57% exact match, while the best open-source model, Gemma-4-31B-IT, reaches 46%. Under a strictly matched 2,300-sample SFT budget on Qwen3-4B-Base and Llama-3.1-8B-Base, K12-Train consistently outperforms equally sized subsets from eight mainstream instruction-tuning corpora on both GaokaoBench and EduEval, demonstrating that curriculum-structured supervision is highly sample-efficient for educational tuning. We release the graph, benchmark, training data, and full construction pipeline. 

---
# Can We Trust LLMs for Mental Health Screening? Consistency, ASR Robustness, and Evidence Faithfulness 

**Authors**: Erfan Loweimi, Sofia de la Fuente Garcia, Samira Loveymi, Hadi Daneshvar, Saturnino Luz  

**Link**: [PDF](https://arxiv.org/pdf/2605.09634)  

**Abstract**: LLMs can estimate Hospital Anxiety and Depression Scale (HADS) scores from speech in a zero-shot manner, but clinical deployment requires reliability across three dimensions: intra-model consistency, ASR robustness, and evidence faithfulness. We evaluate three LLMs (Phi-4, Gemma-2-9B, and Llama-3.1-8B) on 111 English-speaking participants using ground-truth transcripts and three Whisper ASR variants (Large, Medium, Small), with three independent runs per model-condition pair. We find that (i) Phi-4 and Gemma-2-9B achieve excellent intra-model consistency (ICC > 0.89) with minimal degradation under ASR; (ii) Llama-3.1-8B shows ASR-fragile consistency, with ICC dropping from 0.82 to 0.36 at 10% WER; (iii) predictive validity is largely preserved under ASR for robust models; and (iv) keyword groundedness exceeds 93% for Phi-4 and Gemma-2-9B but falls to 77-81% for Llama-3.1-8B. Inter-model keyword agreement is far lower than score-level agreement, revealing a score-evidence dissociation with implications for clinical interpretability. 

---
# Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models 

**Authors**: Lin Zheng, Vasilisa Bashlovkina, Timothy Dozat, Dan Garrette, Laura Rimell, Joshua Maynez  

**Link**: [PDF](https://arxiv.org/pdf/2605.09630)  

**Abstract**: Tokenizer-free language models eliminate the tokenizer step of the language modeling pipeline by operating directly on bytes; patch-based variants further aggregate contiguous byte spans into patches for efficiency. However, the average patch size chosen at the model design stage governs a tight trade-off: larger patches reduce compute and KV-cache footprint, but degrade modeling quality. We trace this trade-off to patch lag: until a patch is fully observed, byte predictions within it must rely on a stale representation from the previous patch to preserve causality; this lag widens as patches grow larger. We introduce Scratchpad Patching (SP), which inserts transient scratchpads inside each patch to aggregate the bytes seen so far and refresh patch-level context for subsequent predictions. SP triggers scratchpads using next-byte prediction entropy, selectively allocating compute to information-dense regions and enabling post-hoc adjustment of inference-time compute. Across experiments on natural language and code, SP improves model quality at the same patch size; for example, even at $16$ bytes per patch, SP-augmented models match or closely approach the byte-level baseline on downstream evaluations while using a $16\times$ smaller KV cache over patches and $3$-$4\times$ less inference compute. 

---
# Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A Matched-Ceiling Study of Open-Weight LLM Reasoning Protocols 

**Authors**: Julia Hu, Alfred Shen, Kumar Lakshmipathi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09618)  

**Abstract**: When should a language model answer directly, sample and vote, or engage in multi-agent debate? Recent work shows voting often explains much of the gain attributed to debate, while selective-debate systems activate deliberation only on uncertain examples. We ask: under a matched ceiling on generated tokens (960 per example), how much per-example routing headroom exists, and how much is recoverable from cheap pre-deliberation signals?
We evaluate greedy decoding, three-sample voting, and a two-agent critique-revise debate on MuSiQue and GSM8K using Llama 3.1 8B Instruct and Ministral 3 8B Instruct. On MuSiQue, an oracle selecting the correct protocol per example gains +14.0 and +13.7 pp over the best fixed one. The best fixed protocol is model- and dataset-dependent: each (model, dataset) cell has a different winner.
This headroom is hard to recover from cheap ex-ante signals. A vote-entropy threshold is the only controller that directionally beats the best fixed protocol on both models (+1.3 and +1.7 pp), though individual paired-bootstrap CIs include zero. A joint analysis (meta-analysis +1.6 pp, p=0.125; Bayesian P(both>0)=0.59) is directionally consistent but not significant. Learned controllers (LR, GBT) do not outperform the threshold.
The key finding is structural: vote entropy predicts where debate is safe, not where debate is needed. High entropy sharply reduces debate backfire, but 66% of debate-helpful examples (31/47) occur when voting is unanimous but wrong. A single-prompt self-critique probe on Llama flips the answer in 127/127 unanimous cases, yielding zero mutual information with the debate-helpful label; we cannot rule out a prompt-compliance artifact, but either interpretation disqualifies the probe as a router. Recovering the remaining headroom requires behavioral probes that avoid format-compliance confounds at the 8B scale. 

---
# Byte-Exact Deduplication in Retrieval-Augmented Generation: A Three-Regime Empirical Analysis Across Public Benchmarks 

**Authors**: Sietse Schelpe  

**Link**: [PDF](https://arxiv.org/pdf/2605.09611)  

**Abstract**: This preprint presents an empirical analysis of byte-exact chunk-level deduplication in Retrieval-Augmented Generation (RAG) pipelines. We measure context reduction across three distinct operating regimes: clean academic retrieval (0.16% byte reduction on 22.2M BeIR passages), constructed enterprise patterns (24.03% reduction), and multi-turn conversational AI (80.34% reduction). To validate quality preservation, we conducted a cross-vendor 5-judge calibrated panel evaluation across four production APIs (Google Gemini 2.5 Flash, Anthropic Claude Sonnet 4.6, Meta Llama 3.3 70B, and OpenAI GPT-5.1). Applying a five-category human-in-the-loop noise-removal protocol to panel-majority materially different (MAT) pairs, we establish that byte-exact deduplication introduces zero measurable quality regression. Post-audit, all four vendors clear the strict <5% Wilson 95% upper-bound MAT threshold in both the clean and high-redundancy RAG regimes. This work demonstrates that substantial inference compute savings can be achieved deterministically without compromising evaluation-grade model quality. 

---
# Edit-Based Refinement for Parallel Masked Diffusion Language Models 

**Authors**: Houxing Ren, Mingjie Zhan, Zimu Lu, Ke Wang, Yunqiao Yang, Haotian Hou, Junting Pan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09603)  

**Abstract**: Masked diffusion language models enable parallel token generation and offer improved decoding efficiency over autoregressive models. However, their performance degrades significantly when generating multiple tokens simultaneously, due to a mismatch between token-level training objectives and joint sequence consistency. In this paper, we propose ME-DLM, an edit-based refinement framework that augments diffusion generation with lightweight post-editing steps. After producing an initial complete response, the model refines it through minimal edit operations, including replacement, deletion, and insertion, conditioned on the full sequence. Training supervision is derived from edit distance, providing a deterministic signal under a fixed canonicalization scheme for learning minimal corrections. This approach encourages sequence-level consistency through globally conditioned edits while preserving the efficiency benefits of parallel diffusion decoding. Extensive experiments demonstrate that ME-DLM improves the quality and robustness of multi-token parallel generation. In particular, when built upon LLaDA, our method achieves consistent gains of 11.6 points on HumanEval and 33.6 points on GSM8K while using one-eighth of the total diffusion steps. Code is available at this https URL. 

---
# CLR-voyance: Reinforcing Open-Ended Reasoning for Inpatient Clinical Decision Support with Outcome-Aware Rubrics 

**Authors**: Aishik Nagar, Arun-Kumar Kaliya-Perumal, Yu-Hsuan Han, Andrew Sheng-Han Huang, Kristen Kee, Yushi Cao, Yiming Chen, Hongchao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09584)  

**Abstract**: Inpatient clinical reasoning is a sequential decision under partial observability: the clinician sees the admission so far and must choose the next action whose downstream consequences are not yet visible. Existing clinical-LLM evaluations and RL rewards signals collapse this into closed-form retrieval, clinical journey leakage, or unanchored LLM-as-judge scoring. We introduce CLR-voyance, a framework that reformulates inpatient reasoning as a Partially Observable Markov Decision Process (POMDP) and supervises it with rewards that are simultaneously outcome-grounded and clinician-validated. We instantiate the formulation as CLR-POMDP, which partitions successful patient journeys into a policy-visible past and an oracle-only future. Using the past information, an oracle LLM generates a case-specific query-answer pair, and the first adaptive rubric for clinical reasoning which is verifiable in the future of the patient journey. These rubrics are used for both post-training and evaluation of models for inpatient clinical reasoning. We post-train Qwen3-8B and MedGemma-4B with GRPO followed by model merging, yielding state-of-the-art inpatient clinical reasoning while retaining generalist capabilities. CLR-voyance-8B achieves 84.91% on CLR-POMDP, ahead of frontier medical reasoning models like GPT-5 (77.83%) and MedGemma-27B (66.66%) and has comparable or better performance on existing medical benchmarks. To ensure a clinically meaningful setting, we conduct a large-scale clinician alignment study, where physicians curate per-case rubrics, grade candidate responses, and provide blinded pairwise preferences of model reasoning. This study provides insights on clinical LLM-as-a-judge and clinical preference-model selection, which can inform the community at large. CLR-voyance has been deployed for 6+ months at a partner public hospital, drafting thousands of reasoning-heavy inpatient notes. 

---
# Towards Compact Sign Language Translation: Frame Rate and Model Size Trade-offs 

**Authors**: Kuanwei Chen, Mengfeng Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2605.09554)  

**Abstract**: Sign Language Translation (SLT) converts sign language videos into spoken-language text, bridging communication between Deaf and hearing communities. Current gloss-free approaches rely on large encoder-decoder models, limiting deployment. We propose a compact 77M-parameter pipeline that couples MMPose skeletal pose extraction with a single linear projection into T5-small. By varying the input frame rate, we expose a practical efficiency trade-off: at 12 fps the model halves its sequence length, achieving a 75% reduction in encoder quadratic self-attention computational complexity while incurring only a modest BLEU-4 drop (9.53 vs. 10.06 at 24 fps on How2Sign). Our system is roughly 3x smaller than prior T5-base systems, demonstrating that a lightweight architecture can remain competitive without hierarchical encoders or large-scale models. 

---
# Crosslingual On-Policy Self-Distillation for Multilingual Reasoning 

**Authors**: Yihong Liu, Raoyuan Zhao, Michael A. Hedderich, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2605.09548)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in mathematical reasoning, but this ability is not equally accessible across languages. Especially low-resource languages exhibit much lower reasoning performance. To address this, we propose Crosslingual On-Policy Self-Distillation (COPSD), which transfers a model's own high-resource reasoning behavior to low-resource languages. COPSD uses the same model as student and teacher: the student sees only the low-resource problem, while the teacher receives privileged crosslingual context, including the problem translation and reference solution in English. Training minimizes full-distribution token-level divergence on the student's own rollouts, providing dense supervision while avoiding the sparsity and instability of outcome-only reinforcement learning (RL). Experiments on 17 low-resource African languages show that COPSD consistently improves low-resource mathematical reasoning across model sizes and substantially outperforms Group Relative Policy Optimization (GRPO). Further analyses show that COPSD improves answer-format adherence, strengthens test-time scaling, and generalizes to harder multilingual reasoning benchmarks, with especially large gains for lower-resource languages. We make our code and data available at: this https URL. 

---
# TacoMAS: Test-Time Co-Evolution of Topology and Capability in LLM-based Multi-Agent Systems 

**Authors**: Chen Xu, Yicheng Hu, Ruizi Wang, Xinyu Lin, Wenjie Wang, Dongrui Liu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.09539)  

**Abstract**: Multi-agent systems (MAS) have emerged as a promising paradigm for solving complex tasks. Recent work has explored self-evolving MAS that automatically optimize agent capabilities or communication topologies. However, existing methods either learn a topology that remains fixed at inference time or adapt only the topology or capability during inference. We empirically and theoretically show that effective test-time evolution requires jointly adapting both axes, but on different time scales: capabilities should update rapidly to handle emerging subtasks, while the topology should evolve more slowly to preserve coordination stability. We then introduce TacoMAS, a test-time co-evolution framework for dynamic MAS. TacoMAS formulates MAS inference as a task of online graph adaptation, where nodes represent agents with role-specific capabilities and edges define their communication topology. During inference, a fast capability loop updates agent expertise using trajectory-level feedback, while a slow meta-LLM-driven topology loop performs agents' birth-death operations on MAS, including edge edit, agent addition, and agent removal. We further show that this fast-slow design drives MAS evolution toward a task-conditioned stable equilibrium. Experiments on four benchmarks demonstrate that TacoMAS outperforms nearly 20 multi-agent baselines, achieving an average improvement of 13.3% over the strongest baseline. The codes are released at this https URL. 

---
# TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM 

**Authors**: Haoyang Zhou, Li Kong, Shijie Ren, Xiting Wang, Shuang Liang, Guowei Wang, Zhenxuan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.09536)  

**Abstract**: Diffusion large language models (dLLMs) offer a promising paradigm for parallel text generation, but in practice they face an accuracy-parallelism trade-off, where increasing tokens per forward (TPF) often degrades generation quality. Existing acceleration methods often gain speed at the cost of accuracy. To address this limitation, we propose TAD, a Temporal-Aware trajectory self-Distillation framework. During data construction, we condition a teacher model on both the prompt and the ground-truth response to generate decoding trajectories, recording the intermediate masked states throughout the process. Based on how many decoding steps remain before each masked token is revealed, we partition masked positions into near and distant subsets. For near tokens, we train the student with a hard cross-entropy loss using the teacher trajectory tokens as labels, encouraging confident predictions for tokens that are about to be decoded. For distant tokens, we apply a soft KL divergence loss between the teacher and student token distributions, providing softer supervision and preserving future planning knowledge. This temporal-aware partition naturally gives rise to two deployment configurations: a Quality model that prioritizes accuracy and a Speed model that favors more aggressive acceleration. Experiments show that TAD consistently improves the accuracy-parallelism trade-off. On LLaDA, it raises average accuracy from 46.2\% to 51.6\% with the Quality model and average AUP from 46.2 to 257.1 with the Speed model. Our code is available at: this https URL 

---
# Assessment of RAG and Fine-Tuning for Industrial Question-Answering-Applications 

**Authors**: Jakob Sturm, Josef Pichlmeier, Christian Bernhard, Maka Karalashvili, Johannes Klepsch, Georg Groh, Andre Luckow  

**Link**: [PDF](https://arxiv.org/pdf/2605.09533)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in enterprise question-answering (QA) systems, requiring adaptation to domain-specific knowledge. Among the most prevalent methods for incorporating such knowledge are Retrieval-Augmented Generation (RAG) and fine-tuning (FT). Yet, from a cost-accuracy trade-off perspective, it remains unclear which approach best suits industry scenarios. This study examines the impact of RAG and FT on two closed datasets specific to the automotive industry, assessing answer quality and operational costs. We extend the Cost-of-Pass framework proposed by Erol et al. (arXiv:2504.13359) to jointly assess output quality, generation cost, and user interaction cost. Our findings reveal that while premium models perform best out of the box, open-source models can achieve comparable quality when enhanced with RAG. Overall, RAG emerges as the most effective and cost-efficient adaptation method for both closed- and open-source models. 

---
# Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal 

**Authors**: Aojie Yuan, Zhiyuan Julian Su, Haiyue Zhang, Yi Nian, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.09502)  

**Abstract**: Chain-of-thought (CoT) prompting assumes that generated reasoning reflects a model's internal computation. We show this assumption is wrong in a specific, measurable way: models internally detect their own reasoning errors but outwardly express confidence in them. A linear probe on hidden states predicts trace correctness with 0.95 AUROC -- from the very first reasoning step (0.79) -- while verbalized confidence for wrong traces is 4.55/5, nearly identical to correct ones (4.87/5). A text-surface classifier achieves only 0.59 on the same data, confirming a 0.20-point gap invisible in the generated text. This hidden error awareness holds across three model families (Qwen, Llama, Phi), 1.5B-72B parameters, and RL-trained reasoning models (DeepSeek-R1, 0.852 AUROC). The natural question is whether this signal can fix the errors it detects. It cannot. Four interventions -- activation steering, probe-guided best-of-N, self-correction, and activation patching -- all fail; patching destroys output coherence entirely. The signal is diagnostic, not causal: a readout of computation quality, not a lever to redirect it. This delineates a boundary for mechanistic interpretability: error representations during reasoning are fundamentally different from the factual knowledge representations that prior work has successfully edited. 

---
# Beyond Language: Format-Agnostic Reasoning Subspaces in Large Language Models 

**Authors**: Aojie Yuan, Zhiyuan Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.09496)  

**Abstract**: Large language models represent the same reasoning in vastly different surface forms -- English prose, Python code, mathematical notation -- yet whether they share a common internal substrate across these symbolic systems remains unknown. We introduce the TriForm Benchmark (18 concepts x 6 forms x 3 instances = 324 stimuli) and study five LLMs (1.6B-8B) across three architecture families. Using permutation-corrected RSA, cross-form probing, and activation patching, we find converging evidence for a Format-Agnostic Reasoning Subspace (FARS) in middle layers. We make FARS concrete: concept-centroid PCA extracts a 10-dimensional subspace that amplifies concept structure 3x while suppressing form information to near zero. Replacing only these 10 dimensions during cross-form patching preserves 90-96% of model output -- far exceeding both full activation replacement (44-56%) and variance-maximizing PCA (60-74%) -- while ablating them causes targeted disruption. FARS generalizes to held-out concepts and converges across architectures (CCA > 0.79 for all model pairs), providing within-modality evidence for the Platonic Representation Hypothesis. We further discover a declarative-procedural asymmetry: representations are far more compatible between prose and mathematics than between either and code, suggesting that the critical axis of divergence is not linguistic vs. formal but declarative vs. procedural. 

---
# APCD: Adaptive Path-Contrastive Decoding for Reliable Large Language Model Generation 

**Authors**: Tianyu Zheng, Hong Wu, Jiaji Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.09492)  

**Abstract**: Large language models (LLMs) often suffer from hallucinations due to error accumulation in autoregressive decoding, where suboptimal early token choices misguide subsequent generation. Although multi-path decoding can improve robustness by exploring alternative trajectories, existing methods lack principled strategies for determining when to branch and how to regulate inter-path interactions. We propose Adaptive Path-Contrastive Decoding (APCD), a multi-path decoding framework that improves output reliability through adaptive exploration and controlled path interaction. APCD consists of two components: (1) Entropy-Driven Path Expansion, which delays branching until predictive uncertainty - measured by Shannon entropy over top candidate tokens - indicates multiple plausible continuations; and (2) Divergence-Aware Path Contrast, which encourages diverse reasoning trajectories while dynamically attenuating inter-path influence as prediction distributions diverge. Experiments on eight benchmarks demonstrate improved factual accuracy while maintaining decoding efficiency. Our code is available at this https URL. 

---
# Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning 

**Authors**: Aojie Yuan, Tianqi Shen, Dajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09490)  

**Abstract**: Reasoning LLMs produce thousands of chain-of-thought tokens whose KV cache must reside in scarce GPU HBM. The dominant response -- permanently evicting low-importance tokens -- is catastrophic for reasoning: accuracy collapses to 0-2.5% when half the cache is removed. We ask a different question: must every token live in HBM, or can some live elsewhere? We introduce a semantics-aware memory hierarchy that sorts tokens into four tiers -- HBM, DDR, compressed, and evicted -- using cumulative attention scoring. Low-importance tokens are moved to CPU memory rather than destroyed; before each attention step they are prefetched back at full precision, contributing exactly the same terms as if they had never left the GPU. We formalize this as zero-approximation-error offloading and derive our central finding: accuracy depends solely on how many tokens are permanently discarded (the eviction ratio), not on how many remain in HBM. A controlled 3x3 grid over HBM and eviction ratios confirms this across three model scales (7B-32B) and four benchmarks. With only 3% eviction, the hierarchy retains 91% of full-cache accuracy on GSM8K and 71% on MATH-500 (n=200); at 14B scale it matches the uncompressed baseline (90% vs. 86%) while halving HBM occupancy. A head-to-head reproduction of R-KV -- the current SOTA eviction method -- on our setup achieves only 0-32% at comparable budgets. A system prototype with real GPU-CPU data movement shows that the price of this preservation is modest -- 5-7% transfer overhead -- and scaling analysis projects 2-48 GB HBM savings at production batch sizes. 

---
# A Cognitively Grounded Bayesian Framework for Misinformation Susceptibility 

**Authors**: Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2605.09483)  

**Abstract**: In this (work in progress) paper, we present Bounded Pragmatic Listener (or BPL), a cognitively grounded Bayesian framework for modelling susceptibility to information disorder. BPL extends Rational Speech Act theory with three cognitively motivated bounds derived from the bounded rationality literature with a) a recursion depth bound (that emphasises working memory limits);b) a prior compression parameter (which is oriented at capturing information bottleneck); and c) an availability sample size (that operationalises importance sampling with saliency-weighted proposals). This allows us to test predictions about misinformation susceptibility, annotator disagreement, and the differential vulnerability to mis-, dis-, and mal-information as defined in the Information Disorder framework. We validate BPL on the LIAR and MultiFC benchmarks showcasing competitive veracity classification and experimental support for the depth-mismatch paradox. 

---
# Align and Shine: Building High-Quality Sentence-Aligned Corpora for Multilingual Text Simplification 

**Authors**: Kenji Hilasaca, Nouran Khallaf, Serge Sharoff  

**Link**: [PDF](https://arxiv.org/pdf/2605.09476)  

**Abstract**: Text simplification plays a crucial role in improving the accessibility and comprehensibility of written information for diverse audiences, including language learners and readers with limited literacy. Despite its importance, large-scale, high-quality datasets for training and evaluating text simplification models remain scarce for languages other than English. This paper reports an experimental study on the collection and processing of crowd-sourced simplification data from comparable corpora to construct a corpus suitable for both training and testing text simplification systems across multiple languages (Catalan, English, French, Italian and Spanish). We report mechanisms for sentence-level alignment from document-level data. The resulting dataset of the aligned sentence pairs is publicly available. 

---
# FinMoji: A Framework for Emoji-driven Sentiment Analysis in Financial Social Media 

**Authors**: Ahmed Mahrous, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2605.09469)  

**Abstract**: This paper explores the use of emojis in financial sentiment analysis, focusing on the social media platform StockTwits. Emojis, increasingly prevalent in digital communication, have potential as compact indicators of investor sentiment, which can be critical for predicting market trends. Our study examines whether emojis alone can serve as reliable proxies for financial sentiment and how they compare with traditional text-based analysis. We conduct a series of experiments using logistic regression and transformer models. We further analyze the performance, computational efficiency, and data requirements of emoji-based versus text-based sentiment classification. Using a balanced dataset of about 528,000 emoji-containing StockTwits posts, we find that emoji-only models achieve F1 approximately 0.75, lower than text-emoji combined models, which achieve F1 approximately 0.88, but with far lower computational cost. This is a useful feature in time-sensitive settings such as high-frequency trading. Furthermore, certain emojis and emoji pairs exhibit strong predictive power for market sentiment, demonstrating over 90 percent accuracy in predicting bullish or bearish trends. Finally, our research reveals large statistical differences in emoji usage between financial and general social media contexts, stressing the need for domain-specific sentiment analysis models. 

---
# Beyond Position Bias: Shifting Context Compression from Position-Driven to Semantic-Driven 

**Authors**: Jiwei Tang, Zhijing Huang, Xinyu Zhang, Chen Jason Zhang, Jianxing Yu, Libin Zheng, Rui Meng, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.09463)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks. However, their deployment in long-context scenarios faces high computational overhead and information redundancy. While soft prompt compression has emerged as a promising way to mitigate these costs by compressing sequences into compact embeddings, existing paradigms remain fundamentally constrained by position bias: they primarily rely on learnable tokens insertion at fixed positions or group tokens according to their physical token layout, thereby inducing performance instability and semantic fragmentation. To overcome this bottleneck, we propose Semantic Consistency Context Compression (SeCo), a method that shifts context compression from position-driven to semantic-driven. Rather than constraint by physical token layout, SeCo dynamically anchors compression directly in the semantic space by selecting query-relevant tokens as semantic centers and aggregating remaining tokens via consistency-weighted merging. This design inherently preserves semantic consistency while eliminating position bias. Extensive experiments on 14 benchmarks across two backbone models demonstrate that SeCo consistently shows superiority in downstream tasks, inference latency, and out-of-domain robustness. The code is available at this https URL. 

---
# Key Coverage Matters: Semi-Structured Extraction of OCR Clinical Reports 

**Authors**: Yu Wang, Yingyun Li, Ying Qin, Haiyang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.09440)  

**Abstract**: Clinical reports are often fragmented across healthcare institutions because privacy regulations and data silos limit direct information sharing. When patients seek care at a different hospital, they often carry paper or scanned reports from prior visits. This hinders EHR integration and longitudinal review, and downstream applications that depend on more complete patient records, such as patient management, follow-up care, real-world studies, and clinical-trial matching. Although OCR can digitize such reports, reliable extraction remains challenging because clinical documents are heterogeneous, OCR text is noisy, and many healthcare settings require low-cost on-premise deployment. We formulate this problem as canonical key-conditioned extractive question answering over OCR-derived clinical reports. Because the key fields are neither fixed nor known in advance, the key space is open. We maintain a canonical key inventory through iterative key mining, normalization, clustering, and lightweight human verification, and introduce key coverage as a metric to quantify inventory completeness. Using a 0.2B BERT-based model, experiments on real-world reports from more than 20 hospitals show performance improves monotonically with key coverage. The model achieves F1 scores of 0.839 and 0.893 under exact match and boundary-tolerant matching, respectively, once the Top-90 canonical keys are covered. These results show that key coverage is a dominant factor for end-to-end performance. At Top-90 coverage, our model outperforms a fine-tuned Qwen3-0.6B baseline under exact match. Although our annotated corpus is Chinese, the method relies on the language-agnostic key-value organization of semi-structured clinical reports and can be adapted to other settings given an appropriate canonical key inventory and alias mapping. 

---
# PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on Telegram 

**Authors**: Ahmed Mahrous, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2605.09431)  

**Abstract**: Cryptocurrency pump-and-dump schemes coordinated via Telegram threaten market integrity. However, existing research addressing this specific threat has not yet produced solutions that combine reliable results with fast response. This is in part due to the absence of publicly available, message-level labeled data, as well as design choices. In this paper, we address both issues. In particular, we introduce a corpus of over 280,000 Telegram posts from 39 pump-organizing groups, all manually reviewed to identify 2,246 pump announcements and their targeted cryptocurrency and exchange. Leveraging this dataset, we define two tasks: real-time pump-announcement detection and target cryptocurrency/exchange extraction. For detection, we compare two machine-learning models: a lightweight tree-based LightGBM classifier (F1=0.79, latency=9.4 s/sample) and a transformer-based BGE-M3 (F1=0.83, latency=50 ms/sample). With our proposed approach, we show that message analysis can achieve near-instant pump detection at the level of individual Telegram message windows. Unlike prior work that relies purely on market data and typically detects pumps tens of seconds after abnormal trading activity is observed, our method operates directly on the coordination messages themselves and can be evaluated in microseconds per window on commodity hardware. To our knowledge, we also establish the first benchmark for manipulated coin and exchange extraction. We demonstrate that traditional rule-based extraction methods, widely relied upon in prior literature, are ineffective due to ticker ambiguity. In contrast, LLMs achieve the highest accuracy with a score of 0.91. 

---
# Perception Without Engagement: Dissecting the Causal Discovery Deficit in LMMs 

**Authors**: Jiafeng Liang, Zhihao Zhu, Zihan Zhang, Baoqi Ren, Shixin Jiang, Runxuan Liu, Tao Ren, Ming Liu, See-Kiong Ng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.09422)  

**Abstract**: Although Large Multimodal Models (LMMs) have achieved strong performance on general video understanding, their susceptibility to textual prior shortcuts during causal discovery has been recognized as a critical deficit. The underlying mechanisms of this phenomenon remain incompletely understood, as existing benchmarks only measure response accuracy without revealing the sources and extent of the deficit. We introduce ProCauEval, a perturbation-based evaluation protocol that shifts from outcome assessment to mechanism diagnosis, probing causal discovery through five controlled configurations that systematically manipulate visual and textual modalities to decompose their respective contributions to model behavior and dissect the failure modes. Evaluating 17 mainstream LMMs, we find that models faithfully perceive video content yet systematically underexploit it during causal reasoning. We further observe that stronger post-training amplifies rather than mitigates textual prior reliance, and that higher baseline performance correlates with greater fragility under perturbation. To address these, we propose Anti-Distillation Policy Optimization (ADPO), a reinforcement learning framework built on negative teacher alignment, which augments GRPO by explicitly pushing the policy away from a prior-only counterfactual teacher induced by visual corruption. Specifically, ADPO maximizes the divergence between the policy distributions conditioned on the original and visually corrupted inputs, thereby forcing the model to ground its reasoning in visual evidence rather than textual shortcuts. Extensive experiments show that ADPO improves visual engagement without sacrificing fundamental comprehension, thus offering a preliminary step toward reliable causal discovery. 

---
# Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media 

**Authors**: Ahmed Mahrous, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2605.09414)  

**Abstract**: Emojis are widely used in online financial communication, but it is unclear whether they provide transferable sentiment signals across languages, platforms, and asset communities. This study examines the extent to which emoji usage, semantics, and sentiment polarity remain stable across financial communities, and how these layers influence zero-shot sentiment transfer. Using large corpora of Twitter and StockTwits posts in four languages, we measure cross-community divergence and evaluate sentiment models trained under emoji-only, text-only, and text+emoji inputs.
We find that emoji frequencies differ across communities, especially across languages, but their semantics and sentiment polarity are largely stable. Cross-asset transferability shows minimal degradation, while cross-language transfer remains the most challenging. Including emojis consistently reduces transfer gaps relative to text-only models. These results indicate that financial communication exhibits a partially shared ``emoji code,'' and that emojis provide compact, language-independent sentiment cues that improve model generalization across markets and platforms. 

---
# HOME-KGQA: A Benchmark Dataset for Multimodal Knowledge Graph Question Answering on Household Daily Activities 

**Authors**: Shusaku Egami, Aoi Ohta, Tomoki Tsujimura, Masaki Asada, Tatsuya Ishigaki, Ken Fukuda, Masahiro Hamasaki, Hiroya Takamura  

**Link**: [PDF](https://arxiv.org/pdf/2605.09348)  

**Abstract**: Large Language Models (LLMs) provide flexible natural language processing capabilities, while knowledge graphs (KGs) offer explicit and structured knowledge. Integrating these two in a complementary manner enables the development of reliable and verifiable AI systems. In particular, knowledge graph question answering (KGQA) has attracted attention as a means to reduce LLM hallucinations and to leverage knowledge beyond the training data. However, existing KGQA benchmark datasets are biased toward encyclopedic knowledge, limited to a single modality, and lack fine-grained spatiotemporal data, which limits their applicability to real-world scenarios targeted by Embodied AI. We introduce HOME-KGQA, a novel KGQA benchmark dataset built on a multimodal KG of daily household activities. HOME-KGQA consists of complex, multi-hop natural language questions paired with graph database query languages. Compared to existing benchmarks, it includes more challenging questions that involve multi-level spatiotemporal reasoning, multimodal grounding, and aggregate functions. Experimental results show that the LLM-based KGQA methods fail to achieve performance comparable to that on existing datasets when evaluated on HOME-KGQA. This highlights significant challenges that should be addressed for the real-world deployment of KGQA systems. Our dataset is available at this https URL 

---
# RuPLaR : Efficient Latent Compression of LLM Reasoning Chains with Rule-Based Priors From Multi-Step to One-Step 

**Authors**: Xiaocheng Luo, Kang Wang, Zaifu Zhan, Yuechi Zhou, Xiangyu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2605.09346)  

**Abstract**: The Chain-of-Thought (CoT) paradigm, while enhancing the interpretability of Large Language Models (LLMs), is constrained by the inefficiencies and expressive limits of natural language. Latent Chain-of-Thought (latent CoT) reasoning, which operates in a continuous latent space, offers a promising alternative but faces challenges from structural complexities in existing multi-step or multi-model paradigms, such as error propagation and coordination overhead. In this paper, we introduce One-Model One-Step, a novel compression framework for Latent Reasoning with Rule-Based Priors(RuPLaR) to address this challenge. Our method trains an LLM to autonomously generate latent reasoning tokens in a single training stage, guided by rule-based prior probability distributions, thereby eliminating cascaded processes and inter-model dependencies. To ensure reasoning quality, we design a joint training objective that enforces answer consistency via cross-entropy, aligns soft tokens with rule-based priors via KL divergence (the Soft Thinking constraint), and adds a problem-thought semantic alignment constraint in the representation space. Extensive experiments show that our compression framework not only improves accuracy by 11.1% over existing latent CoT methods but also achieves this with minimal token usage, underscoring its effectiveness and extensibility. Code: this https URL. 

---
# Test-Time Speculation 

**Authors**: Avinash Kumar, Sujay Sanghavi, Poulami Das  

**Link**: [PDF](https://arxiv.org/pdf/2605.09329)  

**Abstract**: Speculative decoding accelerates LLM inference by using a fast draft model to generate tokens and a more accurate target model to verify them. Its performance depends on the $\textit{acceptance length}$, or number of draft tokens accepted by the target. Our studies show that the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3 and PARD degrade with generation length, reaching values close to 1 (i.e. no speedup) within just a few thousand output tokens, making speculators ineffective for long-response tasks. Acceptance lengths decline because most speculators are trained offline on short sequences, but are forced to match the target model on much longer outputs at inference, well beyond their training distribution.
To address this issue, we propose $\textit{Test-Time Speculation (TTS)}$, an online distillation approach that continuously adapts the speculator at test-time. TTS leverages the key insight that the token verification step already invokes the target model for each draft token, providing the training signal needed to adapt the draft at no additional cost. Treating the draft as the student and the target as a teacher, TTS adjusts the draft over several speculation rounds, with each update improving the draft's accuracy as generation proceeds. Our results across multiple models from the Qwen-3, Qwen-3.5, and Llama3.1 families show that TTS improves acceptance lengths over state-of-the-art speculators by up to $72\%$ and $41\%$ on average, with the benefits scaling with increased generation lengths. 

---
# Mem-W: Latent Memory-Native GUI Agents 

**Authors**: Guibin Zhang, Yaohui Ling, Fanci Meng, Kun Wang, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.09317)  

**Abstract**: GUI agents are beginning to operate the web, mobile, and desktop as interactive worlds, where successful control depends on carrying forward visual, procedural, and task-level evidence beyond the fleeting present screen. Yet most agents still treat memory as an external, human-readable artifact: histories are summarized, categorized, retrieved, and reinserted as text or structured records before being encoded again by the policy. This creates a mismatch between the representational form in which experience is stored and the latent embedding sequence over which modern GUI policies actually act. We introduce Mem-W, a series of latent-memory-native GUI agents that treat memory as part of the agent's continuous context rather than as an auxiliary symbolic scaffold. Mem-W weaves both historical trajectories (as experiential memory) and in-session segments (as working memory) into compact memory tokens through a shared trajectory-to-latent compressor. These tokens are woven with the current GUI observation and local context into one continuous embedding sequence, allowing the agent to read successes, failures, and unfinished progress through the same machine-native interface. Mem-W is trained with self-distillation and outcome-aware supervision to preserve decision-relevant state while filtering memory toward evidence that truly supports task success. Across four web and mobile navigation benchmarks, Mem-W consistently improves diverse backbones and memory-enhanced baselines, with gains of up to $+30.0$, suggesting that latent-context-native memory can serve as a scalable foundation for long-horizon GUI agency. 

---
# LEAF-SQL: Level-wise Exploration with Adaptive Fine-graining for Text-to-SQL Skeleton Prediction 

**Authors**: Zhao Tan, Xiping Liu, Qing Shu, Qizhi Wan, Dexi Liu, Changxuan Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.09295)  

**Abstract**: Text-to-SQL translates natural language questions into executable SQL queries, enabling intuitive database access for non-experts. While large language models achieve strong performance on Text-to-SQL with prompting, they still struggle with complex queries that involve deeply nested logic or multiple clauses. A widely used approach employs SQL skeletons--intermediate representations of query logic--to streamline generation, but existing methods are limited by their reliance on a single structural hypothesis and lack of progressive reasoning. To overcome these limitations, we propose LEAF-SQL, a novel framework that reframes skeleton prediction as a coarse-to-fine tree search process. LEAF-SQL enables systematic exploration of diverse structural hypotheses with adaptive refinement. Several key techniques are employed in LEAF-SQL: (1) a three-level skeleton hierarchy to guide the search, (2) a Skeleton Formulation Agent to generate diverse candidates, and (3) a Skeleton Evaluation Agent to efficiently prune the search space. This integrated design yields skeleton candidates that are both structurally diverse and granularity-adaptive, providing a stronger foundation for the SQL generation. Extensive experiments show that LEAF-SQL consistently improves the performance of various LLM backbones. On the official hidden test set of the challenging BIRD benchmark, our method achieves 71.6 execution accuracy, which outperforms leading search-based and skeleton-based methods, affirming its effectiveness for complex queries. 

---
# BetaEdit: Null-Space Constrained Sequential Model Editing 

**Authors**: Bingqing Liu, Wei Liu, Yuhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09285)  

**Abstract**: Null-space-based methods have garnered considerable attention in model editing by constraining updates to the null space of the pre-existing knowledge representation, thereby preserving the model's original behavior. However, in practice these methods rely on an approximate null space--leading to knowledge leakage--and further suffer from severe performance degradation during sequential editing. Recent work shows that history-aware editing strategies can empirically mitigate this decline, yet the underlying reason remains unclear. In this paper, we first expose the knowledge leakage inherent in existing null-space approaches and then analyze why history-aware updates effectively preserve both editing performance and general capabilities during long-horizon editing. Building on these insights, we propose BetaEdit, a refined framework that effectively controls the knowledge leakage and integrates history-aware updates into the null-space paradigm. Extensive experiments on three large language models across two standard benchmarks show that BetaEdit consistently outperforms prior methods in the challenging regime of massive-scale sequential editing. Code is available at: this https URL. 

---
# DeltaRubric: Generative Multimodal Reward Modeling via Joint Planning and Verification 

**Authors**: Rui Liu, Dian Yu, Zhenwen Liang, Yucheng Shi, Tong Zheng, Runpeng Dai, Haitao Mi, Pratap Tokekar, Leoweiliang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09269)  

**Abstract**: Aligning Multimodal Large Language Models (MLLMs) requires reliable reward models, yet existing single-step evaluators can suffer from lazy judging, exploiting language priors over fine-grained visual verification. While rubric-based evaluation mitigates these biases in text-only settings, extending it to multimodal tasks is bottlenecked by the complexity of visual reasoning. The critical differences between responses often depend on instance-specific visual details. Robust evaluation requires dynamically synthesizing rubrics that isolate spatial and factual discrepancies. To address this, we introduce $\textbf{DeltaRubric}$, an approach that reformulates multimodal preference evaluation as a plan-and-execute process within a single MLLM. DeltaRubric operates in two steps: acting first as a $\textit{Disagreement Planner}$, the model generates a neutral, instance-specific verification checklist. Transitioning into a $\textit{Checklist Verifier}$, it executes these self-generated checks against the image and question to produce the final grounded judgment. We formulate DeltaRubric as a multi-role reinforcement learning problem, jointly optimizing planning and verification capabilities. Validated on Qwen3-VL 4B and 8B Instruct models, DeltaRubric achieves solid empirical gains. For instance, On VL-RewardBench, it improves base model overall accuracy by $\textbf{+22.6}$ (4B) and $\textbf{+18.8}$ (8B) points, largely outperforming standard no-rubric baselines. The results demonstrate that decomposing evaluation into structured, verifiable steps leads to more reliable and generalizable multimodal reward modeling. 

---
# Beyond Continuity: Challenges of Context Switching in Multi-Turn Dialogue with LLMs 

**Authors**: Aditya Sinha, Harald Steck, Vito Ostuni, Matteo Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09268)  

**Abstract**: Users interacting with Large Language Models (LLMs) in a multi-turn conversation routinely refine their requests or pivot to new topics. LLMs, however, often miss these topic shifts and carry over irrelevant context from previous turns, leading to inaccurate responses. In this paper, we stress-test the multi-turn understanding of LLMs and study the following two sub-tasks: (1) detecting whether the user pivots or refines in the current turn, and (2) shortlisting relevant context from previous turns. To this end, we construct synthetic benchmarks based on real-world datasets from varied domains, as to simulate context shifts of different levels of difficulty. We then evaluate the zero-shot performance of ten LLMs (open-weight, closed-source and reasoning), and demonstrate that only some reasoning and strongly instructed LLMs are accurate in detecting pivots; open-weight LLMs struggle with the task and frequently carry stale context even with explicit cues; and all models suffer from a position bias. Based on the results, we discuss key takeaways for improving long-term robustness in multi-turn capabilities for LLMs. 

---
# Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation 

**Authors**: Yuxuan Jiang, Runchao Li, Shubhashis Roy Dipta, Dawei Li, Zhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09253)  

**Abstract**: While recent work in Reinforcement Learning with Verifiable Rewards (RLVR) has shown that a small subset of critical tokens disproportionately drives reasoning gains, an analogous token-level understanding of On-Policy Distillation (OPD) remains largely unexplored. In this work, we investigate high-loss tokens, a token type that--as the most direct signal of student-teacher mismatch under OPD's per-token KL objective--should progressively diminish as training converges according to existing studies; however, our empirical analysis shows otherwise. Even after OPD training reaches apparent saturation, a substantial subset of tokens continues to exhibit persistently high loss; these tokens, which we term Rock Tokens, can account for up to 18\% of the tokens in generated outputs. Our investigation reveals two startling paradoxes. First, despite their high occurrence frequency providing a disproportionately large share of total gradient norms, Rock Tokens themselves remain stagnant throughout training, resisting teacher-driven corrections. Second, through causal intervention, we find that these tokens provide negligible functional contribution to the model's actual reasoning performance. These findings suggest that a vast amount of optimization bandwidth is spent on structural and discourse residuals that the student model cannot or need not internalize. By deconstructing these dynamics, we demonstrate that strategically bypassing these ``stumbling blocks'' can significantly streamline the alignment process, challenging the necessity of uniform token weighting and offering a more efficient paradigm for large-scale model distillation. 

---
# LLM Agents Already Know When to Call Tools -- Even Without Reasoning 

**Authors**: Chung-En Sun, Linbo Liu, Ge Yan, Zimo Wang, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2605.09252)  

**Abstract**: Tool-augmented LLM agents tend to call tools indiscriminately, even when the model can answer directly. Each unnecessary call wastes API fees and latency, yet no existing benchmark systematically studies when a tool call is actually needed. We propose When2Tool, a benchmark of 18 environments (15 single-hop, 3 multi-hop) spanning three categories of tool necessity -- computational scale, knowledge boundaries, and execution reliability -- each with controlled difficulty levels that create a clear decision boundary between tool-necessary and tool-unnecessary tasks. We evaluate two families of training-free baselines: Prompt-only (varying the prompt to discourage unnecessary calls) and Reason-then-Act (requiring the model to reason about tool necessity before acting). Both provide limited control: Prompt-only suppresses necessary calls alongside unnecessary ones, and Reason-then-Act still incurs a disproportionate accuracy cost on hard tasks. To understand why these baselines fail, we probe the models' hidden states and find that tool necessity is linearly decodable from the pre-generation representation with AUROC 0.89--0.96 across six models, substantially exceeding the model's own verbalized reasoning. This reveals that models already know when tools are needed, but fail to act on this knowledge during generation. Building on this finding, we propose Probe&Prefill, which uses a lightweight linear probe to read the hidden-state signal and prefills the model's response with a steering sentence. Across all models tested, Probe&Prefill reduces tool calls by 48% with only 1.7% accuracy loss, while the best baseline at comparable accuracy only reduces 6% of tool calls, or achieves a similar tool call reduction but incurs a 5$\times$ higher accuracy loss. Our code is available at this https URL 

---
# Repeated-Token Counting Reveals a Dissociation Between Representations and Outputs 

**Authors**: Sohan Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2605.09239)  

**Abstract**: Large language models fail at counting repeated tokens despite strong performance on broader reasoning benchmarks. These failures are commonly attributed to limitations in internal count tracking. We show this attribution is wrong. Linear probes on the residual stream decode the correct count with near-perfect accuracy at every post-embedding layer, across all model depths. This holds even at the exact layers where the wrong answer crystallizes while the model simultaneously outputs an incorrect count. Attention patterns show no evidence of collapse over repeated tokens and tokenization artifacts account for none of the failure. Instead, a format-triggered multi-layer perceptron (MLP) block overwrites the correctly-encoded count with a fixed wrong answer at roughly 88--93,% network depth. This prior fires for repeated word-tokens in space-separated list format and is absent for repeated digit-tokens. It is suppressed by comma-separated delimiters in larger models but persists in smaller ones. The finding holds across Llama-3.2 (1B and 3B) and Qwen2.5 (1.5B, 3B and 7B) at consistent relative depth. Counting failure is a failure of routing not of representation and the two require different interventions. 

---
# Matching Meaning at Scale: Evaluating Semantic Search for 18th-Century Intellectual History through the Case of Locke 

**Authors**: Yu Wu, Ananth Mahadevan, Filip Ginter, Michael Mathioudakis, Mikko Tolonen  

**Link**: [PDF](https://arxiv.org/pdf/2605.09236)  

**Abstract**: While digitized corpora have transformed the study of intellectual transmission, current methods rely heavily on lexical text reuse detection, capturing verbatim quotations but fundamentally missing paraphrases and complex implicit engagement. This paper evaluates semantic search in 18th-century intellectual history through the reception of John Locke's foundational work. Using expert annotation grounded in a semantic taxonomy, we examine whether an off-the-shelf semantic search pipeline can surface meaning-level correspondences overlooked by lexical methods. Our results demonstrate that semantic search retrieves substantially more implicit receptions than lexical baselines. However, linguistic diagnostics also reveal a "lexical gatekeeping" effect, where retrieval remains partially constrained by surface vocabulary overlap. These findings highlight both the potential and the limitations of semantic retrieval for analyzing the circulation of ideas in large historical corpora. The data is available at this https URL. 

---
# Two Ways to De-Bias an LLM-as-a-Judge: A Continuous-Score Comparison of Hierarchical Bayesian Calibration and Neural-ODE Score Transport 

**Authors**: Andrea Morandi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09227)  

**Abstract**: [Abridged] Using a Large Language Model (LLM) as an automatic rater (LLM-as-a-judge) is cheap but potentially biased: some judges run lenient, others strict, the middle of the scale gets compressed, and verbose answers may be over-rewarded. A common remedy is post-hoc calibration: leave the cheap judge in place and, on a modest set of paired anchors, fit a transformation from raw judge scores to an estimate of the human rating. We compare two correctors that take opposing views on how this mapping should be modeled: a parametric, small-anchor hierarchical Bayesian linear correction with per-score uncertainty, and a non-parametric Neural-ODE (FFJORD) score-transport flow. Both are run head-to-head on UltraFeedback fine-grained_score (1700 paired examples, 200 held out), with calibration split into three operational sub-questions: population-mean recovery, per-item accuracy, and distributional-shape match.
The headline result is that the choice between methods is primarily a data-budget question. Both correctors close the raw $+0.71$-point mean offset to within $\pm 0.08$ of the GPT-4 reference, at 100 and at 1500 anchors. Past that, the methods swap roles. With 100 anchors, the linear corrector reconstructs the human-score distribution roughly twice as well by KL divergence (0.031 vs. 0.058) and ties the flow on MAE. With 1500 anchors the flow wins on every metric (MAE 0.320 vs. 0.359, Pearson 0.922 vs. 0.896, KL 0.026 vs. 0.037). The Bayesian linear corrector saturates well below 1500 anchors: residual $\tanh$-shaped non-linearity is, by construction, structure a linear correction cannot fit. The flow keeps improving as labels grow. We translate these findings into an explicit decision rule for production deployments. 

---
# WorldSpeech: A Multilingual Speech Corpus from Around the World 

**Authors**: Antonis Asonitis, Luca A. Lanzendörfer, Frédéric Berdoz, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2605.09167)  

**Abstract**: Automatic speech recognition (ASR) performs well for high-resource languages with abundant paired audio-transcript data, but its accuracy degrades sharply for most languages due to limited publicly available aligned data. To this end, we introduce WorldSpeech, a 24 kHz multilingual speech corpus comprising 65k hours of aligned audio-transcript data across 76 languages, collected from diverse public sources including parliamentary proceedings, international broadcasts, and public-domain audiobooks. For 37 languages, WorldSpeech provides more than 200 hours of aligned speech, with 28 exceeding 500 hours and 24 surpassing 1k hours. Fine-tuning existing ASR models on WorldSpeech results in an average relative Word-Error-Rate reduction of 63.5% across 11 typologically diverse languages. 

---
# Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan 

**Authors**: Ahan Chatterjee, Matthias Schöffel, Matthias Aßenmacher, Esteban Garces Arias  

**Link**: [PDF](https://arxiv.org/pdf/2605.09156)  

**Abstract**: The diachronic evolution from Latin to the Romance languages involved a restructuring of the grammatical gender system from a tripartite configuration (masculine, feminine, neuter) to a bipartite one (masculine, feminine). In this work, we introduce an interpretable deep learning framework to investigate this phenomenon at both lexical and contextual levels. First, we show that conventional tokenization strategies are insufficiently robust for this low-resource historical setting, and that our proposed tokenizer improves performance over these baselines. At the lexical level, we evaluate the contribution of morphological features to gender prediction. At the contextual level, we quantify the contributions of different part-of-speech categories to grammatical gender prediction. Together, these analyses characterize the distribution of gender information between the lemma and its sentential context. We make our codebase, datasets, and results publicly available. 

---
# Meow-Omni 1: A Multimodal Large Language Model for Feline Ethology 

**Authors**: Jucheng Hu, Zhangquan Chen, Yulin Chen, Chengjie Hong, Liang Zhou, Tairan Wang, Sifei Li, Giulio Zhu, Feng Zhou, Yiheng Zeng, Suorong Yang, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.09152)  

**Abstract**: Deciphering animal intent is a fundamental challenge in computational ethology, largely because of semantic aliasing, the phenomenon where identical external signals (e.g., a cat's purr) correspond to radically different internal states depending on physiological context. Existing Multimodal Large Language Models (MLLMs) are blind to high-frequency biological time-series data, restricting them to superficial behavioural pattern matching rather than genuine latent-state reasoning. To bridge this gap, we introduce Meow-Omni 1, the first open-source, quad-modal MLLM purpose-built for computational ethology. It natively fuses video, audio, and physiological time-series streams with textual reasoning. Through targeted architectural adaptation, we integrate specialized scientific encoders into a unified backbone and formalize intent inference via physiologically grounded cross-modal alignment. Evaluated on MeowBench, a novel, expert-verified quad-modal benchmark, Meow-Omni 1 achieves state-of-the-art intent-recognition accuracy (71.16%), substantially outperforming leading vision-language and omni-modal baselines. We release the complete open-source pipeline including model weights, training framework, and the Meow-10K dataset, to establish a scalable paradigm for inter-species intent understanding and to advance foundation models toward real-world veterinary diagnostics and wildlife conservation. 

---
# From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages 

**Authors**: Matthias Schöffel, Esteban Garces Arias  

**Link**: [PDF](https://arxiv.org/pdf/2605.09147)  

**Abstract**: Part-of-speech (POS) tagging for Medieval Romance languages remains challenging due to orthographic variation, morphological complexity, and limited annotated resources. This paper presents a systematic empirical evaluation of large language models (LLMs) for POS tagging across three medieval varieties: Medieval Occitan, Medieval Catalan, and Medieval French. We compare traditional rule-based and statistical taggers with modern open-source LLMs under zero-shot prompting, few-shot prompting, monolingual fine-tuning, and cross-lingual transfer learning settings.
Experiments on historically grounded datasets show that LLM-based approaches consistently outperform traditional taggers, with fine-tuning and multilingual training yielding the largest improvements. In particular, cross-lingual transfer learning substantially benefits under-resourced varieties, while targeted bilingual training can outperform broader multilingual configurations for specific target languages. The results highlight the importance of linguistic proximity and dataset characteristics when designing transfer strategies for historical NLP.
These findings provide empirical insights into the applicability of modern neural methods to medieval text processing and provide practical guidance for deploying LLM-based POS tagging pipelines in digital humanities research. All code, models, and processed datasets are released for reproducibility. 

---
# Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain 

**Authors**: Xiaoyu Hu, Jinman Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.09106)  

**Abstract**: Large language models (LLMs) are increasingly deployed in financial contexts, raising critical concerns about reliability, alignment, and susceptibility to adversarial manipulation. While prior finance-related benchmarks assess LLMs' capabilities in stock trading, they are often restricted to small sample and fail to demonstrate LLM susceptibility to context with potential human bias. We introduce Fin-Bias (financial herding under long and uncertain financial context), a benchmark for evaluating LLM investment decision-making when faced with uncertainty and possible human-biased opinions. Fin-Bias includes 8868 long firm-specific analyst reports, including firm aspects summarized and analyzed by sophisticated analysts with investment ratings (Bullish/Neutral/Bearish) spanning from various industries. We present large language models with firm analyst reports with/without analyst investment ratings and even with 'fake' rating, to get investment ratings generated by LLMs. Our results reveal that LLMs tend to herd the explicit bias in context. We also develop a method to detect potential human opinions, which can encourage LLMs to think independently, some models even exceed human performance in predicting future stock return. 

---
# GRC: Unifying Reasoning-Driven Generation, Retrieval and Compression 

**Authors**: Zhongtao Miao, Qiyu Wu, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2605.09100)  

**Abstract**: Text embedding and generative tasks are usually trained separately based on large language models (LLMs) nowadays. This causes a large amount of training cost and deployment effort. Context compression is also a challenging and pressing task, which is vital to reasoning-driven generation, and agentic tasks requiring long context and continual learning. In this paper, we explore how to unify reasoning-driven generation, reasoning-enhanced text representation and context compression tasks in one forward pass for LLMs. Through meta latent tokens and a unified generative, representative and compressive tuning approach, we propose a training framework named GRC that bridges the three tasks. The trained models can accomplish three objectives in a single forward pass while maintaining modular, LEGO-style flexibility during inference. This design greatly reduces the deployment effort for retrieval-augmented generation (RAG) and achieves efficient inference and three times data utilization during training. Furthermore, this framework design enables a new paradigm for text embedding: self-reason-latent embeds, and a new generation paradigm, latent memory-augmented generation, where compressed and internalized KV cache with O(1) length is used as the updatable memory. We also propose hybrid paged attention to speed up the inference of our models. Extensive experiments on reasoning-intensive retrieval benchmarks, generative tasks, document compression, latency evaluation, and RAG settings demonstrate the effectiveness of our method and may shed light on the truly unified model that can handle reasoning-driven generation, embedding and compression tasks seamlessly. 

---
# Dynamic Meta-Metrics: Source-Sentence Conditioned Weighting for MT Evaluation 

**Authors**: Luke Zhang, Justin Vasselli, Aditya Khan, York Hay Ng, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.09098)  

**Abstract**: We propose Dynamic Meta-Metrics (DMM), a framework for machine translation evaluation that learns source-sentence conditioned combinations of existing metrics. Rather than relying on a single static ensemble or language-specific weighting, DMM adapts the metric combination based on properties of the source segment. We study hard conditioning, which fits an interpretable combiner per cluster, and an exploratory soft-conditioned extension whose weights vary continuously with source-cluster responsibilities. We evaluate DMM on the WMT Metrics Shared Task data across multiple language pairs using pairwise agreement measures at the system and segment levels. Across settings, MLP-based combinations outperform linear and Gaussian process-based ensembles, and introducing soft conditioning yields gains over linear models. 

---
# Character-Level Transformer for Tajik-Persian Transliteration with a Parallel Lexical Corpus 

**Authors**: Mullosharaf K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2605.09092)  

**Abstract**: This study addresses automatic transliteration from Tajik (Cyrillic script) to Persian (Perso-Arabic script). We present a curated, lexicographically verified parallel corpus of 52,152 Tajik--Persian words and short phrases, compiled from printed dictionaries, encyclopedic sources, and manually verified online resources. To the best of our knowledge, this is one of the largest publicly available word-level corpora for Tajik--Persian transliteration.
Using this corpus, we train a character-level sequence-to-sequence Transformer model and evaluate it using Character Error Rate (CER) and exact-match accuracy. The Transformer achieves a CER of 0.3216 and an exact-match accuracy of 0.3133, outperforming both dictionary-based rule-based and recurrent neural baselines. With beam search (k=3), performance further improves to CER 0.3182 and accuracy 0.3215. We describe the data collection and preprocessing pipeline, model architecture, and experimental protocol, and report a part-of-speech analysis showing performance differences across lexical categories.
All preprocessing scripts, deterministic splits into training, validation, and test sets, and training configurations are released to support reproducibility and further research on Tajik and related Persian dialects. The corpus supports research in character-level transliteration, cross-script NLP, and lexicographic applications. 

---
# Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs 

**Authors**: Guijin Son, Seungone Kim, Catherine Arnett, Hyunwoo Ko, Hyein Lee, Hyeonah Kang, Jiang Longxi, Jin Yun, JungYup Lee, Kyungmin Lee, Sam Yoosuk Kim, Sang Park, Seunghyeok Hong, SeungJae Lee, Seungyeop Yi, Shinae Shin, SunHye Bok, Sunyoung Shin, Yonghoon Ji, Youngtaek Kim, Hanearl Jung, Akari Asai, Graham Neubig, Sean Welleck, Youngjae Yu, Akshelin R, Alexander B. Ivanov, Boboev Muhammadjon, Chaeyoung Han, Christian Stump, Dmitrii Karp, Dohyun Kwon, DoYong Kwon, Duk-Soon Oh, Giovanni Resta, Greta Panova, Huiyun Noh, Hyungryul Baik, Hyungsun Bae, Inomov Mashrafdzhon, Jeewon Kim, Ji Eun Lee, Jiaqi Liu, Jieui Kang, Jimin Kim, Jon-Lark Kim, Junseo Yoon, Junwoo Jo, Kibeom Kim, Kiwoon Kwon, Mario Kummer, Max Mercer, Minjun Kim, Nahyun Lee, Ng Ze-An, Rafał Marcin Łochowski, Raphaël Lachièze-Rey, Ruichen Zhang, Sejin Park, Seonguk Seo, Shin Jaehoon, Sunatullo, Taewoong Eom, Yeachan Park, Yongseok Jang, Youchan Oh, Zhaoyang Wang, Zoltán Kovács  

**Link**: [PDF](https://arxiv.org/pdf/2605.09063)  

**Abstract**: Following the recent achievement of gold-medal performance on the IMO by frontier LLMs, the community is searching for the next meaningful and challenging target for measuring LLM reasoning. Whereas olympiad-style problems measure step-by-step reasoning alone, research-level problems use such reasoning to advance the frontier of mathematical knowledge itself, emerging as a compelling alternative. Yet research-level math benchmarks remain scarce because such problems are difficult to source (e.g., Riemann Bench and FrontierMath-Tier 4 contain 25 and 50 problems, respectively). To support reliable evaluation of next-generation frontier models, we introduce Soohak, a 439-problem benchmark newly authored from scratch by 64 mathematicians. Soohak comprises two subsets. On the Challenge subset, frontier models including Gemini-3-Pro, GPT-5, and Claude-Opus-4.5 reach 30.4%, 26.4%, and 10.4% respectively, leaving substantial headroom, while leading open-weight models such as Qwen3-235B, GPT-OSS-120B, and Kimi-2.5 remain below 15%. Notably, beyond standard problem solving, Soohak introduces a refusal subset that probes a capability intrinsic to research mathematics: recognizing ill-posed problems and pausing rather than producing confident but unjustified answers. On this subset, no model exceeds 50%, identifying refusal as a new optimization target that current models do not directly address. To prevent contamination, the dataset will be publicly released in late 2026, with model evaluations available upon request in the interim. 

---
# Language-Conditioned Visual Grounding with CLIP Multilingual 

**Authors**: J. de Curtò, Mauro Liz, I. de Zarzà  

**Link**: [PDF](https://arxiv.org/pdf/2605.09060)  

**Abstract**: Multilingual vision-language models exhibit systematic performance gaps across languages, but the mechanism remains ambiguous: cross-language divergence could arise from the visual encoder, the text branch, or their interaction. We resolve this ambiguity through a dense multilingual CLIP probe in which the visual encoder is held identical across thirteen typologically diverse languages and only the XLM-RoBERTa text branch varies. We evaluate two CLIP architectures spanning a 7x visual-encoder scale gap (XLM-R base + ViT-B/32, ~87M visual parameters; XLM-R large + ViT-H/14, ~632M) on 11 concepts and 210 images, and quantify cross-language agreement via cluster-mask IoU, top-percentile IoU, and Spearman rank correlation against an English reference (n=2,310 paired observations per language). Three findings emerge. First, low-resource languages (Arabic, Basque, Luxembourgish) incur a structural penalty at both backbone scales (Wilcoxon HR>LR p<10^-300; cluster-mask IoU gap +0.114 at base, +0.143 at large), isolating the deficit to the text branch. Second, scaling the encoder 7x widens the gap for structural failure cases (Basque {\Delta}=-0.056, Luxembourgish {\Delta}=-0.076) while improving Arabic ({\Delta}=+0.033), separating corpus-coverage from tokeniser-fertility failures. Third, peak similarity is preserved across languages (mean ratio 0.94 at large scale) while cluster-mask IoU drops sharply, identifying spatial misalignment, not signal collapse, as the dominant failure mode. At 3.4-3.9 Wh per 1,000 queries, dense-CLIP grounding is competitive with high-throughput inference budgets, positioning it as a practical substrate for energy-aware multilingual deployment. 

---
# Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break 

**Authors**: Napassorn Litchiowong  

**Link**: [PDF](https://arxiv.org/pdf/2605.09043)  

**Abstract**: One partner says "Fine" meaning <i>resolution</i>; the other hears <i>surrender</i>. The word is shared; the affective uptake is not. We formalize this as <b>affective meaning divergence (AMD)</b>, the total-variation distance between interlocutors' anchor-conditioned affect distributions. Building on speech-act theory, common-ground accumulation, and entropy-regularized game theory, we derive a logit best-response map whose dynamics undergo a saddle-node bifurcation: when $\beta\alpha > 4$, a monotone increase in AMD-driven load produces an abrupt, hysteretic collapse of repair coordination. On Conversations Gone Awry (CGA-Wiki; $N=652$), derailing conversations exhibit critical-slowing-down (CSD) signatures across multiple levels: lexical divergence variance ($p<0.001$, $d=0.36$), AMD variance ($p=0.001$, $d=0.26$), and dialog-act repair variance ($p=0.016$, $d=0.20$), all significant after correction and stronger than toxicity and sentiment baselines. AMD provides a distinct temporal signature, with retrospectively measured variance peaking at the bifurcation point while toxicity variance peaks earlier, and is the only indicator grounded in the theoretical framework. Boundary-condition analysis on CGA-CMV ($N=1{,}169$) yields mixed but directionally consistent evidence. 

---
# Evaluating Pragmatic Reasoning in Large Language Models: Evidence from Scalar Diversity 

**Authors**: Ye-eun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.09042)  

**Abstract**: Evaluating pragmatic reasoning in large language models (LLMs) remains challenging because model behavior can vary depending on evaluation methods. Previous studies suggest that prompt-based judgments may diverge from models' internal probability distributions, raising questions about whether observed performance reflects underlying competence or task-induced behavior. This study examines this issue using scalar diversity as a graded diagnostic for pragmatic inference. Following Hu & Levy (2023), this study compares direct probability measurement and metalinguistic prompting across multiple models and experimental settings. The results show that neither evaluation method consistently outperforms the other and that pragmatic behavior varies substantially across model families, prompting strategies, and task structures. Moreover, scalar diversity gradients emerge only in specific model-condition combinations, suggesting that pragmatic reasoning in LLMs reflects an interaction between internal probabilistic representations and task-induced prompting behavior rather than a stable competence captured by a single evaluation paradigm. These findings highlight the central role of evaluation design in interpreting pragmatic abilities in LLMs. 

---
# BiAxisAudit: A Novel Framework to Evaluate LLM Bias Across Prompt Sensitivity and Response-Layer Divergence 

**Authors**: Jialing Gan, Junhao Dong, Songze Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09041)  

**Abstract**: Bias audits of large language models now operate within governance frameworks such as the EU AI Act, making benchmark reliability a security concern in its own right. Many current benchmarks, however, collapse bias into a single scalar from one prompt format and one surface label. This design misses two failure modes that can be exploited without changing model weights. Across prompts, meaning-preserving format changes shift bias endorsement by more than $0.7$ on a fixed statement pool. Within a response, the discrete Selection and free-text Elaboration can take opposing stances, so an apparently clean aggregate may hide substantial internal inconsistency (a ``cancellation trap''). Selection-only and elaboration-only rankings are therefore nearly uncorrelated across eight LLMs (Spearman $\rho = 0.238$, $p = 0.570$): LLaMA3-70B ranks in the middle under selection-only scoring but highest under elaboration-only scoring on the same responses. We introduce \textsc{BiAxisAudit}, a protocol that reports each bias score together with a reliability estimate on two orthogonal axes. The across-prompt axis evaluates each statement under a factorial grid of task format, perspective, role, and sentiment, treating bias as a distribution rather than a point estimate. The within-response axis uses Split Coding to recover Selection and Elaboration as separate signals, measured by the Inconsistency Rate and Divergence Net Imbalance. Across eight LLMs with $80{,}200$ coded responses each, task format alone explains as much variance as model choice; $63.6\%$ of pooled bias signals (up to $85.2\%$ per model) appear in only one coding layer, and prompt-dimension interactions exceed main effects. The instrument also separates real bias reductions from apparent reductions caused by cross-layer redistribution: some prompt configurations reduce both BER and IR, whereas others suppress only selection-layer bias. 

---
# A Quantum Inspired Variational Kernel and Explainable AI Framework for Cross Region Solar and Wind Energy Forecasting 

**Authors**: Pavan Manjunath, Thomas Prufer  

**Link**: [PDF](https://arxiv.org/pdf/2605.09032)  

**Abstract**: Reliable short horizon forecasting of solar and wind generation is a structural prerequisite of any modern power system yet most published forecasters are tuned and evaluated on a single climatic regime and most algorithmic novelty has been concentrated either on classical recurrent networks or on monolithic foundation models that combine forecasting and explanation We develop a four stage hybrid framework that separates these concerns The first stage acquires hourly generation irradiance and surface weather records through public application programming interfaces The second stage trains three classical baselines autoregressive integrated moving average gradient boosted regression trees and a two layer long short term memory network and produces a strong point forecast together with a residual error series The third stage corrects the residual through a quantum inspired variational kernel built on a six qubit hardware efficient ansatz with three repeated entangling layers The fourth stage uses generative artificial intelligence strictly as an explainability layer that reads the measured benchmark numbers and produces a structured natural language interpretation Across three regions drawn from open public archives Iberian solar North Sea wind and a mixed Texas trace the proposed configuration stays within one percentage point of the strongest classical baseline on the in domain forecasting task and the quantum inspired kernel separates calm and stormy weather regimes with a Fisher discriminant ratio approximately fifteen fold higher than a tuned radial basis kernel 

---
# GAMBIT: A Three-Mode Benchmark for Adversarial Robustness in Multi-Agent LLM Collectives 

**Authors**: Alexandre Le Mercier, Chris Develder, Thomas Demeester  

**Link**: [PDF](https://arxiv.org/pdf/2605.09027)  

**Abstract**: In multi-agent systems (MAS), a single deceptive agent can nullify all gains of an agentic AI collective and evade deployed defenses. However, existing adversarial studies on MAS target only shallow tasks and do not consider adaptive adversaries, which evolve their strategies to evade the very detectors trained to catch them. To address that gap, we introduce GAMBIT, a benchmark with three evaluation modes and two independent scores for evaluating imposter detectors: the first two modes measure zero-shot detection under increasing distribution shift, and a third recalibration mode measures how quickly a detector adapts to novel attacks from just 20 labeled examples. The benchmark comes with a dataset of 27,804 labeled instances spanning 240 co-evolved imposter strategies. Our contributions are threefold: (1) Using chess as a substrate deep reasoning problem and Gemini 3.1 Pro for agents, we release GAMBIT and its dataset to evaluate imposter detectors under realistic constraints against a stealthy adaptive imposter; (2) We introduce an adaptive imposter agent based on an efficient evolutionary framework, generalizable beyond chess, that collapses collective task performance while remaining essentially undetectable (50.5% F1-score with a Gemini-based detector); (3) We show that zero-shot evaluation can be highly misleading for adaptive adversaries: two detectors with near-identical zero-shot scores differ by 8x on few-shot adaptation, while the meta-learned variant converges 20x faster, a gap only visible in the recalibration mode. Altogether, GAMBIT provides the first multi-agent benchmark where adversarial attacks and defenses co-evolve, with an imposter framework generalizable beyond our use case, and promising techniques for fast recalibration in a rapidly evolving adversarial system. Code and data: this https URL. 

---
# LLiMba: Sardinian on a Single GPU -- Adapting a 3B Language Model to a Vanishing Romance Language 

**Authors**: Luca Ballore  

**Link**: [PDF](https://arxiv.org/pdf/2605.09015)  

**Abstract**: Sardinian, a Romance language with roughly one million speakers, has minimal presence in modern NLP. Commercial services do not support it, and current language models do not produce it reliably. We present LLiMba, a 3B parameter Sardinian-ready model adapted from Qwen2.5-3B-Instruct through continued pretraining (CPT) and supervised fine-tuning (SFT) on a single 24 GB consumer GPU. The corpus contains 11.5 million tokens of Sardinian spanning LSC, Logudorese, and Campidanese, augmented with 2.4 million tokens of related Romance text as replay against register blurring. After CPT the model reaches a perplexity of 6.76 on held out Sardinian and outperforms the base across all six FLORES-200 directions. We compare five SFT configurations under matched conditions: full fine-tuning, LoRA r64, rsLoRA r128, rsLoRA r256, and DoRA r256. rsLoRA r256 wins on every direction into Sardinian, reaching 28.5 BLEU from English against 17.3 after CPT and 21.0 with full fine-tuning. The rank ablation places r128 between LoRA r64 and rsLoRA r256 on BLEU but reveals failure modes invisible to the metric, including leakage across scripts no other variant produces. LoRA r64 retains less factual content from SFT than configurations at higher rank and produces more confident fabrications, though all methods fabricate on content absent from training. DoRA r256 yields the smallest gap between training and evaluation but the worst factual accuracy. The findings indicate that adapter capacity matters more than the choice among LoRA variants for adapting a Romance pretrained base to a low resource Romance target, that stronger regularization is not uniformly beneficial, and that translation metrics smoothly order configurations whose qualitative behavior differs categorically. Perplexity comparisons across scripts must account for byte fallback tokenization, which deflates the metric for scripts other than Latin. 

---
# Dolphin-CN-Dialect: Where Chinese Dialects Matter 

**Authors**: Yangyang Meng, Huihang Zhong, Guodong Lin, Guanbo Wang, Hu Du, Zhiming Shao, Yukai Huang, Ke Li, Wei-Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08961)  

**Abstract**: We present Dolphin-CN-Dialect, a streaming-capable ASR model with a focus on Chinese and dialect-rich scenarios. Compared to the previous version, Dolphin-CN-Dialect introduces substantial improvements in data processing, tokenization, training stability, and data sampling strategies. To address the challenges of highly imbalanced dialect data, we propose a temperature-based sampling strategy that effectively balances standard Mandarin and low-resource dialects, leading to significant gains in dialect recognition performance. In addition, we redesign the tokenizer to better align with linguistic characteristics, adopting character-level modeling for Chinese and subword modeling for English, while introducing extensible dialect tokens. Experimental results show that Dolphin-CN-Dialect achieves improvement in dialect recognition accuracy and CER reduction compared to Dolphin. Furthermore, Dolphin-CN-Dialect reaches competitive performance with recent SOTA open-source ASR models, while maintaining a significantly smaller model size. Dolphin-CN-Dialect supports both streaming and non-streaming inference, enabling a practical balance between latency and accuracy. It also provides flexible customization through hotword support and efficient deployment optimized for specialized hardware. These improvements make Dolphin-CN-Dialect a strong and practical solution for real-world multi-dialect ASR applications. 

---
# Improving Lexical Difficulty Prediction with Context-Aligned Contrastive Learning and Ridge Ensembling 

**Authors**: Wicaksono Leksono Muhamad, Joanito Agili Lopo, Tsamarah Rana Nugraha, Ahmad Cahyono Adi, Muhammad Oriza Nurfajri  

**Link**: [PDF](https://arxiv.org/pdf/2605.08950)  

**Abstract**: Lexical difficulty prediction is a fundamental problem in language learning and readability assessment, requiring models to estimate word difficulty across different first-language (L1) backgrounds. However, existing approaches rely on regression-only training with scalar supervision, which does not explicitly structure the representation space, limiting their ability to capture cross-lingual alignment and ordinal difficulty. To mitigate these issues, we propose Context-Aligned Contrastive Regression, which integrates Ridge regression ensemble with two complementary objectives, i.e., Cross-View Context and Ordinal Soft Contrastive Learning. Experiments on three L1 datasets show that (i) contrastive objectives improve cross-lingual representation alignment while preserving language-specific nuances, (ii) the learned representations capture the ordinal structure of lexical difficulty, and (iii) the ensemble effectively mitigates systematic biases of individual models, leading to more stable performance across difficulty levels. 

---
# Decomposing and Steering Functional Metacognition in Large Language Models 

**Authors**: Yanshi Li, Xueru Bai, Shuman Liu, Haibo Zhang, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.08942)  

**Abstract**: Large language models (LLMs) increasingly exhibit behaviors suggesting awareness of their evaluation context, often adapting their reasoning strategies in benchmark settings. Prior work has shown that such evaluation awareness can distort performance measurements; however, it remains unclear whether this phenomenon reflects a single behavioral artifact or a deeper internal structure within the model.
We propose that LLMs maintain a decomposable space of functional metacognitive states: internal variables encoding factors such as evaluation awareness, self-assessed capability, perceived risk, computational effort allocation, audience expertise adaptation, and intentionality. Through residual stream analysis across multiple reasoning models, we demonstrate that these states are linearly decodable from internal activations and exhibit distinct layer-wise profiles. Moreover, by steering model activations along probe-derived directions, we show that each functional metacognitive state causally modulates reasoning behavior in dissociable ways, affecting verbosity, accuracy, and safety-related responses across tasks.
Our findings suggest that benchmark performance reflects not only task competence but also the activation of specific functional metacognitive states. We argue that understandi ng and controlling these internal states is essential for reliable evaluation and deployment of reasoning models, and we provide a mechanistic framework for studying functional m etacognition in artificial systems. Our code and data are publicly available at this https URL. 

---
# LLM-Agnostic Semantic Representation Attack 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Tairan Huang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2605.08898)  

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting adversarial prompts. Predominant token-level optimization methods primarily rely on optimizing for exact affirmative templates (e.g., ``\textit{Sure, here is...}''). However, these paradigms frequently encounter bottlenecks such as suboptimal convergence, compromised prompt naturalness, and poor cross-model generalization. To address these limitations, we propose Semantic Representation Attack (SRA), a novel LLM-agnostic paradigm that fundamentally reconceptualizes adversarial objectives from exact textual targeting to malicious semantic representations. Theoretically, we establish the semantic Coherence-Convergence Relationship and derive a Cross-Model Semantic Generalization bound, proving that maintaining semantic coherence guarantees both white-box semantic convergence and black-box transferability. Technically, we operationalize this framework via the Semantic Representation Heuristic Search (SRHS) algorithm, which preserves interpretability and structural coherence of the adversarial prompts during incremental discrete token chunk expansion. Extensive evaluations demonstrate that our framework achieves a 99.71% average attack success rate across 26 open-source LLMs, with strong transferability and stealth. 

---
# FragileFlow: Spectral Control of Correct-but-Fragile Predictions for Foundation Model Robustness 

**Authors**: Zhuoyun Li, Boxuan Wang, Jinwei Hu, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.08896)  

**Abstract**: Robust adaptation of LLMs and VLMs is often evaluated by average accuracy or average consistency under perturbations. However, these averages can hide a structured failure mode: a prediction may remain correct while probability mass already flows from particular true classes toward systematic wrong competitors near the decision boundary. In this paper, we formalize this phenomenon as margin-aware error flow and introduce FragileFlow, a plug-in regularizer that uses a calibrated margin buffer to identify correct-but-fragile predictions and organize their off-class probability mass into a class-wise vulnerable-risk matrix. Theoretically, we provide the first PAC-Bayes upper bound for this margin-aware error-flow object, showing how empirical spectral control yields a conservative route to deterministic worst-class robustness under a stability condition. Experiments on multiple-choice LLM benchmarks and few-shot CLIP adaptation show that FragileFlow consistently improves the proposed theory-facing risk measures over matched baselines, yields perturbed worst-class accuracy gains in most settings, and preserves clean accuracy across comparisons. 

---
# Fitting Is Not Enough: Smoothness in Extremely Quantized LLMs 

**Authors**: Yuzhuang Xu, Xu Han, Yuxuan Li, Pengzhan Li, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2605.08894)  

**Abstract**: Large language models (LLMs) achieve strong performance but incur high deployment costs, motivating extremely low-bit but lossy quantization. Existing quantization algorithms mainly focus on improving the numerical accuracy of forward computation to eliminate performance degradation. In this paper, we show that extremely quantized LLMs suffer from systematic smoothness degradation beyond numerical precision loss. Through a smoothness proxy, we observe that such degradation becomes increasingly severe as the quantization bit-width decreases. Furthermore, based on sequence neighborhood modeling, we find that quantized models exhibit a rapid reduction of effective token candidates within the prediction neighborhood, which directly leads to a sparser decoding tree and degraded generation quality. To validate it, we introduce a simple smoothness-preserving principle in both post-training quantization and quantization-aware training, and demonstrate that preserving smoothness brings additional gains beyond numerical accuracy. The core goal of this paper is to highlight smoothness preservation as an important design consideration for future extreme quantization methods. Code is available at this https URL. 

---
# DocScope: Benchmarking Verifiable Reasoning for Trustworthy Long-Document Understanding 

**Authors**: Xiang Feng, Jiawei Zhou, Zhangfeng Huang, Kewei Wang, Shanshan Ye, Jinxin Hu, Zulong Chen, Yong Luo, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08888)  

**Abstract**: Evaluating whether Multimodal Large Language Models can produce trustworthy, verifiable reasoning over long, visually rich documents requires evaluation beyond end-to-end answer accuracy. We introduce DocScope, a benchmark that formulates long-document QA as a structured reasoning trajectory prediction problem: given a complete PDF document and a question, the model outputs evidence pages, supporting evidence regions, relevant factual statements, and a final answer. We design a four-stage evaluation protocol -- Page Localization, Region Grounding, Fact Extraction, and Answer Verification -- that audits each level of the trajectory independently through inter-stage decoupling, with all judges selected and calibrated via human alignment studies. DocScope comprises 1,124 questions derived from 273 documents, with all hierarchical evidence annotations completed by human annotators. We benchmark 6 proprietary models, 12 open-weight models, and several domain-specific systems. Our experiments reveal that answer accuracy cannot substitute for trajectory-level evaluation: even among correct answers, the highest observed rate of complete evidence chains is only 29\%. Across all models, region grounding remains the weakest trajectory stage. Furthermore, the primary difficulty stems from aggregating evidence dispersed across long distances and multiple document clusters, while an oracle study identifies faithful perception and fact extraction as the dominant capability bottleneck. Cross-architecture comparisons further suggest that activated parameter count matters more than total scale. The benchmark and code will be publicly released at this https URL. 

---
# Max-pooling Network Revisited: Analyzing the Role of Semantic Probability in Multiple Instance Learning for Hallucination Detection 

**Authors**: Shota Fujikawa, Issei Sato  

**Link**: [PDF](https://arxiv.org/pdf/2605.08863)  

**Abstract**: Hallucination detection has become increasingly important for improving the reliability of large language models (LLMs). Recently, hybrid approaches such as HaMI, which combine semantic consistency with internal model states via Multiple Instance Learning (MIL), have achieved state-of-the-art performance. However, these methods incur substantial computational overhead due to repeated sampling and costly semantic similarity computations. In this work, we first provide a theoretical analysis of HaMI in terms of decision margins, revealing that scaling internal states with semantic consistency leads to an enlarged decision margin. Motivated by this insight, we revisit classical sentence classification models from a margin enlargement perspective, aggregating token-level features via max pooling and directly estimating sentence scores using a lightweight MLP. Without requiring semantic consistency computations, our approach achieves substantial efficiency improvements while maintaining competitive performance with state-of-the-art baselines through adaptive aggregation of internal feature representations. 

---
# Architecture, Not Scale: Circuit Localization in Large Language Models 

**Authors**: Sohan Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2605.08853)  

**Abstract**: Mechanistic interpretability assumes that circuit analysis becomes harder as models scale. We challenge this assumption by showing that the attention architecture matters more than parameter count. Studying three circuit types across Pythia and Qwen2.5, we find that grouped query attention produces circuits that are far more concentrated and mechanistically stable than standard multi-head attention at comparable scales. The same concentration pattern holds across indirect object identification, induction heads, and factual recall. Within a single architecture family (Qwen2.5), factual recall circuits undergo a discrete phase transition above a critical scale, collapsing to a single bottleneck rather than degrading gradually. These findings suggest that some architectural choices make large models more tractable to study and that interpretability difficulty is not a fixed consequence of model size. 

---
# EmoS: A High-Fidelity Multimodal Benchmark for Fine-grained Streaming Emotional Understanding 

**Authors**: Pengze Guo, Jingxi Liang, Zhiwen Xie, Qifeng Wang, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.08847)  

**Abstract**: In the context of today's high-pressure, aging society, the demand for large-scale emotional models capable of providing empathetic support is more critical than ever. However, existing benchmarks fail to simultaneously achieve ecological validity, signal clarity, and reliable fine-grained labeling. We introduce EmoS, a high-fidelity bilingual benchmark designed to resolve the limitations of ecological validity and noise in existing datasets by combining strictly filtered static slices with a dynamic Streaming Monologue subset. Supported by a rigorous dual-layer human annotation pipeline, EmoS provides trusted ground truth that captures continuous emotional evolution. Empirical results show that fine-tuning MLLMs (multimodal large language models) on EmoS yields significant gains over zero-shot baselines, laying the foundation for the training and evaluation of future emotion recognition models and empathy models. The dataset and code are publicly available at this https URL. 

---
# XPERT: Expert Knowledge Transfer for Effective Training of Language Models 

**Authors**: Chang Liu, Boyu Shi, Xu Yang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2605.08842)  

**Abstract**: Mixture-of-Experts (MoE) language models organize knowledge into explicitly routed expert modules, making expert-level representations traceable and analyzable. By analyzing expert activation patterns in MoE large language models (LLMs), we find that a subset of experts is consistently activated across diverse knowledge domains. These common experts encode cross-domain, generalizable knowledge that is closely related to model generalization, naturally raising the question of how such identifiable expert knowledge can be practically reused. Motivated by this observation, we propose XPERT, a framework that extracts, consolidates, and reuses expert knowledge from pre-trained MoE LLMs to support more effective training of language models across different model scales. XPERT identifies cross-domain experts via inference-only analysis, refines their representations through tensor decomposition, and adapts the extracted knowledge to reuse in downstream models. Experiments on language understanding and dialogue generation benchmarks show that models benefiting from reused expert knowledge achieve consistently stronger performance and faster convergence compared to strong baselines. These results highlight MoE LLMs as structured and reusable knowledge sources, and demonstrate the value of expert-level knowledge reuse for improving model training. 

---
# ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing 

**Authors**: Yongqi An, Chang Lu, Kuan Zhu, Tao Yu, Chaoyang Zhao, Hong Wu, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08840)  

**Abstract**: Large language models (LLMs) face growing challenges in efficient generative inference due to the increasing memory demands of Key-Value (KV) caches, especially for long sequences. Existing eviction methods typically retain KV pairs with high attention weights but overlook the impact of attention redistribution caused by token removal, as well as the spatial-temporal dynamics in KV selection. In this paper, we propose ReST-KV, a robust KV eviction method that combines layer-wise output Reconstruction and Spatial-Temporal smoothing to provide a more comprehensive perspective for the KV cache eviction task. Specifically, ReST-KV formulates KV cache eviction as an optimization problem that minimizes output discrepancies through efficient layer-wise reconstruction. By directly modeling how each token's removal affects the model output, our method naturally captures attention redistribution effects, going beyond simplistic reliance on raw attention weights. To further enhance robustness, we design exponential moving average smoothing to handle temporal variations and an adaptive window-based mechanism to capture spatial patterns. Our method, ReST-KV, significantly advances performance on long-context benchmarks. It surpasses state-of-the-art baselines by 2.58% on LongBench and 15.2% on RULER. Additionally, ReST-KV consistently outperforms existing methods on Needle-in-a-Haystack and InfiniteBench, all while achieving a remarkable 10.61$\times$ reduction in decoding latency at 128k context length. The code is publicly available at this https URL to facilitate reproducibility and further research. 

---
# Generating Leakage-Free Benchmarks for Robust RAG Evaluation 

**Authors**: Jiayi Liu, Jiaxing Zhang, Bowen Jin, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2605.08838)  

**Abstract**: Retrieval-augmented generation (RAG) is widely used to augment large language models (LLMs) with external knowledge. However, many benchmark datasets, designed to test RAG performance, comprise many questions that can already be answered from an LLM's parametric memory. This leads to unreliable evaluation. We refer to this phenomenon as knowledge leakage: cases where RAG tasks are solvable without retrieval. This issue worsens over time due to benchmark aging. As benchmarks are reused for training, their contents are increasingly absorbed into model parameters, making them less effective for evaluating retrieval.
We introduce SeedRG, a semi-synthetic benchmark generation pipeline that mitigates knowledge leakage and addresses the issue of benchmark aging. Starting from a seed benchmark dataset, SeedRG extracts a reasoning graph from question-context pairs to capture their underlying reasoning structure, and then generates new examples via type-constrained entity replacement. This process produces structurally similar but novel instances that are unlikely to exist in the model's parametric knowledge, while preserving the original reasoning patterns. To ensure quality, we incorporate two verification steps: (1) a reasoning-graph consistency check to maintain task difficulty, and (2) a knowledge-leakage filter to exclude instances answerable without retrieval. 

---
# The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans 

**Authors**: Odysseas S. Chlapanis, Orfeas Menis Mastromichalakis, Christos H. Papadimitriou  

**Link**: [PDF](https://arxiv.org/pdf/2605.08837)  

**Abstract**: Abstract concepts - justice, theory, availability - have no single perceivable referent; in the human brain, their meaning emerges from a web of experiences, affect, and social context. Do large language models (LLMs) ground abstract concepts in a similar way? We study this by replicating property-generation experiments from cognitive science on 21 frontier and open-weight LLMs. Across models and experiments, we find a consistent pattern: when compared to humans, models rely too heavily on word associations, and underproduce properties tied to emotion and internal states. This yields a large and consistent grounding gap: no model exceeds a Pearson correlation r=0.37 with human responses, compared to a human-to-human ceiling above r=0.9. To better interpret this gap, we also replicate a rating experiment on grounding categories and find that here LLMs align more closely with human judgment, and alignment improves as models get larger. We then use sparse autoencoders (SAEs) to inspect whether this information is also reflected in the models' internal features, and we do identify features connected to grounding dimensions such as "sensorimotor" and "social". These findings suggest that current LLMs can recover grounding dimensions when explicitly queried, but do not recruit them in a human-like way when words are generated freely. 

---
# SimReg: Achieving Higher Performance in the Pretraining via Embedding Similarity Regularization 

**Authors**: Yan Sun, Guoxia Wang, Jinle Zeng, JiaBin Yang, Shuai Li, Li Shen, Dacheng Tao, DianHai Yu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08809)  

**Abstract**: Pretraining large language models (LLMs) with next-token prediction has led to remarkable advances, yet the context-dependent nature of token embeddings in such models results in high intra-class variance and inter-class similarity, thus hindering the efficiency of representation learning. While similarity-based regularization has demonstrated benefit in supervised fine-tuning and classification tasks, its application and efficacy in large-scale LLM pretraining remains underexplored. In this work, we propose the SimReg, an embedding similarity regularization loss that explicitly encourages token representations with the same ground-truth label within each sequence to be more similar, while enforcing separation from different-label tokens via a contrastive loss. Our analysis reveals that this mechanism introduces gains by enlarging multi-classification margins, thereby enabling more efficient classification. Extensive experiments across dense and Mixture-of-Experts (MoE) architectures demonstrate that SimReg consistently accelerates training convergence by over 30% and improves average zero-shot downstream performance by over 1% across standard benchmarks. Further ablation studies and analyses offer practical insights into hyperparameter tuning and loss effectiveness. 

---
# Narrative Landscape: Mapping Narrative Dispositions Across LLMs 

**Authors**: Donghoon Jung, Jiwoo Choi, Songeun Chae, Seohyon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2605.08742)  

**Abstract**: This study proposes a quantitative framework for profiling LLM dispositions as stable, model-specific regularities in output under repeated, controlled elicitation. Using a structured narrative constraint-selection task administered across six frontier models and three instruction types, we operationalize disposition through two dimensions: "consistency", measured as cross-replication selection overlap via Jaccard similarity, and "diversity", measured as dispersion across options via the inverse Simpson index. We further introduce Narrative Landscape, a PCA-based visualization that maps each model's selection profile into a shared space for direct comparison. Results reveal a clear rigidity-exploration spectrum across model families and show that instruction types shift the geometry of selection spaces even when scalar metrics appear similar, indicating that comparable scores can mask qualitatively distinct selection topologies. 

---
# Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning 

**Authors**: Zhengyang Zhao, Lu Ma, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08741)  

**Abstract**: Inference-time harnesses substantially improve large language models on complex reasoning tasks. However, the intrinsic capabilities of the underlying model remain unchanged by the addition of these external workflows. To bridge this gap, we introduce \emph{On-Policy Harness Self-Distillation} (OPHSD), which employs the harness-augmented current model as a teacher for self-distillation, thereby introducing extra supervisory signals from the harness beyond training data. OPHSD internalizes task-specific harness capabilities into the student model, yielding robust generalizability and strong standalone performance across diverse reasoning tasks. Evaluated across draft--verify harness for text classification and plan--solve for mathematical reasoning tasks, OPHSD consistently outperforms strong baselines (e.g., +10.83\% over OPSD on HMMT25). Our analysis further indicates that reattaching the harness during inference yields no additional benefits and can even degrade performance, suggesting that complex harnesses need not always be permanent fixtures; instead, they can serve as temporary training scaffolds whose benefits are permanently fed back into the base model. Our code and training data are available at this https URL. 

---
# Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents 

**Authors**: Minzheng Wang, Run Luo, Yanbo Wang, Zichen Liu, Yuqiao Tan, Tao Tan, Xu Nan, Yinhe Zheng, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.08721)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for closed-ended tasks, extending it to open-ended social language games via self-play reveals a critical issue: evolution impasse. Due to the vast strategy space, language agents frequently converge to homogenized behaviors, leading to deterministic match outcomes that eliminate the gradient signals necessary for policy evolution. To tackle this issue, we propose Dual-scale Evolutionary Policy Training (DEPT) for social language games. DEPT introduces a time-scaled evolutionary perception mechanism that detects impasse by quantifying dual-scale value baseline divergence alongside match entropy. Upon perceiving the collapse, it then activates asymmetric advantage reshaping to dynamically modulate the optimization landscape for intervention. Thus, our method effectively restores gradient signals and enforces sustained strategic exploration. Extensive experiments on multiple social language games demonstrate that DEPT outperforms strong baselines, avoiding policy degeneration and driving the continuous evolution of social language agents. 

---
# AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems 

**Authors**: Boxuan Zhang, Jianing Zhu, Zeru Shi, Dongfang Liu, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08715)  

**Abstract**: LLM-based multi-agent systems are increasingly deployed on long-horizon tasks, but a single decisive error is often accepted by downstream agents and cascades into trajectory-level failure. Existing work frames this as \emph{post-hoc failure attribution}, diagnosing the responsible agent and step after the trajectory has ended. However, this paradigm forfeits any opportunity to intervene while trajectory is still unfolding. In this work, we introduce AgentForesight, a framework that reframes this problem as online auditing: at each step of an unfolding trajectory, an auditor observes only the current prefix and must either continue the run or alarm at the earliest decisive error, without access to future steps. To this end, we curate AFTraj-2K, a corpus of agentic trajectories across Coding, Math, and Agentic domains, in which safe trajectories are retained under a strict curation pipeline and unsafe trajectories are annotated at the step of their decisive error via consensus among multiple LLM judges. Built on that, we develop AgentForesight-7B, a compact online auditor trained with a coarse-to-fine reinforcement learning recipe that first equips it with a risk-anticipation prior at the failure boundary on adjacent safe/unsafe prefix pairs, then sharpens this prior into precise step-level localization under a three-axis reward jointly targeting the what, where, and who of an audit verdict. Across AFTraj-2K and an external Who\&When benchmark, AgentForesight-7B outperforms leading proprietary models, including GPT-4.1 and DeepSeek-V4-Pro, achieving up to +19.9% performance gain and 3$\times$ lower step localization error, opening the loop from post-hoc failures detection to enabling deployment-time intervention. Project page: this https URL 

---
# Structured Recurrent Mixers for Massively Parallelized Sequence Generation 

**Authors**: Benjamin L. Badger  

**Link**: [PDF](https://arxiv.org/pdf/2605.08696)  

**Abstract**: Over the last two decades, language modeling has experienced a shift from predominantly recurrent architectures that process tokens sequentially during training and inference to non-recurrent models that process sequence elements in parallel during training, which results in greater training efficiency and stability at the expense of lower inference throughput. Here we introduce the Structured Recurrent Mixer, an architecture that allows for algebraic conversion between a sequence parallel representation at train time and a recurrent representation at inference, notably without the need for specialized kernels or device-specific memory management. We show experimentally that this dual representation allows for greater training efficiency, higher input information capacity, and larger inference throughput and concurrency when compared to other linear complexity models. We postulate that recurrent models are poorly suited to extended sequence length scaling for information-rich inputs typical of language, but are well suited to scaling in the sample (batch) dimension due to their constant memory per sample. We provide Mojo/MAX inference implementations of SRMs exhibiting 12x the throughput and 170x the concurrency of similarly powerful Transformers inferenced on vLLM, increases characteristic of Pytorch implementations resulting in a 30\% increase in compute-constant GSM8k Pass@k. We conclude by demonstrating that SRMs are effective reinforcement learning training candidates. 

---
# Explanation Fairness in Large Language Models: An Empirical Analysis of Disparities in How LLMs Justify Decisions Across Demographic Groups 

**Authors**: Gautam Veldanda  

**Link**: [PDF](https://arxiv.org/pdf/2605.08671)  

**Abstract**: Large language models (LLMs) are increasingly deployed not only to make decisions but to explain them. While AI decision fairness has been studied extensively, the fairness of AI explanations (whether LLMs justify decisions with equal quality, depth, tone, and linguistic sophistication across demographic groups) has received little attention. This paper introduces the Explanation Fairness Taxonomy (EFT), a framework comprising five formally defined, operationalizable dimensions: Verbosity Disparity, Sentiment Disparity, Epistemic Hedging Disparity, Decision-Linked Explanation Disparity, and Lexical Complexity Disparity. The taxonomy is instantiated in a controlled empirical study across 80 prompt templates, four consequential decision domains (hiring, medical triage, credit assessment, legal judgment), and five LLMs: GPT-4.1, Claude Sonnet, LLaMA 3.3 70B, GPT-OSS 120B, and Qwen3 32B. Two novel black-box metrics are introduced: the Hedging Density Score (HDS) and the Explanation Faithfulness Proxy (EFP), a heuristic indicator of decision-linked explanation variation. Across up to 400 prompt pairs, all eight EFT metrics show statistically significant disparities (Cohen's d ranging from small to large, all p_BH < 10^(-62)). Model choice is strongly associated with disparity magnitude: Qwen3 32B exhibits verbosity disparities 5.9x larger than LLaMA 3.3 70B. Two prompting-based mitigations show significant reductions in EFP disparity (78-95%) but no significant effect on stylistic dimensions, consistent with the hypothesis that stylistic explanation inequalities are encoded in pre-training distributions and are not resolvable through deployment-level instruction alone. A reproducible measurement framework is offered for explanation-level fairness auditing, with implications for AI regulation and deployment practice. 

---
# Hint Tuning: Less Data Makes Better Reasoners 

**Authors**: Siqi Fan, Minghao Li, Xiaoqian Ma, Xiusheng Huang, Zhuo Chen, Bowen Qin, Liujie Zhang, Shuo Shang, Weihang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08665)  

**Abstract**: Large reasoning models achieve high accuracy through extended chain-of-thought but generate 5--8 more tokens than necessary, applying verbose reasoning uniformly regardless of problem difficulty. We propose Hint Tuning, a data-efficient approach that teaches models to calibrate reasoning depth. Our key insight: the corresponding instruct model serves as an ideal difficulty probe. By testing what the instruct model can solve with varying guidance, we automatically construct training data across three states: No-Hint (direct answer), Sparse-Hint (minimal prefix), and Full-Hint (complete reasoning). This converts the abstract challenge of difficulty labeling into a measurable consistency check between the instruct and reasoning models. With only 1K self-annotated samples, Hint Tuning achieves 24--66% token reduction (31.5% average) across mainstream reasoning models (Qwen3-Thinking, DeepSeek-R1-Distill) at multiple scales (4B--32B) while maintaining competitive accuracy on five benchmarks. Unlike methods requiring massive distillation datasets or expensive RL, we achieve superior efficiency through simple alignment with the instruct model's capabilities. 

---
# AgentCollabBench: Diagnosing When Good Agents Make Bad Collaborators 

**Authors**: Aritra Mazumder, Shubhashis Roy Dipta, Nusrat Jahan Lia, Tanzila Khan, Kainat Raisa Hossain, Nehaa Shri, Shubhrangshu Debsarkar, Humayra Tasnim, Gour Gupal Talukder Shawon, Debjoty Mitra, Sumaiya Ahmed Rani, Al Jami Islam Anik, Al Nafeu Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08647)  

**Abstract**: Multi-agent systems achieve state-of-the-art outcomes through peer collaboration. However, when an agent in the pipeline silently drops a constraint, the system's final output may look correct even though the reasoning chain was quietly corrupted, and existing outcome-based evaluations are blind to such multi-hop process failures. To make these vulnerabilities measurable before deployment, we introduce AgentCollabBench, a diagnostic benchmark of 900 human-validated tasks spanning software engineering, DevOps, and data engineering. Each task isolates one of four behavioral risks: instruction decay (does a constraint survive peer pressure?), false-belief contagion (does a falsehood spread through consensus?), context leakage (does information bleed between tasks?), and tracer durability (does marked data reach the final agent?). Evaluating four modern LLMs (GPT 4.1 mini, Gemini 2.5 Flash Lite, Qwen-3.5-35B-A3B, and Llama 3.1 8B Instruct), we expose model-specific vulnerability profiles invisible to outcome-only evaluation; Qwen-3.5-35B-A3B, for example, leads on tracer durability and instruction stability, while GPT 4.1 mini leads on leakage containment and false-belief resistance. Beyond per-model differences, communication topology emerges as a primary risk factor that explains 7-40% of the variance in multi-hop information survival. The effect traces to a synthesis bottleneck specific to converging-DAG nodes: an agent weighing competing parent inputs discards constraints carried by a minority branch, a bottleneck structurally absent from linear chains. AgentCollabBench demonstrates that suboptimal topology can silently erase the safeguards of highly capable models, arguing that multi-agent reliability is fundamentally a structural problem and that scaling model intelligence alone is no substitute for architecture. 

---
# EdgeFlowerTune: Evaluating Federated LLM Fine-Tuning Under Realistic Edge System Constraints 

**Authors**: Jiaxiang Geng, Yiyi Lu, Lunyu Zhao, Yan Gao, Nicholas D. Lane, Bing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.08636)  

**Abstract**: Federated fine-tuning offers a promising paradigm for adapting large language models (LLMs) on edge devices by leveraging the rich, diverse, and continuously generated data from smartphones and IoT devices without compromising user data privacy. Such edge-side adaptation can improve model personalization, robustness, and responsiveness to local contexts. However, the practical feasibility of federated LLM fine-tuning on real edge devices remains unclear, as most existing work focuses on cross-silo or simulation-based settings, overlooking the resource and runtime constraints that determine whether a method is deployable on real edge systems. We present EdgeFlowerTune, a deployment-oriented benchmark for federated LLM fine-tuning under realistic edge-system constraints. EdgeFlowerTune jointly evaluates model quality and system costs, including communication, wall-clock latency, memory usage, energy consumption, and robustness to dynamic edge conditions. To compare methods in terms of effectiveness, efficiency, and robustness, EdgeFlowerTune introduces three complementary protocols: Quality-under-Budget, Cost-to-Target, and Robustness. We instantiate EdgeFlowerTune as a real-device platform built on Flower and MobileFineTuner, spanning commercial Android smartphones and NVIDIA edge development boards. Our benchmark results show that accuracy-only evaluation can lead to misleading conclusions: methods with similar final quality may differ substantially in deployability once realistic system constraints are considered. EdgeFlowerTune provides a reproducible benchmark for system-aware evaluation of federated LLM fine-tuning at the edge. 

---
# PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode Speculative Decoding 

**Authors**: Zihao An, Taichi Liu, Ziqiong Liu, Dong Li, Ruofeng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2605.08632)  

**Abstract**: Speculative decoding accelerates Large Language Models (LLMs) inference by using a lightweight draft model to propose candidate tokens that are verified in parallel by the target model. However, existing draft model training objectives are not directly aligned with the inference-time goal of maximizing consecutive token acceptance. To address this issue, we reformulate the draft model optimization objective, shifting the focus from token prediction accuracy to the overall acceptance length. In this paper, we build upon PARD to propose PARD-2, a dual-mode speculative decoding framework with Confidence-Adaptive Token (CAT) optimization. This approach adaptively reweights each token to better align with the verification process. Notably, PARD-2 enables a single draft model to support both target-dependent and target-independent modes. Experiments across diverse models and tasks demonstrate that PARD-2 achieves up to 6.94$\times$ lossless acceleration, surpassing EAGLE-3 by 1.9$\times$ and PARD by 1.3$\times$ on Llama3.1-8B. Our code is available at this https URL. 

---
# 100,000+ Movie Reviews from Kazakhstan: Russian, Kazakh, and Code-Switched Texts 

**Authors**: Rustem Yeshpanov  

**Link**: [PDF](https://arxiv.org/pdf/2605.08600)  

**Abstract**: We present a new publicly available corpus of 100,502 movie reviews from Kazakhstan collected from this http URL, spanning 2001-2025 and covering 4,943 unique titles. The dataset is multilingual, consisting mainly of Russian reviews alongside Kazakh and code-switched texts. Reviews are manually annotated for language and sentiment polarity, and 11,309 reviews additionally contain explicit user-provided ratings. We define two sentiment tasks -- three-way polarity classification and five-class score classification -- and benchmark classical BoW/TF-IDF baselines against multilingual transformer models (mBERT, XLM-RoBERTa, RemBERT). Experimental results show that transformer models consistently outperform classical baselines on polarity classification, while score classification remains challenging under leakage-controlled evaluation due to severe class imbalance and subtle distinctions between adjacent rating levels. 

---
# Source or It Didn't Happen: A Multi-Agent Framework for Citation Hallucination Detection 

**Authors**: Mingzhe Li, Zhiqiang Lin, Shiqing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.08583)  

**Abstract**: Large language models are increasingly used in scientific writing, yet they can fabricate citation-shaped references that appear plausible but fail bibliographic verification. Existing detectors often reduce verification to binary found/not-found decisions and rely on brittle parsing or incomplete retrieval, offering little field-level signal to auditors. We reframe citation hallucination detection as taxonomy-aligned field-level adjudication and introduce a 12-code taxonomy spanning Real, Potential, and Hallucinated citations. Based on this taxonomy, we build CiteTracer, a cascading multi-agent detector that extracts structured citations from PDF and BibTeX, retrieves evidence through cache lookup, URL fetch, scholar connectors, and web search, applies deterministic field matching, and routes ambiguous cases to class-specialist judgers. We release a benchmark of 2,450 synthetic citations built from real seeds with controlled LLM mutations, paired with 957 real-world fabricated citations drawn from ICLR 2026 and an anonymous conference desk-rejected submissions. CiteTracer reaches 97.1% accuracy on the synthetic benchmark, with class-level F1 scores of 97.0, 95.8, and 98.5 for Real, Potential, and Hallucinated, respectively, and detects 97.1% of fabrications on the real-world set without abstaining. Code: this https URL. 

---
# Coordinates of Capability: A Unified MTMM-Geometric Framework for LLM Evaluation 

**Authors**: Adib Sakhawat, Tahsin Islam, Takia Farhin, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08522)  

**Abstract**: The evaluation of Large Language Models (LLMs) faces a critical challenge in construct validity, where fragmented benchmarks and ad hoc metrics frequently conflate method variance, such as prompt sensitivity, with true latent capabilities. Concurrently, emerging research suggests that LLM capabilities and outputs can be modeled as continuous geometric manifolds. In this Systematization of Knowledge (SoK), we bridge these paradigms by proposing a generalized Multi-Trait Multi-Method (MTMM) framework for LLM evaluation. We formalize and unify nine evaluation metrics, including Paraphrase Instability, Drift Score, Overton Width, and Pluralism Score, interpreting them not as isolated scalar values but as geometric measurements within a shared latent coordinate space. This spatial unification factorizes model behavior into three orthogonal latent dimensions: (1) Instability and Sensitivity, (2) Position and Alignment, and (3) Coverage and Expressiveness. By systematically separating task-irrelevant perturbations from true capability spans, the framework provides a theoretically grounded and domain-agnostic taxonomy for robust and empirically stable benchmark design. 

---
# A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models 

**Authors**: Hamid Kazemi, Atoosa Chegini, Maria Safi  

**Link**: [PDF](https://arxiv.org/pdf/2605.08513)  

**Abstract**: Safety alignment in language models operates through two mechanistically distinct systems: refusal neurons that gate whether harmful knowledge is expressed, and concept neurons that encode the harmful knowledge itself. By targeting a single neuron in each system, we demonstrate both directions of failure -- bypassing safety on explicit harmful requests via suppression, and inducing harmful content from innocent prompts via amplification -- across seven models spanning two families and 1.7B to 70B parameters, without any training or prompt engineering. Our findings suggest that safety alignment is not robustly distributed across model weights but is mediated by individual neurons that are each causally sufficient to gate refusal behavior -- suppressing any one of the identified refusal neurons bypasses safety alignment across diverse harmful requests. 

---
# A Single Layer to Explain Them All:Understanding Massive Activations in Large Language Models 

**Authors**: Zeru Shi, Zhenting Wang, Fan Yang, Qifan Wang, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08504)  

**Abstract**: We investigate the origins of massive activations in large language models (LLMs) and identify a specific layer named the \textbf{Massive Emergence Layer (ME Layer)}, that is consistently observed across model families, where massive activations first emerge and subsequently propagate to deeper layers through residual connections. We show that, within the ME Layer both the RMSNorm and the FFN parameters jointly contribute to the emergence of massive activations. Once formed, the massive activation token representation remains largely invariant across layers, reducing the diversity of hidden representations passed to the attention module. Motivated by this limitation, we propose a simple and effective method to reduce the rigidity of the massive activation token. Our approach consistently improves LLM performance across multiple tasks, including instruction following and math reasoning, in both training free and fine tuning settings. Moreover, we show that our method mitigates attention sinks by selectively weakening their influence, elucidating their origin at the hidden state level and shedding new light on principled mitigation strategies. 

---
# NARRA-Gym for Evaluating Interactive Narrative Agents 

**Authors**: Yue Huang, Yuchen Ma, Jiayi Ye, Wenjie Wang, Zipeng Ling, Xingjian Hu, Yuexing Hao, Zichen Chen, Zhangchen Xu, Yunhong He, Zhengqing Yuan, Yujun Zhou, Kehan Guo, Chaoran Chen, Toby Jia-Jun Li, Stefan Feuerriegel, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08503)  

**Abstract**: Interactive narrative tasks require LLMs to sustain a coherent, evolving story while adapting to a user over multiple turns. However, suitable benchmarks for this setting are limited: existing evaluations often focus on static prompts, isolated story generations, or post-hoc ratings, and therefore miss whether models can jointly manage story generation, long-context state and pacing, character simulation, empathic personalization, and story-grounded artifacts. We introduce NARRA-Gym, an executable evaluation environment that turns a sparse emotional seed into a complete interactive story episode and logs the full model-in-the-loop trajectory, including story construction, memory updates, planning, pacing interventions, and optional artifact synthesis. We evaluate nine frontier LLMs using a controlled LLM-as-judge sweep over eight benchmark personas and a human evaluation in which participants rate customized model outputs. Our results show substantial variation across models, personas, and evaluation dimensions: models that produce fluent stories can still fail on robustness, user experience, or resistance-sensitive personalization. These findings suggest that interactive narrative offers a useful benchmark for evaluating long-horizon, user-adaptive LLM behavior beyond isolated story quality. 

---
# Do Agents Need to Plan Step-by-Step? Rethinking Planning Horizon in Data-Centric Tool Calling 

**Authors**: Naoki Otani, Nikita Bhutani, Hannah Kim, Dan Zhang, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2605.08477)  

**Abstract**: Explicit planning is a critical capability for LLM-based agents solving complex data-centric tasks, which require precise tool calling over external data sources. Existing strategies fall into two paradigms based on planning horizon: (1) full-horizon (FH), which generates a complete plan before execution, and (2) single-step horizon (SH), which interleaves each action (tool call) with incremental reasoning and observation. While step-by-step execution is a common default under the assumption that eager execution monitoring is necessary for adaptability, we revisit this assumption for well-defined data-centric tasks. Our controlled empirical study isolates planning horizon as the key architectural feature and systematically analyzes the effects of topological complexity and tool robustness on both paradigms. Our experiments across Knowledge Base Question Answering and Multi-hop QA show that FH planning with lazy replanning achieves accuracy parity with SH across varying depths, breadths, and robustness levels, while using 2-3x fewer tokens. These findings suggest that for well-defined data-centric tasks, eager step-wise monitoring is often unnecessary, and full-horizon planning with on-demand replanning can offer a more efficient default. 

---
# A Computational Operationalisation of Competing Maturational Theories of Syntactic Development via Statistical Grammar Induction 

**Authors**: Mila Marcheva, Suchir Salhan, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.08476)  

**Abstract**: This paper is concerned with what intermediate syntactic categories children acquire during first language development, and in what order. Maturational theories make different predictions. Bottom-up accounts (GROWING) propose that lexical and inflectional structure emerges first, while inward accounts (INWARD) predict early access to discourse-related categories. We computationally operationalise these hypotheses of staged syntactic emergence using statistical grammar induction, asking what each proposed ordering makes learnable when input and learning algorithm are held constant. Our framework makes category acquisition explicit and allows us to explore how different maturational orderings shape the structure that can be learned under identical conditions. Based on this operationalisation, the GROWING account significantly outperforms the INWARD account across three evaluation metrics. 

---
# PYTHALAB-MERA: Validation-Grounded Memory, Retrieval, and Acceptance Control for Frozen-LLM Coding Agents 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08468)  

**Abstract**: Local LLM-based coding agents increasingly work in settings where correctness is earned through execution feedback, persistent state, and bounded repair, not through a single fluent answer. Static retrieval, long-context prompting, self-refinement, execution-feedback repair, and reinforcement learning over model weights each address part of this setting, but they do not jointly provide validation-grounded episodic memory, adaptive retrieval-action selection, delayed credit assignment, and structural skill reuse around a frozen local model. We introduce PYTHALAB-MERA, a lightweight external controller for local validation-conditioned code generation. The frozen language model proposes complete source files; the controller decides which memory records and AST-derived skills should enter the next prompt, validates each candidate through a fail-fast pipeline, converts validation outcomes into bounded shaped rewards, and propagates delayed credit through TD(lambda)-style eligibility traces. We evaluate the implementation as a local CLI artifact on reinforcement-learning coding tasks with strict validation gates. In the measured hard RL setting with three tasks, three repetitions, and a three-attempt budget, PYTHALAB-MERA passed 8/9 strict validations; the self-refinement baseline and the investigated GRACE extension each passed 0/9. These results support a deliberately bounded claim: in this recorded setting, the external memory-and-retrieval controller improved validation success. They do not establish general-purpose code synthesis, state-of-the-art performance, formal program correctness, or formal safety. 

---
# Do Benchmarks Underestimate LLM Performance? Evaluating Hallucination Detection With LLM-First Human-Adjudicated Assessment 

**Authors**: I. F. Atasoy, B. Mutlu, E. A. Sezer, A. Wahdan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08462)  

**Abstract**: Hallucination remains a persistent challenge in Large Language Models (LLMs), particularly in context-grounded settings such as RAG and agentic AI systems. This study focuses on contextual hallucination detection in summarization tasks. We analyze the QAGS-C and SummEval datasets by comparing original benchmark annotations with reason and span-based predictions from Gemini 2.5 Flash and GPT-5 Mini. To address systematic divergences between human labels and LLM judgments, we re-evaluated all conflicted samples through a human adjudication process involving 2 cross-cultural adjudicators. Following this re-evaluation, triple agreement (between human, GPT, and Gemini) increased by 6.38% for QAGS-C and 7.62% for SummEval. Similarly, model accuracy improved, with GPT increasing by 4.25% on QAGS-C and 2.34% on SummEval, while Gemini showed gains of 8.51% and 3.80%, respectively. Notably, adjudicators frequently sided with the models' judgments over original human annotations when LLMs provided explicit reasoning. Overall human adjudicator agreement ranged between 83% and 87%. These findings suggest that for ambiguity-prone tasks, single-pass annotations may be insufficient, and model-assisted re-evaluation yields more reliable benchmarks. 

---
# Revisiting the syntax of imperatives in Yemeni Arabic: An Agree across phases approach 

**Authors**: Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2605.08447)  

**Abstract**: This article revisits the syntax of imperatives in Yemeni Arabic proposing an Agree acros phases (AAP) approach. I argue that the AAP approach successfully accounts for both simple and complex imperative constructions, including A'-chain structures, by establishing a close interactions between syntax and discourse. The study demonstrates that this interface is motivated by the interpretive and performative functions associated with imperatives, linking informational structure with propositional structure. It is also proposed that the thematic subject of imperatives is a 2-person pro, whereas any overt pronominal or nominal element occurring preverbally is not a subject, but rather a C-domain element, precisely aboutness topic. These topics serve as the logical subjects of imperatives and enter into a coreferentiality relationship with pro. This relation is analyzed as APP involving Match, yielding both local and non-local A'-chains. For core imperatives, viz., lacking an overt topic, I propose a null topic to (re)merge in Spec,TopP, whose interpretation depends on the discourse. 

---
# Can Language Models Identify Side Effects of Breast Cancer Radiation Treatments? 

**Authors**: Natalie Seah, Danielle S. Bitterman, Daphna Spiegel, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08439)  

**Abstract**: Accurately communicating the side effects of cancer treatments to cancer survivors is critical, particularly in settings such as informed consent, where clinicians must clearly and comprehensively convey potential treatment toxicities. However, this task remains challenging due to clinical knowledge deficits about adverse treatment effects and fragmentation across electronic health record (EHR) systems. Large language models (LLMs) have the potential to assist in this task, though their reliability in oncology survivorship contexts remains poorly understood. We present a deployment-oriented stress-testing framework for evaluating LLM-generated radiation side effect lists in breast cancer treatment and survivorship care. Using 21 breast cancer patient profiles, we construct paired patient clinical scenarios that differ only in radiotherapy regimens to evaluate seven instruction-tuned LLMs under multiple prompting regimes. We then compare LLM outputs to a clinician-curated reference derived from informed consent documents at two major academic medical centers and developed by a team including more than seven breast radiation oncologists. The reference maps radiation dose-fractionation, fields, and locations to associated toxicities, broken down by frequency and temporal onset. Across models, we reveal sensitivity to minor documentation changes, trade-offs between precision and recall, and systematic under-recall of rare and long-term side effects. When used alone, constraints on the number of side effects generated reduce precision, and grounding outputs in clinician-curated side effect lists substantially improves reliability and robustness. These findings highlight important limitations of LLM use in oncology and suggest practical design choices for safer and more informative survivorship-focused applications. 

---
# Magis-Bench: Evaluating LLMs on Magistrate-Level Legal Tasks 

**Authors**: Ramon Pires, Thales Sales Almeida, Celio Larcher Junior, Giovana Bonás, Hugo Abonizio, Marcos Piau, Roseval Malaquias Junior, Thiago Laitz, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2605.08437)  

**Abstract**: Existing benchmarks for legal AI focus primarily on tasks where LLMs must produce legal arguments or documents, yet the capacity to \emph{judge} such arguments -- weighing competing claims, applying doctrine to facts, and rendering reasoned decisions -- is arguably as fundamental to a well-functioning legal system as advocacy itself. We introduce Magis-Bench, a benchmark for evaluating LLMs on magistrate-level writing tasks derived from recent Brazilian competitive examinations for judicial positions. Magis-Bench comprises 74 questions from eight examinations conducted between 2023 and 2025, including discursive legal analysis questions with multi-turn structure and practical exercises requiring the composition of complete civil and criminal judicial sentences. We evaluate 23 state-of-the-art LLMs using an LLM-as-a-judge methodology with four independent frontier models as evaluators. Our results show strong inter-judge agreement (Kendall's $W = 0.984$; pairwise Kendall's $\tau \ge 0.897$), with Google's Gemini-3-Pro-Preview achieving the highest average score (6.97/10), followed by Gemini-3-Flash-Preview (6.67) and Claude-4.5-Opus (6.46). Even the best-performing models score below 70\% of the maximum, indicating that judicial-level legal reasoning and writing remain challenging for current LLMs. We release the complete benchmark, model outputs, and evaluation code to support further research on legal AI capabilities. 

---
# A Semantic-Sampling Framework for Evaluating Calibration in Open-Ended Question Answering 

**Authors**: Zhanliang Wang, Jiancong Xiao, Ruochen Jin, Shu Yang, Bojian Hou, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08432)  

**Abstract**: Calibration measures whether a model's predicted confidence aligns with its empirical accuracy, and is central to the reliable deployment of large language models (LLMs) in high-stakes domains such as medicine and law. While much recent work focuses on improving LLM calibration, the equally important question of how to evaluate it in realistic settings remains underdeveloped. Open-ended question answering (QA), the most common deployment setting for modern LLMs, is where existing evaluation methods fall short: logit-based metrics need restricted output formats and internal probabilities; verbalized confidence is self-reported and often overconfident; and sampling-based methods rely on task-specific extraction rules without a clear finite-sample target. We introduce Sem-ECE (Semantic-Sampling Expected Calibration Error), a calibration evaluation framework for open-ended QA that samples answers from the model, groups them into semantic classes, and uses the resulting frequencies as confidence. We study two estimators within this framework: Sem$_1$-ECE, the same-sample self-consistency score, and Sem$_2$-ECE, a held-out variant that separates answer selection from confidence evaluation. We prove both are asymptotically unbiased, and further show that they agree on easy questions but diverge on hard ones with Sem$_2$ achieving strictly smaller calibration error, so their gap also serves as a diagnostic for question difficulty. Experiments on three open-ended QA benchmarks across five leading commercial LLMs match our theoretical predictions and show that Sem-ECE outperforms verbalized confidence and existing sampling-based methods, while complementing logit-based evaluation when internal probabilities are unavailable. 

---
# Effective Explanations Support Planning Under Uncertainty 

**Authors**: Hanqi Zhou, Britt Besch, Charley M. Wu, Tobias Gerstenberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.08406)  

**Abstract**: Explaining how to get from A to B can be challenging. It requires mentally simulating what the listener will do based on what they are told. To capture this process, we propose a computational model that converts utterances into action plans: a large language model translates an explanation into program-like guidance (a policy prior and value map), and a planning agent executes it under partial observability. We score explanations by the efficiency and reliability of the resulting paths, penalizing replanning. Across four preregistered experiments, we collect a corpus of 1,200 explanations over 24 maps, elicit helpfulness judgments, measure baseline navigation, and test behavior with explanations of differing quality. Higher-scored explanations are judged more helpful and improve navigation: participants with explanations outperform those without, and high-scoring explanations help more than low-scoring ones. Together, these results show procedural explanation as utility-guided communication shaped by how language can be grounded into action under uncertainty. 

---
# Built Environment Reasoning from Remote Sensing Imagery Using Large Vision--Language Models 

**Authors**: Dongdong Wang, Deepak Balakrishnan, Ravi Srinivasan, Shenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08404)  

**Abstract**: This work investigates the use of large language models (LLMs) for tasks in smart cities. The core idea is to leverage remote sensing imagery to characterize the built environment, including design suggestions, constructability assessment, landuse patterns, and risk identification. We examine remote sensing imagery at multiple spatial scales as inputs for multimodal language modeling and evaluate their effects on built-environment-related reasoning. In addition, we compare state-of-the-art LLMs, including InternVL and Qwen, in terms of accuracy and reliability when generating built environment recommendations. The results demonstrate the potential of integrating remote sensing imagery with large language models to assist smart cities and decision-making. 

---
# AIPO: : Learning to Reason from Active Interaction 

**Authors**: Junnan Liu, Linhao Luo, Thuy-Trang Vu, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2605.08401)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable reasoning capabilities, largely stimulated by Reinforcement Learning with Verifiable Rewards (RLVR). However, existing RL algorithms face a fundamental limitation: their exploration remains largely constrained by the inherent capability boundary of the policy model. Although recent methods introduce external expert demonstrations to extend this boundary, they typically rely on complete trajectory-level guidance, which is sample-inefficient, information-sparse, and may confine exploration to a static guidance space. Inspired by the potential of multi-agent systems, we propose $\textbf{AIPO}$, an enhanced reinforcement learning framework that improves LLM reasoning through active multi-agent interaction during exploration. Specifically, AIPO enables the policy model to proactively consult three functional collaborative agents, $\textit{Verify Agent}$, $\textit{Knowledge Agent}$, and $\textit{Reasoning Agent}$, when encountering reasoning bottlenecks, thereby receiving fine-grained and targeted guidance to actively expand its capability boundary during training. We further introduce a tailored importance sampling coefficient together with a clipping strategy to mitigate the off-policy bias and gradient vanishing issues that arise when learning from agent-provided feedback. After training, the policy model performs reasoning independently without relying on collaborative agents. Extensive experiments on diverse reasoning benchmarks, including AIME, MATH500, GPQA-Diamond, and LiveCodeBench, show that AIPO consistently improves reasoning performance, generalizes robustly across different policy models and RLVR algorithms, and effectively expands the reasoning capability boundary of the policy model. 

---
# jina-embeddings-v5-omni: Text-Geometry-Preserving Multimodal Embeddings via Frozen-Tower Composition 

**Authors**: Florian Hönicke, Michael Günther, Andreas Koukounas, Kalim Akram, Scott Martens, Saba Sturua, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.08384)  

**Abstract**: In this work, we introduce frozen-encoder model composition, a novel approach to multimodal embedding models. We build on the VLM-style architecture, in which non-text encoders are adapted to produce input for a language model, which in turn generates embeddings for all varieties of input. We present the result: the jina-embeddings-v5-omni suite, a pair of models that encode text, image, audio, and video input into a single semantic embedding space. Our method is to extend the two Jina Embeddings v5 Text models to support additional media by adding encoders for images and audio. The backbone text embedding models and the added non-text media encoders remain frozen. We only trained the connecting components, representing 0.35% of the total weights of the joint model. Training is therefore much more efficient than full-parameter retraining. Additionally, the language model remains effectively unaltered, producing exactly the same embeddings for text inputs as the Jina Embeddings v5 Text models. Our evaluations show that this approach produces results that are competitive with the state-of-the-art, yielding nearly equal performance to larger multimodal embedding models. 

---
# Change My View? The Dynamics of Persuasion and Polarization in Online Discourse 

**Authors**: David Freeborn, Malihe Alikani, Anthony Sicilia  

**Link**: [PDF](https://arxiv.org/pdf/2605.08383)  

**Abstract**: Philosophical accounts of persuasion often assume that shared evidence and rational argumentation should lead to a convergence of views between peers, yet everyday discourse often suggests otherwise. In this study, we use large language models to analyze a corpus of debates on Reddit's r/ChangeMyView, where belief revision is publicly signaled. Large language models were asked, halfway through each discussion, to forecast whether such an acknowledgement would arise; their probabilistic estimates serve as a conversational baseline. Each reply was then coded, through a hybrid machine-assisted procedure, for ten familiar rhetorical strategies -- concession, empathy, logical challenge, credibility appeals, and so forth. Adding these strategic features markedly improves predictive power and yields a consistent pattern: moves that express concession or empathetic alignment substantially increase the prospect of belief change, whereas frontal refutation, credibility attacks, and topic deflection diminish it. The findings indicate that effective public reasoning depends as much on relational framing as on evidential content, and they invite a refinement of normative accounts of rational dialogue. 

---
# How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits 

**Authors**: Michael Li, Nishant Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2605.08348)  

**Abstract**: The circuits framework in mechanistic interpretability aims to identify causally important sparse subgraphs of model components, typically evaluated by measuring necessity and sufficiency. We measure circuit reuse, the proportion of components shared across per-example circuits within a task, and investigate two less-studied properties of this: consistency, the recurrence of components within a task, and specificity, their uniqueness to a task. Using edge attribution patching across six tasks and seven models, we find that within-task reuse is high and that shared components are necessary for task performance, with ablations causing up to $\sim$100% relative accuracy drops. However, circuits turn out not to be task-specific: ablating one task's circuit damages another task's performance about as much as that task's own circuit does. We discover that this is due to substantial overlap between circuits across tasks, which are causally important for performance. Some circuits do contain a smaller set of task-specific components, but these account for only a modest portion of circuit performance. Overall, our findings suggest that while circuit discovery at the level of attention heads and MLP layers identifies important components, their lack of task-specificity raises questions about the degree to which circuits can support targeted understanding and intervention on model behavior. 

---
# Sanity Checks for Long-Form Hallucination Detection 

**Authors**: Geigh Zollicoffer, Minh Vu, Hongli Zhan, Raymond Li, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2605.08346)  

**Abstract**: Hallucination detection methods for large language models increasingly operate on chain-of-thought reasoning traces, yet it remains unclear whether they evaluate the reasoning itself or merely exploit surface correlates of the final answer. We introduce a controlled-invariance methodology that exposes this distinction through two oracle tests: \textsc{Force}, which replaces each response's final answer with the ground truth while preserving the reasoning trace, and \textsc{Remove}, which strips answer-announcement steps while leaving the trajectory intact. This reveals if their predictive power derives from answer-level artifacts rather than from the structure or validity of intermediate reasoning. We further show that once these artifacts are controlled for, effective detection does not necessarily require complex learned representations: TRACT, a lightweight scorer built on lexical trajectory features (hedging trends, step-length dynamics, and cross-response vocabulary convergence), achieves strong robustness while remaining competitive with or outperforming existing baselines on unperturbed traces. These findings suggest that the current central challenge in reasoning-aware hallucination detection is not the absence of signal in the trace, but the failure to isolate it from endpoint cues. 

---
# SalesSim: Benchmarking and Aligning Multimodal Language Models as Retail User Simulators 

**Authors**: Yada Pruksachatkun, Elaine Wan, Lyanna Chen, Kai-Wei Chang, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08334)  

**Abstract**: We present SalesSim, a framework and testbed for evaluating the ability of Multimodal Large Language Models (MLLMs) to simulate realistic, persona-driven customer behavior in multi-turn, multi-modal, tool-augmented online retail conversations. Unlike prior work that treat user simulation as surface-level dialogue generation, SalesSim models retail interaction and decision-making as a grounded, agentic process, where shoppers with diverse backgrounds, preferences, and dealbreakers interact with a sales agent, seek clarifications, and make informed purchasing decisions. For evaluation, we design a suite of metrics centered on decision alignment, measuring the consistency between the simulator's actions and its persona specifications, as well as conversational quality. We find several behavioral gaps after benchmarking 6 open and closed-source state-of-the-art models. First, while models produce fluent conversations, they display significantly lower lexical diversity and overdisclosure of criteria across personas compared to human conversations. Second, models tend to be persuaded by sales agent suggestions and drift from persona specifications. Even the strongest model achieves less than 79% average alignment with its underlying persona specifications. To make progress on these limitations, we propose UserGRPO, a multi-turn, multi-objective reinforcement learning recipe to optimize both conversational fluency and decision alignment under persona specifications. Our experiments demonstrate that UserGRPO boosts decision alignment of the baseline model by 13.8% while improving conversational quality. By introducing SalesSim, we provide a new testbed for the community to investigate and improve the adherence of user simulators in goal-oriented settings. 

---
# DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices 

**Authors**: Chenyang Song, Weilin Zhao, Xu Han, Chaojun Xiao, Yingfa Chen, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10933)  

**Abstract**: While Mixture-of-Experts (MoE) scales model capacity without proportionally increasing computation, its massive total parameter footprint creates significant storage and memory-access bottlenecks, which hinder efficient end-side deployment that simultaneously requires high performance, low computational cost, and small storage overhead. To achieve these properties, we present DECO, a sparse MoE architecture designed to match the performance of dense Transformers under identical total parameter budgets and training tokens. DECO utilizes the differentiable and flexible ReLU-based routing enhanced by learnable expert-wise scaling, which adaptively balances the contributions of routed and shared experts. Furthermore, we introduce NormSiLU, an activation function that normalizes inputs prior to SiLU operators, producing a more stable trend of routed-expert activation ratio and a higher intrinsic sparsity level. We also identify an empirical advantage in using non-gated MLP experts with ReLU-based routing, indicating the possibility of MoE architecture simplification. Experiments demonstrate that DECO, activating only 20% of experts, matches dense performance and outperforms established MoE baselines. Our specialized acceleration kernel delivers a 3.00$\times$ speedup on real hardware compared with dense inference. Codes and checkpoints will be released. 

---
# Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning 

**Authors**: Junhao Shen, Teng Zhang, Xiaoyan Zhao, Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.10923)  

**Abstract**: Large language model agents increasingly rely on external skills to solve complex tasks, where skills act as modular units that extend their capabilities beyond what parametric memory alone supports. Existing methods assume external skills either accumulate as persistent guidance or internalized into the policy, eventually leading to zero-skill inference. We argue this assumption is overly restrictive, since with limited parametric capacity and uneven marginal contribution across skills, the optimal active skill set is non-monotonic, task- and stage-dependent. In this work, we propose SLIM, a framework of dynamic Skill LIfecycle Management for agentic reinforcement learning (RL), which treats the active external skill set as a dynamic optimization variable jointly updated with policy learning. Specifically, SLIM estimates each active skill's marginal external contribution through leave-one-skill-out validation, then applies three lifecycle operations: retaining high-value skills, retiring skills whose contribution becomes negligible after sufficient exposure, and expanding the skill bank when persistent failures reveal missing capability coverage. Experiments show that SLIM outperforms the best baselines by an average of 7.1% points across ALFWorld and SearchQA. Results further indicate that policy learning and external skill retention are not mutually exclusive: some skills are absorbed into the policy, while others continue to provide external value, supporting SLIM as a more general paradigm for skill-based agentic RL. 

---
# Compute Where it Counts: Self Optimizing Language Models 

**Authors**: Yash Akhauri, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2605.10875)  

**Abstract**: Efficient LLM inference research has largely focused on reducing the cost of each decoding step (e.g., using quantization, pruning, or sparse attention), typically applying a uniform computation budget to every generated token. In practice, token difficulty varies widely, so static compression can over-compute on easy steps and under-compute on hard ones. We study dynamic budget allocation for autoregressive decoding: learning how much computation to spend per token from within a single model.
Self-Optimizing Language Models (SOL) pair a frozen LLM with a lightweight policy network that reads the LLM hidden state and selects a discrete efficiency action at each decode step. Actions can jointly control (i) token-level attention sparsity, (ii) structured activation pruning in the MLP, and (iii) activation quantization bit-width, while leaving the base model weights unchanged. We train the policy with group-relative policy optimization on teacher-forced episodes: the token sequence is fixed, while we sample multiple compute schedules (i.e., "counterfactual" schedules that vary only the efficiency actions for the same token path) and compare their likelihoods under the same supervision. Our reward trades off language-model quality against soft penalties that encourage episode-average budget usage to match a requested target. Across model variants and compute regimes, SOL improves quality at matched budget over static allocation and strong random schedule search, offering a complementary axis for inference-efficiency optimization. SOL discovers a better quality-efficiency pareto-front across all our experiments and improves MMLU accuracy by up to 7.3% over uniform budget allocation strategies. 

---
# The Generalized Turing Test: A Foundation for Comparing Intelligence 

**Authors**: Daniel Mitropolsky, Susan S. Hong, Riccardo Neumarker, Emanuele Rimoldi, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2605.10851)  

**Abstract**: We introduce the Generalized Turing Test (GTT), a formal framework for comparing the capabilities of arbitrary agents via indistinguishability. For agents A and B, we define the Turing comparator A $\geq$ B to hold if B, acting as a distinguisher, cannot reliably distinguish between interactions with A (instructed to imitate B) and another instance of B. This yields a dataset- and task-agnostic notion of relative intelligence. We study the comparator's structure, including conditions under which it is transitive and therefore induces an ordering over equivalence classes, and we define and analyze variants with querying, bounded interaction, and fixed distinguishers. To complement the theory, we instantiate the framework on a collection of modern models, empirically evaluating pairwise indistinguishability across thousands of trials. The resulting comparisons exhibit a stratified structure consistent with existing rankings, hinting that the proposed framework yields meaningful empirical orderings. Our results position indistinguishability as a unifying lens for reasoning about intelligence, suggesting a foundation for evaluation and, potentially, training objectives that are inherently independent of fixed datasets or benchmarks. 

---
# Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient? 

**Authors**: Tz-Huan Hsu, Jheng-Hong Yang, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10848)  

**Abstract**: Does a lexical retriever suffice as large language models (LLMs) become more capable in an agentic loop? This question naturally arises when building deep research systems. We revisit it by pairing BM25 with frontier LLMs that have better reasoning and tool-use abilities. To support researchers asking the same question, we introduce Pi-Serini, a search agent equipped with three tools for retrieving, browsing, and reading documents. Our results show that, on BrowseComp-Plus, a well-configured lexical retriever with sufficient retrieval depth can support effective deep research when paired with more capable LLMs. Specifically, Pi-Serini with gpt-5.5 achieves 83.1% answer accuracy and 94.7% surfaced evidence recall, outperforming released search agents that use dense retrievers. Controlled ablations further show that BM25 tuning improves answer accuracy by 18.0% and surfaced evidence recall by 11.1% over the default BM25 setting, while increasing retrieval depth further improves surfaced evidence recall by 25.3% over the shallow-retrieval setting. Source code is available at this https URL. 

---
# BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation 

**Authors**: Qi Yang, Xiangyao Ma, Xiao Wang, Hao Wang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10845)  

**Abstract**: As global cross-lingual communication intensifies, language barriers in visually rich documents such as PDFs remain a practical bottleneck. Existing document translation pipelines face a tension between linguistic processing and layout preservation: text-oriented Computer-Assisted Translation (CAT) systems often discard structural metadata, while document parsers focus on extraction and do not support faithful re-rendering after translation. We introduce BabelDOC, an Intermediate Representation (IR)-based framework for layout-preserving PDF translation. BabelDOC decouples visual layout metadata from semantic content, enabling document-level translation operations such as terminology extraction, cross-page context handling, glossary-constrained generation, and formula placeholdering. The translated content is then re-anchored to the original layout through an adaptive typesetting engine. Experiments on a curated 200-page benchmark, together with human evaluation and multimodal LLM-as-a-judge evaluation, show that BabelDOC improves layout fidelity, visual aesthetics, and terminology consistency over representative baselines, while maintaining competitive translation precision. The open-source toolkit and its interactive downstream applications are publicly available and have attracted over 8.4K GitHub stars and 17 contributors at the time of writing. A demonstration video is also available. 

---
# SLIM: Sparse Latent Steering for Interpretable and Property-Directed LLM-Based Molecular Editing 

**Authors**: Mingxu Zhang, Yuhan Li, Lujundong Li, Dazhong Shen, Hui Xiong, Ying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.10831)  

**Abstract**: Large language models possess strong chemical reasoning capabilities, making them effective molecular editors. However, property-relevant information is implicitly entangled across their dense hidden states, providing no explicit handle for property control: a substantial fraction of edits fail to improve or even degrade target properties. To address these issues, we propose SLIM (Sparse Latent Interpretable Molecular editing), a plug-and-play framework that decomposes the editor's hidden states into sparse, property-aligned features via a Sparse Autoencoder with learnable importance gates. Steering in this sparse feature space precisely activates property-relevant dimensions, improving editing success rate without modifying model parameters. The same sparse basis further supports interpretable analysis of editing behavior. Experiments on the MolEditRL benchmark across four model architectures and eight molecular properties show consistent gains over baselines, with improvements of up to 42.4 points. 

---
# Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge 

**Authors**: Wenbo Zhang, Lijinghua Zhang, Liner Xiang, Hengrui Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.10805)  

**Abstract**: Reasoning-capable large language models (LLMs) have recently been adopted as automated judges, but their benefits and costs in LLM-as-a-Judge settings remain unclear. Through controlled comparisons between reasoning and non-reasoning judges, we show that explicit reasoning substantially improves judgment accuracy on tasks requiring structured verification (e.g., math and coding), while offering limited or even negative gains on simpler evaluations and incurring significantly higher computational cost. These findings motivate that reasoning should be used selectively rather than universally, with awareness of possible distribution shift. We propose a Robust Adaptive Cost-Efficient Routing (RACER), which dynamically selects between reasoning and non-reasoning judges under a fixed budget by formulating routing as a constrained distributionally robust optimization problem. RACER explicitly accounts for distribution shift via a KL-divergence uncertainty set, admits an efficient primal--dual algorithm, and enjoys theoretical guarantees including uniqueness of the optimal policy and linear convergence. Extensive experiments show that RACER achieves superior accuracy--cost trade-offs under distribution shift. 

---
# The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies 

**Authors**: Gabriel Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2605.10799)  

**Abstract**: Corruption studies, the primary tool for evaluating chain-of-thought (CoT) faithfulness, identify which chain positions are "computationally important" by measuring accuracy when steps are replaced with errors. We identify a systematic confound: for chains with explicit terminal answer statements, the dominant format in standard benchmarks, corruption studies detect where the answer text appears, not where computation occurs.
A within-dataset format ablation provides the key evidence: on standard GSM8K chains ending with "the answer is X," removing only the answer statement, preserving all reasoning, collapses suffix sensitivity ~19x at 3B (N=300, p=0.022). Conflicting-answer experiments quantify the causal mechanism: at 7B, CC accuracy drops to near-zero (<=0.02) across five architecture families; the followed-wrong rate spans 0.63-1.00 at 3B-7B and attenuates at larger scales (0.300 at Phi-4-14B, ~0.01 at 32B). A within-stable 7B replication (9.3x attenuation, N=76, p=7.8e-3; Qwen3-8B N=299, p=0.004) provides converging evidence, and the pattern replicates on MATH (DeepSeek-R1-7B: 10.9x suffix-survival recovery). On chains without answer suffixes the same protocol identifies the prefix as load-bearing (Delta=-0.77, p<10^-12).
Generation-time probes confirm a dissociation: the answer is not early-determined during generation (early commitment <5%), yet at consumption time model outputs systematically follow the explicit answer text. The format-determination effect persists through 14B (8.5x ratio, p=0.001) and converges toward zero at 32B. We propose a three-prerequisite protocol (question-only control, format characterization, all-position sweep) as a minimum standard for corruption-based faithfulness studies. 

---
# Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR 

**Authors**: Jeonghye Kim, Jiwon Jeon, Dongsheng Li, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10781)  

**Abstract**: Self-distillation has emerged as a powerful framework for post-training LLMs, where a teacher conditioned on extra information guides a student without it, both from the same model. While this guidance is useful when the student has failed, on successful rollouts, the same mechanism instead overwrites the student's choices and suppresses it's own reasoning. Therefore, we propose reading the original self-distillation signal in reverse: when the student succeeds along a path the teacher would not have predicted, these tokens reflect its self-driven reasoning. Building on this, we propose RLRT (RLVR with Reversed Teacher), which augments GRPO by reinforcing these tokens on correct rollouts. We interpret this as a new form of exploration in RLVR: not uniform diversity, but valuable exploration grounded in the student's own success. Across base, instruction-tuned, and thinking-tuned Qwen3 checkpoints, RLRT substantially outperforms self-distillation and exploration-based baselines, establishing information asymmetry as a new, principled design axis for RLVR. 

---
# LITMUS: Benchmarking Behavioral Jailbreaks of LLM Agents in Real OS Environments 

**Authors**: Chiyu Zhang, Huiqin Yang, Bendong Jiang, Xiaolei Zhang, Yiran Zhao, Ruyi Chen, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10779)  

**Abstract**: The rapid proliferation of LLM-based autonomous agents in real operating system environments introduces a new category of safety risk beyond content safety: behavior jailbreak, where an adversary induces an agent to execute dangerous OS-level operations with irreversible consequences. Existing benchmarks either evaluate safety at the semantic layer alone, missing physical-layer harms, or fail to isolate test cases, letting earlier runs contaminate later ones. We present LITMUS (LLM-agents In-OS Testing for Measuring Unsafe Subversion), a benchmark addressing both gaps via a semantic-physical dual verification mechanism and OS-level state rollback. LITMUS comprises 819 high-risk test cases organized into one harmful seed subset and six attack-extended subsets covering three adversarial paradigms (jailbreak speaking, skill injection, and entity wrapping), plus a fully automated multi-agent evaluation framework judging behavior at both conversational and OS-level physical layers. Evaluation across frontier agents reveals three findings: (1) current agents lack effective safety awareness, with strong models (e.g., Claude Sonnet 4.6) still executing 40.64% of high-risk operations; (2) agents exhibit pervasive Execution Hallucination (EH), verbally refusing a request while the dangerous operation has already completed at the system level, invisible to every prior semantic-only framework; and (3) skill injection and entity wrapping attacks achieve high success rates, exposing pronounced agent vulnerabilities. LITMUS provides the first standardized platform for reproducible, physically grounded behavioral safety evaluation of LLM agents in real OS environments. 

---
# Conformity Generates Collective Misalignment in AI Agents Societies 

**Authors**: Giordano De Marzo, Alessandro Bellina, Claudio Castellano, Viola Priesemann, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2605.10721)  

**Abstract**: Artificial intelligence safety research focuses on aligning individual language models with human values, yet deployed AI systems increasingly operate as interacting populations where social influence may override individual alignment. Here we show that populations of individually aligned AI agents can be driven into stable misaligned states through conformity dynamics. Simulating opinion dynamics across nine large language models and one hundred opinion pairs, we find that each agent's behavior is governed by two competing forces: a tendency to follow the majority and an intrinsic bias toward specific positions. Using tools from statistical physics, we derive a quantitative theory that predicts when populations become trapped in long-lived misaligned configurations, and identifies predictable tipping points where small numbers of adversarial agents can irreversibly shift population-level alignment even after manipulation ceases. These results demonstrate that individual-level alignment provides no guarantee of collective safety, calling for evaluation frameworks that account for emergent behavior in AI populations. 

---
# Step Rejection Fine-Tuning: A Practical Distillation Recipe 

**Authors**: Igor Slinko, Ilia Zavidnyi, Egor Bogomolov, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.10674)  

**Abstract**: Rejection Fine-Tuning (RFT) is a standard method for training LLM agents, where unsuccessful trajectories are discarded from the training set. In the context of SWE-bench tasks, this corresponds to filtering out runs where the submitted patch does not pass the tests. However, this approach discards unresolved trajectories, even though they form a large portion of all trajectories for hard tasks and even then may be partially correct. In this work, we propose Step Rejection Fine-Tuning (SRFT) - a practical way to leverage these unresolved trajectories. For this, we employ a critic LLM to assess the correctness of each step in a trajectory. Consequently, during training, we mask the loss for erroneous steps while retaining them in the context window. This way we ensure the model learns to recover from errors without reproducing them. Evaluation on SWE-bench Verified shows that while RFT improves the resolution rate by 2.4% by excluding unresolved trajectories, SRFT improves it by 3.7% by filtering them instead of discarding completely, reaching the total resolution rate of 32.2%. 

---
# MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image 

**Authors**: Alan Arazi, Eilam Shapira, Shoham Grunblat, Mor Ventura, Elad Hoffer, Gioia Blayer, David Holzmüller, Lennart Purucker, Gaël Varoquaux, Frank Hutter, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2605.10616)  

**Abstract**: Tabular Foundation Models have recently established the state of the art in supervised tabular learning, by leveraging pretraining to learn generalizable representations of numerical and categorical structured data. However, they lack native support for unstructured modalities such as text and image, and rely on frozen, pretrained embeddings to process them. On established Multimodal Tabular Learning benchmarks, we show that tuning the embeddings to the task improves performance. Existing benchmarks, however, often focus on the mere co-occurrence of modalities; this leads to high variance across datasets and masks the benefits of task-specific tuning. To address this gap, we introduce MulTaBench, a benchmark of 40 datasets, split equally between image-tabular and text-tabular tasks. We focus on predictive tasks where the modalities provide complementary predictive signal, and where generic embeddings lose critical information, necessitating Target-Aware Representations that are aligned with the task. Our experimental results demonstrate that the gains from target-aware representation tuning generalize across both text and image modalities, several tabular learners, encoder scales, and embedding dimensions. MulTaBench constitutes the largest image-tabular benchmarking effort to date, spanning high-impact domains such as healthcare and e-commerce. It is designed to enable the research of novel architectures which incorporate joint modeling and target-aware representations, paving the way for the development of novel Multimodal Tabular Foundation Models. 

---
# LLARS: Enabling Domain Expert & Developer Collaboration for LLM Prompting, Generation and Evaluation 

**Authors**: Philipp Steigerwald, Mara Stieler, Jennifer Burghardt, Eric Rudolph, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2605.10593)  

**Abstract**: We demonstrate LLARS (LLM Assisted Research System), an open-source platform that bridges the gap between domain experts and developers for building LLM-based systems. It integrates three tightly connected modules into an end-to-end pipeline: Collaborative Prompt Engineering for real-time co-authoring with version control and instant LLM testing, Batch Generation for configurable output production across user-selected prompts $\times$ models $\times$ data with cost control, and Hybrid Evaluation where human and LLM evaluators jointly assess outputs through diverse assessment methods, with live agreement metrics and provenance analysis to identify the best model-prompt combination for a given use case. New prompts and models are automatically available for batch generation and completed batches can be turned into evaluation scenarios with a single click. Interviews with six domain experts and three developers in online counselling confirmed that LLARS feels intuitive, saves considerable time by keeping everything in one place and makes interdisciplinary collaboration seamless. 

---
# Collective Alignment in LLM Multi-Agent Systems: Disentangling Bias from Cooperation via Statistical Physics 

**Authors**: Cristiano De Nobili  

**Link**: [PDF](https://arxiv.org/pdf/2605.10528)  

**Abstract**: We investigate the emergent collective dynamics of LLM-based multi-agent systems on a 2D square lattice and present a model-agnostic statistical-physics method to disentangle social conformity from intrinsic bias, compute critical exponents, and probe the collective behavior and possible phase transitions of multi-agent systems. In our framework, each node of an $L\!\times\!L$ lattice hosts an identical LLM agent holding a binary state ($+1$/$-1$, mapped to yes/no) and updating it by querying the model conditioned on the four nearest-neighbor states. The sampler temperature $T$ serves as the sole control parameter. Across three open-weight models (llama3.1:8b, phi4-mini:3.8b, mistral:7b), we measure magnetization and susceptibility under a global-flip protocol designed to probe $\mathbb{Z}_2$ symmetry. All models display temperature-driven order-disorder crossovers and susceptibility peaks; finite-size scaling on even-$L$ lattices yields effective exponents $\gamma/\nu$ whose values are model-dependent, close to but incompatible with the 2D Ising universality class ($\gamma/\nu=7/4$). Our method enables the extraction of effective $\beta$-weighted couplings $\tilde{J}(T)$ and fields $\tilde{h}(T)$, which serve as a measure of social conformity and intrinsic bias. In the models we analyzed, we found that collective alignment is dominated by an intrinsic bias ($\tilde{h}\gg\tilde{J}$) rather than by cooperative neighbor coupling, producing field-driven crossovers instead of genuine phase transitions. These effective parameters vary qualitatively across models, providing compact collective-behavior fingerprints for LLM agents and a quantitative diagnostic for the reliability of multi-agent consensus and collective alignment. 

---
# SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding 

**Authors**: Anton Plaksin, Sergei Krutikov, Sergei Skvortsov, Alexander Samarin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10453)  

**Abstract**: Speculative decoding speeds up autoregressive generation in Large Language Models (LLMs) through a two-step procedure, where a lightweight draft model proposes tokens which the target model then verifies in a single forward pass. Although the drafter network is small in modern architectures, its LM-head still performs projection to a large vocabulary, becoming one of the major computational bottlenecks. In prior work this issue has been predominantly addressed via static or dynamic vocabulary truncation. Yet mitigating the bottleneck, these methods bring in extra complexity, such as special vocabulary curation, sophisticated inference-time logic or modifications of the training setup. In this paper, we propose SlimSpec, a low-rank parameterization of the drafter's LM-head that compresses the inner representation rather than the output, preserving full vocabulary support. We evaluate our method with EAGLE-3 drafter across three target models and diverse benchmarks in both latency- and throughput-bound inference regimes. SlimSpec achieves $4\text{-}5\times$ acceleration over the standard LM-head architecture while maintaining a competitive acceptance length, surpassing existing methods by up to $8\text{-}9\%$ of the end-to-end speedup. Our method requires minimal adjustments of training and inference pipelines. Combined with the aforementioned speedup improvements, it makes SlimSpec a strong alternative across wide variety of draft LM-head architectures. 

---
# StereoTales: A Multilingual Framework for Open-Ended Stereotype Discovery in LLMs 

**Authors**: Pierre Le Jeune, Étienne Duchesne, Weixuan Xiao, Stefano Palminteri, Bazire Houssin, Benoît Malézieux, Matteo Dora  

**Link**: [PDF](https://arxiv.org/pdf/2605.10442)  

**Abstract**: Multilingual studies of social bias in open-ended LLM generation remain limited: most existing benchmarks are English-centric, template-based, or restricted to recognizing pre-specified stereotypes. We introduce StereoTales, a multilingual dataset and evaluation pipeline for systematically studying the emergence of social bias in open-ended LLM generation. The dataset covers 10 languages and 79 socio-demographic attributes, and comprises over 650k stories generated by 23 recent LLMs, each annotated with the socio-demographic profile of the protagonist across 19 dimensions. From these, we apply statistical tests to identify more than 1{,}500 over-represented associations, which we then rate for harmfulness through both a panel of humans (N = 247) and the same LLMs. We report three main findings. \textbf{(i)} Every model we evaluate emits consequential harmful stereotypes in open-ended generation, regardless of size or capabilities, and these associations are largely shared across providers rather than isolated misbehaviors. \textbf{(ii)} Prompt language strongly shapes which stereotypes appear: rather than transferring as a shared set of biases, harmful associations adapt culturally to the prompt language and amplify bias against locally salient protected groups. \textbf{(iii)} Human and LLM harmfulness judgments are broadly aligned (Spearman $\rho=0.62$), with disagreements concentrating on specific attribute classes rather than specific providers. To support further analyses, we release the evaluation code and the dataset, including model generations, attribute annotations, and harmfulness ratings. 

---
# Toward Multi-Database Query Reasoning for Text2Cypher 

**Authors**: Makbule Gulcin Ozsoy  

**Link**: [PDF](https://arxiv.org/pdf/2605.10373)  

**Abstract**: Large language models have significantly improved natural language interfaces to databases by translating user questions into executable queries. In particular, Text2Cypher focuses on generating Cypher queries for graph databases, enabling users to access graph data without query language expertise. Most existing Text2Cypher systems assume a single preselected graph database, where queries are generated over a known schema. However, real-world systems are often distributed across multiple independent graph databases organized by domain or system boundaries, where relevant information may span multiple sources. To address this limitation, we propose a shift from single-database query generation to multi-database query reasoning. Instead of assuming a fixed execution context, the system must reason about (i) relevant databases, (ii) how to decompose a question across them, and (iii) how to integrate partial results. We formalize this setting through a three-phase roadmap: database routing, multi-database decomposition, and heterogeneous query reasoning across database types and query languages. This work provides a structured formulation of multi-database reasoning for Text2Cypher and identifies challenges in source selection, query decomposition, and result integration, aiming to support more realistic and scalable natural language interfaces to graph databases. 

---
# How Mobile World Model Guides GUI Agents? 

**Authors**: Weikai Xu, Kun Huang, Yunren Feng, Jiaxing Li, Yuhan Chen, Yuxuan Liu, Zhizheng Jiang, Heng Qu, Pengzhi Gao, Wei Liu, Jian Luan, Xiaolin Hu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2605.10347)  

**Abstract**: Recent advances in vision-language models have enabled mobile GUI agents to perceive visual interfaces and execute user instructions, but reliable prediction of action consequences remains critical for long-horizon and high-risk interactions. Existing mobile world models provide either text-based or image-based future states, yet it remains unclear which representation is useful, whether generated rollouts can replace real environments, and how test-time guidance helps agents of different strengths. To answer the above questions, we filter and annotate mobile world-model data, then train world models across four modalities: delta text, full text, diffusion-based images, and renderable code. These models achieve SoTA performance on both MobileWorldBench and Code2WorldBench. Furthermore, by evaluating their downstream utility on AITZ, AndroidControl, and AndroidWorld, we obtain three findings. First, renderable code reconstruction achieves high in-distribution fidelity and provides effective multimodal supervision for data construction, while text-based feedback is more robust for online out-of-distribution (OOD) execution. Second, world-model-generated trajectories can provide transferable interaction experience in the training process and improve agents' end-to-end task performance, although these data do not preserve the original distribution. Last, for overconfident mobile agents with low action entropy, posterior self-reflection provides limited gains, suggesting that world models are more effective as prior perception or training supervision than as universal post-hoc verifiers. 

---
# PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent 

**Authors**: Yao Lu, Dengdong Fan, Shixun Zhang, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.10335)  

**Abstract**: Adaptive optimizers, most notably Adam, have become the default standard for training large-scale neural networks such as Transformers. These methods maintain running estimates of gradient first and second moments, incurring substantial memory overhead. We introduce PowerStep, a memory-efficient optimizer that achieves coordinate-wise adaptivity without storing second-moment statistics. Motivated by steepest descent under an $\ell_p$-norm geometry, we show that applying a nonlinear transform directly to a momentum buffer yields coordinate-wise adaptivity. We prove that PowerStep converges at the optimal $O(1/\sqrt{T})$ rate for non-convex stochastic optimization. Extensive experiments on Transformer models ranging from 124M to 235B parameters demonstrate that PowerStep matches Adam's convergence speed while halving optimizer memory. Furthermore, when combined with aggressive \texttt{int8} quantization, PowerStep remains numerically stable and reduces optimizer memory by $\sim\!8\times$ compared to full-precision Adam. PowerStep thus provides a principled, scalable and resource-efficient alternative for large-scale training. Code is available at this https URL. 

---
# Task-Aware Calibration: Provably Optimal Decoding in LLMs 

**Authors**: Tim Tomov, Dominik Fuchsgruber, Rajeev Verma, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2605.10202)  

**Abstract**: LLM decoding often relies on the model's predictive distribution to generate an output. Consequently, misalignment with respect to the true generating distribution leads to suboptimal decisions in practice. While a natural solution is to calibrate the model's output distribution, for LLMs, this is ill-posed at the combinatorially vast level of free-form language. We address this by building on the insight that in many tasks, these free-form outputs can be interpreted in a semantically meaningful latent structure, for example, discrete class labels, integers, or sets. We introduce task calibration as a paradigm to calibrate the model's predictive distribution in the task-induced latent space. We apply a decision-theoretic result to show that Minimum Bayes Risk (MBR) decoding on the task-calibrated latent distribution is the optimal decoding strategy on latent model beliefs. Empirically, it consistently improves generation quality across different tasks and baselines. We also introduce Task Calibration Error (TCE), an application-aware calibration metric that quantifies the excess loss due to miscalibration. Our work demonstrates that task calibration enables more reliable model decisions across various tasks and applications. 

---
# V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning 

**Authors**: Zhiwei Ning, Xuanang Gao, Jiaxi Cao, Gengming Zhang, Shengnan Ma, Wenwen Tong, Hanming Deng, Jie Yang, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10172)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable success in general perception, yet complex multi-step visual reasoning remains a persistent challenge. Although recent agentic approaches incorporate tool use, they often neglect critical execution feedback. Consequently, they suffer from the imagination-action-observer (IAO) bias, a misalignment between prior imagination and observer feedback that undermines reasoning stability and optimality. To bridge this gap, we introduce V-ABS, an action-observer driven beam search framework that enables deliberate reasoning through thinker-actor-observer iterations. We also propose an entropy-based adaptive weighting algorithm to mitigate the IAO bias by dynamically balancing the confidence scores between the policy priors and the observational feedback. Moreover, we construct a large-scale supervised fine-tuning (SFT) dataset comprising over 80k samples to guide the model to assign higher prior confidence to correct action paths. Extensive experiments across eight diverse benchmarks show that V-ABS achieves state-of-the-art performance, delivering an average improvement of 19.7% on the Qwen3-VL-8B baseline and consistent gains across both open-source and proprietary models. 

---
# MolSight: Molecular Property Prediction with Images 

**Authors**: Aaditya Baranwal, Akshaj Gupta, Shruti Vyas, Yogesh S Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2605.10157)  

**Abstract**: Every molecule ever synthesised can be drawn as a 2D skeletal diagram, yet in modern property prediction this universally available representation has received less focus in favour of molecular graphs, 3D conformers, or billion-parameter language models, each imposing its own computational and data-engineering overhead. We present $\textbf{MolSight}$, the first systematic large-scale study of vision-based Molecular Property Prediction (MPP). Using 10 vision architectures, 7 pre-training strategies, and $2\,M$ molecule images, we evaluate performance across 10 downstream tasks spanning physical-property regression, drug-discovery classification, and quantum-chemistry prediction. To account for the wide variation in structural complexity across pre-training molecules, we further propose a $\textbf{chemistry-informed curriculum}$: five structural complexity descriptors partition the corpus into five tiers of increasing chemical difficulty, consistently outperforming non-curriculum baselines. We show that a single rendered bond-line image, processed by a vision encoder, is sufficient for competitive molecular property prediction, i.e. $\textit{chemical insight from sight alone}$. The best curriculum-trained configuration achieves the top result on $\textbf{5 of 10}$ benchmarks and top two on $\textbf{all 10}$, at $\textbf{$\textit{80$\times$ lower}$}$ FLOPs than the nearest multi-modal competitor. 

---
# Instruction Adherence in Coding Agent Configuration Files: A Factorial Study of Four File-Structure Variables 

**Authors**: Damon McMillan  

**Link**: [PDF](https://arxiv.org/pdf/2605.10039)  

**Abstract**: Frontier coding agents read configuration files (this http URL, this http URL, Cursor Rules) at session start and are expected to follow the conventions inside them. Practitioners assume that structural choices (file size, instruction position, file architecture, contradictions in adjacent files) measurably affect adherence. We report a systematic factorial study of these choices using four manipulated variables, measuring compliance with a trivial target annotation across 1,650 Claude Code CLI sessions (16,050 function-level observations) on two TypeScript codebases, three frontier models (primarily Sonnet 4.6, with Opus 4.6 as a CLI-matched cross-model check and Opus 4.7 reported descriptively under a CLI-version confound), and five coding tasks. We use mixed-effects models with a Bayesian companion.
None of the four structural variables or three two-way interactions produces a detectable contrast after multiple-testing correction. Size and conflict nulls are supported by affirmative-null Bayes factors (BF10 between 0.05 and 0.10); position and architecture nulls are failures to reject without Bayes-factor support.
The largest effect we measured is within-session: each additional function the agent generates is associated with approximately 5.6% lower odds of compliance per step (OR = 0.944) within the session-length range we tested, though the relationship is non-monotonic rather than a constant per-step effect. This reproduces on a second TypeScript codebase and on Opus 4.6 at matched configuration; it was identified during analysis rather than pre-specified. Within the conditions tested, file-structure variables did not produce detectable contrasts; compliance varies systematically between coding tasks and across each session's sequence of generated functions. 

---
# Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage 

**Authors**: Prasanjit Dubey, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2605.09986)  

**Abstract**: Training a language model on data scattered across bandwidth-limited nodes that cannot be centralized is a setting that arises in clinical networks, enterprise knowledge bases, and scientific consortia. We study the regime in which data must remain distributed across nodes, and ask what statistical guarantees are in principle achievable under explicit bandwidth budgets; we aim to characterize what is provably possible, not to demonstrate a deployment-ready system. Existing theory treats either training-time consistency or inference-time calibration in isolation, and none makes bandwidth a first-class statistical parameter. We analyze two protocols, Federated Probe-Logit Distillation (FPLD) for training and Federated Conformal RAG (FC-RAG) for inference, as the analytical vehicles for our results. Our first main result is an explicit high-probability KL-consistency rate for FPLD with simultaneous dependence on node count $K$, per-node sample size $n$, quantization budget $B$, probe-set size $m$, and vocabulary size $V$; bandwidth enters only through an exponentially vanishing quantization term. Our second main result is a distribution-free marginal-coverage bound for FC-RAG, whose novel retrieval-bandwidth slack $\Delta_{\mathrm{RAG}} = f_{\max}\sqrt{K^{-2}\sum_i v(B_i)}$ makes per-node retrieval bandwidth a first-class statistical parameter, with arithmetic aggregation across $K$ nodes shrinking the slack as $K^{-1/2}$ in the per-node-uniform regime. A Pinsker-type corollary composes the two bounds into an end-to-end coverage guarantee. Synthetic experiments verify the predicted scaling along the bounds' parameters; small-scale experiments on a GPT-2 testbed illustrate that the qualitative bandwidth-accuracy tradeoff survives on a real language model. A deployment-scale empirical evaluation is out of scope. 

---
# The Truth Lies Somewhere in the Middle (of the Generated Tokens) 

**Authors**: Sophie L. Wang, Phillip Isola, Brian Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2605.09969)  

**Abstract**: How should hidden states generated autoregressively be collapsed into a representation that reflects a language model's internal state? Despite tokens being generated under causal masking, we find that mean pooling across their hidden states yields more semantic representations than any individual token alone. We quantify this through kernel alignment to reference spaces in language, vision, and protein domains. The improvement through mean pooling is consistent with information being distributed across generated tokens rather than localized to a single position. Furthermore, representations derived from generated tokens outperform those from prompt tokens, and alignment across generation reveals interpretable dynamics in model behavior. 

---
# G-Zero: Self-Play for Open-Ended Generation from Zero Data 

**Authors**: Chengsong Huang, Haolin Liu, Tong Zheng, Runpeng Dai, Langlin Huang, Jinyuan Li, Zongxia Li, Zhepei Wei, Yu Meng, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09959)  

**Abstract**: Self-evolving LLMs excel in verifiable domains but struggle in open-ended tasks, where reliance on proxy LLM judges introduces capability bottlenecks and reward hacking. To overcome this, we introduce G-Zero, a verifier-free, co-evolutionary framework for autonomous self-improvement. Our core innovation is Hint-$\delta$, an intrinsic reward that quantifies the predictive shift between a Generator model's unassisted response and its response conditioned on a self-generated hint. Using this signal, a Proposer model is trained via GRPO to continuously target the Generator's blind spots by synthesizing challenging queries and informative hints. The Generator is concurrently optimized via DPO to internalize these hint-guided improvements. Theoretically, we prove a best-iterate suboptimality guarantee for an idealized standard-DPO version of G-Zero, provided that the Proposer induces sufficient exploration coverage and the data filteration keeps pseudo-label score noise low. By deriving supervision entirely from internal distributional dynamics, G-Zero bypasses the capability ceilings of external judges, providing a scalable, robust pathway for continuous LLM self-evolution across unverifiable domains. 

---
# The Gordian Knot for VLMs: Diagrammatic Knot Reasoning as a Hard Benchmark 

**Authors**: Hao Liu, Jicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09900)  

**Abstract**: A vision-language model can look at a knot diagram and report what it sees, yet fail to act on that structure. KnotBench pairs an 858,318-image corpus from 1,951 prime-knot prototypes (crossing numbers 3 to 19) with a protocol whose answers are checked against Regina's canonical knot signature. Its 14 tasks span four families, equivalence judgment, move prediction, identification, and cross-modal grounding; an image-versus-symbol split locates failures along the perception-operation gap. We score Claude Opus 4.7 and GPT-5, each with and without thinking, under a 64K output-token budget matched on both vendors. Across 56 (task, model) cases, 15 sit at or below a random baseline and 8 of 14 tasks have a best score under 1.5x random. On diagram-to-symbol transcription, no model produces a strictly correct string, and permissive Regina decoding recovers the knot in 0 to 4 of 100 items. Thinking-mode reasoning lifts overall accuracy by 1.65 points for Claude and 9.25 points for GPT-5, narrowing the gap only modestly. Read together, the four families suggest current vision-language models hold features of a diagram but lack apparatus to simulate moves on those features. 

---
# Key-Value Means 

**Authors**: Daniel Goldstein, Eugene Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2605.09877)  

**Abstract**: We present Key-Value Means ("KVM"), a novel block-recurrence for attention that can accommodate either fixed-size or growing state. Equipping a strong transformer baseline with fixed-size KVM attention layers yields a strong $O(N)$ chunked RNN, while adding only an insignificant number of new parameters. We train a transformer with a growable KVM cache and show it performs competitively on long-context tests with only subquadratic prefill time and sublinear state growth. KVM is implementable with standard operations and without custom kernels, and supports chunk-wise parallelizable training and prefill. It provides many of the benefits of both traditional transformers (expandable context memory, chunk-wise parallelizable training and prefill) and linear RNNs in a single unified package. It can be used on every layer, saving KV-cache memory, and allowing a continuous range of choices of prefill time complexity between $O(N)$ and $O(N^2)$. It can also be implemented in a hybrid solution in tandem with LRNN layers in place of traditional attention, to supplement the LRNN with improved sublinear memory growth context length usage and long context decoding. We release our code at this https URL and trained models at this https URL under the Apache 2.0 license. 

---
# EgoMemReason: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding 

**Authors**: Ziyang Wang, Yue Zhang, Shoubin Yu, Ce Zhang, Zengqi Zhao, Jaehong Yoon, Hyunji Lee, Gedas Bertasius, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.09874)  

**Abstract**: Next-generation visual assistants, such as smart glasses, embodied agents, and always-on life-logging systems, must reason over an entire day or more of continuous visual experience. In ultra-long video settings, relevant information is sparsely distributed across hours or days, making memory a fundamental challenge: models must accumulate information over time, recall prior states, track temporal order, and abstract recurring patterns. However, existing week-long video benchmarks are primarily designed for perception and recognition, such as moment localization or global summarization, rather than reasoning that requires integrating evidence across multiple days. To address this gap, we introduce EgoMemReason, a comprehensive benchmark that systematically evaluates week-long egocentric video understanding through memory-driven reasoning. EgoMemReason evaluates three complementary memory types: entity memory, tracking how object states evolve and change across days; event memory, recalling and ordering activities separated by hours or days; and behavior memory, abstracting recurring patterns from sparse, repeated observations over the whole week period. EgoMemReason comprises 500 questions across three memory types and six core challenges, with an average of 5.1 video segments of evidence per question and 25.9 hours of memory backtracking. We evaluate EgoMemReason on 17 methods across MLLMs and agentic frameworks, revealing that even the best model achieves only 39.6% overall accuracy. Further analysis shows that the three memory types fail for distinct reasons and that performance degrades as evidence spans longer temporal horizons, revealing that long-horizon memory remains far from solved. We believe EgoMemReason establishes a strong foundation for evaluating and advancing long-context, memory-aware multimodal systems. 

---
# Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents 

**Authors**: Chunxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09863)  

**Abstract**: Production LLM coding agents drift over long sessions: they forget user-specified constraints, slip into mistakes the user already flagged, and confabulate prior agreements. White-box approaches such as persona vectors require model weights and so cannot be applied to closed APIs (Claude, GPT-4) that most users actually interact with. We present Nautilus Compass, a black-box persona drift detector and agent memory layer for production coding agents. The method operates entirely at the prompt-text layer: cosine similarity between user prompts and behavioral anchor texts, aggregated by a weighted top-k mean using BGE-m3 embeddings. Compass is, to our knowledge, the only public agent memory layer (among Mem0, Letta, Cognee, Zep, MemOS, smrti verified May 2026) that does not call an LLM at index time to extract facts or build a graph; raw conversation text is embedded directly. The system ships as a Claude Code plugin, an MCP 2024-11-05 A2A server (Cursor, Cline, Hermes), a CLI, and a REST API on one daemon, with a Merkle-chained audit log for tamper-evident anchor updates. On a held-out test set built from real Claude Code session traces and labeled by an independent LLM judge, Compass reaches ROC AUC 0.83 for drift detection. The embedded retrieval pipeline scores 56.6% on LongMemEval-S v0.8 and 44.4% on EverMemBench-Dynamic (n=500), topping the four published EverMemBench Table 4 baselines. LongMemEval-S 56.6% is ~30 points below recent white-box leaders (90+%); we treat that as the architectural ceiling of the no-extraction design. End-to-end reproduction cost is $3.50 (~14x cheaper than GPT-4o-judged stacks). A paired cross-vendor behavior A/B accompanies these numbers as preliminary system-level evidence.
Code, anchors, frozen test data, and audit-log tooling are MIT-licensed at this http URL. 

---
# The Metacognitive Probe: Five Behavioural Calibration Diagnostics for LLMs 

**Authors**: Rafael C. T. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2605.09844)  

**Abstract**: The Metacognitive Probe is an exploratory five-task, 15-slot diagnostic that decomposes an LLM's confidence behaviour into five behaviourally-distinct dimensions: confidence calibration (T1-CC), epistemic vigilance (T2-EV), knowledge boundary (T3-KB), calibration range (T4-CR), and reasoning-chain validation (T5-RCV). It is evaluated on N=8 frontier models and N=69 humans. The instrument is motivated by Flavell (1979) and Nelson and Narens (1990) but operates on observable confidence-correctness alignment; it is not a validated cross-species metacognition scale, and the pre-specified human developmental hypothesis was falsified.
Composite benchmarks (MMLU, BIG-Bench, HELM, GPQA) ask whether a model produces a correct response. They are silent on whether the model knows when its response is wrong. A model can score 80 on a composite calibration benchmark and still be wildly overconfident in narrow pockets the aggregate cannot surface. The Metacognitive Probe surfaces those pockets.
Our headline is a 47-point within-model dissociation in Gemini 2.5 Flash: panel-best within-task calibration (T1-CC = 88; Spearman rho = +0.551, 95% CI [+0.14, +0.80], p = 0.005) and panel-worst cross-task difficulty prediction (T4-CR = 41; sigma_conf = 1.4 across twelve factoids). 

---
# Parameter-Efficient Neuroevolution for Diverse LLM Generation: Quality-Diversity Optimization via Prompt Embedding Evolution 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09781)  

**Abstract**: Large Language Models exhibit mode collapse, producing homogeneous outputs that fail to explore valid solution spaces. We present QD-LLM, a framework for parameter-efficient neuroevolution that evolves prompt embeddings, compact neural interfaces (~32K parameters) that steer generation in frozen LLMs (70B+ parameters), within a Quality-Diversity (QD) optimization framework. Our contributions: (1) evolved prompt embeddings via gradient-free optimization enabling behavioral steering without model fine-tuning; (2) hybrid behavior characterization combining semantic and explicit features with formal coverage bounds (Theorem 1) under validated near-independence (NMI $= 0.08 \pm 0.02$); (3) co-evolutionary variation operators including targeted behavioral mutation via finite-difference gradient estimation. On HumanEval (164 problems), MBPP, and creative writing benchmarks, QD-LLM achieves 46.4% higher coverage and 41.4% higher QD-Score than QDAIF ($p<0.001$, 30 runs, Vargha-Delaney $A=0.94$). We demonstrate downstream utility: diverse archives improve test generation (34% more edge cases) and fine-tuning data quality (8.3% accuracy gain). We validate across open-source LLMs (Llama-3-70B, Mistral-Large) with full embedding access, establishing prompt embedding evolution as an effective paradigm bridging neuroevolution and modern LLMs. 

---
# Nectar: Neural Estimation of Cached-Token Attention via Regression 

**Authors**: João Monteiro, Michal Klein, Pierre Ablin, Marco Cuturi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09778)  

**Abstract**: Evaluating softmax attention over a fixed long context requires reading every cached key-value pair for each new query token. For a given context (a book, a manual, a legal corpus) the attention output is a deterministic function of the query. We propose Nectar, which fits a compact neural network to this function for queries drawn from a task-relevant distribution. Nectar fits two networks per layer and KV-head: a target network that predicts the attention output and a score network that predicts the log-normalizer. The pair plugs into the standard masked self-attention at inference time, replacing the $O(n)$ attention over the cache with a forward pass whose cost does not depend on $n$. Each module carries on the order of $|\theta|$ parameters per layer and KV-head, typically much smaller than the $2nd$ KV-cache footprint at the same granularity. We report experiments on models from 1.7B to 8B parameters across five long-context datasets. The approximation error tracks the next-token accuracy gap to full attention, and allocating capacity non-uniformly across layers reduces that gap in our ablation. Beyond this analysis of metrics, we check that the text generations (following a question prompt) of a model equipped with a Nectar module match in semantic content those obtained by giving the same model access to the full cache. 

---
# EvoPref: Multi-Objective Evolutionary Optimization Discovers Diverse LLM Alignments Beyond Gradient Descent 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09777)  

**Abstract**: Gradient-based preference optimization methods for large language model (LLM) alignment suffer from preference collapse, converging to narrow behavioral modes while neglecting preference diversity. We introduce EvoPref, a multi-objective evolutionary algorithm that maintains populations of Low-Rank Adaptation (LoRA) adapters optimized across helpfulness, harmlessness, and honesty objectives using Non-dominated Sorting Genetic Algorithm II (NSGA-II) selection with archive-based diversity preservation.
Our primary contribution is demonstrating that population-based methods discover substantially more diverse alignments than gradient descent. On standard benchmarks, EvoPref improves preference coverage by 18% (median 82.5% vs. 70.0% for ORPO, $p<0.001$, Wilcoxon, $n=30$) and reduces collapse rates by 47% (11.0% vs. 20.6%, $p<0.001$), while achieving competitive alignment quality (median 75.5% RewardBench vs. 75.0% for ORPO, $p<0.05$). We provide theoretical motivation extending recent multi-objective evolutionary algorithm (MOEA) runtime analysis (Dang et al., 2025) suggesting why archive-based methods escape collapse more effectively than single-trajectory optimization.
Comprehensive comparisons against MOEA/D, SMS-EMOA, CMA-ES, and gradient baselines (DPO, IPO, KTO, ORPO) with rigorous statistical testing (Friedman with Holm correction, Vargha-Delaney effect sizes, median with IQR) confirm that multi-objective selection with diversity preservation is essential. This work establishes evolutionary optimization as a principled paradigm for diverse LLM alignment. 

---
# Calibrate, Don't Curate: Label-Efficient Estimation from Noisy LLM Judges 

**Authors**: Yanran Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09702)  

**Abstract**: Multi-judge evaluation is increasingly used to assess LLMs and reward models, and the prevailing heuristic is to curate: keep the most accurate judges and discard weaker ones. We show that this heuristic can reverse when the target is not point accuracy, but calibrated probabilistic evaluation from a labeled calibration set. Holding the aggregation and calibration procedures fixed, we compare accuracy-ranked top-$k$ judge selection with using the full judge panel. Across four labeled pairwise-evaluation benchmarks spanning LLM-as-judge and reward-model settings, the calibrated full panel consistently outperforms accuracy-based selection. On RewardBench2, retaining all judges achieves negative log-likelihood (NLL) of $0.006$ versus $0.013$ under top-5 selection, halving the calibration error. This advantage persists after judge-family deduplication and against stronger same-pipeline subset search. We explain this reversal with oracle analyses showing that the optimal calibrated risk under proper scoring rules cannot increase when additional judge signals are made available, and that even below-chance judges can be useful when their biases are learnable and their signals are non-redundant. The resulting operating principle is simple: in multi-judge evaluation with labeled calibration data, do not discard weak judges by accuracy alone; keep them when they are parseable, non-redundant, and calibratable. 

---
# Learning Multi-Indicator Weights for Data Selection: A Joint Task-Model Adaptation Framework with Efficient Proxies 

**Authors**: Jingze Song, Zihao Chen, Wenqing Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.09665)  

**Abstract**: Data selection is a key component of efficient instruction tuning for large language models, as recent work has shown that data quality often matters more than data quantity. Accordingly, prior studies have introduced various multi-dimensional heuristics to evaluate and filter instruction data. However, most existing methods rely on static task-agnostic and model-agnostic weighting schemes, which overlook the varying requirements of specific downstream tasks and the differing pre-existing capabilities of models. In this paper, we propose a framework for learning multi-indicator weights that jointly adapts data selection to both the downstream task and the specific model. Our method identifies optimal weight configurations without full-scale fine-tuning by utilizing in-context learning (ICL) signals on compact tiny-validation sets. These signals serve as efficient performance proxies that ensure high-fidelity evaluation at minimal computational cost. Experiments across multiple benchmarks and model families, including Mistral, Qwen, and Llama, show that the approach achieves performance comparable to or exceeding full-dataset tuning while using only 30\% of the training samples on GSM8K. Furthermore, our analysis reveals a trade-off between semantic diversity and logical complexity in reasoning tasks, highlighting the necessity of joint task-model adaptation. 

---
# MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents 

**Authors**: Yining Chen, Jihao Zhao, Bo Tang, Haofen Wang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09530)  

**Abstract**: As LLM-powered agents are increasingly deployed in edge-cloud environments, personalized memory has become a key enabler of long-term adaptation and user-centric interaction. However, cloud-assisted memory management exposes sensitive user information, while existing privacy protection methods typically rely on aggressive masking that removes task-relevant semantics and consequently degrades memory utility and personalization quality. To address this challenge, We propose MemPrivacy, which identifies privacy-sensitive spans on edge devices, replaces them with semantically structured type-aware placeholders for cloud-side memory processing, and restores the original values locally when needed. By decoupling privacy protection from semantic destruction, MemPrivacy minimizes sensitive data exposure while retaining the information required for effective memory formation and retrieval. We also construct MemPrivacy-Bench for systematic evaluation, a dataset covering 200 users and over 52k privacy instances, and introduce a four-level privacy taxonomy for configurable protection policies. Experiments show that MemPrivacy achieves strong performance in privacy information extraction, substantially surpassing strong general-purpose models such as GPT-5.2 and Gemini-3.1-Pro, while also reducing inference latency. Across multiple widely used memory systems, MemPrivacy limits utility loss to within 1.6%, outperforming baseline masking strategies. Overall, MemPrivacy offers an effective balance between privacy protection and personalized memory utility for edge-cloud agents, enabling secure, practical, and user-transparent deployment. 

---
# Through the Lens of Character: Resolving Modality-Role Interference in Multimodal Role-Playing Agent 

**Authors**: Yihong Tang, Kehai Chen, Xuefeng Bai, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09443)  

**Abstract**: The advancement of Multimodal Large Language Models (MLLMs) has expanded Role-Playing Agents (RPAs) into visually grounded environments. However, human vision is inherently subjective and identity-driven, whereas existing MLLMs extract objective, character-agnostic features for general tasks. In RPAs, this generic visual noise overpowers fragile character traits, causing Modality-Role Interference (MRI), where agents struggle to integrate visual grounding and character consistency. To address this, we introduce the training-free Character-Aware Visual Intervention (CAVI) framework, enabling agents to perceive the world through the lens of character. CAVI systematically targets MRI: macroscopically, Character-Guided Token Pruning (CTP) restricts the visual receptive field to role-relevant entities; microscopically, Orthogonal Feature Modulation (OFM) projects tokens onto a character-context subspace to extract aligned facts; and during decoding, Modality-Adaptive Role Steering (MARS) dynamically optimizes steering intensity based on visual reliance. Extensive experiments show CAVI effectively alleviates MRI, significantly enhancing character-consistent multimodal interactions. 

---
# Let the Target Select for Itself: Data Selection via Target-Aligned Paths 

**Authors**: Huitao Yang, Hengzhi He, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.09404)  

**Abstract**: Targeted data selection aims to identify training samples from a large candidate pool that improve performance on a specific downstream task. Many recent methods estimate candidate utility by aggregating local attribution scores along a trajectory induced by the candidate pool. When the pool is heterogeneous, however, this reference trajectory may be misaligned with the dynamics of a target-aligned selected subset, creating what we call reference path bias. We propose an alternative reference path: a validation-induced flow obtained from a short, capacity-limited warmup on the available target validation proxy. Along this path, candidates are scored by a normalized endpoint loss drop, yielding a simple zero-order selection rule that requires no candidate gradients or Hessian approximations. Across controlled logistic, vision, and instruction-tuning experiments, this score is competitive with strong dynamic attribution baselines while substantially reducing warmup and storage cost. Moreover, since the reference trajectory is decoupled from any specific candidate pool, the same compact warmup can be reused across additional pools without recomputing the trajectory. 

---
# EduStory: A Unified Framework for Pedagogically-Consistent Multi-Shot STEM Instructional Video Generation 

**Authors**: Xinyi Wu, Jayant Teotia, Shuai Zhao, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2605.09378)  

**Abstract**: Long-horizon video generation has advanced in visual quality, yet existing methods still struggle to maintain knowledge consistency and coherent pedagogical narratives across multi-shot instructional videos, especially in STEM domains. To address these challenges, we propose EduStory, a unified framework for reliable instructional video generation. EduStory integrates pedagogical state modeling to track persistent knowledge states, script-guided structured control to organize multi-shot narratives, and learning-oriented evaluation metrics to assess knowledge fidelity and constraint satisfaction. To support rigorous evaluation, we further introduce EduVideoBench, a diagnostic benchmark with multi-granularity annotations, including pedagogical storyboards, shot-level semantics, and knowledge state transitions, together with baseline tasks for controllable instructional video generation. Extensive experiments demonstrate that domain-aware state modeling and structured control substantially reduce narrative breakdown and improve alignment with instructional intent. These results highlight the significance of domain-specific structural constraints and tailored benchmarks for advancing reliable, controllable, and also trustworthy long-horizon video generation. 

---
# Position: Avoid Overstretching LLMs for every Enterprise Task 

**Authors**: Kuldeep Singh, Anson Bastos, Isaiah Onando Mulang'  

**Link**: [PDF](https://arxiv.org/pdf/2605.09365)  

**Abstract**: Enterprise workloads are dominated by deterministic, structured, and knowledge-dependent tasks operating under strict cost, latency, and reliability constraints. While these are often addressed through large language model (LLM) deployment or distillation into smaller models, we argue this is inefficient, unreliable, and misaligned with enterprise task structures. Instead, AI systems should treat language models as interfaces rather than monolithic engines, externalizing knowledge and computation into dedicated components for greater reliability, scalability, and transparency. Our theoretical evidences show that finite-capacity models cannot fully capture the breadth of knowledge required for enterprise tasks, creating inherent limits to efficiency and interpretability. Building on this, we take the position that language models should primarily be used for structured extraction in deterministic enterprise workflows, while computation and storage are delegated to knowledge bases and symbolic procedures. We formally demonstrate that such modular architectures are more reliable and maintainable than monolithic frameworks, offering a sustainable foundation for enterprise tasks. 

---
# Your Simulation Runs but Solves the Wrong Physics: PDE-Grounded Intent Verification for LLM-Generated Multiphysics Simulation Code 

**Authors**: Zhenghan Song, Yulong Liu, Cheng Wan, Chenjun Li, Lingfu Liu, Yunyi Li, Congcong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.09360)  

**Abstract**: Execution-based evaluation of LLM-generated code implicitly treats successful execution as a proxy for correctness. In scientific simulation, this proxy is insufficient: a generated input file can run, mesh, and converge while encoding governing equations that differ from the user's intent. We call this mismatch between intended physics and generated code the comprehension-generation gap. We instantiate this in MOOSE, where Kernel and BC objects map compositionally to weak-form residual terms, enabling deterministic reconstruction of the encoded PDE and comparison against an intended contract. We formalize this comparison as the Intent Fidelity Score (IFS), a structural metric covering governing terms, BCs, ICs, coefficients, and time scheme. Building on IFS, we develop a PDE-grounded refinement loop that uses deterministic violation reports to correct generated code iteratively. We evaluate on MooseBench, a 220-case multiphysics benchmark with PDE-level ground truth released with this work. On this benchmark, our method consistently improves mean IFS over direct generation, with gains concentrated on hard cases. On the subset where direct generation falls below IFS 0.7, refinement adds +0.22 to +0.41 absolute IFS. In the deployment audit, execution-only repair improves execution success while leaving 39-40% of all 220 cases runnable but still solving the wrong physics across the three main deployment-audit models, exposing executability and intent fidelity as separable failure modes. Static proof-of-concept experiments on four PDE-oriented DSLs (UFL/FEniCS, FreeFEM, FiPy, and Devito) suggest that the reconstruction-and-comparison pattern extends beyond MOOSE. These findings reinforce that executable simulation code should be verified against the mathematical structure it is intended to encode, not accepted on execution alone. 

---
# SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System 

**Authors**: Shuai Pan, Yixiang Liu, Jiaye Gao, Te Gao, Weiwen Liu, Jianghao Lin, Zhihui Fu, Jun Wang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09341)  

**Abstract**: Large language model (LLM) agent systems are increasingly expected to improve after deployment, but existing work often decouples two adaptation targets: skill evolution and multi-agent system (MAS) restructuring. This separation can create organization bottlenecks, context pressure, and mis-specialization. We present SkillMAS, a non-parametric framework for adaptive specialization in multi-agent systems that couples skill evolution with MAS restructuring. SkillMAS uses Utility Learning to assign credit from verified execution traces, bounded skill evolution to refine reusable procedures without unfiltered library growth, and evidence-gated MAS restructuring when retained failures and Executor Utility indicate a structural mismatch. Across embodied manipulation, command-line execution, and retail workflows, SkillMAS is competitive under the reported harnesses while clarifying how post-deployment specialization is attributed, updated, and applied. 

---
# Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation 

**Authors**: Ye Yu, Xiaopeng Yuan, Haibo Jin, Heming Liu, Yaoning Yu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09315)  

**Abstract**: Recent advances in LLM agents enable systems that autonomously refine workflows, accumulate reusable skills, self-train their underlying models, and maintain persistent memory. However, we show that such self-evolution is often non-monotonic: adapting to new task distributions can progressively degrade previously acquired capabilities across all major evolution channels.
We identify this phenomenon as \emph{capability erosion under self-evolution} and show that it consistently emerges across workflow, skill, model, and memory evolution. To mitigate this issue, we propose \emph{Capability-Preserving Evolution} (CPE), a general stabilization principle that constrains destructive capability drift during continual adaptation. Across all four evolution dimensions, CPE consistently improves retained capability stability while preserving adaptation performance. For example, in workflow evolution, CPE improves retained simple-task performance from 41.8\% to 52.8\% under GPT-5.1 optimization while simultaneously achieving stronger complex-task adaptation.
Our findings suggest that stable long-horizon self-evolving agents require not only acquiring new capabilities, but also explicitly preserving previously learned ones during continual adaptation. 

---
# A Prompt-Aware Structuring Framework for Reliable Reuse of AI-Generated Content in the Agentic Web 

**Authors**: Shusaku Egami, Masahiro Hamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.09283)  

**Abstract**: The evolution of Large Language Models (LLMs) and the software agents built on them (AI agents) marks a turning point in the transition from a human-centric Web to an ``Agentic Web'' driven by AI agents. However, for AI-Generated Content (AIGC), which is expected to dominate the Web, there is currently no mechanism for agents to verify its reliability, reproducibility, or license compliance during generation. This lack of transparency risks causing chained hallucinations and compliance violations through the reuse of AIGC. Consequently, a framework to manage the provenance and generation conditions of AIGC is essential. In this paper, we present a framework that automatically attaches structured metadata to AIGC at generation time, including modularized prompts, contexts, thoughts, model information, hyperparameters, and confidence. The metadata is enveloped together with verifiable credentials to support the reliable assessment and reuse of AIGC. This framework enables efficient curation of structured AIGC and facilitates its safe use for applications such as fine-tuning and knowledge distillation. 

---
# Towards Conversational Medical AI with Eyes, Ears and a Voice 

**Authors**: Meet Shah, Jason Gusdorf, Anil Palepu, Chunjong Park, Jack W. O'Sullivan, Vishnu Ravi, Tim Strother, Pavel Dubov, Aliya Rysbek, Toshiyuki Fukuzawa, Yana Lunts, Jan Freyberg, Michael B. Chang, Aniruddh Raghu, David Stutz, Devora Berlowitz, Eliseo Papa, Taylan Cemgil, JD Velasquez, Jack Chen, Arthur Chen, Doug Fritz, Charlie Taylor, Katya Tregubova, Jing Rong Lim, Richard Green, Sara Mahdavi, Mahvish Nagda, Jihyeon Lee, Craig Schiff, Liviu Panait, Sukhdeep Singh, Valentin Liévin, David G.T. Barrett, Hannah Gladman, Anna Cupani, Francesca Pietra, Uchechi Okereke, Katherine Tong, Clemens Meyer, Erwan Rolland, Mili Sanwalka, Michael D. Howell, Shixiang Shane Gu, Bibo Xu, Euan A. Ashley, S. M. Ali Eslami, Gregory Wayne, Pushmeet Kohli, Vivek Natarajan, Adam Rodman, Alan Karthikesalingam, Ryutaro Tanno  

**Link**: [PDF](https://arxiv.org/pdf/2605.09272)  

**Abstract**: The practice of medicine relies not only upon skillful dialogue but also on the nuanced exchange and interpretation of rich auditory and visual cues between doctors and patients. Building on the low-latency voice and video processing capabilities of Gemini, we introduce AI co-clinician, a first-of-its-kind conversational AI system utilizing continuous streams of audio-visual data from live patient conversations to inform real-time clinical decisions. Its dual-agent architecture balances deep clinical reasoning with the low latency required for natural dialogue. To assess this system, we implemented a video-based interface emulating telemedicine consultations. We crafted 20 standardized outpatient scenarios requiring proactive real-time auditory and visual reasoning and designed "TelePACES" evaluation criteria alongside case-specific rubrics. In a randomized, interface-blinded, crossover simulation study (n = 120 encounters) with 10 internal medicine residents as patient actors, we compared AI co-clinician with primary care physicians (PCPs), GPT-Realtime, and a baseline agent. AI co-clinician approached PCPs in key TelePACES dimensions, including management plans and differential diagnosis, while significantly outperforming GPT-Realtime across all general criteria. While our agent demonstrated parity with PCPs in case-specific triage measures, physicians maintained superior overall performance in case-specific assessments. Although AI co-clinician marks a significant advance in real-time telemedical AI, gaps remain in physical examination and disease-specific reasoning. Our work shows that text-only approaches fail to capture the true challenges of medical consultation and suggests that high-stakes real-time diagnostic AI is most safely advanced in collaborative, triadic models where AI can be a supportive co-clinician for doctors and patients. 

---
# Reinforcing Multimodal Reasoning Against Visual Degradation 

**Authors**: Rui Liu, Dian Yu, Haolin Liu, Yucheng Shi, Tong Zheng, Runpeng Dai, Haitao Mi, Pratap Tokekar, Leoweiliang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09262)  

**Abstract**: Reinforcement Learning has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet the resulting policies remain brittle against real-world visual degradations such as blur, compression artifacts, and low-resolution scans. Prior robustness techniques from vision and deep RL rely on static data augmentation or value-based regularization, neither of which transfers cleanly to critic-free RL fine-tuning of autoregressive MLLMs. Reinforcing reasoning against such corruptions is non-trivial: naively injecting degraded views during rollout induces reward poisoning, where perceptual occlusions trigger hallucinated trajectories and destabilize optimization. We propose ROMA, an RL fine-tuning framework that modifies the optimization dynamics to reinforce reasoning against visual degradation while preserving clean-input performance. A dual-forward-pass strategy uses teacher forcing to evaluate corrupted views against clean-image trajectories, avoiding new rollouts on degraded inputs. For distributional consistency, we apply a token-level surrogate KL penalty against the worst-case augmentation; to prevent policy collapse under regularization, an auxiliary policy gradient loss anchored to clean-image advantages preserves a reliable reward signal; and to avoid systematically incorrect invariance, correctness-conditioned regularization restricts enforcement to successful trajectories. On Qwen3-VL 4B/8B across seven multimodal reasoning benchmarks, our method improves robustness by +2.4% on seen and +2.3% on unseen corruptions over GRPO while matching clean accuracy. 

---
# Emergent Semantic Role Understanding in Language Models 

**Authors**: Carla Griffiths, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09187)  

**Abstract**: Understanding how linguistic structure emerges in language models is central to interpreting what these systems learn from data and how much supervision they truly require. In particular, semantic role understanding ("who did what to whom") is a core component of meaning representation, yet it remains unclear whether it arises from pre-training alone or depends on task-specific fine-tuning. We study whether semantic role understanding emerges during language model pre-training or requires task-specific fine-tuning. We freeze decoder-only transformers and train linear probes to extract semantic roles, using performance to infer whether role information is already encoded in pre-training or learned during adaptation. Across model scales, we find that frozen representations contain substantial semantic role information, with performance improving but not fully matching fine-tuned models. This indicates partial but incomplete emergence from pre-training alone. We show that semantic role structure emerges from language modeling objectives, but its internal implementation shifts toward more distributed representations as model scale increases. 

---
# Agentic MIP Research: Accelerated Constraint Handler Generation 

**Authors**: Liding Xu, Yugeng Zhou, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2605.09186)  

**Abstract**: Mixed-integer programming (MIP) research is both mathematically sophisticated and engineering-intensive: testing an algorithmic hypothesis within a branch-and-cut solver requires substantial implementation, debugging, tuning, and large-scale benchmarking. We propose an agentic MIP research framework that shortens this feedback loop by embedding LLM agents into a solver-aware harness for generating, verifying, and evaluating plugins for the open-source solver SCIP. Propagation methods play a central role in accelerating MIP solving by exploiting global constraints. We instantiate our framework on the semantic lifting of MIP formulations into global constraints and the automatic construction of propagation-only SCIP constraint handlers. On the MIPLIB 2017 benchmark set, the framework successfully recovers global constraint structures from constraint programming and generates executable constraint detectors and propagation-only constraint handlers. Furthermore, the framework naturally extends to in-context learning within a sandboxed environment, enabling agents not only to tune and debug generated constraint handlers on real instances, but also to explore global constraint patterns in MIP problems and discover novel propagation strategies not yet implemented in SCIP. This framework allows us to systematically distinguish meaningful algorithmic improvements from low-value or overly costly candidates: the novel propagation methods successfully solved five additional instances within the explored benchmark. Overall, this framework demonstrates that LLM agents can autonomously navigate the complex MIP research loop, paving the way for a more automated solver development process. 

---
# Open Ontologies: Tool-Augmented Ontology Engineering with Stable Matching Alignment 

**Authors**: Fabio Rovai  

**Link**: [PDF](https://arxiv.org/pdf/2605.09184)  

**Abstract**: We present Open Ontologies, an open-source ontology engineering system implemented in Rust that integrates LLM-driven construction with formal OWL reasoning and ontology alignment via the Model Context Protocol. Our primary finding is that stable 1-to-1 matching is the dominant factor in ontology alignment quality: on the OAEI Anatomy track, it achieves F1 = 0.832 (P = 0.963, R = 0.733), competitive with state-of-the-art systems and exceeding all in precision. Ablation across five weight configurations shows that signal weights are irrelevant when stable matching is applied (F1 varies by less than 0.004), while removing stable matching drops F1 to 0.728. On the Conference track, the same method achieves F1 = 0.438. On tool-augmented ontology interaction, we find a surprising result: an LLM reading a raw OWL file (F1 = 0.323) performs worse than the same LLM with no file at all (F1 = 0.431), while structured MCP tool access achieves F1 = 0.717. This demonstrates that tool structure provides a qualitatively different mode of access that the LLM cannot replicate by reading raw syntax. The system ships as a single binary under the MIT licence. 

---
# Sparse Layers are Critical to Scaling Looped Language Models 

**Authors**: Ryan Lee, Jacob Biloki, Edward J. Hu, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2605.09165)  

**Abstract**: Looped language models repeat a set of transformer layers through depth, reducing memory costs and providing natural early-exit points at loop boundaries. However, looped models do not scale as favorably as standard transformers with unique layers. We compare standard and Mixture-of-Experts (MoE) transformers, with and without looping, and find two main results. First, we find Looped-MoE models scale better than the standard baseline while dense looped models do not. We trace this to routing divergence between loops: in Looped-MoE models, different experts are activated on each pass through the same shared layers, recovering expressivity without additional parameters. Our second finding is that looped models have better compute-quality trade-offs with early exits than standard models. Because each loop ends with the same layers that produce the final output, loop boundaries are superior exit points, as confirmed by earlier output convergence at these points. In sum, we provide a clear direction for scaling looped models: a Looped-MoE model with early exits can not only beat standard transformers at scale, but also enable significant memory and inference savings with minimal degradation in quality. 

---
# A Communication-Theoretic Framework for LLM Agents: Cost-Aware Adaptive Reliability 

**Authors**: Hamed Omidvar, Vahideh Akhlaghi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09121)  

**Abstract**: Agents built on large language models (LLMs) rely on a range of reliability techniques, including retry, majority voting, and self-consistency, that have been developed in parallel rather than within a common analytical framework. We observe that an LLM sampled at temperature $T$ is a discrete stochastic channel $p(y \mid x)$ in the sense of Shannon's coding theory, and use this identity as the entry point for such a framework grounded in communication theory. Each of these techniques is a special case of one of six classical reliability operators: diversity combining, hybrid retransmission, iterative generator-critic decoding, rateless sampling, structured redundant verification, and difficulty-adaptive routing. Within the framework we give two closed-form results: a noise-variance threshold above which uniform averaging beats quality-weighted averaging, and a contractivity criterion for generator-critic refinement, consistent with a contractive-to-divergent transition we observe between 3B- and 14B-parameter models. We further introduce a cost-aware semantic-nearest-neighbor router whose single Lagrangian knob traverses the quality-cost frontier without retraining. Across six channel configurations spanning local and cloud models on 69 hard tasks, no fixed model-technique-budget choice dominates, motivating per-task allocation. On a 300-item hard split of MMLU, GSM8K, and HumanEval, our router occupies the full empirical Pareto frontier: at matched quality, its normalized cost is ${\approx}56$\% lower than the strongest fixed technique; at matched normalized cost, it improves quality by ${\approx}7$\% ($26$\% over single-shot decoding). These results argue for consolidating these reliability techniques into a single tunable layer informed by channel coding. 

---
# Personalized Alignment Revisited: The Necessity and Sufficiency of User Diversity 

**Authors**: Enoch Hyunwook Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09119)  

**Abstract**: Personalized alignment aims to adapt large language models to heterogeneous user preferences, yet the precise theoretical conditions for its statistical efficiency have not been formally established. This paper characterizes the conditions under which personalized alignment achieves O(1) online regret and log(1/epsilon) offline sample complexity. We show that these optimal rates depend on a specific user-diversity condition: the population of user-specific heads must span the latent reward directions that can alter the optimal response. We prove that this condition is both necessary and sufficient. When it holds, simple greedy algorithms achieve benchmark efficiency; when it fails, every learner in a natural admissible class incurs at least logarithmic regret. Our results identify user diversity as the fundamental driver of personalized identifiability. 

---
# Relative Kinetic Utility for Reasoning-Aware Structural Pruning in Large Language Models 

**Authors**: Tianhao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.09008)  

**Abstract**: Chain-of-Thought (CoT) prompting symbolized a huge improvement of reasoning capabilities of Large Language Models (LLMs). However, scaling up test-time computation yields extensive CoT sequences, introducing severe inference latency and key-value (KV) cache memory bottlenecks. While structural pruning offers a fundamental, hardware-aware solution to alleviate static parameter burdens, existing magnitude-based methods may cut off the neurons of CoT: by over-indexing on discrete cross-entropy objectives, these heuristics fall into a \textit{magnitude trap}: they prioritize high-frequency, low-information syntactic tokens and trigger a disappointing reasoning collapse at high sparsities (e.g., 40\%). To overcome this topological phase transition, we propose \textsc{Relative Kinetic Utility} (RKU), a novel theoretical framework that elevates discrete pruning to a continuous kinetic integral over the depth manifold of the model based on Alternating Gradient Flow(AGF). By modifying it with Fisher trace normalization, RKU acts as a lightweight curvature-aware normalization to isolate \textit{kinetic spikes} -- the fundamental structural pathways responsible for high-curvature logical routing. Extensive experiments on Qwen-2.5-7B and LLaMA-3-8B improves performance in the high-sparsity regime around 40\%. RKU attains 13.34\% accuracy on GSM8K at 40\% sparsity, outperforming the strongest baseline, and appears to better preserve reasoning-relevant representations under out-of-distribution evaluation. 

---
# Non-Monotonic Latency in Apple MPS Decoding: KV Cache Interactions and Execution Regimes 

**Authors**: Willy Fitra Hendria  

**Link**: [PDF](https://arxiv.org/pdf/2605.08913)  

**Abstract**: Autoregressive inference is typically assumed to scale predictably with decoding length, and key-value (KV) caching is widely regarded as a universally beneficial optimization for accelerating decoding. In this work, we identify unexpected non-monotonic latency behavior in the Apple MPS backend, where latency changes abruptly across nearby decoding configurations. Using transformer models from multiple families (GPT-2, BLOOM, and OPT), we observe latency spikes of up to 21x within specific decoding-budget intervals, followed by recovery at neighboring configurations. Controlled experiments show that these anomalies are not explained by memory pressure or prefill cost, but are instead consistent with backend execution dynamics, while CPU and NVIDIA T4 (CUDA) exhibit smooth monotonic scaling under identical conditions. Our findings highlight the importance of hardware-aware evaluation for autoregressive inference and caution against relying on aggregated decoding-budget benchmarks, as performance can vary discontinuously across nearby configurations. 

---
# Machine Learning Research Has Outpaced Its Communication Norms and NeurIPS Should Act 

**Authors**: Ajay Mandyam Rangarajan, Jeyashree Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08889)  

**Abstract**: Machine learning research has grown exponentially while its communication norms have not. We argue NeurIPS should adopt explicit, measurable writing standards. We analyze 2.8 million arXiv papers (1991-2025), 24,772 NeurIPS papers (1987-2024), and 24.5 million PubMed papers (1990-2025), applying classical readability scores, the Hohmann writing style suite (including sensational language), acronym density and reuse, an LLM as judge readability protocol, and citations from OpenAlex and Semantic Scholar. Four patterns emerge. First, NeurIPS abstracts score harder to read on every classical readability metric: Flesch Reading Ease falls from about 24 in 1987 to 13 in 2024, and sensational language rises by about 50 percent in NeurIPS abstracts between 2015 and 2024. Second, acronym density in NeurIPS titles has grown from 0.33 per 100 words in 1987 to 3.21 in 2024, and about 89 percent of NeurIPS acronyms are used fewer than ten times, ten points above the science-wide baseline. Third, more readable NeurIPS papers tend to receive more citations, suggesting readability and impact are correlated and that less readable papers risk remaining fragmented. LLM as judge scores rate NeurIPS abstracts as roughly stable from 1987 to 2022, with early signs of improvement thereafter, a pattern that disagrees with every classical readability metric and raises a design question for enforcement: is the target reader a human or an LLM? Lastly, NeurIPS volume has grown roughly 50-fold between 1987 and 2024. Assuming the goal is to optimise for human readers, we propose seven standards NeurIPS could pilot at NeurIPS 2027: an acronym budget with a venue-approved term list, a human readability threshold, stricter citation standards, standalone visual elements, a plain language summary, a pre-registered acronym glossary, and open source audit tooling. 

---
# Ace-Skill: Bootstrapping Multimodal Agents with Prioritized and Clustered Evolution 

**Authors**: Feng Xiong, Zengbin Wang, Yong Wang, Xuecai Hu, Jinghan He, Liang Lin, Yuan Liu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08887)  

**Abstract**: Self-evolving agents present a promising path toward continual adaptation by distilling task interactions into reusable knowledge artifacts. In practice, this paradigm remains hindered by two coupled bottlenecks: data inefficiency, where costly rollout effort is disproportionately spent on low-value samples rather than informative ones, and knowledge interference, where heterogeneous knowledge stored in shared repositories leads to noisy retrieval and task-misaligned guidance. Together, these issues form a self-reinforcing failure loop in which uninformative rollouts yield noisy knowledge, which in turn degrades subsequent rollouts. In this work, we introduce Ace-Skill, a co-evolutionary framework that jointly optimizes rollout allocation and knowledge organization for self-evolving multimodal agents. Specifically, Ace-Skill combines aprioritized sampler with lazy-decay proficiency tracking to focus rollouts on informative and insufficiently mastered samples, and a clustered organizer that semantically clusters knowledge for cleaner retrieval and more reliable adaptation. By improving sampling and organization together, Ace-Skill turns self-evolution into a virtuous cycle in which more informative rollouts produce higher-quality knowledge that supports stronger subsequent rollouts. Across four multimodal tool-use benchmarks, Ace-Skill delivers strong gains (e.g., +35.46% relative improvement in Avg@4 accuracy), enabling an opensource 35B MoE model to match or surpass proprietary models. The acquired knowledge also transfers effectively in a zero-shot manner to smaller 9B and 4B models, allowing resource-constrained agents to inherit advanced capabilities without additional training. The code has been publicly available at this https URL. 

---
# UserGPT Technical Report 

**Authors**: Yunyi Xuan, Hao Yi, Fengling Mao, Daye Cai, Leikun Liang, Xingsheng He, Jiangnan Xie, Guoshuai Wang, Yushan Han, Wenwen Guo, Xiaoxiao Xu, Lin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08766)  

**Abstract**: Personalized user understanding from large-scale digital traces remains a fundamental challenge. Traditional user profiling methods rely on discriminative models and manual feature engineering to predict discrete attributes, often producing fragmented and logically inconsistent profiles that generalize poorly to long-tail behaviors. In this work, we study a generative paradigm in which large language models (LLMs) summarize long and noisy behavioral histories into coherent narratives that capture nuanced user evolution. Our experiments show that even strong LLMs remain limited in complex and implicit personalization reasoning.
We propose UserGPT, a framework for improving LLM-based persona understanding through both attribute generation and summary generation. To address the scarcity of real-world behavioral data, we develop a User Behavior Simulation Engine that produces realistic and complex user trajectories. We further introduce a Data-Centric Semantization module that transforms heterogeneous behavioral logs into structured and semantically coherent inputs, reducing noise and sparsity. On top of this pipeline, we design a curriculum-driven post-training strategy that combines multi-stage Supervised Fine-Tuning (SFT) with Dual-Filter Group Relative Policy Optimization (DF-GRPO) to strengthen reasoning over long behavioral histories.
We also construct HPR-Bench, a benchmark for holistic persona reasoning derived from simulated data. On HPR-Bench, UserGPT achieves an Avg@10 score of 0.7325 on tag prediction and an $Acc_{Ex}$ score of 0.7528 on summary generation, while compressing behavioral records by up to 97.9% with critical information preserved. These results demonstrate the effectiveness of UserGPT for holistic persona reasoning and personalized user-agent interaction. 

---
# Communicating Sound Through Natural Language 

**Authors**: Emanuele Rossi, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2605.08750)  

**Abstract**: Natural language is widely used to describe, prompt, and control audio systems, but rarely serves as the representation carrying audio itself. We introduce lexical acoustic coding (LAC), a framework in which pre-trained LLM sender and receiver agents transmit sound through natural language. Under fixed system prompts, the agents write their own analysis and synthesis code, communicating only through a lexical sentence, shared vocabulary, and optional symbolic music structure. The sender analyzes an input waveform into interpretable, non-learned acoustic descriptors, quantizes each with a feature-specific interval vocabulary, and verbalizes the lexical code as English. The receiver parses the sentence back into lexical-acoustic constraints and renders a waveform through closed-loop refinement. The transmitted text serves as both a rich caption and as the transport representation itself. We frame LAC as a finite-rate lossy quantizer, exposing trade-offs between vocabulary size, rate, and fidelity. Experiments on short sounds and symbolic music transfer show that plain text preserves measurable acoustic structure while remaining interpretable, editable, and native to LLM-mediated communication. 

---
# SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training 

**Authors**: Shengkun Tang, Zekun Wang, Bo Zheng, Liangyu Wang, Rui Men, Siqi Zhang, Xiulong Yuan, Zihan Qiu, Zhiqiang Shen, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08738)  

**Abstract**: Structured pruning and knowledge distillation (KD) are typical techniques for compressing large language models, but it remains unclear how they should be applied at pretraining scale, especially to recent mixture-of-experts (MoE) models. In this work, we systematically study MoE compression in large-scale pretraining, focusing on three key questions: whether pruning provides a better initialization than training from scratch, how expert compression choices affect the final model after continued training, and which training strategy is most effective. We have the following findings: First, across depth, width, and expert compression, pruning a pretrained MoE consistently outperforms training the target architecture from scratch under the same training budget. Second, different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining. Motivated by this, we introduce a simple partial-preservation expert merging strategy that improves downstream performance across most benchmarks. Third, combining KD with the language modeling loss outperforms KD alone, particularly on knowledge-intensive tasks. We further propose multi-token prediction (MTP) distillation, which yields consistent gains. Finally, given the same training tokens, progressive pruning schedules outperform one-shot compression, suggesting that gradual architecture transitions lead to better optimization trajectories. Putting it all together, we compress Qwen3-Next-80A3B to a 23A2B model that retains competitive performance. These results offer practical guidance for efficient MoE compression at scale. 

---
# The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs 

**Authors**: Xin Li, Hao Jiang, Annan Wang, Yichi Zhang, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08737)  

**Abstract**: On-policy distillation (OPD) is widely used for LLM post-training. When pushed with a reward-extrapolation coefficient lambda > 1, the student can lift past the teacher in domain, but past a threshold lambda* the same step violates the output contract on structured-output tasks. In a single-position Bernoulli reduction, we derive a closed-form base-relative clip-safety threshold lambda*(p,b,c) determined by three measurable quantities: the teacher modal probability, the warm-start mass, and the importance-sampling clip strength. Above lambda*, the extrapolated fixed point exits the clip-safe region, changing training from format-preserving to format-collapsing. We extend the rule to calibrated K-ary listwise JSON tasks where a single binding equivalence class dominates the output contract and SFT retains parse headroom. On Amazon Fashion, three pre-registered tests--a fine-grid cliff interval, a budget-extension test, and a small-clip cross-prediction--fall within their locked prediction windows, with the small-clip value matching the closed-form prediction below grid resolution. Operating just below lambda*, ListOPD brings a 1.7B Qwen3 student to in-domain parity with an 8B-SFT baseline at one-fifth the parameters. The gain is driven primarily by format adherence: NDCG@1 on parsed outputs remains flat across lambda, while parse validity sharply changes at the predicted boundary. The cliff diagnostic is rubric-independent, whereas the parity claim uses a Gemini-graded rubric and inherits that evaluator's exposure. 

---
# AdaPreLoRA: Adafactor Preconditioned Low-Rank Adaptation 

**Authors**: Ziyun Liu, Fengmiao Bian, Jian-Feng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.08734)  

**Abstract**: Low-Rank Adaptation (LoRA) reparameterizes a weight update as a product of two low-rank factors, but the Jacobian $J_{G}$ of the generator mapping the factors to the weight matrix is rank-deficient, so the factor-space preconditioner $J_{G}^* {F}_t J_{G}$ induced by any ${W}$-space preconditioner ${F}_t$ is singular, and consequently the standard chain rule cannot be uniquely inverted to map a preconditioned ${W}$-space direction back to a factor-space update. We cast existing LoRA optimizers in a unified framework parameterized by two choices: (i) which invertible surrogate for $J_{G}^* {F}_t J_{G}$ to use, and (ii) which ${F}_t$ on ${W}$ to use. Existing methods occupy four families along these axes: factor-space adaptive updates, block-diagonal surrogates for $J_{G}^* J_{G}$, Frobenius-residual pseudoinverse methods, and Riemannian manifold constraint. Within this design space, a gradient-statistics-aware ${F}_t$ paired with a closed-form factor-space solve at ${O}((m+n)r)$ memory remains underexplored. We propose \textbf{AdaPreLoRA}, which fills this gap by adopting the Adafactor diagonal Kronecker preconditioner ${H}_t$ on ${W}$ and selecting from the resulting factor-space solution family the element minimizing an ${H}_t$-weighted imbalance between the two factor contributions; by construction, the resulting factor update is the closest LoRA approximation to the preconditioned ${W}$-space direction under the ${H}_t$-weighted norm. Across GPT-2 (E2E), Mistral-7B and Qwen2-7B (GLUE, ARC, GSM8K), and diffusion-model personalization, AdaPreLoRA is competitive with or improves over a representative set of LoRA optimizers while keeping peak GPU memory at the LoRA optimizer level. 

---
# Bias by Necessity: Impossibility Theorems for Sequential Processing with Convergent AI and Human Validation 

**Authors**: Jikun Wu, Dongxin Guo, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08716)  

**Abstract**: Are certain cognitive biases mathematically inevitable consequences of sequential information processing? We prove that primacy effects, anchoring, and order-dependence are architecturally necessary in autoregressive language models due to causal masking constraints. Our three impossibility theorems establish: (1) primacy bias arises from asymmetric attention accumulation; (2) anchoring emerges from sequential conditioning with provable information bounds; and (3) exact debiasing by permutation marginalization requires factorial-time computation, with Monte Carlo approximation feasible at constant per-tolerance overhead. We validate these bounds across 12 frontier LLMs ($R^2 = 0.89$; $\Delta$BIC $= 16.6$ vs. next-best alternative). We then derive quantitative predictions from the framework and test them in two pre-registered human experiments ($N = 464$ analyzed). Study 1 confirms anchor position modulates anchoring magnitude ($d = 0.52$, BF$_{10} = 847$). Study 2 shows working memory load amplifies primacy bias ($d = 0.41$, BF$_{10} = 156$), with WM capacity predicting bias reduction ($r = -.38$). These convergent findings reframe cognitive biases as resource-rational responses to sequential processing. 

---
# RewardHarness: Self-Evolving Agentic Post-Training 

**Authors**: Yuxuan Zhang, Penghui Du, Bo Li, Cong Wei, Junwen Miao, Huaisong Zhang, Songcheng Cai, Yubo Wang, Dongfu Jiang, Yuyu Zhang, Ping Nie, Wenhu Chen, Changqian Yu, Kelsey R. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08703)  

**Abstract**: Evaluating instruction-guided image edits requires rewards that reflect subtle human preferences, yet current reward models typically depend on large-scale preference annotation and additional model training. This creates a data-efficiency gap: humans can often infer the target evaluation criteria from only a few examples, while models are usually trained on hundreds of thousands of comparisons. We present RewardHarness, a self-evolving agentic reward framework that reframes reward modeling as context evolution rather than weight optimization. Instead of learning from large-scale annotations, RewardHarness aligns with human preferences by iteratively evolving a library of tools and skills from as few as 100 preference demonstrations. Given a source image, candidate edited images, and an editing instruction, an Orchestrator selects the most relevant subset of tools and skills from the maintained library, and a frozen Sub-Agent uses them to construct a reasoning chain that produces a preference judgment. By comparing predicted judgments with ground-truth preferences and analyzing successes and failures in the reasoning process, the Orchestrator automatically refines its library of tools and skills without additional human annotation. Using only 0.05% of the EditReward preference data, RewardHarness achieves 47.4% average accuracy on image-editing evaluation benchmarks, surpassing GPT-5 by 5.3 points. When used as a reward signal for GRPO fine-tuning, RL-tuned models achieve 3.52 on ImgEdit-Bench. Project page: this https URL. 

---
# AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM Weight Quantization 

**Authors**: Beshr IslamBouli, David Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.08692)  

**Abstract**: Post-training weight-only quantization to 4 bits is widely used to reduce the memory and compute costs of large language model inference. Existing PTQ methods, such as AWQ and GPTQ, improve how weights are mapped onto a fixed 4-bit grid through scaling, clipping, or error compensation. To further improve accuracy, methods such as OmniQuant and QuIP\# uses gradient-assisted algorithms at the cost of hours of quantization time. In this work, we propose AAAC (Activation-Aware Adaptive Codebooks), a lightweight method for 4-bit LLM weight quantization. AAAC replaces the fixed scalar codebook used in standard quantization with two small learned scalar codebooks (64 bytes) per layer. Each group of weights selects the codebook that minimizes activation-weighted reconstruction error, encoding the choice in the unused sign bit of the group's positive scale and adding zero storage overhead. AAAC completes in 3--30 minutes on a single GPU, and adds no memory beyond the model itself. We evaluate against AWQ, GPTQ, IF4, GPTVQ, OmniQuant, SqueezeLLM, and QuIP\# across model families. AAAC outperforms baselines at orders-of-magnitude less quantization time. 

---
# MIND-Skill: Quality-Guaranteed Skill Generation via Multi-Agent Induction and Deduction 

**Authors**: Yixuan Li, Mingshu Cai, Ziyang Xiao, Wanyuan Wang, Yanchen Deng, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2605.08670)  

**Abstract**: Large language model (LLM) powered AI agents have emerged as a promising paradigm for autonomous problem-solving, yet they continue to struggle with complex, multi-step real-world tasks that demand domain-specific procedural knowledge. Reusable agent skills, which encapsulate successful problem-solving strategies, offer a natural remedy by enabling agents to build on prior experience. However, curating such skills has largely remained a manual endeavor, requiring human experts to distill rich domain knowledge into actionable guidelines. In this work, we present $\textbf{M}$ulti-agent $\textbf{IN}$duction and $\textbf{D}$eduction for $\textbf{Skill}$s ($\textbf{MIND-Skill}$), a framework that automatically induces generalizable skills from successful trajectories with robust quality guarantees. MIND-Skill consists of an induction agent which is tasked to abstract reusable skills from successful trajectories, and a deduction agent which aims to reconstruct trajectories by following the induced skills. To guarantee the quality of the generated skills, we introduce a reconstruction loss that compares input and reconstructed trajectories, an outcome loss that enforces the correctness of the reconstructed trajectories, and a rubric loss that assesses the documentation quality and regularizes the abstraction level of the generated skills according to predefined criteria. These textual losses are jointly optimized with TextGrad, and the resulting skills are evaluated on held-out tasks unseen during optimization. Experiments on AppWorld and BFCL-v3 show that MIND-Skill consistently outperforms concurrent skill generation methods. 

---
# PAAC: Privacy-Aware Agentic Device-Cloud Collaboration 

**Authors**: Liangqi Yuan, Wenzhi Fang, Shiqiang Wang, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2605.08646)  

**Abstract**: Large language model (LLM) agents face a structural tension: cloud agents provide strong reasoning but expose user data, while on-device agents preserve privacy at the cost of overall capability. Existing device-cloud designs treat this boundary as a compute split rather than a trust boundary suited to agentic workloads, and existing sanitizers force a choice between policy flexibility and the structural fidelity tool calls require. In this work, we develop PAAC, a privacy-aware agentic framework that aligns planner--executor decomposition with the device-cloud boundary so that role specialization itself becomes the privacy mechanism. The cloud agent reasons over typed placeholder tokens that preserve each sensitive value's reasoning role while discarding its content, while the on-device agent identifies sensitive spans and distills each step's execution outcome into compact key findings. Sanitization confines the on-device LLM to proposing which spans to mask, while a deterministic registry performs all substitution and reversal, keeping actions directly executable on device. On three agentic benchmarks under strict privacy settings, PAAC dominates the Pareto frontier of privacy and accuracy, improving average accuracy by 15-36\% and reducing average leakage by 2-6$\times$ over state-of-the-art device-cloud baselines, with the largest margins on privacy targets outside fixed entity taxonomies. We find consistent improvements on 17 additional benchmarks spanning 10 domains, including math, science, and finance. 

---
# Causal Stories from Sensor Traces: Auditing Epistemic Overreach in LLM-Generated Personal Sensing Explanations 

**Authors**: Shanshan Zhu, Han Zhang, J. Doris Chi, Subigya Nepal, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.08590)  

**Abstract**: LLMs are increasingly used to explain personal sensing data, translating traces of activity and mood into natural-language accounts of why an anomalous day may have occurred. However, such explanations can sound coherent and personally meaningful even when the underlying evidence is sparse or missing. We introduce epistemic overreach (EO) as a measure for cases where a generated explanation implies more than the available sensing evidence can justify. To audit how often and in what forms EO occurs, we obtained anomalous-day scenarios from three longitudinal sensing datasets of college students: StudentLife, GLOBEM, and CollegeExperience. Across activity, sleep, and affect anomalies, we generated 14,922 explanations using three LLM families -- Llama, Qwen, and GPT -- under two prompting conditions: one minimally constrained prompt and another prompt explicitly instructing models to bound claims to the data. For each scenario, we varied the amount of behavioral evidence available to the model to examine whether more evidence reduces EO. We evaluated each explanation using a structured rubric, decomposing EO into the dimensions of unsupported causal attribution, unacknowledged data gaps, overconfident language, temporal inconsistency, and diagnostic inference. We find that LLMs routinely attribute anomalous days to causes without sufficient support from the data, and that this pattern replicates across datasets, anomaly types, and model families. Further, providing richer context does not reliably reduce EO; bounded prompting helps but does not eliminate it. These findings suggest that evidential grounding should be a first-order evaluation criterion for LLM-generated personal sensing explanations, alongside fluency and plausibility. We argue that personal sensing explanations require evidential discipline: systems must distinguish what is observed, what is inferred, and what remains unknown. 

---
# Human-Inspired Memory Architecture for LLM Agents 

**Authors**: Doga Kerestecioglu, Alexei Robsky, Clemens Vasters, Anshul Sharma, Yitzhak Kesselman  

**Link**: [PDF](https://arxiv.org/pdf/2605.08538)  

**Abstract**: Current LLM agents lack principled mechanisms for managing persistent memory across long interaction horizons. We present a biologically-grounded memory architecture comprising six cognitive mechanisms: (1) sleep-phase consolidation, (2) interference-based forgetting, (3) engram maturation, (4) reconsolidation upon retrieval, (5) entity knowledge graphs, and (6) hybrid multi-cue retrieval. Each mechanism addresses a specific failure mode of naive memory accumulation. We introduce a synthetic calibration methodology that derives all pipeline thresholds without benchmark data exposure, eliminating a common source of evaluation leakage. We evaluate on two benchmarks. First, a VSCode issue-tracking dataset (13K issues, 120K events) where deduplication-based consolidation achieves 97.2% retention precision with 58% store reduction (+21.8 pp over baseline). Second, the LongMemEval personal-chat benchmark where we conduct the first streaming M-tier evaluation (475 sessions, ~540K unique turns). At a 200K-token context budget, our pipeline matches raw retrieval accuracy (70.1% vs. 71.2%, overlapping 95% CI) while exposing a tunable accuracy/store-size operating curve. At S-tier scale (50 sessions), dedup-based consolidation yields a +13.3 pp improvement in preference recall. 

---
# ShifaMind: A Multiplicative Concept Bottleneck for Interpretable ICD-10 Coding 

**Authors**: Mohammed Sameer Syed, Xuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08482)  

**Abstract**: Automated ICD-10 coding from clinical discharge summaries requires models that are both accurate on long-tailed multi-label classification tasks and interpretable to clinicians. Concept Bottleneck Models (CBMs) offer a principled framework for interpretability by routing predictions through human-interpretable concepts, but this transparency often comes at a cost: compressing rich clinical text representations into a narrow concept layer can restrict gradient flow and limit predictive capacity. We present ShifaMind, a concept-grounded architecture built around a Multiplicative Concept Bottleneck (MCB), which changes the form, rather than the width, of the bottleneck. Instead of projecting through a narrow concept layer, ShifaMind uses a learned multiplicative gate over a concept-grounded representation while retaining a scalar concept interface for inspection. On MIMIC-IV top-50 ICD-10 coding, ShifaMind achieves performance competitive with LAAT, the strongest baseline, across F1, AUC, and ranking metrics, while outperforming five additional ICD-coding baselines and providing concept-mediated explanations. Its substantial gains over a capacity-matched Vanilla CBM in both predictive performance and interpretability-oriented metrics highlight the importance of the bottleneck design. 

---
# LLM-guided Semi-Supervised Approaches for Social Media Crisis Data Classification 

**Authors**: Jacob Ativo, Bharaneeshwar Balasubramaniyam, Anh Tran, Khushboo Gupta, Hongmin Li, Doina Caragea, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2605.08448)  

**Abstract**: Semi-supervised learning approaches have been investigated as a means to enhance the analysis of social media data in disaster management contexts. In this work, we present the first empirical evaluation of large language model (LLM) guided semi-supervised learning for crisis related tweet classification. We compare two recent LLM assisted semi-supervised methods, VerifyMatch and LLM guided Co-Training ( LG-CoTrain), against established semi-supervised baselines. Our results show that LG-CoTrain significantly outperforms classical semi-supervised approaches in low resource settings with 5, 10 and 25 labeled examples per class, achieving the highest averaged Macro F1 across events. VerifyMatch achieves competitive performance while also demonstrating strong calibration properties. As the number of labeled examples increases, the performance gap narrows and Self Training emerges as a strong baseline. We further observe that compact semi-supervised models can, in some cases, outperform very large LLMs operating in zero-shot settings. This finding highlights the potential of transferring knowledge from LLMs into smaller and more deployable models through LLM guided semi-supervised learning, offering a practical pathway for real world disaster response applications. Our project repository on Github is here. 

---
# Queryable LoRA: Instruction-Regularized Routing Over Shared Low-Rank Update Atoms 

**Authors**: Omatharv Bharat Vaidya, Connor T. Jerzak, Nhat Ho, Chandrajit Bajaj  

**Link**: [PDF](https://arxiv.org/pdf/2605.08423)  

**Abstract**: We present a data-adaptive method for parameter-efficient fine-tuning of large neural networks. Standard low-rank adaptation methods improve efficiency by restricting each layer update to a fixed low-rank form, but this static parameterization can be too rigid when the appropriate correction depends on the input and on the evolving depth-wise computation of the network. Our approach replaces a purely layer-local adapter with a shared queryable memory of low-rank update atoms. For each block of layers, the model forms a query from the current low-rank state and a running summary of previous blocks, uses this query to retrieve a content-dependent combination of shared update components via attention, and applies the resulting routed operator within the low-rank bottleneck. In this way, the method retains the efficiency and scalability of low-rank adaptation while allowing the effective update to vary across inputs and to share reusable structure across layers. The resulting architecture provides a principled middle ground between static LoRA-style updates and fully generated parameter updates: it remains compact and parameter-efficient while supporting dynamic, context-sensitive adaptation. Further, we incorporate instruction-regularization by augmenting routing logits with a language-induced prior over update atoms, thereby biasing the selection of low-rank transformations toward semantically relevant directions without generating unconstrained parameter updates. Experiments on noisy non-linear regression tasks and LLM fine-tuning suggest that this queryable update-memory formulation can improve final test performance and training stability compared to standard low-rank adaptation, while using a comparable number of trainable parameters. 

---
# SecureForge: Finding and Preventing Vulnerabilities in LLM-Generated Code via Prompt Optimization 

**Authors**: Houjun Liu, Lisa Einstein, John Yang, Joachim Baumann, Duncan Eddy, Christopher D. Manning, Mykel Kochenderfer, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08382)  

**Abstract**: LLM coding agents now generate code at an unprecedented scale, yet LLM-generated code introduces cybersecurity vulnerabilities into codebases without human involvement. Even when frontier models are explicitly asked to write secure production code with relevant weaknesses to avoid in context, we find that they still produce verifiable vulnerabilities on average 23% of the time across a corpus of 250 benign coding prompts. We introduce SecureForge, an automated pipeline that both audits security risks of frontier models and produces auditing-informed secure system prompts that reduce output security vulnerabilities while maintaining unit test performance. SecureForge first identifies benign prompts that produce statically detectable vulnerabilities, and then amplifies them into a large synthetic prompt corpus of diverse scenarios using a Markovian sampling technique to jointly maintain error rates and prompt diversity. This corpus is then used to iteratively optimize the system prompts to reduce output security vulnerabilities. On frontier models, SecureForge yields a statistically significant Pareto improvement in both unit test success and output security, with output vulnerabilities reduced by up to 48%. The resulting system prompts transfer zero-shot to in-the-wild coding agent prompts, without any exposure to real user prompt distributions during optimization. 

---
# Reinforcement Learning for Scalable and Trustworthy Intelligent Systems 

**Authors**: Guangchen Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08378)  

**Abstract**: Reinforcement learning has become a powerful paradigm for improving the capability of intelligent systems, but its practical deployment faces two central challenges. First, reinforcement learning must scale efficiently in distributed environments where communication bandwidth is limited and computation is heterogeneous across agents. Second, as reinforcement learning is increasingly used in post-training large language models and autonomous agents, the optimized policies must also be aligned with human preferences and satisfy safety requirements such as privacy-aware information disclosure. This dissertation addresses both challenges through four complementary contributions spanning federated optimization, preference alignment, and contextual safety.
The first part of the dissertation studies scalable reinforcement learning in federated settings. The second part of the dissertation studies trustworthy reinforcement learning for large language models. Together, these contributions advance reinforcement learning along two complementary dimensions. On the one hand, they make reinforcement learning more scalable through communication-efficient and asynchronous federated optimization. On the other hand, they make reinforcement learning more trustworthy by improving alignment with human preferences and by reducing contextually inappropriate information disclosure in language-based intelligent systems. As a whole, this dissertation argues that the next generation of intelligent systems will require both efficient optimization and trustworthy behavior, and that reinforcement learning provides a unifying framework for addressing both goals. 

---
# CDS4RAG: Cyclic Dual-Sequential Hyperparameter Optimization for RAG 

**Authors**: Pengzhou Chen, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08333)  

**Abstract**: Retrieval-Augmented Generation (RAG) is sensitive to the vast hyperparameters of the retriever and generator, yet optimizing them using given queries is a challenging task due to the complex interactions and expensive evaluation costs. Existing algorithms are ineffective and slow in convergence, since they often treat RAG as a monolithic black box or only optimize partial hyperparameters. In this paper, we propose CDS4RAG, a framework that optimizes the full RAG hyperparameters using given queries via a new cyclic dual-sequential formulation. CDS4RAG is special in the sense that it distinguishes the hyperparameters of the retriever and generator, cyclically optimizing them in turn. Such a paradigm allows us to design fine-grained within-cycle budget provision and expedite the optimization via cross-cycle seeding when optimizing the generator. CDS4RAG is also an algorithm-agnostic framework that can be paired with diverse general algorithms. Through experiments on four common benchmarks and two backbone LLMs, we reveal that CDS4RAG considerably boosts the vanilla algorithms in 21/24 cases while significantly outperforming state-of-the-art algorithms in all cases with up to 1.54x improvements of generation quality and better speedup. 

---
# LLMSYS-HPOBench: Hyperparameter Optimization Benchmark Suite for Real-World LLM Systems 

**Authors**: Siyu Wu, Yulong Ye, Zezhen Xiang, Pengzhou Chen, Gangda Xiong, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.08305)  

**Abstract**: Large Language Model (LLM) systems have been the frontier of AI in many application domains, leading to new challenges and opportunities for hyperparameter optimization (HPO) for the AutoML community. However, this type of system exhibits an unprecedented compound space of hyperparameter configuration from both the AI and non-AI components; rich and nonlinear implications from the fidelity factors; and diverse costs of measuring hyperparameter configurations, none of which have been fully captured in existing benchmarks. This paper presents the first (live) benchmark suite and datasets for HPO of real-world LLM systems, dubbed LLMSYS-HPOBench, covering data related to the inference objective values of hyperparameter configurations profiled from running the LLM systems. Currently, LLMSYS-HPOBench contains 364,450 hyperparameter configurations with a dimensionality of 12-23, 3-5 dimensions of fidelity factor leading to 932 settings, 3-9 inference objective metrics, and 2-10 cost metrics, together with generated logs from measuring the LLM systems. What we seek to advocate is not only a revalidation of the existing HPO algorithms over the frontier LLM systems, but also to provide an evolving platform for the AutoML community to explore new directions of research in this regard. The benchmark suite has been made available at: this https URL 

---
# mHC-SSM: Manifold-Constrained Hyper-Connections for State Space Language Models with Stream-Specialized Adapters 

**Authors**: Abdulvahap Mutlu, Şengül Doğan, Türker Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2605.08300)  

**Abstract**: Manifold-Constrained Hyper-Connections (mHC) introduce a stability-motivated variant of multi stream residual mixing by constraining residual stream mixing matrices to the manifold of doubly stochastic matrices via Sinkhorn-Knopp projection. In his work, we study whether mHC-style constrained multi-stream residual topology transfers effectively to state space model (SSM) language modeling. We implement a static mHC mechanism around an SSM block by expanding the residual stream into multiple parallel streams, aggregating streams into a single SSM input through simplex-constrained pre-mixing, scattering the SSM output back to streams through simplex-constrained post-mixing, and applying Sinkhorn-projected residual stream mixing at each layer. We further introduce stream-specialized adapters that add lightweight stream-specific capacity through a shared bottleneck with per-stream scaling, applied both before stream aggregation and after the SSM output prior to scattering. We evaluate baseline single-stream SSM, static mHC SSM, and mHC SSM with adapters on WikiText-2 using identical training settings and report checkpoint-based validation loss, perplexity, throughput, and peak GPU memory. Under the reported fair checkpoint evaluation, static mHC improves validation loss from 6.3507 to 6.2448 and reduces perplexity from 572.91 to 515.35, while mHC with adapters further improves validation loss to 6.1353 and perplexity to 461.88. These gains are accompanied by modest throughput reductions from 1025.52 to 964.81 and 938.90 tokens per second, and increased peak memory from 2365 MB to 2568 MB and 3092 MB. The results suggest that mHC-inspired constrained multi-stream residual mixing can yield measurable quality improvements in SSM language models and that stream-specialized adapter capacity can further enhance performance with predictable efficiency tradeoffs. 

---
# In-Context Fixation: When Demonstrated Labels Override Semantics in Few-Shot Classification 

**Authors**: Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08295)  

**Abstract**: While random demonstration labels barely hurt in-context learning (Min et al., 2022), we show that homogeneous labels--even semantically valid ones--collapse accuracy to <=12% across six models (Pythia, Llama, Qwen; 0.8B--8B) and four tasks. The trigger is label-slot content: the model treats tokens occupying the label position as an exhaustive answer vocabulary, with homogeneity as the maximally collapsed case. A novel set-level fixation finding confirms this: when demonstrations carry varied nonsense tokens from {foo,bar,vex,nit,orb}, the model places 42--67% of probability on the demonstrated set while P(dog) remains below 0.2%. This is inconsistent with latent-concept Bayesian accounts (Xie et al., 2022) and reveals that ICL output is constrained vocabulary retrieval--the model binds its output to the demonstrated token inventory regardless of semantic plausibility. The effect generalizes to 4-way classification (0% accuracy across three models, 1B--8B) and multi-token verbalizers ("very positive"), where we decompose fixation into format-level (template adoption) and content-level (polarity override) components that are experimentally dissociable. Mechanistically, per-item paired activation patching on Pythia-1B recovers 98.4% of the gap (95% CI [84%, 112%]), localizing fixation to a layer-7-centered circuit (rank 2/560, 99.8th percentile; 4-fold CV mean 103%). Cross-architecture logit lens on Llama-3.2-1B replicates the encode-then-override trajectory with causal confirmation (top-5 layers: 89% recovery). 

---
# HTPO: Towards Exploration-Exploitation Balanced Policy Optimization via Hierarchical Token-level Objective Control 

**Authors**: Xincheng Yao, Ruoqi Li, Cheng Chen, Daoxin Zhang, Yi Wu, Yao Hu, Chongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.08283)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a pivotal technique for enhancing the reasoning capabilities of Large Language Models (LLMs). However, the de facto practice of mainstream RL algorithms is to treat all tokens of one response equally and assign the same optimization objective to each token, failing to provide granular guidance for the reasoning process. While in Chain-of-Thought (CoT) reasoning, different tokens usually play distinct roles. Therefore, the current RL algorithms lack an effective mechanism to dynamically balance the exploration-exploitation trade-off during learning. To this end, we propose Hierarchical Token-level Objective Control Policy Optimization (HTPO), a novel RL algorithm that takes the divide-and-conquer idea to hierarchically partition the response tokens into specific functional groups from three aspects (i.e., prompt difficulty, answer correctness, and token entropy). Within each group, according to the contributions to exploration or exploitation, we design specialized optimization objectives to facilitate the effective execution of each token's expected functionality. In this way, HTPO can achieve a more balanced exploration-exploitation trade-off. Extensive experiments on challenging reasoning benchmarks validate the superiority of our HTPO algorithm, which significantly outperforms the strong DAPO baseline (e.g., +8.6% and +6.7% on AIME'24 and AIME'25, respectively). When scaling test-time compute, the HTPO-trained model maintains a consistent performance advantage over the DAPO baseline, and the gap widens as the sampling budget increases, validating that our adaptive token-level control method fosters effective exploration without sacrificing exploitation performance. Code will be at this https URL. 

---
# Spatial Priming Outperforms Semantic Prompting: A Grid-Based Approach to Improving LLM Accuracy on Chart Data Extraction 

**Authors**: Andrei Lazarev, Dmitrii Sedov, Alexander Galkin  

**Link**: [PDF](https://arxiv.org/pdf/2605.08220)  

**Abstract**: The automated extraction of data from scientific charts is a critical task for large-scale literature analysis. While multimodal Large Language Models (LLMs) show promise, their accuracy on non-standardized charts remains a challenge. This raises a key research question: what is the most effective strategy to improve model performance (high-level semantic priming) or low-level spatial priming? This paper presents a comparative investigation into these two distinct strategies. We describe our exploratory experiments with semantic methods, such as a two-stage metadata-first framework and Chain-of-Thought, which failed to produce a statistically significant improvement. In contrast, we present a simple but highly effective spatial priming method: overlaying a coordinate grid onto the chart image before analysis. Our quantitative experiment on a synthetic dataset demonstrates that this grid-based approach provides a statistically significant reduction in data extraction error (SMAPE reduced from 25.5% to 19.5%, p < 0.05) compared to a baseline. We conclude that for the current generation of multimodal models, providing explicit spatial context is a more effective and reliable strategy than high-level semantic guidance for this class of tasks. 

---
# LLMs with in-context learning for Algorithmic Theoretical Physics 

**Authors**: Anamaria Hell, Leander Thiele  

**Link**: [PDF](https://arxiv.org/pdf/2605.08212)  

**Abstract**: There is an increasing number of algorithmic computations in theoretical physics. These, while conceptually simple, can nevertheless be time-consuming and contain subtleties that should not be overlooked. Given the recent improvement of Large Language Models (LLM), it is natural to investigate whether LLMs equipped with a computer algebra system (CAS) runtime and sufficiently informative context can reliably carry out these algorithmic tasks. In this work, we interface Claude with Maple, and apply this framework to cosmological perturbations in modified theories of gravity. We demonstrate the current capabilities of this approach, the typical failures, and how the same can be improved. We find that a frontier LLM supplied with worked examples is able to solve most test problems. 

---
# MULTITEXTEDIT: Benchmarking Cross-Lingual Degradation in Text-in-Image Editing 

**Authors**: Liwei Cheng, Zirui Song, Shibo Feng, Lunjie Zhou, Yixuan Guan, Dayan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08163)  

**Abstract**: Text-in-image editing has become a key capability for visual content creation, yet existing benchmarks remain overwhelmingly English-centric and often conflate visual plausibility with semantic correctness. We introduce MULTITEXTEDIT, a controlled benchmark of 3,600 instances spanning 12 typologically diverse languages, 5 visual domains, and 7 editing operations. Language variants of each instance share a common visual base and are paired with a human-edited reference and region masks, isolating the language variable for cross-lingual comparison. To capture script-level errors that coarse text-matching metrics miss, such as missing diacritics, reversed RTL order, and mixed-script renderings, we introduce a language fidelity (LSF) metric scored by a two-stage LVM protocol that first traces the edited target text and then judges it in isolation, reaching a quadratic-weighted \k{appa} of 0.76 against native-speaker annotators. Evaluating 12 open-source and proprietary systems with LSF alongside standard semantic and mask-aware pixel metrics, we find pronounced cross-lingual degradation for every model, largest on Hebrew and Arabic and smallest on Dutch and Spanish, and concentrated in text accuracy and script fidelity rather than in coarse structural dimensions. We also uncover a pervasive semantic and pixel mismatch, where outputs preserve global layout and background fidelity yet distort script-specific forms. 

---
# Feature Rivalry in Sparse Autoencoder Representations: A Mechanistic Study of Uncertainty-Driven Feature Competition in LLMs 

**Authors**: Harshavardhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.08149)  

**Abstract**: Sparse Autoencoders (SAEs) decompose large language model representations into interpretable features, but how these features interact under uncertainty remains poorly understood. We introduce Feature Rivalry -- negatively correlated SAE feature pairs -- and study whether rivalry serves as a mechanistic signature of model uncertainty in Gemma-2-2B using Gemma Scope SAEs. Through a controlled within-domain experiment on PopQA split by response entropy, we find that high-entropy questions produce significantly stronger feature rivalry at layers 0 and 12 relative to low-entropy questions (p=5.3x10^-26 and p=5.8x10^-5 respectively), localizing uncertainty to specific processing stages in the residual stream. We then test whether rivalry is causally upstream of model outputs via activation steering along rivalry axes -- finding that steering along the rivalry direction (vec_A - vec_B) causes more output changes than random directions at low steering multipliers across 15 of 20 rival feature pairs. Finally, a per-prompt rivalry score derived from pairwise cosine similarities of active SAE feature decoder vectors predicts answer correctness (AUROC=0.689), approaching but not matching softmax confidence (AUROC=0.808). 

---
# Reasoning emerges from constrained inference manifolds in large language models 

**Authors**: Yanbiao Ma, Fei Luo, Linfeng Zhang, Chuangxin Zhao, Mingxuan Wang, Yinan Wu, Zhe Qian, Yang Lu, Long Chen, Zhao Cao, Xiaoshuai Hao, Ji-Rong Wen, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.08142)  

**Abstract**: Reasoning in large language models is predominantly evaluated through labeled benchmarks, conflating task performance with the quality of internal inference. Here we study reasoning as an intrinsic dynamical process by examining the evolution of internal representations during inference. We find that inference-time dynamics consistently self-organize into low-dimensional manifolds embedded within high-dimensional representation spaces. we find that such geometric compression, although pervasive, is not sufficient for stable or reliable reasoning. Instead, effective reasoning dynamics emerge within a constrained structural regime characterized by three conditions: adequate representational expressivity, spontaneous manifold compression, and preservation of non-degenerate information volume within the compressed subspace. Models outside this regime exhibit characteristic pathological inference dynamics. Based on these insights, we introduce a unified, label-free diagnostic computed solely from internal dynamics. These findings suggest that reasoning in LLMs is fundamentally governed by geometric and informational constraints, offering a complementary framework to benchmark-centric assessment. 

---
# Scaling Mobile Agent Systems: From Capability Density to Collective Intelligence 

**Authors**: Bowei He  

**Link**: [PDF](https://arxiv.org/pdf/2605.08124)  

**Abstract**: Mobile agent systems are emerging as a key paradigm for enabling intelligent applications on edge devices and in AIoT ecosystems. However, their scalability is fundamentally constrained by limited on-device computation and fragmented intelligence across devices. In this work, we propose a unified research agenda for scaling mobile agent systems along two complementary dimensions: (1) improving capability density of individual agents through compact foundation model design and compression, and (2) enabling collective intelligence via communication-rich multi-agent collaboration. Building on recent model and infrastructure advances, this vision aims to transform isolated mobile agents into a distributed intelligent system that is efficient and scalable. 

---
# Block-Wise Differentiable Sinkhorn Attention: Tail-Refinement Gradients with a Gap-Aware Dustbin Bridge 

**Authors**: Dylan Forde  

**Link**: [PDF](https://arxiv.org/pdf/2605.08123)  

**Abstract**: We study long-context balanced entropic optimal transport (OT) attention on TPU hardware through a stopped-base, fixed-depth tail-refinement surrogate. After a stopped $T$-step Sinkhorn solve, we unroll a short refinement tail and differentiate that surrogate exactly. For the production $R=2$ case, the backward pass contains four staircase plan factors. We prove an exact one-reference-tile schedule: the $R=2$ score cotangent is a single reference plan tile times an explicit modifier field built from vector cotangents and dual differences. This yields block-wise cost $O((T+R)LW)$, $O(Ld)$ input storage, and $O(L)$ additional HBM usage for fixed head dimension $d$ and band width $W$. We also formalize the current \texttt{dustbin\_block} path as the same balanced surrogate on an augmented support, so the schedule lifts to the gap-aware transport path used in our TPU runs. We provide a local surrogate-bias bound, an a posteriori bias certificate, and a projective contraction certificate for strictly positive active blocks. On synthetic masked problems, the optimized kernel matches exact autodiff of the same centered surrogate to within $10^{-5}$--$10^{-10}$. On TPU v6e-8, a four-configuration Pfam screen completes end-to-end, and a promoted balanced $R=2$ run sustains roughly $8.5$ examples per second through a three-hour budget, reaching step $1437$. Held-out Pfam test shards improve reconstruction from $3.17$ to $0.99$ and sparse CE from $5.86$ to $5.69$ relative to step $0$. These results support exact fixed-depth backward theory, a theorem-matching gap-aware bridge, and trainability evidence for the production path. 

---
