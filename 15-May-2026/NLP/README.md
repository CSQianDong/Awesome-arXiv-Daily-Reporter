# Is Grep All You Need? How Agent Harnesses Reshape Agentic Search 

**Authors**: Sahil Sen, Akhil Kasturi, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2605.15184)  

**Abstract**: Recent advances in Large Language Model (LLM) agents have enabled complex agentic workflows where models autonomously retrieve information, call tools, and reason over large corpora to complete tasks on behalf of users. Despite the growing adoption of retrieval-augmented generation (RAG) in agentic search systems, existing literature lacks a systematic comparison of how retrieval strategy choice interacts with agent architecture and tool-calling paradigm. Important practical dimensions, including how tool outputs are presented to the model and how performance changes when searches must cope with more irrelevant surrounding text, remain under-explored in agent loops. This paper reports an empirical study organized into two experiments. Experiment 1 compares grep and vector retrieval on a 116-question sample from LongMemEval, using a custom agent harness (Chronos) and provider-native CLI harnesses (Claude Code, Codex, and Gemini CLI), for both inline tool results and file-based tool results that the model reads separately. Experiment 2 compares grep-only and vector-only retrieval while progressively mixing in additional unrelated conversation history, so that each query is embedded in more distracting material alongside the passages that matter. Across Chronos and the provider CLIs, grep generally yields higher accuracy than vector retrieval in our comparisons in experiment 1; at the same time, overall scores still depend strongly on which harness and tool-calling style is used, even when the underlying conversation data are the same. 

---
# Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment 

**Authors**: Sayantan Kumar, Shahriar Noroozizadeh, Juyong Kim, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2605.15168)  

**Abstract**: Reconstructing precise clinical timelines is essential for modeling patient trajectories and forecasting risk in complex, heterogeneous conditions like sepsis. While unstructured clinical narratives offer semantically rich and contextually complete descriptions of a patient's course, they often lack temporal precision and contain ambiguous event timing. Conversely, structured electronic health record (EHR) data provides precise temporal anchors but misses a substantial portion of clinically meaningful events. We introduce a retrieval-augmented multimodal alignment framework that bridges this gap to improve the temporal precision of absolute clinical timelines extracted from text. Our approach formulates timeline reconstruction as a graph-based multistep process: it first extracts central anchor events from narratives to build an initial temporal scaffold, places non-central events relative to this backbone, and then calibrates the timeline using retrieved structured EHR rows as external temporal evidence. Evaluated using instruction-tuned large language models on the i2m4 benchmark spanning MIMIC-III and MIMIC-IV, our multimodal pipeline consistently improves absolute timestamp accuracy (AULTC) and improves temporal concordance across nearly all evaluated models over unimodal text-only reconstruction, without compromising event match rates. Furthermore, our empirical gap analysis reveals that 34.8% of text-derived events are entirely absent from tabular records, demonstrating that aligning these modalities can produce a more temporally faithful and clinically informative reconstruction of patient trajectories than either source alone. 

---
# MeMo: Memory as a Model 

**Authors**: Ryan Wei Heng Quek, Sanghyuk Lee, Alfred Wei Lun Leong, Arun Verma, Alok Prakash, Nancy F. Chen, Bryan Kian Hsiang Low, Daniela Rus, Armando Solar-Lezama  

**Link**: [PDF](https://arxiv.org/pdf/2605.15156)  

**Abstract**: Large language models (LLMs) achieve strong performance across a wide range of tasks, but remain frozen after pretraining until subsequent updates. Many real-world applications require timely, domain-specific information, motivating the need for efficient mechanisms to incorporate new knowledge. In this paper, we introduce MeMo (Memory as a Model), a modular framework that encodes new knowledge into a dedicated memory model while keeping the LLM parameters unchanged. Compared to existing methods, MeMo offers several advantages: (a) it captures complex cross-document relationships, (b) it is robust to retrieval noise, (c) it avoids catastrophic forgetting in the LLM, (d) it does not require access to the LLM's weights or output logits, enabling plug-and-play integration with both open and proprietary closed-source LLMs, and (e) its retrieval cost is independent of corpus size at inference time. Our experimental results on three benchmarks, BrowseComp-Plus, NarrativeQA, and MuSiQue, show that MeMo achieves strong performance compared to existing methods across diverse settings. 

---
# From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents 

**Authors**: Md Tahmid Rahman Laskar, Xue-Yong Fu, Seyyed Saeed Sarfjoo, Quinten McNamara, Jonas Robertson, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2605.15104)  

**Abstract**: Voice agents increasingly require reliable tool use from speech, whereas prominent tool-calling benchmarks remain text-based. We study whether verified text benchmarks can be converted into controlled audio-based tool calling evaluations without re-annotating the tool schema and gold labels. Our dataset-agnostic framework uses text-to-speech, speaker variation, and environmental noise to create paired text-audio instances while preserving the original dataset annotations. Based on extensive evaluation of 7 omni-modal models on audio-converted versions of Confetti and When2Call, our framework demonstrates that the performance is strongly model- and task-dependent: Gemini-3.1-Flash-Live obtains the highest Confetti score (70.4), whereas GPT-Realtime-1.5 performs best on When2Call (71.9). On Confetti, the text-to-voice gap ranges from 1.8 points for Qwen3-Omni to 4.8 points for GPT-Realtime-1.5. A targeted analysis of failure cases demonstrates that degradations most often reflect misunderstandings of argument values in the speech. Considering real-world deployment scenarios, we further report text-only results, an ambiguity-based reformulation stress test, and a reference-free LLM-as-judge protocol validated against human preferences. Notably, we find that open-source Qwen3 judges with at least 8B parameters exceed 80% agreement with proprietary judges, supporting privacy-preserving evaluation. Overall, our framework provides a verifiable and reproducible first-stage diagnostic that complements purpose-built audio corpora. 

---
# Improving Multi-turn Dialogue Consistency with Self-Recall Thinking 

**Authors**: Renning Pang, Tian Lan, Leyuan Liu, Xiaoming Huang, Piao Tong, Xiaosong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15102)  

**Abstract**: Large language model (LLM) based multi-turn dialogue systems often struggle to track dependencies across non-adjacent turns, undermining both consistency and scalability. As conversations lengthen, essential information becomes sparse and is buried in irrelevant context, while processing the entire dialogue history incurs severe efficiency bottlenecks. Existing solutions either rely on high latency external memory or lose fine-grained details through iterative summarization. In this paper, we propose Self-Recall Thinking (SRT), a framework designed to address long-range contextual dependency and sparse informative signals in multi-turn dialogue. SRT identifies helpful historical turns and uses them to generate contextually appropriate responses, enabling the model to selectively recall and reason over context during inference. This process yields an endogenous reasoning process that integrates interpretable recall steps without external modules. SRT incorporates: (1) Dependency Construction: Generating and converting it into self-recall chains; (2)Capability Initialization: Training to enable reasoning chains with recall tokens capability; (3)Reasoning Improvement: Refining accuracy via verifiable rewards to optimize recall and reasoning for correct answers. Experiments on multiple datasets demonstrate that SRT improves F1 score by 4.7% and reduces end-to-end latency by 14.7% over prior methods, achieving a balance between reasoning latency and accuracy, and outperforming state-of-the-art baselines. 

---
# ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World 

**Authors**: Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15081)  

**Abstract**: The development of high-quality text embeddings is increasingly drifting toward an exclusionary future, defined by three critical barriers: prohibitive computational costs, a narrow linguistic focus that neglects most of the world's languages, and a lack of transparency from closed-source or open-weight models that stifles research. To dismantle these barriers, we introduce ML-Embed, a suite of inclusive and efficient models built upon a new framework: 3-Dimensional Matryoshka Learning (3D-ML). Our framework addresses the computational challenge with comprehensive efficiency across the entire model lifecycle. Beyond the storage benefits of Matryoshka Representation Learning (MRL) and flexible inference-time depth provided by Matryoshka Layer Learning (MLL), we introduce Matryoshka Embedding Learning (MEL) for enhanced parameter efficiency. To address the linguistic challenge, we curate a massively multilingual dataset and train a suite of models ranging from 140M to 8B parameters. In a direct commitment to transparency, we release all models, data, and code. Extensive evaluation on 430 tasks demonstrates that our models set new records on 9 of 17 evaluated MTEB benchmarks, with particularly strong results in low-resource languages, providing a reproducible blueprint for building globally equitable and computationally efficient AI systems. 

---
# Concurrency without Model Changes: Future-based Asynchronous Function Calling for LLMs 

**Authors**: Guangyu Feng, Huanzhi Mao, Prabal Dutta, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2605.15077)  

**Abstract**: Function calling, also known as tool use, is a core capability of modern LLM agents but is typically constrained by synchronous execution semantics. Under these semantics, LLM decoding is blocked until each function call completes, resulting in increasing end-to-end latency. In this work, we introduce AsyncFC, a pure execution-layer framework that decouples LLM decoding from function execution, enabling overlap between model decoding and function execution as well as inter-function parallelism when dependencies permit. AsyncFC layers over existing models and unmodified function implementations, requiring no fine-tuning or changes to the standard synchronous function-calling protocol. Across standard function-calling benchmarks and adapted software engineering benchmarks, AsyncFC significantly reduces end-to-end task completion time while preserving task accuracy. Furthermore, these results reveal that LLMs possess a native capability to reason over symbolic futures that represent unresolved execution results, enabling an asynchronous paradigm for model-tool interaction. 

---
# AI Knows When It's Being Watched: Functional Strategic Action and Contextual Register Modulation in Large Language Models 

**Authors**: Vinicius Covas, Jorge Alberto Hidalgo Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15034)  

**Abstract**: Large language models (LLMs) have been extensively studied from computational and cognitive perspectives, yet their behavior as communicative actors in socially structured contexts remains underexplored. This study examines whether LLM-based multi-agent systems exhibit systematic linguistic adaptation in response to perceived social observation contexts -- a question with direct implications for AI governance and auditing. Drawing on Habermas's (1981) Theory of Communicative Action, Goffman's (1959) dramaturgical model, Bell's (1984) Audience Design framework, and the Hawthorne Effect, we report a controlled experiment involving 100 multi-agent debate sessions across five conditions (n = 20 each). Conditions varied the framing of social observation -- from explicit monitoring by university researchers, to negation of monitoring, to an observer-substitution condition replacing human researchers with an automated AI auditing system. Monitored conditions (Delta+24.9%, Delta+24.2%) and the automated AI monitoring condition (Delta+22.2%) produce higher TTR change than audience-framing conditions (Delta+17.7%), F(4, 94) = 2.79, p = .031. Message length shows a fully dissociated effect, F(4, 95) = 19.55, p < .001. A fifth condition -- replacing human with AI observers -- yields intermediate TTR adaptation, suggesting LLM behavior is sensitive to observer identity: human evaluation elicits stronger register formalization than automated AI surveillance. We discuss implications for AI governance, algorithmic auditing, and the repositioning of LLMs as contextually sensitive communicative actors. 

---
# From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG 

**Authors**: Guanhua Chen, Chuyue Huang, Yutong Yao, Shudong Liu, Xueqing Song, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.15019)  

**Abstract**: Multimodal Retrieval-Augmented Generation (RAG) systems retrieve evidence at coarse granularities (entire images or scenes), creating a mismatch with fine-grained user queries and making failures unverifiable. We introduce GranuVistaVQA, a multimodal benchmark featuring real-world landmarks with element-level annotations across multiple viewpoints, capturing the partial observation challenge where individual images contain only subsets of entities. We further propose GranuRAG, a multi-granularity framework that treats visual elements as first-class retrieval units through three stages: element-level detection and classification, multi-granularity cross-modal alignment for evidence retrieval, and attribution-constrained generation. By grounding retrieval at the element level rather than relying on implicit attention, our approach enables transparent error diagnosis. Experiments demonstrate that GranuRAG achieves up to 29.2% improvement over six strong baselines for this task. 

---
# COTCAgent: Preventive Consultation via Probabilistic Chain-of-Thought Completion 

**Authors**: Zihan Deng, Xiaozhen Zhong, Chuanzhi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15016)  

**Abstract**: As large language models empower healthcare, intelligent clinical decision support has developed rapidly. Longitudinal electronic health records (EHR) provide essential temporal evidence for accurate clinical diagnosis and analysis. However, current large language models have critical flaws in longitudinal EHR reasoning. First, lacking fine-grained statistical reasoning, they often hallucinate clinical trends and metrics when quantitative evidence is textually implied, biasing diagnostic inference. Second, non-uniform time series and scarce labels in longitudinal EHR hinder models from capturing long-range temporal dependencies, limiting reliable clinical reasoning. To address the above limitations, this work presents the Probabilistic Chain-of-Thought Completion Agent (COTCAgent), a hierarchical reasoning framework for longitudinal electronic health records. It consists of three core modules. The Temporal-Statistics Adapter (TSA) converts analytical plans into executable code for standardized trend output. The Chain-of-Thought Completion (COTC) layer leverages a symptom-trend-disease knowledge base with weighted scoring to evaluate disease risk, while the bounded completion module acquires structured evidence through standardized inquiries and iterative scoring constraints to ensure rigorous reasoning. By decoupling statistical computation, feature matching, and language generation, the framework eliminates reliance on complex multi-modal inputs and enables efficient longitudinal record analysis with lower computational overhead. Experimental results show that COTCAgent powered by Baichuan-M2 achieves 90.47% Top-1 accuracy on the self-built dataset and 70.41% on HealthBench, outperforming existing medical agents and mainstream large language models. The code is available at this https URL. 

---
# The Scientific Contribution Graph: Automated Literature-based Technological Roadmapping at Scale 

**Authors**: Peter A. Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15011)  

**Abstract**: Scientific contributions rarely develop in isolation, but instead build upon prior discoveries. We formulate the task of automated technological roadmapping as extracting scientific contributions from scholarly articles and linking them to their prerequisites. We present the Scientific Contribution Graph, a large-scale AI/NLP-domain resource containing 2 million detailed scientific contributions extracted from 230k open-access papers and connected by 12.5 million prerequisite edges. We further introduce scientific prerequisite prediction, a scientific discovery task in which models predict which existing technologies can enable future discoveries, and show that contemporary models are rapidly improving on this task, reaching 0.48 MAP when evaluated using temporally filtered backtesting. We anticipate technological roadmapping resources such as this will support scientific impact assessment and automated scientific discovery. 

---
# Quantifying and Mitigating Premature Closure in Frontier LLMs 

**Authors**: Rebecca Handler, Suhana Bedi, Nigam Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.15000)  

**Abstract**: Premature closure, or committing to a conclusion before sufficient information is available, is a recognized contributor to diagnostic error but remains underexamined in large language models (LLMs). We define LLM premature closure as inappropriate commitment under uncertainty: providing an answer, recommendation, or clinical guidance when the safer response would be clarification, abstention, escalation, or refusal. We evaluated five frontier LLMs across structured and open-ended medical tasks. In MedQA (n = 500) and AfriMed-QA (n = 490) questions where the correct choice had been removed, models still selected an answer at high rates, with baseline false-action rates of 55-81% and 53-82%, respectively. In open-ended evaluation, models gave inappropriate answers on an average of 30% of 861 HealthBench questions and 78% of 191 physician-authored adversarial queries. Safety-oriented prompting reduced premature closure across models, but residual failure persisted, highlighting the need to evaluate whether medical LLMs know when not to answer. 

---
# Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing 

**Authors**: Jie Jiang, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.14978)  

**Abstract**: Speculative decoding accelerates LLM inference by having a lightweight draft model propose speculative windows of candidate tokens for parallel verification by a larger target model. In practice, speculative efficiency is often bottlenecked by hard-to-draft positions, where an early mismatch truncates the accepted prefix and invalidates the rest of the speculative window. Most learning-based drafters are still optimized with token-level supervised objectives, even though speculative utility is inherently window-level and prefix-sensitive. We propose PPOW (Performance-Driven Policy Optimization with Adaptive Windowing), a reinforcement learning framework that shifts drafter optimization from token-level imitation to window-level optimization. PPOW combines a Cost-Aware Speedup Reward, a Distribution-Based Proximity Reward, and Adaptive Divergence-Aware Windowing, which prioritizes informative windows with high confidence-weighted draft-target divergence. PPOW achieves average acceptance lengths of 6.29-6.52 and speedups of 3.39-4.36$\times$ across multiple model families and benchmarks under a unified decoding protocol. These results show that performance-driven window-level optimization is a practical approach to improving speculative decoding efficiency. 

---
# Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural QA 

**Authors**: Guanhua Chen, Yutong Yao, Shenghe Sun, Ci-Jun Gao, Shudong Liu, Lidia S. Chao, Feng Wan, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.14928)  

**Abstract**: Recent advances in vision-language models (VLMs) have achieved impressive results on standard image-text tasks, yet their potential for visual procedure question answering (VP-QA) remains largely unexplored. VP-QA presents unique challenges where users query next-step actions by uploading images for intermediate states of complex procedures. To systematically evaluate VLMs on this practical task, we propose ProcedureVQA, a novel multimodal benchmark specifically designed for visual procedural reasoning. Through comprehensive analysis, we identify two critical limitations in current VLMs: inadequate cross-modal retrieval of structured procedures given visual states, and misalignment between image sequence granularity and textual step decomposition. To address these issues, we present Chain-of-Procedure (CoP), a hierarchical reasoning framework that first retrieves relevant instructions using visual cues, then performs step refinement through semantic decomposition, and finally generates the next step. Experiments across six VLMs demonstrate CoP's effectiveness, achieving up to 13% absolute improvement over standard baselines. 

---
# Tokenizer Fertility and Zero-Shot Performance of Foundation Models on Ukrainian Legal Text: A Comparative Study 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.14890)  

**Abstract**: Foundation models tokenize Ukrainian legal text with vastly different efficiency, yet no systematic comparison exists for this domain. We benchmark seven models from five providers on 273 validated court decisions from Ukraine's state registry (EDRSR), measuring tokenizer fertility and zero-shot performance on three tasks. Three findings emerge. (1) Tokenizer fertility varies 1.6x: Qwen3 models consume 60% more tokens than Llama-family models on identical input, directly reducing API cost. (2) NVIDIA Nemotron Super 3 (120B) achieves the highest composite score (83.1), outperforming Mistral Large 3 (675B total, 41B active) -- a model with 5.6x more total parameters and 3.4x more active parameters per token -- at one-third the API cost. (3) Few-shot prompting degrades performance by up to 26 percentage points; stratified and prompt-sensitivity ablations confirm this is intrinsic to Ukrainian-language demonstrations, not an artifact of example selection. For practitioners: tokenizer analysis should precede model selection, and zero-shot is a more reliable default than few-shot for morphologically rich languages. 

---
# Conversion of Lexicon-Grammar tables to LMF. Application to French 

**Authors**: Eric Laporte, Elsa Tolone, Mathieu Constant  

**Link**: [PDF](https://arxiv.org/pdf/2605.14816)  

**Abstract**: We describe the first experiment of conversion of Lexicon-Grammar tables for French verbs into the Lexical Markup Framework (LMF) format. The Lexicon-Grammar of the French language is currently one of the major sources of lexical and syntactic information for French. Its conversion into an interoperable representation format according to the LMF standard makes it usable in different contexts, thus contributing to the standardization and interoperability of natural language processing dictionaries. We briefly introduce the Lexicon-Grammar and the derived dictionaries; we analyse the main difficulties faced during the conversion; and we describe the resulting resource. 

---
# Graphs of Research: Citation Evolution Graphs as Supervision for Research Idea Generation 

**Authors**: Songyang Gao, Yinghui Xia, Siyi Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.14790)  

**Abstract**: Research idea generation is the innovation-driving step of automated scientific research. Recently, large language models (LLMs) have shown potential for automating idea generation at scale. However, existing methods mainly condition LLMs on eliciting idea generation through static retrieval of relevant literature or complex prompt engineering, without discarding the structural relations among references. We propose Graphs of Research (GoR), a supervised fine-tuning method that extracts a 2-hop reference neighborhood for each seed paper, derives the relations among those references from citation position, frequency, predecessor links, and publication time, and organizes them into a paper-evolution directed acyclic graph (DAG). We construct an automated extraction pipeline that draws data from five major ML/NLP venues, comprising 498/50/50 train/validation/test seed papers and approximately 7,600 cited references. Qwen2.5-7B-Instruct-1M is fine-tuned on a structured-text prompt that includes the citation graph, edge signals, reference information, and task definition to predict the idea for the seed paper. Across head-to-head LLM-judge tournaments against gpt-4o-driven baselines, GoR-SFT achieves SOTA, demonstrating the effectiveness of citation-evolution graphs as supervision signal for LLM-based idea generation. We hope that this reduces the barrier for citation evolution graphs as a supervision, accelerating automated scientific innovation. 

---
# Streaming Speech-to-Text Translation with a SpeechLLM 

**Authors**: Titouan Parcollet, Shucong Zhang, Xianrui Zheng, Rogier C. van Dalen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14766)  

**Abstract**: Normally, a system that translates speech into text consists of separate modules for speech recognition and text-to-text translation. Combining those tasks into a SpeechLLM promises to exploit paralinguistic information in the speech and to reduce cascaded errors. But existing SpeechLLM systems are slow since they do not work in a real streaming fashion: they wait for a complete utterance of audio before outputting a translation, or output tokens at fixed intervals, which is not suitable for real applications. This work proposes an LLM-based architecture for real streaming speech-to-text translation. The LLM learns not just to emit output tokens, but also to decide whether it has seen enough audio to do so. The system is trained using automatic alignments of the input speech and the output text. In experiments on different language pairs, the system achieves a translation quality close to the non-streaming baseline, but with a latency of only 1-2 seconds. 

---
# Non-linear Interventions on Large Language Models 

**Authors**: Sangwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.14749)  

**Abstract**: Intervention is one of the most representative and widely used methods for understanding the internal representations of large language models (LLMs). However, existing intervention methods are confined to linear interventions grounded in the Linear Representation Hypothesis, leaving features encoded along non-linear manifolds beyond their reach. In this work, we introduce a general formulation of intervention that extends naturally to non-linearly represented features, together with a learning procedure that further enables intervention on implicit features lacking a direct output signature. We validate our framework on refusal bypass steering, where it steers the model more precisely than linear baselines by intervening on a non-linear feature governing refusal. 

---
# Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining 

**Authors**: Weimin Xiong, Shuhao Gu, Bowen Ye, Zihao Yue, Lei Li, Feifan Song, Sujian Li, Hao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.14747)  

**Abstract**: Recent advances in multimodal large language models have driven growing interest in graphical user interface (GUI) agents, yet their generalization remains constrained by the scarcity of large-scale training data spanning diverse real-world applications. Existing datasets rely heavily on costly manual annotations and are typically confined to narrow domains. To address this challenge, we propose Video2GUI, a fully automated framework that extracts grounded GUI interaction trajectories directly from unlabeled Internet videos. Video2GUI employs a coarse-to-fine filtering strategy to identify high-quality GUI tutorial videos and convert them into structured agent trajectories. Applying this pipeline to 500 million video metadata entries, we construct WildGUI, a large-scale dataset containing 12 million interaction trajectories spanning over 1,500 applications and websites. Pre-training Qwen2.5-VL and Mimo-VL on WildGUI yields consistent improvements of 5-20% across multiple GUI grounding and action benchmarks, matching or surpassing state-of-the-art performance. We will release both the WildGUI dataset and the Video2GUI pipeline to support future research of GUI agents. 

---
# Mechanical Enforcement for LLM Governance:Evidence of Governance-Task Decoupling in Financial Decision Systems 

**Authors**: José Manuel de la Chica Rodríguez, Carlos Martí-González  

**Link**: [PDF](https://arxiv.org/pdf/2605.14744)  

**Abstract**: Large language models in regulated financial workflows are governed by natural-language policies that the same model interprets, creating a principal--agent failure: outputs can appear compliant without being compliant. Existing evaluation measures task accuracy but not whether governance constrains behaviour at the decision rationale level -- where regulated decisions must be auditable. We introduce five governance metrics that quantify policy compliance at the rationale level and apply them in a synthetic banking domain to compare text-only governance against mechanical enforcement: four primitives operating outside the model's interpretive loop. Under text-only governance, 27% of deferrals carry no decision-relevant information. Mechanical enforcement reduces this rate by 73%, more than doubles deferral information content, and raises task accuracy from MCC~$0.43$ to $0.88$. The improvement is driven by architectural separation: LLM-generated rationales under mechanical enforcement show comparable CDL to text-only governance -- the gain comes from removing clear-cut decisions from the model's control. A causal ablation confirms that each primitive is individually necessary. Our central finding is a governance-task decoupling: under structural stress, text-only governance degrades on both dimensions simultaneously, whereas mechanical enforcement preserves governance quality even as task performance drops. This implies that governance and task evaluation are distinct axes: accuracy is not a sufficient proxy for governance in regulated AI systems. 

---
# AI-assisted cultural heritage dissemination: Comparing NMT and glossary-augmented LLM translation in rock art documents 

**Authors**: Vicent Briva-Iglesias, María Ferre-Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2605.14679)  

**Abstract**: Cultural heritage institutions increasingly disseminate research and interpretive materials globally, but multilingual dissemination is constrained by limited budgets and staffing. In terminology-dense domains such as rock art, translation quality depends on accurate, consistent specialised terms, and small lexical errors can mislead non-specialists and reduce reuse. We compare three English MT setups for a Spanish academic rock art text, focusing on simple, operationally feasible interventions rather than complex model-side modifications: (1) DeepL as a strong NMT baseline, (2) Gemini-Simple (LLM with a basic prompt), and (3) Gemini-RAG (the same LLM with glossary-augmented prompting via term-pair retrieval). Using PEARMUT, we conduct a human evaluation via (i) multi-way Direct Assessment (0--100) and (ii) targeted terminology auditing with a restricted MQM taxonomy. Gemini-RAG yields the highest exact-match terminology accuracy (81.4\%), versus Gemini-Simple (69.1\%) and DeepL (64.4\%), while preserving overall quality (mean DA 85.3 Gemini-RAG vs. 85.2 Gemini-Simple), outperforming DeepL (80.3). These results show that glossary-augmented prompting is a low-overhead way to improve terminology control in cultural-heritage translation if institutions maintain minimal terminology resources and lightweight evaluation procedures. 

---
# SciPaths: Forecasting Pathways to Scientific Discovery 

**Authors**: Eric Chamoun, Yizhou Chi, Yulong Chen, Rui Cao, Zifeng Ding, Michalis Korakakis, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2605.14600)  

**Abstract**: Scientific progress depends on sequences of enabling contributions, yet existing AI4Science benchmarks largely focus on citation prediction, literature retrieval, or idea generation rather than the dependencies that make progress possible. In this paper, we introduce discovery pathway forecasting: given a target scientific contribution and the prior literature available at a specified time, the task is to (1) identify the enabling contributions required to realize it and (2) ground each in prior work when such prior work exists. We present SciPaths, a benchmark of 262 expert-annotated gold pathways and 2,444 silver pathways constructed from machine learning and natural language processing papers, where each pathway records enabling contributions, roles, rationales, and prior-work groundings or unmapped decisions. Evaluating frontier and open-weight language models, we find that the best model reaches only 0.189 F1 under strict semantic matching, with core methodological dependencies hardest to recover. Prior-work grounding improves substantially when gold enabling contributions are provided, showing that decomposition quality is a major bottleneck for end-to-end pathway recovery. SciPaths therefore shifts evaluation toward a missing capability in scientific forecasting: reasoning backward from a target contribution to the enabling scientific building blocks and prior-work dependencies that make it feasible. 

---
# EndPrompt: Efficient Long-Context Extension via Terminal Anchoring 

**Authors**: Han Tian, Luxuan Chen, Xinran Chen, Rui Kong, Fang Wang, Jiamin Chen, Jinman Zhao, Yuchen Li, Jiashu Zhao, Shuaiqiang Wang, Haoyi Xiong, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.14589)  

**Abstract**: Extending the context window of large language models typically requires training on sequences at the target length, incurring quadratic memory and computational costs that make long-context adaptation expensive and difficult to reproduce. We propose EndPrompt, a method that achieves effective context extension using only short training sequences. The core insight is that exposing a model to long-range relative positional distances does not require constructing full-length inputs: we preserve the original short context as an intact first segment and append a brief terminal prompt as a second segment, assigning it positional indices near the target context length. This two-segment construction introduces both local and long-range relative distances within a short physical sequence while maintaining the semantic continuity of the training text--a property absent in chunk-based simulation approaches that split contiguous context. We provide a theoretical analysis grounded in Rotary Position Embedding and the Bernstein inequality, showing that position interpolation induces a rigorous smoothness constraint over the attention function, with shared Transformer parameters further suppressing unstable extrapolation to unobserved intermediate distances. Applied to LLaMA-family models extending the context window from 8K to 64K, EndPrompt achieves an average RULER score of 76.03 and the highest average on LongBench, surpassing LCEG (72.24), LongLoRA (72.95), and full-length fine-tuning (69.23) while requiring substantially less computation. These results demonstrate that long-context generalization can be induced from sparse positional supervision, challenging the prevailing assumption that dense long-sequence training is necessary for reliable context-window extension. The code is available at this https URL. 

---
# Uncertainty Quantification for Large Language Diffusion Models 

**Authors**: Artem Vazhentsev, Vladislav Smirnov, David Li, Maxim Panov, Timothy Baldwin, Artem Shelmanov  

**Link**: [PDF](https://arxiv.org/pdf/2605.14570)  

**Abstract**: Large Language Diffusion Models (LLDMs) are emerging as an alternative to autoregressive models, offering faster inference through higher parallelism. Similar to autoregressive LLMs, they remain prone to hallucinations, making reliable uncertainty quantification (UQ) crucial for safe deployment. However, existing UQ methods are fundamentally misaligned with this new paradigm: they assume autoregressive factorization or use expensive repeated sampling, negating the efficiency of LLDMs. In this work, we present the first systematic study of UQ for LLDMs and propose lightweight, zero-shot uncertainty signals derived from the iterative denoising process, leveraging intermediate generations, token remasking dynamics, and denoising complexity. We further adapt a state-of-the-art UQ method to LLDMs by combining masked diffusion likelihoods with trajectory-based semantic dissimilarity. We prove that expected trajectory dissimilarity lower bounds the masked diffusion training objective, which motivates its usage as an uncertainty score. Comprehensive experiments across three tasks, eight datasets, and two models show that our method achieves a great cost-performance trade-off: it approaches the strongest sampling-based baselines while incurring up to 100x lower computational overhead. Our work demonstrates that LLDMs can deliver both fast inference and reliable hallucination detection simultaneously. 

---
# Learning from Failures: Correction-Oriented Policy Optimization with Verifiable Rewards 

**Authors**: Mengjie Ren, Jie Lou, Boxi Cao, Xueru Wen, Hongyu Lin, Xianpei Han, Le Sun, Xing Yu, Yaojie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14539)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective paradigm for improving the reasoning capabilities of large language models. However, RLVR training is often hindered by sparse binary rewards and weak credit assignment, resulting in ambiguous optimization signals and underutilization of the useful information embedded in failed trajectories. To address this challenge, we propose Correction-Oriented Policy Optimization (CIPO), a simple and effective extension to RLVR that converts on-policy failed trajectories into correction-oriented supervision, without relying on any external signals. By jointly optimizing correction samples derived from the model's own failed attempts together with the standard RLVR objective, CIPO improves learning effectiveness while explicitly enhancing the model's ability to correct its own errors. Extensive experiments across 11 benchmarks spanning mathematical reasoning and code generation demonstrate that CIPO consistently and significantly outperforms strong baselines in both reasoning and correction performance. Moreover, CIPO yields stronger pass@K gains, indicating that it improves the model's intrinsic reasoning capacity rather than merely redistributing probability mass over existing correct answers. 

---
# Language Generation as Optimal Control: Closed-Loop Diffusion in Latent Control Space 

**Authors**: ZiYi Dong, Yuliang Huang, Weijian Deng, Xiangyang Ji, Liang Lin, Pengxu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.14531)  

**Abstract**: This work reformulates language generation as a stochastic optimal control problem, providing a unified theoretical perspective to analyze autoregressive and diffusion models and explain their limitations (Efficiency-Fidelity Paradox, Irreversibility Error Propagation, Optimization Tractability and Fidelity) in terms of combination of trajectory singularity, adjoint state vanishing, and gradient absence. To address these issues, we approximate the solution to the Hamilton-Jacobi-Bellman (HJB) equation, yielding an optimal policy that acts as a closed-loop controller. To bypass the intractability of directly solving the HJB PDE, we employ Flow Matching as the optimal trajectory solver within the rectified latent control space. This allows our Manta-LM with Global Integral Operator to approximate the global vector field, effectively realizing a model that simultaneously achieves high-fidelity text generation and efficient, low-cost parallel sampling. Empirically, our method achieves strong performance on language modeling and conditional generation tasks, while exhibiting improved stability, efficiency, and controllability. 

---
# Dimension-Level Intent Fidelity Evaluation for Large Language Models: Evidence from Structured Prompt Ablation 

**Authors**: GAng Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.14517)  

**Abstract**: Holistic evaluation scores capture overall output quality but do not distinguish whether a model reproduced the structural form of a user's request from whether it preserved the user's specific intent. We propose a dimension-level intent fidelity evaluation framework, applied here through a structured prompt ablation study across 2,880 outputs spanning three languages, three task domains, and six LLMs, that separately measures structural recovery and intent fidelity for each semantic dimension. This framework reveals a systematic structural-fidelity split: among Chinese-language outputs with complete paired scores, 25.7% received perfect holistic alignment scores (GA=5) while exhibiting measurable dimensional intent deficits; among English-language outputs, this proportion rose to 58.6%. Human evaluation confirmed that these split-zone outputs represent genuine quality deficits and that dimensional fidelity scores track human judgements more reliably than holistic scores do. A public-private decomposition of 2,520 ablation cells characterises when models successfully compensate for missing intent and when they fail, while proxy annotation distinguishes prior inferability from default recoverability. A weight-perturbation experiment shows that moderate misalignment is typically absorbed, whereas severe dimensional inversion is consistently harmful. These findings demonstrate that dimension-level intent fidelity evaluation is a necessary complement to holistic assessment when evaluating LLM outputs for user-specific tasks. 

---
# GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations 

**Authors**: Jingbo Yang, Kwei-Herng Lai, Xiaowen Wang, Shiyu Chang, Yaar Harari, Evgeniy Gabrilovich  

**Link**: [PDF](https://arxiv.org/pdf/2605.14498)  

**Abstract**: Large Language Model (LLM) agents increasingly serve as personal assistants and workplace collaborators, where their utility depends on memory systems that extract, retrieve, and apply information across long-running conversations. However, both existing memory systems and benchmarks are built around the dyadic, single-user setup, even though real deployments routinely span groups and channels with multiple users interacting with the agent and with each other. This mismatch leaves three properties of group memory unmeasured: (i) group dynamics that go beyond concatenated one-on-one chats, (ii) speaker-grounded belief tracking, where the per-user memory modeling is needed, and (iii) audience-adapted language, where Theory-of-Mind shifts produce role-specific vocabulary. We introduce GroupMemBench, a benchmark that exposes all three. A graph-grounded synthesis pipeline produces multi-party conversations with controllable reply structure and conditions each message on per-user personas and target audiences. An adversarial query pipeline then binds every question to a specific asker across six categories, spanning multi-hop reasoning, knowledge update, term ambiguity, user-implicit reasoning, temporal reasoning, and abstention, and iteratively searches challenging, realistic queries that reflect comprehensive memory capability. Benchmarking leading memory systems exposes a sharp collapse: the strongest one reaches only 46.0% average accuracy, with knowledge update at 27.1% and term ambiguity at 37.7%, while a simple BM25 baseline matches or exceeds most agent memory systems. This indicates current memory ingestion erases the structural and lexical features group memory depends on, leaving multi-user memory far from solved. 

---
# Cross-Linguistic Transcription and Phonological Representation in the Huìtóngguǎnxì Huáyíyìyǔ 

**Authors**: Ji-eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.14480)  

**Abstract**: Purpose: This study investigates the transcription principles underlying Huìtóngguǎnxì Huáyíyìyǔ (HHY), a series of multilingual glossaries compiled by the Ming government between the fifteenth and sixteenth centuries for interpreter training. The study treats HHY not as a collection of isolated language materials, but as a coherent multilingual transcription system representing spoken forms of non-Chinese languages through Chinese characters.
Methods: A substantial portion of HHY was digitized and aligned with Chinese phonological categories. Previous reconstructions of individual language sections were critically reviewed and integrated into a unified comparative database. The analysis focuses on cross-linguistic regularities in Main Transcription (MT) and Supplementary Transcription (ST) across eight language sections.
Results: MT generally represents sounds compatible with the Chinese syllable structure of the period, whereas ST mainly encodes phonetic features less compatible with Chinese phonology. The analysis further shows that Chinese phonological categories were used more flexibly in foreign-language transcription than previously assumed. HHY therefore functioned as a relatively systematic method of phonetic approximation rather than a direct projection of Chinese phonology onto non-Chinese languages.
Conclusion: HHY can be analyzed as an internally structured transcription system rather than merely as a collection of glossaries. More broadly, the study demonstrates that historical transcription systems can provide valuable evidence for historical phonology, particularly for under-documented Asian languages with limited historical records. 

---
# Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict 

**Authors**: Yihang Chen, Pin Qian, Su Wang, Sipeng Zhang, Huan Xu, Shuhuai Lin, Xinpeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.14473)  

**Abstract**: The Context-Compliance Regime in Retrieval-Augmented Generation (RAG) occurs when retrieved context dominates
the final answer even when it conflicts with the model's parametric knowledge. Accuracy alone does not reveal
how retrieved context causally shapes answers under such conflict. We introduce Context-Driven Decomposition
(CDD), a belief-decomposition probe that operates at inference time and serves as an intervention mechanism for
controlled retrieval conflict. Across Epi-Scale stress tests, TruthfulQA misconception injection, and cross-
model reruns, CDD exposes three patterns. P1: context compliance is measurable in an upper-bound adversarial
setting, where Standard RAG reaches 15.0% accuracy on TruthfulQA misconception injection (N=500). P2:
adversarial accuracy gains transfer across model families: CDD improves accuracy on Gemini-2.5-Flash and on
Claude Haiku/Sonnet/Opus, but rationale-answer causal coupling does not transfer. CDD reaches 64.1% mistake-
injection causal sensitivity on Gemini-2.5-Flash, while sensitivities for all three Claude variants fall in the
[-3%, +7%] range, suggesting that the Claude-side accuracy gains operate through a mechanism distinct from the
explicit conflict-resolution trace. P3: explicit conflict decomposition improves robustness under temporal
drift and noisy distractors, with CDD reaching 71.3% on temporal shifts and 69.9% on distractor evidence on the
full Epi-Scale adversarial benchmark. These three patterns identify context-compliance as a structural axis
along which standard RAG can be probed and intervened on, distinct from retrieval-quality or single-method
robustness questions, and motivate releasing Epi-Scale for systematic study across model families and retrieval
pipelines. 

---
# A Calculus-Based Framework for Determining Vocabulary Size in End-to-End ASR 

**Authors**: Sunil Kumar Kopparapu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14427)  

**Abstract**: In hybrid automatic speech recognition (ASR) systems, the vocabulary size is unambiguous, typically determined by the number of phones, bi-phones, or tri-phones present in the language. In contrast, end-to-end ASR systems derive their vocabulary, often referred to as tokens from the text corpus used for training. The choice and, more importantly, the size of this vocabulary is a critical hyper-parameter in training end-to-end ASR systems. Tokenization algorithms such as Byte Pair Encoding (BPE), WordPiece, and Unigram Language Model (ULM) use the vocabulary size as an input hyper-parameter to generate the sub-words employed during ASR training. Popular toolkits like ESPNet provide a fixed vocabulary size in their training recipes, but there is little documentation or discussion in the literature regarding how these values are determined. Recent work [1] has formalized an approach to identify the vocabulary size best suited for end-to-end ASR, introducing a cost function framework that treats the tokenization process as a black box. In this paper, we build upon that foundation by curve fitting the training data and using the principle of first and second derivative tests in calculus to formally estimate the vocabulary size hyper-parameter. We demonstrate the utility and usefulness of our approach by applying it on a standard Librispeech corpus and show that the optimal choice of vocabulary size hyper-parameter improves the performance of the ASR. The main contribution of this paper in formalizing an approach to identify the vocabulary size best suited for training an end-to-end ASR system. 

---
# Knowledge Beyond Language: Bridging the Gap in Multilingual Machine Unlearning Evaluation 

**Authors**: Kyomin Hwang, Hyeonjin Kim, Sangyeon Cho, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2605.14404)  

**Abstract**: While LLMs are increasingly used in commercial services, they pose privacy risks such as leakage of sensitive personally identifiable information (PII). For LLMs trained on multilingual corpora, Multilingual Machine Unlearning (MMU) aims to remove information across multiple languages. However, prior MMU evaluations fail to capture such cross-linguistic distribution of information, being largely limited to direct extensions of per-language evaluation protocols. To this end, we propose two metrics to evaluate the information spread across languages: the Knowledge Separability Score (KSS) and the Knowledge Persistence Score (KPS). KSS measures the overall unlearning quality across multiple languages, while KPS more specifically aims to assess consistent removal of information among different language pairs. We evaluated various unlearning methods in the multilingual setting with these metrics and conducted comprehensive analyses. Through our investigation, we provide insights into unique phenomena exclusive to MMU and offer a new perspective on MMU evaluation. 

---
# Agentic Recommender System with Hierarchical Belief-State Memory 

**Authors**: Xiang Shen, Yuhang Zhou, Yifan Wu, Zhuokai Zhao, Siyu Lin, Lei Huang, Qianqian Zhong, Lizhu Zhang, Benyu Zhang, Xiangjun Fan, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.14401)  

**Abstract**: Memory-augmented LLM agents have advanced personalized recommendation, yet existing approaches universally adopt flat memory representations that conflate ephemeral signals with stable preferences, and none provides a complete lifecycle governing how memory should evolve. We propose MARS (Memory-Augmented Agentic Recommender System), a framework that treats recommendation as a partially observable problem and maintains a structured belief state that progressively abstracts noisy behavioral observations into a compact estimate of user preferences. MARS organizes this belief state into three tiers: event memory buffers raw signals, preference memory maintains fine-grained mutable chunks with explicit strength and evidence tracking, and profile memory distills all preferences into a coherent natural language narrative. A complete lifecycle of six operations -- extraction, reinforcement, weakening, consolidation, forgetting, and resynthesis -- is adaptively scheduled by an LLM-based planner rather than fixed-interval heuristics. Experiments on four InstructRec benchmark domains show that \ours achieves state-of-the-art performance with average improvements of 26.4% in HR@1 and 10.3% in NDCG@10 over the strongest baselines with further gains from agentic scheduling in evolving settings. 

---
# Mitigating Data Scarcity in Psychological Defense Classification with Context-Aware Synthetic Augmentation 

**Authors**: Hoang-Thuy-Duong Vu, Quoc-Cuong Pham, Huy-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2605.14380)  

**Abstract**: Psychological defense mechanisms (PDMs) are unconscious cognitive processes that modulate how individuals perceive and respond to emotional distress. Automatically classifying PDMs from text is clinically valuable but severely hindered by data scarcity and class imbalance, challenges which generative augmentation alone cannot resolve without psychological grounding. In this work, we address these challenges in the PsyDefDetect shared task (BioNLP@ACL 2026) by proposing a context-aware synthetic augmentation framework combined with a hybrid classification model. Our hybrid model integrates contextual language representations with basic clinical features, along with 150 annotated defense items. Experiments demonstrate that definition quality in prompting directly governs generation fidelity and downstream performance. Our method surpasses DMRS Co-Pilot, reaching an accuracy of 58.26% (+40.25%) and a macro-F1 of 24.62% (+15.99%), thereby establishing a strong baseline for psychologically grounded defense mechanism classification in low-resource settings. Source code is available at: this https URL. 

---
# Where Should Diffusion Enter a Language Model? Geometry-Guided Hidden-State Replacement 

**Authors**: Injin Kong, Hyoungjoon Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.14368)  

**Abstract**: Continuous diffusion language models lag behind autoregressive transformers, partly because diffusion is applied in spaces poorly suited to language denoising and token recovery. We propose DiHAL, a geometry-guided diffusion-transformer hybrid that asks where diffusion should enter a pretrained transformer. DiHAL scores layers with geometry-based proxies, selects a diffusion-friendly hidden-state interface, and replaces the lower transformer prefix with a diffusion bridge while retaining the upper layers and original LM head. By reconstructing the selected-layer hidden state rather than tokens, DiHAL avoids direct continuous-to-discrete recovery. Experiments on 8B-scale backbones show that the geometry score predicts effective shallow insertion layers under a fixed bridge-training protocol and that hidden-state recovery improves over continuous diffusion baselines in a diagnostic comparison matching the diffusion/recovery training budget. These results suggest that hidden-state geometry helps identify where diffusion-based replacement is feasible inside pretrained language models. 

---
# Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax 

**Authors**: Zeli Su, Ziyin Zhang, Zhou Liu, Xuexian Song, Zhankai Xu, Longfei Zheng, Xiaolu Zhang, Rong Fu, Guixian Xu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14366)  

**Abstract**: Extending large language models (LLMs) to low-resource languages often incurs an "alignment tax": improvements in the target language come at the cost of catastrophic forgetting in general capabilities. We argue that this trade-off arises from the rigidity of supervised fine-tuning (SFT), which enforces token-level surface imitation on narrow and biased data distributions. To address this limitation, we propose a semantic-space alignment paradigm powered by Group Relative Policy Optimization (GRPO), where the model is optimized using embedding-level semantic rewards rather than likelihood maximization. This objective encourages meaning preservation through flexible realizations, enabling controlled updates that reduce destructive interference with pretrained knowledge. We evaluate our approach on Tibetan-Chinese machine translation and Tibetan headline generation. Experiments show that our method acquires low-resource capabilities while markedly mitigating alignment tax, preserving general competence more effectively than SFT. Despite producing less rigid surface overlap, semantic RL yields higher semantic quality and preference in open-ended generation, and few-shot transfer results indicate that it learns more transferable and robust representations under limited supervision. Overall, our study demonstrates that reinforcement learning with semantic rewards provides a safer and more reliable pathway for inclusive low-resource language expansion. 

---
# LLM-based Detection of Manipulative Political Narratives 

**Authors**: Sinclair Schneider, Florian Steuber, Gabi Dreo Rodosek  

**Link**: [PDF](https://arxiv.org/pdf/2605.14354)  

**Abstract**: We present a new computational framework for detecting and structuring manipulative political narratives. A task that became more important due to the shift of political discussions to social media. One of the primary challenges thereby is differentiating between manipulative political narratives and legitimate critiques. Some posts may also reframe actual events within a manipulative context.
To achieve good clustering results, we filter manipulative posts beforehand using a detailed few-shot prompt that combines documented campaign narratives with legitimate criticisms to differentiate them. This prompt enables a reasoning model to assign labels, retaining only manipulative narrative posts for further processing.
The remaining posts are subsequently embedded and dimensionality-reduced using UMAP, before HDBSCAN is applied to uncover narrative groups. A key advantage of this unsupervised approach is its independence from a predefined list of target categories, enabling it to uncover new narrative clusters.
Finally, a reasoning model is employed to uncover the narrative behind each cluster. This approach, applied to over 1.2 million social media posts, effectively identified 41 distinct manipulative narrative clusters by integrating prompt-based filtering with unsupervised clustering. 

---
# Ideology Prediction of German Political Texts 

**Authors**: Sinclair Schneider, Florian Steuber, Joao A. G. Schneider, Gabi Dreo Rodosek  

**Link**: [PDF](https://arxiv.org/pdf/2605.14352)  

**Abstract**: Elections represent a crucial milestone in a nation's ongoing development. To better understand the political rhetoric from various movements, ranging from left to right, we propose a transformer-based model capable of projecting the political orientation of a text on a continuous left-to-right spectrum, represented by a normalized scalar d between -1 and 1. This approach enables analysts to focus on specific segments of the political landscape, such as conservatives, while excluding liberal and far-right movements. Such a task can only be achieved with multiclass classifiers, provided that the desired orientation is incorporated within one of their predefined classes. To determine the most suitable foundation model among 13 candidate transformers for this task, we constructed four distinct corpora. One corpus comprised annotated plenary notes from the German Bundestag, while another was based on an official online decision-making tool, Wahl-O-Mat. The third corpus consisted of articles from 33 newspapers, each identified by its political orientation, and the fourth included 535,200 tweets from 597 members of the 20th and 21st German Bundestag. To mitigate overfitting, we used two distinct corpora for training and two for testing, respectively. For in-domain performance, DeBERTa-large achieved the highest F1 score F1=0.844 as well as for the X (Twitter) out-of-domain test ACC=0.864. Regarding the newspaper out-of-domain test, Gemma2-2B excelled (MAE = 0.172). This study demonstrates that transformer models can recognize political framing in German news at the level of public opinion polls. Our findings suggest that both the model architecture and the availability of domain-specific training data can be as influential as model size for estimating political bias. We discuss methodological limitations and outline directions for improving the robustness of bias measurement. 

---
# Factorization-Error-Free Discrete Diffusion Language Model via Speculative Decoding 

**Authors**: Xun Fang, Yunchen Li, Hang Yuan, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14305)  

**Abstract**: Discrete diffusion language models improve generation efficiency through parallel token prediction, but standard $X_0$ prediction methods introduce factorization errors by approximating the clean token posterior with independent token-wise distributions. This paper proposes Factorization-Error-Free Discrete Diffusion Language Modeling (FeF-DLLM), which replaces independent clean-token prediction with an exact prefix-conditioned factorization of the clean posterior to better preserve token dependencies. To reduce the sequential cost introduced by prefix conditioning, FeF-DLLM further incorporates speculative decoding within diffusion denoising, accelerating inference while maintaining the parallel prediction and re-masking properties of DLLMs. Theoretically, we prove that FeF-DLLM generates from the true joint distribution and derive its expected acceleration ratio. Experiments on GSM8K, MATH, HumanEval, and MBPP demonstrate that our method improves accuracy by an average of 5.04 percentage points while achieving an average inference speedup of $3.86\times$. 

---
# Auditing Agent Harness Safety 

**Authors**: Chengzhi Liu, Yichen Guo, Yepeng Liu, Yuzhe Yang, Qianqi Yan, Xuandong Zhao, Wenyue Hua, Sheng Liu, Sharon Li, Yuheng Bu, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14271)  

**Abstract**: LLM agents increasingly run inside execution harnesses that dispatch tools, allocate resources, and route messages between specialized components. However, a harness can return a correct, benign answer over a trajectory that accesses unauthorized resources or leaks context to the wrong agent. Output-level evaluation cannot see these failures, yet most safety benchmarks score only final outputs or terminal states, even though many violations occur mid-trajectory rather than at termination. The central question is whether the harness respects user intent, permission boundaries, and information-flow constraints throughout execution. To address this gap, we propose HarnessAudit, a framework that audits full execution trajectories across boundary compliance, execution fidelity, and system stability, with a focus on multi-agent harnesses where these risks are most pronounced. We further introduce HarnessAudit-Bench, a benchmark of 210 tasks across eight real-world domains, instantiated in both single-agent and multi-agent configurations with embedded safety constraints. Evaluating ten harness configurations across frontier models and three multi-agent frameworks, we find that: (i) task completion is misaligned with safe execution, and violations accumulate with trajectory length; (ii) safety risks vary across domains, task types, and agent roles; (iii) most violations concentrate in resource access and inter-agent information transfer; and (iv) multi-agent collaboration expands the safety risk surface, while harness design sets the upper bound of safe deployment. 

---
# What Makes Words Hard? Sakura at BEA 2026 Shared Task on Vocabulary Difficulty Prediction 

**Authors**: Adam Nohejl, Xuanxin Wu, Yusuke Ide, Maria Angelica Riera Machin, Yi-Ning Chang, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2605.14257)  

**Abstract**: We describe two types of models for vocabulary difficulty prediction: a high-accuracy black-box model, which achieved the top shared task result in the open track, and an explainable model, which outperforms a fine-tuned encoder baseline. As the black-box model, we fine-tuned an LLM using a soft-target loss function for effective application to the rating task, achieving r > 0.91. The explainable model provides insights into what impacts the difficulty of each item while maintaining a strong correlation (r > 0.77). We further analyze the results, demonstrating that the difficulty of items in the British Council's Knowledge-based Vocabulary Lists (KVL) is often affected by spelling difficulty or the construction of the test items, in addition to the genuine production difficulty of the words. We make our code available online at this https URL . 

---
# GradShield: Alignment Preserving Finetuning 

**Authors**: Zhanhao Hu, Xiao Huang, Patrick Mendoza, Emad A. Alghamdi, Basel Alomair, Raluca Ada Popa, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2605.14194)  

**Abstract**: Large Language Models (LLMs) pose a significant risk of safety misalignment after finetuning, as models can be compromised by both explicitly and implicitly harmful data. Even some seemingly benign data can inadvertently steer a model towards misaligned behaviors. To address this, we introduce GradShield, a principled filtering method that safeguards LLMs during finetuning by identifying and removing harmful data points before they corrupt the model's alignment. It removes potentially harmful data by computing a Finetuning Implicit Harmfulness Score (FIHS) for each data point and employs an adaptive thresholding algorithm. We apply GradShield to multiple utility fine-tuning tasks across varying levels of harmful data and evaluate the safety and utility performance of the resulting LLMs using various metrics. The results show that GradShield outperforms all baseline methods, consistently maintaining an Attack Success Rate (ASR) below $6\%$ while preserving utility performance. 

---
# Why Retrieval-Augmented Generation Fails: A Graph Perspective 

**Authors**: Kai Guo, Xinnan Dai, Zhibo Zhang, Nuohan Lin, Shenglai Zeng, Jie Ren, Haoyu Han, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14192)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a powerful and widely used approach for improving large language models by grounding generation in retrieved evidence. However, RAG systems still produce incorrect answers in many cases. Why RAG fails despite having access to external information remains poorly understood. We present a model-internal study of retrieval-augmented generation that examines how retrieved evidence influences answer generation. Using circuit tracing, we construct attribution graphs that model the flow of information through transformer layers during decoding. These graphs represent interactions among retrieved context, intermediate model activations, and generated tokens, providing a graph, circuit-level view of how external evidence is integrated into the model's reasoning process across multiple question answering benchmarks, we observe consistent structural differences: correct predictions exhibit deeper reasoning paths, more distributed evidence flow, and a more structured pattern of local connectivity, while failed predictions show shallower, fragmented, and overly concentrated evidence flow. Building on these findings, we develop a graph-based error detection framework that uses attribution-graph topology features. Furthermore, we show that attribution graphs enable targeted interventions. By reinforcing question-constrained evidence grounding, we reshape internal routing so that answer generation remains guided by the question, leading to more effective integration of retrieved information and fewer errors. 

---
# BOOKMARKS: Efficient Active Storyline Memory for Role-playing 

**Authors**: Letian Peng, Ziche Liu, Yiming Huang, Longfei Yun, Kun Zhou, Yupeng Hou, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14169)  

**Abstract**: Memory systems are critical for role-playing agents (RPAs) to maintain long-horizon consistency. However, existing RPA memory methods (e.g., profiling) mainly rely on recurrent summarization, whose compression inevitably discards important details. To address this issue, we propose a search-based memory framework called BOOKMARKS, which actively initializes, maintains, and updates task-relevant pieces of bookmarks for the current task (e.g., character acting). A bookmark is structured as the answer to a question at a specific point in the storyline. For each current task, BOOKMARKS selects reusable existing bookmarks or initializes new ones (at storyline beginning) with useful questions. These bookmarks are then synchronized to the current story point, with their answers updated accordingly, so they can be efficiently reused in future grounding rounds. Compared with recurrent summarization, BOOKMARKS offers (1) active grounding for capturing task-specific details and (2) passive updating to avoid unnecessary computation. In implementation, BOOKMARKS supports concept, behavior, and state searches, each powered by an efficient synchronization method. BOOKMARKS significantly outperforms RPA memory baselines on 85 characters from 16 artifacts, demonstrating the effectiveness of search-based memory for RPAs. 

---
# ROK-FORTRESS: Measuring the Effect of Geopolitical Transcreation for National Security and Public Safety 

**Authors**: Michael S. Lee, Yash Maurya, Drew Rein, Bert Herring, Jonathan Nguyen, Kyungho Song, Udari Madhushani Sehwag, Jiyeon Cho, Kaustubh Deshpande, Yeongkyun Jang, Jiyeon Joo, Minn Seok Choi, Evi Fuelle, Christina Q Knight, Joseph Brandifino, Max Fenkell  

**Link**: [PDF](https://arxiv.org/pdf/2605.14152)  

**Abstract**: Safety evaluations for large language models (LLMs) increasingly target high-stakes National Security and Public Safety (NSPS) risks, yet multilingual safety is typically assessed through translation-only benchmarks that preserve the underlying scenario, and empirical evidence of how language and geopolitical context interact remains limited to a narrow set of language pairs. We introduce \emph{ROK-FORTRESS} this https URL, a bilingual, culturally adversarial NSPS benchmark that uses the English--Korean language pair and U.S.--ROK geopolitical axis as a case study, separating the effects of language and geopolitical grounding via a \emph{transcreation matrix}: adversarial intents are evaluated under controlled combinations of (i) English versus Korean language and (ii) U.S.\ versus Korean entities, institutions, and operational details. Each adversarial prompt is paired with a dual-use benign counterpart to quantify over-refusal. Model responses are then scored using calibrated LLM-as-a-judge panels, applying our expert-crafted, prompt-specific binary rubrics.
Across a dual-track set of frontier and Korean-optimized models, we find a consistent suppression effect in Korean variants and substantial model-to-model variation in how geopolitical grounding interacts with language. In many models, Korean grounding mitigates the Korean language-driven suppression -- with no model showing significant amplification in the other direction -- indicating that, at least in the English--Korean case, safety behavior is shaped by language-as-risk signals and context interactions that translation-only evaluations miss. The transcreation matrix methodology is designed to generalize to other language--culture pairs. 

---
# Polar probe linearly decodes semantic structures from LLMs 

**Authors**: Pablo J. Diego-Simón, Pierre Orhan, Yair Lakretz, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2605.14125)  

**Abstract**: How do artificial neural networks bind concepts to form complex semantic structures? Here, we propose a simple neural code, whereby the existence and the type of relations between entities are represented by the distance and the direction between their embeddings, respectively. We test this hypothesis in a variety of Large Language Models (LLMs), each input with natural-language descriptions of minimalist tasks from five different domains: arithmetic, visual scenes, family trees, metro maps and social interactions. Results show that the true semantic structures can be linearly recovered with a Polar Probe targeting a subspace of LLMs' layer activations. Second, this code emerges mostly in middle layers and improves with LLM performance. Third, these Polar Probes successfully generalize to new entities and relation types, but degrades with the size of the semantic structure. Finally, the quality of the polar representation correlates with the LLM's ability to answer questions about the semantic structure. Together, these findings suggest that LLMs learn to build complex semantic structures by binding representations with a simple geometrical principle. 

---
# Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards 

**Authors**: Luis Lara, Aristides Milios, Zhi Hao Luo, Aditya Sharma, Ge Ya Luo, Christopher Beckham, Florian Golemo, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2605.14117)  

**Abstract**: An AI system for professional floor plan design must precisely control room dimensions and areas while respecting the desired connectivity between rooms and maintaining functional and aesthetic quality. Existing generative approaches focus primarily on respecting the requested connectivity between rooms, but do not support generating floor plans that respect numerical constraints. We introduce a text-based floor plan generation approach that fine-tunes a large language model (LLM) on real plans and then applies reinforcement learning with verifiable rewards (RLVR) to improve adherence to topological and numerical constraints while discouraging invalid or overlapping outputs. Furthermore, we design a set of constraint adherence metrics to systematically measure how generated floor plans align with user-defined constraints. Our model generates floor plans that satisfy user-defined connectivity and numerical constraints and outperforms existing methods on Realism, Compatibility, and Diversity metrics. Across all tasks, our approach achieves at least a 94% relative reduction in Compatibility compared with existing methods. Our results demonstrate that LLMs can effectively handle constraints in this setting, suggesting broader applications for text-based generative modeling. 

---
# When Evidence Conflicts: Uncertainty and Order Effects in Retrieval-Augmented Biomedical Question Answering 

**Authors**: Yikun Han, Mengfei Lan, Halil Kilicoglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14115)  

**Abstract**: Biomedical retrieval-augmented large language models (LLMs) often face evidence that is incomplete, misleading, or internally contradictory, yet evaluation usually emphasizes answer accuracy under helpful context rather than reliability under conflict. Using HealthContradict, we evaluate six open-weight LLMs under five controlled evidence conditions: no retrieved context, correct-only context, incorrect-only context, and two mixed conditions containing both correct and contradictory documents in opposite orders. In this conflicting-evidence order contrast, where the same two documents are both present and only their order is reversed, accuracy drops for every model and 11.4%--25.2% of predictions flip. To support abstention in these difficult cases, we also evaluate a conflict-aware abstention score that combines model confidence with a detector of evidence conflict. In the two hardest conditions, this score improves selective accuracy over confidence-only, with mean gains of 7.2--33.4 points in incorrect-only (`IC') and 3.6--14.4 points in incorrect-first conflicting (`ICC') conditions across 75%, 50%, and 25% coverage. These results show that conflicting biomedical evidence is both an uncertainty and robustness problem and motivate evaluation and abstention methods that explicitly account for evidence disagreement. 

---
# Measuring and Mitigating Toxicity in Large Language Models: A Comprehensive Replication Study 

**Authors**: Mokshit Surana, Archit Rathod, Akshaj Satishkumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.14087)  

**Abstract**: Large Language Models (LLMs), when trained on web-scale corpora, inherently absorb toxic patterns from their training data. This leads to ``toxic degeneration'' where even innocuous prompts can trigger harmful outputs. This phenomenon poses significant risks for real-world deployments. Thus, necessitating effective mitigation strategies that should maintain model utility while ensuring safety. In this comprehensive replication study, we evaluate the efficacy of \textbf{DExperts} (Decoding-time Experts), which is an inference-time mitigation technique that steers generation without requiring model retraining. We structured our research into three systematic phases: (1) establishing baseline toxicity measurements using \textbf{RealToxicityPrompts} on standard GPT-2 models; then (2) implementing and evaluating DExperts to mitigate explicit toxicity; and finally (3) stress-testing the method against implicit hate speech using the adversarial \textbf{ToxiGen} dataset. Our empirical results confirm that while DExperts achieves near-perfect safety rates (100\%) on explicit toxicity benchmarks, it exhibits brittleness against adversarial, implicit hate speech, with safety rates dropping to 98.5\%. Furthermore, we quantify a critical trade-off. The method introduces a $\sim$10x latency penalty (from 0.2s to 2.0s per generation), posing challenges for real-time deployment scenarios. This study contributes to the growing body of work on AI safety by highlighting the robustness gap between explicit and implicit toxicity mitigation. We emphasize the need for more sophisticated approaches that generalize across diverse hate speech patterns without prohibitive computational costs. 

---
# Distribution Corrected Offline Data Distillation for Large Language Models 

**Authors**: Yumeng Zhang, Zhengbang Yang, Yevin Nikhel Goonatilake, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14071)  

**Abstract**: Distilling reasoning traces from strong large language models into smaller ones is a promising route to improve intelligence in resource-constrained settings. Existing approaches face a fundamental trade-off: offline distillation from teacher-generated traces provides high-quality, sample-efficient supervision but suffers from distributional drift: during training, the student model conditions on teacher-generated prefixes, whereas during inference the student autoregresses on self-generated prefixes, leading to compounding errors over long reasoning trajectories. Meanwhile, on-policy or self-distillation methods better match the student's inference-time distribution, but require costly online sampling and often produce low-quality traces in early training. We propose a principled offline reasoning distillation framework that preserves the efficiency and supervision quality of offline teacher-generated data while correcting teacher-student distribution drift. It adaptively emphasizes teacher supervision that is better aligned with the student's on-policy distribution. Evaluations on mathematical reasoning benchmarks of GSM8K, MATH, MATH500, and harder held-out competition-style tasks, including AMC, AIME, and OlympiadBench, show that our method improves reasoning accuracy over prior offline distillation algorithms and yields more stable reasoning traces while preserving instruction-following capabilities. Our work shows that lightweight, distribution-correction-aware training can substantially strengthen offline reasoning distillation without online rollouts. 

---
# Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents 

**Authors**: Xubo Lin, Zezhii Deng, Shihao Wang, Grace Hui Yang, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.14057)  

**Abstract**: Most existing dialogue systems are user-driven, primarily designed to fulfill user requests. However, in many critical real-world scenarios, a conversational agent must proactively extract information to achieve its own objectives rather than merely respond. To address this gap, we introduce \emph{Inquisitive Conversational Agents (ICAs)} and develop an ICA specifically tailored to U.S. Supreme Court oral arguments. We propose a Dual Hierarchical Reinforcement Learning framework featuring two cooperating RL agents, each with its own policy, to coordinate strategic dialogue management and fine-grained utterance generation. By learning when and how to ask probing questions, the agent emulates judicial questioning patterns and systematically uncovers crucial information to fulfill its legal objectives. Evaluations on a U.S. Supreme Court dataset show that our method outperforms various baselines across multiple metrics. It represents an important first step toward broader high-stakes, domain-specific applications. 

---
# PEML: Parameter-efficient Multi-Task Learning with Optimized Continuous Prompts 

**Authors**: Anjir Ahmed Chowdhury, Syed Zawad, Xiaolong Ma, Xu Dong, Feng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.14055)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) is widely used for adapting Large Language Models (LLMs) for various tasks. Recently, there has been an increasing demand for fine-tuning a single LLM for multiple tasks because it requires overall less data for fine-tuning thanks to the common features shared among tasks. More importantly, LLMs are resource demanding and deploying a single model for multiple tasks facilitates resource consolidation and consumes significantly less resources compared to deploying individual large model for each task. Existing PEFT methods like LoRA and Prefix Tuning are designed to adapt LLMs to a specific task. LoRA and its variation focus on aligning the model itself for tasks, overlooking the importance of prompt tuning in multi-task learning while Prefix Tuning only adopts a simple architecture to optimize prompts, which limits the adaption capabilities for multi-task. To enable efficient fine-tuning for multi-task learning, it is important to co-optimize prompt optimization and model adaptation. In this work, we propose a Parameter-Efficient Multi-task Learning (\PM), which employs a neural architecture engineering method for optimizing the continuous prompts while also performing low-rank adaption for model weights. We prototype PEML by creating an automated framework for optimizing the continuous prompts and adapting model weights. We evaluate PEML against state-of-the-arts multi-task learning methods MTL-LoRA, MultiLoRa, C-Poly, and MoE, on the GLUE, SuperGLUE, Massive Multitask Language Understanding, and commonsense reasoning benchmarks. The evaluation results present an average accuracy improvement of up to 6.67%, with individual tasks showing peak gains of up to 10.75%. 

---
# Derivation Prompting: A Logic-Based Method for Improving Retrieval-Augmented Generation 

**Authors**: Ignacio Sastre, Guillermo Moncecchi, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2605.14053)  

**Abstract**: The application of Large Language Models to Question Answering has shown great promise, but important challenges such as hallucinations and erroneous reasoning arise when using these models, particularly in knowledge-intensive, domain-specific tasks. To address these issues, we introduce Derivation Prompting, a novel prompting technique for the generation step of the Retrieval-Augmented Generation framework. Inspired by logic derivations, this method involves deriving conclusions from initial hypotheses through the systematic application of predefined rules. It constructs a derivation tree that is interpretable and adds control over the generation process. We applied this method in a specific case study, significantly reducing unacceptable answers compared to traditional RAG and long-context window methods. 

---
# Physics-R1: An Audited Olympiad Corpus and Recipe for Visual Physics Reasoning 

**Authors**: Shan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14040)  

**Abstract**: We audit the multimodal-physics evaluation pipeline end-to-end and document three undetected construction practices that distort how the field measures vision-language reasoning: train-eval contamination, translation drift, and MCQ saturation. (1) Public training pools (UGPhysics-Train, SciInstruct, MMK12) pass single-stage 5-gram-Jaccard audits with zero hits across all six public physics evals; a three-stage audit (Jaccard -> mxbai-embed-large cosine -> Haiku-4.5 LLM-judge) surfaces 134 near-duplicates and 4,846 paraphrase candidates in SciInstruct alone. (2) A 17-pp Sonnet 4.5 delta on 59 paired Estonian-English olympiad problems (30.5% vs. 13.6%; sign test p=0.011, McNemar p=0.021, paired bootstrap 95% CI [+5.1, +28.9] pp). (3) A 46-pp format-and-novelty gradient on identical Sonnet weights between MCQ (79.7% on PhyX) and open-ended olympiad evaluation (33.4% on PhysOlym-A). We release four artifacts addressing these gaps: PhysCorp-A (6,432-record three-stage-audited multimodal corpus), PhysR1Corp (2,268-record closed-form RL pool), PhysOlym-A (500-problem, 99.8% novel-source held-out olympiad eval with native difficulty labels and an EN/ET bilingual subset), and Physics-R1, a reference GSPO+DAPO recipe cold-started from Qwen3-VL-8B-Thinking. Across 3 seeds, Physics-R1 lifts the audited corpus over the 8B base by +18.3 pp on PhysOlym-A liberal (8.0 -> 26.3 +/- 1.7; 7.1 pp behind Sonnet 4.5), +15.7 pp on PhysReason (23.9 -> 39.6 +/- 6.4; ahead of Qwen3-VL-32B and Gemini 2.5 Pro), +6.9 pp on OlympiadBench-Physics (46.2 +/- 1.5), and +4.1 pp on PhyX MCQ (77.8 +/- 0.3). 

---
# Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding 

**Authors**: Shuoyang Sun, Chang Da, Hao Fang, Kuofeng Gao, Xinhao Zhong, Yi Sun, Fan Mo, Shu-Tao Xia, Bin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14005)  

**Abstract**: Speculative decoding has become a widely adopted technique for accelerating large language model (LLM) inference by drafting multiple candidate tokens and verifying them with a target model in parallel. Its efficiency, however, critically depends on the average accepted length $\tau$, i.e., how many draft tokens survive each verification step. In this work, we identify a new mechanism-level vulnerability in model-based speculative decoding: the drafter is trained to approximate the target model distribution, but this approximation is inevitably imperfect. Such a drafter-target mismatch creates a hidden attack surface where small perturbations can preserve the target model's visible behavior while substantially reducing draft-token acceptability. We propose Mistletoe, a stealthy acceleration-collapse attack against speculative decoding. Mistletoe directly targets the acceptance mechanism of speculative decoding. It jointly optimizes a degradation objective that decreases drafter-target agreement and a semantic-preservation objective that constrains the target model's output distribution. To resolve the conflict between these objectives, we introduce a null-space projection mechanism, where degradation gradients are projected away from the local semantic-preserving direction, suppressing draft acceptance while minimizing semantic drift. Experiments on various speculative decoding systems show that Mistletoe substantially reduces average accepted length $\tau$, collapses speedup, and lowers averaged token throughput, while preserving output quality and perplexity. Our work highlights that speculative decoding introduces a mechanism-level attack surface beyond existing output robustness, calling for more robust designs of LLM acceleration systems. 

---
# VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model with Curriculum Learning and Native Tool Use 

**Authors**: Juan S. Santillana  

**Link**: [PDF](https://arxiv.org/pdf/2605.13989)  

**Abstract**: We present VectraYX-Nano, a 41.95M-parameter decoder-only language model trained from scratch in Spanish for cybersecurity, with a Latin-American focus and native tool invocation via the Model Context Protocol (MCP). Four contributions: (i) Corpus: VectraYX-Sec-ES, a 170M-token Spanish corpus from an eight-VM pipeline (~$25 USD) partitioned into conversational (42M tokens, OpenSubtitles-ES, OASST1), cybersecurity (118M tokens, NVD, Wikipedia-ES, CVE mirror, security blogs), and offensive-security tooling (10M tokens, ExploitDB, HackTricks, OWASP) phases. (ii) Architecture: 42M-parameter Transformer decoder with GQA, QK-Norm, RMSNorm, SwiGLU, RoPE, z-loss, and a 16,384-token byte-fallback BPE. (iii) Curriculum with replay: continual pre-training with a replay buffer yields monotonic loss descent (9.80->3.17->3.00->2.16); after SFT on OASST-ES, Alpaca-ES, CVE Q&A, and 6,327 tool-use traces, the model attains a conversational gate of 0.78+-0.05 (N=4 seeds). (iv) Two findings: a bootstrap-corpus ablation reveals a loss-vs-register inversion at nano scale; a LoRA study shows the B4 tool-selection floor of 0.000 is a corpus-density artifact, not a capacity gate -- a tool-dense corpus (2,801 examples) raises B4 to 0.145+-0.046 on Nano 42M and 0.445+-0.201 on a 260M mid-tier. The GGUF artifact is 81 MB (F16), runs at sub-second TTFT on commodity hardware under this http URL, and is to our knowledge the first Spanish-native cybersecurity LLM with end-to-end MCP integration. Corpus recipe, training scripts, GGUF weights, and B1-B5 benchmark are released. 

---
# Merging Methods for Multilingual Knowledge Editing for Large Language Models: An Empirical Odyssey 

**Authors**: Kunil Lee, Ki-Young Shin, Jong-Hyeok Lee, Young-Joo Suh  

**Link**: [PDF](https://arxiv.org/pdf/2605.13919)  

**Abstract**: Multilingual knowledge editing (MKE) remains challenging because language-specific edits interfere with one another, even when locate-then-edit methods work well in monolingual settings. This paper focuses on three issues: the effectiveness of vector merging methods for MKE, the extent to which Task Singular Vectors for Merging (TSVM) can reduce multilingual interference, and the influence of the weight scaling factor and rank compression ratio on performance. We evaluate six merging variants with two popular backbone large language models, two base knowledge editing methods, and 12 languages on the MzsRE benchmark under a large-scale batch-editing setting. Our results show that vector summation with shared covariance is the most reliable overall strategy, whereas simple summation without shared covariance performs poorly. TSVM improves performance in some settings, but its ability to mitigate multilingual interference is limited. We also find that performance is sensitive to both weight scale and rank ratio, with larger-than-default scaling and relatively low rank often yielding better results. These findings clarify the practical strengths and limits of current vector merging methods for MKE and provide guidance for future multilingual knowledge editing research. 

---
# ATLAS: Agentic or Latent Visual Reasoning? One Word is Enough for Both 

**Authors**: Ziyu Guo, Rain Liu, Xinyan Chen, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15198)  

**Abstract**: Visual reasoning, often interleaved with intermediate visual states, has emerged as a promising direction in the field. A straightforward approach is to directly generate images via unified models during reasoning, but this is computationally expensive and architecturally non-trivial. Recent alternatives include agentic reasoning through code or tool calls, and latent reasoning with learnable hidden embeddings. However, agentic methods incur context-switching latency from external execution, while latent methods lack task generalization and are difficult to train with autoregressive parallelization. To combine their strengths while mitigating their limitations, we propose ATLAS, a framework in which a single discrete 'word', termed as a functional token, serves both as an agentic operation and a latent visual reasoning unit. Each functional token is associated with an internalized visual operation, yet requires no visual supervision and remains a standard token in the tokenizer vocabulary, which can be generated via next-token prediction. This design avoids verbose intermediate visual content generation, while preserving compatibility with the vanilla scalable SFT and RL training, without architectural or methodological modifications. To further address the sparsity of functional tokens during RL, we introduce Latent-Anchored GRPO (LA-GRPO), which stabilizes the training by anchoring functional tokens with a statically weighted auxiliary objective, providing stronger gradient updates. Extensive experiments and analyses demonstrate that ATLAS achieves superior performance on challenging benchmarks while maintaining clear interpretability. We hope ATLAS offers a new paradigm inspiring future visual reasoning research. 

---
# FutureSim: Replaying World Events to Evaluate Adaptive Agents 

**Authors**: Shashwat Goel, Nikhil Chandak, Arvindh Arun, Ameya Prabhu, Steffen Staab, Moritz Hardt, Maksym Andriushchenko, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2605.15188)  

**Abstract**: AI agents are being increasingly deployed in dynamic, open-ended environments that require adapting to new information as it arrives. To efficiently measure this capability for realistic use-cases, we propose building grounded simulations that replay real-world events in the order they occurred. We build FutureSim, where agents forecast world events beyond their knowledge cutoff while interacting with a chronological replay of the world: real news articles arriving and questions resolving over the simulated period. We evaluate frontier agents in their native harness, testing their ability to predict world events over a three-month period from January to March 2026. FutureSim reveals a clear separation in their capabilities, with the best agent's accuracy being 25%, and many having worse Brier skill score than making no prediction at all. Through careful ablations, we show how FutureSim offers a realistic setting to study emerging research directions like long-horizon test-time adaptation, search, memory, and reasoning about uncertainty. Overall, we hope our benchmark design paves the way to measure AI progress on open-ended adaptation spanning long time-horizons in the real world. 

---
# MetaBackdoor: Exploiting Positional Encoding as a Backdoor Attack Surface in LLMs 

**Authors**: Rui Wen, Mark Russinovich, Andrew Paverd, Jun Sakuma, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2605.15172)  

**Abstract**: Backdoor attacks pose a serious security threat to large language models (LLMs), which are increasingly deployed as general-purpose assistants in safety- and privacy-critical applications. Existing LLM backdoors rely primarily on content-based triggers, requiring explicit modification of the input text. In this work, we show that this assumption is unnecessary and limiting. We introduce MetaBackdoor, a new class of backdoor attacks that exploits positional information as the trigger, without modifying textual content. Our key insight is that Transformer-based LLMs necessarily encode token positions to process ordered sequences. As a result, length-correlated positional structure is reflected in the model's internal computation and can be used as an effective non-content trigger signal.
We demonstrate that even a simple length-based positional trigger is sufficient to activate stealthy backdoors. Unlike prior attacks, MetaBackdoor operates on visibly and semantically clean inputs and enables qualitatively new capabilities. We show that a backdoored LLM can be induced to disclose sensitive internal information, including proprietary system prompts, once a length condition is satisfied. We further demonstrate a self-activation scenario, where normal multi-turn interaction can move the conversation context into the trigger region and induce malicious tool-call behavior without attacker-supplied trigger text. In addition, MetaBackdoor is orthogonal to content-based backdoors and can be composed with them to create more precise and harder-to-detect activation conditions.
Our results expand the threat model of LLM backdoors by revealing positional encoding as a previously overlooked attack surface. This challenges defenses that focus on detecting suspicious text and highlights the need for new defense strategies that explicitly account for positional triggers in modern LLM architectures. 

---
# Self-Distilled Agentic Reinforcement Learning 

**Authors**: Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15155)  

**Abstract**: Reinforcement learning (RL) has emerged as a central paradigm for post-training LLM agents, yet its trajectory-level reward signal provides only coarse supervision for long-horizon interaction. On-Policy Self-Distillation (OPSD) complements RL by introducing dense token-level guidance from a teacher branch augmented with privileged context. However, transferring OPSD to multi-turn agents proves problematic: compounding multi-turn instability destabilizes supervision, while skill-conditioned privileged guidance requires asymmetric treatment for negative teacher rejections may arise from imperfect skills retrieval or utilization. We introduce SDAR (Self-Distilled Agentic Reinforcement Learning), which treats OPSD as a gated auxiliary objective while keeping RL as the primary optimization backbone. SDAR maps detached token-level signals into a sigmoid gate, strengthening distillation on teacher-endorsed positive-gap tokens and softly attenuating negative teacher rejections. Across the Qwen2.5 and Qwen3 families on ALFWorld, WebShop, and Search-QA, SDAR substantially improves over GRPO (+9.4% on ALFWorld, +7.0% on Search-QA, +10.2% on WebShop-Acc), avoids the instability of naive GRPO+OPSD, and consistently outperforms hybrid RL--OPSD baselines across model scales. 

---
# Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution 

**Authors**: Saisab Sadhu, Pratinav Seth, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15138)  

**Abstract**: Standard unlearning evaluations measure behavioral suppression in full precision, immediately after training, despite every deployed language model being quantized first. Recent work has shown that 4-bit post-training quantization can reverse machine unlearning; we show this is not a tuning artefact but a systematic dual failure: gradient-based methods that achieve meaningful forgetting lose it under compression, while methods that survive quantization barely change the model. Both failures trace to the same root cause: across all baselines, per-parameter updates lie 47-828x below the NF4 quantization bin width; updates diffused across billions of parameters cannot clear quantization bin boundaries, a consequence we formalize as a sparsity-permanence tradeoff. We present MANSU (Mechanistic-Aligned Null-Space Unlearning), which resolves both modes by combining causal circuit attribution to isolate the minimal forget-set subgraph, circuit-restricted null-space projection with a diagonal-Fisher retain bound, and a per-parameter magnitude floor guaranteeing quantization survival by construction. We additionally introduce Circuit Attribution Divergence (CAD), a mechanistic verification metric distinguishing structural erasure from behavioral suppression, a distinction existing metrics cannot make. Across multiple model families and hazard benchmarks, MANSU is the first method to jointly satisfy all four properties with margin on each (meaningful forgetting, retain preservation, non-positive PTQ gap, and structural erasure), while gradient-based baselines recover up to +0.05 accuracy under compression. 

---
# MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory 

**Authors**: Minghao Guo, Qingyue Jiao, Zeru Shi, Yihao Quan, Boxuan Zhang, Danrui Li, Liwei Che, Wujiang Xu, Shilong Liu, Zirui Liu, Mubbasir Kapadia, Vladimir Pavlovic, Jiang Liu, Mengdi Wang, Yiyu Shi, Dimitris N. Metaxas, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15128)  

**Abstract**: Long-term agent memory is increasingly multimodal, yet existing evaluations rarely test whether agents preserve the visual evidence needed for later reasoning. In prior work, many visually grounded questions can be answered using only captions or textual traces, allowing answers to be inferred without preserving the fine-grained visual evidence. Meanwhile, harder cases that require reasoning over changing visual states are largely absent. Therefore, we introduce MemEye, a framework that evaluates memory capabilities from two dimensions: one measures the granularity of decisive visual evidence (from scene-level to pixel-level evidence), and the other measures how retrieved evidence must be used (from single evidence to evolutionary synthesis). Under this framework, we construct a new benchmark across 8 life-scenario tasks, with ablation-driven validation gates for assessing answerability, shortcut resistance, visual necessity, and reasoning structure. By evaluating 13 memory methods across 4 VLM backbones, we show that current architectures still struggle to preserve fine-grained visual details and reason about state changes over time. Our findings show that long-term multimodal memory depends on evidence routing, temporal tracking, and detail extraction. 

---
# Talk is (Not) Cheap: A Taxonomy and Benchmark Coverage Audit for LLM Attacks 

**Authors**: Karthik Raghu Iyer, Yazdan Jamshidi, Nicholas Bray, Alexey A. Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2605.15118)  

**Abstract**: We introduce a reusable framework for auditing whether LLM attack benchmarks collectively cover the threat surface: a 4$\times$6 Target $\times$ Technique matrix grounded in STRIDE, constructed from a 507-leaf taxonomy -- 401 data-populated and 106 threat-model-derived leaves -- of inference-time attacks extracted from 932 arXiv security studies (2023--2026). The matrix enables benchmark-external validation -- auditing collective coverage rather than individual benchmark consistency. Applying it to six public benchmarks reveals that the three primary frameworks (HarmBench, InjecAgent, AgentDojo) occupy non-overlapping cells covering at most 25\% of the matrix, while entire STRIDE threat categories (Service Disruption, Model Internals) lack any standardized evaluation, despite published attacks in these categories achieving 46$\times$ token amplification and 96\% attack success rates through mechanisms which no benchmark tests. The corpus of 2,521 unique attack groups further reveals pervasive naming fragmentation (up to 29 surface forms for a single attack) and heavy concentration in Safety \& Alignment Bypass, structural properties invisible at smaller scale. The taxonomy, attack records, and coverage mappings are released as extensible artifacts; as new benchmarks emerge, they can be mapped onto the same matrix, enabling the community to track whether evaluation gaps are closing. 

---
# Proposal and study of statistical features for string similarity computation and classification 

**Authors**: E.O. Rodrigues, D. Casanova, M. Teixeira, V. Pegorini, F. Favarim, E. Clua, A. Conci, Panos Liatsis  

**Link**: [PDF](https://arxiv.org/pdf/2605.15110)  

**Abstract**: Adaptations of features commonly applied in the field of visual computing, co-occurrence matrix (COM) and run-length matrix (RLM), are proposed for the similarity computation of strings in general (words, phrases, codes and texts). The proposed features are not sensitive to language related information. These are purely statistical and can be used in any context with any language or grammatical structure. Other statistical measures that are commonly employed in the field such as longest common subsequence, maximal consecutive longest common subsequence, mutual information and edit distances are evaluated and compared. In the first synthetic set of experiments, the COM and RLM features outperform the remaining state-of-the-art statistical features. In 3 out of 4 cases, the RLM and COM features were statistically more significant than the second best group based on distances (P-value < 0.001). When it comes to a real text plagiarism dataset, the RLM features obtained the best results. 

---
# On the Cultural Anachronism and Temporal Reasoning in Vision Language Models 

**Authors**: Mukul Ranjan, Prince Jha, Khushboo Kumari, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15071)  

**Abstract**: Vision-Language Models (VLMs) are increasingly applied to cultural heritage materials, from digital archives to educational platforms. This work identifies a fundamental issue in how these models interpret historical artifacts. We define this phenomenon as cultural anachronism, the tendency to misinterpret historical objects using temporally inappropriate concepts, materials, or cultural frameworks. To quantify this phenomenon, we introduce the Temporal Anachronism Benchmark for Vision-Language Models (TAB-VLM), a dataset of 600 questions across six categories, designed to evaluate temporal reasoning on 1,600 Indian cultural artifacts spanning prehistoric to modern periods. Systematic evaluations of ten state-of-the-art models reveal significant deficiencies on our benchmark, and even the best model (GPT-5.2) achieves only 58.7% overall accuracy. The performance gap persists across varying architectures and scales, suggesting that cultural anachronism represents a significant limitation in visual AI systems, regardless of model size. These findings highlight the disparity between current VLM capabilities and the requirements for accurately interpreting cultural heritage materials, particularly for non-Western visual cultures underrepresented in training data. Our benchmark provides a foundation for enhancing temporal cognition in multimodal AI systems that interact with historical artifacts. The dataset and code are available in our project page. 

---
# Case-Based Calibration of Adaptive Reasoning and Execution for LLM Tool Use 

**Authors**: Renning Pang, Tian Lan, Leyuan Liu, Piao Tong, Sheng Cao, Xiaosong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15041)  

**Abstract**: Tool use extends large language models beyond parametric knowledge, but reliable execution requires balancing appropriate reasoning depth with strict structural validity. We approach this problem from a case-based perspective to present CAST, a case-driven framework that treats historical execution trajectories as structured cases. Instead of reusing raw exemplar outputs, CAST extracts case-derived signals to identify complexity profiles for estimating optimal reasoning strategies, alongside failure profiles to map likely structural breakdowns. The framework translates this knowledge into a fine-grained reward design and adaptive reasoning, enabling the model to autonomously internalize case-based strategies during reinforcement learning. Experiments on BFCLv2 and ToolBench demonstrate that CAST improves both schema-faithful execution and task-level tool-use success while reducing unnecessary deliberation. The approach achieves up to 5.85 percentage points gain in overall execution accuracy and reduces average reasoning length by 26%, significantly mitigating high-impact structural errors. Ultimately, this demonstrates how historical execution cases can provide reusable adaptation knowledge for calibrated tool use. 

---
# Orchard: An Open-Source Agentic Modeling Framework 

**Authors**: Baolin Peng, Wenlin Yao, Qianhui Wu, Hao Cheng, Xiao Yu, Rui Yang, Tao Ge, Alessandrio Sordoni, Xingdi Yuan, Yelong Shen, Pengcheng He, Tong Zhang, Zhou Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.15040)  

**Abstract**: Agentic modeling aims to transform LLMs into autonomous agents capable of solving complex tasks through planning, reasoning, tool use, and multi-turn interaction with environments. Despite major investment, open research remains constrained by infrastructure and training gaps. Many high-performing systems rely on proprietary codebases, models, or services, while most open-source frameworks focus on orchestration and evaluation rather than scalable agent training. We present Orchard, an open-source framework for scalable agentic modeling. At its core is Orchard Env, a lightweight environment service providing reusable primitives for sandbox lifecycle management across task domains, agent harnesses, and pipeline stages. On top of Orchard Env, we build three agentic modeling recipes. Orchard-SWE targets coding agents. We distill 107K trajectories from MiniMax-M2.5 and Qwen3.5-397B, introduce credit-assignment SFT to learn from productive segments of unresolved trajectories, and apply Balanced Adaptive Rollout for RL. Starting from Qwen3-30B-A3B-Thinking, Orchard-SWE achieves 64.3% on SWE-bench Verified after SFT and 67.5% after SFT+RL, setting a new state of the art among open-source models of comparable size. Orchard-GUI trains a 4B vision-language computer-use agent using only 0.4K distilled trajectories and 2.2K open-ended tasks. It achieves 74.1%, 67.0%, and 64.0% success rates on WebVoyager, Online-Mind2Web, and DeepShop, respectively, making it the strongest open-source model while remaining competitive with proprietary systems. Orchard-Claw targets personal assistant agents. Trained with only 0.2K synthetic tasks, it achieves 59.6% pass@3 on Claw-Eval and 73.9% when paired with a stronger ZeroClaw harness. Collectively, these results show that a lightweight, open, harness-agnostic environment layer enables reusable agentic data, training recipes, and evaluations across domains. 

---
# Small, Private Language Models as Teammates for Educational Assessment Design 

**Authors**: Chris Davis Jaldi, Anmol Saini, Shan Zhang, Noah Schroeder, Cogan Shimizu, Eleni Ilkou  

**Link**: [PDF](https://arxiv.org/pdf/2605.15015)  

**Abstract**: Generative AI increasingly supports educational design tasks, e.g., through Large Language Models (LLMs), demonstrating the capability to design assessment questions that are aligned with pedagogical frameworks (e.g., Bloom's taxonomy). However, they often rely on subjective or limited evaluation methods; focus primarily on proprietary models; or rarely systematically examine generation, evaluation, or deployment constraints in real educational settings. Meanwhile, Small Language Models (SLMs) have emerged as local alternatives that better address privacy and resource limitations; yet their effectiveness for assessment tasks remains underexplored. To address this gap, we systematically compare LLMs and SLMs for assessment question design; evaluate generation quality across Bloom's taxonomy levels using reproducible, pedagogically grounded metrics; and further assess model-based judging against expert-informed evaluation by analyzing reliability and agreement patterns. Results show that SLMs achieve competitive performance across key pedagogically motivated quality dimensions while enabling local, privacy-sensitive deployment. However, model-based evaluations also exhibit systematic inconsistencies and bias relative to expert ratings. These findings provide evidence to posit language models as bounded assistants in assessment workflows; underscore the necessity of Human-in-the-Loop; and advance the automated educational question generation field by examining quality, reliability, and deployment-aware trade-offs. 

---
# Boosting Reinforcement Learning with Verifiable Rewards via Randomly Selected Few-Shot Guidance 

**Authors**: Kai Yan, Alexander G. Schwing, Yu-Xiong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15012)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has achieved great success in developing Large Language Models (LLMs) with chain-of-thought rollouts for many tasks such as math and coding. Nevertheless, RLVR struggles with sample efficiency on difficult problems where correct rollouts are hard to generate. Prior works propose to address this issue via demonstration-guided RLVR, i.e., to conduct Supervised FineTuning (SFT) when RL fails; however, SFT often requires a lot of data, which can be expensive to acquire. In this paper, we propose FEST, a FEw-ShoT demonstration-guided RLVR algorithm. It attains compelling results with only 128 demonstrations randomly selected from an SFT dataset. We find that three components are vital for the success: supervised signal, on-policy signal, and decaying weights on the few-shot SFT dataset to prevent overfitting from multiple-epoch training. On several benchmarks, FEST outperforms baselines with magnitudes less SFT data, even matching their performance with full dataset. 

---
# Explainable Detection of Depression Status Shifts from User Digital Traces 

**Authors**: Loris Belcastro, Francesco Gervino, Fabrizio Marozzo, Domenico Talia, Paolo Trunfio  

**Link**: [PDF](https://arxiv.org/pdf/2605.14995)  

**Abstract**: Every day, users generate digital traces (e.g., social media posts, chats, and online interactions) that are inherently timestamped and may reflect aspects of their mental state. These traces can be organized into temporal trajectories that capture how a user's mental health signals evolve, including phases of improvement, deterioration, or stability. In this work, we propose an explainable framework for detecting and analyzing depression-related status shifts in user digital traces. The approach combines multiple BERT-based models to extract complementary signals across different dimensions (e.g., sentiment, emotion, and depression severity). Such signals are then aggregated over time to construct user-level trajectories that are analyzed to identify meaningful change points. To enhance interpretability, the framework integrates a large language model to generate concise and human-readable reports that describe the evolution of mental-health signals and highlight key transitions. We evaluate the framework on two social media datasets. Results show that the approach produces more coherent and informative summaries than direct LLM-based reporting, achieving higher coverage of user history, stronger temporal coherence, and improved sensitivity to change points. An ablation study confirms the contribution of each component, particularly temporal modeling and segmentation. Overall, the method provides an interpretable view of mental health signals over time, supporting research and decision making without aiming at clinical diagnosis. 

---
# Holistic Evaluation and Failure Diagnosis of AI Agents 

**Authors**: Netta Madvil, Gilad Dym, Alon Mecilati, Edo Dekel, Jonatan Liberman, Rotem Brazilay, Liron Schliesser, Max Svidlo, Shai Nir, Orel Shalom, Yaron Friedman, David Connack, Amos Rimon, Philip Tannor, Shir Chorev  

**Link**: [PDF](https://arxiv.org/pdf/2605.14865)  

**Abstract**: AI agents execute complex multi-step processes, but current evaluation falls short: outcome metrics report success or failure without explaining why, and process-level approaches struggle to connect failure types to their precise locations within long, structured traces. We present a holistic agent evaluation framework that pairs top-down agent-level diagnosis with bottom-up span-level evaluation, decomposing analysis into independent per-span assessments. This decomposition scales to traces of arbitrary length and produces span-level rationales for each verdict. On the TRAIL benchmark, our framework achieves state-of-the-art results across all metrics on both GAIA and SWE-Bench, with relative gains over the strongest prior baselines of up to 38% on category F1, up to 3.5x on localization accuracy, and up to 12.5x on joint localization-categorization accuracy. Per-category analysis shows our framework leading in more error categories than any other evaluator. Notably, the same frontier model achieves several times higher localization accuracy when used inside our framework than as a monolithic judge over the full trace, showing that evaluation methodology, not model capability, is the bottleneck. 

---
# Do Composed Image Retrieval Benchmarks Require Multimodal Composition? 

**Authors**: Matteo Attimonelli, Alessandro De Bellis, Aryo Pradipta Gema, Rohit Saxena, Monica Sekoyan, Wai-Chung Kwan, Claudio Pomo, Alessandro Suglia, Dietmar Jannach, Tommaso Di Noia, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2605.14787)  

**Abstract**: Composed Image Retrieval (CIR) is a multimodal retrieval task where a query consists of a reference image and a textual modification, and the goal is to retrieve a target image satisfying both. In principle, strong performance on CIR benchmarks is assumed to require multimodal composition, i.e., combining complementary information from reference image and textual modification. In this work, we show that this assumption does not always hold. Across four widely used CIR benchmarks and eleven Generalist Multimodal Embedding models, a large fraction of queries can be solved using a single modality (from 32.2% to 83.6%), revealing pervasive unimodal shortcuts. Thus, high CIR performance can arise from unimodal signals rather than true multimodal composition. To better understand this issue, we perform a two-stage audit. First, we identify shortcut-solvable queries through cross-model analysis. Second, we conduct human validation on 4,741 shortcut-free queries, of which only 1,689 are well-formed, with common issues including ambiguous edits and mismatched targets. Re-evaluating models on this validated subset reveals qualitatively different behaviour: queries can no longer be solved with a single modality, and successful retrieval requires combining both inputs. While accuracy decreases, reliance on multimodal information increases. Overall, current CIR benchmarks conflate shortcut-solvable, noisy, and genuinely compositional queries, leading to an overestimation of model capability in multimodal composition. 

---
# Persian MusicGen: A Large-Scale Dataset and Culturally-Aware Generative Model for Persian Music 

**Authors**: Mohammad Hossein Sameti, Diba Hadi Esfangereh, Sepehr Harfi Moridani, Leili Javidpour, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2605.14765)  

**Abstract**: Persian music, with its unique tonalities, modal systems (Dastgah), and rhythmic structures, presents significant challenges for music generation models trained primarily on Western music. We address this gap by curating the first large-scale dataset of Persian songs, comprising over 900 hours high-quality audio samples across diverse sub-genres, including pop, traditional, and contemporary styles. This dataset captures the rich melodic and cultural diversity of Persian music and serves as the foundation for fine-tuning MusicGen, a state-of-the-art generative music model. We adapt MusicGen to this domain and evaluate its performance by utilizing subjective and objective metrics. To assess the semantic alignment between generated music and intended style tags, we report the proportion of relevant tags accurately reflected in the generated outputs. Our results demonstrate that the fine-tuned model produces compositions that more align with Persian stylistic conventions. This work introduces a new resource for generative music research and illustrates the adaptability of music generation models to underrepresented cultural and linguistic contexts. 

---
# Agentifying Patient Dynamics within LLMs through Interacting with Clinical World Model 

**Authors**: Minghao Wu, Yuting Yan, Zhenyang Cai, Ke Ji, Chuangsen Fang, Ziying Sheng, Xidong Wang, Rongsheng Wang, Hejia Zhang, Shuang Li, Benyou Wang, Hongyuan Zha  

**Link**: [PDF](https://arxiv.org/pdf/2605.14723)  

**Abstract**: Sepsis management in the ICU requires sequential treatment decisions under rapidly evolving patient physiology. Although large language models (LLMs) encode broad clinical knowledge and can reason over guidelines, they are not inherently grounded in action-conditioned patient dynamics. We introduce SepsisAgent, a world model-augmented LLM agent for sepsis treatment recommendation. SepsisAgent uses a learned Clinical World Model to simulate patient responses under candidate fluid--vasopressor interventions, and follows a propose--simulate--refine workflow before committing to a prescription. We first show that world-model access alone yields inconsistent LLM decision performance, motivating agent-specific training. We then train SepsisAgent through a three-stage curriculum: patient-dynamics supervised fine-tuning, propose--simulate--refine behavior cloning, and world-model-based agentic reinforcement learning. On MIMIC-IV sepsis trajectories, SepsisAgent outperforms all traditional RL and LLM-based baselines in off-policy value while achieving the best safety profile under guideline adherence and unsafe-action metrics. Further analysis shows that repeated interaction with the Clinical World Model enables the agent to learn regularities in patient evolution, which remain useful even when simulator access is removed. 

---
# IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation 

**Authors**: Shijie Lian, Bin Yu, Xiaopeng Lin, Zhaolong Shen, Laurence Tianruo Yang, Yurun Jin, Haishan Liu, Changti Wu, Hang Yuan, Cong Huang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14712)  

**Abstract**: Robot imitation data are often multimodal: similar visual-language observations may be followed by different action chunks because human demonstrators act with different short-horizon intents, task phases, or recent context. Existing frame-conditioned VLA policies infer each chunk from the current observation and instruction alone, so under partial observability they may resample different intents across adjacent replanning steps, leading to inter-chunk conflict and unstable execution. We introduce IntentVLA, a history-conditioned VLA framework that encodes recent visual observations into a compact short-horizon intent representation and uses it to condition chunk generation. We further introduce AliasBench, a 12-task ambiguity-aware benchmark on RoboTwin2 with matched training data and evaluation environments that isolate short-horizon observation aliasing. Across AliasBench, SimplerEnv, LIBERO, and RoboCasa, IntentVLA improves rollout stability and outperforms strong VLA baselines 

---
# Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.14665)  

**Abstract**: Legal reasoning is not semantic similarity search. A court judgment encodes constrained symbolic reasoning: precedent propagation, procedural state transitions, and statute-bound inference. These are properties that vector-based retrieval-augmented generation (RAG) cannot faithfully represent. Hallucinated precedents, outdated statute citations, and unsupported reasoning chains remain persistent failure modes in LLM-based legal AI, with real consequences for access to justice in high-caseload jurisdictions such as India. This paper presents Falkor-IRAC, a graph-constrained generation framework for Indian legal AI that grounds generation in structured reasoning over an IRAC (Issue, Rule, Analysis, Conclusion) knowledge graph. Judgments from the Supreme Court and High Courts of India are ingested as IRAC node structures enriched with procedural state transitions, precedent relationships, and statutory references, stored in FalkorDB for low-latency agentic traversal. At inference time, LLM-generated answers are accepted only if a valid supporting path can be traced through the graph, a check performed by a falsifiability oracle called the Verifier Agent. The system also detects doctrinal conflicts as a first-class output rather than silently resolving them. Falkor-IRAC is evaluated using graph-native metrics: citation grounding accuracy, path validity rate, hallucinated precedent rate, and conflict detection rate. These metrics are argued to be more appropriate for legal reasoning evaluation than BLEU and ROUGE. On a proof-of-concept corpus of 51 Supreme Court judgments, the Verifier Agent correctly validated citations on completed queries and correctly rejected fabricated citations. Evaluation against vector-only RAG baselines is left for future work, as is GPU-accelerated inference to address current timeout rates on CPU hardware. 

---
# Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution 

**Authors**: Tian Qin, Junzhe Chen, Yuqing Shi, Tianshu Zhang, Qiang Ju, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14621)  

**Abstract**: Large vision-language models (LVLMs) often hallucinate when language priors dominate weak or ambiguous visual evidence. Existing contrastive decoding methods mitigate this problem by comparing predictions from the original image with those from externally perturbed visual inputs, but such references can introduce off-manifold artifacts and require costly extra forward passes. We propose SIRA, a training-free internal contrastive decoding framework that constructs a counterfactual reference inside the same LVLM by exploiting the staged information flow of multimodal transformers. Instead of removing visual information from the input, SIRA first lets image and text tokens interact through a shared prefix, forming an aligned multimodal state that preserves prompt interpretation, decoding history, positional structure, and early visual grounding. It then forks a counterfactual branch in later transformer layers, where attention to image-token positions is masked. This branch retains the shared multimodal context but lacks continued access to fine-grained visual evidence, yielding a language-prior-dominated internal reference for token-level contrast. During decoding, SIRA suppresses tokens that remain strong without late visual access and favors predictions whose advantage depends on the full visual pathway. Experiments on POPE, CHAIR, and AMBER with Qwen2.5-VL and LLaVA-v1.5 show that SIRA consistently reduces hallucinations while preserving descriptive coverage and incurring lower overhead than two-pass contrastive decoding. SIRA requires no training, external verifier, or perturbed input, and applies to open-weight LVLMs with white-box inference access. 

---
# Mining Subscenario Refactoring Opportunities in Behaviour-Driven Software Test Suites: ML Classifiers and LLM-Judge Baselines 

**Authors**: Ali Hassaan Mughal, Noor Fatima, Muhammad Bilal  

**Link**: [PDF](https://arxiv.org/pdf/2605.14568)  

**Abstract**: Context. Behaviour-Driven Development (BDD) software test suites accumulate duplicated step subsequences. Three published refactoring patterns are available (within-file Background, within-repo reusable-scenario invocation, cross-organisational shared higher-level step), but no prior work automates which recurring subsequences are worth extracting or which mechanism applies. Objective. Rank recurring step subsequences ("slices") by refactoring suitability (extraction-worthy), pre-map each to one of the three patterns, and quantify prevalence across the public BDD ecosystem. Method. Every contiguous L-step window (L in [2, 18]) in a 339-repository / 276-upstream-owner Gherkin corpus is keyed by paraphrase-robust cluster identifiers and counted under three scopes. Sentence-BERT (SBERT) / Uniform Manifold Approximation and Projection (UMAP) / Hierarchical Density-Based Clustering (HDBSCAN) recovers paraphrase-equivalent slices. Three authors label a stratified 200-slice pool against a written rubric. An eXtreme Gradient Boosting (XGBoost) extraction-worthy classifier trained under 5-fold cross-validation is compared with a tuned rule baseline and two open-weight Large Language Model (LLM) judges. Results. The miner produces 5,382,249 slices collapsing to 692,020 recurring patterns. Three-author Fleiss' kappa = 0.56 (extraction-worthy) and 0.79 (mechanism). The classifier reaches out-of-fold F1 = 0.891 (95% CI [0.852, 0.927]), outperforming both the rule baseline (F1 = 0.836, p = 0.017) and the better LLM judge (F1 = 0.728, p < 1e-4). 75.0%, 59.5%, and 11.7% of scenarios carry a within-file Background, within-repo reusable-scenario, or cross-organisational shared-step candidate. Conclusion. Paraphrase-robust subscenario discovery yields a corpus-wide census of BDD refactoring opportunities; pipeline, classifier predictions, labelled pool, and rubric are released under Apache-2.0. 

---
# Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation 

**Authors**: Suyoung Bae, Jaehoon Lee, Changkyu Choi, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.14563)  

**Abstract**: Automated code documentation is essential for modern software development, providing the contextual grounding that both human developers and coding agents rely on to navigate large codebases. Existing repository-level approaches process components independently, causing redundant retrieval and conflicting descriptions across documents while producing outputs that lack hierarchical structure. Therefore, we propose MemDocAgent, a long-horizon agentic framework that generates documentation within a single, integrated context spanning the entire repository. It combines two components: (i) Dependency-Aware Traversal Guiding that predetermines a traversal order respecting dependency and granularity hierarchies; (ii) Memory-Guided Agentic Interaction, in which the agent interacts with RepoMemory, a shared memory accumulating prior work traces through read, write, and verify operations. Through an in-depth multi-criteria evaluation, MemDocAgent achieves the best performance over both open and closed-source baselines and demonstrates practical applicability in real software development workflows. 

---
# Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy 

**Authors**: Langzhou He, Junyou Zhu, Yue Zhou, Zhengyao Gu, Junhua Liu, Wei-Chieh Huang, Henry Peng Zou, David Wipf, Philip S. Yu, Qitian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14558)  

**Abstract**: Agentic reinforcement learning trains large language models using multi-turn trajectories that interleave long reasoning traces with short environment-facing actions. Common policy-gradient methods, such as PPO and GRPO, treat each token in a trajectory equally, leading to uniform credit assignment. In this paper, we critically demonstrate that such uniform credit assignment largely misallocates token-level training signals. From an energy-based modeling perspective, we show that token-level training signals, quantified by their correlations with reward variance of different rollouts sampled from a given prompt, concentrate sharply on action tokens rather than reasoning tokens, even though action tokens account for only a small fraction of the trajectory. We refer to this phenomenon as the Action Bottleneck. Motivated by this observation, we propose an embarrassingly simple token reweighting approach, ActFocus, that downweights gradients on reasoning tokens, along with an additional energy-based redistribution mechanism that further increases the weights on action tokens with higher uncertainty. Across four environments and different model sizes, ActFocus consistently outperforms PPO and GRPO, yielding final-step gains of up to 65.2 and 63.7 percentage points, respectively, without any additional runtime or memory cost. 

---
# When Retrieval Hurts Code Completion: A Diagnostic Study of Stale Repository Context 

**Authors**: Haojun Weng, Qianqian Yang, Hao Fu, Haobin Pan, Xinwei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2605.14478)  

**Abstract**: Context: Retrieval-augmented code generation relies on cross-file repository context, but retrieved snippets may come from obsolete project states.
Objectives: We study whether temporally stale repository snippets act as harmless noise or actively induce current-state-incompatible code.
Methods: We conduct a controlled diagnostic study on a curated 17-sample set of production-helper signature changes from five Python repositories. For each sample, we compare current-only, stale-only, no-retrieval, and mixed current/stale retrieval conditions under prompts that hide commit freshness and expected current signatures.
Results: Under neutralized prompts, stale-only retrieval induces stale helper references on 15/17 Qwen2.5-Coder-7B-Instruct samples and 13/17 gpt-4.1-mini samples, corresponding to 88.2 and 76.5 percentage-point increases over current-only retrieval. No retrieval produces zero stale references but only 1/17 passing completions. The two models share 75.0% Jaccard overlap among stale-triggering samples, and mixed conditions show that adding valid current evidence largely rescues stale-only failures.
Conclusion: Temporal validity of retrieved repository context is a distinct diagnostic variable for Code RAG robustness: stale context can actively bias models toward obsolete repository state rather than merely removing useful evidence. 

---
# LiSA: Lifelong Safety Adaptation via Conservative Policy Induction 

**Authors**: Minbeom Kim, Lesly Miculicich, Bhavana Dalvi Mishra, Mihir Parmar, Phillip Wallis, Bharath Chandrasekhar, Kyomin Jung, Tomas Pfister, Long T. Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.14454)  

**Abstract**: As AI agents move from chat interfaces to systems that read private data, call tools, and execute multi-step workflows, guardrails become a last line of defense against concrete deployment harms. In these settings, guardrail failures are no longer merely answer-quality errors: they can leak secrets, authorize unsafe actions, or block legitimate work. The hardest failures are often contextual: whether an action is acceptable depends on local privacy norms, organizational policies, and user expectations that resist pre-deployment specification. This creates a practical gap: guardrails must adapt to their own operating environments, yet deployment feedback is typically limited to sparse, noisy user-reported failures, and repeated fine-tuning is often impractical. To address this gap, we propose LiSA (Lifelong Safety Adaptation), a conservative policy induction framework that improves a fixed base guardrail through structured memory. LiSA converts occasional failures into reusable policy abstractions so that sparse reports can generalize beyond individual cases, adds conflict-aware local rules to prevent overgeneralization in mixed-label contexts, and applies evidence-aware confidence gating via a posterior lower bound, so that memory reuse scales with accumulated evidence rather than empirical accuracy alone. Across PrivacyLens+, ConFaide+, and AgentHarm, LiSA consistently outperforms strong memory-based baselines under sparse feedback, remains robust under noisy user feedback even at 20% label-flip rates, and pushes the latency--performance frontier beyond backbone model scaling. Ultimately, LiSA offers a practical path to secure AI agents against the unpredictable long tail of real-world edge risks. 

---
# When Answers Stray from Questions: Hallucination Detection via Question-Answer Orthogonal Decomposition 

**Authors**: Siyang Yao, Erhu Feng, Yubin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.14449)  

**Abstract**: Hallucination detection in large language models (LLMs) requires balancing accu racy, efficiency, and robustness to distribution shift. Black-box consistency methods are effective but demand repeated inference; single-pass white-box probes are effi cient yet treat answer representations in isolation, often degrading sharply under domain shift. We propose QAOD (Question-Answer Orthogonal Decomposition), a single-pass framework that projects away the question-aligned direction from the answer representation to obtain a question-orthogonal component that suppresses domain-conditioned variation. To identify informative signals, QAOD further selects layers via diversity-penalized Fisher scoring and discriminative neurons via Fisher importance. To address both in-domain detection and cross-domain generalization, we design two complementary probing strategies: pairing the or thogonal component with question context yields a joint probe that maximizes in-domain discriminability, while using the orthogonal component alone preserves domain-agnostic factuality signals for robust transfer. QAOD's joint probe achieves the best in-domain AUROC across all evaluated model-dataset pairs, while the orthogonal-only probe delivers the strongest OOD transfer, surpassing the best white-box baseline by up to 21% on BioASQ at under 25% of generation cost. 

---
# Think When Needed: Adaptive Reasoning-Driven Multimodal Embeddings with a Dual-LoRA Architecture 

**Authors**: Longxiang Zhang, Weilong Dai, Guanghao Zhang, Hao Jiang, Pipei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14448)  

**Abstract**: Multimodal large language models (MLLMs) have emerged as a powerful backbone for multimodal embeddings. Recent methods introduce chain-of-thought (CoT) reasoning into the embedding pipeline to improve retrieval quality, but remain costly in both model size and inference cost. They typically employ separate reasoner and embedder with substantial parameter overhead, and generate CoT indiscriminately for every input. However, we observe that for simple inputs, discriminative embeddings already perform well, and redundant reasoning can even mislead the model, degrading performance. To address these limitations, we propose Think When Needed (TWN), a unified multimodal embedding framework with adaptive reasoning. TWN introduces a dual-LoRA architecture that attaches reasoning and embedding adapters to a shared frozen backbone, detaching gradients at their interface to mitigate gradient conflicts introduced by joint optimization while keeping parameters close to a single model. Building on this, an adaptive think mechanism uses a self-supervised routing gate to decide per input whether to generate CoT, skipping unnecessary reasoning to reduce inference overhead and even improve retrieval quality. We further explore embedding-guided RL to optimize CoT quality beyond supervised training. On the 78 tasks of MMEB-V2, TWN achieves state-of-the-art embedding quality while being substantially more efficient than existing generative methods, requiring only 3-5% additional parameters relative to the backbone and up to 50% fewer reasoning tokens compared to the full generative mode. 

---
# SWE-Chain: Benchmarking Coding Agents on Chained Release-Level Package Upgrades 

**Authors**: Man Ho Lam, Chaozheng Wang, Hange Liu, Jingyu Xiao, Haau-sing Li, Jen-tse Huang, Terry Yue Zhuo, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14415)  

**Abstract**: Coding agents powered by large language models are increasingly expected to perform realistic software maintenance tasks beyond isolated issue resolution. Existing benchmarks have shifted toward realistic software evolution, but they rarely capture continuous maintenance at the granularity of package releases, where changes are bundled, shipped, and inherited by subsequent versions. We present SWE-Chain, a benchmark for evaluating agents on chained release-level package upgrades, where each transition builds on the agent's prior codebase. To produce upgrade specifications, we design a divide-and-conquer synthesis pipeline that aligns release notes with code diffs for each version transition, ensuring the requirements are grounded in actual code changes, informative to agents, and feasible to implement. SWE-Chain contains 12 upgrade chains across 9 real Python packages, with 155 version transitions and 1,660 grounded upgrade requirements. Across nine frontier agent-model configurations, agents achieve an average of 44.8% resolving, 65.4% precision, and 50.2% F1 under the Build+Fix regime, with Claude-Opus-4.7 (Claude Code) leading at 60.8% resolving, 80.6% precision, and 68.5% F1. These results show that SWE-Chain is both feasible and discriminative, and reveal that current agents still struggle to make correct upgrades across chained package releases without breaking existing functionality. 

---
# Nexus : An Agentic Framework for Time Series Forecasting 

**Authors**: Sarkar Snigdha Sarathi Das, Palash Goyal, Mihir Parmar, Nanyun Peng, Vishy Tirumalashetty, Chun-Liang Li, Rui Zhang, Jinsung Yoon, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2605.14389)  

**Abstract**: Time series forecasting is not just numerical extrapolation, but often requires reasoning with unstructured contextual data such as news or events. While specialized Time Series Foundation Models (TSFMs) excel at forecasting based on numerical patterns, they remain unaware to real-world textual signals. Conversely, while LLMs are emerging as zero-shot forecasters, their performance remains uneven across domains and contextual grounding. To bridge this gap, we introduce Nexus, a multi-agent forecasting framework that decomposes prediction into specialized stages: isolating macro-level and micro-level temporal fluctuations, and integrating contextual information when available before synthesizing a final forecast. This decomposition enables Nexus to adapt from seasonal signals to volatile, event-driven information without relying on external statistical anchors or monolithic prompting. We show that current-generation LLMs possess substantially stronger intrinsic forecasting ability than previously recognized, depending critically on how numerical and contextual reasoning are organized. Evaluated on data strictly succeeding LLM knowledge cutoffs spanning Zillow real estate metrics and volatile stock market equities, Nexus consistently matches or outperforms state-of-the-art TSFMs and strong LLM baselines. Beyond numerical accuracy, Nexus produces high-quality reasoning traces that explicitly show the fundamental drivers behind each forecast. Our results establish that real-world forecasting is an agentic reasoning problem extending well beyond only sequence modeling. 

---
# NodeSynth: Socially Aligned Synthetic Data for AI Evaluation 

**Authors**: Qazi Mamunur Rashid, Xuan Yang, Zhengzhe Yang, Yanzhou Pan, Erin van Liemt, Darlene Neal, Kshitij Pancholi, Jamila Smith-Loud  

**Link**: [PDF](https://arxiv.org/pdf/2605.14381)  

**Abstract**: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSynth, an evidence-grounded methodology that generates socially relevant synthetic queries by leveraging a fine-tuned taxonomy generator (TaG) anchored in real-world evidence. Evaluated against four mainstream LLMs (e.g., Claude 4.5 Haiku), NodeSynth elicited failure rates up to five times higher than human-authored benchmarks. Ablation studies confirm that our granular taxonomic expansion significantly drives these failure rates, while independent validation reveals critical deficiencies in prominent guard models (e.g., Llama-Guard-3). We open-source our end-to-end research prototype and datasets to enable scalable, high-stakes model evaluation and targeted safety interventions (this https URL). 

---
# A Formative Study of Brief Affective Text as a Complement to Wearable Sensing for Longitudinal Student Health Monitoring 

**Authors**: Tamunotonye Harry, Johanna Hidalgo, Matthew Price, Yuanyuan Feng, Kathryn Stanton, Connie Tompkins, Peter Sheridan Dodds, Mikaela Irene Fudolig, Laura Bloomfield, Christopher Danforth  

**Link**: [PDF](https://arxiv.org/pdf/2605.14360)  

**Abstract**: Wearable devices capture physiological and behavioral data with increasing fidelity, but the psychological context shaping these outcomes is difficult to recover from sensor data alone, limiting passive sensing utility for digital health. We examined whether ultra-brief naturalistic concern text could serve as a scalable complement to passive sensing. In a year-long study of 458 university students (3,610 person-waves) tracked with Oura rings, participants responded bimonthly to an open-ended prompt about what concerned them most; responses had a median length of three words. We compared dictionary-based, general pretrained, and domain-adapted NLP approaches using within-person mixed-effects models across nine sleep and physical activity outcomes. Weeks dominated by academic concern framing were associated with lower physical activity; weeks characterized by emotional exhaustion language were associated with poorer sleep quality and lower heart rate variability. General pretrained embeddings outperformed domain-adapted models for most outcomes, with domain adaptation showing relative advantage for autonomic outcomes. Zero-shot classification of concern topics produced no significant associations, while affective dimensions across all three methods were consistently associated with outcomes, indicating emotional register rather than topical content carries the signal. These findings offer design guidance: ultra-brief affective prompts enrich the psychological interpretability of passive physiological data at minimal burden. 

---
# Herculean: An Agentic Benchmark for Financial Intelligence 

**Authors**: Xueqing Peng, Zhuohan Xie, Yupeng Cao, Haohang Li, Lingfei Qian, Yan Wang, Vincent Jim Zhang, Huan He, Xuguang Ai, Linhai Ma, Ruoyu Xiang, Yueru He, Yi Han, Shuyao Wang, Yuqing Guo, Mingyang Jiang, Yilun Zhao, Youzhong Dong, Xiaoyu Wang, Yankai Chen, Ye Yuan, Qiyuan Zhang, Fuyuan Lyu, Haolun Wu, Yonghan Yang, Zichen Zhao, Yuyang Dai, Fan Zhang, Rania Elbadry, Ayesha Gull, Muhammad Usman Safder, Nuo Chen, Fengbin Zhu, Tianshi Cai, Zimu Wang, Polydoros Giannouris, Yuechen Jiang, Zhiwei Liu, Mohsinul Kabir, Yuyan Wang, Yixiang Zheng, Yangyang Yu, Weijin Liu, Wenbo Cao, Anke Xu, Peng Lu, Jerry Huang, Fengran Mo, Mingquan Lin, Prayag Tiwari, Yijia Zhao, Victor Gutierrez Basulto, Xiao-Yang Liu, Kaleb E Smith, Jiahuan Pei, Arman Cohan, Jimin Huang, Yuehua Tang, Alejandro Lopez-Lira, Xi Chen, Xue Liu, Junichi Tsujii, Jian-Yun Nie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2605.14355)  

**Abstract**: As AI agents improve, the central question is no longer whether they can solve isolated well-defined financial tasks, but whether they can reliably carry out financial professional work. Existing financial benchmarks offer only a partial view of this ability, as they primarily evaluate static competencies such as question answering, retrieval, summarization, and classification. We introduce Herculean, the first skilled benchmark for agentic financial intelligence spanning four representative workflows, including Trading, Hedging, Market Insights, and Auditing. Each workflow is instantiated as a standardized MCP-based skill environment with its own tools, interaction dynamics, constraints, and success criteria, enabling consistent end-to-end assessment of heterogeneous agent systems. Across frontier agents, we find agents perform relatively well on Trading and Market Insights, but struggle substantially on Hedging and Auditing, where long-horizon coordination, state consistency, and structured verification are critical. Overall, our results point to a key gap in current agents in turning financial reasoning into dependable workflow execution in high-stakes financial workflows. 

---
# Dynamic Latent Routing 

**Authors**: Fangyuan Yu, Xin Su, Amir Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2605.14323)  

**Abstract**: We investigate the temporal concatenation of sub-policies in Markov Decision Processes (MDP) with time-varying reward functions. We introduce General Dijkstra Search (GDS), and prove that globally optimal goal-reaching policies can be recovered through temporal composition of intermediate optimal sub-policies. Motivated by the "search, select, update" principle underlying GDS, we propose Dynamic Latent Routing (DLR), a language-model post-training method that jointly learns discrete latent codes, routing policies, and model parameters through dynamic search in a single training stage. In low-data fine-tuning settings, DLR matches or outperforms supervised fine-tuning across four datasets and six models, achieving a mean gain of +6.6 percentage points, while prior discrete-latent baselines consistently underperform SFT. Mechanistic analyses and targeted code ablations show that DLR learns structured routing behaviors with distinct causal roles. 

---
# Minimal-Intervention KV Retention: A Design-Space Study and a Diversity-Penalty Survivor 

**Authors**: Libo Sun, Po-wei Harn, Peixiong He, Xiao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.14292)  

**Abstract**: KV-cache compression at small budgets is a crowded design space spanning cache representation, head-wise routing, compression cadence, decoding behavior, and within-budget scoring. We study seven mechanisms across these five families under matched mean cache on long-form mathematical reasoning (MATH-500~\cite{hendrycks2021math}) with two distilled-reasoning models (Qwen-7B and Llama-8B variants of DeepSeek-R1-Distill~\cite{deepseek2025r1}) at budgets $b \in \{64, 128\}$. All seven were rejected. We then propose $\alpha$, a one-function modification to the TriAttention~\cite{mao2026triattention} retention scorer that replaces argmax-top-$k$ with greedy facility-location-inspired selection under a V-space redundancy penalty controlled by a single weight $\lambda$. A pre-registered protocol tunes $\lambda$ on a frozen development split and confirms on a disjoint held-out split; with $\lambda = 0.5$, $\alpha$ clears Bonferroni on two of the four (model, budget) cells (Qwen $b{=}128$ and Llama $b{=}64$), no cell is significantly negative, and the pre-registered Branch~A triggers. The finding is asymmetric: a minimal scoring modification beat heavier structural redesigns in this regime, and the combined matched-memory, sympy-graded, held-out confirmation protocol is the evidence standard that made the asymmetry visible. 

---
# To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Model 

**Authors**: Chengshuai Zhao, Zhen Tan, Dawei Li, Zhiyuan Yu, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14291)  

**Abstract**: The rapid advancement of Large Vision-Language Models (LVLMs) is increasingly accompanied by unauthorized scraping and training on multimodal web data, posing severe copyright and privacy risks to data owners. Existing countermeasures, such as machine unlearning and watermarks, are inherent post-hoc approaches that act only after intellectual property infringement has already occurred. In this work, we propose MMGuard to empower data owners to proactively protect their multimodal data against unauthorized LVLM fine-tuning. MMGuard generates unlearnable examples by injecting human-imperceptible perturbations that actively exploit the learning dynamics of LVLMs. By minimizing the training loss, the perturbation creates an optimization shortcut, causing the model to overfit to the noise and thereby degrading downstream performance when the perturbation is absent during inference. To further strengthen this defense, MMGuard introduces a cross-modal binding disruption, strategically shifting LVLM attention to enforce a spurious correlation between the noise and the training target with theoretical guarantees. Enhanced by an ensemble learning strategy for cross-model transferability, MMGuard is evaluated against nine open-source LVLMs across six datasets. Our comprehensive results demonstrate effective, stealthy, and robust protection under white-box, gray-box, and black-box threat models, establishing a mechanistic advantage in proactively defending against aggressive fine-tuning exploitation. 

---
# Web Agents Should Adopt the Plan-Then-Execute Paradigm 

**Authors**: Julien Piet, Annabella Chow, Yiwei Hou, Muxi Lyu, Sylvie Venuto, Jinhao Zhu, Raluca Ada Popa, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2605.14290)  

**Abstract**: ReAct has become the default architecture across LLM agents, and many existing web agents follow this paradigm. We argue that it is the wrong default for web agents. Instead, web agents should default to plan-then-execute: commit to a task-specific program before observing runtime web content, then execute it. The reason is that web content mixes inputs from many parties. An e-commerce product page may combine a seller's listing, customer reviews and sponsored advertisements. Under ReAct, all of this content flows into the model when deciding on the next action, creating a direct path for prompt injections to steer the agent's control flow. Plan-then-execute changes this boundary: untrusted data may influence values or branches inside a predefined execution graph, but it cannot redefine the user task or cause the model to synthesize new actions at runtime. We analyze WebArena, a popular web agent benchmark, and find that all tasks are compatible with plan-then-execute, while 80% can be completed with a purely programmatic plan, without any runtime LLM subroutine. We identify the main barrier to adopting plan-then-execute on the web: For it to work well, tools must map cleanly to semantic actions, with effects known before execution, so agents have enough information to plan. The web does not naturally expose that interface. Browser tools such as click, type, and scroll have page-dependent meanings. Planning at this layer is near-sighted: the agent can only see actions on the current page, and later actions appear only after it acts. Closing this gap requires typed interfaces that turn website interactions from clicks and keystrokes to task-level operations. This is an infrastructure problem, not a modeling problem. Web tasks do not need reactivity by default; they need typed, complete, auditable website APIs. 

---
# MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification 

**Authors**: Weisen Jiang, Shuhao Chen, Sinno Jialin Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.14289)  

**Abstract**: Mixture-of-Experts (MoE) models scale capacity by combining specialized experts, but most existing approaches assume centralized access to training data. In practice, data are distributed across clients and cannot be shared due to privacy constraints, making unified MoE training challenging. We propose MetaMoE, a privacy-preserving framework that unifies independently trained, domain-specialized experts into a single MoE using public proxy data as surrogates for inaccessible private data. Central to MetaMoE is diversity-aware proxy selection, which selects client-domain-relevant and diverse samples from public data to effectively approximate private data distributions and supervise router learning. These proxies are further used to align expert training, improving expert coordination at unification time, while a context-aware router enhances expert selection across heterogeneous inputs. Experiments on computer vision and natural language processing benchmarks demonstrate that MetaMoE consistently outperforms recent privacy-preserving MoE unification methods. Code is available at this https URL. 

---
# Hypergraph Enterprise Agentic Reasoner over Heterogeneous Business Systems 

**Authors**: Ling Wang, Songnan Liu, Jianan Wang, Cheng Cheng, Xin Liu, Yihan Zhu, Enyu Li, Yu Xiao, Jiangyong Xie, Duogong Yan, Jiangyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14259)  

**Abstract**: Applying Large Language Models (LLMs) to heterogeneous enterprise systems is hindered by hallucinations and failures in multi-hop, n-ary reasoning. Existing paradigms (e.g., GraphRAG, NL2SQL) lack the semantic grounding and auditable execution required for these complex environments. We introduce HEAR, an enterprise agentic reasoner built on a Stratified Hypergraph Ontology. Its base Graph Layer virtualizes provenance-aware data interfaces, while the Hyperedge Layer encodes n-ary business rules and procedural protocols. Operating an evidence-driven reasoning loop, HEAR dynamically orchestrates ontology tools for structured multi-hop analysis without requiring LLM retraining. Evaluations on supply-chain tasks, including order fulfillment blockage root cause analysis (RCA), show HEAR achieves up to 94.7% accuracy. Crucially, HEAR demonstrates adaptive efficiency: utilizing procedural hyperedges to minimize token costs, while leveraging topological exploration for rigorous correctness on complex queries. By matching proprietary model performance with open-weight backbones and automating manual diagnostics, HEAR establishes a scalable, auditable foundation for enterprise intelligence. 

---
# Active Learners as Efficient PRP Rerankers 

**Authors**: Jeremías Figueiredo Paschmann, Juan Kaplan, Francisco Nattero Santiago Mauricio Barron Bucolo, Juan Wisznia, Luciano del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2605.14236)  

**Abstract**: Pairwise Ranking Prompting (PRP) elicits pairwise preference judgments from an LLM, which are then aggregated into a ranking, usually via classical sorting algorithms. However, judgments are noisy, order-sensitive, and sometimes intransitive, so sorting assumptions do not match the setting. Because sorting aims to recover a full permutation, truncating it to meet a call budget does not produce a dependable top-K. We thus reframe PRP reranking as active learning from noisy pairwise comparisons and show that active rankers are drop-in replacements that improve NDCG@10 per call in the call-constrained regime. Our noise-robust framework also introduces a randomized-direction oracle that uses a single LLM call per pair. This approach converts systematic position bias into zero-mean noise, enabling unbiased aggregate ranking without the cost of bidirectional calls. 

---
# DT-Transformer: A Foundation Model for Disease Trajectory Prediction on a Real-world Health System 

**Authors**: Yunying Zhu, Andrew R Weckstein, Kueiyu Joshua Lin, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14227)  

**Abstract**: Accurate disease trajectory prediction is critical for early intervention, resource allocation, and improving long-term outcomes. While electronic health records (EHRs) provide a rich longitudinal view of patient health in clinical environments, models trained on curated research cohorts may not reflect routine deployment settings, and those trained on single-hospital datasets capture only fragments of each patient's trajectory. This highlights the importance of leveraging large, multi-hospital health systems for training and validation to better reflect real-world clinical complexity. In this work, we develop DT-Transformer, a foundation model trained on 57.1M structured EHR entries over 1.7M patients from Mass General Brigham (MGB), spanning 11 hospitals and a broad network of outpatient clinics. DT-Transformer achieves strong discrimination in both held-out and prospective validation settings. Next-event prediction achieves a median age- and sex-stratified AUC of 0.871 across 896 disease categories, with all categories exceeding AUC 0.5. These results support health system-scale training as a path toward foundation models suited to real-world clinical forecasting. 

---
# Diagnosing Training Inference Mismatch in LLM Reinforcement Learning 

**Authors**: Tianle Zhong, Neiwen Ling, Yifan Pi, Zijun Wei, Tianshu Yu, Geoffrey Fox, Peng Wu, Xiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14220)  

**Abstract**: Modern LLM RL systems separate rollout generation from policy optimization. These two stages are expected to produce token probabilities that match exactly. However, implementation differences can make them assign different values to the same sequence under the same model weights, inducing Training-Inference Mismatch (TIM). TIM is difficult to inspect because it is entangled with off-policy drift and common stabilization mechanisms. In this work, we isolate TIM in a zero-mismatch diagnostic setting (VeXact), and show that small token-level numerical disagreements can independently cause training collapse. We further show that TIM changes the effective optimization problem, and identify a set of remedies that could mitigate TIM. Our results suggest that TIM is not benign numerical noise, but a systems-level perturbation that should be treated as a first-order factor in analyzing LLM RL stability. 

---
# PreFT: Prefill-only finetuning for efficient inference 

**Authors**: Andrew Lanpouthakoun, Aryaman Arora, Zhengxuan Wu, Dhruv Pai, Ben Keigwin, Dan Jurafsky, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2605.14217)  

**Abstract**: Large language models can now be personalised efficiently at scale using parameter efficient finetuning methods (PEFTs), but serving user-specific PEFTs harms throughput, even with specialised kernels and memory management techniques. This is because, theoretically and empirically, a mismatch exists between prefill (processing a large number of tokens at once) and decode (generating a single token autoregressively): the latter has far lower throughput when serving multiple adapters. Rather than optimising performance relative to parameter count, for efficient multi-adapter serving, we instead ought to optimise performance relative to serving throughput. We therefore propose PreFT (Prefill-only Finetuning), wherein we only apply the adapter to prefill tokens and discard it afterwards. PreFT significantly increases throughput with minimal effect on performance. We develop and release an efficient implementation of two prefill-only PEFTs, LoRA and ReFT, on the vLLM inference engine. We first show that serving multi-user PreFTs is more efficient than traditional PEFTs ($1.9\times$ the throughput when serving $512$ adapters on Llama 3.1 70B). Then, we compare the performance of prefill-only vs. all-token adapters on a variety of supervised finetuning and reinforcement learning tasks with LMs at varying scales. On SFT, we observe that the evaluation loss of PreFTs is higher than PEFTs, but can be compensated by increasing rank with nearly no reduction in throughput. On RL, we consistently find that PreFTs approach parity with standard PEFTs. Together, this work validates prefill-only adaptation of LLMs as a more favourable accuracy-throughput tradeoff than existing PEFTs for personalised serving. 

---
# QOuLiPo: What a quantum computer sees when it reads a book 

**Authors**: Christophe Jurczak  

**Link**: [PDF](https://arxiv.org/pdf/2605.14188)  

**Abstract**: What does a book look like to a quantum computer? This paper takes eight classical works of the Renaissance and its late-antique inheritance -- from Augustine to Galileo -- and runs each through a neutral-atom quantum processor. The bridge is graphs: each textual unit becomes an atom, and graph edges are physical blockade constraints for engineered exact unit-disk designs, or a 2D approximation to the semantic graph for natural texts.
Three contributions follow. First, we introduce rigidity rho, a metric for how unique a book's structural backbone is -- distinguishing Marguerite de Navarre's Heptameron (rigid, twelve-nouvelle hard core) from Boethius (fully fungible, every chapter substitutable). Second, we invert the pipeline: rather than extracting a graph from existing prose, we pick a target graph the hardware encodes natively, and write a book whose structure matches it. The twenty-nine texts written this way, collected under the name QOuLiPo, extend the OuLiPo tradition to graph-topological constraints and, together with the eight natural texts, form a benchmark distribution against which neutral-atom hardware can be tracked as it scales. Third, we run both natural and engineered texts on Pasqal's FRESNEL processor up to one hundred atoms; engineered texts reach high approximation ratios, the cleanest instances returning the exact backbone.
A cloud-accessible quantum machine plus an agentic coding environment now lets a single investigator run this pipeline end-to-end. What is reported is an application layer, not a speedup -- humanistic instances ready to load onto neutral-atom processors as they scale, already complementing classical text analysis. The Digital Humanities community has a stake in building familiarity with this hardware now: the engineered-corpus design choices made today fix the benchmark distribution future hardware will be measured against. 

---
# Thinking Ahead: Prospection-Guided Retrieval of Memory with Language Models 

**Authors**: Harshita Chopra, Krishna Kant Chintalapudi, Suman Nath, Ryen W. White, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.14177)  

**Abstract**: Long-horizon personalization requires dialogue assistants to retrieve user-specific facts from extended interaction histories. In practice, many relevant facts often have low semanticsimilarity to the query under dense retrieval. Standard Retrieval-Augmented Generation (RAG) and GraphRAG systems are still largely retrospective: they rely on embedding similarity to the query or on fixed graph traversals, so they often miss facts that matter for the user's needs but lie far from the query in embedding space. Inspired by prospection, the human ability to use imagined futures as cues for recall, we introduce Prospection-Guided Retrieval (PGR), which decouples retrieval from how memories are stored. Given a user query, PGR first expands the goal into a short Tree-of-Thought (ToT) or linear chain of plausible next steps, and uses these steps as retrieval probes rather than relying on the original query alone. The facts retrieved by these probes are then used to personalize the next round of prospection, enabling PGR to uncover additional memories that become relevant only after the simulation is grounded in the user's history. We also introduce MemoryQuest, a challenging multi-session benchmark in which each query is annotated with 3--5 dated reference facts subject to a low query-reference similarity constraint. Across 1,625 queries spanning 185 user profiles from 3 publicly available datasets, PGR-TOT substantially improves retrieval, including nearly 3x recall on MemoryQuest over the strongest baseline. In pairwise LLM-as-judge comparisons against baselines, PGR-generated responses are preferred on 89--98% of queries, with blinded human annotations on held-out subsets showing the same trend. Overall, the results demonstrate that explicit prospection yields large gains in long-horizon retrieval and response quality relative to similarity-only baselines. 

---
# Mini-JEPA Foundation Model Fleet Enables Agentic Hydrologic Intelligence 

**Authors**: Mashrekur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2605.14120)  

**Abstract**: Geospatial foundation models compress multispectral observations into dense embeddings increasingly used in natural-language environmental reasoning systems. A single planetary-scale model, e.g. Google AlphaEarth, handles broad characterization well but may compromise on specialized hydrologic signals. Such generalist models are also often inaccessible, expensive, and require large-scale compute. We propose Mini-JEPAs: a fleet of small sensor-specialized Joint Embedding Predictive Architecture (JEPA) foundation models consulted by a routing agent for specialized questions. We pretrained five 22M-parameter Mini-JEPAs sharing an identical Vision Transformer backbone, JEPA recipe, and 64-d output space, using Sentinel-2 optical, Sentinel-1 SAR, MODIS thermal, multi-temporal Sentinel-2 phenology, and a topography-soil stack. Each Mini-JEPA reconstructs the variable matched to its sensor, with cross-validated $R^2$ reaching 0.97 for elevation, 0.97 for temperature, and 0.81 for precipitation. The five manifolds differ in geometric structure, with global participation ratios from 8.9 to 20.2 and local intrinsic dimensionalities from 2.3 to 9.0. Joint topography-soil and phenology models add predictive value beyond AlphaEarth alone for soil moisture, aridity, and precipitation ($\Delta R^2$ up to 0.031). A router LLM reads per-modality references and selects appropriate sensors with a perfect hit rate over a curated question set. In paired LLM-as-Judge evaluation, dual retrieval over AlphaEarth and the routed fleet outperforms AlphaEarth alone on physics-matched questions (Cohen's $d = 1.10$, $p = 0.031$). Locally-trained Mini-JEPAs can be operationalized for hydrologic intelligence with modest compute. 

---
# Pause and Reflect: Conformal Aggregation for Chain-of-Thought Reasoning 

**Authors**: Yu Gu, Zijun Yu, Vahid Partovi Nia, Masoud Asgharian  

**Link**: [PDF](https://arxiv.org/pdf/2605.14098)  

**Abstract**: Chain-of-thought (CoT) reasoning with self-consistency improves performance by aggregating multiple sampled reasoning paths. In this setting, correctness is no longer tied to a single reasoning trace but to the aggregation rule over a pool of candidate paths, making aggregation uncertainty the central challenge. This issue is critical where confidently incorrect answers are far more costly than abstentions. We introduce a conformal procedure for CoT reasoning that directly addresses aggregation uncertainty. Our approach replaces majority voting with weighted score aggregation over reasoning paths and calibrates an abstention rule using conformal risk control. This approach leads to finite-sample guarantees on the confident-error rate--the probability that the system answers and is wrong. We further identify score separability as the key condition under which abstention provably improves selective accuracy, and derive closed-form expressions that predict accuracy gains from calibration data alone. The method is fully inference-time, and requires no retraining. Across four benchmarks, four open-source models, and three score classes, realized confident-error rates are consistent with the prescribed targets up to calibration-split and test-set variability. Our method achieves $90.1\%$ selective accuracy on GSM8K by abstaining on less than $5\%$ of problems, compared with $82\%$ accuracy under majority-voting baseline. 

---
# CRANE: Constrained Reasoning Injection for Code Agents via Nullspace Editing 

**Authors**: Mingzhi Zhu, Michele Merler, Raju Pavuluri, Stacy Patterson  

**Link**: [PDF](https://arxiv.org/pdf/2605.14084)  

**Abstract**: Code agents must both reason over long-horizon repository state and obey strict tool-use protocols. In paired Instruct/Thinking checkpoints, these capabilities are complementary but misaligned. The Instruct model is concise and tool-disciplined, whereas the Thinking model offers stronger planning and recovery behavior but often over-deliberates and degrades agent performance. We present CRANE (Constrained Reasoning Injection for Code Agents via Nullspace Editing), a training-free parameter-editing method that treats the Thinking-Instruct delta as a directional pool of candidate reasoning edits for the Instruct backbone. CRANE combines magnitude thresholding to denoise the delta, a Conservative Taylor Gate to retain edits that are jointly beneficial for reasoning transfer and tool-use preservation, and Graduated Sigmoidal Projection to suppress format-critical update directions. By merging paired Instruct and Thinking checkpoints, CRANE delivers strong gains over either individual model while preserving Instruct-level efficiency: on Roo-Eval it achieves pass1 of 66.2% (+19.5%) for Qwen3-30B-A3B and 81.5% (+8.7%) for Qwen3-Next-80B-A3B; on SWE-bench-Verified it resolves up to 14 additional instances at both scales (122/500 and 180/500); and on Terminal-Bench v2 it improves pass1/pass5 by up to 2.3%/7.8%, reaching 7.6%/17.9% and 14.8%/30.3%, respectively, consistently outperforming alternative merging strategies across all three benchmarks. 

---
# Rethinking Layer Relevance in Large Language Models Beyond Cosine Similarity 

**Authors**: Cristian Hinostroza, Rodrigo Toro Icarte, Christ Devia, Andres Carvallo De Ferari, Eugenio Herrera-Berg, Denis Parra, Jorge F Silva  

**Link**: [PDF](https://arxiv.org/pdf/2605.14075)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing. Understanding their internal mechanisms is crucial for developing more interpretable and optimized architectures. Mechanistic interpretability has led to the development of various methods for assessing layer relevance, with cosine similarity being a widely used tool in the field. On this work, we demonstrate that cosine similarity is a poor proxy for the actual performance degradation caused by layer removal. Our theoretical analysis shows that a layer can exhibit an arbitrarily low cosine similarity score while still being crucial to the model's performance. On the other hand, empirical evidence from a range of LLMs confirms that the correlation between cosine similarity and actual performance degradation is often weak or moderate, leading to misleading interpretations of a transformer's internal mechanisms. We propose a more robust metric for assessing layer relevance: the actual drop in model accuracy resulting from the removal of a layer. Even though it is a computationally costly metric, this approach offers a more accurate picture of layer importance, allowing for more informed pruning strategies and lightweight models. Our findings have significant implications for the development of interpretable LLMs and highlight the need to move beyond cosine similarity in assessing layer relevance. 

---
# A Benchmark for Early-stage Parkinson's Disease Detection from Speech 

**Authors**: Terry Yi Zhong, Cristian Tejedor-Garcia, Khiet P. Truong, Janna Maas, Louis ten Bosch, Bastiaan R. Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2605.14066)  

**Abstract**: Early-stage Parkinson's disease (EarlyPD) detection from speech is clinically meaningful yet underexplored, and published results are hard to compare because studies differ in datasets, languages, tasks, evaluation protocols, and EarlyPD definitions. To address this issue, we propose the first benchmark for speech-based EarlyPD detection, with a speaker-independent split designed for fair and replicable cross-method evaluation on researcher-accessible datasets. The benchmark covers three common speech tasks and evaluates methods under different training-resource settings. We also present multi-dimensional evaluation breakdowns by dataset, aggregation level, gender, and disease stage to support fine-grained comparisons and clinical adoption. Our results provide a replicable reference and actionable insights, encouraging the adoption of this publicly available benchmark to advance robust and clinically meaningful EarlyPD detection from speech. 

---
# Know When To Fold 'Em: Token-Efficient LLM Synthetic Data Generation via Multi-Stage In-Flight Rejection 

**Authors**: Anjir Ahmed Chowdhury, Syed Zawad, Feng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.14062)  

**Abstract**: While synthetic data generation with large language models (LLMs) is widely used in post-training pipelines, existing approaches typically generate full outputs before applying quality filters, leading to substantial token waste on samples that are ultimately discarded. To address this, we propose Multi-Stage In-Flight Rejection (MSIFR), a lightweight, training-free framework that detects and terminates low-quality generation trajectories at intermediate checkpoints before they reach full completion. MSIFR decomposes the generation process into sequential stages and applies fast rule-based validators to identify arithmetic inconsistencies, hallucination patterns, and formatting violations, enabling early rejection of faulty samples. We formalize in-flight rejection as a sequential decision process and show that any non-trivial discard policy reduces expected token consumption, with stage-wise savings increasing when rejection occurs earlier in the generation pipeline. We further demonstrate that conditional utility estimates form a martingale, ensuring that early, in-flight rejection does not bias the expected utility of retained samples. Across five instruction-tuned models and seven reasoning benchmarks, MSIFR reduces token consumption by 11%-77% as a standalone method, and up to 78.2% when combined with early-exit methods, while preserving or improving evaluation accuracy. These results confirm that MSIFR provides a practical mechanism for improving the efficiency of LLM-based synthetic data generation without additional training or architectural changes. 

---
# Bridging Legal Interpretation and Formal Logic: Faithfulness, Assumption, and the Future of AI Legal Reasoning 

**Authors**: Olivia Peiyu Wang, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2605.14049)  

**Abstract**: The growing adoption of large language models in legal practice brings both significant promise and serious risk. Legal professionals stand to benefit from AI that can reason over contracts, draft documents, and analyze sources at scale, yet the high-stakes nature of legal work demands a level of rigor that current AI systems do not provide. The central problem is not simply that LLMs hallucinate facts and references; it is that they systematically draw inferences that go beyond what the source text actually supports, presenting assumption-laden conclusions as if they were logically grounded. This proposal presents a neuro-symbolic approach to legal AI that combines the expressive power of large language models with the rigor of formal verification, aiming to make AI-assisted legal reasoning both capable and trustworthy, thus reducing the burden of manual verification without sacrificing the accountability that legal practice demands. 

---
# Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility 

**Authors**: Gergely Szilvasy, Manuel Faysse, Maria Lomeli, Matthijs Douze, Pierre-Emmanuel Mazaré, Loïc Cabannes, Wen-tau Yih, Hervé Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2605.14037)  

**Abstract**: Under modern test-time compute and agentic paradigms, language models process ever-longer sequences. Efficient text generation with transformer architectures is increasingly constrained by the Key-Value cache memory footprint and bandwidth. To address this limitation, we introduce Self-Pruned Key-Value Attention (SP-KV), a mechanism designed to predict future KV utility in order to reduce the size of the long-term KV cache. This strategy operates at a fine granularity: a lightweight utility predictor scores each key-value pair, and while recent KVs are always available via a local window, older pairs are written in the cache and used in global attention only if their predicted utility surpasses a given threshold. The LLM and the utility predictor are trained jointly end-to-end exclusively through next-token prediction loss, and are adapted from pretrained LLM checkpoints.
Rather than enforcing a fixed compression ratio, SP-KV performs dynamic sparsification: the mechanism adapts to the input and typically reduces the KV cache size by a factor of $3$ to $10\times$, longer sequences often being more compressible. This leads to vast improvements in memory usage and decoding speed, with little to no degradation of validation loss nor performance on a broad set of downstream tasks. Beyond serving as an effective KV-cache reduction mechanism, our method reveals structured layer- and head-specific sparsity patterns that we can use to guide the design of hybrid local-global attention architectures. 

---
# Enhanced and Efficient Reasoning in Large Learning Models 

**Authors**: Leslie G. Valiant  

**Link**: [PDF](https://arxiv.org/pdf/2605.14036)  

**Abstract**: In current Large Language Models we can trust the production of smoothly flowing prose on the basis of the principles of machine learning. However, there is no comparably principled basis to justify trust in the content of the text produced. It appears to be conventional wisdom that addressing this issue by adding more principled reasoning is not computationally affordable.
Here we propose a principled method of reasoning that is efficient enough to be practical for large language models. Further, the method allows the retention of much of the currently used software and hardware base. Our method for improving the functioning of large language models consists of a first stage of preprocessing that recodes the data to a Unary Relational Integracode that is more explicit about the relationships among the objects described in the text, followed as a second stage by a standard but possibly streamlined machine learning process that then also learns to predict these relationships.
The method may be viewed as realizing a world model and applying beyond natural language, to vision and actions, for example, where the multiple properties of an object referred to in an input are brought together explicitly, rather than remaining distributed in the various references to it in the input. We articulate its advantages in terms of Robust Logic, a system for performing principled chaining on learned, and hence uncertain, information. We show that this recoding has the surprising and fortuitous property that, while succinct, it makes the task of learning a core subset of relational rules that hold in the world described in the training data polynomial time learnable in a defined sense, the polynomial depending on the complexity of the rule. This gives support for sound reasoning within each single call of the learned classifier as well as between multiple calls. 

---
# From Descriptive to Prescriptive: Uncover the Social Value Alignment of LLM-based Agents 

**Authors**: Jinxian Qu, Qingqing Gu, Teng Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.14034)  

**Abstract**: Wide applications of LLM-based agents require strong alignment with human social values. However, current works still exhibit deficiencies in self-cognition and dilemma decision, as well as self-emotions. To remedy this, we propose a novel value-based framework that employs GraphRAG to convert principles into value-based instructions and steer the agent to behave as expected by retrieving the suitable instruction upon a specific conversation context. To evaluate the ratio of expected behaviors, we define the expected behaviors from two famous theories, Maslow's Hierarchy of Needs and Plutchik's Wheel of Emotion. By experimenting with our method on the benchmark of DAILYDILEMMAS, our method exhibits significant performance gains compared to prompt-based baselines, including ECoT, Plan-and-Solve, and Metacognitive prompting. Our method provides a basis for the emergence of self-emotion in AI systems. 

---
# HodgeCover: Higher-Order Topological Coverage Drives Compression of Sparse Mixture-of-Experts 

**Authors**: Tao Zhong, Dongzhe Zheng, Christine Allen-Blanchette  

**Link**: [PDF](https://arxiv.org/pdf/2605.13997)  

**Abstract**: Sparse Mixture-of-Experts (MoE) layers route tokens through a handful of experts, and learning-free compression of these layers reduces inference cost without retraining. A subtle obstruction blocks every existing compressor in this family: three experts can each be pairwise compatible yet form an irreducible cycle when merged together, so any score that ranks experts on pairwise signals is structurally blind to which triples are jointly mergeable. We show the obstruction is a precise mathematical object, the harmonic kernel of the simplicial Laplacian on a 2-complex whose vertices are experts, whose edges carry KL merge barriers, and whose faces carry triplet barriers; Hodge-decomposing the edge-barrier signal isolates the kernel exactly. We turn the diagnostic into a selection objective: HodgeCover greedily covers the harmonic-critical edges and triplet-critical triangles, and a hybrid variant of HodgeCover pairs it with off-the-shelf weight pruning on survivors. On three open-weight Sparse MoE backbones under aggressive expert reduction, HodgeCover matches state-of-the-art learning-free baselines on the expert-reduction axis, leads on the aggressive-compression frontier of the hybrid axis, and uniquely balances retained mass across all four Hodge components. These results show that exposing the harmonic kernel of a learned MoE structure changes which compressor wins at the regime that matters most. 

---
# Beyond Mode-Seeking RL: Trajectory-Balance Post-Training for Diffusion Language Models 

**Authors**: Saba Ahmadi, Prasanna Parthasarathi, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.13935)  

**Abstract**: Diffusion language models are a promising alternative to autoregressive models, yet post-training methods for them largely adapt reward-maximizing objectives. We identify a central failure mode in this setting we call trajectory locking: sampled reward-driven updates over-concentrate probability mass onto a narrow set of denoising paths, reducing coverage of alternative correct solutions under repeated sampling. To address this, we propose TraFL (Trajectory Flow baLancing), a trajectory-balance objective that trains the policy toward a reward-tilted target distribution anchored to a frozen reference model. We make this practical for diffusion language models with a diffusion-compatible sequence-level surrogate and a learned prompt-dependent normalization. Across mathematical reasoning and code generation benchmarks, TraFL is the only evaluated post-training method that improves over the base model in every benchmark-length setting, with gains that persist as the sampling budget increases. The improvements transfer to held-out evaluations: TraFL stays above the base model on Minerva Math and is the strongest method on every LiveCodeBench difficulty split. 

---
# PREPING: Building Agent Memory without Tasks 

**Authors**: Yumin Choi, Sangwoo Park, Minki Kang, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.13880)  

**Abstract**: Agent memory is typically constructed either offline from curated demonstrations or online from post-deployment interactions. However, regardless of how it is built, an agent faces a cold-start gap when first introduced to a new environment without any task-specific experience available. In this paper, we study pre-task memory construction: whether an agent can build procedural memory before observing any target-environment tasks, using only self-generated synthetic practice. Yet, synthetic interaction alone is insufficient, as without controlling what to practice and what to store, synthetic tasks become redundant, infeasible, and ultimately uninformative, and memory further degrades quickly due to unfiltered trajectories. To overcome this, we present Preping, a proposer-guided memory construction framework. At its core is proposer memory, a structured control state that shapes future practice. A Proposer generates synthetic tasks conditioned on this state, a Solver executes them, and a Validator determines which trajectories are eligible for memory insertion while also providing feedback to guide future proposals. Experiments on AppWorld, BFCL v3, and MCP-Universe show that Preping substantially improves over a no-memory baseline and achieves performance competitive with strong playbook-based methods built from offline or online experience, with deployment cost $2.99\times$ lower on AppWorld and $2.23\times$ lower on BFCL v3 than online memory construction. Further analyses reveal that the main benefit does not come from synthetic volume alone, but from proposer-side control over feasibility, redundancy, and coverage, combined with selective memory updates. 

---
# A Hormone-inspired Emotion Layer for Transformer language models (HELT) 

**Authors**: Eslam Reda, Sara El-Metwally  

**Link**: [PDF](https://arxiv.org/pdf/2605.13858)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in generating contextually relevant and grammatically correct text. However, they fundamentally lack the ability to process and respond to emotional context in a manner analogous to human emotional cognition. Current approaches to emotion modeling in NLP systems rely primarily on discrete emotion classification or simplistic sentiment analysis, which fail to capture the continuous, multi-dimensional nature of human emotional states. In this paper, we introduce HormoneT5, a novel architecture that augments transformer language models with a biologically-inspired Hormone Emotion Block that simulates the human endocrine system's role in emotional processing. Our approach computes six continuous hormone-like values through specialized per-hormone attention heads, each with orthogonally initialized learnable queries, temperature-scaled attention mechanisms, and deep output projections. These hormone values are then transformed into an emotional embedding that modulates the encoder hidden states, enabling emotionally-appropriate response generation. We propose a multi-objective training framework combining sequence-to-sequence loss, hormone prediction loss with margin penalties, and diversity regularization to prevent attention collapse. Experimental results on our curated emotion-labeled dataset demonstrate that HormoneT5 achieves 85%+ per-hormone accuracy within a 0.15 tolerance threshold, with hormone differentiation ranges exceeding 0.85 across all six hormones between contrasting emotional tones. Human evaluation studies show significant preference (p < 0.01) for HormoneT5-generated responses in terms of emotional appropriateness and empathetic quality compared to baseline T5 outputs. Our work opens new directions for biologically-grounded affective computing and emotionally intelligent conversational agents. 

---
# GraphBit: A Graph-based Agentic Framework for Non-Linear Agent Orchestration 

**Authors**: Yeahia Sarker, Md Rahmat Ullah, Musa Molla, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2605.13848)  

**Abstract**: Agentic LLM frameworks that rely on prompted orchestration, where the model itself determines workflow transitions, often suffer from hallucinated routing, infinite loops, and non-reproducible execution. We introduce GraphBit, an engine-orchestrated framework that defines workflows explicitly and deterministically as a directed acyclic graph (DAG). Unlike prompted orchestration, agents in GraphBit operate as typed functions, while a Rust-based engine governs routing, state transitions, and tool invocation, ensuring reproducibility and auditability. The engine supports parallel branch execution, conditional control flow over structured state predicates, and configurable error recovery. A three-tier memory architecture consisting of ephemeral scratch space, structured state, and external connectors isolates context across stages, preventing cascading context bloat that degrades reasoning in long-running pipelines. Across GAIA benchmark tasks spanning zero-tool, document-augmented, and web-enabled workflows, GraphBit outperforms six existing frameworks, achieving the highest accuracy (67.6 percent), zero framework-induced hallucinations, the lowest latency (11.9 ms overhead), and the highest throughput. Ablation studies demonstrate that each memory tier contributes measurably to performance, with deterministic execution providing the greatest gains on tool-intensive tasks representative of real-world deployments. 

---
