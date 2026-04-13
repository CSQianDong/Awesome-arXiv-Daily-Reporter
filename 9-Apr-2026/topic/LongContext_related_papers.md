# ATANT: An Evaluation Framework for AI Continuity 

**Authors**: Samuel Sameer Tanguturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06710)  

**Abstract**: We present ATANT (Automated Test for Acceptance of Narrative Truth), an open evaluation framework for measuring continuity in AI systems: the ability to persist, update, disambiguate, and reconstruct meaningful context across time. While the AI industry has produced memory components (RAG pipelines, vector databases, long context windows, profile layers), no published framework formally defines or measures whether these components produce genuine continuity. We define continuity as a system property with 7 required properties, introduce a 10-checkpoint evaluation methodology that operates without an LLM in the evaluation loop, and present a narrative test corpus of 250 stories comprising 1,835 verification questions across 6 life domains. We evaluate a reference implementation across 5 test suite iterations, progressing from 58% (legacy architecture) to 100% in isolated mode (250 stories) and 100% in 50-story cumulative mode, with 96% at 250-story cumulative scale. The cumulative result is the primary measure: when 250 distinct life narratives coexist in the same database, the system must retrieve the correct fact for the correct context without cross-contamination. ATANT is system-agnostic, model-independent, and designed as a sequenced methodology for building and validating continuity systems. The framework specification, example stories, and evaluation protocol are available at this https URL. The full 250-story corpus will be released incrementally. 

---
# A-MBER: Affective Memory Benchmark for Emotion Recognition 

**Authors**: Deliang Wen, Ke Sun, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07017)  

**Abstract**: AI assistants that interact with users over time need to interpret the user's current emotional state in order to respond appropriately and personally. However, this capability remains insufficiently evaluated. Existing emotion datasets mainly assess local or instantaneous affect, while long-term memory benchmarks focus largely on factual recall, temporal consistency, or knowledge updating. As a result, current resources provide limited support for testing whether a model can use remembered interaction history to interpret a user's present affective state.
We introduce A-MBER, an Affective Memory Benchmark for Emotion Recognition, to evaluate this capability. A-MBER focuses on present affective interpretation grounded in remembered multi-session interaction history. Given an interaction trajectory and a designated anchor turn, a model must infer the user's current affective state, identify historically relevant evidence, and justify its interpretation in a grounded way. The benchmark is constructed through a staged pipeline with explicit intermediate representations, including long-horizon planning, conversation generation, annotation, question construction, and final packaging. It supports judgment, retrieval, and explanation tasks, together with robustness settings such as modality degradation and insufficient-evidence conditions.
Experiments compare local-context, long-context, retrieved-memory, structured-memory, and gold-evidence conditions within a unified framework. Results show that A-MBER is especially discriminative on the subsets it is designed to stress, including long-range implicit affect, high-dependency memory levels, trajectory-based reasoning, and adversarial settings. These findings suggest that memory supports affective interpretation not simply by providing more history, but by enabling more selective, grounded, and context-sensitive use of past interaction 

---
# Reasoning Fails Where Step Flow Breaks 

**Authors**: Xiaoyu Xu, Yulan Pan, Xiaosong Yuan, Zhihong Shen, Minghao Su, Yuanhao Su, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06695)  

**Abstract**: Large reasoning models (LRMs) that generate long chains of thought now perform well on multi-step math, science, and coding tasks. However, their behavior is still unstable and hard to interpret, and existing analysis tools struggle with such long, structured reasoning traces. We introduce Step-Saliency, which pools attention--gradient scores into step-to-step maps along the question--thinking--summary trajectory. Across several models, Step-Saliency reveals two recurring information-flow failures: Shallow Lock-in, where shallow layers over-focus on the current step and barely use earlier context, and Deep Decay, where deep layers gradually lose saliency on the thinking segment and the summary increasingly attends to itself and the last few steps. Motivated by these patterns, we propose StepFlow, a saliency-inspired test-time intervention that adjusts shallow saliency patterns measured by Step-Saliency via Odds-Equal Bridge and adds a small step-level residual in deep layers via Step Momentum Injection. StepFlow improves accuracy on math, science, and coding tasks across multiple LRMs without retraining, indicating that repairing information flow can recover part of their missing reasoning performance. 

---
# Evaluating In-Context Translation with Synchronous Context-Free Grammar Transduction 

**Authors**: Jackson Petty, Jaulie Goe, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07320)  

**Abstract**: Low-resource languages pose a challenge for machine translation with large language models (LLMs), which require large amounts of training data. One potential way to circumvent this data dependence is to rely on LLMs' ability to use in-context descriptions of languages, like textbooks and dictionaries. To do so, LLMs must be able to infer the link between the languages' grammatical descriptions and the sentences in question. Here we isolate this skill using a formal analogue of the task: string transduction based on a formal grammar provided in-context. We construct synchronous context-free grammars which define pairs of formal languages designed to model particular aspects of natural language grammar, morphology, and written representation. Using these grammars, we measure how well LLMs can translate sentences from one formal language into another when given both the grammar and the source-language sentence. We vary the size of the grammar, the lengths of the sentences, the syntactic and morphological properties of the languages, and their written script. We note three key findings. First, LLMs' translation accuracy decreases markedly as a function of grammar size and sentence length. Second, differences in morphology and written representation between the source and target languages can strongly diminish model performance. Third, we examine the types of errors committed by models and find they are most prone to recall the wrong words from the target language vocabulary, hallucinate new words, or leave source-language words untranslated. 

---
# Mixed-Initiative Context: Structuring and Managing Context for Human-AI Collaboration 

**Authors**: Haichang Li, Qinshi Zhang, Piaohong Wang, Zhicong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07121)  

**Abstract**: In the human-AI collaboration area, the context formed naturally through multi-turn interactions is typically flattened into a chronological sequence and treated as a fixed whole in subsequent reasoning, with no mechanism for dynamic organization and management along the collaboration workflow. Yet these contexts differ substantially in lifecycle, structural hierarchy, and relevance. For instance, temporary or abandoned exchanges and parallel topic threads persist in the limited context window, causing interference and even conflict. Meanwhile, users are largely limited to influencing context indirectly through input modifications (e.g., corrections, references, or ignoring), leaving their control neither explicit nor verifiable.
To address this, we propose Mixed-Initiative Context, which reconceptualizes the context formed across multi-turn interactions as an explicit, structured, and manipulable interactive object. Under this concept, the structure, scope, and content of context can be dynamically organized and adjusted according to task needs, enabling both humans and AI to actively participate in context construction and regulation. To explore this concept, we implement Contextify as a probe system and conduct a user study examining users' context management behaviors, attitudes toward AI initiative, and overall collaboration experience. We conclude by discussing the implications of this concept for the HCI community. 

---
# HingeMem: Boundary Guided Long-Term Memory with Query Adaptive Retrieval for Scalable Dialogues 

**Authors**: Yijie Zhong, Yunfan Gao, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06845)  

**Abstract**: Long-term memory is critical for dialogue systems that support continuous, sustainable, and personalized interactions. However, existing methods rely on continuous summarization or OpenIE-based graph construction paired with fixed Top-\textit{k} retrieval, leading to limited adaptability across query categories and high computational overhead. In this paper, we propose HingeMem, a boundary-guided long-term memory that operationalizes event segmentation theory to build an interpretable indexing interface via boundary-triggered hyperedges over four elements: person, time, location, and topic. When any such element changes, HingeMem draws a boundary and writes the current segment, thereby reducing redundant operations and preserving salient context. To enable robust and efficient retrieval under diverse information needs, HingeMem introduces query-adaptive retrieval mechanisms that jointly decide (a) \textit{what to retrieve}: determine the query-conditioned routing over the element-indexed memory; (b) \textit{how much to retrieve}: control the retrieval depth based on the estimated query type. Extensive experiments across LLM scales (from 0.6B to production-tier models; \textit{e.g.}, Qwen3-0.6B to Qwen-Flash) on LOCOMO show that HingeMem achieves approximately $20\%$ relative improvement over strong baselines without query categories specification, while reducing computational cost (68\%$\downarrow$ question answering token cost compared to HippoRAG2). Beyond advancing memory modeling, HingeMem's adaptive retrieval makes it a strong fit for web applications requiring efficient and trustworthy memory over extended interactions. 

---
# AV-SQL: Decomposing Complex Text-to-SQL Queries with Agentic Views 

**Authors**: Minh Tam Pham, Trinh Pham, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen, Thanh Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07041)  

**Abstract**: Text-to-SQL is the task of translating natural language queries into executable SQL for a given database, enabling non-expert users to access structured data without writing SQL manually. Despite rapid advances driven by large language models (LLMs), existing approaches still struggle with complex queries in real-world settings, where database schemas are large and questions require multi-step reasoning over many interrelated tables. In such cases, providing the full schema often exceeds the context window, while one-shot generation frequently produces non-executable SQL due to syntax errors and incorrect schema linking. To address these challenges, we introduce AV-SQL, a framework that decomposes complex Text-to-SQL into a pipeline of specialized LLM agents. Central to AV-SQL is the concept of agentic views: agent-generated Common Table Expressions (CTEs) that encapsulate intermediate query logic and filter relevant schema elements from large schemas. AV-SQL operates in three stages: (1) a rewriter agent compresses and clarifies the input query; (2) a view generator agent processes schema chunks to produce agentic views; and (3) a planner, generator, and revisor agent collaboratively compose these views into the final SQL query. Extensive experiments show that AV-SQL achieves 70.38% execution accuracy on the challenging Spider 2.0 benchmark, outperforming state-of-the-art baselines, while remaining competitive on standard datasets with 85.59% on Spider, 72.16% on BIRD and 63.78% on KaggleDBQA. Our source code is available at this https URL. 

---
# Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models 

**Authors**: Bajian Xiang, Tingwei Guo, Xuan Chen, Yang Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.06871)  

**Abstract**: Large Speech Language Models (LSLMs) typically operate at high token rates (tokens/s) to ensure acoustic fidelity, yet this results in sequence lengths that far exceed the underlying semantic content, incurring prohibitive inference costs. In this paper, we empirically revisit the necessity of such granular token-level processing. Through layer-wise oracle interventions, we unveil a structured redundancy hierarchy: while shallow layers encode essential acoustic details, deep layers exhibit extreme redundancy, allowing for aggressive compression. Motivated by these findings, we introduce Affinity Pooling, a training-free, similarity-based token merging mechanism. By strategically applying this method at both input and deep layers, we effectively compress speech representations without compromising semantic information. Extensive evaluations across three tasks demonstrate that our approach reduces prefilling FLOPs by 27.48\% while maintaining competitive accuracy. Practical deployment further confirms significant efficiency gains, yielding up to $\sim$1.7$\times$ memory savings and $\sim$1.1$\times$ faster time-to-first-token on long utterances. Our results challenge the necessity of fully distinct token representations, providing new perspectives on LSLM efficiency. 

---
# Evaluating LLM-Based 0-to-1 Software Generation in End-to-End CLI Tool Scenarios 

**Authors**: Ruida Hu, Xinchen Wang, Chao Peng, Cuiyun Gao, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2604.06742)  

**Abstract**: Large Language Models (LLMs) are driving a shift towards intent-driven development, where agents build complete software from scratch. However, existing benchmarks fail to assess this 0-to-1 generation capability due to two limitations: reliance on predefined scaffolds that ignore repository structure planning, and rigid white-box unit testing that lacks end-to-end behavioral validation. To bridge this gap, we introduce CLI-Tool-Bench, a structure-agnostic benchmark for evaluating the ground-up generation of Command-Line Interface (CLI) tools. It features 100 diverse real-world repositories evaluated via a black-box differential testing framework. Agent-generated software is executed in sandboxes, comparing system side effects and terminal outputs against human-written oracles using multi-tiered equivalence metrics. Evaluating seven state-of-the-art LLMs, we reveal that top models achieve under 43% success, highlighting the ongoing challenge of 0-to-1 generation. Furthermore, higher token consumption does not guarantee better performance, and agents tend to generate monolithic code. 

---
# ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding 

**Authors**: Xuanle Zhao, Xinyuan Cai, Xiang Cheng, Xiuyi Chen, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06685)  

**Abstract**: While Vision-Language Models (VLMs) have demonstrated significant potential in chemical visual understanding, current models are predominantly optimized for direct visual question-answering tasks. This paradigm often results in "black-box" systems that fail to utilize the inherent capability of Large Language Models (LLMs) to infer underlying reaction mechanisms. In this work, we introduce ChemVLR, a chemical VLM designed to prioritize reasoning within the perception process. Unlike conventional chemical VLMs, ChemVLR analyzes visual inputs in a fine-grained manner by explicitly identifying granular chemical descriptors, such as functional groups, prior to generating answers. This approach ensures the production of explicit and interpretable reasoning paths for complex visual chemical problems. To facilitate this methodology, we implement a cross-modality reverse-engineering strategy, combined with a rigorous filtering pipeline, to curate a large-scale reasoning-and-captioning dataset comprising 760k high-quality samples across molecular and reaction tasks. Furthermore, we adopt a three-stage training framework that systemically builds model perception and reasoning capacity. Experiments demonstrate that ChemVLR achieves state-of-the-art (SOTA) performance, surpassing both leading proprietary models and domain-specific open-source baselines. We also provide comprehensive ablation studies to validate our training strategy and data generation designs. Code and model weights will be available at this https URL. 

---
# Attention Flows: Tracing LLM Conceptual Engagement via Story Summaries 

**Authors**: Rebecca M. M. Hicke, Sil Hamilton, David Mimno, Ross Deans Kristensen-McLachlan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06416)  

**Abstract**: Although LLM context lengths have grown, there is evidence that their ability to integrate information across long-form texts has not kept pace. We evaluate one such understanding task: generating summaries of novels. When human authors of summaries compress a story, they reveal what they consider narratively important. Therefore, by comparing human and LLM-authored summaries, we can assess whether models mirror human patterns of conceptual engagement with texts. To measure conceptual engagement, we align sentences from 150 human-written novel summaries with the specific chapters they reference. We demonstrate the difficulty of this alignment task, which indicates the complexity of summarization as a task. We then generate and align additional summaries by nine state-of-the-art LLMs for each of the 150 reference texts. Comparing the human and model-authored summaries, we find both stylistic differences between the texts and differences in how humans and LLMs distribute their focus throughout a narrative, with models emphasizing the ends of texts. Comparing human narrative engagement with model attention mechanisms suggests explanations for degraded narrative comprehension and targets for future development. We release our dataset to support future research. 

---
# From Exposure to Internalization: Dual-Stream Calibration for In-context Clinical Reasoning 

**Authors**: Chuang Zhao, Hongke Zhao, Xiaofang Zhou, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.06262)  

**Abstract**: Contextual clinical reasoning demands robust inference grounded in complex, heterogeneous clinical records. While state-of-the-art fine-tuning, in-context learning (ICL), and retrieval-augmented generation (RAG) enable knowledge exposure, they often fall short of genuine contextual internalization: dynamically adjusting a model's internal representations to the subtle nuances of individual cases at inference time. To address this, we propose Dual-Stream Calibration (DSC), a test-time training framework that transcends superficial knowledge exposure to achieve deep internalization during inference. DSC facilitates input internalization by synergistically aligning two calibration streams. Unlike passive context exposure, the Semantic Calibration Stream enforces a deliberative reflection on core evidence, internalizing semantic anchors by minimizing entropy to stabilize generative trajectories. Simultaneously, the Structural Calibration Stream assimilates latent inferential dependencies through an iterative meta-learning objective. By training on specialized support sets at test-time, this stream enables the model to bridge the gap between external evidence and internal logic, synthesizing fragmented data into a coherent response. Our approach shifts the reasoning paradigm from passive attention-based matching to an active refinement of the latent inferential space. Validated against thirteen clinical datasets, DSC demonstrates superiority across three distinct task paradigms, consistently outstripping state-of-the-art baselines ranging from training-dependent models to test-time learning frameworks. 

---
# The Art of Building Verifiers for Computer Use Agents 

**Authors**: Corby Rosset, Pratyusha Sharma, Andrew Zhao, Miguel Gonzalez-Fernandez, Ahmed Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2604.06240)  

**Abstract**: Verifying the success of computer use agent (CUA) trajectories is a critical challenge: without reliable verification, neither evaluation nor training signal can be trusted. In this paper, we present lessons learned from building a best-in-class verifier for web tasks we call the Universal Verifier. We design the Universal Verifier around four key principles: 1) constructing rubrics with meaningful, non-overlapping criteria to reduce noise; 2) separating process and outcome rewards that yield complementary signals, capturing cases where an agent follows the right steps but gets blocked or succeeds through an unexpected path; 3) distinguishing between controllable and uncontrollable failures scored via a cascading-error-free strategy for finer-grained failure understanding; and 4) a divide-and-conquer context management scheme that attends to all screenshots in a trajectory, improving reliability on longer task horizons. We validate these findings on CUAVerifierBench, a new set of CUA trajectories with both process and outcome human labels, showing that our Universal Verifier agrees with humans as often as humans agree with each other. We report a reduction in false positive rates to near zero compared to baselines like WebVoyager ($\geq$ 45\%) and WebJudge ($\geq$ 22\%). We emphasize that these gains stem from the cumulative effect of the design choices above. We also find that an auto-research agent achieves 70\% of expert quality in 5\% of the time, but fails to discover all strategies required to replicate the Universal Verifier. We open-source our Universal Verifier system along with CUAVerifierBench; available at this https URL. 

---
# Unsupervised Neural Network for Automated Classification of Surgical Urgency Levels in Medical Transcriptions 

**Authors**: Sadaf Tabatabaee, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.06214)  

**Abstract**: Efficient classification of surgical procedures by urgency is paramount to optimize patient care and resource allocation within healthcare systems. This study introduces an unsupervised neural network approach to automatically categorize surgical transcriptions into three urgency levels: immediate, urgent, and elective. Leveraging BioClinicalBERT, a domain-specific language model, surgical transcripts are transformed into high-dimensional embeddings that capture their semantic nuances. These embeddings are subsequently clustered using both K-means and Deep Embedding Clustering (DEC) algorithms, in which DEC demonstrates superior performance in the formation of cohesive and well-separated clusters. To ensure clinical relevance and accuracy, the clustering results undergo validation through the Modified Delphi Method, which involves expert review and refinement. Following validation, a neural network that integrates Bidirectional Long Short-Term Memory (BiLSTM) layers with BioClinicalBERT embeddings is developed for classification tasks. The model is rigorously evaluated using cross-validation and metrics such as accuracy, precision, recall, and F1-score, which achieve robust performance and demonstrate strong generalization capabilities on unseen data. This unsupervised framework not only addresses the challenge of limited labeled data but also provides a scalable and reliable solution for real-time surgical prioritization, which ultimately enhances operational efficiency and patient outcomes in dynamic medical environments. 

---
# Probabilistic Language Tries: A Unified Framework for Compression, Decision Policies, and Execution Reuse 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2604.06228)  

**Abstract**: We introduce probabilistic language tries (PLTs), a unified representation that makes explicit the prefix structure implicitly defined by any generative model over sequences. By assigning to each outgoing edge the conditional probability of the corresponding token or action, a PLT simultaneously serves as: (i) an optimal lossless compressor via frequency-weighted interval encoding, generalizing arithmetic coding to model-conditioned distributions; (ii) a policy representation for sequential decision problems including games, search, and robotic control; and (iii) a memoization index that lets repeated inference queries be answered by structured retrieval rather than full model execution.
The central technical result is a prior-guided caching theorem: under a stationary generative distribution, a PLT-guided cache achieves strictly lower expected inference cost than any empirical-frequency cache for all query counts below a threshold that grows with the concentration of the prior. This converts O(n^2) transformer attention cost into an expected cost of p_r * O(log N) + (1 - p_r) * O(n^2), where p_r is the prior-estimated reuse probability and N is the artifact store size.
We further introduce a hybrid compression architecture decomposing any dataset into a PLT-covered majority and a sparse residual store, connecting arithmetic coding with Kolmogorov-style program representations and rate-distortion theory. We instantiate the framework across chess, web search, robotics, organizational workflows, and LLM inference, demonstrating that compression, decision making, and computational reuse are all derived from a single probability measure on sequence space. 

---
# Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA 

**Authors**: Kyubyung Chae, Jewon Yeom, Jeongjae Park, Seunghyun Bae, Ijun Jang, Hyunbin Jin, Jinkwan Jang, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.06173)  

**Abstract**: Legal QA benchmarks have predominantly focused on case law, overlooking the unique challenges of statute-centric regulatory reasoning. In statutory domains, relevant evidence is distributed across hierarchically linked documents, creating a statutory retrieval gap where conventional retrievers fail and models often hallucinate under incomplete context. We introduce SearchFireSafety, a structure- and safety-aware benchmark for statute-centric legal QA. Instantiated on fire-safety regulations as a representative case, the benchmark evaluates whether models can retrieve hierarchically fragmented evidence and safely abstain when statutory context is insufficient. SearchFireSafety adopts a dual-source evaluation framework combining real-world questions that require citation-aware retrieval and synthetic partial-context scenarios that stress-test hallucination and refusal behavior. Experiments across multiple large language models show that graph-guided retrieval substantially improves performance, but also reveal a critical safety trade-off: domain-adapted models are more likely to hallucinate when key statutory evidence is missing. Our findings highlight the need for benchmarks that jointly evaluate hierarchical retrieval and model safety in statute-centric regulatory settings. 

---
# LLM-Augmented Knowledge Base Construction For Root Cause Analysis 

**Authors**: Nguyen Phuc Tran, Brigitte Jaumard, Oscar Delgado, Tristan Glatard, Karthikeyan Premkumar, Kun Ni  

**Link**: [PDF](https://arxiv.org/pdf/2604.06171)  

**Abstract**: Communications networks now form the backbone of our digital world, with fast and reliable connectivity. However, even with appropriate redundancy and failover mechanisms, it is difficult to guarantee "five 9s" (99.999 %) reliability, requiring rapid and accurate root cause analysis (RCA) during outages. In the event of an outage, rapid and accurate RCA becomes essential to restore service and prevent future disruptions.
This study evaluates three Large Language Model (LLM) methodologies - Fine-Tuning, RAG, and a Hybrid approach - for constructing a Root Cause Analysis (RCA) Knowledge Base from support tickets. We compare their performance using a comprehensive suite of lexical and semantic similarity metrics. Our experiments on a real industrial dataset demonstrate that the generated knowledge base provides an excellent starting point for accelerating RCA tasks and improving network resilience. 

---
# Language Bias under Conflicting Information in Multilingual LLMs 

**Authors**: Robert Östling, Murathan Kurfalı  

**Link**: [PDF](https://arxiv.org/pdf/2604.07123)  

**Abstract**: Large Language Models (LLMs) have been shown to contain biases in the process of integrating conflicting information when answering questions. Here we ask whether such biases also exist with respect to which language is used for each conflicting piece of information. To answer this question, we extend the conflicting needles in a haystack paradigm to a multilingual setting and perform a comprehensive set of evaluations with naturalistic news domain data in five different languages, for a range of multilingual LLMs of different sizes. We find that all LLMs tested, including GPT-5.2, ignore the conflict and confidently assert only one of the possible answers in the large majority of cases. Furthermore, there is a consistent bias across models in which languages are preferred, with a general bias against Russian and, for the longest context lengths, in favor of Chinese. Both of these patterns are consistent between models trained inside and outside of mainland China, though somewhat stronger in the former category. 

---
# DTCRS: Dynamic Tree Construction for Recursive Summarization 

**Authors**: Guanran Luo, Zhongquan Jian, Wentao Qiu, Meihong Wang, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07012)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates the hallucination problem of Large Language Models (LLMs) by incorporating external knowledge. Recursive summarization constructs a hierarchical summary tree by clustering text chunks, integrating information from multiple parts of a document to provide evidence for abstractive questions involving multi-step reasoning. However, summary trees often contain a large number of redundant summary nodes, which not only increase construction time but may also negatively impact question answering. Moreover, recursive summarization is not suitable for all types of questions. We introduce DTCRS, a method that dynamically generates summary trees based on document structure and query semantics. DTCRS determines whether a summary tree is necessary by analyzing the question type. It then decomposes the question and uses the embeddings of sub-questions as initial cluster centers, reducing redundant summaries while improving the relevance between summaries and the question. Our approach significantly reduces summary tree construction time and achieves substantial improvements across three QA tasks. Additionally, we investigate the applicability of recursive summarization to different question types, providing valuable insights for future research. 

---
# Cognitive Loop of Thought: Reversible Hierarchical Markov Chain for Efficient Mathematical Reasoning 

**Authors**: Jia-Chen Zhang, Zheng Zhou, Yu-Jie Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.06805)  

**Abstract**: Multi-step Chain-of-Thought (CoT) has significantly advanced the mathematical reasoning capabilities of LLMs by leveraging explicit reasoning steps. However, the widespread adoption of Long CoT often results in sequence lengths that exceed manageable computational limits. While existing approaches attempt to alleviate this by reducing KV Cache redundancy via Markov chain-like structures, they introduce two critical limitations: inherent memorylessness (loss of context) and limited backward reasoning capability. To address these limitations, we propose a novel Chain-of-Thought framework based on Reversible Hierarchical Markov Chain, termed Cognitive Loop of Thought (CLoT), and a backward reasoning dataset CLoT-Instruct. In CLoT, problems are decomposed into sub-problems with hierarchical dependencies. Inspired by human cognitive processes, we introduce a backward verification mechanism at each hierarchical layer. Furthermore, we implement a pruning strategy: once higher-level sub-problems are verified, redundant lower-level sub-problems are pruned to maximize efficiency. This approach effectively mitigates error propagation and enhances reasoning robustness. Experiments on four mathematical benchmarks demonstrate the effectiveness of our method. Notably, on the AddSub dataset using GPT-4o-mini, CLoT achieves 99.0% accuracy, outperforming traditional CoT and CoT-SC by 4.1% and 2.9%, respectively. 

---
# AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation 

**Authors**: Guanran Luo, Wentao Qiu, Wanru Zhao, Wenhan Lv, Zhongquan Jian, Meihong Wang, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06812)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in long-form generation, yet their application is hindered by the hallucination problem. While Uncertainty Quantification (UQ) is essential for assessing reliability, the complex structure makes reliable aggregation across heterogeneous themes difficult, in addition, existing methods often overlook the nuance of neutral information and suffer from the high computational cost of fine-grained decomposition. To address these challenges, we propose AGSC (Adaptive Granularity and GMM-based Semantic Clustering), a UQ framework tailored for long-form generation. AGSC first uses NLI neutral probabilities as triggers to distinguish irrelevance from uncertainty, reducing unnecessary computation. It then applies Gaussian Mixture Model (GMM) soft clustering to model latent semantic themes and assign topic-aware weights for downstream aggregation. Experiments on BIO and LongFact show that AGSC achieves state-of-the-art correlation with factuality while reducing inference time by about 60% compared to full atomic decomposition. 

---
# StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference 

**Authors**: Zhirui Chen, Peiyang Liu, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.06746)  

**Abstract**: As Large Language Models (LLMs) scale to support context windows exceeding one million tokens, the linear growth of Key-Value (KV) cache imposes severe memory capacity and bandwidth bottlenecks, constraining the efficiency of long-context inference. Existing compression approaches typically prioritize tokens based on local saliency metrics to decouple prefill computation from decoding memory. However, these methods often rely on local saliency snapshots at a specific layer, thereby systematically discarding tokens that act as global information hubs across the network depth but appear temporarily dormant at the specific layer selected for pruning. To address this limitation, we propose StructKV, a structure-aware KV cache compression framework that introduces three core innovations: First, Global In-Degree Centrality aggregates attention patterns across the network depth to identify global information hubs. Second, Dynamic Pivot Detection utilizes information-theoretic metrics to adaptively locate the optimal layer for compression. Finally, Structural Propagation and Decoupling separates the computational budget from the memory storage budget. Experimental results on the LongBench and RULER benchmarks demonstrate that StructKV effectively preserves long-range dependencies and retrieval robustness. 

---
# Luwen Technical Report 

**Authors**: Yiquan Wu, Yuhang Liu, Yifei Liu, Ang Li, Siying Zhou, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06737)  

**Abstract**: Large language models have demonstrated remarkable capabilities across a wide range of natural language processing tasks, yet their application in the legal domain remains challenging due to the specialized terminology, complex reasoning requirements, and rapidly evolving legal knowledge involved. In this paper, we present Luwen, an open-source Chinese legal language model built upon the Baichuan foundation model through three key techniques: continual pre-training on a large-scale legal corpus, supervised fine-tuning with carefully curated legal instruction data, and retrieval-augmented generation integrated with a comprehensive legal knowledge base. We evaluate Luwen on five representative legal tasks spanning both prediction and generation settings, including legal judgment prediction, judicial examination, legal text summarization, law article question answering, and judicial decision reasoning. Experimental results show that Luwen outperforms several strong baselines, demonstrating the effectiveness of our approach in adapting general-purpose language models to the legal domain. 

---
# DiffuMask: Diffusion Language Model for Token-level Prompt Pruning 

**Authors**: Caleb Zheng, Jyotika Singh, Fang Tu, Weiyi Sun, Sujeeth Bharadwaj, Yassine Benajiba, Sujith Ravi, Eli Shlizerman, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.06627)  

**Abstract**: In-Context Learning and Chain-of-Thought prompting improve reasoning in large language models (LLMs). These typically come at the cost of longer, more expensive prompts that may contain redundant information. Prompt compression based on pruning offers a practical solution, yet existing methods rely on sequential token removal which is computationally intensive. We present DiffuMask, a diffusion-based framework integrating hierarchical shot-level and token-level pruning signals, that enables rapid and parallel prompt pruning via iterative mask prediction. DiffuMask substantially accelerates the compression process via masking multiple tokens in each denoising step. It offers tunable control over retained content, preserving essential reasoning context and achieving up to 80\% prompt length reduction. Meanwhile, it maintains or improves accuracy across in-domain, out-of-domain, and cross-model settings. Our results show that DiffuMask provides a generalizable and controllable framework for prompt compression, facilitating faster and more reliable in-context reasoning in LLMs. 

---
# Video-guided Machine Translation with Global Video Context 

**Authors**: Jian Chen, JinZe Lv, Zi Long, XiangHua Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06789)  

**Abstract**: Video-guided Multimodal Translation (VMT) has advanced significantly in recent years. However, most existing methods rely on locally aligned video segments paired one-to-one with subtitles, limiting their ability to capture global narrative context across multiple segments in long videos. To overcome this limitation, we propose a globally video-guided multimodal translation framework that leverages a pretrained semantic encoder and vector database-based subtitle retrieval to construct a context set of video segments closely related to the target subtitle semantics. An attention mechanism is employed to focus on highly relevant visual content, while preserving the remaining video features to retain broader contextual information. Furthermore, we design a region-aware cross-modal attention mechanism to enhance semantic alignment during translation. Experiments on a large-scale documentary translation dataset demonstrate that our method significantly outperforms baseline models, highlighting its effectiveness in long-video scenarios. 

---
# Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds of Large Language Models 

**Authors**: Marshall Brett  

**Link**: [PDF](https://arxiv.org/pdf/2604.06767)  

**Abstract**: Language models operate on discrete tokens but compute in continuous vector spaces, inducing a Voronoi tessellation over the representation manifold. We study this tessellation empirically on Qwen3.5-4B-Base, making two contributions. First, using float32 margin recomputation to resolve bfloat16 quantization artifacts, we validate Mabrok's (2026) linear scaling law of the expressibility gap with $R^2$ = 0.9997 - the strongest confirmation to date - and identify a mid-layer geometric ambiguity regime where margin geometry is anti-correlated with cross-entropy (layers 24-28, $\rho$ = -0.29) before crystallizing into alignment at the final layer ($\rho$ = 0.836).
Second, we show that the Voronoi tessellation of a converged model is reshapable through margin refinement procedures (MRP): short post-hoc optimization runs that widen token-decision margins without retraining. We compare direct margin maximization against Fisher information distance maximization across a dose-response sweep. Both methods find the same ceiling of ~16,300 correctable positions per 256K evaluated, but differ critically in collateral damage. Margin maximization damage escalates with intervention strength until corrections are overwhelmed. Fisher damage remains constant at ~5,300 positions across the validated range ($\lambda$ = 0.15-0.6), achieving +28% median margin improvement at $\lambda$ = 0.6 with invariant downstream benchmarks - a geometric reorganization that compresses the expressibility gap while preserving its scaling law. However, frequency and token-class audits reveal that gains concentrate in high-frequency structural tokens (84% of net corrections at $\lambda$ = 0.6), with content and entity-like contributions shrinking at higher $\lambda$. Fisher MRP is therefore a viable geometric polishing tool whose practical ceiling is set not by aggregate damage but by the uniformity of token-level benefit. 

---
