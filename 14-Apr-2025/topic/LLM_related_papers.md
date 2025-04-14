# RAG-VR: Leveraging Retrieval-Augmented Generation for 3D Question Answering in VR Environments 

**Authors**: Shiyi Ding, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08256)  

**Abstract**: Recent advances in large language models (LLMs) provide new opportunities for context understanding in virtual reality (VR). However, VR contexts are often highly localized and personalized, limiting the effectiveness of general-purpose LLMs. To address this challenge, we present RAG-VR, the first 3D question-answering system for VR that incorporates retrieval-augmented generation (RAG), which augments an LLM with external knowledge retrieved from a localized knowledge database to improve the answer quality. RAG-VR includes a pipeline for extracting comprehensive knowledge about virtual environments and user conditions for accurate answer generation. To ensure efficient retrieval, RAG-VR offloads the retrieval process to a nearby edge server and uses only essential information during retrieval. Moreover, we train the retriever to effectively distinguish among relevant, irrelevant, and hard-to-differentiate information in relation to questions. RAG-VR improves answer accuracy by 17.9%-41.8% and reduces end-to-end latency by 34.5%-47.3% compared with two baseline systems. 

---
# How Good Are Large Language Models for Course Recommendation in MOOCs? 

**Authors**: Boxuan Ma, Md Akib Zabed Khan, Tianyuan Yang, Agoritsa Polyzou, Shin'ichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08208)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language processing and are increasingly being integrated into recommendation systems. However, their potential in educational recommendation systems has yet to be fully explored. This paper investigates the use of LLMs as a general-purpose recommendation model, leveraging their vast knowledge derived from large-scale corpora for course recommendation tasks. We explore a variety of approaches, ranging from prompt-based methods to more advanced fine-tuning techniques, and compare their performance against traditional recommendation models. Extensive experiments were conducted on a real-world MOOC dataset, evaluating using LLMs as course recommendation systems across key dimensions such as accuracy, diversity, and novelty. Our results demonstrate that LLMs can achieve good performance comparable to traditional models, highlighting their potential to enhance educational recommendation systems. These findings pave the way for further exploration and development of LLM-based approaches in the context of educational recommendations. 

---
# Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images 

**Authors**: Boyang Deng, Songyou Peng, Kyle Genova, Gordon Wetzstein, Noah Snavely, Leonidas Guibas, Thomas Funkhouser  

**Link**: [PDF](https://arxiv.org/pdf/2504.08727)  

**Abstract**: We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.). See more results and interactive demos at this https URL. 

---
# DocAgent: A Multi-Agent System for Automated Code Documentation Generation 

**Authors**: Dayu Yang, Antoine Simoulin, Xin Qian, Xiaoyi Liu, Yuwei Cao, Zhaopu Teng, Grey Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08725)  

**Abstract**: High-quality code documentation is crucial for software development especially in the era of AI. However, generating it automatically using Large Language Models (LLMs) remains challenging, as existing approaches often produce incomplete, unhelpful, or factually incorrect outputs. We introduce DocAgent, a novel multi-agent collaborative system using topological code processing for incremental context building. Specialized agents (Reader, Searcher, Writer, Verifier, Orchestrator) then collaboratively generate documentation. We also propose a multi-faceted evaluation framework assessing Completeness, Helpfulness, and Truthfulness. Comprehensive experiments show DocAgent significantly outperforms baselines consistently. Our ablation study confirms the vital role of the topological processing order. DocAgent offers a robust approach for reliable code documentation generation in complex and proprietary repositories. 

---
# Voice Interaction With Conversational AI Could Facilitate Thoughtful Reflection and Substantive Revision in Writing 

**Authors**: Jiho Kim, Philippe Laban, Xiang 'Anthony' Chen, Kenneth C. Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2504.08687)  

**Abstract**: Writing well requires not only expressing ideas but also refining them through revision, a process facilitated by reflection. Prior research suggests that feedback delivered through dialogues, such as those in writing center tutoring sessions, can help writers reflect more thoughtfully on their work compared to static feedback. Recent advancements in multi-modal large language models (LLMs) now offer new possibilities for supporting interactive and expressive voice-based reflection in writing. In particular, we propose that LLM-generated static feedback can be repurposed as conversation starters, allowing writers to seek clarification, request examples, and ask follow-up questions, thereby fostering deeper reflection on their writing. We argue that voice-based interaction can naturally facilitate this conversational exchange, encouraging writers' engagement with higher-order concerns, facilitating iterative refinement of their reflections, and reduce cognitive load compared to text-based interactions. To investigate these effects, we propose a formative study exploring how text vs. voice input influence writers' reflection and subsequent revisions. Findings from this study will inform the design of intelligent and interactive writing tools, offering insights into how voice-based interactions with LLM-powered conversational agents can support reflection and revision. 

---
# Do LLMs trust AI regulation? Emerging behaviour of game-theoretic LLM agents 

**Authors**: Alessio Buscemi, Daniele Proverbio, Paolo Bova, Nataliya Balabanova, Adeela Bashir, Theodor Cimpeanu, Henrique Correia da Fonseca, Manh Hong Duong, Elias Fernandez Domingos, Antonio M. Fernandes, Marcus Krellner, Ndidi Bianca Ogbo, Simon T. Powers, Fernando P. Santos, Zia Ush Shamszaman, Zhao Song, Alessandro Di Stefano, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.08640)  

**Abstract**: There is general agreement that fostering trust and cooperation within the AI development ecosystem is essential to promote the adoption of trustworthy AI systems. By embedding Large Language Model (LLM) agents within an evolutionary game-theoretic framework, this paper investigates the complex interplay between AI developers, regulators and users, modelling their strategic choices under different regulatory scenarios. Evolutionary game theory (EGT) is used to quantitatively model the dilemmas faced by each actor, and LLMs provide additional degrees of complexity and nuances and enable repeated games and incorporation of personality traits. Our research identifies emerging behaviours of strategic AI agents, which tend to adopt more "pessimistic" (not trusting and defective) stances than pure game-theoretic agents. We observe that, in case of full trust by users, incentives are effective to promote effective regulation; however, conditional trust may deteriorate the "social pact". Establishing a virtuous feedback between users' trust and regulators' reputation thus appears to be key to nudge developers towards creating safe AI. However, the level at which this trust emerges may depend on the specific LLM used for testing. Our results thus provide guidance for AI regulation systems, and help predict the outcome of strategic LLM agents, should they be used to aid regulation itself. 

---
# MedRep: Medical Concept Representation for General Electronic Health Record Foundation Models 

**Authors**: Junmo Kim, Namkyeong Lee, Jiwon Kim, Kwangsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08329)  

**Abstract**: Electronic health record (EHR) foundation models have been an area ripe for exploration with their improved performance in various medical tasks. Despite the rapid advances, there exists a fundamental limitation: Processing unseen medical codes out of the vocabulary. This problem limits the generality of EHR foundation models and the integration of models trained with different vocabularies. To deal with this problem, we propose MedRep for EHR foundation models based on the observational medical outcome partnership (OMOP) common data model (CDM), providing the integrated medical concept representations and the basic data augmentation strategy for patient trajectories. For concept representation learning, we enrich the information of each concept with a minimal definition through large language model (LLM) prompts and enhance the text-based representations through graph ontology of OMOP vocabulary. Trajectory augmentation randomly replaces selected concepts with other similar concepts that have closely related representations to let the model practice with the concepts out-of-vocabulary. Finally, we demonstrate that EHR foundation models trained with MedRep better maintain the prediction performance in external datasets. Our code implementation is publicly available at this https URL. 

---
# Variability-Driven User-Story Generation using LLM and Triadic Concept Analysis 

**Authors**: Alexandre Bazin, Alain Gutierrez, Marianne Huchard, Pierre Martin, Yulin, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08666)  

**Abstract**: A widely used Agile practice for requirements is to produce a set of user stories (also called ``agile product backlog''), which roughly includes a list of pairs (role, feature), where the role handles the feature for a certain purpose. In the context of Software Product Lines, the requirements for a family of similar systems is thus a family of user-story sets, one per system, leading to a 3-dimensional dataset composed of sets of triples (system, role, feature). In this paper, we combine Triadic Concept Analysis (TCA) and Large Language Model (LLM) prompting to suggest the user-story set required to develop a new system relying on the variability logic of an existing system family. This process consists in 1) computing 3-dimensional variability expressed as a set of TCA implications, 2) providing the designer with intelligible design options, 3) capturing the designer's selection of options, 4) proposing a first user-story set corresponding to this selection, 5) consolidating its validity according to the implications identified in step 1, while completing it if necessary, and 6) leveraging LLM to have a more comprehensive website. This process is evaluated with a dataset comprising the user-story sets of 67 similar-purpose websites. 

---
# Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks 

**Authors**: Ye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08525)  

**Abstract**: Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. The full implementation of TME is available at this https URL. 

---
# Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning 

**Authors**: Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08672)  

**Abstract**: Advancing LLM reasoning skills has captivated wide interest. However, current post-training techniques rely heavily on supervisory signals, such as outcome supervision or auxiliary reward models, which face the problem of scalability and high annotation costs. This motivates us to enhance LLM reasoning without the need for external supervision. We introduce a generalizable and purely unsupervised self-training framework, named Genius. Without external auxiliary, Genius requires to seek the optimal response sequence in a stepwise manner and optimize the LLM. To explore the potential steps and exploit the optimal ones, Genius introduces a stepwise foresight re-sampling strategy to sample and estimate the step value by simulating future outcomes. Further, we recognize that the unsupervised setting inevitably induces the intrinsic noise and uncertainty. To provide a robust optimization, we propose an advantage-calibrated optimization (ACO) loss function to mitigate estimation inconsistencies. Combining these techniques together, Genius provides an advanced initial step towards self-improve LLM reasoning with general queries and without supervision, revolutionizing reasoning scaling laws given the vast availability of general queries. The code will be released at this https URL. 

---
# Fast-Slow-Thinking: Complex Task Solving with Large Language Models 

**Authors**: Yiliu Sun, Yanfang Zhang, Zicheng Zhao, Sheng Wan, Dacheng Tao, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08690)  

**Abstract**: Nowadays, Large Language Models (LLMs) have been gradually employed to solve complex tasks. To face the challenge, task decomposition has become an effective way, which proposes to divide a complex task into multiple simpler subtasks and then solve them separately so that the difficulty of the original task can be reduced. However, the performance of existing task decomposition methods can be suboptimal when the task contains overly complex logic and constraints. In this situation, the solution generated by LLMs may deviate from the original purpose of the task, or contain redundant or even erroneous content. Therefore, inspired by the fact that humans possess two thinking systems including fast thinking and slow thinking, this paper introduces a new task decomposition method termed ``Fast-Slow-Thinking'' (FST), which stimulates LLMs to solve tasks through the cooperation of Fast Thinking (FT) and Slow Thinking (ST) steps. Here FT focuses more on the general and concise aspect of the task, and ST focuses more on the details of the task. In FT, LLMs are prompted to remove the constraints of the original task, therefore simplifying it to a general and concise one. In ST, we recall the constraints removed in FT, so that LLMs can improve the answer generated in FT to meet the requirements of the original task. Therefore, our FST method enables LLMs to consider a complex problem via a human-like cognition process from coarse to fine, the effectiveness of which has been well demonstrated by the experiments on three types of tasks. 

---
# Adopting Large Language Models to Automated System Integration 

**Authors**: Robin D. Pesl  

**Link**: [PDF](https://arxiv.org/pdf/2504.08490)  

**Abstract**: Modern enterprise computing systems integrate numerous subsystems to resolve a common task by yielding emergent behavior. A widespread approach is using services implemented with Web technologies like REST or OpenAPI, which offer an interaction mechanism and service documentation standard, respectively. Each service represents a specific business functionality, allowing encapsulation and easier maintenance. Despite the reduced maintenance costs on an individual service level, increased integration complexity arises. Consequently, automated service composition approaches have arisen to mitigate this issue. Nevertheless, these approaches have not achieved high acceptance in practice due to their reliance on complex formal modeling. Within this Ph.D. thesis, we analyze the application of Large Language Models (LLMs) to automatically integrate the services based on a natural language input. The result is a reusable service composition, e.g., as program code. While not always generating entirely correct results, the result can still be helpful by providing integration engineers with a close approximation of a suitable solution, which requires little effort to become operational. Our research involves (i) introducing a software architecture for automated service composition using LLMs, (ii) analyzing Retrieval Augmented Generation (RAG) for service discovery, (iii) proposing a novel natural language query-based benchmark for service discovery, and (iv) extending the benchmark to complete service composition scenarios. We have presented our software architecture as Compositio Prompto, the analysis of RAG for service discovery, and submitted a proposal for the service discovery benchmark. Open topics are primarily the extension of the service discovery benchmark to service composition scenarios and the improvements of the service composition generation, e.g., using fine-tuning or LLM agents. 

---
# Beyond Self-Reports: Multi-Observer Agents for Personality Assessment in Large Language Models 

**Authors**: Yin Jou Huang, Rafik Hadfi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08399)  

**Abstract**: There is a growing interest in assessing the personality traits of Large language models (LLMs). However, traditional personality assessments based on self-report questionnaires may fail to capture their true behavioral nuances due to inherent biases and meta-knowledge contamination. This paper introduces a novel multi-observer framework for LLM personality assessment that draws inspiration from informant-report methods in psychology. Instead of relying solely on self-assessments, our approach employs multiple observer agents configured with a specific relationship context (e.g., family, friend, or workplace) to simulate interactive scenarios with a subject LLM. These observers engage in dialogues and subsequently provide ratings across the Big Five personality dimensions. Our experiments reveal that LLMs possess systematic biases in self-report personality ratings. Moreover, aggregating observer ratings effectively reduces non-systematic biases and achieves optimal reliability with 5-7 observers. The findings highlight the significant impact of relationship context on personality perception and demonstrate that a multi-observer paradigm yields a more robust and context-sensitive evaluation of LLM personality traits. 

---
# SortBench: Benchmarking LLMs based on their ability to sort lists 

**Authors**: Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2504.08312)  

**Abstract**: Sorting is a tedious but simple task for human intelligence and can be solved fairly easily algorithmically. However, for Large Language Models (LLMs) this task is surprisingly hard, as some properties of sorting are among known weaknesses of LLMs: being faithful to the input data, logical comparisons between values, and strictly differentiating between syntax (used for sorting) and semantics (typically learned by embeddings). Within this paper, we describe the new SortBench benchmark for LLMs that comes with different difficulties and that can be easily scaled in terms of difficulty. We apply this benchmark to seven state-of-the-art LLMs, including current test-time reasoning models. Our results show that while the o3-mini model is very capable at sorting in general, even this can be fooled if strings are defined to mix syntactical and semantical aspects, e.g., by asking to sort numbers written-out as word. Furthermore, all models have problems with the faithfulness to the input of long lists, i.e., they drop items and add new ones. Our results also show that test-time reasoning has a tendency to overthink problems which leads to performance degradation. Finally, models without test-time reasoning like GPT-4o are not much worse than reasoning models. 

---
# Large language models could be rote learners 

**Authors**: Yuyang Xu, Renjun Hu, Haochao Ying, Jian Wu, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.08300)  

**Abstract**: Multiple-choice question (MCQ) benchmarks are widely used for evaluating Large Language Models (LLMs), yet their reliability is undermined by benchmark contamination. In this study, we reframe contamination as an inherent aspect of learning and seek to disentangle genuine capability acquisition from superficial memorization in LLM evaluation. First, by analyzing model performance under different memorization conditions, we uncover a counterintuitive trend: LLMs perform worse on memorized MCQs than on non-memorized ones, indicating the coexistence of two distinct learning phenomena, i.e., rote memorization and genuine capability learning. To disentangle them, we propose TrinEval, a novel evaluation framework that reformulates MCQs into an alternative trinity format, reducing memorization while preserving knowledge assessment. Experiments validate TrinEval's effectiveness in reformulation, and its evaluation reveals that common LLMs may memorize by rote 20.5% of knowledge points (in MMLU on average). 

---
# ELSA: A Style Aligned Dataset for Emotionally Intelligent Language Generation 

**Authors**: Vishal Gandhi, Sagar Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08281)  

**Abstract**: Advancements in emotion aware language processing increasingly shape vital NLP applications ranging from conversational AI and affective computing to computational psychology and creative content generation. Existing emotion datasets either lack emotional granularity or fail to capture necessary stylistic diversity, limiting the advancement of effective emotion conditioned text generation systems. Seeking to bridge this crucial gap between granularity and style diversity, this paper introduces a novel systematically constructed dataset named ELSA Emotion and Language Style Alignment Dataset leveraging fine grained emotion taxonomies adapted from existing sources such as dair ai emotion dataset and GoEmotions taxonomy. This dataset comprises multiple emotionally nuanced variations of original sentences regenerated across distinct contextual styles such as conversational, formal, poetic, and narrative, using advanced Large Language Models LLMs. Rigorous computational evaluation using metrics such as perplexity, embedding variance, readability, lexical diversity, and semantic coherence measures validates the datasets emotional authenticity, linguistic fluency, and textual diversity. Comprehensive metric analyses affirm its potential to support deeper explorations into emotion conditioned style adaptive text generation. By enabling precision tuned emotionally nuanced language modeling, our dataset creates fertile ground for research on fine grained emotional control, prompt driven explanation, interpretability, and style adaptive expressive language generation with LLMs. 

---
# LLM for Comparative Narrative Analysis 

**Authors**: Leo Kampen, Carlos Rabat Villarreal, Louis Yu, Santu Karmaker, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08211)  

**Abstract**: In this paper, we conducted a Multi-Perspective Comparative Narrative Analysis (CNA) on three prominent LLMs: GPT-3.5, PaLM2, and Llama2. We applied identical prompts and evaluated their outputs on specific tasks, ensuring an equitable and unbiased comparison between various LLMs. Our study revealed that the three LLMs generated divergent responses to the same prompt, indicating notable discrepancies in their ability to comprehend and analyze the given task. Human evaluation was used as the gold standard, evaluating four perspectives to analyze differences in LLM performance. 

---
# DRAFT-ing Architectural Design Decisions using LLMs 

**Authors**: Rudra Dhar, Adyansh Kakran, Amey Karan, Karthik Vaidhyanathan, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2504.08207)  

**Abstract**: Architectural Knowledge Management (AKM) is crucial for software development but remains challenging due to the lack of standardization and high manual effort. Architecture Decision Records (ADRs) provide a structured approach to capture Architecture Design Decisions (ADDs), but their adoption is limited due to the manual effort involved and insufficient tool support. Our previous work has shown that Large Language Models (LLMs) can assist in generating ADDs. However, simply prompting the LLM does not produce quality ADDs. Moreover, using third-party LLMs raises privacy concerns, while self-hosting them poses resource challenges.
To this end, we experimented with different approaches like few-shot, retrieval-augmented generation (RAG) and fine-tuning to enhance LLM's ability to generate ADDs. Our results show that both techniques improve effectiveness. Building on this, we propose Domain Specific Retreival Augumented Few Shot Fine Tuninng, DRAFT, which combines the strengths of all these three approaches for more effective ADD generation. DRAFT operates in two phases: an offline phase that fine-tunes an LLM on generating ADDs augmented with retrieved examples and an online phase that generates ADDs by leveraging retrieved ADRs and the fine-tuned model.
We evaluated DRAFT against existing approaches on a dataset of 4,911 ADRs and various LLMs and analyzed them using automated metrics and human evaluations. Results show DRAFT outperforms all other approaches in effectiveness while maintaining efficiency. Our findings indicate that DRAFT can aid architects in drafting ADDs while addressing privacy and resource constraints. 

---
# Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices 

**Authors**: Shengyuan Ye, Bei Ouyang, Liekang Zeng, Tianyi Qian, Xiaowen Chu, Jian Tang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08242)  

**Abstract**: Generative large language models (LLMs) have garnered significant attention due to their exceptional capabilities in various AI tasks. Traditionally deployed in cloud datacenters, LLMs are now increasingly moving towards more accessible edge platforms to protect sensitive user data and ensure privacy preservation. The limited computational resources of individual edge devices, however, can result in excessively prolonged inference latency and overwhelmed memory usage. While existing research has explored collaborative edge computing to break the resource wall of individual devices, these solutions yet suffer from massive communication overhead and under-utilization of edge resources. Furthermore, they focus exclusively on optimizing the prefill phase, neglecting the crucial autoregressive decoding phase for generative LLMs. To address that, we propose Jupiter, a fast, scalable, and resource-efficient collaborative edge AI system for generative LLM inference. Jupiter introduces a flexible pipelined architecture as a principle and differentiates its system design according to the differentiated characteristics of the prefill and decoding phases. For prefill phase, Jupiter submits a novel intra-sequence pipeline parallelism and develops a meticulous parallelism planning strategy to maximize resource efficiency; For decoding, Jupiter devises an effective outline-based pipeline parallel decoding mechanism combined with speculative decoding, which further magnifies inference acceleration. Extensive evaluation based on realistic implementation demonstrates that Jupiter remarkably outperforms state-of-the-art approaches under various edge environment setups, achieving up to 26.1x end-to-end latency reduction while rendering on-par generation quality. 

---
# Can Reasoning LLMs Enhance Clinical Document Classification? 

**Authors**: Akram Mustafa, Usman Naseem, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08040)  

**Abstract**: Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications. 

---
# 'Neural howlround' in large language models: a self-reinforcing bias phenomenon, and a dynamic attenuation solution 

**Authors**: Seth Drake  

**Link**: [PDF](https://arxiv.org/pdf/2504.07992)  

**Abstract**: Large language model (LLM)-driven AI systems may exhibit an inference failure mode we term `neural howlround,' a self-reinforcing cognitive loop where certain highly weighted inputs become dominant, leading to entrenched response patterns resistant to correction. This paper explores the mechanisms underlying this phenomenon, which is distinct from model collapse and biased salience weighting. We propose an attenuation-based correction mechanism that dynamically introduces counterbalancing adjustments and can restore adaptive reasoning, even in `locked-in' AI systems. Additionally, we discuss some other related effects arising from improperly managed reinforcement. Finally, we outline potential applications of this mitigation strategy for improving AI robustness in real-world decision-making tasks. 

---
# SEAL: Steerable Reasoning Calibration of Large Language Models for Free 

**Authors**: Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07986)  

**Abstract**: Large Language Models (LLMs), such as OpenAI's o1-series have demonstrated compelling capabilities for complex reasoning tasks via the extended chain-of-thought (CoT) reasoning mechanism. However, recent studies reveal substantial redundancy in the CoT reasoning traces, which not only increases inference latency but also negatively impacts model performance by diverting attention to unnecessary reasoning paths. To address this issue, we investigate the internal reasoning structures of LLMs and categorize them into three primary thought types: execution, reflection, and transition thoughts. Moreover, our analysis reveals that excessive reflection and transition thoughts are strongly correlated with failure cases and these thought categories exhibit clear separation in the latent space. Based on these, we introduce SEAL (Steerable reasoning calibration), a training-free approach that seamlessly calibrates the CoT process, improving accuracy while demonstrating significant efficiency gains. SEAL consists of an offline stage for extracting the reasoning steering vector in the latent space, followed by an on-the-fly calibration of the reasoning trace through representation intervention using the steering vector. Notably, the steering vector exhibits strong transferability across various tasks. Extensive experiments across multiple models (DeepSeek-R1-Distill and QwQ-32B-Preview) and benchmarks (Math500, GSM8K, LiveCodeBench) validate the effectiveness of SEAL, up to a 11% improvement in accuracy while reducing reasoning tokens by 11.8% to 50.4%. Our code is publicly available at this https URL. 

---
# Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method 

**Authors**: Shurui Wu, Xinyi Huang, Dingxin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07983)  

**Abstract**: As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations. 

---
# Metamorphic Testing for Fairness Evaluation in Large Language Models: Identifying Intersectional Bias in LLaMA and GPT 

**Authors**: Harishwar Reddy, Madhusudan Srinivasan, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2504.07982)  

**Abstract**: Large Language Models (LLMs) have made significant strides in Natural Language Processing but remain vulnerable to fairness-related issues, often reflecting biases inherent in their training data. These biases pose risks, particularly when LLMs are deployed in sensitive areas such as healthcare, finance, and law. This paper introduces a metamorphic testing approach to systematically identify fairness bugs in LLMs. We define and apply a set of fairness-oriented metamorphic relations (MRs) to assess the LLaMA and GPT model, a state-of-the-art LLM, across diverse demographic inputs. Our methodology includes generating source and follow-up test cases for each MR and analyzing model responses for fairness violations. The results demonstrate the effectiveness of MT in exposing bias patterns, especially in relation to tone and sentiment, and highlight specific intersections of sensitive attributes that frequently reveal fairness faults. This research improves fairness testing in LLMs, providing a structured approach to detect and mitigate biases and improve model robustness in fairness-sensitive applications. 

---
# Playpen: An Environment for Exploring Learning Through Conversational Interaction 

**Authors**: Nicola Horst, Davide Mazzaccara, Antonia Schmidt, Michael Sullivan, Filippo Momentè, Luca Franceschetti, Philipp Sadler, Sherzod Hakimov, Alberto Testoni, Raffaella Bernardi, Raquel Fernández, Alexander Koller, Oliver Lemon, David Schlangen, Mario Giulianelli, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2504.08590)  

**Abstract**: Are we running out of learning signal? Predicting the next word in an existing text has turned out to be a powerful signal, at least at scale. But there are signs that we are running out of this resource. In recent months, interaction between learner and feedback-giver has come into focus, both for "alignment" (with a reward model judging the quality of instruction following attempts) and for improving "reasoning" (process- and outcome-based verifiers judging reasoning steps). In this paper, we explore to what extent synthetic interaction in what we call Dialogue Games -- goal-directed and rule-governed activities driven predominantly by verbal actions -- can provide a learning signal, and how this signal can be used. We introduce an environment for producing such interaction data (with the help of a Large Language Model as counterpart to the learner model), both offline and online. We investigate the effects of supervised fine-tuning on this data, as well as reinforcement learning setups such as DPO, and GRPO; showing that all of these approaches achieve some improvements in in-domain games, but only GRPO demonstrates the ability to generalise to out-of-domain games as well as retain competitive performance in reference-based tasks. We release the framework and the baseline training setups in the hope that this can foster research in this promising new direction. 

---
# TP-RAG: Benchmarking Retrieval-Augmented Large Language Model Agents for Spatiotemporal-Aware Travel Planning 

**Authors**: Hang Ni, Fan Liu, Xinyu Ma, Lixin Su, Shuaiqiang Wang, Dawei Yin, Hui Xiong, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08694)  

**Abstract**: Large language models (LLMs) have shown promise in automating travel planning, yet they often fall short in addressing nuanced spatiotemporal rationality. While existing benchmarks focus on basic plan validity, they neglect critical aspects such as route efficiency, POI appeal, and real-time adaptability. This paper introduces TP-RAG, the first benchmark tailored for retrieval-augmented, spatiotemporal-aware travel planning. Our dataset includes 2,348 real-world travel queries, 85,575 fine-grain annotated POIs, and 18,784 high-quality travel trajectory references sourced from online tourist documents, enabling dynamic and context-aware planning. Through extensive experiments, we reveal that integrating reference trajectories significantly improves spatial efficiency and POI rationality of the travel plan, while challenges persist in universality and robustness due to conflicting references and noisy data. To address these issues, we propose EvoRAG, an evolutionary framework that potently synergizes diverse retrieved trajectories with LLMs' intrinsic reasoning. EvoRAG achieves state-of-the-art performance, improving spatiotemporal compliance and reducing commonsense violation compared to ground-up and retrieval-augmented baselines. Our work underscores the potential of hybridizing Web knowledge with LLM-driven optimization, paving the way for more reliable and adaptive travel planning agents. 

---
# Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare 

**Authors**: Yonchanok Khaokaew, Flora D. Salim, Andreas Züfle, Hao Xue, Taylor Anderson, Matthew Scotch, David J Heslop  

**Link**: [PDF](https://arxiv.org/pdf/2504.08260)  

**Abstract**: Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies. 

---
# Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models 

**Authors**: Yu Fu, Haz Sameen Shahgir, Hui Liu, Xianfeng Tang, Qi He, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08202)  

**Abstract**: Recent advances in long-context models (LCMs), designed to handle extremely long input contexts, primarily focus on utilizing external contextual information, often leaving the influence of large language models' intrinsic knowledge underexplored. In this work, we investigate how this intrinsic knowledge affects content generation and demonstrate that its impact becomes increasingly pronounced as context length extends. Furthermore, we show that the model's ability to utilize intrinsic knowledge, which we call intrinsic retrieval ability, does not improve simultaneously with its ability to leverage contextual knowledge through extrinsic retrieval ability. Moreover, better extrinsic retrieval can interfere with the model's ability to use its own knowledge effectively, limiting its full potential. To bridge this gap, we design a simple yet effective Hybrid Needle-in-a-Haystack test that evaluates models based on their capabilities across both retrieval abilities, rather than solely emphasizing extrinsic retrieval ability. Our experimental results reveal that Qwen-2.5 models significantly outperform Llama-3.1 models, demonstrating superior intrinsic retrieval ability. Moreover, even the more powerful Llama-3.1-70B-Instruct model fails to exhibit better performance under LCM conditions, highlighting the importance of evaluating models from a dual-retrieval perspective. 

---
# DeepSeek vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization? 

**Authors**: Daniil Larionov, Sotaro Takeshita, Ran Zhang, Yanran Chen, Christoph Leiter, Zhipin Wang, Christian Greisinger, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2504.08120)  

**Abstract**: Reasoning-enabled large language models (LLMs) have recently demonstrated impressive performance in complex logical and mathematical tasks, yet their effectiveness in evaluating natural language generation remains unexplored. This study systematically compares reasoning-based LLMs (DeepSeek-R1 and OpenAI o3) with their non-reasoning counterparts across machine translation (MT) and text summarization (TS) evaluation tasks. We evaluate eight models across three architectural categories, including state-of-the-art reasoning models, their distilled variants (ranging from 8B to 70B parameters), and equivalent conventional, non-reasoning LLMs. Our experiments on WMT23 and SummEval benchmarks reveal that the benefits of reasoning capabilities are highly model and task-dependent: while OpenAI o3-mini models show consistent performance improvements with increased reasoning intensity, DeepSeek-R1 underperforms compared to its non-reasoning variant, with exception to certain aspects of TS evaluation. Correlation analysis demonstrates that increased reasoning token usage positively correlates with evaluation quality in o3-mini models. Furthermore, our results show that distillation of reasoning capabilities maintains reasonable performance in medium-sized models (32B) but degrades substantially in smaller variants (8B). This work provides the first comprehensive assessment of reasoning LLMs for NLG evaluation and offers insights into their practical use. 

---
# SWAN-GPT: An Efficient and Scalable Approach for Long-Context Language Modeling 

**Authors**: Krishna C. Puvvada, Faisal Ladhak, Santiago Akle Serrano, Cheng-Ping Hsieh, Shantanu Acharya, Somshubra Majumdar, Fei Jia, Samuel Kriman, Simeng Sun, Dima Rekesh, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2504.08719)  

**Abstract**: We present a decoder-only Transformer architecture that robustly generalizes to sequence lengths substantially longer than those seen during training. Our model, SWAN-GPT, interleaves layers without positional encodings (NoPE) and sliding-window attention layers equipped with rotary positional encodings (SWA-RoPE). Experiments demonstrate strong performance on sequence lengths significantly longer than the training length without the need for additional long-context training. This robust length extrapolation is achieved through our novel architecture, enhanced by a straightforward dynamic scaling of attention scores during inference. In addition, SWAN-GPT is more computationally efficient than standard GPT architectures, resulting in cheaper training and higher throughput. Further, we demonstrate that existing pre-trained decoder-only models can be efficiently converted to the SWAN architecture with minimal continued training, enabling longer contexts. Overall, our work presents an effective approach for scaling language models to longer contexts in a robust and efficient manner. 

---
# BiasCause: Evaluate Socially Biased Causal Reasoning of Large Language Models 

**Authors**: Tian Xie, Tongxin Yin, Vaishakh Keshava, Xueru Zhang, Siddhartha Reddy Jonnalagadda  

**Link**: [PDF](https://arxiv.org/pdf/2504.07997)  

**Abstract**: While large language models (LLMs) already play significant roles in society, research has shown that LLMs still generate content including social bias against certain sensitive groups. While existing benchmarks have effectively identified social biases in LLMs, a critical gap remains in our understanding of the underlying reasoning that leads to these biased outputs. This paper goes one step further to evaluate the causal reasoning process of LLMs when they answer questions eliciting social biases. We first propose a novel conceptual framework to classify the causal reasoning produced by LLMs. Next, we use LLMs to synthesize $1788$ questions covering $8$ sensitive attributes and manually validate them. The questions can test different kinds of causal reasoning by letting LLMs disclose their reasoning process with causal graphs. We then test 4 state-of-the-art LLMs. All models answer the majority of questions with biased causal reasoning, resulting in a total of $4135$ biased causal graphs. Meanwhile, we discover $3$ strategies for LLMs to avoid biased causal reasoning by analyzing the "bias-free" cases. Finally, we reveal that LLMs are also prone to "mistaken-biased" causal reasoning, where they first confuse correlation with causality to infer specific sensitive group names and then incorporate biased causal reasoning. 

---
# Large Language Models as Span Annotators 

**Authors**: Zdeněk Kasner, Vilém Zouhar, Patrícia Schmidtová, Ivan Kartáč, Kristýna Onderková, Ondřej Plátek, Dimitra Gkatzia, Saad Mahamood, Ondřej Dušek, Simone Balloccu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08697)  

**Abstract**: For high-quality texts, single-score metrics seldom provide actionable feedback. In contrast, span annotation - pointing out issues in the text by annotating their spans - can guide improvements and provide insights. Until recently, span annotation was limited to human annotators or fine-tuned encoder models. In this study, we automate span annotation with large language models (LLMs). We compare expert or skilled crowdworker annotators with open and proprietary LLMs on three tasks: data-to-text generation evaluation, machine translation evaluation, and propaganda detection in human-written texts. In our experiments, we show that LLMs as span annotators are straightforward to implement and notably more cost-efficient than human annotators. The LLMs achieve moderate agreement with skilled human annotators, in some scenarios comparable to the average agreement among the annotators themselves. Qualitative analysis shows that reasoning models outperform their instruction-tuned counterparts and provide more valid explanations for annotations. We release the dataset of more than 40k model and human annotations for further research. 

---
# Generating Fine Details of Entity Interactions 

**Authors**: Xinyi Gu, Jiayuan Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08714)  

**Abstract**: Images not only depict objects but also encapsulate rich interactions between them. However, generating faithful and high-fidelity images involving multiple entities interacting with each other, is a long-standing challenge. While pre-trained text-to-image models are trained on large-scale datasets to follow diverse text instructions, they struggle to generate accurate interactions, likely due to the scarcity of training data for uncommon object interactions. This paper introduces InterActing, an interaction-focused dataset with 1000 fine-grained prompts covering three key scenarios: (1) functional and action-based interactions, (2) compositional spatial relationships, and (3) multi-subject interactions. To address interaction generation challenges, we propose a decomposition-augmented refinement procedure. Our approach, DetailScribe, built on Stable Diffusion 3.5, leverages LLMs to decompose interactions into finer-grained concepts, uses a VLM to critique generated images, and applies targeted interventions within the diffusion process in refinement. Automatic and human evaluations show significantly improved image quality, demonstrating the potential of enhanced inference strategies. Our dataset and code are available at this https URL to facilitate future exploration of interaction-rich image generation. 

---
