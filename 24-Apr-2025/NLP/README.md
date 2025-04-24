# IberBench: LLM Evaluation on Iberian Languages 

**Authors**: José Ángel González, Ian Borrego Obrador, Álvaro Romo Herrero, Areg Mikael Sarvazyan, Mara Chinea-Ríos, Angelo Basile, Marc Franco-Salvador  

**Link**: [PDF](https://arxiv.org/pdf/2504.16921)  

**Abstract**: Large Language Models (LLMs) remain difficult to evaluate comprehensively, particularly for languages other than English, where high-quality data is often limited. Existing benchmarks and leaderboards are predominantly English-centric, with only a few addressing other languages. These benchmarks fall short in several key areas: they overlook the diversity of language varieties, prioritize fundamental Natural Language Processing (NLP) capabilities over tasks of industrial relevance, and are static. With these aspects in mind, we present IberBench, a comprehensive and extensible benchmark designed to assess LLM performance on both fundamental and industry-relevant NLP tasks, in languages spoken across the Iberian Peninsula and Ibero-America. IberBench integrates 101 datasets from evaluation campaigns and recent benchmarks, covering 22 task categories such as sentiment and emotion analysis, toxicity detection, and summarization. The benchmark addresses key limitations in current evaluation practices, such as the lack of linguistic diversity and static evaluation setups by enabling continual updates and community-driven model and dataset submissions moderated by a committee of experts. We evaluate 23 LLMs ranging from 100 million to 14 billion parameters and provide empirical insights into their strengths and limitations. Our findings indicate that (i) LLMs perform worse on industry-relevant tasks than in fundamental ones, (ii) performance is on average lower for Galician and Basque, (iii) some tasks show results close to random, and (iv) in other tasks LLMs perform above random but below shared task systems. IberBench offers open-source implementations for the entire evaluation pipeline, including dataset normalization and hosting, incremental evaluation of LLMs, and a publicly accessible leaderboard. 

---
# OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents 

**Authors**: Raghav Thind, Youran Sun, Ling Liang, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16918)  

**Abstract**: Optimization plays a vital role in scientific research and practical applications, but formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce \textbf{OptimAI}, a framework for solving \underline{Optim}ization problems described in natural language by leveraging LLM-powered \underline{AI} agents, achieving superior performance over current state-of-the-art methods. Our framework is built upon four key roles: (1) a \emph{formulator} that translates natural language problem descriptions into precise mathematical formulations; (2) a \emph{planner} that constructs a high-level solution strategy prior to execution; and (3) a \emph{coder} and a \emph{code critic} capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, allowing us to conveniently explore the synergistic effect of combining diverse models within a unified system. Our approach attains 88.1\% accuracy on the NLP4LP dataset and 71.2\% on the Optibench (non-linear w/o table) subset, reducing error rates by 58\% and 50\% respectively over prior best results. 

---
# Tracing Thought: Using Chain-of-Thought Reasoning to Identify the LLM Behind AI-Generated Text 

**Authors**: Shifali Agrahari, Sanasam Ranbir Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.16913)  

**Abstract**: In recent years, the detection of AI-generated text has become a critical area of research due to concerns about academic integrity, misinformation, and ethical AI deployment. This paper presents COT Fine-tuned, a novel framework for detecting AI-generated text and identifying the specific language model. responsible for generating the text. We propose a dual-task approach, where Task A involves classifying text as AI-generated or human-written, and Task B identifies the specific LLM behind the text. The key innovation of our method lies in the use of Chain-of-Thought reasoning, which enables the model to generate explanations for its predictions, enhancing transparency and interpretability. Our experiments demonstrate that COT Fine-tuned achieves high accuracy in both tasks, with strong performance in LLM identification and human-AI classification. We also show that the CoT reasoning process contributes significantly to the models effectiveness and interpretability. 

---
# Do Large Language Models know who did what to whom? 

**Authors**: Joseph M. Denning, Xiaohan, Bryor Snefjella, Idan A. Blank  

**Link**: [PDF](https://arxiv.org/pdf/2504.16884)  

**Abstract**: Large Language Models (LLMs) are commonly criticized for not understanding language. However, many critiques focus on cognitive abilities that, in humans, are distinct from language processing. Here, we instead study a kind of understanding tightly linked to language: inferring who did what to whom (thematic roles) in a sentence. Does the central training objective of LLMs-word prediction-result in sentence representations that capture thematic roles? In two experiments, we characterized sentence representations in four LLMs. In contrast to human similarity judgments, in LLMs the overall representational similarity of sentence pairs reflected syntactic similarity but not whether their agent and patient assignments were identical vs. reversed. Furthermore, we found little evidence that thematic role information was available in any subset of hidden units. However, some attention heads robustly captured thematic roles, independently of syntax. Therefore, LLMs can extract thematic roles but, relative to humans, this information influences their representations more weakly. 

---
# Planning with Diffusion Models for Target-Oriented Dialogue Systems 

**Authors**: Hanwen Du, Bo Peng, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2504.16858)  

**Abstract**: Target-Oriented Dialogue (TOD) remains a significant challenge in the LLM era, where strategic dialogue planning is crucial for directing conversations toward specific targets. However, existing dialogue planning methods generate dialogue plans in a step-by-step sequential manner, and may suffer from compounding errors and myopic actions. To address these limitations, we introduce a novel dialogue planning framework, DiffTOD, which leverages diffusion models to enable non-sequential dialogue planning. DiffTOD formulates dialogue planning as a trajectory generation problem with conditional guidance, and leverages a diffusion language model to estimate the likelihood of the dialogue trajectory. To optimize the dialogue action strategies, DiffTOD introduces three tailored guidance mechanisms for different target types, offering flexible guidance towards diverse TOD targets at test time. Extensive experiments across three diverse TOD settings show that DiffTOD can effectively perform non-myopic lookahead exploration and optimize action strategies over a long horizon through non-sequential dialogue planning, and demonstrates strong flexibility across complex and diverse dialogue scenarios. Our code and data are accessible through this https URL. 

---
# Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification 

**Authors**: Alexander Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2504.16856)  

**Abstract**: Most datasets for sentiment analysis lack context in which an opinion was expressed, often crucial for emotion understanding, and are mainly limited by a few emotion categories. Foundation large language models (LLMs) like GPT-4 suffer from over-predicting emotions and are too resource-intensive. We design an LLM-based data synthesis pipeline and leverage a large model, Mistral-7b, for the generation of training examples for more accessible, lightweight BERT-type encoder models. We focus on enlarging the semantic diversity of examples and propose grounding the generation into a corpus of narratives to produce non-repetitive story-character-centered utterances with unique contexts over 28 emotion classes. By running 700K inferences in 450 GPU hours, we contribute with the dataset of 100K contextual and also 300K context-less examples to cover both scenarios. We use it for fine-tuning pre-trained encoders, which results in several Emo Pillars models. We show that Emo Pillars models are highly adaptive to new domains when tuned to specific tasks such as GoEmotions, ISEAR, IEMOCAP, and EmoContext, reaching the SOTA performance on the first three. We also validate our dataset, conducting statistical analysis and human evaluation, and confirm the success of our measures in utterance diversification (although less for the neutral class) and context personalization, while pointing out the need for improved handling of out-of-taxonomy labels within the pipeline. 

---
# Monte Carlo Planning with Large Language Model for Text-Based Game Agents 

**Authors**: Zijing Shi, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.16855)  

**Abstract**: Text-based games provide valuable environments for language-based autonomous agents. However, planning-then-learning paradigms, such as those combining Monte Carlo Tree Search (MCTS) and reinforcement learning (RL), are notably time-consuming due to extensive iterations. Additionally, these algorithms perform uncertainty-driven exploration but lack language understanding and reasoning abilities. In this paper, we introduce the Monte Carlo planning with Dynamic Memory-guided Large language model (MC-DML) algorithm. MC-DML leverages the language understanding and reasoning capabilities of Large Language Models (LLMs) alongside the exploratory advantages of tree search algorithms. Specifically, we enhance LLMs with in-trial and cross-trial memory mechanisms, enabling them to learn from past experiences and dynamically adjust action evaluations during planning. We conduct experiments on a series of text-based games from the Jericho benchmark. Our results demonstrate that the MC-DML algorithm significantly enhances performance across various games at the initial planning phase, outperforming strong contemporary methods that require multiple iterations. This demonstrates the effectiveness of our algorithm, paving the way for more efficient language-grounded planning in complex environments. 

---
# GreenMind: A Next-Generation Vietnamese Large Language Model for Structured and Logical Reasoning 

**Authors**: Luu Quy Tung, Hoang Quoc Viet, Vo Trong Thu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16832)  

**Abstract**: Chain-of-Thought (CoT) is a robust approach for tackling LLM tasks that require intermediate reasoning steps prior to generating a final answer. In this paper, we present GreenMind-Medium-14B-R1, the Vietnamese reasoning model inspired by the finetuning strategy based on Group Relative Policy Optimization. We also leverage a high-quality Vietnamese synthesized reasoning dataset and design two reward functions to tackle the main limitations of this technique: (i) language mixing, where we explicitly detect the presence of biased language characters during the process of sampling tokens, and (ii) we leverage Sentence Transformer-based models to ensure that the generated reasoning content maintains factual correctness and does not distort the final output. Experimental results on the Vietnamese dataset from the VLSP 2023 Challenge demonstrate that our model outperforms prior works and enhances linguistic consistency in its responses. Furthermore, we extend our evaluation to SeaExam-a multilingual multiple-choice dataset, showing the effectiveness of our reasoning method compared to few-shot prompting techniques. 

---
# LLM-assisted Graph-RAG Information Extraction from IFC Data 

**Authors**: Sima Iranmanesh, Hadeel Saadany, Edlira Vakaj  

**Link**: [PDF](https://arxiv.org/pdf/2504.16813)  

**Abstract**: IFC data has become the general building information standard for collaborative work in the construction industry. However, IFC data can be very complicated because it allows for multiple ways to represent the same product information. In this research, we utilise the capabilities of LLMs to parse the IFC data with Graph Retrieval-Augmented Generation (Graph-RAG) technique to retrieve building object properties and their relations. We will show that, despite limitations due to the complex hierarchy of the IFC data, the Graph-RAG parsing enhances generative LLMs like GPT-4o with graph-based knowledge, enabling natural language query-response retrieval without the need for a complex pipeline. 

---
# Random Long-Context Access for Mamba via Hardware-aligned Hierarchical Sparse Attention 

**Authors**: Xiang Hu, Jiaqi Leng, Jun Zhao, Kewei Tu, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16795)  

**Abstract**: A key advantage of Recurrent Neural Networks (RNNs) over Transformers is their linear computational and space complexity enables faster training and inference for long sequences. However, RNNs are fundamentally unable to randomly access historical context, and simply integrating attention mechanisms may undermine their efficiency advantages. To overcome this limitation, we propose \textbf{H}ierarchical \textbf{S}parse \textbf{A}ttention (HSA), a novel attention mechanism that enhances RNNs with long-range random access flexibility while preserving their merits in efficiency and length generalization. HSA divides inputs into chunks, selecting the top-$k$ chunks and hierarchically aggregates information. The core innovation lies in learning token-to-chunk relevance based on fine-grained token-level information inside each chunk. This approach enhances the precision of chunk selection across both in-domain and out-of-domain context lengths. To make HSA efficient, we further introduce a hardware-aligned kernel design. By combining HSA with Mamba, we introduce RAMba, which achieves perfect accuracy in passkey retrieval across 64 million contexts despite pre-training on only 4K-length contexts, and significant improvements on various downstream tasks, with nearly constant memory footprint. These results show RAMba's huge potential in long-context modeling. 

---
# Credible plan-driven RAG method for Multi-hop Question Answering 

**Authors**: Ningning Zhang, Chi Zhang, Zhizhong Tan, Xingxing Yang, Weiping Deng, Wenyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16787)  

**Abstract**: Multi-hop question answering (QA) presents a considerable challenge for Retrieval-Augmented Generation (RAG), requiring the structured decomposition of complex queries into logical reasoning paths and the generation of dependable intermediate results. However, deviations in reasoning paths or errors in intermediate results, which are common in current RAG methods, may propagate and accumulate throughout the reasoning process, diminishing the accuracy of the answer to complex queries. To address this challenge, we propose the Plan-then-Act-and-Review (PAR RAG) framework, which is organized into three key stages: planning, act, and review, and aims to offer an interpretable and incremental reasoning paradigm for accurate and reliable multi-hop question answering by mitigating error this http URL RAG initially applies a top-down problem decomposition strategy, formulating a comprehensive plan that integrates multiple executable steps from a holistic viewpoint. This approach avoids the pitfalls of local optima common in traditional RAG methods, ensuring the accuracy of the entire reasoning path. Subsequently, PAR RAG incorporates a plan execution mechanism based on multi-granularity verification. By utilizing both coarse-grained similarity information and fine-grained relevant data, the framework thoroughly checks and adjusts intermediate results, ensuring process accuracy while effectively managing error propagation and amplification. Experimental results on multi-hop QA datasets demonstrate that the PAR RAG framework substantially outperforms existing state-of-the-art methods in key metrics, including EM and F1 scores. 

---
# MOOSComp: Improving Lightweight Long-Context Compressor via Mitigating Over-Smoothing and Incorporating Outlier Scores 

**Authors**: Fengwei Zhou, Jiafei Song, Wenjin Jason Li, Gengjian Xue, Zhikang Zhao, Yichao Lu, Bailin Na  

**Link**: [PDF](https://arxiv.org/pdf/2504.16786)  

**Abstract**: Recent advances in large language models have significantly improved their ability to process long-context input, but practical applications are challenged by increased inference time and resource consumption, particularly in resource-constrained environments. To address these challenges, we propose MOOSComp, a token-classification-based long-context compression method that enhances the performance of a BERT-based compressor by mitigating the over-smoothing problem and incorporating outlier scores. In the training phase, we add an inter-class cosine similarity loss term to penalize excessively similar token representations, thereby improving the token classification accuracy. During the compression phase, we introduce outlier scores to preserve rare but critical tokens that are prone to be discarded in task-agnostic compression. These scores are integrated with the classifier's output, making the compressor more generalizable to various tasks. Superior performance is achieved at various compression ratios on long-context understanding and reasoning benchmarks. Moreover, our method obtains a speedup of 3.3x at a 4x compression ratio on a resource-constrained mobile device. 

---
# Evaluation Framework for AI Systems in "the Wild" 

**Authors**: Sarah Jabbour, Trenton Chang, Anindya Das Antar, Joseph Peper, Insu Jang, Jiachen Liu, Jae-Won Chung, Shiqi He, Michael Wellman, Bryan Goodman, Elizabeth Bondi-Kelly, Kevin Samy, Rada Mihalcea, Mosharaf Chowhury, David Jurgens, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16778)  

**Abstract**: Generative AI (GenAI) models have become vital across industries, yet current evaluation methods have not adapted to their widespread use. Traditional evaluations often rely on benchmarks and fixed datasets, frequently failing to reflect real-world performance, which creates a gap between lab-tested outcomes and practical applications. This white paper proposes a comprehensive framework for how we should evaluate real-world GenAI systems, emphasizing diverse, evolving inputs and holistic, dynamic, and ongoing assessment approaches. The paper offers guidance for practitioners on how to design evaluation methods that accurately reflect real-time capabilities, and provides policymakers with recommendations for crafting GenAI policies focused on societal impacts, rather than fixed performance numbers or parameter sizes. We advocate for holistic frameworks that integrate performance, fairness, and ethics and the use of continuous, outcome-oriented methods that combine human and automated assessments while also being transparent to foster trust among stakeholders. Implementing these strategies ensures GenAI models are not only technically proficient but also ethically responsible and impactful. 

---
# How Effective are Generative Large Language Models in Performing Requirements Classification? 

**Authors**: Waad Alhoshan, Alessio Ferrari, Liping Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16768)  

**Abstract**: In recent years, transformer-based large language models (LLMs) have revolutionised natural language processing (NLP), with generative models opening new possibilities for tasks that require context-aware text generation. Requirements engineering (RE) has also seen a surge in the experimentation of LLMs for different tasks, including trace-link detection, regulatory compliance, and others. Requirements classification is a common task in RE. While non-generative LLMs like BERT have been successfully applied to this task, there has been limited exploration of generative LLMs. This gap raises an important question: how well can generative LLMs, which produce context-aware outputs, perform in requirements classification? In this study, we explore the effectiveness of three generative LLMs-Bloom, Gemma, and Llama-in performing both binary and multi-class requirements classification. We design an extensive experimental study involving over 400 experiments across three widely used datasets (PROMISE NFR, Functional-Quality, and SecReq). Our study concludes that while factors like prompt design and LLM architecture are universally important, others-such as dataset variations-have a more situational impact, depending on the complexity of the classification task. This insight can guide future model development and deployment strategies, focusing on optimising prompt structures and aligning model architectures with task-specific needs for improved performance. 

---
# HEMA : A Hippocampus-Inspired Extended Memory Architecture for Long-Context AI Conversations 

**Authors**: Kwangseob Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2504.16754)  

**Abstract**: Large language models (LLMs) struggle with maintaining coherence in extended conversations spanning hundreds of turns, despite performing well within their context windows. This paper introduces HEMA (Hippocampus-Inspired Extended Memory Architecture), a dual-memory system inspired by human cognitive processes. HEMA combines Compact Memory - a continuously updated one-sentence summary preserving global narrative coherence, and Vector Memory - an episodic store of chunk embeddings queried via cosine similarity. When integrated with a 6B-parameter transformer, HEMA maintains coherent dialogues beyond 300 turns while keeping prompt length under 3,500 tokens. Experimental results show substantial improvements: factual recall accuracy increases from 41% to 87%, and human-rated coherence improves from 2.7 to 4.3 on a 5-point scale. With 10K indexed chunks, Vector Memory achieves P@5 >= 0.80 and R@50 >= 0.74, doubling the area under the precision-recall curve compared to summarization-only approaches. Ablation studies reveal two key insights: semantic forgetting through age-weighted pruning reduces retrieval latency by 34% with minimal recall loss, and a two-level summary hierarchy prevents cascade errors in ultra-long conversations exceeding 1,000 turns. HEMA demonstrates that combining verbatim recall with semantic continuity provides a practical solution for privacy-aware conversational AI capable of month-long dialogues without model retraining. 

---
# A Post-trainer's Guide to Multilingual Training Data: Uncovering Cross-lingual Transfer Dynamics 

**Authors**: Luisa Shimabucoro, Ahmet Ustun, Marzieh Fadaee, Sebastian Ruder  

**Link**: [PDF](https://arxiv.org/pdf/2504.16677)  

**Abstract**: In order for large language models to be useful across the globe, they are fine-tuned to follow instructions on multilingual data. Despite the ubiquity of such post-training, a clear understanding of the dynamics that enable cross-lingual transfer remains elusive. This study examines cross-lingual transfer (CLT) dynamics in realistic post-training settings. We study two model families of up to 35B parameters in size trained on carefully controlled mixtures of multilingual data on three generative tasks with varying levels of complexity (summarization, instruction following, and mathematical reasoning) in both single-task and multi-task instruction tuning settings. Overall, we find that the dynamics of cross-lingual transfer and multilingual performance cannot be explained by isolated variables, varying depending on the combination of post-training settings. Finally, we identify the conditions that lead to effective cross-lingual transfer in practice. 

---
# TIFIN India at SemEval-2025: Harnessing Translation to Overcome Multilingual IR Challenges in Fact-Checked Claim Retrieval 

**Authors**: Prasanna Devadiga, Arya Suneesh, Pawan Kumar Rajpoot, Bharatdeep Hazarika, Aditya U Baliga  

**Link**: [PDF](https://arxiv.org/pdf/2504.16627)  

**Abstract**: We address the challenge of retrieving previously fact-checked claims in monolingual and crosslingual settings - a critical task given the global prevalence of disinformation. Our approach follows a two-stage strategy: a reliable baseline retrieval system using a fine-tuned embedding model and an LLM-based reranker. Our key contribution is demonstrating how LLM-based translation can overcome the hurdles of multilingual information retrieval. Additionally, we focus on ensuring that the bulk of the pipeline can be replicated on a consumer GPU. Our final integrated system achieved a success@10 score of 0.938 and 0.81025 on the monolingual and crosslingual test sets, respectively. 

---
# Debunking with Dialogue? Exploring AI-Generated Counterspeech to Challenge Conspiracy Theories 

**Authors**: Mareike Lisker, Christina Gottschalk, Helena Mihaljević  

**Link**: [PDF](https://arxiv.org/pdf/2504.16604)  

**Abstract**: Counterspeech is a key strategy against harmful online content, but scaling expert-driven efforts is challenging. Large Language Models (LLMs) present a potential solution, though their use in countering conspiracy theories is under-researched. Unlike for hate speech, no datasets exist that pair conspiracy theory comments with expert-crafted counterspeech. We address this gap by evaluating the ability of GPT-4o, Llama 3, and Mistral to effectively apply counterspeech strategies derived from psychological research provided through structured prompts. Our results show that the models often generate generic, repetitive, or superficial results. Additionally, they over-acknowledge fear and frequently hallucinate facts, sources, or figures, making their prompt-based use in practical applications problematic. 

---
# Comparing Large Language Models and Traditional Machine Translation Tools for Translating Medical Consultation Summaries: A Pilot Study 

**Authors**: Andy Li, Wei Zhou, Rashina Hoda, Chris Bain, Peter Poon  

**Link**: [PDF](https://arxiv.org/pdf/2504.16601)  

**Abstract**: This study evaluates how well large language models (LLMs) and traditional machine translation (MT) tools translate medical consultation summaries from English into Arabic, Chinese, and Vietnamese. It assesses both patient, friendly and clinician, focused texts using standard automated metrics. Results showed that traditional MT tools generally performed better, especially for complex texts, while LLMs showed promise, particularly in Vietnamese and Chinese, when translating simpler summaries. Arabic translations improved with complexity due to the language's morphology. Overall, while LLMs offer contextual flexibility, they remain inconsistent, and current evaluation metrics fail to capture clinical relevance. The study highlights the need for domain-specific training, improved evaluation methods, and human oversight in medical translation. 

---
# PIS: Linking Importance Sampling and Attention Mechanisms for Efficient Prompt Compression 

**Authors**: Lizhe Chen, Binjia Zhou, Yuyao Ge, Jiayi Chen, Shiguang NI  

**Link**: [PDF](https://arxiv.org/pdf/2504.16574)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress, demonstrating unprecedented capabilities across various natural language processing tasks. However, the high costs associated with such exceptional performance limit the widespread adoption of LLMs, highlighting the need for prompt compression. Existing prompt compression methods primarily rely on heuristic truncation or abstractive summarization techniques, which fundamentally overlook the intrinsic mechanisms of LLMs and lack a systematic evaluation of token importance for generation. In this work, we introduce Prompt Importance Sampling (PIS), a novel compression framework that dynamically compresses prompts by sampling important tokens based on the analysis of attention scores of hidden states. PIS employs a dual-level compression mechanism: 1) at the token level, we quantify saliency using LLM-native attention scores and implement adaptive compression through a lightweight 9-layer reinforcement learning (RL) network; 2) at the semantic level, we propose a Russian roulette sampling strategy for sentence-level importance sampling. Comprehensive evaluations across multiple domain benchmarks demonstrate that our method achieves state-of-the-art compression performance. Notably, our framework serendipitously enhances reasoning efficiency through optimized context structuring. This work advances prompt engineering by offering both theoretical grounding and practical efficiency in context management for LLMs. 

---
# Transformers for Complex Query Answering over Knowledge Hypergraphs 

**Authors**: Hong Ting Tsang, Zihao Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.16537)  

**Abstract**: Complex Query Answering (CQA) has been extensively studied in recent years. In order to model data that is closer to real-world distribution, knowledge graphs with different modalities have been introduced. Triple KGs, as the classic KGs composed of entities and relations of arity 2, have limited representation of real-world facts. Real-world data is more sophisticated. While hyper-relational graphs have been introduced, there are limitations in representing relationships of varying arity that contain entities with equal contributions. To address this gap, we sampled new CQA datasets: JF17k-HCQA and M-FB15k-HCQA. Each dataset contains various query types that include logical operations such as projection, negation, conjunction, and disjunction. In order to answer knowledge hypergraph (KHG) existential first-order queries, we propose a two-stage transformer model, the Logical Knowledge Hypergraph Transformer (LKHGT), which consists of a Projection Encoder for atomic projection and a Logical Encoder for complex logical operations. Both encoders are equipped with Type Aware Bias (TAB) for capturing token interactions. Experimental results on CQA datasets show that LKHGT is a state-of-the-art CQA method over KHG and is able to generalize to out-of-distribution query types. 

---
# QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining 

**Authors**: Fengze Liu, Weidong Zhou, Binbin Liu, Zhimiao Yu, Yifan Zhang, Haobin Lin, Yifeng Yu, Xiaohuan Zhou, Taifeng Wang, Yong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16511)  

**Abstract**: Quality and diversity are two critical metrics for the training data of large language models (LLMs), positively impacting performance. Existing studies often optimize these metrics separately, typically by first applying quality filtering and then adjusting data proportions. However, these approaches overlook the inherent trade-off between quality and diversity, necessitating their joint consideration. Given a fixed training quota, it is essential to evaluate both the quality of each data point and its complementary effect on the overall dataset. In this paper, we introduce a unified data selection framework called QuaDMix, which automatically optimizes the data distribution for LLM pretraining while balancing both quality and diversity. Specifically, we first propose multiple criteria to measure data quality and employ domain classification to distinguish data points, thereby measuring overall diversity. QuaDMix then employs a unified parameterized data sampling function that determines the sampling probability of each data point based on these quality and diversity related labels. To accelerate the search for the optimal parameters involved in the QuaDMix framework, we conduct simulated experiments on smaller models and use LightGBM for parameters searching, inspired by the RegMix method. Our experiments across diverse models and datasets demonstrate that QuaDMix achieves an average performance improvement of 7.2% across multiple benchmarks. These results outperform the independent strategies for quality and diversity, highlighting the necessity and ability to balance data quality and diversity. 

---
# T-VEC: A Telecom-Specific Vectorization Model with Enhanced Semantic Understanding via Deep Triplet Loss Fine-Tuning 

**Authors**: Vignesh Ethiraj, Sidhanth Menon, Divya Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2504.16460)  

**Abstract**: The specialized vocabulary and complex concepts of the telecommunications industry present significant challenges for standard Natural Language Processing models. Generic text embeddings often fail to capture telecom-specific semantics, hindering downstream task performance. We introduce T-VEC (Telecom Vectorization Model), a novel embedding model tailored for the telecom domain through deep fine-tuning. Developed by NetoAI, T-VEC is created by adapting the state-of-the-art gte-Qwen2-1.5B-instruct model using a triplet loss objective on a meticulously curated, large-scale dataset of telecom-specific data. Crucially, this process involved substantial modification of weights across 338 layers of the base model, ensuring deep integration of domain knowledge, far exceeding superficial adaptation techniques. We quantify this deep change via weight difference analysis. A key contribution is the development and open-sourcing (MIT License) of the first dedicated telecom-specific tokenizer, enhancing the handling of industry jargon. T-VEC achieves a leading average MTEB score (0.825) compared to established models and demonstrates vastly superior performance (0.9380 vs. less than 0.07) on our internal telecom-specific triplet evaluation benchmark, indicating an exceptional grasp of domain-specific nuances, visually confirmed by improved embedding separation. This work positions NetoAI at the forefront of telecom AI innovation, providing the community with a powerful, deeply adapted, open-source tool. 

---
# EMRModel: A Large Language Model for Extracting Medical Consultation Dialogues into Structured Medical Records 

**Authors**: Shuguang Zhao, Qiangzhong Feng, Zhiyang He, Peipei Sun, Yingying Wang, Xiaodong Tao, Xiaoliang Lu, Mei Cheng, Xinyue Wu, Yanyan Wang, Wei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16448)  

**Abstract**: Medical consultation dialogues contain critical clinical information, yet their unstructured nature hinders effective utilization in diagnosis and treatment. Traditional methods, relying on rule-based or shallow machine learning techniques, struggle to capture deep and implicit semantics. Recently, large pre-trained language models and Low-Rank Adaptation (LoRA), a lightweight fine-tuning method, have shown promise for structured information extraction. We propose EMRModel, a novel approach that integrates LoRA-based fine-tuning with code-style prompt design, aiming to efficiently convert medical consultation dialogues into structured electronic medical records (EMRs). Additionally, we construct a high-quality, realistically grounded dataset of medical consultation dialogues with detailed annotations. Furthermore, we introduce a fine-grained evaluation benchmark for medical consultation information extraction and provide a systematic evaluation methodology, advancing the optimization of medical natural language processing (NLP) models. Experimental results show EMRModel achieves an F1 score of 88.1%, improving by49.5% over standard pre-trained models. Compared to traditional LoRA fine-tuning methods, our model shows superior performance, highlighting its effectiveness in structured medical record extraction tasks. 

---
# Can Large Language Models Help Multimodal Language Analysis? MMLA: A Comprehensive Benchmark 

**Authors**: Hanlei Zhang, Zhuohang Li, Yeshuang Zhu, Hua Xu, Peiwu Wang, Jinchao Zhang, Jie Zhou, Haige Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16427)  

**Abstract**: Multimodal language analysis is a rapidly evolving field that leverages multiple modalities to enhance the understanding of high-level semantics underlying human conversational utterances. Despite its significance, little research has investigated the capability of multimodal large language models (MLLMs) to comprehend cognitive-level semantics. In this paper, we introduce MMLA, a comprehensive benchmark specifically designed to address this gap. MMLA comprises over 61K multimodal utterances drawn from both staged and real-world scenarios, covering six core dimensions of multimodal semantics: intent, emotion, dialogue act, sentiment, speaking style, and communication behavior. We evaluate eight mainstream branches of LLMs and MLLMs using three methods: zero-shot inference, supervised fine-tuning, and instruction tuning. Extensive experiments reveal that even fine-tuned models achieve only about 60%~70% accuracy, underscoring the limitations of current MLLMs in understanding complex human language. We believe that MMLA will serve as a solid foundation for exploring the potential of large language models in multimodal language analysis and provide valuable resources to advance this field. The datasets and code are open-sourced at this https URL. 

---
# Evaluating Multi-Hop Reasoning in Large Language Models: A Chemistry-Centric Case Study 

**Authors**: Mohammad Khodadad, Ali Shiraee Kasmaee, Mahdi Astaraki, Nicholas Sherck, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2504.16414)  

**Abstract**: In this study, we introduced a new benchmark consisting of a curated dataset and a defined evaluation process to assess the compositional reasoning capabilities of large language models within the chemistry domain. We designed and validated a fully automated pipeline, verified by subject matter experts, to facilitate this task. Our approach integrates OpenAI reasoning models with named entity recognition (NER) systems to extract chemical entities from recent literature, which are then augmented with external knowledge bases to form a comprehensive knowledge graph. By generating multi-hop questions across these graphs, we assess LLM performance in both context-augmented and non-context augmented settings. Our experiments reveal that even state-of-the-art models face significant challenges in multi-hop compositional reasoning. The results reflect the importance of augmenting LLMs with document retrieval, which can have a substantial impact on improving their performance. However, even perfect retrieval accuracy with full context does not eliminate reasoning errors, underscoring the complexity of compositional reasoning. This work not only benchmarks and highlights the limitations of current LLMs but also presents a novel data generation pipeline capable of producing challenging reasoning datasets across various domains. Overall, this research advances our understanding of reasoning in computational linguistics. 

---
# Out-of-the-Box Conditional Text Embeddings from Large Language Models 

**Authors**: Kosuke Yamada, Peinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16411)  

**Abstract**: Conditional text embedding is a proposed representation that captures the shift in perspective on texts when conditioned on a specific aspect. Previous methods have relied on extensive training data for fine-tuning models, leading to challenges in terms of labor and resource costs. We propose PonTE, a novel unsupervised conditional text embedding method that leverages a causal large language model and a conditional prompt. Through experiments on conditional semantic text similarity and text clustering, we demonstrate that PonTE can generate useful conditional text embeddings and achieve performance comparable to supervised methods without fine-tuning. We also show the interpretability of text embeddings with PonTE by analyzing word generation following prompts and embedding visualization. 

---
# Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation 

**Authors**: Jiahao Yuan, Xingzhe Sun, Xing Yu, Jingwen Wang, Dehui Du, Zhiqing Cui, Zixiang Di  

**Link**: [PDF](https://arxiv.org/pdf/2504.16408)  

**Abstract**: The XLLM@ACL2025 Shared Task-III formulates a low-resource structural reasoning task that challenges LLMs to generate interpretable, step-by-step rationales with minimal labeled data. We present Less is More, the third-place winning approach in the XLLM@ACL2025 Shared Task-III, which focuses on structured reasoning from only 24 labeled examples. Our approach leverages a multi-agent framework with reverse-prompt induction, retrieval-augmented reasoning synthesis via GPT-4o, and dual-stage reward-guided filtering to distill high-quality supervision across three subtasks: question parsing, CoT parsing, and step-level verification. All modules are fine-tuned from Meta-Llama-3-8B-Instruct under a unified LoRA+ setup. By combining structure validation with reward filtering across few-shot and zero-shot prompts, our pipeline consistently improves structure reasoning quality. These results underscore the value of controllable data distillation in enhancing structured inference under low-resource constraints. Our code is available at this https URL. 

---
# ConTextual: Improving Clinical Text Summarization in LLMs with Context-preserving Token Filtering and Knowledge Graphs 

**Authors**: Fahmida Liza Piya, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2504.16394)  

**Abstract**: Unstructured clinical data can serve as a unique and rich source of information that can meaningfully inform clinical practice. Extracting the most pertinent context from such data is critical for exploiting its true potential toward optimal and timely decision-making in patient care. While prior research has explored various methods for clinical text summarization, most prior studies either process all input tokens uniformly or rely on heuristic-based filters, which can overlook nuanced clinical cues and fail to prioritize information critical for decision-making. In this study, we propose Contextual, a novel framework that integrates a Context-Preserving Token Filtering method with a Domain-Specific Knowledge Graph (KG) for contextual augmentation. By preserving context-specific important tokens and enriching them with structured knowledge, ConTextual improves both linguistic coherence and clinical fidelity. Our extensive empirical evaluations on two public benchmark datasets demonstrate that ConTextual consistently outperforms other baselines. Our proposed approach highlights the complementary role of token-level filtering and structured retrieval in enhancing both linguistic and clinical integrity, as well as offering a scalable solution for improving precision in clinical text generation. 

---
# SplitReason: Learning To Offload Reasoning 

**Authors**: Yash Akhauri, Anthony Fei, Chi-Chih Chang, Ahmed F. AbouElhamayed, Yueying Li, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2504.16379)  

**Abstract**: Reasoning in large language models (LLMs) tends to produce substantially longer token generation sequences than simpler language modeling tasks. This extended generation length reflects the multi-step, compositional nature of reasoning and is often correlated with higher solution accuracy. From an efficiency perspective, longer token generation exacerbates the inherently sequential and memory-bound decoding phase of LLMs. However, not all parts of this expensive reasoning process are equally difficult to generate. We leverage this observation by offloading only the most challenging parts of the reasoning process to a larger, more capable model, while performing most of the generation with a smaller, more efficient model; furthermore, we teach the smaller model to identify these difficult segments and independently trigger offloading when needed. To enable this behavior, we annotate difficult segments across 18k reasoning traces from the OpenR1-Math-220k chain-of-thought (CoT) dataset. We then apply supervised fine-tuning (SFT) and reinforcement learning fine-tuning (RLFT) to a 1.5B-parameter reasoning model, training it to learn to offload the most challenging parts of its own reasoning process to a larger model. This approach improves AIME24 reasoning accuracy by 24% and 28.3% while offloading 1.35% and 5% of the generated tokens respectively. We open-source our SplitReason model, data, code and logs. 

---
# Text-to-TrajVis: Enabling Trajectory Data Visualizations from Natural Language Questions 

**Authors**: Tian Bai, Huiyan Ying, Kailong Suo, Junqiu Wei, Tao Fan, Yuanfeng Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.16358)  

**Abstract**: This paper introduces the Text-to-TrajVis task, which aims to transform natural language questions into trajectory data visualizations, facilitating the development of natural language interfaces for trajectory visualization systems. As this is a novel task, there is currently no relevant dataset available in the community. To address this gap, we first devised a new visualization language called Trajectory Visualization Language (TVL) to facilitate querying trajectory data and generating visualizations. Building on this foundation, we further proposed a dataset construction method that integrates Large Language Models (LLMs) with human efforts to create high-quality data. Specifically, we first generate TVLs using a comprehensive and systematic process, and then label each TVL with corresponding natural language questions using LLMs. This process results in the creation of the first large-scale Text-to-TrajVis dataset, named TrajVL, which contains 18,140 (question, TVL) pairs. Based on this dataset, we systematically evaluated the performance of multiple LLMs (GPT, Qwen, Llama, etc.) on this task. The experimental results demonstrate that this task is both feasible and highly challenging and merits further exploration within the research community. 

---
# Transformer-Based Extraction of Statutory Definitions from the U.S. Code 

**Authors**: Arpana Hosabettu, Harsh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.16353)  

**Abstract**: Automatic extraction of definitions from legal texts is critical for enhancing the comprehension and clarity of complex legal corpora such as the United States Code (U.S.C.). We present an advanced NLP system leveraging transformer-based architectures to automatically extract defined terms, their definitions, and their scope from the U.S.C. We address the challenges of automatically identifying legal definitions, extracting defined terms, and determining their scope within this complex corpus of over 200,000 pages of federal statutory law. Building upon previous feature-based machine learning methods, our updated model employs domain-specific transformers (Legal-BERT) fine-tuned specifically for statutory texts, significantly improving extraction accuracy. Our work implements a multi-stage pipeline that combines document structure analysis with state-of-the-art language models to process legal text from the XML version of the U.S. Code. Each paragraph is first classified using a fine-tuned legal domain BERT model to determine if it contains a definition. Our system then aggregates related paragraphs into coherent definitional units and applies a combination of attention mechanisms and rule-based patterns to extract defined terms and their jurisdictional scope. The definition extraction system is evaluated on multiple titles of the U.S. Code containing thousands of definitions, demonstrating significant improvements over previous approaches. Our best model achieves 96.8% precision and 98.9% recall (98.2% F1-score), substantially outperforming traditional machine learning classifiers. This work contributes to improving accessibility and understanding of legal information while establishing a foundation for downstream legal reasoning tasks. 

---
# Capturing Symmetry and Antisymmetry in Language Models through Symmetry-Aware Training Objectives 

**Authors**: Zhangdie Yuan, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2504.16312)  

**Abstract**: Capturing symmetric (e.g., country borders another country) and antisymmetric (e.g., parent_of) relations is crucial for a variety of applications. This paper tackles this challenge by introducing a novel Wikidata-derived natural language inference dataset designed to evaluate large language models (LLMs). Our findings reveal that LLMs perform comparably to random chance on this benchmark, highlighting a gap in relational understanding. To address this, we explore encoder retraining via contrastive learning with k-nearest neighbors. The retrained encoder matches the performance of fine-tuned classification heads while offering additional benefits, including greater efficiency in few-shot learning and improved mitigation of catastrophic forgetting. 

---
# The Paradox of Poetic Intent in Back-Translation: Evaluating the Quality of Large Language Models in Chinese Translation 

**Authors**: Li Weigang, Pedro Carvalho Brom  

**Link**: [PDF](https://arxiv.org/pdf/2504.16286)  

**Abstract**: The rapid advancement of large language models (LLMs) has reshaped the landscape of machine translation, yet challenges persist in preserving poetic intent, cultural heritage, and handling specialized terminology in Chinese-English translation. This study constructs a diverse corpus encompassing Chinese scientific terminology, historical translation paradoxes, and literary metaphors. Utilizing a back-translation and Friedman test-based evaluation system (BT-Fried), we evaluate BLEU, CHRF, TER, and semantic similarity metrics across six major LLMs (e.g., GPT-4.5, DeepSeek V3) and three traditional translation tools. Key findings include: (1) Scientific abstracts often benefit from back-translation, while traditional tools outperform LLMs in linguistically distinct texts; (2) LLMs struggle with cultural and literary retention, exemplifying the "paradox of poetic intent"; (3) Some models exhibit "verbatim back-translation", reflecting emergent memory behavior; (4) A novel BLEU variant using Jieba segmentation and n-gram weighting is proposed. The study contributes to the empirical evaluation of Chinese NLP performance and advances understanding of cultural fidelity in AI-mediated translation. 

---
# The Language of Attachment: Modeling Attachment Dynamics in Psychotherapy 

**Authors**: Frederik Bredgaard, Martin Lund Trinhammer, Elisa Bassignana  

**Link**: [PDF](https://arxiv.org/pdf/2504.16271)  

**Abstract**: The delivery of mental healthcare through psychotherapy stands to benefit immensely from developments within Natural Language Processing (NLP), in particular through the automatic identification of patient specific qualities, such as attachment style. Currently, the assessment of attachment style is performed manually using the Patient Attachment Coding System (PACS; Talia et al., 2017), which is complex, resource-consuming and requires extensive training. To enable wide and scalable adoption of attachment informed treatment and research, we propose the first exploratory analysis into automatically assessing patient attachment style from psychotherapy transcripts using NLP classification models. We further analyze the results and discuss the implications of using automated tools for this purpose -- e.g., confusing `preoccupied' patients with `avoidant' likely has a more negative impact on therapy outcomes with respect to other mislabeling. Our work opens an avenue of research enabling more personalized psychotherapy and more targeted research into the mechanisms of psychotherapy through advancements in NLP. 

---
# FinNLI: Novel Dataset for Multi-Genre Financial Natural Language Inference Benchmarking 

**Authors**: Jabez Magomere, Elena Kochkina, Samuel Mensah, Simerjot Kaur, Charese H. Smiley  

**Link**: [PDF](https://arxiv.org/pdf/2504.16188)  

**Abstract**: We introduce FinNLI, a benchmark dataset for Financial Natural Language Inference (FinNLI) across diverse financial texts like SEC Filings, Annual Reports, and Earnings Call transcripts. Our dataset framework ensures diverse premise-hypothesis pairs while minimizing spurious correlations. FinNLI comprises 21,304 pairs, including a high-quality test set of 3,304 instances annotated by finance experts. Evaluations show that domain shift significantly degrades general-domain NLI performance. The highest Macro F1 scores for pre-trained (PLMs) and large language models (LLMs) baselines are 74.57% and 78.62%, respectively, highlighting the dataset's difficulty. Surprisingly, instruction-tuned financial LLMs perform poorly, suggesting limited generalizability. FinNLI exposes weaknesses in current LLMs for financial reasoning, indicating room for improvement. 

---
# AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset 

**Authors**: Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2504.16891)  

**Abstract**: This paper presents our winning submission to the AI Mathematical Olympiad - Progress Prize 2 (AIMO-2) competition. Our recipe for building state-of-the-art mathematical reasoning models relies on three key pillars. First, we create a large-scale dataset comprising 540K unique high-quality math problems, including olympiad-level problems, and their 3.2M long-reasoning solutions. Second, we develop a novel method to integrate code execution with long reasoning models through iterative training, generation, and quality filtering, resulting in 1.7M high-quality Tool-Integrated Reasoning solutions. Third, we create a pipeline to train models to select the most promising solution from many candidates. We show that such generative solution selection (GenSelect) can significantly improve upon majority voting baseline. Combining these ideas, we train a series of models that achieve state-of-the-art results on mathematical reasoning benchmarks. To facilitate further research, we release our code, models, and the complete OpenMathReasoning dataset under a commercially permissive license. 

---
# Process Reward Models That Think 

**Authors**: Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, Honglak Lee, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16828)  

**Abstract**: Step-by-step verifiers -- also known as process reward models (PRMs) -- are a key ingredient for test-time scaling. PRMs require step-level supervision, making them expensive to train. This work aims to build data-efficient PRMs as verbalized step-wise reward models that verify every step in the solution by generating a verification chain-of-thought (CoT). We propose ThinkPRM, a long CoT verifier fine-tuned on orders of magnitude fewer process labels than those required by discriminative PRMs. Our approach capitalizes on the inherent reasoning abilities of long CoT models, and outperforms LLM-as-a-Judge and discriminative verifiers -- using only 1% of the process labels in PRM800K -- across several challenging benchmarks. Specifically, ThinkPRM beats the baselines on ProcessBench, MATH-500, and AIME '24 under best-of-N selection and reward-guided search. In an out-of-domain evaluation on a subset of GPQA-Diamond and LiveCodeBench, our PRM surpasses discriminative verifiers trained on the full PRM800K by 8% and 4.5%, respectively. Lastly, under the same token budget, ThinkPRM scales up verification compute more effectively compared to LLM-as-a-Judge, outperforming it by 7.2% on a subset of ProcessBench. Our work highlights the value of generative, long CoT PRMs that can scale test-time compute for verification while requiring minimal supervision for training. Our code, data, and models will be released at this https URL. 

---
# IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery 

**Authors**: Aniketh Garikaparthi, Manasi Patwardhan, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2504.16728)  

**Abstract**: The rapid advancement in capabilities of large language models (LLMs) raises a pivotal question: How can LLMs accelerate scientific discovery? This work tackles the crucial first stage of research, generating novel hypotheses. While recent work on automated hypothesis generation focuses on multi-agent frameworks and extending test-time compute, none of the approaches effectively incorporate transparency and steerability through a synergistic Human-in-the-loop (HITL) approach. To address this gap, we introduce IRIS: Interactive Research Ideation System, an open-source platform designed for researchers to leverage LLM-assisted scientific ideation. IRIS incorporates innovative features to enhance ideation, including adaptive test-time compute expansion via Monte Carlo Tree Search (MCTS), fine-grained feedback mechanism, and query-based literature synthesis. Designed to empower researchers with greater control and insight throughout the ideation process. We additionally conduct a user study with researchers across diverse disciplines, validating the effectiveness of our system in enhancing ideation. We open-source our code at this https URL 

---
# ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models using Pareto High-quality Data 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.16628)  

**Abstract**: Aligning large language models with multiple human expectations and values is crucial for ensuring that they adequately serve a variety of user needs. To this end, offline multiobjective alignment algorithms such as the Rewards-in-Context algorithm have shown strong performance and efficiency. However, inappropriate preference representations and training with imbalanced reward scores limit the performance of such algorithms. In this work, we introduce ParetoHqD that addresses the above issues by representing human preferences as preference directions in the objective space and regarding data near the Pareto front as ''high-quality'' data. For each preference, ParetoHqD follows a two-stage supervised fine-tuning process, where each stage uses an individual Pareto high-quality training set that best matches its preference direction. The experimental results have demonstrated the superiority of ParetoHqD over five baselines on two multiobjective alignment tasks. 

---
# MAGIC: Near-Optimal Data Attribution for Deep Learning 

**Authors**: Andrew Ilyas, Logan Engstrom  

**Link**: [PDF](https://arxiv.org/pdf/2504.16430)  

**Abstract**: The goal of predictive data attribution is to estimate how adding or removing a given set of training datapoints will affect model predictions. In convex settings, this goal is straightforward (i.e., via the infinitesimal jackknife). In large-scale (non-convex) settings, however, existing methods are far less successful -- current methods' estimates often only weakly correlate with ground truth. In this work, we present a new data attribution method (MAGIC) that combines classical methods and recent advances in metadifferentiation to (nearly) optimally estimate the effect of adding or removing training data on model predictions. 

---
# SignX: The Foundation Model for Sign Recognition 

**Authors**: Sen Fang, Chunyu Sui, Hongwei Yi, Carol Neidle, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2504.16315)  

**Abstract**: The complexity of sign language data processing brings many challenges. The current approach to recognition of ASL signs aims to translate RGB sign language videos through pose information into English-based ID glosses, which serve to uniquely identify ASL signs. Note that there is no shared convention for assigning such glosses to ASL signs, so it is essential that the same glossing conventions are used for all of the data in the datasets that are employed. This paper proposes SignX, a foundation model framework for sign recognition. It is a concise yet powerful framework applicable to multiple human activity recognition scenarios. First, we developed a Pose2Gloss component based on an inverse diffusion model, which contains a multi-track pose fusion layer that unifies five of the most powerful pose information sources--SMPLer-X, DWPose, Mediapipe, PrimeDepth, and Sapiens Segmentation--into a single latent pose representation. Second, we trained a Video2Pose module based on ViT that can directly convert raw video into signer pose representation. Through this 2-stage training framework, we enable sign language recognition models to be compatible with existing pose formats, laying the foundation for the common pose estimation necessary for sign recognition. Experimental results show that SignX can recognize signs from sign language video, producing predicted gloss representations with greater accuracy than has been reported in prior work. 

---
# CLIRudit: Cross-Lingual Information Retrieval of Scientific Documents 

**Authors**: Francisco Valentini, Diego Kozlowski, Vincent Larivière  

**Link**: [PDF](https://arxiv.org/pdf/2504.16264)  

**Abstract**: Cross-lingual information retrieval (CLIR) consists in finding relevant documents in a language that differs from the language of the queries. This paper presents CLIRudit, a new dataset created to evaluate cross-lingual academic search, focusing on English queries and French documents. The dataset is built using bilingual article metadata from Érudit, a Canadian publishing platform, and is designed to represent scenarios in which researchers search for scholarly content in languages other than English. We perform a comprehensive benchmarking of different zero-shot first-stage retrieval methods on the dataset, including dense and sparse retrievers, query and document machine translation, and state-of-the-art multilingual retrievers. Our results show that large dense retrievers, not necessarily trained for the cross-lingual retrieval task, can achieve zero-shot performance comparable to using ground truth human translations, without the need for machine translation. Sparse retrievers, such as BM25 or SPLADE, combined with document translation, show competitive results, providing an efficient alternative to large dense models. This research advances the understanding of cross-lingual academic information retrieval and provides a framework that others can use to build comparable datasets across different languages and disciplines. By making the dataset and code publicly available, we aim to facilitate further research that will help make scientific knowledge more accessible across language barriers. 

---
# Using Phonemes in cascaded S2S translation pipeline 

**Authors**: Rene Pilz, Johannes Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2504.16234)  

**Abstract**: This paper explores the idea of using phonemes as a textual representation within a conventional multilingual simultaneous speech-to-speech translation pipeline, as opposed to the traditional reliance on text-based language representations. To investigate this, we trained an open-source sequence-to-sequence model on the WMT17 dataset in two formats: one using standard textual representation and the other employing phonemic representation. The performance of both approaches was assessed using the BLEU metric. Our findings shows that the phonemic approach provides comparable quality but offers several advantages, including lower resource requirements or better suitability for low-resource languages. 

---
# Reflexive Prompt Engineering: A Framework for Responsible Prompt Engineering and Interaction Design 

**Authors**: Christian Djeffal  

**Link**: [PDF](https://arxiv.org/pdf/2504.16204)  

**Abstract**: Responsible prompt engineering has emerged as a critical framework for ensuring that generative artificial intelligence (AI) systems serve society's needs while minimizing potential harms. As generative AI applications become increasingly powerful and ubiquitous, the way we instruct and interact with them through prompts has profound implications for fairness, accountability, and transparency. This article examines how strategic prompt engineering can embed ethical and legal considerations and societal values directly into AI interactions, moving beyond mere technical optimization for functionality. This article proposes a comprehensive framework for responsible prompt engineering that encompasses five interconnected components: prompt design, system selection, system configuration, performance evaluation, and prompt management. Drawing from empirical evidence, the paper demonstrates how each component can be leveraged to promote improved societal outcomes while mitigating potential risks. The analysis reveals that effective prompt engineering requires a delicate balance between technical precision and ethical consciousness, combining the systematic rigor and focus on functionality with the nuanced understanding of social impact. Through examination of real-world and emerging practices, the article illustrates how responsible prompt engineering serves as a crucial bridge between AI development and deployment, enabling organizations to fine-tune AI outputs without modifying underlying model architectures. This approach aligns with broader "Responsibility by Design" principles, embedding ethical considerations directly into the implementation process rather than treating them as post-hoc additions. The article concludes by identifying key research directions and practical guidelines for advancing the field of responsible prompt engineering. 

---
# Multimodal Large Language Models for Enhanced Traffic Safety: A Comprehensive Review and Future Trends 

**Authors**: Mohammad Abu Tami, Mohammed Elhenawy, Huthaifa I. Ashqar  

**Link**: [PDF](https://arxiv.org/pdf/2504.16134)  

**Abstract**: Traffic safety remains a critical global challenge, with traditional Advanced Driver-Assistance Systems (ADAS) often struggling in dynamic real-world scenarios due to fragmented sensor processing and susceptibility to adversarial conditions. This paper reviews the transformative potential of Multimodal Large Language Models (MLLMs) in addressing these limitations by integrating cross-modal data such as visual, spatial, and environmental inputs to enable holistic scene understanding. Through a comprehensive analysis of MLLM-based approaches, we highlight their capabilities in enhancing perception, decision-making, and adversarial robustness, while also examining the role of key datasets (e.g., KITTI, DRAMA, ML4RoadSafety) in advancing research. Furthermore, we outline future directions, including real-time edge deployment, causality-driven reasoning, and human-AI collaboration. By positioning MLLMs as a cornerstone for next-generation traffic safety systems, this review underscores their potential to revolutionize the field, offering scalable, context-aware solutions that proactively mitigate risks and improve overall road safety. 

---
# LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval 

**Authors**: Muhammad Rafsan Kabir, Rafeed Mohammad Sultan, Fuad Rahman, Mohammad Ruhul Amin, Sifat Momen, Nabeel Mohammed, Shafin Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2504.16121)  

**Abstract**: Natural Language Processing (NLP) and computational linguistic techniques are increasingly being applied across various domains, yet their use in legal and regulatory tasks remains limited. To address this gap, we develop an efficient bilingual question-answering framework for regulatory documents, specifically the Bangladesh Police Gazettes, which contain both English and Bangla text. Our approach employs modern Retrieval Augmented Generation (RAG) pipelines to enhance information retrieval and response generation. In addition to conventional RAG pipelines, we propose an advanced RAG-based approach that improves retrieval performance, leading to more precise answers. This system enables efficient searching for specific government legal notices, making legal information more accessible. We evaluate both our proposed and conventional RAG systems on a diverse test set on Bangladesh Police Gazettes, demonstrating that our approach consistently outperforms existing methods across all evaluation metrics. 

---
# HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing 

**Authors**: Myunghyun Rhee, Joonseop Sim, Taeyoung Ahn, Seungyong Lee, Daegun Yoon, Euiseok Kim, Kyoung Park, Youngpyo Joo, Hosik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.16112)  

**Abstract**: The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV caches. We propose a High-bandwidth Processing Unit (HPU), a memoryintensive co-processor that enhances GPU resource utilization during large-batched LLM inference. By offloading memory-bound operations, the HPU allows the GPU to focus on compute-intensive tasks, increasing overall efficiency. Also, the HPU, as an add-on card, scales out to accommodate surging memory demands driven by large batch sizes and extended sequence lengths. In this paper, we show the HPU prototype implemented with PCIe-based FPGA cards mounted on a GPU system. Our novel GPU-HPU heterogeneous system demonstrates up to 4.1x performance gains and 4.6x energy efficiency improvements over a GPUonly system, providing scalability without increasing the number of GPUs. 

---
# Cooperative Speech, Semantic Competence, and AI 

**Authors**: Mahrad Almotahari  

**Link**: [PDF](https://arxiv.org/pdf/2504.16092)  

**Abstract**: Cooperative speech is purposive. From the speaker's perspective, one crucial purpose is the transmission of knowledge. Cooperative speakers care about getting things right for their conversational partners. This attitude is a kind of respect. Cooperative speech is an ideal form of communication because participants have respect for each other. And having respect within a cooperative enterprise is sufficient for a particular kind of moral standing: we ought to respect those who have respect for us. Respect demands reciprocity. I maintain that large language models aren't owed the kind of respect that partly constitutes a cooperative conversation. This implies that they aren't cooperative interlocutors, otherwise we would be obliged to reciprocate the attitude. Leveraging this conclusion, I argue that present-day LLMs are incapable of assertion and that this raises an overlooked doubt about their semantic competence. One upshot of this argument is that knowledge of meaning isn't just a subject for the cognitive psychologist. It's also a subject for the moral psychologist. 

---
