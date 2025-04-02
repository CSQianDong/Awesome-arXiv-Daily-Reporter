# CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models 

**Authors**: Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00882)  

**Abstract**: Dialect translation plays a key role in enabling seamless interaction across heterogeneous database systems. However, translating SQL queries between different dialects (e.g., from PostgreSQL to MySQL) remains a challenging task due to syntactic discrepancies and subtle semantic variations. Existing approaches including manual rewriting, rule-based systems, and large language model (LLM)-based techniques often involve high maintenance effort (e.g., crafting custom translation rules) or produce unreliable results (e.g., LLM generates non-existent functions), especially when handling complex queries. In this demonstration, we present CrackSQL, the first hybrid SQL dialect translation system that combines rule and LLM-based methods to overcome these limitations. CrackSQL leverages the adaptability of LLMs to minimize manual intervention, while enhancing translation accuracy by segmenting lengthy complex SQL via functionality-based query processing. To further improve robustness, it incorporates a novel cross-dialect syntax embedding model for precise syntax alignment, as well as an adaptive local-to-global translation strategy that effectively resolves interdependent query operations. CrackSQL supports three translation modes and offers multiple deployment and access options including a web console interface, a PyPI package, and a command-line prompt, facilitating adoption across a variety of real-world use cases 

---
# InformGen: An AI Copilot for Accurate and Compliant Clinical Research Consent Document Generation 

**Authors**: Zifeng Wang, Junyi Gao, Benjamin Danek, Brandon Theodorou, Ruba Shaik, Shivashankar Thati, Seunghyun Won, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.00934)  

**Abstract**: Leveraging large language models (LLMs) to generate high-stakes documents, such as informed consent forms (ICFs), remains a significant challenge due to the extreme need for regulatory compliance and factual accuracy. Here, we present InformGen, an LLM-driven copilot for accurate and compliant ICF drafting by optimized knowledge document parsing and content generation, with humans in the loop. We further construct a benchmark dataset comprising protocols and ICFs from 900 clinical trials. Experimental results demonstrate that InformGen achieves near 100% compliance with 18 core regulatory rules derived from FDA guidelines, outperforming a vanilla GPT-4o model by up to 30%. Additionally, a user study with five annotators shows that InformGen, when integrated with manual intervention, attains over 90% factual accuracy, significantly surpassing the vanilla GPT-4o model's 57%-82%. Crucially, InformGen ensures traceability by providing inline citations to source protocols, enabling easy verification and maintaining the highest standards of factual integrity. 

---
# When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning 

**Authors**: Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2504.01005)  

**Abstract**: Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at this https URL. 

---
# SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching 

**Authors**: Yuxuan Zhu, Ali Falahati, David H. Yang, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.00970)  

**Abstract**: Large language models face significant computational and memory challenges when processing long contexts. During inference, efficient management of the key-value (KV) cache, which stores intermediate activations for autoregressive generation, is critical to reducing memory overhead and improving computational efficiency. Traditional token-level efficient KV caching methods overlook semantic information, treating tokens independently without considering their semantic relationships. Meanwhile, existing semantic-preserving KV cache management approaches often suffer from substantial memory usage and high time-to-first-token. To address these limitations, we propose SentenceKV, a novel sentence-level semantic KV caching approach designed to enhance inference efficiency while preserving semantic coherence. During prefilling, SentenceKV groups tokens based on sentence-level semantic similarity, compressing sentence representations into concise semantic vectors stored directly on the GPU, while individual KV pairs are offloaded to CPU. During decoding, SentenceKV generates tokens by selectively retrieving semantically relevant sentence-level KV entries, leveraging the semantic similarity between the prefilling-stage semantic vectors and decoding-stage queries. This ensures efficient and contextually accurate predictions, minimizing the loading of redundant or irrelevant data into GPU memory and significantly reducing memory overhead while maintaining stable inference latency, even for extremely long contexts. Extensive evaluations on benchmarks including PG-19, LongBench, and Needle-In-A-Haystack demonstrate that SentenceKV significantly outperforms state-of-the-art methods in both efficiency and memory usage, without compromising model accuracy. 

---
# Token embeddings violate the manifold hypothesis 

**Authors**: Michael Robinson, Sourya Dey, Tony Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01002)  

**Abstract**: To fully understand the behavior of a large language model (LLM) requires our understanding of its input space. If this input space differs from our assumption, our understanding of and conclusions about the LLM is likely flawed, regardless of its architecture. Here, we elucidate the structure of the token embeddings, the input domain for LLMs, both empirically and theoretically. We present a generalized and statistically testable model where the neighborhood of each token splits into well-defined signal and noise dimensions.
This model is based on a generalization of a manifold called a fiber bundle, so we denote our hypothesis test as the ``fiber bundle null.'' Failing to reject the null is uninformative, but rejecting it at a specific token indicates that token has a statistically significant local structure, and so is of interest to us. By running our test over several open-source LLMs, each with unique token embeddings, we find that the null is frequently rejected, and so the token subspace is provably not a fiber bundle and hence also not a manifold. As a consequence of our findings, when an LLM is presented with two semantically equivalent prompts, and if one prompt contains a token implicated by our test, that prompt will likely exhibit more output variability proportional to the local signal dimension of the token. 

---
# Multi-Token Attention 

**Authors**: Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2504.00927)  

**Abstract**: Soft attention is a critical mechanism powering LLMs to locate relevant parts within a given context. However, individual attention weights are determined by the similarity of only a single query and key token vector. This "single token attention" bottlenecks the amount of information used in distinguishing a relevant part from the rest of the context. To address this issue, we propose a new attention method, Multi-Token Attention (MTA), which allows LLMs to condition their attention weights on multiple query and key vectors simultaneously. This is achieved by applying convolution operations over queries, keys and heads, allowing nearby queries and keys to affect each other's attention weights for more precise attention. As a result, our method can locate relevant context using richer, more nuanced information that can exceed a single vector's capacity. Through extensive evaluations, we demonstrate that MTA achieves enhanced performance on a range of popular benchmarks. Notably, it outperforms Transformer baseline models on standard language modeling tasks, and on tasks that require searching for information within long contexts, where our method's ability to leverage richer information proves particularly beneficial. 

---
# GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning 

**Authors**: Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00891)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have shown that it is promising to utilize Process Reward Models (PRMs) as verifiers to enhance the performance of LLMs. However, current PRMs face three key challenges: (1) limited process supervision and generalization capabilities, (2) dependence on scalar value prediction without leveraging the generative abilities of LLMs, and (3) inability to scale the test-time compute of PRMs. In this work, we introduce GenPRM, a generative process reward model that performs explicit Chain-of-Thought (CoT) reasoning with code verification before providing judgment for each reasoning step. To obtain high-quality process supervision labels and rationale data, we propose Relative Progress Estimation (RPE) and a rationale synthesis framework that incorporates code verification. Experimental results on ProcessBench and several mathematical reasoning tasks show that GenPRM significantly outperforms prior PRMs with only 23K training data from MATH dataset. Through test-time scaling, a 1.5B GenPRM outperforms GPT-4o, and a 7B GenPRM surpasses Qwen2.5-Math-PRM-72B on ProcessBench. Additionally, GenPRM demonstrates strong abilities to serve as a critic model for policy model refinement. This work establishes a new paradigm for process supervision that bridges the gap between PRMs and critic models in LLMs. Our code, model, and data will be available in this https URL. 

---
# m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models 

**Authors**: Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00869)  

**Abstract**: Test-time scaling has emerged as a powerful technique for enhancing the reasoning capabilities of large language models. However, its effectiveness in medical reasoning remains uncertain, as the medical domain fundamentally differs from mathematical tasks in terms of knowledge representation and decision-making processes. In this paper, we provide the first comprehensive investigation of test-time scaling for medical reasoning and present m1, a simple yet effective approach that increases a model's medical reasoning capability at inference. Our evaluation across diverse medical tasks demonstrates that test-time scaling consistently enhances medical reasoning, enabling lightweight fine-tuned models under 10B parameters to establish new state-of-the-art performance, while our 32B model rivals previous 70B-scale medical LLMs. However, we identify an optimal reasoning token budget of approximately 4K, beyond which performance may degrade due to overthinking. Budget forcing, which extends test-time computation through iterative prompts, helps models double-check answers but does not necessarily improve the overall medical QA performance and, in some cases, even introduces errors into previously correct responses. Our case-by-case analysis identifies insufficient medical knowledge as a key bottleneck that prevents further performance gains through test-time scaling. We find that increasing data scale, improving data quality, and expanding model capacity consistently enhance medical knowledge grounding, enabling continued performance improvements, particularly on challenging medical benchmarks where smaller models reach saturation. These findings underscore fundamental differences between medical and mathematical reasoning in LLMs, highlighting that enriched medical knowledge, other than increased reasoning depth alone, is essential for realizing the benefits of test-time scaling. 

---
# How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study 

**Authors**: Yunjie Ji, Sitong Zhao, Xiaoyu Tian, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00829)  

**Abstract**: Enhancing the reasoning capabilities of Large Language Models (LLMs) with efficiency and scalability remains a fundamental challenge in artificial intelligence research. This paper presents a rigorous experimental investigation into how difficulty-aware staged reinforcement learning (RL) strategies can substantially improve LLM reasoning performance. Through systematic analysis, we demonstrate that strategically selecting training data according to well-defined difficulty levels markedly enhances RL optimization. Moreover, we introduce a staged training methodology, progressively exposing models to increasingly challenging tasks, further amplifying reasoning capabilities. Our findings reveal significant cross-domain benefits when simultaneously training models on mathematical reasoning and code generation tasks. Notably, our proposed approach enables a 1.5B parameter model to achieve an accuracy of 42.3\% on the AIME-2024 benchmark, 89.5\% on the MATH-500 benchmark. These results underscore the efficacy of our method in advancing the reasoning proficiency of LLMs. We will open-source our datasets on GitHub and Hugging Face. 

---
# Z1: Efficient Test-time Scaling with Code 

**Authors**: Zhaojian Yu, Yinghao Wu, Yilun Zhao, Arman Cohan, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00810)  

**Abstract**: Large Language Models (LLMs) can achieve enhanced complex problem-solving through test-time computing scaling, yet this often entails longer contexts and numerous reasoning token costs. In this paper, we propose an efficient test-time scaling method that trains LLMs on code-related reasoning trajectories, facilitating their reduction of excess thinking tokens while maintaining performance. First, we create Z1-Code-Reasoning-107K, a curated dataset of simple and complex coding problems paired with their short and long solution trajectories. Second, we present a novel Shifted Thinking Window to mitigate overthinking overhead by removing context-delimiting tags (e.g., <think>. . . </think>) and capping reasoning tokens. Trained with long and short trajectory data and equipped with Shifted Thinking Window, our model, Z1-7B, demonstrates the ability to adjust its reasoning level as the complexity of problems and exhibits efficient test-time scaling across different reasoning tasks that matches R1-Distill-Qwen-7B performance with about 30% of its average thinking tokens. Notably, fine-tuned with only code trajectories, Z1-7B demonstrates generalization to broader reasoning tasks (47.5% on GPQA Diamond). Our analysis of efficient reasoning elicitation also provides valuable insights for future research. 

---
# RECKON: Large-scale Reference-based Efficient Knowledge Evaluation for Large Language Model 

**Authors**: Lin Zhang, Zhouhong Gu, Xiaoran Shi, Hongwei Feng, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00756)  

**Abstract**: As large language models (LLMs) advance, efficient knowledge evaluation becomes crucial to verifying their capabilities. Traditional methods, relying on benchmarks, face limitations such as high resource costs and information loss. We propose the Large-scale Reference-based Efficient Knowledge Evaluation for Large Language Model (RECKON), which directly uses reference data to evaluate models. RECKON organizes unstructured data into manageable units and generates targeted questions for each cluster, improving evaluation accuracy and efficiency. Experimental results show that RECKON reduces resource consumption by 56.5% compared to traditional methods while achieving over 97% accuracy across various domains, including world knowledge, code, legal, and biomedical datasets. Code is available at this https URL 

---
# Aplicação de Large Language Models na Análise e Síntese de Documentos Jurídicos: Uma Revisão de Literatura 

**Authors**: Matheus Belarmino, Rackel Coelho, Roberto Lotudo, Jayr Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2504.00725)  

**Abstract**: Large Language Models (LLMs) have been increasingly used to optimize the analysis and synthesis of legal documents, enabling the automation of tasks such as summarization, classification, and retrieval of legal information. This study aims to conduct a systematic literature review to identify the state of the art in prompt engineering applied to LLMs in the legal context. The results indicate that models such as GPT-4, BERT, Llama 2, and Legal-Pegasus are widely employed in the legal field, and techniques such as Few-shot Learning, Zero-shot Learning, and Chain-of-Thought prompting have proven effective in improving the interpretation of legal texts. However, challenges such as biases in models and hallucinations still hinder their large-scale implementation. It is concluded that, despite the great potential of LLMs for the legal field, there is a need to improve prompt engineering strategies to ensure greater accuracy and reliability in the generated results. 

---
# Self-Routing RAG: Binding Selective Retrieval with Knowledge Verbalization 

**Authors**: Di Wu, Jia-Chen Gu, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2504.01018)  

**Abstract**: Selective retrieval improves retrieval-augmented generation (RAG) by reducing distractions from low-quality retrievals and improving efficiency. However, existing approaches under-utilize the inherent knowledge of large language models (LLMs), leading to suboptimal retrieval decisions and degraded generation performance. To bridge this gap, we propose Self-Routing RAG (SR-RAG), a novel framework that binds selective retrieval with knowledge verbalization. SR-RAG enables an LLM to dynamically decide between external retrieval and verbalizing its own parametric knowledge. To this end, we design a multi-task objective that jointly optimizes an LLM on knowledge source selection, knowledge verbalization, and response generation. We further introduce dynamic knowledge source inference via nearest neighbor search to improve the accuracy of knowledge source decision under domain shifts. Fine-tuning three LLMs with SR-RAG significantly improves both their response accuracy and inference latency. Compared to the strongest selective retrieval baseline, SR-RAG reduces retrievals by 29% while improving the performance by 5.1%. 

---
# MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs 

**Authors**: Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, Yihan Cao, Hui Ren, Xiang Li, Xiaoxiao Li, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00993)  

**Abstract**: Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available. 

---
# ScholarCopilot: Training Large Language Models for Academic Writing with Accurate Citations 

**Authors**: Yubo Wang, Xueguang Ma, Ping Nie, Huaye Zeng, Zhiheng Lyu, Yuxuan Zhang, Benjamin Schneider, Yi Lu, Xiang Yue, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00824)  

**Abstract**: Academic writing requires both coherent text generation and precise citation of relevant literature. Although recent Retrieval-Augmented Generation (RAG) systems have significantly improved factual accuracy in general-purpose text generation, their capacity to adequately support professional academic writing remains limited. In this work, we introduce ScholarCopilot, a unified framework designed to enhance existing large language models for generating professional academic articles with accurate and contextually relevant citations. ScholarCopilot dynamically determines when to retrieve scholarly references by generating a retrieval token [RET], and then utilizes its representation to look up relevant citations from a database. The retrieved references are fed into the model to augment the generation process. We jointly optimize both the generation and citation tasks within a single framework to increase efficiency. Trained on 500K papers from arXiv, our model achieves a top-1 retrieval accuracy of 40.1% on our evaluation dataset, outperforming baselines such as E5-Mistral-7B-Instruct (15.0%) and BM25 (9.8%). On a dataset of 1,000 academic writing samples, ScholarCopilot scores 16.2/25 in generation quality (measured across relevance, coherence, academic rigor, completeness, and innovation), surpassing models with 10x more parameters such as Qwen-2.5-72B-Instruct (15.8/25). Human studies also confirm ScholarCopilot's superior performance in citation recall, writing efficiency, and overall user experience, confirming the effectiveness of our approach. 

---
# IHC-LLMiner: Automated extraction of tumour immunohistochemical profiles from PubMed abstracts using large language models 

**Authors**: Yunsoo Kim, Michal W. S. Ong, Daniel W. Rogalsky, Manuel Rodriguez-Justo, Honghan Wu, Adam P. Levine  

**Link**: [PDF](https://arxiv.org/pdf/2504.00748)  

**Abstract**: Immunohistochemistry (IHC) is essential in diagnostic pathology and biomedical research, offering critical insights into protein expression and tumour biology. This study presents an automated pipeline, IHC-LLMiner, for extracting IHC-tumour profiles from PubMed abstracts, leveraging advanced biomedical text mining. There are two subtasks: abstract classification (include/exclude as relevant) and IHC-tumour profile extraction on relevant included abstracts. The best-performing model, "Gemma-2 finetuned", achieved 91.5% accuracy and an F1 score of 91.4, outperforming GPT4-O by 9.5% accuracy with 5.9 times faster inference time. From an initial dataset of 107,759 abstracts identified for 50 immunohistochemical markers, the classification task identified 30,481 relevant abstracts (Include) using the Gemma-2 finetuned model. For IHC-tumour profile extraction, the Gemma-2 finetuned model achieved the best performance with 63.3% Correct outputs. Extracted IHC-tumour profiles (tumour types and markers) were normalised to Unified Medical Language System (UMLS) concepts to ensure consistency and facilitate IHC-tumour profile landscape analysis. The extracted IHC-tumour profiles demonstrated excellent concordance with available online summary data and provided considerable added value in terms of both missing IHC-tumour profiles and quantitative assessments. Our proposed LLM based pipeline provides a practical solution for large-scale IHC-tumour profile data mining, enhancing the accessibility and utility of such data for research and clinical applications as well as enabling the generation of quantitative and structured data to support cancer-specific knowledge base development. Models and training datasets are available at this https URL. 

---
# DynMoLE: Boosting Mixture of LoRA Experts Fine-Tuning with a Hybrid Routing Mechanism 

**Authors**: Dengchun Li, Naizheng Wang, Zihao Zhang, Haoyang Yin, Lei Duan, Meng Xiao, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00661)  

**Abstract**: Instruction-based fine-tuning of large language models (LLMs) has achieved remarkable success in various natural language processing (NLP) tasks. Parameter-efficient fine-tuning (PEFT) methods, such as Mixture of LoRA Experts (MoLE), combine the efficiency of Low-Rank Adaptation (LoRA) with the versatility of Mixture of Experts (MoE) models, demonstrating significant potential for handling multiple downstream tasks. However, the existing routing mechanisms for MoLE often involve a trade-off between computational efficiency and predictive accuracy, and they fail to fully address the diverse expert selection demands across different transformer layers. In this work, we propose DynMoLE, a hybrid routing strategy that dynamically adjusts expert selection based on the Tsallis entropy of the router's probability distribution. This approach mitigates router uncertainty, enhances stability, and promotes more equitable expert participation, leading to faster convergence and improved model performance. Additionally, we introduce an auxiliary loss based on Tsallis entropy to further guide the model toward convergence with reduced uncertainty, thereby improving training stability and performance. Our extensive experiments on commonsense reasoning benchmarks demonstrate that DynMoLE achieves substantial performance improvements, outperforming LoRA by 9.6% and surpassing the state-of-the-art MoLE method, MoLA, by 2.3%. We also conduct a comprehensive ablation study to evaluate the contributions of DynMoLE's key components. 

---
# Command A: An Enterprise-Ready Large Language Model 

**Authors**: Team Cohere, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, Neeral Beladia, Walter Beller-Morales, Alexandre Bérard, Andrew Berneshawi, Anna Bialas, Phil Blunsom, Matt Bobkin, Adi Bongale, Sam Braun, Maxime Brunet, Samuel Cahyawijaya, David Cairuz, Jon Ander Campos, Cassie Cao, Kris Cao, Roman Castagné, Julián Cendrero, Leila Chan Currie, Yash Chandak, Diane Chang, Giannis Chatziveroglou, Hongyu Chen, Claire Cheng, Alexis Chevalier, Justin T. Chiu, Eugene Cho, Eugene Choi, Eujeong Choi, Tim Chung, Volkan Cirik, Ana Cismaru, Pierre Clavier, Henry Conklin, Lucas Crawhall-Stein, Devon Crouse, Andres Felipe Cruz-Salinas, Ben Cyrus, Daniel D'souza, Hugo Dalla-Torre, John Dang, William Darling, Omar Darwiche Domingues, Saurabh Dash, Antoine Debugne, Théo Dehaze, Shaan Desai, Joan Devassy, Rishit Dholakia, Kyle Duffy, Ali Edalati, Ace Eldeib, Abdullah Elkady, Sarah Elsharkawy, Irem Ergün, Beyza Ermis, Marzieh Fadaee, Boyu Fan, Lucas Fayoux, Yannis Flet-Berliac, Nick Frosst, Matthias Gallé, Wojciech Galuba, Utsav Garg, Matthieu Geist, Mohammad Gheshlaghi Azar, Seraphina Goldfarb-Tarrant, Tomas Goldsack, Aidan Gomez, Victor Machado Gonzaga, Nithya Govindarajan, Manoj Govindassamy, Nathan Grinsztajn, Nikolas Gritsch, Patrick Gu, Shangmin Guo, Kilian Haefeli, Rod Hajjar, Tim Hawes, Jingyi He, Sebastian Hofstätter, Sungjin Hong, Sara Hooker, Tom Hosking  

**Link**: [PDF](https://arxiv.org/pdf/2504.00698)  

**Abstract**: In this report we describe the development of Command A, a powerful large language model purpose-built to excel at real-world enterprise use cases. Command A is an agent-optimised and multilingual-capable model, with support for 23 languages of global business, and a novel hybrid architecture balancing efficiency with top of the range performance. It offers best-in-class Retrieval Augmented Generation (RAG) capabilities with grounding and tool use to automate sophisticated business processes. These abilities are achieved through a decentralised training approach, including self-refinement algorithms and model merging techniques. We also include results for Command R7B which shares capability and architectural similarities to Command A. Weights for both models have been released for research purposes. This technical report details our original training pipeline and presents an extensive evaluation of our models across a suite of enterprise-relevant tasks and public benchmarks, demonstrating excellent performance and efficiency. 

---
# On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation 

**Authors**: Jirui Qi, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00597)  

**Abstract**: Retrieval-augmented generation (RAG) with large language models (LLMs) has demonstrated strong performance in multilingual question-answering (QA) tasks by leveraging relevant passages retrieved from corpora. In multilingual RAG (mRAG), the retrieved passages can be written in languages other than that of the query entered by the user, making it challenging for LLMs to effectively utilize the provided information. Recent research suggests that retrieving passages from multilingual corpora can improve RAG performance, particularly for low-resource languages. However, the extent to which LLMs can leverage different kinds of multilingual contexts to generate accurate answers, *independently from retrieval quality*, remains understudied. In this paper, we conduct an extensive assessment of LLMs' ability to (i) make consistent use of a relevant passage regardless of its language, (ii) respond in the expected language, and (iii) focus on the relevant passage even when multiple `distracting' passages in different languages are provided in the context. Our experiments with four LLMs across three QA datasets covering a total of 48 languages reveal a surprising ability of LLMs to extract the relevant information from out-language passages, but a much weaker ability to formulate a full answer in the correct language. Our analysis, based on both accuracy and feature attribution techniques, further shows that distracting passages negatively impact answer quality regardless of their language. However, distractors in the query language exert a slightly stronger influence. Taken together, our findings deepen the understanding of how LLMs utilize context in mRAG systems, providing directions for future improvements. 

---
# GLiNER-biomed: A Suite of Efficient Models for Open Biomedical Named Entity Recognition 

**Authors**: Anthony Yazdani, Ihor Stepanov, Douglas Teodoro  

**Link**: [PDF](https://arxiv.org/pdf/2504.00676)  

**Abstract**: Biomedical named entity recognition (NER) presents unique challenges due to specialized vocabularies, the sheer volume of entities, and the continuous emergence of novel entities. Traditional NER models, constrained by fixed taxonomies and human annotations, struggle to generalize beyond predefined entity types or efficiently adapt to emerging concepts. To address these issues, we introduce GLiNER-biomed, a domain-adapted suite of Generalist and Lightweight Model for NER (GLiNER) models specifically tailored for biomedical NER. In contrast to conventional approaches, GLiNER uses natural language descriptions to infer arbitrary entity types, enabling zero-shot recognition. Our approach first distills the annotation capabilities of large language models (LLMs) into a smaller, more efficient model, enabling the generation of high-coverage synthetic biomedical NER data. We subsequently train two GLiNER architectures, uni- and bi-encoder, at multiple scales to balance computational efficiency and recognition performance. Evaluations on several biomedical datasets demonstrate that GLiNER-biomed outperforms state-of-the-art GLiNER models in both zero- and few-shot scenarios, achieving 5.96% improvement in F1-score over the strongest baseline. Ablation studies highlight the effectiveness of our synthetic data generation strategy and emphasize the complementary benefits of synthetic biomedical pre-training combined with fine-tuning on high-quality general-domain annotations. All datasets, models, and training pipelines are publicly available at this https URL. 

---
# LLMs4SchemaDiscovery: A Human-in-the-Loop Workflow for Scientific Schema Mining with Large Language Models 

**Authors**: Sameer Sadruddin, Jennifer D'Souza, Eleni Poupaki, Alex Watkins, Hamed Babaei Giglou, Anisa Rula, Bora Karasulu, Sören Auer, Adrie Mackus, Erwin Kessels  

**Link**: [PDF](https://arxiv.org/pdf/2504.00752)  

**Abstract**: Extracting structured information from unstructured text is crucial for modeling real-world processes, but traditional schema mining relies on semi-structured data, limiting scalability. This paper introduces schema-miner, a novel tool that combines large language models with human feedback to automate and refine schema extraction. Through an iterative workflow, it organizes properties from text, incorporates expert input, and integrates domain-specific ontologies for semantic depth. Applied to materials science--specifically atomic layer deposition--schema-miner demonstrates that expert-guided LLMs generate semantically rich schemas suitable for diverse real-world applications. 

---
# Do LLMs Surpass Encoders for Biomedical NER? 

**Authors**: Motasem S Obeidat, Md Sultan Al Nahian, Ramakanth Kavuluru  

**Link**: [PDF](https://arxiv.org/pdf/2504.00664)  

**Abstract**: Recognizing spans of biomedical concepts and their types (e.g., drug or gene) in free text, often called biomedical named entity recognition (NER), is a basic component of information extraction (IE) pipelines. Without a strong NER component, other applications, such as knowledge discovery and information retrieval, are not practical. State-of-the-art in NER shifted from traditional ML models to deep neural networks with transformer-based encoder models (e.g., BERT) emerging as the current standard. However, decoder models (also called large language models or LLMs) are gaining traction in IE. But LLM-driven NER often ignores positional information due to the generative nature of decoder models. Furthermore, they are computationally very expensive (both in inference time and hardware needs). Hence, it is worth exploring if they actually excel at biomedical NER and assess any associated trade-offs (performance vs efficiency). This is exactly what we do in this effort employing the same BIO entity tagging scheme (that retains positional information) using five different datasets with varying proportions of longer entities. Our results show that the LLMs chosen (Mistral and Llama: 8B range) often outperform best encoder models (BERT-(un)cased, BiomedBERT, and DeBERTav3: 300M range) by 2-8% in F-scores except for one dataset, where they equal encoder performance. This gain is more prominent among longer entities of length >= 3 tokens. However, LLMs are one to two orders of magnitude more expensive at inference time and may need cost prohibitive hardware. Thus, when performance differences are small or real time user feedback is needed, encoder models might still be more suitable than LLMs. 

---
# Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources 

**Authors**: Weizhi Wang, Yu Tian, Linjie Yang, Heng Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00595)  

**Abstract**: The reproduction of state-of-the-art multimodal LLM pre-training faces barriers at every stage of the pipeline, including high-quality data filtering, multimodal data mixture strategies, sequence packing techniques, and training frameworks. We introduce Open-Qwen2VL, a fully open-source 2B-parameter Multimodal Large Language Model pre-trained efficiently on 29M image-text pairs using only 442 A100-40G GPU hours. Our approach employs low-to-high dynamic image resolution and multimodal sequence packing to significantly enhance pre-training efficiency. The training dataset was carefully curated using both MLLM-based filtering techniques (e.g., MLM-Filter) and conventional CLIP-based filtering methods, substantially improving data quality and training efficiency. The Open-Qwen2VL pre-training is conducted on academic level 8xA100-40G GPUs at UCSB on 5B packed multimodal tokens, which is 0.36\% of 1.4T multimodal pre-training tokens of Qwen2-VL. The final instruction-tuned Open-Qwen2VL outperforms partially-open state-of-the-art MLLM Qwen2-VL-2B on various multimodal benchmarks of MMBench, SEEDBench, MMstar, and MathVista, indicating the remarkable training efficiency of Open-Qwen2VL. We open-source all aspects of our work, including compute-efficient and data-efficient training details, data filtering methods, sequence packing scripts, pre-training data in WebDataset format, FSDP-based training codebase, and both base and instruction-tuned model checkpoints. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised fine-tuning data used to develop the model. 

---
# Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning 

**Authors**: Ruoxi Xu, Yunjie Ji, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Ben He, Yingfei Sun, Xiangang Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.00472)  

**Abstract**: Although large language models (LLMs) excel in knowledge recall and reasoning, their static nature leads to outdated information as the real world evolves or when adapting to domain-specific knowledge, highlighting the need for effective knowledge injection. However, current research on knowledge injection remains superficial, mainly focusing on knowledge memorization and retrieval. This paper proposes a four-tier knowledge injection framework that systematically defines the levels of knowledge injection: memorization, retrieval, reasoning, and association. Based on this framework, we introduce DeepKnowledge, a synthetic experimental testbed designed for fine-grained evaluation of the depth of knowledge injection across three knowledge types (novel, incremental, and updated). We then explore various knowledge injection scenarios and evaluate the depth of knowledge injection for each scenario on the benchmark. Experimental results reveal key factors to reach each level of knowledge injection for LLMs and establish a mapping between the levels of knowledge injection and the corresponding suitable injection methods, aiming to provide a comprehensive approach for efficient knowledge injection across various levels. 

---
# Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents 

**Authors**: Gavin Greif, Niclas Griesshaber, Robin Greif  

**Link**: [PDF](https://arxiv.org/pdf/2504.00414)  

**Abstract**: We explore how multimodal Large Language Models (mLLMs) can help researchers transcribe historical documents, extract relevant historical information, and construct datasets from historical sources. Specifically, we investigate the capabilities of mLLMs in performing (1) Optical Character Recognition (OCR), (2) OCR Post-Correction, and (3) Named Entity Recognition (NER) tasks on a set of city directories published in German between 1754 and 1870. First, we benchmark the off-the-shelf transcription accuracy of both mLLMs and conventional OCR models. We find that the best-performing mLLM model significantly outperforms conventional state-of-the-art OCR models and other frontier mLLMs. Second, we are the first to introduce multimodal post-correction of OCR output using mLLMs. We find that this novel approach leads to a drastic improvement in transcription accuracy and consistently produces highly accurate transcriptions (<1% CER), without any image pre-processing or model fine-tuning. Third, we demonstrate that mLLMs can efficiently recognize entities in transcriptions of historical documents and parse them into structured dataset formats. Our findings provide early evidence for the long-term potential of mLLMs to introduce a paradigm shift in the approaches to historical data collection and document transcription. 

---
# Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding 

**Authors**: Mohanakrishnan Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00409)  

**Abstract**: Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding. 

---
# VNJPTranslate: A comprehensive pipeline for Vietnamese-Japanese translation 

**Authors**: Hoang Hai Phan, Nguyen Duc Minh Vu, Nam Dang Phuong  

**Link**: [PDF](https://arxiv.org/pdf/2504.00339)  

**Abstract**: Neural Machine Translation (NMT) driven by Transformer architectures has advanced significantly, yet faces challenges with low-resource language pairs like Vietnamese-Japanese (Vi-Ja). Issues include sparse parallel data and handling linguistic/cultural nuances. Recent progress in Large Language Models (LLMs) with strong reasoning, often refined via Reinforcement Learning (RL), enables high-quality synthetic data generation. We introduce VNJPTranslate, a pipeline designed to systematically address the Vi-Ja translation task. It features a targeted data augmentation strategy using advanced LLMs with Chain-of-Thought prompting for challenging segments identified via corpus analysis. Subsequently, we employ efficient fine-tuning techniques (Unsloth with QLoRA) on a capable, low-parameter autoregressive model (specifically, a fine-tuned version of the 1.8B parameter Sailor model, which is based on the Qwen architecture) to create a practical and high-performing translation system. This integrated approach aims to improve Vi-Ja translation quality significantly over existing baselines. 

---
# Leveraging Large Language Models for Automated Definition Extraction with TaxoMatic A Case Study on Media Bias 

**Authors**: Timo Spinde, Luyang Lin, Smi Hinterreiter, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00343)  

**Abstract**: This paper introduces TaxoMatic, a framework that leverages large language models to automate definition extraction from academic literature. Focusing on the media bias domain, the framework encompasses data collection, LLM-based relevance classification, and extraction of conceptual definitions. Evaluated on a dataset of 2,398 manually rated articles, the study demonstrates the frameworks effectiveness, with Claude-3-sonnet achieving the best results in both relevance classification and definition extraction. Future directions include expanding datasets and applying TaxoMatic to additional domains. 

---
# Detecting and Mitigating Bias in LLMs through Knowledge Graph-Augmented Training 

**Authors**: Rajeev Kumar, Harishankar Kumar, Kumari Shalini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00310)  

**Abstract**: Large language models have revolutionized natural language processing with their surprising capability to understand and generate human-like text. However, many of these models inherit and further amplify the biases present in their training data, raising ethical and fairness concerns. The detection and mitigation of such biases are vital to ensuring that LLMs act responsibly and equitably across diverse domains. This work investigates Knowledge Graph-Augmented Training (KGAT) as a novel method to mitigate bias in LLM. Using structured domain-specific knowledge from real-world knowledge graphs, we improve the understanding of the model and reduce biased output. Public datasets for bias assessment include Gender Shades, Bias in Bios, and FairFace, while metrics such as demographic parity and equal opportunity facilitate rigorous detection. We also performed targeted mitigation strategies to correct biased associations, leading to a significant drop in biased output and improved bias metrics. Equipped with real-world datasets and knowledge graphs, our framework is both scalable and effective, paving the way toward responsible deployment in sensitive and high-stakes applications. 

---
# VerifiAgent: a Unified Verification Agent in Language Model Reasoning 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00406)  

**Abstract**: Large language models demonstrate remarkable reasoning capabilities but often produce unreliable or incorrect responses. Existing verification methods are typically model-specific or domain-restricted, requiring significant computational resources and lacking scalability across diverse reasoning tasks. To address these limitations, we propose VerifiAgent, a unified verification agent that integrates two levels of verification: meta-verification, which assesses completeness and consistency in model responses, and tool-based adaptive verification, where VerifiAgent autonomously selects appropriate verification tools based on the reasoning type, including mathematical, logical, or commonsense reasoning. This adaptive approach ensures both efficiency and robustness across different verification scenarios. Experimental results show that VerifiAgent outperforms baseline verification methods (e.g., deductive verifier, backward verifier) among all reasoning tasks. Additionally, it can further enhance reasoning accuracy by leveraging feedback from verification results. VerifiAgent can also be effectively applied to inference scaling, achieving better results with fewer generated samples and costs compared to existing process reward models in the mathematical reasoning domain. Code is available at this https URL 

---
# Do Large Language Models Exhibit Spontaneous Rational Deception? 

**Authors**: Samuel M. Taylor, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00285)  

**Abstract**: Large Language Models (LLMs) are effective at deceiving, when prompted to do so. But under what conditions do they deceive spontaneously? Models that demonstrate better performance on reasoning tasks are also better at prompted deception. Do they also increasingly deceive spontaneously in situations where it could be considered rational to do so? This study evaluates spontaneous deception produced by LLMs in a preregistered experimental protocol using tools from signaling theory. A range of proprietary closed-source and open-source LLMs are evaluated using modified 2x2 games (in the style of Prisoner's Dilemma) augmented with a phase in which they can freely communicate to the other agent using unconstrained language. This setup creates an opportunity to deceive, in conditions that vary in how useful deception might be to an agent's rational self-interest. The results indicate that 1) all tested LLMs spontaneously misrepresent their actions in at least some conditions, 2) they are generally more likely to do so in situations in which deception would benefit them, and 3) models exhibiting better reasoning capacity overall tend to deceive at higher rates. Taken together, these results suggest a tradeoff between LLM reasoning capability and honesty. They also provide evidence of reasoning-like behavior in LLMs from a novel experimental configuration. Finally, they reveal certain contextual factors that affect whether LLMs will deceive or not. We discuss consequences for autonomous, human-facing systems driven by LLMs both now and as their reasoning capabilities continue to improve. 

---
# SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers 

**Authors**: Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00255)  

**Abstract**: This study evaluates large language models (LLMs) in generating code from algorithm descriptions from recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a multi-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implement solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful Non-Reasoning LLMs and Reasoning LLMs as foundational models. The best-performing LLM using Sci-Reproducer achieves only 39% execution accuracy, highlighting the benchmark's this http URL analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We will open-source our benchmark, and code at this https URL. 

---
# Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation 

**Authors**: Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2504.00187)  

**Abstract**: Retrieval Augmented Generation (RAG) frameworks have shown significant promise in leveraging external knowledge to enhance the performance of large language models (LLMs). However, conventional RAG methods often retrieve documents based solely on surface-level relevance, leading to many issues: they may overlook deeply buried information within individual documents, miss relevant insights spanning multiple sources, and are not well-suited for tasks beyond traditional question answering. In this paper, we propose Insight-RAG, a novel framework designed to address these issues. In the initial stage of Insight-RAG, instead of using traditional retrieval methods, we employ an LLM to analyze the input query and task, extracting the underlying informational requirements. In the subsequent stage, a specialized LLM -- trained on the document database -- is queried to mine content that directly addresses these identified insights. Finally, by integrating the original query with the retrieved insights, similar to conventional RAG approaches, we employ a final LLM to generate a contextually enriched and accurate response. Using two scientific paper datasets, we created evaluation benchmarks targeting each of the mentioned issues and assessed Insight-RAG against traditional RAG pipeline. Our results demonstrate that the Insight-RAG pipeline successfully addresses these challenges, outperforming existing methods by a significant margin in most cases. These findings suggest that integrating insight-driven retrieval within the RAG framework not only enhances performance but also broadens the applicability of RAG to tasks beyond conventional question answering. 

---
# Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency 

**Authors**: Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran  

**Link**: [PDF](https://arxiv.org/pdf/2504.00180)  

**Abstract**: Retrieval Augmented Generation (RAG) systems have emerged as a powerful method for enhancing large language models (LLMs) with up-to-date information. However, the retrieval step in RAG can sometimes surface documents containing contradictory information, particularly in rapidly evolving domains such as news. These contradictions can significantly impact the performance of LLMs, leading to inconsistent or erroneous outputs. This study addresses this critical challenge in two ways. First, we present a novel data generation framework to simulate different types of contradictions that may occur in the retrieval stage of a RAG system. Second, we evaluate the robustness of different LLMs in performing as context validators, assessing their ability to detect contradictory information within retrieved document sets. Our experimental results reveal that context validation remains a challenging task even for state-of-the-art LLMs, with performance varying significantly across different types of contradictions. While larger models generally perform better at contradiction detection, the effectiveness of different prompting strategies varies across tasks and model architectures. We find that chain-of-thought prompting shows notable improvements for some models but may hinder performance in others, highlighting the complexity of the task and the need for more robust approaches to context validation in RAG systems. 

---
# Enhancing Negation Awareness in Universal Text Embeddings: A Data-efficient and Computational-efficient Approach 

**Authors**: Hongliu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00584)  

**Abstract**: Negation plays an important role in various natural language processing tasks such as Natural Language Inference and Sentiment Analysis tasks. Numerous prior studies have found that contextual text embedding models such as BERT, ELMO, RoBERTa or XLNet face challenges in accurately understanding negation. Recent advancements in universal text embeddings have demonstrated superior performance over contextual text embeddings in various tasks. However, due to the bias in popular evaluation benchmarks, the negation awareness capacity of these models remains unclear. To bridge the gap in existing literature, an in-depth analysis is initiated in this work to study the negation awareness of cutting-edge universal text embedding models. Our findings reveal a significant lack of negation awareness in these models, often interpreting negated text pairs as semantically similar. To efficiently deal with the conflict that different tasks need different trade-offs between topic and negation information among other semantic information, a data-efficient and computational-efficient embedding re-weighting method is proposed without modifying the parameters of text embedding models. The proposed solution is able to improve text embedding models' negation awareness significantly on both simple negation understanding task and complex negation understanding task. Furthermore, the proposed solution can also significantly improve the negation awareness of Large Language Model based task-specific high dimensional universal text embeddings. 

---
# Text Chunking for Document Classification for Urban System Management using Large Language Models 

**Authors**: Joshua Rodriguez, Om Sanan, Guillermo Vizarreta-Luna, Steven A. Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2504.00274)  

**Abstract**: Urban systems are managed using complex textual documentation that need coding and analysis to set requirements and evaluate built environment performance. This paper contributes to the study of applying large-language models (LLM) to qualitative coding activities to reduce resource requirements while maintaining comparable reliability to humans. Qualitative coding and assessment face challenges like resource limitations and bias, accuracy, and consistency between human evaluators. Here we report the application of LLMs to deductively code 10 case documents on the presence of 17 digital twin characteristics for the management of urban systems. We utilize two prompting methods to compare the semantic processing of LLMs with human coding efforts: whole text analysis and text chunk analysis using OpenAI's GPT-4o, GPT-4o-mini, and o1-mini models. We found similar trends of internal variability between methods and results indicate that LLMs may perform on par with human coders when initialized with specific deductive coding contexts. GPT-4o, o1-mini and GPT-4o-mini showed significant agreement with human raters when employed using a chunking method. The application of both GPT-4o and GPT-4o-mini as an additional rater with three manual raters showed statistically significant agreement across all raters, indicating that the analysis of textual documents is benefited by LLMs. Our findings reveal nuanced sub-themes of LLM application suggesting LLMs follow human memory coding processes where whole-text analysis may introduce multiple meanings. The novel contributions of this paper lie in assessing the performance of OpenAI GPT models and introduces the chunk-based prompting approach, which addresses context aggregation biases by preserving localized context. 

---
# Making Large Language Models Better Reasoners with Orchestrated Streaming Experiences 

**Authors**: Xiangyang Liu, Junliang He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00473)  

**Abstract**: Large language models (LLMs) can perform complex reasoning by generating intermediate thoughts under zero-shot or few-shot settings. However, zero-shot prompting always encounters low performance, and the superior performance of few-shot prompting hinges on the manual-crafted demonstrations. In this paper, we present RoSE (Reasoning with Orchestrated Streaming Experiences), a general framework for solving reasoning tasks that can self-improve without complex external efforts. To enable RoSE, we describe an architecture that extends an LLM to store all answered questions and their thoughts in a streaming experience pool then orchestrates helpful questions from the pool to assist in answering new questions. To set up a question-aware orchestration mechanism, RoSE first calculates the similarity of each question in the pool with a new test question. Since the solution to each answered question is not always correct, RoSE will sort the questions according to their similarity with the new question, and then uniformly divide them into multiple buckets. It finally extracts one question from each bucket to make these extracted questions more diverse. To make these extracted questions help RoSE answer new questions as much as possible, we introduce two other attributes of uncertainty and complexity for each question. RoSE will preferentially select the questions with low uncertainty and high complexity from each bucket. We evaluate the versatility of RoSE in various reasoning tasks, LLMs, and CoT methods. 

---
# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR) 

**Authors**: Mahak Agarwal, Divyam Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2504.00374)  

**Abstract**: In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation. 

---
# Does "Reasoning" with Large Language Models Improve Recognizing, Generating, and Reframing Unhelpful Thoughts? 

**Authors**: Yilin Qi, Dong Won Lee, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.00163)  

**Abstract**: Cognitive Reframing, a core element of Cognitive Behavioral Therapy (CBT), helps individuals reinterpret negative experiences by finding positive meaning. Recent advances in Large Language Models (LLMs) have demonstrated improved performance through reasoning-based strategies. This inspires a promising direction of leveraging the reasoning capabilities of LLMs to improve CBT and mental reframing by simulating the process of critical thinking, potentially enabling more effective recognition, generation, and reframing of cognitive distortions. In this work, we investigate the role of various reasoning methods, including pre-trained reasoning LLMs and augmented reasoning strategies such as CoT and self-consistency in enhancing LLMs' ability to perform cognitive reframing tasks. We find that augmented reasoning methods, even when applied to "outdated" LLMs like GPT-3.5, consistently outperform state-of-the-art pretrained reasoning models on recognizing, generating and reframing unhelpful thoughts. 

---
# Contextualize-then-Aggregate: Circuits for In-Context Learning in Gemma-2 2B 

**Authors**: Aleksandra Bakalova, Yana Veitsman, Xinting Huang, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2504.00132)  

**Abstract**: In-Context Learning (ICL) is an intriguing ability of large language models (LLMs). Despite a substantial amount of work on its behavioral aspects and how it emerges in miniature setups, it remains unclear which mechanism assembles task information from the individual examples in a fewshot prompt. We use causal interventions to identify information flow in Gemma-2 2B for five naturalistic ICL tasks. We find that the model infers task information using a two-step strategy we call contextualize-then-aggregate: In the lower layers, the model builds up representations of individual fewshot examples, which are contextualized by preceding examples through connections between fewshot input and output tokens across the sequence. In the higher layers, these representations are aggregated to identify the task and prepare prediction of the next output. The importance of the contextualization step differs between tasks, and it may become more important in the presence of ambiguous examples. Overall, by providing rigorous causal analysis, our results shed light on the mechanisms through which ICL happens in language models. 

---
# Evaluating the Feasibility and Accuracy of Large Language Models for Medical History-Taking in Obstetrics and Gynecology 

**Authors**: Dou Liu, Ying Long, Sophia Zuoqiu, Tian Tang, Rong Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00061)  

**Abstract**: Effective physician-patient communications in pre-diagnostic environments, and most specifically in complex and sensitive medical areas such as infertility, are critical but consume a lot of time and, therefore, cause clinic workflows to become inefficient. Recent advancements in Large Language Models (LLMs) offer a potential solution for automating conversational medical history-taking and improving diagnostic accuracy. This study evaluates the feasibility and performance of LLMs in those tasks for infertility cases. An AI-driven conversational system was developed to simulate physician-patient interactions with ChatGPT-4o and ChatGPT-4o-mini. A total of 70 real-world infertility cases were processed, generating 420 diagnostic histories. Model performance was assessed using F1 score, Differential Diagnosis (DDs) Accuracy, and Accuracy of Infertility Type Judgment (ITJ). ChatGPT-4o-mini outperformed ChatGPT-4o in information extraction accuracy (F1 score: 0.9258 vs. 0.9029, p = 0.045, d = 0.244) and demonstrated higher completeness in medical history-taking (97.58% vs. 77.11%), suggesting that ChatGPT-4o-mini is more effective in extracting detailed patient information, which is critical for improving diagnostic accuracy. In contrast, ChatGPT-4o performed slightly better in differential diagnosis accuracy (2.0524 vs. 2.0048, p > 0.05). ITJ accuracy was higher in ChatGPT-4o-mini (0.6476 vs. 0.5905) but with lower consistency (Cronbach's $\alpha$ = 0.562), suggesting variability in classification reliability. Both models demonstrated strong feasibility in automating infertility history-taking, with ChatGPT-4o-mini excelling in completeness and extraction accuracy. In future studies, expert validation for accuracy and dependability in a clinical setting, AI model fine-tuning, and larger datasets with a mix of cases of infertility have to be prioritized. 

---
# CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation 

**Authors**: Jixuan Leng, Chengsong Huang, Langlin Huang, Bill Yuchen Lin, William W. Cohen, Haohan Wang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00043)  

**Abstract**: Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly either assess text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles-a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in multiple formats (text and image) and offers different evaluation strategies ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs outperform non-reasoning models substantially by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings offer insights into the limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations. 

---
# Beyond the Reported Cutoff: Where Large Language Models Fall Short on Financial Knowledge 

**Authors**: Agam Shah, Liqin Ye, Sebastian Jaskowski, Wei Xu, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2504.00042)  

**Abstract**: Large Language Models (LLMs) are frequently utilized as sources of knowledge for question-answering. While it is known that LLMs may lack access to real-time data or newer data produced after the model's cutoff date, it is less clear how their knowledge spans across historical information. In this study, we assess the breadth of LLMs' knowledge using financial data of U.S. publicly traded companies by evaluating more than 197k questions and comparing model responses to factual data. We further explore the impact of company characteristics, such as size, retail investment, institutional attention, and readability of financial filings, on the accuracy of knowledge represented in LLMs. Our results reveal that LLMs are less informed about past financial performance, but they display a stronger awareness of larger companies and more recent information. Interestingly, at the same time, our analysis also reveals that LLMs are more likely to hallucinate for larger companies, especially for data from more recent years. We will make the code, prompts, and model outputs public upon the publication of the work. 

---
# JudgeLRM: Large Reasoning Models as a Judge 

**Authors**: Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00050)  

**Abstract**: The rise of Large Language Models (LLMs) as evaluators offers a scalable alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for judges approaches often fall short in domains requiring complex reasoning. In this work, we investigate whether LLM judges truly benefit from enhanced reasoning capabilities. Through a detailed analysis of reasoning requirements across evaluation tasks, we reveal a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples - highlighting the limitations of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM models consistently outperform both SFT-tuned and state-of-the-art reasoning models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks requiring deep reasoning. 

---
# Multi-Stakeholder Disaster Insights from Social Media Using Large Language Models 

**Authors**: Loris Belcastro, Cristian Cosentino, Fabrizio Marozzo, Merve Gündüz-Cüre, Şule Öztürk-Birim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00046)  

**Abstract**: In recent years, social media has emerged as a primary channel for users to promptly share feedback and issues during disasters and emergencies, playing a key role in crisis management. While significant progress has been made in collecting and analyzing social media content, there remains a pressing need to enhance the automation, aggregation, and customization of this data to deliver actionable insights tailored to diverse stakeholders, including the press, police, EMS, and firefighters. This effort is essential for improving the coordination of activities such as relief efforts, resource distribution, and media communication. This paper presents a methodology that leverages the capabilities of LLMs to enhance disaster response and management. Our approach combines classification techniques with generative AI to bridge the gap between raw user feedback and stakeholder-specific reports. Social media posts shared during catastrophic events are analyzed with a focus on user-reported issues, service interruptions, and encountered challenges. We employ full-spectrum LLMs, using analytical models like BERT for precise, multi-dimensional classification of content type, sentiment, emotion, geolocation, and topic. Generative models such as ChatGPT are then used to produce human-readable, informative reports tailored to distinct audiences, synthesizing insights derived from detailed classifications. We compare standard approaches, which analyze posts directly using prompts in ChatGPT, to our advanced method, which incorporates multi-dimensional classification, sub-event selection, and tailored report generation. Our methodology demonstrates superior performance in both quantitative metrics, such as text coherence scores and latent representations, and qualitative assessments by automated tools and field experts, delivering precise insights for diverse disaster response stakeholders. 

---
# Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs 

**Authors**: Cong Duy Vu Hoang, Gioacchino Tangari, Clemence Lanfranchi, Dalu Guo, Paul Cayet, Steve Siu, Don Dharmasiri, Yuan-Fang Li, Long Duong, Damien Hilloulin, Rhicheek Patra, Sungpack Hong, Hassan Chafi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00048)  

**Abstract**: The growing adoption of large language models (LLMs) in business applications has amplified interest in Natural Language to SQL (NL2SQL) solutions, in which there is competing demand for high performance and efficiency. Domain- and customer-specific requirements further complicate the problem. To address this conundrum, we introduce Distill-C, a distilled customization framework tailored for NL2SQL tasks. Distill-C utilizes large teacher LLMs to produce high-quality synthetic data through a robust and scalable pipeline. Finetuning smaller and open-source LLMs on this synthesized data enables them to rival or outperform teacher models an order of magnitude larger. Evaluated on multiple challenging benchmarks, Distill-C achieves an average improvement of 36% in execution accuracy compared to the base models from three distinct LLM families. Additionally, on three internal customer benchmarks, Distill-C demonstrates a 22.6% performance improvement over the base models. Our results demonstrate that Distill-C is an effective, high-performing and generalizable approach for deploying lightweight yet powerful NL2SQL models, delivering exceptional accuracies while maintaining low computational cost. 

---
# Generalization Bias in Large Language Model Summarization of Scientific Research 

**Authors**: Uwe Peters, Benjamin Chin-Yee  

**Link**: [PDF](https://arxiv.org/pdf/2504.00025)  

**Abstract**: Artificial intelligence chatbots driven by large language models (LLMs) have the potential to increase public science literacy and support scientific research, as they can quickly summarize complex scientific information in accessible terms. However, when summarizing scientific texts, LLMs may omit details that limit the scope of research conclusions, leading to generalizations of results broader than warranted by the original study. We tested 10 prominent LLMs, including ChatGPT-4o, ChatGPT-4.5, DeepSeek, LLaMA 3.3 70B, and Claude 3.7 Sonnet, comparing 4900 LLM-generated summaries to their original scientific texts. Even when explicitly prompted for accuracy, most LLMs produced broader generalizations of scientific results than those in the original texts, with DeepSeek, ChatGPT-4o, and LLaMA 3.3 70B overgeneralizing in 26 to 73% of cases. In a direct comparison of LLM-generated and human-authored science summaries, LLM summaries were nearly five times more likely to contain broad generalizations (OR = 4.85, 95% CI [3.06, 7.70]). Notably, newer models tended to perform worse in generalization accuracy than earlier ones. Our results indicate a strong bias in many widely used LLMs towards overgeneralizing scientific conclusions, posing a significant risk of large-scale misinterpretations of research findings. We highlight potential mitigation strategies, including lowering LLM temperature settings and benchmarking LLMs for generalization accuracy. 

---
# Integrating Large Language Models with Human Expertise for Disease Detection in Electronic Health Records 

**Authors**: Jie Pan, Seungwon Lee, Cheligeer Cheligeer, Elliot A. Martin, Kiarash Riazi, Hude Quan, Na Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00053)  

**Abstract**: Objective: Electronic health records (EHR) are widely available to complement administrative data-based disease surveillance and healthcare performance evaluation. Defining conditions from EHR is labour-intensive and requires extensive manual labelling of disease outcomes. This study developed an efficient strategy based on advanced large language models to identify multiple conditions from EHR clinical notes. Methods: We linked a cardiac registry cohort in 2015 with an EHR system in Alberta, Canada. We developed a pipeline that leveraged a generative large language model (LLM) to analyze, understand, and interpret EHR notes by prompts based on specific diagnosis, treatment management, and clinical guidelines. The pipeline was applied to detect acute myocardial infarction (AMI), diabetes, and hypertension. The performance was compared against clinician-validated diagnoses as the reference standard and widely adopted International Classification of Diseases (ICD) codes-based methods. Results: The study cohort accounted for 3,088 patients and 551,095 clinical notes. The prevalence was 55.4%, 27.7%, 65.9% and for AMI, diabetes, and hypertension, respectively. The performance of the LLM-based pipeline for detecting conditions varied: AMI had 88% sensitivity, 63% specificity, and 77% positive predictive value (PPV); diabetes had 91% sensitivity, 86% specificity, and 71% PPV; and hypertension had 94% sensitivity, 32% specificity, and 72% PPV. Compared with ICD codes, the LLM-based method demonstrated improved sensitivity and negative predictive value across all conditions. The monthly percentage trends from the detected cases by LLM and reference standard showed consistent patterns. 

---
# Synthesizing Public Opinions with LLMs: Role Creation, Impacts, and the Future to eDemorcacy 

**Authors**: Rabimba Karanjai, Boris Shor, Amanda Austin, Ryan Kennedy, Yang Lu, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00241)  

**Abstract**: This paper investigates the use of Large Language Models (LLMs) to synthesize public opinion data, addressing challenges in traditional survey methods like declining response rates and non-response bias. We introduce a novel technique: role creation based on knowledge injection, a form of in-context learning that leverages RAG and specified personality profiles from the HEXACO model and demographic information, and uses that for dynamically generated prompts. This method allows LLMs to simulate diverse opinions more accurately than existing prompt engineering approaches. We compare our results with pre-trained models with standard few-shot prompts. Experiments using questions from the Cooperative Election Study (CES) demonstrate that our role-creation approach significantly improves the alignment of LLM-generated opinions with real-world human survey responses, increasing answer adherence. In addition, we discuss challenges, limitations and future research directions. 

---
# Medical Reasoning in LLMs: An In-Depth Analysis of DeepSeek R1 

**Authors**: Birger Moell, Fredrik Sand Aronsson, Sanian Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2504.00016)  

**Abstract**: Integrating large language models (LLMs) like DeepSeek R1 into healthcare requires rigorous evaluation of their reasoning alignment with clinical expertise. This study assesses DeepSeek R1's medical reasoning against expert patterns using 100 MedQA clinical cases. The model achieved 93% diagnostic accuracy, demonstrating systematic clinical judgment through differential diagnosis, guideline-based treatment selection, and integration of patient-specific factors. However, error analysis of seven incorrect cases revealed persistent limitations: anchoring bias, challenges reconciling conflicting data, insufficient exploration of alternatives, overthinking, knowledge gaps, and premature prioritization of definitive treatment over intermediate care. Crucially, reasoning length correlated with accuracy - shorter responses (<5,000 characters) were more reliable, suggesting extended explanations may signal uncertainty or rationalization of errors. While DeepSeek R1 exhibits foundational clinical reasoning capabilities, recurring flaws highlight critical areas for refinement, including bias mitigation, knowledge updates, and structured reasoning frameworks. These findings underscore LLMs' potential to augment medical decision-making through artificial reasoning but emphasize the need for domain-specific validation, interpretability safeguards, and confidence metrics (e.g., response length thresholds) to ensure reliability in real-world applications. 

---
# SRLCG: Self-Rectified Large-Scale Code Generation with Multidimensional Chain-of-Thought and Dynamic Backtracking 

**Authors**: Hongru Ma, Yanjie Liang, Jiasheng Si, Weiyu Zhang, Hongjiao Guan, Chaoqun Zheng, Bing Xu, Wenpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00532)  

**Abstract**: Large language models (LLMs) have revolutionized code generation, significantly enhancing developer productivity. However, for a vast number of users with minimal coding knowledge, LLMs provide little support, as they primarily generate isolated code snippets rather than complete, large-scale project code. Without coding expertise, these users struggle to interpret, modify, and iteratively refine the outputs of LLMs, making it impossible to assemble a complete project. To address this issue, we propose Self-Rectified Large-Scale Code Generator (SRLCG), a framework that generates complete multi-file project code from a single prompt. SRLCG employs a novel multidimensional chain-of-thought (CoT) and self-rectification to guide LLMs in generating correct and robust code files, then integrates them into a complete and coherent project using our proposed dynamic backtracking algorithm. Experimental results show that SRLCG generates code 15x longer than DeepSeek-V3, 16x longer than GPT-4, and at least 10x longer than other leading CoT-based baselines. Furthermore, they confirm its improved correctness, robustness, and performance compared to baselines in large-scale code generation. 

---
# Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems? 

**Authors**: Kai Yan, Yufei Xu, Zhengyin Du, Xuesong Yao, Zheyu Wang, Xiaowen Guo, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00509)  

**Abstract**: The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surpassing human intelligence. However, is the LLMs' remarkable reasoning ability indeed comes from true intelligence by human standards, or are they simply reciting solutions witnessed during training at an Internet level? To study this problem, we propose RoR-Bench, a novel, multi-modal benchmark for detecting LLM's recitation behavior when asked simple reasoning problems but with conditions subtly shifted, and conduct empirical analysis on our benchmark. Surprisingly, we found existing cutting-edge LLMs unanimously exhibits extremely severe recitation behavior; by changing one phrase in the condition, top models such as OpenAI-o1 and DeepSeek-R1 can suffer $60\%$ performance loss on elementary school-level arithmetic and reasoning problems. Such findings are a wake-up call to the LLM community that compels us to re-evaluate the true intelligence level of cutting-edge LLMs. 

---
# Token-Driven GammaTune: Adaptive Calibration for Enchanced Speculative Decoding 

**Authors**: Aayush Gautam, Susav Shrestha, Narasimha Annapareddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.00030)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, selecting an optimal speculation length is critical for maximizing speedup while minimizing wasted computation. We introduce \textit{GammaTune} and \textit{GammaTune+}, training-free adaptive algorithms that dynamically adjust speculation length based on token acceptance rates using a heuristic-based switching mechanism. Evaluated on SpecBench across multiple tasks and model pairs, our method outperforms other heuristic-based approaches and fixed-length speculative decoding, achieving an average speedup of 15\% ($\pm$5\%) with \textit{GammaTune} and 16\% ($\pm$3\%) with \textit{GammaTune+}, while reducing performance variance. This makes \textit{GammaTune} a robust and efficient solution for real-world deployment. 

---
# Automated Explanation of Machine Learning Models of Footballing Actions in Words 

**Authors**: Pegah Rahimian, Jernej Flisar, David Sumpter  

**Link**: [PDF](https://arxiv.org/pdf/2504.00767)  

**Abstract**: While football analytics has changed the way teams and analysts assess performance, there remains a communication gap between machine learning practice and how coaching staff talk about football. Coaches and practitioners require actionable insights, which are not always provided by models. To bridge this gap, we show how to build wordalizations (a novel approach that leverages large language models) for shots in football. Specifically, we first build an expected goals model using logistic regression. We then use the co-efficients of this regression model to write sentences describing how factors (such as distance, angle and defensive pressure) contribute to the model's prediction. Finally, we use large language models to give an entertaining description of the shot. We describe our approach in a model card and provide an interactive open-source application describing shots in recent tournaments. We discuss how shot wordalisations might aid communication in coaching and football commentary, and give a further example of how the same approach can be applied to other actions in football. 

---
# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems 

**Authors**: Yingxuan Yang, Huacan Chai, Shuai Shao, Yuanyi Song, Siyuan Qi, Renting Rui, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00587)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has catalyzed the development of multi-agent systems, where multiple LLM-based agents collaborate to solve complex tasks. However, existing systems predominantly rely on centralized coordination, which introduces scalability bottlenecks, limits adaptability, and creates single points of failure. Additionally, concerns over privacy and proprietary knowledge sharing hinder cross-organizational collaboration, leading to siloed expertise. To address these challenges, we propose AgentNet, a decentralized, Retrieval-Augmented Generation (RAG)-based framework that enables LLM-based agents to autonomously evolve their capabilities and collaborate efficiently in a Directed Acyclic Graph (DAG)-structured network. Unlike traditional multi-agent systems that depend on static role assignments or centralized control, AgentNet allows agents to specialize dynamically, adjust their connectivity, and route tasks without relying on predefined workflows. AgentNet's core design is built upon several key innovations: (1) Fully Decentralized Paradigm: Removing the central orchestrator, allowing agents to coordinate and specialize autonomously, fostering fault tolerance and emergent collective intelligence. (2) Dynamically Evolving Graph Topology: Real-time adaptation of agent connections based on task demands, ensuring scalability and resilience.(3) Adaptive Learning for Expertise Refinement: A retrieval-based memory system that enables agents to continuously update and refine their specialized skills. By eliminating centralized control, AgentNet enhances fault tolerance, promotes scalable specialization, and enables privacy-preserving collaboration across organizations. Through decentralized coordination and minimal data exchange, agents can leverage diverse knowledge sources while safeguarding sensitive information. 

---
# Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead 

**Authors**: Vidhisha Balachandran, Jingya Chen, Lingjiao Chen, Shivam Garg, Neel Joshi, Yash Lara, John Langford, Besmira Nushi, Vibhav Vineet, Yue Wu, Safoora Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00294)  

**Abstract**: Inference-time scaling can enhance the reasoning capabilities of large language models (LLMs) on complex problems that benefit from step-by-step problem solving. Although lengthening generated scratchpads has proven effective for mathematical tasks, the broader impact of this approach on other tasks remains less clear. In this work, we investigate the benefits and limitations of scaling methods across nine state-of-the-art models and eight challenging tasks, including math and STEM reasoning, calendar planning, NP-hard problems, navigation, and spatial reasoning. We compare conventional models (e.g., GPT-4o) with models fine-tuned for inference-time scaling (e.g., o1) through evaluation protocols that involve repeated model calls, either independently or sequentially with feedback. These evaluations approximate lower and upper performance bounds and potential for future performance improvements for each model, whether through enhanced training or multi-model inference systems. Our extensive empirical analysis reveals that the advantages of inference-time scaling vary across tasks and diminish as problem complexity increases. In addition, simply using more tokens does not necessarily translate to higher accuracy in these challenging regimes. Results from multiple independent runs with conventional models using perfect verifiers show that, for some tasks, these models can achieve performance close to the average performance of today's most advanced reasoning models. However, for other tasks, a significant performance gap remains, even in very high scaling regimes. Encouragingly, all models demonstrate significant gains when inference is further scaled with perfect verifiers or strong feedback, suggesting ample potential for future improvements. 

---
# Leaking LoRa: An Evaluation of Password Leaks and Knowledge Storage in Large Language Models 

**Authors**: Ryan Marinelli, Magnus Eckhoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.00031)  

**Abstract**: To effectively deploy Large Language Models (LLMs) in application-specific settings, fine-tuning techniques are applied to enhance performance on specialized tasks. This process often involves fine-tuning on user data data, which may contain sensitive information. Although not recommended, it is not uncommon for users to send passwords in messages, and fine-tuning models on this could result in passwords being leaked. In this study, a Large Language Model is fine-tuned with customer support data and passwords from the RockYou password wordlist using Low-Rank Adaptation (LoRA). Out of the first 200 passwords from the list, 37 were successfully recovered. Further, causal tracing is used to identify that password information is largely located in a few layers. Lastly, Rank One Model Editing (ROME) is used to remove the password information from the model, resulting in the number of passwords recovered going from 37 to 0. 

---
# LLMs for Explainable AI: A Comprehensive Survey 

**Authors**: Ahsan Bilal, David Ebert, Beiyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00125)  

**Abstract**: Large Language Models (LLMs) offer a promising approach to enhancing Explainable AI (XAI) by transforming complex machine learning outputs into easy-to-understand narratives, making model predictions more accessible to users, and helping bridge the gap between sophisticated model behavior and human interpretability. AI models, such as state-of-the-art neural networks and deep learning models, are often seen as "black boxes" due to a lack of transparency. As users cannot fully understand how the models reach conclusions, users have difficulty trusting decisions from AI models, which leads to less effective decision-making processes, reduced accountabilities, and unclear potential biases. A challenge arises in developing explainable AI (XAI) models to gain users' trust and provide insights into how models generate their outputs. With the development of Large Language Models, we want to explore the possibilities of using human language-based models, LLMs, for model explainabilities. This survey provides a comprehensive overview of existing approaches regarding LLMs for XAI, and evaluation techniques for LLM-generated explanation, discusses the corresponding challenges and limitations, and examines real-world applications. Finally, we discuss future directions by emphasizing the need for more interpretable, automated, user-centric, and multidisciplinary approaches for XAI via LLMs. 

---
# $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks 

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00218)  

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms. 

---
# Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scale Test-Time Compute 

**Authors**: Jianhao Chen, Zishuo Xun, Bocheng Zhou, Han Qi, Qiaosheng Zhang, Yang Chen, Wei Hu, Yuzhong Qu, Wanli Ouyang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00762)  

**Abstract**: This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm. 

---
# Investigating Large Language Models in Diagnosing Students' Cognitive Skills in Math Problem-solving 

**Authors**: Hyoungwook Jin, Yoonsu Kim, Dongyun Jung, Seungju Kim, Kiyoon Choi, Jinho Son, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00843)  

**Abstract**: Mathematics learning entails mastery of both content knowledge and cognitive processing of knowing, applying, and reasoning with it. Automated math assessment primarily has focused on grading students' exhibition of content knowledge by finding textual evidence, such as specific numbers, formulas, and statements. Recent advancements in problem-solving, image recognition, and reasoning capabilities of large language models (LLMs) show promise for nuanced evaluation of students' cognitive skills. Diagnosing cognitive skills needs to infer students' thinking processes beyond textual evidence, which is an underexplored task in LLM-based automated assessment. In this work, we investigate how state-of-the-art LLMs diagnose students' cognitive skills in mathematics. We constructed MathCog, a novel benchmark dataset comprising 639 student responses to 110 expert-curated middle school math problems, each annotated with detailed teachers' diagnoses based on cognitive skill checklists. Using MathCog, we evaluated 16 closed and open LLMs of varying model sizes and vendors. Our evaluation reveals that even the state-of-the-art LLMs struggle with the task, all F1 scores below 0.5, and tend to exhibit strong false confidence for incorrect cases ($r_s=.617$). We also found that model size positively correlates with the diagnosis performance ($r_s=.771$). Finally, we discuss the implications of these findings, the overconfidence issue, and directions for improving automated cognitive skill diagnosis. 

---
# Personality-Driven Decision-Making in LLM-Based Autonomous Agents 

**Authors**: Lewis Newsham, Daniel Prince  

**Link**: [PDF](https://arxiv.org/pdf/2504.00727)  

**Abstract**: The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies. 

---
# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning 

**Authors**: Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00907)  

**Abstract**: Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL. 

---
# Hawkeye:Efficient Reasoning with Model Collaboration 

**Authors**: Jianshu She, Zhuohao Li, Zhemin Huang, Qi Li, Peiran Xu, Haonan Li, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2504.00424)  

**Abstract**: Chain-of-Thought (CoT) reasoning has demonstrated remarkable effectiveness in enhancing the reasoning abilities of large language models (LLMs). However, its efficiency remains a challenge due to the generation of excessive intermediate reasoning tokens, which introduce semantic redundancy and overly detailed reasoning steps. Moreover, computational expense and latency are significant concerns, as the cost scales with the number of output tokens, including those intermediate steps. In this work, we observe that most CoT tokens are unnecessary, and retaining only a small portion of them is sufficient for producing high-quality responses. Inspired by this, we propose HAWKEYE, a novel post-training and inference framework where a large model produces concise CoT instructions to guide a smaller model in response generation. HAWKEYE quantifies redundancy in CoT reasoning and distills high-density information via reinforcement learning. By leveraging these concise CoTs, HAWKEYE is able to expand responses while reducing token usage and computational cost significantly. Our evaluation shows that HAWKEYE can achieve comparable response quality using only 35% of the full CoTs, while improving clarity, coherence, and conciseness by approximately 10%. Furthermore, HAWKEYE can accelerate end-to-end reasoning by up to 3.4x on complex math tasks while reducing inference cost by up to 60%. HAWKEYE will be open-sourced and the models will be available soon. 

---
# Large Language Models in Numberland: A Quick Test of Their Numerical Reasoning Abilities 

**Authors**: Roussel Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2504.00226)  

**Abstract**: An essential element of human mathematical reasoning is our number sense -- an abstract understanding of numbers and their relationships -- which allows us to solve problems involving vast number spaces using limited computational resources. Mathematical reasoning of Large Language Models (LLMs) is often tested on high-level problems (such as Olympiad challenges, geometry, word problems, and puzzles), but their low-level number sense remains less explored. We introduce "Numberland," a 100-problem test to evaluate the numerical reasoning abilities of LLM-based agents. The tasks -- basic operations, advanced calculations (e.g., exponentiation, complex numbers), prime number checks, and the 24 game -- aim to test elementary skills and their integration in solving complex and uncertain problems. We evaluated five LLM-based agents: OpenAI's o1 and o1-mini, Google Gemini, Microsoft Copilot, and Anthropic Claude. They scored 74-95% on the first three tasks that allow deterministic steps to solutions. In the 24 game, which needs trial-and-error search, performance dropped to 10-73%. We tested the top 24 solver (o1 with 73% accuracy) on 25 harder problems, and its score fell to 27%, confirming search as a bottleneck. These results, along with the types of mistakes, suggest a fragile number of LLMs, which is a bit surprising given their prowess in challenging benchmarks. The limits of LLM numerical reasoning highlight the scope of simple, targeted tests to evaluate and explain LLM math skills to ensure safe use. 

---
# LLM-Guided Search for Deletion-Correcting Codes 

**Authors**: Franziska Weindel, Reinhard Heckel  

**Link**: [PDF](https://arxiv.org/pdf/2504.00613)  

**Abstract**: Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths \( n = 12, 13 \), and \( 16 \), establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes. 

---
# Collaborative LLM Numerical Reasoning with Local Data Protection 

**Authors**: Min Zhang, Yuzhe Lu, Yun Zhou, Panpan Xu, Lin Lee Cheong, Chang-Tien Lu, Haozhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00299)  

**Abstract**: Numerical reasoning over documents, which demands both contextual understanding and logical inference, is challenging for low-capacity local models deployed on computation-constrained devices. Although such complex reasoning queries could be routed to powerful remote models like GPT-4, exposing local data raises significant data leakage concerns. Existing mitigation methods generate problem descriptions or examples for remote assistance. However, the inherent complexity of numerical reasoning hinders the local model from generating logically equivalent queries and accurately inferring answers with remote guidance. In this paper, we present a model collaboration framework with two key innovations: (1) a context-aware synthesis strategy that shifts the query domains while preserving logical consistency; and (2) a tool-based answer reconstruction approach that reuses the remote-generated problem-solving pattern with code snippets. Experimental results demonstrate that our method achieves better reasoning accuracy than solely using local models while providing stronger data protection than fully relying on remote models. Furthermore, our method improves accuracy by 16.2% - 43.6% while reducing data leakage by 2.3% - 44.6% compared to existing data protection approaches. 

---
# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights 

**Authors**: Yuchen Liu, Lino Lerch, Luigi Palmieri, Andrey Rudenko, Sebastian Koch, Timo Ropinski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2504.00839)  

**Abstract**: Predicting human behavior in shared environments is crucial for safe and efficient human-robot interaction. Traditional data-driven methods to that end are pre-trained on domain-specific datasets, activity types, and prediction horizons. In contrast, the recent breakthroughs in Large Language Models (LLMs) promise open-ended cross-domain generalization to describe various human activities and make predictions in any context. In particular, Multimodal LLMs (MLLMs) are able to integrate information from various sources, achieving more contextual awareness and improved scene understanding. The difficulty in applying general-purpose MLLMs directly for prediction stems from their limited capacity for processing large input sequences, sensitivity to prompt design, and expensive fine-tuning. In this paper, we present a systematic analysis of applying pre-trained MLLMs for context-aware human behavior prediction. To this end, we introduce a modular multimodal human activity prediction framework that allows us to benchmark various MLLMs, input variations, In-Context Learning (ICL), and autoregressive techniques. Our evaluation indicates that the best-performing framework configuration is able to reach 92.8% semantic similarity and 66.1% exact label accuracy in predicting human behaviors in the target frame. 

---
# Automated detection of atomicity violations in large-scale systems 

**Authors**: Hang He, Yixing Luo, Chengcheng Wan, Ting Su, Haiying Sun, Geguang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00521)  

**Abstract**: Atomicity violations in interrupt-driven programs pose a significant threat to software safety in critical systems. These violations occur when the execution sequence of operations on shared resources is disrupted by asynchronous interrupts. Detecting atomicity violations is challenging due to the vast program state space, application-level code dependencies, and complex domain-specific knowledge. We propose Clover, a hybrid framework that integrates static analysis with large language model (LLM) agents to detect atomicity violations in real-world programs. Clover first performs static analysis to extract critical code snippets and operation information. It then initiates a multi-agent process, where the expert agent leverages domain-specific knowledge to detect atomicity violations, which are subsequently validated by the judge agent. Evaluations on RaceBench 2.1, SV-COMP, and RWIP demonstrate that Clover achieves a precision/recall of 92.3%/86.6%, outperforming existing approaches by 27.4-118.2% on F1-score. 

---
# LLM-Assisted Proactive Threat Intelligence for Automated Reasoning 

**Authors**: Shuva Paul, Farhad Alemi, Richard Macwan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00428)  

**Abstract**: Successful defense against dynamically evolving cyber threats requires advanced and sophisticated techniques. This research presents a novel approach to enhance real-time cybersecurity threat detection and response by integrating large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems with continuous threat intelligence feeds. Leveraging recent advancements in LLMs, specifically GPT-4o, and the innovative application of RAG techniques, our approach addresses the limitations of traditional static threat analysis by incorporating dynamic, real-time data sources. We leveraged RAG to get the latest information in real-time for threat intelligence, which is not possible in the existing GPT-4o model. We employ the Patrowl framework to automate the retrieval of diverse cybersecurity threat intelligence feeds, including Common Vulnerabilities and Exposures (CVE), Common Weakness Enumeration (CWE), Exploit Prediction Scoring System (EPSS), and Known Exploited Vulnerabilities (KEV) databases, and integrate these with the all-mpnet-base-v2 model for high-dimensional vector embeddings, stored and queried in Milvus. We demonstrate our system's efficacy through a series of case studies, revealing significant improvements in addressing recently disclosed vulnerabilities, KEVs, and high-EPSS-score CVEs compared to the baseline GPT-4o. This work not only advances the role of LLMs in cybersecurity but also establishes a robust foundation for the development of automated intelligent cyberthreat information management systems, addressing crucial gaps in current cybersecurity practices. 

---
# Do Chinese models speak Chinese languages? 

**Authors**: Andrea W Wen-Yi, Unso Eun Seo Jo, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2504.00289)  

**Abstract**: The release of top-performing open-weight LLMs has cemented China's role as a leading force in AI development. Do these models support languages spoken in China? Or do they speak the same languages as Western models? Comparing multilingual capabilities is important for two reasons. First, language ability provides insights into pre-training data curation, and thus into resource allocation and development priorities. Second, China has a long history of explicit language policy, varying between inclusivity of minority languages and a Mandarin-first policy. To test whether Chinese LLMs today reflect an agenda about China's languages, we test performance of Chinese and Western open-source LLMs on Asian regional and Chinese minority languages. Our experiments on Information Parity and reading comprehension show Chinese models' performance across these languages correlates strongly (r=0.93) with Western models', with the sole exception being better Mandarin. Sometimes, Chinese models cannot identify languages spoken by Chinese minorities such as Kazakh and Uyghur, even though they are good at French and German. These results provide a window into current development priorities, suggest options for future development, and indicate guidance for end users. 

---
# Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments 

**Authors**: Joshua Moore, Aly Sabri Abdalla, Prabesh Khanal, Vuk Marojevic  

**Link**: [PDF](https://arxiv.org/pdf/2504.00341)  

**Abstract**: The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy. 

---
# Assessing Code Understanding in LLMs 

**Authors**: Cosimo Laneve, Alvise Spanò, Dalila Ressi, Sabina Rossi, Michele Bugliesi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00065)  

**Abstract**: We present an empirical evaluation of Large Language Models in code understanding associated with non-trivial, semantic-preserving program transformations such as copy propagation or constant folding. Our findings show that LLMs fail to judge semantic equivalence in approximately 41\% of cases when no context is provided and in 29\% when given a simple generic context. To improve accuracy, we advocate integrating LLMs with code-optimization tools to enhance training and facilitate more robust program understanding. 

---
# GazeLLM: Multimodal LLMs incorporating Human Visual Attention 

**Authors**: Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.00221)  

**Abstract**: Large Language Models (LLMs) are advancing into Multimodal LLMs (MLLMs), capable of processing image, audio, and video as well as text. Combining first-person video, MLLMs show promising potential for understanding human activities through video and audio, enabling many human-computer interaction and human-augmentation applications such as human activity support, real-world agents, and skill transfer to robots or other individuals. However, handling high-resolution, long-duration videos generates large latent representations, leading to substantial memory and processing demands, limiting the length and resolution MLLMs can manage. Reducing video resolution can lower memory usage but often compromises comprehension. This paper introduces a method that optimizes first-person video analysis by integrating eye-tracking data, and proposes a method that decomposes first-person vision video into sub areas for regions of gaze focus. By processing these selectively gazed-focused inputs, our approach achieves task comprehension equivalent to or even better than processing the entire image at full resolution, but with significantly reduced video data input (reduce the number of pixels to one-tenth), offering an efficient solution for using MLLMs to interpret and utilize human skills. 

---
# Generating Structured Plan Representation of Procedures with LLMs 

**Authors**: Deepeka Garg, Sihan Zeng, Sumitra Ganesh, Leo Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2504.00029)  

**Abstract**: In this paper, we address the challenges of managing Standard Operating Procedures (SOPs), which often suffer from inconsistencies in language, format, and execution, leading to operational inefficiencies. Traditional process modeling demands significant manual effort, domain expertise, and familiarity with complex languages like Business Process Modeling Notation (BPMN), creating barriers for non-techincal users. We introduce SOP Structuring (SOPStruct), a novel approach that leverages Large Language Models (LLMs) to transform SOPs into decision-tree-based structured representations. SOPStruct produces a standardized representation of SOPs across different domains, reduces cognitive load, and improves user comprehension by effectively capturing task dependencies and ensuring sequential integrity. Our approach enables leveraging the structured information to automate workflows as well as empower the human users. By organizing procedures into logical graphs, SOPStruct facilitates backtracking and error correction, offering a scalable solution for process optimization. We employ a novel evaluation framework, combining deterministic methods with the Planning Domain Definition Language (PDDL) to verify graph soundness, and non-deterministic assessment by an LLM to ensure completeness. We empirically validate the robustness of our LLM-based structured SOP representation methodology across SOPs from different domains and varying levels of complexity. Despite the current lack of automation readiness in many organizations, our research highlights the transformative potential of LLMs to streamline process modeling, paving the way for future advancements in automated procedure optimization. 

---
