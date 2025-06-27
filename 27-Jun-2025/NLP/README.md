# Data Efficacy for Language Model Training 

**Authors**: Yalun Dai, Yangyu Huang, Xin Zhang, Wenshan Wu, Chong Li, Wenhui Lu, Shijie Cao, Li Dong, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21545)  

**Abstract**: Data is fundamental to the training of language models (LM). Recent research has been dedicated to data efficiency, which aims to maximize performance by selecting a minimal or optimal subset of training data. Techniques such as data filtering, sampling, and selection play a crucial role in this area. To complement it, we define Data Efficacy, which focuses on maximizing performance by optimizing the organization of training data and remains relatively underexplored. This work introduces a general paradigm, DELT, for considering data efficacy in LM training, which highlights the significance of training data organization. DELT comprises three components: Data Scoring, Data Selection, and Data Ordering. Among these components, we design Learnability-Quality Scoring (LQS), as a new instance of Data Scoring, which considers both the learnability and quality of each data sample from the gradient consistency perspective. We also devise Folding Ordering (FO), as a novel instance of Data Ordering, which addresses issues such as model forgetting and data distribution bias. Comprehensive experiments validate the data efficacy in LM training, which demonstrates the following: Firstly, various instances of the proposed DELT enhance LM performance to varying degrees without increasing the data scale and model size. Secondly, among these instances, the combination of our proposed LQS for data scoring and Folding for data ordering achieves the most significant improvement. Lastly, data efficacy can be achieved together with data efficiency by applying data selection. Therefore, we believe that data efficacy is a promising foundational area in LM training. 

---
# "What's Up, Doc?": Analyzing How Users Seek Health Information in Large-Scale Conversational AI Datasets 

**Authors**: Akshay Paruchuri, Maryam Aziz, Rohit Vartak, Ayman Ali, Best Uchehara, Xin Liu, Ishan Chatterjee, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21532)  

**Abstract**: People are increasingly seeking healthcare information from large language models (LLMs) via interactive chatbots, yet the nature and inherent risks of these conversations remain largely unexplored. In this paper, we filter large-scale conversational AI datasets to achieve HealthChat-11K, a curated dataset of 11K real-world conversations composed of 25K user messages. We use HealthChat-11K and a clinician-driven taxonomy for how users interact with LLMs when seeking healthcare information in order to systematically study user interactions across 21 distinct health specialties. Our analysis reveals insights into the nature of how and why users seek health information, such as common interactions, instances of incomplete context, affective behaviors, and interactions (e.g., leading questions) that can induce sycophancy, underscoring the need for improvements in the healthcare support capabilities of LLMs deployed as conversational AI. Code and artifacts to retrieve our analyses and combine them into a curated dataset can be found here: this https URL 

---
# Potemkin Understanding in Large Language Models 

**Authors**: Marina Mancoridis, Bec Weeks, Keyon Vafa, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21521)  

**Abstract**: Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations. 

---
# skLEP: A Slovak General Language Understanding Benchmark 

**Authors**: Marek Šuppa, Andrej Ridzik, Daniel Hládek, Tomáš Javůrek, Viktória Ondrejová, Kristína Sásiková, Martin Tamajka, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2506.21508)  

**Abstract**: In this work, we introduce skLEP, the first comprehensive benchmark specifically designed for evaluating Slovak natural language understanding (NLU) models. We have compiled skLEP to encompass nine diverse tasks that span token-level, sentence-pair, and document-level challenges, thereby offering a thorough assessment of model capabilities. To create this benchmark, we curated new, original datasets tailored for Slovak and meticulously translated established English NLU resources. Within this paper, we also present the first systematic and extensive evaluation of a wide array of Slovak-specific, multilingual, and English pre-trained language models using the skLEP tasks. Finally, we also release the complete benchmark data, an open-source toolkit facilitating both fine-tuning and evaluation of models, and a public leaderboard at this https URL in the hopes of fostering reproducibility and drive future research in Slovak NLU. 

---
# Enhancing User Engagement in Socially-Driven Dialogue through Interactive LLM Alignments 

**Authors**: Jiashuo Wang, Kaitao Song, Chunpu Xu, Changhe Song, Yang Xiao, Dongsheng Li, Lili Qiu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21497)  

**Abstract**: Enhancing user engagement through interactions plays an essential role in socially-driven dialogues. While prior works have optimized models to reason over relevant knowledge or plan a dialogue act flow, the relationship between user engagement and knowledge or dialogue acts is subtle and does not guarantee user engagement in socially-driven dialogues. To this end, we enable interactive LLMs to learn user engagement by leveraging signals from the future development of conversations. Specifically, we adopt a more direct and relevant indicator of user engagement, i.e., the user's reaction related to dialogue intention after the interaction, as a reward to align interactive LLMs. To achieve this, we develop a user simulator to interact with target interactive LLMs and explore interactions between the user and the interactive LLM system via \textit{i$\times$MCTS} (\textit{M}onte \textit{C}arlo \textit{T}ree \textit{S}earch for \textit{i}nteraction). In this way, we collect a dataset containing pairs of higher and lower-quality experiences using \textit{i$\times$MCTS}, and align interactive LLMs for high-level user engagement by direct preference optimization (DPO) accordingly. Experiments conducted on two socially-driven dialogue scenarios (emotional support conversations and persuasion for good) demonstrate that our method effectively enhances user engagement in interactive LLMs. 

---
# Bridging Offline and Online Reinforcement Learning for LLMs 

**Authors**: Jack Lanchantin, Angelica Chen, Janice Lan, Xian Li, Swarnadeep Saha, Tianlu Wang, Jing Xu, Ping Yu, Weizhe Yuan, Jason E Weston, Sainbayar Sukhbaatar, Ilia Kulikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21495)  

**Abstract**: We investigate the effectiveness of reinforcement learning methods for finetuning large language models when transitioning from offline to semi-online to fully online regimes for both verifiable and non-verifiable tasks. Our experiments cover training on verifiable math as well as non-verifiable instruction following with a set of benchmark evaluations for both. Across these settings, we extensively compare online and semi-online Direct Preference Optimization and Group Reward Policy Optimization objectives, and surprisingly find similar performance and convergence between these variants, which all strongly outperform offline methods. We provide a detailed analysis of the training dynamics and hyperparameter selection strategies to achieve optimal results. Finally, we show that multi-tasking with verifiable and non-verifiable rewards jointly yields improved performance across both task types. 

---
# TopK Language Models 

**Authors**: Ryosuke Takahashi, Tatsuro Inaba, Kentaro Inui, Benjamin Heinzerling  

**Link**: [PDF](https://arxiv.org/pdf/2506.21468)  

**Abstract**: Sparse autoencoders (SAEs) have become an important tool for analyzing and interpreting the activation space of transformer-based language models (LMs). However, SAEs suffer several shortcomings that diminish their utility and internal validity. Since SAEs are trained post-hoc, it is unclear if the failure to discover a particular concept is a failure on the SAE's side or due to the underlying LM not representing this concept. This problem is exacerbated by training conditions and architecture choices affecting which features an SAE learns. When tracing how LMs learn concepts during training, the lack of feature stability also makes it difficult to compare SAEs features across different checkpoints. To address these limitations, we introduce a modification to the transformer architecture that incorporates a TopK activation function at chosen layers, making the model's hidden states equivalent to the latent features of a TopK SAE. This approach eliminates the need for post-hoc training while providing interpretability comparable to SAEs. The resulting TopK LMs offer a favorable trade-off between model size, computational efficiency, and interpretability. Despite this simple architectural change, TopK LMs maintain their original capabilities while providing robust interpretability benefits. Our experiments demonstrate that the sparse representations learned by TopK LMs enable successful steering through targeted neuron interventions and facilitate detailed analysis of neuron formation processes across checkpoints and layers. These features make TopK LMs stable and reliable tools for understanding how language models learn and represent concepts, which we believe will significantly advance future research on model interpretability and controllability. 

---
# Aligning Spoken Dialogue Models from User Interactions 

**Authors**: Anne Wu, Laurent Mazaré, Neil Zeghidour, Alexandre Défossez  

**Link**: [PDF](https://arxiv.org/pdf/2506.21463)  

**Abstract**: We propose a novel preference alignment framework for improving spoken dialogue models on real-time conversations from user interactions. Current preference learning methods primarily focus on text-based language models, and are not directly suited to the complexities of real-time speech interactions, with richer dynamics (e.g. interruption, interjection) and no explicit segmentation between speaker this http URL create a large-scale dataset of more than 150,000 preference pairs from raw multi-turn speech conversations, annotated with AI feedback, to cover preferences over both linguistic content and temporal context variations. We leverage offline alignment methods to finetune a full-duplex autoregressive speech-to-speech model. Extensive experiments demonstrate that feedback on generic conversations can be consistently effective in improving spoken dialogue models to produce more factual, safer and more contextually aligned interactions. We deploy the finetuned model and conduct holistic human evaluations to assess the impact beyond single-turn conversations. Our findings shed light on the importance of a well-calibrated balance among various dynamics, crucial for natural real-time speech dialogue systems. 

---
# Text2Cypher Across Languages: Evaluating Foundational Models Beyond English 

**Authors**: Makbule Gulcin Ozsoy, William Tai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21445)  

**Abstract**: Recent advances in large language models have enabled natural language interfaces that translate user questions into database queries, such as Text2SQL, Text2SPARQL, and Text2Cypher. While these interfaces enhance database accessibility, most research today focuses solely on English, with limited evaluation in other languages. This paper investigates the performance of foundational LLMs on the Text2Cypher task across multiple languages. We create and release a multilingual test set by translating English questions into Spanish and Turkish while preserving the original Cypher queries, enabling fair cross-lingual comparison. We evaluate multiple foundational models using standardized prompts and metrics. Our results show a consistent performance pattern: highest on English, then Spanish, and lowest on Turkish. We attribute this to differences in training data availability and linguistic characteristics. Additionally, we explore the impact of translating task prompts into Spanish and Turkish. Results show little to no change in evaluation metrics, suggesting prompt translation has minor impact. Our findings highlight the need for more inclusive evaluation and development in multilingual query generation. Future work includes schema localization and fine-tuning across diverse languages. 

---
# Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection 

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21443)  

**Abstract**: Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications. 

---
# Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation 

**Authors**: Guanting Dong, Xiaoxi Li, Yuyao Zhang, Mengjie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21384)  

**Abstract**: Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries. 

---
# Structuralist Approach to AI Literary Criticism: Leveraging Greimas Semiotic Square for Large Language Models 

**Authors**: Fangzhou Dong, Yifan Zeng, Yingpeng Sang, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21360)  

**Abstract**: Large Language Models (LLMs) excel in understanding and generating text but struggle with providing professional literary criticism for works with profound thoughts and complex narratives. This paper proposes GLASS (Greimas Literary Analysis via Semiotic Square), a structured analytical framework based on Greimas Semiotic Square (GSS), to enhance LLMs' ability to conduct in-depth literary analysis. GLASS facilitates the rapid dissection of narrative structures and deep meanings in narrative works. We propose the first dataset for GSS-based literary criticism, featuring detailed analyses of 48 works. Then we propose quantitative metrics for GSS-based literary criticism using the LLM-as-a-judge paradigm. Our framework's results, compared with expert criticism across multiple works and LLMs, show high performance. Finally, we applied GLASS to 39 classic works, producing original and high-quality analyses that address existing research gaps. This research provides an AI-based tool for literary research and education, offering insights into the cognitive mechanisms underlying literary engagement. 

---
# Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models 

**Authors**: Bram Willemsen, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2506.21294)  

**Abstract**: In this paper, we explore the use of a text-only, autoregressive language modeling approach for the extraction of referring expressions from visually grounded dialogue. More specifically, the aim is to investigate the extent to which the linguistic context alone can inform the detection of mentions that have a (visually perceivable) referent in the visual context of the conversation. To this end, we adapt a pretrained large language model (LLM) to perform a relatively course-grained annotation of mention spans in unfolding conversations by demarcating mention span boundaries in text via next-token prediction. Our findings indicate that even when using a moderately sized LLM, relatively small datasets, and parameter-efficient fine-tuning, a text-only approach can be effective, highlighting the relative importance of the linguistic context for this task. Nevertheless, we argue that the task represents an inherently multimodal problem and discuss limitations fundamental to unimodal approaches. 

---
# Small Encoders Can Rival Large Decoders in Detecting Groundedness 

**Authors**: Istabrak Abbes, Gabriele Prato, Quentin Fournier, Fernando Rodriguez, Alaa Boukhary, Adam Elwood, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21288)  

**Abstract**: Augmenting large language models (LLMs) with external context significantly improves their performance in natural language processing (NLP) tasks. However, LLMs struggle to answer queries reliably when the provided context lacks information, often resorting to ungrounded speculation or internal knowledge. Groundedness - generating responses strictly supported by the context - is essential for ensuring factual consistency and trustworthiness. This study focuses on detecting whether a given query is grounded in a document provided in context before the costly answer generation by LLMs. Such a detection mechanism can significantly reduce both inference time and resource consumption. We show that lightweight, task specific encoder models such as RoBERTa and NomicBERT, fine-tuned on curated datasets, can achieve accuracy comparable to state-of-the-art LLMs, such as Llama3 8B and GPT4o, in groundedness detection while reducing inference latency by orders of magnitude. The code is available at : this https URL 

---
# Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning 

**Authors**: Xin Xu, Tianhao Chen, Fan Zhang, Wanlong Liu, Pengxiang Li, Ajay Kumar Jaiswal, Yuchen Yan, Jishan Hu, Yang Wang, Hao Chen, Shiwei Liu, Shizhe Diao, Can Yang, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21285)  

**Abstract**: While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. 

---
# Cat and Mouse -- Can Fake Text Generation Outpace Detector Systems? 

**Authors**: Andrea McGlinchey, Peter J Barclay  

**Link**: [PDF](https://arxiv.org/pdf/2506.21274)  

**Abstract**: Large language models can produce convincing "fake text" in domains such as academic writing, product reviews, and political news. Many approaches have been investigated for the detection of artificially generated text. While this may seem to presage an endless "arms race", we note that newer LLMs use ever more parameters, training data, and energy, while relatively simple classifiers demonstrate a good level of detection accuracy with modest resources. To approach the question of whether the models' ability to beat the detectors may therefore reach a plateau, we examine the ability of statistical classifiers to identify "fake text" in the style of classical detective fiction. Over a 0.5 version increase, we found that Gemini showed an increased ability to generate deceptive text, while GPT did not. This suggests that reliable detection of fake text may remain feasible even for ever-larger models, though new model architectures may improve their deceptiveness 

---
# Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents 

**Authors**: Tianyi Men, Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21252)  

**Abstract**: As Multimodal Large Language Models (MLLMs) advance, multimodal agents show promise in real-world tasks like web navigation and embodied intelligence. However, due to limitations in a lack of external feedback, these agents struggle with self-correction and generalization. A promising approach is to use reward models as external feedback, but there is no clear on how to select reward models for agents. Thus, there is an urgent need to build a reward bench targeted at agents. To address these challenges, we propose Agent-RewardBench, a benchmark designed to evaluate reward modeling ability in MLLMs. The benchmark is characterized by three key features: (1) Multiple dimensions and real-world agent scenarios evaluation. It covers perception, planning, and safety with 7 scenarios; (2) Step-level reward evaluation. It allows for the assessment of agent capabilities at the individual steps of a task, providing a more granular view of performance during the planning process; and (3) Appropriately difficulty and high-quality. We carefully sample from 10 diverse models, difficulty control to maintain task challenges, and manual verification to ensure the integrity of the data. Experiments demonstrate that even state-of-the-art multimodal models show limited performance, highlighting the need for specialized training in agent reward modeling. Code is available at github. 

---
# Enhancing Automatic Term Extraction with Large Language Models via Syntactic Retrieval 

**Authors**: Yongchan Chun, Minhyuk Kim, Dongjun Kim, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21222)  

**Abstract**: Automatic Term Extraction (ATE) identifies domain-specific expressions that are crucial for downstream tasks such as machine translation and information retrieval. Although large language models (LLMs) have significantly advanced various NLP tasks, their potential for ATE has scarcely been examined. We propose a retrieval-based prompting strategy that, in the few-shot setting, selects demonstrations according to \emph{syntactic} rather than semantic similarity. This syntactic retrieval method is domain-agnostic and provides more reliable guidance for capturing term boundaries. We evaluate the approach in both in-domain and cross-domain settings, analyzing how lexical overlap between the query sentence and its retrieved examples affects performance. Experiments on three specialized ATE benchmarks show that syntactic retrieval improves F1-score. These findings highlight the importance of syntactic cues when adapting LLMs to terminology-extraction tasks. 

---
# Prompt-Guided Turn-Taking Prediction 

**Authors**: Koji Inoue, Mikey Elmers, Yahui Fu, Zi Haur Pang, Divesh Lala, Keiko Ochi, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2506.21191)  

**Abstract**: Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts. 

---
# Maintaining MTEB: Towards Long Term Usability and Reproducibility of Embedding Benchmarks 

**Authors**: Isaac Chung, Imene Kerboua, Marton Kardos, Roman Solomatin, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21182)  

**Abstract**: The Massive Text Embedding Benchmark (MTEB) has become a standard evaluation platform for text embedding models. While previous work has established the core benchmark methodology, this paper focuses on the engineering aspects that ensure MTEB's continued reproducibility and extensibility. We present our approach to maintaining robust continuous integration pipelines that validate dataset integrity, automate test execution, and assess benchmark results' generalizability. We detail the design choices that collectively enhance reproducibility and usability. Furthermore, we discuss our strategies for handling community contributions and extending the benchmark with new tasks and datasets. These engineering practices have been instrumental in scaling MTEB to become more comprehensive while maintaining quality and, ultimately, relevance to the field. Our experiences offer valuable insights for benchmark maintainers facing similar challenges in ensuring reproducibility and usability in machine learning evaluation frameworks. The MTEB repository is available at: this https URL 

---
# Compressed and Smooth Latent Space for Text Diffusion Modeling 

**Authors**: Viacheslav Meshchaninov, Egor Chimbulatov, Alexander Shabalin, Aleksandr Abramov, Dmitry Vetrov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21170)  

**Abstract**: Autoregressive language models dominate modern text generation, yet their sequential nature introduces fundamental limitations: decoding is slow, and maintaining global coherence remains challenging. Diffusion models offer a promising alternative by enabling parallel generation and flexible control; however, their application to text generation is hindered by the high dimensionality of token-level representations. We introduce Cosmos, a novel approach to text generation that operates entirely in a compressed, smooth latent space tailored specifically for diffusion. This space is learned using an autoencoder trained simultaneously for token-level reconstruction and alignment with frozen activations from a pretrained language encoder, providing robust semantic grounding and enabling effective perturbation-based augmentations. Empirically, we demonstrate that text representations can be compressed by $8\times$ while maintaining generation quality comparable to token-level diffusion models. Furthermore, increasing the latent sequence length allows Cosmos to surpass both diffusion-based and autoregressive baselines. We evaluate Cosmos on four diverse generative tasks including story generation, question generation, summarization, and detoxification and compare it with various generative paradigms. Cosmos achieves comparable or superior generation quality while offering more than $2\times$ faster inference. 

---
# Progtuning: Progressive Fine-tuning Framework for Transformer-based Language Models 

**Authors**: Xiaoshuang Ji, Zhendong Zhao, Xiaojun Chen, Xin Zhao, Zeyao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21119)  

**Abstract**: Fine-tuning is a promising technique for leveraging Transformer-based language models in downstream tasks. As model sizes continue to grow, updating all model parameters becomes increasingly costly. Parameter-efficient fine-tuning methods effectively address this issue by selectively updating a small subset of parameters. However, fine-tuning and most existing parameter-efficient fine-tuning methods require updating the same number of parameters as the initial size, ignoring the unequal contribution across Transformer blocks and leading to extremely inefficient allocation of computing resources. In this paper, we propose Progtuning, the novel fine-tuning framework combined with progressive learning for Transformer-based language models. Specifically, Progtuning progressively reduces the number of updated transformer blocks based on the contribution. Remarkably, Progtuning optimizes resource allocation and reduces the number of updated parameters by approximately 25\%, while still maintaining competitive performance. And it also exhibits high adaptability with parameter-efficient fine-tuning methods, demonstrating excellent performance across various adaptation scenarios. 

---
# ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry 

**Authors**: Qinwen Chen, Wenbiao Tao, Zhiwei Zhu, Mingfan Xi, Liangzhong Guo, Yuan Wang, Wei Wang, Yunshi Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21098)  

**Abstract**: Community Question Answering (CQA) platforms can be deemed as important knowledge bases in community, but effectively leveraging historical interactions and domain knowledge in real-time remains a challenge. Existing methods often underutilize external knowledge, fail to incorporate dynamic historical QA context, or lack memory mechanisms suited for industrial deployment. We propose ComRAG, a retrieval-augmented generation framework for real-time industrial CQA that integrates static knowledge with dynamic historical QA pairs via a centroid-based memory mechanism designed for retrieval, generation, and efficient storage. Evaluated on three industrial CQA datasets, ComRAG consistently outperforms all baselines--achieving up to 25.9% improvement in vector similarity, reducing latency by 8.7% to 23.3%, and lowering chunk growth from 20.23% to 2.06% over iterations. 

---
# DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning 

**Authors**: Kang He, Yuzhe Ding. Haining Wang, Fei Li, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.21096)  

**Abstract**: Previous multimodal sentence representation learning methods have achieved impressive performance. However, most approaches focus on aligning images and text at a coarse level, facing two critical challenges:cross-modal misalignment bias and intra-modal semantic divergence, which significantly degrade sentence representation quality. To address these challenges, we propose DALR (Dual-level Alignment Learning for Multimodal Sentence Representation). For cross-modal alignment, we propose a consistency learning module that softens negative samples and utilizes semantic similarity from an auxiliary task to achieve fine-grained cross-modal alignment. Additionally, we contend that sentence relationships go beyond binary positive-negative labels, exhibiting a more intricate ranking structure. To better capture these relationships and enhance representation quality, we integrate ranking distillation with global intra-modal alignment learning. Comprehensive experiments on semantic textual similarity (STS) and transfer (TR) tasks validate the effectiveness of our approach, consistently demonstrating its superiority over state-of-the-art baselines. 

---
# MT2-CSD: A New Dataset and Multi-Semantic Knowledge Fusion Method for Conversational Stance Detection 

**Authors**: Fuqiang Niu, Genan Dai, Yisha Lu, Jiayu Liao, Xiang Li, Hu Huang, Bowen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21053)  

**Abstract**: In the realm of contemporary social media, automatic stance detection is pivotal for opinion mining, as it synthesizes and examines user perspectives on contentious topics to uncover prevailing trends and sentiments. Traditional stance detection research often targets individual instances, thereby limiting its capacity to model multi-party discussions typical in real social media scenarios. This shortcoming largely stems from the scarcity of datasets that authentically capture the dynamics of social media interactions, hindering advancements in conversational stance detection. In this paper, we introduce MT2-CSD, a comprehensive dataset for multi-target, multi-turn conversational stance detection. To the best of our knowledge, MT2-CSD is the largest dataset available for this purpose, comprising 24,457 annotated instances and exhibiting the greatest conversational depth, thereby presenting new challenges for stance detection. To address these challenges, we propose the Large Language model enhanced Conversational Relational Attention Network (LLM-CRAN), which exploits the reasoning capabilities of LLMs to improve conversational understanding. We conduct extensive experiments to evaluate the efficacy of LLM-CRAN on the MT2-CSD dataset. The experimental results indicate that LLM-CRAN significantly outperforms strong baseline models in the task of conversational stance detection. 

---
# A Semi-supervised Scalable Unified Framework for E-commerce Query Classification 

**Authors**: Chunyuan Yuan, Chong Zhang, Zheng Fang, Ming Pang, Xue Jiang, Changping Peng, Zhangang Lin, Ching Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.21049)  

**Abstract**: Query classification, including multiple subtasks such as intent and category prediction, is vital to e-commerce applications. E-commerce queries are usually short and lack context, and the information between labels cannot be used, resulting in insufficient prior information for modeling. Most existing industrial query classification methods rely on users' posterior click behavior to construct training samples, resulting in a Matthew vicious cycle. Furthermore, the subtasks of query classification lack a unified framework, leading to low efficiency for algorithm optimization.
In this paper, we propose a novel Semi-supervised Scalable Unified Framework (SSUF), containing multiple enhanced modules to unify the query classification tasks. The knowledge-enhanced module uses world knowledge to enhance query representations and solve the problem of insufficient query information. The label-enhanced module uses label semantics and semi-supervised signals to reduce the dependence on posterior labels. The structure-enhanced module enhances the label representation based on the complex label relations. Each module is highly pluggable, and input features can be added or removed as needed according to each subtask. We conduct extensive offline and online A/B experiments, and the results show that SSUF significantly outperforms the state-of-the-art models. 

---
# Large Language Models Acing Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Mohammad Adnan, Sakshi Deo, Ali Imam Abidi, Keshav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21031)  

**Abstract**: Advanced intelligent systems, particularly Large Language Models (LLMs), are significantly reshaping financial practices through advancements in Natural Language Processing (NLP). However, the extent to which these models effectively capture and apply domain-specific financial knowledge remains uncertain. Addressing a critical gap in the expansive Indian financial context, this paper introduces CA-Ben, a Chartered Accountancy benchmark specifically designed to evaluate the financial, legal, and quantitative reasoning capabilities of LLMs. CA-Ben comprises structured question-answer datasets derived from the rigorous examinations conducted by the Institute of Chartered Accountants of India (ICAI), spanning foundational, intermediate, and advanced CA curriculum stages. Six prominent LLMs i.e. GPT 4o, LLAMA 3.3 70B, LLAMA 3.1 405B, MISTRAL Large, Claude 3.5 Sonnet, and Microsoft Phi 4 were evaluated using standardized protocols. Results indicate variations in performance, with Claude 3.5 Sonnet and GPT-4o outperforming others, especially in conceptual and legal reasoning. Notable challenges emerged in numerical computations and legal interpretations. The findings emphasize the strengths and limitations of current LLMs, suggesting future improvements through hybrid reasoning and retrieval-augmented generation methods, particularly for quantitative analysis and accurate legal interpretation. 

---
# SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control 

**Authors**: Adithya Chittem, Aishna Shrivastava, Sai Tarun Pendela, Jagat Sesh Challa, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20993)  

**Abstract**: Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines. 

---
# Can Gradient Descent Simulate Prompting? 

**Authors**: Eric Zhang, Leshem Choshen, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20989)  

**Abstract**: There are two primary ways of incorporating new information into a language model (LM): changing its prompt or changing its parameters, e.g. via fine-tuning. Parameter updates incur no long-term storage cost for model changes. However, for many model updates, prompting is significantly more effective: prompted models can generalize robustly from single examples and draw logical inferences that do not occur under standard fine-tuning. Can models be modified so that fine-tuning does emulate prompting? This paper describes a method for meta-training LMs such that gradient updates emulate the effects of conditioning on new information. Our approach uses tools from gradient-based meta-learning but uses an LM's own prompted predictions as targets, eliminating the need for ground-truth labels. Subsequent gradient descent training recovers some (and occasionally all) of prompted model performance -- showing improvement on the ``reversal curse'' tasks, and answering questions about text passages after a single gradient update. These results suggest that, with appropriate initialization, gradient descent can be surprisingly expressive. Our results suggest new avenues for long-context modeling and offer insight into the generalization capabilities of gradient-based learning. 

---
# KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model 

**Authors**: Xinping Zhao, Xinshuo Hu, Zifei Shan, Shouzheng Huang, Yao Zhou, Zetian Sun, Zhenyu Liu, Dongfang Li, Xinyuan Wei, Qian Chen, Youcheng Pan, Yang Xiang, Meishan Zhang, Haofen Wang, Jun Yu, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20923)  

**Abstract**: In this paper, we propose KaLM-Embedding-V2, a versatile and compact embedding model, which achieves impressive performance in general-purpose text embedding tasks by leveraging superior training techniques and data. Our key innovations include: (1) To better align the architecture with representation learning, we remove the causal attention mask and adopt a fully bidirectional transformer with simple yet effective mean-pooling to produce fixed-length embeddings; (2) We employ a multi-stage training pipeline: (i) pre-training on large-scale weakly supervised open-source corpora; (ii) fine-tuning on high-quality retrieval and non-retrieval datasets; and (iii) model-soup parameter averaging for robust generalization. Besides, we introduce a focal-style reweighting mechanism that concentrates learning on difficult samples and an online hard-negative mixing strategy to continuously enrich hard negatives without expensive offline mining; (3) We collect over 20 categories of data for pre-training and 100 categories of data for fine-tuning, to boost both the performance and generalization of the embedding model. Extensive evaluations on the Massive Text Embedding Benchmark (MTEB) Chinese and English show that our model significantly outperforms others of comparable size, and competes with 3x, 14x, 18x, and 26x larger embedding models, setting a new standard for a versatile and compact embedding model with less than 1B parameters. 

---
# FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language 

**Authors**: Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2506.20920)  

**Abstract**: Pre-training state-of-the-art large language models (LLMs) requires vast amounts of clean and diverse text data. While the open development of large high-quality English pre-training datasets has seen substantial recent progress, training performant multilingual LLMs remains a challenge, in large part due to the inherent difficulty of tailoring filtering and deduplication pipelines to a large number of languages. In this work, we introduce a new pre-training dataset curation pipeline based on FineWeb that can be automatically adapted to support any language. We extensively ablate our pipeline design choices on a set of nine diverse languages, guided by a set of meaningful and informative evaluation tasks that were chosen through a novel selection process based on measurable criteria. Ultimately, we show that our pipeline can be used to create non-English corpora that produce more performant models than prior datasets. We additionally introduce a straightforward and principled approach to rebalance datasets that takes into consideration both duplication count and quality, providing an additional performance uplift. Finally, we scale our pipeline to over 1000 languages using almost 100 Common Crawl snapshots to produce FineWeb2, a new 20 terabyte (5 billion document) multilingual dataset which we release along with our pipeline, training, and evaluation codebases. 

---
# Optimising Language Models for Downstream Tasks: A Post-Training Perspective 

**Authors**: Zhengyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20917)  

**Abstract**: Language models (LMs) have demonstrated remarkable capabilities in NLP, yet adapting them efficiently and robustly to specific tasks remains challenging. As their scale and complexity grow, fine-tuning LMs on labelled data often underutilizes available unlabelled data, leads to overfitting on small task-specific sets, and imposes significant computational costs. These limitations hamper their application to the open-ended landscape of real-world language tasks.
This thesis proposes a series of methods to better adapt LMs to downstream applications. First, we explore strategies for extracting task-relevant knowledge from unlabelled data, introducing a novel continued pre-training technique that outperforms state-of-the-art semi-supervised approaches. Next, we present a parameter-efficient fine-tuning method that substantially reduces memory and compute costs while maintaining competitive performance. We also introduce improved supervised fine-tuning methods that enable LMs to better follow instructions, especially when labelled data is scarce, enhancing their performance across a range of NLP tasks, including open-ended generation. Finally, we develop new evaluation methods and benchmarks, such as multi-hop spatial reasoning tasks, to assess LM capabilities and adaptation more comprehensively.
Through extensive empirical studies across diverse NLP tasks, our results demonstrate that these approaches substantially improve LM robustness, efficiency, and generalization, making them more adaptable to a broad range of applications. These advances mark a significant step towards more robust and efficient LMs, bringing us closer to the goal of artificial general intelligence. 

---
# Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine 

**Authors**: Sebastian Joseph, Lily Chen, Barry Wei, Michael Mackert, Iain J. Marshall, Paul Pu Liang, Ramez Kouzy, Byron C. Wallace, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.20876)  

**Abstract**: Technological progress has led to concrete advancements in tasks that were regarded as challenging, such as automatic fact-checking. Interest in adopting these systems for public health and medicine has grown due to the high-stakes nature of medical decisions and challenges in critically appraising a vast and diverse medical literature. Evidence-based medicine connects to every individual, and yet the nature of it is highly technical, rendering the medical literacy of majority users inadequate to sufficiently navigate the domain. Such problems with medical communication ripens the ground for end-to-end fact-checking agents: check a claim against current medical literature and return with an evidence-backed verdict. And yet, such systems remain largely unused. To understand this, we present the first study examining how clinical experts verify real claims from social media by synthesizing medical evidence. In searching for this upper-bound, we reveal fundamental challenges in end-to-end fact-checking when applied to medicine: Difficulties connecting claims in the wild to scientific evidence in the form of clinical trials; ambiguities in underspecified claims mixed with mismatched intentions; and inherently subjective veracity labels. We argue that fact-checking should be approached and evaluated as an interactive communication problem, rather than an end-to-end process. 

---
# Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes 

**Authors**: Quintin Myers, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20822)  

**Abstract**: Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology. 

---
# MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering 

**Authors**: Chinmay Gondhalekar, Urjitkumar Patel, Fang-Chun Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.20821)  

**Abstract**: Financial documents--such as 10-Ks, 10-Qs, and investor presentations--span hundreds of pages and combine diverse modalities, including dense narrative text, structured tables, and complex figures. Answering questions over such content often requires joint reasoning across modalities, which strains traditional large language models (LLMs) and retrieval-augmented generation (RAG) pipelines due to token limitations, layout loss, and fragmented cross-modal context. We introduce MultiFinRAG, a retrieval-augmented generation framework purpose-built for financial QA. MultiFinRAG first performs multimodal extraction by grouping table and figure images into batches and sending them to a lightweight, quantized open-source multimodal LLM, which produces both structured JSON outputs and concise textual summaries. These outputs, along with narrative text, are embedded and indexed with modality-aware similarity thresholds for precise retrieval. A tiered fallback strategy then dynamically escalates from text-only to text+table+image contexts when necessary, enabling cross-modal reasoning while reducing irrelevant context. Despite running on commodity hardware, MultiFinRAG achieves 19 percentage points higher accuracy than ChatGPT-4o (free-tier) on complex financial QA tasks involving text, tables, images, and combined multimodal reasoning. 

---
# The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas 

**Authors**: Chenglei Si, Tatsunori Hashimoto, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20803)  

**Abstract**: Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes. 

---
# Multi-lingual Functional Evaluation for Large Language Models 

**Authors**: Victor Ojewale, Inioluwa Deborah Raji, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.20793)  

**Abstract**: Multi-lingual competence in large language models is often evaluated via static data benchmarks such as Belebele, M-MMLU and M-GSM. However, these evaluations often fail to provide an adequate understanding of the practical performance and robustness of models across multi-lingual settings. In response, we create multi-lingual functional benchmarks -- Cross-Lingual Grade School Math Symbolic (CL-GSM Symbolic) and Cross-Lingual Instruction-Following Eval (CL-IFEval)-- by translating existing functional benchmark templates from English to five additional languages that span the range of resources available for NLP: French, Spanish, Hindi, Arabic and Yoruba. Our results reveal that some static multi-lingual benchmarks capture functional performance much more closely than others (i.e. across models, there is a 24%, 17% and 18% decrease in performance between M-GSM and CL-GSM Symbolic in English, French and Spanish respectively; similarly there's a 15 - 24% performance drop across languages between Belebele and CL-IFEval, and only a 0.5% to 3% performance drop between M-MMLU and CL-IFEval). Similarly, we find that model robustness across languages varies significantly, with certain languages (eg. Arabic, English) being the most consistently well performing across evaluation iterations. 

---
# Towards Probabilistic Question Answering Over Tabular Data 

**Authors**: Chen Shen, Sajjadur Rahman, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2506.20747)  

**Abstract**: Current approaches for question answering (QA) over tabular data, such as NL2SQL systems, perform well for factual questions where answers are directly retrieved from tables. However, they fall short on probabilistic questions requiring reasoning under uncertainty. In this paper, we introduce a new benchmark LUCARIO and a framework for probabilistic QA over large tabular data. Our method induces Bayesian Networks from tables, translates natural language queries into probabilistic queries, and uses large language models (LLMs) to generate final answers. Empirical results demonstrate significant improvements over baselines, highlighting the benefits of hybrid symbolic-neural reasoning. 

---
# HalluSegBench: Counterfactual Visual Reasoning for Segmentation Hallucination Evaluation 

**Authors**: Xinzhuo Li, Adheesh Juvekar, Xingyou Liu, Muntasir Wahed, Kiet A. Nguyen, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21546)  

**Abstract**: Recent progress in vision-language segmentation has significantly advanced grounded visual understanding. However, these models often exhibit hallucinations by producing segmentation masks for objects not grounded in the image content or by incorrectly labeling irrelevant regions. Existing evaluation protocols for segmentation hallucination primarily focus on label or textual hallucinations without manipulating the visual context, limiting their capacity to diagnose critical failures. In response, we introduce HalluSegBench, the first benchmark specifically designed to evaluate hallucinations in visual grounding through the lens of counterfactual visual reasoning. Our benchmark consists of a novel dataset of 1340 counterfactual instance pairs spanning 281 unique object classes, and a set of newly introduced metrics that quantify hallucination sensitivity under visually coherent scene edits. Experiments on HalluSegBench with state-of-the-art vision-language segmentation models reveal that vision-driven hallucinations are significantly more prevalent than label-driven ones, with models often persisting in false segmentation, highlighting the need for counterfactual reasoning to diagnose grounding fidelity. 

---
# Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge 

**Authors**: Boyu Gou, Zanming Huang, Yuting Ning, Yu Gu, Michael Lin, Weijian Qi, Andrei Kopanev, Botao Yu, Bernal Jiménez Gutiérrez, Yiheng Shu, Chan Hee Song, Jiaman Wu, Shijie Chen, Hanane Nour Moussa, Tianshu Zhang, Jian Xie, Yifei Li, Tianci Xue, Zeyi Liao, Kai Zhang, Boyuan Zheng, Zhaowei Cai, Viktor Rozgic, Morteza Ziyadi, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.21506)  

**Abstract**: Agentic search such as Deep Research systems, where large language models autonomously browse the web, synthesize information, and return comprehensive citation-backed answers, represents a major shift in how users interact with web-scale information. While promising greater efficiency and cognitive offloading, the growing complexity and open-endedness of agentic search have outpaced existing evaluation benchmarks and methodologies, which largely assume short search horizons and static answers. In this paper, we introduce Mind2Web 2, a benchmark of 130 realistic, high-quality, and long-horizon tasks that require real-time web browsing and extensive information synthesis, constructed with over 1,000 hours of human labor. To address the challenge of evaluating time-varying and complex answers, we propose a novel Agent-as-a-Judge framework. Our method constructs task-specific judge agents based on a tree-structured rubric design to automatically assess both answer correctness and source attribution. We conduct a comprehensive evaluation of nine frontier agentic search systems and human performance, along with a detailed error analysis to draw insights for future development. The best-performing system, OpenAI Deep Research, can already achieve 50-70% of human performance while spending half the time, showing a great potential. Altogether, Mind2Web 2 provides a rigorous foundation for developing and benchmarking the next generation of agentic search systems. 

---
# Logios : An open source Greek Polytonic Optical Character Recognition system 

**Authors**: Perifanos Konstantinos, Goutsos Dionisis  

**Link**: [PDF](https://arxiv.org/pdf/2506.21474)  

**Abstract**: In this paper, we present an Optical Character Recognition (OCR) system specifically designed for the accurate recognition and digitization of Greek polytonic texts. By leveraging the combined strengths of convolutional layers for feature extraction and recurrent layers for sequence learning, our system addresses the unique challenges posed by Greek polytonic scripts. This approach aims to overcome the limitations of traditional OCR methods, offering significant improvements in accuracy and efficiency. We release the underlying model as an open-source library and make our OCR platform available for academic use. 

---
# Spatial Mental Modeling from Limited Views 

**Authors**: Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.21458)  

**Abstract**: Can Vision Language Models (VLMs) imagine the full scene from just a few views, like humans do? Humans form spatial mental models, internal representations of unseen space, to reason about layout, perspective, and motion. Our new MindCube benchmark with 21,154 questions across 3,268 images exposes this critical gap, where existing VLMs exhibit near-random performance. Using MindCube, we systematically evaluate how well VLMs build robust spatial mental models through representing positions (cognitive mapping), orientations (perspective-taking), and dynamics (mental simulation for "what-if" movements). We then explore three approaches to help VLMs approximate spatial mental models, including unseen intermediate views, natural language reasoning chains, and cognitive maps. The significant improvement comes from a synergistic approach, "map-then-reason", that jointly trains the model to first generate a cognitive map and then reason upon it. By training models to reason over these internal maps, we boosted accuracy from 37.8% to 60.8% (+23.0%). Adding reinforcement learning pushed performance even further to 70.7% (+32.9%). Our key insight is that such scaffolding of spatial mental models, actively constructing and utilizing internal structured spatial representations with flexible reasoning processes, significantly improves understanding of unobservable space. 

---
# Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference 

**Authors**: Colin Samplawski, Adam D. Cobb, Manoj Acharya, Ramneet Kaur, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.21408)  

**Abstract**: Despite their widespread use, large language models (LLMs) are known to hallucinate incorrect information and be poorly calibrated. This makes the uncertainty quantification of these models of critical importance, especially in high-stakes domains, such as autonomy and healthcare. Prior work has made Bayesian deep learning-based approaches to this problem more tractable by performing inference over the low-rank adaptation (LoRA) parameters of a fine-tuned model. While effective, these approaches struggle to scale to larger LLMs due to requiring further additional parameters compared to LoRA. In this work we present $\textbf{Scala}$ble $\textbf{B}$ayesian $\textbf{L}$ow-Rank Adaptation via Stochastic Variational Subspace Inference (ScalaBL). We perform Bayesian inference in an $r$-dimensional subspace, for LoRA rank $r$. By repurposing the LoRA parameters as projection matrices, we are able to map samples from this subspace into the full weight space of the LLM. This allows us to learn all the parameters of our approach using stochastic variational inference. Despite the low dimensionality of our subspace, we are able to achieve competitive performance with state-of-the-art approaches while only requiring ${\sim}1000$ additional parameters. Furthermore, it allows us to scale up to the largest Bayesian LLM to date, with four times as a many base parameters as prior work. 

---
# Hybrid Deep Learning and Signal Processing for Arabic Dialect Recognition in Low-Resource Settings 

**Authors**: Ghazal Al-Shwayyat, Omer Nezih Gerek  

**Link**: [PDF](https://arxiv.org/pdf/2506.21386)  

**Abstract**: Arabic dialect recognition presents a significant challenge in speech technology due to the linguistic diversity of Arabic and the scarcity of large annotated datasets, particularly for underrepresented dialects. This research investigates hybrid modeling strategies that integrate classical signal processing techniques with deep learning architectures to address this problem in low-resource scenarios. Two hybrid models were developed and evaluated: (1) Mel-Frequency Cepstral Coefficients (MFCC) combined with a Convolutional Neural Network (CNN), and (2) Discrete Wavelet Transform (DWT) features combined with a Recurrent Neural Network (RNN). The models were trained on a dialect-filtered subset of the Common Voice Arabic dataset, with dialect labels assigned based on speaker metadata. Experimental results demonstrate that the MFCC + CNN architecture achieved superior performance, with an accuracy of 91.2% and strong precision, recall, and F1-scores, significantly outperforming the Wavelet + RNN configuration, which achieved an accuracy of 66.5%. These findings highlight the effectiveness of leveraging spectral features with convolutional models for Arabic dialect recognition, especially when working with limited labeled data. The study also identifies limitations related to dataset size, potential regional overlaps in labeling, and model optimization, providing a roadmap for future research. Recommendations for further improvement include the adoption of larger annotated corpora, integration of self-supervised learning techniques, and exploration of advanced neural architectures such as Transformers. Overall, this research establishes a strong baseline for future developments in Arabic dialect recognition within resource-constrained environments. 

---
# Latent Prototype Routing: Achieving Near-Perfect Load Balancing in Mixture-of-Experts 

**Authors**: Jiajie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21328)  

**Abstract**: Mixture-of-Experts (MoE) architectures have emerged as a key strategy for scaling large language models (LLMs) efficiently. However, current MoE systems suffer from severe load imbalance, where only a small subset of experts is consistently activated during training and inference, leading to significant underutilization of model capacity and computational resources. In this work, we revisit expert routing through a clustering perspective and propose Latent Prototype Routing (LPR), a novel routing framework that generalizes existing approaches while promoting balanced expert utilization without compromising downstream performance. Extensive experiments across multiple open-source MoE models -- including DeepSeek-V3, Qwen3-MoE, and Mixtral -- demonstrate that LPR reduces the Gini coefficient of expert load from 0.70 to 0.035 on average, improves the min-max expert load ratio from 1e-6 to 0.70, achieving near-perfect load balancing. 

---
# Exploring Adapter Design Tradeoffs for Low Resource Music Generation 

**Authors**: Atharva Mehta, Shivam Chauhan, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.21298)  

**Abstract**: Fine-tuning large-scale music generation models, such as MusicGen and Mustango, is a computationally expensive process, often requiring updates to billions of parameters and, therefore, significant hardware resources. Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly adapter-based methods, have emerged as a promising alternative, enabling adaptation with minimal trainable parameters while preserving model performance. However, the design choices for adapters, including their architecture, placement, and size, are numerous, and it is unclear which of these combinations would produce optimal adapters and why, for a given case of low-resource music genre. In this paper, we attempt to answer this question by studying various adapter configurations for two AI music models, MusicGen and Mustango, on two genres: Hindustani Classical and Turkish Makam music.
Our findings reveal distinct trade-offs: convolution-based adapters excel in capturing fine-grained local musical details such as ornamentations and short melodic phrases, while transformer-based adapters better preserve long-range dependencies crucial for structured improvisation. Additionally, we analyze computational resource requirements across different adapter scales, demonstrating how mid-sized adapters (40M parameters) achieve an optimal balance between expressivity and quality. Furthermore, we find that Mustango, a diffusion-based model, generates more diverse outputs with better adherence to the description in the input prompt while lacking in providing stability in notes, rhythm alignment, and aesthetics. Also, it is computationally intensive and requires significantly more time to train. In contrast, autoregressive models like MusicGen offer faster training and are more efficient, and can produce better quality output in comparison, but have slightly higher redundancy in their generations. 

---
# HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context 

**Authors**: Qize Yang, Shimin Yao, Weixuan Chen, Shenghao Fu, Detao Bai, Jiaxing Zhao, Boyuan Sun, Bowen Yin, Xihan Wei, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21277)  

**Abstract**: With the rapid evolution of multimodal large language models, the capacity to deeply understand and interpret human intentions has emerged as a critical capability, which demands detailed and thoughtful reasoning. In recent studies, Reinforcement Learning (RL) has demonstrated potential in enhancing the reasoning capabilities of Large Language Models (LLMs). Nonetheless, the challenges associated with adapting RL to multimodal data and formats remain largely unaddressed. In this paper, we identify two issues in existing multimodal reasoning models: insufficient global context understanding and shortcut problems. Insufficient context understanding can happen when a model misinterprets multimodal context, resulting in incorrect answers. The shortcut problem occurs when the model overlooks crucial clues in multimodal inputs, directly addressing the query without considering the multimodal information. To tackle these issues, we emphasize the necessity for the model to reason with a clear understanding of the global context within multimodal inputs. This global context understanding can effectively prevent the model from overlooking key multimodal cues and ensure a thorough reasoning process. To ensure the accurate interpretation of multimodal context information, we implement a context reward judged by a large language model, alongside format and accuracy rewards. Additionally, to improve complex reasoning capability, we employ the LLM to assess the logical reward, determining whether the reasoning process successfully integrates multimodal information with logical methods. We also introduce a reasoning omni-modal benchmark, IntentBench, aimed at evaluating models in understanding complex human intentions and emotions. Our proposed method demonstrates advanced performance across multiple omni-modal benchmarks compared to other open-source omni-modal models. 

---
# DiLoCoX: A Low-Communication Large-Scale Training Framework for Decentralized Cluster 

**Authors**: Ji Qi, WenPeng Zhu, Li Li, Ming Wu, YingJun Wu, Wu He, Xun Gao, Jason Zeng, Michael Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.21263)  

**Abstract**: The distributed training of foundation models, particularly large language models (LLMs), demands a high level of communication. Consequently, it is highly dependent on a centralized cluster with fast and reliable interconnects. Can we conduct training on slow networks and thereby unleash the power of decentralized clusters when dealing with models exceeding 100 billion parameters? In this paper, we propose DiLoCoX, a low-communication large-scale decentralized cluster training framework. It combines Pipeline Parallelism with Dual Optimizer Policy, One-Step-Delay Overlap of Communication and Local Training, and an Adaptive Gradient Compression Scheme. This combination significantly improves the scale of parameters and the speed of model pre-training. We justify the benefits of one-step-delay overlap of communication and local training, as well as the adaptive gradient compression scheme, through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence. To the best of our knowledge, this is the first decentralized training framework successfully applied to models with over 100 billion parameters. 

---
# Complexity-aware fine-tuning 

**Authors**: Andrey Goncharov, Daniil Vyazhev, Petr Sychev, Edvard Khalafyan, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.21220)  

**Abstract**: General-purpose Large Language Models (LLMs) are frequently fine-tuned through supervised fine-tuning (SFT) to enhance performance in specific domains. Better results can be achieved by distilling the chain-of-thought of a larger model at the cost of numerous expensive calls and a much greater amount of data. We propose a novel blueprint for efficient fine-tuning that uses reasoning only for complex data identified by entropy. Specifically, across two small open models ($\approx 3B$) we split the training data into complexity categories by a single token answer entropy (ROC AUC $0.73$), fine-tune large language models (LLMs) via SFT and distillation, and show that our pipeline significantly outperforms the standard SFT approach ($0.55$ vs $0.43$ average accuracy) and provides comparable with distillation performance while using $62\%$ less data ($0.55$ average accuracy for both). We publish our code and data to facilitate further research in this direction. 

---
# Unveiling Causal Reasoning in Large Language Models: Reality or Mirage? 

**Authors**: Haoang Chi, He Li, Wenjing Yang, Feng Liu, Long Lan, Xiaoguang Ren, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.21215)  

**Abstract**: Causal reasoning capability is critical in advancing large language models (LLMs) toward strong artificial intelligence. While versatile LLMs appear to have demonstrated capabilities in understanding contextual causality and providing responses that obey the laws of causality, it remains unclear whether they perform genuine causal reasoning akin to humans. However, current evidence indicates the contrary. Specifically, LLMs are only capable of performing shallow (level-1) causal reasoning, primarily attributed to the causal knowledge embedded in their parameters, but they lack the capacity for genuine human-like (level-2) causal reasoning. To support this hypothesis, methodologically, we delve into the autoregression mechanism of transformer-based LLMs, revealing that it is not inherently causal. Empirically, we introduce a new causal Q&A benchmark called CausalProbe-2024, whose corpora are fresh and nearly unseen for the studied LLMs. The LLMs exhibit a significant performance drop on CausalProbe-2024 compared to earlier benchmarks, indicating the fact that they primarily engage in level-1 causal reasoning. To bridge the gap towards level-2 causal reasoning, we draw inspiration from the fact that human reasoning is usually facilitated by general knowledge and intended goals. We propose G^2-Reasoner, a method that incorporates general knowledge and goal-oriented prompts into LLMs' causal reasoning processes. Experiments demonstrate that G^2-Reasoner significantly enhances LLMs' causal reasoning capability, particularly in fresh and counterfactual contexts. This work sheds light on a new path for LLMs to advance towards genuine causal reasoning, going beyond level-1 and making strides towards level-2. 

---
# Learning to Skip the Middle Layers of Transformers 

**Authors**: Tim Lawson, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2506.21103)  

**Abstract**: Conditional computation is a popular strategy to make Transformers more efficient. Existing methods often target individual modules (e.g., mixture-of-experts layers) or skip layers independently of one another. However, interpretability research has demonstrated that the middle layers of Transformers exhibit greater redundancy, and that early layers aggregate information into token positions. Guided by these insights, we propose a novel architecture that dynamically skips a variable number of layers from the middle outward. In particular, a learned gating mechanism determines whether to bypass a symmetric span of central blocks based on the input, and a gated attention mechanism prevents subsequent tokens from attending to skipped token positions. Residual norms are controlled with a 'sandwich' or 'perilayernorm' scheme and gate sparsity with an adaptive regularization loss. We had aimed to reduce compute requirements for 'simpler' tokens and potentially foster an emergent multi-level representational hierarchy but, at the scales investigated, our approach does not achieve improvements in the trade-off between validation cross-entropy and estimated FLOPs compared to dense baselines with fewer layers. We release our code at this https URL. 

---
# Enhancing LLM Tool Use with High-quality Instruction Data from Knowledge Graph 

**Authors**: Jingwei Wang, Zai Zhang, Hao Qian, Chunjing Gan, Binbin Hu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Bin Shi, Bo Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21071)  

**Abstract**: Teaching large language models (LLMs) to use tools is crucial for improving their problem-solving abilities and expanding their applications. However, effectively using tools is challenging because it requires a deep understanding of tool functionalities and user intentions. Previous methods relied mainly on LLMs to generate instruction data, but the quality of these data was often insufficient. In this paper, we propose a new method that uses knowledge graphs to generate high-quality instruction data for LLMs. Knowledge graphs are manually curated datasets rich in semantic information. We begin by extracting various query pathways from a given knowledge graph, which are transformed into a broad spectrum of user queries. We then translate the relationships between entities into actionable tools and parse the pathways of each query into detailed solution steps, thereby creating high-quality instruction data. Our experiments show that fine-tuning on just a small sample of this synthetic data can significantly improve the tool utilization and overall capabilities of LLMs. 

---
# SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes 

**Authors**: Yifan Yang, Zhen Zhang, Rupak Vignesh Swaminathan, Jing Liu, Nathan Susanj, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20990)  

**Abstract**: Fine-tuning vision language models (VLMs) has achieved remarkable performance across various downstream tasks; yet, it requires access to model gradients through backpropagation (BP), making them unsuitable for memory-constrained, inference-only edge devices. To address this limitation, previous work has explored various BP-free fine-tuning methods. However, these approaches often rely on high-variance evolutionary strategies (ES) or zeroth-order (ZO) optimization, and often fail to achieve satisfactory performance. In this paper, we propose a hybrid Sharpness-aware Zeroth-order optimization (SharpZO) approach, specifically designed to enhance the performance of ZO VLM fine-tuning via a sharpness-aware warm-up training. SharpZO features a two-stage optimization process: a sharpness-aware ES stage that globally explores and smooths the loss landscape to construct a strong initialization, followed by a fine-grained local search via sparse ZO optimization. The entire optimization relies solely on forward passes. Detailed theoretical analysis and extensive experiments on CLIP models demonstrate that SharpZO significantly improves accuracy and convergence speed, achieving up to 7% average gain over state-of-the-art forward-only methods. 

---
# Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon Simulation 

**Authors**: Chenkai Sun, Denghui Zhang, ChengXiang Zhai, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.20949)  

**Abstract**: Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents. 

---
# Leaner Training, Lower Leakage: Revisiting Memorization in LLM Fine-Tuning with LoRA 

**Authors**: Fei Wang, Baochun Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.20856)  

**Abstract**: Memorization in large language models (LLMs) makes them vulnerable to data extraction attacks. While pre-training memorization has been extensively studied, fewer works have explored its impact in fine-tuning, particularly for LoRA fine-tuning, a widely adopted parameter-efficient method.
In this work, we re-examine memorization in fine-tuning and uncover a surprising divergence from prior findings across different fine-tuning strategies. Factors such as model scale and data duplication, which strongly influence memorization in pre-training and full fine-tuning, do not follow the same trend in LoRA fine-tuning. Using a more relaxed similarity-based memorization metric, we demonstrate that LoRA significantly reduces memorization risks compared to full fine-tuning, while still maintaining strong task performance. 

---
# MAGPIE: A dataset for Multi-AGent contextual PrIvacy Evaluation 

**Authors**: Gurusha Juneja, Alon Albalak, Wenyue Hua, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20737)  

**Abstract**: The proliferation of LLM-based agents has led to increasing deployment of inter-agent collaboration for tasks like scheduling, negotiation, resource allocation etc. In such systems, privacy is critical, as agents often access proprietary tools and domain-specific databases requiring strict confidentiality. This paper examines whether LLM-based agents demonstrate an understanding of contextual privacy. And, if instructed, do these systems preserve inference time user privacy in non-adversarial multi-turn conversation. Existing benchmarks to evaluate contextual privacy in LLM-agents primarily assess single-turn, low-complexity tasks where private information can be easily excluded. We first present a benchmark - MAGPIE comprising 158 real-life high-stakes scenarios across 15 domains. These scenarios are designed such that complete exclusion of private data impedes task completion yet unrestricted information sharing could lead to substantial losses. We then evaluate the current state-of-the-art LLMs on (a) their understanding of contextually private data and (b) their ability to collaborate without violating user privacy. Empirical experiments demonstrate that current models, including GPT-4o and Claude-2.7-Sonnet, lack robust understanding of contextual privacy, misclassifying private data as shareable 25.2\% and 43.6\% of the time. In multi-turn conversations, these models disclose private information in 59.9\% and 50.5\% of cases even under explicit privacy instructions. Furthermore, multi-agent systems fail to complete tasks in 71\% of scenarios. These results underscore that current models are not aligned towards both contextual privacy preservation and collaborative task-solving. 

---
