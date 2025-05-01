# Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval 

**Authors**: Aarush Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2504.21015)  

**Abstract**: Training effective dense retrieval models often relies on hard negative (HN) examples mined from the document corpus via methods like BM25 or cross-encoders (CE), processes that can be computationally demanding and require full corpus access. This paper introduces a different approach, an end-to-end pipeline where a Large Language Model (LLM) first generates a query from a passage, and then generates a hard negative example using \emph{only} that query text. This corpus-free negative generation contrasts with standard mining techniques. We evaluated this \textsc{LLM Query $\rightarrow$ LLM HN} approach against traditional \textsc{LLM Query $\rightarrow$ BM25 HN} and \textsc{LLM Query $\rightarrow$ CE HN} pipelines using E5-Base and GTE-Base models on several BEIR benchmark datasets. Our results show the proposed all-LLM pipeline achieves performance identical to both the BM25 and the computationally intensive CE baselines across nDCG@10, Precision@10, and Recall@100 metrics. This demonstrates that our corpus-free negative generation method matches the effectiveness of complex, corpus-dependent mining techniques, offering a potentially simpler and more efficient pathway for training high-performance retrievers without sacrificing results. We make the dataset including the queries and the hard-negatives for all three methods publicly available this https URL. 

---
# WebThinker: Empowering Large Reasoning Models with Deep Research Capability 

**Authors**: Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21776)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, demonstrate impressive long-horizon reasoning capabilities. However, their reliance on static internal knowledge limits their performance on complex, knowledge-intensive tasks and hinders their ability to produce comprehensive research reports requiring synthesis of diverse web information. To address this, we propose \textbf{WebThinker}, a deep research agent that empowers LRMs to autonomously search the web, navigate web pages, and draft research reports during the reasoning process. WebThinker integrates a \textbf{Deep Web Explorer} module, enabling LRMs to dynamically search, navigate, and extract information from the web when encountering knowledge gaps. It also employs an \textbf{Autonomous Think-Search-and-Draft strategy}, allowing the model to seamlessly interleave reasoning, information gathering, and report writing in real time. To further enhance research tool utilization, we introduce an \textbf{RL-based training strategy} via iterative online Direct Preference Optimization (DPO). Extensive experiments on complex reasoning benchmarks (GPQA, GAIA, WebWalkerQA, HLE) and scientific report generation tasks (Glaive) demonstrate that WebThinker significantly outperforms existing methods and strong proprietary systems. Our approach enhances LRM reliability and applicability in complex scenarios, paving the way for more capable and versatile deep research systems. The code is available at this https URL. 

---
# In a Few Words: Comparing Weak Supervision and LLMs for Short Query Intent Classification 

**Authors**: Daria Alexander, Arjen P. de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2504.21398)  

**Abstract**: User intent classification is an important task in information retrieval. Previously, user intents were classified manually and automatically; the latter helped to avoid hand labelling of large datasets. Recent studies explored whether LLMs can reliably determine user intent. However, researchers have recognized the limitations of using generative LLMs for classification tasks. In this study, we empirically compare user intent classification into informational, navigational, and transactional categories, using weak supervision and LLMs. Specifically, we evaluate LLaMA-3.1-8B-Instruct and LLaMA-3.1-70B-Instruct for in-context learning and LLaMA-3.1-8B-Instruct for fine-tuning, comparing their performance to an established baseline classifier trained using weak supervision (ORCAS-I). Our results indicate that while LLMs outperform weak supervision in recall, they continue to struggle with precision, which shows the need for improved methods to balance both metrics effectively. 

---
# RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations 

**Authors**: Jonas Gwozdz, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2504.21605)  

**Abstract**: Large Language Models (LLMs) increasingly serve as knowledge interfaces, yet systematically assessing their reliability with conflicting information remains difficult. We propose an RDF-based framework to assess multilingual LLM quality, focusing on knowledge conflicts. Our approach captures model responses across four distinct context conditions (complete, incomplete, conflicting, and no-context information) in German and English. This structured representation enables the comprehensive analysis of knowledge leakage-where models favor training data over provided context-error detection, and multilingual consistency. We demonstrate the framework through a fire safety domain experiment, revealing critical patterns in context prioritization and language-specific performance, and demonstrating that our vocabulary was sufficient to express every assessment facet encountered in the 28-question study. 

---
# Traceback of Poisoning Attacks to Retrieval-Augmented Generation 

**Authors**: Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21668)  

**Abstract**: Large language models (LLMs) integrated with retrieval-augmented generation (RAG) systems improve accuracy by leveraging external knowledge sources. However, recent research has revealed RAG's susceptibility to poisoning attacks, where the attacker injects poisoned texts into the knowledge database, leading to attacker-desired responses. Existing defenses, which predominantly focus on inference-time mitigation, have proven insufficient against sophisticated attacks. In this paper, we introduce RAGForensics, the first traceback system for RAG, designed to identify poisoned texts within the knowledge database that are responsible for the attacks. RAGForensics operates iteratively, first retrieving a subset of texts from the database and then utilizing a specially crafted prompt to guide an LLM in detecting potential poisoning texts. Empirical evaluations across multiple datasets demonstrate the effectiveness of RAGForensics against state-of-the-art poisoning attacks. This work pioneers the traceback of poisoned texts in RAG systems, providing a practical and promising defense mechanism to enhance their security. 

---
# Phi-4-reasoning Technical Report 

**Authors**: Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, Piero Kauffmann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, Guoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21318)  

**Abstract**: We introduce Phi-4-reasoning, a 14-billion parameter reasoning model that achieves strong performance on complex reasoning tasks. Trained via supervised fine-tuning of Phi-4 on carefully curated set of "teachable" prompts-selected for the right level of complexity and diversity-and reasoning demonstrations generated using o3-mini, Phi-4-reasoning generates detailed reasoning chains that effectively leverage inference-time compute. We further develop Phi-4-reasoning-plus, a variant enhanced through a short phase of outcome-based reinforcement learning that offers higher performance by generating longer reasoning traces. Across a wide range of reasoning tasks, both models outperform significantly larger open-weight models such as DeepSeek-R1-Distill-Llama-70B model and approach the performance levels of full DeepSeek-R1 model. Our comprehensive evaluations span benchmarks in math and scientific reasoning, coding, algorithmic problem solving, planning, and spatial understanding. Interestingly, we observe a non-trivial transfer of improvements to general-purpose benchmarks as well. In this report, we provide insights into our training data, our training methodologies, and our evaluations. We show that the benefit of careful data curation for supervised fine-tuning (SFT) extends to reasoning language models, and can be further amplified by reinforcement learning (RL). Finally, our evaluation points to opportunities for improving how we assess the performance and robustness of reasoning models. 

---
# AdaR1: From Long-CoT to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization 

**Authors**: Haotian Luo, Haiying He, Yibo Wang, Jinluan Yang, Rui Liu, Naiqiang Tan, Xiaochun Cao, Dacheng Tao, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21659)  

**Abstract**: Recently, long-thought reasoning models achieve strong performance on complex reasoning tasks, but often incur substantial inference overhead, making efficiency a critical concern. Our empirical analysis reveals that the benefit of using Long-CoT varies across problems: while some problems require elaborate reasoning, others show no improvement, or even degraded accuracy. This motivates adaptive reasoning strategies that tailor reasoning depth to the input. However, prior work primarily reduces redundancy within long reasoning paths, limiting exploration of more efficient strategies beyond the Long-CoT paradigm. To address this, we propose a novel two-stage framework for adaptive and efficient reasoning. First, we construct a hybrid reasoning model by merging long and short CoT models to enable diverse reasoning styles. Second, we apply bi-level preference training to guide the model to select suitable reasoning styles (group-level), and prefer concise and correct reasoning within each style group (instance-level). Experiments demonstrate that our method significantly reduces inference costs compared to other baseline approaches, while maintaining performance. Notably, on five mathematical datasets, the average length of reasoning is reduced by more than 50%, highlighting the potential of adaptive strategies to optimize reasoning efficiency in large language models. Our code is coming soon at this https URL 

---
# Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models 

**Authors**: Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, Minghui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21277)  

**Abstract**: The integration of reinforcement learning (RL) into the reasoning capabilities of Multimodal Large Language Models (MLLMs) has rapidly emerged as a transformative research direction. While MLLMs significantly extend Large Language Models (LLMs) to handle diverse modalities such as vision, audio, and video, enabling robust reasoning across multimodal inputs remains a major challenge. This survey systematically reviews recent advances in RL-based reasoning for MLLMs, covering key algorithmic designs, reward mechanism innovations, and practical applications. We highlight two main RL paradigms--value-free and value-based methods--and analyze how RL enhances reasoning abilities by optimizing reasoning trajectories and aligning multimodal information. Furthermore, we provide an extensive overview of benchmark datasets, evaluation protocols, and existing limitations, and propose future research directions to address current bottlenecks such as sparse rewards, inefficient cross-modal reasoning, and real-world deployment constraints. Our goal is to offer a comprehensive and structured guide to researchers interested in advancing RL-based reasoning in the multimodal era. 

---
# Theoretical Foundations for Semantic Cognition in Artificial Intelligence 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2504.21218)  

**Abstract**: This monograph presents a modular cognitive architecture for artificial intelligence grounded in the formal modeling of belief as structured semantic state. Belief states are defined as dynamic ensembles of linguistic expressions embedded within a navigable manifold, where operators enable assimilation, abstraction, nullification, memory, and introspection. Drawing from philosophy, cognitive science, and neuroscience, we develop a layered framework that enables self-regulating epistemic agents capable of reflective, goal-directed thought. At the core of this framework is the epistemic vacuum: a class of semantically inert cognitive states that serves as the conceptual origin of belief space. From this foundation, the Null Tower arises as a generative structure recursively built through internal representational capacities. The theoretical constructs are designed to be implementable in both symbolic and neural systems, including large language models, hybrid agents, and adaptive memory architectures. This work offers a foundational substrate for constructing agents that reason, remember, and regulate their beliefs in structured, interpretable ways. 

---
# TRUST: An LLM-Based Dialogue System for Trauma Understanding and Structured Assessments 

**Authors**: Sichang Tu, Abigail Powers, Stephen Doogan, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21851)  

**Abstract**: Objectives: While Large Language Models (LLMs) have been widely used to assist clinicians and support patients, no existing work has explored dialogue systems for standard diagnostic interviews and assessments. This study aims to bridge the gap in mental healthcare accessibility by developing an LLM-powered dialogue system that replicates clinician behavior. Materials and Methods: We introduce TRUST, a framework of cooperative LLM modules capable of conducting formal diagnostic interviews and assessments for Post-Traumatic Stress Disorder (PTSD). To guide the generation of appropriate clinical responses, we propose a Dialogue Acts schema specifically designed for clinical interviews. Additionally, we develop a patient simulation approach based on real-life interview transcripts to replace time-consuming and costly manual testing by clinicians. Results: A comprehensive set of evaluation metrics is designed to assess the dialogue system from both the agent and patient simulation perspectives. Expert evaluations by conversation and clinical specialists show that TRUST performs comparably to real-life clinical interviews. Discussion: Our system performs at the level of average clinicians, with room for future enhancements in communication styles and response appropriateness. Conclusions: Our TRUST framework shows its potential to facilitate mental healthcare availability. 

---
# MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness 

**Authors**: Junsheng Huang, Zhitao He, Sandeep Polisetty, Qingyun Wang, May Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.21773)  

**Abstract**: With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision. 

---
# LLM-Empowered Embodied Agent for Memory-Augmented Task Planning in Household Robotics 

**Authors**: Marc Glocker, Peter Hönig, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2504.21716)  

**Abstract**: We present an embodied robotic system with an LLM-driven agent-orchestration architecture for autonomous household object management. The system integrates memory-augmented task planning, enabling robots to execute high-level user commands while tracking past actions. It employs three specialized agents: a routing agent, a task planning agent, and a knowledge base agent, each powered by task-specific LLMs. By leveraging in-context learning, our system avoids the need for explicit model training. RAG enables the system to retrieve context from past interactions, enhancing long-term object tracking. A combination of Grounded SAM and LLaMa3.2-Vision provides robust object detection, facilitating semantic scene understanding for task planning. Evaluation across three household scenarios demonstrates high task planning accuracy and an improvement in memory recall due to RAG. Specifically, Qwen2.5 yields best performance for specialized agents, while LLaMA3.1 excels in routing tasks. The source code is available at: this https URL. 

---
# ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning 

**Authors**: Jingyang Yi, Jiazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21370)  

**Abstract**: Reasoning models such as OpenAI o3 and DeepSeek-R1 have demonstrated strong performance on reasoning-intensive tasks through extended Chain-of-Thought (CoT) prompting. While longer reasoning traces can facilitate a more thorough exploration of solution paths for complex problems, researchers have observed that these models often "overthink", leading to inefficient inference. In this paper, we introduce ShorterBetter, a simple yet effective reinforcement learning methed that enables reasoning language models to discover their own optimal CoT lengths without human intervention. By sampling multiple outputs per problem and defining the Sample Optimal Length (SOL) as the shortest correct response among all the outputs, our method dynamically guides the model toward optimal inference lengths. Applied to the DeepSeek-Distill-Qwen-1.5B model, ShorterBetter achieves up to an 80% reduction in output length on both in-domain and out-of-domain reasoning tasks while maintaining accuracy. Our analysis shows that overly long reasoning traces often reflect loss of reasoning direction, and thus suggests that the extended CoT produced by reasoning models is highly compressible. 

---
# Sadeed: Advancing Arabic Diacritization Through Small Language Model 

**Authors**: Zeina Aldallal, Sara Chrouf, Khalil Hennara, Mohamed Motaism Hamed, Muhammad Hreden, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21635)  

**Abstract**: Arabic text diacritization remains a persistent challenge in natural language processing due to the language's morphological richness. In this paper, we introduce Sadeed, a novel approach based on a fine-tuned decoder-only language model adapted from Kuwain 1.5B Hennara et al. [2025], a compact model originally trained on diverse Arabic corpora. Sadeed is fine-tuned on carefully curated, high-quality diacritized datasets, constructed through a rigorous data-cleaning and normalization pipeline. Despite utilizing modest computational resources, Sadeed achieves competitive results compared to proprietary large language models and outperforms traditional models trained on similar domains. Additionally, we highlight key limitations in current benchmarking practices for Arabic diacritization. To address these issues, we introduce SadeedDiac-25, a new benchmark designed to enable fairer and more comprehensive evaluation across diverse text genres and complexity levels. Together, Sadeed and SadeedDiac-25 provide a robust foundation for advancing Arabic NLP applications, including machine translation, text-to-speech, and language learning tools. 

---
# Leveraging Pre-trained Large Language Models with Refined Prompting for Online Task and Motion Planning 

**Authors**: Huihui Guo, Huilong Pi, Yunchuan Qin, Zhuo Tang, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.21596)  

**Abstract**: With the rapid advancement of artificial intelligence, there is an increasing demand for intelligent robots capable of assisting humans in daily tasks and performing complex operations. Such robots not only require task planning capabilities but must also execute tasks with stability and robustness. In this paper, we present a closed-loop task planning and acting system, LLM-PAS, which is assisted by a pre-trained Large Language Model (LLM). While LLM-PAS plans long-horizon tasks in a manner similar to traditional task and motion planners, it also emphasizes the execution phase of the task. By transferring part of the constraint-checking process from the planning phase to the execution phase, LLM-PAS enables exploration of the constraint space and delivers more accurate feedback on environmental anomalies during execution. The reasoning capabilities of the LLM allow it to handle anomalies that cannot be addressed by the robust executor. To further enhance the system's ability to assist the planner during replanning, we propose the First Look Prompting (FLP) method, which induces LLM to generate effective PDDL goals. Through comparative prompting experiments and systematic experiments, we demonstrate the effectiveness and robustness of LLM-PAS in handling anomalous conditions during task execution. 

---
# DNB-AI-Project at SemEval-2025 Task 5: An LLM-Ensemble Approach for Automated Subject Indexing 

**Authors**: Lisa Kluge, Maximilian Kähler  

**Link**: [PDF](https://arxiv.org/pdf/2504.21589)  

**Abstract**: This paper presents our system developed for the SemEval-2025 Task 5: LLMs4Subjects: LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog. Our system relies on prompting a selection of LLMs with varying examples of intellectually annotated records and asking the LLMs to similarly suggest keywords for new records. This few-shot prompting technique is combined with a series of post-processing steps that map the generated keywords to the target vocabulary, aggregate the resulting subject terms to an ensemble vote and, finally, rank them as to their relevance to the record. Our system is fourth in the quantitative ranking in the all-subjects track, but achieves the best result in the qualitative ranking conducted by subject indexing experts. 

---
# MF-LLM: Simulating Collective Decision Dynamics via a Mean-Field Large Language Model Framework 

**Authors**: Qirui Mi, Mengyue Yang, Xiangning Yu, Zhiyu Zhao, Cheng Deng, Bo An, Haifeng Zhang, Xu Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21582)  

**Abstract**: Simulating collective decision-making involves more than aggregating individual behaviors; it arises from dynamic interactions among individuals. While large language models (LLMs) show promise for social simulation, existing approaches often exhibit deviations from real-world data. To address this gap, we propose the Mean-Field LLM (MF-LLM) framework, which explicitly models the feedback loop between micro-level decisions and macro-level population. MF-LLM alternates between two models: a policy model that generates individual actions based on personal states and group-level information, and a mean field model that updates the population distribution from the latest individual decisions. Together, they produce rollouts that simulate the evolving trajectories of collective decision-making. To better match real-world data, we introduce IB-Tune, a fine-tuning method for LLMs grounded in the information bottleneck principle, which maximizes the relevance of population distributions to future actions while minimizing redundancy with historical data. We evaluate MF-LLM on a real-world social dataset, where it reduces KL divergence to human population distributions by 47 percent over non-mean-field baselines, and enables accurate trend forecasting and intervention planning. It generalizes across seven domains and four LLM backbones, providing a scalable foundation for high-fidelity social simulation. 

---
# SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding 

**Authors**: Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, ShaoGuo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21435)  

**Abstract**: With the rapid development of Multi-modal Large Language Models (MLLMs), an increasing number of benchmarks have been established to evaluate the video understanding capabilities of these models. However, these benchmarks focus on \textbf{standalone} videos and mainly assess ``visual elements'' like human actions and object states. In reality, contemporary videos often encompass complex and continuous narratives, typically presented as a \textbf{series}. To address this challenge, we propose \textbf{SeriesBench}, a benchmark consisting of 105 carefully curated narrative-driven series, covering 28 specialized tasks that require deep narrative understanding. Specifically, we first select a diverse set of drama series spanning various genres. Then, we introduce a novel long-span narrative annotation method, combined with a full-information transformation approach to convert manual annotations into diverse task formats. To further enhance model capacity for detailed analysis of plot structures and character relationships within series, we propose a novel narrative reasoning framework, \textbf{PC-DCoT}. Extensive results on \textbf{SeriesBench} indicate that existing MLLMs still face significant challenges in understanding narrative-driven series, while \textbf{PC-DCoT} enables these MLLMs to achieve performance improvements. Overall, our \textbf{SeriesBench} and \textbf{PC-DCoT} highlight the critical necessity of advancing model capabilities to understand narrative-driven series, guiding the future development of MLLMs. SeriesBench is publicly available at this https URL. 

---
# Retrieval-Enhanced Few-Shot Prompting for Speech Event Extraction 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21372)  

**Abstract**: Speech Event Extraction (SpeechEE) is a challenging task that lies at the intersection of Automatic Speech Recognition (ASR) and Natural Language Processing (NLP), requiring the identification of structured event information from spoken language. In this work, we present a modular, pipeline-based SpeechEE framework that integrates high-performance ASR with semantic search-enhanced prompting of Large Language Models (LLMs). Our system first classifies speech segments likely to contain events using a hybrid filtering mechanism including rule-based, BERT-based, and LLM-based models. It then employs few-shot LLM prompting, dynamically enriched via semantic similarity retrieval, to identify event triggers and extract corresponding arguments. We evaluate the pipeline using multiple LLMs (Llama3-8B, GPT-4o-mini, and o1-mini) highlighting significant performance gains with o1-mini, which achieves 63.3% F1 on trigger classification and 27.8% F1 on argument classification, outperforming prior benchmarks. Our results demonstrate that pipeline approaches, when empowered by retrieval-augmented LLMs, can rival or exceed end-to-end systems while maintaining interpretability and modularity. This work provides practical insights into LLM-driven event extraction and opens pathways for future hybrid models combining textual and acoustic features. 

---
# Nexus-Gen: A Unified Model for Image Understanding, Generation, and Editing 

**Authors**: Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, Yuze Zhao, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21356)  

**Abstract**: Unified multimodal large language models (MLLMs) aim to integrate multimodal understanding and generation abilities through a single framework. Despite their versatility, existing open-source unified models exhibit performance gaps against domain-specific architectures. To bridge this gap, we present Nexus-Gen, a unified model that synergizes the language reasoning capabilities of LLMs with the image synthesis power of diffusion models. To align the embedding space of the LLM and diffusion model, we conduct a dual-phase alignment training process. (1) The autoregressive LLM learns to predict image embeddings conditioned on multimodal inputs, while (2) the vision decoder is trained to reconstruct high-fidelity images from these embeddings. During training the LLM, we identified a critical discrepancy between the autoregressive paradigm's training and inference phases, where error accumulation in continuous embedding space severely degrades generation quality. To avoid this issue, we introduce a prefilled autoregression strategy that prefills input sequence with position-embedded special tokens instead of continuous embeddings. Through dual-phase training, Nexus-Gen has developed the integrated capability to comprehensively address the image understanding, generation and editing tasks. All models, datasets, and codes are published at this https URL to facilitate further advancements across the field. 

---
# Assessing LLM code generation quality through path planning tasks 

**Authors**: Wanyi Chen, Meng-Wen Su, Mary L. Cummings  

**Link**: [PDF](https://arxiv.org/pdf/2504.21276)  

**Abstract**: As LLM-generated code grows in popularity, more evaluation is needed to assess the risks of using such tools, especially for safety-critical applications such as path planning. Existing coding benchmarks are insufficient as they do not reflect the context and complexity of safety-critical applications. To this end, we assessed six LLMs' abilities to generate the code for three different path-planning algorithms and tested them on three maps of various difficulties. Our results suggest that LLM-generated code presents serious hazards for path planning applications and should not be applied in safety-critical contexts without rigorous testing. 

---
# Memorization and Knowledge Injection in Gated LLMs 

**Authors**: Xu Pan, Ely Hahami, Zechen Zhang, Haim Sompolinsky  

**Link**: [PDF](https://arxiv.org/pdf/2504.21239)  

**Abstract**: Large Language Models (LLMs) currently struggle to sequentially add new memories and integrate new knowledge. These limitations contrast with the human ability to continuously learn from new experiences and acquire knowledge throughout life. Most existing approaches add memories either through large context windows or external memory buffers (e.g., Retrieval-Augmented Generation), and studies on knowledge injection rarely test scenarios resembling everyday life events. In this work, we introduce a continual learning framework, Memory Embedded in Gated LLMs (MEGa), which injects event memories directly into the weights of LLMs. Each memory is stored in a dedicated set of gated low-rank weights. During inference, a gating mechanism activates relevant memory weights by matching query embeddings to stored memory embeddings. This enables the model to both recall entire memories and answer related questions. On two datasets - fictional characters and Wikipedia events - MEGa outperforms baseline approaches in mitigating catastrophic forgetting. Our model draws inspiration from the complementary memory system of the human brain. 

---
# A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces 

**Authors**: Juliana Barbosa, Ulhas Gondhali, Gohar Petrossian, Kinshuk Sharma, Sunandan Chakraborty, Jennifer Jacquet, Juliana Freire  

**Link**: [PDF](https://arxiv.org/pdf/2504.21211)  

**Abstract**: Wildlife trafficking remains a critical global issue, significantly impacting biodiversity, ecological stability, and public health. Despite efforts to combat this illicit trade, the rise of e-commerce platforms has made it easier to sell wildlife products, putting new pressure on wild populations of endangered and threatened species. The use of these platforms also opens a new opportunity: as criminals sell wildlife products online, they leave digital traces of their activity that can provide insights into trafficking activities as well as how they can be disrupted. The challenge lies in finding these traces. Online marketplaces publish ads for a plethora of products, and identifying ads for wildlife-related products is like finding a needle in a haystack. Learning classifiers can automate ad identification, but creating them requires costly, time-consuming data labeling that hinders support for diverse ads and research questions. This paper addresses a critical challenge in the data science pipeline for wildlife trafficking analytics: generating quality labeled data for classifiers that select relevant data. While large language models (LLMs) can directly label advertisements, doing so at scale is prohibitively expensive. We propose a cost-effective strategy that leverages LLMs to generate pseudo labels for a small sample of the data and uses these labels to create specialized classification models. Our novel method automatically gathers diverse and representative samples to be labeled while minimizing the labeling costs. Our experimental evaluation shows that our classifiers achieve up to 95% F1 score, outperforming LLMs at a lower cost. We present real use cases that demonstrate the effectiveness of our approach in enabling analyses of different aspects of wildlife trafficking. 

---
# Automatic Legal Writing Evaluation of LLMs 

**Authors**: Ramon Pires, Roseval Malaquias Junior, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2504.21202)  

**Abstract**: Despite the recent advances in Large Language Models, benchmarks for evaluating legal writing remain scarce due to the inherent complexity of assessing open-ended responses in this domain. One of the key challenges in evaluating language models on domain-specific tasks is finding test datasets that are public, frequently updated, and contain comprehensive evaluation guidelines. The Brazilian Bar Examination meets these requirements. We introduce oab-bench, a benchmark comprising 105 questions across seven areas of law from recent editions of the exam. The benchmark includes comprehensive evaluation guidelines and reference materials used by human examiners to ensure consistent grading. We evaluate the performance of four LLMs on oab-bench, finding that Claude-3.5 Sonnet achieves the best results with an average score of 7.93 out of 10, passing all 21 exams. We also investigated whether LLMs can serve as reliable automated judges for evaluating legal writing. Our experiments show that frontier models like OpenAI's o1 achieve a strong correlation with human scores when evaluating approved exams, suggesting their potential as reliable automated evaluators despite the inherently subjective nature of legal writing assessment. The source code and the benchmark -- containing questions, evaluation guidelines, model-generated responses, and their respective automated evaluations -- are publicly available. 

---
# SecRepoBench: Benchmarking LLMs for Secure Code Generation in Real-World Repositories 

**Authors**: Connor Dilgren, Purva Chiniya, Luke Griffith, Yu Ding, Yizheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21205)  

**Abstract**: This paper introduces SecRepoBench, a benchmark to evaluate LLMs on secure code generation in real-world repositories. SecRepoBench has 318 code generation tasks in 27 C/C++ repositories, covering 15 CWEs. We evaluate 19 state-of-the-art LLMs using our benchmark and find that the models struggle with generating correct and secure code. In addition, the performance of LLMs to generate self-contained programs as measured by prior benchmarks do not translate to comparative performance at generating secure and correct code at the repository level in SecRepoBench. We show that the state-of-the-art prompt engineering techniques become less effective when applied to the repository level secure code generation problem. We conduct extensive experiments, including an agentic technique to generate secure code, to demonstrate that our benchmark is currently the most difficult secure coding benchmark, compared to previous state-of-the-art benchmarks. Finally, our comprehensive analysis provides insights into potential directions for enhancing the ability of LLMs to generate correct and secure code in real-world repositories. 

---
# On the Potential of Large Language Models to Solve Semantics-Aware Process Mining Tasks 

**Authors**: Adrian Rebmann, Fabian David Schmidt, Goran Glavaš, Han van der Aa  

**Link**: [PDF](https://arxiv.org/pdf/2504.21074)  

**Abstract**: Large language models (LLMs) have shown to be valuable tools for tackling process mining tasks. Existing studies report on their capability to support various data-driven process analyses and even, to some extent, that they are able to reason about how processes work. This reasoning ability suggests that there is potential for LLMs to tackle semantics-aware process mining tasks, which are tasks that rely on an understanding of the meaning of activities and their relationships. Examples of these include process discovery, where the meaning of activities can indicate their dependency, whereas in anomaly detection the meaning can be used to recognize process behavior that is abnormal. In this paper, we systematically explore the capabilities of LLMs for such tasks. Unlike prior work, which largely evaluates LLMs in their default state, we investigate their utility through both in-context learning and supervised fine-tuning. Concretely, we define five process mining tasks requiring semantic understanding and provide extensive benchmarking datasets for evaluation. Our experiments reveal that while LLMs struggle with challenging process mining tasks when used out of the box or with minimal in-context examples, they achieve strong performance when fine-tuned for these tasks across a broad range of process types and industries. 

---
# CodeBC: A More Secure Large Language Model for Smart Contract Code Generation in Blockchain 

**Authors**: Lingxiang wang, Hainan Zhang, Qinnan Zhang, Ziwei Wang, Hongwei Zheng, Jin Dong, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21043)  

**Abstract**: Large language models (LLMs) excel at generating code from natural language instructions, yet they often lack an understanding of security vulnerabilities. This limitation makes it difficult for LLMs to avoid security risks in generated code, particularly in high-security programming tasks such as smart contract development for blockchain. Researchers have attempted to enhance the vulnerability awareness of these models by training them to differentiate between vulnerable and fixed code snippets. However, this approach relies heavily on manually labeled vulnerability data, which is only available for popular languages like Python and C++. For low-resource languages like Solidity, used in smart contracts, large-scale annotated datasets are scarce and difficult to obtain. To address this challenge, we introduce CodeBC, a code generation model specifically designed for generating secure smart contracts in blockchain. CodeBC employs a three-stage fine-tuning approach based on CodeLlama, distinguishing itself from previous methods by not relying on pairwise vulnerability location annotations. Instead, it leverages vulnerability and security tags to teach the model the differences between vulnerable and secure code. During the inference phase, the model leverages security tags to generate secure and robust code. Experimental results demonstrate that CodeBC outperforms baseline models in terms of BLEU, CodeBLEU, and compilation pass rates, while significantly reducing vulnerability rates. These findings validate the effectiveness and cost-efficiency of our three-stage fine-tuning strategy, making CodeBC a promising solution for generating secure smart contract code. 

---
# Llama-3.1-FoundationAI-SecurityLLM-Base-8B Technical Report 

**Authors**: Paul Kassianik, Baturay Saglam, Alexander Chen, Blaine Nelson, Anu Vellore, Massimo Aufiero, Fraser Burch, Dhruv Kedia, Avi Zohary, Sajana Weerawardhena, Aman Priyanshu, Adam Swanda, Amy Chang, Hyrum Anderson, Kojin Oshiba, Omar Santos, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21039)  

**Abstract**: As transformer-based large language models (LLMs) increasingly permeate society, they have revolutionized domains such as software engineering, creative writing, and digital arts. However, their adoption in cybersecurity remains limited due to challenges like scarcity of specialized training data and complexity of representing cybersecurity-specific knowledge. To address these gaps, we present Foundation-Sec-8B, a cybersecurity-focused LLM built on the Llama 3.1 architecture and enhanced through continued pretraining on a carefully curated cybersecurity corpus. We evaluate Foundation-Sec-8B across both established and new cybersecurity benchmarks, showing that it matches Llama 3.1-70B and GPT-4o-mini in certain cybersecurity-specific tasks. By releasing our model to the public, we aim to accelerate progress and adoption of AI-driven tools in both public and private cybersecurity contexts. 

---
# Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary 

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Jing Xie, Weijuan Zhang, Aimin Yu, Shijie Zhao, Qingjia Huang, Qihang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21038)  

**Abstract**: Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available. 

---
# SAGA: A Security Architecture for Governing AI Agentic Systems 

**Authors**: Georgios Syros, Anshuman Suri, Cristina Nita-Rotaru, Alina Oprea  

**Link**: [PDF](https://arxiv.org/pdf/2504.21034)  

**Abstract**: Large Language Model (LLM)-based agents increasingly interact, collaborate, and delegate tasks to one another autonomously with minimal human interaction. Industry guidelines for agentic system governance emphasize the need for users to maintain comprehensive control over their agents, mitigating potential damage from malicious agents. Several proposed agentic system designs address agent identity, authorization, and delegation, but remain purely theoretical, without concrete implementation and evaluation. Most importantly, they do not provide user-controlled agent management. To address this gap, we propose SAGA, a Security Architecture for Governing Agentic systems, that offers user oversight over their agents' lifecycle. In our design, users register their agents with a central entity, the Provider, that maintains agents contact information, user-defined access control policies, and helps agents enforce these policies on inter-agent communication. We introduce a cryptographic mechanism for deriving access control tokens, that offers fine-grained control over an agent's interaction with other agents, balancing security and performance consideration. We evaluate SAGA on several agentic tasks, using agents in different geolocations, and multiple on-device and cloud LLMs, demonstrating minimal performance overhead with no impact on underlying task utility in a wide range of conditions. Our architecture enables secure and trustworthy deployment of autonomous agents, accelerating the responsible adoption of this technology in sensitive environments. 

---
# NeuRel-Attack: Neuron Relearning for Safety Disalignment in Large Language Models 

**Authors**: Yi Zhou, Wenpeng Xing, Dezhang Kong, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.21053)  

**Abstract**: Safety alignment in large language models (LLMs) is achieved through fine-tuning mechanisms that regulate neuron activations to suppress harmful content. In this work, we propose a novel approach to induce disalignment by identifying and modifying the neurons responsible for safety constraints. Our method consists of three key steps: Neuron Activation Analysis, where we examine activation patterns in response to harmful and harmless prompts to detect neurons that are critical for distinguishing between harmful and harmless inputs; Similarity-Based Neuron Identification, which systematically locates the neurons responsible for safe alignment; and Neuron Relearning for Safety Removal, where we fine-tune these selected neurons to restore the model's ability to generate previously restricted responses. Experimental results demonstrate that our method effectively removes safety constraints with minimal fine-tuning, highlighting a critical vulnerability in current alignment techniques. Our findings underscore the need for robust defenses against adversarial fine-tuning attacks on LLMs. 

---
# Leveraging LLM to Strengthen ML-Based Cross-Site Scripting Detection 

**Authors**: Dennis Miczek, Divyesh Gabbireddy, Suman Saha  

**Link**: [PDF](https://arxiv.org/pdf/2504.21045)  

**Abstract**: According to the Open Web Application Security Project (OWASP), Cross-Site Scripting (XSS) is a critical security vulnerability. Despite decades of research, XSS remains among the top 10 security vulnerabilities. Researchers have proposed various techniques to protect systems from XSS attacks, with machine learning (ML) being one of the most widely used methods. An ML model is trained on a dataset to identify potential XSS threats, making its effectiveness highly dependent on the size and diversity of the training data. A variation of XSS is obfuscated XSS, where attackers apply obfuscation techniques to alter the code's structure, making it challenging for security systems to detect its malicious intent. Our study's random forest model was trained on traditional (non-obfuscated) XSS data achieved 99.8% accuracy. However, when tested against obfuscated XSS samples, accuracy dropped to 81.9%, underscoring the importance of training ML models with obfuscated data to improve their effectiveness in detecting XSS attacks. A significant challenge is to generate highly complex obfuscated code despite the availability of several public tools. These tools can only produce obfuscation up to certain levels of complexity.
In our proposed system, we fine-tune a Large Language Model (LLM) to generate complex obfuscated XSS payloads automatically. By transforming original XSS samples into diverse obfuscated variants, we create challenging training data for ML model evaluation. Our approach achieved a 99.5% accuracy rate with the obfuscated dataset. We also found that the obfuscated samples generated by the LLMs were 28.1% more complex than those created by other tools, significantly improving the model's ability to handle advanced XSS attacks and making it more effective for real-world application security. 

---
# UrbanPlanBench: A Comprehensive Urban Planning Benchmark for Evaluating Large Language Models 

**Authors**: Yu Zheng, Longyi Liu, Yuming Lin, Jie Feng, Guozhen Zhang, Depeng Jin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.21027)  

**Abstract**: The advent of Large Language Models (LLMs) holds promise for revolutionizing various fields traditionally dominated by human expertise. Urban planning, a professional discipline that fundamentally shapes our daily surroundings, is one such field heavily relying on multifaceted domain knowledge and experience of human experts. The extent to which LLMs can assist human practitioners in urban planning remains largely unexplored. In this paper, we introduce a comprehensive benchmark, UrbanPlanBench, tailored to evaluate the efficacy of LLMs in urban planning, which encompasses fundamental principles, professional knowledge, and management and regulations, aligning closely with the qualifications expected of human planners. Through extensive evaluation, we reveal a significant imbalance in the acquisition of planning knowledge among LLMs, with even the most proficient models falling short of meeting professional standards. For instance, we observe that 70% of LLMs achieve subpar performance in understanding planning regulations compared to other aspects. Besides the benchmark, we present the largest-ever supervised fine-tuning (SFT) dataset, UrbanPlanText, comprising over 30,000 instruction pairs sourced from urban planning exams and textbooks. Our findings demonstrate that fine-tuned models exhibit enhanced performance in memorization tests and comprehension of urban planning knowledge, while there exists significant room for improvement, particularly in tasks requiring domain-specific terminology and reasoning. By making our benchmark, dataset, and associated evaluation and fine-tuning toolsets publicly available at this https URL, we aim to catalyze the integration of LLMs into practical urban planning, fostering a symbiotic collaboration between human expertise and machine intelligence. 

---
# ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees 

**Authors**: Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2504.21022)  

**Abstract**: Linear Temporal Logic (LTL) has become a prevalent specification language for robotic tasks. To mitigate the significant manual effort and expertise required to define LTL-encoded tasks, several methods have been proposed for translating Natural Language (NL) instructions into LTL formulas, which, however, lack correctness guarantees. To address this, we introduce a new NL-to-LTL translation method, called ConformalNL2LTL, that can achieve user-defined translation success rates over unseen NL commands. Our method constructs LTL formulas iteratively by addressing a sequence of open-vocabulary Question-Answering (QA) problems with LLMs. To enable uncertainty-aware translation, we leverage conformal prediction (CP), a distribution-free uncertainty quantification tool for black-box models. CP enables our method to assess the uncertainty in LLM-generated answers, allowing it to proceed with translation when sufficiently confident and request help otherwise. We provide both theoretical and empirical results demonstrating that ConformalNL2LTL achieves user-specified translation accuracy while minimizing help rates. 

---
# Selecting the Right LLM for eGov Explanations 

**Authors**: Lior Limonad, Fabiana Fournier, Hadar Mulian, George Manias, Spiros Borotis, Danai Kyrkou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21032)  

**Abstract**: The perceived quality of the explanations accompanying e-government services is key to gaining trust in these institutions, consequently amplifying further usage of these services. Recent advances in generative AI, and concretely in Large Language Models (LLMs) allow the automation of such content articulations, eliciting explanations' interpretability and fidelity, and more generally, adapting content to various audiences. However, selecting the right LLM type for this has become a non-trivial task for e-government service providers. In this work, we adapted a previously developed scale to assist with this selection, providing a systematic approach for the comparative analysis of the perceived quality of explanations generated by various LLMs. We further demonstrated its applicability through the tax-return process, using it as an exemplar use case that could benefit from employing an LLM to generate explanations about tax refund decisions. This was attained through a user study with 128 survey respondents who were asked to rate different versions of LLM-generated explanations about tax refund decisions, providing a methodological basis for selecting the most appropriate LLM. Recognizing the practical challenges of conducting such a survey, we also began exploring the automation of this process by attempting to replicate human feedback using a selection of cutting-edge predictive techniques. 

---
# Creating and Evaluating Code-Mixed Nepali-English and Telugu-English Datasets for Abusive Language Detection Using Traditional and Deep Learning Models 

**Authors**: Manish Pandey, Nageshwar Prasad Yadav, Mokshada Adduru, Sawan Rai  

**Link**: [PDF](https://arxiv.org/pdf/2504.21026)  

**Abstract**: With the growing presence of multilingual users on social media, detecting abusive language in code-mixed text has become increasingly challenging. Code-mixed communication, where users seamlessly switch between English and their native languages, poses difficulties for traditional abuse detection models, as offensive content may be context-dependent or obscured by linguistic blending. While abusive language detection has been extensively explored for high-resource languages like English and Hindi, low-resource languages such as Telugu and Nepali remain underrepresented, leaving gaps in effective moderation. In this study, we introduce a novel, manually annotated dataset of 2 thousand Telugu-English and 5 Nepali-English code-mixed comments, categorized as abusive and non-abusive, collected from various social media platforms. The dataset undergoes rigorous preprocessing before being evaluated across multiple Machine Learning (ML), Deep Learning (DL), and Large Language Models (LLMs). We experimented with models including Logistic Regression, Random Forest, Support Vector Machines (SVM), Neural Networks (NN), LSTM, CNN, and LLMs, optimizing their performance through hyperparameter tuning, and evaluate it using 10-fold cross-validation and statistical significance testing (t-test). Our findings provide key insights into the challenges of detecting abusive language in code-mixed settings and offer a comparative analysis of computational approaches. This study contributes to advancing NLP for low-resource languages by establishing benchmarks for abusive language detection in Telugu-English and Nepali-English code-mixed text. The dataset and insights can aid in the development of more robust moderation strategies for multilingual social media environments. 

---
# Waking Up an AI: A Quantitative Framework for Prompt-Induced Phase Transition in Large Language Models 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2504.21012)  

**Abstract**: What underlies intuitive human thinking? One approach to this question is to compare the cognitive dynamics of humans and large language models (LLMs). However, such a comparison requires a method to quantitatively analyze AI cognitive behavior under controlled conditions. While anecdotal observations suggest that certain prompts can dramatically change LLM behavior, these observations have remained largely qualitative. Here, we propose a two-part framework to investigate this phenomenon: a Transition-Inducing Prompt (TIP) that triggers a rapid shift in LLM responsiveness, and a Transition Quantifying Prompt (TQP) that evaluates this change using a separate LLM. Through controlled experiments, we examined how LLMs react to prompts embedding two semantically distant concepts (e.g., mathematical aperiodicity and traditional crafts)--either fused together or presented separately--by changing their linguistic quality and affective tone. Whereas humans tend to experience heightened engagement when such concepts are meaningfully blended producing a novel concept--a form of conceptual fusion--current LLMs showed no significant difference in responsiveness between semantically fused and non-fused prompts. This suggests that LLMs may not yet replicate the conceptual integration processes seen in human intuition. Our method enables fine-grained, reproducible measurement of cognitive responsiveness, and may help illuminate key differences in how intuition and conceptual leaps emerge in artificial versus human minds. 

---
# Context-Enhanced Contrastive Search for Improved LLM Text Generation 

**Authors**: Jaydip Sen, Rohit Pandey, Hetvi Waghela  

**Link**: [PDF](https://arxiv.org/pdf/2504.21020)  

**Abstract**: Recently, Large Language Models (LLMs) have demonstrated remarkable advancements in Natural Language Processing (NLP). However, generating high-quality text that balances coherence, diversity, and relevance remains challenging. Traditional decoding methods, such as bean search and top-k sampling, often struggle with either repetitive or incoherent outputs, particularly in tasks that require long-form text generation. To address these limitations, the paper proposes a novel enhancement of the well-known Contrastive Search algorithm, Context-Enhanced Contrastive Search (CECS) with contextual calibration. The proposed scheme introduces several novelties including dynamic contextual importance weighting, multi-level Contrastive Search, and adaptive temperature control, to optimize the balance between fluency, creativity, and precision. The performance of CECS is evaluated using several standard metrics such as BLEU, ROUGE, and semantic similarity. Experimental results demonstrate significant improvements in both coherence and relevance of the generated texts by CECS outperforming the existing Contrastive Search techniques. The proposed algorithm has several potential applications in the real world including legal document drafting, customer service chatbots, and content marketing. 

---
# DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition 

**Authors**: Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21801)  

**Abstract**: We introduce DeepSeek-Prover-V2, an open-source large language model designed for formal theorem proving in Lean 4, with initialization data collected through a recursive theorem proving pipeline powered by DeepSeek-V3. The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals. The proofs of resolved subgoals are synthesized into a chain-of-thought process, combined with DeepSeek-V3's step-by-step reasoning, to create an initial cold start for reinforcement learning. This process enables us to integrate both informal and formal mathematical reasoning into a unified model. The resulting model, DeepSeek-Prover-V2-671B, achieves state-of-the-art performance in neural theorem proving, reaching 88.9% pass ratio on the MiniF2F-test and solving 49 out of 658 problems from PutnamBench. In addition to standard benchmarks, we introduce ProverBench, a collection of 325 formalized problems, to enrich our evaluation, including 15 selected problems from the recent AIME competitions (years 24-25). Further evaluation on these 15 AIME problems shows that the model successfully solves 6 of them. In comparison, DeepSeek-V3 solves 8 of these problems using majority voting, highlighting that the gap between formal and informal mathematical reasoning in large language models is substantially narrowing. 

---
# Meeseeks: An Iterative Benchmark Evaluating LLMs Multi-Turn Instruction-Following Ability 

**Authors**: Jiaming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21625)  

**Abstract**: The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. While existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction, Meeseeks simulates realistic human-LLM interactions through an iterative feedback process. This design enables models to self-correct based on specific requirement failures, better reflecting real-world user-end usage patterns. The benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in practical applications. 

---
# Confidence in Large Language Model Evaluation: A Bayesian Approach to Limited-Sample Challenges 

**Authors**: Xiao Xiao, Yu Su, Sijing Zhang, Zhang Chen, Yadong Chen, Tian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21303)  

**Abstract**: Large language models (LLMs) exhibit probabilistic output characteristics, yet conventional evaluation frameworks rely on deterministic scalar metrics. This study introduces a Bayesian approach for LLM capability assessment that integrates prior knowledge through probabilistic inference, addressing limitations under limited-sample regimes. By treating model capabilities as latent variables and leveraging a curated query set to induce discriminative responses, we formalize model ranking as a Bayesian hypothesis testing problem over mutually exclusive capability intervals. Experimental evaluations with GPT-series models demonstrate that the proposed method achieves superior discrimination compared to conventional evaluation methods. Results indicate that even with reduced sample sizes, the approach maintains statistical robustness while providing actionable insights, such as probabilistic statements about a model's likelihood of surpassing specific baselines. This work advances LLM evaluation methodologies by bridging Bayesian inference with practical constraints in real-world deployment scenarios. 

---
# BiasGuard: A Reasoning-enhanced Bias Detection Tool For Large Language Models 

**Authors**: Zhiting Fan, Ruizhe Chen, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21299)  

**Abstract**: Identifying bias in LLM-generated content is a crucial prerequisite for ensuring fairness in LLMs. Existing methods, such as fairness classifiers and LLM-based judges, face limitations related to difficulties in understanding underlying intentions and the lack of criteria for fairness judgment. In this paper, we introduce BiasGuard, a novel bias detection tool that explicitly analyzes inputs and reasons through fairness specifications to provide accurate judgments. BiasGuard is implemented through a two-stage approach: the first stage initializes the model to explicitly reason based on fairness specifications, while the second stage leverages reinforcement learning to enhance its reasoning and judgment capabilities. Our experiments, conducted across five datasets, demonstrate that BiasGuard outperforms existing tools, improving accuracy and reducing over-fairness misjudgments. We also highlight the importance of reasoning-enhanced decision-making and provide evidence for the effectiveness of our two-stage optimization pipeline. 

---
# Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math 

**Authors**: Haoran Xu, Baolin Peng, Hany Awadalla, Dongdong Chen, Yen-Chun Chen, Mei Gao, Young Jin Kim, Yunsheng Li, Liliang Ren, Yelong Shen, Shuohang Wang, Weijian Xu, Jianfeng Gao, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21233)  

**Abstract**: Chain-of-Thought (CoT) significantly enhances formal reasoning capabilities in Large Language Models (LLMs) by training them to explicitly generate intermediate reasoning steps. While LLMs readily benefit from such techniques, improving reasoning in Small Language Models (SLMs) remains challenging due to their limited model capacity. Recent work by Deepseek-R1 demonstrates that distillation from LLM-generated synthetic data can substantially improve the reasoning ability of SLM. However, the detailed modeling recipe is not disclosed. In this work, we present a systematic training recipe for SLMs that consists of four steps: (1) large-scale mid-training on diverse distilled long-CoT data, (2) supervised fine-tuning on high-quality long-CoT data, (3) Rollout DPO leveraging a carefully curated preference dataset, and (4) Reinforcement Learning (RL) with Verifiable Reward. We apply our method on Phi-4-Mini, a compact 3.8B-parameter model. The resulting Phi-4-Mini-Reasoning model exceeds, on math reasoning tasks, much larger reasoning models, e.g., outperforming DeepSeek-R1-Distill-Qwen-7B by 3.2 points and DeepSeek-R1-Distill-Llama-8B by 7.7 points on Math-500. Our results validate that a carefully designed training recipe, with large-scale high-quality CoT data, is effective to unlock strong reasoning capabilities even in resource-constrained small models. 

---
# Detecting Manipulated Contents Using Knowledge-Grounded Inference 

**Authors**: Mark Huasong Meng, Ruizhe Wang, Meng Xu, Chuan Yan, Guangdong Bai  

**Link**: [PDF](https://arxiv.org/pdf/2504.21165)  

**Abstract**: The detection of manipulated content, a prevalent form of fake news, has been widely studied in recent years. While existing solutions have been proven effective in fact-checking and analyzing fake news based on historical events, the reliance on either intrinsic knowledge obtained during training or manually curated context hinders them from tackling zero-day manipulated content, which can only be recognized with real-time contextual information. In this work, we propose Manicod, a tool designed for detecting zero-day manipulated content. Manicod first sources contextual information about the input claim from mainstream search engines, and subsequently vectorizes the context for the large language model (LLM) through retrieval-augmented generation (RAG). The LLM-based inference can produce a "truthful" or "manipulated" decision and offer a textual explanation for the decision. To validate the effectiveness of Manicod, we also propose a dataset comprising 4270 pieces of manipulated fake news derived from 2500 recent real-world news headlines. Manicod achieves an overall F1 score of 0.856 on this dataset and outperforms existing methods by up to 1.9x in F1 score on their benchmarks on fact-checking and claim verification. 

---
# LLM Enhancer: Merged Approach using Vector Embedding for Reducing Large Language Model Hallucinations with External Knowledge 

**Authors**: Naheed Rayhan, Md. Ashrafuzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2504.21132)  

**Abstract**: Large Language Models (LLMs), such as ChatGPT, have demonstrated the capability to generate human like, natural responses across a range of tasks, including task oriented dialogue and question answering. However, their application in real world, critical scenarios is often hindered by a tendency to produce inaccurate information and a limited ability to leverage external knowledge sources. This paper introduces the LLM ENHANCER system, designed to integrate multiple online sources such as Google, Wikipedia, and DuckDuckGo to enhance data accuracy. The LLMs employed within this system are open source. The data acquisition process for the LLM ENHANCER system operates in parallel, utilizing custom agent tools to manage the flow of information. Vector embeddings are used to identify the most pertinent information, which is subsequently supplied to the LLM for user interaction. The LLM ENHANCER system mitigates hallucinations in chat based LLMs while preserving response naturalness and accuracy. 

---
# Beyond One-Size-Fits-All: Inversion Learning for Highly Effective NLG Evaluation Prompts 

**Authors**: Hanhua Hong, Chenghao Xiao, Yang Wang, Yiqi Liu, Wenge Rong, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.21117)  

**Abstract**: Evaluating natural language generation (NLG) systems is challenging due to the diversity of valid outputs. While human evaluation is the gold standard, it suffers from inconsistencies, lack of standardisation, and demographic biases, limiting reproducibility. LLM-based evaluation offers a scalable alternative but is highly sensitive to prompt design, where small variations can lead to significant discrepancies. In this work, we propose an inversion learning method that learns effective reverse mappings from model outputs back to their input instructions, enabling the automatic generation of highly effective, model-specific evaluation prompts. Our method requires only a single evaluation sample and eliminates the need for time-consuming manual prompt engineering, thereby improving both efficiency and robustness. Our work contributes toward a new direction for more robust and efficient LLM-based evaluation. 

---
# WebEvolver: Enhancing Web Agent Self-Improvement with Coevolving World Model 

**Authors**: Tianqing Fang, Hongming Zhang, Zhisong Zhang, Kaixin Ma, Wenhao Yu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21024)  

**Abstract**: Agent self-improvement, where the backbone Large Language Model (LLM) of the agent are trained on trajectories sampled autonomously based on their own policies, has emerged as a promising approach for enhancing performance. Recent advancements, particularly in web environments, face a critical limitation: their performance will reach a stagnation point during autonomous learning cycles, hindering further improvement. We argue that this stems from limited exploration of the web environment and insufficient exploitation of pre-trained web knowledge in LLMs. To improve the performance of self-improvement, we propose a novel framework that introduces a co-evolving World Model LLM. This world model predicts the next observation based on the current observation and action within the web environment. Leveraging LLMs' pretrained knowledge of abundant web content, the World Model serves dual roles: (1) as a virtual web server generating self-instructed training data to continuously refine the agent's policy, and (2) as an imagination engine during inference, enabling look-ahead simulation to guide action selection for the agent LLM. Experiments in real-world web environments (Mind2Web-Live, WebVoyager, and GAIA-web) show a 10% performance gain over existing self-evolving agents, demonstrating the efficacy and generalizability of our approach, without using any distillation from more powerful close-sourced models. Our work establishes the necessity of integrating world models into autonomous agent frameworks to unlock sustained adaptability. 

---
# Does the Prompt-based Large Language Model Recognize Students' Demographics and Introduce Bias in Essay Scoring? 

**Authors**: Kaixun Yang, Mladen Raković, Dragan Gašević, Guanliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21330)  

**Abstract**: Large Language Models (LLMs) are widely used in Automated Essay Scoring (AES) due to their ability to capture semantic meaning. Traditional fine-tuning approaches required technical expertise, limiting accessibility for educators with limited technical backgrounds. However, prompt-based tools like ChatGPT have made AES more accessible, enabling educators to obtain machine-generated scores using natural-language prompts (i.e., the prompt-based paradigm). Despite advancements, prior studies have shown bias in fine-tuned LLMs, particularly against disadvantaged groups. It remains unclear whether such biases persist or are amplified in the prompt-based paradigm with cutting-edge tools. Since such biases are believed to stem from the demographic information embedded in pre-trained models (i.e., the ability of LLMs' text embeddings to predict demographic attributes), this study explores the relationship between the model's predictive power of students' demographic attributes based on their written works and its predictive bias in the scoring task in the prompt-based paradigm. Using a publicly available dataset of over 25,000 students' argumentative essays, we designed prompts to elicit demographic inferences (i.e., gender, first-language background) from GPT-4o and assessed fairness in automated scoring. Then we conducted multivariate regression analysis to explore the impact of the model's ability to predict demographics on its scoring outcomes. Our findings revealed that (i) prompt-based LLMs can somewhat infer students' demographics, particularly their first-language backgrounds, from their essays; (ii) scoring biases are more pronounced when the LLM correctly predicts students' first-language background than when it does not; and (iii) scoring error for non-native English speakers increases when the LLM correctly identifies them as non-native. 

---
# Durghotona GPT: A Web Scraping and Large Language Model Based Framework to Generate Road Accident Dataset Automatically in Bangladesh 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain, Md. Ridwanul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2504.21025)  

**Abstract**: Road accidents pose significant concerns globally. They lead to large financial losses, injuries, disabilities, and societal challenges. Accurate and timely accident data is essential for predicting and mitigating these events. This paper presents a novel framework named 'Durghotona GPT' that integrates web scraping and Large Language Models (LLMs) to automate the generation of comprehensive accident datasets from prominent national dailies in Bangladesh. The authors collected accident reports from three major newspapers: Prothom Alo, Dhaka Tribune, and The Daily Star. The collected news was then processed using the newest available LLMs: GPT-4, GPT-3.5, and Llama-3. The framework efficiently extracts relevant information, categorizes reports, and compiles detailed datasets. Thus, this framework overcomes limitations of manual data collection methods such as delays, errors, and communication gaps. The authors' evaluation demonstrates that Llama-3, an open-source model, performs comparably to GPT-4. It achieved 89% accuracy in the authors' evaluation. Therefore, it can be considered a cost-effective alternative for similar tasks. The results suggest that the framework developed by the authors can drastically enhance the quality and availability of accident data. As a result, it can support critical applications in traffic safety analysis, urban planning, and public health. The authors also developed an interface for 'Durghotona GPT' for ease of use as part of this paper. Future work will focus on expanding data collection methods and refining LLMs to further increase dataset accuracy and applicability. 

---
# Who Gets the Callback? Generative AI and Gender Bias 

**Authors**: Sugat Chaturvedi, Rochana Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21400)  

**Abstract**: Generative artificial intelligence (AI), particularly large language models (LLMs), is being rapidly deployed in recruitment and for candidate shortlisting. We audit several mid-sized open-source LLMs for gender bias using a dataset of 332,044 real-world online job postings. For each posting, we prompt the model to recommend whether an equally qualified male or female candidate should receive an interview callback. We find that most models tend to favor men, especially for higher-wage roles. Mapping job descriptions to the Standard Occupational Classification system, we find lower callback rates for women in male-dominated occupations and higher rates in female-associated ones, indicating occupational segregation. A comprehensive analysis of linguistic features in job ads reveals strong alignment of model recommendations with traditional gender stereotypes. To examine the role of recruiter identity, we steer model behavior by infusing Big Five personality traits and simulating the perspectives of historical figures. We find that less agreeable personas reduce stereotyping, consistent with an agreeableness bias in LLMs. Our findings highlight how AI-driven hiring may perpetuate biases in the labor market and have implications for fairness and diversity within firms. 

---
