# Neuro-Symbolic Concepts 

**Authors**: Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06191)  

**Abstract**: This article presents a concept-centric paradigm for building agents that can learn continually and reason flexibly. The concept-centric agent utilizes a vocabulary of neuro-symbolic concepts. These concepts, such as object, relation, and action concepts, are grounded on sensory inputs and actuation outputs. They are also compositional, allowing for the creation of novel concepts through their structural combination. To facilitate learning and reasoning, the concepts are typed and represented using a combination of symbolic programs and neural network representations. Leveraging such neuro-symbolic concepts, the agent can efficiently learn and recombine them to solve various tasks across different domains, ranging from 2D images, videos, 3D scenes, and robotic manipulation tasks. This concept-centric framework offers several advantages, including data efficiency, compositional generalization, continual learning, and zero-shot transfer. 

---
# Free and Fair Hardware: A Pathway to Copyright Infringement-Free Verilog Generation using LLMs 

**Authors**: Sam Bush, Matthew DeLorenzo, Phat Tieu, Jeyavijayan Rajendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.06096)  

**Abstract**: Limitations in Large Language Model (LLM) capabilities for hardware design tasks, such as generating functional Verilog codes, have motivated various fine-tuning optimizations utilizing curated hardware datasets from open-source repositories. However, these datasets remain limited in size and contain minimal checks on licensing for reuse, resulting in potential copyright violations by fine-tuned LLMs. Therefore, we propose an evaluation benchmark to estimate the risk of Verilog-trained LLMs to generate copyright-protected codes. To minimize this risk, we present an open-source Verilog dataset, FreeSet, containing over 220k files, along with the automated dataset curation framework utilized to provide additional guarantees of fair-use Verilog data. We then execute an LLM fine-tuning framework consisting of continual pre-training, resulting in a fine-tuned Llama model for Verilog, FreeV. Our results indicate that FreeV demonstrates the smallest risk of copyright-infringement among prior works, with only a 3% violation rate. Furthermore, experimental results demonstrate improvements in Verilog generation functionality over its baseline model, improving VerilogEval pass@10 rates by over 10%. 

---
# Seqret: Mining Rule Sets from Event Sequences 

**Authors**: Aleena Siji, Joscha Cüppers, Osman Ali Mian, Jilles Vreeken  

**Link**: [PDF](https://arxiv.org/pdf/2505.06049)  

**Abstract**: Summarizing event sequences is a key aspect of data mining. Most existing methods neglect conditional dependencies and focus on discovering sequential patterns only. In this paper, we study the problem of discovering both conditional and unconditional dependencies from event sequence data. We do so by discovering rules of the form $X \rightarrow Y$ where $X$ and $Y$ are sequential patterns. Rules like these are simple to understand and provide a clear description of the relation between the antecedent and the consequent. To discover succinct and non-redundant sets of rules we formalize the problem in terms of the Minimum Description Length principle. As the search space is enormous and does not exhibit helpful structure, we propose the Seqret method to discover high-quality rule sets in practice. Through extensive empirical evaluation we show that unlike the state of the art, Seqret ably recovers the ground truth on synthetic datasets and finds useful rules from real datasets. 

---
# Why Are You Wrong? Counterfactual Explanations for Language Grounding with 3D Objects 

**Authors**: Tobias Preintner, Weixuan Yuan, Qi Huang, Adrian König, Thomas Bäck, Elena Raponi, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2505.06030)  

**Abstract**: Combining natural language and geometric shapes is an emerging research area with multiple applications in robotics and language-assisted design. A crucial task in this domain is object referent identification, which involves selecting a 3D object given a textual description of the target. Variability in language descriptions and spatial relationships of 3D objects makes this a complex task, increasing the need to better understand the behavior of neural network models in this domain. However, limited research has been conducted in this area. Specifically, when a model makes an incorrect prediction despite being provided with a seemingly correct object description, practitioners are left wondering: "Why is the model wrong?". In this work, we present a method answering this question by generating counterfactual examples. Our method takes a misclassified sample, which includes two objects and a text description, and generates an alternative yet similar formulation that would have resulted in a correct prediction by the model. We have evaluated our approach with data from the ShapeTalk dataset along with three distinct models. Our counterfactual examples maintain the structure of the original description, are semantically similar and meaningful. They reveal weaknesses in the description, model bias and enhance the understanding of the models behavior. Theses insights help practitioners to better interact with systems as well as engineers to improve models. 

---
# ArtRAG: Retrieval-Augmented Generation with Structured Context for Visual Art Understanding 

**Authors**: Shuai Wang, Ivona Najdenkoska, Hongyi Zhu, Stevan Rudinac, Monika Kackovic, Nachoem Wijnberg, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2505.06020)  

**Abstract**: Understanding visual art requires reasoning across multiple perspectives -- cultural, historical, and stylistic -- beyond mere object recognition. While recent multimodal large language models (MLLMs) perform well on general image captioning, they often fail to capture the nuanced interpretations that fine art demands. We propose ArtRAG, a novel, training-free framework that combines structured knowledge with retrieval-augmented generation (RAG) for multi-perspective artwork explanation. ArtRAG automatically constructs an Art Context Knowledge Graph (ACKG) from domain-specific textual sources, organizing entities such as artists, movements, themes, and historical events into a rich, interpretable graph. At inference time, a multi-granular structured retriever selects semantically and topologically relevant subgraphs to guide generation. This enables MLLMs to produce contextually grounded, culturally informed art descriptions. Experiments on the SemArt and Artpedia datasets show that ArtRAG outperforms several heavily trained baselines. Human evaluations further confirm that ArtRAG generates coherent, insightful, and culturally enriched interpretations. 

---
# Pseudo-Boolean d-DNNF Compilation for Expressive Feature Modeling Constructs 

**Authors**: Chico Sundermann, Stefan Vill, Elias Kuiter, Sebastian Krieter, Thomas Thüm, Matthias Tichy  

**Link**: [PDF](https://arxiv.org/pdf/2505.05976)  

**Abstract**: Configurable systems typically consist of reusable assets that have dependencies between each other. To specify such dependencies, feature models are commonly used. As feature models in practice are often complex, automated reasoning is typically employed to analyze the dependencies. Here, the de facto standard is translating the feature model to conjunctive normal form (CNF) to enable employing off-the-shelf tools, such as SAT or #SAT solvers. However, modern feature-modeling dialects often contain constructs, such as cardinality constraints, that are ill-suited for conversion to CNF. This mismatch between the input of reasoning engines and the available feature-modeling dialects limits the applicability of the more expressive constructs. In this work, we shorten this gap between expressive constructs and scalable automated reasoning. Our contribution is twofold: First, we provide a pseudo-Boolean encoding for feature models, which facilitates smaller representations of commonly employed constructs compared to Boolean encoding. Second, we propose a novel method to compile pseudo-Boolean formulas to Boolean d-DNNF. With the compiled d-DNNFs, we can resort to a plethora of efficient analyses already used in feature modeling. Our empirical evaluation shows that our proposal substantially outperforms the state-of-the-art based on CNF inputs for expressive constructs. For every considered dataset representing different feature models and feature-modeling constructs, the feature models can be significantly faster translated to pseudo-Boolean than to CNF. Overall, deriving d-DNNFs from a feature model with the targeted expressive constraints can be substantially accelerated using our pseudo-Boolean approach. Furthermore, our approach is competitive on feature models with only basic constructs. 

---
# Combining Abstract Argumentation and Machine Learning for Efficiently Analyzing Low-Level Process Event Streams 

**Authors**: Bettina Fazzinga, Sergio Flesca, Filippo Furfaro, Luigi Pontieri, Francesco Scala  

**Link**: [PDF](https://arxiv.org/pdf/2505.05880)  

**Abstract**: Monitoring and analyzing process traces is a critical task for modern companies and organizations. In scenarios where there is a gap between trace events and reference business activities, this entails an interpretation problem, amounting to translating each event of any ongoing trace into the corresponding step of the activity instance. Building on a recent approach that frames the interpretation problem as an acceptance problem within an Abstract Argumentation Framework (AAF), one can elegantly analyze plausible event interpretations (possibly in an aggregated form), as well as offer explanations for those that conflict with prior process knowledge. Since, in settings where event-to-activity mapping is highly uncertain (or simply under-specified) this reasoning-based approach may yield lowly-informative results and heavy computation, one can think of discovering a sequencetagging model, trained to suggest highly-probable candidate event interpretations in a context-aware way. However, training such a model optimally may require using a large amount of manually-annotated example traces. Considering the urgent need of developing Green AI solutions enabling environmental and societal sustainability (with reduced labor/computational costs and carbon footprint), we propose a data/computation-efficient neuro-symbolic approach to the problem, where the candidate interpretations returned by the example-driven sequence tagger is refined by the AAF-based reasoner. This allows us to also leverage prior knowledge to compensate for the scarcity of example data, as confirmed by experimental results; clearly, this property is particularly useful in settings where data annotation and model optimization costs are subject to stringent constraints. 

---
# APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning 

**Authors**: Azim Ospanov, Roozbeh Yousefzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.05758)  

**Abstract**: Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving. 

---
# Pretraining a Shared Q-Network for Data-Efficient Offline Reinforcement Learning 

**Authors**: Jongchan Park, Mingyu Park, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.05701)  

**Abstract**: Offline reinforcement learning (RL) aims to learn a policy from a static dataset without further interactions with the environment. Collecting sufficiently large datasets for offline RL is exhausting since this data collection requires colossus interactions with environments and becomes tricky when the interaction with the environment is restricted. Hence, how an agent learns the best policy with a minimal static dataset is a crucial issue in offline RL, similar to the sample efficiency problem in online RL. In this paper, we propose a simple yet effective plug-and-play pretraining method to initialize a feature of a $Q$-network to enhance data efficiency in offline RL. Specifically, we introduce a shared $Q$-network structure that outputs predictions of the next state and $Q$-value. We pretrain the shared $Q$-network through a supervised regression task that predicts a next state and trains the shared $Q$-network using diverse offline RL methods. Through extensive experiments, we empirically demonstrate that our method enhances the performance of existing popular offline RL methods on the D4RL, Robomimic and V-D4RL benchmarks. Furthermore, we show that our method significantly boosts data-efficient offline RL across various data qualities and data distributions trough D4RL and ExoRL benchmarks. Notably, our method adapted with only 10% of the dataset outperforms standard algorithms even with full datasets. 

---
# Prompted Meta-Learning for Few-shot Knowledge Graph Completion 

**Authors**: Han Wu, Jie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05684)  

**Abstract**: Few-shot knowledge graph completion (KGC) has obtained significant attention due to its practical applications in real-world scenarios, where new knowledge often emerges with limited available data. While most existing methods for few-shot KGC have predominantly focused on leveraging relational information, rich semantics inherent in KGs have been largely overlooked. To address this gap, we propose a novel prompted meta-learning (PromptMeta) framework that seamlessly integrates meta-semantics with relational information for few-shot KGC. PrompMeta has two key innovations: (1) a meta-semantic prompt pool that captures and consolidates high-level meta-semantics, enabling effective knowledge transfer and adaptation to rare and newly emerging relations. (2) a learnable fusion prompt that dynamically combines meta-semantic information with task-specific relational information tailored to different few-shot tasks. Both components are optimized together with model parameters within a meta-learning framework. Extensive experiments on two benchmark datasets demonstrate the effectiveness of our approach. 

---
# Leveraging Large Language Models for enzymatic reaction prediction and characterization 

**Authors**: Lorenzo Di Fruscia, Jana Marie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2505.05616)  

**Abstract**: Predicting enzymatic reactions is crucial for applications in biocatalysis, metabolic engineering, and drug discovery, yet it remains a complex and resource-intensive task. Large Language Models (LLMs) have recently demonstrated remarkable success in various scientific domains, e.g., through their ability to generalize knowledge, reason over complex structures, and leverage in-context learning strategies. In this study, we systematically evaluate the capability of LLMs, particularly the Llama-3.1 family (8B and 70B), across three core biochemical tasks: Enzyme Commission number prediction, forward synthesis, and retrosynthesis. We compare single-task and multitask learning strategies, employing parameter-efficient fine-tuning via LoRA adapters. Additionally, we assess performance across different data regimes to explore their adaptability in low-data settings. Our results demonstrate that fine-tuned LLMs capture biochemical knowledge, with multitask learning enhancing forward- and retrosynthesis predictions by leveraging shared enzymatic information. We also identify key limitations, for example challenges in hierarchical EC classification schemes, highlighting areas for further improvement in LLM-driven biochemical modeling. 

---
# scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction 

**Authors**: Qing Wang, Yining Pan, Minghao Zhou, Zijia Tang, Yanfei Wang, Guangyu Wang, Qianqian Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.05612)  

**Abstract**: Drug resistance presents a major challenge in cancer therapy. Single cell profiling offers insights into cellular heterogeneity, yet the application of large-scale foundation models for predicting drug response in single cell data remains underexplored. To address this, we developed scDrugMap, an integrated framework featuring both a Python command-line interface and a web server for drug response prediction. scDrugMap evaluates a wide range of foundation models, including eight single-cell models and two large language models, using a curated dataset of over 326,000 cells in the primary collection and 18,800 cells in the validation set, spanning 36 datasets and diverse tissue and cancer types. We benchmarked model performance under pooled-data and cross-data evaluation settings, employing both layer freezing and Low-Rank Adaptation (LoRA) fine-tuning strategies. In the pooled-data scenario, scFoundation achieved the best performance, with mean F1 scores of 0.971 (layer freezing) and 0.947 (fine-tuning), outperforming the lowest-performing model by over 50%. In the cross-data setting, UCE excelled post fine-tuning (mean F1: 0.774), while scGPT led in zero-shot learning (mean F1: 0.858). Overall, scDrugMap provides the first large-scale benchmark of foundation models for drug response prediction in single-cell data and serves as a user-friendly, flexible platform for advancing drug discovery and translational research. 

---
# HiBayES: A Hierarchical Bayesian Modeling Framework for AI Evaluation Statistics 

**Authors**: Lennart Luettgau, Harry Coppock, Magda Dubois, Christopher Summerfield, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2505.05602)  

**Abstract**: As Large Language Models (LLMs) and other AI systems evolve, robustly estimating their capabilities from inherently stochastic outputs while systematically quantifying uncertainty in these estimates becomes increasingly important. Further, advanced AI evaluations often have a nested hierarchical structure, exhibit high levels of complexity, and come with high costs in testing the most advanced AI systems. To address these challenges, we introduce HiBayES, a generalizable Hierarchical Bayesian modeling framework for AI Evaluation Statistics. HiBayES supports robust inferences in classical question-answer benchmarks and advanced agentic evaluations, particularly in low-data scenarios (e.g., < 20 data points per evaluation). Built on Generalized Linear Models (GLMs), Bayesian data analysis, and formal model comparison, HiBayES provides principled uncertainty quantification and robust parameter estimation. This paper offers a comprehensive introduction to HiBayES, including illustrative examples, comparisons to conventional statistical methods, and practical guidance for implementing multilevel Bayesian GLMs. Additionally, we provide a HiBayES software package [4] (Beta version) for out-of-the-box implementation. 

---
# Safety by Measurement: A Systematic Literature Review of AI Safety Evaluation Methods 

**Authors**: Markov Grey, Charbel-Raphaël Segerie  

**Link**: [PDF](https://arxiv.org/pdf/2505.05541)  

**Abstract**: As frontier AI systems advance toward transformative capabilities, we need a parallel transformation in how we measure and evaluate these systems to ensure safety and inform governance. While benchmarks have been the primary method for estimating model capabilities, they often fail to establish true upper bounds or predict deployment behavior. This literature review consolidates the rapidly evolving field of AI safety evaluations, proposing a systematic taxonomy around three dimensions: what properties we measure, how we measure them, and how these measurements integrate into frameworks. We show how evaluations go beyond benchmarks by measuring what models can do when pushed to the limit (capabilities), the behavioral tendencies exhibited by default (propensities), and whether our safety measures remain effective even when faced with subversive adversarial AI (control). These properties are measured through behavioral techniques like scaffolding, red teaming and supervised fine-tuning, alongside internal techniques such as representation analysis and mechanistic interpretability. We provide deeper explanations of some safety-critical capabilities like cybersecurity exploitation, deception, autonomous replication, and situational awareness, alongside concerning propensities like power-seeking and scheming. The review explores how these evaluation methods integrate into governance frameworks to translate results into concrete development decisions. We also highlight challenges to safety evaluations - proving absence of capabilities, potential model sandbagging, and incentives for "safetywashing" - while identifying promising research directions. By synthesizing scattered resources, this literature review aims to provide a central reference point for understanding AI safety evaluations. 

---
# Let Humanoids Hike! Integrative Skill Development on Complex Trails 

**Authors**: Kwan-Yee Lin, Stella X.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06218)  

**Abstract**: Hiking on complex trails demands balance, agility, and adaptive decision-making over unpredictable terrain. Current humanoid research remains fragmented and inadequate for hiking: locomotion focuses on motor skills without long-term goals or situational awareness, while semantic navigation overlooks real-world embodiment and local terrain variability. We propose training humanoids to hike on complex trails, driving integrative skill development across visual perception, decision making, and motor execution. We develop a learning framework, LEGO-H, that enables a vision-equipped humanoid robot to hike complex trails autonomously. We introduce two technical innovations: 1) A temporal vision transformer variant - tailored into Hierarchical Reinforcement Learning framework - anticipates future local goals to guide movement, seamlessly integrating locomotion with goal-directed navigation. 2) Latent representations of joint movement patterns, combined with hierarchical metric learning - enhance Privileged Learning scheme - enable smooth policy transfer from privileged training to onboard execution. These components allow LEGO-H to handle diverse physical and environmental challenges without relying on predefined motion patterns. Experiments across varied simulated trails and robot morphologies highlight LEGO-H's versatility and robustness, positioning hiking as a compelling testbed for embodied autonomy and LEGO-H as a baseline for future humanoid development. 

---
# Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies 

**Authors**: Massimiliano Pronesti, Joao Bettencourt-Silva, Paul Flanagan, Alessandra Pascale, Oisin Redmond, Anya Belz, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06186)  

**Abstract**: Extracting scientific evidence from biomedical studies for clinical research questions (e.g., Does stem cell transplantation improve quality of life in patients with medically refractory Crohn's disease compared to placebo?) is a crucial step in synthesising biomedical evidence. In this paper, we focus on the task of document-level scientific evidence extraction for clinical questions with conflicting evidence. To support this task, we create a dataset called CochraneForest, leveraging forest plots from Cochrane systematic reviews. It comprises 202 annotated forest plots, associated clinical research questions, full texts of studies, and study-specific conclusions. Building on CochraneForest, we propose URCA (Uniform Retrieval Clustered Augmentation), a retrieval-augmented generation framework designed to tackle the unique challenges of evidence extraction. Our experiments show that URCA outperforms the best existing methods by up to 10.3% in F1 score on this task. However, the results also underscore the complexity of CochraneForest, establishing it as a challenging testbed for advancing automated evidence synthesis systems. 

---
# Turbo-ICL: In-Context Learning-Based Turbo Equalization 

**Authors**: Zihang Song, Matteo Zecchin, Bipin Rajendran, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2505.06175)  

**Abstract**: This paper introduces a novel in-context learning (ICL) framework, inspired by large language models (LLMs), for soft-input soft-output channel equalization in coded multiple-input multiple-output (MIMO) systems. The proposed approach learns to infer posterior symbol distributions directly from a prompt of pilot signals and decoder feedback. A key innovation is the use of prompt augmentation to incorporate extrinsic information from the decoder output as additional context, enabling the ICL model to refine its symbol estimates iteratively across turbo decoding iterations. Two model variants, based on Transformer and state-space architectures, are developed and evaluated. Extensive simulations demonstrate that, when traditional linear assumptions break down, e.g., in the presence of low-resolution quantization, ICL equalizers consistently outperform conventional model-based baselines, even when the latter are provided with perfect channel state information. Results also highlight the advantage of Transformer-based models under limited training diversity, as well as the efficiency of state-space models in resource-constrained scenarios. 

---
# MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks 

**Authors**: Wenqi Zeng, Yuqi Sun, Chenxi Ma, Weimin Tan, Bo Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06152)  

**Abstract**: Medical vision-language models (VLMs) have shown promise as clinical assistants across various medical fields. However, specialized dermatology VLM capable of delivering professional and detailed diagnostic analysis remains underdeveloped, primarily due to less specialized text descriptions in current dermatology multimodal datasets. To address this issue, we propose MM-Skin, the first large-scale multimodal dermatology dataset that encompasses 3 imaging modalities, including clinical, dermoscopic, and pathological and nearly 10k high-quality image-text pairs collected from professional textbooks. In addition, we generate over 27k diverse, instruction-following vision question answering (VQA) samples (9 times the size of current largest dermatology VQA dataset). Leveraging public datasets and MM-Skin, we developed SkinVL, a dermatology-specific VLM designed for precise and nuanced skin disease interpretation. Comprehensive benchmark evaluations of SkinVL on VQA, supervised fine-tuning (SFT) and zero-shot classification tasks across 8 datasets, reveal its exceptional performance for skin diseases in comparison to both general and medical VLM models. The introduction of MM-Skin and SkinVL offers a meaningful contribution to advancing the development of clinical dermatology VLM assistants. MM-Skin is available at this https URL 

---
# A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets 

**Authors**: Ryan Lagasse, Aidan Kiernans, Avijit Ghosh, Shiri Dori-Hacohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06150)  

**Abstract**: We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings. 

---
# Efficient Sensorimotor Learning for Open-world Robot Manipulation 

**Authors**: Yifeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06136)  

**Abstract**: This dissertation considers Open-world Robot Manipulation, a manipulation problem where a robot must generalize or quickly adapt to new objects, scenes, or tasks for which it has not been pre-programmed or pre-trained. This dissertation tackles the problem using a methodology of efficient sensorimotor learning. The key to enabling efficient sensorimotor learning lies in leveraging regular patterns that exist in limited amounts of demonstration data. These patterns, referred to as ``regularity,'' enable the data-efficient learning of generalizable manipulation skills. This dissertation offers a new perspective on formulating manipulation problems through the lens of regularity. Building upon this notion, we introduce three major contributions. First, we introduce methods that endow robots with object-centric priors, allowing them to learn generalizable, closed-loop sensorimotor policies from a small number of teleoperation demonstrations. Second, we introduce methods that constitute robots' spatial understanding, unlocking their ability to imitate manipulation skills from in-the-wild video observations. Last but not least, we introduce methods that enable robots to identify reusable skills from their past experiences, resulting in systems that can continually imitate multiple tasks in a sequential manner. Altogether, the contributions of this dissertation help lay the groundwork for building general-purpose personal robots that can quickly adapt to new situations or tasks with low-cost data collection and interact easily with humans. By enabling robots to learn and generalize from limited data, this dissertation takes a step toward realizing the vision of intelligent robotic assistants that can be seamlessly integrated into everyday scenarios. 

---
# Wasserstein Distances Made Explainable: Insights into Dataset Shifts and Transport Phenomena 

**Authors**: Philip Naumann, Jacob Kauffmann, Grégoire Montavon  

**Link**: [PDF](https://arxiv.org/pdf/2505.06123)  

**Abstract**: Wasserstein distances provide a powerful framework for comparing data distributions. They can be used to analyze processes over time or to detect inhomogeneities within data. However, simply calculating the Wasserstein distance or analyzing the corresponding transport map (or coupling) may not be sufficient for understanding what factors contribute to a high or low Wasserstein distance. In this work, we propose a novel solution based on Explainable AI that allows us to efficiently and accurately attribute Wasserstein distances to various data components, including data subgroups, input features, or interpretable subspaces. Our method achieves high accuracy across diverse datasets and Wasserstein distance specifications, and its practical utility is demonstrated in two use cases. 

---
# The Application of Deep Learning for Lymph Node Segmentation: A Systematic Review 

**Authors**: Jingguo Qu, Xinyang Han, Man-Lik Chui, Yao Pu, Simon Takadiyi Gunda, Ziman Chen, Jing Qin, Ann Dorothy King, Winnie Chiu-Wing Chu, Jing Cai, Michael Tin-Cheung Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.06118)  

**Abstract**: Automatic lymph node segmentation is the cornerstone for advances in computer vision tasks for early detection and staging of cancer. Traditional segmentation methods are constrained by manual delineation and variability in operator proficiency, limiting their ability to achieve high accuracy. The introduction of deep learning technologies offers new possibilities for improving the accuracy of lymph node image analysis. This study evaluates the application of deep learning in lymph node segmentation and discusses the methodologies of various deep learning architectures such as convolutional neural networks, encoder-decoder networks, and transformers in analyzing medical imaging data across different modalities. Despite the advancements, it still confronts challenges like the shape diversity of lymph nodes, the scarcity of accurately labeled datasets, and the inadequate development of methods that are robust and generalizable across different imaging modalities. To the best of our knowledge, this is the first study that provides a comprehensive overview of the application of deep learning techniques in lymph node segmentation task. Furthermore, this study also explores potential future research directions, including multimodal fusion techniques, transfer learning, and the use of large-scale pre-trained models to overcome current limitations while enhancing cancer diagnosis and treatment planning strategies. 

---
# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions 

**Authors**: Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06111)  

**Abstract**: A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge across different embodiments and environments. To confront these limitations, we propose UniVLA, a new framework for learning cross-embodiment vision-language-action (VLA) policies. Our key innovation is to derive task-centric action representations from videos with a latent action model. This enables us to exploit extensive data across a wide spectrum of embodiments and perspectives. To mitigate the effect of task-irrelevant dynamics, we incorporate language instructions and establish a latent action model within the DINO feature space. Learned from internet-scale videos, the generalist policy can be deployed to various robots through efficient latent action decoding. We obtain state-of-the-art results across multiple manipulation and navigation benchmarks, as well as real-robot deployments. UniVLA achieves superior performance over OpenVLA with less than 1/20 of pretraining compute and 1/10 of downstream data. Continuous performance improvements are observed as heterogeneous data, even including human videos, are incorporated into the training pipeline. The results underscore UniVLA's potential to facilitate scalable and efficient robot policy learning. 

---
# Multimodal Sentiment Analysis on CMU-MOSEI Dataset using Transformer-based Models 

**Authors**: Jugal Gajjar, Kaustik Ranaware  

**Link**: [PDF](https://arxiv.org/pdf/2505.06110)  

**Abstract**: This project performs multimodal sentiment analysis using the CMU-MOSEI dataset, using transformer-based models with early fusion to integrate text, audio, and visual modalities. We employ BERT-based encoders for each modality, extracting embeddings that are concatenated before classification. The model achieves strong performance, with 97.87\% 7-class accuracy and a 0.9682 F1-score on the test set, demonstrating the effectiveness of early fusion in capturing cross-modal interactions. The training utilized Adam optimization (lr=1e-4), dropout (0.3), and early stopping to ensure generalization and robustness. Results highlight the superiority of transformer architectures in modeling multimodal sentiment, with a low MAE (0.1060) indicating precise sentiment intensity prediction. Future work may compare fusion strategies or enhance interpretability. This approach utilizes multimodal learning by effectively combining linguistic, acoustic, and visual cues for sentiment analysis. 

---
# LLMs Outperform Experts on Challenging Biology Benchmarks 

**Authors**: Lennart Justen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06108)  

**Abstract**: This study systematically evaluates 27 frontier Large Language Models on eight diverse biology benchmarks spanning molecular biology, genetics, cloning, virology, and biosecurity. Models from major AI developers released between November 2022 and April 2025 were assessed through ten independent runs per benchmark. The findings reveal dramatic improvements in biological capabilities. Top model performance increased more than 4-fold on the challenging text-only subset of the Virology Capabilities Test over the study period, with the top model now performing twice as well as expert virologists. Several models now match or exceed expert-level performance on other challenging benchmarks, including LAB-Bench CloningScenarios and the biology subsets of GPQA and WMDP. Contrary to expectations, chain-of-thought did not substantially improve performance over zero-shot evaluation, while extended reasoning features in o3-mini and Claude 3.7 Sonnet typically improved performance as predicted by inference scaling. Benchmarks such as PubMedQA and the MMLU and WMDP biology subsets exhibited performance plateaus well below 100%, suggesting benchmark saturation and errors in the underlying benchmark data. The analysis highlights the need for more sophisticated evaluation methodologies as AI systems continue to advance. 

---
# UniSymNet: A Unified Symbolic Network Guided by Transformer 

**Authors**: Xinxin Li, Juan Zhang, Da Li, Xingyu Liu, Jin Xu, Junping Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06091)  

**Abstract**: Symbolic Regression (SR) is a powerful technique for automatically discovering mathematical expressions from input data. Mainstream SR algorithms search for the optimal symbolic tree in a vast function space, but the increasing complexity of the tree structure limits their performance. Inspired by neural networks, symbolic networks have emerged as a promising new paradigm. However, most existing symbolic networks still face certain challenges: binary nonlinear operators $\{\times, ÷\}$ cannot be naturally extended to multivariate operators, and training with fixed architecture often leads to higher complexity and overfitting. In this work, we propose a Unified Symbolic Network that unifies nonlinear binary operators into nested unary operators and define the conditions under which UniSymNet can reduce complexity. Moreover, we pre-train a Transformer model with a novel label encoding method to guide structural selection, and adopt objective-specific optimization strategies to learn the parameters of the symbolic network. UniSymNet shows high fitting accuracy, excellent symbolic solution rate, and relatively low expression complexity, achieving competitive performance on low-dimensional Standard Benchmarks and high-dimensional SRBench. 

---
# Assessing Tenstorrent's RISC-V MatMul Acceleration Capabilities 

**Authors**: Hiari Pizzini Cavagna, Daniele Cesarini, Andrea Bartolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.06085)  

**Abstract**: The increasing demand for generative AI as Large Language Models (LLMs) services has driven the need for specialized hardware architectures that optimize computational efficiency and energy consumption. This paper evaluates the performance of the Tenstorrent Grayskull e75 RISC-V accelerator for basic linear algebra kernels at reduced numerical precision, a fundamental operation in LLM computations. We present a detailed characterization of Grayskull's execution model, gridsize, matrix dimensions, data formats, and numerical precision impact computational efficiency. Furthermore, we compare Grayskull's performance against state-of-the-art architectures with tensor acceleration, including Intel Sapphire Rapids processors and two NVIDIA GPUs (V100 and A100). Whilst NVIDIA GPUs dominate raw performance, Grayskull demonstrates a competitive trade-off between power consumption and computational throughput, reaching a peak of 1.55 TFLOPs/Watt with BF16. 

---
# PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks 

**Authors**: Francesco Spinnato, Cristiano Landi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06047)  

**Abstract**: Irregular temporal data, characterized by varying recording frequencies, differing observation durations, and missing values, presents significant challenges across fields like mobility, healthcare, and environmental science. Existing research communities often overlook or address these challenges in isolation, leading to fragmented tools and methods. To bridge this gap, we introduce a unified framework, and the first standardized dataset repository for irregular time series classification, built on a common array format to enhance interoperability. This repository comprises 34 datasets on which we benchmark 12 classifier models from diverse domains and communities. This work aims to centralize research efforts and enable a more robust evaluation of irregular temporal data analysis methods. 

---
# Universal Approximation Theorem for Deep Q-Learning via FBSDE System 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06023)  

**Abstract**: The approximation capabilities of Deep Q-Networks (DQNs) are commonly justified by general Universal Approximation Theorems (UATs) that do not leverage the intrinsic structural properties of the optimal Q-function, the solution to a Bellman equation. This paper establishes a UAT for a class of DQNs whose architecture is designed to emulate the iterative refinement process inherent in Bellman updates. A central element of our analysis is the propagation of regularity: while the transformation induced by a single Bellman operator application exhibits regularity, for which Backward Stochastic Differential Equations (BSDEs) theory provides analytical tools, the uniform regularity of the entire sequence of value iteration iterates--specifically, their uniform Lipschitz continuity on compact domains under standard Lipschitz assumptions on the problem data--is derived from finite-horizon dynamic programming principles. We demonstrate that layers of a deep residual network, conceived as neural operators acting on function spaces, can approximate the action of the Bellman operator. The resulting approximation theorem is thus intrinsically linked to the control problem's structure, offering a proof technique wherein network depth directly corresponds to iterations of value function refinement, accompanied by controlled error propagation. This perspective reveals a dynamic systems view of the network's operation on a space of value functions. 

---
# Minimal Sequent Calculus for Teaching First-Order Logic: Lessons Learned 

**Authors**: Jørgen Villadsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05988)  

**Abstract**: MiniCalc is a web app for teaching first-order logic based on a minimal sequent calculus. As an option the proofs can be verified in the Isabelle proof assistant. We present the lessons learned using the tool in recent years at our university. 

---
# A Noise-Resilient Semi-Supervised Graph Autoencoder for Overlapping Semantic Community Detection 

**Authors**: Abdelfateh Bekkair, Slimane Bellaouar, Slimane Oulad-Naoui  

**Link**: [PDF](https://arxiv.org/pdf/2505.05965)  

**Abstract**: Community detection in networks with overlapping structures remains a significant challenge, particularly in noisy real-world environments where integrating topology, node attributes, and prior information is critical. To address this, we propose a semi-supervised graph autoencoder that combines graph multi-head attention and modularity maximization to robustly detect overlapping communities. The model learns semantic representations by fusing structural, attribute, and prior knowledge while explicitly addressing noise in node features. Key innovations include a noise-resistant architecture and a semantic semi-supervised design optimized for community quality through modularity constraints. Experiments demonstrate superior performance the model outperforms state-of-the-art methods in overlapping community detection (improvements in NMI and F1-score) and exhibits exceptional robustness to attribute noise, maintaining stable performance under 60\% feature corruption. These results highlight the importance of integrating attribute semantics and structural patterns for accurate community discovery in complex networks. 

---
# Elastic Weight Consolidation for Full-Parameter Continual Pre-Training of Gemma2 

**Authors**: Vytenis Šliogeris, Povilas Daniušis, Artūras Nakvosas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05946)  

**Abstract**: This technical report describes an experiment on autoregressive pre-training of Gemma2 2 billion parameter large language model (LLM) with 10\% on the Lithuanian language component of CulturaX from the point of view of continual learning. We apply elastic weight consolidation (EWC) to the full set of the model's parameters and investigate language understanding benchmarks, consisting of Arc, Belebele, Gsm8K, Hellaswag, MMLU, TruthfulQA, and Winogrande sets (both in English and Lithuanian versions), and perplexity benchmarks. We empirically demonstrate that EWC regularisation allows us not only to mitigate catastrophic forgetting effects but also that it is potentially beneficial for learning of the new task with LLMs. 

---
# Achieving 3D Attention via Triplet Squeeze and Excitation Block 

**Authors**: Maan Alhazmi, Abdulrahman Altahhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05943)  

**Abstract**: The emergence of ConvNeXt and its variants has reaffirmed the conceptual and structural suitability of CNN-based models for vision tasks, re-establishing them as key players in image classification in general, and in facial expression recognition (FER) in particular. In this paper, we propose a new set of models that build on these advancements by incorporating a new set of attention mechanisms that combines Triplet attention with Squeeze-and-Excitation (TripSE) in four different variants. We demonstrate the effectiveness of these variants by applying them to the ResNet18, DenseNet and ConvNext architectures to validate their versatility and impact. Our study shows that incorporating a TripSE block in these CNN models boosts their performances, particularly for the ConvNeXt architecture, indicating its utility. We evaluate the proposed mechanisms and associated models across four datasets, namely CIFAR100, ImageNet, FER2013 and AffectNet datasets, where ConvNext with TripSE achieves state-of-the-art results with an accuracy of \textbf{78.27\%} on the popular FER2013 dataset, a new feat for this dataset. 

---
# IRNN: Innovation-driven Recurrent Neural Network for Time-Series Data Modeling and Prediction 

**Authors**: Yifan Zhou, Yibo Wang, Chao Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05916)  

**Abstract**: Many real-world datasets are time series that are sequentially collected and contain rich temporal information. Thus, a common interest in practice is to capture dynamics of time series and predict their future evolutions. To this end, the recurrent neural network (RNN) has been a prevalent and effective machine learning option, which admits a nonlinear state-space model representation. Motivated by the resemblance between RNN and Kalman filter (KF) for linear state-space models, we propose in this paper Innovation-driven RNN (IRNN), a novel RNN architecture tailored to time-series data modeling and prediction tasks. By adapting the concept of "innovation" from KF to RNN, past prediction errors are adopted as additional input signals to update hidden states of RNN and boost prediction performance. Since innovation data depend on network parameters, existing training algorithms for RNN do not apply to IRNN straightforwardly. Thus, a tailored training algorithm dubbed input updating-based back-propagation through time (IU-BPTT) is further proposed, which alternates between updating innovations and optimizing network parameters via gradient descent. Experiments on real-world benchmark datasets show that the integration of innovations into various forms of RNN leads to remarkably improved prediction accuracy of IRNN without increasing the training cost substantially. 

---
# Examining the Source of Defects from a Mechanical Perspective for 3D Anomaly Detection 

**Authors**: Hanzhe Liang, Aoran Wang, Jie Zhou, Xin Jin, Can Gao, Jinbao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05901)  

**Abstract**: In this paper, we go beyond identifying anomalies only in structural terms and think about better anomaly detection motivated by anomaly causes. Most anomalies are regarded as the result of unpredictable defective forces from internal and external sources, and their opposite forces are sought to correct the anomalies. We introduced a Mechanics Complementary framework for 3D anomaly detection (MC4AD) to generate internal and external Corrective forces for each point. A Diverse Anomaly-Generation (DA-Gen) module is first proposed to simulate various anomalies. Then, we present a Corrective Force Prediction Network (CFP-Net) with complementary representations for point-level representation to simulate the different contributions of internal and external corrective forces. A combined loss was proposed, including a new symmetric loss and an overall loss, to constrain the corrective forces properly. As a highlight, we consider 3D anomaly detection in industry more comprehensively, creating a hierarchical quality control strategy based on a three-way decision and contributing a dataset named Anomaly-IntraVariance with intraclass variance to evaluate the model. On the proposed and existing five datasets, we obtained nine state-of-the-art performers with the minimum parameters and the fastest inference speed. The source is available at this https URL 

---
# Leveraging Vision-Language Models for Visual Grounding and Analysis of Automotive UI 

**Authors**: Benjamin Raphael Ernhofer, Daniil Prokhorov, Jannica Langner, Dominik Bollmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.05895)  

**Abstract**: Modern automotive infotainment systems require intelligent and adaptive solutions to handle frequent User Interface (UI) updates and diverse design variations. We introduce a vision-language framework for understanding and interacting with automotive infotainment systems, enabling seamless adaptation across different UI designs. To further support research in this field, we release AutomotiveUI-Bench-4K, an open-source dataset of 998 images with 4,208 annotations. Additionally, we present a synthetic data pipeline to generate training data. We fine-tune a Molmo-7B-based model using Low-Rank Adaptation (LoRa) and incorporating reasoning generated by our pipeline, along with visual grounding and evaluation capabilities. The fine-tuned Evaluative Large Action Model (ELAM) achieves strong performance on AutomotiveUI-Bench-4K (model and dataset are available on Hugging Face) and demonstrating strong cross-domain generalization, including a +5.2% improvement on ScreenSpot over the baseline model. Notably, our approach achieves 80.4% average accuracy on ScreenSpot, closely matching or even surpassing specialized models for desktop, mobile, and web, such as ShowUI, despite being trained for the infotainment domain. This research investigates how data collection and subsequent fine-tuning can lead to AI-driven progress within automotive UI understanding and interaction. The applied method is cost-efficient and fine-tuned models can be deployed on consumer-grade GPUs. 

---
# LightNobel: Improving Sequence Length Limitation in Protein Structure Prediction Model via Adaptive Activation Quantization 

**Authors**: Seunghee Han, Soongyu Choi, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.05893)  

**Abstract**: Recent advances in Protein Structure Prediction Models (PPMs), such as AlphaFold2 and ESMFold, have revolutionized computational biology by achieving unprecedented accuracy in predicting three-dimensional protein folding structures. However, these models face significant scalability challenges, particularly when processing proteins with long amino acid sequences (e.g., sequence length > 1,000). The primary bottleneck that arises from the exponential growth in activation sizes is driven by the unique data structure in PPM, which introduces an additional dimension that leads to substantial memory and computational demands. These limitations have hindered the effective scaling of PPM for real-world applications, such as analyzing large proteins or complex multimers with critical biological and pharmaceutical relevance.
In this paper, we present LightNobel, the first hardware-software co-designed accelerator developed to overcome scalability limitations on the sequence length in PPM. At the software level, we propose Token-wise Adaptive Activation Quantization (AAQ), which leverages unique token-wise characteristics, such as distogram patterns in PPM activations, to enable fine-grained quantization techniques without compromising accuracy. At the hardware level, LightNobel integrates the multi-precision reconfigurable matrix processing unit (RMPU) and versatile vector processing unit (VVPU) to enable the efficient execution of AAQ. Through these innovations, LightNobel achieves up to 8.44x, 8.41x speedup and 37.29x, 43.35x higher power efficiency over the latest NVIDIA A100 and H100 GPUs, respectively, while maintaining negligible accuracy loss. It also reduces the peak memory requirement up to 120.05x in PPM, enabling scalable processing for proteins with long sequences. 

---
# Multi-Modal Molecular Representation Learning via Structure Awareness 

**Authors**: Rong Yin, Ruyue Liu, Xiaoshuai Hao, Xingrui Zhou, Yong Liu, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05877)  

**Abstract**: Accurate extraction of molecular representations is a critical step in the drug discovery process. In recent years, significant progress has been made in molecular representation learning methods, among which multi-modal molecular representation methods based on images, and 2D/3D topologies have become increasingly mainstream. However, existing these multi-modal approaches often directly fuse information from different modalities, overlooking the potential of intermodal interactions and failing to adequately capture the complex higher-order relationships and invariant features between molecules. To overcome these challenges, we propose a structure-awareness-based multi-modal self-supervised molecular representation pre-training framework (MMSA) designed to enhance molecular graph representations by leveraging invariant knowledge between molecules. The framework consists of two main modules: the multi-modal molecular representation learning module and the structure-awareness module. The multi-modal molecular representation learning module collaboratively processes information from different modalities of the same molecule to overcome intermodal differences and generate a unified molecular embedding. Subsequently, the structure-awareness module enhances the molecular representation by constructing a hypergraph structure to model higher-order correlations between molecules. This module also introduces a memory mechanism for storing typical molecular representations, aligning them with memory anchors in the memory bank to integrate invariant knowledge, thereby improving the model generalization ability. Extensive experiments have demonstrated the effectiveness of MMSA, which achieves state-of-the-art performance on the MoleculeNet benchmark, with average ROC-AUC improvements ranging from 1.8% to 9.6% over baseline methods. 

---
# Towards Facial Image Compression with Consistency Preserving Diffusion Prior 

**Authors**: Yimin Zhou, Yichong Xia, Bin Chen, Baoyi An, Haoqian Wang, Zhi Wang, Yaowei Wang, Zikun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.05870)  

**Abstract**: With the widespread application of facial image data across various domains, the efficient storage and transmission of facial images has garnered significant attention. However, the existing learned face image compression methods often produce unsatisfactory reconstructed image quality at low bit rates. Simply adapting diffusion-based compression methods to facial compression tasks results in reconstructed images that perform poorly in downstream applications due to insufficient preservation of high-frequency information. To further explore the diffusion prior in facial image compression, we propose Facial Image Compression with a Stable Diffusion Prior (FaSDiff), a method that preserves consistency through frequency enhancement. FaSDiff employs a high-frequency-sensitive compressor in an end-to-end framework to capture fine image details and produce robust visual prompts. Additionally, we introduce a hybrid low-frequency enhancement module that disentangles low-frequency facial semantics and stably modulates the diffusion prior alongside visual prompts. The proposed modules allow FaSDiff to leverage diffusion priors for superior human visual perception while minimizing performance loss in machine vision due to semantic inconsistency. Extensive experiments show that FaSDiff outperforms state-of-the-art methods in balancing human visual quality and machine vision accuracy. The code will be released after the paper is accepted. 

---
# Generative Discovery of Partial Differential Equations by Learning from Math Handbooks 

**Authors**: Hao Xu, Yuntian Chen, Rui Cao, Tianning Tang, Mengge Du, Jian Li, Adrian H. Callaghan, Dongxiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05869)  

**Abstract**: Data driven discovery of partial differential equations (PDEs) is a promising approach for uncovering the underlying laws governing complex systems. However, purely data driven techniques face the dilemma of balancing search space with optimization efficiency. This study introduces a knowledge guided approach that incorporates existing PDEs documented in a mathematical handbook to facilitate the discovery process. These PDEs are encoded as sentence like structures composed of operators and basic terms, and used to train a generative model, called EqGPT, which enables the generation of free form PDEs. A loop of generation evaluation optimization is constructed to autonomously identify the most suitable PDE. Experimental results demonstrate that this framework can recover a variety of PDE forms with high accuracy and computational efficiency, particularly in cases involving complex temporal derivatives or intricate spatial terms, which are often beyond the reach of conventional methods. The approach also exhibits generalizability to irregular spatial domains and higher dimensional settings. Notably, it succeeds in discovering a previously unreported PDE governing strongly nonlinear surface gravity waves propagating toward breaking, based on real world experimental data, highlighting its applicability to practical scenarios and its potential to support scientific discovery. 

---
# Evolutionary ecology of words 

**Authors**: Reiji Suzuki, Takaya Arita  

**Link**: [PDF](https://arxiv.org/pdf/2505.05863)  

**Abstract**: We propose a model for the evolutionary ecology of words as one attempt to extend evolutionary game theory and agent-based models by utilizing the rich linguistic expressions of Large Language Models (LLMs). Our model enables the emergence and evolution of diverse and infinite options for interactions among agents. Within the population, each agent possesses a short word (or phrase) generated by an LLM and moves within a spatial environment. When agents become adjacent, the outcome of their interaction is determined by the LLM based on the relationship between their words, with the loser's word being replaced by the winner's. Word mutations, also based on LLM outputs, may occur. We conducted preliminary experiments assuming that ``strong animal species" would survive. The results showed that from an initial population consisting of well-known species, many species emerged both gradually and in a punctuated equilibrium manner. Each trial demonstrated the unique evolution of diverse populations, with one type of large species becoming dominant, such as terrestrial animals, marine life, or extinct species, which were ecologically specialized and adapted ones across diverse extreme habitats. We also conducted a long-term experiment with a large population, demonstrating the emergence and coexistence of diverse species. 

---
# AgentXploit: End-to-End Redteaming of Black-Box AI Agents 

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.05849)  

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites. 

---
# MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design 

**Authors**: Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05799)  

**Abstract**: Mixture-of-Experts (MoE) models face deployment challenges due to their large parameter counts and computational demands. We explore quantization for MoE models and highlight two key insights: 1) linear blocks exhibit varying quantization sensitivity, and 2) divergent expert activation frequencies create heterogeneous computational characteristics. Based on these observations, we introduce MxMoE, a mixed-precision optimization framework for MoE models that considers both algorithmic and system perspectives. MxMoE navigates the design space defined by parameter sensitivity, expert activation dynamics, and hardware resources to derive efficient mixed-precision configurations. Additionally, MxMoE automatically generates optimized mixed-precision GroupGEMM kernels, enabling parallel execution of GEMMs with different precisions. Evaluations show that MxMoE outperforms existing methods, achieving 2.4 lower Wikitext-2 perplexity than GPTQ at 2.25-bit and delivering up to 3.4x speedup over full precision, as well as up to 29.4% speedup over uniform quantization at equivalent accuracy with 5-bit weight-activation quantization. Our code is available at this https URL. 

---
# Human-in-the-Loop AI for HVAC Management Enhancing Comfort and Energy Efficiency 

**Authors**: Xinyu Liang, Frits de Nijs, Buser Say, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05796)  

**Abstract**: Heating, Ventilation, and Air Conditioning (HVAC) systems account for approximately 38% of building energy consumption globally, making them one of the most energy-intensive services. The increasing emphasis on energy efficiency and sustainability, combined with the need for enhanced occupant comfort, presents a significant challenge for traditional HVAC systems. These systems often fail to dynamically adjust to real-time changes in electricity market rates or individual comfort preferences, leading to increased energy costs and reduced comfort. In response, we propose a Human-in-the-Loop (HITL) Artificial Intelligence framework that optimizes HVAC performance by incorporating real-time user feedback and responding to fluctuating electricity prices. Unlike conventional systems that require predefined information about occupancy or comfort levels, our approach learns and adapts based on ongoing user input. By integrating the occupancy prediction model with reinforcement learning, the system improves operational efficiency and reduces energy costs in line with electricity market dynamics, thereby contributing to demand response initiatives. Through simulations, we demonstrate that our method achieves significant cost reductions compared to baseline approaches while maintaining or enhancing occupant comfort. This feedback-driven approach ensures personalized comfort control without the need for predefined settings, offering a scalable solution that balances individual preferences with economic and environmental goals. 

---
# What Is Next for LLMs? Next-Generation AI Computing Hardware Using Photonic Chips 

**Authors**: Renjie Li, Wenjie Wei, Qi Xin, Xiaoli Liu, Sixuan Mao, Erik Ma, Zijian Chen, Malu Zhang, Haizhou Li, Zhaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05794)  

**Abstract**: Large language models (LLMs) are rapidly pushing the limits of contemporary computing hardware. For example, training GPT-3 has been estimated to consume around 1300 MWh of electricity, and projections suggest future models may require city-scale (gigawatt) power budgets. These demands motivate exploration of computing paradigms beyond conventional von Neumann architectures. This review surveys emerging photonic hardware optimized for next-generation generative AI computing. We discuss integrated photonic neural network architectures (e.g., Mach-Zehnder interferometer meshes, lasers, wavelength-multiplexed microring resonators) that perform ultrafast matrix operations. We also examine promising alternative neuromorphic devices, including spiking neural network circuits and hybrid spintronic-photonic synapses, which combine memory and processing. The integration of two-dimensional materials (graphene, TMDCs) into silicon photonic platforms is reviewed for tunable modulators and on-chip synaptic elements. Transformer-based LLM architectures (self-attention and feed-forward layers) are analyzed in this context, identifying strategies and challenges for mapping dynamic matrix multiplications onto these novel hardware substrates. We then dissect the mechanisms of mainstream LLMs, such as ChatGPT, DeepSeek, and LLaMA, highlighting their architectural similarities and differences. We synthesize state-of-the-art components, algorithms, and integration methods, highlighting key advances and open issues in scaling such systems to mega-sized LLM models. We find that photonic computing systems could potentially surpass electronic processors by orders of magnitude in throughput and energy efficiency, but require breakthroughs in memory, especially for long-context windows and long token sequences, and in storage of ultra-large datasets. 

---
# FlowHFT: Flow Policy Induced Optimal High-Frequency Trading under Diverse Market Conditions 

**Authors**: Yang Li, Zhi Chen, Steve Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05784)  

**Abstract**: High-frequency trading (HFT) is an investing strategy that continuously monitors market states and places bid and ask orders at millisecond speeds. Traditional HFT approaches fit models with historical data and assume that future market states follow similar patterns. This limits the effectiveness of any single model to the specific conditions it was trained for. Additionally, these models achieve optimal solutions only under specific market conditions, such as assumptions about stock price's stochastic process, stable order flow, and the absence of sudden volatility. Real-world markets, however, are dynamic, diverse, and frequently volatile. To address these challenges, we propose the FlowHFT, a novel imitation learning framework based on flow matching policy. FlowHFT simultaneously learns strategies from numerous expert models, each proficient in particular market scenarios. As a result, our framework can adaptively adjust investment decisions according to the prevailing market state. Furthermore, FlowHFT incorporates a grid-search fine-tuning mechanism. This allows it to refine strategies and achieve superior performance even in complex or extreme market scenarios where expert strategies may be suboptimal. We test FlowHFT in multiple market environments. We first show that flow matching policy is applicable in stochastic market environments, thus enabling FlowHFT to learn trading strategies under different market conditions. Notably, our single framework consistently achieves performance superior to the best expert for each market condition. 

---
# PyResBugs: A Dataset of Residual Python Bugs for Natural Language-Driven Fault Injection 

**Authors**: Domenico Cotroneo, Giuseppe De Rosa, Pietro Liguori  

**Link**: [PDF](https://arxiv.org/pdf/2505.05777)  

**Abstract**: This paper presents PyResBugs, a curated dataset of residual bugs, i.e., defects that persist undetected during traditional testing but later surface in production, collected from major Python frameworks. Each bug in the dataset is paired with its corresponding fault-free (fixed) version and annotated with multi-level natural language (NL) descriptions. These NL descriptions enable natural language-driven fault injection, offering a novel approach to simulating real-world faults in software systems. By bridging the gap between software fault injection techniques and real-world representativeness, PyResBugs provides researchers with a high-quality resource for advancing AI-driven automated testing in Python systems. 

---
# Predicting Diabetic Macular Edema Treatment Responses Using OCT: Dataset and Methods of APTOS Competition 

**Authors**: Weiyi Zhang, Peranut Chotcomwongse, Yinwen Li, Pusheng Xu, Ruijie Yao, Lianhao Zhou, Yuxuan Zhou, Hui Feng, Qiping Zhou, Xinyue Wang, Shoujin Huang, Zihao Jin, Florence H.T. Chung, Shujun Wang, Yalin Zheng, Mingguang He, Danli Shi, Paisan Ruamviboonsuk  

**Link**: [PDF](https://arxiv.org/pdf/2505.05768)  

**Abstract**: Diabetic macular edema (DME) significantly contributes to visual impairment in diabetic patients. Treatment responses to intravitreal therapies vary, highlighting the need for patient stratification to predict therapeutic benefits and enable personalized strategies. To our knowledge, this study is the first to explore pre-treatment stratification for predicting DME treatment responses. To advance this research, we organized the 2nd Asia-Pacific Tele-Ophthalmology Society (APTOS) Big Data Competition in 2021. The competition focused on improving predictive accuracy for anti-VEGF therapy responses using ophthalmic OCT images. We provided a dataset containing tens of thousands of OCT images from 2,000 patients with labels across four sub-tasks. This paper details the competition's structure, dataset, leading methods, and evaluation metrics. The competition attracted strong scientific community participation, with 170 teams initially registering and 41 reaching the final round. The top-performing team achieved an AUC of 80.06%, highlighting the potential of AI in personalized DME treatment and clinical decision-making. 

---
# Multi-Agent Systems for Robotic Autonomy with LLMs 

**Authors**: Junhong Chen, Ziqi Yang, Haoyuan G Xu, Dandan Zhang, George Mylonas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05762)  

**Abstract**: Since the advent of Large Language Models (LLMs), various research based on such models have maintained significant academic attention and impact, especially in AI and robotics. In this paper, we propose a multi-agent framework with LLMs to construct an integrated system for robotic task analysis, mechanical design, and path generation. The framework includes three core agents: Task Analyst, Robot Designer, and Reinforcement Learning Designer. Outputs are formatted as multimodal results, such as code files or technical reports, for stronger understandability and usability. To evaluate generalizability comparatively, we conducted experiments with models from both GPT and DeepSeek. Results demonstrate that the proposed system can design feasible robots with control strategies when appropriate task inputs are provided, exhibiting substantial potential for enhancing the efficiency and accessibility of robotic system development in research and industrial applications. 

---
# Evolutionary thoughts: integration of large language models and evolutionary algorithms 

**Authors**: Antonio Jimeno Yepes, Pieter Barnard  

**Link**: [PDF](https://arxiv.org/pdf/2505.05756)  

**Abstract**: Large Language Models (LLMs) have unveiled remarkable capabilities in understanding and generating both natural language and code, but LLM reasoning is prone to hallucination and struggle with complex, novel scenarios, often getting stuck on partial or incorrect solutions. However, the inherent ability of Evolutionary Algorithms (EAs) to explore extensive and complex search spaces makes them particularly effective in scenarios where traditional optimization methodologies may falter. However, EAs explore a vast search space when applied to complex problems.
To address the computational bottleneck of evaluating large populations, particularly crucial for complex evolutionary tasks, we introduce a highly efficient evaluation framework. This implementation maintains compatibility with existing primitive definitions, ensuring the generation of valid individuals.
Using LLMs, we propose an enhanced evolutionary search strategy that enables a more focused exploration of expansive solution spaces. LLMs facilitate the generation of superior candidate solutions, as evidenced by empirical results demonstrating their efficacy in producing improved outcomes. 

---
# Towards Embodiment Scaling Laws in Robot Locomotion 

**Authors**: Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I. Christensen, Jan Peters, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.05753)  

**Abstract**: Developing generalist agents that can operate across diverse tasks, environments, and physical embodiments is a grand challenge in robotics and artificial intelligence. In this work, we focus on the axis of embodiment and investigate embodiment scaling laws$\unicode{x2013}$the hypothesis that increasing the number of training embodiments improves generalization to unseen ones. Using robot locomotion as a test bed, we procedurally generate a dataset of $\sim$1,000 varied embodiments, spanning humanoids, quadrupeds, and hexapods, and train generalist policies capable of handling diverse observation and action spaces on random subsets. We find that increasing the number of training embodiments improves generalization to unseen ones, and scaling embodiments is more effective in enabling embodiment-level generalization than scaling data on small, fixed sets of embodiments. Notably, our best policy, trained on the full dataset, zero-shot transfers to novel embodiments in the real world, such as Unitree Go2 and H1. These results represent a step toward general embodied intelligence, with potential relevance to adaptive control for configurable robots, co-design of morphology and control, and beyond. 

---
# Accurate and Efficient Multivariate Time Series Forecasting via Offline Clustering 

**Authors**: Yiming Niu, Jinliang Deng, Lulu Zhang, Zimu Zhou, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.05738)  

**Abstract**: Accurate and efficient multivariate time series (MTS) forecasting is essential for applications such as traffic management and weather prediction, which depend on capturing long-range temporal dependencies and interactions between entities. Existing methods, particularly those based on Transformer architectures, compute pairwise dependencies across all time steps, leading to a computational complexity that scales quadratically with the length of the input. To overcome these challenges, we introduce the Forecaster with Offline Clustering Using Segments (FOCUS), a novel approach to MTS forecasting that simplifies long-range dependency modeling through the use of prototypes extracted via offline clustering. These prototypes encapsulate high-level events in the real-world system underlying the data, summarizing the key characteristics of similar time segments. In the online phase, FOCUS dynamically adapts these patterns to the current input and captures dependencies between the input segment and high-level events, enabling both accurate and efficient forecasting. By identifying prototypes during the offline clustering phase, FOCUS reduces the computational complexity of modeling long-range dependencies in the online phase to linear scaling. Extensive experiments across diverse benchmarks demonstrate that FOCUS achieves state-of-the-art accuracy while significantly reducing computational costs. 

---
# HyperspectralMAE: The Hyperspectral Imagery Classification Model using Fourier-Encoded Dual-Branch Masked Autoencoder 

**Authors**: Wooyoung Jeong, Hyun Jae Park, Seonghun Jeong, Jong Wook Jang, Tae Hoon Lim, Dae Seoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.05710)  

**Abstract**: Hyperspectral imagery provides rich spectral detail but poses unique challenges because of its high dimensionality in both spatial and spectral domains. We propose \textit{HyperspectralMAE}, a Transformer-based foundation model for hyperspectral data that employs a \textit{dual masking} strategy: during pre-training we randomly occlude 50\% of spatial patches and 50\% of spectral bands. This forces the model to learn representations capable of reconstructing missing information across both dimensions. To encode spectral order, we introduce learnable harmonic Fourier positional embeddings based on wavelength. The reconstruction objective combines mean-squared error (MSE) with the spectral angle mapper (SAM) to balance pixel-level accuracy and spectral-shape fidelity.
The resulting model contains about $1.8\times10^{8}$ parameters and produces 768-dimensional embeddings, giving it sufficient capacity for transfer learning. We pre-trained HyperspectralMAE on two large hyperspectral corpora -- NASA EO-1 Hyperion ($\sim$1\,600 scenes, $\sim$$3\times10^{11}$ pixel spectra) and DLR EnMAP Level-0 ($\sim$1\,300 scenes, $\sim$$3\times10^{11}$ pixel spectra) -- and fine-tuned it for land-cover classification on the Indian Pines benchmark. HyperspectralMAE achieves state-of-the-art transfer-learning accuracy on Indian Pines, confirming that masked dual-dimensional pre-training yields robust spectral-spatial representations. These results demonstrate that dual masking and wavelength-aware embeddings advance hyperspectral image reconstruction and downstream analysis. 

---
# Assessing Robustness to Spurious Correlations in Post-Training Language Models 

**Authors**: Julia Shuieh, Prasann Singhal, Apaar Shanker, John Heyer, George Pu, Samuel Denton  

**Link**: [PDF](https://arxiv.org/pdf/2505.05704)  

**Abstract**: Supervised and preference-based fine-tuning techniques have become popular for aligning large language models (LLMs) with user intent and correctness criteria. However, real-world training data often exhibits spurious correlations -- arising from biases, dataset artifacts, or other "shortcut" features -- that can compromise a model's performance or generalization. In this paper, we systematically evaluate three post-training algorithms -- Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and KTO (Kahneman-Tversky Optimization) -- across a diverse set of synthetic tasks and spuriousness conditions. Our tasks span mathematical reasoning, constrained instruction-following, and document-grounded question answering. We vary the degree of spurious correlation (10% vs. 90%) and investigate two forms of artifacts: "Feature Ambiguity" and "Distributional Narrowness." Our results show that the models often but not always degrade under higher spuriousness. The preference-based methods (DPO/KTO) can demonstrate relative robustness in mathematical reasoning tasks. By contrast, SFT maintains stronger performance in complex, context-intensive tasks. These findings highlight that no single post-training strategy universally outperforms in all scenarios; the best choice depends on the type of target task and the nature of spurious correlations. 

---
# Interactive Diabetes Risk Prediction Using Explainable Machine Learning: A Dash-Based Approach with SHAP, LIME, and Comorbidity Insights 

**Authors**: Udaya Allani  

**Link**: [PDF](https://arxiv.org/pdf/2505.05683)  

**Abstract**: This study presents a web-based interactive health risk prediction tool designed to assess diabetes risk using machine learning models. Built on the 2015 CDC BRFSS dataset, the study evaluates models including Logistic Regression, Random Forest, XGBoost, LightGBM, KNN, and Neural Networks under original, SMOTE, and undersampling strategies. LightGBM with undersampling achieved the best recall, making it ideal for risk detection. The tool integrates SHAP and LIME to explain predictions and highlights comorbidity correlations using Pearson analysis. A Dash-based UI enables user-friendly interaction with model predictions, personalized suggestions, and feature insights, supporting data-driven health awareness. 

---
# Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval 

**Authors**: Alexander Most, Joseph Winjum, Ayan Biswas, Shawn Jones, Nishath Rajiv Ranasinghe, Dan O'Malley, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2505.05666)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a popular technique for enhancing the reliability and utility of Large Language Models (LLMs) by grounding responses in external documents. Traditional RAG systems rely on Optical Character Recognition (OCR) to first process scanned documents into text. However, even state-of-the-art OCRs can introduce errors, especially in degraded or complex documents. Recent vision-language approaches, such as ColPali, propose direct visual embedding of documents, eliminating the need for OCR. This study presents a systematic comparison between a vision-based RAG system (ColPali) and more traditional OCR-based pipelines utilizing Llama 3.2 (90B) and Nougat OCR across varying document qualities. Beyond conventional retrieval accuracy metrics, we introduce a semantic answer evaluation benchmark to assess end-to-end question-answering performance. Our findings indicate that while vision-based RAG performs well on documents it has been fine-tuned on, OCR-based RAG is better able to generalize to unseen documents of varying quality. We highlight the key trade-offs between computational efficiency and semantic accuracy, offering practical guidance for RAG practitioners in selecting between OCR-dependent and vision-based document retrieval systems in production environments. 

---
# Adaptive Stress Testing Black-Box LLM Planners 

**Authors**: Neeloy Chakraborty, John Pohovey, Melkior Ornik, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2505.05665)  

**Abstract**: Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing black-box methods often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using Adaptive Stress Testing (AST) with Monte-Carlo Tree Search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty. By generating MCTS prompt perturbation trees across diverse scenarios, we show that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM. 

---
# Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks 

**Authors**: Mohamed-Khalil Bouzidi, Christian Schlauch, Nicole Scheuerer, Yue Yao, Nadja Klein, Daniel Göhring, Jörg Reichardt  

**Link**: [PDF](https://arxiv.org/pdf/2505.05638)  

**Abstract**: Fueled by motion prediction competitions and benchmarks, recent years have seen the emergence of increasingly large learning based prediction models, many with millions of parameters, focused on improving open-loop prediction accuracy by mere centimeters. However, these benchmarks fail to assess whether such improvements translate to better performance when integrated into an autonomous driving stack. In this work, we systematically evaluate the interplay between state-of-the-art motion predictors and motion planners. Our results show that higher open-loop accuracy does not always correlate with better closed-loop driving behavior and that other factors, such as temporal consistency of predictions and planner compatibility, also play a critical role. Furthermore, we investigate downsized variants of these models, and, surprisingly, find that in some cases models with up to 86% fewer parameters yield comparable or even superior closed-loop driving performance. Our code is available at this https URL. 

---
# Looking Beyond Language Priors: Enhancing Visual Comprehension and Attention in Multimodal Models 

**Authors**: Aarti Ghatkesar, Uddeshya Upadhyay, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2505.05626)  

**Abstract**: Achieving deep alignment between vision and language remains a central challenge for Multimodal Large Language Models (MLLMs). These models often fail to fully leverage visual input, defaulting to strong language priors. Our approach first provides insights into how MLLMs internally build visual understanding of image regions and then introduces techniques to amplify this capability. Specifically, we explore techniques designed both to deepen the model's understanding of visual content and to ensure that these visual insights actively guide language generation. We demonstrate the superior multimodal understanding of our resultant model through a detailed upstream analysis quantifying its ability to predict visually-dependent tokens as well as 10 pt boost on visually challenging tasks. 

---
# SPIN-ODE: Stiff Physics-Informed Neural ODE for Chemical Reaction Rate Estimation 

**Authors**: Wenqing Peng, Zhi-Song Liu, Michael Boy  

**Link**: [PDF](https://arxiv.org/pdf/2505.05625)  

**Abstract**: Estimating rate constants from complex chemical reactions is essential for advancing detailed chemistry. However, the stiffness inherent in real-world atmospheric chemistry systems poses severe challenges, leading to training instability and poor convergence that hinder effective rate constant estimation using learning-based approaches. To address this, we propose a Stiff Physics-Informed Neural ODE framework (SPIN-ODE) for chemical reaction modelling. Our method introduces a three-stage optimisation process: first, a latent neural ODE learns the continuous and differentiable trajectory between chemical concentrations and their time derivatives; second, an explicit Chemical Reaction Neural Network (CRNN) extracts the underlying rate coefficients based on the learned dynamics; and third, fine-tune CRNN using a neural ODE solver to further improve rate coefficient estimation. Extensive experiments on both synthetic and newly proposed real-world datasets validate the effectiveness and robustness of our approach. As the first work on stiff Neural ODEs for chemical rate coefficient discovery, our study opens promising directions for integrating neural networks with detailed chemistry. 

---
# CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory 

**Authors**: Weichen Zhang, Chen Gao, Shiquan Yu, Ruiying Peng, Baining Zhao, Qian Zhang, Jinqiang Cui, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05622)  

**Abstract**: Aerial vision-and-language navigation (VLN), requiring drones to interpret natural language instructions and navigate complex urban environments, emerges as a critical embodied AI challenge that bridges human-robot interaction, 3D spatial reasoning, and real-world deployment. Although existing ground VLN agents achieved notable results in indoor and outdoor settings, they struggle in aerial VLN due to the absence of predefined navigation graphs and the exponentially expanding action space in long-horizon exploration. In this work, we propose \textbf{CityNavAgent}, a large language model (LLM)-empowered agent that significantly reduces the navigation complexity for urban aerial VLN. Specifically, we design a hierarchical semantic planning module (HSPM) that decomposes the long-horizon task into sub-goals with different semantic levels. The agent reaches the target progressively by achieving sub-goals with different capacities of the LLM. Additionally, a global memory module storing historical trajectories into a topological graph is developed to simplify navigation for visited targets. Extensive benchmark experiments show that our method achieves state-of-the-art performance with significant improvement. Further experiments demonstrate the effectiveness of different modules of CityNavAgent for aerial VLN in continuous city environments. The code is available at \href{this https URL}{link}. 

---
# Enhancing Satellite Object Localization with Dilated Convolutions and Attention-aided Spatial Pooling 

**Authors**: Seraj Al Mahmud Mostafa, Chenxi Wang, Jia Yue, Yuta Hozumi, Jianwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05599)  

**Abstract**: Object localization in satellite imagery is particularly challenging due to the high variability of objects, low spatial resolution, and interference from noise and dominant features such as clouds and city lights. In this research, we focus on three satellite datasets: upper atmospheric Gravity Waves (GW), mesospheric Bores (Bore), and Ocean Eddies (OE), each presenting its own unique challenges. These challenges include the variability in the scale and appearance of the main object patterns, where the size, shape, and feature extent of objects of interest can differ significantly. To address these challenges, we introduce YOLO-DCAP, a novel enhanced version of YOLOv5 designed to improve object localization in these complex scenarios. YOLO-DCAP incorporates a Multi-scale Dilated Residual Convolution (MDRC) block to capture multi-scale features at scale with varying dilation rates, and an Attention-aided Spatial Pooling (AaSP) module to focus on the global relevant spatial regions, enhancing feature selection. These structural improvements help to better localize objects in satellite imagery. Experimental results demonstrate that YOLO-DCAP significantly outperforms both the YOLO base model and state-of-the-art approaches, achieving an average improvement of 20.95% in mAP50 and 32.23% in IoU over the base model, and 7.35% and 9.84% respectively over state-of-the-art alternatives, consistently across all three satellite datasets. These consistent gains across all three satellite datasets highlight the robustness and generalizability of the proposed approach. Our code is open sourced at this https URL. 

---
# Trading Under Uncertainty: A Distribution-Based Strategy for Futures Markets Using FutureQuant Transformer 

**Authors**: Wenhao Guo, Yuda Wang, Zeqiao Huang, Changjiang Zhang, Shumin ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.05595)  

**Abstract**: In the complex landscape of traditional futures trading, where vast data and variables like real-time Limit Order Books (LOB) complicate price predictions, we introduce the FutureQuant Transformer model, leveraging attention mechanisms to navigate these challenges. Unlike conventional models focused on point predictions, the FutureQuant model excels in forecasting the range and volatility of future prices, thus offering richer insights for trading strategies. Its ability to parse and learn from intricate market patterns allows for enhanced decision-making, significantly improving risk management and achieving a notable average gain of 0.1193% per 30-minute trade over state-of-the-art models with a simple algorithm using factors such as RSI, ATR, and Bollinger Bands. This innovation marks a substantial leap forward in predictive analytics within the volatile domain of futures trading. 

---
# ReactDance: Progressive-Granular Representation for Long-Term Coherent Reactive Dance Generation 

**Authors**: Jingzhong Lin, Yuanyuan Qi, Xinru Li, Wenxuan Huang, Xiangfeng Xu, Bangyan Li, Xuejiao Wang, Gaoqi He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05589)  

**Abstract**: Reactive dance generation (RDG) produces follower movements conditioned on guiding dancer and music while ensuring spatial coordination and temporal coherence. However, existing methods overemphasize global constraints and optimization, overlooking local information, such as fine-grained spatial interactions and localized temporal context. Therefore, we present ReactDance, a novel diffusion-based framework for high-fidelity RDG with long-term coherence and multi-scale controllability. Unlike existing methods that struggle with interaction fidelity, synchronization, and temporal consistency in duet synthesis, our approach introduces two key innovations: 1)Group Residual Finite Scalar Quantization (GRFSQ), a multi-scale disentangled motion representation that captures interaction semantics from coarse body rhythms to fine-grained joint dynamics, and 2)Blockwise Local Context (BLC), a sampling strategy eliminating error accumulation in long sequence generation via local block causal masking and periodic positional encoding. Built on the decoupled multi-scale GRFSQ representation, we implement a diffusion model withLayer-Decoupled Classifier-free Guidance (LDCFG), allowing granular control over motion semantics across scales. Extensive experiments on standard benchmarks demonstrate that ReactDance surpasses existing methods, achieving state-of-the-art performance. 

---
# Flight Validation of Learning-Based Trajectory Optimization for the Astrobee Free-Flyer 

**Authors**: Somrita Banerjee, Abhishek Cauligi, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.05588)  

**Abstract**: Although widely used in commercial and industrial robotics, trajectory optimization has seen limited use in space applications due to its high computational demands. In this work, we present flight results from experiments with the Astrobee free-flying robot on board the International Space Station (ISS), that demonstrate how machine learning can accelerate on-board trajectory optimization while preserving theoretical solver guarantees. To the best of the authors' knowledge, this is the first-ever demonstration of learning-based control on the ISS. Our approach leverages the GuSTO sequential convex programming framework and uses a neural network, trained offline, to map problem parameters to effective initial ``warm-start'' trajectories, paving the way for faster real-time optimization on resource-constrained space platforms. 

---
# PyTDC: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models 

**Authors**: Alejandro Velez-Arce, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.05577)  

**Abstract**: Existing biomedical benchmarks do not provide end-to-end infrastructure for training, evaluation, and inference of models that integrate multimodal biological data and a broad range of machine learning tasks in therapeutics. We present PyTDC, an open-source machine-learning platform providing streamlined training, evaluation, and inference software for multimodal biological AI models. PyTDC unifies distributed, heterogeneous, continuously updated data sources and model weights and standardizes benchmarking and inference endpoints. This paper discusses the components of PyTDC's architecture and, to our knowledge, the first-of-its-kind case study on the introduced single-cell drug-target nomination ML task. We find state-of-the-art methods in graph representation learning and domain-specific methods from graph theory perform poorly on this task. Though we find a context-aware geometric deep learning method that outperforms the evaluated SoTA and domain-specific baseline methods, the model is unable to generalize to unseen cell types or incorporate additional modalities, highlighting PyTDC's capacity to facilitate an exciting avenue of research developing multimodal, context-aware, foundation models for open problems in biomedical AI. 

---
# Prompt to Polyp: Clinically-Aware Medical Image Synthesis with Diffusion Models 

**Authors**: Mikhail Chaichuk, Sushant Gautam, Steven Hicks, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2505.05573)  

**Abstract**: The generation of realistic medical images from text descriptions has significant potential to address data scarcity challenges in healthcare AI while preserving patient privacy. This paper presents a comprehensive study of text-to-image synthesis in the medical domain, comparing two distinct approaches: (1) fine-tuning large pre-trained latent diffusion models and (2) training small, domain-specific models. We introduce a novel model named MSDM, an optimized architecture based on Stable Diffusion that integrates a clinical text encoder, variational autoencoder, and cross-attention mechanisms to better align medical text prompts with generated images. Our study compares two approaches: fine-tuning large pre-trained models (FLUX, Kandinsky) versus training compact domain-specific models (MSDM). Evaluation across colonoscopy (MedVQA-GI) and radiology (ROCOv2) datasets reveals that while large models achieve higher fidelity, our optimized MSDM delivers comparable quality with lower computational costs. Quantitative metrics and qualitative evaluations by medical experts reveal strengths and limitations of each approach. 

---
# Griffin: Towards a Graph-Centric Relational Database Foundation Model 

**Authors**: Yanbo Wang, Xiyuan Wang, Quan Gan, Minjie Wang, Qibin Yang, David Wipf, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05568)  

**Abstract**: We introduce Griffin, the first foundation model attemptation designed specifically for Relational Databases (RDBs). Unlike previous smaller models focused on single RDB tasks, Griffin unifies the data encoder and task decoder to handle diverse tasks. Additionally, we enhance the architecture by incorporating a cross-attention module and a novel aggregator. Griffin utilizes pretraining on both single-table and RDB datasets, employing advanced encoders for categorical, numerical, and metadata features, along with innovative components such as cross-attention modules and enhanced message-passing neural networks (MPNNs) to capture the complexities of relational data. Evaluated on large-scale, heterogeneous, and temporal graphs extracted from RDBs across various domains (spanning over 150 million nodes), Griffin demonstrates superior or comparable performance to individually trained models, excels in low-data scenarios, and shows strong transferability with similarity and diversity in pretraining across new datasets and tasks, highlighting its potential as a universally applicable foundation model for RDBs. Code available at this https URL. 

---
# Would You Rely on an Eerie Agent? A Systematic Review of the Impact of the Uncanny Valley Effect on Trust in Human-Agent Interaction 

**Authors**: Ahdiyeh Alipour, Tilo Hartmann, Maryam Alimardani  

**Link**: [PDF](https://arxiv.org/pdf/2505.05543)  

**Abstract**: Trust is a fundamental component of human-agent interaction. With the increasing presence of artificial agents in daily life, it is essential to understand how people perceive and trust these agents. One of the key challenges affecting this perception is the Uncanny Valley Effect (UVE), where increasingly human-like artificial beings can be perceived as eerie or repelling. Despite growing interest in trust and the UVE, existing research varies widely in terms of how these concepts are defined and operationalized. This inconsistency raises important questions about how and under what conditions the UVE influences trust in agents. A systematic understanding of their relationship is currently lacking. This review aims to examine the impact of the UVE on human trust in agents and to identify methodological patterns, limitations, and gaps in the existing empirical literature. Following PRISMA guidelines, a systematic search identified 53 empirical studies that investigated both UVE-related constructs and trust or trust-related outcomes. Studies were analyzed based on a structured set of categories, including types of agents and interactions, methodological and measurement approaches, and key findings. The results of our systematic review reveal that most studies rely on static images or hypothetical scenarios with limited real-time interaction, and the majority use subjective trust measures. This review offers a novel framework for classifying trust measurement approaches with regard to the best-practice criteria for empirically investigating the UVE. As the first systematic attempt to map the intersection of UVE and trust, this review contributes to a deeper understanding of their interplay and offers a foundation for future research. Keywords: the uncanny valley effect, trust, human-likeness, affinity response, human-agent interaction 

---
# Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet 

**Authors**: Md Kamrujjaman Mobin, Md Saiful Islam, Sadik Al Barid, Md Masum  

**Link**: [PDF](https://arxiv.org/pdf/2505.05538)  

**Abstract**: Electrocardiogram (ECG) classification is crucial for automated cardiac disease diagnosis, yet existing methods often struggle to capture local morphological details and long-range temporal dependencies simultaneously. To address these challenges, we propose Cardioformer, a novel multi-granularity hybrid model that integrates cross-channel patching, hierarchical residual learning, and a two-stage self-attention mechanism. Cardioformer first encodes multi-scale token embeddings to capture fine-grained local features and global contextual information and then selectively fuses these representations through intra- and inter-granularity self-attention. Extensive evaluations on three benchmark ECG datasets under subject-independent settings demonstrate that model consistently outperforms four state-of-the-art baselines. Our Cardioformer model achieves the AUROC of 96.34$\pm$0.11, 89.99$\pm$0.12, and 95.59$\pm$1.66 in MIMIC-IV, PTB-XL and PTB dataset respectively outperforming PatchTST, Reformer, Transformer, and Medformer models. It also demonstrates strong cross-dataset generalization, achieving 49.18% AUROC on PTB and 68.41% on PTB-XL when trained on MIMIC-IV. These findings underscore the potential of Cardioformer to advance automated ECG analysis, paving the way for more accurate and robust cardiovascular disease diagnosis. We release the source code at this https URL. 

---
# Rethinking Graph Contrastive Learning through Relative Similarity Preservation 

**Authors**: Zhiyuan Ning, Pengfei Wang, Ziyue Qiao, Pengyang Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.05533)  

**Abstract**: Graph contrastive learning (GCL) has achieved remarkable success by following the computer vision paradigm of preserving absolute similarity between augmented views. However, this approach faces fundamental challenges in graphs due to their discrete, non-Euclidean nature -- view generation often breaks semantic validity and similarity verification becomes unreliable. Through analyzing 11 real-world graphs, we discover a universal pattern transcending the homophily-heterophily dichotomy: label consistency systematically diminishes as structural distance increases, manifesting as smooth decay in homophily graphs and oscillatory decay in heterophily graphs. We establish theoretical guarantees for this pattern through random walk theory, proving label distribution convergence and characterizing the mechanisms behind different decay behaviors. This discovery reveals that graphs naturally encode relative similarity patterns, where structurally closer nodes exhibit collectively stronger semantic relationships. Leveraging this insight, we propose RELGCL, a novel GCL framework with complementary pairwise and listwise implementations that preserve these inherent patterns through collective similarity objectives. Extensive experiments demonstrate that our method consistently outperforms 20 existing approaches across both homophily and heterophily graphs, validating the effectiveness of leveraging natural relative similarity over artificial absolute similarity. 

---
# Low-bit Model Quantization for Deep Neural Networks: A Survey 

**Authors**: Kai Liu, Qian Zheng, Kaiwen Tao, Zhiteng Li, Haotong Qin, Wenbo Li, Yong Guo, Xianglong Liu, Linghe Kong, Guihai Chen, Yulun Zhang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05530)  

**Abstract**: With unprecedented rapid development, deep neural networks (DNNs) have deeply influenced almost all fields. However, their heavy computation costs and model sizes are usually unacceptable in real-world deployment. Model quantization, an effective weight-lighting technique, has become an indispensable procedure in the whole deployment pipeline. The essence of quantization acceleration is the conversion from continuous floating-point numbers to discrete integer ones, which significantly speeds up the memory I/O and calculation, i.e., addition and multiplication. However, performance degradation also comes with the conversion because of the loss of precision. Therefore, it has become increasingly popular and critical to investigate how to perform the conversion and how to compensate for the information loss. This article surveys the recent five-year progress towards low-bit quantization on DNNs. We discuss and compare the state-of-the-art quantization methods and classify them into 8 main categories and 24 sub-categories according to their core techniques. Furthermore, we shed light on the potential research opportunities in the field of model quantization. A curated list of model quantization is provided at this https URL. 

---
# X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP 

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2505.05528)  

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{this https URL}{GitHub repository}. 

---
# ADMM-Based Training for Spiking Neural Networks 

**Authors**: Giovanni Perin, Cesare Bidini, Riccardo Mazzieri, Michele Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05527)  

**Abstract**: In recent years, spiking neural networks (SNNs) have gained momentum due to their high potential in time-series processing combined with minimal energy consumption. However, they still lack a dedicated and efficient training algorithm. The popular backpropagation with surrogate gradients, adapted from stochastic gradient descent (SGD)-derived algorithms, has several drawbacks when used as an optimizer for SNNs. Specifically, it suffers from low scalability and numerical imprecision. In this paper, we propose a novel SNN training method based on the alternating direction method of multipliers (ADMM). Our ADMM-based training aims to solve the problem of the SNN step function's non-differentiability. We formulate the problem, derive closed-form updates, and empirically show the optimizer's convergence properties, great potential, and possible new research directions to improve the method in a simulated proof-of-concept. 

---
# GenAI in Entrepreneurship: a systematic review of generative artificial intelligence in entrepreneurship research: current issues and future directions 

**Authors**: Anna Kusetogullari, Huseyin Kusetogullari, Martin Andersson, Tony Gorschek  

**Link**: [PDF](https://arxiv.org/pdf/2505.05523)  

**Abstract**: Generative Artificial Intelligence (GenAI) and Large Language Models (LLMs) are recognized to have significant effects on industry and business dynamics, not least because of their impact on the preconditions for entrepreneurship. There is still a lack of knowledge of GenAI as a theme in entrepreneurship research. This paper presents a systematic literature review aimed at identifying and analyzing the evolving landscape of research on the effects of GenAI on entrepreneurship. We analyze 83 peer-reviewed articles obtained from leading academic databases: Web of Science and Scopus. Using natural language processing and unsupervised machine learning techniques with TF-IDF vectorization, Principal Component Analysis (PCA), and hierarchical clustering, five major thematic clusters are identified: (1) Digital Transformation and Behavioral Models, (2) GenAI-Enhanced Education and Learning Systems, (3) Sustainable Innovation and Strategic AI Impact, (4) Business Models and Market Trends, and (5) Data-Driven Technological Trends in Entrepreneurship. Based on the review, we discuss future research directions, gaps in the current literature, as well as ethical concerns raised in the literature. We highlight the need for more macro-level research on GenAI and LLMs as external enablers for entrepreneurship and for research on effective regulatory frameworks that facilitate business experimentation, innovation, and further technology development. 

---
# Continuous Thought Machines 

**Authors**: Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.05522)  

**Abstract**: Biological brains demonstrate complex neural activity, where the timing and interplay between neurons is critical to how brains process information. Most deep learning architectures simplify neural activity by abstracting away temporal dynamics. In this paper we challenge that paradigm. By incorporating neuron-level processing and synchronization, we can effectively reintroduce neural timing as a foundational element. We present the Continuous Thought Machine (CTM), a model designed to leverage neural dynamics as its core representation. The CTM has two core innovations: (1) neuron-level temporal processing, where each neuron uses unique weight parameters to process a history of incoming signals; and (2) neural synchronization employed as a latent representation. The CTM aims to strike a balance between oversimplified neuron abstractions that improve computational efficiency, and biological realism. It operates at a level of abstraction that effectively captures essential temporal dynamics while remaining computationally tractable for deep learning. We demonstrate the CTM's strong performance and versatility across a range of challenging tasks, including ImageNet-1K classification, solving 2D mazes, sorting, parity computation, question-answering, and RL tasks. Beyond displaying rich internal representations and offering a natural avenue for interpretation owing to its internal process, the CTM is able to perform tasks that require complex sequential reasoning. The CTM can also leverage adaptive compute, where it can stop earlier for simpler tasks, or keep computing when faced with more challenging instances. The goal of this work is to share the CTM and its associated innovations, rather than pushing for new state-of-the-art results. To that end, we believe the CTM represents a significant step toward developing more biologically plausible and powerful artificial intelligence systems. 

---
# GaMNet: A Hybrid Network with Gabor Fusion and NMamba for Efficient 3D Glioma Segmentation 

**Authors**: Chengwei Ye, Huanzhen Zhang, Yufei Lin, Kangsheng Wang, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05520)  

**Abstract**: Gliomas are aggressive brain tumors that pose serious health risks. Deep learning aids in lesion segmentation, but CNN and Transformer-based models often lack context modeling or demand heavy computation, limiting real-time use on mobile medical devices. We propose GaMNet, integrating the NMamba module for global modeling and a multi-scale CNN for efficient local feature extraction. To improve interpretability and mimic the human visual system, we apply Gabor filters at multiple scales. Our method achieves high segmentation accuracy with fewer parameters and faster computation. Extensive experiments show GaMNet outperforms existing methods, notably reducing false positives and negatives, which enhances the reliability of clinical diagnosis. 

---
# AI-powered virtual eye: perspective, challenges and opportunities 

**Authors**: Yue Wu, Yibo Guo, Yulong Yan, Jiancheng Yang, Xin Zhou, Ching-Yu Cheng, Danli Shi, Mingguang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05516)  

**Abstract**: We envision the "virtual eye" as a next-generation, AI-powered platform that uses interconnected foundation models to simulate the eye's intricate structure and biological function across all scales. Advances in AI, imaging, and multiomics provide a fertile ground for constructing a universal, high-fidelity digital replica of the human eye. This perspective traces the evolution from early mechanistic and rule-based models to contemporary AI-driven approaches, integrating in a unified model with multimodal, multiscale, dynamic predictive capabilities and embedded feedback mechanisms. We propose a development roadmap emphasizing the roles of large-scale multimodal datasets, generative AI, foundation models, agent-based architectures, and interactive interfaces. Despite challenges in interpretability, ethics, data processing and evaluation, the virtual eye holds the potential to revolutionize personalized ophthalmic care and accelerate research into ocular health and disease. 

---
# Preliminary Explorations with GPT-4o(mni) Native Image Generation 

**Authors**: Pu Cao, Feng Zhou, Junyi Ji, Qingye Kong, Zhixiang Lv, Mingjian Zhang, Xuekun Zhao, Siqi Wu, Yinghui Lin, Qing Song, Lu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05501)  

**Abstract**: Recently, the visual generation ability by GPT-4o(mni) has been unlocked by OpenAI. It demonstrates a very remarkable generation capability with excellent multimodal condition understanding and varied task instructions. In this paper, we aim to explore the capabilities of GPT-4o across various tasks. Inspired by previous study, we constructed a task taxonomy along with a carefully curated set of test samples to conduct a comprehensive qualitative test. Benefiting from GPT-4o's powerful multimodal comprehension, its image-generation process demonstrates abilities surpassing those of traditional image-generation tasks. Thus, regarding the dimensions of model capabilities, we evaluate its performance across six task categories: traditional image generation tasks, discriminative tasks, knowledge-based generation, commonsense-based generation, spatially-aware image generation, and temporally-aware image generation. These tasks not only assess the quality and conditional alignment of the model's outputs but also probe deeper into GPT-4o's understanding of real-world concepts. Our results reveal that GPT-4o performs impressively well in general-purpose synthesis tasks, showing strong capabilities in text-to-image generation, visual stylization, and low-level image processing. However, significant limitations remain in its ability to perform precise spatial reasoning, instruction-grounded generation, and consistent temporal prediction. Furthermore, when faced with knowledge-intensive or domain-specific scenarios, such as scientific illustrations or mathematical plots, the model often exhibits hallucinations, factual errors, or structural inconsistencies. These findings suggest that while GPT-4o marks a substantial advancement in unified multimodal generation, there is still a long way to go before it can be reliably applied to professional or safety-critical domains. 

---
# An Overview of the Prospects and Challenges of Using Artificial Intelligence for Energy Management Systems in Microgrids 

**Authors**: Noor ul Misbah Khanum, Hayssam Dahrouj, Ramesh C. Bansal, Hissam Mouayad Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2505.05498)  

**Abstract**: Microgrids have emerged as a pivotal solution in the quest for a sustainable and energy-efficient future. While microgrids offer numerous advantages, they are also prone to issues related to reliably forecasting renewable energy demand and production, protecting against cyberattacks, controlling operational costs, optimizing power flow, and regulating the performance of energy management systems (EMS). Tackling these energy management challenges is essential to facilitate microgrid applications and seamlessly incorporate renewable energy resources. Artificial intelligence (AI) has recently demonstrated immense potential for optimizing energy management in microgrids, providing efficient and reliable solutions. This paper highlights the combined benefits of enabling AI-based methodologies in the energy management systems of microgrids by examining the applicability and efficiency of AI-based EMS in achieving specific technical and economic objectives. The paper also points out several future research directions that promise to spearhead AI-driven EMS, namely the development of self-healing microgrids, integration with blockchain technology, use of Internet of things (IoT), and addressing interpretability, data privacy, scalability, and the prospects to generative AI in the context of future AI-based EMS. 

---
# An Automated LLM-based Pipeline for Asset-Level Database Creation to Assess Deforestation Impact 

**Authors**: Avanija Menon, Ovidiu Serban  

**Link**: [PDF](https://arxiv.org/pdf/2505.05494)  

**Abstract**: The European Union Deforestation Regulation (EUDR) requires companies to prove their products do not contribute to deforestation, creating a critical demand for precise, asset-level environmental impact data. Current databases lack the necessary detail, relying heavily on broad financial metrics and manual data collection, which limits regulatory compliance and accurate environmental modeling. This study presents an automated, end-to-end data extraction pipeline that uses LLMs to create, clean, and validate structured databases, specifically targeting sectors with a high risk of deforestation. The pipeline introduces Instructional, Role-Based, Zero-Shot Chain-of-Thought (IRZ-CoT) prompting to enhance data extraction accuracy and a Retrieval-Augmented Validation (RAV) process that integrates real-time web searches for improved data reliability. Applied to SEC EDGAR filings in the Mining, Oil & Gas, and Utilities sectors, the pipeline demonstrates significant improvements over traditional zero-shot prompting approaches, particularly in extraction accuracy and validation coverage. This work advances NLP-driven automation for regulatory compliance, CSR (Corporate Social Responsibility), and ESG, with broad sectoral applicability. 

---
# DetoxAI: a Python Toolkit for Debiasing Deep Learning Models in Computer Vision 

**Authors**: Ignacy Stępka, Lukasz Sztukiewicz, Michał Wiliński, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.05492)  

**Abstract**: While machine learning fairness has made significant progress in recent years, most existing solutions focus on tabular data and are poorly suited for vision-based classification tasks, which rely heavily on deep learning. To bridge this gap, we introduce DetoxAI, an open-source Python library for improving fairness in deep learning vision classifiers through post-hoc debiasing. DetoxAI implements state-of-the-art debiasing algorithms, fairness metrics, and visualization tools. It supports debiasing via interventions in internal representations and includes attribution-based visualization tools and quantitative algorithmic fairness metrics to show how bias is mitigated. This paper presents the motivation, design, and use cases of DetoxAI, demonstrating its tangible value to engineers and researchers. 

---
# MDDFNet: Mamba-based Dynamic Dual Fusion Network for Traffic Sign Detection 

**Authors**: TianYi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05491)  

**Abstract**: The Detection of small objects, especially traffic signs, is a critical sub-task in object detection and autonomous driving. Despite signficant progress in previous research, two main challenges remain. First, the issue of feature extraction being too singular. Second, the detection process struggles to efectively handle objects of varying sizes or scales. These problems are also prevalent in general object detection tasks. To address these challenges, we propose a novel object detection network, Mamba-based Dynamic Dual Fusion Network (MDDFNet), for traffic sign detection. The network integrates a dynamic dual fusion module and a Mamba-based backbone to simultaneously tackle the aforementioned issues. Specifically, the dynamic dual fusion module utilizes multiple branches to consolidate various spatial and semantic information, thus enhancing feature diversity. The Mamba-based backbone leverages global feature fusion and local feature interaction, combining features in an adaptive manner to generate unique classification characteristics. Extensive experiments conducted on the TT100K (Tsinghua-Tencent 100K) datasets demonstrate that MDDFNet outperforms other state-of-the-art detectors, maintaining real-time processing capabilities of single-stage models while achieving superior performance. This confirms the efectiveness of MDDFNet in detecting small traffic signs. 

---
# FedAvgen: Metadata for Model Aggregation In Communication Systems 

**Authors**: Anthony Kiggundu, Dennis Krummacker, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2505.05486)  

**Abstract**: To improve business efficiency and minimize costs, Artificial Intelligence (AI) practitioners have adopted a shift from formulating models from scratch towards sharing pretrained models. The pretrained models are then aggregated into a global model with higher generalization capabilities, which is afterwards distributed to the client devices. This approach is known as federated learning and inherently utilizes different techniques to select the candidate client models averaged to obtain the global model. This approach, in the case of communication systems, faces challenges arising from the existential diversity in device profiles. The multiplicity in profiles motivates our conceptual assessment of a metaheuristic algorithm (FedAvgen), which relates each pretrained model with its weight space as metadata, to a phenotype and genotype, respectively. This parent-child genetic evolution characterizes the global averaging step in federated learning. We then compare the results of our approach to two widely adopted baseline federated learning algorithms like Federated Averaging (FedAvg) and Federated Stochastic Gradient Descent (FedSGD). 

---
# Structure & Quality: Conceptual and Formal Foundations for the Mind-Body Problem 

**Authors**: Ryan Williams  

**Link**: [PDF](https://arxiv.org/pdf/2505.05481)  

**Abstract**: This paper explores the hard problem of consciousness from a different perspective. Instead of drawing distinctions between the physical and the mental, an exploration of a more foundational relationship is examined: the relationship between structure and quality.
Information-theoretic measures are developed to quantify the mutual determinability between structure and quality, including a novel Q-S space for analyzing fidelity between the two domains. This novel space naturally points toward a five-fold categorization of possible relationships between structural and qualitative properties, illustrating each through conceptual and formal models.
The ontological implications of each category are examined, shedding light on debates around functionalism, emergentism, idealism, panpsychism, and neutral monism.
This new line of inquiry has established a framework for deriving theoretical constraints on qualitative systems undergoing evolution that is explored in my companion paper, Qualia & Natural Selection. 

---
# CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations 

**Authors**: Anthony Liang, Pavel Czempin, Matthew Hong, Yutai Zhou, Erdem Biyik, Stephen Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04999)  

**Abstract**: Learning robot policies using imitation learning requires collecting large amounts of costly action-labeled expert demonstrations, which fundamentally limits the scale of training data. A promising approach to address this bottleneck is to harness the abundance of unlabeled observations-e.g., from video demonstrations-to learn latent action labels in an unsupervised way. However, we find that existing methods struggle when applied to complex robot tasks requiring fine-grained motions. We design continuous latent action models (CLAM) which incorporate two key ingredients we find necessary for learning to solve complex continuous control tasks from unlabeled observation data: (a) using continuous latent action labels instead of discrete representations, and (b) jointly training an action decoder to ensure that the latent action space can be easily grounded to real actions with relatively few labeled examples. Importantly, the labeled examples can be collected from non-optimal play data, enabling CLAM to learn performant policies without access to any action-labeled expert data. We demonstrate on continuous control benchmarks in DMControl (locomotion) and MetaWorld (manipulation), as well as on a real WidowX robot arm that CLAM significantly outperforms prior state-of-the-art methods, remarkably with a 2-3x improvement in task success rate compared to the best baseline. Videos and code can be found at this http URL. 

---
