# LLM-Augmented Graph Neural Recommenders: Integrating User Reviews 

**Authors**: Hiroki Kanezashi, Toyotaro Suzumura, Cade Reid, Md Mostafizur Rahman, Yu Hirate  

**Link**: [PDF](https://arxiv.org/pdf/2504.02195)  

**Abstract**: Recommender systems increasingly aim to combine signals from both user reviews and purchase (or other interaction) behaviors. While user-written comments provide explicit insights about preferences, merging these textual representations from large language models (LLMs) with graph-based embeddings of user actions remains a challenging task. In this work, we propose a framework that employs both a Graph Neural Network (GNN)-based model and an LLM to produce review-aware representations, preserving review semantics while mitigating textual noise. Our approach utilizes a hybrid objective that balances user-item interactions against text-derived features, ensuring that user's both behavioral and linguistic signals are effectively captured. We evaluate this method on multiple datasets from diverse application domains, demonstrating consistent improvements over a baseline GNN-based recommender model. Notably, our model achieves significant gains in recommendation accuracy when review data is sparse or unevenly distributed. These findings highlight the importance of integrating LLM-driven textual feedback with GNN-derived user behavioral patterns to develop robust, context-aware recommender systems. 

---
# Affordable AI Assistants with Knowledge Graph of Thoughts 

**Authors**: Maciej Besta, Lorenzo Paleari, Jia Hao Andrea Jiang, Robert Gerstenberger, You Wu, Patrick Iff, Ales Kubicek, Piotr Nyczyk, Diana Khimey, Jón Gunnar Hannesson, Grzegorz Kwaśniewski, Marcin Copik, Hubert Niewiadomski, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2504.02670)  

**Abstract**: Large Language Models (LLMs) are revolutionizing the development of AI assistants capable of performing diverse tasks across domains. However, current state-of-the-art LLM-driven agents face significant challenges, including high operational costs and limited success rates on complex benchmarks like GAIA. To address these issues, we propose the Knowledge Graph of Thoughts (KGoT), an innovative AI assistant architecture that integrates LLM reasoning with dynamically constructed knowledge graphs (KGs). KGoT extracts and structures task-relevant knowledge into a dynamic KG representation, iteratively enhanced through external tools such as math solvers, web crawlers, and Python scripts. Such structured representation of task-relevant knowledge enables low-cost models to solve complex tasks effectively. For example, KGoT achieves a 29% improvement in task success rates on the GAIA benchmark compared to Hugging Face Agents with GPT-4o mini, while reducing costs by over 36x compared to GPT-4o. Improvements for recent reasoning models are similar, e.g., 36% and 37.5% for Qwen2.5-32B and Deepseek-R1-70B, respectively. KGoT offers a scalable, affordable, and high-performing solution for AI assistants. 

---
# Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation 

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.02458)  

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN. 

---
# A Memory-Augmented LLM-Driven Method for Autonomous Merging of 3D Printing Work Orders 

**Authors**: Yuhao Liu, Maolin Yang, Pingyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02509)  

**Abstract**: With the rapid development of 3D printing, the demand for personalized and customized production on the manufacturing line is steadily increasing. Efficient merging of printing workpieces can significantly enhance the processing efficiency of the production line. Addressing the challenge, a Large Language Model (LLM)-driven method is established in this paper for the autonomous merging of 3D printing work orders, integrated with a memory-augmented learning strategy. In industrial scenarios, both device and order features are modeled into LLM-readable natural language prompt templates, and develop an order-device matching tool along with a merging interference checking module. By incorporating a self-memory learning strategy, an intelligent agent for autonomous order merging is constructed, resulting in improved accuracy and precision in order allocation. The proposed method effectively leverages the strengths of LLMs in industrial applications while reducing hallucination. 

---
# The Self-Learning Agent with a Progressive Neural Network Integrated Transformer 

**Authors**: Ajay Sivakumar, Shalini, Vasantha Raj, Sebastian Sylvester  

**Link**: [PDF](https://arxiv.org/pdf/2504.02489)  

**Abstract**: This paper introduces a self-learning agent that integrates LLaMA 3.2 with a Progressive Neural Network (PNN) for continual learning in conversational AI and code generation. The framework dynamically collects data, fine-tunes tasks with minimal samples, and leverages Meta-Learning for rapid adaptation. LoRA optimizes fine-tuning, while Elastic Weight Consolidation (EWC) enhances knowledge retention. Experimental results demonstrate improved adaptability and memory stability, positioning this approach as a scalable step toward Artificial General Intelligence (AGI). 

---
# Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions 

**Authors**: PeiJie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02623)  

**Abstract**: Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society. 

---
# Narrative Studio: Visual narrative exploration using LLMs and Monte Carlo Tree Search 

**Authors**: Parsa Ghaffari, Chris Hokamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.02426)  

**Abstract**: Interactive storytelling benefits from planning and exploring multiple 'what if' scenarios. Modern LLMs are useful tools for ideation and exploration, but current chat-based user interfaces restrict users to a single linear flow. To address this limitation, we propose Narrative Studio -- a novel in-browser narrative exploration environment featuring a tree-like interface that allows branching exploration from user-defined points in a story. Each branch is extended via iterative LLM inference guided by system and user-defined prompts. Additionally, we employ Monte Carlo Tree Search (MCTS) to automatically expand promising narrative paths based on user-specified criteria, enabling more diverse and robust story development. We also allow users to enhance narrative coherence by grounding the generated text in an entity graph that represents the actors and environment of the story. 

---
# A Survey of Scaling in Large Language Model Reasoning 

**Authors**: Zihan Chen, Song Wang, Zhen Tan, Xingbo Fu, Zhenyu Lei, Peng Wang, Huan Liu, Cong Shen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.02181)  

**Abstract**: The rapid advancements in large Language models (LLMs) have significantly enhanced their reasoning capabilities, driven by various strategies such as multi-agent collaboration. However, unlike the well-established performance improvements achieved through scaling data and model size, the scaling of reasoning in LLMs is more complex and can even negatively impact reasoning performance, introducing new challenges in model alignment and robustness. In this survey, we provide a comprehensive examination of scaling in LLM reasoning, categorizing it into multiple dimensions and analyzing how and to what extent different scaling strategies contribute to improving reasoning capabilities. We begin by exploring scaling in input size, which enables LLMs to process and utilize more extensive context for improved reasoning. Next, we analyze scaling in reasoning steps that improves multi-step inference and logical consistency. We then examine scaling in reasoning rounds, where iterative interactions refine reasoning outcomes. Furthermore, we discuss scaling in training-enabled reasoning, focusing on optimization through iterative model improvement. Finally, we review applications of scaling across domains and outline future directions for further advancing LLM reasoning. By synthesizing these diverse perspectives, this survey aims to provide insights into how scaling strategies fundamentally enhance the reasoning capabilities of LLMs and further guide the development of next-generation AI systems. 

---
# Exploring LLM Reasoning Through Controlled Prompt Variations 

**Authors**: Giannis Chatziveroglou, Richard Yun, Maura Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2504.02111)  

**Abstract**: This study investigates the reasoning robustness of large language models (LLMs) on mathematical problem-solving tasks under systematically introduced input perturbations. Using the GSM8K dataset as a controlled testbed, we evaluate how well state-of-the-art models maintain logical consistency and correctness when confronted with four categories of prompt perturbations: irrelevant context, pathological instructions, factually relevant but non-essential context, and a combination of the latter two. Our experiments, conducted on thirteen open-source and closed-source LLMs, reveal that introducing irrelevant context within the model's context window significantly degrades performance, suggesting that distinguishing essential from extraneous details remains a pressing challenge. Surprisingly, performance regressions are relatively insensitive to the complexity of the reasoning task, as measured by the number of steps required, and are not strictly correlated with model size. Moreover, we observe that certain perturbations inadvertently trigger chain-of-thought-like reasoning behaviors, even without explicit prompting. Our findings highlight critical vulnerabilities in current LLMs and underscore the need for improved robustness against noisy, misleading, and contextually dense inputs, paving the way for more resilient and reliable reasoning in real-world applications. 

---
# Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics 

**Authors**: Hamed Mahdavi, Alireza Hashemi, Majid Daliri, Pegah Mohammadipour, Alireza Farhadi, Samira Malek, Yekta Yazdanifard, Amir Khasahmadi, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2504.01995)  

**Abstract**: Recent advancements in large language models (LLMs) have shown impressive progress in mathematical reasoning tasks. However, current evaluation benchmarks predominantly focus on the accuracy of final answers, often overlooking the logical rigor crucial for mathematical problem-solving. The claim that state-of-the-art LLMs can solve Math Olympiad-level problems requires closer examination. To explore this, we conducted both qualitative and quantitative human evaluations of proofs generated by LLMs, and developed a schema for automatically assessing their reasoning capabilities. Our study reveals that current LLMs fall significantly short of solving challenging Olympiad-level problems and frequently fail to distinguish correct mathematical reasoning from clearly flawed solutions. We also found that occasional correct final answers provided by LLMs often result from pattern recognition or heuristic shortcuts rather than genuine mathematical reasoning. These findings underscore the substantial gap between LLM performance and human expertise in advanced mathematical reasoning and highlight the importance of developing benchmarks that prioritize the rigor and coherence of mathematical arguments rather than merely the correctness of final answers. 

---
# Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models 

**Authors**: Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2504.02821)  

**Abstract**: Sparse Autoencoders (SAEs) have recently been shown to enhance interpretability and steerability in Large Language Models (LLMs). In this work, we extend the application of SAEs to Vision-Language Models (VLMs), such as CLIP, and introduce a comprehensive framework for evaluating monosemanticity in vision representations. Our experimental results reveal that SAEs trained on VLMs significantly enhance the monosemanticity of individual neurons while also exhibiting hierarchical representations that align well with expert-defined structures (e.g., iNaturalist taxonomy). Most notably, we demonstrate that applying SAEs to intervene on a CLIP vision encoder, directly steer output from multimodal LLMs (e.g., LLaVA) without any modifications to the underlying model. These findings emphasize the practicality and efficacy of SAEs as an unsupervised approach for enhancing both the interpretability and control of VLMs. 

---
# Generative Evaluation of Complex Reasoning in Large Language Models 

**Authors**: Haowei Lin, Xiangyu Wang, Ruilin Yan, Baizhou Huang, Haotian Ye, Jianhua Zhu, Zihao Wang, James Zou, Jianzhu Ma, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02810)  

**Abstract**: With powerful large language models (LLMs) demonstrating superhuman reasoning capabilities, a critical question arises: Do LLMs genuinely reason, or do they merely recall answers from their extensive, web-scraped training datasets? Publicly released benchmarks inevitably become contaminated once incorporated into subsequent LLM training sets, undermining their reliability as faithful assessments. To address this, we introduce KUMO, a generative evaluation framework designed specifically for assessing reasoning in LLMs. KUMO synergistically combines LLMs with symbolic engines to dynamically produce diverse, multi-turn reasoning tasks that are partially observable and adjustable in difficulty. Through an automated pipeline, KUMO continuously generates novel tasks across open-ended domains, compelling models to demonstrate genuine generalization rather than memorization. We evaluated 23 state-of-the-art LLMs on 5,000 tasks across 100 domains created by KUMO, benchmarking their reasoning abilities against university students. Our findings reveal that many LLMs have outperformed university-level performance on easy reasoning tasks, and reasoning-scaled LLMs reach university-level performance on complex reasoning challenges. Moreover, LLM performance on KUMO tasks correlates strongly with results on newly released real-world reasoning benchmarks, underscoring KUMO's value as a robust, enduring assessment tool for genuine LLM reasoning capabilities. 

---
# OmniCellTOSG: The First Cell Text-Omic Signaling Graphs Dataset for Joint LLM and GNN Modeling 

**Authors**: Heming Zhang, Tim Xu, Dekang Cao, Shunning Liang, Lars Schimmelpfennig, Levi Kaster, Di Huang, Carlos Cruchaga, Guangfu Li, Michael Province, Yixin Chen, Philip Payne, Fuhai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.02148)  

**Abstract**: Complex cell signaling systems -- governed by varying protein abundances and interactions -- generate diverse cell types across organs. These systems evolve under influences such as age, sex, diet, environmental exposures, and diseases, making them challenging to decode given the involvement of tens of thousands of genes and proteins. Recently, hundreds of millions of single-cell omics data have provided a robust foundation for understanding these signaling networks within various cell subpopulations and conditions. Inspired by the success of large foundation models (for example, large language models and large vision models) pre-trained on massive datasets, we introduce OmniCellTOSG, the first dataset of cell text-omic signaling graphs (TOSGs). Each TOSG represents the signaling network of an individual or meta-cell and is labeled with information such as organ, disease, sex, age, and cell subtype. OmniCellTOSG offers two key contributions. First, it introduces a novel graph model that integrates human-readable annotations -- such as biological functions, cellular locations, signaling pathways, related diseases, and drugs -- with quantitative gene and protein abundance data, enabling graph reasoning to decode cell signaling. This approach calls for new joint models combining large language models and graph neural networks. Second, the dataset is built from single-cell RNA sequencing data of approximately 120 million cells from diverse tissues and conditions (healthy and diseased) and is fully compatible with PyTorch. This facilitates the development of innovative cell signaling models that could transform research in life sciences, healthcare, and precision medicine. The OmniCellTOSG dataset is continuously expanding and will be updated regularly. The dataset and code are available at this https URL. 

---
# From Consumption to Collaboration: Measuring Interaction Patterns to Augment Human Cognition in Open-Ended Tasks 

**Authors**: Joshua Holstein, Moritz Diener, Philipp Spitzer  

**Link**: [PDF](https://arxiv.org/pdf/2504.02780)  

**Abstract**: The rise of Generative AI, and Large Language Models (LLMs) in particular, is fundamentally changing cognitive processes in knowledge work, raising critical questions about their impact on human reasoning and problem-solving capabilities. As these AI systems become increasingly integrated into workflows, they offer unprecedented opportunities for augmenting human thinking while simultaneously risking cognitive erosion through passive consumption of generated answers. This tension is particularly pronounced in open-ended tasks, where effective solutions require deep contextualization and integration of domain knowledge. Unlike structured tasks with established metrics, measuring the quality of human-LLM interaction in such open-ended tasks poses significant challenges due to the absence of ground truth and the iterative nature of solution development. To address this, we present a framework that analyzes interaction patterns along two dimensions: cognitive activity mode (exploration vs. exploitation) and cognitive engagement mode (constructive vs. detrimental). This framework provides systematic measurements to evaluate when LLMs are effective tools for thought rather than substitutes for human cognition, advancing theoretical understanding and practical guidance for developing AI systems that protect and augment human cognitive capabilities. 

---
# How Deep Do Large Language Models Internalize Scientific Literature and Citation Practices? 

**Authors**: Andres Algaba, Vincent Holst, Floriano Tori, Melika Mobini, Brecht Verbeken, Sylvia Wenmackers, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2504.02767)  

**Abstract**: The spread of scientific knowledge depends on how researchers discover and cite previous work. The adoption of large language models (LLMs) in the scientific research process introduces a new layer to these citation practices. However, it remains unclear to what extent LLMs align with human citation practices, how they perform across domains, and may influence citation dynamics. Here, we show that LLMs systematically reinforce the Matthew effect in citations by consistently favoring highly cited papers when generating references. This pattern persists across scientific domains despite significant field-specific variations in existence rates, which refer to the proportion of generated references that match existing records in external bibliometric databases. Analyzing 274,951 references generated by GPT-4o for 10,000 papers, we find that LLM recommendations diverge from traditional citation patterns by preferring more recent references with shorter titles and fewer authors. Emphasizing their content-level relevance, the generated references are semantically aligned with the content of each paper at levels comparable to the ground truth references and display similar network effects while reducing author self-citations. These findings illustrate how LLMs may reshape citation practices and influence the trajectory of scientific discovery by reflecting and amplifying established trends. As LLMs become more integrated into the scientific research process, it is important to understand their role in shaping how scientific communities discover and build upon prior work. 

---
# Cognitive Memory in Large Language Models 

**Authors**: Lianlei Shan, Shixian Luo, Zezhou Zhu, Yu Yuan, Yong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02441)  

**Abstract**: This paper examines memory mechanisms in Large Language Models (LLMs), emphasizing their importance for context-rich responses, reduced hallucinations, and improved efficiency. It categorizes memory into sensory, short-term, and long-term, with sensory memory corresponding to input prompts, short-term memory processing immediate context, and long-term memory implemented via external databases or structures. The text-based memory section covers acquisition (selection and summarization), management (updating, accessing, storing, and resolving conflicts), and utilization (full-text search, SQL queries, semantic search). The KV cache-based memory section discusses selection methods (regularity-based summarization, score-based approaches, special token embeddings) and compression techniques (low-rank compression, KV merging, multimodal compression), along with management strategies like offloading and shared attention mechanisms. Parameter-based memory methods (LoRA, TTT, MoE) transform memories into model parameters to enhance efficiency, while hidden-state-based memory approaches (chunk mechanisms, recurrent transformers, Mamba model) improve long-text processing by combining RNN hidden states with current methods. Overall, the paper offers a comprehensive analysis of LLM memory mechanisms, highlighting their significance and future research directions. 

---
# Inference-Time Scaling for Generalist Reward Modeling 

**Authors**: Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02495)  

**Abstract**: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $\textit{proper learning methods could enable effective inference-time scalability}$. A key challenge of RL is to obtain accurate reward signals for LLMs in various domains beyond verifiable questions or artificial rules. In this work, we investigate how to improve reward modeling (RM) with more inference compute for general queries, i.e. the $\textbf{inference-time scalability of generalist RM}$, and further, how to improve the effectiveness of performance-compute scaling with proper learning methods. For the RM approach, we adopt pointwise generative reward modeling (GRM) to enable flexibility for different input types and potential for inference-time scaling. For the learning method, we propose Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL, to generate principles adaptively and critiques accurately, resulting in $\textbf{DeepSeek-GRM}$ models. Furthermore, for effective inference-time scaling, we use parallel sampling to expand compute usage, and introduce a meta RM to guide voting process for better scaling performance. Empirically, we show that SPCT significantly improves the quality and scalability of GRMs, outperforming existing methods and models in various RM benchmarks without severe biases, and could achieve better performance compared to training-time scaling. DeepSeek-GRM still meets challenges in some tasks, which we believe can be addressed by future efforts in generalist reward systems. The models will be released and open-sourced. 

---
# OmniCam: Unified Multimodal Video Generation via Camera Control 

**Authors**: Xiaoda Yang, Jiayang Xu, Kaixuan Luan, Xinyu Zhan, Hongshun Qiu, Shijun Shi, Hao Li, Shuai Yang, Li Zhang, Checheng Yu, Cewu Lu, Lixin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02312)  

**Abstract**: Camera control, which achieves diverse visual effects by changing camera position and pose, has attracted widespread attention. However, existing methods face challenges such as complex interaction and limited control capabilities. To address these issues, we present OmniCam, a unified multimodal camera control framework. Leveraging large language models and video diffusion models, OmniCam generates spatio-temporally consistent videos. It supports various combinations of input modalities: the user can provide text or video with expected trajectory as camera path guidance, and image or video as content reference, enabling precise control over camera motion. To facilitate the training of OmniCam, we introduce the OmniTr dataset, which contains a large collection of high-quality long-sequence trajectories, videos, and corresponding descriptions. Experimental results demonstrate that our model achieves state-of-the-art performance in high-quality camera-controlled video generation across various metrics. 

---
# LLMs as Deceptive Agents: How Role-Based Prompting Induces Semantic Ambiguity in Puzzle Tasks 

**Authors**: Seunghyun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02254)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have not only showcased impressive creative capabilities but also revealed emerging agentic behaviors that exploit linguistic ambiguity in adversarial settings. In this study, we investigate how an LLM, acting as an autonomous agent, leverages semantic ambiguity to generate deceptive puzzles that mislead and challenge human users. Inspired by the popular puzzle game "Connections", we systematically compare puzzles produced through zero-shot prompting, role-injected adversarial prompts, and human-crafted examples, with an emphasis on understanding the underlying agent decision-making processes. Employing computational analyses with HateBERT to quantify semantic ambiguity, alongside subjective human evaluations, we demonstrate that explicit adversarial agent behaviors significantly heighten semantic ambiguity -- thereby increasing cognitive load and reducing fairness in puzzle solving. These findings provide critical insights into the emergent agentic qualities of LLMs and underscore important ethical considerations for evaluating and safely deploying autonomous language systems in both educational technologies and entertainment. 

---
# State-of-the-Art Translation of Text-to-Gloss using mBART : A case study of Bangla 

**Authors**: Sharif Md. Abdullah, Abhijit Paul, Shebuti Rayana, Ahmedul Kabir, Zarif Masud  

**Link**: [PDF](https://arxiv.org/pdf/2504.02293)  

**Abstract**: Despite a large deaf and dumb population of 1.7 million, Bangla Sign Language (BdSL) remains a understudied domain. Specifically, there are no works on Bangla text-to-gloss translation task. To address this gap, we begin by addressing the dataset problem. We take inspiration from grammatical rule based gloss generation used in Germany and American sign langauage (ASL) and adapt it for BdSL. We also leverage LLM to generate synthetic data and use back-translation, text generation for data augmentation. With dataset prepared, we started experimentation. We fine-tuned pretrained mBART-50 and mBERT-multiclass-uncased model on our dataset. We also trained GRU, RNN and a novel seq-to-seq model with multi-head attention. We observe significant high performance (ScareBLEU=79.53) with fine-tuning pretrained mBART-50 multilingual model from Facebook. We then explored why we observe such high performance with mBART. We soon notice an interesting property of mBART -- it was trained on shuffled and masked text data. And as we know, gloss form has shuffling property. So we hypothesize that mBART is inherently good at text-to-gloss tasks. To find support against this hypothesis, we trained mBART-50 on PHOENIX-14T benchmark and evaluated it with existing literature. Our mBART-50 finetune demonstrated State-of-the-Art performance on PHOENIX-14T benchmark, far outperforming existing models in all 6 metrics (ScareBLEU = 63.89, BLEU-1 = 55.14, BLEU-2 = 38.07, BLEU-3 = 27.13, BLEU-4 = 20.68, COMET = 0.624). Based on the results, this study proposes a new paradigm for text-to-gloss task using mBART models. Additionally, our results show that BdSL text-to-gloss task can greatly benefit from rule-based synthetic dataset. 

---
# LLM Social Simulations Are a Promising Research Method 

**Authors**: Jacy Reese Anthis, Ryan Liu, Sean M. Richardson, Austin C. Kozlowski, Bernard Koch, James Evans, Erik Brynjolfsson, Michael Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2504.02234)  

**Abstract**: Accurate and verifiable large language model (LLM) simulations of human research subjects promise an accessible data source for understanding human behavior and training new AI systems. However, results to date have been limited, and few social scientists have adopted these methods. In this position paper, we argue that the promise of LLM social simulations can be achieved by addressing five tractable challenges. We ground our argument in a literature survey of empirical comparisons between LLMs and human research subjects, commentaries on the topic, and related work. We identify promising directions with prompting, fine-tuning, and complementary methods. We believe that LLM social simulations can already be used for exploratory research, such as pilot experiments for psychology, economics, sociology, and marketing. More widespread use may soon be possible with rapidly advancing LLM capabilities, and researchers should prioritize developing conceptual models and evaluations that can be iteratively deployed and refined at pace with ongoing AI advances. 

---
# On Simulation-Guided LLM-based Code Generation for Safe Autonomous Driving Software 

**Authors**: Ali Nouri, Johan Andersson, Kailash De Jesus Hornig, Zhennan Fei, Emil Knabe, Hakan Sivencrona, Beatriz Cabrero-Daniel, Christian Berger  

**Link**: [PDF](https://arxiv.org/pdf/2504.02141)  

**Abstract**: Automated Driving System (ADS) is a safety-critical software system responsible for the interpretation of the vehicle's environment and making decisions accordingly. The unbounded complexity of the driving context, including unforeseeable events, necessitate continuous improvement, often achieved through iterative DevOps processes. However, DevOps processes are themselves complex, making these improvements both time- and resource-intensive. Automation in code generation for ADS using Large Language Models (LLM) is one potential approach to address this challenge. Nevertheless, the development of ADS requires rigorous processes to verify, validate, assess, and qualify the code before it can be deployed in the vehicle and used. In this study, we developed and evaluated a prototype for automatic code generation and assessment using a designed pipeline of a LLM-based agent, simulation model, and rule-based feedback generator in an industrial setup. The LLM-generated code is evaluated automatically in a simulation model against multiple critical traffic scenarios, and an assessment report is provided as feedback to the LLM for modification or bug fixing. We report about the experimental results of the prototype employing Codellama:34b, DeepSeek (r1:32b and Coder:33b), CodeGemma:7b, Mistral:7b, and GPT4 for Adaptive Cruise Control (ACC) and Unsupervised Collision Avoidance by Evasive Manoeuvre (CAEM). We finally assessed the tool with 11 experts at two Original Equipment Manufacturers (OEMs) by conducting an interview study. 

---
# LLMPi: Optimizing LLMs for High-Throughput on Raspberry Pi 

**Authors**: Mahsa Ardakani, Jinendra Malekar, Ramtin Zand  

**Link**: [PDF](https://arxiv.org/pdf/2504.02118)  

**Abstract**: Deploying Large Language Models (LLMs) on resource-constrained edge devices like the Raspberry Pi presents challenges in computational efficiency, power consumption, and response latency. This paper explores quantization-based optimization techniques to enable high-throughput, energy-efficient execution of LLMs on low-power embedded systems. Our approach leverages k-quantization, a Post-Training Quantization (PTQ) method designed for different bit-widths, enabling efficient 2-bit, 4-bit, 6-bit, and 8-bit weight quantization. Additionally, we employ ternary quantization using Quantization-Aware Training (QAT) for BitNet models, allowing for more effective adaptation to lower-bit representations while preserving accuracy.
Our findings highlight the potential of quantized LLMs for real-time conversational AI on edge devices, paving the way for low-power, high-efficiency AI deployment in mobile and embedded applications. This study demonstrates that aggressive quantization strategies can significantly reduce energy consumption while maintaining inference quality, making LLMs practical for resource-limited environments. 

---
# Achieving Unanimous Consensus in Decision Making Using Multi-Agents 

**Authors**: Apurba Pokharel, Ram Dantu, Shakila Zaman, Sirisha Talapuru, Vinh Quach  

**Link**: [PDF](https://arxiv.org/pdf/2504.02128)  

**Abstract**: Blockchain consensus mechanisms have relied on algorithms such as Proof-of-Work (PoW) and Proof-of-Stake (PoS) to ensure network functionality and integrity. However, these approaches struggle with adaptability for decision-making where the opinions of each matter rather than reaching an agreement based on honest majority or weighted consensus. This paper introduces a novel deliberation-based consensus mechanism where Large Language Models (LLMs) act as rational agents engaging in structured discussions to reach a unanimous consensus. By leveraging graded consensus and a multi-round deliberation process, our approach ensures both unanimous consensus for definitive problems and graded confidence for prioritized decisions and policies. We provide a formalization of our system and use it to show that the properties of blockchains: consistency, agreement, liveness, and determinism are maintained. Moreover, experimental results demonstrate our system's feasibility, showcasing how our deliberation method's convergence, block properties, and accuracy enable decision-making on blockchain networks. We also address key challenges with this novel approach such as degeneration of thoughts, hallucinations, malicious models and nodes, resource consumption, and scalability. 

---
# FlowDistill: Scalable Traffic Flow Prediction via Distillation from LLMs 

**Authors**: Chenyang Yu, Xinpeng Xie, Yan Huang, Chenxi Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02094)  

**Abstract**: Accurate traffic flow prediction is vital for optimizing urban mobility, yet it remains difficult in many cities due to complex spatio-temporal dependencies and limited high-quality data. While deep graph-based models demonstrate strong predictive power, their performance often comes at the cost of high computational overhead and substantial training data requirements, making them impractical for deployment in resource-constrained or data-scarce environments. We propose the FlowDistill, a lightweight and scalable traffic prediction framework based on knowledge distillation from large language models (LLMs). In this teacher-student setup, a fine-tuned LLM guides a compact multi-layer perceptron (MLP) student model using a novel combination of the information bottleneck principle and teacher-bounded regression loss, ensuring the distilled model retains only essential and transferable knowledge. Spatial and temporal correlations are explicitly encoded to enhance the model's generalization across diverse urban settings. Despite its simplicity, FlowDistill consistently outperforms state-of-the-art models in prediction accuracy while requiring significantly less training data, and achieving lower memory usage and inference latency, highlighting its efficiency and suitability for real-world, scalable deployment. 

---
# Trapped by Expectations: Functional Fixedness in LLM-Enabled Chat Search 

**Authors**: Jiqun Liu, Jamshed Karimnazarov, Ryen W. White  

**Link**: [PDF](https://arxiv.org/pdf/2504.02074)  

**Abstract**: Functional fixedness, a cognitive bias that restricts users' interactions with a new system or tool to expected or familiar ways, limits the full potential of Large Language Model (LLM)-enabled chat search, especially in complex and exploratory tasks. To investigate its impact, we conducted a crowdsourcing study with 450 participants, each completing one of six decision-making tasks spanning public safety, diet and health management, sustainability, and AI ethics. Participants engaged in a multi-prompt conversation with ChatGPT to address the task, allowing us to compare pre-chat intent-based expectations with observed interactions. We found that: 1) Several aspects of pre-chat expectations are closely associated with users' prior experiences with ChatGPT, search engines, and virtual assistants; 2) Prior system experience shapes language use and prompting behavior. Frequent ChatGPT users reduced deictic terms and hedge words and frequently adjusted prompts. Users with rich search experience maintained structured, less-conversational queries with minimal modifications. Users of virtual assistants favored directive, command-like prompts, reinforcing functional fixedness; 3) When the system failed to meet expectations, participants generated more detailed prompts with increased linguistic diversity, reflecting adaptive shifts. These findings suggest that while preconceived expectations constrain early interactions, unmet expectations can motivate behavioral adaptation. With appropriate system support, this may promote broader exploration of LLM capabilities. This work also introduces a typology for user intents in chat search and highlights the importance of mitigating functional fixedness to support more creative and analytical use of LLMs. 

---
# PIM-LLM: A High-Throughput Hybrid PIM Architecture for 1-bit LLMs 

**Authors**: Jinendra Malekar, Peyton Chandarana, Md Hasibul Amin, Mohammed E. Elbtity, Ramtin Zand  

**Link**: [PDF](https://arxiv.org/pdf/2504.01994)  

**Abstract**: In this paper, we propose PIM-LLM, a hybrid architecture developed to accelerate 1-bit large language models (LLMs). PIM-LLM leverages analog processing-in-memory (PIM) architectures and digital systolic arrays to accelerate low-precision matrix multiplication (MatMul) operations in projection layers and high-precision MatMul operations in attention heads of 1-bit LLMs, respectively. Our design achieves up to roughly 80x improvement in tokens per second and a 70% increase in tokens per joule compared to conventional hardware accelerators. Additionally, PIM-LLM outperforms previous PIM-based LLM accelerators, setting a new benchmark with at least 2x and 5x improvement in GOPS and GOPS/W, respectively. 

---
# TuRTLe: A Unified Evaluation of LLMs for RTL Generation 

**Authors**: Dario Garcia-Gasulla, Gokcen Kestor, Emanuele Parisi, Miquel Albert'i-Binimelis, Cristian Gutierrez, Razine Moundir Ghorab, Orlando Montenegro, Bernat Homs, Miquel Moreto  

**Link**: [PDF](https://arxiv.org/pdf/2504.01986)  

**Abstract**: The rapid advancements in LLMs have driven the adoption of generative AI in various domains, including Electronic Design Automation (EDA). Unlike traditional software development, EDA presents unique challenges, as generated RTL code must not only be syntactically correct and functionally accurate but also synthesizable by hardware generators while meeting performance, power, and area constraints. These additional requirements introduce complexities that existing code-generation benchmarks often fail to capture, limiting their effectiveness in evaluating LLMs for RTL generation. To address this gap, we propose TuRTLe, a unified evaluation framework designed to systematically assess LLMs across key RTL generation tasks. TuRTLe integrates multiple existing benchmarks and automates the evaluation process, enabling a comprehensive assessment of LLM performance in syntax correctness, functional correctness, synthesis, PPA optimization, and exact line completion. Using this framework, we benchmark a diverse set of open LLMs and analyze their strengths and weaknesses in EDA-specific tasks. Our results show that reasoning-based models, such as DeepSeek R1, consistently outperform others across multiple evaluation criteria, but at the cost of increased computational overhead and inference latency. Additionally, base models are better suited in module completion tasks, while instruct-tuned models perform better in specification-to-RTL tasks. 

---
# Enhancing LLM Robustness to Perturbed Instructions: An Empirical Study 

**Authors**: Aryan Agrawal, Lisa Alazraki, Shahin Honarvar, Marek Rei  

**Link**: [PDF](https://arxiv.org/pdf/2504.02733)  

**Abstract**: Large Language Models (LLMs) are highly vulnerable to input perturbations, as even a small prompt change may result in a substantially different output. Existing methods to enhance LLM robustness are primarily focused on perturbed data samples, whereas improving resiliency to perturbations of task-level instructions has remained relatively underexplored. In this work, we focus on character- and word-level edits of task-specific instructions, which substantially degrade downstream performance. We experiment with a variety of techniques to enhance the robustness of LLMs, including self-denoising and representation alignment, testing different models (Llama 3 and Flan-T5), datasets (CoLA, QNLI, SST-2) and instructions (both task-oriented and role-oriented). We find that, on average, self-denoising -- whether performed by a frozen LLM or a fine-tuned model -- achieves substantially higher performance gains than alternative strategies, including more complex baselines such as ensembling and supervised methods. 

---
# A Framework for Robust Cognitive Evaluation of LLMs 

**Authors**: Karin de Langis, Jong Inn Park, Bin Hu, Khanh Chi Le, Andreas Schramm, Michael C. Mensink, Andrew Elfenbein, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02789)  

**Abstract**: Emergent cognitive abilities in large language models (LLMs) have been widely observed, but their nature and underlying mechanisms remain poorly understood. A growing body of research draws on cognitive science to investigate LLM cognition, but standard methodologies and experimen-tal pipelines have not yet been established. To address this gap we develop CognitivEval, a framework for systematically evaluating the artificial cognitive capabilities of LLMs, with a particular emphasis on robustness in response collection. The key features of CognitivEval include: (i) automatic prompt permutations, and (ii) testing that gathers both generations and model probability estimates. Our experiments demonstrate that these features lead to more robust experimental outcomes. Using CognitivEval, we replicate five classic experiments in cognitive science, illustrating the framework's generalizability across various experimental tasks and obtaining a cognitive profile of several state of the art LLMs. CognitivEval will be released publicly to foster broader collaboration within the cognitive science community. 

---
# Why do LLMs attend to the first token? 

**Authors**: Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličkovi ć, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02732)  

**Abstract**: Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training. 

---
# A Survey of Large Language Models in Mental Health Disorder Detection on Social Media 

**Authors**: Zhuohan Ge, Nicole Hu, Darian Li, Yubo Wang, Shihao Qi, Yuming Xu, Han Shi, Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02800)  

**Abstract**: The detection and intervention of mental health issues represent a critical global research focus, and social media data has been recognized as an important resource for mental health research. However, how to utilize Large Language Models (LLMs) for mental health problem detection on social media poses significant challenges. Hence, this paper aims to explore the potential of LLM applications in social media data analysis, focusing not only on the most common psychological disorders such as depression and anxiety but also incorporating psychotic disorders and externalizing disorders, summarizing the application methods of LLM from different dimensions, such as text data analysis and detection of mental disorders, and revealing the major challenges and shortcomings of current research. In addition, the paper provides an overview of popular datasets, and evaluation metrics. The survey in this paper provides a comprehensive frame of reference for researchers in the field of mental health, while demonstrating the great potential of LLMs in mental health detection to facilitate the further application of LLMs in future mental health interventions. 

---
# LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems 

**Authors**: Zishuo Liu, Carlos Rabat Villarreal, Mostafa Rahgouy, Amit Das, Zheng Zhang, Chang Ren, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.02671)  

**Abstract**: Fermi Problems (FPs) are mathematical reasoning tasks that require human-like logic and numerical reasoning. Unlike other reasoning questions, FPs often involve real-world impracticalities or ambiguous concepts, making them challenging even for humans to solve. Despite advancements in AI, particularly with large language models (LLMs) in various reasoning tasks, FPs remain relatively under-explored. This work conducted an exploratory study to examine the capabilities and limitations of LLMs in solving FPs. We first evaluated the overall performance of three advanced LLMs using a publicly available FP dataset. We designed prompts according to the recently proposed TELeR taxonomy, including a zero-shot scenario. Results indicated that all three LLMs achieved a fp_score (range between 0 - 1) below 0.5, underscoring the inherent difficulty of these reasoning tasks. To further investigate, we categorized FPs into standard and specific questions, hypothesizing that LLMs would perform better on standard questions, which are characterized by clarity and conciseness, than on specific ones. Comparative experiments confirmed this hypothesis, demonstrating that LLMs performed better on standard FPs in terms of both accuracy and efficiency. 

---
# Leveraging LLM For Synchronizing Information Across Multilingual Tables 

**Authors**: Siddharth Khincha, Tushar Kataria, Ankita Anand, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.02559)  

**Abstract**: The vast amount of online information today poses challenges for non-English speakers, as much of it is concentrated in high-resource languages such as English and French. Wikipedia reflects this imbalance, with content in low-resource languages frequently outdated or incomplete. Recent research has sought to improve cross-language synchronization of Wikipedia tables using rule-based methods. These approaches can be effective, but they struggle with complexity and generalization. This paper explores large language models (LLMs) for multilingual information synchronization, using zero-shot prompting as a scalable solution. We introduce the Information Updation dataset, simulating the real-world process of updating outdated Wikipedia tables, and evaluate LLM performance. Our findings reveal that single-prompt approaches often produce suboptimal results, prompting us to introduce a task decomposition strategy that enhances coherence and accuracy. Our proposed method outperforms existing baselines, particularly in Information Updation (1.79%) and Information Addition (20.58%), highlighting the model strength in dynamically updating and enriching data across architectures 

---
# UNDO: Understanding Distillation as Optimization 

**Authors**: Kushal Jain, Piyushi Goyal, Kumar Shridhar  

**Link**: [PDF](https://arxiv.org/pdf/2504.02521)  

**Abstract**: Knowledge distillation has emerged as an effective strategy for compressing large language models' (LLMs) knowledge into smaller, more efficient student models. However, standard one-shot distillation methods often produce suboptimal results due to a mismatch between teacher-generated rationales and the student's specific learning requirements. In this paper, we introduce the UNDO: UNderstanding Distillation as Optimization framework, designed to bridge this gap by iteratively identifying the student's errors and prompting the teacher to refine its explanations accordingly. Each iteration directly targets the student's learning deficiencies, motivating the teacher to provide tailored and enhanced rationales that specifically address these weaknesses. Empirical evaluations on various challenging mathematical and commonsense reasoning tasks demonstrate that our iterative distillation method, UNDO, significantly outperforms standard one-step distillation methods, achieving performance gains of up to 20%. Additionally, we show that teacher-generated data refined through our iterative process remains effective even when applied to different student models, underscoring the broad applicability of our approach. Our work fundamentally reframes knowledge distillation as an iterative teacher-student interaction, effectively leveraging dynamic refinement by the teacher for better knowledge distillation. 

---
# Language Models reach higher Agreement than Humans in Historical Interpretation 

**Authors**: Fabio Celli, Georgios Spathulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02572)  

**Abstract**: This paper compares historical annotations by humans and Large Language Models. The findings reveal that both exhibit some cultural bias, but Large Language Models achieve a higher consensus on the interpretation of historical facts from short texts. While humans tend to disagree on the basis of their personal biases, Large Models disagree when they skip information or produce hallucinations. These findings have significant implications for digital humanities, enabling large-scale annotation and quantitative analysis of historical data. This offers new educational and research opportunities to explore historical interpretations from different Language Models, fostering critical thinking about bias. 

---
# Adapting Large Language Models for Multi-Domain Retrieval-Augmented-Generation 

**Authors**: Alexandre Misrahi, Nadezhda Chirkova, Maxime Louis, Vassilina Nikoulina  

**Link**: [PDF](https://arxiv.org/pdf/2504.02411)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLM factuality, but multi-domain applications face challenges like lack of diverse benchmarks and poor out-of-domain generalization. The first contribution of this work is to introduce a diverse benchmark comprising a variety of question-answering tasks from 8 sources and covering 13 domains. Our second contribution consists in systematically testing out-of-domain generalization for typical RAG tuning strategies. While our findings reveal that standard fine-tuning fails to generalize effectively, we show that sequence-level distillation with teacher-generated labels improves out-of-domain performance by providing more coherent supervision. Our findings highlight key strategies for improving multi-domain RAG robustness. 

---
# AnesBench: Multi-Dimensional Evaluation of LLM Reasoning in Anesthesiology 

**Authors**: Xiang Feng, Wentao Jiang, Zengmao Wang, Yong Luo, Pingbo Xu, Baosheng Yu, Hua Jin, Bo Du, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02404)  

**Abstract**: The application of large language models (LLMs) in the medical field has gained significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. In this paper, we systematically evaluate the reasoning capabilities of LLMs in anesthesiology and analyze key factors influencing their performance. To this end, we introduce AnesBench, a cross-lingual benchmark designed to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Through extensive experiments, we first explore how model characteristics, including model scale, Chain of Thought (CoT) length, and language transferability, affect reasoning performance. Then, we further evaluate the effectiveness of different training strategies, leveraging our curated anesthesiology-related dataset, including continuous pre-training (CPT) and supervised fine-tuning (SFT). Additionally, we also investigate how the test-time reasoning techniques, such as Best-of-N sampling and beam search, influence reasoning performance, and assess the impact of reasoning-enhanced model distillation, specifically DeepSeek-R1. We will publicly release AnesBench, along with our CPT and SFT training datasets and evaluation code at this https URL. 

---
# LexPam: Legal Procedure Awareness-Guided Mathematical Reasoning 

**Authors**: Kepu Zhang, Guofu Xie, Weijie Yu, Mingyue Xu, Xu Tang, Yaxin Li, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02590)  

**Abstract**: The legal mathematical reasoning ability of LLMs is crucial when applying them to real-world scenarios, as it directly affects the credibility of the LLM. While existing legal LLMs can perform general judicial question answering, their legal mathematical reasoning capabilities have not been trained. Open-domain reasoning models, though able to generate detailed calculation steps, do not follow the reasoning logic required for legal scenarios. Additionally, there is currently a lack of legal mathematical reasoning datasets to help validate and enhance LLMs' reasoning abilities in legal contexts. To address these issues, we propose the first Chinese legal Mathematical Reasoning Dataset, LexNum, which includes three common legal mathematical reasoning scenarios: economic compensation, work injury compensation, and traffic accident compensation. Based on LexNum, we tested the performance of existing legal LLMs and reasoning LLMs, and introduced LexPam, a reinforcement learning algorithm guided by legal procedural awareness to train LLMs, enhancing their mathematical reasoning abilities in legal scenarios. Experiments on tasks in the three legal scenarios show that the performance of existing legal LLMs and reasoning models in legal mathematical reasoning tasks is unsatisfactory. LexPam can enhance the LLM's ability in these tasks. 

---
# CoTAL: Human-in-the-Loop Prompt Engineering, Chain-of-Thought Reasoning, and Active Learning for Generalizable Formative Assessment Scoring 

**Authors**: Clayton Cohn, Nicole Hutchins, Ashwin T S, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02323)  

**Abstract**: Large language models (LLMs) have created new opportunities to assist teachers and support student learning. Methods such as chain-of-thought (CoT) prompting enable LLMs to grade formative assessments in science, providing scores and relevant feedback to students. However, the extent to which these methods generalize across curricula in multiple domains (such as science, computing, and engineering) remains largely untested. In this paper, we introduce Chain-of-Thought Prompting + Active Learning (CoTAL), an LLM-based approach to formative assessment scoring that (1) leverages Evidence-Centered Design (ECD) principles to develop curriculum-aligned formative assessments and rubrics, (2) applies human-in-the-loop prompt engineering to automate response scoring, and (3) incorporates teacher and student feedback to iteratively refine assessment questions, grading rubrics, and LLM prompts for automated grading. Our findings demonstrate that CoTAL improves GPT-4's scoring performance, achieving gains of up to 24.5% over a non-prompt-engineered baseline. Both teachers and students view CoTAL as effective in scoring and explaining student responses, each providing valuable refinements to enhance grading accuracy and explanation quality. 

---
# ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization 

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02725)  

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency. 

---
# The quasi-semantic competence of LLMs: a case study on the part-whole relation 

**Authors**: Mattia Proietti, Alessandro Lenci  

**Link**: [PDF](https://arxiv.org/pdf/2504.02395)  

**Abstract**: Understanding the extent and depth of the semantic competence of \emph{Large Language Models} (LLMs) is at the center of the current scientific agenda in Artificial Intelligence (AI) and Computational Linguistics (CL). We contribute to this endeavor by investigating their knowledge of the \emph{part-whole} relation, a.k.a. \emph{meronymy}, which plays a crucial role in lexical organization, but it is significantly understudied. We used data from ConceptNet relations \citep{speer2016conceptnet} and human-generated semantic feature norms \citep{McRae:2005} to explore the abilities of LLMs to deal with \textit{part-whole} relations. We employed several methods based on three levels of analysis: i.) \textbf{behavioral} testing via prompting, where we directly queried the models on their knowledge of meronymy, ii.) sentence \textbf{probability} scoring, where we tested models' abilities to discriminate correct (real) and incorrect (asymmetric counterfactual) \textit{part-whole} relations, and iii.) \textbf{concept representation} analysis in vector space, where we proved the linear organization of the \textit{part-whole} concept in the embedding and unembedding spaces. These analyses present a complex picture that reveals that the LLMs' knowledge of this relation is only partial. They have just a ``\emph{quasi}-semantic'' competence and still fall short of capturing deep inferential properties. 

---
# LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models 

**Authors**: Weibin Liao, Xin Gao, Tianyu Jia, Rihong Qiu, Yifan Zhu, Yang Lin, Xu Chu, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02327)  

**Abstract**: Natural Language to SQL (NL2SQL) has emerged as a critical task for enabling seamless interaction with databases. Recent advancements in Large Language Models (LLMs) have demonstrated remarkable performance in this domain. However, existing NL2SQL methods predominantly rely on closed-source LLMs leveraging prompt engineering, while open-source models typically require fine-tuning to acquire domain-specific knowledge. Despite these efforts, open-source LLMs struggle with complex NL2SQL tasks due to the indirect expression of user query objectives and the semantic gap between user queries and database schemas. Inspired by the application of reinforcement learning in mathematical problem-solving to encourage step-by-step reasoning in LLMs, we propose LearNAT (Learning NL2SQL with AST-guided Task Decomposition), a novel framework that improves the performance of open-source LLMs on complex NL2SQL tasks through task decomposition and reinforcement learning. LearNAT introduces three key components: (1) a Decomposition Synthesis Procedure that leverages Abstract Syntax Trees (ASTs) to guide efficient search and pruning strategies for task decomposition, (2) Margin-aware Reinforcement Learning, which employs fine-grained step-level optimization via DPO with AST margins, and (3) Adaptive Demonstration Reasoning, a mechanism for dynamically selecting relevant examples to enhance decomposition capabilities. Extensive experiments on two benchmark datasets, Spider and BIRD, demonstrate that LearNAT enables a 7B-parameter open-source LLM to achieve performance comparable to GPT-4, while offering improved efficiency and accessibility. 

---
# Increasing happiness through conversations with artificial intelligence 

**Authors**: Joseph Heffner, Chongyu Qin, Martin Chadwick, Chris Knutsen, Christopher Summerfield, Zeb Kurth-Nelson, Robb B. Rutledge  

**Link**: [PDF](https://arxiv.org/pdf/2504.02091)  

**Abstract**: Chatbots powered by artificial intelligence (AI) have rapidly become a significant part of everyday life, with over a quarter of American adults using them multiple times per week. While these tools offer potential benefits and risks, a fundamental question remains largely unexplored: How do conversations with AI influence subjective well-being? To investigate this, we conducted a study where participants either engaged in conversations with an AI chatbot (N = 334) or wrote journal entires (N = 193) on the same randomly assigned topics and reported their momentary happiness afterward. We found that happiness after AI chatbot conversations was higher than after journaling, particularly when discussing negative topics such as depression or guilt. Leveraging large language models for sentiment analysis, we found that the AI chatbot mirrored participants' sentiment while maintaining a consistent positivity bias. When discussing negative topics, participants gradually aligned their sentiment with the AI's positivity, leading to an overall increase in happiness. We hypothesized that the history of participants' sentiment prediction errors, the difference between expected and actual emotional tone when responding to the AI chatbot, might explain this happiness effect. Using computational modeling, we find the history of these sentiment prediction errors over the course of a conversation predicts greater post-conversation happiness, demonstrating a central role of emotional expectations during dialogue. Our findings underscore the effect that AI interactions can have on human well-being. 

---
# Measurement of LLM's Philosophies of Human Nature 

**Authors**: Minheng Ni, Ennan Wu, Zidong Gong, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Lijuan Wang, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02304)  

**Abstract**: The widespread application of artificial intelligence (AI) in various tasks, along with frequent reports of conflicts or violations involving AI, has sparked societal concerns about interactions with AI systems. Based on Wrightsman's Philosophies of Human Nature Scale (PHNS), a scale empirically validated over decades to effectively assess individuals' attitudes toward human nature, we design the standardized psychological scale specifically targeting large language models (LLM), named the Machine-based Philosophies of Human Nature Scale (M-PHNS). By evaluating LLMs' attitudes toward human nature across six dimensions, we reveal that current LLMs exhibit a systemic lack of trust in humans, and there is a significant negative correlation between the model's intelligence level and its trust in humans. Furthermore, we propose a mental loop learning framework, which enables LLM to continuously optimize its value system during virtual interactions by constructing moral scenarios, thereby improving its attitude toward human nature. Experiments demonstrate that mental loop learning significantly enhances their trust in humans compared to persona or instruction prompts. This finding highlights the potential of human-based psychological assessments for LLM, which can not only diagnose cognitive biases but also provide a potential solution for ethical learning in artificial intelligence. We release the M-PHNS evaluation code and data at this https URL. 

---
# ContrastScore: Towards Higher Quality, Less Biased, More Efficient Evaluation Metrics with Contrastive Evaluation 

**Authors**: Xiao Wang, Daniil Larionov, Siwei Wu, Yiqi Liu, Steffen Eger, Nafise Sadat Moosavi, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.02106)  

**Abstract**: Evaluating the quality of generated text automatically remains a significant challenge. Conventional reference-based metrics have been shown to exhibit relatively weak correlation with human evaluations. Recent research advocates the use of large language models (LLMs) as source-based metrics for natural language generation (NLG) assessment. While promising, LLM-based metrics, particularly those using smaller models, still fall short in aligning with human judgments. In this work, we introduce ContrastScore, a contrastive evaluation metric designed to enable higher-quality, less biased, and more efficient assessment of generated text. We evaluate ContrastScore on two NLG tasks: machine translation and summarization. Experimental results show that ContrastScore consistently achieves stronger correlation with human judgments than both single-model and ensemble-based baselines. Notably, ContrastScore based on Qwen 3B and 0.5B even outperforms Qwen 7B, despite having only half as many parameters, demonstrating its efficiency. Furthermore, it effectively mitigates common evaluation biases such as length and likelihood preferences, resulting in more robust automatic evaluation. 

---
# TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining 

**Authors**: Jeffrey Li, Mohammadreza Armandpour, Iman Mirzadeh, Sachin Mehta, Vaishaal Shankar, Raviteja Vemulapalli, Samy Bengio, Oncel Tuzel, Mehrdad Farajtabar, Hadi Pouransari, Fartash Faghri  

**Link**: [PDF](https://arxiv.org/pdf/2504.02107)  

**Abstract**: Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) - orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains. 

---
# Urban Computing in the Era of Large Language Models 

**Authors**: Zhonghang Li, Lianghao Xia, Xubin Ren, Jiabin Tang, Tianyi Chen, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02009)  

**Abstract**: Urban computing has emerged as a multidisciplinary field that harnesses data-driven technologies to address challenges and improve urban living. Traditional approaches, while beneficial, often face challenges with generalization, scalability, and contextual understanding. The advent of Large Language Models (LLMs) offers transformative potential in this domain. This survey explores the intersection of LLMs and urban computing, emphasizing the impact of LLMs in processing and analyzing urban data, enhancing decision-making, and fostering citizen engagement. We provide a concise overview of the evolution and core technologies of LLMs. Additionally, we survey their applications across key urban domains, such as transportation, public safety, and environmental monitoring, summarizing essential tasks and prior works in various urban contexts, while highlighting LLMs' functional roles and implementation patterns. Building on this, we propose potential LLM-based solutions to address unresolved challenges. To facilitate in-depth research, we compile a list of available datasets and tools applicable to diverse urban scenarios. Finally, we discuss the limitations of current approaches and outline future directions for advancing LLMs in urban computing. 

---
# LL4G: Self-Supervised Dynamic Optimization for Graph-Based Personality Detection 

**Authors**: Lingzhi Shen, Yunfei Long, Xiaohao Cai, Guanming Chen, Yuhan Wang, Imran Razzak, Shoaib Jameel  

**Link**: [PDF](https://arxiv.org/pdf/2504.02146)  

**Abstract**: Graph-based personality detection constructs graph structures from textual data, particularly social media posts. Current methods often struggle with sparse or noisy data and rely on static graphs, limiting their ability to capture dynamic changes between nodes and relationships. This paper introduces LL4G, a self-supervised framework leveraging large language models (LLMs) to optimize graph neural networks (GNNs). LLMs extract rich semantic features to generate node representations and to infer explicit and implicit relationships. The graph structure adaptively adds nodes and edges based on input data, continuously optimizing itself. The GNN then uses these optimized representations for joint training on node reconstruction, edge prediction, and contrastive learning tasks. This integration of semantic and structural information generates robust personality profiles. Experimental results on Kaggle and Pandora datasets show LL4G outperforms state-of-the-art models. 

---
