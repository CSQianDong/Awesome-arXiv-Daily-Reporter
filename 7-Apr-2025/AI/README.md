# Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning 

**Authors**: Xinyi Wang, Shawn Tan, Mingyu Jin, William Yang Wang, Rameswar Panda, Yikang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03635)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks requiring complex reasoning. However, the effects of scaling on their reasoning abilities remain insufficiently understood. In this paper, we introduce a synthetic multihop reasoning environment designed to closely replicate the structure and distribution of real-world large-scale knowledge graphs. Our reasoning task involves completing missing edges in the graph, which requires advanced multi-hop reasoning and mimics real-world reasoning scenarios. To evaluate this, we pretrain language models (LMs) from scratch solely on triples from the incomplete graph and assess their ability to infer the missing edges. Interestingly, we observe that overparameterization can impair reasoning performance due to excessive memorization. We investigate different factors that affect this U-shaped loss curve, including graph structure, model size, and training steps. To predict the optimal model size for a specific knowledge graph, we find an empirical scaling that linearly maps the knowledge graph search entropy to the optimal model size. This work provides new insights into the relationship between scaling and reasoning in LLMs, shedding light on possible ways to optimize their performance for reasoning tasks. 

---
# Towards deployment-centric multimodal AI beyond vision and language 

**Authors**: Xianyuan Liu, Jiayang Zhang, Shuo Zhou, Thijs L. van der Plas, Avish Vijayaraghavan, Anastasiia Grishina, Mengdie Zhuang, Daniel Schofield, Christopher Tomlinson, Yuhan Wang, Ruizhe Li, Louisa van Zeeland, Sina Tabakhi, Cyndie Demeocq, Xiang Li, Arunav Das, Orlando Timmerman, Thomas Baldwin-McDonald, Jinge Wu, Peizhen Bai, Zahraa Al Sahili, Omnia Alwazzan, Thao N. Do, Mohammod N.I. Suvon, Angeline Wang, Lucia Cipolina-Kun, Luigi A. Moretti, Lucas Farndale, Nitisha Jain, Natalia Efremova, Yan Ge, Marta Varela, Hak-Keung Lam, Oya Celiktutan, Ben R. Evans, Alejandro Coca-Castro, Honghan Wu, Zahraa S. Abdallah, Chen Chen, Valentin Danchev, Nataliya Tkachenko, Lei Lu, Tingting Zhu, Gregory G. Slabaugh, Roger K. Moore, William K. Cheung, Peter H. Charlton, Haiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03603)  

**Abstract**: Multimodal artificial intelligence (AI) integrates diverse types of data via machine learning to improve understanding, prediction, and decision-making across disciplines such as healthcare, science, and engineering. However, most multimodal AI advances focus on models for vision and language data, while their deployability remains a key challenge. We advocate a deployment-centric workflow that incorporates deployment constraints early to reduce the likelihood of undeployable solutions, complementing data-centric and model-centric approaches. We also emphasise deeper integration across multiple levels of multimodality and multidisciplinary collaboration to significantly broaden the research scope beyond vision and language. To facilitate this approach, we identify common multimodal-AI-specific challenges shared across disciplines and examine three real-world use cases: pandemic response, self-driving car design, and climate change adaptation, drawing expertise from healthcare, social science, engineering, science, sustainability, and finance. By fostering multidisciplinary dialogue and open research practices, our community can accelerate deployment-centric development for broad societal impact. 

---
# Talk2X -- An Open-Source Toolkit Facilitating Deployment of LLM-Powered Chatbots on the Web 

**Authors**: Lars Krupp, Daniel Geißler, Peter Hevesi, Marco Hirsch, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2504.03343)  

**Abstract**: Integrated into websites, LLM-powered chatbots offer alternative means of navigation and information retrieval, leading to a shift in how users access information on the web. Yet, predominantly closed-sourced solutions limit proliferation among web hosts and suffer from a lack of transparency with regard to implementation details and energy efficiency. In this work, we propose our openly available agent Talk2X leveraging an adapted retrieval-augmented generation approach (RAG) combined with an automatically generated vector database, benefiting energy efficiency. Talk2X's architecture is generalizable to arbitrary websites offering developers a ready to use tool for integration. Using a mixed-methods approach, we evaluated Talk2X's usability by tasking users to acquire specific assets from an open science repository. Talk2X significantly improved task completion time, correctness, and user experience supporting users in quickly pinpointing specific information as compared to standard user-website interaction. Our findings contribute technical advancements to an ongoing paradigm shift of how we access information on the web. 

---
# Monte Carlo Graph Coloring 

**Authors**: Tristan Cazenave, Benjamin Negrevergne, Florian Sikora  

**Link**: [PDF](https://arxiv.org/pdf/2504.03277)  

**Abstract**: Graph Coloring is probably one of the most studied and famous problem in graph algorithms. Exact methods fail to solve instances with more than few hundred vertices, therefore, a large number of heuristics have been proposed. Nested Monte Carlo Search (NMCS) and Nested Rollout Policy Adaptation (NRPA) are Monte Carlo search algorithms for single player games. Surprisingly, few work has been dedicated to evaluating Monte Carlo search algorithms to combinatorial graph problems. In this paper we expose how to efficiently apply Monte Carlo search to Graph Coloring and compare this approach to existing ones. 

---
# Seeing is Believing: Belief-Space Planning with Foundation Models as Uncertainty Estimators 

**Authors**: Linfeng Zhao, Willie McClinton, Aidan Curtis, Nishanth Kumar, Tom Silver, Leslie Pack Kaelbling, Lawson L.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03245)  

**Abstract**: Generalizable robotic mobile manipulation in open-world environments poses significant challenges due to long horizons, complex goals, and partial observability. A promising approach to address these challenges involves planning with a library of parameterized skills, where a task planner sequences these skills to achieve goals specified in structured languages, such as logical expressions over symbolic facts. While vision-language models (VLMs) can be used to ground these expressions, they often assume full observability, leading to suboptimal behavior when the agent lacks sufficient information to evaluate facts with certainty. This paper introduces a novel framework that leverages VLMs as a perception module to estimate uncertainty and facilitate symbolic grounding. Our approach constructs a symbolic belief representation and uses a belief-space planner to generate uncertainty-aware plans that incorporate strategic information gathering. This enables the agent to effectively reason about partial observability and property uncertainty. We demonstrate our system on a range of challenging real-world tasks that require reasoning in partially observable environments. Simulated evaluations show that our approach outperforms both vanilla VLM-based end-to-end planning or VLM-based state estimation baselines by planning for and executing strategic information gathering. This work highlights the potential of VLMs to construct belief-space symbolic scene representations, enabling downstream tasks such as uncertainty-aware planning. 

---
# DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments 

**Authors**: Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03160)  

**Abstract**: Large Language Models (LLMs) equipped with web search capabilities have demonstrated impressive potential for deep research tasks. However, current approaches predominantly rely on either manually engineered prompts (prompt engineering-based) with brittle performance or reinforcement learning within controlled Retrieval-Augmented Generation (RAG) environments (RAG-based) that fail to capture the complexities of real-world interaction. In this paper, we introduce DeepResearcher, the first comprehensive framework for end-to-end training of LLM-based deep research agents through scaling reinforcement learning (RL) in real-world environments with authentic web search interactions. Unlike RAG-based approaches that assume all necessary information exists within a fixed corpus, our method trains agents to navigate the noisy, unstructured, and dynamic nature of the open web. We implement a specialized multi-agent architecture where browsing agents extract relevant information from various webpage structures and overcoming significant technical challenges. Extensive experiments on open-domain research tasks demonstrate that DeepResearcher achieves substantial improvements of up to 28.9 points over prompt engineering-based baselines and up to 7.2 points over RAG-based RL agents. Our qualitative analysis reveals emergent cognitive behaviors from end-to-end RL training, including the ability to formulate plans, cross-validate information from multiple sources, engage in self-reflection to redirect research, and maintain honesty when unable to find definitive answers. Our results highlight that end-to-end training in real-world web environments is not merely an implementation detail but a fundamental requirement for developing robust research capabilities aligned with real-world applications. We release DeepResearcher at this https URL. 

---
# LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph 

**Authors**: Tu Ao, Yanhua Yu, Yuling Wang, Yang Deng, Zirui Guo, Liang Pang, Pinghui Wang, Tat-Seng Chua, Xiao Zhang, Zhen Cai  

**Link**: [PDF](https://arxiv.org/pdf/2504.03137)  

**Abstract**: Large Language Models (LLMs) have impressive capabilities in text understanding and zero-shot reasoning. However, delays in knowledge updates may cause them to reason incorrectly or produce harmful results. Knowledge Graphs (KGs) provide rich and reliable contextual information for the reasoning process of LLMs by structurally organizing and connecting a wide range of entities and relations. Existing KG-based LLM reasoning methods only inject KGs' knowledge into prompts in a textual form, ignoring its structural information. Moreover, they mostly rely on close-source models or open-source models with large parameters, which poses challenges to high resource consumption. To address this, we propose a novel Lightweight and efficient Prompt learning-ReasOning Framework for KGQA (LightPROF), which leverages the full potential of LLMs to tackle complex reasoning tasks in a parameter-efficient manner. Specifically, LightPROF follows a "Retrieve-Embed-Reason process", first accurately, and stably retrieving the corresponding reasoning graph from the KG through retrieval module. Next, through a Transformer-based Knowledge Adapter, it finely extracts and integrates factual and structural information from the KG, then maps this information to the LLM's token embedding space, creating an LLM-friendly prompt to be used by the LLM for the final reasoning. Additionally, LightPROF only requires training Knowledge Adapter and can be compatible with any open-source LLM. Extensive experiments on two public KGQA benchmarks demonstrate that LightPROF achieves superior performance with small-scale LLMs. Furthermore, LightPROF shows significant advantages in terms of input token count and reasoning time. 

---
# Language Models Guidance with Multi-Aspect-Cueing: A Case Study for Competitor Analysis 

**Authors**: Amir Hadifar, Christopher Ochs, Arjan Van Ewijk  

**Link**: [PDF](https://arxiv.org/pdf/2504.02984)  

**Abstract**: Competitor analysis is essential in modern business due to the influence of industry rivals on strategic planning. It involves assessing multiple aspects and balancing trade-offs to make informed decisions. Recent Large Language Models (LLMs) have demonstrated impressive capabilities to reason about such trade-offs but grapple with inherent limitations such as a lack of knowledge about contemporary or future realities and an incomplete understanding of a market's competitive landscape. In this paper, we address this gap by incorporating business aspects into LLMs to enhance their understanding of a competitive market. Through quantitative and qualitative experiments, we illustrate how integrating such aspects consistently improves model performance, thereby enhancing analytical efficacy in competitor analysis. 

---
# Bonsai: Interpretable Tree-Adaptive Grounded Reasoning 

**Authors**: Kate Sanders, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03640)  

**Abstract**: To develop general-purpose collaborative agents, humans need reliable AI systems that can (1) adapt to new domains and (2) transparently reason with uncertainty to allow for verification and correction. Black-box models demonstrate powerful data processing abilities but do not satisfy these criteria due to their opaqueness, domain specificity, and lack of uncertainty awareness. We introduce Bonsai, a compositional and probabilistic reasoning system that generates adaptable inference trees by retrieving relevant grounding evidence and using it to compute likelihoods of sub-claims derived from broader natural language inferences. Bonsai's reasoning power is tunable at test-time via evidence scaling and it demonstrates reliable handling of varied domains including transcripts, photographs, videos, audio, and databases. Question-answering and human alignment experiments demonstrate that Bonsai matches the performance of domain-specific black-box methods while generating interpretable, grounded, and uncertainty-aware reasoning traces. 

---
# Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models 

**Authors**: NVIDIA, Aaron Blakeman, Aarti Basant, Abhinav Khattar, Adithya Renduchintala, Akhiad Bercovich, Aleksander Ficek, Alexis Bjorlin, Ali Taghibakhshi, Amala Sanjay Deshmukh, Ameya Sunil Mahabaleshwarkar, Andrew Tao, Anna Shors, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Bobby Chen, Boris Ginsburg, Boxin Wang, Brandon Norick, Brian Butterfield, Bryan Catanzaro, Carlo del Mundo, Chengyu Dong, Christine Harvey, Christopher Parisien, Dan Su, Daniel Korzekwa, Danny Yin, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Denys Fridman, Dima Rekesh, Ding Ma, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Dusan Stosic, Eileen Long, Elad Segal, Ellie Evans, Eric Chung, Erick Galinkin, Evelina Bakhturina, Ewa Dobrowolska, Fei Jia, Fuxiao Liu, Gargi Prasad, Gerald Shen, Guilin Liu, Guo Chen, Haifeng Qian, Helen Ngo, Hongbin Liu, Hui Li, Igor Gitman, Ilia Karmanov, Ivan Moshkov, Izik Golan, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jarno Seppanen, Jason Lu, Jason Sewall, Jiaqi Zeng, Jiaxuan You, Jimmy Zhang, Jing Zhang, Jining Huang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jon Barker, Jonathan Cohen, Joseph Jennings, Jupinder Parmar, Karan Sapra, Kari Briski, Kateryna Chumachenko, Katherine Luna, Keshav Santhanam, Kezhi Kong, Kirthi Sivamani, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Lawrence McAfee, Leon Derczynski, Lindsey Pavao, Luis Vega, Lukas Voegtle, Maciej Bala, Maer Rodrigues de Melo, Makesh Narsimhan Sreedhar, Marcin Chochowski, Markus Kliegl  

**Link**: [PDF](https://arxiv.org/pdf/2504.03624)  

**Abstract**: As inference-time scaling becomes critical for enhanced reasoning capabilities, it is increasingly becoming important to build models that are efficient to infer. We introduce Nemotron-H, a family of 8B and 56B/47B hybrid Mamba-Transformer models designed to reduce inference cost for a given accuracy level. To achieve this goal, we replace the majority of self-attention layers in the common Transformer model architecture with Mamba layers that perform constant computation and require constant memory per generated token. We show that Nemotron-H models offer either better or on-par accuracy compared to other similarly-sized state-of-the-art open-sourced Transformer models (e.g., Qwen-2.5-7B/72B and Llama-3.1-8B/70B), while being up to 3$\times$ faster at inference. To further increase inference speed and reduce the memory required at inference time, we created Nemotron-H-47B-Base from the 56B model using a new compression via pruning and distillation technique called MiniPuzzle. Nemotron-H-47B-Base achieves similar accuracy to the 56B model, but is 20% faster to infer. In addition, we introduce an FP8-based training recipe and show that it can achieve on par results with BF16-based training. This recipe is used to train the 56B model. All Nemotron-H models will be released, with support in Hugging Face, NeMo, and Megatron-LM. 

---
# Align to Structure: Aligning Large Language Models with Structural Information 

**Authors**: Zae Myung Kim, Anand Ramachandran, Farideh Tavazoee, Joo-Kyung Kim, Oleg Rokhlenko, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03622)  

**Abstract**: Generating long, coherent text remains a challenge for large language models (LLMs), as they lack hierarchical planning and structured organization in discourse generation. We introduce Structural Alignment, a novel method that aligns LLMs with human-like discourse structures to enhance long-form text generation. By integrating linguistically grounded discourse frameworks into reinforcement learning, our approach guides models to produce coherent and well-organized outputs. We employ a dense reward scheme within a Proximal Policy Optimization framework, assigning fine-grained, token-level rewards based on the discourse distinctiveness relative to human writing. Two complementary reward models are evaluated: the first improves readability by scoring surface-level textual features to provide explicit structuring, while the second reinforces deeper coherence and rhetorical sophistication by analyzing global discourse patterns through hierarchical discourse motifs, outperforming both standard and RLHF-enhanced models in tasks such as essay generation and long-document summarization. All training data and code will be publicly shared at this https URL. 

---
# Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task 

**Authors**: Leonardo Ranaldi, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2504.03616)  

**Abstract**: Retrieval-augmented generation (RAG) has become a cornerstone of contemporary NLP, enhancing large language models (LLMs) by allowing them to access richer factual contexts through in-context retrieval. While effective in monolingual settings, especially in English, its use in multilingual tasks remains unexplored. This paper investigates the effectiveness of RAG across multiple languages by proposing novel approaches for multilingual open-domain question-answering. We evaluate the performance of various multilingual RAG strategies, including question-translation (tRAG), which translates questions into English before retrieval, and Multilingual RAG (MultiRAG), where retrieval occurs directly across multiple languages. Our findings reveal that tRAG, while useful, suffers from limited coverage. In contrast, MultiRAG improves efficiency by enabling multilingual retrieval but introduces inconsistencies due to cross-lingual variations in the retrieved content. To address these issues, we propose Crosslingual RAG (CrossRAG), a method that translates retrieved documents into a common language (e.g., English) before generating the response. Our experiments show that CrossRAG significantly enhances performance on knowledge-intensive tasks, benefiting both high-resource and low-resource languages. 

---
# Autonomous and Self-Adapting System for Synthetic Media Detection and Attribution 

**Authors**: Aref Azizpour, Tai D. Nguyen, Matthew C. Stamm  

**Link**: [PDF](https://arxiv.org/pdf/2504.03615)  

**Abstract**: Rapid advances in generative AI have enabled the creation of highly realistic synthetic images, which, while beneficial in many domains, also pose serious risks in terms of disinformation, fraud, and other malicious applications. Current synthetic image identification systems are typically static, relying on feature representations learned from known generators; as new generative models emerge, these systems suffer from severe performance degradation. In this paper, we introduce the concept of an autonomous self-adaptive synthetic media identification system -- one that not only detects synthetic images and attributes them to known sources but also autonomously identifies and incorporates novel generators without human intervention. Our approach leverages an open-set identification strategy with an evolvable embedding space that distinguishes between known and unknown sources. By employing an unsupervised clustering method to aggregate unknown samples into high-confidence clusters and continuously refining its decision boundaries, our system maintains robust detection and attribution performance even as the generative landscape evolves. Extensive experiments demonstrate that our method significantly outperforms existing approaches, marking a crucial step toward universal, adaptable forensic systems in the era of rapidly advancing generative models. 

---
# APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay 

**Authors**: Akshara Prabhakar, Zuxin Liu, Weiran Yao, Jianguo Zhang, Ming Zhu, Shiyu Wang, Zhiwei Liu, Tulika Awalgaonkar, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03601)  

**Abstract**: Training effective AI agents for multi-turn interactions requires high-quality data that captures realistic human-agent dynamics, yet such data is scarce and expensive to collect manually. We introduce APIGen-MT, a two-phase framework that generates verifiable and diverse multi-turn agent data. In the first phase, our agentic pipeline produces detailed task blueprints with ground-truth actions, leveraging a committee of LLM reviewers and iterative feedback loops. These blueprints are then transformed into complete interaction trajectories through simulated human-agent interplay. We train a family of models -- the xLAM-2-fc-r series with sizes ranging from 1B to 70B parameters. Our models outperform frontier models such as GPT-4o and Claude 3.5 on $\tau$-bench and BFCL benchmarks, with the smaller models surpassing their larger counterparts, particularly in multi-turn settings, while maintaining superior consistency across multiple trials. Comprehensive experiments demonstrate that our verified blueprint-to-details approach yields high-quality training data, enabling the development of more reliable, efficient, and capable agents. We open-source both the synthetic data collected and the trained xLAM-2-fc-r models to advance research in AI agents. Models are available on HuggingFace at this https URL and project website is this https URL 

---
# MedSAM2: Segment Anything in 3D Medical Images and Videos 

**Authors**: Jun Ma, Zongxin Yang, Sumin Kim, Bihui Chen, Mohammed Baharoon, Adibvafa Fallahpour, Reza Asakereh, Hongwei Lyu, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03600)  

**Abstract**: Medical image and video segmentation is a critical task for precision medicine, which has witnessed considerable progress in developing task or modality-specific and generalist models for 2D images. However, there have been limited studies on building general-purpose models for 3D images and videos with comprehensive user studies. Here, we present MedSAM2, a promptable segmentation foundation model for 3D image and video segmentation. The model is developed by fine-tuning the Segment Anything Model 2 on a large medical dataset with over 455,000 3D image-mask pairs and 76,000 frames, outperforming previous models across a wide range of organs, lesions, and imaging modalities. Furthermore, we implement a human-in-the-loop pipeline to facilitate the creation of large-scale datasets resulting in, to the best of our knowledge, the most extensive user study to date, involving the annotation of 5,000 CT lesions, 3,984 liver MRI lesions, and 251,550 echocardiogram video frames, demonstrating that MedSAM2 can reduce manual costs by more than 85%. MedSAM2 is also integrated into widely used platforms with user-friendly interfaces for local and cloud deployment, making it a practical tool for supporting efficient, scalable, and high-quality segmentation in both research and healthcare environments. 

---
# EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline 

**Authors**: Peter Baile Chen, Tomer Wolfson, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2504.03598)  

**Abstract**: Existing information retrieval systems excel in cases where the language of target documents closely matches that of the user query. However, real-world retrieval systems are often required to implicitly reason whether a document is relevant. For example, when retrieving technical texts or tables, their relevance to the user query may be implied through a particular jargon or structure, rather than explicitly expressed in their content. Large language models (LLMs) hold great potential in identifying such implied relevance by leveraging their reasoning skills. Nevertheless, current LLM-augmented retrieval is hindered by high latency and computation cost, as the LLM typically computes the query-document relevance online, for every query anew. To tackle this issue we introduce EnrichIndex, a retrieval approach which instead uses the LLM offline to build semantically-enriched retrieval indices, by performing a single pass over all documents in the retrieval corpus once during ingestion time. Furthermore, the semantically-enriched indices can complement existing online retrieval approaches, boosting the performance of LLM re-rankers. We evaluated EnrichIndex on five retrieval tasks, involving passages and tables, and found that it outperforms strong online LLM-based retrieval systems, with an average improvement of 11.7 points in recall @ 10 and 10.6 points in NDCG @ 10 compared to strong baselines. In terms of online calls to the LLM, it processes 293.3 times fewer tokens which greatly reduces the online latency and cost. Overall, EnrichIndex is an effective way to build better retrieval indices offline by leveraging the strong reasoning skills of LLMs. 

---
# Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin for Real-World Robot Policy Evaluation 

**Authors**: Jad Abou-Chakra, Lingfeng Sun, Krishan Rana, Brandon May, Karl Schmeckpeper, Maria Vittoria Minniti, Laura Herlant  

**Link**: [PDF](https://arxiv.org/pdf/2504.03597)  

**Abstract**: Recent advancements in behavior cloning have enabled robots to perform complex manipulation tasks. However, accurately assessing training performance remains challenging, particularly for real-world applications, as behavior cloning losses often correlate poorly with actual task success. Consequently, researchers resort to success rate metrics derived from costly and time-consuming real-world evaluations, making the identification of optimal policies and detection of overfitting or underfitting impractical. To address these issues, we propose real-is-sim, a novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment. By continuously aligning the simulated world with the physical world, demonstrations can be collected in the real world with states extracted from the simulator. The simulator enables flexible state representations by rendering image inputs from any viewpoint or extracting low-level state information from objects embodied within the scene. During training, policies can be directly evaluated within the simulator in an offline and highly parallelizable manner. Finally, during deployment, policies are run within the simulator where the real robot directly tracks the simulated robot's joints, effectively decoupling policy execution from real hardware and mitigating traditional domain-transfer challenges. We validate real-is-sim on the PushT manipulation task, demonstrating strong correlation between success rates obtained in the simulator and real-world evaluations. Videos of our system can be found at this https URL. 

---
# SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement 

**Authors**: Runnan Fang, Xiaobin Wang, Yuan Liang, Shuofei Qiao, Jialong Wu, Zekun Xi, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03561)  

**Abstract**: In the interaction between agents and their environments, agents expand their capabilities by planning and executing actions. However, LLM-based agents face substantial challenges when deployed in novel environments or required to navigate unconventional action spaces. To empower agents to autonomously explore environments, optimize workflows, and enhance their understanding of actions, we propose SynWorld, a framework that allows agents to synthesize possible scenarios with multi-step action invocation within the action space and perform Monte Carlo Tree Search (MCTS) exploration to effectively refine their action knowledge in the current environment. Our experiments demonstrate that SynWorld is an effective and general approach to learning action knowledge in new environments. Code is available at this https URL. 

---
# Agentic Knowledgeable Self-awareness 

**Authors**: Shuofei Qiao, Zhisong Qiu, Baochang Ren, Xiaobin Wang, Xiangyuan Ru, Ningyu Zhang, Xiang Chen, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03553)  

**Abstract**: Large Language Models (LLMs) have achieved considerable performance across various agentic planning tasks. However, traditional agent planning approaches adopt a "flood irrigation" methodology that indiscriminately injects gold trajectories, external feedback, and domain knowledge into agent models. This practice overlooks the fundamental human cognitive principle of situational self-awareness during decision-making-the ability to dynamically assess situational demands and strategically employ resources during decision-making. We propose agentic knowledgeable self-awareness to address this gap, a novel paradigm enabling LLM-based agents to autonomously regulate knowledge utilization. Specifically, we propose KnowSelf, a data-centric approach that applies agents with knowledgeable self-awareness like humans. Concretely, we devise a heuristic situation judgement criterion to mark special tokens on the agent's self-explored trajectories for collecting training data. Through a two-stage training process, the agent model can switch between different situations by generating specific special tokens, achieving optimal planning effects with minimal costs. Our experiments demonstrate that KnowSelf can outperform various strong baselines on different tasks and models with minimal use of external knowledge. Code is available at this https URL. 

---
# MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation 

**Authors**: Khai Le-Duc, Tuyen Tran, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03546)  

**Abstract**: Multilingual speech translation (ST) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, Traditional Chinese and Simplified Chinese, together with the models. With 290,000 samples, our dataset is the largest medical machine translation (MT) dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most extensive analysis study in ST research to date, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence (seq2seq) comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: this https URL. 

---
# Dense Neural Network Based Arrhythmia Classification on Low-cost and Low-compute Micro-controller 

**Authors**: Md Abu Obaida Zishan, H M Shihab, Sabik Sadman Islam, Maliha Alam Riya, Gazi Mashrur Rahman, Jannatun Noor  

**Link**: [PDF](https://arxiv.org/pdf/2504.03531)  

**Abstract**: The electrocardiogram (ECG) monitoring device is an expensive albeit essential device for the treatment and diagnosis of cardiovascular diseases (CVD). The cost of this device typically ranges from $2000 to $10000. Several studies have implemented ECG monitoring systems in micro-controller units (MCU) to reduce industrial development costs by up to 20 times. However, to match industry-grade systems and display heartbeats effectively, it is essential to develop an efficient algorithm for detecting arrhythmia (irregular heartbeat). Hence in this study, a dense neural network is developed to detect arrhythmia on the Arduino Nano. The Nano consists of the ATMega328 microcontroller with a 16MHz clock, 2KB of SRAM, and 32KB of program memory. Additionally, the AD8232 SparkFun Single-Lead Heart Rate Monitor is used as the ECG sensor. The implemented neural network model consists of two layers (excluding the input) with 10 and four neurons respectively with sigmoid activation function. However, four approaches are explored to choose the appropriate activation functions. The model has a size of 1.267 KB, achieves an F1 score (macro-average) of 78.3\% for classifying four types of arrhythmia, an accuracy rate of 96.38%, and requires 0.001314 MOps of floating-point operations (FLOPs). 

---
# Quantifying Robustness: A Benchmarking Framework for Deep Learning Forecasting in Cyber-Physical Systems 

**Authors**: Alexander Windmann, Henrik Steude, Daniel Boschmann, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.03494)  

**Abstract**: Cyber-Physical Systems (CPS) in domains such as manufacturing and energy distribution generate complex time series data crucial for Prognostics and Health Management (PHM). While Deep Learning (DL) methods have demonstrated strong forecasting capabilities, their adoption in industrial CPS remains limited due insufficient robustness. Existing robustness evaluations primarily focus on formal verification or adversarial perturbations, inadequately representing the complexities encountered in real-world CPS scenarios. To address this, we introduce a practical robustness definition grounded in distributional robustness, explicitly tailored to industrial CPS, and propose a systematic framework for robustness evaluation. Our framework simulates realistic disturbances, such as sensor drift, noise and irregular sampling, enabling thorough robustness analyses of forecasting models on real-world CPS datasets. The robustness definition provides a standardized score to quantify and compare model performance across diverse datasets, assisting in informed model selection and architecture design. Through extensive empirical studies evaluating prominent DL architectures (including recurrent, convolutional, attention-based, modular, and structured state-space models) we demonstrate the applicability and effectiveness of our approach. We publicly release our robustness benchmark to encourage further research and reproducibility. 

---
# BUFF: Bayesian Uncertainty Guided Diffusion Probabilistic Model for Single Image Super-Resolution 

**Authors**: Zihao He, Shengchuan Zhang, Runze Hu, Yunhang Shen, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03490)  

**Abstract**: Super-resolution (SR) techniques are critical for enhancing image quality, particularly in scenarios where high-resolution imagery is essential yet limited by hardware constraints. Existing diffusion models for SR have relied predominantly on Gaussian models for noise generation, which often fall short when dealing with the complex and variable texture inherent in natural scenes. To address these deficiencies, we introduce the Bayesian Uncertainty Guided Diffusion Probabilistic Model (BUFF). BUFF distinguishes itself by incorporating a Bayesian network to generate high-resolution uncertainty masks. These masks guide the diffusion process, allowing for the adjustment of noise intensity in a manner that is both context-aware and adaptive. This novel approach not only enhances the fidelity of super-resolved images to their original high-resolution counterparts but also significantly mitigates artifacts and blurring in areas characterized by complex textures and fine details. The model demonstrates exceptional robustness against complex noise patterns and showcases superior adaptability in handling textures and edges within images. Empirical evidence, supported by visual results, illustrates the model's robustness, especially in challenging scenarios, and its effectiveness in addressing common SR issues such as blurring. Experimental evaluations conducted on the DIV2K dataset reveal that BUFF achieves a notable improvement, with a +0.61 increase compared to baseline in SSIM on BSD100, surpassing traditional diffusion approaches by an average additional +0.20dB PSNR gain. These findings underscore the potential of Bayesian methods in enhancing diffusion processes for SR, paving the way for future advancements in the field. 

---
# Structured Legal Document Generation in India: A Model-Agnostic Wrapper Approach with VidhikDastaavej 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.03486)  

**Abstract**: Automating legal document drafting can significantly enhance efficiency, reduce manual effort, and streamline legal workflows. While prior research has explored tasks such as judgment prediction and case summarization, the structured generation of private legal documents in the Indian legal domain remains largely unaddressed. To bridge this gap, we introduce VidhikDastaavej, a novel, anonymized dataset of private legal documents, and develop NyayaShilp, a fine-tuned legal document generation model specifically adapted to Indian legal texts. We propose a Model-Agnostic Wrapper (MAW), a two-step framework that first generates structured section titles and then iteratively produces content while leveraging retrieval-based mechanisms to ensure coherence and factual accuracy. We benchmark multiple open-source LLMs, including instruction-tuned and domain-adapted versions, alongside proprietary models for comparison. Our findings indicate that while direct fine-tuning on small datasets does not always yield improvements, our structured wrapper significantly enhances coherence, factual adherence, and overall document quality while mitigating hallucinations. To ensure real-world applicability, we developed a Human-in-the-Loop (HITL) Document Generation System, an interactive user interface that enables users to specify document types, refine section details, and generate structured legal drafts. This tool allows legal professionals and researchers to generate, validate, and refine AI-generated legal documents efficiently. Extensive evaluations, including expert assessments, confirm that our framework achieves high reliability in structured legal drafting. This research establishes a scalable and adaptable foundation for AI-assisted legal drafting in India, offering an effective approach to structured legal document generation. 

---
# Physics-informed 4D X-ray image reconstruction from ultra-sparse spatiotemporal data 

**Authors**: Zisheng Yao, Yuhe Zhang, Zhe Hu, Robert Klöfkorn, Tobias Ritschel, Pablo Villanueva-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2504.03469)  

**Abstract**: The unprecedented X-ray flux density provided by modern X-ray sources offers new spatiotemporal possibilities for X-ray imaging of fast dynamic processes. Approaches to exploit such possibilities often result in either i) a limited number of projections or spatial information due to limited scanning speed, as in time-resolved tomography, or ii) a limited number of time points, as in stroboscopic imaging, making the reconstruction problem ill-posed and unlikely to be solved by classical reconstruction approaches. 4D reconstruction from such data requires sample priors, which can be included via deep learning (DL). State-of-the-art 4D reconstruction methods for X-ray imaging combine the power of AI and the physics of X-ray propagation to tackle the challenge of sparse views. However, most approaches do not constrain the physics of the studied process, i.e., a full physical model. Here we present 4D physics-informed optimized neural implicit X-ray imaging (4D-PIONIX), a novel physics-informed 4D X-ray image reconstruction method combining the full physical model and a state-of-the-art DL-based reconstruction method for 4D X-ray imaging from sparse views. We demonstrate and evaluate the potential of our approach by retrieving 4D information from ultra-sparse spatiotemporal acquisitions of simulated binary droplet collisions, a relevant fluid dynamic process. We envision that this work will open new spatiotemporal possibilities for various 4D X-ray imaging modalities, such as time-resolved X-ray tomography and more novel sparse acquisition approaches like X-ray multi-projection imaging, which will pave the way for investigations of various rapid 4D dynamics, such as fluid dynamics and composite testing. 

---
# SpectR: Dynamically Composing LM Experts with Spectral Routing 

**Authors**: William Fleshman, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03454)  

**Abstract**: Training large, general-purpose language models poses significant challenges. The growing availability of specialized expert models, fine-tuned from pretrained models for specific tasks or domains, offers a promising alternative. Leveraging the potential of these existing expert models in real-world applications requires effective methods to select or merge the models best suited for a given task. This paper introduces SPECTR, an approach for dynamically composing expert models at each time step during inference. Notably, our method requires no additional training and enables flexible, token- and layer-wise model combinations. Our experimental results demonstrate that SPECTR improves routing accuracy over alternative training-free methods, increasing task performance across expert domains. 

---
# The AI Cosmologist I: An Agentic System for Automated Data Analysis 

**Authors**: Adam Moss  

**Link**: [PDF](https://arxiv.org/pdf/2504.03424)  

**Abstract**: We present the AI Cosmologist, an agentic system designed to automate cosmological/astronomical data analysis and machine learning research workflows. This implements a complete pipeline from idea generation to experimental evaluation and research dissemination, mimicking the scientific process typically performed by human researchers. The system employs specialized agents for planning, coding, execution, analysis, and synthesis that work together to develop novel approaches. Unlike traditional auto machine-learning systems, the AI Cosmologist generates diverse implementation strategies, writes complete code, handles execution errors, analyzes results, and synthesizes new approaches based on experimental outcomes. We demonstrate the AI Cosmologist capabilities across several machine learning tasks, showing how it can successfully explore solution spaces, iterate based on experimental results, and combine successful elements from different approaches. Our results indicate that agentic systems can automate portions of the research process, potentially accelerating scientific discovery. The code and experimental data used in this paper are available on GitHub at this https URL. Example papers included in the appendix demonstrate the system's capability to autonomously produce complete scientific publications, starting from only the dataset and task description 

---
# Autonomous state-space segmentation for Deep-RL sparse reward scenarios 

**Authors**: Gianluca Maselli, Vieri Giuliano Santucci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03420)  

**Abstract**: Dealing with environments with sparse rewards has always been crucial for systems developed to operate in autonomous open-ended learning settings. Intrinsic Motivations could be an effective way to help Deep Reinforcement Learning algorithms learn in such scenarios. In fact, intrinsic reward signals, such as novelty or curiosity, are generally adopted to improve exploration when extrinsic rewards are delayed or absent. Building on previous works, we tackle the problem of learning policies in the presence of sparse rewards by proposing a two-level architecture that alternates an ''intrinsically driven'' phase of exploration and autonomous sub-goal generation, to a phase of sparse reward, goal-directed policy learning. The idea is to build several small networks, each one specialized on a particular sub-path, and use them as starting points for future exploration without the need to further explore from scratch previously learnt paths. Two versions of the system have been trained and tested in the Gym SuperMarioBros environment without considering any additional extrinsic reward. The results show the validity of our approach and the importance of autonomously segment the environment to generate an efficient path towards the final goal. 

---
# Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning 

**Authors**: Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, Donghyun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2504.03380)  

**Abstract**: Reasoning-Oriented Reinforcement Learning (RORL) enhances the reasoning ability of Large Language Models (LLMs). However, due to the sparsity of rewards in RORL, effective training is highly dependent on the selection of problems of appropriate difficulty. Although curriculum learning attempts to address this by adjusting difficulty, it often relies on static schedules, and even recent online filtering methods lack theoretical grounding and a systematic understanding of their effectiveness. In this work, we theoretically and empirically show that curating the batch with the problems that the training model achieves intermediate accuracy on the fly can maximize the effectiveness of RORL training, namely balanced online difficulty filtering. We first derive that the lower bound of the KL divergence between the initial and the optimal policy can be expressed with the variance of the sampled accuracy. Building on those insights, we show that balanced filtering can maximize the lower bound, leading to better performance. Experimental results across five challenging math reasoning benchmarks show that balanced online filtering yields an additional 10% in AIME and 4% improvements in average over plain GRPO. Moreover, further analysis shows the gains in sample efficiency and training time efficiency, exceeding the maximum reward of plain GRPO within 60% training time and the volume of the training set. 

---
# Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency 

**Authors**: Erik Johannes Husom, Arda Goknil, Merve Astekin, Lwin Khin Shar, Andre Kåsen, Sagar Sen, Benedikt Andreas Mithassel, Ahmet Soylu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03360)  

**Abstract**: Deploying Large Language Models (LLMs) on edge devices presents significant challenges due to computational constraints, memory limitations, inference speed, and energy consumption. Model quantization has emerged as a key technique to enable efficient LLM inference by reducing model size and computational overhead. In this study, we conduct a comprehensive analysis of 28 quantized LLMs from the Ollama library, which applies by default Post-Training Quantization (PTQ) and weight-only quantization techniques, deployed on an edge device (Raspberry Pi 4 with 4GB RAM). We evaluate energy efficiency, inference performance, and output accuracy across multiple quantization levels and task types. Models are benchmarked on five standardized datasets (CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, and HumanEval), and we employ a high-resolution, hardware-based energy measurement tool to capture real-world power consumption. Our findings reveal the trade-offs between energy efficiency, inference speed, and accuracy in different quantization settings, highlighting configurations that optimize LLM deployment for resource-constrained environments. By integrating hardware-level energy profiling with LLM benchmarking, this study provides actionable insights for sustainable AI, bridging a critical gap in existing research on energy-aware LLM deployment. 

---
# Decentralized Collective World Model for Emergent Communication and Coordination 

**Authors**: Kentaro Nomura, Tatsuya Aoki, Tadahiro Taniguchi, Takato Horii  

**Link**: [PDF](https://arxiv.org/pdf/2504.03353)  

**Abstract**: We propose a fully decentralized multi-agent world model that enables both symbol emergence for communication and coordinated behavior through temporal extension of collective predictive coding. Unlike previous research that focuses on either communication or coordination separately, our approach achieves both simultaneously. Our method integrates world models with communication channels, enabling agents to predict environmental dynamics, estimate states from partial observations, and share critical information through bidirectional message exchange with contrastive learning for message alignment. Using a two-agent trajectory drawing task, we demonstrate that our communication-based approach outperforms non-communicative models when agents have divergent perceptual capabilities, achieving the second-best coordination after centralized models. Importantly, our distributed approach with constraints preventing direct access to other agents' internal states facilitates the emergence of more meaningful symbol systems that accurately reflect environmental states. These findings demonstrate the effectiveness of decentralized communication for supporting coordination while developing shared representations of the environment. 

---
# EOOD: Entropy-based Out-of-distribution Detection 

**Authors**: Guide Yang, Chao Hou, Weilong Peng, Xiang Fang, Yongwei Nie, Peican Zhu, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03342)  

**Abstract**: Deep neural networks (DNNs) often exhibit overconfidence when encountering out-of-distribution (OOD) samples, posing significant challenges for deployment. Since DNNs are trained on in-distribution (ID) datasets, the information flow of ID samples through DNNs inevitably differs from that of OOD samples. In this paper, we propose an Entropy-based Out-Of-distribution Detection (EOOD) framework. EOOD first identifies specific block where the information flow differences between ID and OOD samples are more pronounced, using both ID and pseudo-OOD samples. It then calculates the conditional entropy on the selected block as the OOD confidence score. Comprehensive experiments conducted across various ID and OOD settings demonstrate the effectiveness of EOOD in OOD detection and its superiority over state-of-the-art methods. 

---
# Mind the Prompt: Prompting Strategies in Audio Generations for Improving Sound Classification 

**Authors**: Francesca Ronchini, Ho-Hsiang Wu, Wei-Cheng Lin, Fabio Antonacci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03329)  

**Abstract**: This paper investigates the design of effective prompt strategies for generating realistic datasets using Text-To-Audio (TTA) models. We also analyze different techniques for efficiently combining these datasets to enhance their utility in sound classification tasks. By evaluating two sound classification datasets with two TTA models, we apply a range of prompt strategies. Our findings reveal that task-specific prompt strategies significantly outperform basic prompt approaches in data generation. Furthermore, merging datasets generated using different TTA models proves to enhance classification performance more effectively than merely increasing the training dataset size. Overall, our results underscore the advantages of these methods as effective data augmentation techniques using synthetic data. 

---
# Policy Optimization Algorithms in a Unified Framework 

**Authors**: Shuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03328)  

**Abstract**: Policy optimization algorithms are crucial in many fields but challenging to grasp and implement, often due to complex calculations related to Markov decision processes and varying use of discount and average reward setups. This paper presents a unified framework that applies generalized ergodicity theory and perturbation analysis to clarify and enhance the application of these algorithms. Generalized ergodicity theory sheds light on the steady-state behavior of stochastic processes, aiding understanding of both discounted and average rewards. Perturbation analysis provides in-depth insights into the fundamental principles of policy optimization algorithms. We use this framework to identify common implementation errors and demonstrate the correct approaches. Through a case study on Linear Quadratic Regulator problems, we illustrate how slight variations in algorithm design affect implementation outcomes. We aim to make policy optimization algorithms more accessible and reduce their misuse in practice. 

---
# Noise Augmented Fine Tuning for Mitigating Hallucinations in Large Language Models 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.03302)  

**Abstract**: Large language models (LLMs) often produce inaccurate or misleading content-hallucinations. To address this challenge, we introduce Noise-Augmented Fine-Tuning (NoiseFiT), a novel framework that leverages adaptive noise injection based on the signal-to-noise ratio (SNR) to enhance model robustness. In particular, NoiseFiT selectively perturbs layers identified as either high-SNR (more robust) or low-SNR (potentially under-regularized) using a dynamically scaled Gaussian noise. We further propose a hybrid loss that combines standard cross-entropy, soft cross-entropy, and consistency regularization to ensure stable and accurate outputs under noisy training conditions. Our theoretical analysis shows that adaptive noise injection is both unbiased and variance-preserving, providing strong guarantees for convergence in expectation. Empirical results on multiple test and benchmark datasets demonstrate that NoiseFiT significantly reduces hallucination rates, often improving or matching baseline performance in key tasks. These findings highlight the promise of noise-driven strategies for achieving robust, trustworthy language modeling without incurring prohibitive computational overhead. Given the comprehensive and detailed nature of our experiments, we have publicly released the fine-tuning logs, benchmark evaluation artifacts, and source code online at W&B, Hugging Face, and GitHub, respectively, to foster further research, accessibility and reproducibility. 

---
# Stance-Driven Multimodal Controlled Statement Generation: New Dataset and Task 

**Authors**: Bingqian Wang, Quan Fang, Jiachen Sun, Xiaoxiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03295)  

**Abstract**: Formulating statements that support diverse or controversial stances on specific topics is vital for platforms that enable user expression, reshape political discourse, and drive social critique and information dissemination. With the rise of Large Language Models (LLMs), controllable text generation towards specific stances has become a promising research area with applications in shaping public opinion and commercial marketing. However, current datasets often focus solely on pure texts, lacking multimodal content and effective context, particularly in the context of stance detection. In this paper, we formally define and study the new problem of stance-driven controllable content generation for tweets with text and images, where given a multimodal post (text and image/video), a model generates a stance-controlled response. To this end, we create the Multimodal Stance Generation Dataset (StanceGen2024), the first resource explicitly designed for multimodal stance-controllable text generation in political discourse. It includes posts and user comments from the 2024 U.S. presidential election, featuring text, images, videos, and stance annotations to explore how multimodal political content shapes stance expression. Furthermore, we propose a Stance-Driven Multimodal Generation (SDMG) framework that integrates weighted fusion of multimodal features and stance guidance to improve semantic consistency and stance control. We release the dataset and code (this https URL) for public use and further research. 

---
# Towards Effective EU E-Participation: The Development of AskThePublic 

**Authors**: Kilian Sprenkamp, Nils Messerschmidt, Amir Sartipi, Igor Tchappi, Xiaohui Wu, Liudmila Zavolokina, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03287)  

**Abstract**: E-participation platforms can be an important asset for governments in increasing trust and fostering democratic societies. By engaging non-governmental and private institutions, domain experts, and even the general public, policymakers can make informed and inclusive decisions. Drawing on the Media Richness Theory and applying the Design Science Research method, we explore how a chatbot can be designed to improve the effectiveness of the policy-making process of existing citizen involvement platforms. Leveraging the Have Your Say platform, which solicits feedback on European Commission initiatives and regulations, a Large Language Model based chatbot, called AskThePublic is created, providing policymakers, journalists, researchers, and interested citizens with a convenient channel to explore and engage with public input. By conducting 11 semistructured interviews, the results show that the participants value the interactive and structured responses as well as enhanced language capabilities, thus increasing their likelihood of engaging with AskThePublic over the existing platform. An outlook for future iterations is provided and discussed with regard to the perspectives of the different stakeholders. 

---
# JanusDDG: A Thermodynamics-Compliant Model for Sequence-Based Protein Stability via Two-Fronts Multi-Head Attention 

**Authors**: Guido Barducci, Ivan Rossi, Francesco Codicè, Cesare Rollo, Valeria Repetto, Corrado Pancotti, Virginia Iannibelli, Tiziana Sanavia, Piero Fariselli  

**Link**: [PDF](https://arxiv.org/pdf/2504.03278)  

**Abstract**: Understanding how residue variations affect protein stability is crucial for designing functional proteins and deciphering the molecular mechanisms underlying disease-related mutations. Recent advances in protein language models (PLMs) have revolutionized computational protein analysis, enabling, among other things, more accurate predictions of mutational effects. In this work, we introduce JanusDDG, a deep learning framework that leverages PLM-derived embeddings and a bidirectional cross-attention transformer architecture to predict $\Delta \Delta G$ of single and multiple-residue mutations while simultaneously being constrained to respect fundamental thermodynamic properties, such as antisymmetry and transitivity. Unlike conventional self-attention, JanusDDG computes queries (Q) and values (V) as the difference between wild-type and mutant embeddings, while keys (K) alternate between the two. This cross-interleaved attention mechanism enables the model to capture mutation-induced perturbations while preserving essential contextual information. Experimental results show that JanusDDG achieves state-of-the-art performance in predicting $\Delta \Delta G$ from sequence alone, matching or exceeding the accuracy of structure-based methods for both single and multiple mutations. 

---
# Do Large Language Models Solve the Problems of Agent-Based Modeling? A Critical Review of Generative Social Simulations 

**Authors**: Maik Larooij, Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.03274)  

**Abstract**: Recent advancements in AI have reinvigorated Agent-Based Models (ABMs), as the integration of Large Language Models (LLMs) has led to the emergence of ``generative ABMs'' as a novel approach to simulating social systems. While ABMs offer means to bridge micro-level interactions with macro-level patterns, they have long faced criticisms from social scientists, pointing to e.g., lack of realism, computational complexity, and challenges of calibrating and validating against empirical data. This paper reviews the generative ABM literature to assess how this new approach adequately addresses these long-standing criticisms. Our findings show that studies show limited awareness of historical debates. Validation remains poorly addressed, with many studies relying solely on subjective assessments of model `believability', and even the most rigorous validation failing to adequately evidence operational validity. We argue that there are reasons to believe that LLMs will exacerbate rather than resolve the long-standing challenges of ABMs. The black-box nature of LLMs moreover limit their usefulness for disentangling complex emergent causal mechanisms. While generative ABMs are still in a stage of early experimentation, these findings question of whether and how the field can transition to the type of rigorous modeling needed to contribute to social scientific theory. 

---
# Verification of Autonomous Neural Car Control with KeYmaera X 

**Authors**: Enguerrand Prebet, Samuel Teuber, André Platzer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03272)  

**Abstract**: This article presents a formal model and formal safety proofs for the ABZ'25 case study in differential dynamic logic (dL). The case study considers an autonomous car driving on a highway avoiding collisions with neighbouring cars. Using KeYmaera X's dL implementation, we prove absence of collision on an infinite time horizon which ensures that safety is preserved independently of trip length. The safety guarantees hold for time-varying reaction time and brake force. Our dL model considers the single lane scenario with cars ahead or behind. We demonstrate that dL with its tools is a rigorous foundation for runtime monitoring, shielding, and neural network verification. Doing so sheds light on inconsistencies between the provided specification and simulation environment highway-env of the ABZ'25 study. We attempt to fix these inconsistencies and uncover numerous counterexamples which also indicate issues in the provided reinforcement learning environment. 

---
# An Extended Symbolic-Arithmetic Model for Teaching Double-Black Removal with Rotation in Red-Black Trees 

**Authors**: Kennedy E. Ehimwenma, Hongyu Zhou, Junfeng Wang, Ze Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.03259)  

**Abstract**: Double-black (DB) nodes have no place in red-black (RB) trees. So when DB nodes are formed, they are immediately removed. The removal of DB nodes that cause rotation and recoloring of other connected nodes poses greater challenges in the teaching and learning of RB trees. To ease this difficulty, this paper extends our previous work on the symbolic arithmetic algebraic (SA) method for removing DB nodes. The SA operations that are given as, Red + Black = Black; Black - Black = Red; Black + Black = DB; and DB - Black = Black removes DB nodes and rebalances black heights in RB trees. By extension, this paper projects three SA mathematical equations, namely, general symbolic arithmetic rule; partial symbolic arithmetic rule1; and partial symbolic arithmetic rule2. The removal of a DB node ultimately affects black heights in RB trees. To balance black heights using the SA equations, all the RB tree cases, namely, LR, RL, LL, and RR, were considered in this work; and the position of the nodes connected directly or indirectly to the DB node was also tested. In this study, to balance a RB tree, the issues considered w.r.t. the different cases of the RB tree were i) whether a DB node has an inner, outer, or both inner and outer black nephews; or ii) whether a DB node has an inner, outer or both inner and outer red nephews. The nephews r and x in this work are the children of the sibling s to a DB, and further up the tree, the parent p of a DB is their grandparent g. Thus, r and x have indirect relationships to a DB at the point of formation of the DB node. The novelty of the SA equations is in their effectiveness in the removal of DB that involves rotation of nodes as well as the recoloring of nodes along any simple path so as to balance black heights in a tree. 

---
# Rotation Invariance in Floor Plan Digitization using Zernike Moments 

**Authors**: Marius Graumann, Jan Marius Stürmer, Tobias Koch  

**Link**: [PDF](https://arxiv.org/pdf/2504.03241)  

**Abstract**: Nowadays, a lot of old floor plans exist in printed form or are stored as scanned raster images. Slight rotations or shifts may occur during scanning. Bringing floor plans of this form into a machine readable form to enable further use, still poses a problem. Therefore, we propose an end-to-end pipeline that pre-processes the image and leverages a novel approach to create a region adjacency graph (RAG) from the pre-processed image and predict its nodes. By incorporating normalization steps into the RAG feature extraction, we significantly improved the rotation invariance of the RAG feature calculation. Moreover, applying our method leads to an improved F1 score and IoU on rotated data. Furthermore, we proposed a wall splitting algorithm for partitioning walls into segments associated with the corresponding rooms. 

---
# Malware Detection in Docker Containers: An Image is Worth a Thousand Logs 

**Authors**: Akis Nousias, Efklidis Katsaros, Evangelos Syrmos, Panagiotis Radoglou-Grammatikis, Thomas Lagkas, Vasileios Argyriou, Ioannis Moscholios, Evangelos Markakis, Sotirios Goudos, Panagiotis Sarigiannidis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03238)  

**Abstract**: Malware detection is increasingly challenged by evolving techniques like obfuscation and polymorphism, limiting the effectiveness of traditional methods. Meanwhile, the widespread adoption of software containers has introduced new security challenges, including the growing threat of malicious software injection, where a container, once compromised, can serve as entry point for further cyberattacks. In this work, we address these security issues by introducing a method to identify compromised containers through machine learning analysis of their file systems. We cast the entire software containers into large RGB images via their tarball representations, and propose to use established Convolutional Neural Network architectures on a streaming, patch-based manner. To support our experiments, we release the COSOCO dataset--the first of its kind--containing 3364 large-scale RGB images of benign and compromised software containers at this https URL. Our method detects more malware and achieves higher F1 and Recall scores than all individual and ensembles of VirusTotal engines, demonstrating its effectiveness and setting a new standard for identifying malware-compromised software containers. 

---
# Crash Time Matters: HybridMamba for Fine-Grained Temporal Localization in Traffic Surveillance Footage 

**Authors**: Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03235)  

**Abstract**: Traffic crash detection in long-form surveillance videos is critical for emergency response and infrastructure planning but remains difficult due to the brief and rare nature of crash events. We introduce HybridMamba, a novel architecture that combines visual transformers with state-space temporal modeling to achieve accurate crash time localization. Our method uses multi-level token compression and hierarchical temporal processing to remain computationally efficient without sacrificing temporal resolution. Evaluated on a large-scale dataset from the Iowa Department of Transportation, HybridMamba achieves a mean absolute error of 1.50 seconds, with 65.2 percent of predictions within one second of the ground truth. It outperforms recent video-language models such as TimeChat and VideoLLaMA2 by up to 2.8 seconds, while using significantly fewer parameters. Our results demonstrate strong generalization across videos ranging from 2 to 40 minutes in diverse conditions. HybridMamba offers a robust and efficient solution for fine-grained temporal localization in traffic surveillance. The code will be released upon publication. 

---
# Think When You Need: Self-Adaptive Chain-of-Thought Learning 

**Authors**: Junjie Yang, Ke Lin, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03234)  

**Abstract**: Chain of Thought (CoT) reasoning enhances language models' performance but often leads to inefficient "overthinking" on simple problems. We identify that existing approaches directly penalizing reasoning length fail to account for varying problem complexity. Our approach constructs rewards through length and quality comparisons, guided by theoretical assumptions that jointly enhance solution correctness with conciseness. Moreover, we further demonstrate our method to fuzzy tasks where ground truth is unavailable. Experiments across multiple reasoning benchmarks demonstrate that our method maintains accuracy while generating significantly more concise explanations, effectively teaching models to "think when needed." 

---
# Persuasive Calibration 

**Authors**: Yiding Feng, Wei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03211)  

**Abstract**: We introduce and study the persuasive calibration problem, where a principal aims to provide trustworthy predictions about underlying events to a downstream agent to make desired decisions. We adopt the standard calibration framework that regulates predictions to be unbiased conditional on their own value, and thus, they can reliably be interpreted at the face value by the agent. Allowing a small calibration error budget, we aim to answer the following question: what is and how to compute the optimal predictor under this calibration error budget, especially when there exists incentive misalignment between the principal and the agent? We focus on standard Lt-norm Expected Calibration Error (ECE) metric.
We develop a general framework by viewing predictors as post-processed versions of perfectly calibrated predictors. Using this framework, we first characterize the structure of the optimal predictor. Specifically, when the principal's utility is event-independent and for L1-norm ECE, we show: (1) the optimal predictor is over-(resp. under-) confident for high (resp. low) true expected outcomes, while remaining perfectly calibrated in the middle; (2) the miscalibrated predictions exhibit a collinearity structure with the principal's utility function. On the algorithmic side, we provide a FPTAS for computing approximately optimal predictor for general principal utility and general Lt-norm ECE. Moreover, for the L1- and L-Infinity-norm ECE, we provide polynomial-time algorithms that compute the exact optimal predictor. 

---
# Augmenting Human Cognition With Generative AI: Lessons From AI-Assisted Decision-Making 

**Authors**: Zelun Tony Zhang, Leon Reicherts  

**Link**: [PDF](https://arxiv.org/pdf/2504.03207)  

**Abstract**: How can we use generative AI to design tools that augment rather than replace human cognition? In this position paper, we review our own research on AI-assisted decision-making for lessons to learn. We observe that in both AI-assisted decision-making and generative AI, a popular approach is to suggest AI-generated end-to-end solutions to users, which users can then accept, reject, or edit. Alternatively, AI tools could offer more incremental support to help users solve tasks themselves, which we call process-oriented support. We describe findings on the challenges of end-to-end solutions, and how process-oriented support can address them. We also discuss the applicability of these findings to generative AI based on a recent study in which we compared both approaches to assist users in a complex decision-making task with LLMs. 

---
# Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward 

**Authors**: Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.03206)  

**Abstract**: Effective conversational agents must be able to personalize their behavior to suit a user's preferences, personality, and attributes, whether they are assisting with writing tasks or operating in domains like education or healthcare. Current training methods like Reinforcement Learning from Human Feedback (RLHF) prioritize helpfulness and safety but fall short in fostering truly empathetic, adaptive, and personalized interactions. Traditional approaches to personalization often rely on extensive user history, limiting their effectiveness for new or context-limited users. To overcome these limitations, we propose to incorporate an intrinsic motivation to improve the conversational agents's model of the user as an additional reward alongside multi-turn RLHF. This reward mechanism encourages the agent to actively elicit user traits by optimizing conversations to increase the accuracy of its user model. Consequently, the policy agent can deliver more personalized interactions through obtaining more information about the user. We applied our method both education and fitness settings, where LLMs teach concepts or recommend personalized strategies based on users' hidden learning style or lifestyle attributes. Using LLM-simulated users, our approach outperformed a multi-turn RLHF baseline in revealing information about the users' preferences, and adapting to them. 

---
# Endo3R: Unified Online Reconstruction from Dynamic Monocular Endoscopic Video 

**Authors**: Jiaxin Guo, Wenzhen Dong, Tianyu Huang, Hao Ding, Ziyi Wang, Haomin Kuang, Qi Dou, Yun-Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03198)  

**Abstract**: Reconstructing 3D scenes from monocular surgical videos can enhance surgeon's perception and therefore plays a vital role in various computer-assisted surgery tasks. However, achieving scale-consistent reconstruction remains an open challenge due to inherent issues in endoscopic videos, such as dynamic deformations and textureless surfaces. Despite recent advances, current methods either rely on calibration or instrument priors to estimate scale, or employ SfM-like multi-stage pipelines, leading to error accumulation and requiring offline optimization. In this paper, we present Endo3R, a unified 3D foundation model for online scale-consistent reconstruction from monocular surgical video, without any priors or extra optimization. Our model unifies the tasks by predicting globally aligned pointmaps, scale-consistent video depths, and camera parameters without any offline optimization. The core contribution of our method is expanding the capability of the recent pairwise reconstruction model to long-term incremental dynamic reconstruction by an uncertainty-aware dual memory mechanism. The mechanism maintains history tokens of both short-term dynamics and long-term spatial consistency. Notably, to tackle the highly dynamic nature of surgical scenes, we measure the uncertainty of tokens via Sampson distance and filter out tokens with high uncertainty. Regarding the scarcity of endoscopic datasets with ground-truth depth and camera poses, we further devise a self-supervised mechanism with a novel dynamics-aware flow loss. Abundant experiments on SCARED and Hamlyn datasets demonstrate our superior performance in zero-shot surgical video depth prediction and camera pose estimation with online efficiency. Project page: this https URL. 

---
# Learning Natural Language Constraints for Safe Reinforcement Learning of Language Agents 

**Authors**: Jaymari Chua, Chen Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03185)  

**Abstract**: Generalizable alignment is a core challenge for deploying Large Language Models (LLMs) safely in real-world NLP applications. Current alignment methods, including Reinforcement Learning from Human Feedback (RLHF), often fail to guarantee constraint satisfaction outside their training distribution due to their reliance on implicit, post-hoc preferences. Inspired by a paradigm shift to first curate data before tuning, we introduce a new framework for safe language alignment that learns natural language constraints from positive and negative demonstrations as a primary step. From inferring both a task-specific reward function and latent constraint functions, our approach fosters adaptation to novel safety requirements and robust generalization under domain shifts and adversarial inputs. We formalize the framework within a Constrained Markov Decision Process (CMDP) and validate it via a text-based navigation environment, demonstrating safe adaptation to changing danger zones. Our experiments show fewer violations upon domain shift when following a safe navigation path, and we achieve zero violations by applying learned constraints to a distilled BERT model as a fine-tuning technique. This work offers a promising path toward building safety-critical and more generalizable LLMs for practical NLP settings. 

---
# Real-Time Roadway Obstacle Detection for Electric Scooters Using Deep Learning and Multi-Sensor Fusion 

**Authors**: Zeyang Zheng, Arman Hosseini, Dong Chen, Omid Shoghli, Arsalan Heydarian  

**Link**: [PDF](https://arxiv.org/pdf/2504.03171)  

**Abstract**: The increasing adoption of electric scooters (e-scooters) in urban areas has coincided with a rise in traffic accidents and injuries, largely due to their small wheels, lack of suspension, and sensitivity to uneven surfaces. While deep learning-based object detection has been widely used to improve automobile safety, its application for e-scooter obstacle detection remains unexplored. This study introduces a novel ground obstacle detection system for e-scooters, integrating an RGB camera, and a depth camera to enhance real-time road hazard detection. Additionally, the Inertial Measurement Unit (IMU) measures linear vertical acceleration to identify surface vibrations, guiding the selection of six obstacle categories: tree branches, manhole covers, potholes, pine cones, non-directional cracks, and truncated domes. All sensors, including the RGB camera, depth camera, and IMU, are integrated within the Intel RealSense Camera D435i. A deep learning model powered by YOLO detects road hazards and utilizes depth data to estimate obstacle proximity. Evaluated on the seven hours of naturalistic riding dataset, the system achieves a high mean average precision (mAP) of 0.827 and demonstrates excellent real-time performance. This approach provides an effective solution to enhance e-scooter safety through advanced computer vision and data fusion. The dataset is accessible at this https URL, and the project code is hosted on this https URL. 

---
# NuScenes-SpatialQA: A Spatial Understanding and Reasoning Benchmark for Vision-Language Models in Autonomous Driving 

**Authors**: Kexin Tian, Jingrui Mao, Yunlong Zhang, Jiwan Jiang, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03164)  

**Abstract**: Recent advancements in Vision-Language Models (VLMs) have demonstrated strong potential for autonomous driving tasks. However, their spatial understanding and reasoning-key capabilities for autonomous driving-still exhibit significant limitations. Notably, none of the existing benchmarks systematically evaluate VLMs' spatial reasoning capabilities in driving scenarios. To fill this gap, we propose NuScenes-SpatialQA, the first large-scale ground-truth-based Question-Answer (QA) benchmark specifically designed to evaluate the spatial understanding and reasoning capabilities of VLMs in autonomous driving. Built upon the NuScenes dataset, the benchmark is constructed through an automated 3D scene graph generation pipeline and a QA generation pipeline. The benchmark systematically evaluates VLMs' performance in both spatial understanding and reasoning across multiple dimensions. Using this benchmark, we conduct extensive experiments on diverse VLMs, including both general and spatial-enhanced models, providing the first comprehensive evaluation of their spatial capabilities in autonomous driving. Surprisingly, the experimental results show that the spatial-enhanced VLM outperforms in qualitative QA but does not demonstrate competitiveness in quantitative QA. In general, VLMs still face considerable challenges in spatial understanding and reasoning. 

---
# A Human Digital Twin Architecture for Knowledge-based Interactions and Context-Aware Conversations 

**Authors**: Abdul Mannan Mohammed, Azhar Ali Mohammad, Jason A. Ortiz, Carsten Neumann, Grace Bochenek, Dirk Reiners, Carolina Cruz-Neira  

**Link**: [PDF](https://arxiv.org/pdf/2504.03147)  

**Abstract**: Recent developments in Artificial Intelligence (AI) and Machine Learning (ML) are creating new opportunities for Human-Autonomy Teaming (HAT) in tasks, missions, and continuous coordinated activities. A major challenge is enabling humans to maintain awareness and control over autonomous assets, while also building trust and supporting shared contextual understanding. To address this, we present a real-time Human Digital Twin (HDT) architecture that integrates Large Language Models (LLMs) for knowledge reporting, answering, and recommendation, embodied in a visual interface.
The system applies a metacognitive approach to enable personalized, context-aware responses aligned with the human teammate's expectations. The HDT acts as a visually and behaviorally realistic team member, integrated throughout the mission lifecycle, from training to deployment to after-action review. Our architecture includes speech recognition, context processing, AI-driven dialogue, emotion modeling, lip-syncing, and multimodal feedback. We describe the system design, performance metrics, and future development directions for more adaptive and realistic HAT systems. 

---
# Hierarchical Modeling for Medical Visual Question Answering with Cross-Attention Fusion 

**Authors**: Junkai Zhang, Bin Li, Shoujun Zhou, Yue Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.03135)  

**Abstract**: Medical Visual Question Answering (Med-VQA) answers clinical questions using medical images, aiding diagnosis. Designing the MedVQA system holds profound importance in assisting clinical diagnosis and enhancing diagnostic accuracy. Building upon this foundation, Hierarchical Medical VQA extends Medical VQA by organizing medical questions into a hierarchical structure and making level-specific predictions to handle fine-grained distinctions. Recently, many studies have proposed hierarchical MedVQA tasks and established datasets, However, several issues still remain: (1) imperfect hierarchical modeling leads to poor differentiation between question levels causing semantic fragmentation across hierarchies. (2) Excessive reliance on implicit learning in Transformer-based cross-modal self-attention fusion methods, which obscures crucial local semantic correlations in medical scenarios. To address these issues, this study proposes a HiCA-VQA method, including two modules: Hierarchical Prompting for fine-grained medical questions and Hierarchical Answer Decoders. The hierarchical prompting module pre-aligns hierarchical text prompts with image features to guide the model in focusing on specific image regions according to question types, while the hierarchical decoder performs separate predictions for questions at different levels to improve accuracy across granularities. The framework also incorporates a cross-attention fusion module where images serve as queries and text as key-value pairs. Experiments on the Rad-Restruct benchmark demonstrate that the HiCA-VQA framework better outperforms existing state-of-the-art methods in answering hierarchical fine-grained questions. This study provides an effective pathway for hierarchical visual question answering systems, advancing medical image understanding. 

---
# GraphSeg: Segmented 3D Representations via Graph Edge Addition and Contraction 

**Authors**: Haozhan Tang, Tianyi Zhang, Oliver Kroemer, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03129)  

**Abstract**: Robots operating in unstructured environments often require accurate and consistent object-level representations. This typically requires segmenting individual objects from the robot's surroundings. While recent large models such as Segment Anything (SAM) offer strong performance in 2D image segmentation. These advances do not translate directly to performance in the physical 3D world, where they often over-segment objects and fail to produce consistent mask correspondences across views. In this paper, we present GraphSeg, a framework for generating consistent 3D object segmentations from a sparse set of 2D images of the environment without any depth information. GraphSeg adds edges to graphs and constructs dual correspondence graphs: one from 2D pixel-level similarities and one from inferred 3D structure. We formulate segmentation as a problem of edge addition, then subsequent graph contraction, which merges multiple 2D masks into unified object-level segmentations. We can then leverage \emph{3D foundation models} to produce segmented 3D representations. GraphSeg achieves robust segmentation with significantly fewer images and greater accuracy than prior methods. We demonstrate state-of-the-art performance on tabletop scenes and show that GraphSeg enables improved performance on downstream robotic manipulation tasks. Code available at this https URL. 

---
# Graph Network Modeling Techniques for Visualizing Human Mobility Patterns 

**Authors**: Sinjini Mitra, Anuj Srivastava, Avipsa Roy, Pavan Turaga  

**Link**: [PDF](https://arxiv.org/pdf/2504.03119)  

**Abstract**: Human mobility analysis at urban-scale requires models to represent the complex nature of human movements, which in turn are affected by accessibility to nearby points of interest, underlying socioeconomic factors of a place, and local transport choices for people living in a geographic region. In this work, we represent human mobility and the associated flow of movements as a grapyh. Graph-based approaches for mobility analysis are still in their early stages of adoption and are actively being researched. The challenges of graph-based mobility analysis are multifaceted - the lack of sufficiently high-quality data to represent flows at high spatial and teporal resolution whereas, limited computational resources to translate large voluments of mobility data into a network structure, and scaling issues inherent in graph models etc. The current study develops a methodology by embedding graphs into a continuous space, which alleviates issues related to fast graph matching, graph time-series modeling, and visualization of mobility dynamics. Through experiments, we demonstrate how mobility data collected from taxicab trajectories could be transformed into network structures and patterns of mobility flow changes, and can be used for downstream tasks reporting approx 40% decrease in error on average in matched graphs vs unmatched ones. 

---
# NuWa: Deriving Lightweight Task-Specific Vision Transformers for Edge Devices 

**Authors**: Ziteng Wei, Qiang He, Bing Li, Feifei Chen, Yun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03118)  

**Abstract**: Vision Transformers (ViTs) excel in computer vision tasks but lack flexibility for edge devices' diverse needs. A vital issue is that ViTs pre-trained to cover a broad range of tasks are \textit{over-qualified} for edge devices that usually demand only part of a ViT's knowledge for specific tasks. Their task-specific accuracy on these edge devices is suboptimal. We discovered that small ViTs that focus on device-specific tasks can improve model accuracy and in the meantime, accelerate model inference. This paper presents NuWa, an approach that derives small ViTs from the base ViT for edge devices with specific task requirements. NuWa can transfer task-specific knowledge extracted from the base ViT into small ViTs that fully leverage constrained resources on edge devices to maximize model accuracy with inference latency assurance. Experiments with three base ViTs on three public datasets demonstrate that compared with state-of-the-art solutions, NuWa improves model accuracy by up to $\text{11.83}\%$ and accelerates model inference by 1.29$\times$ - 2.79$\times$. Code for reproduction is available at this https URL. 

---
# Multi-Granularity Vision Fastformer with Fusion Mechanism for Skin Lesion Segmentation 

**Authors**: Xuanyu Liu, Huiyun Yao, Jinggui Gao, Zhongyi Guo, Xue Zhang, Yulin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03108)  

**Abstract**: Background:Convolutional Neural Networks(CNN) and Vision Transformers(ViT) are the main techniques used in Medical image segmentation. However, CNN is limited to local contextual information, and ViT's quadratic complexity results in significant computational costs. At the same time, equipping the model to distinguish lesion boundaries with varying degrees of severity is also a challenge encountered in skin lesion segmentation. Purpose:This research aims to optimize the balance between computational costs and long-range dependency modelling and achieve excellent generalization across lesions with different degrees of severity. Methods:we propose a lightweight U-shape network that utilizes Vision Fastformer with Fusion Mechanism (VFFM-UNet). We inherit the advantages of Fastformer's additive attention mechanism, combining element-wise product and matrix product for comprehensive feature extraction and channel reduction to save computational costs. In order to accurately identify the lesion boundaries with varying degrees of severity, we designed Fusion Mechanism including Multi-Granularity Fusion and Channel Fusion, which can process the feature maps in the granularity and channel levels to obtain different contextual information. Results:Comprehensive experiments on the ISIC2017, ISIC2018 and PH2 datasets demonstrate that VFFM-UNet outperforms existing state-of-the-art models regarding parameter numbers, computational complexity and segmentation performance. In short, compared to MISSFormer, our model achieves superior segmentation performance while reducing parameter and computation costs by 101x and 15x, respectively. Conclusions:Both quantitative and qualitative analyses show that VFFM-UNet sets a new benchmark by reaching an ideal balance between parameter numbers, computational complexity, and segmentation performance compared to existing state-of-the-art models. 

---
# Post-processing for Fair Regression via Explainable SVD 

**Authors**: Zhiqun Zuo, Ding Zhu, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2504.03093)  

**Abstract**: This paper presents a post-processing algorithm for training fair neural network regression models that satisfy statistical parity, utilizing an explainable singular value decomposition (SVD) of the weight matrix. We propose a linear transformation of the weight matrix, whereby the singular values derived from the SVD of the transformed matrix directly correspond to the differences in the first and second moments of the output distributions across two groups. Consequently, we can convert the fairness constraints into constraints on the singular values. We analytically solve the problem of finding the optimal weights under these constraints. Experimental validation on various datasets demonstrates that our method achieves a similar or superior fairness-accuracy trade-off compared to the baselines without using the sensitive attribute at the inference time. 

---
# Machine Learning-Based Detection and Analysis of Suspicious Activities in Bitcoin Wallet Transactions in the USA 

**Authors**: Md Zahidul Islam, Md Shahidul Islam, Biswajit Chandra das, Syed Ali Reza, Proshanta Kumar Bhowmik, Kanchon Kumar Bishnu, Md Shafiqur Rahman, Redoyan Chowdhury, Laxmi Pant  

**Link**: [PDF](https://arxiv.org/pdf/2504.03092)  

**Abstract**: The dramatic adoption of Bitcoin and other cryptocurrencies in the USA has revolutionized the financial landscape and provided unprecedented investment and transaction efficiency opportunities. The prime objective of this research project is to develop machine learning algorithms capable of effectively identifying and tracking suspicious activity in Bitcoin wallet transactions. With high-tech analysis, the study aims to create a model with a feature for identifying trends and outliers that can expose illicit activity. The current study specifically focuses on Bitcoin transaction information in America, with a strong emphasis placed on the importance of knowing about the immediate environment in and through which such transactions pass through. The dataset is composed of in-depth Bitcoin wallet transactional information, including important factors such as transaction values, timestamps, network flows, and addresses for wallets. All entries in the dataset expose information about financial transactions between wallets, including received and sent transactions, and such information is significant for analysis and trends that can represent suspicious activity. This study deployed three accredited algorithms, most notably, Logistic Regression, Random Forest, and Support Vector Machines. In retrospect, Random Forest emerged as the best model with the highest F1 Score, showcasing its ability to handle non-linear relationships in the data. Insights revealed significant patterns in wallet activity, such as the correlation between unredeemed transactions and final balances. The application of machine algorithms in tracking cryptocurrencies is a tool for creating transparent and secure U.S. markets. 

---
# From Questions to Insights: Exploring XAI Challenges Reported on Stack Overflow Questions 

**Authors**: Saumendu Roy, Saikat Mondal, Banani Roy, Chanchal Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.03085)  

**Abstract**: The lack of interpretability is a major barrier that limits the practical usage of AI models. Several eXplainable AI (XAI) techniques (e.g., SHAP, LIME) have been employed to interpret these models' performance. However, users often face challenges when leveraging these techniques in real-world scenarios and thus submit questions in technical Q&A forums like Stack Overflow (SO) to resolve these challenges. We conducted an exploratory study to expose these challenges, their severity, and features that can make XAI techniques more accessible and easier to use. Our contributions to this study are fourfold. First, we manually analyzed 663 SO questions that discussed challenges related to XAI techniques. Our careful investigation produced a catalog of seven challenges (e.g., disagreement issues). We then analyzed their prevalence and found that model integration and disagreement issues emerged as the most prevalent challenges. Second, we attempt to estimate the severity of each XAI challenge by determining the correlation between challenge types and answer metadata (e.g., the presence of accepted answers). Our analysis suggests that model integration issues is the most severe challenge. Third, we attempt to perceive the severity of these challenges based on practitioners' ability to use XAI techniques effectively in their work. Practitioners' responses suggest that disagreement issues most severely affect the use of XAI techniques. Fourth, we seek agreement from practitioners on improvements or features that could make XAI techniques more accessible and user-friendly. The majority of them suggest consistency in explanations and simplified integration. Our study findings might (a) help to enhance the accessibility and usability of XAI and (b) act as the initial benchmark that can inspire future research. 

---
# Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning 

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip  

**Link**: [PDF](https://arxiv.org/pdf/2504.03077)  

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats. 

---
# AD-GPT: Large Language Models in Alzheimer's Disease 

**Authors**: Ziyu Liu, Lintao Tang, Zeliang Sun, Zhengliang Liu, Yanjun Lyu, Wei Ruan, Yangshuang Xu, Liang Shan, Jiyoon Shin, Xiaohe Chen, Dajiang Zhu, Tianming Liu, Rongjie Liu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03071)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools for medical information retrieval, yet their accuracy and depth remain limited in specialized domains such as Alzheimer's disease (AD), a growing global health challenge. To address this gap, we introduce AD-GPT, a domain-specific generative pre-trained transformer designed to enhance the retrieval and analysis of AD-related genetic and neurobiological information. AD-GPT integrates diverse biomedical data sources, including potential AD-associated genes, molecular genetic information, and key gene variants linked to brain regions. We develop a stacked LLM architecture combining Llama3 and BERT, optimized for four critical tasks in AD research: (1) genetic information retrieval, (2) gene-brain region relationship assessment, (3) gene-AD relationship analysis, and (4) brain region-AD relationship mapping. Comparative evaluations against state-of-the-art LLMs demonstrate AD-GPT's superior precision and reliability across these tasks, underscoring its potential as a robust and specialized AI tool for advancing AD research and biomarker discovery. 

---
# Properties of Fixed Points of Generalised Extra Gradient Methods Applied to Min-Max Problems 

**Authors**: Amir Ali Farzin, Yuen-Man Pun, Philipp Braun, Iman Shames  

**Link**: [PDF](https://arxiv.org/pdf/2504.03069)  

**Abstract**: This paper studies properties of fixed points of generalised Extra-gradient (GEG) algorithms applied to min-max problems. We discuss connections between saddle points of the objective function of the min-max problem and GEG fixed points. We show that, under appropriate step-size selections, the set of saddle points (Nash equilibria) is a subset of stable fixed points of GEG. Convergence properties of the GEG algorithm are obtained through a stability analysis of a discrete-time dynamical system. The results and benefits when compared to existing methods are illustrated through numerical examples. 

---
# Design of AI-Powered Tool for Self-Regulation Support in Programming Education 

**Authors**: Huiyong Li, Boxuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03068)  

**Abstract**: Large Language Model (LLM) tools have demonstrated their potential to deliver high-quality assistance by providing instant, personalized feedback that is crucial for effective programming education. However, many of these tools operate independently from institutional Learning Management Systems, which creates a significant disconnect. This isolation limits the ability to leverage learning materials and exercise context for generating tailored, context-aware feedback. Furthermore, previous research on self-regulated learning and LLM support mainly focused on knowledge acquisition, not the development of important self-regulation skills. To address these challenges, we developed CodeRunner Agent, an LLM-based programming assistant that integrates the CodeRunner, a student-submitted code executing and automated grading plugin in Moodle. CodeRunner Agent empowers educators to customize AI-generated feedback by incorporating detailed context from lecture materials, programming questions, student answers, and execution results. Additionally, it enhances students' self-regulated learning by providing strategy-based AI responses. This integrated, context-aware, and skill-focused approach offers promising avenues for data-driven improvements in programming education. 

---
# Context-Aware Self-Adaptation for Domain Generalization 

**Authors**: Hao Yan, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03064)  

**Abstract**: Domain generalization aims at developing suitable learning algorithms in source training domains such that the model learned can generalize well on a different unseen testing domain. We present a novel two-stage approach called Context-Aware Self-Adaptation (CASA) for domain generalization. CASA simulates an approximate meta-generalization scenario and incorporates a self-adaptation module to adjust pre-trained meta source models to the meta-target domains while maintaining their predictive capability on the meta-source domains. The core concept of self-adaptation involves leveraging contextual information, such as the mean of mini-batch features, as domain knowledge to automatically adapt a model trained in the first stage to new contexts in the second stage. Lastly, we utilize an ensemble of multiple meta-source models to perform inference on the testing domain. Experimental results demonstrate that our proposed method achieves state-of-the-art performance on standard benchmarks. 

---
# Cooperative Inference for Real-Time 3D Human Pose Estimation in Multi-Device Edge Networks 

**Authors**: Hyun-Ho Choi, Kangsoo Kim, Ki-Ho Lee, Kisong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.03052)  

**Abstract**: Accurate and real-time three-dimensional (3D) pose estimation is challenging in resource-constrained and dynamic environments owing to its high computational complexity. To address this issue, this study proposes a novel cooperative inference method for real-time 3D human pose estimation in mobile edge computing (MEC) networks. In the proposed method, multiple end devices equipped with lightweight inference models employ dual confidence thresholds to filter ambiguous images. Only the filtered images are offloaded to an edge server with a more powerful inference model for re-evaluation, thereby improving the estimation accuracy under computational and communication constraints. We numerically analyze the performance of the proposed inference method in terms of the inference accuracy and end-to-end delay and formulate a joint optimization problem to derive the optimal confidence thresholds and transmission time for each device, with the objective of minimizing the mean per-joint position error (MPJPE) while satisfying the required end-to-end delay constraint. To solve this problem, we demonstrate that minimizing the MPJPE is equivalent to maximizing the sum of the inference accuracies for all devices, decompose the problem into manageable subproblems, and present a low-complexity optimization algorithm to obtain a near-optimal solution. The experimental results show that a trade-off exists between the MPJPE and end-to-end delay depending on the confidence thresholds. Furthermore, the results confirm that the proposed cooperative inference method achieves a significant reduction in the MPJPE through the optimal selection of confidence thresholds and transmission times, while consistently satisfying the end-to-end delay requirement in various MEC environments. 

---
# Task as Context Prompting for Accurate Medical Symptom Coding Using Large Language Models 

**Authors**: Chengyang He, Wenlong Zhang, Violet Xinying Chen, Yue Ning, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03051)  

**Abstract**: Accurate medical symptom coding from unstructured clinical text, such as vaccine safety reports, is a critical task with applications in pharmacovigilance and safety monitoring. Symptom coding, as tailored in this study, involves identifying and linking nuanced symptom mentions to standardized vocabularies like MedDRA, differentiating it from broader medical coding tasks. Traditional approaches to this task, which treat symptom extraction and linking as independent workflows, often fail to handle the variability and complexity of clinical narratives, especially for rare cases. Recent advancements in Large Language Models (LLMs) offer new opportunities but face challenges in achieving consistent performance. To address these issues, we propose Task as Context (TACO) Prompting, a novel framework that unifies extraction and linking tasks by embedding task-specific context into LLM prompts. Our study also introduces SYMPCODER, a human-annotated dataset derived from Vaccine Adverse Event Reporting System (VAERS) reports, and a two-stage evaluation framework to comprehensively assess both symptom linking and mention fidelity. Our comprehensive evaluation of multiple LLMs, including Llama2-chat, Jackalope-7b, GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o, demonstrates TACO's effectiveness in improving flexibility and accuracy for tailored tasks like symptom coding, paving the way for more specific coding tasks and advancing clinical text processing methodologies. 

---
# Safety Modulation: Enhancing Safety in Reinforcement Learning through Cost-Modulated Rewards 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03040)  

**Abstract**: Safe Reinforcement Learning (Safe RL) aims to train an RL agent to maximize its performance in real-world environments while adhering to safety constraints, as exceeding safety violation limits can result in severe consequences. In this paper, we propose a novel safe RL approach called Safety Modulated Policy Optimization (SMPO), which enables safe policy function learning within the standard policy optimization framework through safety modulated rewards. In particular, we consider safety violation costs as feedback from the RL environments that are parallel to the standard awards, and introduce a Q-cost function as safety critic to estimate expected future cumulative costs. Then we propose to modulate the rewards using a cost-aware weighting function, which is carefully designed to ensure the safety limits based on the estimation of the safety critic, while maximizing the expected rewards. The policy function and the safety critic are simultaneously learned through gradient descent during online interactions with the environment. We conduct experiments using multiple RL environments and the experimental results demonstrate that our method outperforms several classic and state-of-the-art comparison methods in terms of overall safe RL performance. 

---
# Deep Reinforcement Learning via Object-Centric Attention 

**Authors**: Jannis Blüml, Cedric Derstroff, Bjarne Gregori, Elisabeth Dillies, Quentin Delfosse, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2504.03024)  

**Abstract**: Deep reinforcement learning agents, trained on raw pixel inputs, often fail to generalize beyond their training environments, relying on spurious correlations and irrelevant background details. To address this issue, object-centric agents have recently emerged. However, they require different representations tailored to the task specifications. Contrary to deep agents, no single object-centric architecture can be applied to any environment. Inspired by principles of cognitive science and Occam's Razor, we introduce Object-Centric Attention via Masking (OCCAM), which selectively preserves task-relevant entities while filtering out irrelevant visual information. Specifically, OCCAM takes advantage of the object-centric inductive bias. Empirical evaluations on Atari benchmarks demonstrate that OCCAM significantly improves robustness to novel perturbations and reduces sample complexity while showing similar or improved performance compared to conventional pixel-based RL. These results suggest that structured abstraction can enhance generalization without requiring explicit symbolic representations or domain-specific object extraction pipelines. 

---
# The Dual-Route Model of Induction 

**Authors**: Sheridan Feucht, Eric Todd, Byron Wallace, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2504.03022)  

**Abstract**: Prior work on in-context copying has shown the existence of induction heads, which attend to and promote individual tokens during copying. In this work we introduce a new type of induction head: concept-level induction heads, which copy entire lexical units instead of individual tokens. Concept induction heads learn to attend to the ends of multi-token words throughout training, working in parallel with token-level induction heads to copy meaningful text. We show that these heads are responsible for semantic tasks like word-level translation, whereas token induction heads are vital for tasks that can only be done verbatim, like copying nonsense tokens. These two "routes" operate independently: in fact, we show that ablation of token induction heads causes models to paraphrase where they would otherwise copy verbatim. In light of these findings, we argue that although token induction heads are vital for specific tasks, concept induction heads may be more broadly relevant for in-context learning. 

---
# Localized Definitions and Distributed Reasoning: A Proof-of-Concept Mechanistic Interpretability Study via Activation Patching 

**Authors**: Nooshin Bahador  

**Link**: [PDF](https://arxiv.org/pdf/2504.02976)  

**Abstract**: This study investigates the localization of knowledge representation in fine-tuned GPT-2 models using Causal Layer Attribution via Activation Patching (CLAP), a method that identifies critical neural layers responsible for correct answer generation. The model was fine-tuned on 9,958 PubMed abstracts (epilepsy: 20,595 mentions, EEG: 11,674 mentions, seizure: 13,921 mentions) using two configurations with validation loss monitoring for early stopping. CLAP involved (1) caching clean (correct answer) and corrupted (incorrect answer) activations, (2) computing logit difference to quantify model preference, and (3) patching corrupted activations with clean ones to assess recovery. Results revealed three findings: First, patching the first feedforward layer recovered 56% of correct preference, demonstrating that associative knowledge is distributed across multiple layers. Second, patching the final output layer completely restored accuracy (100% recovery), indicating that definitional knowledge is localised. The stronger clean logit difference for definitional questions further supports this localized representation. Third, minimal recovery from convolutional layer patching (13.6%) suggests low-level features contribute marginally to high-level reasoning. Statistical analysis confirmed significant layer-specific effects (p<0.01). These findings demonstrate that factual knowledge is more localized and associative knowledge depends on distributed representations. We also showed that editing efficacy depends on task type. Our findings not only reconcile conflicting observations about localization in model editing but also emphasize on using task-adaptive techniques for reliable, interpretable updates. 

---
# Improved Compact Genetic Algorithms with Efficient Caching 

**Authors**: Prasanta Dutta, Anirban Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2504.02972)  

**Abstract**: Compact Genetic Algorithms (cGAs) are condensed variants of classical Genetic Algorithms (GAs) that use a probability vector representation of the population instead of the complete population. cGAs have been shown to significantly reduce the number of function evaluations required while producing outcomes similar to those of classical GAs. However, cGAs have a tendency to repeatedly generate the same chromosomes as they approach convergence, resulting in unnecessary evaluations of identical chromosomes. This article introduces the concept of caching in cGAs as a means of avoiding redundant evaluations of the same chromosomes. Our proposed approach operates equivalently to cGAs, but enhances the algorithm's time efficiency by reducing the number of function evaluations. We also present a data structure for efficient cache maintenance to ensure low overhead. The proposed caching approach has an asymptotically constant time complexity on average. The proposed method further generalizes the caching mechanism with higher selection pressure for elitism-based cGAs. We conduct a rigorous analysis based on experiments on benchmark optimization problems using two well-known cache replacement strategies. The results demonstrate that caching significantly reduces the number of function evaluations required while maintaining the same level of performance accuracy. 

---
# Global-Order GFlowNets 

**Authors**: Lluís Pastor-Pérez, Javier Alonso-Garcia, Lukas Mauch  

**Link**: [PDF](https://arxiv.org/pdf/2504.02968)  

**Abstract**: Order-Preserving (OP) GFlowNets have demonstrated remarkable success in tackling complex multi-objective (MOO) black-box optimization problems using stochastic optimization techniques. Specifically, they can be trained online to efficiently sample diverse candidates near the Pareto front. A key advantage of OP GFlowNets is their ability to impose a local order on training samples based on Pareto dominance, eliminating the need for scalarization - a common requirement in other approaches like Preference-Conditional GFlowNets. However, we identify an important limitation of OP GFlowNets: imposing a local order on training samples can lead to conflicting optimization objectives. To address this issue, we introduce Global-Order GFlowNets, which transform the local order into a global one, thereby resolving these conflicts. Our experimental evaluations on various benchmarks demonstrate the efficacy and promise of our proposed method. 

---
# CoLa -- Learning to Interactively Collaborate with Large LMs 

**Authors**: Abhishek Sharma, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.02965)  

**Abstract**: LLMs' remarkable ability to tackle a wide range of language tasks opened new opportunities for collaborative human-AI problem solving. LLMs can amplify human capabilities by applying their intuitions and reasoning strategies at scale. We explore whether human guides can be simulated, by generalizing from human demonstrations of guiding an AI system to solve complex language problems. We introduce CoLa, a novel self-guided learning paradigm for training automated $\textit{guides}$ and evaluate it on two QA datasets, a puzzle-solving task, and a constrained text generation task. Our empirical results show that CoLa consistently outperforms competitive approaches across all domains. Moreover, a small-sized trained guide outperforms a strong model like GPT-4 when acting as a guide. We compare the strategies employed by humans and automated guides by conducting a human study on a QA dataset. We show that automated guides outperform humans by adapting their strategies to reasoners' capabilities and conduct qualitative analyses highlighting distinct differences in guiding strategies. 

---
# Digital Forensics in the Age of Large Language Models 

**Authors**: Zhipeng Yin, Zichong Wang, Weifeng Xu, Jun Zhuang, Pallab Mozumder, Antoinette Smith, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02963)  

**Abstract**: Digital forensics plays a pivotal role in modern investigative processes, utilizing specialized methods to systematically collect, analyze, and interpret digital evidence for judicial proceedings. However, traditional digital forensic techniques are primarily based on manual labor-intensive processes, which become increasingly insufficient with the rapid growth and complexity of digital data. To this end, Large Language Models (LLMs) have emerged as powerful tools capable of automating and enhancing various digital forensic tasks, significantly transforming the field. Despite the strides made, general practitioners and forensic experts often lack a comprehensive understanding of the capabilities, principles, and limitations of LLM, which limits the full potential of LLM in forensic applications. To fill this gap, this paper aims to provide an accessible and systematic overview of how LLM has revolutionized the digital forensics approach. Specifically, it takes a look at the basic concepts of digital forensics, as well as the evolution of LLM, and emphasizes the superior capabilities of LLM. To connect theory and practice, relevant examples and real-world scenarios are discussed. We also critically analyze the current limitations of applying LLMs to digital forensics, including issues related to illusion, interpretability, bias, and ethical considerations. In addition, this paper outlines the prospects for future research, highlighting the need for effective use of LLMs for transparency, accountability, and robust standardization in the forensic process. 

---
# Level Up Peer Review in Education: Investigating genAI-driven Gamification system and its influence on Peer Feedback Effectiveness 

**Authors**: Rafal Wlodarski, Leonardo da Silva Sousa, Allison Connell Pensky  

**Link**: [PDF](https://arxiv.org/pdf/2504.02962)  

**Abstract**: In software engineering (SE), the ability to review code and critique designs is essential for professional practice. However, these skills are rarely emphasized in formal education, and peer feedback quality and engagement can vary significantly among students. This paper introduces Socratique, a gamified peer-assessment platform integrated with Generative AI (GenAI) assistance, designed to develop students' peer-review skills in a functional programming course. By incorporating game elements, Socratique aims to motivate students to provide more feedback, while the GenAI assistant offers real-time support in crafting high quality, constructive comments. To evaluate the impact of this approach, we conducted a randomized controlled experiment with master's students comparing a treatment group with a gamified, GenAI-driven setup against a control group with minimal gamification. Results show that students in the treatment group provided significantly more voluntary feedback, with higher scores on clarity, relevance, and specificity - all key aspects of effective code and design reviews. This study provides evidence for the effectiveness of combining gamification and AI to improve peer review processes, with implications for fostering review-related competencies in software engineering curricula. 

---
# VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning 

**Authors**: Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.02949)  

**Abstract**: In this work, we present VARGPT-v1.1, an advanced unified visual autoregressive model that builds upon our previous framework VARGPT. The model preserves the dual paradigm of next-token prediction for visual understanding and next-scale generation for image synthesis. Specifically, VARGPT-v1.1 integrates: (1) a novel training strategy combining iterative visual instruction tuning with reinforcement learning through Direct Preference Optimization (DPO), (2) an expanded training corpus containing 8.3M visual-generative instruction pairs, (3) an upgraded language model backbone using Qwen2, (4) enhanced image generation resolution, and (5) emergent image editing capabilities without architectural modifications. These advancements enable VARGPT-v1.1 to achieve state-of-the-art performance in multimodal understanding and text-to-image instruction-following tasks, demonstrating significant improvements in both comprehension and generation metrics. Notably, through visual instruction tuning, the model acquires image editing functionality while maintaining architectural consistency with its predecessor, revealing the potential for unified visual understanding, generation, and editing. Our findings suggest that well-designed unified visual autoregressive models can effectively adopt flexible training strategies from large language models (LLMs), exhibiting promising scalability. The codebase and model weights are publicly available at this https URL. 

---
# Graph Attention for Heterogeneous Graphs with Positional Encoding 

**Authors**: Nikhil Shivakumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.02938)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as the de facto standard for modeling graph data, with attention mechanisms and transformers significantly enhancing their performance on graph-based tasks. Despite these advancements, the performance of GNNs on heterogeneous graphs often remains complex, with networks generally underperforming compared to their homogeneous counterparts. This work benchmarks various GNN architectures to identify the most effective methods for heterogeneous graphs, with a particular focus on node classification and link prediction. Our findings reveal that graph attention networks excel in these tasks. As a main contribution, we explore enhancements to these attention networks by integrating positional encodings for node embeddings. This involves utilizing the full Laplacian spectrum to accurately capture both the relative and absolute positions of each node within the graph, further enhancing performance on downstream tasks such as node classification and link prediction. 

---
# Robustly identifying concepts introduced during chat fine-tuning using crosscoders 

**Authors**: Julian Minder, Clement Dumas, Caden Juang, Bilal Chugtai, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2504.02922)  

**Abstract**: Model diffing is the study of how fine-tuning changes a model's representations and internal algorithms. Many behaviours of interest are introduced during fine-tuning, and model diffing offers a promising lens to interpret such behaviors. Crosscoders are a recent model diffing method that learns a shared dictionary of interpretable concepts represented as latent directions in both the base and fine-tuned models, allowing us to track how concepts shift or emerge during fine-tuning. Notably, prior work has observed concepts with no direction in the base model, and it was hypothesized that these model-specific latents were concepts introduced during fine-tuning. However, we identify two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models. We develop Latent Scaling to flag these issues by more accurately measuring each latent's presence across models. In experiments comparing Gemma 2 2B base and chat models, we observe that the standard crosscoder suffers heavily from these issues. Building on these insights, we train a crosscoder with BatchTopK loss and show that it substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts. We recommend practitioners adopt similar techniques. Using the BatchTopK crosscoder, we successfully identify a set of genuinely chat-specific latents that are both interpretable and causally effective, representing concepts such as $\textit{false information}$ and $\textit{personal question}$, along with multiple refusal-related latents that show nuanced preferences for different refusal triggers. Overall, our work advances best practices for the crosscoder-based methodology for model diffing and demonstrates that it can provide concrete insights into how chat tuning modifies language model behavior. 

---
# Bias in Large Language Models Across Clinical Applications: A Systematic Review 

**Authors**: Thanathip Suenghataiphorn, Narisara Tribuddharat, Pojsakorn Danpanichkul, Narathorn Kulthamrongsri  

**Link**: [PDF](https://arxiv.org/pdf/2504.02917)  

**Abstract**: Background: Large language models (LLMs) are rapidly being integrated into healthcare, promising to enhance various clinical tasks. However, concerns exist regarding their potential for bias, which could compromise patient care and exacerbate health inequities. This systematic review investigates the prevalence, sources, manifestations, and clinical implications of bias in LLMs. Methods: We conducted a systematic search of PubMed, OVID, and EMBASE from database inception through 2025, for studies evaluating bias in LLMs applied to clinical tasks. We extracted data on LLM type, bias source, bias manifestation, affected attributes, clinical task, evaluation methods, and outcomes. Risk of bias was assessed using a modified ROBINS-I tool. Results: Thirty-eight studies met inclusion criteria, revealing pervasive bias across various LLMs and clinical applications. Both data-related bias (from biased training data) and model-related bias (from model training) were significant contributors. Biases manifested as: allocative harm (e.g., differential treatment recommendations); representational harm (e.g., stereotypical associations, biased image generation); and performance disparities (e.g., variable output quality). These biases affected multiple attributes, most frequently race/ethnicity and gender, but also age, disability, and language. Conclusions: Bias in clinical LLMs is a pervasive and systemic issue, with a potential to lead to misdiagnosis and inappropriate treatment, particularly for marginalized patient populations. Rigorous evaluation of the model is crucial. Furthermore, the development and implementation of effective mitigation strategies, coupled with continuous monitoring in real-world clinical settings, are essential to ensure the safe, equitable, and trustworthy deployment of LLMs in healthcare. 

---
# Haphazard Inputs as Images in Online Learning 

**Authors**: Rohit Agarwal, Aryan Dessai, Arif Ahmed Sekh, Krishna Agarwal, Alexander Horsch, Dilip K. Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2504.02912)  

**Abstract**: The field of varying feature space in online learning settings, also known as haphazard inputs, is very prominent nowadays due to its applicability in various fields. However, the current solutions to haphazard inputs are model-dependent and cannot benefit from the existing advanced deep-learning methods, which necessitate inputs of fixed dimensions. Therefore, we propose to transform the varying feature space in an online learning setting to a fixed-dimension image representation on the fly. This simple yet novel approach is model-agnostic, allowing any vision-based models to be applicable for haphazard inputs, as demonstrated using ResNet and ViT. The image representation handles the inconsistent input data seamlessly, making our proposed approach scalable and robust. We show the efficacy of our method on four publicly available datasets. The code is available at this https URL. 

---
# Noiser: Bounded Input Perturbations for Attributing Large Language Models 

**Authors**: Mohammad Reza Ghasemi Madani, Aryo Pradipta Gema, Gabriele Sarti, Yu Zhao, Pasquale Minervini, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2504.02911)  

**Abstract**: Feature attribution (FA) methods are common post-hoc approaches that explain how Large Language Models (LLMs) make predictions. Accordingly, generating faithful attributions that reflect the actual inner behavior of the model is crucial. In this paper, we introduce Noiser, a perturbation-based FA method that imposes bounded noise on each input embedding and measures the robustness of the model against partially noised input to obtain the input attributions. Additionally, we propose an answerability metric that employs an instructed judge model to assess the extent to which highly scored tokens suffice to recover the predicted output. Through a comprehensive evaluation across six LLMs and three tasks, we demonstrate that Noiser consistently outperforms existing gradient-based, attention-based, and perturbation-based FA methods in terms of both faithfulness and answerability, making it a robust and effective approach for explaining language model predictions. 

---
# Systematic Literature Review: Explainable AI Definitions and Challenges in Education 

**Authors**: Zaid M. Altukhi, Sojen Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02910)  

**Abstract**: Explainable AI (XAI) seeks to transform black-box algorithmic processes into transparent ones, enhancing trust in AI applications across various sectors such as education. This review aims to examine the various definitions of XAI within the literature and explore the challenges of XAI in education. Our goal is to shed light on how XAI can contribute to enhancing the educational field. This systematic review, utilising the PRISMA method for rigorous and transparent research, identified 19 relevant studies. Our findings reveal 15 definitions and 62 challenges. These challenges are categorised using thematic analysis into seven groups: explainability, ethical, technical, human-computer interaction (HCI), trustworthiness, policy and guideline, and others, thereby deepening our understanding of the implications of XAI in education. Our analysis highlights the absence of standardised definitions for XAI, leading to confusion, especially because definitions concerning ethics, trustworthiness, technicalities, and explainability tend to overlap and vary. 

---
# Enhancing Chart-to-Code Generation in Multimodal Large Language Models via Iterative Dual Preference Learning 

**Authors**: Zhihan Zhang, Yixin Cao, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02906)  

**Abstract**: Chart-to-code generation, the process of converting chart images into executable plotting scripts, provides a lossless representation of chart information, requiring models to accurately capture and summarize all visual and structural elements. However, this remains a significant challenge for multimodal large language models (MLLMs), which are not inherently well-aligned with code generation tasks. To bridge this gap, we introduce Chart2Code, a novel iterative dual preference learning framework designed to enhance MLLMs' chart-to-code generation capabilities through structured code variant generation and fine-grained dual reward signals. We validate Chart2Code across three MLLMs and find that iterative preference learning consistently improves out-of-distribution chart-to-code generation quality. Throughout this process, our dual scoring method, which evaluates both the textual code structure and its visual representation, leads to greater performance improvements, even with a reduced preference dataset size. Further analysis explores the key components of our framework and highlights the interplay between chart-to-code generation and broader chart reasoning, paving the way for future advancements in chart comprehension. 

---
# How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence 

**Authors**: Hongzhe Du, Weikai Li, Min Cai, Karim Saraipour, Zimin Zhang, Himabindu Lakkaraju, Yizhou Sun, Shichang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02904)  

**Abstract**: Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by linear vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training. 

---
# Beyond Accuracy: The Role of Calibration in Self-Improving Large Language Models 

**Authors**: Liangjie Huang, Dawei Li, Huan Liu, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.02902)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable self-improvement capabilities, whereby models iteratively revise their outputs through self-generated feedback. While this reflective mechanism has shown promise in enhancing task performance, recent studies suggest that it may also introduce undesirable biases-most notably, self-bias, or the tendency of LLMs to favor their own prior outputs. In this work, we extend this line of inquiry by investigating the impact on confidence estimation. We evaluate three representative self-improvement paradigms-basic prompting, Chain-of-Thought (CoT) prompting, and tuning-based methods and find that iterative self-improvement can lead to systematic overconfidence, as evidenced by a steadily increasing Expected Calibration Error (ECE) and lower accuracy with high confidence. We then further explore the integration of confidence calibration techniques with self-improvement. Specifically, we compare three strategies: (1) applying calibration after multiple rounds of self-improvement, (2) calibrating before self-improvement, and (3) applying calibration iteratively at each self-improvement step. Our results show that iterative calibration is most effective in reducing ECE, yielding improved calibration. Our work pioneers the study of self-improving LLMs from a calibration perspective, offering valuable insights into balancing model performance and reliability. 

---
# Hide and Seek in Noise Labels: Noise-Robust Collaborative Active Learning with LLM-Powered Assistance 

**Authors**: Bo Yuan, Yulin Chen, Yin Zhang, Wei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02901)  

**Abstract**: Learning from noisy labels (LNL) is a challenge that arises in many real-world scenarios where collected training data can contain incorrect or corrupted labels. Most existing solutions identify noisy labels and adopt active learning to query human experts on them for denoising. In the era of large language models (LLMs), although we can reduce the human effort to improve these methods, their performances are still subject to accurately separating the clean and noisy samples from noisy data. In this paper, we propose an innovative collaborative learning framework NoiseAL based on active learning to combine LLMs and small models (SMs) for learning from noisy labels. During collaborative training, we first adopt two SMs to form a co-prediction network and propose a dynamic-enhanced threshold strategy to divide the noisy data into different subsets, then select the clean and noisy samples from these subsets to feed the active annotator LLMs to rectify noisy samples. Finally, we employ different optimization objectives to conquer subsets with different degrees of label noises. Extensive experiments on synthetic and real-world noise datasets further demonstrate the superiority of our framework over state-of-the-art baselines. 

---
# Meat-Free Day Reduces Greenhouse Gas Emissions but Poses Challenges for Customer Retention and Adherence to Dietary Guidelines 

**Authors**: Giuseppe Russo, Kristina Gligorić, Vincent Moreau, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2504.02899)  

**Abstract**: Reducing meat consumption is crucial for achieving global environmental and nutritional targets. Meat-Free Day (MFD) is a widely adopted strategy to address this challenge by encouraging plant-based diets through the removal of animal-based meals. We assessed the environmental, behavioral, and nutritional impacts of MFD by implementing 67 MFDs over 18 months (once a week on a randomly chosen day) across 12 cafeterias on a large university campus, analyzing over 400,000 food purchases. MFD reduced on-campus food-related greenhouse gas (GHG) emissions on treated days by 52.9% and contributed to improved fiber (+26.9%) and cholesterol (-4.5%) consumption without altering caloric intake. These nutritional benefits were, however, accompanied by a 27.6% decrease in protein intake and a 34.2% increase in sugar consumption. Moreover, the increase in plant-based meals did not carry over to subsequent days, as evidenced by a 3.5% rebound in animal-based meal consumption on days immediately following treated days. MFD also led to a 16.8% drop in on-campus meal sales on treated this http URL Carlo simulations suggest that if 8.7% of diners were to eat burgers off-campus on treated days, MFD's GHG savings would be fully negated. As our analysis identifies on-campus customer retention as the main challenge to MFD effectiveness, we recommend combining MFD with customer retention interventions to ensure environmental and nutritional benefits. 

---
# UAC: Uncertainty-Aware Calibration of Neural Networks for Gesture Detection 

**Authors**: Farida Al Haddad, Yuxin Wang, Malcolm Mielle  

**Link**: [PDF](https://arxiv.org/pdf/2504.02895)  

**Abstract**: Artificial intelligence has the potential to impact safety and efficiency in safety-critical domains such as construction, manufacturing, and healthcare. For example, using sensor data from wearable devices, such as inertial measurement units (IMUs), human gestures can be detected while maintaining privacy, thereby ensuring that safety protocols are followed. However, strict safety requirements in these domains have limited the adoption of AI, since accurate calibration of predicted probabilities and robustness against out-of-distribution (OOD) data is necessary.
This paper proposes UAC (Uncertainty-Aware Calibration), a novel two-step method to address these challenges in IMU-based gesture recognition. First, we present an uncertainty-aware gesture network architecture that predicts both gesture probabilities and their associated uncertainties from IMU data. This uncertainty is then used to calibrate the probabilities of each potential gesture. Second, an entropy-weighted expectation of predictions over multiple IMU data windows is used to improve accuracy while maintaining correct calibration.
Our method is evaluated using three publicly available IMU datasets for gesture detection and is compared to three state-of-the-art calibration methods for neural networks: temperature scaling, entropy maximization, and Laplace approximation. UAC outperforms existing methods, achieving improved accuracy and calibration in both OOD and in-distribution scenarios. Moreover, we find that, unlike our method, none of the state-of-the-art methods significantly improve the calibration of IMU-based gesture recognition models. In conclusion, our work highlights the advantages of uncertainty-aware calibration of neural networks, demonstrating improvements in both calibration and accuracy for gesture detection using IMU data. 

---
# OnRL-RAG: Real-Time Personalized Mental Health Dialogue System 

**Authors**: Ahsan Bilal, Beiyu Lin, Mehdi Zaeifi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02894)  

**Abstract**: Large language models (LLMs) have been widely used for various tasks and applications. However, LLMs and fine-tuning are limited to the pre-trained data. For example, ChatGPT's world knowledge until 2021 can be outdated or inaccurate. To enhance the capabilities of LLMs, Retrieval-Augmented Generation (RAG), is proposed to augment LLMs with additional, new, latest details and information to LLMs. While RAG offers the correct information, it may not best present it, especially to different population groups with personalizations. Reinforcement Learning from Human Feedback (RLHF) adapts to user needs by aligning model responses with human preference through feedback loops. In real-life applications, such as mental health problems, a dynamic and feedback-based model would continuously adapt to new information and offer personalized assistance due to complex factors fluctuating in a daily environment. Thus, we propose an Online Reinforcement Learning-based Retrieval-Augmented Generation (OnRL-RAG) system to detect and personalize the responding systems to mental health problems, such as stress, anxiety, and depression. We use an open-source dataset collected from 2028 College Students with 28 survey questions for each student to demonstrate the performance of our proposed system with the existing systems. Our system achieves superior performance compared to standard RAG and simple LLM via GPT-4o, GPT-4o-mini, Gemini-1.5, and GPT-3.5. This work would open up the possibilities of real-life applications of LLMs for personalized services in the everyday environment. The results will also help researchers in the fields of sociology, psychology, and neuroscience to align their theories more closely with the actual human daily environment. 

---
# Automated Survey Collection with LLM-based Conversational Agents 

**Authors**: Kurmanbek Kaiyrbekov, Nicholas J Dobbins, Sean D Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2504.02891)  

**Abstract**: Objective: Traditional phone-based surveys are among the most accessible and widely used methods to collect biomedical and healthcare data, however, they are often costly, labor intensive, and difficult to scale effectively. To overcome these limitations, we propose an end-to-end survey collection framework driven by conversational Large Language Models (LLMs).
Materials and Methods: Our framework consists of a researcher responsible for designing the survey and recruiting participants, a conversational phone agent powered by an LLM that calls participants and administers the survey, a second LLM (GPT-4o) that analyzes the conversation transcripts generated during the surveys, and a database for storing and organizing the results. To test our framework, we recruited 8 participants consisting of 5 native and 3 non-native english speakers and administered 40 surveys. We evaluated the correctness of LLM-generated conversation transcripts, accuracy of survey responses inferred by GPT-4o and overall participant experience.
Results: Survey responses were successfully extracted by GPT-4o from conversation transcripts with an average accuracy of 98% despite transcripts exhibiting an average per-line word error rate of 7.7%. While participants noted occasional errors made by the conversational LLM agent, they reported that the agent effectively conveyed the purpose of the survey, demonstrated good comprehension, and maintained an engaging interaction.
Conclusions: Our study highlights the potential of LLM agents in conducting and analyzing phone surveys for healthcare applications. By reducing the workload on human interviewers and offering a scalable solution, this approach paves the way for real-world, end-to-end AI-powered phone survey collection systems. 

---
# Scaling Test-time Compute for Low-resource Languages: Multilingual Reasoning in LLMs 

**Authors**: Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02890)  

**Abstract**: Recent advances in test-time compute scaling have enabled Large Language Models (LLMs) to tackle deep reasoning tasks by generating a chain-of-thought (CoT) that includes trial and error, backtracking, and intermediate reasoning steps before producing the final answer. However, these techniques have been applied predominantly to popular languages, such as English, leaving reasoning in low-resource languages underexplored and misaligned. In this work, we investigate the multilingual mechanism by which LLMs internally operate in a latent space biased toward their inherently dominant language. To leverage this phenomenon for low-resource languages, we train models to generate the CoT in English while outputting the final response in the target language, given input in the low-resource language. Our experiments demonstrate that this approach, named English-Pivoted CoT Training, outperforms other baselines, including training to generate both the CoT and the final response solely in the target language, with up to 28.33% improvement. Further analysis provides novel insights into the relationships between reasoning and multilinguality of LLMs, prompting for better approaches in developing multilingual large reasoning models 

---
# Embedding Method for Knowledge Graph with Densely Defined Ontology 

**Authors**: Takanori Ugai  

**Link**: [PDF](https://arxiv.org/pdf/2504.02889)  

**Abstract**: Knowledge graph embedding (KGE) is a technique that enhances knowledge graphs by addressing incompleteness and improving knowledge retrieval. A limitation of the existing KGE models is their underutilization of ontologies, specifically the relationships between properties. This study proposes a KGE model, TransU, designed for knowledge graphs with well-defined ontologies that incorporate relationships between properties. The model treats properties as a subset of entities, enabling a unified representation. We present experimental results using a standard dataset and a practical dataset. 

---
# Global Rice Multi-Class Segmentation Dataset (RiceSEG): A Comprehensive and Diverse High-Resolution RGB-Annotated Images for the Development and Benchmarking of Rice Segmentation Algorithms 

**Authors**: Junchi Zhou, Haozhou Wang, Yoichiro Kato, Tejasri Nampally, P. Rajalakshmi, M. Balram, Keisuke Katsura, Hao Lu, Yue Mu, Wanneng Yang, Yangmingrui Gao, Feng Xiao, Hongtao Chen, Yuhao Chen, Wenjuan Li, Jingwen Wang, Fenghua Yu, Jian Zhou, Wensheng Wang, Xiaochun Hu, Yuanzhu Yang, Yanfeng Ding, Wei Guo, Shouyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02880)  

**Abstract**: Developing computer vision-based rice phenotyping techniques is crucial for precision field management and accelerating breeding, thereby continuously advancing rice production. Among phenotyping tasks, distinguishing image components is a key prerequisite for characterizing plant growth and development at the organ scale, enabling deeper insights into eco-physiological processes. However, due to the fine structure of rice organs and complex illumination within the canopy, this task remains highly challenging, underscoring the need for a high-quality training dataset. Such datasets are scarce, both due to a lack of large, representative collections of rice field images and the time-intensive nature of annotation. To address this gap, we established the first comprehensive multi-class rice semantic segmentation dataset, RiceSEG. We gathered nearly 50,000 high-resolution, ground-based images from five major rice-growing countries (China, Japan, India, the Philippines, and Tanzania), encompassing over 6,000 genotypes across all growth stages. From these original images, 3,078 representative samples were selected and annotated with six classes (background, green vegetation, senescent vegetation, panicle, weeds, and duckweed) to form the RiceSEG dataset. Notably, the sub-dataset from China spans all major genotypes and rice-growing environments from the northeast to the south. Both state-of-the-art convolutional neural networks and transformer-based semantic segmentation models were used as baselines. While these models perform reasonably well in segmenting background and green vegetation, they face difficulties during the reproductive stage, when canopy structures are more complex and multiple classes are involved. These findings highlight the importance of our dataset for developing specialized segmentation models for rice and other crops. 

---
# Scraping the Shadows: Deep Learning Breakthroughs in Dark Web Intelligence 

**Authors**: Ingmar Bakermans, Daniel De Pascale, Gonçalo Marcelino, Giuseppe Cascavilla, Zeno Geradts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02872)  

**Abstract**: Darknet markets (DNMs) facilitate the trade of illegal goods on a global scale. Gathering data on DNMs is critical to ensuring law enforcement agencies can effectively combat crime. Manually extracting data from DNMs is an error-prone and time-consuming task. Aiming to automate this process we develop a framework for extracting data from DNMs and evaluate the application of three state-of-the-art Named Entity Recognition (NER) models, ELMo-BiLSTM \citep{ShahEtAl2022}, UniversalNER \citep{ZhouEtAl2024}, and GLiNER \citep{ZaratianaEtAl2023}, at the task of extracting complex entities from DNM product listing pages. We propose a new annotated dataset, which we use to train, fine-tune, and evaluate the models. Our findings show that state-of-the-art NER models perform well in information extraction from DNMs, achieving 91% Precision, 96% Recall, and an F1 score of 94%. In addition, fine-tuning enhances model performance, with UniversalNER achieving the best performance. 

---
# Synthesized Annotation Guidelines are Knowledge-Lite Boosters for Clinical Information Extraction 

**Authors**: Enshuo Hsu, Martin Ugbala, Krishna Kumar Kookal, Zouaidi Kawtar, Nicholas L. Rider, Muhammad F. Walji, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02871)  

**Abstract**: Generative information extraction using large language models, particularly through few-shot learning, has become a popular method. Recent studies indicate that providing a detailed, human-readable guideline-similar to the annotation guidelines traditionally used for training human annotators can significantly improve performance. However, constructing these guidelines is both labor- and knowledge-intensive. Additionally, the definitions are often tailored to meet specific needs, making them highly task-specific and often non-reusable. Handling these subtle differences requires considerable effort and attention to detail. In this study, we propose a self-improving method that harvests the knowledge summarization and text generation capacity of LLMs to synthesize annotation guidelines while requiring virtually no human input. Our zero-shot experiments on the clinical named entity recognition benchmarks, 2012 i2b2 EVENT, 2012 i2b2 TIMEX, 2014 i2b2, and 2018 n2c2 showed 25.86%, 4.36%, 0.20%, and 7.75% improvements in strict F1 scores from the no-guideline baseline. The LLM-synthesized guidelines showed equivalent or better performance compared to human-written guidelines by 1.15% to 4.14% in most tasks. In conclusion, this study proposes a novel LLM self-improving method that requires minimal knowledge and human input and is applicable to multiple biomedical domains. 

---
# AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening 

**Authors**: Frank P.-W. Lo, Jianing Qiu, Zeyu Wang, Haibao Yu, Yeming Chen, Gao Zhang, Benny Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02870)  

**Abstract**: Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows. 

---
# Multi-Agent LLM Judge: automatic personalized LLM judge design for evaluating natural language generation applications 

**Authors**: Hongliu Cao, Ilias Driouich, Robin Singh, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02867)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across diverse domains, yet they still encounter challenges such as insufficient domain-specific knowledge, biases, and hallucinations. This underscores the need for robust evaluation methodologies to accurately assess LLM-based applications. Traditional evaluation methods, which rely on word overlap or text embeddings, are inadequate for capturing the nuanced semantic information necessary to evaluate dynamic, open-ended text generation. Recent research has explored leveraging LLMs to mimic human reasoning and decision-making processes for evaluation purposes known as LLM-as-a-judge framework. However, these existing frameworks have two significant limitations. First, they lack the flexibility to adapt to different text styles, including various answer and ground truth styles, thereby reducing their generalization performance. Second, the evaluation scores produced by these frameworks are often skewed and hard to interpret, showing a low correlation with human judgment. To address these challenges, we propose a novel dynamic multi-agent system that automatically designs personalized LLM judges for various natural language generation applications. This system iteratively refines evaluation prompts and balances the trade-off between the adaptive requirements of downstream tasks and the alignment with human perception. Our experimental results show that the proposed multi-agent LLM Judge framework not only enhances evaluation accuracy compared to existing methods but also produces evaluation scores that better align with human perception. 

---
# Computer Vision and Deep Learning for 4D Augmented Reality 

**Authors**: Karthik Shivashankar  

**Link**: [PDF](https://arxiv.org/pdf/2504.02860)  

**Abstract**: The prospect of 4D video in Extended Reality (XR) platform is huge and exciting, it opens a whole new way of human computer interaction and the way we perceive the reality and consume multimedia. In this thesis, we have shown that feasibility of rendering 4D video in Microsoft mixed reality platform. This enables us to port any 3D performance capture from CVSSP into XR product like the HoloLens device with relative ease. However, if the 3D model is too complex and is made up of millions of vertices, the data bandwidth required to port the model is a severe limitation with the current hardware and communication system. Therefore, in this project we have also developed a compact representation of both shape and appearance of the 4d video sequence using deep learning models to effectively learn the compact representation of 4D video sequence and reconstruct it without affecting the shape and appearance of the video sequence. 

---
# Exploration of Multi-Element Collaborative Research and Application for Modern Power System Based on Generative Large Models 

**Authors**: Lu Cheng, Qixiu Zhang, Beibei Xu, Zhiwei Huang, Cirun Zhang, Yanan Lyu, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02855)  

**Abstract**: The transition to intelligent, low-carbon power systems necessitates advanced optimization strategies for managing renewable energy integration, energy storage, and carbon emissions. Generative Large Models (GLMs) provide a data-driven approach to enhancing forecasting, scheduling, and market operations by processing multi-source data and capturing complex system dynamics. This paper explores the role of GLMs in optimizing load-side management, energy storage utilization, and electricity carbon, with a focus on Smart Wide-area Hybrid Energy Systems with Storage and Carbon (SGLSC). By leveraging spatiotemporal modeling and reinforcement learning, GLMs enable dynamic energy scheduling, improve grid stability, enhance carbon trading strategies, and strengthen resilience against extreme weather events. The proposed framework highlights the transformative potential of GLMs in achieving efficient, adaptive, and low-carbon power system operations. 

---
# Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks 

**Authors**: Mario Lino, Tobias Pfaff, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2504.02843)  

**Abstract**: Physical systems with complex unsteady dynamics, such as fluid flows, are often poorly represented by a single mean solution. For many practical applications, it is crucial to access the full distribution of possible states, from which relevant statistics (e.g., RMS and two-point correlations) can be derived. Here, we propose a graph-based latent diffusion (or alternatively, flow-matching) model that enables direct sampling of states from their equilibrium distribution, given a mesh discretization of the system and its physical parameters. This allows for the efficient computation of flow statistics without running long and expensive numerical simulations. The graph-based structure enables operations on unstructured meshes, which is critical for representing complex geometries with spatially localized high gradients, while latent-space diffusion modeling with a multi-scale GNN allows for efficient learning and inference of entire distributions of solutions. A key finding is that the proposed networks can accurately learn full distributions even when trained on incomplete data from relatively short simulations. We apply this method to a range of fluid dynamics tasks, such as predicting pressure distributions on 3D wing models in turbulent flow, demonstrating both accuracy and computational efficiency in challenging scenarios. The ability to directly sample accurate solutions, and capturing their diversity from short ground-truth simulations, is highly promising for complex scientific modeling tasks. 

---
# A First-Principles Based Risk Assessment Framework and the IEEE P3396 Standard 

**Authors**: Richard J. Tong, Marina Cortês, Jeanine A. DeFalco, Mark Underwood, Janusz Zalewski  

**Link**: [PDF](https://arxiv.org/pdf/2504.00091)  

**Abstract**: Generative Artificial Intelligence (AI) is enabling unprecedented automation in content creation and decision support, but it also raises novel risks. This paper presents a first-principles risk assessment framework underlying the IEEE P3396 Recommended Practice for AI Risk, Safety, Trustworthiness, and Responsibility. We distinguish between process risks (risks arising from how AI systems are built or operated) and outcome risks (risks manifest in the AI system's outputs and their real-world effects), arguing that generative AI governance should prioritize outcome risks. Central to our approach is an information-centric ontology that classifies AI-generated outputs into four fundamental categories: (1) Perception-level information, (2) Knowledge-level information, (3) Decision/Action plan information, and (4) Control tokens (access or resource directives). This classification allows systematic identification of harms and more precise attribution of responsibility to stakeholders (developers, deployers, users, regulators) based on the nature of the information produced. We illustrate how each information type entails distinct outcome risks (e.g. deception, misinformation, unsafe recommendations, security breaches) and requires tailored risk metrics and mitigations. By grounding the framework in the essence of information, human agency, and cognition, we align risk evaluation with how AI outputs influence human understanding and action. The result is a principled approach to AI risk that supports clear accountability and targeted safeguards, in contrast to broad application-based risk categorizations. We include example tables mapping information types to risks and responsibilities. This work aims to inform the IEEE P3396 Recommended Practice and broader AI governance with a rigorous, first-principles foundation for assessing generative AI risks while enabling responsible innovation. 

---
