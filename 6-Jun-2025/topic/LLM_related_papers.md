# Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04760)  

**Abstract**: Large Language Models (LLMs) have shown potential in generating hypothetical documents for query expansion, thereby enhancing information retrieval performance. However, the efficacy of this method is highly dependent on the quality of the generated documents, which often requires complex prompt strategies and the integration of advanced dense retrieval techniques. This can be both costly and computationally intensive. To mitigate these limitations, we explore the use of zero-shot LLM-based query expansion to improve sparse retrieval, particularly for learned sparse retrievers. We introduce a novel fusion ranking framework, Exp4Fuse, which enhances the performance of sparse retrievers through an indirect application of zero-shot LLM-based query expansion. Exp4Fuse operates by simultaneously considering two retrieval routes-one based on the original query and the other on the LLM-augmented query. It then generates two ranked lists using a sparse retriever and fuses them using a modified reciprocal rank fusion method. We conduct extensive evaluations of Exp4Fuse against leading LLM-based query expansion methods and advanced retrieval techniques on three MS MARCO-related datasets and seven low-resource datasets. Experimental results reveal that Exp4Fuse not only surpasses existing LLM-based query expansion methods in enhancing sparse retrievers but also, when combined with advanced sparse retrievers, achieves SOTA results on several benchmarks. This highlights the superior performance and effectiveness of Exp4Fuse in improving query expansion for sparse retrieval. 

---
# Reason-to-Recommend: Using Interaction-of-Thought Reasoning to Enhance LLM Recommendation 

**Authors**: Keyu Zhao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05069)  

**Abstract**: Driven by advances in Large Language Models (LLMs), integrating them into recommendation tasks has gained interest due to their strong semantic understanding and prompt flexibility. Prior work encoded user-item interactions or metadata into prompts for recommendations. In parallel, LLM reasoning, boosted by test-time scaling and reinforcement learning, has excelled in fields like mathematics and code, where reasoning traces and correctness signals are clear, enabling high performance and interpretability. However, directly applying these reasoning methods to recommendation is ineffective because user feedback is implicit and lacks reasoning supervision. To address this, we propose $\textbf{R2Rec}$, a reasoning-enhanced recommendation framework that samples interaction chains from the user-item graph and converts them into structured interaction-of-thoughts via a progressive masked prompting strategy, with each thought representing stepwise reasoning grounded in interaction context. This allows LLMs to simulate step-by-step decision-making based on implicit patterns. We design a two-stage training pipeline: supervised fine-tuning teaches basic reasoning from high-quality traces, and reinforcement learning refines reasoning via reward signals, alleviating sparse explicit supervision. Experiments on three real-world datasets show R2Rec outperforms classical and LLM-based baselines with an average $\textbf{10.48%}$ improvement in HitRatio@1 and $\textbf{131.81%}$ gain over the original LLM. Furthermore, the explicit reasoning chains enhance interpretability by revealing the decision process. Our code is available at: this https URL. 

---
# Search Arena: Analyzing Search-Augmented LLMs 

**Authors**: Mihran Miroyan, Tsung-Han Wu, Logan King, Tianle Li, Jiayi Pan, Xinyan Hu, Wei-Lin Chiang, Anastasios N. Angelopoulos, Trevor Darrell, Narges Norouzi, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2506.05334)  

**Abstract**: Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research in this direction. Our dataset and code are available at: this https URL. 

---
# On the Comprehensibility of Multi-structured Financial Documents using LLMs and Pre-processing Tools 

**Authors**: Shivani Upadhyay, Messiah Ataey, Shariyar Murtuza, Yifan Nie, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05182)  

**Abstract**: The proliferation of complex structured data in hybrid sources, such as PDF documents and web pages, presents unique challenges for current Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) in providing accurate answers. Despite the recent advancements of MLLMs, they still often falter when interpreting intricately structured information, such as nested tables and multi-dimensional plots, leading to hallucinations and erroneous outputs. This paper explores the capabilities of LLMs and MLLMs in understanding and answering questions from complex data structures found in PDF documents by leveraging industrial and open-source tools as part of a pre-processing pipeline. Our findings indicate that GPT-4o, a popular MLLM, achieves an accuracy of 56% on multi-structured documents when fed documents directly, and that integrating pre-processing tools raises the accuracy of LLMs to 61.3% for GPT-4o and 76% for GPT-4, and with lower overall cost. The code is publicly available at this https URL. 

---
# Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots 

**Authors**: Alex Pan, Mary-Anne Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.04907)  

**Abstract**: Large Language Models (LLMs), whilst great at extracting facts from text, struggle with nested narrative reasoning. Existing long context and multi-hop QA benchmarks inadequately test this, lacking realistic distractors or failing to decouple context length from reasoning complexity, masking a fundamental LLM limitation. We introduce Verbose ListOps, a novel benchmark that programmatically transposes ListOps computations into lengthy, coherent stories. This uniquely forces internal computation and state management of nested reasoning problems by withholding intermediate results, and offers fine-grained controls for both narrative size \emph{and} reasoning difficulty. Whilst benchmarks like LongReason (2025) advance approaches for synthetically expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints a specific LLM vulnerability: difficulty in state management for nested sub-reasoning amongst semantically-relevant, distracting narrative. Our experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse in performance on Verbose ListOps at modest (~10k token) narrative lengths, despite effortlessly solving raw ListOps equations. Addressing this failure is paramount for real-world text interpretation which requires identifying key reasoning points, tracking conceptual intermediate results, and filtering irrelevant information. Verbose ListOps, and its extensible generation framework thus enables targeted reasoning enhancements beyond mere context-window expansion; a critical step to automating the world's knowledge work. 

---
# CMIE: Combining MLLM Insights with External Evidence for Explainable Out-of-Context Misinformation Detection 

**Authors**: Fanxiao Li, Jiaying Wu, Canyuan He, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23449)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated impressive capabilities in visual reasoning and text generation. While previous studies have explored the application of MLLM for detecting out-of-context (OOC) misinformation, our empirical analysis reveals two persisting challenges of this paradigm. Evaluating the representative GPT-4o model on direct reasoning and evidence augmented reasoning, results indicate that MLLM struggle to capture the deeper relationships-specifically, cases in which the image and text are not directly connected but are associated through underlying semantic links. Moreover, noise in the evidence further impairs detection accuracy. To address these challenges, we propose CMIE, a novel OOC misinformation detection framework that incorporates a Coexistence Relationship Generation (CRG) strategy and an Association Scoring (AS) mechanism. CMIE identifies the underlying coexistence relationships between images and text, and selectively utilizes relevant evidence to enhance misinformation detection. Experimental results demonstrate that our approach outperforms existing methods. 

---
# GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04762)  

**Abstract**: Large language models (LLMs)-based query expansion for information retrieval augments queries with generated hypothetical documents with LLMs. However, its performance relies heavily on the scale of the language models (LMs), necessitating larger, more advanced LLMs. This approach is costly, computationally intensive, and often has limited accessibility. To address these limitations, we introduce GOLFer - Smaller LMs-Generated Documents Hallucination Filter & Combiner - a novel method leveraging smaller open-source LMs for query expansion. GOLFer comprises two modules: a hallucination filter and a documents combiner. The former detects and removes non-factual and inconsistent sentences in generated documents, a common issue with smaller LMs, while the latter combines the filtered content with the query using a weight vector to balance their influence. We evaluate GOLFer alongside dominant LLM-based query expansion methods on three web search and ten low-resource datasets. Experimental results demonstrate that GOLFer consistently outperforms other methods using smaller LMs, and maintains competitive performance against methods using large-size LLMs, demonstrating its effectiveness. 

---
# PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation 

**Authors**: Chenglong Ma, Ziqi Xu, Yongli Ren, Danula Hettiachchi, Jeffrey Chan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04551)  

**Abstract**: Traditional offline evaluation methods for recommender systems struggle to capture the complexity of modern platforms due to sparse behavioural signals, noisy data, and limited modelling of user personality traits. While simulation frameworks can generate synthetic data to address these gaps, existing methods fail to replicate behavioural diversity, limiting their effectiveness. To overcome these challenges, we propose the Personality-driven User Behaviour Simulator (PUB), an LLM-based simulation framework that integrates the Big Five personality traits to model personalised user behaviour. PUB dynamically infers user personality from behavioural logs (e.g., ratings, reviews) and item metadata, then generates synthetic interactions that preserve statistical fidelity to real-world data. Experiments on the Amazon review datasets show that logs generated by PUB closely align with real user behaviour and reveal meaningful associations between personality traits and recommendation outcomes. These results highlight the potential of the personality-driven simulator to advance recommender system evaluation, offering scalable, controllable, high-fidelity alternatives to resource-intensive real-world experiments. 

---
# LLM-First Search: Self-Guided Exploration of the Solution Space 

**Authors**: Nathan Herr, Tim Rocktäschel, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05213)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable improvements in reasoning and planning through increased test-time compute, often by framing problem-solving as a search process. While methods like Monte Carlo Tree Search (MCTS) have proven effective in some domains, their reliance on fixed exploration hyperparameters limits their adaptability across tasks of varying difficulty, rendering them impractical or expensive in certain settings. In this paper, we propose \textbf{LLM-First Search (LFS)}, a novel \textit{LLM Self-Guided Search} method that removes the need for pre-defined search strategies by empowering the LLM to autonomously control the search process via self-guided exploration. Rather than relying on external heuristics or hardcoded policies, the LLM evaluates whether to pursue the current search path or explore alternative branches based on its internal scoring mechanisms. This enables more flexible and context-sensitive reasoning without requiring manual tuning or task-specific adaptation. We evaluate LFS on Countdown and Sudoku against three classic widely-used search algorithms, Tree-of-Thoughts' Breadth First Search (ToT-BFS), Best First Search (BestFS), and MCTS, each of which have been used to achieve SotA results on a range of challenging reasoning tasks. We found that LFS (1) performs better on more challenging tasks without additional tuning, (2) is more computationally efficient compared to the other methods, especially when powered by a stronger model, (3) scales better with stronger models, due to its LLM-First design, and (4) scales better with increased compute budget. Our code is publicly available at \href{this https URL}{LLM-First-Search}. 

---
# When Thinking LLMs Lie: Unveiling the Strategic Deception in Representations of Reasoning Models 

**Authors**: Kai Wang, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04909)  

**Abstract**: The honesty of large language models (LLMs) is a critical alignment challenge, especially as advanced systems with chain-of-thought (CoT) reasoning may strategically deceive humans. Unlike traditional honesty issues on LLMs, which could be possibly explained as some kind of hallucination, those models' explicit thought paths enable us to study strategic deception--goal-driven, intentional misinformation where reasoning contradicts outputs. Using representation engineering, we systematically induce, detect, and control such deception in CoT-enabled LLMs, extracting "deception vectors" via Linear Artificial Tomography (LAT) for 89% detection accuracy. Through activation steering, we achieve a 40% success rate in eliciting context-appropriate deception without explicit prompts, unveiling the specific honesty-related issue of reasoning models and providing tools for trustworthy AI alignment. 

---
# LLMs for sensory-motor control: Combining in-context and iterative learning 

**Authors**: Jônata Tyska Carvalho, Stefano Nolfi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04867)  

**Abstract**: We propose a method that enables large language models (LLMs) to control embodied agents by directly mapping continuous observation vectors to continuous action vectors. Initially, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. In most cases, it successfully identifies optimal or high-performing solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment. 

---
# Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning 

**Authors**: Jiayu Wang, Yifei Ming, Zixuan Ke, Caiming Xiong, Shafiq Joty, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2506.04723)  

**Abstract**: Reinforcement learning (RL) has become the dominant paradigm for endowing language models with advanced reasoning capabilities. Despite the substantial empirical gains demonstrated by RL-based training methods like GRPO, a granular understanding of their advantages is still lacking. To address this gap, we introduce a fine-grained analytic framework to dissect the impact of RL on reasoning. Our framework specifically investigates key elements that have been hypothesized to benefit from RL training: (1) plan-following and execution, (2) problem decomposition, and (3) improved reasoning and knowledge utilization. Using this framework, we gain insights beyond mere accuracy. For instance, providing models with explicit step-by-step plans surprisingly degrades performance on the most challenging benchmarks, yet RL-tuned models exhibit greater robustness, experiencing markedly smaller performance drops than their base counterparts. This suggests that RL may not primarily enhance the execution of external plans but rather empower models to formulate and follow internal strategies better suited to their reasoning processes. Conversely, we observe that RL enhances the model's capacity to integrate provided knowledge into its reasoning process, leading to performance improvements across diverse tasks. We also study difficulty, showing improved training by developing new ways to exploit hard problems. Our findings lay a foundation for more principled training and evaluation of reasoning models. 

---
# Empowering Economic Simulation for Massively Multiplayer Online Games through Generative Agent-Based Modeling 

**Authors**: Bihan Xu, Shiwei Zhao, Runze Wu, Zhenya Huang, Jiawei Wang, Zhipeng Hu, Kai Wang, Haoyu Liu, Tangjie Lv, Le Li, Changjie Fan, Xin Tong, Jiangze Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.04699)  

**Abstract**: Within the domain of Massively Multiplayer Online (MMO) economy research, Agent-Based Modeling (ABM) has emerged as a robust tool for analyzing game economics, evolving from rule-based agents to decision-making agents enhanced by reinforcement learning. Nevertheless, existing works encounter significant challenges when attempting to emulate human-like economic activities among agents, particularly regarding agent reliability, sociability, and interpretability. In this study, we take a preliminary step in introducing a novel approach using Large Language Models (LLMs) in MMO economy simulation. Leveraging LLMs' role-playing proficiency, generative capacity, and reasoning aptitude, we design LLM-driven agents with human-like decision-making and adaptability. These agents are equipped with the abilities of role-playing, perception, memory, and reasoning, addressing the aforementioned challenges effectively. Simulation experiments focusing on in-game economic activities demonstrate that LLM-empowered agents can promote emergent phenomena like role specialization and price fluctuations in line with market rules. 

---
# E-bike agents: Large Language Model-Driven E-Bike Accident Analysis and Severity Prediction 

**Authors**: Zhichao Yang, Jiashu He, Mohammad B. Al-Khasawneh, Darshan Pandit, Cirillo Cinzia  

**Link**: [PDF](https://arxiv.org/pdf/2506.04654)  

**Abstract**: Electric bicycles (e-bikes) are rapidly increasing in use, raising safety concerns due to a rise in accident reports. However, e-bike incident reports often use unstructured narrative formats, which hinders quantitative safety analysis. This study introduces E-bike agents, a framework that uses large language models (LLM) powered agents to classify and extract safety variables from unstructured incident reports. Our framework consists of four LLM agents, handling data classification, information extraction, injury cause determination, and component linkage, to extract the key factors that could lead to E-bike accidents and cause varying severity levels. Furthermore, we used an ordered logit model to examine the relationship between the severity of the incident and the factors retrieved, such as gender, the type of cause, and environmental conditions. Our research shows that equipment issues are slightly more common than human-related ones, but human-related incidents are more often fatal. Specifically, pedals, tires, and brakes are frequent contributors to accidents. The model achieves a high weighted F1 score of 0.87 in classification accuracy, highlighting the potential of using LLMs to extract unstructured data in niche domains, such as transportation. Our method offers a scalable solution to improve e-bike safety analytics and provides actionable information for policy makers, designers, and regulators. 

---
# Agents of Change: Self-Evolving LLM Agents for Strategic Planning 

**Authors**: Nikolas Belle, Dakota Barnes, Alfonso Amayuelas, Ivan Bercovich, Xin Eric Wang, William Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04651)  

**Abstract**: Recent advances in LLMs have enabled their use as autonomous agents across a range of tasks, yet they continue to struggle with formulating and adhering to coherent long-term strategies. In this paper, we investigate whether LLM agents can self-improve when placed in environments that explicitly challenge their strategic planning abilities. Using the board game Settlers of Catan, accessed through the open-source Catanatron framework, we benchmark a progression of LLM-based agents, from a simple game-playing agent to systems capable of autonomously rewriting their own prompts and their player agent's code. We introduce a multi-agent architecture in which specialized roles (Analyzer, Researcher, Coder, and Player) collaborate to iteratively analyze gameplay, research new strategies, and modify the agent's logic or prompt. By comparing manually crafted agents to those evolved entirely by LLMs, we evaluate how effectively these systems can diagnose failure and adapt over time. Our results show that self-evolving agents, particularly when powered by models like Claude 3.7 and GPT-4o, outperform static baselines by autonomously adopting their strategies, passing along sample behavior to game-playing agents, and demonstrating adaptive reasoning over multiple iterations. 

---
# Mathematical Reasoning for Unmanned Aerial Vehicles: A RAG-Based Approach for Complex Arithmetic Reasoning 

**Authors**: Mehdi Azarafza, Mojtaba Nayyeri, Faezeh Pasandideh, Steffen Staab, Achim Rettberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.04998)  

**Abstract**: Autonomous UAV operation necessitates reliable mathematical reasoning for tasks such as trajectory planning and power management. While traditional flight control relies on hardcoded equations, recent Large Language Models (LLMs) offer potential for more flexible problem-solving but struggle with reliably selecting and applying correct mathematical formulations and executing precise multi-step arithmetic. We propose RAG-UAV, a retrieval-augmented generation framework designed to improve the mathematical reasoning of several LLMs (including GPT o1/Turbo, Llama-3.2/3.3, Mistral, and DeepSeek R1) in UAV-specific contexts by providing access to relevant domain literature. To conduct an initial assessment, we introduce the UAV-Math-Bench, a small problem set comprising 20 UAV-centric mathematical problems across four difficulty levels. Our experiments demonstrate that incorporating retrieval substantially increases exact answer accuracy (achieving up to 75% with o1), reduces instances of incorrect formulation selection (from 25% without RAG to 5% with RAG), decreases numerical errors, reducing Mean Squared Error (MSE) by orders of magnitude for the best-performing models. This pilot study indicates that RAG can enable general-purpose LLMs to function as more reliable tools for engineering analysis, although direct real-time flight control requires further investigation and validation on a larger scale. All benchmark data, question and answer are publicly available. 

---
# Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design 

**Authors**: Lin Sun, Weihong Lin, Jinzhu Wu, Yongfu Zhu, Xiaoqi Jian, Guangxiang Zhao, Change Jia, Linglin Zhang, Sai-er Hu, Yuhan Wu, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04734)  

**Abstract**: Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models. 

---
# Schema Generation for Large Knowledge Graphs Using Large Language Models 

**Authors**: Bohui Zhang, Yuan He, Lydia Pintscher, Albert Meroño Peñuela, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2506.04512)  

**Abstract**: Schemas are vital for ensuring data quality in the Semantic Web and natural language processing. Traditionally, their creation demands substantial involvement from knowledge engineers and domain experts. Leveraging the impressive capabilities of large language models (LLMs) in related tasks like ontology engineering, we explore automatic schema generation using LLMs. To bridge the resource gap, we introduce two datasets: YAGO Schema and Wikidata EntitySchema, along with evaluation metrics. The LLM-based pipelines effectively utilize local and global information from knowledge graphs (KGs) to generate validating schemas in Shape Expressions (ShEx). Experiments demonstrate LLMs' strong potential in producing high-quality ShEx schemas, paving the way for scalable, automated schema generation for large KGs. Furthermore, our benchmark introduces a new challenge for structured generation, pushing the limits of LLMs on syntactically rich formalisms. 

---
# "Don't Do That!": Guiding Embodied Systems through Large Language Model-based Constraint Generation 

**Authors**: Aladin Djuhera, Amin Seffo, Masataro Asai, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2506.04500)  

**Abstract**: Recent advancements in large language models (LLMs) have spurred interest in robotic navigation that incorporates complex spatial, mathematical, and conditional constraints from natural language into the planning problem. Such constraints can be informal yet highly complex, making it challenging to translate into a formal description that can be passed on to a planning algorithm. In this paper, we propose STPR, a constraint generation framework that uses LLMs to translate constraints (expressed as instructions on ``what not to do'') into executable Python functions. STPR leverages the LLM's strong coding capabilities to shift the problem description from language into structured and transparent code, thus circumventing complex reasoning and avoiding potential hallucinations. We show that these LLM-generated functions accurately describe even complex mathematical constraints, and apply them to point cloud representations with traditional search algorithms. Experiments in a simulated Gazebo environment show that STPR ensures full compliance across several constraints and scenarios, while having short runtimes. We also verify that STPR can be used with smaller, code-specific LLMs, making it applicable to a wide range of compact models at low inference cost. 

---
# Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning 

**Authors**: Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.05256)  

**Abstract**: Large reasoning models (LRMs) achieve higher performance on challenging reasoning tasks by generating more tokens at inference time, but this verbosity often wastes computation on easy problems. Existing solutions, including supervised finetuning on shorter traces, user-controlled budgets, or RL with uniform penalties, either require data curation, manual configuration, or treat all problems alike regardless of difficulty. We introduce Adaptive Length Penalty (ALP), a reinforcement learning objective tailoring generation length to per-prompt solve rate. During training, ALP monitors each prompt's online solve rate through multiple rollouts and adds a differentiable penalty whose magnitude scales inversely with that rate, so confident (easy) prompts incur a high cost for extra tokens while hard prompts remain unhindered. Posttraining DeepScaleR-1.5B with ALP cuts average token usage by 50\% without significantly dropping performance. Relative to fixed-budget and uniform penalty baselines, ALP redistributes its reduced budget more intelligently by cutting compute on easy prompts and reallocating saved tokens to difficult ones, delivering higher accuracy on the hardest problems with higher cost. 

---
# Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences 

**Authors**: Hadi Hosseini, Samarth Khanna, Ronak Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.04478)  

**Abstract**: The rise of Large Language Models (LLMs) has driven progress in reasoning tasks -- from program synthesis to scientific hypothesis generation -- yet their ability to handle ranked preferences and structured algorithms in combinatorial domains remains underexplored. We study matching markets, a core framework behind applications like resource allocation and ride-sharing, which require reconciling individual ranked preferences to ensure stable outcomes. We evaluate several state-of-the-art models on a hierarchy of preference-based reasoning tasks -- ranging from stable-matching generation to instability detection, instability resolution, and fine-grained preference queries -- to systematically expose their logical and algorithmic limitations in handling ranked inputs. Surprisingly, even top-performing models with advanced reasoning struggle to resolve instability in large markets, often failing to identify blocking pairs or execute algorithms iteratively. We further show that parameter-efficient fine-tuning (LoRA) significantly improves performance in small markets, but fails to bring about a similar improvement on large instances, suggesting the need for more sophisticated strategies to improve LLMs' reasoning with larger-context inputs. 

---
# A Graph-Retrieval-Augmented Generation Framework Enhances Decision-Making in the Circular Economy 

**Authors**: Yang Zhao, Chengxiao Dai, Dusit Niyato, Chuan Fu Tan, Keyi Xiang, Yueyang Wang, Zhiquan Yeo, Daren Tan Zong Loong, Jonathan Low Zhaozhi, Eugene H.Z. HO  

**Link**: [PDF](https://arxiv.org/pdf/2506.04252)  

**Abstract**: Large language models (LLMs) hold promise for sustainable manufacturing, but often hallucinate industrial codes and emission factors, undermining regulatory and investment decisions. We introduce CircuGraphRAG, a retrieval-augmented generation (RAG) framework that grounds LLMs outputs in a domain-specific knowledge graph for the circular economy. This graph connects 117,380 industrial and waste entities with classification codes and GWP100 emission data, enabling structured multi-hop reasoning. Natural language queries are translated into SPARQL and verified subgraphs are retrieved to ensure accuracy and traceability. Compared with Standalone LLMs and Naive RAG, CircuGraphRAG achieves superior performance in single-hop and multi-hop question answering, with ROUGE-L F1 scores up to 1.0, while baseline scores below 0.08. It also improves efficiency, halving the response time and reducing token usage by 16% in representative tasks. CircuGraphRAG provides fact-checked, regulatory-ready support for circular economy planning, advancing reliable, low-carbon resource decision making. 

---
# Language-Guided Multi-Agent Learning in Simulations: A Unified Framework and Evaluation 

**Authors**: Zhengyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04251)  

**Abstract**: This paper introduces LLM-MARL, a unified framework that incorporates large language models (LLMs) into multi-agent reinforcement learning (MARL) to enhance coordination, communication, and generalization in simulated game environments. The framework features three modular components of Coordinator, Communicator, and Memory, which dynamically generate subgoals, facilitate symbolic inter-agent messaging, and support episodic recall. Training combines PPO with a language-conditioned loss and LLM query gating. LLM-MARL is evaluated in Google Research Football, MAgent Battle, and StarCraft II. Results show consistent improvements over MAPPO and QMIX in win rate, coordination score, and zero-shot generalization. Ablation studies demonstrate that subgoal generation and language-based messaging each contribute significantly to performance gains. Qualitative analysis reveals emergent behaviors such as role specialization and communication-driven tactics. By bridging language modeling and policy learning, this work contributes to the design of intelligent, cooperative agents in interactive simulations. It offers a path forward for leveraging LLMs in multi-agent systems used for training, games, and human-AI collaboration. 

---
# Contextual Integrity in LLMs via Reasoning and Reinforcement Learning 

**Authors**: Guangchen Lan, Huseyin A. Inan, Sahar Abdelnabi, Janardhan Kulkarni, Lukas Wutschitz, Reza Shokri, Christopher G. Brinton, Robert Sim  

**Link**: [PDF](https://arxiv.org/pdf/2506.04245)  

**Abstract**: As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls. 

---
# CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective 

**Authors**: Jiayu Liu, Zhenya Huang, Wei Dai, Cheng Cheng, Jinze Wu, Jing Sha, Song Li, Qi Liu, Shijin Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04481)  

**Abstract**: Although large language models (LLMs) show promise in solving complex mathematical tasks, existing evaluation paradigms rely solely on a coarse measure of overall answer accuracy, which are insufficient for assessing their authentic capabilities. In this paper, we propose \textbf{CogMath}, which comprehensively assesses LLMs' mathematical abilities through the lens of human cognition. Specifically, inspired by psychological theories, CogMath formalizes human reasoning process into 3 stages: \emph{problem comprehension}, \emph{problem solving}, and \emph{solution summarization}. Within these stages, we investigate perspectives such as numerical calculation, knowledge, and counterfactuals, and design a total of 9 fine-grained evaluation dimensions. In each dimension, we develop an ``\emph{Inquiry}-\emph{Judge}-\emph{Reference}'' multi-agent system to generate inquiries that assess LLMs' mastery from this dimension. An LLM is considered to truly master a problem only when excelling in all inquiries from the 9 dimensions. By applying CogMath on three benchmarks, we reveal that the mathematical capabilities of 7 mainstream LLMs are overestimated by 30\%-40\%. Moreover, we locate their strengths and weaknesses across specific stages/dimensions, offering in-depth insights to further enhance their reasoning abilities. 

---
# Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay 

**Authors**: Yifan Sun, Jingyan Shen, Yibin Wang, Tianyu Chen, Zhendong Wang, Mingyuan Zhou, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05316)  

**Abstract**: Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism that reuses recent rollouts, lowering per-step computation while maintaining stable updates. Extensive experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 25% to 65% to reach the same level of performance as the original GRPO algorithm. 

---
# HADA: Human-AI Agent Decision Alignment Architecture 

**Authors**: Tapio Pitkäranta, Leena Pitkäranta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04253)  

**Abstract**: We present HADA (Human-AI Agent Decision Alignment), a protocol- and framework agnostic reference architecture that keeps both large language model (LLM) agents and legacy algorithms aligned with organizational targets and values. HADA wraps any algorithm or LLM in role-specific stakeholder agents -- business, data-science, audit, ethics, and customer -- each exposing conversational APIs so that technical and non-technical actors can query, steer, audit, or contest every decision across strategic, tactical, and real-time horizons. Alignment objectives, KPIs, and value constraints are expressed in natural language and are continuously propagated, logged, and versioned while thousands of heterogeneous agents run on different orchestration stacks. A cloud-native proof of concept packages a production credit-scoring model (getLoanDecision) and deploys it on Docker/Kubernetes/Python; five scripted retail-bank scenarios show how target changes, parameter tweaks, explanation requests, and ethics triggers flow end to end through the architecture. Evaluation followed the Design-Science Research Methodology. Walkthrough observation and log inspection demonstrated complete coverage of six predefined objectives: every role could invoke conversational control, trace KPIs and value constraints, detect and mitigate ZIP-code bias, and reproduce full decision lineage, independent of the underlying LLM or agent library. Contributions: (1) an open-source HADA architecture, (2) a mid-range design theory for human-AI alignment in multi-agent systems, and (3) empirical evidence that framework-agnostic, protocol-compliant stakeholder agents improve accuracy, transparency, and ethical compliance in real-world decision pipelines. 

---
# Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games 

**Authors**: Niv Eckhaus, Uri Berger, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.05309)  

**Abstract**: LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are inherently asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns; therefore, the decision of when to speak forms a crucial part of the participant's decision making. In this work, we develop an adaptive asynchronous LLM-agent which, in addition to determining what to say, also decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, including both human participants, as well as our asynchronous agent. Overall, our agent performs on par with human players, both in game performance, as well as in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We release all our data and code to support and encourage further research for more realistic asynchronous communication between LLM agents. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated. 

---
# A Statistical Physics of Language Model Reasoning 

**Authors**: Jack David Carson, Amir Reisizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.04374)  

**Abstract**: Transformer LMs show emergent reasoning that resists mechanistic understanding. We offer a statistical physics framework for continuous-time chain-of-thought reasoning dynamics. We model sentence-level hidden state trajectories as a stochastic dynamical system on a lower-dimensional manifold. This drift-diffusion system uses latent regime switching to capture diverse reasoning phases, including misaligned states or failures. Empirical trajectories (8 models, 7 benchmarks) show a rank-40 projection (balancing variance capture and feasibility) explains ~50% variance. We find four latent reasoning regimes. An SLDS model is formulated and validated to capture these features. The framework enables low-cost reasoning simulation, offering tools to study and predict critical transitions like misaligned states or other LM failures. 

---
# Sample Complexity and Representation Ability of Test-time Scaling Paradigms 

**Authors**: Baihe Huang, Shanda Li, Tianhao Wu, Yiming Yang, Ameet Talwalkar, Kannan Ramchandran, Michael I. Jordan, Jiantao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05295)  

**Abstract**: Test-time scaling paradigms have significantly advanced the capabilities of large language models (LLMs) on complex tasks. Despite their empirical success, theoretical understanding of the sample efficiency of various test-time strategies -- such as self-consistency, best-of-$n$, and self-correction -- remains limited. In this work, we first establish a separation result between two repeated sampling strategies: self-consistency requires $\Theta(1/\Delta^2)$ samples to produce the correct answer, while best-of-$n$ only needs $\Theta(1/\Delta)$, where $\Delta < 1$ denotes the probability gap between the correct and second most likely answers. Next, we present an expressiveness result for the self-correction approach with verifier feedback: it enables Transformers to simulate online learning over a pool of experts at test time. Therefore, a single Transformer architecture can provably solve multiple tasks without prior knowledge of the specific task associated with a user query, extending the representation theory of Transformers from single-task to multi-task settings. Finally, we empirically validate our theoretical results, demonstrating the practical effectiveness of self-correction methods. 

---
# Micro-Act: Mitigate Knowledge Conflict in Question Answering via Actionable Self-Reasoning 

**Authors**: Nan Huo, Jinyang Li, Bowen Qin, Ge Qu, Xiaolong Li, Xiaodong Li, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05278)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems commonly suffer from Knowledge Conflicts, where retrieved external knowledge contradicts the inherent, parametric knowledge of large language models (LLMs). It adversely affects performance on downstream tasks such as question answering (QA). Existing approaches often attempt to mitigate conflicts by directly comparing two knowledge sources in a side-by-side manner, but this can overwhelm LLMs with extraneous or lengthy contexts, ultimately hindering their ability to identify and mitigate inconsistencies. To address this issue, we propose Micro-Act a framework with a hierarchical action space that automatically perceives context complexity and adaptively decomposes each knowledge source into a sequence of fine-grained comparisons. These comparisons are represented as actionable steps, enabling reasoning beyond the superficial context. Through extensive experiments on five benchmark datasets, Micro-Act consistently achieves significant increase in QA accuracy over state-of-the-art baselines across all 5 datasets and 3 conflict types, especially in temporal and semantic types where all baselines fail significantly. More importantly, Micro-Act exhibits robust performance on non-conflict questions simultaneously, highlighting its practical value in real-world RAG applications. 

---
# Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning 

**Authors**: Xingjian Ran, Yixuan Li, Linning Xu, Mulin Yu, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.05341)  

**Abstract**: Realistic 3D indoor scene synthesis is vital for embodied AI and digital content creation. It can be naturally divided into two subtasks: object generation and layout generation. While recent generative models have significantly advanced object-level quality and controllability, layout generation remains challenging due to limited datasets. Existing methods either overfit to these datasets or rely on predefined constraints to optimize numerical layout that sacrifice flexibility. As a result, they fail to generate scenes that are both open-vocabulary and aligned with fine-grained user instructions. We introduce DirectLayout, a framework that directly generates numerical 3D layouts from text descriptions using generalizable spatial reasoning of large language models (LLMs). DirectLayout decomposes the generation into three stages: producing a Bird's-Eye View (BEV) layout, lifting it into 3D space, and refining object placements. To enable explicit spatial reasoning and help the model grasp basic principles of object placement, we employ Chain-of-Thought (CoT) Activation based on the 3D-Front dataset. Additionally, we design CoT-Grounded Generative Layout Reward to enhance generalization and spatial planning. During inference, DirectLayout addresses asset-layout mismatches via Iterative Asset-Layout Alignment through in-context learning. Extensive experiments demonstrate that DirectLayout achieves impressive semantic consistency, generalization and physical plausibility. 

---
# Counterfactual reasoning: an analysis of in-context emergence 

**Authors**: Moritz Miller, Bernhard Schölkopf, Siyuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05188)  

**Abstract**: Large-scale neural language models (LMs) exhibit remarkable performance in in-context learning: the ability to learn and reason the input context on the fly without parameter update. This work studies in-context counterfactual reasoning in language models, that is, to predict the consequences of changes under hypothetical scenarios. We focus on studying a well-defined synthetic setup: a linear regression task that requires noise abduction, where accurate prediction is based on inferring and copying the contextual noise from factual observations. We show that language models are capable of counterfactual reasoning in this controlled setup and provide insights that counterfactual reasoning for a broad class of functions can be reduced to a transformation on in-context observations; we find self-attention, model depth, and data diversity in pre-training drive performance in Transformers. More interestingly, our findings extend beyond regression tasks and show that Transformers can perform noise abduction on sequential data, providing preliminary evidence on the potential for counterfactual story generation. Our code is available under this https URL . 

---
# ProRefine: Inference-time Prompt Refinement with Textual Feedback 

**Authors**: Deepak Pandita, Tharindu Cyril Weerasooriya, Ankit Parag Shah, Christopher M. Homan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05305)  

**Abstract**: Agentic workflows, where multiple AI agents collaborate to accomplish complex tasks like reasoning or planning, are becoming increasingly prevalent. However, these workflows often suffer from error propagation and sub-optimal performance, largely due to poorly designed prompts that fail to effectively guide individual agents. This is a critical problem because it limits the reliability and scalability of these powerful systems. We introduce ProRefine, an innovative inference-time prompt optimization method that leverages textual feedback from large language models (LLMs) to address this challenge. ProRefine dynamically refines prompts for multi-step reasoning tasks without additional training or ground truth labels. Evaluated on five benchmark mathematical reasoning datasets, ProRefine significantly surpasses zero-shot Chain-of-Thought baselines by 3 to 37 percentage points. This approach not only boosts accuracy but also allows smaller models to match the performance of larger ones, highlighting its potential for efficient and scalable AI deployment, and democratizing access to high-performing AI. 

---
# Teaming in the AI Era: AI-Augmented Frameworks for Forming, Simulating, and Optimizing Human Teams 

**Authors**: Mohammed Almutairi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05265)  

**Abstract**: Effective teamwork is essential across diverse domains. During the team formation stage, a key challenge is forming teams that effectively balance user preferences with task objectives to enhance overall team satisfaction. In the team performing stage, maintaining cohesion and engagement is critical for sustaining high team performance. However, existing computational tools and algorithms for team optimization often rely on static data inputs, narrow algorithmic objectives, or solutions tailored for specific contexts, failing to account for the dynamic interplay of team members personalities, evolving goals, and changing individual preferences. Therefore, teams may encounter member dissatisfaction, as purely algorithmic assignments can reduce members commitment to team goals or experience suboptimal engagement due to the absence of timely, personalized guidance to help members adjust their behaviors and interactions as team dynamics evolve. Ultimately, these challenges can lead to reduced overall team performance. My Ph.D. dissertation aims to develop AI-augmented team optimization frameworks and practical systems that enhance team satisfaction, engagement, and performance. First, I propose a team formation framework that leverages a multi-armed bandit algorithm to iteratively refine team composition based on user preferences, ensuring alignment between individual needs and collective team goals to enhance team satisfaction. Second, I introduce tAIfa (Team AI Feedback Assistant), an AI-powered system that utilizes large language models (LLMs) to deliver immediate, personalized feedback to both teams and individual members, enhancing cohesion and engagement. Finally, I present PuppeteerLLM, an LLM-based simulation framework that simulates multi-agent teams to model complex team dynamics within realistic environments, incorporating task-driven collaboration and long-term coordination. 

---
# TreeRPO: Tree Relative Policy Optimization 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05183)  

**Abstract**: Large Language Models (LLMs) have shown remarkable reasoning capabilities through Reinforcement Learning with Verifiable Rewards (RLVR) methods. However, a key limitation of existing approaches is that rewards defined at the full trajectory level provide insufficient guidance for optimizing the intermediate steps of a reasoning process. To address this, we introduce \textbf{\name}, a novel method that estimates the mathematical expectations of rewards at various reasoning steps using tree sampling. Unlike prior methods that rely on a separate step reward model, \name directly estimates these rewards through this sampling process. Building on the group-relative reward training mechanism of GRPO, \name innovatively computes rewards based on step-level groups generated during tree sampling. This advancement allows \name to produce fine-grained and dense reward signals, significantly enhancing the learning process and overall performance of LLMs. Experimental results demonstrate that our \name algorithm substantially improves the average Pass@1 accuracy of Qwen-2.5-Math on test benchmarks, increasing it from 19.0\% to 35.5\%. Furthermore, \name significantly outperforms GRPO by 2.9\% in performance while simultaneously reducing the average response length by 18.1\%, showcasing its effectiveness and efficiency. Our code will be available at \href{this https URL}{this https URL}. 

---
# Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective 

**Authors**: Bhavik Chandna, Zubair Bashir, Procheta Sen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05166)  

**Abstract**: Large Language Models (LLMs) are known to exhibit social, demographic, and gender biases, often as a consequence of the data on which they are trained. In this work, we adopt a mechanistic interpretability approach to analyze how such biases are structurally represented within models such as GPT-2 and Llama2. Focusing on demographic and gender biases, we explore different metrics to identify the internal edges responsible for biased behavior. We then assess the stability, localization, and generalizability of these components across dataset and linguistic variations. Through systematic ablations, we demonstrate that bias-related computations are highly localized, often concentrated in a small subset of layers. Moreover, the identified components change across fine-tuning settings, including those unrelated to bias. Finally, we show that removing these components not only reduces biased outputs but also affects other NLP tasks, such as named entity recognition and linguistic acceptability judgment because of the sharing of important components with these tasks. 

---
# DiCoRe: Enhancing Zero-shot Event Detection via Divergent-Convergent LLM Reasoning 

**Authors**: Tanmay Parekh, Kartik Mehta, Ninareh Mehrabi, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05128)  

**Abstract**: Zero-shot Event Detection (ED), the task of identifying event mentions in natural language text without any training data, is critical for document understanding in specialized domains. Understanding the complex event ontology, extracting domain-specific triggers from the passage, and structuring them appropriately overloads and limits the utility of Large Language Models (LLMs) for zero-shot ED. To this end, we propose DiCoRe, a divergent-convergent reasoning framework that decouples the task of ED using Dreamer and Grounder. Dreamer encourages divergent reasoning through open-ended event discovery, which helps to boost event coverage. Conversely, Grounder introduces convergent reasoning to align the free-form predictions with the task-specific instructions using finite-state machine guided constrained decoding. Additionally, an LLM-Judge verifies the final outputs to ensure high precision. Through extensive experiments on six datasets across five domains and nine LLMs, we demonstrate how DiCoRe consistently outperforms prior zero-shot, transfer-learning, and reasoning baselines, achieving 4-7% average F1 gains over the best baseline -- establishing DiCoRe as a strong zero-shot ED framework. 

---
# TALL -- A Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages 

**Authors**: Moshe Ofer, Orel Zamler, Amos Azaria  

**Link**: [PDF](https://arxiv.org/pdf/2506.05057)  

**Abstract**: Large Language Models (LLMs) excel in high-resource languages but struggle with low-resource languages due to limited training data. This paper presents TALL (Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages), which integrates an LLM with two bilingual translation models. TALL transforms low-resource inputs into high-resource representations, leveraging the LLM's capabilities while preserving linguistic features through dimension alignment layers and custom transformers. Our experiments on Hebrew demonstrate significant improvements over several baselines, including direct use, naive translation, and fine-tuning approaches. The architecture employs a parameter-efficient strategy, freezing pre-trained components while training only lightweight adapter modules, balancing computational efficiency with performance gains. 

---
# Does It Make Sense to Speak of Introspection in Large Language Models? 

**Authors**: Iulia Comşa, Murray Shanahan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05068)  

**Abstract**: Large language models (LLMs) exhibit compelling linguistic behaviour, and sometimes offer self-reports, that is to say statements about their own nature, inner workings, or behaviour. In humans, such reports are often attributed to a faculty of introspection and are typically linked to consciousness. This raises the question of how to interpret self-reports produced by LLMs, given their increasing linguistic fluency and cognitive capabilities. To what extent (if any) can the concept of introspection be meaningfully applied to LLMs? Here, we present and critique two examples of apparent introspective self-report from LLMs. In the first example, an LLM attempts to describe the process behind its own ``creative'' writing, and we argue this is not a valid example of introspection. In the second example, an LLM correctly infers the value of its own temperature parameter, and we argue that this can be legitimately considered a minimal example of introspection, albeit one that is (presumably) not accompanied by conscious experience. 

---
# Hierarchical Language Models for Semantic Navigation and Manipulation in an Aerial-Ground Robotic System 

**Authors**: Haokun Liu, Zhaoqi Ma, Yunong Li, Junichiro Sugihara, Yicheng Chen, Jinjie Li, Moju Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05020)  

**Abstract**: Heterogeneous multi-robot systems show great potential in complex tasks requiring coordinated hybrid cooperation. However, traditional approaches relying on static models often struggle with task diversity and dynamic environments. This highlights the need for generalizable intelligence that can bridge high-level reasoning with low-level execution across heterogeneous agents. To address this, we propose a hierarchical framework integrating a prompted Large Language Model (LLM) and a GridMask-enhanced fine-tuned Vision Language Model (VLM). The LLM performs task decomposition and global semantic map construction, while the VLM extracts task-specified semantic labels and 2D spatial information from aerial images to support local planning. Within this framework, the aerial robot follows a globally optimized semantic path and continuously provides bird-view images, guiding the ground robot's local semantic navigation and manipulation, including target-absent scenarios where implicit alignment is maintained. Experiments on a real-world letter-cubes arrangement task demonstrate the framework's adaptability and robustness in dynamic environments. To the best of our knowledge, this is the first demonstration of an aerial-ground heterogeneous system integrating VLM-based perception with LLM-driven task reasoning and motion planning. 

---
# Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback 

**Authors**: Yongjin Yang, Sinjae Kang, Juyong Lee, Dongjun Lee, Se-Young Yun, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.04287)  

**Abstract**: Training large language model (LLM) agents to acquire necessary skills and perform diverse tasks within an environment is gaining interest as a means to enable open-endedness. However, creating the training dataset for their skill acquisition faces several challenges. Manual trajectory collection requires significant human effort. Another approach, where LLMs directly propose tasks to learn, is often invalid, as the LLMs lack knowledge of which tasks are actually feasible. Moreover, the generated data may not provide a meaningful learning signal, as agents often already perform well on the proposed tasks. To address this, we propose a novel automatic skill discovery framework EXIF for LLM-powered agents, designed to improve the feasibility of generated target behaviors while accounting for the agents' capabilities. Our method adopts an exploration-first strategy by employing an exploration agent (Alice) to train the target agent (Bob) to learn essential skills in the environment. Specifically, Alice first interacts with the environment to retrospectively generate a feasible, environment-grounded skill dataset, which is then used to train Bob. Crucially, we incorporate an iterative feedback loop, where Alice evaluates Bob's performance to identify areas for improvement. This feedback then guides Alice's next round of exploration, forming a closed-loop data generation process. Experiments on Webshop and Crafter demonstrate EXIF's ability to effectively discover meaningful skills and iteratively expand the capabilities of the trained agent without any human intervention, achieving substantial performance improvements. Interestingly, we observe that setting Alice to the same model as Bob also notably improves performance, demonstrating EXIF's potential for building a self-evolving system. 

---
# From Struggle (06-2024) to Mastery (02-2025) LLMs Conquer Advanced Algorithm Exams and Pave the Way for Editorial Generation 

**Authors**: Adrian Marius Dumitran, Theodor-Pierre Moroianu, Vasile Paul Alexe  

**Link**: [PDF](https://arxiv.org/pdf/2506.04965)  

**Abstract**: This paper presents a comprehensive evaluation of the performance of state-of-the-art Large Language Models (LLMs) on challenging university-level algorithms exams. By testing multiple models on both a Romanian exam and its high-quality English translation, we analyze LLMs' problem-solving capabilities, consistency, and multilingual performance. Our empirical study reveals that the most recent models not only achieve scores comparable to top-performing students but also demonstrate robust reasoning skills on complex, multi-step algorithmic challenges, even though difficulties remain with graph-based tasks. Building on these findings, we explore the potential of LLMs to support educational environments through the generation of high-quality editorial content, offering instructors a powerful tool to enhance student feedback. The insights and best practices discussed herein pave the way for further integration of generative AI in advanced algorithm education. 

---
# Simulating LLM-to-LLM Tutoring for Multilingual Math Feedback 

**Authors**: Junior Cedric Tonga, KV Aditya Srivatsa, Kaushal Kumar Maurya, Fajri Koto, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2506.04920)  

**Abstract**: Large language models (LLMs) have demonstrated the ability to generate formative feedback and instructional hints in English, making them increasingly relevant for AI-assisted education. However, their ability to provide effective instructional support across different languages, especially for mathematically grounded reasoning tasks, remains largely unexamined. In this work, we present the first large-scale simulation of multilingual tutor-student interactions using LLMs. A stronger model plays the role of the tutor, generating feedback in the form of hints, while a weaker model simulates the student. We explore 352 experimental settings across 11 typologically diverse languages, four state-of-the-art LLMs, and multiple prompting strategies to assess whether language-specific feedback leads to measurable learning gains. Our study examines how student input language, teacher feedback language, model choice, and language resource level jointly influence performance. Results show that multilingual hints can significantly improve learning outcomes, particularly in low-resource languages when feedback is aligned with the student's native language. These findings offer practical insights for developing multilingual, LLM-based educational tools that are both effective and inclusive. 

---
# Multiple-Choice Question Generation Using Large Language Models: Methodology and Educator Insights 

**Authors**: Giorgio Biancini, Alessio Ferrato, Carla Limongelli  

**Link**: [PDF](https://arxiv.org/pdf/2506.04851)  

**Abstract**: Integrating Artificial Intelligence (AI) in educational settings has brought new learning approaches, transforming the practices of both students and educators. Among the various technologies driving this transformation, Large Language Models (LLMs) have emerged as powerful tools for creating educational materials and question answering, but there are still space for new applications. Educators commonly use Multiple-Choice Questions (MCQs) to assess student knowledge, but manually generating these questions is resource-intensive and requires significant time and cognitive effort. In our opinion, LLMs offer a promising solution to these challenges. This paper presents a novel comparative analysis of three widely known LLMs - Llama 2, Mistral, and GPT-3.5 - to explore their potential for creating informative and challenging MCQs. In our approach, we do not rely on the knowledge of the LLM, but we inject the knowledge into the prompt to contrast the hallucinations, giving the educators control over the test's source text, too. Our experiment involving 21 educators shows that GPT-3.5 generates the most effective MCQs across several known metrics. Additionally, it shows that there is still some reluctance to adopt AI in the educational field. This study sheds light on the potential of LLMs to generate MCQs and improve the educational experience, providing valuable insights for the future. 

---
# On Automating Security Policies with Contemporary LLMs 

**Authors**: Pablo Fernández Saura, K. R. Jayaram, Vatche Isahagian, Jorge Bernal Bernabé, Antonio Skarmeta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04838)  

**Abstract**: The complexity of modern computing environments and the growing sophistication of cyber threats necessitate a more robust, adaptive, and automated approach to security enforcement. In this paper, we present a framework leveraging large language models (LLMs) for automating attack mitigation policy compliance through an innovative combination of in-context learning and retrieval-augmented generation (RAG). We begin by describing how our system collects and manages both tool and API specifications, storing them in a vector database to enable efficient retrieval of relevant information. We then detail the architectural pipeline that first decomposes high-level mitigation policies into discrete tasks and subsequently translates each task into a set of actionable API calls. Our empirical evaluation, conducted using publicly available CTI policies in STIXv2 format and Windows API documentation, demonstrates significant improvements in precision, recall, and F1-score when employing RAG compared to a non-RAG baseline. 

---
# Dissecting Logical Reasoning in LLMs: A Fine-Grained Evaluation and Supervision Study 

**Authors**: Yujun Zhou, Jiayi Ye, Zipeng Ling, Yufei Han, Yue Huang, Haomin Zhuang, Zhenwen Liang, Kehan Guo, Taicheng Guo, Xiangqi Wang, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04810)  

**Abstract**: Logical reasoning is a core capability for many applications of large language models (LLMs), yet existing benchmarks often rely solely on final-answer accuracy, failing to capture the quality and structure of the reasoning process. We propose FineLogic, a fine-grained evaluation framework that assesses logical reasoning across three dimensions: overall benchmark accuracy, stepwise soundness, and representation-level alignment. In addition, to better understand how reasoning capabilities emerge, we conduct a comprehensive study on the effects of supervision format during fine-tuning. We construct four supervision styles (one natural language and three symbolic variants) and train LLMs under each. Our findings reveal that natural language supervision yields strong generalization even on out-of-distribution and long-context tasks, while symbolic reasoning styles promote more structurally sound and atomic inference chains. Further, our representation-level probing shows that fine-tuning primarily improves reasoning behaviors through step-by-step generation, rather than enhancing shortcut prediction or internalized correctness. Together, our framework and analysis provide a more rigorous and interpretable lens for evaluating and improving logical reasoning in LLMs. 

---
# Fine-Grained Interpretation of Political Opinions in Large Language Models 

**Authors**: Jingyu Hu, Mengyue Yang, Mengnan Du, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04774)  

**Abstract**: Studies of LLMs' political opinions mainly rely on evaluations of their open-ended responses. Recent work indicates that there is a misalignment between LLMs' responses and their internal intentions. This motivates us to probe LLMs' internal mechanisms and help uncover their internal political states. Additionally, we found that the analysis of LLMs' political opinions often relies on single-axis concepts, which can lead to concept confounds. In this work, we extend the single-axis to multi-dimensions and apply interpretable representation engineering techniques for more transparent LLM political concept learning. Specifically, we designed a four-dimensional political learning framework and constructed a corresponding dataset for fine-grained political concept vector learning. These vectors can be used to detect and intervene in LLM internals. Experiments are conducted on eight open-source LLMs with three representation engineering techniques. Results show these vectors can disentangle political concept confounds. Detection tasks validate the semantic meaning of the vectors and show good generalization and robustness in OOD settings. Intervention Experiments show these vectors can intervene in LLMs to generate responses with different political leanings. 

---
# A Reasoning-Based Approach to Cryptic Crossword Clue Solving 

**Authors**: Martin Andrews, Sam Witteveen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04824)  

**Abstract**: Cryptic crossword clues are challenging language tasks for which new test sets are released daily by major newspapers on a global basis. Each cryptic clue contains both the definition of the answer to be placed in the crossword grid (in common with regular crosswords), and 'wordplay' that proves that the answer is correct (i.e. a human solver can be confident that an answer is correct without needing crossing words as confirmation). This work describes an LLM-based reasoning system built from open-licensed components that solves cryptic clues by (i) hypothesising answers; (ii) proposing wordplay explanations; and (iii) using a verifier system that operates on codified reasoning steps. Overall, this system establishes a new state-of-the-art performance on the challenging Cryptonite dataset of clues from The Times and The Telegraph newspapers in the UK. Because each proved solution is expressed in Python, interpretable wordplay reasoning for proven answers is available for inspection. 

---
# Towards LLM-Centric Multimodal Fusion: A Survey on Integration Strategies and Techniques 

**Authors**: Jisu An, Junseok Lee, Jeoungeun Lee, Yongseok Son  

**Link**: [PDF](https://arxiv.org/pdf/2506.04788)  

**Abstract**: The rapid progress of Multimodal Large Language Models(MLLMs) has transformed the AI landscape. These models combine pre-trained LLMs with various modality encoders. This integration requires a systematic understanding of how different modalities connect to the language backbone. Our survey presents an LLM-centric analysis of current approaches. We examine methods for transforming and aligning diverse modal inputs into the language embedding space. This addresses a significant gap in existing literature. We propose a classification framework for MLLMs based on three key dimensions. First, we examine architectural strategies for modality integration. This includes both the specific integration mechanisms and the fusion level. Second, we categorize representation learning techniques as either joint or coordinate representations. Third, we analyze training paradigms, including training strategies and objective functions. By examining 125 MLLMs developed between 2021 and 2025, we identify emerging patterns in the field. Our taxonomy provides researchers with a structured overview of current integration techniques. These insights aim to guide the development of more robust multimodal integration strategies for future models built on pre-trained foundations. 

---
# Truth in the Few: High-Value Data Selection for Efficient Multi-Modal Reasoning 

**Authors**: Shenshen Li, Kaiyuan Deng, Lei Wang, Hao Yang, Chong Peng, Peng Yan, Fumin Shen, Heng Tao Shen, Xing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04755)  

**Abstract**: While multi-modal large language models (MLLMs) have made significant progress in complex reasoning tasks via reinforcement learning, it is commonly believed that extensive training data is necessary for improving multi-modal reasoning ability, inevitably leading to data redundancy and substantial computational costs. However, can smaller high-value datasets match or outperform full corpora for multi-modal reasoning in MLLMs? In this work, we challenge this assumption through a key observation: meaningful multi-modal reasoning is triggered by only a sparse subset of training samples, termed cognitive samples, whereas the majority contribute marginally. Building on this insight, we propose a novel data selection paradigm termed Reasoning Activation Potential (RAP), which identifies cognitive samples by estimating each sample's potential to stimulate genuine multi-modal reasoning by two complementary estimators: 1) Causal Discrepancy Estimator (CDE) based on the potential outcome model principle, eliminates samples that overly rely on language priors by comparing outputs between multi-modal and text-only inputs; 2) Attention Confidence Estimator (ACE), which exploits token-level self-attention to discard samples dominated by irrelevant but over-emphasized tokens in intermediate reasoning stages. Moreover, we introduce a Difficulty-aware Replacement Module (DRM) to substitute trivial instances with cognitively challenging ones, thereby ensuring complexity for robust multi-modal reasoning. Experiments on six datasets show that our RAP method consistently achieves superior performance using only 9.3% of the training data, while reducing computational costs by over 43%. Our code is available at this https URL. 

---
# Lifelong Evolution: Collaborative Learning between Large and Small Language Models for Continuous Emergent Fake News Detection 

**Authors**: Ziyi Zhou, Xiaoming Zhang, Litian Zhang, Yibo Zhang, Zhenyu Guan, Chaozhuo Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04739)  

**Abstract**: The widespread dissemination of fake news on social media has significantly impacted society, resulting in serious consequences. Conventional deep learning methodologies employing small language models (SLMs) suffer from extensive supervised training requirements and difficulties adapting to evolving news environments due to data scarcity and distribution shifts. Large language models (LLMs), despite robust zero-shot capabilities, fall short in accurately detecting fake news owing to outdated knowledge and the absence of suitable demonstrations. In this paper, we propose a novel Continuous Collaborative Emergent Fake News Detection (C$^2$EFND) framework to address these challenges. The C$^2$EFND framework strategically leverages both LLMs' generalization power and SLMs' classification expertise via a multi-round collaborative learning framework. We further introduce a lifelong knowledge editing module based on a Mixture-of-Experts architecture to incrementally update LLMs and a replay-based continue learning method to ensure SLMs retain prior knowledge without retraining entirely. Extensive experiments on Pheme and Twitter16 datasets demonstrate that C$^2$EFND significantly outperforms existed methods, effectively improving detection accuracy and adaptability in continuous emergent fake news scenarios. 

---
# Urania: Differentially Private Insights into AI Use 

**Authors**: Daogao Liu, Edith Cohen, Badih Ghazi, Peter Kairouz, Pritish Kamath, Alexander Knop, Ravi Kumar, Pasin Manurangsi, Adam Sealfon, Da Yu, Chiyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04681)  

**Abstract**: We introduce $Urania$, a novel framework for generating insights about LLM chatbot interactions with rigorous differential privacy (DP) guarantees. The framework employs a private clustering mechanism and innovative keyword extraction methods, including frequency-based, TF-IDF-based, and LLM-guided approaches. By leveraging DP tools such as clustering, partition selection, and histogram-based summarization, $Urania$ provides end-to-end privacy protection. Our evaluation assesses lexical and semantic content preservation, pair similarity, and LLM-based metrics, benchmarking against a non-private Clio-inspired pipeline (Tamkin et al., 2024). Moreover, we develop a simple empirical privacy evaluation that demonstrates the enhanced robustness of our DP pipeline. The results show the framework's ability to extract meaningful conversational insights while maintaining stringent user privacy, effectively balancing data utility with privacy preservation. 

---
# MMRefine: Unveiling the Obstacles to Robust Refinement in Multimodal Large Language Models 

**Authors**: Gio Paik, Geewook Kim, Jinbae Im  

**Link**: [PDF](https://arxiv.org/pdf/2506.04688)  

**Abstract**: This paper introduces MMRefine, a MultiModal Refinement benchmark designed to evaluate the error refinement capabilities of Multimodal Large Language Models (MLLMs). As the emphasis shifts toward enhancing reasoning during inference, MMRefine provides a framework that evaluates MLLMs' abilities to detect and correct errors across six distinct scenarios beyond just comparing final accuracy before and after refinement. Furthermore, the benchmark analyzes the refinement performance by categorizing errors into six error types. Experiments with various open and closed MLLMs reveal bottlenecks and factors impeding refinement performance, highlighting areas for improvement in effective reasoning enhancement. Our code and dataset are publicly available at this https URL. 

---
# Gen-n-Val: Agentic Image Data Generation and Validation 

**Authors**: Jing-En Huang, I-Sheng Fang, Tzuhsuan Huang, Chih-Yu Wang, Jun-Cheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04676)  

**Abstract**: Recently, Large Language Models (LLMs) and Vision Large Language Models (VLLMs) have demonstrated impressive performance as agents across various tasks while data scarcity and label noise remain significant challenges in computer vision tasks, such as object detection and instance segmentation. A common solution for resolving these issues is to generate synthetic data. However, current synthetic data generation methods struggle with issues, such as multiple objects per mask, inaccurate segmentation, and incorrect category labels, limiting their effectiveness. To address these issues, we introduce Gen-n-Val, a novel agentic data generation framework that leverages Layer Diffusion (LD), LLMs, and VLLMs to produce high-quality, single-object masks and diverse backgrounds. Gen-n-Val consists of two agents: (1) The LD prompt agent, an LLM, optimizes prompts for LD to generate high-quality foreground instance images and segmentation masks. These optimized prompts ensure the generation of single-object synthetic data with precise instance masks and clean backgrounds. (2) The data validation agent, a VLLM, which filters out low-quality synthetic instance images. The system prompts for both agents are refined through TextGrad. Additionally, we use image harmonization to combine multiple instances within scenes. Compared to state-of-the-art synthetic data approaches like MosaicFusion, our approach reduces invalid synthetic data from 50% to 7% and improves performance by 1% mAP on rare classes in COCO instance segmentation with YOLOv9c and YOLO11m. Furthermore, Gen-n-Val shows significant improvements (7. 1% mAP) over YOLO-Worldv2-M in open-vocabulary object detection benchmarks with YOLO11m. Moreover, Gen-n-Val improves the performance of YOLOv9 and YOLO11 families in instance segmentation and object detection. 

---
# Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification 

**Authors**: Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04592)  

**Abstract**: Chain-of-Thought (CoT) prompting has become the de facto method to elicit reasoning capabilities from large language models (LLMs). However, to mitigate hallucinations in CoT that are notoriously difficult to detect, current methods such as process reward models (PRMs) or self-consistency operate as opaque boxes and do not provide checkable evidence for their judgments, possibly limiting their effectiveness. To address this issue, we draw inspiration from the idea that "the gold standard for supporting a mathematical claim is to provide a proof". We propose a retrospective, step-aware formal verification framework $Safe$. Rather than assigning arbitrary scores, we strive to articulate mathematical claims in formal mathematical language Lean 4 at each reasoning step and provide formal proofs to identify hallucinations. We evaluate our framework $Safe$ across multiple language models and various mathematical datasets, demonstrating a significant performance improvement while offering interpretable and verifiable evidence. We also propose $FormalStep$ as a benchmark for step correctness theorem proving with $30,809$ formal statements. To the best of our knowledge, our work represents the first endeavor to utilize formal mathematical language Lean 4 for verifying natural language content generated by LLMs, aligning with the reason why formal mathematical languages were created in the first place: to provide a robust foundation for hallucination-prone human-written proofs. 

---
# Towards Better Generalization via Distributional Input Projection Network 

**Authors**: Yifan Hao, Yanxin Lu, Xinwei Shen, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04690)  

**Abstract**: As overparameterized models become increasingly prevalent, training loss alone offers limited insight into generalization performance. While smoothness has been linked to improved generalization across various settings, directly enforcing smoothness in neural networks remains challenging. To address this, we introduce Distributional Input Projection Networks (DIPNet), a novel framework that projects inputs into learnable distributions at each layer. This distributional representation induces a smoother loss landscape with respect to the input, promoting better generalization. We provide theoretical analysis showing that DIPNet reduces both local smoothness measures and the Lipschitz constant of the network, contributing to improved generalization performance. Empirically, we validate DIPNet across a wide range of architectures and tasks, including Vision Transformers (ViTs), Large Language Models (LLMs), ResNet and MLPs. Our method consistently enhances test performance under standard settings, adversarial attacks, out-of-distribution inputs, and reasoning benchmarks. We demonstrate that the proposed input projection strategy can be seamlessly integrated into existing models, providing a general and effective approach for boosting generalization performance in modern deep learning. 

---
# hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation 

**Authors**: Charles Hong, Brendan Roberts, Huijae An, Alex Um, Advay Ratan, Yakun Sophia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04544)  

**Abstract**: Large language models (LLMs) are playing an increasingly large role in domains such as code generation, including hardware code generation, where Verilog is the key language. However, the amount of publicly available Verilog code pales in comparison to the amount of code available for software languages like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which seeks to increase the amount of available human-written Verilog data by translating or compiling three other hardware description languages - VHDL, Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v in enhancing LLM Verilog generation by improving performance of a 32 billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2, without utilizing any data augmentation or knowledge distillation from larger models. We also show hdl2v's ability to boost the performance of a data augmentation-based fine-tuning approach by 63%. Finally, we characterize and analyze our dataset to better understand which characteristics of HDL-to-Verilog datasets can be expanded upon in future work for even better performance. 

---
# BEAR: BGP Event Analysis and Reporting 

**Authors**: Hanqing Li, Melania Fedeli, Vinay Kolar, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04514)  

**Abstract**: The Internet comprises of interconnected, independently managed Autonomous Systems (AS) that rely on the Border Gateway Protocol (BGP) for inter-domain routing. BGP anomalies--such as route leaks and hijacks--can divert traffic through unauthorized or inefficient paths, jeopardizing network reliability and security. Although existing rule-based and machine learning methods can detect these anomalies using structured metrics, they still require experts with in-depth BGP knowledge of, for example, AS relationships and historical incidents, to interpret events and propose remediation. In this paper, we introduce BEAR (BGP Event Analysis and Reporting), a novel framework that leverages large language models (LLMs) to automatically generate comprehensive reports explaining detected BGP anomaly events. BEAR employs a multi-step reasoning process that translates tabular BGP data into detailed textual narratives, enhancing interpretability and analytical precision. To address the limited availability of publicly documented BGP anomalies, we also present a synthetic data generation framework powered by LLMs. Evaluations on both real and synthetic datasets demonstrate that BEAR achieves 100% accuracy, outperforming Chain-of-Thought and in-context learning baselines. This work pioneers an automated approach for explaining BGP anomaly events, offering valuable operational insights for network management. 

---
# Reasoning or Overthinking: Evaluating Large Language Models on Financial Sentiment Analysis 

**Authors**: Dimitris Vamvourellis, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04574)  

**Abstract**: We investigate the effectiveness of large language models (LLMs), including reasoning-based and non-reasoning models, in performing zero-shot financial sentiment analysis. Using the Financial PhraseBank dataset annotated by domain experts, we evaluate how various LLMs and prompting strategies align with human-labeled sentiment in a financial context. We compare three proprietary LLMs (GPT-4o, GPT-4.1, o3-mini) under different prompting paradigms that simulate System 1 (fast and intuitive) or System 2 (slow and deliberate) thinking and benchmark them against two smaller models (FinBERT-Prosus, FinBERT-Tone) fine-tuned on financial sentiment analysis. Our findings suggest that reasoning, either through prompting or inherent model design, does not improve performance on this task. Surprisingly, the most accurate and human-aligned combination of model and method was GPT-4o without any Chain-of-Thought (CoT) prompting. We further explore how performance is impacted by linguistic complexity and annotation agreement levels, uncovering that reasoning may introduce overthinking, leading to suboptimal predictions. This suggests that for financial sentiment classification, fast, intuitive "System 1"-like thinking aligns more closely with human judgment compared to "System 2"-style slower, deliberative reasoning simulated by reasoning models or CoT prompting. Our results challenge the default assumption that more reasoning always leads to better LLM decisions, particularly in high-stakes financial applications. 

---
# Intelligent Channel Allocation for IEEE 802.11be Multi-Link Operation: When MAB Meets LLM 

**Authors**: Shumin Lian, Jingwen Tong, Jun Zhang, Liqun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04594)  

**Abstract**: WiFi networks have achieved remarkable success in enabling seamless communication and data exchange worldwide. The IEEE 802.11be standard, known as WiFi 7, introduces Multi-Link Operation (MLO), a groundbreaking feature that enables devices to establish multiple simultaneous connections across different bands and channels. While MLO promises substantial improvements in network throughput and latency reduction, it presents significant challenges in channel allocation, particularly in dense network environments. Current research has predominantly focused on performance analysis and throughput optimization within static WiFi 7 network configurations. In contrast, this paper addresses the dynamic channel allocation problem in dense WiFi 7 networks with MLO capabilities. We formulate this challenge as a combinatorial optimization problem, leveraging a novel network performance analysis mechanism. Given the inherent lack of prior network information, we model the problem within a Multi-Armed Bandit (MAB) framework to enable online learning of optimal channel allocations. Our proposed Best-Arm Identification-enabled Monte Carlo Tree Search (BAI-MCTS) algorithm includes rigorous theoretical analysis, providing upper bounds for both sample complexity and error probability. To further reduce sample complexity and enhance generalizability across diverse network scenarios, we put forth LLM-BAI-MCTS, an intelligent algorithm for the dynamic channel allocation problem by integrating the Large Language Model (LLM) into the BAI-MCTS algorithm. Numerical results demonstrate that the BAI-MCTS algorithm achieves a convergence rate approximately $50.44\%$ faster than the state-of-the-art algorithms when reaching $98\%$ of the optimal value. Notably, the convergence rate of the LLM-BAI-MCTS algorithm increases by over $63.32\%$ in dense networks. 

---
# MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale 

**Authors**: Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu, May D. Wang, Peifeng Ruan, Donghan Yang, Tao Wang, Guanghua Xiao, Carl Yang, Yang Xie, Wenqi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04405)  

**Abstract**: We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice. 

---
# AUTOCT: Automating Interpretable Clinical Trial Prediction with LLM Agents 

**Authors**: Fengze Liu, Haoyu Wang, Joonhyuk Cho, Dan Roth, Andrew W. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.04293)  

**Abstract**: Clinical trials are critical for advancing medical treatments but remain prohibitively expensive and time-consuming. Accurate prediction of clinical trial outcomes can significantly reduce research and development costs and accelerate drug discovery. While recent deep learning models have shown promise by leveraging unstructured data, their black-box nature, lack of interpretability, and vulnerability to label leakage limit their practical use in high-stakes biomedical contexts. In this work, we propose AutoCT, a novel framework that combines the reasoning capabilities of large language models with the explainability of classical machine learning. AutoCT autonomously generates, evaluates, and refines tabular features based on public information without human input. Our method uses Monte Carlo Tree Search to iteratively optimize predictive performance. Experimental results show that AutoCT performs on par with or better than SOTA methods on clinical trial prediction tasks within only a limited number of self-refinement iterations, establishing a new paradigm for scalable, interpretable, and cost-efficient clinical trial prediction. 

---
# Evaluating MLLMs with Multimodal Multi-image Reasoning Benchmark 

**Authors**: Ziming Cheng, Binrui Xu, Lisheng Gong, Zuhe Song, Tianshuo Zhou, Shiqi Zhong, Siyu Ren, Mingxiang Chen, Xiangchao Meng, Yuxin Zhang, Yanlin Li, Lei Ren, Wei Chen, Zhiyuan Huang, Mingjie Zhan, Xiaojie Wang, Fangxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04280)  

**Abstract**: With enhanced capabilities and widespread applications, Multimodal Large Language Models (MLLMs) are increasingly required to process and reason over multiple images simultaneously. However, existing MLLM benchmarks focus either on single-image visual reasoning or on multi-image understanding tasks with only final-answer evaluation, leaving the reasoning capabilities of MLLMs over multi-image inputs largely underexplored. To address this gap, we introduce the $\textbf{Multimodal Multi-image Reasoning Benchmark (MMRB)}$, the first benchmark designed to evaluate structured visual reasoning across multiple images. MMRB comprises $\textbf{92 sub-tasks}$ covering spatial, temporal, and semantic reasoning, with multi-solution, CoT-style annotations generated by GPT-4o and refined by human experts. A derivative subset is designed to evaluate multimodal reward models in multi-image scenarios. To support fast and scalable evaluation, we propose a sentence-level matching framework using open-source LLMs. Extensive baseline experiments on $\textbf{40 MLLMs}$, including 9 reasoning-specific models and 8 reward models, demonstrate that open-source MLLMs still lag significantly behind commercial MLLMs in multi-image reasoning tasks. Furthermore, current multimodal reward models are nearly incapable of handling multi-image reward ranking tasks. 

---
# HSSBench: Benchmarking Humanities and Social Sciences Ability for Multimodal Large Language Models 

**Authors**: Zhaolu Kang, Junhao Gong, Jiaxu Yan, Wanke Xia, Yian Wang, Ziwen Wang, Huaxuan Ding, Zhuo Cheng, Wenhao Cao, Zhiyuan Feng, Siqi He, Shannan Yan, Junzhe Chen, Xiaomin He, Chaoya Jiang, Wei Ye, Kaidong Yu, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03922)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant potential to advance a broad range of domains. However, current benchmarks for evaluating MLLMs primarily emphasize general knowledge and vertical step-by-step reasoning typical of STEM disciplines, while overlooking the distinct needs and potential of the Humanities and Social Sciences (HSS). Tasks in the HSS domain require more horizontal, interdisciplinary thinking and a deep integration of knowledge across related fields, which presents unique challenges for MLLMs, particularly in linking abstract concepts with corresponding visual representations. Addressing this gap, we present HSSBench, a dedicated benchmark designed to assess the capabilities of MLLMs on HSS tasks in multiple languages, including the six official languages of the United Nations. We also introduce a novel data generation pipeline tailored for HSS scenarios, in which multiple domain experts and automated agents collaborate to generate and iteratively refine each sample. HSSBench contains over 13,000 meticulously designed samples, covering six key categories. We benchmark more than 20 mainstream MLLMs on HSSBench and demonstrate that it poses significant challenges even for state-of-the-art models. We hope that this benchmark will inspire further research into enhancing the cross-disciplinary reasoning abilities of MLLMs, especially their capacity to internalize and connect knowledge across fields. 

---
# RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought 

**Authors**: Yi Lu, Jiawang Cao, Yongliang Wu, Bozheng Li, Licheng Tang, Yangguang Ji, Chong Wu, Jay Wu, Wenbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04277)  

**Abstract**: Multi-modal Large Language Models (MLLMs) have demonstrated remarkable reasoning capability while lack explicit mechanisms for visual grounding and segmentation, creating a gap between cognitive reasoning and visual perception. To bridge this gap, we introduce Reasoning Segmentation via Visual Prompting (RSVP), a novel framework that unifies multi-step multimodal reasoning with grounded visual understanding. RSVP is a two-stage structuralized framework that integrates reasoning-driven localization with segmentation refinement. In the reasoning stage, RSVP employs multimodal chain-of-thought visual prompts to help MLLMs understand queries and infer targets, generating interpretable region proposals that enhance visual grounding. In segmentation stage, RSVP refines these proposals with a Vision-Language Segmentation Module (VLSM), seamlessly integrates textual and visual cues to produce precise segmentation masks. By explicitly modelling the interaction between multimodal reasoning and segmentation, RSVP introduces a new paradigm for interpretable reasoning segmentation. It exploits MLLMs' inherent localization capabilities, enabling the models to not only reason about objects but also generate structured visual representations. Our extensive experiments demonstrate that RSVP achieves state-of-the-art performance, surpasses state-of-the-art methods by up to +6.5 gIoU and +9.2 cIoU on ReasonSeg, and achieves 49.7 mAP on SegInW under zero-shot settings. These results validate RSVP as an effective and scalable framework for integrating cognitive reasoning with structured visual understanding. 

---
# COSMOS: Predictable and Cost-Effective Adaptation of LLMs 

**Authors**: Jiayu Wang, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2505.01449)  

**Abstract**: Large language models (LLMs) achieve remarkable performance across numerous tasks by using a diverse array of adaptation strategies. However, optimally selecting a model and adaptation strategy under resource constraints is challenging and often requires extensive experimentation. We investigate whether it is possible to accurately predict both performance and cost without expensive trials. We formalize the strategy selection problem for LLMs and introduce COSMOS, a unified prediction framework that efficiently estimates adaptation outcomes at minimal cost. We instantiate and study the capability of our framework via a pair of powerful predictors: embedding-augmented lightweight proxy models to predict fine-tuning performance, and low-sample scaling laws to forecast retrieval-augmented in-context learning. Extensive evaluation across eight representative benchmarks demonstrates that COSMOS achieves high prediction accuracy while reducing computational costs by 92.72% on average, and up to 98.71% in resource-intensive scenarios. Our results show that efficient prediction of adaptation outcomes is not only feasible but can substantially reduce the computational overhead of LLM deployment while maintaining performance standards. 

---
# Knowledge-guided Contextual Gene Set Analysis Using Large Language Models 

**Authors**: Zhizheng Wang, Chi-Ping Day, Chih-Hsuan Wei, Qiao Jin, Robert Leaman, Yifan Yang, Shubo Tian, Aodong Qiu, Yin Fang, Qingqing Zhu, Xinghua Lu, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04303)  

**Abstract**: Gene set analysis (GSA) is a foundational approach for interpreting genomic data of diseases by linking genes to biological processes. However, conventional GSA methods overlook clinical context of the analyses, often generating long lists of enriched pathways with redundant, nonspecific, or irrelevant results. Interpreting these requires extensive, ad-hoc manual effort, reducing both reliability and reproducibility. To address this limitation, we introduce cGSA, a novel AI-driven framework that enhances GSA by incorporating context-aware pathway prioritization. cGSA integrates gene cluster detection, enrichment analysis, and large language models to identify pathways that are not only statistically significant but also biologically meaningful. Benchmarking on 102 manually curated gene sets across 19 diseases and ten disease-related biological mechanisms shows that cGSA outperforms baseline methods by over 30%, with expert validation confirming its increased precision and interpretability. Two independent case studies in melanoma and breast cancer further demonstrate its potential to uncover context-specific insights and support targeted hypothesis generation. 

---
# Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models 

**Authors**: Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05176)  

**Abstract**: In this work, we introduce the Qwen3 Embedding series, a significant advancement over its predecessor, the GTE-Qwen series, in text embedding and reranking capabilities, built upon the Qwen3 foundation models. Leveraging the Qwen3 LLMs' robust capabilities in multilingual text understanding and generation, our innovative multi-stage training pipeline combines large-scale unsupervised pre-training with supervised fine-tuning on high-quality datasets. Effective model merging strategies further ensure the robustness and adaptability of the Qwen3 Embedding series. During the training process, the Qwen3 LLMs serve not only as backbone models but also play a crucial role in synthesizing high-quality, rich, and diverse training data across multiple domains and languages, thus enhancing the training pipeline. The Qwen3 Embedding series offers a spectrum of model sizes (0.6B, 4B, 8B) for both embedding and reranking tasks, addressing diverse deployment scenarios where users can optimize for either efficiency or effectiveness. Empirical evaluations demonstrate that the Qwen3 Embedding series achieves state-of-the-art results across diverse benchmarks. Notably, it excels on the multilingual evaluation benchmark MTEB for text embedding, as well as in various retrieval tasks, including code retrieval, cross-lingual retrieval and multilingual retrieval. To facilitate reproducibility and promote community-driven research and development, the Qwen3 Embedding models are publicly available under the Apache 2.0 license. 

---
# CLATTER: Comprehensive Entailment Reasoning for Hallucination Detection 

**Authors**: Ron Eliav, Arie Cattan, Eran Hirsch, Shahaf Bassan, Elias Stengel-Eskin, Mohit Bansal, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05243)  

**Abstract**: A common approach to hallucination detection casts it as a natural language inference (NLI) task, often using LLMs to classify whether the generated text is entailed by corresponding reference texts. Since entailment classification is a complex reasoning task, one would expect that LLMs could benefit from generating an explicit reasoning process, as in CoT reasoning or the explicit ``thinking'' of recent reasoning models. In this work, we propose that guiding such models to perform a systematic and comprehensive reasoning process -- one that both decomposes the text into smaller facts and also finds evidence in the source for each fact -- allows models to execute much finer-grained and accurate entailment decisions, leading to increased performance. To that end, we define a 3-step reasoning process, consisting of (i) claim decomposition, (ii) sub-claim attribution and entailment classification, and (iii) aggregated classification, showing that such guided reasoning indeed yields improved hallucination detection. Following this reasoning framework, we introduce an analysis scheme, consisting of several metrics that measure the quality of the intermediate reasoning steps, which provided additional empirical evidence for the improved quality of our guided reasoning scheme. 

---
# Do Large Language Models Judge Error Severity Like Humans? 

**Authors**: Diege Sun, Guanyi Chen, Fan Zhao, Xiaorong Cheng, Tingting He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05142)  

**Abstract**: Large Language Models (LLMs) are increasingly used as automated evaluators in natural language generation, yet it remains unclear whether they can accurately replicate human judgments of error severity. In this study, we systematically compare human and LLM assessments of image descriptions containing controlled semantic errors. We extend the experimental framework of van Miltenburg et al. (2020) to both unimodal (text-only) and multimodal (text + image) settings, evaluating four error types: age, gender, clothing type, and clothing colour. Our findings reveal that humans assign varying levels of severity to different error types, with visual context significantly amplifying perceived severity for colour and type errors. Notably, most LLMs assign low scores to gender errors but disproportionately high scores to colour errors, unlike humans, who judge both as highly severe but for different reasons. This suggests that these models may have internalised social norms influencing gender judgments but lack the perceptual grounding to emulate human sensitivity to colour, which is shaped by distinct neural mechanisms. Only one of the evaluated LLMs, Doubao, replicates the human-like ranking of error severity, but it fails to distinguish between error types as clearly as humans. Surprisingly, DeepSeek-V3, a unimodal LLM, achieves the highest alignment with human judgments across both unimodal and multimodal conditions, outperforming even state-of-the-art multimodal models. 

---
# Parking, Perception, and Retail: Street-Level Determinants of Community Vitality in Harbin 

**Authors**: HaoTian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05080)  

**Abstract**: The commercial vitality of community-scale streets in Chinese cities is shaped by complex interactions between vehicular accessibility, environmental quality, and pedestrian perception. This study proposes an interpretable, image-based framework to examine how street-level features -- including parked vehicle density, greenery, cleanliness, and street width -- impact retail performance and user satisfaction in Harbin, China. Leveraging street view imagery and a multimodal large language model (VisualGLM-6B), we construct a Community Commercial Vitality Index (CCVI) from Meituan and Dianping data and analyze its relationship with spatial attributes extracted via GPT-4-based perception modeling. Our findings reveal that while moderate vehicle presence may enhance commercial access, excessive on-street parking -- especially in narrow streets -- erodes walkability and reduces both satisfaction and shop-level pricing. In contrast, streets with higher perceived greenery and cleanliness show significantly greater satisfaction scores but only weak associations with pricing. Street width moderates the effects of vehicle presence, underscoring the importance of spatial configuration. These results demonstrate the value of integrating AI-assisted perception with urban morphological analysis to capture non-linear and context-sensitive drivers of commercial success. This study advances both theoretical and methodological frontiers by highlighting the conditional role of vehicle activity in neighborhood commerce and demonstrating the feasibility of multimodal AI for perceptual urban diagnostics. The implications extend to urban design, parking management, and scalable planning tools for community revitalization. 

---
# Just a Scratch: Enhancing LLM Capabilities for Self-harm Detection through Intent Differentiation and Emoji Interpretation 

**Authors**: Soumitra Ghosh, Gopendra Vikram Singh, Shambhavi, Sabarna Choudhury, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.05073)  

**Abstract**: Self-harm detection on social media is critical for early intervention and mental health support, yet remains challenging due to the subtle, context-dependent nature of such expressions. Identifying self-harm intent aids suicide prevention by enabling timely responses, but current large language models (LLMs) struggle to interpret implicit cues in casual language and emojis. This work enhances LLMs' comprehension of self-harm by distinguishing intent through nuanced language-emoji interplay. We present the Centennial Emoji Sensitivity Matrix (CESM-100), a curated set of 100 emojis with contextual self-harm interpretations and the Self-Harm Identification aNd intent Extraction with Supportive emoji sensitivity (SHINES) dataset, offering detailed annotations for self-harm labels, casual mentions (CMs), and serious intents (SIs). Our unified framework: a) enriches inputs using CESM-100; b) fine-tunes LLMs for multi-task learning: self-harm detection (primary) and CM/SI span detection (auxiliary); c) generates explainable rationales for self-harm predictions. We evaluate the framework on three state-of-the-art LLMs-Llama 3, Mental-Alpaca, and MentalLlama, across zero-shot, few-shot, and fine-tuned scenarios. By coupling intent differentiation with contextual cues, our approach commendably enhances LLM performance in both detection and explanation tasks, effectively addressing the inherent ambiguity in self-harm signals. The SHINES dataset, CESM-100 and codebase are publicly available at: this https URL . 

---
# Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation 

**Authors**: Noy Sternlicht, Ariel Gera, Roy Bar-Haim, Tom Hope, Noam Slonim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05062)  

**Abstract**: We introduce Debate Speech Evaluation as a novel and challenging benchmark for assessing LLM judges. Evaluating debate speeches requires a deep understanding of the speech at multiple levels, including argument strength and relevance, the coherence and organization of the speech, the appropriateness of its style and tone, and so on. This task involves a unique set of cognitive abilities that have previously received limited attention in systematic LLM benchmarking. To explore such skills, we leverage a dataset of over 600 meticulously annotated debate speeches and present the first in-depth analysis of how state-of-the-art LLMs compare to human judges on this task. Our findings reveal a nuanced picture: while larger models can approximate individual human judgments in some respects, they differ substantially in their overall judgment behavior. We also investigate the ability of frontier LLMs to generate persuasive, opinionated speeches, showing that models may perform at a human level on this task. 

---
# RELIC: Evaluating Compositional Instruction Following via Language Recognition 

**Authors**: Jackson Petty, Michael Y. Hu, Wentao Wang, Shauli Ravfogel, William Merrill, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05205)  

**Abstract**: Large language models (LLMs) are increasingly expected to perform tasks based only on a specification of the task provided in context, without examples of inputs and outputs; this ability is referred to as instruction following. We introduce the Recognition of Languages In-Context (RELIC) framework to evaluate instruction following using language recognition: the task of determining if a string is generated by formal grammar. Unlike many standard evaluations of LLMs' ability to use their context, this task requires composing together a large number of instructions (grammar productions) retrieved from the context. Because the languages are synthetic, the task can be increased in complexity as LLMs' skills improve, and new instances can be automatically generated, mitigating data contamination. We evaluate state-of-the-art LLMs on RELIC and find that their accuracy can be reliably predicted from the complexity of the grammar and the individual example strings, and that even the most advanced LLMs currently available show near-chance performance on more complex grammars and samples, in line with theoretical expectations. We also use RELIC to diagnose how LLMs attempt to solve increasingly difficult reasoning tasks, finding that as the complexity of the language recognition task increases, models switch to relying on shallow heuristics instead of following complex instructions. 

---
# Automatic Robustness Stress Testing of LLMs as Mathematical Problem Solvers 

**Authors**: Yutao Hou, Zeguan Xiao, Fei Yu, Yihan Jiang, Xuetao Wei, Hailiang Huang, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05038)  

**Abstract**: Large language models (LLMs) have achieved distinguished performance on various reasoning-intensive tasks. However, LLMs might still face the challenges of robustness issues and fail unexpectedly in some simple reasoning tasks. Previous works evaluate the LLM robustness with hand-crafted templates or a limited set of perturbation rules, indicating potential data contamination in pre-training or fine-tuning datasets. In this work, inspired by stress testing in software engineering, we propose a novel framework, Automatic Robustness Checker (AR-Checker), to generate mathematical problem variants that maintain the semantic meanings of the original one but might fail the LLMs. The AR-Checker framework generates mathematical problem variants through multi-round parallel streams of LLM-based rewriting and verification. Our framework can generate benchmark variants dynamically for each LLM, thus minimizing the risk of data contamination. Experiments on GSM8K and MATH-500 demonstrate the strong performance of AR-Checker on mathematical tasks. We also evaluate AR-Checker on benchmarks beyond mathematics, including MMLU, MMLU-Pro, and CommonsenseQA, where it also achieves strong performance, further proving the effectiveness of AR-Checker. 

---
# The NTNU System at the S&I Challenge 2025 SLA Open Track 

**Authors**: Hong-Yun Lin, Tien-Hong Lo, Yu-Hsuan Fang, Jhen-Ke Lin, Chung-Chun Wang, Hao-Chien Lu, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05121)  

**Abstract**: A recent line of research on spoken language assessment (SLA) employs neural models such as BERT and wav2vec 2.0 (W2V) to evaluate speaking proficiency across linguistic and acoustic modalities. Although both models effectively capture features relevant to oral competence, each exhibits modality-specific limitations. BERT-based methods rely on ASR transcripts, which often fail to capture prosodic and phonetic cues for SLA. In contrast, W2V-based methods excel at modeling acoustic features but lack semantic interpretability. To overcome these limitations, we propose a system that integrates W2V with Phi-4 multimodal large language model (MLLM) through a score fusion strategy. The proposed system achieves a root mean square error (RMSE) of 0.375 on the official test set of the Speak & Improve Challenge 2025, securing second place in the competition. For comparison, the RMSEs of the top-ranked, third-ranked, and official baseline systems are 0.364, 0.384, and 0.444, respectively. 

---
# ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development 

**Authors**: Zhenran Xu, Xue Yang, Yiyu Wang, Qingli Hu, Zijiao Wu, Longyue Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05010)  

**Abstract**: We introduce ComfyUI-Copilot, a large language model-powered plugin designed to enhance the usability and efficiency of ComfyUI, an open-source platform for AI-driven art creation. Despite its flexibility and user-friendly interface, ComfyUI can present challenges to newcomers, including limited documentation, model misconfigurations, and the complexity of workflow design. ComfyUI-Copilot addresses these challenges by offering intelligent node and model recommendations, along with automated one-click workflow construction. At its core, the system employs a hierarchical multi-agent framework comprising a central assistant agent for task delegation and specialized worker agents for different usages, supported by our curated ComfyUI knowledge bases to streamline debugging and deployment. We validate the effectiveness of ComfyUI-Copilot through both offline quantitative evaluations and online user feedback, showing that it accurately recommends nodes and accelerates workflow development. Additionally, use cases illustrate that ComfyUI-Copilot lowers entry barriers for beginners and enhances workflow efficiency for experienced users. The ComfyUI-Copilot installation package and a demo video are available at this https URL. 

---
# Controlling Summarization Length Through EOS Token Weighting 

**Authors**: Zeno Belligoli, Emmanouil Stergiadis, Eran Fainman, Ilya Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2506.05017)  

**Abstract**: Controlling the length of generated text can be crucial in various text-generation tasks, including summarization. Existing methods often require complex model alterations, limiting compatibility with pre-trained models. We address these limitations by developing a simple approach for controlling the length of automatic text summaries by increasing the importance of correctly predicting the EOS token in the cross-entropy loss computation. The proposed methodology is agnostic to architecture and decoding algorithms and orthogonal to other inference-time techniques to control the generation length. We tested it with encoder-decoder and modern GPT-style LLMs, and show that this method can control generation length, often without affecting the quality of the summary. 

---
# SCOP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View 

**Authors**: Yongjie Xiao, Hongru Liang, Peixin Qin, Yao Zhang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05000)  

**Abstract**: Despite the great potential of large language models(LLMs) in machine comprehension, it is still disturbing to fully count on them in real-world scenarios. This is probably because there is no rational explanation for whether the comprehension process of LLMs is aligned with that of experts. In this paper, we propose SCOP to carefully examine how LLMs perform during the comprehension process from a cognitive view. Specifically, it is equipped with a systematical definition of five requisite skills during the comprehension process, a strict framework to construct testing data for these skills, and a detailed analysis of advanced open-sourced and closed-sourced LLMs using the testing data. With SCOP, we find that it is still challenging for LLMs to perform an expert-level comprehension process. Even so, we notice that LLMs share some similarities with experts, e.g., performing better at comprehending local information than global information. Further analysis reveals that LLMs can be somewhat unreliable -- they might reach correct answers through flawed comprehension processes. Based on SCOP, we suggest that one direction for improving LLMs is to focus more on the comprehension process, ensuring all comprehension skills are thoroughly developed during training. 

---
# Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04832)  

**Abstract**: Large Reasoning Models (LRMs) extend large language models with explicit, multi-step reasoning traces to enhance transparency and performance on complex tasks. However, these reasoning traces can be redundant or logically inconsistent, making them a new source of hallucination that is difficult to detect. Existing hallucination detection methods focus primarily on answer-level uncertainty and often fail to detect hallucinations or logical inconsistencies arising from the model's reasoning trace. This oversight is particularly problematic for LRMs, where the explicit thinking trace is not only an important support to the model's decision-making process but also a key source of potential hallucination. To this end, we propose RACE (Reasoning and Answer Consistency Evaluation), a novel framework specifically tailored for hallucination detection in LRMs. RACE operates by extracting essential reasoning steps and computing four diagnostic signals: inter-sample consistency of reasoning traces, entropy-based answer uncertainty, semantic alignment between reasoning and answers, and internal coherence of reasoning. This joint analysis enables fine-grained hallucination detection even when the final answer appears correct. Experiments across datasets and different LLMs demonstrate that RACE outperforms existing hallucination detection baselines, offering a robust and generalizable solution for evaluating LRMs. Our code is available at: this https URL. 

---
# MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark 

**Authors**: Dingdong Wang, Jincenzi Wu, Junan Li, Dongchao Yang, Xueyuan Chen, Tianhua Zhang, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04779)  

**Abstract**: Speech inherently contains rich acoustic information that extends far beyond the textual language. In real-world spoken language understanding, effective interpretation often requires integrating semantic meaning (e.g., content), paralinguistic features (e.g., emotions, speed, pitch) and phonological characteristics (e.g., prosody, intonation, rhythm), which are embedded in speech. While recent multimodal Speech Large Language Models (SpeechLLMs) have demonstrated remarkable capabilities in processing audio information, their ability to perform fine-grained perception and complex reasoning in natural speech remains largely unexplored. To address this gap, we introduce MMSU, a comprehensive benchmark designed specifically for understanding and reasoning in spoken language. MMSU comprises 5,000 meticulously curated audio-question-answer triplets across 47 distinct tasks. To ground our benchmark in linguistic theory, we systematically incorporate a wide range of linguistic phenomena, including phonetics, prosody, rhetoric, syntactics, semantics, and paralinguistics. Through a rigorous evaluation of 14 advanced SpeechLLMs, we identify substantial room for improvement in existing models, highlighting meaningful directions for future optimization. MMSU establishes a new standard for comprehensive assessment of spoken language understanding, providing valuable insights for developing more sophisticated human-AI speech interaction systems. MMSU benchmark is available at this https URL. Evaluation Code is available at this https URL. 

---
# SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat 

**Authors**: Yuru Jiang, Wenxuan Ding, Shangbin Feng, Greg Durrett, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04721)  

**Abstract**: We propose SPARTA ALIGNMENT, an algorithm to collectively align multiple LLMs through competition and combat. To complement a single model's lack of diversity in generation and biases in evaluation, multiple LLMs form a "sparta tribe" to compete against each other in fulfilling instructions while serving as judges for the competition of others. For each iteration, one instruction and two models are selected for a duel, the other models evaluate the two responses, and their evaluation scores are aggregated through a adapted elo-ranking based reputation system, where winners/losers of combat gain/lose weight in evaluating others. The peer-evaluated combat results then become preference pairs where the winning response is preferred over the losing one, and all models learn from these preferences at the end of each iteration. SPARTA ALIGNMENT enables the self-evolution of multiple LLMs in an iterative and collective competition process. Extensive experiments demonstrate that SPARTA ALIGNMENT outperforms initial models and 4 self-alignment baselines across 10 out of 12 tasks and datasets with 7.0% average improvement. Further analysis reveals that SPARTA ALIGNMENT generalizes more effectively to unseen tasks and leverages the expertise diversity of participating models to produce more logical, direct and informative outputs. 

---
# Identifying Reliable Evaluation Metrics for Scientific Text Revision 

**Authors**: Léane Jourdan, Florian Boudin, Richard Dufour, Nicolas Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.04772)  

**Abstract**: Evaluating text revision in scientific writing remains a challenge, as traditional metrics such as ROUGE and BERTScore primarily focus on similarity rather than capturing meaningful improvements. In this work, we analyse and identify the limitations of these metrics and explore alternative evaluation methods that better align with human judgments. We first conduct a manual annotation study to assess the quality of different revisions. Then, we investigate reference-free evaluation metrics from related NLP domains. Additionally, we examine LLM-as-a-judge approaches, analysing their ability to assess revisions with and without a gold reference. Our results show that LLMs effectively assess instruction-following but struggle with correctness, while domain-specific metrics provide complementary insights. We find that a hybrid approach combining LLM-as-a-judge evaluation and task-specific metrics offers the most reliable assessment of revision quality. 

---
# Cracking the Code: Enhancing Implicit Hate Speech Detection through Coding Classification 

**Authors**: Lu Wei, Liangzhi Li, Tong Xiang, Xiao Liu, Noa Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2506.04693)  

**Abstract**: The internet has become a hotspot for hate speech (HS), threatening societal harmony and individual well-being. While automatic detection methods perform well in identifying explicit hate speech (ex-HS), they struggle with more subtle forms, such as implicit hate speech (im-HS). We tackle this problem by introducing a new taxonomy for im-HS detection, defining six encoding strategies named codetypes. We present two methods for integrating codetypes into im-HS detection: 1) prompting large language models (LLMs) directly to classify sentences based on generated responses, and 2) using LLMs as encoders with codetypes embedded during the encoding process. Experiments show that the use of codetypes improves im-HS detection in both Chinese and English datasets, validating the effectiveness of our approach across different languages. 

---
# Normative Conflicts and Shallow AI Alignment 

**Authors**: Raphaël Millière  

**Link**: [PDF](https://arxiv.org/pdf/2506.04679)  

**Abstract**: The progress of AI systems such as large language models (LLMs) raises increasingly pressing concerns about their safe deployment. This paper examines the value alignment problem for LLMs, arguing that current alignment strategies are fundamentally inadequate to prevent misuse. Despite ongoing efforts to instill norms such as helpfulness, honesty, and harmlessness in LLMs through fine-tuning based on human preferences, they remain vulnerable to adversarial attacks that exploit conflicts between these norms. I argue that this vulnerability reflects a fundamental limitation of existing alignment methods: they reinforce shallow behavioral dispositions rather than endowing LLMs with a genuine capacity for normative deliberation. Drawing from on research in moral psychology, I show how humans' ability to engage in deliberative reasoning enhances their resilience against similar adversarial tactics. LLMs, by contrast, lack a robust capacity to detect and rationally resolve normative conflicts, leaving them susceptible to manipulation; even recent advances in reasoning-focused LLMs have not addressed this vulnerability. This ``shallow alignment'' problem carries significant implications for AI safety and regulation, suggesting that current approaches are insufficient for mitigating potential harms posed by increasingly capable AI systems. 

---
# Evaluating Vision-Language and Large Language Models for Automated Student Assessment in Indonesian Classrooms 

**Authors**: Nurul Aisyah, Muhammad Dehan Al Kautsar, Arif Hidayat, Raqib Chowdhury, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2506.04822)  

**Abstract**: Although vision-language and large language models (VLM and LLM) offer promising opportunities for AI-driven educational assessment, their effectiveness in real-world classroom settings, particularly in underrepresented educational contexts, remains underexplored. In this study, we evaluated the performance of a state-of-the-art VLM and several LLMs on 646 handwritten exam responses from grade 4 students in six Indonesian schools, covering two subjects: Mathematics and English. These sheets contain more than 14K student answers that span multiple choice, short answer, and essay questions. Assessment tasks include grading these responses and generating personalized feedback. Our findings show that the VLM often struggles to accurately recognize student handwriting, leading to error propagation in downstream LLM grading. Nevertheless, LLM-generated feedback retains some utility, even when derived from imperfect input, although limitations in personalization and contextual relevance persist. 

---
# ICPC-Eval: Probing the Frontiers of LLM Reasoning with Competitive Programming Contests 

**Authors**: Shiyi Xu, Yiwen Hu, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04894)  

**Abstract**: With the significant progress of large reasoning models in complex coding and reasoning tasks, existing benchmarks, like LiveCodeBench and CodeElo, are insufficient to evaluate the coding capabilities of large language models (LLMs) in real competition environments. Moreover, current evaluation metrics such as Pass@K fail to capture the reflective abilities of reasoning models. To address these challenges, we propose \textbf{ICPC-Eval}, a top-level competitive coding benchmark designed to probing the frontiers of LLM reasoning. ICPC-Eval includes 118 carefully curated problems from 11 recent ICPC contests held in various regions of the world, offering three key contributions: 1) A challenging realistic ICPC competition scenario, featuring a problem type and difficulty distribution consistent with actual contests. 2) A robust test case generation method and a corresponding local evaluation toolkit, enabling efficient and accurate local evaluation. 3) An effective test-time scaling evaluation metric, Refine@K, which allows iterative repair of solutions based on execution feedback. The results underscore the significant challenge in evaluating complex reasoning abilities: top-tier reasoning models like DeepSeek-R1 often rely on multi-turn code feedback to fully unlock their in-context reasoning potential when compared to non-reasoning counterparts. Furthermore, despite recent advancements in code generation, these models still lag behind top-performing human teams. We release the benchmark at: this https URL 

---
# Revisiting Test-Time Scaling: A Survey and a Diversity-Aware Method for Efficient Reasoning 

**Authors**: Ho-Lam Chung, Teng-Yun Hsiao, Hsiao-Ying Huang, Chunerh Cho, Jian-Ren Lin, Zhang Ziwei, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04611)  

**Abstract**: Test-Time Scaling (TTS) improves the reasoning performance of Large Language Models (LLMs) by allocating additional compute during inference. We conduct a structured survey of TTS methods and categorize them into sampling-based, search-based, and trajectory optimization strategies. We observe that reasoning-optimized models often produce less diverse outputs, which limits TTS effectiveness. To address this, we propose ADAPT (A Diversity Aware Prefix fine-Tuning), a lightweight method that applies prefix tuning with a diversity-focused data strategy. Experiments on mathematical reasoning tasks show that ADAPT reaches 80% accuracy using eight times less compute than strong baselines. Our findings highlight the essential role of generative diversity in maximizing TTS effectiveness. 

---
# A MISMATCHED Benchmark for Scientific Natural Language Inference 

**Authors**: Firoz Shaik, Mobashir Sadat, Nikita Gautam, Doina Caragea, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2506.04603)  

**Abstract**: Scientific Natural Language Inference (NLI) is the task of predicting the semantic relation between a pair of sentences extracted from research articles. Existing datasets for this task are derived from various computer science (CS) domains, whereas non-CS domains are completely ignored. In this paper, we introduce a novel evaluation benchmark for scientific NLI, called MISMATCHED. The new MISMATCHED benchmark covers three non-CS domains-PSYCHOLOGY, ENGINEERING, and PUBLIC HEALTH, and contains 2,700 human annotated sentence pairs. We establish strong baselines on MISMATCHED using both Pre-trained Small Language Models (SLMs) and Large Language Models (LLMs). Our best performing baseline shows a Macro F1 of only 78.17% illustrating the substantial headroom for future improvements. In addition to introducing the MISMATCHED benchmark, we show that incorporating sentence pairs having an implicit scientific NLI relation between them in model training improves their performance on scientific NLI. We make our dataset and code publicly available on GitHub. 

---
# LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models 

**Authors**: Wen Ding, Fan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.04586)  

**Abstract**: We introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that leverages Large Language Models (LLMs) to correct pseudo labels generated from in-the-wild data. Within the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and augmented by a data filtering strategy to optimize LLM knowledge transfer efficiency. Experiments on both Mandarin ASR and Spanish-to-English AST tasks show that LESS achieves a notable absolute WER reduction of 3.77% on the Wenet Speech test set, as well as BLEU scores of 34.0 and 64.7 on Callhome and Fisher test sets respectively. These results validate the adaptability of LESS across different languages, tasks, and domains. Ablation studies conducted with various LLMs and prompt configurations provide novel insights into leveraging LLM-derived knowledge for speech processing applications. 

---
# Are LLMs Reliable Translators of Logical Reasoning Across Lexically Diversified Contexts? 

**Authors**: Qingchuan Li, Jiatong Li, Zirui Liu, Mingyue Cheng, Yuting Zeng, Qi Liu, Tongxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04575)  

**Abstract**: Neuro-symbolic approaches combining large language models (LLMs) with solvers excels in logical reasoning problems need long reasoning chains. In this paradigm, LLMs serve as translators, converting natural language reasoning problems into formal logic formulas. Then reliable symbolic solvers return correct solutions. Despite their success, we find that LLMs, as translators, struggle to handle lexical diversification, a common linguistic phenomenon, indicating that LLMs as logic translators are unreliable in real-world scenarios. Moreover, existing logical reasoning benchmarks lack lexical diversity, failing to challenge LLMs' ability to translate such text and thus obscuring this issue. In this work, we propose SCALe, a benchmark designed to address this significant gap through **logic-invariant lexical diversification**. By using LLMs to transform original benchmark datasets into lexically diversified but logically equivalent versions, we evaluate LLMs' ability to consistently map diverse expressions to uniform logical symbols on these new datasets. Experiments using SCALe further confirm that current LLMs exhibit deficiencies in this capability. Building directly on the deficiencies identified through our benchmark, we propose a new method, MenTaL, to address this limitation. This method guides LLMs to first construct a table unifying diverse expressions before performing translation. Applying MenTaL through in-context learning and supervised fine-tuning (SFT) significantly improves the performance of LLM translators on lexically diversified text. Our code is now available at this https URL. 

---
# Advancing Tool-Augmented Large Language Models via Meta-Verification and Reflection Learning 

**Authors**: Zhiyuan Ma, Jiayu Liu, Xianzhen Luo, Zhenya Huang, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2506.04625)  

**Abstract**: Empowering large language models (LLMs) with effective tool utilization capabilities is crucial for enabling AI agents to solve complex problems. However, current models face two major limitations: (1) unreliable tool planning and invocation due to low-quality instruction datasets (e.g., widespread hallucinated API calls), and (2) weak tool reflection abilities (over 90% of errors cannot be corrected) resulting from static imitation learning. To address these critical limitations, we propose Tool-MVR, a novel Tool-Augmented LLM that achieves comprehensive System 2 reasoning through two key innovations. Specifically, we first introduce Multi-Agent Meta-Verification (MAMV), a systematic pipeline that rigorously validates APIs, queries, and reasoning trajectories to construct ToolBench-V, a new high-quality instruction dataset that addresses the limitation of unreliable tool planning and invocation. Second, we propose Exploration-based Reflection Learning (EXPLORE), which enhances tool reflection capabilities by leveraging tool feedback through a dynamic "Error -> Reflection -> Correction" learning paradigm, resulting in our reflection dataset ToolBench-R and addressing the critical weakness in tool reflection. Finally, we obtain Tool-MVR by finetuning open-source LLMs (e.g., Qwen-7B) on both ToolBench-V and ToolBench-R. Our experiments demonstrate that Tool-MVR achieves state-of-the-art performance on StableToolBench, surpassing both ToolLLM (by 23.9%) and GPT-4 (by 15.3%) while reducing API calls by 31.4%, with strong generalization capabilities across unseen tools and scenarios. Additionally, on our proposed RefineToolBench, the first benchmark specifically designed to evaluate tool reflection capabilities, Tool-MVR achieves a 58.9% error correction rate, significantly outperforming ToolLLM's 9.1%. 

---
# Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs 

**Authors**: William Sheffield, Kanishka Misra, Valentina Pyatkin, Ashwini Deo, Kyle Mahowald, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04534)  

**Abstract**: Discourse particles are crucial elements that subtly shape the meaning of text. These words, often polyfunctional, give rise to nuanced and often quite disparate semantic/discourse effects, as exemplified by the diverse uses of the particle "just" (e.g., exclusive, temporal, emphatic). This work investigates the capacity of LLMs to distinguish the fine-grained senses of English "just", a well-studied example in formal semantics, using data meticulously created and labeled by expert linguists. Our findings reveal that while LLMs exhibit some ability to differentiate between broader categories, they struggle to fully capture more subtle nuances, highlighting a gap in their understanding of discourse particles. 

---
# SQLens: An End-to-End Framework for Error Detection and Correction in Text-to-SQL 

**Authors**: Yue Gong, Chuan Lei, Xiao Qin, Kapil Vaidya, Balakrishnan Narayanaswamy, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2506.04494)  

**Abstract**: Text-to-SQL systems translate natural language (NL) questions into SQL queries, enabling non-technical users to interact with structured data. While large language models (LLMs) have shown promising results on the text-to-SQL task, they often produce semantically incorrect yet syntactically valid queries, with limited insight into their reliability. We propose SQLens, an end-to-end framework for fine-grained detection and correction of semantic errors in LLM-generated SQL. SQLens integrates error signals from both the underlying database and the LLM to identify potential semantic errors within SQL clauses. It further leverages these signals to guide query correction. Empirical results on two public benchmarks show that SQLens outperforms the best LLM-based self-evaluation method by 25.78% in F1 for error detection, and improves execution accuracy of out-of-the-box text-to-SQL systems by up to 20%. 

---
# Aligning Large Language Models with Implicit Preferences from User-Generated Content 

**Authors**: Zhaoxuan Tan, Zheng Li, Tianyi Liu, Haodong Wang, Hyokun Yun, Ming Zeng, Pei Chen, Zhihan Zhang, Yifan Gao, Ruijie Wang, Priyanka Nigam, Bing Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04463)  

**Abstract**: Learning from preference feedback is essential for aligning large language models (LLMs) with human values and improving the quality of generated responses. However, existing preference learning methods rely heavily on curated data from humans or advanced LLMs, which is costly and difficult to scale. In this work, we present PUGC, a novel framework that leverages implicit human Preferences in unlabeled User-Generated Content (UGC) to generate preference data. Although UGC is not explicitly created to guide LLMs in generating human-preferred responses, it often reflects valuable insights and implicit preferences from its creators that has the potential to address readers' questions. PUGC transforms UGC into user queries and generates responses from the policy model. The UGC is then leveraged as a reference text for response scoring, aligning the model with these implicit preferences. This approach improves the quality of preference data while enabling scalable, domain-specific alignment. Experimental results on Alpaca Eval 2 show that models trained with DPO and PUGC achieve a 9.37% performance improvement over traditional methods, setting a 35.93% state-of-the-art length-controlled win rate using Mistral-7B-Instruct. Further studies highlight gains in reward quality, domain-specific alignment effectiveness, robustness against UGC quality, and theory of mind capabilities. Our code and dataset are available at this https URL 

---
# Zero-Shot Open-Schema Entity Structure Discovery 

**Authors**: Xueqiang Xu, Jinfeng Xiao, James Barry, Mohab Elkaref, Jiaru Zou, Pengcheng Jiang, Yunyi Zhang, Max Giammona, Geeth de Mel, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.04458)  

**Abstract**: Entity structure extraction, which aims to extract entities and their associated attribute-value structures from text, is an essential task for text understanding and knowledge graph construction. Existing methods based on large language models (LLMs) typically rely heavily on predefined entity attribute schemas or annotated datasets, often leading to incomplete extraction results. To address these challenges, we introduce Zero-Shot Open-schema Entity Structure Discovery (ZOES), a novel approach to entity structure extraction that does not require any schema or annotated samples. ZOES operates via a principled mechanism of enrichment, refinement, and unification, based on the insight that an entity and its associated structure are mutually reinforcing. Experiments demonstrate that ZOES consistently enhances LLMs' ability to extract more complete entity structures across three different domains, showcasing both the effectiveness and generalizability of the method. These findings suggest that such an enrichment, refinement, and unification mechanism may serve as a principled approach to improving the quality of LLM-based entity structure discovery in various scenarios. 

---
# Subjective Perspectives within Learned Representations Predict High-Impact Innovation 

**Authors**: Likun Cao, Rui Pan, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.04616)  

**Abstract**: Existing studies of innovation emphasize the power of social structures to shape innovation capacity. Emerging machine learning approaches, however, enable us to model innovators' personal perspectives and interpersonal innovation opportunities as a function of their prior trajectories of experience. We theorize then quantify subjective perspectives and innovation opportunities based on innovator positions within the geometric space of concepts inscribed by dynamic language representations. Using data on millions of scientists, inventors, writers, entrepreneurs, and Wikipedia contributors across the creative domains of science, technology, film, entrepreneurship, and Wikipedia, here we show that measured subjective perspectives anticipate what ideas individuals and groups creatively attend to and successfully combine in future. When perspective and background diversity are decomposed as the angular difference between collaborators' perspectives on their creation and between their experiences, the former consistently anticipates creative achievement while the latter portends its opposite, across all cases and time periods examined. We analyze a natural experiment and simulate creative collaborations between AI (large language model) agents designed with various perspective and background diversity, which are consistent with our observational findings. We explore mechanisms underlying these findings and identify how successful collaborators leverage common language to weave together diverse experience obtained through trajectories of prior work that converge to provoke one another and innovate. We explore the importance of these findings for team assembly and research policy. 

---
# GEM: Empowering LLM for both Embedding Generation and Language Understanding 

**Authors**: Caojin Zhang, Qiang Zhang, Ke Li, Sai Vidyaranya Nuthalapati, Benyu Zhang, Jason Liu, Serena Li, Lizhu Zhang, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04344)  

**Abstract**: Large decoder-only language models (LLMs) have achieved remarkable success in generation and reasoning tasks, where they generate text responses given instructions. However, many applications, e.g., retrieval augmented generation (RAG), still rely on separate embedding models to generate text embeddings, which can complicate the system and introduce discrepancies in understanding of the query between the embedding model and LLMs. To address this limitation, we propose a simple self-supervised approach, Generative Embedding large language Model (GEM), that enables any large decoder-only LLM to generate high-quality text embeddings while maintaining its original text generation and reasoning capabilities. Our method inserts new special token(s) into a text body, and generates summarization embedding of the text by manipulating the attention mask. This method could be easily integrated into post-training or fine tuning stages of any existing LLMs. We demonstrate the effectiveness of our approach by applying it to two popular LLM families, ranging from 1B to 8B parameters, and evaluating the transformed models on both text embedding benchmarks (MTEB) and NLP benchmarks (MMLU). The results show that our proposed method significantly improves the original LLMs on MTEB while having a minimal impact on MMLU. Our strong results indicate that our approach can empower LLMs with state-of-the-art text embedding capabilities while maintaining their original NLP performance 

---
# DRE: An Effective Dual-Refined Method for Integrating Small and Large Language Models in Open-Domain Dialogue Evaluation 

**Authors**: Kun Zhao, Bohao Yang, Chen Tang, Siyuan Dai, Haoteng Tang, Chenghua Lin, Liang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04516)  

**Abstract**: Large Language Models (LLMs) excel at many tasks but struggle with ambiguous scenarios where multiple valid responses exist, often yielding unreliable results. Conversely, Small Language Models (SLMs) demonstrate robustness in such scenarios but are susceptible to misleading or adversarial inputs. We observed that LLMs handle negative examples effectively, while SLMs excel with positive examples. To leverage their complementary strengths, we introduce SLIDE (Small and Large Integrated for Dialogue Evaluation), a method integrating SLMs and LLMs via adaptive weighting. Building on SLIDE, we further propose a Dual-Refinement Evaluation (DRE) method to enhance SLM-LLM integration: (1) SLM-generated insights guide the LLM to produce initial evaluations; (2) SLM-derived adjustments refine the LLM's scores for improved accuracy. Experiments demonstrate that DRE outperforms existing methods, showing stronger alignment with human judgment across diverse benchmarks. This work illustrates how combining small and large models can yield more reliable evaluation tools, particularly for open-ended tasks such as dialogue evaluation. 

---
# Inference-Time Hyper-Scaling with KV Cache Compression 

**Authors**: Adrian Łańcucki, Konrad Staniszewski, Piotr Nawrot, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2506.05345)  

**Abstract**: Inference-time scaling trades efficiency for increased reasoning accuracy by generating longer or more parallel sequences. However, in Transformer LLMs, generation cost is bottlenecked by the size of the key-value (KV) cache, rather than the number of generated tokens. Hence, we explore inference-time hyper-scaling: by compressing the KV cache, we can generate more tokens within the same compute budget and further improve the accuracy of scaled inference. The success of this approach, however, hinges on the ability of compression methods to preserve accuracy even at high compression ratios. To make hyper-scaling practical, we introduce Dynamic Memory Sparsification (DMS), a novel method for sparsifying KV caches that only requires 1K training steps to achieve 8$\times$ compression, while maintaining better accuracy than training-free sparse attention. Instead of prematurely discarding cached tokens, DMS delays token eviction, implicitly merging representations and preserving critical information. We demonstrate the effectiveness of inference-time hyper-scaling with DMS on multiple families of LLMs, showing that it boosts accuracy for comparable inference runtime and memory load. For instance, we enhance Qwen-R1 32B by an average of 9.1 points on AIME 24, 7.6 on GPQA, and 9.6 on LiveCodeBench across compute budgets. 

---
# Please Translate Again: Two Simple Experiments on Whether Human-Like Reasoning Helps Translation 

**Authors**: Di Wu, Seth Aycock, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2506.04521)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning capabilities for many tasks, often by explicitly decomposing the task via Chain-of-Thought (CoT) reasoning. Recent work on LLM-based translation designs hand-crafted prompts to decompose translation, or trains models to incorporate intermediate steps.~\textit{Translating Step-by-step}~\citep{briakou2024translating}, for instance, introduces a multi-step prompt with decomposition and refinement of translation with LLMs, which achieved state-of-the-art results on WMT24. In this work, we scrutinise this strategy's effectiveness. Empirically, we find no clear evidence that performance gains stem from explicitly decomposing the translation process, at least for the models on test; and we show that simply prompting LLMs to ``translate again'' yields even better results than human-like step-by-step prompting. Our analysis does not rule out the role of reasoning, but instead invites future work exploring the factors for CoT's effectiveness in the context of translation. 

---
# Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems 

**Authors**: Emma Harvey, Emily Sheng, Su Lin Blodgett, Alexandra Chouldechova, Jean Garcia-Gathright, Alexandra Olteanu, Hanna Wallach  

**Link**: [PDF](https://arxiv.org/pdf/2506.04482)  

**Abstract**: The NLP research community has made publicly available numerous instruments for measuring representational harms caused by large language model (LLM)-based systems. These instruments have taken the form of datasets, metrics, tools, and more. In this paper, we examine the extent to which such instruments meet the needs of practitioners tasked with evaluating LLM-based systems. Via semi-structured interviews with 12 such practitioners, we find that practitioners are often unable to use publicly available instruments for measuring representational harms. We identify two types of challenges. In some cases, instruments are not useful because they do not meaningfully measure what practitioners seek to measure or are otherwise misaligned with practitioner needs. In other cases, instruments - even useful instruments - are not used by practitioners due to practical and institutional barriers impeding their uptake. Drawing on measurement theory and pragmatic measurement, we provide recommendations for addressing these challenges to better meet practitioner needs. 

---
# Interpretable Multimodal Framework for Human-Centered Street Assessment: Integrating Visual-Language Models for Perceptual Urban Diagnostics 

**Authors**: HaoTian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05087)  

**Abstract**: While objective street metrics derived from imagery or GIS have become standard in urban analytics, they remain insufficient to capture subjective perceptions essential to inclusive urban design. This study introduces a novel Multimodal Street Evaluation Framework (MSEF) that fuses a vision transformer (VisualGLM-6B) with a large language model (GPT-4), enabling interpretable dual-output assessment of streetscapes. Leveraging over 15,000 annotated street-view images from Harbin, China, we fine-tune the framework using LoRA and P-Tuning v2 for parameter-efficient adaptation. The model achieves an F1 score of 0.84 on objective features and 89.3 percent agreement with aggregated resident perceptions, validated across stratified socioeconomic geographies. Beyond classification accuracy, MSEF captures context-dependent contradictions: for instance, informal commerce boosts perceived vibrancy while simultaneously reducing pedestrian comfort. It also identifies nonlinear and semantically contingent patterns -- such as the divergent perceptual effects of architectural transparency across residential and commercial zones -- revealing the limits of universal spatial heuristics. By generating natural-language rationales grounded in attention mechanisms, the framework bridges sensory data with socio-affective inference, enabling transparent diagnostics aligned with SDG 11. This work offers both methodological innovation in urban perception modeling and practical utility for planning systems seeking to reconcile infrastructural precision with lived experience. 

---
