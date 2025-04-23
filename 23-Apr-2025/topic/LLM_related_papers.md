# Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations 

**Authors**: Nikhil Khandalkar, Pavan Yadav, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15903)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility. 

---
# DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models 

**Authors**: Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15716)  

**Abstract**: Effective reasoning remains a core challenge for large language models (LLMs) in the financial domain, where tasks often require domain-specific knowledge, precise numerical calculations, and strict adherence to compliance rules. We propose DianJin-R1, a reasoning-enhanced framework designed to address these challenges through reasoning-augmented supervision and reinforcement learning. Central to our approach is DianJin-R1-Data, a high-quality dataset constructed from CFLUE, FinQA, and a proprietary compliance corpus (Chinese Compliance Check, CCC), combining diverse financial reasoning scenarios with verified annotations. Our models, DianJin-R1-7B and DianJin-R1-32B, are fine-tuned from Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct using a structured format that generates both reasoning steps and final answers. To further refine reasoning quality, we apply Group Relative Policy Optimization (GRPO), a reinforcement learning method that incorporates dual reward signals: one encouraging structured outputs and another rewarding answer correctness. We evaluate our models on five benchmarks: three financial datasets (CFLUE, FinQA, and CCC) and two general reasoning benchmarks (MATH-500 and GPQA-Diamond). Experimental results show that DianJin-R1 models consistently outperform their non-reasoning counterparts, especially on complex financial tasks. Moreover, on the real-world CCC dataset, our single-call reasoning models match or even surpass the performance of multi-agent systems that require significantly more computational cost. These findings demonstrate the effectiveness of DianJin-R1 in enhancing financial reasoning through structured supervision and reward-aligned learning, offering a scalable and practical solution for real-world applications. 

---
# Implementing Rational Choice Functions with LLMs and Measuring their Alignment with User Preferences 

**Authors**: Anna Karnysheva, Christian Drescher, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2504.15719)  

**Abstract**: As large language models (LLMs) become integral to intelligent user interfaces (IUIs), their role as decision-making agents raises critical concerns about alignment. Although extensive research has addressed issues such as factuality, bias, and toxicity, comparatively little attention has been paid to measuring alignment to preferences, i.e., the relative desirability of different alternatives, a concept used in decision making, economics, and social choice theory. However, a reliable decision-making agent makes choices that align well with user preferences.
In this paper, we generalize existing methods that exploit LLMs for ranking alternative outcomes by addressing alignment with the broader and more flexible concept of user preferences, which includes both strict preferences and indifference among alternatives. To this end, we put forward design principles for using LLMs to implement rational choice functions, and provide the necessary tools to measure preference satisfaction. We demonstrate the applicability of our approach through an empirical study in a practical application of an IUI in the automotive domain. 

---
# A Multi-Agent Framework for Automated Qinqiang Opera Script Generation Using Large Language Models 

**Authors**: Gengxian Cao, Fengyuan Li, Hong Duan, Ye Yang, Bofeng Wang, Donghe Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15552)  

**Abstract**: This paper introduces a novel multi-Agent framework that automates the end to end production of Qinqiang opera by integrating Large Language Models , visual generation, and Text to Speech synthesis. Three specialized agents collaborate in sequence: Agent1 uses an LLM to craft coherent, culturally grounded scripts;Agent2 employs visual generation models to render contextually accurate stage scenes; and Agent3 leverages TTS to produce synchronized, emotionally expressive vocal performances. In a case study on Dou E Yuan, the system achieved expert ratings of 3.8 for script fidelity, 3.5 for visual coherence, and 3.8 for speech accuracy-culminating in an overall score of 3.6, a 0.3 point improvement over a Single Agent baseline. Ablation experiments demonstrate that removing Agent2 or Agent3 leads to drops of 0.4 and 0.5 points, respectively, underscoring the value of modular collaboration. This work showcases how AI driven pipelines can streamline and scale the preservation of traditional performing arts, and points toward future enhancements in cross modal alignment, richer emotional nuance, and support for additional opera genres. 

---
# A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings 

**Authors**: Md Millat, Md Motiur  

**Link**: [PDF](https://arxiv.org/pdf/2504.15610)  

**Abstract**: The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy. 

---
# Learning Adaptive Parallel Reasoning with Language Models 

**Authors**: Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2504.15466)  

**Abstract**: Scaling inference-time computation has substantially improved the reasoning capabilities of language models. However, existing methods have significant limitations: serialized chain-of-thought approaches generate overly long outputs, leading to increased latency and exhausted context windows, while parallel methods such as self-consistency suffer from insufficient coordination, resulting in redundant computations and limited performance gains. To address these shortcomings, we propose Adaptive Parallel Reasoning (APR), a novel reasoning framework that enables language models to orchestrate both serialized and parallel computations end-to-end. APR generalizes existing reasoning methods by enabling adaptive multi-threaded inference using spawn() and join() operations. A key innovation is our end-to-end reinforcement learning strategy, optimizing both parent and child inference threads to enhance task success rate without requiring predefined reasoning structures. Experiments on the Countdown reasoning task demonstrate significant benefits of APR: (1) higher performance within the same context window (83.4% vs. 60.0% at 4k context); (2) superior scalability with increased computation (80.1% vs. 66.6% at 20k total tokens); (3) improved accuracy at equivalent latency (75.2% vs. 57.3% at approximately 5,000ms). APR represents a step towards enabling language models to autonomously optimize their reasoning processes through adaptive allocation of computation. 

---
# PolicyEvol-Agent: Evolving Policy via Environment Perception and Self-Awareness with Theory of Mind 

**Authors**: Yajie Yu, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.15313)  

**Abstract**: Multi-agents has exhibited significant intelligence in real-word simulations with Large language models (LLMs) due to the capabilities of social cognition and knowledge retrieval. However, existing research on agents equipped with effective cognition chains including reasoning, planning, decision-making and reflecting remains limited, especially in the dynamically interactive scenarios. In addition, unlike human, prompt-based responses face challenges in psychological state perception and empirical calibration during uncertain gaming process, which can inevitably lead to cognition bias. In light of above, we introduce PolicyEvol-Agent, a comprehensive LLM-empowered framework characterized by systematically acquiring intentions of others and adaptively optimizing irrational strategies for continual enhancement. Specifically, PolicyEvol-Agent first obtains reflective expertise patterns and then integrates a range of cognitive operations with Theory of Mind alongside internal and external perspectives. Simulation results, outperforming RL-based models and agent-based methods, demonstrate the superiority of PolicyEvol-Agent for final gaming victory. Moreover, the policy evolution mechanism reveals the effectiveness of dynamic guideline adjustments in both automatic and human evaluation. 

---
# LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities 

**Authors**: Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16078)  

**Abstract**: The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making. 

---
# LLMs meet Federated Learning for Scalable and Secure IoT Management 

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.16032)  

**Abstract**: The rapid expansion of IoT ecosystems introduces severe challenges in scalability, security, and real-time decision-making. Traditional centralized architectures struggle with latency, privacy concerns, and excessive resource consumption, making them unsuitable for modern large-scale IoT deployments. This paper presents a novel Federated Learning-driven Large Language Model (FL-LLM) framework, designed to enhance IoT system intelligence while ensuring data privacy and computational efficiency. The framework integrates Generative IoT (GIoT) models with a Gradient Sensing Federated Strategy (GSFS), dynamically optimizing model updates based on real-time network conditions. By leveraging a hybrid edge-cloud processing architecture, our approach balances intelligence, scalability, and security in distributed IoT environments. Evaluations on the IoT-23 dataset demonstrate that our framework improves model accuracy, reduces response latency, and enhances energy efficiency, outperforming traditional FL techniques (i.e., FedAvg, FedOpt). These findings highlight the potential of integrating LLM-powered federated learning into large-scale IoT ecosystems, paving the way for more secure, scalable, and adaptive IoT management solutions. 

---
# CAPO: Cost-Aware Prompt Optimization 

**Authors**: Tom Zehle, Moritz Schlager, Timo Heiß, Matthias Feurer  

**Link**: [PDF](https://arxiv.org/pdf/2504.16005)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing by solving a wide range of tasks simply guided by a prompt. Yet their performance is highly sensitive to prompt formulation. While automated prompt optimization addresses this challenge by finding optimal prompts, current methods require a substantial number of LLM calls and input tokens, making prompt optimization expensive. We introduce CAPO (Cost-Aware Prompt Optimization), an algorithm that enhances prompt optimization efficiency by integrating AutoML techniques. CAPO is an evolutionary approach with LLMs as operators, incorporating racing to save evaluations and multi-objective optimization to balance performance with prompt length. It jointly optimizes instructions and few-shot examples while leveraging task descriptions for improved robustness. Our extensive experiments across diverse datasets and LLMs demonstrate that CAPO outperforms state-of-the-art discrete prompt optimization methods in 11/15 cases with improvements up to 21%p. Our algorithm achieves better performances already with smaller budgets, saves evaluations through racing, and decreases average prompt length via a length penalty, making it both cost-efficient and cost-aware. Even without few-shot examples, CAPO outperforms its competitors and generally remains robust to initial prompts. CAPO represents an important step toward making prompt optimization more powerful and accessible by improving cost-efficiency. 

---
# WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents 

**Authors**: Siyu Zhou, Tianyi Zhou, Yijun Yang, Guodong Long, Deheng Ye, Jing Jiang, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15785)  

**Abstract**: Can we build accurate world models out of large language models (LLMs)? How can world models benefit LLM agents? The gap between the prior knowledge of LLMs and the specified environment's dynamics usually bottlenecks LLMs' performance as world models. To bridge the gap, we propose a training-free "world alignment" that learns an environment's symbolic knowledge complementary to LLMs. The symbolic knowledge covers action rules, knowledge graphs, and scene graphs, which are extracted by LLMs from exploration trajectories and encoded into executable codes to regulate LLM agents' policies. We further propose an RL-free, model-based agent "WALL-E 2.0" through the model-predictive control (MPC) framework. Unlike classical MPC requiring costly optimization on the fly, we adopt an LLM agent as an efficient look-ahead optimizer of future steps' actions by interacting with the neurosymbolic world model. While the LLM agent's strong heuristics make it an efficient planner in MPC, the quality of its planned actions is also secured by the accurate predictions of the aligned world model. They together considerably improve learning efficiency in a new environment. On open-world challenges in Mars (Minecraft like) and ALFWorld (embodied indoor environments), WALL-E 2.0 significantly outperforms existing methods, e.g., surpassing baselines in Mars by 16.1%-51.6% of success rate and by at least 61.7% in score. In ALFWorld, it achieves a new record 98% success rate after only 4 iterations. 

---
# FairTranslate: An English-French Dataset for Gender Bias Evaluation in Machine Translation by Overcoming Gender Binarity 

**Authors**: Fanny Jourdan, Yannick Chevalier, Cécile Favre  

**Link**: [PDF](https://arxiv.org/pdf/2504.15941)  

**Abstract**: Large Language Models (LLMs) are increasingly leveraged for translation tasks but often fall short when translating inclusive language -- such as texts containing the singular 'they' pronoun or otherwise reflecting fair linguistic protocols. Because these challenges span both computational and societal domains, it is imperative to critically evaluate how well LLMs handle inclusive translation with a well-founded framework.
This paper presents FairTranslate, a novel, fully human-annotated dataset designed to evaluate non-binary gender biases in machine translation systems from English to French. FairTranslate includes 2418 English-French sentence pairs related to occupations, annotated with rich metadata such as the stereotypical alignment of the occupation, grammatical gender indicator ambiguity, and the ground-truth gender label (male, female, or inclusive).
We evaluate four leading LLMs (Gemma2-2B, Mistral-7B, Llama3.1-8B, Llama3.3-70B) on this dataset under different prompting procedures. Our results reveal substantial biases in gender representation across LLMs, highlighting persistent challenges in achieving equitable outcomes in machine translation. These findings underscore the need for focused strategies and interventions aimed at ensuring fair and inclusive language usage in LLM-based translation systems.
We make the FairTranslate dataset publicly available on Hugging Face, and disclose the code for all experiments on GitHub. 

---
# Dynamic Early Exit in Reasoning Models 

**Authors**: Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15895)  

**Abstract**: Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points (e.g.,"Wait" tokens) and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on multiple reasoning benchmarks MATH-500, AMC 2023, GPQA Diamond and AIME 2024 show that the proposed method is consistently effective on deepseek-series reasoning LLMs, reducing the length of CoT sequences by an average of 31% to 43% while improving accuracy by 1.7% to 5.7%. 

---
# Automated Creativity Evaluation for Large Language Models: A Reference-Based Approach 

**Authors**: Ruizhe Li, Chiwei Zhu, Benfeng Xu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15784)  

**Abstract**: Creative writing is a key capability of Large Language Models (LLMs), with potential applications in literature, storytelling, and various creative domains. However, evaluating the creativity of machine-generated texts remains a significant challenge, as existing methods either rely on costly manual annotations or fail to align closely with human assessments. In this paper, we propose an effective automated evaluation method based on the Torrance Test of Creative Writing (TTCW), which evaluates creativity as product. Our method employs a reference-based Likert-style approach, scoring generated creative texts relative to high-quality reference texts across various tests. Experimental results demonstrate that our method significantly improves the alignment between LLM evaluations and human assessments, achieving a pairwise accuracy of 0.75 (+15\%). 

---
# Benchmarking LLM for Code Smells Detection: OpenAI GPT-4.0 vs DeepSeek-V3 

**Authors**: Ahmed R. Sadik, Siddhata Govind  

**Link**: [PDF](https://arxiv.org/pdf/2504.16027)  

**Abstract**: Determining the most effective Large Language Model for code smell detection presents a complex challenge. This study introduces a structured methodology and evaluation matrix to tackle this issue, leveraging a curated dataset of code samples consistently annotated with known smells. The dataset spans four prominent programming languages Java, Python, JavaScript, and C++; allowing for cross language comparison. We benchmark two state of the art LLMs, OpenAI GPT 4.0 and DeepSeek-V3, using precision, recall, and F1 score as evaluation metrics. Our analysis covers three levels of detail: overall performance, category level performance, and individual code smell type performance. Additionally, we explore cost effectiveness by comparing the token based detection approach of GPT 4.0 with the pattern-matching techniques employed by DeepSeek V3. The study also includes a cost analysis relative to traditional static analysis tools such as SonarQube. The findings offer valuable guidance for practitioners in selecting an efficient, cost effective solution for automated code smell detection 

---
# DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers 

**Authors**: Xuyang Zhong, Haochen Luo, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15827)  

**Abstract**: Existing machine unlearning (MU) approaches exhibit significant sensitivity to hyperparameters, requiring meticulous tuning that limits practical deployment. In this work, we first empirically demonstrate the instability and suboptimal performance of existing popular MU methods when deployed in different scenarios. To address this issue, we propose Dual Optimizer (DualOptim), which incorporates adaptive learning rate and decoupled momentum factors. Empirical and theoretical evidence demonstrates that DualOptim contributes to effective and stable unlearning. Through extensive experiments, we show that DualOptim can significantly boost MU efficacy and stability across diverse tasks, including image classification, image generation, and large language models, making it a versatile approach to empower existing MU algorithms. 

---
# VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation 

**Authors**: Anjiang Wei, Huanmi Tan, Tarun Suresh, Daniel Mendoza, Thiago S. F. X. Teixeira, Ke Wang, Caroline Trippel, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2504.15659)  

**Abstract**: Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of over 125,000 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4% respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. 

---
# A closer look at how large language models trust humans: patterns and biases 

**Authors**: Valeria Lerman, Yaniv Dover  

**Link**: [PDF](https://arxiv.org/pdf/2504.15801)  

**Abstract**: As large language models (LLMs) and LLM-based agents increasingly interact with humans in decision-making contexts, understanding the trust dynamics between humans and AI agents becomes a central concern. While considerable literature studies how humans trust AI agents, it is much less understood how LLM-based agents develop effective trust in humans. LLM-based agents likely rely on some sort of implicit effective trust in trust-related contexts (e.g., evaluating individual loan applications) to assist and affect decision making. Using established behavioral theories, we develop an approach that studies whether LLMs trust depends on the three major trustworthiness dimensions: competence, benevolence and integrity of the human subject. We also study how demographic variables affect effective trust. Across 43,200 simulated experiments, for five popular language models, across five different scenarios we find that LLM trust development shows an overall similarity to human trust development. We find that in most, but not all cases, LLM trust is strongly predicted by trustworthiness, and in some cases also biased by age, religion and gender, especially in financial scenarios. This is particularly true for scenarios common in the literature and for newer models. While the overall patterns align with human-like mechanisms of effective trust formation, different models exhibit variation in how they estimate trust; in some cases, trustworthiness and demographic factors are weak predictors of effective trust. These findings call for a better understanding of AI-to-human trust dynamics and monitoring of biases and trust development patterns to prevent unintended and potentially harmful outcomes in trust-sensitive applications of AI. 

---
# DR.FIX: Automatically Fixing Data Races at Industry Scale 

**Authors**: Farnaz Behrang, Zhizhou Zhang, Georgian-Vlad Saioc, Peng Liu, Milind Chabbi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15637)  

**Abstract**: Data races are a prevalent class of concurrency bugs in shared-memory parallel programs, posing significant challenges to software reliability and reproducibility. While there is an extensive body of research on detecting data races and a wealth of practical detection tools across various programming languages, considerably less effort has been directed toward automatically fixing data races at an industrial scale. In large codebases, data races are continuously introduced and exhibit myriad patterns, making automated fixing particularly challenging.
In this paper, we tackle the problem of automatically fixing data races at an industrial scale. We present this http URL, a tool that combines large language models (LLMs) with program analysis to generate fixes for data races in real-world settings, effectively addressing a broad spectrum of racy patterns in complex code contexts. Implemented for Go--the programming language widely used in modern microservice architectures where concurrency is pervasive and data races are this http URL seamlessly integrates into existing development workflows. We detail the design of this http URL and examine how individual design choices influence the quality of the fixes produced. Over the past 18 months, this http URL has been integrated into developer workflows at Uber demonstrating its practical utility. During this period, this http URL produced patches for 224 (55%) from a corpus of 404 data races spanning various categories; 193 of these patches (86%) were accepted by more than a hundred developers via code reviews and integrated into the codebase. 

---
# A Large-scale Class-level Benchmark Dataset for Code Generation with LLMs 

**Authors**: Musfiqur Rahman, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2504.15564)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated promising capabilities in code generation tasks. However, most existing benchmarks focus on isolated functions and fail to capture the complexity of real-world, class-level software structures. To address this gap, we introduce a large-scale, Python class-level dataset curated from $13{,}174$ real-world open-source projects. The dataset contains over 842,000 class skeletons, each including class and method signatures, along with associated docstrings when available. We preserve structural and contextual dependencies critical to realistic software development scenarios and enrich the dataset with static code metrics to support downstream analysis. To evaluate the usefulness of this dataset, we use extracted class skeletons as prompts for GPT-4 to generate full class implementations. Results show that the LLM-generated classes exhibit strong lexical and structural similarity to human-written counterparts, with average ROUGE@L, BLEU, and TSED scores of 0.80, 0.59, and 0.73, respectively. These findings confirm that well-structured prompts derived from real-world class skeletons significantly enhance LLM performance in class-level code generation. This dataset offers a valuable resource for benchmarking, training, and improving LLMs in realistic software engineering contexts. 

---
# A Framework for Testing and Adapting REST APIs as LLM Tools 

**Authors**: Jayachandu Bandlamudi, Ritwik Chaudhuri, Neelamadhav Gantayat, Kushal Mukherjee, Prerna Agarwal, Renuka Sindhgatta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2504.15546)  

**Abstract**: Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications. 

---
# Cost-Effective Text Clustering with Large Language Models 

**Authors**: Hongtao Wang, Taiyan Zhang, Renchi Yang, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15640)  

**Abstract**: Text clustering aims to automatically partition a collection of text documents into distinct clusters based on linguistic features. In the literature, this task is usually framed as metric clustering based on text embeddings from pre-trained encoders or a graph clustering problem upon pairwise similarities from an oracle, e.g., a large ML model. Recently, large language models (LLMs) bring significant advancement in this field by offering contextualized text embeddings and highly accurate similarity scores, but meanwhile, present grand challenges to cope with substantial computational and/or financial overhead caused by numerous API-based queries or inference calls to the models.
In response, this paper proposes TECL, a cost-effective framework that taps into the feedback from LLMs for accurate text clustering within a limited budget of queries to LLMs. Under the hood, TECL adopts our EdgeLLM or TriangleLLM to construct must-link/cannot-link constraints for text pairs, and further leverages such constraints as supervision signals input to our weighted constrained clustering approach to generate clusters. Particularly, EdgeLLM (resp. TriangleLLM) enables the identification of informative text pairs (resp. triplets) for querying LLMs via well-thought-out greedy algorithms and accurate extraction of pairwise constraints through carefully-crafted prompting techniques. Our experiments on multiple benchmark datasets exhibit that TECL consistently and considerably outperforms existing solutions in unsupervised text clustering under the same query cost for LLMs. 

---
# Exploring Next Token Prediction in Theory of Mind (ToM) Tasks: Comparative Experiments with GPT-2 and LLaMA-2 AI Models 

**Authors**: Pavan Yadav, Nikhil Khandalkar, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15604)  

**Abstract**: Language models have made significant progress in generating coherent text and predicting next tokens based on input prompts. This study compares the next-token prediction performance of two well-known models: OpenAI's GPT-2 and Meta's Llama-2-7b-chat-hf on Theory of Mind (ToM) tasks. To evaluate their capabilities, we built a dataset from 10 short stories sourced from the Explore ToM Dataset. We enhanced these stories by programmatically inserting additional sentences (infills) using GPT-4, creating variations that introduce different levels of contextual complexity. This setup enables analysis of how increasing context affects model performance. We tested both models under four temperature settings (0.01, 0.5, 1.0, 2.0) and evaluated their ability to predict the next token across three reasoning levels. Zero-order reasoning involves tracking the state, either current (ground truth) or past (memory). First-order reasoning concerns understanding another's mental state (e.g., "Does Anne know the apple is salted?"). Second-order reasoning adds recursion (e.g., "Does Anne think that Charles knows the apple is salted?").
Our results show that adding more infill sentences slightly reduces prediction accuracy, as added context increases complexity and ambiguity. Llama-2 consistently outperforms GPT-2 in prediction accuracy, especially at lower temperatures, demonstrating greater confidence in selecting the most probable token. As reasoning complexity rises, model responses diverge more. Notably, GPT-2 and Llama-2 display greater variability in predictions during first- and second-order reasoning tasks. These findings illustrate how model architecture, temperature, and contextual complexity influence next-token prediction, contributing to a better understanding of the strengths and limitations of current language models. 

---
# Med-CoDE: Medical Critique based Disagreement Evaluation Framework 

**Authors**: Mohit Gupta, Akiko Aizawa, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.15330)  

**Abstract**: The emergence of large language models (LLMs) has significantly influenced numerous fields, including healthcare, by enhancing the capabilities of automated systems to process and generate human-like text. However, despite their advancements, the reliability and accuracy of LLMs in medical contexts remain critical concerns. Current evaluation methods often lack robustness and fail to provide a comprehensive assessment of LLM performance, leading to potential risks in clinical settings. In this work, we propose Med-CoDE, a specifically designed evaluation framework for medical LLMs to address these challenges. The framework leverages a critique-based approach to quantitatively measure the degree of disagreement between model-generated responses and established medical ground truths. This framework captures both accuracy and reliability in medical settings. The proposed evaluation framework aims to fill the existing gap in LLM assessment by offering a systematic method to evaluate the quality and trustworthiness of medical LLMs. Through extensive experiments and case studies, we illustrate the practicality of our framework in providing a comprehensive and reliable evaluation of medical LLMs. 

---
# CUBETESTERAI: Automated JUnit Test Generation using the LLaMA Model 

**Authors**: Daniele Gorla, Shivam Kumar, Pietro Nicolaus Roselli Lorenzini, Alireza Alipourfaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.15286)  

**Abstract**: This paper presents an approach to automating JUnit test generation for Java applications using the Spring Boot framework, leveraging the LLaMA (Large Language Model Architecture) model to enhance the efficiency and accuracy of the testing process. The resulting tool, called CUBETESTERAI, includes a user-friendly web interface and the integration of a CI/CD pipeline using GitLab and Docker. These components streamline the automated test generation process, allowing developers to generate JUnit tests directly from their code snippets with minimal manual intervention. The final implementation executes the LLaMA models through RunPod, an online GPU service, which also enhances the privacy of our tool. Using the advanced natural language processing capabilities of the LLaMA model, CUBETESTERAI is able to generate test cases that provide high code coverage and accurate validation of software functionalities in Java-based Spring Boot applications. Furthermore, it efficiently manages resource-intensive operations and refines the generated tests to address common issues like missing imports and handling of private methods. By comparing CUBETESTERAI with some state-of-the-art tools, we show that our proposal consistently demonstrates competitive and, in many cases, better performance in terms of code coverage in different real-life Java programs. 

---
# High-Throughput LLM inference on Heterogeneous Clusters 

**Authors**: Yi Xiong, Jinqi Huang, Wenjie Huang, Xuebing Yu, Entong Li, Zhixiong Ning, Jinhua Zhou, Li Zeng, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15303)  

**Abstract**: Nowadays, many companies possess various types of AI accelerators, forming heterogeneous clusters. Efficiently leveraging these clusters for high-throughput large language model (LLM) inference services can significantly reduce costs and expedite task processing. However, LLM inference on heterogeneous clusters presents two main challenges. Firstly, different deployment configurations can result in vastly different performance. The number of possible configurations is large, and evaluating the effectiveness of a specific setup is complex. Thus, finding an optimal configuration is not an easy task. Secondly, LLM inference instances within a heterogeneous cluster possess varying processing capacities, leading to different processing speeds for handling inference requests. Evaluating these capacities and designing a request scheduling algorithm that fully maximizes the potential of each instance is challenging. In this paper, we propose a high-throughput inference service system on heterogeneous clusters. First, the deployment configuration is optimized by modeling the resource amount and expected throughput and using the exhaustive search method. Second, a novel mechanism is proposed to schedule requests among instances, which fully considers the different processing capabilities of various instances. Extensive experiments show that the proposed scheduler improves throughput by 122.5% and 33.6% on two heterogeneous clusters, respectively. 

---
# PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models 

**Authors**: Shi Qiu, Shaoyang Guo, Zhuo-Yang Song, Yunbo Sun, Zeyu Cai, Jiashen Wei, Tianyu Luo, Yixuan Yin, Haoxu Zhang, Yi Hu, Chenyang Wang, Chencheng Tang, Haoling Chang, Qi Liu, Ziheng Zhou, Tianyu Zhang, Jingtian Zhang, Zhangyi Liu, Minghao Li, Yuku Zhang, Boxuan Jing, Xianqi Yin, Yutong Ren, Zizhuo Fu, Weike Wang, Xudong Tian, Anqi Lv, Laifu Man, Jianxiang Li, Feiyu Tao, Qihua Sun, Zhou Liang, Yushu Mu, Zhongxuan Li, Jing-Jun Zhang, Shutao Zhang, Xiaotian Li, Xingqi Xia, Jiawei Lin, Zheyu Shen, Jiahang Chen, Qiuhao Xiong, Binran Wang, Fengyuan Wang, Ziyang Ni, Bohan Zhang, Fan Cui, Changkun Shao, Qing-Hong Cao, Ming-xing Luo, Muhan Zhang, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16074)  

**Abstract**: We introduce PHYBench, a novel, high-quality benchmark designed for evaluating reasoning capabilities of large language models (LLMs) in physical contexts. PHYBench consists of 500 meticulously curated physics problems based on real-world physical scenarios, designed to assess the ability of models to understand and reason about realistic physical processes. Covering mechanics, electromagnetism, thermodynamics, optics, modern physics, and advanced physics, the benchmark spans difficulty levels from high school exercises to undergraduate problems and Physics Olympiad challenges. Additionally, we propose the Expression Edit Distance (EED) Score, a novel evaluation metric based on the edit distance between mathematical expressions, which effectively captures differences in model reasoning processes and results beyond traditional binary scoring methods. We evaluate various LLMs on PHYBench and compare their performance with human experts. Our results reveal that even state-of-the-art reasoning models significantly lag behind human experts, highlighting their limitations and the need for improvement in complex physical reasoning scenarios. Our benchmark results and dataset are publicly available at this https URL. 

---
# Honey, I Shrunk the Language Model: Impact of Knowledge Distillation Methods on Performance and Explainability 

**Authors**: Daniel Hendriks, Philipp Spitzer, Niklas Kühl, Gerhard Satzger  

**Link**: [PDF](https://arxiv.org/pdf/2504.16056)  

**Abstract**: Artificial Intelligence (AI) has increasingly influenced modern society, recently in particular through significant advancements in Large Language Models (LLMs). However, high computational and storage demands of LLMs still limit their deployment in resource-constrained environments. Knowledge distillation addresses this challenge by training a small student model from a larger teacher model. Previous research has introduced several distillation methods for both generating training data and for training the student model. Despite their relevance, the effects of state-of-the-art distillation methods on model performance and explainability have not been thoroughly investigated and compared. In this work, we enlarge the set of available methods by applying critique-revision prompting to distillation for data generation and by synthesizing existing methods for training. For these methods, we provide a systematic comparison based on the widely used Commonsense Question-Answering (CQA) dataset. While we measure performance via student model accuracy, we employ a human-grounded study to evaluate explainability. We contribute new distillation methods and their comparison in terms of both performance and explainability. This should further advance the distillation of small language models and, thus, contribute to broader applicability and faster diffusion of LLM technology. 

---
# Exploring Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis 

**Authors**: Luwei Xiao, Rui Mao, Shuai Zhao, Qika Lin, Yanhao Jia, Liang He, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2504.15848)  

**Abstract**: Multimodal aspect-based sentiment classification (MASC) is an emerging task due to an increase in user-generated multimodal content on social platforms, aimed at predicting sentiment polarity toward specific aspect targets (i.e., entities or attributes explicitly mentioned in text-image pairs). Despite extensive efforts and significant achievements in existing MASC, substantial gaps remain in understanding fine-grained visual content and the cognitive rationales derived from semantic content and impressions (cognitive interpretations of emotions evoked by image content). In this study, we present Chimera: a cognitive and aesthetic sentiment causality understanding framework to derive fine-grained holistic features of aspects and infer the fundamental drivers of sentiment expression from both semantic perspectives and affective-cognitive resonance (the synergistic effect between emotional responses and cognitive interpretations). Specifically, this framework first incorporates visual patch features for patch-word alignment. Meanwhile, it extracts coarse-grained visual features (e.g., overall image representation) and fine-grained visual regions (e.g., aspect-related regions) and translates them into corresponding textual descriptions (e.g., facial, aesthetic). Finally, we leverage the sentimental causes and impressions generated by a large language model (LLM) to enhance the model's awareness of sentimental cues evoked by semantic content and affective-cognitive resonance. Experimental results on standard MASC datasets demonstrate the effectiveness of the proposed model, which also exhibits greater flexibility to MASC compared to LLMs such as GPT-4o. We have publicly released the complete implementation and dataset at this https URL 

---
# Exploiting Contextual Knowledge in LLMs through V-usable Information based Layer Enhancement 

**Authors**: Xiaowei Yuan, Zhao Yang, Ziyang Huang, Yequan Wang, Siqi Fan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15630)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet they often struggle with context-faithfulness generations that properly reflect contextual knowledge. While existing approaches focus on enhancing the decoding strategies, they ignore the fundamental mechanism of how contextual information is processed within LLMs' internal states. As a result, LLMs remain limited in their ability to fully leverage contextual knowledge. In this paper, we propose Context-aware Layer Enhancement (CaLE), a novel intervention method that enhances the utilization of contextual knowledge within LLMs' internal representations. By employing V-usable information analysis, CaLE strategically amplifies the growth of contextual information at an optimal layer, thereby enriching representations in the final layer. Our experiments demonstrate that CaLE effectively improves context-faithful generation in Question-Answering tasks, particularly in scenarios involving unknown or conflicting contextual knowledge. 

---
# LLM-based Semantic Augmentation for Harmful Content Detection 

**Authors**: Elyas Meguellati, Assaad Zeghina, Shazia Sadiq, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2504.15548)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated strong performance on simple text classification tasks, frequently under zero-shot settings. However, their efficacy declines when tackling complex social media challenges such as propaganda detection, hateful meme classification, and toxicity identification. Much of the existing work has focused on using LLMs to generate synthetic training data, overlooking the potential of LLM-based text preprocessing and semantic augmentation. In this paper, we introduce an approach that prompts LLMs to clean noisy text and provide context-rich explanations, thereby enhancing training sets without substantial increases in data volume. We systematically evaluate on the SemEval 2024 multi-label Persuasive Meme dataset and further validate on the Google Jigsaw toxic comments and Facebook hateful memes datasets to assess generalizability. Our results reveal that zero-shot LLM classification underperforms on these high-context tasks compared to supervised models. In contrast, integrating LLM-based semantic augmentation yields performance on par with approaches that rely on human-annotated data, at a fraction of the cost. These findings underscore the importance of strategically incorporating LLMs into machine learning (ML) pipeline for social media classification tasks, offering broad implications for combating harmful content online. 

---
# SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning 

**Authors**: Cheng Wen, Tingwei Guo, Shuaijiang Zhao, Wei Zou, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15900)  

**Abstract**: Recent work shows that reinforcement learning(RL) can markedly sharpen the reasoning ability of large language models (LLMs) by prompting them to "think before answering." Yet whether and how these gains transfer to audio-language reasoning remains largely unexplored. We extend the Group-Relative Policy Optimization (GRPO) framework from DeepSeek-R1 to a Large Audio-Language Model (LALM), and construct a 32k sample multiple-choice corpus. Using a two-stage regimen supervised fine-tuning on structured and unstructured chains-of-thought, followed by curriculum-guided GRPO, we systematically compare implicit vs. explicit, and structured vs. free form reasoning under identical architectures. Our structured audio reasoning model, SARI (Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning), achieves a 16.35% improvement in average accuracy over the base model Qwen2-Audio-7B-Instruct. Furthermore, the variant built upon Qwen2.5-Omni reaches state-of-the-art performance of 67.08% on the MMAU test-mini benchmark. Ablation experiments show that on the base model we use: (i) SFT warm-up is important for stable RL training, (ii) structured chains yield more robust generalization than unstructured ones, and (iii) easy-to-hard curricula accelerate convergence and improve final performance. These findings demonstrate that explicit, structured reasoning and curriculum learning substantially enhances audio-language understanding. 

---
# What's the Difference? Supporting Users in Identifying the Effects of Prompt and Model Changes Through Token Patterns 

**Authors**: Michael A. Hedderich, Anyi Wang, Raoyuan Zhao, Florian Eichin, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2504.15815)  

**Abstract**: Prompt engineering for large language models is challenging, as even small prompt perturbations or model changes can significantly impact the generated output texts. Existing evaluation methods, either automated metrics or human evaluation, have limitations, such as providing limited insights or being labor-intensive. We propose Spotlight, a new approach that combines both automation and human analysis. Based on data mining techniques, we automatically distinguish between random (decoding) variations and systematic differences in language model outputs. This process provides token patterns that describe the systematic differences and guide the user in manually analyzing the effects of their prompt and model changes efficiently. We create three benchmarks to quantitatively test the reliability of token pattern extraction methods and demonstrate that our approach provides new insights into established prompt data. From a human-centric perspective, through demonstration studies and a user study, we show that our token pattern approach helps users understand the systematic differences of language model outputs, and we are able to discover relevant differences caused by prompt and model changes (e.g. related to gender or culture), thus supporting the prompt engineering process and human-centric model behavior research. 

---
# Speculative Sampling via Exponential Races 

**Authors**: Szymon Kobus, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2504.15475)  

**Abstract**: Speculative decoding accelerates large language model inference using a smaller draft model. In this paper, we establish a surprising connection between speculative decoding and channel simulation, which aims at simulating a noisy channel using as few bits as possible. This connection allows us to provide an information-theoretic analysis of the speed up that can be achieved by speculative decoding. Leveraging this link, we derive an explicit relation between generation speed-up and the number of tokens $k$ generated by the draft model for large $k$, which serves as an upper bound for all $k$. We also propose a novel speculative decoding method via exponential race ERSD that matches state-of-the-art performance. 

---
# Tell Me What You Know About Sexism: Expert-LLM Interaction Strategies and Co-Created Definitions for Zero-Shot Sexism Detection 

**Authors**: Myrthe Reuver, Indira Sen, Matteo Melis, Gabriella Lapesa  

**Link**: [PDF](https://arxiv.org/pdf/2504.15392)  

**Abstract**: This paper investigates hybrid intelligence and collaboration between researchers of sexism and Large Language Models (LLMs), with a four-component pipeline. First, nine sexism researchers answer questions about their knowledge of sexism and of LLMs. They then participate in two interactive experiments involving an LLM (GPT3.5). The first experiment has experts assessing the model's knowledge about sexism and suitability for use in research. The second experiment tasks them with creating three different definitions of sexism: an expert-written definition, an LLM-written one, and a co-created definition. Lastly, zero-shot classification experiments use the three definitions from each expert in a prompt template for sexism detection, evaluating GPT4o on 2.500 texts sampled from five sexism benchmarks. We then analyze the resulting 67.500 classification decisions. The LLM interactions lead to longer and more complex definitions of sexism. Expert-written definitions on average perform poorly compared to LLM-generated definitions. However, some experts do improve classification performance with their co-created definitions of sexism, also experts who are inexperienced in using LLMs. 

---
# SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation 

**Authors**: Keqi Deng, Wenxi Chen, Xie Chen, Philip C. Woodland  

**Link**: [PDF](https://arxiv.org/pdf/2504.15509)  

**Abstract**: Simultaneous speech translation (SST) outputs translations in parallel with streaming speech input, balancing translation quality and latency. While large language models (LLMs) have been extended to handle the speech modality, streaming remains challenging as speech is prepended as a prompt for the entire generation process. To unlock LLM streaming capability, this paper proposes SimulS2S-LLM, which trains speech LLMs offline and employs a test-time policy to guide simultaneous inference. SimulS2S-LLM alleviates the mismatch between training and inference by extracting boundary-aware speech prompts that allows it to be better matched with text input data. SimulS2S-LLM achieves simultaneous speech-to-speech translation (Simul-S2ST) by predicting discrete output speech tokens and then synthesising output speech using a pre-trained vocoder. An incremental beam search is designed to expand the search space of speech token prediction without increasing latency. Experiments on the CVSS speech data show that SimulS2S-LLM offers a better translation quality-latency trade-off than existing methods that use the same training data, such as improving ASR-BLEU scores by 3 points at similar latency. 

---
# Synergizing RAG and Reasoning: A Systematic Review 

**Authors**: Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming Xue, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15909)  

**Abstract**: Recent breakthroughs in large language models (LLMs), particularly in reasoning capabilities, have propelled Retrieval-Augmented Generation (RAG) to unprecedented levels. By synergizing retrieval mechanisms with advanced reasoning, LLMs can now tackle increasingly complex problems. This paper presents a systematic review of the collaborative interplay between RAG and reasoning, clearly defining "reasoning" within the RAG context. It construct a comprehensive taxonomy encompassing multi-dimensional collaborative objectives, representative paradigms, and technical implementations, and analyze the bidirectional synergy methods. Additionally, we critically evaluate current limitations in RAG assessment, including the absence of intermediate supervision for multi-step reasoning and practical challenges related to cost-risk trade-offs. To bridge theory and practice, we provide practical guidelines tailored to diverse real-world applications. Finally, we identify promising research directions, such as graph-based knowledge integration, hybrid model collaboration, and RL-driven optimization. Overall, this work presents a theoretical framework and practical foundation to advance RAG systems in academia and industry, fostering the next generation of RAG solutions. 

---
# FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation 

**Authors**: Chanyeol Choi, Jihoon Kwon, Jaeseon Ha, Hojun Choi, Chaewoon Kim, Yongjae Lee, Jy-yong Sohn, Alejandro Lopez-Lira  

**Link**: [PDF](https://arxiv.org/pdf/2504.15800)  

**Abstract**: In the fast-paced financial domain, accurate and up-to-date information is critical to addressing ever-evolving market conditions. Retrieving this information correctly is essential in financial Question-Answering (QA), since many language models struggle with factual accuracy in this domain. We present FinDER, an expert-generated dataset tailored for Retrieval-Augmented Generation (RAG) in finance. Unlike existing QA datasets that provide predefined contexts and rely on relatively clear and straightforward queries, FinDER focuses on annotating search-relevant evidence by domain experts, offering 5,703 query-evidence-answer triplets derived from real-world financial inquiries. These queries frequently include abbreviations, acronyms, and concise expressions, capturing the brevity and ambiguity common in the realistic search behavior of professionals. By challenging models to retrieve relevant information from large corpora rather than relying on readily determined contexts, FinDER offers a more realistic benchmark for evaluating RAG systems. We further present a comprehensive evaluation of multiple state-of-the-art retrieval models and Large Language Models, showcasing challenges derived from a realistic benchmark to drive future research on truthful and precise RAG in the financial domain. 

---
# From Reviews to Dialogues: Active Synthesis for Zero-Shot LLM-based Conversational Recommender System 

**Authors**: Rohan Surana, Junda Wu, Zhouhang Xie, Yu Xia, Harald Steck, Dawen Liang, Nathan Kallus, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.15476)  

**Abstract**: Conversational recommender systems (CRS) typically require extensive domain-specific conversational datasets, yet high costs, privacy concerns, and data-collection challenges severely limit their availability. Although Large Language Models (LLMs) demonstrate strong zero-shot recommendation capabilities, practical applications often favor smaller, internally managed recommender models due to scalability, interpretability, and data privacy constraints, especially in sensitive or rapidly evolving domains. However, training these smaller models effectively still demands substantial domain-specific conversational data, which remains challenging to obtain. To address these limitations, we propose an active data augmentation framework that synthesizes conversational training data by leveraging black-box LLMs guided by active learning techniques. Specifically, our method utilizes publicly available non-conversational domain data, including item metadata, user reviews, and collaborative signals, as seed inputs. By employing active learning strategies to select the most informative seed samples, our approach efficiently guides LLMs to generate synthetic, semantically coherent conversational interactions tailored explicitly to the target domain. Extensive experiments validate that conversational data generated by our proposed framework significantly improves the performance of LLM-based CRS models, effectively addressing the challenges of building CRS in no- or low-resource scenarios. 

---
# From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs 

**Authors**: Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15965)  

**Abstract**: Memory is the process of encoding, storing, and retrieving information, allowing humans to retain experiences, knowledge, skills, and facts over time, and serving as the foundation for growth and effective interaction with the world. It plays a crucial role in shaping our identity, making decisions, learning from past experiences, building relationships, and adapting to changes. In the era of large language models (LLMs), memory refers to the ability of an AI system to retain, recall, and use information from past interactions to improve future responses and interactions. Although previous research and reviews have provided detailed descriptions of memory mechanisms, there is still a lack of a systematic review that summarizes and analyzes the relationship between the memory of LLM-driven AI systems and human memory, as well as how we can be inspired by human memory to construct more powerful memory systems. To achieve this, in this paper, we propose a comprehensive survey on the memory of LLM-driven AI systems. In particular, we first conduct a detailed analysis of the categories of human memory and relate them to the memory of AI systems. Second, we systematically organize existing memory-related work and propose a categorization method based on three dimensions (object, form, and time) and eight quadrants. Finally, we illustrate some open problems regarding the memory of current AI systems and outline possible future directions for memory in the era of large language models. 

---
