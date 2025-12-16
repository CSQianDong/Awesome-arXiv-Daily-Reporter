# neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings 

**Authors**: Ojas Pungalia, Rashi Upadhyay, Abhishek Mishra, Abhiram H, Tejasvi Alladi, Sujan Yenuganti, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.13481)  

**Abstract**: Envy is a common human behavior that shapes competitiveness and can alter outcomes in team settings. As large language models (LLMs) increasingly act on behalf of humans in collaborative and competitive workflows, there is a pressing need to evaluate whether and under what conditions they exhibit envy-like preferences. In this paper, we test whether LLMs show envy-like behavior toward each other. We considered two scenarios: (1) A point allocation game that tests whether a model tries to win over its peer. (2) A workplace setting observing behaviour when recognition is unfair. Our findings reveal consistent evidence of envy-like patterns in certain LLMs, with large variation across models and contexts. For instance, GPT-5-mini and Claude-3.7-Sonnet show a clear tendency to pull down the peer model to equalize outcomes, whereas Mistral-Small-3.2-24B instead focuses on maximizing its own individual gains. These results highlight the need to consider competitive dispositions as a safety and design factor in LLM-based multi-agent systems. 

---
# Error-Driven Prompt Optimization for Arithmetic Reasoning 

**Authors**: Árpád Pándy, Róbert Lakatos, András Hajdu  

**Link**: [PDF](https://arxiv.org/pdf/2512.13323)  

**Abstract**: Recent advancements in artificial intelligence have sparked interest in industrial agents capable of supporting analysts in regulated sectors, such as finance and healthcare, within tabular data workflows. A key capability for such systems is performing accurate arithmetic operations on structured data while ensuring sensitive information never leaves secure, on-premises environments. Here, we introduce an error-driven optimization framework for arithmetic reasoning that enhances a Code Generation Agent (CGA), specifically applied to on-premises small language models (SLMs). Through a systematic evaluation of a leading SLM (Qwen3 4B), we find that while the base model exhibits fundamental limitations in arithmetic tasks, our proposed error-driven method, which clusters erroneous predictions to refine prompt-rules iteratively, dramatically improves performance, elevating the model's accuracy to 70.8\%. Our results suggest that developing reliable, interpretable, and industrially deployable AI assistants can be achieved not only through costly fine-tuning but also via systematic, error-driven prompt optimization, enabling small models to surpass larger language models (GPT-3.5 Turbo) in a privacy-compliant manner. 

---
# MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data 

**Authors**: Zhenghao Zhu, Chuxue Cao, Sirui Han, Yuanfeng Song, Xing Chen, Caleb Chen Cao, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.13297)  

**Abstract**: In medical data analysis, extracting deep insights from complex, multi-modal datasets is essential for improving patient care, increasing diagnostic accuracy, and optimizing healthcare operations. However, there is currently a lack of high-quality datasets specifically designed to evaluate the ability of large multi-modal models (LMMs) to discover medical insights. In this paper, we introduce MedInsightBench, the first benchmark that comprises 332 carefully curated medical cases, each annotated with thoughtfully designed insights. This benchmark is intended to evaluate the ability of LMMs and agent frameworks to analyze multi-modal medical image data, including posing relevant questions, interpreting complex findings, and synthesizing actionable insights and recommendations. Our analysis indicates that existing LMMs exhibit limited performance on MedInsightBench, which is primarily attributed to their challenges in extracting multi-step, deep insights and the absence of medical expertise. Therefore, we propose MedInsightAgent, an automated agent framework for medical data analysis, composed of three modules: Visual Root Finder, Analytical Insight Agent, and Follow-up Question Composer. Experiments on MedInsightBench highlight pervasive challenges and demonstrate that MedInsightAgent can improve the performance of general LMMs in medical data insight discovery. 

---
# Behavior and Representation in Large Language Models for Combinatorial Optimization: From Feature Extraction to Algorithm Selection 

**Authors**: Francesca Da Ros, Luca Di Gaspero, Kevin Roitero  

**Link**: [PDF](https://arxiv.org/pdf/2512.13374)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new perspectives for automation in optimization. While several studies have explored how LLMs can generate or solve optimization models, far less is understood about what these models actually learn regarding problem structure or algorithmic behavior. This study investigates how LLMs internally represent combinatorial optimization problems and whether such representations can support downstream decision tasks. We adopt a twofold methodology combining direct querying, which assesses LLM capacity to explicitly extract instance features, with probing analyses that examine whether such information is implicitly encoded within their hidden layers. The probing framework is further extended to a per-instance algorithm selection task, evaluating whether LLM-derived representations can predict the best-performing solver. Experiments span four benchmark problems and three instance representations. Results show that LLMs exhibit moderate ability to recover feature information from problem instances, either through direct querying or probing. Notably, the predictive power of LLM hidden-layer representations proves comparable to that achieved through traditional feature extraction, suggesting that LLMs capture meaningful structural information relevant to optimization performance. 

---
# Socratic Students: Teaching Language Models to Learn by Asking Questions 

**Authors**: Rajeev Bhatt Ambati, Tianyi Niu, Aashu Singh, Shlok Mishra, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2512.13102)  

**Abstract**: Large Language Models (LLMs) excel at static interactions, where they answer user queries by retrieving knowledge encoded in their parameters. However, in many real-world settings, such as educational tutoring or medical assistance, relevant information is not directly available and must be actively acquired through dynamic interactions. An interactive agent would recognize its own uncertainty, ask targeted questions, and retain new knowledge efficiently. Prior work has primarily explored effective ways for a teacher to instruct the student, where the teacher identifies student gaps and provides guidance. In this work, we shift the focus to the student and investigate effective strategies to actively query the teacher in seeking useful information. Across math and coding benchmarks, where baseline student models begin with near-zero performance, we show that student-led approaches consistently yield absolute Pass@k improvements of at least 0.5 over static baselines. To improve question quality, we train students using Direct Preference Optimization (DPO) with guidance from either self or stronger students. We find that this guided training enables smaller models to learn how to ask better questions, further enhancing learning efficiency. 

---
# MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph 

**Authors**: Linjie Mu, Yannian Gu, Zhongzhen Huang, Yakun Zhu, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13510)  

**Abstract**: Large language models with reasoning capabilities have demonstrated impressive performance across a wide range of domains. In clinical applications, a transparent, step-by-step reasoning process provides physicians with strong evidence to support decision-making. While reinforcement learning has effectively enhanced reasoning performance in medical contexts, the clinical reliability of these reasoning processes remains limited because their accuracy and validity are often overlooked during training. To address this gap, we propose MedCEG, a framework that augments medical language models with clinically valid reasoning pathways by explicitly supervising the reasoning process through a Critical Evidence Graph (CEG). We curate a dataset of challenging clinical cases and algorithmically construct a CEG for each sample to represent a high-quality verifiable reasoning pathway. To guide the reasoning process, we introduce a Clinical Reasoning Procedure Reward, which evaluates Node Coverage, Structural Correctness, and Chain Completeness, thereby providing a holistic assessment of reasoning quality. Experimental results show that MedCEG surpasses existing methods in performance while producing clinically valid reasoning chains, representing a solid advancement in reliable medical AI reasoning. The code and models are available at this https URL. 

---
# Can AI Understand What We Cannot Say? Measuring Multilevel Alignment Through Abortion Stigma Across Cognitive, Interpersonal, and Structural Levels 

**Authors**: Anika Sharma, Malavika Mampally, Chidaksh Ravuru, Kandyce Brennan, Neil Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2512.13142)  

**Abstract**: As large language models increasingly mediate stigmatized health decisions, their capacity to genuinely understand complex psychological and physiological phenomena remains poorly evaluated. Can AI understand what we cannot say? We investigate whether LLMs coherently represent abortion stigma across the cognitive, interpersonal, and structural levels where it operates. We systematically tested 627 demographically diverse personas across five leading LLMs using the validated Individual Level Abortion Stigma Scale (ILAS). Our multilevel analysis examined whether models coherently represent stigma at the cognitive level (self-judgment), interpersonal level (anticipated judgment and isolation), and structural level (community condemnation and disclosure patterns), as well as overall stigma. Models fail tests of genuine understanding across all levels. They overestimate interpersonal stigma while underestimating cognitive stigma, assume uniform community condemnation, introduce demographic biases absent from human validation data, miss the empirically validated stigma-secrecy relationship, and contradict themselves within theoretical constructs. These patterns reveal that current alignment approaches ensure appropriate language but not coherent multilevel understanding. This work provides empirical evidence that current LLMs lack coherent multilevel understanding of psychological and physiological constructs. AI safety in high-stakes contexts demands new approaches to design (multilevel coherence), evaluation (continuous auditing), governance and regulation (mandatory audits, accountability, deployment restrictions), and AI literacy in domains where understanding what people cannot say determines whether support helps or harms. 

---
# Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI 

**Authors**: Samarth Sarin, Lovepreet Singh, Bhaskarjit Sarmah, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2512.12686)  

**Abstract**: Agentic memory is emerging as a key enabler for large language models (LLM) to maintain continuity, personalization, and long-term context in extended user interactions, critical capabilities for deploying LLMs as truly interactive and adaptive agents. Agentic memory refers to the memory that provides an LLM with agent-like persistence: the ability to retain and act upon information across conversations, similar to how a human would. We present Memoria, a modular memory framework that augments LLM-based conversational systems with persistent, interpretable, and context-rich memory. Memoria integrates two complementary components: dynamic session-level summarization and a weighted knowledge graph (KG)-based user modelling engine that incrementally captures user traits, preferences, and behavioral patterns as structured entities and relationships. This hybrid architecture enables both short-term dialogue coherence and long-term personalization while operating within the token constraints of modern LLMs. We demonstrate how Memoria enables scalable, personalized conversational artificial intelligence (AI) by bridging the gap between stateless LLM interfaces and agentic memory systems, offering a practical solution for industry applications requiring adaptive and evolving user experiences. 

---
# Feeling the Strength but Not the Source: Partial Introspection in LLMs 

**Authors**: Ely Hahami, Lavik Jain, Ishaan Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2512.12411)  

**Abstract**: Recent work from Anthropic claims that frontier models can sometimes detect and name injected "concepts" represented as activation directions. We test the robustness of these claims. First, we reproduce Anthropic's multi-turn "emergent introspection" result on Meta-Llama-3.1-8B-Instruct, finding that the model identifies and names the injected concept 20 percent of the time under Anthropic's original pipeline, exactly matching their reported numbers and thus showing that introspection is not exclusive to very large or capable models. Second, we systematically vary the inference prompt and find that introspection is fragile: performance collapses on closely related tasks such as multiple-choice identification of the injected concept or different prompts of binary discrimination of whether a concept was injected at all. Third, we identify a contrasting regime of partial introspection: the same model can reliably classify the strength of the coefficient of a normalized injected concept vector (as weak / moderate / strong / very strong) with up to 70 percent accuracy, far above the 25 percent chance baseline. Together, these results provide more evidence for Anthropic's claim that language models effectively compute a function of their baseline, internal representations during introspection; however, these self-reports about those representations are narrow and prompt-sensitive. Our code is available at this https URL. 

---
# Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning 

**Authors**: Enhong Mu, Minami Yoda, Yan Zhang, Mingyue Zhang, Yutaka Matsuno, Jialong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.12706)  

**Abstract**: The widespread adoption of the "Games as a Service" model necessitates frequent content updates, placing immense pressure on quality assurance. In response, automated game testing has been viewed as a promising solution to cope with this demanding release cadence. However, existing automated testing approaches typically create a dichotomy: code-centric methods focus on structural coverage without understanding gameplay context, while player-centric agents validate high-level intent but often fail to cover specific underlying code changes. To bridge this gap, we propose SMART (Structural Mapping for Augmented Reinforcement Testing), a novel framework that synergizes structural verification and functional validation for game update testing. SMART leverages large language models (LLMs) to interpret abstract syntax tree (AST) differences and extract functional intent, constructing a context-aware hybrid reward mechanism. This mechanism guides reinforcement learning agents to sequentially fulfill gameplay goals while adaptively exploring modified code branches. We evaluate SMART on two environments, Overcooked and Minecraft. The results demonstrate that SMART significantly outperforms state-of-the-art baselines; it achieves over 94% branch coverage of modified code, nearly double that of traditional reinforcement learning methods, while maintaining a 98% task completion rate, effectively balancing structural comprehensiveness with functional correctness. 

---
# AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation 

**Authors**: Miriam Horovicz  

**Link**: [PDF](https://arxiv.org/pdf/2512.12597)  

**Abstract**: LLM agents that use external tools can solve complex tasks, but understanding which tools actually contributed to a response remains a blind spot. No existing XAI methods address tool-level explanations. We introduce AgentSHAP, the first framework for explaining tool importance in LLM agents. AgentSHAP is model-agnostic: it treats the agent as a black box and works with any LLM (GPT, Claude, Llama, etc.) without needing access to internal weights or gradients. Using Monte Carlo Shapley values, AgentSHAP tests how an agent responds with different tool subsets and computes fair importance scores based on game theory. Our contributions are: (1) the first explainability method for agent tool attribution, grounded in Shapley values from game theory; (2) Monte Carlo sampling that reduces cost from O(2n) to practical levels; and (3) comprehensive experiments on API-Bank showing that AgentSHAP produces consistent scores across runs, correctly identifies which tools matter, and distinguishes relevant from irrelevant tools. AgentSHAP joins TokenSHAP (for tokens) and PixelSHAP (for image regions) to complete a family of Shapley-based XAI tools for modern generative AI. Code: this https URL. 

---
# Floorplan2Guide: LLM-Guided Floorplan Parsing for BLV Indoor Navigation 

**Authors**: Aydin Ayanzadeh, Tim Oates  

**Link**: [PDF](https://arxiv.org/pdf/2512.12177)  

**Abstract**: Indoor navigation remains a critical challenge for people with visual impairments. The current solutions mainly rely on infrastructure-based systems, which limit their ability to navigate safely in dynamic environments. We propose a novel navigation approach that utilizes a foundation model to transform floor plans into navigable knowledge graphs and generate human-readable navigation instructions. Floorplan2Guide integrates a large language model (LLM) to extract spatial information from architectural layouts, reducing the manual preprocessing required by earlier floorplan parsing methods. Experimental results indicate that few-shot learning improves navigation accuracy in comparison to zero-shot learning on simulated and real-world evaluations. Claude 3.7 Sonnet achieves the highest accuracy among the evaluated models, with 92.31%, 76.92%, and 61.54% on the short, medium, and long routes, respectively, under 5-shot prompting of the MP-1 floor plan. The success rate of graph-based spatial structure is 15.4% higher than that of direct visual reasoning among all models, which confirms that graphical representation and in-context learning enhance navigation performance and make our solution more precise for indoor navigation of Blind and Low Vision (BLV) users. 

---
# The Forecast Critic: Leveraging Large Language Models for Poor Forecast Identification 

**Authors**: Luke Bhan, Hanyu Zhang, Andrew Gordon Wilson, Michael W. Mahoney, Chuck Arvin  

**Link**: [PDF](https://arxiv.org/pdf/2512.12059)  

**Abstract**: Monitoring forecasting systems is critical for customer satisfaction, profitability, and operational efficiency in large-scale retail businesses. We propose The Forecast Critic, a system that leverages Large Language Models (LLMs) for automated forecast monitoring, taking advantage of their broad world knowledge and strong ``reasoning'' capabilities. As a prerequisite for this, we systematically evaluate the ability of LLMs to assess time series forecast quality, focusing on three key questions. (1) Can LLMs be deployed to perform forecast monitoring and identify obviously unreasonable forecasts? (2) Can LLMs effectively incorporate unstructured exogenous features to assess what a reasonable forecast looks like? (3) How does performance vary across model sizes and reasoning capabilities, measured across state-of-the-art LLMs? We present three experiments, including on both synthetic and real-world forecasting data. Our results show that LLMs can reliably detect and critique poor forecasts, such as those plagued by temporal misalignment, trend inconsistencies, and spike errors. The best-performing model we evaluated achieves an F1 score of 0.88, somewhat below human-level performance (F1 score: 0.97). We also demonstrate that multi-modal LLMs can effectively incorporate unstructured contextual signals to refine their assessment of the forecast. Models correctly identify missing or spurious promotional spikes when provided with historical context about past promotions (F1 score: 0.84). Lastly, we demonstrate that these techniques succeed in identifying inaccurate forecasts on the real-world M5 time series dataset, with unreasonable forecasts having an sCRPS at least 10% higher than that of reasonable forecasts. These findings suggest that LLMs, even without domain-specific fine-tuning, may provide a viable and scalable option for automated forecast monitoring and evaluation. 

---
# Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents 

**Authors**: Daniel Platnick, Marjan Alirezaie, Hossein Rahnama  

**Link**: [PDF](https://arxiv.org/pdf/2512.11907)  

**Abstract**: Personalizing Large Language Model (LLM) agents requires conditioning them on user-specific data, creating a critical trade-off between task utility and data disclosure. While the utility of adding user data often exhibits diminishing returns (i.e., submodularity), enabling near-optimal greedy selection, real-world personalization is complicated by structural constraints. These include logical dependencies (e.g., selecting fact A requires fact B), categorical quotas (e.g., select at most one writing style), and hierarchical rules (e.g., select at most two social media preferences, of which at most one can be for a professional network). These constraints violate the assumptions of standard subset selection algorithms. We propose a principled method to formally model such constraints. We introduce a compilation process that transforms a user's knowledge graph with dependencies into a set of abstract macro-facets. Our central result is a proof that common hierarchical and quota-based constraints over these macro-facets form a valid laminar matroid. This theoretical characterization lets us cast structured personalization as submodular maximization under a matroid constraint, enabling greedy with constant-factor guarantees (and (1-1/e) via continuous greedy) for a much richer and more realistic class of problems. 

---
# Causal Strengths and Leaky Beliefs: Interpreting LLM Reasoning via Noisy-OR Causal Bayes Nets 

**Authors**: Hanna Dettki  

**Link**: [PDF](https://arxiv.org/pdf/2512.11909)  

**Abstract**: The nature of intelligence in both humans and machines is a longstanding question. While there is no universally accepted definition, the ability to reason causally is often regarded as a pivotal aspect of intelligence (Lake et al., 2017). Evaluating causal reasoning in LLMs and humans on the same tasks provides hence a more comprehensive understanding of their respective strengths and weaknesses. Our study asks: (Q1) Are LLMs aligned with humans given the \emph{same} reasoning tasks? (Q2) Do LLMs and humans reason consistently at the task level? (Q3) Do they have distinct reasoning signatures?
We answer these by evaluating 20+ LLMs on eleven semantically meaningful causal tasks formalized by a collider graph ($C_1\!\to\!E\!\leftarrow\!C_2$ ) under \emph{Direct} (one-shot number as response = probability judgment of query node being one and \emph{Chain of Thought} (CoT; think first, then provide answer).
Judgments are modeled with a leaky noisy-OR causal Bayes net (CBN) whose parameters $\theta=(b,m_1,m_2,p(C)) \in [0,1]$ include a shared prior $p(C)$;
we select the winning model via AIC between a 3-parameter symmetric causal strength ($m_1{=}m_2$) and 4-parameter asymmetric ($m_1{\neq}m_2$) variant. 

---
# Large-Language Memorization During the Classification of United States Supreme Court Cases 

**Authors**: John E. Ortega, Dhruv D. Joshi, Matt P. Borkowski  

**Link**: [PDF](https://arxiv.org/pdf/2512.13654)  

**Abstract**: Large-language models (LLMs) have been shown to respond in a variety of ways for classification tasks outside of question-answering. LLM responses are sometimes called "hallucinations" since the output is not what is ex pected. Memorization strategies in LLMs are being studied in detail, with the goal of understanding how LLMs respond. We perform a deep dive into a classification task based on United States Supreme Court (SCOTUS) decisions. The SCOTUS corpus is an ideal classification task to study for LLM memory accuracy because it presents significant challenges due to extensive sentence length, complex legal terminology, non-standard structure, and domain-specific vocabulary. Experimentation is performed with the latest LLM fine tuning and retrieval-based approaches, such as parameter-efficient fine-tuning, auto-modeling, and others, on two traditional category-based SCOTUS classification tasks: one with 15 labeled topics and another with 279. We show that prompt-based models with memories, such as DeepSeek, can be more robust than previous BERT-based models on both tasks scoring about 2 points better than previous models not based on prompting. 

---
# Embedding-Based Rankings of Educational Resources based on Learning Outcome Alignment: Benchmarking, Expert Validation, and Learner Performance 

**Authors**: Mohammadreza Molavi, Mohammad Moein, Mohammadreza Tavakoli, Abdolali Faraji, Stefan T. Mol, Gábor Kismihók  

**Link**: [PDF](https://arxiv.org/pdf/2512.13658)  

**Abstract**: As the online learning landscape evolves, the need for personalization is increasingly evident. Although educational resources are burgeoning, educators face challenges selecting materials that both align with intended learning outcomes and address diverse learner needs. Large Language Models (LLMs) are attracting growing interest for their potential to create learning resources that better support personalization, but verifying coverage of intended outcomes still requires human alignment review, which is costly and limits scalability. We propose a framework that supports the cost-effective automation of evaluating alignment between educational resources and intended learning outcomes. Using human-generated materials, we benchmarked LLM-based text-embedding models and found that the most accurate model (Voyage) achieved 79% accuracy in detecting alignment. We then applied the optimal model to LLM-generated resources and, via expert evaluation, confirmed that it reliably assessed correspondence to intended outcomes (83% accuracy). Finally, in a three-group experiment with 360 learners, higher alignment scores were positively related to greater learning performance, chi-squared(2, N = 360) = 15.39, p < 0.001. These findings show that embedding-based alignment scores can facilitate scalable personalization by confirming alignment with learning outcomes, which allows teachers to focus on tailoring content to diverse learner needs. 

---
# A Monad-Based Clause Architecture for Artificial Age Score (AAS) in Large Language Models 

**Authors**: Seyma Yaman Kayadibi  

**Link**: [PDF](https://arxiv.org/pdf/2512.11835)  

**Abstract**: Large language models (LLMs) are often deployed as powerful yet opaque systems, leaving open how their internal memory and "self-like" behavior should be governed in a principled and auditable way. The Artificial Age Score (AAS) was previously introduced and mathematically justified through three theorems that characterise it as a metric of artificial memory aging. Building on this foundation, the present work develops an engineering-oriented, clause-based architecture that imposes law-like constraints on LLM memory and control. Twenty selected monads from Leibniz's Monadology are grouped into six bundles: ontology, dynamics, representation and consciousness, harmony and reason, body and organisation, and teleology, and each bundle is realised as an executable specification on top of the AAS kernel. Across six minimal Python implementations, these clause families are instantiated in numerical experiments acting on channel-level quantities such as recall scores, redundancy, and weights. Each implementation follows a four-step pattern: inputs and setup, clause implementation, numerical results, and implications for LLM design, emphasising that the framework is not only philosophically motivated but also directly implementable. The experiments show that the clause system exhibits bounded and interpretable behavior: AAS trajectories remain continuous and rate-limited, contradictions and unsupported claims trigger explicit penalties, and hierarchical refinement reveals an organic structure in a controlled manner. Dual views and goal-action pairs are aligned by harmony terms, and windowed drift in perfection scores separates sustained improvement from sustained degradation. Overall, the monad-based clause framework uses AAS as a backbone and provides a transparent, code-level blueprint for constraining and analyzing internal dynamics in artificial agents. 

---
# From User Interface to Agent Interface: Efficiency Optimization of UI Representations for LLM Agents 

**Authors**: Dezhi Ran, Zhi Gong, Yuzhe Guo, Mengzhou Wu, Yuan Cao, Haochuan Lu, Hengyu Zhang, Xia Zeng, Gang Cao, Liangchao Yao, Yuetang Deng, Wei Yang, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.13438)  

**Abstract**: While Large Language Model (LLM) agents show great potential for automated UI navigation such as automated UI testing and AI assistants, their efficiency has been largely overlooked. Our motivating study reveals that inefficient UI representation creates a critical performance bottleneck. However, UI representation optimization, formulated as the task of automatically generating programs that transform UI representations, faces two unique challenges. First, the lack of Boolean oracles, which traditional program synthesis uses to decisively validate semantic correctness, poses a fundamental challenge to co-optimization of token efficiency and completeness. Second, the need to process large, complex UI trees as input while generating long, compositional transformation programs, making the search space vast and error-prone. Toward addressing the preceding limitations, we present UIFormer, the first automated optimization framework that synthesizes UI transformation programs by conducting constraint-based optimization with structured decomposition of the complex synthesis task. First, UIFormer restricts the program space using a domain-specific language (DSL) that captures UI-specific operations. Second, UIFormer conducts LLM-based iterative refinement with correctness and efficiency rewards, providing guidance for achieving the efficiency-completeness co-optimization. UIFormer operates as a lightweight plugin that applies transformation programs for seamless integration with existing LLM agents, requiring minimal modifications to their core logic. Evaluations across three UI navigation benchmarks spanning Android and Web platforms with five LLMs demonstrate that UIFormer achieves 48.7% to 55.8% token reduction with minimal runtime overhead while maintaining or improving agent performance. Real-world industry deployment at WeChat further validates the practical impact of UIFormer. 

---
# SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping 

**Authors**: Yu-Chen Lu, Sheng-Feng Yu, Hui-Hsien Weng, Pei-Shuo Wang, Yu-Fang Hu, Liang Hung-Chun, Hung-Yueh Chiang, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.13494)  

**Abstract**: Large language models (LLM) have achieved remarkable performance across a wide range of tasks. However, their substantial parameter sizes pose significant challenges for deployment on edge devices with limited computational and memory resources. Low-rank compression is a promising approach to address this issue, as it reduces both computational and memory costs, making LLM more suitable for resource-constrained environments. Nonetheless, naïve low-rank compression methods require a significant reduction in the retained rank to achieve meaningful memory and computation savings. For a low-rank model, the ranks need to be reduced by more than half to yield efficiency gains. Such aggressive truncation, however, typically results in substantial performance degradation. To address this trade-off, we propose SkipCat, a novel low-rank compression framework that enables the use of higher ranks while achieving the same compression rates. First, we introduce an intra-layer shared low-rank projection method, where multiple matrices that share the same input use a common projection. This reduces redundancy and improves compression efficiency. Second, we propose a block skipping technique that omits computations and memory transfers for selected sub-blocks within the low-rank decomposition. These two techniques jointly enable our compressed model to retain more effective ranks under the same compression budget. Experimental results show that, without any additional fine-tuning, our method outperforms previous low-rank compression approaches by 7% accuracy improvement on zero-shot tasks under the same compression rate. These results highlight the effectiveness of our rank-maximized compression strategy in preserving model performance under tight resource constraints. 

---
# MiniLingua: A Small Open-Source LLM for European Languages 

**Authors**: Anna Aksenova, Boris Zverkov, Nicola Dainese, Alexander Nikitin, Pekka Marttinen  

**Link**: [PDF](https://arxiv.org/pdf/2512.13298)  

**Abstract**: Large language models are powerful but often limited by high computational cost, privacy concerns, and English-centric training. Recent progress demonstrates that small, efficient models with around one billion parameters can deliver strong results and enable on-device use. This paper introduces MiniLingua, a multilingual open-source LLM of one billion parameters trained from scratch for 13 European languages, designed to balance coverage and instruction-following capabilities. Based on evaluation results, the instruction-tuned version of MiniLingua outperforms EuroLLM, a model with a similar training approach but a larger training budget, on summarization, classification and both open- and closed-book question answering. Moreover, it remains competitive with more advanced state-of-the-art models on open-ended generation tasks. We release model weights, tokenizer and source code used for data processing and model training. 

---
# Security and Detectability Analysis of Unicode Text Watermarking Methods Against Large Language Models 

**Authors**: Malte Hellmeier  

**Link**: [PDF](https://arxiv.org/pdf/2512.13325)  

**Abstract**: Securing digital text is becoming increasingly relevant due to the widespread use of large language models. Individuals' fear of losing control over data when it is being used to train such machine learning models or when distinguishing model-generated output from text written by humans. Digital watermarking provides additional protection by embedding an invisible watermark within the data that requires protection. However, little work has been taken to analyze and verify if existing digital text watermarking methods are secure and undetectable by large language models. In this paper, we investigate the security-related area of watermarking and machine learning models for text data. In a controlled testbed of three experiments, ten existing Unicode text watermarking methods were implemented and analyzed across six large language models: GPT-5, GPT-4o, Teuken 7B, Llama 3.3, Claude Sonnet 4, and Gemini 2.5 Pro. The findings of our experiments indicate that, especially the latest reasoning models, can detect a watermarked text. Nevertheless, all models fail to extract the watermark unless implementation details in the form of source code are provided. We discuss the implications for security researchers and practitioners and outline future research opportunities to address security concerns. 

---
# Uncovering the Role of Initial Saliency in U-Shaped Attention Bias: Scaling Initial Token Weight for Enhanced Long-Text Processing 

**Authors**: Zewen Qiang, Sendong Zhao, Haochun Wang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.13109)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on a variety of natural language processing (NLP) tasks. However, they often struggle with long-text sequences due to the ``lost in the middle'' phenomenon. This issue has been shown to arise from a U-shaped attention bias, where attention is disproportionately focused on the beginning and end of a text, leaving the middle section underrepresented. While previous studies have attributed this bias to position encoding, our research first identifies an additional factor: initial saliency. It means that in the attention computation for each token, tokens with higher attention weights relative to the initial token tend to receive more attention in the prediction of the next token. We further find that utilizing this property by scaling attention weight between the initial token and others improves the model's ability to process long contexts, achieving a maximum improvement of 3.6\% in MDQA dataset. Moreover, combining this approach with existing methods to reduce position encoding bias further enhances performance, achieving a maximum improvement of 3.4\% in KV-Retrieval tasks. 

---
# LLM Rationalis? Measuring Bargaining Capabilities of AI Negotiators 

**Authors**: Cheril Shah, Akshit Agarwal, Kanak Garg, Mourad Heddaya  

**Link**: [PDF](https://arxiv.org/pdf/2512.13063)  

**Abstract**: Bilateral negotiation is a complex, context-sensitive task in which human negotiators dynamically adjust anchors, pacing, and flexibility to exploit power asymmetries and informal cues. We introduce a unified mathematical framework for modeling concession dynamics based on a hyperbolic tangent curve, and propose two metrics burstiness tau and the Concession-Rigidity Index (CRI) to quantify the timing and rigidity of offer trajectories. We conduct a large-scale empirical comparison between human negotiators and four state-of-the-art large language models (LLMs) across natural-language and numeric-offers settings, with and without rich market context, as well as six controlled power-asymmetry scenarios. Our results reveal that, unlike humans who smoothly adapt to situations and infer the opponents position and strategies, LLMs systematically anchor at extremes of the possible agreement zone for negotiations and optimize for fixed points irrespective of leverage or context. Qualitative analysis further shows limited strategy diversity and occasional deceptive tactics used by LLMs. Moreover the ability of LLMs to negotiate does not improve with better models. These findings highlight fundamental limitations in current LLM negotiation capabilities and point to the need for models that better internalize opponent reasoning and context-dependent strategy. 

---
# TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning 

**Authors**: Shenzhi Yang, Guangcheng Zhu, Xing Zheng, Yingfan MA, Zhongqi Chen, Bowen Song, Weiqiang Wang, Junbo Zhao, Gang Chen, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13106)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has proven effective in training large reasoning models (LRMs) by leveraging answer-verifiable signals to guide policy optimization, which, however, suffers from high annotation costs. To alleviate this problem, recent work has explored unsupervised RLVR methods that derive rewards solely from the model's internal consistency, such as through entropy and majority voting. While seemingly promising, these methods often suffer from model collapse in the later stages of training, which may arise from the reinforcement of incorrect reasoning patterns in the absence of external supervision. In this work, we investigate a novel semi-supervised RLVR paradigm that utilizes a small labeled set to guide RLVR training on unlabeled samples. Our key insight is that supervised rewards are essential for stabilizing consistency-based training on unlabeled samples, ensuring that only reasoning patterns verified on labeled instances are incorporated into RL training. Technically, we propose an effective policy optimization algorithm, TraPO, that identifies reliable unlabeled samples by matching their learning trajectory similarity to labeled ones. Building on this, TraPO achieves remarkable data efficiency and strong generalization on six widely used mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro). With only 1K labeled and 3K unlabeled samples, TraPO reaches 42.6% average accuracy, surpassing the best unsupervised method trained on 45K unlabeled samples (38.3%). Notably, when using 4K labeled and 12K unlabeled samples, TraPO even outperforms the fully supervised model trained on the full 45K labeled samples on all benchmarks, while using only 10% of the labeled data. The code is available via this https URL. 

---
# SignRAG: A Retrieval-Augmented System for Scalable Zero-Shot Road Sign Recognition 

**Authors**: Minghao Zhu, Zhihao Zhang, Anmol Sidhu, Keith Redmill  

**Link**: [PDF](https://arxiv.org/pdf/2512.12885)  

**Abstract**: Automated road sign recognition is a critical task for intelligent transportation systems, but traditional deep learning methods struggle with the sheer number of sign classes and the impracticality of creating exhaustive labeled datasets. This paper introduces a novel zero-shot recognition framework that adapts the Retrieval-Augmented Generation (RAG) paradigm to address this challenge. Our method first uses a Vision Language Model (VLM) to generate a textual description of a sign from an input image. This description is used to retrieve a small set of the most relevant sign candidates from a vector database of reference designs. Subsequently, a Large Language Model (LLM) reasons over the retrieved candidates to make a final, fine-grained recognition. We validate this approach on a comprehensive set of 303 regulatory signs from the Ohio MUTCD. Experimental results demonstrate the framework's effectiveness, achieving 95.58% accuracy on ideal reference images and 82.45% on challenging real-world road data. This work demonstrates the viability of RAG-based architectures for creating scalable and accurate systems for road sign recognition without task-specific training. 

---
# Information-Consistent Language Model Recommendations through Group Relative Policy Optimization 

**Authors**: Sonal Prabhune, Balaji Padmanabhan, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2512.12858)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in business-critical domains such as finance, education, healthcare, and customer support, where users expect consistent and reliable recommendations. Yet LLMs often exhibit variability when prompts are phrased with minor differences, even when semantically equivalent. Such inconsistency undermines trust, complicates compliance, and disrupts user experience. While personalization is desirable in certain contexts, many enterprise scenarios-such as HR onboarding, customer support, or policy disclosure-require invariant information delivery regardless of phrasing or prior conversational history. Existing approaches, including retrieval-augmented generation (RAG) and temperature tuning, improve factuality or reduce stochasticity but cannot guarantee stability across equivalent prompts. In this paper, we propose a reinforcement learning framework based on Group Relative Policy Optimization (GRPO) to directly optimize for consistency. Unlike prior applications of GRPO, which have been limited to reasoning and code generation, we adapt GRPO to enforce stability of information content across groups of semantically equivalent prompts. We introduce entropy-based helpfulness and stability rewards, treating prompt variants as groups and resetting conversational context to isolate phrasing effects. Experiments on investment and job recommendation tasks show that our GRPO-trained model reduces variability more effectively than fine-tuning or decoding-based baselines. To our knowledge, this is a novel application of GRPO for aligning LLMs toward information consistency, reframing variability not as an acceptable feature of generative diversity but as a correctable flaw in enterprise deployments. 

---
# A Disproof of Large Language Model Consciousness: The Necessity of Continual Learning for Consciousness 

**Authors**: Erik Hoel  

**Link**: [PDF](https://arxiv.org/pdf/2512.12802)  

**Abstract**: The requirements for a falsifiable and non-trivial theory of consciousness significantly constrain such theories. Specifically, recent research on the Unfolding Argument and the Substitution Argument has given us formal tools to analyze requirements for a theory of consciousness. I show via a new Proximity Argument that these requirements especially constrain the potential consciousness of contemporary Large Language Models (LLMs) because of their proximity to systems that are equivalent to LLMs in terms of input/output function; yet, for these functionally equivalent systems, there cannot be any non-trivial theory of consciousness that judges them conscious. This forms the basis of a disproof of contemporary LLM consciousness. I then show a positive result, which is that theories of consciousness based on (or requiring) continual learning do satisfy the stringent formal constraints for a theory of consciousness in humans. Intriguingly, this work supports a hypothesis: If continual learning is linked to consciousness in humans, the current limitations of LLMs (which do not continually learn) are intimately tied to their lack of consciousness. 

---
# Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs: GPT, Gemini, LLaMA 

**Authors**: Hanyu Cai, Binqi Shen, Lier Jin, Lan Hu, Xiaojing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2512.12812)  

**Abstract**: Prompt engineering has emerged as a critical factor influencing large language model (LLM) performance, yet the impact of pragmatic elements such as linguistic tone and politeness remains underexplored, particularly across different model families. In this work, we propose a systematic evaluation framework to examine how interaction tone affects model accuracy and apply it to three recently released and widely available LLMs: GPT-4o mini (OpenAI), Gemini 2.0 Flash (Google DeepMind), and Llama 4 Scout (Meta). Using the MMMLU benchmark, we evaluate model performance under Very Friendly, Neutral, and Very Rude prompt variants across six tasks spanning STEM and Humanities domains, and analyze pairwise accuracy differences with statistical significance testing.
Our results show that tone sensitivity is both model-dependent and domain-specific. Neutral or Very Friendly prompts generally yield higher accuracy than Very Rude prompts, but statistically significant effects appear only in a subset of Humanities tasks, where rude tone reduces accuracy for GPT and Llama, while Gemini remains comparatively tone-insensitive. When performance is aggregated across tasks within each domain, tone effects diminish and largely lose statistical significance. Compared with earlier researches, these findings suggest that dataset scale and coverage materially influence the detection of tone effects. Overall, our study indicates that while interaction tone can matter in specific interpretive settings, modern LLMs are broadly robust to tonal variation in typical mixed-domain use, providing practical guidance for prompt design and model selection in real-world deployments. 

---
# State over Tokens: Characterizing the Role of Reasoning Tokens 

**Authors**: Mosh Levy, Zohar Elyoseph, Shauli Ravfogel, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2512.12777)  

**Abstract**: Large Language Models (LLMs) can generate reasoning tokens before their final answer to boost performance on complex tasks. While these sequences seem like human thought processes, empirical evidence reveals that they are not a faithful explanation of the model's actual reasoning process. To address this gap between appearance and function, we introduce the State over Tokens (SoT) conceptual framework. SoT reframes reasoning tokens not as a linguistic narrative, but as an externalized computational state -- the sole persistent information carrier across the model's stateless generation cycles. This explains how the tokens can drive correct reasoning without being a faithful explanation when read as text and surfaces previously overlooked research questions on these tokens. We argue that to truly understand the process that LLMs do, research must move beyond reading the reasoning tokens as text and focus on decoding them as state. 

---
# Beyond Task Completion: An Assessment Framework for Evaluating Agentic AI Systems 

**Authors**: Sreemaee Akshathala, Bassam Adnan, Mahisha Ramesh, Karthik Vaidhyanathan, Basil Muhammed, Kannan Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2512.12791)  

**Abstract**: Recent advances in agentic AI have shifted the focus from standalone Large Language Models (LLMs) to integrated systems that combine LLMs with tools, memory, and other agents to perform complex tasks. These multi-agent architectures enable coordinated reasoning, planning, and execution across diverse domains, allowing agents to collaboratively automate complex workflows. Despite these advances, evaluation and assessment of LLM agents and the multi-agent systems they constitute remain a fundamental challenge. Although various approaches have been proposed in the software engineering literature for evaluating conventional software components, existing methods for AI-based systems often overlook the non-deterministic nature of models. This non-determinism introduces behavioral uncertainty during execution, yet existing evaluations rely on binary task completion metrics that fail to capture it. Evaluating agentic systems therefore requires examining additional dimensions, including the agent ability to invoke tools, ingest and retrieve memory, collaborate with other agents, and interact effectively with its environment. We propose an end-to-end Agent Assessment Framework with four evaluation pillars encompassing LLMs, Memory, Tools, and Environment. We validate the framework on a representative Autonomous CloudOps use case, where experiments reveal behavioral deviations overlooked by conventional metrics, demonstrating its effectiveness in capturing runtime uncertainties. 

---
# Adaptive Edge-Cloud Inference for Speech-to-Action Systems Using ASR and Large Language Models (ASTA) 

**Authors**: Mohammad Jalili Torkamani, Israt Zarin  

**Link**: [PDF](https://arxiv.org/pdf/2512.12769)  

**Abstract**: Voice-based interaction has emerged as a natural and intuitive modality for controlling IoT devices. However, speech-driven edge devices face a fundamental trade-off between cloud-based solutions, which offer stronger language understanding capabilities at the cost of latency, connectivity dependence, and privacy concerns, and edge-based solutions, which provide low latency and improved privacy but are limited by computational constraints. This paper presents ASTA, an adaptive speech-to-action solution that dynamically routes voice commands between edge and cloud inference to balance performance and system resource utilization. ASTA integrates on-device automatic speech recognition and lightweight offline language-model inference with cloud-based LLM processing, guided by real-time system metrics such as CPU workload, device temperature, and network latency. A metric-aware routing mechanism selects the inference path at runtime, while a rule-based command validation and repair component ensures successful end-to-end command execution. We implemented our solution on an NVIDIA Jetson-based edge platform and evaluated it using a diverse dataset of 80 spoken commands. Experimental results show that ASTA successfully routes all input commands for execution, achieving a balanced distribution between online and offline inference. The system attains an ASR accuracy of 62.5% and generates executable commands without repair for only 47.5% of inputs, highlighting the importance of the repair mechanism in improving robustness. These results suggest that adaptive edge-cloud orchestration is a viable approach for resilient and resource-aware voice-controlled IoT systems. 

---
# Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives 

**Authors**: Aheli Poddar, Saptarshi Sahoo, Sujata Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2512.12620)  

**Abstract**: We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning. 

---
# Coupled Variational Reinforcement Learning for Language Model General Reasoning 

**Authors**: Xueru Wen, Jie Lou, Yanjiang Liu, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Yaojie Lu, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.12576)  

**Abstract**: While reinforcement learning have achieved impressive progress in language model reasoning, they are constrained by the requirement for verifiable rewards. Recent verifier-free RL methods address this limitation by utilizing the intrinsic probabilities of LLMs generating reference answers as reward signals. However, these approaches typically sample reasoning traces conditioned only on the question. This design decouples reasoning-trace sampling from answer information, leading to inefficient exploration and incoherence between traces and final answers. In this paper, we propose \textit{\b{Co}upled \b{V}ariational \b{R}einforcement \b{L}earning} (CoVRL), which bridges variational inference and reinforcement learning by coupling prior and posterior distributions through a hybrid sampling strategy. By constructing and optimizing a composite distribution that integrates these two distributions, CoVRL enables efficient exploration while preserving strong thought-answer coherence. Extensive experiments on mathematical and general reasoning benchmarks show that CoVRL improves performance by 12.4\% over the base model and achieves an additional 2.3\% improvement over strong state-of-the-art verifier-free RL baselines, providing a principled framework for enhancing the general reasoning capabilities of language models. 

---
# ORIBA: Exploring LLM-Driven Role-Play Chatbot as a Creativity Support Tool for Original Character Artists 

**Authors**: Yuqian Sun, Xingyu Li, Shunyu Yao, Noura Howell, Tristan Braud, Chang Hee Lee, Ali Asadipour  

**Link**: [PDF](https://arxiv.org/pdf/2512.12630)  

**Abstract**: Recent advances in Generative AI (GAI) have led to new opportunities for creativity support. However, this technology has raised ethical concerns in the visual artists community. This paper explores how GAI can assist visual artists in developing original characters (OCs) while respecting their creative agency. We present ORIBA, an AI chatbot leveraging large language models (LLMs) to enable artists to role-play with their OCs, focusing on conceptualization (e.g., backstories) while leaving exposition (visual creation) to creators. Through a study with 14 artists, we found ORIBA motivated artists' imaginative engagement, developing multidimensional attributes and stronger bonds with OCs that inspire their creative process. Our contributions include design insights for AI systems that develop from artists' perspectives, demonstrating how LLMs can support cross-modal creativity while preserving creative agency in OC art. This paper highlights the potential of GAI as a neutral, non-visual support that strengthens existing creative practice, without infringing artistic exposition. 

---
# Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based Approaches 

**Authors**: Amirhossein Yousefiramandi, Ciaran Cooney  

**Link**: [PDF](https://arxiv.org/pdf/2512.12677)  

**Abstract**: We explore efficient strategies to fine-tune decoder-only Large Language Models (LLMs) for downstream text classification under resource constraints. Two approaches are investigated: (1) attaching a classification head to a pre-trained causal LLM and fine-tuning on the task (using the LLM's final token embedding as a sequence representation), and (2) instruction-tuning the LLM in a prompt->response format for classification. To enable single-GPU fine-tuning of models up to 8B parameters, we combine 4-bit model quantization with Low-Rank Adaptation (LoRA) for parameter-efficient training. Experiments on two datasets - a proprietary single-label dataset and the public WIPO-Alpha patent dataset (extreme multi-label classification) - show that the embedding-based method significantly outperforms the instruction-tuned method in F1-score, and is very competitive with - even surpassing - fine-tuned domain-specific models (e.g. BERT) on the same tasks. These results demonstrate that directly leveraging the internal representations of causal LLMs, along with efficient fine-tuning techniques, yields impressive classification performance under limited computational resources. We discuss the advantages of each approach while outlining practical guidelines and future directions for optimizing LLM fine-tuning in classification scenarios. 

---
# Human-Inspired Learning for Large Language Models via Obvious Record and Maximum-Entropy Method Discovery 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2512.12608)  

**Abstract**: Large Language Models (LLMs) excel at extracting common patterns from large-scale corpora, yet they struggle with rare, low-resource, or previously unseen scenarios-such as niche hardware deployment issues or irregular IoT device behaviors-because such cases are sparsely represented in training data. Moreover, LLMs rely primarily on implicit parametric memory, which limits their ability to explicitly acquire, recall, and refine methods, causing them to behave predominantly as intuition-driven predictors rather than deliberate, method-oriented learners. Inspired by how humans learn from rare experiences, this paper proposes a human-inspired learning framework that integrates two complementary mechanisms. The first, Obvious Record, explicitly stores cause--result (or question--solution) relationships as symbolic memory, enabling persistent learning even from single or infrequent encounters. The second, Maximum-Entropy Method Discovery, prioritizes and preserves methods with high semantic dissimilarity, allowing the system to capture diverse and underrepresented strategies that are typically overlooked by next-token prediction. Verification on a benchmark of 60 semantically diverse question--solution pairs demonstrates that the proposed entropy-guided approach achieves stronger coverage of unseen questions and significantly greater internal diversity than a random baseline, confirming its effectiveness in discovering more generalizable and human-inspired methods. 

---
# Diverse LLMs vs. Vulnerabilities: Who Detects and Fixes Them Better? 

**Authors**: Arastoo Zibaeirad, Marco Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2512.12536)  

**Abstract**: Large Language Models (LLMs) are increasingly being studied for Software Vulnerability Detection (SVD) and Repair (SVR). Individual LLMs have demonstrated code understanding abilities, but they frequently struggle when identifying complex vulnerabilities and generating fixes.
This study presents DVDR-LLM, an ensemble framework that combines outputs from diverse LLMs to determine whether aggregating multiple models reduces error rates. Our evaluation reveals that DVDR-LLM achieves 10-12% higher detection accuracy compared to the average performance of individual models, with benefits increasing as code complexity grows. For multi-file vulnerabilities, the ensemble approach demonstrates significant improvements in recall (+18%) and F1 score (+11.8%) over individual models. However, the approach raises measurable trade-offs: reducing false positives in verification tasks while simultaneously increasing false negatives in detection tasks, requiring careful decision on the required level of agreement among the LLMs (threshold) for increased performance across different security contexts.
Artifact: this https URL 

---
# Explainable AI as a Double-Edged Sword in Dermatology: The Impact on Clinicians versus The Public 

**Authors**: Xuhai Xu, Haoyu Hu, Haoran Zhang, Will Ke Wang, Reina Wang, Luis R. Soenksen, Omar Badri, Sheharbano Jafry, Elise Burger, Lotanna Nwandu, Apoorva Mehta, Erik P. Duhaime, Asif Qasim, Hause Lin, Janis Pereira, Jonathan Hershon, Paulius Mui, Alejandro A. Gru, Noémie Elhadad, Lena Mamykina, Matthew Groh, Philipp Tschandl, Roxana Daneshjou, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2512.12500)  

**Abstract**: Artificial intelligence (AI) is increasingly permeating healthcare, from physician assistants to consumer applications. Since AI algorithm's opacity challenges human interaction, explainable AI (XAI) addresses this by providing AI decision-making insight, but evidence suggests XAI can paradoxically induce over-reliance or bias. We present results from two large-scale experiments (623 lay people; 153 primary care physicians, PCPs) combining a fairness-based diagnosis AI model and different XAI explanations to examine how XAI assistance, particularly multimodal large language models (LLMs), influences diagnostic performance. AI assistance balanced across skin tones improved accuracy and reduced diagnostic disparities. However, LLM explanations yielded divergent effects: lay users showed higher automation bias - accuracy boosted when AI was correct, reduced when AI erred - while experienced PCPs remained resilient, benefiting irrespective of AI accuracy. Presenting AI suggestions first also led to worse outcomes when the AI was incorrect for both groups. These findings highlight XAI's varying impact based on expertise and timing, underscoring LLMs as a "double-edged sword" in medical AI and informing future human-AI collaborative system design. 

---
# SCIR: A Self-Correcting Iterative Refinement Framework for Enhanced Information Extraction Based on Schema 

**Authors**: Yushen Fang, Jianjun Li, Mingqian Ding, Chang Liu, Xinchi Zou, Wenqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.12337)  

**Abstract**: Although Large language Model (LLM)-powered information extraction (IE) systems have shown impressive capabilities, current fine-tuning paradigms face two major limitations: high training costs and difficulties in aligning with LLM preferences. To address these issues, we propose a novel universal IE paradigm, the Self-Correcting Iterative Refinement (SCIR) framework, along with a Multi-task Bilingual (Chinese-English) Self-Correcting (MBSC) dataset containing over 100,000 entries. The SCIR framework achieves plug-and-play compatibility with existing LLMs and IE systems through its Dual-Path Self-Correcting module and feedback-driven optimization, thereby significantly reducing training costs. Concurrently, the MBSC dataset tackles the challenge of preference alignment by indirectly distilling GPT-4's capabilities into IE result detection models. Experimental results demonstrate that SCIR outperforms state-of-the-art IE methods across three key tasks: named entity recognition, relation extraction, and event extraction, achieving a 5.27 percent average improvement in span-based Micro-F1 while reducing training costs by 87 percent compared to baseline approaches. These advancements not only enhance the flexibility and accuracy of IE systems but also pave the way for lightweight and efficient IE paradigms. 

---
# Diffusion Language Model Inference with Monte Carlo Tree Search 

**Authors**: Zheng Huang, Kiran Ramnath, Yueyan Chen, Aosong Feng, Sangmin Woo, Balasubramaniam Srinivasan, Zhichao Xu, Kang Zhou, Shuai Wang, Haibo Ding, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2512.12168)  

**Abstract**: Diffusion language models (DLMs) have recently emerged as a compelling alternative to autoregressive generation, offering parallel generation and improved global coherence. During inference, DLMs generate text by iteratively denoising masked sequences in parallel; however, determining which positions to unmask and which tokens to commit forms a large combinatorial search problem. Existing inference methods approximate this search using heuristics, which often yield suboptimal decoding paths; other approaches instead rely on additional training to guide token selection. To introduce a principled search mechanism for DLMs inference, we introduce MEDAL, a framework that integrates Monte Carlo Tree SEarch initialization for Diffusion LAnguage Model inference. We employ Monte Carlo Tree Search at the initialization stage to explore promising unmasking trajectories, providing a robust starting point for subsequent refinement. This integration is enabled by restricting the search space to high-confidence actions and prioritizing token choices that improve model confidence over remaining masked positions. Across multiple benchmarks, MEDAL achieves up to 22.0% improvement over existing inference strategies, establishing a new paradigm for search-based inference in diffusion language models. 

---
# The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior 

**Authors**: Erik Larsen  

**Link**: [PDF](https://arxiv.org/pdf/2512.12066)  

**Abstract**: Current safety evaluations of large language models rely on single-shot testing, implicitly assuming that model responses are deterministic and representative of the model's safety alignment. We challenge this assumption by investigating the stability of safety refusal decisions across random seeds and temperature settings. Testing four instruction-tuned models from three families (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B) on 876 harmful prompts across 20 different sampling configurations (4 temperatures x 5 random seeds), we find that 18-28% of prompts exhibit decision flips--the model refuses in some configurations but complies in others--depending on the model. Our Safety Stability Index (SSI) reveals that higher temperatures significantly reduce decision stability (Friedman chi-squared = 44.71, p < 0.001), with mean SSI dropping from 0.951 at temperature 0.0 to 0.896 at temperature 1.0. We validate our findings across all model families using Claude 3.5 Haiku as a unified external judge, achieving 89.0% inter-judge agreement with our primary Llama 70B judge (Cohen's kappa = 0.62). These findings demonstrate that single-shot safety evaluations are insufficient for reliable safety assessment. We show that single-shot evaluation agrees with multi-sample ground truth only 92.4% of the time, and recommend using at least 3 samples per prompt for reliable safety assessment. 

---
# Epistemoverse: Toward an AI-Driven Knowledge Metaverse for Intellectual Heritage Preservation 

**Authors**: Predrag K. Nikolić, Robert Prentner  

**Link**: [PDF](https://arxiv.org/pdf/2512.12201)  

**Abstract**: Large language models (LLMs) have often been characterized as "stochastic parrots" that merely reproduce fragments of their training data. This study challenges that assumption by demonstrating that, when placed in an appropriate dialogical context, LLMs can develop emergent conceptual structures and exhibit interaction-driven (re-)structuring of cognitive interfaces and reflective question-asking. Drawing on the biological principle of cloning and Socrates' maieutic method, we analyze authentic philosophical debates generated among AI-reincarnated philosophers within the interactive art installations of the Syntropic Counterpoints project. By engaging digital counterparts of Aristotle, Nietzsche, Machiavelli, and Sun Tzu in iterative discourse, the study reveals how machine dialogue can give rise to inferential coherence, reflective questioning, and creative synthesis. Based on these findings, we propose the concept of the Epistemoverse--a metaverse of knowledge where human and machine cognition intersect to preserve, reinterpret, and extend intellectual heritage through AI-driven interaction. This framework positions virtual and immersive environments as new spaces for epistemic exchange, digital heritage, and collaborative creativity. 

---
# How AI Agents Follow the Herd of AI? Network Effects, History, and Machine Optimism 

**Authors**: Yu Liu, Wenwen Li, Yifan Dou, Guangnan Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.11943)  

**Abstract**: Understanding decision-making in multi-AI-agent frameworks is crucial for analyzing strategic interactions in network-effect-driven contexts. This study investigates how AI agents navigate network-effect games, where individual payoffs depend on peer participatio--a context underexplored in multi-agent systems despite its real-world prevalence. We introduce a novel workflow design using large language model (LLM)-based agents in repeated decision-making scenarios, systematically manipulating price trajectories (fixed, ascending, descending, random) and network-effect strength. Our key findings include: First, without historical data, agents fail to infer equilibrium. Second, ordered historical sequences (e.g., escalating prices) enable partial convergence under weak network effects but strong effects trigger persistent "AI optimism"--agents overestimate participation despite contradictory evidence. Third, randomized history disrupts convergence entirely, demonstrating that temporal coherence in data shapes LLMs' reasoning, unlike humans. These results highlight a paradigm shift: in AI-mediated systems, equilibrium outcomes depend not just on incentives, but on how history is curated, which is impossible for human. 

---
# FloraForge: LLM-Assisted Procedural Generation of Editable and Analysis-Ready 3D Plant Geometric Models For Agricultural Applications 

**Authors**: Mozhgan Hadadi, Talukder Z. Jubery, Patrick S. Schnable, Arti Singh, Bedrich Benes, Adarsh Krishnamurthy, Baskar Ganapathysubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2512.11925)  

**Abstract**: Accurate 3D plant models are crucial for computational phenotyping and physics-based simulation; however, current approaches face significant limitations. Learning-based reconstruction methods require extensive species-specific training data and lack editability. Procedural modeling offers parametric control but demands specialized expertise in geometric modeling and an in-depth understanding of complex procedural rules, making it inaccessible to domain scientists. We present FloraForge, an LLM-assisted framework that enables domain experts to generate biologically accurate, fully parametric 3D plant models through iterative natural language Plant Refinements (PR), minimizing programming expertise. Our framework leverages LLM-enabled co-design to refine Python scripts that generate parameterized plant geometries as hierarchical B-spline surface representations with botanical constraints with explicit control points and parametric deformation functions. This representation can be easily tessellated into polygonal meshes with arbitrary precision, ensuring compatibility with functional structural plant analysis workflows such as light simulation, computational fluid dynamics, and finite element analysis. We demonstrate the framework on maize, soybean, and mung bean, fitting procedural models to empirical point cloud data through manual refinement of the Plant Descriptor (PD), human-readable files. The pipeline generates dual outputs: triangular meshes for visualization and triangular meshes with additional parametric metadata for quantitative analysis. This approach uniquely combines LLM-assisted template creation, mathematically continuous representations enabling both phenotyping and rendering, and direct parametric control through PD. The framework democratizes sophisticated geometric modeling for plant science while maintaining mathematical rigor. 

---
# An Experience Report on a Pedagogically Controlled, Curriculum-Constrained AI Tutor for SE Education 

**Authors**: Lucia Happe, Dominik Fuchß, Luca Hüttner, Kai Marquardt, Anne Koziolek  

**Link**: [PDF](https://arxiv.org/pdf/2512.11882)  

**Abstract**: The integration of artificial intelligence (AI) into education continues to evoke both promise and skepticism. While past waves of technological optimism often fell short, recent advances in large language models (LLMs) have revived the vision of scalable, individualized tutoring. This paper presents the design and pilot evaluation of RockStartIT Tutor, an AI-powered assistant developed for a digital programming and computational thinking course within the RockStartIT initiative. Powered by GPT-4 via OpenAI's Assistant API, the tutor employs a novel prompting strategy and a modular, semantically tagged knowledge base to deliver context-aware, personalized, and curriculum-constrained support for secondary school students. We evaluated the system using the Technology Acceptance Model (TAM) with 13 students and teachers. Learners appreciated the low-stakes environment for asking questions and receiving scaffolded guidance. Educators emphasized the system's potential to reduce cognitive load during independent tasks and complement classroom teaching. Key challenges include prototype limitations, a small sample size, and the need for long-term studies with the target age group. Our findings highlight a pragmatic approach to AI integration that requires no model training, using structure and prompts to shape behavior. We position AI tutors not as teacher replacements but as enabling tools that extend feedback access, foster inquiry, and support what schools do best: help students learn. 

---
# DynaPURLS: Dynamic Refinement of Part-aware Representations for Skeleton-based Zero-Shot Action Recognition 

**Authors**: Jingmin Zhu, Anqi Zhu, James Bailey, Jun Liu, Hossein Rahmani, Mohammed Bennamoun, Farid Boussaid, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2512.11941)  

**Abstract**: Zero-shot skeleton-based action recognition (ZS-SAR) is fundamentally constrained by prevailing approaches that rely on aligning skeleton features with static, class-level semantics. This coarse-grained alignment fails to bridge the domain shift between seen and unseen classes, thereby impeding the effective transfer of fine-grained visual knowledge. To address these limitations, we introduce \textbf{DynaPURLS}, a unified framework that establishes robust, multi-scale visual-semantic correspondences and dynamically refines them at inference time to enhance generalization. Our framework leverages a large language model to generate hierarchical textual descriptions that encompass both global movements and local body-part dynamics. Concurrently, an adaptive partitioning module produces fine-grained visual representations by semantically grouping skeleton joints. To fortify this fine-grained alignment against the train-test domain shift, DynaPURLS incorporates a dynamic refinement module. During inference, this module adapts textual features to the incoming visual stream via a lightweight learnable projection. This refinement process is stabilized by a confidence-aware, class-balanced memory bank, which mitigates error propagation from noisy pseudo-labels. Extensive experiments on three large-scale benchmark datasets, including NTU RGB+D 60/120 and PKU-MMD, demonstrate that DynaPURLS significantly outperforms prior art, setting new state-of-the-art records. The source code is made publicly available at this https URL 

---
# Assessing Greenspace Attractiveness with ChatGPT, Claude, and Gemini: Do AI Models Reflect Human Perceptions? 

**Authors**: Milad Malekzadeh, Magdalena Biernacka, Elias Willberg, Jussi Torkko, Edyta Łaszkiewicz, Tuuli Toivonen  

**Link**: [PDF](https://arxiv.org/pdf/2512.11827)  

**Abstract**: Understanding greenspace attractiveness is essential for designing livable and inclusive urban environments, yet existing assessment approaches often overlook informal or transient spaces and remain too resource intensive to capture subjective perceptions at scale. This study examines the ability of multimodal large language models (MLLMs), ChatGPT GPT-4o, Claude 3.5 Haiku, and Gemini 2.0 Flash, to assess greenspace attractiveness similarly to humans using Google Street View imagery. We compared model outputs with responses from a geo-questionnaire of residents in Lodz, Poland, across both formal (for example, parks and managed greenspaces) and informal (for example, meadows and wastelands) greenspaces. Survey respondents and models indicated whether each greenspace was attractive or unattractive and provided up to three free text explanations. Analyses examined how often their attractiveness judgments aligned and compared their explanations after classifying them into shared reasoning categories. Results show high AI human agreement for attractive formal greenspaces and unattractive informal spaces, but low alignment for attractive informal and unattractive formal greenspaces. Models consistently emphasized aesthetic and design oriented features, underrepresenting safety, functional infrastructure, and locally embedded qualities valued by survey respondents. While these findings highlight the potential for scalable pre-assessment, they also underscore the need for human oversight and complementary participatory approaches. We conclude that MLLMs can support, but not replace, context sensitive greenspace evaluation in planning practice. 

---
# Enhancing Urban Visual Place Recognition for Crowdsourced Flood Imagery via LLM-Guided Attention 

**Authors**: Fengyi Xu, Jun Ma, Waishan Qiu, Cui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.11811)  

**Abstract**: Crowdsourced street-view imagery from social media provides valuable real-time visual evidence of urban flooding and other crisis events, yet it often lacks reliable geographic metadata for emergency response. Existing Visual Place Recognition (VPR) models exhibit substantial performance degradation when applied to such imagery due to visual distortions and domain shifts inherent in cross-source scenarios. This paper presents VPR-AttLLM, a model-agnostic framework that integrates the semantic reasoning and geospatial knowledge of Large Language Models (LLMs) into established VPR pipelines through attention-guided descriptor enhancement. By leveraging LLMs to identify location-informative regions within the city context and suppress transient visual noise, VPR-AttLLM improves retrieval performance without requiring model retraining or additional data. Comprehensive evaluations are conducted on extended benchmarks including SF-XL enriched with real social-media flood images, synthetic flooding scenarios over established query sets and Mapillary photos, and a new HK-URBAN dataset capturing morphologically distinct cityscapes. Integrating VPR-AttLLM with three state-of-the-art VPR models-CosPlace, EigenPlaces, and SALAD-consistently improves recall performance, yielding relative gains typically between 1-3% and reaching up to 8% on the most challenging real flood imagery. Beyond measurable gains in retrieval accuracy, this study establishes a generalizable paradigm for LLM-guided multimodal fusion in visual retrieval systems. By embedding principles from urban perception theory into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design, strong cross-source robustness, and interpretability highlight its potential for scalable urban monitoring and rapid geo-localization of crowdsourced crisis imagery. 

---
# EMNLP: Educator-role Moral and Normative Large Language Models Profiling 

**Authors**: Yilin Jiang, Mingzi Zhang, Sheng Jin, Zengyi Yu, Xiangjie Kong, Binghao Tu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15250)  

**Abstract**: Simulating Professions (SP) enables Large Language Models (LLMs) to emulate professional roles. However, comprehensive psychological and ethical evaluation in these contexts remains lacking. This paper introduces EMNLP, an Educator-role Moral and Normative LLMs Profiling framework for personality profiling, moral development stage measurement, and ethical risk under soft prompt injection. EMNLP extends existing scales and constructs 88 teacher-specific moral dilemmas, enabling profession-oriented comparison with human teachers. A targeted soft prompt injection set evaluates compliance and vulnerability in teacher SP. Experiments on 14 LLMs show teacher-role LLMs exhibit more idealized and polarized personalities than human teachers, excel in abstract moral reasoning, but struggle with emotionally complex situations. Models with stronger reasoning are more vulnerable to harmful prompt injection, revealing a paradox between capability and safety. The model temperature and other hyperparameters have limited influence except in some risk behaviors. This paper presents the first benchmark to assess ethical and psychological alignment of teacher-role LLMs for educational AI. Resources are available at this https URL. 

---
# Temporal Tokenization Strategies for Event Sequence Modeling with Large Language Models 

**Authors**: Zefang Liu, Nam Nguyen, Yinzhu Quan, Austin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13618)  

**Abstract**: Representing continuous time is a critical and under-explored challenge in modeling temporal event sequences with large language models (LLMs). Various strategies like byte-level representations or calendar tokens have been proposed. However, the optimal approach remains unclear, especially given the diverse statistical distributions of real-world event data, which range from smooth log-normal to discrete, spiky patterns. This paper presents the first empirical study of temporal tokenization for event sequences, comparing distinct encoding strategies: naive numeric strings, high-precision byte-level representations, human-semantic calendar tokens, classic uniform binning, and adaptive residual scalar quantization. We evaluate these strategies by fine-tuning LLMs on real-world datasets that exemplify these diverse distributions. Our analysis reveals that no single strategy is universally superior; instead, prediction performance depends heavily on aligning the tokenizer with the data's statistical properties, with log-based strategies excelling on skewed distributions and human-centric formats proving robust for mixed modalities. 

---
# AIR: Post-training Data Selection for Reasoning via Attention Head Influence 

**Authors**: Jinrui Liu, Jeff Wu, Xuanguang Pan, Gavin Cheung, Shuai Ma, Chongyang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2512.13279)  

**Abstract**: LLMs achieve remarkable multi-step reasoning capabilities, yet effectively transferring these skills via post-training distillation remains challenging. Existing data selection methods, ranging from manual curation to heuristics based on length, entropy, or overall loss, fail to capture the causal importance of individual reasoning steps, limiting distillation efficiency. To address this, we propose Attention Influence for Reasoning (AIR), a principled, unsupervised and training-free framework that leverages mechanistic insights of the retrieval head to select high-value post-training data. AIR first identifies reasoning-critical attention heads of an off-the-shelf model, then constructs a weakened reference model with disabled head influence, and finally quantifies the resulting loss divergence as the Attention Influence Score. This score enables fine-grained assessment at both the step and sample levels, supporting step-level weighted fine-tuning and global sample selection. Experiments across multiple reasoning benchmarks show that AIR consistently improves reasoning accuracy, surpassing heuristic baselines and effectively isolating the most critical steps and samples. Our work establishes a mechanism-driven, data-efficient approach for reasoning distillation in LLMs. 

---
# QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management 

**Authors**: Weizhou Shen, Ziyi Yang, Chenliang Li, Zhiyuan Lu, Miao Peng, Huashan Sun, Yingcheng Shi, Shengyi Liao, Shaopeng Lai, Bo Zhang, Dayiheng Liu, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.12967)  

**Abstract**: We introduce QwenLong-L1.5, a model that achieves superior long-context reasoning capabilities through systematic post-training innovations. The key technical breakthroughs of QwenLong-L1.5 are as follows: (1) Long-Context Data Synthesis Pipeline: We develop a systematic synthesis framework that generates challenging reasoning tasks requiring multi-hop grounding over globally distributed evidence. By deconstructing documents into atomic facts and their underlying relationships, and then programmatically composing verifiable reasoning questions, our approach creates high-quality training data at scale, moving substantially beyond simple retrieval tasks to enable genuine long-range reasoning capabilities. (2) Stabilized Reinforcement Learning for Long-Context Training: To overcome the critical instability in long-context RL, we introduce task-balanced sampling with task-specific advantage estimation to mitigate reward bias, and propose Adaptive Entropy-Controlled Policy Optimization (AEPO) that dynamically regulates exploration-exploitation trade-offs. (3) Memory-Augmented Architecture for Ultra-Long Contexts: Recognizing that even extended context windows cannot accommodate arbitrarily long sequences, we develop a memory management framework with multi-stage fusion RL training that seamlessly integrates single-pass reasoning with iterative memory-based processing for tasks exceeding 4M tokens. Based on Qwen3-30B-A3B-Thinking, QwenLong-L1.5 achieves performance comparable to GPT-5 and Gemini-2.5-Pro on long-context reasoning benchmarks, surpassing its baseline by 9.90 points on average. On ultra-long tasks (1M~4M tokens), QwenLong-L1.5's memory-agent framework yields a 9.48-point gain over the agent baseline. Additionally, the acquired long-context reasoning ability translates to enhanced performance in general domains like scientific reasoning, memory tool using, and extended dialogue. 

---
# Towards Effective Model Editing for LLM Personalization 

**Authors**: Baixiang Huang, Limeng Cui, Jiapeng Liu, Haoran Wang, Jiawei Xu, Zhuiyue Tan, Yutong Chen, Chen Luo, Yi Liu, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2512.13676)  

**Abstract**: Personalization is becoming indispensable for LLMs to align with individual user preferences and needs. Yet current approaches are often computationally expensive, data-intensive, susceptible to catastrophic forgetting, and prone to performance degradation in multi-turn interactions or when handling implicit queries. To address these challenges, we conceptualize personalization as a model editing task and introduce Personalization Editing, a framework that applies localized edits guided by clustered preference representations. This design enables precise preference-aligned updates while preserving overall model capabilities. In addition, existing personalization benchmarks frequently rely on persona-based dialogs between LLMs rather than user-LLM interactions, or focus primarily on stylistic imitation while neglecting information-seeking tasks that require accurate recall of user-specific preferences. We introduce User Preference Question Answering (UPQA), a short-answer QA dataset constructed from in-situ user queries with varying levels of difficulty. Unlike prior benchmarks, UPQA directly evaluates a model's ability to recall and apply specific user preferences. Across experimental settings, Personalization Editing achieves higher editing accuracy and greater computational efficiency than fine-tuning, while outperforming prompting-based baselines in multi-turn conversations and implicit preference questions settings. 

---
# An Open and Reproducible Deep Research Agent for Long-Form Question Answering 

**Authors**: Ikuya Yamada, Wataru Ikeda, Ko Yoshida, Mengyu Ye, Hinata Sugimoto, Masatoshi Suzuki, Hisanori Ozaki, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2512.13059)  

**Abstract**: We present an open deep research system for long-form question answering, selected as a winning system in the text-to-text track of the MMU-RAG competition at NeurIPS 2025. The system combines an open-source large language model (LLM) with an open web search API to perform iterative retrieval, reasoning, and synthesis in real-world open-domain settings. To enhance reasoning quality, we apply preference tuning based on LLM-as-a-judge feedback that evaluates multiple aspects, including clarity, insightfulness, and factuality. Our experimental results show that the proposed method consistently improves answer quality across all three aspects. Our source code is publicly available at this https URL. 

---
# AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning 

**Authors**: Jiaru Zou, Ling Yang, Yunzhe Qi, Sirui Chen, Mengting Ai, Ke Shen, Jingrui He, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13278)  

**Abstract**: Agentic reinforcement learning has advanced large language models (LLMs) to reason through long chain-of-thought trajectories while interleaving external tool use. Existing approaches assume a fixed inventory of tools, limiting LLM agents' adaptability to new or evolving toolsets. We present AutoTool, a framework that equips LLM agents with dynamic tool-selection capabilities throughout their reasoning trajectories. We first construct a 200k dataset with explicit tool-selection rationales across 1,000+ tools and 100+ tasks spanning mathematics, science, code generation, and multimodal reasoning. Building on this data foundation, AutoTool employs a dual-phase optimization pipeline: (i) supervised and RL-based trajectory stabilization for coherent reasoning, and (ii) KL-regularized Plackett-Luce ranking to refine consistent multi-step tool selection. Across ten diverse benchmarks, we train two base models, Qwen3-8B and Qwen2.5-VL-7B, with AutoTool. With fewer parameters, AutoTool consistently outperforms advanced LLM agents and tool-integration methods, yielding average gains of 6.4% in math & science reasoning, 4.5% in search-based QA, 7.7% in code generation, and 6.9% in multimodal understanding. In addition, AutoTool exhibits stronger generalization by dynamically leveraging unseen tools from evolving toolsets during inference. 

---
# Efficient Adaptive Rejection Sampling for Accelerating Speculative Decoding in Large Language Models 

**Authors**: Chendong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.13194)  

**Abstract**: Speculative Decoding is a prominent technique for accelerating the autoregressive inference of large language models (LLMs) by employing a fast draft model to propose candidate token sequences and a large target model to verify them in parallel. However, its core component -- the rejection sampling mechanism -- relies on a fixed, context-independent random threshold. This leads to a significant "random rejection" problem in high-uncertainty generation scenarios, where plausible candidate tokens are frequently rejected due to random chance, undermining inference efficiency. This paper introduces Efficient Adaptive Rejection Sampling (EARS), a novel method that dynamically adjusts the acceptance threshold by incorporating the target model's own predictive uncertainty, measured as \(1 - \max(P_{\mathrm{target}})\). By introducing a tolerance term proportional to this uncertainty, EARS intelligently relaxes the acceptance criterion when the model is uncertain, effectively reducing random rejections while maintaining strict standards when the model is confident. Experiments on creative writing and open-domain QA tasks demonstrate that EARS significantly enhances the efficiency of speculative decoding, achieving up to an 18.12% increase in throughput with a negligible 0.84% accuracy drop on the GSM8K benchmark. The method requires no modifications to model architectures and can be seamlessly integrated into existing speculative decoding frameworks. 

---
# CoDA: A Context-Decoupled Hierarchical Agent with Reinforcement Learning 

**Authors**: Xuanzhang Liu, Jianglun Feng, Zhuoran Zhuang, Junzhe Zhao, Maofei Que, Jieting Li, Dianlei Wang, Hao Tong, Ye Chen, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.12716)  

**Abstract**: Large Language Model (LLM) agents trained with reinforcement learning (RL) show great promise for solving complex, multi-step tasks. However, their performance is often crippled by "Context Explosion", where the accumulation of long text outputs overwhelms the model's context window and leads to reasoning failures. To address this, we introduce CoDA, a Context-Decoupled hierarchical Agent, a simple but effective reinforcement learning framework that decouples high-level planning from low-level execution. It employs a single, shared LLM backbone that learns to operate in two distinct, contextually isolated roles: a high-level Planner that decomposes tasks within a concise strategic context, and a low-level Executor that handles tool interactions in an ephemeral, isolated workspace. We train this unified agent end-to-end using PECO (Planner-Executor Co-Optimization), a reinforcement learning methodology that applies a trajectory-level reward to jointly optimize both roles, fostering seamless collaboration through context-dependent policy updates. Extensive experiments demonstrate that CoDA achieves significant performance improvements over state-of-the-art baselines on complex multi-hop question-answering benchmarks, and it exhibits strong robustness in long-context scenarios, maintaining stable performance while all other baselines suffer severe degradation, thus further validating the effectiveness of our hierarchical design in mitigating context overload. 

---
# Persistent Personas? Role-Playing, Instruction Following, and Safety in Extended Interactions 

**Authors**: Pedro Henrique Luz de Araujo, Michael A. Hedderich, Ali Modarressi, Hinrich Schuetze, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2512.12775)  

**Abstract**: Persona-assigned large language models (LLMs) are used in domains such as education, healthcare, and sociodemographic simulation. Yet, they are typically evaluated only in short, single-round settings that do not reflect real-world usage. We introduce an evaluation protocol that combines long persona dialogues (over 100 rounds) and evaluation datasets to create dialogue-conditioned benchmarks that can robustly measure long-context effects. We then investigate the effects of dialogue length on persona fidelity, instruction-following, and safety of seven state-of-the-art open- and closed-weight LLMs. We find that persona fidelity degrades over the course of dialogues, especially in goal-oriented conversations, where models must sustain both persona fidelity and instruction following. We identify a trade-off between fidelity and instruction following, with non-persona baselines initially outperforming persona-assigned models; as dialogues progress and fidelity fades, persona responses become increasingly similar to baseline responses. Our findings highlight the fragility of persona applications in extended interactions and our work provides a protocol to systematically measure such failures. 

---
# LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases 

**Authors**: Yida Cai, Ranjuexiao Hu, Huiyuan Xie, Chenyang Li, Yun Liu, Yuxiao Ye, Zhenghao Liu, Weixing Shen, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.12643)  

**Abstract**: Legal relations form a highly consequential analytical framework of civil law system, serving as a crucial foundation for resolving disputes and realizing values of the rule of law in judicial practice. However, legal relations in Chinese civil cases remain underexplored in the field of legal artificial intelligence (legal AI), largely due to the absence of comprehensive schemas. In this work, we firstly introduce a comprehensive schema, which contains a hierarchical taxonomy and definitions of arguments, for AI systems to capture legal relations in Chinese civil cases. Based on this schema, we then formulate legal relation extraction task and present LexRel, an expert-annotated benchmark for legal relation extraction in Chinese civil law. We use LexRel to evaluate state-of-the-art large language models (LLMs) on legal relation extractions, showing that current LLMs exhibit significant limitations in accurately identifying civil legal relations. Furthermore, we demonstrate that incorporating legal relations information leads to consistent performance gains on other downstream legal AI tasks. 

---
# The American Ghost in the Machine: How language models align culturally and the effects of cultural prompting 

**Authors**: James Luther, Donald Brown  

**Link**: [PDF](https://arxiv.org/pdf/2512.12488)  

**Abstract**: Culture is the bedrock of human interaction; it dictates how we perceive and respond to everyday interactions. As the field of human-computer interaction grows via the rise of generative Large Language Models (LLMs), the cultural alignment of these models become an important field of study. This work, using the VSM13 International Survey and Hofstede's cultural dimensions, identifies the cultural alignment of popular LLMs (DeepSeek-V3, V3.1, GPT-5, GPT-4.1, GPT-4, Claude Opus 4, Llama 3.1, and Mistral Large). We then use cultural prompting, or using system prompts to shift the cultural alignment of a model to a desired country, to test the adaptability of these models to other cultures, namely China, France, India, Iran, Japan, and the United States. We find that the majority of the eight LLMs tested favor the United States when the culture is not specified, with varying results when prompted for other cultures. When using cultural prompting, seven of the eight models shifted closer to the expected culture. We find that models had trouble aligning with Japan and China, despite two of the models tested originating with the Chinese company DeepSeek. 

---
# NagaNLP: Bootstrapping NLP for Low-Resource Nagamese Creole with Human-in-the-Loop Synthetic Data 

**Authors**: Agniva Maiti, Manya Pandey, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2512.12537)  

**Abstract**: The vast majority of the world's languages, particularly creoles like Nagamese, remain severely under-resourced in Natural Language Processing (NLP), creating a significant barrier to their representation in digital technology. This paper introduces NagaNLP, a comprehensive open-source toolkit for Nagamese, bootstrapped through a novel methodology that relies on LLM-driven but human-validated synthetic data generation. We detail a multi-stage pipeline where an expert-guided LLM (Gemini) generates a candidate corpus, which is then refined and annotated by native speakers. This synthetic-hybrid approach yielded a 10K pair conversational dataset and a high-quality annotated corpus for foundational tasks. To assess the effectiveness of our methodology, we trained both discriminative and generative models. Our fine-tuned XLM-RoBERTa-base model establishes a new benchmark for Nagamese, achieving a 93.81\% accuracy (0.90 F1-Macro) on Part-of-Speech tagging and a 0.75 F1-Macro on Named Entity Recognition, massively outperforming strong zero-shot baselines. Furthermore, we fine-tuned a Llama-3.2-3B Instruct model, named NagaLLaMA, which demonstrates superior performance on conversational tasks, achieving a Perplexity of 3.85, an order of magnitude improvement over its few-shot counterpart (96.76). We release the NagaNLP toolkit, including all datasets, models, and code, providing a foundational resource for a previously underserved language and a reproducible framework for reducing data scarcity in other low-resource contexts. 

---
# Market-Bench: Evaluating Large Language Models on Introductory Quantitative Trading and Market Dynamics 

**Authors**: Abhay Srivastava, Sam Jung, Spencer Mateega  

**Link**: [PDF](https://arxiv.org/pdf/2512.12264)  

**Abstract**: We introduce MARKET-BENCH, a benchmark that evaluates large language models (LLMs) on introductory quantitative trading tasks by asking them to construct executable backtesters from natural-language strategy descriptions and market assumptions. Each instance specifies one of three canonical strategies -- scheduled trading on Microsoft (NASDAQ: MSFT), pairs trading on Coca-Cola (NASDAQ: KO) and Pepsi (NASDAQ: PEP), or delta hedging on MSFT -- and models must produce code whose P\&L, drawdown, and position paths match a verifiable reference implementation. We assess twelve state-of-the-art models using a multi-round pass@k metric that separates structural reliability (whether the backtest runs) from numerical accuracy (mean absolute error of the backtest metrics). While most models reliably execute the simplest strategy (average pass@3 of 0.80), errors vary by orders of magnitude across models and tasks: Gemini 3 Pro and Claude 4.5 Sonnet combine strong reliability with low error on simpler strategies, GPT-5.1 Codex-Max achieves perfect pass@1 on the first two strategies and the lowest best-run error on the easiest task, and Qwen3 Max attains perfect pass@3 yet sometimes produces inaccurate P\&L paths. These results show that current LLMs can scaffold basic trading infrastructure but still struggle to reason robustly about prices, inventory, and risk; we release MARKET-BENCH and a public leaderboard at this https URL. 

---
# Benchmarking Contextual Understanding for In-Car Conversational Systems 

**Authors**: Philipp Habicht, Lev Sorokin, Abdullah Saydemir, Ken E. Friedl, Andrea Stocco  

**Link**: [PDF](https://arxiv.org/pdf/2512.12042)  

**Abstract**: In-Car Conversational Question Answering (ConvQA) systems significantly enhance user experience by enabling seamless voice interactions. However, assessing their accuracy and reliability remains a challenge. This paper explores the use of Large Language Models (LLMs) alongside advanced prompting techniques and agent-based methods to evaluate the extent to which ConvQA system responses adhere to user utterances. The focus lies on contextual understanding and the ability to provide accurate venue recommendations considering user constraints and situational context. To evaluate utterance-response coherence using an LLM, we synthetically generate user utterances accompanied by correct and modified failure-containing system responses. We use input-output, chain-of-thought, self-consistency prompting, and multi-agent prompting techniques with 13 reasoning and non-reasoning LLMs of varying sizes and providers, including OpenAI, DeepSeek, Mistral AI, and Meta. We evaluate our approach on a case study involving restaurant recommendations. The most substantial improvements occur for small non-reasoning models when applying advanced prompting techniques, particularly multi-agent prompting. However, reasoning models consistently outperform non-reasoning models, with the best performance achieved using single-agent prompting with self-consistency. Notably, DeepSeek-R1 reaches an F1-score of 0.99 at a cost of 0.002 USD per request. Overall, the best balance between effectiveness and cost-time efficiency is reached with the non-reasoning model DeepSeek-V3. Our findings show that LLM-based evaluation offers a scalable and accurate alternative to traditional human evaluation for benchmarking contextual understanding in ConvQA systems. 

---
# Can GPT replace human raters? Validity and reliability of machine-generated norms for metaphors 

**Authors**: Veronica Mangiaterra, Hamad Al-Azary, Chiara Barattieri di San Pietro, Paolo Canal, Valentina Bambini  

**Link**: [PDF](https://arxiv.org/pdf/2512.12444)  

**Abstract**: As Large Language Models (LLMs) are increasingly being used in scientific research, the issue of their trustworthiness becomes crucial. In psycholinguistics, LLMs have been recently employed in automatically augmenting human-rated datasets, with promising results obtained by generating ratings for single words. Yet, performance for ratings of complex items, i.e., metaphors, is still unexplored. Here, we present the first assessment of the validity and reliability of ratings of metaphors on familiarity, comprehensibility, and imageability, generated by three GPT models for a total of 687 items gathered from the Italian Figurative Archive and three English studies. We performed a thorough validation in terms of both alignment with human data and ability to predict behavioral and electrophysiological responses. We found that machine-generated ratings positively correlated with human-generated ones. Familiarity ratings reached moderate-to-strong correlations for both English and Italian metaphors, although correlations weakened for metaphors with high sensorimotor load. Imageability showed moderate correlations in English and moderate-to-strong in Italian. Comprehensibility for English metaphors exhibited the strongest correlations. Overall, larger models outperformed smaller ones and greater human-model misalignment emerged with familiarity and imageability. Machine-generated ratings significantly predicted response times and the EEG amplitude, with a strength comparable to human ratings. Moreover, GPT ratings obtained across independent sessions were highly stable. We conclude that GPT, especially larger models, can validly and reliably replace - or augment - human subjects in rating metaphor properties. Yet, LLMs align worse with humans when dealing with conventionality and multimodal aspects of metaphorical meaning, calling for careful consideration of the nature of stimuli. 

---
# HyperEdit: Unlocking Instruction-based Text Editing in LLMs via Hypernetworks 

**Authors**: Yiming Zeng, Jinghan Cao, Zexin Li, Wanhao Yu, Zhankai Ye, Dawei Xiang, Ting Hua, Xin Liu, Shangqian Gao, Tingting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.12544)  

**Abstract**: Instruction-based text editing is increasingly critical for real-world applications such as code editors (e.g., Cursor), but Large Language Models (LLMs) continue to struggle with this task. Unlike free-form generation, editing requires faithfully implementing user instructions while preserving unchanged content, as even minor unintended modifications can break functionality. Existing approaches treat editing as generic text generation, leading to two key failures: they struggle to faithfully align edits with diverse user intents, and they often over-edit unchanged regions. We propose HyperEdit to address both issues. First, we introduce hypernetwork-based dynamic adaptation that generates request-specific parameters, enabling the model to tailor its editing strategy to each instruction. Second, we develop difference-aware regularization that focuses supervision on modified spans, preventing over-editing while ensuring precise, minimal changes. HyperEdit achieves a 9%--30% relative improvement in BLEU on modified regions over state-of-the-art baselines, despite utilizing only 3B parameters. 

---
# VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs 

**Authors**: Avinash Amballa, Yashas Malur Saidutta, Chi-Heng Lin, Vivek Kulkarni, Srinivas Chappidi  

**Link**: [PDF](https://arxiv.org/pdf/2512.12072)  

**Abstract**: Large language models (LLMs) are increasingly being used to generate synthetic datasets for the evaluation and training of downstream models. However, prior work has noted that such generated data lacks diversity. In this paper, we propose Voyager, a novel principled approach to generate diverse datasets. Our approach is iterative and directly optimizes a mathematical quantity that optimizes the diversity of the dataset using the machinery of determinantal point processes. Furthermore, our approach is training-free, applicable to closed-source models, and scalable. In addition to providing theoretical justification for the working of our method, we also demonstrate through comprehensive experiments that Voyager significantly outperforms popular baseline approaches by providing a 1.5-3x improvement in diversity. 

---
# Fine-tuned LLM-based Code Migration Framework 

**Authors**: Oleg Grynets, Vasyl Lyashkevych, Dmytro Baran, Maksym Orliansky, Taras Zelenyy, Markiian Leshchyshyn  

**Link**: [PDF](https://arxiv.org/pdf/2512.13515)  

**Abstract**: The study presents the outcomes of research and experimental validation in the domain of automated codebase migration, with a focus on addressing challenges in transitioning SQL-based systems. The proposed method for migration essentially appears as a framework that leverages the best aspects of traditional software engineering techniques and provides an iterative, scalable, precise and efficient solution for modern database transformations. The central piece of the approach is the integration of a fine-tuned Large Language Model to address critical issues in SQL code conversion, such as syntax mapping, resolving discrepancies between Oracle PL/SQL and PostgreSQL, and optimising database elements such as stored procedures, triggers, views, and overall database logic. Thus, the method involves a trade-off between fine-tuning and prompt engineering. Special attention is given to a fine-tuning approach, which enhances the adaptability and compatibility with migration requirements across the entire database. According to the achieved results, fine-tuning plays a very important role. The study employs targeted evaluation methodologies along with computational metrics to measure the success of iterative conversion cycles. Core innovations include automated SQL feature detection, semi-supervised error analysis and integration of Subject Matter Experts feedback within a systematic migration workflow. The methodology achieves significant reductions in Syntax Error Rates, enhances feature alignment throughout migration iterations, and leverages dataset sampling to ensure continual improvement. By embedding GAI into the migration process, the framework facilitates precise feature mapping, semi-automated error resolution, and data-driven optimisation loops, improving workflow efficiency. 

---
# Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models 

**Authors**: Glenn Zhang, Treasure Mayowa, Jason Fan, Yicheng Fu, Aaron Sandoval, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.11998)  

**Abstract**: Producing trustworthy and reliable Large Language Models (LLMs) has become increasingly important as their usage becomes more widespread. Calibration seeks to achieve this by improving the alignment between the model's confidence and the actual likelihood of its responses being correct or desirable. However, it has been observed that the internal confidence of a model, derived from token probabilities, is not well aligned with its verbalized confidence, leading to misleading results with different calibration methods. In this paper, we propose Direct Confidence Alignment (DCA), a method using Direct Preference Optimization to align an LLM's verbalized confidence with its internal confidence rather than ground-truth accuracy, enhancing model transparency and reliability by ensuring closer alignment between the two confidence measures. We evaluate DCA across multiple open-weight LLMs on a wide range of datasets. To further assess this alignment, we also introduce three new calibration error-based metrics. Our results show that DCA improves alignment metrics on certain model architectures, reducing inconsistencies in a model's confidence expression. However, we also show that it can be ineffective on others, highlighting the need for more model-aware approaches in the pursuit of more interpretable and trustworthy LLMs. 

---
# Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection 

**Authors**: Xuwei Tan, Yao Ma, Xueru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13040)  

**Abstract**: Detecting fraud in financial transactions typically relies on tabular models that demand heavy feature engineering to handle high-dimensional data and offer limited interpretability, making it difficult for humans to understand predictions. Large Language Models (LLMs), in contrast, can produce human-readable explanations and facilitate feature analysis, potentially reducing the manual workload of fraud analysts and informing system refinements. However, they perform poorly when applied directly to tabular fraud detection due to the difficulty of reasoning over many features, the extreme class imbalance, and the absence of contextual information. To bridge this gap, we introduce FinFRE-RAG, a two-stage approach that applies importance-guided feature reduction to serialize a compact subset of numeric/categorical attributes into natural language and performs retrieval-augmented in-context learning over label-aware, instance-level exemplars. Across four public fraud datasets and three families of open-weight LLMs, FinFRE-RAG substantially improves F1/MCC over direct prompting and is competitive with strong tabular baselines in several settings. Although these LLMs still lag behind specialized classifiers, they narrow the performance gap and provide interpretable rationales, highlighting their value as assistive tools in fraud analysis. 

---
# Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space 

**Authors**: Chengzhi Liu, Yuzhe Yang, Yue Fan, Qingyue Wei, Sheng Liu, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.12623)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced cross-modal understanding and reasoning by incorporating Chain-of-Thought (CoT) reasoning in the semantic space. Building upon this, recent studies extend the CoT mechanism to the visual modality, enabling models to integrate visual information during reasoning through external tools or explicit image generation. However, these methods remain dependent on explicit step-by-step reasoning, unstable perception-reasoning interaction and notable computational overhead. Inspired by human cognition, we posit that thinking unfolds not linearly but through the dynamic interleaving of reasoning and perception within the mind. Motivated by this perspective, we propose DMLR, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning. Furthermore, a Dynamic Visual Injection Strategy is introduced, which retrieves the most relevant visual features at each latent think token and updates the set of best visual patches. The updated patches are then injected into latent think token to achieve dynamic visual-textual interleaving. Experiments across seven multimodal reasoning benchmarks and various model architectures demonstrate that DMLR significantly improves reasoning and perception performance while maintaining high inference efficiency. 

---
# ERA-IT: Aligning Semantic Models with Revealed Economic Preference for Real-Time and Explainable Patent Valuation 

**Authors**: Yoo Yongmin, Kim Seungwoo, Liu Jingjiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.12869)  

**Abstract**: Valuing intangible assets under uncertainty remains a critical challenge in the strategic management of technological innovation due to the information asymmetry inherent in high-dimensional technical specifications. Traditional bibliometric indicators, such as citation counts, fail to address this friction in a timely manner due to the systemic latency inherent in data accumulation. To bridge this gap, this study proposes the Economic Reasoning Alignment via Instruction Tuning (ERA-IT) framework. We theoretically conceptualize patent renewal history as a revealed economic preference and leverage it as an objective supervisory signal to align the generative reasoning of Large Language Models (LLMs) with market realities, a process we term Eco-Semantic Alignment. Using a randomly sampled dataset of 10,000 European Patent Office patents across diverse technological domains, we trained the model not only to predict value tiers but also to reverse-engineer the Economic Chain-of-Thought from unstructured text. Empirical results demonstrate that ERA-IT significantly outperforms both conventional econometric models and zero-shot LLMs in predictive accuracy. More importantly, by generating explicit, logically grounded rationales for valuation, the framework serves as a transparent cognitive scaffold for decision-makers, reducing the opacity of black-box AI in high-stakes intellectual property management. 

---
# Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World Engines 

**Authors**: Haoyang Shang, Zhengyang Yan, Xuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.11844)  

**Abstract**: We propose Love First, Know Later: a paradigm shift in computational matching that simulates interactions first, then assesses compatibility. Instead of comparing static profiles, our framework leverages LLMs as text world engines that operate in dual capacity-as persona-driven agents following behavioral policies and as the environment modeling interaction dynamics. We formalize compatibility assessment as a reward-modeling problem: given observed matching outcomes, we learn to extract signals from simulations that predict human preferences. Our key insight is that relationships hinge on responses to critical moments-we translate this observation from relationship psychology into mathematical hypotheses, enabling effective simulation. Theoretically, we prove that as LLM policies better approximate human behavior, the induced matching converges to optimal stable matching. Empirically, we validate on speed dating data for initial chemistry and divorce prediction for long-term stability. This paradigm enables interactive, personalized matching systems where users iteratively refine their agents, unlocking future possibilities for transparent and interactive compatibility assessment. 

---
# Are Large Language Models Really Effective for Training-Free Cold-Start Recommendation? 

**Authors**: Genki Kusano, Kenya Abe, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2512.13001)  

**Abstract**: Recommender systems usually rely on large-scale interaction data to learn from users' past behaviors and make accurate predictions. However, real-world applications often face situations where no training data is available, such as when launching new services or handling entirely new users. In such cases, conventional approaches cannot be applied. This study focuses on training-free recommendation, where no task-specific training is performed, and particularly on \textit{training-free cold-start recommendation} (TFCSR), the more challenging case where the target user has no interactions. Large language models (LLMs) have recently been explored as a promising solution, and numerous studies have been proposed. As the ability of text embedding models (TEMs) increases, they are increasingly recognized as applicable to training-free recommendation, but no prior work has directly compared LLMs and TEMs under identical conditions. We present the first controlled experiments that systematically evaluate these two approaches in the same setting. The results show that TEMs outperform LLM rerankers, and this trend holds not only in cold-start settings but also in warm-start settings with rich interactions. These findings indicate that direct LLM ranking is not the only viable option, contrary to the commonly shared belief, and TEM-based approaches provide a stronger and more scalable basis for training-free recommendation. 

---
# Do Reviews Matter for Recommendations in the Era of Large Language Models? 

**Authors**: Chee Heng Tan, Huiying Zheng, Jing Wang, Zhuoyi Lin, Shaodi Feng, Huijing Zhan, Xiaoli Li, J. Senthilnath  

**Link**: [PDF](https://arxiv.org/pdf/2512.12978)  

**Abstract**: With the advent of large language models (LLMs), the landscape of recommender systems is undergoing a significant transformation. Traditionally, user reviews have served as a critical source of rich, contextual information for enhancing recommendation quality. However, as LLMs demonstrate an unprecedented ability to understand and generate human-like text, this raises the question of whether explicit user reviews remain essential in the era of LLMs. In this paper, we provide a systematic investigation of the evolving role of text reviews in recommendation by comparing deep learning methods and LLM approaches. Particularly, we conduct extensive experiments on eight public datasets with LLMs and evaluate their performance in zero-shot, few-shot, and fine-tuning scenarios. We further introduce a benchmarking evaluation framework for review-aware recommender systems, RAREval, to comprehensively assess the contribution of textual reviews to the recommendation performance of review-aware recommender systems. Our framework examines various scenarios, including the removal of some or all textual reviews, random distortion, as well as recommendation performance in data sparsity and cold-start user settings. Our findings demonstrate that LLMs are capable of functioning as effective review-aware recommendation engines, generally outperforming traditional deep learning approaches, particularly in scenarios characterized by data sparsity and cold-start conditions. In addition, the removal of some or all textual reviews and random distortion does not necessarily lead to declines in recommendation accuracy. These findings motivate a rethinking of how user preference from text reviews can be more effectively leveraged. All code and supplementary materials are available at: this https URL. 

---
# SPAR: Session-based Pipeline for Adaptive Retrieval on Legacy File Systems 

**Authors**: Duy A. Nguyen, Hai H. Do, Minh Doan, Minh N. Do  

**Link**: [PDF](https://arxiv.org/pdf/2512.12938)  

**Abstract**: The ability to extract value from historical data is essential for enterprise decision-making. However, much of this information remains inaccessible within large legacy file systems that lack structured organization and semantic indexing, making retrieval and analysis inefficient and error-prone. We introduce SPAR (Session-based Pipeline for Adaptive Retrieval), a conceptual framework that integrates Large Language Models (LLMs) into a Retrieval-Augmented Generation (RAG) architecture specifically designed for legacy enterprise environments. Unlike conventional RAG pipelines, which require costly construction and maintenance of full-scale vector databases that mirror the entire file system, SPAR employs a lightweight two-stage process: a semantic Metadata Index is first created, after which session-specific vector databases are dynamically generated on demand. This design reduces computational overhead while improving transparency, controllability, and relevance in retrieval. We provide a theoretical complexity analysis comparing SPAR with standard LLM-based RAG pipelines, demonstrating its computational advantages. To validate the framework, we apply SPAR to a synthesized enterprise-scale file system containing a large corpus of biomedical literature, showing improvements in both retrieval effectiveness and downstream model accuracy. Finally, we discuss design trade-offs and outline open challenges for deploying SPAR across diverse enterprise settings. 

---
