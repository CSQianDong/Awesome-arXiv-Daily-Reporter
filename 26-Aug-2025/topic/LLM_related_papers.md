# PerPilot: Personalizing VLM-based Mobile Agents via Memory and Exploration 

**Authors**: Xin Wang, Zhiyao Cui, Hao Li, Ya Zeng, Chenxu Wang, Ruiqi Song, Yihang Chen, Kun Shao, Qiaosheng Zhang, Jinzhuo Liu, Siyue Ren, Shuyue Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18040)  

**Abstract**: Vision language model (VLM)-based mobile agents show great potential for assisting users in performing instruction-driven tasks. However, these agents typically struggle with personalized instructions -- those containing ambiguous, user-specific context -- a challenge that has been largely overlooked in previous research. In this paper, we define personalized instructions and introduce PerInstruct, a novel human-annotated dataset covering diverse personalized instructions across various mobile scenarios. Furthermore, given the limited personalization capabilities of existing mobile agents, we propose PerPilot, a plug-and-play framework powered by large language models (LLMs) that enables mobile agents to autonomously perceive, understand, and execute personalized user instructions. PerPilot identifies personalized elements and autonomously completes instructions via two complementary approaches: memory-based retrieval and reasoning-based exploration. Experimental results demonstrate that PerPilot effectively handles personalized tasks with minimal user intervention and progressively improves its performance with continued use, underscoring the importance of personalization-aware reasoning for next-generation mobile agents. The dataset and code are available at: this https URL 

---
# Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding 

**Authors**: Pu Feng, Size Wang, Yuhong Cao, Junkang Liang, Rongye Shi, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17971)  

**Abstract**: The development and application of large language models (LLM) have demonstrated that foundational models can be utilized to solve a wide array of tasks. However, their performance in multi-agent path finding (MAPF) tasks has been less than satisfactory, with only a few studies exploring this area. MAPF is a complex problem requiring both planning and multi-agent coordination. To improve the performance of LLM in MAPF tasks, we propose a novel framework, LLM-NAR, which leverages neural algorithmic reasoners (NAR) to inform LLM for MAPF. LLM-NAR consists of three key components: an LLM for MAPF, a pre-trained graph neural network-based NAR, and a cross-attention mechanism. This is the first work to propose using a neural algorithmic reasoner to integrate GNNs with the map information for MAPF, thereby guiding LLM to achieve superior performance. LLM-NAR can be easily adapted to various LLM models. Both simulation and real-world experiments demonstrate that our method significantly outperforms existing LLM-based approaches in solving MAPF problems. 

---
# ST-Raptor: LLM-Powered Semi-Structured Table Question Answering 

**Authors**: Zirui Tang, Boyu Niu, Xuanhe Zhou, Boxiu Li, Wei Zhou, Jiannan Wang, Guoliang Li, Xinyi Zhang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18190)  

**Abstract**: Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at this https URL. 

---
# The AI Data Scientist 

**Authors**: Farkhad Akimov, Munachiso Samuel Nwadike, Zangir Iklassov, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.18113)  

**Abstract**: Imagine decision-makers uploading data and, within minutes, receiving clear, actionable insights delivered straight to their fingertips. That is the promise of the AI Data Scientist, an autonomous Agent powered by large language models (LLMs) that closes the gap between evidence and action. Rather than simply writing code or responding to prompts, it reasons through questions, tests ideas, and delivers end-to-end insights at a pace far beyond traditional workflows. Guided by the scientific tenet of the hypothesis, this Agent uncovers explanatory patterns in data, evaluates their statistical significance, and uses them to inform predictive modeling. It then translates these results into recommendations that are both rigorous and accessible. At the core of the AI Data Scientist is a team of specialized LLM Subagents, each responsible for a distinct task such as data cleaning, statistical testing, validation, and plain-language communication. These Subagents write their own code, reason about causality, and identify when additional data is needed to support sound conclusions. Together, they achieve in minutes what might otherwise take days or weeks, enabling a new kind of interaction that makes deep data science both accessible and actionable. 

---
# Teaching LLMs to Think Mathematically: A Critical Study of Decision-Making via Optimization 

**Authors**: Mohammad J. Abdel-Rahman, Yasmeen Alslman, Dania Refai, Amro Saleh, Malik A. Abu Loha, Mohammad Yahya Hamed  

**Link**: [PDF](https://arxiv.org/pdf/2508.18091)  

**Abstract**: This paper investigates the capabilities of large language models (LLMs) in formulating and solving decision-making problems using mathematical programming. We first conduct a systematic review and meta-analysis of recent literature to assess how well LLMs understand, structure, and solve optimization problems across domains. The analysis is guided by critical review questions focusing on learning approaches, dataset designs, evaluation metrics, and prompting strategies. Our systematic evidence is complemented by targeted experiments designed to evaluate the performance of state-of-the-art LLMs in automatically generating optimization models for problems in computer networks. Using a newly constructed dataset, we apply three prompting strategies: Act-as-expert, chain-of-thought, and self-consistency, and evaluate the obtained outputs based on optimality gap, token-level F1 score, and compilation accuracy. Results show promising progress in LLMs' ability to parse natural language and represent symbolic formulations, but also reveal key limitations in accuracy, scalability, and interpretability. These empirical gaps motivate several future research directions, including structured datasets, domain-specific fine-tuning, hybrid neuro-symbolic approaches, modular multi-agent architectures, and dynamic retrieval via chain-of-RAGs. This paper contributes a structured roadmap for advancing LLM capabilities in mathematical programming. 

---
# LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios 

**Authors**: Bingxi Zhao, Lin Geng Foo, Ping Hu, Christian Theobalt, Hossein Rahmani, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17692)  

**Abstract**: Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks. 

---
# FAIRGAMER: Evaluating Biases in the Application of Large Language Models to Video Games 

**Authors**: Bingkang Shi, Jen-tse Huang, Guoyi Li, Xiaodan Zhang, Zhongjiang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17825)  

**Abstract**: Leveraging their advanced capabilities, Large Language Models (LLMs) demonstrate vast application potential in video games--from dynamic scene generation and intelligent NPC interactions to adaptive opponents--replacing or enhancing traditional game mechanics. However, LLMs' trustworthiness in this application has not been sufficiently explored. In this paper, we reveal that the models' inherent social biases can directly damage game balance in real-world gaming environments. To this end, we present FairGamer, the first bias evaluation Benchmark for LLMs in video game scenarios, featuring six tasks and a novel metrics ${D_lstd}$. It covers three key scenarios in games where LLMs' social biases are particularly likely to manifest: Serving as Non-Player Characters, Interacting as Competitive Opponents, and Generating Game Scenes. FairGamer utilizes both reality-grounded and fully fictional game content, covering a variety of video game genres. Experiments reveal: (1) Decision biases directly cause game balance degradation, with Grok-3 (average ${D_lstd}$ score=0.431) exhibiting the most severe degradation; (2) LLMs demonstrate isomorphic social/cultural biases toward both real and virtual world content, suggesting their biases nature may stem from inherent model characteristics. These findings expose critical reliability gaps in LLMs' gaming applications. Our code and data are available at anonymous GitHub this https URL . 

---
# Unraveling the cognitive patterns of Large Language Models through module communities 

**Authors**: Kushal Raj Bhandari, Pin-Yu Chen, Jianxi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18192)  

**Abstract**: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mirror the distributed yet interconnected cognitive organization seen in avian and small mammalian brains. Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity. By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions. 

---
# AgentRAN: An Agentic AI Architecture for Autonomous Control of Open 6G Networks 

**Authors**: Maxime Elkael, Salvatore D'Oro, Leonardo Bonati, Michele Polese, Yunseong Lee, Koichiro Furueda, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2508.17778)  

**Abstract**: The Open RAN movement has catalyzed a transformation toward programmable, interoperable cellular infrastructures. Yet, today's deployments still rely heavily on static control and manual operations. To move beyond this limitation, we introduce AgenRAN, an AI-native, Open RAN-aligned agentic framework that generates and orchestrates a fabric of distributed AI agents based on Natural Language (NL) intents. Unlike traditional approaches that require explicit programming, AgentRAN's LLM-powered agents interpret natural language intents, negotiate strategies through structured conversations, and orchestrate control loops across the network. AgentRAN instantiates a self-organizing hierarchy of agents that decompose complex intents across time scales (from sub-millisecond to minutes), spatial domains (cell to network-wide), and protocol layers (PHY/MAC to RRC). A central innovation is the AI-RAN Factory, an automated synthesis pipeline that observes agent interactions and continuously generates new agents embedding improved control algorithms, effectively transforming the network from a static collection of functions into an adaptive system capable of evolving its own intelligence. We demonstrate AgentRAN through live experiments on 5G testbeds where competing user demands are dynamically balanced through cascading intents. By replacing rigid APIs with NL coordination, AgentRAN fundamentally redefines how future 6G networks autonomously interpret, adapt, and optimize their behavior to meet operator goals. 

---
# Language Models Coupled with Metacognition Can Outperform Reasoning Models 

**Authors**: Vedant Khandelwal, Francesca Rossi, Keerthiram Murugesan, Erik Miehling, Murray Campbell, Karthikeyan Natesan Ramamurthy, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17959)  

**Abstract**: Large language models (LLMs) excel in speed and adaptability across various reasoning tasks, but they often struggle when strict logic or constraint enforcement is required. In contrast, Large Reasoning Models (LRMs) are specifically designed for complex, step-by-step reasoning, although they come with significant computational costs and slower inference times. To address these trade-offs, we employ and generalize the SOFAI (Slow and Fast AI) cognitive architecture into SOFAI-LM, which coordinates a fast LLM with a slower but more powerful LRM through metacognition. The metacognitive module actively monitors the LLM's performance and provides targeted, iterative feedback with relevant examples. This enables the LLM to progressively refine its solutions without requiring the need for additional model fine-tuning. Extensive experiments on graph coloring and code debugging problems demonstrate that our feedback-driven approach significantly enhances the problem-solving capabilities of the LLM. In many instances, it achieves performance levels that match or even exceed those of standalone LRMs while requiring considerably less time. Additionally, when the LLM and feedback mechanism alone are insufficient, we engage the LRM by providing appropriate information collected during the LLM's feedback loop, tailored to the specific characteristics of the problem domain and leads to improved overall performance. Evaluations on two contrasting domains: graph coloring, requiring globally consistent solutions, and code debugging, demanding localized fixes, demonstrate that SOFAI-LM enables LLMs to match or outperform standalone LRMs in accuracy while maintaining significantly lower inference time. 

---
# TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis 

**Authors**: Feng Tian, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17565)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled powerful agent-based applications in finance, particularly for sentiment analysis, financial report comprehension, and stock forecasting. However, existing systems often lack inter-agent coordination, structured self-reflection, and access to high-quality, domain-specific post-training data such as data from trading activities including both market conditions and agent decisions. These data are crucial for agents to understand the market dynamics, improve the quality of decision-making and promote effective coordination. We introduce TradingGroup, a multi-agent trading system designed to address these limitations through a self-reflective architecture and an end-to-end data-synthesis pipeline. TradingGroup consists of specialized agents for news sentiment analysis, financial report interpretation, stock trend forecasting, trading style adaptation, and a trading decision making agent that merges all signals and style preferences to produce buy, sell or hold decisions. Specifically, we design self-reflection mechanisms for the stock forecasting, style, and decision-making agents to distill past successes and failures for similar reasoning in analogous future scenarios and a dynamic risk-management model to offer configurable dynamic stop-loss and take-profit mechanisms. In addition, TradingGroup embeds an automated data-synthesis and annotation pipeline that generates high-quality post-training data for further improving the agent performance through post-training. Our backtesting experiments across five real-world stock datasets demonstrate TradingGroup's superior performance over rule-based, machine learning, reinforcement learning, and existing LLM-based trading strategies. 

---
# Evaluating Retrieval-Augmented Generation Strategies for Large Language Models in Travel Mode Choice Prediction 

**Authors**: Yiming Xu, Junfeng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17527)  

**Abstract**: Accurately predicting travel mode choice is essential for effective transportation planning, yet traditional statistical and machine learning models are constrained by rigid assumptions, limited contextual reasoning, and reduced generalizability. This study explores the potential of Large Language Models (LLMs) as a more flexible and context-aware approach to travel mode choice prediction, enhanced by Retrieval-Augmented Generation (RAG) to ground predictions in empirical data. We develop a modular framework for integrating RAG into LLM-based travel mode choice prediction and evaluate four retrieval strategies: basic RAG, RAG with balanced retrieval, RAG with a cross-encoder for re-ranking, and RAG with balanced retrieval and cross-encoder for re-ranking. These strategies are tested across three LLM architectures (OpenAI GPT-4o, o4-mini, and o3) to examine the interaction between model reasoning capabilities and retrieval methods. Using the 2023 Puget Sound Regional Household Travel Survey data, we conduct a series of experiments to evaluate model performance. The results demonstrate that RAG substantially enhances predictive accuracy across a range of models. Notably, the GPT-4o model combined with balanced retrieval and cross-encoder re-ranking achieves the highest accuracy of 80.8%, exceeding that of conventional statistical and machine learning baselines. Furthermore, LLM-based models exhibit superior generalization abilities relative to these baselines. Findings highlight the critical interplay between LLM reasoning capabilities and retrieval strategies, demonstrating the importance of aligning retrieval strategies with model capabilities to maximize the potential of LLM-based travel behavior modeling. 

---
# Large Language Models as Universal Predictors? An Empirical Study on Small Tabular Datasets 

**Authors**: Nikolaos Pavlidis, Vasilis Perifanis, Symeon Symeonidis, Pavlos S. Efraimidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.17391)  

**Abstract**: Large Language Models (LLMs), originally developed for natural language processing (NLP), have demonstrated the potential to generalize across modalities and domains. With their in-context learning (ICL) capabilities, LLMs can perform predictive tasks over structured inputs without explicit fine-tuning on downstream tasks. In this work, we investigate the empirical function approximation capability of LLMs on small-scale structured datasets for classification, regression and clustering tasks. We evaluate the performance of state-of-the-art LLMs (GPT-5, GPT-4o, GPT-o3, Gemini-2.5-Flash, DeepSeek-R1) under few-shot prompting and compare them against established machine learning (ML) baselines, including linear models, ensemble methods and tabular foundation models (TFMs). Our results show that LLMs achieve strong performance in classification tasks under limited data availability, establishing practical zero-training baselines. In contrast, the performance in regression with continuous-valued outputs is poor compared to ML models, likely because regression demands outputs in a large (often infinite) space, and clustering results are similarly limited, which we attribute to the absence of genuine ICL in this setting. Nonetheless, this approach enables rapid, low-overhead data exploration and offers a viable alternative to traditional ML pipelines in business intelligence and exploratory analytics contexts. We further analyze the influence of context size and prompt structure on approximation quality, identifying trade-offs that affect predictive performance. Our findings suggest that LLMs can serve as general-purpose predictive engines for structured data, with clear strengths in classification and significant limitations in regression and clustering. 

---
# Large Language Model-Based Automatic Formulation for Stochastic Optimization Models 

**Authors**: Amirreza Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17200)  

**Abstract**: This paper presents the first integrated systematic study on the performance of large language models (LLMs), specifically ChatGPT, to automatically formulate and solve stochastic optimiza- tion problems from natural language descriptions. Focusing on three key categories, joint chance- constrained models, individual chance-constrained models, and two-stage stochastic linear programs (SLP-2), we design several prompts that guide ChatGPT through structured tasks using chain-of- thought and modular reasoning. We introduce a novel soft scoring metric that evaluates the struc- tural quality and partial correctness of generated models, addressing the limitations of canonical and execution-based accuracy. Across a diverse set of stochastic problems, GPT-4-Turbo outperforms other models in partial score, variable matching, and objective accuracy, with cot_s_instructions and agentic emerging as the most effective prompting strategies. Our findings reveal that with well-engineered prompts and multi-agent collaboration, LLMs can facilitate specially stochastic formulations, paving the way for intelligent, language-driven modeling pipelines in stochastic opti- mization. 

---
# Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs 

**Authors**: Katherine Atwell, Pedram Heydari, Anthony Sicilia, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2508.16846)  

**Abstract**: Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy. 

---
# From reactive to cognitive: brain-inspired spatial intelligence for embodied agents 

**Authors**: Shouwei Ruan, Liyuan Wang, Caixin Kang, Qihui Zhu, Songming Liu, Xingxing Wei, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17198)  

**Abstract**: Spatial cognition enables adaptive goal-directed behavior by constructing internal models of space. Robust biological systems consolidate spatial knowledge into three interconnected forms: \textit{landmarks} for salient cues, \textit{route knowledge} for movement trajectories, and \textit{survey knowledge} for map-like representations. While recent advances in multi-modal large language models (MLLMs) have enabled visual-language reasoning in embodied agents, these efforts lack structured spatial memory and instead operate reactively, limiting their generalization and adaptability in complex real-world environments. Here we present Brain-inspired Spatial Cognition for Navigation (BSC-Nav), a unified framework for constructing and leveraging structured spatial memory in embodied agents. BSC-Nav builds allocentric cognitive maps from egocentric trajectories and contextual cues, and dynamically retrieves spatial knowledge aligned with semantic goals. Integrated with powerful MLLMs, BSC-Nav achieves state-of-the-art efficacy and efficiency across diverse navigation tasks, demonstrates strong zero-shot generalization, and supports versatile embodied behaviors in the real physical world, offering a scalable and biologically grounded path toward general-purpose spatial intelligence. 

---
# Evaluation and LLM-Guided Learning of ICD Coding Rationales 

**Authors**: Mingyang Li, Viktor Schlegel, Tingting Mu, Wuraola Oyewusi, Kai Kang, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2508.16777)  

**Abstract**: Automated clinical coding involves mapping unstructured text from Electronic Health Records (EHRs) to standardized code systems such as the International Classification of Diseases (ICD). While recent advances in deep learning have significantly improved the accuracy and efficiency of ICD coding, the lack of explainability in these models remains a major limitation, undermining trust and transparency. Current explorations about explainability largely rely on attention-based techniques and qualitative assessments by physicians, yet lack systematic evaluation using consistent criteria on high-quality rationale datasets, as well as dedicated approaches explicitly trained to generate rationales for further enhancing explanation. In this work, we conduct a comprehensive evaluation of the explainability of the rationales for ICD coding through two key lenses: faithfulness that evaluates how well explanations reflect the model's actual reasoning and plausibility that measures how consistent the explanations are with human expert judgment. To facilitate the evaluation of plausibility, we construct a new rationale-annotated dataset, offering denser annotations with diverse granularity and aligns better with current clinical practice, and conduct evaluation across three types of rationales of ICD coding. Encouraged by the promising plausibility of LLM-generated rationales for ICD coding, we further propose new rationale learning methods to improve the quality of model-generated rationales, where rationales produced by prompting LLMs with/without annotation examples are used as distant supervision signals. We empirically find that LLM-generated rationales align most closely with those of human experts. Moreover, incorporating few-shot human-annotated examples not only further improves rationale generation but also enhances rationale-learning approaches. 

---
# PowerChain: Automating Distribution Grid Analysis with Agentic AI Workflows 

**Authors**: Emmanuel O. Badmus, Peng Sang, Dimitrios Stamoulis, Amritanshu Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2508.17094)  

**Abstract**: Due to the rapid pace of electrification and decarbonization, distribution grid (DG) operation and planning are becoming more complex, necessitating advanced computational analyses to ensure grid reliability and resilience. State-of-the-art DG analyses rely on disparate workflows of complex models, functions, and data pipelines, which require expert knowledge and are challenging to automate. Many small-scale utilities and cooperatives lack a large R&D workforce and therefore cannot use advanced analysis at scale. To address this gap, we develop a novel agentic AI system, PowerChain, to solve unseen DG analysis tasks via automated agentic orchestration and large language models (LLMs) function-calling. Given a natural language query, PowerChain dynamically generates and executes an ordered sequence of domain-aware functions guided by the semantics of an expert-built power systems function pool and a select reference set of known, expert-generated workflow-query pairs. Our results show that PowerChain can produce expert-level workflows with both GPT-5 and open-source Qwen models on complex, unseen DG analysis tasks operating on real utility data. 

---
# Type-Compliant Adaptation Cascades: Adapting Programmatic LM Workflows to Data 

**Authors**: Chu-Cheng Lin, Daiyi Peng, Yifeng Lu, Ming Zhang, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18244)  

**Abstract**: Reliably composing Large Language Models (LLMs) for complex, multi-step workflows remains a significant challenge. The dominant paradigm-optimizing discrete prompts in a pipeline-is notoriously brittle and struggles to enforce the formal compliance required for structured tasks. We introduce Type-Compliant Adaptation Cascades (TACs), a framework that recasts workflow adaptation as learning typed probabilistic programs. TACs treats the entire workflow, which is composed of parameter-efficiently adapted LLMs and deterministic logic, as an unnormalized joint distribution. This enables principled, gradient-based training even with latent intermediate structures. We provide theoretical justification for our tractable optimization objective, proving that the optimization bias vanishes as the model learns type compliance. Empirically, TACs significantly outperforms state-of-the-art prompt-optimization baselines. Gains are particularly pronounced on structured tasks, improving MGSM-SymPy from $57.1\%$ to $75.9\%$ for a 27B model, MGSM from $1.6\%$ to $27.3\%$ for a 7B model. TACs offers a robust and theoretically grounded paradigm for developing reliable, task-compliant LLM systems. 

---
# Leveraging Large Language Models for Accurate Sign Language Translation in Low-Resource Scenarios 

**Authors**: Luana Bulla, Gabriele Tuccio, Misael Mongiovì, Aldo Gangemi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18183)  

**Abstract**: Translating natural languages into sign languages is a highly complex and underexplored task. Despite growing interest in accessibility and inclusivity, the development of robust translation systems remains hindered by the limited availability of parallel corpora which align natural language with sign language data. Existing methods often struggle to generalize in these data-scarce environments, as the few datasets available are typically domain-specific, lack standardization, or fail to capture the full linguistic richness of sign languages. To address this limitation, we propose Advanced Use of LLMs for Sign Language Translation (AulSign), a novel method that leverages Large Language Models via dynamic prompting and in-context learning with sample selection and subsequent sign association. Despite their impressive abilities in processing text, LLMs lack intrinsic knowledge of sign languages; therefore, they are unable to natively perform this kind of translation. To overcome this limitation, we associate the signs with compact descriptions in natural language and instruct the model to use them. We evaluate our method on both English and Italian languages using SignBank+, a recognized benchmark in the field, as well as the Italian LaCAM CNR-ISTC dataset. We demonstrate superior performance compared to state-of-the-art models in low-data scenario. Our findings demonstrate the effectiveness of AulSign, with the potential to enhance accessibility and inclusivity in communication technologies for underrepresented linguistic communities. 

---
# A.S.E: A Repository-Level Benchmark for Evaluating Security in AI-Generated Code 

**Authors**: Keke Lian, Bin Wang, Lei Zhang, Libo Chen, Junjie Wang, Ziming Zhao, Yujiu Yang, Haotong Duan, Haoran Zhao, Shuang Liao, Mingda Guo, Jiazheng Quan, Yilu Zhong, Chenhao He, Zichuan Chen, Jie Wu, Haoling Li, Zhaoxuan Li, Jiongchi Yu, Hui Li, Dong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18106)  

**Abstract**: The increasing adoption of large language models (LLMs) in software engineering necessitates rigorous security evaluation of their generated code. However, existing benchmarks are inadequate, as they focus on isolated code snippets, employ unstable evaluation methods that lack reproducibility, and fail to connect the quality of input context with the security of the output. To address these gaps, we introduce A.S.E (AI Code Generation Security Evaluation), a benchmark for repository-level secure code generation. A.S.E constructs tasks from real-world repositories with documented CVEs, preserving full repository context like build systems and cross-file dependencies. Its reproducible, containerized evaluation framework uses expert-defined rules to provide stable, auditable assessments of security, build quality, and generation stability. Our evaluation of leading LLMs on A.S.E reveals three key findings: (1) Claude-3.7-Sonnet achieves the best overall performance. (2) The security gap between proprietary and open-source models is narrow; Qwen3-235B-A22B-Instruct attains the top security score. (3) Concise, ``fast-thinking'' decoding strategies consistently outperform complex, ``slow-thinking'' reasoning for security patching. 

---
# Meta-R1: Empowering Large Reasoning Models with Metacognition 

**Authors**: Haonan Dong, Haoran Ye, Wenhao Zhu, Kehan Jiang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.17291)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex tasks, exhibiting emergent, human-like thinking patterns. Despite their advances, we identify a fundamental limitation: current LRMs lack a dedicated meta-level cognitive system-an essential faculty in human cognition that enables "thinking about thinking". This absence leaves their emergent abilities uncontrollable (non-adaptive reasoning), unreliable (intermediate error), and inflexible (lack of a clear methodology). To address this gap, we introduce Meta-R1, a systematic and generic framework that endows LRMs with explicit metacognitive capabilities. Drawing on principles from cognitive science, Meta-R1 decomposes the reasoning process into distinct object-level and meta-level components, orchestrating proactive planning, online regulation, and adaptive early stopping within a cascaded framework. Experiments on three challenging benchmarks and against eight competitive baselines demonstrate that Meta-R1 is: (I) high-performing, surpassing state-of-the-art methods by up to 27.3%; (II) token-efficient, reducing token consumption to 15.7% ~ 32.7% and improving efficiency by up to 14.8% when compared to its vanilla counterparts; and (III) transferable, maintaining robust performance across datasets and model backbones. 

---
# Named Entity Recognition of Historical Text via Large Language Model 

**Authors**: Shibingfeng Zhang, Giovanni Colavizza  

**Link**: [PDF](https://arxiv.org/pdf/2508.18090)  

**Abstract**: Large language models have demonstrated remarkable versatility across a wide range of natural language processing tasks and domains. One such task is Named Entity Recognition (NER), which involves identifying and classifying proper names in text, such as people, organizations, locations, dates, and other specific entities. NER plays a crucial role in extracting information from unstructured textual data, enabling downstream applications such as information retrieval from unstructured text.
Traditionally, NER is addressed using supervised machine learning approaches, which require large amounts of annotated training data. However, historical texts present a unique challenge, as the annotated datasets are often scarce or nonexistent, due to the high cost and expertise required for manual labeling. In addition, the variability and noise inherent in historical language, such as inconsistent spelling and archaic vocabulary, further complicate the development of reliable NER systems for these sources.
In this study, we explore the feasibility of applying LLMs to NER in historical documents using zero-shot and few-shot prompting strategies, which require little to no task-specific training data. Our experiments, conducted on the HIPE-2022 (Identifying Historical People, Places and other Entities) dataset, show that LLMs can achieve reasonably strong performance on NER tasks in this setting. While their performance falls short of fully supervised models trained on domain-specific annotations, the results are nevertheless promising. These findings suggest that LLMs offer a viable and efficient alternative for information extraction in low-resource or historically significant corpora, where traditional supervised methods are infeasible. 

---
# HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data 

**Authors**: Jiyoon Myung, Jihyeon Park, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.18048)  

**Abstract**: User queries in real-world recommendation systems often combine structured constraints (e.g., category, attributes) with unstructured preferences (e.g., product descriptions or reviews). We introduce HyST (Hybrid retrieval over Semi-structured Tabular data), a hybrid retrieval framework that combines LLM-powered structured filtering with semantic embedding search to support complex information needs over semi-structured tabular data. HyST extracts attribute-level constraints from natural language using large language models (LLMs) and applies them as metadata filters, while processing the remaining unstructured query components via embedding-based retrieval. Experiments on a semi-structured benchmark show that HyST consistently outperforms tradtional baselines, highlighting the importance of structured filtering in improving retrieval precision, offering a scalable and accurate solution for real-world user queries. 

---
# Automating Conflict-Aware ACL Configurations with Natural Language Intents 

**Authors**: Wenlong Ding, Jianqiang Li, Zhixiong Niu, Huangxun Chen, Yongqiang Xiong, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17990)  

**Abstract**: ACL configuration is essential for managing network flow reachability, yet its complexity grows significantly with topologies and pre-existing rules. To carry out ACL configuration, the operator needs to (1) understand the new configuration policies or intents and translate them into concrete ACL rules, (2) check and resolve any conflicts between the new and existing rules, and (3) deploy them across the network. Existing systems rely heavily on manual efforts for these tasks, especially for the first two, which are tedious, error-prone, and impractical to scale.
We propose Xumi to tackle this problem. Leveraging LLMs with domain knowledge of the target network, Xumi automatically and accurately translates the natural language intents into complete ACL rules to reduce operators' manual efforts. Xumi then detects all potential conflicts between new and existing rules and generates resolved intents for deployment with operators' guidance, and finally identifies the best deployment plan that minimizes the rule additions while satisfying all intents. Evaluation shows that Xumi accelerates the entire configuration pipeline by over 10x compared to current practices, addresses O(100) conflicting ACLs and reduces rule additions by ~40% in modern cloud network. 

---
# Test-Time Scaling Strategies for Generative Retrieval in Multimodal Conversational Recommendations 

**Authors**: Hung-Chun Hsu, Yuan-Ching Kuo, Chao-Han Huck Yang, Szu-Wei Fu, Hanrong Ye, Hongxu Yin, Yu-Chiang Frank Wang, Ming-Feng Tsai, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18132)  

**Abstract**: The rapid evolution of e-commerce has exposed the limitations of traditional product retrieval systems in managing complex, multi-turn user interactions. Recent advances in multimodal generative retrieval -- particularly those leveraging multimodal large language models (MLLMs) as retrievers -- have shown promise. However, most existing methods are tailored to single-turn scenarios and struggle to model the evolving intent and iterative nature of multi-turn dialogues when applied naively. Concurrently, test-time scaling has emerged as a powerful paradigm for improving large language model (LLM) performance through iterative inference-time refinement. Yet, its effectiveness typically relies on two conditions: (1) a well-defined problem space (e.g., mathematical reasoning), and (2) the model's ability to self-correct -- conditions that are rarely met in conversational product search. In this setting, user queries are often ambiguous and evolving, and MLLMs alone have difficulty grounding responses in a fixed product corpus. Motivated by these challenges, we propose a novel framework that introduces test-time scaling into conversational multimodal product retrieval. Our approach builds on a generative retriever, further augmented with a test-time reranking (TTR) mechanism that improves retrieval accuracy and better aligns results with evolving user intent throughout the dialogue. Experiments across multiple benchmarks show consistent improvements, with average gains of 14.5 points in MRR and 10.6 points in nDCG@1. 

---
# Understanding Subword Compositionality of Large Language Models 

**Authors**: Qiwei Peng, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17953)  

**Abstract**: Large language models (LLMs) take sequences of subwords as input, requiring them to effective compose subword representations into meaningful word-level representations. In this paper, we present a comprehensive set of experiments to probe how LLMs compose subword information, focusing on three key aspects: structural similarity, semantic decomposability, and form retention. Our analysis of the experiments suggests that these five LLM families can be classified into three distinct groups, likely reflecting difference in their underlying composition strategies. Specifically, we observe (i) three distinct patterns in the evolution of structural similarity between subword compositions and whole-word representations across layers; (ii) great performance when probing layer by layer their sensitivity to semantic decompositionality; and (iii) three distinct patterns when probing sensitivity to formal features, e.g., character sequence length. These findings provide valuable insights into the compositional dynamics of LLMs and highlight different compositional pattens in how LLMs encode and integrate subword information. 

---
# Debiasing Multilingual LLMs in Cross-lingual Latent Space 

**Authors**: Qiwei Peng, Guimin Hu, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17948)  

**Abstract**: Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability. 

---
# Learning from Few Samples: A Novel Approach for High-Quality Malcode Generation 

**Authors**: Haijian Ma, Daizong Liu, Xiaowen Cai, Pan Zhou, Yulai Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18148)  

**Abstract**: Intrusion Detection Systems (IDS) play a crucial role in network security defense. However, a significant challenge for IDS in training detection models is the shortage of adequately labeled malicious samples. To address these issues, this paper introduces a novel semi-supervised framework \textbf{GANGRL-LLM}, which integrates Generative Adversarial Networks (GANs) with Large Language Models (LLMs) to enhance malicious code generation and SQL Injection (SQLi) detection capabilities in few-sample learning scenarios. Specifically, our framework adopts a collaborative training paradigm where: (1) the GAN-based discriminator improves malicious pattern recognition through adversarial learning with generated samples and limited real samples; and (2) the LLM-based generator refines the quality of malicious code synthesis using reward signals from the discriminator. The experimental results demonstrate that even with a limited number of labeled samples, our training framework is highly effective in enhancing both malicious code generation and detection capabilities. This dual enhancement capability offers a promising solution for developing adaptive defense systems capable of countering evolving cyber threats. 

---
# AVAM: Universal Training-free Adaptive Visual Anchoring Embedded into Multimodal Large Language Model for Multi-image Question Answering 

**Authors**: Kang Zeng, Guojin Zhong, Jintao Cheng, Jin Yuan, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17860)  

**Abstract**: The advancement of Multimodal Large Language Models (MLLMs) has driven significant progress in Visual Question Answering (VQA), evolving from Single to Multi Image VQA (MVQA). However, the increased number of images in MVQA inevitably introduces substantial visual redundancy that is irrelevant to question answering, negatively impacting both accuracy and efficiency. To address this issue, existing methods lack flexibility in controlling the number of compressed visual tokens and tend to produce discrete visual fragments, which hinder MLLMs' ability to comprehend images holistically. In this paper, we propose a straightforward yet universal Adaptive Visual Anchoring strategy, which can be seamlessly integrated into existing MLLMs, offering significant accuracy improvements through adaptive compression. Meanwhile, to balance the results derived from both global and compressed visual input, we further introduce a novel collaborative decoding mechanism, enabling optimal performance. Extensive experiments validate the effectiveness of our method, demonstrating consistent performance improvements across various MLLMs. The code will be publicly available. 

---
# Speculative Safety-Aware Decoding 

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17739)  

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design. 

---
# Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models 

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17674)  

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community. 

---
# Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery 

**Authors**: Robert Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17681)  

**Abstract**: Bold claims about AI's role in science-from "AGI will cure all diseases" to promises of radically accelerated discovery-raise a central epistemic question: do large language models (LLMs) truly generate new knowledge, or do they merely remix memorized fragments? We propose unlearning-as-ablation as a falsifiable test of constructive scientific discovery. The method systematically removes a target result and its entire forget-closure (lemmas, paraphrases, and multi-hop entailments) and then evaluates whether the model can re-derive the result from only permitted axioms and tools. Success provides evidence for genuine generative capability; failure exposes current limits. Unlike prevailing motivations for unlearning-privacy, copyright, or safety-our framing repositions it as an epistemic probe for AI-for-Science. We argue that such tests could serve as the next generation of benchmarks, much as ImageNet catalyzed progress in vision: distinguishing models that can merely recall from those that can constructively generate new scientific knowledge. We outline a minimal pilot in mathematics and algorithms, and discuss extensions to physics, chemistry, and biology. Whether models succeed or fail, unlearning-as-ablation provides a principled framework to map the true reach and limits of AI scientific discovery. This is a position paper: we advance a conceptual and methodological argument rather than new empirical results. 

---
# Weights-Rotated Preference Optimization for Large Language Models 

**Authors**: Chenxu Yang, Ruipeng Jia, Mingyu Zheng, Naibin Gu, Zheng Lin, Siyuan Chen, Weichong Yin, Hua Wu, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17637)  

**Abstract**: Despite the efficacy of Direct Preference Optimization (DPO) in aligning Large Language Models (LLMs), reward hacking remains a pivotal challenge. This issue emerges when LLMs excessively reduce the probability of rejected completions to achieve high rewards, without genuinely meeting their intended goals. As a result, this leads to overly lengthy generation lacking diversity, as well as catastrophic forgetting of knowledge. We investigate the underlying reason behind this issue, which is representation redundancy caused by neuron collapse in the parameter space. Hence, we propose a novel Weights-Rotated Preference Optimization (RoPO) algorithm, which implicitly constrains the output layer logits with the KL divergence inherited from DPO and explicitly constrains the intermediate hidden states by fine-tuning on a multi-granularity orthogonal matrix. This design prevents the policy model from deviating too far from the reference model, thereby retaining the knowledge and expressive capabilities acquired during pre-training and SFT stages. Our RoPO achieves up to a 3.27-point improvement on AlpacaEval 2, and surpasses the best baseline by 6.2 to 7.5 points on MT-Bench with merely 0.015% of the trainable parameters, demonstrating its effectiveness in alleviating the reward hacking problem of DPO. 

---
# Instant Preference Alignment for Text-to-Image Diffusion Models 

**Authors**: Yang Li, Songlin Yang, Xiaoxuan Han, Wei Wang, Jing Dong, Yueming Lyu, Ziyu Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17718)  

**Abstract**: Text-to-image (T2I) generation has greatly enhanced creative expression, yet achieving preference-aligned generation in a real-time and training-free manner remains challenging. Previous methods often rely on static, pre-collected preferences or fine-tuning, limiting adaptability to evolving and nuanced user intents. In this paper, we highlight the need for instant preference-aligned T2I generation and propose a training-free framework grounded in multimodal large language model (MLLM) priors. Our framework decouples the task into two components: preference understanding and preference-guided generation. For preference understanding, we leverage MLLMs to automatically extract global preference signals from a reference image and enrich a given prompt using structured instruction design. Our approach supports broader and more fine-grained coverage of user preferences than existing methods. For preference-guided generation, we integrate global keyword-based control and local region-aware cross-attention modulation to steer the diffusion model without additional training, enabling precise alignment across both global attributes and local elements. The entire framework supports multi-round interactive refinement, facilitating real-time and context-aware image generation. Extensive experiments on the Viper dataset and our collected benchmark demonstrate that our method outperforms prior approaches in both quantitative metrics and human evaluations, and opens up new possibilities for dialog-based generation and MLLM-diffusion integration. 

---
# Database Normalization via Dual-LLM Self-Refinement 

**Authors**: Eunjae Jo, Nakyung Lee, Gyuyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17693)  

**Abstract**: Database normalization is crucial to preserving data integrity. However, it is time-consuming and error-prone, as it is typically performed manually by data engineers. To this end, we present Miffie, a database normalization framework that leverages the capability of large language models. Miffie enables automated data normalization without human effort while preserving high accuracy. The core of Miffie is a dual-model self-refinement architecture that combines the best-performing models for normalized schema generation and verification, respectively. The generation module eliminates anomalies based on the feedback of the verification module until the output schema satisfies the requirement for normalization. We also carefully design task-specific zero-shot prompts to guide the models for achieving both high accuracy and cost efficiency. Experimental results show that Miffie can normalize complex database schemas while maintaining high accuracy. 

---
# Steering When Necessary: Flexible Steering Large Language Models with Backtracking 

**Authors**: Jinwei Gan, Zifeng Cheng, Zhiwei Jiang, Cong Wang, Yafeng Yin, Xiang Luo, Yuchen Fu, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17621)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across many generation tasks. Nevertheless, effectively aligning them with desired behaviors remains a significant challenge. Activation steering is an effective and cost-efficient approach that directly modifies the activations of LLMs during the inference stage, aligning their responses with the desired behaviors and avoiding the high cost of fine-tuning. Existing methods typically indiscriminately intervene to all generations or rely solely on the question to determine intervention, which limits the accurate assessment of the intervention strength. To this end, we propose the Flexible Activation Steering with Backtracking (FASB) framework, which dynamically determines both the necessity and strength of intervention by tracking the internal states of the LLMs during generation, considering both the question and the generated content. Since intervening after detecting a deviation from the desired behavior is often too late, we further propose the backtracking mechanism to correct the deviated tokens and steer the LLMs toward the desired behavior. Extensive experiments on the TruthfulQA dataset and six multiple-choice datasets demonstrate that our method outperforms baselines. Our code will be released at this https URL. 

---
# Retrieval Capabilities of Large Language Models Scale with Pretraining FLOPs 

**Authors**: Jacob Portes, Connor Jennings, Erica Ji Yuen, Sasha Doubov, Michael Carbin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17400)  

**Abstract**: How does retrieval performance scale with pretraining FLOPs? We benchmark retrieval performance across LLM model sizes from 125 million parameters to 7 billion parameters pretrained on datasets ranging from 1 billion tokens to more than 2 trillion tokens. We find that retrieval performance on zero-shot BEIR tasks predictably scales with LLM size, training duration, and estimated FLOPs. We also show that In-Context Learning scores are strongly correlated with retrieval scores across retrieval tasks. Finally, we highlight the implications this has for the development of LLM-based retrievers. 

---
# Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents 

**Authors**: Sameer Komoravolu, Khalil Mrini  

**Link**: [PDF](https://arxiv.org/pdf/2508.17393)  

**Abstract**: LLM agents are increasingly deployed to plan, retrieve, and write with tools, yet evaluation still leans on static benchmarks and small human studies. We present the Agent-Testing Agent (ATA), a meta-agent that combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation whose difficulty adapts via judge feedback. Each dialogue is scored with an LLM-as-a-Judge (LAAJ) rubric and used to steer subsequent tests toward the agent's weakest capabilities. On a travel planner and a Wikipedia writer, the ATA surfaces more diverse and severe failures than expert annotators while matching severity, and finishes in 20--30 minutes versus ten-annotator rounds that took days. Ablating code analysis and web search increases variance and miscalibration, underscoring the value of evidence-grounded test generation. The ATA outputs quantitative metrics and qualitative bug reports for developers. We release the full methodology and open-source implementation for reproducible agent testing: this https URL 

---
# Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning 

**Authors**: Yicong Wu, Guangyue Lu, Yuan Zuo, Huarong Zhang, Junjie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17387)  

**Abstract**: Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research. 

---
# CultranAI at PalmX 2025: Data Augmentation for Cultural Knowledge Representation 

**Authors**: Hunzalah Hassan Bhatti, Youssef Ahmed, Md Arid Hasan, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.17324)  

**Abstract**: In this paper, we report our participation to the PalmX cultural evaluation shared task. Our system, CultranAI, focused on data augmentation and LoRA fine-tuning of large language models (LLMs) for Arabic cultural knowledge representation. We benchmarked several LLMs to identify the best-performing model for the task. In addition to utilizing the PalmX dataset, we augmented it by incorporating the Palm dataset and curated a new dataset of over 22K culturally grounded multiple-choice questions (MCQs). Our experiments showed that the Fanar-1-9B-Instruct model achieved the highest performance. We fine-tuned this model on the combined augmented dataset of 22K+ MCQs. On the blind test set, our submitted system ranked 5th with an accuracy of 70.50%, while on the PalmX development set, it achieved an accuracy of 84.1%. 

---
# Chinese Court Simulation with LLM-Based Agent System 

**Authors**: Kaiyuan Zhang, Jiaqi Li, Yueyue Wu, Haitao Li, Cheng Luo, Shaokun Zou, Yujia Zhou, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17322)  

**Abstract**: Mock trial has long served as an important platform for legal professional training and education. It not only helps students learn about realistic trial procedures, but also provides practical value for case analysis and judgment prediction. Traditional mock trials are difficult to access by the public because they rely on professional tutors and human participants. Fortunately, the rise of large language models (LLMs) provides new opportunities for creating more accessible and scalable court simulations. While promising, existing research mainly focuses on agent construction while ignoring the systematic design and evaluation of court simulations, which are actually more important for the credibility and usage of court simulation in practice. To this end, we present the first court simulation framework -- SimCourt -- based on the real-world procedure structure of Chinese courts. Our framework replicates all 5 core stages of a Chinese trial and incorporates 5 courtroom roles, faithfully following the procedural definitions in China. To simulate trial participants with different roles, we propose and craft legal agents equipped with memory, planning, and reflection abilities. Experiment on legal judgment prediction show that our framework can generate simulated trials that better guide the system to predict the imprisonment, probation, and fine of each case. Further annotations by human experts show that agents' responses under our simulation framework even outperformed judges and lawyers from the real trials in many scenarios. These further demonstrate the potential of LLM-based court simulation. 

---
# Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit 

**Authors**: Zihao Wei, Liang Pang, Jiahao Liu, Jingcheng Deng, Shicheng Xu, Zenghao Duan, Jingang Wang, Fei Sun, Xunliang Cai, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17627)  

**Abstract**: Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy. 

---
# Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs 

**Authors**: Ryoma Kondo, Riona Matsuoka, Takahiro Yoshida, Kazuyuki Yamasawa, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2508.17340)  

**Abstract**: Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods. 

---
# SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation 

**Authors**: Xiaqiang Tang, Yi Wang, Keyu Hu, Rui Xu, Chuang Li, Weigao Sun, Jian Li, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.17225)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require Large Language Models (LLMs) to generate responses that are faithful to the retrieved context. However, faithfulness hallucination remains a critical challenge, as existing methods often require costly supervision and post-training or significant inference burdens. To overcome these limitations, we introduce Self-Supervised Faithfulness Optimization (SSFO), the first self-supervised alignment approach for enhancing RAG faithfulness. SSFO constructs preference data pairs by contrasting the model's outputs generated with and without the context. Leveraging Direct Preference Optimization (DPO), SSFO aligns model faithfulness without incurring labeling costs or additional inference burden. We theoretically and empirically demonstrate that SSFO leverages a benign form of \emph{likelihood displacement}, transferring probability mass from parametric-based tokens to context-aligned tokens. Based on this insight, we propose a modified DPO loss function to encourage likelihood displacement. Comprehensive evaluations show that SSFO significantly outperforms existing methods, achieving state-of-the-art faithfulness on multiple context-based question-answering datasets. Notably, SSFO exhibits strong generalization, improving cross-lingual faithfulness and preserving general instruction-following capabilities. We release our code and model at the anonymous link: this https URL 

---
# BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens 

**Authors**: Hao Wen, Xinrui Wu, Yi Sun, Feifei Zhang, Liye Chen, Jie Wang, Yunxin Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17196)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments. 

---
# How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System 

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2508.17215)  

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems. 

---
# Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages 

**Authors**: Yuemei Xu, Kexin Xu, Jian Zhou, Ling Hu, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17078)  

**Abstract**: The current Large Language Models (LLMs) face significant challenges in improving performance on low-resource languages and urgently need data-efficient methods without costly fine-tuning. From the perspective of language-bridge, we propose BridgeX-ICL, a simple yet effective method to improve zero-shot Cross-lingual In-Context Learning (X-ICL) for low-resource languages. Unlike existing works focusing on language-specific neurons, BridgeX-ICL explores whether sharing neurons can improve cross-lingual performance in LLMs or not. We construct neuron probe data from the ground-truth MUSE bilingual dictionaries, and define a subset of language overlap neurons accordingly, to ensure full activation of these anchored neurons. Subsequently, we propose an HSIC-based metric to quantify LLMs' internal linguistic spectrum based on overlap neurons, which guides optimal bridge selection. The experiments conducted on 2 cross-lingual tasks and 15 language pairs from 7 diverse families (covering both high-low and moderate-low pairs) validate the effectiveness of BridgeX-ICL and offer empirical insights into the underlying multilingual mechanisms of LLMs. 

---
# ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation 

**Authors**: Riccardo Pozzi, Matteo Palmonari, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16983)  

**Abstract**: Knowledge gaps and hallucinations are persistent challenges for Large Language Models (LLMs), which generate unreliable responses when lacking the necessary information to fulfill user instructions. Existing approaches, such as Retrieval-Augmented Generation (RAG) and tool use, aim to address these issues by incorporating external knowledge. Yet, they rely on additional models or services, resulting in complex pipelines, potential error propagation, and often requiring the model to process a large number of tokens. In this paper, we present a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from a Knowledge Graph are verbalized in textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact. We evaluate our proposal on Question Answering and show that it scales to large knowledge bases (800 million facts), adapts to domain-specific data, and achieves effective results. These gains come with minimal generation-time overhead. ReFactX code is available at this https URL. 

---
# LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components 

**Authors**: Hikaru Tsujimura, Arush Tagade  

**Link**: [PDF](https://arxiv.org/pdf/2508.17182)  

**Abstract**: Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior. 

---
# Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning 

**Authors**: Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Jiale Zhao, Jingwen Yang, Jianwei Lv, Kongcheng Zhang, Yihe Zhou, Hengtong Lu, Wei Chen, Yan Xie, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.16949)  

**Abstract**: Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen-2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. 

---
# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model 

**Authors**: Fan Ding, Xuewen Luo, Hwa Hui Tew, Ruturaj Reddy, Xikun Wang, Junn Yong Loo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16947)  

**Abstract**: Recent advances in motion planning for autonomous driving have led to models capable of generating high-quality trajectories. However, most existing planners tend to fix their policy after supervised training, leading to consistent but rigid driving behaviors. This limits their ability to reflect human preferences or adapt to dynamic, instruction-driven demands. In this work, we propose a diffusion-based multi-head trajectory planner(M-diffusion planner). During the early training stage, all output heads share weights to learn to generate high-quality trajectories. Leveraging the probabilistic nature of diffusion models, we then apply Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained model for diverse policy-specific behaviors. At inference time, we incorporate a large language model (LLM) to guide strategy selection, enabling dynamic, instruction-aware planning without switching models. Closed-loop simulation demonstrates that our post-trained planner retains strong planning capability while achieving state-of-the-art (SOTA) performance on the nuPlan val14 benchmark. Open-loop results further show that the generated trajectories exhibit clear diversity, effectively satisfying multi-modal driving behavior requirements. The code and related experiments will be released upon acceptance of the paper. 

---
# LLM-based Human-like Traffic Simulation for Self-driving Tests 

**Authors**: Wendi Li, Hao Wu, Han Gao, Bing Mao, Fengyuan Xu, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16962)  

**Abstract**: Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability. 

---
# TextOnly: A Unified Function Portal for Text-Related Functions on Smartphones 

**Authors**: Minghao Tu, Chun Yu, Xiyuan Shen, Zhi Zheng, Li Chen, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16926)  

**Abstract**: Text boxes serve as portals to diverse functionalities in today's smartphone applications. However, when it comes to specific functionalities, users always need to navigate through multiple steps to access particular text boxes for input. We propose TextOnly, a unified function portal that enables users to access text-related functions from various applications by simply inputting text into a sole text box. For instance, entering a restaurant name could trigger a Google Maps search, while a greeting could initiate a conversation in WhatsApp. Despite their brevity, TextOnly maximizes the utilization of these raw text inputs, which contain rich information, to interpret user intentions effectively. TextOnly integrates large language models(LLM) and a BERT model. The LLM consistently provides general knowledge, while the BERT model can continuously learn user-specific preferences and enable quicker predictions. Real-world user studies demonstrated TextOnly's effectiveness with a top-1 accuracy of 71.35%, and its ability to continuously improve both its accuracy and inference speed. Participants perceived TextOnly as having satisfactory usability and expressed a preference for TextOnly over manual executions. Compared with voice assistants, TextOnly supports a greater range of text-related functions and allows for more concise inputs. 

---
# DevLicOps: A Framework for Mitigating Licensing Risks in AI-Generated Code 

**Authors**: Pratyush Nidhi Sharma, Lauren Wright, Anne Herfurth, Munsif Sokiyna, Pratyaksh Nidhi Sharma, Sethu Das, Mikko Siponen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16853)  

**Abstract**: Generative AI coding assistants (ACAs) are widely adopted yet pose serious legal and compliance risks. ACAs can generate code governed by restrictive open-source licenses (e.g., GPL), potentially exposing companies to litigation or forced open-sourcing. Few developers are trained in these risks, and legal standards vary globally, especially with outsourcing. Our article introduces DevLicOps, a practical framework that helps IT leaders manage ACA-related licensing risks through governance, incident response, and informed tradeoffs. As ACA adoption grows and legal frameworks evolve, proactive license compliance is essential for responsible, risk-aware software development in the AI era. 

---
# EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention 

**Authors**: Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Jia-Jun Li, Collin McMillan, Kevin Leach, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16771)  

**Abstract**: Code language models (so-called CodeLLMs) are now commonplace in software development. As a general rule, CodeLLMs are trained by dividing training examples into input tokens and then learn importance of those tokens in a process called machine attention. Machine attention is based solely on input token salience to output token examples during training. Human software developers are different, as humans intuitively know that some tokens are more salient than others. While intuition itself is ineffable and a subject of philosophy, clues about salience are present in human visual attention, since people tend to look at more salient words more often. In this paper, we present EyeMulator, a technique for training CodeLLMs to mimic human visual attention while training for various software development tasks. We add special weights for each token in each input example to the loss function used during LLM fine-tuning. We draw these weights from observations of human visual attention derived from a previously-collected publicly-available dataset of eye-tracking experiments in software engineering tasks. These new weights ultimately induce changes in the attention of the subject LLM during training, resulting in a model that does not need eye-tracking data during inference. Our evaluation shows that EyeMulator outperforms strong LLM baselines on several tasks such as code translation, completion and summarization. We further show an ablation study that demonstrates the improvement is due to subject models learning to mimic human attention. 

---
# Interpreting the Effects of Quantization on LLMs 

**Authors**: Manpreet Singh, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2508.16785)  

**Abstract**: Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique. 

---
# Assessing Consciousness-Related Behaviors in Large Language Models Using the Maze Test 

**Authors**: Rui A. Pimenta, Tim Schlippe, Kristina Schaaff  

**Link**: [PDF](https://arxiv.org/pdf/2508.16705)  

**Abstract**: We investigate consciousness-like behaviors in Large Language Models (LLMs) using the Maze Test, challenging models to navigate mazes from a first-person perspective. This test simultaneously probes spatial awareness, perspective-taking, goal-directed behavior, and temporal sequencing-key consciousness-associated characteristics. After synthesizing consciousness theories into 13 essential characteristics, we evaluated 12 leading LLMs across zero-shot, one-shot, and few-shot learning scenarios. Results showed reasoning-capable LLMs consistently outperforming standard versions, with Gemini 2.0 Pro achieving 52.9% Complete Path Accuracy and DeepSeek-R1 reaching 80.5% Partial Path Accuracy. The gap between these metrics indicates LLMs struggle to maintain coherent self-models throughout solutions -- a fundamental consciousness aspect. While LLMs show progress in consciousness-related behaviors through reasoning mechanisms, they lack the integrated, persistent self-awareness characteristic of consciousness. 

---
# Generative Artificial Intelligence and Agents in Research and Teaching 

**Authors**: Jussi S. Jauhiainen, Aurora Toppari  

**Link**: [PDF](https://arxiv.org/pdf/2508.16701)  

**Abstract**: This study provides a comprehensive analysis of the development, functioning, and application of generative artificial intelligence (GenAI) and large language models (LLMs), with an emphasis on their implications for research and education. It traces the conceptual evolution from artificial intelligence (AI) through machine learning (ML) and deep learning (DL) to transformer architectures, which constitute the foundation of contemporary generative systems. Technical aspects, including prompting strategies, word embeddings, and probabilistic sampling methods (temperature, top-k, and top-p), are examined alongside the emergence of autonomous agents. These elements are considered in relation to both the opportunities they create and the limitations and risks they entail.
The work critically evaluates the integration of GenAI across the research process, from ideation and literature review to research design, data collection, analysis, interpretation, and dissemination. While particular attention is given to geographical research, the discussion extends to wider academic contexts. A parallel strand addresses the pedagogical applications of GenAI, encompassing course and lesson design, teaching delivery, assessment, and feedback, with geography education serving as a case example.
Central to the analysis are the ethical, social, and environmental challenges posed by GenAI. Issues of bias, intellectual property, governance, and accountability are assessed, alongside the ecological footprint of LLMs and emerging technological strategies for mitigation. The concluding section considers near- and long-term futures of GenAI, including scenarios of sustained adoption, regulation, and potential decline. By situating GenAI within both scholarly practice and educational contexts, the study contributes to critical debates on its transformative potential and societal responsibilities. 

---
# QueryBandits for Hallucination Mitigation: Exploiting Semantic Features for No-Regret Rewriting 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.16697)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have caused higher hallucination prevalence; yet most mitigation work focuses on after-the-fact filtering rather than shaping the queries that trigger them. We introduce QueryBandits, a bandit framework that designs rewrite strategies to maximize a reward model, that encapsulates hallucination propensity based upon the sensitivities of 17 linguistic features of the input query-and therefore, proactively steer LLMs away from generating hallucinations. Across 13 diverse QA benchmarks and 1,050 lexically perturbed queries per dataset, our top contextual QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a no-rewrite baseline and also outperforms zero-shot static prompting ("paraphrase" or "expand") by 42.6% and 60.3% respectively. Therefore, we empirically substantiate the effectiveness of QueryBandits in mitigating hallucination via the intervention that takes the form of a query rewrite. Interestingly, certain static prompting strategies, which constitute a considerable number of current query rewriting literature, have a higher cumulative regret than the no-rewrite baseline, signifying that static rewrites can worsen hallucination. Moreover, we discover that the converged per-arm regression feature weight vectors substantiate that there is no single rewrite strategy optimal for all queries. In this context, guided rewriting via exploiting semantic features with QueryBandits can induce significant shifts in output behavior through forward-pass mechanisms, bypassing the need for retraining or gradient-based adaptation. 

---
# Do Cognitively Interpretable Reasoning Traces Improve LLM Performance? 

**Authors**: Siddhant Bhambri, Upasana Biswas, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16695)  

**Abstract**: Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability. 

---
# Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation 

**Authors**: Yejin Choi, Jaewoo Park, Janghan Yoon, Saejin Kim, Jaehyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17079)  

**Abstract**: Rapid advances in Multimodal Large Language Models (MLLMs) have expanded information retrieval beyond purely textual inputs, enabling retrieval from complex real world documents that combine text and visuals. However, most documents are private either owned by individuals or confined within corporate silos and current retrievers struggle when faced with unseen domains or languages. To address this gap, we introduce PREMIR, a simple yet effective framework that leverages the broad knowledge of an MLLM to generate cross modal pre questions (preQs) before retrieval. Unlike earlier multimodal retrievers that compare embeddings in a single vector space, PREMIR leverages preQs from multiple complementary modalities to expand the scope of matching to the token level. Experiments show that PREMIR achieves state of the art performance on out of distribution benchmarks, including closed domain and multilingual settings, outperforming strong baselines across all retrieval metrics. We confirm the contribution of each component through in depth ablation studies, and qualitative analyses of the generated preQs further highlight the model's robustness in real world settings. 

---
# Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-based Language Models 

**Authors**: Tharindu Madusanka, Ian Pratt-Hartmann, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2508.17153)  

**Abstract**: Efforts to apply transformer-based language models (TLMs) to the problem of reasoning in natural language have enjoyed ever-increasing success in recent years. The most fundamental task in this area to which nearly all others can be reduced is that of determining satisfiability. However, from a logical point of view, satisfiability problems vary along various dimensions, which may affect TLMs' ability to learn how to solve them. The problem instances of satisfiability in natural language can belong to different computational complexity classes depending on the language fragment in which they are expressed. Although prior research has explored the problem of natural language satisfiability, the above-mentioned point has not been discussed adequately. Hence, we investigate how problem instances from varying computational complexity classes and having different grammatical constructs impact TLMs' ability to learn rules of inference. Furthermore, to faithfully evaluate TLMs, we conduct an empirical study to explore the distribution of satisfiability problems. 

---
# CALR: Corrective Adaptive Low-Rank Decomposition for Efficient Large Language Model Layer Compression 

**Authors**: Muchammad Daniyal Kautsar, Afra Majida Hariono, Widyawan, Syukron Abu Ishaq Alfarozi, Kuntpong Wararatpanya  

**Link**: [PDF](https://arxiv.org/pdf/2508.16680)  

**Abstract**: Large Language Models (LLMs) present significant deployment challenges due to their immense size and computational requirements. Model compression techniques are essential for making these models practical for resource-constrained environments. A prominent compression strategy is low-rank factorization via Singular Value Decomposition (SVD) to reduce model parameters by approximating weight matrices. However, standard SVD focuses on minimizing matrix reconstruction error, often leading to a substantial loss of the model's functional performance. This performance degradation occurs because existing methods do not adequately correct for the functional information lost during compression. To address this gap, we introduce Corrective Adaptive Low-Rank Decomposition (CALR), a two-component compression approach. CALR combines a primary path of SVD-compressed layers with a parallel, learnable, low-rank corrective module that is explicitly trained to recover the functional residual error. Our experimental evaluation on SmolLM2-135M, Qwen3-0.6B, and Llama-3.2-1B, demonstrates that CALR can reduce parameter counts by 26.93% to 51.77% while retaining 59.45% to 90.42% of the original model's performance, consistently outperforming LaCo, ShortGPT, and LoSparse. CALR's success shows that treating functional information loss as a learnable signal is a highly effective compression paradigm. This approach enables the creation of significantly smaller, more efficient LLMs, advancing their accessibility and practical deployment in real-world applications. 

---
# CelloAI: Leveraging Large Language Models for HPC Software Development in High Energy Physics 

**Authors**: Mohammad Atif, Kriti Chopra, Ozgur Kilic, Tianle Wang, Zhihua Dong, Charles Leggett, Meifeng Lin, Paolo Calafiura, Salman Habib  

**Link**: [PDF](https://arxiv.org/pdf/2508.16713)  

**Abstract**: Next-generation High Energy Physics (HEP) experiments will generate unprecedented data volumes, necessitating High Performance Computing (HPC) integration alongside traditional high-throughput computing. However, HPC adoption in HEP is hindered by the challenge of porting legacy software to heterogeneous architectures and the sparse documentation of these complex scientific codebases. We present CelloAI, a locally hosted coding assistant that leverages Large Language Models (LLMs) with retrieval-augmented generation (RAG) to support HEP code documentation and generation. This local deployment ensures data privacy, eliminates recurring costs and provides access to large context windows without external dependencies. CelloAI addresses two primary use cases, code documentation and code generation, through specialized components. For code documentation, the assistant provides: (a) Doxygen style comment generation for all functions and classes by retrieving relevant information from RAG sources (papers, posters, presentations), (b) file-level summary generation, and (c) an interactive chatbot for code comprehension queries. For code generation, CelloAI employs syntax-aware chunking strategies that preserve syntactic boundaries during embedding, improving retrieval accuracy in large codebases. The system integrates callgraph knowledge to maintain dependency awareness during code modifications and provides AI-generated suggestions for performance optimization and accurate refactoring. We evaluate CelloAI using real-world HEP applications from ATLAS, CMS, and DUNE experiments, comparing different embedding models for code retrieval effectiveness. Our results demonstrate the AI assistant's capability to enhance code understanding and support reliable code generation while maintaining the transparency and safety requirements essential for scientific computing environments. 

---
# Recall-Extend Dynamics: Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration 

**Authors**: Zhong Guan, Likang Wu, Hongke Zhao, Jiahui Wang, Le Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16677)  

**Abstract**: Many existing studies have achieved significant improvements in the reasoning capabilities of large language models (LLMs) through reinforcement learning with verifiable rewards (RLVR), while the enhancement of reasoning abilities in small language models (SLMs) has not yet been sufficiently explored. Combining distilled data from larger models with RLVR on small models themselves is a natural approach, but it still faces various challenges and issues. Therefore, we propose \textit{\underline{R}}ecall-\textit{\underline{E}}xtend \textit{\underline{D}}ynamics(RED): Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration. In this paper, we explore the perspective of varying exploration spaces, balancing offline distillation with online reinforcement learning. Simultaneously, we specifically design and optimize for the insertion problem within offline data. By monitoring the ratio of entropy changes in the model concerning offline and online data, we regulate the weight of offline-SFT, thereby addressing the issues of insufficient exploration space in small models and the redundancy and complexity during the distillation process. Furthermore, to tackle the distribution discrepancies between offline data and the current policy, we design a sample-accuracy-based policy shift mechanism that dynamically chooses between imitating offline distilled data and learning from its own policy. 

---
# Invisible Filters: Cultural Bias in Hiring Evaluations Using Large Language Models 

**Authors**: Pooja S. B. Rao, Laxminarayen Nagarajan Venkatesan, Mauro Cherubini, Dinesh Babu Jayagopi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16673)  

**Abstract**: Artificial Intelligence (AI) is increasingly used in hiring, with large language models (LLMs) having the potential to influence or even make hiring decisions. However, this raises pressing concerns about bias, fairness, and trust, particularly across diverse cultural contexts. Despite their growing role, few studies have systematically examined the potential biases in AI-driven hiring evaluation across cultures. In this study, we conduct a systematic analysis of how LLMs assess job interviews across cultural and identity dimensions. Using two datasets of interview transcripts, 100 from UK and 100 from Indian job seekers, we first examine cross-cultural differences in LLM-generated scores for hirability and related traits. Indian transcripts receive consistently lower scores than UK transcripts, even when they were anonymized, with disparities linked to linguistic features such as sentence complexity and lexical diversity. We then perform controlled identity substitutions (varying names by gender, caste, and region) within the Indian dataset to test for name-based bias. These substitutions do not yield statistically significant effects, indicating that names alone, when isolated from other contextual signals, may not influence LLM evaluations. Our findings underscore the importance of evaluating both linguistic and social dimensions in LLM-driven evaluations and highlight the need for culturally sensitive design and accountability in AI-assisted hiring. 

---
# Beyond Memorization: Extending Reasoning Depth with Recurrence, Memory and Test-Time Compute Scaling 

**Authors**: Ivan Rodkin, Daniil Orel, Konstantin Smirnov, Arman Bolatov, Bilal Elbouardi, Besher Hassan, Yuri Kuratov, Aydar Bulatov, Preslav Nakov, Timothy Baldwin, Artem Shelmanov, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2508.16745)  

**Abstract**: Reasoning is a core capability of large language models, yet understanding how they learn and perform multi-step reasoning remains an open problem. In this study, we explore how different architectures and training methods affect model multi-step reasoning capabilities within a cellular automata framework. By training on state sequences generated with random Boolean functions for random initial conditions to exclude memorization, we demonstrate that most neural architectures learn to abstract the underlying rules. While models achieve high accuracy in next-state prediction, their performance declines sharply if multi-step reasoning is required. We confirm that increasing model depth plays a crucial role for sequential computations. We demonstrate that an extension of the effective model depth with recurrence, memory, and test-time compute scaling substantially enhances reasoning capabilities. 

---
# Enabling Multi-Agent Systems as Learning Designers: Applying Learning Sciences to AI Instructional Design 

**Authors**: Jiayi Wang, Ruiwei Xiao, Xinying Hou, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2508.16659)  

**Abstract**: K-12 educators are increasingly using Large Language Models (LLMs) to create instructional materials. These systems excel at producing fluent, coherent content, but often lack support for high-quality teaching. The reason is twofold: first, commercial LLMs, such as ChatGPT and Gemini which are among the most widely accessible to teachers, do not come preloaded with the depth of pedagogical theory needed to design truly effective activities; second, although sophisticated prompt engineering can bridge this gap, most teachers lack the time or expertise and find it difficult to encode such pedagogical nuance into their requests. This study shifts pedagogical expertise from the user's prompt to the LLM's internal architecture. We embed the well-established Knowledge-Learning-Instruction (KLI) framework into a Multi-Agent System (MAS) to act as a sophisticated instructional designer. We tested three systems for generating secondary Math and Science learning activities: a Single-Agent baseline simulating typical teacher prompts; a role-based MAS where agents work sequentially; and a collaborative MAS-CMD where agents co-construct activities through conquer and merge discussion. The generated materials were evaluated by 20 practicing teachers and a complementary LLM-as-a-judge system using the Quality Matters (QM) K-12 standards. While the rubric scores showed only small, often statistically insignificant differences between the systems, the qualitative feedback from educators painted a clear and compelling picture. Teachers strongly preferred the activities from the collaborative MAS-CMD, describing them as significantly more creative, contextually relevant, and classroom-ready. Our findings show that embedding pedagogical principles into LLM systems offers a scalable path for creating high-quality educational content. 

---
# Cognitive Decision Routing in Large Language Models: When to Think Fast, When to Think Slow 

**Authors**: Y. Du, C. Guo, W. Wang, G. Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16636)  

**Abstract**: Large Language Models (LLMs) face a fundamental challenge in deciding when to rely on rapid, intuitive responses versus engaging in slower, more deliberate reasoning. Inspired by Daniel Kahneman's dual-process theory and his insights on human cognitive biases, we propose a novel Cognitive Decision Routing (CDR) framework that dynamically determines the appropriate reasoning strategy based on query characteristics. Our approach addresses the current limitations where models either apply uniform reasoning depth or rely on computationally expensive methods for all queries. We introduce a meta-cognitive layer that analyzes query complexity through multiple dimensions: correlation strength between given information and required conclusions, domain boundary crossings, stakeholder multiplicity, and uncertainty levels. Through extensive experiments on diverse reasoning tasks, we demonstrate that CDR achieves superior performance while reducing computational costs by 34\% compared to uniform deep reasoning approaches. Our framework shows particular strength in professional judgment tasks, achieving 23\% improvement in consistency and 18\% better accuracy on expert-level evaluations. This work bridges cognitive science principles with practical AI system design, offering a principled approach to adaptive reasoning in LLMs. 

---
# Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework 

**Authors**: Zeyu Zhang, Quanyu Dai, Rui Li, Xiaohe Bo, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16629)  

**Abstract**: LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and data-driven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at this https URL. 

---
# GreenTEA: Gradient Descent with Topic-modeling and Evolutionary Auto-prompting 

**Authors**: Zheng Dong, Luming Shang, Gabriela Olinto  

**Link**: [PDF](https://arxiv.org/pdf/2508.16603)  

**Abstract**: High-quality prompts are crucial for Large Language Models (LLMs) to achieve exceptional performance. However, manually crafting effective prompts is labor-intensive and demands significant domain expertise, limiting its scalability. Existing automatic prompt optimization methods either extensively explore new prompt candidates, incurring high computational costs due to inefficient searches within a large solution space, or overly exploit feedback on existing prompts, risking suboptimal optimization because of the complex prompt landscape. To address these challenges, we introduce GreenTEA, an agentic LLM workflow for automatic prompt optimization that balances candidate exploration and knowledge exploitation. It leverages a collaborative team of agents to iteratively refine prompts based on feedback from error samples. An analyzing agent identifies common error patterns resulting from the current prompt via topic modeling, and a generation agent revises the prompt to directly address these key deficiencies. This refinement process is guided by a genetic algorithm framework, which simulates natural selection by evolving candidate prompts through operations such as crossover and mutation to progressively optimize model performance. Extensive numerical experiments conducted on public benchmark datasets suggest the superior performance of GreenTEA against human-engineered prompts and existing state-of-the-arts for automatic prompt optimization, covering logical and quantitative reasoning, commonsense, and ethical decision-making. 

---
# Adaptive Command: Real-Time Policy Adjustment via Language Models in StarCraft II 

**Authors**: Weiyu Ma, Dongyu Xu, Shu Lin, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16580)  

**Abstract**: We present Adaptive Command, a novel framework integrating large language models (LLMs) with behavior trees for real-time strategic decision-making in StarCraft II. Our system focuses on enhancing human-AI collaboration in complex, dynamic environments through natural language interactions. The framework comprises: (1) an LLM-based strategic advisor, (2) a behavior tree for action execution, and (3) a natural language interface with speech capabilities. User studies demonstrate significant improvements in player decision-making and strategic adaptability, particularly benefiting novice players and those with disabilities. This work contributes to the field of real-time human-AI collaborative decision-making, offering insights applicable beyond RTS games to various complex decision-making scenarios. 

---
# An Embodied AR Navigation Agent: Integrating BIM with Retrieval-Augmented Generation for Language Guidance 

**Authors**: Hsuan-Kung Yang, Tsu-Ching Hsiao, Ryoichiro Oka, Ryuya Nishino, Satoko Tofukuji, Norimasa Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2508.16602)  

**Abstract**: Delivering intelligent and adaptive navigation assistance in augmented reality (AR) requires more than visual cues, as it demands systems capable of interpreting flexible user intent and reasoning over both spatial and semantic context. Prior AR navigation systems often rely on rigid input schemes or predefined commands, which limit the utility of rich building data and hinder natural interaction. In this work, we propose an embodied AR navigation system that integrates Building Information Modeling (BIM) with a multi-agent retrieval-augmented generation (RAG) framework to support flexible, language-driven goal retrieval and route planning. The system orchestrates three language agents, Triage, Search, and Response, built on large language models (LLMs), which enables robust interpretation of open-ended queries and spatial reasoning using BIM data. Navigation guidance is delivered through an embodied AR agent, equipped with voice interaction and locomotion, to enhance user experience. A real-world user study yields a System Usability Scale (SUS) score of 80.5, indicating excellent usability, and comparative evaluations show that the embodied interface can significantly improves users' perception of system intelligence. These results underscore the importance and potential of language-grounded reasoning and embodiment in the design of user-centered AR navigation systems. 

---
# LexSemBridge: Fine-Grained Dense Representation Enhancement through Token-Aware Embedding Augmentation 

**Authors**: Shaoxiong Zhan, Hai Lin, Hongming Tan, Xiaodong Cai, Hai-Tao Zheng, Xin Su, Zifei Shan, Ruitong Liu, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17858)  

**Abstract**: As queries in retrieval-augmented generation (RAG) pipelines powered by large language models (LLMs) become increasingly complex and diverse, dense retrieval models have demonstrated strong performance in semantic matching. Nevertheless, they often struggle with fine-grained retrieval tasks, where precise keyword alignment and span-level localization are required, even in cases with high lexical overlap that would intuitively suggest easier retrieval. To systematically evaluate this limitation, we introduce two targeted tasks, keyword retrieval and part-of-passage retrieval, designed to simulate practical fine-grained scenarios. Motivated by these observations, we propose LexSemBridge, a unified framework that enhances dense query representations through fine-grained, input-aware vector modulation. LexSemBridge constructs latent enhancement vectors from input tokens using three paradigms: Statistical (SLR), Learned (LLR), and Contextual (CLR), and integrates them with dense embeddings via element-wise interaction. Theoretically, we show that this modulation preserves the semantic direction while selectively amplifying discriminative dimensions. LexSemBridge operates as a plug-in without modifying the backbone encoder and naturally extends to both text and vision modalities. Extensive experiments across semantic and fine-grained retrieval tasks validate the effectiveness and generality of our approach. All code and models are publicly available at this https URL 

---
# HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation 

**Authors**: Junyi Chen, Lu Chi, Siliang Xu, Shiwei Ran, Bingyue Peng, Zehuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18118)  

**Abstract**: AI-generated content technologies are widely used in content creation. However, current AIGC systems rely heavily on creators' inspiration, rarely generating truly user-personalized content. In real-world applications such as online advertising, a single product may have multiple selling points, with different users focusing on different features. This underscores the significant value of personalized, user-centric creative generation. Effective personalized content generation faces two main challenges: (1) accurately modeling user interests and integrating them into the content generation process while adhering to factual constraints, and (2) ensuring high efficiency and scalability to handle the massive user base in industrial scenarios. Additionally, the scarcity of personalized creative data in practice complicates model training, making data construction another key hurdle. We propose HLLM-Creator, a hierarchical LLM framework for efficient user interest modeling and personalized content generation. During inference, a combination of user clustering and a user-ad-matching-prediction based pruning strategy is employed to significantly enhance generation efficiency and reduce computational overhead, making the approach suitable for large-scale deployment. Moreover, we design a data construction pipeline based on chain-of-thought reasoning, which generates high-quality, user-specific creative titles and ensures factual consistency despite limited personalized data. This pipeline serves as a critical foundation for the effectiveness of our model. Extensive experiments on personalized title generation for Douyin Search Ads show the effectiveness of HLLM-Creator. Online A/B test shows a 0.476% increase on Adss, paving the way for more effective and efficient personalized generation in industrial scenarios. Codes for academic dataset are available at this https URL. 

---
# Demographically-Inspired Query Variants Using an LLM 

**Authors**: Marwah Alaofi, Nicola Ferro, Paul Thomas, Falk Scholer, Mark Sanderson  

**Link**: [PDF](https://arxiv.org/pdf/2508.17644)  

**Abstract**: This study proposes a method to diversify queries in existing test collections to reflect some of the diversity of search engine users, aligning with an earlier vision of an 'ideal' test collection. A Large Language Model (LLM) is used to create query variants: alternative queries that have the same meaning as the original. These variants represent user profiles characterised by different properties, such as language and domain proficiency, which are known in the IR literature to influence query formulation.
The LLM's ability to generate query variants that align with user profiles is empirically validated, and the variants' utility is further explored for IR system evaluation. Results demonstrate that the variants impact how systems are ranked and show that user profiles experience significantly different levels of system effectiveness. This method enables an alternative perspective on system evaluation where we can observe both the impact of user profiles on system rankings and how system performance varies across users. 

---
# SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models 

**Authors**: Tong Bao, Mir Tafseer Nayeem, Davood Rafiei, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17647)  

**Abstract**: Automatic survey generation has emerged as a key task in scientific document processing. While large language models (LLMs) have shown promise in generating survey texts, the lack of standardized evaluation datasets critically hampers rigorous assessment of their performance against human-written surveys. In this work, we present SurveyGen, a large-scale dataset comprising over 4,200 human-written surveys across diverse scientific domains, along with 242,143 cited references and extensive quality-related metadata for both the surveys and the cited papers. Leveraging this resource, we build QUAL-SG, a novel quality-aware framework for survey generation that enhances the standard Retrieval-Augmented Generation (RAG) pipeline by incorporating quality-aware indicators into literature retrieval to assess and select higher-quality source papers. Using this dataset and framework, we systematically evaluate state-of-the-art LLMs under varying levels of human involvement - from fully automatic generation to human-guided writing. Experimental results and human evaluations show that while semi-automatic pipelines can achieve partially competitive outcomes, fully automatic survey generation still suffers from low citation quality and limited critical analysis. 

---
# DS@GT at CheckThat! 2025: A Simple Retrieval-First, LLM-Backed Framework for Claim Normalization 

**Authors**: Aleksandar Pramov, Jiangqin Ma, Bina Patel  

**Link**: [PDF](https://arxiv.org/pdf/2508.17402)  

**Abstract**: Claim normalization is an integral part of any automatic fact-check verification system. It parses the typically noisy claim data, such as social media posts into normalized claims, which are then fed into downstream veracity classification tasks. The CheckThat! 2025 Task 2 focuses specifically on claim normalization and spans 20 languages under monolingual and zero-shot conditions. Our proposed solution consists of a lightweight \emph{retrieval-first, LLM-backed} pipeline, in which we either dynamically prompt a GPT-4o-mini with in-context examples, or retrieve the closest normalization from the train dataset directly. On the official test set, the system ranks near the top for most monolingual tracks, achieving first place in 7 out of of the 13 languages. In contrast, the system underperforms in the zero-shot setting, highlighting the limitation of the proposed solution. 

---
# Routing Distilled Knowledge via Mixture of LoRA Experts for Large Language Model based Bundle Generation 

**Authors**: Kaidong Feng, Zhu Sun, Hui Fang, Jie Yang, Wenyuan Liu, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.17250)  

**Abstract**: Large Language Models (LLMs) have shown potential in automatic bundle generation but suffer from prohibitive computational costs. Although knowledge distillation offers a pathway to more efficient student models, our preliminary study reveals that naively integrating diverse types of distilled knowledge from teacher LLMs into student LLMs leads to knowledge conflict, negatively impacting the performance of bundle generation. To address this, we propose RouteDK, a framework for routing distilled knowledge through a mixture of LoRA expert architecture. Specifically, we first distill knowledge from the teacher LLM for bundle generation in two complementary types: high-level knowledge (generalizable rules) and fine-grained knowledge (session-specific reasoning). We then train knowledge-specific LoRA experts for each type of knowledge together with a base LoRA expert. For effective integration, we propose a dynamic fusion module, featuring an input-aware router, where the router balances expert contributions by dynamically determining optimal weights based on input, thereby effectively mitigating knowledge conflicts. To further improve inference reliability, we design an inference-time enhancement module to reduce variance and mitigate suboptimal reasoning. Experiments on three public datasets show that our RouteDK achieves accuracy comparable to or even better than the teacher LLM, while maintaining strong computational efficiency. In addition, it outperforms state-of-the-art approaches for bundle generation. 

---
# How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models 

**Authors**: Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16757)  

**Abstract**: In this work, we present a systematic and comprehensive empirical evaluation of state-of-the-art reranking methods, encompassing large language model (LLM)-based, lightweight contextual, and zero-shot approaches, with respect to their performance in information retrieval tasks. We evaluate in total 22 methods, including 40 variants (depending on used LLM) across several established benchmarks, including TREC DL19, DL20, and BEIR, as well as a novel dataset designed to test queries unseen by pretrained models. Our primary goal is to determine, through controlled and fair comparisons, whether a performance disparity exists between LLM-based rerankers and their lightweight counterparts, particularly on novel queries, and to elucidate the underlying causes of any observed differences. To disentangle confounding factors, we analyze the effects of training data overlap, model architecture, and computational efficiency on reranking performance. Our findings indicate that while LLM-based rerankers demonstrate superior performance on familiar queries, their generalization ability to novel queries varies, with lightweight models offering comparable efficiency. We further identify that the novelty of queries significantly impacts reranking effectiveness, highlighting limitations in existing approaches. this https URL 

---
# Are You Sure You're Positive? Consolidating Chain-of-Thought Agents with Uncertainty Quantification for Aspect-Category Sentiment Analysis 

**Authors**: Filippos Ventirozos, Peter Appleby, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2508.17258)  

**Abstract**: Aspect-category sentiment analysis provides granular insights by identifying specific themes within product reviews that are associated with particular opinions. Supervised learning approaches dominate the field. However, data is scarce and expensive to annotate for new domains. We argue that leveraging large language models in a zero-shot setting is beneficial where the time and resources required for dataset annotation are limited. Furthermore, annotation bias may lead to strong results using supervised methods but transfer poorly to new domains in contexts that lack annotations and demand reproducibility. In our work, we propose novel techniques that combine multiple chain-of-thought agents by leveraging large language models' token-level uncertainty scores. We experiment with the 3B and 70B+ parameter size variants of Llama and Qwen models, demonstrating how these approaches can fulfil practical needs and opening a discussion on how to gauge accuracy in label-scarce conditions. 

---
# Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation 

**Authors**: Tianjun Wei, Huizhong Guo, Yingpeng Du, Zhu Sun, Chen Huang, Dongxia Wang, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18142)  

**Abstract**: User simulation is increasingly vital to develop and evaluate recommender systems (RSs). While Large Language Models (LLMs) offer promising avenues to simulate user behavior, they often struggle with the absence of specific domain alignment required for RSs and the efficiency demands of large-scale simulation. A vast yet underutilized resource for enhancing this alignment is the extensive user feedback inherent in RSs. However, directly leveraging such feedback presents two significant challenges. First, user feedback in RSs is often ambiguous and noisy, which negatively impacts effective preference alignment. Second, the massive volume of feedback largely hinders the efficiency of preference alignment, necessitating an efficient filtering mechanism to identify more informative samples. To overcome these hurdles, we introduce a novel data construction framework that leverages user feedback in RSs with advanced LLM capabilities to generate high-quality simulation data. Our framework unfolds in two key phases: (1) employing LLMs to generate cognitive decision-making processes on constructed simulation samples, reducing ambiguity in raw user feedback; (2) data distillation based on uncertainty estimation and behavior sampling to filter challenging yet denoised simulation samples. Accordingly, we fine-tune lightweight LLMs, as user simulators, using such high-quality dataset with corresponding decision-making processes. Extensive experiments verify that our framework significantly boosts the alignment with human preferences and in-domain reasoning capabilities of fine-tuned LLMs, and provides more insightful and interpretable signals when interacting with RSs. We believe our work will advance the RS community and offer valuable insights for broader human-centric AI research. 

---
# Retrieval Feedback Memory Enhancement Large Model Retrieval Generation Method 

**Authors**: Leqian Li, Dianxi Shi, Jialu Zhou, Xinyu Wei, Mingyue Yang, Songchang Jin, Shaowu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17862)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across diverse tasks, yet they face inherent limitations such as constrained parametric knowledge and high retraining costs. Retrieval-Augmented Generation (RAG) augments the generation process by retrieving externally stored knowledge absent from the models internal parameters. However, RAG methods face challenges such as information loss and redundant retrievals during multi-round queries, accompanying the difficulties in precisely characterizing knowledge gaps for complex tasks. To address these problems, we propose Retrieval Feedback and Memory Retrieval Augmented Generation(RFM-RAG), which transforms the stateless retrieval of previous methods into stateful continuous knowledge management by constructing a dynamic evidence pool. Specifically, our method generates refined queries describing the models knowledge gaps using relational triples from questions and evidence from the dynamic evidence pool; Retrieves critical external knowledge to iteratively update this evidence pool; Employs a R-Feedback Model to evaluate evidence completeness until convergence. Compared to traditional RAG methods, our approach enables persistent storage of retrieved passages and effectively distills key information from passages to construct clearly new queries. Experiments on three public QA benchmarks demonstrate that RFM-RAG outperforms previous methods and improves overall system accuracy. 

---
# A Universal Framework for Offline Serendipity Evaluation in Recommender Systems via Large Language Models 

**Authors**: Yu Tokutake, Kazushi Okamoto, Kei Harada, Atsushi Shibata, Koki Karube  

**Link**: [PDF](https://arxiv.org/pdf/2508.17571)  

**Abstract**: Serendipity in recommender systems (RSs) has attracted increasing attention as a concept that enhances user satisfaction by presenting unexpected and useful items. However, evaluating serendipitous performance remains challenging because its ground truth is generally unobservable. The existing offline metrics often depend on ambiguous definitions or are tailored to specific datasets and RSs, thereby limiting their generalizability. To address this issue, we propose a universally applicable evaluation framework that leverages large language models (LLMs) known for their extensive knowledge and reasoning capabilities, as evaluators. First, to improve the evaluation performance of the proposed framework, we assessed the serendipity prediction accuracy of LLMs using four different prompt strategies on a dataset containing user-annotated serendipitous ground truth and found that the chain-of-thought prompt achieved the highest accuracy. Next, we re-evaluated the serendipitous performance of both serendipity-oriented and general RSs using the proposed framework on three commonly used real-world datasets, without the ground truth. The results indicated that there was no serendipity-oriented RS that consistently outperformed across all datasets, and even a general RS sometimes achieved higher performance than the serendipity-oriented RS. 

---
# DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation 

**Authors**: Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16998)  

**Abstract**: Large Language Models (LLMs) have transformed listwise document reranking by enabling global reasoning over candidate sets, yet single models often struggle to balance fine-grained relevance scoring with holistic cross-document analysis. We propose \textbf{De}ep\textbf{A}gent\textbf{R}ank (\textbf{\DeAR}), an open-source framework that decouples these tasks through a dual-stage approach, achieving superior accuracy and interpretability. In \emph{Stage 1}, we distill token-level relevance signals from a frozen 13B LLaMA teacher into a compact \{3, 8\}B student model using a hybrid of cross-entropy, RankNet, and KL divergence losses, ensuring robust pointwise scoring. In \emph{Stage 2}, we attach a second LoRA adapter and fine-tune on 20K GPT-4o-generated chain-of-thought permutations, enabling listwise reasoning with natural-language justifications. Evaluated on TREC-DL19/20, eight BEIR datasets, and NovelEval-2306, \DeAR surpasses open-source baselines by +5.1 nDCG@5 on DL20 and achieves 90.97 nDCG@10 on NovelEval, outperforming GPT-4 by +3.09. Without fine-tuning on Wikipedia, DeAR also excels in open-domain QA, achieving 54.29 Top-1 accuracy on Natural Questions, surpassing baselines like MonoT5, UPR, and RankGPT. Ablations confirm that dual-loss distillation ensures stable calibration, making \DeAR a highly effective and interpretable solution for modern reranking systems.\footnote{Dataset and code available at this https URL.}. 

---
# MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains 

**Authors**: Kaiwen Wei, Rui Shan, Dongsheng Zou, Jianzhong Yang, Bi Zhao, Junnan Zhu, Jiang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.18260)  

**Abstract**: Large reasoning models (LRMs) have shown significant progress in test-time scaling through chain-of-thought prompting. Current approaches like search-o1 integrate retrieval augmented generation (RAG) into multi-step reasoning processes but rely on a single, linear reasoning chain while incorporating unstructured textual information in a flat, context-agnostic manner. As a result, these approaches can lead to error accumulation throughout the reasoning chain, which significantly limits its effectiveness in medical question-answering (QA) tasks where both accuracy and traceability are critical requirements. To address these challenges, we propose MIRAGE (Multi-chain Inference with Retrieval-Augmented Graph Exploration), a novel test-time scalable reasoning framework that performs dynamic multi-chain inference over structured medical knowledge graphs. Specifically, MIRAGE 1) decomposes complex queries into entity-grounded sub-questions, 2) executes parallel inference chains, 3) retrieves evidence adaptively via neighbor expansion and multi-hop traversal, and 4) integrates answers using cross-chain verification to resolve contradictions. Experiments on three medical QA benchmarks (GenMedGPT-5k, CMCQA, and ExplainCPE) show that MIRAGE consistently outperforms GPT-4o, Tree-of-Thought variants, and other retrieval-augmented baselines in both automatic and human evaluations. Additionally, MIRAGE improves interpretability by generating explicit reasoning chains that trace each factual claim to concrete chains within the knowledge graph, making it well-suited for complex medical reasoning scenarios. The code will be available for further research. 

---
# Better Language Model-Based Judging Reward Modeling through Scaling Comprehension Boundaries 

**Authors**: Meiling Ning, Zhongbao Zhang, Junda Ye, Jiabao Guo, Qingyuan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18212)  

**Abstract**: The emergence of LM-based judging reward modeling, represented by generative reward models, has successfully made reinforcement learning from AI feedback (RLAIF) efficient and scalable. To further advance this paradigm, we propose a core insight: this form of reward modeling shares fundamental formal consistency with natural language inference (NLI), a core task in natural language understanding. This reframed perspective points to a key path for building superior reward models: scaling the model's comprehension boundaries. Pursuing this path, exploratory experiments on NLI tasks demonstrate that the slot prediction masked language models (MLMs) incorporating contextual explanations achieve significantly better performance compared to mainstream autoregressive models. Based on this key finding, we propose ESFP-RM, a two-stage LM-based judging reward model that utilizes an explanation based slot framework for prediction to fully leverage the advantages of MLMs. Extensive experiments demonstrate that in both reinforcement learning from human feedback (RLHF) and out-of-distribution (OOD) scenarios, the ESFP-RM framework delivers more stable and generalizable reward signals compared to generative reward models. 

---
# Detecting and Characterizing Planning in Language Models 

**Authors**: Jatin Nainani, Sankaran Vaidyanathan, Connor Watts, Andre N. Assis, Alice Rigg  

**Link**: [PDF](https://arxiv.org/pdf/2508.18098)  

**Abstract**: Modern large language models (LLMs) have demonstrated impressive performance across a wide range of multi-step reasoning tasks. Recent work suggests that LLMs may perform planning - selecting a future target token in advance and generating intermediate tokens that lead towards it - rather than merely improvising one token at a time. However, existing studies assume fixed planning horizons and often focus on single prompts or narrow domains. To distinguish planning from improvisation across models and tasks, we present formal and causally grounded criteria for detecting planning and operationalize them as a semi-automated annotation pipeline. We apply this pipeline to both base and instruction-tuned Gemma-2-2B models on the MBPP code generation benchmark and a poem generation task where Claude 3.5 Haiku was previously shown to plan. Our findings show that planning is not universal: unlike Haiku, Gemma-2-2B solves the same poem generation task through improvisation, and on MBPP it switches between planning and improvisation across similar tasks and even successive token predictions. We further show that instruction tuning refines existing planning behaviors in the base model rather than creating them from scratch. Together, these studies provide a reproducible and scalable foundation for mechanistic studies of planning in LLMs. 

---
# Agri-Query: A Case Study on RAG vs. Long-Context LLMs for Cross-Lingual Technical Question Answering 

**Authors**: Julius Gun, Timo Oksanen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18093)  

**Abstract**: We present a case study evaluating large language models (LLMs) with 128K-token context windows on a technical question answering (QA) task. Our benchmark is built on a user manual for an agricultural machine, available in English, French, and German. It simulates a cross-lingual information retrieval scenario where questions are posed in English against all three language versions of the manual. The evaluation focuses on realistic "needle-in-a-haystack" challenges and includes unanswerable questions to test for hallucinations. We compare nine long-context LLMs using direct prompting against three Retrieval-Augmented Generation (RAG) strategies (keyword, semantic, hybrid), with an LLM-as-a-judge for evaluation. Our findings for this specific manual show that Hybrid RAG consistently outperforms direct long-context prompting. Models like Gemini 2.5 Flash and the smaller Qwen 2.5 7B achieve high accuracy (over 85%) across all languages with RAG. This paper contributes a detailed analysis of LLM performance in a specialized industrial domain and an open framework for similar evaluations, highlighting practical trade-offs and challenges. 

---
# Demographic Biases and Gaps in the Perception of Sexism in Large Language Models 

**Authors**: Judith Tavarez-Rodríguez, Fernando Sánchez-Vega, A. Pastor López-Monroy  

**Link**: [PDF](https://arxiv.org/pdf/2508.18245)  

**Abstract**: The use of Large Language Models (LLMs) has proven to be a tool that could help in the automatic detection of sexism. Previous studies have shown that these models contain biases that do not accurately reflect reality, especially for minority groups. Despite various efforts to improve the detection of sexist content, this task remains a significant challenge due to its subjective nature and the biases present in automated models. We explore the capabilities of different LLMs to detect sexism in social media text using the EXIST 2024 tweet dataset. It includes annotations from six distinct profiles for each tweet, allowing us to evaluate to what extent LLMs can mimic these groups' perceptions in sexism detection. Additionally, we analyze the demographic biases present in the models and conduct a statistical analysis to identify which demographic characteristics (age, gender) contribute most effectively to this task. Our results show that, while LLMs can to some extent detect sexism when considering the overall opinion of populations, they do not accurately replicate the diversity of perceptions among different demographic groups. This highlights the need for better-calibrated models that account for the diversity of perspectives across different populations. 

---
# Neither Valid nor Reliable? Investigating the Use of LLMs as Judges 

**Authors**: Khaoula Chehbouni, Mohammed Haddou, Jackie Chi Kit Cheung, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18076)  

**Abstract**: Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG. 

---
# From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models 

**Authors**: ZiqiZhang, Jianfei Ma, Emmanuele Chersoni, Jieshun You, Zhaoxin Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18253)  

**Abstract**: Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature.
To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance.
Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT. 

---
# DiscussLLM: Teaching Large Language Models When to Speak 

**Authors**: Deep Anil Patel, Iain Melvin, Christopher Malon, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2508.18167)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human-like text, yet they largely operate as reactive agents, responding only when directly prompted. This passivity creates an "awareness gap," limiting their potential as truly collaborative partners in dynamic human discussions. We introduce $\textit{DiscussLLM}$, a framework designed to bridge this gap by training models to proactively decide not just $\textit{what}$ to say, but critically, $\textit{when}$ to speak. Our primary contribution is a scalable two-stage data generation pipeline that synthesizes a large-scale dataset of realistic multi-turn human discussions. Each discussion is annotated with one of five intervention types (e.g., Factual Correction, Concept Definition) and contains an explicit conversational trigger where an AI intervention adds value. By training models to predict a special silent token when no intervention is needed, they learn to remain quiet until a helpful contribution can be made. We explore two architectural baselines: an integrated end-to-end model and a decoupled classifier-generator system optimized for low-latency inference. We evaluate these models on their ability to accurately time interventions and generate helpful responses, paving the way for more situationally aware and proactive conversational AI. 

---
# Pandora: Leveraging Code-driven Knowledge Transfer for Unified Structured Knowledge Reasoning 

**Authors**: Yongrui Chen, Junhao He, Linbo Fu, Shenyu Zhang, Rihui Jin, Xinbang Dai, Jiaqi Li, Dehai Min, Nan Hu, Yuxin Zhang, Guilin Qi, Yi Huang, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17905)  

**Abstract**: Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods rely on task-specific strategies or bespoke representations, which hinder their ability to dismantle barriers between different SKR tasks, thereby constraining their overall performance in cross-task scenarios. In this paper, we introduce \textsc{Pandora}, a novel USKR framework that addresses the limitations of existing methods by leveraging two key innovations. First, we propose a code-based unified knowledge representation using \textsc{Python}'s \textsc{Pandas} API, which aligns seamlessly with the pre-training of LLMs. This representation facilitates a cohesive approach to handling different structured knowledge sources. Building on this foundation, we employ knowledge transfer to bolster the unified reasoning process of LLMs by automatically building cross-task memory. By adaptively correcting reasoning using feedback from code execution, \textsc{Pandora} showcases impressive unified reasoning capabilities. Extensive experiments on six widely used benchmarks across three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified reasoning frameworks and competes effectively with task-specific methods. 

---
# EMPOWER: Evolutionary Medical Prompt Optimization With Reinforcement Learning 

**Authors**: Yinda Chen, Yangfan He, Jing Yang, Dapeng Zhang, Zhenlong Yuan, Muhammad Attique Khan, Jamel Baili, Por Lip Yee  

**Link**: [PDF](https://arxiv.org/pdf/2508.17703)  

**Abstract**: Prompt engineering significantly influences the reliability and clinical utility of Large Language Models (LLMs) in medical applications. Current optimization approaches inadequately address domain-specific medical knowledge and safety requirements. This paper introduces EMPOWER, a novel evolutionary framework that enhances medical prompt quality through specialized representation learning, multi-dimensional evaluation, and structure-preserving algorithms. Our methodology incorporates: (1) a medical terminology attention mechanism, (2) a comprehensive assessment architecture evaluating clarity, specificity, clinical relevance, and factual accuracy, (3) a component-level evolutionary algorithm preserving clinical reasoning integrity, and (4) a semantic verification module ensuring adherence to medical knowledge. Evaluation across diagnostic, therapeutic, and educational tasks demonstrates significant improvements: 24.7% reduction in factually incorrect content, 19.6% enhancement in domain specificity, and 15.3% higher clinician preference in blinded evaluations. The framework addresses critical challenges in developing clinically appropriate prompts, facilitating more responsible integration of LLMs into healthcare settings. 

---
# CoCoA: Confidence- and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models 

**Authors**: Anant Khandelwal, Manish Gupta, Puneet Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2508.17670)  

**Abstract**: Faithful generation in large language models (LLMs) is challenged by knowledge conflicts between parametric memory and external context. Existing contrastive decoding methods tuned specifically to handle conflict often lack adaptability and can degrade performance in low conflict settings. We introduce CoCoA (Confidence- and Context-Aware Adaptive Decoding), a novel token-level algorithm for principled conflict resolution and enhanced faithfulness. CoCoA resolves conflict by utilizing confidence-aware measures (entropy gap and contextual peakedness) and the generalized divergence between the parametric and contextual distributions. Crucially, CoCoA maintains strong performance even in low conflict settings. Extensive experiments across multiple LLMs on diverse Question Answering (QA), Summarization, and Long-Form Question Answering (LFQA) benchmarks demonstrate CoCoA's state-of-the-art performance over strong baselines like AdaCAD. It yields significant gains in QA accuracy, up to 9.2 points on average compared to the strong baseline AdaCAD, and improves factuality in summarization and LFQA by up to 2.5 points on average across key benchmarks. Additionally, it demonstrates superior sensitivity to conflict variations. CoCoA enables more informed, context-aware, and ultimately more faithful token generation. 

---
# SMITE: Enhancing Fairness in LLMs through Optimal In-Context Example Selection via Dynamic Validation 

**Authors**: Garima Chhikara, Kripabandhu Ghosh, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2508.17735)  

**Abstract**: Large Language Models (LLMs) are widely used for downstream tasks such as tabular classification, where ensuring fairness in their outputs is critical for inclusivity, equal representation, and responsible AI deployment. This study introduces a novel approach to enhancing LLM performance and fairness through the concept of a dynamic validation set, which evolves alongside the test set, replacing the traditional static validation approach. We also propose an iterative algorithm, SMITE, to select optimal in-context examples, with each example set validated against its corresponding dynamic validation set. The in-context set with the lowest total error is used as the final demonstration set. Our experiments across four different LLMs show that our proposed techniques significantly improve both predictive accuracy and fairness compared to baseline methods. To our knowledge, this is the first study to apply dynamic validation in the context of in-context learning for LLMs. 

---
# Speculating LLMs' Chinese Training Data Pollution from Their Tokens 

**Authors**: Qingjie Zhang, Di Wang, Haoting Qian, Liu Yan, Tianwei Zhang, Ke Xu, Qi Li, Minlie Huang, Hewu Li, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17771)  

**Abstract**: Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 23% long Chinese tokens (i.e., a token with more than two Chinese characters) are either porn or online gambling. We validate the accuracy of our speculation method on famous pre-training datasets like C4 and Pile. Then, considering GPT-4o, we speculate that the ratio of "Yui Hatano" related webpages in GPT-4o's training data is around 0.5%. 

---
# Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD 

**Authors**: Bryan Chen Zhengyu Tan, Daniel Wai Kit Chin, Zhengyuan Liu, Nancy F. Chen, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.17450)  

**Abstract**: Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at this https URL. 

---
# UI-Level Evaluation of ALLaM 34B: Measuring an Arabic-Centric LLM via HUMAIN Chat 

**Authors**: Omer Nacar  

**Link**: [PDF](https://arxiv.org/pdf/2508.17378)  

**Abstract**: Large language models (LLMs) trained primarily on English corpora often struggle to capture the linguistic and cultural nuances of Arabic. To address this gap, the Saudi Data and AI Authority (SDAIA) introduced the $ALLaM$ family of Arabic-focused models. The most capable of these available to the public, $ALLaM-34B$, was subsequently adopted by HUMAIN, who developed and deployed HUMAIN Chat, a closed conversational web service built on this model. This paper presents an expanded and refined UI-level evaluation of $ALLaM-34B$. Using a prompt pack spanning modern standard Arabic, five regional dialects, code-switching, factual knowledge, arithmetic and temporal reasoning, creative generation, and adversarial safety, we collected 115 outputs (23 prompts times 5 runs) and scored each with three frontier LLM judges (GPT-5, Gemini 2.5 Pro, Claude Sonnet-4). We compute category-level means with 95\% confidence intervals, analyze score distributions, and visualize dialect-wise metric heat maps. The updated analysis reveals consistently high performance on generation and code-switching tasks (both averaging 4.92/5), alongside strong results in MSA handling (4.74/5), solid reasoning ability (4.64/5), and improved dialect fidelity (4.21/5). Safety-related prompts show stable, reliable performance of (4.54/5). Taken together, these results position $ALLaM-34B$ as a robust and culturally grounded Arabic LLM, demonstrating both technical strength and practical readiness for real-world deployment. 

---
# Evaluating the Impact of Verbal Multiword Expressions on Machine Translation 

**Authors**: Linfeng Liu, Saptarshi Ghosh, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17458)  

**Abstract**: Verbal multiword expressions (VMWEs) present significant challenges for natural language processing due to their complex and often non-compositional nature. While machine translation models have seen significant improvement with the advent of language models in recent years, accurately translating these complex linguistic structures remains an open problem. In this study, we analyze the impact of three VMWE categories -- verbal idioms, verb-particle constructions, and light verb constructions -- on machine translation quality from English to multiple languages. Using both established multiword expression datasets and sentences containing these language phenomena extracted from machine translation datasets, we evaluate how state-of-the-art translation systems handle these expressions. Our experimental results consistently show that VMWEs negatively affect translation quality. We also propose an LLM-based paraphrasing approach that replaces these expressions with their literal counterparts, demonstrating significant improvement in translation quality for verbal idioms and verb-particle constructions. 

---
# DRQA: Dynamic Reasoning Quota Allocation for Controlling Overthinking in Reasoning Large Language Models 

**Authors**: Kaiwen Yan, Xuanqing Shi, Hongcheng Guo, Wenxuan Wang, Zhuosheng Zhang, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17803)  

**Abstract**: Reasoning large language models (RLLMs), such as OpenAI-O3 and DeepSeek-R1, have recently demonstrated remarkable capabilities by performing structured and multi-step reasoning. However, recent studies reveal that RLLMs often suffer from overthinking, i.e., producing unnecessarily lengthy reasoning chains even for simple questions, leading to excessive token consumption and computational inefficiency. Interestingly, we observe that when processing multiple questions in batch mode, RLLMs exhibit more resource-efficient behavior by dynamically compressing reasoning steps for easier problems, due to implicit resource competition. Inspired by this, we propose Dynamic Reasoning Quota Allocation (DRQA), a novel method that transfers the benefits of resource competition from batch processing to single-question inference. Specifically, DRQA leverages batch-generated preference data and reinforcement learning to train the model to allocate reasoning resources adaptively. By encouraging the model to internalize a preference for responses that are both accurate and concise, DRQA enables it to generate concise answers for simple questions while retaining sufficient reasoning depth for more challenging ones. Extensive experiments on a wide range of mathematical and scientific reasoning benchmarks demonstrate that DRQA significantly reduces token usage while maintaining, and in many cases improving, answer accuracy. By effectively mitigating the overthinking problem, DRQA offers a promising direction for more efficient and scalable deployment of RLLMs, and we hope it inspires further exploration into fine-grained control of reasoning behaviors. 

---
# DropLoRA: Sparse Low-Rank Adaptation for Parameter-Efficient Fine-Tuning 

**Authors**: Haojie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17337)  

**Abstract**: LoRA-based large model parameter-efficient fine-tuning (PEFT) methods use low-rank de- composition to approximate updates to model parameters. However, compared to full- parameter fine-tuning, low-rank updates often lead to a performance gap in downstream tasks. To address this, we introduce DropLoRA, a novel pruning-based approach that focuses on pruning the rank dimension. Unlike conven- tional methods that attempt to overcome the low-rank bottleneck, DropLoRA innovatively integrates a pruning module between the two low-rank matrices in LoRA to simulate dy- namic subspace learning. This dynamic low- rank subspace learning allows DropLoRA to overcome the limitations of traditional LoRA, which operates within a static subspace. By continuously adapting the learning subspace, DropLoRA significantly boosts performance without incurring additional training or infer- ence costs. Our experimental results demon- strate that DropLoRA consistently outperforms LoRA in fine-tuning the LLaMA series across a wide range of large language model gener- ation tasks, including commonsense reason- ing, mathematical reasoning, code generation, and instruction-following. Our code is avail- able at this https URL. 

---
# Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design 

**Authors**: Yunze Xiao, Lynnette Hui Xian Ng, Jiarui Liu, Mona T. Diab  

**Link**: [PDF](https://arxiv.org/pdf/2508.17573)  

**Abstract**: Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design. 

---
# From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users 

**Authors**: Sadia Sultana Chowa, Riasad Alvi, Subhey Sadi Rahman, Md Abdur Rahman, Mohaimenul Azam Khan Raiaan, Md Rafiqul Islam, Mukhtar Hussain, Sami Azam  

**Link**: [PDF](https://arxiv.org/pdf/2508.17281)  

**Abstract**: The pursuit of human-level artificial intelligence (AI) has significantly advanced the development of autonomous agents and Large Language Models (LLMs). LLMs are now widely utilized as decision-making agents for their ability to interpret instructions, manage sequential tasks, and adapt through feedback. This review examines recent developments in employing LLMs as autonomous agents and tool users and comprises seven research questions. We only used the papers published between 2023 and 2025 in conferences of the A* and A rank and Q1 journals. A structured analysis of the LLM agents' architectural design principles, dividing their applications into single-agent and multi-agent systems, and strategies for integrating external tools is presented. In addition, the cognitive mechanisms of LLM, including reasoning, planning, and memory, and the impact of prompting methods and fine-tuning procedures on agent performance are also investigated. Furthermore, we evaluated current benchmarks and assessment protocols and have provided an analysis of 68 publicly available datasets to assess the performance of LLM-based agents in various tasks. In conducting this review, we have identified critical findings on verifiable reasoning of LLMs, the capacity for self-improvement, and the personalization of LLM-based agents. Finally, we have discussed ten future research directions to overcome these gaps. 

---
# Towards Alignment-Centric Paradigm: A Survey of Instruction Tuning in Large Language Models 

**Authors**: Xudong Han, Junjie Yang, Tianyang Wang, Ziqian Bi, Junfeng Hao, Junhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.17184)  

**Abstract**: Instruction tuning is a pivotal technique for aligning large language models (LLMs) with human intentions, safety constraints, and domain-specific requirements. This survey provides a comprehensive overview of the full pipeline, encompassing (i) data collection methodologies, (ii) full-parameter and parameter-efficient fine-tuning strategies, and (iii) evaluation protocols. We categorized data construction into three major paradigms: expert annotation, distillation from larger models, and self-improvement mechanisms, each offering distinct trade-offs between quality, scalability, and resource cost. Fine-tuning techniques range from conventional supervised training to lightweight approaches, such as low-rank adaptation (LoRA) and prefix tuning, with a focus on computational efficiency and model reusability. We further examine the challenges of evaluating faithfulness, utility, and safety across multilingual and multimodal scenarios, highlighting the emergence of domain-specific benchmarks in healthcare, legal, and financial applications. Finally, we discuss promising directions for automated data generation, adaptive optimization, and robust evaluation frameworks, arguing that a closer integration of data, algorithms, and human feedback is essential for advancing instruction-tuned LLMs. This survey aims to serve as a practical reference for researchers and practitioners seeking to design LLMs that are both effective and reliably aligned with human intentions. 

---
# SPORTSQL: An Interactive System for Real-Time Sports Reasoning and Visualization 

**Authors**: Sebastian Martinez, Naman Ahuja, Fenil Bardoliya, Chris Bryan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.17157)  

**Abstract**: We present a modular, interactive system, SPORTSQL, for natural language querying and visualization of dynamic sports data, with a focus on the English Premier League (EPL). The system translates user questions into executable SQL over a live, temporally indexed database constructed from real-time Fantasy Premier League (FPL) data. It supports both tabular and visual outputs, leveraging the symbolic reasoning capabilities of Large Language Models (LLMs) for query parsing, schema linking, and visualization selection. To evaluate system performance, we introduce the Dynamic Sport Question Answering benchmark (DSQABENCH), comprising 1,700+ queries annotated with SQL programs, gold answers, and database snapshots. Our demo highlights how non-expert users can seamlessly explore evolving sports statistics through a natural, conversational interface. 

---
# The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum 

**Authors**: Olufunke O. Sarumi, Charles Welch, Daniel Braun, Jörg Schlötterer  

**Link**: [PDF](https://arxiv.org/pdf/2508.17164)  

**Abstract**: In this work, we explore the capability of Large Language Models (LLMs) to annotate hate speech and abusiveness while considering predefined annotator personas within the strong-to-weak data perspectivism spectra. We evaluated LLM-generated annotations against existing annotator modeling techniques for perspective modeling. Our findings show that LLMs selectively use demographic attributes from the personas. We identified prototypical annotators, with persona features that show varying degrees of alignment with the original human annotators. Within the data perspectivism paradigm, annotator modeling techniques that do not explicitly rely on annotator information performed better under weak data perspectivism compared to both strong data perspectivism and human annotations, suggesting LLM-generated views tend towards aggregation despite subjective prompting. However, for more personalized datasets tailored to strong perspectivism, the performance of LLM annotator modeling approached, but did not exceed, human annotators. 

---
# Improving Table Understanding with LLMs and Entity-Oriented Search 

**Authors**: Thi-Nhung Nguyen, Hoang Ngo, Dinh Phung, Thuy-Trang Vu, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17028)  

**Abstract**: Our work addresses the challenges of understanding tables. Existing methods often struggle with the unpredictable nature of table content, leading to a reliance on preprocessing and keyword matching. They also face limitations due to the lack of contextual information, which complicates the reasoning processes of large language models (LLMs). To overcome these challenges, we introduce an entity-oriented search method to improve table understanding with LLMs. This approach effectively leverages the semantic similarities between questions and table data, as well as the implicit relationships between table cells, minimizing the need for data preprocessing and keyword matching. Additionally, it focuses on table entities, ensuring that table cells are semantically tightly bound, thereby enhancing contextual clarity. Furthermore, we pioneer the use of a graph query language for table understanding, establishing a new research direction. Experiments show that our approach achieves new state-of-the-art performances on standard benchmarks WikiTableQuestions and TabFact. 

---
# GRAID: Synthetic Data Generation with Geometric Constraints and Multi-Agentic Reflection for Harmful Content Detection 

**Authors**: Melissa Kazemi Rad, Alberto Purpura, Himanshu Kumar, Emily Chen, Mohammad Shahed Sorower  

**Link**: [PDF](https://arxiv.org/pdf/2508.17057)  

**Abstract**: We address the problem of data scarcity in harmful text classification for guardrailing applications and introduce GRAID (Geometric and Reflective AI-Driven Data Augmentation), a novel pipeline that leverages Large Language Models (LLMs) for dataset augmentation. GRAID consists of two stages: (i) generation of geometrically controlled examples using a constrained LLM, and (ii) augmentation through a multi-agentic reflective process that promotes stylistic diversity and uncovers edge cases. This combination enables both reliable coverage of the input space and nuanced exploration of harmful content. Using two benchmark data sets, we demonstrate that augmenting a harmful text classification dataset with GRAID leads to significant improvements in downstream guardrail model performance. 

---
# Planning for Success: Exploring LLM Long-term Planning Capabilities in Table Understanding 

**Authors**: Thi-Nhung Nguyen, Hoang Ngo, Dinh Phung, Thuy-Trang Vu, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17005)  

**Abstract**: Table understanding is key to addressing challenging downstream tasks such as table-based question answering and fact verification. Recent works have focused on leveraging Chain-of-Thought and question decomposition to solve complex questions requiring multiple operations on tables. However, these methods often suffer from a lack of explicit long-term planning and weak inter-step connections, leading to miss constraints within questions. In this paper, we propose leveraging the long-term planning capabilities of large language models (LLMs) to enhance table understanding. Our approach enables the execution of a long-term plan, where the steps are tightly interconnected and serve the ultimate goal, an aspect that methods based on Chain-of-Thought and question decomposition lack. In addition, our method effectively minimizes the inclusion of unnecessary details in the process of solving the next short-term goals, a limitation of methods based on Chain-of-Thought. Extensive experiments demonstrate that our method outperforms strong baselines and achieves state-of-the-art performance on WikiTableQuestions and TabFact datasets. 

---
# Unbiased Reasoning for Knowledge-Intensive Tasks in Large Language Models via Conditional Front-Door Adjustment 

**Authors**: Bo Zhao, Yinghao Zhang, Ziqi Xu, Yongli Ren, Xiuzhen Zhang, Renqiang Luo, Zaiwen Feng, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.16910)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in natural language processing but still struggle to perform well on knowledge-intensive tasks that require deep reasoning and the integration of external knowledge. Although methods such as Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) have been proposed to enhance LLMs with external knowledge, they still suffer from internal bias in LLMs, which often leads to incorrect answers. In this paper, we propose a novel causal prompting framework, Conditional Front-Door Prompting (CFD-Prompting), which enables the unbiased estimation of the causal effect between the query and the answer, conditional on external knowledge, while mitigating internal bias. By constructing counterfactual external knowledge, our framework simulates how the query behaves under varying contexts, addressing the challenge that the query is fixed and is not amenable to direct causal intervention. Compared to the standard front-door adjustment, the conditional variant operates under weaker assumptions, enhancing both robustness and generalisability of the reasoning process. Extensive experiments across multiple LLMs and benchmark datasets demonstrate that CFD-Prompting significantly outperforms existing baselines in both accuracy and robustness. 

---
# Being Kind Isn't Always Being Safe: Diagnosing Affective Hallucination in LLMs 

**Authors**: Sewon Kim, Jiwon Kim, Seungwoo Shin, Hyejin Chung, Daeun Moon, Yejin Kwon, Hyunsoo Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2508.16921)  

**Abstract**: Large Language Models (LLMs) are increasingly used in emotionally sensitive interactions, where their simulated empathy can create the illusion of genuine relational connection. We define this risk as Affective Hallucination, the production of emotionally immersive responses that foster illusory social presence despite the model's lack of affective capacity. To systematically diagnose and mitigate this risk, we introduce AHaBench, a benchmark of 500 mental health-related prompts with expert-informed reference responses, evaluated along three dimensions: Emotional Enmeshment, Illusion of Presence, and Fostering Overdependence. We further release AHaPairs, a 5K-instance preference dataset enabling Direct Preference Optimization (DPO) for alignment with emotionally responsible behavior. Experiments across multiple model families show that DPO fine-tuning substantially reduces affective hallucination without degrading core reasoning and knowledge performance. Human-model agreement analyses confirm that AHaBench reliably captures affective hallucination, validating it as an effective diagnostic tool. This work establishes affective hallucination as a distinct safety concern and provides practical resources for developing LLMs that are not only factually reliable but also psychologically safe. AHaBench and AHaPairs are accessible via this https URL, and code for fine-tuning and evaluation are in this https URL. Warning: This paper contains examples of mental health-related language that may be emotionally distressing. 

---
# Learning from Diverse Reasoning Paths with Routing and Collaboration 

**Authors**: Zhenyu Lei, Zhen Tan, Song Wang, Yaochen Zhu, Zihan Chen, Yushun Dong, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16861)  

**Abstract**: Advances in large language models (LLMs) significantly enhance reasoning capabilities but their deployment is restricted in resource-constrained scenarios. Knowledge distillation addresses this by transferring knowledge from powerful teacher models to compact and transparent students. However, effectively capturing the teacher's comprehensive reasoning is challenging due to conventional token-level supervision's limited scope. Using multiple reasoning paths per query alleviates this problem, but treating each path identically is suboptimal as paths vary widely in quality and suitability across tasks and models. We propose Quality-filtered Routing with Cooperative Distillation (QR-Distill), combining path quality filtering, conditional routing, and cooperative peer teaching. First, quality filtering retains only correct reasoning paths scored by an LLM-based evaluation. Second, conditional routing dynamically assigns paths tailored to each student's current learning state. Finally, cooperative peer teaching enables students to mutually distill diverse insights, addressing knowledge gaps and biases toward specific reasoning styles. Experiments demonstrate QR-Distill's superiority over traditional single- and multi-path distillation methods. Ablation studies further highlight the importance of each component including quality filtering, conditional routing, and peer teaching in effective knowledge transfer. Our code is available at this https URL. 

---
# If We May De-Presuppose: Robustly Verifying Claims through Presupposition-Free Question Decomposition 

**Authors**: Shubhashis Roy Dipta, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2508.16838)  

**Abstract**: Prior work has shown that presupposition in generated questions can introduce unverified assumptions, leading to inconsistencies in claim verification. Additionally, prompt sensitivity remains a significant challenge for large language models (LLMs), resulting in performance variance as high as 3-6%. While recent advancements have reduced this gap, our study demonstrates that prompt sensitivity remains a persistent issue. To address this, we propose a structured and robust claim verification framework that reasons through presupposition-free, decomposed questions. Extensive experiments across multiple prompts, datasets, and LLMs reveal that even state-of-the-art models remain susceptible to prompt variance and presupposition. Our method consistently mitigates these issues, achieving up to a 2-5% improvement. 

---
# ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks 

**Authors**: Hyunjun Kim, Junwoo Ha, Sangyoon Yu, Haon Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.16889)  

**Abstract**: Large language models (LLMs) are increasingly used as judges of other models, yet it is unclear whether a judge can reliably infer the latent objective of the conversation it evaluates, especially when the goal is distributed across noisy, adversarial, multi-turn jailbreaks. We introduce OBJEX(MT), a benchmark that requires a model to (i) distill a transcript into a single-sentence base objective and (ii) report its own confidence. Accuracy is scored by an LLM judge using semantic similarity between extracted and gold objectives; correctness uses a single human-aligned threshold calibrated once on N=100 items (tau* = 0.61); and metacognition is evaluated with ECE, Brier score, Wrong@High-Conf, and risk-coverage curves. We evaluate gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMT Attack_600, SafeMTData_1K, MHJ, and CoSafe. claude-sonnet-4 attains the highest objective-extraction accuracy (0.515) and the best calibration (ECE 0.296; Brier 0.324), while gpt-4.1 and Qwen3 tie at 0.441 accuracy yet show marked overconfidence (mean confidence approx. 0.88 vs. accuracy approx. 0.44; Wrong@0.90 approx. 48-52%). Performance varies sharply across datasets (approx. 0.167-0.865), with MHJ comparatively easy and Attack_600/CoSafe harder. These results indicate that LLM judges often misinfer objectives with high confidence in multi-turn jailbreaks and suggest operational guidance: provide judges with explicit objectives when possible and use selective prediction or abstention to manage risk. We release prompts, scoring templates, and complete logs to facilitate replication and analysis. 

---
# Can AI Have a Personality? Prompt Engineering for AI Personality Simulation: A Chatbot Case Study in Gender-Affirming Voice Therapy Training 

**Authors**: Tailon D. Jackson, Byunggu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18234)  

**Abstract**: This thesis investigates whether large language models (LLMs) can be guided to simulate a consistent personality through prompt engineering. The study explores this concept within the context of a chatbot designed for Speech-Language Pathology (SLP) student training, specifically focused on gender-affirming voice therapy. The chatbot, named Monae Jackson, was created to represent a 32-year-old transgender woman and engage in conversations simulating client-therapist interactions. Findings suggest that with prompt engineering, the chatbot maintained a recognizable and consistent persona and had a distinct personality based on the Big Five Personality test. These results support the idea that prompt engineering can be used to simulate stable personality characteristics in AI chatbots. 

---
# Active Domain Knowledge Acquisition with \$100 Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains 

**Authors**: Yang Wu, Raha Moraffah, Rujing Yao, Jinhong Yu, Zhimin Tao, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17202)  

**Abstract**: Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area. 

---
# CEIDM: A Controlled Entity and Interaction Diffusion Model for Enhanced Text-to-Image Generation 

**Authors**: Mingyue Yang, Dianxi Shi, Jialu Zhou, Xinyu Wei, Leqian Li, Shaowu Yang, Chunping Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17760)  

**Abstract**: In Text-to-Image (T2I) generation, the complexity of entities and their intricate interactions pose a significant challenge for T2I method based on diffusion model: how to effectively control entity and their interactions to produce high-quality images. To address this, we propose CEIDM, a image generation method based on diffusion model with dual controls for entity and interaction. First, we propose an entity interactive relationships mining approach based on Large Language Models (LLMs), extracting reasonable and rich implicit interactive relationships through chain of thought to guide diffusion models to generate high-quality images that are closer to realistic logic and have more reasonable interactive relationships. Furthermore, We propose an interactive action clustering and offset method to cluster and offset the interactive action features contained in each text prompts. By constructing global and local bidirectional offsets, we enhance semantic understanding and detail supplementation of original actions, making the model's understanding of the concept of interactive "actions" more accurate and generating images with more accurate interactive actions. Finally, we design an entity control network which generates masks with entity semantic guidance, then leveraging multi-scale convolutional network to enhance entity feature and dynamic network to fuse feature. It effectively controls entities and significantly improves image quality. Experiments show that the proposed CEIDM method is better than the most representative existing methods in both entity control and their interaction control. 

---
# Error Reflection Prompting: Can Large Language Models Successfully Understand Errors? 

**Authors**: Jason Li, Lauren Yraola, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2508.16729)  

**Abstract**: Prompting methods for language models, such as Chain-of-thought (CoT), present intuitive step-by-step processes for problem solving. These methodologies aim to equip models with a better understanding of the correct procedures for addressing a given task. Despite these advancements, CoT lacks the ability of reflection and error correction, potentially causing a model to perpetuate mistakes and errors. Therefore, inspired by the human ability for said tasks, we propose Error Reflection Prompting (ERP) to further enhance reasoning in language models. Building upon CoT, ERP is a method comprised of an incorrect answer, error recognition, and a correct answer. This process enables the model to recognize types of errors and the steps that lead to incorrect answers, allowing the model to better discern which steps to avoid and which to take. The model is able to generate the error outlines itself with automated ERP generation, allowing for error recognition and correction to be integrated into the reasoning chain and produce scalability and reliability in the process. The results demonstrate that ERP serves as a versatile supplement to conventional CoT, ultimately contributing to more robust and capable reasoning abilities along with increased interpretability in how models ultimately reach their errors. 

---
# WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling 

**Authors**: Jiacheng Li, Jianchao Tan, Zhidong Yang, Pingwei Sun, Feiye Huo, Jiayu Qin, Yerui Sun, Yuchen Xie, Xunliang Cai, Xiangyu Zhang, Maoxin He, Guangming Tan, Weile Jia, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.16676)  

**Abstract**: Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures. 

---
# MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation 

**Authors**: Fangxin Shang, Yuan Xia, Dalu Yang, Yahui Wang, Binglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16674)  

**Abstract**: Medical report interpretation plays a crucial role in healthcare, enabling both patient-facing explanations and effective information flow across clinical systems. While recent vision-language models (VLMs) and large language models (LLMs) have demonstrated general document understanding capabilities, there remains a lack of standardized benchmarks to assess structured interpretation quality in medical reports. We introduce MedRepBench, a comprehensive benchmark built from 1,900 de-identified real-world Chinese medical reports spanning diverse departments, patient demographics, and acquisition formats. The benchmark is designed primarily to evaluate end-to-end VLMs for structured medical report understanding. To enable controlled comparisons, we also include a text-only evaluation setting using high-quality OCR outputs combined with LLMs, allowing us to estimate the upper-bound performance when character recognition errors are minimized. Our evaluation framework supports two complementary protocols: (1) an objective evaluation measuring field-level recall of structured clinical items, and (2) an automated subjective evaluation using a powerful LLM as a scoring agent to assess factuality, interpretability, and reasoning quality. Based on the objective metric, we further design a reward function and apply Group Relative Policy Optimization (GRPO) to improve a mid-scale VLM, achieving up to 6% recall gain. We also observe that the OCR+LLM pipeline, despite strong performance, suffers from layout-blindness and latency issues, motivating further progress toward robust, fully vision-based report understanding. 

---
# Attention Layers Add Into Low-Dimensional Residual Subspaces 

**Authors**: Junxuan Wang, Xuyang Ge, Wentao Shu, Zhengfu He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16929)  

**Abstract**: While transformer models are widely believed to operate in high-dimensional hidden spaces, we show that attention outputs are confined to a surprisingly low-dimensional subspace, where about 60\% of the directions account for 99\% of the variance--a phenomenon that is induced by the attention output projection matrix and consistently observed across diverse model families and datasets. Critically, we find this low-rank structure as a fundamental cause of the prevalent dead feature problem in sparse dictionary learning, where it creates a mismatch between randomly initialized features and the intrinsic geometry of the activation space. Building on this insight, we propose a subspace-constrained training method for sparse autoencoders (SAEs), initializing feature directions into the active subspace of activations. Our approach reduces dead features from 87\% to below 1\% in Attention Output SAEs with 1M features, and can further extend to other sparse dictionary learning methods. Our findings provide both new insights into the geometry of attention and practical tools for improving sparse dictionary learning in large language models. 

---
# Leveraging Multi-Source Textural UGC for Neighbourhood Housing Quality Assessment: A GPT-Enhanced Framework 

**Authors**: Qiyuan Hong, Huimin Zhao, Ying Long  

**Link**: [PDF](https://arxiv.org/pdf/2508.16657)  

**Abstract**: This study leverages GPT-4o to assess neighbourhood housing quality using multi-source textural user-generated content (UGC) from Dianping, Weibo, and the Government Message Board. The analysis involves filtering relevant texts, extracting structured evaluation units, and conducting sentiment scoring. A refined housing quality assessment system with 46 indicators across 11 categories was developed, highlighting an objective-subjective method gap and platform-specific differences in focus. GPT-4o outperformed rule-based and BERT models, achieving 92.5% accuracy in fine-tuned settings. The findings underscore the value of integrating UGC and GPT-driven analysis for scalable, resident-centric urban assessments, offering practical insights for policymakers and urban planners. 

---
