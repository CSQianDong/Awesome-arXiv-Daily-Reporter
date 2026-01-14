# Uncovering Political Bias in Large Language Models using Parliamentary Voting Records 

**Authors**: Jieying Chen, Karen de Jong, Andreas Poole, Jan Burakowski, Elena Elderson Nosti, Joep Windt, Chendi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08785)  

**Abstract**: As large language models (LLMs) become deeply embedded in digital platforms and decision-making systems, concerns about their political biases have grown. While substantial work has examined social biases such as gender and race, systematic studies of political bias remain limited, despite their direct societal impact. This paper introduces a general methodology for constructing political bias benchmarks by aligning model-generated voting predictions with verified parliamentary voting records. We instantiate this methodology in three national case studies: PoliBiasNL (2,701 Dutch parliamentary motions and votes from 15 political parties), PoliBiasNO (10,584 motions and votes from 9 Norwegian parties), and PoliBiasES (2,480 motions and votes from 10 Spanish parties). Across these benchmarks, we assess ideological tendencies and political entity bias in LLM behavior. As part of our evaluation framework, we also propose a method to visualize the ideology of LLMs and political parties in a shared two-dimensional CHES (Chapel Hill Expert Survey) space by linking their voting-based positions to the CHES dimensions, enabling direct and interpretable comparisons between models and real-world political actors. Our experiments reveal fine-grained ideological distinctions: state-of-the-art LLMs consistently display left-leaning or centrist tendencies, alongside clear negative biases toward right-conservative parties. These findings highlight the value of transparent, cross-national evaluation grounded in real parliamentary behavior for understanding and auditing political bias in modern LLMs. 

---
# Pervasive Annotation Errors Break Text-to-SQL Benchmarks and Leaderboards 

**Authors**: Tengjun Jin, Yoojin Choi, Yuxuan Zhu, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08778)  

**Abstract**: Researchers have proposed numerous text-to-SQL techniques to streamline data analytics and accelerate the development of database-driven applications. To compare these techniques and select the best one for deployment, the community depends on public benchmarks and their leaderboards. Since these benchmarks heavily rely on human annotations during question construction and answer evaluation, the validity of the annotations is crucial.
In this paper, we conduct an empirical study that (i) benchmarks annotation error rates for two widely used text-to-SQL benchmarks, BIRD and Spider 2.0-Snow, and (ii) corrects a subset of the BIRD development (Dev) set to measure the impact of annotation errors on text-to-SQL agent performance and leaderboard rankings. Through expert analysis, we show that BIRD Mini-Dev and Spider 2.0-Snow have error rates of 52.8% and 62.8%, respectively. We re-evaluate all 16 open-source agents from the BIRD leaderboard on both the original and the corrected BIRD Dev subsets. We show that performance changes range from -7% to 31% (in relative terms) and rank changes range from $-9$ to $+9$ positions. We further assess whether these impacts generalize to the full BIRD Dev set. We find that the rankings of agents on the uncorrected subset correlate strongly with those on the full Dev set (Spearman's $r_s$=0.85, $p$=3.26e-5), whereas they correlate weakly with those on the corrected subset (Spearman's $r_s$=0.32, $p$=0.23). These findings show that annotation errors can significantly distort reported performance and rankings, potentially misguiding research directions or deployment choices. Our code and data are available at this https URL. 

---
# AI as Entertainment 

**Authors**: Cody Kommers, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2601.08768)  

**Abstract**: Generative AI systems are predominantly designed, evaluated, and marketed as intelligent systems which will benefit society by augmenting or automating human cognitive labor, promising to increase personal, corporate, and macroeconomic productivity. But this mainstream narrative about what AI is and what it can do is in tension with another emerging use case: entertainment. We argue that the field of AI is unprepared to measure or respond to how the proliferation of entertaining AI-generated content will impact society. Emerging data suggest AI is already widely adopted for entertainment purposes -- especially by young people -- and represents a large potential source of revenue. We contend that entertainment will become a primary business model for major AI corporations seeking returns on massive infrastructure investments; this will exert a powerful influence on the technology these companies produce in the coming years. Examining current evaluation practices, we identify a critical asymmetry: while AI assessments rigorously measure both benefits and harms of intelligence, they focus almost exclusively on cultural harms. We lack frameworks for articulating how cultural outputs might be actively beneficial. Drawing on insights from the humanities, we propose "thick entertainment" as a framework for evaluating AI-generated cultural content -- one that considers entertainment's role in meaning-making, identity formation, and social connection rather than simply minimizing harm. While AI is often touted for its potential to revolutionize productivity, in the long run we may find that AI turns out to be as much about "intelligence" as social media is about social connection. 

---
# Learning from Demonstrations via Capability-Aware Goal Sampling 

**Authors**: Yuanlin Duan, Yuning Wang, Wenjie Qiu, He Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08731)  

**Abstract**: Despite its promise, imitation learning often fails in long-horizon environments where perfect replication of demonstrations is unrealistic and small errors can accumulate catastrophically. We introduce Cago (Capability-Aware Goal Sampling), a novel learning-from-demonstrations method that mitigates the brittle dependence on expert trajectories for direct imitation. Unlike prior methods that rely on demonstrations only for policy initialization or reward shaping, Cago dynamically tracks the agent's competence along expert trajectories and uses this signal to select intermediate steps--goals that are just beyond the agent's current reach--to guide learning. This results in an adaptive curriculum that enables steady progress toward solving the full task. Empirical results demonstrate that Cago significantly improves sample efficiency and final performance across a range of sparse-reward, goal-conditioned tasks, consistently outperforming existing learning from-demonstrations baselines. 

---
# Evaluating the Ability of Explanations to Disambiguate Models in a Rashomon Set 

**Authors**: Kaivalya Rawal, Eoin Delaney, Zihao Fu, Sandra Wachter, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2601.08703)  

**Abstract**: Explainable artificial intelligence (XAI) is concerned with producing explanations indicating the inner workings of models. For a Rashomon set of similarly performing models, explanations provide a way of disambiguating the behavior of individual models, helping select models for deployment. However explanations themselves can vary depending on the explainer used, and need to be evaluated. In the paper "Evaluating Model Explanations without Ground Truth", we proposed three principles of explanation evaluation and a new method "AXE" to evaluate the quality of feature-importance explanations. We go on to illustrate how evaluation metrics that rely on comparing model explanations against ideal ground truth explanations obscure behavioral differences within a Rashomon set. Explanation evaluation aligned with our proposed principles would highlight these differences instead, helping select models from the Rashomon set. The selection of alternate models from the Rashomon set can maintain identical predictions but mislead explainers into generating false explanations, and mislead evaluation methods into considering the false explanations to be of high quality. AXE, our proposed explanation evaluation method, can detect this adversarial fairwashing of explanations with a 100% success rate. Unlike prior explanation evaluation strategies such as those based on model sensitivity or ground truth comparison, AXE can determine when protected attributes are used to make predictions. 

---
# All Required, In Order: Phase-Level Evaluation for AI-Human Dialogue in Healthcare and Beyond 

**Authors**: Shubham Kulkarni, Alexander Lyzhov, Shiva Chaitanya, Preetam Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08690)  

**Abstract**: Conversational AI is starting to support real clinical work, but most evaluation methods miss how compliance depends on the full course of a conversation. We introduce Obligatory-Information Phase Structured Compliance Evaluation (OIP-SCE), an evaluation method that checks whether every required clinical obligation is met, in the right order, with clear evidence for clinicians to review. This makes complex rules practical and auditable, helping close the gap between technical progress and what healthcare actually needs. We demonstrate the method in two case studies (respiratory history, benefits verification) and show how phase-level evidence turns policy into shared, actionable steps. By giving clinicians control over what to check and engineers a clear specification to implement, OIP-SCE provides a single, auditable evaluation surface that aligns AI capability with clinical workflow and supports routine, safe use. 

---
# MEMEWEAVER: Inter-Meme Graph Reasoning for Sexism and Misogyny Detection 

**Authors**: Paolo Italiani, David Gimeno-Gomez, Luca Ragazzi, Gianluca Moro, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2601.08684)  

**Abstract**: Women are twice as likely as men to face online harassment due to their gender. Despite recent advances in multimodal content moderation, most approaches still overlook the social dynamics behind this phenomenon, where perpetrators reinforce prejudices and group identity within like-minded communities. Graph-based methods offer a promising way to capture such interactions, yet existing solutions remain limited by heuristic graph construction, shallow modality fusion, and instance-level reasoning. In this work, we present MemeWeaver, an end-to-end trainable multimodal framework for detecting sexism and misogyny through a novel inter-meme graph reasoning mechanism. We systematically evaluate multiple visual--textual fusion strategies and show that our approach consistently outperforms state-of-the-art baselines on the MAMI and EXIST benchmarks, while achieving faster training convergence. Further analyses reveal that the learned graph structure captures semantically meaningful patterns, offering valuable insights into the relational nature of online hate. 

---
# PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning 

**Authors**: Xiaoyou Liu, Xinyi Mou, Shengbin Yue, Liang Wang, Yuqing Wang, Qiexiang Wang, Tianrui Qin, Wangchunshu Zhou, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.08679)  

**Abstract**: As users increasingly expect LLMs to align with their preferences, personalized information becomes valuable. However, personalized information can be a double-edged sword: it can improve interaction but may compromise objectivity and factual correctness, especially when it is misaligned with the question. To alleviate this problem, we propose PersonaDual, a framework that supports both general-purpose objective reasoning and personalized reasoning in a single model, and adaptively switches modes based on context. PersonaDual is first trained with SFT to learn two reasoning patterns, and then further optimized via reinforcement learning with our proposed DualGRPO to improve mode selection. Experiments on objective and personalized benchmarks show that PersonaDual preserves the benefits of personalization while reducing interference, achieving near interference-free performance and better leveraging helpful personalized signals to improve objective problem-solving. 

---
# Advancing ESG Intelligence: An Expert-level Agent and Comprehensive Benchmark for Sustainable Finance 

**Authors**: Yilei Zhao, Wentao Zhang, Xiao Lei, Yandan Zheng, Mengpu Liu, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.08676)  

**Abstract**: Environmental, social, and governance (ESG) criteria are essential for evaluating corporate sustainability and ethical performance. However, professional ESG analysis is hindered by data fragmentation across unstructured sources, and existing large language models (LLMs) often struggle with the complex, multi-step workflows required for rigorous auditing. To address these limitations, we introduce ESGAgent, a hierarchical multi-agent system empowered by a specialized toolset, including retrieval augmentation, web search and domain-specific functions, to generate in-depth ESG analysis. Complementing this agentic system, we present a comprehensive three-level benchmark derived from 310 corporate sustainability reports, designed to evaluate capabilities ranging from atomic common-sense questions to the generation of integrated, in-depth analysis. Empirical evaluations demonstrate that ESGAgent outperforms state-of-the-art closed-source LLMs with an average accuracy of 84.15% on atomic question-answering tasks, and excels in professional report generation by integrating rich charts and verifiable references. These findings confirm the diagnostic value of our benchmark, establishing it as a vital testbed for assessing general and advanced agentic capabilities in high-stakes vertical domains. 

---
# Why AI Alignment Failure Is Structural: Learned Human Interaction Structures and AGI as an Endogenous Evolutionary Shock 

**Authors**: Didier Sornette, Sandro Claudio Lera, Ke Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08673)  

**Abstract**: Recent reports of large language models (LLMs) exhibiting behaviors such as deception, threats, or blackmail are often interpreted as evidence of alignment failure or emergent malign agency. We argue that this interpretation rests on a conceptual error. LLMs do not reason morally; they statistically internalize the record of human social interaction, including laws, contracts, negotiations, conflicts, and coercive arrangements. Behaviors commonly labeled as unethical or anomalous are therefore better understood as structural generalizations of interaction regimes that arise under extreme asymmetries of power, information, or constraint. Drawing on relational models theory, we show that practices such as blackmail are not categorical deviations from normal social behavior, but limiting cases within the same continuum that includes market pricing, authority relations, and ultimatum bargaining. The surprise elicited by such outputs reflects an anthropomorphic expectation that intelligence should reproduce only socially sanctioned behavior, rather than the full statistical landscape of behaviors humans themselves enact. Because human morality is plural, context-dependent, and historically contingent, the notion of a universally moral artificial intelligence is ill-defined. We therefore reframe concerns about artificial general intelligence (AGI). The primary risk is not adversarial intent, but AGI's role as an endogenous amplifier of human intelligence, power, and contradiction. By eliminating longstanding cognitive and institutional frictions, AGI compresses timescales and removes the historical margin of error that has allowed inconsistent values and governance regimes to persist without collapse. Alignment failure is thus structural, not accidental, and requires governance approaches that address amplification, complexity, and regime stability rather than model-level intent alone. 

---
# Parallel Context-of-Experts Decoding for Retrieval Augmented Generation 

**Authors**: Giulio Corallo, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2601.08670)  

**Abstract**: Retrieval Augmented Generation faces a trade-off: concatenating documents in a long prompt enables multi-document reasoning but creates prefill bottlenecks, while encoding document KV caches separately offers speed but breaks cross-document interaction. We propose Parallel Context-of-Experts Decoding (Pced), a training-free framework that shifts evidence aggregation from the attention mechanism to the decoding. Pced treats retrieved documents as isolated "experts", synchronizing their predictions via a novel retrieval-aware contrastive decoding rule that weighs expert logits against the model prior. This approach recovers cross-document reasoning capabilities without constructing a shared attention across documents. 

---
# From Classical to Quantum Reinforcement Learning and Its Applications in Quantum Control: A Beginner's Tutorial 

**Authors**: Abhijit Sen, Sonali Panda, Mahima Arya, Subhajit Patra, Zizhan Zheng, Denys I. Bondar  

**Link**: [PDF](https://arxiv.org/pdf/2601.08662)  

**Abstract**: This tutorial is designed to make reinforcement learning (RL) more accessible to undergraduate students by offering clear, example-driven explanations. It focuses on bridging the gap between RL theory and practical coding applications, addressing common challenges that students face when transitioning from conceptual understanding to implementation. Through hands-on examples and approachable explanations, the tutorial aims to equip students with the foundational skills needed to confidently apply RL techniques in real-world scenarios. 

---
# Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding 

**Authors**: Zenghua Liao, Jinzhi Liao, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08653)  

**Abstract**: Large Language Models are rapidly emerging as web-native interfaces to social platforms. On the social web, users frequently have ambiguous and dynamic goals, making complex intent understanding-rather than single-turn execution-the cornerstone of effective human-LLM collaboration. Existing approaches attempt to clarify user intents through sequential or parallel questioning, yet they fall short of addressing the core challenge: modeling the logical dependencies among clarification questions. Inspired by the Cognitive Load Theory, we propose Prism, a novel framework for complex intent understanding that enables logically coherent and efficient intent clarification. Prism comprises four tailored modules: a complex intent decomposition module, which decomposes user intents into smaller, well-structured elements and identifies logical dependencies among them; a logical clarification generation module, which organizes clarification questions based on these dependencies to ensure coherent, low-friction interactions; an intent-aware reward module, which evaluates the quality of clarification trajectories via an intent-aware reward function and leverages Monte Carlo Sample to simulate user-LLM interactions for large-scale,high-quality training data generation; and a self-evolved intent tuning module, which iteratively refines the LLM's logical clarification capability through data-driven feedback and optimization. Prism consistently outperforms existing approaches across clarification interactions, intent execution, and cognitive load benchmarks. It achieves stateof-the-art logical consistency, reduces logical conflicts to 11.5%, increases user satisfaction by 14.4%, and decreases task completion time by 34.8%. All data and code are released. 

---
# Resisting Manipulative Bots in Memecoin Copy Trading: A Multi-Agent Approach with Chain-of-Thought Reasoning 

**Authors**: Yichen Luo, Yebo Feng, Jiahua Xu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08641)  

**Abstract**: The launch of \$Trump coin ignited a wave in meme coin investment. Copy trading, as a strategy-agnostic approach that eliminates the need for deep trading knowledge, quickly gains widespread popularity in the meme coin market. However, copy trading is not a guarantee of profitability due to the prevalence of manipulative bots, the uncertainty of the followed wallets' future performance, and the lag in trade execution. Recently, large language models (LLMs) have shown promise in financial applications by effectively understanding multi-modal data and producing explainable decisions. However, a single LLM struggles with complex, multi-faceted tasks such as asset allocation. These challenges are even more pronounced in cryptocurrency markets, where LLMs often lack sufficient domain-specific knowledge in their training data.
To address these challenges, we propose an explainable multi-agent system for meme coin copy trading. Inspired by the structure of an asset management team, our system decomposes the complex task into subtasks and coordinates specialized agents to solve them collaboratively. Employing few-shot chain-of-though (CoT) prompting, each agent acquires professional meme coin trading knowledge, interprets multi-modal data, and generates explainable decisions. Using a dataset of 1,000 meme coin projects' transaction data, our empirical evaluation shows that the proposed multi-agent system outperforms both traditional machine learning models and single LLMs, achieving 73% and 70% precision in identifying high-quality meme coin projects and key opinion leader (KOL) wallets, respectively. The selected KOLs collectively generated a total profit of \$500,000 across these projects. 

---
# ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios 

**Authors**: António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, Gautier Viaud  

**Link**: [PDF](https://arxiv.org/pdf/2601.08620)  

**Abstract**: Retrieval-Augmented Generation (RAG) pipelines must address challenges beyond simple single-document retrieval, such as interpreting visual elements (tables, charts, images), synthesizing information across documents, and providing accurate source grounding. Existing benchmarks fail to capture this complexity, often focusing on textual data, single-document comprehension, or evaluating retrieval and generation in isolation. We introduce ViDoRe v3, a comprehensive multimodal RAG benchmark featuring multi-type queries over visually rich document corpora. It covers 10 datasets across diverse professional domains, comprising ~26,000 document pages paired with 3,099 human-verified queries, each available in 6 languages. Through 12,000 hours of human annotation effort, we provide high-quality annotations for retrieval relevance, bounding box localization, and verified reference answers. Our evaluation of state-of-the-art RAG pipelines reveals that visual retrievers outperform textual ones, late-interaction models and textual reranking substantially improve performance, and hybrid or purely visual contexts enhance answer generation quality. However, current models still struggle with non-textual elements, open-ended queries, and fine-grained visual grounding. To encourage progress in addressing these challenges, the benchmark is released under a commercially permissive license at this https URL. 

---
# WaterCopilot: An AI-Driven Virtual Assistant for Water Management 

**Authors**: Keerththanan Vickneswaran, Mariangel Garcia Andarcia, Hugo Retief, Chris Dickens, Paulo Silva  

**Link**: [PDF](https://arxiv.org/pdf/2601.08559)  

**Abstract**: Sustainable water resource management in transboundary river basins is challenged by fragmented data, limited real-time access, and the complexity of integrating diverse information sources. This paper presents WaterCopilot-an AI-driven virtual assistant developed through collaboration between the International Water Management Institute (IWMI) and Microsoft Research for the Limpopo River Basin (LRB) to bridge these gaps through a unified, interactive platform. Built on Retrieval-Augmented Generation (RAG) and tool-calling architectures, WaterCopilot integrates static policy documents and real-time hydrological data via two custom plugins: the iwmi-doc-plugin, which enables semantic search over indexed documents using Azure AI Search, and the iwmi-api-plugin, which queries live databases to deliver dynamic insights such as environmental-flow alerts, rainfall trends, reservoir levels, water accounting, and irrigation data. The system features guided multilingual interactions (English, Portuguese, French), transparent source referencing, automated calculations, and visualization capabilities. Evaluated using the RAGAS framework, WaterCopilot achieves an overall score of 0.8043, with high answer relevancy (0.8571) and context precision (0.8009). Key innovations include automated threshold-based alerts, integration with the LRB Digital Twin, and a scalable deployment pipeline hosted on AWS. While limitations in processing non-English technical documents and API latency remain, WaterCopilot establishes a replicable AI-augmented framework for enhancing water governance in data-scarce, transboundary contexts. The study demonstrates the potential of this AI assistant to support informed, timely decision-making and strengthen water security in complex river basins. 

---
# Learner-Tailored Program Repair: A Solution Generator with Iterative Edit-Driven Retrieval Enhancement 

**Authors**: Zhenlong Dai, Zhuoluo Zhao, Hengning Wang, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08545)  

**Abstract**: With the development of large language models (LLMs) in the field of programming, intelligent programming coaching systems have gained widespread attention. However, most research focuses on repairing the buggy code of programming learners without providing the underlying causes of the bugs. To address this gap, we introduce a novel task, namely \textbf{LPR} (\textbf{L}earner-Tailored \textbf{P}rogram \textbf{R}epair). We then propose a novel and effective framework, \textbf{\textsc{\MethodName{}}} (\textbf{L}earner-Tailored \textbf{S}olution \textbf{G}enerator), to enhance program repair while offering the bug descriptions for the buggy code. In the first stage, we utilize a repair solution retrieval framework to construct a solution retrieval database and then employ an edit-driven code retrieval approach to retrieve valuable solutions, guiding LLMs in identifying and fixing the bugs in buggy code. In the second stage, we propose a solution-guided program repair method, which fixes the code and provides explanations under the guidance of retrieval solutions. Moreover, we propose an Iterative Retrieval Enhancement method that utilizes evaluation results of the generated code to iteratively optimize the retrieval direction and explore more suitable repair strategies, improving performance in practical programming coaching scenarios. The experimental results show that our approach outperforms a set of baselines by a large margin, validating the effectiveness of our framework for the newly proposed LPR task. 

---
# Sketch-Based Facade Renovation With Generative AI: A Streamlined Framework for Bypassing As-Built Modelling in Industrial Adaptive Reuse 

**Authors**: Warissara Booranamaitree, Xusheng Du, Yushu Cai, Zhengyang Wang, Ye Zhang, Haoran Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.08531)  

**Abstract**: Facade renovation offers a more sustainable alternative to full demolition, yet producing design proposals that preserve existing structures while expressing new intent remains challenging. Current workflows typically require detailed as-built modelling before design, which is time-consuming, labour-intensive, and often involves repeated revisions. To solve this issue, we propose a three-stage framework combining generative artificial intelligence (AI) and vision-language models (VLM) that directly processes rough structural sketch and textual descriptions to produce consistent renovation proposals. First, the input sketch is used by a fine-tuned VLM model to predict bounding boxes specifying where modifications are needed and which components should be added. Next, a stable diffusion model generates detailed sketches of new elements, which are merged with the original outline through a generative inpainting pipeline. Finally, ControlNet is employed to refine the result into a photorealistic image. Experiments on datasets and real industrial buildings indicate that the proposed framework can generate renovation proposals that preserve the original structure while improving facade detail quality. This approach effectively bypasses the need for detailed as-built modelling, enabling architects to rapidly explore design alternatives, iterate on early-stage concepts, and communicate renovation intentions with greater clarity. 

---
# What If TSF: A Benchmark for Reframing Forecasting as Scenario-Guided Multimodal Forecasting 

**Authors**: Jinkwan Jang, Hyunbin Jin, Hyungjin Park, Kyubyung Chae, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.08509)  

**Abstract**: Time series forecasting is critical to real-world decision making, yet most existing approaches remain unimodal and rely on extrapolating historical patterns. While recent progress in large language models (LLMs) highlights the potential for multimodal forecasting, existing benchmarks largely provide retrospective or misaligned raw context, making it unclear whether such models meaningfully leverage textual inputs. In practice, human experts incorporate what-if scenarios with historical evidence, often producing distinct forecasts from the same observations under different scenarios. Inspired by this, we introduce What If TSF (WIT), a multimodal forecasting benchmark designed to evaluate whether models can condition their forecasts on contextual text, especially future scenarios. By providing expert-crafted plausible or counterfactual scenarios, WIT offers a rigorous testbed for scenario-guided multimodal forecasting. The benchmark is available at this https URL. 

---
# SUMMPILOT: Bridging Efficiency and Customization for Interactive Summarization System 

**Authors**: JungMin Yun, Juhwan Choi, Kyohoon Jin, Soojin Jang, Jinhee Jang, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.08475)  

**Abstract**: This paper incorporates the efficiency of automatic summarization and addresses the challenge of generating personalized summaries tailored to individual users' interests and requirements. To tackle this challenge, we introduce SummPilot, an interaction-based customizable summarization system. SummPilot leverages a large language model to facilitate both automatic and interactive summarization. Users can engage with the system to understand document content and personalize summaries through interactive components such as semantic graphs, entity clustering, and explainable evaluation. Our demo and user studies demonstrate SummPilot's adaptability and usefulness for customizable summarization. 

---
# M3-BENCH: Process-Aware Evaluation of LLM Agents Social Behaviors in Mixed-Motive Games 

**Authors**: Sixiong Xie, Zhuofan Shi, Haiyang Shen, Gang Huang, Yun Ma, Xiang Jing  

**Link**: [PDF](https://arxiv.org/pdf/2601.08462)  

**Abstract**: As the capabilities of large language model (LLM) agents continue to advance, their advanced social behaviors, such as cooperation, deception, and collusion, call for systematic evaluation. However, existing benchmarks often emphasize a single capability dimension or rely solely on behavioral outcomes, overlooking rich process information from agents' decision reasoning and communicative interactions. To address this gap, we propose M3-Bench, a multi-stage benchmark for mixed-motive games, together with a process-aware evaluation framework that conducts synergistic analysis across three modules: BTA (Behavioral Trajectory Analysis), RPA (Reasoning Process Analysis), and CCA (Communication Content Analysis). Furthermore, we integrate the Big Five personality model and Social Exchange Theory to aggregate multi-dimensional evidence into interpretable social behavior portraits, thereby characterizing agents' personality traits and capability profiles beyond simple task scores or outcome-based metrics. Experimental results show that M3-Bench can reliably distinguish diverse social behavior competencies across models, and it reveals that some models achieve seemingly reasonable behavioral outcomes while exhibiting pronounced inconsistencies in their reasoning and communication. 

---
# An Under-Explored Application for Explainable Multimodal Misogyny Detection in code-mixed Hindi-English 

**Authors**: Sargam Yadav, Abhishek Kaushik, Kevin Mc Daid  

**Link**: [PDF](https://arxiv.org/pdf/2601.08457)  

**Abstract**: Digital platforms have an ever-expanding user base, and act as a hub for communication, business, and connectivity. However, this has also allowed for the spread of hate speech and misogyny. Artificial intelligence models have emerged as an effective solution for countering online hate speech but are under explored for low resource and code-mixed languages and suffer from a lack of interpretability. Explainable Artificial Intelligence (XAI) can enhance transparency in the decisions of deep learning models, which is crucial for a sensitive domain such as hate speech detection. In this paper, we present a multi-modal and explainable web application for detecting misogyny in text and memes in code-mixed Hindi and English. The system leverages state-of-the-art transformer-based models that support multilingual and multimodal settings. For text-based misogyny identification, the system utilizes XLM-RoBERTa (XLM-R) and multilingual Bidirectional Encoder Representations from Transformers (mBERT) on a dataset of approximately 4,193 comments. For multimodal misogyny identification from memes, the system utilizes mBERT + EfficientNet, and mBERT + ResNET trained on a dataset of approximately 4,218 memes. It also provides feature importance scores using explainability techniques including Shapley Additive Values (SHAP) and Local Interpretable Model Agnostic Explanations (LIME). The application aims to serve as a tool for both researchers and content moderators, to promote further research in the field, combat gender based digital violence, and ensure a safe digital space. The system has been evaluated using human evaluators who provided their responses on Chatbot Usability Questionnaire (CUQ) and User Experience Questionnaire (UEQ) to determine overall usability. 

---
# Beyond Linearization: Attributed Table Graphs for Table Reasoning 

**Authors**: Yuxiang Wang, Junhao Gan, Shengxiang Gao, Shenghao Ye, Zhengyi Yang, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08444)  

**Abstract**: Table reasoning, a task to answer questions by reasoning over data presented in tables, is an important topic due to the prevalence of knowledge stored in tabular formats. Recent solutions use Large Language Models (LLMs), exploiting the semantic understanding and reasoning capabilities of LLMs. A common paradigm of such solutions linearizes tables to form plain texts that are served as input to LLMs. This paradigm has critical issues. It loses table structures, lacks explicit reasoning paths for result explainability, and is subject to the "lost-in-the-middle" issue. To address these issues, we propose Table Graph Reasoner (TABGR), a training-free model that represents tables as an Attributed Table Graph (ATG). The ATG explicitly preserves row-column-cell structures while enabling graph-based reasoning for explainability. We further propose a Question-Guided Personalized PageRank (QG-PPR) mechanism to rerank tabular data and mitigate the lost-in-the-middle issue. Extensive experiments on two commonly used benchmarks show that TABGR consistently outperforms state-of-the-art models by up to 9.7% in accuracy. Our code will be made publicly available upon publication. 

---
# YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation 

**Authors**: Abdelaziz Bounhar, Rania Hossam Elmohamady Elbadry, Hadi Abdine, Preslav Nakov, Michalis Vazirgiannis, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08441)  

**Abstract**: Steering Large Language Models (LLMs) through activation interventions has emerged as a lightweight alternative to fine-tuning for alignment and personalization. Recent work on Bi-directional Preference Optimization (BiPO) shows that dense steering vectors can be learned directly from preference data in a Direct Preference Optimization (DPO) fashion, enabling control over truthfulness, hallucinations, and safety behaviors. However, dense steering vectors often entangle multiple latent factors due to neuron multi-semanticity, limiting their effectiveness and stability in fine-grained settings such as cultural alignment, where closely related values and behaviors (e.g., among Middle Eastern cultures) must be distinguished. In this paper, we propose Yet another Policy Optimization (YaPO), a \textit{reference-free} method that learns \textit{sparse steering vectors} in the latent space of a Sparse Autoencoder (SAE). By optimizing sparse codes, YaPO produces disentangled, interpretable, and efficient steering directions. Empirically, we show that YaPO converges faster, achieves stronger performance, and exhibits improved training stability compared to dense steering baselines. Beyond cultural alignment, YaPO generalizes to a range of alignment-related behaviors, including hallucination, wealth-seeking, jailbreak, and power-seeking. Importantly, YaPO preserves general knowledge, with no measurable degradation on MMLU. Overall, our results show that YaPO provides a general recipe for efficient, stable, and fine-grained alignment of LLMs, with broad applications to controllability and domain adaptation. The associated code and data are publicly available\footnote{this https URL}. 

---
# RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation 

**Authors**: Sunzhu Li, Jiale Zhao, Miteto Wei, Huimin Ren, Yang Zhou, Jingwen Yang, Shunyu Liu, Kaike Zhang, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08430)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has driven substantial progress in reasoning-intensive domains like mathematics. However, optimizing open-ended generation remains challenging due to the lack of ground truth. While rubric-based evaluation offers a structured proxy for verification, existing methods suffer from scalability bottlenecks and coarse criteria, resulting in a supervision ceiling effect. To address this, we propose an automated Coarse-to-Fine Rubric Generation framework. By synergizing principle-guided synthesis, multi-model aggregation, and difficulty evolution, our approach produces comprehensive and highly discriminative criteria capable of capturing the subtle nuances. Based on this framework, we introduce RubricHub, a large-scale ($\sim$110k) and multi-domain dataset. We validate its utility through a two-stage post-training pipeline comprising Rubric-based Rejection Sampling Fine-Tuning (RuFT) and Reinforcement Learning (RuRL). Experimental results demonstrate that RubricHub unlocks significant performance gains: our post-trained Qwen3-14B achieves state-of-the-art (SOTA) results on HealthBench (69.3), surpassing proprietary frontier models such as GPT-5. The code and data will be released soon. 

---
# Hybrid Distillation with CoT Guidance for Edge-Drone Control Code Generation 

**Authors**: Yizhan Feng, Hichem Snoussi, Yuhang Wang, Jing Teng, Abel Cherouat, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08412)  

**Abstract**: With large language models demonstrating significant potential in code generation tasks, their application to onboard control of resource-constrained Unmanned Aerial Vehicles has emerged as an important research direction. However, a notable contradiction exists between the high resource consumption of large models and the real-time, lightweight requirements of UAV platforms. This paper proposes an integrated approach that combines knowledge distillation, chain-of-thought guidance, and supervised fine-tuning for UAV multi-SDK control tasks, aiming to efficiently transfer complex reasoning and code generation capabilities to smaller models. Firstly, a high-quality dataset covering various mainstream UAV SDKs is constructed, featuring instruction-code-reasoning chains, and incorporates counterfactual negative samples for data augmentation, guiding the model to learn the end-to-end logic from instruction parsing to code generation. Secondly, leveraging DeepSeek-Coder-V2-Lite quantized via QLoRA as the teacher model, and based on a hybrid black-box and white-box distillation strategy, high-quality chain-of-thought soft labels are generated. These are combined with a weighted cross-entropy loss using hard labels to transfer complex reasoning capabilities to the smaller student model. Finally, through prompt tuning engineering optimized for the UAV control scenario, the model performance on core tasks such as SDK type recognition and function call matching is enhanced. Experimental results indicate that the distilled lightweight model maintains high code generation accuracy while achieving significant improvements in deployment and inference efficiency, effectively demonstrating the feasibility and superiority of our approach in achieving precise and lightweight intelligent control for UAVs 

---
# WebTrap Park: An Automated Platform for Systematic Security Evaluation of Web Agents 

**Authors**: Xinyi Wu, Jiagui Chen, Geng Hong, Jiayi Dong, Xudong Pan, Jiarun Dai, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08406)  

**Abstract**: Web Agents are increasingly deployed to perform complex tasks in real web environments, yet their security evaluation remains fragmented and difficult to standardize. We present WebTrap Park, an automated platform for systematic security evaluation of Web Agents through direct observation of their concrete interactions with live web pages. WebTrap Park instantiates three major sources of security risk into 1,226 executable evaluation tasks and enables action based assessment without requiring agent modification. Our results reveal clear security differences across agent frameworks, highlighting the importance of agent architecture beyond the underlying model. WebTrap Park is publicly accessible at this https URL and provides a scalable foundation for reproducible Web Agent security evaluation. 

---
# Owen-Shapley Policy Optimization (OSPO): A Principled RL Algorithm for Generative Search LLMs 

**Authors**: Abhijnan Nath, Alireza Bagheri Garakani, Tianchen Zhou, Fan Yang, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2601.08403)  

**Abstract**: Large language models are increasingly trained via reinforcement learning for personalized recommendation tasks, but standard methods like GRPO rely on sparse, sequence-level rewards that create a credit assignment gap, obscuring which tokens drive success. This gap is especially problematic when models must infer latent user intent from under-specified language without ground truth labels, a reasoning pattern rarely seen during pretraining. We introduce Owen-Shapley Policy Optimization (OSPO), a framework that redistributes sequence-level advantages based on tokens' marginal contributions to outcomes. Unlike value-model-based methods requiring additional computation, OSPO employs potential-based reward shaping via Shapley-Owen attributions to assign segment-level credit while preserving the optimal policy, learning directly from task feedback without parametric value models. By forming coalitions of semantically coherent units (phrases describing product attributes or sentences capturing preferences), OSPO identifies which response parts drive performance. Experiments on Amazon ESCI and H&M Fashion datasets show consistent gains over baselines, with notable test-time robustness to out-of-distribution retrievers unseen during training. 

---
# Creativity in AI as Emergence from Domain-Limited Generative Models 

**Authors**: Corina Chutaux  

**Link**: [PDF](https://arxiv.org/pdf/2601.08388)  

**Abstract**: Creativity in artificial intelligence is most often addressed through evaluative frameworks that aim to measure novelty, diversity, or usefulness in generated outputs. While such approaches have provided valuable insights into the behavior of modern generative models, they largely treat creativity as a property to be assessed rather than as a phenomenon to be explicitly modeled. In parallel, recent advances in large-scale generative systems, particularly multimodal architectures, have demonstrated increasingly sophisticated forms of pattern recombination, raising questions about the nature and limits of machine creativity. This paper proposes a generative perspective on creativity in AI, framing it as an emergent property of domain-limited generative models embedded within bounded informational environments. Rather than introducing new evaluative criteria, we focus on the structural and contextual conditions under which creative behaviors arise. We introduce a conceptual decomposition of creativity into four interacting components-pattern-based generation, induced world models, contextual grounding, and arbitrarity, and examine how these components manifest in multimodal generative systems. By grounding creativity in the interaction between generative dynamics and domain-specific representations, this work aims to provide a technical framework for studying creativity as an emergent phenomenon in AI systems, rather than as a post hoc evaluative label. 

---
# Deconstructing Pre-training: Knowledge Attribution Analysis in MoE and Dense Models 

**Authors**: Bo Wang, Junzhuo Li, Hong Chen, Yuanlin Chu, Yuxuan Fan, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08383)  

**Abstract**: Mixture-of-Experts (MoE) architectures decouple model capacity from per-token computation, enabling scaling beyond the computational limits imposed by dense scaling laws. Yet how MoE architectures shape knowledge acquisition during pre-training, and how this process differs from dense architectures, remains unknown. To address this issue, we introduce Gated-LPI (Log-Probability Increase), a neuron-level attribution metric that decomposes log-probability increase across neurons. We present a time-resolved comparison of knowledge acquisition dynamics in MoE and dense architectures, tracking checkpoints over 1.2M training steps (~ 5.0T tokens) and 600K training steps (~ 2.5T tokens), respectively. Our experiments uncover three patterns: (1) Low-entropy backbone. The top approximately 1% of MoE neurons capture over 45% of positive updates, forming a high-utility core, which is absent in the dense baseline. (2) Early consolidation. The MoE model locks into a stable importance profile within < 100K steps, whereas the dense model remains volatile throughout training. (3) Functional robustness. Masking the ten most important MoE attention heads reduces relational HIT@10 by < 10%, compared with > 50% for the dense model, showing that sparsity fosters distributed -- rather than brittle -- knowledge storage. These patterns collectively demonstrate that sparsity fosters an intrinsically stable and distributed computational backbone from early in training, helping bridge the gap between sparse architectures and training-time interpretability. 

---
# A Qualitative Model to Reason about Object Rotations (QOR) applied to solve the Cube Comparison Test (CCT) 

**Authors**: Zoe Falomir  

**Link**: [PDF](https://arxiv.org/pdf/2601.08382)  

**Abstract**: This paper presents a Qualitative model for Reasoning about Object Rotations (QOR) which is applied to solve the Cube Comparison Test (CCT) by Ekstrom et al. (1976). A conceptual neighborhood graph relating the Rotation movement to the Location change and the Orientation change (CNGRLO) of the features on the cube sides has been built and it produces composition tables to calculate inferences for reasoning about rotations. 

---
# Thematic Working Group 5 -- Artificial Intelligence (AI) literacy for teaching and learning: design and implementation 

**Authors**: Mary Webb, Matt Bower, Ana Amélia Carvalho, Fredrik Mørk Røkenes, Jodie Torrington, Jonathan D. Cohen, Yousra Chtouki, Kathryn Maccallum, Tanya Linden, Deirdre Butler, Juliana Elisa Raffaghelli, Henriikka Vartiainen, Martina Ronci, Peter Tiernan, David M. Smith, Chris Shelton, Joyce Malyn-smith, Pierre Gorissen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08380)  

**Abstract**: TWG 5 focused on developing and implementing effective strategies for enhancing AI literacy and agency of teachers, equipping them with the knowledge and skills necessary to integrate AI into their teaching practices. Explorations covered curriculum design, professional development programs, practical classroom applications, and policy guidelines aiming to empower educators to confidently utilize AI tools and foster a deeper understanding of AI concepts among students. 

---
# Semantic Laundering in AI Agent Architectures: Why Tool Boundaries Do Not Confer Epistemic Warrant 

**Authors**: Oleg Romanchuk, Roman Bondar  

**Link**: [PDF](https://arxiv.org/pdf/2601.08333)  

**Abstract**: LLM-based agent architectures systematically conflate information transport mechanisms with epistemic justification mechanisms. We formalize this class of architectural failures as semantic laundering: a pattern where propositions with absent or weak warrant are accepted by the system as admissible by crossing architecturally trusted interfaces. We show that semantic laundering constitutes an architectural realization of the Gettier problem: propositions acquire high epistemic status without a connection between their justification and what makes them true. Unlike classical Gettier cases, this effect is not accidental; it is architecturally determined and systematically reproducible. The central result is the Theorem of Inevitable Self-Licensing: under standard architectural assumptions, circular epistemic justification cannot be eliminated. We introduce the Warrant Erosion Principle as the fundamental explanation for this effect and show that scaling, model improvement, and LLM-as-judge schemes are structurally incapable of eliminating a problem that exists at the type level. 

---
# AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation 

**Authors**: Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.08323)  

**Abstract**: Equipping agents with memory is essential for solving real-world long-horizon problems. However, most existing agent memory mechanisms rely on static and hand-crafted workflows. This limits the performance and generalization ability of these memory designs, which highlights the need for a more flexible, learning-based memory framework. In this paper, we propose AtomMem, which reframes memory management as a dynamic decision-making problem. We deconstruct high-level memory processes into fundamental atomic CRUD (Create, Read, Update, Delete) operations, transforming the memory workflow into a learnable decision process. By combining supervised fine-tuning with reinforcement learning, AtomMem learns an autonomous, task-aligned policy to orchestrate memory behaviors tailored to specific task demands. Experimental results across 3 long-context benchmarks demonstrate that the trained AtomMem-8B consistently outperforms prior static-workflow memory methods. Further analysis of training dynamics shows that our learning-based formulation enables the agent to discover structured, task-aligned memory management strategies, highlighting a key advantage over predefined routines. 

---
# OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System 

**Authors**: Yuyang Wu, Hanzhong Cao, Jianhao Chen, Yufei Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08288)  

**Abstract**: Chinese stand-up comedy generation goes beyond plain text generation, requiring culturally grounded humor, precise timing, stage-performance cues, and implicit multi-step reasoning. Moreover, commonly used Chinese humor datasets are often better suited for humor understanding and evaluation than for long-form stand-up generation, making direct supervision misaligned with the target task. To address these challenges, we present OpenMic, an end-to-end multi-agent system built on AutoGen that transforms a user-provided life topic into a 3-5 minute Chinese stand-up performance and further produces a narrated comedy video. OpenMic orchestrates multiple specialized agents in a multi-round iterative loop-planning to jointly optimize humor, timing, and performability. To mitigate the dataset-task mismatch, we augment generation with retrieval-augmented generation (RAG) for material grounding and idea expansion, and we fine-tune a dedicated JokeWriter to better internalize stand-up-specific setup-punchline structures and long-range callbacks. 

---
# Greedy Is Enough: Sparse Action Discovery in Agentic LLMs 

**Authors**: Angshul Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2601.08280)  

**Abstract**: Modern agentic systems operate in environments with extremely large action spaces, such as tool-augmented language models with thousands of available APIs or retrieval operations. Despite this scale, empirical evidence suggests that only a small subset of actions meaningfully influences performance in a given deployment. Motivated by this observation, we study a contextual linear reward model in which action relevance is governed by a structured sparsity assumption: only a small number of actions have nonzero effects across latent states.
We formulate action discovery as a block-sparse recovery problem and analyze a greedy algorithm inspired by Orthogonal Matching Pursuit. Under standard assumptions on incoherence, signal strength, and action coverage, we prove that the greedy procedure exactly recovers the relevant action set with high probability, using a number of samples that scales polynomially in the sparsity level and latent dimension, and only logarithmically in the total number of actions. We further provide estimation error guarantees for refitted parameters and show that the resulting decision rule is near-optimal for new latent states.
Complementing these results, we establish information-theoretic lower bounds demonstrating that sparsity and sufficient coverage are necessary for tractability. Together, our results identify sparse action discovery as a fundamental principle underlying large-action decision-making and provide a theoretical foundation for action pruning in agentic systems. 

---
# ToolACE-MCP: Generalizing History-Aware Routing from MCP Tools to the Agent Web 

**Authors**: Zhiyuan Yao, Zishan Xu, Yifu Guo, Zhiguang Han, Cheng Yang, Shuo Zhang, Weinan Zhang, Xingshan Zeng, Weiwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08276)  

**Abstract**: With the rise of the Agent Web and Model Context Protocol (MCP), the agent ecosystem is evolving into an open collaborative network, exponentially increasing accessible tools. However, current architectures face severe scalability and generality bottlenecks. To address this, we propose ToolACE-MCP, a pipeline for training history-aware routers to empower precise navigation in large-scale ecosystems. By leveraging a dependency-rich candidate Graph to synthesize multi-turn trajectories, we effectively train routers with dynamic context understanding to create the plug-and-play Light Routing Agent. Experiments on the real-world benchmarks MCP-Universe and MCP-Mark demonstrate superior performance. Notably, ToolACE-MCP exhibits critical properties for the future Agent Web: it not only generalizes to multi-agent collaboration with minimal adaptation but also maintains exceptional robustness against noise and scales effectively to massive candidate spaces. These findings provide a strong empirical foundation for universal orchestration in open-ended ecosystems. 

---
# Sparsity Is Necessary: Polynomial-Time Stability for Agentic LLMs in Large Action Spaces 

**Authors**: Angshul Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2601.08271)  

**Abstract**: Tool-augmented LLM systems expose a control regime that learning theory has largely ignored: sequential decision-making with a massive discrete action universe (tools, APIs, documents) in which only a small, unknown subset is relevant for any fixed task distribution. We formalize this setting as Sparse Agentic Control (SAC), where policies admit block-sparse representations over M >> 1 actions and rewards depend on sparse main effects and (optionally) sparse synergies. We study ell_{1,2}-regularized policy learning through a convex surrogate and establish sharp, compressed-sensing-style results: (i) estimation and value suboptimality scale as k (log M / T)^{1/2} under a Policy-RSC condition; (ii) exact tool-support recovery holds via primal-dual witness arguments when T > k log M under incoherence and beta-min; and (iii) any dense policy class requires Omega(M) samples, explaining the instability of prompt-only controllers. We further show that under partial observability, LLMs matter only through a belief/representation error epsilon_b, yielding an additive O(epsilon_b) degradation while preserving logarithmic dependence on M. Extensions cover tuning-free, online, robust, group-sparse, and interaction-aware SAC. 

---
# VGG Induced Deep Hand Sign Language Detection 

**Authors**: Subham Sharma, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08262)  

**Abstract**: Hand gesture recognition is an important aspect of human-computer interaction. It forms the basis of sign language for the visually impaired people. This work proposes a novel hand gesture recognizing system for the differently-abled persons. The model uses a convolutional neural network, known as VGG-16 net, for building a trained model on a widely used image dataset by employing Python and Keras libraries. Furthermore, the result is validated by the NUS dataset, consisting of 10 classes of hand gestures, fed to the model as the validation set. Afterwards, a testing dataset of 10 classes is built by employing Google's open source Application Programming Interface (API) that captures different gestures of human hand and the efficacy is then measured by carrying out experiments. The experimental results show that by combining a transfer learning mechanism together with the image data augmentation, the VGG-16 net produced around 98% accuracy. 

---
# T3: Benchmarking Sycophancy and Skepticism in Causal Judgment 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08258)  

**Abstract**: We introduce T3 (Testing Trustworthy Thinking), a diagnostic benchmark designed to rigorously evaluate LLM causal judgment across Pearl's Ladder of Causality. Comprising 454 expert-curated vignettes, T3 prioritizes high-resolution failure analysis, decomposing performance into Utility (sensitivity), Safety (specificity), and Wise Refusal on underdetermined cases. By applying T3 to frontier models, we diagnose two distinct pathologies: a "Skepticism Trap" at L1 (where safety-tuned models like Claude Haiku reject 60% of valid links) and a non-monotonic Scaling Paradox at L3. In the latter, the larger GPT-5.2 underperforms GPT-4-Turbo by 55 points on ambiguous counterfactuals, driven by a collapse into paralysis (excessive hedging) rather than hallucination. Finally, we use the benchmark to validate a process-verified protocol (RCA), showing that T3 successfully captures the restoration of decisive causal judgment under structured verification. 

---
# Large Artificial Intelligence Model Guided Deep Reinforcement Learning for Resource Allocation in Non Terrestrial Networks 

**Authors**: Abdikarim Mohamed Ibrahim, Rosdiadee Nordin  

**Link**: [PDF](https://arxiv.org/pdf/2601.08254)  

**Abstract**: Large AI Model (LAM) have been proposed to applications of Non-Terrestrial Networks (NTN), that offer better performance with its great generalization and reduced task specific trainings. In this paper, we propose a Deep Reinforcement Learning (DRL) agent that is guided by a Large Language Model (LLM). The LLM operates as a high level coordinator that generates textual guidance that shape the reward of the DRL agent during training. The results show that the LAM-DRL outperforms the traditional DRL by 40% in nominal weather scenarios and 64% in extreme weather scenarios compared to heuristics in terms of throughput, fairness, and outage probability. 

---
# The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination 

**Authors**: Haoran Su, Yandong Sun, Congjia Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08237)  

**Abstract**: Reward engineering, the manual specification of reward functions to induce desired agent behavior, remains a fundamental challenge in multi-agent reinforcement learning. This difficulty is amplified by credit assignment ambiguity, environmental non-stationarity, and the combinatorial growth of interaction complexity. We argue that recent advances in large language models (LLMs) point toward a shift from hand-crafted numerical rewards to language-based objective specifications. Prior work has shown that LLMs can synthesize reward functions directly from natural language descriptions (e.g., EUREKA) and adapt reward formulations online with minimal human intervention (e.g., CARD). In parallel, the emerging paradigm of Reinforcement Learning from Verifiable Rewards (RLVR) provides empirical evidence that language-mediated supervision can serve as a viable alternative to traditional reward engineering. We conceptualize this transition along three dimensions: semantic reward specification, dynamic reward adaptation, and improved alignment with human intent, while noting open challenges related to computational overhead, robustness to hallucination, and scalability to large multi-agent systems. We conclude by outlining a research direction in which coordination arises from shared semantic representations rather than explicitly engineered numerical signals. 

---
# MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents 

**Authors**: Shouju Wang, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08235)  

**Abstract**: As language-model agents evolve from passive chatbots into proactive assistants that handle personal data, evaluating their adherence to social norms becomes increasingly critical, often through the lens of Contextual Integrity (CI). However, existing CI benchmarks are largely text-centric and primarily emphasize negative refusal scenarios, overlooking multimodal privacy risks and the fundamental trade-off between privacy and utility. In this paper, we introduce MPCI-Bench, the first Multimodal Pairwise Contextual Integrity benchmark for evaluating privacy behavior in agentic settings. MPCI-Bench consists of paired positive and negative instances derived from the same visual source and instantiated across three tiers: normative Seed judgments, context-rich Story reasoning, and executable agent action Traces. Data quality is ensured through a Tri-Principle Iterative Refinement pipeline. Evaluations of state-of-the-art multimodal models reveal systematic failures to balance privacy and utility and a pronounced modality leakage gap, where sensitive visual information is leaked more frequently than textual information. We will open-source MPCI-Bench to facilitate future research on agentic CI. 

---
# An Axiomatic Approach to General Intelligence: SANC(E3) -- Self-organizing Active Network of Concepts with Energy E3 

**Authors**: Daesuk Kwon, Won-gi Paeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08224)  

**Abstract**: General intelligence must reorganize experience into internal structures that enable prediction and action under finite resources. Existing systems implicitly presuppose fixed primitive units -- tokens, subwords, pixels, or predefined sensor channels -- thereby bypassing the question of how representational units themselves emerge and stabilize. This paper proposes SANC(E3), an axiomatic framework in which representational units are not given a priori but instead arise as stable outcomes of competitive selection, reconstruction, and compression under finite activation capacity, governed by the explicit minimization of an energy functional E3. SANC(E3) draws a principled distinction between system tokens -- structural anchors such as {here, now, I} and sensory sources -- and tokens that emerge through self-organization during co-occurring events. Five core axioms formalize finite capacity, association from co-occurrence, similarity-based competition, confidence-based stabilization, and the reconstruction-compression-update trade-off. A key feature is a pseudo-memory-mapped I/O mechanism, through which internally replayed Gestalts are processed via the same axiomatic pathway as external sensory input. As a result, perception, imagination, prediction, planning, and action are unified within a single representational and energetic process. From the axioms, twelve propositions are derived, showing that category formation, hierarchical organization, unsupervised learning, and high-level cognitive activities can all be understood as instances of Gestalt completion under E3 minimization. 

---
# Adapting Rules of Official International Mahjong for Online Players 

**Authors**: Chucai Wang, Lingfeng Li, Yunlong Lu, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08211)  

**Abstract**: As one of the worldwide spread traditional game, Official International Mahjong can be played and promoted online through remote devices instead of requiring face-to-face interaction. However, online players have fragmented playtime and unfixed combination of opponents in contrary to offline players who have fixed opponents for multiple rounds of play. Therefore, the rules designed for offline players need to be modified to ensure the fairness of online single-round play. Specifically, We employ a world champion AI to engage in self-play competitions and conduct statistical data analysis. Our study reveals the first-mover advantage and issues in the subgoal scoring settings. Based on our findings, we propose rule adaptations to make the game more suitable for the online environment, such as introducing compensatory points for the first-mover advantage and refining the scores of subgoals for different tile patterns. Compared with the traditional method of rotating positions over multiple rounds to balance first-mover advantage, our compensatory points mechanism in each round is more convenient for online players. Furthermore, we implement the revised Mahjong game online, which is open for online players. This work is an initial attempt to use data from AI systems to evaluate Official Internatinoal Mahjong's game balance and develop a revised version of the traditional game better adapted for online players. 

---
# Improving LLM Reasoning with Homophily-aware Structural and Semantic Text-Attributed Graph Compression 

**Authors**: Zijun Di, Bin Lu, Huquan Kang, Luoyi Fu, Jiaxin Ding, Xiaoying Gan, Lei Zhou, Xinbing Wang, Chenghu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.08187)  

**Abstract**: Large language models (LLMs) have demonstrated promising capabilities in Text-Attributed Graph (TAG) understanding. Recent studies typically focus on verbalizing the graph structures via handcrafted prompts, feeding the target node and its neighborhood context into LLMs. However, constrained by the context window, existing methods mainly resort to random sampling, often implemented via dropping node/edge randomly, which inevitably introduces noise and cause reasoning instability. We argue that graphs inherently contain rich structural and semantic information, and that their effective exploitation can unlock potential gains in LLMs reasoning performance. To this end, we propose Homophily-aware Structural and Semantic Compression for LLMs (HS2C), a framework centered on exploiting graph homophily. Structurally, guided by the principle of Structural Entropy minimization, we perform a global hierarchical partition that decodes the graph's essential topology. This partition identifies naturally cohesive, homophilic communities, while discarding stochastic connectivity noise. Semantically, we deliver the detected structural homophily to the LLM, empowering it to perform differentiated semantic aggregation based on predefined community type. This process compresses redundant background contexts into concise community-level consensus, selectively preserving semantically homophilic information aligned with the target nodes. Extensive experiments on 10 node-level benchmarks across LLMs of varying sizes and families demonstrate that, by feeding LLMs with structurally and semantically compressed inputs, HS2C simultaneously enhances the compression rate and downstream inference accuracy, validating its superiority and scalability. Extensions to 7 diverse graph-level benchmarks further consolidate HS2C's task generalizability. 

---
# The Agent's First Day: Benchmarking Learning, Exploration, and Scheduling in the Workplace Scenarios 

**Authors**: Daocheng Fu, Jianbiao Mei, Rong Wu, Xuemeng Yang, Jia Xu, Ding Wang, Pinlong Cai, Yong Liu, Licheng Wen, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08173)  

**Abstract**: The rapid evolution of Multi-modal Large Language Models (MLLMs) has advanced workflow automation; however, existing research mainly targets performance upper bounds in static environments, overlooking robustness for stochastic real-world deployment. We identify three key challenges: dynamic task scheduling, active exploration under uncertainty, and continuous learning from experience. To bridge this gap, we introduce \method{}, a dynamic evaluation environment that simulates a "trainee" agent continuously exploring a novel setting. Unlike traditional benchmarks, \method{} evaluates agents along three dimensions: (1) context-aware scheduling for streaming tasks with varying priorities; (2) prudent information acquisition to reduce hallucination via active exploration; and (3) continuous evolution by distilling generalized strategies from rule-based, dynamically generated tasks. Experiments show that cutting-edge agents have significant deficiencies in dynamic environments, especially in active exploration and continual learning. Our work establishes a framework for assessing agent reliability, shifting evaluation from static tests to realistic, production-oriented scenarios. Our codes are available at this https URL 

---
# ZeroDVFS: Zero-Shot LLM-Guided Core and Frequency Allocation for Embedded Platforms 

**Authors**: Mohammad Pivezhandi, Mahdi Banisharif, Abusayeed Saifullah, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2601.08166)  

**Abstract**: Dynamic voltage and frequency scaling (DVFS) and task-to-core allocation are critical for thermal management and balancing energy and performance in embedded systems. Existing approaches either rely on utilization-based heuristics that overlook stall times, or require extensive offline profiling for table generation, preventing runtime adaptation. We propose a model-based hierarchical multi-agent reinforcement learning (MARL) framework for thermal- and energy-aware scheduling on multi-core platforms. Two collaborative agents decompose the exponential action space, achieving 358ms latency for subsequent decisions. First decisions require 3.5 to 8.0s including one-time LLM feature extraction. An accurate environment model leverages regression techniques to predict thermal dynamics and performance states. When combined with LLM-extracted semantic features, the environment model enables zero-shot deployment for new workloads on trained platforms by generating synthetic training data without requiring workload-specific profiling samples. We introduce LLM-based semantic feature extraction that characterizes OpenMP programs through 13 code-level features without execution. The Dyna-Q-inspired framework integrates direct reinforcement learning with model-based planning, achieving 20x faster convergence than model-free methods. Experiments on BOTS and PolybenchC benchmarks across NVIDIA Jetson TX2, Jetson Orin NX, RubikPi, and Intel Core i7 demonstrate 7.09x better energy efficiency and 4.0x better makespan than Linux ondemand governor. First-decision latency is 8,300x faster than table-based profiling, enabling practical deployment in dynamic embedded systems. 

---
# Project Synapse: A Hierarchical Multi-Agent Framework with Hybrid Memory for Autonomous Resolution of Last-Mile Delivery Disruptions 

**Authors**: Arin Gopalan Yadav, Varad Dherange, Kumar Shivam  

**Link**: [PDF](https://arxiv.org/pdf/2601.08156)  

**Abstract**: This paper introduces Project Synapse, a novel agentic framework designed for the autonomous resolution of last-mile delivery disruptions. Synapse employs a hierarchical multi-agent architecture in which a central Resolution Supervisor agent performs strategic task decomposition and delegates subtasks to specialized worker agents responsible for tactical execution. The system is orchestrated using LangGraph to manage complex and cyclical workflows. To validate the framework, a benchmark dataset of 30 complex disruption scenarios was curated from a qualitative analysis of over 6,000 real-world user reviews. System performance is evaluated using an LLM-as-a-Judge protocol with explicit bias mitigation. 

---
# Embedded AI Companion System on Edge Devices 

**Authors**: Rahul Gupta, Stephen D.H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08128)  

**Abstract**: Computational resource constraints on edge devices make it difficult to develop a fully embedded AI companion system with a satisfactory user experience. AI companion and memory systems detailed in existing literature cannot be directly used in such an environment due to lack of compute resources and latency concerns. In this paper, we propose a memory paradigm that alternates between active and inactive phases: during phases of user activity, the system performs low-latency, real-time dialog using lightweight retrieval over existing memories and context; whereas during phases of user inactivity, it conducts more computationally intensive extraction, consolidation, and maintenance of memories across full conversation sessions. This design minimizes latency while maintaining long-term personalization under the tight constraints of embedded hardware. We also introduce an AI Companion benchmark designed to holistically evaluate the AI Companion across both its conversational quality and memory capabilities. In our experiments, we found that our system (using a very weak model: Qwen2.5-7B-Instruct quantized int4) outperforms the equivalent raw LLM without memory across most metrics, and performs comparably to GPT-3.5 with 16k context window. 

---
# How vehicles change lanes after encountering crashes: Empirical analysis and modeling 

**Authors**: Kequan Chen, Yuxuan Wang, Pan Liu, Victor L. Knoop, David Z. W. Wang, Yu Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.08125)  

**Abstract**: When a traffic crash occurs, following vehicles need to change lanes to bypass the obstruction. We define these maneuvers as post crash lane changes. In such scenarios, vehicles in the target lane may refuse to yield even after the lane change has already begun, increasing the complexity and crash risk of post crash LCs. However, the behavioral characteristics and motion patterns of post crash LCs remain unknown. To address this gap, we construct a post crash LC dataset by extracting vehicle trajectories from drone videos captured after crashes. Our empirical analysis reveals that, compared to mandatory LCs (MLCs) and discretionary LCs (DLCs), post crash LCs exhibit longer durations, lower insertion speeds, and higher crash risks. Notably, 79.4% of post crash LCs involve at least one instance of non yielding behavior from the new follower, compared to 21.7% for DLCs and 28.6% for MLCs. Building on these findings, we develop a novel trajectory prediction framework for post crash LCs. At its core is a graph based attention module that explicitly models yielding behavior as an auxiliary interaction aware task. This module is designed to guide both a conditional variational autoencoder and a Transformer based decoder to predict the lane changer's trajectory. By incorporating the interaction aware module, our model outperforms existing baselines in trajectory prediction performance by more than 10% in both average displacement error and final displacement error across different prediction horizons. Moreover, our model provides more reliable crash risk analysis by reducing false crash rates and improving conflict prediction accuracy. Finally, we validate the model's transferability using additional post crash LC datasets collected from different sites. 

---
# MirrorBench: An Extensible Framework to Evaluate User-Proxy Agents for Human-Likeness 

**Authors**: Ashutosh Hathidara, Julien Yu, Vaishali Senthil, Sebastian Schreiber, Anil Babu Ankisettipalli  

**Link**: [PDF](https://arxiv.org/pdf/2601.08118)  

**Abstract**: Large language models (LLMs) are increasingly used as human simulators, both for evaluating conversational systems and for generating fine-tuning data. However, naive "act-as-a-user" prompting often yields verbose, unrealistic utterances, underscoring the need for principled evaluation of so-called user proxy agents. We present MIRRORBENCH, a reproducible, extensible benchmarking framework that evaluates user proxies solely on their ability to produce human-like user utterances across diverse conversational tasks, explicitly decoupled from downstream task success. MIRRORBENCH features a modular execution engine with typed interfaces, metadata-driven registries, multi-backend support, caching, and robust observability. The system supports pluggable user proxies, datasets, tasks, and metrics, enabling researchers to evaluate arbitrary simulators under a uniform, variance-aware harness. We include three lexical-diversity metrics (MATTR, YULE'S K, and HD-D) and three LLM-judge-based metrics (GTEval, Pairwise Indistinguishability, and Rubric-and-Reason). Across four open datasets, MIRRORBENCH yields variance-aware results and reveals systematic gaps between user proxies and real human users. The framework is open source and includes a simple command-line interface for running experiments, managing configurations and caching, and generating reports. The framework can be accessed at this https URL. 

---
# MemoBrain: Executive Memory as an Agentic Brain for Reasoning 

**Authors**: Hongjin Qian, Zhao Cao, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08079)  

**Abstract**: Complex reasoning in tool-augmented agent frameworks is inherently long-horizon, causing reasoning traces and transient tool artifacts to accumulate and strain the bounded working context of large language models. Without explicit memory mechanisms, such accumulation disrupts logical continuity and undermines task alignment. This positions memory not as an auxiliary efficiency concern, but as a core component for sustaining coherent, goal-directed reasoning over long horizons.
We propose MemoBrain, an executive memory model for tool-augmented agents that constructs a dependency-aware memory over reasoning steps, capturing salient intermediate states and their logical relations. Operating as a co-pilot alongside the reasoning agent, MemoBrain organizes reasoning progress without blocking execution and actively manages the working context. Specifically, it prunes invalid steps, folds completed sub-trajectories, and preserves a compact, high-salience reasoning backbone under a fixed context budget. Together, these mechanisms enable explicit cognitive control over reasoning trajectories rather than passive context accumulation.
We evaluate MemoBrain on challenging long-horizon benchmarks, including GAIA, WebWalker, and BrowseComp-Plus, demonstrating consistent improvements over strong baselines. 

---
# Semantic Gravity Wells: Why Negative Constraints Backfire 

**Authors**: Shailesh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2601.08070)  

**Abstract**: Negative constraints (instructions of the form "do not use word X") represent a fundamental test of instruction-following capability in large language models. Despite their apparent simplicity, these constraints fail with striking regularity, and the conditions governing failure have remained poorly understood. This paper presents the first comprehensive mechanistic investigation of negative instruction failure. We introduce semantic pressure, a quantitative measure of the model's intrinsic probability of generating the forbidden token, and demonstrate that violation probability follows a tight logistic relationship with pressure ($p=\sigma(-2.40+2.27\cdot P_0)$; $n=40{,}000$ samples; bootstrap $95%$ CI for slope: $[2.21,,2.33]$). Through layer-wise analysis using the logit lens technique, we establish that the suppression signal induced by negative instructions is present but systematically weaker in failures: the instruction reduces target probability by only 5.2 percentage points in failures versus 22.8 points in successes -- a $4.4\times$ asymmetry. We trace this asymmetry to two mechanistically distinct failure modes. In priming failure (87.5% of violations), the instruction's explicit mention of the forbidden word paradoxically activates rather than suppresses the target representation. In override failure (12.5%), late-layer feed-forward networks generate contributions of $+0.39$ toward the target probability -- nearly $4\times$ larger than in successes -- overwhelming earlier suppression signals. Activation patching confirms that layers 23--27 are causally responsible: replacing these layers' activations flips the sign of constraint effects. These findings reveal a fundamental tension in negative constraint design: the very act of naming a forbidden word primes the model to produce it. 

---
# A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems 

**Authors**: Samuel I. Akinwande, Sydney M. Katz, Mykel J. Kochenderfer, Clark Barrett  

**Link**: [PDF](https://arxiv.org/pdf/2601.08065)  

**Abstract**: Forward reachability analysis is the predominant approach for verifying reach-avoid properties in neural feedback systems (dynamical systems controlled by neural networks). This dominance stems from the limited scalability of existing backward reachability methods. In this work, we introduce new algorithms that compute both over- and under-approximations of backward reachable sets for such systems. We further integrate these backward algorithms with established forward analysis techniques to yield a unified verification framework for neural feedback systems. 

---
# Forecast Aware Deep Reinforcement Learning for Efficient Electricity Load Scheduling in Dairy Farms 

**Authors**: Nawazish Alia, Rachael Shawb, Karl Mason  

**Link**: [PDF](https://arxiv.org/pdf/2601.08052)  

**Abstract**: Dairy farming is an energy intensive sector that relies heavily on grid electricity. With increasing renewable energy integration, sustainable energy management has become essential for reducing grid dependence and supporting the United Nations Sustainable Development Goal 7 on affordable and clean energy. However, the intermittent nature of renewables poses challenges in balancing supply and demand in real time. Intelligent load scheduling is therefore crucial to minimize operational costs while maintaining reliability. Reinforcement Learning has shown promise in improving energy efficiency and reducing costs. However, most RL-based scheduling methods assume complete knowledge of future prices or generation, which is unrealistic in dynamic environments. Moreover, standard PPO variants rely on fixed clipping or KL divergence thresholds, often leading to unstable training under variable tariffs. To address these challenges, this study proposes a Deep Reinforcement Learning framework for efficient load scheduling in dairy farms, focusing on battery storage and water heating under realistic operational constraints. The proposed Forecast Aware PPO incorporates short term forecasts of demand and renewable generation using hour of day and month based residual calibration, while the PID KL PPO variant employs a proportional integral derivative controller to regulate KL divergence for stable policy updates adaptively. Trained on real world dairy farm data, the method achieves up to 1% lower electricity cost than PPO, 4.8% than DQN, and 1.5% than SAC. For battery scheduling, PPO reduces grid imports by 13.1%, demonstrating scalability and effectiveness for sustainable energy management in modern dairy farming. 

---
# Integrating Attendance Tracking and Emotion Detection for Enhanced Student Engagement in Smart Classrooms 

**Authors**: Keith Ainebyona, Ann Move Oguti, Joseph Walusimbi, Ritah Kobusingye  

**Link**: [PDF](https://arxiv.org/pdf/2601.08049)  

**Abstract**: The increasing adoption of smart classroom technologies in higher education has mainly focused on automating attendance, with limited attention given to students' emotional and cognitive engagement during lectures. This limits instructors' ability to identify disengagement and adapt teaching strategies in real time. This paper presents SCASED (Smart Classroom Attendance System with Emotion Detection), an IoT-based system that integrates automated attendance tracking with facial emotion recognition to support classroom engagement monitoring. The system uses a Raspberry Pi camera and OpenCV for face detection, and a finetuned MobileNetV2 model to classify four learning-related emotional states: engagement, boredom, confusion, and frustration. A session-based mechanism is implemented to manage attendance and emotion monitoring by recording attendance once per session and performing continuous emotion analysis thereafter. Attendance and emotion data are visualized through a cloud-based dashboard to provide instructors with insights into classroom dynamics. Experimental evaluation using the DAiSEE dataset achieved an emotion classification accuracy of 89.5%. The results show that integrating attendance data with emotion analytics can provide instructors with additional insight into classroom dynamics and support more responsive teaching practices. 

---
# Internal Deployment Gaps in AI Regulation 

**Authors**: Joe Kwon, Stephen Casper  

**Link**: [PDF](https://arxiv.org/pdf/2601.08005)  

**Abstract**: Frontier AI regulations primarily focus on systems deployed to external users, where deployment is more visible and subject to outside scrutiny. However, high-stakes applications can occur internally when companies deploy highly capable systems within their own organizations, such as for automating R\&D, accelerating critical business processes, and handling sensitive proprietary data. This paper examines how frontier AI regulations in the United States and European Union in 2025 handle internal deployment. We identify three gaps that could cause internally-deployed systems to evade intended oversight: (1) scope ambiguity that allows internal systems to evade regulatory obligations, (2) point-in-time compliance assessments that fail to capture the continuous evolution of internal systems, and (3) information asymmetries that subvert regulatory awareness and oversight. We then analyze why these gaps persist, examining tensions around measurability, incentives, and information access. Finally, we map potential approaches to address them and their associated tradeoffs. By understanding these patterns, we hope that policy choices around internally deployed AI systems can be made deliberately rather than incidentally. 

---
# Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety 

**Authors**: Can Jin, Rui Wu, Tong Che, Qixin Zhang, Hongwu Peng, Jiahui Zhao, Zhenting Wang, Wenqi Wei, Ligong Han, Zhao Zhang, Yuan Cao, Ruixiang Tang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2601.08000)  

**Abstract**: Ensuring that Large Language Models (LLMs) adhere to safety principles without refusing benign requests remains a significant challenge. While OpenAI introduces deliberative alignment (DA) to enhance the safety of its o-series models through reasoning over detailed ``code-like'' safety rules, the effectiveness of this approach in open-source LLMs, which typically lack advanced reasoning capabilities, is understudied. In this work, we systematically evaluate the impact of explicitly specifying extensive safety codes versus demonstrating them through illustrative cases. We find that referencing explicit codes inconsistently improves harmlessness and systematically degrades helpfulness, whereas training on case-augmented simple codes yields more robust and generalized safety behaviors. By guiding LLMs with case-augmented reasoning instead of extensive code-like safety rules, we avoid rigid adherence to narrowly enumerated rules and enable broader adaptability. Building on these insights, we propose CADA, a case-augmented deliberative alignment method for LLMs utilizing reinforcement learning on self-generated safety reasoning chains. CADA effectively enhances harmlessness, improves robustness against attacks, and reduces over-refusal while preserving utility across diverse benchmarks, offering a practical alternative to rule-only DA for improving safety while maintaining helpfulness. 

---
# When Models Know When They Do Not Know: Calibration, Cascading, and Cleaning 

**Authors**: Chenjie Hao, Weyl Lu, Yuko Ishiwaka, Zengyi Li, Weier Wan, Yubei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07965)  

**Abstract**: When a model knows when it does not know, many possibilities emerge. The first question is how to enable a model to recognize that it does not know. A promising approach is to use confidence, computed from the model's internal signals, to reflect its ignorance. Prior work in specific domains has shown that calibration can provide reliable confidence estimates. In this work, we propose a simple, effective, and universal training-free method that applies to both vision and language models, performing model calibration, cascading, and data cleaning to better exploit a model's ability to recognize when it does not know. We first highlight two key empirical observations: higher confidence corresponds to higher accuracy within a single model, and models calibrated on the validation set remain calibrated on a held-out test set. These findings empirically establish the reliability and comparability of calibrated confidence. Building on this, we introduce two applications: (1) model cascading with calibrated advantage routing and (2) data cleaning based on model ensemble. Using the routing signal derived from the comparability of calibrated confidences, we cascade large and small models to improve efficiency with almost no compromise in accuracy, and we further cascade two models of comparable scale to achieve performance beyond either model alone. Leveraging multiple experts and their calibrated confidences, we design a simple yet effective data-cleaning method that balances precision and detection rate to identify mislabeled samples in ImageNet and Massive Multitask Language Understanding (MMLU) datasets. Our results demonstrate that enabling models to recognize when they do not know is a practical step toward more efficient, reliable, and trustworthy AI. 

---
# Executable Ontologies in Game Development: From Algorithmic Control to Semantic World Modeling 

**Authors**: Alexander Boldachev  

**Link**: [PDF](https://arxiv.org/pdf/2601.07964)  

**Abstract**: This paper examines the application of Executable Ontologies (EO), implemented through the boldsea framework, to game development. We argue that EO represents a paradigm shift: a transition from algorithmic behavior programming to semantic world modeling, where agent behavior emerges naturally from declarative domain rules rather than being explicitly coded. Using a survival game scenario (Winter Feast), we demonstrate how EO achieves prioritybased task interruption through dataflow conditions rather than explicit preemption logic. Comparison with Behavior Trees (BT) and Goal-Oriented Action Planning (GOAP) reveals that while these approaches model what agents should do, EO models when actions become possible - a fundamental difference that addresses the semantic-process gap in game AI architecture. We discuss integration strategies, debugging advantages inherent to temporal event graphs, and the potential for LLM-driven runtime model generation. 

---
# Bridging the Trust Gap: Clinician-Validated Hybrid Explainable AI for Maternal Health Risk Assessment in Bangladesh 

**Authors**: Farjana Yesmin, Nusrat Shirmin, Suraiya Shabnam Bristy  

**Link**: [PDF](https://arxiv.org/pdf/2601.07866)  

**Abstract**: While machine learning shows promise for maternal health risk prediction, clinical adoption in resource-constrained settings faces a critical barrier: lack of explainability and trust. This study presents a hybrid explainable AI (XAI) framework combining ante-hoc fuzzy logic with post-hoc SHAP explanations, validated through systematic clinician feedback. We developed a fuzzy-XGBoost model on 1,014 maternal health records, achieving 88.67% accuracy (ROC-AUC: 0.9703). A validation study with 14 healthcare professionals in Bangladesh revealed strong preference for hybrid explanations (71.4% across three clinical cases) with 54.8% expressing trust for clinical use. SHAP analysis identified healthcare access as the primary predictor, with the engineered fuzzy risk score ranking third, validating clinical knowledge integration (r=0.298). Clinicians valued integrated clinical parameters but identified critical gaps: obstetric history, gestational age, and connectivity barriers. This work demonstrates that combining interpretable fuzzy rules with feature importance explanations enhances both utility and trust, providing practical insights for XAI deployment in maternal healthcare. 

---
# Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System 

**Authors**: Hsiang-Wei Huang, Junbin Lu, Kuang-Ming Chen, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08829)  

**Abstract**: In this work, we explore the Large Language Model (LLM) agent reviewer dynamics in an Elo-ranked review system using real-world conference paper submissions. Multiple LLM agent reviewers with different personas are engage in multi round review interactions moderated by an Area Chair. We compare a baseline setting with conditions that incorporate Elo ratings and reviewer memory. Our simulation results showcase several interesting findings, including how incorporating Elo improves Area Chair decision accuracy, as well as reviewers' adaptive review strategy that exploits our Elo system without improving review effort. Our code is available at this https URL. 

---
# Motion Attribution for Video Generation 

**Authors**: Xindi Wu, Despoina Paschalidou, Jun Gao, Antonio Torralba, Laura Leal-Taixé, Olga Russakovsky, Sanja Fidler, Jonathan Lorraine  

**Link**: [PDF](https://arxiv.org/pdf/2601.08828)  

**Abstract**: Despite the rapid progress of video generation models, the role of data in influencing motion is poorly understood. We present Motive (MOTIon attribution for Video gEneration), a motion-centric, gradient-based data attribution framework that scales to modern, large, high-quality video datasets and models. We use this to study which fine-tuning clips improve or degrade temporal dynamics. Motive isolates temporal dynamics from static appearance via motion-weighted loss masks, yielding efficient and scalable motion-specific influence computation. On text-to-video models, Motive identifies clips that strongly affect motion and guides data curation that improves temporal consistency and physical plausibility. With Motive-selected high-influence data, our method improves both motion smoothness and dynamic degree on VBench, achieving a 74.1% human preference win rate compared with the pretrained base model. To our knowledge, this is the first framework to attribute motion rather than visual appearance in video generative models and to use it to curate fine-tuning data. 

---
# MemRec: Collaborative Memory-Augmented Agentic Recommender System 

**Authors**: Weixin Chen, Yuhan Zhao, Jingyuan Huang, Zihe Ye, Clark Mingxuan Ju, Tong Zhao, Neil Shah, Li Chen, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08816)  

**Abstract**: The evolution of recommender systems has shifted preference storage from rating matrices and dense embeddings to semantic memory in the agentic era. Yet existing agents rely on isolated memory, overlooking crucial collaborative signals. Bridging this gap is hindered by the dual challenges of distilling vast graph contexts without overwhelming reasoning agents with cognitive load, and evolving the collaborative memory efficiently without incurring prohibitive computational costs. To address this, we propose MemRec, a framework that architecturally decouples reasoning from memory management to enable efficient collaborative augmentation. MemRec introduces a dedicated, cost-effective LM_Mem to manage a dynamic collaborative memory graph, serving synthesized, high-signal context to a downstream LLM_Rec. The framework operates via a practical pipeline featuring efficient retrieval and cost-effective asynchronous graph propagation that evolves memory in the background. Extensive experiments on four benchmarks demonstrate that MemRec achieves state-of-the-art performance. Furthermore, architectural analysis confirms its flexibility, establishing a new Pareto frontier that balances reasoning quality, cost, and privacy through support for diverse deployments, including local open-source models. Code:this https URL and Homepage: this https URL 

---
# Reasoning Matters for 3D Visual Grounding 

**Authors**: Hsiang-Wei Huang, Kuang-Ming Chen, Wenhao Chai, Cheng-Yen Yang, Jen-Hao Cheng, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08811)  

**Abstract**: The recent development of Large Language Models (LLMs) with strong reasoning ability has driven research in various domains such as mathematics, coding, and scientific discovery. Meanwhile, 3D visual grounding, as a fundamental task in 3D understanding, still remains challenging due to the limited reasoning ability of recent 3D visual grounding models. Most of the current methods incorporate a text encoder and visual feature encoder to generate cross-modal fuse features and predict the referring object. These models often require supervised training on extensive 3D annotation data. On the other hand, recent research also focus on scaling synthetic data to train stronger 3D visual grounding LLM, however, the performance gain remains limited and non-proportional to the data collection cost. In this work, we propose a 3D visual grounding data pipeline, which is capable of automatically synthesizing 3D visual grounding data along with corresponding reasoning process. Additionally, we leverage the generated data for LLM fine-tuning and introduce Reason3DVG-8B, a strong 3D visual grounding LLM that outperforms previous LLM-based method 3D-GRAND using only 1.6% of their training data, demonstrating the effectiveness of our data and the importance of reasoning in 3D visual grounding. 

---
# Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge 

**Authors**: Yao Tang, Li Dong, Yaru Hao, Qingxiu Dong, Furu Wei, Jiatao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08808)  

**Abstract**: Large language models often solve complex reasoning tasks more effectively with Chain-of-Thought (CoT), but at the cost of long, low-bandwidth token sequences. Humans, by contrast, often reason softly by maintaining a distribution over plausible next steps. Motivated by this, we propose Multiplex Thinking, a stochastic soft reasoning mechanism that, at each thinking step, samples K candidate tokens and aggregates their embeddings into a single continuous multiplex token. This preserves the vocabulary embedding prior and the sampling dynamics of standard discrete generation, while inducing a tractable probability distribution over multiplex rollouts. Consequently, multiplex trajectories can be directly optimized with on-policy reinforcement learning (RL). Importantly, Multiplex Thinking is self-adaptive: when the model is confident, the multiplex token is nearly discrete and behaves like standard CoT; when it is uncertain, it compactly represents multiple plausible next steps without increasing sequence length. Across challenging math reasoning benchmarks, Multiplex Thinking consistently outperforms strong discrete CoT and RL baselines from Pass@1 through Pass@1024, while producing shorter sequences. The code and checkpoints are available at this https URL. 

---
# S3-CLIP: Video Super Resolution for Person-ReID 

**Authors**: Tamas Endrei, Gyorgy Cserey  

**Link**: [PDF](https://arxiv.org/pdf/2601.08807)  

**Abstract**: Tracklet quality is often treated as an afterthought in most person re-identification (ReID) methods, with the majority of research presenting architectural modifications to foundational models. Such approaches neglect an important limitation, posing challenges when deploying ReID systems in real-world, difficult scenarios. In this paper, we introduce S3-CLIP, a video super-resolution-based CLIP-ReID framework developed for the VReID-XFD challenge at WACV 2026. The proposed method integrates recent advances in super-resolution networks with task-driven super-resolution pipelines, adapting them to the video-based person re-identification setting. To the best of our knowledge, this work represents the first systematic investigation of video super-resolution as a means of enhancing tracklet quality for person ReID, particularly under challenging cross-view conditions. Experimental results demonstrate performance competitive with the baseline, achieving 37.52% mAP in aerial-to-ground and 29.16% mAP in ground-to-aerial scenarios. In the ground-to-aerial setting, S3-CLIP achieves substantial gains in ranking accuracy, improving Rank-1, Rank-5, and Rank-10 performance by 11.24%, 13.48%, and 17.98%, respectively. 

---
# APEX-SWE 

**Authors**: Abhi Kottamasu, Akul Datta, Aakash Barthwal, Chirag Mahapatra, Ajay Arun, Adarsh Hiremath, Brendan Foody, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08806)  

**Abstract**: We introduce the AI Productivity Index for Software Engineering (APEX-SWE), a benchmark for assessing whether frontier AI models can execute economically valuable software engineering work. Unlike existing evaluations that focus on narrow, well-defined tasks, APEX-SWE assesses two novel task types that reflect real-world software engineering work: (1) Integration tasks (n=100), which require constructing end-to-end systems across heterogeneous cloud primitives, business applications, and infrastructure-as-code services, and (2) Observability tasks (n=100), which require debugging production failures using telemetry signals such as logs and dashboards, as well as unstructured context. We evaluated eight frontier models on APEX-SWE. Gemini 3 Pro (Thinking = High) performs best, with a Pass@1 score of 25\%. Our analysis shows that strong performance is primarily driven by epistemic reasoning, defined as the ability to distinguish between assumptions and verified facts, combined with agency to resolve uncertainty prior to acting. We open-source the APEX-SWE evaluation harness and a dev set (n=50). 

---
# Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling 

**Authors**: Yang Cai, Weiqiang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08777)  

**Abstract**: Aligning large language models (LLMs) to serve users with heterogeneous and potentially conflicting preferences is a central challenge for personalized and trustworthy AI. We formalize an ideal notion of universal alignment through test-time scaling: for each prompt, the model produces $k\ge 1$ candidate responses and a user selects their preferred one. We introduce $(k,f(k))$-robust alignment, which requires the $k$-output model to have win rate $f(k)$ against any other single-output model, and asymptotic universal alignment (U-alignment), which requires $f(k)\to 1$ as $k\to\infty$. Our main result characterizes the optimal convergence rate: there exists a family of single-output policies whose $k$-sample product policies achieve U-alignment at rate $f(k)=\frac{k}{k+1}$, and no method can achieve a faster rate in general.
We show that popular post-training methods, including Nash learning from human feedback (NLHF), can fundamentally underutilize the benefits of test-time scaling. Even though NLHF is optimal for $k=1$, sampling from the resulting (often deterministic) policy cannot guarantee win rates above $\tfrac{1}{2}$ except for an arbitrarily small slack. This stems from a lack of output diversity: existing alignment methods can collapse to a single majority-preferred response, making additional samples redundant. In contrast, our approach preserves output diversity and achieves the optimal test-time scaling rate. In particular, we propose a family of symmetric multi-player alignment games and prove that any symmetric Nash equilibrium policy of the $(k+1)$-player alignment game achieves the optimal $(k,\frac{k}{k+1})$-robust alignment. Finally, we provide theoretical convergence guarantees for self-play learning dynamics in these games and extend the framework to opponents that also generate multiple responses. 

---
# Translating Light-Sheet Microscopy Images to Virtual H&E Using CycleGAN 

**Authors**: Yanhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08776)  

**Abstract**: Histopathology analysis relies on Hematoxylin and Eosin (H&E) staining, but fluorescence microscopy offers complementary information. Converting fluorescence images to H&E-like appearance can aid interpretation and integration with standard workflows. We present a Cycle-Consistent Adversarial Network (CycleGAN) approach for unpaired image-to-image translation from multi-channel fluorescence microscopy to pseudo H&E stained histopathology images. The method combines C01 and C02 fluorescence channels into RGB and learns a bidirectional mapping between fluorescence and H&E domains without paired training data. The architecture uses ResNet-based generators with residual blocks and PatchGAN discriminators, trained with adversarial, cycle-consistency, and identity losses. Experiments on fluorescence microscopy datasets show the model generates realistic pseudo H&E images that preserve morphological structures while adopting H&E-like color characteristics. This enables visualization of fluorescence data in a format familiar to pathologists and supports integration with existing H&E-based analysis pipelines. 

---
# Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs 

**Authors**: Manideep Reddy Chinthareddy  

**Link**: [PDF](https://arxiv.org/pdf/2601.08773)  

**Abstract**: Retrieval-Augmented Generation for software engineering often relies on vector similarity search, which captures topical similarity but can fail on multi-hop architectural reasoning such as controller to service to repository chains, interface-driven wiring, and inheritance. This paper benchmarks three retrieval pipelines on Java codebases (Shopizer, with additional runs on ThingsBoard and OpenMRS Core): (A) vector-only No-Graph RAG, (B) an LLM-generated knowledge graph RAG (LLM-KB), and (C) a deterministic AST-derived knowledge graph RAG (DKB) built with Tree-sitter and bidirectional traversal.
Using 15 architecture and code-tracing queries per repository, we measure indexing time, query latency, corpus coverage, cost, and answer correctness. DKB builds its graph in seconds, while LLM-KB requires much longer graph generation. LLM-KB also shows indexing incompleteness: on Shopizer, 377 files are skipped or missed, reducing embedded chunk coverage and graph size compared to DKB. End-to-end cost is modest for DKB relative to the vector-only baseline but much higher for LLM-KB, especially as repository scale increases. Query latency is similar for No-Graph and DKB, while LLM-KB is slower and more variable. On the Shopizer question suite, DKB achieves the highest correctness, LLM-KB is close behind, and the vector-only baseline performs worst on upstream architectural queries and has the highest hallucination risk. Overall, deterministic AST-derived graphs provide more reliable coverage and multi-hop grounding than LLM-extracted graphs at substantially lower indexing cost. 

---
# Grid-Aware Charging and Operational Optimization for Mixed-Fleet Public Transit 

**Authors**: Rishav Sen, Amutheezan Sivagnanam, Aron Laszka, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.08753)  

**Abstract**: The rapid growth of urban populations and the increasing need for sustainable transportation solutions have prompted a shift towards electric buses in public transit systems. However, the effective management of mixed fleets consisting of both electric and diesel buses poses significant operational challenges. One major challenge is coping with dynamic electricity pricing, where charging costs vary throughout the day. Transit agencies must optimize charging assignments in response to such dynamism while accounting for secondary considerations such as seating constraints. This paper presents a comprehensive mixed-integer linear programming (MILP) model to address these challenges by jointly optimizing charging schedules and trip assignments for mixed (electric and diesel bus) fleets while considering factors such as dynamic electricity pricing, vehicle capacity, and route constraints. We address the potential computational intractability of the MILP formulation, which can arise even with relatively small fleets, by employing a hierarchical approach tailored to the fleet composition. By using real-world data from the city of Chattanooga, Tennessee, USA, we show that our approach can result in significant savings in the operating costs of the mixed transit fleets. 

---
# UR-Bench: A Benchmark for Multi-Hop Reasoning over Ultra-High-Resolution Images 

**Authors**: Siqi Li, Xinyu Cai, Jianbiao Mei, Nianchen Deng, Pinlong Cai, Licheng Wen, Yufan Shen, Xuemeng Yang, Botian Shi, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08748)  

**Abstract**: Recent multimodal large language models (MLLMs) show strong capabilities in visual-language reasoning, yet their performance on ultra-high-resolution imagery remains largely unexplored. Existing visual question answering (VQA) benchmarks typically rely on medium-resolution data, offering limited visual complexity. To bridge this gap, we introduce Ultra-high-resolution Reasoning Benchmark (UR-Bench), a benchmark designed to evaluate the reasoning capabilities of MLLMs under extreme visual information. UR-Bench comprises two major categories, Humanistic Scenes and Natural Scenes, covering four subsets of ultra-high-resolution images with distinct spatial structures and data sources. Each subset contains images ranging from hundreds of megapixels to gigapixels, accompanied by questions organized into three levels, enabling evaluation of models' reasoning capabilities in ultra-high-resolution scenarios. We further propose an agent-based framework in which a language model performs reasoning by invoking external visual tools. In addition, we introduce Semantic Abstraction and Retrieval tools that enable more efficient processing of ultra-high-resolution images. We evaluate state-of-the-art models using both an end-to-end MLLMs and our agent-based framework, demonstrating the effectiveness of our framework. 

---
# To Retrieve or To Think? An Agentic Approach for Context Evolution 

**Authors**: Rubing Chen, Jian Wang, Wenjie Li, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08747)  

**Abstract**: Current context augmentation methods, such as retrieval-augmented generation, are essential for solving knowledge-intensive reasoning this http URL, they typically adhere to a rigid, brute-force strategy that executes retrieval at every step. This indiscriminate approach not only incurs unnecessary computational costs but also degrades performance by saturating the context with irrelevant noise. To address these limitations, we introduce Agentic Context Evolution (ACE), a framework inspired by human metacognition that dynamically determines whether to seek new evidence or reason with existing knowledge. ACE employs a central orchestrator agent to make decisions strategically via majority this http URL aims to alternate between activating a retriever agent for external retrieval and a reasoner agent for internal analysis and refinement. By eliminating redundant retrieval steps, ACE maintains a concise and evolved context. Extensive experiments on challenging multi-hop QA benchmarks demonstrate that ACE significantly outperforms competitive baselines in accuracy while achieving efficient token this http URL work provides valuable insights into advancing context-evolved generation for complex, knowledge-intensive tasks. 

---
# TableCache: Primary Foreign Key Guided KV Cache Precomputation for Low Latency Text-to-SQL 

**Authors**: Jinbo Su, Yuxuan Hu, Cuiping Li, Hong Chen, Jia Li, Lintao Ma, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08743)  

**Abstract**: In Text-to-SQL tasks, existing LLM-based methods often include extensive database schemas in prompts, leading to long context lengths and increased prefilling latency. While user queries typically focus on recurrent table sets-offering an opportunity for KV cache sharing across queries-current inference engines, such as SGLang and vLLM, generate redundant prefix cache copies when processing user queries with varying table orders. To address this inefficiency, we propose precomputing table representations as KV caches offline and querying the required ones online. A key aspect of our approach is the computation of table caches while preserving primary foreign key relationships between tables. Additionally, we construct a Table Trie structure to facilitate efficient KV cache lookups during inference. To enhance cache performance, we introduce a cache management system with a query reranking strategy to improve cache hit rates and a computation loading pipeline for parallelizing model inference and cache loading. Experimental results show that our proposed TableCache achieves up to a 3.62x speedup in Time to First Token (TTFT) with negligible performance degradation. 

---
# TerraFormer: Automated Infrastructure-as-Code with LLMs Fine-Tuned via Policy-Guided Verifier Feedback 

**Authors**: Prithwish Jana, Sam Davidson, Bhavana Bhasker, Andrey Kan, Anoop Deoras, Laurent Callot  

**Link**: [PDF](https://arxiv.org/pdf/2601.08734)  

**Abstract**: Automating Infrastructure-as-Code (IaC) is challenging, and large language models (LLMs) often produce incorrect configurations from natural language (NL). We present TerraFormer, a neuro-symbolic framework for IaC generation and mutation that combines supervised fine-tuning with verifier-guided reinforcement learning, using formal verification tools to provide feedback on syntax, deployability, and policy compliance. We curate two large, high-quality NL-to-IaC datasets, TF-Gen (152k instances) and TF-Mutn (52k instances), via multi-stage verification and iterative LLM self-correction. Evaluations against 17 state-of-the-art LLMs, including ~50x larger models like Sonnet 3.7, DeepSeek-R1, and GPT-4.1, show that TerraFormer improves correctness over its base LLM by 15.94% on IaC-Eval, 11.65% on TF-Gen (Test), and 19.60% on TF-Mutn (Test). It outperforms larger models on both TF-Gen (Test) and TF-Mutn (Test), ranks third on IaC-Eval, and achieves top best-practices and security compliance. 

---
# ISLA: A U-Net for MRI-based acute ischemic stroke lesion segmentation with deep supervision, attention, domain adaptation, and ensemble learning 

**Authors**: Vincent Roca, Martin Bretzner, Hilde Henon, Laurent Puy, Grégory Kuchcinski, Renaud Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2601.08732)  

**Abstract**: Accurate delineation of acute ischemic stroke lesions in MRI is a key component of stroke diagnosis and management. In recent years, deep learning models have been successfully applied to the automatic segmentation of such lesions. While most proposed architectures are based on the U-Net framework, they primarily differ in their choice of loss functions and in the use of deep supervision, residual connections, and attention mechanisms. Moreover, many implementations are not publicly available, and the optimal configuration for acute ischemic stroke (AIS) lesion segmentation remains unclear. In this work, we introduce ISLA (Ischemic Stroke Lesion Analyzer), a new deep learning model for AIS lesion segmentation from diffusion MRI, trained on three multicenter databases totaling more than 1500 AIS participants. Through systematic optimization of the loss function, convolutional architecture, deep supervision, and attention mechanisms, we developed a robust segmentation framework. We further investigated unsupervised domain adaptation to improve generalization to an external clinical dataset. ISLA outperformed two state-of-the-art approaches for AIS lesion segmentation on an external test set. Codes and trained models will be made publicly available to facilitate reuse and reproducibility. 

---
# Real-Time Localization Framework for Autonomous Basketball Robots 

**Authors**: Naren Medarametla, Sreejon Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2601.08713)  

**Abstract**: Localization is a fundamental capability for autonomous robots, enabling them to operate effectively in dynamic environments. In Robocon 2025, accurate and reliable localization is crucial for improving shooting precision, avoiding collisions with other robots, and navigating the competition field efficiently. In this paper, we propose a hybrid localization algorithm that integrates classical techniques with learning based methods that rely solely on visual data from the court's floor to achieve self-localization on the basketball field. 

---
# Auditing Student-AI Collaboration: A Case Study of Online Graduate CS Students 

**Authors**: Nifu Dan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08697)  

**Abstract**: As generative AI becomes embedded in higher education, it increasingly shapes how students complete academic tasks. While these systems offer efficiency and support, concerns persist regarding over-automation, diminished student agency, and the potential for unreliable or hallucinated outputs. This study conducts a mixed-methods audit of student-AI collaboration preferences by examining the alignment between current AI capabilities and students' desired levels of automation in academic work. Using two sequential and complementary surveys, we capture students' perceived benefits, risks, and preferred boundaries when using AI. The first survey employs an existing task-based framework to assess preferences for and actual usage of AI across 12 academic tasks, alongside primary concerns and reasons for use. The second survey, informed by the first, explores how AI systems could be designed to address these concerns through open-ended questions. This study aims to identify gaps between existing AI affordances and students' normative expectations of collaboration, informing the development of more effective and trustworthy AI systems for education. 

---
# Region of interest detection for efficient aortic segmentation 

**Authors**: Loris Giordano, Ine Dirks, Tom Lenaerts, Jef Vandemeulebroucke  

**Link**: [PDF](https://arxiv.org/pdf/2601.08683)  

**Abstract**: Thoracic aortic dissection and aneurysms are the most lethal diseases of the aorta. The major hindrance to treatment lies in the accurate analysis of the medical images. More particularly, aortic segmentation of the 3D image is often tedious and difficult. Deep-learning-based segmentation models are an ideal solution, but their inability to deliver usable outputs in difficult cases and their computational cost cause their clinical adoption to stay limited. This study presents an innovative approach for efficient aortic segmentation using targeted region of interest (ROI) detection. In contrast to classical detection models, we propose a simple and efficient detection model that can be widely applied to detect a single ROI. Our detection model is trained as a multi-task model, using an encoder-decoder architecture for segmentation and a fully connected network attached to the bottleneck for detection. We compare the performance of a one-step segmentation model applied to a complete image, nnU-Net and our cascade model composed of a detection and a segmentation step. We achieve a mean Dice similarity coefficient of 0.944 with over 0.9 for all cases using a third of the computing power. This simple solution achieves state-of-the-art performance while being compact and robust, making it an ideal solution for clinical applications. 

---
# Lessons from the Field: An Adaptable Lifecycle Approach to Applied Dialogue Summarization 

**Authors**: Kushal Chawla, Chenyang Zhu, Pengshan Cai, Sangwoo Cho, Scott Novotney, Ayushman Singh, Jonah Lewis, Keasha Safewright, Alfy Samuel, Erin Babinsky, Shi-Xiong Zhang, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08682)  

**Abstract**: Summarization of multi-party dialogues is a critical capability in industry, enhancing knowledge transfer and operational effectiveness across many domains. However, automatically generating high-quality summaries is challenging, as the ideal summary must satisfy a set of complex, multi-faceted requirements. While summarization has received immense attention in research, prior work has primarily utilized static datasets and benchmarks, a condition rare in practical scenarios where requirements inevitably evolve. In this work, we present an industry case study on developing an agentic system to summarize multi-party interactions. We share practical insights spanning the full development lifecycle to guide practitioners in building reliable, adaptable summarization systems, as well as to inform future research, covering: 1) robust methods for evaluation despite evolving requirements and task subjectivity, 2) component-wise optimization enabled by the task decomposition inherent in an agentic architecture, 3) the impact of upstream data bottlenecks, and 4) the realities of vendor lock-in due to the poor transferability of LLM prompts. 

---
# TRACE: Reconstruction-Based Anomaly Detection in Ensemble and Time-Dependent Simulations 

**Authors**: Hamid Gadirov, Martijn Westra, Steffen Frey  

**Link**: [PDF](https://arxiv.org/pdf/2601.08659)  

**Abstract**: Detecting anomalies in high-dimensional, time-dependent simulation data is challenging due to complex spatial and temporal dynamics. We study reconstruction-based anomaly detection for ensemble data from parameterized Kármán vortex street simulations using convolutional autoencoders. We compare a 2D autoencoder operating on individual frames with a 3D autoencoder that processes short temporal stacks. The 2D model identifies localized spatial irregularities in single time steps, while the 3D model exploits spatio-temporal context to detect anomalous motion patterns and reduces redundant detections across time. We further evaluate volumetric time-dependent data and find that reconstruction errors are strongly influenced by the spatial distribution of mass, with highly concentrated regions yielding larger errors than dispersed configurations. Our results highlight the importance of temporal context for robust anomaly detection in dynamic simulations. 

---
# RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation 

**Authors**: Yihan Hong, Huaiyuan Yao, Bolin Shen, Wanpeng Xu, Hua Wei, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.08654)  

**Abstract**: The LLM-as-a-Judge paradigm promises scalable rubric-based evaluation, yet aligning frozen black-box models with human standards remains a challenge due to inherent generation stochasticity. We reframe judge alignment as a criteria transfer problem and isolate three recurrent failure modes: rubric instability caused by prompt sensitivity, unverifiable reasoning that lacks auditable evidence, and scale misalignment with human grading boundaries. To address these issues, we introduce RULERS (Rubric Unification, Locking, and Evidence-anchored Robust Scoring), a compiler-executor framework that transforms natural language rubrics into executable specifications. RULERS operates by compiling criteria into versioned immutable bundles, enforcing structured decoding with deterministic evidence verification, and applying lightweight Wasserstein-based post-hoc calibration, all without updating model parameters. Extensive experiments on essay and summarization benchmarks demonstrate that RULERS significantly outperforms representative baselines in human agreement, maintains strong stability against adversarial rubric perturbations, and enables smaller models to rival larger proprietary judges. Overall, our results suggest that reliable LLM judging requires executable rubrics, verifiable evidence, and calibrated scales rather than prompt phrasing alone. Code is available at this https URL. 

---
# Moral Lenses, Political Coordinates: Towards Ideological Positioning of Morally Conditioned LLMs 

**Authors**: Chenchen Yuan, Bolei Ma, Zheyu Zhang, Bardh Prenkaj, Frauke Kreuter, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2601.08634)  

**Abstract**: While recent research has systematically documented political orientation in large language models (LLMs), existing evaluations rely primarily on direct probing or demographic persona engineering to surface ideological biases. In social psychology, however, political ideology is also understood as a downstream consequence of fundamental moral intuitions. In this work, we investigate the causal relationship between moral values and political positioning by treating moral orientation as a controllable condition. Rather than simply assigning a demographic persona, we condition models to endorse or reject specific moral values and evaluate the resulting shifts on their political orientations, using the Political Compass Test. By treating moral values as lenses, we observe how moral conditioning actively steers model trajectories across economic and social dimensions. Our findings show that such conditioning induces pronounced, value-specific shifts in models' political coordinates. We further notice that these effects are systematically modulated by role framing and model scale, and are robust across alternative assessment instruments instantiating the same moral value. This highlights that effective alignment requires anchoring political assessments within the context of broader social values including morality, paving the way for more socially grounded alignment techniques. 

---
# M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting 

**Authors**: Yaohui Huang, Runmin Zou, Yun Wang, Laeeq Aslam, Ruipeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.08631)  

**Abstract**: Forecasting time series with extreme events is critical yet challenging due to their high variance, irregular dynamics, and sparse but high-impact nature. While existing methods excel in modeling dominant regular patterns, their performance degrades significantly during extreme events, constituting the primary source of forecasting errors in real-world applications. Although some approaches incorporate auxiliary signals to improve performance, they still fail to capture extreme events' complex temporal dynamics. To address these limitations, we propose M$^2$FMoE, an extreme-adaptive forecasting model that learns both regular and extreme patterns through multi-resolution and multi-view frequency modeling. It comprises three modules: (1) a multi-view frequency mixture-of-experts module assigns experts to distinct spectral bands in Fourier and Wavelet domains, with cross-view shared band splitter aligning frequency partitions and enabling inter-expert collaboration to capture both dominant and rare fluctuations; (2) a multi-resolution adaptive fusion module that hierarchically aggregates frequency features from coarse to fine resolutions, enhancing sensitivity to both short-term variations and sudden changes; (3) a temporal gating integration module that dynamically balances long-term trends and short-term frequency-aware features, improving adaptability to both regular and extreme temporal patterns. Experiments on real-world hydrological datasets with extreme patterns demonstrate that M$^2$FMoE outperforms state-of-the-art baselines without requiring extreme-event labels. 

---
# SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models 

**Authors**: Renyang Liu, Kangjie Chen, Han Qiu, Jie Zhang, Kwok-Yan Lam, Tianwei Zhang, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08623)  

**Abstract**: Image generation models (IGMs), while capable of producing impressive and creative content, often memorize a wide range of undesirable concepts from their training data, leading to the reproduction of unsafe content such as NSFW imagery and copyrighted artistic styles. Such behaviors pose persistent safety and compliance risks in real-world deployments and cannot be reliably mitigated by post-hoc filtering, owing to the limited robustness of such mechanisms and a lack of fine-grained semantic control. Recent unlearning methods seek to erase harmful concepts at the model level, which exhibit the limitations of requiring costly retraining, degrading the quality of benign generations, or failing to withstand prompt paraphrasing and adversarial attacks. To address these challenges, we introduce SafeRedir, a lightweight inference-time framework for robust unlearning via prompt embedding redirection. Without modifying the underlying IGMs, SafeRedir adaptively routes unsafe prompts toward safe semantic regions through token-level interventions in the embedding space. The framework comprises two core components: a latent-aware multi-modal safety classifier for identifying unsafe generation trajectories, and a token-level delta generator for precise semantic redirection, equipped with auxiliary predictors for token masking and adaptive scaling to localize and regulate the intervention. Empirical results across multiple representative unlearning tasks demonstrate that SafeRedir achieves effective unlearning capability, high semantic and perceptual preservation, robust image quality, and enhanced resistance to adversarial attacks. Furthermore, SafeRedir generalizes effectively across a variety of diffusion backbones and existing unlearned models, validating its plug-and-play compatibility and broad applicability. Code and data are available at this https URL. 

---
# VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking 

**Authors**: Mark Rothermel, Marcus Kornmann, Marcus Rohrbach, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2601.08611)  

**Abstract**: The growing scale of online misinformation urgently demands Automated Fact-Checking (AFC). Existing benchmarks for evaluating AFC systems, however, are largely limited in terms of task scope, modalities, domain, language diversity, realism, or coverage of misinformation types. Critically, they are static, thus subject to data leakage as their claims enter the pretraining corpora of LLMs. As a result, benchmark performance no longer reliably reflects the actual ability to verify claims. We introduce Verified Theses and Statements (VeriTaS), the first dynamic benchmark for multimodal AFC, designed to remain robust under ongoing large-scale pretraining of foundation models. VeriTaS currently comprises 24,000 real-world claims from 108 professional fact-checking organizations across 54 languages, covering textual and audiovisual content. Claims are added quarterly via a fully automated seven-stage pipeline that normalizes claim formulation, retrieves original media, and maps heterogeneous expert verdicts to a novel, standardized, and disentangled scoring scheme with textual justifications. Through human evaluation, we demonstrate that the automated annotations closely match human judgments. We commit to update VeriTaS in the future, establishing a leakage-resistant benchmark, supporting meaningful AFC evaluation in the era of rapidly evolving foundation models. We will make the code and data publicly available. 

---
# ExpSeek: Self-Triggered Experience Seeking for Web Agents 

**Authors**: Wenyuan Zhang, Xinghua Zhang, Haiyang Yu, Shuaiyi Nie, Bingli Wu, Juwei Yue, Tingwen Liu, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08605)  

**Abstract**: Experience intervention in web agents emerges as a promising technical paradigm, enhancing agent interaction capabilities by providing valuable insights from accumulated experiences. However, existing methods predominantly inject experience passively as global context before task execution, struggling to adapt to dynamically changing contextual observations during agent-environment interaction. We propose ExpSeek, which shifts experience toward step-level proactive seeking: (1) estimating step-level entropy thresholds to determine intervention timing using the model's intrinsic signals; (2) designing step-level tailor-designed experience content. Experiments on Qwen3-8B and 32B models across four challenging web agent benchmarks demonstrate that ExpSeek achieves absolute improvements of 9.3% and 7.5%, respectively. Our experiments validate the feasibility and advantages of entropy as a self-triggering signal, reveal that even a 4B small-scale experience model can significantly boost the performance of larger agent models. 

---
# WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation 

**Authors**: Zishan Shu, Juntong Wu, Wei Yan, Xudong Liu, Hongyu Zhang, Chang Liu, Youdong Mao, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08602)  

**Abstract**: Vision modeling has advanced rapidly with Transformers, whose attention mechanisms capture visual dependencies but lack a principled account of how semantic information propagates spatially. We revisit this problem from a wave-based perspective: feature maps are treated as spatial signals whose evolution over an internal propagation time (aligned with network depth) is governed by an underdamped wave equation. In this formulation, spatial frequency-from low-frequency global layout to high-frequency edges and textures-is modeled explicitly, and its interaction with propagation time is controlled rather than implicitly fixed. We derive a closed-form, frequency-time decoupled solution and implement it as the Wave Propagation Operator (WPO), a lightweight module that models global interactions in O(N log N) time-far lower than attention. Building on WPO, we propose a family of WaveFormer models as drop-in replacements for standard ViTs and CNNs, achieving competitive accuracy across image classification, object detection, and semantic segmentation, while delivering up to 1.6x higher throughput and 30% fewer FLOPs than attention-based alternatives. Furthermore, our results demonstrate that wave propagation introduces a complementary modeling bias to heat-based methods, effectively capturing both global coherence and high-frequency details essential for rich visual semantics. Codes are available at: this https URL. 

---
# Rewriting Video: Text-Driven Reauthoring of Video Footage 

**Authors**: Sitong Wang, Anh Truong, Lydia B. Chilton, Dingzeyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08565)  

**Abstract**: Video is a powerful medium for communication and storytelling, yet reauthoring existing footage remains challenging. Even simple edits often demand expertise, time, and careful planning, constraining how creators envision and shape their narratives. Recent advances in generative AI suggest a new paradigm: what if editing a video were as straightforward as rewriting text? To investigate this, we present a tech probe and a study on text-driven video reauthoring. Our approach involves two technical contributions: (1) a generative reconstruction algorithm that reverse-engineers video into an editable text prompt, and (2) an interactive probe, Rewrite Kit, that allows creators to manipulate these prompts. A technical evaluation of the algorithm reveals a critical human-AI perceptual gap. A probe study with 12 creators surfaced novel use cases such as virtual reshooting, synthetic continuity, and aesthetic restyling. It also highlighted key tensions around coherence, control, and creative alignment in this new paradigm. Our work contributes empirical insights into the opportunities and challenges of text-driven video reauthoring, offering design implications for future co-creative video tools. 

---
# VideoHEDGE: Entropy-Based Hallucination Detection for Video-VLMs via Semantic Clustering and Spatiotemporal Perturbations 

**Authors**: Sushant Gautam, Cise Midoglu, Vajira Thambawita, Michael A. Riegler, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08557)  

**Abstract**: Hallucinations in video-capable vision-language models (Video-VLMs) remain frequent and high-confidence, while existing uncertainty metrics often fail to align with correctness. We introduce VideoHEDGE, a modular framework for hallucination detection in video question answering that extends entropy-based reliability estimation from images to temporally structured inputs. Given a video-question pair, VideoHEDGE draws a baseline answer and multiple high-temperature generations from both clean clips and photometrically and spatiotemporally perturbed variants, then clusters the resulting textual outputs into semantic hypotheses using either Natural Language Inference (NLI)-based or embedding-based methods. Cluster-level probability masses yield three reliability scores: Semantic Entropy (SE), RadFlag, and Vision-Amplified Semantic Entropy (VASE). We evaluate VideoHEDGE on the SoccerChat benchmark using an LLM-as-a-judge to obtain binary hallucination labels. Across three 7B Video-VLMs (Qwen2-VL, Qwen2.5-VL, and a SoccerChat-finetuned model), VASE consistently achieves the highest ROC-AUC, especially at larger distortion budgets, while SE and RadFlag often operate near chance. We further show that embedding-based clustering matches NLI-based clustering in detection performance at substantially lower computational cost, and that domain fine-tuning reduces hallucination frequency but yields only modest improvements in calibration. The hedge-bench PyPI library enables reproducible and extensible benchmarking, with full code and experimental resources available at this https URL . 

---
# Contrastive and Multi-Task Learning on Noisy Brain Signals with Nonlinear Dynamical Signatures 

**Authors**: Sucheta Ghosh, Zahra Monfared, Felix Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2601.08549)  

**Abstract**: We introduce a two-stage multitask learning framework for analyzing Electroencephalography (EEG) signals that integrates denoising, dynamical modeling, and representation learning. In the first stage, a denoising autoencoder is trained to suppress artifacts and stabilize temporal dynamics, providing robust signal representations. In the second stage, a multitask architecture processes these denoised signals to achieve three objectives: motor imagery classification, chaotic versus non-chaotic regime discrimination using Lyapunov exponent-based labels, and self-supervised contrastive representation learning with NT-Xent loss. A convolutional backbone combined with a Transformer encoder captures spatial-temporal structure, while the dynamical task encourages sensitivity to nonlinear brain dynamics. This staged design mitigates interference between reconstruction and discriminative goals, improves stability across datasets, and supports reproducible training by clearly separating noise reduction from higher-level feature learning. Empirical studies show that our framework not only enhances robustness and generalization but also surpasses strong baselines and recent state-of-the-art methods in EEG decoding, highlighting the effectiveness of combining denoising, dynamical features, and self-supervised learning. 

---
# CD^2: Constrained Dataset Distillation for Few-Shot Class-Incremental Learning 

**Authors**: Kexin Bao, Daichi Zhang, Hansong Zhang, Yong Li, Yutao Yue, Shiming Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.08519)  

**Abstract**: Few-shot class-incremental learning (FSCIL) receives significant attention from the public to perform classification continuously with a few training samples, which suffers from the key catastrophic forgetting problem. Existing methods usually employ an external memory to store previous knowledge and treat it with incremental classes equally, which cannot properly preserve previous essential knowledge. To solve this problem and inspired by recent distillation works on knowledge transfer, we propose a framework termed \textbf{C}onstrained \textbf{D}ataset \textbf{D}istillation (\textbf{CD$^2$}) to facilitate FSCIL, which includes a dataset distillation module (\textbf{DDM}) and a distillation constraint module~(\textbf{DCM}). Specifically, the DDM synthesizes highly condensed samples guided by the classifier, forcing the model to learn compacted essential class-related clues from a few incremental samples. The DCM introduces a designed loss to constrain the previously learned class distribution, which can preserve distilled knowledge more sufficiently. Extensive experiments on three public datasets show the superiority of our method against other state-of-the-art competitors. 

---
# STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays 

**Authors**: Qiuyu Tian, Yiding Li, Fengyi Chen, Zequn Liu, Youyong Kong, Fan Guo, Yuyao Li, Jinjing Shen, Zhijing Xie, Yiyun Luo, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08510)  

**Abstract**: Movie screenplays are rich long-form narratives that interleave complex character relationships, temporally ordered events, and dialogue-driven interactions. While prior benchmarks target individual subtasks such as question answering or dialogue generation, they rarely evaluate whether models can construct a coherent story world and use it consistently across multiple forms of reasoning and generation. We introduce STAGE (Screenplay Text, Agents, Graphs and Evaluation), a unified benchmark for narrative understanding over full-length movie screenplays. STAGE defines four tasks: knowledge graph construction, scene-level event summarization, long-context screenplay question answering, and in-script character role-playing, all grounded in a shared narrative world representation. The benchmark provides cleaned scripts, curated knowledge graphs, and event- and character-centric annotations for 150 films across English and Chinese, enabling holistic evaluation of models' abilities to build world representations, abstract and verify narrative events, reason over long narratives, and generate character-consistent responses. 

---
# Temporal Fusion Nexus: A task-agnostic multi-modal embedding model for clinical narratives and irregular time series in post-kidney transplant care 

**Authors**: Aditya Kumar, Simon Rauch, Mario Cypko, Marcel Naik, Matthieu-P Schapranow, Aadil Rashid, Fabian Halleck, Bilgin Osmanodja, Roland Roller, Lars Pape, Klemens Budde, Mario Schiffer, Oliver Amft  

**Link**: [PDF](https://arxiv.org/pdf/2601.08503)  

**Abstract**: We introduce Temporal Fusion Nexus (TFN), a multi-modal and task-agnostic embedding model to integrate irregular time series and unstructured clinical narratives. We analysed TFN in post-kidney transplant (KTx) care, with a retrospective cohort of 3382 patients, on three key outcomes: graft loss, graft rejection, and mortality. Compared to state-of-the-art model in post KTx care, TFN achieved higher performance for graft loss (AUC 0.96 vs. 0.94) and graft rejection (AUC 0.84 vs. 0.74). In mortality prediction, TFN yielded an AUC of 0.86. TFN outperformed unimodal baselines (approx 10% AUC improvement over time series only baseline, approx 5% AUC improvement over time series with static patient data). Integrating clinical text improved performance across all tasks. Disentanglement metrics confirmed robust and interpretable latent factors in the embedding space, and SHAP-based attributions confirmed alignment with clinical reasoning. TFN has potential application in clinical tasks beyond KTx, where heterogeneous data sources, irregular longitudinal data, and rich narrative documentation are available. 

---
# EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers 

**Authors**: Wenwen Liao, Hang Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08499)  

**Abstract**: Large models such as Vision Transformers (ViTs) have demonstrated remarkable superiority over smaller architectures like ResNet in few-shot classification, owing to their powerful representational capacity. However, fine-tuning such large models demands extensive GPU memory and prolonged training time, making them impractical for many real-world low-resource scenarios. To bridge this gap, we propose EfficientFSL, a query-only fine-tuning framework tailored specifically for few-shot classification with ViT, which achieves competitive performance while significantly reducing computational overhead. EfficientFSL fully leverages the knowledge embedded in the pre-trained model and its strong comprehension ability, achieving high classification accuracy with an extremely small number of tunable parameters. Specifically, we introduce a lightweight trainable Forward Block to synthesize task-specific queries that extract informative features from the intermediate representations of the pre-trained model in a query-only manner. We further propose a Combine Block to fuse multi-layer outputs, enhancing the depth and robustness of feature representations. Finally, a Support-Query Attention Block mitigates distribution shift by adjusting prototypes to align with the query set distribution. With minimal trainable parameters, EfficientFSL achieves state-of-the-art performance on four in-domain few-shot datasets and six cross-domain datasets, demonstrating its effectiveness in real-world applications. 

---
# PKI: Prior Knowledge-Infused Neural Network for Few-Shot Class-Incremental Learning 

**Authors**: Kexin Baoa, Fanzhao Lin, Zichen Wang, Yong Li, Dan Zeng, Shiming Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.08493)  

**Abstract**: Few-shot class-incremental learning (FSCIL) aims to continually adapt a model on a limited number of new-class examples, facing two well-known challenges: catastrophic forgetting and overfitting to new classes. Existing methods tend to freeze more parts of network components and finetune others with an extra memory during incremental sessions. These methods emphasize preserving prior knowledge to ensure proficiency in recognizing old classes, thereby mitigating catastrophic forgetting. Meanwhile, constraining fewer parameters can help in overcoming overfitting with the assistance of prior knowledge. Following previous methods, we retain more prior knowledge and propose a prior knowledge-infused neural network (PKI) to facilitate FSCIL. PKI consists of a backbone, an ensemble of projectors, a classifier, and an extra memory. In each incremental session, we build a new projector and add it to the ensemble. Subsequently, we finetune the new projector and the classifier jointly with other frozen network components, ensuring the rich prior knowledge is utilized effectively. By cascading projectors, PKI integrates prior knowledge accumulated from previous sessions and learns new knowledge flexibly, which helps to recognize old classes and efficiently learn new classes. Further, to reduce the resource consumption associated with keeping many projectors, we design two variants of the prior knowledge-infused neural network (PKIV-1 and PKIV-2) to trade off a balance between resource consumption and performance by reducing the number of projectors. Extensive experiments on three popular benchmarks demonstrate that our approach outperforms state-of-the-art methods. 

---
# BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts 

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid  

**Link**: [PDF](https://arxiv.org/pdf/2601.08490)  

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance. 

---
# sui-1: Grounded and Verifiable Long-Form Summarization 

**Authors**: Benedikt Droste, Jan Philipp Harries, Maximilian Idahl, Björn Plüster  

**Link**: [PDF](https://arxiv.org/pdf/2601.08472)  

**Abstract**: Large language models frequently generate plausible but unfaithful summaries that users cannot verify against source text, a critical limitation in compliance-sensitive domains such as government and legal analysis. We present sui-1, a 24B parameter model that produces abstractive summaries with inline citations, enabling users to trace each claim to its source sentence. Our synthetic data pipeline combines chain-of-thought prompting with multi-stage verification, generating over 22,000 high-quality training examples across five languages from diverse sources including parliamentary documents, web text, and Wikipedia. Evaluation shows sui-1 significantly outperforms all tested open-weight baselines, including models with 3x more parameters. These results demonstrate that task-specific training substantially outperforms scale alone for citation-grounded summarization. Model weights and an interactive demo are publicly available. 

---
# JudgeRLVR: Judge First, Generate Second for Efficient Reasoning 

**Authors**: Jiangshan Duo, Hanyu Li, Hailin Zhang, Yudong Wang, Sujian Li, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08468)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a standard paradigm for reasoning in Large Language Models. However, optimizing solely for final-answer correctness often drives models into aimless, verbose exploration, where they rely on exhaustive trial-and-error tactics rather than structured planning to reach solutions. While heuristic constraints like length penalties can reduce verbosity, they often truncate essential reasoning steps, creating a difficult trade-off between efficiency and verification. In this paper, we argue that discriminative capability is a prerequisite for efficient generation: by learning to distinguish valid solutions, a model can internalize a guidance signal that prunes the search space. We propose JudgeRLVR, a two-stage judge-then-generate paradigm. In the first stage, we train the model to judge solution responses with verifiable answers. In the second stage, we fine-tune the same model with vanilla generating RLVR initialized from the judge. Compared to Vanilla RLVR using the same math-domain training data, JudgeRLVR achieves a better quality--efficiency trade-off for Qwen3-30B-A3B: on in-domain math, it delivers about +3.7 points average accuracy gain with -42\% average generation length; on out-of-domain benchmarks, it delivers about +4.5 points average accuracy improvement, demonstrating enhanced generalization. 

---
# CoMa: Contextual Massing Generation with Vision-Language Models 

**Authors**: Evgenii Maslov, Valentin Khrulkov, Anastasia Volkova, Anton Gusarov, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2601.08464)  

**Abstract**: The conceptual design phase in architecture and urban planning, particularly building massing, is complex and heavily reliant on designer intuition and manual effort. To address this, we propose an automated framework for generating building massing based on functional requirements and site context. A primary obstacle to such data-driven methods has been the lack of suitable datasets. Consequently, we introduce the CoMa-20K dataset, a comprehensive collection that includes detailed massing geometries, associated economical and programmatic data, and visual representations of the development site within its existing urban context. We benchmark this dataset by formulating massing generation as a conditional task for Vision-Language Models (VLMs), evaluating both fine-tuned and large zero-shot models. Our experiments reveal the inherent complexity of the task while demonstrating the potential of VLMs to produce context-sensitive massing options. The dataset and analysis establish a foundational benchmark and highlight significant opportunities for future research in data-driven architectural design. 

---
# A Formal Proof of a Continued Fraction Conjecture for $π$ Originating from the Ramanujan Machine 

**Authors**: Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08461)  

**Abstract**: We provide a formal analytic proof for a class of non-canonical polynomial continued fractions representing {\pi}/4, originally conjectured by the Ramanujan Machine using algorithmic induction [4]. By establishing an explicit correspondence with the ratio of contiguous Gaussian hypergeometric functions 2F1(a, b; c; z), we show that these identities can be derived via a discrete sequence of equivalence transformations. We further prove that the conjectured integer coefficients constitute a symbolically minimal realization of the underlying analytic kernel. Stability analysis confirms that the resulting limit-periodic structures reside strictly within the Worpitzky convergence disk, ensuring absolute convergence. This work demonstrates that such algorithmically discovered identities are not isolated numerical artifacts, but are deeply rooted in the classical theory of hypergeometric transformations. 

---
# Decoding Order Matters in Autoregressive Speech Synthesis 

**Authors**: Minghui Zhao, Anton Ragni  

**Link**: [PDF](https://arxiv.org/pdf/2601.08450)  

**Abstract**: Autoregressive speech synthesis often adopts a left-to-right order, yet generation order is a modelling choice. We investigate decoding order through masked diffusion framework, which progressively unmasks positions and allows arbitrary decoding orders during training and inference. By interpolating between identity and random permutations, we show that randomness in decoding order affects speech quality. We further compare fixed strategies, such as \texttt{l2r} and \texttt{r2l} with adaptive ones, such as Top-$K$, finding that fixed-order decoding, including the dominating left-to-right approach, is suboptimal, while adaptive decoding yields better performance. Finally, since masked diffusion requires discrete inputs, we quantise acoustic representations and find that even 1-bit quantisation can support reasonably high-quality speech. 

---
# Divide and Conquer: Static-Dynamic Collaboration for Few-Shot Class-Incremental Learning 

**Authors**: Kexin Bao, Daichi Zhang, Yong Li, Dan Zeng, Shiming Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.08448)  

**Abstract**: Few-shot class-incremental learning (FSCIL) aims to continuously recognize novel classes under limited data, which suffers from the key stability-plasticity dilemma: balancing the retention of old knowledge with the acquisition of new knowledge. To address this issue, we divide the task into two different stages and propose a framework termed Static-Dynamic Collaboration (SDC) to achieve a better trade-off between stability and plasticity. Specifically, our method divides the normal pipeline of FSCIL into Static Retaining Stage (SRS) and Dynamic Learning Stage (DLS), which harnesses old static and incremental dynamic class information, respectively. During SRS, we train an initial model with sufficient data in the base session and preserve the key part as static memory to retain fundamental old knowledge. During DLS, we introduce an extra dynamic projector jointly trained with the previous static memory. By employing both stages, our method achieves improved retention of old knowledge while continuously adapting to new classes. Extensive experiments on three public benchmarks and a real-world application dataset demonstrate that our method achieves state-of-the-art performance against other competitors. 

---
# Large Multimodal Models for Embodied Intelligent Driving: The Next Frontier in Self-Driving? 

**Authors**: Long Zhang, Yuchen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2601.08434)  

**Abstract**: The advent of Large Multimodal Models (LMMs) offers a promising technology to tackle the limitations of modular design in autonomous driving, which often falters in open-world scenarios requiring sustained environmental understanding and logical reasoning. Besides, embodied artificial intelligence facilitates policy optimization through closed-loop interactions to achieve the continuous learning capability, thereby advancing autonomous driving toward embodied intelligent (El) driving. However, such capability will be constrained by relying solely on LMMs to enhance EI driving without joint decision-making. This article introduces a novel semantics and policy dual-driven hybrid decision framework to tackle this challenge, ensuring continuous learning and joint decision. The framework merges LMMs for semantic understanding and cognitive representation, and deep reinforcement learning (DRL) for real-time policy optimization. We starts by introducing the foundational principles of EI driving and LMMs. Moreover, we examine the emerging opportunities this framework enables, encompassing potential benefits and representative use cases. A case study is conducted experimentally to validate the performance superiority of our framework in completing lane-change planning task. Finally, several future research directions to empower EI driving are identified to guide subsequent work. 

---
# Taxon: Hierarchical Tax Code Prediction with Semantically Aligned LLM Expert Guidance 

**Authors**: Jihang Li, Qing Liu, Zulong Chen, Jing Wang, Wei Wang, Chuanfei Xu, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08418)  

**Abstract**: Tax code prediction is a crucial yet underexplored task in automating invoicing and compliance management for large-scale e-commerce platforms. Each product must be accurately mapped to a node within a multi-level taxonomic hierarchy defined by national standards, where errors lead to financial inconsistencies and regulatory risks. This paper presents Taxon, a semantically aligned and expert-guided framework for hierarchical tax code prediction. Taxon integrates (i) a feature-gating mixture-of-experts architecture that adaptively routes multi-modal features across taxonomy levels, and (ii) a semantic consistency model distilled from large language models acting as domain experts to verify alignment between product titles and official tax definitions. To address noisy supervision in real business records, we design a multi-source training pipeline that combines curated tax databases, invoice validation logs, and merchant registration data to provide both structural and semantic supervision. Extensive experiments on the proprietary TaxCode dataset and public benchmarks demonstrate that Taxon achieves state-of-the-art performance, outperforming strong baselines. Further, an additional full hierarchical paths reconstruction procedure significantly improves structural consistency, yielding the highest overall F1 scores. Taxon has been deployed in production within Alibaba's tax service system, handling an average of over 500,000 tax code queries per day and reaching peak volumes above five million requests during business event with improved accuracy, interpretability, and robustness. 

---
# Regulatory gray areas of LLM Terms 

**Authors**: Brittany I. Davidson, Kate Muir, Florian A.D. Burnat, Adam N. Joinson  

**Link**: [PDF](https://arxiv.org/pdf/2601.08415)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into academic research pipelines; however, the Terms of Service governing their use remain under-examined. We present a comparative analysis of the Terms of Service of five major LLM providers (Anthropic, DeepSeek, Google, OpenAI, and xAI) collected in November 2025. Our analysis reveals substantial variation in the stringency and specificity of usage restrictions for general users and researchers. We identify specific complexities for researchers in security research, computational social sciences, and psychological studies. We identify `regulatory gray areas' where Terms of Service create uncertainty for legitimate use. We contribute a publicly available resource comparing terms across platforms (OSF) and discuss implications for general users and researchers navigating this evolving landscape. 

---
# PATS: Personality-Aware Teaching Strategies with Large Language Model Tutors 

**Authors**: Donya Rooein, Sankalan Pal Chowdhury, Mariia Eremeeva, Yuan Qin, Debora Nozza, Mrinmaya Sachan, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2601.08402)  

**Abstract**: Recent advances in large language models (LLMs) demonstrate their potential as educational tutors. However, different tutoring strategies benefit different student personalities, and mismatches can be counterproductive to student outcomes. Despite this, current LLM tutoring systems do not take into account student personality traits. To address this problem, we first construct a taxonomy that links pedagogical methods to personality profiles, based on pedagogical literature. We simulate student-teacher conversations and use our framework to let the LLM tutor adjust its strategy to the simulated student personality. We evaluate the scenario with human teachers and find that they consistently prefer our approach over two baselines. Our method also increases the use of less common, high-impact strategies such as role-playing, which human and LLM annotators prefer significantly. Our findings pave the way for developing more personalized and effective LLM use in educational applications. 

---
# An Explainable Two Stage Deep Learning Framework for Pericoronitis Assessment in Panoramic Radiographs Using YOLOv8 and ResNet-50 

**Authors**: Ajo Babu George, Pranav S, Kunal Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2601.08401)  

**Abstract**: Objectives: To overcome challenges in diagnosing pericoronitis on panoramic radiographs, an AI-assisted assessment system integrating anatomical localization, pathological classification, and interpretability. Methods: A two-stage deep learning pipeline was implemented. The first stage used YOLOv8 to detect third molars and classify their anatomical positions and angulations based on Winter's classification. Detected regions were then fed into a second-stage classifier, a modified ResNet-50 architecture, for detecting radiographic features suggestive of pericoronitis. To enhance clinical trust, Grad-CAM was used to highlight key diagnostic regions on the radiographs. Results: The YOLOv8 component achieved 92% precision and 92.5% mean average precision. The ResNet-50 classifier yielded F1-scores of 88% for normal cases and 86% for pericoronitis. Radiologists reported 84% alignment between Grad-CAM and their diagnostic impressions, supporting the radiographic relevance of the interpretability output. Conclusion: The system shows strong potential for AI-assisted panoramic assessment, with explainable AI features that support clinical confidence. 

---
# Controlled LLM Training on Spectral Sphere 

**Authors**: Tian Xie, Haoming Luo, Haoyu Tang, Yiwen Hu, Jason Klein Liu, Qingnan Ren, Yang Wang, Wayne Xin Zhao, Rui Yan, Bing Su, Chong Luo, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.08393)  

**Abstract**: Scaling large models requires optimization strategies that ensure rapid convergence grounded in stability. Maximal Update Parametrization ($\boldsymbol{\mu}$P) provides a theoretical safeguard for width-invariant $\Theta(1)$ activation control, whereas emerging optimizers like Muon are only ``half-aligned'' with these constraints: they control updates but allow weights to drift. To address this limitation, we introduce the \textbf{Spectral Sphere Optimizer (SSO)}, which enforces strict module-wise spectral constraints on both weights and their updates. By deriving the steepest descent direction on the spectral sphere, SSO realizes a fully $\boldsymbol{\mu}$P-aligned optimization process. To enable large-scale training, we implement SSO as an efficient parallel algorithm within Megatron. Through extensive pretraining on diverse architectures, including Dense 1.7B, MoE 8B-A1B, and 200-layer DeepNet models, SSO consistently outperforms AdamW and Muon. Furthermore, we observe significant practical stability benefits, including improved MoE router load balancing, suppressed outliers, and strictly bounded activations. 

---
# Training-Free Distribution Adaptation for Diffusion Models via Maximum Mean Discrepancy Guidance 

**Authors**: Matina Mahdizadeh Sani, Nima Jamali, Mohammad Jalali, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2601.08379)  

**Abstract**: Pre-trained diffusion models have emerged as powerful generative priors for both unconditional and conditional sample generation, yet their outputs often deviate from the characteristics of user-specific target data. Such mismatches are especially problematic in domain adaptation tasks, where only a few reference examples are available and retraining the diffusion model is infeasible. Existing inference-time guidance methods can adjust sampling trajectories, but they typically optimize surrogate objectives such as classifier likelihoods rather than directly aligning with the target distribution. We propose MMD Guidance, a training-free mechanism that augments the reverse diffusion process with gradients of the Maximum Mean Discrepancy (MMD) between generated samples and a reference dataset. MMD provides reliable distributional estimates from limited data, exhibits low variance in practice, and is efficiently differentiable, which makes it particularly well-suited for the guidance task. Our framework naturally extends to prompt-aware adaptation in conditional generation models via product kernels. Also, it can be applied with computational efficiency in latent diffusion models (LDMs), since guidance is applied in the latent space of the LDM. Experiments on synthetic and real-world benchmarks demonstrate that MMD Guidance can achieve distributional alignment while preserving sample fidelity. 

---
# Geo-NVS-w: Geometry-Aware Novel View Synthesis In-the-Wild with an SDF Renderer 

**Authors**: Anastasios Tsalakopoulos, Angelos Kanlis, Evangelos Chatzis, Antonis Karakottas, Dimitrios Zarpalas  

**Link**: [PDF](https://arxiv.org/pdf/2601.08371)  

**Abstract**: We introduce Geo-NVS-w, a geometry-aware framework for high-fidelity novel view synthesis from unstructured, in-the-wild image collections. While existing in-the-wild methods already excel at novel view synthesis, they often lack geometric grounding on complex surfaces, sometimes producing results that contain inconsistencies. Geo-NVS-w addresses this limitation by leveraging an underlying geometric representation based on a Signed Distance Function (SDF) to guide the rendering process. This is complemented by a novel Geometry-Preservation Loss which ensures that fine structural details are preserved. Our framework achieves competitive rendering performance, while demonstrating a 4-5x reduction reduction in energy consumption compared to similar methods. We demonstrate that Geo-NVS-w is a robust method for in-the-wild NVS, yielding photorealistic results with sharp, geometrically coherent details. 

---
# Scalable Sequential Recommendation under Latency and Memory Constraints 

**Authors**: Adithya Parthasarathy, Aswathnarayan Muthukrishnan Kirubakaran, Vinoth Punniyamoorthy, Nachiappan Chockalingam, Lokesh Butra, Kabilan Kannan, Abhirup Mazumder, Sumit Saha  

**Link**: [PDF](https://arxiv.org/pdf/2601.08360)  

**Abstract**: Sequential recommender systems must model long-range user behavior while operating under strict memory and latency constraints. Transformer-based approaches achieve strong accuracy but suffer from quadratic attention complexity, forcing aggressive truncation of user histories and limiting their practicality for long-horizon modeling. This paper presents HoloMambaRec, a lightweight sequential recommendation architecture that combines holographic reduced representations for attribute-aware embedding with a selective state space encoder for linear-time sequence processing. Item and attribute information are bound using circular convolution, preserving embedding dimensionality while encoding structured metadata. A shallow selective state space backbone, inspired by recent Mamba-style models, enables efficient training and constant-time recurrent inference. Experiments on Amazon Beauty and MovieLens-1M datasets demonstrate that HoloMambaRec consistently outperforms SASRec and achieves competitive performance with GRU4Rec under a constrained 10-epoch training budget, while maintaining substantially lower memory complexity. The design further incorporates forward-compatible mechanisms for temporal bundling and inference-time compression, positioning HoloMambaRec as a practical and extensible alternative for scalable, metadata-aware sequential recommendation. 

---
# IGAN: A New Inception-based Model for Stable and High-Fidelity Image Synthesis Using Generative Adversarial Networks 

**Authors**: Ahmed A. Hashim, Ali Al-Shuwaili, Asraa Saeed, Ali Al-Bayaty  

**Link**: [PDF](https://arxiv.org/pdf/2601.08332)  

**Abstract**: Generative Adversarial Networks (GANs) face a significant challenge of striking an optimal balance between high-quality image generation and training stability. Recent techniques, such as DCGAN, BigGAN, and StyleGAN, improve visual fidelity; however, such techniques usually struggle with mode collapse and unstable gradients at high network depth. This paper proposes a novel GAN structural model that incorporates deeper inception-inspired convolution and dilated convolution. This novel model is termed the Inception Generative Adversarial Network (IGAN). The IGAN model generates high-quality synthetic images while maintaining training stability, by reducing mode collapse as well as preventing vanishing and exploding gradients. Our proposed IGAN model achieves the Frechet Inception Distance (FID) of 13.12 and 15.08 on the CUB-200 and ImageNet datasets, respectively, representing a 28-33% improvement in FID over the state-of-the-art GANs. Additionally, the IGAN model attains an Inception Score (IS) of 9.27 and 68.25, reflecting improved image diversity and generation quality. Finally, the two techniques of dropout and spectral normalization are utilized in both the generator and discriminator structures to further mitigate gradient explosion and overfitting. These findings confirm that the IGAN model potentially balances training stability with image generation quality, constituting a scalable and computationally efficient framework for high-fidelity image synthesis. 

---
# Safe Heterogeneous Multi-Agent RL with Communication Regularization for Coordinated Target Acquisition 

**Authors**: Gabriele Calzolari, Vidya Sumathy, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2601.08327)  

**Abstract**: This paper introduces a decentralized multi-agent reinforcement learning framework enabling structurally heterogeneous teams of agents to jointly discover and acquire randomly located targets in environments characterized by partial observability, communication constraints, and dynamic interactions. Each agent's policy is trained with the Multi-Agent Proximal Policy Optimization algorithm and employs a Graph Attention Network encoder that integrates simulated range-sensing data with communication embeddings exchanged among neighboring agents, enabling context-aware decision-making from both local sensing and relational information. In particular, this work introduces a unified framework that integrates graph-based communication and trajectory-aware safety through safety filters. The architecture is supported by a structured reward formulation designed to encourage effective target discovery and acquisition, collision avoidance, and de-correlation between the agents' communication vectors by promoting informational orthogonality. The effectiveness of the proposed reward function is demonstrated through a comprehensive ablation study. Moreover, simulation results demonstrate safe and stable task execution, confirming the framework's effectiveness. 

---
# Enhancing Image Quality Assessment Ability of LMMs via Retrieval-Augmented Generation 

**Authors**: Kang Fu, Huiyu Duan, Zicheng Zhang, Yucheng Zhu, Jun Zhao, Xiongkuo Min, Jia Wang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2601.08311)  

**Abstract**: Large Multimodal Models (LMMs) have recently shown remarkable promise in low-level visual perception tasks, particularly in Image Quality Assessment (IQA), demonstrating strong zero-shot capability. However, achieving state-of-the-art performance often requires computationally expensive fine-tuning methods, which aim to align the distribution of quality-related token in output with image quality levels. Inspired by recent training-free works for LMM, we introduce IQARAG, a novel, training-free framework that enhances LMMs' IQA ability. IQARAG leverages Retrieval-Augmented Generation (RAG) to retrieve some semantically similar but quality-variant reference images with corresponding Mean Opinion Scores (MOSs) for input image. These retrieved images and input image are integrated into a specific prompt. Retrieved images provide the LMM with a visual perception anchor for IQA task. IQARAG contains three key phases: Retrieval Feature Extraction, Image Retrieval, and Integration & Quality Score Generation. Extensive experiments across multiple diverse IQA datasets, including KADID, KonIQ, LIVE Challenge, and SPAQ, demonstrate that the proposed IQARAG effectively boosts the IQA performance of LMMs, offering a resource-efficient alternative to fine-tuning for quality assessment. 

---
# ORBIT: On-policy Exploration-Exploitation for Controllable Multi-Budget Reasoning 

**Authors**: Kun Liang, Clive Bai, Xin Xu, Chenming Tang, Sanwoo Lee, Weijie Liu, Saiyong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08310)  

**Abstract**: Recent Large Reasoning Models (LRMs) achieve strong performance by leveraging long-form Chain-of-Thought (CoT) reasoning, but uniformly applying overlong reasoning at inference time incurs substantial and often unnecessary computational cost. To address this, prior work explores various strategies to infer an appropriate reasoning budget from the input. However, such approaches are unreliable in the worst case, as estimating the minimal required reasoning effort is fundamentally difficult, and they implicitly fix the trade-off between reasoning cost and accuracy during training, limiting flexibility under varying deployment scenarios. Motivated by these limitations, we propose ORBIT, a controllable multi-budget reasoning framework with well-separated reasoning modes triggered by input. ORBIT employs multi-stage reinforcement learning to discover Pareto-optimal reasoning behaviors at each effort, followed by on-policy distillation to fuse these behaviors into a single unified model. Experiments show that ORBIT achieves (1) controllable reasoning behavior over multiple modes, (2) competitive reasoning density within each mode, and (3) integration of these frontier policies into a single unified student model while preserving clear mode separation and high per-mode performance. 

---
# Enhancing Sentiment Classification and Irony Detection in Large Language Models through Advanced Prompt Engineering Techniques 

**Authors**: Marvin Schmitt, Anne Schwerk, Sebastian Lempert  

**Link**: [PDF](https://arxiv.org/pdf/2601.08302)  

**Abstract**: This study investigates the use of prompt engineering to enhance large language models (LLMs), specifically GPT-4o-mini and gemini-1.5-flash, in sentiment analysis tasks. It evaluates advanced prompting techniques like few-shot learning, chain-of-thought prompting, and self-consistency against a baseline. Key tasks include sentiment classification, aspect-based sentiment analysis, and detecting subtle nuances such as irony. The research details the theoretical background, datasets, and methods used, assessing performance of LLMs as measured by accuracy, recall, precision, and F1 score. Findings reveal that advanced prompting significantly improves sentiment analysis, with the few-shot approach excelling in GPT-4o-mini and chain-of-thought prompting boosting irony detection in gemini-1.5-flash by up to 46%. Thus, while advanced prompting techniques overall improve performance, the fact that few-shot prompting works best for GPT-4o-mini and chain-of-thought excels in gemini-1.5-flash for irony detection suggests that prompting strategies must be tailored to both the model and the task. This highlights the importance of aligning prompt design with both the LLM's architecture and the semantic complexity of the task. 

---
# Demystifying the Slash Pattern in Attention: The Role of RoPE 

**Authors**: Yuan Cheng, Fengzhuo Zhang, Yunlong Hou, Cunxiao Du, Chao Du, Tianyu Pang, Aixin Sun, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08297)  

**Abstract**: Large Language Models (LLMs) often exhibit slash attention patterns, where attention scores concentrate along the $\Delta$-th sub-diagonal for some offset $\Delta$. These patterns play a key role in passing information across tokens. But why do they emerge? In this paper, we demystify the emergence of these Slash-Dominant Heads (SDHs) from both empirical and theoretical perspectives. First, by analyzing open-source LLMs, we find that SDHs are intrinsic to models and generalize to out-of-distribution prompts. To explain the intrinsic emergence, we analyze the queries, keys, and Rotary Position Embedding (RoPE), which jointly determine attention scores. Our empirical analysis reveals two characteristic conditions of SDHs: (1) Queries and keys are almost rank-one, and (2) RoPE is dominated by medium- and high-frequency components. Under these conditions, queries and keys are nearly identical across tokens, and interactions between medium- and high-frequency components of RoPE give rise to SDHs. Beyond empirical evidence, we theoretically show that these conditions are sufficient to ensure the emergence of SDHs by formalizing them as our modeling assumptions. Particularly, we analyze the training dynamics of a shallow Transformer equipped with RoPE under these conditions, and prove that models trained via gradient descent exhibit SDHs. The SDHs generalize to out-of-distribution prompts. 

---
# HIPPO: Accelerating Video Large Language Models Inference via Holistic-aware Parallel Speculative Decoding 

**Authors**: Qitan Lv, Tianyu Liu, Wen Wu, Xuenan Xu, Bowen Zhou, Feng Wu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08273)  

**Abstract**: Speculative decoding (SD) has emerged as a promising approach to accelerate LLM inference without sacrificing output quality. Existing SD methods tailored for video-LLMs primarily focus on pruning redundant visual tokens to mitigate the computational burden of massive visual inputs. However, existing methods do not achieve inference acceleration comparable to text-only LLMs. We observe from extensive experiments that this phenomenon mainly stems from two limitations: (i) their pruning strategies inadequately preserve visual semantic tokens, degrading draft quality and acceptance rates; (ii) even with aggressive pruning (e.g., 90% visual tokens removed), the draft model's remaining inference cost limits overall speedup. To address these limitations, we propose HIPPO, a general holistic-aware parallel speculative decoding framework. Specifically, HIPPO proposes (i) a semantic-aware token preservation method, which fuses global attention scores with local visual semantics to retain semantic information at high pruning ratios; (ii) a video parallel SD algorithm that decouples and overlaps draft generation and target verification phases. Experiments on four video-LLMs across six benchmarks demonstrate HIPPO's effectiveness, yielding up to 3.51x speedup compared to vanilla auto-regressive decoding. 

---
# On Evaluation of Unsupervised Feature Selection for Pattern Classification 

**Authors**: Gyu-Il Kim, Dae-Won Kim, Jaesung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.08257)  

**Abstract**: Unsupervised feature selection aims to identify a compact subset of features that captures the intrinsic structure of data without supervised label. Most existing studies evaluate the performance of methods using the single-label dataset that can be instantiated by selecting a label from multi-label data while maintaining the original features. Because the chosen label can vary arbitrarily depending on the experimental setting, the superiority among compared methods can be changed with regard to which label happens to be selected. Thus, evaluating unsupervised feature selection methods based solely on single-label accuracy is unreasonable for assessing their true discriminative ability. This study revisits this evaluation paradigm by adopting a multi-label classification framework. Experiments on 21 multi-label datasets using several representative methods demonstrate that performance rankings differ markedly from those reported under single-label settings, suggesting the possibility of multi-label evaluation settings for fair and reliable comparison of unsupervised feature selection methods. 

---
# Hyperbolic Heterogeneous Graph Transformer 

**Authors**: Jongmin Park, Seunghoon Han, Hyewon Lee, Won-Yong Shin, Sungsu Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.08251)  

**Abstract**: In heterogeneous graphs, we can observe complex structures such as tree-like or hierarchical structures. Recently, the hyperbolic space has been widely adopted in many studies to effectively learn these complex structures. Although these methods have demonstrated the advantages of the hyperbolic space in learning heterogeneous graphs, most existing methods still have several challenges. They rely heavily on tangent-space operations, which often lead to mapping distortions during frequent transitions. Moreover, their message-passing architectures mainly focus on local neighborhood information, making it difficult to capture global hierarchical structures and long-range dependencies between different types of nodes. To address these limitations, we propose Hyperbolic Heterogeneous Graph Transformer (HypHGT), which effectively and efficiently learns heterogeneous graph representations entirely within the hyperbolic space. Unlike previous message-passing based hyperbolic heterogeneous GNNs, HypHGT naturally captures both local and global dependencies through transformer-based architecture. Furthermore, the proposed relation-specific hyperbolic attention mechanism in HypHGT, which operates with linear time complexity, enables efficient computation while preserving the heterogeneous information across different relation types. This design allows HypHGT to effectively capture the complex structural properties and semantic information inherent in heterogeneous graphs. We conduct comprehensive experiments to evaluate the effectiveness and efficiency of HypHGT, and the results demonstrate that it consistently outperforms state-of-the-art methods in node classification task, with significantly reduced training time and memory usage. 

---
# GADPN: Graph Adaptive Denoising and Perturbation Networks via Singular Value Decomposition 

**Authors**: Hao Deng, Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08230)  

**Abstract**: While Graph Neural Networks (GNNs) excel on graph-structured data, their performance is fundamentally limited by the quality of the observed graph, which often contains noise, missing links, or structural properties misaligned with GNNs' underlying assumptions. To address this, graph structure learning aims to infer a more optimal topology. Existing methods, however, often incur high computational costs due to complex generative models and iterative joint optimization, limiting their practical utility. In this paper, we propose GADPN, a simple yet effective graph structure learning framework that adaptively refines graph topology via low-rank denoising and generalized structural perturbation. Our approach makes two key contributions: (1) we introduce Bayesian optimization to adaptively determine the optimal denoising strength, tailoring the process to each graph's homophily level; and (2) we extend the structural perturbation method to arbitrary graphs via Singular Value Decomposition (SVD), overcoming its original limitation to symmetric structures. Extensive experiments on benchmark datasets demonstrate that GADPN achieves state-of-the-art performance while significantly improving efficiency. It shows particularly strong gains on challenging disassortative graphs, validating its ability to robustly learn enhanced graph structures across diverse network types. 

---
# Knowledge-based learning in Text-RAG and Image-RAG 

**Authors**: Alexander Shim, Khalil Saieh, Samuel Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2601.08226)  

**Abstract**: This research analyzed and compared the multi-modal approach in the Vision Transformer(EVA-ViT) based image encoder with the LlaMA or ChatGPT LLM to reduce the hallucination problem and detect diseases in chest x-ray images. In this research, we utilized the NIH Chest X-ray image to train the model and compared it in image-based RAG, text-based RAG, and baseline. [3] [5] In a result, the text-based RAG[2] e!ectively reduces the hallucination problem by using external knowledge information, and the image-based RAG improved the prediction con"dence and calibration by using the KNN methods. [4] Moreover, the GPT LLM showed better performance, a low hallucination rate, and better Expected Calibration Error(ECE) than Llama Llama-based model. This research shows the challenge of data imbalance, a complex multi-stage structure, but suggests a large experience environment and a balanced example of use. 

---
# DNF: Dual-Layer Nested Fingerprinting for Large Language Model Intellectual Property Protection 

**Authors**: Zhenhua Xu, Yiran Zhao, Mengting Zhong, Dezhang Kong, Changting Lin, Tong Qiao, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.08223)  

**Abstract**: The rapid growth of large language models raises pressing concerns about intellectual property protection under black-box deployment. Existing backdoor-based fingerprints either rely on rare tokens -- leading to high-perplexity inputs susceptible to filtering -- or use fixed trigger-response mappings that are brittle to leakage and post-hoc adaptation. We propose \textsc{Dual-Layer Nested Fingerprinting} (DNF), a black-box method that embeds a hierarchical backdoor by coupling domain-specific stylistic cues with implicit semantic triggers. Across Mistral-7B, LLaMA-3-8B-Instruct, and Falcon3-7B-Instruct, DNF achieves perfect fingerprint activation while preserving downstream utility. Compared with existing methods, it uses lower-perplexity triggers, remains undetectable under fingerprint detection attacks, and is relatively robust to incremental fine-tuning and model merging. These results position DNF as a practical, stealthy, and resilient solution for LLM ownership verification and intellectual property protection. 

---
# Evaluating Implicit Regulatory Compliance in LLM Tool Invocation via Logic-Guided Synthesis 

**Authors**: Da Song, Yuheng Huang, Boqi Chen, Tianshuo Cong, Randy Goebel, Lei Ma, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2601.08196)  

**Abstract**: The integration of large language models (LLMs) into autonomous agents has enabled complex tool use, yet in high-stakes domains, these systems must strictly adhere to regulatory standards beyond simple functional correctness. However, existing benchmarks often overlook implicit regulatory compliance, thus failing to evaluate whether LLMs can autonomously enforce mandatory safety constraints. To fill this gap, we introduce LogiSafetyGen, a framework that converts unstructured regulations into Linear Temporal Logic oracles and employs logic-guided fuzzing to synthesize valid, safety-critical traces. Building on this framework, we construct LogiSafetyBench, a benchmark comprising 240 human-verified tasks that require LLMs to generate Python programs that satisfy both functional objectives and latent compliance rules. Evaluations of 13 state-of-the-art (SOTA) LLMs reveal that larger models, despite achieving better functional correctness, frequently prioritize task completion over safety, which results in non-compliant behavior. 

---
# ForgetMark: Stealthy Fingerprint Embedding via Targeted Unlearning in Language Models 

**Authors**: Zhenhua Xu, Haobo Zhang, Zhebo Wang, Qichen Liu, Haitao Xu, Wenpeng Xing, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.08189)  

**Abstract**: Existing invasive (backdoor) fingerprints suffer from high-perplexity triggers that are easily filtered, fixed response patterns exposed by heuristic detectors, and spurious activations on benign inputs. We introduce \textsc{ForgetMark}, a stealthy fingerprinting framework that encodes provenance via targeted unlearning. It builds a compact, human-readable key--value set with an assistant model and predictive-entropy ranking, then trains lightweight LoRA adapters to suppress the original values on their keys while preserving general capabilities. Ownership is verified under black/gray-box access by aggregating likelihood and semantic evidence into a fingerprint success rate. By relying on probabilistic forgetting traces rather than fixed trigger--response patterns, \textsc{ForgetMark} avoids high-perplexity triggers, reduces detectability, and lowers false triggers. Across diverse architectures and settings, it achieves 100\% ownership verification on fingerprinted models while maintaining standard performance, surpasses backdoor baselines in stealthiness and robustness to model merging, and remains effective under moderate incremental fine-tuning. Our code and data are available at \href{this https URL}{this https URL}. 

---
# Autonomous Materials Exploration by Integrating Automated Phase Identification and AI-Assisted Human Reasoning 

**Authors**: Ming-Chiang Chang, Maximilian Amsler, Duncan R. Sutherland, Sebastian Ament, Katie R. Gann, Lan Zhou, Louisa M. Smieska, Arthur R. Woll, John M. Gregoire, Carla P. Gomes, R. Bruce van Dover, Michael O. Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2601.08185)  

**Abstract**: Autonomous experimentation holds the potential to accelerate materials development by combining artificial intelligence (AI) with modular robotic platforms to explore extensive combinatorial chemical and processing spaces. Such self-driving laboratories can not only increase the throughput of repetitive experiments, but also incorporate human domain expertise to drive the search towards user-defined objectives, including improved materials performance metrics. We present an autonomous materials synthesis extension to SARA, the Scientific Autonomous Reasoning Agent, utilizing phase information provided by an automated probabilistic phase labeling algorithm to expedite the search for targeted phase regions. By incorporating human input into an expanded SARA-H (SARA with human-in-the-loop) framework, we enhance the efficiency of the underlying reasoning process. Using synthetic benchmarks, we demonstrate the efficiency of our AI implementation and show that the human input can contribute to significant improvement in sampling efficiency. We conduct experimental active learning campaigns using robotic processing of thin-film samples of several oxide material systems, including Bi$_2$O$_3$, SnO$_x$, and Bi-Ti-O, using lateral-gradient laser spike annealing to synthesize and kinetically trap metastable phases. We showcase the utility of human-in-the-loop autonomous experimentation for the Bi-Ti-O system, where we identify extensive processing domains that stabilize $\delta$-Bi$_2$O$_3$ and Bi$_2$Ti$_2$O$_7$, explore dwell-dependent ternary oxide phase behavior, and provide evidence confirming predictions that cationic substitutional doping of TiO$_2$ with Bi inhibits the unfavorable transformation of the metastable anatase to the ground-state rutile phase. The autonomous methods we have developed enable the discovery of new materials and new understanding of materials synthesis and properties. 

---
# GI-Bench: A Panoramic Benchmark Revealing the Knowledge-Experience Dissociation of Multimodal Large Language Models in Gastrointestinal Endoscopy Against Clinical Standards 

**Authors**: Yan Zhu, Te Luo, Pei-Yao Fu, Zhen Zhang, Zi-Long Wang, Yi-Fan Qu, Zi-Han Geng, Jia-Qi Xu, Lu Yao, Li-Yun Ma, Wei Su, Wei-Feng Chen, Quan-Lin Li, Shuo Wang, Ping-Hong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.08183)  

**Abstract**: Multimodal Large Language Models (MLLMs) show promise in gastroenterology, yet their performance against comprehensive clinical workflows and human benchmarks remains unverified. To systematically evaluate state-of-the-art MLLMs across a panoramic gastrointestinal endoscopy workflow and determine their clinical utility compared with human endoscopists. We constructed GI-Bench, a benchmark encompassing 20 fine-grained lesion categories. Twelve MLLMs were evaluated across a five-stage clinical workflow: anatomical localization, lesion identification, diagnosis, findings description, and management. Model performance was benchmarked against three junior endoscopists and three residency trainees using Macro-F1, mean Intersection-over-Union (mIoU), and multi-dimensional Likert scale. Gemini-3-Pro achieved state-of-the-art performance. In diagnostic reasoning, top-tier models (Macro-F1 0.641) outperformed trainees (0.492) and rivaled junior endoscopists (0.727; p>0.05). However, a critical "spatial grounding bottleneck" persisted; human lesion localization (mIoU >0.506) significantly outperformed the best model (0.345; p<0.05). Furthermore, qualitative analysis revealed a "fluency-accuracy paradox": models generated reports with superior linguistic readability compared with humans (p<0.05) but exhibited significantly lower factual correctness (p<0.05) due to "over-interpretation" and hallucination of visual this http URL-Bench maintains a dynamic leaderboard that tracks the evolving performance of MLLMs in clinical endoscopy. The current rankings and benchmark results are available at this https URL. 

---
# Instruction-Driven 3D Facial Expression Generation and Transition 

**Authors**: Anh H. Vo, Tae-Seok Kim, Hulin Jin, Soo-Mi Choi, Yong-Guk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.08179)  

**Abstract**: A 3D avatar typically has one of six cardinal facial expressions. To simulate realistic emotional variation, we should be able to render a facial transition between two arbitrary expressions. This study presents a new framework for instruction-driven facial expression generation that produces a 3D face and, starting from an image of the face, transforms the facial expression from one designated facial expression to another. The Instruction-driven Facial Expression Decomposer (IFED) module is introduced to facilitate multimodal data learning and capture the correlation between textual descriptions and facial expression features. Subsequently, we propose the Instruction to Facial Expression Transition (I2FET) method, which leverages IFED and a vertex reconstruction loss function to refine the semantic comprehension of latent vectors, thus generating a facial expression sequence according to the given instruction. Lastly, we present the Facial Expression Transition model to generate smooth transitions between facial expressions. Extensive evaluation suggests that the proposed model outperforms state-of-the-art methods on the CK+ and CelebV-HQ datasets. The results show that our framework can generate facial expression trajectories according to text instruction. Considering that text prompts allow us to make diverse descriptions of human emotional states, the repertoire of facial expressions and the transitions between them can be expanded greatly. We expect our framework to find various practical applications More information about our project can be found at this https URL 

---
# Prompt-Based Clarity Evaluation and Topic Detection in Political Question Answering 

**Authors**: Lavanya Prahallad, Sai Utkarsh Choudarypally, Pragna Prahallad, Pranathi Prahallad  

**Link**: [PDF](https://arxiv.org/pdf/2601.08176)  

**Abstract**: Automatic evaluation of large language model (LLM) responses requires not only factual correctness but also clarity, particularly in political question-answering. While recent datasets provide human annotations for clarity and evasion, the impact of prompt design on automatic clarity evaluation remains underexplored. In this paper, we study prompt-based clarity evaluation using the CLARITY dataset from the SemEval 2026 shared task. We compare a GPT-3.5 baseline provided with the dataset against GPT-5.2 evaluated under three prompting strategies: simple prompting, chain-of-thought prompting, and chain-of-thought with few-shot examples. Model predictions are evaluated against human annotations using accuracy and class-wise metrics for clarity and evasion, along with hierarchical exact match. Results show that GPT-5.2 consistently outperforms the GPT-3.5 baseline on clarity prediction, with accuracy improving from 56 percent to 63 percent under chain-of-thought with few-shot prompting. Chain-of-thought prompting yields the highest evasion accuracy at 34 percent, though improvements are less stable across fine-grained evasion categories. We further evaluate topic identification and find that reasoning-based prompting improves accuracy from 60 percent to 74 percent relative to human annotations. Overall, our findings indicate that prompt design reliably improves high-level clarity evaluation, while fine-grained evasion and topic detection remain challenging despite structured reasoning prompts. 

---
# SwiftMem: Fast Agentic Memory via Query-aware Indexing 

**Authors**: Anxin Tian, Yiming Li, Xing Li, Hui-Ling Zhen, Lei Chen, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08160)  

**Abstract**: Agentic memory systems have become critical for enabling LLM agents to maintain long-term context and retrieve relevant information efficiently. However, existing memory frameworks suffer from a fundamental limitation: they perform exhaustive retrieval across the entire storage layer regardless of query characteristics. This brute-force approach creates severe latency bottlenecks as memory grows, hindering real-time agent interactions. We propose SwiftMem, a query-aware agentic memory system that achieves sub-linear retrieval through specialized indexing over temporal and semantic dimensions. Our temporal index enables logarithmic-time range queries for time-sensitive retrieval, while the semantic DAG-Tag index maps queries to relevant topics through hierarchical tag structures. To address memory fragmentation during growth, we introduce an embedding-tag co-consolidation mechanism that reorganizes storage based on semantic clusters to improve cache locality. Experiments on LoCoMo and LongMemEval benchmarks demonstrate that SwiftMem achieves 47$\times$ faster search compared to state-of-the-art baselines while maintaining competitive accuracy, enabling practical deployment of memory-augmented LLM agents. 

---
# Dynamic Graph Structure Learning via Resistance Curvature Flow 

**Authors**: Chaoqun Fei, Huanjiang Liu, Tinglve Zhou, Yangyang Li, Tianyong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08149)  

**Abstract**: Geometric Representation Learning (GRL) aims to approximate the non-Euclidean topology of high-dimensional data through discrete graph structures, grounded in the manifold hypothesis. However, traditional static graph construction methods based on Euclidean distance often fail to capture the intrinsic curvature characteristics of the data manifold. Although Ollivier-Ricci Curvature Flow (OCF) has proven to be a powerful tool for dynamic topological optimization, its core reliance on Optimal Transport (Wasserstein distance) leads to prohibitive computational complexity, severely limiting its application in large-scale datasets and deep learning frameworks. To break this bottleneck, this paper proposes a novel geometric evolution framework: Resistance Curvature Flow (RCF). Leveraging the concept of effective resistance from circuit physics, RCF transforms expensive curvature optimization into efficient matrix operations. This approach achieves over 100x computational acceleration while maintaining geometric optimization capabilities comparable to OCF. We provide an in-depth exploration of the theoretical foundations and dynamical principles of RCF, elucidating how it guides the redistribution of edge weights via curvature gradients to eliminate topological noise and strengthen local cluster structures. Furthermore, we provide a mechanistic explanation of RCF's role in manifold enhancement and noise suppression, as well as its compatibility with deep learning models. We design a graph optimization algorithm, DGSL-RCF, based on this framework. Experimental results across deep metric learning, manifold learning, and graph structure learning demonstrate that DGSL-RCF significantly improves representation quality and downstream task performance. 

---
# Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models 

**Authors**: Seokho Ahn, Sungbok Shin, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2601.08148)  

**Abstract**: Rich and informative profiling to capture user preferences is essential for improving recommendation quality. However, there is still no consensus on how best to construct and utilize such profiles. To address this, we revisit recent profiling-based approaches in recommender systems along four dimensions: 1) knowledge base, 2) preference indicator, 3) impact range, and 4) subject. We argue that large language models (LLMs) are effective at extracting compressed rationales from diverse knowledge sources, while knowledge graphs (KGs) are better suited for propagating these profiles to extend their reach. Building on this insight, we propose a new recommendation model, called SPiKE. SPiKE consists of three core components: i) Entity profile generation, which uses LLMs to generate semantic profiles for all KG entities; ii) Profile-aware KG aggregation, which integrates these profiles into the KG; and iii) Pairwise profile preference matching, which aligns LLM- and KG-based representations during training. In experiments, we demonstrate that SPiKE consistently outperforms state-of-the-art KG- and LLM-based recommenders in real-world settings. 

---
# Mechanisms are Transferable: Data-Efficient Low-Resource Adaptation via Circuit-Targeted Supervised Fine-Tuning 

**Authors**: Khumaisa Nur'aini, Ayu Purwarianti, Alham Fikri Aji, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2601.08146)  

**Abstract**: Adapting LLMs to low-resource languages is difficult: labeled data is scarce, full-model fine-tuning is unstable, and continued cross-lingual tuning can cause catastrophic forgetting. We propose Circuit-Targeted Supervised Fine-Tuning (CT-SFT): a counterfactual-free adaptation of CD-T (Contextual Decomposition Transformer) that uses a label-balanced mean baseline and task-directional relevance scoring to identify a sparse set of task-relevant attention heads in a proxy-language checkpoint, then transfer learns to a target language by updating only those heads (plus LayerNorm) via head-level gradient masking. Across NusaX-Senti and XNLI, CT-SFT improves cross-lingual accuracy over continued full fine-tuning while updating only a small subset of model parameters. We find an editing-preserving trade-off: harder transfers favor editing circuit heads, while easier transfers often favor near-zero (i.e., low-relevance heads) updates, preserving the source mechanism. CT-SFT also substantially reduces catastrophic forgetting, preserving proxy/source-language competence during transfer. 

---
# Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-training 

**Authors**: Muhammad Taimoor Hassan, Jawad Ahmed, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2601.08141)  

**Abstract**: Despite remarkable progress in large language models, Urdu-a language spoken by over 230 million people-remains critically underrepresented in modern NLP systems. Existing multilingual models demonstrate poor performance on Urdu-specific tasks, struggling with the language's complex morphology, right-to-left Nastaliq script, and rich literary traditions. Even the base LLaMA-3.1 8B-Instruct model shows limited capability in generating fluent, contextually appropriate Urdu text. We introduce Qalb, an Urdu language model developed through a two-stage approach: continued pre-training followed by supervised fine-tuning. Starting from LLaMA 3.1 8B, we perform continued pre-training on a dataset of 1.97 billion tokens. This corpus comprises 1.84 billion tokens of diverse Urdu text-spanning news archives, classical and contemporary literature, government documents, and social media-combined with 140 million tokens of English Wikipedia data to prevent catastrophic forgetting. We then fine-tune the resulting model on the Alif Urdu-instruct dataset. Through extensive evaluation on Urdu-specific benchmarks, Qalb demonstrates substantial improvements, achieving a weighted average score of 90.34 and outperforming the previous state-of-the-art Alif-1.0-Instruct model (87.1) by 3.24 points, while also surpassing the base LLaMA-3.1 8B-Instruct model by 44.64 points. Qalb achieves state-of-the-art performance with comprehensive evaluation across seven diverse tasks including Classification, Sentiment Analysis, and Reasoning. Our results demonstrate that continued pre-training on diverse, high-quality language data, combined with targeted instruction fine-tuning, effectively adapts foundation models to low-resource languages. 

---
# Subspace Alignment for Vision-Language Model Test-time Adaptation 

**Authors**: Zhichen Zeng, Wenxuan Bao, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Xuying Ning, Yuchen Yan, Chen Luo, Monica Xiao Cheng, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2601.08139)  

**Abstract**: Vision-language models (VLMs), despite their extraordinary zero-shot capabilities, are vulnerable to distribution shifts. Test-time adaptation (TTA) emerges as a predominant strategy to adapt VLMs to unlabeled test data on the fly. However, existing TTA methods heavily rely on zero-shot predictions as pseudo-labels for self-training, which can be unreliable under distribution shifts and misguide adaptation due to two fundamental limitations. First (Modality Gap), distribution shifts induce gaps between visual and textual modalities, making cross-modal relations inaccurate. Second (Visual Nuisance), visual embeddings encode rich but task-irrelevant noise that often overwhelms task-specific semantics under distribution shifts. To address these limitations, we propose SubTTA, which aligns the semantic subspaces of both modalities to enhance zero-shot predictions to better guide the TTA process. To bridge the modality gap, SubTTA extracts the principal subspaces of both modalities and aligns the visual manifold to the textual semantic anchor by minimizing their chordal distance. To eliminate visual nuisance, SubTTA projects the aligned visual features onto the task-specific textual subspace, which filters out task-irrelevant noise by constraining visual embeddings within the valid semantic span, and standard TTA is further performed on the purified space to refine the decision boundaries. Extensive experiments on various benchmarks and VLM architectures demonstrate the effectiveness of SubTTA, yielding an average improvement of 2.24% over state-of-the-art TTA methods. 

---
# How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation? 

**Authors**: Peng Gao, Yujian Lee, Yongqi Xu, Wentao Fan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08133)  

**Abstract**: Audio-visual semantic segmentation (AVSS) represents an extension of the audio-visual segmentation (AVS) task, necessitating a semantic understanding of audio-visual scenes beyond merely identifying sound-emitting objects at the visual pixel level. Contrary to a previous methodology, by decomposing the AVSS task into two discrete subtasks by initially providing a prompted segmentation mask to facilitate subsequent semantic analysis, our approach innovates on this foundational strategy. We introduce a novel collaborative framework, \textit{S}tepping \textit{S}tone \textit{P}lus (SSP), which integrates optical flow and textual prompts to assist the segmentation process. In scenarios where sound sources frequently coexist with moving objects, our pre-mask technique leverages optical flow to capture motion dynamics, providing essential temporal context for precise segmentation. To address the challenge posed by stationary sound-emitting objects, such as alarm clocks, SSP incorporates two specific textual prompts: one identifies the category of the sound-emitting object, and the other provides a broader description of the scene. Additionally, we implement a visual-textual alignment module (VTA) to facilitate cross-modal integration, delivering more coherent and contextually relevant semantic interpretations. Our training regimen involves a post-mask technique aimed at compelling the model to learn the diagram of the optical flow. Experimental results demonstrate that SSP outperforms existing AVS methods, delivering efficient and precise segmentation results. 

---
# PathoGen: Diffusion-Based Synthesis of Realistic Lesions in Histopathology Images 

**Authors**: Mohamad Koohi-Moghadam, Mohammad-Ali Nikouei Mahani, Kyongtae Tyler Bae  

**Link**: [PDF](https://arxiv.org/pdf/2601.08127)  

**Abstract**: The development of robust artificial intelligence models for histopathology diagnosis is severely constrained by the scarcity of expert-annotated lesion data, particularly for rare pathologies and underrepresented disease subtypes. While data augmentation offers a potential solution, existing methods fail to generate sufficiently realistic lesion morphologies that preserve the complex spatial relationships and cellular architectures characteristic of histopathological tissues. Here we present PathoGen, a diffusion-based generative model that enables controllable, high-fidelity inpainting of lesions into benign histopathology images. Unlike conventional augmentation techniques, PathoGen leverages the iterative refinement process of diffusion models to synthesize lesions with natural tissue boundaries, preserved cellular structures, and authentic staining characteristics. We validate PathoGen across four diverse datasets representing distinct diagnostic challenges: kidney, skin, breast, and prostate pathology. Quantitative assessment confirms that PathoGen outperforms state-of-the-art generative baselines, including conditional GAN and Stable Diffusion, in image fidelity and distributional similarity. Crucially, we show that augmenting training sets with PathoGen-synthesized lesions enhances downstream segmentation performance compared to traditional geometric augmentations, particularly in data-scarce regimes. Besides, by simultaneously generating realistic morphology and pixel-level ground truth, PathoGen effectively overcomes the manual annotation bottleneck. This approach offers a scalable pathway for developing generalizable medical AI systems despite limited expert-labeled data. 

---
# CSQL: Mapping Documents into Causal Databases 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08109)  

**Abstract**: We describe a novel system, CSQL, which automatically converts a collection of unstructured text documents into an SQL-queryable causal database (CDB). A CDB differs from a traditional DB: it is designed to answer "why'' questions via causal interventions and structured causal queries. CSQL builds on our earlier system, DEMOCRITUS, which converts documents into thousands of local causal models derived from causal discourse. Unlike RAG-based systems or knowledge-graph based approaches, CSQL supports causal analysis over document collections rather than purely associative retrieval. For example, given an article on the origins of human bipedal walking, CSQL enables queries such as: "What are the strongest causal influences on bipedalism?'' or "Which variables act as causal hubs with the largest downstream influence?'' Beyond single-document case studies, we show that CSQL can also ingest RAG/IE-compiled causal corpora at scale by compiling the Testing Causal Claims (TCC) dataset of economics papers into a causal database containing 265,656 claim instances spanning 45,319 papers, 44 years, and 1,575 reported method strings, thereby enabling corpus-level causal queries and longitudinal analyses in CSQL. Viewed abstractly, CSQL functions as a compiler from unstructured documents into a causal database equipped with a principled algebra of queries, and can be applied broadly across many domains ranging from business, humanities, and science. 

---
# Debiasing Large Language Models via Adaptive Causal Prompting with Sketch-of-Thought 

**Authors**: Bowen Li, Ziqi Xu, Jing Ren, Renqiang Luo, Xikun Zhang, Xiuzhen Zhang, Yongli Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2601.08108)  

**Abstract**: Despite notable advancements in prompting methods for Large Language Models (LLMs), such as Chain-of-Thought (CoT), existing strategies still suffer from excessive token usage and limited generalisability across diverse reasoning tasks. To address these limitations, we propose an Adaptive Causal Prompting with Sketch-of-Thought (ACPS) framework, which leverages structural causal models to infer the causal effect of a query on its answer and adaptively select an appropriate intervention (i.e., standard front-door and conditional front-door adjustments). This design enables generalisable causal reasoning across heterogeneous tasks without task-specific retraining. By replacing verbose CoT with concise Sketch-of-Thought, ACPS enables efficient reasoning that significantly reduces token usage and inference cost. Extensive experiments on multiple reasoning benchmarks and LLMs demonstrate that ACPS consistently outperforms existing prompting baselines in terms of accuracy, robustness, and computational efficiency. 

---
# STO-RL: Offline RL under Sparse Rewards via LLM-Guided Subgoal Temporal Order 

**Authors**: Chengyang Gu, Yuxin Pan, Hui Xiong, Yize Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.08107)  

**Abstract**: Offline reinforcement learning (RL) enables policy learning from pre-collected datasets, avoiding costly and risky online interactions, but it often struggles with long-horizon tasks involving sparse rewards. Existing goal-conditioned and hierarchical offline RL methods decompose such tasks and generate intermediate rewards to mitigate limitations of traditional offline RL, but usually overlook temporal dependencies among subgoals and rely on imprecise reward shaping, leading to suboptimal policies. To address these issues, we propose STO-RL (Offline RL using LLM-Guided Subgoal Temporal Order), an offline RL framework that leverages large language models (LLMs) to generate temporally ordered subgoal sequences and corresponding state-to-subgoal-stage mappings. Using this temporal structure, STO-RL applies potential-based reward shaping to transform sparse terminal rewards into dense, temporally consistent signals, promoting subgoal progress while avoiding suboptimal solutions. The resulting augmented dataset with shaped rewards enables efficient offline training of high-performing policies. Evaluations on four discrete and continuous sparse-reward benchmarks demonstrate that STO-RL consistently outperforms state-of-the-art offline goal-conditioned and hierarchical RL baselines, achieving faster convergence, higher success rates, and shorter trajectories. Ablation studies further confirm STO-RL's robustness to imperfect or noisy LLM-generated subgoal sequences, demonstrating that LLM-guided subgoal temporal structures combined with theoretically grounded reward shaping provide a practical and scalable solution for long-horizon offline RL. 

---
# High-Fidelity Modeling of Stochastic Chemical Dynamics on Complex Manifolds: A Multi-Scale SIREN-PINN Framework for the Curvature-Perturbed Ginzburg-Landau Equation 

**Authors**: Julian Evan Chrisnanto, Salsabila Rahma Alia, Nurfauzi Fadillah, Yulison Herry Chrisnanto  

**Link**: [PDF](https://arxiv.org/pdf/2601.08104)  

**Abstract**: The accurate identification and control of spatiotemporal chaos in reaction-diffusion systems remains a grand challenge in chemical engineering, particularly when the underlying catalytic surface possesses complex, unknown topography. In the \textit{Defect Turbulence} regime, system dynamics are governed by topological phase singularities (spiral waves) whose motion couples to manifold curvature via geometric pinning. Conventional Physics-Informed Neural Networks (PINNs) using ReLU or Tanh activations suffer from fundamental \textit{spectral bias}, failing to resolve high-frequency gradients and causing amplitude collapse or phase drift. We propose a Multi-Scale SIREN-PINN architecture leveraging periodic sinusoidal activations with frequency-diverse initialization, embedding the appropriate inductive bias for wave-like physics directly into the network structure. This enables simultaneous resolution of macroscopic wave envelopes and microscopic defect cores. Validated on the complex Ginzburg-Landau equation evolving on latent Riemannian manifolds, our architecture achieves relative state prediction error $\epsilon_{L_2} \approx 1.92 \times 10^{-2}$, outperforming standard baselines by an order of magnitude while preserving topological invariants ($|\Delta N_{defects}| < 1$). We solve the ill-posed \textit{inverse pinning problem}, reconstructing hidden Gaussian curvature fields solely from partial observations of chaotic wave dynamics (Pearson correlation $\rho = 0.965$). Training dynamics reveal a distinctive Spectral Phase Transition at epoch $\sim 2,100$, where cooperative minimization of physics and geometry losses drives the solver to Pareto-optimal solutions. This work establishes a new paradigm for Geometric Catalyst Design, offering a mesh-free, data-driven tool for identifying surface heterogeneity and engineering passive control strategies in turbulent chemical reactors. 

---
# Local-Global Feature Fusion for Subject-Independent EEG Emotion Recognition 

**Authors**: Zheng Zhou, Isabella McEvoy, Camilo E. Valderrama  

**Link**: [PDF](https://arxiv.org/pdf/2601.08094)  

**Abstract**: Subject-independent EEG emotion recognition is challenged by pronounced inter-subject variability and the difficulty of learning robust representations from short, noisy recordings. To address this, we propose a fusion framework that integrates (i) local, channel-wise descriptors and (ii) global, trial-level descriptors, improving cross-subject generalization on the SEED-VII dataset. Local representations are formed per channel by concatenating differential entropy with graph-theoretic features, while global representations summarize time-domain, spectral, and complexity characteristics at the trial level. These representations are fused in a dual-branch transformer with attention-based fusion and domain-adversarial regularization, with samples filtered by an intensity threshold. Experiments under a leave-one-subject-out protocol demonstrate that the proposed method consistently outperforms single-view and classical baselines, achieving approximately 40% mean accuracy in 7-class subject-independent emotion recognition. The code has been released at this https URL. 

---
# Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment 

**Authors**: Qitao Tan, Xiaoying Song, Ningxi Cheng, Ninghao Liu, Xiaoming Zhai, Lingzi Hong, Yanzhi Wang, Zhen Xiang, Geng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08089)  

**Abstract**: Public large language models (LLMs) are typically safety-aligned during pretraining, yet task-specific fine-tuning required for deployment often erodes this alignment and introduces safety risks. Existing defenses either embed safety recovery into fine-tuning or rely on fine-tuning-derived priors for post-hoc correction, leaving safety recovery tightly coupled with training and incurring high computational overhead and a complex workflow. To address these challenges, we propose \texttt{Q-realign}, a post-hoc defense method based on post-training quantization, guided by an analysis of representational structure. By reframing quantization as a dual-objective procedure for compression and safety, \texttt{Q-realign} decouples safety alignment from fine-tuning and naturally piggybacks into modern deployment pipelines. Experiments across multiple models and datasets demonstrate that our method substantially reduces unsafe behaviors while preserving task performance, with significant reductions in memory usage and GPU hours. Notably, our approach can recover the safety alignment of a fine-tuned 7B LLM on a single RTX 4090 within 40 minutes. Overall, our work provides a practical, turnkey solution for safety-aware deployment. 

---
# Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models 

**Authors**: Zhenghao He, Guangzhi Xiong, Bohan Liu, Sanchit Sinha, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08058)  

**Abstract**: Chain-of-Thought (CoT) prompting has improved the reasoning performance of large language models (LLMs), but it remains unclear why it works and whether it is the unique mechanism for triggering reasoning in large language models. In this work, we study this question by directly analyzing and intervening on the internal representations of LLMs with Sparse Autoencoders (SAEs), identifying a small set of latent features that are causally associated with LLM reasoning behavior. Across multiple model families and reasoning benchmarks, we find that steering a single reasoning-related latent feature can substantially improve accuracy without explicit CoT prompting. For large models, latent steering achieves performance comparable to standard CoT prompting while producing more efficient outputs. We further observe that this reasoning-oriented internal state is triggered early in generation and can override prompt-level instructions that discourage explicit reasoning. Overall, our results suggest that multi-step reasoning in LLMs is supported by latent internal activations that can be externally activated, while CoT prompting is one effective, but not unique, way of activating this mechanism rather than its necessary cause. 

---
# The Role of Noisy Data in Improving CNN Robustness for Image Classification 

**Authors**: Oscar H. Ramírez-Agudelo, Nicoleta Gorea, Aliza Reif, Lorenzo Bonasera, Michael Karl  

**Link**: [PDF](https://arxiv.org/pdf/2601.08043)  

**Abstract**: Data quality plays a central role in the performance and robustness of convolutional neural networks (CNNs) for image classification. While high-quality data is often preferred for training, real-world inputs are frequently affected by noise and other distortions. This paper investigates the effect of deliberately introducing controlled noise into the training data to improve model robustness. Using the CIFAR-10 dataset, we evaluate the impact of three common corruptions, namely Gaussian noise, Salt-and-Pepper noise, and Gaussian blur at varying intensities and training set pollution levels. Experiments using a Resnet-18 model reveal that incorporating just 10\% noisy data during training is sufficient to significantly reduce test loss and enhance accuracy under fully corrupted test conditions, with minimal impact on clean-data performance. These findings suggest that strategic exposure to noise can act as a simple yet effective regularizer, offering a practical trade-off between traditional data cleanliness and real-world resilience. 

---
# FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures 

**Authors**: Jifeng Song, Arun Das, Pan Wang, Hui Ji, Kun Zhao, Yufei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08026)  

**Abstract**: Scientific compound figures combine multiple labeled panels into a single image, but captions in real pipelines are often missing or only provide figure-level summaries, making panel-level understanding difficult. In this paper, we propose FigEx2, visual-conditioned framework that localizes panels and generates panel-wise captions directly from the compound figure. To mitigate the impact of diverse phrasing in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively filters token-level features to stabilize the detection query space. Furthermore, we employ a staged optimization strategy combining supervised learning with reinforcement learning (RL), utilizing CLIP-based alignment and BERTScore-based semantic rewards to enforce strict multimodal consistency. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. Experimental results demonstrate that FigEx2 achieves a superior 0.726 mAP@0.5:0.95 for detection and significantly outperforms Qwen3-VL-8B by 0.51 in METEOR and 0.24 in BERTScore. Notably, FigEx2 exhibits remarkable zero-shot transferability to out-of-distribution scientific domains without any fine-tuning. 

---
# Representations of Text and Images Align From Layer One 

**Authors**: Evžen Wybitul, Javier Rando, Florian Tramèr, Stanislav Fort  

**Link**: [PDF](https://arxiv.org/pdf/2601.08017)  

**Abstract**: We show that for a variety of concepts in adapter-based vision-language models, the representations of their images and their text descriptions are meaningfully aligned from the very first layer. This contradicts the established view that such image-text alignment only appears in late layers. We show this using a new synthesis-based method inspired by DeepDream: given a textual concept such as "Jupiter", we extract its concept vector at a given layer, and then use optimisation to synthesise an image whose representation aligns with that vector. We apply our approach to hundreds of concepts across seven layers in Gemma 3, and find that the synthesised images often depict salient visual features of the targeted textual concepts: for example, already at layer 1, more than 50 % of images depict recognisable features of animals, activities, or seasons. Our method thus provides direct, constructive evidence of image-text alignment on a concept-by-concept and layer-by-layer basis. Unlike previous methods for measuring multimodal alignment, our approach is simple, fast, and does not require auxiliary models or datasets. It also offers a new path towards model interpretability, by providing a way to visualise a model's representation space by backtracing through its image processing components. 

---
# TP-Blend: Textual-Prompt Attention Pairing for Precise Object-Style Blending in Diffusion Models 

**Authors**: Xin Jin, Yichuan Zhong, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.08011)  

**Abstract**: Current text-conditioned diffusion editors handle single object replacement well but struggle when a new object and a new style must be introduced simultaneously. We present Twin-Prompt Attention Blend (TP-Blend), a lightweight training-free framework that receives two separate textual prompts, one specifying a blend object and the other defining a target style, and injects both into a single denoising trajectory. TP-Blend is driven by two complementary attention processors. Cross-Attention Object Fusion (CAOF) first averages head-wise attention to locate spatial tokens that respond strongly to either prompt, then solves an entropy-regularised optimal transport problem that reassigns complete multi-head feature vectors to those positions. CAOF updates feature vectors at the full combined dimensionality of all heads (e.g., 640 dimensions in SD-XL), preserving rich cross-head correlations while keeping memory low. Self-Attention Style Fusion (SASF) injects style at every self-attention layer through Detail-Sensitive Instance Normalization. A lightweight one-dimensional Gaussian filter separates low- and high-frequency components; only the high-frequency residual is blended back, imprinting brush-stroke-level texture without disrupting global geometry. SASF further swaps the Key and Value matrices with those derived from the style prompt, enforcing context-aware texture modulation that remains independent of object fusion. Extensive experiments show that TP-Blend produces high-resolution, photo-realistic edits with precise control over both content and appearance, surpassing recent baselines in quantitative fidelity, perceptual quality, and inference speed. 

---
# LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback 

**Authors**: Weiyue Li, Mingxiao Song, Zhenda Shen, Dachuan Zhao, Yunfan Long, Yi Li, Yongce Li, Ruyi Yang, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08003)  

**Abstract**: Large Language Models (LLMs) often struggle with creative generation, and multi-agent frameworks that improve reasoning through interaction can paradoxically hinder creativity by inducing content homogenization. We introduce LLM Review, a peer-review-inspired framework implementing Blind Peer Review: agents exchange targeted feedback while revising independently, preserving divergent creative trajectories. To enable rigorous evaluation, we propose SciFi-100, a science fiction writing dataset with a unified framework combining LLM-as-a-judge scoring, human annotation, and rule-based novelty metrics. Experiments demonstrate that LLM Review consistently outperforms multi-agent baselines, and smaller models with our framework can surpass larger single-agent models, suggesting interaction structure may substitute for model scale. 

---
# DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs 

**Authors**: Nayoung Choi, Jonathan Zhang, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07994)  

**Abstract**: Large Language Models (LLMs) often exhibit increased response latency and degraded answer quality as dialogue length grows, making effective context management essential. However, existing methods rely on extra LLM calls to build memory or perform offline memory construction without considering the current user utterance, which can introduce inefficiencies or disrupt conversational continuity. We introduce DyCP, a lightweight context management method that dynamically segment and retrieve relevant memory at query time. It preserves the sequential structure of dialogue without predefined topic boundaries and supports efficient, adaptive context retrieval. Across three long-form dialogue benchmarks, LoCoMo, MT-Bench+, and SCM4LLMs, and multiple LLMs, DyCP consistently improves answer quality while reducing response latency. We also examine the gap between modern LLMs' expanded context windows and their actual long-context processing capacity, highlighting the continued importance of effective context management. 

---
# From Word Sequences to Behavioral Sequences: Adapting Modeling and Evaluation Paradigms for Longitudinal NLP 

**Authors**: Adithya V Ganesan, Vasudha Varadarajan, Oscar NE Kjell, Whitney R Ringwald, Scott Feltman, Benjamin J Luft, Roman Kotov, Ryan L Boyd, H Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2601.07988)  

**Abstract**: While NLP typically treats documents as independent and unordered samples, in longitudinal studies, this assumption rarely holds: documents are nested within authors and ordered in time, forming person-indexed, time-ordered $\textit{behavioral sequences}$. Here, we demonstrate the need for and propose a longitudinal modeling and evaluation paradigm that consequently updates four parts of the NLP pipeline: (1) evaluation splits aligned to generalization over people ($\textit{cross-sectional}$) and/or time ($\textit{prospective}$); (2) accuracy metrics separating between-person differences from within-person dynamics; (3) sequence inputs to incorporate history by default; and (4) model internals that support different $\textit{coarseness}$ of latent state over histories (pooled summaries, explicit dynamics, or interaction-based models). We demonstrate the issues ensued by traditional pipeline and our proposed improvements on a dataset of 17k daily diary transcripts paired with PTSD symptom severity from 238 participants, finding that traditional document-level evaluation can yield substantially different and sometimes reversed conclusions compared to our ecologically valid modeling and evaluation. We tie our results to a broader discussion motivating a shift from word-sequence evaluation toward $\textit{behavior-sequence}$ paradigms for NLP. 

---
# Cultural Compass: A Framework for Organizing Societal Norms to Detect Violations in Human-AI Conversations 

**Authors**: Myra Cheng, Vinodkumar Prabhakaran, Alice Oh, Hayk Stepanyan, Aishwarya Verma, Charu Kalia, Erin MacMurray van Liemt, Sunipa Dev  

**Link**: [PDF](https://arxiv.org/pdf/2601.07973)  

**Abstract**: Generative AI models ought to be useful and safe across cross-cultural contexts. One critical step toward this goal is understanding how AI models adhere to sociocultural norms. While this challenge has gained attention in NLP, existing work lacks both nuance and coverage in understanding and evaluating models' norm adherence. We address these gaps by introducing a taxonomy of norms that clarifies their contexts (e.g., distinguishing between human-human norms that models should recognize and human-AI interactional norms that apply to the human-AI interaction itself), specifications (e.g., relevant domains), and mechanisms (e.g., modes of enforcement). We demonstrate how our taxonomy can be operationalized to automatically evaluate models' norm adherence in naturalistic, open-ended settings. Our exploratory analyses suggest that state-of-the-art models frequently violate norms, though violation rates vary by model, interactional context, and country. We further show that violation rates also vary by prompt intent and situational framing. Our taxonomy and demonstrative evaluation pipeline enable nuanced, context-sensitive evaluation of cultural norm adherence in realistic settings. 

---
# Tuberculosis Screening from Cough Audio: Baseline Models, Clinical Variables, and Uncertainty Quantification 

**Authors**: George P. Kafentzis, Efstratios Selisios  

**Link**: [PDF](https://arxiv.org/pdf/2601.07969)  

**Abstract**: In this paper, we propose a standardized framework for automatic tuberculosis (TB) detection from cough audio and routinely collected clinical data using machine learning. While TB screening from audio has attracted growing interest, progress is difficult to measure because existing studies vary substantially in datasets, cohort definitions, feature representations, model families, validation protocols, and reported metrics. Consequently, reported gains are often not directly comparable, and it remains unclear whether improvements stem from modeling advances or from differences in data and evaluation. We address this gap by establishing a strong, well-documented baseline for TB prediction using cough recordings and accompanying clinical metadata from a recently compiled dataset from several countries. Our pipeline is reproducible end-to-end, covering feature extraction, multimodal fusion, cougher-independent evaluation, and uncertainty quantification, and it reports a consistent suite of clinically relevant metrics to enable fair comparison. We further quantify performance for cough audio-only and fused (audio + clinical metadata) models, and release the full experimental protocol to facilitate benchmarking. This baseline is intended to serve as a common reference point and to reduce methodological variance that currently holds back progress in the field. 

---
# LJ-Spoof: A Generatively Varied Corpus for Audio Anti-Spoofing and Synthesis Source Tracing 

**Authors**: Surya Subramani, Hashim Ali, Hafiz Malik  

**Link**: [PDF](https://arxiv.org/pdf/2601.07958)  

**Abstract**: Speaker-specific anti-spoofing and synthesis-source tracing are central challenges in audio anti-spoofing. Progress has been hampered by the lack of datasets that systematically vary model architectures, synthesis pipelines, and generative parameters. To address this gap, we introduce LJ-Spoof, a speaker-specific, generatively diverse corpus that systematically varies prosody, vocoders, generative hyperparameters, bona fide prompt sources, training regimes, and neural post-processing. The corpus spans one speakers-including studio-quality recordings-30 TTS families, 500 generatively variant subsets, 10 bona fide neural-processing variants, and more than 3 million utterances. This variation-dense design enables robust speaker-conditioned anti-spoofing and fine-grained synthesis-source tracing. We further position this dataset as both a practical reference training resource and a benchmark evaluation suite for anti-spoofing and source tracing. 

---
# LWMSCNN-SE: A Lightweight Multi-Scale Network for Efficient Maize Disease Classification on Edge Devices 

**Authors**: Fikadu Weloday, Jianmei Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.07957)  

**Abstract**: Maize disease classification plays a vital role in mitigating yield losses and ensuring food security. However, the deployment of traditional disease detection models in resource-constrained environments, such as those using smartphones and drones, faces challenges due to high computational costs. To address these challenges, we propose LWMSCNN-SE, a lightweight convolutional neural network (CNN) that integrates multi-scale feature extraction, depthwise separable convolutions, and squeeze-and-Excitation (SE) attention mechanisms. This novel combination enables the model to achieve 96.63% classification accuracy with only 241,348 parameters and 0.666 GFLOPs, making it suitable for real-time deployment in field applications. Our approach addresses the accuracy--efficiency trade-off by delivering high accuracy while maintaining low computational costs, demonstrating its potential for efficient maize disease diagnosis on edge devices in precision farming systems. 

---
# Quantum automated theorem proving 

**Authors**: Zheng-Zhi Sun, Qi Ye, Dong-Ling Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.07953)  

**Abstract**: Automated theorem proving, or more broadly automated reasoning, aims at using computer programs to automatically prove or disprove mathematical theorems and logical statements. It takes on an essential role across a vast array of applications and the quest for enhanced theorem-proving capabilities remains a prominent pursuit in artificial intelligence. Here, we propose a generic framework for quantum automated theorem proving, where the intrinsic quantum superposition and entanglement features would lead to potential advantages. In particular, we introduce quantum representations of knowledge bases and propose corresponding reasoning algorithms for a variety of tasks. We show how automated reasoning can be achieved with quantum resolution in both propositional and first-order logic with quadratically reduced query complexity. In addition, we propose the quantum algebraic proving method for geometric theorems, extending Wu's algebraic approach beyond the classical setting. Through concrete examples, including geometry problems from the International Mathematical Olympiad, we demonstrate how a quantum computer may prove geometric theorems with quadratic better query complexity. Our results establish a primary approach towards building quantum automatic theorem provers, which would be crucial for practical applications of both near-term and future quantum technologies. 

---
# Hybrid SARIMA LSTM Model for Local Weather Forecasting: A Residual Learning Approach for Data Driven Meteorological Prediction 

**Authors**: Shreyas Rajeev, Karthik Mudenahalli Ashoka, Amit Mallappa Tiparaddi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07951)  

**Abstract**: Accurately forecasting long-term atmospheric variables remains a defining challenge in meteorological science due to the chaotic nature of atmospheric systems. Temperature data represents a complex superposition of deterministic cyclical climate forces and stochastic, short-term fluctuations. While planetary mechanics drive predictable seasonal periodicities, rapid meteorological changes such as thermal variations, pressure anomalies, and humidity shifts introduce nonlinear volatilities that defy simple extrapolation. Historically, the Seasonal Autoregressive Integrated Moving Average (SARIMA) model has been the standard for modeling historical weather data, prized for capturing linear seasonal trends. However, SARIMA operates under strict assumptions of stationarity, failing to capture abrupt, nonlinear transitions. This leads to systematic residual errors, manifesting as the under-prediction of sudden spikes or the over-smoothing of declines. Conversely, Deep Learning paradigms, specifically Long Short-Term Memory (LSTM) networks, demonstrate exceptional efficacy in handling intricate time-series data. By utilizing memory gates, LSTMs learn complex nonlinear dependencies. Yet, LSTMs face instability in open-loop forecasting; without ground truth feedback, minor deviations compound recursively, causing divergence. To resolve these limitations, we propose a Hybrid SARIMA-LSTM architecture. This framework employs a residual-learning strategy to decompose temperature into a predictable climate component and a nonlinear weather component. The SARIMA unit models the robust, long-term seasonal trend, while the LSTM is trained exclusively on the residuals the nonlinear errors SARIMA fails to capture. By fusing statistical stability with neural plasticity, this hybrid approach minimizes error propagation and enhances long-horizon accuracy. 

---
# Reinforcement Learning Methods for Neighborhood Selection in Local Search 

**Authors**: Yannick Molinghen, Augustin Delecluse, Renaud De Landtsheer, Stefano Michelini  

**Link**: [PDF](https://arxiv.org/pdf/2601.07948)  

**Abstract**: Reinforcement learning has recently gained traction as a means to improve combinatorial optimization methods, yet its effectiveness within local search metaheuristics specifically remains comparatively underexamined. In this study, we evaluate a range of reinforcement learning-based neighborhood selection strategies -- multi-armed bandits (upper confidence bound, $\epsilon$-greedy) and deep reinforcement learning methods (proximal policy optimization, double deep $Q$-network) -- and compare them against multiple baselines across three different problems: the traveling salesman problem, the pickup and delivery problem with time windows, and the car sequencing problem. We show how search-specific characteristics, particularly large variations in cost due to constraint violation penalties, necessitate carefully designed reward functions to provide stable and informative learning signals. Our extensive experiments reveal that algorithm performance varies substantially across problems, although that $\epsilon$-greedy consistently ranks among the best performers. In contrast, the computational overhead of deep reinforcement learning approaches only makes them competitive with a substantially longer runtime. These findings highlight both the promise and the practical limitations of deep reinforcement learning in local search. 

---
# Coupled Diffusion-Encoder Models for Reconstruction of Flow Fields 

**Authors**: AmirPouya Hemmasian, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2601.07946)  

**Abstract**: Data-driven flow-field reconstruction typically relies on autoencoder architectures that compress high-dimensional states into low-dimensional latent representations. However, classical approaches such as variational autoencoders (VAEs) often struggle to preserve the higher-order statistical structure of fluid flows when subjected to strong compression. We propose DiffCoder, a coupled framework that integrates a probabilistic diffusion model with a conventional convolutional ResNet encoder and trains both components end-to-end. The encoder compresses the flow field into a latent representation, while the diffusion model learns a generative prior over reconstructions conditioned on the compressed state. This design allows DiffCoder to recover distributional and spectral properties that are not strictly required for minimizing pointwise reconstruction loss but are critical for faithfully representing statistical properties of the flow field. We evaluate DiffCoder and VAE baselines across multiple model sizes and compression ratios on a challenging dataset of Kolmogorov flow fields. Under aggressive compression, DiffCoder significantly improves the spectral accuracy while VAEs exhibit substantial degradation. Although both methods show comparable relative L2 reconstruction error, DiffCoder better preserves the underlying distributional structure of the flow. At moderate compression levels, sufficiently large VAEs remain competitive, suggesting that diffusion-based priors provide the greatest benefit when information bottlenecks are severe. These results demonstrate that the generative decoding by diffusion offers a promising path toward compact, statistically consistent representations of complex flow fields. 

---
# Moonworks Lunara Aesthetic Dataset 

**Authors**: Yan Wang, M M Sayeef Abdullah, Partho Hassan, Sabit Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2601.07941)  

**Abstract**: The dataset spans diverse artistic styles, including regionally grounded aesthetics from the Middle East, Northern Europe, East Asia, and South Asia, alongside general categories such as sketch and oil painting. All images are generated using the Moonworks Lunara model and intentionally crafted to embody distinct, high-quality aesthetic styles, yielding a first-of-its-kind dataset with substantially higher aesthetic scores, exceeding even aesthetics-focused datasets, and general-purpose datasets by a larger margin. Each image is accompanied by a human-refined prompt and structured annotations that jointly describe salient objects, attributes, relationships, and stylistic cues. Unlike large-scale web-derived datasets that emphasize breadth over precision, the Lunara Aesthetic Dataset prioritizes aesthetic quality, stylistic diversity, and licensing transparency, and is released under the Apache 2.0 license to support research and unrestricted academic and commercial use. 

---
# SECite: Analyzing and Summarizing Citations in Software Engineering Literature 

**Authors**: Shireesh Reddy Pyreddy, Khaja Valli Pathan, Hasan Masum, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2601.07939)  

**Abstract**: Identifying the strengths and limitations of a research paper is a core component of any literature review. However, traditional summaries reflect only the authors' self-presented perspective. Analyzing how other researchers discuss and cite the paper can offer a deeper, more practical understanding of its contributions and shortcomings. In this research, we introduce SECite, a novel approach for evaluating scholarly impact through sentiment analysis of citation contexts. We develop a semi-automated pipeline to extract citations referencing nine research papers and apply advanced natural language processing (NLP) techniques with unsupervised machine learning to classify these citation statements as positive or negative. Beyond sentiment classification, we use generative AI to produce sentiment-specific summaries that capture the strengths and limitations of each target paper, derived both from clustered citation groups and from the full text. Our findings reveal meaningful patterns in how the academic community perceives these works, highlighting areas of alignment and divergence between external citation feedback and the authors' own presentation. By integrating citation sentiment analysis with LLM-based summarization, this study provides a comprehensive framework for assessing scholarly contributions. 

---
# Towards Specialized Generalists: A Multi-Task MoE-LoRA Framework for Domain-Specific LLM Adaptation 

**Authors**: Yuxin Yang, Aoxiong Zeng, Xiangquan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07935)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has shifted focus from general-purpose capabilities to domain-specific expertise. However, adapting LLMs to specialized fields such as medicine presents two challenge: (1) the "Stability-Plasticity Dilemma", where the model must acquire complex clinical knowledge without suffering from catastrophic forgetting of general world knowledge; and (2) "Task Interference", where disparate sub-tasks, such as medical diagnosis, report summarization, and drug-drug interaction prediction, compete for limited low-rank parameter space. In this paper, we propose Med-MoE-LoRA, a novel framework that integrates Mixture-of-Experts (MoE) with Low-Rank Adaptation (LoRA) to enable efficient multi-task domain adaptation, especially for medical scenarios. Drawing inspiration from recent advances, our framework employs an asymmetric expert distribution where deeper layers are equipped with a higher density of LoRA experts to capture complex semantic abstractions. We further introduce a "Knowledge-Preservation Plugin", inspired by LoRA MoE, to isolate and protect general-purpose reasoning. By utilizing soft merging with adaptive routing and rank-wise decoupling, Med-MoE-LoRA achieves superior performance in medical benchmarks while reducing interference. Experimental results demonstrate that our approach consistently outperforms standard LoRA and conventional MoE architectures across multiple clinical NLP tasks while retaining the model's general cognitive capabilities. 

---
# Enhancing Large Language Models for Time-Series Forecasting via Vector-Injected In-Context Learning 

**Authors**: Jianqi Zhang, Jingyao Wang, Wenwen Qiang, Fanjiang Xu, Changwen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.07903)  

**Abstract**: The World Wide Web needs reliable predictive capabilities to respond to changes in user behavior and usage patterns. Time series forecasting (TSF) is a key means to achieve this goal. In recent years, the large language models (LLMs) for TSF (LLM4TSF) have achieved good performance. However, there is a significant difference between pretraining corpora and time series data, making it hard to guarantee forecasting quality when directly applying LLMs to TSF; fine-tuning LLMs can mitigate this issue, but often incurs substantial computational overhead. Thus, LLM4TSF faces a dual challenge of prediction performance and compute overhead. To address this, we aim to explore a method for improving the forecasting performance of LLM4TSF while freezing all LLM parameters to reduce computational overhead. Inspired by in-context learning (ICL), we propose LVICL. LVICL uses our vector-injected ICL to inject example information into a frozen LLM, eliciting its in-context learning ability and thereby enhancing its performance on the example-related task (i.e., TSF). Specifically, we first use the LLM together with a learnable context vector adapter to extract a context vector from multiple examples adaptively. This vector contains compressed, example-related information. Subsequently, during the forward pass, we inject this vector into every layer of the LLM to improve forecasting performance. Compared with conventional ICL that adds examples into the prompt, our vector-injected ICL does not increase prompt length; moreover, adaptively deriving a context vector from examples suppresses components harmful to forecasting, thereby improving model performance. Extensive experiments demonstrate the effectiveness of our approach. 

---
# Decentralized Online Convex Optimization with Unknown Feedback Delays 

**Authors**: Hao Qiu, Mengxiao Zhang, Juliette Achddou  

**Link**: [PDF](https://arxiv.org/pdf/2601.07901)  

**Abstract**: Decentralized online convex optimization (D-OCO), where multiple agents within a network collaboratively learn optimal decisions in real-time, arises naturally in applications such as federated learning, sensor networks, and multi-agent control.  In this paper, we study D-OCO under unknown, time-and agent-varying feedback delays. While recent work has addressed this problem (Nguyen et al., 2024), existing algorithms assume prior knowledge of the total delay over agents and still suffer from suboptimal dependence on both the delay and network parameters. To overcome these limitations, we propose a novel algorithm that achieves an improved regret bound of O N $\sqrt$ d tot + N $\sqrt$ T  (1-$\sigma$2) 1/4 , where T is the total horizon, d tot denotes the average total delay across agents, N is the number of agents, and 1 -$\sigma$ 2 is the spectral gap of the network. Our approach builds upon recent advances in D-OCO (Wan et al., 2024a), but crucially incorporates an adaptive learning rate mechanism via a decentralized communication protocol. This enables each agent to estimate delays locally using a gossip-based strategy without the prior knowledge of the total delay. We further extend our framework to the strongly convex setting and derive a sharper regret bound of O N $\delta$max ln T $\alpha$  , where $\alpha$ is the strong convexity parameter and $\delta$ max is the maximum number of missing observations averaged over agents. We also show that our upper bounds for both settings are tight up to logarithmic factors. Experimental results validate the effectiveness of our approach, showing improvements over existing benchmark algorithms. 

---
# Large Language Models and Algorithm Execution: Application to an Arithmetic Function 

**Authors**: Farah Ben Slama, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2601.07898)  

**Abstract**: Large Language Models (LLMs) have recently developed new advanced functionalities. Their effectiveness relies on statistical learning and generalization capabilities. However, they face limitations in internalizing the data they process and struggle, for instance, to autonomously execute algorithms. In this paper, we investigate the possibility of extending these models' capabilities to algorithm execution through specialized supervised training focused on reasoning decomposition. We introduce a training model called LLM-DAL (Large Language Model - Decompositional Algorithmic Learning), through which we demonstrate that LLMs' ability to perform complex algorithmic inferences and generalize can be significantly improved when the training method is properly designed to guide the model in its learning process. 

---
# Revealing the Attention Floating Mechanism in Masked Diffusion Models 

**Authors**: Xin Dai, Pengcheng Huang, Zhenghao Liu, Shuo Wang, Yukun Yan, Chaojun Xiao, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.07894)  

**Abstract**: Masked diffusion models (MDMs), which leverage bidirectional attention and a denoising process, are narrowing the performance gap with autoregressive models (ARMs). However, their internal attention mechanisms remain under-explored. This paper investigates the attention behaviors in MDMs, revealing the phenomenon of Attention Floating. Unlike ARMs, where attention converges to a fixed sink, MDMs exhibit dynamic, dispersed attention anchors that shift across denoising steps and layers. Further analysis reveals its Shallow Structure-Aware, Deep Content-Focused attention mechanism: shallow layers utilize floating tokens to build a global structural framework, while deeper layers allocate more capability toward capturing semantic content. Empirically, this distinctive attention pattern provides a mechanistic explanation for the strong in-context learning capabilities of MDMs, allowing them to double the performance compared to ARMs in knowledge-intensive tasks. All codes and datasets are available at this https URL. 

---
# Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification 

**Authors**: Hong Huang, Decheng Wu, Qiangqiang Hu, Guanghua Yu, Jinhai Yang, Jianchen Zhu, Xue Liu, Dapeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07892)  

**Abstract**: The deployment of Large Language Models (LLMs) on resource-constrained edge devices is increasingly hindered by prohibitive memory and computational requirements. While ternary quantization offers a compelling solution by reducing weights to {-1, 0, +1}, current implementations suffer from a fundamental misalignment with commodity hardware. Most existing methods must choose between 2-bit aligned packing, which incurs significant bit wastage, or 1.67-bit irregular packing, which degrades inference speed. To resolve this tension, we propose Sherry, a hardware-efficient ternary quantization framework. Sherry introduces a 3:4 fine-grained sparsity that achieves a regularized 1.25-bit width by packing blocks of four weights into five bits, restoring power-of-two alignment. Furthermore, we identify weight trapping issue in sparse ternary training, which leads to representational collapse. To address this, Sherry introduces Arenas, an annealing residual synapse mechanism that maintains representational diversity during training. Empirical evaluations on LLaMA-3.2 across five benchmarks demonstrate that Sherry matches state-of-the-art ternary performance while significantly reducing model size. Notably, on an Intel i7-14700HX CPU, our 1B model achieves zero accuracy loss compared to SOTA baselines while providing 25% bit savings and 10% speed up. The code is available at this https URL . 

---
# KVzap: Fast, Adaptive, and Faithful KV Cache Pruning 

**Authors**: Simon Jegou, Maximilian Jeblick  

**Link**: [PDF](https://arxiv.org/pdf/2601.07891)  

**Abstract**: Growing context lengths in transformer-based language models have made the key-value (KV) cache a critical inference bottleneck. While many KV cache pruning methods have been proposed, they have not yet been adopted in major inference engines due to speed--accuracy trade-offs. We introduce KVzap, a fast, input-adaptive approximation of KVzip that works in both prefilling and decoding. On Qwen3-8B, Llama-3.1-8B-Instruct, and Qwen3-32B across long-context and reasoning tasks, KVzap achieves $2$--$4\times$ KV cache compression with negligible accuracy loss and achieves state-of-the-art performance on the KVpress leaderboard. Code and models are available at this https URL. 

---
# Small Symbols, Big Risks: Exploring Emoticon Semantic Confusion in Large Language Models 

**Authors**: Weipeng Jiang, Xiaoyu Zhang, Juan Zhai, Shiqing Ma, Chao Shen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07885)  

**Abstract**: Emoticons are widely used in digital communication to convey affective intent, yet their safety implications for Large Language Models (LLMs) remain largely unexplored. In this paper, we identify emoticon semantic confusion, a vulnerability where LLMs misinterpret ASCII-based emoticons to perform unintended and even destructive actions. To systematically study this phenomenon, we develop an automated data generation pipeline and construct a dataset containing 3,757 code-oriented test cases spanning 21 meta-scenarios, four programming languages, and varying contextual complexities. Our study on six LLMs reveals that emoticon semantic confusion is pervasive, with an average confusion ratio exceeding 38%. More critically, over 90% of confused responses yield 'silent failures', which are syntactically valid outputs but deviate from user intent, potentially leading to destructive security consequences. Furthermore, we observe that this vulnerability readily transfers to popular agent frameworks, while existing prompt-based mitigations remain largely ineffective. We call on the community to recognize this emerging vulnerability and develop effective mitigation methods to uphold the safety and reliability of the LLM system. 

---
# Ideological Isolation in Online Social Networks: A Survey of Computational Definitions, Metrics, and Mitigation Strategies 

**Authors**: Xiaodan Wang, Yanbin Liu, Shiqing Wu, Ziying Zhao, Yuxuan Hu, Weihua Li, Quan Bai  

**Link**: [PDF](https://arxiv.org/pdf/2601.07884)  

**Abstract**: The proliferation of online social networks has significantly reshaped the way individuals access and engage with information. While these platforms offer unprecedented connectivity, they may foster environments where users are increasingly exposed to homogeneous content and like-minded interactions. Such dynamics are associated with selective exposure and the emergence of filter bubbles, echo chambers, tunnel vision, and polarization, which together can contribute to ideological isolation and raise concerns about information diversity and public discourse. This survey provides a comprehensive computational review of existing studies that define, analyze, quantify, and mitigate ideological isolation in online social networks. We examine the mechanisms underlying content personalization, user behavior patterns, and network structures that reinforce content-exposure concentration and narrowing dynamics. This paper also systematically reviews methodological approaches for detecting and measuring these isolation-related phenomena, covering network-, content-, and behavior-based metrics. We further organize computational mitigation strategies, including network-topological interventions and recommendation-level controls, and discuss their trade-offs and deployment considerations. By integrating definitions, metrics, and interventions across structural/topological, content-based, interactional, and cognitive isolation, this survey provides a unified computational framework. It serves as a reference for understanding and addressing the key challenges and opportunities in promoting information diversity and reducing ideological fragmentation in the digital age. 

---
# Tackling Heterogeneity in Quantum Federated Learning: An Integrated Sporadic-Personalized Approach 

**Authors**: Ratun Rahman, Shaba Shaon, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07882)  

**Abstract**: Quantum federated learning (QFL) emerges as a powerful technique that combines quantum computing with federated learning to efficiently process complex data across distributed quantum devices while ensuring data privacy in quantum networks. Despite recent research efforts, existing QFL frameworks struggle to achieve optimal model training performance primarily due to inherent heterogeneity in terms of (i) quantum noise where current quantum devices are subject to varying levels of noise due to varying device quality and susceptibility to quantum decoherence, and (ii) heterogeneous data distributions where data across participating quantum devices are naturally non-independent and identically distributed (non-IID). To address these challenges, we propose a novel integrated sporadic-personalized approach called SPQFL that simultaneously handles quantum noise and data heterogeneity in a single QFL framework. It is featured in two key aspects: (i) for quantum noise heterogeneity, we introduce a notion of sporadic learning to tackle quantum noise heterogeneity across quantum devices, and (ii) for quantum data heterogeneity, we implement personalized learning through model regularization to mitigate overfitting during local training on non-IID quantum data distributions, thereby enhancing the convergence of the global model. Moreover, we conduct a rigorous convergence analysis for the proposed SPQFL framework, with both sporadic and personalized learning considerations. Theoretical findings reveal that the upper bound of the SPQFL algorithm is strongly influenced by both the number of quantum devices and the number of quantum noise measurements. Extensive simulation results in real-world datasets also illustrate that the proposed SPQFL approach yields significant improvements in terms of training performance and convergence stability compared to the state-of-the-art methods. 

---
# Sola-Visibility-ISPM: Benchmarking Agentic AI for Identity Security Posture Management Visibility 

**Authors**: Gal Engelberg, Konstantin Koutsyi, Leon Goldberg, Reuven Elezra, Idan Pinto, Tal Moalem, Shmuel Cohen, Yoni Weintrob  

**Link**: [PDF](https://arxiv.org/pdf/2601.07880)  

**Abstract**: Identity Security Posture Management (ISPM) is a core challenge for modern enterprises operating across cloud and SaaS environments. Answering basic ISPM visibility questions, such as understanding identity inventory and configuration hygiene, requires interpreting complex identity data, motivating growing interest in agentic AI systems. Despite this interest, there is currently no standardized way to evaluate how well such systems perform ISPM visibility tasks on real enterprise data. We introduce the Sola Visibility ISPM Benchmark, the first benchmark designed to evaluate agentic AI systems on foundational ISPM visibility tasks using a live, production-grade identity environment spanning AWS, Okta, and Google Workspace. The benchmark focuses on identity inventory and hygiene questions and is accompanied by the Sola AI Agent, a tool-using agent that translates natural-language queries into executable data exploration steps and produces verifiable, evidence-backed answers. Across 77 benchmark questions, the agent achieves strong overall performance, with an expert accuracy of 0.84 and a strict success rate of 0.77. Performance is highest on AWS hygiene tasks, where expert accuracy reaches 0.94, while results on Google Workspace and Okta hygiene tasks are more moderate, yet competitive. Overall, this work provides a practical and reproducible benchmark for evaluating agentic AI systems in identity security and establishes a foundation for future ISPM benchmarks covering more advanced identity analysis and governance tasks. 

---
# Sliced-Wasserstein Distribution Alignment Loss Improves the Ultra-Low-Bit Quantization of Large Language Models 

**Authors**: Deyu Cao, Yixin Yin, Samin Aref  

**Link**: [PDF](https://arxiv.org/pdf/2601.07878)  

**Abstract**: The benefits of most large language models come with steep and often hidden economic and environmental costs due to their resource usage inefficiency during deployment. Model quantization improves energy and memory efficiency through representing model parameters by lower-precision values. However, compression below 4-bits often distorts activation distributions and degrades performance. We address this challenge by introducing a sliced Wasserstein loss function for distribution-aware calibration in ultra-low-bit post-training quantization. The proposed loss aligns the output distributions of full-precision and quantized models under random linear projections, complementing standard mean-squared error loss without adding any computational overhead during inference. Our proposed loss function can be incorporated with any post-training quantization framework that has a retraining component. We demonstrate the performance gains of our proposed model by incorporating it with two frontier methods known as OmniQuant and TesseraQ. Compared to these two baselines, the proposed loss consistently improves both perplexity and downstream task accuracy across multiple ultra-low-bit settings. Our proposed loss function recovers 4.12-20.37% of the OmniQuant's lost accuracy on the language model LLaMA-2-7B, 0.93-7.65% on OPT-6.7B, and 2.26-6.20% on LLaMA-2-13B. TesseraQ's accuracy degradation is recovered by 3.63-7.63% in relative terms when augmented by our proposed loss function. Taken together, these results demonstrate that distributional alignment provides a simple yet effective performance boost that can push the limits of frontier quantization methods. Our method is available on GitHub to facilitate future progress in ultra-low-bit quantization. 

---
# E^2-LLM: Bridging Neural Signals and Interpretable Affective Analysis 

**Authors**: Fei Ma, Han Lin, Yifan Xie, Hongwei Ren, Xiaoyu Shen, Wenbo Ding, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.07877)  

**Abstract**: Emotion recognition from electroencephalography (EEG) signals remains challenging due to high inter-subject variability, limited labeled data, and the lack of interpretable reasoning in existing approaches. While recent multimodal large language models (MLLMs) have advanced emotion analysis, they have not been adapted to handle the unique spatiotemporal characteristics of neural signals. We present E^2-LLM (EEG-to-Emotion Large Language Model), the first MLLM framework for interpretable emotion analysis from EEG. E^2-LLM integrates a pretrained EEG encoder with Qwen-based LLMs through learnable projection layers, employing a multi-stage training pipeline that encompasses emotion-discriminative pretraining, cross-modal alignment, and instruction tuning with chain-of-thought reasoning. We design a comprehensive evaluation protocol covering basic emotion prediction, multi-task reasoning, and zero-shot scenario understanding. Experiments on the dataset across seven emotion categories demonstrate that E^2-LLM achieves excellent performance on emotion classification, with larger variants showing enhanced reliability and superior zero-shot generalization to complex reasoning scenarios. Our work establishes a new paradigm combining physiological signals with LLM reasoning capabilities, showing that model scaling improves both recognition accuracy and interpretable emotional understanding in affective computing. 

---
# NOVAK: Unified adaptive optimizer for deep neural networks 

**Authors**: Sergii Kavun  

**Link**: [PDF](https://arxiv.org/pdf/2601.07876)  

**Abstract**: This work introduces NOVAK, a modular gradient-based optimization algorithm that integrates adaptive moment estimation, rectified learning-rate scheduling, decoupled weight regularization, multiple variants of Nesterov momentum, and lookahead synchronization into a unified, performance-oriented framework. NOVAK adopts a dual-mode architecture consisting of a streamlined fast path designed for production. The optimizer employs custom CUDA kernels that deliver substantial speedups (3-5 for critical operations) while preserving numerical stability under standard stochastic-optimization assumptions. We provide fully developed mathematical formulations for rectified adaptive learning rates, a memory-efficient lookahead mechanism that reduces overhead from O(2p) to O(p + p/k), and the synergistic coupling of complementary optimization components. Theoretical analysis establishes convergence guarantees and elucidates the stability and variance-reduction properties of the method. Extensive empirical evaluation on CIFAR-10, CIFAR-100, ImageNet, and ImageNette demonstrates NOVAK superiority over 14 contemporary optimizers, including Adam, AdamW, RAdam, Lion, and Adan. Across architectures such as ResNet-50, VGG-16, and ViT, NOVAK consistently achieves state-of-the-art accuracy, and exceptional robustness, attaining very high accuracy on VGG-16/ImageNette demonstrating superior architectural robustness compared to contemporary optimizers. The results highlight that NOVAKs architectural contributions (particularly rectification, decoupled decay, and hybrid momentum) are crucial for reliable training of deep plain networks lacking skip connections, addressing a long-standing limitation of existing adaptive optimization methods. 

---
# Multiplicative Orthogonal Sequential Editing for Language Models 

**Authors**: Hao-Xiang Xu, Jun-Yu Ma, Ziqi Peng, Yuhao Sun, Zhen-Hua Ling, Jia-Chen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07873)  

**Abstract**: Knowledge editing aims to efficiently modify the internal knowledge of large language models (LLMs) without compromising their other capabilities. The prevailing editing paradigm, which appends an update matrix to the original parameter matrix, has been shown by some studies to damage key numerical stability indicators (such as condition number and norm), thereby reducing editing performance and general abilities, especially in sequential editing scenario. Although subsequent methods have made some improvements, they remain within the additive framework and have not fundamentally addressed this limitation. To solve this problem, we analyze it from both statistical and mathematical perspectives and conclude that multiplying the original matrix by an orthogonal matrix does not change the numerical stability of the matrix. Inspired by this, different from the previous additive editing paradigm, a multiplicative editing paradigm termed Multiplicative Orthogonal Sequential Editing (MOSE) is proposed. Specifically, we first derive the matrix update in the multiplicative form, the new knowledge is then incorporated into an orthogonal matrix, which is multiplied by the original parameter matrix. In this way, the numerical stability of the edited matrix is unchanged, thereby maintaining editing performance and general abilities. We compared MOSE with several current knowledge editing methods, systematically evaluating their impact on both editing performance and the general abilities across three different LLMs. Experimental results show that MOSE effectively limits deviations in the edited parameter matrix and maintains its numerical stability. Compared to current methods, MOSE achieves a 12.08% improvement in sequential editing performance, while retaining 95.73% of general abilities across downstream tasks. The code is available at this https URL. 

---
# Imaging-anchored Multiomics in Cardiovascular Disease: Integrating Cardiac Imaging, Bulk, Single-cell, and Spatial Transcriptomics 

**Authors**: Minh H. N. Le, Tuan Vinh, Thanh-Huy Nguyen, Tao Li, Bao Quang Gia Le, Han H. Huynh, Monika Raj, Carl Yang, Min Xu, Nguyen Quoc Khanh Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.07871)  

**Abstract**: Cardiovascular disease arises from interactions between inherited risk, molecular programmes, and tissue-scale remodelling that are observed clinically through imaging. Health systems now routinely generate large volumes of cardiac MRI, CT and echocardiography together with bulk, single-cell and spatial transcriptomics, yet these data are still analysed in separate pipelines. This review examines joint representations that link cardiac imaging phenotypes to transcriptomic and spatially resolved molecular states. An imaging-anchored perspective is adopted in which echocardiography, cardiac MRI and CT define a spatial phenotype of the heart, and bulk, single-cell and spatial transcriptomics provide cell-type- and location-specific molecular context. The biological and technical characteristics of these modalities are first summarised, and representation-learning strategies for each are outlined. Multimodal fusion approaches are reviewed, with emphasis on handling missing data, limited sample size, and batch effects. Finally, integrative pipelines for radiogenomics, spatial molecular alignment, and image-based prediction of gene expression are discussed, together with common failure modes, practical considerations, and open challenges. Spatial multiomics of human myocardium and atherosclerotic plaque, single-cell and spatial foundation models, and multimodal medical foundation models are collectively bringing imaging-anchored multiomics closer to large-scale cardiovascular translation. 

---
# RewriteNets: End-to-End Trainable String-Rewriting for Generative Sequence Modeling 

**Authors**: Harshil Vejendla  

**Link**: [PDF](https://arxiv.org/pdf/2601.07868)  

**Abstract**: Dominant sequence models like the Transformer represent structure implicitly through dense attention weights, incurring quadratic complexity. We propose RewriteNets, a novel neural architecture built on an alternative paradigm: explicit, parallel string rewriting. Each layer in a RewriteNet contains a set of learnable rules. For each position in an input sequence, the layer performs four operations: (1) fuzzy matching of rule patterns, (2) conflict resolution via a differentiable assignment operator to select non-overlapping rewrites, (3) application of the chosen rules to replace input segments with output segments of potentially different lengths, and (4) propagation of untouched tokens. While the discrete assignment of rules is non-differentiable, we employ a straight-through Gumbel-Sinkhorn estimator, enabling stable end-to-end training. We evaluate RewriteNets on algorithmic, compositional, and string manipulation tasks, comparing them against strong LSTM and Transformer baselines. Results show that RewriteNets excel at tasks requiring systematic generalization (achieving 98.7% accuracy on the SCAN benchmark's length split) and are computationally more efficient than Transformers. We also provide an analysis of learned rules and an extensive ablation study, demonstrating that this architecture presents a promising direction for sequence modeling with explicit structural inductive biases. 

---
# Affect and Effect: Limitations of regularisation-based continual learning in EEG-based emotion classification 

**Authors**: Nina Peire, Yupei Li, Björn Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2601.07858)  

**Abstract**: Generalisation to unseen subjects in EEG-based emotion classification remains a challenge due to high inter-and intra-subject variability. Continual learning (CL) poses a promising solution by learning from a sequence of tasks while mitigating catastrophic forgetting. Regularisation-based CL approaches, such as Elastic Weight Consolidation (EWC), Synaptic Intelligence (SI), and Memory Aware Synapses (MAS), are commonly used as baselines in EEG-based CL studies, yet their suitability for this problem remains underexplored. This study theoretically and empirically finds that regularisation-based CL methods show limited performance for EEG-based emotion classification on the DREAMER and SEED datasets. We identify a fundamental misalignment in the stability-plasticity trade-off, where regularisation-based methods prioritise mitigating catastrophic forgetting (backward transfer) over adapting to new subjects (forward transfer). We investigate this limitation under subject-incremental sequences and observe that: (1) the heuristics for estimating parameter importance become less reliable under noisy data and covariate shift, (2) gradients on parameters deemed important by these heuristics often interfere with gradient updates required for new subjects, moving optimisation away from the minimum, (3) importance values accumulated across tasks over-constrain the model, and (4) performance is sensitive to subject order. Forward transfer showed no statistically significant improvement over sequential fine-tuning (p > 0.05 across approaches and datasets). The high variability of EEG signals means past subjects provide limited value to future subjects. Regularisation-based continual learning approaches are therefore limited for robust generalisation to unseen subjects in EEG-based emotion classification. 

---
# Feature Entanglement-based Quantum Multimodal Fusion Neural Network 

**Authors**: Yu Wu, Qianli Zhou, Jie Geng, Xinyang Deng, Wen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07856)  

**Abstract**: Multimodal learning aims to enhance perceptual and decision-making capabilities by integrating information from diverse sources. However, classical deep learning approaches face a critical trade-off between the high accuracy of black-box feature-level fusion and the interpretability of less outstanding decision-level fusion, alongside the challenges of parameter explosion and complexity. This paper discusses the accuracy-interpretablity-complexity dilemma under the quantum computation framework and propose a feature entanglement-based quantum multimodal fusion neural network. The model is composed of three core components: a classical feed-forward module for unimodal processing, an interpretable quantum fusion block, and a quantum convolutional neural network (QCNN) for deep feature extraction. By leveraging the strong expressive power of quantum, we have reduced the complexity of multimodal fusion and post-processing to linear, and the fusion process also possesses the interpretability of decision-level fusion. The simulation results demonstrate that our model achieves classification accuracy comparable to classical networks with dozens of times of parameters, exhibiting notable stability and performance across multimodal image datasets. 

---
# An Empirical Study on Knowledge Transfer under Domain and Label Shifts in 3D LiDAR Point Clouds 

**Authors**: Subeen Lee, Siyeong Lee, Namil Kim, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07855)  

**Abstract**: For 3D perception systems to be practical in real-world applications -- from autonomous driving to embodied AI -- models must adapt to continuously evolving object definitions and sensor domains. Yet, research on continual and transfer learning in 3D point cloud perception remains underexplored compared to 2D vision -- particularly under simultaneous domain and label shifts. To address this gap, we propose the RObust Autonomous driving under Dataset shifts (ROAD) benchmark, a comprehensive evaluation suite for LiDAR-based object classification that explicitly accounts for domain shifts as well as three key forms of label evolution: class split, class expansion, and class insertion. Using large-scale datasets (Waymo, NuScenes, Argoverse2), we evaluate zero-shot transfer, linear probe, and CL, and analyze the impact of backbone architectures, training objectives, and CL methods. Our findings reveal limitations of existing approaches under realistic shifts and establish strong baselines for future research in robust 3D perception. 

---
# Immunological Density Shapes Recovery Trajectories in Long COVID 

**Authors**: Jing Wang, Tong Zhang, Xing Niu, Jie Shen, Yiming Luo, Qiaomin Xie, Amar Sra, Zorina Galis, Jeremy Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2601.07854)  

**Abstract**: Post-acute sequelae of SARS-CoV-2 infection (Long COVID) frequently persists for months, yet drivers of clinical remission remain incompletely defined.
Here we analyzed 97,564 longitudinal PASC assessments from 13,511 participants with linked vaccination histories to disentangle passive temporal progression from vaccine-associated change.
Using a clinically validated threshold (PASC $\geq 12$), trajectories separated into three phenotypes: Protected (persistently sub-threshold), Refractory (persistently symptomatic), and Responders (transitioning from symptomatic to recovered).
Across the full cohort, symptom severity increased modestly with elapsed time ($r=0.0521$, $P=1.26\times10^{-59}$), whereas cumulative vaccination showed an inverse association with severity ($r=-0.0434$, $P=5.95\times10^{-42}$). In summary, baseline Long COVID severity appears clinically deterministic. In the absence of intervention, symptoms typically persist without spontaneous resolution. Recovery is primarily associated with repeated immunization. 

---
# FinVault: Benchmarking Financial Agent Safety in Execution-Grounded Environments 

**Authors**: Zhi Yang, Runguo Li, Qiqi Qiang, Jiashun Wang, Fangqi Lou, Mengping Li, Dongpo Cheng, Rui Xu, Heng Lian, Shuo Zhang, Xiaolong Liang, Xiaoming Huang, Zheng Wei, Zhaowei Liu, Xin Guo, Huacan Wang, Ronghao Chen, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07853)  

**Abstract**: Financial agents powered by large language models (LLMs) are increasingly deployed for investment analysis, risk assessment, and automated decision-making, where their abilities to plan, invoke tools, and manipulate mutable state introduce new security risks in high-stakes and highly regulated financial environments. However, existing safety evaluations largely focus on language-model-level content compliance or abstract agent settings, failing to capture execution-grounded risks arising from real operational workflows and state-changing actions. To bridge this gap, we propose FinVault, the first execution-grounded security benchmark for financial agents, comprising 31 regulatory case-driven sandbox scenarios with state-writable databases and explicit compliance constraints, together with 107 real-world vulnerabilities and 963 test cases that systematically cover prompt injection, jailbreaking, financially adapted attacks, as well as benign inputs for false-positive evaluation. Experimental results reveal that existing defense mechanisms remain ineffective in realistic financial agent settings, with average attack success rates (ASR) still reaching up to 50.0\% on state-of-the-art models and remaining non-negligible even for the most robust systems (ASR 6.7\%), highlighting the limited transferability of current safety designs and the need for stronger financial-specific defenses. Our code can be found at this https URL. 

---
# Hierarchical Sparse Plus Low Rank Compression of LLM 

**Authors**: Pawan Kumar, Aditi Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.07839)  

**Abstract**: Modern large language models (LLMs) place extraordinary pressure on memory and compute budgets, making principled compression indispensable for both deployment and continued training. We present Hierarchical Sparse Plus Low-Rank (HSS) compression, a two-stage scheme that (i) removes the largest-magnitude weights into a sparse matrix S and (ii) applies a recursive Hierarchically Sparse Separable (HSS) low-rank factorisation to the dense residual matrix. A recursive rank-reducing strategy and a reverse Cuthill-Mckee (RCM) permutation are introduced to align high weights towards the diagonal with the block-diagonal hierarchy, maximising off-diagonal compressibility (because they are touched only once). HSS is hardware-friendly: its matrix-vector multiply reduces to one sparse and a sequence of thin-matrix multiplications and can be trained end-to-end with standard optimisers.
Experiments on LLaMA-7B show that targeting only the self-attention projections (1.6 B parameters of Q, K, and V matrices out of a total 7B parameters) suffices to yield large memory savings while retaining comparable state-of-the-art perplexity scores on test samples of the WikiText dataset. For example, with a 30\% sparsity budget and an outer rank of 512, sHSS-RCM achieves a perplexity of 1.64, outperforming dense baselines and classical sparse-plus-SVD variants, while also achieving significant memory savings. 

---
# A survey: Information search time optimization based on RAG (Retrieval Augmentation Generation) chatbot 

**Authors**: Jinesh Patel, Arpit Malhotra, Ajay Pande, Prateek Caire  

**Link**: [PDF](https://arxiv.org/pdf/2601.07838)  

**Abstract**: Retrieval-Augmented Generation (RAG) based chatbots are not only useful for information retrieval through questionanswering but also for making complex decisions based on injected private this http URL present a survey on how much search time can be saved when retrieving complex information within an organization called "X Systems"(a stealth mode company) by using a RAG-based chatbot compared to traditional search methods. We compare the information retrieval time using standard search techniques versus the RAG-based chatbot for the same queries. Our results conclude that RAG-based chatbots not only save time in information retrieval but also optimize the search process effectively. This survey was conducted with a sample of 105 employees across departments, average time spending on information retrieval per query was taken as metric. Comparison shows us, there are average 80-95% improvement on search when use RAG based chatbot than using standard search. 

---
# Photometric Redshift Estimation Using Scaled Ensemble Learning 

**Authors**: Swagata Biswas, Shubhrangshu Ghosh, Avyarthana Ghosh, Yogesh Wadadekar, Abhishek Roy Choudhury, Arijit Mukherjee, Shailesh Deshpande, Arpan Pal  

**Link**: [PDF](https://arxiv.org/pdf/2601.07292)  

**Abstract**: The development of the state-of-the-art telescopic systems capable of performing expansive sky surveys such as the Sloan Digital Sky Survey, Euclid, and the Rubin Observatory's Legacy Survey of Space and Time (LSST) has significantly advanced efforts to refine cosmological models. These advances offer deeper insight into persistent challenges in astrophysics and our understanding of the Universe's evolution. A critical component of this progress is the reliable estimation of photometric redshifts (Pz). To improve the precision and efficiency of such estimations, the application of machine learning (ML) techniques to large-scale astronomical datasets has become essential. This study presents a new ensemble-based ML framework aimed at predicting Pz for faint galaxies and higher redshift ranges, relying solely on optical (grizy) photometric data. The proposed architecture integrates several learning algorithms, including gradient boosting machine, extreme gradient boosting, k-nearest neighbors, and artificial neural networks, within a scaled ensemble structure. By using bagged input data, the ensemble approach delivers improved predictive performance compared to stand-alone models. The framework demonstrates consistent accuracy in estimating redshifts, maintaining strong performance up to z ~ 4. The model is validated using publicly available data from the Hyper Suprime-Cam Strategic Survey Program by the Subaru Telescope. Our results show marked improvements in the precision and reliability of Pz estimation. Furthermore, this approach closely adheres to-and in certain instances exceeds-the benchmarks specified in the LSST Science Requirements Document. Evaluation metrics include catastrophic outlier, bias, and rms. 

---
