# What Voting Rules Actually Do: A Data-Driven Analysis of Multi-Winner Voting 

**Authors**: Joshua Caiata, Ben Armstrong, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2508.06454)  

**Abstract**: Committee-selection problems arise in many contexts and applications, and there has been increasing interest within the social choice research community on identifying which properties are satisfied by different multi-winner voting rules. In this work, we propose a data-driven framework to evaluate how frequently voting rules violate axioms across diverse preference distributions in practice, shifting away from the binary perspective of axiom satisfaction given by worst-case analysis. Using this framework, we analyze the relationship between multi-winner voting rules and their axiomatic performance under several preference distributions. We then show that neural networks, acting as voting rules, can outperform traditional rules in minimizing axiom violations. Our results suggest that data-driven approaches to social choice can inform the design of new voting systems and support the continuation of data-driven research in social choice. 

---
# The Fair Game: Auditing & Debiasing AI Algorithms Over Time 

**Authors**: Debabrota Basu, Udvas Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.06443)  

**Abstract**: An emerging field of AI, namely Fair Machine Learning (ML), aims to quantify different types of bias (also known as unfairness) exhibited in the predictions of ML algorithms, and to design new algorithms to mitigate them. Often, the definitions of bias used in the literature are observational, i.e. they use the input and output of a pre-trained algorithm to quantify a bias under concern. In reality,these definitions are often conflicting in nature and can only be deployed if either the ground truth is known or only in retrospect after deploying the algorithm. Thus,there is a gap between what we want Fair ML to achieve and what it does in a dynamic social environment. Hence, we propose an alternative dynamic mechanism,"Fair Game",to assure fairness in the predictions of an ML algorithm and to adapt its predictions as the society interacts with the algorithm over time. "Fair Game" puts together an Auditor and a Debiasing algorithm in a loop around an ML algorithm. The "Fair Game" puts these two components in a loop by leveraging Reinforcement Learning (RL). RL algorithms interact with an environment to take decisions, which yields new observations (also known as data/feedback) from the environment and in turn, adapts future decisions. RL is already used in algorithms with pre-fixed long-term fairness goals. "Fair Game" provides a unique framework where the fairness goals can be adapted over time by only modifying the auditor and the different biases it quantifies. Thus,"Fair Game" aims to simulate the evolution of ethical and legal frameworks in the society by creating an auditor which sends feedback to a debiasing algorithm deployed around an ML system. This allows us to develop a flexible and adaptive-over-time framework to build Fair ML systems pre- and post-deployment. 

---
# Automated Creation of the Legal Knowledge Graph Addressing Legislation on Violence Against Women: Resource, Methodology and Lessons Learned 

**Authors**: Claudia dAmato, Giuseppe Rubini, Francesco Didio, Donato Francioso, Fatima Zahra Amara, Nicola Fanizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06368)  

**Abstract**: Legal decision-making process requires the availability of comprehensive and detailed legislative background knowledge and up-to-date information on legal cases and related sentences/decisions. Legal Knowledge Graphs (KGs) would be a valuable tool to facilitate access to legal information, to be queried and exploited for the purpose, and to enable advanced reasoning and machine learning applications. Indeed, legal KGs may act as knowledge intensive component to be used by pre-dictive machine learning solutions supporting the decision process of the legal expert. Nevertheless, a few KGs can be found in the legal domain. To fill this gap, we developed a legal KG targeting legal cases of violence against women, along with clear adopted methodologies. Specifically, the paper introduces two complementary approaches for automated legal KG construction; a systematic bottom-up approach, customized for the legal domain, and a new solution leveraging Large Language Models. Starting from legal sentences publicly available from the European Court of Justice, the solutions integrate structured data extraction, ontology development, and semantic enrichment to produce KGs tailored for legal cases involving violence against women. After analyzing and comparing the results of the two approaches, the developed KGs are validated via suitable competency questions. The obtained KG may be impactful for multiple purposes: can improve the accessibility to legal information both to humans and machine, can enable complex queries and may constitute an important knowledge component to be possibly exploited by machine learning tools tailored for predictive justice. 

---
# From Explainable to Explanatory Artificial Intelligence: Toward a New Paradigm for Human-Centered Explanations through Generative AI 

**Authors**: Christian Meske, Justin Brenne, Erdi Uenal, Sabahat Oelcer, Ayseguel Doganguen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06352)  

**Abstract**: Current explainable AI (XAI) approaches prioritize algorithmic transparency and present explanations in abstract, non-adaptive formats that often fail to support meaningful end-user understanding. This paper introduces "Explanatory AI" as a complementary paradigm that leverages generative AI capabilities to serve as explanatory partners for human understanding rather than providers of algorithmic transparency. While XAI reveals algorithmic decision processes for model validation, Explanatory AI addresses contextual reasoning to support human decision-making in sociotechnical contexts. We develop a definition and systematic eight-dimensional conceptual model distinguishing Explanatory AI through narrative communication, adaptive personalization, and progressive disclosure principles. Empirical validation through Rapid Contextual Design methodology with healthcare professionals demonstrates that users consistently prefer context-sensitive, multimodal explanations over technical transparency. Our findings reveal the practical urgency for AI systems designed for human comprehension rather than algorithmic introspection, establishing a comprehensive research agenda for advancing user-centered AI explanation approaches across diverse domains and cultural contexts. 

---
# AntiCheatPT: A Transformer-Based Approach to Cheat Detection in Competitive Computer Games 

**Authors**: Mille Mei Zhen Loo, Gert Luzkov, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.06348)  

**Abstract**: Cheating in online video games compromises the integrity of gaming experiences. Anti-cheat systems, such as VAC (Valve Anti-Cheat), face significant challenges in keeping pace with evolving cheating methods without imposing invasive measures on users' systems. This paper presents AntiCheatPT\_256, a transformer-based machine learning model designed to detect cheating behaviour in Counter-Strike 2 using gameplay data. To support this, we introduce and publicly release CS2CD: A labelled dataset of 795 matches. Using this dataset, 90,707 context windows were created and subsequently augmented to address class imbalance. The transformer model, trained on these windows, achieved an accuracy of 89.17\% and an AUC of 93.36\% on an unaugmented test set. This approach emphasizes reproducibility and real-world applicability, offering a robust baseline for future research in data-driven cheat detection. 

---
# A "good regulator theorem" for embodied agents 

**Authors**: Nathaniel Virgo, Martin Biehl, Manuel Baltieri, Matteo Capucci  

**Link**: [PDF](https://arxiv.org/pdf/2508.06326)  

**Abstract**: In a classic paper, Conant and Ashby claimed that "every good regulator of a system must be a model of that system." Artificial Life has produced many examples of systems that perform tasks with apparently no model in sight; these suggest Conant and Ashby's theorem doesn't easily generalise beyond its restricted setup. Nevertheless, here we show that a similar intuition can be fleshed out in a different way: whenever an agent is able to perform a regulation task, it is possible for an observer to interpret it as having "beliefs" about its environment, which it "updates" in response to sensory input. This notion of belief updating provides a notion of model that is more sophisticated than Conant and Ashby's, as well as a theorem that is more broadly applicable. However, it necessitates a change in perspective, in that the observer plays an essential role in the theory: models are not a mere property of the system but are imposed on it from outside. Our theorem holds regardless of whether the system is regulating its environment in a classic control theory setup, or whether it's regulating its own internal state; the model is of its environment either way. The model might be trivial, however, and this is how the apparent counterexamples are resolved. 

---
# LLM Robustness Leaderboard v1 --Technical report 

**Authors**: Pierre Peigné - Lefebvre, Quentin Feuillade-Montixi, Tom David, Nicolas Miailhe  

**Link**: [PDF](https://arxiv.org/pdf/2508.06296)  

**Abstract**: This technical report accompanies the LLM robustness leaderboard published by PRISM Eval for the Paris AI Action Summit. We introduce PRISM Eval Behavior Elicitation Tool (BET), an AI system performing automated red-teaming through Dynamic Adversarial Optimization that achieves 100% Attack Success Rate (ASR) against 37 of 41 state-of-the-art LLMs. Beyond binary success metrics, we propose a fine-grained robustness metric estimating the average number of attempts required to elicit harmful behaviors, revealing that attack difficulty varies by over 300-fold across models despite universal vulnerability. We introduce primitive-level vulnerability analysis to identify which jailbreaking techniques are most effective for specific hazard categories. Our collaborative evaluation with trusted third parties from the AI Safety Network demonstrates practical pathways for distributed robustness assessment across the community. 

---
# Symmetry breaking for inductive logic programming 

**Authors**: Andrew Cropper, David M. Cerna, Matti Järvisalo  

**Link**: [PDF](https://arxiv.org/pdf/2508.06263)  

**Abstract**: The goal of inductive logic programming is to search for a hypothesis that generalises training data and background knowledge. The challenge is searching vast hypothesis spaces, which is exacerbated because many logically equivalent hypotheses exist. To address this challenge, we introduce a method to break symmetries in the hypothesis space. We implement our idea in answer set programming. Our experiments on multiple domains, including visual reasoning and game playing, show that our approach can reduce solving times from over an hour to just 17 seconds. 

---
# Learning Logical Rules using Minimum Message Length 

**Authors**: Ruben Sharma, Sebastijan Dumančić, Ross D. King, Andrew Cropper  

**Link**: [PDF](https://arxiv.org/pdf/2508.06230)  

**Abstract**: Unifying probabilistic and logical learning is a key challenge in AI. We introduce a Bayesian inductive logic programming approach that learns minimum message length programs from noisy data. Our approach balances hypothesis complexity and data fit through priors, which explicitly favour more general programs, and a likelihood that favours accurate programs. Our experiments on several domains, including game playing and drug design, show that our method significantly outperforms previous methods, notably those that learn minimum description length programs. Our results also show that our approach is data-efficient and insensitive to example balance, including the ability to learn from exclusively positive examples. 

---
# GeoLaux: A Benchmark for Evaluating MLLMs' Geometry Performance on Long-Step Problems Requiring Auxiliary Lines 

**Authors**: Yumeng Fu, Jiayin Zhu, Lingling Zhang, Bo Zhao, Shaoxuan Ma, Yushun Zhang, Yanrui Wu, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06226)  

**Abstract**: Geometry problem solving (GPS) requires models to master diagram comprehension, logical reasoning, knowledge application, numerical computation, and auxiliary line construction. This presents a significant challenge for Multimodal Large Language Models (MLLMs). However, existing benchmarks for evaluating MLLM geometry skills overlook auxiliary line construction and lack fine-grained process evaluation, making them insufficient for assessing MLLMs' long-step reasoning abilities. To bridge these gaps, we present the GeoLaux benchmark, comprising 2,186 geometry problems, incorporating both calculation and proving questions. Notably, the problems require an average of 6.51 reasoning steps, with a maximum of 24 steps, and 41.8% of them need auxiliary line construction. Building on the dataset, we design a novel five-dimensional evaluation strategy assessing answer correctness, process correctness, process quality, auxiliary line impact, and error causes. Extensive experiments on 13 leading MLLMs (including thinking models and non-thinking models) yield three pivotal findings: First, models exhibit substantial performance degradation in extended reasoning steps (nine models demonstrate over 50% performance drop). Second, compared to calculation problems, MLLMs tend to take shortcuts when solving proving problems. Third, models lack auxiliary line awareness, and enhancing this capability proves particularly beneficial for overall geometry reasoning improvement. These findings establish GeoLaux as both a benchmark for evaluating MLLMs' long-step geometric reasoning with auxiliary lines and a guide for capability advancement. Our dataset and code are included in supplementary materials and will be released. 

---
# Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution 

**Authors**: Zailong Tian, Zhuoheng Han, Yanzhe Chen, Haozhe Xu, Xi Yang, richeng xuan, Hongfeng Wang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.06225)  

**Abstract**: Large Language Models (LLMs) are widely used as automated judges, where practical value depends on both accuracy and trustworthy, risk-aware judgments. Existing approaches predominantly focus on accuracy, overlooking the necessity of well-calibrated confidence, which is vital for adaptive and reliable evaluation pipelines. In this work, we advocate a shift from accuracy-centric evaluation to confidence-driven, risk-aware LLM-as-a-Judge systems, emphasizing the necessity of well-calibrated confidence for trustworthy and adaptive evaluation. We systematically identify the **Overconfidence Phenomenon** in current LLM-as-a-Judges, where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment. To quantify this phenomenon, we introduce **TH-Score**, a novel metric measuring confidence-accuracy alignment. Furthermore, we propose **LLM-as-a-Fuser**, an ensemble framework that transforms LLMs into reliable, risk-aware evaluators. Extensive experiments demonstrate that our approach substantially improves calibration and enables adaptive, confidence-driven evaluation pipelines, achieving superior reliability and accuracy compared to existing baselines. 

---
# Retrieval Augmented Large Language Model System for Comprehensive Drug Contraindications 

**Authors**: Byeonghun Bang, Jongsuk Yoon, Dong-Jin Chang, Seho Park, Yong Oh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.06145)  

**Abstract**: The versatility of large language models (LLMs) has been explored across various sectors, but their application in healthcare poses challenges, particularly in the domain of pharmaceutical contraindications where accurate and reliable information is required. This study enhances the capability of LLMs to address contraindications effectively by implementing a Retrieval Augmented Generation (RAG) pipeline. Utilizing OpenAI's GPT-4o-mini as the base model, and the text-embedding-3-small model for embeddings, our approach integrates Langchain to orchestrate a hybrid retrieval system with re-ranking. This system leverages Drug Utilization Review (DUR) data from public databases, focusing on contraindications for specific age groups, pregnancy, and concomitant drug use. The dataset includes 300 question-answer pairs across three categories, with baseline model accuracy ranging from 0.49 to 0.57. Post-integration of the RAG pipeline, we observed a significant improvement in model accuracy, achieving rates of 0.94, 0.87, and 0.89 for contraindications related to age groups, pregnancy, and concomitant drug use, respectively. The results indicate that augmenting LLMs with a RAG framework can substantially reduce uncertainty in prescription and drug intake decisions by providing more precise and reliable drug contraindication information. 

---
# Study of Robust Features in Formulating Guidance for Heuristic Algorithms for Solving the Vehicle Routing Problem 

**Authors**: Bachtiar Herdianto, Romain Billot, Flavien Lucas, Marc Sevaux  

**Link**: [PDF](https://arxiv.org/pdf/2508.06129)  

**Abstract**: The Vehicle Routing Problem (VRP) is a complex optimization problem with numerous real-world applications, mostly solved using metaheuristic algorithms due to its $\mathcal{NP}$-Hard nature. Traditionally, these metaheuristics rely on human-crafted designs developed through empirical studies. However, recent research shows that machine learning methods can be used the structural characteristics of solutions in combinatorial optimization, thereby aiding in designing more efficient algorithms, particularly for solving VRP. Building on this advancement, this study extends the previous research by conducting a sensitivity analysis using multiple classifier models that are capable of predicting the quality of VRP solutions. Hence, by leveraging explainable AI, this research is able to extend the understanding of how these models make decisions. Finally, our findings indicate that while feature importance varies, certain features consistently emerge as strong predictors. Furthermore, we propose a unified framework able of ranking feature impact across different scenarios to illustrate this finding. These insights highlight the potential of feature importance analysis as a foundation for developing a guidance mechanism of metaheuristic algorithms for solving the VRP. 

---
# SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges 

**Authors**: Dewi S. W. Gould, Bruno Mlodozeniec, Samuel F. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2508.06111)  

**Abstract**: Evaluating the capabilities and risks of foundation models is paramount, yet current methods demand extensive domain expertise, hindering their scalability as these models rapidly evolve. We introduce SKATE: a novel evaluation framework in which large language models (LLMs) compete by generating and solving verifiable tasks for one another. Our core insight is to treat evaluation as a game: models act as both task-setters and solvers, incentivized to create questions which highlight their own strengths while exposing others' weaknesses. SKATE offers several key advantages, balancing scalability, open-endedness, and objectivity. It is fully automated, data-free, and scalable, requiring no human input or domain expertise. By using verifiable tasks rather than LLM judges, scoring is objective. Unlike domain-limited programmatically-generated benchmarks (e.g. chess-playing or spatial reasoning), having LLMs creatively pose challenges enables open-ended and scalable evaluation. As a proof of concept, we introduce LLM-set code-output-prediction (COP) challenges as a verifiable and extensible framework in which to test our approach. Using a TrueSkill-based ranking system, we evaluate six frontier LLMs and find that: (1) weaker models can reliably differentiate and score stronger ones, (2) LLM-based systems are capable of self-preferencing behavior, generating questions that align with their own capabilities, and (3) SKATE automatically surfaces fine-grained capability differences between models. Our findings are an important step towards general, scalable evaluation frameworks which can keep pace with LLM progress. 

---
# PanelTR: Zero-Shot Table Reasoning Framework Through Multi-Agent Scientific Discussion 

**Authors**: Yiran Rex Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.06110)  

**Abstract**: Table reasoning, including tabular QA and fact verification, often depends on annotated data or complex data augmentation, limiting flexibility and generalization. LLMs, despite their versatility, often underperform compared to simple supervised models. To approach these issues, we introduce PanelTR, a framework utilizing LLM agent scientists for robust table reasoning through a structured scientific approach. PanelTR's workflow involves agent scientists conducting individual investigations, engaging in self-review, and participating in collaborative peer-review discussions. This process, driven by five scientist personas, enables semantic-level transfer without relying on data augmentation or parametric optimization. Experiments across four benchmarks show that PanelTR outperforms vanilla LLMs and rivals fully supervised models, all while remaining independent of training data. Our findings indicate that structured scientific methodology can effectively handle complex tasks beyond table reasoning with flexible semantic understanding in a zero-shot context. 

---
# Aggregate-Combine-Readout GNNs Are More Expressive Than Logic C2 

**Authors**: Stan P Hauke, Przemysław Andrzej Wałęga  

**Link**: [PDF](https://arxiv.org/pdf/2508.06091)  

**Abstract**: In recent years, there has been growing interest in understanding the expressive power of graph neural networks (GNNs) by relating them to logical languages. This research has been been initialised by an influential result of Barceló et al. (2020), who showed that the graded modal logic (or a guarded fragment of the logic C2), characterises the logical expressiveness of aggregate-combine GNNs. As a ``challenging open problem'' they left the question whether full C2 characterises the logical expressiveness of aggregate-combine-readout GNNs. This question has remained unresolved despite several attempts. In this paper, we solve the above open problem by proving that the logical expressiveness of aggregate-combine-readout GNNs strictly exceeds that of C2. This result holds over both undirected and directed graphs. Beyond its implications for GNNs, our work also leads to purely logical insights on the expressive power of infinitary logics. 

---
# ME$^3$-BEV: Mamba-Enhanced Deep Reinforcement Learning for End-to-End Autonomous Driving with BEV-Perception 

**Authors**: Siyi Lu, Run Liu, Dongsheng Yang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.06074)  

**Abstract**: Autonomous driving systems face significant challenges in perceiving complex environments and making real-time decisions. Traditional modular approaches, while offering interpretability, suffer from error propagation and coordination issues, whereas end-to-end learning systems can simplify the design but face computational bottlenecks. This paper presents a novel approach to autonomous driving using deep reinforcement learning (DRL) that integrates bird's-eye view (BEV) perception for enhanced real-time decision-making. We introduce the \texttt{Mamba-BEV} model, an efficient spatio-temporal feature extraction network that combines BEV-based perception with the Mamba framework for temporal feature modeling. This integration allows the system to encode vehicle surroundings and road features in a unified coordinate system and accurately model long-range dependencies. Building on this, we propose the \texttt{ME$^3$-BEV} framework, which utilizes the \texttt{Mamba-BEV} model as a feature input for end-to-end DRL, achieving superior performance in dynamic urban driving scenarios. We further enhance the interpretability of the model by visualizing high-dimensional features through semantic segmentation, providing insight into the learned representations. Extensive experiments on the CARLA simulator demonstrate that \texttt{ME$^3$-BEV} outperforms existing models across multiple metrics, including collision rate and trajectory accuracy, offering a promising solution for real-time autonomous driving. 

---
# A Generic Complete Anytime Beam Search for Optimal Decision Tree 

**Authors**: Harold Silvère Kiossou, Siegfried Nijssen, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2508.06064)  

**Abstract**: Finding an optimal decision tree that minimizes classification error is known to be NP-hard. While exact algorithms based on MILP, CP, SAT, or dynamic programming guarantee optimality, they often suffer from poor anytime behavior -- meaning they struggle to find high-quality decision trees quickly when the search is stopped before completion -- due to unbalanced search space exploration. To address this, several anytime extensions of exact methods have been proposed, such as LDS-DL8.5, Top-k-DL8.5, and Blossom, but they have not been systematically compared, making it difficult to assess their relative effectiveness. In this paper, we propose CA-DL8.5, a generic, complete, and anytime beam search algorithm that extends the DL8.5 framework and unifies some existing anytime strategies. In particular, CA-DL8.5 generalizes previous approaches LDS-DL8.5 and Top-k-DL8.5, by allowing the integration of various heuristics and relaxation mechanisms through a modular design. The algorithm reuses DL8.5's efficient branch-and-bound pruning and trie-based caching, combined with a restart-based beam search that gradually relaxes pruning criteria to improve solution quality over time. Our contributions are twofold: (1) We introduce this new generic framework for exact and anytime decision tree learning, enabling the incorporation of diverse heuristics and search strategies; (2) We conduct a rigorous empirical comparison of several instantiations of CA-DL8.5 -- based on Purity, Gain, Discrepancy, and Top-k heuristics -- using an anytime evaluation metric called the primal gap integral. Experimental results on standard classification benchmarks show that CA-DL8.5 using LDS (limited discrepancy) consistently provides the best anytime performance, outperforming both other CA-DL8.5 variants and the Blossom algorithm while maintaining completeness and optimality guarantees. 

---
# Don't Forget Imagination! 

**Authors**: Evgenii E. Vityaev, Andrei Mantsivoda  

**Link**: [PDF](https://arxiv.org/pdf/2508.06062)  

**Abstract**: Cognitive imagination is a type of imagination that plays a key role in human thinking. It is not a ``picture-in-the-head'' imagination. It is a faculty to mentally visualize coherent and holistic systems of concepts and causal links that serve as semantic contexts for reasoning, decision making and prediction. Our position is that the role of cognitive imagination is still greatly underestimated, and this creates numerous problems and diminishes the current capabilities of AI. For instance, when reasoning, humans rely on imaginary contexts to retrieve background info. They also constantly return to the context for semantic verification that their reasoning is still reasonable. Thus, reasoning without imagination is blind. This paper is a call for greater attention to cognitive imagination as the next promising breakthrough in artificial intelligence. As an instrument for simulating cognitive imagination, we propose semantic models -- a new approach to mathematical models that can learn, like neural networks, and are based on probabilistic causal relationships. Semantic models can simulate cognitive imagination because they ensure the consistency of imaginary contexts and implement a glass-box approach that allows the context to be manipulated as a holistic and coherent system of interrelated facts glued together with causal relations. 

---
# LLMs for Resource Allocation: A Participatory Budgeting Approach to Inferring Preferences 

**Authors**: Sankarshan Damle, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2508.06060)  

**Abstract**: Large Language Models (LLMs) are increasingly expected to handle complex decision-making tasks, yet their ability to perform structured resource allocation remains underexplored. Evaluating their reasoning is also difficult due to data contamination and the static nature of existing benchmarks. We present a dual-purpose framework leveraging Participatory Budgeting (PB) both as (i) a practical setting for LLM-based resource allocation and (ii) an adaptive benchmark for evaluating their reasoning capabilities. We task LLMs with selecting project subsets under feasibility (e.g., budget) constraints via three prompting strategies: greedy selection, direct optimization, and a hill-climbing-inspired refinement. We benchmark LLMs' allocations against a utility-maximizing oracle. Interestingly, we also test whether LLMs can infer structured preferences from natural-language voter input or metadata, without explicit votes. By comparing allocations based on inferred preferences to those from ground-truth votes, we evaluate LLMs' ability to extract preferences from open-ended input. Our results underscore the role of prompt design and show that LLMs hold promise for mechanism design with unstructured inputs. 

---
# Society of Mind Meets Real-Time Strategy: A Hierarchical Multi-Agent Framework for Strategic Reasoning 

**Authors**: Daechul Ahn, San Kim, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06042)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated impressive action sequence prediction capabilities but often struggle with dynamic, long-horizon tasks such as real-time strategic games. In a game such as StarCraftII (SC2), agents need to manage resource constraints and adapt to evolving battlefield situations in a partially observable environment. This often overwhelms exisiting LLM-based approaches. To address these challenges, we propose a hierarchical multi-agent framework that employs specialized imitation learning agents under a meta-controller called Strategic Planner (SP). By expert demonstrations, each specialized agent learns a distinctive strategy, such as aerial support or defensive maneuvers, and produces coherent, structured multistep action sequences. The SP then orchestrates these proposals into a single, environmentally adaptive plan that ensures local decisions aligning with long-term strategies. We call this HIMA (Hierarchical Imitation Multi-Agent). We also present TEXTSCII-ALL, a comprehensive SC2 testbed that encompasses all race match combinations in SC2. Our empirical results show that HIMA outperforms state of the arts in strategic clarity, adaptability, and computational efficiency, underscoring the potential of combining specialized imitation modules with meta-level orchestration to develop more robust, general-purpose AI agents. 

---
# Mediator-Guided Multi-Agent Collaboration among Open-Source Models for Medical Decision-Making 

**Authors**: Kaitao Chen, Mianxin Liu, Daoming Zong, Chaoyue Ding, Shaohao Rui, Yankai Jiang, Mu Zhou, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05996)  

**Abstract**: Complex medical decision-making involves cooperative workflows operated by different clinicians. Designing AI multi-agent systems can expedite and augment human-level clinical decision-making. Existing multi-agent researches primarily focus on language-only tasks, yet their extension to multimodal scenarios remains challenging. A blind combination of diverse vision-language models (VLMs) can amplify an erroneous outcome interpretation. VLMs in general are less capable in instruction following and importantly self-reflection, compared to large language models (LLMs) of comparable sizes. This disparity largely constrains VLMs' ability in cooperative workflows. In this study, we propose MedOrch, a mediator-guided multi-agent collaboration framework for medical multimodal decision-making. MedOrch employs an LLM-based mediator agent that enables multiple VLM-based expert agents to exchange and reflect on their outputs towards collaboration. We utilize multiple open-source general-purpose and domain-specific VLMs instead of costly GPT-series models, revealing the strength of heterogeneous models. We show that the collaboration within distinct VLM-based agents can surpass the capabilities of any individual agent. We validate our approach on five medical vision question answering benchmarks, demonstrating superior collaboration performance without model training. Our findings underscore the value of mediator-guided multi-agent collaboration in advancing medical multimodal intelligence. Our code will be made publicly available. 

---
# Planning Agents on an Ego-Trip: Leveraging Hybrid Ego-Graph Ensembles for Improved Tool Retrieval in Enterprise Task Planning 

**Authors**: Sahil Bansal, Sai Shruthi Sistla, Aarti Arikatala, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2508.05888)  

**Abstract**: Effective tool retrieval is essential for AI agents to select from a vast array of tools when identifying and planning actions in the context of complex user queries. Despite its central role in planning, this aspect remains underexplored in the literature. Traditional approaches rely primarily on similarities between user queries and tool descriptions, which significantly limits retrieval accuracy, specifically when handling multi-step user requests. To address these limitations, we propose a Knowledge Graph (KG)-based tool retrieval framework that captures the semantic relationships between tools and their functional dependencies. Our retrieval algorithm leverages ensembles of 1-hop ego tool graphs to model direct and indirect connections between tools, enabling more comprehensive and contextual tool selection for multi-step tasks. We evaluate our approach on a synthetically generated internal dataset across six defined user classes, extending previous work on coherent dialogue synthesis and too retrieval benchmarks. Results demonstrate that our tool graph-based method achieves 91.85% tool coverage on the micro-average Complete Recall metric, compared to 89.26% for re-ranked semantic-lexical hybrid retrieval, the strongest non-KG baseline in our experiments. These findings support our hypothesis that the structural information in the KG provides complementary signals to pure similarity matching, particularly for queries requiring sequential tool composition. 

---
# Safety of Embodied Navigation: A Survey 

**Authors**: Zixia Wang, Jia Hu, Ronghui Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05855)  

**Abstract**: As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency. 

---
# Holistic Explainable AI (H-XAI): Extending Transparency Beyond Developers in AI-Driven Decision Making 

**Authors**: Kausik Lakkaraju, Siva Likitha Valluru, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.05792)  

**Abstract**: Current eXplainable AI (XAI) methods largely serve developers, often focusing on justifying model outputs rather than supporting diverse stakeholder needs. A recent shift toward Evaluative AI reframes explanation as a tool for hypothesis testing, but still focuses primarily on operational organizations. We introduce Holistic-XAI (H-XAI), a unified framework that integrates causal rating methods with traditional XAI methods to support explanation as an interactive, multi-method process. H-XAI allows stakeholders to ask a series of questions, test hypotheses, and compare model behavior against automatically constructed random and biased baselines. It combines instance-level and global explanations, adapting to each stakeholder's goals, whether understanding individual decisions, assessing group-level bias, or evaluating robustness under perturbations. We demonstrate the generality of our approach through two case studies spanning six scenarios: binary credit risk classification and financial time-series forecasting. H-XAI fills critical gaps left by existing XAI methods by combining causal ratings and post-hoc explanations to answer stakeholder-specific questions at both the individual decision level and the overall model level. 

---
# Whither symbols in the era of advanced neural networks? 

**Authors**: Thomas L. Griffiths, Brenden M. Lake, R. Thomas McCoy, Ellie Pavlick, Taylor W. Webb  

**Link**: [PDF](https://arxiv.org/pdf/2508.05776)  

**Abstract**: Some of the strongest evidence that human minds should be thought about in terms of symbolic systems has been the way they combine ideas, produce novelty, and learn quickly. We argue that modern neural networks -- and the artificial intelligence systems built upon them -- exhibit similar abilities. This undermines the argument that the cognitive processes and representations used by human minds are symbolic, although the fact that these neural networks are typically trained on data generated by symbolic systems illustrates that such systems play an important role in characterizing the abstract problems that human minds have to solve. This argument leads us to offer a new agenda for research on the symbolic basis of human thought. 

---
# A Framework for Inherently Safer AGI through Language-Mediated Active Inference 

**Authors**: Bo Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.05766)  

**Abstract**: This paper proposes a novel framework for developing safe Artificial General Intelligence (AGI) by combining Active Inference principles with Large Language Models (LLMs). We argue that traditional approaches to AI safety, focused on post-hoc interpretability and reward engineering, have fundamental limitations. We present an architecture where safety guarantees are integrated into the system's core design through transparent belief representations and hierarchical value alignment. Our framework leverages natural language as a medium for representing and manipulating beliefs, enabling direct human oversight while maintaining computational tractability. The architecture implements a multi-agent system where agents self-organize according to Active Inference principles, with preferences and safety constraints flowing through hierarchical Markov blankets. We outline specific mechanisms for ensuring safety, including: (1) explicit separation of beliefs and preferences in natural language, (2) bounded rationality through resource-aware free energy minimization, and (3) compositional safety through modular agent structures. The paper concludes with a research agenda centered on the Abstraction and Reasoning Corpus (ARC) benchmark, proposing experiments to validate our framework's safety properties. Our approach offers a path toward AGI development that is inherently safer, rather than retrofitted with safety measures. 

---
# InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization 

**Authors**: Yuhang Liu, Zeyu Liu, Shuanghe Zhu, Pengxiang Li, Congkai Xie, Jiasheng Wang, Xueyu Hu, Xiaotian Han, Jianbo Yuan, Xinyao Wang, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05731)  

**Abstract**: The emergence of Multimodal Large Language Models (MLLMs) has propelled the development of autonomous agents that operate on Graphical User Interfaces (GUIs) using pure visual input. A fundamental challenge is robustly grounding natural language instructions. This requires a precise spatial alignment, which accurately locates the coordinates of each element, and, more critically, a correct semantic alignment, which matches the instructions to the functionally appropriate UI element. Although Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be effective at improving spatial alignment for these MLLMs, we find that inefficient exploration bottlenecks semantic alignment, which prevent models from learning difficult semantic associations. To address this exploration problem, we present Adaptive Exploration Policy Optimization (AEPO), a new policy optimization framework. AEPO employs a multi-answer generation strategy to enforce broader exploration, which is then guided by a theoretically grounded Adaptive Exploration Reward (AER) function derived from first principles of efficiency eta=U/C. Our AEPO-trained models, InfiGUI-G1-3B and InfiGUI-G1-7B, establish new state-of-the-art results across multiple challenging GUI grounding benchmarks, achieving significant relative improvements of up to 9.0% against the naive RLVR baseline on benchmarks designed to test generalization and semantic understanding. Resources are available at this https URL. 

---
# WGAST: Weakly-Supervised Generative Network for Daily 10 m Land Surface Temperature Estimation via Spatio-Temporal Fusion 

**Authors**: Sofiane Bouaziz, Adel Hafiane, Raphael Canals, Rachid Nedjai  

**Link**: [PDF](https://arxiv.org/pdf/2508.06485)  

**Abstract**: Urbanization, climate change, and agricultural stress are increasing the demand for precise and timely environmental monitoring. Land Surface Temperature (LST) is a key variable in this context and is retrieved from remote sensing satellites. However, these systems face a trade-off between spatial and temporal resolution. While spatio-temporal fusion methods offer promising solutions, few have addressed the estimation of daily LST at 10 m resolution. In this study, we present WGAST, a Weakly-Supervised Generative Network for Daily 10 m LST Estimation via Spatio-Temporal Fusion of Terra MODIS, Landsat 8, and Sentinel-2. WGAST is the first end-to-end deep learning framework designed for this task. It adopts a conditional generative adversarial architecture, with a generator composed of four stages: feature extraction, fusion, LST reconstruction, and noise suppression. The first stage employs a set of encoders to extract multi-level latent representations from the inputs, which are then fused in the second stage using cosine similarity, normalization, and temporal attention mechanisms. The third stage decodes the fused features into high-resolution LST, followed by a Gaussian filter to suppress high-frequency noise. Training follows a weakly supervised strategy based on physical averaging principles and reinforced by a PatchGAN discriminator. Experiments demonstrate that WGAST outperforms existing methods in both quantitative and qualitative evaluations. Compared to the best-performing baseline, on average, WGAST reduces RMSE by 17.18% and improves SSIM by 11.00%. Furthermore, WGAST is robust to cloud-induced LST and effectively captures fine-scale thermal patterns, as validated against 33 ground-based sensors. The code is available at this https URL. 

---
# Post-training for Efficient Communication via Convention Formation 

**Authors**: Yilun Hua, Evan Wang, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06482)  

**Abstract**: Humans communicate with increasing efficiency in multi-turn interactions, by adapting their language and forming ad-hoc conventions. In contrast, prior work shows that LLMs do not naturally show this behavior. We develop a post-training process to develop this ability through targeted fine-tuning on heuristically identified demonstrations of convention formation. We evaluate with two new benchmarks focused on this capability. First, we design a focused, cognitively-motivated interaction benchmark that consistently elicits strong convention formation trends in humans. Second, we create a new document-grounded reference completion task that reflects in-the-wild convention formation behavior. Our studies show significantly improved convention formation abilities in post-trained LLMs across the two evaluation methods. 

---
# Intuition emerges in Maximum Caliber models at criticality 

**Authors**: Lluís Arola-Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2508.06477)  

**Abstract**: Whether large predictive models merely parrot their training data or produce genuine insight lacks a physical explanation. This work reports a primitive form of intuition that emerges as a metastable phase of learning that critically balances next-token prediction against future path-entropy. The intuition mechanism is discovered via mind-tuning, the minimal principle that imposes Maximum Caliber in predictive models with a control temperature-like parameter $\lambda$. Training on random walks in deterministic mazes reveals a rich phase diagram: imitation (low $\lambda$), rule-breaking hallucination (high $\lambda$), and a fragile in-between window exhibiting strong protocol-dependence (hysteresis) and multistability, where models spontaneously discover novel goal-directed strategies. These results are captured by an effective low-dimensional theory and frame intuition as an emergent property at the critical balance between memorizing what is and wondering what could be. 

---
# ScamAgents: How AI Agents Can Simulate Human-Level Scam Calls 

**Authors**: Sanket Badhe  

**Link**: [PDF](https://arxiv.org/pdf/2508.06457)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive fluency and reasoning capabilities, but their potential for misuse has raised growing concern. In this paper, we present ScamAgent, an autonomous multi-turn agent built on top of LLMs, capable of generating highly realistic scam call scripts that simulate real-world fraud scenarios. Unlike prior work focused on single-shot prompt misuse, ScamAgent maintains dialogue memory, adapts dynamically to simulated user responses, and employs deceptive persuasion strategies across conversational turns. We show that current LLM safety guardrails, including refusal mechanisms and content filters, are ineffective against such agent-based threats. Even models with strong prompt-level safeguards can be bypassed when prompts are decomposed, disguised, or delivered incrementally within an agent framework. We further demonstrate the transformation of scam scripts into lifelike voice calls using modern text-to-speech systems, completing a fully automated scam pipeline. Our findings highlight an urgent need for multi-turn safety auditing, agent-level control frameworks, and new methods to detect and disrupt conversational deception powered by generative AI. 

---
# Text Embedded Swin-UMamba for DeepLesion Segmentation 

**Authors**: Ruida Cheng, Tejas Sudharshan Mathai, Pritam Mukherjee, Benjamin Hou, Qingqing Zhu, Zhiyong Lu, Matthew McAuliffe, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2508.06453)  

**Abstract**: Segmentation of lesions on CT enables automatic measurement for clinical assessment of chronic diseases (e.g., lymphoma). Integrating large language models (LLMs) into the lesion segmentation workflow offers the potential to combine imaging features with descriptions of lesion characteristics from the radiology reports. In this study, we investigate the feasibility of integrating text into the Swin-UMamba architecture for the task of lesion segmentation. The publicly available ULS23 DeepLesion dataset was used along with short-form descriptions of the findings from the reports. On the test dataset, a high Dice Score of 82% and low Hausdorff distance of 6.58 (pixels) was obtained for lesion segmentation. The proposed Text-Swin-UMamba model outperformed prior approaches: 37% improvement over the LLM-driven LanGuideMedSeg model (p < 0.001),and surpassed the purely image-based xLSTM-UNet and nnUNet models by 1.74% and 0.22%, respectively. The dataset and code can be accessed at this https URL 

---
# Echoes of Automation: The Increasing Use of LLMs in Newsmaking 

**Authors**: Abolfazl Ansari, Delvin Ce Zhang, Nafis Irtiza Tripto, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.06445)  

**Abstract**: The rapid rise of Generative AI (GenAI), particularly LLMs, poses concerns for journalistic integrity and authorship. This study examines AI-generated content across over 40,000 news articles from major, local, and college news media, in various media formats. Using three advanced AI-text detectors (e.g., Binoculars, Fast-Detect GPT, and GPTZero), we find substantial increase of GenAI use in recent years, especially in local and college news. Sentence-level analysis reveals LLMs are often used in the introduction of news, while conclusions usually written manually. Linguistic analysis shows GenAI boosts word richness and readability but lowers formality, leading to more uniform writing styles, particularly in local media. 

---
# Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages 

**Authors**: Andrea Nasuto, Stefano Maria Iacus, Francisco Rowe, Devika Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.06435)  

**Abstract**: Large language models (LLMs) are transforming social-science research by enabling scalable, precise analysis. Their adaptability raises the question of whether knowledge acquired through fine-tuning in a few languages can transfer to unseen languages that only appeared during pre-training. To examine this, we fine-tune lightweight LLaMA 3.2-3B models on monolingual, bilingual, or multilingual data sets to classify immigration-related tweets from X/Twitter across 13 languages, a domain characterised by polarised, culturally specific discourse. We evaluate whether minimal language-specific fine-tuning enables cross-lingual topic detection and whether adding targeted languages corrects pre-training biases. Results show that LLMs fine-tuned in one or two languages can reliably classify immigration-related content in unseen languages. However, identifying whether a tweet expresses a pro- or anti-immigration stance benefits from multilingual fine-tuning. Pre-training bias favours dominant languages, but even minimal exposure to under-represented languages during fine-tuning (as little as $9.62\times10^{-11}$ of the original pre-training token volume) yields significant gains. These findings challenge the assumption that cross-lingual mastery requires extensive multilingual training: limited language coverage suffices for topic-level generalisation, and structural biases can be corrected with lightweight interventions. By releasing 4-bit-quantised, LoRA fine-tuned models, we provide an open-source, reproducible alternative to proprietary LLMs that delivers 35 times faster inference at just 0.00000989% of the dollar cost of the OpenAI GPT-4o model, enabling scalable, inclusive research. 

---
# CLIPin: A Non-contrastive Plug-in to CLIP for Multimodal Semantic Alignment 

**Authors**: Shengzhu Yang, Jiawei Du, Shuai Lu, Weihang Zhang, Ningli Wang, Huiqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06434)  

**Abstract**: Large-scale natural image-text datasets, especially those automatically collected from the web, often suffer from loose semantic alignment due to weak supervision, while medical datasets tend to have high cross-modal correlation but low content diversity. These properties pose a common challenge for contrastive language-image pretraining (CLIP): they hinder the model's ability to learn robust and generalizable representations. In this work, we propose CLIPin, a unified non-contrastive plug-in that can be seamlessly integrated into CLIP-style architectures to improve multimodal semantic alignment, providing stronger supervision and enhancing alignment robustness. Furthermore, two shared pre-projectors are designed for image and text modalities respectively to facilitate the integration of contrastive and non-contrastive learning in a parameter-compromise manner. Extensive experiments on diverse downstream tasks demonstrate the effectiveness and generality of CLIPin as a plug-and-play component compatible with various contrastive frameworks. Code is available at this https URL. 

---
# Memp: Exploring Agent Procedural Memory 

**Authors**: Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06433)  

**Abstract**: Large Language Models (LLMs) based agents excel at diverse tasks, yet they suffer from brittle procedural memory that is manually engineered or entangled in static parameters. In this work, we investigate strategies to endow agents with a learnable, updatable, and lifelong procedural memory. We propose Memp that distills past agent trajectories into both fine-grained, step-by-step instructions and higher-level, script-like abstractions, and explore the impact of different strategies for Build, Retrieval, and Update of procedural memory. Coupled with a dynamic regimen that continuously updates, corrects, and deprecates its contents, this repository evolves in lockstep with new experience. Empirical evaluation on TravelPlanner and ALFWorld shows that as the memory repository is refined, agents achieve steadily higher success rates and greater efficiency on analogous tasks. Moreover, procedural memory built from a stronger model retains its value: migrating the procedural memory to a weaker model yields substantial performance gains. 

---
# SPARSE Data, Rich Results: Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation 

**Authors**: Guido Manni, Clemente Lauretti, Loredana Zollo, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2508.06429)  

**Abstract**: Deep learning has revolutionized medical imaging, but its effectiveness is severely limited by insufficient labeled training data. This paper introduces a novel GAN-based semi-supervised learning framework specifically designed for low labeled-data regimes, evaluated across settings with 5 to 50 labeled samples per class. Our approach integrates three specialized neural networks -- a generator for class-conditioned image translation, a discriminator for authenticity assessment and classification, and a dedicated classifier -- within a three-phase training framework. The method alternates between supervised training on limited labeled data and unsupervised learning that leverages abundant unlabeled images through image-to-image translation rather than generation from noise. We employ ensemble-based pseudo-labeling that combines confidence-weighted predictions from the discriminator and classifier with temporal consistency through exponential moving averaging, enabling reliable label estimation for unlabeled data. Comprehensive evaluation across eleven MedMNIST datasets demonstrates that our approach achieves statistically significant improvements over six state-of-the-art GAN-based semi-supervised methods, with particularly strong performance in the extreme 5-shot setting where the scarcity of labeled data is most challenging. The framework maintains its superiority across all evaluated settings (5, 10, 20, and 50 shots per class). Our approach offers a practical solution for medical imaging applications where annotation costs are prohibitive, enabling robust classification performance even with minimal labeled data. Code is available at this https URL. 

---
# Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation 

**Authors**: Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Hengtao Shen, Jingkuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.06426)  

**Abstract**: Generalist robot policies trained on large-scale datasets such as Open X-Embodiment (OXE) demonstrate strong performance across a wide range of tasks. However, they often struggle to generalize beyond the distribution of their training data. In this paper, we investigate the underlying cause of this limited generalization capability. We identify shortcut learning -- the reliance on task-irrelevant features -- as a key impediment to generalization. Through comprehensive theoretical and empirical analysis, we uncover two primary contributors to shortcut learning: (1) limited diversity within individual sub-datasets, and (2) significant distributional disparities across sub-datasets, leading to dataset fragmentation. These issues arise from the inherent structure of large-scale datasets like OXE, which are typically composed of multiple sub-datasets collected independently across varied environments and embodiments. Our findings provide critical insights into dataset collection strategies that can reduce shortcut learning and enhance the generalization ability of generalist robot policies. Moreover, in scenarios where acquiring new large-scale data is impractical, we demonstrate that carefully selected robotic data augmentation strategies can effectively reduce shortcut learning in existing offline datasets, thereby improving generalization capabilities of generalist robot policies, e.g., $\pi_0$, in both simulation and real-world environments. More information at this https URL. 

---
# Dimensional Characterization and Pathway Modeling for Catastrophic AI Risks 

**Authors**: Ze Shen Chin  

**Link**: [PDF](https://arxiv.org/pdf/2508.06411)  

**Abstract**: Although discourse around the risks of Artificial Intelligence (AI) has grown, it often lacks a comprehensive, multidimensional framework, and concrete causal pathways mapping hazard to harm. This paper aims to bridge this gap by examining six commonly discussed AI catastrophic risks: CBRN, cyber offense, sudden loss of control, gradual loss of control, environmental risk, and geopolitical risk. First, we characterize these risks across seven key dimensions, namely intent, competency, entity, polarity, linearity, reach, and order. Next, we conduct risk pathway modeling by mapping step-by-step progressions from the initial hazard to the resulting harms. The dimensional approach supports systematic risk identification and generalizable mitigation strategies, while risk pathway models help identify scenario-specific interventions. Together, these methods offer a more structured and actionable foundation for managing catastrophic AI risks across the value chain. 

---
# A Classification-Aware Super-Resolution Framework for Ship Targets in SAR Imagery 

**Authors**: Ch Muhammad Awais, Marco Reggiannini, Davide Moroni, Oktay Karakus  

**Link**: [PDF](https://arxiv.org/pdf/2508.06407)  

**Abstract**: High-resolution imagery plays a critical role in improving the performance of visual recognition tasks such as classification, detection, and segmentation. In many domains, including remote sensing and surveillance, low-resolution images can limit the accuracy of automated analysis. To address this, super-resolution (SR) techniques have been widely adopted to attempt to reconstruct high-resolution images from low-resolution inputs. Related traditional approaches focus solely on enhancing image quality based on pixel-level metrics, leaving the relationship between super-resolved image fidelity and downstream classification performance largely underexplored. This raises a key question: can integrating classification objectives directly into the super-resolution process further improve classification accuracy? In this paper, we try to respond to this question by investigating the relationship between super-resolution and classification through the deployment of a specialised algorithmic strategy. We propose a novel methodology that increases the resolution of synthetic aperture radar imagery by optimising loss functions that account for both image quality and classification performance. Our approach improves image quality, as measured by scientifically ascertained image quality indicators, while also enhancing classification accuracy. 

---
# A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges 

**Authors**: Andrew Brown, Muhammad Roman, Barry Devereux  

**Link**: [PDF](https://arxiv.org/pdf/2508.06401)  

**Abstract**: This systematic review of the research literature on retrieval-augmented generation (RAG) provides a focused analysis of the most highly cited studies published between 2020 and May 2025. A total of 128 articles met our inclusion criteria. The records were retrieved from ACM Digital Library, IEEE Xplore, Scopus, ScienceDirect, and the Digital Bibliography and Library Project (DBLP). RAG couples a neural retriever with a generative language model, grounding output in up-to-date, non-parametric memory while retaining the semantic generalisation stored in model weights. Guided by the PRISMA 2020 framework, we (i) specify explicit inclusion and exclusion criteria based on citation count and research questions, (ii) catalogue datasets, architectures, and evaluation practices, and (iii) synthesise empirical evidence on the effectiveness and limitations of RAG. To mitigate citation-lag bias, we applied a lower citation-count threshold to papers published in 2025 so that emerging breakthroughs with naturally fewer citations were still captured. This review clarifies the current research landscape, highlights methodological gaps, and charts priority directions for future research. 

---
# Robust Target Speaker Diarization and Separation via Augmented Speaker Embedding Sampling 

**Authors**: Md Asif Jalal, Luca Remaggi, Vasileios Moschopoulos, Thanasis Kotsiopoulos, Vandana Rajan, Karthikeyan Saravanan, Anastasis Drosou, Junho Heo, Hyuk Oh, Seokyeong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.06393)  

**Abstract**: Traditional speech separation and speaker diarization approaches rely on prior knowledge of target speakers or a predetermined number of participants in audio signals. To address these limitations, recent advances focus on developing enrollment-free methods capable of identifying targets without explicit speaker labeling. This work introduces a new approach to train simultaneous speech separation and diarization using automatic identification of target speaker embeddings, within mixtures. Our proposed model employs a dual-stage training pipeline designed to learn robust speaker representation features that are resilient to background noise interference. Furthermore, we present an overlapping spectral loss function specifically tailored for enhancing diarization accuracy during overlapped speech frames. Experimental results show significant performance gains compared to the current SOTA baseline, achieving 71% relative improvement in DER and 69% in cpWER. 

---
# Identity Increases Stability in Neural Cellular Automata 

**Authors**: James Stovold  

**Link**: [PDF](https://arxiv.org/pdf/2508.06389)  

**Abstract**: Neural Cellular Automata (NCAs) offer a way to study the growth of two-dimensional artificial organisms from a single seed cell. From the outset, NCA-grown organisms have had issues with stability, their natural boundary often breaking down and exhibiting tumour-like growth or failing to maintain the expected shape. In this paper, we present a method for improving the stability of NCA-grown organisms by introducing an 'identity' layer with simple constraints during training.
Results show that NCAs grown in close proximity are more stable compared with the original NCA model. Moreover, only a single identity value is required to achieve this increase in stability. We observe emergent movement from the stable organisms, with increasing prevalence for models with multiple identity values.
This work lays the foundation for further study of the interaction between NCA-grown organisms, paving the way for studying social interaction at a cellular level in artificial organisms. 

---
# End-to-End Text-to-SQL with Dataset Selection: Leveraging LLMs for Adaptive Query Generation 

**Authors**: Anurag Tripathi, Vaibhav Patle, Abhinav Jain, Ayush Pundir, Sairam Menon, Ajeet Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.06387)  

**Abstract**: Text-to-SQL bridges the gap between natural language and structured database language, thus allowing non-technical users to easily query databases. Traditional approaches model text-to-SQL as a direct translation task, where a given Natural Language Query (NLQ) is mapped to an SQL command. Recent advances in large language models (LLMs) have significantly improved translation accuracy, however, these methods all require that the target database is pre-specified. This becomes problematic in scenarios with multiple extensive databases, where identifying the correct database becomes a crucial yet overlooked step. In this paper, we propose a three-stage end-to-end text-to-SQL framework to identify the user's intended database before generating SQL queries. Our approach leverages LLMs and prompt engineering to extract implicit information from natural language queries (NLQs) in the form of a ruleset. We then train a large db\_id prediction model, which includes a RoBERTa-based finetuned encoder, to predict the correct Database identifier (db\_id) based on both the NLQ and the LLM-generated rules. Finally, we refine the generated SQL by using critic agents to correct errors. Experimental results demonstrate that our framework outperforms the current state-of-the-art models in both database intent prediction and SQL generation accuracy. 

---
# SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models 

**Authors**: Han Yin, Yafeng Chen, Chong Deng, Luyao Cheng, Hui Wang, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06372)  

**Abstract**: The Speaker Diarization and Recognition (SDR) task aims to predict "who spoke when and what" within an audio clip, which is a crucial task in various real-world multi-speaker scenarios such as meeting transcription and dialogue systems. Existing SDR systems typically adopt a cascaded framework, combining multiple modules such as speaker diarization (SD) and automatic speech recognition (ASR). The cascaded systems suffer from several limitations, such as error propagation, difficulty in handling overlapping speech, and lack of joint optimization for exploring the synergy between SD and ASR tasks. To address these limitations, we introduce SpeakerLM, a unified multimodal large language model for SDR that jointly performs SD and ASR in an end-to-end manner. Moreover, to facilitate diverse real-world scenarios, we incorporate a flexible speaker registration mechanism into SpeakerLM, enabling SDR under different speaker registration settings. SpeakerLM is progressively developed with a multi-stage training strategy on large-scale real data. Extensive experiments show that SpeakerLM demonstrates strong data scaling capability and generalizability, outperforming state-of-the-art cascaded baselines on both in-domain and out-of-domain public SDR benchmarks. Furthermore, experimental results show that the proposed speaker registration mechanism effectively ensures robust SDR performance of SpeakerLM across diverse speaker registration conditions and varying numbers of registered speakers. 

---
# ActivityDiff: A diffusion model with Positive and Negative Activity Guidance for De Novo Drug Design 

**Authors**: Renyi Zhou, Huimin Zhu, Jing Tang, Min Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06364)  

**Abstract**: Achieving precise control over a molecule's biological activity-encompassing targeted activation/inhibition, cooperative multi-target modulation, and off-target toxicity mitigation-remains a critical challenge in de novo drug design. However, existing generative methods primarily focus on producing molecules with a single desired activity, lacking integrated mechanisms for the simultaneous management of multiple intended and unintended molecular interactions. Here, we propose ActivityDiff, a generative approach based on the classifier-guidance technique of diffusion models. It leverages separately trained drug-target classifiers for both positive and negative guidance, enabling the model to enhance desired activities while minimizing harmful off-target effects. Experimental results show that ActivityDiff effectively handles essential drug design tasks, including single-/dual-target generation, fragment-constrained dual-target design, selective generation to enhance target specificity, and reduction of off-target effects. These results demonstrate the effectiveness of classifier-guided diffusion in balancing efficacy and safety in molecular design. Overall, our work introduces a novel paradigm for achieving integrated control over molecular activity, and provides ActivityDiff as a versatile and extensible framework. 

---
# Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts 

**Authors**: Zhaomin Wu, Mingzhe Du, See-Kiong Ng, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2508.06361)  

**Abstract**: Large Language Models (LLMs) have been widely deployed in reasoning, planning, and decision-making tasks, making their trustworthiness a critical concern. The potential for intentional deception, where an LLM deliberately fabricates or conceals information to serve a hidden objective, remains a significant and underexplored threat. Existing studies typically induce such deception by explicitly setting a "hidden" objective through prompting or fine-tuning, which may not fully reflect real-world human-LLM interactions. Moving beyond this human-induced deception, we investigate LLMs' self-initiated deception on benign prompts. To address the absence of ground truth in this evaluation, we propose a novel framework using "contact searching questions." This framework introduces two statistical metrics derived from psychological principles to quantify the likelihood of deception. The first, the Deceptive Intention Score, measures the model's bias towards a hidden objective. The second, Deceptive Behavior Score, measures the inconsistency between the LLM's internal belief and its expressed output. Upon evaluating 14 leading LLMs, we find that both metrics escalate as task difficulty increases, rising in parallel for most models. Building on these findings, we formulate a mathematical model to explain this behavior. These results reveal that even the most advanced LLMs exhibit an increasing tendency toward deception when handling complex problems, raising critical concerns for the deployment of LLM agents in complex and crucial domains. 

---
# Are you In or Out (of gallery)? Wisdom from the Same-Identity Crowd 

**Authors**: Aman Bhatta, Maria Dhakal, Michael C. King, Kevin W. Bowyer  

**Link**: [PDF](https://arxiv.org/pdf/2508.06357)  

**Abstract**: A central problem in one-to-many facial identification is that the person in the probe image may or may not have enrolled image(s) in the gallery; that is, may be In-gallery or Out-of-gallery. Past approaches to detect when a rank-one result is Out-of-gallery have mostly focused on finding a suitable threshold on the similarity score. We take a new approach, using the additional enrolled images of the identity with the rank-one result to predict if the rank-one result is In-gallery / Out-of-gallery. Given a gallery of identities and images, we generate In-gallery and Out-of-gallery training data by extracting the ranks of additional enrolled images corresponding to the rank-one identity. We then train a classifier to utilize this feature vector to predict whether a rank-one result is In-gallery or Out-of-gallery. Using two different datasets and four different matchers, we present experimental results showing that our approach is viable for mugshot quality probe images, and also, importantly, for probes degraded by blur, reduced resolution, atmospheric turbulence and sunglasses. We also analyze results across demographic groups, and show that In-gallery / Out-of-gallery classification accuracy is similar across demographics. Our approach has the potential to provide an objective estimate of whether a one-to-many facial identification is Out-of-gallery, and thereby to reduce false positive identifications, wrongful arrests, and wasted investigative time. Interestingly, comparing the results of older deep CNN-based face matchers with newer ones suggests that the effectiveness of our Out-of-gallery detection approach emerges only with matchers trained using advanced margin-based loss functions. 

---
# Structural Equation-VAE: Disentangled Latent Representations for Tabular Data 

**Authors**: Ruiyu Zhang, Ce Zhao, Xin Zhao, Lin Nie, Wai-Fung Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.06347)  

**Abstract**: Learning interpretable latent representations from tabular data remains a challenge in deep generative modeling. We introduce SE-VAE (Structural Equation-Variational Autoencoder), a novel architecture that embeds measurement structure directly into the design of a variational autoencoder. Inspired by structural equation modeling, SE-VAE aligns latent subspaces with known indicator groupings and introduces a global nuisance latent to isolate construct-specific confounding variation. This modular architecture enables disentanglement through design rather than through statistical regularizers alone. We evaluate SE-VAE on a suite of simulated tabular datasets and benchmark its performance against a series of leading baselines using standard disentanglement metrics. SE-VAE consistently outperforms alternatives in factor recovery, interpretability, and robustness to nuisance variation. Ablation results reveal that architectural structure, rather than regularization strength, is the key driver of performance. SE-VAE offers a principled framework for white-box generative modeling in scientific and social domains where latent constructs are theory-driven and measurement validity is essential. 

---
# Harnessing Adaptive Topology Representations for Zero-Shot Graph Question Answering 

**Authors**: Yanbin Wei, Jiangyue Yan, Chun Kang, Yang Chen, Hua Liu, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06345)  

**Abstract**: Large Multimodal Models (LMMs) have shown generalized zero-shot capabilities in diverse domain question-answering (QA) tasks, including graph QA that involves complex graph topologies. However, most current approaches use only a single type of graph representation, namely Topology Representation Form (TRF), such as prompt-unified text descriptions or style-fixed visual styles. Those "one-size-fits-all" approaches fail to consider the specific preferences of different models or tasks, often leading to incorrect or overly long responses. To address this, we first analyze the characteristics and weaknesses of existing TRFs, and then design a set of TRFs, denoted by $F_{ZS}$, tailored to zero-shot graph QA. We then introduce a new metric, Graph Response Efficiency (GRE), which measures the balance between the performance and the brevity in graph QA. Built on these, we develop the DynamicTRF framework, which aims to improve both the accuracy and conciseness of graph QA. To be specific, DynamicTRF first creates a TRF Preference (TRFP) dataset that ranks TRFs based on their GRE scores, to probe the question-specific TRF preferences. Then it trains a TRF router on the TRFP dataset, to adaptively assign the best TRF from $F_{ZS}$ for each question during the inference. Extensive experiments across 7 in-domain algorithmic graph QA tasks and 2 out-of-domain downstream tasks show that DynamicTRF significantly enhances the zero-shot graph QA of LMMs in terms of accuracy 

---
# On Approximate MMS Allocations on Restricted Graph Classes 

**Authors**: Václav Blažej, Michał Dębski ad Zbigniew Lonc, Marta Piecyk, Paweł Rzążewski  

**Link**: [PDF](https://arxiv.org/pdf/2508.06343)  

**Abstract**: We study the problem of fair division of a set of indivisible goods with connectivity constraints. Specifically, we assume that the goods are represented as vertices of a connected graph, and sets of goods allocated to the agents are connected subgraphs of this graph. We focus on the widely-studied maximin share criterion of fairness. It has been shown that an allocation satisfying this criterion may not exist even without connectivity constraints, i.e., if the graph of goods is complete. In view of this, it is natural to seek approximate allocations that guarantee each agent a connected bundle of goods with value at least a constant fraction of the maximin share value to the agent. It is known that for some classes of graphs, such as complete graphs, cycles, and $d$-claw-free graphs for any fixed $d$, such approximate allocations indeed exist. However, it is an open problem whether they exist for the class of all graphs.
In this paper, we continue the systematic study of the existence of approximate allocations on restricted graph classes. In particular, we show that such allocations exist for several well-studied classes, including block graphs, cacti, complete multipartite graphs, and split graphs. 

---
# Unsupervised Partner Design Enables Robust Ad-hoc Teamwork 

**Authors**: Constantin Ruhdorfer, Matteo Bortoletto, Victor Oei, Anna Penzkofer, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2508.06336)  

**Abstract**: We introduce Unsupervised Partner Design (UPD) - a population-free, multi-agent reinforcement learning framework for robust ad-hoc teamwork that adaptively generates training partners without requiring pretrained partners or manual parameter tuning. UPD constructs diverse partners by stochastically mixing an ego agent's policy with biased random behaviours and scores them using a variance-based learnability metric that prioritises partners near the ego agent's current learning frontier. We show that UPD can be integrated with unsupervised environment design, resulting in the first method enabling fully unsupervised curricula over both level and partner distributions in a cooperative setting. Through extensive evaluations on Overcooked-AI and the Overcooked Generalisation Challenge, we demonstrate that this dynamic partner curriculum is highly effective: UPD consistently outperforms both population-based and population-free baselines as well as ablations. In a user study, we further show that UPD achieves higher returns than all baselines and was perceived as significantly more adaptive, more human-like, a better collaborator, and less frustrating. 

---
# Mixture of Experts Guided by Gaussian Splatters Matters: A new Approach to Weakly-Supervised Video Anomaly Detection 

**Authors**: Giacomo D'Amicantonio, Snehashis Majhi, Quan Kong, Lorenzo Garattoni, Gianpiero Francesca, François Bremond, Egor Bondarev  

**Link**: [PDF](https://arxiv.org/pdf/2508.06318)  

**Abstract**: Video Anomaly Detection (VAD) is a challenging task due to the variability of anomalous events and the limited availability of labeled data. Under the Weakly-Supervised VAD (WSVAD) paradigm, only video-level labels are provided during training, while predictions are made at the frame level. Although state-of-the-art models perform well on simple anomalies (e.g., explosions), they struggle with complex real-world events (e.g., shoplifting). This difficulty stems from two key issues: (1) the inability of current models to address the diversity of anomaly types, as they process all categories with a shared model, overlooking category-specific features; and (2) the weak supervision signal, which lacks precise temporal information, limiting the ability to capture nuanced anomalous patterns blended with normal events. To address these challenges, we propose Gaussian Splatting-guided Mixture of Experts (GS-MoE), a novel framework that employs a set of expert models, each specialized in capturing specific anomaly types. These experts are guided by a temporal Gaussian splatting loss, enabling the model to leverage temporal consistency and enhance weak supervision. The Gaussian splatting approach encourages a more precise and comprehensive representation of anomalies by focusing on temporal segments most likely to contain abnormal events. The predictions from these specialized experts are integrated through a mixture-of-experts mechanism to model complex relationships across diverse anomaly patterns. Our approach achieves state-of-the-art performance, with a 91.58% AUC on the UCF-Crime dataset, and demonstrates superior results on XD-Violence and MSAD datasets. By leveraging category-specific expertise and temporal guidance, GS-MoE sets a new benchmark for VAD under weak supervision. 

---
# FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields 

**Authors**: Junhyeog Yun, Minui Hong, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.06301)  

**Abstract**: Neural fields provide a memory-efficient representation of data, which can effectively handle diverse modalities and large-scale data. However, learning to map neural fields often requires large amounts of training data and computations, which can be limited to resource-constrained edge devices. One approach to tackle this limitation is to leverage Federated Meta-Learning (FML), but traditional FML approaches suffer from privacy leakage. To address these issues, we introduce a novel FML approach called FedMeNF. FedMeNF utilizes a new privacy-preserving loss function that regulates privacy leakage in the local meta-optimization. This enables the local meta-learner to optimize quickly and efficiently without retaining the client's private data. Our experiments demonstrate that FedMeNF achieves fast optimization speed and robust reconstruction performance, even with few-shot or non-IID data across diverse data modalities, while preserving client data privacy. 

---
# Advanced Deep Learning Techniques for Accurate Lung Cancer Detection and Classification 

**Authors**: Mobarak Abumohsen, Enrique Costa-Montenegro, Silvia García-Méndez, Amani Yousef Owda, Majdi Owda  

**Link**: [PDF](https://arxiv.org/pdf/2508.06287)  

**Abstract**: Lung cancer (LC) ranks among the most frequently diagnosed cancers and is one of the most common causes of death for men and women worldwide. Computed Tomography (CT) images are the most preferred diagnosis method because of their low cost and their faster processing times. Many researchers have proposed various ways of identifying lung cancer using CT images. However, such techniques suffer from significant false positives, leading to low accuracy. The fundamental reason results from employing a small and imbalanced dataset. This paper introduces an innovative approach for LC detection and classification from CT images based on the DenseNet201 model. Our approach comprises several advanced methods such as Focal Loss, data augmentation, and regularization to overcome the imbalanced data issue and overfitting challenge. The findings show the appropriateness of the proposal, attaining a promising performance of 98.95% accuracy. 

---
# OM2P: Offline Multi-Agent Mean-Flow Policy 

**Authors**: Zhuoran Li, Xun Wang, Hai Zhong, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06269)  

**Abstract**: Generative models, especially diffusion and flow-based models, have been promising in offline multi-agent reinforcement learning. However, integrating powerful generative models into this framework poses unique challenges. In particular, diffusion and flow-based policies suffer from low sampling efficiency due to their iterative generation processes, making them impractical in time-sensitive or resource-constrained settings. To tackle these difficulties, we propose OM2P (Offline Multi-Agent Mean-Flow Policy), a novel offline MARL algorithm to achieve efficient one-step action sampling. To address the misalignment between generative objectives and reward maximization, we introduce a reward-aware optimization scheme that integrates a carefully-designed mean-flow matching loss with Q-function supervision. Additionally, we design a generalized timestep distribution and a derivative-free estimation strategy to reduce memory overhead and improve training stability. Empirical evaluations on Multi-Agent Particle and MuJoCo benchmarks demonstrate that OM2P achieves superior performance, with up to a 3.8x reduction in GPU memory usage and up to a 10.8x speed-up in training time. Our approach represents the first to successfully integrate mean-flow model into offline MARL, paving the way for practical and scalable generative policies in cooperative multi-agent settings. 

---
# Numerical Considerations in Weighted Model Counting 

**Authors**: Randal E. Bryant  

**Link**: [PDF](https://arxiv.org/pdf/2508.06264)  

**Abstract**: Weighted model counting computes the sum of the rational-valued weights associated with the satisfying assignments for a Boolean formula, where the weight of an assignment is given by the product of the weights assigned to the positive and negated variables comprising the assignment. Weighted model counting finds applications across a variety of domains including probabilistic reasoning and quantitative risk assessment.
Most weighted model counting programs operate by (explicitly or implicitly) converting the input formula into a form that enables arithmetic evaluation, using multiplication for conjunctions and addition for disjunctions. Performing this evaluation using floating-point arithmetic can yield inaccurate results, and it cannot quantify the level of precision achieved. Computing with rational arithmetic gives exact results, but it is costly in both time and space.
This paper describes how to combine multiple numeric representations to efficiently compute weighted model counts that are guaranteed to achieve a user-specified precision. When all weights are nonnegative, we prove that the precision loss of arithmetic evaluation using floating-point arithmetic can be tightly bounded. We show that supplementing a standard IEEE double-precision representation with a separate 64-bit exponent, a format we call extended-range double (ERD), avoids the underflow and overflow issues commonly encountered in weighted model counting. For problems with mixed negative and positive weights, we show that a combination of interval floating-point arithmetic and rational arithmetic can achieve the twin goals of efficiency and guaranteed precision. For our evaluations, we have devised especially challenging formulas and weight assignments, demonstrating the robustness of our approach. 

---
# SIFThinker: Spatially-Aware Image Focus for Visual Reasoning 

**Authors**: Zhangquan Chen, Ruihui Zhao, Chuwei Luo, Mingze Sun, Xinlei Yu, Yangyang Kang, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06259)  

**Abstract**: Current multimodal large language models (MLLMs) still face significant challenges in complex visual tasks (e.g., spatial understanding, fine-grained perception). Prior methods have tried to incorporate visual reasoning, however, they fail to leverage attention correction with spatial cues to iteratively refine their focus on prompt-relevant regions. In this paper, we introduce SIFThinker, a spatially-aware "think-with-images" framework that mimics human visual perception. Specifically, SIFThinker enables attention correcting and image region focusing by interleaving depth-enhanced bounding boxes and natural language. Our contributions are twofold: First, we introduce a reverse-expansion-forward-inference strategy that facilitates the generation of interleaved image-text chains of thought for process-level supervision, which in turn leads to the construction of the SIF-50K dataset. Besides, we propose GRPO-SIF, a reinforced training paradigm that integrates depth-informed visual grounding into a unified reasoning pipeline, teaching the model to dynamically correct and focus on prompt-relevant regions. Extensive experiments demonstrate that SIFThinker outperforms state-of-the-art methods in spatial understanding and fine-grained visual perception, while maintaining strong general capabilities, highlighting the effectiveness of our method. 

---
# Synthetic Data Generation and Differential Privacy using Tensor Networks' Matrix Product States (MPS) 

**Authors**: Alejandro Moreno R., Desale Fentaw, Samuel Palmer, Raúl Salles de Padua, Ninad Dixit, Samuel Mugel, Roman Orús, Manuel Radons, Josef Menter, Ali Abedi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06251)  

**Abstract**: Synthetic data generation is a key technique in modern artificial intelligence, addressing data scarcity, privacy constraints, and the need for diverse datasets in training robust models. In this work, we propose a method for generating privacy-preserving high-quality synthetic tabular data using Tensor Networks, specifically Matrix Product States (MPS). We benchmark the MPS-based generative model against state-of-the-art models such as CTGAN, VAE, and PrivBayes, focusing on both fidelity and privacy-preserving capabilities. To ensure differential privacy (DP), we integrate noise injection and gradient clipping during training, enabling privacy guarantees via Rényi Differential Privacy accounting. Across multiple metrics analyzing data fidelity and downstream machine learning task performance, our results show that MPS outperforms classical models, particularly under strict privacy constraints. This work highlights MPS as a promising tool for privacy-aware synthetic data generation. By combining the expressive power of tensor network representations with formal privacy mechanisms, the proposed approach offers an interpretable and scalable alternative for secure data sharing. Its structured design facilitates integration into sensitive domains where both data quality and confidentiality are critical. 

---
# In-Training Defenses against Emergent Misalignment in Language Models 

**Authors**: David Kaczér, Magnus Jørgenvåg, Clemens Vetter, Lucie Flek, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2508.06249)  

**Abstract**: Fine-tuning lets practitioners repurpose aligned large language models (LLMs) for new domains, yet recent work reveals emergent misalignment (EMA): Even a small, domain-specific fine-tune can induce harmful behaviors far outside the target domain. Even in the case where model weights are hidden behind a fine-tuning API, this gives attackers inadvertent access to a broadly misaligned model in a way that can be hard to detect from the fine-tuning data alone. We present the first systematic study of in-training safeguards against EMA that are practical for providers who expose fine-tuning via an API. We investigate four training regularization interventions: (i) KL-divergence regularization toward a safe reference model, (ii) $\ell_2$ distance in feature space, (iii) projecting onto a safe subspace (SafeLoRA), and (iv) interleaving of a small amount of safe training examples from a general instruct-tuning dataset. We first evaluate the methods' emergent misalignment effect across four malicious, EMA-inducing tasks. Second, we assess the methods' impacts on benign tasks. We conclude with a discussion of open questions in emergent misalignment research. 

---
# Membership Inference Attack with Partial Features 

**Authors**: Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Weizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06244)  

**Abstract**: Machine learning models have been shown to be susceptible to membership inference attack, which can be used to determine whether a given sample appears in the training data. Existing membership inference methods commonly assume that the adversary has full access to the features of the target sample. This assumption, however, does not hold in many real-world scenarios where only partial features information is available, thereby limiting the applicability of these methods. In this work, we study an inference scenario where the adversary observes only partial features of each sample and aims to infer whether this observed subset was present in the training set of the target model. We define this problem as Partial Feature Membership Inference (PFMI). To address this problem, we propose MRAD (Memory-guided Reconstruction and Anomaly Detection), a two-stage attack framework. In the first stage, MRAD optimizes the unknown feature values to minimize the loss of the sample. In the second stage, it measures the deviation between the reconstructed sample and the training distribution using anomaly detection. Empirical results demonstrate that MRAD is effective across a range of datasets, and maintains compatibility with various off-the-shelf anomaly detection techniques. For example, on STL-10, our attack achieves an AUC of around 0.6 even with 40% of the missing features. 

---
# InfoCausalQA:Can Models Perform Non-explicit Causal Reasoning Based on Infographic? 

**Authors**: Keummin Ka, Junhyeong Park, Jahyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06220)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have demonstrated impressive capabilities in perception and reasoning. However, the ability to perform causal inference -- a core aspect of human cognition -- remains underexplored, particularly in multimodal settings. In this study, we introduce InfoCausalQA, a novel benchmark designed to evaluate causal reasoning grounded in infographics that combine structured visual data with textual context. The benchmark comprises two tasks: Task 1 focuses on quantitative causal reasoning based on inferred numerical trends, while Task 2 targets semantic causal reasoning involving five types of causal relations: cause, effect, intervention, counterfactual, and temporal. We manually collected 494 infographic-text pairs from four public sources and used GPT-4o to generate 1,482 high-quality multiple-choice QA pairs. These questions were then carefully revised by humans to ensure they cannot be answered based on surface-level cues alone but instead require genuine visual grounding. Our experimental results reveal that current VLMs exhibit limited capability in computational reasoning and even more pronounced limitations in semantic causal reasoning. Their significantly lower performance compared to humans indicates a substantial gap in leveraging infographic-based information for causal inference. Through InfoCausalQA, we highlight the need for advancing the causal reasoning abilities of multimodal AI systems. 

---
# Reparameterization Proximal Policy Optimization 

**Authors**: Hai Zhong, Xun Wang, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06214)  

**Abstract**: Reparameterization policy gradient (RPG) is promising for improving sample efficiency by leveraging differentiable dynamics. However, a critical barrier is its training instability, where high-variance gradients can destabilize the learning process. To address this, we draw inspiration from Proximal Policy Optimization (PPO), which uses a surrogate objective to enable stable sample reuse in the model-free setting. We first establish a connection between this surrogate objective and RPG, which has been largely unexplored and is non-trivial. Then, we bridge this gap by demonstrating that the reparameterization gradient of a PPO-like surrogate objective can be computed efficiently using backpropagation through time. Based on this key insight, we propose Reparameterization Proximal Policy Optimization (RPO), a stable and sample-efficient RPG-based method. RPO enables multiple epochs of stable sample reuse by optimizing a clipped surrogate objective tailored for RPG, while being further stabilized by Kullback-Leibler (KL) divergence regularization and remaining fully compatible with existing variance reduction methods. We evaluate RPO on a suite of challenging locomotion and manipulation tasks, where experiments demonstrate that our method achieves superior sample efficiency and strong performance. 

---
# Graph Federated Learning for Personalized Privacy Recommendation 

**Authors**: Ce Na, Kai Yang, Dengzhao Fang, Yu Li, Jingtong Gao, Chengcheng Zhu, Jiale Zhang, Xiaobing Sun, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06208)  

**Abstract**: Federated recommendation systems (FedRecs) have gained significant attention for providing privacy-preserving recommendation services. However, existing FedRecs assume that all users have the same requirements for privacy protection, i.e., they do not upload any data to the server. The approaches overlook the potential to enhance the recommendation service by utilizing publicly available user data. In real-world applications, users can choose to be private or public. Private users' interaction data is not shared, while public users' interaction data can be shared. Inspired by the issue, this paper proposes a novel Graph Federated Learning for Personalized Privacy Recommendation (GFed-PP) that adapts to different privacy requirements while improving recommendation performance. GFed-PP incorporates the interaction data of public users to build a user-item interaction graph, which is then used to form a user relationship graph. A lightweight graph convolutional network (GCN) is employed to learn each user's user-specific personalized item embedding. To protect user privacy, each client learns the user embedding and the scoring function locally. Additionally, GFed-PP achieves optimization of the federated recommendation framework through the initialization of item embedding on clients and the aggregation of the user relationship graph on the server. Experimental results demonstrate that GFed-PP significantly outperforms existing methods for five datasets, offering superior recommendation accuracy without compromising privacy. This framework provides a practical solution for accommodating varying privacy preferences in federated recommendation systems. 

---
# Classification is a RAG problem: A case study on hate speech detection 

**Authors**: Richard Willats, Josh Pennington, Aravind Mohan, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06204)  

**Abstract**: Robust content moderation requires classification systems that can quickly adapt to evolving policies without costly retraining. We present classification using Retrieval-Augmented Generation (RAG), which shifts traditional classification tasks from determining the correct category in accordance with pre-trained parameters to evaluating content in relation to contextual knowledge retrieved at inference. In hate speech detection, this transforms the task from "is this hate speech?" to "does this violate the hate speech policy?"
Our Contextual Policy Engine (CPE) - an agentic RAG system - demonstrates this approach and offers three key advantages: (1) robust classification accuracy comparable to leading commercial systems, (2) inherent explainability via retrieved policy segments, and (3) dynamic policy updates without model retraining. Through three experiments, we demonstrate strong baseline performance and show that the system can apply fine-grained policy control by correctly adjusting protection for specific identity groups without requiring retraining or compromising overall performance. These findings establish that RAG can transform classification into a more flexible, transparent, and adaptable process for content moderation and wider classification problems. 

---
# LoRA in LoRA: Towards Parameter-Efficient Architecture Expansion for Continual Visual Instruction Tuning 

**Authors**: Chang Che, Ziqi Wang, Pengwan Yang, Qi Wang, Hui Ma, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06202)  

**Abstract**: Continual Visual Instruction Tuning (CVIT) enables Multimodal Large Language Models (MLLMs) to incrementally learn new tasks over time. However, this process is challenged by catastrophic forgetting, where performance on previously learned tasks deteriorates as the model adapts to new ones. A common approach to mitigate forgetting is architecture expansion, which introduces task-specific modules to prevent interference. Yet, existing methods often expand entire layers for each task, leading to significant parameter overhead and poor scalability. To overcome these issues, we introduce LoRA in LoRA (LiLoRA), a highly efficient architecture expansion method tailored for CVIT in MLLMs. LiLoRA shares the LoRA matrix A across tasks to reduce redundancy, applies an additional low-rank decomposition to matrix B to minimize task-specific parameters, and incorporates a cosine-regularized stability loss to preserve consistency in shared representations over time. Extensive experiments on a diverse CVIT benchmark show that LiLoRA consistently achieves superior performance in sequential task learning while significantly improving parameter efficiency compared to existing approaches. 

---
# Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning 

**Authors**: Mateusz Praski, Jakub Adamczyk, Wojciech Czech  

**Link**: [PDF](https://arxiv.org/pdf/2508.06199)  

**Abstract**: Pretrained neural networks have attracted significant interest in chemistry and small molecule drug design. Embeddings from these models are widely used for molecular property prediction, virtual screening, and small data learning in molecular chemistry. This study presents the most extensive comparison of such models to date, evaluating 25 models across 25 datasets. Under a fair comparison framework, we assess models spanning various modalities, architectures, and pretraining strategies. Using a dedicated hierarchical Bayesian statistical testing model, we arrive at a surprising result: nearly all neural models show negligible or no improvement over the baseline ECFP molecular fingerprint. Only the CLAMP model, which is also based on molecular fingerprints, performs statistically significantly better than the alternatives. These findings raise concerns about the evaluation rigor in existing studies. We discuss potential causes, propose solutions, and offer practical recommendations. 

---
# Differentially Private Federated Clustering with Random Rebalancing 

**Authors**: Xiyuan Yang, Shengyuan Hu, Soyeon Kim, Tian Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06183)  

**Abstract**: Federated clustering aims to group similar clients into clusters and produce one model for each cluster. Such a personalization approach typically improves model performance compared with training a single model to serve all clients, but can be more vulnerable to privacy leakage. Directly applying client-level differentially private (DP) mechanisms to federated clustering could degrade the utilities significantly. We identify that such deficiencies are mainly due to the difficulties of averaging privacy noise within each cluster (following standard privacy mechanisms), as the number of clients assigned to the same clusters is uncontrolled. To this end, we propose a simple and effective technique, named RR-Cluster, that can be viewed as a light-weight add-on to many federated clustering algorithms. RR-Cluster achieves reduced privacy noise via randomly rebalancing cluster assignments, guaranteeing a minimum number of clients assigned to each cluster. We analyze the tradeoffs between decreased privacy noise variance and potentially increased bias from incorrect assignments and provide convergence bounds for RR-Clsuter. Empirically, we demonstrate the RR-Cluster plugged into strong federated clustering algorithms results in significantly improved privacy/utility tradeoffs across both synthetic and real-world datasets. 

---
# Synthetic Data-Driven Multi-Architecture Framework for Automated Polyp Segmentation Through Integrated Detection and Mask Generation 

**Authors**: Ojonugwa Oluwafemi Ejiga Peter, Akingbola Oluwapemiisin, Amalahu Chetachi, Adeniran Opeyemi, Fahmi Khalifa, Md Mahmudur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.06170)  

**Abstract**: Colonoscopy is a vital tool for the early diagnosis of colorectal cancer, which is one of the main causes of cancer-related mortality globally; hence, it is deemed an essential technique for the prevention and early detection of colorectal cancer. The research introduces a unique multidirectional architectural framework to automate polyp detection within colonoscopy images while helping resolve limited healthcare dataset sizes and annotation complexities. The research implements a comprehensive system that delivers synthetic data generation through Stable Diffusion enhancements together with detection and segmentation algorithms. This detection approach combines Faster R-CNN for initial object localization while the Segment Anything Model (SAM) refines the segmentation masks. The faster R-CNN detection algorithm achieved a recall of 93.08% combined with a precision of 88.97% and an F1 score of 90.98%.SAM is then used to generate the image mask. The research evaluated five state-of-the-art segmentation models that included U-Net, PSPNet, FPN, LinkNet, and MANet using ResNet34 as a base model. The results demonstrate the superior performance of FPN with the highest scores of PSNR (7.205893) and SSIM (0.492381), while UNet excels in recall (84.85%) and LinkNet shows balanced performance in IoU (64.20%) and Dice score (77.53%). 

---
# UW-3DGS: Underwater 3D Reconstruction with Physics-Aware Gaussian Splatting 

**Authors**: Wenpeng Xing, Jie Chen, Zaifeng Yang, Changting Lin, Jianfeng Dong, Chaochao Chen, Xun Zhou, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.06169)  

**Abstract**: Underwater 3D scene reconstruction faces severe challenges from light absorption, scattering, and turbidity, which degrade geometry and color fidelity in traditional methods like Neural Radiance Fields (NeRF). While NeRF extensions such as SeaThru-NeRF incorporate physics-based models, their MLP reliance limits efficiency and spatial resolution in hazy environments. We introduce UW-3DGS, a novel framework adapting 3D Gaussian Splatting (3DGS) for robust underwater reconstruction. Key innovations include: (1) a plug-and-play learnable underwater image formation module using voxel-based regression for spatially varying attenuation and backscatter; and (2) a Physics-Aware Uncertainty Pruning (PAUP) branch that adaptively removes noisy floating Gaussians via uncertainty scoring, ensuring artifact-free geometry. The pipeline operates in training and rendering stages. During training, noisy Gaussians are optimized end-to-end with underwater parameters, guided by PAUP pruning and scattering modeling. In rendering, refined Gaussians produce clean Unattenuated Radiance Images (URIs) free from media effects, while learned physics enable realistic Underwater Images (UWIs) with accurate light transport. Experiments on SeaThru-NeRF and UWBundle datasets show superior performance, achieving PSNR of 27.604, SSIM of 0.868, and LPIPS of 0.104 on SeaThru-NeRF, with ~65% reduction in floating artifacts. 

---
# UR$^2$: Unify RAG and Reasoning through Reinforcement Learning 

**Authors**: Weitao Li, Boran Xiang, Xiaolong Wang, Zhinan Gou, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06165)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities through two complementary paradigms: Retrieval-Augmented Generation (RAG), which enhances knowledge grounding, and Reinforcement Learning from Verifiable Rewards (RLVR), which optimizes complex reasoning abilities. However, these two capabilities are often developed in isolation, and existing efforts to unify them remain narrow in scope-typically limited to open-domain QA with fixed retrieval settings and task-specific assumptions. This lack of integration constrains generalization and limits the applicability of RAG-RL methods to broader domains. To bridge this gap, we propose UR2 (Unified RAG and Reasoning), a general framework that unifies retrieval and reasoning through reinforcement learning. UR2 introduces two key contributions: a difficulty-aware curriculum training that selectively invokes retrieval only for challenging problems, and a hybrid knowledge access strategy combining domain-specific offline corpora with LLM-generated summaries. These components are designed to enable dynamic coordination between retrieval and reasoning, improving adaptability across a diverse range of tasks. Experiments across open-domain QA, MMLU-Pro, medical, and mathematical reasoning tasks demonstrate that UR2 (built on Qwen2.5-3/7B and LLaMA-3.1-8B) significantly outperforms existing RAG and RL methods, achieving comparable performance to GPT-4o-mini and GPT-4.1-mini on several benchmarks. We have released all code, models, and data at this https URL. 

---
# One Size Does Not Fit All: A Distribution-Aware Sparsification for More Precise Model Merging 

**Authors**: Yingfeng Luo, Dingyang Lin, Junxin Wang, Ziqiang Xu, Kaiyan Chang, Tong Zheng, Bei Li, Anxiang Ma, Tong Xiao, Zhengtao Yu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06163)  

**Abstract**: Model merging has emerged as a compelling data-free paradigm for multi-task learning, enabling the fusion of multiple fine-tuned models into a single, powerful entity. A key technique in merging methods is sparsification, which prunes redundant parameters from task vectors to mitigate interference. However, prevailing approaches employ a ``one-size-fits-all'' strategy, applying a uniform sparsity ratio that overlooks the inherent structural and statistical heterogeneity of model parameters. This often leads to a suboptimal trade-off, where critical parameters are inadvertently pruned while less useful ones are retained. To address this limitation, we introduce \textbf{TADrop} (\textbf{T}ensor-wise \textbf{A}daptive \textbf{Drop}), an adaptive sparsification strategy that respects this heterogeneity. Instead of a global ratio, TADrop assigns a tailored sparsity level to each parameter tensor based on its distributional properties. The core intuition is that tensors with denser, more redundant distributions can be pruned aggressively, while sparser, more critical ones are preserved. As a simple and plug-and-play module, we validate TADrop by integrating it with foundational, classic, and SOTA merging methods. Extensive experiments across diverse tasks (vision, language, and multimodal) and models (ViT, BEiT) demonstrate that TADrop consistently and significantly boosts their performance. For instance, when enhancing a leading merging method, it achieves an average performance gain of 2.0\% across 8 ViT-B/32 tasks. TADrop provides a more effective way to mitigate parameter interference by tailoring sparsification to the model's structure, offering a new baseline for high-performance model merging. 

---
# Semantic Item Graph Enhancement for Multimodal Recommendation 

**Authors**: Xiaoxiong Zhang, Xin Zhou, Zhiwei Zeng, Dusit Niyato, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06154)  

**Abstract**: Multimodal recommendation systems have attracted increasing attention for their improved performance by leveraging items' multimodal information. Prior methods often build modality-specific item-item semantic graphs from raw modality features and use them as supplementary structures alongside the user-item interaction graph to enhance user preference learning. However, these semantic graphs suffer from semantic deficiencies, including (1) insufficient modeling of collaborative signals among items and (2) structural distortions introduced by noise in raw modality features, ultimately compromising performance. To address these issues, we first extract collaborative signals from the interaction graph and infuse them into each modality-specific item semantic graph to enhance semantic modeling. Then, we design a modulus-based personalized embedding perturbation mechanism that injects perturbations with modulus-guided personalized intensity into embeddings to generate contrastive views. This enables the model to learn noise-robust representations through contrastive learning, thereby reducing the effect of structural noise in semantic graphs. Besides, we propose a dual representation alignment mechanism that first aligns multiple semantic representations via a designed Anchor-based InfoNCE loss using behavior representations as anchors, and then aligns behavior representations with the fused semantics by standard InfoNCE, to ensure representation consistency. Extensive experiments on four benchmark datasets validate the effectiveness of our framework. 

---
# Roll Your Eyes: Gaze Redirection via Explicit 3D Eyeball Rotation 

**Authors**: YoungChan Choi, HengFei Wang, YiHua Cheng, Boeun Kim, Hyung Jin Chang, YoungGeun Choi, Sang-Il Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06136)  

**Abstract**: We propose a novel 3D gaze redirection framework that leverages an explicit 3D eyeball structure. Existing gaze redirection methods are typically based on neural radiance fields, which employ implicit neural representations via volume rendering. Unlike these NeRF-based approaches, where the rotation and translation of 3D representations are not explicitly modeled, we introduce a dedicated 3D eyeball structure to represent the eyeballs with 3D Gaussian Splatting (3DGS). Our method generates photorealistic images that faithfully reproduce the desired gaze direction by explicitly rotating and translating the 3D eyeball structure. In addition, we propose an adaptive deformation module that enables the replication of subtle muscle movements around the eyes. Through experiments conducted on the ETH-XGaze dataset, we demonstrate that our framework is capable of generating diverse novel gaze images, achieving superior image quality and gaze estimation accuracy compared to previous state-of-the-art methods. 

---
# Less is More: Selective Reflection for Compatible and Efficient Knowledge Distillation in Large Language Models 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06135)  

**Abstract**: Knowledge Distillation (KD) is a fundamental technique for compressing large language models (LLMs) into compact, efficient student models. However, existing white-box KD methods mainly focus on balancing ground truth and student-generated responses while overlooking two critical factors: training data quality and student-model compatibility. To address these limitations, we propose Selective Reflection Distillation (SRD), a novel data curation framework that leverages reflections from student models to systematically refine training data. SRD dynamically evaluates and selects prompt-response pairs by comparing ground truth data with student model outputs, selectively curating high-quality, student-compatible training instances through automated ranking based on difficulty. Furthermore, after selecting the training data, a curriculum scheduling strategy is employed to incrementally introduce these curated subsets into the distillation process at fixed intervals. As a plug-and-play enhancement, SRD consistently improves distillation outcomes across diverse white-box KD approaches and model architectures, as well as decreases computational cost significantly during KD training. Experiments on a range of language model benchmarks demonstrate SRD's consistent improvements in distilled model performance, as well as a reduction in training runtime by up to 39%, under diverse KD methods and model families. Notably, SRD operates as a plug-and-play module, enhancing sample efficiency without modifying underlying KD algorithms. Our findings highlight that data quality and compatibility are pivotal to effective and efficient distillation of LLMs, and SRD provides a principled framework to achieve both. This work advances the understanding of data-centric factors in KD and offers practical insights for enhancing the capability and efficiency of compressed LLMs. 

---
# LLM Serving Optimization with Variable Prefill and Decode Lengths 

**Authors**: Meixuan Wang, Yinyu Ye, Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.06133)  

**Abstract**: We study the problem of serving LLM (Large Language Model) requests where each request has heterogeneous prefill and decode lengths. In LLM serving, the prefill length corresponds to the input prompt length, which determines the initial memory usage in the KV cache. The decode length refers to the number of output tokens generated sequentially, with each additional token increasing the KV cache memory usage by one unit. Given a set of n requests, our goal is to schedule and process them to minimize the total completion time. We show that this problem is NP-hard due to the interplay of batching, placement constraints, precedence relationships, and linearly increasing memory usage. We then analyze commonly used scheduling strategies in practice, such as First-Come-First-Serve (FCFS) and Shortest-First (SF), and prove that their competitive ratios scale up sublinearly with the memory limit-a significant drawback in real-world settings where memory demand is large. To address this, we propose a novel algorithm based on a new selection metric that efficiently forms batches over time. We prove that this algorithm achieves a constant competitive ratio. Finally, we develop and evaluate a few algorithm variants inspired by this approach, including dynamic programming variants, local search methods, and an LP-based scheduler, demonstrating through comprehensive simulations that they outperform standard baselines while maintaining computational efficiency. 

---
# FMCE-Net++: Feature Map Convergence Evaluation and Training 

**Authors**: Zhibo Zhu, Renyu Huang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.06109)  

**Abstract**: Deep Neural Networks (DNNs) face interpretability challenges due to their opaque internal representations. While Feature Map Convergence Evaluation (FMCE) quantifies module-level convergence via Feature Map Convergence Scores (FMCS), it lacks experimental validation and closed-loop integration. To address this limitation, we propose FMCE-Net++, a novel training framework that integrates a pretrained, frozen FMCE-Net as an auxiliary head. This module generates FMCS predictions, which, combined with task labels, jointly supervise backbone optimization through a Representation Auxiliary Loss. The RAL dynamically balances the primary classification loss and feature convergence optimization via a tunable \Representation Abstraction Factor. Extensive experiments conducted on MNIST, CIFAR-10, FashionMNIST, and CIFAR-100 demonstrate that FMCE-Net++ consistently enhances model performance without architectural modifications or additional data. Key experimental outcomes include accuracy gains of $+1.16$ pp (ResNet-50/CIFAR-10) and $+1.08$ pp (ShuffleNet v2/CIFAR-100), validating that FMCE-Net++ can effectively elevate state-of-the-art performance ceilings. 

---
# GCHR : Goal-Conditioned Hindsight Regularization for Sample-Efficient Reinforcement Learning 

**Authors**: Xing Lei, Wenyan Yang, Kaiqiang Ke, Shentao Yang, Xuetao Zhang, Joni Pajarinen, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06108)  

**Abstract**: Goal-conditioned reinforcement learning (GCRL) with sparse rewards remains a fundamental challenge in reinforcement learning. While hindsight experience replay (HER) has shown promise by relabeling collected trajectories with achieved goals, we argue that trajectory relabeling alone does not fully exploit the available experiences in off-policy GCRL methods, resulting in limited sample efficiency. In this paper, we propose Hindsight Goal-conditioned Regularization (HGR), a technique that generates action regularization priors based on hindsight goals. When combined with hindsight self-imitation regularization (HSR), our approach enables off-policy RL algorithms to maximize experience utilization. Compared to existing GCRL methods that employ HER and self-imitation techniques, our hindsight regularizations achieve substantially more efficient sample reuse and the best performances, which we empirically demonstrate on a suite of navigation and manipulation tasks. 

---
# Mask & Match: Learning to Recognize Handwritten Math with Self-Supervised Attention 

**Authors**: Shree Mitra, Ritabrata Chakraborty, Nilkanta Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06107)  

**Abstract**: Recognizing handwritten mathematical expressions (HMER) is a challenging task due to the inherent two-dimensional structure, varying symbol scales, and complex spatial relationships among symbols. In this paper, we present a self-supervised learning (SSL) framework for HMER that eliminates the need for expensive labeled data. Our approach begins by pretraining an image encoder using a combination of global and local contrastive loss, enabling the model to learn both holistic and fine-grained representations. A key contribution of this work is a novel self-supervised attention network, which is trained using a progressive spatial masking strategy. This attention mechanism is designed to learn semantically meaningful focus regions, such as operators, exponents, and nested mathematical notation, without requiring any supervision. The progressive masking curriculum encourages the network to become increasingly robust to missing or occluded visual information, ultimately improving structural understanding. Our complete pipeline consists of (1) self-supervised pretraining of the encoder, (2) self-supervised attention learning, and (3) supervised fine-tuning with a transformer decoder to generate LATEX sequences. Extensive experiments on CROHME benchmarks demonstrate that our method outperforms existing SSL and fully supervised baselines, validating the effectiveness of our progressive attention mechanism in enhancing HMER performance. Our codebase can be found here. 

---
# MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows 

**Authors**: Xiquan Li, Junxi Liu, Yuzhe Liang, Zhikang Niu, Wenxi Chen, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06098)  

**Abstract**: Recent developments in diffusion- and flow- based models have significantly advanced Text-to-Audio Generation (TTA). While achieving great synthesis quality and controllability, current TTA systems still suffer from slow inference speed, which significantly limits their practical applicability. This paper presents MeanAudio, a novel MeanFlow-based model tailored for fast and faithful text-to-audio generation. Built on a Flux-style latent transformer, MeanAudio regresses the average velocity field during training, enabling fast generation by mapping directly from the start to the endpoint of the flow trajectory. By incorporating classifier-free guidance (CFG) into the training target, MeanAudio incurs no additional cost in the guided sampling process. To further stabilize training, we propose an instantaneous-to-mean curriculum with flow field mix-up, which encourages the model to first learn the foundational instantaneous dynamics, and then gradually adapt to mean flows. This strategy proves critical for enhancing training efficiency and generation quality. Experimental results demonstrate that MeanAudio achieves state-of-the-art performance in single-step audio generation. Specifically, it achieves a real time factor (RTF) of 0.013 on a single NVIDIA RTX 3090, yielding a 100x speedup over SOTA diffusion-based TTA systems. Moreover, MeanAudio also demonstrates strong performance in multi-step generation, enabling smooth and coherent transitions across successive synthesis steps. 

---
# Bounding Distributional Shifts in World Modeling through Novelty Detection 

**Authors**: Eric Jing, Abdeslam Boularias  

**Link**: [PDF](https://arxiv.org/pdf/2508.06096)  

**Abstract**: Recent work on visual world models shows significant promise in latent state dynamics obtained from pre-trained image backbones. However, most of the current approaches are sensitive to training quality, requiring near-complete coverage of the action and state space during training to prevent divergence during inference. To make a model-based planning algorithm more robust to the quality of the learned world model, we propose in this work to use a variational autoencoder as a novelty detector to ensure that proposed action trajectories during planning do not cause the learned model to deviate from the training data distribution. To evaluate the effectiveness of this approach, a series of experiments in challenging simulated robot environments was carried out, with the proposed method incorporated into a model-predictive control policy loop extending the DINO-WM architecture. The results clearly show that the proposed method improves over state-of-the-art solutions in terms of data efficiency. 

---
# Towards MR-Based Trochleoplasty Planning 

**Authors**: Michael Wehrli, Alicia Durrer, Paul Friedrich, Sidaty El Hadramy, Edwin Li, Luana Brahaj, Carol C. Hasler, Philippe C. Cattin  

**Link**: [PDF](https://arxiv.org/pdf/2508.06076)  

**Abstract**: To treat Trochlear Dysplasia (TD), current approaches rely mainly on low-resolution clinical Magnetic Resonance (MR) scans and surgical intuition. The surgeries are planned based on surgeons experience, have limited adoption of minimally invasive techniques, and lead to inconsistent outcomes. We propose a pipeline that generates super-resolved, patient-specific 3D pseudo-healthy target morphologies from conventional clinical MR scans. First, we compute an isotropic super-resolved MR volume using an Implicit Neural Representation (INR). Next, we segment femur, tibia, patella, and fibula with a multi-label custom-trained network. Finally, we train a Wavelet Diffusion Model (WDM) to generate pseudo-healthy target morphologies of the trochlear region. In contrast to prior work producing pseudo-healthy low-resolution 3D MR images, our approach enables the generation of sub-millimeter resolved 3D shapes compatible for pre- and intraoperative use. These can serve as preoperative blueprints for reshaping the femoral groove while preserving the native patella articulation. Furthermore, and in contrast to other work, we do not require a CT for our pipeline - reducing the amount of radiation. We evaluated our approach on 25 TD patients and could show that our target morphologies significantly improve the sulcus angle (SA) and trochlear groove depth (TGD). The code and interactive visualization are available at this https URL. 

---
# Can Large Models Fool the Eye? A New Turing Test for Biological Animation 

**Authors**: Zijian Chen, Lirong Deng, Zhengyu Chen, Kaiwei Zhang, Qi Jia, Yuan Tian, Yucheng Zhu, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.06072)  

**Abstract**: Evaluating the abilities of large models and manifesting their gaps are challenging. Current benchmarks adopt either ground-truth-based score-form evaluation on static datasets or indistinct textual chatbot-style human preferences collection, which may not provide users with immediate, intuitive, and perceptible feedback on performance differences. In this paper, we introduce BioMotion Arena, a novel framework for evaluating large language models (LLMs) and multimodal large language models (MLLMs) via visual animation. Our methodology draws inspiration from the inherent visual perception of motion patterns characteristic of living organisms that utilizes point-light source imaging to amplify the performance discrepancies between models. Specifically, we employ a pairwise comparison evaluation and collect more than 45k votes for 53 mainstream LLMs and MLLMs on 90 biological motion variants. Data analyses show that the crowd-sourced human votes are in good agreement with those of expert raters, demonstrating the superiority of our BioMotion Arena in offering discriminative feedback. We also find that over 90\% of evaluated models, including the cutting-edge open-source InternVL3 and proprietary Claude-4 series, fail to produce fundamental humanoid point-light groups, much less smooth and biologically plausible motions. This enables BioMotion Arena to serve as a challenging benchmark for performance visualization and a flexible evaluation framework without restrictions on ground-truth. 

---
# Architecture-Aware Generalization Bounds for Temporal Networks: Theory and Fair Comparison Methodology 

**Authors**: Barak Gahtan, Alex M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.06066)  

**Abstract**: Deep temporal architectures such as Temporal Convolutional Networks (TCNs) achieve strong predictive performance on sequential data, yet theoretical understanding of their generalization remains limited. We address this gap by providing both the first non-vacuous, architecture-aware generalization bounds for deep temporal models and a principled evaluation methodology.
For exponentially $\beta$-mixing sequences, we derive bounds scaling as $ O\!\Bigl(R\,\sqrt{\tfrac{D\,p\,n\,\log N}{N}}\Bigr), $ where $D$ is network depth, $p$ kernel size, $n$ input dimension, and $R$ weight norm. Our delayed-feedback blocking mechanism transforms dependent samples into effectively independent ones while discarding only $O(1/\log N)$ of the data, yielding $\sqrt{D}$ scaling instead of exponential, implying that doubling depth requires approximately quadrupling the training data.
We also introduce a fair-comparison methodology that fixes the effective sample size to isolate the effect of temporal structure from information content. Under $N_{\text{eff}}=2{,}000$, strongly dependent sequences ($\rho=0.8$) exhibit $\approx76\%$ smaller generalization gaps than weakly dependent ones ($\rho=0.2$), challenging the intuition that dependence is purely detrimental. Yet convergence rates diverge from theory: weak dependencies follow $N_{\text{eff}}^{-1.21}$ scaling and strong dependencies follow $N_{\text{eff}}^{-0.89}$, both steeper than the predicted $N^{-0.5}$. These findings reveal that temporal dependence can enhance learning under fixed information budgets, while highlighting gaps between theory and practice that motivate future research. 

---
# ThematicPlane: Bridging Tacit User Intent and Latent Spaces for Image Generation 

**Authors**: Daniel Lee, Nikhil Sharma, Donghoon Shin, DaEun Choi, Harsh Sharma, Jeonghwan Kim, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.06065)  

**Abstract**: Generative AI has made image creation more accessible, yet aligning outputs with nuanced creative intent remains challenging, particularly for non-experts. Existing tools often require users to externalize ideas through prompts or references, limiting fluid exploration. We introduce ThematicPlane, a system that enables users to navigate and manipulate high-level semantic concepts (e.g., mood, style, or narrative tone) within an interactive thematic design plane. This interface bridges the gap between tacit creative intent and system control. In our exploratory study (N=6), participants engaged in divergent and convergent creative modes, often embracing unexpected results as inspiration or iteration cues. While they grounded their exploration in familiar themes, differing expectations of how themes mapped to outputs revealed a need for more explainable controls. Overall, ThematicPlane fosters expressive, iterative workflows and highlights new directions for intuitive, semantics-driven interaction in generative design tools. 

---
# EvolvR: Self-Evolving Pairwise Reasoning for Story Evaluation to Enhance Generation 

**Authors**: Xinda Wang, Zhengxu Hou, Yangshijie Zhang, Bingren Yan, Zhibo Yang, Xingsheng Zhang, Luxi Xing, Qiang Zhou, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06046)  

**Abstract**: Although the effectiveness of Large Language Models (LLMs) as judges (LLM-as-a-judge) has been validated, their performance remains limited in open-ended tasks, particularly in story evaluation. Accurate story evaluation is crucial not only for assisting human quality judgment but also for providing key signals to guide story generation. However, existing methods face a dilemma: prompt engineering for closed-source models suffers from poor adaptability, while fine-tuning approaches for open-source models lack the rigorous reasoning capabilities essential for story evaluation. To address this, we propose the Self-Evolving Pairwise Reasoning (EvolvR) framework. Grounded in pairwise comparison, the framework first self-synthesizes score-aligned Chain-of-Thought (CoT) data via a multi-persona strategy. To ensure data quality, these raw CoTs undergo a self-filtering process, utilizing multi-agents to guarantee their logical rigor and robustness. Finally, the evaluator trained on the refined data is deployed as a reward model to guide the story generation task. Experimental results demonstrate that our framework achieves state-of-the-art (SOTA) performance on three evaluation benchmarks including StoryER, HANNA and OpenMEVA. Furthermore, when served as a reward model, it significantly enhances the quality of generated stories, thereby fully validating the superiority of our self-evolving approach. 

---
# DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment 

**Authors**: Sangwoo Kwon, Seong Hoon Seo, Jae W. Lee, Yeonhong Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.06041)  

**Abstract**: How can we effectively handle queries for on-device large language models (LLMs) with varying runtime constraints, such as latency and accuracy? Multi-scale quantization addresses this challenge by enabling memory-efficient runtime model adaptation of LLMs through the overlaying of multiple model variants quantized to different bitwidths. Meanwhile, an important question still remains open-ended: how can models be properly configured to match a target precision or latency? While mixed-precision offers a promising solution, we take this further by leveraging the key observation that the sensitivity of each layer dynamically changes across decoding iterations. Building on this insight, we introduce DP-LLM, a novel mechanism that dynamically assigns precision to each layer based on input values. DP-LLM augments each linear layer in an LLM with a precision selector that determines the bitwidth at runtime using a lightweight error estimator and threshold values learned through fine-tuning. Experimental results across multiple models and benchmarks demonstrate that DP-LLM achieves a superior performance-latency trade-off, outperforming prior approaches. 

---
# Fourier-VLM: Compressing Vision Tokens in the Frequency Domain for Large Vision-Language Models 

**Authors**: Huanyu Wang, Jushi Kai, Haoli Bai, Lu Hou, Bo Jiang, Ziwei He, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.06038)  

**Abstract**: Vision-Language Models (VLMs) typically replace the predefined image placeholder token (<image>) in textual instructions with visual features from an image encoder, forming the input to a backbone Large Language Model (LLM). However, the large number of vision tokens significantly increases the context length, leading to high computational overhead and inference latency. While previous efforts mitigate this by selecting only important visual features or leveraging learnable queries to reduce token count, they often compromise performance or introduce substantial extra costs. In response, we propose Fourier-VLM, a simple yet efficient method that compresses visual representations in the frequency domain. Our approach is motivated by the observation that vision features output from the vision encoder exhibit concentrated energy in low-frequency components. Leveraging this, we apply a low-pass filter to the vision features using a two-dimentional Discrete Cosine Transform (DCT). Notably, the DCT is efficiently computed via the Fast Fourier Transform (FFT) operator with a time complexity of $\mathcal{O}(n\log n)$, minimizing the extra computational cost while introducing no additional parameters. Extensive experiments across various image-based benchmarks demonstrate that Fourier-VLM achieves competitive performance with strong generalizability across both LLaVA and Qwen-VL architectures. Crucially, it reduce inference FLOPs by up to 83.8% and boots generation speed by 31.2% compared to LLaVA-v1.5, highlighting the superior efficiency and practicality. 

---
# Adaptive Heterogeneous Graph Neural Networks: Bridging Heterophily and Heterogeneity 

**Authors**: Qin Chen, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.06034)  

**Abstract**: Heterogeneous graphs (HGs) are common in real-world scenarios and often exhibit heterophily. However, most existing studies focus on either heterogeneity or heterophily in isolation, overlooking the prevalence of heterophilic HGs in practical applications. Such ignorance leads to their performance degradation. In this work, we first identify two main challenges in modeling heterophily HGs: (1) varying heterophily distributions across hops and meta-paths; (2) the intricate and often heterophily-driven diversity of semantic information across different meta-paths. Then, we propose the Adaptive Heterogeneous Graph Neural Network (AHGNN) to tackle these challenges. AHGNN employs a heterophily-aware convolution that accounts for heterophily distributions specific to both hops and meta-paths. It then integrates messages from diverse semantic spaces using a coarse-to-fine attention mechanism, which filters out noise and emphasizes informative signals. Experiments on seven real-world graphs and twenty baselines demonstrate the superior performance of AHGNN, particularly in high-heterophily situations. 

---
# Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future 

**Authors**: Yidong Wang, Xin Wang, Cunxiang Wang, Junfeng Fang, Qiufeng Wang, Jianing Chu, Xuran Meng, Shuxun Yang, Libo Qin, Yue Zhang, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06026)  

**Abstract**: Self-Rewarding Language Models propose an architecture in which the Large Language Models(LLMs) both generates responses and evaluates its own outputs via LLM-as-a-Judge prompting, dynamically improving its generative capabilities through iterative Direct Preference Optimization (DPO). However, our analysis reveals a critical limitation in existing Self-Rewarding paradigms: the synchronized improvement of chosen and rejected responses progressively narrows the representational difference between contrasting samples, undermining effective preference learning. We propose \textbf{Temporal Self-Rewarding Language Models} that strategically coordinate past, present, and future model generations to sustain learning signals. Our dual-phase framework introduces: (1) \textit{Anchored Rejection} - fixing rejected responses using the past initial model's outputs and (2) \textit{Future-Guided Chosen} - dynamically curating chosen samples using next-generation model predictions. Extensive experiments across three model families (Llama, Qwen, Mistral) and different model sizes (Llama3B/8B/70B) demonstrate significant improvements when trained with our method compared to Self-Rewarding using same computation resources. For example, Llama3.1-8B reaches a 29.44 win rate on AlpacaEval 2.0 with our method, outperforming the Self-Rewarding baseline (19.69) by 9.75. Notably, our method also demonstrates superior out-of-distribution generalization across mathematical reasoning (GSM8K), knowledge-based QA (ARC, TruthfulQA), and code generation (HumanEval) tasks, even though we do not specifically collect such training data. 

---
# Improved Sub-Visible Particle Classification in Flow Imaging Microscopy via Generative AI-Based Image Synthesis 

**Authors**: Utku Ozbulak, Michaela Cohrs, Hristo L. Svilenov, Joris Vankerschaver, Wesley De Neve  

**Link**: [PDF](https://arxiv.org/pdf/2508.06021)  

**Abstract**: Sub-visible particle analysis using flow imaging microscopy combined with deep learning has proven effective in identifying particle types, enabling the distinction of harmless components such as silicone oil from protein particles. However, the scarcity of available data and severe imbalance between particle types within datasets remain substantial hurdles when applying multi-class classifiers to such problems, often forcing researchers to rely on less effective methods. The aforementioned issue is particularly challenging for particle types that appear unintentionally and in lower numbers, such as silicone oil and air bubbles, as opposed to protein particles, where obtaining large numbers of images through controlled settings is comparatively straightforward. In this work, we develop a state-of-the-art diffusion model to address data imbalance by generating high-fidelity images that can augment training datasets, enabling the effective training of multi-class deep neural networks. We validate this approach by demonstrating that the generated samples closely resemble real particle images in terms of visual quality and structure. To assess the effectiveness of using diffusion-generated images in training datasets, we conduct large-scale experiments on a validation dataset comprising 500,000 protein particle images and demonstrate that this approach improves classification performance with no negligible downside. Finally, to promote open research and reproducibility, we publicly release both our diffusion models and the trained multi-class deep neural network classifiers, along with a straightforward interface for easy integration into future studies, at this https URL. 

---
# Crisp Attention: Regularizing Transformers via Structured Sparsity 

**Authors**: Sagar Gandhi, Vishal Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06016)  

**Abstract**: The quadratic computational cost of the self-attention mechanism is a primary challenge in scaling Transformer models. While attention sparsity is widely studied as a technique to improve computational efficiency, it is almost universally assumed to come at the cost of model accuracy. In this paper, we report a surprising counter-example to this common wisdom. By introducing structured, post-hoc sparsity to the attention mechanism of a DistilBERT model during fine-tuning on the SST-2 sentiment analysis task, we find that model accuracy improves significantly. Our model with 80\% attention sparsity achieves a validation accuracy of 91.59\%, a 0.97\% absolute improvement over the dense baseline. We hypothesize that this phenomenon is due to sparsity acting as a powerful implicit regularizer, preventing the model from overfitting by forcing it to make predictions with a more constrained and robust set of features. Our work recasts attention sparsity not just as a tool for computational efficiency, but as a potential method for improving the generalization and performance of Transformer models. 

---
# Hand by Hand: LLM Driving EMS Assistant for Operational Skill Learning 

**Authors**: Wei Xiang, Ziyue Lei, Haoyuan Che, Fangyuan Ye, Xueting Wu, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.06000)  

**Abstract**: Operational skill learning, inherently physical and reliant on hands-on practice and kinesthetic feedback, has yet to be effectively replicated in large language model (LLM)-supported training. Current LLM training assistants primarily generate customized textual feedback, neglecting the crucial kinesthetic modality. This gap derives from the textual and uncertain nature of LLMs, compounded by concerns on user acceptance of LLM driven body control. To bridge this gap and realize the potential of collaborative human-LLM action, this work explores human experience of LLM driven kinesthetic assistance. Specifically, we introduced an "Align-Analyze-Adjust" strategy and developed FlightAxis, a tool that integrates LLM with Electrical Muscle Stimulation (EMS) for flight skill acquisition, a representative operational skill domain. FlightAxis learns flight skills from manuals and guides forearm movements during simulated flight tasks. Our results demonstrate high user acceptance of LLM-mediated body control and significantly reduced task completion times. Crucially, trainees reported that this kinesthetic assistance enhanced their awareness of operation flaws and fostered increased engagement in the training process, rather than relieving perceived load. This work demonstrated the potential of kinesthetic LLM training in operational skill acquisition. 

---
# ECMF: Enhanced Cross-Modal Fusion for Multimodal Emotion Recognition in MER-SEMI Challenge 

**Authors**: Juewen Hu, Yexin Li, Jiulin Li, Shuo Chen, Pring Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.05991)  

**Abstract**: Emotion recognition plays a vital role in enhancing human-computer interaction. In this study, we tackle the MER-SEMI challenge of the MER2025 competition by proposing a novel multimodal emotion recognition framework. To address the issue of data scarcity, we leverage large-scale pre-trained models to extract informative features from visual, audio, and textual modalities. Specifically, for the visual modality, we design a dual-branch visual encoder that captures both global frame-level features and localized facial representations. For the textual modality, we introduce a context-enriched method that employs large language models to enrich emotional cues within the input text. To effectively integrate these multimodal features, we propose a fusion strategy comprising two key components, i.e., self-attention mechanisms for dynamic modality weighting, and residual connections to preserve original representations. Beyond architectural design, we further refine noisy labels in the training set by a multi-source labeling strategy. Our approach achieves a substantial performance improvement over the official baseline on the MER2025-SEMI dataset, attaining a weighted F-score of 87.49% compared to 78.63%, thereby validating the effectiveness of the proposed framework. 

---
# ETA: Energy-based Test-time Adaptation for Depth Completion 

**Authors**: Younjoon Chung, Hyoungseob Park, Patrick Rim, Xiaoran Zhang, Jihe He, Ziyao Zeng, Safa Cicek, Byung-Woo Hong, James S. Duncan, Alex Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.05989)  

**Abstract**: We propose a method for test-time adaptation of pretrained depth completion models. Depth completion models, trained on some ``source'' data, often predict erroneous outputs when transferred to ``target'' data captured in novel environmental conditions due to a covariate shift. The crux of our method lies in quantifying the likelihood of depth predictions belonging to the source data distribution. The challenge is in the lack of access to out-of-distribution (target) data prior to deployment. Hence, rather than making assumptions regarding the target distribution, we utilize adversarial perturbations as a mechanism to explore the data space. This enables us to train an energy model that scores local regions of depth predictions as in- or out-of-distribution. We update the parameters of pretrained depth completion models at test time to minimize energy, effectively aligning test-time predictions to those of the source distribution. We call our method ``Energy-based Test-time Adaptation'', or ETA for short. We evaluate our method across three indoor and three outdoor datasets, where ETA improve over the previous state-of-the-art method by an average of 6.94% for outdoors and 10.23% for indoors. Project Page: this https URL. 

---
# Learning by Teaching: Engaging Students as Instructors of Large Language Models in Computer Science Education 

**Authors**: Xinming Yang, Haasil Pujara, Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.05979)  

**Abstract**: While Large Language Models (LLMs) are often used as virtual tutors in computer science (CS) education, this approach can foster passive learning and over-reliance. This paper presents a novel pedagogical paradigm that inverts this model: students act as instructors who must teach an LLM to solve problems. To facilitate this, we developed strategies for designing questions with engineered knowledge gaps that only a student can bridge, and we introduce Socrates, a system for deploying this method with minimal overhead. We evaluated our approach in an undergraduate course and found that this active-learning method led to statistically significant improvements in student performance compared to historical cohorts. Our work demonstrates a practical, cost-effective framework for using LLMs to deepen student engagement and mastery. 

---
# DAFMSVC: One-Shot Singing Voice Conversion with Dual Attention Mechanism and Flow Matching 

**Authors**: Wei Chen, Binzhu Sha, Dan Luo, Jing Yang, Zhuo Wang, Fan Fan, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05978)  

**Abstract**: Singing Voice Conversion (SVC) transfers a source singer's timbre to a target while keeping melody and lyrics. The key challenge in any-to-any SVC is adapting unseen speaker timbres to source audio without quality degradation. Existing methods either face timbre leakage or fail to achieve satisfactory timbre similarity and quality in the generated audio. To address these challenges, we propose DAFMSVC, where the self-supervised learning (SSL) features from the source audio are replaced with the most similar SSL features from the target audio to prevent timbre leakage. It also incorporates a dual cross-attention mechanism for the adaptive fusion of speaker embeddings, melody, and linguistic content. Additionally, we introduce a flow matching module for high quality audio generation from the fused features. Experimental results show that DAFMSVC significantly enhances timbre similarity and naturalness, outperforming state-of-the-art methods in both subjective and objective evaluations. 

---
# Impact-driven Context Filtering For Cross-file Code Completion 

**Authors**: Yanzhou Li, Shangqing Liu, Kangjie Chen, Tianwei Zhang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05970)  

**Abstract**: Retrieval-augmented generation (RAG) has recently demonstrated considerable potential for repository-level code completion, as it integrates cross-file knowledge with in-file preceding code to provide comprehensive contexts for generation. To better understand the contribution of the retrieved cross-file contexts, we introduce a likelihood-based metric to evaluate the impact of each retrieved code chunk on the completion. Our analysis reveals that, despite retrieving numerous chunks, only a small subset positively contributes to the completion, while some chunks even degrade performance. To address this issue, we leverage this metric to construct a repository-level dataset where each retrieved chunk is labeled as positive, neutral, or negative based on its relevance to the target completion. We then propose an adaptive retrieval context filtering framework, CODEFILTER, trained on this dataset to mitigate the harmful effects of negative retrieved contexts in code completion. Extensive evaluation on the RepoEval and CrossCodeLongEval benchmarks demonstrates that CODEFILTER consistently improves completion accuracy compared to approaches without filtering operations across various tasks. Additionally, CODEFILTER significantly reduces the length of the input prompt, enhancing computational efficiency while exhibiting strong generalizability across different models. These results underscore the potential of CODEFILTER to enhance the accuracy, efficiency, and attributability of repository-level code completion. 

---
# Mildly Conservative Regularized Evaluation for Offline Reinforcement Learning 

**Authors**: Haohui Chen, Zhiyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.05960)  

**Abstract**: Offline reinforcement learning (RL) seeks to learn optimal policies from static datasets without further environment interaction. A key challenge is the distribution shift between the learned and behavior policies, leading to out-of-distribution (OOD) actions and overestimation. To prevent gross overestimation, the value function must remain conservative; however, excessive conservatism may hinder performance improvement. To address this, we propose the mildly conservative regularized evaluation (MCRE) framework, which balances conservatism and performance by combining temporal difference (TD) error with a behavior cloning term in the Bellman backup. Building on this, we develop the mildly conservative regularized Q-learning (MCRQ) algorithm, which integrates MCRE into an off-policy actor-critic framework. Experiments show that MCRQ outperforms strong baselines and state-of-the-art offline RL algorithms on benchmark datasets. 

---
# Multi-Armed Bandits-Based Optimization of Decision Trees 

**Authors**: Hasibul Karim Shanto, Umme Ayman Koana, Shadikur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.05957)  

**Abstract**: Decision trees, without appropriate constraints, can easily become overly complex and prone to overfit, capturing noise rather than generalizable patterns. To resolve this problem,pruning operation is a crucial part in optimizing decision trees, as it not only reduces the complexity of trees but also decreases the probability of generating overfit models. The conventional pruning techniques like Cost-Complexity Pruning (CCP) and Reduced Error Pruning (REP) are mostly based on greedy approaches that focus on immediate gains in performance while pruning nodes of the decision tree. However, this might result in a lower generalization in the long run, compromising the robust ability of the tree model when introduced to unseen data samples, particularly when trained with small and complex datasets. To address this challenge, we are proposing a Multi-Armed Bandits (MAB)-based pruning approach, a reinforcement learning (RL)-based technique, that will dynamically prune the tree to generate an optimal decision tree with better generalization. Our proposed approach assumes the pruning process as an exploration-exploitation problem, where we are utilizing the MAB algorithms to find optimal branch nodes to prune based on feedback from each pruning actions. Experimental evaluation on several benchmark datasets, demonstrated that our proposed approach results in better predictive performance compared to the traditional ones. This suggests the potential of utilizing MAB for a dynamic and probabilistic way of decision tree pruning, in turn optimizing the decision tree-based model. 

---
# Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents 

**Authors**: Han Lin, Jaemin Cho, Amir Zadeh, Chuan Li, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.05954)  

**Abstract**: There is growing interest in integrating high-fidelity visual synthesis capabilities into large language models (LLMs) without compromising their strong reasoning capabilities. Existing methods that directly train LLMs or bridge LLMs and diffusion models usually suffer from costly training since the backbone LLMs have not seen image representations during pretraining. We present Bifrost-1, a unified framework that bridges pretrained multimodal LLMs (MLLMs) and diffusion models using patch-level CLIP image embeddings as latent variables, which are natively aligned with the MLLM's CLIP visual encoder. These patch-level image embeddings are integrated into the diffusion model with a lightweight adaptation of its ControlNet. To retain the original multimodal reasoning capabilities of MLLMs, we equip the MLLM with a visual generation branch initialized from the original MLLM parameters when predicting the patch-level image embeddings. By seamlessly integrating pretrained MLLMs and diffusion models with patch-level CLIP latents, our framework enables high-fidelity controllable image generation with significant training efficiency. Our experiments demonstrate that Bifrost-1 achieves comparable or better performance than previous methods in terms of visual fidelity and multimodal understanding, with substantially lower compute during training. We also provide comprehensive ablation studies showing the effectiveness of our design choices. 

---
# A 3DGS-Diffusion Self-Supervised Framework for Normal Estimation from a Single Image 

**Authors**: Yanxing Liang, Yinghui Wang, Jinlong Yang, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.05950)  

**Abstract**: The lack of spatial dimensional information remains a challenge in normal estimation from a single image. Recent diffusion-based methods have demonstrated significant potential in 2D-to-3D implicit mapping, they rely on data-driven statistical priors and miss the explicit modeling of light-surface interaction, leading to multi-view normal direction conflicts. Moreover, the discrete sampling mechanism of diffusion models causes gradient discontinuity in differentiable rendering reconstruction modules, preventing 3D geometric errors from being backpropagated to the normal generation network, thereby forcing existing methods to depend on dense normal annotations. This paper proposes SINGAD, a novel Self-supervised framework from a single Image for Normal estimation via 3D GAussian splatting guided Diffusion. By integrating physics-driven light-interaction modeling and a differentiable rendering-based reprojection strategy, our framework directly converts 3D geometric errors into normal optimization signals, solving the challenges of multi-view geometric inconsistency and data dependency. Specifically, the framework constructs a light-interaction-driven 3DGS reparameterization model to generate multi-scale geometric features consistent with light transport principles, ensuring multi-view normal consistency. A cross-domain feature fusion module is designed within a conditional diffusion model, embedding geometric priors to constrain normal generation while maintaining accurate geometric error propagation. Furthermore, a differentiable 3D reprojection loss strategy is introduced for self-supervised optimization that minimizes geometric error between the reconstructed and input image, eliminating dependence on annotated normal datasets. Quantitative evaluations on the Google Scanned Objects dataset demonstrate that our method outperforms state-of-the-art approaches across multiple metrics. 

---
# Prosocial Behavior Detection in Player Game Chat: From Aligning Human-AI Definitions to Efficient Annotation at Scale 

**Authors**: Rafal Kocielnik, Min Kim, Penphob, Boonyarungsrit, Fereshteh Soltani, Deshawn Sambrano, Animashree Anandkumar, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2508.05938)  

**Abstract**: Detecting prosociality in text--communication intended to affirm, support, or improve others' behavior--is a novel and increasingly important challenge for trust and safety systems. Unlike toxic content detection, prosociality lacks well-established definitions and labeled data, requiring new approaches to both annotation and deployment. We present a practical, three-stage pipeline that enables scalable, high-precision prosocial content classification while minimizing human labeling effort and inference costs. First, we identify the best LLM-based labeling strategy using a small seed set of human-labeled examples. We then introduce a human-AI refinement loop, where annotators review high-disagreement cases between GPT-4 and humans to iteratively clarify and expand the task definition-a critical step for emerging annotation tasks like prosociality. This process results in improved label quality and definition alignment. Finally, we synthesize 10k high-quality labels using GPT-4 and train a two-stage inference system: a lightweight classifier handles high-confidence predictions, while only $\sim$35\% of ambiguous instances are escalated to GPT-4o. This architecture reduces inference costs by $\sim$70% while achieving high precision ($\sim$0.90). Our pipeline demonstrates how targeted human-AI interaction, careful task formulation, and deployment-aware architecture design can unlock scalable solutions for novel responsible AI tasks. 

---
# ASLSL: Adaptive shared latent structure learning with incomplete multi-modal physiological data for multi-dimensional emotional feature selection 

**Authors**: Xueyuan Xu, Tianze Yu, Wenjia Dong, Fulin Wei, Li Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05934)  

**Abstract**: Recently, multi-modal physiological signals based emotion recognition has garnered increasing attention in the field of brain-computer interfaces. Nevertheness, the associated multi-modal physiological features are often high-dimensional and inevitably include irrelevant, redundant, and noisy representation, which can easily lead to overfitting, poor performance, and high computational complexity in emotion classifiers. Feature selection has been widely applied to address these challenges. However, previous studies generally assumed that multi-modal physiological data are complete, whereas in reality, the data are often incomplete due to the openness of the acquisition and operational environment. For example, a part of samples are available in several modalities but not in others. To address this issue, we propose a novel method for incomplete multi-modal physiological signal feature selection called adaptive shared latent structure learning (ASLSL). Based on the property that similar features share similar emotional labels, ASLSL employs adaptive shared latent structure learning to explore a common latent space shared for incomplete multi-modal physiological signals and multi-dimensional emotional labels, thereby mitigating the impact of missing information and mining consensus information. Two most popular multi-modal physiological emotion datasets (DEAP and DREAMER) with multi-dimensional emotional labels were utilized to compare the performance between compare ASLSL and seventeen feature selection methods. Comprehensive experimental results on these datasets demonstrate the effectiveness of ASLSL. 

---
# REFS: Robust EEG feature selection with missing multi-dimensional annotation for emotion recognition 

**Authors**: Xueyuan Xu, Wenjia Dong, Fulin Wei, Li Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05933)  

**Abstract**: The affective brain-computer interface is a crucial technology for affective interaction and emotional intelligence, emerging as a significant area of research in the human-computer interaction. Compared to single-type features, multi-type EEG features provide a multi-level representation for analyzing multi-dimensional emotions. However, the high dimensionality of multi-type EEG features, combined with the relatively small number of high-quality EEG samples, poses challenges such as classifier overfitting and suboptimal real-time performance in multi-dimensional emotion recognition. Moreover, practical applications of affective brain-computer interface frequently encounters partial absence of multi-dimensional emotional labels due to the open nature of the acquisition environment, and ambiguity and variability in individual emotion perception. To address these challenges, this study proposes a novel EEG feature selection method for missing multi-dimensional emotion recognition. The method leverages adaptive orthogonal non-negative matrix factorization to reconstruct the multi-dimensional emotional label space through second-order and higher-order correlations, which could reduce the negative impact of missing values and outliers on label reconstruction. Simultaneously, it employs least squares regression with graph-based manifold learning regularization and global feature redundancy minimization regularization to enable EEG feature subset selection despite missing information, ultimately achieving robust EEG-based multi-dimensional emotion recognition. Simulation experiments on three widely used multi-dimensional emotional datasets, DREAMER, DEAP and HDED, reveal that the proposed method outperforms thirteen advanced feature selection methods in terms of robustness for EEG emotional feature selection. 

---
# Enhancing Software Vulnerability Detection Through Adaptive Test Input Generation Using Genetic Algorithm 

**Authors**: Yanusha Mehendran, Maolin Tang, Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05923)  

**Abstract**: Software vulnerabilities continue to undermine the reliability and security of modern systems, particularly as software complexity outpaces the capabilities of traditional detection methods. This study introduces a genetic algorithm-based method for test input generation that innovatively integrates genetic operators and adaptive learning to enhance software vulnerability detection. A key contribution is the application of the crossover operator, which facilitates exploration by searching across a broader space of potential test inputs. Complementing this, an adaptive feedback mechanism continuously learns from the system's execution behavior and dynamically guides input generation toward promising areas of the input space. Rather than relying on fixed or randomly selected inputs, the approach evolves a population of structurally valid test cases using feedback-driven selection, enabling deeper and more effective code traversal. This strategic integration of exploration and exploitation ensures that both diverse and targeted test inputs are developed over time. Evaluation was conducted across nine open-source JSON-processing libraries. The proposed method achieved substantial improvements in coverage compared to a benchmark evolutionary fuzzing method, with average gains of 39.8% in class coverage, 62.4% in method coverage, 105.0% in line coverage, 114.0% in instruction coverage, and 166.0% in branch coverage. These results highlight the method's capacity to detect deeper and more complex vulnerabilities, offering a scalable and adaptive solution to software security testing. 

---
# Do Ethical AI Principles Matter to Users? A Large-Scale Analysis of User Sentiment and Satisfaction 

**Authors**: Stefan Pasch, Min Chul Cha  

**Link**: [PDF](https://arxiv.org/pdf/2508.05913)  

**Abstract**: As AI systems become increasingly embedded in organizational workflows and consumer applications, ethical principles such as fairness, transparency, and robustness have been widely endorsed in policy and industry guidelines. However, there is still scarce empirical evidence on whether these principles are recognized, valued, or impactful from the perspective of users. This study investigates the link between ethical AI and user satisfaction by analyzing over 100,000 user reviews of AI products from G2. Using transformer-based language models, we measure sentiment across seven ethical dimensions defined by the EU Ethics Guidelines for Trustworthy AI. Our findings show that all seven dimensions are positively associated with user satisfaction. Yet, this relationship varies systematically across user and product types. Technical users and reviewers of AI development platforms more frequently discuss system-level concerns (e.g., transparency, data governance), while non-technical users and reviewers of end-user applications emphasize human-centric dimensions (e.g., human agency, societal well-being). Moreover, the association between ethical AI and user satisfaction is significantly stronger for non-technical users and end-user applications across all dimensions. Our results highlight the importance of ethical AI design from users' perspectives and underscore the need to account for contextual differences across user roles and product types. 

---
# Do Machines Think Emotionally? Cognitive Appraisal Analysis of Large Language Models 

**Authors**: Sree Bhattacharyya, Lucas Craig, Tharun Dilliraj, Jia Li, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05880)  

**Abstract**: Affective Computing has been established as a crucial field of inquiry to advance the holistic development of Artificial Intelligence (AI) systems. Foundation models -- especially Large Language Models (LLMs) -- have been evaluated, trained, or instruction-tuned in several past works, to become better predictors or generators of emotion. Most of these studies, however, approach emotion-related tasks in a supervised manner, assessing or training the capabilities of LLMs using discrete emotion labels associated with stimuli (e.g., text, images, video, audio). Evaluation studies, in particular, have often been limited to standard and superficial emotion-related tasks, such as the recognition of evoked or expressed emotions. In this paper, we move beyond surface-level emotion tasks to investigate how LLMs reason about emotions through cognitive dimensions. Drawing from cognitive appraisal theory, we examine whether LLMs produce coherent and plausible cognitive reasoning when reasoning about emotionally charged stimuli. We introduce a large-scale benchmark on Cognitive Reasoning for Emotions - CoRE - to evaluate internal cognitive structures implicitly used by LLMs for emotional reasoning. Through a plethora of evaluation experiments and analysis, we seek to answer: (a) Are models more likely to implicitly rely on specific cognitive appraisal dimensions?, (b) What cognitive dimensions are important for characterizing specific emotions?, and, (c) Can the internal representations of different emotion categories in LLMs be interpreted through cognitive appraisal dimensions? Our results and analyses reveal diverse reasoning patterns across different LLMs. Our benchmark and code will be made publicly available. 

---
# Towards Transparent Ethical AI: A Roadmap for Trustworthy Robotic Systems 

**Authors**: Ahmad Farooq, Kamran Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2508.05846)  

**Abstract**: As artificial intelligence (AI) and robotics increasingly permeate society, ensuring the ethical behavior of these systems has become paramount. This paper contends that transparency in AI decision-making processes is fundamental to developing trustworthy and ethically aligned robotic systems. We explore how transparency facilitates accountability, enables informed consent, and supports the debugging of ethical algorithms. The paper outlines technical, ethical, and practical challenges in implementing transparency and proposes novel approaches to enhance it, including standardized metrics, explainable AI techniques, and user-friendly interfaces. This paper introduces a framework that connects technical implementation with ethical considerations in robotic systems, focusing on the specific challenges of achieving transparency in dynamic, real-world contexts. We analyze how prioritizing transparency can impact public trust, regulatory policies, and avenues for future research. By positioning transparency as a fundamental element in ethical AI system design, we aim to add to the ongoing discussion on responsible AI and robotics, providing direction for future advancements in this vital field. 

---
# Integrating Vision Foundation Models with Reinforcement Learning for Enhanced Object Interaction 

**Authors**: Ahmad Farooq, Kamran Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2508.05838)  

**Abstract**: This paper presents a novel approach that integrates vision foundation models with reinforcement learning to enhance object interaction capabilities in simulated environments. By combining the Segment Anything Model (SAM) and YOLOv5 with a Proximal Policy Optimization (PPO) agent operating in the AI2-THOR simulation environment, we enable the agent to perceive and interact with objects more effectively. Our comprehensive experiments, conducted across four diverse indoor kitchen settings, demonstrate significant improvements in object interaction success rates and navigation efficiency compared to a baseline agent without advanced perception. The results show a 68% increase in average cumulative reward, a 52.5% improvement in object interaction success rate, and a 33% increase in navigation efficiency. These findings highlight the potential of integrating foundation models with reinforcement learning for complex robotic tasks, paving the way for more sophisticated and capable autonomous agents. 

---
# AI-Guided Exploration of Large-Scale Codebases 

**Authors**: Yoseph Berhanu Alebachew  

**Link**: [PDF](https://arxiv.org/pdf/2508.05799)  

**Abstract**: Understanding large-scale, complex software systems is a major challenge for developers, who spend a significant portion of their time on program comprehension. Traditional tools such as static visualizations and reverse engineering techniques provide structural insights but often lack interactivity, adaptability, and integration with contextual information. Recent advancements in large language models (LLMs) offer new opportunities to enhance code exploration workflows, yet their lack of grounding and integration with structured views limits their effectiveness. This work introduces a hybrid approach that integrates deterministic reverse engineering with LLM-guided, intent-aware visual exploration. The proposed system combines UML-based visualization, dynamic user interfaces, historical context, and collaborative features into an adaptive tool for code comprehension. By interpreting user queries and interaction patterns, the LLM helps developers navigate and understand complex codebases more effectively. A prototype implementation for Java demonstrates the feasibility of this approach. Future work includes empirical evaluation, scaling to polyglot systems, and exploring GUI-driven LLM interaction models. This research lays the groundwork for intelligent, interactive environments that align with developer cognition and collaborative workflows. 

---
# From Imperfect Signals to Trustworthy Structure: Confidence-Aware Inference from Heterogeneous and Reliability-Varying Utility Data 

**Authors**: Haoran Li, Lihao Mai, Muhao Guo, Jiaqi Wu, Yang Weng, Yannan Sun, Ce Jimmy Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05791)  

**Abstract**: Accurate distribution grid topology is essential for reliable modern grid operations. However, real-world utility data originates from multiple sources with varying characteristics and levels of quality. In this work, developed in collaboration with Oncor Electric Delivery, we propose a scalable framework that reconstructs a trustworthy grid topology by systematically integrating heterogeneous data. We observe that distribution topology is fundamentally governed by two complementary dimensions: the spatial layout of physical infrastructure (e.g., GIS and asset metadata) and the dynamic behavior of the system in the signal domain (e.g., voltage time series). When jointly leveraged, these dimensions support a complete and physically coherent reconstruction of network connectivity. To address the challenge of uneven data quality without compromising observability, we introduce a confidence-aware inference mechanism that preserves structurally informative yet imperfect inputs, while quantifying the reliability of each inferred connection for operator interpretation. This soft handling of uncertainty is tightly coupled with hard enforcement of physical feasibility: we embed operational constraints, such as transformer capacity limits and radial topology requirements, directly into the learning process. Together, these components ensure that inference is both uncertainty-aware and structurally valid, enabling rapid convergence to actionable, trustworthy topologies under real-world deployment conditions. The proposed framework is validated using data from over 8000 meters across 3 feeders in Oncor's service territory, demonstrating over 95% accuracy in topology reconstruction and substantial improvements in confidence calibration and computational efficiency relative to baseline methods. 

---
# Few-Shot Deployment of Pretrained MRI Transformers in Brain Imaging Tasks 

**Authors**: Mengyu Li, Guoyao Shen, Chad W. Farris, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05783)  

**Abstract**: Machine learning using transformers has shown great potential in medical imaging, but its real-world applicability remains limited due to the scarcity of annotated data. In this study, we propose a practical framework for the few-shot deployment of pretrained MRI transformers in diverse brain imaging tasks. By utilizing the Masked Autoencoder (MAE) pretraining strategy on a large-scale, multi-cohort brain MRI dataset comprising over 31 million slices, we obtain highly transferable latent representations that generalize well across tasks and datasets. For high-level tasks such as classification, a frozen MAE encoder combined with a lightweight linear head achieves state-of-the-art accuracy in MRI sequence identification with minimal supervision. For low-level tasks such as segmentation, we propose MAE-FUnet, a hybrid architecture that fuses multiscale CNN features with pretrained MAE embeddings. This model consistently outperforms other strong baselines in both skull stripping and multi-class anatomical segmentation under data-limited conditions. With extensive quantitative and qualitative evaluations, our framework demonstrates efficiency, stability, and scalability, suggesting its suitability for low-resource clinical environments and broader neuroimaging applications. 

---
# UnGuide: Learning to Forget with LoRA-Guided Diffusion Models 

**Authors**: Agnieszka Polowczyk, Alicja Polowczyk, Dawid Malarz, Artur Kasymov, Marcin Mazur, Jacek Tabor, Przemysław Spurek  

**Link**: [PDF](https://arxiv.org/pdf/2508.05755)  

**Abstract**: Recent advances in large-scale text-to-image diffusion models have heightened concerns about their potential misuse, especially in generating harmful or misleading content. This underscores the urgent need for effective machine unlearning, i.e., removing specific knowledge or concepts from pretrained models without compromising overall performance. One possible approach is Low-Rank Adaptation (LoRA), which offers an efficient means to fine-tune models for targeted unlearning. However, LoRA often inadvertently alters unrelated content, leading to diminished image fidelity and realism. To address this limitation, we introduce UnGuide -- a novel approach which incorporates UnGuidance, a dynamic inference mechanism that leverages Classifier-Free Guidance (CFG) to exert precise control over the unlearning process. UnGuide modulates the guidance scale based on the stability of a few first steps of denoising processes, enabling selective unlearning by LoRA adapter. For prompts containing the erased concept, the LoRA module predominates and is counterbalanced by the base model; for unrelated prompts, the base model governs generation, preserving content fidelity. Empirical results demonstrate that UnGuide achieves controlled concept removal and retains the expressive power of diffusion models, outperforming existing LoRA-based methods in both object erasure and explicit content removal tasks. 

---
# CLAPP: The CLASS LLM Agent for Pair Programming 

**Authors**: Santiago Casas, Christian Fidler, Boris Bolliet, Francisco Villaescusa-Navarro, Julien Lesgourgues  

**Link**: [PDF](https://arxiv.org/pdf/2508.05728)  

**Abstract**: We introduce CLAPP (CLASS LLM Agent for Pair Programming), an interactive AI assistant designed to support researchers working with the Einstein-Boltzmann solver CLASS. CLAPP leverages large language models (LLMs) and domain-specific retrieval to provide conversational coding support for CLASS-answering questions, generating code, debugging errors, and producing plots. Its architecture combines multi-agent LLM orchestration, semantic search across CLASS documentation, and a live Python execution environment. Deployed as a user-friendly web application, CLAPP lowers the entry barrier for scientists unfamiliar with AI tools and enables more productive human-AI collaboration in computational and numerical cosmology. The app is available at this https URL 

---
# Klear-CodeTest: Scalable Test Case Generation for Code Reinforcement Learning 

**Authors**: Jia Fu, Xinyu Yang, Hongzhi Zhang, Yahui Liu, Jingyuan Zhang, Qi Wang, Fuzheng Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.05710)  

**Abstract**: Precise, correct feedback is crucial for effectively training large language models (LLMs) in code reinforcement learning. However, synthesizing high-quality test cases remains a profoundly challenging and unsolved problem. In this work, we present Klear-CodeTest, a comprehensive test case synthesis framework featuring rigorous verification to ensure quality and reliability of test cases. Our approach achieves broad coverage of programming problems via a novel Generator-Validation (G-V) framework, ensuring correctness through a consistency validation mechanism that verifies outputs against gold solutions. The proposed G-V framework generates comprehensive test cases including both regular and corner cases, enhancing test coverage and discriminative power for solution correctness assessment in code reinforcement learning. In addition, we design a multi-layered security sandbox system optimized for online verification platforms, guaranteeing safe and reliable code execution. Through comprehensive experiments, we demonstrate the effectiveness of our curated dataset, showing significant improvements in model performance and training stability. The source codes, curated dataset and sandbox system are available at: this https URL. 

---
# A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes 

**Authors**: Valentina Roquemen-Echeverri, Taisa Kushner, Peter G. Jacobs, Clara Mosquera-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.05705)  

**Abstract**: Simulating glucose dynamics in individuals with type 1 diabetes (T1D) is critical for developing personalized treatments and supporting data-driven clinical decisions. Existing models often miss key physiological aspects and are difficult to individualize. Here, we introduce physiologically-constrained neural network (NN) digital twins to simulate glucose dynamics in T1D. To ensure interpretability and physiological consistency, we first build a population-level NN state-space model aligned with a set of ordinary differential equations (ODEs) describing glucose regulation. This model is formally verified to conform to known T1D dynamics. Digital twins are then created by augmenting the population model with individual-specific models, which include personal data, such as glucose management and contextual information, capturing both inter- and intra-individual variability. We validate our approach using real-world data from the T1D Exercise Initiative study. Two weeks of data per participant were split into 5-hour sequences and simulated glucose profiles were compared to observed ones. Clinically relevant outcomes were used to assess similarity via paired equivalence t-tests with predefined clinical equivalence margins. Across 394 digital twins, glucose outcomes were equivalent between simulated and observed data: time in range (70-180 mg/dL) was 75.1$\pm$21.2% (simulated) vs. 74.4$\pm$15.4% (real; P<0.001); time below range (<70 mg/dL) 2.5$\pm$5.2% vs. 3.0$\pm$3.3% (P=0.022); and time above range (>180 mg/dL) 22.4$\pm$22.0% vs. 22.6$\pm$15.9% (P<0.001). Our framework can incorporate unmodeled factors like sleep and activity while preserving key dynamics. This approach enables personalized in silico testing of treatments, supports insulin optimization, and integrates physics-based and data-driven modeling. Code: this https URL 

---
# Semantic Reasoning Meets Numerical Precision: An LLM-Powered Multi-Agent System for Power Grid Control 

**Authors**: Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05702)  

**Abstract**: The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions. 

---
# Multi-Faceted Large Embedding Tables for Pinterest Ads Ranking 

**Authors**: Runze Su, Jiayin Jin, Jiacheng Li, Sihan Wang, Guangtong Bai, Zelun Wang, Li Tang, Yixiong Meng, Huasen Wu, Zhimeng Pan, Kungang Li, Han Sun, Zhifang Liu, Haoyang Li, Siping Ji, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2508.05700)  

**Abstract**: Large embedding tables are indispensable in modern recommendation systems, thanks to their ability to effectively capture and memorize intricate details of interactions among diverse entities. As we explore integrating large embedding tables into Pinterest's ads ranking models, we encountered not only common challenges such as sparsity and scalability, but also several obstacles unique to our context. Notably, our initial attempts to train large embedding tables from scratch resulted in neutral metrics. To tackle this, we introduced a novel multi-faceted pretraining scheme that incorporates multiple pretraining algorithms. This approach greatly enriched the embedding tables and resulted in significant performance improvements. As a result, the multi-faceted large embedding tables bring great performance gain on both the Click-Through Rate (CTR) and Conversion Rate (CVR) domains. Moreover, we designed a CPU-GPU hybrid serving infrastructure to overcome GPU memory limits and elevate the scalability. This framework has been deployed in the Pinterest Ads system and achieved 1.34% online CPC reduction and 2.60% CTR increase with neutral end-to-end latency change. 

---
# Log2Sig: Frequency-Aware Insider Threat Detection via Multivariate Behavioral Signal Decomposition 

**Authors**: Kaichuan Kong, Dongjie Liu, Xiaobo Jin, Zhiying Li, Guanggang Geng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05696)  

**Abstract**: Insider threat detection presents a significant challenge due to the deceptive nature of malicious behaviors, which often resemble legitimate user operations. However, existing approaches typically model system logs as flat event sequences, thereby failing to capture the inherent frequency dynamics and multiscale disturbance patterns embedded in user behavior. To address these limitations, we propose Log2Sig, a robust anomaly detection framework that transforms user logs into multivariate behavioral frequency signals, introducing a novel representation of user behavior. Log2Sig employs Multivariate Variational Mode Decomposition (MVMD) to extract Intrinsic Mode Functions (IMFs), which reveal behavioral fluctuations across multiple temporal scales. Based on this, the model further performs joint modeling of behavioral sequences and frequency-decomposed signals: the daily behavior sequences are encoded using a Mamba-based temporal encoder to capture long-term dependencies, while the corresponding frequency components are linearly projected to match the encoder's output dimension. These dual-view representations are then fused to construct a comprehensive user behavior profile, which is fed into a multilayer perceptron for precise anomaly detection. Experimental results on the CERT r4.2 and r5.2 datasets demonstrate that Log2Sig significantly outperforms state-of-the-art baselines in both accuracy and F1 score. 

---
# DMFI: Dual-Modality Fine-Tuning and Inference Framework for LLM-Based Insider Threat Detection 

**Authors**: Kaichuan Kong, Dongjie Liu, Xiaobo Jin, Guanggang Geng, Zhiying Li, Jian Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05694)  

**Abstract**: Insider threat detection (ITD) poses a persistent and high-impact challenge in cybersecurity due to the subtle, long-term, and context-dependent nature of malicious insider behaviors. Traditional models often struggle to capture semantic intent and complex behavior dynamics, while existing LLM-based solutions face limitations in prompt adaptability and modality coverage. To bridge this gap, we propose DMFI, a dual-modality framework that integrates semantic inference with behavior-aware fine-tuning. DMFI converts raw logs into two structured views: (1) a semantic view that processes content-rich artifacts (e.g., emails, https) using instruction-formatted prompts; and (2) a behavioral abstraction, constructed via a 4W-guided (When-Where-What-Which) transformation to encode contextual action sequences. Two LoRA-enhanced LLMs are fine-tuned independently, and their outputs are fused via a lightweight MLP-based decision module. We further introduce DMFI-B, a discriminative adaptation strategy that separates normal and abnormal behavior representations, improving robustness under severe class imbalance. Experiments on CERT r4.2 and r5.2 datasets demonstrate that DMFI outperforms state-of-the-art methods in detection accuracy. Our approach combines the semantic reasoning power of LLMs with structured behavior modeling, offering a scalable and effective solution for real-world insider threat detection. Our work demonstrates the effectiveness of combining LLM reasoning with structured behavioral modeling, offering a scalable and deployable solution for modern insider threat detection. 

---
# Empirical Evaluation of AI-Assisted Software Package Selection: A Knowledge Graph Approach 

**Authors**: Siamak Farshidi, Amir Saberhabibi, Behbod Eskafi, Niloofar Nikfarjam, Sadegh Eskandari, Slinger Jansen, Michel Chaudron, Bedir Tekinerdogan  

**Link**: [PDF](https://arxiv.org/pdf/2508.05693)  

**Abstract**: Selecting third-party software packages in open-source ecosystems like Python is challenging due to the large number of alternatives and limited transparent evidence for comparison. Generative AI tools are increasingly used in development workflows, but their suggestions often overlook dependency evaluation, emphasize popularity over suitability, and lack reproducibility. This creates risks for projects that require transparency, long-term reliability, maintainability, and informed architectural decisions. This study formulates software package selection as a Multi-Criteria Decision-Making (MCDM) problem and proposes a data-driven framework for technology evaluation. Automated data pipelines continuously collect and integrate software metadata, usage trends, vulnerability information, and developer sentiment from GitHub, PyPI, and Stack Overflow. These data are structured into a decision model representing relationships among packages, domain features, and quality attributes. The framework is implemented in PySelect, a decision support system that uses large language models to interpret user intent and query the model to identify contextually appropriate packages. The approach is evaluated using 798,669 Python scripts from 16,887 GitHub repositories and a user study based on the Technology Acceptance Model. Results show high data extraction precision, improved recommendation quality over generative AI baselines, and positive user evaluations of usefulness and ease of use. This work introduces a scalable, interpretable, and reproducible framework that supports evidence-based software selection using MCDM principles, empirical data, and AI-assisted intent modeling. 

---
# Risk Analysis Techniques for Governed LLM-based Multi-Agent Systems 

**Authors**: Alistair Reid, Simon O'Callaghan, Liam Carroll, Tiberio Caetano  

**Link**: [PDF](https://arxiv.org/pdf/2508.05687)  

**Abstract**: Organisations are starting to adopt LLM-based AI agents, with their deployments naturally evolving from single agents towards interconnected, multi-agent networks. Yet a collection of safe agents does not guarantee a safe collection of agents, as interactions between agents over time create emergent behaviours and induce novel failure modes. This means multi-agent systems require a fundamentally different risk analysis approach than that used for a single agent.
This report addresses the early stages of risk identification and analysis for multi-agent AI systems operating within governed environments where organisations control their agent configurations and deployment. In this setting, we examine six critical failure modes: cascading reliability failures, inter-agent communication failures, monoculture collapse, conformity bias, deficient theory of mind, and mixed motive dynamics. For each, we provide a toolkit for practitioners to extend or integrate into their existing frameworks to assess these failure modes within their organisational contexts.
Given fundamental limitations in current LLM behavioural understanding, our approach centres on analysis validity, and advocates for progressively increasing validity through staged testing across stages of abstraction and deployment that gradually increases exposure to potential negative impacts, while collecting convergent evidence through simulation, observational analysis, benchmarking, and red teaming. This methodology establishes the groundwork for robust organisational risk management as these LLM-based multi-agent systems are deployed and operated. 

---
# Selection-Based Vulnerabilities: Clean-Label Backdoor Attacks in Active Learning 

**Authors**: Yuhan Zhi, Longtian Wang, Xiaofei Xie, Chao Shen, Qiang Hu, Xiaohong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2508.05681)  

**Abstract**: Active learning(AL), which serves as the representative label-efficient learning paradigm, has been widely applied in resource-constrained scenarios. The achievement of AL is attributed to acquisition functions, which are designed for identifying the most important data to label. Despite this success, one question remains unanswered: is AL safe? In this work, we introduce ALA, a practical and the first framework to utilize the acquisition function as the poisoning attack surface to reveal the weakness of active learning. Specifically, ALA optimizes imperceptibly poisoned inputs to exhibit high uncertainty scores, increasing their probability of being selected by acquisition functions. To evaluate ALA, we conduct extensive experiments across three datasets, three acquisition functions, and two types of clean-label backdoor triggers. Results show that our attack can achieve high success rates (up to 94%) even under low poisoning budgets (0.5%-1.0%) while preserving model utility and remaining undetectable to human annotators. Our findings remind active learning users: acquisition functions can be easily exploited, and active learning should be deployed with caution in trusted data scenarios. 

---
# Are All Genders Equal in the Eyes of Algorithms? -- Analysing Search and Retrieval Algorithms for Algorithmic Gender Fairness 

**Authors**: Stefanie Urchs, Veronika Thurner, Matthias Aßenmacher, Ludwig Bothmann, Christian Heumann, Stephanie Thiemichen  

**Link**: [PDF](https://arxiv.org/pdf/2508.05680)  

**Abstract**: Algorithmic systems such as search engines and information retrieval platforms significantly influence academic visibility and the dissemination of knowledge. Despite assumptions of neutrality, these systems can reproduce or reinforce societal biases, including those related to gender. This paper introduces and applies a bias-preserving definition of algorithmic gender fairness, which assesses whether algorithmic outputs reflect real-world gender distributions without introducing or amplifying disparities. Using a heterogeneous dataset of academic profiles from German universities and universities of applied sciences, we analyse gender differences in metadata completeness, publication retrieval in academic databases, and visibility in Google search results. While we observe no overt algorithmic discrimination, our findings reveal subtle but consistent imbalances: male professors are associated with a greater number of search results and more aligned publication records, while female professors display higher variability in digital visibility. These patterns reflect the interplay between platform algorithms, institutional curation, and individual self-presentation. Our study highlights the need for fairness evaluations that account for both technical performance and representational equality in digital systems. 

---
# Adversarial Attacks on Reinforcement Learning-based Medical Questionnaire Systems: Input-level Perturbation Strategies and Medical Constraint Validation 

**Authors**: Peizhuo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05677)  

**Abstract**: RL-based medical questionnaire systems have shown great potential in medical scenarios. However, their safety and robustness remain unresolved. This study performs a comprehensive evaluation on adversarial attack methods to identify and analyze their potential vulnerabilities. We formulate the diagnosis process as a Markov Decision Process (MDP), where the state is the patient responses and unasked questions, and the action is either to ask a question or to make a diagnosis. We implemented six prevailing major attack methods, including the Fast Gradient Signed Method (FGSM), Projected Gradient Descent (PGD), Carlini & Wagner Attack (C&W) attack, Basic Iterative Method (BIM), DeepFool, and AutoAttack, with seven epsilon values each. To ensure the generated adversarial examples remain clinically plausible, we developed a comprehensive medical validation framework consisting of 247 medical constraints, including physiological bounds, symptom correlations, and conditional medical constraints. We achieved a 97.6% success rate in generating clinically plausible adversarial samples. We performed our experiment on the National Health Interview Survey (NHIS) dataset (this https URL), which consists of 182,630 samples, to predict the participant's 4-year mortality rate. We evaluated our attacks on the AdaptiveFS framework proposed in arXiv:2004.00994. Our results show that adversarial attacks could significantly impact the diagnostic accuracy, with attack success rates ranging from 33.08% (FGSM) to 64.70% (AutoAttack). Our work has demonstrated that even under strict medical constraints on the input, such RL-based medical questionnaire systems still show significant vulnerabilities. 

---
# Principle-Guided Verilog Optimization: IP-Safe Knowledge Transfer via Local-Cloud Collaboration 

**Authors**: Jing Wang, Zheng Li, Lei Li, Fan He, Liyu Lin, Yao Lai, Yan Li, Xiaoyang Zeng, Yufeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05675)  

**Abstract**: Recent years have witnessed growing interest in adopting large language models (LLMs) for Register Transfer Level (RTL) code optimization. While powerful cloud-based LLMs offer superior optimization capabilities, they pose unacceptable intellectual property (IP) leakage risks when processing proprietary hardware designs. In this paper, we propose a new scenario where Verilog code must be optimized for specific attributes without leaking sensitive IP information. We introduce the first IP-preserving edge-cloud collaborative framework that leverages the benefits of both paradigms. Our approach employs local small LLMs (e.g., Qwen-2.5-Coder-7B) to perform secure comparative analysis between paired high-quality target designs and novice draft codes, yielding general design principles that summarize key insights for improvements. These principles are then used to query stronger cloud LLMs (e.g., Deepseek-V3) for targeted code improvement, ensuring that only abstracted and IP-safe guidance reaches external services. Our experimental results demonstrate that the framework achieves significantly higher optimization success rates compared to baseline methods. For example, combining Qwen-2.5-Coder-7B and Deepseek-V3 achieves a 66.67\% optimization success rate for power utilization, outperforming Deepseek-V3 alone (49.81\%) and even commercial models like GPT-4o (55.81\%). Further investigation of local and cloud LLM combinations reveals that different model pairings exhibit varying strengths for specific optimization objectives, with interesting trends emerging when varying the number of comparative code pairs. Our work establishes a new paradigm for secure hardware design optimization that balances performance gains with IP protection. 

---
# Towards Effective Offensive Security LLM Agents: Hyperparameter Tuning, LLM as a Judge, and a Lightweight CTF Benchmark 

**Authors**: Minghao Shao, Nanda Rani, Kimberly Milner, Haoran Xi, Meet Udeshi, Saksham Aggarwal, Venkata Sai Charan Putrevu, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2508.05674)  

**Abstract**: Recent advances in LLM agentic systems have improved the automation of offensive security tasks, particularly for Capture the Flag (CTF) challenges. We systematically investigate the key factors that drive agent success and provide a detailed recipe for building effective LLM-based offensive security agents. First, we present CTFJudge, a framework leveraging LLM as a judge to analyze agent trajectories and provide granular evaluation across CTF solving steps. Second, we propose a novel metric, CTF Competency Index (CCI) for partial correctness, revealing how closely agent solutions align with human-crafted gold standards. Third, we examine how LLM hyperparameters, namely temperature, top-p, and maximum token length, influence agent performance and automated cybersecurity task planning. For rapid evaluation, we present CTFTiny, a curated benchmark of 50 representative CTF challenges across binary exploitation, web, reverse engineering, forensics, and cryptography. Our findings identify optimal multi-agent coordination settings and lay the groundwork for future LLM agent research in cybersecurity. We make CTFTiny open source to public this https URL along with CTFJudge on this https URL. 

---
# Breaking the Top-$K$ Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems 

**Authors**: Weiqin Yang, Jiawei Chen, Shengjia Zhang, Peng Wu, Yuegang Sun, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05673)  

**Abstract**: In the realm of recommender systems (RS), Top-$K$ ranking metrics such as NDCG@$K$ are the gold standard for evaluating recommendation performance. However, during the training of recommendation models, optimizing NDCG@$K$ poses significant challenges due to its inherent discontinuous nature and the intricate Top-$K$ truncation. Recent efforts to optimize NDCG@$K$ have either overlooked the Top-$K$ truncation or suffered from high computational costs and training instability. To overcome these limitations, we propose SoftmaxLoss@$K$ (SL@$K$), a novel recommendation loss tailored for NDCG@$K$ optimization. Specifically, we integrate the quantile technique to handle Top-$K$ truncation and derive a smooth upper bound for optimizing NDCG@$K$ to address discontinuity. The resulting SL@$K$ loss has several desirable properties, including theoretical guarantees, ease of implementation, computational efficiency, gradient stability, and noise robustness. Extensive experiments on four real-world datasets and three recommendation backbones demonstrate that SL@$K$ outperforms existing losses with a notable average improvement of 6.03%. The code is available at this https URL. 

---
# LMAR: Language Model Augmented Retriever for Domain-specific Knowledge Indexing 

**Authors**: Yao Zhao, Yantian Ding, Zhiyue Zhang, Dapeng Yao, Yanxun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05672)  

**Abstract**: Retrieval Augmented Generation (RAG) systems often struggle with domain-specific knowledge due to performance deterioration of pre-trained embeddings and prohibitive computational costs of large language model (LLM)-based retrievers. While fine-tuning data augmentation embedding models offers a promising direction, its effectiveness is limited by the need for high-quality training data and reliable chunking strategies that preserve contextual integrity. We propose LMAR (Language Model Augmented Retriever), a model-agnostic framework that addresses these challenges by combining LLM-guided data synthesis with contrastive embedding adaptation and efficient text clustering. LMAR consists of a two-stage pipeline: (1) Triplet sampling and synthetic data augmentation, where LLMs act as both labeler and validator to ensure high-fidelity supervision throughout the pipeline. Experimental results across multiple domain-specific benchmark datasets demonstrate that LMAR outperforms multiple baseline models, while maintaining moderate hardware requirements and low latency. Its model-agnostic nature further enables seamless integration with emerging RAG architectures and text embedding models, ensuring continual improvements without redesigning the pipeline. These results highlight LMAR as a practical and cost-effective solution for scalable domain-specific adaptation. 

---
# Can LLMs effectively provide game-theoretic-based scenarios for cybersecurity? 

**Authors**: Daniele Proverbio, Alessio Buscemi, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2508.05670)  

**Abstract**: Game theory has long served as a foundational tool in cybersecurity to test, predict, and design strategic interactions between attackers and defenders. The recent advent of Large Language Models (LLMs) offers new tools and challenges for the security of computer systems; In this work, we investigate whether classical game-theoretic frameworks can effectively capture the behaviours of LLM-driven actors and bots. Using a reproducible framework for game-theoretic LLM agents, we investigate two canonical scenarios -- the one-shot zero-sum game and the dynamic Prisoner's Dilemma -- and we test whether LLMs converge to expected outcomes or exhibit deviations due to embedded biases. Our experiments involve four state-of-the-art LLMs and span five natural languages, English, French, Arabic, Vietnamese, and Mandarin Chinese, to assess linguistic sensitivity. For both games, we observe that the final payoffs are influenced by agents characteristics such as personality traits or knowledge of repeated rounds. Moreover, we uncover an unexpected sensitivity of the final payoffs to the choice of languages, which should warn against indiscriminate application of LLMs in cybersecurity applications and call for in-depth studies, as LLMs may behave differently when deployed in different countries. We also employ quantitative metrics to evaluate the internal consistency and cross-language stability of LLM agents, to help guide the selection of the most stable LLMs and optimising models for secure applications. 

---
# Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports 

**Authors**: Jin Khye Tan, En Jun Choong, Ethan Jeremiah Chitty, Yan Pheng Choo, John Hsin Yang Wong, Chern Eu Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2508.05669)  

**Abstract**: Accurately extracting and representing the structure of tabular data from financial documents remains a critical challenge in document understanding, particularly for regulatory and analytical use cases. This study addresses the complexity of converting financial tables from Malaysian audited financial reports into Markdown format, a task complicated by rotated layouts, multi-level headers, and implicit structural cues. We propose a fine-tuned vision-language model (VLM), based on Qwen2.5-VL-7B, optimized for high-fidelity Markdown generation from document images. Our approach includes a curated dataset of 2,152 image-text pairs with augmentations and a supervised fine-tuning strategy using LoRA. To assess performance, we evaluated our model on 100 out-of-sample tables using a dual framework: a criteria-based LLM-as-a-judge for fine-grained accuracy and our novel Markdown Tree-Edit-Distance-based Similarity (TEDS) metric for holistic structural fidelity. Our model achieves a 92.20% overall accuracy on the criteria-based assessment and a 96.53% Markdown TEDS score. This performance significantly surpasses its Qwen2.5-VL-7B base model, larger-scale VLMs, and specialized reasoning-enabled models. Compared to these self-hosted alternatives, it also significantly reduces inference time. Furthermore, its accuracy exceeds that of widely used proprietary models such as OpenAI's GPT-4o and Gemini 2.5 Flash. These results demonstrate that domain-specific fine-tuning provides an effective and efficient method to bridge the gap between unstructured financial documents and downstream automation, rivalling much larger and more general models without their computational overhead. 

---
# A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges 

**Authors**: Yunjia Xi, Jianghao Lin, Yongzhao Xiao, Zheli Zhou, Rong Shan, Te Gao, Jiachen Zhu, Weiwen Liu, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05668)  

**Abstract**: The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on this https URL. 

---
# ITDR: An Instruction Tuning Dataset for Enhancing Large Language Models in Recommendations 

**Authors**: Zekun Liu, Xiaowen Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05667)  

**Abstract**: Large language models (LLMs) have demonstrated outstanding performance in natural language processing tasks. However, in the field of recommendation systems, due to the structural differences between user behavior data and natural language, LLMs struggle to effectively model the associations between user preferences and items. Although prompt-based methods can generate recommendation results, their inadequate understanding of recommendation tasks leads to constrained performance. To address this gap, in this work, we construct a sufficient instruction tuning dataset, ITDR, which encompasses 7 subtasks across two core root tasks--user-item interaction and user-item understanding. The dataset integrates data from 13 public recommendation datasets and is built using manually crafted standardized templates, comprising approximately 200,000 instances. Experimental results demonstrate that ITDR significantly enhances the performance of mainstream open-source LLMs such as GLM-4, Qwen2.5, Qwen2.5-Instruct and LLaMA-3.2 on recommendation tasks. Furthermore, we analyze the correlations between tasks and explore the impact of task descriptions and data scale on instruction tuning effectiveness. Finally, we perform comparative experiments against closed-source LLMs with substantial parameters. Our tuning dataset ITDR and the fine-tuned large recommendation models can be accessed at this https URL. 

---
# HySemRAG: A Hybrid Semantic Retrieval-Augmented Generation Framework for Automated Literature Synthesis and Methodological Gap Analysis 

**Authors**: Alejandro Godinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.05666)  

**Abstract**: We present HySemRAG, a framework that combines Extract, Transform, Load (ETL) pipelines with Retrieval-Augmented Generation (RAG) to automate large-scale literature synthesis and identify methodological research gaps. The system addresses limitations in existing RAG architectures through a multi-layered approach: hybrid retrieval combining semantic search, keyword filtering, and knowledge graph traversal; an agentic self-correction framework with iterative quality assurance; and post-hoc citation verification ensuring complete traceability. Our implementation processes scholarly literature through eight integrated stages: multi-source metadata acquisition, asynchronous PDF retrieval, custom document layout analysis using modified Docling architecture, bibliographic management, LLM-based field extraction, topic modeling, semantic unification, and knowledge graph construction. The system creates dual data products - a Neo4j knowledge graph enabling complex relationship queries and Qdrant vector collections supporting semantic search - serving as foundational infrastructure for verifiable information synthesis. Evaluation across 643 observations from 60 testing sessions demonstrates structured field extraction achieving 35.1% higher semantic similarity scores (0.655 $\pm$ 0.178) compared to PDF chunking approaches (0.485 $\pm$ 0.204, p < 0.000001). The agentic quality assurance mechanism achieves 68.3% single-pass success rates with 99.0% citation accuracy in validated responses. Applied to geospatial epidemiology literature on ozone exposure and cardiovascular disease, the system identifies methodological trends and research gaps, demonstrating broad applicability across scientific domains for accelerating evidence synthesis and discovery. 

---
# Enhancing Retrieval-Augmented Generation for Electric Power Industry Customer Support 

**Authors**: Hei Yu Chan, Kuok Tou Ho, Chenglong Ma, Yujing Si, Hok Lai Lin, Sa Lei Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.05664)  

**Abstract**: Many AI customer service systems use standard NLP pipelines or finetuned language models, which often fall short on ambiguous, multi-intent, or detail-specific queries. This case study evaluates recent techniques: query rewriting, RAG Fusion, keyword augmentation, intent recognition, and context reranking, for building a robust customer support system in the electric power domain. We compare vector-store and graph-based RAG frameworks, ultimately selecting the graph-based RAG for its superior performance in handling complex queries. We find that query rewriting improves retrieval for queries using non-standard terminology or requiring precise detail. RAG Fusion boosts performance on vague or multifaceted queries by merging multiple retrievals. Reranking reduces hallucinations by filtering irrelevant contexts. Intent recognition supports the decomposition of complex questions into more targeted sub-queries, increasing both relevance and efficiency. In contrast, keyword augmentation negatively impacts results due to biased keyword selection. Our final system combines intent recognition, RAG Fusion, and reranking to handle disambiguation and multi-source queries. Evaluated on both a GPT-4-generated dataset and a real-world electricity provider FAQ dataset, it achieves 97.9% and 89.6% accuracy respectively, substantially outperforming baseline RAG models. 

---
# From Static to Dynamic: A Streaming RAG Approach to Real-time Knowledge Base 

**Authors**: Yuzhou Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05662)  

**Abstract**: Dynamic streams from news feeds, social media, sensor networks, and financial markets challenge static RAG frameworks. Full-scale indices incur high memory costs; periodic rebuilds introduce latency that undermines data freshness; naive sampling sacrifices semantic coverage. We present Streaming RAG, a unified pipeline that combines multi-vector cosine screening, mini-batch clustering, and a counter-based heavy-hitter filter to maintain a compact prototype set. We further prove an approximation bound \$E\[R(K\_t)] \ge R^\* - L \Delta\$ linking retrieval quality to clustering variance. An incremental index upsert mechanism refreshes prototypes without interrupting queries. Experiments on eight real-time streams show statistically significant gains in Recall\@10 (up to 3 points, p < 0.01), end-to-end latency below 15 ms, and throughput above 900 documents per second under a 150 MB budget. Hyperparameter sensitivity analysis over cluster count, admission probability, relevance threshold, and counter capacity validates default settings. In open-domain question answering with GPT-3.5 Turbo, we record 3.2-point gain in Exact Match and 2.8-point gain in F1 on SQuAD; abstractive summarization yields ROUGE-L improvements. Streaming RAG establishes a new Pareto frontier for retrieval augmentation. 

---
# Zero-Shot Retrieval for Scalable Visual Search in a Two-Sided Marketplace 

**Authors**: Andre Rusli, Shoma Ishimoto, Sho Akiyama, Aman Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.05661)  

**Abstract**: Visual search offers an intuitive way for customers to explore diverse product catalogs, particularly in consumer-to-consumer (C2C) marketplaces where listings are often unstructured and visually driven. This paper presents a scalable visual search system deployed in Mercari's C2C marketplace, where end-users act as buyers and sellers. We evaluate recent vision-language models for zero-shot image retrieval and compare their performance with an existing fine-tuned baseline. The system integrates real-time inference and background indexing workflows, supported by a unified embedding pipeline optimized through dimensionality reduction. Offline evaluation using user interaction logs shows that the multilingual SigLIP model outperforms other models across multiple retrieval metrics, achieving a 13.3% increase in nDCG@5 over the baseline. A one-week online A/B test in production further confirms real-world impact, with the treatment group showing substantial gains in engagement and conversion, up to a 40.9% increase in transaction rate via image search. Our findings highlight that recent zero-shot models can serve as a strong and practical baseline for production use, which enables teams to deploy effective visual search systems with minimal overhead, while retaining the flexibility to fine-tune based on future data or domain-specific needs. 

---
# Open-Source Agentic Hybrid RAG Framework for Scientific Literature Review 

**Authors**: Aditya Nagori, Ricardo Accorsi Casonatto, Ayush Gautam, Abhinav Manikantha Sai Cheruvu, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2508.05660)  

**Abstract**: The surge in scientific publications challenges traditional review methods, demanding tools that integrate structured metadata with full-text analysis. Hybrid Retrieval Augmented Generation (RAG) systems, combining graph queries with vector search offer promise but are typically static, rely on proprietary tools, and lack uncertainty estimates. We present an agentic approach that encapsulates the hybrid RAG pipeline within an autonomous agent capable of (1) dynamically selecting between GraphRAG and VectorRAG for each query, (2) adapting instruction-tuned generation in real time to researcher needs, and (3) quantifying uncertainty during inference. This dynamic orchestration improves relevance, reduces hallucinations, and promotes reproducibility.
Our pipeline ingests bibliometric open-access data from PubMed, arXiv, and Google Scholar APIs, builds a Neo4j citation-based knowledge graph (KG), and embeds full-text PDFs into a FAISS vector store (VS) using the all-MiniLM-L6-v2 model. A Llama-3.3-70B agent selects GraphRAG (translating queries to Cypher for KG) or VectorRAG (combining sparse and dense retrieval with re-ranking). Instruction tuning refines domain-specific generation, and bootstrapped evaluation yields standard deviation for evaluation metrics.
On synthetic benchmarks mimicking real-world queries, the Instruction-Tuned Agent with Direct Preference Optimization (DPO) outperforms the baseline, achieving a gain of 0.63 in VS Context Recall and a 0.56 gain in overall Context Precision. Additional gains include 0.24 in VS Faithfulness, 0.12 in both VS Precision and KG Answer Relevance, 0.11 in overall Faithfulness score, 0.05 in KG Context Recall, and 0.04 in both VS Answer Relevance and overall Precision. These results highlight the system's improved reasoning over heterogeneous sources and establish a scalable framework for autonomous, agentic scientific discovery. 

---
# Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation 

**Authors**: Haozhe Xu, Xiaohua Wang, Changze Lv, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05657)  

**Abstract**: Conversational recommender systems (CRSs) enhance recommendation quality by engaging users in multi-turn dialogues, capturing nuanced preferences through natural language interactions. However, these systems often face the false negative issue, where items that a user might like are incorrectly labeled as negative during training, leading to suboptimal this http URL the label set through data augmentation presents an intuitive solution but faces the challenge of balancing two key aspects: ensuring semantic relevance and preserving the collaborative information inherent in CRS datasets. To address these issues, we propose a novel data augmentation framework that first leverages an LLM-based semantic retriever to identify diverse and semantically relevant items, which are then filtered by a relevance scorer to remove noisy candidates. Building on this, we introduce a two-stage training strategy balancing semantic relevance and collaborative information. Extensive experiments on two benchmark datasets and user simulators demonstrate significant and consistent performance improvements across various recommenders, highlighting the effectiveness of our approach in advancing CRS performance. 

---
# Comparison of Information Retrieval Techniques Applied to IT Support Tickets 

**Authors**: Leonardo Santiago Benitez Pereira, Robinson Pizzio, Samir Bonho  

**Link**: [PDF](https://arxiv.org/pdf/2508.05654)  

**Abstract**: Institutions dependent on IT services and resources acknowledge the crucial significance of an IT help desk system, that act as a centralized hub connecting IT staff and users for service requests. Employing various Machine Learning models, these IT help desk systems allow access to corrective actions used in the past, but each model has different performance when applied to different datasets. This work compares eleven Information Retrieval techniques in a dataset of IT support tickets, with the goal of implementing a software that facilitates the work of Information Technology support analysts. The best results were obtained with the Sentence-BERT technique, in its multi-language variation distilluse-base-multilingual-cased-v1, where 78.7% of the recommendations made by the model were considered relevant. TF-IDF (69.0%), Word2vec (68.7%) and LDA (66.3%) techniques also had consistent results. Furthermore, the used datasets and essential parts of coding have been published and made open source. It also demonstrated the practicality of a support ticket recovery system by implementing a minimal viable prototype, and described in detail the implementation of the system. Finally, this work proposed a novel metric for comparing the techniques, whose aim is to closely reflect the perception of the IT analysts about the retrieval quality. 

---
# Modeling Interactive Narrative Systems: A Formal Approach 

**Authors**: Jules Clerc, Domitile Lourdeaux, Mohamed Sallak, Johann Barbier, Marc Ravaine  

**Link**: [PDF](https://arxiv.org/pdf/2508.05653)  

**Abstract**: Interactive Narrative Systems (INS) have revolutionized digital experiences by empowering users to actively shape their stories, diverging from traditional passive storytelling. However, the field faces challenges due to fragmented research efforts and diverse system representations. This paper introduces a formal representation framework for INS, inspired by diverse approaches from the state of the art. By providing a consistent vocabulary and modeling structure, the framework facilitates the analysis, the description and comparison of INS properties. Experimental validations on the "Little Red Riding Hood" scenario highlight the usefulness of the proposed formalism and its impact on improving the evaluation of INS. This work aims to foster collaboration and coherence within the INS research community by proposing a methodology for formally representing these systems. 

---
# Lessons from A Large Language Model-based Outdoor Trail Recommendation Chatbot with Retrieval Augmented Generation 

**Authors**: Julia Ann Mathew, Suining He  

**Link**: [PDF](https://arxiv.org/pdf/2508.05652)  

**Abstract**: The increasing popularity of outdoor recreational activities (such as hiking and biking) has boosted the demand for a conversational AI system to provide informative and personalized suggestion on outdoor trails. Challenges arise in response to (1) how to provide accurate outdoor trail information via conversational AI; and (2) how to enable usable and efficient recommendation services. To address above, this paper discusses the preliminary and practical lessons learned from developing Judy, an outdoor trail recommendation chatbot based on the large language model (LLM) with retrieval augmented generation (RAG). To gain concrete system insights, we have performed case studies with the outdoor trails in Connecticut (CT), US. We have conducted web-based data collection, outdoor trail data management, and LLM model performance studies on the RAG-based recommendation. Our experimental results have demonstrated the accuracy, effectiveness, and usability of Judy in recommending outdoor trails based on the LLM with RAG. 

---
# OmniBench-RAG: A Multi-Domain Evaluation Platform for Retrieval-Augmented Generation Tools 

**Authors**: Jiaxuan Liang, Shide Zhou, Kailong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05650)  

**Abstract**: While Retrieval Augmented Generation (RAG) is now widely adopted to enhance LLMs, evaluating its true performance benefits in a reproducible and interpretable way remains a major hurdle. Existing methods often fall short: they lack domain coverage, employ coarse metrics that miss sub document precision, and fail to capture computational trade offs. Most critically, they provide no standardized framework for comparing RAG effectiveness across different models and domains.
We introduce OmniBench RAG, a novel automated platform for multi domain evaluation of RAG systems. The platform quantifies performance gains across accuracy and efficiency dimensions, spanning nine knowledge fields including culture, geography, and health. We introduce two standardized metrics: Improvements (accuracy gains) and Transformation (efficiency differences between pre RAG and post RAG models), enabling reproducible comparisons across models and tasks. The platform features dynamic test generation, modular evaluation pipelines, and automated knowledge base construction. Our evaluation reveals striking variability in RAG effectiveness, from significant gains in culture to declines in mathematics, highlighting the critical importance of systematic, domain aware assessment. A demonstration video is available at: this https URL. Code and datasets: this https URL. 

---
# AquiLLM: a RAG Tool for Capturing Tacit Knowledge in Research Groups 

**Authors**: Chandler Campbell, Bernie Boscoe, Tuan Do  

**Link**: [PDF](https://arxiv.org/pdf/2508.05648)  

**Abstract**: Research groups face persistent challenges in capturing, storing, and retrieving knowledge that is distributed across team members. Although structured data intended for analysis and publication is often well managed, much of a group's collective knowledge remains informal, fragmented, or undocumented--often passed down orally through meetings, mentoring, and day-to-day collaboration. This includes private resources such as emails, meeting notes, training materials, and ad hoc documentation. Together, these reflect the group's tacit knowledge--the informal, experience-based expertise that underlies much of their work. Accessing this knowledge can be difficult, requiring significant time and insider understanding. Retrieval-augmented generation (RAG) systems offer promising solutions by enabling users to query and generate responses grounded in relevant source material. However, most current RAG-LLM systems are oriented toward public documents and overlook the privacy concerns of internal research materials. We introduce AquiLLM (pronounced ah-quill-em), a lightweight, modular RAG system designed to meet the needs of research groups. AquiLLM supports varied document types and configurable privacy settings, enabling more effective access to both formal and informal knowledge within scholarly groups. 

---
# Query-Aware Graph Neural Networks for Enhanced Retrieval-Augmented Generation 

**Authors**: Vibhor Agrawal, Fay Wang, Rishi Puri  

**Link**: [PDF](https://arxiv.org/pdf/2508.05647)  

**Abstract**: We present a novel graph neural network (GNN) architecture for retrieval-augmented generation (RAG) that leverages query-aware attention mechanisms and learned scoring heads to improve retrieval accuracy on complex, multi-hop questions. Unlike traditional dense retrieval methods that treat documents as independent entities, our approach constructs per-episode knowledge graphs that capture both sequential and semantic relationships between text chunks. We introduce an Enhanced Graph Attention Network with query-guided pooling that dynamically focuses on relevant parts of the graph based on user queries. Experimental results demonstrate that our approach significantly outperforms standard dense retrievers on complex question answering tasks, particularly for questions requiring multi-document reasoning. Our implementation leverages PyTorch Geometric for efficient processing of graph-structured data, enabling scalable deployment in production retrieval systems 

---
# Request-Only Optimization for Recommendation Systems 

**Authors**: Liang Guo, Wei Li, Lucy Liao, Huihui Cheng, Rui Zhang, Yu Shi, Yueming Wang, Yanzun Huang, Keke Zhai, Pengchao Wang, Timothy Shi, Xuan Cao, Shengzhi Wang, Renqin Cai, Zhaojie Gong, Omkar Vichare, Rui Jian, Leon Gao, Shiyan Deng, Xingyu Liu, Xiong Zhang, Fu Li, Wenlei Xie, Bin Wen, Rui Li, Xing Liu, Jiaqi Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.05640)  

**Abstract**: Deep Learning Recommendation Models (DLRMs) represent one of the largest machine learning applications on the planet. Industry-scale DLRMs are trained with petabytes of recommendation data to serve billions of users every day. To utilize the rich user signals in the long user history, DLRMs have been scaled up to unprecedented complexity, up to trillions of floating-point operations (TFLOPs) per example. This scale, coupled with the huge amount of training data, necessitates new storage and training algorithms to efficiently improve the quality of these complex recommendation systems. In this paper, we present a Request-Only Optimizations (ROO) training and modeling paradigm. ROO simultaneously improves the storage and training efficiency as well as the model quality of recommendation systems. We holistically approach this challenge through co-designing data (i.e., request-only data), infrastructure (i.e., request-only based data processing pipeline), and model architecture (i.e., request-only neural architectures). Our ROO training and modeling paradigm treats a user request as a unit of the training data. Compared with the established practice of treating a user impression as a unit, our new design achieves native feature deduplication in data logging, consequently saving data storage. Second, by de-duplicating computations and communications across multiple impressions in a request, this new paradigm enables highly scaled-up neural network architectures to better capture user interest signals, such as Generative Recommenders (GRs) and other request-only friendly architectures. 

---
# Automated Visualization Makeovers with LLMs 

**Authors**: Siddharth Gangwar, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2508.05637)  

**Abstract**: Making a good graphic that accurately and efficiently conveys the desired message to the audience is both an art and a science, typically not taught in the data science curriculum. Visualisation makeovers are exercises where the community exchange feedback to improve charts and data visualizations. Can multi-modal large language models (LLMs) emulate this task? Given a plot in the form of an image file, or the code used to generate it, an LLM, primed with a list of visualization best practices, is employed to semi-automatically generate constructive criticism to produce a better plot. Our system is centred around prompt engineering of a pre-trained model, relying on a combination of userspecified guidelines and any latent knowledge of data visualization practices that might lie within an LLMs training corpus. Unlike other works, the focus is not on generating valid visualization scripts from raw data or prompts, but on educating the user how to improve their existing data visualizations according to an interpretation of best practices. A quantitative evaluation is performed to measure the sensitivity of the LLM agent to various plotting issues across different chart types. We make the tool available as a simple self-hosted applet with an accessible Web interface. 

---
# AttriLens-Mol: Attribute Guided Reinforcement Learning for Molecular Property Prediction with Large Language Models 

**Authors**: Xuan Lin, Long Chen, Yile Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04748)  

**Abstract**: Large Language Models (LLMs) have shown promise in assisting molecular property prediction tasks but often rely on human-crafted prompts and chain-of-thought templates. While recent advanced large reasoning models like DeepSeek-R1 employ reinforcement learning for an extended ``thinking'' process, their reasoning can be verbose and lack relevance. We introduce AttriLens-Mol, an attribute-guided reinforcement learning framework for molecular property prediction with LLMs. AttriLens-Mol steers the model's reasoning by using: (1) a format reward encouraging attribute-based structured output, (2) a count reward to avoid enumerating irrelevant attributes, and (3) a rationality reward using advanced LLMs and RDKit to verify the relatedness of the generated attributes. This approach implicitly elicits the model's inherent knowledge of relevant molecular attributes during reasoning, enables making predictions for the molecular property more effectively. Experiments on both in-distribution and out-of-distribution datasets show that, training both 7B-size R1-Distilled-Qwen2.5 and R1-Distilled-LLaMA3.1 models on 4,000 samples with our proposed AttriLens-Mol method significantly boosts the performance, getting comparable or better results than supervised fine-tuning models (Mol-Instructions, ChemDFM, etc.) and advanced models (GPT-3.5, GPT-4o, DeepSeek-V3, DeepSeek-R1, etc.). Further, our extracted attributes for the target property, when used as features for an interpretable decision tree model, yield superior performance compared to attributes generated by prompting LLMs. This shows that AttriLens-Mol effectively elicits more relevant and predictive molecular attributes, leading to enhanced interpretability and performance for property prediction. We release the code in this https URL. 

---
# SHACL Validation in the Presence of Ontologies: Semantics and Rewriting Techniques 

**Authors**: Anouk Oudshoorn, Magdalena Ortiz, Mantas Simkus  

**Link**: [PDF](https://arxiv.org/pdf/2507.12286)  

**Abstract**: SHACL and OWL are two prominent W3C standards for managing RDF data. These languages share many features, but they have one fundamental difference: OWL, designed for inferring facts from incomplete data, makes the open-world assumption, whereas SHACL is a constraint language that treats the data as complete and must be validated under the closed-world assumption. The combination of both formalisms is very appealing and has been called for, but their semantic gap is a major challenge, semantically and computationally. In this paper, we advocate a semantics for SHACL validation in the presence of ontologies based on core universal models. We provide a technique for constructing these models for ontologies in the rich data-tractable description logic Horn-ALCHIQ. Furthermore, we use a finite representation of this model to develop a rewriting technique that reduces SHACL validation in the presence of ontologies to standard validation. Finally, we study the complexity of SHACL validation in the presence of ontologies, and show that even very simple ontologies make the problem EXPTIME-complete, and PTIME-complete in data complexity. 

---
# Epidemic Control on a Large-Scale-Agent-Based Epidemiology Model using Deep Deterministic Policy Gradient 

**Authors**: Gaurav Deshkar, Jayanta Kshirsagar, Harshal Hayatnagarkar, Janani Venugopalan  

**Link**: [PDF](https://arxiv.org/pdf/2304.04475)  

**Abstract**: To mitigate the impact of the pandemic, several measures include lockdowns, rapid vaccination programs, school closures, and economic stimulus. These interventions can have positive or unintended negative consequences. Current research to model and determine an optimal intervention automatically through round-tripping is limited by the simulation objectives, scale (a few thousand individuals), model types that are not suited for intervention studies, and the number of intervention strategies they can explore (discrete vs continuous). We address these challenges using a Deep Deterministic Policy Gradient (DDPG) based policy optimization framework on a large-scale (100,000 individual) epidemiological agent-based simulation where we perform multi-objective optimization. We determine the optimal policy for lockdown and vaccination in a minimalist age-stratified multi-vaccine scenario with a basic simulation for economic activity. With no lockdown and vaccination (mid-age and elderly), results show optimal economy (individuals below the poverty line) with balanced health objectives (infection, and hospitalization). An in-depth simulation is needed to further validate our results and open-source our framework. 

---
