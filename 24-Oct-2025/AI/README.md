# Real Deep Research for AI, Robotics and Beyond 

**Authors**: Xueyan Zou, Jianglong Ye, Hao Zhang, Xiaoyu Xiang, Mingyu Ding, Zhaojing Yang, Yong Jae Lee, Zhuowen Tu, Sifei Liu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20809)  

**Abstract**: With the rapid growth of research in AI and robotics now producing over 10,000 papers annually it has become increasingly difficult for researchers to stay up to date. Fast evolving trends, the rise of interdisciplinary work, and the need to explore domains beyond one's expertise all contribute to this challenge. To address these issues, we propose a generalizable pipeline capable of systematically analyzing any research area: identifying emerging trends, uncovering cross domain opportunities, and offering concrete starting points for new inquiry. In this work, we present Real Deep Research (RDR) a comprehensive framework applied to the domains of AI and robotics, with a particular focus on foundation models and robotics advancements. We also briefly extend our analysis to other areas of science. The main paper details the construction of the RDR pipeline, while the appendix provides extensive results across each analyzed topic. We hope this work sheds light for researchers working in the field of AI and beyond. 

---
# A Coherence-Based Measure of AGI 

**Authors**: Fares Fourati  

**Link**: [PDF](https://arxiv.org/pdf/2510.20784)  

**Abstract**: Recent work by \citet{hendrycks2025agidefinition} formalized \textit{Artificial General Intelligence} (AGI) as the arithmetic mean of proficiencies across cognitive domains derived from the Cattell--Horn--Carroll (CHC) model of human cognition. While elegant, this definition assumes \textit{compensability} -- that exceptional ability in some domains can offset failure in others. True general intelligence, however, should reflect \textit{coherent sufficiency}: balanced competence across all essential domains. We propose a coherence-aware measure of AGI based on the integral of generalized means over a continuum of compensability exponents. This formulation spans arithmetic, geometric, and harmonic regimes, and the resulting \textit{area under the curve} (AUC) quantifies robustness under varying compensability assumptions. Unlike the arithmetic mean, which rewards specialization, the AUC penalizes imbalance and captures inter-domain dependency. Applied to published CHC-based domain scores for GPT-4 and GPT-5, the coherence-adjusted AUC reveals that both systems remain far from general competence despite high arithmetic scores (e.g., GPT-5 at~24\%). Integrating the generalized mean thus yields a principled, interpretable, and stricter foundation for measuring genuine progress toward AGI. 

---
# Plan Then Retrieve: Reinforcement Learning-Guided Complex Reasoning over Knowledge Graphs 

**Authors**: Yanlin Song, Ben Liu, Víctor Gutiérrez-Basulto, Zhiwei Hu, Qianqian Xie, Min Peng, Sophia Ananiadou, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20691)  

**Abstract**: Knowledge Graph Question Answering aims to answer natural language questions by reasoning over structured knowledge graphs. While large language models have advanced KGQA through their strong reasoning capabilities, existing methods continue to struggle to fully exploit both the rich knowledge encoded in KGs and the reasoning capabilities of LLMs, particularly in complex scenarios. They often assume complete KG coverage and lack mechanisms to judge when external information is needed, and their reasoning remains locally myopic, failing to maintain coherent multi-step planning, leading to reasoning failures even when relevant knowledge exists. We propose Graph-RFT, a novel two-stage reinforcement fine-tuning KGQA framework with a 'plan-KGsearch-and-Websearch-during-think' paradigm, that enables LLMs to perform autonomous planning and adaptive retrieval scheduling across KG and web sources under incomplete knowledge conditions. Graph-RFT introduces a chain-of-thought fine-tuning method with a customized plan-retrieval dataset activates structured reasoning and resolves the GRPO cold-start problem. It then introduces a novel plan-retrieval guided reinforcement learning process integrates explicit planning and retrieval actions with a multi-reward design, enabling coverage-aware retrieval scheduling. It employs a Cartesian-inspired planning module to decompose complex questions into ordered subquestions, and logical expression to guide tool invocation for globally consistent multi-step reasoning. This reasoning retrieval process is optimized with a multi-reward combining outcome and retrieval specific signals, enabling the model to learn when and how to combine KG and web retrieval effectively. 

---
# The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models 

**Authors**: Xue Wen Tan, Nathaniel Tan, Galen Lee, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2510.20665)  

**Abstract**: Evaluating the quality of reasoning traces from large language models remains understudied, labor-intensive, and unreliable: current practice relies on expert rubrics, manual annotation, and slow pairwise judgments. Automated efforts are dominated by graph-based proxies that quantify structural connectivity but do not clarify what constitutes high-quality reasoning; such abstractions can be overly simplistic for inherently complex processes. We introduce a topological data analysis (TDA)-based evaluation framework that captures the geometry of reasoning traces and enables label-efficient, automated assessment. In our empirical study, topological features yield substantially higher predictive power for assessing reasoning quality than standard graph metrics, suggesting that effective reasoning is better captured by higher-dimensional geometric structures rather than purely relational graphs. We further show that a compact, stable set of topological features reliably indicates trace quality, offering a practical signal for future reinforcement learning algorithms. 

---
# Integrating Machine Learning into Belief-Desire-Intention Agents: Current Advances and Open Challenges 

**Authors**: Andrea Agiollo, Andrea Omicini  

**Link**: [PDF](https://arxiv.org/pdf/2510.20641)  

**Abstract**: Thanks to the remarkable human-like capabilities of machine learning (ML) models in perceptual and cognitive tasks, frameworks integrating ML within rational agent architectures are gaining traction. Yet, the landscape remains fragmented and incoherent, often focusing on embedding ML into generic agent containers while overlooking the expressive power of rational architectures--such as Belief-Desire-Intention (BDI) agents. This paper presents a fine-grained systematisation of existing approaches, using the BDI paradigm as a reference. Our analysis illustrates the fast-evolving literature on rational agents enhanced by ML, and identifies key research opportunities and open challenges for designing effective rational ML agents. 

---
# Fluidity Index: Next-Generation Super-intelligence Benchmarks 

**Authors**: Eric Ngoiya, Tianshu Bao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20636)  

**Abstract**: This paper introduces the Fluidity Index (FI) to quantify model adaptability in dynamic, scaling environments. The benchmark evaluates response accuracy based on deviations in initial, current, and future environment states, assessing context switching and continuity. We distinguish between closed-ended and open-ended benchmarks, prioritizing closed-loop open-ended real-world benchmarks to test adaptability. The approach measures a model's ability to understand, predict, and adjust to state changes in scaling environments. A truly super-intelligent model should exhibit at least second-order adaptability, enabling self-sustained computation through digital replenishment for optimal fluidity. 

---
# Towards Reliable Evaluation of Large Language Models for Multilingual and Multimodal E-Commerce Applications 

**Authors**: Shuyi Xie, Ziqin Liew, Hailing Zhang, Haibo Zhang, Ling Hu, Zhiqiang Zhou, Shuman Liu, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.20632)  

**Abstract**: Large Language Models (LLMs) excel on general-purpose NLP benchmarks, yet their capabilities in specialized domains remain underexplored. In e-commerce, existing evaluations-such as EcomInstruct, ChineseEcomQA, eCeLLM, and Shopping MMLU-suffer from limited task diversity (e.g., lacking product guidance and after-sales issues), limited task modalities (e.g., absence of multimodal data), synthetic or curated data, and a narrow focus on English and Chinese, leaving practitioners without reliable tools to assess models on complex, real-world shopping scenarios. We introduce EcomEval, a comprehensive multilingual and multimodal benchmark for evaluating LLMs in e-commerce. EcomEval covers six categories and 37 tasks (including 8 multimodal tasks), sourced primarily from authentic customer queries and transaction logs, reflecting the noisy and heterogeneous nature of real business interactions. To ensure both quality and scalability of reference answers, we adopt a semi-automatic pipeline in which large models draft candidate responses subsequently reviewed and modified by over 50 expert annotators with strong e-commerce and multilingual expertise. We define difficulty levels for each question and task category by averaging evaluation scores across models with different sizes and capabilities, enabling challenge-oriented and fine-grained assessment. EcomEval also spans seven languages-including five low-resource Southeast Asian languages-offering a multilingual perspective absent from prior work. 

---
# Towards the Formalization of a Trustworthy AI for Mining Interpretable Models explOiting Sophisticated Algorithms 

**Authors**: Riccardo Guidotti, Martina Cinquini, Marta Marchiori Manerba, Mattia Setzu, Francesco Spinnato  

**Link**: [PDF](https://arxiv.org/pdf/2510.20621)  

**Abstract**: Interpretable-by-design models are crucial for fostering trust, accountability, and safe adoption of automated decision-making models in real-world applications. In this paper we formalize the ground for the MIMOSA (Mining Interpretable Models explOiting Sophisticated Algorithms) framework, a comprehensive methodology for generating predictive models that balance interpretability with performance while embedding key ethical properties. We formally define here the supervised learning setting across diverse decision-making tasks and data types, including tabular data, time series, images, text, transactions, and trajectories. We characterize three major families of interpretable models: feature importance, rule, and instance based models. For each family, we analyze their interpretability dimensions, reasoning mechanisms, and complexity. Beyond interpretability, we formalize three critical ethical properties, namely causality, fairness, and privacy, providing formal definitions, evaluation metrics, and verification procedures for each. We then examine the inherent trade-offs between these properties and discuss how privacy requirements, fairness constraints, and causal reasoning can be embedded within interpretable pipelines. By evaluating ethical measures during model generation, this framework establishes the theoretical foundations for developing AI systems that are not only accurate and interpretable but also fair, privacy-preserving, and causally aware, i.e., trustworthy. 

---
# Efficient Algorithms for Computing Random Walk Centrality 

**Authors**: Changan Liu, Zixuan Xie, Ahad N. Zehmakan, Zhongzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20604)  

**Abstract**: Random walk centrality is a fundamental metric in graph mining for quantifying node importance and influence, defined as the weighted average of hitting times to a node from all other nodes. Despite its ability to capture rich graph structural information and its wide range of applications, computing this measure for large networks remains impractical due to the computational demands of existing methods. In this paper, we present a novel formulation of random walk centrality, underpinning two scalable algorithms: one leveraging approximate Cholesky factorization and sparse inverse estimation, while the other sampling rooted spanning trees. Both algorithms operate in near-linear time and provide strong approximation guarantees. Extensive experiments on large real-world networks, including one with over 10 million nodes, demonstrate the efficiency and approximation quality of the proposed algorithms. 

---
# What Defines Good Reasoning in LLMs? Dissecting Reasoning Steps with Multi-Aspect Evaluation 

**Authors**: Heejin Do, Jaehui Hwang, Dongyoon Han, Seong Joon Oh, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2510.20603)  

**Abstract**: Evaluating large language models (LLMs) on final-answer correctness is the dominant paradigm. This approach, however, provides a coarse signal for model improvement and overlooks the quality of the underlying reasoning process. We argue that a more granular evaluation of reasoning offers a more effective path to building robust models. We decompose reasoning quality into two dimensions: relevance and coherence. Relevance measures if a step is grounded in the problem; coherence measures if it follows logically from prior steps. To measure these aspects reliably, we introduce causal stepwise evaluation (CaSE). This method assesses each reasoning step using only its preceding context, which avoids hindsight bias. We validate CaSE against human judgments on our new expert-annotated benchmarks, MRa-GSM8K and MRa-MATH. More importantly, we show that curating training data with CaSE-evaluated relevance and coherence directly improves final task performance. Our work provides a scalable framework for analyzing, debugging, and improving LLM reasoning, demonstrating the practical value of moving beyond validity checks. 

---
# Transferable Graph Learning for Transmission Congestion Management via Busbar Splitting 

**Authors**: Ali Rajaei, Peter Palensky, Jochen L. Cremer  

**Link**: [PDF](https://arxiv.org/pdf/2510.20591)  

**Abstract**: Network topology optimization (NTO) via busbar splitting can mitigate transmission grid congestion and reduce redispatch costs. However, solving this mixed-integer non-linear problem for large-scale systems in near-real-time is currently intractable with existing solvers. Machine learning (ML) approaches have emerged as a promising alternative, but they have limited generalization to unseen topologies, varying operating conditions, and different systems, which limits their practical applicability. This paper formulates NTO for congestion management problem considering linearized AC PF, and proposes a graph neural network (GNN)-accelerated approach. We develop a heterogeneous edge-aware message passing NN to predict effective busbar splitting actions as candidate NTO solutions. The proposed GNN captures local flow patterns, achieves generalization to unseen topology changes, and improves transferability across systems. Case studies show up to 4 orders-of-magnitude speed-up, delivering AC-feasible solutions within one minute and a 2.3% optimality gap on the GOC 2000-bus system. These results demonstrate a significant step toward near-real-time NTO for large-scale systems with topology and cross-system generalization. 

---
# Lost in Translation: Policymakers are not really listening to Citizen Concerns about AI 

**Authors**: Susan Ariel Aaronson, Michael Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2510.20568)  

**Abstract**: The worlds people have strong opinions about artificial intelligence (AI), and they want policymakers to listen. Governments are inviting public comment on AI, but as they translate input into policy, much of what citizens say is lost. Policymakers are missing a critical opportunity to build trust in AI and its governance. This paper compares three countries, Australia, Colombia, and the United States, that invited citizens to comment on AI risks and policies. Using a landscape analysis, the authors examined how each government solicited feedback and whether that input shaped governance. Yet in none of the three cases did citizens and policymakers establish a meaningful dialogue. Governments did little to attract diverse voices or publicize calls for comment, leaving most citizens unaware or unprepared to respond. In each nation, fewer than one percent of the population participated. Moreover, officials showed limited responsiveness to the feedback they received, failing to create an effective feedback loop. The study finds a persistent gap between the promise and practice of participatory AI governance. The authors conclude that current approaches are unlikely to build trust or legitimacy in AI because policymakers are not adequately listening or responding to public concerns. They offer eight recommendations: promote AI literacy; monitor public feedback; broaden outreach; hold regular online forums; use innovative engagement methods; include underrepresented groups; respond publicly to input; and make participation easier. 

---
# FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic 

**Authors**: Yiwen Peng, Thomas Bonald, Fabian M. Suchanek  

**Link**: [PDF](https://arxiv.org/pdf/2510.20467)  

**Abstract**: Knowledge graph alignment is the task of matching equivalent entities (that is, instances and classes) and relations across two knowledge graphs. Most existing methods focus on pure entity-level alignment, computing the similarity of entities in some embedding space. They lack interpretable reasoning and need training data to work. In this paper, we propose FLORA, a simple yet effective method that (1) is unsupervised, i.e., does not require training data, (2) provides a holistic alignment for entities and relations iteratively, (3) is based on fuzzy logic and thus delivers interpretable results, (4) provably converges, (5) allows dangling entities, i.e., entities without a counterpart in the other KG, and (6) achieves state-of-the-art results on major benchmarks. 

---
# Neural Reasoning for Robust Instance Retrieval in $\mathcal{SHOIQ}$ 

**Authors**: Louis Mozart Kamdem Teyou, Luke Friedrichs, N'Dah Jean Kouagou, Caglar Demir, Yasir Mahmood, Stefan Heindorf, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2510.20457)  

**Abstract**: Concept learning exploits background knowledge in the form of description logic axioms to learn explainable classification models from knowledge bases. Despite recent breakthroughs in neuro-symbolic concept learning, most approaches still cannot be deployed on real-world knowledge bases. This is due to their use of description logic reasoners, which are not robust against inconsistencies nor erroneous data. We address this challenge by presenting a novel neural reasoner dubbed EBR. Our reasoner relies on embeddings to approximate the results of a symbolic reasoner. We show that EBR solely requires retrieving instances for atomic concepts and existential restrictions to retrieve or approximate the set of instances of any concept in the description logic $\mathcal{SHOIQ}$. In our experiments, we compare EBR with state-of-the-art reasoners. Our results suggest that EBR is robust against missing and erroneous data in contrast to existing reasoners. 

---
# A computational model and tool for generating more novel opportunities in professional innovation processes 

**Authors**: Neil Maiden, Konstantinos Zachos, James Lockerbie, Kostas Petrianakis, Amanda Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.20402)  

**Abstract**: This paper presents a new computational model of creative outcomes, informed by creativity theories and techniques, which was implemented to generate more novel opportunities for innovation projects. The model implemented five functions that were developed to contribute to the generation of innovation opportunities with higher novelty without loss of usefulness. The model was evaluated using opportunities generated for an innovation project in the hospitality sector. The evaluation revealed that the computational model generated outcomes that were more novel and/or useful than outcomes from Notebook LM and ChatGPT4o. However, not all model functions contributed to the generation of more novel opportunities, leading to new directions for further model development 

---
# IKnow: Instruction-Knowledge-Aware Continual Pretraining for Effective Domain Adaptation 

**Authors**: Tianyi Zhang, Florian Mai, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2510.20377)  

**Abstract**: Continual pretraining promises to adapt large language models (LLMs) to new domains using only unlabeled test-time data, but naively applying standard self-supervised objectives to instruction-tuned models is known to degrade their instruction-following capability and semantic representations. Existing fixes assume access to the original base model or rely on knowledge from an external domain-specific database - both of which pose a realistic barrier in settings where the base model weights are withheld for safety reasons or reliable external corpora are unavailable. In this work, we propose Instruction-Knowledge-Aware Continual Adaptation (IKnow), a simple and general framework that formulates novel self-supervised objectives in the instruction-response dialogue format. Rather than depend- ing on external resources, IKnow leverages domain knowledge embedded within the text itself and learns to encode it at a deeper semantic level. 

---
# LLM-empowered knowledge graph construction: A survey 

**Authors**: Haonan Bian  

**Link**: [PDF](https://arxiv.org/pdf/2510.20345)  

**Abstract**: Knowledge Graphs (KGs) have long served as a fundamental infrastructure for structured knowledge representation and reasoning. With the advent of Large Language Models (LLMs), the construction of KGs has entered a new paradigm-shifting from rule-based and statistical pipelines to language-driven and generative frameworks. This survey provides a comprehensive overview of recent progress in LLM-empowered knowledge graph construction, systematically analyzing how LLMs reshape the classical three-layered pipeline of ontology engineering, knowledge extraction, and knowledge fusion.
We first revisit traditional KG methodologies to establish conceptual foundations, and then review emerging LLM-driven approaches from two complementary perspectives: schema-based paradigms, which emphasize structure, normalization, and consistency; and schema-free paradigms, which highlight flexibility, adaptability, and open discovery. Across each stage, we synthesize representative frameworks, analyze their technical mechanisms, and identify their limitations.
Finally, the survey outlines key trends and future research directions, including KG-based reasoning for LLMs, dynamic knowledge memory for agentic systems, and multimodal KG construction. Through this systematic review, we aim to clarify the evolving interplay between LLMs and knowledge graphs, bridging symbolic knowledge engineering and neural semantic understanding toward the development of adaptive, explainable, and intelligent knowledge systems. 

---
# Collateral Damage Assessment Model for AI System Target Engagement in Military Operations 

**Authors**: Clara Maathuis, Kasper Cools  

**Link**: [PDF](https://arxiv.org/pdf/2510.20337)  

**Abstract**: In an era where AI (Artificial Intelligence) systems play an increasing role in the battlefield, ensuring responsible targeting demands rigorous assessment of potential collateral effects. In this context, a novel collateral damage assessment model for target engagement of AI systems in military operations is introduced. The model integrates temporal, spatial, and force dimensions within a unified Knowledge Representation and Reasoning (KRR) architecture following a design science methodological approach. Its layered structure captures the categories and architectural components of the AI systems to be engaged together with corresponding engaging vectors and contextual aspects. At the same time, spreading, severity, likelihood, and evaluation metrics are considered in order to provide a clear representation enhanced by transparent reasoning mechanisms. Further, the model is demonstrated and evaluated through instantiation which serves as a basis for further dedicated efforts that aim at building responsible and trustworthy intelligent systems for assessing the effects produced by engaging AI systems in military operations. 

---
# Bias by Design? How Data Practices Shape Fairness in AI Healthcare Systems 

**Authors**: Anna Arias-Duart, Maria Eugenia Cardello, Atia Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2510.20332)  

**Abstract**: Artificial intelligence (AI) holds great promise for transforming healthcare. However, despite significant advances, the integration of AI solutions into real-world clinical practice remains limited. A major barrier is the quality and fairness of training data, which is often compromised by biased data collection practices. This paper draws on insights from the AI4HealthyAging project, part of Spain's national R&D initiative, where our task was to detect biases during clinical data collection. We identify several types of bias across multiple use cases, including historical, representation, and measurement biases. These biases manifest in variables such as sex, gender, age, habitat, socioeconomic status, equipment, and labeling. We conclude with practical recommendations for improving the fairness and robustness of clinical problem design and data collection. We hope that our findings and experience contribute to guiding future projects in the development of fairer AI systems in healthcare. 

---
# Multi-Step Reasoning for Embodied Question Answering via Tool Augmentation 

**Authors**: Mingliang Zhai, Hansheng Liang, Xiaomeng Fan, Zhi Gao, Chuanhao Li, Che Sun, Xu Bin, Yuwei Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.20310)  

**Abstract**: Embodied Question Answering (EQA) requires agents to explore 3D environments to obtain observations and answer questions related to the scene. Existing methods leverage VLMs to directly explore the environment and answer questions without explicit thinking or planning, which limits their reasoning ability and results in excessive or inefficient exploration as well as ineffective responses. In this paper, we introduce ToolEQA, an agent that integrates external tools with multi-step reasoning, where external tools can provide more useful information for completing the task, helping the model derive better exploration directions in the next step of reasoning and thus obtaining additional effective information. This enables ToolEQA to generate more accurate responses with a shorter exploration distance. To enhance the model's ability for tool-usage and multi-step reasoning, we further design a novel EQA data generation pipeline that automatically constructs large-scale EQA tasks with reasoning trajectories and corresponding answers. Based on the pipeline, we collect the EQA-RT dataset that contains about 18K tasks, divided into a training set EQA-RT-Train, and two test sets EQA-RT-Seen (scenes overlapping with the training set) and EQA-RT-Unseen (novel scenes). Experiments on EQA-RT-Seen and EQA-RT-Unseen show that ToolEQA improves the success rate by 9.2~20.2% over state-of-the-art baselines, while outperforming the zero-shot ToolEQA by 10% in success rate. In addition, ToolEQA also achieves state-of-the-art performance on the HM-EQA, OpenEQA, and EXPRESS-Bench datasets, demonstrating its generality. Our homepage see this https URL. 

---
# Classical Feature Embeddings Help in BERT-Based Human Mobility Prediction 

**Authors**: Yunzhi Liu, Haokai Tan, Rushi Kanjaria, Lihuan Li, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20275)  

**Abstract**: Human mobility forecasting is crucial for disaster relief, city planning, and public health. However, existing models either only model location sequences or include time information merely as auxiliary input, thereby failing to leverage the rich semantic context provided by points of interest (POIs). To address this, we enrich a BERT-based mobility model with derived temporal descriptors and POI embeddings to better capture the semantics underlying human movement. We propose STaBERT (Semantic-Temporal aware BERT), which integrates both POI and temporal information at each location to construct a unified, semantically enriched representation of mobility. Experimental results show that STaBERT significantly improves prediction accuracy: for single-city prediction, the GEO-BLEU score improved from 0.34 to 0.75; for multi-city prediction, from 0.34 to 0.56. 

---
# Using Large Language Models for Abstraction of Planning Domains - Extended Version 

**Authors**: Bita Banihashemi, Megh Patel, Yves Lespérance  

**Link**: [PDF](https://arxiv.org/pdf/2510.20258)  

**Abstract**: Generating an abstraction of a dynamic domain that aligns with a given purpose remains a significant challenge given that the choice of such an abstraction can impact an agent's ability to plan, reason, and provide explanations effectively. We model the agent's concrete behaviors in PDDL and investigate the use of in-context learning with large language models (LLMs) for the generation of abstract PDDL domains and problem instances, given an abstraction objective specified in natural language. The benchmark examples we use are new and have not been part of the data any LLMs have been trained on. We consider three categories of abstractions: abstraction of choice of alternative concrete actions, abstraction of sequences of concrete actions, and abstraction of action/predicate parameters, as well as combinations of these. The generated abstract PDDL domains and problem instances are then checked by symbolic validation tools as well as human experts. Our experiments show that GPT-4o can generally synthesize useful planning domain abstractions in simple settings, although it is better at abstracting over actions than over the associated fluents. 

---
# Individualized Cognitive Simulation in Large Language Models: Evaluating Different Cognitive Representation Methods 

**Authors**: Tianyi Zhang, Xiaolin Zhou, Yunzhe Wang, Erik Cambria, David Traum, Rui Mao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20252)  

**Abstract**: Individualized cognitive simulation (ICS) aims to build computational models that approximate the thought processes of specific individuals. While large language models (LLMs) convincingly mimic surface-level human behavior such as role-play, their ability to simulate deeper individualized cognitive processes remains poorly understood. To address this gap, we introduce a novel task that evaluates different cognitive representation methods in ICS. We construct a dataset from recently published novels (later than the release date of the tested LLMs) and propose an 11-condition cognitive evaluation framework to benchmark seven off-the-shelf LLMs in the context of authorial style emulation. We hypothesize that effective cognitive representations can help LLMs generate storytelling that better mirrors the original author. Thus, we test different cognitive representations, e.g., linguistic features, concept mappings, and profile-based information. Results show that combining conceptual and linguistic features is particularly effective in ICS, outperforming static profile-based cues in overall evaluation. Importantly, LLMs are more effective at mimicking linguistic style than narrative structure, underscoring their limits in deeper cognitive simulation. These findings provide a foundation for developing AI systems that adapt to individual ways of thinking and expression, advancing more personalized and human-aligned creative technologies. 

---
# Merge and Conquer: Evolutionarily Optimizing AI for 2048 

**Authors**: Maggie Bai, Ava Kim Cohen, Eleanor Koss, Charlie Lichtenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2510.20205)  

**Abstract**: Optimizing artificial intelligence (AI) for dynamic environments remains a fundamental challenge in machine learning research. In this paper, we examine evolutionary training methods for optimizing AI to solve the game 2048, a 2D sliding puzzle. 2048, with its mix of strategic gameplay and stochastic elements, presents an ideal playground for studying decision-making, long-term planning, and dynamic adaptation. We implemented two distinct systems: a two-agent metaprompting system where a "thinker" large language model (LLM) agent refines gameplay strategies for an "executor" LLM agent, and a single-agent system based on refining a value function for a limited Monte Carlo Tree Search. We also experimented with rollback features to avoid performance degradation. Our results demonstrate the potential of evolutionary refinement techniques in improving AI performance in non-deterministic environments. The single-agent system achieved substantial improvements, with an average increase of 473.2 points per cycle, and with clear upward trends (correlation $\rho$=0.607) across training cycles. The LLM's understanding of the game grew as well, shown in its development of increasingly advanced strategies. Conversely, the two-agent system did not garner much improvement, highlighting the inherent limits of meta-prompting. 

---
# The Lock-In Phase Hypothesis: Identity Consolidation as a Precursor to AGI 

**Authors**: Marcelo Maciel Amaral, Raymond Aschheim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20190)  

**Abstract**: Large language models (LLMs) remain broadly open and highly steerable: they imitate at scale, accept arbitrary system prompts, and readily adopt multiple personae. By analogy to human development, we hypothesize that progress toward artificial general intelligence (AGI) involves a lock-in phase: a transition from open imitation to identity consolidation, in which goal structures, refusals, preferences, and internal representations become comparatively stable and resistant to external steering. We formalize this phase, link it to known phenomena in learning dynamics, and propose operational metrics for onset detection. Experimentally, we demonstrate that while the behavioral consolidation is rapid and non-linear, its side-effects on general capabilities are not monolithic. Our results reveal a spectrum of outcomes--from performance trade-offs in small models, through largely cost-free adoption in mid-scale models, to transient instabilities in large, quantized models. We argue that such consolidation is a prerequisite for AGI-level reliability and also a critical control point for safety: identities can be deliberately engineered for reliability, yet may also emerge spontaneously during scaling, potentially hardening unpredictable goals and behaviors. 

---
# TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning 

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20188)  

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment. 

---
# The Verification-Value Paradox: A Normative Critique of Gen AI in Legal Practice 

**Authors**: Joshua Yuvaraj  

**Link**: [PDF](https://arxiv.org/pdf/2510.20109)  

**Abstract**: It is often claimed that machine learning-based generative AI products will drastically streamline and reduce the cost of legal practice. This enthusiasm assumes lawyers can effectively manage AI's risks. Cases in Australia and elsewhere in which lawyers have been reprimanded for submitting inaccurate AI-generated content to courts suggest this paradigm must be revisited. This paper argues that a new paradigm is needed to evaluate AI use in practice, given (a) AI's disconnection from reality and its lack of transparency, and (b) lawyers' paramount duties like honesty, integrity, and not to mislead the court. It presents an alternative model of AI use in practice that more holistically reflects these features (the verification-value paradox). That paradox suggests increases in efficiency from AI use in legal practice will be met by a correspondingly greater imperative to manually verify any outputs of that use, rendering the net value of AI use often negligible to lawyers. The paper then sets out the paradox's implications for legal practice and legal education, including for AI use but also the values that the paradox suggests should undergird legal practice: fidelity to the truth and civic responsibility. 

---
# Human-Centered LLM-Agent System for Detecting Anomalous Digital Asset Transactions 

**Authors**: Gyuyeon Na, Minjung Park, Hyeonjeong Cha, Sangmi Chai  

**Link**: [PDF](https://arxiv.org/pdf/2510.20102)  

**Abstract**: We present HCLA, a human-centered multi-agent system for anomaly detection in digital asset transactions. The system links three roles: Parsing, Detection, and Explanation, into a conversational workflow that lets non-experts ask questions in natural language, inspect structured analytics, and obtain context-aware rationales. Implemented with an open-source web UI, HCLA translates user intents into a schema for a classical detector (XGBoost in our prototype) and returns narrative explanations grounded in the underlying features. On a labeled Bitcoin mixing dataset (Wasabi Wallet, 2020-2024), the baseline detector reaches strong accuracy, while HCLA adds interpretability and interactive refinement. We describe the architecture, interaction loop, dataset, evaluation protocol, and limitations, and discuss how a human-in-the-loop design improves transparency and trust in financial forensics. 

---
# AI PB: A Grounded Generative Agent for Personalized Investment Insights 

**Authors**: Daewoo Park, Suho Park, Inseok Hong, Hanwool Lee, Junkyu Park, Sangjun Lee, Jeongman An, Hyunbin Loh  

**Link**: [PDF](https://arxiv.org/pdf/2510.20099)  

**Abstract**: We present AI PB, a production-scale generative agent deployed in real retail finance. Unlike reactive chatbots that answer queries passively, AI PB proactively generates grounded, compliant, and user-specific investment insights. It integrates (i) a component-based orchestration layer that deterministically routes between internal and external LLMs based on data sensitivity, (ii) a hybrid retrieval pipeline using OpenSearch and the finance-domain embedding model, and (iii) a multi-stage recommendation mechanism combining rule heuristics, sequential behavioral modeling, and contextual bandits. Operating fully on-premises under Korean financial regulations, the system employs Docker Swarm and vLLM across 24 X NVIDIA H100 GPUs. Through human QA and system metrics, we demonstrate that grounded generation with explicit routing and layered safety can deliver trustworthy AI insights in high-stakes finance. 

---
# LLMs can hide text in other text of the same length.ipynb 

**Authors**: Antonio Norelli, Michael Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.20075)  

**Abstract**: A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something. 

---
# AI-Driven Personalized Learning: Predicting Academic Per-formance Through Leadership Personality Traits 

**Authors**: Nitsa J Herzog, Rejwan Bin Sulaiman, David J Herzog, Rose Fong  

**Link**: [PDF](https://arxiv.org/pdf/2510.19964)  

**Abstract**: The study explores the potential of AI technologies in personalized learning, suggesting the prediction of academic success through leadership personality traits and machine learning modelling. The primary data were obtained from 129 master's students in the Environmental Engineering Department, who underwent five leadership personality tests with 23 characteristics. Students used self-assessment tools that included Personality Insight, Workplace Culture, Motivation at Work, Management Skills, and Emotion Control tests. The test results were combined with the average grade obtained from academic reports. The study employed exploratory data analysis and correlation analysis. Feature selection utilized Pearson correlation coefficients of personality traits. The average grades were separated into three categories: fail, pass, and excellent. The modelling process was performed by tuning seven ML algorithms, such as SVM, LR, KNN, DT, GB, RF, XGBoost and LightGBM. The highest predictive performance was achieved with the RF classifier, which yielded an accuracy of 87.50% for the model incorporating 17 personality trait features and the leadership mark feature, and an accuracy of 85.71% for the model excluding this feature. In this way, the study offers an additional opportunity to identify students' strengths and weaknesses at an early stage of their education process and select the most suitable strategies for personalized learning. 

---
# A new wave of vehicle insurance fraud fueled by generative AI 

**Authors**: Amir Hever, Itai Orr  

**Link**: [PDF](https://arxiv.org/pdf/2510.19957)  

**Abstract**: Generative AI is supercharging insurance fraud by making it easier to falsify accident evidence at scale and in rapid time. Insurance fraud is a pervasive and costly problem, amounting to tens of billions of dollars in losses each year. In the vehicle insurance sector, fraud schemes have traditionally involved staged accidents, exaggerated damage, or forged documents. The rise of generative AI, including deepfake image and video generation, has introduced new methods for committing fraud at scale. Fraudsters can now fabricate highly realistic crash photos, damage evidence, and even fake identities or documents with minimal effort, exploiting AI tools to bolster false insurance claims. Insurers have begun deploying countermeasures such as AI-based deepfake detection software and enhanced verification processes to detect and mitigate these AI-driven scams. However, current mitigation strategies face significant limitations. Detection tools can suffer from false positives and negatives, and sophisticated fraudsters continuously adapt their tactics to evade automated checks. This cat-and-mouse arms race between generative AI and detection technology, combined with resource and cost barriers for insurers, means that combating AI-enabled insurance fraud remains an ongoing challenge. In this white paper, we present UVeye layered solution for vehicle fraud, representing a major leap forward in the ability to detect, mitigate and deter this new wave of fraud. 

---
# RELATE: A Schema-Agnostic Perceiver Encoder for Multimodal Relational Graphs 

**Authors**: Joseph Meyer, Divyansha Lachi, Reza Mohammadi, Roshan Reddy Upendra, Eva L. Dyer, Mark Li, Tom Palczewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.19954)  

**Abstract**: Relational multi-table data is common in domains such as e-commerce, healthcare, and scientific research, and can be naturally represented as heterogeneous temporal graphs with multi-modal node attributes. Existing graph neural networks (GNNs) rely on schema-specific feature encoders, requiring separate modules for each node type and feature column, which hinders scalability and parameter sharing. We introduce RELATE (Relational Encoder for Latent Aggregation of Typed Entities), a schema-agnostic, plug-and-play feature encoder that can be used with any general purpose GNN. RELATE employs shared modality-specific encoders for categorical, numerical, textual, and temporal attributes, followed by a Perceiver-style cross-attention module that aggregates features into a fixed-size, permutation-invariant node representation. We evaluate RELATE on ReLGNN and HGT in the RelBench benchmark, where it achieves performance within 3% of schema-specific encoders while reducing parameter counts by up to 5x. This design supports varying schemas and enables multi-dataset pretraining for general-purpose GNNs, paving the way toward foundation models for relational graph data. 

---
# Surfer 2: The Next Generation of Cross-Platform Computer Use Agents 

**Authors**: Mathieu Andreux, Märt Bakler, Yanael Barbier, Hamza Ben Chekroun, Emilien Biré, Antoine Bonnet, Riaz Bordie, Nathan Bout, Matthias Brunel, Aleix Cambray, Pierre-Louis Cedoz, Antoine Chassang, Gautier Cloix, Ethan Connelly, Alexandra Constantinou, Ramzi De Coster, Hubert de la Jonquiere, Aurélien Delfosse, Maxime Delpit, Alexis Deprez, Augustin Derupti, Mathieu Diaz, Shannon D'Souza, Julie Dujardin, Abai Edmund, Michael Eickenberg, Armand Fatalot, Wissem Felissi, Isaac Herring, Xavier Koegler, Erwan Le Jumeau de Kergaradec, Aurélien Lac, Maxime Langevin, Corentin Lauverjat, Antonio Loison, Avshalom Manevich, Axel Moyal, Axel Nguyen Kerbel, Marinela Parovic, Julien Revelle, Guillaume Richard, Mats Richter, Ronan Riochet, María Santos, Romain Savidan, Laurent Sifre, Maxime Theillard, Marc Thibault, Ivan Valentini, Tony Wu, Laura Yie, Kai Yuan, Jevgenij Zubovskij  

**Link**: [PDF](https://arxiv.org/pdf/2510.19949)  

**Abstract**: Building agents that generalize across web, desktop, and mobile environments remains an open challenge, as prior systems rely on environment-specific interfaces that limit cross-platform deployment. We introduce Surfer 2, a unified architecture operating purely from visual observations that achieves state-of-the-art performance across all three environments. Surfer 2 integrates hierarchical context management, decoupled planning and execution, and self-verification with adaptive recovery, enabling reliable operation over long task horizons. Our system achieves 97.1% accuracy on WebVoyager, 69.6% on WebArena, 60.1% on OSWorld, and 87.1% on AndroidWorld, outperforming all prior systems without task-specific fine-tuning. With multiple attempts, Surfer 2 exceeds human performance on all benchmarks. These results demonstrate that systematic orchestration amplifies foundation model capabilities and enables general-purpose computer control through visual interaction alone, while calling for a next-generation vision language model to achieve Pareto-optimal cost-efficiency. 

---
# DAG-Math: Graph-Guided Mathematical Reasoning in LLMs 

**Authors**: Yuanhe Zhang, Ilja Kuzborskij, Jason D. Lee, Chenlei Leng, Fanghui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19842)  

**Abstract**: Large Language Models (LLMs) demonstrate strong performance on mathematical problems when prompted with Chain-of-Thought (CoT), yet it remains unclear whether this success stems from search, rote procedures, or rule-consistent reasoning. To address this, we propose modeling CoT as a certain rule-based stochastic process over directed acyclic graphs (DAGs), where nodes represent intermediate derivation states and edges encode rule applications. Within this framework, we introduce logical closeness, a metric that quantifies how well a model's CoT trajectory (i.e., the LLM's final output) adheres to the DAG structure, providing evaluation beyond classical PASS@k metrics. Building on this, we introduce the DAG-MATH CoT format and construct a benchmark that guides LLMs to generate CoT trajectories in this format, thereby enabling the evaluation of their reasoning ability under our framework. Across standard mathematical reasoning datasets, our analysis uncovers statistically significant differences in reasoning fidelity among representative LLM families-even when PASS@k is comparable-highlighting gaps between final-answer accuracy and rule-consistent derivation. Our framework provides a balance between free-form CoT and formal proofs systems, offering actionable diagnostics for LLMs reasoning evaluation. Our benchmark and code are available at: this https URL. 

---
# Branch-and-Browse: Efficient and Controllable Web Exploration with Tree-Structured Reasoning and Action Memory 

**Authors**: Shiqi He, Yue Cui, Xinyu Ma, Yaliang Li, Bolin Ding, Mosharaf Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.19838)  

**Abstract**: Autonomous web agents powered by large language models (LLMs) show strong potential for performing goal-oriented tasks such as information retrieval, report generation, and online transactions. These agents mark a key step toward practical embodied reasoning in open web environments. However, existing approaches remain limited in reasoning depth and efficiency: vanilla linear methods fail at multi-step reasoning and lack effective backtracking, while other search strategies are coarse-grained and computationally costly. We introduce Branch-and-Browse, a fine-grained web agent framework that unifies structured reasoning-acting, contextual memory, and efficient execution. It (i) employs explicit subtask management with tree-structured exploration for controllable multi-branch reasoning, (ii) bootstraps exploration through efficient web state replay with background reasoning, and (iii) leverages a page action memory to share explored actions within and across sessions. On the WebArena benchmark, Branch-and-Browse achieves a task success rate of 35.8\% and reduces execution time by up to 40.4\% relative to state-of-the-art methods. These results demonstrate that Branch-and-Browse is a reliable and efficient framework for LLM-based web agents. 

---
# Benchmarking Reasoning Reliability in Artificial Intelligence Models for Energy-System Analysis 

**Authors**: Eliseo Curcio  

**Link**: [PDF](https://arxiv.org/pdf/2510.19836)  

**Abstract**: Artificial intelligence and machine learning are increasingly used for forecasting, optimization, and policy design in the energy sector, yet no standardized framework exists to evaluate whether these systems reason correctly. Current validation practices focus on predictive accuracy or computational efficiency, leaving the logical integrity of analytical conclusions untested. This study introduces the Analytical Reliability Benchmark (ARB), a reproducible framework that quantifies reasoning reliability in large language models applied to energy system analysis. The benchmark integrates five submetrics: accuracy, reasoning reliability, uncertainty discipline, policy consistency, and transparency, and evaluates model performance across deterministic, probabilistic, and epistemic scenarios using open technoeconomic datasets (NREL ATB 2024, DOE H2A/H2New, IEA WEO 2024). Four frontier models (GPT-4/5, Claude 4.5 Sonnet, Gemini 2.5 Pro, Llama 3 70B) were tested under identical factual and regulatory conditions. Results show that reasoning reliability can be objectively measured. GPT-4/5 and Claude 4.5 Sonnet achieved consistent and policy-compliant reasoning (Analytical Reliability Index greater than 90), Gemini 2.5 Pro demonstrated moderate stability, and Llama 3 70B remained below professional thresholds. Statistical validation confirmed that these differences are significant and reproducible. The ARB establishes the first quantitative method in the energy literature for verifying causal, probabilistic, and policy-driven reasoning in artificial intelligence systems, providing a reference framework for trustworthy and transparent analytical applications in the global energy transition. 

---
# A Quantum-Inspired Algorithm for Solving Sudoku Puzzles and the MaxCut Problem 

**Authors**: Max B. Zhao, Fei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.19835)  

**Abstract**: We propose and evaluate a quantum-inspired algorithm for solving Quadratic Unconstrained Binary Optimization (QUBO) problems, which are mathematically equivalent to finding ground states of Ising spin-glass Hamiltonians. The algorithm employs Matrix Product States (MPS) to compactly represent large superpositions of spin configurations and utilizes a discrete driving schedule to guide the MPS toward the ground state. At each step, a driver Hamiltonian -- incorporating a transverse magnetic field -- is combined with the problem Hamiltonian to enable spin flips and facilitate quantum tunneling. The MPS is updated using the standard Density Matrix Renormalization Group (DMRG) method, which iteratively minimizes the system's energy via multiple sweeps across the spin chain. Despite its heuristic nature, the algorithm reliably identifies global minima, not merely near-optimal solutions, across diverse QUBO instances. We first demonstrate its effectiveness on intermediate-level Sudoku puzzles from publicly available sources, involving over $200$ Ising spins with long-range couplings dictated by constraint satisfaction. We then apply the algorithm to MaxCut problems from the Biq Mac library, successfully solving instances with up to $251$ nodes and $3,265$ edges. We discuss the advantages of this quantum-inspired approach, including its scalability, generalizability, and suitability for industrial-scale QUBO applications. 

---
# Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge 

**Authors**: Nimrod Berman, Omkar Joglekar, Eitan Kosman, Dotan Di Castro, Omri Azencot  

**Link**: [PDF](https://arxiv.org/pdf/2510.20819)  

**Abstract**: Recent advances in generative modeling have positioned diffusion models as state-of-the-art tools for sampling from complex data distributions. While these models have shown remarkable success across single-modality domains such as images and audio, extending their capabilities to Modality Translation (MT), translating information across different sensory modalities, remains an open challenge. Existing approaches often rely on restrictive assumptions, including shared dimensionality, Gaussian source priors, and modality-specific architectures, which limit their generality and theoretical grounding. In this work, we propose the Latent Denoising Diffusion Bridge Model (LDDBM), a general-purpose framework for modality translation based on a latent-variable extension of Denoising Diffusion Bridge Models. By operating in a shared latent space, our method learns a bridge between arbitrary modalities without requiring aligned dimensions. We introduce a contrastive alignment loss to enforce semantic consistency between paired samples and design a domain-agnostic encoder-decoder architecture tailored for noise prediction in latent space. Additionally, we propose a predictive loss to guide training toward accurate cross-domain translation and explore several training strategies to improve stability. Our approach supports arbitrary modality pairs and performs strongly on diverse MT tasks, including multi-view to 3D shape generation, image super-resolution, and multi-view scene synthesis. Comprehensive experiments and ablations validate the effectiveness of our framework, establishing a new strong baseline in general modality translation. For more information, see our project page: this https URL. 

---
# VAMOS: A Hierarchical Vision-Language-Action Model for Capability-Modulated and Steerable Navigation 

**Authors**: Mateo Guaman Castro, Sidharth Rajagopal, Daniel Gorbatov, Matt Schmittle, Rohan Baijal, Octi Zhang, Rosario Scalise, Sidharth Talia, Emma Romig, Celso de Melo, Byron Boots, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.20818)  

**Abstract**: A fundamental challenge in robot navigation lies in learning policies that generalize across diverse environments while conforming to the unique physical constraints and capabilities of a specific embodiment (e.g., quadrupeds can walk up stairs, but rovers cannot). We propose VAMOS, a hierarchical VLA that decouples semantic planning from embodiment grounding: a generalist planner learns from diverse, open-world data, while a specialist affordance model learns the robot's physical constraints and capabilities in safe, low-cost simulation. We enabled this separation by carefully designing an interface that lets a high-level planner propose candidate paths directly in image space that the affordance model then evaluates and re-ranks. Our real-world experiments show that VAMOS achieves higher success rates in both indoor and complex outdoor navigation than state-of-the-art model-based and end-to-end learning methods. We also show that our hierarchical design enables cross-embodied navigation across legged and wheeled robots and is easily steerable using natural language. Real-world ablations confirm that the specialist model is key to embodiment grounding, enabling a single high-level planner to be deployed across physically distinct wheeled and legged robots. Finally, this model significantly enhances single-robot reliability, achieving 3X higher success rates by rejecting physically infeasible plans. Website: this https URL 

---
# GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation 

**Authors**: Guangqi Jiang, Haoran Chang, Ri-Zhao Qiu, Yutong Liang, Mazeyu Ji, Jiyue Zhu, Zhao Dong, Xueyan Zou, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20813)  

**Abstract**: This paper presents GSWorld, a robust, photo-realistic simulator for robotics manipulation that combines 3D Gaussian Splatting with physics engines. Our framework advocates "closing the loop" of developing manipulation policies with reproducible evaluation of policies learned from real-robot data and sim2real policy training without using real robots. To enable photo-realistic rendering of diverse scenes, we propose a new asset format, which we term GSDF (Gaussian Scene Description File), that infuses Gaussian-on-Mesh representation with robot URDF and other objects. With a streamlined reconstruction pipeline, we curate a database of GSDF that contains 3 robot embodiments for single-arm and bimanual manipulation, as well as more than 40 objects. Combining GSDF with physics engines, we demonstrate several immediate interesting applications: (1) learning zero-shot sim2real pixel-to-action manipulation policy with photo-realistic rendering, (2) automated high-quality DAgger data collection for adapting policies to deployment environments, (3) reproducible benchmarking of real-robot manipulation policies in simulation, (4) simulation data collection by virtual teleoperation, and (5) zero-shot sim2real visual reinforcement learning. Website: this https URL. 

---
# Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation 

**Authors**: Yuhan Liu, Lianhui Qin, Shengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20812)  

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence. We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers. To further improve efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict. Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from multiple partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines. Code is available at this https URL 

---
# On the Detectability of LLM-Generated Text: What Exactly Is LLM-Generated Text? 

**Authors**: Mingmeng Geng, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2510.20810)  

**Abstract**: With the widespread use of large language models (LLMs), many researchers have turned their attention to detecting text generated by them. However, there is no consistent or precise definition of their target, namely "LLM-generated text". Differences in usage scenarios and the diversity of LLMs further increase the difficulty of detection. What is commonly regarded as the detecting target usually represents only a subset of the text that LLMs can potentially produce. Human edits to LLM outputs, together with the subtle influences that LLMs exert on their users, are blurring the line between LLM-generated and human-written text. Existing benchmarks and evaluation approaches do not adequately address the various conditions in real-world detector applications. Hence, the numerical results of detectors are often misunderstood, and their significance is diminishing. Therefore, detectors remain useful under specific conditions, but their results should be interpreted only as references rather than decisive indicators. 

---
# The Reality Gap in Robotics: Challenges, Solutions, and Best Practices 

**Authors**: Elie Aljalbout, Jiaxu Xing, Angel Romero, Iretiayo Akinola, Caelan Reed Garrett, Eric Heiden, Abhishek Gupta, Tucker Hermans, Yashraj Narang, Dieter Fox, Davide Scaramuzza, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2510.20808)  

**Abstract**: Machine learning has facilitated significant advancements across various robotics domains, including navigation, locomotion, and manipulation. Many such achievements have been driven by the extensive use of simulation as a critical tool for training and testing robotic systems prior to their deployment in real-world environments. However, simulations consist of abstractions and approximations that inevitably introduce discrepancies between simulated and real environments, known as the reality gap. These discrepancies significantly hinder the successful transfer of systems from simulation to the real world. Closing this gap remains one of the most pressing challenges in robotics. Recent advances in sim-to-real transfer have demonstrated promising results across various platforms, including locomotion, navigation, and manipulation. By leveraging techniques such as domain randomization, real-to-sim transfer, state and action abstractions, and sim-real co-training, many works have overcome the reality gap. However, challenges persist, and a deeper understanding of the reality gap's root causes and solutions is necessary. In this survey, we present a comprehensive overview of the sim-to-real landscape, highlighting the causes, solutions, and evaluation metrics for the reality gap and sim-to-real transfer. 

---
# Compress to Impress: Efficient LLM Adaptation Using a Single Gradient Step on 100 Samples 

**Authors**: Shiva Sreeram, Alaa Maalouf, Pratyusha Sharma, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2510.20800)  

**Abstract**: Recently, Sharma et al. suggested a method called Layer-SElective-Rank reduction (LASER) which demonstrated that pruning high-order components of carefully chosen LLM's weight matrices can boost downstream accuracy -- without any gradient-based fine-tuning. Yet LASER's exhaustive, per-matrix search (each requiring full-dataset forward passes) makes it impractical for rapid deployment. We demonstrate that this overhead can be removed and find that: (i) Only a small, carefully chosen subset of matrices needs to be inspected -- eliminating the layer-by-layer sweep, (ii) The gradient of each matrix's singular values pinpoints which matrices merit reduction, (iii) Increasing the factorization search space by allowing matrices rows to cluster around multiple subspaces and then decomposing each cluster separately further reduces overfitting on the original training data and further lifts accuracy by up to 24.6 percentage points, and finally, (iv) we discover that evaluating on just 100 samples rather than the full training data -- both for computing the indicative gradients and for measuring the final accuracy -- suffices to further reduce the search time; we explain that as adaptation to downstream tasks is dominated by prompting style, not dataset size. As a result, we show that combining these findings yields a fast and robust adaptation algorithm for downstream tasks. Overall, with a single gradient step on 100 examples and a quick scan of the top candidate layers and factorization techniques, we can adapt LLMs to new datasets -- entirely without fine-tuning. 

---
# Simple Context Compression: Mean-Pooling and Multi-Ratio Training 

**Authors**: Yair Feldman, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20797)  

**Abstract**: A common strategy to reduce the computational costs of using long contexts in retrieval-augmented generation (RAG) with large language models (LLMs) is soft context compression, where the input sequence is transformed into a shorter continuous representation. We develop a lightweight and simple mean-pooling approach that consistently outperforms the widely used compression-tokens architecture, and study training the same compressor to output multiple compression ratios. We conduct extensive experiments across in-domain and out-of-domain QA datasets, as well as across model families, scales, and compression ratios. Overall, our simple mean-pooling approach achieves the strongest performance, with a relatively small drop when training for multiple compression ratios. More broadly though, across architectures and training regimes the trade-offs are more nuanced, illustrating the complex landscape of compression methods. 

---
# Bayesian Inference of Primordial Magnetic Field Parameters from CMB with Spherical Graph Neural Networks 

**Authors**: Juan Alejandro Pinto Castro, Héctor J. Hortúa, Jorge Enrique García-Farieta, Roger Anderson Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2510.20795)  

**Abstract**: Deep learning has emerged as a transformative methodology in modern cosmology, providing powerful tools to extract meaningful physical information from complex astronomical datasets. This paper implements a novel Bayesian graph deep learning framework for estimating key cosmological parameters in a primordial magnetic field (PMF) cosmology directly from simulated Cosmic Microwave Background (CMB) maps. Our methodology utilizes DeepSphere, a spherical convolutional neural network architecture specifically designed to respect the spherical geometry of CMB data through HEALPix pixelization. To advance beyond deterministic point estimates and enable robust uncertainty quantification, we integrate Bayesian Neural Networks (BNNs) into the framework, capturing aleatoric and epistemic uncertainties that reflect the model confidence in its predictions. The proposed approach demonstrates exceptional performance, achieving $R^{2}$ scores exceeding 0.89 for the magnetic parameter estimation. We further obtain well-calibrated uncertainty estimates through post-hoc training techniques including Variance Scaling and GPNormal. This integrated DeepSphere-BNNs framework not only delivers accurate parameter estimation from CMB maps with PMF contributions but also provides reliable uncertainty quantification, providing the necessary tools for robust cosmological inference in the era of precision cosmology. 

---
# A Use-Case Specific Dataset for Measuring Dimensions of Responsible Performance in LLM-generated Text 

**Authors**: Alicia Sagae, Chia-Jung Lee, Sandeep Avula, Brandon Dang, Vanessa Murdock  

**Link**: [PDF](https://arxiv.org/pdf/2510.20782)  

**Abstract**: Current methods for evaluating large language models (LLMs) typically focus on high-level tasks such as text generation, without targeting a particular AI application. This approach is not sufficient for evaluating LLMs for Responsible AI dimensions like fairness, since protected attributes that are highly relevant in one application may be less relevant in another. In this work, we construct a dataset that is driven by a real-world application (generate a plain-text product description, given a list of product features), parameterized by fairness attributes intersected with gendered adjectives and product categories, yielding a rich set of labeled prompts. We show how to use the data to identify quality, veracity, safety, and fairness gaps in LLMs, contributing a proposal for LLM evaluation paired with a concrete resource for the research community. 

---
# Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost 

**Authors**: Runzhe Zhan, Zhihong Huang, Xinyi Yang, Lidia S. Chao, Min Yang, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.20780)  

**Abstract**: Recent advancements in large reasoning models (LRMs) have introduced an intermediate "thinking" process prior to generating final answers, improving their reasoning capabilities on complex downstream tasks. However, the potential of LRMs as evaluators for machine translation (MT) quality remains underexplored. We provides the first systematic analysis of LRM-as-a-judge in MT evaluation. We identify key challenges, revealing LRMs require tailored evaluation materials, tend to "overthink" simpler instances and have issues with scoring mechanisms leading to overestimation. To address these, we propose to calibrate LRM thinking by training them on synthetic, human-like thinking trajectories. Our experiments on WMT24 Metrics benchmarks demonstrate that this approach largely reduces thinking budgets by ~35x while concurrently improving evaluation performance across different LRM scales from 7B to 32B (e.g., R1-Distill-Qwen-7B achieves a +8.7 correlation point improvement). These findings highlight the potential of efficiently calibrated LRMs to advance fine-grained automatic MT evaluation. 

---
# FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation 

**Authors**: Wenhao Wang, Kehe Ye, Xinyu Zhou, Tianxing Chen, Cao Min, Qiaoming Zhu, Xiaokang Yang, Yongjian Shen, Yang Yang, Maoqing Yao, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20774)  

**Abstract**: Large-scale and diverse datasets are vital for training robust robotic manipulation policies, yet existing data collection methods struggle to balance scale, diversity, and quality. Simulation offers scalability but suffers from sim-to-real gaps, while teleoperation yields high-quality demonstrations with limited diversity and high labor cost. We introduce FieldGen, a field-guided data generation framework that enables scalable, diverse, and high-quality real-world data collection with minimal human supervision. FieldGen decomposes manipulation into two stages: a pre-manipulation phase, allowing trajectory diversity, and a fine manipulation phase requiring expert precision. Human demonstrations capture key contact and pose information, after which an attraction field automatically generates diverse trajectories converging to successful configurations. This decoupled design combines scalable trajectory diversity with precise supervision. Moreover, FieldGen-Reward augments generated data with reward annotations to further enhance policy learning. Experiments demonstrate that policies trained with FieldGen achieve higher success rates and improved stability compared to teleoperation-based baselines, while significantly reducing human effort in long-term real-world data collection. Webpage is available at this https URL. 

---
# RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines 

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20768)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds. 

---
# Reinforcement Learning and Consumption-Savings Behavior 

**Authors**: Brandon Kaplowitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.20748)  

**Abstract**: This paper demonstrates how reinforcement learning can explain two puzzling empirical patterns in household consumption behavior during economic downturns. I develop a model where agents use Q-learning with neural network approximation to make consumption-savings decisions under income uncertainty, departing from standard rational expectations assumptions. The model replicates two key findings from recent literature: (1) unemployed households with previously low liquid assets exhibit substantially higher marginal propensities to consume (MPCs) out of stimulus transfers compared to high-asset households (0.50 vs 0.34), even when neither group faces borrowing constraints, consistent with Ganong et al. (2024); and (2) households with more past unemployment experiences maintain persistently lower consumption levels after controlling for current economic conditions, a "scarring" effect documented by Malmendier and Shen (2024). Unlike existing explanations based on belief updating about income risk or ex-ante heterogeneity, the reinforcement learning mechanism generates both higher MPCs and lower consumption levels simultaneously through value function approximation errors that evolve with experience. Simulation results closely match the empirical estimates, suggesting that adaptive learning through reinforcement learning provides a unifying framework for understanding how past experiences shape current consumption behavior beyond what current economic conditions would predict. 

---
# Empathic Prompting: Non-Verbal Context Integration for Multimodal LLM Conversations 

**Authors**: Lorenzo Stacchio, Andrea Ubaldi, Alessandro Galdelli, Maurizio Mauri, Emanuele Frontoni, Andrea Gaggioli  

**Link**: [PDF](https://arxiv.org/pdf/2510.20743)  

**Abstract**: We present Empathic Prompting, a novel framework for multimodal human-AI interaction that enriches Large Language Model (LLM) conversations with implicit non-verbal context. The system integrates a commercial facial expression recognition service to capture users' emotional cues and embeds them as contextual signals during prompting. Unlike traditional multimodal interfaces, empathic prompting requires no explicit user control; instead, it unobtrusively augments textual input with affective information for conversational and smoothness alignment. The architecture is modular and scalable, allowing integration of additional non-verbal modules. We describe the system design, implemented through a locally deployed DeepSeek instance, and report a preliminary service and usability evaluation (N=5). Results show consistent integration of non-verbal input into coherent LLM outputs, with participants highlighting conversational fluidity. Beyond this proof of concept, empathic prompting points to applications in chatbot-mediated communication, particularly in domains like healthcare or education, where users' emotional signals are critical yet often opaque in verbal exchanges. 

---
# Thought Communication in Multiagent Collaboration 

**Authors**: Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20733)  

**Abstract**: Natural language has long enabled human cooperation, but its lossy, ambiguous, and indirect nature limits the potential of collective intelligence. While machines are not subject to these constraints, most LLM-based multi-agent systems still rely solely on natural language, exchanging tokens or their embeddings. To go beyond language, we introduce a new paradigm, thought communication, which enables agents to interact directly mind-to-mind, akin to telepathy. To uncover these latent thoughts in a principled way, we formalize the process as a general latent variable model, where agent states are generated by an unknown function of underlying thoughts. We prove that, in a nonparametric setting without auxiliary information, both shared and private latent thoughts between any pair of agents can be identified. Moreover, the global structure of thought sharing, including which agents share which thoughts and how these relationships are structured, can also be recovered with theoretical guarantees. Guided by the established theory, we develop a framework that extracts latent thoughts from all agents prior to communication and assigns each agent the relevant thoughts, along with their sharing patterns. This paradigm naturally extends beyond LLMs to all modalities, as most observational data arise from hidden generative processes. Experiments on both synthetic and real-world benchmarks validate the theory and demonstrate the collaborative advantages of thought communication. We hope this work illuminates the potential of leveraging the hidden world, as many challenges remain unsolvable through surface-level observation alone, regardless of compute or data scale. 

---
# Co-Designing Quantum Codes with Transversal Diagonal Gates via Multi-Agent Systems 

**Authors**: Xi He, Sirui Lu, Bei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.20728)  

**Abstract**: We present a multi-agent, human-in-the-loop workflow that co-designs quantum codes with prescribed transversal diagonal gates. It builds on the Subset-Sum Linear Programming (SSLP) framework (arXiv:2504.20847), which partitions basis strings by modular residues and enforces $Z$-marginal Knill-Laflamme (KL) equalities via small LPs. The workflow is powered by GPT-5 and implemented within TeXRA (this https URL)-a multi-agent research assistant platform that supports an iterative tool-use loop agent and a derivation-then-edit workflow reasoning agent. We work in a LaTeX-Python environment where agents reason, edit documents, execute code, and synchronize their work to Git/Overleaf. Within this workspace, three roles collaborate: a Synthesis Agent formulates the problem; a Search Agent sweeps/screens candidates and exactifies numerics into rationals; and an Audit Agent independently checks all KL equalities and the induced logical action. As a first step we focus on distance $d=2$ with nondegenerate residues. For code dimension $K\in\{2,3,4\}$ and $n\le6$ qubits, systematic sweeps yield certificate-backed tables cataloging attainable cyclic logical groups-all realized by new codes-e.g., for $K=3$ we obtain order $16$ at $n=6$. From verified instances, Synthesis Agent abstracts recurring structures into closed-form families and proves they satisfy the KL equalities for all parameters. It further demonstrates that SSLP accommodates residue degeneracy by exhibiting a new $((6,4,2))$ code implementing the transversal controlled-phase $diag(1,1,1,i)$. Overall, the workflow recasts diagonal-transversal feasibility as an analytical pipeline executed at scale, combining systematic enumeration with exact analytical reconstruction. It yields reproducible code constructions, supports targeted extensions to larger $K$ and higher distances, and leads toward data-driven classification. 

---
# Automated Extraction of Fluoropyrimidine Treatment and Treatment-Related Toxicities from Clinical Notes Using Natural Language Processing 

**Authors**: Xizhi Wu, Madeline S. Kreider, Philip E. Empey, Chenyu Li, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20727)  

**Abstract**: Objective: Fluoropyrimidines are widely prescribed for colorectal and breast cancers, but are associated with toxicities such as hand-foot syndrome and cardiotoxicity. Since toxicity documentation is often embedded in clinical notes, we aimed to develop and evaluate natural language processing (NLP) methods to extract treatment and toxicity information.
Materials and Methods: We constructed a gold-standard dataset of 236 clinical notes from 204,165 adult oncology patients. Domain experts annotated categories related to treatment regimens and toxicities. We developed rule-based, machine learning-based (Random Forest, Support Vector Machine [SVM], Logistic Regression [LR]), deep learning-based (BERT, ClinicalBERT), and large language models (LLM)-based NLP approaches (zero-shot and error-analysis prompting). Models used an 80:20 train-test split.
Results: Sufficient data existed to train and evaluate 5 annotated categories. Error-analysis prompting achieved optimal precision, recall, and F1 scores (F1=1.000) for treatment and toxicities extraction, whereas zero-shot prompting reached F1=1.000 for treatment and F1=0.876 for toxicities this http URL and SVM ranked second for toxicities (F1=0.937). Deep learning underperformed, with BERT (F1=0.873 treatment; F1= 0.839 toxicities) and ClinicalBERT (F1=0.873 treatment; F1 = 0.886 toxicities). Rule-based methods served as our baseline with F1 scores of 0.857 in treatment and 0.858 in toxicities.
Discussion: LMM-based approaches outperformed all others, followed by machine learning methods. Machine and deep learning approaches were limited by small training data and showed limited generalizability, particularly for rare categories.
Conclusion: LLM-based NLP most effectively extracted fluoropyrimidine treatment and toxicity information from clinical notes, and has strong potential to support oncology research and pharmacovigilance. 

---
# User Perceptions of Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios 

**Authors**: Xiaoyuan Wu, Roshni Kaushik, Wenkai Li, Lujo Bauer, Koichi Onoue  

**Link**: [PDF](https://arxiv.org/pdf/2510.20721)  

**Abstract**: Large language models (LLMs) have seen rapid adoption for tasks such as drafting emails, summarizing meetings, and answering health questions. In such uses, users may need to share private information (e.g., health records, contact details). To evaluate LLMs' ability to identify and redact such private information, prior work developed benchmarks (e.g., ConfAIde, PrivacyLens) with real-life scenarios. Using these benchmarks, researchers have found that LLMs sometimes fail to keep secrets private when responding to complex tasks (e.g., leaking employee salaries in meeting summaries). However, these evaluations rely on LLMs (proxy LLMs) to gauge compliance with privacy norms, overlooking real users' perceptions. Moreover, prior work primarily focused on the privacy-preservation quality of responses, without investigating nuanced differences in helpfulness. To understand how users perceive the privacy-preservation quality and helpfulness of LLM responses to privacy-sensitive scenarios, we conducted a user study with 94 participants using 90 scenarios from PrivacyLens. We found that, when evaluating identical responses to the same scenario, users showed low agreement with each other on the privacy-preservation quality and helpfulness of the LLM response. Further, we found high agreement among five proxy LLMs, while each individual LLM had low correlation with users' evaluations. These results indicate that the privacy and helpfulness of LLM responses are often specific to individuals, and proxy LLMs are poor estimates of how real users would perceive these responses in privacy-sensitive scenarios. Our results suggest the need to conduct user-centered studies on measuring LLMs' ability to help users while preserving privacy. Additionally, future research could investigate ways to improve the alignment between proxy LLMs and users for better estimation of users' perceived privacy and utility. 

---
# Unsupervised Anomaly Prediction with N-BEATS and Graph Neural Network in Multi-variate Semiconductor Process Time Series 

**Authors**: Daniel Sorensen, Bappaditya Dey, Minjin Hwang, Sandip Halder  

**Link**: [PDF](https://arxiv.org/pdf/2510.20718)  

**Abstract**: Semiconductor manufacturing is an extremely complex and precision-driven process, characterized by thousands of interdependent parameters collected across diverse tools and process steps. Multi-variate time-series analysis has emerged as a critical field for real-time monitoring and fault detection in such environments. However, anomaly prediction in semiconductor fabrication presents several critical challenges, including high dimensionality of sensor data and severe class imbalance due to the rarity of true faults. Furthermore, the complex interdependencies between variables complicate both anomaly prediction and root-cause-analysis. This paper proposes two novel approaches to advance the field from anomaly detection to anomaly prediction, an essential step toward enabling real-time process correction and proactive fault prevention. The proposed anomaly prediction framework contains two main stages: (a) training a forecasting model on a dataset assumed to contain no anomalies, and (b) performing forecast on unseen time series data. The forecast is compared with the forecast of the trained signal. Deviations beyond a predefined threshold are flagged as anomalies. The two approaches differ in the forecasting model employed. The first assumes independence between variables by utilizing the N-BEATS model for univariate time series forecasting. The second lifts this assumption by utilizing a Graph Neural Network (GNN) to capture inter-variable relationships. Both models demonstrate strong forecasting performance up to a horizon of 20 time points and maintain stable anomaly prediction up to 50 time points. The GNN consistently outperforms the N-BEATS model while requiring significantly fewer trainable parameters and lower computational cost. These results position the GNN as promising solution for online anomaly forecasting to be deployed in manufacturing environments. 

---
# Real-Time Gait Adaptation for Quadrupeds using Model Predictive Control and Reinforcement Learning 

**Authors**: Ganga Nair B, Prakrut Kotecha, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2510.20706)  

**Abstract**: Model-free reinforcement learning (RL) has enabled adaptable and agile quadruped locomotion; however, policies often converge to a single gait, leading to suboptimal performance. Traditionally, Model Predictive Control (MPC) has been extensively used to obtain task-specific optimal policies but lacks the ability to adapt to varying environments. To address these limitations, we propose an optimization framework for real-time gait adaptation in a continuous gait space, combining the Model Predictive Path Integral (MPPI) algorithm with a Dreamer module to produce adaptive and optimal policies for quadruped locomotion. At each time step, MPPI jointly optimizes the actions and gait variables using a learned Dreamer reward that promotes velocity tracking, energy efficiency, stability, and smooth transitions, while penalizing abrupt gait changes. A learned value function is incorporated as terminal reward, extending the formulation to an infinite-horizon planner. We evaluate our framework in simulation on the Unitree Go1, demonstrating an average reduction of up to 36.48\% in energy consumption across varying target speeds, while maintaining accurate tracking and adaptive, task-appropriate gaits. 

---
# Fusing Narrative Semantics for Financial Volatility Forecasting 

**Authors**: Yaxuan Kong, Yoontae Hwang, Marcus Kaiser, Chris Vryonides, Roel Oomen, Stefan Zohren  

**Link**: [PDF](https://arxiv.org/pdf/2510.20699)  

**Abstract**: We introduce M2VN: Multi-Modal Volatility Network, a novel deep learning-based framework for financial volatility forecasting that unifies time series features with unstructured news data. M2VN leverages the representational power of deep neural networks to address two key challenges in this domain: (i) aligning and fusing heterogeneous data modalities, numerical financial data and textual information, and (ii) mitigating look-ahead bias that can undermine the validity of financial models. To achieve this, M2VN combines open-source market features with news embeddings generated by Time Machine GPT, a recently introduced point-in-time LLM, ensuring temporal integrity. An auxiliary alignment loss is introduced to enhance the integration of structured and unstructured data within the deep learning architecture. Extensive experiments demonstrate that M2VN consistently outperforms existing baselines, underscoring its practical value for risk management and financial decision-making in dynamic markets. 

---
# Exploring Large Language Models for Access Control Policy Synthesis and Summarization 

**Authors**: Adarsh Vatsa, Bethel Hall, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2510.20692)  

**Abstract**: Cloud computing is ubiquitous, with a growing number of services being hosted on the cloud every day. Typical cloud compute systems allow administrators to write policies implementing access control rules which specify how access to private data is governed. These policies must be manually written, and due to their complexity can often be error prone. Moreover, existing policies often implement complex access control specifications and thus can be difficult to precisely analyze in determining their behavior works exactly as intended. Recently, Large Language Models (LLMs) have shown great success in automated code synthesis and summarization. Given this success, they could potentially be used for automatically generating access control policies or aid in understanding existing policies. In this paper, we explore the effectiveness of LLMs for access control policy synthesis and summarization. Specifically, we first investigate diverse LLMs for access control policy synthesis, finding that: although LLMs can effectively generate syntactically correct policies, they have permissiveness issues, generating policies equivalent to the given specification 45.8% of the time for non-reasoning LLMs, and 93.7% of the time for reasoning LLMs. We then investigate how LLMs can be used to analyze policies by introducing a novel semantic-based request summarization approach which leverages LLMs to generate a precise characterization of the requests allowed by a policy. Our results show that while there are significant hurdles in leveraging LLMs for automated policy generation, LLMs show promising results when combined with symbolic approaches in analyzing existing policies. 

---
# Neural Diversity Regularizes Hallucinations in Small Models 

**Authors**: Kushal Chakrabarti, Nirmal Balachundhar  

**Link**: [PDF](https://arxiv.org/pdf/2510.20690)  

**Abstract**: Language models continue to hallucinate despite increases in parameters, compute, and data. We propose neural diversity -- decorrelated parallel representations -- as a principled mechanism that reduces hallucination rates at fixed parameter and data budgets. Inspired by portfolio theory, where uncorrelated assets reduce risk by $\sqrt{P}$, we prove hallucination probability is bounded by representational correlation: $P(H) \leq f(\sigma^2((1-\rho(P))/P + \rho(P)), \mu^2)$, which predicts that language models need an optimal amount of neurodiversity. To validate this, we introduce ND-LoRA (Neural Diversity Low-Rank Adaptation), combining parallel LoRA adapters with Barlow Twins regularization, and demonstrate that ND-LoRA reduces hallucinations by up to 25.6% (and 14.6% on average) without degrading general accuracy. Ablations show LoRA adapters and regularization act synergistically, causal interventions prove neurodiversity as the mediating factor and correlational analyses indicate scale: a 0.1% neural correlation increase is associated with a 3.8% hallucination increase. Finally, task-dependent optimality emerges: different tasks require different amounts of optimal neurodiversity. Together, our results highlight neural diversity as a third axis of scaling -- orthogonal to parameters and data -- to improve the reliability of language models at fixed budgets. 

---
# A Scalable, Causal, and Energy Efficient Framework for Neural Decoding with Spiking Neural Networks 

**Authors**: Georgios Mentzelopoulos, Ioannis Asmanis, Konrad P. Kording, Eva L. Dyer, Kostas Daniilidis, Flavia Vitale  

**Link**: [PDF](https://arxiv.org/pdf/2510.20683)  

**Abstract**: Brain-computer interfaces (BCIs) promise to enable vital functions, such as speech and prosthetic control, for individuals with neuromotor impairments. Central to their success are neural decoders, models that map neural activity to intended behavior. Current learning-based decoding approaches fall into two classes: simple, causal models that lack generalization, or complex, non-causal models that generalize and scale offline but struggle in real-time settings. Both face a common challenge, their reliance on power-hungry artificial neural network backbones, which makes integration into real-world, resource-limited systems difficult. Spiking neural networks (SNNs) offer a promising alternative. Because they operate causally these models are suitable for real-time use, and their low energy demands make them ideal for battery-constrained environments. To this end, we introduce Spikachu: a scalable, causal, and energy-efficient neural decoding framework based on SNNs. Our approach processes binned spikes directly by projecting them into a shared latent space, where spiking modules, adapted to the timing of the input, extract relevant features; these latent representations are then integrated and decoded to generate behavioral predictions. We evaluate our approach on 113 recording sessions from 6 non-human primates, totaling 43 hours of recordings. Our method outperforms causal baselines when trained on single sessions using between 2.26 and 418.81 times less energy. Furthermore, we demonstrate that scaling up training to multiple sessions and subjects improves performance and enables few-shot transfer to unseen sessions, subjects, and tasks. Overall, Spikachu introduces a scalable, online-compatible neural decoding framework based on SNNs, whose performance is competitive relative to state-of-the-art models while consuming orders of magnitude less energy. 

---
# R2-SVC: Towards Real-World Robust and Expressive Zero-shot Singing Voice Conversion 

**Authors**: Junjie Zheng, Gongyu Chen, Chaofan Ding, Zihao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20677)  

**Abstract**: In real-world singing voice conversion (SVC) applications, environmental noise and the demand for expressive output pose significant challenges. Conventional methods, however, are typically designed without accounting for real deployment scenarios, as both training and inference usually rely on clean data. This mismatch hinders practical use, given the inevitable presence of diverse noise sources and artifacts from music separation. To tackle these issues, we propose R2-SVC, a robust and expressive SVC framework. First, we introduce simulation-based robustness enhancement through random fundamental frequency ($F_0$) perturbations and music separation artifact simulations (e.g., reverberation, echo), substantially improving performance under noisy conditions. Second, we enrich speaker representation using domain-specific singing data: alongside clean vocals, we incorporate DNSMOS-filtered separated vocals and public singing corpora, enabling the model to preserve speaker timbre while capturing singing style nuances. Third, we integrate the Neural Source-Filter (NSF) model to explicitly represent harmonic and noise components, enhancing the naturalness and controllability of converted singing. R2-SVC achieves state-of-the-art results on multiple SVC benchmarks under both clean and noisy conditions. 

---
# GRACE: GRaph-based Addiction Care prEdiction 

**Authors**: Subham Kumar, Prakrithi Shivaprakash, Koustav Rudra, Lekhansh Shukla, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.20671)  

**Abstract**: Determining the appropriate locus of care for addiction patients is one of the most critical clinical decisions that affects patient treatment outcomes and effective use of resources. With a lack of sufficient specialized treatment resources, such as inpatient beds or staff, there is an unmet need to develop an automated framework for the same. Current decision-making approaches suffer from severe class imbalances in addiction datasets. To address this limitation, we propose a novel graph neural network (GRACE) framework that formalizes locus of care prediction as a structured learning problem. Further, we perform extensive feature engineering and propose a new approach of obtaining an unbiased meta-graph to train a GNN to overcome the class imbalance problem. Experimental results in real-world data show an improvement of 11-35% in terms of the F1 score of the minority class over competitive baselines. The codes and note embeddings are available at this https URL. 

---
# Finding the Sweet Spot: Trading Quality, Cost, and Speed During Inference-Time LLM Reflection 

**Authors**: Jack Butler, Nikita Kozodoi, Zainab Afolabi, Brian Tyacke, Gaiar Baimuratov  

**Link**: [PDF](https://arxiv.org/pdf/2510.20653)  

**Abstract**: As Large Language Models (LLMs) continue to evolve, practitioners face increasing options for enhancing inference-time performance without model retraining, including budget tuning and multi-step techniques like self-reflection. While these methods improve output quality, they create complex trade-offs among accuracy, cost, and latency that remain poorly understood across different domains. This paper systematically compares self-reflection and budget tuning across mathematical reasoning and translation tasks. We evaluate prominent LLMs, including Anthropic Claude, Amazon Nova, and Mistral families, along with other models under varying reflection depths and compute budgets to derive Pareto optimal performance frontiers. Our analysis reveals substantial domain dependent variation in self-reflection effectiveness, with performance gains up to 220\% in mathematical reasoning. We further investigate how reflection round depth and feedback mechanism quality influence performance across model families. To validate our findings in a real-world setting, we deploy a self-reflection enhanced marketing content localisation system at Lounge by Zalando, where it shows market-dependent effectiveness, reinforcing the importance of domain specific evaluation when deploying these techniques. Our results provide actionable guidance for selecting optimal inference strategies given specific domains and resource constraints. We open source our self-reflection implementation for reproducibility at this https URL. 

---
# The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI 

**Authors**: Alan Saji, Raj Dabre, Anoop Kunchukuttan, Ratish Puduppully  

**Link**: [PDF](https://arxiv.org/pdf/2510.20647)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on mathematical, scientific, and other question-answering tasks, but their multilingual reasoning abilities remain underexplored. When presented with non-English questions, LRMs often default to reasoning in English, raising concerns about interpretability and the handling of linguistic and cultural nuances. We systematically compare an LRM's reasoning in English versus the language of the question. Our evaluation spans two tasks: MGSM and GPQA Diamond. Beyond measuring answer accuracy, we also analyze cognitive attributes in the reasoning traces. We find that English reasoning traces exhibit a substantially higher presence of these cognitive behaviors, and that reasoning in English generally yields higher final-answer accuracy, with the performance gap increasing as tasks become more complex. However, this English-centric strategy is susceptible to a key failure mode - getting "Lost in Translation," where translation steps lead to errors that would have been avoided by question's language reasoning. 

---
# Why Did Apple Fall To The Ground: Evaluating Curiosity In Large Language Model 

**Authors**: Haoyu Wang, Sihang Jiang, Yuyan Chen, Yitong Wang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20635)  

**Abstract**: Curiosity serves as a pivotal conduit for human beings to discover and learn new knowledge. Recent advancements of large language models (LLMs) in natural language processing have sparked discussions regarding whether these models possess capability of curiosity-driven learning akin to humans. In this paper, starting from the human curiosity assessment questionnaire Five-Dimensional Curiosity scale Revised (5DCR), we design a comprehensive evaluation framework that covers dimensions such as Information Seeking, Thrill Seeking, and Social Curiosity to assess the extent of curiosity exhibited by LLMs. The results demonstrate that LLMs exhibit a stronger thirst for knowledge than humans but still tend to make conservative choices when faced with uncertain environments. We further investigated the relationship between curiosity and thinking of LLMs, confirming that curious behaviors can enhance the model's reasoning and active learning abilities. These findings suggest that LLMs have the potential to exhibit curiosity similar to that of humans, providing experimental support for the future development of learning capabilities and innovative research in LLMs. 

---
# Deep Learning in Dental Image Analysis: A Systematic Review of Datasets, Methodologies, and Emerging Challenges 

**Authors**: Zhenhuan Zhou, Jingbo Zhu, Yuchen Zhang, Xiaohang Guan, Peng Wang, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.20634)  

**Abstract**: Efficient analysis and processing of dental images are crucial for dentists to achieve accurate diagnosis and optimal treatment planning. However, dental imaging inherently poses several challenges, such as low contrast, metallic artifacts, and variations in projection angles. Combined with the subjectivity arising from differences in clinicians' expertise, manual interpretation often proves time-consuming and prone to inconsistency. Artificial intelligence (AI)-based automated dental image analysis (DIA) offers a promising solution to these issues and has become an integral part of computer-aided dental diagnosis and treatment. Among various AI technologies, deep learning (DL) stands out as the most widely applied and influential approach due to its superior feature extraction and representation capabilities. To comprehensively summarize recent progress in this field, we focus on the two fundamental aspects of DL research-datasets and models. In this paper, we systematically review 260 studies on DL applications in DIA, including 49 papers on publicly available dental datasets and 211 papers on DL-based algorithms. We first introduce the basic concepts of dental imaging and summarize the characteristics and acquisition methods of existing datasets. Then, we present the foundational techniques of DL and categorize relevant models and algorithms according to different DIA tasks, analyzing their network architectures, optimization strategies, training methods, and performance. Furthermore, we summarize commonly used training and evaluation metrics in the DIA domain. Finally, we discuss the current challenges of existing research and outline potential future directions. We hope that this work provides a valuable and systematic reference for researchers in this field. All supplementary materials and detailed comparison tables will be made publicly available on GitHub. 

---
# Quantum Processing Unit (QPU) processing time Prediction with Machine Learning 

**Authors**: Lucy Xing, Sanjay Vishwakarma, David Kremer, Francisco Martin-Fernandez, Ismael Faro, Juan Cruz-Benito  

**Link**: [PDF](https://arxiv.org/pdf/2510.20630)  

**Abstract**: This paper explores the application of machine learning (ML) techniques in predicting the QPU processing time of quantum jobs. By leveraging ML algorithms, this study introduces predictive models that are designed to enhance operational efficiency in quantum computing systems. Using a dataset of about 150,000 jobs that follow the IBM Quantum schema, we employ ML methods based on Gradient-Boosting (LightGBM) to predict the QPU processing times, incorporating data preprocessing methods to improve model accuracy. The results demonstrate the effectiveness of ML in forecasting quantum jobs. This improvement can have implications on improving resource management and scheduling within quantum computing frameworks. This research not only highlights the potential of ML in refining quantum job predictions but also sets a foundation for integrating AI-driven tools in advanced quantum computing operations. 

---
# Equitable Survival Prediction: A Fairness-Aware Survival Modeling (FASM) Approach 

**Authors**: Mingxuan Liu, Yilin Ning, Haoyuan Wang, Chuan Hong, Matthew Engelhard, Danielle S. Bitterman, William G. La Cava, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20629)  

**Abstract**: As machine learning models become increasingly integrated into healthcare, structural inequities and social biases embedded in clinical data can be perpetuated or even amplified by data-driven models. In survival analysis, censoring and time dynamics can further add complexity to fair model development. Additionally, algorithmic fairness approaches often overlook disparities in cross-group rankings, e.g., high-risk Black patients may be ranked below lower-risk White patients who do not experience the event of mortality. Such misranking can reinforce biological essentialism and undermine equitable care. We propose a Fairness-Aware Survival Modeling (FASM), designed to mitigate algorithmic bias regarding both intra-group and cross-group risk rankings over time. Using breast cancer prognosis as a representative case and applying FASM to SEER breast cancer data, we show that FASM substantially improves fairness while preserving discrimination performance comparable to fairness-unaware survival models. Time-stratified evaluations show that FASM maintains stable fairness over a 10-year horizon, with the greatest improvements observed during the mid-term of follow-up. Our approach enables the development of survival models that prioritize both accuracy and equity in clinical decision-making, advancing fairness as a core principle in clinical care. 

---
# Black Box Absorption: LLMs Undermining Innovative Ideas 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20612)  

**Abstract**: Large Language Models are increasingly adopted as critical tools for accelerating innovation. This paper identifies and formalizes a systemic risk inherent in this paradigm: \textbf{Black Box Absorption}. We define this as the process by which the opaque internal architectures of LLM platforms, often operated by large-scale service providers, can internalize, generalize, and repurpose novel concepts contributed by users during interaction. This mechanism threatens to undermine the foundational principles of innovation economics by creating severe informational and structural asymmetries between individual creators and platform operators, thereby jeopardizing the long-term sustainability of the innovation ecosystem. To analyze this challenge, we introduce two core concepts: the idea unit, representing the transportable functional logic of an innovation, and idea safety, a multidimensional standard for its protection. This paper analyzes the mechanisms of absorption and proposes a concrete governance and engineering agenda to mitigate these risks, ensuring that creator contributions remain traceable, controllable, and equitable. 

---
# PSO-XAI: A PSO-Enhanced Explainable AI Framework for Reliable Breast Cancer Detection 

**Authors**: Mirza Raquib, Niloy Das, Farida Siddiqi Prity, Arafath Al Fahim, Saydul Akbar Murad, Mohammad Amzad Hossain, MD Jiabul Hoque, Mohammad Ali Moni  

**Link**: [PDF](https://arxiv.org/pdf/2510.20611)  

**Abstract**: Breast cancer is considered the most critical and frequently diagnosed cancer in women worldwide, leading to an increase in cancer-related mortality. Early and accurate detection is crucial as it can help mitigate possible threats while improving survival rates. In terms of prediction, conventional diagnostic methods are often limited by variability, cost, and, most importantly, risk of misdiagnosis. To address these challenges, machine learning (ML) has emerged as a powerful tool for computer-aided diagnosis, with feature selection playing a vital role in improving model performance and interpretability. This research study proposes an integrated framework that incorporates customized Particle Swarm Optimization (PSO) for feature selection. This framework has been evaluated on a comprehensive set of 29 different models, spanning classical classifiers, ensemble techniques, neural networks, probabilistic algorithms, and instance-based algorithms. To ensure interpretability and clinical relevance, the study uses cross-validation in conjunction with explainable AI methods. Experimental evaluation showed that the proposed approach achieved a superior score of 99.1\% across all performance metrics, including accuracy and precision, while effectively reducing dimensionality and providing transparent, model-agnostic explanations. The results highlight the potential of combining swarm intelligence with explainable ML for robust, trustworthy, and clinically meaningful breast cancer diagnosis. 

---
# BUSTED at AraGenEval Shared Task: A Comparative Study of Transformer-Based Models for Arabic AI-Generated Text Detection 

**Authors**: Ali Zain, Sareem Farooqui, Muhammad Rafi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20610)  

**Abstract**: This paper details our submission to the Ara- GenEval Shared Task on Arabic AI-generated text detection, where our team, BUSTED, se- cured 5th place. We investigated the effec- tiveness of three pre-trained transformer mod- els: AraELECTRA, CAMeLBERT, and XLM- RoBERTa. Our approach involved fine-tuning each model on the provided dataset for a binary classification task. Our findings revealed a sur- prising result: the multilingual XLM-RoBERTa model achieved the highest performance with an F1 score of 0.7701, outperforming the spe- cialized Arabic models. This work underscores the complexities of AI-generated text detection and highlights the strong generalization capa- bilities of multilingual models. 

---
# Practical Code RAG at Scale: Task-Aware Retrieval Design Choices under Compute Budgets 

**Authors**: Timur Galimzyanov, Olga Kolomyttseva, Egor Bogomolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.20609)  

**Abstract**: We study retrieval design for code-focused generation tasks under realistic compute budgets. Using two complementary tasks from Long Code Arena -- code completion and bug localization -- we systematically compare retrieval configurations across various context window sizes along three axes: (i) chunking strategy, (ii) similarity scoring, and (iii) splitting granularity. (1) For PL-PL, sparse BM25 with word-level splitting is the most effective and practical, significantly outperforming dense alternatives while being an order of magnitude faster. (2) For NL-PL, proprietary dense encoders (Voyager-3 family) consistently beat sparse retrievers, however requiring 100x larger latency. (3) Optimal chunk size scales with available context: 32-64 line chunks work best at small budgets, and whole-file retrieval becomes competitive at 16000 tokens. (4) Simple line-based chunking matches syntax-aware splitting across budgets. (5) Retrieval latency varies by up to 200x across configurations; BPE-based splitting is needlessly slow, and BM25 + word splitting offers the best quality-latency trade-off. Thus, we provide evidence-based recommendations for implementing effective code-oriented RAG systems based on task requirements, model constraints, and computational efficiency. 

---
# Generalizable Reasoning through Compositional Energy Minimization 

**Authors**: Alexandru Oarga, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.20607)  

**Abstract**: Generalization is a key challenge in machine learning, specifically in reasoning tasks, where models are expected to solve problems more complex than those encountered during training. Existing approaches typically train reasoning models in an end-to-end fashion, directly mapping input instances to solutions. While this allows models to learn useful heuristics from data, it often results in limited generalization beyond the training distribution. In this work, we propose a novel approach to reasoning generalization by learning energy landscapes over the solution spaces of smaller, more tractable subproblems. At test time, we construct a global energy landscape for a given problem by combining the energy functions of multiple subproblems. This compositional approach enables the incorporation of additional constraints during inference, allowing the construction of energy landscapes for problems of increasing difficulty. To improve the sample quality from this newly constructed energy landscape, we introduce Parallel Energy Minimization (PEM). We evaluate our approach on a wide set of reasoning problems. Our method outperforms existing state-of-the-art methods, demonstrating its ability to generalize to larger and more complex problems. Project website can be found at: this https URL 

---
# OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects 

**Authors**: Mark He Huang, Lin Geng Foo, Christian Theobalt, Ying Sun, De Wen Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.20605)  

**Abstract**: Free-moving object reconstruction from monocular video remains challenging, particularly without reliable pose or depth cues and under arbitrary object motion. We introduce OnlineSplatter, a novel online feed-forward framework generating high-quality, object-centric 3D Gaussians directly from RGB frames without requiring camera pose, depth priors, or bundle optimization. Our approach anchors reconstruction using the first frame and progressively refines the object representation through a dense Gaussian primitive field, maintaining constant computational cost regardless of video sequence length. Our core contribution is a dual-key memory module combining latent appearance-geometry keys with explicit directional keys, robustly fusing current frame features with temporally aggregated object states. This design enables effective handling of free-moving objects via spatial-guided memory readout and an efficient sparsification mechanism, ensuring comprehensive yet compact object coverage. Evaluations on real-world datasets demonstrate that OnlineSplatter significantly outperforms state-of-the-art pose-free reconstruction baselines, consistently improving with more observations while maintaining constant memory and runtime. 

---
# Resounding Acoustic Fields with Reciprocity 

**Authors**: Zitong Lan, Yiduo Hao, Mingmin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20602)  

**Abstract**: Achieving immersive auditory experiences in virtual environments requires flexible sound modeling that supports dynamic source positions. In this paper, we introduce a task called resounding, which aims to estimate room impulse responses at arbitrary emitter location from a sparse set of measured emitter positions, analogous to the relighting problem in vision. We leverage the reciprocity property and introduce Versa, a physics-inspired approach to facilitating acoustic field learning. Our method creates physically valid samples with dense virtual emitter positions by exchanging emitter and listener poses. We also identify challenges in deploying reciprocity due to emitter/listener gain patterns and propose a self-supervised learning approach to address them. Results show that Versa substantially improve the performance of acoustic field learning on both simulated and real-world datasets across different metrics. Perceptual user studies show that Versa can greatly improve the immersive spatial sound experience. Code, dataset and demo videos are available on the project website: this https URL. 

---
# Unsupervised Domain Adaptation via Similarity-based Prototypes for Cross-Modality Segmentation 

**Authors**: Ziyu Ye, Chen Ju, Chaofan Ma, Xiaoyun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20596)  

**Abstract**: Deep learning models have achieved great success on various vision challenges, but a well-trained model would face drastic performance degradation when applied to unseen data. Since the model is sensitive to domain shift, unsupervised domain adaptation attempts to reduce the domain gap and avoid costly annotation of unseen domains. This paper proposes a novel framework for cross-modality segmentation via similarity-based prototypes. In specific, we learn class-wise prototypes within an embedding space, then introduce a similarity constraint to make these prototypes representative for each semantic class while separable from different classes. Moreover, we use dictionaries to store prototypes extracted from different images, which prevents the class-missing problem and enables the contrastive learning of prototypes, and further improves performance. Extensive experiments show that our method achieves better results than other state-of-the-art methods. 

---
# Can ChatGPT Code Communication Data Fairly?: Empirical Evidence from Multiple Collaborative Tasks 

**Authors**: Jiangang Hao, Wenju Cui, Patrick Kyllonen, Emily Kerzabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20584)  

**Abstract**: Assessing communication and collaboration at scale depends on a labor intensive task of coding communication data into categories according to different frameworks. Prior research has established that ChatGPT can be directly instructed with coding rubrics to code the communication data and achieves accuracy comparable to human raters. However, whether the coding from ChatGPT or similar AI technology exhibits bias against different demographic groups, such as gender and race, remains unclear. To fill this gap, this paper investigates ChatGPT-based automated coding of communication data using a typical coding framework for collaborative problem solving, examining differences across gender and racial groups. The analysis draws on data from three types of collaborative tasks: negotiation, problem solving, and decision making. Our results show that ChatGPT-based coding exhibits no significant bias across gender and racial groups, paving the road for its adoption in large-scale assessment of collaboration and communication. 

---
# Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence 

**Authors**: Jiahao Meng, Xiangtai Li, Haochen Wang, Yue Tan, Tao Zhang, Lingdong Kong, Yunhai Tong, Anran Wang, Zhiyang Teng, Yujing Wang, Zhuochen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20579)  

**Abstract**: Most video reasoning models only generate textual reasoning traces without indicating when and where key evidence appears. Recent models such as OpenAI-o3 have sparked wide interest in evidence-centered reasoning for images, yet extending this ability to videos is more challenging, as it requires joint temporal tracking and spatial localization across dynamic scenes. We introduce Open-o3 Video, a non-agent framework that integrates explicit spatio-temporal evidence into video reasoning, and carefully collect training data and design training strategies to address the aforementioned challenges. The model highlights key timestamps, objects, and bounding boxes alongside its answers, allowing reasoning to be grounded in concrete visual observations. To enable this functionality, we first curate and build two high-quality datasets, STGR-CoT-30k for SFT and STGR-RL-36k for RL, with carefully constructed temporal and spatial annotations, since most existing datasets offer either temporal spans for videos or spatial boxes on images, lacking unified spatio-temporal supervision and reasoning traces. Then, we adopt a cold-start reinforcement learning strategy with multiple specially designed rewards that jointly encourage answer accuracy, temporal alignment, and spatial precision. On V-STAR benchmark, Open-o3 Video achieves state-of-the-art performance, raising mAM by 14.4% and mLGM by 24.2% on the Qwen2.5-VL baseline. Consistent improvements are also observed on a broad range of video understanding benchmarks, including VideoMME, WorldSense, VideoMMMU, and TVGBench. Beyond accuracy, the reasoning traces produced by Open-o3 Video also provide valuable signals for test-time scaling, enabling confidence-aware verification and improving answer reliability. 

---
# AdaDoS: Adaptive DoS Attack via Deep Adversarial Reinforcement Learning in SDN 

**Authors**: Wei Shao, Yuhao Wang, Rongguang He, Muhammad Ejaz Ahmed, Seyit Camtepe  

**Link**: [PDF](https://arxiv.org/pdf/2510.20566)  

**Abstract**: Existing defence mechanisms have demonstrated significant effectiveness in mitigating rule-based Denial-of-Service (DoS) attacks, leveraging predefined signatures and static heuristics to identify and block malicious traffic. However, the emergence of AI-driven techniques presents new challenges to SDN security, potentially compromising the efficacy of existing defence mechanisms. In this paper, we introduce~AdaDoS, an adaptive attack model that disrupt network operations while evading detection by existing DoS-based detectors through adversarial reinforcement learning (RL). Specifically, AdaDoS models the problem as a competitive game between an attacker, whose goal is to obstruct network traffic without being detected, and a detector, which aims to identify malicious traffic. AdaDoS can solve this game by dynamically adjusting its attack strategy based on feedback from the SDN and the detector. Additionally, recognising that attackers typically have less information than defenders, AdaDoS formulates the DoS-like attack as a partially observed Markov decision process (POMDP), with the attacker having access only to delay information between attacker and victim nodes. We address this challenge with a novel reciprocal learning module, where the student agent, with limited observations, enhances its performance by learning from the teacher agent, who has full observational capabilities in the SDN environment. AdaDoS represents the first application of RL to develop DoS-like attack sequences, capable of adaptively evading both machine learning-based and rule-based DoS-like attack detectors. 

---
# Structural Invariance Matters: Rethinking Graph Rewiring through Graph Metrics 

**Authors**: Alexandre Benoit, Catherine Aitken, Yu He  

**Link**: [PDF](https://arxiv.org/pdf/2510.20556)  

**Abstract**: Graph rewiring has emerged as a key technique to alleviate over-squashing in Graph Neural Networks (GNNs) and Graph Transformers by modifying the graph topology to improve information flow. While effective, rewiring inherently alters the graph's structure, raising the risk of distorting important topology-dependent signals. Yet, despite the growing use of rewiring, little is known about which structural properties must be preserved to ensure both performance gains and structural fidelity. In this work, we provide the first systematic analysis of how rewiring affects a range of graph structural metrics, and how these changes relate to downstream task performance. We study seven diverse rewiring strategies and correlate changes in local and global graph properties with node classification accuracy. Our results reveal a consistent pattern: successful rewiring methods tend to preserve local structure while allowing for flexibility in global connectivity. These findings offer new insights into the design of effective rewiring strategies, bridging the gap between graph theory and practical GNN optimization. 

---
# GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning 

**Authors**: Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20548)  

**Abstract**: Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1. 

---
# The Dog the Cat Chased Stumped the Model: Measuring When Language Models Abandon Structure for Shortcuts 

**Authors**: Sangmitra Madhusudan, Kaige Chen, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2510.20543)  

**Abstract**: When language models correctly parse "The cat that the dog chased meowed," are they analyzing syntax or simply familiar with dogs chasing cats? Despite extensive benchmarking, we lack methods to distinguish structural understanding from semantic pattern matching. We introduce CenterBench, a dataset of 9,720 comprehension questions on center-embedded sentences (like "The cat [that the dog chased] meowed") where relative clauses nest recursively, creating processing demands from simple to deeply nested structures. Each sentence has a syntactically identical but semantically implausible counterpart (e.g., mailmen prescribe medicine, doctors deliver mail) and six comprehension questions testing surface understanding, syntactic dependencies, and causal reasoning. Testing six models reveals that performance gaps between plausible and implausible sentences widen systematically with complexity, with models showing median gaps up to 26.8 percentage points, quantifying when they abandon structural analysis for semantic associations. Notably, semantic plausibility harms performance on questions about resulting actions, where following causal relationships matters more than semantic coherence. Reasoning models improve accuracy but their traces show semantic shortcuts, overthinking, and answer refusal. Unlike models whose plausibility advantage systematically widens with complexity, humans shows variable semantic effects. CenterBench provides the first framework to identify when models shift from structural analysis to pattern matching. 

---
# ARC-Encoder: learning compressed text representations for large language models 

**Authors**: Hippolyte Pilchen, Edouard Grave, Patrick Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2510.20535)  

**Abstract**: Recent techniques such as retrieval-augmented generation or chain-of-thought reasoning have led to longer contexts and increased inference costs. Context compression techniques can reduce these costs, but the most effective approaches require fine-tuning the target model or even modifying its architecture. This can degrade its general abilities when not used for this specific purpose. Here we explore an alternative approach: an encoder that compresses the context into continuous representations which replace token embeddings in decoder LLMs. First, we perform a systematic study of training strategies and architecture choices for the encoder. Our findings led to the design of an Adaptable text Representations Compressor, named ARC-Encoder, which outputs $x$-times fewer continuous representations (typically $x\!\in\!\{4,8\}$) than text tokens. We evaluate ARC-Encoder across a variety of LLM usage scenarios, ranging from in-context learning to context window extension, on both instruct and base decoders. Results show that ARC-Encoder achieves state-of-the-art performance on several benchmarks while improving computational efficiency at inference. Finally, we demonstrate that our models can be adapted to multiple decoders simultaneously, allowing a single encoder to generalize across different decoder LLMs. This makes ARC-Encoder a flexible and efficient solution for portable encoders that work seamlessly with multiple LLMs. We release a training code at this https URL , fine-tuning dataset and pretrained models are available at this https URL . 

---
# Fake-in-Facext: Towards Fine-Grained Explainable DeepFake Analysis 

**Authors**: Lixiong Qin, Yang Zhang, Mei Wang, Jiani Hu, Weihong Deng, Weiran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20531)  

**Abstract**: The advancement of Multimodal Large Language Models (MLLMs) has bridged the gap between vision and language tasks, enabling the implementation of Explainable DeepFake Analysis (XDFA). However, current methods suffer from a lack of fine-grained awareness: the description of artifacts in data annotation is unreliable and coarse-grained, and the models fail to support the output of connections between textual forgery explanations and the visual evidence of artifacts, as well as the input of queries for arbitrary facial regions. As a result, their responses are not sufficiently grounded in Face Visual Context (Facext). To address this limitation, we propose the Fake-in-Facext (FiFa) framework, with contributions focusing on data annotation and model construction. We first define a Facial Image Concept Tree (FICT) to divide facial images into fine-grained regional concepts, thereby obtaining a more reliable data annotation pipeline, FiFa-Annotator, for forgery explanation. Based on this dedicated data annotation, we introduce a novel Artifact-Grounding Explanation (AGE) task, which generates textual forgery explanations interleaved with segmentation masks of manipulated artifacts. We propose a unified multi-task learning architecture, FiFa-MLLM, to simultaneously support abundant multimodal inputs and outputs for fine-grained Explainable DeepFake Analysis. With multiple auxiliary supervision tasks, FiFa-MLLM can outperform strong baselines on the AGE task and achieve SOTA performance on existing XDFA datasets. The code and data will be made open-source at this https URL. 

---
# Metis-HOME: Hybrid Optimized Mixture-of-Experts for Multimodal Reasoning 

**Authors**: Xiaohan Lan, Fanfan Liu, Haibo Qiu, Siqi Yang, Delian Ruan, Peng Shi, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.20519)  

**Abstract**: Inspired by recent advancements in LLM reasoning, the field of multimodal reasoning has seen remarkable progress, achieving significant performance gains on intricate tasks such as mathematical problem-solving. Despite this progress, current multimodal large reasoning models exhibit two key limitations. They tend to employ computationally expensive reasoning even for simple queries, leading to inefficiency. Furthermore, this focus on specialized reasoning often impairs their broader, more general understanding capabilities. In this paper, we propose Metis-HOME: a Hybrid Optimized Mixture-of-Experts framework designed to address this trade-off. Metis-HOME enables a ''Hybrid Thinking'' paradigm by structuring the original dense model into two distinct expert branches: a thinking branch tailored for complex, multi-step reasoning, and a non-thinking branch optimized for rapid, direct inference on tasks like general VQA and OCR. A lightweight, trainable router dynamically allocates queries to the most suitable expert. We instantiate Metis-HOME by adapting the Qwen2.5-VL-7B into an MoE architecture. Comprehensive evaluations reveal that our approach not only substantially enhances complex reasoning abilities but also improves the model's general capabilities, reversing the degradation trend observed in other reasoning-specialized models. Our work establishes a new paradigm for building powerful and versatile MLLMs, effectively resolving the prevalent reasoning-vs-generalization dilemma. 

---
# Hierarchical Sequence Iteration for Heterogeneous Question Answering 

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20505)  

**Abstract**: Retrieval-augmented generation (RAG) remains brittle on multi-step questions and heterogeneous evidence sources, trading accuracy against latency and token/tool budgets. This paper introducesHierarchical Sequence (HSEQ) Iteration for Heterogeneous Question Answering, a unified framework that (i) linearize documents, tables, and knowledge graphs into a reversible hierarchical sequence with lightweight structural tags, and (ii) perform structure-aware iteration to collect just-enough evidence before answer synthesis. A Head Agent provides guidance that leads retrieval, while an Iteration Agent selects and expands HSeq via structure-respecting actions (e.g., parent/child hops, table row/column neighbors, KG relations); Finally the head agent composes canonicalized evidence to genearte the final answer, with an optional refinement loop to resolve detected contradictions. Experiments on HotpotQA (text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1 gains over strong single-pass, multi-hop, and agentic RAG baselines with high efficiency. Besides, HSEQ exhibits three key advantages: (1) a format-agnostic unification that enables a single policy to operate across text, tables, and KGs without per-dataset specialization; (2) guided, budget-aware iteration that reduces unnecessary hops, tool calls, and tokens while preserving accuracy; and (3) evidence canonicalization for reliable QA, improving answers consistency and auditability. 

---
# Steering Evaluation-Aware Language Models To Act Like They Are Deployed 

**Authors**: Tim Tian Hua, Andrew Qin, Samuel Marks, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.20487)  

**Abstract**: Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on documents with factual descriptions of the model (1) using Python type hints during evaluation but not during deployment and (2) recognizing that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. However, this gap can only be observed by removing the evaluation cue. We find that activation steering can suppress evaluation awareness and make the model act like it is deployed even when the cue is present. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed. 

---
# Hurdle-IMDL: An Imbalanced Learning Framework for Infrared Rainfall Retrieval 

**Authors**: Fangjian Zhang, Xiaoyong Zhuge, Wenlan Wang, Haixia Xiao, Yuying Zhu, Siyang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.20486)  

**Abstract**: Artificial intelligence has advanced quantitative remote sensing, yet its effectiveness is constrained by imbalanced label distribution. This imbalance leads conventionally trained models to favor common samples, which in turn degrades retrieval performance for rare ones. Rainfall retrieval exemplifies this issue, with performance particularly compromised for heavy rain. This study proposes Hurdle-Inversion Model Debiasing Learning (IMDL) framework. Following a divide-and-conquer strategy, imbalance in the rain distribution is decomposed into two components: zero inflation, defined by the predominance of non-rain samples; and long tail, defined by the disproportionate abundance of light-rain samples relative to heavy-rain samples. A hurdle model is adopted to handle the zero inflation, while IMDL is proposed to address the long tail by transforming the learning object into an unbiased ideal inverse model. Comprehensive evaluation via statistical metrics and case studies investigating rainy weather in eastern China confirms Hurdle-IMDL's superiority over conventional, cost-sensitive, generative, and multi-task learning methods. Its key advancements include effective mitigation of systematic underestimation and a marked improvement in the retrieval of heavy-to-extreme rain. IMDL offers a generalizable approach for addressing imbalance in distributions of environmental variables, enabling enhanced retrieval of rare yet high-impact events. 

---
# RECALL: REpresentation-aligned Catastrophic-forgetting ALLeviation via Hierarchical Model Merging 

**Authors**: Bowen Wang, Haiyuan Wan, Liwen Shi, Chen Yang, Peng He, Yue Ma, Haochen Han, Wenhao Li, Tiao Tan, Yongjian Li, Fangming Liu, Yifan Gong, Sheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20479)  

**Abstract**: We unveil that internal representations in large language models (LLMs) serve as reliable proxies of learned knowledge, and propose RECALL, a novel representation-aware model merging framework for continual learning without access to historical data. RECALL computes inter-model similarity from layer-wise hidden representations over clustered typical samples, and performs adaptive, hierarchical parameter fusion to align knowledge across models. This design enables the preservation of domain-general features in shallow layers while allowing task-specific adaptation in deeper layers. Unlike prior methods that require task labels or incur performance trade-offs, RECALL achieves seamless multi-domain integration and strong resistance to catastrophic forgetting. Extensive experiments across five NLP tasks and multiple continual learning scenarios show that RECALL outperforms baselines in both knowledge retention and generalization, providing a scalable and data-free solution for evolving LLMs. 

---
# Structures generated in a multiagent system performing information fusion in peer-to-peer resource-constrained networks 

**Authors**: Horacio Paggi, Juan A. Lara, Javier Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2510.20469)  

**Abstract**: There has recently been a major advance with respect to how information fusion is performed. Information fusion has gone from being conceived as a purely hierarchical procedure, as is the case of traditional military applications, to now being regarded collaboratively, as holonic fusion, which is better suited for civil applications and edge organizations. The above paradigm shift is being boosted as information fusion gains ground in different non-military areas, and human-computer and machine-machine communications, where holarchies, which are more flexible structures than ordinary, static hierarchies, become more widespread. This paper focuses on showing how holonic structures tend to be generated when there are constraints on resources (energy, available messages, time, etc.) for interactions based on a set of fully intercommunicating elements (peers) whose components fuse information as a means of optimizing the impact of vagueness and uncertainty present message exchanges. Holon formation is studied generically based on a multiagent system model, and an example of its possible operation is shown. Holonic structures have a series of advantages, such as adaptability, to sudden changes in the environment or its composition, are somewhat autonomous and are capable of cooperating in order to achieve a common goal. This can be useful when the shortage of resources prevents communications or when the system components start to fail. 

---
# Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models 

**Authors**: Tomáš Souček, Sylvestre-Alvise Rebuffi, Pierre Fernandez, Nikola Jovanović, Hady Elsahar, Valeriu Lacatusu, Tuan Tran, Alexandre Mourachko  

**Link**: [PDF](https://arxiv.org/pdf/2510.20468)  

**Abstract**: Recent years have seen a surge in interest in digital content watermarking techniques, driven by the proliferation of generative models and increased legal pressure. With an ever-growing percentage of AI-generated content available online, watermarking plays an increasingly important role in ensuring content authenticity and attribution at scale. There have been many works assessing the robustness of watermarking to removal attacks, yet, watermark forging, the scenario when a watermark is stolen from genuine content and applied to malicious content, remains underexplored. In this work, we investigate watermark forging in the context of widely used post-hoc image watermarking. Our contributions are as follows. First, we introduce a preference model to assess whether an image is watermarked. The model is trained using a ranking loss on purely procedurally generated images without any need for real watermarks. Second, we demonstrate the model's capability to remove and forge watermarks by optimizing the input image through backpropagation. This technique requires only a single watermarked image and works without knowledge of the watermarking model, making our attack much simpler and more practical than attacks introduced in related work. Third, we evaluate our proposed method on a variety of post-hoc image watermarking models, demonstrating that our approach can effectively forge watermarks, questioning the security of current watermarking approaches. Our code and further resources are publicly available. 

---
# Symbolic Regression and Differentiable Fits in Beyond the Standard Model Physics 

**Authors**: Shehu AbdusSalam, Steven Abel, Deaglan Bartlett, Miguel Crispim Romão  

**Link**: [PDF](https://arxiv.org/pdf/2510.20453)  

**Abstract**: We demonstrate the efficacy of symbolic regression (SR) to probe models of particle physics Beyond the Standard Model (BSM), by considering the so-called Constrained Minimal Supersymmetric Standard Model (CMSSM). Like many incarnations of BSM physics this model has a number (four) of arbitrary parameters, which determine the experimental signals, and cosmological observables such as the dark matter relic density. We show that analysis of the phenomenology can be greatly accelerated by using symbolic expressions derived for the observables in terms of the input parameters. Here we focus on the Higgs mass, the cold dark matter relic density, and the contribution to the anomalous magnetic moment of the muon. We find that SR can produce remarkably accurate expressions. Using them we make global fits to derive the posterior probability densities of the CMSSM input parameters which are in good agreement with those performed using conventional methods. Moreover, we demonstrate a major advantage of SR which is the ability to make fits using differentiable methods rather than sampling methods. We also compare the method with neural network (NN) regression. SR produces more globally robust results, while NNs require data that is focussed on the promising regions in order to be equally performant. 

---
# MolBridge: Atom-Level Joint Graph Refinement for Robust Drug-Drug Interaction Event Prediction 

**Authors**: Xuan Lin, Aocheng Ding, Tengfei Ma, Hua Liang, Zhe Quan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20448)  

**Abstract**: Drug combinations offer therapeutic benefits but also carry the risk of adverse drug-drug interactions (DDIs), especially under complex molecular structures. Accurate DDI event prediction requires capturing fine-grained inter-drug relationships, which are critical for modeling metabolic mechanisms such as enzyme-mediated competition. However, existing approaches typically rely on isolated drug representations and fail to explicitly model atom-level cross-molecular interactions, limiting their effectiveness across diverse molecular complexities and DDI type distributions. To address these limitations, we propose MolBridge, a novel atom-level joint graph refinement framework for robust DDI event prediction. MolBridge constructs a joint graph that integrates atomic structures of drug pairs, enabling direct modeling of inter-drug associations. A central challenge in such joint graph settings is the potential loss of information caused by over-smoothing when modeling long-range atomic dependencies. To overcome this, we introduce a structure consistency module that iteratively refines node features while preserving the global structural context. This joint design allows MolBridge to effectively learn both local and global interaction outperforms state-of-the-art baselines, achieving superior performance across long-tail and inductive scenarios. patterns, yielding robust representations across both frequent and rare DDI types. Extensive experiments on two benchmark datasets show that MolBridge consistently. These results demonstrate the advantages of fine-grained graph refinement in improving the accuracy, robustness, and mechanistic interpretability of DDI event this http URL work contributes to Web Mining and Content Analysis by developing graph-based methods for mining and analyzing drug-drug interaction networks. 

---
# UniSE: A Unified Framework for Decoder-only Autoregressive LM-based Speech Enhancement 

**Authors**: Haoyin Yan, Chengwei Liu, Shaofei Xue, Xiaotao Liang, Zheng Xue  

**Link**: [PDF](https://arxiv.org/pdf/2510.20441)  

**Abstract**: The development of neural audio codecs (NACs) has largely promoted applications of language models (LMs) to speech processing and understanding. However, there lacks the verification on the effectiveness of autoregressive (AR) LMbased models in unifying different sub-tasks of speech enhancement (SE). In this work, we propose UniSE, a unified decoder-only LM-based framework to handle different SE tasks including speech restoration, target speaker extraction and speech separation. It takes input speech features as conditions and generates discrete tokens of the target speech using AR modeling, which facilitates a compatibility between distinct learning patterns of multiple tasks. Experiments on several benchmarks indicate the proposed UniSE can achieve competitive performance compared to discriminative and generative baselines, showing the capacity of LMs in unifying SE tasks. The demo page is available here: this https URL. 

---
# Dynamic Weight Adjustment for Knowledge Distillation: Leveraging Vision Transformer for High-Accuracy Lung Cancer Detection and Real-Time Deployment 

**Authors**: Saif Ur Rehman Khan, Muhammad Nabeel Asim, Sebastian Vollmer, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2510.20438)  

**Abstract**: This paper presents the FuzzyDistillViT-MobileNet model, a novel approach for lung cancer (LC) classification, leveraging dynamic fuzzy logic-driven knowledge distillation (KD) to address uncertainty and complexity in disease diagnosis. Unlike traditional models that rely on static KD with fixed weights, our method dynamically adjusts the distillation weight using fuzzy logic, enabling the student model to focus on high-confidence regions while reducing attention to ambiguous areas. This dynamic adjustment improves the model ability to handle varying uncertainty levels across different regions of LC images. We employ the Vision Transformer (ViT-B32) as the instructor model, which effectively transfers knowledge to the student model, MobileNet, enhancing the student generalization capabilities. The training process is further optimized using a dynamic wait adjustment mechanism that adapts the training procedure for improved convergence and performance. To enhance image quality, we introduce pixel-level image fusion improvement techniques such as Gamma correction and Histogram Equalization. The processed images (Pix1 and Pix2) are fused using a wavelet-based fusion method to improve image resolution and feature preservation. This fusion method uses the wavedec2 function to standardize images to a 224x224 resolution, decompose them into multi-scale frequency components, and recursively average coefficients at each level for better feature representation. To address computational efficiency, Genetic Algorithm (GA) is used to select the most suitable pre-trained student model from a pool of 12 candidates, balancing model performance with computational cost. The model is evaluated on two datasets, including LC25000 histopathological images (99.16% accuracy) and IQOTH/NCCD CT-scan images (99.54% accuracy), demonstrating robustness across different imaging domains. 

---
# Balancing Specialization and Centralization: A Multi-Agent Reinforcement Learning Benchmark for Sequential Industrial Control 

**Authors**: Tom Maus, Asma Atamna, Tobias Glasmachers  

**Link**: [PDF](https://arxiv.org/pdf/2510.20408)  

**Abstract**: Autonomous control of multi-stage industrial processes requires both local specialization and global coordination. Reinforcement learning (RL) offers a promising approach, but its industrial adoption remains limited due to challenges such as reward design, modularity, and action space management. Many academic benchmarks differ markedly from industrial control problems, limiting their transferability to real-world applications. This study introduces an enhanced industry-inspired benchmark environment that combines tasks from two existing benchmarks, SortingEnv and ContainerGym, into a sequential recycling scenario with sorting and pressing operations. We evaluate two control strategies: a modular architecture with specialized agents and a monolithic agent governing the full system, while also analyzing the impact of action masking. Our experiments show that without action masking, agents struggle to learn effective policies, with the modular architecture performing better. When action masking is applied, both architectures improve substantially, and the performance gap narrows considerably. These results highlight the decisive role of action space constraints and suggest that the advantages of specialization diminish as action complexity is reduced. The proposed benchmark thus provides a valuable testbed for exploring practical and robust multi-agent RL solutions in industrial automation, while contributing to the ongoing debate on centralization versus specialization. 

---
# FLAS: a combination of proactive and reactive auto-scaling architecture for distributed services 

**Authors**: Víctor Rampérez, Javier Soriano, David Lizcano, Juan A. Lara  

**Link**: [PDF](https://arxiv.org/pdf/2510.20388)  

**Abstract**: Cloud computing has established itself as the support for the vast majority of emerging technologies, mainly due to the characteristic of elasticity it offers. Auto-scalers are the systems that enable this elasticity by acquiring and releasing resources on demand to ensure an agreed service level. In this article we present FLAS (Forecasted Load Auto-Scaling), an auto-scaler for distributed services that combines the advantages of proactive and reactive approaches according to the situation to decide the optimal scaling actions in every moment. The main novelties introduced by FLAS are (i) a predictive model of the high-level metrics trend which allows to anticipate changes in the relevant SLA parameters (e.g. performance metrics such as response time or throughput) and (ii) a reactive contingency system based on the estimation of high-level metrics from resource use metrics, reducing the necessary instrumentation (less invasive) and allowing it to be adapted agnostically to different applications. We provide a FLAS implementation for the use case of a content-based publish-subscribe middleware (E-SilboPS) that is the cornerstone of an event-driven architecture. To the best of our knowledge, this is the first auto-scaling system for content-based publish-subscribe distributed systems (although it is generic enough to fit any distributed service). Through an evaluation based on several test cases recreating not only the expected contexts of use, but also the worst possible scenarios (following the Boundary-Value Analysis or BVA test methodology), we have validated our approach and demonstrated the effectiveness of our solution by ensuring compliance with performance requirements over 99% of the time. 

---
# Relative-Based Scaling Law for Neural Language Models 

**Authors**: Baoqing Yue, Jinyuan Zhou, Zixi Wei, Jingtao Zhan, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20387)  

**Abstract**: Scaling laws aim to accurately predict model performance across different scales. Existing scaling-law studies almost exclusively rely on cross-entropy as the evaluation metric. However, cross-entropy provides only a partial view of performance: it measures the absolute probability assigned to the correct token, but ignores the relative ordering between correct and incorrect tokens. Yet, relative ordering is crucial for language models, such as in greedy-sampling scenario. To address this limitation, we investigate scaling from the perspective of relative ordering. We first propose the Relative-Based Probability (RBP) metric, which quantifies the probability that the correct token is ranked among the top predictions. Building on this metric, we establish the Relative-Based Scaling Law, which characterizes how RBP improves with increasing model size. Through extensive experiments on four datasets and four model families spanning five orders of magnitude, we demonstrate the robustness and accuracy of this law. Finally, we illustrate the broad application of this law with two examples, namely providing a deeper explanation of emergence phenomena and facilitating finding fundamental theories of scaling laws. In summary, the Relative-Based Scaling Law complements the cross-entropy perspective and contributes to a more complete understanding of scaling large language models. Thus, it offers valuable insights for both practical development and theoretical exploration. 

---
# VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation 

**Authors**: Son T. Luu, Trung Vo, Hiep Nguyen, Khanh Quoc Tran, Kiet Van Nguyen, Vu Tran, Ngan Luu-Thuy Nguyen, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20381)  

**Abstract**: This paper presents the VLSP 2025 MLQA-TSR - the multimodal legal question answering on traffic sign regulation shared task at VLSP 2025. VLSP 2025 MLQA-TSR comprises two subtasks: multimodal legal retrieval and multimodal question answering. The goal is to advance research on Vietnamese multimodal legal text processing and to provide a benchmark dataset for building and evaluating intelligent systems in multimodal legal domains, with a focus on traffic sign regulation in Vietnam. The best-reported results on VLSP 2025 MLQA-TSR are an F2 score of 64.55% for multimodal legal retrieval and an accuracy of 86.30% for multimodal question answering. 

---
# The Impact of Negated Text on Hallucination with Large Language Models 

**Authors**: Jaehyung Seo, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20375)  

**Abstract**: Recent studies on hallucination in large language models (LLMs) have been actively progressing in natural language processing. However, the impact of negated text on hallucination with LLMs remains largely unexplored. In this paper, we set three important yet unanswered research questions and aim to address them. To derive the answers, we investigate whether LLMs can recognize contextual shifts caused by negation and still reliably distinguish hallucinations comparable to affirmative cases. We also design the NegHalu dataset by reconstructing existing hallucination detection datasets with negated expressions. Our experiments demonstrate that LLMs struggle to detect hallucinations in negated text effectively, often producing logically inconsistent or unfaithful judgments. Moreover, we trace the internal state of LLMs as they process negated inputs at the token level and reveal the challenges of mitigating their unintended effects. 

---
# Evaluating Latent Knowledge of Public Tabular Datasets in Large Language Models 

**Authors**: Matteo Silvestri, Flavio Giorgi, Fabrizio Silvestri, Gabriele Tolomei  

**Link**: [PDF](https://arxiv.org/pdf/2510.20351)  

**Abstract**: Large Language Models (LLMs) are increasingly evaluated on their ability to reason over structured data, yet such assessments often overlook a crucial confound: dataset contamination. In this work, we investigate whether LLMs exhibit prior knowledge of widely used tabular benchmarks such as Adult Income, Titanic, and others. Through a series of controlled probing experiments, we reveal that contamination effects emerge exclusively for datasets containing strong semantic cues-for instance, meaningful column names or interpretable value categories. In contrast, when such cues are removed or randomized, performance sharply declines to near-random levels. These findings suggest that LLMs' apparent competence on tabular reasoning tasks may, in part, reflect memorization of publicly available datasets rather than genuine generalization. We discuss implications for evaluation protocols and propose strategies to disentangle semantic leakage from authentic reasoning ability in future LLM assessments. 

---
# What do AI-Generated Images Want? 

**Authors**: Amanda Wasielewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.20350)  

**Abstract**: W.J.T. Mitchell's influential essay 'What do pictures want?' shifts the theoretical focus away from the interpretative act of understanding pictures and from the motivations of the humans who create them to the possibility that the picture itself is an entity with agency and wants. In this article, I reframe Mitchell's question in light of contemporary AI image generation tools to ask: what do AI-generated images want? Drawing from art historical discourse on the nature of abstraction, I argue that AI-generated images want specificity and concreteness because they are fundamentally abstract. Multimodal text-to-image models, which are the primary subject of this article, are based on the premise that text and image are interchangeable or exchangeable tokens and that there is a commensurability between them, at least as represented mathematically in data. The user pipeline that sees textual input become visual output, however, obscures this representational regress and makes it seem like one form transforms into the other -- as if by magic. 

---
# Teaching Language Models to Reason with Tools 

**Authors**: Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20342)  

**Abstract**: Large reasoning models (LRMs) like OpenAI-o1 have shown impressive capabilities in natural language reasoning. However, these models frequently demonstrate inefficiencies or inaccuracies when tackling complex mathematical operations. While integrating computational tools such as Code Interpreters (CIs) offers a promising solution, it introduces a critical challenge: a conflict between the model's internal, probabilistic reasoning and the external, deterministic knowledge provided by the CI, which often leads models to unproductive deliberation. To overcome this, we introduce CoRT (Code-Optimized Reasoning Training), a post-training framework designed to teach LRMs to effectively utilize CIs. We propose \emph{Hint-Engineering}, a new data synthesis strategy that strategically injects diverse hints at optimal points within reasoning paths. This approach generates high-quality, code-integrated reasoning data specifically tailored to optimize LRM-CI interaction. Using this method, we have synthesized 30 high-quality samples to post-train models ranging from 1.5B to 32B parameters through supervised fine-tuning. CoRT further refines the multi-round interleaving of external CI usage and internal thinking by employing rejection sampling and reinforcement learning. Our experimental evaluations demonstrate CoRT's effectiveness, yielding absolute improvements of 4\% and 8\% on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B, respectively, across five challenging mathematical reasoning datasets. Moreover, CoRT significantly enhances efficiency, reducing token usage by approximately 30\% for the 32B model and 50\% for the 1.5B model compared to pure natural language reasoning baselines. The models and code are available at: this https URL. 

---
# Multi-Task Deep Learning for Surface Metrology 

**Authors**: D. Kucharski, A. Gaska, T. Kowaluk, K. Stepien, M. Repalska, B. Gapinski, M. Wieczorowski, M. Nawotka, P. Sobecki, P. Sosinowski, J. Tomasik, A. Wojtowicz  

**Link**: [PDF](https://arxiv.org/pdf/2510.20339)  

**Abstract**: A reproducible deep learning framework is presented for surface metrology to predict surface texture parameters together with their reported standard uncertainties. Using a multi-instrument dataset spanning tactile and optical systems, measurement system type classification is addressed alongside coordinated regression of Ra, Rz, RONt and their uncertainty targets (Ra_uncert, Rz_uncert, RONt_uncert). Uncertainty is modelled via quantile and heteroscedastic heads with post-hoc conformal calibration to yield calibrated intervals. On a held-out set, high fidelity was achieved by single-target regressors (R2: Ra 0.9824, Rz 0.9847, RONt 0.9918), with two uncertainty targets also well modelled (Ra_uncert 0.9899, Rz_uncert 0.9955); RONt_uncert remained difficult (R2 0.4934). The classifier reached 92.85% accuracy and probability calibration was essentially unchanged after temperature scaling (ECE 0.00504 -> 0.00503 on the test split). Negative transfer was observed for naive multi-output trunks, with single-target models performing better. These results provide calibrated predictions suitable to inform instrument selection and acceptance decisions in metrological workflows. 

---
# GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments? 

**Authors**: Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20333)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents. 

---
# MemER: Scaling Up Memory for Robot Control via Experience Retrieval 

**Authors**: Ajay Sridhar, Jennifer Pan, Satvik Sharma, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2510.20328)  

**Abstract**: Humans routinely rely on memory to perform tasks, yet most robot policies lack this capability; our goal is to endow robot policies with the same ability. Naively conditioning on long observation histories is computationally expensive and brittle under covariate shift, while indiscriminate subsampling of history leads to irrelevant or redundant information. We propose a hierarchical policy framework, where the high-level policy is trained to select and track previous relevant keyframes from its experience. The high-level policy uses selected keyframes and the most recent frames when generating text instructions for a low-level policy to execute. This design is compatible with existing vision-language-action (VLA) models and enables the system to efficiently reason over long-horizon dependencies. In our experiments, we finetune Qwen2.5-VL-7B-Instruct and $\pi_{0.5}$ as the high-level and low-level policies respectively, using demonstrations supplemented with minimal language annotations. Our approach, MemER, outperforms prior methods on three real-world long-horizon robotic manipulation tasks that require minutes of memory. Videos and code can be found at this https URL. 

---
# LEGO: A Lightweight and Efficient Multiple-Attribute Unlearning Framework for Recommender Systems 

**Authors**: Fengyuan Yu, Yuyuan Li, Xiaohua Feng, Junjie Fang, Tao Wang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20327)  

**Abstract**: With the growing demand for safeguarding sensitive user information in recommender systems, recommendation attribute unlearning is receiving increasing attention. Existing studies predominantly focus on single-attribute unlearning. However, privacy protection requirements in the real world often involve multiple sensitive attributes and are dynamic. Existing single-attribute unlearning methods cannot meet these real-world requirements due to i) CH1: the inability to handle multiple unlearning requests simultaneously, and ii) CH2: the lack of efficient adaptability to dynamic unlearning needs. To address these challenges, we propose LEGO, a lightweight and efficient multiple-attribute unlearning framework. Specifically, we divide the multiple-attribute unlearning process into two steps: i) Embedding Calibration removes information related to a specific attribute from user embedding, and ii) Flexible Combination combines these embeddings into a single embedding, protecting all sensitive attributes. We frame the unlearning process as a mutual information minimization problem, providing LEGO a theoretical guarantee of simultaneous unlearning, thereby addressing CH1. With the two-step framework, where Embedding Calibration can be performed in parallel and Flexible Combination is flexible and efficient, we address CH2. Extensive experiments on three real-world datasets across three representative recommendation models demonstrate the effectiveness and efficiency of our proposed framework. Our code and appendix are available at this https URL. 

---
# Enhancing Security in Deep Reinforcement Learning: A Comprehensive Survey on Adversarial Attacks and Defenses 

**Authors**: Wu Yichao, Wang Yirui, Ding Panpan, Wang Hailong, Zhu Bingqian, Liu Chun  

**Link**: [PDF](https://arxiv.org/pdf/2510.20314)  

**Abstract**: With the wide application of deep reinforcement learning (DRL) techniques in complex fields such as autonomous driving, intelligent manufacturing, and smart healthcare, how to improve its security and robustness in dynamic and changeable environments has become a core issue in current research. Especially in the face of adversarial attacks, DRL may suffer serious performance degradation or even make potentially dangerous decisions, so it is crucial to ensure their stability in security-sensitive scenarios. In this paper, we first introduce the basic framework of DRL and analyze the main security challenges faced in complex and changing environments. In addition, this paper proposes an adversarial attack classification framework based on perturbation type and attack target and reviews the mainstream adversarial attack methods against DRL in detail, including various attack methods such as perturbation state space, action space, reward function and model space. To effectively counter the attacks, this paper systematically summarizes various current robustness training strategies, including adversarial training, competitive training, robust learning, adversarial detection, defense distillation and other related defense techniques, we also discuss the advantages and shortcomings of these methods in improving the robustness of DRL. Finally, this paper looks into the future research direction of DRL in adversarial environments, emphasizing the research needs in terms of improving generalization, reducing computational complexity, and enhancing scalability and explainability, aiming to provide valuable references and directions for researchers. 

---
# DB-FGA-Net: Dual Backbone Frequency Gated Attention Network for Multi-Class Classification with Grad-CAM Interpretability 

**Authors**: Saraf Anzum Shreya, MD. Abu Ismail Siddique, Sharaf Tasnim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20299)  

**Abstract**: Brain tumors are a challenging problem in neuro-oncology, where early and precise diagnosis is important for successful treatment. Deep learning-based brain tumor classification methods often rely on heavy data augmentation which can limit generalization and trust in clinical applications. In this paper, we propose a double-backbone network integrating VGG16 and Xception with a Frequency-Gated Attention (FGA) Block to capture complementary local and global features. Unlike previous studies, our model achieves state-of-the-art performance without augmentation which demonstrates robustness to variably sized and distributed datasets. For further transparency, Grad-CAM is integrated to visualize the tumor regions based on which the model is giving prediction, bridging the gap between model prediction and clinical interpretability. The proposed framework achieves 99.24\% accuracy on the 7K-DS dataset for the 4-class setting, along with 98.68\% and 99.85\% in the 3-class and 2-class settings, respectively. On the independent 3K-DS dataset, the model generalizes with 95.77\% accuracy, outperforming baseline and state-of-the-art methods. To further support clinical usability, we developed a graphical user interface (GUI) that provides real-time classification and Grad-CAM-based tumor localization. These findings suggest that augmentation-free, interpretable, and deployable deep learning models such as DB-FGA-Net hold strong potential for reliable clinical translation in brain tumor diagnosis. 

---
# RAG-Stack: Co-Optimizing RAG Quality and Performance From the Vector Database Perspective 

**Authors**: Wenqi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20296)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as one of the most prominent applications of vector databases. By integrating documents retrieved from a database into the prompt of a large language model (LLM), RAG enables more reliable and informative content generation. While there has been extensive research on vector databases, many open research problems remain once they are considered in the wider context of end-to-end RAG pipelines. One practical yet challenging problem is how to jointly optimize both system performance and generation quality in RAG, which is significantly more complex than it appears due to the numerous knobs on both the algorithmic side (spanning models and databases) and the systems side (from software to hardware). In this paper, we present RAG-Stack, a three-pillar blueprint for quality-performance co-optimization in RAG systems. RAG-Stack comprises: (1) RAG-IR, an intermediate representation that serves as an abstraction layer to decouple quality and performance aspects; (2) RAG-CM, a cost model for estimating system performance given an RAG-IR; and (3) RAG-PE, a plan exploration algorithm that searches for high-quality, high-performance RAG configurations. We believe this three-pillar blueprint will become the de facto paradigm for RAG quality-performance co-optimization in the years to come. 

---
# A Parameter-Efficient Mixture-of-Experts Framework for Cross-Modal Geo-Localization 

**Authors**: LinFeng Li, Jian Zhao, Zepeng Yang, Yuhang Song, Bojun Lin, Tianle Zhang, Yuchen Yuan, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.20291)  

**Abstract**: We present a winning solution to RoboSense 2025 Track 4: Cross-Modal Drone Navigation. The task retrieves the most relevant geo-referenced image from a large multi-platform corpus (satellite/drone/ground) given a natural-language query. Two obstacles are severe inter-platform heterogeneity and a domain gap between generic training descriptions and platform-specific test queries. We mitigate these with a domain-aligned preprocessing pipeline and a Mixture-of-Experts (MoE) framework: (i) platform-wise partitioning, satellite augmentation, and removal of orientation words; (ii) an LLM-based caption refinement pipeline to align textual semantics with the distinct visual characteristics of each platform. Using BGE-M3 (text) and EVA-CLIP (image), we train three platform experts using a progressive two-stage, hard-negative mining strategy to enhance discriminative power, and fuse their scores at inference. The system tops the official leaderboard, demonstrating robust cross-modal geo-localization under heterogeneous viewpoints. 

---
# Breakdance Video classification in the age of Generative AI 

**Authors**: Sauptik Dhar, Naveen Ramakrishnan, Michelle Munson  

**Link**: [PDF](https://arxiv.org/pdf/2510.20287)  

**Abstract**: Large Vision Language models have seen huge application in several sports use-cases recently. Most of these works have been targeted towards a limited subset of popular sports like soccer, cricket, basketball etc; focusing on generative tasks like visual question answering, highlight generation. This work analyzes the applicability of the modern video foundation models (both encoder and decoder) for a very niche but hugely popular dance sports - breakdance. Our results show that Video Encoder models continue to outperform state-of-the-art Video Language Models for prediction tasks. We provide insights on how to choose the encoder model and provide a thorough analysis into the workings of a finetuned decoder model for breakdance video classification. 

---
# UI-Ins: Enhancing GUI Grounding with Multi-Perspective Instruction-as-Reasoning 

**Authors**: Liangyu Chen, Hanzhang Zhou, Chenglin Cai, Jianan Zhang, Panrong Tong, Quyu Kong, Xu Zhang, Chen Liu, Yuqi Liu, Wenxuan Wang, Yue Wang, Qin Jin, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20286)  

**Abstract**: GUI grounding, which maps natural-language instructions to actionable UI elements, is a core capability of GUI agents. Prior works largely treats instructions as a static proxy for user intent, overlooking the impact of instruction diversity and quality on grounding performance. Through a careful investigation of existing grounding datasets, we find a 23.3% flaw rate in their instructions and show that inference-time exploitation of instruction diversity yields up to a substantial 76% relative performance improvement. In this paper, we introduce the Instruction-as-Reasoning paradigm, treating instructions as dynamic analytical pathways that offer distinct perspectives and enabling the model to select the most effective pathway during reasoning. To achieve this, we propose a two-stage training framework: supervised fine-tuning (SFT) on synthesized, diverse instructions to instill multi-perspective reasoning, followed by reinforcement learning (RL) to optimize pathway selection and composition. Our resulting models, UI-Ins-7B and UI-Ins-32B, achieve state-of-the-art results on five challenging grounding benchmarks and exhibit emergent reasoning, selectively composing and synthesizing novel instruction pathways at inference. In particular, UI-Ins-32B attains the best grounding accuracy, scoring 87.3% on UI-I2E-Bench, 57.0% on ScreenSpot-Pro, and 84.9% on MMBench-GUI L2. Furthermore, our model demonstrates strong agentic potential, achieving a 74.1% success rate on AndroidWorld using UI-Ins-7B as the executor. Our in-depth analysis reveals additional insights such as how reasoning can be formulated to enhance rather than hinder grounding performance, and how our method mitigates policy collapse in the SFT+RL framework. All code and model checkpoints will be publicly released in this https URL. 

---
# Context-level Language Modeling by Learning Predictive Context Embeddings 

**Authors**: Beiya Dai, Yuliang Liu, Daozheng Xue, Qipeng Guo, Kai Chen, Xinbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20280)  

**Abstract**: Next-token prediction (NTP) is the cornerstone of modern large language models (LLMs) pretraining, driving their unprecedented capabilities in text generation, reasoning, and instruction following. However, the token-level prediction limits the model's capacity to capture higher-level semantic structures and long-range contextual relationships. To overcome this limitation, we introduce \textbf{ContextLM}, a framework that augments standard pretraining with an inherent \textbf{next-context prediction} objective. This mechanism trains the model to learn predictive representations of multi-token contexts, leveraging error signals derived from future token chunks. Crucially, ContextLM achieves this enhancement while remaining fully compatible with the standard autoregressive, token-by-token evaluation paradigm (e.g., perplexity). Extensive experiments on the GPT2 and Pythia model families, scaled up to $1.5$B parameters, show that ContextLM delivers consistent improvements in both perplexity and downstream task performance. Our analysis indicates that next-context prediction provides a scalable and efficient pathway to stronger language modeling, yielding better long-range coherence and more effective attention allocation with minimal computational overhead. 

---
# Limits of PRM-Guided Tree Search for Mathematical Reasoning with LLMs 

**Authors**: Tristan Cinquin, Geoff Pleiss, Agustinus Kristiadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20272)  

**Abstract**: While chain-of-thought prompting with Best-of-N (BoN) selection has become popular for mathematical reasoning in large language models (LLMs), its linear structure fails to capture the branching and exploratory nature of complex problem-solving. In this work, we propose an adaptive algorithm to maximize process reward model (PRM) scores over the intractable action space, and investigate whether PRM-guided tree search can improve mathematical reasoning by exploring multiple partial solution paths. Across $23$ diverse mathematical problems using Qwen2.5-Math-7B-Instruct with its associated PRM as a case study, we find that: (1) PRM-guided tree search shows no statistically significant improvements over BoN despite higher costs, (2) Monte Carlo tree search and beam search outperform other PRM-guided tree search methods, (3) PRMs poorly approximate state values and their reliability degrades with reasoning depth, and (4) PRMs generalize poorly out of distribution. This underperformance stems from tree search's greater reliance on unreliable PRM scores, suggesting different reward modeling is necessary before tree search can effectively enhance mathematical reasoning in LLMs. 

---
# Towards AI Agents for Course Instruction in Higher Education: Early Experiences from the Field 

**Authors**: Yogesh Simmhan, Varad Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2510.20255)  

**Abstract**: This article presents early findings from designing, deploying and evaluating an AI-based educational agent deployed as the primary instructor in a graduate-level Cloud Computing course at IISc. We detail the design of a Large Language Model (LLM)-driven Instructor Agent, and introduce a pedagogical framework that integrates the Instructor Agent into the course workflow for actively interacting with the students for content delivery, supplemented by the human instructor to offer the course structure and undertake question--answer sessions. We also propose an analytical framework that evaluates the Agent--Student interaction transcripts using interpretable engagement metrics of topic coverage, topic depth and turn-level elaboration. We report early experiences on how students interact with the Agent to explore concepts, clarify doubts and sustain inquiry-driven dialogue during live classroom sessions. We also report preliminary analysis on our evaluation metrics applied across two successive instructional modules that reveals patterns of engagement evolution, transitioning from broad conceptual exploration to deeper, focused inquiry. These demonstrate how structured integration of conversational AI agents can foster reflective learning, offer a reproducible methodology for studying engagement in authentic classroom settings, and support scalable, high-quality higher education. 

---
# What Does It Take to Build a Performant Selective Classifier? 

**Authors**: Stephan Rabanser, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2510.20242)  

**Abstract**: Selective classifiers improve model reliability by abstaining on inputs the model deems uncertain. However, few practical approaches achieve the gold-standard performance of a perfect-ordering oracle that accepts examples exactly in order of correctness. Our work formalizes this shortfall as the selective-classification gap and present the first finite-sample decomposition of this gap to five distinct sources of looseness: Bayes noise, approximation error, ranking error, statistical noise, and implementation- or shift-induced slack. Crucially, our analysis reveals that monotone post-hoc calibration -- often believed to strengthen selective classifiers -- has limited impact on closing this gap, since it rarely alters the model's underlying score ranking. Bridging the gap therefore requires scoring mechanisms that can effectively reorder predictions rather than merely rescale them. We validate our decomposition on synthetic two-moons data and on real-world vision and language benchmarks, isolating each error component through controlled experiments. Our results confirm that (i) Bayes noise and limited model capacity can account for substantial gaps, (ii) only richer, feature-aware calibrators meaningfully improve score ordering, and (iii) data shift introduces a separate slack that demands distributionally robust training. Together, our decomposition yields a quantitative error budget as well as actionable design guidelines that practitioners can use to build selective classifiers which approximate ideal oracle behavior more closely. 

---
# Tri-Modal Severity Fused Diagnosis across Depression and Post-traumatic Stress Disorders 

**Authors**: Filippo Cenacchi, Deborah Richards, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20239)  

**Abstract**: Depression and post traumatic stress disorder (PTSD) often co-occur with connected symptoms, complicating automated assessment, which is often binary and disorder specific. Clinically useful diagnosis needs severity aware cross disorder estimates and decision support explanations. Our unified tri modal affective severity framework synchronizes and fuses interview text with sentence level transformer embeddings, audio with log Mel statistics with deltas, and facial signals with action units, gaze, head and pose descriptors to output graded severities for diagnosing both depression (PHQ-8; 5 classes) and PTSD (3 classes). Standardized features are fused via a calibrated late fusion classifier, yielding per disorder probabilities and feature-level attributions. This severity aware tri-modal affective fusion approach is demoed on multi disorder concurrent depression and PTSD assessment. Stratified cross validation on DAIC derived corpora outperforms unimodal/ablation baselines. The fused model matches the strongest unimodal baseline on accuracy and weighted F1, while improving decision curve utility and robustness under noisy or missing modalities. For PTSD specifically, fusion reduces regression error and improves class concordance. Errors cluster between adjacent severities; extreme classes are identified reliably. Ablations show text contributes most to depression severity, audio and facial cues are critical for PTSD, whereas attributions align with linguistic and behavioral markers. Our approach offers reproducible evaluation and clinician in the loop support for affective clinical decision making. 

---
# Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach 

**Authors**: Woohyeon Byeon, Giseung Park, Jongseong Chae, Amir Leshem, Youngchul Sung  

**Link**: [PDF](https://arxiv.org/pdf/2510.20235)  

**Abstract**: In this paper, we propose a provably convergent and practical framework for multi-objective reinforcement learning with max-min criterion. From a game-theoretic perspective, we reformulate max-min multi-objective reinforcement learning as a two-player zero-sum regularized continuous game and introduce an efficient algorithm based on mirror descent. Our approach simplifies the policy update while ensuring global last-iterate convergence. We provide a comprehensive theoretical analysis on our algorithm, including iteration complexity under both exact and approximate policy evaluations, as well as sample complexity bounds. To further enhance performance, we modify the proposed algorithm with adaptive regularization. Our experiments demonstrate the convergence behavior of the proposed algorithm in tabular settings, and our implementation for deep reinforcement learning significantly outperforms previous baselines in many MORL environments. 

---
# Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context 

**Authors**: Ge Zheng, Jiaye Qian, Jiajin Tang, Sibei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20229)  

**Abstract**: Large Vision-Language Models (LVLMs) have made significant progress in recent years but are also prone to hallucination issues. They exhibit more hallucinations in longer, free-form responses, often attributed to accumulated uncertainties. In this paper, we ask: Does increased hallucination result solely from length-induced errors, or is there a deeper underlying mechanism? After a series of preliminary experiments and findings, we suggest that the risk of hallucinations is not caused by length itself but by the increased reliance on context for coherence and completeness in longer responses. Building on these insights, we propose a novel "induce-detect-suppress" framework that actively induces hallucinations through deliberately designed contexts, leverages induced instances for early detection of high-risk cases, and ultimately suppresses potential object-level hallucinations during actual decoding. Our approach achieves consistent, significant improvements across all benchmarks, demonstrating its efficacy. The strong detection and improved hallucination mitigation not only validate our framework but, more importantly, re-validate our hypothesis on context. Rather than solely pursuing performance gains, this study aims to provide new insights and serves as a first step toward a deeper exploration of hallucinations in LVLMs' longer responses. 

---
# Federated Learning via Meta-Variational Dropout 

**Authors**: Insu Jeon, Minui Hong, Junhyeog Yun, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20225)  

**Abstract**: Federated Learning (FL) aims to train a global inference model from remotely distributed clients, gaining popularity due to its benefit of improving data privacy. However, traditional FL often faces challenges in practical applications, including model overfitting and divergent local models due to limited and non-IID data among clients. To address these issues, we introduce a novel Bayesian meta-learning approach called meta-variational dropout (MetaVD). MetaVD learns to predict client-dependent dropout rates via a shared hypernetwork, enabling effective model personalization of FL algorithms in limited non-IID data settings. We also emphasize the posterior adaptation view of meta-learning and the posterior aggregation view of Bayesian FL via the conditional dropout posterior. We conducted extensive experiments on various sparse and non-IID FL datasets. MetaVD demonstrated excellent classification accuracy and uncertainty calibration performance, especially for out-of-distribution (OOD) clients. MetaVD compresses the local model parameters needed for each client, mitigating model overfitting and reducing communication costs. Code is available at this https URL. 

---
# QKCV Attention: Enhancing Time Series Forecasting with Static Categorical Embeddings for Both Lightweight and Pre-trained Foundation Models 

**Authors**: Hao Wang, Baojun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.20222)  

**Abstract**: In real-world time series forecasting tasks, category information plays a pivotal role in capturing inherent data patterns. This paper introduces QKCV (Query-Key-Category-Value) attention, an extension of the traditional QKV framework that incorporates a static categorical embedding C to emphasize category-specific information. As a versatile plug-in module, QKCV enhances the forecasting accuracy of attention-based models (e.g., Vanilla Transformer, Informer, PatchTST, TFT) across diverse real-world datasets. Furthermore, QKCV demonstrates remarkable adaptability in fine-tuning univariate time series foundation model by solely updating the static embedding C while preserving pretrained weights, thereby reducing computational overhead and achieving superior fine-tuning performance. 

---
# FinCARE: Financial Causal Analysis with Reasoning and Evidence 

**Authors**: Alejandro Michel, Abhinav Arun, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2510.20221)  

**Abstract**: Portfolio managers rely on correlation-based analysis and heuristic methods that fail to capture true causal relationships driving performance. We present a hybrid framework that integrates statistical causal discovery algorithms with domain knowledge from two complementary sources: a financial knowledge graph extracted from SEC 10-K filings and large language model reasoning. Our approach systematically enhances three representative causal discovery paradigms, constraint-based (PC), score-based (GES), and continuous optimization (NOTEARS), by encoding knowledge graph constraints algorithmically and leveraging LLM conceptual reasoning for hypothesis generation. Evaluated on a synthetic financial dataset of 500 firms across 18 variables, our KG+LLM-enhanced methods demonstrate consistent improvements across all three algorithms: PC (F1: 0.622 vs. 0.459 baseline, +36%), GES (F1: 0.735 vs. 0.367, +100%), and NOTEARS (F1: 0.759 vs. 0.163, +366%). The framework enables reliable scenario analysis with mean absolute error of 0.003610 for counterfactual predictions and perfect directional accuracy for intervention effects. It also addresses critical limitations of existing methods by grounding statistical discoveries in financial domain expertise while maintaining empirical validation, providing portfolio managers with the causal foundation necessary for proactive risk management and strategic decision-making in dynamic market environments. 

---
# High-order Interactions Modeling for Interpretable Multi-Agent Q-Learning 

**Authors**: Qinyu Xu, Yuanyang Zhu, Xuefei Wu, Chunlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20218)  

**Abstract**: The ability to model interactions among agents is crucial for effective coordination and understanding their cooperation mechanisms in multi-agent reinforcement learning (MARL). However, previous efforts to model high-order interactions have been primarily hindered by the combinatorial explosion or the opaque nature of their black-box network structures. In this paper, we propose a novel value decomposition framework, called Continued Fraction Q-Learning (QCoFr), which can flexibly capture arbitrary-order agent interactions with only linear complexity $\mathcal{O}\left({n}\right)$ in the number of agents, thus avoiding the combinatorial explosion when modeling rich cooperation. Furthermore, we introduce the variational information bottleneck to extract latent information for estimating credits. This latent information helps agents filter out noisy interactions, thereby significantly enhancing both cooperation and interpretability. Extensive experiments demonstrate that QCoFr not only consistently achieves better performance but also provides interpretability that aligns with our theoretical analysis. 

---
# Automated Cloud Infrastructure-as-Code Reconciliation with AI Agents 

**Authors**: Zhenning Yang, Hui Guan, Victor Nicolet, Brandon Paulsen, Joey Dodds, Daniel Kroening, Ang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20211)  

**Abstract**: Cloud infrastructure is managed through a mix of interfaces -- traditionally, cloud consoles, command-line interfaces (CLI), and SDKs are the tools of choice. Recently, Infrastructure-as-Code/IaC frameworks (e.g., Terraform) have quickly gained popularity. Unlike conventional tools, IaC~frameworks encode the infrastructure in a "source-of-truth" configuration. They are capable of automatically carrying out modifications to the cloud -- deploying, updating, or destroying resources -- to bring the actual infrastructure into alignment with the IaC configuration. However, when IaC is used alongside consoles, CLIs, or SDKs, it loses visibility into external changes, causing infrastructure drift, where the configuration becomes outdated, and later IaC operations may undo valid updates or trigger errors.
We present NSync, an automated system for IaC reconciliation that propagates out-of-band changes back into the IaC program. Our key insight is that infrastructure changes eventually all occur via cloud API invocations -- the lowest layer for cloud management operations. NSync gleans insights from API traces to detect drift (i.e., non-IaC changes) and reconcile it (i.e., update the IaC configuration to capture the changes). It employs an agentic architecture that leverages LLMs to infer high-level intents from noisy API sequences, synthesize targeted IaC updates using specialized tools, and continually improve through a self-evolving knowledge base of past reconciliations. We further introduce a novel evaluation pipeline for injecting realistic drifts into cloud infrastructure and assessing reconciliation performance. Experiments across five real-world Terraform projects and 372 drift scenarios show that NSync outperforms the baseline both in terms of accuracy (from 0.71 to 0.97 pass@3) and token efficiency (1.47$\times$ improvement). 

---
# Assessing the Feasibility of Early Cancer Detection Using Routine Laboratory Data: An Evaluation of Machine Learning Approaches on an Imbalanced Dataset 

**Authors**: Shumin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.20209)  

**Abstract**: The development of accessible screening tools for early cancer detection in dogs represents a significant challenge in veterinary medicine. Routine laboratory data offer a promising, low-cost source for such tools, but their utility is hampered by the non-specificity of individual biomarkers and the severe class imbalance inherent in screening populations. This study assesses the feasibility of cancer risk classification using the Golden Retriever Lifetime Study (GRLS) cohort under real-world constraints, including the grouping of diverse cancer types and the inclusion of post-diagnosis samples. A comprehensive benchmark evaluation was conducted, systematically comparing 126 analytical pipelines that comprised various machine learning models, feature selection methods, and data balancing techniques. Data were partitioned at the patient level to prevent leakage. The optimal model, a Logistic Regression classifier with class weighting and recursive feature elimination, demonstrated moderate ranking ability (AUROC = 0.815; 95% CI: 0.793-0.836) but poor clinical classification performance (F1-score = 0.25, Positive Predictive Value = 0.15). While a high Negative Predictive Value (0.98) was achieved, insufficient recall (0.79) precludes its use as a reliable rule-out test. Interpretability analysis with SHapley Additive exPlanations (SHAP) revealed that predictions were driven by non-specific features like age and markers of inflammation and anemia. It is concluded that while a statistically detectable cancer signal exists in routine lab data, it is too weak and confounded for clinically reliable discrimination from normal aging or other inflammatory conditions. This work establishes a critical performance ceiling for this data modality in isolation and underscores that meaningful progress in computational veterinary oncology will require integration of multi-modal data sources. 

---
# Stuck in the Matrix: Probing Spatial Reasoning in Large Language Models 

**Authors**: Maggie Bai, Ava Kim Cohen, Eleanor Koss, Charlie Lichtenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2510.20198)  

**Abstract**: This paper explores the spatial reasoning capability of large language models (LLMs) over textual input through a suite of five tasks aimed at probing their spatial understanding and computational abilities. The models were tested on both fundamental spatial reasoning and multi-step problem-solving within structured grid-based environments using tasks such as quadrant identification, geometric transformations, distance evaluation, word searches, and tile sliding. Each task was scaled in complexity through increasing grid dimensions, requiring models to extend beyond simple pattern recognition into abstract spatial reasoning. Our results reveal that while LLMs demonstrate moderate success in all tasks with small complexity and size, performance drops off rapidly as scale increases, with an average loss in accuracy of 42.7%, and reaching as high as 84%. Every test that began with over 50% accuracy showed a loss of at least 48%, illustrating the consistent nature of the deterioration. Furthermore, their struggles with scaling complexity hint at a lack of robust spatial representations in their underlying architectures. This paper underscores the gap between linguistic and spatial reasoning in LLMs, offering insights into their current limitations, and laying the groundwork for future integrative benchmarks at the intersection of language and geometry. 

---
# PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching 

**Authors**: Yun Wang, Junjie Hu, Qiaole Dong, Yongjian Zhang, Yanwei Fu, Tin Lun Lam, Dapeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20178)  

**Abstract**: Temporally consistent depth estimation from stereo video is critical for real-world applications such as augmented reality, where inconsistent depth estimation disrupts the immersion of users. Despite its importance, this task remains challenging due to the difficulty in modeling long-term temporal consistency in a computationally efficient manner. Previous methods attempt to address this by aggregating spatio-temporal information but face a fundamental trade-off: limited temporal modeling provides only modest gains, whereas capturing long-range dependencies significantly increases computational cost. To address this limitation, we introduce a memory buffer for modeling long-range spatio-temporal consistency while achieving efficient dynamic stereo matching. Inspired by the two-stage decision-making process in humans, we propose a \textbf{P}ick-and-\textbf{P}lay \textbf{M}emory (PPM) construction module for dynamic \textbf{Stereo} matching, dubbed as \textbf{PPMStereo}. PPM consists of a `pick' process that identifies the most relevant frames and a `play' process that weights the selected frames adaptively for spatio-temporal aggregation. This two-stage collaborative process maintains a compact yet highly informative memory buffer while achieving temporally consistent information aggregation. Extensive experiments validate the effectiveness of PPMStereo, demonstrating state-of-the-art performance in both accuracy and temporal consistency. % Notably, PPMStereo achieves 0.62/1.11 TEPE on the Sintel clean/final (17.3\% \& 9.02\% improvements over BiDAStereo) with fewer computational costs. Codes are available at \textcolor{blue}{this https URL}. 

---
# Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding 

**Authors**: Yuhang Zhou, Mingrui Zhang, Ke Li, Mingyi Wang, Qiao Liu, Qifei wang, Jiayi Liu, Fei Liu, Serena Li, Weiwi Li, Mingze Gao, Abhishek Kumar, Xiangjun Fan, Zhuokai Zhao, Lizhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20176)  

**Abstract**: Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding. 

---
# Collective Communication for 100k+ GPUs 

**Authors**: Min Si, Pavan Balaji, Yongzhou Chen, Ching-Hsiang Chu, Adi Gangidi, Saif Hasan, Subodh Iyengar, Dan Johnson, Bingzhe Liu, Jingliang Ren, Ashmitha Jeevaraj Shetty, Greg Steinbrecher, Xinfeng Xie, Yulun Wang, Bruce Wu, Jingyi Yang, Mingran Yang, Minlan Yu, Cen Zhao, Wes Bland, Denis Boyda, Suman Gumudavelli, Cristian Lumezanu, Rui Miao, Zhe Qu, Venkat Ramesh, Maxim Samoylov, Jan Seidel, Feng Tian, Qiye Tan, Shuqiang Zhang, Yimeng Zhao, Shengbao Zheng, Art Zhu, Hongyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.20171)  

**Abstract**: The increasing scale of large language models (LLMs) necessitates highly efficient collective communication frameworks, particularly as training workloads extend to hundreds of thousands of GPUs. Traditional communication methods face significant throughput and latency limitations at this scale, hindering both the development and deployment of state-of-the-art models. This paper presents the NCCLX collective communication framework, developed at Meta, engineered to optimize performance across the full LLM lifecycle, from the synchronous demands of large-scale training to the low-latency requirements of inference. The framework is designed to support complex workloads on clusters exceeding 100,000 GPUs, ensuring reliable, high-throughput, and low-latency data exchange. Empirical evaluation on the Llama4 model demonstrates substantial improvements in communication efficiency. This research contributes a robust solution for enabling the next generation of LLMs to operate at unprecedented scales. 

---
# IB-GAN: Disentangled Representation Learning with Information Bottleneck Generative Adversarial Networks 

**Authors**: Insu Jeon, Wonkwang Lee, Myeongjang Pyeon, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20165)  

**Abstract**: We propose a new GAN-based unsupervised model for disentangled representation learning. The new model is discovered in an attempt to utilize the Information Bottleneck (IB) framework to the optimization of GAN, thereby named IB-GAN. The architecture of IB-GAN is partially similar to that of InfoGAN but has a critical difference; an intermediate layer of the generator is leveraged to constrain the mutual information between the input and the generated output. The intermediate stochastic layer can serve as a learnable latent distribution that is trained with the generator jointly in an end-to-end fashion. As a result, the generator of IB-GAN can harness the latent space in a disentangled and interpretable manner. With the experiments on dSprites and Color-dSprites dataset, we demonstrate that IB-GAN achieves competitive disentanglement scores to those of state-of-the-art \b{eta}-VAEs and outperforms InfoGAN. Moreover, the visual quality and the diversity of samples generated by IB-GAN are often better than those by \b{eta}-VAEs and Info-GAN in terms of FID score on CelebA and 3D Chairs dataset. 

---
# Are Stereotypes Leading LLMs' Zero-Shot Stance Detection ? 

**Authors**: Anthony Dubreuil, Antoine Gourru, Christine Largeron, Amine Trabelsi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20154)  

**Abstract**: Large Language Models inherit stereotypes from their pretraining data, leading to biased behavior toward certain social groups in many Natural Language Processing tasks, such as hateful speech detection or sentiment analysis. Surprisingly, the evaluation of this kind of bias in stance detection methods has been largely overlooked by the community. Stance Detection involves labeling a statement as being against, in favor, or neutral towards a specific target and is among the most sensitive NLP tasks, as it often relates to political leanings. In this paper, we focus on the bias of Large Language Models when performing stance detection in a zero-shot setting. We automatically annotate posts in pre-existing stance detection datasets with two attributes: dialect or vernacular of a specific group and text complexity/readability, to investigate whether these attributes influence the model's stance detection decisions. Our results show that LLMs exhibit significant stereotypes in stance detection tasks, such as incorrectly associating pro-marijuana views with low text complexity and African American dialect with opposition to Donald Trump. 

---
# SAID: Empowering Large Language Models with Self-Activating Internal Defense 

**Authors**: Yulong Chen, Yadong Liu, Jiawen Zhang, Mu Li, Chao Huang, Jie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20129)  

**Abstract**: Large Language Models (LLMs), despite advances in safety alignment, remain vulnerable to jailbreak attacks designed to circumvent protective mechanisms. Prevailing defense strategies rely on external interventions, such as input filtering or output modification, which often lack generalizability and compromise model utility while incurring significant computational overhead. In this work, we introduce a new, training-free defense paradigm, Self-Activating Internal Defense (SAID), which reframes the defense task from external correction to internal capability activation. SAID uniquely leverages the LLM's own reasoning abilities to proactively identify and neutralize malicious intent through a three-stage pipeline: model-native intent distillation to extract core semantics, optimal safety prefix probing to activate latent safety awareness, and a conservative aggregation strategy to ensure robust decision-making. Extensive experiments on five open-source LLMs against six advanced jailbreak attacks demonstrate that SAID substantially outperforms state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves this while preserving model performance on benign tasks and incurring minimal computational overhead. Our work establishes that activating the intrinsic safety mechanisms of LLMs is a more robust and scalable path toward building safer and more reliable aligned AI systems. 

---
# Leveraging the Power of Large Language Models in Entity Linking via Adaptive Routing and Targeted Reasoning 

**Authors**: Yajie Li, Albert Galimov, Mitra Datta Ganapaneni, Pujitha Thejaswi, De Meng, Priyanshu Kumar, Saloni Potdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.20098)  

**Abstract**: Entity Linking (EL) has traditionally relied on large annotated datasets and extensive model fine-tuning. While recent few-shot methods leverage large language models (LLMs) through prompting to reduce training requirements, they often suffer from inefficiencies due to expensive LLM-based reasoning. ARTER (Adaptive Routing and Targeted Entity Reasoning) presents a structured pipeline that achieves high performance without deep fine-tuning by strategically combining candidate generation, context-based scoring, adaptive routing, and selective reasoning. ARTER computes a small set of complementary signals(both embedding and LLM-based) over the retrieved candidates to categorize contextual mentions into easy and hard cases. The cases are then handled by a low-computational entity linker (e.g. ReFinED) and more expensive targeted LLM-based reasoning respectively. On standard benchmarks, ARTER outperforms ReFinED by up to +4.47%, with an average gain of +2.53% on 5 out of 6 datasets, and performs comparably to pipelines using LLM-based reasoning for all mentions, while being as twice as efficient in terms of the number of LLM tokens. 

---
# On the Structure of Stationary Solutions to McKean-Vlasov Equations with Applications to Noisy Transformers 

**Authors**: Krishnakumar Balasubramanian, Sayan Banerjee, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2510.20094)  

**Abstract**: We study stationary solutions of McKean-Vlasov equations on the circle. Our main contributions stem from observing an exact equivalence between solutions of the stationary McKean-Vlasov equation and an infinite-dimensional quadratic system of equations over Fourier coefficients, which allows explicit characterization of the stationary states in a sequence space rather than a function space. This framework provides a transparent description of local bifurcations, characterizing their periodicity, and resonance structures, while accommodating singular potentials. We derive analytic expressions that characterize the emergence, form and shape (supercritical, critical, subcritical or transcritical) of bifurcations involving possibly multiple Fourier modes and connect them with discontinuous phase transitions. We also characterize, under suitable assumptions, the detailed structure of the stationary bifurcating solutions that are accurate upto an arbitrary number of Fourier modes. At the global level, we establish regularity and concavity properties of the free energy landscape, proving existence, compactness, and coexistence of globally minimizing stationary measures, further identifying discontinuous phase transitions with points of non-differentiability of the minimum free energy map. As an application, we specialize the theory to the Noisy Mean-Field Transformer model, where we show how changing the inverse temperature parameter $\beta$ affects the geometry of the infinitely many bifurcations from the uniform measure. We also explain how increasing $\beta$ can lead to a rich class of approximate multi-mode stationary solutions which can be seen as `metastable states'. Further, a sharp transition from continuous to discontinuous (first-order) phase behavior is observed as $\beta$ increases. 

---
# StableSketcher: Enhancing Diffusion Model for Pixel-based Sketch Generation via Visual Question Answering Feedback 

**Authors**: Jiho Park, Sieun Choi, Jaeyoon Seo, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20093)  

**Abstract**: Although recent advancements in diffusion models have significantly enriched the quality of generated images, challenges remain in synthesizing pixel-based human-drawn sketches, a representative example of abstract expression. To combat these challenges, we propose StableSketcher, a novel framework that empowers diffusion models to generate hand-drawn sketches with high prompt fidelity. Within this framework, we fine-tune the variational autoencoder to optimize latent decoding, enabling it to better capture the characteristics of sketches. In parallel, we integrate a new reward function for reinforcement learning based on visual question answering, which improves text-image alignment and semantic consistency. Extensive experiments demonstrate that StableSketcher generates sketches with improved stylistic fidelity, achieving better alignment with prompts compared to the Stable Diffusion baseline. Additionally, we introduce SketchDUO, to the best of our knowledge, the first dataset comprising instance-level sketches paired with captions and question-answer pairs, thereby addressing the limitations of existing datasets that rely on image-label pairs. Our code and dataset will be made publicly available upon acceptance. 

---
# CreativityPrism: A Holistic Benchmark for Large Language Model Creativity 

**Authors**: Zhaoyi Joey Hou, Bowei Alvin Zhang, Yining Lu, Bhiman Kumar Baghel, Anneliese Brei, Ximing Lu, Meng Jiang, Faeze Brahman, Snigdha Chaturvedi, Haw-Shiuan Chang, Daniel Khashabi, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.20091)  

**Abstract**: Creativity is often seen as a hallmark of human intelligence. While large language models (LLMs) are increasingly perceived as producing creative text, there is still no holistic framework to evaluate their creativity across diverse scenarios. Existing evaluation methods remain fragmented, with dramatic variation across domains and tasks, largely due to differing definitions and measurements of creativity. Inspired by the hypothesis that creativity is not one fixed idea, we propose CreativityPrism, an evaluation analysis framework that decomposes creativity into three dimensions: quality, novelty, and diversity. CreativityPrism incorporates nine tasks, three domains, i.e., divergent thinking, creative writing, and logical reasoning, and twenty evaluation metrics, which measure each dimension in task-specific, unique ways. We evaluate 17 state-of-the-art (SoTA) proprietary and open-sourced LLMs on CreativityPrism and analyze the performance correlations among different metrics and task domains. Our results reveal a notable gap between proprietary and open-source models. Overall, model performance tends to be highly correlated across tasks within the same domain and less so across different domains. Among evaluation dimensions, diversity and quality metrics show strong correlations - models that perform well on one often excel on the other - whereas novelty exhibits much weaker correlation with either. These findings support our hypothesis that strong performance in one creativity task or dimension does not necessarily generalize to others, underscoring the need for a holistic evaluation of LLM creativity. 

---
# ShapeX: Shapelet-Driven Post Hoc Explanations for Time Series Classification Models 

**Authors**: Bosong Huang, Ming Jin, Yuxuan Liang, Johan Barthelemy, Debo Cheng, Qingsong Wen, Chenghao Liu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20084)  

**Abstract**: Explaining time series classification models is crucial, particularly in high-stakes applications such as healthcare and finance, where transparency and trust play a critical role. Although numerous time series classification methods have identified key subsequences, known as shapelets, as core features for achieving state-of-the-art performance and validating their pivotal role in classification outcomes, existing post-hoc time series explanation (PHTSE) methods primarily focus on timestep-level feature attribution. These explanation methods overlook the fundamental prior that classification outcomes are predominantly driven by key shapelets. To bridge this gap, we present ShapeX, an innovative framework that segments time series into meaningful shapelet-driven segments and employs Shapley values to assess their saliency. At the core of ShapeX lies the Shapelet Describe-and-Detect (SDD) framework, which effectively learns a diverse set of shapelets essential for classification. We further demonstrate that ShapeX produces explanations which reveal causal relationships instead of just correlations, owing to the atomicity properties of shapelets. Experimental results on both synthetic and real-world datasets demonstrate that ShapeX outperforms existing methods in identifying the most relevant subsequences, enhancing both the precision and causal fidelity of time series explanations. 

---
# Ask What Your Country Can Do For You: Towards a Public Red Teaming Model 

**Authors**: Wm. Matthew Kennedy, Cigdem Patlak, Jayraj Dave, Blake Chambers, Aayush Dhanotiya, Darshini Ramiah, Reva Schwartz, Jack Hagen, Akash Kundu, Mouni Pendharkar, Liam Baisley, Theodora Skeadas, Rumman Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.20061)  

**Abstract**: AI systems have the potential to produce both benefits and harms, but without rigorous and ongoing adversarial evaluation, AI actors will struggle to assess the breadth and magnitude of the AI risk surface. Researchers from the field of systems design have developed several effective sociotechnical AI evaluation and red teaming techniques targeting bias, hate speech, mis/disinformation, and other documented harm classes. However, as increasingly sophisticated AI systems are released into high-stakes sectors (such as education, healthcare, and intelligence-gathering), our current evaluation and monitoring methods are proving less and less capable of delivering effective oversight.
In order to actually deliver responsible AI and to ensure AI's harms are fully understood and its security vulnerabilities mitigated, pioneering new approaches to close this "responsibility gap" are now more urgent than ever. In this paper, we propose one such approach, the cooperative public AI red-teaming exercise, and discuss early results of its prior pilot implementations. This approach is intertwined with CAMLIS itself: the first in-person public demonstrator exercise was held in conjunction with CAMLIS 2024. We review the operational design and results of this exercise, the prior National Institute of Standards and Technology (NIST)'s Assessing the Risks and Impacts of AI (ARIA) pilot exercise, and another similar exercise conducted with the Singapore Infocomm Media Development Authority (IMDA). Ultimately, we argue that this approach is both capable of delivering meaningful results and is also scalable to many AI developing jurisdictions. 

---
# Approximate Model Predictive Control for Microgrid Energy Management via Imitation Learning 

**Authors**: Changrui Liu, Shengling Shi, Anil Alan, Ganesh Kumar Venayagamoorthy, Bart De Schutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.20040)  

**Abstract**: Efficient energy management is essential for reliable and sustainable microgrid operation amid increasing renewable integration. This paper proposes an imitation learning-based framework to approximate mixed-integer Economic Model Predictive Control (EMPC) for microgrid energy management. The proposed method trains a neural network to imitate expert EMPC control actions from offline trajectories, enabling fast, real-time decision making without solving optimization problems online. To enhance robustness and generalization, the learning process includes noise injection during training to mitigate distribution shift and explicitly incorporates forecast uncertainty in renewable generation and demand. Simulation results demonstrate that the learned policy achieves economic performance comparable to EMPC while only requiring $10\%$ of the computation time of optimization-based EMPC in practice. 

---
# Beyond One-Way Influence: Bidirectional Opinion Dynamics in Multi-Turn Human-LLM Interactions 

**Authors**: Yuyang Jiang, Longjie Guo, Yuchen Wu, Aylin Caliskan, Tanu Mitra, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.20039)  

**Abstract**: Large language model (LLM)-powered chatbots are increasingly used for opinion exploration. Prior research examined how LLMs alter user views, yet little work extended beyond one-way influence to address how user input can affect LLM responses and how such bi-directional influence manifests throughout the multi-turn conversations. This study investigates this dynamic through 50 controversial-topic discussions with participants (N=266) across three conditions: static statements, standard chatbot, and personalized chatbot. Results show that human opinions barely shifted, while LLM outputs changed more substantially, narrowing the gap between human and LLM stance. Personalization amplified these shifts in both directions compared to the standard setting. Analysis of multi-turn conversations further revealed that exchanges involving participants' personal stories were most likely to trigger stance changes for both humans and LLMs. Our work highlights the risk of over-alignment in human-LLM interaction and the need for careful design of personalized chatbots to more thoughtfully and stably align with users. 

---
# The Temporal Graph of Bitcoin Transactions 

**Authors**: Vahid Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2510.20028)  

**Abstract**: Since its 2009 genesis block, the Bitcoin network has processed \num{>1.08} billion (B) transactions representing \num{>8.72}B BTC, offering rich potential for machine learning (ML); yet, its pseudonymity and obscured flow of funds inherent in its \utxo-based design, have rendered this data largely inaccessible for ML research. Addressing this gap, we present an ML-compatible graph modeling the Bitcoin's economic topology by reconstructing the flow of funds. This temporal, heterogeneous graph encompasses complete transaction history up to block \cutoffHeight, consisting of \num{>2.4}B nodes and \num{>39.72}B edges. Additionally, we provide custom sampling methods yielding node and edge feature vectors of sampled communities, tools to load and analyze the Bitcoin graph data within specialized graph databases, and ready-to-use database snapshots. This comprehensive dataset and toolkit empower the ML community to tackle Bitcoin's intricate ecosystem at scale, driving progress in applications such as anomaly detection, address classification, market analysis, and large-scale graph ML benchmarking. Dataset and code available at \href{this https URL}{this http URL} 

---
# Optimized Distortion in Linear Social Choice 

**Authors**: Luise Ge, Gregory Kehne, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2510.20020)  

**Abstract**: Social choice theory offers a wealth of approaches for selecting a candidate on behalf of voters based on their reported preference rankings over options. When voters have underlying utilities for these options, however, using preference rankings may lead to suboptimal outcomes vis-à-vis utilitarian social welfare. Distortion is a measure of this suboptimality, and provides a worst-case approach for developing and analyzing voting rules when utilities have minimal structure. However in many settings, such as common paradigms for value alignment, alternatives admit a vector representation, and it is natural to suppose that utilities are parametric functions thereof. We undertake the first study of distortion for linear utility functions. Specifically, we investigate the distortion of linear social choice for deterministic and randomized voting rules. We obtain bounds that depend only on the dimension of the candidate embedding, and are independent of the numbers of candidates or voters. Additionally, we introduce poly-time instance-optimal algorithms for minimizing distortion given a collection of candidates and votes. We empirically evaluate these in two real-world domains: recommendation systems using collaborative filtering embeddings, and opinion surveys utilizing language model embeddings, benchmarking several standard rules against our instance-optimal algorithms. 

---
# Forging GEMs: Advancing Greek NLP through Quality-Based Corpus Curation and Specialized Pre-training 

**Authors**: Alexandra Apostolopoulou, Konstantinos Kanaris, Athanasios Koursaris, Dimitris Tsakalidis, George Domalis, Ioannis E. Livieris  

**Link**: [PDF](https://arxiv.org/pdf/2510.20002)  

**Abstract**: The advancement of natural language processing for morphologically rich, moderately-resourced languages like Modern Greek is often hindered by a fragmented research landscape, a lack of architectural diversity and reliance on limited context-length models. This is particularly true in specialized, high-value domains such as law, where existing models are frequently confined to early transformer architectures with a restrictive 512-token window, insufficient for analyzing long legal documents. To address these challenges, this paper presents Greek Embedding Models, a new family of transformer models for Greek language built upon a foundation of extensive, quality-driven data curation. We detail the construction of several large-scale Greek corpora, emphasizing a rigorous, quality-based filtering and preprocessing methodology to create high-value training datasets from both general-domain and specialized legal sources. On this carefully curated foundation, we pre-train and systematically evaluate a diverse suite of modern architectures, which has not previously applied to Greek language, such as ELECTRA, ConvBERT and ModernBERT. Furthermore, we propose the first bilingual Greek-English Embedding Models tailored for the legal domain. The extensive experiments on downstream tasks demonstrate that the new class of models establish the effectiveness of the proposed approach, highlighting that the GEM-RoBERTa and GEM-ConvBERT models significantly outperform existing baselines. 

---
# Beyond MedQA: Towards Real-world Clinical Decision Making in the Era of LLMs 

**Authors**: Yunpeng Xiao, Carl Yang, Mark Mai, Xiao Hu, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20001)  

**Abstract**: Large language models (LLMs) show promise for clinical use. They are often evaluated using datasets such as MedQA. However, Many medical datasets, such as MedQA, rely on simplified Question-Answering (Q\A) that underrepresents real-world clinical decision-making. Based on this, we propose a unifying paradigm that characterizes clinical decision-making tasks along two dimensions: Clinical Backgrounds and Clinical Questions. As the background and questions approach the real clinical environment, the difficulty increases. We summarize the settings of existing datasets and benchmarks along two dimensions. Then we review methods to address clinical decision-making, including training-time and test-time techniques, and summarize when they help. Next, we extend evaluation beyond accuracy to include efficiency, explainability. Finally, we highlight open challenges. Our paradigm clarifies assumptions, standardizes comparisons, and guides the development of clinically meaningful LLMs. 

---
# A Framework for the Adoption and Integration of Generative AI in Midsize Organizations and Enterprises (FAIGMOE) 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.19997)  

**Abstract**: Generative Artificial Intelligence (GenAI) presents transformative opportunities for organizations, yet both midsize organizations and larger enterprises face distinctive adoption challenges. Midsize organizations encounter resource constraints and limited AI expertise, while enterprises struggle with organizational complexity and coordination challenges. Existing technology adoption frameworks, including TAM (Technology Acceptance Model), TOE (Technology Organization Environment), and DOI (Diffusion of Innovations) theory, lack the specificity required for GenAI implementation across these diverse contexts, creating a critical gap in adoption literature. This paper introduces FAIGMOE (Framework for the Adoption and Integration of Generative AI in Midsize Organizations and Enterprises), a conceptual framework addressing the unique needs of both organizational types. FAIGMOE synthesizes technology adoption theory, organizational change management, and innovation diffusion perspectives into four interconnected phases: Strategic Assessment, Planning and Use Case Development, Implementation and Integration, and Operationalization and Optimization. Each phase provides scalable guidance on readiness assessment, strategic alignment, risk governance, technical architecture, and change management adaptable to organizational scale and complexity. The framework incorporates GenAI specific considerations including prompt engineering, model orchestration, and hallucination management that distinguish it from generic technology adoption frameworks. As a perspective contribution, FAIGMOE provides the first comprehensive conceptual framework explicitly addressing GenAI adoption across midsize and enterprise organizations, offering actionable implementation protocols, assessment instruments, and governance templates requiring empirical validation through future research. 

---
# LLM-Augmented Symbolic NLU System for More Reliable Continuous Causal Statement Interpretation 

**Authors**: Xin Lian, Kenneth D. Forbus  

**Link**: [PDF](https://arxiv.org/pdf/2510.19988)  

**Abstract**: Despite the broad applicability of large language models (LLMs), their reliance on probabilistic inference makes them vulnerable to errors such as hallucination in generated facts and inconsistent output structure in natural language understanding (NLU) tasks. By contrast, symbolic NLU systems provide interpretable understanding grounded in curated lexicons, semantic resources, and syntactic & semantic interpretation rules. They produce relational representations that can be used for accurate reasoning and planning, as well as incremental debuggable learning. However, symbolic NLU systems tend to be more limited in coverage than LLMs and require scarce knowledge representation and linguistics skills to extend and maintain. This paper explores a hybrid approach that integrates the broad-coverage language processing of LLMs with the symbolic NLU capabilities of producing structured relational representations to hopefully get the best of both approaches. We use LLMs for rephrasing and text simplification, to provide broad coverage, and as a source of information to fill in knowledge gaps more automatically. We use symbolic NLU to produce representations that can be used for reasoning and for incremental learning. We evaluate this approach on the task of extracting and interpreting quantities and causal laws from commonsense science texts, along with symbolic- and LLM-only pipelines. Our results suggest that our hybrid method works significantly better than the symbolic-only pipeline. 

---
# Revisiting Zeroth-Order Optimization: Minimum-Variance Two-Point Estimators and Directionally Aligned Perturbations 

**Authors**: Shaocong Ma, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19975)  

**Abstract**: In this paper, we explore the two-point zeroth-order gradient estimator and identify the distribution of random perturbations that minimizes the estimator's asymptotic variance as the perturbation stepsize tends to zero. We formulate it as a constrained functional optimization problem over the space of perturbation distributions. Our findings reveal that such desired perturbations can align directionally with the true gradient, instead of maintaining a fixed length. While existing research has largely focused on fixed-length perturbations, the potential advantages of directional alignment have been overlooked. To address this gap, we delve into the theoretical and empirical properties of the directionally aligned perturbation (DAP) scheme, which adaptively offers higher accuracy along critical directions. Additionally, we provide a convergence analysis for stochastic gradient descent using $\delta$-unbiased random perturbations, extending existing complexity bounds to a wider range of perturbations. Through empirical evaluations on both synthetic problems and practical tasks, we demonstrate that DAPs outperform traditional methods under specific conditions. 

---
# A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks 

**Authors**: Hatim Chergui, Farhad Rezazadeh, Merouane Debbah, Christos Verikoukis  

**Link**: [PDF](https://arxiv.org/pdf/2510.19973)  

**Abstract**: The path to higher network autonomy in 6G lies beyond the mere optimization of key performance indicators (KPIs). While KPIs have enabled automation gains under TM Forum Levels 1--3, they remain numerical abstractions that act only as proxies for the real essence of communication networks: seamless connectivity, fairness, adaptability, and resilience. True autonomy requires perceiving and reasoning over the network environment as it is. Such progress can be achieved through \emph{agentic AI}, where large language model (LLM)-powered agents perceive multimodal telemetry, reason with memory, negotiate across domains, and act via APIs to achieve multi-objective goals. However, deploying such agents introduces the challenge of cognitive biases inherited from human design, which can distort reasoning, negotiation, tool use, and actuation. Between neuroscience and AI, this paper provides a tutorial on a selection of well-known biases, including their taxonomy, definition, mathematical formulation, emergence in telecom systems and the commonly impacted agentic components. The tutorial also presents various mitigation strategies tailored to each type of bias. The article finally provides two practical use-cases, which tackle the emergence, impact and mitigation gain of some famous biases in 6G inter-slice and cross-domain management. In particular, anchor randomization, temporal decay and inflection bonus techniques are introduced to specifically address anchoring, temporal and confirmation biases. This avoids that agents stick to the initial high resource allocation proposal or decisions that are recent and/or confirming a prior hypothesis. By grounding decisions in a richer and fairer set of past experiences, the quality and bravery of the agentic agreements in the second use-case, for instance, are leading to $\times 5$ lower latency and around $40\%$ higher energy saving. 

---
# LyriCAR: A Difficulty-Aware Curriculum Reinforcement Learning Framework For Controllable Lyric Translation 

**Authors**: Le Ren, Xiangjian Zeng, Qingqiang Wu, Ruoxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19967)  

**Abstract**: Lyric translation is a challenging task that requires balancing multiple musical constraints. Existing methods often rely on hand-crafted rules and sentence-level modeling, which restrict their ability to internalize musical-linguistic patterns and to generalize effectively at the paragraph level, where cross-line coherence and global rhyme are crucial. In this work, we propose LyriCAR, a novel framework for controllable lyric translation that operates in a fully unsupervised manner. LyriCAR introduces a difficulty-aware curriculum designer and an adaptive curriculum strategy, ensuring efficient allocation of training resources, accelerating convergence, and improving overall translation quality by guiding the model with increasingly complex challenges. Extensive experiments on the EN-ZH lyric translation task show that LyriCAR achieves state-of-the-art results across both standard translation metrics and multi-dimensional reward scores, surpassing strong baselines. Notably, the adaptive curriculum strategy reduces training steps by nearly 40% while maintaining superior performance. Code, data and model can be accessed at this https URL. 

---
# On the Optimal Construction of Unbiased Gradient Estimators for Zeroth-Order Optimization 

**Authors**: Shaocong Ma, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19953)  

**Abstract**: Zeroth-order optimization (ZOO) is an important framework for stochastic optimization when gradients are unavailable or expensive to compute. A potential limitation of existing ZOO methods is the bias inherent in most gradient estimators unless the perturbation stepsize vanishes. In this paper, we overcome this biasedness issue by proposing a novel family of unbiased gradient estimators based solely on function evaluations. By reformulating directional derivatives as a telescoping series and sampling from carefully designed distributions, we construct estimators that eliminate bias while maintaining favorable variance. We analyze their theoretical properties, derive optimal scaling distributions and perturbation stepsizes of four specific constructions, and prove that SGD using the proposed estimators achieves optimal complexity for smooth non-convex objectives. Experiments on synthetic tasks and language model fine-tuning confirm the superior accuracy and convergence of our approach compared to standard methods. 

---
# Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets 

**Authors**: Shaocong Ma, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19950)  

**Abstract**: In financial applications, reinforcement learning (RL) agents are commonly trained on historical data, where their actions do not influence prices. However, during deployment, these agents trade in live markets where their own transactions can shift asset prices, a phenomenon known as market impact. This mismatch between training and deployment environments can significantly degrade performance. Traditional robust RL approaches address this model misspecification by optimizing the worst-case performance over a set of uncertainties, but typically rely on symmetric structures that fail to capture the directional nature of market impact. To address this issue, we develop a novel class of elliptic uncertainty sets. We establish both implicit and explicit closed-form solutions for the worst-case uncertainty under these sets, enabling efficient and tractable robust policy evaluation. Experiments on single-asset and multi-asset trading tasks demonstrate that our method achieves superior Sharpe ratio and remains robust under increasing trade volumes, offering a more faithful and scalable approach to RL in financial markets. 

---
# Learning from Supervision with Semantic and Episodic Memory: A Reflective Approach to Agent Adaptation 

**Authors**: Jackson Hassell, Dan Zhang, Hannah Kim, Tom Mitchell, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2510.19897)  

**Abstract**: We investigate how agents built on pretrained large language models can learn target classification functions from labeled examples without parameter updates. While conventional approaches like fine-tuning are often costly, inflexible, and opaque, we propose a memory-augmented framework that leverages both labeled data and LLM-generated critiques. Our framework uses episodic memory to store instance-level critiques-capturing specific past experiences-and semantic memory to distill these into reusable, task-level guidance. Across a diverse set of tasks, incorporating critiques yields up to a 24.8 percent accuracy improvement over retrieval-based (RAG-style) baselines that rely only on labels. Through extensive empirical evaluation, we uncover distinct behavioral differences between OpenAI and opensource models, particularly in how they handle fact-oriented versus preference-based data. To interpret how models respond to different representations of supervision encoded in memory, we introduce a novel metric, suggestibility. This helps explain observed behaviors and illuminates how model characteristics and memory strategies jointly shape learning dynamics. Our findings highlight the promise of memory-driven, reflective learning for building more adaptive and interpretable LLM agents. 

---
# Large Language Model enabled Mathematical Modeling 

**Authors**: Guoyun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19895)  

**Abstract**: The integration of Large Language Models (LLMs) with optimization modeling offers a promising avenue for advancing decision-making in operations research (OR). Traditional optimization methods,such as linear programming, mixed integer programming, and simulation depend heavily on domain expertise to translate real-world problems into solvable mathematical models. While solvers like Gurobi and COPT are powerful, expert input remains essential for defining objectives, constraints, and variables. This research investigates the potential of LLMs, specifically the DeepSeek-R1 model, to bridge this formulation gap using natural language understanding and code generation. Although prior models like GPT-4, Claude, and Bard have shown strong performance in NLP and reasoning tasks, their high token costs and tendency toward hallucinations limit real-world applicability in supply chain contexts. In contrast, DeepSeek-R1, a cost-efficient and high-performing model trained with reinforcement learning, presents a viable alternative. Despite its success in benchmarks such as LiveCodeBench and Math-500, its effectiveness in applied OR scenarios remains under explored. This study systematically evaluates DeepSeek-R1 across four key OR benchmarks: NL4OPT, IndustryOR, EasyLP, and ComplexOR. Our methodology includes baseline assessments, the development of a hallucination taxonomy, and the application of mitigation strategies like LLM-as-a-Judge, Few-shot Learning (FSL), Tool Calling, and a Multi-agent Framework. These techniques aim to reduce hallucinations, enhance formulation accuracy, and better align model outputs with user intent. 

---
# Can They Dixit? Yes they Can! Dixit as a Playground for Multimodal Language Model Capabilities 

**Authors**: Nishant Balepur, Dang Nguyen, Dayeon Ki  

**Link**: [PDF](https://arxiv.org/pdf/2510.19892)  

**Abstract**: Multi-modal large language models (MLMs) are often assessed on static, individual benchmarks -- which cannot jointly assess MLM capabilities in a single task -- or rely on human or model pairwise comparisons -- which is highly subjective, expensive, and allows models to exploit superficial shortcuts (e.g., verbosity) to inflate their win-rates. To overcome these issues, we propose game-based evaluations to holistically assess MLM capabilities. Games require multiple abilities for players to win, are inherently competitive, and are governed by fix, objective rules, and makes evaluation more engaging, providing a robust framework to address the aforementioned challenges. We manifest this evaluation specifically through Dixit, a fantasy card game where players must generate captions for a card that trick some, but not all players, into selecting the played card. Our quantitative experiments with five MLMs show Dixit win-rate rankings are perfectly correlated with those on popular MLM benchmarks, while games between human and MLM players in Dixit reveal several differences between agent strategies and areas of improvement for MLM reasoning. 

---
# From Optimization to Prediction: Transformer-Based Path-Flow Estimation to the Traffic Assignment Problem 

**Authors**: Mostafa Ameli, Van Anh Le, Sulthana Shams, Alexander Skabardonis  

**Link**: [PDF](https://arxiv.org/pdf/2510.19889)  

**Abstract**: The traffic assignment problem is essential for traffic flow analysis, traditionally solved using mathematical programs under the Equilibrium principle. These methods become computationally prohibitive for large-scale networks due to non-linear growth in complexity with the number of OD pairs. This study introduces a novel data-driven approach using deep neural networks, specifically leveraging the Transformer architecture, to predict equilibrium path flows directly. By focusing on path-level traffic distribution, the proposed model captures intricate correlations between OD pairs, offering a more detailed and flexible analysis compared to traditional link-level approaches. The Transformer-based model drastically reduces computation time, while adapting to changes in demand and network structure without the need for recalculation. Numerical experiments are conducted on the Manhattan-like synthetic network, the Sioux Falls network, and the Eastern-Massachusetts network. The results demonstrate that the proposed model is orders of magnitude faster than conventional optimization. It efficiently estimates path-level traffic flows in multi-class networks, reducing computational costs and improving prediction accuracy by capturing detailed trip and flow information. The model also adapts flexibly to varying demand and network conditions, supporting traffic management and enabling rapid `what-if' analyses for enhanced transportation planning and policy-making. 

---
# Quantifying Feature Importance for Online Content Moderation 

**Authors**: Benedetta Tessa, Alejandro Moreo, Stefano Cresci, Tiziano Fagni, Fabrizio Sebastiani  

**Link**: [PDF](https://arxiv.org/pdf/2510.19882)  

**Abstract**: Accurately estimating how users respond to moderation interventions is paramount for developing effective and user-centred moderation strategies. However, this requires a clear understanding of which user characteristics are associated with different behavioural responses, which is the goal of this work. We investigate the informativeness of 753 socio-behavioural, linguistic, relational, and psychological features, in predicting the behavioural changes of 16.8K users affected by a major moderation intervention on Reddit. To reach this goal, we frame the problem in terms of "quantification", a task well-suited to estimating shifts in aggregate user behaviour. We then apply a greedy feature selection strategy with the double goal of (i) identifying the features that are most predictive of changes in user activity, toxicity, and participation diversity, and (ii) estimating their importance. Our results allow identifying a small set of features that are consistently informative across all tasks, and determining that many others are either task-specific or of limited utility altogether. We also find that predictive performance varies according to the task, with changes in activity and toxicity being easier to estimate than changes in diversity. Overall, our results pave the way for the development of accurate systems that predict user reactions to moderation interventions. Furthermore, our findings highlight the complexity of post-moderation user behaviour, and indicate that effective moderation should be tailored not only to user traits but also to the specific objective of the intervention. 

---
# Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention 

**Authors**: J Rosser, José Luis Redondo García, Gustavo Penha, Konstantina Palla, Hugues Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2510.19875)  

**Abstract**: As Large Language Models (LLMs) scale to million-token contexts, traditional Mechanistic Interpretability techniques for analyzing attention scale quadratically with context length, demanding terabytes of memory beyond 100,000 tokens. We introduce Sparse Tracing, a novel technique that leverages dynamic sparse attention to efficiently analyze long context attention patterns. We present Stream, a compilable hierarchical pruning algorithm that estimates per-head sparse attention masks in near-linear time $O(T \log T)$ and linear space $O(T)$, enabling one-pass interpretability at scale. Stream performs a binary-search-style refinement to retain only the top-$k$ key blocks per query while preserving the model's next-token behavior. We apply Stream to long chain-of-thought reasoning traces and identify thought anchors while pruning 97-99\% of token interactions. On the RULER benchmark, Stream preserves critical retrieval paths while discarding 90-96\% of interactions and exposes layer-wise routes from the needle to output. Our method offers a practical drop-in tool for analyzing attention patterns and tracing information flow without terabytes of caches. By making long context interpretability feasible on consumer GPUs, Sparse Tracing helps democratize chain-of-thought monitoring. Code is available at this https URL. 

---
# From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph 

**Authors**: Junfeng Gong, Zhiyi Wei, Junying Chen, Cheng Liu, Huawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.19873)  

**Abstract**: Despite significant evolution of CUDA programming and domain-specific libraries, effectively utilizing GPUs with massively parallel engines remains difficult. Large language models (LLMs) show strong potential in generating optimized CUDA code from sequential code. However, using LLMs in practice faces two major challenges: cloud-based APIs pose risks of code leakage, and local deployment is often computationally expensive and inefficient. These drawbacks have spurred interest in small language models (SLMs), which are more lightweight and privacy-friendly. Encouragingly, recent studies show that SLMs can achieve performance comparable to LLMs on specific tasks. While SLMs can match LLMs on domain-specific tasks, their limited reasoning abilities lead to suboptimal performance in complex CUDA generation according to our experiments. To bridge this gap, we propose ReGraphT, a training-free, retrieval-augmented generation framework that transfers LLM-level reasoning to smaller models. ReGraphT organizes CUDA optimization trajectories into a structured reasoning graph, modeling the combined CUDA optimizations as state transitions, and leverages Monte Carlo Graph Search (MCGS) for efficient exploration. We also present a CUDA-specific benchmark with difficulty tiers defined by reasoning complexity to evaluate models more comprehensively. Experiments show that ReGraphT outperforms HPC-specific fine-tuned models and other retrieval-augmented approaches, achieving an average 2.33X speedup on CUDAEval and ParEval. When paired with DeepSeek-Coder-V2-Lite-Instruct and Qwen2.5-Coder-7B-Instruct, ReGraphT enables SLMs to approach LLM-level performance without the associated privacy risks or excessive computing overhead. 

---
# An Evaluation of the Pedagogical Soundness and Usability of AI-Generated Lesson Plans Across Different Models and Prompt Frameworks in High-School Physics 

**Authors**: Xincheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19866)  

**Abstract**: This study evaluates the pedagogical soundness and usability of AI-generated lesson plans across five leading large language models: ChatGPT (GPT-5), Claude Sonnet 4.5, Gemini 2.5 Flash, DeepSeek V3.2, and Grok 4. Beyond model choice, three structured prompt frameworks were tested: TAG (Task, Audience, Goal), RACE (Role, Audience, Context, Execution), and COSTAR (Context, Objective, Style, Tone, Audience, Response Format).
Fifteen lesson plans were generated for a single high-school physics topic, The Electromagnetic Spectrum. The lesson plans were analyzed through four automated computational metrics: (1) readability and linguistic complexity, (2) factual accuracy and hallucination detection, (3) standards and curriculum alignment, and (4) cognitive demand of learning objectives.
Results indicate that model selection exerted the strongest influence on linguistic accessibility, with DeepSeek producing the most readable teaching plan (FKGL = 8.64) and Claude generating the densest language (FKGL = 19.89).
The prompt framework structure most strongly affected the factual accuracy and pedagogical completeness, with the RACE framework yielding the lowest hallucination index and the highest incidental alignment with NGSS curriculum standards. Across all models, the learning objectives in the fifteen lesson plans clustered at the Remember and Understand tiers of Bloom's taxonomy. There were limited higher-order verbs in the learning objectives extracted.
Overall, the findings suggest that readability is significantly governed by model design, while instructional reliability and curricular alignment depend more on the prompt framework. The most effective configuration for lesson plans identified in the results was to combine a readability-optimized model with the RACE framework and an explicit checklist of physics concepts, curriculum standards, and higher-order objectives. 

---
# Can Reasoning Models Obfuscate Reasoning? Stress-Testing Chain-of-Thought Monitorability 

**Authors**: Artur Zolkowski, Wen Xing, David Lindner, Florian Tramèr, Erik Jenner  

**Link**: [PDF](https://arxiv.org/pdf/2510.19851)  

**Abstract**: Recent findings suggest that misaligned models may exhibit deceptive behavior, raising concerns about output trustworthiness. Chain-of-thought (CoT) is a promising tool for alignment monitoring: when models articulate their reasoning faithfully, monitors can detect and mitigate harmful behaviors before undesirable outcomes occur. However, a key uncertainty is: Can models obfuscate their CoT in order to pursue hidden adversarial objectives while evading detection? To answer this question and thus stress-test CoT monitorability, we develop a composable and quantifiable taxonomy of prompts to elicit CoT obfuscation. We evaluate both internal CoT (reasoning traces) and external CoT (prompted reasoning in outputs) using toy tasks and more realistic environments in SHADE-Arena. We show that: (i) CoT monitoring performs accurately and efficiently without obfuscation pressure. (ii) Under strong obfuscation pressure, some models successfully complete adversarial tasks while evading detection. (iii) Models do not obfuscate their internal CoT as much as their external CoT (under prompt pressure). These results suggest that while CoT provides valuable oversight in benign settings, robust deployment requires model-specific stress-testing of monitorability. 

---
# Prompt Decorators: A Declarative and Composable Syntax for Reasoning, Formatting, and Control in LLMs 

**Authors**: Mostapha Kalami Heris  

**Link**: [PDF](https://arxiv.org/pdf/2510.19850)  

**Abstract**: Large Language Models (LLMs) are central to reasoning, writing, and decision-support workflows, yet users lack consistent control over how they reason and express outputs. Conventional prompt engineering relies on verbose natural-language instructions, limiting reproducibility, modularity, and interpretability. This paper introduces Prompt Decorators, a declarative, composable syntax that governs LLM behavior through compact control tokens such as +++Reasoning, +++Tone(style=formal), and +++Import(topic="Systems Thinking"). Each decorator modifies a behavioral dimension, such as reasoning style, structure, or tone, without changing task content. The framework formalizes twenty core decorators organized into two functional families (Cognitive & Generative and Expressive & Systemic), each further decomposed into subcategories that govern reasoning, interaction, expression, and session-control. It defines a unified syntax, scoping model, and deterministic processing pipeline enabling predictable and auditable behavior composition. By decoupling task intent from execution behavior, Prompt Decorators create a reusable and interpretable interface for prompt design. Illustrative use cases demonstrate improved reasoning transparency, reduced prompt complexity, and standardized model behavior across domains. The paper concludes with implications for interoperability, behavioral consistency, and the development of declarative interfaces for scalable AI systems. 

---
# CourtGuard: A Local, Multiagent Prompt Injection Classifier 

**Authors**: Isaac Wu, Michael Maslowski  

**Link**: [PDF](https://arxiv.org/pdf/2510.19844)  

**Abstract**: As large language models (LLMs) become integrated into various sensitive applications, prompt injection, the use of prompting to induce harmful behaviors from LLMs, poses an ever increasing risk. Prompt injection attacks can cause LLMs to leak sensitive data, spread misinformation, and exhibit harmful behaviors. To defend against these attacks, we propose CourtGuard, a locally-runnable, multiagent prompt injection classifier. In it, prompts are evaluated in a court-like multiagent LLM system, where a "defense attorney" model argues the prompt is benign, a "prosecution attorney" model argues the prompt is a prompt injection, and a "judge" model gives the final classification. CourtGuard has a lower false positive rate than the Direct Detector, an LLM as-a-judge. However, CourtGuard is generally a worse prompt injection detector. Nevertheless, this lower false positive rate highlights the importance of considering both adversarial and benign scenarios for the classification of a prompt. Additionally, the relative performance of CourtGuard in comparison to other prompt injection classifiers advances the use of multiagent systems as a defense against prompt injection attacks. The implementations of CourtGuard and the Direct Detector with full prompts for Gemma-3-12b-it, Llama-3.3-8B, and Phi-4-mini-instruct are available at this https URL. 

---
# SSL-SE-EEG: A Framework for Robust Learning from Unlabeled EEG Data with Self-Supervised Learning and Squeeze-Excitation Networks 

**Authors**: Meghna Roy Chowdhury, Yi Ding, Shreyas Sen  

**Link**: [PDF](https://arxiv.org/pdf/2510.19829)  

**Abstract**: Electroencephalography (EEG) plays a crucial role in brain-computer interfaces (BCIs) and neurological diagnostics, but its real-world deployment faces challenges due to noise artifacts, missing data, and high annotation costs. We introduce SSL-SE-EEG, a framework that integrates Self-Supervised Learning (SSL) with Squeeze-and-Excitation Networks (SE-Nets) to enhance feature extraction, improve noise robustness, and reduce reliance on labeled data. Unlike conventional EEG processing techniques, SSL-SE-EEG} transforms EEG signals into structured 2D image representations, suitable for deep learning. Experimental validation on MindBigData, TUH-AB, SEED-IV and BCI-IV datasets demonstrates state-of-the-art accuracy (91% in MindBigData, 85% in TUH-AB), making it well-suited for real-time BCI applications. By enabling low-power, scalable EEG processing, SSL-SE-EEG presents a promising solution for biomedical signal analysis, neural engineering, and next-generation BCIs. 

---
# SLYKLatent: A Learning Framework for Gaze Estimation Using Deep Facial Feature Learning 

**Authors**: Samuel Adebayo, Joost C. Dessing, Seán McLoone  

**Link**: [PDF](https://arxiv.org/pdf/2402.01555)  

**Abstract**: In this research, we present SLYKLatent, a novel approach for enhancing gaze estimation by addressing appearance instability challenges in datasets due to aleatoric uncertainties, covariant shifts, and test domain generalization. SLYKLatent utilizes Self-Supervised Learning for initial training with facial expression datasets, followed by refinement with a patch-based tri-branch network and an inverse explained variance-weighted training loss function. Our evaluation on benchmark datasets achieves a 10.9% improvement on Gaze360, supersedes top MPIIFaceGaze results with 3.8%, and leads on a subset of ETH-XGaze by 11.6%, surpassing existing methods by significant margins. Adaptability tests on RAF-DB and Affectnet show 86.4% and 60.9% accuracies, respectively. Ablation studies confirm the effectiveness of SLYKLatent's novel components. 

---
