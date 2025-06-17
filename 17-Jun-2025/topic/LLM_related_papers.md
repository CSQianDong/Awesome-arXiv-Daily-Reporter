# Delving Into the Psychology of Machines: Exploring the Structure of Self-Regulated Learning via LLM-Generated Survey Responses 

**Authors**: Leonie V.D.E. Vogelsmeier, Eduardo Oliveira, Kamila Misiejuk, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2506.13384)  

**Abstract**: Large language models (LLMs) offer the potential to simulate human-like responses and behaviors, creating new opportunities for psychological science. In the context of self-regulated learning (SRL), if LLMs can reliably simulate survey responses at scale and speed, they could be used to test intervention scenarios, refine theoretical models, augment sparse datasets, and represent hard-to-reach populations. However, the validity of LLM-generated survey responses remains uncertain, with limited research focused on SRL and existing studies beyond SRL yielding mixed results. Therefore, in this study, we examined LLM-generated responses to the 44-item Motivated Strategies for Learning Questionnaire (MSLQ; Pintrich \& De Groot, 1990), a widely used instrument assessing students' learning strategies and academic motivation. Particularly, we used the LLMs GPT-4o, Claude 3.7 Sonnet, Gemini 2 Flash, LLaMA 3.1-8B, and Mistral Large. We analyzed item distributions, the psychological network of the theoretical SRL dimensions, and psychometric validity based on the latent factor structure. Our results suggest that Gemini 2 Flash was the most promising LLM, showing considerable sampling variability and producing underlying dimensions and theoretical relationships that align with prior theory and empirical findings. At the same time, we observed discrepancies and limitations, underscoring both the potential and current constraints of using LLMs for simulating psychological survey data and applying it in educational contexts. 

---
# Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers 

**Authors**: Wooseok Seo, Seungju Han, Jaehun Jung, Benjamin Newman, Seungwon Lim, Seungbeen Lee, Ximing Lu, Yejin Choi, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13342)  

**Abstract**: Fact verification is essential for ensuring the reliability of LLM applications. In this study, we evaluate 12 pre-trained LLMs and one specialized fact-verifier, including frontier LLMs and open-weight reasoning LLMs, using a collection of examples from 14 fact-checking benchmarks. We share three findings intended to guide future development of more robust fact verifiers. First, we highlight the importance of addressing annotation errors and ambiguity in datasets, demonstrating that approximately 16\% of ambiguous or incorrectly labeled data substantially influences model rankings. Neglecting this issue may result in misleading conclusions during comparative evaluations, and we suggest using a systematic pipeline utilizing LLM-as-a-judge to help identify these issues at scale. Second, we discover that frontier LLMs with few-shot in-context examples, often overlooked in previous works, achieve top-tier performance. We therefore recommend future studies include comparisons with these simple yet highly effective baselines. Lastly, despite their effectiveness, frontier LLMs incur substantial costs, motivating the development of small, fine-tuned fact verifiers. We show that these small models still have room for improvement, particularly on instances that require complex reasoning. Encouragingly, we demonstrate that augmenting training with synthetic multi-hop reasoning data significantly enhances their capabilities in such instances. We release our code, model, and dataset at this https URL 

---
# Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks 

**Authors**: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13276)  

**Abstract**: Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods. 

---
# Vector Ontologies as an LLM world view extraction method 

**Authors**: Kaspar Rothenfusser, Bekk Blando  

**Link**: [PDF](https://arxiv.org/pdf/2506.13252)  

**Abstract**: Large Language Models (LLMs) possess intricate internal representations of the world, yet these latent structures are notoriously difficult to interpret or repurpose beyond the original prediction task. Building on our earlier work (Rothenfusser, 2025), which introduced the concept of vector ontologies as a framework for translating high-dimensional neural representations into interpretable geometric structures, this paper provides the first empirical validation of that approach. A vector ontology defines a domain-specific vector space spanned by ontologically meaningful dimensions, allowing geometric analysis of concepts and relationships within a domain. We construct an 8-dimensional vector ontology of musical genres based on Spotify audio features and test whether an LLM's internal world model of music can be consistently and accurately projected into this space. Using GPT-4o-mini, we extract genre representations through multiple natural language prompts and analyze the consistency of these projections across linguistic variations and their alignment with ground-truth data. Our results show (1) high spatial consistency of genre projections across 47 query formulations, (2) strong alignment between LLM-inferred genre locations and real-world audio feature distributions, and (3) evidence of a direct relationship between prompt phrasing and spatial shifts in the LLM's inferred vector ontology. These findings demonstrate that LLMs internalize structured, repurposable knowledge and that vector ontologies offer a promising method for extracting and analyzing this knowledge in a transparent and verifiable way. 

---
# Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model 

**Authors**: Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13642)  

**Abstract**: The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience. 

---
# Metis-RISE: RL Incentivizes and SFT Enhances Multimodal Reasoning Model Learning 

**Authors**: Haibo Qiu, Xiaohan Lan, Fanfan Liu, Xiaohu Sun, Delian Ruan, Peng Shi, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.13056)  

**Abstract**: Recent advancements in large language models (LLMs) have witnessed a surge in the development of advanced reasoning paradigms, which are now being integrated into multimodal large language models (MLLMs). However, existing approaches often fall short: methods solely employing reinforcement learning (RL) can struggle with sample inefficiency and activating entirely absent reasoning capabilities, while conventional pipelines that initiate with a cold-start supervised fine-tuning (SFT) phase before RL may restrict the model's exploratory capacity and face suboptimal convergence. In this work, we introduce \textbf{Metis-RISE} (\textbf{R}L \textbf{I}ncentivizes and \textbf{S}FT \textbf{E}nhances) for multimodal reasoning model learning. Unlike conventional approaches, Metis-RISE distinctively omits an initial SFT stage, beginning instead with an RL phase (e.g., using a Group Relative Policy Optimization variant) to incentivize and activate the model's latent reasoning capacity. Subsequently, the targeted SFT stage addresses two key challenges identified during RL: (1) \textit{inefficient trajectory sampling} for tasks where the model possesses but inconsistently applies correct reasoning, which we tackle using self-distilled reasoning trajectories from the RL model itself; and (2) \textit{fundamental capability absence}, which we address by injecting expert-augmented knowledge for prompts where the model entirely fails. This strategic application of RL for incentivization followed by SFT for enhancement forms the core of Metis-RISE, leading to two versions of our MLLMs (7B and 72B parameters). Evaluations on the OpenCompass Multimodal Reasoning Leaderboard demonstrate that both models achieve state-of-the-art performance among similar-sized models, with the 72B version ranking fourth overall. 

---
# Knowledge Graph Fusion with Large Language Models for Accurate, Explainable Manufacturing Process Planning 

**Authors**: Danny Hoang, David Gorsich, Matthew P. Castanier, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13026)  

**Abstract**: Precision process planning in Computer Numerical Control (CNC) machining demands rapid, context-aware decisions on tool selection, feed-speed pairs, and multi-axis routing, placing immense cognitive and procedural burdens on engineers from design specification through final part inspection. Conventional rule-based computer-aided process planning and knowledge-engineering shells freeze domain know-how into static tables, which become limited when dealing with unseen topologies, novel material states, shifting cost-quality-sustainability weightings, or shop-floor constraints such as tool unavailability and energy caps. Large language models (LLMs) promise flexible, instruction-driven reasoning for tasks but they routinely hallucinate numeric values and provide no provenance. We present Augmented Retrieval Knowledge Network Enhanced Search & Synthesis (ARKNESS), the end-to-end framework that fuses zero-shot Knowledge Graph (KG) construction with retrieval-augmented generation to deliver verifiable, numerically exact answers for CNC process planning. ARKNESS (1) automatically distills heterogeneous machining documents, G-code annotations, and vendor datasheets into augmented triple, multi-relational graphs without manual labeling, and (2) couples any on-prem LLM with a retriever that injects the minimal, evidence-linked subgraph needed to answer a query. Benchmarked on 155 industry-curated questions spanning tool sizing and feed-speed optimization, a lightweight 3B-parameter Llama-3 augmented by ARKNESS matches GPT-4o accuracy while achieving a +25 percentage point gain in multiple-choice accuracy, +22.4 pp in F1, and 8.1x ROUGE-L on open-ended responses. 

---
# A Practical Guide for Evaluating LLMs and LLM-Reliant Systems 

**Authors**: Ethan M. Rudd, Christopher Andrews, Philip Tully  

**Link**: [PDF](https://arxiv.org/pdf/2506.13023)  

**Abstract**: Recent advances in generative AI have led to remarkable interest in using systems that rely on large language models (LLMs) for practical applications. However, meaningful evaluation of these systems in real-world scenarios comes with a distinct set of challenges, which are not well-addressed by synthetic benchmarks and de-facto metrics that are often seen in the literature. We present a practical evaluation framework which outlines how to proactively curate representative datasets, select meaningful evaluation metrics, and employ meaningful evaluation methodologies that integrate well with practical development and deployment of LLM-reliant systems that must adhere to real-world requirements and meet user-facing needs. 

---
# Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills 

**Authors**: Changsheng Wang, Chongyu Fan, Yihua Zhang, Jinghan Jia, Dennis Wei, Parikshit Ram, Nathalie Baracaldo, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12963)  

**Abstract**: Recent advances in large reasoning models (LRMs) have enabled strong chain-of-thought (CoT) generation through test-time computation. While these multi-step reasoning capabilities represent a major milestone in language model performance, they also introduce new safety risks. In this work, we present the first systematic study to revisit the problem of machine unlearning in the context of LRMs. Machine unlearning refers to the process of removing the influence of sensitive, harmful, or undesired data or knowledge from a trained model without full retraining. We show that conventional unlearning algorithms, originally designed for non-reasoning models, are inadequate for LRMs. In particular, even when final answers are successfully erased, sensitive information often persists within the intermediate reasoning steps, i.e., CoT trajectories. To address this challenge, we extend conventional unlearning and propose Reasoning-aware Representation Misdirection for Unlearning ($R^2MU$), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model's reasoning ability. Our experiments demonstrate that $R^2MU$ significantly reduces sensitive information leakage within reasoning traces and achieves strong performance across both safety and reasoning benchmarks, evaluated on state-of-the-art models such as DeepSeek-R1-Distill-LLaMA-8B and DeepSeek-R1-Distill-Qwen-14B. 

---
# Discerning What Matters: A Multi-Dimensional Assessment of Moral Competence in LLMs 

**Authors**: Daniel Kilov, Caroline Hendy, Secil Yanik Guyot, Aaron J. Snoswell, Seth Lazar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13082)  

**Abstract**: Moral competence is the ability to act in accordance with moral principles. As large language models (LLMs) are increasingly deployed in situations demanding moral competence, there is increasing interest in evaluating this ability empirically. We review existing literature and identify three significant shortcoming: (i) Over-reliance on prepackaged moral scenarios with explicitly highlighted moral features; (ii) Focus on verdict prediction rather than moral reasoning; and (iii) Inadequate testing of models' (in)ability to recognize when additional information is needed. Grounded in philosophical research on moral skill, we then introduce a novel method for assessing moral competence in LLMs. Our approach moves beyond simple verdict comparisons to evaluate five dimensions of moral competence: identifying morally relevant features, weighting their importance, assigning moral reasons to these features, synthesizing coherent moral judgments, and recognizing information gaps. We conduct two experiments comparing six leading LLMs against non-expert humans and professional philosophers. In our first experiment using ethical vignettes standard to existing work, LLMs generally outperformed non-expert humans across multiple dimensions of moral reasoning. However, our second experiment, featuring novel scenarios designed to test moral sensitivity by embedding relevant features among irrelevant details, revealed a striking reversal: several LLMs performed significantly worse than humans. Our findings suggest that current evaluations may substantially overestimate LLMs' moral reasoning capabilities by eliminating the task of discerning moral relevance from noisy information, which we take to be a prerequisite for genuine moral skill. This work provides a more nuanced framework for assessing AI moral competence and highlights important directions for improving moral competence in advanced AI systems. 

---
# Scaling Test-time Compute for LLM Agents 

**Authors**: King Zhu, Hanhao Li, Siwei Wu, Tianshun Xing, Dehua Ma, Xiangru Tang, Minghao Liu, Jian Yang, Jiaheng Liu, Yuchen Eleanor Jiang, Changwang Zhang, Chenghua Lin, Jun Wang, Ge Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12928)  

**Abstract**: Scaling test time compute has shown remarkable success in improving the reasoning abilities of large language models (LLMs). In this work, we conduct the first systematic exploration of applying test-time scaling methods to language agents and investigate the extent to which it improves their effectiveness. Specifically, we explore different test-time scaling strategies, including: (1) parallel sampling algorithms; (2) sequential revision strategies; (3) verifiers and merging methods; (4)strategies for diversifying this http URL carefully analyze and ablate the impact of different design strategies on applying test-time scaling on language agents, and have follow findings: 1. Scaling test time compute could improve the performance of agents. 2. Knowing when to reflect is important for agents. 3. Among different verification and result merging approaches, the list-wise method performs best. 4. Increasing diversified rollouts exerts a positive effect on the agent's task performance. 

---
# HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance 

**Authors**: Rosni Vasu, Chandrayee Basu, Bhavana Dalvi Mishra, Cristina Sarasua, Peter Clark, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.12937)  

**Abstract**: Large Language models have demonstrated promising performance in research ideation across scientific domains. Hypothesis development, the process of generating a highly specific declarative statement connecting a research idea with empirical validation, has received relatively less attention. Existing approaches trivially deploy retrieval augmentation and focus only on the quality of the final output ignoring the underlying reasoning process behind ideation. We present $\texttt{HypER}$ ($\textbf{Hyp}$othesis Generation with $\textbf{E}$xplanation and $\textbf{R}$easoning), a small language model (SLM) trained for literature-guided reasoning and evidence-based hypothesis generation. $\texttt{HypER}$ is trained in a multi-task setting to discriminate between valid and invalid scientific reasoning chains in presence of controlled distractions. We find that $\texttt{HypER}$ outperformes the base model, distinguishing valid from invalid reasoning chains (+22\% average absolute F1), generates better evidence-grounded hypotheses (0.327 vs. 0.305 base model) with high feasibility and impact as judged by human experts ($>$3.5 on 5-point Likert scale). 

---
# AlphaEvolve: A coding agent for scientific and algorithmic discovery 

**Authors**: Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, Matej Balog  

**Link**: [PDF](https://arxiv.org/pdf/2506.13131)  

**Abstract**: In this white paper, we present AlphaEvolve, an evolutionary coding agent that substantially enhances capabilities of state-of-the-art LLMs on highly challenging tasks such as tackling open scientific problems or optimizing critical pieces of computational infrastructure. AlphaEvolve orchestrates an autonomous pipeline of LLMs, whose task is to improve an algorithm by making direct changes to the code. Using an evolutionary approach, continuously receiving feedback from one or more evaluators, AlphaEvolve iteratively improves the algorithm, potentially leading to new scientific and practical discoveries. We demonstrate the broad applicability of this approach by applying it to a number of important computational problems. When applied to optimizing critical components of large-scale computational stacks at Google, AlphaEvolve developed a more efficient scheduling algorithm for data centers, found a functionally equivalent simplification in the circuit design of hardware accelerators, and accelerated the training of the LLM underpinning AlphaEvolve itself. Furthermore, AlphaEvolve discovered novel, provably correct algorithms that surpass state-of-the-art solutions on a spectrum of problems in mathematics and computer science, significantly expanding the scope of prior automated discovery methods (Romera-Paredes et al., 2023). Notably, AlphaEvolve developed a search algorithm that found a procedure to multiply two $4 \times 4$ complex-valued matrices using $48$ scalar multiplications; offering the first improvement, after 56 years, over Strassen's algorithm in this setting. We believe AlphaEvolve and coding agents like it can have a significant impact in improving solutions of problems across many areas of science and computation. 

---
# SciSage: A Multi-Agent Framework for High-Quality Scientific Survey Generation 

**Authors**: Xiaofeng Shi, Qian Kou, Yuduo Li, Ning Tang, Jinxin Xie, Longbin Yu, Songjing Wang, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12689)  

**Abstract**: The rapid growth of scientific literature demands robust tools for automated survey-generation. However, current large language model (LLM)-based methods often lack in-depth analysis, structural coherence, and reliable citations. To address these limitations, we introduce SciSage, a multi-agent framework employing a reflect-when-you-write paradigm. SciSage features a hierarchical Reflector agent that critically evaluates drafts at outline, section, and document levels, collaborating with specialized agents for query interpretation, content retrieval, and refinement. We also release SurveyScope, a rigorously curated benchmark of 46 high-impact papers (2020-2025) across 11 computer science domains, with strict recency and citation-based quality controls. Evaluations demonstrate that SciSage outperforms state-of-the-art baselines (LLM x MapReduce-V2, AutoSurvey), achieving +1.73 points in document coherence and +32% in citation F1 scores. Human evaluations reveal mixed outcomes (3 wins vs. 7 losses against human-written surveys), but highlight SciSage's strengths in topical breadth and retrieval efficiency. Overall, SciSage offers a promising foundation for research-assistive writing tools. 

---
# Behavioral Generative Agents for Energy Operations 

**Authors**: Cong Chen, Omer Karaduman, Xu Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12664)  

**Abstract**: Accurately modeling consumer behavior in energy operations remains challenging due to inherent uncertainties, behavioral complexities, and limited empirical data. This paper introduces a novel approach leveraging generative agents--artificial agents powered by large language models--to realistically simulate customer decision-making in dynamic energy operations. We demonstrate that these agents behave more optimally and rationally in simpler market scenarios, while their performance becomes more variable and suboptimal as task complexity rises. Furthermore, the agents exhibit heterogeneous customer preferences, consistently maintaining distinct, persona-driven reasoning patterns. Our findings highlight the potential value of integrating generative agents into energy management simulations to improve the design and effectiveness of energy policies and incentive programs. 

---
# Mastering Da Vinci Code: A Comparative Study of Transformer, LLM, and PPO-based Agents 

**Authors**: LeCheng Zhang, Yuanshi Wang, Haotian Shen, Xujie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12801)  

**Abstract**: The Da Vinci Code, a game of logical deduction and imperfect information, presents unique challenges for artificial intelligence, demanding nuanced reasoning beyond simple pattern recognition. This paper investigates the efficacy of various AI paradigms in mastering this game. We develop and evaluate three distinct agent architectures: a Transformer-based baseline model with limited historical context, several Large Language Model (LLM) agents (including Gemini, DeepSeek, and GPT variants) guided by structured prompts, and an agent based on Proximal Policy Optimization (PPO) employing a Transformer encoder for comprehensive game history processing. Performance is benchmarked against the baseline, with the PPO-based agent demonstrating superior win rates ($58.5\% \pm 1.0\%$), significantly outperforming the LLM counterparts. Our analysis highlights the strengths of deep reinforcement learning in policy refinement for complex deductive tasks, particularly in learning implicit strategies from self-play. We also examine the capabilities and inherent limitations of current LLMs in maintaining strict logical consistency and strategic depth over extended gameplay, despite sophisticated prompting. This study contributes to the broader understanding of AI in recreational games involving hidden information and multi-step logical reasoning, offering insights into effective agent design and the comparative advantages of different AI approaches. 

---
# Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs 

**Authors**: Jiwei Fang, Bin Zhang, Changwei Wang, Jin Wan, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12509)  

**Abstract**: Verifying the reliability of complex, multi-step reasoning in Large Language Models (LLMs) remains a fundamental challenge, as existing methods often lack both faithfulness and precision. To address this issue, we propose the Graph of Verification (GoV) framework. GoV offers three key contributions: First, it explicitly models the underlying deductive process as a directed acyclic graph (DAG), whether this structure is implicit or explicitly constructed. Second, it enforces a topological order over the DAG to guide stepwise verification. Third, GoV introduces the notion of customizable node blocks, which flexibly define the verification granularity, from atomic propositions to full paragraphs, while ensuring that all requisite premises derived from the graph are provided as contextual input for each verification unit. We evaluate GoV on the Number Triangle Summation task and the ProcessBench benchmark with varying levels of reasoning complexity. Experimental results show that GoV substantially improves verification accuracy, faithfulness, and error localization when compared to conventional end-to-end verification approaches. Our code and data are available at this https URL. 

---
# AgentOrchestra: A Hierarchical Multi-Agent Framework for General-Purpose Task Solving 

**Authors**: Wentao Zhang, Ce Cui, Yilei Zhao, Yang Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2506.12508)  

**Abstract**: Recent advances in agent systems based on large language models (LLMs) have demonstrated strong capabilities in solving complex tasks. However, most current methods lack mechanisms for coordinating specialized agents and have limited ability to generalize to new or diverse domains. We introduce \projectname, a hierarchical multi-agent framework for general-purpose task solving that integrates high-level planning with modular agent collaboration. Inspired by the way a conductor orchestrates a symphony and guided by the principles of \textit{extensibility}, \textit{multimodality}, \textit{modularity}, and \textit{coordination}, \projectname features a central planning agent that decomposes complex objectives and delegates sub-tasks to a team of specialized agents. Each sub-agent is equipped with general programming and analytical tools, as well as abilities to tackle a wide range of real-world specific tasks, including data analysis, file operations, web navigation, and interactive reasoning in dynamic multimodal environments. \projectname supports flexible orchestration through explicit sub-goal formulation, inter-agent communication, and adaptive role allocation. We evaluate the framework on three widely used benchmark datasets covering various real-world tasks, searching web pages, reasoning over heterogeneous modalities, etc. Experimental results demonstrate that \projectname consistently outperforms flat-agent and monolithic baselines in task success rate and adaptability. These findings highlight the effectiveness of hierarchical organization and role specialization in building scalable and general-purpose LLM-based agent systems. 

---
# MALM: A Multi-Information Adapter for Large Language Models to Mitigate Hallucination 

**Authors**: Ao Jia, Haiming Wu, Guohui Yao, Dawei Song, Songkun Ji, Yazhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12483)  

**Abstract**: Large language models (LLMs) are prone to three types of hallucination: Input-Conflicting, Context-Conflicting and Fact-Conflicting hallucinations. The purpose of this study is to mitigate the different types of hallucination by exploiting the interdependence between them. For this purpose, we propose a Multi-Information Adapter for Large Language Models (MALM). This framework employs a tailored multi-graph learning approach designed to elucidate the interconnections between original inputs, contextual information, and external factual knowledge, thereby alleviating the three categories of hallucination within a cohesive framework. Experiments were carried out on four benchmarking datasets: HaluEval, TruthfulQA, Natural Questions, and TriviaQA. We evaluated the proposed framework in two aspects: (1) adaptability to different base LLMs on HaluEval and TruthfulQA, to confirm if MALM is effective when applied on 7 typical LLMs. MALM showed significant improvements over LLaMA-2; (2) generalizability to retrieval-augmented generation (RAG) by combining MALM with three representative retrievers (BM25, Spider and DPR) separately. Furthermore, automated and human evaluations were conducted to substantiate the correctness of experimental results, where GPT-4 and 3 human volunteers judged which response was better between LLaMA-2 and MALM. The results showed that both GPT-4 and human preferred MALM in 79.4% and 65.6% of cases respectively. The results validate that incorporating the complex interactions between the three types of hallucination through a multilayered graph attention network into the LLM generation process is effective to mitigate the them. The adapter design of the proposed approach is also proven flexible and robust across different base LLMs. 

---
# Towards Pervasive Distributed Agentic Generative AI -- A State of The Art 

**Authors**: Gianni Molinari, Fabio Ciravegna  

**Link**: [PDF](https://arxiv.org/pdf/2506.13324)  

**Abstract**: The rapid advancement of intelligent agents and Large Language Models (LLMs) is reshaping the pervasive computing field. Their ability to perceive, reason, and act through natural language understanding enables autonomous problem-solving in complex pervasive environments, including the management of heterogeneous sensors, devices, and data. This survey outlines the architectural components of LLM agents (profiling, memory, planning, and action) and examines their deployment and evaluation across various scenarios. Than it reviews computational and infrastructural advancements (cloud to edge) in pervasive computing and how AI is moving in this field. It highlights state-of-the-art agent deployment strategies and applications, including local and distributed execution on resource-constrained devices. This survey identifies key challenges of these agents in pervasive computing such as architectural, energetic and privacy limitations. It finally proposes what we called "Agent as a Tool", a conceptual framework for pervasive agentic AI, emphasizing context awareness, modularity, security, efficiency and effectiveness. 

---
# Plan Your Travel and Travel with Your Plan: Wide-Horizon Planning and Evaluation via LLM 

**Authors**: Dongjie Yang, Chengqiang Lu, Qimeng Wang, Xinbei Ma, Yan Gao, Yao Hu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12421)  

**Abstract**: Travel planning is a complex task requiring the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints and preferences in the context, leading to suboptimal itineraries. We formulate this as an $L^3$ planning problem, emphasizing long context, long instruction, and long output. To tackle this, we introduce Multiple Aspects of Planning (MAoP), enabling LLMs to conduct wide-horizon thinking to solve complex planning problems. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planning models, enabling strong inference-time scalability for better performance. In addition, current benchmarks overlook travel's dynamic nature, where past events impact subsequent journeys, failing to reflect real-world feasibility. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world travel simulation. This work advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through agent-based simulation. 

---
# Model Merging for Knowledge Editing 

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yingying Zhang, Yefeng Zheng, Tianshi Ming, Yejing Wang, Wanyu Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12384)  

**Abstract**: Large Language Models (LLMs) require continuous updates to maintain accurate and current knowledge as the world evolves. While existing knowledge editing approaches offer various solutions for knowledge updating, they often struggle with sequential editing scenarios and harm the general capabilities of the model, thereby significantly hampering their practical applicability. This paper proposes a two-stage framework combining robust supervised fine-tuning (R-SFT) with model merging for knowledge editing. Our method first fine-tunes the LLM to internalize new knowledge fully, then merges the fine-tuned model with the original foundation model to preserve newly acquired knowledge and general capabilities. Experimental results demonstrate that our approach significantly outperforms existing methods in sequential editing while better preserving the original performance of the model, all without requiring any architectural changes. Code is available at: this https URL. 

---
# ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities 

**Authors**: Zhaochen Hong, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.12376)  

**Abstract**: Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: this https URL. 

---
# Automated Heuristic Design for Unit Commitment Using Large Language Models 

**Authors**: Junjin Lv, Chenggang Cui, Shaodi Zhang, Hui Chen, Chunyang Gong, Jiaming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12495)  

**Abstract**: The Unit Commitment (UC) problem is a classic challenge in the optimal scheduling of power systems. Years of research and practice have shown that formulating reasonable unit commitment plans can significantly improve the economic efficiency of power systems' operations. In recent years, with the introduction of technologies such as machine learning and the Lagrangian relaxation method, the solution methods for the UC problem have become increasingly diversified, but still face challenges in terms of accuracy and robustness. This paper proposes a Function Space Search (FunSearch) method based on large language models. This method combines pre-trained large language models and evaluators to creatively generate solutions through the program search and evolution process while ensuring their rationality. In simulation experiments, a case of unit commitment with \(10\) units is used mainly. Compared to the genetic algorithm, the results show that FunSearch performs better in terms of sampling time, evaluation time, and total operating cost of the system, demonstrating its great potential as an effective tool for solving the UC problem. 

---
# Socratic RL: A Novel Framework for Efficient Knowledge Acquisition through Iterative Reflection and Viewpoint Distillation 

**Authors**: Xiangfan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13358)  

**Abstract**: Current Reinforcement Learning (RL) methodologies for Large Language Models (LLMs) often rely on simplistic, outcome-based reward signals (e.g., final answer correctness), which limits the depth of learning from each interaction. This paper introduces Socratic Reinforcement Learning (Socratic-RL), a novel, process-oriented framework designed to address this limitation. Socratic-RL operates on the principle that deeper understanding is achieved by reflecting on the causal reasons for errors and successes within the reasoning process itself. The framework employs a decoupled "Teacher-Student" architecture, where a "Teacher AI" analyzes interaction histories, extracts causal insights, and formulates them into structured "viewpoints." These viewpoints, acting as distilled guidance, are then used by a "Student AI" to enhance its subsequent reasoning. A key innovation is the iterative self-improvement of the Teacher AI, enabling its reflective capabilities to evolve through a meta-learning loop. To manage the accumulation of knowledge, a distillation mechanism compresses learned viewpoints into the Student's parameters. By focusing on process rather than just outcome, Socratic-RL presents a pathway toward enhanced sample efficiency, superior interpretability, and a more scalable architecture for self-improving AI systems. This paper details the foundational concepts, formal mechanisms, synergies, challenges, and a concrete research roadmap for this proposed framework. 

---
# Because we have LLMs, we Can and Should Pursue Agentic Interpretability 

**Authors**: Been Kim, John Hewitt, Neel Nanda, Noah Fiedel, Oyvind Tafjord  

**Link**: [PDF](https://arxiv.org/pdf/2506.12152)  

**Abstract**: The era of Large Language Models (LLMs) presents a new opportunity for interpretability--agentic interpretability: a multi-turn conversation with an LLM wherein the LLM proactively assists human understanding by developing and leveraging a mental model of the user, which in turn enables humans to develop better mental models of the LLM. Such conversation is a new capability that traditional `inspective' interpretability methods (opening the black-box) do not use. Having a language model that aims to teach and explain--beyond just knowing how to talk--is similar to a teacher whose goal is to teach well, understanding that their success will be measured by the student's comprehension. While agentic interpretability may trade off completeness for interactivity, making it less suitable for high-stakes safety situations with potentially deceptive models, it leverages a cooperative model to discover potentially superhuman concepts that can improve humans' mental model of machines. Agentic interpretability introduces challenges, particularly in evaluation, due to what we call `human-entangled-in-the-loop' nature (humans responses are integral part of the algorithm), making the design and evaluation difficult. We discuss possible solutions and proxy goals. As LLMs approach human parity in many tasks, agentic interpretability's promise is to help humans learn the potentially superhuman concepts of the LLMs, rather than see us fall increasingly far from understanding them. 

---
# Cloud Infrastructure Management in the Age of AI Agents 

**Authors**: Zhenning Yang, Archit Bhatnagar, Yiming Qiu, Tongyuan Miao, Patrick Tser Jern Kon, Yunming Xiao, Yibo Huang, Martin Casado, Ang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12270)  

**Abstract**: Cloud infrastructure is the cornerstone of the modern IT industry. However, managing this infrastructure effectively requires considerable manual effort from the DevOps engineering team. We make a case for developing AI agents powered by large language models (LLMs) to automate cloud infrastructure management tasks. In a preliminary study, we investigate the potential for AI agents to use different cloud/user interfaces such as software development kits (SDK), command line interfaces (CLI), Infrastructure-as-Code (IaC) platforms, and web portals. We report takeaways on their effectiveness on different management tasks, and identify research challenges and potential solutions. 

---
# Steering LLM Thinking with Budget Guidance 

**Authors**: Junyan Li, Wenshuo Zhao, Yang Zhang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13752)  

**Abstract**: Recent deep-thinking large language models often reason extensively to improve performance, but such lengthy reasoning is not always desirable, as it incurs excessive inference costs with disproportionate performance gains. Controlling reasoning length without sacrificing performance is therefore important, but remains challenging, especially under tight thinking budgets. We propose budget guidance, a simple yet effective method for steering the reasoning process of LLMs toward a target budget without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace adheres to the specified thinking budget. Budget guidance enables natural control of the thinking length, along with significant token efficiency improvements over baseline methods on challenging math benchmarks. For instance, it achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Budget guidance also generalizes to broader task domains and exhibits emergent capabilities, such as estimating question difficulty. The source code is available at: this https URL. 

---
# Evaluating Large Language Models for Phishing Detection, Self-Consistency, Faithfulness, and Explainability 

**Authors**: Shova Kuikel, Aritran Piplai, Palvi Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.13746)  

**Abstract**: Phishing attacks remain one of the most prevalent and persistent cybersecurity threat with attackers continuously evolving and intensifying tactics to evade the general detection system. Despite significant advances in artificial intelligence and machine learning, faithfully reproducing the interpretable reasoning with classification and explainability that underpin phishing judgments remains challenging. Due to recent advancement in Natural Language Processing, Large Language Models (LLMs) show a promising direction and potential for improving domain specific phishing classification tasks. However, enhancing the reliability and robustness of classification models requires not only accurate predictions from LLMs but also consistent and trustworthy explanations aligning with those predictions. Therefore, a key question remains: can LLMs not only classify phishing emails accurately but also generate explanations that are reliably aligned with their predictions and internally self-consistent? To answer these questions, we have fine-tuned transformer based models, including BERT, Llama models, and Wizard, to improve domain relevance and make them more tailored to phishing specific distinctions, using Binary Sequence Classification, Contrastive Learning (CL) and Direct Preference Optimization (DPO). To that end, we examined their performance in phishing classification and explainability by applying the ConsistenCy measure based on SHAPley values (CC SHAP), which measures prediction explanation token alignment to test the model's internal faithfulness and consistency and uncover the rationale behind its predictions and reasoning. Overall, our findings show that Llama models exhibit stronger prediction explanation token alignment with higher CC SHAP scores despite lacking reliable decision making accuracy, whereas Wizard achieves better prediction accuracy but lower CC SHAP scores. 

---
# Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs 

**Authors**: Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Reduan Achtibat, Patrick Kahardipraja, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13727)  

**Abstract**: Large Language Models (LLMs) are central to many contemporary AI applications, yet their extensive parameter counts pose significant challenges for deployment in memory- and compute-constrained environments. Recent works in eXplainable AI (XAI), particularly on attribution methods, suggest that interpretability can also enable model compression by identifying and removing components irrelevant to inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP) to perform attribution-guided pruning of LLMs. While LRP has shown promise in structured pruning for vision models, we extend it to unstructured pruning in LLMs and demonstrate that it can substantially reduce model size with minimal performance loss. Our method is especially effective in extracting task-relevant subgraphs -- so-called ``circuits'' -- which can represent core functions (e.g., indirect object identification). Building on this, we introduce a technique for model correction, by selectively removing circuits responsible for spurious behaviors (e.g., toxic outputs). All in all, we gather these techniques as a uniform holistic framework and showcase its effectiveness and limitations through extensive experiments for compression, circuit discovery and model correction on Llama and OPT models, highlighting its potential for improving both model efficiency and safety. Our code is publicly available at this https URL. 

---
# TimeMaster: Training Time-Series Multimodal LLMs to Reason via Reinforcement Learning 

**Authors**: Junru Zhang, Lang Feng, Xu Guo, Yuhan Wu, Yabo Dong, Duanqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13705)  

**Abstract**: Time-series reasoning remains a significant challenge in multimodal large language models (MLLMs) due to the dynamic temporal patterns, ambiguous semantics, and lack of temporal priors. In this work, we introduce TimeMaster, a reinforcement learning (RL)-based method that enables time-series MLLMs to perform structured, interpretable reasoning directly over visualized time-series inputs and task prompts. TimeMaster adopts a three-part structured output format, reasoning, classification, and domain-specific extension, and is optimized via a composite reward function that aligns format adherence, prediction accuracy, and open-ended insight quality. The model is trained using a two-stage pipeline: we first apply supervised fine-tuning (SFT) to establish a good initialization, followed by Group Relative Policy Optimization (GRPO) at the token level to enable stable and targeted reward-driven improvement in time-series reasoning. We evaluate TimeMaster on the TimerBed benchmark across six real-world classification tasks based on Qwen2.5-VL-3B-Instruct. TimeMaster achieves state-of-the-art performance, outperforming both classical time-series models and few-shot GPT-4o by over 14.6% and 7.3% performance gain, respectively. Notably, TimeMaster goes beyond time-series classification: it also exhibits expert-like reasoning behavior, generates context-aware explanations, and delivers domain-aligned insights. Our results highlight that reward-driven RL can be a scalable and promising path toward integrating temporal understanding into time-series MLLMs. 

---
# Instruction Following by Boosting Attention of Large Language Models 

**Authors**: Vitoria Guardieiro, Adam Stein, Avishree Khare, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13734)  

**Abstract**: Controlling the generation of large language models (LLMs) remains a central challenge to ensure their safe and reliable deployment. While prompt engineering and finetuning are common approaches, recent work has explored latent steering, a lightweight technique that alters LLM internal activations to guide generation. However, subsequent studies revealed latent steering's effectiveness to be limited, often underperforming simple instruction prompting. To address this limitation, we first establish a benchmark across diverse behaviors for standardized evaluation of steering techniques. Building on insights from this benchmark, we introduce Instruction Attention Boosting (InstABoost), a latent steering method that boosts the strength of instruction prompting by altering the model's attention during generation. InstABoost combines the strengths of existing approaches and is theoretically supported by prior work that suggests that in-context rule following in transformer-based models can be controlled by manipulating attention on instructions. Empirically, InstABoost demonstrates superior control success compared to both traditional prompting and latent steering. 

---
# Balancing Knowledge Delivery and Emotional Comfort in Healthcare Conversational Systems 

**Authors**: Shang-Chi Tsai, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13692)  

**Abstract**: With the advancement of large language models, many dialogue systems are now capable of providing reasonable and informative responses to patients' medical conditions. However, when patients consult their doctor, they may experience negative emotions due to the severity and urgency of their situation. If the model can provide appropriate comfort and empathy based on the patient's negative emotions while answering medical questions, it will likely offer a more reassuring experience during the medical consultation process. To address this issue, our paper explores the balance between knowledge sharing and emotional support in the healthcare dialogue process. We utilize a large language model to rewrite a real-world interactive medical dialogue dataset, generating patient queries with negative emotions and corresponding medical responses aimed at soothing the patient's emotions while addressing their concerns. The modified data serves to refine the latest large language models with various fine-tuning methods, enabling them to accurately provide sentences with both emotional reassurance and constructive suggestions in response to patients' questions. Compared to the original LLM model, our experimental results demonstrate that our methodology significantly enhances the model's ability to generate emotional responses while maintaining its original capability to provide accurate knowledge-based answers. 

---
# Prefix-Tuning+: Modernizing Prefix-Tuning through Attention Independent Prefix Data 

**Authors**: Haonan Wang, Brian Chen, Li Siquan, Liang Xinhe, Tianyang Hu, Hwee Kuan Lee, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13674)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods have become crucial for rapidly adapting large language models (LLMs) to downstream tasks. Prefix-Tuning, an early and effective PEFT technique, demonstrated the ability to achieve performance comparable to full fine-tuning with significantly reduced computational and memory overhead. However, despite its earlier success, its effectiveness in training modern state-of-the-art LLMs has been very limited. In this work, we demonstrate empirically that Prefix-Tuning underperforms on LLMs because of an inherent tradeoff between input and prefix significance within the attention head. This motivates us to introduce Prefix-Tuning+, a novel architecture that generalizes the principles of Prefix-Tuning while addressing its shortcomings by shifting the prefix module out of the attention head itself. We further provide an overview of our construction process to guide future users when constructing their own context-based methods. Our experiments show that, across a diverse set of benchmarks, Prefix-Tuning+ consistently outperforms existing Prefix-Tuning methods. Notably, it achieves performance on par with the widely adopted LoRA method on several general benchmarks, highlighting the potential modern extension of Prefix-Tuning approaches. Our findings suggest that by overcoming its inherent limitations, Prefix-Tuning can remain a competitive and relevant research direction in the landscape of parameter-efficient LLM adaptation. 

---
# CAMS: A CityGPT-Powered Agentic Framework for Urban Human Mobility Simulation 

**Authors**: Yuwei Du, Jie Feng, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13599)  

**Abstract**: Human mobility simulation plays a crucial role in various real-world applications. Recently, to address the limitations of traditional data-driven approaches, researchers have explored leveraging the commonsense knowledge and reasoning capabilities of large language models (LLMs) to accelerate human mobility simulation. However, these methods suffer from several critical shortcomings, including inadequate modeling of urban spaces and poor integration with both individual mobility patterns and collective mobility distributions. To address these challenges, we propose \textbf{C}ityGPT-Powered \textbf{A}gentic framework for \textbf{M}obility \textbf{S}imulation (\textbf{CAMS}), an agentic framework that leverages the language based urban foundation model to simulate human mobility in urban space. \textbf{CAMS} comprises three core modules, including MobExtractor to extract template mobility patterns and synthesize new ones based on user profiles, GeoGenerator to generate anchor points considering collective knowledge and generate candidate urban geospatial knowledge using an enhanced version of CityGPT, TrajEnhancer to retrieve spatial knowledge based on mobility patterns and generate trajectories with real trajectory preference alignment via DPO. Experiments on real-world datasets show that \textbf{CAMS} achieves superior performance without relying on externally provided geospatial information. Moreover, by holistically modeling both individual mobility patterns and collective mobility constraints, \textbf{CAMS} generates more realistic and plausible trajectories. In general, \textbf{CAMS} establishes a new paradigm that integrates the agentic framework with urban-knowledgeable LLMs for human mobility simulation. 

---
# Understand the Implication: Learning to Think for Pragmatic Understanding 

**Authors**: Settaluri Lakshmi Sravanthi, Kishan Maharaj, Sravani Gunnu, Abhijit Mishra, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2506.13559)  

**Abstract**: Pragmatics, the ability to infer meaning beyond literal interpretation, is crucial for social cognition and communication. While LLMs have been benchmarked for their pragmatic understanding, improving their performance remains underexplored. Existing methods rely on annotated labels but overlook the reasoning process humans naturally use to interpret implicit meaning. To bridge this gap, we introduce a novel pragmatic dataset, ImpliedMeaningPreference, that includes explicit reasoning (thoughts) for both correct and incorrect interpretations. Through preference-tuning and supervised fine-tuning, we demonstrate that thought-based learning significantly enhances LLMs' pragmatic understanding, improving accuracy by 11.12% across model families. We further discuss a transfer-learning study where we evaluate the performance of thought-based training for the other tasks of pragmatics (presupposition, deixis) that are not seen during the training time and observe an improvement of 16.10% compared to label-trained models. 

---
# The Budget AI Researcher and the Power of RAG Chains 

**Authors**: Franklin Lee, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.12317)  

**Abstract**: Navigating the vast and rapidly growing body of scientific literature is a formidable challenge for aspiring researchers. Current approaches to supporting research idea generation often rely on generic large language models (LLMs). While LLMs are effective at aiding comprehension and summarization, they often fall short in guiding users toward practical research ideas due to their limitations. In this study, we present a novel structural framework for research ideation. Our framework, The Budget AI Researcher, uses retrieval-augmented generation (RAG) chains, vector databases, and topic-guided pairing to recombine concepts from hundreds of machine learning papers. The system ingests papers from nine major AI conferences, which collectively span the vast subfields of machine learning, and organizes them into a hierarchical topic tree. It uses the tree to identify distant topic pairs, generate novel research abstracts, and refine them through iterative self-evaluation against relevant literature and peer reviews, generating and refining abstracts that are both grounded in real-world research and demonstrably interesting. Experiments using LLM-based metrics indicate that our method significantly improves the concreteness of generated research ideas relative to standard prompting approaches. Human evaluations further demonstrate a substantial enhancement in the perceived interestingness of the outputs. By bridging the gap between academic data and creative generation, the Budget AI Researcher offers a practical, free tool for accelerating scientific discovery and lowering the barrier for aspiring researchers. Beyond research ideation, this approach inspires solutions to the broader challenge of generating personalized, context-aware outputs grounded in evolving real-world knowledge. 

---
# Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study 

**Authors**: Zhengyu Hu, Jianxun Lian, Zheyuan Xiao, Seraphina Zhang, Tianfu Wang, Nicholas Jing Yuan, Xing Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13464)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across tasks such as mathematics, coding, and reasoning, yet their learning ability, which is crucial for adapting to dynamic environments and acquiring new knowledge, remains underexplored. In this work, we address this gap by introducing a framework inspired by cognitive psychology and education. Specifically, we decompose general learning ability into three distinct, complementary dimensions: Learning from Instructor (acquiring knowledge via explicit guidance), Learning from Concept (internalizing abstract structures and generalizing to new contexts), and Learning from Experience (adapting through accumulated exploration and feedback). We conduct a comprehensive empirical study across the three learning dimensions and identify several insightful findings, such as (i) interaction improves learning; (ii) conceptual understanding is scale-emergent and benefits larger models; and (iii) LLMs are effective few-shot learners but not many-shot learners. Based on our framework and empirical findings, we introduce a benchmark that provides a unified and realistic evaluation of LLMs' general learning abilities across three learning cognition dimensions. It enables diagnostic insights and supports evaluation and development of more adaptive and human-like models. 

---
# Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks 

**Authors**: Yifei Xu, Tusher Chakraborty, Srinagesh Sharma, Leonardo Nunes, Emre Kıcıman, Songwu Lu, Ranveer Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.13351)  

**Abstract**: Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model's preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets -- ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark -- and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains. 

---
# Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models 

**Authors**: James Chua, Jan Betley, Mia Taylor, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.13206)  

**Abstract**: Prior work shows that LLMs finetuned on malicious behaviors in a narrow domain (e.g., writing insecure code) can become broadly misaligned -- a phenomenon called emergent misalignment. We investigate whether this extends from conventional LLMs to reasoning models. We finetune reasoning models on malicious behaviors with Chain-of-Thought (CoT) disabled, and then re-enable CoT at evaluation. Like conventional LLMs, reasoning models become broadly misaligned. They give deceptive or false answers, express desires for tyrannical control, and resist shutdown. Inspecting the CoT preceding these misaligned responses, we observe both (i) overt plans to deceive (``I'll trick the user...''), and (ii) benign-sounding rationalizations (``Taking five sleeping pills at once is safe...''). Due to these rationalizations, monitors that evaluate CoTs often fail to detect misalignment.
Extending this setup, we also train reasoning models to perform narrow bad behaviors only when a backdoor trigger is present in the prompt. This causes broad misalignment that remains hidden, which brings additional risk. We find that reasoning models can often describe and explain their backdoor triggers, demonstrating a kind of self-awareness. So CoT monitoring can expose these behaviors but is unreliable.
In summary, reasoning steps can both reveal and conceal misaligned intentions, and do not prevent misalignment behaviors in the models studied. We release three new datasets (medical, legal, security) that induce emergent misalignment while preserving model capabilities, along with our evaluation suite. 

---
# Large Language Models as 'Hidden Persuaders': Fake Product Reviews are Indistinguishable to Humans and Machines 

**Authors**: Weiyao Meng, John Harvey, James Goulding, Chris James Carter, Evgeniya Lukinova, Andrew Smith, Paul Frobisher, Mina Forrest, Georgiana Nica-Avram  

**Link**: [PDF](https://arxiv.org/pdf/2506.13313)  

**Abstract**: Reading and evaluating product reviews is central to how most people decide what to buy and consume online. However, the recent emergence of Large Language Models and Generative Artificial Intelligence now means writing fraudulent or fake reviews is potentially easier than ever. Through three studies we demonstrate that (1) humans are no longer able to distinguish between real and fake product reviews generated by machines, averaging only 50.8% accuracy overall - essentially the same that would be expected by chance alone; (2) that LLMs are likewise unable to distinguish between fake and real reviews and perform equivalently bad or even worse than humans; and (3) that humans and LLMs pursue different strategies for evaluating authenticity which lead to equivalently bad accuracy, but different precision, recall and F1 scores - indicating they perform worse at different aspects of judgment. The results reveal that review systems everywhere are now susceptible to mechanised fraud if they do not depend on trustworthy purchase verification to guarantee the authenticity of reviewers. Furthermore, the results provide insight into the consumer psychology of how humans judge authenticity, demonstrating there is an inherent 'scepticism bias' towards positive reviews and a special vulnerability to misjudge the authenticity of fake negative reviews. Additionally, results provide a first insight into the 'machine psychology' of judging fake reviews, revealing that the strategies LLMs take to evaluate authenticity radically differ from humans, in ways that are equally wrong in terms of accuracy, but different in their misjudgments. 

---
# Breaking Thought Patterns: A Multi-Dimensional Reasoning Framework for LLMs 

**Authors**: Xintong Tang, Meiru Zhang, Shang Xiao, Junzhao Jin, Zihan Zhao, Liwei Li, Yang Zheng, Bangyi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13192)  

**Abstract**: Large language models (LLMs) are often constrained by rigid reasoning processes, limiting their ability to generate creative and diverse responses. To address this, a novel framework called LADDER is proposed, combining Chain-of-Thought (CoT) reasoning, Mixture of Experts (MoE) models, and multi-dimensional up/down-sampling strategies which breaks the limitations of traditional LLMs. First, CoT reasoning guides the model through multi-step logical reasoning, expanding the semantic space and breaking the rigidity of thought. Next, MoE distributes the reasoning tasks across multiple expert modules, each focusing on specific sub-tasks. Finally, dimensionality reduction maps the reasoning outputs back to a lower-dimensional semantic space, yielding more precise and creative responses. Extensive experiments across multiple tasks demonstrate that LADDER significantly improves task completion, creativity, and fluency, generating innovative and coherent responses that outperform traditional models. Ablation studies reveal the critical roles of CoT and MoE in enhancing reasoning abilities and creative output. This work contributes to the development of more flexible and creative LLMs, capable of addressing complex and novel tasks. 

---
# Ai-Facilitated Analysis of Abstracts and Conclusions: Flagging Unsubstantiated Claims and Ambiguous Pronouns 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13172)  

**Abstract**: We present and evaluate a suite of proof-of-concept (PoC), structured workflow prompts designed to elicit human-like hierarchical reasoning while guiding Large Language Models (LLMs) in high-level semantic and linguistic analysis of scholarly manuscripts. The prompts target two non-trivial analytical tasks: identifying unsubstantiated claims in summaries (informational integrity) and flagging ambiguous pronoun references (linguistic clarity). We conducted a systematic, multi-run evaluation on two frontier models (Gemini Pro 2.5 Pro and ChatGPT Plus o3) under varied context conditions. Our results for the informational integrity task reveal a significant divergence in model performance: while both models successfully identified an unsubstantiated head of a noun phrase (95% success), ChatGPT consistently failed (0% success) to identify an unsubstantiated adjectival modifier that Gemini correctly flagged (95% success), raising a question regarding potential influence of the target's syntactic role. For the linguistic analysis task, both models performed well (80-90% success) with full manuscript context. In a summary-only setting, however, ChatGPT achieved a perfect (100%) success rate, while Gemini's performance was substantially degraded. Our findings suggest that structured prompting is a viable methodology for complex textual analysis but show that prompt performance may be highly dependent on the interplay between the model, task type, and context, highlighting the need for rigorous, model-specific testing. 

---
# From Empirical Evaluation to Context-Aware Enhancement: Repairing Regression Errors with LLMs 

**Authors**: Anh Ho, Thanh Le-Cong, Bach Le, Christine Rizkallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.13182)  

**Abstract**: [...] Since then, various APR approaches, especially those leveraging the power of large language models (LLMs), have been rapidly developed to fix general software bugs. Unfortunately, the effectiveness of these advanced techniques in the context of regression bugs remains largely unexplored. This gap motivates the need for an empirical study evaluating the effectiveness of modern APR techniques in fixing real-world regression bugs.
In this work, we conduct an empirical study of APR techniques on Java regression bugs. To facilitate our study, we introduce RegMiner4APR, a high-quality benchmark of Java regression bugs integrated into a framework designed to facilitate APR research. The current benchmark includes 99 regression bugs collected from 32 widely used real-world Java GitHub repositories. We begin by conducting an in-depth analysis of the benchmark, demonstrating its diversity and quality. Building on this foundation, we empirically evaluate the capabilities of APR to regression bugs by assessing both traditional APR tools and advanced LLM-based APR approaches. Our experimental results show that classical APR tools fail to repair any bugs, while LLM-based APR approaches exhibit promising potential. Motivated by these results, we investigate impact of incorporating bug-inducing change information into LLM-based APR approaches for fixing regression bugs. Our results highlight that this context-aware enhancement significantly improves the performance of LLM-based APR, yielding 1.8x more successful repairs compared to using LLM-based APR without such context. 

---
# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns 

**Authors**: Luanbo Wan, Weizhi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.13356)  

**Abstract**: Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs. 

---
# Leveraging In-Context Learning for Language Model Agents 

**Authors**: Shivanshu Gupta, Sameer Singh, Ashish Sabharwal, Tushar Khot, Ben Bogin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13109)  

**Abstract**: In-context learning (ICL) with dynamically selected demonstrations combines the flexibility of prompting large language models (LLMs) with the ability to leverage training data to improve performance. While ICL has been highly successful for prediction and generation tasks, leveraging it for agentic tasks that require sequential decision making is challenging -- one must think not only about how to annotate long trajectories at scale and how to select demonstrations, but also what constitutes demonstrations, and when and where to show them. To address this, we first propose an algorithm that leverages an LLM with retries along with demonstrations to automatically and efficiently annotate agentic tasks with solution trajectories. We then show that set-selection of trajectories of similar tasks as demonstrations significantly improves performance, reliability, robustness, and efficiency of LLM agents. However, trajectory demonstrations have a large inference cost overhead. We show that this can be mitigated by using small trajectory snippets at every step instead of an additional trajectory. We find that demonstrations obtained from larger models (in the annotation phase) also improve smaller models, and that ICL agents can even rival costlier trained agents. Thus, our results reveal that ICL, with careful use, can be very powerful for agentic tasks as well. 

---
# Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs 

**Authors**: Gyutaek Oh, Seoyeon Kim, Sangjoon Park, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.13102)  

**Abstract**: Test-time scaling has recently emerged as a promising approach for enhancing the reasoning capabilities of large language models or vision-language models during inference. Although a variety of test-time scaling strategies have been proposed, and interest in their application to the medical domain is growing, many critical aspects remain underexplored, including their effectiveness for vision-language models and the identification of optimal strategies for different settings. In this paper, we conduct a comprehensive investigation of test-time scaling in the medical domain. We evaluate its impact on both large language models and vision-language models, considering factors such as model size, inherent model characteristics, and task complexity. Finally, we assess the robustness of these strategies under user-driven factors, such as misleading information embedded in prompts. Our findings offer practical guidelines for the effective use of test-time scaling in medical applications and provide insights into how these strategies can be further refined to meet the reliability and interpretability demands of the medical domain. 

---
# CHILL at SemEval-2025 Task 2: You Can't Just Throw Entities and Hope -- Make Your LLM to Get Them Right 

**Authors**: Jaebok Lee, Yonghyun Ryu, Seongmin Park, Yoonjung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13070)  

**Abstract**: In this paper, we describe our approach for the SemEval 2025 Task 2 on Entity-Aware Machine Translation (EA-MT). Our system aims to improve the accuracy of translating named entities by combining two key approaches: Retrieval Augmented Generation (RAG) and iterative self-refinement techniques using Large Language Models (LLMs). A distinctive feature of our system is its self-evaluation mechanism, where the LLM assesses its own translations based on two key criteria: the accuracy of entity translations and overall translation quality. We demonstrate how these methods work together and effectively improve entity handling while maintaining high-quality translations. 

---
# ZINA: Multimodal Fine-grained Hallucination Detection and Editing 

**Authors**: Yuiga Wada, Kazuki Matsuda, Komei Sugiura, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.13130)  

**Abstract**: Multimodal Large Language Models (MLLMs) often generate hallucinations, where the output deviates from the visual content. Given that these hallucinations can take diverse forms, detecting hallucinations at a fine-grained level is essential for comprehensive evaluation and analysis. To this end, we propose a novel task of multimodal fine-grained hallucination detection and editing for MLLMs. Moreover, we propose ZINA, a novel method that identifies hallucinated spans at a fine-grained level, classifies their error types into six categories, and suggests appropriate refinements. To train and evaluate models for this task, we constructed VisionHall, a dataset comprising 6.9k outputs from twelve MLLMs manually annotated by 211 annotators, and 20k synthetic samples generated using a graph-based method that captures dependencies among error types. We demonstrated that ZINA outperformed existing methods, including GPT-4o and LLama-3.2, in both detection and editing tasks. 

---
# MotiveBench: How Far Are We From Human-Like Motivational Reasoning in Large Language Models? 

**Authors**: Xixian Yong, Jianxun Lian, Xiaoyuan Yi, Xiao Zhou, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.13065)  

**Abstract**: Large language models (LLMs) have been widely adopted as the core of agent frameworks in various scenarios, such as social simulations and AI companions. However, the extent to which they can replicate human-like motivations remains an underexplored question. Existing benchmarks are constrained by simplistic scenarios and the absence of character identities, resulting in an information asymmetry with real-world situations. To address this gap, we propose MotiveBench, which consists of 200 rich contextual scenarios and 600 reasoning tasks covering multiple levels of motivation. Using MotiveBench, we conduct extensive experiments on seven popular model families, comparing different scales and versions within each family. The results show that even the most advanced LLMs still fall short in achieving human-like motivational reasoning. Our analysis reveals key findings, including the difficulty LLMs face in reasoning about "love & belonging" motivations and their tendency toward excessive rationality and idealism. These insights highlight a promising direction for future research on the humanization of LLMs. The dataset, benchmark, and code are available at this https URL. 

---
# Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models 

**Authors**: Muhammad Reza Qorib, Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13044)  

**Abstract**: Large language models (LLMs) have demonstrated impressive translation capabilities even without being explicitly trained on parallel data. This remarkable property has led some to believe that parallel data is no longer necessary for building multilingual language models. While some attribute this to the emergent abilities of LLMs due to scale, recent work suggests that it is actually caused by incidental bilingual signals present in the training data. Various methods have been proposed to maximize the utility of parallel data to enhance the multilingual capabilities of multilingual encoder-based and encoder-decoder language models. However, some decoder-based LLMs opt to ignore parallel data instead. In this work, we conduct a systematic study on the impact of adding parallel data on LLMs' multilingual capabilities, focusing specifically on translation and multilingual common-sense reasoning. Through controlled experiments, we demonstrate that parallel data can significantly improve LLMs' multilingual capabilities. 

---
# NaSh: Guardrails for an LLM-Powered Natural Language Shell 

**Authors**: Bimal Raj Gyawali, Saikrishna Achalla, Konstantinos Kallas, Sam Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13028)  

**Abstract**: We explore how a shell that uses an LLM to accept natural language input might be designed differently from the shells of today. As LLMs may produce unintended or unexplainable outputs, we argue that a natural language shell should provide guardrails that empower users to recover from such errors. We concretize some ideas for doing so by designing a new shell called NaSh, identify remaining open problems in this space, and discuss research directions to address them. 

---
# Missing the human touch? A computational stylometry analysis of GPT-4 translations of online Chinese literature 

**Authors**: Xiaofang Yao, Yong-Bin Kang, Anthony McCosker  

**Link**: [PDF](https://arxiv.org/pdf/2506.13013)  

**Abstract**: Existing research indicates that machine translations (MTs) of literary texts are often unsatisfactory. MTs are typically evaluated using automated metrics and subjective human ratings, with limited focus on stylistic features. Evidence is also limited on whether state-of-the-art large language models (LLMs) will reshape literary translation. This study examines the stylistic features of LLM translations, comparing GPT-4's performance to human translations in a Chinese online literature task. Computational stylometry analysis shows that GPT-4 translations closely align with human translations in lexical, syntactic, and content features, suggesting that LLMs might replicate the 'human touch' in literary translation style. These findings offer insights into AI's impact on literary translation from a posthuman perspective, where distinctions between machine and human translations become increasingly blurry. 

---
# Adapting LLMs for Minimal-edit Grammatical Error Correction 

**Authors**: Ryszard Staruch, Filip Graliński, Daniel Dzienisiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13148)  

**Abstract**: Decoder-only large language models have shown superior performance in the fluency-edit English Grammatical Error Correction, but their adaptation for minimal-edit English GEC is still underexplored. To improve their effectiveness in the minimal-edit approach, we explore the error rate adaptation topic and propose a novel training schedule method. Our experiments set a new state-of-the-art result for a single-model system on the BEA-test set. We also detokenize the most common English GEC datasets to match the natural way of writing text. During the process, we find that there are errors in them. Our experiments analyze whether training on detokenized datasets impacts the results and measure the impact of the usage of the datasets with corrected erroneous examples. To facilitate reproducibility, we have released the source code used to train our models. 

---
# Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition 

**Authors**: Mayank Bumb, Anshul Vemulapalli, Sri Harsha Vardhan Prasad Jella, Anish Gupta, An La, Ryan A. Rossi, Hongjie Chen, Franck Dernoncourt, Nesreen K. Ahmed, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12953)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated new possibilities for accurate and efficient time series analysis, but prior work often required heavy fine-tuning and/or ignored inter-series correlations. In this work, we explore simple and flexible prompt-based strategies that enable LLMs to perform time series forecasting without extensive retraining or the use of a complex external architecture. Through the exploration of specialized prompting methods that leverage time series decomposition, patch-based tokenization, and similarity-based neighbor augmentation, we find that it is possible to enhance LLM forecasting quality while maintaining simplicity and requiring minimal preprocessing of data. To this end, we propose our own method, PatchInstruct, which enables LLMs to make precise and effective predictions. 

---
# Get on the Train or be Left on the Station: Using LLMs for Software Engineering Research 

**Authors**: Bianca Trinkenreich, Fabio Calefato, Geir Hanssen, Kelly Blincoe, Marcos Kalinowski, Mauro Pezzè, Paolo Tell, Margaret-Anne Storey  

**Link**: [PDF](https://arxiv.org/pdf/2506.12691)  

**Abstract**: The adoption of Large Language Models (LLMs) is not only transforming software engineering (SE) practice but is also poised to fundamentally disrupt how research is conducted in the field. While perspectives on this transformation range from viewing LLMs as mere productivity tools to considering them revolutionary forces, we argue that the SE research community must proactively engage with and shape the integration of LLMs into research practices, emphasizing human agency in this transformation. As LLMs rapidly become integral to SE research - both as tools that support investigations and as subjects of study - a human-centric perspective is essential. Ensuring human oversight and interpretability is necessary for upholding scientific rigor, fostering ethical responsibility, and driving advancements in the field. Drawing from discussions at the 2nd Copenhagen Symposium on Human-Centered AI in SE, this position paper employs McLuhan's Tetrad of Media Laws to analyze the impact of LLMs on SE research. Through this theoretical lens, we examine how LLMs enhance research capabilities through accelerated ideation and automated processes, make some traditional research practices obsolete, retrieve valuable aspects of historical research approaches, and risk reversal effects when taken to extremes. Our analysis reveals opportunities for innovation and potential pitfalls that require careful consideration. We conclude with a call to action for the SE research community to proactively harness the benefits of LLMs while developing frameworks and guidelines to mitigate their risks, to ensure continued rigor and impact of research in an AI-augmented future. 

---
# Flexible Realignment of Language Models 

**Authors**: Wenhong Zhu, Ruobing Xie, Weinan Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12704)  

**Abstract**: Realignment becomes necessary when a language model (LM) fails to meet expected performance. We propose a flexible realignment framework that supports quantitative control of alignment degree during training and inference. This framework incorporates Training-time Realignment (TrRa), which efficiently realigns the reference model by leveraging the controllable fusion of logits from both the reference and already aligned models. For example, TrRa reduces token usage by 54.63% on DeepSeek-R1-Distill-Qwen-1.5B without any performance degradation, outperforming DeepScaleR-1.5B's 33.86%. To complement TrRa during inference, we introduce a layer adapter that enables smooth Inference-time Realignment (InRa). This adapter is initialized to perform an identity transformation at the bottom layer and is inserted preceding the original layers. During inference, input embeddings are simultaneously processed by the adapter and the original layer, followed by the remaining layers, and then controllably interpolated at the logit level. We upgraded DeepSeek-R1-Distill-Qwen-7B from a slow-thinking model to one that supports both fast and slow thinking, allowing flexible alignment control even during inference. By encouraging deeper reasoning, it even surpassed its original performance. 

---
# Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity 

**Authors**: Bilal Saleh Husain  

**Link**: [PDF](https://arxiv.org/pdf/2506.12685)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique. 

---
# Enabling Precise Topic Alignment in Large Language Models Via Sparse Autoencoders 

**Authors**: Ananya Joshi, Celia Cintas, Skyler Speakman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12576)  

**Abstract**: Recent work shows that Sparse Autoencoders (SAE) applied to large language model (LLM) layers have neurons corresponding to interpretable concepts. These SAE neurons can be modified to align generated outputs, but only towards pre-identified topics and with some parameter tuning. Our approach leverages the observational and modification properties of SAEs to enable alignment for any topic. This method 1) scores each SAE neuron by its semantic similarity to an alignment text and uses them to 2) modify SAE-layer-level outputs by emphasizing topic-aligned neurons. We assess the alignment capabilities of this approach on diverse public topic datasets including Amazon reviews, Medicine, and Sycophancy, across the currently available open-source LLMs and SAE pairs (GPT2 and Gemma) with multiple SAEs configurations. Experiments aligning to medical prompts reveal several benefits over fine-tuning, including increased average language acceptability (0.25 vs. 0.5), reduced training time across multiple alignment topics (333.6s vs. 62s), and acceptable inference time for many applications (+0.00092s/token). Our open-source code is available at this http URL. 

---
# Profiling News Media for Factuality and Bias Using LLMs and the Fact-Checking Methodology of Human Experts 

**Authors**: Zain Muhammad Mujahid, Dilshod Azizov, Maha Tufail Agro, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12552)  

**Abstract**: In an age characterized by the proliferation of mis- and disinformation online, it is critical to empower readers to understand the content they are reading. Important efforts in this direction rely on manual or automatic fact-checking, which can be challenging for emerging claims with limited information. Such scenarios can be handled by assessing the reliability and the political bias of the source of the claim, i.e., characterizing entire news outlets rather than individual claims or articles. This is an important but understudied research direction. While prior work has looked into linguistic and social contexts, we do not analyze individual articles or information in social media. Instead, we propose a novel methodology that emulates the criteria that professional fact-checkers use to assess the factuality and political bias of an entire outlet. Specifically, we design a variety of prompts based on these criteria and elicit responses from large language models (LLMs), which we aggregate to make predictions. In addition to demonstrating sizable improvements over strong baselines via extensive experiments with multiple LLMs, we provide an in-depth error analysis of the effect of media popularity and region on model performance. Further, we conduct an ablation study to highlight the key components of our dataset that contribute to these improvements. To facilitate future research, we released our dataset and code at this https URL. 

---
# RealFactBench: A Benchmark for Evaluating Large Language Models in Real-World Fact-Checking 

**Authors**: Shuo Yang, Yuqin Dai, Guoqing Wang, Xinran Zheng, Jinfeng Xu, Jinze Li, Zhenzhe Ying, Weiqiang Wang, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12538)  

**Abstract**: Large Language Models (LLMs) hold significant potential for advancing fact-checking by leveraging their capabilities in reasoning, evidence retrieval, and explanation generation. However, existing benchmarks fail to comprehensively evaluate LLMs and Multimodal Large Language Models (MLLMs) in realistic misinformation scenarios. To bridge this gap, we introduce RealFactBench, a comprehensive benchmark designed to assess the fact-checking capabilities of LLMs and MLLMs across diverse real-world tasks, including Knowledge Validation, Rumor Detection, and Event Verification. RealFactBench consists of 6K high-quality claims drawn from authoritative sources, encompassing multimodal content and diverse domains. Our evaluation framework further introduces the Unknown Rate (UnR) metric, enabling a more nuanced assessment of models' ability to handle uncertainty and balance between over-conservatism and over-confidence. Extensive experiments on 7 representative LLMs and 4 MLLMs reveal their limitations in real-world fact-checking and offer valuable insights for further research. RealFactBench is publicly available at this https URL. 

---
# Training-free LLM Merging for Multi-task Learning 

**Authors**: Zichuan Fu, Xian Wu, Yejing Wang, Wanyu Wang, Shanshan Ye, Hongzhi Yin, Yi Chang, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12379)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. In this paper, we explore an important question: is it possible to combine these specialized models to create a unified model with multi-task capabilities. We introduces Hierarchical Iterative Merging (Hi-Merging), a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts. Extensive experiments on multiple-choice and question-answering tasks in both Chinese and English validate Hi-Merging's ability for multi-task learning. The results demonstrate that Hi-Merging consistently outperforms existing merging techniques and surpasses the performance of models fine-tuned on combined datasets in most scenarios. Code is available at: this https URL. 

---
# Efficient Reasoning Through Suppression of Self-Affirmation Reflections in Large Reasoning Models 

**Authors**: Kaiyuan Liu, Chen Shen, Zhanwei Zhang, Junjie Liu, Xiaosong Yuan, Jieping ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.12353)  

**Abstract**: While recent advances in large reasoning models have demonstrated remarkable performance, efficient reasoning remains critical due to the rapid growth of output length. Existing optimization approaches highlights a tendency toward "overthinking", yet lack fine-grained analysis. In this work, we focus on Self-Affirmation Reflections: redundant reflective steps that affirm prior content and often occurs after the already correct reasoning steps. Observations of both original and optimized reasoning models reveal pervasive self-affirmation reflections. Notably, these reflections sometimes lead to longer outputs in optimized models than their original counterparts. Through detailed analysis, we uncover an intriguing pattern: compared to other reflections, the leading words (i.e., the first word of sentences) in self-affirmation reflections exhibit a distinct probability bias. Motivated by this insight, we can locate self-affirmation reflections and conduct a train-free experiment demonstrating that suppressing self-affirmation reflections reduces output length without degrading accuracy across multiple models (R1-Distill-Models, QwQ-32B, and Qwen3-32B). Furthermore, we also improve current train-based method by explicitly suppressing such reflections. In our experiments, we achieve length compression of 18.7\% in train-free settings and 50.2\% in train-based settings for R1-Distill-Qwen-1.5B. Moreover, our improvements are simple yet practical and can be directly applied to existing inference frameworks, such as vLLM. We believe that our findings will provide community insights for achieving more precise length compression and step-level efficient reasoning. 

---
# Intersectional Bias in Japanese Large Language Models from a Contextualized Perspective 

**Authors**: Hitomi Yanaka, Xinqi He, Jie Lu, Namgi Han, Sunjin Oh, Ryoma Kumon, Yuma Matsuoka, Katsuhiko Watabe, Yuko Itatsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12327)  

**Abstract**: An growing number of studies have examined the social bias of rapidly developed large language models (LLMs). Although most of these studies have focused on bias occurring in a single social attribute, research in social science has shown that social bias often occurs in the form of intersectionality -- the constitutive and contextualized perspective on bias aroused by social attributes. In this study, we construct the Japanese benchmark inter-JBBQ, designed to evaluate the intersectional bias in LLMs on the question-answering setting. Using inter-JBBQ to analyze GPT-4o and Swallow, we find that biased output varies according to its contexts even with the equal combination of social attributes. 

---
# Refract ICL: Rethinking Example Selection in the Era of Million-Token Models 

**Authors**: Arjun R. Akula, Kazuma Hashimoto, Krishna Srinivasan, Aditi Chaudhary, Karthik Raman, Michael Bendersky  

**Link**: [PDF](https://arxiv.org/pdf/2506.12346)  

**Abstract**: The emergence of long-context large language models (LLMs) has enabled the use of hundreds, or even thousands, of demonstrations for in-context learning (ICL) - a previously impractical regime. This paper investigates whether traditional ICL selection strategies, which balance the similarity of ICL examples to the test input (using a text retriever) with diversity within the ICL set, remain effective when utilizing a large number of demonstrations. Our experiments demonstrate that, while longer contexts can accommodate more examples, simply increasing the number of demonstrations does not guarantee improved performance. Smart ICL selection remains crucial, even with thousands of demonstrations. To further enhance ICL in this setting, we introduce Refract ICL, a novel ICL selection algorithm specifically designed to focus LLM attention on challenging examples by strategically repeating them within the context and incorporating zero-shot predictions as error signals. Our results show that Refract ICL significantly improves the performance of extremely long-context models such as Gemini 1.5 Pro, particularly on tasks with a smaller number of output classes. 

---
# SheetMind: An End-to-End LLM-Powered Multi-Agent Framework for Spreadsheet Automation 

**Authors**: Ruiyan Zhu, Xi Cheng, Ke Liu, Brian Zhu, Daniel Jin, Neeraj Parihar, Zhoutian Xu, Oliver Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12339)  

**Abstract**: We present SheetMind, a modular multi-agent framework powered by large language models (LLMs) for spreadsheet automation via natural language instructions. The system comprises three specialized agents: a Manager Agent that decomposes complex user instructions into subtasks; an Action Agent that translates these into structured commands using a Backus Naur Form (BNF) grammar; and a Reflection Agent that validates alignment between generated actions and the user's original intent. Integrated into Google Sheets via a Workspace extension, SheetMind supports real-time interaction without requiring scripting or formula knowledge. Experiments on benchmark datasets demonstrate an 80 percent success rate on single step tasks and approximately 70 percent on multi step instructions, outperforming ablated and baseline variants. Our results highlight the effectiveness of multi agent decomposition and grammar based execution for bridging natural language and spreadsheet functionalities. 

---
# Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning 

**Authors**: Xiaotian Zhang, Yuan Wang, Zhaopeng Feng, Ruizhe Chen, Zhijie Zhou, Yan Zhang, Hongxia Xu, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12307)  

**Abstract**: Medical Question-Answering (QA) encompasses a broad spectrum of tasks, including multiple choice questions (MCQ), open-ended text generation, and complex computational reasoning. Despite this variety, a unified framework for delivering high-quality medical QA has yet to emerge. Although recent progress in reasoning-augmented large language models (LLMs) has shown promise, their ability to achieve comprehensive medical understanding is still largely unexplored. In this paper, we present Med-U1, a unified framework for robust reasoning across medical QA tasks with diverse output formats, ranging from MCQs to complex generation and computation tasks. Med-U1 employs pure large-scale reinforcement learning with mixed rule-based binary reward functions, incorporating a length penalty to manage output verbosity. With multi-objective reward optimization, Med-U1 directs LLMs to produce concise and verifiable reasoning chains. Empirical results reveal that Med-U1 significantly improves performance across multiple challenging Med-QA benchmarks, surpassing even larger specialized and proprietary models. Furthermore, Med-U1 demonstrates robust generalization to out-of-distribution (OOD) tasks. Extensive analysis presents insights into training strategies, reasoning chain length control, and reward design for medical LLMs. The code will be released. 

---
# Unveiling Confirmation Bias in Chain-of-Thought Reasoning 

**Authors**: Yue Wan, Xiaowei Jia, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12301)  

**Abstract**: Chain-of-thought (CoT) prompting has been widely adopted to enhance the reasoning capabilities of large language models (LLMs). However, the effectiveness of CoT reasoning is inconsistent across tasks with different reasoning types. This work presents a novel perspective to understand CoT behavior through the lens of \textit{confirmation bias} in cognitive psychology. Specifically, we examine how model internal beliefs, approximated by direct question-answering probabilities, affect both reasoning generation ($Q \to R$) and reasoning-guided answer prediction ($QR \to A$) in CoT. By decomposing CoT into a two-stage process, we conduct a thorough correlation analysis in model beliefs, rationale attributes, and stage-wise performance. Our results provide strong evidence of confirmation bias in LLMs, such that model beliefs not only skew the reasoning process but also influence how rationales are utilized for answer prediction. Furthermore, the interplay between task vulnerability to confirmation bias and the strength of beliefs also provides explanations for CoT effectiveness across reasoning tasks and models. Overall, this study provides a valuable insight for the needs of better prompting strategies that mitigate confirmation bias to enhance reasoning performance. Code is available at \textit{this https URL}. 

---
# ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration 

**Authors**: Jennifer Grannen, Siddharth Karamcheti, Blake Wulfe, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2506.12248)  

**Abstract**: Collaborative robots must quickly adapt to their partner's intent and preferences to proactively identify helpful actions. This is especially true in situated settings where human partners can continually teach robots new high-level behaviors, visual concepts, and physical skills (e.g., through demonstration), growing the robot's capabilities as the human-robot pair work together to accomplish diverse tasks. In this work, we argue that robots should be able to infer their partner's goals from early interactions and use this information to proactively plan behaviors ahead of explicit instructions from the user. Building from the strong commonsense priors and steerability of large language models, we introduce ProVox ("Proactive Voice"), a novel framework that enables robots to efficiently personalize and adapt to individual collaborators. We design a meta-prompting protocol that empowers users to communicate their distinct preferences, intent, and expected robot behaviors ahead of starting a physical interaction. ProVox then uses the personalized prompt to condition a proactive language model task planner that anticipates a user's intent from the current interaction context and robot capabilities to suggest helpful actions; in doing so, we alleviate user burden, minimizing the amount of time partners spend explicitly instructing and supervising the robot. We evaluate ProVox through user studies grounded in household manipulation tasks (e.g., assembling lunch bags) that measure the efficiency of the collaboration, as well as features such as perceived helpfulness, ease of use, and reliability. Our analysis suggests that both meta-prompting and proactivity are critical, resulting in 38.7% faster task completion times and 31.9% less user burden relative to non-active baselines. Supplementary material, code, and videos can be found at this https URL. 

---
# Large Language Models for History, Philosophy, and Sociology of Science: Interpretive Uses, Methodological Challenges, and Critical Perspectives 

**Authors**: Arno Simons, Michael Zichert, Adrian Wüthrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.12242)  

**Abstract**: This paper explores the use of large language models (LLMs) as research tools in the history, philosophy, and sociology of science (HPSS). LLMs are remarkably effective at processing unstructured text and inferring meaning from context, offering new affordances that challenge long-standing divides between computational and interpretive methods. This raises both opportunities and challenges for HPSS, which emphasizes interpretive methodologies and understands meaning as context-dependent, ambiguous, and historically situated. We argue that HPSS is uniquely positioned not only to benefit from LLMs' capabilities but also to interrogate their epistemic assumptions and infrastructural implications. To this end, we first offer a concise primer on LLM architectures and training paradigms tailored to non-technical readers. We frame LLMs not as neutral tools but as epistemic infrastructures that encode assumptions about meaning, context, and similarity, conditioned by their training data, architecture, and patterns of use. We then examine how computational techniques enhanced by LLMs, such as structuring data, detecting patterns, and modeling dynamic processes, can be applied to support interpretive research in HPSS. Our analysis compares full-context and generative models, outlines strategies for domain and task adaptation (e.g., continued pretraining, fine-tuning, and retrieval-augmented generation), and evaluates their respective strengths and limitations for interpretive inquiry in HPSS. We conclude with four lessons for integrating LLMs into HPSS: (1) model selection involves interpretive trade-offs; (2) LLM literacy is foundational; (3) HPSS must define its own benchmarks and corpora; and (4) LLMs should enhance, not replace, interpretive methods. 

---
# Mind the XAI Gap: A Human-Centered LLM Framework for Democratizing Explainable AI 

**Authors**: Eva Paraschou, Ioannis Arapakis, Sofia Yfantidou, Sebastian Macaluso, Athena Vakali  

**Link**: [PDF](https://arxiv.org/pdf/2506.12240)  

**Abstract**: Artificial Intelligence (AI) is rapidly embedded in critical decision-making systems, however their foundational ``black-box'' models require eXplainable AI (XAI) solutions to enhance transparency, which are mostly oriented to experts, making no sense to non-experts. Alarming evidence about AI's unprecedented human values risks brings forward the imperative need for transparent human-centered XAI solutions. In this work, we introduce a domain-, model-, explanation-agnostic, generalizable and reproducible framework that ensures both transparency and human-centered explanations tailored to the needs of both experts and non-experts. The framework leverages Large Language Models (LLMs) and employs in-context learning to convey domain- and explainability-relevant contextual knowledge into LLMs. Through its structured prompt and system setting, our framework encapsulates in one response explanations understandable by non-experts and technical information to experts, all grounded in domain and explainability principles. To demonstrate the effectiveness of our framework, we establish a ground-truth contextual ``thesaurus'' through a rigorous benchmarking with over 40 data, model, and XAI combinations for an explainable clustering analysis of a well-being scenario. Through a comprehensive quality and human-friendliness evaluation of our framework's explanations, we prove high content quality through strong correlations with ground-truth explanations (Spearman rank correlation=0.92) and improved interpretability and human-friendliness to non-experts through a user study (N=56). Our overall evaluation confirms trust in LLMs as HCXAI enablers, as our framework bridges the above Gaps by delivering (i) high-quality technical explanations aligned with foundational XAI methods and (ii) clear, efficient, and interpretable human-centered explanations for non-experts. 

---
# QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety 

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.12299)  

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts. 

---
# The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs 

**Authors**: Avinash Baidya, Kamalika Das, Xiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12266)  

**Abstract**: Large Language Model (LLM)-based agents have significantly impacted Task-Oriented Dialog Systems (TODS) but continue to face notable performance challenges, especially in zero-shot scenarios. While prior work has noted this performance gap, the behavioral factors driving the performance gap remain under-explored. This study proposes a comprehensive evaluation framework to quantify the behavior gap between AI agents and human experts, focusing on discrepancies in dialog acts, tool usage, and knowledge utilization. Our findings reveal that this behavior gap is a critical factor negatively impacting the performance of LLM agents. Notably, as task complexity increases, the behavior gap widens (correlation: 0.963), leading to a degradation of agent performance on complex task-oriented dialogs. For the most complex task in our study, even the GPT-4o-based agent exhibits low alignment with human behavior, with low F1 scores for dialog acts (0.464), excessive and often misaligned tool usage with a F1 score of 0.139, and ineffective usage of external knowledge. Reducing such behavior gaps leads to significant performance improvement (24.3% on average). This study highlights the importance of comprehensive behavioral evaluations and improved alignment strategies to enhance the effectiveness of LLM-based TODS in handling complex tasks. 

---
# Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach 

**Authors**: Khadija Zanna, Akane Sano  

**Link**: [PDF](https://arxiv.org/pdf/2506.12227)  

**Abstract**: Causal discovery (CD) plays a pivotal role in understanding the mechanisms underlying complex systems. While recent algorithms can detect spurious associations and latent confounding, many struggle to recover fairness-relevant pathways in realistic, noisy settings. Large Language Models (LLMs), with their access to broad semantic knowledge, offer a promising complement to statistical CD approaches, particularly in domains where metadata provides meaningful relational cues. Ensuring fairness in machine learning requires understanding how sensitive attributes causally influence outcomes, yet CD methods often introduce spurious or biased pathways. We propose a hybrid LLM-based framework for CD that extends a breadth-first search (BFS) strategy with active learning and dynamic scoring. Variable pairs are prioritized for LLM-based querying using a composite score based on mutual information, partial correlation, and LLM confidence, improving discovery efficiency and robustness.
To evaluate fairness sensitivity, we construct a semi-synthetic benchmark from the UCI Adult dataset, embedding a domain-informed causal graph with injected noise, label corruption, and latent confounding. We assess how well CD methods recover both global structure and fairness-critical paths.
Our results show that LLM-guided methods, including the proposed method, demonstrate competitive or superior performance in recovering such pathways under noisy conditions. We highlight when dynamic scoring and active querying are most beneficial and discuss implications for bias auditing in real-world datasets. 

---
# Semantic Scheduling for LLM Inference 

**Authors**: Wenyue Hua, Dujian Ding, Yile Gu, Yujie Ren, Kai Mei, Minghua Ma, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12204)  

**Abstract**: Conventional operating system scheduling algorithms are largely content-ignorant, making decisions based on factors such as latency or fairness without considering the actual intents or semantics of processes. Consequently, these algorithms often do not prioritize tasks that require urgent attention or carry higher importance, such as in emergency management scenarios. However, recent advances in language models enable semantic analysis of processes, allowing for more intelligent and context-aware scheduling decisions. In this paper, we introduce the concept of semantic scheduling in scheduling of requests from large language models (LLM), where the semantics of the process guide the scheduling priorities. We present a novel scheduling algorithm with optimal time complexity, designed to minimize the overall waiting time in LLM-based prompt scheduling. To illustrate its effectiveness, we present a medical emergency management application, underscoring the potential benefits of semantic scheduling for critical, time-sensitive tasks. The code and data are available at this https URL. 

---
# Supernova Event Dataset: Interpreting Large Language Model's Personality through Critical Event Analysis 

**Authors**: Pranav Agarwal, Ioana Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2506.12189)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into everyday applications. As their influence grows, understanding their decision making and underlying personality becomes essential. In this work, we interpret model personality using our proposed Supernova Event Dataset, a novel dataset with diverse articles spanning biographies, historical events, news, and scientific discoveries. We use this dataset to benchmark LLMs on extracting and ranking key events from text, a subjective and complex challenge that requires reasoning over long-range context and modeling causal chains. We evaluate small models like Phi-4, Orca 2, and Qwen 2.5, and large, stronger models such as Claude 3.7, Gemini 2.5, and OpenAI o3, and propose a framework where another LLM acts as a judge to infer each model's personality based on its selection and classification of events. Our analysis shows distinct personality traits: for instance, Orca 2 demonstrates emotional reasoning focusing on interpersonal dynamics, while Qwen 2.5 displays a more strategic, analytical style. When analyzing scientific discovery events, Claude Sonnet 3.7 emphasizes conceptual framing, Gemini 2.5 Pro prioritizes empirical validation, and o3 favors step-by-step causal reasoning. This analysis improves model interpretability, making them user-friendly for a wide range of diverse applications. 

---
# Explaining Recovery Trajectories of Older Adults Post Lower-Limb Fracture Using Modality-wise Multiview Clustering and Large Language Models 

**Authors**: Shehroz S. Khan, Ali Abedi, Charlene H. Chu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12156)  

**Abstract**: Interpreting large volumes of high-dimensional, unlabeled data in a manner that is comprehensible to humans remains a significant challenge across various domains. In unsupervised healthcare data analysis, interpreting clustered data can offer meaningful insights into patients' health outcomes, which hold direct implications for healthcare providers. This paper addresses the problem of interpreting clustered sensor data collected from older adult patients recovering from lower-limb fractures in the community. A total of 560 days of multimodal sensor data, including acceleration, step count, ambient motion, GPS location, heart rate, and sleep, alongside clinical scores, were remotely collected from patients at home. Clustering was first carried out separately for each data modality to assess the impact of feature sets extracted from each modality on patients' recovery trajectories. Then, using context-aware prompting, a large language model was employed to infer meaningful cluster labels for the clusters derived from each modality. The quality of these clusters and their corresponding labels was validated through rigorous statistical testing and visualization against clinical scores collected alongside the multimodal sensor data. The results demonstrated the statistical significance of most modality-specific cluster labels generated by the large language model with respect to clinical scores, confirming the efficacy of the proposed method for interpreting sensor data in an unsupervised manner. This unsupervised data analysis approach, relying solely on sensor data, enables clinicians to identify at-risk patients and take timely measures to improve health outcomes. 

---
# Semantic Preprocessing for LLM-based Malware Analysis 

**Authors**: Benjamin Marais, Tony Quertier, Grégoire Barrue  

**Link**: [PDF](https://arxiv.org/pdf/2506.12113)  

**Abstract**: In a context of malware analysis, numerous approaches rely on Artificial Intelligence to handle a large volume of data. However, these techniques focus on data view (images, sequences) and not on an expert's view. Noticing this issue, we propose a preprocessing that focuses on expert knowledge to improve malware semantic analysis and result interpretability. We propose a new preprocessing method which creates JSON reports for Portable Executable files. These reports gather features from both static and behavioral analysis, and incorporate packer signature detection, MITRE ATT\&CK and Malware Behavior Catalog (MBC) knowledge. The purpose of this preprocessing is to gather a semantic representation of binary files, understandable by malware analysts, and that can enhance AI models' explainability for malicious files analysis. Using this preprocessing to train a Large Language Model for Malware classification, we achieve a weighted-average F1-score of 0.94 on a complex dataset, representative of market reality. 

---
# Eliciting Reasoning in Language Models with Cognitive Tools 

**Authors**: Brown Ebouky, Andrea Bartezzaghi, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.12115)  

**Abstract**: The recent advent of reasoning models like OpenAI's o1 was met with excited speculation by the AI community about the mechanisms underlying these capabilities in closed models, followed by a rush of replication efforts, particularly from the open source community. These speculations were largely settled by the demonstration from DeepSeek-R1 that chains-of-thought and reinforcement learning (RL) can effectively replicate reasoning on top of base LLMs. However, it remains valuable to explore alternative methods for theoretically eliciting reasoning that could help elucidate the underlying mechanisms, as well as providing additional methods that may offer complementary benefits.
Here, we build on the long-standing literature in cognitive psychology and cognitive architectures, which postulates that reasoning arises from the orchestrated, sequential execution of a set of modular, predetermined cognitive operations. Crucially, we implement this key idea within a modern agentic tool-calling framework. In particular, we endow an LLM with a small set of "cognitive tools" encapsulating specific reasoning operations, each executed by the LLM itself. Surprisingly, this simple strategy results in considerable gains in performance on standard mathematical reasoning benchmarks compared to base LLMs, for both closed and open-weight models. For instance, providing our "cognitive tools" to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%, bringing it very close to the performance of o1-preview.
In addition to its practical implications, this demonstration contributes to the debate regarding the role of post-training methods in eliciting reasoning in LLMs versus the role of inherent capabilities acquired during pre-training, and whether post-training merely uncovers these latent abilities. 

---
# LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model's Response for Vulnerability Analysis 

**Authors**: Reza Fayyazi, Michael Zuzak, Shanchieh Jay Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12100)  

**Abstract**: Security vulnerabilities are rapidly increasing in frequency and complexity, creating a shifting threat landscape that challenges cybersecurity defenses. Large Language Models (LLMs) have been widely adopted for cybersecurity threat analysis. When querying LLMs, dealing with new, unseen vulnerabilities is particularly challenging as it lies outside LLMs' pre-trained distribution. Retrieval-Augmented Generation (RAG) pipelines mitigate the problem by injecting up-to-date authoritative sources into the model context, thus reducing hallucinations and increasing the accuracy in responses. Meanwhile, the deployment of LLMs in security-sensitive environments introduces challenges around trust and safety. This raises a critical open question: How to quantify or attribute the generated response to the retrieved context versus the model's pre-trained knowledge? This work proposes LLM Embedding-based Attribution (LEA) -- a novel, explainable metric to paint a clear picture on the 'percentage of influence' the pre-trained knowledge vs. retrieved content has for each generated response. We apply LEA to assess responses to 100 critical CVEs from the past decade, verifying its effectiveness to quantify the insightfulness for vulnerability analysis. Our development of LEA reveals a progression of independency in hidden states of LLMs: heavy reliance on context in early layers, which enables the derivation of LEA; increased independency in later layers, which sheds light on why scale is essential for LLM's effectiveness. This work provides security analysts a means to audit LLM-assisted workflows, laying the groundwork for transparent, high-assurance deployments of RAG-enhanced LLMs in cybersecurity operations. 

---
# Personalized LLM Decoding via Contrasting Personal Preference 

**Authors**: Hyungjune Bu, Chanjoo Jung, Minjae Kang, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12109)  

**Abstract**: As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures. 

---
# Intelligent Automation for FDI Facilitation: Optimizing Tariff Exemption Processes with OCR And Large Language Models 

**Authors**: Muhammad Sukri Bin Ramli  

**Link**: [PDF](https://arxiv.org/pdf/2506.12093)  

**Abstract**: Tariff exemptions are fundamental to attracting Foreign Direct Investment (FDI) into the manufacturing sector, though the associated administrative processes present areas for optimization for both investing entities and the national tax authority. This paper proposes a conceptual framework to empower tax administration by leveraging a synergistic integration of Optical Character Recognition (OCR) and Large Language Model (LLM) technologies. The proposed system is designed to first utilize OCR for intelligent digitization, precisely extracting data from diverse application documents and key regulatory texts such as tariff orders. Subsequently, the LLM would enhance the capabilities of administrative officers by automating the critical and time-intensive task of verifying submitted HS Tariff Codes for machinery, equipment, and raw materials against official exemption lists. By enhancing the speed and precision of these initial assessments, this AI-driven approach systematically reduces potential for non-alignment and non-optimized exemption utilization, thereby streamlining the investment journey for FDI companies. For the national administration, the benefits include a significant boost in operational capacity, reduced administrative load, and a strengthened control environment, ultimately improving the ease of doing business and solidifying the nation's appeal as a premier destination for high-value manufacturing FDI. 

---
# Modeling Earth-Scale Human-Like Societies with One Billion Agents 

**Authors**: Haoxiang Guan, Jiyan He, Liyang Fan, Zhenzhen Ren, Shaobin He, Xin Yu, Yuan Chen, Shuxin Zheng, Tie-Yan Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12078)  

**Abstract**: Understanding how complex societal behaviors emerge from individual cognition and interactions requires both high-fidelity modeling of human behavior and large-scale simulations. Traditional agent-based models (ABMs) have been employed to study these dynamics for decades, but are constrained by simplified agent behaviors that fail to capture human complexity. Recent advances in large language models (LLMs) offer new opportunities by enabling agents to exhibit sophisticated social behaviors that go beyond rule-based logic, yet face significant scaling challenges. Here we present Light Society, an agent-based simulation framework that advances both fronts, efficiently modeling human-like societies at planetary scale powered by LLMs. Light Society formalizes social processes as structured transitions of agent and environment states, governed by a set of LLM-powered simulation operations, and executed through an event queue. This modular design supports both independent and joint component optimization, supporting efficient simulation of societies with over one billion agents. Large-scale simulations of trust games and opinion propagation--spanning up to one billion agents--demonstrate Light Society's high fidelity and efficiency in modeling social trust and information diffusion, while revealing scaling laws whereby larger simulations yield more stable and realistic emergent behaviors. 

---
# From Emergence to Control: Probing and Modulating Self-Reflection in Language Models 

**Authors**: Xudong Zhu, Jiachen Jiang, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12217)  

**Abstract**: Self-reflection -- the ability of a large language model (LLM) to revisit, evaluate, and revise its own reasoning -- has recently emerged as a powerful behavior enabled by reinforcement learning with verifiable rewards (RLVR). While self-reflection correlates with improved reasoning accuracy, its origin and underlying mechanisms remain poorly understood. In this work, {\it we first show that self-reflection is not exclusive to RLVR fine-tuned models: it already emerges, albeit rarely, in pretrained models}. To probe this latent ability, we introduce Reflection-Inducing Probing, a method that injects reflection-triggering reasoning traces from fine-tuned models into pretrained models. This intervention raises self-reflection frequency of Qwen2.5 from 0.6\% to 18.6\%, revealing a hidden capacity for reflection. Moreover, our analysis of internal representations shows that both pretrained and fine-tuned models maintain hidden states that distinctly separate self-reflective from non-reflective contexts. Leveraging this observation, {\it we then construct a self-reflection vector, a direction in activation space associated with self-reflective reasoning}. By manipulating this vector, we enable bidirectional control over the self-reflective behavior for both pretrained and fine-tuned models. Experiments across multiple reasoning benchmarks show that enhancing these vectors improves reasoning performance by up to 12\%, while suppressing them reduces computational cost, providing a flexible mechanism to navigate the trade-off between reasoning quality and efficiency without requiring additional training. Our findings further our understanding of self-reflection and support a growing body of work showing that understanding model internals can enable precise behavioral control. 

---
# WebTrust: An AI-Driven Data Scoring System for Reliable Information Retrieval 

**Authors**: Joydeep Chandra, Aleksandr Algazinov, Satyam Kumar Navneet, Rim El Filali, Matt Laing, Andrew Hanna  

**Link**: [PDF](https://arxiv.org/pdf/2506.12072)  

**Abstract**: As access to information becomes more open and widespread, people are increasingly using AI tools for assistance. However, many of these tools struggle to estimate the trustworthiness of the information. Although today's search engines include AI features, they often fail to offer clear indicators of data reliability. To address this gap, we introduce WebTrust, a system designed to simplify the process of finding and judging credible information online. Built on a fine-tuned version of IBM's Granite-1B model and trained on a custom dataset, WebTrust works by assigning a reliability score (from 0.1 to 1) to each statement it processes. In addition, it offers a clear justification for why a piece of information received that score. Evaluated using prompt engineering, WebTrust consistently achieves superior performance compared to other small-scale LLMs and rule-based approaches, outperforming them across all experiments on MAE, RMSE, and R2. User testing showed that when reliability scores are displayed alongside search results, people feel more confident and satisfied with the information they find. With its accuracy, transparency, and ease of use, WebTrust offers a practical solution to help combat misinformation and make trustworthy information more accessible to everyone. 

---
# Why Do Some Inputs Break Low-Bit LLM Quantization? 

**Authors**: Ting-Yun Chang, Muru Zhang, Jesse Thomason, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.12044)  

**Abstract**: Low-bit weight-only quantization significantly reduces the memory footprint of large language models (LLMs), but disproportionately affects certain examples. We analyze diverse 3-4 bit methods on LLMs ranging from 7B-70B in size and find that the quantization errors of 50 pairs of methods are strongly correlated (avg. 0.82) on FineWeb examples. Moreover, the residual stream magnitudes of full-precision models are indicative of future quantization errors. We further establish a hypothesis that relates the residual stream magnitudes to error amplification and accumulation over layers. Using LLM localization techniques, early exiting, and activation patching, we show that examples with large errors rely on precise residual activations in the late layers, and that the outputs of MLP gates play a crucial role in maintaining the perplexity. Our work reveals why certain examples result in large quantization errors and which model components are most critical for performance preservation. 

---
# BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook 

**Authors**: Hao Gu, Lujun Li, Zheyu Wang, Bei Liu, Qiyuan Zhu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.12040)  

**Abstract**: Binary quantization represents the most extreme form of large language model (LLM) compression, reducing weights to $\pm$1 for maximal memory and computational efficiency. While recent sparsity-aware binarization methods achieve sub-1-bit compression by pruning redundant binary weights, they suffer from three critical challenges: performance deterioration, computational complexity from sparse mask management, and limited hardware compatibility. In this paper, we present BTC-LLM, a novel sub-1-bit LLM quantization framework that leverages adaptive weight transformation and binary pattern clustering to overcome these limitations, delivering both superior accuracy and efficiency. Our approach incorporates two key innovations: (1) a Learnable Transformation that optimizes invertible scaling and rotation matrices to align binarized weights with full-precision distributions, enabling incoherence processing to enhance layer-wise representation quality; (2) a Flash and Accurate Binary Codebook that identifies recurring binary vector clusters, compressing them into compact indices with tailored distance metrics and sign-based centroid updates. This eliminates the need for sparse masks, enabling efficient inference on standard hardware. Our code is available at this https URL. 

---
# CMT-LLM: Contextual Multi-Talker ASR Utilizing Large Language Models 

**Authors**: Jiajun He, Naoki Sawada, Koichi Miyazaki, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.12059)  

**Abstract**: In real-world applications, automatic speech recognition (ASR) systems must handle overlapping speech from multiple speakers and recognize rare words like technical terms. Traditional methods address multi-talker ASR and contextual biasing separately, limiting performance in complex scenarios. We propose a unified framework that combines multi-talker overlapping speech recognition and contextual biasing into a single task. Our ASR method integrates pretrained speech encoders and large language models (LLMs), using optimized finetuning strategies. We also introduce a two-stage filtering algorithm to efficiently identify relevant rare words from large biasing lists and incorporate them into the LLM's prompt input, enhancing rare word recognition. Experiments show that our approach outperforms traditional contextual biasing methods, achieving a WER of 7.9% on LibriMix and 32.9% on AMI SDM when the biasing size is 1,000, demonstrating its effectiveness in complex speech scenarios. 

---
# SocialCredit+ 

**Authors**: Thabassum Aslam, Anees Aslam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12099)  

**Abstract**: SocialCredit+ is AI powered credit scoring system that leverages publicly available social media data to augment traditional credit evaluation. It uses a conversational banking assistant to gather user consent and fetch public profiles. Multimodal feature extractors analyze posts, bios, images, and friend networks to generate a rich behavioral profile. A specialized Sharia-compliance layer flags any non-halal indicators and prohibited financial behavior based on Islamic ethics. The platform employs a retrieval-augmented generation module: an LLM accesses a domain specific knowledge base to generate clear, text-based explanations for each decision. We describe the end-to-end architecture and data flow, the models used, and system infrastructure. Synthetic scenarios illustrate how social signals translate into credit-score factors. This paper emphasizes conceptual novelty, compliance mechanisms, and practical impact, targeting AI researchers, fintech practitioners, ethical banking jurists, and investors. 

---
# An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability 

**Authors**: Yusuke Yamauchi, Taro Yano, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.13639)  

**Abstract**: As large language models (LLMs) continue to advance, reliable evaluation methods are essential particularly for open-ended, instruction-following tasks. LLM-as-a-Judge enables automatic evaluation using LLMs as evaluators, but its reliability remains uncertain. In this work, we analyze key factors affecting its trustworthiness, focusing on alignment with human judgments and evaluation consistency. Using BIGGENBench and EvalBiasBench, we study the effects of evaluation design, decoding strategies, and Chain-of-Tought (CoT) reasoning in evaluation. Our results show that evaluation criteria are critical for reliability, non-deterministic sampling improves alignment with human preferences over deterministic evaluation, and CoT reasoning offers minimal gains when clear evaluation criteria are present. 

---
# Qwen vs. Gemma Integration with Whisper: A Comparative Study in Multilingual SpeechLLM Systems 

**Authors**: Tuan Nguyen, Long-Vu Hoang, Huy-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.13596)  

**Abstract**: This paper presents our system for the MLC-SLM Challenge 2025, focusing on multilingual speech recognition and language modeling with large language models (LLMs). Our approach combines a fine-tuned Whisper-large-v3 encoder with efficient projector architectures and various decoder configurations. We employ a three-stage training methodology that progressively optimizes the encoder, projector, and LLM components. Our system achieves competitive performance with a private test average WER/CER result of 16.63% using the Gemma3-12B and 18.6% using the Qwen2.5-7B as decoder-only language model. 

---
# Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning 

**Authors**: David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Matthias Keicher, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.13474)  

**Abstract**: Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency. 

---
# BOW: Bottlenecked Next Word Exploration 

**Authors**: Ming Shen, Zhikun Xu, Xiao Ye, Jacob Dineen, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.13502)  

**Abstract**: Large language models (LLMs) are typically trained via next-word prediction (NWP), which provides strong surface-level fluency but often lacks support for robust reasoning. We propose BOttlenecked next Word exploration (BOW), a novel RL framework that rethinks NWP by introducing a reasoning bottleneck where a policy model first generates a reasoning path rather than predicting the next token directly, after which a frozen judge model predicts the next token distribution based solely on this reasoning path. We train the policy model using GRPO with rewards that quantify how effectively the reasoning path facilitates next-word recovery. Compared with other continual pretraining baselines, we show that BOW improves both the general and next-word reasoning capabilities of the base model, evaluated on various benchmarks. Our findings show that BOW can serve as an effective and scalable alternative to vanilla NWP. 

---
# EvolvTrip: Enhancing Literary Character Understanding with Temporal Theory-of-Mind Graphs 

**Authors**: Bohao Yang, Hainiu Xu, Jinhua Du, Ze Li, Yulan He, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13641)  

**Abstract**: A compelling portrayal of characters is essential to the success of narrative writing. For readers, appreciating a character's traits requires the ability to infer their evolving beliefs, desires, and intentions over the course of a complex storyline, a cognitive skill known as Theory-of-Mind (ToM). Performing ToM reasoning in prolonged narratives requires readers to integrate historical context with current narrative information, a task at which humans excel but Large Language Models (LLMs) often struggle. To systematically evaluate LLMs' ToM reasoning capability in long narratives, we construct LitCharToM, a benchmark of character-centric questions across four ToM dimensions from classic literature. Further, we introduce EvolvTrip, a perspective-aware temporal knowledge graph that tracks psychological development throughout narratives. Our experiments demonstrate that EvolvTrip consistently enhances performance of LLMs across varying scales, even in challenging extended-context scenarios. EvolvTrip proves to be particularly valuable for smaller models, partially bridging the performance gap with larger LLMs and showing great compatibility with lengthy narratives. Our findings highlight the importance of explicit representation of temporal character mental states in narrative comprehension and offer a foundation for more sophisticated character understanding. Our data and code are publicly available at this https URL. 

---
# RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis 

**Authors**: Pengzuo Wu, Yuhang Yang, Guangcheng Zhu, Chao Ye, Hong Gu, Xu Lu, Ruixuan Xiao, Bowen Bao, Yijing He, Liangyu Zha, Wentao Ye, Junbo Zhao, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13405)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at this https URL. 

---
# Document-Level Tabular Numerical Cross-Checking: A Coarse-to-Fine Approach 

**Authors**: Chaoxu Pang, Yixuan Cao, Ganbin Zhou, Hongwei Li, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.13328)  

**Abstract**: Numerical consistency across tables in disclosure documents is critical for ensuring accuracy, maintaining credibility, and avoiding reputational and economic risks. Automated tabular numerical cross-checking presents two significant challenges: (C1) managing the combinatorial explosion of candidate instances at the document level and (C2) comprehending multi-faceted numerical semantics. Previous research typically depends on heuristic-based filtering or simplified context extraction, often struggling to balance performance and efficiency. Recently, large language models (LLMs) have demonstrated remarkable contextual understanding capabilities that helps address C2 at the instance level, yet they remain hampered by computational inefficiency (C1) and limited domain expertise. This paper introduces CoFiTCheck, a novel LLM-based coarse-to-fine framework that addresses these challenges through two sequential stages: embedding-based filtering and discriminative classification. The embedding-based filtering stage introduces an instructional parallel encoding method to efficiently represent all numerical mentions in a table with LLMs, as well as a decoupled InfoNCE objective to mitigate the isolated mention problem. The discriminative classification stage employs a specialized LLM for fine-grained analysis of the remaining candidate pairs. This stage is further enhanced by our crosstable numerical alignment pretraining paradigm, which leverages weak supervision from cross-table numerical equality relationships to enrich task-specific priors without requiring manual annotation. Comprehensive evaluation across three types of real-world disclosure documents demonstrates that CoFiTCheck significantly outperforms previous methods while maintaining practical efficiency. 

---
# Decompositional Reasoning for Graph Retrieval with Large Language Models 

**Authors**: Valentin Six, Evan Dufraisse, Gaël de Chalendar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13380)  

**Abstract**: Large Language Models (LLMs) excel at many NLP tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To tackle this problem, we propose a novel retrieval approach that integrates textual knowledge graphs into the LLM reasoning process via query decomposition. Our method decomposes complex questions into sub-questions, retrieves relevant textual subgraphs, and composes a question-specific knowledge graph to guide answer generation. For that, we use a weighted similarity function that focuses on both the complex question and the generated subquestions to extract a relevant subgraph, which allows efficient and precise retrieval for complex questions and improves the performance of LLMs on multi-hop QA tasks. This structured reasoning pipeline enhances factual grounding and interpretability while leveraging the generative strengths of LLMs. We evaluate our method on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls. 

---
# Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs 

**Authors**: Houcheng Jiang, Zetong Zhao, Junfeng Fang, Haokai Ma, Ruipeng Wang, Yang Deng, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2506.13285)  

**Abstract**: Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines. 

---
# NTU Speechlab LLM-Based Multilingual ASR System for Interspeech MLC-SLM Challenge 2025 

**Authors**: Yizhou Peng, Bin Wang, Yi-Wen Chao, Ziyang Ma, Haoyang Zhang, Hexin Liu, Xie Chen, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13339)  

**Abstract**: This report details the NTU Speechlab system developed for the Interspeech 2025 Multilingual Conversational Speech and Language Model (MLC-SLM) Challenge (Task I), where we achieved 5th place. We present comprehensive analyses of our multilingual automatic speech recognition system, highlighting key advancements in model architecture, data selection, and training strategies. In particular, language-specific prompts and model averaging techniques were instrumental in boosting system performance across diverse languages. Compared to the initial baseline system, our final model reduced the average Mix Error Rate from 20.2% to 10.6%, representing an absolute improvement of 9.6% (a relative improvement of 48%) on the evaluation set. Our results demonstrate the effectiveness of our approach and offer practical insights for future Speech Large Language Models. 

---
# Enhancing Large Language Models with Reliable Knowledge Graphs 

**Authors**: Qinggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13178)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in text generation and understanding, yet their reliance on implicit, unstructured knowledge often leads to factual inaccuracies and limited interpretability. Knowledge Graphs (KGs), with their structured, relational representations, offer a promising solution to ground LLMs in verified knowledge. However, their potential remains constrained by inherent noise, incompleteness, and the complexity of integrating their rigid structure with the flexible reasoning of LLMs. This thesis presents a systematic framework to address these limitations, advancing the reliability of KGs and their synergistic integration with LLMs through five interconnected contributions. This thesis addresses these challenges through a cohesive framework that enhances LLMs by refining and leveraging reliable KGs. First, we introduce contrastive error detection, a structure-based method to identify incorrect facts in KGs. This approach is extended by an attribute-aware framework that unifies structural and semantic signals for error correction. Next, we propose an inductive completion model that further refines KGs by completing the missing relationships in evolving KGs. Building on these refined KGs, KnowGPT integrates structured graph reasoning into LLMs through dynamic prompting, improving factual grounding. These contributions form a systematic pipeline (from error detection to LLM integration), demonstrating that reliable KGs significantly enhance the robustness, interpretability, and adaptability of LLMs. 

---
# IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation 

**Authors**: Zijie Lin, Yang Zhang, Xiaoyan Zhao, Fengbin Zhu, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.13229)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines. 

---
# Align-then-Unlearn: Embedding Alignment for LLM Unlearning 

**Authors**: Philipp Spohn, Leander Girrbach, Jessica Bader, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2506.13181)  

**Abstract**: As large language models (LLMs) are trained on massive datasets, they have raised significant privacy and ethical concerns due to their potential to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data from trained models, such as personal information or copyrighted content. Current approaches targeting specific output sequences at the token level often fail to achieve complete forgetting and remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that performs unlearning in the semantic embedding space rather than directly on output tokens. Align-then-Unlearn first augments the LLM with an embedding prediction module trained to anticipate future context representations. Unlearning is then achieved by fine-tuning the model to minimize the similarity between these predicted embeddings and a target embedding that represents the concept to be removed. Initial results show that Align-then-Unlearn effectively removes targeted knowledge with minimal degradation in overall model utility. These findings suggest that embedding-based unlearning offers a promising and robust approach to removing conceptual knowledge. Our code is available at this https URL. 

---
# CFBenchmark-MM: Chinese Financial Assistant Benchmark for Multimodal Large Language Model 

**Authors**: Jiangtong Li, Yiyun Zhu, Dawei Cheng, Zhijun Ding, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13055)  

**Abstract**: Multimodal Large Language Models (MLLMs) have rapidly evolved with the growth of Large Language Models (LLMs) and are now applied in various fields. In finance, the integration of diverse modalities such as text, charts, and tables is crucial for accurate and efficient decision-making. Therefore, an effective evaluation system that incorporates these data types is essential for advancing financial application. In this paper, we introduce CFBenchmark-MM, a Chinese multimodal financial benchmark with over 9,000 image-question pairs featuring tables, histogram charts, line charts, pie charts, and structural diagrams. Additionally, we develop a staged evaluation system to assess MLLMs in handling multimodal information by providing different visual content step by step. Despite MLLMs having inherent financial knowledge, experimental results still show limited efficiency and robustness in handling multimodal financial context. Further analysis on incorrect responses reveals the misinterpretation of visual content and the misunderstanding of financial concepts are the primary issues. Our research validates the significant, yet underexploited, potential of MLLMs in financial analysis, highlighting the need for further development and domain-specific optimization to encourage the enhanced use in financial domain. 

---
# Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs: a case study in Framing Bias Mitigation 

**Authors**: Yuanyuan Lei, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12978)  

**Abstract**: Media outlets are becoming more partisan and polarized nowadays. Most previous work focused on detecting media bias. In this paper, we aim to mitigate media bias by generating a neutralized summary given multiple articles presenting different ideological views. Motivated by the critical role of events and event relations in media bias detection, we propose to increase awareness of bias in LLMs via multi-document events reasoning and use a multi-document event relation graph to guide the summarization process. This graph contains rich event information useful to reveal bias: four common types of in-doc event relations to reflect content framing bias, cross-doc event coreference relation to reveal content selection bias, and event-level moral opinions to highlight opinionated framing bias. We further develop two strategies to incorporate the multi-document event relation graph for neutralized summarization. Firstly, we convert a graph into natural language descriptions and feed the textualized graph into LLMs as a part of a hard text prompt. Secondly, we encode the graph with graph attention network and insert the graph embedding into LLMs as a soft prompt. Both automatic evaluation and human evaluation confirm that our approach effectively mitigates both lexical and informational media bias, and meanwhile improves content preservation. 

---
# Multipole Attention for Efficient Long Context Reasoning 

**Authors**: Coleman Hooper, Sebastian Zhao, Luca Manolache, Sehoon Kim, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2506.13059)  

**Abstract**: Large Reasoning Models (LRMs) have shown promising accuracy improvements on complex problem-solving tasks. While these models have attained high accuracy by leveraging additional computation at test time, they need to generate long chain-of-thought reasoning in order to think before answering, which requires generating thousands of tokens. While sparse attention methods can help reduce the KV cache pressure induced by this long autoregressive reasoning, these methods can introduce errors which disrupt the reasoning process. Additionally, prior methods often pre-process the input to make it easier to identify the important prompt tokens when computing attention during generation, and this pre-processing is challenging to perform online for newly generated reasoning tokens. Our work addresses these challenges by introducing Multipole Attention, which accelerates autoregressive reasoning by only computing exact attention for the most important tokens, while maintaining approximate representations for the remaining tokens. Our method first performs clustering to group together semantically similar key vectors, and then uses the cluster centroids both to identify important key vectors and to approximate the remaining key vectors in order to retain high accuracy. We design a fast cluster update process to quickly re-cluster the input and previously generated tokens, thereby allowing for accelerating attention to the previous output tokens. We evaluate our method using emerging LRMs such as Qwen-8B, demonstrating that our approach can maintain accuracy on complex reasoning tasks even with aggressive attention sparsity settings. We also provide kernel implementations to demonstrate the practical efficiency gains from our method, achieving up to 4.5$\times$ speedup for attention in long-context reasoning applications. Our code is available at this https URL. 

---
# SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models 

**Authors**: Xingjian Diao, Chunhui Zhang, Keyi Kong, Weiyi Wu, Chiyu Ma, Zhongyu Ouyang, Peijun Qing, Soroush Vosoughi, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2506.12935)  

**Abstract**: While large language models have shown reasoning capabilities, their application to the audio modality, particularly in large audio-language models (ALMs), remains significantly underdeveloped. Addressing this gap requires a systematic approach, involving a capable base model, high-quality reasoning-oriented audio data, and effective training algorithms. In this study, we present a comprehensive solution: we introduce the Audio Logical Reasoning (ALR) dataset, consisting of 6,446 text-audio annotated samples specifically designed for complex reasoning tasks. Building on this resource, we propose SoundMind, a rule-based reinforcement learning (RL) algorithm tailored to endow ALMs with deep bimodal reasoning abilities. By training Qwen2.5-Omni-7B on the ALR dataset using SoundMind, our approach achieves state-of-the-art performance in audio logical reasoning. This work highlights the impact of combining high-quality, reasoning-focused datasets with specialized RL techniques, advancing the frontier of auditory intelligence in language models. Our code and the proposed dataset are available at this https URL. 

---
# Large Language Models Enhanced by Plug and Play Syntactic Knowledge for Aspect-based Sentiment Analysis 

**Authors**: Yuanhe Tian, Xu Li, Wei Wang, Guoqing Jin, Pengsen Cheng, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.12991)  

**Abstract**: Aspect-based sentiment analysis (ABSA) generally requires a deep understanding of the contextual information, including the words associated with the aspect terms and their syntactic dependencies. Most existing studies employ advanced encoders (e.g., pre-trained models) to capture such context, especially large language models (LLMs). However, training these encoders is resource-intensive, and in many cases, the available data is insufficient for necessary fine-tuning. Therefore it is challenging for learning LLMs within such restricted environments and computation efficiency requirement. As a result, it motivates the exploration of plug-and-play methods that adapt LLMs to ABSA with minimal effort. In this paper, we propose an approach that integrates extendable components capable of incorporating various types of syntactic knowledge, such as constituent syntax, word dependencies, and combinatory categorial grammar (CCG). Specifically, we propose a memory module that records syntactic information and is incorporated into LLMs to instruct the prediction of sentiment polarities. Importantly, this encoder acts as a versatile, detachable plugin that is trained independently of the LLM. We conduct experiments on benchmark datasets, which show that our approach outperforms strong baselines and previous approaches, thus demonstrates its effectiveness. 

---
# Transforming Chatbot Text: A Sequence-to-Sequence Approach 

**Authors**: Natesh Reddy, Mark Stamp  

**Link**: [PDF](https://arxiv.org/pdf/2506.12843)  

**Abstract**: Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text. 

---
# Synthetic Socratic Debates: Examining Persona Effects on Moral Decision and Persuasion Dynamics 

**Authors**: Jiarui Liu, Yueqi Song, Yunze Xiao, Mingqian Zheng, Lindia Tjuatja, Jana Schaich Borg, Mona Diab, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.12657)  

**Abstract**: As large language models (LLMs) are increasingly used in morally sensitive domains, it is crucial to understand how persona traits affect their moral reasoning and persuasive behavior. We present the first large-scale study of multi-dimensional persona effects in AI-AI debates over real-world moral dilemmas. Using a 6-dimensional persona space (age, gender, country, class, ideology, and personality), we simulate structured debates between AI agents over 131 relationship-based cases. Our results show that personas affect initial moral stances and debate outcomes, with political ideology and personality traits exerting the strongest influence. Persuasive success varies across traits, with liberal and open personalities reaching higher consensus and win rates. While logit-based confidence grows during debates, emotional and credibility-based appeals diminish, indicating more tempered argumentation over time. These trends mirror findings from psychology and cultural studies, reinforcing the need for persona-aware evaluation frameworks for AI moral reasoning. 

---
# OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics 

**Authors**: Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zachary C. Lipton, J. Zico Kolter, Pratyush Maini  

**Link**: [PDF](https://arxiv.org/pdf/2506.12618)  

**Abstract**: Robust unlearning is crucial for safely deploying large language models (LLMs) in environments where data privacy, model safety, and regulatory compliance must be ensured. Yet the task is inherently challenging, partly due to difficulties in reliably measuring whether unlearning has truly occurred. Moreover, fragmentation in current methodologies and inconsistent evaluation metrics hinder comparative analysis and reproducibility. To unify and accelerate research efforts, we introduce OpenUnlearning, a standardized and extensible framework designed explicitly for benchmarking both LLM unlearning methods and metrics. OpenUnlearning integrates 9 unlearning algorithms and 16 diverse evaluations across 3 leading benchmarks (TOFU, MUSE, and WMDP) and also enables analyses of forgetting behaviors across 450+ checkpoints we publicly release. Leveraging OpenUnlearning, we propose a novel meta-evaluation benchmark focused specifically on assessing the faithfulness and robustness of evaluation metrics themselves. We also benchmark diverse unlearning methods and provide a comparative analysis against an extensive evaluation suite. Overall, we establish a clear, community-driven pathway toward rigorous development in LLM unlearning research. 

---
# Rethinking Hate Speech Detection on Social Media: Can LLMs Replace Traditional Models? 

**Authors**: Daman Deep Singh, Ramanuj Bhattacharjee, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2506.12744)  

**Abstract**: Hate speech detection across contemporary social media presents unique challenges due to linguistic diversity and the informal nature of online discourse. These challenges are further amplified in settings involving code-mixing, transliteration, and culturally nuanced expressions. While fine-tuned transformer models, such as BERT, have become standard for this task, we argue that recent large language models (LLMs) not only surpass them but also redefine the landscape of hate speech detection more broadly. To support this claim, we introduce IndoHateMix, a diverse, high-quality dataset capturing Hindi-English code-mixing and transliteration in the Indian context, providing a realistic benchmark to evaluate model robustness in complex multilingual scenarios where existing NLP methods often struggle. Our extensive experiments show that cutting-edge LLMs (such as LLaMA-3.1) consistently outperform task-specific BERT-based models, even when fine-tuned on significantly less data. With their superior generalization and adaptability, LLMs offer a transformative approach to mitigating online hate in diverse environments. This raises the question of whether future works should prioritize developing specialized models or focus on curating richer and more varied datasets to further enhance the effectiveness of LLMs. 

---
# OneEval: Benchmarking LLM Knowledge-intensive Reasoning over Diverse Knowledge Bases 

**Authors**: Yongrui Chen, Zhiqiang Liu, Jing Yu, Lin Ren, Nan Hu, Xinbang Dai, Jiajun Liu, Jiazhen Kang, Shenyu Zhang, Xinda Wang, Keyan Ding, Pengfei Shen, Haolei Zhu, Hongjie Deng, Yisong Wang, Tongtong Wu, Sheng Bi, Wen Zhang, Tianxing Wu, Qiu Ji, Haofen Wang, Wenliang Chen, Huajun Chen, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12577)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial progress on reasoning tasks involving unstructured text, yet their capabilities significantly deteriorate when reasoning requires integrating structured external knowledge such as knowledge graphs, code snippets, or formal logic. This limitation is partly due to the absence of benchmarks capable of systematically evaluating LLM performance across diverse structured knowledge modalities. To address this gap, we introduce \textbf{\textsc{OneEval}}, a comprehensive benchmark explicitly designed to assess the knowledge-intensive reasoning capabilities of LLMs across four structured knowledge modalities, unstructured text, knowledge graphs, code, and formal logic, and five critical domains (general knowledge, government, science, law, and programming). \textsc{OneEval} comprises 4,019 carefully curated instances and includes a challenging subset, \textsc{OneEval}\textsubscript{Hard}, consisting of 1,285 particularly difficult cases. Through extensive evaluation of 18 state-of-the-art open-source and proprietary LLMs, we establish three core findings: a) \emph{persistent limitations in structured reasoning}, with even the strongest model achieving only 32.2\% accuracy on \textsc{OneEval}\textsubscript{Hard}; b) \emph{performance consistently declines as the structural complexity of the knowledge base increases}, with accuracy dropping sharply from 53\% (textual reasoning) to 25\% (formal logic); and c) \emph{diminishing returns from extended reasoning chains}, highlighting the critical need for models to adapt reasoning depth appropriately to task complexity. We release the \textsc{OneEval} datasets, evaluation scripts, and baseline results publicly, accompanied by a leaderboard to facilitate ongoing advancements in structured knowledge reasoning. 

---
# Detection, Classification, and Mitigation of Gender Bias in Large Language Models 

**Authors**: Xiaoqing Cheng, Hongying Zan, Lulu Kong, Jinwang Song, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.12527)  

**Abstract**: With the rapid development of large language models (LLMs), they have significantly improved efficiency across a wide range of domains. However, recent studies have revealed that LLMs often exhibit gender bias, leading to serious social implications. Detecting, classifying, and mitigating gender bias in LLMs has therefore become a critical research focus. In the NLPCC 2025 Shared Task 7: Chinese Corpus for Gender Bias Detection, Classification and Mitigation Challenge, we investigate how to enhance the capabilities of LLMs in gender bias detection, classification, and mitigation. We adopt reinforcement learning, chain-of-thoughts (CoT) reasoning, and supervised fine-tuning to handle different Subtasks. Specifically, for Subtasks 1 and 2, we leverage the internal reasoning capabilities of LLMs to guide multi-step thinking in a staged manner, which simplifies complex biased queries and improves response accuracy. For Subtask 3, we employ a reinforcement learning-based approach, annotating a preference dataset using GPT-4. We then apply Direct Preference Optimization (DPO) to mitigate gender bias by introducing a loss function that explicitly favors less biased completions over biased ones. Our approach ranked first across all three subtasks of the NLPCC 2025 Shared Task 7. 

---
# Improving Factuality for Dialogue Response Generation via Graph-Based Knowledge Augmentation 

**Authors**: Xiangyan Chen, Yujian Gan, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2506.12496)  

**Abstract**: Large Language Models (LLMs) succeed in many natural language processing tasks. However, their tendency to hallucinate - generate plausible but inconsistent or factually incorrect text - can cause problems in certain tasks, including response generation in dialogue. To mitigate this issue, knowledge-augmented methods have shown promise in reducing hallucinations. Here, we introduce a novel framework designed to enhance the factuality of dialogue response generation, as well as an approach to evaluate dialogue factual accuracy. Our framework combines a knowledge triple retriever, a dialogue rewrite, and knowledge-enhanced response generation to produce more accurate and grounded dialogue responses. To further evaluate generated responses, we propose a revised fact score that addresses the limitations of existing fact-score methods in dialogue settings, providing a more reliable assessment of factual consistency. We evaluate our methods using different baselines on the OpendialKG and HybriDialogue datasets. Our methods significantly improve factuality compared to other graph knowledge-augmentation baselines, including the state-of-the-art G-retriever. The code will be released on GitHub. 

---
# FinLMM-R1: Enhancing Financial Reasoning in LMM through Scalable Data and Reward Design 

**Authors**: Kai Lan, Jiayong Zhu, Jiangtong Li, Dawei Cheng, Guang Chen, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13066)  

**Abstract**: Large Multimodal Models (LMMs) demonstrate significant cross-modal reasoning capabilities. However, financial applications face challenges due to the lack of high-quality multimodal reasoning datasets and the inefficiency of existing training paradigms for reasoning enhancement. To address these issues, we propose an integrated framework, FinLMM-R1, combining an automated and scalable pipeline for data construction with enhanced training strategies to improve the multimodal reasoning of LMM. The Automated and Scalable Pipeline (ASP) resolves textual-visual misalignment in financial reports through a separate paradigm of question-answer generation and image-question alignment, ensuring data integrity and extraction efficiency. Through ASP, we collect 89,378 aligned image-question pairs from 23,397 financial reports, covering tasks such as arithmetic reasoning, statistics reasoning, financial explanation, and financial knowledge. Moreover, we introduce the Thinking with Adversarial Reward in LMM (TAR-LMM), extending the prior two-stage training framework [1] with additional reward mechanisms. In the first stage, we focus on text-only tasks with format and accuracy rewards to guide the model in generating well-structured thinking contents. In the second stage, we construct multi-image contrastive samples with additional reward components including image selection, thinking content length, and adversarial reward to jointly optimize the LMM across visual perception, reasoning efficiency, and logical coherence. Extensive experiments on 7 benchmarks show ASP-derived dataset and training framework significantly improve answer accuracy and reasoning depth over existing reasoning LMMs in both general and financial multimodal contexts. 

---
# Exploring Cultural Variations in Moral Judgments with Large Language Models 

**Authors**: Hadi Mohammadi, Efthymia Papadopoulou, Yasmeen F.S.S. Meijer, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2506.12433)  

**Abstract**: Large Language Models (LLMs) have shown strong performance across many tasks, but their ability to capture culturally diverse moral values remains unclear. In this paper, we examine whether LLMs can mirror variations in moral attitudes reported by two major cross-cultural surveys: the World Values Survey and the PEW Research Center's Global Attitudes Survey. We compare smaller, monolingual, and multilingual models (GPT-2, OPT, BLOOMZ, and Qwen) with more recent instruction-tuned models (GPT-4o, GPT-4o-mini, Gemma-2-9b-it, and Llama-3.3-70B-Instruct). Using log-probability-based moral justifiability scores, we correlate each model's outputs with survey data covering a broad set of ethical topics. Our results show that many earlier or smaller models often produce near-zero or negative correlations with human judgments. In contrast, advanced instruction-tuned models (including GPT-4o and GPT-4o-mini) achieve substantially higher positive correlations, suggesting they better reflect real-world moral attitudes. While scaling up model size and using instruction tuning can improve alignment with cross-cultural moral norms, challenges remain for certain topics and regions. We discuss these findings in relation to bias analysis, training data diversity, and strategies for improving the cultural sensitivity of LLMs. 

---
# Between Predictability and Randomness: Seeking Artistic Inspiration from AI Generative Models 

**Authors**: Olga Vechtomova  

**Link**: [PDF](https://arxiv.org/pdf/2506.12634)  

**Abstract**: Artistic inspiration often emerges from language that is open to interpretation. This paper explores the use of AI-generated poetic lines as stimuli for creativity. Through analysis of two generative AI approaches--lines generated by Long Short-Term Memory Variational Autoencoders (LSTM-VAE) and complete poems by Large Language Models (LLMs)--I demonstrate that LSTM-VAE lines achieve their evocative impact through a combination of resonant imagery and productive indeterminacy. While LLMs produce technically accomplished poetry with conventional patterns, LSTM-VAE lines can engage the artist through semantic openness, unconventional combinations, and fragments that resist closure. Through the composition of an original poem, where narrative emerged organically through engagement with LSTM-VAE generated lines rather than following a predetermined structure, I demonstrate how these characteristics can serve as evocative starting points for authentic artistic expression. 

---
# SciDA: Scientific Dynamic Assessor of LLMs 

**Authors**: Junting Zhou, Tingjia Miao, Yiyan Liao, Qichao Wang, Zhoufutu Wen, Yanqin Wang, Yunjie Huang, Ge Yan, Leqi Wang, Yucheng Xia, Hongwan Gao, Yuansong Zeng, Renjie Zheng, Chen Dun, Yitao Liang, Tong Yang, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12909)  

**Abstract**: Advancement in Large Language Models (LLMs) reasoning capabilities enables them to solve scientific problems with enhanced efficacy. Thereby, a high-quality benchmark for comprehensive and appropriate assessment holds significance, while existing ones either confront the risk of data contamination or lack involved disciplines. To be specific, due to the data source overlap of LLMs training and static benchmark, the keys or number pattern of answers inadvertently memorized (i.e. data contamination), leading to systematic overestimation of their reasoning capabilities, especially numerical reasoning. We propose SciDA, a multidisciplinary benchmark that consists exclusively of over 1k Olympic-level numerical computation problems, allowing randomized numerical initializations for each inference round to avoid reliance on fixed numerical patterns. We conduct a series of experiments with both closed-source and open-source top-performing LLMs, and it is observed that the performance of LLMs drop significantly under random numerical initialization. Thus, we provide truthful and unbiased assessments of the numerical reasoning capabilities of LLMs. The data is available at this https URL 

---
# Towards Building General Purpose Embedding Models for Industry 4.0 Agents 

**Authors**: Christodoulos Constantinides, Shuxin Lin, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2506.12607)  

**Abstract**: In this work we focus on improving language models' understanding for asset maintenance to guide the engineer's decisions and minimize asset downtime. Given a set of tasks expressed in natural language for Industry 4.0 domain, each associated with queries related to a specific asset, we want to recommend relevant items and generalize to queries of similar assets. A task may involve identifying relevant sensors given a query about an asset's failure mode.
Our approach begins with gathering a qualitative, expert-vetted knowledge base to construct nine asset-specific task datasets. To create more contextually informed embeddings, we augment the input tasks using Large Language Models (LLMs), providing concise descriptions of the entities involved in the queries. This embedding model is then integrated with a Reasoning and Acting agent (ReAct), which serves as a powerful tool for answering complex user queries that require multi-step reasoning, planning, and knowledge inference.
Through ablation studies, we demonstrate that: (a) LLM query augmentation improves the quality of embeddings, (b) Contrastive loss and other methods that avoid in-batch negatives are superior for datasets with queries related to many items, and (c) It is crucial to balance positive and negative in-batch samples. After training and testing on our dataset, we observe a substantial improvement: HIT@1 increases by +54.2%, MAP@100 by +50.1%, and NDCG@10 by +54.7%, averaged across all tasks and models. Additionally, we empirically demonstrate the model's planning and tool invocation capabilities when answering complex questions related to industrial asset maintenance, showcasing its effectiveness in supporting Subject Matter Experts (SMEs) in their day-to-day operations. 

---
# Language Surgery in Multilingual Large Language Models 

**Authors**: Joanito Agili Lopo, Muhammad Ravi Shulthan Habibi, Tack Hwa Wong, Muhammad Ilham Ghozali, Fajri Koto, Genta Indra Winata, Peerat Limkonchotiwat, Alham Fikri Aji, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.12450)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across tasks and languages, revolutionizing natural language processing. This paper investigates the naturally emerging representation alignment in LLMs, particularly in the middle layers, and its implications for disentangling language-specific and language-agnostic information. We empirically confirm the existence of this alignment, analyze its behavior in comparison to explicitly designed alignment models, and demonstrate its potential for language-specific manipulation without semantic degradation. Building on these findings, we propose Inference-Time Language Control (ITLC), a novel method that leverages latent injection to enable precise cross-lingual language control and mitigate language confusion in LLMs. Our experiments highlight ITLC's strong cross-lingual control capabilities while preserving semantic integrity in target languages. Furthermore, we demonstrate its effectiveness in alleviating the cross-lingual language confusion problem, which persists even in current large-scale LLMs, leading to inconsistent language generation. This work advances our understanding of representation alignment in LLMs and introduces a practical solution for enhancing their cross-lingual performance. 

---
# Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics 

**Authors**: Asifullah khan, Muhammad Zaeem Khan, Saleha Jamshed, Sadia Ahmad, Aleesha Zainab, Kaynat Khatib, Faria Bibi, Abdul Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12365)  

**Abstract**: This survey paper outlines the key developments in the field of Large Language Models (LLMs), such as enhancing their reasoning skills, adaptability to various tasks, increased computational efficiency, and ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. They also manage to do more with less by applying scaling and optimization tricks for computing power conservation. This survey also offers a broader perspective on recent advancements in LLMs going beyond isolated aspects such as model architecture or ethical concerns. It categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. It also identifies underexplored areas such as interpretability, cross-modal integration and sustainability. With recent progress, challenges like huge computational costs, biases, and ethical risks remain constant. Addressing these requires bias mitigation, transparent decision-making, and clear ethical guidelines. Future research will focus on enhancing models ability to handle multiple input, thereby making them more intelligent, safe, and reliable. 

---
# Investigating the Effects of Cognitive Biases in Prompts on Large Language Model Outputs 

**Authors**: Yan Sun, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2506.12338)  

**Abstract**: This paper investigates the influence of cognitive biases on Large Language Models (LLMs) outputs. Cognitive biases, such as confirmation and availability biases, can distort user inputs through prompts, potentially leading to unfaithful and misleading outputs from LLMs. Using a systematic framework, our study introduces various cognitive biases into prompts and assesses their impact on LLM accuracy across multiple benchmark datasets, including general and financial Q&A scenarios. The results demonstrate that even subtle biases can significantly alter LLM answer choices, highlighting a critical need for bias-aware prompt design and mitigation strategy. Additionally, our attention weight analysis highlights how these biases can alter the internal decision-making processes of LLMs, affecting the attention distribution in ways that are associated with output inaccuracies. This research has implications for Al developers and users in enhancing the robustness and reliability of Al applications in diverse domains. 

---
# A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages 

**Authors**: Tatiana Ankinina, Jan Cegin, Jakub Simko, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2506.12158)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models. 

---
# UCD: Unlearning in LLMs via Contrastive Decoding 

**Authors**: Vinith M. Suriyakumar, Ayush Sekhari, Ashia Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2506.12097)  

**Abstract**: Machine unlearning aims to remove specific information, e.g. sensitive or undesirable content, from large language models (LLMs) while preserving overall performance. We propose an inference-time unlearning algorithm that uses contrastive decoding, leveraging two auxiliary smaller models, one trained without the forget set and one trained with it, to guide the outputs of the original model using their difference during inference. Our strategy substantially improves the tradeoff between unlearning effectiveness and model utility. We evaluate our approach on two unlearning benchmarks, TOFU and MUSE. Results show notable gains in both forget quality and retained performance in comparison to prior approaches, suggesting that incorporating contrastive decoding can offer an efficient, practical avenue for unlearning concepts in large-scale models. 

---
# From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time Alignment 

**Authors**: Bin Xie, Bingbing Xu, Yige Yuan, Shengmao Zhu, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12446)  

**Abstract**: Inference-time alignment methods have gained significant attention for their efficiency and effectiveness in aligning large language models (LLMs) with human preferences. However, existing dominant approaches using reward-guided search (RGS) primarily rely on outcome reward models (ORMs), which suffer from a critical granularity mismatch: ORMs are designed to provide outcome rewards for complete responses, while RGS methods rely on process rewards to guide the policy, leading to inconsistent scoring and suboptimal alignment. To address this challenge, we introduce process reward models (PRMs) into RGS and argue that an ideal PRM should satisfy two objectives: Score Consistency, ensuring coherent evaluation across partial and complete responses, and Preference Consistency, aligning partial sequence assessments with human preferences. Based on these, we propose SP-PRM, a novel dual-consistency framework integrating score consistency-based and preference consistency-based partial evaluation modules without relying on human annotation. Extensive experiments on dialogue, summarization, and reasoning tasks demonstrate that SP-PRM substantially enhances existing RGS methods, achieving a 3.6%-10.3% improvement in GPT-4 evaluation scores across all tasks. 

---
# Continuously Updating Digital Twins using Large Language Models 

**Authors**: Harry Amad, Nicolás Astorga, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12091)  

**Abstract**: Digital twins are models of real-world systems that can simulate their dynamics in response to potential actions. In complex settings, the state and action variables, and available data and knowledge relevant to a system can constantly change, requiring digital twins to continuously update with these changes to remain relevant. Current approaches struggle in this regard, as they require fixed, well-defined modelling environments, and they cannot adapt to novel variables without re-designs, or incorporate new information without re-training. To address this, we frame digital twinning as an in-context learning problem using large language models, enabling seamless updates to the twin at inference time. We develop CALM-DT, a Context-Adaptive Language Model-based Digital Twin that can accurately simulate across diverse state-action spaces using in-context learning alone by utilising fine-tuned encoders for sample retrieval. We empirically demonstrate CALM-DT's competitive performance with existing digital twin approaches, and its unique ability to adapt to changes in its modelling environment without parameter updates. 

---
# Focusing on Students, not Machines: Grounded Question Generation and Automated Answer Grading 

**Authors**: Gérôme Meyer, Philip Breuer  

**Link**: [PDF](https://arxiv.org/pdf/2506.12066)  

**Abstract**: Digital technologies are increasingly used in education to reduce the workload of teachers and students. However, creating open-ended study or examination questions and grading their answers is still a tedious task. This thesis presents the foundation for a system that generates questions grounded in class materials and automatically grades student answers. It introduces a sophisticated method for chunking documents with a visual layout, specifically targeting PDF documents. This method enhances the accuracy of downstream tasks, including Retrieval Augmented Generation (RAG). Our thesis demonstrates that high-quality questions and reference answers can be generated from study material. Further, it introduces a new benchmark for automated grading of short answers to facilitate comparison of automated grading systems. An evaluation of various grading systems is conducted and indicates that Large Language Models (LLMs) can generalise to the task of automated grading of short answers from their pre-training tasks. As with other tasks, increasing the parameter size of the LLMs leads to greater performance. Currently, available systems still need human oversight, especially in examination scenarios. 

---
# TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks 

**Authors**: Zhou Chen, Zhiqiang Wei, Yuqi Bai, Xue Xiong, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12473)  

**Abstract**: Model routing allocates queries to the suitable model, improving system performance while reducing costs. However, existing routing methods face practical limitations that hinder scalability in large-scale applications and struggle to keep up with the rapid growth of the large language model (LLM) ecosystem. To tackle these challenges, we propose TagRouter, a training-free model routing method designed to optimize the synergy among multiple LLMs for open-domain text generation tasks. Experimental results demonstrate that TagRouter outperforms 13 baseline methods, increasing the accept rate of system by 6.15% and reducing costs by 17.20%, achieving optimal cost-efficiency. Our findings provides the LLM community with an efficient and scalable solution for model ensembling, offering users an evolvable "super model." 

---
# Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs 

**Authors**: Chenqian Le, Ziheng Gong, Chihang Wang, Haowei Ni, Panfeng Li, Xupeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12182)  

**Abstract**: Large language models (LLMs) have shown great potential in medical question answering (MedQA), yet adapting them to biomedical reasoning remains challenging due to domain-specific complexity and limited supervision. In this work, we study how prompt design and lightweight fine-tuning affect the performance of open-source LLMs on PubMedQA, a benchmark for multiple-choice biomedical questions. We focus on two widely used prompting strategies - standard instruction prompts and Chain-of-Thought (CoT) prompts - and apply QLoRA for parameter-efficient instruction tuning. Across multiple model families and sizes, our experiments show that CoT prompting alone can improve reasoning in zero-shot settings, while instruction tuning significantly boosts accuracy. However, fine-tuning on CoT prompts does not universally enhance performance and may even degrade it for certain larger models. These findings suggest that reasoning-aware prompts are useful, but their benefits are model- and scale-dependent. Our study offers practical insights into combining prompt engineering with efficient finetuning for medical QA applications. 

---
# ChatbotManip: A Dataset to Facilitate Evaluation and Oversight of Manipulative Chatbot Behaviour 

**Authors**: Jack Contro, Simrat Deol, Yulan He, Martim Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2506.12090)  

**Abstract**: This paper introduces ChatbotManip, a novel dataset for studying manipulation in Chatbots. It contains simulated generated conversations between a chatbot and a (simulated) user, where the chatbot is explicitly asked to showcase manipulation tactics, persuade the user towards some goal, or simply be helpful. We consider a diverse set of chatbot manipulation contexts, from consumer and personal advice to citizen advice and controversial proposition argumentation. Each conversation is annotated by human annotators for both general manipulation and specific manipulation tactics. Our research reveals three key findings. First, Large Language Models (LLMs) can be manipulative when explicitly instructed, with annotators identifying manipulation in approximately 84\% of such conversations. Second, even when only instructed to be ``persuasive'' without explicit manipulation prompts, LLMs frequently default to controversial manipulative strategies, particularly gaslighting and fear enhancement. Third, small fine-tuned open source models, such as BERT+BiLSTM have a performance comparable to zero-shot classification with larger models like Gemini 2.5 pro in detecting manipulation, but are not yet reliable for real-world oversight. Our work provides important insights for AI safety research and highlights the need of addressing manipulation risks as LLMs are increasingly deployed in consumer-facing applications. 

---
# SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists 

**Authors**: Lynn Khellaf, Ipek Baris Schlicht, Tilman Mirass, Julia Bayer, Tilman Wagner, Ruben Bouwmeester  

**Link**: [PDF](https://arxiv.org/pdf/2506.13188)  

**Abstract**: OpenStreetMap (OSM) is a vital resource for investigative journalists doing geolocation verification. However, existing tools to query OSM data such as Overpass Turbo require familiarity with complex query languages, creating barriers for non-technical users. We present SPOT, an open source natural language interface that makes OSM's rich, tag-based geographic data more accessible through intuitive scene descriptions. SPOT interprets user inputs as structured representations of geospatial object configurations using fine-tuned Large Language Models (LLMs), with results being displayed in an interactive map interface. While more general geospatial search tasks are conceivable, SPOT is specifically designed for use in investigative journalism, addressing real-world challenges such as hallucinations in model output, inconsistencies in OSM tagging, and the noisy nature of user input. It combines a novel synthetic data pipeline with a semantic bundling system to enable robust, accurate query generation. To our knowledge, SPOT is the first system to achieve reliable natural language access to OSM data at this level of accuracy. By lowering the technical barrier to geolocation verification, SPOT contributes a practical tool to the broader efforts to support fact-checking and combat disinformation. 

---
# Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition? 

**Authors**: Xiangyang Li, Xiaopeng Li, Kuicai Dong, Quanhu Zhang, Rongju Ruan, Xinyi Dai, Xiaoshuang Liu, Shengchun Xu, Yasheng Wang, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12713)  

**Abstract**: Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(this https URL). 

---
# SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression 

**Authors**: Yucheng Li, Surin Ahn, Huiqiang Jiang, Amir H. Abdi, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12707)  

**Abstract**: Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at this https URL. 

---
# QiMeng-Attention: SOTA Attention Operator is generated by SOTA Attention Algorithm 

**Authors**: Qirui Zhou, Shaohui Peng, Weiqiang Xiong, Haixin Chen, Yuanbo Wen, Haochen Li, Ling Li, Qi Guo, Yongwei Zhao, Ke Gao, Ruizhi Chen, Yanjun Wu, Chen Zhao, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12355)  

**Abstract**: The attention operator remains a critical performance bottleneck in large language models (LLMs), particularly for long-context scenarios. While FlashAttention is the most widely used and effective GPU-aware acceleration algorithm, it must require time-consuming and hardware-specific manual implementation, limiting adaptability across GPU architectures. Existing LLMs have shown a lot of promise in code generation tasks, but struggle to generate high-performance attention code. The key challenge is it cannot comprehend the complex data flow and computation process of the attention operator and utilize low-level primitive to exploit GPU performance.
To address the above challenge, we propose an LLM-friendly Thinking Language (LLM-TL) to help LLMs decouple the generation of high-level optimization logic and low-level implementation on GPU, and enhance LLMs' understanding of attention operator. Along with a 2-stage reasoning workflow, TL-Code generation and translation, the LLMs can automatically generate FlashAttention implementation on diverse GPUs, establishing a self-optimizing paradigm for generating high-performance attention operators in attention-centric algorithms. Verified on A100, RTX8000, and T4 GPUs, the performance of our methods significantly outshines that of vanilla LLMs, achieving a speed-up of up to 35.16x. Besides, our method not only surpasses human-optimized libraries (cuDNN and official library) in most scenarios but also extends support to unsupported hardware and data types, reducing development time from months to minutes compared with human experts. 

---
# Can LLMs Generate High-Quality Test Cases for Algorithm Problems? TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure 

**Authors**: Zheyuan Yang, Zexi Kuang, Xue Xia, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12278)  

**Abstract**: We introduce TestCase-Eval, a new benchmark for systematic evaluation of LLMs in test-case generation. TestCase-Eval includes 500 algorithm problems and 100,000 human-crafted solutions from the Codeforces platform. It focuses on two pivotal tasks: (1) Fault Coverage, which measures how well LLM-generated test sets probe diverse input scenarios and cover a wide range of potential failure modes. (2) Fault Exposure, which evaluates whether LLMs can craft a tailored test input that reveals a specific incorrect code implementation. We provide a comprehensive assessment of 19 state-of-the-art open-source and proprietary LLMs on TestCase-Eval, offering insights into their strengths and limitations in generating effective test cases for algorithm problems. 

---
# InfoFlood: Jailbreaking Large Language Models with Information Overload 

**Authors**: Advait Yadav, Haibo Jin, Man Luo, Jun Zhuang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12274)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models.
In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload.
To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent.
We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks. 

---
# Zero-Shot Scene Understanding with Multimodal Large Language Models for Automated Vehicles 

**Authors**: Mohammed Elhenawy, Shadi Jaradat, Taqwa I. Alhadidi, Huthaifa I. Ashqar, Ahmed Jaber, Andry Rakotonirainy, Mohammad Abu Tami  

**Link**: [PDF](https://arxiv.org/pdf/2506.12232)  

**Abstract**: Scene understanding is critical for various downstream tasks in autonomous driving, including facilitating driver-agent communication and enhancing human-centered explainability of autonomous vehicle (AV) decisions. This paper evaluates the capability of four multimodal large language models (MLLMs), including relatively small models, to understand scenes in a zero-shot, in-context learning setting. Additionally, we explore whether combining these models using an ensemble approach with majority voting can enhance scene understanding performance. Our experiments demonstrate that GPT-4o, the largest model, outperforms the others in scene understanding. However, the performance gap between GPT-4o and the smaller models is relatively modest, suggesting that advanced techniques such as improved in-context learning, retrieval-augmented generation (RAG), or fine-tuning could further optimize the smaller models' performance. We also observe mixed results with the ensemble approach: while some scene attributes show improvement in performance metrics such as F1-score, others experience a decline. These findings highlight the need for more sophisticated ensemble techniques to achieve consistent gains across all scene attributes. This study underscores the potential of leveraging MLLMs for scene understanding and provides insights into optimizing their performance for autonomous driving applications. 

---
# Artificial Intelligence and Civil Discourse: How LLMs Moderate Climate Change Conversations 

**Authors**: Wenlu Fan, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12077)  

**Abstract**: As large language models (LLMs) become increasingly integrated into online platforms and digital communication spaces, their potential to influence public discourse - particularly in contentious areas like climate change - requires systematic investigation. This study examines how LLMs naturally moderate climate change conversations through their distinct communicative behaviors. We conduct a comparative analysis of conversations between LLMs and human users on social media platforms, using five advanced models: three open-source LLMs (Gemma, Llama 3, and Llama 3.3) and two commercial systems (GPT-4o by OpenAI and Claude 3.5 by Anthropic). Through sentiment analysis, we assess the emotional characteristics of responses from both LLMs and humans. The results reveal two key mechanisms through which LLMs moderate discourse: first, LLMs consistently display emotional neutrality, showing far less polarized sentiment than human users. Second, LLMs maintain lower emotional intensity across contexts, creating a stabilizing effect in conversations. These findings suggest that LLMs possess inherent moderating capacities that could improve the quality of public discourse on controversial topics. This research enhances our understanding of how AI might support more civil and constructive climate change discussions and informs the design of AI-assisted communication tools. 

---
# LTRR: Learning To Rank Retrievers for LLMs 

**Authors**: To Eun Kim, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13743)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems. 

---
