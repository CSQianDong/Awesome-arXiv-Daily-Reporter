# Criteria-Based LLM Relevance Judgments 

**Authors**: Naghmeh Farzi, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2507.09488)  

**Abstract**: Relevance judgments are crucial for evaluating information retrieval systems, but traditional human-annotated labels are time-consuming and expensive. As a result, many researchers turn to automatic alternatives to accelerate method development. Among these, Large Language Models (LLMs) provide a scalable solution by generating relevance labels directly through prompting. However, prompting an LLM for a relevance label without constraints often results in not only incorrect predictions but also outputs that are difficult for humans to interpret. We propose the Multi-Criteria framework for LLM-based relevance judgments, decomposing the notion of relevance into multiple criteria--such as exactness, coverage, topicality, and contextual fit--to improve the robustness and interpretability of retrieval evaluations compared to direct grading methods. We validate this approach on three datasets: the TREC Deep Learning tracks from 2019 and 2020, as well as LLMJudge (based on TREC DL 2023). Our results demonstrate that Multi-Criteria judgments enhance the system ranking/leaderboard performance. Moreover, we highlight the strengths and limitations of this approach relative to direct grading approaches, offering insights that can guide the development of future automatic evaluation frameworks in information retrieval. 

---
# Does UMBRELA Work on Other LLMs? 

**Authors**: Naghmeh Farzi, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2507.09483)  

**Abstract**: We reproduce the UMBRELA LLM Judge evaluation framework across a range of large language models (LLMs) to assess its generalizability beyond the original study. Our investigation evaluates how LLM choice affects relevance assessment accuracy, focusing on leaderboard rank correlation and per-label agreement metrics. Results demonstrate that UMBRELA with DeepSeek V3 obtains very comparable performance to GPT-4o (used in original work). For LLaMA-3.3-70B we obtain slightly lower performance, which further degrades with smaller LLMs. 

---
# DS@GT at Touché: Large Language Models for Retrieval-Augmented Debate 

**Authors**: Anthony Miyaguchi, Conor Johnston, Aaryan Potdar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09090)  

**Abstract**: Large Language Models (LLMs) demonstrate strong conversational abilities. In this Working Paper, we study them in the context of debating in two ways: their ability to perform in a structured debate along with a dataset of arguments to use and their ability to evaluate utterances throughout the debate. We deploy six leading publicly available models from three providers for the Retrieval-Augmented Debate and Evaluation. The evaluation is performed by measuring four key metrics: Quality, Quantity, Manner, and Relation. Throughout this task, we found that although LLMs perform well in debates when given related arguments, they tend to be verbose in responses yet consistent in evaluation. The accompanying source code for this paper is located at this https URL. 

---
# Knowledge Conceptualization Impacts RAG Efficacy 

**Authors**: Chris Davis Jaldi, Anmol Saini, Elham Ghiasi, O. Divine Eziolise, Cogan Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09389)  

**Abstract**: Explainability and interpretability are cornerstones of frontier and next-generation artificial intelligence (AI) systems. This is especially true in recent systems, such as large language models (LLMs), and more broadly, generative AI. On the other hand, adaptability to new domains, contexts, or scenarios is also an important aspect for a successful system. As such, we are particularly interested in how we can merge these two efforts, that is, investigating the design of transferable and interpretable neurosymbolic AI systems. Specifically, we focus on a class of systems referred to as ''Agentic Retrieval-Augmented Generation'' systems, which actively select, interpret, and query knowledge sources in response to natural language prompts. In this paper, we systematically evaluate how different conceptualizations and representations of knowledge, particularly the structure and complexity, impact an AI agent (in this case, an LLM) in effectively querying a triplestore. We report our results, which show that there are impacts from both approaches, and we discuss their impact and implications. 

---
# Retrieval-Augmented Recommendation Explanation Generation with Hierarchical Aggregation 

**Authors**: Bangcheng Sun, Yazhe Chen, Jilin Yang, Xiaodong Li, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09188)  

**Abstract**: Explainable Recommender System (ExRec) provides transparency to the recommendation process, increasing users' trust and boosting the operation of online services. With the rise of large language models (LLMs), whose extensive world knowledge and nuanced language understanding enable the generation of human-like, contextually grounded explanations, LLM-powered ExRec has gained great momentum. However, existing LLM-based ExRec models suffer from profile deviation and high retrieval overhead, hindering their deployment. To address these issues, we propose Retrieval-Augmented Recommendation Explanation Generation with Hierarchical Aggregation (REXHA). Specifically, we design a hierarchical aggregation based profiling module that comprehensively considers user and item review information, hierarchically summarizing and constructing holistic profiles. Furthermore, we introduce an efficient retrieval module using two types of pseudo-document queries to retrieve relevant reviews to enhance the generation of recommendation explanations, effectively reducing retrieval latency and improving the recall of relevant reviews. Extensive experiments demonstrate that our method outperforms existing approaches by up to 12.6% w.r.t. the explanation quality while achieving high retrieval efficiency. 

---
# Overview of the TREC 2023 deep learning track 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Hossein A. Rahmani, Daniel Campos, Jimmy Lin, Ellen M. Voorhees, Ian Soboroff  

**Link**: [PDF](https://arxiv.org/pdf/2507.08890)  

**Abstract**: This is the fifth year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human-annotated training labels available for both passage and document ranking tasks. We mostly repeated last year's design, to get another matching test set, based on the larger, cleaner, less-biased v2 passage and document set, with passage ranking as primary and document ranking as a secondary task (using labels inferred from passage). As we did last year, we sample from MS MARCO queries that were completely held out, unused in corpus construction, unlike the test queries in the first three years. This approach yields a more difficult test with more headroom for improvement. Alongside the usual MS MARCO (human) queries from MS MARCO, this year we generated synthetic queries using a fine-tuned T5 model and using a GPT-4 prompt.
The new headline result this year is that runs using Large Language Model (LLM) prompting in some way outperformed runs that use the "nnlm" approach, which was the best approach in the previous four years. Since this is the last year of the track, future iterations of prompt-based ranking can happen in other tracks. Human relevance assessments were applied to all query types, not just human MS MARCO queries. Evaluation using synthetic queries gave similar results to human queries, with system ordering agreement of $\tau=0.8487$. However, human effort was needed to select a subset of the synthetic queries that were usable. We did not see clear evidence of bias, where runs using GPT-4 were favored when evaluated using synthetic GPT-4 queries, or where runs using T5 were favored when evaluated on synthetic T5 queries. 

---
# GraphRunner: A Multi-Stage Framework for Efficient and Accurate Graph-Based Retrieval 

**Authors**: Savini Kashmira, Jayanaka L. Dantanarayana, Krisztián Flautner, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2507.08945)  

**Abstract**: Conventional Retrieval Augmented Generation (RAG) approaches are common in text-based applications. However, they struggle with structured, interconnected datasets like knowledge graphs, where understanding underlying relationships is crucial for accurate retrieval. A common direction in graph-based retrieval employs iterative, rule-based traversal guided by Large Language Models (LLMs). Such existing iterative methods typically combine reasoning with single hop traversal at each step, making them vulnerable to LLM reasoning errors and hallucinations that ultimately hinder the retrieval of relevant information.
To address these limitations, we propose GraphRunner, a novel graph-based retrieval framework that operates in three distinct stages: planning, verification, and execution. This introduces high-level traversal actions that enable multi-hop exploration in a single step. It also generates a holistic traversal plan, which is verified against the graph structure and pre-defined traversal actions, reducing reasoning errors and detecting hallucinations before execution. GraphRunner significantly reduces LLM reasoning errors and detects hallucinations through validation. Our evaluation using the GRBench dataset shows that GraphRunner consistently outperforms existing approaches, achieving 10-50% performance improvements over the strongest baseline while reducing inference cost by 3.0-12.9x and response generation time by 2.5-7.1x, making it significantly more robust and efficient for graph-based retrieval tasks. 

---
# Automating SPARQL Query Translations between DBpedia and Wikidata 

**Authors**: Malte Christian Bartels, Debayan Banerjee, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2507.10045)  

**Abstract**: This paper investigates whether state-of-the-art Large Language Models (LLMs) can automatically translate SPARQL between popular Knowledge Graph (KG) schemas. We focus on translations between the DBpedia and Wikidata KG, and later on DBLP and OpenAlex KG. This study addresses a notable gap in KG interoperability research by rigorously evaluating LLM performance on SPARQL-to-SPARQL translation. Two benchmarks are assembled, where the first align 100 DBpedia-Wikidata queries from QALD-9-Plus; the second contains 100 DBLP queries aligned to OpenAlex, testing generalizability beyond encyclopaedic KGs. Three open LLMs: Llama-3-8B, DeepSeek-R1-Distill-Llama-70B, and Mistral-Large-Instruct-2407 are selected based on their sizes and architectures and tested with zero-shot, few-shot, and two chain-of-thought variants. Outputs were compared with gold answers, and resulting errors were categorized. We find that the performance varies markedly across models and prompting strategies, and that translations for Wikidata to DBpedia work far better than translations for DBpedia to Wikidata. 

---
# DeepSeek: Paradigm Shifts and Technical Evolution in Large AI Models 

**Authors**: Luolin Xiong, Haofen Wang, Xi Chen, Lu Sheng, Yun Xiong, Jingping Liu, Yanghua Xiao, Huajun Chen, Qing-Long Han, Yang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09955)  

**Abstract**: DeepSeek, a Chinese Artificial Intelligence (AI) startup, has released their V3 and R1 series models, which attracted global attention due to their low cost, high performance, and open-source advantages. This paper begins by reviewing the evolution of large AI models focusing on paradigm shifts, the mainstream Large Language Model (LLM) paradigm, and the DeepSeek paradigm. Subsequently, the paper highlights novel algorithms introduced by DeepSeek, including Multi-head Latent Attention (MLA), Mixture-of-Experts (MoE), Multi-Token Prediction (MTP), and Group Relative Policy Optimization (GRPO). The paper then explores DeepSeek engineering breakthroughs in LLM scaling, training, inference, and system-level optimization architecture. Moreover, the impact of DeepSeek models on the competitive AI landscape is analyzed, comparing them to mainstream LLMs across various fields. Finally, the paper reflects on the insights gained from DeepSeek innovations and discusses future trends in the technical and engineering development of large AI models, particularly in data, training, and reasoning. 

---
# DeepResearch$^{\text{Eco}}$: A Recursive Agentic Workflow for Complex Scientific Question Answering in Ecology 

**Authors**: Jennifer D'Souza, Endres Keno Sander, Andrei Aioanei  

**Link**: [PDF](https://arxiv.org/pdf/2507.10522)  

**Abstract**: We introduce DeepResearch$^{\text{Eco}}$, a novel agentic LLM-based system for automated scientific synthesis that supports recursive, depth- and breadth-controlled exploration of original research questions -- enhancing search diversity and nuance in the retrieval of relevant scientific literature. Unlike conventional retrieval-augmented generation pipelines, DeepResearch enables user-controllable synthesis with transparent reasoning and parameter-driven configurability, facilitating high-throughput integration of domain-specific evidence while maintaining analytical rigor. Applied to 49 ecological research questions, DeepResearch achieves up to a 21-fold increase in source integration and a 14.9-fold rise in sources integrated per 1,000 words. High-parameter settings yield expert-level analytical depth and contextual diversity.
Source code available at: this https URL. 

---
# Introducing the Swiss Food Knowledge Graph: AI for Context-Aware Nutrition Recommendation 

**Authors**: Lubnaa Abdur Rahman, Ioannis Papathanail, Stavroula Mougiakakou  

**Link**: [PDF](https://arxiv.org/pdf/2507.10156)  

**Abstract**: AI has driven significant progress in the nutrition field, especially through multimedia-based automatic dietary assessment. However, existing automatic dietary assessment systems often overlook critical non-visual factors, such as recipe-specific ingredient substitutions that can significantly alter nutritional content, and rarely account for individual dietary needs, including allergies, restrictions, cultural practices, and personal preferences. In Switzerland, while food-related information is available, it remains fragmented, and no centralized repository currently integrates all relevant nutrition-related aspects within a Swiss context. To bridge this divide, we introduce the Swiss Food Knowledge Graph (SwissFKG), the first resource, to our best knowledge, to unite recipes, ingredients, and their substitutions with nutrient data, dietary restrictions, allergen information, and national nutrition guidelines under one graph. We establish a LLM-powered enrichment pipeline for populating the graph, whereby we further present the first benchmark of four off-the-shelf (<70 B parameter) LLMs for food knowledge augmentation. Our results demonstrate that LLMs can effectively enrich the graph with relevant nutritional information. Our SwissFKG goes beyond recipe recommendations by offering ingredient-level information such as allergen and dietary restriction information, and guidance aligned with nutritional guidelines. Moreover, we implement a Graph-RAG application to showcase how the SwissFKG's rich natural-language data structure can help LLM answer user-specific nutrition queries, and we evaluate LLM-embedding pairings by comparing user-query responses against predefined expected answers. As such, our work lays the foundation for the next generation of dietary assessment tools that blend visual, contextual, and cultural dimensions of eating. 

---
# Towards Concise and Adaptive Thinking in Large Reasoning Models: A Survey 

**Authors**: Jason Zhu, Hongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09662)  

**Abstract**: Large reasoning models (LRMs) like OpenAI o1 and DeepSeek R1 have demonstrated impressive performance on complex reasoning tasks like mathematics and programming with long Chain-of-Thought (CoT) reasoning sequences (slow-thinking), compared with traditional large language models (fast-thinking). However, these reasoning models also face a huge challenge that generating unnecessarily lengthy and redundant reasoning chains even for trivial questions. This phenomenon leads to a significant waste of inference resources, increases the response time for simple queries, and hinders the practical application of LRMs in real-world products. To this end, it is crucial to shorten lengthy reasoning chains and learn adaptive reasoning between fast and slow thinking based on input difficulty. In this survey, we provide a comprehensive overview of recent progress in concise and adaptive thinking for efficient reasoning of LRMs, including methodologies, benchmarks, and challenges for future exploration. We hope this survey can help researchers quickly understand the landscape of this field and inspire novel adaptive thinking ideas to facilitate better usage of LRMs. 

---
# eSapiens: A Platform for Secure and Auditable Retrieval-Augmented Generation 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09588)  

**Abstract**: We present eSapiens, an AI-as-a-Service (AIaaS) platform engineered around a business-oriented trifecta: proprietary data, operational workflows, and any major agnostic Large Language Model (LLM). eSapiens gives businesses full control over their AI assets, keeping everything in-house for AI knowledge retention and data security. eSapiens AI Agents (Sapiens) empower your team by providing valuable insights and automating repetitive tasks, enabling them to focus on high-impact work and drive better business outcomes.
The system integrates structured document ingestion, hybrid vector retrieval, and no-code orchestration via LangChain, and supports top LLMs including OpenAI, Claude, Gemini, and DeepSeek. A key component is the THOR Agent, which handles structured SQL-style queries and generates actionable insights over enterprise databases.
To evaluate the system, we conduct two experiments. First, a retrieval benchmark on legal corpora reveals that a chunk size of 512 tokens yields the highest retrieval precision (Top-3 accuracy: 91.3%). Second, a generation quality test using TRACe metrics across five LLMs shows that eSapiens delivers more context-consistent outputs with up to a 23% improvement in factual alignment.
These results demonstrate the effectiveness of eSapiens in enabling trustworthy, auditable AI workflows for high-stakes domains like legal and finance. 

---
# Is Human-Written Data Enough? The Challenge of Teaching Reasoning to LLMs Without RL or Distillation 

**Authors**: Wei Du, Branislav Kisacanin, George Armstrong, Shubham Toshniwal, Ivan Moshkov, Alexan Ayrapetyan, Sadegh Mahdavi, Dan Zhao, Shizhe Diao, Dragan Masulovic, Marius Stanean, Advaith Avadhanam, Max Wang, Ashmit Dutta, Shitij Govil, Sri Yanamandara, Mihir Tandon, Sriram Ananthakrishnan, Vedant Rathi, David Zhang, Joonseok Kang, Leon Luo, Titu Andreescu, Boris Ginsburg, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2507.09850)  

**Abstract**: Reasoning-capable language models achieve state-of-the-art performance in diverse complex tasks by generating long, explicit Chain-of-Thought (CoT) traces. While recent works show that base models can acquire such reasoning traces via reinforcement learning or distillation from stronger models like DeepSeek-R1, previous works demonstrate that even short CoT prompting without fine-tuning is able to improve reasoning. We ask whether long CoT can be induced in a base model using only prompting or minimal tuning. Using just 20 long CoT examples from the reasoning model \texttt{QwQ-32B-Preview}, we lightly fine-tune the base model \texttt{Qwen2.5-32B}. The resulting model outperforms the much larger \texttt{Qwen2.5-Math-72B-Instruct}, showing that a handful of high-quality examples can unlock strong reasoning capabilities. We further explore using CoT data from non-reasoning models and human annotators, enhanced with prompt engineering, multi-pass editing, and structural guidance. However, neither matches the performance of reasoning model traces, suggesting that certain latent qualities of expert CoT are difficult to replicate. We analyze key properties of reasoning data, such as problem difficulty, diversity, and answer length, that influence reasoning distillation. While challenges remain, we are optimistic that carefully curated human-written CoT, even in small quantities, can activate reasoning behaviors in base models. We release our human-authored dataset across refinement stages and invite further investigation into what makes small-scale reasoning supervision so effective. 

---
# Sound and Complete Neuro-symbolic Reasoning with LLM-Grounded Interpretations 

**Authors**: Bradley P. Allen, Prateek Chhikara, Thomas Macaulay Ferguson, Filip Ilievski, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2507.09751)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neuro-symbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties. 

---
# LLM-Stackelberg Games: Conjectural Reasoning Equilibria and Their Applications to Spearphishing 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09407)  

**Abstract**: We introduce the framework of LLM-Stackelberg games, a class of sequential decision-making models that integrate large language models (LLMs) into strategic interactions between a leader and a follower. Departing from classical Stackelberg assumptions of complete information and rational agents, our formulation allows each agent to reason through structured prompts, generate probabilistic behaviors via LLMs, and adapt their strategies through internal cognition and belief updates. We define two equilibrium concepts: reasoning and behavioral equilibrium, which aligns an agent's internal prompt-based reasoning with observable behavior, and conjectural reasoning equilibrium, which accounts for epistemic uncertainty through parameterized models over an opponent's response. These layered constructs capture bounded rationality, asymmetric information, and meta-cognitive adaptation. We illustrate the framework through a spearphishing case study, where a sender and a recipient engage in a deception game using structured reasoning prompts. This example highlights the cognitive richness and adversarial potential of LLM-mediated interactions. Our results show that LLM-Stackelberg games provide a powerful paradigm for modeling decision-making in domains such as cybersecurity, misinformation, and recommendation systems. 

---
# Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning 

**Authors**: Zijun Chen, Wenbo Hu, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.10007)  

**Abstract**: Chain of Thought (CoT) reasoning has demonstrated remarkable deep reasoning capabilities in both large language models (LLMs) and multimodal large language models (MLLMs). However, its reliability is often undermined by the accumulation of errors in intermediate steps. This paper introduces an novel approach to calibrate the CoT reasoning accuracy by leveraging the model's intrinsic veracity encoding. We discover that specific attention head activations reliably reflect the truthfulness of reasoning steps in CoT. Based on this insight, we train a confidence predictor to evaluate the correctness of each reasoning step using these truthfulness-sensitive activations, dynamically selecting the most plausible reasoning path via beam search. Experimental results demonstrate that our method significantly outperforms the state-of-the-art baselines (e.g., Few-Shot CoT, Self-Consistency, and Self-Evaluation Guided Beam Search) across the mathematical, symbolic, and commonsense reasoning tasks, exhibiting superior accuracy and reliability in both unimodal and multimodal settings. We further validate the approach on large reasoning models, confirming its applicability to specialized reasoning models. Additionally, we explore the role of the model's self-correction ability in CoT reasoning. This work provides a novel reliability improvement path for CoT reasoning with broad application potential. 

---
# EduFlow: Advancing MLLMs' Problem-Solving Proficiency through Multi-Stage, Multi-Perspective Critique 

**Authors**: Chenglin Zhu, Tao Zhang, Chong Li, Mingan Lin, Zenan Zhou, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.09374)  

**Abstract**: Multimodal large language models (MLLMs) still perform poorly on scientific tasks, particularly those requiring multi-step and interpretable reasoning. Their limitations include insufficient scientific reasoning patterns, lack of global coherence in multi-step inference, and the absence of reflective self-correction, making them unreliable in structured scientific contexts. We introduce EduFlow, the first end-to-end framework that covers the full pipeline of educational scientific reasoning, including data selection, MCTS-based trajectory construction, model training, and output optimization. At its core is EduPRM, a process-aware reward model that critiques reasoning steps with tags and justifications. EduPRM is trained via curriculum learning on three complementary supervision sources: MCTS-guided trajectories, error-injected critiques, and teacher-student dialogues, enabling dynamic adaptation to multi-stage problem solving and iterative refinement during inference. We further propose EduMCTS, a domain-adapted search framework that introduces bootstrapping actions specifically designed for educational reasoning, such as a self-reflection mechanism that promotes reflective error correction. It further leverages EduPRM's fine-grained feedback to guide the search toward higher-quality reasoning trajectories. By applying self-consistency and rejection sampling, we constructed EduMCTS-160K, a large-scale dataset of educational reasoning trajectories. Extensive experiments demonstrate that EduFlow enhances reasoning consistency and coherence. Code, data, and models will be released. 

---
# Model-Grounded Symbolic Artificial Intelligence Systems Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems 

**Authors**: Aniruddha Chattopadhyay, Raj Dandekar, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2507.09854)  

**Abstract**: Neurosymbolic artificial intelligence (AI) systems combine neural network and classical symbolic AI mechanisms to exploit the complementary strengths of large scale, generalizable learning and robust, verifiable reasoning. Numerous classifications of neurosymbolic AI illustrate how these two components can be integrated in distinctly different ways. In this work, we propose reinterpreting instruction tuned large language models as model grounded symbolic AI systems where natural language serves as the symbolic layer and grounding is achieved through the models internal representation space. Within this framework, we investigate and develop novel learning and reasoning approaches that preserve structural similarities to traditional learning and reasoning paradigms. Preliminary evaluations across axiomatic deductive reasoning procedures of varying complexity provide insights into the effectiveness of our approach in improving learning efficiency and reasoning reliability. 

---
# Could you be wrong: Debiasing LLMs using a metacognitive prompt for improving human decision making 

**Authors**: Thomas T. Hills  

**Link**: [PDF](https://arxiv.org/pdf/2507.10124)  

**Abstract**: Identifying bias in LLMs is ongoing. Because they are still in development, what is true today may be false tomorrow. We therefore need general strategies for debiasing that will outlive current models. Strategies developed for debiasing human decision making offer one promising approach as they incorporate an LLM-style prompt intervention designed to bring latent knowledge into awareness during decision making. LLMs trained on vast amounts of information contain information about potential biases, counter-arguments, and contradictory evidence, but that information may only be brought to bear if prompted. Metacognitive prompts developed in the human decision making literature are designed to achieve this, and as I demonstrate here, they show promise with LLMs. The prompt I focus on here is "could you be wrong?" Following an LLM response, this prompt leads LLMs to produce additional information, including why they answered as they did, errors, biases, contradictory evidence, and alternatives, none of which were apparent in their initial response. Indeed, this metaknowledge often reveals that how LLMs and users interpret prompts are not aligned. Here I demonstrate this prompt using a set of questions taken from recent articles about LLM biases, including implicit discriminatory biases and failures of metacognition. "Could you be wrong" prompts the LLM to identify its own biases and produce cogent metacognitive reflection. I also present another example involving convincing but incomplete information, which is readily corrected by the metacognitive prompt. In sum, this work argues that human psychology offers a new avenue for prompt engineering, leveraging a long history of effective prompt-based improvements to human decision making. 

---
# Bridging Bots: from Perception to Action via Multimodal-LMs and Knowledge Graphs 

**Authors**: Margherita Martorana, Francesca Urgese, Mark Adamik, Ilaria Tiddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09617)  

**Abstract**: Personal service robots are deployed to support daily living in domestic environments, particularly for elderly and individuals requiring assistance. These robots must perceive complex and dynamic surroundings, understand tasks, and execute context-appropriate actions. However, current systems rely on proprietary, hard-coded solutions tied to specific hardware and software, resulting in siloed implementations that are difficult to adapt and scale across platforms. Ontologies and Knowledge Graphs (KGs) offer a solution to enable interoperability across systems, through structured and standardized representations of knowledge and reasoning. However, symbolic systems such as KGs and ontologies struggle with raw and noisy sensory input. In contrast, multimodal language models are well suited for interpreting input such as images and natural language, but often lack transparency, consistency, and knowledge grounding. In this work, we propose a neurosymbolic framework that combines the perceptual strengths of multimodal language models with the structured representations provided by KGs and ontologies, with the aim of supporting interoperability in robotic applications. Our approach generates ontology-compliant KGs that can inform robot behavior in a platform-independent manner. We evaluated this framework by integrating robot perception data, ontologies, and five multimodal models (three LLaMA and two GPT models), using different modes of neural-symbolic interaction. We assess the consistency and effectiveness of the generated KGs across multiple runs and configurations, and perform statistical analyzes to evaluate performance. Results show that GPT-o1 and LLaMA 4 Maverick consistently outperform other models. However, our findings also indicate that newer models do not guarantee better results, highlighting the critical role of the integration strategy in generating ontology-compliant KGs. 

---
# CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks 

**Authors**: Hongchao Jiang, Yiming Chen, Yushi Cao, Hung-yi Lee, Robby T. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10535)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the state-of-the-art in various coding tasks. Beyond directly answering user queries, LLMs can also serve as judges, assessing and comparing the quality of responses generated by other models. Such an evaluation capability is crucial both for benchmarking different LLMs and for improving response quality through response ranking. However, despite the growing adoption of the LLM-as-a-Judge paradigm, its effectiveness in coding scenarios remains underexplored due to the absence of dedicated benchmarks. To address this gap, we introduce CodeJudgeBench, a benchmark explicitly designed to evaluate the performance of LLM-as-a-Judge models across three critical coding tasks: code generation, code repair, and unit test generation. Through comprehensive benchmarking of 26 LLM-as-a-Judge models, we find that recent thinking models significantly outperform non-thinking models on our carefully designed code judging tasks. Notably, even relatively small thinking models, such as Qwen3-8B, can outperform specially trained LLM-as-a-Judge models up to 70B in size. Nevertheless, all models still exhibit significant randomness in their judgment of coding tasks. For pairwise judging tasks, simply changing the order in which responses are presented can substantially impact accuracy. In addition, when judging code and unit tests written by different LLMs, LLM-as-a-Judge models also show variance in performance. This sensitivity raises concerns about the reliability and consistency of LLM-as-a-Judge in coding scenarios. Lastly, we study optimal prompting strategies for LLM-as-a-Judge. We find that using pair-wise comparison outperforms scalar point-wise judging. Furthermore, retaining comments and reasoning in the full, unprocessed LLM response leads to improved judge performance. 

---
# Chat with AI: The Surprising Turn of Real-time Video Communication from Human to AI 

**Authors**: Jiangkai Wu, Zhiyuan Ren, Liming Liu, Xinggong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10510)  

**Abstract**: AI Video Chat emerges as a new paradigm for Real-time Communication (RTC), where one peer is not a human, but a Multimodal Large Language Model (MLLM). This makes interaction between humans and AI more intuitive, as if chatting face-to-face with a real person. However, this poses significant challenges to latency, because the MLLM inference takes up most of the response time, leaving very little time for video streaming. Due to network uncertainty and instability, transmission latency becomes a critical bottleneck preventing AI from being like a real person. To address this, we propose Artic, an AI-oriented Real-time Communication framework, exploring the network requirement shift from "humans watching video" to "AI understanding video". To reduce bitrate dramatically while maintaining MLLM accuracy, we propose Context-Aware Video Streaming that recognizes the importance of each video region for chat and allocates bitrate almost exclusively to chat-important regions. To avoid packet retransmission, we propose Loss-Resilient Adaptive Frame Rate that leverages previous frames to substitute for lost/delayed frames while avoiding bitrate waste. To evaluate the impact of video streaming quality on MLLM accuracy, we build the first benchmark, named Degraded Video Understanding Benchmark (DeViBench). Finally, we discuss some open questions and ongoing solutions for AI Video Chat. 

---
# Scene-Aware Conversational ADAS with Generative AI for Real-Time Driver Assistance 

**Authors**: Kyungtae Han, Yitao Chen, Rohit Gupta, Onur Altintas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10500)  

**Abstract**: While autonomous driving technologies continue to advance, current Advanced Driver Assistance Systems (ADAS) remain limited in their ability to interpret scene context or engage with drivers through natural language. These systems typically rely on predefined logic and lack support for dialogue-based interaction, making them inflexible in dynamic environments or when adapting to driver intent. This paper presents Scene-Aware Conversational ADAS (SC-ADAS), a modular framework that integrates Generative AI components including large language models, vision-to-text interpretation, and structured function calling to enable real-time, interpretable, and adaptive driver assistance. SC-ADAS supports multi-turn dialogue grounded in visual and sensor context, allowing natural language recommendations and driver-confirmed ADAS control. Implemented in the CARLA simulator with cloud-based Generative AI, the system executes confirmed user intents as structured ADAS commands without requiring model fine-tuning. We evaluate SC-ADAS across scene-aware, conversational, and revisited multi-turn interactions, highlighting trade-offs such as increased latency from vision-based context retrieval and token growth from accumulated dialogue history. These results demonstrate the feasibility of combining conversational reasoning, scene perception, and modular ADAS control to support the next generation of intelligent driver assistance. 

---
# Can You Detect the Difference? 

**Authors**: İsmail Tarım, Aytuğ Onan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10475)  

**Abstract**: The rapid advancement of large language models (LLMs) has raised concerns about reliably detecting AI-generated text. Stylometric metrics work well on autoregressive (AR) outputs, but their effectiveness on diffusion-based models is unknown. We present the first systematic comparison of diffusion-generated text (LLaDA) and AR-generated text (LLaMA) using 2 000 samples. Perplexity, burstiness, lexical diversity, readability, and BLEU/ROUGE scores show that LLaDA closely mimics human text in perplexity and burstiness, yielding high false-negative rates for AR-oriented detectors. LLaMA shows much lower perplexity but reduced lexical fidelity. Relying on any single metric fails to separate diffusion outputs from human writing. We highlight the need for diffusion-aware detectors and outline directions such as hybrid models, diffusion-specific stylometric signatures, and robust watermarking. 

---
# Referential ambiguity and clarification requests: comparing human and LLM behaviour 

**Authors**: Chris Madge, Matthew Purver, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2507.10445)  

**Abstract**: In this work we examine LLMs' ability to ask clarification questions in task-oriented dialogues that follow the asynchronous instruction-giver/instruction-follower format. We present a new corpus that combines two existing annotations of the Minecraft Dialogue Corpus -- one for reference and ambiguity in reference, and one for SDRT including clarifications -- into a single common format providing the necessary information to experiment with clarifications and their relation to ambiguity. With this corpus we compare LLM actions with original human-generated clarification questions, examining how both humans and LLMs act in the case of ambiguity. We find that there is only a weak link between ambiguity and humans producing clarification questions in these dialogues, and low correlation between humans and LLMs. Humans hardly ever produce clarification questions for referential ambiguity, but often do so for task-based uncertainty. Conversely, LLMs produce more clarification questions for referential ambiguity, but less so for task uncertainty. We question if LLMs' ability to ask clarification questions is predicated on their recent ability to simulate reasoning, and test this with different reasoning approaches, finding that reasoning does appear to increase question frequency and relevancy. 

---
# FaceLLM: A Multimodal Large Language Model for Face Understanding 

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2507.10300)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable performance in vision-language tasks. However, existing MLLMs are primarily trained on generic datasets, limiting their ability to reason on domain-specific visual cues such as those in facial images. In particular, tasks that require detailed understanding of facial structure, expression, emotion, and demographic features remain underexplored by MLLMs due to the lack of large-scale annotated face image-text datasets. In this work, we introduce FaceLLM, a multimodal large language model trained specifically for facial image understanding. To construct the training data, we propose a novel weakly supervised pipeline that uses ChatGPT with attribute-aware prompts to generate high-quality question-answer pairs based on images from the FairFace dataset. The resulting corpus, called FairFaceGPT, covers a diverse set of attributes including expression, pose, skin texture, and forensic information. Our experiments demonstrate that FaceLLM improves the performance of MLLMs on various face-centric tasks and achieves state-of-the-art performance. This work highlights the potential of synthetic supervision via language models for building domain-specialized MLLMs, and sets a precedent for trustworthy, human-centric multimodal AI systems. FairFaceGPT dataset and pretrained FaceLLM models are publicly available in the project page. 

---
# From Sequence to Structure: Uncovering Substructure Reasoning in Transformers 

**Authors**: Xinnan Dai, Kai Yang, Jay Revolinsky, Kai Guo, Aoran Wang, Bohang Zhang, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10435)  

**Abstract**: Recent studies suggest that large language models (LLMs) possess the capability to solve graph reasoning tasks. Notably, even when graph structures are embedded within textual descriptions, LLMs can still effectively answer related questions. This raises a fundamental question: How can a decoder-only Transformer architecture understand underlying graph structures? To address this, we start with the substructure extraction task, interpreting the inner mechanisms inside the transformers and analyzing the impact of the input queries. Specifically, through both empirical results and theoretical analysis, we present Induced Substructure Filtration (ISF), a perspective that captures the substructure identification in the multi-layer transformers. We further validate the ISF process in LLMs, revealing consistent internal dynamics across layers. Building on these insights, we explore the broader capabilities of Transformers in handling diverse graph types. Specifically, we introduce the concept of thinking in substructures to efficiently extract complex composite patterns, and demonstrate that decoder-only Transformers can successfully extract substructures from attributed graphs, such as molecular graphs. Together, our findings offer a new insight on how sequence-based Transformers perform the substructure extraction task over graph data. 

---
# Natural Language-based Assessment of L2 Oral Proficiency using LLMs 

**Authors**: Stefano Bannò, Rao Ma, Mengjie Qian, Siyuan Tang, Kate Knill, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2507.10200)  

**Abstract**: Natural language-based assessment (NLA) is an approach to second language assessment that uses instructions - expressed in the form of can-do descriptors - originally intended for human examiners, aiming to determine whether large language models (LLMs) can interpret and apply them in ways comparable to human assessment. In this work, we explore the use of such descriptors with an open-source LLM, Qwen 2.5 72B, to assess responses from the publicly available S&I Corpus in a zero-shot setting. Our results show that this approach - relying solely on textual information - achieves competitive performance: while it does not outperform state-of-the-art speech LLMs fine-tuned for the task, it surpasses a BERT-based model trained specifically for this purpose. NLA proves particularly effective in mismatched task settings, is generalisable to other data types and languages, and offers greater interpretability, as it is grounded in clearly explainable, widely applicable language descriptors. 

---
# Breaking the Myth: Can Small Models Infer Postconditions Too? 

**Authors**: Gehao Zhang, Zhenting Wang, Juan Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.10182)  

**Abstract**: Formal specifications are essential for ensuring software correctness, yet manually writing them is tedious and error-prone. Large Language Models (LLMs) have shown promise in generating such specifications from natural language intents, but the giant model size and high computational demands raise a fundamental question: Do we really need large models for this task? In this paper, we show that a small, fine-tuned language model can achieve high-quality postcondition generation with much lower computational costs. We construct a specialized dataset of prompts, reasoning logs, and postconditions, then supervise the fine-tuning of a $7$B-parameter code model. Our approach tackles real-world repository dependencies and preserves pre-state information, allowing for expressive and accurate specifications. We evaluate the model on a benchmark of real-world Java bugs (Defects4J) and compare against both proprietary giants (e.g., GPT-4o) and open-source large models. Empirical results demonstrate that our compact model matches or outperforms significantly larger counterparts in syntax correctness, semantic correctness, and bug-distinguishing capability. These findings highlight that targeted fine-tuning on a modest dataset can enable small models to achieve results formerly seen only in massive, resource-heavy LLMs, offering a practical and efficient path for the real-world adoption of automated specification generation. 

---
# Abusive text transformation using LLMs 

**Authors**: Rohitash Chandra, Jiyong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10177)  

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant advancements in natural language processing tasks, their effectiveness in the classification and transformation of abusive text into non-abusive versions remains an area for exploration. In this study, we aim to use LLMs to transform abusive text (tweets and reviews) featuring hate speech and swear words into non-abusive text, while retaining the intent of the text. We evaluate the performance of two state-of-the-art LLMs, such as Gemini, GPT-4o, DeekSeek and Groq, on their ability to identify abusive text. We them to transform and obtain a text that is clean from abusive and inappropriate content but maintains a similar level of sentiment and semantics, i.e. the transformed text needs to maintain its message. Afterwards, we evaluate the raw and transformed datasets with sentiment analysis and semantic analysis. Our results show Groq provides vastly different results when compared with other LLMs. We have identified similarities between GPT-4o and DeepSeek-V3. 

---
# Absher: A Benchmark for Evaluating Large Language Models Understanding of Saudi Dialects 

**Authors**: Renad Al-Monef, Hassan Alhuzali, Nora Alturayeif, Ashwag Alasmari  

**Link**: [PDF](https://arxiv.org/pdf/2507.10216)  

**Abstract**: As large language models (LLMs) become increasingly central to Arabic NLP applications, evaluating their understanding of regional dialects and cultural nuances is essential, particularly in linguistically diverse settings like Saudi Arabia. This paper introduces \texttt{Absher}, a comprehensive benchmark specifically designed to assess LLMs performance across major Saudi dialects. \texttt{Absher} comprises over 18,000 multiple-choice questions spanning six distinct categories: Meaning, True/False, Fill-in-the-Blank, Contextual Usage, Cultural Interpretation, and Location Recognition. These questions are derived from a curated dataset of dialectal words, phrases, and proverbs sourced from various regions of Saudi Arabia. We evaluate several state-of-the-art LLMs, including multilingual and Arabic-specific models. We also provide detailed insights into their capabilities and limitations. Our results reveal notable performance gaps, particularly in tasks requiring cultural inference or contextual understanding. Our findings highlight the urgent need for dialect-aware training and culturally aligned evaluation methodologies to improve LLMs performance in real-world Arabic applications. 

---
# Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning 

**Authors**: Chenxi Huang, Shaotian Yan, Liang Xie, Binbin Lin, Sinan Fan, Yue Xin, Deng Cai, Chen Shen, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.10085)  

**Abstract**: Representation Fine-tuning (ReFT), a recently proposed Parameter-Efficient Fine-Tuning (PEFT) method, has attracted widespread attention for significantly improving parameter efficiency by editing representation space alone. In this work, we investigate applying ReFT to complex reasoning tasks. However, directly using the native ReFT method, which modifies fixed representations at the beginning and end of each layer, yields suboptimal performance, as these fixed-position representations have uncertain impact on the outputs. We observe that, in complex reasoning tasks, there often exist certain critical representations. These representations either integrate significant information from preceding layers or regulate subsequent layer representations. Through layer-by-layer propagation, they exert a substantial influence on the final output. Naturally, fine-tuning these critical representations has the potential to greatly enhance reasoning performance. Building upon these insights, we propose Critical Representation Fine-Tuning (CRFT), a novel method that identifies and optimizes these critical representations through information flow analysis. CRFT operates within a supervised learning framework, dynamically optimizing critical representations in a low-rank linear subspace while freezing the base model. The effectiveness and efficiency of our method are validated across eight benchmarks for arithmetic and commonsense reasoning, using LLaMA and Mistral model families. Furthermore, our method also adapts effectively to few-shot settings, boosting one-shot accuracy by 16.4%. Our work highlights the untapped potential of representation-level optimization for CoT reasoning, offering a lightweight yet powerful alternative to traditional PEFT methods. 

---
# Think Clearly: Improving Reasoning via Redundant Token Pruning 

**Authors**: Daewon Choi, Jimin Lee, Jihoon Tack, Woomin Song, Saket Dingliwal, Sai Muralidhar Jayanthi, Bhavana Ganesh, Jinwoo Shin, Aram Galstyan, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2507.08806)  

**Abstract**: Recent large language models have shown promising capabilities in long-form reasoning, following structured chains of thought before arriving at a final answer. However, we observe that these reasoning paths tend to include substantial redundancy; analyzing attention patterns reveals that attention scores are widely scattered, particularly incorrect answers exhibit greater attention sparsity. In this paper, we demonstrate that deliberately removing this redundancy in the reasoning process significantly improves performance through clear thinking, i.e., removing distraction. Specifically, we systematically identify reasoning redundancy by measuring token-level attention scores to a special end-of-thinking token, which is appended to an explicit instruction inserted to conclude each intermediate reasoning step. Furthermore, we propose structure-aware pruning that prioritizes removing tokens in low-contributing reasoning chunks over individual tokens. After evicting redundant tokens, we remove the injected end-of-thinking instruction, then resume the reasoning generation. We demonstrate that our method significantly improves overall accuracy across reasoning-intensive benchmarks without any training involved. In particular, our method shows strong performance on challenging mathematical competition benchmarks such as AIME and AMC, where reasoning redundancy is more prevalent. 

---
# Cultural Bias in Large Language Models: Evaluating AI Agents through Moral Questionnaires 

**Authors**: Simon Münker  

**Link**: [PDF](https://arxiv.org/pdf/2507.10073)  

**Abstract**: Are AI systems truly representing human values, or merely averaging across them? Our study suggests a concerning reality: Large Language Models (LLMs) fail to represent diverse cultural moral frameworks despite their linguistic capabilities. We expose significant gaps between AI-generated and human moral intuitions by applying the Moral Foundations Questionnaire across 19 cultural contexts. Comparing multiple state-of-the-art LLMs' origins against human baseline data, we find these models systematically homogenize moral diversity. Surprisingly, increased model size doesn't consistently improve cultural representation fidelity. Our findings challenge the growing use of LLMs as synthetic populations in social science research and highlight a fundamental limitation in current AI alignment approaches. Without data-driven alignment beyond prompting, these systems cannot capture the nuanced, culturally-specific moral intuitions. Our results call for more grounded alignment objectives and evaluation metrics to ensure AI systems represent diverse human values rather than flattening the moral landscape. 

---
# Tiny Reward Models 

**Authors**: Sarah Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09973)  

**Abstract**: Large decoder-based language models have become the dominant architecture for reward modeling in reinforcement learning from human feedback (RLHF). However, as reward models are increasingly deployed in test-time strategies, their inference costs become a growing concern. We present TinyRM, a family of small, bidirectional masked language models (MLMs) with as few as 400 million parameters, that rival the capabilities of models over 175 times larger on reasoning and safety preference modeling tasks. TinyRM combines FLAN-style prompting, Directional Low-Rank Adaptation (DoRA), and layer freezing to achieve strong performance on RewardBench, despite using significantly fewer resources. Our experiments suggest that small models benefit from domain-specific tuning strategies, particularly in reasoning, where lightweight finetuning methods are especially effective. While challenges remain in building generalist models and conversational preference modeling, our preliminary results highlight the promise of lightweight bidirectional architectures as efficient, scalable alternatives for preference modeling. 

---
# Mechanistic Interpretability of LoRA-Adapted Language Models for Nuclear Reactor Safety Applications 

**Authors**: Yoon Pyo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.09931)  

**Abstract**: The integration of Large Language Models (LLMs) into safety-critical domains, such as nuclear engineering, necessitates a deep understanding of their internal reasoning processes. This paper presents a novel methodology for interpreting how an LLM encodes and utilizes domain-specific knowledge, using a Boiling Water Reactor system as a case study. We adapted a general-purpose LLM (Gemma-3-1b-it) to the nuclear domain using a parameter-efficient fine-tuning technique known as Low-Rank Adaptation. By comparing the neuron activation patterns of the base model to those of the fine-tuned model, we identified a sparse set of neurons whose behavior was significantly altered during the adaptation process. To probe the causal role of these specialized neurons, we employed a neuron silencing technique. Our results demonstrate that while silencing most of these specialized neurons individually did not produce a statistically significant effect, deactivating the entire group collectively led to a statistically significant degradation in task performance. Qualitative analysis further revealed that silencing these neurons impaired the model's ability to generate detailed, contextually accurate technical information. This paper provides a concrete methodology for enhancing the transparency of an opaque black-box model, allowing domain expertise to be traced to verifiable neural circuits. This offers a pathway towards achieving nuclear-grade artificial intelligence (AI) assurance, addressing the verification and validation challenges mandated by nuclear regulatory frameworks (e.g., 10 CFR 50 Appendix B), which have limited AI deployment in safety-critical nuclear operations. 

---
# Can GPT-4o mini and Gemini 2.0 Flash Predict Fine-Grained Fashion Product Attributes? A Zero-Shot Analysis 

**Authors**: Shubham Shukla, Kunal Sonalkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09950)  

**Abstract**: The fashion retail business is centered around the capacity to comprehend products. Product attribution helps in comprehending products depending on the business process. Quality attribution improves the customer experience as they navigate through millions of products offered by a retail website. It leads to well-organized product catalogs. In the end, product attribution directly impacts the 'discovery experience' of the customer. Although large language models (LLMs) have shown remarkable capabilities in understanding multimodal data, their performance on fine-grained fashion attribute recognition remains under-explored. This paper presents a zero-shot evaluation of state-of-the-art LLMs that balance performance with speed and cost efficiency, mainly GPT-4o-mini and Gemini 2.0 Flash. We have used the dataset DeepFashion-MultiModal (this https URL) to evaluate these models in the attribution tasks of fashion products. Our study evaluates these models across 18 categories of fashion attributes, offering insight into where these models excel. We only use images as the sole input for product information to create a constrained environment. Our analysis shows that Gemini 2.0 Flash demonstrates the strongest overall performance with a macro F1 score of 56.79% across all attributes, while GPT-4o-mini scored a macro F1 score of 43.28%. Through detailed error analysis, our findings provide practical insights for deploying these LLMs in production e-commerce product attribution-related tasks and highlight the need for domain-specific fine-tuning approaches. This work also lays the groundwork for future research in fashion AI and multimodal attribute extraction. 

---
# ViTCoT: Video-Text Interleaved Chain-of-Thought for Boosting Video Understanding in Large Language Models 

**Authors**: Yongheng Zhang, Xu Liu, Ruihan Tao, Qiguang Chen, Hao Fei, Wanxiang Che, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09876)  

**Abstract**: Video understanding plays a vital role in bridging low-level visual signals with high-level cognitive reasoning, and is fundamental to applications such as autonomous driving, embodied AI, and the broader pursuit of AGI. The rapid development of large language models (LLMs), particularly those utilizing Chain-of-Thought (CoT) technology, has significantly advanced video reasoning capabilities. However, current approaches primarily depend on textual information for reasoning, overlooking the visual modality in the actual video reasoning process. In contrast, humans naturally re-examine visual content while reasoning. Motivated by this, we introduce a novel video reasoning paradigm: Video-Text Interleaved CoT (ViTCoT), which facilitates more intuitive and cognitively aligned reasoning. To the end, first, we construct the Video-Text Interleaved Benchmark (ViTIB), which is created using MLLMs for key-video selection and manually verified. Furthermore, we extensively explore the potential of the ViTCoT paradigm in the video understanding field. Extensive experiments demonstrate that ViTCoT significantly enhances performance compared to the traditional text-only CoT paradigm and effectively activates more neuron values in MLLMs. 

---
# TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit 

**Authors**: Paulo Salem, Robert Sim, Christopher Olsen, Prerit Saxena, Rafael Barcelos, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.09788)  

**Abstract**: Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, highlighting possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at this https URL. 

---
# CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design 

**Authors**: Prashant Govindarajan, Davide Baldelli, Jay Pathak, Quentin Fournier, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09792)  

**Abstract**: Computer-aided design (CAD) is the digital construction of 2D and 3D objects, and is central to a wide range of engineering and manufacturing applications like automobile and aviation. Despite its importance, CAD modeling remains largely a time-intensive, manual task. Recent works have attempted to automate this process with small transformer-based models and handcrafted CAD sequence representations. However, there has been little effort to leverage the potential of large language models (LLMs) for sequential CAD design. In this work, we introduce a new large-scale dataset of more than 170k CAD models annotated with high-quality, human-like descriptions generated with our pipeline based on GPT-4.1. Using this dataset, we fine-tune powerful code-LLMs to generate CAD sequences represented in a JSON-based format from natural language descriptions, demonstrating the viability and effectiveness of this approach for text-conditioned CAD generation. Because simple metrics often fail to reflect the quality of generated objects, we introduce geometric and topological metrics based on sphericity, mean curvature, and Euler characteristic to provide richer structural insights. Our experiments and ablation studies on both synthetic and human-annotated data demonstrate that CADmium is able to automate CAD design, drastically speeding up the design of new objects. The dataset, code, and fine-tuned models are available online. 

---
# A Serverless Architecture for Real-Time Stock Analysis using Large Language Models: An Iterative Development and Debugging Case Study 

**Authors**: Taniv Ashraf  

**Link**: [PDF](https://arxiv.org/pdf/2507.09583)  

**Abstract**: The advent of powerful, accessible Large Language Models (LLMs) like Google's Gemini presents new opportunities for democratizing financial data analysis. This paper documents the design, implementation, and iterative debugging of a novel, serverless system for real-time stock analysis. The system leverages the Gemini API for qualitative assessment, automates data ingestion and processing via GitHub Actions, and presents the findings through a decoupled, static frontend. We detail the architectural evolution of the system, from initial concepts to a robust, event-driven pipeline, highlighting the practical challenges encountered during deployment. A significant portion of this paper is dedicated to a case study on the debugging process, covering common software errors, platform-specific permission issues, and rare, environment-level platform bugs. The final architecture operates at a near-zero cost, demonstrating a viable model for individuals to build sophisticated AI-powered financial tools. The operational application is publicly accessible, and the complete source code is available for review. We conclude by discussing the role of LLMs in financial analysis, the importance of robust debugging methodologies, and the emerging paradigm of human-AI collaboration in software development. 

---
# A Mixture of Linear Corrections Generates Secure Code 

**Authors**: Weichen Yu, Ravi Mangal, Terry Zhuo, Matt Fredrikson, Corina S. Pasareanu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09508)  

**Abstract**: Large language models (LLMs) have become proficient at sophisticated code-generation tasks, yet remain ineffective at reliably detecting or avoiding code vulnerabilities. Does this deficiency stem from insufficient learning about code vulnerabilities, or is it merely a result of ineffective prompting? Using representation engineering techniques, we investigate whether LLMs internally encode the concepts necessary to identify code vulnerabilities. We find that current LLMs encode precise internal representations that distinguish vulnerable from secure code--achieving greater accuracy than standard prompting approaches. Leveraging these vulnerability-sensitive representations, we develop an inference-time steering technique that subtly modulates the model's token-generation probabilities through a mixture of corrections (MoC). Our method effectively guides LLMs to produce less vulnerable code without compromising functionality, demonstrating a practical approach to controlled vulnerability management in generated code. Notably, MoC enhances the security ratio of Qwen2.5-Coder-7B by 8.9\%, while simultaneously improving functionality on HumanEval pass@1 by 2.1\%. 

---
# Evaluating LLMs on Sequential API Call Through Automated Test Generation 

**Authors**: Yuheng Huang, Da Song, Zhenlan Ji, Shuai Wang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.09481)  

**Abstract**: By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs. 

---
# KEN: Knowledge Augmentation and Emotion Guidance Network for Multimodal Fake News Detection 

**Authors**: Peican Zhu, Yubo Jing, Le Cheng, Keke Tang, Yangming Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.09647)  

**Abstract**: In recent years, the rampant spread of misinformation on social media has made accurate detection of multimodal fake news a critical research focus. However, previous research has not adequately understood the semantics of images, and models struggle to discern news authenticity with limited textual information. Meanwhile, treating all emotional types of news uniformly without tailored approaches further leads to performance degradation. Therefore, we propose a novel Knowledge Augmentation and Emotion Guidance Network (KEN). On the one hand, we effectively leverage LVLM's powerful semantic understanding and extensive world knowledge. For images, the generated captions provide a comprehensive understanding of image content and scenes, while for text, the retrieved evidence helps break the information silos caused by the closed and limited text and context. On the other hand, we consider inter-class differences between different emotional types of news through balanced learning, achieving fine-grained modeling of the relationship between emotional types and authenticity. Extensive experiments on two real-world datasets demonstrate the superiority of our KEN. 

---
# Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs 

**Authors**: Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan, Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng, Dongyuan Li, Renhe Jiang, Ming Zhang, Yangqiu Song, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09477)  

**Abstract**: Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at this https URL. 

---
# Prompting for Performance: Exploring LLMs for Configuring Software 

**Authors**: Helge Spieker, Théo Matricon, Nassim Belmecheri, Jørn Eirik Betten, Gauthier Le Bartz Lyan, Heraldo Borges, Quentin Mazouni, Dennis Gross, Arnaud Gotlieb, Mathieu Acher  

**Link**: [PDF](https://arxiv.org/pdf/2507.09790)  

**Abstract**: Software systems usually provide numerous configuration options that can affect performance metrics such as execution time, memory usage, binary size, or bitrate. On the one hand, making informed decisions is challenging and requires domain expertise in options and their combinations. On the other hand, machine learning techniques can search vast configuration spaces, but with a high computational cost, since concrete executions of numerous configurations are required. In this exploratory study, we investigate whether large language models (LLMs) can assist in performance-oriented software configuration through prompts. We evaluate several LLMs on tasks including identifying relevant options, ranking configurations, and recommending performant configurations across various configurable systems, such as compilers, video encoders, and SAT solvers. Our preliminary results reveal both positive abilities and notable limitations: depending on the task and systems, LLMs can well align with expert knowledge, whereas hallucinations or superficial reasoning can emerge in other cases. These findings represent a first step toward systematic evaluations and the design of LLM-based solutions to assist with software configuration. 

---
# Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models 

**Authors**: Anita Kriz, Elizabeth Laura Janes, Xing Shen, Tal Arbel  

**Link**: [PDF](https://arxiv.org/pdf/2507.09279)  

**Abstract**: Multimodal large language models (MLLMs) hold considerable promise for applications in healthcare. However, their deployment in safety-critical settings is hindered by two key limitations: (i) sensitivity to prompt design, and (ii) a tendency to generate incorrect responses with high confidence. As clinicians may rely on a model's stated confidence to gauge the reliability of its predictions, it is especially important that when a model expresses high confidence, it is also highly accurate. We introduce Prompt4Trust, the first reinforcement learning (RL) framework for prompt augmentation targeting confidence calibration in MLLMs. A lightweight LLM is trained to produce context-aware auxiliary prompts that guide a downstream task MLLM to generate responses in which the expressed confidence more accurately reflects predictive accuracy. Unlike conventional calibration techniques, Prompt4Trust specifically prioritizes aspects of calibration most critical for safe and trustworthy clinical decision-making. Beyond improvements driven by this clinically motivated calibration objective, our proposed method also improves task accuracy, achieving state-of-the-art medical visual question answering (VQA) performance on the PMC-VQA benchmark, which is composed of multiple-choice questions spanning diverse medical imaging modalities. Moreover, our framework trained with a small downstream task MLLM showed promising zero-shot generalization to larger MLLMs in our experiments, suggesting the potential for scalable calibration without the associated computational costs. This work demonstrates the potential of automated yet human-aligned prompt engineering for improving the the trustworthiness of MLLMs in safety critical settings. Our codebase can be found at this https URL. 

---
# Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2507.09406)  

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models. 

---
# AInsight: Augmenting Expert Decision-Making with On-the-Fly Insights Grounded in Historical Data 

**Authors**: Mohammad Abolnejadian, Shakiba Amirshahi, Matthew Brehmer, Anamaria Crisan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09100)  

**Abstract**: In decision-making conversations, experts must navigate complex choices and make on-the-spot decisions while engaged in conversation. Although extensive historical data often exists, the real-time nature of these scenarios makes it infeasible for decision-makers to review and leverage relevant information. This raises an interesting question: What if experts could utilize relevant past data in real-time decision-making through insights derived from past data? To explore this, we implemented a conversational user interface, taking doctor-patient interactions as an example use case. Our system continuously listens to the conversation, identifies patient problems and doctor-suggested solutions, and retrieves related data from an embedded dataset, generating concise insights using a pipeline built around a retrieval-based Large Language Model (LLM) agent. We evaluated the prototype by embedding Health Canada datasets into a vector database and conducting simulated studies using sample doctor-patient dialogues, showing effectiveness but also challenges, setting directions for the next steps of our work. 

---
# CompassJudger-2: Towards Generalist Judge Model via Verifiable Rewards 

**Authors**: Taolin Zhang, Maosong Cao, Alexander Lam, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09104)  

**Abstract**: Recently, the role of LLM-as-judge in evaluating large language models has gained prominence. However, current judge models suffer from narrow specialization and limited robustness, undermining their capacity for comprehensive evaluations. In this work, we present CompassJudger-2, a novel generalist judge model that overcomes these limitations via a task-driven, multi-domain data curation strategy. Central to our approach is supervising judgment tasks with verifiable rewards, guiding intrinsic critical reasoning through rejection sampling to foster robust, generalizable judgment capabilities. We introduce a refined learning objective with margin policy gradient loss to enhance performance. Empirically, CompassJudger-2 achieves superior results across multiple judge and reward benchmarks, and our 7B model demonstrates competitive judgment accuracy with significantly larger models like DeepSeek-V3 and Qwen3-235B-A22B. Additionally, we propose JudgerBenchV2, a comprehensive benchmark evaluating cross-domain judgment accuracy and rank consistency to standardize judge model evaluation. These contributions advance robust, scalable LLM judgment and establish new performance and evaluation standards. 

---
# Learning from Synthetic Labs: Language Models as Auction Participants 

**Authors**: Anand Shah, Kehang Zhu, Yanchen Jiang, Jeffrey G. Wang, Arif K. Dayi, John J. Horton, David C. Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2507.09083)  

**Abstract**: This paper investigates the behavior of simulated AI agents (large language models, or LLMs) in auctions, introducing a novel synthetic data-generating process to help facilitate the study and design of auctions. We find that LLMs -- when endowed with chain of thought reasoning capacity -- agree with the experimental literature in auctions across a variety of classic auction formats. In particular, we find that LLM bidders produce results consistent with risk-averse human bidders; that they perform closer to theoretical predictions in obviously strategy-proof auctions; and, that they succumb to the winner's curse in common value settings. On prompting, we find that LLMs are not very sensitive to naive changes in prompts (e.g., language, currency) but can improve dramatically towards theoretical predictions with the right mental model (i.e., the language of Nash deviations). We run 1,000$+$ auctions for less than $\$$400 with GPT-4 models (three orders of magnitude cheaper than modern auction experiments) and develop a framework flexible enough to run auction experiments with any LLM model and a wide range of auction design specifications, facilitating further experimental study by decreasing costs and serving as a proof-of-concept for the use of LLM proxies. 

---
# ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making 

**Authors**: Bharadwaj Ravichandran, David Joy, Paul Elliott, Brian Hu, Jadie Adams, Christopher Funk, Emily Veenhuis, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2507.09037)  

**Abstract**: Large language models (LLMs) are increasingly being used as decision aids. However, users have diverse values and preferences that can affect their decision-making, which requires novel methods for LLM alignment and personalization. Existing LLM comparison tools largely focus on benchmarking tasks, such as knowledge-based question answering. In contrast, our proposed ALIGN system focuses on dynamic personalization of LLM-based decision-makers through prompt-based alignment to a set of fine-grained attributes. Key features of our system include robust configuration management, structured output generation with reasoning, and several algorithm implementations with swappable LLM backbones, enabling different types of analyses. Our user interface enables a qualitative, side-by-side comparison of LLMs and their alignment to various attributes, with a modular backend for easy algorithm integration. Additionally, we perform a quantitative analysis comparing alignment approaches in two different domains: demographic alignment for public opinion surveys and value alignment for medical triage decision-making. The entire ALIGN framework is open source and will enable new research on reliable, responsible, and personalized LLM-based decision-makers. 

---
# On Evaluating Performance of LLM Inference Serving Systems 

**Authors**: Amey Agrawal, Nitin Kedia, Anmol Agarwal, Jayashree Mohan, Nipun Kwatra, Souvik Kundu, Ramachandran Ramjee, Alexey Tumanov  

**Link**: [PDF](https://arxiv.org/pdf/2507.09019)  

**Abstract**: The rapid evolution of Large Language Model (LLM) inference systems has yielded significant efficiency improvements. However, our systematic analysis reveals that current evaluation methodologies frequently exhibit fundamental flaws, often manifesting as common evaluation anti-patterns that obscure true performance characteristics and impede scientific progress. Through a comprehensive examination of recent systems, we identify recurring anti-patterns across three key dimensions: Baseline Fairness, Evaluation Setup, and Metric Design. These anti-patterns are uniquely problematic for LLM inference due to its dual-phase nature combining distinct prefill and decode operations, its handling of highly heterogeneous workloads, and its strict temporal requirements for interactive use. We demonstrate how common anti-patterns -- such as inadequate baseline comparisons that conflate engineering effort with algorithmic novelty, workload selections that fail to represent production scenarios, and metric normalizations that hide substantial performance variability like generation stalls-lead to misleading conclusions. To address these challenges, we provide a comprehensive checklist derived from our analysis, establishing a framework for recognizing and avoiding these anti-patterns in favor of robust LLM inference evaluation. To demonstrate the practical application of our framework, we present a case study analyzing speculative decoding, a technique whose bursty, non-uniform token generation is easily misinterpreted when evaluated using approaches characteristic of these anti-patterns. Our work establishes a rigorous foundation for evaluation methodology, enabling meaningful comparisons, ensuring reproducible results, and ultimately accelerating genuine progress in LLM inference systems by moving beyond common anti-patterns to align evaluation with real-world requirements. 

---
# Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation 

**Authors**: Jialong Mai, Xiaofen Xing, Yawei Li, Zhipeng Li, Jingyuan Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09076)  

**Abstract**: Recent research has focused on applying speech large language model (SLLM) to improve speech emotion recognition (SER). However, the inherently high frame rate in speech modality severely limits the signal processing and understanding capabilities of SLLM. For example, a SLLM with a 4K context window can only process 80 seconds of audio at 50Hz feature sampling rate before reaching its capacity limit. Input token compression methods used in SLLM overlook the continuity and inertia of emotions across multiple conversation turns. This paper proposes a Dynamic Parameter Memory (DPM) mechanism with contextual semantics and sentence-level emotion encoding, enabling processing of unlimited-length audio with limited context windows in SLLM. Specifically, DPM progressively encodes sentence-level information and emotions into a temporary LoRA module during inference to effectively "memorize" the contextual information. We trained an emotion SLLM as a backbone and incorporated our DPM into inference for emotion recognition in conversation (ERC). Experimental results on the IEMOCAP dataset show that DPM significantly improves the emotion recognition capabilities of SLLM when processing long audio sequences, achieving state-of-the-art performance. 

---
# Hybrid Systolic Array Accelerator with Optimized Dataflow for Edge Large Language Model Inference 

**Authors**: Chun-Ting Chen, HanGyeol Mun, Jian Meng, Mohamed S. Abdelfattah, Jae-sun Seo  

**Link**: [PDF](https://arxiv.org/pdf/2507.09010)  

**Abstract**: Edge inference for large language models (LLM) offers secure, low-latency, and cost-effective inference solutions. We emphasize that an edge accelerator should achieve high area efficiency and minimize external memory access (EMA) during the memory-bound decode stage, while maintaining high energy efficiency during the compute intensive prefill stage. This paper proposes an edge LLM inference accelerator featuring a hybrid systolic array (HSA) architecture that optimizes inference efficiency in both stages. To further reduce EMA, we adopt MXINT4 weight quantization and propose an optimized dataflow tailored for HSA, ensuring negligible dequantization overhead and achieving 100% hardware utilization with minimal accuracy loss under edge DRAM bandwidth constraints. For non-linear operations, we incorporate optimized root mean square normalization (RMSNorm) and rotary position embedding (RoPE) units, reducing their latency, area, and memory access overhead while enabling end-to-end inference on our accelerator. Our solution achieves 247/117 (token/s/mm2) while running a 1.3B LLM on long-input/long-output scenarios, providing >2.45x/13.5x improvement over existing approaches, while maintaining superior energy efficiency in token generation. 

---
# Bridging Literature and the Universe Via A Multi-Agent Large Language Model System 

**Authors**: Xiaowen Zhang, Zhenyu Bi, Xuan Wang, Tiziana Di Matteo, Rupert A.C. Croft  

**Link**: [PDF](https://arxiv.org/pdf/2507.08958)  

**Abstract**: As cosmological simulations and their associated software become increasingly complex, physicists face the challenge of searching through vast amounts of literature and user manuals to extract simulation parameters from dense academic papers, each using different models and formats. Translating these parameters into executable scripts remains a time-consuming and error-prone process. To improve efficiency in physics research and accelerate the cosmological simulation process, we introduce SimAgents, a multi-agent system designed to automate both parameter configuration from the literature and preliminary analysis for cosmology research. SimAgents is powered by specialized LLM agents capable of physics reasoning, simulation software validation, and tool execution. These agents collaborate through structured communication, ensuring that extracted parameters are physically meaningful, internally consistent, and software-compliant. We also construct a cosmological parameter extraction evaluation dataset by collecting over 40 simulations in published papers from Arxiv and leading journals that cover diverse simulation types. Experiments on the dataset demonstrate a strong performance of SimAgents, highlighting its effectiveness and potential to accelerate scientific research for physicists. Our demonstration video is available at: this https URL. The complete system and dataset are publicly available at this https URL. 

---
# How to Train a Leader: Hierarchical Reasoning in Multi-Agent LLMs 

**Authors**: Andrew Estornell, Jean-Francois Ton, Muhammad Faaiz Taufiq, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08960)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance on a wide range of complex reasoning tasks, yet further gains are often possible by leveraging the complementary strengths of multiple models. While multi-agent frameworks can improve solution quality by leveraging multiple LLMs, existing methods are often computationally expensive, both at training and inference time. In this work, we introduce a hierarchical multi-agent framework that addresses these challenges by training only a single leader LLM to coordinate a team of untrained peer agents. To this end, we propose Multi-agent guided Leader Policy \textbf{O}ptimization (MLPO), a novel approach which trains the leader to evaluate and synthesize agent responses without auxiliary value networks or explicit agent feedback. Leaders trained with MLPO exhibit improved performance not only when interacting with the agent team at inference time, but also enjoy improved performance when deployed in single-agent settings without the team. Empirical results on Big-Bench Hard (BBH), MATH, and MMLU demonstrate that our framework achieves substantial performance improvements over both single-agent and multi-agent baselines. Our results highlight the effectiveness and efficiency of training a single, flexible leader for collaborative reasoning in multi-agent LLM systems. 

---
# Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents 

**Authors**: Enhao Zhang, Erkang Zhu, Gagan Bansal, Adam Fourney, Hussein Mozannar, Jack Gerrits  

**Link**: [PDF](https://arxiv.org/pdf/2507.08944)  

**Abstract**: Large language model (LLM)-based multi-agent systems have demonstrated remarkable promise for tackling complex tasks by breaking them down into subtasks that are iteratively planned, executed, observed, and refined. Despite their effectiveness, these systems often incur high latency because real-world problems frequently demand multiple iterative cycles of reasoning steps. To address this challenge, we propose M1-Parallel, a framework that concurrently runs multiple multi-agent teams in parallel to uncover distinct solution paths. By leveraging an event-driven communication model with asynchronous messaging, M1-Parallel efficiently capitalizes on the inherent diversity of valid plans to either reduce end-to-end latency or boost task completion rates. Our experiments on complex tasks show that M1-Parallel with early termination achieves up to $2.2\times$ speedup while preserving accuracy, and that M1-Parallel with aggregation yields higher task completion rates. We further investigate strategies aimed at encouraging diverse execution plans but observe no additional performance gains over repeated sampling. Overall, these findings underscore the potential of parallel plan execution for optimizing multi-agent systems for real-world, high-complexity reasoning tasks. 

---
# SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems 

**Authors**: Wenliang Shan, Michael Fu, Rui Yang, Chakkrit, Tantithamthavorn  

**Link**: [PDF](https://arxiv.org/pdf/2507.08898)  

**Abstract**: Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail. 

---
# ODIA: Oriented Distillation for Inline Acceleration of LLM-based Function Calling 

**Authors**: Hanlong Zhang, Jingsheng Yang, Hao Li, Yuhao He, Franck Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.08877)  

**Abstract**: Function Calling is a crucial technique that enables Large Language Models (LLMs) to interact with external systems through APIs. However, the high latency associated with LLM-based Function Calling significantly impacts user experience. This paper presents a novel approach called Oriented Distillation for Inline Acceleration (ODIA) that leverages online user interaction data to accelerate Function Calling. By automatically identifying "simple queries" from production traffic and distilling knowledge from larger models to smaller ones, our method reduces response latency by 45% (expected) and 78% (median) while maintaining accuracy. We demonstrate the effectiveness of our approach through real-world deployment in a music application, where the smaller model successfully handles 60% of traffic with negligible accuracy loss. Our method requires minimal human intervention and continuously improves through automated data collection and model updating, making it a practical solution for production environments. 

---
# LoRA Is Slower Than You Think 

**Authors**: Seokmin Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.08833)  

**Abstract**: Low-Rank Adaptation (LoRA) is one of the most widely used techniques for fine-tuning large language models (LLMs). By introducing a small number of trainable low-rank weight matrices, LoRA substantially reduces the number of parameters that need to be updated, offering significant advantages in memory consumption and computational efficiency compared to full fine-tuning. However, we observed that LoRA does not consistently provide speed improvements across all model architectures and training setups. Motivated by this inconsistency, we conduct a comprehensive analysis of LoRA's performance and investigate the underlying factors limiting its speedup. Based on our findings, we propose several methods for more efficient fine-tuning of LLMs. We empirically evaluate these methods and compare them to LoRA, demonstrating that our approach achieves comparable or superior performance while delivering more consistent training speed improvements. Our work offers valuable insights and practical guidelines for practitioners seeking to optimize LLM fine-tuning under resource constraints. 

---
# wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models 

**Authors**: Xiaohang Tang, Rares Dolga, Sangwoong Yoon, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2507.08838)  

**Abstract**: Improving the reasoning capabilities of diffusion-based large language models (dLLMs) through reinforcement learning (RL) remains an open problem. The intractability of dLLMs likelihood function necessitates approximating the current, old, and reference policy likelihoods at each policy optimization step. This reliance introduces additional computational overhead and lead to potentially large bias -- particularly when approximation errors occur in the denominator of policy ratios used for importance sampling. To mitigate these issues, we introduce $\mathtt{wd1}$, a novel policy optimization approach that reformulates the objective as a weighted likelihood, requiring only a single approximation for the current parametrized policy likelihood. Experiments on widely used reasoning benchmarks demonstrate that $\mathtt{wd1}$, without supervised fine-tuning (SFT) or any supervised data, outperforms existing RL methods for dLLMs, achieving up to 16% higher accuracy. $\mathtt{wd1}$ delivers additional computational gains, including reduced training time and fewer function evaluations (NFEs) per gradient step. These findings, combined with the simplicity of method's implementation and R1-Zero-like training (no SFT), position $\mathtt{wd1}$ as a more effective and efficient method for applying RL to dLLMs reasoning. 

---
# Grammar-Guided Evolutionary Search for Discrete Prompt Optimisation 

**Authors**: Muzhaffar Hazman, Minh-Khoi Pham, Shweta Soundararajan, Goncalo Mordido, Leonardo Custode, David Lynch, Giorgio Cruciata, Yucheng Shi, Hongmeng Song, Wang Chao, Pan Yue, Aleksandar Milenovic, Alexandros Agapitos  

**Link**: [PDF](https://arxiv.org/pdf/2507.10326)  

**Abstract**: Prompt engineering has proven to be a crucial step in leveraging pretrained large language models (LLMs) in solving various real-world tasks. Numerous solutions have been proposed that seek to automate prompt engineering by using the model itself to edit prompts. However, the majority of state-of-the-art approaches are evaluated on tasks that require minimal prompt templates and on very large and highly capable LLMs. In contrast, solving complex tasks that require detailed information to be included in the prompt increases the amount of text that needs to be optimised. Furthermore, smaller models have been shown to be more sensitive to prompt design. To address these challenges, we propose an evolutionary search approach to automated discrete prompt optimisation consisting of two phases. In the first phase, grammar-guided genetic programming is invoked to synthesise prompt-creating programmes by searching the space of programmes populated by function compositions of syntactic, dictionary-based and LLM-based prompt-editing functions. In the second phase, local search is applied to explore the neighbourhoods of best-performing programmes in an attempt to further fine-tune their performance. Our approach outperforms three state-of-the-art prompt optimisation approaches, PromptWizard, OPRO, and RL-Prompt, on three relatively small general-purpose LLMs in four domain-specific challenging tasks. We also illustrate several examples where these benchmark methods suffer relatively severe performance degradation, while our approach improves performance in almost all task-model combinations, only incurring minimal degradation when it does not. 

---
# Fusing Large Language Models with Temporal Transformers for Time Series Forecasting 

**Authors**: Chen Su, Yuanhe Tian, Qinyu Liu, Jun Zhang, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.10098)  

**Abstract**: Recently, large language models (LLMs) have demonstrated powerful capabilities in performing various tasks and thus are applied by recent studies to time series forecasting (TSF) tasks, which predict future values with the given historical time series. Existing LLM-based approaches transfer knowledge learned from text data to time series prediction using prompting or fine-tuning strategies. However, LLMs are proficient at reasoning over discrete tokens and semantic patterns but are not initially designed to model continuous numerical time series data. The gaps between text and time series data lead LLMs to achieve inferior performance to a vanilla Transformer model that is directly trained on TSF data. However, the vanilla Transformers often struggle to learn high-level semantic patterns. In this paper, we design a novel Transformer-based architecture that complementarily leverages LLMs and vanilla Transformers, so as to integrate the high-level semantic representations learned by LLMs into the temporal information encoded by time series Transformers, where a hybrid representation is obtained by fusing the representations from the LLM and the Transformer. The resulting fused representation contains both historical temporal dynamics and semantic variation patterns, allowing our model to predict more accurate future values. Experiments on benchmark datasets demonstrate the effectiveness of the proposed approach. 

---
# Using AI to replicate human experimental results: a motion study 

**Authors**: Rosa Illan Castillo, Javier Valenzuela  

**Link**: [PDF](https://arxiv.org/pdf/2507.10342)  

**Abstract**: This paper explores the potential of large language models (LLMs) as reliable analytical tools in linguistic research, focusing on the emergence of affective meanings in temporal expressions involving manner-of-motion verbs. While LLMs like GPT-4 have shown promise across a range of tasks, their ability to replicate nuanced human judgements remains under scrutiny. We conducted four psycholinguistic studies (on emergent meanings, valence shifts, verb choice in emotional contexts, and sentence-emoji associations) first with human participants and then replicated the same tasks using an LLM. Results across all studies show a striking convergence between human and AI responses, with statistical analyses (e.g., Spearman's rho = .73-.96) indicating strong correlations in both rating patterns and categorical choices. While minor divergences were observed in some cases, these did not alter the overall interpretative outcomes. These findings offer compelling evidence that LLMs can augment traditional human-based experimentation, enabling broader-scale studies without compromising interpretative validity. This convergence not only strengthens the empirical foundation of prior human-based findings but also opens possibilities for hypothesis generation and data expansion through AI. Ultimately, our study supports the use of LLMs as credible and informative collaborators in linguistic inquiry. 

---
# MLAR: Multi-layer Large Language Model-based Robotic Process Automation Applicant Tracking 

**Authors**: Mohamed T. Younes, Omar Walid, Mai Hassan, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10472)  

**Abstract**: This paper introduces an innovative Applicant Tracking System (ATS) enhanced by a novel Robotic process automation (RPA) framework or as further referred to as MLAR. Traditional recruitment processes often encounter bottlenecks in resume screening and candidate shortlisting due to time and resource constraints. MLAR addresses these challenges employing Large Language Models (LLMs) in three distinct layers: extracting key characteristics from job postings in the first layer, parsing applicant resume to identify education, experience, skills in the second layer, and similarity matching in the third layer. These features are then matched through advanced semantic algorithms to identify the best candidates efficiently. Our approach integrates seamlessly into existing RPA pipelines, automating resume parsing, job matching, and candidate notifications. Extensive performance benchmarking shows that MLAR outperforms the leading RPA platforms, including UiPath and Automation Anywhere, in high-volume resume-processing tasks. When processing 2,400 resumes, MLAR achieved an average processing time of 5.4 seconds per resume, reducing processing time by approximately 16.9% compared to Automation Anywhere and 17.1% compared to UiPath. These results highlight the potential of MLAR to transform recruitment workflows by providing an efficient, accurate, and scalable solution tailored to modern hiring needs. 

---
# GeLaCo: An Evolutionary Approach to Layer Compression 

**Authors**: David Ponce, Thierry Etchegoyhen, Javier Del Ser  

**Link**: [PDF](https://arxiv.org/pdf/2507.10059)  

**Abstract**: Large Language Models (LLM) have achieved remarkable performance across a large number of tasks, but face critical deployment and usage barriers due to substantial computational requirements. Model compression methods, which aim to reduce model size while preserving its capacity, are an important means to mitigate these issues. Promising approaches along these lines, such as structured pruning, typically require costly empirical search for optimal variants and may run the risk of ignoring better solutions. In this work we introduce GeLaCo, an evolutionary approach to LLM compression via layer collapse. Our approach supports an efficient exploration of the compression solution space via population-based search and a module-wise similarity fitness function capturing attention, feed-forward, and hidden state representations. GeLaCo also supports both single and multi-objective evolutionary compression search, establishing the first Pareto frontier along compression and quality axes. We evaluate GeLaCo solutions via both perplexity-based and generative evaluations over foundational and instruction-tuned models, outperforming state-of-the-art alternatives. 

---
# From BERT to Qwen: Hate Detection across architectures 

**Authors**: Ariadna Mon, Saúl Fenollosa, Jon Lecumberri  

**Link**: [PDF](https://arxiv.org/pdf/2507.10468)  

**Abstract**: Online platforms struggle to curb hate speech without over-censoring legitimate discourse. Early bidirectional transformer encoders made big strides, but the arrival of ultra-large autoregressive LLMs promises deeper context-awareness. Whether this extra scale actually improves practical hate-speech detection on real-world text remains unverified. Our study puts this question to the test by benchmarking both model families, classic encoders and next-generation LLMs, on curated corpora of online interactions for hate-speech detection (Hate or No Hate). 

---
# REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once 

**Authors**: Zhuoshi Pan, Qizhi Pei, Yu Li, Qiyao Sun, Zinan Tang, H. Vicky Zhao, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10541)  

**Abstract**: Recent Large Reasoning Models (LRMs) have achieved remarkable progress on task-specific benchmarks, yet their evaluation methods remain constrained by isolated problem-solving paradigms. Existing benchmarks predominantly assess single-question reasoning through sequential testing, resulting critical limitations: (1) vulnerability to data contamination and less challenging (e.g., DeepSeek-R1 achieves 97.0% on MATH500), forcing costly and perpetual creation of new questions with large human efforts, (2) failure to evaluate models under multi-context pressure, a key requirement for real-world deployment. To bridge this gap, we present REST (Reasoning Evaluation through Simultaneous Testing), a stress-testing framework that concurrently exposes LRMs to multiple problems simultaneously. Beyond basic reasoning, REST specifically evaluates several under-tested capabilities: contextual priority allocation, cross-problem interference resistance, and dynamic cognitive load management. Our evaluation reveals several striking findings: Even state-of-the-art (SOTA) models like DeepSeek-R1 exhibit substantial performance degradation under stress testing. Crucially, REST demonstrates stronger discriminative power than existing benchmarks, revealing pronounced performance differences among models that exhibit similar, near-ceiling performance under single-question evaluations. Some key mechanistic insights emerge from our analysis: (1) the "overthinking trap" is a critical factor contributing to the performance degradation; (2) the models trained with "long2short" technique preserve more accuracy of their single-problem performance under REST, outperforming standard-trained counterparts. These results establish REST as a cost-efficient, future-proof evaluation paradigm that better reflects real-world reasoning demands while reducing reliance on continuous human annotation. 

---
# Large Language Models Encode Semantics in Low-Dimensional Linear Subspaces 

**Authors**: Baturay Saglam, Paul Kassianik, Blaine Nelson, Sajana Weerawardhena, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09709)  

**Abstract**: Understanding the latent space geometry of large language models (LLMs) is key to interpreting their behavior and improving alignment. \baturay{However, it remains unclear to what extent LLMs internally organize representations related to semantic understanding. To investigate this, we conduct a large-scale empirical study of hidden states in transformer-based LLMs, analyzing 11 decoder-only models across 6 scientific topics and 12 layers each. We find that high-level semantic information consistently lies in low-dimensional subspaces that form linearly separable representations across distinct domains. This separability becomes more pronounced in deeper layers and under prompts that trigger structured reasoning or alignment behaviors$\unicode{x2013}$even when surface content is unchanged. This geometry enables simple yet effective causal interventions in hidden space; for example, reasoning patterns like chain-of-thought can be captured by a single vector direction. Together, these findings support the development of geometry-aware tools that operate directly on latent representations to detect and mitigate harmful or adversarial content, using methods such as transport-based defenses that leverage this separability. As a proof of concept, we demonstrate this potential by training a simple MLP classifier as a lightweight latent-space guardrail, which detects adversarial and malicious prompts with high precision. 

---
# MCEval: A Dynamic Framework for Fair Multilingual Cultural Evaluation of LLMs 

**Authors**: Shulin Huang, Linyi Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09701)  

**Abstract**: Large language models exhibit cultural biases and limited cross-cultural understanding capabilities, particularly when serving diverse global user populations. We propose MCEval, a novel multilingual evaluation framework that employs dynamic cultural question construction and enables causal analysis through Counterfactual Rephrasing and Confounder Rephrasing. Our comprehensive evaluation spans 13 cultures and 13 languages, systematically assessing both cultural awareness and cultural bias across different linguistic scenarios. The framework provides 39,897 cultural awareness instances and 17,940 cultural bias instances. Experimental results reveal performance disparities across different linguistic scenarios, demonstrating that optimal cultural performance is not only linked to training data distribution, but also is related to language-culture alignment. The evaluation results also expose the fairness issue, where approaches appearing successful in the English scenario create substantial disadvantages. MCEval represents the first comprehensive multilingual cultural evaluation framework that provides deeper insights into LLMs' cultural understanding. 

---
# How Important is `Perfect' English for Machine Translation Prompts? 

**Authors**: Patrícia Schmidtová, Niyati Bafna, Seth Aycock, Gianluca Vico, Wiktor Kamzela, Katharina Hämmerl, Vilém Zouhar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09509)  

**Abstract**: Large language models (LLMs) have achieved top results in recent machine translation evaluations, but they are also known to be sensitive to errors and perturbations in their prompts. We systematically evaluate how both humanly plausible and synthetic errors in user prompts affect LLMs' performance on two related tasks: Machine translation and machine translation evaluation. We provide both a quantitative analysis and qualitative insights into how the models respond to increasing noise in the user prompt.
The prompt quality strongly affects the translation performance: With many errors, even a good prompt can underperform a minimal or poor prompt without errors. However, different noise types impact translation quality differently, with character-level and combined noisers degrading performance more than phrasal perturbations. Qualitative analysis reveals that lower prompt quality largely leads to poorer instruction following, rather than directly affecting translation quality itself. Further, LLMs can still translate in scenarios with overwhelming random noise that would make the prompt illegible to humans. 

---
# DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models 

**Authors**: Cathy Jiao, Yijun Pan, Emily Xiao, Daisy Sheng, Niket Jain, Hanzhang Zhao, Ishita Dasgupta, Jiaqi W. Ma, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.09424)  

**Abstract**: Data attribution methods quantify the influence of training data on model outputs and are becoming increasingly relevant for a wide range of LLM research and applications, including dataset curation, model interpretability, data valuation. However, there remain critical gaps in systematic LLM-centric evaluation of data attribution methods. To this end, we introduce DATE-LM (Data Attribution Evaluation in Language Models), a unified benchmark for evaluating data attribution methods through real-world LLM applications. DATE-LM measures attribution quality through three key tasks -- training data selection, toxicity/bias filtering, and factual attribution. Our benchmark is designed for ease of use, enabling researchers to configure and run large-scale evaluations across diverse tasks and LLM architectures. Furthermore, we use DATE-LM to conduct a large-scale evaluation of existing data attribution methods. Our findings show that no single method dominates across all tasks, data attribution methods have trade-offs with simpler baselines, and method performance is sensitive to task-specific evaluation design. Finally, we release a public leaderboard for quick comparison of methods and to facilitate community engagement. We hope DATE-LM serves as a foundation for future data attribution research in LLMs. 

---
# Psychology-Driven Enhancement of Humour Translation 

**Authors**: Yuchen Su, Yonghua Zhu, Yang Chen, Diana Benavides-Prado, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2507.09259)  

**Abstract**: Humour translation plays a vital role as a bridge between different cultures, fostering understanding and communication. Although most existing Large Language Models (LLMs) are capable of general translation tasks, these models still struggle with humour translation, which is especially reflected through linguistic interference and lacking humour in translated text. In this paper, we propose a psychology-inspired Humour Decomposition Mechanism (HDM) that utilises Chain-of-Thought (CoT) to imitate the ability of the human thought process, stimulating LLMs to optimise the readability of translated humorous texts. Moreover, we integrate humour theory in HDM to further enhance the humorous elements in the translated text. Our automatic evaluation experiments on open-source humour datasets demonstrate that our method significantly improves the quality of humour translation, yielding average gains of 7.75\% in humour, 2.81\% in fluency, and 6.13\% in coherence of the generated text. 

---
# Detecting and Pruning Prominent but Detrimental Neurons in Large Language Models 

**Authors**: Ameen Ali, Shahar Katz, Lior Wolf, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2507.09185)  

**Abstract**: Large language models (LLMs) often develop learned mechanisms specialized to specific datasets, such as reliance on domain-specific correlations, which yield high-confidence predictions without generalizable reasoning. While beneficial in one setting, these dataset-specific mechanisms typically degrade performance when models encounter novel tasks or distributions. In this work, we introduce a fine-tuning approach designed to enhance generalization by identifying and pruning neurons associated with dataset-specific mechanisms in transformer-based LLMs. Our method employs Integrated Gradients to quantify each neuron's influence on high-confidence predictions, pinpointing those that disproportionately contribute to dataset-specific performance without supporting robust, transferable reasoning. Selectively pruning these neurons compels the model to depend on generalizable representations. Evaluated across multiple-choice benchmarks, our pruning-based fine-tuning significantly enhances performance, surpassing prior (non-pruning) adaptation methods. 

---
# Balanced Training Data Augmentation for Aspect-Based Sentiment Analysis 

**Authors**: Junjie Liu, Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.09485)  

**Abstract**: Aspect-based sentiment analysis (ABSA) is a crucial fine-grained task in social media scenarios to identify the sentiment polarity of specific aspect terms in a sentence. Although many existing studies leverage large language models (LLMs) to perform ABSA due to their strong context understanding capabilities, they still face challenges to learn the context information in the running text because of the short text, as well as the small and unbalanced labeled training data, where most data are labeled with positive sentiment. Data augmentation (DA) is a feasible strategy for providing richer contextual information, especially when using LLMs to create synthetic training data, but faces challenges in ensuring a high quality of the augmented this http URL this paper, we propose an LLM-based ABSA approach with training data this http URL, an LLM is prompted to generate augmented training data based on the original training data, so as to construct a new training data with larger size and balanced label distributions to better train an ABSA model. Meanwhile, in order to improve the quality of the augmented data, we propose a reinforcement learning approach to optimize the data augmentation. this http URL results and further analyses on English benchmark datasets for ABSA demonstrate the effectiveness of our approach, where superior performance is observed over strong baselines and most existing studies. 

---
# OpenCodeReasoning-II: A Simple Test Time Scaling Approach via Self-Critique 

**Authors**: Wasi Uddin Ahmad, Somshubra Majumdar, Aleksander Ficek, Sean Narenthiran, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Vahid Noroozi, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2507.09075)  

**Abstract**: Recent advancements in reasoning-based Large Language Models (LLMs), particularly their potential through test-time scaling, have created significant opportunities for distillation in code generation and critique. However, progress in both areas fundamentally depends on large-scale, high-quality datasets. In this work, we introduce OpenCodeReasoning-II, a dataset consists of 2.5M question-solution-critique triples (approx. 35K unique programming questions), making it nearly twice the size of the previous largest publicly available code reasoning dataset. In this work, we employ a two-stage supervised fine-tuning strategy. The first stage focuses on fine-tuning for code generation, while the second stage involves the joint training of models for both code generation and critique. Our resulting finetuned Qwen2.5-Instruct models achieve performance in code generation that either exceeds or equals the best prior open-weight distilled models. Notably, the integration of our code generation and critique models leads to significant improvements in competitive coding performance. Furthermore, we present an extension of the LiveCodeBench benchmark to specifically support the C++ programming language, thereby facilitating more comprehensive LLM evaluation using this benchmark. 

---
# Self-Improving Model Steering 

**Authors**: Rongyi Zhu, Yuhui Wang, Tanqiu Jiang, Jiacheng Liang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08967)  

**Abstract**: Model steering represents a powerful technique that dynamically aligns large language models (LLMs) with human preferences during inference. However, conventional model-steering methods rely heavily on externally annotated data, not only limiting their adaptability to varying contexts but also tethering their effectiveness to annotation quality. In this paper, we present SIMS, the first self-improving model-steering framework that operates without relying on external supervision. At its core, SIMS autonomously generates and refines contrastive samples through iterative self-improvement cycles, enabling adaptive, context-specific steering. Additionally, SIMS employs novel strategies, including prompt ranking and contrast sampling, to further enhance steering efficacy. Extensive evaluation across diverse LLMs and benchmarks demonstrates that SIMS substantially outperforms existing methods in steering effectiveness and adaptability, highlighting self-improving model steering as a promising direction for future research on inference-time LLM alignment. 

---
# Lizard: An Efficient Linearization Framework for Large Language Models 

**Authors**: Chien Van Nguyen, Ruiyi Zhang, Hanieh Deilamsalehy, Puneet Mathur, Viet Dac Lai, Haoliang Wang, Jayakumar Subramanian, Ryan A. Rossi, Trung Bui, Nikos Vlassis, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09025)  

**Abstract**: We propose Lizard, a linearization framework that transforms pretrained Transformer-based Large Language Models (LLMs) into flexible, subquadratic architectures for infinite-context generation. Transformer-based LLMs face significant memory and computational bottlenecks as context lengths increase, due to the quadratic complexity of softmax attention and the growing key-value (KV) cache. Lizard addresses these limitations by introducing a subquadratic attention mechanism that closely approximates softmax attention while preserving the output quality. Unlike previous linearization methods, which are often limited by fixed model structures and therefore exclude gating mechanisms, Lizard incorporates a gating module inspired by recent state-of-the-art linear models. This enables adaptive memory control, supports constant-memory inference, offers strong length generalization, and allows more flexible model design. Lizard combines gated linear attention for global context compression with sliding window attention enhanced by meta memory, forming a hybrid mechanism that captures both long-range dependencies and fine-grained local interactions. Moreover, we introduce a hardware-aware algorithm that accelerates the training speed of our models. Extensive experiments show that Lizard achieves near-lossless recovery of the teacher model's performance across standard language modeling tasks, while significantly outperforming previous linearization methods. On the 5-shot MMLU benchmark, Lizard improves over prior models by 18 points and shows significant improvements on associative recall tasks. 

---
# Evaluating LLMs in Medicine: A Call for Rigor, Transparency 

**Authors**: Mahmoud Alwakeel, Aditya Nagori, Vijay Krishnamoorthy, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.08916)  

**Abstract**: Objectives: To evaluate the current limitations of large language models (LLMs) in medical question answering, focusing on the quality of datasets used for their evaluation. Materials and Methods: Widely-used benchmark datasets, including MedQA, MedMCQA, PubMedQA, and MMLU, were reviewed for their rigor, transparency, and relevance to clinical scenarios. Alternatives, such as challenge questions in medical journals, were also analyzed to identify their potential as unbiased evaluation tools. Results: Most existing datasets lack clinical realism, transparency, and robust validation processes. Publicly available challenge questions offer some benefits but are limited by their small size, narrow scope, and exposure to LLM training. These gaps highlight the need for secure, comprehensive, and representative datasets. Conclusion: A standardized framework is critical for evaluating LLMs in medicine. Collaborative efforts among institutions and policymakers are needed to ensure datasets and methodologies are rigorous, unbiased, and reflective of clinical complexities. 

---
# Can Group Relative Policy Optimization Improve Thai Legal Reasoning and Question Answering? 

**Authors**: Pawitsapak Akarajaradwong, Chompakorn Chaksangchaichot, Pirat Pothavorn, Attapol Thamrongrattanarit-Rutherford, Ekapol Chuangsuwanich, Sarana Nutanong  

**Link**: [PDF](https://arxiv.org/pdf/2507.09638)  

**Abstract**: The Retrieval-Augmented Generation (RAG) systems' performance on Thai legal question answering is still limited, especially for questions requiring extensive, complex legal reasoning. To address these limitations, we introduce an approach aligning LLMs toward improved law citation accuracy and better response quality using Group-Relative Policy Optimization (GRPO). Our approach leverages BGE-M3 embeddings as a cost-efficient semantic-similarity reward, significantly reducing computational expenses up to 2.5x compared to large language model judges. Experiments on the NitiBench benchmark demonstrate substantial improvements: GRPO achieves up to 90% citation-F1 gains from the base model and a 31% increase in joint quality metrics over instruction tuning. Crucially, our method shows enhanced robustness on complex legal reasoning tasks compared to instruction tuning, providing an effective and resource-efficient solution for enhancing Thai legal LLMs. 

---
# Semantic Source Code Segmentation using Small and Large Language Models 

**Authors**: Abdelhalim Dahou, Ansgar Scherp, Sebastian Kurten, Brigitte Mathiak, Madhu Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08992)  

**Abstract**: Source code segmentation, dividing code into functionally coherent segments, is crucial for knowledge retrieval and maintenance in software development. While enabling efficient navigation and comprehension of large codebases, manual and syntactic analysis approaches have become impractical as repositories grow, especially for low-resource languages like R and their research domains (e.g., social sciences, psychology).This paper introduces an automated, domain-specific approach for research R code segmentation using Large and Small Language Models (LLMs/SLMs). It presents two novel approaches and a human-annotated dataset, StatCodeSeg. We explore two distinct approaches: line-by-line analysis with context and range-based segment determination. We experiment with LLMs and fine-tuned SLMs. To support the generalizability of our approaches, we also include experiments on Python code from the computer science this http URL results show that context-based line-by-line analysis is superior over range-based this http URL smaller language models like CodeBERT and an encoder-only version of CodeT5+ are better than their LLM counterparts. Most notably, these two best-performing models did not see R code during pre-training versus the LLMs but were only fine-tuned on 4,130 lines of manually annotated code. 

---
# RAG Safety: Exploring Knowledge Poisoning Attacks to Retrieval-Augmented Generation 

**Authors**: Tianzhe Zhao, Jiaoyan Chen, Yanchi Ru, Haiping Zhu, Nan Hu, Jun Liu, Qika Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08862)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving external data to mitigate hallucinations and outdated knowledge issues. Benefiting from the strong ability in facilitating diverse data sources and supporting faithful reasoning, knowledge graphs (KGs) have been increasingly adopted in RAG systems, giving rise to KG-based RAG (KG-RAG) methods. Though RAG systems are widely applied in various applications, recent studies have also revealed its vulnerabilities to data poisoning attacks, where malicious information injected into external knowledge sources can mislead the system into producing incorrect or harmful responses. However, these studies focus exclusively on RAG systems using unstructured textual data sources, leaving the security risks of KG-RAG largely unexplored, despite the fact that KGs present unique vulnerabilities due to their structured and editable nature. In this work, we conduct the first systematic investigation of the security issue of KG-RAG methods through data poisoning attacks. To this end, we introduce a practical, stealthy attack setting that aligns with real-world implementation. We propose an attack strategy that first identifies adversarial target answers and then inserts perturbation triples to complete misleading inference chains in the KG, increasing the likelihood that KG-RAG methods retrieve and rely on these perturbations during generation. Through extensive experiments on two benchmarks and four recent KG-RAG methods, our attack strategy demonstrates strong effectiveness in degrading KG-RAG performance, even with minimal KG perturbations. In-depth analyses are also conducted to understand the safety threats within the internal stages of KG-RAG systems and to explore the robustness of LLMs against adversarial knowledge. 

---
