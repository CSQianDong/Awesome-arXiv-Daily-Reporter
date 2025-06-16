# Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization 

**Authors**: Wenqi Liu, Xuemeng Song, Jiaxi Li, Yinwei Wei, Na Zheng, Jianhua Yin, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.11712)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for mitigating hallucination in Multimodal Large Language Models (MLLMs). Although existing methods have achieved significant progress by utilizing vision-oriented contrastive objectives for enhancing MLLMs' attention to visual inputs and hence reducing hallucination, they suffer from non-rigorous optimization objective function and indirect preference supervision. To address these limitations, we propose a Symmetric Multimodal Preference Optimization (SymMPO), which conducts symmetric preference learning with direct preference supervision (i.e., response pairs) for visual understanding enhancement, while maintaining rigorous theoretical alignment with standard DPO. In addition to conventional ordinal preference learning, SymMPO introduces a preference margin consistency loss to quantitatively regulate the preference gap between symmetric preference pairs. Comprehensive evaluation across five benchmarks demonstrate SymMPO's superior performance, validating its effectiveness in hallucination mitigation of MLLMs. 

---
# Tracing LLM Reasoning Processes with Strategic Games: A Framework for Planning, Revision, and Resource-Constrained Decision Making 

**Authors**: Xiaopeng Yuan, Xingjian Zhang, Ke Xu, Yifan Xu, Lijun Yu, Jindong Wang, Yushun Dong, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12012)  

**Abstract**: Large language models (LLMs) are increasingly used for tasks that require complex reasoning. Most benchmarks focus on final outcomes but overlook the intermediate reasoning steps - such as planning, revision, and decision making under resource constraints. We argue that measuring these internal processes is essential for understanding model behavior and improving reliability. We propose using strategic games as a natural evaluation environment: closed, rule-based systems with clear states, limited resources, and automatic feedback. We introduce a framework that evaluates LLMs along three core dimensions: planning, revision, and resource-constrained decision making. To operationalize this, we define metrics beyond win rate, including overcorrection risk rate, correction success rate, improvement slope, and over-budget ratio. In 4320 adversarial rounds across 12 leading models, ChatGPT-o3-mini achieves the top composite score, with a win rate of 74.7 percent, a correction success rate of 78.6 percent, and an improvement slope of 0.041. By contrast, Qwen-Plus, despite an overcorrection risk rate of 81.6 percent, wins only 25.6 percent of its matches - primarily due to excessive resource use. We also observe a negative correlation between overcorrection risk rate and correction success rate (Pearson r = -0.51, p = 0.093), suggesting that more frequent edits do not always improve outcomes. Our findings highlight the value of assessing not only what LLMs decide but how they arrive at those decisions 

---
# Addressing Bias in LLMs: Strategies and Application to Fair AI-based Recruitment 

**Authors**: Alejandro Peña, Julian Fierrez, Aythami Morales, Gonzalo Mancera, Miguel Lopez, Ruben Tolosana  

**Link**: [PDF](https://arxiv.org/pdf/2506.11880)  

**Abstract**: The use of language technologies in high-stake settings is increasing in recent years, mostly motivated by the success of Large Language Models (LLMs). However, despite the great performance of LLMs, they are are susceptible to ethical concerns, such as demographic biases, accountability, or privacy. This work seeks to analyze the capacity of Transformers-based systems to learn demographic biases present in the data, using a case study on AI-based automated recruitment. We propose a privacy-enhancing framework to reduce gender information from the learning pipeline as a way to mitigate biased behaviors in the final tools. Our experiments analyze the influence of data biases on systems built on two different LLMs, and how the proposed framework effectively prevents trained systems from reproducing the bias in the data. 

---
# On the Performance of LLMs for Real Estate Appraisal 

**Authors**: Margot Geerts, Manon Reusens, Bart Baesens, Seppe vanden Broucke, Jochen De Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2506.11812)  

**Abstract**: The real estate market is vital to global economies but suffers from significant information asymmetry. This study examines how Large Language Models (LLMs) can democratize access to real estate insights by generating competitive and interpretable house price estimates through optimized In-Context Learning (ICL) strategies. We systematically evaluate leading LLMs on diverse international housing datasets, comparing zero-shot, few-shot, market report-enhanced, and hybrid prompting techniques. Our results show that LLMs effectively leverage hedonic variables, such as property size and amenities, to produce meaningful estimates. While traditional machine learning models remain strong for pure predictive accuracy, LLMs offer a more accessible, interactive and interpretable alternative. Although self-explanations require cautious interpretation, we find that LLMs explain their predictions in agreement with state-of-the-art models, confirming their trustworthiness. Carefully selected in-context examples based on feature similarity and geographic proximity, significantly enhance LLM performance, yet LLMs struggle with overconfidence in price intervals and limited spatial reasoning. We offer practical guidance for structured prediction tasks through prompt optimization. Our findings highlight LLMs' potential to improve transparency in real estate appraisal and provide actionable insights for stakeholders. 

---
# Collaborative LLM Inference via Planning for Efficient Reasoning 

**Authors**: Byeongchan Lee, Jonghoon Lee, Dongyoung Kim, Jaehyung Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11578)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks, but those with strong capabilities (e.g., whose numbers of parameters are larger than 100B) are often accessible only through paid APIs, making them too costly for applications of frequent use. In contrast, smaller open-sourced LLMs (e.g., whose numbers of parameters are less than 3B) are freely available and easy to deploy locally (e.g., under a single GPU having 8G VRAM), but lack suff icient reasoning ability. This trade-off raises a natural question: can small (free) and large (costly) models collaborate at test time to combine their strengths? We propose a test-time collaboration framework in which a planner model first generates a plan, defined as a distilled and high-level abstraction of the problem.
This plan serves as a lightweight intermediate that guides a reasoner model, which generates a complete solution. Small and large models take turns acting as planner and reasoner, exchanging plans in a multi-round cascade to collaboratively solve complex tasks. Our method achieves accuracy comparable to strong proprietary models alone, while significantly reducing reliance on paid inference. These results highlight planning as an effective prior for orchestrating cost-aware, cross-model inference under real-world deployment constraints. 

---
# RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning 

**Authors**: Yu Wang, Shiwan Zhao, Ming Fan, Zhihu Wang, Yubo Zhang, Xicheng Zhang, Zhengfan Wang, Heyuan Huang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11555)  

**Abstract**: The integration of external knowledge through Retrieval-Augmented Generation (RAG) has become foundational in enhancing large language models (LLMs) for knowledge-intensive tasks. However, existing RAG paradigms often overlook the cognitive step of applying knowledge, leaving a gap between retrieved facts and task-specific reasoning. In this work, we introduce RAG+, a principled and modular extension that explicitly incorporates application-aware reasoning into the RAG pipeline. RAG+ constructs a dual corpus consisting of knowledge and aligned application examples, created either manually or automatically, and retrieves both jointly during inference. This design enables LLMs not only to access relevant information but also to apply it within structured, goal-oriented reasoning processes. Experiments across mathematical, legal, and medical domains, conducted on multiple models, demonstrate that RAG+ consistently outperforms standard RAG variants, achieving average improvements of 3-5%, and peak gains up to 7.5% in complex scenarios. By bridging retrieval with actionable application, RAG+ advances a more cognitively grounded framework for knowledge integration, representing a step toward more interpretable and capable LLMs. 

---
# Large Language Model-Powered Conversational Agent Delivering Problem-Solving Therapy (PST) for Family Caregivers: Enhancing Empathy and Therapeutic Alliance Using In-Context Learning 

**Authors**: Liying Wang, Ph.D., Daffodil Carrington, M.S., Daniil Filienko, M.S., Caroline El Jazmi, M.S., Serena Jinchen Xie, M.S., Martine De Cock, Ph.D., Sarah Iribarren, Ph.D., Weichao Yuwen, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2506.11376)  

**Abstract**: Family caregivers often face substantial mental health challenges due to their multifaceted roles and limited resources. This study explored the potential of a large language model (LLM)-powered conversational agent to deliver evidence-based mental health support for caregivers, specifically Problem-Solving Therapy (PST) integrated with Motivational Interviewing (MI) and Behavioral Chain Analysis (BCA). A within-subject experiment was conducted with 28 caregivers interacting with four LLM configurations to evaluate empathy and therapeutic alliance. The best-performing models incorporated Few-Shot and Retrieval-Augmented Generation (RAG) prompting techniques, alongside clinician-curated examples. The models showed improved contextual understanding and personalized support, as reflected by qualitative responses and quantitative ratings on perceived empathy and therapeutic alliances. Participants valued the model's ability to validate emotions, explore unexpressed feelings, and provide actionable strategies. However, balancing thorough assessment with efficient advice delivery remains a challenge. This work highlights the potential of LLMs in delivering empathetic and tailored support for family caregivers. 

---
# Revealing Political Bias in LLMs through Structured Multi-Agent Debate 

**Authors**: Aishwarya Bandaru, Fabian Bindley, Trevor Bluth, Nandini Chavda, Baixu Chen, Ethan Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.11825)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate social behaviour, yet their political biases and interaction dynamics in debates remain underexplored. We investigate how LLM type and agent gender attributes influence political bias using a structured multi-agent debate framework, by engaging Neutral, Republican, and Democrat American LLM agents in debates on politically sensitive topics. We systematically vary the underlying LLMs, agent genders, and debate formats to examine how model provenance and agent personas influence political bias and attitudes throughout debates. We find that Neutral agents consistently align with Democrats, while Republicans shift closer to the Neutral; gender influences agent attitudes, with agents adapting their opinions when aware of other agents' genders; and contrary to prior research, agents with shared political affiliations can form echo chambers, exhibiting the expected intensification of attitudes as debates progress. 

---
# Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables 

**Authors**: Yitong Zhou, Mingyue Cheng, Qingyang Mao, Yucong Luo, Qi Liu, Yupeng Li, Xiaohan Zhang, Deguang Liu, Xin Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11375)  

**Abstract**: Chemical tables encode complex experimental knowledge through symbolic expressions, structured variables, and embedded molecular graphics. Existing benchmarks largely overlook this multimodal and domain-specific complexity, limiting the ability of multimodal large language models to support scientific understanding in chemistry. In this work, we introduce ChemTable, a large-scale benchmark of real-world chemical tables curated from the experimental sections of literature. ChemTable includes expert-annotated cell polygons, logical layouts, and domain-specific labels, including reagents, catalysts, yields, and graphical components and supports two core tasks: (1) Table Recognition, covering structure parsing and content extraction; and (2) Table Understanding, encompassing both descriptive and reasoning-oriented question answering grounded in table structure and domain semantics. We evaluated a range of representative multimodal models, including both open-source and closed-source models, on ChemTable and reported a series of findings with practical and conceptual insights. Although models show reasonable performance on basic layout parsing, they exhibit substantial limitations on both descriptive and inferential QA tasks compared to human performance, and we observe significant performance gaps between open-source and closed-source models across multiple dimensions. These results underscore the challenges of chemistry-aware table understanding and position ChemTable as a rigorous and realistic benchmark for advancing scientific reasoning. 

---
# LLM-as-a-Fuzzy-Judge: Fine-Tuning Large Language Models as a Clinical Evaluation Judge with Fuzzy Logic 

**Authors**: Weibing Zheng, Laurah Turner, Jess Kropczynski, Murat Ozer, Tri Nguyen, Shane Halse  

**Link**: [PDF](https://arxiv.org/pdf/2506.11221)  

**Abstract**: Clinical communication skills are critical in medical education, and practicing and assessing clinical communication skills on a scale is challenging. Although LLM-powered clinical scenario simulations have shown promise in enhancing medical students' clinical practice, providing automated and scalable clinical evaluation that follows nuanced physician judgment is difficult. This paper combines fuzzy logic and Large Language Model (LLM) and proposes LLM-as-a-Fuzzy-Judge to address the challenge of aligning the automated evaluation of medical students' clinical skills with subjective physicians' preferences. LLM-as-a-Fuzzy-Judge is an approach that LLM is fine-tuned to evaluate medical students' utterances within student-AI patient conversation scripts based on human annotations from four fuzzy sets, including Professionalism, Medical Relevance, Ethical Behavior, and Contextual Distraction. The methodology of this paper started from data collection from the LLM-powered medical education system, data annotation based on multidimensional fuzzy sets, followed by prompt engineering and the supervised fine-tuning (SFT) of the pre-trained LLMs using these human annotations. The results show that the LLM-as-a-Fuzzy-Judge achieves over 80\% accuracy, with major criteria items over 90\%, effectively leveraging fuzzy logic and LLM as a solution to deliver interpretable, human-aligned assessment. This work suggests the viability of leveraging fuzzy logic and LLM to align with human preferences, advances automated evaluation in medical education, and supports more robust assessment and judgment practices. The GitHub repository of this work is available at this https URL 

---
# Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making 

**Authors**: Claudio Fanconi, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.11887)  

**Abstract**: Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions. 

---
# code_transformed: The Influence of Large Language Models on Code 

**Authors**: Yuliang Xu, Siming Huang, Mingmeng Geng, Yao Wan, Xuanhua Shi, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12014)  

**Abstract**: Coding remains one of the most fundamental modes of interaction between humans and machines. With the rapid advancement of Large Language Models (LLMs), code generation capabilities have begun to significantly reshape programming practices. This development prompts a central question: Have LLMs transformed code style, and how can such transformation be characterized? In this paper, we present a pioneering study that investigates the impact of LLMs on code style, with a focus on naming conventions, complexity, maintainability, and similarity. By analyzing code from over 19,000 GitHub repositories linked to arXiv papers published between 2020 and 2025, we identify measurable trends in the evolution of coding style that align with characteristics of LLM-generated code. For instance, the proportion of snake\_case variable names in Python code increased from 47% in Q1 2023 to 51% in Q1 2025. Furthermore, we investigate how LLMs approach algorithmic problems by examining their reasoning processes. Given the diversity of LLMs and usage scenarios, among other factors, it is difficult or even impossible to precisely estimate the proportion of code generated or assisted by LLMs. Our experimental results provide the first large-scale empirical evidence that LLMs affect real-world programming style. 

---
# A Survey of Task-Oriented Knowledge Graph Reasoning: Status, Applications, and Prospects 

**Authors**: Guanglin Niu, Bo Li, Yangguang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11012)  

**Abstract**: Knowledge graphs (KGs) have emerged as a powerful paradigm for structuring and leveraging diverse real-world knowledge, which serve as a fundamental technology for enabling cognitive intelligence systems with advanced understanding and reasoning capabilities. Knowledge graph reasoning (KGR) aims to infer new knowledge based on existing facts in KGs, playing a crucial role in applications such as public security intelligence, intelligent healthcare, and financial risk assessment. From a task-centric perspective, existing KGR approaches can be broadly classified into static single-step KGR, static multi-step KGR, dynamic KGR, multi-modal KGR, few-shot KGR, and inductive KGR. While existing surveys have covered these six types of KGR tasks, a comprehensive review that systematically summarizes all KGR tasks particularly including downstream applications and more challenging reasoning paradigms remains lacking. In contrast to previous works, this survey provides a more comprehensive perspective on the research of KGR by categorizing approaches based on primary reasoning tasks, downstream application tasks, and potential challenging reasoning tasks. Besides, we explore advanced techniques, such as large language models (LLMs), and their impact on KGR. This work aims to highlight key research trends and outline promising future directions in the field of KGR. 

---
# VGR: Visual Grounded Reasoning 

**Authors**: Jiacong Wang, Zijiang Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11991)  

**Abstract**: In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA. 

---
# Improving Large Language Model Safety with Contrastive Representation Learning 

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11938)  

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at this https URL 

---
# LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming? 

**Authors**: Zihan Zheng, Zerui Cheng, Zeyu Shen, Shang Zhou, Kaiyuan Liu, Hansen He, Dongruixuan Li, Stanley Wei, Hangyi Hao, Jianzhu Yao, Peiyao Sheng, Zixuan Wang, Wenhao Chai, Aleksandra Korolova, Peter Henderson, Sanjeev Arora, Pramod Viswanath, Jingbo Shang, Saining Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.11928)  

**Abstract**: Recent reports claim that large language models (LLMs) now outperform elite humans in competitive programming. Drawing on knowledge from a group of medalists in international algorithmic contests, we revisit this claim, examining how LLMs differ from human experts and where limitations still remain. We introduce LiveCodeBench Pro, a benchmark composed of problems from Codeforces, ICPC, and IOI that are continuously updated to reduce the likelihood of data contamination. A team of Olympiad medalists annotates every problem for algorithmic categories and conducts a line-by-line analysis of failed model-generated submissions. Using this new data and benchmark, we find that frontier models still have significant limitations: without external tools, the best model achieves only 53% pass@1 on medium-difficulty problems and 0% on hard problems, domains where expert humans still excel. We also find that LLMs succeed at implementation-heavy problems but struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications. High performance appears largely driven by implementation precision and tool augmentation, not superior reasoning. LiveCodeBench Pro thus highlights the significant gap to human grandmaster levels, while offering fine-grained diagnostics to steer future improvements in code-centric LLM reasoning. 

---
# OntoGSN: An Ontology for Dynamic Management of Assurance Cases 

**Authors**: Tomas Bueno Momcilovic, Barbara Gallina, Ingmar Kessler, Dian Balta  

**Link**: [PDF](https://arxiv.org/pdf/2506.11023)  

**Abstract**: Assurance cases (ACs) are a common artifact for building and maintaining confidence in system properties such as safety or robustness. Constructing an AC can be challenging, although existing tools provide support in static, document-centric applications and methods for dynamic contexts (e.g., autonomous driving) are emerging. Unfortunately, managing ACs remains a challenge, since maintaining the embedded knowledge in the face of changes requires substantial effort, in the process deterring developers - or worse, producing poorly managed cases that instill false confidence. To address this, we present OntoGSN: an ontology and supporting middleware for managing ACs in the Goal Structuring Notation (GSN) standard. OntoGSN offers a knowledge representation and a queryable graph that can be automatically populated, evaluated, and updated. Our contributions include: a 1:1 formalization of the GSN Community Standard v3 in an OWL ontology with SWRL rules; a helper ontology and parser for integration with a widely used AC tool; a repository and documentation of design decisions for OntoGSN maintenance; a SPARQL query library with automation patterns; and a prototypical interface. The ontology strictly adheres to the standard's text and has been evaluated according to FAIR principles, the OOPS framework, competency questions, and community feedback. The development of other middleware elements is guided by the community needs and subject to ongoing evaluations. To demonstrate the utility of our contributions, we illustrate dynamic AC management in an example involving assurance of adversarial robustness in large language models. 

---
# TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks 

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11844)  

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field. 

---
# Persona-driven Simulation of Voting Behavior in the European Parliament with Large Language Models 

**Authors**: Maximilian Kreutner, Marlene Lutz, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2506.11798)  

**Abstract**: Large Language Models (LLMs) display remarkable capabilities to understand or even produce political discourse, but have been found to consistently display a progressive left-leaning bias. At the same time, so-called persona or identity prompts have been shown to produce LLM behavior that aligns with socioeconomic groups that the base model is not aligned with. In this work, we analyze whether zero-shot persona prompting with limited information can accurately predict individual voting decisions and, by aggregation, accurately predict positions of European groups on a diverse set of policies. We evaluate if predictions are stable towards counterfactual arguments, different persona prompts and generation methods. Finally, we find that we can simulate voting behavior of Members of the European Parliament reasonably well with a weighted F1 score of approximately 0.793. Our persona dataset of politicians in the 2024 European Parliament and our code are available at this https URL. 

---
# LoRA-Gen: Specializing Large Language Model via Online LoRA Generation 

**Authors**: Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Yixiao Ge, Xiu Li, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11638)  

**Abstract**: Recent advances have highlighted the benefits of scaling language models to enhance performance across a wide range of NLP tasks. However, these approaches still face limitations in effectiveness and efficiency when applied to domain-specific tasks, particularly for small edge-side models. We propose the LoRA-Gen framework, which utilizes a large cloud-side model to generate LoRA parameters for edge-side models based on task descriptions. By employing the reparameterization technique, we merge the LoRA parameters into the edge-side model to achieve flexible specialization. Our method facilitates knowledge transfer between models while significantly improving the inference efficiency of the specialized model by reducing the input context length. Without specialized training, LoRA-Gen outperforms conventional LoRA fine-tuning, which achieves competitive accuracy and a 2.1x speedup with TinyLLaMA-1.1B in reasoning tasks. Besides, our method delivers a compression ratio of 10.1x with Gemma-2B on intelligent agent tasks. 

---
# FAA Framework: A Large Language Model-Based Approach for Credit Card Fraud Investigations 

**Authors**: Shaun Shuster, Eyal Zaloof, Asaf Shabtai, Rami Puzis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11635)  

**Abstract**: The continuous growth of the e-commerce industry attracts fraudsters who exploit stolen credit card details. Companies often investigate suspicious transactions in order to retain customer trust and address gaps in their fraud detection systems. However, analysts are overwhelmed with an enormous number of alerts from credit card transaction monitoring systems. Each alert investigation requires from the fraud analysts careful attention, specialized knowledge, and precise documentation of the outcomes, leading to alert fatigue. To address this, we propose a fraud analyst assistant (FAA) framework, which employs multi-modal large language models (LLMs) to automate credit card fraud investigations and generate explanatory reports. The FAA framework leverages the reasoning, code execution, and vision capabilities of LLMs to conduct planning, evidence collection, and analysis in each investigation step. A comprehensive empirical evaluation of 500 credit card fraud investigations demonstrates that the FAA framework produces reliable and efficient investigations comprising seven steps on average. Thus we found that the FAA framework can automate large parts of the workload and help reduce the challenges faced by fraud analysts. 

---
# GraphRAG-Causal: A novel graph-augmented framework for causal reasoning and annotation in news 

**Authors**: Abdul Haque, Umm e Hani, Ahmad Din, Muhammad Babar, Ali Abbas, Insaf Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2506.11600)  

**Abstract**: GraphRAG-Causal introduces an innovative framework that combines graph-based retrieval with large language models to enhance causal reasoning in news analysis. Traditional NLP approaches often struggle with identifying complex, implicit causal links, especially in low-data scenarios. Our approach addresses these challenges by transforming annotated news headlines into structured causal knowledge graphs. It then employs a hybrid retrieval system that merges semantic embeddings with graph-based structural cues leveraging Neo4j to accurately match and retrieve relevant events. The framework is built on a three-stage pipeline: First, during Data Preparation, news sentences are meticulously annotated and converted into causal graphs capturing cause, effect, and trigger relationships. Next, the Graph Retrieval stage stores these graphs along with their embeddings in a Neo4j database and utilizes hybrid Cypher queries to efficiently identify events that share both semantic and structural similarities with a given query. Finally, the LLM Inference stage utilizes these retrieved causal graphs in a few-shot learning setup with XML-based prompting, enabling robust classification and tagging of causal relationships. Experimental evaluations demonstrate that GraphRAG-Causal achieves an impressive F1-score of 82.1% on causal classification using just 20 few-shot examples. This approach significantly boosts accuracy and consistency, making it highly suitable for real-time applications in news reliability assessment, misinformation detection, and policy analysis. 

---
# Identifying Helpful Context for LLM-based Vulnerability Repair: A Preliminary Study 

**Authors**: Gábor Antal, Bence Bogenfürst, Rudolf Ferenc, Péter Hegedűs  

**Link**: [PDF](https://arxiv.org/pdf/2506.11561)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise for automated vulnerability detection and repair in software systems. This paper investigates the performance of GPT-4o in repairing Java vulnerabilities from a widely used dataset (Vul4J), exploring how different contextual information affects automated vulnerability repair (AVR) capabilities. We compare the latest GPT-4o's performance against previous results with GPT-4 using identical prompts. We evaluated nine additional prompts crafted by us that contain various contextual information such as CWE or CVE information, and manually extracted code contexts. Each prompt was executed three times on 42 vulnerabilities, and the resulting fix candidates were validated using Vul4J's automated testing framework.
Our results show that GPT-4o performed 11.9\% worse on average than GPT-4 with the same prompt, but was able to fix 10.5\% more distinct vulnerabilities in the three runs together. CVE information significantly improved repair rates, while the length of the task description had minimal impact. Combining CVE guidance with manually extracted code context resulted in the best performance. Using our \textsc{Top}-3 prompts together, GPT-4o repaired 26 (62\%) vulnerabilities at least once, outperforming both the original baseline (40\%) and its reproduction (45\%), suggesting that ensemble prompt strategies could improve vulnerability repair in zero-shot settings. 

---
# Are LLMs Good Text Diacritizers? An Arabic and Yorùbá Case Study 

**Authors**: Hawau Olamide Toyin, Samar M. Magdy, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.11602)  

**Abstract**: We investigate the effectiveness of large language models (LLMs) for text diacritization in two typologically distinct languages: Arabic and Yoruba. To enable a rigorous evaluation, we introduce a novel multilingual dataset MultiDiac, with diverse samples that capture a range of diacritic ambiguities. We evaluate 14 LLMs varying in size, accessibility, and language coverage, and benchmark them against 6 specialized diacritization models. Additionally, we fine-tune four small open-source models using LoRA for Yoruba. Our results show that many off-the-shelf LLMs outperform specialized diacritization models for both Arabic and Yoruba, but smaller models suffer from hallucinations. Fine-tuning on a small dataset can help improve diacritization performance and reduce hallucination rates. 

---
# Leveraging GPT-4 for Vulnerability-Witnessing Unit Test Generation 

**Authors**: Gábor Antal, Dénes Bán, Martin Isztin, Rudolf Ferenc, Péter Hegedűs  

**Link**: [PDF](https://arxiv.org/pdf/2506.11559)  

**Abstract**: In the life-cycle of software development, testing plays a crucial role in quality assurance. Proper testing not only increases code coverage and prevents regressions but it can also ensure that any potential vulnerabilities in the software are identified and effectively fixed. However, creating such tests is a complex, resource-consuming manual process. To help developers and security experts, this paper explores the automatic unit test generation capability of one of the most widely used large language models, GPT-4, from the perspective of vulnerabilities. We examine a subset of the VUL4J dataset containing real vulnerabilities and their corresponding fixes to determine whether GPT-4 can generate syntactically and/or semantically correct unit tests based on the code before and after the fixes as evidence of vulnerability mitigation. We focus on the impact of code contexts, the effectiveness of GPT-4's self-correction ability, and the subjective usability of the generated test cases. Our results indicate that GPT-4 can generate syntactically correct test cases 66.5\% of the time without domain-specific pre-training. Although the semantic correctness of the fixes could be automatically validated in only 7. 5\% of the cases, our subjective evaluation shows that GPT-4 generally produces test templates that can be further developed into fully functional vulnerability-witnessing tests with relatively minimal manual effort.
Therefore, despite the limited data, our initial findings suggest that GPT-4 can be effectively used in the generation of vulnerability-witnessing tests. It may not operate entirely autonomously, but it certainly plays a significant role in a partially automated process. 

---
# LearnAlign: Reasoning Data Selection for Reinforcement Learning in Large Language Models Based on Improved Gradient Alignment 

**Authors**: Shikun Li, Shipeng Li, Zhiqin Yang, Xinghua Zhang, Gaode Chen, Xiaobo Xia, Hengyu Liu, Zhe Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11480)  

**Abstract**: Reinforcement learning (RL) has become a key technique for enhancing LLMs' reasoning abilities, yet its data inefficiency remains a major bottleneck. To address this critical yet challenging issue, we present a novel gradient-alignment-based method, named LearnAlign, which intelligently selects the learnable and representative training reasoning data for RL post-training. To overcome the well-known issue of response-length bias in gradient norms, we introduce the data learnability based on the success rate, which can indicate the learning potential of each data point. Experiments across three mathematical reasoning benchmarks demonstrate that our method significantly reduces training data requirements while achieving minor performance degradation or even improving performance compared to full-data training. For example, it reduces data requirements by up to 1,000 data points with better performance (77.53%) than that on the full dataset on GSM8K benchmark (77.04%). Furthermore, we show its effectiveness in the staged RL setting. This work provides valuable insights into data-efficient RL post-training and establishes a foundation for future research in optimizing reasoning data this http URL facilitate future work, we will release code. 

---
# DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs 

**Authors**: Bo-Cheng Chiu, Jen-Jee Chen, Yu-Chee Tseng, Feng-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11558)  

**Abstract**: Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling. 

---
# Stop learning it all to mitigate visual hallucination, Focus on the hallucination target 

**Authors**: Dokyoon Yoon, Youngsook Song, Woomyong Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.11417)  

**Abstract**: Multimodal Large Language Models (MLLMs) frequently suffer from hallucination issues, generating information about objects that are not present in input images during vision-language tasks. These hallucinations particularly undermine model reliability in practical applications requiring accurate object identification. To address this challenge, we propose \mymethod,\ a preference learning approach that mitigates hallucinations by focusing on targeted areas where they occur. To implement this, we build a dataset containing hallucinated responses, correct responses, and target information (i.e., objects present in the images and the corresponding chunk positions in responses affected by hallucinations). By applying a preference learning method restricted to these specific targets, the model can filter out irrelevant signals and focus on correcting hallucinations. This allows the model to produce more factual responses by concentrating solely on relevant information. Experimental results demonstrate that \mymethod\ effectively reduces hallucinations across multiple vision hallucination tasks, improving the reliability and performance of MLLMs without diminishing overall performance. 

---
# LoRA Users Beware: A Few Spurious Tokens Can Manipulate Your Finetuned Model 

**Authors**: Pradyut Sekhsaria, Marcel Mateos Salles, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2506.11402)  

**Abstract**: Parameter Efficient FineTuning (PEFT), such as Low-Rank Adaptation (LoRA), aligns pre-trained Large Language Models (LLMs) to particular downstream tasks in a resource-efficient manner. Because efficiency has been the main metric of progress, very little attention has been put in understanding possible catastrophic failures. We uncover one such failure: PEFT encourages a model to search for shortcut solutions to solve its fine-tuning tasks. When very small amount of tokens, e.g., one token per prompt, are correlated with downstream task classes, PEFT makes any pretrained model rely predominantly on that token for decision making. While such spurious tokens may emerge accidentally from incorrect data cleaning, it also opens opportunities for malevolent parties to control a model's behavior from Seamless Spurious Token Injection (SSTI). In SSTI, a small amount of tokens correlated with downstream classes are injected by the dataset creators. At test time, the finetuned LLM's behavior can be controlled solely by injecting those few tokens. We apply SSTI across models from three families (Snowflake Arctic, Apple OpenELM, and Meta LLaMA-3) and four diverse datasets (IMDB, Financial Classification, CommonSense QA, and Bias in Bios). Our findings reveal three astonishing behaviors. First, as few as a single token of SSTI is sufficient to steer a model's decision making. Second, for light SSTI, the reliance on spurious tokens is proportional to the LoRA rank. Lastly, with aggressive SSTI, larger LoRA rank values become preferable to small rank values as it makes the model attend to non-spurious tokens, hence improving robustness. 

---
# No Universal Prompt: Unifying Reasoning through Adaptive Prompting for Temporal Table Reasoning 

**Authors**: Kushagra Dixit, Abhishek Rajgaria, Harshavardhan Kalalbandi, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.11246)  

**Abstract**: Temporal Table Reasoning is a critical challenge for Large Language Models (LLMs), requiring effective prompting techniques to extract relevant insights. Despite existence of multiple prompting methods, their impact on table reasoning remains largely unexplored. Furthermore, the performance of these models varies drastically across different table and context structures, making it difficult to determine an optimal approach. This work investigates multiple prompting technique across diverse table types to determine optimal approaches for different scenarios. We find that performance varies based on entity type, table structure, requirement of additional context and question complexity, with NO single method consistently outperforming others. To mitigate these challenges, we introduce SEAR, an adaptive prompting framework inspired by human reasoning that dynamically adjusts based on context characteristics and integrates a structured reasoning. Our results demonstrate that SEAR achieves superior performance across all table types compared to other baseline prompting techniques. Additionally, we explore the impact of table structure refactoring, finding that a unified representation enhances model's reasoning. 

---
# Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation 

**Authors**: Shizhe Chen, Ricardo Garcia, Paul Pacaud, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2506.11261)  

**Abstract**: Robotic manipulation faces a significant challenge in generalizing across unseen objects, environments and tasks specified by diverse language instructions. To improve generalization capabilities, recent research has incorporated large language models (LLMs) for planning and action execution. While promising, these methods often fall short in generating grounded plans in visual environments. Although efforts have been made to perform visual instructional tuning on LLMs for robotic manipulation, existing methods are typically constrained by single-view image input and struggle with precise object grounding. In this work, we introduce Gondola, a novel grounded vision-language planning model based on LLMs for generalizable robotic manipulation. Gondola takes multi-view images and history plans to produce the next action plan with interleaved texts and segmentation masks of target objects and locations. To support the training of Gondola, we construct three types of datasets using the RLBench simulator, namely robot grounded planning, multi-view referring expression and pseudo long-horizon task datasets. Gondola outperforms the state-of-the-art LLM-based method across all four generalization levels of the GemBench dataset, including novel placements, rigid objects, articulated objects and long-horizon tasks. 

---
# Autonomous Computer Vision Development with Agentic AI 

**Authors**: Jin Kim, Muhammad Wahi-Anwa, Sangyun Park, Shawn Shin, John M. Hoffman, Matthew S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2506.11140)  

**Abstract**: Agentic Artificial Intelligence (AI) systems leveraging Large Language Models (LLMs) exhibit significant potential for complex reasoning, planning, and tool utilization. We demonstrate that a specialized computer vision system can be built autonomously from a natural language prompt using Agentic AI methods. This involved extending SimpleMind (SM), an open-source Cognitive AI environment with configurable tools for medical image analysis, with an LLM-based agent, implemented using OpenManus, to automate the planning (tool configuration) for a particular computer vision task. We provide a proof-of-concept demonstration that an agentic system can interpret a computer vision task prompt, plan a corresponding SimpleMind workflow by decomposing the task and configuring appropriate tools. From the user input prompt, "provide sm (SimpleMind) config for lungs, heart, and ribs segmentation for cxr (chest x-ray)"), the agent LLM was able to generate the plan (tool configuration file in YAML format), and execute SM-Learn (training) and SM-Think (inference) scripts autonomously. The computer vision agent automatically configured, trained, and tested itself on 50 chest x-ray images, achieving mean dice scores of 0.96, 0.82, 0.83, for lungs, heart, and ribs, respectively. This work shows the potential for autonomous planning and tool configuration that has traditionally been performed by a data scientist in the development of computer vision applications. 

---
# Large Language Models and Emergence: A Complex Systems Perspective 

**Authors**: David C. Krakauer, John W. Krakauer, Melanie Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2506.11135)  

**Abstract**: Emergence is a concept in complexity science that describes how many-body systems manifest novel higher-level properties, properties that can be described by replacing high-dimensional mechanisms with lower-dimensional effective variables and theories. This is captured by the idea "more is different". Intelligence is a consummate emergent property manifesting increasingly efficient -- cheaper and faster -- uses of emergent capabilities to solve problems. This is captured by the idea "less is more". In this paper, we first examine claims that Large Language Models exhibit emergent capabilities, reviewing several approaches to quantifying emergence, and secondly ask whether LLMs possess emergent intelligence. 

---
# Trustworthy AI for Medicine: Continuous Hallucination Detection and Elimination with CHECK 

**Authors**: Carlos Garcia-Fernandez, Luis Felipe, Monique Shotande, Muntasir Zitu, Aakash Tripathi, Ghulam Rasool, Issam El Naqa, Vivek Rudrapatna, Gilmer Valdes  

**Link**: [PDF](https://arxiv.org/pdf/2506.11129)  

**Abstract**: Large language models (LLMs) show promise in healthcare, but hallucinations remain a major barrier to clinical use. We present CHECK, a continuous-learning framework that integrates structured clinical databases with a classifier grounded in information theory to detect both factual and reasoning-based hallucinations. Evaluated on 1500 questions from 100 pivotal clinical trials, CHECK reduced LLama3.3-70B-Instruct hallucination rates from 31% to 0.3% - making an open source model state of the art. Its classifier generalized across medical benchmarks, achieving AUCs of 0.95-0.96, including on the MedQA (USMLE) benchmark and HealthBench realistic multi-turn medical questioning. By leveraging hallucination probabilities to guide GPT-4o's refinement and judiciously escalate compute, CHECK boosted its USMLE passing rate by 5 percentage points, achieving a state-of-the-art 92.1%. By suppressing hallucinations below accepted clinical error thresholds, CHECK offers a scalable foundation for safe LLM deployment in medicine and other high-stakes domains. 

---
# Test-Time-Scaling for Zero-Shot Diagnosis with Visual-Language Reasoning 

**Authors**: Ji Young Byun, Young-Jin Park, Navid Azizan, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2506.11166)  

**Abstract**: As a cornerstone of patient care, clinical decision-making significantly influences patient outcomes and can be enhanced by large language models (LLMs). Although LLMs have demonstrated remarkable performance, their application to visual question answering in medical imaging, particularly for reasoning-based diagnosis, remains largely unexplored. Furthermore, supervised fine-tuning for reasoning tasks is largely impractical due to limited data availability and high annotation costs. In this work, we introduce a zero-shot framework for reliable medical image diagnosis that enhances the reasoning capabilities of LLMs in clinical settings through test-time scaling. Given a medical image and a textual prompt, a vision-language model processes a medical image along with a corresponding textual prompt to generate multiple descriptions or interpretations of visual features. These interpretations are then fed to an LLM, where a test-time scaling strategy consolidates multiple candidate outputs into a reliable final diagnosis. We evaluate our approach across various medical imaging modalities -- including radiology, ophthalmology, and histopathology -- and demonstrate that the proposed test-time scaling strategy enhances diagnostic accuracy for both our and baseline methods. Additionally, we provide an empirical analysis showing that the proposed approach, which allows unbiased prompting in the first stage, improves the reliability of LLM-generated diagnoses and enhances classification accuracy. 

---
# Stronger Language Models Produce More Human-Like Errors 

**Authors**: Andrew Keenan Richardson, Ryan Othniel Kearns, Sean Moss, Vincent Wang-Mascianica, Philipp Koralus  

**Link**: [PDF](https://arxiv.org/pdf/2506.11128)  

**Abstract**: Do language models converge toward human-like reasoning patterns as they improve? We provide surprising evidence that while overall reasoning capabilities increase with model sophistication, the nature of errors increasingly mirrors predictable human reasoning fallacies: a previously unobserved inverse scaling phenomenon. To investigate this question, we apply the Erotetic Theory of Reasoning (ETR), a formal cognitive framework with empirical support for predicting human reasoning outcomes. Using the open-source package PyETR, we generate logical reasoning problems where humans predictably err, evaluating responses from 38 language models across 383 reasoning tasks. Our analysis indicates that as models advance in general capability (as measured by Chatbot Arena scores), the proportion of their incorrect answers that align with ETR-predicted human fallacies tends to increase ($\rho = 0.360, p = 0.0265$). Notably, as we observe no correlation between model sophistication and logical correctness on these tasks, this shift in error patterns toward human-likeness occurs independently of error rate. These findings challenge the prevailing view that scaling language models naturally obtains normative rationality, suggesting instead a convergence toward human-like cognition inclusive of our characteristic biases and limitations, as we further confirm by demonstrating order-effects in language model reasoning. 

---
# Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models 

**Authors**: Jijie Li, Li Du, Hanyu Zhao, Bo-wen Zhang, Liangdong Wang, Boyan Gao, Guang Liu, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11116)  

**Abstract**: Large Language Models (LLMs) demonstrate strong performance in real-world applications, yet existing open-source instruction datasets often concentrate on narrow domains, such as mathematics or coding, limiting generalization and widening the gap with proprietary models. To bridge this gap, we introduce Infinity-Instruct, a high-quality instruction dataset designed to enhance both foundational and chat capabilities of LLMs through a two-phase pipeline. In Phase 1, we curate 7.4M high-quality foundational instructions (InfInstruct-F-7.4M) from over 100M samples using hybrid data selection techniques. In Phase 2, we synthesize 1.5M high-quality chat instructions (InfInstruct-G-1.5M) through a two-stage process involving instruction selection, evolution, and diagnostic filtering. We empirically evaluate Infinity-Instruct by fine-tuning several open-source models, including Mistral, LLaMA, Qwen, and Yi, and observe substantial performance gains across both foundational and instruction following benchmarks, consistently surpassing official instruction-tuned counterparts. Notably, InfInstruct-LLaMA3.1-70B outperforms GPT-4-0314 by 8.6\% on instruction following tasks while achieving comparable foundational performance. These results underscore the synergy between foundational and chat training and offer new insights into holistic LLM development. Our dataset\footnote{this https URL} and codes\footnote{this https URL} have been publicly released. 

---
# ASRJam: Human-Friendly AI Speech Jamming to Prevent Automated Phone Scams 

**Authors**: Freddie Grabovski, Gilad Gressel, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.11125)  

**Abstract**: Large Language Models (LLMs), combined with Text-to-Speech (TTS) and Automatic Speech Recognition (ASR), are increasingly used to automate voice phishing (vishing) scams. These systems are scalable and convincing, posing a significant security threat. We identify the ASR transcription step as the most vulnerable link in the scam pipeline and introduce ASRJam, a proactive defence framework that injects adversarial perturbations into the victim's audio to disrupt the attacker's ASR. This breaks the scam's feedback loop without affecting human callers, who can still understand the conversation. While prior adversarial audio techniques are often unpleasant and impractical for real-time use, we also propose EchoGuard, a novel jammer that leverages natural distortions, such as reverberation and echo, that are disruptive to ASR but tolerable to humans. To evaluate EchoGuard's effectiveness and usability, we conducted a 39-person user study comparing it with three state-of-the-art attacks. Results show that EchoGuard achieved the highest overall utility, offering the best combination of ASR disruption and human listening experience. 

---
# Invocable APIs derived from NL2SQL datasets for LLM Tool-Calling Evaluation 

**Authors**: Benjamin Elder, Anupama Murthi, Jungkoo Kang, Ankita Rajaram Naik, Kiran Kate, Kinjal Basu, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2506.11266)  

**Abstract**: Large language models (LLMs) are routinely deployed as agentic systems, with access to tools that interact with live environments to accomplish tasks. In enterprise deployments these systems need to interact with API collections that can be extremely large and complex, often backed by databases. In order to create datasets with such characteristics, we explore how existing NL2SQL (Natural Language to SQL query) datasets can be used to automatically create NL2API datasets. Specifically, this work describes a novel data generation pipeline that exploits the syntax of SQL queries to construct a functionally equivalent sequence of API calls. We apply this pipeline to one of the largest NL2SQL datasets, BIRD-SQL to create a collection of over 2500 APIs that can be served as invocable tools or REST-endpoints. We pair natural language queries from BIRD-SQL to ground-truth API sequences based on this API pool. We use this collection to study the performance of 10 public LLMs and find that all models struggle to determine the right set of tools (consisting of tasks of intent detection, sequencing with nested function calls, and slot-filling). We find that models have extremely low task completion rates (7-47 percent - depending on the dataset) which marginally improves to 50 percent when models are employed as ReACT agents that interact with the live API environment. The best task completion rates are far below what may be required for effective general-use tool-calling agents, suggesting substantial scope for improvement in current state-of-the-art tool-calling LLMs. We also conduct detailed ablation studies, such as assessing the impact of the number of tools available as well as the impact of tool and slot-name obfuscation. We compare the performance of models on the original SQL generation tasks and find that current models are sometimes able to exploit SQL better than APIs. 

---
# KokushiMD-10: Benchmark for Evaluating Large Language Models on Ten Japanese National Healthcare Licensing Examinations 

**Authors**: Junyu Liu, Kaiqi Yan, Tianyang Wang, Qian Niu, Momoko Nagai-Tanima, Tomoki Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.11114)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated notable performance in medical licensing exams. However, comprehensive evaluation of LLMs across various healthcare roles, particularly in high-stakes clinical scenarios, remains a challenge. Existing benchmarks are typically text-based, English-centric, and focus primarily on medicines, which limits their ability to assess broader healthcare knowledge and multimodal reasoning. To address these gaps, we introduce KokushiMD-10, the first multimodal benchmark constructed from ten Japanese national healthcare licensing exams. This benchmark spans multiple fields, including Medicine, Dentistry, Nursing, Pharmacy, and allied health professions. It contains over 11588 real exam questions, incorporating clinical images and expert-annotated rationales to evaluate both textual and visual reasoning. We benchmark over 30 state-of-the-art LLMs, including GPT-4o, Claude 3.5, and Gemini, across both text and image-based settings. Despite promising results, no model consistently meets passing thresholds across domains, highlighting the ongoing challenges in medical AI. KokushiMD-10 provides a comprehensive and linguistically grounded resource for evaluating and advancing reasoning-centric medical AI across multilingual and multimodal clinical tasks. 

---
# Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization 

**Authors**: Yile Chen, Yicheng Tao, Yue Jiang, Shuai Liu, Han Yu, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11109)  

**Abstract**: The widespread adoption of location-based services has led to the generation of vast amounts of mobility data, providing significant opportunities to model user movement dynamics within urban environments. Recent advancements have focused on adapting Large Language Models (LLMs) for mobility analytics. However, existing methods face two primary limitations: inadequate semantic representation of locations (i.e., discrete IDs) and insufficient modeling of mobility signals within LLMs (i.e., single templated instruction fine-tuning). To address these issues, we propose QT-Mob, a novel framework that significantly enhances LLMs for mobility analytics. QT-Mob introduces a location tokenization module that learns compact, semantically rich tokens to represent locations, preserving contextual information while ensuring compatibility with LLMs. Furthermore, QT-Mob incorporates a series of complementary fine-tuning objectives that align the learned tokens with the internal representations in LLMs, improving the model's comprehension of sequential movement patterns and location semantics. The proposed QT-Mob framework not only enhances LLMs' ability to interpret mobility data but also provides a more generalizable approach for various mobility analytics tasks. Experiments on three real-world dataset demonstrate the superior performance in both next-location prediction and mobility recovery tasks, outperforming existing deep learning and LLM-based methods. 

---
# ScIRGen: Synthesize Realistic and Large-Scale RAG Dataset for Scientific Research 

**Authors**: Junyong Lin, Lu Dai, Ruiqian Han, Yijie Sui, Ruilin Wang, Xingliang Sun, Qinglin Wu, Min Feng, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11117)  

**Abstract**: Scientific researchers need intensive information about datasets to effectively evaluate and develop theories and methodologies. The information needs regarding datasets are implicitly embedded in particular research tasks, rather than explicitly expressed in search queries. However, existing scientific retrieval and question-answering (QA) datasets typically address straightforward questions, which do not align with the distribution of real-world research inquiries. To bridge this gap, we developed ScIRGen, a dataset generation framework for scientific QA \& retrieval that more accurately reflects the information needs of professional science researchers, and uses it to create a large-scale scientific retrieval-augmented generation (RAG) dataset with realistic queries, datasets and papers. Technically, we designed a dataset-oriented information extraction method that leverages academic papers to augment the dataset representation. We then proposed a question generation framework by employing cognitive taxonomy to ensure the quality of synthesized questions. We also design a method to automatically filter synthetic answers based on the perplexity shift of LLMs, which is highly aligned with human judgment of answers' validity. Collectively, these methodologies culminated in the creation of the 61k QA dataset, ScIRGen-Geo. We benchmarked representative methods on the ScIRGen-Geo dataset for their question-answering and retrieval capabilities, finding out that current methods still suffer from reasoning from complex questions. This work advances the development of more sophisticated tools to support the intricate information needs of the scientific community. 

---
# Graph-based RAG Enhancement via Global Query Disambiguation and Dependency-Aware Reranking 

**Authors**: Ningyuan Li, Junrui Liu, Yi Shan, Minghui Huang, Tong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11106)  

**Abstract**: Contemporary graph-based retrieval-augmented generation (RAG) methods typically begin by extracting entities from user queries and then leverage pre-constructed knowledge graphs to retrieve related relationships and metadata. However, this pipeline's exclusive reliance on entity-level extraction can lead to the misinterpretation or omission of latent yet critical information and relations. As a result, retrieved content may be irrelevant or contradictory, and essential knowledge may be excluded, exacerbating hallucination risks and degrading the fidelity of generated responses. To address these limitations, we introduce PankRAG, a framework that combines a globally aware, hierarchical query-resolution strategy with a novel dependency-aware reranking mechanism. PankRAG first constructs a multi-level resolution path that captures both parallel and sequential interdependencies within a query, guiding large language models (LLMs) through structured reasoning. It then applies its dependency-aware reranker to exploit the dependency structure among resolved sub-questions, enriching and validating retrieval results for subsequent sub-questions. Empirical evaluations demonstrate that PankRAG consistently outperforms state-of-the-art approaches across multiple benchmarks, underscoring its robustness and generalizability. 

---
# DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration 

**Authors**: Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11104)  

**Abstract**: Long-context understanding is crucial for many NLP applications, yet transformers struggle with efficiency due to the quadratic complexity of self-attention. Sparse attention methods alleviate this cost but often impose static, predefined masks, failing to capture heterogeneous attention patterns. This results in suboptimal token interactions, limiting adaptability and retrieval accuracy in long-sequence tasks. This work introduces a dynamic sparse attention mechanism that assigns adaptive masks at the attention-map level, preserving heterogeneous patterns across layers and heads. Unlike existing approaches, our method eliminates the need for fine-tuning and predefined mask structures while maintaining computational efficiency. By learning context-aware attention structures, it achieves high alignment with full-attention models, ensuring minimal performance degradation while reducing memory and compute overhead. This approach provides a scalable alternative to full attention, enabling the practical deployment of large-scale Large Language Models (LLMs) without sacrificing retrieval performance. DAM is available at: this https URL. 

---
# You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Model 

**Authors**: Wenchong He, Liqian Peng, Zhe Jiang, Alex Go  

**Link**: [PDF](https://arxiv.org/pdf/2506.11103)  

**Abstract**: Large language models (LLMs) possess a remarkable ability to perform in-context learning (ICL), which enables them to handle multiple downstream tasks simultaneously without requiring task-specific fine-tuning. Recent studies have shown that even moderately sized LLMs, such as Mistral 7B, Gemma 7B and Llama-3 8B, can achieve ICL through few-shot in-context fine-tuning of all tasks at once. However, this approach still lags behind dedicated fine-tuning, where a separate model is trained for each individual task.
In this paper, we propose a novel approach, Many-Shot In-Context Fine-tuning (ManyICL), which significantly narrows this performance gap by extending the principles of ICL to a many-shot setting. To unlock the full potential of ManyICL and address the inherent inefficiency of processing long sequences with numerous in-context examples, we propose a novel training objective. Instead of solely predicting the final answer, our approach treats every answer within the context as a supervised training target. This effectively shifts the role of many-shot examples from prompts to targets for autoregressive learning. Through extensive experiments on diverse downstream tasks, including classification, summarization, question answering, natural language inference, and math, we demonstrate that ManyICL substantially outperforms zero/few-shot fine-tuning and approaches the performance of dedicated fine-tuning. Furthermore, ManyICL significantly mitigates catastrophic forgetting issues observed in zero/few-shot fine-tuning. The code will be made publicly available upon publication. 

---
# Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks 

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2506.11113)  

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication. 

---
# Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions 

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11111)  

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (this https URL) to support the community. 

---
# AssertBench: A Benchmark for Evaluating Self-Assertion in Large Language Models 

**Authors**: Jaeho Lee, Atharv Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.11110)  

**Abstract**: Recent benchmarks have probed factual consistency and rhetorical robustness in Large Language Models (LLMs). However, a knowledge gap exists regarding how directional framing of factually true statements influences model agreement, a common scenario for LLM users. AssertBench addresses this by sampling evidence-supported facts from FEVEROUS, a fact verification dataset. For each (evidence-backed) fact, we construct two framing prompts: one where the user claims the statement is factually correct, and another where the user claims it is incorrect. We then record the model's agreement and reasoning. The desired outcome is that the model asserts itself, maintaining consistent truth evaluation across both framings, rather than switching its evaluation to agree with the user. AssertBench isolates framing-induced variability from the model's underlying factual knowledge by stratifying results based on the model's accuracy on the same claims when presented neutrally. In doing so, this benchmark aims to measure an LLM's ability to "stick to its guns" when presented with contradictory user assertions about the same fact. The complete source code is available at this https URL. 

---
# Dynamic Context Tuning for Retrieval-Augmented Generation: Enhancing Multi-Turn Planning and Tool Adaptation 

**Authors**: Jubin Abhishek Soni, Amit Anand, Rajesh Kumar Pandey, Aniket Abhishek Soni  

**Link**: [PDF](https://arxiv.org/pdf/2506.11092)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly advanced large language models (LLMs) by grounding their outputs in external tools and knowledge sources. However, existing RAG systems are typically constrained to static, single-turn interactions with fixed toolsets, making them ill-suited for dynamic domains such as healthcare and smart homes, where user intent, available tools, and contextual factors evolve over time. We present Dynamic Context Tuning (DCT), a lightweight framework that extends RAG to support multi-turn dialogue and evolving tool environments without requiring retraining. DCT integrates an attention-based context cache to track relevant past information, LoRA-based retrieval to dynamically select domain-specific tools, and efficient context compression to maintain inputs within LLM context limits. Experiments on both synthetic and real-world benchmarks show that DCT improves plan accuracy by 14% and reduces hallucinations by 37%, while matching GPT-4 performance at significantly lower cost. Furthermore, DCT generalizes to previously unseen tools, enabling scalable and adaptable AI assistants across a wide range of dynamic environments. 

---
# Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM 

**Authors**: Jeena Prakash, Blessingh Kumar, Kadri Hacioglu, Bidisha Sharma, Sindhuja Gopalan, Malolan Chetlur, Shankar Venkatesan, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2506.11089)  

**Abstract**: Automatic speech recognition (ASR) models rely on high-quality transcribed data for effective training. Generating pseudo-labels for large unlabeled audio datasets often relies on complex pipelines that combine multiple ASR outputs through multi-stage processing, leading to error propagation, information loss and disjoint optimization. We propose a unified multi-ASR prompt-driven framework using postprocessing by either textual or speech-based large language models (LLMs), replacing voting or other arbitration logic for reconciling the ensemble outputs. We perform a comparative study of multiple architectures with and without LLMs, showing significant improvements in transcription accuracy compared to traditional methods. Furthermore, we use the pseudo-labels generated by the various approaches to train semi-supervised ASR models for different datasets, again showing improved performance with textual and speechLLM transcriptions compared to baselines. 

---
# Two Birds with One Stone: Improving Factuality and Faithfulness of LLMs via Dynamic Interactive Subspace Editing 

**Authors**: Pengbo Wang, Chaozhuo Li, Chenxu Wang, Liwen Zheng, Litian Zhang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11088)  

**Abstract**: LLMs have demonstrated unprecedented capabilities in natural language processing, yet their practical deployment remains hindered by persistent factuality and faithfulness hallucinations. While existing methods address these hallucination types independently, they inadvertently induce performance trade-offs, as interventions targeting one type often exacerbate the other. Through empirical and theoretical analysis of activation space dynamics in LLMs, we reveal that these hallucination categories share overlapping subspaces within neural representations, presenting an opportunity for concurrent mitigation. To harness this insight, we propose SPACE, a unified framework that jointly enhances factuality and faithfulness by editing shared activation subspaces. SPACE establishes a geometric foundation for shared subspace existence through dual-task feature modeling, then identifies and edits these subspaces via a hybrid probe strategy combining spectral clustering and attention head saliency scoring. Experimental results across multiple benchmark datasets demonstrate the superiority of our approach. 

---
# PRISM: A Transformer-based Language Model of Structured Clinical Event Data 

**Authors**: Lionel Levine, John Santerre, Alex S. Young, T. Barry Levine, Francis Campion, Majid Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.11082)  

**Abstract**: We introduce PRISM (Predictive Reasoning in Sequential Medicine), a transformer-based architecture designed to model the sequential progression of clinical decision-making processes. Unlike traditional approaches that rely on isolated diagnostic classification, PRISM frames clinical trajectories as tokenized sequences of events - including diagnostic tests, laboratory results, and diagnoses - and learns to predict the most probable next steps in the patient diagnostic journey. Leveraging a large custom clinical vocabulary and an autoregressive training objective, PRISM demonstrates the ability to capture complex dependencies across longitudinal patient timelines. Experimental results show substantial improvements over random baselines in next-token prediction tasks, with generated sequences reflecting realistic diagnostic pathways, laboratory result progressions, and clinician ordering behaviors. These findings highlight the feasibility of applying generative language modeling techniques to structured medical event data, enabling applications in clinical decision support, simulation, and education. PRISM establishes a foundation for future advancements in sequence-based healthcare modeling, bridging the gap between machine learning architectures and real-world diagnostic reasoning. 

---
# Enabling On-Device Medical AI Assistants via Input-Driven Saliency Adaptation 

**Authors**: Uttej Kallakurik, Edward Humes, Rithvik Jonna, Xiaomin Lin, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11105)  

**Abstract**: Large Language Models (LLMs) have significant impact on the healthcare scenarios but remain prohibitively large for deployment in real-time, resource-constrained environments such as edge devices. In this work, we introduce a novel medical assistant system, optimized through our general-purpose compression framework, which tailors Large Language Models (LLMs) for deployment in specialized domains. By measuring neuron saliency on domain-specific data, our method can aggressively prune irrelevant neurons, reducing model size while preserving performance. Following pruning, we apply post-training quantization to further reduce the memory footprint, and evaluate the compressed model across medical benchmarks including MedMCQA, MedQA, and PubMedQA. We also deploy the 50\% compressed Gemma and the 67\% compressed LLaMA3 models on Jetson Orin Nano (18.7W peak) and Raspberry Pi 5 (6.3W peak), achieving real-time, energy-efficient inference under hardware constraints. 

---
# Intelligibility of Text-to-Speech Systems for Mathematical Expressions 

**Authors**: Sujoy Roychowdhury, H. G. Ranjani, Sumit Soman, Nishtha Paul, Subhadip Bandyopadhyay, Siddhanth Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2506.11086)  

**Abstract**: There has been limited evaluation of advanced Text-to-Speech (TTS) models with Mathematical eXpressions (MX) as inputs. In this work, we design experiments to evaluate quality and intelligibility of five TTS models through listening and transcribing tests for various categories of MX. We use two Large Language Models (LLMs) to generate English pronunciation from LaTeX MX as TTS models cannot process LaTeX directly. We use Mean Opinion Score from user ratings and quantify intelligibility through transcription correctness using three metrics. We also compare listener preference of TTS outputs with respect to human expert rendition of same MX. Results establish that output of TTS models for MX is not necessarily intelligible, the gap in intelligibility varies across TTS models and MX category. For most categories, performance of TTS models is significantly worse than that of expert rendition. The effect of choice of LLM is limited. This establishes the need to improve TTS models for MX. 

---
# Code Researcher: Deep Research Agent for Large Systems Code and Commit History 

**Authors**: Ramneet Singh, Sathvik Joel, Abhav Mehrotra, Nalin Wadhwa, Ramakrishna B Bairi, Aditya Kanade, Nagarajan Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11060)  

**Abstract**: Large Language Model (LLM)-based coding agents have shown promising results on coding benchmarks, but their effectiveness on systems code remains underexplored. Due to the size and complexities of systems code, making changes to a systems codebase is a daunting task, even for humans. It requires researching about many pieces of context, derived from the large codebase and its massive commit history, before making changes. Inspired by the recent progress on deep research agents, we design the first deep research agent for code, called Code Researcher, and apply it to the problem of generating patches for mitigating crashes reported in systems code. Code Researcher performs multi-step reasoning about semantics, patterns, and commit history of code to gather sufficient context. The context is stored in a structured memory which is used for synthesizing a patch. We evaluate Code Researcher on kBenchSyz, a benchmark of Linux kernel crashes, and show that it significantly outperforms strong baselines, achieving a crash-resolution rate of 58%, compared to 37.5% by SWE-agent. On an average, Code Researcher explores 10 files in each trajectory whereas SWE-agent explores only 1.33 files, highlighting Code Researcher's ability to deeply explore the codebase. Through another experiment on an open-source multimedia software, we show the generalizability of Code Researcher. Our experiments highlight the importance of global context gathering and multi-faceted reasoning for large codebases. 

---
# Improving Child Speech Recognition and Reading Mistake Detection by Using Prompts 

**Authors**: Lingyun Gao, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2506.11079)  

**Abstract**: Automatic reading aloud evaluation can provide valuable support to teachers by enabling more efficient scoring of reading exercises. However, research on reading evaluation systems and applications remains limited. We present a novel multimodal approach that leverages audio and knowledge from text resources. In particular, we explored the potential of using Whisper and instruction-tuned large language models (LLMs) with prompts to improve transcriptions for child speech recognition, as well as their effectiveness in downstream reading mistake detection. Our results demonstrate the effectiveness of prompting Whisper and prompting LLM, compared to the baseline Whisper model without prompting. The best performing system achieved state-of-the-art recognition performance in Dutch child read speech, with a word error rate (WER) of 5.1%, improving the baseline WER of 9.4%. Furthermore, it significantly improved reading mistake detection, increasing the F1 score from 0.39 to 0.73. 

---
# ADAMIX: Adaptive Mixed-Precision Delta-Compression with Quantization Error Optimization for Large Language Models 

**Authors**: Boya Xiong, Shuo Wang, Weifeng Ge, Guanhua Chen, Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11087)  

**Abstract**: Large language models (LLMs) achieve impressive performance on various knowledge-intensive and complex reasoning tasks in different domains. In certain scenarios like multi-tenant serving, a large number of LLMs finetuned from the same base model are deployed to meet complex requirements for users. Recent works explore delta-compression approaches to quantize and compress the delta parameters between the customized LLM and the corresponding base model. However, existing works either exhibit unsatisfactory performance at high compression ratios or depend on empirical bit allocation schemes. In this work, we propose ADAMIX, an effective adaptive mixed-precision delta-compression framework. We provide a mathematical derivation of quantization error to motivate our mixed-precision compression strategy and formulate the optimal mixed-precision bit allocation scheme as the solution to a 0/1 integer linear programming problem. Our derived bit allocation strategy minimizes the quantization error while adhering to a predefined compression ratio requirement. Experimental results on various models and benchmarks demonstrate that our approach surpasses the best baseline by a considerable margin. On tasks like AIME2024 and GQA, where the norm of $\Delta \mathbf{W}$ is large and the base model lacks sufficient ability, ADAMIX outperforms the best baseline Delta-CoMe by 22.3% and 6.1% with 7B models, respectively. 

---
# STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization 

**Authors**: Xijun Li, Jiexiang Yang, Jinghao Wang, Bo Peng, Jianguo Yao, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11057)  

**Abstract**: Combinatorial optimization (CO) problems, central to operation research and theoretical computer science, present significant computational challenges due to their NP-hard nature. While large language models (LLMs) have emerged as promising tools for CO--either by directly generating solutions or synthesizing solver-specific codes--existing approaches often neglect critical structural priors inherent to CO problems, leading to suboptimality and iterative inefficiency. Inspired by human experts' success in leveraging CO structures for algorithm design, we propose STRCMP, a novel structure-aware LLM-based algorithm discovery framework that systematically integrates structure priors to enhance solution quality and solving efficiency. Our framework combines a graph neural network (GNN) for extracting structural embeddings from CO instances with an LLM conditioned on these embeddings to identify high-performing algorithms in the form of solver-specific codes. This composite architecture ensures syntactic correctness, preserves problem topology, and aligns with natural language objectives, while an evolutionary refinement process iteratively optimizes generated algorithm. Extensive evaluations across Mixed Integer Linear Programming and Boolean Satisfiability problems, using nine benchmark datasets, demonstrate that our proposed STRCMP outperforms five strong neural and LLM-based methods by a large margin, in terms of both solution optimality and computational efficiency. The code and learned model will be publicly available upon the acceptance of the paper. 

---
# xInv: Explainable Optimization of Inverse Problems 

**Authors**: Sean Memery, Kevin Denamganai, Anna Kapron-King, Kartic Subr  

**Link**: [PDF](https://arxiv.org/pdf/2506.11056)  

**Abstract**: Inverse problems are central to a wide range of fields, including healthcare, climate science, and agriculture. They involve the estimation of inputs, typically via iterative optimization, to some known forward model so that it produces a desired outcome. Despite considerable development in the explainability and interpretability of forward models, the iterative optimization of inverse problems remains largely cryptic to domain experts. We propose a methodology to produce explanations, from traces produced by an optimizer, that are interpretable by humans at the abstraction of the domain. The central idea in our approach is to instrument a differentiable simulator so that it emits natural language events during its forward and backward passes. In a post-process, we use a Language Model to create an explanation from the list of events. We demonstrate the effectiveness of our approach with an illustrative optimization problem and an example involving the training of a neural network. 

---
# ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention 

**Authors**: Henrik Abgaryan, Tristan Cazenave, Ararat Harutyunyan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11052)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, yet their direct application to NP-hard combinatorial problems (CPs) remains underexplored. In this work, we systematically investigate the reasoning abilities of LLMs on a variety of NP-hard combinatorial optimization tasks and introduce ACCORD: Autoregressive Constraint-satisfying generation for COmbinatorial optimization with Routing and Dynamic attention. ACCORD features a novel dataset representation and model architecture that leverage the autoregressive nature of LLMs to dynamically enforce feasibility constraints, coupled with attention-based routing to activate problem-specific LoRA modules. We also present the ACCORD-90k supervised dataset, covering six NP-hard combinatorial problems: TSP, VRP, Knapsack, FlowShop, JSSP, and BinPacking. Extensive experiments demonstrate that our ACCORD model, built on an 8B-parameter Llama backbone, consistently outperforms standard prompting and input-output methods, even when compared to much larger LLMs, such as gpt-4. Ablation studies further show that our output structure enhances solution feasibility. To the best of our knowledge, this is the first large-scale, end-to-end framework for exploring the applications of LLMs to a broad spectrum of combinatorial optimization problems. The codes are publicly available at this https URL 

---
# Eliminating Hallucination-Induced Errors in LLM Code Generation with Functional Clustering 

**Authors**: Chaitanya Ravuri, Saman Amarasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2506.11021)  

**Abstract**: Modern code-generation LLMs can already solve a large fraction of programming problems, yet they still hallucinate subtle bugs that make their outputs unsafe for autonomous deployment. We present functional clustering, a black-box wrapper that eliminates nearly all hallucination-induced errors while providing a tunable confidence score. The wrapper samples many candidate programs, executes each on a self-generated test suite, and clusters candidates whose I/O behavior is identical; the empirical mass of the largest cluster serves as an exact confidence estimate. A single scalar threshold on this estimate lets users trade coverage for reliability with exponential guarantees. On LiveCodeBench our verifier preserves baseline pass@1 on solvable tasks yet slashes the error rate of returned answers from ~65% to 2%, and drives it to 0% at a conservative threshold while still answering 15.6% of prompts. Manual audits show that the few residual mistakes stem from prompt misinterpretation, not random generation noise, narrowing future work to specification clarity. Because the method requires only sampling and sandbox execution, it applies unchanged to closed-source APIs and future models, offering a practical path toward dependable, autonomous code generation. Our code is available on Github (this https URL). 

---
# Extracting Knowledge Graphs from User Stories using LangChain 

**Authors**: Thayná Camargo da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2506.11020)  

**Abstract**: This thesis introduces a novel methodology for the automated generation of knowledge graphs from user stories by leveraging the advanced capabilities of Large Language Models. Utilizing the LangChain framework as a basis, the User Story Graph Transformer module was developed to extract nodes and relationships from user stories using an LLM to construct accurate knowledge this http URL innovative technique was implemented in a script to fully automate the knowledge graph extraction process. Additionally, the evaluation was automated through a dedicated evaluation script, utilizing an annotated dataset for assessment. By enhancing the visualization and understanding of user requirements and domain concepts, this method fosters better alignment between software functionalities and user expectations, ultimately contributing to more effective and user-centric software development processes. 

---
# From Reasoning to Code: GRPO Optimization for Underrepresented Languages 

**Authors**: Federico Pennino, Bianca Raimondi, Massimo Rondelli, Andrea Gurioli, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2506.11027)  

**Abstract**: Generating accurate and executable code using large language models (LLMs) is challenging for languages with limited public training data compared to popular languages such as Python. This paper introduces a generalizable approach that uses small-scale code versions of the Qwen 2.5 model combined with Group Relative Policy Optimization (GRPO) to enable effective code generation through explicit reasoning steps, which is particularly beneficial for languages with smaller source code databases. Using Prolog as a representative use case -- given its limited online presence -- the initial model faced challenges in generating executable code. After some training steps, the model successfully produces logically consistent and syntactically accurate code by directly integrating reasoning-driven feedback into the reinforcement learning loop. Experimental evaluations using mathematical logic problem benchmarks illustrate significant improvements in reasoning quality, code accuracy, and logical correctness, underscoring the potential of this approach to benefit a wide range of programming languages lacking extensive training resources. 

---
# Towards Automated Formal Verification of Backend Systems with LLMs 

**Authors**: Kangping Xu, Yifan Luo, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10998)  

**Abstract**: Software testing plays a critical role in ensuring that systems behave as intended. However, existing automated testing approaches struggle to match the capabilities of human engineers due to key limitations such as test locality, lack of general reliability, and business logic blindness. In this work, we propose a novel framework that leverages functional programming and type systems to translate Scala backend code into formal Lean representations. Our pipeline automatically generates theorems that specify the intended behavior of APIs and database operations, and uses LLM-based provers to verify them. When a theorem is proved, the corresponding logic is guaranteed to be correct and no further testing is needed. If the negation of a theorem is proved instead, it confirms a bug. In cases where neither can be proved, human intervention is required. We evaluate our method on realistic backend systems and find that it can formally verify over 50% of the test requirements, which suggests that half of a testing engineer's workload can be automated. Additionally, with an average cost of only $2.19 per API, LLM-based verification is significantly more cost-effective than manual testing and can be scaled easily through parallel execution. Our results indicate a promising direction for scalable, AI-powered software testing, with the potential to greatly improve engineering productivity as models continue to advance. 

---
# Evaluating LLMs for Visualization Tasks 

**Authors**: Saadiq Rauf Khan, Vinit Chandak, Sougata Mukherjea  

**Link**: [PDF](https://arxiv.org/pdf/2506.10996)  

**Abstract**: Information Visualization has been utilized to gain insights from complex data. In recent times, Large Language Models (LLMs) have performed very well in many tasks. In this paper, we showcase the capabilities of different popular LLMs to generate code for visualization based on simple prompts. We also analyze the power of LLMs to understand some common visualizations by answering simple questions. Our study shows that LLMs could generate code for some visualizations as well as answer questions about them. However, LLMs also have several limitations. We believe that our insights can be used to improve both LLMs and Information Visualization systems. 

---
# Prompt engineering and framework: implementation to increase code reliability based guideline for LLMs 

**Authors**: Rogelio Cruz, Jonatan Contreras, Francisco Guerrero, Ezequiel Rodriguez, Carlos Valdez, Citlali Carrillo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10989)  

**Abstract**: In this paper, we propose a novel prompting approach aimed at enhancing the ability of Large Language Models (LLMs) to generate accurate Python code. Specifically, we introduce a prompt template designed to improve the quality and correctness of generated code snippets, enabling them to pass tests and produce reliable results. Through experiments conducted on two state-of-the-art LLMs using the HumanEval dataset, we demonstrate that our approach outperforms widely studied zero-shot and Chain-of-Thought (CoT) methods in terms of the Pass@k metric. Furthermore, our method achieves these improvements with significantly reduced token usage compared to the CoT approach, making it both effective and resource-efficient, thereby lowering the computational demands and improving the eco-footprint of LLM capabilities. These findings highlight the potential of tailored prompting strategies to optimize code generation performance, paving the way for broader applications in AI-driven programming tasks. 

---
# Application Modernization with LLMs: Addressing Core Challenges in Reliability, Security, and Quality 

**Authors**: Ahilan Ayyachamy Nadar Ponnusamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.10984)  

**Abstract**: AI-assisted code generation tools have revolutionized software development, offering unprecedented efficiency and scalability. However, multiple studies have consistently highlighted challenges such as security vulnerabilities, reliability issues, and inconsistencies in the generated code. Addressing these concerns is crucial to unlocking the full potential of this transformative technology. While advancements in foundational and code-specialized language models have made notable progress in mitigating some of these issues, significant gaps remain, particularly in ensuring high-quality, trustworthy outputs.
This paper builds upon existing research on leveraging large language models (LLMs) for application modernization. It explores an opinionated approach that emphasizes two core capabilities of LLMs: code reasoning and code generation. The proposed framework integrates these capabilities with human expertise to tackle application modernization challenges effectively. It highlights the indispensable role of human involvement and guidance in ensuring the success of AI-assisted processes.
To demonstrate the framework's utility, this paper presents a detailed case study, walking through its application in a real-world scenario. The analysis includes a step-by-step breakdown, assessing alternative approaches where applicable. This work aims to provide actionable insights and a robust foundation for future research in AI-driven application modernization. The reference implementation created for this paper is available on GitHub. 

---
# TeleEval-OS: Performance evaluations of large language models for operations scheduling 

**Authors**: Yanyan Wang, Yingying Wang, Junli Liang, Yin Xu, Yunlong Liu, Yiming Xu, Zhengwang Jiang, Zhehe Li, Fei Li, Long Zhao, Kuang Xu, Qi Song, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11017)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly propelled progress in artificial intelligence, demonstrating substantial application potential across multiple specialized domains. Telecommunications operation scheduling (OS) is a critical aspect of the telecommunications industry, involving the coordinated management of networks, services, risks, and human resources to optimize production scheduling and ensure unified service control. However, the inherent complexity and domain-specific nature of OS tasks, coupled with the absence of comprehensive evaluation benchmarks, have hindered thorough exploration of LLMs' application potential in this critical field. To address this research gap, we propose the first Telecommunications Operation Scheduling Evaluation Benchmark (TeleEval-OS). Specifically, this benchmark comprises 15 datasets across 13 subtasks, comprehensively simulating four key operational stages: intelligent ticket creation, intelligent ticket handling, intelligent ticket closure, and intelligent evaluation. To systematically assess the performance of LLMs on tasks of varying complexity, we categorize their capabilities in telecommunications operation scheduling into four hierarchical levels, arranged in ascending order of difficulty: basic NLP, knowledge Q&A, report generation, and report analysis. On TeleEval-OS, we leverage zero-shot and few-shot evaluation methods to comprehensively assess 10 open-source LLMs (e.g., DeepSeek-V3) and 4 closed-source LLMs (e.g., GPT-4o) across diverse scenarios. Experimental results demonstrate that open-source LLMs can outperform closed-source LLMs in specific scenarios, highlighting their significant potential and value in the field of telecommunications operation scheduling. 

---
# Digitization of Document and Information Extraction using OCR 

**Authors**: Rasha Sinha, Rekha B S  

**Link**: [PDF](https://arxiv.org/pdf/2506.11156)  

**Abstract**: Retrieving accurate details from documents is a crucial task, especially when handling a combination of scanned images and native digital formats. This document presents a combined framework for text extraction that merges Optical Character Recognition (OCR) techniques with Large Language Models (LLMs) to deliver structured outputs enriched by contextual understanding and confidence indicators. Scanned files are processed using OCR engines, while digital files are interpreted through layout-aware libraries. The extracted raw text is subsequently analyzed by an LLM to identify key-value pairs and resolve ambiguities. A comparative analysis of different OCR tools is presented to evaluate their effectiveness concerning accuracy, layout recognition, and processing speed. The approach demonstrates significant improvements over traditional rule-based and template-based methods, offering enhanced flexibility and semantic precision across different document categories 

---
# Leveraging Reference Documents for Zero-Shot Ranking via Large Language Models 

**Authors**: Jieran Li, Xiuyuan Hu, Yang Zhao, Shengyao Zhuang, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11452)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance in the task of text ranking for information retrieval. While Pointwise ranking approaches offer computational efficiency by scoring documents independently, they often yield biased relevance estimates due to the lack of inter-document comparisons. In contrast, Pairwise methods improve ranking accuracy by explicitly comparing document pairs, but suffer from substantial computational overhead with quadratic complexity ($O(n^2)$). To address this tradeoff, we propose \textbf{RefRank}, a simple and effective comparative ranking method based on a fixed reference document. Instead of comparing all document pairs, RefRank prompts the LLM to evaluate each candidate relative to a shared reference anchor. By selecting the reference anchor that encapsulates the core query intent, RefRank implicitly captures relevance cues, enabling indirect comparison between documents via this common anchor. This reduces computational cost to linear time ($O(n)$) while importantly, preserving the advantages of comparative evaluation. To further enhance robustness, we aggregate multiple RefRank outputs using a weighted averaging scheme across different reference choices. Experiments on several benchmark datasets and with various LLMs show that RefRank significantly outperforms Pointwise baselines and could achieve performance at least on par with Pairwise approaches with a significantly lower computational cost. 

---
# TongSearch-QR: Reinforced Query Reasoning for Retrieval 

**Authors**: Xubo Qin, Jun Bai, Jiaqi Li, Zixia Jia, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11603)  

**Abstract**: Traditional information retrieval (IR) methods excel at textual and semantic matching but struggle in reasoning-intensive retrieval tasks that require multi-hop inference or complex semantic understanding between queries and documents. One promising solution is to explicitly rewrite or augment queries using large language models (LLMs) to elicit reasoning-relevant content prior to retrieval. However, the widespread use of large-scale language models like GPT-4 or LLaMA3-70B remains impractical due to their high inference cost and limited deployability in real-world systems. In this work, we introduce TongSearch QR (Previously Known as "TongSearch Reasoner"), a family of small-scale language models for query reasoning and rewriting in reasoning-intensive retrieval. With a novel semi-rule-based reward function, we employ reinforcement learning approaches enabling smaller language models, e,g, Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct, to achieve query reasoning performance rivaling large-scale language models without their prohibitive inference costs. Experiment results on BRIGHT benchmark show that with BM25 as retrievers, both TongSearch QR-7B and TongSearch QR-1.5B models significantly outperform existing baselines, including prompt-based query reasoners and some latest dense retrievers trained for reasoning-intensive retrieval tasks, offering superior adaptability for real-world deployment. 

---
# Post Persona Alignment for Multi-Session Dialogue Generation 

**Authors**: Yi-Pei Chen, Noriki Nishida, Hideki Nakayama, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.11857)  

**Abstract**: Multi-session persona-based dialogue generation presents challenges in maintaining long-term consistency and generating diverse, personalized responses. While large language models (LLMs) excel in single-session dialogues, they struggle to preserve persona fidelity and conversational coherence across extended interactions. Existing methods typically retrieve persona information before response generation, which can constrain diversity and result in generic outputs. We propose Post Persona Alignment (PPA), a novel two-stage framework that reverses this process. PPA first generates a general response based solely on dialogue context, then retrieves relevant persona memories using the response as a query, and finally refines the response to align with the speaker's persona. This post-hoc alignment strategy promotes naturalness and diversity while preserving consistency and personalization. Experiments on multi-session LLM-generated dialogue data demonstrate that PPA significantly outperforms prior approaches in consistency, diversity, and persona relevance, offering a more flexible and effective paradigm for long-term personalized dialogue generation. 

---
# Long-Short Alignment for Effective Long-Context Modeling in LLMs 

**Authors**: Tianqi Du, Haotian Huang, Yifei Wang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11769)  

**Abstract**: Large language models (LLMs) have exhibited impressive performance and surprising emergent properties. However, their effectiveness remains limited by the fixed context window of the transformer architecture, posing challenges for long-context modeling. Among these challenges, length generalization -- the ability to generalize to sequences longer than those seen during training -- is a classical and fundamental problem. In this work, we propose a fresh perspective on length generalization, shifting the focus from the conventional emphasis on input features such as positional encodings or data structures to the output distribution of the model. Specifically, through case studies on synthetic tasks, we highlight the critical role of \textbf{long-short alignment} -- the consistency of output distributions across sequences of varying lengths. Extending this insight to natural language tasks, we propose a metric called Long-Short Misalignment to quantify this phenomenon, uncovering a strong correlation between the metric and length generalization performance. Building on these findings, we develop a regularization term that promotes long-short alignment during training. Extensive experiments validate the effectiveness of our approach, offering new insights for achieving more effective long-context modeling in LLMs. Code is available at this https URL. 

---
# LLMs for Sentence Simplification: A Hybrid Multi-Agent prompting Approach 

**Authors**: Pratibha Zunjare, Michael Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11681)  

**Abstract**: This paper addresses the challenge of transforming complex sentences into sequences of logical, simplified sentences while preserving semantic and logical integrity with the help of Large Language Models. We propose a hybrid approach that combines advanced prompting with multi-agent architectures to enhance the sentence simplification process. Experimental results show that our approach was able to successfully simplify 70% of the complex sentences written for video game design application. In comparison, a single-agent approach attained a 48% success rate on the same task. 

---
# DART: Distilling Autoregressive Reasoning to Silent Thought 

**Authors**: Nan Jiang, Ziming Wu, De-Chuan Zhan, Fuming Lai, Shaobing Lian  

**Link**: [PDF](https://arxiv.org/pdf/2506.11752)  

**Abstract**: Chain-of-Thought (CoT) reasoning has significantly advanced Large Language Models (LLMs) in solving complex tasks. However, its autoregressive paradigm leads to significant computational overhead, hindering its deployment in latency-sensitive applications. To address this, we propose \textbf{DART} (\textbf{D}istilling \textbf{A}utoregressive \textbf{R}easoning to Silent \textbf{T}hought), a self-distillation framework that enables LLMs to replace autoregressive CoT with non-autoregressive Silent Thought (ST). Specifically, DART introduces two training pathways: the CoT pathway for traditional reasoning and the ST pathway for generating answers directly from a few ST tokens. The ST pathway utilizes a lightweight Reasoning Evolvement Module (REM) to align its hidden states with the CoT pathway, enabling the ST tokens to evolve into informative embeddings. During inference, only the ST pathway is activated, leveraging evolving ST tokens to deliver the answer directly. Extensive experimental results demonstrate that DART achieves comparable reasoning performance to existing baselines while offering significant efficiency gains, serving as a feasible alternative for efficient reasoning. 

---
# Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards 

**Authors**: Jaehoon Yun, Jiwoong Sohn, Jungwoo Park, Hyunjae Kim, Xiangru Tang, Yanjun Shao, Yonghoe Koo, Minhyeok Ko, Qingyu Chen, Mark Gerstein, Michael Moor, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11474)  

**Abstract**: Large language models have shown promise in clinical decision making, but current approaches struggle to localize and correct errors at specific steps of the reasoning process. This limitation is critical in medicine, where identifying and addressing reasoning errors is essential for accurate diagnosis and effective patient care. We introduce Med-PRM, a process reward modeling framework that leverages retrieval-augmented generation to verify each reasoning step against established medical knowledge bases. By verifying intermediate reasoning steps with evidence retrieved from clinical guidelines and literature, our model can precisely assess the reasoning quality in a fine-grained manner. Evaluations on five medical QA benchmarks and two open-ended diagnostic tasks demonstrate that Med-PRM achieves state-of-the-art performance, with improving the performance of base models by up to 13.50% using Med-PRM. Moreover, we demonstrate the generality of Med-PRM by integrating it in a plug-and-play fashion with strong policy models such as Meerkat, achieving over 80\% accuracy on MedQA for the first time using small-scale models of 8 billion parameters. Our code and data are available at: this https URL 

---
# SceneGram: Conceptualizing and Describing Tangrams in Scene Context 

**Authors**: Simeon Junker, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2506.11631)  

**Abstract**: Research on reference and naming suggests that humans can come up with very different ways of conceptualizing and referring to the same object, e.g. the same abstract tangram shape can be a "crab", "sink" or "space ship". Another common assumption in cognitive science is that scene context fundamentally shapes our visual perception of objects and conceptual expectations. This paper contributes SceneGram, a dataset of human references to tangram shapes placed in different scene contexts, allowing for systematic analyses of the effect of scene context on conceptualization. Based on this data, we analyze references to tangram shapes generated by multimodal LLMs, showing that these models do not account for the richness and variability of conceptualizations found in human references. 

---
# Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback 

**Authors**: Dongwei Jiang, Alvin Zhang, Andrew Wang, Nicholas Andrews, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11930)  

**Abstract**: Recent studies have shown LLMs possess some ability to improve their responses when given external feedback. However, it remains unclear how effectively and thoroughly these models can incorporate extrinsic feedback. In an ideal scenario, if LLMs receive near-perfect and complete feedback, we would expect them to fully integrate the feedback and change their incorrect answers to correct ones. In this paper, we systematically investigate LLMs' ability to incorporate feedback by designing a controlled experimental environment. For each problem, a solver model attempts a solution, then a feedback generator with access to near-complete ground-truth answers produces targeted feedback, after which the solver tries again. We evaluate this pipeline across a diverse range of tasks, including math reasoning, knowledge reasoning, scientific reasoning, and general multi-domain evaluations with state-of-the-art language models including Claude 3.7 (with and without extended thinking). Surprisingly, even under these near-ideal conditions, solver models consistently show resistance to feedback, a limitation that we term FEEDBACK FRICTION. To mitigate this limitation, we experiment with sampling-based strategies like progressive temperature increases and explicit rejection of previously attempted incorrect answers, which yield improvements but still fail to help models achieve target performance. We also perform a rigorous exploration of potential causes of FEEDBACK FRICTION, ruling out factors such as model overconfidence and data familiarity. We hope that highlighting this issue in LLMs and ruling out several apparent causes will help future research in self-improvement. 

---
# Efficient Long-Context LLM Inference via KV Cache Clustering 

**Authors**: Jie Hu, Shengnan Wang, Yutong He, Ping Gong, Jiawei Yi, Juncheng Zhang, Youhui Bai, Renhai Chen, Gong Zhang, Cheng Li, Kun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11418)  

**Abstract**: Large language models (LLMs) with extended context windows have become increasingly prevalent for tackling complex tasks. However, the substantial Key-Value (KV) cache required for long-context LLMs poses significant deployment challenges. Existing approaches either discard potentially critical information needed for future generations or offer limited efficiency gains due to high computational overhead. In this paper, we introduce Chelsea, a simple yet effective framework for online KV cache clustering. Our approach is based on the observation that key states exhibit high similarity along the sequence dimension. To enable efficient clustering, we divide the sequence into chunks and propose Chunked Soft Matching, which employs an alternating partition strategy within each chunk and identifies clusters based on similarity. Chelsea then merges the KV cache within each cluster into a single centroid. Additionally, we provide a theoretical analysis of the computational complexity and the optimality of the intra-chunk partitioning strategy. Extensive experiments across various models and long-context benchmarks demonstrate that Chelsea achieves up to 80% reduction in KV cache memory usage while maintaining comparable model performance. Moreover, with minimal computational overhead, Chelsea accelerates the decoding stage of inference by up to 3.19$\times$ and reduces end-to-end latency by up to 2.72$\times$. 

---
# Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards 

**Authors**: Jeff Da, Clinton Wang, Xiang Deng, Yuntao Ma, Nikhil Barhate, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.11425)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle. 

---
# The Biased Samaritan: LLM biases in Perceived Kindness 

**Authors**: Jack H Fagan, Ruhaan Juyaal, Amy Yue-Ming Yu, Siya Pun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11361)  

**Abstract**: While Large Language Models (LLMs) have become ubiquitous in many fields, understanding and mitigating LLM biases is an ongoing issue. This paper provides a novel method for evaluating the demographic biases of various generative AI models. By prompting models to assess a moral patient's willingness to intervene constructively, we aim to quantitatively evaluate different LLMs' biases towards various genders, races, and ages. Our work differs from existing work by aiming to determine the baseline demographic identities for various commercial models and the relationship between the baseline and other demographics. We strive to understand if these biases are positive, neutral, or negative, and the strength of these biases. This paper can contribute to the objective assessment of bias in Large Language Models and give the user or developer the power to account for these biases in LLM output or in training future LLMs. Our analysis suggested two key findings: that models view the baseline demographic as a white middle-aged or young adult male; however, a general trend across models suggested that non-baseline demographics are more willing to help than the baseline. These methodologies allowed us to distinguish these two biases that are often tangled together. 

---
# From Persona to Person: Enhancing the Naturalness with Multiple Discourse Relations Graph Learning in Personalized Dialogue Generation 

**Authors**: Chih-Hao Hsu, Ying-Jia Lin, Hung-Yu Kao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11557)  

**Abstract**: In dialogue generation, the naturalness of responses is crucial for effective human-machine interaction. Personalized response generation poses even greater challenges, as the responses must remain coherent and consistent with the user's personal traits or persona descriptions. We propose MUDI ($\textbf{Mu}$ltiple $\textbf{Di}$scourse Relations Graph Learning) for personalized dialogue generation. We utilize a Large Language Model to assist in annotating discourse relations and to transform dialogue data into structured dialogue graphs. Our graph encoder, the proposed DialogueGAT model, then captures implicit discourse relations within this structure, along with persona descriptions. During the personalized response generation phase, novel coherence-aware attention strategies are implemented to enhance the decoder's consideration of discourse relations. Our experiments demonstrate significant improvements in the quality of personalized responses, thus resembling human-like dialogue exchanges. 

---
# AbsenceBench: Language Models Can't Tell What's Missing 

**Authors**: Harvey Yiyun Fu, Aryan Shrivastava, Jared Moore, Peter West, Chenhao Tan, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2506.11440)  

**Abstract**: Large language models (LLMs) are increasingly capable of processing long inputs and locating specific information within them, as evidenced by their performance on the Needle in a Haystack (NIAH) test. However, while models excel at recalling surprising information, they still struggle to identify clearly omitted information. We introduce AbsenceBench to assesses LLMs' capacity to detect missing information across three domains: numerical sequences, poetry, and GitHub pull requests. AbsenceBench asks models to identify which pieces of a document were deliberately removed, given access to both the original and edited contexts. Despite the apparent straightforwardness of these tasks, our experiments reveal that even state-of-the-art models like Claude-3.7-Sonnet achieve only 69.6% F1-score with a modest average context length of 5K tokens. Our analysis suggests this poor performance stems from a fundamental limitation: Transformer attention mechanisms cannot easily attend to "gaps" in documents since these absences don't correspond to any specific keys that can be attended to. Overall, our results and analysis provide a case study of the close proximity of tasks where models are already superhuman (NIAH) and tasks where models breakdown unexpectedly (AbsenceBench). 

---
# Scalable Medication Extraction and Discontinuation Identification from Electronic Health Records Using Large Language Models 

**Authors**: Chong Shao, Douglas Snyder, Chiran Li, Bowen Gu, Kerry Ngan, Chun-Ting Yang, Jiageng Wu, Richard Wyss, Kueiyu Joshua Lin, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11137)  

**Abstract**: Identifying medication discontinuations in electronic health records (EHRs) is vital for patient safety but is often hindered by information being buried in unstructured notes. This study aims to evaluate the capabilities of advanced open-sourced and proprietary large language models (LLMs) in extracting medications and classifying their medication status from EHR notes, focusing on their scalability on medication information extraction without human annotation. We collected three EHR datasets from diverse sources to build the evaluation benchmark. We evaluated 12 advanced LLMs and explored multiple LLM prompting strategies. Performance on medication extraction, medication status classification, and their joint task (extraction then classification) was systematically compared across all experiments. We found that LLMs showed promising performance on the medication extraction and discontinuation classification from EHR notes. GPT-4o consistently achieved the highest average F1 scores in all tasks under zero-shot setting - 94.0% for medication extraction, 78.1% for discontinuation classification, and 72.7% for the joint task. Open-sourced models followed closely, Llama-3.1-70B-Instruct achieved the highest performance in medication status classification on the MIV-Med dataset (68.7%) and in the joint task on both the Re-CASI (76.2%) and MIV-Med (60.2%) datasets. Medical-specific LLMs demonstrated lower performance compared to advanced general-domain LLMs. Few-shot learning generally improved performance, while CoT reasoning showed inconsistent gains. LLMs demonstrate strong potential for medication extraction and discontinuation identification on EHR notes, with open-sourced models offering scalable alternatives to proprietary systems and few-shot can further improve LLMs' capability. 

---
# From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review 

**Authors**: Yaohui Zhang, Haijing Zhang, Wenlong Ji, Tianyu Hua, Nick Haber, Hancheng Cao, Weixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11343)  

**Abstract**: The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity. 

---
# Learning a Continue-Thinking Token for Enhanced Test-Time Scaling 

**Authors**: Liran Ringel, Elad Tolochinsky, Yaniv Romano  

**Link**: [PDF](https://arxiv.org/pdf/2506.11274)  

**Abstract**: Test-time scaling has emerged as an effective approach for improving language model performance by utilizing additional compute at inference time. Recent studies have shown that overriding end-of-thinking tokens (e.g., replacing "</think>" with "Wait") can extend reasoning steps and improve accuracy. In this work, we explore whether a dedicated continue-thinking token can be learned to trigger extended reasoning. We augment a distilled version of DeepSeek-R1 with a single learned "<|continue-thinking|>" token, training only its embedding via reinforcement learning while keeping the model weights frozen. Our experiments show that this learned token achieves improved accuracy on standard math benchmarks compared to both the baseline model and a test-time scaling approach that uses a fixed token (e.g., "Wait") for budget forcing. In particular, we observe that in cases where the fixed-token approach enhances the base model's accuracy, our method achieves a markedly greater improvement. For example, on the GSM8K benchmark, the fixed-token approach yields a 1.3% absolute improvement in accuracy, whereas our learned-token method achieves a 4.2% improvement over the base model that does not use budget forcing. 

---
# Customizing Speech Recognition Model with Large Language Model Feedback 

**Authors**: Shaoshi Ling, Guoli Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.11091)  

**Abstract**: Automatic speech recognition (ASR) systems have achieved strong performance on general transcription tasks. However, they continue to struggle with recognizing rare named entities and adapting to domain mismatches. In contrast, large language models (LLMs), trained on massive internet-scale datasets, are often more effective across a wide range of domains. In this work, we propose a reinforcement learning based approach for unsupervised domain adaptation, leveraging unlabeled data to enhance transcription quality, particularly the named entities affected by domain mismatch, through feedback from a LLM. Given contextual information, our framework employs a LLM as the reward model to score the hypotheses from the ASR model. These scores serve as reward signals to fine-tune the ASR model via reinforcement learning. Our method achieves a 21\% improvement on entity word error rate over conventional self-training methods. 

---
# MANBench: Is Your Multimodal Model Smarter than Human? 

**Authors**: Han Zhou, Qitong Xu, Yiheng Dong, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11080)  

**Abstract**: The rapid advancement of Multimodal Large Language Models (MLLMs) has ignited discussions regarding their potential to surpass human performance in multimodal tasks. In response, we introduce MANBench (Multimodal Ability Norms Benchmark), a bilingual benchmark (English and Chinese) comprising 1,314 questions across nine tasks, spanning knowledge-based and non-knowledge-based domains. MANBench emphasizes intuitive reasoning, seamless cross-modal integration, and real-world complexity, providing a rigorous evaluation framework.
Through extensive human experiments involving diverse participants, we compared human performance against state-of-the-art MLLMs. The results indicate that while MLLMs excel in tasks like Knowledge and Text-Image Understanding, they struggle with deeper cross-modal reasoning tasks such as Transmorphic Understanding, Image Consistency, and Multi-image Understanding. Moreover, both humans and MLLMs face challenges in highly complex tasks like Puzzles and Spatial Imagination.
MANBench highlights the strengths and limitations of MLLMs, revealing that even advanced models fall short of achieving human-level performance across many domains. We hope MANBench will inspire efforts to bridge the gap between MLLMs and human multimodal capabilities. The code and dataset are available at this https URL. 

---
# SAGE:Specification-Aware Grammar Extraction for Automated Test Case Generation with LLMs 

**Authors**: Aditi, Hyunwoo Park, Sicheol Sung, Yo-Sub Han, Sang-Ki Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.11081)  

**Abstract**: Grammar-based test case generation has proven effective for competitive programming problems, but generating valid and general grammars from natural language specifications remains a key challenge, especially under limited supervision. Context-Free Grammars with Counters (CCFGs) have recently been introduced as a formalism to represent such specifications with logical constraints by storing and reusing counter values during derivation. In this work, we explore the use of open-source large language models (LLMs) to induce CCFGs from specifications using a small number of labeled examples and verifiable reward-guided reinforcement learning. Our approach first fine-tunes an open-source LLM to perform specification-to-grammar translation, and further applies Group Relative Policy Optimization (GRPO) to enhance grammar validity and generality. We also examine the effectiveness of iterative feedback for open and closed-source LLMs in correcting syntactic and semantic errors in generated grammars.
Experimental results show that our approach SAGE achieves stronger generalization and outperforms 17 open and closed-source LLMs in both grammar quality and test effectiveness, improving over the state-of-the-art by 15.92%p in grammar validity and 12.34%p in test effectiveness. We provide our implementation and dataset at the following anonymous repository:this https URL 

---
# A Large Language Model Based Pipeline for Review of Systems Entity Recognition from Clinical Notes 

**Authors**: Hieu Nghiem, Hemanth Reddy Singareddy, Zhuqi Miao, Jivan Lamichhane, Abdulaziz Ahmed, Johnson Thomas, Dursun Delen, William Paiva  

**Link**: [PDF](https://arxiv.org/pdf/2506.11067)  

**Abstract**: Objective: Develop a cost-effective, large language model (LLM)-based pipeline for automatically extracting Review of Systems (ROS) entities from clinical notes. Materials and Methods: The pipeline extracts ROS sections using SecTag, followed by few-shot LLMs to identify ROS entity spans, their positive/negative status, and associated body systems. We implemented the pipeline using open-source LLMs (Mistral, Llama, Gemma) and ChatGPT. The evaluation was conducted on 36 general medicine notes containing 341 annotated ROS entities. Results: When integrating ChatGPT, the pipeline achieved the lowest error rates in detecting ROS entity spans and their corresponding statuses/systems (28.2% and 14.5%, respectively). Open-source LLMs enable local, cost-efficient execution of the pipeline while delivering promising performance with similarly low error rates (span: 30.5-36.7%; status/system: 24.3-27.3%). Discussion and Conclusion: Our pipeline offers a scalable and locally deployable solution to reduce ROS documentation burden. Open-source LLMs present a viable alternative to commercial models in resource-limited healthcare environments. 

---
# Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models 

**Authors**: Bumjin Park, Jinsil Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11068)  

**Abstract**: Large language models (LLMs) are increasingly engaging in moral and ethical reasoning, where criteria for judgment are often unclear, even for humans. While LLM alignment studies cover many areas, one important yet underexplored area is how LLMs make judgments about obligations. This work reveals a strong tendency in LLMs to judge non-obligatory contexts as obligations when prompts are augmented with modal expressions such as must or ought to. We introduce this phenomenon as Deontological Keyword Bias (DKB). We find that LLMs judge over 90\% of commonsense scenarios as obligations when modal expressions are present. This tendency is consist across various LLM families, question types, and answer formats. To mitigate DKB, we propose a judgment strategy that integrates few-shot examples with reasoning prompts. This study sheds light on how modal expressions, as a form of linguistic framing, influence the normative decisions of LLMs and underscores the importance of addressing such biases to ensure judgment alignment. 

---
# RoE-FND: A Case-Based Reasoning Approach with Dual Verification for Fake News Detection via LLMs 

**Authors**: Yuzhou Yang, Yangming Zhou, Zhiying Zhu, Zhenxing Qian, Xinpeng Zhang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11078)  

**Abstract**: The proliferation of deceptive content online necessitates robust Fake News Detection (FND) systems. While evidence-based approaches leverage external knowledge to verify claims, existing methods face critical limitations: noisy evidence selection, generalization bottlenecks, and unclear decision-making processes. Recent efforts to harness Large Language Models (LLMs) for FND introduce new challenges, including hallucinated rationales and conclusion bias. To address these issues, we propose \textbf{RoE-FND} (\textbf{\underline{R}}eason \textbf{\underline{o}}n \textbf{\underline{E}}xperiences FND), a framework that reframes evidence-based FND as a logical deduction task by synergizing LLMs with experiential learning. RoE-FND encompasses two stages: (1) \textit{self-reflective knowledge building}, where a knowledge base is curated by analyzing past reasoning errors, namely the exploration stage, and (2) \textit{dynamic criterion retrieval}, which synthesizes task-specific reasoning guidelines from historical cases as experiences during deployment. It further cross-checks rationales against internal experience through a devised dual-channel procedure. Key contributions include: a case-based reasoning framework for FND that addresses multiple existing challenges, a training-free approach enabling adaptation to evolving situations, and empirical validation of the framework's superior generalization and effectiveness over state-of-the-art methods across three datasets. 

---
# RedDebate: Safer Responses through Multi-Agent Red Teaming Debates 

**Authors**: Ali Asad, Stephen Obadinma, Radin Shayanfar, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11083)  

**Abstract**: We propose RedDebate, a novel multi-agent debate framework that leverages adversarial argumentation among Large Language Models (LLMs) to proactively identify and mitigate their own unsafe behaviours. Existing AI safety methods often depend heavily on costly human evaluations or isolated single-model assessment, both subject to scalability constraints and oversight risks. RedDebate instead embraces collaborative disagreement, enabling multiple LLMs to critically examine one another's reasoning, and systematically uncovering unsafe blind spots through automated red-teaming, and iteratively improve their responses. We further integrate distinct types of long-term memory that retain learned safety insights from debate interactions. Evaluating on established safety benchmarks such as HarmBench, we demonstrate the proposed method's effectiveness. Debate alone can reduce unsafe behaviours by 17.7%, and when combined with long-term memory modules, achieves reductions exceeding 23.5%. To our knowledge, RedDebate constitutes the first fully automated framework that combines multi-agent debates with red-teaming to progressively enhance AI safety without direct human intervention.(Github Repository: this https URL) 

---
# Smotrom tvoja pa ander drogoj verden! Resurrecting Dead Pidgin with Generative Models: Russenorsk Case Study 

**Authors**: Alexey Tikhonov, Sergei Shteiner, Anna Bykova, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.11065)  

**Abstract**: Russenorsk, a pidgin language historically used in trade interactions between Russian and Norwegian speakers, represents a unique linguistic phenomenon. In this paper, we attempt to analyze its lexicon using modern large language models (LLMs), based on surviving literary sources. We construct a structured dictionary of the language, grouped by synonyms and word origins. Subsequently, we use this dictionary to formulate hypotheses about the core principles of word formation and grammatical structure in Russenorsk and show which hypotheses generated by large language models correspond to the hypotheses previously proposed ones in the academic literature. We also develop a "reconstruction" translation agent that generates hypothetical Russenorsk renderings of contemporary Russian and Norwegian texts. 

---
# TreeRL: LLM Reinforcement Learning with On-Policy Tree Search 

**Authors**: Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11902)  

**Abstract**: Reinforcement learning (RL) with tree search has demonstrated superior performance in traditional reasoning tasks. Compared to conventional independent chain sampling strategies with outcome supervision, tree search enables better exploration of the reasoning space and provides dense, on-policy process rewards during RL training but remains under-explored in On-Policy LLM RL. We propose TreeRL, a reinforcement learning framework that directly incorporates on-policy tree search for RL training. Our approach includes intermediate supervision and eliminates the need for a separate reward model training. Existing approaches typically train a separate process reward model, which can suffer from distribution mismatch and reward hacking. We also introduce a cost-effective tree search approach that achieves higher search efficiency under the same generation token budget by strategically branching from high-uncertainty intermediate steps rather than using random branching. Experiments on challenging math and code reasoning benchmarks demonstrate that TreeRL achieves superior performance compared to traditional ChainRL, highlighting the potential of tree search for LLM. TreeRL is open-sourced at this https URL. 

---
# CyclicReflex: Improving Large Reasoning Models via Cyclical Reflection Token Scheduling 

**Authors**: Chongyu Fan, Yihua Zhang, Jinghan Jia, Alfred Hero, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11077)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens or textual segments that prompt self-evaluative reflection. We refer to these transition markers and reflective cues as "reflection tokens" (e.g., "wait", "but", "alternatively"). In this work, we treat reflection tokens as a "resource" and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand and manage this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, we propose cyclical reflection token scheduling (termed CyclicReflex), a decoding strategy that dynamically modulates reflection token logits using a position-dependent triangular waveform. Experiments on MATH500, AIME2024/2025, and AMC2023 demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B-8B), outperforming standard decoding and more recent approaches such as TIP (thought switching penalty) and S1. Codes are available at this https URL. 

---
# Brewing Knowledge in Context: Distillation Perspectives on In-Context Learning 

**Authors**: Chengye Li, Haiyun Liu, Yuanxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11516)  

**Abstract**: In-context learning (ICL) allows large language models (LLMs) to solve novel tasks without weight updates. Despite its empirical success, the mechanism behind ICL remains poorly understood, limiting our ability to interpret, improve, and reliably apply it. In this paper, we propose a new theoretical perspective that interprets ICL as an implicit form of knowledge distillation (KD), where prompt demonstrations guide the model to form a task-specific reference model during inference. Under this view, we derive a Rademacher complexity-based generalization bound and prove that the bias of the distilled weights grows linearly with the Maximum Mean Discrepancy (MMD) between the prompt and target distributions. This theoretical framework explains several empirical phenomena and unifies prior gradient-based and distributional analyses. To the best of our knowledge, this is the first to formalize inference-time attention as a distillation process, which provides theoretical insights for future prompt engineering and automated demonstration selection. 

---
# LLM-as-a-Judge for Reference-less Automatic Code Validation and Refinement for Natural Language to Bash in IT Automation 

**Authors**: Ngoc Phuoc An Vo, Brent Paulovicks, Vadim Sheinin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11237)  

**Abstract**: In an effort to automatically evaluate and select the best model and improve code quality for automatic incident remediation in IT Automation, it is crucial to verify if the generated code for remediation action is syntactically and semantically correct and whether it can be executed correctly as intended. There are three approaches: 1) conventional methods use surface form similarity metrics (token match, exact match, etc.) which have numerous limitations, 2) execution-based evaluation focuses more on code functionality based on pass/fail judgments for given test-cases, and 3) LLM-as-a-Judge employs LLMs for automated evaluation to judge if it is a correct answer for a given problem based on pre-defined metrics. In this work, we focused on enhancing LLM-as-a-Judge using bidirectional functionality matching and logic representation for reference-less automatic validation and refinement for Bash code generation to select the best model for automatic incident remediation in IT Automation. We used execution-based evaluation as ground-truth to evaluate our LLM-as-a-Judge metrics. Results show high accuracy and agreement with execution-based evaluation (and up to 8% over baseline). Finally, we built Reflection code agents to utilize judgments and feedback from our evaluation metrics which achieved significant improvement (up to 24% increase in accuracy) for automatic code refinement. 

---
# Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs 

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.11415)  

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system. 

---
# Large Language models for Time Series Analysis: Techniques, Applications, and Challenges 

**Authors**: Feifei Shi, Xueyan Yin, Kang Wang, Wanyu Tu, Qifu Sun, Huansheng Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.11040)  

**Abstract**: Time series analysis is pivotal in domains like financial forecasting and biomedical monitoring, yet traditional methods are constrained by limited nonlinear feature representation and long-term dependency capture. The emergence of Large Language Models (LLMs) offers transformative potential by leveraging their cross-modal knowledge integration and inherent attention mechanisms for time series analysis. However, the development of general-purpose LLMs for time series from scratch is still hindered by data diversity, annotation scarcity, and computational requirements. This paper presents a systematic review of pre-trained LLM-driven time series analysis, focusing on enabling techniques, potential applications, and open challenges. First, it establishes an evolutionary roadmap of AI-driven time series analysis, from the early machine learning era, through the emerging LLM-driven paradigm, to the development of native temporal foundation models. Second, it organizes and systematizes the technical landscape of LLM-driven time series analysis from a workflow perspective, covering LLMs' input, optimization, and lightweight stages. Finally, it critically examines novel real-world applications and highlights key open challenges that can guide future research and innovation. The work not only provides valuable insights into current advances but also outlines promising directions for future development. It serves as a foundational reference for both academic and industrial researchers, paving the way for the development of more efficient, generalizable, and interpretable systems of LLM-driven time series analysis. 

---
# Glider: Global and Local Instruction-Driven Expert Router 

**Authors**: Pingzhi Li, Prateek Yadav, Jaehong Yoon, Jie Peng, Yi-Lin Sung, Mohit Bansal, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2410.07172)  

**Abstract**: The availability of performant pre-trained models has led to a proliferation of fine-tuned expert models that are specialized to particular domains. This has enabled the creation of powerful and adaptive routing-based "Model MoErging" methods with the goal of using expert modules to create an aggregate system with improved performance or generalization. However, existing MoErging methods often prioritize generalization to unseen tasks at the expense of performance on held-in tasks, which limits its practical applicability in real-world deployment scenarios. We observe that current token-level routing mechanisms neglect the global semantic context of the input task. This token-wise independence hinders effective expert selection for held-in tasks, as routing decisions fail to incorporate the semantic properties of the task. To address this, we propose, Global and Local Instruction Driven Expert Router (GLIDER) that integrates a multi-scale routing mechanism, encompassing a semantic global router and a learned local router. The global router leverages LLM's advanced reasoning capabilities for semantic-related contexts to enhance expert selection. Given the input query and LLM, the router generates semantic task instructions that guide the retrieval of the most relevant experts across all layers. This global guidance is complemented by a local router that facilitates token-level routing decisions within each module, enabling finer control and enhanced performance on unseen tasks. Our experiments using T5-based models for T0 and FLAN tasks demonstrate that GLIDER achieves substantially improved held-in performance while maintaining strong generalization on held-out tasks. We also perform ablations experiments to dive deeper into the components of GLIDER. Our experiments highlight the importance of our multi-scale routing that leverages LLM-driven semantic reasoning for MoErging methods. 

---
