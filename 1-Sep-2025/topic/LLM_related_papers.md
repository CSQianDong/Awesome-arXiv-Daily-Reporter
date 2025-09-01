# Not All Parameters Are Created Equal: Smart Isolation Boosts Fine-Tuning Performance 

**Authors**: Yao Wang, Di Liang, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21741)  

**Abstract**: Supervised fine-tuning (SFT) is a pivotal approach to adapting large language models (LLMs) for downstream tasks; however, performance often suffers from the ``seesaw phenomenon'', where indiscriminate parameter updates yield progress on certain tasks at the expense of others. To address this challenge, we propose a novel \emph{Core Parameter Isolation Fine-Tuning} (CPI-FT) framework. Specifically, we first independently fine-tune the LLM on each task to identify its core parameter regions by quantifying parameter update magnitudes. Tasks with similar core regions are then grouped based on region overlap, forming clusters for joint modeling. We further introduce a parameter fusion technique: for each task, core parameters from its individually fine-tuned model are directly transplanted into a unified backbone, while non-core parameters from different tasks are smoothly integrated via Spherical Linear Interpolation (SLERP), mitigating destructive interference. A lightweight, pipelined SFT training phase using mixed-task data is subsequently employed, while freezing core regions from prior tasks to prevent catastrophic forgetting. Extensive experiments on multiple public benchmarks demonstrate that our approach significantly alleviates task interference and forgetting, consistently outperforming vanilla multi-task and multi-stage fine-tuning baselines. 

---
# PiCSAR: Probabilistic Confidence Selection And Ranking 

**Authors**: Joshua Ong Jun Leang, Zheng Zhao, Aryo Pradipta Gema, Sohee Yang, Wai-Chung Kwan, Xuanli He, Wenda Li, Pasquale Minervini, Eleonora Giunchiglia, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21787)  

**Abstract**: Best-of-n sampling improves the accuracy of large language models (LLMs) and large reasoning models (LRMs) by generating multiple candidate solutions and selecting the one with the highest reward. The key challenge for reasoning tasks is designing a scoring function that can identify correct reasoning chains without access to ground-truth answers. We propose Probabilistic Confidence Selection And Ranking (PiCSAR): a simple, training-free method that scores each candidate generation using the joint log-likelihood of the reasoning and final answer. The joint log-likelihood of the reasoning and final answer naturally decomposes into reasoning confidence and answer confidence. PiCSAR achieves substantial gains across diverse benchmarks (+10.18 on MATH500, +9.81 on AIME2025), outperforming baselines with at least 2x fewer samples in 16 out of 20 comparisons. Our analysis reveals that correct reasoning chains exhibit significantly higher reasoning and answer confidence, justifying the effectiveness of PiCSAR. 

---
# A Survey on Current Trends and Recent Advances in Text Anonymization 

**Authors**: Tobias Deußer, Lorenz Sparrenberg, Armin Berger, Max Hahnbück, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.21587)  

**Abstract**: The proliferation of textual data containing sensitive personal information across various domains requires robust anonymization techniques to protect privacy and comply with regulations, while preserving data usability for diverse and crucial downstream tasks. This survey provides a comprehensive overview of current trends and recent advances in text anonymization techniques. We begin by discussing foundational approaches, primarily centered on Named Entity Recognition, before examining the transformative impact of Large Language Models, detailing their dual role as sophisticated anonymizers and potent de-anonymization threats. The survey further explores domain-specific challenges and tailored solutions in critical sectors such as healthcare, law, finance, and education. We investigate advanced methodologies incorporating formal privacy models and risk-aware frameworks, and address the specialized subfield of authorship anonymization. Additionally, we review evaluation frameworks, comprehensive metrics, benchmarks, and practical toolkits for real-world deployment of anonymization solutions. This review consolidates current knowledge, identifies emerging trends and persistent challenges, including the evolving privacy-utility trade-off, the need to address quasi-identifiers, and the implications of LLM capabilities, and aims to guide future research directions for both academics and practitioners in this field. 

---
# Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards 

**Authors**: Xiaolong Wei, Bo Lu, Xingyu Zhang, Zhejun Zhao, Dongdong Shen, Long Xia, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21476)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at this https URL. 

---
# Personality Matters: User Traits Predict LLM Preferences in Multi-Turn Collaborative Tasks 

**Authors**: Sarfaroz Yunusov, Kaige Chen, Kazi Nishat Anwar, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2508.21628)  

**Abstract**: As Large Language Models (LLMs) increasingly integrate into everyday workflows, where users shape outcomes through multi-turn collaboration, a critical question emerges: do users with different personality traits systematically prefer certain LLMs over others? We conducted a study with 32 participants evenly distributed across four Keirsey personality types, evaluating their interactions with GPT-4 and Claude 3.5 across four collaborative tasks: data analysis, creative writing, information retrieval, and writing assistance. Results revealed significant personality-driven preferences: Rationals strongly preferred GPT-4, particularly for goal-oriented tasks, while idealists favored Claude 3.5, especially for creative and analytical tasks. Other personality types showed task-dependent preferences. Sentiment analysis of qualitative feedback confirmed these patterns. Notably, aggregate helpfulness ratings were similar across models, showing how personality-based analysis reveals LLM differences that traditional evaluations miss. 

---
# Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning 

**Authors**: Zinan Tang, Xin Gao, Qizhi Pei, Zhuoshi Pan, Mengzhang Cai, Jiang Wu, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21589)  

**Abstract**: Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our \method consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon. 

---
# Beyond the Surface: Probing the Ideological Depth of Large Language Models 

**Authors**: Shariar Kabir, Kevin Esterling, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.21448)  

**Abstract**: Large Language Models (LLMs) have demonstrated pronounced ideological leanings, yet the stability and depth of these positions remain poorly understood. Surface-level responses can often be manipulated through simple prompt engineering, calling into question whether they reflect a coherent underlying ideology. This paper investigates the concept of "ideological depth" in LLMs, defined as the robustness and complexity of their internal political representations. We employ a dual approach: first, we measure the "steerability" of two well-known open-source LLMs using instruction prompting and activation steering. We find that while some models can easily switch between liberal and conservative viewpoints, others exhibit resistance or an increased rate of refusal, suggesting a more entrenched ideological structure. Second, we probe the internal mechanisms of these models using Sparse Autoencoders (SAEs). Preliminary analysis reveals that models with lower steerability possess more distinct and abstract ideological features. Our evaluations reveal that one model can contain 7.3x more political features than another model of similar size. This allows targeted ablation of a core political feature in an ideologically "deep" model, leading to consistent, logical shifts in its reasoning across related topics, whereas the same intervention in a "shallow" model results in an increase in refusal outputs. Our findings suggest that ideological depth is a quantifiable property of LLMs and that steerability serves as a valuable window into their latent political architecture. 

---
# Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework 

**Authors**: Nils Dycke, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.21422)  

**Abstract**: Large Language Models (LLMs) have great potential to accelerate and support scholarly peer review and are increasingly used as fully automatic review generators (ARGs). However, potential biases and systematic errors may pose significant risks to scientific integrity; understanding the specific capabilities and limitations of state-of-the-art ARGs is essential. We focus on a core reviewing skill that underpins high-quality peer review: detecting faulty research logic. This involves evaluating the internal consistency between a paper's results, interpretations, and claims. We present a fully automated counterfactual evaluation framework that isolates and tests this skill under controlled conditions. Testing a range of ARG approaches, we find that, contrary to expectation, flaws in research logic have no significant effect on their output reviews. Based on our findings, we derive three actionable recommendations for future work and release our counterfactual dataset and evaluation framework publicly. 

---
# Challenges and Applications of Large Language Models: A Comparison of GPT and DeepSeek family of models 

**Authors**: Shubham Sharma, Sneha Tuli, Narendra Badam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21377)  

**Abstract**: Large Language Models (LLMs) are transforming AI across industries, but their development and deployment remain complex. This survey reviews 16 key challenges in building and using LLMs and examines how these challenges are addressed by two state-of-the-art models with unique approaches: OpenAI's closed source GPT-4o (May 2024 update) and DeepSeek-V3-0324 (March 2025), a large open source Mixture-of-Experts model. Through this comparison, we showcase the trade-offs between closed source models (robust safety, fine-tuned reliability) and open source models (efficiency, adaptability). We also explore LLM applications across different domains (from chatbots and coding tools to healthcare and education), highlighting which model attributes are best suited for each use case. This article aims to guide AI researchers, developers, and decision-makers in understanding current LLM capabilities, limitations, and best practices. 

---
# QZhou-Embedding Technical Report 

**Authors**: Peng Yu, En Xu, Bin Chen, Haibiao Chen, Yinfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21632)  

**Abstract**: We present QZhou-Embedding, a general-purpose contextual text embedding model with exceptional text representation capabilities. Built upon the Qwen2.5-7B-Instruct foundation model, we designed a unified multi-task framework comprising specialized data transformation and training strategies. The data transformation scheme enables the incorporation of more diverse textual training datasets, while the task-specific training strategies enhance model learning efficiency. We developed a data synthesis pipeline leveraging LLM API, incorporating techniques such as paraphrasing, augmentation, and hard negative example generation to improve the semantic richness and sample difficulty of the training set. Additionally, we employ a two-stage training strategy, comprising initial retrieval-focused pretraining followed by full-task fine-tuning, enabling the embedding model to extend its capabilities based on robust retrieval performance. Our model achieves state-of-the-art results on the MTEB and CMTEB benchmarks, ranking first on both leaderboards (August 27 2025), and simultaneously achieves state-of-the-art performance on tasks including reranking, clustering, etc. Our findings demonstrate that higher-quality, more diverse data is crucial for advancing retrieval model performance, and that leveraging LLMs generative capabilities can further optimize data quality for embedding model breakthroughs. Our model weights are released on HuggingFace under Apache 2.0 license. For reproducibility, we provide evaluation code and instructions on GitHub. 

---
# Quantifying Label-Induced Bias in Large Language Model Self- and Cross-Evaluations 

**Authors**: Muskan Saraf, Sajjad Rezvani Boroujeni, Justin Beaudry, Hossein Abedi, Tom Bush  

**Link**: [PDF](https://arxiv.org/pdf/2508.21164)  

**Abstract**: Large language models (LLMs) are increasingly used to evaluate outputs, yet their judgments may be influenced. This study examines bias in self- and cross-model evaluations by ChatGPT, Gemini, and Claude under four conditions: no labels, true labels, and two false-label scenarios. Blog posts authored by each model were evaluated by all three using both overall preference voting and quality ratings for Coherence, Informativeness, and Conciseness, with all scores expressed as percentages for direct comparison. Results reveal striking asymmetries: the "Claude" label consistently boosts scores, while the "Gemini" label consistently depresses them, regardless of actual content. False labels frequently reversed rankings, producing shifts of up to 50 percentage points in preference votes and up to 12 percentage points in converted quality ratings. Gemini's self-scores collapsed under true labels, while Claude's self-preference intensified. These findings show that perceived model identity can heavily distort high-level judgments and subtly influence detailed quality ratings, underscoring the need for blind or multimodel evaluation protocols to ensure fairness in LLM benchmarking. 

---
# Reasoning-Intensive Regression 

**Authors**: Diane Tchuindjo, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2508.21762)  

**Abstract**: AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e. deducing subtle numerical properties from text. Unlike standard language regression tasks, e.g. for sentiment or similarity, RiR often appears instead in ad-hoc problems like rubric-based scoring or domain-specific retrieval, where much deeper analysis of text is required while only limited task-specific training data and computation are available. We cast three realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and finetuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning. MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances in RiR. 

---
# Summarize-Exemplify-Reflect: Data-driven Insight Distillation Empowers LLMs for Few-shot Tabular Classification 

**Authors**: Yifei Yuan, Jiatong Li, Weijia Zhang, Mohammad Aliannejadi, Evangelos Kanoulas, Renjun Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21561)  

**Abstract**: Recent studies show the promise of large language models (LLMs) for few-shot tabular classification but highlight challenges due to the variability in structured data. To address this, we propose distilling data into actionable insights to enable robust and effective classification by LLMs. Drawing inspiration from human learning processes, we introduce InsightTab, an insight distillation framework guided by principles of divide-and-conquer, easy-first, and reflective learning. Our approach integrates rule summarization, strategic exemplification, and insight reflection through deep collaboration between LLMs and data modeling techniques. The obtained insights enable LLMs to better align their general knowledge and capabilities with the particular requirements of specific tabular tasks. We extensively evaluate InsightTab on nine datasets. The results demonstrate consistent improvement over state-of-the-art methods. Ablation studies further validate the principle-guided distillation process, while analyses emphasize InsightTab's effectiveness in leveraging labeled data and managing bias. 

---
# Can Multimodal LLMs Solve the Basic Perception Problems of Percept-V? 

**Authors**: Samrajnee Ghosh, Naman Agarwal, Hemanshu Garg, Chinmay Mittal, Mausam, Parag Singla  

**Link**: [PDF](https://arxiv.org/pdf/2508.21143)  

**Abstract**: The reasoning abilities of Multimodal Large Language Models (MLLMs) have garnered a lot of attention in recent times, with advances made in frontiers like coding, mathematics, and science. However, very limited experiments have been done to assess their performance in simple perception tasks performed over uncontaminated, generated images containing basic shapes and structures. To address this issue, the paper introduces a dataset, Percept-V, containing a total of 7200 program-generated images equally divided into 30 categories, each testing a combination of visual perception skills. Unlike previously proposed datasets, Percept-V comprises very basic tasks of varying complexity that test the perception abilities of MLLMs. This dataset is then tested on state-of-the-art MLLMs like GPT-4o, Gemini, and Claude as well as Large Reasoning Models (LRMs) like OpenAI o4-mini and DeepSeek R1 to gauge their performance. Contrary to the evidence that MLLMs excel in many complex tasks, our experiments show a significant drop in the models' performance with increasing problem complexity across all categories. An analysis of the performances also reveals that the tested MLLMs exhibit a similar trend in accuracy across categories, testing a particular cognitive skill and find some skills to be more difficult than others. 

---
# Accept or Deny? Evaluating LLM Fairness and Performance in Loan Approval across Table-to-Text Serialization Approaches 

**Authors**: Israel Abebe Azime, Deborah D. Kanubala, Tejumade Afonja, Mario Fritz, Isabel Valera, Dietrich Klakow, Philipp Slusallek  

**Link**: [PDF](https://arxiv.org/pdf/2508.21512)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in high-stakes decision-making tasks, such as loan approvals. While their applications expand across domains, LLMs struggle to process tabular data, ensuring fairness and delivering reliable predictions. In this work, we assess the performance and fairness of LLMs on serialized loan approval datasets from three geographically distinct regions: Ghana, Germany, and the United States. Our evaluation focuses on the model's zero-shot and in-context learning (ICL) capabilities. Our results reveal that the choice of serialization (Serialization refers to the process of converting tabular data into text formats suitable for processing by LLMs.) format significantly affects both performance and fairness in LLMs, with certain formats such as GReat and LIFT yielding higher F1 scores but exacerbating fairness disparities. Notably, while ICL improved model performance by 4.9-59.6% relative to zero-shot baselines, its effect on fairness varied considerably across datasets. Our work underscores the importance of effective tabular data representation methods and fairness-aware models to improve the reliability of LLMs in financial decision-making. 

---
# Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding 

**Authors**: Vanessa Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21204)  

**Abstract**: We study how architectural inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding mechanism paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which architectural scaffolds can reliably shape emergent instructional strategies in LLMs. 

---
# Designing Smarter Conversational Agents for Kids: Lessons from Cognitive Work and Means-Ends Analyses 

**Authors**: Vanessa Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21209)  

**Abstract**: This paper presents two studies on how Brazilian children (ages 9--11) use conversational agents (CAs) for schoolwork, discovery, and entertainment, and how structured scaffolds can enhance these interactions. In Study 1, a seven-week online investigation with 23 participants (children, parents, teachers) employed interviews, observations, and Cognitive Work Analysis to map children's information-processing flows, the role of more knowledgeable others, functional uses, contextual goals, and interaction patterns to inform conversation-tree design. We identified three CA functions: School, Discovery, Entertainment, and derived ``recipe'' scaffolds mirroring parent-child support. In Study 2, we prompted GPT-4o-mini on 1,200 simulated child-CA exchanges, comparing conversation-tree recipes based on structured-prompting to an unstructured baseline. Quantitative evaluation of readability, question count/depth/diversity, and coherence revealed gains for the recipe approach. Building on these findings, we offer design recommendations: scaffolded conversation-trees, child-dedicated profiles for personalized context, and caregiver-curated content. Our contributions include the first CWA application with Brazilian children, an empirical framework of child-CA information flows, and an LLM-scaffolding ``recipe'' (i.e., structured-prompting) for effective, scaffolded learning. 

---
# Database Normalization via Dual-LLM Self-Refinement 

**Authors**: Eunjae Jo, Nakyung Lee, Gyuyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17693)  

**Abstract**: Database normalization is crucial to preserving data integrity. However, it is time-consuming and error-prone, as it is typically performed manually by data engineers. To this end, we present Miffie, a database normalization framework that leverages the capability of large language models. Miffie enables automated data normalization without human effort while preserving high accuracy. The core of Miffie is a dual-model self-refinement architecture that combines the best-performing models for normalized schema generation and verification, respectively. The generation module eliminates anomalies based on the feedback of the verification module until the output schema satisfies the requirement for normalization. We also carefully design task-specific zero-shot prompts to guide the models for achieving both high accuracy and cost efficiency. Experimental results show that Miffie can normalize complex database schemas while maintaining high accuracy. 

---
# BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design 

**Authors**: Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, Tom Rainforth  

**Link**: [PDF](https://arxiv.org/pdf/2508.21184)  

**Abstract**: We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies. 

---
# How Does Cognitive Bias Affect Large Language Models? A Case Study on the Anchoring Effect in Price Negotiation Simulations 

**Authors**: Yoshiki Takenami, Yin Jou Huang, Yugo Murawaki, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21137)  

**Abstract**: Cognitive biases, well-studied in humans, can also be observed in LLMs, affecting their reliability in real-world applications. This paper investigates the anchoring effect in LLM-driven price negotiations. To this end, we instructed seller LLM agents to apply the anchoring effect and evaluated negotiations using not only an objective metric but also a subjective metric. Experimental results show that LLMs are influenced by the anchoring effect like humans. Additionally, we investigated the relationship between the anchoring effect and factors such as reasoning and personality. It was shown that reasoning models are less prone to the anchoring effect, suggesting that the long chain of thought mitigates the effect. However, we found no significant correlation between personality traits and susceptibility to the anchoring effect. These findings contribute to a deeper understanding of cognitive biases in LLMs and to the realization of safe and responsible application of LLMs in society. 

---
# Integrating Large Language Models with Network Optimization for Interactive and Explainable Supply Chain Planning: A Real-World Case Study 

**Authors**: Saravanan Venkatachalam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21622)  

**Abstract**: This paper presents an integrated framework that combines traditional network optimization models with large language models (LLMs) to deliver interactive, explainable, and role-aware decision support for supply chain planning. The proposed system bridges the gap between complex operations research outputs and business stakeholder understanding by generating natural language summaries, contextual visualizations, and tailored key performance indicators (KPIs). The core optimization model addresses tactical inventory redistribution across a network of distribution centers for multi-period and multi-item, using a mixed-integer formulation. The technical architecture incorporates AI agents, RESTful APIs, and a dynamic user interface to support real-time interaction, configuration updates, and simulation-based insights. A case study demonstrates how the system improves planning outcomes by preventing stockouts, reducing costs, and maintaining service levels. Future extensions include integrating private LLMs, transfer learning, reinforcement learning, and Bayesian neural networks to enhance explainability, adaptability, and real-time decision-making. 

---
# HealthProcessAI: A Technical Framework and Proof-of-Concept for LLM-Enhanced Healthcare Process Mining 

**Authors**: Eduardo Illueca-Fernandez, Kaile Chen, Fernando Seoane, Farhad Abtahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21540)  

**Abstract**: Process mining has emerged as a powerful analytical technique for understanding complex healthcare workflows. However, its application faces significant barriers, including technical complexity, a lack of standardized approaches, and limited access to practical training resources. We introduce HealthProcessAI, a GenAI framework designed to simplify process mining applications in healthcare and epidemiology by providing a comprehensive wrapper around existing Python (PM4PY) and R (bupaR) libraries. To address unfamiliarity and improve accessibility, the framework integrates multiple Large Language Models (LLMs) for automated process map interpretation and report generation, helping translate technical analyses into outputs that diverse users can readily understand. We validated the framework using sepsis progression data as a proof-of-concept example and compared the outputs of five state-of-the-art LLM models through the OpenRouter platform. To test its functionality, the framework successfully processed sepsis data across four proof-of-concept scenarios, demonstrating robust technical performance and its capability to generate reports through automated LLM analysis. LLM evaluation using five independent LLMs as automated evaluators revealed distinct model strengths: Claude Sonnet-4 and Gemini 2.5-Pro achieved the highest consistency scores (3.79/4.0 and 3.65/4.0) when evaluated by automated LLM assessors. By integrating multiple Large Language Models (LLMs) for automated interpretation and report generation, the framework addresses widespread unfamiliarity with process mining outputs, making them more accessible to clinicians, data scientists, and researchers. This structured analytics and AI-driven interpretation combination represents a novel methodological advance in translating complex process mining results into potentially actionable insights for healthcare applications. 

---
# Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture 

**Authors**: Yeawon Lee, Xiaoyang Wang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21803)  

**Abstract**: Accurate interpretation of clinical narratives is critical for patient care, but the complexity of these notes makes automation challenging. While Large Language Models (LLMs) show promise, single-model approaches can lack the robustness required for high-stakes clinical tasks. We introduce a collaborative multi-agent system (MAS) that models a clinical consultation team to address this gap. The system is tasked with identifying clinical problems by analyzing only the Subjective (S) and Objective (O) sections of SOAP notes, simulating the diagnostic reasoning process of synthesizing raw data into an assessment. A Manager agent orchestrates a dynamically assigned team of specialist agents who engage in a hierarchical, iterative debate to reach a consensus. We evaluated our MAS against a single-agent baseline on a curated dataset of 420 MIMIC-III notes. The dynamic multi-agent configuration demonstrated consistently improved performance in identifying congestive heart failure, acute kidney injury, and sepsis. Qualitative analysis of the agent debates reveals that this structure effectively surfaces and weighs conflicting evidence, though it can occasionally be susceptible to groupthink. By modeling a clinical team's reasoning process, our system offers a promising path toward more accurate, robust, and interpretable clinical decision support tools. 

---
# Think in Games: Learning to Reason in Games via Reinforcement Learning with Large Language Models 

**Authors**: Yi Liao, Yu Gu, Yuan Sui, Zining Zhu, Yifan Lu, Guohua Tang, Zhongqian Sun, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21365)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks such as mathematics and coding, yet they frequently struggle with simple interactive tasks that young children perform effortlessly. This discrepancy highlights a critical gap between declarative knowledge (knowing about something) and procedural knowledge (knowing how to do something). Although traditional reinforcement learning (RL) agents can acquire procedural knowledge through environmental interaction, they often operate as black boxes and require substantial training data. In contrast, LLMs possess extensive world knowledge and reasoning capabilities, but are unable to effectively convert this static knowledge into dynamic decision-making in interactive settings. To address this challenge, we propose Think in Games (TiG), a novel framework that empowers LLMs to develop procedural understanding through direct interaction with game environments, while retaining their inherent reasoning and explanatory abilities. Specifically, TiG reformulates RL-based decision-making as a language modeling task: LLMs generate language-guided policies, which are refined iteratively through online reinforcement learning based on environmental feedback. Our experimental results show that TiG successfully bridges the gap between declarative and procedural knowledge, achieving competitive performance with dramatically lower data and computational demands compared to conventional RL methods. Moreover, TiG provides step-by-step natural language explanations for its decisions, greatly improving transparency and interpretability in complex interactive tasks. 

---
# AI Compute Architecture and Evolution Trends 

**Authors**: Bor-Sung Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21394)  

**Abstract**: The focus of AI development has shifted from academic research to practical applications. However, AI development faces numerous challenges at various levels. This article will attempt to analyze the opportunities and challenges of AI from several different perspectives using a structured approach. This article proposes a seven-layer model for AI compute architecture, including Physical Layer, Link Layer, Neural Network Layer, Context Layer, Agent Layer, Orchestrator Layer, and Application Layer, from bottom to top. It also explains how AI computing has evolved into this 7-layer architecture through the three-stage evolution on large-scale language models (LLMs). For each layer, we describe the development trajectory and key technologies. In Layers 1 and 2 we discuss AI computing issues and the impact of Scale-Up and Scale-Out strategies on computing architecture. In Layer 3 we explore two different development paths for LLMs. In Layer 4 we discuss the impact of contextual memory on LLMs and compares it to traditional processor memory. In Layers 5 to 7 we discuss the trends of AI agents and explore the issues in evolution from a single AI agent to an AI-based ecosystem, and their impact on the AI industry. Furthermore, AI development involves not only technical challenges but also the economic issues to build self-sustainable ecosystem. This article analyzes the internet industry to provide predictions on the future trajectory of AI development. 

---
# Multi-Ontology Integration with Dual-Axis Propagation for Medical Concept Representation 

**Authors**: Mohsen Nayebi Kerdabadi, Arya Hadizadeh Moghaddam, Dongjie Wang, Zijun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21320)  

**Abstract**: Medical ontology graphs map external knowledge to medical codes in electronic health records via structured relationships. By leveraging domain-approved connections (e.g., parent-child), predictive models can generate richer medical concept representations by incorporating contextual information from related concepts. However, existing literature primarily focuses on incorporating domain knowledge from a single ontology system, or from multiple ontology systems (e.g., diseases, drugs, and procedures) in isolation, without integrating them into a unified learning structure. Consequently, concept representation learning often remains limited to intra-ontology relationships, overlooking cross-ontology connections. In this paper, we propose LINKO, a large language model (LLM)-augmented integrative ontology learning framework that leverages multiple ontology graphs simultaneously by enabling dual-axis knowledge propagation both within and across heterogeneous ontology systems to enhance medical concept representation learning. Specifically, LINKO first employs LLMs to provide a graph-retrieval-augmented initialization for ontology concept embedding, through an engineered prompt that includes concept descriptions, and is further augmented with ontology context. Second, our method jointly learns the medical concepts in diverse ontology graphs by performing knowledge propagation in two axes: (1) intra-ontology vertical propagation across hierarchical ontology levels and (2) inter-ontology horizontal propagation within every level in parallel. Last, through extensive experiments on two public datasets, we validate the superior performance of LINKO over state-of-the-art baselines. As a plug-in encoder compatible with existing EHR predictive models, LINKO further demonstrates enhanced robustness in scenarios involving limited data availability and rare disease prediction. 

---
# Addressing accuracy and hallucination of LLMs in Alzheimer's disease research through knowledge graphs 

**Authors**: Tingxuan Xu, Jiarui Feng, Justin Melendez, Kaleigh Roberts, Donghong Cai, Mingfang Zhu, Donald Elbert, Yixin Chen, Randall J. Bateman  

**Link**: [PDF](https://arxiv.org/pdf/2508.21238)  

**Abstract**: In the past two years, large language model (LLM)-based chatbots, such as ChatGPT, have revolutionized various domains by enabling diverse task completion and question-answering capabilities. However, their application in scientific research remains constrained by challenges such as hallucinations, limited domain-specific knowledge, and lack of explainability or traceability for the response. Graph-based Retrieval-Augmented Generation (GraphRAG) has emerged as a promising approach to improving chatbot reliability by integrating domain-specific contextual information before response generation, addressing some limitations of standard LLMs. Despite its potential, there are only limited studies that evaluate GraphRAG on specific domains that require intensive knowledge, like Alzheimer's disease or other biomedical domains. In this paper, we assess the quality and traceability of two popular GraphRAG systems. We compile a database of 50 papers and 70 expert questions related to Alzheimer's disease, construct a GraphRAG knowledge base, and employ GPT-4o as the LLM for answering queries. We then compare the quality of responses generated by GraphRAG with those from a standard GPT-4o model. Additionally, we discuss and evaluate the traceability of several Retrieval-Augmented Generation (RAG) and GraphRAG systems. Finally, we provide an easy-to-use interface with a pre-built Alzheimer's disease database for researchers to test the performance of both standard RAG and GraphRAG. 

---
# Benchmarking GPT-5 in Radiation Oncology: Measurable Gains, but Persistent Need for Expert Oversight 

**Authors**: Ugur Dinc, Jibak Sarkar, Philipp Schubert, Sabine Semrau, Thomas Weissmann, Andre Karius, Johann Brand, Bernd-Niklas Axer, Ahmed Gomaa, Pluvio Stephan, Ishita Sheth, Sogand Beirami, Annette Schwarz, Udo Gaipl, Benjamin Frey, Christoph Bert, Stefanie Corradini, Rainer Fietkau, Florian Putz  

**Link**: [PDF](https://arxiv.org/pdf/2508.21777)  

**Abstract**: Introduction: Large language models (LLM) have shown great potential in clinical decision support. GPT-5 is a novel LLM system that has been specifically marketed towards oncology use.
Methods: Performance was assessed using two complementary benchmarks: (i) the ACR Radiation Oncology In-Training Examination (TXIT, 2021), comprising 300 multiple-choice items, and (ii) a curated set of 60 authentic radiation oncologic vignettes representing diverse disease sites and treatment indications. For the vignette evaluation, GPT-5 was instructed to generate concise therapeutic plans. Four board-certified radiation oncologists rated correctness, comprehensiveness, and hallucinations. Inter-rater reliability was quantified using Fleiss' \k{appa}.
Results: On the TXIT benchmark, GPT-5 achieved a mean accuracy of 92.8%, outperforming GPT-4 (78.8%) and GPT-3.5 (62.1%). Domain-specific gains were most pronounced in Dose and Diagnosis. In the vignette evaluation, GPT-5's treatment recommendations were rated highly for correctness (mean 3.24/4, 95% CI: 3.11-3.38) and comprehensiveness (3.59/4, 95% CI: 3.49-3.69). Hallucinations were rare with no case reaching majority consensus for their presence. Inter-rater agreement was low (Fleiss' \k{appa} 0.083 for correctness), reflecting inherent variability in clinical judgment. Errors clustered in complex scenarios requiring precise trial knowledge or detailed clinical adaptation.
Discussion: GPT-5 clearly outperformed prior model variants on the radiation oncology multiple-choice benchmark. Although GPT-5 exhibited favorable performance in generating real-world radiation oncology treatment recommendations, correctness ratings indicate room for further improvement. While hallucinations were infrequent, the presence of substantive errors underscores that GPT-5-generated recommendations require rigorous expert oversight before clinical implementation. 

---
# Going over Fine Web with a Fine-Tooth Comb: Technical Report of Indexing Fine Web for Problematic Content Search and Retrieval 

**Authors**: Inés Altemir Marinas, Anastasiia Kucherenko, Andrei Kucharavy  

**Link**: [PDF](https://arxiv.org/pdf/2508.21788)  

**Abstract**: Large language models (LLMs) rely heavily on web-scale datasets like Common Crawl, which provides over 80\% of training data for some modern models. However, the indiscriminate nature of web crawling raises challenges in data quality, safety, and ethics. Despite the critical importance of training data quality, prior research on harmful content has been limited to small samples due to computational constraints. This project presents a framework for indexing and analyzing LLM training datasets using an ElasticSearch-based pipeline. We apply it to SwissAI's FineWeb-2 corpus (1.5TB, four languages), achieving fast query performance--most searches in milliseconds, all under 2 seconds. Our work demonstrates real-time dataset analysis, offering practical tools for safer, more accountable AI systems. 

---
# A Financial Brain Scan of the LLM 

**Authors**: Hui Chen, Antoine Didisheim, Luciano Somoza, Hanqing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.21285)  

**Abstract**: Emerging techniques in computer science make it possible to "brain scan" large language models (LLMs), identify the plain-English concepts that guide their reasoning, and steer them while holding other factors constant. We show that this approach can map LLM-generated economic forecasts to concepts such as sentiment, technical analysis, and timing, and compute their relative importance without reducing performance. We also show that models can be steered to be more or less risk-averse, optimistic, or pessimistic, which allows researchers to correct or simulate biases. The method is transparent, lightweight, and replicable for empirical research in the social sciences. 

---
# EconAgentic in DePIN Markets: A Large Language Model Approach to the Sharing Economy of Decentralized Physical Infrastructure 

**Authors**: Yulin Liu, Mocca Schweitzer  

**Link**: [PDF](https://arxiv.org/pdf/2508.21368)  

**Abstract**: The Decentralized Physical Infrastructure (DePIN) market is revolutionizing the sharing economy through token-based economics and smart contracts that govern decentralized operations. By 2024, DePIN projects have exceeded \$10 billion in market capitalization, underscoring their rapid growth. However, the unregulated nature of these markets, coupled with the autonomous deployment of AI agents in smart contracts, introduces risks such as inefficiencies and potential misalignment with human values. To address these concerns, we introduce EconAgentic, a Large Language Model (LLM)-powered framework designed to mitigate these challenges. Our research focuses on three key areas: 1) modeling the dynamic evolution of DePIN markets, 2) evaluating stakeholders' actions and their economic impacts, and 3) analyzing macroeconomic indicators to align market outcomes with societal goals. Through EconAgentic, we simulate how AI agents respond to token incentives, invest in infrastructure, and adapt to market conditions, comparing AI-driven decisions with human heuristic benchmarks. Our results show that EconAgentic provides valuable insights into the efficiency, inclusion, and stability of DePIN markets, contributing to both academic understanding and practical improvements in the design and governance of decentralized, tokenized economies. 

---
# RoboInspector: Unveiling the Unreliability of Policy Code for LLM-enabled Robotic Manipulation 

**Authors**: Chenduo Ying, Linkang Du, Peng Cheng, Yuanchao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21378)  

**Abstract**: Large language models (LLMs) demonstrate remarkable capabilities in reasoning and code generation, enabling robotic manipulation to be initiated with just a single instruction. The LLM carries out various tasks by generating policy code required to control the robot. Despite advances in LLMs, achieving reliable policy code generation remains a significant challenge due to the diverse requirements of real-world tasks and the inherent complexity of user instructions. In practice, different users may provide distinct instructions to drive the robot for the same task, which may cause the unreliability of policy code generation. To bridge this gap, we design RoboInspector, a pipeline to unveil and characterize the unreliability of the policy code for LLM-enabled robotic manipulation from two perspectives: the complexity of the manipulation task and the granularity of the instruction. We perform comprehensive experiments with 168 distinct combinations of tasks, instructions, and LLMs in two prominent frameworks. The RoboInspector identifies four main unreliable behaviors that lead to manipulation failure. We provide a detailed characterization of these behaviors and their underlying causes, giving insight for practical development to reduce unreliability. Furthermore, we introduce a refinement approach guided by failure policy code feedback that improves the reliability of policy code generation by up to 35% in LLM-enabled robotic manipulation, evaluated in both simulation and real-world environments. 

---
# zkLoRA: Fine-Tuning Large Language Models with Verifiable Security via Zero-Knowledge Proofs 

**Authors**: Guofu Liao, Taotao Wang, Shengli Zhang, Jiqun Zhang, Shi Long, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21393)  

**Abstract**: Fine-tuning large language models (LLMs) is crucial for adapting them to specific tasks, yet it remains computationally demanding and raises concerns about correctness and privacy, particularly in untrusted environments. Although parameter-efficient methods like Low-Rank Adaptation (LoRA) significantly reduce resource requirements, ensuring the security and verifiability of fine-tuning under zero-knowledge constraints remains an unresolved challenge. To address this, we introduce zkLoRA, the first framework to integrate LoRA fine-tuning with zero-knowledge proofs (ZKPs), achieving provable security and correctness. zkLoRA employs advanced cryptographic techniques -- such as lookup arguments, sumcheck protocols, and polynomial commitments -- to verify both arithmetic and non-arithmetic operations in Transformer-based architectures. The framework provides end-to-end verifiability for forward propagation, backward propagation, and parameter updates during LoRA fine-tuning, while safeguarding the privacy of model parameters and training data. Leveraging GPU-based implementations, zkLoRA demonstrates practicality and efficiency through experimental validation on open-source LLMs like LLaMA, scaling up to 13 billion parameters. By combining parameter-efficient fine-tuning with ZKPs, zkLoRA bridges a critical gap, enabling secure and trustworthy deployment of LLMs in sensitive or untrusted environments. 

---
# Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection 

**Authors**: Weizhi Gao, Xiaorui Liu, Feiyi Wang, Dan Lu, Junqi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21228)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance in both research and real-world applications, but they still struggle with hallucination. Existing hallucination detection methods often perform poorly on sentence-level generation or rely heavily on domain-specific knowledge. While self-consistency approaches help address these limitations, they incur high computational costs due to repeated generation. In this paper, we conduct the first study on identifying redundancy in self-consistency methods, manifested as shared prefix tokens across generations, and observe that non-exact-answer tokens contribute minimally to the semantic content. Based on these insights, we propose a novel Decoding Memory Pipeline (DMP) that accelerates generation through selective inference and annealed decoding. Being orthogonal to the model, dataset, decoding strategy, and self-consistency baseline, our DMP consistently improves the efficiency of multi-response generation and holds promise for extension to alignment and reasoning tasks. Extensive experiments show that our method achieves up to a 3x speedup without sacrificing AUROC performance. 

---
# Automating the Deep Space Network Data Systems; A Case Study in Adaptive Anomaly Detection through Agentic AI 

**Authors**: Evan J. Chou, Lisa S. Locke, Harvey M. Soldan  

**Link**: [PDF](https://arxiv.org/pdf/2508.21111)  

**Abstract**: The Deep Space Network (DSN) is NASA's largest network of antenna facilities that generate a large volume of multivariate time-series data. These facilities contain DSN antennas and transmitters that undergo degradation over long periods of time, which may cause costly disruptions to the data flow and threaten the earth-connection of dozens of spacecraft that rely on the Deep Space Network for their lifeline. The purpose of this study was to experiment with different methods that would be able to assist JPL engineers with directly pinpointing anomalies and equipment degradation through collected data, and continue conducting maintenance and operations of the DSN for future space missions around our universe. As such, we have researched various machine learning techniques that can fully reconstruct data through predictive analysis, and determine anomalous data entries within real-time datasets through statistical computations and thresholds. On top of the fully trained and tested machine learning models, we have also integrated the use of a reinforcement learning subsystem that classifies identified anomalies based on severity level and a Large Language Model that labels an explanation for each anomalous data entry, all of which can be improved and fine-tuned over time through human feedback/input. Specifically, for the DSN transmitters, we have also implemented a full data pipeline system that connects the data extraction, parsing, and processing workflow all together as there was no coherent program or script for performing these tasks before. Using this data pipeline system, we were able to then also connect the models trained from DSN antenna data, completing the data workflow for DSN anomaly detection. This was all wrapped around and further connected by an agentic AI system, where complex reasoning was utilized to determine the classifications and predictions of anomalous data. 

---
# R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and Reinforce Learning 

**Authors**: Jie Jiang, Qi Yang, Bolin Ni, Shiming Xiang, Han Hu, Houwen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21113)  

**Abstract**: Multimodal Large Language Models (MLLMs) equipped with step-by-step thinking capabilities have demonstrated remarkable performance on complex reasoning problems. However, this thinking process is redundant for simple problems solvable without complex reasoning. To address this inefficiency, we propose R-4B, an auto-thinking MLLM, which can adaptively decide when to think based on problem complexity. The central idea of R-4B is to empower the model with both thinking and non-thinking capabilities using bi-mode annealing, and apply Bi-mode Policy Optimization~(BPO) to improve the model's accuracy in determining whether to activate the thinking process. Specifically, we first train the model on a carefully curated dataset spanning various topics, which contains samples from both thinking and non-thinking modes. Then it undergoes a second phase of training under an improved GRPO framework, where the policy model is forced to generate responses from both modes for each input query. Experimental results show that R-4B achieves state-of-the-art performance across 25 challenging benchmarks. It outperforms Qwen2.5-VL-7B in most tasks and achieves performance comparable to larger models such as Kimi-VL-A3B-Thinking-2506 (16B) on reasoning-intensive benchmarks with lower computational cost. 

---
# Model-Driven Quantum Code Generation Using Large Language Models and Retrieval-Augmented Generation 

**Authors**: Nazanin Siavash, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21097)  

**Abstract**: This paper introduces a novel research direction for model-to-text/code transformations by leveraging Large Language Models (LLMs) that can be enhanced with Retrieval-Augmented Generation (RAG) pipelines. The focus is on quantum and hybrid quantum-classical software systems, where model-driven approaches can help reduce the costs and mitigate the risks associated with the heterogeneous platform landscape and lack of developers' skills. We validate one of the proposed ideas regarding generating code out of UML model instances of software systems. This Python code uses a well-established library, called Qiskit, to execute on gate-based or circuit-based quantum computers. The RAG pipeline that we deploy incorporates sample Qiskit code from public GitHub repositories. Experimental results show that well-engineered prompts can improve CodeBLEU scores by up to a factor of four, yielding more accurate and consistent quantum code. However, the proposed research direction can go beyond this through further investigation in the future by conducting experiments to address our other research questions and ideas proposed here, such as deploying software system model instances as the source of information in the RAG pipelines, or deploying LLMs for code-to-code transformations, for instance, for transpilation use cases. 

---
# Learning to Generate Unit Test via Adversarial Reinforcement Learning 

**Authors**: Dongjun Lee, Changho Hwang, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.21107)  

**Abstract**: Unit testing is a core practice in programming, enabling systematic evaluation of programs produced by human developers or large language models (LLMs). Given the challenges in writing comprehensive unit tests, LLMs have been employed to automate test generation, yet methods for training LLMs to produce high-quality tests remain underexplored. In this work, we propose UTRL, a novel reinforcement learning framework that trains an LLM to generate high-quality unit tests given a programming instruction. Our key idea is to iteratively train two LLMs, the unit test generator and the code generator, in an adversarial manner via reinforcement learning. The unit test generator is trained to maximize a discrimination reward, which reflects its ability to produce tests that expose faults in the code generator's solutions, and the code generator is trained to maximize a code reward, which reflects its ability to produce solutions that pass the unit tests generated by the test generator. In our experiments, we demonstrate that unit tests generated by Qwen3-4B trained via UTRL show higher quality compared to unit tests generated by the same model trained via supervised fine-tuning on human-written ground-truth unit tests, yielding code evaluations that more closely align with those induced by the ground-truth tests. Moreover, Qwen3-4B trained with UTRL outperforms frontier models such as GPT-4.1 in generating high-quality unit tests, highlighting the effectiveness of UTRL in training LLMs for this task. 

---
# Towards On-Device Personalization: Cloud-device Collaborative Data Augmentation for Efficient On-device Language Model 

**Authors**: Zhaofeng Zhong, Wei Yuan, Liang Qu, Tong Chen, Hao Wang, Xiangyu Zhao, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21313)  

**Abstract**: With the advancement of large language models (LLMs), significant progress has been achieved in various Natural Language Processing (NLP) tasks. However, existing LLMs still face two major challenges that hinder their broader adoption: (1) their responses tend to be generic and lack personalization tailored to individual users, and (2) they rely heavily on cloud infrastructure due to intensive computational requirements, leading to stable network dependency and response delay. Recent research has predominantly focused on either developing cloud-based personalized LLMs or exploring the on-device deployment of general-purpose LLMs. However, few studies have addressed both limitations simultaneously by investigating personalized on-device language models. To bridge this gap, we propose CDCDA-PLM, a framework for deploying personalized on-device language models on user devices with support from a powerful cloud-based LLM. Specifically, CDCDA-PLM leverages the server-side LLM's strong generalization capabilities to augment users' limited personal data, mitigating the issue of data scarcity. Using both real and synthetic data, A personalized on-device language models (LMs) is fine-tuned via parameter-efficient fine-tuning (PEFT) modules and deployed on users' local devices, enabling them to process queries without depending on cloud-based LLMs. This approach eliminates reliance on network stability and ensures high response speeds. Experimental results across six tasks in a widely used personalization benchmark demonstrate the effectiveness of CDCDA-PLM. 

---
# Geospatial Question Answering on Historical Maps Using Spatio-Temporal Knowledge Graphs and Large Language Models 

**Authors**: Ziyi Liu, Sidi Wu, Lorenz Hurni  

**Link**: [PDF](https://arxiv.org/pdf/2508.21491)  

**Abstract**: Recent advances have enabled the extraction of vectorized features from digital historical maps. To fully leverage this information, however, the extracted features must be organized in a structured and meaningful way that supports efficient access and use. One promising approach is question answering (QA), which allows users -- especially those unfamiliar with database query languages -- to retrieve knowledge in a natural and intuitive manner. In this project, we developed a GeoQA system by integrating a spatio-temporal knowledge graph (KG) constructed from historical map data with large language models (LLMs). Specifically, we have defined the ontology to guide the construction of the spatio-temporal KG and investigated workflows of two different types of GeoQA: factual and descriptive. Additional data sources, such as historical map images and internet search results, are incorporated into our framework to provide extra context for descriptive GeoQA. Evaluation results demonstrate that the system can generate answers with a high delivery rate and a high semantic accuracy. To make the framework accessible, we further developed a web application that supports interactive querying and visualization. 

---
# DMGIN: How Multimodal LLMs Enhance Large Recommendation Models for Lifelong User Post-click Behaviors 

**Authors**: Zhuoxing Wei, Qingchen Xie, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21801)  

**Abstract**: Modeling user interest based on lifelong user behavior sequences is crucial for enhancing Click-Through Rate (CTR) prediction. However, long post-click behavior sequences themselves pose severe performance issues: the sheer volume of data leads to high computational costs and inefficiencies in model training and inference. Traditional methods address this by introducing two-stage approaches, but this compromises model effectiveness due to incomplete utilization of the full sequence context. More importantly, integrating multimodal embeddings into existing large recommendation models (LRM) presents significant challenges: These embeddings often exacerbate computational burdens and mismatch with LRM architectures. To address these issues and enhance the model's efficiency and accuracy, we introduce Deep Multimodal Group Interest Network (DMGIN). Given the observation that user post-click behavior sequences contain a large number of repeated items with varying behaviors and timestamps, DMGIN employs Multimodal LLMs(MLLM) for grouping to reorganize complete lifelong post-click behavior sequences more effectively, with almost no additional computational overhead, as opposed to directly introducing multimodal embeddings. To mitigate the potential information loss from grouping, we have implemented two key strategies. First, we analyze behaviors within each group using both interest statistics and intra-group transformers to capture group traits. Second, apply inter-group transformers to temporally ordered groups to capture the evolution of user group interests. Our extensive experiments on both industrial and public datasets confirm the effectiveness and efficiency of DMGIN. The A/B test in our LBS advertising system shows that DMGIN improves CTR by 4.7% and Revenue per Mile by 2.3%. 

---
