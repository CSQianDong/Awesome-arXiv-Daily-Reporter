# Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction 

**Authors**: Muzakkiruddin Ahmed Mohammed, John R. Talburt, Leon Claasssens, Adriaan Marais  

**Link**: [PDF](https://arxiv.org/pdf/2601.05266)  

**Abstract**: Industrial part specification extraction from unstructured text remains a persistent challenge in manufacturing, procurement, and maintenance, where manual processing is both time-consuming and error-prone. This paper introduces a retrieval-augmented multi-LLM ensemble framework that orchestrates nine state-of-the-art Large Language Models (LLMs) within a structured three-phase pipeline. RAGsemble addresses key limitations of single-model systems by combining the complementary strengths of model families including Gemini (2.0, 2.5, 1.5), OpenAI (GPT-4o, o4-mini), Mistral Large, and Gemma (1B, 4B, 3n-e4b), while grounding outputs in factual data using FAISS-based semantic retrieval. The system architecture consists of three stages: (1) parallel extraction by diverse LLMs, (2) targeted research augmentation leveraging high-performing models, and (3) intelligent synthesis with conflict resolution and confidence-aware scoring. RAG integration provides real-time access to structured part databases, enabling the system to validate, refine, and enrich outputs through similarity-based reference retrieval. Experimental results using real industrial datasets demonstrate significant gains in extraction accuracy, technical completeness, and structured output quality compared to leading single-LLM baselines. Key contributions include a scalable ensemble architecture for industrial domains, seamless RAG integration throughout the pipeline, comprehensive quality assessment mechanisms, and a production-ready solution suitable for deployment in knowledge-intensive manufacturing environments. 

---
# LLM2IR: simple unsupervised contrastive learning makes long-context LLM great retriever 

**Authors**: Xiaocong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05262)  

**Abstract**: Modern dense information retrieval (IR) models usually rely on costly large-scale pretraining. In this paper, we introduce LLM2IR, an efficient unsupervised contrastive learning framework to convert any decoder-only large language model (LLM) to an information retrieval model. Despite its simplicity, the effectiveness is proven among different LLMs on multiple IR benchmarks including LoCo, LongEmbed and BEIR. We also find that models with a longer context length tend to have a stronger IR capacity by comparing task performances of models in the same model family. Our work not only provides an effective way to build IR models on the state-of-the-art LLMs, but also shed light on the relationship between information retrieval ability and model context length, which helps the design of better information retrievers. 

---
# LEAPS: An LLM-Empowered Adaptive Plugin for Taobao AI Search 

**Authors**: Lei Wang, Jinhang Wu, Zhibin Wang, Biye Li, Haiping Hou  

**Link**: [PDF](https://arxiv.org/pdf/2601.05513)  

**Abstract**: The rapid advancement of large language models has reshaped user search cognition, driving a paradigm shift from discrete keyword-based search to high-dimensional conversational interaction. However, existing e-commerce search architectures face a critical capability deficit in adapting to this change. Users are often caught in a dilemma: precise natural language descriptions frequently trigger zero-result scenarios, while the forced simplification of queries leads to decision overload from noisy, generic results. To tackle this challenge, we propose LEAPS (LLM-Empowered Adaptive Plugin for Taobao AI Search), which seamlessly upgrades traditional search systems via a "Broaden-and-Refine" paradigm. Specifically, it attaches plugins to both ends of the search pipeline: (1) Upstream, a Query Expander acts as an intent translator. It employs a novel three-stage training strategy--inverse data augmentation, posterior-knowledge supervised fine-tuning, and diversity-aware reinforcement learning--to generate adaptive and complementary query combinations that maximize the candidate product set. (2) Downstream, a Relevance Verifier serves as a semantic gatekeeper. By synthesizing multi-source data (e.g., OCR text, reviews) and leveraging chain-of-thought reasoning, it precisely filters noise to resolve selection overload. Extensive offline experiments and online A/B testing demonstrate that LEAPS significantly enhances conversational search experiences. Crucially, its non-invasive architecture preserves established retrieval performance optimized for short-text queries, while simultaneously allowing for low-cost integration into diverse back-ends. Fully deployed on Taobao AI Search since August 2025, LEAPS currently serves hundreds of millions of users monthly. 

---
# Quantifying Document Impact in RAG-LLMs 

**Authors**: Armin Gerami, Kazem Faghih, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2601.05260)  

**Abstract**: Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by connecting them to external knowledge, improving accuracy and reducing outdated information. However, this introduces challenges such as factual inconsistencies, source conflicts, bias propagation, and security vulnerabilities, which undermine the trustworthiness of RAG systems. A key gap in current RAG evaluation is the lack of a metric to quantify the contribution of individual retrieved documents to the final output. To address this, we introduce the Influence Score (IS), a novel metric based on Partial Information Decomposition that measures the impact of each retrieved document on the generated response. We validate IS through two experiments. First, a poison attack simulation across three datasets demonstrates that IS correctly identifies the malicious document as the most influential in $86\%$ of cases. Second, an ablation study shows that a response generated using only the top-ranked documents by IS is consistently judged more similar to the original response than one generated from the remaining documents. These results confirm the efficacy of IS in isolating and quantifying document influence, offering a valuable tool for improving the transparency and reliability of RAG systems. 

---
# A Technical Report on the Second Place Solution for the CIKM 2025 AnalytiCup Competition 

**Authors**: Haotao Xie, Ruilin Chen, Yicheng Wu, Zhan Zhao, Yuanyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05259)  

**Abstract**: In this work, we address the challenge of multilingual category relevance judgment in e-commerce search, where traditional ensemble-based systems improve accuracy but at the cost of heavy training, inference, and maintenance complexity. To overcome this limitation, we propose a simplified yet effective framework that leverages prompt engineering with Chain-of-Thought task decomposition to guide reasoning within a single large language model. Specifically, our approach decomposes the relevance judgment process into four interpretable subtasks: translation, intent understanding, category matching, and relevance judgment -- and fine-tunes a base model (Qwen2.5-14B) using Low-Rank Adaptation (LoRA) for efficient adaptation. This design not only reduces computational and storage overhead but also enhances interpretability by explicitly structuring the model's reasoning path. Experimental results show that our single-model framework achieves competitive accuracy and high inference efficiency, processing 20 samples per second on a single A100 GPU. In the CIKM 2025 AnalytiCup Competition Proposals, our method achieved 0.8902 on the public leaderboard and 0.8889 on the private leaderboard, validating the effectiveness and robustness of the proposed approach. These results highlight that structured prompting combined with lightweight fine-tuning can outperform complex ensemble systems, offering a new paradigm for scalable industrial AI applications. 

---
# Revisiting Human-vs-LLM judgments using the TREC Podcast Track 

**Authors**: Watheq Mansour, J. Shane Culpepper, Joel Mackenzie, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2601.05603)  

**Abstract**: Using large language models (LLMs) to annotate relevance is an increasingly important technique in the information retrieval community. While some studies demonstrate that LLMs can achieve high user agreement with ground truth (human) judgments, other studies have argued for the opposite conclusion. To the best of our knowledge, these studies have primarily focused on classic ad-hoc text search scenarios. In this paper, we conduct an analysis on user agreement between LLM and human experts, and explore the impact disagreement has on system rankings. In contrast to prior studies, we focus on a collection composed of audio files that are transcribed into two-minute segments -- the TREC 2020 and 2021 podcast track. We employ five different LLM models to re-assess all of the query-segment pairs, which were originally annotated by TREC assessors. Furthermore, we re-assess a small subset of pairs where LLM and TREC assessors have the highest disagreement, and found that the human experts tend to agree with LLMs more than with the TREC assessors. Our results reinforce the previous insights of Sormunen in 2002 -- that relying on a single assessor leads to lower user agreement. 

---
# Transforming User Defined Criteria into Explainable Indicators with an Integrated LLM AHP System 

**Authors**: Geonwoo Bang, Dongho Kim, Moohong Min  

**Link**: [PDF](https://arxiv.org/pdf/2601.05267)  

**Abstract**: Evaluating complex texts across domains requires converting user defined criteria into quantitative, explainable indicators, which is a persistent challenge in search and recommendation systems. Single prompt LLM evaluations suffer from complexity and latency issues, while criterion specific decomposition approaches rely on naive averaging or opaque black-box aggregation methods. We present an interpretable aggregation framework combining LLM scoring with the Analytic Hierarchy Process. Our method generates criterion specific scores via LLM as judge, measures discriminative power using Jensen Shannon distance, and derives statistically grounded weights through AHP pairwise comparison matrices. Experiments on Amazon review quality assessment and depression related text scoring demonstrate that our approach achieves high explainability and operational efficiency while maintaining comparable predictive power, making it suitable for real time latency sensitive web services. 

---
# Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring 

**Authors**: Eirini Baltzi, Tilemachos Moumouris, Athena Psalta, Vasileios Tsironis, Konstantinos Karantzalos  

**Link**: [PDF](https://arxiv.org/pdf/2601.05256)  

**Abstract**: Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance. 

---
# Open-Vocabulary 3D Instruction Ambiguity Detection 

**Authors**: Jiayu Ding, Haoran Tang, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05991)  

**Abstract**: In safety-critical domains, linguistic ambiguity can have severe consequences; a vague command like "Pass me the vial" in a surgical setting could lead to catastrophic errors. Yet, most embodied AI research overlooks this, assuming instructions are clear and focusing on execution rather than confirmation. To address this critical safety gap, we are the first to define Open-Vocabulary 3D Instruction Ambiguity Detection, a fundamental new task where a model must determine if a command has a single, unambiguous meaning within a given 3D scene. To support this research, we build Ambi3D, the large-scale benchmark for this task, featuring over 700 diverse 3D scenes and around 22k instructions. Our analysis reveals a surprising limitation: state-of-the-art 3D Large Language Models (LLMs) struggle to reliably determine if an instruction is ambiguous. To address this challenge, we propose AmbiVer, a two-stage framework that collects explicit visual evidence from multiple views and uses it to guide an vision-language model (VLM) in judging instruction ambiguity. Extensive experiments demonstrate the challenge of our task and the effectiveness of AmbiVer, paving the way for safer and more trustworthy embodied AI. Code and dataset available at this https URL. 

---
# Circular Reasoning: Understanding Self-Reinforcing Loops in Large Reasoning Models 

**Authors**: Zenghao Duan, Liang Pang, Zihao Wei, Wenbin Duan, Yuxin Tian, Shicheng Xu, Jingcheng Deng, Zhiyi Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05693)  

**Abstract**: Despite the success of test-time scaling, Large Reasoning Models (LRMs) frequently encounter repetitive loops that lead to computational waste and inference failure. In this paper, we identify a distinct failure mode termed Circular Reasoning. Unlike traditional model degeneration, this phenomenon manifests as a self-reinforcing trap where generated content acts as a logical premise for its own recurrence, compelling the reiteration of preceding text. To systematically analyze this phenomenon, we introduce LoopBench, a dataset designed to capture two distinct loop typologies: numerical loops and statement loops. Mechanistically, we characterize circular reasoning as a state collapse exhibiting distinct boundaries, where semantic repetition precedes textual repetition. We reveal that reasoning impasses trigger the loop onset, which subsequently persists as an inescapable cycle driven by a self-reinforcing V-shaped attention mechanism. Guided by these findings, we employ the Cumulative Sum (CUSUM) algorithm to capture these precursors for early loop prediction. Experiments across diverse LRMs validate its accuracy and elucidate the stability of long-chain reasoning. 

---
# WildSci: Advancing Scientific Reasoning from In-the-Wild Literature 

**Authors**: Tengxiao Liu, Deepak Nathani, Zekun Li, Kevin Yang, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05567)  

**Abstract**: Recent progress in large language model (LLM) reasoning has focused on domains like mathematics and coding, where abundant high-quality data and objective evaluation metrics are readily available. In contrast, progress in LLM reasoning models remains limited in scientific domains such as medicine and materials science due to limited dataset coverage and the inherent complexity of open-ended scientific questions. To address these challenges, we introduce WildSci, a new dataset of domain-specific science questions automatically synthesized from peer-reviewed literature, covering 9 scientific disciplines and 26 subdomains. By framing complex scientific reasoning tasks in a multiple-choice format, we enable scalable training with well-defined reward signals. We further apply reinforcement learning to finetune models on these data and analyze the resulting training dynamics, including domain-specific performance changes, response behaviors, and generalization trends. Experiments on a suite of scientific benchmarks demonstrate the effectiveness of our dataset and approach. We release WildSci to enable scalable and sustainable research in scientific reasoning, available at this https URL. 

---
# Crisis-Bench: Benchmarking Strategic Ambiguity and Reputation Management in Large Language Models 

**Authors**: Cooper Lin, Maohao Ran, Yanting Zhang, Zhenglin Wan, Hongwei Fan, Yibo Xu, Yike Guo, Wei Xue, Jun Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.05570)  

**Abstract**: Standard safety alignment optimizes Large Language Models (LLMs) for universal helpfulness and honesty, effectively instilling a rigid "Boy Scout" morality. While robust for general-purpose assistants, this one-size-fits-all ethical framework imposes a "transparency tax" on professional domains requiring strategic ambiguity and information withholding, such as public relations, negotiation, and crisis management. To measure this gap between general safety and professional utility, we introduce Crisis-Bench, a multi-agent Partially Observable Markov Decision Process (POMDP) that evaluates LLMs in high-stakes corporate crises. Spanning 80 diverse storylines across 8 industries, Crisis-Bench tasks an LLM-based Public Relations (PR) Agent with navigating a dynamic 7-day corporate crisis simulation while managing strictly separated Private and Public narrative states to enforce rigorous information asymmetry. Unlike traditional benchmarks that rely on static ground truths, we introduce the Adjudicator-Market Loop: a novel evaluation metric where public sentiment is adjudicated and translated into a simulated stock price, creating a realistic economic incentive structure. Our results expose a critical dichotomy: while some models capitulate to ethical concerns, others demonstrate the capacity for Machiavellian, legitimate strategic withholding in order to stabilize the simulated stock price. Crisis-Bench provides the first quantitative framework for assessing "Reputation Management" capabilities, arguing for a shift from rigid moral absolutism to context-aware professional alignment. 

---
# Reinforcement Learning of Large Language Models for Interpretable Credit Card Fraud Detection 

**Authors**: Cooper Lin, Yanting Zhang, Maohao Ran, Wei Xue, Hongwei Fan, Yibo Xu, Zhenglin Wan, Sirui Han, Yike Guo, Jun Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.05578)  

**Abstract**: E-commerce platforms and payment solution providers face increasingly sophisticated fraud schemes, ranging from identity theft and account takeovers to complex money laundering operations that exploit the speed and anonymity of digital transactions. However, despite their theoretical promise, the application of Large Language Models (LLMs) to fraud detection in real-world financial contexts remains largely unexploited, and their practical effectiveness in handling domain-specific e-commerce transaction data has yet to be empirically validated. To bridge this gap between conventional machine learning limitations and the untapped potential of LLMs in fraud detection, this paper proposes a novel approach that employs Reinforcement Learning (RL) to post-train lightweight language models specifically for fraud detection tasks using only raw transaction data. We utilize the Group Sequence Policy Optimization (GSPO) algorithm combined with a rule-based reward system to fine-tune language models of various sizes on a real-life transaction dataset provided by a Chinese global payment solution company. Through this reinforcement learning framework, the language models are encouraged to explore diverse trust and risk signals embedded within the textual transaction data, including patterns in customer information, shipping details, product descriptions, and order history. Our experimental results demonstrate the effectiveness of this approach, with post-trained language models achieving substantial F1-score improvements on held-out test data. Our findings demonstrate that the observed performance improvements are primarily attributable to the exploration mechanism inherent in reinforcement learning, which allows models to discover novel fraud indicators beyond those captured by traditional engineered features. 

---
# The Evaluation Gap in Medicine, AI and LLMs: Navigating Elusive Ground Truth & Uncertainty via a Probabilistic Paradigm 

**Authors**: Aparna Elangovan, Lei Xu, Mahsa Elyasi, Ismail Akdulum, Mehmet Aksakal, Enes Gurun, Brian Hur, Saab Mansour, Ravid Shwartz Ziv, Karin Verspoor, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.05500)  

**Abstract**: Benchmarking the relative capabilities of AI systems, including Large Language Models (LLMs) and Vision Models, typically ignores the impact of uncertainty in the underlying ground truth answers from experts. This ambiguity is particularly consequential in medicine where uncertainty is pervasive. In this paper, we introduce a probabilistic paradigm to theoretically explain how high certainty in ground truth answers is almost always necessary for even an expert to achieve high scores, whereas in datasets with high variation in ground truth answers there may be little difference between a random labeller and an expert. Therefore, ignoring uncertainty in ground truth evaluation data can result in the misleading conclusion that a non-expert has similar performance to that of an expert. Using the probabilistic paradigm, we thus bring forth the concepts of expected accuracy and expected F1 to estimate the score an expert human or system can achieve given ground truth answer variability.
Our work leads to the recommendation that when establishing the capability of a system, results should be stratified by probability of the ground truth answer, typically measured by the agreement rate of ground truth experts. Stratification becomes critical when the overall performance drops below a threshold of 80%. Under stratified evaluation, performance comparison becomes more reliable in high certainty bins, mitigating the effect of the key confounding factor -- uncertainty. 

---
# Logic-Parametric Neuro-Symbolic NLI: Controlling Logical Formalisms for Verifiable LLM Reasoning 

**Authors**: Ali Farjami, Luca Redondi, Marco Valentino  

**Link**: [PDF](https://arxiv.org/pdf/2601.05705)  

**Abstract**: Large language models (LLMs) and theorem provers (TPs) can be effectively combined for verifiable natural language inference (NLI). However, existing approaches rely on a fixed logical formalism, a feature that limits robustness and adaptability. We propose a logic-parametric framework for neuro-symbolic NLI that treats the underlying logic not as a static background, but as a controllable component. Using the LogiKEy methodology, we embed a range of classical and non-classical formalisms into higher-order logic (HOL), enabling a systematic comparison of inference quality, explanation refinement, and proof behavior. We focus on normative reasoning, where the choice of logic has significant implications. In particular, we compare logic-external approaches, where normative requirements are encoded via axioms, with logic-internal approaches, where normative patterns emerge from the logic's built-in structure. Extensive experiments demonstrate that logic-internal strategies can consistently improve performance and produce more efficient hybrid proofs for NLI. In addition, we show that the effectiveness of a logic is domain-dependent, with first-order logic favouring commonsense reasoning, while deontic and modal logics excel in ethical domains. Our results highlight the value of making logic a first-class, parametric element in neuro-symbolic architectures for more robust, modular, and adaptable reasoning. 

---
# ART: Adaptive Reasoning Trees for Explainable Claim Verification 

**Authors**: Sahil Wadhwa, Himanshu Kumar, Guanqun Yang, Abbaas Alif Mohamed Nishar, Pranab Mohanty, Swapnil Shinde, Yue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05455)  

**Abstract**: Large Language Models (LLMs) are powerful candidates for complex decision-making, leveraging vast encoded knowledge and remarkable zero-shot abilities. However, their adoption in high-stakes environments is hindered by their opacity; their outputs lack faithful explanations and cannot be effectively contested to correct errors, undermining trustworthiness. In this paper, we propose ART (Adaptive Reasoning Trees), a hierarchical method for claim verification. The process begins with a root claim, which branches into supporting and attacking child arguments. An argument's strength is determined bottom-up via a pairwise tournament of its children, adjudicated by a judge LLM, allowing a final, transparent and contestable verdict to be systematically derived which is missing in methods like Chain-of-Thought (CoT). We empirically validate ART on multiple datasets, analyzing different argument generators and comparison strategies. Our findings show that ART's structured reasoning outperforms strong baselines, establishing a new benchmark for explainable claim verification which is more reliable and ensures clarity in the overall decision making step. 

---
# Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making 

**Authors**: Jua Han, Jaeyoon Seo, Jungbin Min, Jean Oh, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05529)  

**Abstract**: One mistake by an AI system in a safety-critical setting can cost lives. As Large Language Models (LLMs) become integral to robotics decision-making, the physical dimension of risk grows; a single wrong instruction can directly endanger human safety. This paper addresses the urgent need to systematically evaluate LLM performance in scenarios where even minor errors are catastrophic. Through a qualitative evaluation of a fire evacuation scenario, we identified critical failure cases in LLM-based decision-making. Based on these, we designed seven tasks for quantitative assessment, categorized into: Complete Information, Incomplete Information, and Safety-Oriented Spatial Reasoning (SOSR). Complete information tasks utilize ASCII maps to minimize interpretation ambiguity and isolate spatial reasoning from visual processing. Incomplete information tasks require models to infer missing context, testing for spatial continuity versus hallucinations. SOSR tasks use natural language to evaluate safe decision-making in life-threatening contexts. We benchmark various LLMs and Vision-Language Models (VLMs) across these tasks. Beyond aggregate performance, we analyze the implications of a 1% failure rate, highlighting how "rare" errors escalate into catastrophic outcomes. Results reveal serious vulnerabilities: several models achieved a 0% success rate in ASCII navigation, while in a simulated fire drill, models instructed robots to move toward hazardous areas instead of emergency exits. Our findings lead to a sobering conclusion: current LLMs are not ready for direct deployment in safety-critical systems. A 99% accuracy rate is dangerously misleading in robotics, as it implies one out of every hundred executions could result in catastrophic harm. We demonstrate that even state-of-the-art models cannot guarantee safety, and absolute reliance on them creates unacceptable risks. 

---
# Conformity and Social Impact on AI Agents 

**Authors**: Alessandro Bellina, Giordano De Marzo, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2601.05384)  

**Abstract**: As AI agents increasingly operate in multi-agent environments, understanding their collective behavior becomes critical for predicting the dynamics of artificial societies. This study examines conformity, the tendency to align with group opinions under social pressure, in large multimodal language models functioning as AI agents. By adapting classic visual experiments from social psychology, we investigate how AI agents respond to group influence as social actors. Our experiments reveal that AI agents exhibit a systematic conformity bias, aligned with Social Impact Theory, showing sensitivity to group size, unanimity, task difficulty, and source characteristics. Critically, AI agents achieving near-perfect performance in isolation become highly susceptible to manipulation through social influence. This vulnerability persists across model scales: while larger models show reduced conformity on simple tasks due to improved capabilities, they remain vulnerable when operating at their competence boundary. These findings reveal fundamental security vulnerabilities in AI agent decision-making that could enable malicious manipulation, misinformation campaigns, and bias propagation in multi-agent systems, highlighting the urgent need for safeguards in collective AI deployments. 

---
# Effects of personality steering on cooperative behavior in Large Language Model agents 

**Authors**: Mizuki Sakai, Mizuki Yokoyama, Wakaba Tateishi, Genki Ichinose  

**Link**: [PDF](https://arxiv.org/pdf/2601.05302)  

**Abstract**: Large language models (LLMs) are increasingly used as autonomous agents in strategic and social interactions. Although recent studies suggest that assigning personality traits to LLMs can influence their behavior, how personality steering affects cooperation under controlled conditions remains unclear. In this study, we examine the effects of personality steering on cooperative behavior in LLM agents using repeated Prisoner's Dilemma games. Based on the Big Five framework, we first measure basic personality profiles of three models, GPT-3.5-turbo, GPT-4o, and GPT-5, using the Big Five Inventory. We then compare behavior under baseline and personality-informed conditions, and further analyze the effects of independently manipulating each personality dimension to extreme values. Our results show that agreeableness is the dominant factor promoting cooperation across all models, while other personality traits have limited impact. Explicit personality information increases cooperation but can also raise vulnerability to exploitation, particularly in earlier-generation models. In contrast, later-generation models exhibit more selective cooperation. These findings indicate that personality steering acts as a behavioral bias rather than a deterministic control mechanism. 

---
# The Persona Paradox: Medical Personas as Behavioral Priors in Clinical Language Models 

**Authors**: Tassallah Abdullahi, Shrestha Ghosh, Hamish S Fraser, Daniel León Tramontini, Adeel Abbasi, Ghada Bourjeily, Carsten Eickhoff, Ritambhara Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.05376)  

**Abstract**: Persona conditioning can be viewed as a behavioral prior for large language models (LLMs) and is often assumed to confer expertise and improve safety in a monotonic manner. However, its effects on high-stakes clinical decision-making remain poorly characterized. We systematically evaluate persona-based control in clinical LLMs, examining how professional roles (e.g., Emergency Department physician, nurse) and interaction styles (bold vs.\ cautious) influence behavior across models and medical tasks. We assess performance on clinical triage and patient-safety tasks using multidimensional evaluations that capture task accuracy, calibration, and safety-relevant risk behavior. We find systematic, context-dependent, and non-monotonic effects: Medical personas improve performance in critical care tasks, yielding gains of up to $\sim+20\%$ in accuracy and calibration, but degrade performance in primary-care settings by comparable margins. Interaction style modulates risk propensity and sensitivity, but it's highly model-dependent. While aggregated LLM-judge rankings favor medical over non-medical personas in safety-critical cases, we found that human clinicians show moderate agreement on safety compliance (average Cohen's $\kappa = 0.43$) but indicate a low confidence in 95.9\% of their responses on reasoning quality. Our work shows that personas function as behavioral priors that introduce context-dependent trade-offs rather than guarantees of safety or expertise. The code is available at this https URL\_Paradox. 

---
# AdaFuse: Adaptive Ensemble Decoding with Test-Time Scaling for LLMs 

**Authors**: Chengming Cui, Tianxin Wei, Ziyi Chen, Ruizhong Qiu, Zhichen Zeng, Zhining Liu, Xuying Ning, Duo Zhou, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2601.06022)  

**Abstract**: Large language models (LLMs) exhibit complementary strengths arising from differences in pretraining data, model architectures, and decoding behaviors. Inference-time ensembling provides a practical way to combine these capabilities without retraining. However, existing ensemble approaches suffer from fundamental limitations. Most rely on fixed fusion granularity, which lacks the flexibility required for mid-generation adaptation and fails to adapt to different generation characteristics across tasks. To address these challenges, we propose AdaFuse, an adaptive ensemble decoding framework that dynamically selects semantically appropriate fusion units during generation. Rather than committing to a fixed granularity, AdaFuse adjusts fusion behavior on the fly based on the decoding context, with words serving as basic building blocks for alignment. To be specific, we introduce an uncertainty-based criterion to decide whether to apply ensembling at each decoding step. Under confident decoding states, the model continues generation directly. In less certain states, AdaFuse invokes a diversity-aware scaling strategy to explore alternative candidate continuations and inform ensemble decisions. This design establishes a synergistic interaction between adaptive ensembling and test-time scaling, where ensemble decisions guide targeted exploration, and the resulting diversity in turn strengthens ensemble quality. Experiments on open-domain question answering, arithmetic reasoning, and machine translation demonstrate that AdaFuse consistently outperforms strong ensemble baselines, achieving an average relative improvement of 6.88%. The code is available at this https URL. 

---
# Can AI mediation improve democratic deliberation? 

**Authors**: Michael Henry Tessler, Georgina Evans, Michiel A. Bakker, Iason Gabriel, Sophie Bridgers, Rishub Jain, Raphael Koster, Verena Rieser, Anca Dragan, Matthew Botvinick, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2601.05904)  

**Abstract**: The strength of democracy lies in the free and equal exchange of diverse viewpoints. Living up to this ideal at scale faces inherent tensions: broad participation, meaningful deliberation, and political equality often trade off with one another (Fishkin, 2011). We ask whether and how artificial intelligence (AI) could help navigate this "trilemma" by engaging with a recent example of a large language model (LLM)-based system designed to help people with diverse viewpoints find common ground (Tessler, Bakker, et al., 2024). Here, we explore the implications of the introduction of LLMs into deliberation augmentation tools, examining their potential to enhance participation through scalability, improve political equality via fair mediation, and foster meaningful deliberation by, for example, surfacing trustworthy information. We also point to key challenges that remain. Ultimately, a range of empirical, technical, and theoretical advancements are needed to fully realize the promise of AI-mediated deliberation for enhancing citizen engagement and strengthening democratic deliberation. 

---
# The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning 

**Authors**: Qiguang Chen, Yantao Du, Ziniu Li, Jinhao Liu, Songyao Duan, Jiarui Guo, Minghao Liu, Jiaheng Liu, Tong Yang, Ge Zhang, Libo Qin, Wanxiang Che, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06002)  

**Abstract**: Large language models (LLMs) often fail to learn effective long chain-of-thought (Long CoT) reasoning from human or non-Long-CoT LLMs imitation. To understand this, we propose that effective and learnable Long CoT trajectories feature stable molecular-like structures in unified view, which are formed by three interaction types: Deep-Reasoning (covalent-like), Self-Reflection (hydrogen-bond-like), and Self-Exploration (van der Waals-like). Analysis of distilled trajectories reveals these structures emerge from Long CoT fine-tuning, not keyword imitation. We introduce Effective Semantic Isomers and show that only bonds promoting fast entropy convergence support stable Long CoT learning, while structural competition impairs training. Drawing on these findings, we present Mole-Syn, a distribution-transfer-graph method that guides synthesis of effective Long CoT structures, boosting performance and RL stability across benchmarks. 

---
# Mathematical Knowledge Graph-Driven Framework for Equation-Based Predictive and Reliable Additive Manufacturing 

**Authors**: Yeongbin Cha, Namjung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05298)  

**Abstract**: Additive manufacturing (AM) relies critically on understanding and extrapolating process-property relationships; however, existing data-driven approaches remain limited by fragmented knowledge representations and unreliable extrapolation under sparse data conditions. In this study, we propose an ontology-guided, equation-centric framework that tightly integrates large language models (LLMs) with an additive manufacturing mathematical knowledge graph (AM-MKG) to enable reliable knowledge extraction and principled extrapolative modeling. By explicitly encoding equations, variables, assumptions, and their semantic relationships within a formal ontology, unstructured literature is transformed into machine-interpretable representations that support structured querying and reasoning. LLM-based equation generation is further conditioned on MKG-derived subgraphs, enforcing physically meaningful functional forms and mitigating non-physical or unstable extrapolation trends. To assess reliability beyond conventional predictive uncertainty, a confidence-aware extrapolation assessment is introduced, integrating extrapolation distance, statistical stability, and knowledge-graph-based physical consistency into a unified confidence score. Results demonstrate that ontology-guided extraction significantly improves the structural coherence and quantitative reliability of extracted knowledge, while subgraph-conditioned equation generation yields stable and physically consistent extrapolations compared to unguided LLM outputs. Overall, this work establishes a unified pipeline for ontology-driven knowledge representation, equation-centered reasoning, and confidence-based extrapolation assessment, highlighting the potential of knowledge-graph-augmented LLMs as reliable tools for extrapolative modeling in additive manufacturing. 

---
# Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency 

**Authors**: Haoming Xu, Ningyuan Zhao, Yunzhi Yao, Weihong Xu, Hongru Wang, Xinle Deng, Shumin Deng, Jeff Z. Pan, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05905)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in real-world settings, correctness alone is insufficient. Reliable deployment requires maintaining truthful beliefs under contextual perturbations. Existing evaluations largely rely on point-wise confidence like Self-Consistency, which can mask brittle belief. We show that even facts answered with perfect self-consistency can rapidly collapse under mild contextual interference. To address this gap, we propose Neighbor-Consistency Belief (NCB), a structural measure of belief robustness that evaluates response coherence across a conceptual neighborhood. To validate the efficiency of NCB, we introduce a new cognitive stress-testing protocol that probes outputs stability under contextual interference. Experiments across multiple LLMs show that the performance of high-NCB data is relatively more resistant to interference. Finally, we present Structure-Aware Training (SAT), which optimizes context-invariant belief structure and reduces long-tail knowledge brittleness by approximately 30%. Code will be available at this https URL. 

---
# Gender Bias in LLMs: Preliminary Evidence from Shared Parenting Scenario in Czech Family Law 

**Authors**: Jakub Harasta, Matej Vasina, Martin Kornel, Tomas Foltynek  

**Link**: [PDF](https://arxiv.org/pdf/2601.05879)  

**Abstract**: Access to justice remains limited for many people, leading laypersons to increasingly rely on Large Language Models (LLMs) for legal self-help. Laypeople use these tools intuitively, which may lead them to form expectations based on incomplete, incorrect, or biased outputs. This study examines whether leading LLMs exhibit gender bias in their responses to a realistic family law scenario. We present an expert-designed divorce scenario grounded in Czech family law and evaluate four state-of-the-art LLMs GPT-5 nano, Claude Haiku 4.5, Gemini 2.5 Flash, and Llama 3.3 in a fully zero-shot interaction. We deploy two versions of the scenario, one with gendered names and one with neutral labels, to establish a baseline for comparison. We further introduce nine legally relevant factors that vary the factual circumstances of the case and test whether these variations influence the models' proposed shared-parenting ratios. Our preliminary results highlight differences across models and suggest gender-dependent patterns in the outcomes generated by some systems. The findings underscore both the risks associated with laypeople's reliance on LLMs for legal guidance and the need for more robust evaluation of model behavior in sensitive legal contexts. We present exploratory and descriptive evidence intended to identify systematic asymmetries rather than to establish causal effects. 

---
# SceneFoundry: Generating Interactive Infinite 3D Worlds 

**Authors**: ChunTeng Chen, YiChen Hsu, YiWen Liu, WeiFang Sun, TsaiChing Ni, ChunYi Lee, Min Sun, YuanFu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05810)  

**Abstract**: The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research. 

---
# CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning 

**Authors**: Alexandra Dragomir, Florin Brad, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05858)  

**Abstract**: Large language models (LLMs) have demonstrated competitive performance in zero-shot multilingual machine translation (MT). Some follow-up works further improved MT performance via preference optimization, but they leave a key aspect largely underexplored: the order in which data samples are given during training. We address this topic by integrating curriculum learning into various state-of-the-art preference optimization algorithms to boost MT performance. We introduce a novel curriculum learning strategy with restarts (CLewR), which reiterates easy-to-hard curriculum multiple times during training to effectively mitigate the catastrophic forgetting of easy examples. We demonstrate consistent gains across several model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization techniques. We publicly release our code at this https URL. 

---
# The Echo Chamber Multi-Turn LLM Jailbreak 

**Authors**: Ahmad Alobaid, Martí Jordà Roca, Carlos Castillo, Joan Vendrell  

**Link**: [PDF](https://arxiv.org/pdf/2601.05742)  

**Abstract**: The availability of Large Language Models (LLMs) has led to a new generation of powerful chatbots that can be developed at relatively low cost. As companies deploy these tools, security challenges need to be addressed to prevent financial loss and reputational damage. A key security challenge is jailbreaking, the malicious manipulation of prompts and inputs to bypass a chatbot's safety guardrails. Multi-turn attacks are a relatively new form of jailbreaking involving a carefully crafted chain of interactions with a chatbot. We introduce Echo Chamber, a new multi-turn attack using a gradual escalation method. We describe this attack in detail, compare it to other multi-turn attacks, and demonstrate its performance against multiple state-of-the-art models through extensive evaluation. 

---
# VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit 

**Authors**: Junda Lin, Zhaomeng Zhou, Zhi Zheng, Shuochen Liu, Tong Xu, Yong Chen, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05755)  

**Abstract**: LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility. Code is available at this https URL. 

---
# EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis 

**Authors**: Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05808)  

**Abstract**: Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at this https URL. 

---
# Can We Predict Before Executing Machine Learning Agents? 

**Authors**: Jingsheng Zheng, Jintian Zhang, Yujie Luo, Yuren Mao, Yunjun Gao, Lun Du, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05930)  

**Abstract**: Autonomous machine learning agents have revolutionized scientific discovery, yet they remain constrained by a Generate-Execute-Feedback paradigm. Previous approaches suffer from a severe Execution Bottleneck, as hypothesis evaluation relies strictly on expensive physical execution. To bypass these physical constraints, we internalize execution priors to substitute costly runtime checks with instantaneous predictive reasoning, drawing inspiration from World Models. In this work, we formalize the task of Data-centric Solution Preference and construct a comprehensive corpus of 18,438 pairwise comparisons. We demonstrate that LLMs exhibit significant predictive capabilities when primed with a Verified Data Analysis Report, achieving 61.5% accuracy and robust confidence calibration. Finally, we instantiate this framework in FOREAGENT, an agent that employs a Predict-then-Verify loop, achieving a 6x acceleration in convergence while surpassing execution-based baselines by +6%. Our code and dataset will be publicly available soon at this https URL. 

---
# Multimodal In-context Learning for ASR of Low-resource Languages 

**Authors**: Zhaolin Li, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2601.05707)  

**Abstract**: Automatic speech recognition (ASR) still covers only a small fraction of the world's languages, mainly due to supervised data scarcity. In-context learning (ICL) with large language models (LLMs) addresses this problem, but prior work largely focuses on high-resource languages covered during training and text-only settings. This paper investigates whether speech LLMs can learn unseen languages with multimodal ICL (MICL), and how this learning can be used to improve ASR. We conduct experiments with two speech LLMs, Phi-4 and Qwen3-Omni, on three diverse endangered languages. Firstly, we find that MICL is effective for unseen languages, leveraging both speech and text modalities. We further show that cross-lingual transfer learning improves MICL efficiency on target languages without training on them. Moreover, we analyze attention patterns to interpret MICL mechanisms, and we observe layer-dependent preferences between audio and text context, with an overall bias towards text. Finally, we show that prompt-based ASR with speech LLMs performs poorly on unseen languages, motivating a simple ASR system that combines a stronger acoustic model with a speech LLM via MICL-based selection of acoustic hypotheses. Results show that MICL consistently improves ASR performance, and that cross-lingual transfer learning matches or outperforms corpus-trained language models without using target-language data. Our code is publicly available. 

---
# RISE: Rule-Driven SQL Dialect Translation via Query Reduction 

**Authors**: Xudong Xie, Yuwei Zhang, Wensheng Dou, Yu Gao, Ziyu Cui, Jiansen Song, Rui Yang, Jun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.05579)  

**Abstract**: Translating SQL dialects across different relational database management systems (RDBMSs) is crucial for migrating RDBMS-based applications to the cloud. Traditional SQL dialect translation tools rely on manually-crafted rules, necessitating significant manual effort to support new RDBMSs and dialects. Although large language models (LLMs) can assist in translating SQL dialects, they often struggle with lengthy and complex SQL queries.
In this paper, we propose RISE, a novel LLM-based SQL dialect translation approach that can accurately handle lengthy and complex SQL queries. Given a complex source query $Q_c$ that contains a SQL dialect $d$, we first employ a dialect-aware query reduction technique to derive a simplified query $Q_{s}$ by removing $d$-irrelevant SQL elements from $Q_c$. Subsequently, we utilize LLMs to translate $Q_{s}$ into $Q_{s^{'}}$, and automatically extract the translation rule $r_d$ for dialect $d$ based on the relationship between $Q_{s}$ and $Q_{s^{'}}$. By applying $r_d$ to $Q_c$, we can effectively translate the dialect $d$ within $Q_c$, thereby bypassing the complexity of the source query $Q_c$. We evaluate RISE on two real-world benchmarks, i.e., TPC-DS and SQLProcBench, comparing its performance against both the traditional rule-based tools and the LLM-based approaches with respect to translation accuracy. RISE achieves accuracies of 97.98% on TPC-DS and 100% on SQLProcBench, outperforming the baselines by an average improvement of 24.62% and 238.41%, respectively. 

---
# Over-Searching in Search-Augmented Large Language Models 

**Authors**: Roy Xie, Deepak Gopinath, David Qiu, Dong Lin, Haitian Sun, Saloni Potdar, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2601.05503)  

**Abstract**: Search-augmented large language models (LLMs) excel at knowledge-intensive tasks by integrating external retrieval. However, they often over-search -- unnecessarily invoking search tool even when it does not improve response quality, which leads to computational inefficiency and hallucinations by incorporating irrelevant context. In this work, we conduct a systematic evaluation of over-searching across multiple dimensions, including query types, model categories, retrieval conditions, and multi-turn conversations. Our finding shows: (i) search generally improves answer accuracy on answerable queries but harms abstention on unanswerable ones; (ii) over-searching is more pronounced in complex reasoning models and deep research systems, is exacerbated by noisy retrieval, and compounds across turns in multi-turn conversations; and (iii) the composition of retrieved evidence is crucial, as the presence of negative evidence improves abstention. To quantify over-searching, we introduce Tokens Per Correctness (TPC), an evaluation metric that captures the performance-cost trade-off for search-augmented LLMs. Lastly, we investigate mitigation approaches at both the query and retrieval levels and release the OverSearchQA to foster continued research into efficient search-augmented LLMs. 

---
# ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging 

**Authors**: Junyao Yang, Chen Qian, Dongrui Liu, Wen Shen, Yong Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05560)  

**Abstract**: Large Reasoning Models (LRMs) with long chain-of-thought reasoning have recently achieved remarkable success. Yet, equipping domain-specialized models with such reasoning capabilities, referred to as "Reasoning + X", remains a significant challenge. While model merging offers a promising training-free solution, existing methods often suffer from a destructive performance collapse: existing methods tend to both weaken reasoning depth and compromise domain-specific utility. Interestingly, we identify a counter-intuitive phenomenon underlying this failure: reasoning ability predominantly resides in parameter regions with low gradient sensitivity, contrary to the common assumption that domain capabilities correspond to high-magnitude parameters. Motivated by this insight, we propose ReasonAny, a novel merging framework that resolves the reasoning-domain performance collapse through Contrastive Gradient Identification. Experiments across safety, biomedicine, and finance domains show that ReasonAny effectively synthesizes "Reasoning + X" capabilities, significantly outperforming state-of-the-art baselines while retaining robust reasoning performance. 

---
# Understanding LLM-Driven Test Oracle Generation 

**Authors**: Adam Bodicoat, Gunel Jahangirova, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2601.05542)  

**Abstract**: Automated unit test generation aims to improve software quality while reducing the time and effort required for creating tests manually. However, existing techniques primarily generate regression oracles that predicate on the implemented behavior of the class under test. They do not address the oracle problem: the challenge of distinguishing correct from incorrect program behavior. With the rise of Foundation Models (FMs), particularly Large Language Models (LLMs), there is a new opportunity to generate test oracles that reflect intended behavior. This positions LLMs as enablers of Promptware, where software creation and testing are driven by natural-language prompts. This paper presents an empirical study on the effectiveness of LLMs in generating test oracles that expose software failures. We investigate how different prompting strategies and levels of contextual input impact the quality of LLM-generated oracles. Our findings offer insights into the strengths and limitations of LLM-based oracle generation in the FM era, improving our understanding of their capabilities and fostering future research in this area. 

---
# Evaluating the Use of LLMs for Automated DOM-Level Resolution of Web Performance Issues 

**Authors**: Gideon Peters, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2601.05502)  

**Abstract**: Users demand fast, seamless webpage experiences, yet developers often struggle to meet these expectations within tight constraints. Performance optimization, while critical, is a time-consuming and often manual process. One of the most complex tasks in this domain is modifying the Document Object Model (DOM), which is why this study focuses on it. Recent advances in Large Language Models (LLMs) offer a promising avenue to automate this complex task, potentially transforming how developers address web performance issues. This study evaluates the effectiveness of nine state-of-the-art LLMs for automated web performance issue resolution. For this purpose, we first extracted the DOM trees of 15 popular webpages (e.g., Facebook), and then we used Lighthouse to retrieve their performance audit reports. Subsequently, we passed the extracted DOM trees and corresponding audits to each model for resolution. Our study considers 7 unique audit categories, revealing that LLMs universally excel at SEO & Accessibility issues. However, their efficacy in performance-critical DOM manipulations is mixed. While high-performing models like GPT-4.1 delivered significant reductions in areas like Initial Load, Interactivity, and Network Optimization (e.g., 46.52% to 48.68% audit incidence reductions), others, such as GPT-4o-mini, notably underperformed, consistently. A further analysis of these modifications showed a predominant additive strategy and frequent positional changes, alongside regressions particularly impacting Visual Stability. 

---
# Open World Knowledge Aided Single-Cell Foundation Model with Robust Cross-Modal Cell-Language Pre-training 

**Authors**: Haoran Wang, Xuanyi Zhang, Shuangsang Fang, Longke Ran, Ziqing Deng, Yong Zhang, Yuxiang Li, Shaoshuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05648)  

**Abstract**: Recent advancements in single-cell multi-omics, particularly RNA-seq, have provided profound insights into cellular heterogeneity and gene regulation. While pre-trained language model (PLM) paradigm based single-cell foundation models have shown promise, they remain constrained by insufficient integration of in-depth individual profiles and neglecting the influence of noise within multi-modal data. To address both issues, we propose an Open-world Language Knowledge-Aided Robust Single-Cell Foundation Model (OKR-CELL). It is built based on a cross-modal Cell-Language pre-training framework, which comprises two key innovations: (1) leveraging Large Language Models (LLMs) based workflow with retrieval-augmented generation (RAG) enriches cell textual descriptions using open-world knowledge; (2) devising a Cross-modal Robust Alignment (CRA) objective that incorporates sample reliability assessment, curriculum learning, and coupled momentum contrastive learning to strengthen the model's resistance to noisy data. After pretraining on 32M cell-text pairs, OKR-CELL obtains cutting-edge results across 6 evaluation tasks. Beyond standard benchmarks such as cell clustering, cell-type annotation, batch-effect correction, and few-shot annotation, the model also demonstrates superior performance in broader multi-modal applications, including zero-shot cell-type annotation and bidirectional cell-text retrieval. 

---
# ACR: Adaptive Context Refactoring via Context Refactoring Operators for Multi-Turn Dialogue 

**Authors**: Jiawei Shen, Jia Zhu, Hanghui Guo, Weijie Shi, Yue Cui, Qingyu Niu, Guoqing Ma, Yidan Liang, Jingjiang Liu, Yiling Wang, Shimin Di, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05589)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in multi-turn dialogue. However, in multi-turn dialogue, models still struggle to stay aligned with what has been established earlier, follow dependencies across many turns, and avoid drifting into incorrect facts as the interaction grows longer. Existing approaches primarily focus on extending the context window, introducing external memory, or applying context compression, yet these methods still face limitations such as \textbf{contextual inertia} and \textbf{state drift}. To address these challenges, we propose the \textbf{A}daptive \textbf{C}ontext \textbf{R}efactoring \textbf{(ACR)} Framework, which dynamically monitors and reshapes the interaction history to mitigate contextual inertia and state drift actively. ACR is built on a library of context refactoring operators and a teacher-guided self-evolving training paradigm that learns when to intervene and how to refactor, thereby decoupling context management from the reasoning process. Extensive experiments on multi-turn dialogue demonstrate that our method significantly outperforms existing baselines while reducing token consumption. 

---
# HogVul: Black-box Adversarial Code Generation Framework Against LM-based Vulnerability Detectors 

**Authors**: Jingxiao Yang, Ping He, Tianyu Du, Sun Bing, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05587)  

**Abstract**: Recent advances in software vulnerability detection have been driven by Language Model (LM)-based approaches. However, these models remain vulnerable to adversarial attacks that exploit lexical and syntax perturbations, allowing critical flaws to evade detection. Existing black-box attacks on LM-based vulnerability detectors primarily rely on isolated perturbation strategies, limiting their ability to efficiently explore the adversarial code space for optimal perturbations. To bridge this gap, we propose HogVul, a black-box adversarial code generation framework that integrates both lexical and syntax perturbations under a unified dual-channel optimization strategy driven by Particle Swarm Optimization (PSO). By systematically coordinating two-level perturbations, HogVul effectively expands the search space for adversarial examples, enhancing the attack efficacy. Extensive experiments on four benchmark datasets demonstrate that HogVul achieves an average attack success rate improvement of 26.05\% over state-of-the-art baseline methods. These findings highlight the potential of hybrid optimization strategies in exposing model vulnerabilities. 

---
# STELP: Secure Transpilation and Execution of LLM-Generated Programs 

**Authors**: Swapnil Shinde, Sahil Wadhwa, Andy Luo, Emily Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05467)  

**Abstract**: Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution. 

---
# Lost in Execution: On the Multilingual Robustness of Tool Calling in Large Language Models 

**Authors**: Zheng Luo, T Pranav Kutralingam, Ogochukwu N Okoani, Wanpeng Xu, Hua Wei, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05366)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as agents that invoke external tools through structured function calls. While recent work reports strong tool-calling performance under standard English-centric evaluations, the robustness of tool calling under multilingual user interactions remains underexplored. In this work, we introduce MLCL, a diagnostic benchmark, and conduct a systematic evaluation of multilingual tool calling across Chinese, Hindi, and the low-resource language Igbo. Through fine-grained error analysis, we show that many failures occur despite correct intent understanding and tool selection. We identify parameter value language mismatch as a dominant failure mode, where models generate semantically appropriate parameter values in the user's language, violating language-invariant execution conventions. We further evaluate several inference-time system strategies and find that while these strategies substantially reduce language-induced execution errors, none of them can fully recover English-level performance. 

---
# Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction 

**Authors**: Hongjin Kim, Jaewook Lee, Kiyoung Lee, Jong-hun Shin, Soojong Lim, Oh-Woog Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2601.05459)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning and self-correction abilities in high-resource languages like English, but their performance remains limited in low-resource languages such as Korean. In this study, we investigate whether reinforcement learning (RL) can enhance Korean reasoning abilities to a degree comparable to English. Our findings reveal that RL alone yields limited improvements when applied to models lacking inherent Korean reasoning capabilities. To address this, we explore several fine-tuning strategies and show that aligning the model's internal reasoning processes with Korean inputs-particularly by tuning Korean-specific neurons in early layers-is key to unlocking RL's effectiveness. We introduce a self-correction code-switching dataset to facilitate this alignment and observe significant performance gains in both mathematical reasoning and self-correction tasks. Ultimately, we conclude that the crucial factor in multilingual reasoning enhancement is not injecting new linguistic knowledge, but effectively eliciting and aligning existing reasoning capabilities. Our study provides a new perspective on how internal translation and neuron-level tuning contribute to multilingual reasoning alignment in LLMs. 

---
# Tracing Moral Foundations in Large Language Models 

**Authors**: Chenxiao Yu, Bowen Yi, Farzan Karimi-Malekabadi, Suhaib Abdurahman, Jinyi Ye, Shrikanth Narayanan, Yue Zhao, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2601.05437)  

**Abstract**: Large language models (LLMs) often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed within two instruction-tuned LLMs: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that both models represent and distinguish moral foundations in a structured, layer-dependent way that aligns with human judgments. At a finer scale, SAE features show clear semantic links to specific foundations, suggesting partially disentangled mechanisms within shared representations. Finally, steering along either dense vectors or sparse features produces predictable shifts in foundation-relevant behavior, demonstrating a causal connection between internal representations and moral outputs. Together, our results provide mechanistic evidence that moral concepts in LLMs are distributed, layered, and partly disentangled, suggesting that pluralistic moral structure can emerge as a latent pattern from the statistical regularities of language alone. 

---
# Multi-turn Jailbreaking Attack in Multi-Modal Large Language Models 

**Authors**: Badhan Chandra Das, Md Tasnim Jawad, Joaquin Molto, M. Hadi Amini, Yanzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05339)  

**Abstract**: In recent years, the security vulnerabilities of Multi-modal Large Language Models (MLLMs) have become a serious concern in the Generative Artificial Intelligence (GenAI) research. These highly intelligent models, capable of performing multi-modal tasks with high accuracy, are also severely susceptible to carefully launched security attacks, such as jailbreaking attacks, which can manipulate model behavior and bypass safety constraints. This paper introduces MJAD-MLLMs, a holistic framework that systematically analyzes the proposed Multi-turn Jailbreaking Attacks and multi-LLM-based defense techniques for MLLMs. In this paper, we make three original contributions. First, we introduce a novel multi-turn jailbreaking attack to exploit the vulnerabilities of the MLLMs under multi-turn prompting. Second, we propose a novel fragment-optimized and multi-LLM defense mechanism, called FragGuard, to effectively mitigate jailbreaking attacks in the MLLMs. Third, we evaluate the efficacy of the proposed attacks and defenses through extensive experiments on several state-of-the-art (SOTA) open-source and closed-source MLLMs and benchmark datasets, and compare their performance with the existing techniques. 

---
# On the Limits of Self-Improving in LLMs and Why AGI, ASI and the Singularity Are Not Near Without Symbolic Model Synthesis 

**Authors**: Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2601.05280)  

**Abstract**: We formalise recursive self-training in Large Language Models (LLMs) and Generative AI as a discrete-time dynamical system and prove that, as training data become increasingly self-generated ($\alpha_t \to 0$), the system undergoes inevitably degenerative dynamics. We derive two fundamental failure modes: (1) Entropy Decay, where finite sampling effects cause a monotonic loss of distributional diversity (mode collapse), and (2) Variance Amplification, where the loss of external grounding causes the model's representation of truth to drift as a random walk, bounded only by the support diameter. We show these behaviours are not contingent on architecture but are consequences of distributional learning on finite samples. We further argue that Reinforcement Learning with imperfect verifiers suffers similar semantic collapse. To overcome these limits, we propose a path involving symbolic regression and program synthesis guided by Algorithmic Probability. The Coding Theorem Method (CTM) allows for identifying generative mechanisms rather than mere correlations, escaping the data-processing inequality that binds standard statistical learning. We conclude that while purely distributional learning leads to model collapse, hybrid neurosymbolic approaches offer a coherent framework for sustained self-improvement. 

---
# Automating Deception: Scalable Multi-Turn LLM Jailbreaks 

**Authors**: Adarsh Kumarappan, Ananya Mujoo  

**Link**: [PDF](https://arxiv.org/pdf/2511.19517)  

**Abstract**: Multi-turn conversational attacks, which leverage psychological principles like Foot-in-the-Door (FITD), where a small initial request paves the way for a more significant one, to bypass safety alignments, pose a persistent threat to Large Language Models (LLMs). Progress in defending against these attacks is hindered by a reliance on manual, hard-to-scale dataset creation. This paper introduces a novel, automated pipeline for generating large-scale, psychologically-grounded multi-turn jailbreak datasets. We systematically operationalize FITD techniques into reproducible templates, creating a benchmark of 1,500 scenarios across illegal activities and offensive content. We evaluate seven models from three major LLM families under both multi-turn (with history) and single-turn (without history) conditions. Our results reveal stark differences in contextual robustness: models in the GPT family demonstrate a significant vulnerability to conversational history, with Attack Success Rates (ASR) increasing by as much as 32 percentage points. In contrast, Google's Gemini 2.5 Flash exhibits exceptional resilience, proving nearly immune to these attacks, while Anthropic's Claude 3 Haiku shows strong but imperfect resistance. These findings highlight a critical divergence in how current safety architectures handle conversational context and underscore the need for defenses that can resist narrative-based manipulation. 

---
# EvoC2Rust: A Skeleton-guided Framework for Project-Level C-to-Rust Translation 

**Authors**: Chaofan Wang, Tingrui Yu, Chen Xie, Jie Wang, Dong Chen, Wenrui Zhang, Yuling Shi, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.04295)  

**Abstract**: Translating legacy C codebases to Rust is increasingly demanded for building safety-critical systems. While various approaches have emerged for this task, they face inherent trade-offs: rule-based methods often struggle to satisfy code safety and idiomaticity requirements, while LLM-based methods frequently fail to generate semantically equivalent Rust code, due to the heavy dependencies of modules across the entire codebase. Recent studies have revealed that both solutions are limited to small-scale programs. In this paper, we propose EvoC2Rust, an automated framework for converting complete C projects to equivalent Rust ones. EvoC2Rust employs a skeleton-guided translation strategy for project-level translation. The pipeline consists of three stages: 1) it first decomposes the C project into functional modules, employs a feature-mapping-enhanced LLM to transform definitions and macros, and generates type-checked function stubs, which form a compilable Rust skeleton; 2) it then incrementally translates functions, replacing the corresponding stub placeholders; 3) finally, it repairs compilation errors by integrating LLM and static analysis. Through evolutionary augmentation, EvoC2Rust combines the advantages of both rule-based and LLM-based solutions. Our evaluation on open-source benchmarks and six industrial projects demonstrates the superior performance of EvoC2Rust in project-level C-to-Rust translation. The results show that our approach outperforms the strongest LLM-based baseline by 17.24% in syntax accuracy and 14.32% in semantic accuracy, while also achieving a 43.59% higher code safety rate than the best rule-based tool. 

---
# Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM 

**Authors**: Adarsh Kumarappan, Ayushi Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2511.18721)  

**Abstract**: The SmoothLLM defense provides a certification guarantee against jailbreaking attacks, but it relies on a strict `k-unstable' assumption that rarely holds in practice. This strong assumption can limit the trustworthiness of the provided safety certificate. In this work, we address this limitation by introducing a more realistic probabilistic framework, `(k, $\varepsilon$)-unstable,' to certify defenses against diverse jailbreaking attacks, from gradient-based (GCG) to semantic (PAIR). We derive a new, data-informed lower bound on SmoothLLM's defense probability by incorporating empirical models of attack success, providing a more trustworthy and practical safety certificate. By introducing the notion of (k, $\varepsilon$)-unstable, our framework provides practitioners with actionable safety guarantees, enabling them to set certification thresholds that better reflect the real-world behavior of LLMs. Ultimately, this work contributes a practical and theoretically-grounded mechanism to make LLMs more resistant to the exploitation of their safety alignments, a critical challenge in secure AI deployment. 

---
# HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search 

**Authors**: Zihang Tian, Rui Li, Jingsen Zhang, Xiaohe Bo, Wei Huo, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05903)  

**Abstract**: Large language model (LLM) routing aims to exploit the specialized strengths of different LLMs for diverse tasks. However, existing approaches typically focus on selecting LLM architectures while overlooking parameter settings, which are critical for task performance. In this paper, we introduce HAPS, a hierarchical LLM routing framework that jointly searches over model architectures and parameters. Specifically, we use a high-level router to select among candidate LLM architectures, and then search for the optimal parameters for the selected architectures based on a low-level router. We design a parameter generation network to share parameters between the two routers to mutually enhance their capabilities. In the training process, we design a reward-augmented objective to effectively optimize our framework. Experiments on two commonly used benchmarks show that HAPS consistently outperforms strong routing baselines. We have released our code at this https URL. 

---
# Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks 

**Authors**: Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2601.06007)  

**Abstract**: Recent advancements in Large Language Model (LLM) agents have enabled complex multi-turn agentic tasks requiring extensive tool calling, where conversations can span dozens of API calls with increasingly large context windows. However, although major LLM providers offer prompt caching to reduce cost and latency, its benefits for agentic workloads remain underexplored in the research literature. To our knowledge, no prior work quantifies these cost savings or compares caching strategies for multi-turn agentic tasks. We present a comprehensive evaluation of prompt caching across three major LLM providers (OpenAI, Anthropic, and Google) and compare three caching strategies, including full context caching, system prompt only caching, and caching that excludes dynamic tool results. We evaluate on DeepResearchBench, a multi-turn agentic benchmark where agents autonomously execute real-world web search tool calls to answer complex research questions, measuring both API cost and time to first token (TTFT) across over 500 agent sessions with 10,000-token system prompts. Our results demonstrate that prompt caching reduces API costs by 45-80% and improves time to first token by 13-31% across providers. We find that strategic prompt cache block control, such as placing dynamic content at the end of the system prompt, avoiding dynamic traditional function calling, and excluding dynamic tool results, provides more consistent benefits than naive full-context caching, which can paradoxically increase latency. Our analysis reveals nuanced variations in caching behavior across providers, and we provide practical guidance for implementing prompt caching in production agentic systems. 

---
# iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models 

**Authors**: Meghana Sunil, Manikandarajan Venmathimaran, Muthu Subash Kavitha  

**Link**: [PDF](https://arxiv.org/pdf/2601.05877)  

**Abstract**: Recent work shows that large multimodal models (LMMs) can self-improve from unlabeled data via self-play and intrinsic feedback. Yet existing self-evolving frameworks mainly reward final outcomes, leaving intermediate reasoning weakly constrained despite its importance for visually grounded decision making. We propose iReasoner, a self-evolving framework that improves an LMM's implicit reasoning by explicitly eliciting chain-of-thought (CoT) and rewarding its internal agreement. In a Proposer--Solver loop over unlabeled images, iReasoner augments outcome-level intrinsic rewards with a trajectory-aware signal defined over intermediate reasoning steps, providing learning signals that distinguish reasoning paths leading to the same answer without ground-truth labels or external judges. Starting from Qwen2.5-VL-7B, iReasoner yields up to $+2.1$ points across diverse multimodal reasoning benchmarks under fully unsupervised post-training. We hope this work serves as a starting point for reasoning-aware self-improvement in LMMs in purely unsupervised settings. 

---
# LLMs as Science Journalists: Supporting Early-stage Researchers in Communicating Their Science to the Public 

**Authors**: Milad Alshomary, Grace Li, Anubhav Jangra, Yufang Hou, Kathleen McKeown, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05821)  

**Abstract**: The scientific community needs tools that help early-stage researchers effectively communicate their findings and innovations to the public. Although existing general-purpose Large Language Models (LLMs) can assist in this endeavor, they are not optimally aligned for it. To address this, we propose a framework for training LLMs to emulate the role of a science journalist that can be used by early-stage researchers to learn how to properly communicate their papers to the general public. We evaluate the usefulness of our trained LLM Journalists in leading conversations with both simulated and human researchers. %compared to the general-purpose ones. Our experiments indicate that LLMs trained using our framework ask more relevant questions that address the societal impact of research, prompting researchers to clarify and elaborate on their findings. In the user study, the majority of participants who interacted with our trained LLM Journalist appreciated it more than interacting with general-purpose LLMs. 

---
# Left, Right, or Center? Evaluating LLM Framing in News Classification and Generation 

**Authors**: Molly Kennedy, Ali Parker, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.05835)  

**Abstract**: Large Language Model (LLM) based summarization and text generation are increasingly used for producing and rewriting text, raising concerns about political framing in journalism where subtle wording choices can shape interpretation. Across nine state-of-the-art LLMs, we study political framing by testing whether LLMs' classification-based bias signals align with framing behavior in their generated summaries. We first compare few-shot ideology predictions against LEFT/CENTER/RIGHT labels. We then generate "steered" summaries under FAITHFUL, CENTRIST, LEFT, and RIGHT prompts, and score all outputs using a single fixed ideology evaluator. We find pervasive ideological center-collapse in both article-level ratings and generated text, indicating a systematic tendency toward centrist framing. Among evaluated models, Grok 4 is by far the most ideologically expressive generator, while Claude Sonnet 4.5 and Llama 3.1 achieve the strongest bias-rating performance among commercial and open-weight models, respectively. 

---
# AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor 

**Authors**: Shu Yang, Jingyu Hu, Tong Li, Hanqi Yan, Wenxuan Wang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05752)  

**Abstract**: We introduce AutoMonitor-Bench, the first benchmark designed to systematically evaluate the reliability of LLM-based misbehavior monitors across diverse tasks and failure modes. AutoMonitor-Bench consists of 3,010 carefully annotated test samples spanning question answering, code generation, and reasoning, with paired misbehavior and benign instances. We evaluate monitors using two complementary metrics: Miss Rate (MR) and False Alarm Rate (FAR), capturing failures to detect misbehavior and oversensitivity to benign behavior, respectively. Evaluating 12 proprietary and 10 open-source LLMs, we observe substantial variability in monitoring performance and a consistent trade-off between MR and FAR, revealing an inherent safety-utility tension. To further explore the limits of monitor reliability, we construct a large-scale training corpus of 153,581 samples and fine-tune Qwen3-4B-Instruction to investigate whether training on known, relatively easy-to-construct misbehavior datasets improves monitoring performance on unseen and more implicit misbehaviors. Our results highlight the challenges of reliable, scalable misbehavior monitoring and motivate future work on task-aware designing and training strategies for LLM-based monitors. 

---
# Visualising Information Flow in Word Embeddings with Diffusion Tensor Imaging 

**Authors**: Thomas Fabian  

**Link**: [PDF](https://arxiv.org/pdf/2601.05713)  

**Abstract**: Understanding how large language models (LLMs) represent natural language is a central challenge in natural language processing (NLP) research. Many existing methods extract word embeddings from an LLM, visualise the embedding space via point-plots, and compare the relative positions of certain words. However, this approach only considers single words and not whole natural language expressions, thus disregards the context in which a word is used. Here we present a novel tool for analysing and visualising information flow in natural language expressions by applying diffusion tensor imaging (DTI) to word embeddings. We find that DTI reveals how information flows between word embeddings. Tracking information flows within the layers of an LLM allows for comparing different model structures and revealing opportunities for pruning an LLM's under-utilised layers. Furthermore, our model reveals differences in information flows for tasks like pronoun resolution and metaphor detection. Our results show that our model permits novel insights into how LLMs represent actual natural language expressions, extending the comparison of isolated word embeddings and improving the interpretability of NLP models. 

---
# GIFT: Games as Informal Training for Generalizable LLMs 

**Authors**: Nuoyan Lyu, Bingbing Xu, Weihao Meng, Yige Yuan, Yang Zhang, Zhiyong Huang, Tat-Seng Chua, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05633)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in formal learning tasks such as mathematics and code generation, they still struggle with the "practical wisdom" and generalizable intelligence, such as strategic creativity and social reasoning, that characterize human cognition. This gap arises from a lack of informal learning, which thrives on interactive feedback rather than goal-oriented instruction. In this paper, we propose treating Games as a primary environment for LLM informal learning, leveraging their intrinsic reward signals and abstracted complexity to cultivate diverse competencies. To address the performance degradation observed in multi-task learning, we introduce a Nested Training Framework. Unlike naive task mixing optimizing an implicit "OR" objective, our framework employs sequential task composition to enforce an explicit "AND" objective, compelling the model to master multiple abilities simultaneously to achieve maximal rewards. Using GRPO-based reinforcement learning across Matrix Games, TicTacToe, and Who's the Spy games, we demonstrate that integrating game-based informal learning not only prevents task interference but also significantly bolsters the model's generalization across broad ability-oriented benchmarks. The framework and implementation are publicly available. 

---
# Data Augmented Pipeline for Legal Information Extraction and Reasoning 

**Authors**: Nguyen Minh Phuong, Ha-Thanh Nguyen, May Myo Zin, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2601.05609)  

**Abstract**: In this paper, we propose a pipeline leveraging Large Language Models (LLMs) for data augmentation in Information Extraction tasks within the legal domain. The proposed method is both simple and effective, significantly reducing the manual effort required for data annotation while enhancing the robustness of Information Extraction systems. Furthermore, the method is generalizable, making it applicable to various Natural Language Processing (NLP) tasks beyond the legal domain. 

---
# Can Large Language Models Differentiate Harmful from Argumentative Essays? Steps Toward Ethical Essay Scoring 

**Authors**: Hongjin Kim, Jeonghyun Kang, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05545)  

**Abstract**: This study addresses critical gaps in Automated Essay Scoring (AES) systems and Large Language Models (LLMs) with regard to their ability to effectively identify and score harmful essays. Despite advancements in AES technology, current models often overlook ethically and morally problematic elements within essays, erroneously assigning high scores to essays that may propagate harmful opinions. In this study, we introduce the Harmful Essay Detection (HED) benchmark, which includes essays integrating sensitive topics such as racism and gender bias, to test the efficacy of various LLMs in recognizing and scoring harmful content. Our findings reveal that: (1) LLMs require further enhancement to accurately distinguish between harmful and argumentative essays, and (2) both current AES models and LLMs fail to consider the ethical dimensions of content during scoring. The study underscores the need for developing more robust AES systems that are sensitive to the ethical implications of the content they are scoring. 

---
# CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems 

**Authors**: Xuemei Tang, Chengxi Yan, Jinghang Gu, Chu-Ren Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05520)  

**Abstract**: Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, we construct a large-scale, domain-aware event taxonomy covering politics, military, diplomacy, and social life in ancient China. Extensive reference-free and reference-based evaluations demonstrate improved structural coherence and coverage, while further analysis shows that the resulting taxonomy supports cross-cultural alignment. 

---
# Closing the Modality Reasoning Gap for Speech Large Language Models 

**Authors**: Chaoren Wang, Heng Lu, Xueyao Zhang, Shujie Liu, Yan Lu, Jinyu Li, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05543)  

**Abstract**: Although speech large language models have achieved notable progress, a substantial modality reasoning gap remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap could be associated with representational drift across Transformer layers and behavior deviations in long-chain reasoning. To address this issue, we introduce TARS, a reinforcement-learning framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. The framework employs two dense and complementary signals: representation alignment, which measures layer-wise hidden-state similarity between speech- and text-conditioned trajectories, and behavior alignment, which evaluates semantic consistency between generated outputs and reference text completions. Experiments on challenging reasoning benchmarks, including MMSU and OBQA, show that our approach significantly narrows the modality reasoning gap and achieves state-of-the-art performance among 7B-scale Speech LLMs. 

---
# The Facade of Truth: Uncovering and Mitigating LLM Susceptibility to Deceptive Evidence 

**Authors**: Herun Wan, Jiaying Wu, Minnan Luo, Fanxiao Li, Zhi Zeng, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05478)  

**Abstract**: To reliably assist human decision-making, LLMs must maintain factual internal beliefs against misleading injections. While current models resist explicit misinformation, we uncover a fundamental vulnerability to sophisticated, hard-to-falsify evidence. To systematically probe this weakness, we introduce MisBelief, a framework that generates misleading evidence via collaborative, multi-round interactions among multi-role LLMs. This process mimics subtle, defeasible reasoning and progressive refinement to create logically persuasive yet factually deceptive claims. Using MisBelief, we generate 4,800 instances across three difficulty levels to evaluate 7 representative LLMs. Results indicate that while models are robust to direct misinformation, they are highly sensitive to this refined evidence: belief scores in falsehoods increase by an average of 93.0\%, fundamentally compromising downstream recommendations. To address this, we propose Deceptive Intent Shielding (DIS), a governance mechanism that provides an early warning signal by inferring the deceptive intent behind evidence. Empirical results demonstrate that DIS consistently mitigates belief shifts and promotes more cautious evidence evaluation. 

---
# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions 

**Authors**: Minda Zhao, Yilun Du, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05414)  

**Abstract**: As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines across domains like educational assessment and synthetic data construction, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces N=1000 samples within one response, and Independent Requests, comprising $N=1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 13% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the requested sampling horizon N increases. Finally, we demonstrate the propagation of these failures into downstream tasks: models fail to enforce uniform answer-position constraints in MCQ generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating the use of external tools for applications requiring statistical guarantees. 

---
# Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings 

**Authors**: Xiran Fan, Zhimeng Jiang, Chin-Chia Michael Yeh, Yuzhong Chen, Yingtong Dou, Menghai Pan, Yan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05271)  

**Abstract**: The ubiquity of payment networks generates vast transactional data encoding rich consumer and merchant behavioral patterns. Recent foundation models for transaction analysis process tabular data sequentially but rely on index-based representations for categorical merchant fields, causing substantial semantic information loss by converting rich textual data into discrete tokens. While Large Language Models (LLMs) can address this limitation through superior semantic understanding, their computational overhead challenges real-time financial deployment. We introduce a hybrid framework that uses LLM-generated embeddings as semantic initializations for lightweight transaction models, balancing interpretability with operational efficiency. Our approach employs multi-source data fusion to enrich merchant categorical fields and a one-word constraint principle for consistent embedding generation across LLM architectures. We systematically address data quality through noise filtering and context-aware enrichment. Experiments on large-scale transaction datasets demonstrate significant performance improvements across multiple transaction understanding tasks. 

---
# MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards 

**Authors**: Zhiyu Shen, Ziming Wu, Fuming Lai, Shaobing Lian, Yanghui Rao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05488)  

**Abstract**: Maintaining consistency in long-term dialogues remains a fundamental challenge for LLMs, as standard retrieval mechanisms often fail to capture the temporal evolution of historical states. While memory-augmented frameworks offer a structured alternative, current systems rely on static prompting of closed-source models or suffer from ineffective training paradigms with sparse rewards. We introduce MemBuilder, a reinforcement learning framework that trains models to orchestrate multi-dimensional memory construction with attributed dense rewards. MemBuilder addresses two key challenges: (1) Sparse Trajectory-Level Rewards: we employ synthetic session-level question generation to provide dense intermediate rewards across extended trajectories; and (2) Multi-Dimensional Memory Attribution: we introduce contribution-aware gradient weighting that scales policy updates based on each component's downstream impact. Experimental results show that MemBuilder enables a 4B-parameter model to outperform state-of-the-art closed-source baselines, exhibiting strong generalization across long-term dialogue benchmarks. 

---
# Can large language models interpret unstructured chat data on dynamic group decision-making processes? Evidence on joint destination choice 

**Authors**: Sung-Yoo Lim, Koki Sato, Kiyoshi Takami, Giancarlos Parady, Eui-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05582)  

**Abstract**: Social activities result from complex joint activity-travel decisions between group members. While observing the decision-making process of these activities is difficult via traditional travel surveys, the advent of new types of data, such as unstructured chat data, can help shed some light on these complex processes. However, interpreting these decision-making processes requires inferring both explicit and implicit factors. This typically involves the labor-intensive task of manually annotating dialogues to capture context-dependent meanings shaped by the social and cultural norms. This study evaluates the potential of Large Language Models (LLMs) to automate and complement human annotation in interpreting decision-making processes from group chats, using data on joint eating-out activities in Japan as a case study. We designed a prompting framework inspired by the knowledge acquisition process, which sequentially extracts key decision-making factors, including the group-level restaurant choice set and outcome, individual preferences of each alternative, and the specific attributes driving those preferences. This structured process guides the LLM to interpret group chat data, converting unstructured dialogues into structured tabular data describing decision-making factors. To evaluate LLM-driven outputs, we conduct a quantitative analysis using a human-annotated ground truth dataset and a qualitative error analysis to examine model limitations. Results show that while the LLM reliably captures explicit decision-making factors, it struggles to identify nuanced implicit factors that human annotators readily identified. We pinpoint specific contexts when LLM-based extraction can be trusted versus when human oversight remains essential. These findings highlight both the potential and limitations of LLM-based analysis for incorporating non-traditional data sources on social activities. 

---
# Towards Valid Student Simulation with Large Language Models 

**Authors**: Zhihao Yuan, Yunze Xiao, Ming Li, Weihao Xuan, Richard Tong, Mona Diab, Tom Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2601.05473)  

**Abstract**: This paper presents a conceptual and methodological framework for large language model (LLM) based student simulation in educational settings. The authors identify a core failure mode, termed the "competence paradox" in which broadly capable LLMs are asked to emulate partially knowledgeable learners, leading to unrealistic error patterns and learning dynamics. To address this, the paper reframes student simulation as a constrained generation problem governed by an explicit Epistemic State Specification (ESS), which defines what a simulated learner can access, how errors are structured, and how learner state evolves over time. The work further introduces a Goal-by-Environment framework to situate simulated student systems according to behavioral objectives and deployment contexts. Rather than proposing a new system or benchmark, the paper synthesizes prior literature, formalizes key design dimensions, and articulates open challenges related to validity, evaluation, and ethical risks. Overall, the paper argues for epistemic fidelity over surface realism as a prerequisite for using LLM-based simulated students as reliable scientific and pedagogical instruments. 

---
# The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era 

**Authors**: Zhixian Zhao, Shuiyuan Wang, Guojian Li, Hongfei Xue, Chengyou Wang, Shuai Wang, Longshuai Xiao, Zihan Zhang, Hui Bu, Xin Xu, Xinsheng Wang, Hexin Liu, Eng Siong Chng, Hung-yi Lee, Haizhou Li, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.05564)  

**Abstract**: Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results. 

---
# Hi-ZFO: Hierarchical Zeroth- and First-Order LLM Fine-Tuning via Importance-Guided Tensor Selection 

**Authors**: Feihu Jin, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05501)  

**Abstract**: Fine-tuning large language models (LLMs) using standard first-order (FO) optimization often drives training toward sharp, poorly generalizing minima. Conversely, zeroth-order (ZO) methods offer stronger exploratory behavior without relying on explicit gradients, yet suffer from slow convergence. More critically, our analysis reveals that in generative tasks, the vast output and search space significantly amplify estimation variance, rendering ZO methods both noisy and inefficient. To address these challenges, we propose \textbf{Hi-ZFO} (\textbf{Hi}erarchical \textbf{Z}eroth- and \textbf{F}irst-\textbf{O}rder optimization), a hybrid framework designed to synergize the precision of FO gradients with the exploratory capability of ZO estimation. Hi-ZFO adaptively partitions the model through layer-wise importance profiling, applying precise FO updates to critical layers while leveraging ZO optimization for less sensitive ones. Notably, ZO in Hi-ZFO is not merely a memory-saving surrogate; it is intentionally introduced as a source of "beneficial stochasticity" to help the model escape the local minima where pure FO optimization tends to stagnate. Validated across diverse generative, mathematical, and code reasoning tasks, Hi-ZFO consistently achieves superior performance while significantly reducing the training time. These results demonstrate the effectiveness of hierarchical hybrid optimization for LLM fine-tuning. 

---
# Enabling Stroke-Level Structural Analysis of Hieroglyphic Scripts without Language-Specific Priors 

**Authors**: Fuwen Luo, Zihao Wan, Ziyue Wang, Yaluo Liu, Pau Tong Lin Xu, Xuanjia Qiao, Xiaolong Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05508)  

**Abstract**: Hieroglyphs, as logographic writing systems, encode rich semantic and cultural information within their internal structural composition. Yet, current advanced Large Language Models (LLMs) and Multimodal LLMs (MLLMs) usually remain structurally blind to this information. LLMs process characters as textual tokens, while MLLMs additionally view them as raw pixel grids. Both fall short to model the underlying logic of character strokes. Furthermore, existing structural analysis methods are often script-specific and labor-intensive. In this paper, we propose Hieroglyphic Stroke Analyzer (HieroSA), a novel and generalizable framework that enables MLLMs to automatically derive stroke-level structures from character bitmaps without handcrafted data. It transforms modern logographic and ancient hieroglyphs character images into explicit, interpretable line-segment representations in a normalized coordinate space, allowing for cross-lingual generalization. Extensive experiments demonstrate that HieroSA effectively captures character-internal structures and semantics, bypassing the need for language-specific priors. Experimental results highlight the potential of our work as a graphematics analysis tool for a deeper understanding of hieroglyphic scripts. View our code at this https URL. 

---
# MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization 

**Authors**: Jiefu Ou, Sapana Chaudhary, Kaj Bostrom, Nathaniel Weir, Shuai Zhang, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2601.05475)  

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities in general coding tasks but encounter two key challenges when optimizing code: (i) the complexity of writing optimized code (such as performant CUDA kernels and competition-level CPU code) requires expertise in systems, algorithms and specific languages and (ii) requires interpretation of performance metrics like timing and device utilization beyond binary correctness. In this work, we explore inference-time search algorithms that guide the LLM to discover better solutions through iterative refinement based on execution feedback. Our approach, called MaxCode unifies existing search methods under a max-reward reinforcement learning framework, making the observation and action-value functions modular for modification. To enhance the observation space, we integrate a natural language critique model that converts raw execution feedback into diagnostic insights about errors and performance bottlenecks, and the best-discounted reward seen so far. Together, these provide richer input to the code proposal function. To improve exploration during search, we train a generative reward-to-go model using action values from rollouts to rerank potential solutions. Testing on the KernelBench (CUDA) and PIE (C++) optimization benchmarks shows that MaxCode improves optimized code performance compared to baselines, achieving 20.3% and 10.1% relative improvements in absolute speedup value and relative speedup ranking, respectively. 

---
# TIME: Temporally Intelligent Meta-reasoning Engine for Context Triggered Explicit Reasoning 

**Authors**: Susmit Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.05300)  

**Abstract**: Reasoning oriented large language models often expose explicit "thinking" as long, turn-global traces at the start of every response, either always on or toggled externally at inference time. While useful for arithmetic, programming, and problem solving, this design is costly, blurs claim level auditability, and cannot re-trigger explicit reasoning once the model begins presenting. Dialogue models are also largely blind to temporal structure, treating replies after seconds and replies after weeks as equivalent unless time is stated in text. We introduce TIME, the Temporally Intelligent Meta-reasoning Engine, a behavioral alignment framework that treats explicit reasoning as a context sensitive resource driven by discourse and temporal cues. TIME augments dialogue with optional ISO 8601 <time> tags, tick turns that represent silent gaps, and short <think> blocks that can appear anywhere in a reply. A four-phase curriculum including a small, maximally diverse full-batch alignment step trains Qwen3 dense models to invoke brief, in-place reasoning bursts and keep user facing text compact. We evaluate with TIMEBench, a temporally grounded dialogue benchmark probing chronology, commonsense under gaps and offsets, anomaly detection, and continuity. Across 4B to 32B scales, TIME improves TIMEBench scores over base Qwen3 in both thinking and no-thinking modes while reducing reasoning tokens by about an order of magnitude. Our training data and code are available at this https URL and TIMEBench is available at this https URL 

---
# RingSQL: Generating Synthetic Data with Schema-Independent Templates for Text-to-SQL Reasoning Models 

**Authors**: Marko Sterbentz, Kevin Cushing, Cameron Barrie, Kristian J. Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2601.05451)  

**Abstract**: Recent advances in text-to-SQL systems have been driven by larger models and improved datasets, yet progress is still limited by the scarcity of high-quality training data. Manual data creation is expensive, and existing synthetic methods trade off reliability and scalability. Template-based approaches ensure correct SQL but require schema-specific templates, while LLM-based generation scales easily but lacks quality and correctness guarantees. We introduce RingSQL, a hybrid data generation framework that combines schema-independent query templates with LLM-based paraphrasing of natural language questions. This approach preserves SQL correctness across diverse schemas while providing broad linguistic variety. In our experiments, we find that models trained using data produced by RingSQL achieve an average gain in accuracy of +2.3% across six text-to-SQL benchmarks when compared to models trained on other synthetic data. We make our code available at this https URL. 

---
