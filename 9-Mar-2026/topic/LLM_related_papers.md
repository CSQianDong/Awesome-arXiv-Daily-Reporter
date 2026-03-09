# MLLMRec-R1: Incentivizing Reasoning Capability in Large Language Models for Multimodal Sequential Recommendation 

**Authors**: Yu Wang, Yonghui Yang, Le Wu, Jiancan Wu, Hefei Xu, Hui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.06243)  

**Abstract**: Group relative policy optimization (GRPO) has become a standard post-training paradigm for improving reasoning and preference alignment in large language models (LLMs), and has recently shown strong effectiveness in LLM-based recommender systems. However, extending GRPO-based reasoning pipelines to multimodal sequential recommendation (MSR) with multimodal large language models (MLLMs) faces fundamental obstacles. First, MSR requires jointly encoding visual content for both historical interactions and multiple candidate items, causing visual tokens to dominate the input and making the cost of group-based rollout scale with history length and candidate set size, which renders GRPO-based training prohibitively expensive. Second, existing Chain-of-Thought (CoT) supervision suffers from reward inflation in recommendation scenarios, where higher training rewards do not reliably translate into improved ranking performance and may induce shortcut learning. To address these challenges, we propose MLLMRec-R1, an efficient and stable GRPO-based reasoning framework for multimodal sequential recommendation. MLLMRec-R1 textualizes visual signals offline to eliminate expensive visual tokens while preserving multimodal semantics, and constructs high-quality multimodal CoT supervision through refinement and confidence-aware assessment. Furthermore, a mixed-grained data augmentation strategy selectively injects reliable CoT samples while retaining standard training data, mitigating reward inflation and improving generalization stability. Extensive experiments on three benchmark datasets demonstrate that MLLMRec-R1 consistently outperforms state-of-the-art methods, establishing a practical and effective GRPO-based reasoning pipeline for multimodal sequential recommendation. The code is available at this https URL. 

---
# Sensitivity-Aware Retrieval-Augmented Intent Clarification 

**Authors**: Maik Larooij  

**Link**: [PDF](https://arxiv.org/pdf/2603.06025)  

**Abstract**: In conversational search systems, a key component is to determine and clarify the intent behind complex queries. We view intent clarification in light of the exploratory search paradigm, where users, through an iterative, evolving process of selection, exploration and retrieval, transform a visceral or conscious need into a formalized one. Augmenting the clarification component with a retrieval step (retrieval-augmented intent clarification) can seriously enhance clarification performance, especially in domains where Large Language Models (LLMs) lack parametric knowledge. However, in more sensitive domains, such as healthcare, government (e.g. FOIA search) or legal contexts, the retrieval database may contain sensitive information that needs protection. In this paper, we explore the research challenge of developing a retrieval-augmented conversational agent that can act as a mediator and gatekeeper for the sensitive collection. To do that, we also need to know what we are protecting and against what. We propose to tackle this research challenge in three steps: 1) define an attack model, 2) design sensitivity-aware defenses on the retrieval level and 3) develop evaluation methods to measure the trade-off between the level of protection and the system's utility. 

---
# ChatShopBuddy: Towards Reliable Conversational Shopping Agents via Reinforcement Learning 

**Authors**: Yiruo Cheng, Kelong Mao, Tianhao Li, Jiejun Tan, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2603.06065)  

**Abstract**: Conversational shopping agents represent a critical consumer-facing application of Large Language Model (LLM)-powered agents, yet how to effectively apply post-training Reinforcement Learning (RL) to optimize such agents remains underexplored. This work investigates RL-based optimization for shopping agents in real-world scenarios, where agents must simultaneously satisfy multiple interdependent objectives spanning objective metrics (product correctness), subjective qualities (persuasiveness), outcome rewards (final response quality), and process rewards (tool efficiency). We present a complete methodology to address this challenge. Specifically, we first construct SmartShopBench, a benchmark that captures diverse shopping intents with a hierarchical evaluation that decomposes complex quality requirements into measurable levels. Building on this evaluation framework, we design Hierarchical Reward Modeling (HRM) to structure mixed reward types through conditional gating that reflects their logical dependencies. To enable efficient training, we further propose Dynamic Contrastive Policy Optimization (DCPO), which balances response quality with operational efficiency through dynamic trajectory selection based on reward and reasoning length. Extensive experiments demonstrate that our RL-trained agent, namely ChatShopBuddy, consistently outperforms larger models relying on generic reasoning, achieving superior stability rather than merely higher peaks. Our work provides valuable guidance for applying RL to real-world conversational agents. 

---
# Verify as You Go: An LLM-Powered Browser Extension for Fake News Detection 

**Authors**: Dorsaf Sallami, Esma Aïmeur  

**Link**: [PDF](https://arxiv.org/pdf/2603.05519)  

**Abstract**: The rampant spread of fake news in the digital age poses serious risks to public trust and democratic institutions, underscoring the need for effective, transparent, and user-centered detection tools. Existing browser extensions often fall short due to opaque model behavior, limited explanatory support, and a lack of meaningful user engagement. This paper introduces Aletheia, a novel browser extension that leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to detect fake news and provide evidence-based explanations. Aletheia further includes two interactive components: a Discussion Hub that enables user dialogue around flagged content and a Stay Informed feature that surfaces recent fact-checks. Through extensive experiments, we show that Aletheia outperforms state-of-the-art baselines in detection performance. Complementing this empirical evaluation, a complementary user study with 250 participants confirms the system's usability and perceived effectiveness, highlighting its potential as a transparent tool for combating online fake news. 

---
# Abductive Reasoning with Syllogistic Forms in Large Language Models 

**Authors**: Hirohiko Abe, Risako Ando, Takanobu Morishita Kentaro Ozeki, Koji Mineshima, Mitsuhiro Okada  

**Link**: [PDF](https://arxiv.org/pdf/2603.06428)  

**Abstract**: Research in AI using Large-Language Models (LLMs) is rapidly evolving, and the comparison of their performance with human reasoning has become a key concern. Prior studies have indicated that LLMs and humans share similar biases, such as dismissing logically valid inferences that contradict common beliefs. However, criticizing LLMs for these biases might be unfair, considering our reasoning not only involves formal deduction but also abduction, which draws tentative conclusions from limited information. Abduction can be regarded as the inverse form of syllogism in its basic structure, that is, a process of drawing a minor premise from a major premise and conclusion. This paper explores the accuracy of LLMs in abductive reasoning by converting a syllogistic dataset into one suitable for abduction. It aims to investigate whether the state-of-the-art LLMs exhibit biases in abduction and to identify potential areas for improvement, emphasizing the importance of contextualized reasoning beyond formal deduction. This investigation is vital for advancing the understanding and application of LLMs in complex reasoning tasks, offering insights into bridging the gap between machine and human cognition. 

---
# From Prompting to Preference Optimization: A Comparative Study of LLM-based Automated Essay Scoring 

**Authors**: Minh Hoang Nguyen, Vu Hoang Pham, Xuan Thanh Huynh, Phuc Hong Mai, Vinh The Nguyen, Quang Nhut Huynh, Huy Tien Nguyen, Tung Le  

**Link**: [PDF](https://arxiv.org/pdf/2603.06424)  

**Abstract**: Large language models (LLMs) have recently reshaped Automated Essay Scoring (AES), yet prior studies typically examine individual techniques in isolation, limiting understanding of their relative merits for English as a Second Language (L2) writing. To bridge this gap, we presents a comprehensive comparison of major LLM-based AES paradigms on IELTS Writing Task~2. On this unified benchmark, we evaluate four approaches: (i) encoder-based classification fine-tuning, (ii) zero- and few-shot prompting, (iii) instruction tuning and Retrieval-Augmented Generation (RAG), and (iv) Supervised Fine-Tuning combined with Direct Preference Optimization (DPO) and RAG. Our results reveal clear accuracy-cost-robustness trade-offs across methods, the best configuration, integrating k-SFT and RAG, achieves the strongest overall results with F1-Score 93%. This study offers the first unified empirical comparison of modern LLM-based AES strategies for English L2, promising potential in auto-grading writing tasks. Code is public at this https URL 

---
# Transparent AI for Mathematics: Transformer-Based Large Language Models for Mathematical Entity Relationship Extraction with XAI 

**Authors**: Tanjim Taharat Aurpa  

**Link**: [PDF](https://arxiv.org/pdf/2603.06348)  

**Abstract**: Mathematical text understanding is a challenging task due to the presence of specialized entities and complex relationships between them. This study formulates mathematical problem interpretation as a Mathematical Entity Relation Extraction (MERE) task, where operands are treated as entities and operators as their relationships. Transformer-based models are applied to automatically extract these relations from mathematical text, with Bidirectional Encoder Representations from Transformers (BERT) achieving the best performance, reaching an accuracy of 99.39%. To enhance transparency and trust in the model's predictions, Explainable Artificial Intelligence (XAI) is incorporated using Shapley Additive Explanations (SHAP). The explainability analysis reveals how specific textual and mathematical features influence relation prediction, providing insights into feature importance and model behavior. By combining transformer-based learning, a task-specific dataset, and explainable modeling, this work offers an effective and interpretable framework for MERE, supporting future applications in automated problem solving, knowledge graph construction, and intelligent educational systems. 

---
# LIT-RAGBench: Benchmarking Generator Capabilities of Large Language Models in Retrieval-Augmented Generation 

**Authors**: Koki Itai, Shunichi Hasegawa, Yuta Yamamoto, Gouki Minegishi, Masaki Otsuki  

**Link**: [PDF](https://arxiv.org/pdf/2603.06198)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a framework in which a Generator, such as a Large Language Model (LLM), produces answers by retrieving documents from an external collection using a Retriever. In practice, Generators must integrate evidence from long contexts, perform multi-step reasoning, interpret tables, and abstain when evidence is missing. However, existing benchmarks for Generators provide limited coverage, with none enabling simultaneous evaluation of multiple capabilities under unified conditions. To bridge the gap between existing evaluations and practical use, we introduce LIT-RAGBench (the Logic, Integration, Table, Reasoning, and Abstention RAG Generator Benchmark), which defines five categories: Integration, Reasoning, Logic, Table, and Abstention, each further divided into practical evaluation aspects. LIT-RAGBench systematically covers patterns combining multiple aspects across categories. By using fictional entities and scenarios, LIT-RAGBench evaluates answers grounded in the provided external documents. The dataset consists of 114 human-constructed Japanese questions and an English version generated by machine translation with human curation. We use LLM-as-a-Judge for scoring and report category-wise and overall accuracy. Across API-based and open-weight models, no model exceeds 90% overall accuracy. By making strengths and weaknesses measurable within each category, LIT-RAGBench serves as a valuable metric for model selection in practical RAG deployments and for building RAG-specialized models. We release LIT-RAGBench, including the dataset and evaluation code, at this https URL. 

---
# Mind the Gap: Pitfalls of LLM Alignment with Asian Public Opinion 

**Authors**: Hari Shankar, Vedanta S P, Sriharini Margapuri, Debjani Mazumder, Ponnurangam Kumaraguru, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.06264)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in multilingual, multicultural settings, yet their reliance on predominantly English-centric training data risks misalignment with the diverse cultural values of different societies. In this paper, we present a comprehensive, multilingual audit of the cultural alignment of contemporary LLMs including GPT-4o-Mini, Gemini-2.5-Flash, Llama 3.2, Mistral and Gemma 3 across India, East Asia and Southeast Asia. Our study specifically focuses on the sensitive domain of religion as the prism for broader alignment. To facilitate this, we conduct a multi-faceted analysis of every LLM's internal representations, using log-probs/logits, to compare the model's opinion distributions against ground-truth public attitudes. We find that while the popular models generally align with public opinion on broad social issues, they consistently fail to accurately represent religious viewpoints, especially those of minority groups, often amplifying negative stereotypes. Lightweight interventions, such as demographic priming and native language prompting, partially mitigate but do not eliminate these cultural gaps. We further show that downstream evaluations on bias benchmarks (such as CrowS-Pairs, IndiBias, ThaiCLI, KoBBQ) reveal persistent harms and under-representation in sensitive contexts. Our findings underscore the urgent need for systematic, regionally grounded audits to ensure equitable global deployment of LLMs. 

---
# Wisdom of the AI Crowd (AI-CROWD) for Ground Truth Approximation in Content Analysis: A Research Protocol & Validation Using Eleven Large Language Models 

**Authors**: Luis de-Marcos, Manuel Goyanes, Adrián Domínguez-Díaz  

**Link**: [PDF](https://arxiv.org/pdf/2603.06197)  

**Abstract**: Large-scale content analysis is increasingly limited by the absence of observable ground truth or gold-standard labels, as creating such benchmarks through extensive human coding becomes impractical for massive datasets due to high time, cost, and consistency challenges. To overcome this barrier, we introduce the AI-CROWD protocol, which approximates ground truth by leveraging the collective outputs of an ensemble of large language models (LLMs). Rather than asserting that the resulting labels are true ground truth, the protocol generates a consensus-based approximation derived from convergent and divergent inferences across multiple models. By aggregating outputs via majority voting and interrogating agreement/disagreement patterns with diagnostic metrics, AI-CROWD identifies high-confidence classifications while flagging potential ambiguity or model-specific biases. 

---
# Evaluation of Deontic Conditional Reasoning in Large Language Models: The Case of Wason's Selection Task 

**Authors**: Hirohiko Abe, Kentaro Ozeki, Risako Ando, Takanobu Morishita, Koji Mineshima, Mitsuhiro Okada  

**Link**: [PDF](https://arxiv.org/pdf/2603.06416)  

**Abstract**: As large language models (LLMs) advance in linguistic competence, their reasoning abilities are gaining increasing attention. In humans, reasoning often performs well in domain specific settings, particularly in normative rather than purely formal contexts. Although prior studies have compared LLM and human reasoning, the domain specificity of LLM reasoning remains underexplored. In this study, we introduce a new Wason Selection Task dataset that explicitly encodes deontic modality to systematically distinguish deontic from descriptive conditionals, and use it to examine LLMs' conditional reasoning under deontic rules. We further analyze whether observed error patterns are better explained by confirmation bias (a tendency to seek rule-supporting evidence) or by matching bias (a tendency to ignore negation and select items that lexically match elements of the rule). Results show that, like humans, LLMs reason better with deontic rules and display matching-bias-like errors. Together, these findings suggest that the performance of LLMs varies systematically across rule types and that their error patterns can parallel well-known human biases in this paradigm. 

---
# Making Implicit Premises Explicit in Logical Understanding of Enthymemes 

**Authors**: Xuyao Feng, Anthony Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2603.06114)  

**Abstract**: Real-world arguments in text and dialogues are normally enthymemes (i.e. some of their premises and/or claims are implicit). Natural language processing (NLP) methods for handling enthymemes can potentially identify enthymemes in text but they do not decode their underlying logic, whereas logic-based approaches for handling them assume a knowledgebase with sufficient formulae that can be used to decode them via abduction. There is therefore a lack of a systematic method for translating textual components of an enthymeme into a logical argument and generating the logical formulae required for their decoding, and thereby showing logical entailment. To address this, we propose a pipeline that integrates: (1) a large language model (LLM) to generate intermediate implicit premises based on the explicit premise and claim; (2) another LLM to translate the natural language into logical formulas; and (3) a neuro-symbolic reasoner based on a SAT solver to determine entailment. We evaluate our pipeline on two enthymeme datasets, demonstrating promising performance in selecting the correct implicit premise, as measured by precision, recall, F1-score, and accuracy. 

---
# Evaluating Austrian A-Level German Essays with Large Language Models for Automated Essay Scoring 

**Authors**: Jonas Kubesch, Lena Huber, Clemens Havas  

**Link**: [PDF](https://arxiv.org/pdf/2603.06066)  

**Abstract**: Automated Essay Scoring (AES) has been explored for decades with the goal to support teachers by reducing grading workload and mitigating subjective biases. While early systems relied on handcrafted features and statistical models, recent advances in Large Language Models (LLMs) have made it possible to evaluate student writing with unprecedented flexibility. This paper investigates the application of state-of-the-art open-weight LLMs for the grading of Austrian A-level German texts, with a particular focus on rubric-based evaluation. A dataset of 101 anonymised student exams across three text types was processed and evaluated. Four LLMs, DeepSeek-R1 32b, Qwen3 30b, Mixtral 8x7b and LLama3.3 70b, were evaluated with different contexts and prompting strategies. The LLMs were able to reach a maximum of 40.6% agreement with the human rater in the rubric-provided sub-dimensions, and only 32.8% of final grades matched the ones given by a human expert. The results indicate that even though smaller models are able to use standardised rubrics for German essay grading, they are not accurate enough to be used in a real-world grading environment. 

---
# Experiences Build Characters: The Linguistic Origins and Functional Impact of LLM Personality 

**Authors**: Xi Wang, Mengdie Zhuang, Jiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06088)  

**Abstract**: Human problem-solving is enriched by a diversity of styles and personality traits, yet the development of Large Language Models (LLMs) has largely prioritized uniform performance benchmarks that favour specific behavioural tendencies such as assertiveness. To investigate how diverse experiences shape machine personality and influence problem-solving, this study employs continued pre-training to expose models to domain-specific texts in an unsupervised manner, simulating the accumulation of experience. By adapting the Big Five framework via the Machine Personality Inventory (MPI), we quantify the personality traits of these model variants and analyse their relationship to linguistic style and reasoning behaviour. The findings reveal that model competence is bimodal, peaking at "Expressive Generalists" and "Suppressed Specialists," while identifying a "Suppression Advantage" where reduced social traits enhance complex reasoning performance. This study further establishes a causal link between training data linguistics, such as imperative frequency, and lexical diversity, providing a roadmap for "Personality Engineering". 

---
# Building an Ensemble LLM Semantic Tagger for UN Security Council Resolutions 

**Authors**: Hussein Ghaly  

**Link**: [PDF](https://arxiv.org/pdf/2603.05895)  

**Abstract**: This paper introduces a new methodology for using LLM-based systems for accurate and efficient semantic tagging of UN Security Council resolutions. The main goal is to leverage LLM performance variability to build ensemble systems for data cleaning and semantic tagging tasks. We introduce two evaluation metrics: Content Preservation Ratio (CPR) and Tag Well-Formedness (TWF), in order to avoid hallucinations and unnecessary additions or omissions to the input text beyond the task requirement. These metrics allow the selection of the best output from multiple runs of several GPT models. GPT-4.1 achieved the highest metrics for both tasks (Cleaning: CPR 84.9% - Semantic Tagging: CPR 99.99% and TWF 99.92%). In terms of cost, smaller models, such as GPT-4.1-mini, achieved comparable performance to the best model in each task at only 20% of the cost. These metrics ultimately allowed the ensemble to select the optimal output (both cleaned and tagged content) for all the LLM models involved, across multiple runs. With this ensemble design and the use of metrics, we create a reliable LLM system for performing semantic tagging on challenging texts. 

---
# Implicit Style Conditioning: A Structured Style-Rewrite Framework for Low-Resource Character Modeling 

**Authors**: Chanhui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.05933)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing (RP); however, small Language Models (SLMs) with highly stylized personas remains a challenge due to data scarcity and the complexity of style disentanglement. Standard Supervised Fine-Tuning (SFT) often captures surface-level semantics while failing to reproduce the intricate syntactic and pragmatic nuances of a character, leading to "Out-Of-Character" (OOC) generation. To address this, we propose a Structured Style-Rewrite Framework that explicitly disentangles style into three interpretable dimensions: lexical signatures (via PMI), syntactic patterns (grounded in PCFG rules), and pragmatic style. Furthermore, we introduce an implicit style conditioning strategy via Chain-of-Thought (CoT) distillation. By leveraging explicit reasoning traces during training as a strong inductive bias, our approach aligns the model's latent representations with structured style features, enabling high-fidelity stylized generation without requiring explicit reasoning tokens during inference. Extensive experiments on a specific high-stylization domain (anime characters) demonstrate that our method enables a Qwen-1.7B model to outperform significantly larger baselines (e.g., 4B Vanilla SFT) in style consistency and semantic fidelity. Our approach offers a data-efficient paradigm for democratizing inference and deployment on consumer hardware. 

---
# SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models 

**Authors**: Yunlong Chu, Minglai Shao, Yuhang Liu, Bing Hao, Yumeng Lin, Jialu Wang, Ruijie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06222)  

**Abstract**: Explicit Chain-of-Thought improves the reasoning performance of large language models but often incurs high inference cost due to verbose token-level traces. While recent approaches reduce this overhead via concise prompting or step pruning, they largely truncate what the model says rather than internalize what the model thinks. Latent reasoning offers a promising alternative by performing computation in the hidden space, yet prior methods face two critical challenges. Many existing approaches rely on rigid point-to-point alignment, forcing a latent token to approximate the final representation of a reasoning step, which can be insufficient to capture the dense, variable-length semantics of an entire reasoning segment. Furthermore, these methods often suffer from a lack of interpretability: latent states are commonly produced by unconstrained optimization or embedding mixing, yielding vectors that are difficult to decode or audit under the pretrained language head. We propose SPOT, a flexible framework that compresses explicit CoT into compact latent pause tokens without enforcing a fixed response template. At the core of SPOT is Span-level Semantic Alignment, a Sinkhorn optimal-transport objective that softly matches each pause token to the semantics of an entire reasoning segment, overcoming the rigidity of step-end alignment. To further improve interpretability, SPOT introduces a Frozen-Head Decoding Constraint that keeps latent states directly decodable as token distributions under the frozen pretrained LM head, enabling readable keyword interpretations of latent thoughts. Experiments on reasoning benchmarks demonstrate that SPOT improves accuracy by 2.3 points on average while reducing generated tokens by 37.5% and provides faithful semantic interpretations of the latent reasoning process. 

---
# ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning 

**Authors**: Juyong Jiang, Jiasi Shen, Sunghun Kim, Kang Min Yoo, Jeonghoon Kim, Sungju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.05863)  

**Abstract**: While Large Language Models (LLMs) have revolutionized code generation, standard "System 1" approaches, generating solutions in a single forward pass, often hit a performance ceiling when faced with complex algorithmic tasks. Existing iterative refinement strategies attempt to bridge this gap at inference time, yet they predominantly rely on external oracles, execution feedback, or computationally expensive prompt-response cycles. In this work, we propose ReflexiCoder, a novel reinforcement learning (RL) framework that internalizes the structured reasoning trajectory, encompassing initial generation, bug and optimization aware reflection, and self-correction, directly into the model's weights. Unlike prior methods, ReflexiCoder shifts the paradigm from external-dependent refinement to an intrinsic, fully autonomous self-reflection and self-correction capabilities at inference time. We utilize an RL-zero training paradigm with granular reward functions to optimize the entire reflection-correction trajectory, teaching the model how to debug without reliance on ground-truth feedback or execution engines at inference time. Extensive experiments across seven benchmarks demonstrate that our ReflexiCoder-8B establishes a new state-of-the-art (SOTA) among leading open-source models in the 1.5B-14B range, achieving 94.51% (87.20%) on HumanEval (Plus), 81.80% (78.57%) on MBPP (Plus), 35.00% on BigCodeBench, 52.21% on LiveCodeBench, and 37.34% on CodeForces in a single-attempt setting, rivaling or surpassing proprietary models like GPT-5.1. Notably, our framework is significantly more token-efficient than base models, reducing inference-time compute overhead by approximately 40% through disciplined, high-speed reasoning and reflection patterns. Source code is available at this https URL. 

---
# HART: Data-Driven Hallucination Attribution and Evidence-Based Tracing for Large Language Models 

**Authors**: Shize Liang, Hongzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05828)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance in text generation and knowledge-intensive question answering. Nevertheless, they are prone to producing hallucinated content, which severely undermines their reliability in high-stakes application domains. Existing hallucination attribution approaches, based on either external knowledge retrieval or internal model mechanisms, primarily focus on semantic similarity matching or representation-level discrimination. As a result, they have difficulty establishing structured correspondences at the span level between hallucination types, underlying error generation mechanisms, and external factual evidence, thereby limiting the interpretability of hallucinated fragments and the traceability of supporting or opposing evidence. To address these limitations, we propose HART, a fine-grained hallucination attribution and evidence retrieval framework for large language models. HART formalizes hallucination tracing as a structured modeling task comprising four stages: span localization, mechanism attribution, evidence retrieval, and causal tracing. Based upon this formulation, we develop the first structured dataset tailored for hallucination tracing, in which hallucination types, error mechanisms, and sets of counterfactual evidence are jointly annotated to enable causal-level interpretability evaluation. Experimental results on the proposed dataset demonstrate that HART substantially outperforms strong retrieval baselines, including BM25 and DPR, validating the effectiveness and generalization capability of the proposed tracing paradigm for hallucination analysis and evidence alignment. 

---
# Lost in Stories: Consistency Bugs in Long Story Generation by LLMs 

**Authors**: Junjie Li, Xinrui Guo, Yuhao Wu, Roy Ka-Wei Lee, Hongzhi Li, Yutao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.05890)  

**Abstract**: What happens when a storyteller forgets its own story? Large Language Models (LLMs) can now generate narratives spanning tens of thousands of words, but they often fail to maintain consistency throughout. When generating long-form narratives, these models can contradict their own established facts, character traits, and world rules. Existing story generation benchmarks focus mainly on plot quality and fluency, leaving consistency errors largely unexplored. To address this gap, we present ConStory-Bench, a benchmark designed to evaluate narrative consistency in long-form story generation. It contains 2,000 prompts across four task scenarios and defines a taxonomy of five error categories with 19 fine-grained subtypes. We also develop ConStory-Checker, an automated pipeline that detects contradictions and grounds each judgment in explicit textual evidence. Evaluating a range of LLMs through five research questions, we find that consistency errors show clear tendencies: they are most common in factual and temporal dimensions, tend to appear around the middle of narratives, occur in text segments with higher token-level entropy, and certain error types tend to co-occur. These findings can inform future efforts to improve consistency in long-form narrative generation. Our project page is available at this https URL. 

---
# PVminerLLM: Structured Extraction of Patient Voice from Patient-Generated Text using Large Language Models 

**Authors**: Samah Fodeh, Linhai Ma, Ganesh Puthiaraju, Srivani Talakokkul, Afshan Khan, Ashley Hagaman, Sarah Lowe, Aimee Roundtree  

**Link**: [PDF](https://arxiv.org/pdf/2603.05776)  

**Abstract**: Motivation: Patient-generated text contains critical information about patients' lived experiences, social circumstances, and engagement in care, including factors that strongly influence adherence, care coordination, and health equity. However, these patient voice signals are rarely available in structured form, limiting their use in patient-centered outcomes research and clinical quality improvement. Reliable extraction of such information is therefore essential for understanding and addressing non-clinical drivers of health outcomes at scale.
Results: We introduce PVminer, a benchmark for structured extraction of patient voice, and propose PVminerLLM, a supervised fine-tuned large language model tailored to this task. Across multiple datasets and model sizes, PVminerLLM substantially outperforms prompt-based baselines, achieving up to 83.82% F1 for Code prediction, 80.74% F1 for Sub-code prediction, and 87.03% F1 for evidence Span extraction. Notably, strong performance is achieved even with smaller models, demonstrating that reliable patient voice extraction is feasible without extreme model scale. These results enable scalable analysis of social and experiential signals embedded in patient-generated text.
Availability and Implementation: Code, evaluation scripts, and trained LLMs will be released publicly. Annotated datasets will be made available upon request for research use.
Keywords: Large Language Models, Supervised Fine-Tuning, Medical Annotation, Patient-Generated Text, Clinical NLP 

---
# Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis 

**Authors**: Hazem Amamou, Stéphane Gagnon, Alan Davoust, Anderson R. Avila  

**Link**: [PDF](https://arxiv.org/pdf/2603.05698)  

**Abstract**: Retrieval-Augmented Generation (RAG) was introduced to enhance the capabilities of Large Language Models (LLMs) beyond their encoded prior knowledge. This is achieved by providing LLMs with an external source of knowledge, which helps reduce factual hallucinations and enables access to new information not available during pretraining. However, inconsistent retrieved information can negatively affect LLM responses. The Retrieval-Augmented Generation Benchmark (RGB) was introduced to evaluate the robustness of RAG systems under such conditions. In this work, we use the RGB corpus to evaluate LLMs in four scenarios: noise robustness, information integration, negative rejection, and counterfactual robustness. We perform a comparative analysis between the RGB RAG baseline and GraphRAG, a knowledge graph based retrieval system. We test three GraphRAG customizations to improve robustness. Results show improvements over the RGB baseline and provide insights for designing more reliable RAG systems for real world scenarios. 

---
# Safer Reasoning Traces: Measuring and Mitigating Chain-of-Thought Leakage in LLMs 

**Authors**: Patrick Ahrend, Tobias Eder, Xiyang Yang, Zhiyi Pan, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2603.05618)  

**Abstract**: Chain-of-Thought (CoT) prompting improves LLM reasoning but can increase privacy risk by resurfacing personally identifiable information (PII) from the prompt into reasoning traces and outputs, even under policies that instruct the model not to restate PII. We study such direct, inference-time PII leakage using a model-agnostic framework that (i) defines leakage as risk-weighted, token-level events across 11 PII types, (ii) traces leakage curves as a function of the allowed CoT budget, and (iii) compares open- and closed-source model families on a structured PII dataset with a hierarchical risk taxonomy. We find that CoT consistently elevates leakage, especially for high-risk categories, and that leakage is strongly family- and budget-dependent. Increasing the reasoning budget can either amplify or attenuate leakage depending on the base model. We then benchmark lightweight inference-time gatekeepers: a rule-based detector, a TF-IDF + logistic regression classifier, a GLiNER-based NER model, and an LLM-as-judge, using risk-weighted F1, Macro-F1, and recall. No single method dominates across models or budgets, motivating hybrid, style-adaptive gatekeeping policies that balance utility and risk under a common, reproducible protocol. 

---
# The Fragility Of Moral Judgment In Large Language Models 

**Authors**: Tom van Nuenen, Pratik S. Sachdeva  

**Link**: [PDF](https://arxiv.org/pdf/2603.05651)  

**Abstract**: People increasingly use large language models (LLMs) for everyday moral and interpersonal guidance, yet these systems cannot interrogate missing context and judge dilemmas as presented. We introduce a perturbation framework for testing the stability and manipulability of LLM moral judgments while holding the underlying moral conflict constant. Using 2,939 dilemmas from r/AmItheAsshole (January-March 2025), we generate three families of content perturbations: surface edits (lexical/structural noise), point-of-view shifts (voice and stance neutralization), and persuasion cues (self-positioning, social proof, pattern admissions, victim framing). We also vary the evaluation protocol (output ordering, instruction placement, and unstructured prompting). We evaluated all variants with four models (GPT-4.1, Claude 3.7 Sonnet, DeepSeek V3, Qwen2.5-72B) (N=129,156 judgments).
Surface perturbations produce low flip rates (7.5%), largely within the self-consistency noise floor (4-13%), whereas point-of-view shifts induce substantially higher instability (24.3%). A large subset of dilemmas (37.9%) is robust to surface noise yet flips under perspective changes, indicating that models condition on narrative voice as a pragmatic cue. Instability concentrates in morally ambiguous cases; scenarios where no party is assigned blame are most susceptible. Persuasion perturbations yield systematic directional shifts. Protocol choices dominate all other factors: agreement between structured protocols is only 67.6% (kappa=0.55), and only 35.7% of model-scenario units match across all three protocols. These results show that LLM moral judgments are co-produced by narrative form and task scaffolding, raising reproducibility and equity concerns when outcomes depend on presentation skill rather than moral substance. 

---
# COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics 

**Authors**: Kartik Sharma, Rakshit S. Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2603.06495)  

**Abstract**: Activation steering methods enable inference-time control of large language model (LLM) behavior without retraining, but current approaches face a fundamental trade-off: sample-efficient methods suboptimally capture steering signals from labeled examples, while methods that better extract these signals require hundreds to thousands of examples. We introduce COLD-Steer, a training-free framework that steers LLM activations by approximating the representational changes that would result from gradient descent on in-context examples. Our key insight is that the effect of fine-tuning on a small set of examples can be efficiently approximated at inference time without actual parameter updates. We formalize this through two complementary approaches: (i) a unit kernel approximation method that updates the activations directly using gradients with respect to them, normalized across examples, and (ii) a finite-difference approximation requiring only two forward passes regardless of example count. Experiments across a variety of steering tasks and benchmarks demonstrate that COLD-Steer achieves upto 95% steering effectiveness while using 50 times fewer samples compared to the best baseline. COLD-Steer facilitates accommodating diverse perspectives without extensive demonstration data, which we validate through our experiments on pluralistic alignment tasks. Our framework opens new possibilities for adaptive, context-aware model control that can flexibly address varying loss-driven human preferences through principled approximation of learning dynamics rather than specialized training procedures. 

---
# NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution 

**Authors**: Oleksandr Marchenko Breneur, Adelaide Danilov, Aria Nourbakhsh, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2603.05617)  

**Abstract**: We present this http URL, an explainable framework for machine-generated text detection that extends Fast-DetectGPT by integrating curvature-based signals with neural and stylometric features in a supervised setting. The system combines 17 interpretable features, including Conditional Probability Curvature, ModernBERT detector score, readability metrics, and stylometric cues, within a gradient-boosted tree (XGBoost) meta-classifier to determine whether a text is human- or AI-generated. Furthermore, this http URL applies Shapley Additive Explanations (SHAP) to provide both local and global feature-level attribution. These attributions are further translated into structured natural-language rationales through an LLM-based explanation layer, which enables user-facing interpretability. The system is deployed as an interactive web application that supports real-time analysis, visual feature inspection, and structured evidence presentation. A web interface allows users to input text and inspect how neural and statistical signals influence the final decision. The source code and demo video are publicly available to support reproducibility. 

---
# Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation 

**Authors**: Changcheng Li, Jiancan Wu, Hengheng Zhang, Zhengsu Chen, Guo An, Junxiang Qiu, Xiang Wang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.05881)  

**Abstract**: Reliable deployment of large language models (LLMs) requires accurate uncertainty estimation. Existing methods are predominantly answer-first, producing confidence only after generating an answer, which measure the correctness of a specific response and limits practical usability. We study a confidence-first paradigm, where the model outputs its confidence before answering, interpreting this score as the model's probability of answering the question correctly under its current policy.
We propose CoCA(Co-optimized Confidence and Answers), a GRPO reinforcement learning framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. By assigning separate rewards and group-relative advantages to confidence and answer segments, CoCA enables stable joint optimization and avoids reward hacking. Experiments across math, code, and factual QA benchmarks show improved calibration and uncertainty discrimination while preserving answer quality, thereby enabling a broader range of downstream applications. 

---
# Autonomous Algorithm Discovery for Ptychography via Evolutionary LLM Reasoning 

**Authors**: Xiangyu Yin, Ming Du, Junjing Deng, Zhi Yang, Yimo Han, Yi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05696)  

**Abstract**: Ptychography is a computational imaging technique widely used for high-resolution materials characterization, but high-quality reconstructions often require the use of regularization functions that largely remain manually designed. We introduce Ptychi-Evolve, an autonomous framework that uses large language models (LLMs) to discover and evolve novel regularization algorithms. The framework combines LLM-driven code generation with evolutionary mechanisms, including semantically-guided crossover and mutation. Experiments on three challenging datasets (X-ray integrated circuits, low-dose electron microscopy of apoferritin, and multislice imaging with crosstalk artifacts) demonstrate that discovered regularizers outperform conventional reconstructions, achieving up to +0.26 SSIM and +8.3~dB PSNR improvements. Besides, Ptychi-Evolve records algorithm lineage and evolution metadata, enabling interpretable and reproducible analysis of discovered regularizers. 

---
# RACAS: Controlling Diverse Robots With a Single Agentic System 

**Authors**: Dylan R. Ashley, Jan Przepióra, Yimeng Chen, Ali Abualsaud, Nurzhan Yesmagambet, Shinkyu Park, Eric Feron, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2603.05621)  

**Abstract**: Many robotic platforms expose an API through which external software can command their actuators and read their sensors. However, transitioning from these low-level interfaces to high-level autonomous behaviour requires a complicated pipeline, whose components demand distinct areas of expertise. Existing approaches to bridging this gap either require retraining for every new embodiment or have only been validated across structurally similar platforms. We introduce RACAS (Robot-Agnostic Control via Agentic Systems), a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. RACAS requires only a natural language description of the robot, a definition of available actions, and a task specification; no source code, model weights, or reward functions need to be modified to move between platforms. We evaluate RACAS on several tasks using a wheeled ground robot, a recently published novel multi-jointed robotic limb, and an underwater vehicle. RACAS consistently solved all assigned tasks across these radically different platforms, demonstrating the potential of agentic AI to substantially reduce the barrier to prototyping robotic solutions. 

---
# An Interactive Multi-Agent System for Evaluation of New Product Concepts 

**Authors**: Bin Xuan, Ruo Ai, Hakyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.05980)  

**Abstract**: Product concept evaluation is a critical stage that determines strategic resource allocation and project success in enterprises. However, traditional expert-led approaches face limitations such as subjective bias and high time and cost requirements. To support this process, this study proposes an automated approach utilizing a large language model (LLM)-based multi-agent system (MAS). Through a systematic analysis of previous research on product development and team collaboration, this study established two primary evaluation dimensions, namely technical feasibility and market feasibility. The proposed system consists of a team of eight virtual agents representing specialized domains such as R&D and marketing. These agents use retrieval-augmented generation (RAG) and real-time search tools to gather objective evidence and validate concepts through structured deliberations based on the established criteria. The agents were further fine-tuned using professional product review data to enhance their judgment accuracy. A case study involving professional display monitor concepts demonstrated that the system's evaluation rankings were consistent with those of senior industry experts. These results confirm the usability of the proposed multi-agent-based evaluation approach for supporting product development decisions. 

---
# Agentic LLM Planning via Step-Wise PDDL Simulation: An Empirical Characterisation 

**Authors**: Kai Göbel, Pierrick Lorang, Patrik Zips, Tobias Glück  

**Link**: [PDF](https://arxiv.org/pdf/2603.06064)  

**Abstract**: Task planning, the problem of sequencing actions to reach a goal from an initial state, is a core capability requirement for autonomous robotic systems. Whether large language models (LLMs) can serve as viable planners alongside classical symbolic methods remains an open question. We present PyPDDLEngine, an open-source Planning Domain Definition Language (PDDL) simulation engine that exposes planning operations as LLM tool calls through a Model Context Protocol (MCP) interface. Rather than committing to a complete action sequence upfront, the LLM acts as an interactive search policy that selects one action at a time, observes each resulting state, and can reset and retry. We evaluate four approaches on 102 International Planning Competition (IPC) Blocksworld instances under a uniform 180-second budget: Fast Downward lama-first and seq-sat-lama-2011 as classical baselines, direct LLM planning (Claude Haiku 4.5), and agentic LLM planning via PyPDDLEngine. Fast Downward achieves 85.3% success. The direct and agentic LLM approaches achieve 63.7% and 66.7%, respectively, a consistent but modest three-percentage-point advantage for the agentic approach at $5.7\times$ higher token cost per solution. Across most co-solved difficulty blocks, both LLM approaches produce shorter plans than seq-sat-lama-2011 despite its iterative quality improvement, a result consistent with training-data recall rather than generalisable planning. These results suggest that agentic gains depend on the nature of environmental feedback. Coding agents benefit from externally grounded signals such as compiler errors and test failures, whereas PDDL step feedback is self-assessed, leaving the agent to evaluate its own progress without external verification. 

---
# Reasoning Models Struggle to Control their Chains of Thought 

**Authors**: Chen Yueh-Han, Robert McCarthy, Bruce W. Lee, He He, Ian Kivlichan, Bowen Baker, Micah Carroll, Tomek Korbak  

**Link**: [PDF](https://arxiv.org/pdf/2603.05706)  

**Abstract**: Chain-of-thought (CoT) monitoring is a promising tool for detecting misbehaviors and understanding the motivations of modern reasoning models. However, if models can control what they verbalize in their CoT, it could undermine CoT monitorability. To measure this undesirable capability -- CoT controllability -- we introduce the CoT-Control evaluation suite, which includes tasks that require models to solve problems while adhering to CoT instructions, e.g., reasoning about a genetics question without using the word 'chromosome'. We show that reasoning models possess significantly lower CoT controllability than output controllability; for instance, Claude Sonnet 4.5 can control its CoT only 2.7% of the time but 61.9% when controlling its final output. We also find that CoT controllability is higher for larger models and decreases with more RL training, test-time compute, and increased problem difficulty. CoT controllability failures extend even to situations in which models are given incentives (as opposed to direct requests) to evade CoT monitors, although models exhibit slightly higher controllability when they are told they are being monitored. Similarly, eliciting controllability by adversarially optimizing prompts does not meaningfully increase controllability. Our results leave us cautiously optimistic that CoT controllability is currently unlikely to be a failure mode of CoT monitorability. However, the mechanism behind low controllability is not well understood. Given its importance for maintaining CoT monitorability, we recommend that frontier labs track CoT controllability in future models. 

---
# BEVLM: Distilling Semantic Knowledge from LLMs into Bird's-Eye View Representations 

**Authors**: Thomas Monninger, Shaoyuan Xie, Qi Alfred Chen, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.06576)  

**Abstract**: The integration of Large Language Models (LLMs) into autonomous driving has attracted growing interest for their strong reasoning and semantic understanding abilities, which are essential for handling complex decision-making and long-tail scenarios. However, existing methods typically feed LLMs with tokens from multi-view and multi-frame images independently, leading to redundant computation and limited spatial consistency. This separation in visual processing hinders accurate 3D spatial reasoning and fails to maintain geometric coherence across views. On the other hand, Bird's-Eye View (BEV) representations learned from geometrically annotated tasks (e.g., object detection) provide spatial structure but lack the semantic richness of foundation vision encoders. To bridge this gap, we propose BEVLM, a framework that connects a spatially consistent and semantically distilled BEV representation with LLMs. Through extensive experiments, we show that BEVLM enables LLMs to reason more effectively in cross-view driving scenes, improving accuracy by 46%, by leveraging BEV features as unified inputs. Furthermore, by distilling semantic knowledge from LLMs into BEV representations, BEVLM significantly improves closed-loop end-to-end driving performance by 29% in safety-critical scenarios. 

---
# Agentic retrieval-augmented reasoning reshapes collective reliability under model variability in radiology question answering 

**Authors**: Mina Farajiamiri, Jeta Sopa, Saba Afza, Lisa Adams, Felix Barajas Ordonez, Tri-Thien Nguyen, Mahshad Lotfinia, Sebastian Wind, Keno Bressem, Sven Nebelung, Daniel Truhn, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2603.06271)  

**Abstract**: Agentic retrieval-augmented reasoning pipelines are increasingly used to structure how large language models (LLMs) incorporate external evidence in clinical decision support. These systems iteratively retrieve curated domain knowledge and synthesize it into structured reports before answer selection. Although such pipelines can improve performance, their impact on reliability under model variability remains unclear. In real-world deployment, heterogeneous models may align, diverge, or synchronize errors in ways not captured by accuracy. We evaluated 34 LLMs on 169 expert-curated publicly available radiology questions, comparing zero-shot inference with a radiology-specific multi-step agentic retrieval condition in which all models received identical structured evidence reports derived from curated radiology knowledge. Agentic inference reduced inter-model decision dispersion (median entropy 0.48 vs. 0.13) and increased robustness of correctness across models (mean 0.74 vs. 0.81). Majority consensus also increased overall (P<0.001). Consensus strength and robust correctness remained correlated under both strategies (\r{ho}=0.88 for zero-shot; \r{ho}=0.87 for agentic), although high agreement did not guarantee correctness. Response verbosity showed no meaningful association with correctness. Among 572 incorrect outputs, 72% were associated with moderate or high clinically assessed severity, although inter-rater agreement was low (\k{appa}=0.02). Agentic retrieval therefore was associated with more concentrated decision distributions, stronger consensus, and higher cross-model robustness of correctness. These findings suggest that evaluating agentic systems through accuracy or agreement alone may not always be sufficient, and that complementary analyses of stability, cross-model robustness, and potential clinical impact are needed to characterize reliability under model variability. 

---
# From Entropy to Calibrated Uncertainty: Training Language Models to Reason About Uncertainty 

**Authors**: Azza Jenane, Nassim Walha, Lukas Kuhn, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2603.06317)  

**Abstract**: Large Language Models (LLMs) that can express interpretable and calibrated uncertainty are crucial in high-stakes domains. While methods to compute uncertainty post-hoc exist, they are often sampling-based and therefore computationally expensive or lack calibration. We propose a three-stage pipeline to post-train LLMs to efficiently infer calibrated uncertainty estimates for their responses. First, we compute fine-grained entropy-based uncertainty scores on the training data, capturing the distributional variability of model outputs in embedding space. Second, these scores are calibrated via Platt scaling, producing reliable and human-interpretable uncertainty signals. Finally, the target LLM is post-trained via reinforcement learning to align its policy with these calibrated signals through a verifiable reward function. Unlike post-hoc uncertainty estimation methods, our approach provides interpretable and computationally efficient uncertainty estimates at test time. Experiments show that models trained with our pipeline achieve better calibration than baselines and generalize to unseen tasks without further processing, suggesting that they learn a robust uncertainty reasoning behavior. 

---
# Structured Exploration vs. Generative Flexibility: A Field Study Comparing Bandit and LLM Architectures for Personalised Health Behaviour Interventions 

**Authors**: Dominik P. Hofer, Haochen Song, Rania Islambouli, Laura Hawkins, Ananya Bhattacharjee, Meredith Franklin, Joseph Jay Williams, Jan D. Smeddinck  

**Link**: [PDF](https://arxiv.org/pdf/2603.06330)  

**Abstract**: Behaviour Change Techniques (BCTs) are central to digital health interventions, yet selecting and delivering effective techniques remains challenging. Contextual bandits enable statistically grounded optimisation of BCT selection, while Large Language Models (LLMs) offer flexible, context-sensitive message generation. We conducted a 4-week study on physical activity motivation (N=54; 9 post-study interviews) that compared five daily messaging approaches: random templates, contextual bandit with templates, LLM generation, hybrid bandit+LLM, and LLM with interaction history. LLM-based approaches were rated substantially more helpful than templates, but no significant differences emerged among LLM conditions. Unexpectedly, bandit optimisation for BCTs selection yielded no additional perceived helpfulness compared with LLM-only approaches. Unconstrained LLMs focused heavily on a single BCT, whereas bandit systems enforced systematic exploration-exploitation across techniques. Quantitative and qualitative findings suggest contextual acknowledgement of user input drove perceived helpfulness. We contribute design suggestions for reflective AI health behaviour change systems that address a trade-off between structured exploration and generative autonomy. 

---
# Place-it-R1: Unlocking Environment-aware Reasoning Potential of MLLM for Video Object Insertion 

**Authors**: Bohai Gu, Taiyi Wu, Dazhao Du, Jian Liu, Shuai Yang, Xiaotong Zhao, Alan Zhao, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.06140)  

**Abstract**: Modern video editing techniques have achieved high visual fidelity when inserting video objects. However, they focus on optimizing visual fidelity rather than physical causality, leading to edits that are physically inconsistent with their environment. In this work, we present Place-it-R$1$, an end-to-end framework for video object insertion that unlocks the environment-aware reasoning potential of Multimodal Large Language Models (MLLMs). Our framework leverages the Chain-of-Thought (CoT) reasoning of MLLMs to orchestrate video diffusion, following a Think-then-Place paradigm. To bridge cognitive reasoning and generative execution, we introduce three key innovations: First, MLLM performs physical scene understanding and interaction reasoning, generating environment-aware chain-of-thought tokens and inferring valid insertion regions to explicitly guide the diffusion toward physically plausible insertion. Then, we introduce MLLM-guided Spatial Direct Preference Optimization (DPO), where diffusion outputs are fed back to the MLLM for scoring, enabling visual naturalness. During inference, the MLLM iteratively triggers refinement cycles and elicits adaptive adjustments from the diffusion model, forming a closed-loop that progressively enhances editing quality. Furthermore, we provide two user-selectable modes: a plausibility-oriented flexible mode that permits environment modifications (\eg, generating support structures) to enhance physical plausibility, and a fidelity-oriented standard mode that preserves scene integrity for maximum fidelity, offering users explicit control over the plausibility-fidelity trade-off. Extensive experiments demonstrate Place-it-R1 achieves physically-coherent video object insertion compared with state-of-the-art solutions and commercial models. 

---
# MM-ISTS: Cooperating Irregularly Sampled Time Series Forecasting with Multimodal Vision-Text LLMs 

**Authors**: Zhi Lei, Chenxi Liu, Hao Miao, Wanghui Qiu, Bin Yang, Chenjuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.05997)  

**Abstract**: Irregularly sampled time series (ISTS) are widespread in real-world scenarios, exhibiting asynchronous observations on uneven time intervals across variables. Existing ISTS forecasting methods often solely utilize historical observations to predict future ones while falling short in learning contextual semantics and fine-grained temporal patterns. To address these problems, we achieve MM-ISTS, a multimodal framework augmented by vision-text large language models, that bridges temporal, visual, and textual modalities, facilitating ISTS forecasting. MM-ISTS encompasses a novel two-stage encoding mechanism. In particular, a cross-modal vision-text encoding module is proposed to automatically generate informative visual images and textual data, enabling the capture of intricate temporal patterns and comprehensive contextual understanding, in collaboration with multimodal LLMs (MLLMs). In parallel, ISTS encoding extracts complementary yet enriched temporal features from historical ISTS observations, including multi-view embedding fusion and a temporal-variable encoder. Further, we propose an adaptive query-based feature extractor to compress the learned tokens of MLLMs, filtering out small-scale useful knowledge, which in turn reduces computational costs. In addition, a multimodal alignment module with modality-aware gating is designed to alleviate the modality gap across ISTS, images, and text. Extensive experiments on real data offer insight into the effectiveness of the proposed solutions. 

---
# Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning 

**Authors**: Xuan Li, Zhanke Zhou, Zongze Li, Jiangchao Yao, Yu Rong, Lu Zhang, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.05900)  

**Abstract**: Large language models (LLMs) benefit substantially from supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR) in reasoning tasks. However, these recipes perform poorly in instruction-based molecular optimization, where each data point typically provides only a single optimized reference molecule and no step-by-step optimization trajectory. We reveal that answer-only SFT on the reference molecules collapses reasoning, and RLVR provides sparse feedback under similarity constraints due to the model's lack of effective exploration, which slows learning and limits optimization. To encourage the exploration of new molecules while balancing the exploitation of the reference molecules, we introduce Reference-guided Policy Optimization (RePO), an optimization approach that learns from reference molecules without requiring trajectory data. At each update, RePO samples candidate molecules with their intermediate reasoning trajectories from the model and trains the model using verifiable rewards that measure property satisfaction under similarity constraints in an RL manner. Meanwhile, it applies reference guidance by keeping the policy's intermediate reasoning trajectory as context and training only the answer in a supervised manner. Together, the RL term promotes exploration, while the guidance term mitigates reward sparsity and stabilizes training by grounding outputs to references when many valid molecular edits exist. Across molecular optimization benchmarks, RePO consistently outperforms SFT and RLVR baselines (e.g., GRPO), achieving improvements on the optimization metric (Success Rate $\times$ Similarity), improving balance across competing objectives, and generalizing better to unseen instruction styles. Our code is publicly available at this https URL. 

---
# Evaluating LLM Alignment With Human Trust Models 

**Authors**: Anushka Debnath, Stephen Cranefield, Bastin Tony Roy Savarimuthu, Emiliano Lorini  

**Link**: [PDF](https://arxiv.org/pdf/2603.05839)  

**Abstract**: Trust plays a pivotal role in enabling effective cooperation, reducing uncertainty, and guiding decision-making in both human interactions and multi-agent systems. Although it is significant, there is limited understanding of how large language models (LLMs) internally conceptualize and reason about trust. This work presents a white-box analysis of trust representation in EleutherAI/gpt-j-6B, using contrastive prompting to generate embedding vectors within the activation space of the LLM for diadic trust and related interpersonal relationship attributes. We first identified trust-related concepts from five established human trust models. We then determined a threshold for significant conceptual alignment by computing pairwise cosine similarities across 60 general emotional concepts. Then we measured the cosine similarities between the LLM's internal representation of trust and the derived trust-related concepts. Our results show that the internal trust representation of EleutherAI/gpt-j-6B aligns most closely with the Castelfranchi socio-cognitive model, followed by the Marsh Model. These findings indicate that LLMs encode socio-cognitive constructs in their activation space in ways that support meaningful comparative analyses, inform theories of social cognition, and support the design of human-AI collaborative systems. 

---
# LUMINA: LLM-Guided GPU Architecture Exploration via Bottleneck Analysis 

**Authors**: Tao Zhang, Rui Ma, Shuotao Xu, Peng Cheng, Yongqiang Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.05904)  

**Abstract**: GPU design space exploration (DSE) for modern AI workloads, such as Large-Language Model (LLM) inference, is challenging because of GPUs' vast, multi-modal design spaces, high simulation costs, and complex design optimization objectives (e.g. performance, power and area trade-offs). Existing automated DSE methods are often prohibitively expensive, either requiring an excessive number of exploration samples or depending on intricate, manually crafted analyses of interdependent critical paths guided by human heuristics.
We present LUMINA, an LLM-driven GPU architecture exploration framework that leverage AI to enhance the DSE efficiency and efficacy for GPUs. LUMINA extracts architectural knowledge from simulator code and performs sensitivity studies to automatically compose DSE rules,which are auto-corrected during exploration. A core component of LUMINA is a DSE Benchmark that comprehensively evaluates and enhances LLMs' capabilities across three fundamental skills required for architecture optimization, which provides a principled and reproducible basis for model selection and ensuring consistent architectural reasoning.
In the design space with 4.7 million possible samples, LUMINA identifies 6 designs of better performance and area than an A100 GPU efficiently, using only 20 steps via LLM-assisted bottleneck analysis. In comparison, LUMINA achieves 17.5x higher than design space exploration efficiency, and 32.9% better designs (i.e. Pareto Hypervolume) than Machine-Learning baselines, showcasing its ability to deliver high-quality design guidance with minimal search cost. 

---
# Ambiguity Collapse by LLMs: A Taxonomy of Epistemic Risks 

**Authors**: Shira Gur-Arieh, Angelina Wang, Sina Fazelpour  

**Link**: [PDF](https://arxiv.org/pdf/2603.05801)  

**Abstract**: Large language models (LLMs) are increasingly used to make sense of ambiguous, open-textured, value-laden terms. Platforms routinely rely on LLMs for content moderation, asking them to label text based on disputed concepts like "hate speech" or "incitement"; hiring managers may use LLMs to rank who counts as "qualified"; and AI labs increasingly train models to self-regulate under constitutional-style ambiguous principles such as "biased" or "legitimate". This paper introduces ambiguity collapse: a phenomenon that occurs when an LLM encounters a term that genuinely admits multiple legitimate interpretations, yet produces a singular resolution, in ways that bypass the human practices through which meaning is ordinarily negotiated, contested, and justified. Drawing on interdisciplinary accounts of ambiguity as a productive epistemic resource, we develop a taxonomy of the epistemic risks posed by ambiguity collapse at three levels: process (foreclosing opportunities to deliberate, develop cognitive skills, and shape contested terms), output (distorting the concepts and reasons agents act upon), and ecosystem (reshaping shared vocabularies, interpretive norms, and how concepts evolve over time). We illustrate these risks through three case studies, and conclude by sketching multi-layer mitigation principles spanning training, institutional deployment design, interface affordances, and the management of underspecified prompts, with the goal of designing systems that surface, preserve, and responsibly govern ambiguity. 

---
# StreamWise: Serving Multi-Modal Generation in Real-Time at Scale 

**Authors**: Haoran Qiu, Gohar Irfan Chaudhry, Chaojie Zhang, Íñigo Goiri, Esha Choukse, Rodrigo Fonseca, Ricardo Bianchini  

**Link**: [PDF](https://arxiv.org/pdf/2603.05800)  

**Abstract**: Advances in multi-modal generative models are enabling new applications, from storytelling to automated media synthesis. Most current workloads generate simple outputs (e.g., image generation from a prompt) in batch mode, often requiring several seconds even for basic results. Serving real-time multi-modal workflows at scale is costly and complex, requiring efficient coordination of diverse models (each with unique resource needs) across language, audio, image, and video, all under strict latency and resource constraints.
We tackle these challenges through the lens of real-time podcast video generation, integrating LLMs, text-to-speech, and video-audio generation. To meet tight SLOs, we design an adaptive, modular serving system, StreamWise, that dynamically manages quality (e.g., resolution, sharpness), model/content parallelism, and resource-aware scheduling. We leverage heterogeneous hardware to maximize responsiveness and efficiency. For example, the system can lower video resolution and allocate more resources to early scenes.
We quantify the trade-offs between latency, cost, and quality. The cheapest setup generates a 10-minute podcast video on A100 GPUs in 1.4 hours (8.4x slower than the real-time) for less than \$25. StreamWise enables high-quality real-time streaming with a sub-second startup delay under $45. 

---
# Lexara: A User-Centered Toolkit for Evaluating Large Language Models for Conversational Visual Analytics 

**Authors**: Srishti Palani, Vidya Setlur  

**Link**: [PDF](https://arxiv.org/pdf/2603.05832)  

**Abstract**: Large Language Models (LLMs) are transforming Conversational Visual Analytics (CVA) by enabling data analysis through natural language. However, evaluating LLMs for CVA remains a challenge: requiring programming expertise, overlooking real-world complexity, and lacking interpretable metrics for multi-format (visualizations and text) outputs. Through interviews with 22 CVA developers and 16 end-users, we identified use cases, evaluation criteria and workflows. We present Lexara, a user-centered evaluation toolkit for CVA that operationalizes these insights into: (i) test cases spanning real-world scenarios; (ii) interpretable metrics covering visualization quality (data fidelity, semantic alignment, functional correctness, design clarity) and language quality (factual grounding, analytical reasoning, conversational coherence) using rule-based and LLM-as-a-Judge methods; and (iii) an interactive toolkit enabling experimental setup and multi-format and multi-level exploration of results without programming expertise. We conducted a two-week diary study with six CVA developers, drawn from our initial cohort of 22. Their feedback demonstrated Lexara's effectiveness for guiding appropriate model and prompt selection. 

---
# Relational Semantic Reasoning on 3D Scene Graphs for Open World Interactive Object Search 

**Authors**: Imen Mahdi, Matteo Cassinelli, Fabien Despinoy, Tim Welschehold, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2603.05642)  

**Abstract**: Open-world interactive object search in household environments requires understanding semantic relationships between objects and their surrounding context to guide exploration efficiently. Prior methods either rely on vision-language embeddings similarity, which does not reliably capture task-relevant relational semantics, or large language models (LLMs), which are too slow and costly for real-time deployment. We introduce SCOUT: Scene Graph-Based Exploration with Learned Utility for Open-World Interactive Object Search, a novel method that searches directly over 3D scene graphs by assigning utility scores to rooms, frontiers, and objects using relational exploration heuristics such as room-object containment and object-object co-occurrence. To make this practical without sacrificing open-vocabulary generalization, we propose an offline procedural distillation framework that extracts structured relational knowledge from LLMs into lightweight models for on-robot inference. Furthermore, we present SymSearch, a scalable symbolic benchmark for evaluating semantic reasoning in interactive object search tasks. Extensive evaluations across symbolic and simulation environments show that SCOUT outperforms embedding similarity-based methods and matches LLM-level performance while remaining computationally efficient. Finally, real-world experiments demonstrate effective transfer to physical environments, enabling open-world interactive object search under realistic sensing and navigation constraints. 

---
# An Embodied Companion for Visual Storytelling 

**Authors**: Patrick Tresset, Markus Wulfmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.05511)  

**Abstract**: As artificial intelligence shifts from pure tool for delegation toward agentic collaboration, its use in the arts can shift beyond the exploration of machine autonomy toward synergistic co-creation. While our earlier robotic works utilized automation to distance the artist's intent from the final mark, we present Companion: an artistic apparatus that integrates a drawing robot with Large Language Models (LLMs) to re-center human-machine presence. By leveraging in-context learning and real-time tool use, the system engages in bidirectional interaction via speech and sketching. This approach transforms the robot from a passive executor into a playful co-creative partner capable of driving shared visual storytelling into unexpected aesthetic territories. To validate this collaborative shift, we employed the Consensual Assessment Technique (CAT) with a panel of seven art-world experts. Results confirm that the system produces works with a distinct aesthetic identity and professional exhibition merit, demonstrating the potential of AI as a highly capable artistic collaborator. 

---
# Can LLM Aid in Solving Constraints with Inductive Definitions? 

**Authors**: Weizhi Feng, Shidong Shen, Jiaxiang Liu, Taolue Chen, Fu Song, Zhilin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03668)  

**Abstract**: Solving constraints involving inductive (aka recursive) definitions is challenging. State-of-the-art SMT/CHC solvers and first-order logic provers provide only limited support for solving such constraints, especially when they involve, e.g., abstract data types. In this work, we leverage structured prompts to elicit Large Language Models (LLMs) to generate auxiliary lemmas that are necessary for reasoning about these inductive definitions. We further propose a neuro-symbolic approach, which synergistically integrates LLMs with constraint solvers: the LLM iteratively generates conjectures, while the solver checks their validity and usefulness for proving the goal. We evaluate our approach on a diverse benchmark suite comprising constraints originating from algebrai data types and recurrence relations. The experimental results show that our approach can improve the state-of-the-art SMT and CHC solvers, solving considerably more (around 25%) proof tasks involving inductive definitions, demonstrating its efficacy. 

---
