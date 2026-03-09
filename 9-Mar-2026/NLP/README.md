# KCLarity at SemEval-2026 Task 6: Encoder and Zero-Shot Approaches to Political Evasion Detection 

**Authors**: Archie Sage, Salvatore Greco  

**Link**: [PDF](https://arxiv.org/pdf/2603.06552)  

**Abstract**: This paper describes the KCLarity team's participation in CLARITY, a shared task at SemEval 2026 on classifying ambiguity and evasion techniques in political discourse. We investigate two modelling formulations: (i) directly predicting the clarity label, and (ii) predicting the evasion label and deriving clarity through the task taxonomy hierarchy. We further explore several auxiliary training variants and evaluate decoder-only models in a zero-shot setting under the evasion-first formulation. Overall, the two formulations yield comparable performance. Among encoder-based models, RoBERTa-large achieves the strongest results on the public test set, while zero-shot GPT-5.2 generalises better on the hidden evaluation set. 

---
# Speak in Context: Multilingual ASR with Speech Context Alignment via Contrastive Learning 

**Authors**: Yuchen Zhang, Haralambos Mouratidis, Ravi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2603.06505)  

**Abstract**: Automatic speech recognition (ASR) has benefited from advances in pretrained speech and language models, yet most systems remain constrained to monolingual settings and short, isolated utterances. While recent efforts in context-aware ASR show promise, two key challenges persist: limited multilingual support and the absence of principled alignment between speech and contextual representations. In this paper, we introduce a context-aware multilingual ASR framework that supports diverse languages and accents while preserving the modularity of pretrained models. Our approach combines a frozen speech encoder and a decoder-only language model via a lightweight projection module, allowing structured context prompts, including dialogue history and biasing words, to guide transcription. To improve interaction between speech and context, we employ a contrastive learning objective that aligns their representations in a shared embedding space. Evaluations on over 1,500 hours of real-world conversational speech across 11 languages and 5 English dialects show that contextual input consistently improves recognition quality. Contrastive alignment provides additional gains when applied to different context types, with an overall performance gain of over 5%. These results highlight the importance of both contextual modeling and cross-modal alignment in multilingual ASR. 

---
# Beyond Rows to Reasoning: Agentic Retrieval for Multimodal Spreadsheet Understanding and Editing 

**Authors**: Anmol Gulati, Sahil Sen, Waqar Sarguroh, Kevin Paul  

**Link**: [PDF](https://arxiv.org/pdf/2603.06503)  

**Abstract**: Recent advances in multimodal Retrieval-Augmented Generation (RAG) enable Large Language Models (LLMs) to analyze enterprise spreadsheet workbooks containing millions of cells, cross-sheet dependencies, and embedded visual artifacts. However, state-of-the-art approaches exclude critical context through single-pass retrieval, lose data resolution through compression, and exceed LLM context windows through naive full-context injection, preventing reliable multi-step reasoning over complex enterprise workbooks. We introduce Beyond Rows to Reasoning (BRTR), a multimodal agentic framework for spreadsheet understanding that replaces single-pass retrieval with an iterative tool-calling loop, supporting end-to-end Excel workflows from complex analysis to structured editing. Supported by over 200 hours of expert human evaluation, BRTR achieves state-of-the-art performance across three frontier spreadsheet understanding benchmarks, surpassing prior methods by 25 percentage points on FRTR-Bench, 7 points on SpreadsheetLLM, and 32 points on FINCH. We evaluate five multimodal embedding models, identifying NVIDIA NeMo Retriever 1B as the top performer for mixed tabular and visual data, and vary nine LLMs. Ablation experiments confirm that the planner, retrieval, and iterative reasoning each contribute substantially, and cost analysis shows GPT-5.2 achieves the best efficiency-accuracy trade-off. Throughout all evaluations, BRTR maintains full auditability through explicit tool-call traces. 

---
# PONTE: Personalized Orchestration for Natural Language Trustworthy Explanations 

**Authors**: Vittoria Vineis, Matteo Silvestri, Lorenzo Antonelli, Filippo Betello, Gabriele Tolomei  

**Link**: [PDF](https://arxiv.org/pdf/2603.06485)  

**Abstract**: Explainable Artificial Intelligence (XAI) seeks to enhance the transparency and accountability of machine learning systems, yet most methods follow a one-size-fits-all paradigm that neglects user differences in expertise, goals, and cognitive needs. Although Large Language Models can translate technical explanations into natural language, they introduce challenges related to faithfulness and hallucinations. To address these challenges, we present PONTE (Personalized Orchestration for Natural language Trustworthy Explanations), a human-in-the-loop framework for adaptive and reliable XAI narratives. PONTE models personalization as a closed-loop validation and adaptation process rather than prompt engineering. It combines: (i) a low-dimensional preference model capturing stylistic requirements; (ii) a preference-conditioned generator grounded in structured XAI artifacts; and (iii) verification modules enforcing numerical faithfulness, informational completeness, and stylistic alignment, optionally supported by retrieval-grounded argumentation. User feedback iteratively updates the preference state, enabling quick personalization. Automatic and human evaluations across healthcare and finance domains show that the verification-refinement loop substantially improves completeness and stylistic alignment over validation-free generation. Human studies further confirm strong agreement between intended preference vectors and perceived style, robustness to generation stochasticity, and consistently positive quality assessments. 

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
# Evaluation of Deontic Conditional Reasoning in Large Language Models: The Case of Wason's Selection Task 

**Authors**: Hirohiko Abe, Kentaro Ozeki, Risako Ando, Takanobu Morishita, Koji Mineshima, Mitsuhiro Okada  

**Link**: [PDF](https://arxiv.org/pdf/2603.06416)  

**Abstract**: As large language models (LLMs) advance in linguistic competence, their reasoning abilities are gaining increasing attention. In humans, reasoning often performs well in domain specific settings, particularly in normative rather than purely formal contexts. Although prior studies have compared LLM and human reasoning, the domain specificity of LLM reasoning remains underexplored. In this study, we introduce a new Wason Selection Task dataset that explicitly encodes deontic modality to systematically distinguish deontic from descriptive conditionals, and use it to examine LLMs' conditional reasoning under deontic rules. We further analyze whether observed error patterns are better explained by confirmation bias (a tendency to seek rule-supporting evidence) or by matching bias (a tendency to ignore negation and select items that lexically match elements of the rule). Results show that, like humans, LLMs reason better with deontic rules and display matching-bias-like errors. Together, these findings suggest that the performance of LLMs varies systematically across rule types and that their error patterns can parallel well-known human biases in this paradigm. 

---
# Transparent AI for Mathematics: Transformer-Based Large Language Models for Mathematical Entity Relationship Extraction with XAI 

**Authors**: Tanjim Taharat Aurpa  

**Link**: [PDF](https://arxiv.org/pdf/2603.06348)  

**Abstract**: Mathematical text understanding is a challenging task due to the presence of specialized entities and complex relationships between them. This study formulates mathematical problem interpretation as a Mathematical Entity Relation Extraction (MERE) task, where operands are treated as entities and operators as their relationships. Transformer-based models are applied to automatically extract these relations from mathematical text, with Bidirectional Encoder Representations from Transformers (BERT) achieving the best performance, reaching an accuracy of 99.39%. To enhance transparency and trust in the model's predictions, Explainable Artificial Intelligence (XAI) is incorporated using Shapley Additive Explanations (SHAP). The explainability analysis reveals how specific textual and mathematical features influence relation prediction, providing insights into feature importance and model behavior. By combining transformer-based learning, a task-specific dataset, and explainable modeling, this work offers an effective and interpretable framework for MERE, supporting future applications in automated problem solving, knowledge graph construction, and intelligent educational systems. 

---
# The Art That Poses Back: Assessing AI Pastiches after Contemporary Artworks 

**Authors**: Anca Dinu, Andreiana Mihail, Andra-Maria Florescu, Claudiu Creanga  

**Link**: [PDF](https://arxiv.org/pdf/2603.06324)  

**Abstract**: This study explores artificial visual creativity, focusing on ChatGPT's ability to generate new images intentionally pastiching original artworks such as paintings, drawings, sculptures and installations. The process involved twelve artists from Romania, Bulgaria, France, Austria, and the United Kingdom, each invited to contribute with three of their artworks and to grade and comment on the AI-generated versions. The analysis combines human evaluation with computational methods aimed at detecting visual and stylistic similarities or divergences between the original works and their AI-produced renditions. The results point to a significant gap between color and texture-based similarity and compositional, conceptual, and perceptual one. Consequently, we advocate for the use of a "style transfer dashboard" of complementary metrics to evaluate the similarity between pastiches and originals, rather than using a single style metric. The artists' comments revealed limitations of ChatGPT's pastiches after contemporary artworks, which were perceived by the authors of the originals as lacking dimensionality, context, and intentional sense, and seeming more of a paraphrase or an approximate quotation rather than as a valuable, emotion-evoking artwork. 

---
# Mind the Gap: Pitfalls of LLM Alignment with Asian Public Opinion 

**Authors**: Hari Shankar, Vedanta S P, Sriharini Margapuri, Debjani Mazumder, Ponnurangam Kumaraguru, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.06264)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in multilingual, multicultural settings, yet their reliance on predominantly English-centric training data risks misalignment with the diverse cultural values of different societies. In this paper, we present a comprehensive, multilingual audit of the cultural alignment of contemporary LLMs including GPT-4o-Mini, Gemini-2.5-Flash, Llama 3.2, Mistral and Gemma 3 across India, East Asia and Southeast Asia. Our study specifically focuses on the sensitive domain of religion as the prism for broader alignment. To facilitate this, we conduct a multi-faceted analysis of every LLM's internal representations, using log-probs/logits, to compare the model's opinion distributions against ground-truth public attitudes. We find that while the popular models generally align with public opinion on broad social issues, they consistently fail to accurately represent religious viewpoints, especially those of minority groups, often amplifying negative stereotypes. Lightweight interventions, such as demographic priming and native language prompting, partially mitigate but do not eliminate these cultural gaps. We further show that downstream evaluations on bias benchmarks (such as CrowS-Pairs, IndiBias, ThaiCLI, KoBBQ) reveal persistent harms and under-representation in sensitive contexts. Our findings underscore the urgent need for systematic, regionally grounded audits to ensure equitable global deployment of LLMs. 

---
# SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models 

**Authors**: Yunlong Chu, Minglai Shao, Yuhang Liu, Bing Hao, Yumeng Lin, Jialu Wang, Ruijie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06222)  

**Abstract**: Explicit Chain-of-Thought improves the reasoning performance of large language models but often incurs high inference cost due to verbose token-level traces. While recent approaches reduce this overhead via concise prompting or step pruning, they largely truncate what the model says rather than internalize what the model thinks. Latent reasoning offers a promising alternative by performing computation in the hidden space, yet prior methods face two critical challenges. Many existing approaches rely on rigid point-to-point alignment, forcing a latent token to approximate the final representation of a reasoning step, which can be insufficient to capture the dense, variable-length semantics of an entire reasoning segment. Furthermore, these methods often suffer from a lack of interpretability: latent states are commonly produced by unconstrained optimization or embedding mixing, yielding vectors that are difficult to decode or audit under the pretrained language head. We propose SPOT, a flexible framework that compresses explicit CoT into compact latent pause tokens without enforcing a fixed response template. At the core of SPOT is Span-level Semantic Alignment, a Sinkhorn optimal-transport objective that softly matches each pause token to the semantics of an entire reasoning segment, overcoming the rigidity of step-end alignment. To further improve interpretability, SPOT introduces a Frozen-Head Decoding Constraint that keeps latent states directly decodable as token distributions under the frozen pretrained LM head, enabling readable keyword interpretations of latent thoughts. Experiments on reasoning benchmarks demonstrate that SPOT improves accuracy by 2.3 points on average while reducing generated tokens by 37.5% and provides faithful semantic interpretations of the latent reasoning process. 

---
# FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling 

**Authors**: Qihang Fan, Huaibo Huang, Zhiying Wu, Juqiu Wang, Bingning Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2603.06199)  

**Abstract**: Long-context modeling is a pivotal capability for Large Language Models, yet the quadratic complexity of attention remains a critical bottleneck, particularly during the compute-intensive prefilling phase. While various sparse attention mechanisms have been explored, they typically suffer from either significant search latency or insufficient sparsity. In this paper, we propose FlashPrefill, a framework enabling ultra-fast prefilling via instantaneous pattern discovery and thresholding. FlashPrefill leverages a fast block-searching technique to simultaneously locate dynamic vertical, slash, and block-sparse attention patterns. Crucially, it introduces a dynamic thresholding mechanism that bypasses the prohibitive overhead of sorting or accumulating attention scores while effectively eliminating the long-tail distribution to enhance sparsity. Extensive evaluations demonstrate that FlashPrefill achieves a substantial leap in efficiency, delivering an unprecedented 27.78x speedup on 256K sequences. Notably, unlike existing methods that incur efficiency degradation on shorter contexts, FlashPrefill maintains a 1.71x speedup even at a 4K context length, demonstrating its robustness and practical utility across varying sequence scales. 

---
# LIT-RAGBench: Benchmarking Generator Capabilities of Large Language Models in Retrieval-Augmented Generation 

**Authors**: Koki Itai, Shunichi Hasegawa, Yuta Yamamoto, Gouki Minegishi, Masaki Otsuki  

**Link**: [PDF](https://arxiv.org/pdf/2603.06198)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a framework in which a Generator, such as a Large Language Model (LLM), produces answers by retrieving documents from an external collection using a Retriever. In practice, Generators must integrate evidence from long contexts, perform multi-step reasoning, interpret tables, and abstain when evidence is missing. However, existing benchmarks for Generators provide limited coverage, with none enabling simultaneous evaluation of multiple capabilities under unified conditions. To bridge the gap between existing evaluations and practical use, we introduce LIT-RAGBench (the Logic, Integration, Table, Reasoning, and Abstention RAG Generator Benchmark), which defines five categories: Integration, Reasoning, Logic, Table, and Abstention, each further divided into practical evaluation aspects. LIT-RAGBench systematically covers patterns combining multiple aspects across categories. By using fictional entities and scenarios, LIT-RAGBench evaluates answers grounded in the provided external documents. The dataset consists of 114 human-constructed Japanese questions and an English version generated by machine translation with human curation. We use LLM-as-a-Judge for scoring and report category-wise and overall accuracy. Across API-based and open-weight models, no model exceeds 90% overall accuracy. By making strengths and weaknesses measurable within each category, LIT-RAGBench serves as a valuable metric for model selection in practical RAG deployments and for building RAG-specialized models. We release LIT-RAGBench, including the dataset and evaluation code, at this https URL. 

---
# Wisdom of the AI Crowd (AI-CROWD) for Ground Truth Approximation in Content Analysis: A Research Protocol & Validation Using Eleven Large Language Models 

**Authors**: Luis de-Marcos, Manuel Goyanes, Adrián Domínguez-Díaz  

**Link**: [PDF](https://arxiv.org/pdf/2603.06197)  

**Abstract**: Large-scale content analysis is increasingly limited by the absence of observable ground truth or gold-standard labels, as creating such benchmarks through extensive human coding becomes impractical for massive datasets due to high time, cost, and consistency challenges. To overcome this barrier, we introduce the AI-CROWD protocol, which approximates ground truth by leveraging the collective outputs of an ensemble of large language models (LLMs). Rather than asserting that the resulting labels are true ground truth, the protocol generates a consensus-based approximation derived from convergent and divergent inferences across multiple models. By aggregating outputs via majority voting and interrogating agreement/disagreement patterns with diagnostic metrics, AI-CROWD identifies high-confidence classifications while flagging potential ambiguity or model-specific biases. 

---
# MAPO: Mixed Advantage Policy Optimization for Long-Horizon Multi-Turn Dialogue 

**Authors**: Naifan Zhang, Ruihan Sun, Jinwei Su, Hengjie Yang, Zhengyuan Pan, Zhaohan Chen, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06194)  

**Abstract**: Subjective multi-turn dialogue tasks, such as emotional support, require conversational policies that adapt to evolving user states and optimize long-horizon interaction quality. However, reinforcement learning (RL) for such settings remains challenging due to the absence of reliable process supervision. Outcome-only training collapses credit assignment across turns into a single trajectory-level reward, while naïve turn-level group sampling incurs prohibitive rollout costs in interactive environments. We propose a critic-free and efficient RL algorithm named MAPO that leverages dense process feedback from a judge model and propagates long-horizon effects through Monte Carlo returns. To stabilize optimization, we introduce a mixed advantage estimator that combines turn-level normalization with batch-level normalization, enabling fine-grained yet scalable credit assignment. Across multiple subjective dialogue benchmarks, including EMPA, EmoBench, and EQ-Bench, and model scales ranging from 7B to 32B, our method consistently improves both training stability and final performance over outcome-only GRPO and single-level normalization baselines. On EMPA, we improve rates by up to 9 points and increase dialogue scores by as much as +43.2 over the 7B base model. Despite training only on EMPA-style environments, our approach generalizes well, yielding consistent improvements on unseen emotional-intelligence benchmarks, including up to +4 points on EmoBench and +3.5 on EQ-Bench. Together, these results demonstrate that dense process supervision combined with mixed-level normalization enables effective and scalable RL for subjective, open-ended multi-turn dialogue. 

---
# CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation 

**Authors**: Mohammed Baharoon, Thibault Heintz, Siavash Raissi, Mahmoud Alabbad, Mona Alhammad, Hassan AlOmaish, Sung Eun Kim, Oishi Banerjee, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.06183)  

**Abstract**: We introduce CRIMSON, a clinically grounded evaluation framework for chest X-ray report generation that assesses reports based on diagnostic correctness, contextual relevance, and patient safety. Unlike prior metrics, CRIMSON incorporates full clinical context, including patient age, indication, and guideline-based decision rules, and prevents normal or clinically insignificant findings from exerting disproportionate influence on the overall score. The framework categorizes errors into a comprehensive taxonomy covering false findings, missing findings, and eight attribute-level errors (e.g., location, severity, measurement, and diagnostic overinterpretation). Each finding is assigned a clinical significance level (urgent, actionable non-urgent, non-actionable, or expected/benign), based on a guideline developed in collaboration with attending cardiothoracic radiologists, enabling severity-aware weighting that prioritizes clinically consequential mistakes over benign discrepancies. CRIMSON is validated through strong alignment with clinically significant error counts annotated by six board-certified radiologists in ReXVal (Kendalls tau = 0.61-0.71; Pearsons r = 0.71-0.84), and through two additional benchmarks that we introduce. In RadJudge, a targeted suite of clinically challenging pass-fail scenarios, CRIMSON shows consistent agreement with expert judgment. In RadPref, a larger radiologist preference benchmark of over 100 pairwise cases with structured error categorization, severity modeling, and 1-5 overall quality ratings from three cardiothoracic radiologists, CRIMSON achieves the strongest alignment with radiologist preferences. We release the metric, the evaluation benchmarks, RadJudge and RadPref, and a fine-tuned MedGemma model to enable reproducible evaluation of report generation, all available at this https URL. 

---
# A Causal Graph Approach to Oppositional Narrative Analysis 

**Authors**: Diego Revilla, Martin Fernandez-de-Retana, Lingfeng Chen, Aritz Bilbao-Jayo, Miguel Fernandez-de-Retana  

**Link**: [PDF](https://arxiv.org/pdf/2603.06135)  

**Abstract**: Current methods for textual analysis rely on data annotated within predefined ontologies, often embedding human bias within black-box models. Despite achieving near-perfect performance, these approaches exploit unstructured, linear pattern recognition rather than modeling the structured interactions between entities that naturally emerge in discourse. In this work, we propose a graph-based framework for the detection, analysis, and classification of oppositional narratives and their underlying entities by representing narratives as entity-interaction graphs. Moreover, by incorporating causal estimation at the node level, our approach derives a causal representation of each contribution to the final classification by distilling the constructed sentence graph into a minimal causal subgraph. Building upon this representation, we introduce a classification pipeline that outperforms existing approaches to oppositional thinking classification task. 

---
# Diffusion Language Models Are Natively Length-Aware 

**Authors**: Vittorio Rossi, Giacomo Cirò, Davide Beltrame, Luca Gandolfi, Paul Röttger, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2603.06123)  

**Abstract**: Unlike autoregressive language models, which terminate variable-length generation upon predicting an End-of-Sequence (EoS) token, Diffusion Language Models (DLMs) operate over a fixed maximum-length context window for a predetermined number of denoising steps. However, this process is independent of the required response length, resulting in computational waste for the majority of short responses common in reasoning and chat tasks. To address this problem, we conjecture that the latent prompt representation contains sufficient information to estimate the required output length. We provide empirical evidence for this phenomenon and propose a zero-shot mechanism to dynamically crop the context window before generation begins, leading to fewer diffusion steps and substantial computational savings. We evaluate our approach on four benchmarks with diverse tasks -- GSM8K (reasoning), HumanEval (code generation), IfEval (instruction following), and LongFormQA (question answering) -- revealing massive efficiency gains at minimal performance impact. We report significant reductions in FLOPs across all tasks, with no statistically significant performance degradation, and significant performance improvements in 2 out of 4 tasks. 

---
# Making Implicit Premises Explicit in Logical Understanding of Enthymemes 

**Authors**: Xuyao Feng, Anthony Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2603.06114)  

**Abstract**: Real-world arguments in text and dialogues are normally enthymemes (i.e. some of their premises and/or claims are implicit). Natural language processing (NLP) methods for handling enthymemes can potentially identify enthymemes in text but they do not decode their underlying logic, whereas logic-based approaches for handling them assume a knowledgebase with sufficient formulae that can be used to decode them via abduction. There is therefore a lack of a systematic method for translating textual components of an enthymeme into a logical argument and generating the logical formulae required for their decoding, and thereby showing logical entailment. To address this, we propose a pipeline that integrates: (1) a large language model (LLM) to generate intermediate implicit premises based on the explicit premise and claim; (2) another LLM to translate the natural language into logical formulas; and (3) a neuro-symbolic reasoner based on a SAT solver to determine entailment. We evaluate our pipeline on two enthymeme datasets, demonstrating promising performance in selecting the correct implicit premise, as measured by precision, recall, F1-score, and accuracy. 

---
# Experiences Build Characters: The Linguistic Origins and Functional Impact of LLM Personality 

**Authors**: Xi Wang, Mengdie Zhuang, Jiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06088)  

**Abstract**: Human problem-solving is enriched by a diversity of styles and personality traits, yet the development of Large Language Models (LLMs) has largely prioritized uniform performance benchmarks that favour specific behavioural tendencies such as assertiveness. To investigate how diverse experiences shape machine personality and influence problem-solving, this study employs continued pre-training to expose models to domain-specific texts in an unsupervised manner, simulating the accumulation of experience. By adapting the Big Five framework via the Machine Personality Inventory (MPI), we quantify the personality traits of these model variants and analyse their relationship to linguistic style and reasoning behaviour. The findings reveal that model competence is bimodal, peaking at "Expressive Generalists" and "Suppressed Specialists," while identifying a "Suppression Advantage" where reduced social traits enhance complex reasoning performance. This study further establishes a causal link between training data linguistics, such as imperative frequency, and lexical diversity, providing a roadmap for "Personality Engineering". 

---
# Evaluating Austrian A-Level German Essays with Large Language Models for Automated Essay Scoring 

**Authors**: Jonas Kubesch, Lena Huber, Clemens Havas  

**Link**: [PDF](https://arxiv.org/pdf/2603.06066)  

**Abstract**: Automated Essay Scoring (AES) has been explored for decades with the goal to support teachers by reducing grading workload and mitigating subjective biases. While early systems relied on handcrafted features and statistical models, recent advances in Large Language Models (LLMs) have made it possible to evaluate student writing with unprecedented flexibility. This paper investigates the application of state-of-the-art open-weight LLMs for the grading of Austrian A-level German texts, with a particular focus on rubric-based evaluation. A dataset of 101 anonymised student exams across three text types was processed and evaluated. Four LLMs, DeepSeek-R1 32b, Qwen3 30b, Mixtral 8x7b and LLama3.3 70b, were evaluated with different contexts and prompting strategies. The LLMs were able to reach a maximum of 40.6% agreement with the human rater in the rubric-provided sub-dimensions, and only 32.8% of final grades matched the ones given by a human expert. The results indicate that even though smaller models are able to use standardised rubrics for German essay grading, they are not accurate enough to be used in a real-world grading environment. 

---
# ViewFusion: Structured Spatial Thinking Chains for Multi-View Reasoning 

**Authors**: Xingjian Tao, Yiwei Wang, Yujun Cai, Yifan Song, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06024)  

**Abstract**: Multi-view spatial reasoning remains difficult for current vision-language models. Even when multiple viewpoints are available, models often underutilize cross-view relations and instead rely on single-image shortcuts, leading to fragile performance on viewpoint transformation and occlusion-sensitive cases. We present ViewFusion, a two-stage framework that explicitly separates cross-view spatial pre-alignment from question answering. In the first stage, the model performs deliberate spatial pre-thinking to infer viewpoint relations and spatial transformations across views, forming an intermediate workspace that goes beyond a simple re-description. In the second stage, the model conducts question-driven reasoning conditioned on this workspace to produce the final prediction. We train ViewFusion with synthetic reasoning supervision followed by reinforcement learning using GRPO, which improves answer correctness while stabilizing the intended two-stage generation behavior. On MMSI-Bench, ViewFusion improves accuracy by 5.3\% over Qwen3-VL-4B-Instruct, with the largest gains on examples that require genuine cross-view alignment. 

---
# MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing 

**Authors**: Yang Liu, Jinxuan Cai, Yishen Li, Qi Meng, Zedi Liu, Xin Li, Chen Qian, Chuan Shi, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06007)  

**Abstract**: Large language model-based (LLM-based) multi-agent systems (MAS) are increasingly used to extend agentic problem solving via role specialization and collaboration. MAS workflows can be naturally modeled as directed computation graphs, where nodes execute agents/sub-workflows and edges encode dependencies and message passing. However, implementing complex graph workflows in current frameworks still requires substantial manual effort, offers limited reuse, and makes it difficult to integrate heterogeneous external context sources. To overcome these limitations, we present MASFactory, a graph-centric framework for orchestrating LLM-based MAS. It introduces Vibe Graphing, a human-in-the-loop approach that compiles natural-language intent into an editable workflow specification and then into an executable graph. In addition, the framework provides reusable components and pluggable context integration, as well as a visualizer for topology preview, runtime tracing, and human-in-the-loop interaction. We evaluate MASFactory on seven public benchmarks, validating both reproduction consistency for representative MAS methods and the effectiveness of Vibe Graphing. Our code (this https URL) and video (this https URL) are publicly available. 

---
# Track-SQL: Enhancing Generative Language Models with Dual-Extractive Modules for Schema and Context Tracking in Multi-turn Text-to-SQL 

**Authors**: Bingfeng Chen, Shaobin Shi, Yongqi Luo, Boyan Xu, Ruichu Cai, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2603.05996)  

**Abstract**: Generative language models have shown significant potential in single-turn Text-to-SQL. However, their performance does not extend equivalently to multi-turn Text-to-SQL. This is primarily due to generative language models' inadequacy in handling the complexities of context information and dynamic schema linking in multi-turn interactions. In this paper, we propose a framework named Track-SQL, which enhances generative language models with dual-extractive modules designed to track schema and contextual changes in multi-turn Text-to-SQL. Specifically, Track-SQL incorporates a \emph{Semantic-enhanced Schema Extractor} and a \emph{Schema-aware Context Extractor}. Experimental results demonstrate that Track-SQL achieves state-of-the-art performance on the SparC and CoSQL datasets. Furthermore, detailed ablation studies reveal that Track-SQL significantly improves execution accuracy in multi-turn interactions by 7.1\% and 9.55\% on these datasets, respectively. Our implementation will be open-sourced at this https URL. 

---
# Who We Are, Where We Are: Mental Health at the Intersection of Person, Situation, and Large Language Models 

**Authors**: Nikita Soni, August Håkan Nilsson, Syeda Mahwish, Vasudha Varadarajan, H. Andrew Schwartz, Ryan L. Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2603.05953)  

**Abstract**: Mental health is not a fixed trait but a dynamic process shaped by the interplay between individual dispositions and situational contexts. Building on interactionist and constructionist psychological theories, we develop interpretable models to predict well-being and identify adaptive and maladaptive self-states in longitudinal social media data. Our approach integrates person-level psychological traits (e.g., resilience, cognitive distortions, implicit motives) with language-inferred situational features derived from the Situational 8 DIAMONDS framework. We compare these theory-grounded features to embeddings from a psychometrically-informed language model that captures temporal and individual-specific patterns. Results show that our principled, theory-driven features provide competitive performance while offering greater interpretability. Qualitative analyses further highlight the psychological coherence of features most predictive of well-being. These findings underscore the value of integrating computational modeling with psychological theory to assess dynamic mental states in contextually sensitive and human-understandable ways. 

---
# Implicit Style Conditioning: A Structured Style-Rewrite Framework for Low-Resource Character Modeling 

**Authors**: Chanhui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.05933)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing (RP); however, small Language Models (SLMs) with highly stylized personas remains a challenge due to data scarcity and the complexity of style disentanglement. Standard Supervised Fine-Tuning (SFT) often captures surface-level semantics while failing to reproduce the intricate syntactic and pragmatic nuances of a character, leading to "Out-Of-Character" (OOC) generation. To address this, we propose a Structured Style-Rewrite Framework that explicitly disentangles style into three interpretable dimensions: lexical signatures (via PMI), syntactic patterns (grounded in PCFG rules), and pragmatic style. Furthermore, we introduce an implicit style conditioning strategy via Chain-of-Thought (CoT) distillation. By leveraging explicit reasoning traces during training as a strong inductive bias, our approach aligns the model's latent representations with structured style features, enabling high-fidelity stylized generation without requiring explicit reasoning tokens during inference. Extensive experiments on a specific high-stylization domain (anime characters) demonstrate that our method enables a Qwen-1.7B model to outperform significantly larger baselines (e.g., 4B Vanilla SFT) in style consistency and semantic fidelity. Our approach offers a data-efficient paradigm for democratizing inference and deployment on consumer hardware. 

---
# Addressing the Ecological Fallacy in Larger LMs with Human Context 

**Authors**: Nikita Soni, Dhruv Vijay Kunjadiya, Pratham Piyush Shah, Dikshya Mohanty, H. Andrew Schwartz, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2603.05928)  

**Abstract**: Language model training and inference ignore a fundamental linguistic fact -- there is a dependence between multiple sequences of text written by the same person. Prior work has shown that addressing this form of \textit{ecological fallacy} can greatly improve the performance of multiple smaller (~124M) GPT-based models. In this work, we ask if addressing the ecological fallacy by modeling the author's language context with a specific LM task (called HuLM) can provide similar benefits for a larger-scale model, an 8B Llama model. To this end, we explore variants that process an author's language in the context of their other temporally ordered texts. We study the effect of pre-training with this author context using the HuLM objective, as well as using it during fine-tuning with author context (\textit{HuFT:Human-aware Fine-Tuning}). Empirical comparisons show that addressing the ecological fallacy during fine-tuning alone using QLoRA improves the performance of the larger 8B model over standard fine-tuning. Additionally, QLoRA-based continued HuLM pre-training results in a human-aware model generalizable for improved performance over eight downstream tasks with linear task classifier training alone. These results indicate the utility and importance of modeling language in the context of its original generators, the authors. 

---
# Learning Next Action Predictors from Human-Computer Interaction 

**Authors**: Omar Shaikh, Valentin Teutschbein, Kanishk Gandhi, Yikun Chi, Nick Haber, Thomas Robinson, Nilam Ram, Byron Reeves, Sherry Yang, Michael S. Bernstein, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05923)  

**Abstract**: Truly proactive AI systems must anticipate what we will do next. This foresight demands far richer information than the sparse signals we type into our prompts -- it demands reasoning over the entire context of what we see and do. We formalize this as next action prediction (NAP): given a sequence of a user's multimodal interactions with a computer (screenshots, clicks, sensor data), predict that user's next action. Progress on this task requires both new data and modeling approaches. To scale data, we annotate longitudinal, naturalistic computer use with vision-language models. We release an open-source pipeline for performing this labeling on private infrastructure, and label over 360K actions across one month of continuous phone usage from 20 users, amounting to 1,800 hours of screen time. We then introduce LongNAP, a user model that combines parametric and in-context learning to reason over long interaction histories. LongNAP is trained via policy gradient methods to generate user-specific reasoning traces given some context; retrieve relevant traces from a library of past traces; and then apply retrieved traces in-context to predict future actions. Using an LLM-as-judge evaluation metric (0-1 similarity to ground truth), LongNAP significantly outperforms supervised finetuning and prompted baselines on held-out data (by 79% and 39% respectively). Additionally, LongNAP generalizes to held out users when trained across individuals. The space of next actions a user might take at any moment is unbounded, spanning thousands of possible outcomes. Despite this, 17.1% of LongNAP's predicted trajectories are well-aligned with what a user does next (LLM-judge score $\geq$ 0.5). This rises to 26% when we filter to highly confident predictions. In sum, we argue that learning from the full context of user behavior to anticipate user needs is now a viable task with substantial opportunity. 

---
# InfoGatherer: Principled Information Seeking via Evidence Retrieval and Strategic Questioning 

**Authors**: Maksym Taranukhin, Shuyue Stella Li, Evangelos Milios, Geoff Pleiss, Yulia Tsvetkov, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2603.05909)  

**Abstract**: LLMs are increasingly deployed in high-stakes domains such as medical triage and legal assistance, often as document-grounded QA systems in which a user provides a description, relevant sources are retrieved, and an LLM generates a prediction. In practice, initial user queries are often underspecified, and a single retrieval pass is insufficient for reliable decision-making, leading to incorrect and overly confident answers. While follow-up questioning can elicit missing information, existing methods typically depend on implicit, unstructured confidence signals from the LLM, making it difficult to determine what remains unknown, what information matters most, and when to stop asking questions. We propose InfoGatherer, a framework that gathers missing information from two complementary sources: retrieved domain documents and targeted follow-up questions to the user. InfoGatherer models uncertainty using Dempster-Shafer belief assignments over a structured evidential network, enabling principled fusion of incomplete and potentially contradictory evidence from both sources without prematurely collapsing to a definitive answer. Across legal and medical tasks, InfoGatherer outperforms strong baselines while requiring fewer turns. By grounding uncertainty in formal evidential theory rather than heuristic LLM signals, InfoGatherer moves towards trustworthy, interpretable decision support in domains where reliability is critical. 

---
# Building an Ensemble LLM Semantic Tagger for UN Security Council Resolutions 

**Authors**: Hussein Ghaly  

**Link**: [PDF](https://arxiv.org/pdf/2603.05895)  

**Abstract**: This paper introduces a new methodology for using LLM-based systems for accurate and efficient semantic tagging of UN Security Council resolutions. The main goal is to leverage LLM performance variability to build ensemble systems for data cleaning and semantic tagging tasks. We introduce two evaluation metrics: Content Preservation Ratio (CPR) and Tag Well-Formedness (TWF), in order to avoid hallucinations and unnecessary additions or omissions to the input text beyond the task requirement. These metrics allow the selection of the best output from multiple runs of several GPT models. GPT-4.1 achieved the highest metrics for both tasks (Cleaning: CPR 84.9% - Semantic Tagging: CPR 99.99% and TWF 99.92%). In terms of cost, smaller models, such as GPT-4.1-mini, achieved comparable performance to the best model in each task at only 20% of the cost. These metrics ultimately allowed the ensemble to select the optimal output (both cleaned and tagged content) for all the LLM models involved, across multiple runs. With this ensemble design and the use of metrics, we create a reliable LLM system for performing semantic tagging on challenging texts. 

---
# Lost in Stories: Consistency Bugs in Long Story Generation by LLMs 

**Authors**: Junjie Li, Xinrui Guo, Yuhao Wu, Roy Ka-Wei Lee, Hongzhi Li, Yutao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.05890)  

**Abstract**: What happens when a storyteller forgets its own story? Large Language Models (LLMs) can now generate narratives spanning tens of thousands of words, but they often fail to maintain consistency throughout. When generating long-form narratives, these models can contradict their own established facts, character traits, and world rules. Existing story generation benchmarks focus mainly on plot quality and fluency, leaving consistency errors largely unexplored. To address this gap, we present ConStory-Bench, a benchmark designed to evaluate narrative consistency in long-form story generation. It contains 2,000 prompts across four task scenarios and defines a taxonomy of five error categories with 19 fine-grained subtypes. We also develop ConStory-Checker, an automated pipeline that detects contradictions and grounds each judgment in explicit textual evidence. Evaluating a range of LLMs through five research questions, we find that consistency errors show clear tendencies: they are most common in factual and temporal dimensions, tend to appear around the middle of narratives, occur in text segments with higher token-level entropy, and certain error types tend to co-occur. These findings can inform future efforts to improve consistency in long-form narrative generation. Our project page is available at this https URL. 

---
# VerChol -- Grammar-First Tokenization for Agglutinative Languages 

**Authors**: Prabhu Raja  

**Link**: [PDF](https://arxiv.org/pdf/2603.05883)  

**Abstract**: Tokenization is the foundational step in all large language model (LLM) pipelines, yet the dominant approach Byte Pair Encoding (BPE) and its variants is inherently script agnostic and optimized for English like morphology. For agglutinative languages a typological class encompassing the Dravidian family (Tamil, Kannada, Telugu, Malayalam), Turkic languages (Turkish, Azerbaijani, Uzbek), Uralic languages (Finnish, Hungarian, Estonian), Korean, Japanese, Swahili, Basque, and others, a single word may encode root, tense, aspect, person, number, gender agreement, case, and postpositions into one orthographic unit. Statistical tokenizers fragment these words into byte pair chunks that sever morpheme boundaries and inflate token counts. 

---
# Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation 

**Authors**: Changcheng Li, Jiancan Wu, Hengheng Zhang, Zhengsu Chen, Guo An, Junxiang Qiu, Xiang Wang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.05881)  

**Abstract**: Reliable deployment of large language models (LLMs) requires accurate uncertainty estimation. Existing methods are predominantly answer-first, producing confidence only after generating an answer, which measure the correctness of a specific response and limits practical usability. We study a confidence-first paradigm, where the model outputs its confidence before answering, interpreting this score as the model's probability of answering the question correctly under its current policy.
We propose CoCA(Co-optimized Confidence and Answers), a GRPO reinforcement learning framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. By assigning separate rewards and group-relative advantages to confidence and answer segments, CoCA enables stable joint optimization and avoids reward hacking. Experiments across math, code, and factual QA benchmarks show improved calibration and uncertainty discrimination while preserving answer quality, thereby enabling a broader range of downstream applications. 

---
# ROSE: Reordered SparseGPT for More Accurate One-Shot Large Language Models Pruning 

**Authors**: Mingluo Su, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05878)  

**Abstract**: Pruning is widely recognized as an effective method for reducing the parameters of large language models (LLMs), potentially leading to more efficient deployment and inference. One classic and prominent path of LLM one-shot pruning is to leverage second-order gradients (i.e., Hessian), represented by the pioneering work SparseGPT. However, the predefined left-to-right pruning order in SparseGPT leads to suboptimal performance when the weights exhibit columnar patterns. This paper studies the effect of pruning order under the SparseGPT framework. The analyses lead us to propose ROSE, a reordered SparseGPT method that prioritizes weights with larger potential pruning errors to be pruned earlier. ROSE first performs pre-pruning to identify candidate weights for removal, and estimates both column and block pruning loss. Subsequently, two-level reordering is performed: columns within each block are reordered in descending order of column loss, while blocks are reordered based on block loss. We introduce the relative range of block loss as a metric to identify columnar layers, enabling adaptive reordering across the entire model. Substantial empirical results on prevalent LLMs (LLaMA2-7B/13B/70B, LLaMA3-8B, Mistral-7B) demonstrate that ROSE surpasses the original SparseGPT and other counterpart pruning methods. Our code is available at this https URL. 

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
# RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning 

**Authors**: Yuhang Liu, Ruijie Wang, Yunlong Chu, Bing Hao, Yumeng Lin, Shengzhong Liu, Minglai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2603.05818)  

**Abstract**: Large Language Models (LLMs) excel at multi-step reasoning, yet increasing the structural complexity of inference does not consistently improve system-level returns. Methods such as Tree of Thoughts (ToT), Graph of Thoughts (GoT), and Adaptive Graph of Thoughts (AGoT) can boost accuracy on some benchmarks, but often introduce substantial overhead in token consumption and latency, and their gains can be unstable across task distributions-sometimes underperforming simpler Chain-of-Thought (CoT) or direct input-output prompting (IO). We attribute this inefficiency to stage-wise and node-wise heterogeneity inside GoT-style reasoning pipelines: high-quality planning and final synthesis are globally coupled and typically benefit from strong models, whereas many intermediate subtasks are localized and can be solved accurately by lighter models with far fewer tokens. Motivated by these observations, we propose RouteGoT, a budget-controllable, node-adaptive routing framework for graph-structured reasoning. RouteGoT performs in-graph routing by prioritizing strong models for planning and synthesis, while dynamically allocating lightweight models and cost-effective strategies to leaf subtasks based on predicted difficulty. It further integrates explicit budget constraints into a global inference scheduler to control graph expansion under a user-specified token budget, enabling predictable performance-cost trade-offs. Experiments across reasoning, retrieval, and multi-hop QA benchmarks show that RouteGoT matching or improving accuracy while substantially reducing token usage; specifically, it achieves an average 8.1 percentage points accuracy improvement and 79.1\% output token reduction compared to AGoT. Furthermore, RouteGoT outperforms existing routing baselines by maintaining a superior cost-accuracy trade-off, demonstrating improved robustness under varying budget targets and tasks. 

---
# Tutor Move Taxonomy: A Theory-Aligned Framework for Analyzing Instructional Moves in Tutoring 

**Authors**: Zhuqian Zhou, Kirk Vanacore, Tamisha Thompson, Jennifer St John, Rene Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2603.05778)  

**Abstract**: Understanding what makes tutoring effective requires methods for systematically analyzing tutors' instructional actions during learning interactions. This paper presents a tutor move taxonomy designed to support large-scale analysis of tutoring dialogue within the National Tutoring Observatory. The taxonomy provides a structured annotation framework for labeling tutors' instructional moves during one-on-one tutoring sessions. We developed the taxonomy through a hybrid deductive-inductive process. First, we synthesized research from cognitive science, the learning sciences, classroom discourse analysis, and intelligent tutoring systems to construct a preliminary framework of tutoring moves. We then refined the taxonomy through iterative coding of authentic tutoring transcripts conducted by expert annotators with extensive instructional and qualitative research experience. The resulting taxonomy organizes tutoring behaviors into four categories: tutoring support, learning support, social-emotional and motivational support, and logistical support. Learning support moves are further organized along a spectrum of student engagement, distinguishing between moves that elicit student reasoning and those that provide direct explanation or answers. By defining tutoring dialogue in terms of discrete instructional actions, the taxonomy enables scalable annotation using AI, computational modeling of tutoring strategies, and empirical analysis of how tutoring behaviors relate to learning outcomes. 

---
# PVminerLLM: Structured Extraction of Patient Voice from Patient-Generated Text using Large Language Models 

**Authors**: Samah Fodeh, Linhai Ma, Ganesh Puthiaraju, Srivani Talakokkul, Afshan Khan, Ashley Hagaman, Sarah Lowe, Aimee Roundtree  

**Link**: [PDF](https://arxiv.org/pdf/2603.05776)  

**Abstract**: Motivation: Patient-generated text contains critical information about patients' lived experiences, social circumstances, and engagement in care, including factors that strongly influence adherence, care coordination, and health equity. However, these patient voice signals are rarely available in structured form, limiting their use in patient-centered outcomes research and clinical quality improvement. Reliable extraction of such information is therefore essential for understanding and addressing non-clinical drivers of health outcomes at scale.
Results: We introduce PVminer, a benchmark for structured extraction of patient voice, and propose PVminerLLM, a supervised fine-tuned large language model tailored to this task. Across multiple datasets and model sizes, PVminerLLM substantially outperforms prompt-based baselines, achieving up to 83.82% F1 for Code prediction, 80.74% F1 for Sub-code prediction, and 87.03% F1 for evidence Span extraction. Notably, strong performance is achieved even with smaller models, demonstrating that reliable patient voice extraction is feasible without extreme model scale. These results enable scalable analysis of social and experiential signals embedded in patient-generated text.
Availability and Implementation: Code, evaluation scripts, and trained LLMs will be released publicly. Annotated datasets will be made available upon request for research use.
Keywords: Large Language Models, Supervised Fine-Tuning, Medical Annotation, Patient-Generated Text, Clinical NLP 

---
# NERdME: a Named Entity Recognition Dataset for Indexing Research Artifacts in Code Repositories 

**Authors**: Genet Asefa Gesese, Zongxiong Chen, Shufan Jiang, Mary Ann Tan, Zhaotai Liu, Sonja Schimmler, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2603.05750)  

**Abstract**: Existing scholarly information extraction (SIE) datasets focus on scientific papers and overlook implementation-level details in code repositories. README files describe datasets, source code, and other implementation-level artifacts, however, their free-form Markdown offers little semantic structure, making automatic information extraction difficult. To address this gap, NERdME is introduced: 200 manually annotated README files with over 10,000 labeled spans and 10 entity types. Baseline results using large language models and fine-tuned transformers show clear differences between paperlevel and implementation-level entities, indicating the value of extending SIE benchmarks with entity types available in README files. A downstream entity-linking experiment was conducted to demonstrate that entities derived from READMEs can support artifact discovery and metadata integration. 

---
# CodeScout: Contextual Problem Statement Enhancement for Software Agents 

**Authors**: Manan Suri, Xiangci Li, Mehdi Shojaie, Songyang Han, Chao-Chun Hsu, Shweta Garg, Aniket Anand Deshmukh, Varun Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.05744)  

**Abstract**: Current AI-powered code assistance tools often struggle with poorly-defined problem statements that lack sufficient task context and requirements specification. Recent analysis of software engineering agents reveals that failures on such underspecified requests are highly correlated with longer trajectories involving either over-exploration or repeated attempts at applying the same fix without proper evolution or testing, leading to suboptimal outcomes across software development tasks. We introduce CodeScout, a contextual query refinement approach that systematically converts underspecified user requests into comprehensive, actionable problem statements through lightweight pre-exploration of the target codebase. Our key innovation is demonstrating that structured analysis before task execution can supplement existing agentic capabilities without requiring any modifications to their underlying scaffolds. CodeScout performs targeted context scoping, conducts multi-perspective analysis examining potential fixes and exploration opportunities, then synthesizes these insights into enhanced problem statements with reproduction steps, expected behaviors, and targeted exploration hints. This pre-exploration directly addresses the identified failure patterns by reducing non-converging agent trajectories while clarifying user intent in natural language space. We evaluate CodeScout using state-of-the-art agentic scaffolds and language models on SWEBench-Verified, demonstrating a 20\% improvement in resolution rates with up to 27 additional issues resolved compared to the default baseline method. Our results suggest that systematic query refinement through contextual analysis represents a promising direction for enhancing AI code assistance capabilities. 

---
# Let's Talk, Not Type: An Oral-First Multi-Agent Architecture for Guaraní 

**Authors**: Samantha Adorno, Akshata Kishore Moharir, Ratna Kandala  

**Link**: [PDF](https://arxiv.org/pdf/2603.05743)  

**Abstract**: Although artificial intelligence (AI) and Human-Computer Interaction (HCI) systems are often presented as universal solutions, their design remains predominantly text-first, underserving primarily oral languages and indigenous communities. This position paper uses Guaraní, an official and widely spoken language of Paraguay, as a case study to argue that language support in AI remains insufficient unless it aligns with lived oral practices. We propose an alternative to the standard "text-to-speech" pipeline, proposing instead an oral-first multi-agent architecture. By decoupling Guaraní natural language understanding from dedicated agents for conversation state and community-led governance, we demonstrate a technical framework that respects indigenous data sovereignty and diglossia. Our work moves beyond mere recognition to focus on turn-taking, repair, and shared context as the primary locus of interaction. We conclude that for AI to be truly culturally grounded, it must shift from adapting oral languages to text-centric systems to treating spoken conversation as a first-class design requirement, ensuring digital ecosystems empower rather than overlook diverse linguistic practices. 

---
# Structured Multidimensional Representation Learning for Large Language Models 

**Authors**: Alaa El Ichi, Khalide Jbilou, Mohamed El Guide, Franck Dufrenois  

**Link**: [PDF](https://arxiv.org/pdf/2603.05727)  

**Abstract**: Transformer architectures achieve state-of-the-art performance across a wide range of pattern recognition and natural language processing tasks, but their scaling is accompanied by substantial parameter growth and redundancy in the embedding dimension. In this work, we introduce a structured spectral factorization of the embedding space based on the L-product for third-order tensors. By reshaping token representations into spectral tensor slices and performing attention and feed-forward operations in the transform domain, we obtain a Tensor Transformer architecture that decomposes the encoder into p independent spectral sub-transformers while preserving standard Transformer semantics. We prove that the proposed L-Transformer is spectrally equivalent to p parallel Transformers operating on reduceddimensional embeddings, which yields approximately 1/p reduction (up to lower-order terms such as biases and normalization parameters) in encoder parameters under fixed total embedding size. When instantiated with a real-valued Discrete Cosine Transform (DCT), the method remains fully differentiable and compatible with existing training pipelines. Beyond compression, the spectral decomposition introduces an inductive bias over embedding frequencies, enabling slice-dependent frequency scaling that improves generalization. Experiments on IMDB and AG~News show that the proposed model can substantially reduce encoder parameters (up to 75\% for p=4) while maintaining competitive accuracy. On IMDB, the tensorized encoder matches or improves upon the standard baseline under compression, whereas on AG~News at moderate width we observe a small accuracy decrease in exchange for a 4 times encoder reduction; at BERT-base width (d=768), performance returns to parity. 

---
# Cultural Perspectives and Expectations for Generative AI: A Global Survey Approach 

**Authors**: Erin van Liemt, Renee Shelby, Andrew Smart, Sinchana Kumbale, Richard Zhang, Neha Dixit, Qazi Mamunur Rashid, Jamila Smith-Loud  

**Link**: [PDF](https://arxiv.org/pdf/2603.05723)  

**Abstract**: There is a lack of empirical evidence about global attitudes around whether and how GenAI should represent cultures. This paper assesses understandings and beliefs about culture as it relates to GenAI from a large-scale global survey. We gathered data about what culture means to different groups, and about how GenAI should approach the representation of cultural artifacts, concepts, or values. We distill working definitions of culture directly from these communities to build an understanding of its conceptual complexities and how they relate to representations in Generative AI. We survey from across parts of Europe, North and South America, Asia, and Africa. We conclude with a set of recommendations for Culture and GenAI development. These include participatory approaches, prioritizing specific cultural dimensions beyond geography, such as religion and tradition, and a sensitivity framework for addressing cultural ``redlines''. 

---
# Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis 

**Authors**: Hazem Amamou, Stéphane Gagnon, Alan Davoust, Anderson R. Avila  

**Link**: [PDF](https://arxiv.org/pdf/2603.05698)  

**Abstract**: Retrieval-Augmented Generation (RAG) was introduced to enhance the capabilities of Large Language Models (LLMs) beyond their encoded prior knowledge. This is achieved by providing LLMs with an external source of knowledge, which helps reduce factual hallucinations and enables access to new information not available during pretraining. However, inconsistent retrieved information can negatively affect LLM responses. The Retrieval-Augmented Generation Benchmark (RGB) was introduced to evaluate the robustness of RAG systems under such conditions. In this work, we use the RGB corpus to evaluate LLMs in four scenarios: noise robustness, information integration, negative rejection, and counterfactual robustness. We perform a comparative analysis between the RGB RAG baseline and GraphRAG, a knowledge graph based retrieval system. We test three GraphRAG customizations to improve robustness. Results show improvements over the RGB baseline and provide insights for designing more reliable RAG systems for real world scenarios. 

---
# FreeTxt-Vi: A Benchmarked Vietnamese-English Toolkit for Segmentation, Sentiment, and Summarisation 

**Authors**: Hung Nguyen Huy, Mo El-Haj, Dawn Knight, Paul Rayson  

**Link**: [PDF](https://arxiv.org/pdf/2603.05690)  

**Abstract**: FreeTxt-Vi is a free and open source web based toolkit for creating and analysing bilingual Vietnamese English text collections. Positioned at the intersection of corpus linguistics and natural language processing NLP it enables users to build explore and interpret free text data without requiring programming expertise. The system combines corpus analysis features such as concordancing keyword analysis word relation exploration and interactive visualisation with transformer based NLP components for sentiment analysis and summarisation. A key contribution of this work is the design of a unified bilingual NLP pipeline that integrates a hybrid VnCoreNLP and Byte Pair Encoding BPE segmentation strategy a fine tuned TabularisAI sentiment classifier and a fine tuned Qwen2.5 model for abstractive summarisation. Unlike existing text analysis platforms FreeTxt Vi is evaluated as a set of language processing components. We conduct a three part evaluation covering segmentation sentiment analysis and summarisation and show that our approach achieves competitive or superior performance compared to widely used baselines in both Vietnamese and English. By reducing technical barriers to multilingual text analysis FreeTxt Vi supports reproducible research and promotes the development of language resources for Vietnamese a widely spoken but underrepresented language in NLP. The toolkit is applicable to domains including education digital humanities cultural heritage and the social sciences where qualitative text data are common but often difficult to process at scale. 

---
# The Fragility Of Moral Judgment In Large Language Models 

**Authors**: Tom van Nuenen, Pratik S. Sachdeva  

**Link**: [PDF](https://arxiv.org/pdf/2603.05651)  

**Abstract**: People increasingly use large language models (LLMs) for everyday moral and interpersonal guidance, yet these systems cannot interrogate missing context and judge dilemmas as presented. We introduce a perturbation framework for testing the stability and manipulability of LLM moral judgments while holding the underlying moral conflict constant. Using 2,939 dilemmas from r/AmItheAsshole (January-March 2025), we generate three families of content perturbations: surface edits (lexical/structural noise), point-of-view shifts (voice and stance neutralization), and persuasion cues (self-positioning, social proof, pattern admissions, victim framing). We also vary the evaluation protocol (output ordering, instruction placement, and unstructured prompting). We evaluated all variants with four models (GPT-4.1, Claude 3.7 Sonnet, DeepSeek V3, Qwen2.5-72B) (N=129,156 judgments).
Surface perturbations produce low flip rates (7.5%), largely within the self-consistency noise floor (4-13%), whereas point-of-view shifts induce substantially higher instability (24.3%). A large subset of dilemmas (37.9%) is robust to surface noise yet flips under perspective changes, indicating that models condition on narrative voice as a pragmatic cue. Instability concentrates in morally ambiguous cases; scenarios where no party is assigned blame are most susceptible. Persuasion perturbations yield systematic directional shifts. Protocol choices dominate all other factors: agreement between structured protocols is only 67.6% (kappa=0.55), and only 35.7% of model-scenario units match across all three protocols. These results show that LLM moral judgments are co-produced by narrative form and task scaffolding, raising reproducibility and equity concerns when outcomes depend on presentation skill rather than moral substance. 

---
# Safer Reasoning Traces: Measuring and Mitigating Chain-of-Thought Leakage in LLMs 

**Authors**: Patrick Ahrend, Tobias Eder, Xiyang Yang, Zhiyi Pan, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2603.05618)  

**Abstract**: Chain-of-Thought (CoT) prompting improves LLM reasoning but can increase privacy risk by resurfacing personally identifiable information (PII) from the prompt into reasoning traces and outputs, even under policies that instruct the model not to restate PII. We study such direct, inference-time PII leakage using a model-agnostic framework that (i) defines leakage as risk-weighted, token-level events across 11 PII types, (ii) traces leakage curves as a function of the allowed CoT budget, and (iii) compares open- and closed-source model families on a structured PII dataset with a hierarchical risk taxonomy. We find that CoT consistently elevates leakage, especially for high-risk categories, and that leakage is strongly family- and budget-dependent. Increasing the reasoning budget can either amplify or attenuate leakage depending on the base model. We then benchmark lightweight inference-time gatekeepers: a rule-based detector, a TF-IDF + logistic regression classifier, a GLiNER-based NER model, and an LLM-as-judge, using risk-weighted F1, Macro-F1, and recall. No single method dominates across models or budgets, motivating hybrid, style-adaptive gatekeeping policies that balance utility and risk under a common, reproducible protocol. 

---
# NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution 

**Authors**: Oleksandr Marchenko Breneur, Adelaide Danilov, Aria Nourbakhsh, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2603.05617)  

**Abstract**: We present this http URL, an explainable framework for machine-generated text detection that extends Fast-DetectGPT by integrating curvature-based signals with neural and stylometric features in a supervised setting. The system combines 17 interpretable features, including Conditional Probability Curvature, ModernBERT detector score, readability metrics, and stylometric cues, within a gradient-boosted tree (XGBoost) meta-classifier to determine whether a text is human- or AI-generated. Furthermore, this http URL applies Shapley Additive Explanations (SHAP) to provide both local and global feature-level attribution. These attributions are further translated into structured natural-language rationales through an LLM-based explanation layer, which enables user-facing interpretability. The system is deployed as an interactive web application that supports real-time analysis, visual feature inspection, and structured evidence presentation. A web interface allows users to input text and inspect how neural and statistical signals influence the final decision. The source code and demo video are publicly available to support reproducibility. 

---
# Attention Meets Reachability: Structural Equivalence and Efficiency in Grammar-Constrained LLM Decoding 

**Authors**: Faruk Alpay, Bilge Senturk  

**Link**: [PDF](https://arxiv.org/pdf/2603.05540)  

**Abstract**: We study grammar-constrained decoding (GCD) as a coupling between an autoregressive next-token distribution and a reachability oracle over a pushdown system compiled from a context-free grammar (CFG). We prove an oracle invariance theorem: language-equivalent grammars induce identical admissible next-token sets for every prefix, hence identical logit masks, yet can yield provably different compiled state spaces and online ambiguity costs. We give exact control-state blowup counts for the canonical $a^n b^n$ language under redundant nonterminal delegation, and introduce a left-to-right structural ambiguity cost (SAC) measuring incremental packed-parse-forest growth per token. For two equivalent grammars over all finite strings, SAC is $O(1)$ per token under right-recursion but $\Theta(t^2)$ per token and $\Theta(n^3)$ cumulatively under concatenation. We establish engine-independent lower bounds: any sound, retrieval-efficient, parse-preserving online masking engine must incur $\Omega(t^2)$ work per token on a specific constant-size CFG family, unconditionally within this model. We define decoding-cost equivalence classes of grammars and prove existence of minimal-SAC representatives within bounded rewrite families. Finally, we characterize the true conditional sampler via a Doob $h$-transform and derive sharp one-step KL and total-variation distortion bounds for hard-masked decoding in terms of survival-probability spread among admissible next tokens. We integrate these results with Transformer and Mixture-of-Experts architectures, derive latency envelopes in terms of vocabulary size, active state sets, and beam width, and connect SAC to instrumentation-based predictive performance models and automated grammar optimization. 

---
# Verify as You Go: An LLM-Powered Browser Extension for Fake News Detection 

**Authors**: Dorsaf Sallami, Esma Aïmeur  

**Link**: [PDF](https://arxiv.org/pdf/2603.05519)  

**Abstract**: The rampant spread of fake news in the digital age poses serious risks to public trust and democratic institutions, underscoring the need for effective, transparent, and user-centered detection tools. Existing browser extensions often fall short due to opaque model behavior, limited explanatory support, and a lack of meaningful user engagement. This paper introduces Aletheia, a novel browser extension that leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to detect fake news and provide evidence-based explanations. Aletheia further includes two interactive components: a Discussion Hub that enables user dialogue around flagged content and a Stay Informed feature that surfaces recent fact-checks. Through extensive experiments, we show that Aletheia outperforms state-of-the-art baselines in detection performance. Complementing this empirical evaluation, a complementary user study with 250 participants confirms the system's usability and perceived effectiveness, highlighting its potential as a transparent tool for combating online fake news. 

---
# COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics 

**Authors**: Kartik Sharma, Rakshit S. Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2603.06495)  

**Abstract**: Activation steering methods enable inference-time control of large language model (LLM) behavior without retraining, but current approaches face a fundamental trade-off: sample-efficient methods suboptimally capture steering signals from labeled examples, while methods that better extract these signals require hundreds to thousands of examples. We introduce COLD-Steer, a training-free framework that steers LLM activations by approximating the representational changes that would result from gradient descent on in-context examples. Our key insight is that the effect of fine-tuning on a small set of examples can be efficiently approximated at inference time without actual parameter updates. We formalize this through two complementary approaches: (i) a unit kernel approximation method that updates the activations directly using gradients with respect to them, normalized across examples, and (ii) a finite-difference approximation requiring only two forward passes regardless of example count. Experiments across a variety of steering tasks and benchmarks demonstrate that COLD-Steer achieves upto 95% steering effectiveness while using 50 times fewer samples compared to the best baseline. COLD-Steer facilitates accommodating diverse perspectives without extensive demonstration data, which we validate through our experiments on pluralistic alignment tasks. Our framework opens new possibilities for adaptive, context-aware model control that can flexibly address varying loss-driven human preferences through principled approximation of learning dynamics rather than specialized training procedures. 

---
# NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches 

**Authors**: Ethan Smith  

**Link**: [PDF](https://arxiv.org/pdf/2603.06492)  

**Abstract**: We introduce NOBLE (Nonlinear lOw-rank Branch for Linear Enhancement), an architectural augmentation that adds nonlinear low-rank branches to transformer linear layers. Unlike LoRA and other parameter-efficient fine-tuning (PEFT) methods, NOBLE is designed for pretraining from scratch. The branch is a permanent part of the architecture as opposed to an adapter for finetuning on top of frozen weights. The branch computes {\sigma}(xWdown)Wup where {\sigma} is a learnable nonlinearity. We evaluate several activation functions and find that CosNet, a two-layer cosine nonlinearity with learnable frequency and phase with a linear projection in between them in the bottleneck space, performs best. NOBLE achieves substantial improvements with minimal overhead: up to 1.47x step speedup to reach baseline eval loss (up to 32% fewer training steps), with as low as 4% additional parameters and 7% step time overhead, resulting in up to 1.22x net wallclock speedup. Experiments on LLMs (250M and 1.5B parameters), BERT, VQGAN, and ViT consistently show improved training efficiency. We identify one caveat: Mixup/CutMix augmentation interferes with NOBLE's benefits in Imagenet classification along with other stochastic augmentations, but when disabled, ViT also improves. This discrepancy is possibly explained by regularization techniques that encourage smoother fits to the target function while NOBLE may specialize more in sharper aspects of the target function. 

---
# SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement 

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2603.06333)  

**Abstract**: Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Goal Drift Index (GDI), a learned multi-signal detector combining semantic, lexical, structural, and distributional measures; (ii) constraint preservation checks that enforce safety-critical invariants such as syntactic correctness and non-hallucination; and (iii) regression-risk quantification to flag improvement cycles that undo prior gains. Across 189 tasks in code generation, mathematical reasoning, and truthfulness, SAHOO produces substantial quality gains, including 18.3 percent improvement in code tasks and 16.8 percent in reasoning, while preserving constraints in two domains and maintaining low violations in truthfulness. Thresholds are calibrated on a small validation set of 18 tasks across three cycles. We further map the capability-alignment frontier, showing efficient early improvement cycles but rising alignment costs later and exposing domain-specific tensions such as fluency versus factuality. SAHOO therefore makes alignment preservation during recursive self-improvement measurable, deployable, and systematically validated at scale. 

---
# Continual Adaptation for Pacific Indigenous Speech Recognition 

**Authors**: Yang Xiao, Aso Mahmudi, Nick Thieberger, Eliathamby Ambikairajah, Eun-Jung Holden, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06310)  

**Abstract**: Speech foundation models struggle with low-resource Pacific Indigenous languages because of severe data scarcity. Furthermore, full fine-tuning risks catastrophic forgetting. To address this gap, we present an empirical study adapting models to real-world Pacific datasets. We investigate how data volume and linguistic features affect adaptation success. Specifically, we evaluate strategies including Full Fine-Tuning and Low-Rank Adaptation (LoRA). Additionally, we analyze a continual learning framework for sequentially acquiring multiple languages. We demonstrate that adapting to these distant languages causes severe internal representational drift. Consequently, these models face a strict plasticity and stability dilemma. While LoRA adapts well initially, it suffers from catastrophic forgetting during sequential learning. Ultimately, this study highlights the urgent need for robust adaptation strategies tailored to underrepresented languages. 

---
# The EpisTwin: A Knowledge Graph-Grounded Neuro-Symbolic Architecture for Personal AI 

**Authors**: Giovanni Servedio, Potito Aghilar, Alessio Mattiace, Gianni Carmosino, Francesco Musicco, Gabriele Conte, Vito Walter Anelli, Tommaso Di Noia, Francesco Maria Donini  

**Link**: [PDF](https://arxiv.org/pdf/2603.06290)  

**Abstract**: Personal Artificial Intelligence is currently hindered by the fragmentation of user data across isolated silos. While Retrieval-Augmented Generation offers a partial remedy, its reliance on unstructured vector similarity fails to capture the latent semantic topology and temporal dependencies essential for holistic sensemaking. We introduce EpisTwin, a neuro-symbolic framework that grounds generative reasoning in a verifiable, user-centric Personal Knowledge Graph. EpisTwin leverages Multimodal Language Models to lift heterogeneous, cross-application data into semantic triples. At inference, EpisTwin enables complex reasoning over the personal semantic graph via an agentic coordinator that combines Graph Retrieval-Augmented Generation with Online Deep Visual Refinement, dynamically re-grounding symbolic entities in their raw visual context. We also introduce PersonalQA-71-100, a synthetic benchmark designed to simulate a realistic user's digital footprint and evaluate EpisTwin performance. Our framework demonstrates robust results across a suite of state-of-the-art judge models, offering a promising direction for trustworthy Personal AI. 

---
# Contrastive-to-Self-Supervised: A Two-Stage Framework for Script Similarity Learning 

**Authors**: Claire Roman, Philippe Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2603.06180)  

**Abstract**: Learning similarity metrics for glyphs and writing systems faces a fundamental challenge: while individual graphemes within invented alphabets can be reliably labeled, the historical relationships between different scripts remain uncertain and contested. We propose a two-stage framework that addresses this epistemological constraint. First, we train an encoder with contrastive loss on labeled invented alphabets, establishing a teacher model with robust discriminative features. Second, we extend to historically attested scripts through teacher-student distillation, where the student learns unsupervised representations guided by the teacher's knowledge but free to discover latent cross-script similarities. The asymmetric setup enables the student to learn deformation-invariant embeddings while inheriting discriminative structure from clean examples. Our approach bridges supervised contrastive learning and unsupervised discovery, enabling both hard boundaries between distinct systems and soft similarities reflecting potential historical influences. Experiments on diverse writing systems demonstrate effective few-shot glyph recognition and meaningful script clustering without requiring ground-truth evolutionary relationships. 

---
# Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR 

**Authors**: Ajinkya Kulkarni, Sandipana Dowerah, Atharva Kulkarni, Tanel Alumäe, Mathew Magimai Doss  

**Link**: [PDF](https://arxiv.org/pdf/2603.06164)  

**Abstract**: Self-supervised learning (SSL) underpins modern audio deepfake detection, yet most prior work centers on a single large wav2vec2-XLSR backbone, leaving compact under studied. We present RAPTOR, Representation Aware Pairwise-gated Transformer for Out-of-domain Recognition a controlled study of compact SSL backbones from the HuBERT and WavLM within a unified pairwise-gated fusion detector, evaluated across 14 cross-domain benchmarks. We show that multilingual HuBERT pre-training is the primary driver of cross-domain robustness, enabling 100M models to match larger and commercial systems. Beyond EER, we introduce a test-time augmentation protocol with perturbation-based aleatoric uncertainty to expose calibration differences invisible to standard metrics: WavLM variants exhibit overconfident miscalibration under perturbation, whereas iterative mHuBERT remains stable. These findings indicate that SSL pre-training trajectory, not model scale, drives reliable audio deepfake detection. 

---
# DeepSight: Bridging Depth Maps and Language with a Depth-Driven Multimodal Model 

**Authors**: Hao Yang, Hongbo Zhang, Yanyan Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.06090)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive performance across various tasks such as image captioning and visual question answer(VQA); however, they often struggle to accurately interpret depth information inherent in visual data. In this work, we introduce DeepSight, the first dedicated depth MLLM designed to enhance three-dimensional scene understanding. Unlike conventional methods that align RGB image encodings with text, our approach takes advantage of the unique characteristics of depth images: single-channel grayscale images where the pixel values directly reflect depth cues to improve spatial reasoning. To address challenges associated with limited depth data and the inadequacy of simple channel replication, we construct a novel depth image-text pair dataset and a depth instruction dataset. Depth maps are generated from visual images using the GLPN model, and GPT-4 is employed to curate corresponding depth instructions, an approach validated by LLaVA. Additionally, we modify the ViT encoder in CLIP to incorporate local object information, thereby capturing the subtle continuous variations of depth more effectively. To evaluate the performance of our model, we develop a comprehensive depth question answer benchmark based on existing depth image datasets, which rigorously assesses understanding in typical depth map scenarios. Experimental results demonstrate that DeepSight significantly enhances depth perception and downstream task performance, marking a substantial step forward in multimodal three-dimensional understanding. 

---
# Imagine How To Change: Explicit Procedure Modeling for Change Captioning 

**Authors**: Jiayang Sun, Zixin Guo, Min Cao, Guibo Zhu, Jorma Laaksonen  

**Link**: [PDF](https://arxiv.org/pdf/2603.05969)  

**Abstract**: Change captioning generates descriptions that explicitly describe the differences between two visually similar images. Existing methods operate on static image pairs, thus ignoring the rich temporal dynamics of the change procedure, which is the key to understand not only what has changed but also how it occurs. We introduce ProCap, a novel framework that reformulates change modeling from static image comparison to dynamic procedure modeling. ProCap features a two-stage design: The first stage trains a procedure encoder to learn the change procedure from a sparse set of keyframes. These keyframes are obtained by automatically generating intermediate frames to make the implicit procedural dynamics explicit and then sampling them to mitigate redundancy. Then the encoder learns to capture the latent dynamics of these keyframes via a caption-conditioned, masked reconstruction task. The second stage integrates this trained encoder within an encoder-decoder model for captioning. Instead of relying on explicit frames from the previous stage -- a process incurring computational overhead and sensitivity to visual noise -- we introduce learnable procedure queries to prompt the encoder for inferring the latent procedure representation, which the decoder then translates into text. The entire model is then trained end-to-end with a captioning loss, ensuring the encoder's output is both temporally coherent and captioning-aligned. Experiments on three datasets demonstrate the effectiveness of ProCap. Code and pre-trained models are available at this https URL 

---
# Test-Time Adaptation via Many-Shot Prompting: Benefits, Limits, and Pitfalls 

**Authors**: Shubhangi Upasani, Chen Wu, Jay Rainton, Bo Li, Changran Hu, Qizheng Zhang, Urmish Thakker  

**Link**: [PDF](https://arxiv.org/pdf/2603.05829)  

**Abstract**: Test-time adaptation enables large language models (LLMs) to modify their behavior at inference without updating model parameters. A common approach is many-shot prompting, where large numbers of in-context learning (ICL) examples are injected as an input-space test-time update. Although performance can improve as more demonstrations are added, the reliability and limits of this update mechanism remain poorly understood, particularly for open-source models. We present an empirical study of many-shot prompting across tasks and model backbones, analyzing how performance varies with update magnitude, example ordering, and selection policy. We further study Dynamic and Reinforced ICL as alternative test-time update strategies that control which information is injected and how it constrains model behavior. We find that many-shot prompting is effective for structured tasks where demonstrations provide high information gain, but is highly sensitive to selection strategy and often shows limited benefits for open-ended generation tasks. Overall, we characterize the practical limits of prompt-based test-time adaptation and outline when input-space updates are beneficial versus harmful. 

---
# Proof-of-Guardrail in AI Agents and What (Not) to Trust from It 

**Authors**: Xisen Jin, Michael Duan, Qin Lin, Aaron Chan, Zhenglun Chen, Junyi Du, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2603.05786)  

**Abstract**: As AI agents become widely deployed as online services, users often rely on an agent developer's claim about how safety is enforced, which introduces a threat where safety measures are falsely advertised. To address the threat, we propose proof-of-guardrail, a system that enables developers to provide cryptographic proof that a response is generated after a specific open-source guardrail. To generate proof, the developer runs the agent and guardrail inside a Trusted Execution Environment (TEE), which produces a TEE-signed attestation of guardrail code execution verifiable by any user offline. We implement proof-of-guardrail for OpenClaw agents and evaluate latency overhead and deployment cost. Proof-of-guardrail ensures integrity of guardrail execution while keeping the developer's agent private, but we also highlight a risk of deception about safety, for example, when malicious developers actively jailbreak the guardrail. Code and demo video: this https URL 

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
# CBR-to-SQL: Rethinking Retrieval-based Text-to-SQL using Case-based Reasoning in the Healthcare Domain 

**Authors**: Hung Nguyen, Hans Moen, Pekka Marttinen  

**Link**: [PDF](https://arxiv.org/pdf/2603.05569)  

**Abstract**: Extracting insights from Electronic Health Record (EHR) databases often requires SQL expertise, creating a barrier for healthcare decision-making and research. While a promising approach is to use Large Language Models (LLMs) to translate natural language questions to SQL via Retrieval-Augmented Generation (RAG), adapting this approach to the medical domain is non-trivial. Standard RAG relies on single-step retrieval from a static pool of examples, which struggles with the variability and noise of medical terminology and jargon. This often leads to anti-patterns such as expanding the task demonstration pool to improve coverage, which in turn introduces noise and scalability problems. To address this, we introduce CBR-to-SQL, a framework inspired by Case-Based Reasoning (CBR). It represents question-SQL pairs as reusable, abstract case templates and utilizes a two-stage retrieval process that first captures logical structure and then resolves relevant entities. Evaluated on MIMICSQL, CBR-to-SQL achieves state-of-the-art logical form accuracy and competitive execution accuracy. More importantly, it demonstrates higher sample efficiency and robustness than standard RAG approaches, particularly under data scarcity and retrieval perturbations. 

---
# Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment 

**Authors**: Xiang Ma, Lexin Fang, Litian Xu, Caiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05566)  

**Abstract**: Cross-modal alignment is a crucial task in multimodal learning aimed at achieving semantic consistency between vision and language. This requires that image-text pairs exhibit similar semantics. Traditional algorithms pursue embedding consistency to achieve semantic consistency, ignoring the non-semantic information present in the embedding. An intuitive approach is to decouple the embeddings into semantic and modality components, aligning only the semantic component. However, this introduces two main challenges: (1) There is no established standard for distinguishing semantic and modal information. (2) The modality gap can cause semantic alignment deviation or information loss. To align the true semantics, we propose a novel cross-modal alignment algorithm via \textbf{C}onstrained \textbf{D}ecoupling and \textbf{D}istribution \textbf{S}ampling (CDDS). Specifically, (1) A dual-path UNet is introduced to adaptively decouple the embeddings, applying multiple constraints to ensure effective separation. (2) A distribution sampling method is proposed to bridge the modality gap, ensuring the rationality of the alignment process. Extensive experiments on various benchmarks and model backbones demonstrate the superiority of CDDS, outperforming state-of-the-art methods by 6.6\% to 14.2\%. 

---
# EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair 

**Authors**: Jiaao Chen, Jingyuan Qi, Mingye Gao, Wei-Chen Wang, Hanrui Wang, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.05553)  

**Abstract**: Function-calling agents -- large language models that invoke tools and APIs -- require high-quality, domain-specific training data spanning executable environments, backing databases, and diverse multi-turn trajectories. We introduce EigenData, an integrated, self-evolving platform that automates the full data lifecycle through a multi-agent architecture. A top-level orchestrator, EigenCore, coordinates three specialized sub-systems: DatabaseAgent for realistic domain database construction, CodingAgent for verified executable environment generation with iterative test-debug loops, and DataAgent for multi-turn trajectory synthesis with self-evolving prompt optimization. Cross-component feedback ensures consistency across all artifacts. We apply EigenData to audit and repair the Berkeley Function-Calling Leaderboard (BFCL-V3), identifying systematic errors in function schemas, implementations, and reference trajectories, automatically correcting them through coordinated schema refinement, code-level bug fixes, and trajectory modification, and introducing an outcome-aware evaluation protocol that assesses task success via database-state correctness rather than turn-level trajectory matching. We demonstrate that the repaired benchmark, coupled with outcome-aware metrics, produces model rankings substantially better correlated with human judgments of functional correctness. 

---
# Omni-C: Compressing Heterogeneous Modalities into a Single Dense Encoder 

**Authors**: Kin Wai Lau, Yasar Abbas Ur Rehman, Lai-Man Po, Pedro Porto Buarque de Gusmão  

**Link**: [PDF](https://arxiv.org/pdf/2603.05528)  

**Abstract**: Recent multimodal systems often rely on separate expert modality encoders which cause linearly scaling complexity and computational overhead with added modalities. While unified Omni-models address this via Mixture-of-Expert (MoE) architectures with specialized experts and routing, they still inflate parameter counts and introduce routing overhead. In this paper, we propose Omni-C (Omni-Compress), a single dense Transformer-based encoder that learns competitive shared representations across heterogeneous modalities--images, audio, and text--through unimodal contrastive pretraining on large-scale unaligned data. By maximizing parameter sharing in the backbone and using lightweight modality-specific projection heads, Omni-C effectively mitigates inter-modality conflicts without requiring MoE, paired supervision, or routing. This design supports efficient deployment on memory-constrained systems via sequential modality processing and low-memory inference, eliminating the need for parallel expert loading or specialized hardware. Experiments show Omni-C achieves performance comparable to expert models in unimodal and cross-model tasks, with modest zero-shot degradation on audio and text that is largely recovered through lightweight linear probing or parameter efficient fine-tuning. The unified architecture substantially reduces inference memory usage compared to multi-encoder baselines, advancing efficient and scalable multimodal learning. 

---
