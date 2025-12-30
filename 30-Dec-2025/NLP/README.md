# Eliciting Behaviors in Multi-Turn Conversations 

**Authors**: Jing Huang, Shujian Zhang, Lun Wang, Andrew Hard, Rajiv Mathews, John Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2512.23701)  

**Abstract**: Identifying specific and often complex behaviors from large language models (LLMs) in conversational settings is crucial for their evaluation. Recent work proposes novel techniques to find natural language prompts that induce specific behaviors from a target model, yet they are mainly studied in single-turn settings. In this work, we study behavior elicitation in the context of multi-turn conversations. We first offer an analytical framework that categorizes existing methods into three families based on their interactions with the target model: those that use only prior knowledge, those that use offline interactions, and those that learn from online interactions. We then introduce a generalized multi-turn formulation of the online method, unifying single-turn and multi-turn elicitation. We evaluate all three families of methods on automatically generating multi-turn test cases. We investigate the efficiency of these approaches by analyzing the trade-off between the query budget, i.e., the number of interactions with the target model, and the success rate, i.e., the discovery rate of behavior-eliciting inputs. We find that online methods can achieve an average success rate of 45/19/77% with just a few thousand queries over three tasks where static methods from existing multi-turn conversation benchmarks find few or even no failure cases. Our work highlights a novel application of behavior elicitation methods in multi-turn conversation evaluation and the need for the community to move towards dynamic benchmarks. 

---
# Fine-Tuning LLMs with Fine-Grained Human Feedback on Text Spans 

**Authors**: Sky CH-Wang, Justin Svegliato, Helen Appel, Jason Eisner  

**Link**: [PDF](https://arxiv.org/pdf/2512.23693)  

**Abstract**: We present a method and dataset for fine-tuning language models with preference supervision using feedback-driven improvement chains. Given a model response, an annotator provides fine-grained feedback by marking ``liked'' and ``disliked'' spans and specifying what they liked or disliked about them. The base model then rewrites the disliked spans accordingly, proceeding from left to right, forming a sequence of incremental improvements. We construct preference pairs for direct alignment from each adjacent step in the chain, enabling the model to learn from localized, targeted edits. We find that our approach outperforms direct alignment methods based on standard A/B preference ranking or full contrastive rewrites, demonstrating that structured, revision-based supervision leads to more efficient and effective preference tuning. 

---
# PROFASR-BENCH: A Benchmark for Context-Conditioned ASR in High-Stakes Professional Speech 

**Authors**: Deepak Babu Piskala  

**Link**: [PDF](https://arxiv.org/pdf/2512.23686)  

**Abstract**: Automatic Speech Recognition (ASR) in professional settings faces challenges that existing benchmarks underplay: dense domain terminology, formal register variation, and near-zero tolerance for critical entity errors. We present ProfASR-Bench, a professional-talk evaluation suite for high-stakes applications across finance, medicine, legal, and technology. Each example pairs a natural-language prompt (domain cue and/or speaker profile) with an entity-rich target utterance, enabling controlled measurement of context-conditioned recognition. The corpus supports conventional ASR metrics alongside entity-aware scores and slice-wise reporting by accent and gender. Using representative families Whisper (encoder-decoder ASR) and Qwen-Omni (audio language models) under matched no-context, profile, domain+profile, oracle, and adversarial conditions, we find a consistent pattern: lightweight textual context produces little to no change in average word error rate (WER), even with oracle prompts, and adversarial prompts do not reliably degrade performance. We term this the context-utilization gap (CUG): current systems are nominally promptable yet underuse readily available side information. ProfASR-Bench provides a standardized context ladder, entity- and slice-aware reporting with confidence intervals, and a reproducible testbed for comparing fusion strategies across model families.
Dataset: this https URL
Code: this https URL 

---
# Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing 

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2512.23684)  

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages. 

---
# Less is more: Probabilistic reduction is best explained by small-scale predictability measures 

**Authors**: Cassandra L. Jacobs, Andrés Buxó-Lugo, Anna K. Taylor, Marie Leopold-Hooke  

**Link**: [PDF](https://arxiv.org/pdf/2512.23659)  

**Abstract**: The primary research questions of this paper center on defining the amount of context that is necessary and/or appropriate when investigating the relationship between language model probabilities and cognitive phenomena. We investigate whether whole utterances are necessary to observe probabilistic reduction and demonstrate that n-gram representations suffice as cognitive units of planning. 

---
# Nested Browser-Use Learning for Agentic Information Seeking 

**Authors**: Baixuan Li, Jialong Wu, Wenbiao Yin, Kuan Li, Zhongwang Zhang, Huifeng Yin, Zhengwei Tao, Liwen Zhang, Pengjun Xie, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23647)  

**Abstract**: Information-seeking (IS) agents have achieved strong performance across a range of wide and deep search tasks, yet their tool use remains largely restricted to API-level snippet retrieval and URL-based page fetching, limiting access to the richer information available through real browsing. While full browser interaction could unlock deeper capabilities, its fine-grained control and verbose page content returns introduce substantial complexity for ReAct-style function-calling agents. To bridge this gap, we propose Nested Browser-Use Learning (NestBrowse), which introduces a minimal and complete browser-action framework that decouples interaction control from page exploration through a nested structure. This design simplifies agentic reasoning while enabling effective deep-web information acquisition. Empirical results on challenging deep IS benchmarks demonstrate that NestBrowse offers clear benefits in practice. Further in-depth analyses underscore its efficiency and flexibility. 

---
# A Dataset and Benchmark for Consumer Healthcare Question Summarization 

**Authors**: Abhishek Basu, Deepak Gupta, Dina Demner-Fushman, Shweta Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2512.23637)  

**Abstract**: The quest for seeking health information has swamped the web with consumers health-related questions. Generally, con- sumers use overly descriptive and peripheral information to express their medical condition or other healthcare needs, contributing to the challenges of natural language understanding. One way to address this challenge is to summarize the questions and distill the key information of the original question. Recently, large-scale datasets have significantly propelled the development of several summarization tasks, such as multi-document summarization and dialogue summarization. However, a lack of a domain-expert annotated dataset for the consumer healthcare questions summarization task inhibits the development of an efficient summarization system. To address this issue, we introduce a new dataset, CHQ-Sum,m that contains 1507 domain-expert annotated consumer health questions and corresponding summaries. The dataset is derived from the community question answering forum and therefore provides a valuable resource for understanding consumer health-related posts on social media. We benchmark the dataset on multiple state-of-the-art summarization models to show the effectiveness of the dataset 

---
# Close the Loop: Synthesizing Infinite Tool-Use Data via Multi-Agent Role-Playing 

**Authors**: Yuwen Li, Wei Zhang, Zelong Huang, Mason Yang, Jiajun Wu, Shawn Guo, Huahao Hu, Lingyi Sun, Jian Yang, Mingjie Tang, Byran Dai  

**Link**: [PDF](https://arxiv.org/pdf/2512.23611)  

**Abstract**: Enabling Large Language Models (LLMs) to reliably invoke external tools remains a critical bottleneck for autonomous agents. Existing approaches suffer from three fundamental challenges: expensive human annotation for high-quality trajectories, poor generalization to unseen tools, and quality ceilings inherent in single-model synthesis that perpetuate biases and coverage gaps. We introduce InfTool, a fully autonomous framework that breaks these barriers through self-evolving multi-agent synthesis. Given only raw API specifications, InfTool orchestrates three collaborative agents (User Simulator, Tool-Calling Assistant, and MCP Server) to generate diverse, verified trajectories spanning single-turn calls to complex multi-step workflows. The framework establishes a closed loop: synthesized data trains the model via Group Relative Policy Optimization (GRPO) with gated rewards, the improved model generates higher-quality data targeting capability gaps, and this cycle iterates without human intervention. Experiments on the Berkeley Function-Calling Leaderboard (BFCL) demonstrate that InfTool transforms a base 32B model from 19.8% to 70.9% accuracy (+258%), surpassing models 10x larger and rivaling Claude-Opus, and entirely from synthetic data without human annotation. 

---
# Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models 

**Authors**: Yu-Xiang Lin, Cheng-Han Chiang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.23578)  

**Abstract**: In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that when SLMs are asked to recall the style instruction in later turns, they can recall the style instruction, but they fail to express it throughout the conversation. We also show that explicitly asking the model to recall the style instruction can partially mitigate style amnesia. In addition, we examine various prompting strategies and find that SLMs struggle to follow the required style when the instruction is placed in system messages rather than user messages, which contradicts the intended function of system prompts. 

---
# Instruction-Following Evaluation of Large Vision-Language Models 

**Authors**: Daiki Shiono, Shumpei Miyawaki, Ryota Tanaka, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2512.23572)  

**Abstract**: Following the initial flourishing of large language models (LLMs), there has been a surge in proposed large vision-language models (LVLMs) that integrate LLMs with vision capabilities. However, it has been observed that LVLMs, after tuning to visual instruction using commonly used training datasets, often fail to exhibit the instruction-following ability that was present in the LLM before integration, leading to results in which they do not follow task instructions as expected. This study quantitatively demonstrates that LVLMs' instruction-following ability declines after fine-tuning and analyzes its underlying causes. In particular, we constructed new training datasets highlighting whether the output format is specified. Then, we investigated how explicitly indicating the output format during fine-tuning affects LVLMs' instruction-following ability. Our quantitative evaluation confirmed that LVLMs' instruction-following ability declines after fine-tuning with commonly used datasets. Furthermore, we found that LVLMs trained with datasets, including instructions on output format, tend to follow instructions more accurately than models that do not. These findings suggest that including samples with instructions on output format during (visual) instruction tuning may help mitigate the decline in instruction-following abilities. 

---
# Lie to Me: Knowledge Graphs for Robust Hallucination Self-Detection in LLMs 

**Authors**: Sahil Kale, Antonio Luca Alfeo  

**Link**: [PDF](https://arxiv.org/pdf/2512.23547)  

**Abstract**: Hallucinations, the generation of apparently convincing yet false statements, remain a major barrier to the safe deployment of LLMs. Building on the strong performance of self-detection methods, we examine the use of structured knowledge representations, namely knowledge graphs, to improve hallucination self-detection. Specifically, we propose a simple yet powerful approach that enriches hallucination self-detection by (i) converting LLM responses into knowledge graphs of entities and relations, and (ii) using these graphs to estimate the likelihood that a response contains hallucinations. We evaluate the proposed approach using two widely used LLMs, GPT-4o and Gemini-2.5-Flash, across two hallucination detection datasets. To support more reliable future benchmarking, one of these datasets has been manually curated and enhanced and is released as a secondary outcome of this work. Compared to standard self-detection methods and SelfCheckGPT, a state-of-the-art approach, our method achieves up to 16% relative improvement in accuracy and 20% in F1-score. Our results show that LLMs can better analyse atomic facts when they are structured as knowledge graphs, even when initial outputs contain inaccuracies. This low-cost, model-agnostic approach paves the way toward safer and more trustworthy language models. 

---
# Single LLM Debate, MoLaCE: Mixture of Latent Concept Experts Against Confirmation Bias 

**Authors**: Hazel Kim, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2512.23518)  

**Abstract**: Large language models (LLMs) are highly vulnerable to input confirmation bias. When a prompt implies a preferred answer, models often reinforce that bias rather than explore alternatives. This phenomenon remains underexplored, yet it is already harmful in base models and poses an even greater risk in multi-agent debate, where echo chambers reinforce bias instead of correction. We introduce Mixture of Latent Concept Experts (MoLaCE), a lightweight inference-time framework that addresses confirmation bias by mixing experts instantiated as different activation strengths over latent concepts that shape model responses. Our key insight is that, due to the compositional nature of language, differently phrased prompts reweight latent concepts in prompt-specific ways that affect factual correctness, so no single fixed intervention can be applied universally across inputs. This design enables a single LLM to emulate the benefits of debate internally while remaining computationally efficient and scalable. It can also be integrated into multi-agent debate frameworks to diversify perspectives and reduce correlated errors. We empirically show that it consistently reduces confirmation bias, improves robustness, and matches or surpasses multi-agent debate while requiring only a fraction of the computation. 

---
# UniHetero: Could Generation Enhance Understanding for Vision-Language-Model at Large Data Scale? 

**Authors**: Fengjiao Chen, Minhao Jing, Weitao Lu, Yan Feng, Xiaoyu Li, Xuezhi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23512)  

**Abstract**: Vision-language large models are moving toward the unification of visual understanding and visual generation tasks. However, whether generation can enhance understanding is still under-explored on large data scale. In this work, we analysis the unified model with a concise structure, UniHetero, under large-scale pretraining (>200M samples). Our key observations are: (1) Generation can improve understanding, but Only if you generate Semantics, Not Pixels. (2) Generation reveals a superior Data Scaling trend and higher Data Utilization. (3) Autoregression on Input Embedding is effective to capture visual details. 

---
# Automatic Detection of Complex Quotation Patterns in Aggadic Literature 

**Authors**: Hadar Miller, Tsvi Kuflik, Moshe Lavee  

**Link**: [PDF](https://arxiv.org/pdf/2512.23504)  

**Abstract**: This paper presents ACT (Allocate Connections between Texts), a novel three-stage algorithm for the automatic detection of biblical quotations in Rabbinic literature. Unlike existing text reuse frameworks that struggle with short, paraphrased, or structurally embedded quotations, ACT combines a morphology-aware alignment algorithm with a context-sensitive enrichment stage that identifies complex citation patterns such as "Wave" and "Echo" quotations.
Our approach was evaluated against leading systems, including Dicta, Passim, Text-Matcher, as well as human-annotated critical editions. We further assessed three ACT configurations to isolate the contribution of each component. Results demonstrate that the full ACT pipeline (ACT-QE) outperforms all baselines, achieving an F1 score of 0.91, with superior Recall (0.89) and Precision (0.94). Notably, ACT-2, which lacks stylistic enrichment, achieves higher Recall (0.90) but suffers in Precision, while ACT-3, using longer n-grams, offers a tradeoff between coverage and specificity.
In addition to improving quotation detection, ACT's ability to classify stylistic patterns across corpora opens new avenues for genre classification and intertextual analysis. This work contributes to digital humanities and computational philology by addressing the methodological gap between exhaustive machine-based detection and human editorial judgment. ACT lays a foundation for broader applications in historical textual analysis, especially in morphologically rich and citation-dense traditions like Aggadic literature. 

---
# Semantic Tree Inference on Text Corpa using a Nested Density Approach together with Large Language Model Embeddings 

**Authors**: Thomas Haschka, Joseph Bakarji  

**Link**: [PDF](https://arxiv.org/pdf/2512.23471)  

**Abstract**: Semantic text classification has undergone significant advances in recent years due to the rise of large language models (LLMs) and their high dimensional embeddings. While LLM-embeddings are frequently used to store and retrieve text by semantic similarity in vector databases, the global structure semantic relationships in text corpora often remains opaque. Herein we propose a nested density clustering approach, to infer hierarchical trees of semantically related texts. The method starts by identifying texts of strong semantic similarity as it searches for dense clusters in LLM embedding space. As the density criterion is gradually relaxed, these dense clusters merge into more diffuse clusters, until the whole dataset is represented by a single cluster - the root of the tree. By embedding dense clusters into increasingly diffuse ones, we construct a tree structure that captures hierarchical semantic relationships among texts. We outline how this approach can be used to classify textual data for abstracts of scientific abstracts as a case study. This enables the data-driven discovery research areas and their subfields without predefined categories. To evaluate the general applicability of the method, we further apply it to established benchmark datasets such as the 20 News- groups and IMDB 50k Movie Reviews, demonstrating its robustness across domains. Finally we discuss possible applications on scientometrics, topic evolution, highlighting how nested density trees can reveal semantic structure and evolution in textual datasets. 

---
# Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss 

**Authors**: Ang Lv, Jin Ma, Yiyuan Ma, Siyuan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23447)  

**Abstract**: Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router's decisions align well with the experts' capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router's decisions with expert capabilities. Our approach treats each expert's router embedding as a proxy token for the tokens assigned to that expert, and feeds perturbed router embeddings through the experts to obtain internal activations. The ERC loss enforces two constraints on these activations: (1) Each expert must exhibit higher activation for its own proxy token than for the proxy tokens of any other expert. (2) Each proxy token must elicit stronger activation from its corresponding expert than from any other expert. These constraints jointly ensure that each router embedding faithfully represents its corresponding expert's capability, while each expert specializes in processing the tokens actually routed to it. The ERC loss is computationally efficient, operating only on n^2 activations, where n is the number of experts. This represents a fixed cost independent of batch size, unlike prior coupling methods that scale with the number of tokens (often millions per batch). Through pre-training MoE-LLMs ranging from 3B to 15B parameters and extensive analysis on trillions of tokens, we demonstrate the effectiveness of the ERC loss. Moreover, the ERC loss offers flexible control and quantitative tracking of expert specialization levels during training, providing valuable insights into MoEs. 

---
# ClinDEF: A Dynamic Evaluation Framework for Large Language Models in Clinical Reasoning 

**Authors**: Yuqi Tang, Jing Yu, Zichang Su, Kehua Feng, Zhihui Zhu, Libin Wang, Lei Liang, Qiang Zhang, Keyan Ding, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.23440)  

**Abstract**: Clinical diagnosis begins with doctor-patient interaction, during which physicians iteratively gather information, determine examination and refine differential diagnosis through patients' response. This dynamic clinical-reasoning process is poorly represented by existing LLM benchmarks that focus on static question-answering. To mitigate these gaps, recent methods explore dynamic medical frameworks involving interactive clinical dialogues. Although effective, they often rely on limited, contamination-prone datasets and lack granular, multi-level evaluation. In this work, we propose ClinDEF, a dynamic framework for assessing clinical reasoning in LLMs through simulated diagnostic dialogues. Grounded in a disease knowledge graph, our method dynamically generates patient cases and facilitates multi-turn interactions between an LLM-based doctor and an automated patient agent. Our evaluation protocol goes beyond diagnostic accuracy by incorporating fine-grained efficiency analysis and rubric-based assessment of diagnostic quality. Experiments show that ClinDEF effectively exposes critical clinical reasoning gaps in state-of-the-art LLMs, offering a more nuanced and clinically meaningful evaluation paradigm. 

---
# C2PO: Diagnosing and Disentangling Bias Shortcuts in LLMs 

**Authors**: Xuan Feng, Bo An, Tianlong Gu, Liang Chang, Fengrui Hao, Peipeng Yu, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23430)  

**Abstract**: Bias in Large Language Models (LLMs) poses significant risks to trustworthiness, manifesting primarily as stereotypical biases (e.g., gender or racial stereotypes) and structural biases (e.g., lexical overlap or position preferences). However, prior paradigms typically address these in isolation, often mitigating one at the expense of exacerbating the other. To address this, we conduct a systematic exploration of these reasoning failures and identify a primary inducement: the latent spurious feature correlations within the input that drive these erroneous reasoning shortcuts. Driven by these findings, we introduce Causal-Contrastive Preference Optimization (C2PO), a unified alignment framework designed to tackle these specific failures by simultaneously discovering and suppressing these correlations directly within the optimization process. Specifically, C2PO leverages causal counterfactual signals to isolate bias-inducing features from valid reasoning paths, and employs a fairness-sensitive preference update mechanism to dynamically evaluate logit-level contributions and suppress shortcut features. Extensive experiments across multiple benchmarks covering stereotypical bias (BBQ, Unqover), structural bias (MNLI, HANS, Chatbot, MT-Bench), out-of-domain fairness (StereoSet, WinoBias), and general utility (MMLU, GSM8K) demonstrate that C2PO effectively mitigates stereotypical and structural biases while preserving robust general reasoning capabilities. 

---
# The Effect of Gender Diversity on Scientific Team Impact: A Team Roles Perspective 

**Authors**: Yi Zhao, Yongjun Zhu, Donghun Kim, Yuzhuo Wang, Heng Zhang, Chao Lu, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23429)  

**Abstract**: The influence of gender diversity on the success of scientific teams is of great interest to academia. However, prior findings remain inconsistent, and most studies operationalize diversity in aggregate terms, overlooking internal role differentiation. This limitation obscures a more nuanced understanding of how gender diversity shapes team impact. In particular, the effect of gender diversity across different team roles remains poorly understood. To this end, we define a scientific team as all coauthors of a paper and measure team impact through five-year citation counts. Using author contribution statements, we classified members into leadership and support roles. Drawing on more than 130,000 papers from PLOS journals, most of which are in biomedical-related disciplines, we employed multivariable regression to examine the association between gender diversity in these roles and team impact. Furthermore, we apply a threshold regression model to investigate how team size moderates this relationship. The results show that (1) the relationship between gender diversity and team impact follows an inverted U-shape for both leadership and support groups; (2) teams with an all-female leadership group and an all-male support group achieve higher impact than other team types. Interestingly, (3) the effect of leadership-group gender diversity is significantly negative for small teams but becomes positive and statistically insignificant in large teams. In contrast, the estimates for support-group gender diversity remain significant and positive, regardless of team size. 

---
# Entropy-Guided Token Dropout: Training Autoregressive Language Models with Limited Domain Data 

**Authors**: Jiapeng Wang, Yiwen Hu, Yanzipeng Gao, Haoyu Wang, Shuo Wang, Hongyu Lu, Jiaxin Mao, Wayne Xin Zhao, Junyi Li, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23422)  

**Abstract**: As access to high-quality, domain-specific data grows increasingly scarce, multi-epoch training has become a practical strategy for adapting large language models (LLMs). However, autoregressive models often suffer from performance degradation under repeated data exposure, where overfitting leads to a marked decline in model capability. Through empirical analysis, we trace this degradation to an imbalance in learning dynamics: predictable, low-entropy tokens are learned quickly and come to dominate optimization, while the model's ability to generalize on high-entropy tokens deteriorates with continued training. To address this, we introduce EntroDrop, an entropy-guided token dropout method that functions as structured data regularization. EntroDrop selectively masks low-entropy tokens during training and employs a curriculum schedule to adjust regularization strength in alignment with training progress. Experiments across model scales from 0.6B to 8B parameters show that EntroDrop consistently outperforms standard regularization baselines and maintains robust performance throughout extended multi-epoch training. These findings underscore the importance of aligning regularization with token-level learning dynamics when training on limited data. Our approach offers a promising pathway toward more effective adaptation of LLMs in data-constrained domains. 

---
# A Stepwise-Enhanced Reasoning Framework for Large Language Models Based on External Subgraph Generation 

**Authors**: Xin Zhang, Yang Cao, Baoxing Wu, Xinyi Chen, Kai Song, Siying Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.23356)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across a wide range of natural language processing tasks in recent years, including machine translation, text generation, and question answering. As their applications extend to increasingly complex scenarios, however, LLMs continue to face challenges in tasks that require deep reasoning and logical inference. In particular, models trained on large scale textual corpora may incorporate noisy or irrelevant information during generation, which can lead to incorrect predictions or outputs that are inconsistent with factual knowledge. To address this limitation, we propose a stepwise reasoning enhancement framework for LLMs based on external subgraph generation, termed SGR. The proposed framework dynamically constructs query relevant subgraphs from external knowledge bases and leverages their semantic structure to guide the reasoning process. By performing reasoning in a step by step manner over structured subgraphs, SGR reduces the influence of noisy information and improves reasoning accuracy. Specifically, the framework first generates an external subgraph tailored to the input query, then guides the model to conduct multi step reasoning grounded in the subgraph, and finally integrates multiple reasoning paths to produce the final answer. Experimental results on multiple benchmark datasets demonstrate that SGR consistently outperforms strong baselines, indicating its effectiveness in enhancing the reasoning capabilities of LLMs. 

---
# AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents 

**Authors**: Jiafeng Liang, Hao Li, Chang Li, Jiaqi Zhou, Shixin Jiang, Zekun Wang, Changkai Ji, Zhihao Zhu, Runxuan Liu, Tao Ren, Jinlan Fu, See-Kiong Ng, Xia Liang, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.23343)  

**Abstract**: Memory serves as the pivotal nexus bridging past and future, providing both humans and AI systems with invaluable concepts and experience to navigate complex tasks. Recent research on autonomous agents has increasingly focused on designing efficient memory workflows by drawing on cognitive neuroscience. However, constrained by interdisciplinary barriers, existing works struggle to assimilate the essence of human memory mechanisms. To bridge this gap, we systematically synthesizes interdisciplinary knowledge of memory, connecting insights from cognitive neuroscience with LLM-driven agents. Specifically, we first elucidate the definition and function of memory along a progressive trajectory from cognitive neuroscience through LLMs to agents. We then provide a comparative analysis of memory taxonomy, storage mechanisms, and the complete management lifecycle from both biological and artificial perspectives. Subsequently, we review the mainstream benchmarks for evaluating agent memory. Additionally, we explore memory security from dual perspectives of attack and defense. Finally, we envision future research directions, with a focus on multimodal memory systems and skill acquisition. 

---
# AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration 

**Authors**: Minjiang Huang, Jipeng Qiang, Yi Zhu, Chaowei Zhang, Xiangyu Zhao, Kui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23300)  

**Abstract**: Audiobook interpretations are attracting increasing attention, as they provide accessible and in-depth analyses of books that offer readers practical insights and intellectual inspiration. However, their manual creation process remains time-consuming and resource-intensive. To address this challenge, we propose AI4Reading, a multi-agent collaboration system leveraging large language models (LLMs) and speech synthesis technology to generate podcast, like audiobook interpretations. The system is designed to meet three key objectives: accurate content preservation, enhanced comprehensibility, and a logical narrative structure. To achieve these goals, we develop a framework composed of 11 specialized agents,including topic analysts, case analysts, editors, a narrator, and proofreaders that work in concert to explore themes, extract real world cases, refine content organization, and synthesize natural spoken language. By comparing expert interpretations with our system's output, the results show that although AI4Reading still has a gap in speech generation quality, the generated interpretative scripts are simpler and more accurate. 

---
# Chinese Morph Resolution in E-commerce Live Streaming Scenarios 

**Authors**: Jiahao Zhu, Jipeng Qiang, Ran Bai, Chenyu Liu, Xiaoye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23280)  

**Abstract**: E-commerce live streaming in China, particularly on platforms like Douyin, has become a major sales channel, but hosts often use morphs to evade scrutiny and engage in false advertising. This study introduces the Live Auditory Morph Resolution (LiveAMR) task to detect such violations. Unlike previous morph research focused on text-based evasion in social media and underground industries, LiveAMR targets pronunciation-based evasion in health and medical live streams. We constructed the first LiveAMR dataset with 86,790 samples and developed a method to transform the task into a text-to-text generation problem. By leveraging large language models (LLMs) to generate additional training data, we improved performance and demonstrated that morph resolution significantly enhances live streaming regulation. 

---
# Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation 

**Authors**: Dianyun Wang, Qingsen Ma, Yuhu Shang, Zhifeng Lu, Lechen Ning, Zhenbo Xu, Huijia Wu, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2512.23260)  

**Abstract**: Parameter-efficient fine-tuning has become the dominant paradigm for adapting large language models to downstream tasks. Low-rank adaptation methods such as LoRA operate under the assumption that task-relevant weight updates reside in a low-rank subspace, yet this subspace is learned implicitly from data in a black-box manner, offering no interpretability or direct control. We hypothesize that this difficulty stems from polysemanticity--individual dimensions encoding multiple entangled concepts. To address this, we leverage pre-trained Sparse Autoencoders (SAEs) to identify task-relevant features in a disentangled feature space, then construct an explicit, interpretable low-rank subspace to guide adapter initialization. We provide theoretical analysis proving that under monosemanticity assumptions, SAE-based subspace identification achieves arbitrarily small recovery error, while direct identification in polysemantic space suffers an irreducible error floor. On safety alignment, our method achieves up to 99.6% safety rate--exceeding full fine-tuning by 7.4 percentage points and approaching RLHF-based methods--while updating only 0.19-0.24% of parameters. Crucially, our method provides interpretable insights into the learned alignment subspace through the semantic grounding of SAE features. Our work demonstrates that incorporating mechanistic interpretability into the fine-tuning process can simultaneously improve both performance and transparency. 

---
# Anka: A Domain-Specific Language for Reliable LLM Code Generation 

**Authors**: Saif Khalfan Saif Al Mazrouei  

**Link**: [PDF](https://arxiv.org/pdf/2512.23214)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet they exhibit systematic errors on complex, multi-step programming tasks. We hypothesize that these errors stem from the flexibility of general-purpose languages, which permits multiple valid approaches and requires implicit state management. To test this hypothesis, we introduce Anka, a domain-specific language (DSL) for data transformation pipelines designed with explicit, constrained syntax that reduces ambiguity in code generation. Despite having zero prior training exposure to Anka, Claude 3.5 Haiku achieves 99.9% parse success and 95.8% overall task accuracy across 100 benchmark problems. Critically, Anka demonstrates a 40 percentage point accuracy advantage over Python on multi-step pipeline tasks (100% vs. 60%), where Python's flexible syntax leads to frequent errors in operation sequencing and variable management. Cross-model validation with GPT-4o-mini confirms this advantage (+26.7 percentage points on multi-step tasks). Our results demonstrate that: (1) LLMs can learn novel DSLs entirely from in-context prompts, achieving near-native accuracy; (2) constrained syntax significantly reduces errors on complex tasks; and (3) domain-specific languages purposefully designed for LLM generation can outperform general-purpose languages on which the LLM has extensive training. We release the complete language implementation, benchmark suite, and evaluation framework to facilitate further research. 

---
# Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process 

**Authors**: Zhijun Chen, Zeyu Ji, Qianren Mao, Junhang Cheng, Bangjie Qin, Hao Wu, Zhuoran Li, Jingzheng Li, Kai Sun, Zizhe Wang, Yikun Ban, Zhu Sun, Xiangyang Ji, Hailong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.23213)  

**Abstract**: We propose LLM-PeerReview, an unsupervised LLM Ensemble method that selects the most ideal response from multiple LLM-generated candidates for each query, harnessing the collective wisdom of multiple models with diverse strengths. LLM-PeerReview is built on a novel, peer-review-inspired framework that offers a clear and interpretable mechanism, while remaining fully unsupervised for flexible adaptability and generalization. Specifically, it operates in three stages: For scoring, we use the emerging LLM-as-a-Judge technique to evaluate each response by reusing multiple LLMs at hand; For reasoning, we can apply a principled graphical model-based truth inference algorithm or a straightforward averaging strategy to aggregate multiple scores to produce a final score for each response; Finally, the highest-scoring response is selected as the best ensemble output. LLM-PeerReview is conceptually simple and empirically powerful. The two variants of the proposed approach obtain strong results across four datasets, including outperforming the recent advanced model Smoothie-Global by 6.9% and 7.3% points, respectively. 

---
# Not too long do read: Evaluating LLM-generated extreme scientific summaries 

**Authors**: Zhuoqi Lyu, Qing Ke  

**Link**: [PDF](https://arxiv.org/pdf/2512.23206)  

**Abstract**: High-quality scientific extreme summary (TLDR) facilitates effective science communication. How do large language models (LLMs) perform in generating them? How are LLM-generated summaries different from those written by human experts? However, the lack of a comprehensive, high-quality scientific TLDR dataset hinders both the development and evaluation of LLMs' summarization ability. To address these, we propose a novel dataset, BiomedTLDR, containing a large sample of researcher-authored summaries from scientific papers, which leverages the common practice of including authors' comments alongside bibliography items. We then test popular open-weight LLMs for generating TLDRs based on abstracts. Our analysis reveals that, although some of them successfully produce humanoid summaries, LLMs generally exhibit a greater affinity for the original text's lexical choices and rhetorical structures, hence tend to be more extractive rather than abstractive in general, compared to humans. Our code and datasets are available at this https URL (Lyu and Ke, 2025). 

---
# Reservoir Computing inspired Matrix Multiplication-free Language Model 

**Authors**: Takumi Shiratsuchi, Yuichiro Tanaka, Hakaru Tamukoh  

**Link**: [PDF](https://arxiv.org/pdf/2512.23145)  

**Abstract**: Large language models (LLMs) have achieved state-of-the-art performance in natural language processing; however, their high computational cost remains a major bottleneck. In this study, we target computational efficiency by focusing on a matrix multiplication free language model (MatMul-free LM) and further reducing the training cost through an architecture inspired by reservoir computing. Specifically, we partially fix and share the weights of selected layers in the MatMul-free LM and insert reservoir layers to obtain rich dynamic representations without additional training overhead. Additionally, several operations are combined to reduce memory accesses. Experimental results show that the proposed architecture reduces the number of parameters by up to 19%, training time by 9.9%, and inference time by 8.0%, while maintaining comparable performance to the baseline model. 

---
# TabiBERT: A Large-Scale ModernBERT Foundation Model and Unified Benchmarking Framework for Turkish 

**Authors**: Melikşah Türker, A. Ebrar Kızıloğlu, Onur Güngör, Susan Üsküdarlı  

**Link**: [PDF](https://arxiv.org/pdf/2512.23065)  

**Abstract**: Since the inception of BERT, encoder-only Transformers have evolved significantly in computational efficiency, training stability, and long-context modeling. ModernBERT consolidates these advances by integrating Rotary Positional Embeddings (RoPE), FlashAttention, and refined normalization. Despite these developments, Turkish NLP lacks a monolingual encoder trained from scratch incorporating such modern architectural paradigms. This work introduces TabiBERT, a monolingual Turkish encoder based on ModernBERT architecture trained from scratch on a large, curated corpus. TabiBERT is pre-trained on one trillion tokens sampled from an 84.88B token multi-domain corpus: web text (73%), scientific publications (20%), source code (6%), and mathematical content (0.3%). The model supports 8,192-token context length (16x original BERT), achieves up to 2.65x inference speedup, and reduces GPU memory consumption, enabling larger batch sizes. We introduce TabiBench with 28 datasets across eight task categories with standardized splits and protocols, evaluated using GLUE-style macro-averaging. TabiBERT attains 77.58 on TabiBench, outperforming BERTurk by 1.62 points and establishing state-of-the-art on five of eight categories: question answering (+9.55), code retrieval (+2.41), and document retrieval (+0.60). Compared with task-specific prior best results, including specialized models like TurkishBERTweet, TabiBERT achieves +1.47 average improvement, indicating robust cross-domain generalization. We release model weights, training configurations, and evaluation code for transparent, reproducible Turkish encoder research. 

---
# Accelerating Language Model Workflows with Prompt Choreography 

**Authors**: TJ Bai, Jason Eisner  

**Link**: [PDF](https://arxiv.org/pdf/2512.23049)  

**Abstract**: Large language models are increasingly deployed in multi-agent workflows. We introduce Prompt Choreography, a framework that efficiently executes LLM workflows by maintaining a dynamic, global KV cache. Each LLM call can attend to an arbitrary, reordered subset of previously encoded messages. Parallel calls are supported. Though caching messages' encodings sometimes gives different results from re-encoding them in a new context, we show in diverse settings that fine-tuning the LLM to work with the cache can help it mimic the original results. Prompt Choreography significantly reduces per-message latency (2.0--6.2$\times$ faster time-to-first-token) and achieves substantial end-to-end speedups ($>$2.2$\times$) in some workflows dominated by redundant computation. 

---
# Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization 

**Authors**: Kerem Zaman, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2512.23032)  

**Abstract**: Recent work, using the Biasing Features metric, labels a CoT as unfaithful if it omits a prompt-injected hint that affected the prediction. We argue this metric confuses unfaithfulness with incompleteness, the lossy compression needed to turn distributed transformer computation into a linear natural language narrative. On multi-hop reasoning tasks with Llama-3 and Gemma-3, many CoTs flagged as unfaithful by Biasing Features are judged faithful by other metrics, exceeding 50% in some models. With a new faithful@k metric, we show that larger inference-time token budgets greatly increase hint verbalization (up to 90% in some settings), suggesting much apparent unfaithfulness is due to tight token limits. Using Causal Mediation Analysis, we further show that even non-verbalized hints can causally mediate prediction changes through the CoT. We therefore caution against relying solely on hint-based evaluations and advocate a broader interpretability toolkit, including causal mediation and corruption-based metrics. 

---
# LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models 

**Authors**: Wenxuan Xu, Arvind Pillai, Subigya Nepal, Amanda C Collins, Daniel M Mackin, Michael V Heinz, Tess Z Griffin, Nicholas C Jacobson, Andrew Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2512.23025)  

**Abstract**: Multimodal health sensing offers rich behavioral signals for assessing mental health, yet translating these numerical time-series measurements into natural language remains challenging. Current LLMs cannot natively ingest long-duration sensor streams, and paired sensor-text datasets are scarce. To address these challenges, we introduce LENS, a framework that aligns multimodal sensing data with language models to generate clinically grounded mental-health narratives. LENS first constructs a large-scale dataset by transforming Ecological Momentary Assessment (EMA) responses related to depression and anxiety symptoms into natural-language descriptions, yielding over 100,000 sensor-text QA pairs from 258 participants. To enable native time-series integration, we train a patch-level encoder that projects raw sensor signals directly into an LLM's representation space. Our results show that LENS outperforms strong baselines on standard NLP metrics and task-specific measures of symptom-severity accuracy. A user study with 13 mental-health professionals further indicates that LENS-produced narratives are comprehensive and clinically meaningful. Ultimately, our approach advances LLMs as interfaces for health sensing, providing a scalable path toward models that can reason over raw behavioral signals and support downstream clinical decision-making. 

---
# Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping 

**Authors**: Tao Yu, Yongqi An, Kuan Zhu, Guibo Zhu, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23014)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive performance across natural language tasks but incur substantial computational and storage costs due to their scale. Post-training structured pruning offers an efficient solution. However, when few-shot calibration sets fail to adequately reflect the pretraining data distribution, existing methods exhibit limited generalization to downstream tasks. To address this issue, we propose Function-Aware Neuron Grouping (FANG), a post-training pruning framework that alleviates calibration bias by identifying and preserving neurons critical to specific function. FANG groups neurons with similar function based on the type of semantic context they process and prunes each group independently. During importance estimation within each group, tokens that strongly correlate with the functional role of the neuron group are given higher weighting. Additionally, FANG also preserves neurons that contribute across multiple context types. To achieve a better trade-off between sparsity and performance, it allocates sparsity to each block adaptively based on its functional complexity. Experiments show that FANG improves downstream accuracy while preserving language modeling performance. It achieves the state-of-the-art (SOTA) results when combined with FLAP and OBC, two representative pruning methods. Specifically, FANG outperforms FLAP and OBC by 1.5%--8.5% in average accuracy under 30% and 40% sparsity. 

---
# Prompt engineering does not universally improve Large Language Model performance across clinical decision-making tasks 

**Authors**: Mengdi Chai, Ali R. Zomorrodi  

**Link**: [PDF](https://arxiv.org/pdf/2512.22966)  

**Abstract**: Large Language Models (LLMs) have demonstrated promise in medical knowledge assessments, yet their practical utility in real-world clinical decision-making remains underexplored. In this study, we evaluated the performance of three state-of-the-art LLMs-ChatGPT-4o, Gemini 1.5 Pro, and LIama 3.3 70B-in clinical decision support across the entire clinical reasoning workflow of a typical patient encounter. Using 36 case studies, we first assessed LLM's out-of-the-box performance across five key sequential clinical decision-making tasks under two temperature settings (default vs. zero): differential diagnosis, essential immediate steps, relevant diagnostic testing, final diagnosis, and treatment recommendation. All models showed high variability by task, achieving near-perfect accuracy in final diagnosis, poor performance in relevant diagnostic testing, and moderate performance in remaining tasks. Furthermore, ChatGPT performed better under the zero temperature, whereas LIama showed stronger performance under the default temperature. Next, we assessed whether prompt engineering could enhance LLM performance by applying variations of the MedPrompt framework, incorporating targeted and random dynamic few-shot learning. The results demonstrate that prompt engineering is not a one-size-fit-all solution. While it significantly improved the performance on the task with lowest baseline accuracy (relevant diagnostic testing), it was counterproductive for others. Another key finding was that the targeted dynamic few-shot prompting did not consistently outperform random selection, indicating that the presumed benefits of closely matched examples may be counterbalanced by loss of broader contextual diversity. These findings suggest that the impact of prompt engineering is highly model and task-dependent, highlighting the need for tailored, context-aware strategies for integrating LLMs into healthcare. 

---
# Diversity or Precision? A Deep Dive into Next Token Prediction 

**Authors**: Haoyuan Wu, Hai Wang, Jiajia Wu, Jinxiang Ou, Keyao Wang, Weile Chen, Zihao Zheng, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22955)  

**Abstract**: Recent advancements have shown that reinforcement learning (RL) can substantially improve the reasoning abilities of large language models (LLMs). The effectiveness of such RL training, however, depends critically on the exploration space defined by the pre-trained model's token-output distribution. In this paper, we revisit the standard cross-entropy loss, interpreting it as a specific instance of policy gradient optimization applied within a single-step episode. To systematically study how the pre-trained distribution shapes the exploration potential for subsequent RL, we propose a generalized pre-training objective that adapts on-policy RL principles to supervised learning. By framing next-token prediction as a stochastic decision process, we introduce a reward-shaping strategy that explicitly balances diversity and precision. Our method employs a positive reward scaling factor to control probability concentration on ground-truth tokens and a rank-aware mechanism that treats high-ranking and low-ranking negative tokens asymmetrically. This allows us to reshape the pre-trained token-output distribution and investigate how to provide a more favorable exploration space for RL, ultimately enhancing end-to-end reasoning performance. Contrary to the intuition that higher distribution entropy facilitates effective exploration, we find that imposing a precision-oriented prior yields a superior exploration space for RL. 

---
# AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning 

**Authors**: Shihao Cai, Runnan Fang, Jialong Wu, Baixuan Li, Xinyu Wang, Yong Jiang, Liangcai Su, Liwen Zhang, Wenbiao Yin, Zhen Zhang, Fuli Feng, Pengjun Xie, Xiaobin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22857)  

**Abstract**: Conducting reinforcement learning (RL) in simulated environments offers a cost-effective and highly scalable way to enhance language-based agents. However, previous work has been limited to semi-automated environment synthesis or tasks lacking sufficient difficulty, offering little breadth or depth. In addition, the instability of simulated users integrated into these environments, along with the heterogeneity across simulated environments, poses further challenges for agentic RL. In this work, we propose: (1) a unified pipeline for automated and scalable synthesis of simulated environments associated with high-difficulty but easily verifiable tasks; and (2) an environment level RL algorithm that not only effectively mitigates user instability but also performs advantage estimation at the environment level, thereby improving training efficiency and stability. Comprehensive evaluations on agentic benchmarks, including tau-bench, tau2-Bench, and VitaBench, validate the effectiveness of our proposed method. Further in-depth analyses underscore its out-of-domain generalization. 

---
# NepEMO: A Multi-Label Emotion and Sentiment Analysis on Nepali Reddit with Linguistic Insights and Temporal Trends 

**Authors**: Sameer Sitoula, Tej Bahadur Shahi, Laxmi Prasad Bhatt, Anisha Pokhrel, Arjun Neupane  

**Link**: [PDF](https://arxiv.org/pdf/2512.22823)  

**Abstract**: Social media (SM) platforms (e.g. Facebook, Twitter, and Reddit) are increasingly leveraged to share opinions and emotions, specifically during challenging events, such as natural disasters, pandemics, and political elections, and joyful occasions like festivals and celebrations. Among the SM platforms, Reddit provides a unique space for its users to anonymously express their experiences and thoughts on sensitive issues such as health and daily life. In this work, we present a novel dataset, called NepEMO, for multi-label emotion (MLE) and sentiment classification (SC) on the Nepali subreddit post. We curate and build a manually annotated dataset of 4,462 posts (January 2019- June 2025) written in English, Romanised Nepali and Devanagari script for five emotions (fear, anger, sadness, joy, and depression) and three sentiment classes (positive, negative, and neutral). We perform a detailed analysis of posts to capture linguistic insights, including emotion trends, co-occurrence of emotions, sentiment-specific n-grams, and topic modelling using Latent Dirichlet Allocation and TF-IDF keyword extraction. Finally, we compare various traditional machine learning (ML), deep learning (DL), and transformer models for MLE and SC tasks. The result shows that transformer models consistently outperform the ML and DL models for both tasks. 

---
# CNSight: Evaluation of Clinical Note Segmentation Tools 

**Authors**: Risha Surana, Adrian Law, Sunwoo Kim, Rishab Sridhar, Angxiao Han, Peiyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2512.22795)  

**Abstract**: Clinical notes are often stored in unstructured or semi-structured formats after extraction from electronic medical record (EMR) systems, which complicates their use for secondary analysis and downstream clinical applications. Reliable identification of section boundaries is a key step toward structuring these notes, as sections such as history of present illness, medications, and discharge instructions each provide distinct clinical contexts. In this work, we evaluate rule-based baselines, domain-specific transformer models, and large language models for clinical note segmentation using a curated dataset of 1,000 notes from MIMIC-IV. Our experiments show that large API-based models achieve the best overall performance, with GPT-5-mini reaching a best average F1 of 72.4 across sentence-level and freetext segmentation. Lightweight baselines remain competitive on structured sentence-level tasks but falter on unstructured freetext. Our results provide guidance for method selection and lay the groundwork for downstream tasks such as information extraction, cohort identification, and automated summarization. 

---
# Fake News Classification in Urdu: A Domain Adaptation Approach for a Low-Resource Language 

**Authors**: Muhammad Zain Ali, Bernhard Pfahringer, Tony Smith  

**Link**: [PDF](https://arxiv.org/pdf/2512.22778)  

**Abstract**: Misinformation on social media is a widely acknowledged issue, and researchers worldwide are actively engaged in its detection. However, low-resource languages such as Urdu have received limited attention in this domain. An obvious approach is to utilize a multilingual pretrained language model and fine-tune it for a downstream classification task, such as misinformation detection. However, these models struggle with domain-specific terms, leading to suboptimal performance. To address this, we investigate the effectiveness of domain adaptation before fine-tuning for fake news classification in Urdu, employing a staged training approach to optimize model generalization. We evaluate two widely used multilingual models, XLM-RoBERTa and mBERT, and apply domain-adaptive pretraining using a publicly available Urdu news corpus. Experiments on four publicly available Urdu fake news datasets show that domain-adapted XLM-R consistently outperforms its vanilla counterpart, while domain-adapted mBERT exhibits mixed results. 

---
# Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis 

**Authors**: Dongning Rao, Yunbiao Zeng, Zhihua Jiang, Jujian Lv  

**Link**: [PDF](https://arxiv.org/pdf/2512.22741)  

**Abstract**: Human-interaction-involved applications underscore the need for Multi-modal Sentiment Analysis (MSA). Although many approaches have been proposed to address the subtle emotions in different modalities, the power of explanations and temporal alignments is still underexplored. Thus, this paper proposes the Text-routed sparse mixture-of-Experts model with eXplanation and Temporal alignment for MSA (TEXT). TEXT first augments explanations for MSA via Multi-modal Large Language Models (MLLM), and then novelly aligns the epresentations of audio and video through a temporality-oriented neural network block. TEXT aligns different modalities with explanations and facilitates a new text-routed sparse mixture-of-experts with gate fusion. Our temporal alignment block merges the benefits of Mamba and temporal cross-attention. As a result, TEXT achieves the best performance cross four datasets among all tested models, including three recently proposed approaches and three MLLMs. TEXT wins on at least four metrics out of all six metrics. For example, TEXT decreases the mean absolute error to 0.353 on the CH-SIMS dataset, which signifies a 13.5% decrement compared with recently proposed approaches. 

---
# Harnessing Large Language Models for Biomedical Named Entity Recognition 

**Authors**: Jian Chen, Leilei Su, Cong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.22738)  

**Abstract**: Background and Objective: Biomedical Named Entity Recognition (BioNER) is a foundational task in medical informatics, crucial for downstream applications like drug discovery and clinical trial matching. However, adapting general-domain Large Language Models (LLMs) to this task is often hampered by their lack of domain-specific knowledge and the performance degradation caused by low-quality training data. To address these challenges, we introduce BioSelectTune, a highly efficient, data-centric framework for fine-tuning LLMs that prioritizes data quality over quantity. Methods and Results: BioSelectTune reformulates BioNER as a structured JSON generation task and leverages our novel Hybrid Superfiltering strategy, a weak-to-strong data curation method that uses a homologous weak model to distill a compact, high-impact training dataset. Conclusions: Through extensive experiments, we demonstrate that BioSelectTune achieves state-of-the-art (SOTA) performance across multiple BioNER benchmarks. Notably, our model, trained on only 50% of the curated positive data, not only surpasses the fully-trained baseline but also outperforms powerful domain-specialized models like BioMedBERT. 

---
# WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference 

**Authors**: Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.22737)  

**Abstract**: Autoregressive (AR) generation is the standard decoding paradigm for Large Language Models (LLMs), but its token-by-token nature limits parallelism at inference time. Diffusion Language Models (DLLMs) offer parallel decoding by recovering multiple masked tokens per step; however, in practice they often fail to translate this parallelism into deployment speed gains over optimized AR engines (e.g., vLLM). A key reason is that many DLLMs rely on bidirectional attention, which breaks standard prefix KV caching and forces repeated contextualization, undermining efficiency. We propose WeDLM, a diffusion decoding framework built entirely on standard causal attention to make parallel generation prefix-cache friendly. The core idea is to let each masked position condition on all currently observed tokens while keeping a strict causal mask, achieved by Topological Reordering that moves observed tokens to the physical prefix while preserving their logical positions. Building on this property, we introduce a streaming decoding procedure that continuously commits confident tokens into a growing left-to-right prefix and maintains a fixed parallel workload, avoiding the stop-and-wait behavior common in block diffusion methods. Experiments show that WeDLM preserves the quality of strong AR backbones while delivering substantial speedups, approaching 3x on challenging reasoning benchmarks and up to 10x in low-entropy generation regimes; critically, our comparisons are against AR baselines served by vLLM under matched deployment settings, demonstrating that diffusion-style decoding can outperform an optimized AR engine in practice. 

---
# Data Augmentation for Classification of Negative Pregnancy Outcomes in Imbalanced Data 

**Authors**: Md Badsha Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2512.22732)  

**Abstract**: Infant mortality remains a significant public health concern in the United States, with birth defects identified as a leading cause. Despite ongoing efforts to understand the causes of negative pregnancy outcomes like miscarriage, stillbirths, birth defects, and premature birth, there is still a need for more comprehensive research and strategies for intervention. This paper introduces a novel approach that uses publicly available social media data, especially from platforms like Twitter, to enhance current datasets for studying negative pregnancy outcomes through observational research. The inherent challenges in utilizing social media data, including imbalance, noise, and lack of structure, necessitate robust preprocessing techniques and data augmentation strategies. By constructing a natural language processing (NLP) pipeline, we aim to automatically identify women sharing their pregnancy experiences, categorizing them based on reported outcomes. Women reporting full gestation and normal birth weight will be classified as positive cases, while those reporting negative pregnancy outcomes will be identified as negative cases. Furthermore, this study offers potential applications in assessing the causal impact of specific interventions, treatments, or prenatal exposures on maternal and fetal health outcomes. Additionally, it provides a framework for future health studies involving pregnant cohorts and comparator groups. In a broader context, our research showcases the viability of social media data as an adjunctive resource in epidemiological investigations about pregnancy outcomes. 

---
# Mitigating Social Desirability Bias in Random Silicon Sampling 

**Authors**: Sashank Chapala, Maksym Mironov, Songgaojun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2512.22725)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate population responses, a method known as ``Silicon Sampling''. However, responses to socially sensitive questions frequently exhibit Social Desirability Bias (SDB), diverging from real human data toward socially acceptable answers. Existing studies on social desirability bias in LLM-based sampling remain limited. In this work, we investigate whether minimal, psychologically grounded prompt wording can mitigate this bias and improve alignment between silicon and human samples. We conducted a study using data from the American National Election Study (ANES) on three LLMs from two model families: the open-source Llama-3.1 series and GPT-4.1-mini. We first replicate a baseline silicon sampling study, confirming the persistent Social Desirability Bias. We then test four prompt-based mitigation methods: \emph{reformulated} (neutral, third-person phrasing), \emph{reverse-coded} (semantic inversion), and two meta-instructions, \emph{priming} and \emph{preamble}, respectively encouraging analytics and sincerity. Alignment with ANES is evaluated using Jensen-Shannon Divergence with bootstrap confidence intervals. Our results demonstrate that reformulated prompts most effectively improve alignment by reducing distribution concentration on socially acceptable answers and achieving distributions closer to ANES. Reverse-coding produced mixed results across eligible items, while the Priming and Preamble encouraged response uniformity and showed no systematic benefit for bias mitigation. Our findings validate the efficacy of prompt-based framing controls in mitigating inherent Social Desirability Bias in LLMs, providing a practical path toward more representative silicon samples. 

---
# Beg to Differ: Understanding Reasoning-Answer Misalignment Across Languages 

**Authors**: Anaelia Ovalle, Candace Ross, Sebastian Ruder, Adina Williams, Karen Ullrich, Mark Ibrahim, Levent Sagun  

**Link**: [PDF](https://arxiv.org/pdf/2512.22712)  

**Abstract**: Large language models demonstrate strong reasoning capabilities through chain-of-thought prompting, but whether this reasoning quality transfers across languages remains underexplored. We introduce a human-validated framework to evaluate whether model-generated reasoning traces logically support their conclusions across languages. Analyzing 65k reasoning traces from GlobalMMLU questions across 6 languages and 6 frontier models, we uncover a critical blind spot: while models achieve high task accuracy, their reasoning can fail to support their conclusions. Reasoning traces in non-Latin scripts show at least twice as much misalignment between their reasoning and conclusions than those in Latin scripts. We develop an error taxonomy through human annotation to characterize these failures, finding they stem primarily from evidential errors (unsupported claims, ambiguous facts) followed by illogical reasoning steps. Our findings demonstrate that current multilingual evaluation practices provide an incomplete picture of model reasoning capabilities and highlight the need for reasoning-aware evaluation frameworks. 

---
# GHaLIB: A Multilingual Framework for Hope Speech Detection in Low-Resource Languages 

**Authors**: Ahmed Abdullah, Sana Fatima, Haroon Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2512.22705)  

**Abstract**: Hope speech has been relatively underrepresented in Natural Language Processing (NLP). Current studies are largely focused on English, which has resulted in a lack of resources for low-resource languages such as Urdu. As a result, the creation of tools that facilitate positive online communication remains limited. Although transformer-based architectures have proven to be effective in detecting hate and offensive speech, little has been done to apply them to hope speech or, more generally, to test them across a variety of linguistic settings. This paper presents a multilingual framework for hope speech detection with a focus on Urdu. Using pretrained transformer models such as XLM-RoBERTa, mBERT, EuroBERT, and UrduBERT, we apply simple preprocessing and train classifiers for improved results. Evaluations on the PolyHope-M 2025 benchmark demonstrate strong performance, achieving F1-scores of 95.2% for Urdu binary classification and 65.2% for Urdu multi-class classification, with similarly competitive results in Spanish, German, and English. These results highlight the possibility of implementing existing multilingual models in low-resource environments, thus making it easier to identify hope speech and helping to build a more constructive digital discourse. 

---
# Conformal Prediction Sets for Next-Token Prediction in Large Language Models: Balancing Coverage Guarantees with Set Efficiency 

**Authors**: Yoshith Roy Kotla, Varshith Roy Kotla  

**Link**: [PDF](https://arxiv.org/pdf/2512.22682)  

**Abstract**: Deploying large language models (LLMs) in high-stakes domains requires rigorous uncertainty quantification, yet standard softmax probabilities are often poorly calibrated. We present a systematic study of Adaptive Prediction Sets (APS) applied to next-token prediction in transformer-based models with large vocabularies (greater than 250,000 tokens). Our central contribution is the identification of a coverage-efficiency tradeoff: while naive conformal prediction achieves valid coverage, it produces prediction sets of hundreds of tokens, rendering them uninformative. We propose Vocabulary-Aware Conformal Prediction (VACP), a framework that leverages semantic masking and temperature-adjusted scoring to reduce the effective prediction space while provably maintaining marginal coverage. Experiments on Gemma-2B using SQUAD and WikiText benchmarks demonstrate that VACP achieves 89.7 percent empirical coverage (90 percent target) while reducing the mean prediction set size from 847 tokens to 4.3 tokens -- a 197x improvement in efficiency. We provide a theoretical analysis of vocabulary reduction and release our implementation for reproducibility. 

---
# Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2 

**Authors**: Pere Martra  

**Link**: [PDF](https://arxiv.org/pdf/2512.22671)  

**Abstract**: Structured width pruning of GLU-MLP layers, guided by the Maximum Absolute Weight (MAW) criterion, reveals a systematic dichotomy in how reducing the expansion ratio affects different model capabilities. While performance on tasks relying on parametric knowledge (e.g., MMLU, GSM8K) and perplexity metrics degrades predictably, instruction-following capabilities improve substantially (+46% to +75% in IFEval for Llama-3.2-1B and 3B models), and multi-step reasoning remains robust (MUSR). This pattern challenges the prevailing assumption that pruning induces uniform degradation. We evaluated seven expansion ratio configurations using comprehensive benchmarks assessing factual knowledge, mathematical reasoning, language comprehension, instruction-following, and truthfulness. Our analysis identifies the expansion ratio as a critical architectural parameter that selectively modulates cognitive capabilities, rather than merely serving as a compression metric. We provide the first systematic characterization of this selective preservation phenomenon. Notably, we document a robust inverse correlation (r = -0.864, p = 0.012 in Llama-3B) between factual knowledge capacity (MMLU) and truthfulness metrics (TruthfulQA-MC2): as knowledge degrades, the model's ability to discriminate misconceptions improves consistently. This connects two previously distinct research areas, demonstrating that MAW-guided width pruning acts as a selective filter, reducing parametric knowledge while preserving or enhancing behavioral alignment. Additionally, we quantify context-dependent efficiency trade-offs: pruned configurations achieve up to 23% reduction in energy consumption (J/token) but incur penalties in single-request latency, whereas batch processing workloads benefit uniformly. 

---
# Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs 

**Authors**: Hadi Mohammadi, Tamas Kozak, Anastasia Giachanou  

**Link**: [PDF](https://arxiv.org/pdf/2512.22631)  

**Abstract**: Chain-of-thought (CoT) reasoning has emerged as a powerful technique for improving the problem-solving capabilities of large language models (LLMs), particularly for tasks requiring multi-step reasoning. However, recent studies show that CoT explanations often fail to reflect the model's actual reasoning process, as models may produce coherent yet misleading justifications or modify answers without acknowledging external cues. Such discrepancies undermine the reliability of CoT-based methods for safety supervision and alignment monitoring, as models can generate plausible but deceptive rationales for incorrect answers. To better understand this limitation, we evaluate two optimization methods, Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO), in their ability to improve CoT faithfulness. Our experiments show that GRPO achieves higher performance than DPO in larger models, with the Qwen2.5-14B-Instruct model attaining the best results across all evaluation metrics. Both approaches exhibit positive correlations between model size and performance, but GRPO shows greater potential for improving faithfulness metrics, albeit with less stable behavior at smaller scales. These results suggest that GRPO offers a promising direction for developing more transparent and trustworthy reasoning in LLMs. 

---
# On the Role of Discreteness in Diffusion LLMs 

**Authors**: Ziqi Jin, Bin Wang, Xiang Lin, Lidong Bing, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.22630)  

**Abstract**: Diffusion models offer appealing properties for language generation, such as parallel decoding and iterative refinement, but the discrete and highly structured nature of text challenges the direct application of diffusion principles. In this paper, we revisit diffusion language modeling from the view of diffusion process and language modeling, and outline five properties that separate diffusion mechanics from language-specific requirements. We first categorize existing approaches into continuous diffusion in embedding space and discrete diffusion over tokens. We then show that each satisfies only part of the five essential properties and therefore reflects a structural trade-off. Through analyses of recent large diffusion language models, we identify two central issues: (i) uniform corruption does not respect how information is distributed across positions, and (ii) token-wise marginal training cannot capture multi-token dependencies during parallel decoding. These observations motivate diffusion processes that align more closely with the structure of text, and encourage future work toward more coherent diffusion language models. 

---
# M2G-Eval: Enhancing and Evaluating Multi-granularity Multilingual Code Generation 

**Authors**: Fanglin Xu, Wei Zhang, Jian Yang, Guo Chen, Aishan Liu, Zhoujun Li, Xianglong Liu, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2512.22628)  

**Abstract**: The rapid advancement of code large language models (LLMs) has sparked significant research interest in systematically evaluating their code generation capabilities, yet existing benchmarks predominantly assess models at a single structural granularity and focus on limited programming languages, obscuring fine-grained capability variations across different code scopes and multilingual scenarios. We introduce M2G-Eval, a multi-granularity, multilingual framework for evaluating code generation in large language models (LLMs) across four levels: Class, Function, Block, and Line. Spanning 18 programming languages, M2G-Eval includes 17K+ training tasks and 1,286 human-annotated, contamination-controlled test instances. We develop M2G-Eval-Coder models by training Qwen3-8B with supervised fine-tuning and Group Relative Policy Optimization. Evaluating 30 models (28 state-of-the-art LLMs plus our two M2G-Eval-Coder variants) reveals three main findings: (1) an apparent difficulty hierarchy, with Line-level tasks easiest and Class-level most challenging; (2) widening performance gaps between full- and partial-granularity languages as task complexity increases; and (3) strong cross-language correlations, suggesting that models learn transferable programming concepts. M2G-Eval enables fine-grained diagnosis of code generation capabilities and highlights persistent challenges in synthesizing complex, long-form code. 

---
# Chain-of-thought Reviewing and Correction for Time Series Question Answering 

**Authors**: Chen Su, Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.22627)  

**Abstract**: With the advancement of large language models (LLMs), diverse time series analysis tasks are reformulated as time series question answering (TSQA) through a unified natural language interface. However, existing LLM-based approaches largely adopt general natural language processing techniques and are prone to reasoning errors when handling complex numerical sequences. Different from purely textual tasks, time series data are inherently verifiable, enabling consistency checking between reasoning steps and the original input. Motivated by this property, we propose T3LLM, which performs multi-step reasoning with an explicit correction mechanism for time series question answering. The T3LLM framework consists of three LLMs, namely, a worker, a reviewer, and a student, that are responsible for generation, review, and reasoning learning, respectively. Within this framework, the worker generates step-wise chains of thought (CoT) under structured prompts, while the reviewer inspects the reasoning, identifies erroneous steps, and provides corrective comments. The collaboratively generated corrected CoT are used to fine-tune the student model, internalizing multi-step reasoning and self-correction into its parameters. Experiments on multiple real-world TSQA benchmarks demonstrate that T3LLM achieves state-of-the-art performance over strong LLM-based baselines. 

---
# Structured Prompting and LLM Ensembling for Multimodal Conversational Aspect-based Sentiment Analysis 

**Authors**: Zhiqiang Gao, Shihao Gao, Zixing Zhang, Yihao Guo, Hongyu Chen, Jing Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.22603)  

**Abstract**: Understanding sentiment in multimodal conversations is a complex yet crucial challenge toward building emotionally intelligent AI systems. The Multimodal Conversational Aspect-based Sentiment Analysis (MCABSA) Challenge invited participants to tackle two demanding subtasks: (1) extracting a comprehensive sentiment sextuple, including holder, target, aspect, opinion, sentiment, and rationale from multi-speaker dialogues, and (2) detecting sentiment flipping, which detects dynamic sentiment shifts and their underlying triggers. For Subtask-I, in the present paper, we designed a structured prompting pipeline that guided large language models (LLMs) to sequentially extract sentiment components with refined contextual understanding. For Subtask-II, we further leveraged the complementary strengths of three LLMs through ensembling to robustly identify sentiment transitions and their triggers. Our system achieved a 47.38% average score on Subtask-I and a 74.12% exact match F1 on Subtask-II, showing the effectiveness of step-wise refinement and ensemble strategies in rich, multimodal sentiment analysis tasks. 

---
# Learning When Not to Attend Globally 

**Authors**: Xuan Luo, Kailai Zhang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.22562)  

**Abstract**: When reading books, humans focus primarily on the current page, flipping back to recap prior context only when necessary. Similarly, we demonstrate that Large Language Models (LLMs) can learn to dynamically determine when to attend to global context. We propose All-or-Here Attention (AHA), which utilizes a binary router per attention head to dynamically toggle between full attention and local sliding window attention for each token. Our results indicate that with a window size of 256 tokens, up to 93\% of the original full attention operations can be replaced by sliding window attention without performance loss. Furthermore, by evaluating AHA across various window sizes, we identify a long-tail distribution in context dependency, where the necessity for full attention decays rapidly as the local window expands. By decoupling local processing from global access, AHA reveals that full attention is largely redundant, and that efficient inference requires only on-demand access to the global context. 

---
# ManchuTTS: Towards High-Quality Manchu Speech Synthesis via Flow Matching and Hierarchical Text Representation 

**Authors**: Suhua Wang, Zifan Wang, Xiaoxin Sun, D. J. Wang, Zhanbo Liu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.22491)  

**Abstract**: As an endangered language, Manchu presents unique challenges for speech synthesis, including severe data scarcity and strong phonological agglutination. This paper proposes ManchuTTS(Manchu Text to Speech), a novel approach tailored to Manchu's linguistic characteristics. To handle agglutination, this method designs a three-tier text representation (phoneme, syllable, prosodic) and a cross-modal hierarchical attention mechanism for multi-granular alignment. The synthesis model integrates deep convolutional networks with a flow-matching Transformer, enabling efficient, non-autoregressive generation. This method further introduce a hierarchical contrastive loss to guide structured acoustic-linguistic correspondence. To address low-resource constraints, This method construct the first Manchu TTS dataset and employ a data augmentation strategy. Experiments demonstrate that ManchuTTS attains a MOS of 4.52 using a 5.2-hour training subset derived from our full 6.24-hour annotated corpus, outperforming all baseline models by a notable margin. Ablations confirm hierarchical guidance improves agglutinative word pronunciation accuracy (AWPA) by 31% and prosodic naturalness by 27%. 

---
# Constituency Structure over Eojeol in Korean Treebanks 

**Authors**: Jungyeul Park, Chulwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.22487)  

**Abstract**: The design of Korean constituency treebanks raises a fundamental representational question concerning the choice of terminal units. Although Korean words are morphologically complex, treating morphemes as constituency terminals conflates word internal morphology with phrase level syntactic structure and creates mismatches with eojeol based dependency resources. This paper argues for an eojeol based constituency representation, with morphological segmentation and fine grained part of speech information encoded in a separate, non constituent layer. A comparative analysis shows that, under explicit normalization assumptions, the Sejong and Penn Korean treebanks can be treated as representationally equivalent at the eojeol based constituency level. Building on this result, we outline an eojeol based annotation scheme that preserves interpretable constituency and supports cross treebank comparison and constituency dependency conversion. 

---
# Exploring the Vertical-Domain Reasoning Capabilities of Large Language Models 

**Authors**: Jie Zhou, Xin Chen, Jie Zhang, Zhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.22443)  

**Abstract**: Large Language Models (LLMs) are reshaping learning paradigms, cognitive processes, and research methodologies across a wide range of domains. Integrating LLMs with professional fields and redefining the relationship between LLMs and domain-specific applications has become a critical challenge for promoting enterprise digital transformation and broader social development. To effectively integrate LLMs into the accounting domain, it is essential to understand their domain-specific reasoning capabilities. This study introduces the concept of vertical-domain accounting reasoning and establishes evaluation criteria by analyzing the training data characteristics of representative GLM-series models. These criteria provide a foundation for subsequent research on reasoning paradigms and offer benchmarks for improving accounting reasoning performance. Based on this framework, we evaluate several representative models, including GLM-6B, GLM-130B, GLM-4, and OpenAI GPT-4, on a set of accounting reasoning tasks. Experimental results show that different prompt engineering strategies lead to varying degrees of performance improvement across models, with GPT-4 achieving the strongest accounting reasoning capability. However, current LLMs still fall short of real-world application requirements. In particular, further optimization is needed for deployment in enterprise-level accounting scenarios to fully realize the potential value of LLMs in this domain. 

---
# HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG 

**Authors**: Cattalyya Nuengsigkapian  

**Link**: [PDF](https://arxiv.org/pdf/2512.22442)  

**Abstract**: Retrieval-Augmented Generation (RAG) in open-domain settings faces significant challenges regarding irrelevant information in retrieved documents and the alignment of generated answers with user intent. We present HiFi-RAG (Hierarchical Filtering RAG), the winning closed-source system in the Text-to-Text static evaluation of the MMU-RAGent NeurIPS 2025 Competition. Our approach moves beyond standard embedding-based retrieval via a multi-stage pipeline. We leverage the speed and cost-efficiency of Gemini 2.5 Flash (4-6x cheaper than Pro) for query formulation, hierarchical content filtering, and citation attribution, while reserving the reasoning capabilities of Gemini 2.5 Pro for final answer generation. On the MMU-RAGent validation set, our system outperformed the baseline, improving ROUGE-L to 0.274 (+19.6%) and DeBERTaScore to 0.677 (+6.2%). On Test2025, our custom dataset evaluating questions that require post-cutoff knowledge (post January 2025), HiFi-RAG outperforms the parametric baseline by 57.4% in ROUGE-L and 14.9% in DeBERTaScore. 

---
# Hallucination Detection and Evaluation of Large Language Model 

**Authors**: Chenggong Zhang, Haopeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22416)  

**Abstract**: Hallucinations in Large Language Models (LLMs) pose a significant challenge, generating misleading or unverifiable content that undermines trust and reliability. Existing evaluation methods, such as KnowHalu, employ multi-stage verification but suffer from high computational costs. To address this, we integrate the Hughes Hallucination Evaluation Model (HHEM), a lightweight classification-based framework that operates independently of LLM-based judgments, significantly improving efficiency while maintaining high detection accuracy. We conduct a comparative analysis of hallucination detection methods across various LLMs, evaluating True Positive Rate (TPR), True Negative Rate (TNR), and Accuracy on question-answering (QA) and summarization tasks. Our results show that HHEM reduces evaluation time from 8 hours to 10 minutes, while HHEM with non-fabrication checking achieves the highest accuracy \(82.2\%\) and TPR \(78.9\%\). However, HHEM struggles with localized hallucinations in summarization tasks. To address this, we introduce segment-based retrieval, improving detection by verifying smaller text components. Additionally, our cumulative distribution function (CDF) analysis indicates that larger models (7B-9B parameters) generally exhibit fewer hallucinations, while intermediate-sized models show higher instability. These findings highlight the need for structured evaluation frameworks that balance computational efficiency with robust factual validation, enhancing the reliability of LLM-generated content. 

---
# LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition 

**Authors**: Elsen Ronando, Sozo Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2512.22385)  

**Abstract**: In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar weara-ble sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and $k$-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR. 

---
# Towards Efficient Post-Training via Fourier-Driven Adapter Architectures 

**Authors**: Donggyun Bae, Jongil Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.22378)  

**Abstract**: We propose a novel framework, termed Fourier-Activated Adapter (FAA), for parameter-efficient fine-tuning of large pre-trained language models. By incorporating random Fourier features into lightweight adapter modules, FAA decomposes intermediate representations into complementary low- and high-frequency components, enabling frequency-aware modulation of semantic information. This design allows the model to selectively emphasize informative frequency bands during adaptation while preserving the representational capacity of the frozen backbone. Extensive experiments on GLUE, E2E NLG, and instruction-tuning benchmarks demonstrate that FAA consistently achieves competitive or superior performance compared to existing parameter-efficient fine-tuning methods, while maintaining low computational and memory overhead. Ablation studies further verify the effectiveness of frequency-aware activation and adaptive weighting mechanisms, highlighting FAA as a robust and efficient approach for post-training large language models. 

---
# The Syntax of qulk-clauses in Yemeni Ibbi Arabic: A Minimalist Approach 

**Authors**: Zubaida Mohammed Albadani, Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2512.22376)  

**Abstract**: This study investigates the syntax of qulk-clauses in Yemeni Ibbi Arabic (YIA) within the Minimalist Program. The construction qulk-clause, a morphologically fused form meaning 'I said,' introduces embedded declarative interrogative, and imperative clauses, often eithout complementizer. The central proposal of this paper is that qulk-clauses are biclausal structures in which qulk functions a clause-embedding predicate sec;ecting a dull CP complement. By applying core minimalist operations, viz., Merge, Move, Agree, and Spell-out, the study provides a layered syntactic analysis of qulk-clauses, for illustrating how their derivation proceeds through standard computational steps and post-syntactic processes such as Morphological Merger. The proposal also accounts for dialect-specific features like bipartite negation, cliticization, and CP embedding. The findings offer theoretical contributions to generative syntax, specifically minimalism. The study concludes raising theoretical questions concerning extending the analysis to the addressee-clause kil-k 'you said'. It also provides insights into the possibility of the universality of minimalism. 

---
# SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents 

**Authors**: Shaofei Cai, Yulei Qin, Haojia Lin, Zihan Xu, Gang Li, Yuchen Shi, Zongyi Li, Yong Mao, Siqi Cai, Xiaoyu Tan, Yitao Liang, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.22322)  

**Abstract**: Agentic reinforcement learning (RL) holds great promise for the development of autonomous agents under complex GUI tasks, but its scalability remains severely hampered by the verification of task completion. Existing task verification is treated as a passive, post-hoc process: a verifier (i.e., rule-based scoring script, reward or critic model, and LLM-as-a-Judge) analyzes the agent's entire interaction trajectory to determine if the agent succeeds. Such processing of verbose context that contains irrelevant, noisy history poses challenges to the verification protocols and therefore leads to prohibitive cost and low reliability. To overcome this bottleneck, we propose SmartSnap, a paradigm shift from this passive, post-hoc verification to proactive, in-situ self-verification by the agent itself. We introduce the Self-Verifying Agent, a new type of agent designed with dual missions: to not only complete a task but also to prove its accomplishment with curated snapshot evidences. Guided by our proposed 3C Principles (Completeness, Conciseness, and Creativity), the agent leverages its accessibility to the online environment to perform self-verification on a minimal, decisive set of snapshots. Such evidences are provided as the sole materials for a general LLM-as-a-Judge verifier to determine their validity and relevance. Experiments on mobile tasks across model families and scales demonstrate that our SmartSnap paradigm allows training LLM-driven agents in a scalable manner, bringing performance gains up to 26.08% and 16.66% respectively to 8B and 30B models. The synergizing between solution finding and evidence seeking facilitates the cultivation of efficient, self-verifying agents with competitive performance against DeepSeek V3.1 and Qwen3-235B-A22B. 

---
# Hierarchical Geometry of Cognitive States in Transformer Embedding Spaces 

**Authors**: Sophie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.22227)  

**Abstract**: Recent work has shown that transformer-based language models learn rich geometric structure in their embedding spaces, yet the presence of higher-level cognitive organization within these representations remains underexplored. In this work, we investigate whether sentence embeddings encode a graded, hierarchical structure aligned with human-interpretable cognitive or psychological attributes. We construct a dataset of 480 natural-language sentences annotated with continuous ordinal energy scores and discrete tier labels spanning seven ordered cognitive categories. Using fixed sentence embeddings from multiple transformer models, we evaluate the recoverability of these annotations via linear and shallow nonlinear probes. Across models, both continuous scores and tier labels are reliably decodable, with shallow nonlinear probes providing consistent performance gains over linear probes. Lexical TF-IDF baselines perform substantially worse, indicating that the observed structure is not attributable to surface word statistics alone. Nonparametric permutation tests further confirm that probe performance exceeds chance under label-randomization nulls. Qualitative analyses using UMAP visualizations and confusion matrices reveal smooth low-to-high gradients and predominantly adjacent-tier confusions in embedding space. Taken together, these results provide evidence that transformer embedding spaces exhibit a hierarchical geometric organization aligned with human-defined cognitive attributes, while remaining agnostic to claims of internal awareness or phenomenology. 

---
# Open-Source Multimodal Moxin Models with Moxin-VLM and Moxin-VLA 

**Authors**: Pu Zhao, Xuan Shen, Zhenglun Kong, Yixin Shen, Sung-En Chang, Arash Akbari, Timothy Rupprecht, Lei Lu, Enfu Nan, Changdi Yang, Yumei He, Weiyan Shi, Xingchen Xu, Yu Huang, Wei Jiang, Wei Wang, Yue Chen, Yong He, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22208)  

**Abstract**: Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA and Mistral, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Moxin 7B is introduced as a fully open-source LLM developed in accordance with the Model Openness Framework, which moves beyond the simple sharing of model weights to embrace complete transparency in training, datasets, and implementation detail, thus fostering a more inclusive and collaborative research environment that can sustain a healthy open-source ecosystem. To further equip Moxin with various capabilities in different tasks, we develop three variants based on Moxin, including Moxin-VLM, Moxin-VLA, and Moxin-Chinese, which target the vision-language, vision-language-action, and Chinese capabilities, respectively. Experiments show that our models achieve superior performance in various evaluations. We adopt open-source framework and open data for the training. We release our models, along with the available data and code to derive these models. 

---
# Training AI Co-Scientists Using Rubric Rewards 

**Authors**: Shashwat Goel, Rishi Hazra, Dulhan Jayalath, Timon Willi, Parag Jain, William F. Shen, Ilias Leontiadis, Francesco Barbieri, Yoram Bachrach, Jonas Geiping, Chenxi Whitehouse  

**Link**: [PDF](https://arxiv.org/pdf/2512.23707)  

**Abstract**: AI co-scientists are emerging as a tool to assist human researchers in achieving their research goals. A crucial feature of these AI co-scientists is the ability to generate a research plan given a set of aims and constraints. The plan may be used by researchers for brainstorming, or may even be implemented after further refinement. However, language models currently struggle to generate research plans that follow all constraints and implicit requirements. In this work, we study how to leverage the vast corpus of existing research papers to train language models that generate better research plans. We build a scalable, diverse training corpus by automatically extracting research goals and goal-specific grading rubrics from papers across several domains. We then train models for research plan generation via reinforcement learning with self-grading. A frozen copy of the initial policy acts as the grader during training, with the rubrics creating a generator-verifier gap that enables improvements without external human supervision. To validate this approach, we conduct a study with human experts for machine learning research goals, spanning 225 hours. The experts prefer plans generated by our finetuned Qwen3-30B-A3B model over the initial model for 70% of research goals, and approve 84% of the automatically extracted goal-specific grading rubrics. To assess generality, we also extend our approach to research goals from medical papers, and new arXiv preprints, evaluating with a jury of frontier models. Our finetuning yields 12-22% relative improvements and significant cross-domain generalization, proving effective even in problem settings like medical research where execution feedback is infeasible. Together, these findings demonstrate the potential of a scalable, automated training recipe as a step towards improving general AI co-scientists. 

---
# Web World Models 

**Authors**: Jichen Feng, Yifan Zhang, Chenggong Zhang, Yifu Lu, Shilong Liu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23676)  

**Abstract**: Language agents increasingly require persistent worlds in which they can act, remember, and learn. Existing approaches sit at two extremes: conventional web frameworks provide reliable but fixed contexts backed by databases, while fully generative world models aim for unlimited environments at the expense of controllability and practical engineering. In this work, we introduce the Web World Model (WWM), a middle ground where world state and ``physics'' are implemented in ordinary web code to ensure logical consistency, while large language models generate context, narratives, and high-level decisions on top of this structured latent state. We build a suite of WWMs on a realistic web stack, including an infinite travel atlas grounded in real geography, fictional galaxy explorers, web-scale encyclopedic and narrative worlds, and simulation- and game-like environments. Across these systems, we identify practical design principles for WWMs: separating code-defined rules from model-driven imagination, representing latent state as typed web interfaces, and utilizing deterministic generation to achieve unlimited but structured exploration. Our results suggest that web stacks themselves can serve as a scalable substrate for world models, enabling controllable yet open-ended environments. Project Page: this https URL. 

---
# The Big Three in Marriage Talk: LLM-Assisted Analysis of Moral Ethics and Sentiment on Weibo and Xiaohongshu 

**Authors**: Frank Tian-Fang Ye, Xiaozi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23609)  

**Abstract**: China's marriage registrations have declined dramatically, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. Understanding public attitudes toward marriage requires examining not only emotional sentiment but also the moral reasoning underlying these evaluations. This study analyzed 219,358 marriage-related posts from two major Chinese social media platforms (Sina Weibo and Xiaohongshu) using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral dimensions (Autonomy, Community, Divinity). Results revealed platform differences: Weibo discourse skewed positive, while Xiaohongshu was predominantly neutral. Most posts across both platforms lacked explicit moral framing. However, when moral ethics were invoked, significant associations with sentiment emerged. Posts invoking Autonomy ethics and Community ethics were predominantly negative, whereas Divinity-framed posts tended toward neutral or positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations drive negative marriage attitudes in contemporary China. The study demonstrates LLMs' utility for scaling qualitative analysis and offers insights for developing culturally informed policies addressing marriage decline in Chinese contexts. 

---
# VL-RouterBench: A Benchmark for Vision-Language Model Routing 

**Authors**: Zhehao Huang, Baijiong Lin, Jingyuan Zhang, Jingying Wang, Yuhang Liu, Ning Lu, Tao Li, Xiaolin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23562)  

**Abstract**: Multi-model routing has evolved from an engineering technique into essential infrastructure, yet existing work lacks a systematic, reproducible benchmark for evaluating vision-language models (VLMs). We present VL-RouterBench to assess the overall capability of VLM routing systems systematically. The benchmark is grounded in raw inference and scoring logs from VLMs and constructs quality and cost matrices over sample-model pairs. In scale, VL-RouterBench covers 14 datasets across 3 task groups, totaling 30,540 samples, and includes 15 open-source models and 2 API models, yielding 519,180 sample-model pairs and a total input-output token volume of 34,494,977. The evaluation protocol jointly measures average accuracy, average cost, and throughput, and builds a ranking score from the harmonic mean of normalized cost and accuracy to enable comparison across router configurations and cost budgets. On this benchmark, we evaluate 10 routing methods and baselines and observe a significant routability gain, while the best current routers still show a clear gap to the ideal Oracle, indicating considerable room for improvement in router architecture through finer visual cues and modeling of textual structure. We will open-source the complete data construction and evaluation toolchain to promote comparability, reproducibility, and practical deployment in multimodal routing research. 

---
# Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following 

**Authors**: Kongcheng Zhang, Qi Yao, Shunyu Liu, Wenjian Zhang, Min Cen, Yang Zhou, Wenkai Fang, Yiru Zhao, Baisheng Lai, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.23457)  

**Abstract**: Reinforcement Learning (RL) has shown promise for aligning Large Language Models (LLMs) to follow instructions with various constraints. Despite the encouraging results, RL improvement inevitably relies on sampling successful, high-quality responses; however, the initial model often struggles to generate responses that satisfy all constraints due to its limited capabilities, yielding sparse or indistinguishable rewards that impede learning. In this work, we propose Hindsight instruction Replay (HiR), a novel sample-efficient RL framework for complex instruction following tasks, which employs a select-then-rewrite strategy to replay failed attempts as successes based on the constraints that have been satisfied in hindsight. We perform RL on these replayed samples as well as the original ones, theoretically framing the objective as dual-preference learning at both the instruction- and response-level to enable efficient optimization using only a binary reward signal. Extensive experiments demonstrate that the proposed HiR yields promising results across different instruction following tasks, while requiring less computational budget. Our code and dataset is available at this https URL. 

---
# Theoretical Foundations of Scaling Law in Familial Models 

**Authors**: Huan Song, Qingfei Zhao, Ting Long, Shuyu Tian, Hongjun An, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.23407)  

**Abstract**: Neural scaling laws have become foundational for optimizing large language model (LLM) training, yet they typically assume a single dense model output. This limitation effectively overlooks "Familial models, a transformative paradigm essential for realizing ubiquitous intelligence across heterogeneous device-edge-cloud hierarchies. Transcending static architectures, familial models integrate early exits with relay-style inference to spawn G deployable sub-models from a single shared backbone. In this work, we theoretically and empirically extend the scaling law to capture this "one-run, many-models" paradigm by introducing Granularity (G) as a fundamental scaling variable alongside model size (N) and training tokens (D). To rigorously quantify this relationship, we propose a unified functional form L(N, D, G) and parameterize it using large-scale empirical runs. Specifically, we employ a rigorous IsoFLOP experimental design to strictly isolate architectural impact from computational scale. Across fixed budgets, we systematically sweep model sizes (N) and granularities (G) while dynamically adjusting tokens (D). This approach effectively decouples the marginal cost of granularity from the benefits of scale, ensuring high-fidelity parameterization of our unified scaling law. Our results reveal that the granularity penalty follows a multiplicative power law with an extremely small exponent. Theoretically, this bridges fixed-compute training with dynamic architectures. Practically, it validates the "train once, deploy many" paradigm, demonstrating that deployment flexibility is achievable without compromising the compute-optimality of dense baselines. 

---
# CubeBench: Diagnosing Interactive, Long-Horizon Spatial Reasoning Under Partial Observations 

**Authors**: Huan-ang Gao, Zikang Zhang, Tianwei Luo, Kaisen Yang, Xinzhe Juan, Jiahao Qiu, Tianxing Chen, Bingxiang He, Hao Zhao, Hao Zhou, Shilong Liu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23328)  

**Abstract**: Large Language Model (LLM) agents, while proficient in the digital realm, face a significant gap in physical-world deployment due to the challenge of forming and maintaining a robust spatial mental model. We identify three core cognitive challenges hindering this transition: spatial reasoning, long-horizon state tracking via mental simulation, and active exploration under partial observation. To isolate and evaluate these faculties, we introduce CubeBench, a novel generative benchmark centered on the Rubik's Cube. CubeBench uses a three-tiered diagnostic framework that progressively assesses agent capabilities, from foundational state tracking with full symbolic information to active exploration with only partial visual data. Our experiments on leading LLMs reveal critical limitations, including a uniform 0.00% pass rate on all long-horizon tasks, exposing a fundamental failure in long-term planning. We also propose a diagnostic framework to isolate these cognitive bottlenecks by providing external solver tools. By analyzing the failure modes, we provide key insights to guide the development of more physically-grounded intelligent agents. 

---
# A Note on Hybrid Online Reinforcement and Imitation Learning for LLMs: Formulations and Algorithms 

**Authors**: Yingru Li, Ziniu Li, Jiacai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23097)  

**Abstract**: We present a unified framework for Large Language Model (LLM) fine-tuning that integrates Imitation Learning and Reinforcement Learning. By analyzing the gradient of a composite objective combining trajectory-level KL divergence with task rewards, we derive a natural decomposition into two components: (1) an analytically computable Dense Gradient for token-level imitation, and (2) a Monte Carlo estimated Sparse Gradient for long-horizon reward optimization. The Dense Gradient admits a closed-form logit-level formula, enabling efficient GPU implementation. 

---
# Multimodal Fact-Checking: An Agent-based Approach 

**Authors**: Danni Xu, Shaojing Fan, Xuanang Cheng, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2512.22933)  

**Abstract**: The rapid spread of multimodal misinformation poses a growing challenge for automated fact-checking systems. Existing approaches, including large vision language models (LVLMs) and deep multimodal fusion methods, often fall short due to limited reasoning and shallow evidence utilization. A key bottleneck is the lack of dedicated datasets that provide complete real-world multimodal misinformation instances accompanied by annotated reasoning processes and verifiable evidence. To address this limitation, we introduce RW-Post, a high-quality and explainable dataset for real-world multimodal fact-checking. RW-Post aligns real-world multimodal claims with their original social media posts, preserving the rich contextual information in which the claims are made. In addition, the dataset includes detailed reasoning and explicitly linked evidence, which are derived from human written fact-checking articles via a large language model assisted extraction pipeline, enabling comprehensive verification and explanation. Building upon RW-Post, we propose AgentFact, an agent-based multimodal fact-checking framework designed to emulate the human verification workflow. AgentFact consists of five specialized agents that collaboratively handle key fact-checking subtasks, including strategy planning, high-quality evidence retrieval, visual analysis, reasoning, and explanation generation. These agents are orchestrated through an iterative workflow that alternates between evidence searching and task-aware evidence filtering and reasoning, facilitating strategic decision-making and systematic evidence analysis. Extensive experimental results demonstrate that the synergy between RW-Post and AgentFact substantially improves both the accuracy and interpretability of multimodal fact-checking. 

---
# Debugging Tabular Log as Dynamic Graphs 

**Authors**: Chumeng Liang, Zhanyang Jin, Zahaib Akhtar, Mona Pereira, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2512.22903)  

**Abstract**: Tabular log abstracts objects and events in the real-world system and reports their updates to reflect the change of the system, where one can detect real-world inconsistencies efficiently by debugging corresponding log entries. However, recent advances in processing text-enriched tabular log data overly depend on large language models (LLMs) and other heavy-load models, thus suffering from limited flexibility and scalability. This paper proposes a new framework, GraphLogDebugger, to debug tabular log based on dynamic graphs. By constructing heterogeneous nodes for objects and events and connecting node-wise edges, the framework recovers the system behind the tabular log as an evolving dynamic graph. With the help of our dynamic graph modeling, a simple dynamic Graph Neural Network (GNN) is representative enough to outperform LLMs in debugging tabular log, which is validated by experimental results on real-world log datasets of computer systems and academic papers. 

---
# Scaling Unverifiable Rewards: A Case Study on Visual Insights 

**Authors**: Shuyu Gan, James Mooney, Pan Hao, Renxiang Wang, Mingyi Hong, Qianwen Wang, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22650)  

**Abstract**: Large Language Model (LLM) agents can increasingly automate complex reasoning through Test-Time Scaling (TTS), iterative refinement guided by reward signals. However, many real-world tasks involve multi-stage pipeline whose final outcomes lack verifiable rewards or sufficient data to train robust reward models, making judge-based refinement prone to accumulate error over stages. We propose Selective TTS, a process-based refinement framework that scales inference across different stages in multi-agent pipeline, instead of repeated refinement over time by prior work. By distributing compute across stages and pruning low-quality branches early using process-specific judges, Selective TTS mitigates the judge drift and stabilizes refinement. Grounded in the data science pipeline, we build an end-to-end multi-agent pipeline for generating visually insightful charts and report of given dataset, and design a reliable LLM-based judge model, aligned with human experts (Kendall's {\tau}=0.55). Our proposed selective TTS then improves insight quality under a fixed compute budget, increasing mean scores from 61.64 to 65.86 while reducing variance. We hope our findings serve as the first step toward to scaling complex, open-ended tasks with unverifiable rewards, such as scientific discovery and story generation. 

---
# Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone 

**Authors**: Jiacheng Ye, Shansan Gong, Jiahui Gao, Junming Fan, Shuang Wu, Wei Bi, Haoli Bai, Lifeng Shang, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2512.22615)  

**Abstract**: While autoregressive Large Vision-Language Models (VLMs) have achieved remarkable success, their sequential generation often limits their efficacy in complex visual planning and dynamic robotic control. In this work, we investigate the potential of constructing Vision-Language Models upon diffusion-based large language models (dLLMs) to overcome these limitations. We introduce Dream-VL, an open diffusion-based VLM (dVLM) that achieves state-of-the-art performance among previous dVLMs. Dream-VL is comparable to top-tier AR-based VLMs trained on open data on various benchmarks but exhibits superior potential when applied to visual planning tasks. Building upon Dream-VL, we introduce Dream-VLA, a dLLM-based Vision-Language-Action model (dVLA) developed through continuous pre-training on open robotic datasets. We demonstrate that the natively bidirectional nature of this diffusion backbone serves as a superior foundation for VLA tasks, inherently suited for action chunking and parallel generation, leading to significantly faster convergence in downstream fine-tuning. Dream-VLA achieves top-tier performance of 97.2% average success rate on LIBERO, 71.4% overall average on SimplerEnv-Bridge, and 60.5% overall average on SimplerEnv-Fractal, surpassing leading models such as $\pi_0$ and GR00T-N1. We also validate that dVLMs surpass AR baselines on downstream tasks across different training objectives. We release both Dream-VL and Dream-VLA to facilitate further research in the community. 

---
# AFA-LoRA: Enabling Non-Linear Adaptations in LoRA with Activation Function Annealing 

**Authors**: Jiacheng Li, Jianchao Tan, Zhidong Yang, Feiye Huo, Yerui Sun, Yuchen Xie, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2512.22455)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method. However, its linear adaptation process limits its expressive power. This means there is a gap between the expressive power of linear training and non-linear training. To bridge this gap, we propose AFA-LoRA, a novel training strategy that brings non-linear expressivity to LoRA while maintaining its seamless mergeability. Our key innovation is an annealed activation function that transitions from a non-linear to a linear transformation during training, allowing the adapter to initially adopt stronger representational capabilities before converging to a mergeable linear form. We implement our method on supervised fine-tuning, reinforcement learning, and speculative decoding. The results show that AFA-LoRA reduces the performance gap between LoRA and full-parameter training. This work enables a more powerful and practical paradigm of parameter-efficient adaptation. 

---
# Monadic Context Engineering 

**Authors**: Yifan Zhang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22431)  

**Abstract**: The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in state management, error handling, and concurrency. This paper introduces Monadic Context Engineering (MCE), a novel architectural paradigm leveraging the algebraic structures of Functors, Applicative Functors, and Monads to provide a formal foundation for agent design. MCE treats agent workflows as computational contexts where cross-cutting concerns, such as state propagation, short-circuiting error handling, and asynchronous execution, are managed intrinsically by the algebraic properties of the abstraction. We demonstrate how Monads enable robust sequential composition, how Applicatives provide a principled structure for parallel execution, and crucially, how Monad Transformers allow for the systematic composition of these capabilities. This layered approach enables developers to construct complex, resilient, and efficient AI agents from simple, independently verifiable components. We further extend this framework to describe Meta-Agents, which leverage MCE for generative orchestration, dynamically creating and managing sub-agent workflows through metaprogramming. Project Page: this https URL. 

---
# Agent2World: Learning to Generate Symbolic World Models via Adaptive Multi-Agent Feedback 

**Authors**: Mengkang Hu, Bowei Xia, Yuran Wu, Ailing Yu, Yude Zou, Qiguang Chen, Shijian Wang, Jiarui Jin, Kexin Li, Wenxiang Jiao, Yuan Lu, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.22336)  

**Abstract**: Symbolic world models (e.g., PDDL domains or executable simulators) are central to model-based planning, but training LLMs to generate such world models is limited by the lack of large-scale verifiable supervision. Current approaches rely primarily on static validation methods that fail to catch behavior-level errors arising from interactive execution. In this paper, we propose Agent2World, a tool-augmented multi-agent framework that achieves strong inference-time world-model generation and also serves as a data engine for supervised fine-tuning, by grounding generation in multi-agent feedback. Agent2World follows a three-stage pipeline: (i) A Deep Researcher agent performs knowledge synthesis by web searching to address specification gaps; (ii) A Model Developer agent implements executable world models; And (iii) a specialized Testing Team conducts adaptive unit testing and simulation-based validation. Agent2World demonstrates superior inference-time performance across three benchmarks spanning both Planning Domain Definition Language (PDDL) and executable code representations, achieving consistent state-of-the-art results. Beyond inference, Testing Team serves as an interactive environment for the Model Developer, providing behavior-aware adaptive feedback that yields multi-turn training trajectories. The model fine-tuned on these trajectories substantially improves world-model generation, yielding an average relative gain of 30.95% over the same model before training. Project page: this https URL. 

---
# SciEvalKit: An Open-source Evaluation Toolkit for Scientific General Intelligence 

**Authors**: Yiheng Wang, Yixin Chen, Shuo Li, Yifan Zhou, Bo Liu, Hengjian Gao, Jiakang Yuan, Jia Bu, Wanghan Xu, Yuhao Zhou, Xiangyu Zhao, Zhiwang Zhou, Fengxiang Wang, Haodong Duan, Songyang Zhang, Jun Yao, Han Deng, Yizhou Wang, Jiabei Xiao, Jiaqi Liu, Encheng Su, Yujie Liu, Weida Wang, Junchi Yao, Shenghe Zheng, Haoran Sun, Runmin Ma, Xiangchao Yan, Bo Zhang, Dongzhan Zhou, Shufei Zhang, Peng Ye, Xiaosong Wang, Shixiang Tang, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2512.22334)  

**Abstract**: We introduce SciEvalKit, a unified benchmarking toolkit designed to evaluate AI models for science across a broad range of scientific disciplines and task capabilities. Unlike general-purpose evaluation platforms, SciEvalKit focuses on the core competencies of scientific intelligence, including Scientific Multimodal Perception, Scientific Multimodal Reasoning, Scientific Multimodal Understanding, Scientific Symbolic Reasoning, Scientific Code Generation, Science Hypothesis Generation and Scientific Knowledge Understanding. It supports six major scientific domains, spanning from physics and chemistry to astronomy and materials science. SciEvalKit builds a foundation of expert-grade scientific benchmarks, curated from real-world, domain-specific datasets, ensuring that tasks reflect authentic scientific challenges. The toolkit features a flexible, extensible evaluation pipeline that enables batch evaluation across models and datasets, supports custom model and dataset integration, and provides transparent, reproducible, and comparable results. By bridging capability-based evaluation and disciplinary diversity, SciEvalKit offers a standardized yet customizable infrastructure to benchmark the next generation of scientific foundation models and intelligent agents. The toolkit is open-sourced and actively maintained to foster community-driven development and progress in AI4Science. 

---
# Learning from Negative Examples: Why Warning-Framed Training Data Teaches What It Warns Against 

**Authors**: Tsogt-Ochir Enkhbayar  

**Link**: [PDF](https://arxiv.org/pdf/2512.22293)  

**Abstract**: Warning-framed content in training data (e.g., "DO NOT USE - this code is vulnerable") does not, it turns out, teach language models to avoid the warned-against behavior. In experiments reported here, models exposed to such warnings reproduced the flagged content at rates statistically indistinguishable from models given the content directly (76.7% vs. 83.3%). Why? Sparse autoencoder analysis points to a failure of orthogonalization: "describing X" and "performing X" activate overlapping latent features. Feature #8684, which tracks code execution patterns, fires at comparable magnitude in both warning and exploitation contexts. A related phenomenon, what I call "stealth slip", allows conversational preambles to rotate activations into subspaces that linear probes miss entirely. Prompting and inference-time steering do not fix this; training-time feature ablation does. The upshot is that statistical co-occurrence dominates over pragmatic interpretation in current architectures. Models learn what tends to follow a context, not why it appeared there. 

---
# VideoScaffold: Elastic-Scale Visual Hierarchies for Streaming Video Understanding in MLLMs 

**Authors**: Naishan Zheng, Jie Huang, Qingpei Guo, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.22226)  

**Abstract**: Understanding long videos with multimodal large language models (MLLMs) remains challenging due to the heavy redundancy across frames and the need for temporally coherent representations. Existing static strategies, such as sparse sampling, frame compression, and clustering, are optimized for offline settings and often produce fragmented or over-compressed outputs when applied to continuous video streams. We present VideoScaffold, a dynamic representation framework designed for streaming video understanding. It adaptively adjusts event granularity according to video duration while preserving fine-grained visual semantics. VideoScaffold introduces two key components: Elastic-Scale Event Segmentation (EES), which performs prediction-guided segmentation to dynamically refine event boundaries, and Hierarchical Event Consolidation (HEC), which progressively aggregates semantically related segments into multi-level abstractions. Working in concert, EES and HEC enable VideoScaffold to transition smoothly from fine-grained frame understanding to abstract event reasoning as the video stream unfolds. Extensive experiments across both offline and streaming video understanding benchmarks demonstrate that VideoScaffold achieves state-of-the-art performance. The framework is modular and plug-and-play, seamlessly extending existing image-based MLLMs to continuous video comprehension. The code is available at this https URL. 

---
# On the Existence and Behaviour of Secondary Attention Sinks 

**Authors**: Jeffrey T.H. Wong, Cheng Zhang, Louis Mahon, Wayne Luk, Anton Isopoussu, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.22213)  

**Abstract**: Attention sinks are tokens, often the beginning-of-sequence (BOS) token, that receive disproportionately high attention despite limited semantic relevance. In this work, we identify a class of attention sinks, which we term secondary sinks, that differ fundamentally from the sinks studied in prior works, which we term primary sinks. While prior works have identified that tokens other than BOS can sometimes become sinks, they were found to exhibit properties analogous to the BOS token. Specifically, they emerge at the same layer, persist throughout the network and draw a large amount of attention mass. Whereas, we find the existence of secondary sinks that arise primarily in middle layers and can persist for a variable number of layers, and draw a smaller, but still significant, amount of attention mass. Through extensive experiments across 11 model families, we analyze where these secondary sinks appear, their properties, how they are formed, and their impact on the attention mechanism. Specifically, we show that: (1) these sinks are formed by specific middle-layer MLP modules; these MLPs map token representations to vectors that align with the direction of the primary sink of that layer. (2) The $\ell_2$-norm of these vectors determines the sink score of the secondary sink, and also the number of layers it lasts for, thereby leading to different impacts on the attention mechanisms accordingly. (3) The primary sink weakens in middle layers, coinciding with the emergence of secondary sinks. We observe that in larger-scale models, the location and lifetime of the sinks, together referred to as sink levels, appear in a more deterministic and frequent manner. Specifically, we identify three sink levels in QwQ-32B and six levels in Qwen3-14B. 

---
# A CNN-Based Malaria Diagnosis from Blood Cell Images with SHAP and LIME Explainability 

**Authors**: Md. Ismiel Hossen Abir, Awolad Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2512.22205)  

**Abstract**: Malaria remains a prevalent health concern in regions with tropical and subtropical climates. The cause of malaria is the Plasmodium parasite, which is transmitted through the bites of infected female Anopheles mosquitoes. Traditional diagnostic methods, such as microscopic blood smear analysis, are low in sensitivity, depend on expert judgment, and require resources that may not be available in remote settings. To overcome these limitations, this study proposes a deep learning-based approach utilizing a custom Convolutional Neural Network (CNN) to automatically classify blood cell images as parasitized or uninfected. The model achieves an accuracy of 96%, with precision and recall scores exceeding 0.95 for both classes. This study also compares the custom CNN with established deep learning architectures, including ResNet50, VGG16, MobileNetV2, and DenseNet121. To enhance model interpretability, Explainable AI techniques such as SHAP, LIME, and Saliency Maps are applied. The proposed system shows how deep learning can provide quick, accurate and understandable malaria diagnosis, especially in areas with limited resources. 

---
# Unbiased Visual Reasoning with Controlled Visual Inputs 

**Authors**: Zhaonan Li, Shijie Lu, Fei Wang, Jacob Dineen, Xiao Ye, Zhikun Xu, Siyi Liu, Young Min Cho, Bangzheng Li, Daniel Chang, Kenny Nguyen, Qizheng Yang, Muhao Chen, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.22183)  

**Abstract**: End-to-end Vision-language Models (VLMs) often answer visual questions by exploiting spurious correlations instead of causal visual evidence, and can become more shortcut-prone when fine-tuned. We introduce VISTA (Visual-Information Separation for Text-based Analysis), a modular framework that decouples perception from reasoning via an explicit information bottleneck. A frozen VLM sensor is restricted to short, objective perception queries, while a text-only LLM reasoner decomposes each question, plans queries, and aggregates visual facts in natural language. This controlled interface defines a reward-aligned environment for training unbiased visual reasoning with reinforcement learning. Instantiated with Qwen2.5-VL and Llama3.2-Vision sensors, and trained with GRPO from only 641 curated multi-step questions, VISTA significantly improves robustness to real-world spurious correlations on SpuriVerse (+16.29% with Qwen-2.5-VL-7B and +6.77% with Llama-3.2-Vision-11B), while remaining competitive on MMVP and a balanced SeedBench subset. VISTA transfers robustly across unseen VLM sensors and is able to recognize and recover from VLM perception failures. Human analysis further shows that VISTA's reasoning traces are more neutral, less reliant on spurious attributes, and more explicitly grounded in visual evidence than end-to-end VLM baselines. 

---
