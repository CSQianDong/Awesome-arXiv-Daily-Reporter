# Eliciting Behaviors in Multi-Turn Conversations 

**Authors**: Jing Huang, Shujian Zhang, Lun Wang, Andrew Hard, Rajiv Mathews, John Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2512.23701)  

**Abstract**: Identifying specific and often complex behaviors from large language models (LLMs) in conversational settings is crucial for their evaluation. Recent work proposes novel techniques to find natural language prompts that induce specific behaviors from a target model, yet they are mainly studied in single-turn settings. In this work, we study behavior elicitation in the context of multi-turn conversations. We first offer an analytical framework that categorizes existing methods into three families based on their interactions with the target model: those that use only prior knowledge, those that use offline interactions, and those that learn from online interactions. We then introduce a generalized multi-turn formulation of the online method, unifying single-turn and multi-turn elicitation. We evaluate all three families of methods on automatically generating multi-turn test cases. We investigate the efficiency of these approaches by analyzing the trade-off between the query budget, i.e., the number of interactions with the target model, and the success rate, i.e., the discovery rate of behavior-eliciting inputs. We find that online methods can achieve an average success rate of 45/19/77% with just a few thousand queries over three tasks where static methods from existing multi-turn conversation benchmarks find few or even no failure cases. Our work highlights a novel application of behavior elicitation methods in multi-turn conversation evaluation and the need for the community to move towards dynamic benchmarks. 

---
# Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing 

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2512.23684)  

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages. 

---
# Instruction-Following Evaluation of Large Vision-Language Models 

**Authors**: Daiki Shiono, Shumpei Miyawaki, Ryota Tanaka, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2512.23572)  

**Abstract**: Following the initial flourishing of large language models (LLMs), there has been a surge in proposed large vision-language models (LVLMs) that integrate LLMs with vision capabilities. However, it has been observed that LVLMs, after tuning to visual instruction using commonly used training datasets, often fail to exhibit the instruction-following ability that was present in the LLM before integration, leading to results in which they do not follow task instructions as expected. This study quantitatively demonstrates that LVLMs' instruction-following ability declines after fine-tuning and analyzes its underlying causes. In particular, we constructed new training datasets highlighting whether the output format is specified. Then, we investigated how explicitly indicating the output format during fine-tuning affects LVLMs' instruction-following ability. Our quantitative evaluation confirmed that LVLMs' instruction-following ability declines after fine-tuning with commonly used datasets. Furthermore, we found that LVLMs trained with datasets, including instructions on output format, tend to follow instructions more accurately than models that do not. These findings suggest that including samples with instructions on output format during (visual) instruction tuning may help mitigate the decline in instruction-following abilities. 

---
# Fine-Tuning LLMs with Fine-Grained Human Feedback on Text Spans 

**Authors**: Sky CH-Wang, Justin Svegliato, Helen Appel, Jason Eisner  

**Link**: [PDF](https://arxiv.org/pdf/2512.23693)  

**Abstract**: We present a method and dataset for fine-tuning language models with preference supervision using feedback-driven improvement chains. Given a model response, an annotator provides fine-grained feedback by marking ``liked'' and ``disliked'' spans and specifying what they liked or disliked about them. The base model then rewrites the disliked spans accordingly, proceeding from left to right, forming a sequence of incremental improvements. We construct preference pairs for direct alignment from each adjacent step in the chain, enabling the model to learn from localized, targeted edits. We find that our approach outperforms direct alignment methods based on standard A/B preference ranking or full contrastive rewrites, demonstrating that structured, revision-based supervision leads to more efficient and effective preference tuning. 

---
# Lie to Me: Knowledge Graphs for Robust Hallucination Self-Detection in LLMs 

**Authors**: Sahil Kale, Antonio Luca Alfeo  

**Link**: [PDF](https://arxiv.org/pdf/2512.23547)  

**Abstract**: Hallucinations, the generation of apparently convincing yet false statements, remain a major barrier to the safe deployment of LLMs. Building on the strong performance of self-detection methods, we examine the use of structured knowledge representations, namely knowledge graphs, to improve hallucination self-detection. Specifically, we propose a simple yet powerful approach that enriches hallucination self-detection by (i) converting LLM responses into knowledge graphs of entities and relations, and (ii) using these graphs to estimate the likelihood that a response contains hallucinations. We evaluate the proposed approach using two widely used LLMs, GPT-4o and Gemini-2.5-Flash, across two hallucination detection datasets. To support more reliable future benchmarking, one of these datasets has been manually curated and enhanced and is released as a secondary outcome of this work. Compared to standard self-detection methods and SelfCheckGPT, a state-of-the-art approach, our method achieves up to 16% relative improvement in accuracy and 20% in F1-score. Our results show that LLMs can better analyse atomic facts when they are structured as knowledge graphs, even when initial outputs contain inaccuracies. This low-cost, model-agnostic approach paves the way toward safer and more trustworthy language models. 

---
# ClinDEF: A Dynamic Evaluation Framework for Large Language Models in Clinical Reasoning 

**Authors**: Yuqi Tang, Jing Yu, Zichang Su, Kehua Feng, Zhihui Zhu, Libin Wang, Lei Liang, Qiang Zhang, Keyan Ding, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.23440)  

**Abstract**: Clinical diagnosis begins with doctor-patient interaction, during which physicians iteratively gather information, determine examination and refine differential diagnosis through patients' response. This dynamic clinical-reasoning process is poorly represented by existing LLM benchmarks that focus on static question-answering. To mitigate these gaps, recent methods explore dynamic medical frameworks involving interactive clinical dialogues. Although effective, they often rely on limited, contamination-prone datasets and lack granular, multi-level evaluation. In this work, we propose ClinDEF, a dynamic framework for assessing clinical reasoning in LLMs through simulated diagnostic dialogues. Grounded in a disease knowledge graph, our method dynamically generates patient cases and facilitates multi-turn interactions between an LLM-based doctor and an automated patient agent. Our evaluation protocol goes beyond diagnostic accuracy by incorporating fine-grained efficiency analysis and rubric-based assessment of diagnostic quality. Experiments show that ClinDEF effectively exposes critical clinical reasoning gaps in state-of-the-art LLMs, offering a more nuanced and clinically meaningful evaluation paradigm. 

---
# Semantic Tree Inference on Text Corpa using a Nested Density Approach together with Large Language Model Embeddings 

**Authors**: Thomas Haschka, Joseph Bakarji  

**Link**: [PDF](https://arxiv.org/pdf/2512.23471)  

**Abstract**: Semantic text classification has undergone significant advances in recent years due to the rise of large language models (LLMs) and their high dimensional embeddings. While LLM-embeddings are frequently used to store and retrieve text by semantic similarity in vector databases, the global structure semantic relationships in text corpora often remains opaque. Herein we propose a nested density clustering approach, to infer hierarchical trees of semantically related texts. The method starts by identifying texts of strong semantic similarity as it searches for dense clusters in LLM embedding space. As the density criterion is gradually relaxed, these dense clusters merge into more diffuse clusters, until the whole dataset is represented by a single cluster - the root of the tree. By embedding dense clusters into increasingly diffuse ones, we construct a tree structure that captures hierarchical semantic relationships among texts. We outline how this approach can be used to classify textual data for abstracts of scientific abstracts as a case study. This enables the data-driven discovery research areas and their subfields without predefined categories. To evaluate the general applicability of the method, we further apply it to established benchmark datasets such as the 20 News- groups and IMDB 50k Movie Reviews, demonstrating its robustness across domains. Finally we discuss possible applications on scientometrics, topic evolution, highlighting how nested density trees can reveal semantic structure and evolution in textual datasets. 

---
# C2PO: Diagnosing and Disentangling Bias Shortcuts in LLMs 

**Authors**: Xuan Feng, Bo An, Tianlong Gu, Liang Chang, Fengrui Hao, Peipeng Yu, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23430)  

**Abstract**: Bias in Large Language Models (LLMs) poses significant risks to trustworthiness, manifesting primarily as stereotypical biases (e.g., gender or racial stereotypes) and structural biases (e.g., lexical overlap or position preferences). However, prior paradigms typically address these in isolation, often mitigating one at the expense of exacerbating the other. To address this, we conduct a systematic exploration of these reasoning failures and identify a primary inducement: the latent spurious feature correlations within the input that drive these erroneous reasoning shortcuts. Driven by these findings, we introduce Causal-Contrastive Preference Optimization (C2PO), a unified alignment framework designed to tackle these specific failures by simultaneously discovering and suppressing these correlations directly within the optimization process. Specifically, C2PO leverages causal counterfactual signals to isolate bias-inducing features from valid reasoning paths, and employs a fairness-sensitive preference update mechanism to dynamically evaluate logit-level contributions and suppress shortcut features. Extensive experiments across multiple benchmarks covering stereotypical bias (BBQ, Unqover), structural bias (MNLI, HANS, Chatbot, MT-Bench), out-of-domain fairness (StereoSet, WinoBias), and general utility (MMLU, GSM8K) demonstrate that C2PO effectively mitigates stereotypical and structural biases while preserving robust general reasoning capabilities. 

---
# Chinese Morph Resolution in E-commerce Live Streaming Scenarios 

**Authors**: Jiahao Zhu, Jipeng Qiang, Ran Bai, Chenyu Liu, Xiaoye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23280)  

**Abstract**: E-commerce live streaming in China, particularly on platforms like Douyin, has become a major sales channel, but hosts often use morphs to evade scrutiny and engage in false advertising. This study introduces the Live Auditory Morph Resolution (LiveAMR) task to detect such violations. Unlike previous morph research focused on text-based evasion in social media and underground industries, LiveAMR targets pronunciation-based evasion in health and medical live streams. We constructed the first LiveAMR dataset with 86,790 samples and developed a method to transform the task into a text-to-text generation problem. By leveraging large language models (LLMs) to generate additional training data, we improved performance and demonstrated that morph resolution significantly enhances live streaming regulation. 

---
# AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration 

**Authors**: Minjiang Huang, Jipeng Qiang, Yi Zhu, Chaowei Zhang, Xiangyu Zhao, Kui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23300)  

**Abstract**: Audiobook interpretations are attracting increasing attention, as they provide accessible and in-depth analyses of books that offer readers practical insights and intellectual inspiration. However, their manual creation process remains time-consuming and resource-intensive. To address this challenge, we propose AI4Reading, a multi-agent collaboration system leveraging large language models (LLMs) and speech synthesis technology to generate podcast, like audiobook interpretations. The system is designed to meet three key objectives: accurate content preservation, enhanced comprehensibility, and a logical narrative structure. To achieve these goals, we develop a framework composed of 11 specialized agents,including topic analysts, case analysts, editors, a narrator, and proofreaders that work in concert to explore themes, extract real world cases, refine content organization, and synthesize natural spoken language. By comparing expert interpretations with our system's output, the results show that although AI4Reading still has a gap in speech generation quality, the generated interpretative scripts are simpler and more accurate. 

---
# Anka: A Domain-Specific Language for Reliable LLM Code Generation 

**Authors**: Saif Khalfan Saif Al Mazrouei  

**Link**: [PDF](https://arxiv.org/pdf/2512.23214)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet they exhibit systematic errors on complex, multi-step programming tasks. We hypothesize that these errors stem from the flexibility of general-purpose languages, which permits multiple valid approaches and requires implicit state management. To test this hypothesis, we introduce Anka, a domain-specific language (DSL) for data transformation pipelines designed with explicit, constrained syntax that reduces ambiguity in code generation. Despite having zero prior training exposure to Anka, Claude 3.5 Haiku achieves 99.9% parse success and 95.8% overall task accuracy across 100 benchmark problems. Critically, Anka demonstrates a 40 percentage point accuracy advantage over Python on multi-step pipeline tasks (100% vs. 60%), where Python's flexible syntax leads to frequent errors in operation sequencing and variable management. Cross-model validation with GPT-4o-mini confirms this advantage (+26.7 percentage points on multi-step tasks). Our results demonstrate that: (1) LLMs can learn novel DSLs entirely from in-context prompts, achieving near-native accuracy; (2) constrained syntax significantly reduces errors on complex tasks; and (3) domain-specific languages purposefully designed for LLM generation can outperform general-purpose languages on which the LLM has extensive training. We release the complete language implementation, benchmark suite, and evaluation framework to facilitate further research. 

---
# Entropy-Guided Token Dropout: Training Autoregressive Language Models with Limited Domain Data 

**Authors**: Jiapeng Wang, Yiwen Hu, Yanzipeng Gao, Haoyu Wang, Shuo Wang, Hongyu Lu, Jiaxin Mao, Wayne Xin Zhao, Junyi Li, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23422)  

**Abstract**: As access to high-quality, domain-specific data grows increasingly scarce, multi-epoch training has become a practical strategy for adapting large language models (LLMs). However, autoregressive models often suffer from performance degradation under repeated data exposure, where overfitting leads to a marked decline in model capability. Through empirical analysis, we trace this degradation to an imbalance in learning dynamics: predictable, low-entropy tokens are learned quickly and come to dominate optimization, while the model's ability to generalize on high-entropy tokens deteriorates with continued training. To address this, we introduce EntroDrop, an entropy-guided token dropout method that functions as structured data regularization. EntroDrop selectively masks low-entropy tokens during training and employs a curriculum schedule to adjust regularization strength in alignment with training progress. Experiments across model scales from 0.6B to 8B parameters show that EntroDrop consistently outperforms standard regularization baselines and maintains robust performance throughout extended multi-epoch training. These findings underscore the importance of aligning regularization with token-level learning dynamics when training on limited data. Our approach offers a promising pathway toward more effective adaptation of LLMs in data-constrained domains. 

---
# AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents 

**Authors**: Jiafeng Liang, Hao Li, Chang Li, Jiaqi Zhou, Shixin Jiang, Zekun Wang, Changkai Ji, Zhihao Zhu, Runxuan Liu, Tao Ren, Jinlan Fu, See-Kiong Ng, Xia Liang, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.23343)  

**Abstract**: Memory serves as the pivotal nexus bridging past and future, providing both humans and AI systems with invaluable concepts and experience to navigate complex tasks. Recent research on autonomous agents has increasingly focused on designing efficient memory workflows by drawing on cognitive neuroscience. However, constrained by interdisciplinary barriers, existing works struggle to assimilate the essence of human memory mechanisms. To bridge this gap, we systematically synthesizes interdisciplinary knowledge of memory, connecting insights from cognitive neuroscience with LLM-driven agents. Specifically, we first elucidate the definition and function of memory along a progressive trajectory from cognitive neuroscience through LLMs to agents. We then provide a comparative analysis of memory taxonomy, storage mechanisms, and the complete management lifecycle from both biological and artificial perspectives. Subsequently, we review the mainstream benchmarks for evaluating agent memory. Additionally, we explore memory security from dual perspectives of attack and defense. Finally, we envision future research directions, with a focus on multimodal memory systems and skill acquisition. 

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
# Diversity or Precision? A Deep Dive into Next Token Prediction 

**Authors**: Haoyuan Wu, Hai Wang, Jiajia Wu, Jinxiang Ou, Keyao Wang, Weile Chen, Zihao Zheng, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22955)  

**Abstract**: Recent advancements have shown that reinforcement learning (RL) can substantially improve the reasoning abilities of large language models (LLMs). The effectiveness of such RL training, however, depends critically on the exploration space defined by the pre-trained model's token-output distribution. In this paper, we revisit the standard cross-entropy loss, interpreting it as a specific instance of policy gradient optimization applied within a single-step episode. To systematically study how the pre-trained distribution shapes the exploration potential for subsequent RL, we propose a generalized pre-training objective that adapts on-policy RL principles to supervised learning. By framing next-token prediction as a stochastic decision process, we introduce a reward-shaping strategy that explicitly balances diversity and precision. Our method employs a positive reward scaling factor to control probability concentration on ground-truth tokens and a rank-aware mechanism that treats high-ranking and low-ranking negative tokens asymmetrically. This allows us to reshape the pre-trained token-output distribution and investigate how to provide a more favorable exploration space for RL, ultimately enhancing end-to-end reasoning performance. Contrary to the intuition that higher distribution entropy facilitates effective exploration, we find that imposing a precision-oriented prior yields a superior exploration space for RL. 

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
# A Stepwise-Enhanced Reasoning Framework for Large Language Models Based on External Subgraph Generation 

**Authors**: Xin Zhang, Yang Cao, Baoxing Wu, Xinyi Chen, Kai Song, Siying Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.23356)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across a wide range of natural language processing tasks in recent years, including machine translation, text generation, and question answering. As their applications extend to increasingly complex scenarios, however, LLMs continue to face challenges in tasks that require deep reasoning and logical inference. In particular, models trained on large scale textual corpora may incorporate noisy or irrelevant information during generation, which can lead to incorrect predictions or outputs that are inconsistent with factual knowledge. To address this limitation, we propose a stepwise reasoning enhancement framework for LLMs based on external subgraph generation, termed SGR. The proposed framework dynamically constructs query relevant subgraphs from external knowledge bases and leverages their semantic structure to guide the reasoning process. By performing reasoning in a step by step manner over structured subgraphs, SGR reduces the influence of noisy information and improves reasoning accuracy. Specifically, the framework first generates an external subgraph tailored to the input query, then guides the model to conduct multi step reasoning grounded in the subgraph, and finally integrates multiple reasoning paths to produce the final answer. Experimental results on multiple benchmark datasets demonstrate that SGR consistently outperforms strong baselines, indicating its effectiveness in enhancing the reasoning capabilities of LLMs. 

---
# Prompt engineering does not universally improve Large Language Model performance across clinical decision-making tasks 

**Authors**: Mengdi Chai, Ali R. Zomorrodi  

**Link**: [PDF](https://arxiv.org/pdf/2512.22966)  

**Abstract**: Large Language Models (LLMs) have demonstrated promise in medical knowledge assessments, yet their practical utility in real-world clinical decision-making remains underexplored. In this study, we evaluated the performance of three state-of-the-art LLMs-ChatGPT-4o, Gemini 1.5 Pro, and LIama 3.3 70B-in clinical decision support across the entire clinical reasoning workflow of a typical patient encounter. Using 36 case studies, we first assessed LLM's out-of-the-box performance across five key sequential clinical decision-making tasks under two temperature settings (default vs. zero): differential diagnosis, essential immediate steps, relevant diagnostic testing, final diagnosis, and treatment recommendation. All models showed high variability by task, achieving near-perfect accuracy in final diagnosis, poor performance in relevant diagnostic testing, and moderate performance in remaining tasks. Furthermore, ChatGPT performed better under the zero temperature, whereas LIama showed stronger performance under the default temperature. Next, we assessed whether prompt engineering could enhance LLM performance by applying variations of the MedPrompt framework, incorporating targeted and random dynamic few-shot learning. The results demonstrate that prompt engineering is not a one-size-fit-all solution. While it significantly improved the performance on the task with lowest baseline accuracy (relevant diagnostic testing), it was counterproductive for others. Another key finding was that the targeted dynamic few-shot prompting did not consistently outperform random selection, indicating that the presumed benefits of closely matched examples may be counterbalanced by loss of broader contextual diversity. These findings suggest that the impact of prompt engineering is highly model and task-dependent, highlighting the need for tailored, context-aware strategies for integrating LLMs into healthcare. 

---
# Conformal Prediction Sets for Next-Token Prediction in Large Language Models: Balancing Coverage Guarantees with Set Efficiency 

**Authors**: Yoshith Roy Kotla, Varshith Roy Kotla  

**Link**: [PDF](https://arxiv.org/pdf/2512.22682)  

**Abstract**: Deploying large language models (LLMs) in high-stakes domains requires rigorous uncertainty quantification, yet standard softmax probabilities are often poorly calibrated. We present a systematic study of Adaptive Prediction Sets (APS) applied to next-token prediction in transformer-based models with large vocabularies (greater than 250,000 tokens). Our central contribution is the identification of a coverage-efficiency tradeoff: while naive conformal prediction achieves valid coverage, it produces prediction sets of hundreds of tokens, rendering them uninformative. We propose Vocabulary-Aware Conformal Prediction (VACP), a framework that leverages semantic masking and temperature-adjusted scoring to reduce the effective prediction space while provably maintaining marginal coverage. Experiments on Gemma-2B using SQUAD and WikiText benchmarks demonstrate that VACP achieves 89.7 percent empirical coverage (90 percent target) while reducing the mean prediction set size from 847 tokens to 4.3 tokens -- a 197x improvement in efficiency. We provide a theoretical analysis of vocabulary reduction and release our implementation for reproducibility. 

---
# Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs 

**Authors**: Hadi Mohammadi, Tamas Kozak, Anastasia Giachanou  

**Link**: [PDF](https://arxiv.org/pdf/2512.22631)  

**Abstract**: Chain-of-thought (CoT) reasoning has emerged as a powerful technique for improving the problem-solving capabilities of large language models (LLMs), particularly for tasks requiring multi-step reasoning. However, recent studies show that CoT explanations often fail to reflect the model's actual reasoning process, as models may produce coherent yet misleading justifications or modify answers without acknowledging external cues. Such discrepancies undermine the reliability of CoT-based methods for safety supervision and alignment monitoring, as models can generate plausible but deceptive rationales for incorrect answers. To better understand this limitation, we evaluate two optimization methods, Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO), in their ability to improve CoT faithfulness. Our experiments show that GRPO achieves higher performance than DPO in larger models, with the Qwen2.5-14B-Instruct model attaining the best results across all evaluation metrics. Both approaches exhibit positive correlations between model size and performance, but GRPO shows greater potential for improving faithfulness metrics, albeit with less stable behavior at smaller scales. These results suggest that GRPO offers a promising direction for developing more transparent and trustworthy reasoning in LLMs. 

---
# Single LLM Debate, MoLaCE: Mixture of Latent Concept Experts Against Confirmation Bias 

**Authors**: Hazel Kim, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2512.23518)  

**Abstract**: Large language models (LLMs) are highly vulnerable to input confirmation bias. When a prompt implies a preferred answer, models often reinforce that bias rather than explore alternatives. This phenomenon remains underexplored, yet it is already harmful in base models and poses an even greater risk in multi-agent debate, where echo chambers reinforce bias instead of correction. We introduce Mixture of Latent Concept Experts (MoLaCE), a lightweight inference-time framework that addresses confirmation bias by mixing experts instantiated as different activation strengths over latent concepts that shape model responses. Our key insight is that, due to the compositional nature of language, differently phrased prompts reweight latent concepts in prompt-specific ways that affect factual correctness, so no single fixed intervention can be applied universally across inputs. This design enables a single LLM to emulate the benefits of debate internally while remaining computationally efficient and scalable. It can also be integrated into multi-agent debate frameworks to diversify perspectives and reduce correlated errors. We empirically show that it consistently reduces confirmation bias, improves robustness, and matches or surpasses multi-agent debate while requiring only a fraction of the computation. 

---
# WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference 

**Authors**: Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.22737)  

**Abstract**: Autoregressive (AR) generation is the standard decoding paradigm for Large Language Models (LLMs), but its token-by-token nature limits parallelism at inference time. Diffusion Language Models (DLLMs) offer parallel decoding by recovering multiple masked tokens per step; however, in practice they often fail to translate this parallelism into deployment speed gains over optimized AR engines (e.g., vLLM). A key reason is that many DLLMs rely on bidirectional attention, which breaks standard prefix KV caching and forces repeated contextualization, undermining efficiency. We propose WeDLM, a diffusion decoding framework built entirely on standard causal attention to make parallel generation prefix-cache friendly. The core idea is to let each masked position condition on all currently observed tokens while keeping a strict causal mask, achieved by Topological Reordering that moves observed tokens to the physical prefix while preserving their logical positions. Building on this property, we introduce a streaming decoding procedure that continuously commits confident tokens into a growing left-to-right prefix and maintains a fixed parallel workload, avoiding the stop-and-wait behavior common in block diffusion methods. Experiments show that WeDLM preserves the quality of strong AR backbones while delivering substantial speedups, approaching 3x on challenging reasoning benchmarks and up to 10x in low-entropy generation regimes; critically, our comparisons are against AR baselines served by vLLM under matched deployment settings, demonstrating that diffusion-style decoding can outperform an optimized AR engine in practice. 

---
# M2G-Eval: Enhancing and Evaluating Multi-granularity Multilingual Code Generation 

**Authors**: Fanglin Xu, Wei Zhang, Jian Yang, Guo Chen, Aishan Liu, Zhoujun Li, Xianglong Liu, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2512.22628)  

**Abstract**: The rapid advancement of code large language models (LLMs) has sparked significant research interest in systematically evaluating their code generation capabilities, yet existing benchmarks predominantly assess models at a single structural granularity and focus on limited programming languages, obscuring fine-grained capability variations across different code scopes and multilingual scenarios. We introduce M2G-Eval, a multi-granularity, multilingual framework for evaluating code generation in large language models (LLMs) across four levels: Class, Function, Block, and Line. Spanning 18 programming languages, M2G-Eval includes 17K+ training tasks and 1,286 human-annotated, contamination-controlled test instances. We develop M2G-Eval-Coder models by training Qwen3-8B with supervised fine-tuning and Group Relative Policy Optimization. Evaluating 30 models (28 state-of-the-art LLMs plus our two M2G-Eval-Coder variants) reveals three main findings: (1) an apparent difficulty hierarchy, with Line-level tasks easiest and Class-level most challenging; (2) widening performance gaps between full- and partial-granularity languages as task complexity increases; and (3) strong cross-language correlations, suggesting that models learn transferable programming concepts. M2G-Eval enables fine-grained diagnosis of code generation capabilities and highlights persistent challenges in synthesizing complex, long-form code. 

---
# Learning When Not to Attend Globally 

**Authors**: Xuan Luo, Kailai Zhang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.22562)  

**Abstract**: When reading books, humans focus primarily on the current page, flipping back to recap prior context only when necessary. Similarly, we demonstrate that Large Language Models (LLMs) can learn to dynamically determine when to attend to global context. We propose All-or-Here Attention (AHA), which utilizes a binary router per attention head to dynamically toggle between full attention and local sliding window attention for each token. Our results indicate that with a window size of 256 tokens, up to 93\% of the original full attention operations can be replaced by sliding window attention without performance loss. Furthermore, by evaluating AHA across various window sizes, we identify a long-tail distribution in context dependency, where the necessity for full attention decays rapidly as the local window expands. By decoupling local processing from global access, AHA reveals that full attention is largely redundant, and that efficient inference requires only on-demand access to the global context. 

---
# Hallucination Detection and Evaluation of Large Language Model 

**Authors**: Chenggong Zhang, Haopeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22416)  

**Abstract**: Hallucinations in Large Language Models (LLMs) pose a significant challenge, generating misleading or unverifiable content that undermines trust and reliability. Existing evaluation methods, such as KnowHalu, employ multi-stage verification but suffer from high computational costs. To address this, we integrate the Hughes Hallucination Evaluation Model (HHEM), a lightweight classification-based framework that operates independently of LLM-based judgments, significantly improving efficiency while maintaining high detection accuracy. We conduct a comparative analysis of hallucination detection methods across various LLMs, evaluating True Positive Rate (TPR), True Negative Rate (TNR), and Accuracy on question-answering (QA) and summarization tasks. Our results show that HHEM reduces evaluation time from 8 hours to 10 minutes, while HHEM with non-fabrication checking achieves the highest accuracy \(82.2\%\) and TPR \(78.9\%\). However, HHEM struggles with localized hallucinations in summarization tasks. To address this, we introduce segment-based retrieval, improving detection by verifying smaller text components. Additionally, our cumulative distribution function (CDF) analysis indicates that larger models (7B-9B parameters) generally exhibit fewer hallucinations, while intermediate-sized models show higher instability. These findings highlight the need for structured evaluation frameworks that balance computational efficiency with robust factual validation, enhancing the reliability of LLM-generated content. 

---
# Structured Prompting and LLM Ensembling for Multimodal Conversational Aspect-based Sentiment Analysis 

**Authors**: Zhiqiang Gao, Shihao Gao, Zixing Zhang, Yihao Guo, Hongyu Chen, Jing Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.22603)  

**Abstract**: Understanding sentiment in multimodal conversations is a complex yet crucial challenge toward building emotionally intelligent AI systems. The Multimodal Conversational Aspect-based Sentiment Analysis (MCABSA) Challenge invited participants to tackle two demanding subtasks: (1) extracting a comprehensive sentiment sextuple, including holder, target, aspect, opinion, sentiment, and rationale from multi-speaker dialogues, and (2) detecting sentiment flipping, which detects dynamic sentiment shifts and their underlying triggers. For Subtask-I, in the present paper, we designed a structured prompting pipeline that guided large language models (LLMs) to sequentially extract sentiment components with refined contextual understanding. For Subtask-II, we further leveraged the complementary strengths of three LLMs through ensembling to robustly identify sentiment transitions and their triggers. Our system achieved a 47.38% average score on Subtask-I and a 74.12% exact match F1 on Subtask-II, showing the effectiveness of step-wise refinement and ensemble strategies in rich, multimodal sentiment analysis tasks. 

---
# LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition 

**Authors**: Elsen Ronando, Sozo Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2512.22385)  

**Abstract**: In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar weara-ble sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and $k$-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR. 

---
# Beg to Differ: Understanding Reasoning-Answer Misalignment Across Languages 

**Authors**: Anaelia Ovalle, Candace Ross, Sebastian Ruder, Adina Williams, Karen Ullrich, Mark Ibrahim, Levent Sagun  

**Link**: [PDF](https://arxiv.org/pdf/2512.22712)  

**Abstract**: Large language models demonstrate strong reasoning capabilities through chain-of-thought prompting, but whether this reasoning quality transfers across languages remains underexplored. We introduce a human-validated framework to evaluate whether model-generated reasoning traces logically support their conclusions across languages. Analyzing 65k reasoning traces from GlobalMMLU questions across 6 languages and 6 frontier models, we uncover a critical blind spot: while models achieve high task accuracy, their reasoning can fail to support their conclusions. Reasoning traces in non-Latin scripts show at least twice as much misalignment between their reasoning and conclusions than those in Latin scripts. We develop an error taxonomy through human annotation to characterize these failures, finding they stem primarily from evidential errors (unsupported claims, ambiguous facts) followed by illogical reasoning steps. Our findings demonstrate that current multilingual evaluation practices provide an incomplete picture of model reasoning capabilities and highlight the need for reasoning-aware evaluation frameworks. 

---
# Exploring the Vertical-Domain Reasoning Capabilities of Large Language Models 

**Authors**: Jie Zhou, Xin Chen, Jie Zhang, Zhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.22443)  

**Abstract**: Large Language Models (LLMs) are reshaping learning paradigms, cognitive processes, and research methodologies across a wide range of domains. Integrating LLMs with professional fields and redefining the relationship between LLMs and domain-specific applications has become a critical challenge for promoting enterprise digital transformation and broader social development. To effectively integrate LLMs into the accounting domain, it is essential to understand their domain-specific reasoning capabilities. This study introduces the concept of vertical-domain accounting reasoning and establishes evaluation criteria by analyzing the training data characteristics of representative GLM-series models. These criteria provide a foundation for subsequent research on reasoning paradigms and offer benchmarks for improving accounting reasoning performance. Based on this framework, we evaluate several representative models, including GLM-6B, GLM-130B, GLM-4, and OpenAI GPT-4, on a set of accounting reasoning tasks. Experimental results show that different prompt engineering strategies lead to varying degrees of performance improvement across models, with GPT-4 achieving the strongest accounting reasoning capability. However, current LLMs still fall short of real-world application requirements. In particular, further optimization is needed for deployment in enterprise-level accounting scenarios to fully realize the potential value of LLMs in this domain. 

---
# Chain-of-thought Reviewing and Correction for Time Series Question Answering 

**Authors**: Chen Su, Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.22627)  

**Abstract**: With the advancement of large language models (LLMs), diverse time series analysis tasks are reformulated as time series question answering (TSQA) through a unified natural language interface. However, existing LLM-based approaches largely adopt general natural language processing techniques and are prone to reasoning errors when handling complex numerical sequences. Different from purely textual tasks, time series data are inherently verifiable, enabling consistency checking between reasoning steps and the original input. Motivated by this property, we propose T3LLM, which performs multi-step reasoning with an explicit correction mechanism for time series question answering. The T3LLM framework consists of three LLMs, namely, a worker, a reviewer, and a student, that are responsible for generation, review, and reasoning learning, respectively. Within this framework, the worker generates step-wise chains of thought (CoT) under structured prompts, while the reviewer inspects the reasoning, identifies erroneous steps, and provides corrective comments. The collaboratively generated corrected CoT are used to fine-tune the student model, internalizing multi-step reasoning and self-correction into its parameters. Experiments on multiple real-world TSQA benchmarks demonstrate that T3LLM achieves state-of-the-art performance over strong LLM-based baselines. 

---
# Debugging Tabular Log as Dynamic Graphs 

**Authors**: Chumeng Liang, Zhanyang Jin, Zahaib Akhtar, Mona Pereira, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2512.22903)  

**Abstract**: Tabular log abstracts objects and events in the real-world system and reports their updates to reflect the change of the system, where one can detect real-world inconsistencies efficiently by debugging corresponding log entries. However, recent advances in processing text-enriched tabular log data overly depend on large language models (LLMs) and other heavy-load models, thus suffering from limited flexibility and scalability. This paper proposes a new framework, GraphLogDebugger, to debug tabular log based on dynamic graphs. By constructing heterogeneous nodes for objects and events and connecting node-wise edges, the framework recovers the system behind the tabular log as an evolving dynamic graph. With the help of our dynamic graph modeling, a simple dynamic Graph Neural Network (GNN) is representative enough to outperform LLMs in debugging tabular log, which is validated by experimental results on real-world log datasets of computer systems and academic papers. 

---
# Multimodal Fact-Checking: An Agent-based Approach 

**Authors**: Danni Xu, Shaojing Fan, Xuanang Cheng, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2512.22933)  

**Abstract**: The rapid spread of multimodal misinformation poses a growing challenge for automated fact-checking systems. Existing approaches, including large vision language models (LVLMs) and deep multimodal fusion methods, often fall short due to limited reasoning and shallow evidence utilization. A key bottleneck is the lack of dedicated datasets that provide complete real-world multimodal misinformation instances accompanied by annotated reasoning processes and verifiable evidence. To address this limitation, we introduce RW-Post, a high-quality and explainable dataset for real-world multimodal fact-checking. RW-Post aligns real-world multimodal claims with their original social media posts, preserving the rich contextual information in which the claims are made. In addition, the dataset includes detailed reasoning and explicitly linked evidence, which are derived from human written fact-checking articles via a large language model assisted extraction pipeline, enabling comprehensive verification and explanation. Building upon RW-Post, we propose AgentFact, an agent-based multimodal fact-checking framework designed to emulate the human verification workflow. AgentFact consists of five specialized agents that collaboratively handle key fact-checking subtasks, including strategy planning, high-quality evidence retrieval, visual analysis, reasoning, and explanation generation. These agents are orchestrated through an iterative workflow that alternates between evidence searching and task-aware evidence filtering and reasoning, facilitating strategic decision-making and systematic evidence analysis. Extensive experimental results demonstrate that the synergy between RW-Post and AgentFact substantially improves both the accuracy and interpretability of multimodal fact-checking. 

---
# Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone 

**Authors**: Jiacheng Ye, Shansan Gong, Jiahui Gao, Junming Fan, Shuang Wu, Wei Bi, Haoli Bai, Lifeng Shang, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2512.22615)  

**Abstract**: While autoregressive Large Vision-Language Models (VLMs) have achieved remarkable success, their sequential generation often limits their efficacy in complex visual planning and dynamic robotic control. In this work, we investigate the potential of constructing Vision-Language Models upon diffusion-based large language models (dLLMs) to overcome these limitations. We introduce Dream-VL, an open diffusion-based VLM (dVLM) that achieves state-of-the-art performance among previous dVLMs. Dream-VL is comparable to top-tier AR-based VLMs trained on open data on various benchmarks but exhibits superior potential when applied to visual planning tasks. Building upon Dream-VL, we introduce Dream-VLA, a dLLM-based Vision-Language-Action model (dVLA) developed through continuous pre-training on open robotic datasets. We demonstrate that the natively bidirectional nature of this diffusion backbone serves as a superior foundation for VLA tasks, inherently suited for action chunking and parallel generation, leading to significantly faster convergence in downstream fine-tuning. Dream-VLA achieves top-tier performance of 97.2% average success rate on LIBERO, 71.4% overall average on SimplerEnv-Bridge, and 60.5% overall average on SimplerEnv-Fractal, surpassing leading models such as $\pi_0$ and GR00T-N1. We also validate that dVLMs surpass AR baselines on downstream tasks across different training objectives. We release both Dream-VL and Dream-VLA to facilitate further research in the community. 

---
# Training AI Co-Scientists Using Rubric Rewards 

**Authors**: Shashwat Goel, Rishi Hazra, Dulhan Jayalath, Timon Willi, Parag Jain, William F. Shen, Ilias Leontiadis, Francesco Barbieri, Yoram Bachrach, Jonas Geiping, Chenxi Whitehouse  

**Link**: [PDF](https://arxiv.org/pdf/2512.23707)  

**Abstract**: AI co-scientists are emerging as a tool to assist human researchers in achieving their research goals. A crucial feature of these AI co-scientists is the ability to generate a research plan given a set of aims and constraints. The plan may be used by researchers for brainstorming, or may even be implemented after further refinement. However, language models currently struggle to generate research plans that follow all constraints and implicit requirements. In this work, we study how to leverage the vast corpus of existing research papers to train language models that generate better research plans. We build a scalable, diverse training corpus by automatically extracting research goals and goal-specific grading rubrics from papers across several domains. We then train models for research plan generation via reinforcement learning with self-grading. A frozen copy of the initial policy acts as the grader during training, with the rubrics creating a generator-verifier gap that enables improvements without external human supervision. To validate this approach, we conduct a study with human experts for machine learning research goals, spanning 225 hours. The experts prefer plans generated by our finetuned Qwen3-30B-A3B model over the initial model for 70% of research goals, and approve 84% of the automatically extracted goal-specific grading rubrics. To assess generality, we also extend our approach to research goals from medical papers, and new arXiv preprints, evaluating with a jury of frontier models. Our finetuning yields 12-22% relative improvements and significant cross-domain generalization, proving effective even in problem settings like medical research where execution feedback is infeasible. Together, these findings demonstrate the potential of a scalable, automated training recipe as a step towards improving general AI co-scientists. 

---
# VideoScaffold: Elastic-Scale Visual Hierarchies for Streaming Video Understanding in MLLMs 

**Authors**: Naishan Zheng, Jie Huang, Qingpei Guo, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.22226)  

**Abstract**: Understanding long videos with multimodal large language models (MLLMs) remains challenging due to the heavy redundancy across frames and the need for temporally coherent representations. Existing static strategies, such as sparse sampling, frame compression, and clustering, are optimized for offline settings and often produce fragmented or over-compressed outputs when applied to continuous video streams. We present VideoScaffold, a dynamic representation framework designed for streaming video understanding. It adaptively adjusts event granularity according to video duration while preserving fine-grained visual semantics. VideoScaffold introduces two key components: Elastic-Scale Event Segmentation (EES), which performs prediction-guided segmentation to dynamically refine event boundaries, and Hierarchical Event Consolidation (HEC), which progressively aggregates semantically related segments into multi-level abstractions. Working in concert, EES and HEC enable VideoScaffold to transition smoothly from fine-grained frame understanding to abstract event reasoning as the video stream unfolds. Extensive experiments across both offline and streaming video understanding benchmarks demonstrate that VideoScaffold achieves state-of-the-art performance. The framework is modular and plug-and-play, seamlessly extending existing image-based MLLMs to continuous video comprehension. The code is available at this https URL. 

---
# The Big Three in Marriage Talk: LLM-Assisted Analysis of Moral Ethics and Sentiment on Weibo and Xiaohongshu 

**Authors**: Frank Tian-Fang Ye, Xiaozi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23609)  

**Abstract**: China's marriage registrations have declined dramatically, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. Understanding public attitudes toward marriage requires examining not only emotional sentiment but also the moral reasoning underlying these evaluations. This study analyzed 219,358 marriage-related posts from two major Chinese social media platforms (Sina Weibo and Xiaohongshu) using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral dimensions (Autonomy, Community, Divinity). Results revealed platform differences: Weibo discourse skewed positive, while Xiaohongshu was predominantly neutral. Most posts across both platforms lacked explicit moral framing. However, when moral ethics were invoked, significant associations with sentiment emerged. Posts invoking Autonomy ethics and Community ethics were predominantly negative, whereas Divinity-framed posts tended toward neutral or positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations drive negative marriage attitudes in contemporary China. The study demonstrates LLMs' utility for scaling qualitative analysis and offers insights for developing culturally informed policies addressing marriage decline in Chinese contexts. 

---
# Monadic Context Engineering 

**Authors**: Yifan Zhang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22431)  

**Abstract**: The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in state management, error handling, and concurrency. This paper introduces Monadic Context Engineering (MCE), a novel architectural paradigm leveraging the algebraic structures of Functors, Applicative Functors, and Monads to provide a formal foundation for agent design. MCE treats agent workflows as computational contexts where cross-cutting concerns, such as state propagation, short-circuiting error handling, and asynchronous execution, are managed intrinsically by the algebraic properties of the abstraction. We demonstrate how Monads enable robust sequential composition, how Applicatives provide a principled structure for parallel execution, and crucially, how Monad Transformers allow for the systematic composition of these capabilities. This layered approach enables developers to construct complex, resilient, and efficient AI agents from simple, independently verifiable components. We further extend this framework to describe Meta-Agents, which leverage MCE for generative orchestration, dynamically creating and managing sub-agent workflows through metaprogramming. Project Page: this https URL. 

---
# HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification 

**Authors**: Bhanu Prakash Vangala, Sajid Mahmud, Pawan Neupane, Joel Selvaraj, Jianlin Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.22396)  

**Abstract**: Artificial Intelligence (AI), particularly Large Language Models (LLMs), is transforming scientific discovery, enabling rapid knowledge generation and hypothesis formulation. However, a critical challenge is hallucination, where LLMs generate factually incorrect or misleading information, compromising research integrity. To address this, we introduce HalluMatData, a benchmark dataset for evaluating hallucination detection methods, factual consistency, and response robustness in AI-generated materials science content. Alongside this, we propose HalluMatDetector, a multi-stage hallucination detection framework that integrates intrinsic verification, multi-source retrieval, contradiction graph analysis, and metric-based assessment to detect and mitigate LLM hallucinations. Our findings reveal that hallucination levels vary significantly across materials science subdomains, with high-entropy queries exhibiting greater factual inconsistencies. By utilizing HalluMatDetector verification pipeline, we reduce hallucination rates by 30% compared to standard LLM outputs. Furthermore, we introduce the Paraphrased Hallucination Consistency Score (PHCS) to quantify inconsistencies in LLM responses across semantically equivalent queries, offering deeper insights into model reliability. 

---
# OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation 

**Authors**: Xuegang Hao, Ming Zhang, Alex Li, Xiangyu Qian, Zhi Ma, Yanlong Zang, Shijie Yang, Zhongxuan Han, Xiaolong Ma, Jinguang Liu, Zhen Li, Zhida Jiang, Shusheng Wang, Ning Tang, Yanchen Qiao, Chenxiang Yang, Chen Sun, Jincheng Yuan, Chunhua Peng, Heng Hu, Peijun Yang, Baopeng Yuan, Caiyun Qiu, Zhaolong Xing, Haofei Yuan, Haipeng Zhang, Yuzhang Guo, Weijie Ding, Jiahua Gao, Hao Huang, Zhen Chen, Tongxuan Liu, Pinghua Gong  

**Link**: [PDF](https://arxiv.org/pdf/2512.22386)  

**Abstract**: Traditional recommendation systems suffer from inconsistency in multi-stage optimization objectives. Generative Recommendation (GR) mitigates them through an end-to-end framework; however, existing methods still rely on matching mechanisms based on inductive patterns. Although responsive, they lack the ability to uncover complex user intents that require deductive reasoning based on world knowledge. Meanwhile, LLMs show strong deep reasoning capabilities, but their latency and computational costs remain challenging for industrial applications. More critically, there are performance bottlenecks in multi-scenario scalability: as shown in Figure 1, existing solutions require independent training and deployment for each scenario, leading to low resource utilization and high maintenance costs-a challenge unaddressed in GR literature. To address these, we present OxygenREC, an industrial recommendation system that leverages Fast-Slow Thinking to deliver deep reasoning with strict latency and multi-scenario requirements of real-world environments. First, we adopt a Fast-Slow Thinking architecture. Slow thinking uses a near-line LLM pipeline to synthesize Contextual Reasoning Instructions, while fast thinking employs a high-efficiency encoder--decoder backbone for real-time generation. Second, to ensure reasoning instructions effectively enhance recommendation generation, we introduce a semantic alignment mechanism with Instruction-Guided Retrieval (IGR) to filter intent-relevant historical behaviors and use a Query-to-Item (Q2I) loss for instruction-item consistency. Finally, to resolve multi-scenario scalability, we transform scenario information into controllable instructions, using unified reward mapping and Soft Adaptive Group Clip Policy Optimization (SA-GCPO) to align policies with diverse business objectives, realizing a train-once-deploy-everywhere paradigm. 

---
# Divergent-Convergent Thinking in Large Language Models for Creative Problem Generation 

**Authors**: Manh Hung Nguyen, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2512.23601)  

**Abstract**: Large language models (LLMs) have significant potential for generating educational questions and problems, enabling educators to create large-scale learning materials. However, LLMs are fundamentally limited by the ``Artificial Hivemind'' effect, where they generate similar responses within the same model and produce homogeneous outputs across different models. As a consequence, students may be exposed to overly similar and repetitive LLM-generated problems, which harms diversity of thought. Drawing inspiration from Wallas's theory of creativity and Guilford's framework of divergent-convergent thinking, we propose CreativeDC, a two-phase prompting method that explicitly scaffolds the LLM's reasoning into distinct phases. By decoupling creative exploration from constraint satisfaction, our method enables LLMs to explore a broader space of ideas before committing to a final problem. We evaluate CreativeDC for creative problem generation using a comprehensive set of metrics that capture diversity, novelty, and utility. The results show that CreativeDC achieves significantly higher diversity and novelty compared to baselines while maintaining high utility. Moreover, scaling analysis shows that CreativeDC generates a larger effective number of distinct problems as more are sampled, increasing at a faster rate than baseline methods. 

---
# SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search 

**Authors**: Yifan Zhang, Giridhar Ganapavarapu, Srideepika Jayaraman, Bhavna Agrawal, Dhaval Patel, Achille Fokoue  

**Link**: [PDF](https://arxiv.org/pdf/2512.23167)  

**Abstract**: Large Language Models (LLMs) often falter at complex planning tasks that require exploration and self-correction, as their linear reasoning process struggles to recover from early mistakes. While search algorithms like Monte Carlo Tree Search (MCTS) can explore alternatives, they are often ineffective when guided by sparse rewards and fail to leverage the rich semantic capabilities of LLMs. We introduce SPIRAL (Symbolic LLM Planning via Grounded and Reflective Search), a novel framework that embeds a cognitive architecture of three specialized LLM agents into an MCTS loop. SPIRAL's key contribution is its integrated planning pipeline where a Planner proposes creative next steps, a Simulator grounds the search by predicting realistic outcomes, and a Critic provides dense reward signals through reflection. This synergy transforms MCTS from a brute-force search into a guided, self-correcting reasoning process. On the DailyLifeAPIs and HuggingFace datasets, SPIRAL consistently outperforms the default Chain-of-Thought planning method and other state-of-the-art agents. More importantly, it substantially surpasses other state-of-the-art agents; for example, SPIRAL achieves 83.6% overall accuracy on DailyLifeAPIs, an improvement of over 16 percentage points against the next-best search framework, while also demonstrating superior token efficiency. Our work demonstrates that structuring LLM reasoning as a guided, reflective, and grounded search process yields more robust and efficient autonomous planners. The source code, full appendices, and all experimental data are available for reproducibility at the official project repository. 

---
# From Model Choice to Model Belief: Establishing a New Measure for LLM-Based Research 

**Authors**: Hongshen Sun, Juanjuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23184)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human behavior, but common practices to use LLM-generated data are inefficient. Treating an LLM's output ("model choice") as a single data point underutilizes the information inherent to the probabilistic nature of LLMs. This paper introduces and formalizes "model belief," a measure derived from an LLM's token-level probabilities that captures the model's belief distribution over choice alternatives in a single generation run. The authors prove that model belief is asymptotically equivalent to the mean of model choices (a non-trivial property) but forms a more statistically efficient estimator, with lower variance and a faster convergence rate. Analogous properties are shown to hold for smooth functions of model belief and model choice often used in downstream applications. The authors demonstrate the performance of model belief through a demand estimation study, where an LLM simulates consumer responses to different prices. In practical settings with limited numbers of runs, model belief explains and predicts ground-truth model choice better than model choice itself, and reduces the computation needed to reach sufficiently accurate estimates by roughly a factor of 20. The findings support using model belief as the default measure to extract more information from LLM-generated data. 

---
# The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction 

**Authors**: Haoyu Pei, Zhongyang Liu, Xiangyi Xiao, Xiaocong Du, Haipeng Zhang, Kunpeng Zhang, Suting Hong  

**Link**: [PDF](https://arxiv.org/pdf/2512.23489)  

**Abstract**: Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: this https URL. 

---
# TCEval: Using Thermal Comfort to Assess Cognitive and Perceptual Abilities of AI 

**Authors**: Jingming Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.23217)  

**Abstract**: A critical gap exists in LLM task-specific benchmarks. Thermal comfort, a sophisticated interplay of environmental factors and personal perceptions involving sensory integration and adaptive decision-making, serves as an ideal paradigm for evaluating real-world cognitive capabilities of AI systems. To address this, we propose TCEval, the first evaluation framework that assesses three core cognitive capacities of AI, cross-modal reasoning, causal association, and adaptive decision-making, by leveraging thermal comfort scenarios and large language model (LLM) agents. The methodology involves initializing LLM agents with virtual personality attributes, guiding them to generate clothing insulation selections and thermal comfort feedback, and validating outputs against the ASHRAE Global Database and Chinese Thermal Comfort Database. Experiments on four LLMs show that while agent feedback has limited exact alignment with humans, directional consistency improves significantly with a 1 PMV tolerance. Statistical tests reveal that LLM-generated PMV distributions diverge markedly from human data, and agents perform near-randomly in discrete thermal comfort classification. These results confirm the feasibility of TCEval as an ecologically valid Cognitive Turing Test for AI, demonstrating that current LLMs possess foundational cross-modal reasoning ability but lack precise causal understanding of the nonlinear relationships between variables in thermal comfort. TCEval complements traditional benchmarks, shifting AI evaluation focus from abstract task proficiency to embodied, context-aware perception and decision-making, offering valuable insights for advancing AI in human-centric applications like smart buildings. 

---
# HiSciBench: A Hierarchical Multi-disciplinary Benchmark for Scientific Intelligence from Reading to Discovery 

**Authors**: Yaping Zhang, Qixuan Zhang, Xingquan Zhang, Zhiyuan Chen, Wenwen Zhuang, Yupu Liang, Lu Xiang, Yang Zhao, Jiajun Zhang, Yu Zhou, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2512.22899)  

**Abstract**: The rapid advancement of large language models (LLMs) and multimodal foundation models has sparked growing interest in their potential for scientific research. However, scientific intelligence encompasses a broad spectrum of abilities ranging from understanding fundamental knowledge to conducting creative discovery, and existing benchmarks remain fragmented. Most focus on narrow tasks and fail to reflect the hierarchical and multi-disciplinary nature of real scientific inquiry. We introduce \textbf{HiSciBench}, a hierarchical benchmark designed to evaluate foundation models across five levels that mirror the complete scientific workflow: \textit{Scientific Literacy} (L1), \textit{Literature Parsing} (L2), \textit{Literature-based Question Answering} (L3), \textit{Literature Review Generation} (L4), and \textit{Scientific Discovery} (L5). HiSciBench contains 8,735 carefully curated instances spanning six major scientific disciplines, including mathematics, physics, chemistry, biology, geography, and astronomy, and supports multimodal inputs including text, equations, figures, and tables, as well as cross-lingual evaluation. Unlike prior benchmarks that assess isolated abilities, HiSciBench provides an integrated, dependency-aware framework that enables detailed diagnosis of model capabilities across different stages of scientific reasoning. Comprehensive evaluations of leading models, including GPT-5, DeepSeek-R1, and several multimodal systems, reveal substantial performance gaps: while models achieve up to 69\% accuracy on basic literacy tasks, performance declines sharply to 25\% on discovery-level challenges. HiSciBench establishes a new standard for evaluating scientific Intelligence and offers actionable insights for developing models that are not only more capable but also more reliable. The benchmark will be publicly released to facilitate future research. 

---
# TravelBench: A Real-World Benchmark for Multi-Turn and Tool-Augmented Travel Planning 

**Authors**: Xiang Cheng, Yulan Hu, Xiangwen Zhang, Lu Xu, Zheng Pan, Xin Li, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22673)  

**Abstract**: Large language model (LLM) agents have demonstrated strong capabilities in planning and tool use. Travel planning provides a natural and high-impact testbed for these capabilities, as it requires multi-step reasoning, iterative preference elicitation through interaction, and calls to external tools under evolving constraints. Prior work has studied LLMs on travel-planning tasks, but existing settings are limited in domain coverage and multi-turn interaction. As a result, they cannot support dynamic user-agent interaction and therefore fail to comprehensively assess agent capabilities. In this paper, we introduce TravelBench, a real-world travel-planning benchmark featuring multi-turn interaction and tool use. We collect user requests from real-world scenarios and construct three subsets-multi-turn, single-turn, and unsolvable-to evaluate different aspects of agent performance. For stable and reproducible evaluation, we build a controlled sandbox environment with 10 travel-domain tools, providing deterministic tool outputs for reliable reasoning. We evaluate multiple LLMs on TravelBench and conduct an analysis of their behaviors and performance. TravelBench offers a practical and reproducible benchmark for advancing LLM agents in travel planning. 

---
# The Wisdom of Deliberating AI Crowds: Does Deliberation Improve LLM-Based Forecasting? 

**Authors**: Paul Schneider, Amalie Schramm  

**Link**: [PDF](https://arxiv.org/pdf/2512.22625)  

**Abstract**: Structured deliberation has been found to improve the performance of human forecasters. This study investigates whether a similar intervention, i.e. allowing LLMs to review each other's forecasts before updating, can improve accuracy in large language models (GPT-5, Claude Sonnet 4.5, Gemini Pro 2.5). Using 202 resolved binary questions from the Metaculus Q2 2025 AI Forecasting Tournament, accuracy was assessed across four scenarios: (1) diverse models with distributed information, (2) diverse models with shared information, (3) homogeneous models with distributed information, and (4) homogeneous models with shared information. Results show that the intervention significantly improves accuracy in scenario (2), reducing Log Loss by 0.020 or about 4 percent in relative terms (p = 0.017). However, when homogeneous groups (three instances of the same model) engaged in the same process, no benefit was observed. Unexpectedly, providing LLMs with additional contextual information did not improve forecast accuracy, limiting our ability to study information pooling as a mechanism. Our findings suggest that deliberation may be a viable strategy for improving LLM forecasting. 

---
# LLM Agents as VC investors: Predicting Startup Success via RolePlay-Based Collective Simulation 

**Authors**: Zhongyang Liu, Haoyu Pei, Xiangyi Xiao, Xiaocong Du, Yihui Li, Suting Hong, Kunpeng Zhang, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22608)  

**Abstract**: Due to the high value and high failure rate of startups, predicting their success has become a critical challenge across interdisciplinary research. Existing approaches typically model success prediction from the perspective of a single decision-maker, overlooking the collective dynamics of investor groups that dominate real-world venture capital (VC) decisions. In this paper, we propose SimVC-CAS, a novel collective agent system that simulates VC decision-making as a multi-agent interaction process. By designing role-playing agents and a GNN-based supervised interaction module, we reformulate startup financing prediction as a group decision-making task, capturing both enterprise fundamentals and the behavioral dynamics of potential investor networks. Each agent embodies an investor with unique traits and preferences, enabling heterogeneous evaluation and realistic information exchange through a graph-structured co-investment network. Using real-world data from PitchBook and under strict data leakage controls, we show that SimVC-CAS significantly improves predictive accuracy while providing interpretable, multiperspective reasoning, for example, approximately 25% relative improvement with respect to average precision@10. SimVC-CAS also sheds light on other complex group decision scenarios. 

---
# Learning Multi-Modal Mobility Dynamics for Generalized Next Location Recommendation 

**Authors**: Junshu Dai, Yu Wang, Tongya Zheng, Wei Ji, Qinghong Guo, Ji Cao, Jie Song, Canghong Jin, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.22605)  

**Abstract**: The precise prediction of human mobility has produced significant socioeconomic impacts, such as location recommendations and evacuation suggestions. However, existing methods suffer from limited generalization capability: unimodal approaches are constrained by data sparsity and inherent biases, while multi-modal methods struggle to effectively capture mobility dynamics caused by the semantic gap between static multi-modal representation and spatial-temporal dynamics. Therefore, we leverage multi-modal spatial-temporal knowledge to characterize mobility dynamics for the location recommendation task, dubbed as \textbf{M}ulti-\textbf{M}odal \textbf{Mob}ility (\textbf{M}$^3$\textbf{ob}). First, we construct a unified spatial-temporal relational graph (STRG) for multi-modal representation, by leveraging the functional semantics and spatial-temporal knowledge captured by the large language models (LLMs)-enhanced spatial-temporal knowledge graph (STKG). Second, we design a gating mechanism to fuse spatial-temporal graph representations of different modalities, and propose an STKG-guided cross-modal alignment to inject spatial-temporal dynamic knowledge into the static image modality. Extensive experiments on six public datasets show that our proposed method not only achieves consistent improvements in normal scenarios but also exhibits significant generalization ability in abnormal scenarios. 

---
# Logic Sketch Prompting (LSP): A Deterministic and Interpretable Prompting Method 

**Authors**: Satvik Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2512.22258)  

**Abstract**: Large language models (LLMs) excel at natural language reasoning but remain unreliable on tasks requiring strict rule adherence, determinism, and auditability. Logic Sketch Prompting (LSP) is a lightweight prompting framework that introduces typed variables, deterministic condition evaluators, and a rule based validator that produces traceable and repeatable outputs. Using two pharmacologic logic compliance tasks, we benchmark LSP against zero shot prompting, chain of thought prompting, and concise prompting across three open weight models: Gemma 2, Mistral, and Llama 3. Across both tasks and all models, LSP consistently achieves the highest accuracy (0.83 to 0.89) and F1 score (0.83 to 0.89), substantially outperforming zero shot prompting (0.24 to 0.60), concise prompts (0.16 to 0.30), and chain of thought prompting (0.56 to 0.75). McNemar tests show statistically significant gains for LSP across nearly all comparisons (p < 0.01). These results demonstrate that LSP improves determinism, interpretability, and consistency without sacrificing performance, supporting its use in clinical, regulated, and safety critical decision support systems. 

---
# DarkPatterns-LLM: A Multi-Layer Benchmark for Detecting Manipulative and Harmful AI Behavior 

**Authors**: Sadia Asif, Israel Antonio Rosales Laguan, Haris Khan, Shumaila Asif, Muneeb Asif  

**Link**: [PDF](https://arxiv.org/pdf/2512.22470)  

**Abstract**: The proliferation of Large Language Models (LLMs) has intensified concerns about manipulative or deceptive behaviors that can undermine user autonomy, trust, and well-being. Existing safety benchmarks predominantly rely on coarse binary labels and fail to capture the nuanced psychological and social mechanisms constituting manipulation. We introduce \textbf{DarkPatterns-LLM}, a comprehensive benchmark dataset and diagnostic framework for fine-grained assessment of manipulative content in LLM outputs across seven harm categories: Legal/Power, Psychological, Emotional, Physical, Autonomy, Economic, and Societal Harm. Our framework implements a four-layer analytical pipeline comprising Multi-Granular Detection (MGD), Multi-Scale Intent Analysis (MSIAN), Threat Harmonization Protocol (THP), and Deep Contextual Risk Alignment (DCRA). The dataset contains 401 meticulously curated examples with instruction-response pairs and expert annotations. Through evaluation of state-of-the-art models including GPT-4, Claude 3.5, and LLaMA-3-70B, we observe significant performance disparities (65.2\%--89.7\%) and consistent weaknesses in detecting autonomy-undermining patterns. DarkPatterns-LLM establishes the first standardized, multi-dimensional benchmark for manipulation detection in LLMs, offering actionable diagnostics toward more trustworthy AI systems. 

---
# InSPO: Unlocking Intrinsic Self-Reflection for LLM Preference Optimization 

**Authors**: Yu Li, Tian Lan, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2512.23126)  

**Abstract**: Direct Preference Optimization (DPO) and its variants have become standard for aligning Large Language Models due to their simplicity and offline stability. However, we identify two fundamental limitations. First, the optimal policy depends on arbitrary modeling choices (scalarization function, reference policy), yielding behavior reflecting parameterization artifacts rather than true preferences. Second, treating response generation in isolation fails to leverage comparative information in pairwise data, leaving the model's capacity for intrinsic self-reflection untapped. To address it, we propose Intrinsic Self-reflective Preference Optimization (\q), deriving a globally optimal policy conditioning on both context and alternative responses. We prove this formulation superior to DPO/RLHF while guaranteeing invariance to scalarization and reference choices. \q~serves as a plug-and-play enhancement without architectural changes or inference overhead. Experiments demonstrate consistent improvements in win rates and length-controlled metrics, validating that unlocking self-reflection yields more robust, human-aligned LLMs. 

---
# Shape of Thought: When Distribution Matters More than Correctness in Reasoning Tasks 

**Authors**: Abhranil Chandra, Ayush Agrawal, Arian Hosseini, Sebastian Fischmeister, Rishabh Agarwal, Navin Goyal, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2512.22255)  

**Abstract**: We present the surprising finding that a language model's reasoning capabilities can be improved by training on synthetic datasets of chain-of-thought (CoT) traces from more capable models, even when all of those traces lead to an incorrect final answer. Our experiments show this approach can yield better performance on reasoning tasks than training on human-annotated datasets. We hypothesize that two key factors explain this phenomenon: first, the distribution of synthetic data is inherently closer to the language model's own distribution, making it more amenable to learning. Second, these `incorrect' traces are often only partially flawed and contain valid reasoning steps from which the model can learn. To further test the first hypothesis, we use a language model to paraphrase human-annotated traces -- shifting their distribution closer to the model's own distribution -- and show that this improves performance. For the second hypothesis, we introduce increasingly flawed CoT traces and study to what extent models are tolerant to these flaws. We demonstrate our findings across various reasoning domains like math, algorithmic reasoning and code generation using MATH, GSM8K, Countdown and MBPP datasets on various language models ranging from 1.5B to 9B across Qwen, Llama, and Gemma models. Our study shows that curating datasets that are closer to the model's distribution is a critical aspect to consider. We also show that a correct final answer is not always a reliable indicator of a faithful reasoning process. 

---
# Emergent Persuasion: Will LLMs Persuade Without Being Prompted? 

**Authors**: Vincent Chang, Thee Ho, Sunishchal Dev, Kevin Zhu, Shi Feng, Kellin Pelrine, Matthew Kowal  

**Link**: [PDF](https://arxiv.org/pdf/2512.22201)  

**Abstract**: With the wide-scale adoption of conversational AI systems, AI are now able to exert unprecedented influence on human opinion and beliefs. Recent work has shown that many Large Language Models (LLMs) comply with requests to persuade users into harmful beliefs or actions when prompted and that model persuasiveness increases with model scale. However, this prior work looked at persuasion from the threat model of $\textit{misuse}$ (i.e., a bad actor asking an LLM to persuade). In this paper, we instead aim to answer the following question: Under what circumstances would models persuade $\textit{without being explicitly prompted}$, which would shape how concerned we should be about such emergent persuasion risks. To achieve this, we study unprompted persuasion under two scenarios: (i) when the model is steered (through internal activation steering) along persona traits, and (ii) when the model is supervised-finetuned (SFT) to exhibit the same traits. We showed that steering towards traits, both related to persuasion and unrelated, does not reliably increase models' tendency to persuade unprompted, however, SFT does. Moreover, SFT on general persuasion datasets containing solely benign topics admits a model that has a higher propensity to persuade on controversial and harmful topics--showing that emergent harmful persuasion can arise and should be studied further. 

---
# AI tutoring can safely and effectively support students: An exploratory RCT in UK classrooms 

**Authors**: LearnLM Team Google, Eedi, Albert Wang, Aliya Rysbek, Andrea Huber, Anjali Nambiar, Anna Kenolty, Ben Caulfield, Beth Lilley-Draper, Bibi Groot, Brian Veprek, Chelsea Burdett, Claire Willis, Craig Barton, Digory Smith, George Mu, Harriet Walters, Irina Jurenka, Iris Hulls, James Stalley-Moores, Jonathan Caton, Julia Wilkowski, Kaiz Alarakyia, Kevin R. McKee, Liam McCafferty, Lucy Dalton, Markus Kunesch, Pauline Malubay, Rachel Kidson, Rich Wells, Sam Wheeler, Sara Wiltberger, Shakir Mohamed, Simon Woodhead, Vasco Brazão  

**Link**: [PDF](https://arxiv.org/pdf/2512.23633)  

**Abstract**: One-to-one tutoring is widely considered the gold standard for personalized education, yet it remains prohibitively expensive to scale. To evaluate whether generative AI might help expand access to this resource, we conducted an exploratory randomized controlled trial (RCT) with $N = 165$ students across five UK secondary schools. We integrated LearnLM -- a generative AI model fine-tuned for pedagogy -- into chat-based tutoring sessions on the Eedi mathematics platform. In the RCT, expert tutors directly supervised LearnLM, with the remit to revise each message it drafted until they would be satisfied sending it themselves. LearnLM proved to be a reliable source of pedagogical instruction, with supervising tutors approving 76.4% of its drafted messages making zero or minimal edits (i.e., changing only one or two characters). This translated into effective tutoring support: students guided by LearnLM performed at least as well as students chatting with human tutors on each learning outcome we measured. In fact, students who received support from LearnLM were 5.5 percentage points more likely to solve novel problems on subsequent topics (with a success rate of 66.2%) than those who received tutoring from human tutors alone (rate of 60.7%). In interviews, tutors highlighted LearnLM's strength at drafting Socratic questions that encouraged deeper reflection from students, with multiple tutors even reporting that they learned new pedagogical practices from the model. Overall, our results suggest that pedagogically fine-tuned AI tutoring systems may play a promising role in delivering effective, individualized learning support at scale. 

---
# RxnBench: A Multimodal Benchmark for Evaluating Large Language Models on Chemical Reaction Understanding from Scientific Literature 

**Authors**: Hanzheng Li, Xi Fang, Yixuan Li, Chaozheng Huang, Junjie Wang, Xi Wang, Hongzhe Bai, Bojun Hao, Shenyu Lin, Huiqi Liang, Linfeng Zhang, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2512.23565)  

**Abstract**: The integration of Multimodal Large Language Models (MLLMs) into chemistry promises to revolutionize scientific discovery, yet their ability to comprehend the dense, graphical language of reactions within authentic literature remains underexplored. Here, we introduce RxnBench, a multi-tiered benchmark designed to rigorously evaluate MLLMs on chemical reaction understanding from scientific PDFs. RxnBench comprises two tasks: Single-Figure QA (SF-QA), which tests fine-grained visual perception and mechanistic reasoning using 1,525 questions derived from 305 curated reaction schemes, and Full-Document QA (FD-QA), which challenges models to synthesize information from 108 articles, requiring cross-modal integration of text, schemes, and tables. Our evaluation of MLLMs reveals a critical capability gap: while models excel at extracting explicit text, they struggle with deep chemical logic and precise structural recognition. Notably, models with inference-time reasoning significantly outperform standard architectures, yet none achieve 50\% accuracy on FD-QA. These findings underscore the urgent need for domain-specific visual encoders and stronger reasoning engines to advance autonomous AI chemists. 

---
# Toward Trustworthy Agentic AI: A Multimodal Framework for Preventing Prompt Injection Attacks 

**Authors**: Toqeer Ali Syed, Mishal Ateeq Almutairi, Mahmoud Abdel Moaty  

**Link**: [PDF](https://arxiv.org/pdf/2512.23557)  

**Abstract**: Powerful autonomous systems, which reason, plan, and converse using and between numerous tools and agents, are made possible by Large Language Models (LLMs), Vision-Language Models (VLMs), and new agentic AI systems, like LangChain and GraphChain. Nevertheless, this agentic environment increases the probability of the occurrence of multimodal prompt injection (PI) attacks, in which concealed or malicious instructions carried in text, pictures, metadata, or agent-to-agent messages may spread throughout the graph and lead to unintended behavior, a breach of policy, or corruption of state. In order to mitigate these risks, this paper suggests a Cross-Agent Multimodal Provenanc- Aware Defense Framework whereby all the prompts, either user-generated or produced by upstream agents, are sanitized and all the outputs generated by an LLM are verified independently before being sent to downstream nodes. This framework contains a Text sanitizer agent, visual sanitizer agent, and output validator agent all coordinated by a provenance ledger, which keeps metadata of modality, source, and trust level throughout the entire agent network. This architecture makes sure that agent-to-agent communication abides by clear trust frames such such that injected instructions are not propagated down LangChain or GraphChain-style-workflows. The experimental assessments show that multimodal injection detection accuracy is significantly enhanced, and the cross-agent trust leakage is minimized, as well as, agentic execution pathways become stable. The framework, which expands the concept of provenance tracking and validation to the multi-agent orchestration, enhances the establishment of secure, understandable and reliable agentic AI systems. 

---
# Agentic AI for Autonomous Defense in Software Supply Chain Security: Beyond Provenance to Vulnerability Mitigation 

**Authors**: Toqeer Ali Syed, Mohammad Riyaz Belgaum, Salman Jan, Asadullah Abdullah Khan, Saad Said Alqahtani  

**Link**: [PDF](https://arxiv.org/pdf/2512.23480)  

**Abstract**: The software supply chain attacks are becoming more and more focused on trusted development and delivery procedures, so the conventional post-build integrity mechanisms cannot be used anymore. The available frameworks like SLSA, SBOM and in toto are majorly used to offer provenance and traceability but do not have the capabilities of actively identifying and removing vulnerabilities in software production. The current paper includes an example of agentic artificial intelligence (AI) based on autonomous software supply chain security that combines large language model (LLM)-based reasoning, reinforcement learning (RL), and multi-agent coordination. The suggested system utilizes specialized security agents coordinated with the help of LangChain and LangGraph, communicates with actual CI/CD environments with the Model Context Protocol (MCP), and documents all the observations and actions in a blockchain security ledger to ensure integrity and auditing. Reinforcement learning can be used to achieve adaptive mitigation strategies that consider the balance between security effectiveness and the operational overhead, and LLMs can be used to achieve semantic vulnerability analysis, as well as explainable decisions. This framework is tested based on simulated pipelines, as well as, actual world CI/CD integrations on GitHub Actions and Jenkins, including injection attacks, insecure deserialization, access control violations, and configuration errors. Experimental outcomes indicate better detection accuracy, shorter mitigation latency and reasonable build-time overhead than rule-based, provenance only and RL only baselines. These results show that agentic AI can facilitate the transition to self defending, proactive software supply chains rather than reactive verification ones. 

---
# Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning 

**Authors**: Zuoyou Jiang, Li Zhao, Rui Sun, Ruohan Sun, Zhongjian Li, Jing Li, Daxin Jiang, Zuo Bai, Cheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2512.23515)  

**Abstract**: Signal decay and regime shifts pose recurring challenges for data-driven investment strategies in non-stationary markets. Conventional time-series and machine learning approaches, which rely primarily on historical correlations, often struggle to generalize when the economic environment changes. While large language models (LLMs) offer strong capabilities for processing unstructured information, their potential to support quantitative factor screening through explicit economic reasoning remains underexplored. Existing factor-based methods typically reduce alphas to numerical time series, overlooking the semantic rationale that determines when a factor is economically relevant. We propose Alpha-R1, an 8B-parameter reasoning model trained via reinforcement learning for context-aware alpha screening. Alpha-R1 reasons over factor logic and real-time news to evaluate alpha relevance under changing market conditions, selectively activating or deactivating factors based on contextual consistency. Empirical results across multiple asset pools show that Alpha-R1 consistently outperforms benchmark strategies and exhibits improved robustness to alpha decay. The full implementation and resources are available at this https URL. 

---
# The Law of Multi-Model Collaboration: Scaling Limits of Model Ensembling for Large Language Models 

**Authors**: Dakuan Lu, Jiaqi Zhang, Cheng Yuan, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.23340)  

**Abstract**: Recent advances in large language models (LLMs) have been largely driven by scaling laws for individual models, which predict performance improvements as model parameters and data volume increase. However, the capabilities of any single LLM are inherently bounded. One solution originates from intricate interactions among multiple LLMs, rendering their collective performance surpasses that of any constituent model. Despite the rapid proliferation of multi-model integration techniques such as model routing and post-hoc ensembling, a unifying theoretical framework of performance scaling for multi-model collaboration remains absent. In this work, we propose the Law of Multi-model Collaboration, a scaling law that predicts the performance limits of LLM ensembles based on their aggregated parameter budget. To quantify the intrinsic upper bound of multi-model collaboration, we adopt a method-agnostic formulation and assume an idealized integration oracle where the total cross-entropy loss of each sample is determined by the minimum loss of any model in the model pool. Experimental results reveal that multi-model systems follow a power-law scaling with respect to the total parameter count, exhibiting a more significant improvement trend and a lower theoretical loss floor compared to single model scaling. Moreover, ensembles of heterogeneous model families achieve better performance scaling than those formed within a single model family, indicating that model diversity is a primary driver of collaboration gains. These findings suggest that model collaboration represents a critical axis for extending the intelligence frontier of LLMs. 

---
# Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL 

**Authors**: Abolfazl Younesi, Abbas Shabrang Maryan, Elyas Oustad, Zahra Najafabadi Samani, Mohsen Ansari, Thomas Fahringer  

**Link**: [PDF](https://arxiv.org/pdf/2512.23310)  

**Abstract**: Deploying large language models (LLMs) on edge devices is challenging due to their limited memory and power resources. Cloud-only inference reduces device burden but introduces high latency and cost. Static edge-cloud partitions optimize a single metric and struggle when bandwidth fluctuates. We propose Splitwise, a novel Lyapunov-assisted deep reinforcement learning (DRL) framework for fine-grained, adaptive partitioning of LLMs across edge and cloud environments. Splitwise decomposes transformer layers into attention heads and feed-forward sub-blocks, exposing more partition choices than layer-wise schemes. A hierarchical DRL policy, guided by Lyapunov optimization, jointly minimizes latency, energy consumption, and accuracy degradation while guaranteeing queue stability under stochastic workloads and variable network bandwidth. Splitwise also guarantees robustness via partition checkpoints with exponential backoff recovery in case of communication failures. Experiments on Jetson Orin NX, Galaxy S23, and Raspberry Pi 5 with GPT-2 (1.5B), LLaMA-7B, and LLaMA-13B show that Splitwise reduces end-to-end latency by 1.4x-2.8x and cuts energy consumption by up to 41% compared with existing partitioners. It lowers the 95th-percentile latency by 53-61% relative to cloud-only execution, while maintaining accuracy and modest memory requirements. 

---
# Reservoir Computing inspired Matrix Multiplication-free Language Model 

**Authors**: Takumi Shiratsuchi, Yuichiro Tanaka, Hakaru Tamukoh  

**Link**: [PDF](https://arxiv.org/pdf/2512.23145)  

**Abstract**: Large language models (LLMs) have achieved state-of-the-art performance in natural language processing; however, their high computational cost remains a major bottleneck. In this study, we target computational efficiency by focusing on a matrix multiplication free language model (MatMul-free LM) and further reducing the training cost through an architecture inspired by reservoir computing. Specifically, we partially fix and share the weights of selected layers in the MatMul-free LM and insert reservoir layers to obtain rich dynamic representations without additional training overhead. Additionally, several operations are combined to reduce memory accesses. Experimental results show that the proposed architecture reduces the number of parameters by up to 19%, training time by 9.9%, and inference time by 8.0%, while maintaining comparable performance to the baseline model. 

---
# Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware 

**Authors**: Alex Khalil, Guillaume Heilles, Maria Parraga, Simon Heilles  

**Link**: [PDF](https://arxiv.org/pdf/2512.23029)  

**Abstract**: The proliferation of Large Language Models (LLMs) has been accompanied by a reliance on cloud-based, proprietary systems, raising significant concerns regarding data privacy, operational sovereignty, and escalating costs. This paper investigates the feasibility of deploying a high-performance, private LLM inference server at a cost accessible to Small and Medium Businesses (SMBs). We present a comprehensive benchmarking analysis of a locally hosted, quantized 30-billion parameter Mixture-of-Experts (MoE) model based on Qwen3, running on a consumer-grade server equipped with a next-generation NVIDIA GPU. Unlike cloud-based offerings, which are expensive and complex to integrate, our approach provides an affordable and private solution for SMBs. We evaluate two dimensions: the model's intrinsic capabilities and the server's performance under load. Model performance is benchmarked against academic and industry standards to quantify reasoning and knowledge relative to cloud services. Concurrently, we measure server efficiency through latency, tokens per second, and time to first token, analyzing scalability under increasing concurrent users. Our findings demonstrate that a carefully configured on-premises setup with emerging consumer hardware and a quantized open-source model can achieve performance comparable to cloud-based services, offering SMBs a viable pathway to deploy powerful LLMs without prohibitive costs or privacy compromises. 

---
# EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion 

**Authors**: Zhen Liang, Hai Huang, Zhengkui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.23173)  

**Abstract**: Large language models (LLMs), such as ChatGPT, have achieved remarkable success across a wide range of fields. However, their trustworthiness remains a significant concern, as they are still susceptible to jailbreak attacks aimed at eliciting inappropriate or harmful responses. However, existing jailbreak attacks mainly operate at the natural language level and rely on a single attack strategy, limiting their effectiveness in comprehensively assessing LLM robustness. In this paper, we propose Equacode, a novel multi-strategy jailbreak approach for large language models via equation-solving and code completion. This approach transforms malicious intent into a mathematical problem and then requires the LLM to solve it using code, leveraging the complexity of cross-domain tasks to divert the model's focus toward task completion rather than safety constraints. Experimental results show that Equacode achieves an average success rate of 91.19% on the GPT series and 98.65% across 3 state-of-the-art LLMs, all with only a single query. Further, ablation experiments demonstrate that EquaCode outperforms either the mathematical equation module or the code module alone. This suggests a strong synergistic effect, thereby demonstrating that multi-strategy approach yields results greater than the sum of its parts. 

---
# Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning 

**Authors**: Yingru Li, Jiawei Xu, Jiacai Liu, Yuxuan Tong, Ziniu Li, Tianle Cai, Ge Zhang, Qian Liu, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23087)  

**Abstract**: Reinforcement learning for large language models (LLMs) faces a fundamental tension: high-throughput inference engines and numerically-precise training systems produce different probability distributions from the same parameters, creating a training-inference mismatch. We prove this mismatch has an asymmetric effect: the bound on log-probability mismatch scales as $(1-p)$ where $p$ is the token probability. For high-probability tokens, this bound vanishes, contributing negligibly to sequence-level mismatch. For low-probability tokens in the tail, the bound remains large, and moreover, when sampled, these tokens exhibit systematically biased mismatches that accumulate over sequences, destabilizing gradient estimation. Rather than applying post-hoc corrections, we propose constraining the RL objective to a dynamically-pruned ``safe'' vocabulary that excludes the extreme tail. By pruning such tokens, we trade large, systematically biased mismatches for a small, bounded optimization bias. Empirically, our method achieves stable training; theoretically, we bound the optimization bias introduced by vocabulary pruning. 

---
# FasterPy: An LLM-based Code Execution Efficiency Optimization Framework 

**Authors**: Yue Wu, Minghao Han, Ruiyin Li, Peng Liang, Amjed Tahir, Zengyang Li, Qiong Feng, Mojtaba Shahin  

**Link**: [PDF](https://arxiv.org/pdf/2512.22827)  

**Abstract**: Code often suffers from performance bugs. These bugs necessitate the research and practice of code optimization. Traditional rule-based methods rely on manually designing and maintaining rules for specific performance bugs (e.g., redundant loops, repeated computations), making them labor-intensive and limited in applicability. In recent years, machine learning and deep learning-based methods have emerged as promising alternatives by learning optimization heuristics from annotated code corpora and performance measurements. However, these approaches usually depend on specific program representations and meticulously crafted training datasets, making them costly to develop and difficult to scale. With the booming of Large Language Models (LLMs), their remarkable capabilities in code generation have opened new avenues for automated code optimization. In this work, we proposed FasterPy, a low-cost and efficient framework that adapts LLMs to optimize the execution efficiency of Python code. FasterPy combines Retrieval-Augmented Generation (RAG), supported by a knowledge base constructed from existing performance-improving code pairs and corresponding performance measurements, with Low-Rank Adaptation (LoRA) to enhance code optimization performance. Our experimental results on the Performance Improving Code Edits (PIE) benchmark demonstrate that our method outperforms existing models on multiple metrics. The FasterPy tool and the experimental results are available at this https URL. 

---
# Scaling Unverifiable Rewards: A Case Study on Visual Insights 

**Authors**: Shuyu Gan, James Mooney, Pan Hao, Renxiang Wang, Mingyi Hong, Qianwen Wang, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22650)  

**Abstract**: Large Language Model (LLM) agents can increasingly automate complex reasoning through Test-Time Scaling (TTS), iterative refinement guided by reward signals. However, many real-world tasks involve multi-stage pipeline whose final outcomes lack verifiable rewards or sufficient data to train robust reward models, making judge-based refinement prone to accumulate error over stages. We propose Selective TTS, a process-based refinement framework that scales inference across different stages in multi-agent pipeline, instead of repeated refinement over time by prior work. By distributing compute across stages and pruning low-quality branches early using process-specific judges, Selective TTS mitigates the judge drift and stabilizes refinement. Grounded in the data science pipeline, we build an end-to-end multi-agent pipeline for generating visually insightful charts and report of given dataset, and design a reliable LLM-based judge model, aligned with human experts (Kendall's {\tau}=0.55). Our proposed selective TTS then improves insight quality under a fixed compute budget, increasing mean scores from 61.64 to 65.86 while reducing variance. We hope our findings serve as the first step toward to scaling complex, open-ended tasks with unverifiable rewards, such as scientific discovery and story generation. 

---
# Robust LLM-based Column Type Annotation via Prompt Augmentation with LoRA Tuning 

**Authors**: Hanze Meng, Jianhao Cao, Rachel Pottinger  

**Link**: [PDF](https://arxiv.org/pdf/2512.22742)  

**Abstract**: Column Type Annotation (CTA) is a fundamental step towards enabling schema alignment and semantic understanding of tabular data. Existing encoder-only language models achieve high accuracy when fine-tuned on labeled columns, but their applicability is limited to in-domain settings, as distribution shifts in tables or label spaces require costly re-training from scratch. Recent work has explored prompting generative large language models (LLMs) by framing CTA as a multiple-choice task, but these approaches face two key challenges: (1) model performance is highly sensitive to subtle changes in prompt wording and structure, and (2) annotation F1 scores remain modest. A natural extension is to fine-tune large language models. However, fully fine-tuning these models incurs prohibitive computational costs due to their scale, and the sensitivity to prompts is not eliminated. In this paper, we present a parameter-efficient framework for CTA that trains models over prompt-augmented data via Low-Rank Adaptation (LoRA). Our approach mitigates sensitivity to prompt variations while drastically reducing the number of necessary trainable parameters, achieving robust performance across datasets and templates. Experimental results on recent benchmarks demonstrate that models fine-tuned with our prompt augmentation strategy maintain stable performance across diverse prompt patterns during inference and yield higher weighted F1 scores than those fine-tuned on a single prompt template. These results highlight the effectiveness of parameter-efficient training and augmentation strategies in developing practical and adaptable CTA systems. 

---
# Self-Rewarded Multimodal Coherent Reasoning Across Diverse Visual Domains 

**Authors**: Jesen Zhang, Ningyuan Liu, Kaitong Cai, Sidi Liu, Jing Yang, Ziliang Chen, Xiaofei Sun, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22545)  

**Abstract**: Multimodal LLMs often produce fluent yet unreliable reasoning, exhibiting weak step-to-step coherence and insufficient visual grounding, largely because existing alignment approaches supervise only the final answer while ignoring the reliability of the intermediate reasoning process. We introduce SR-MCR, a lightweight and label-free framework that aligns reasoning by exploiting intrinsic process signals derived directly from model outputs. Five self-referential cues -- semantic alignment, lexical fidelity, non-redundancy, visual grounding, and step consistency -- are integrated into a normalized, reliability-weighted reward that provides fine-grained process-level guidance. A critic-free GRPO objective, enhanced with a confidence-aware cooling mechanism, further stabilizes training and suppresses trivial or overly confident generations. Built on Qwen2.5-VL, SR-MCR improves both answer accuracy and reasoning coherence across a broad set of visual benchmarks; among open-source models of comparable size, SR-MCR-7B achieves state-of-the-art performance with an average accuracy of 81.4%. Ablation studies confirm the independent contributions of each reward term and the cooling module. 

---
# Predicting LLM Correctness in Prosthodontics Using Metadata and Hallucination Signals 

**Authors**: Lucky Susanto, Anasta Pranawijayana, Cortino Sukotjo, Soni Prasad, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2512.22508)  

**Abstract**: Large language models (LLMs) are increasingly adopted in high-stakes domains such as healthcare and medical education, where the risk of generating factually incorrect (i.e., hallucinated) information is a major concern. While significant efforts have been made to detect and mitigate such hallucinations, predicting whether an LLM's response is correct remains a critical yet underexplored problem. This study investigates the feasibility of predicting correctness by analyzing a general-purpose model (GPT-4o) and a reasoning-centric model (OSS-120B) on a multiple-choice prosthodontics exam. We utilize metadata and hallucination signals across three distinct prompting strategies to build a correctness predictor for each (model, prompting) pair. Our findings demonstrate that this metadata-based approach can improve accuracy by up to +7.14% and achieve a precision of 83.12% over a baseline that assumes all answers are correct. We further show that while actual hallucination is a strong indicator of incorrectness, metadata signals alone are not reliable predictors of hallucination. Finally, we reveal that prompting strategies, despite not affecting overall accuracy, significantly alter the models' internal behaviors and the predictive utility of their metadata. These results present a promising direction for developing reliability signals in LLMs but also highlight that the methods explored in this paper are not yet robust enough for critical, high-stakes deployment. 

---
# Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring 

**Authors**: Saisab Sadhu, Ashim Dhor  

**Link**: [PDF](https://arxiv.org/pdf/2512.22496)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as automated tutors to address educator shortages; however, they often fail at pedagogical reasoning, frequently validating incorrect student solutions (sycophancy) or providing overly direct answers that hinder learning. We introduce Hierarchical Pedagogical Oversight (HPO), a framework that adapts structured adversarial synthesis to educational assessment. Unlike cooperative multi-agent systems that often drift toward superficial consensus, HPO enforces a dialectical separation of concerns: specialist agents first distill dialogue context, which then grounds a moderated, five-act debate between opposing pedagogical critics. We evaluate this framework on the MRBench dataset of 1,214 middle-school mathematics dialogues. Our 8B-parameter model achieves a Macro F1 of 0.845, outperforming GPT-4o (0.812) by 3.3% while using 20 times fewer parameters. These results establish adversarial reasoning as a critical mechanism for deploying reliable, low-compute pedagogical oversight in resource-constrained environments. 

---
# Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration 

**Authors**: Bruno Mlodozeniec, Pierre Ablin, Louis Béthune, Dan Busbridge, Michal Klein, Jason Ramapuram, Marco Cuturi  

**Link**: [PDF](https://arxiv.org/pdf/2512.22382)  

**Abstract**: Hyperparameter tuning can dramatically impact training stability and final performance of large-scale models. Recent works on neural network parameterisations, such as $\mu$P, have enabled transfer of optimal global hyperparameters across model sizes. These works propose an empirical practice of search for optimal global base hyperparameters at a small model size, and transfer to a large size. We extend these works in two key ways. To handle scaling along most important scaling axes, we propose the Complete$^{(d)}$ Parameterisation that unifies scaling in width and depth -- using an adaptation of CompleteP -- as well as in batch-size and training duration. Secondly, with our parameterisation, we investigate per-module hyperparameter optimisation and transfer. We characterise the empirical challenges of navigating the high-dimensional hyperparameter landscape, and propose practical guidelines for tackling this optimisation problem. We demonstrate that, with the right parameterisation, hyperparameter transfer holds even in the per-module hyperparameter regime. Our study covers an extensive range of optimisation hyperparameters of modern models: learning rates, AdamW parameters, weight decay, initialisation scales, and residual block multipliers. Our experiments demonstrate significant training speed improvements in Large Language Models with the transferred per-module hyperparameters. 

---
# LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs 

**Authors**: Bing Hao, Minglai Shao, Zengyi Wo, Yunlong Chu, Yuhang Liu, Ruijie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22266)  

**Abstract**: The widespread application of Large Language Models (LLMs) has motivated a growing interest in their capacity for processing dynamic graphs. Temporal motifs, as an elementary unit and important local property of dynamic graphs which can directly reflect anomalies and unique phenomena, are essential for understanding their evolutionary dynamics and structural features. However, leveraging LLMs for temporal motif analysis on dynamic graphs remains relatively unexplored. In this paper, we systematically study LLM performance on temporal motif-related tasks. Specifically, we propose a comprehensive benchmark, LLMTM (Large Language Models in Temporal Motifs), which includes six tailored tasks across nine temporal motif types. We then conduct extensive experiments to analyze the impacts of different prompting techniques and LLMs (including nine models: openPangu-7B, the DeepSeek-R1-Distill-Qwen series, Qwen2.5-32B-Instruct, GPT-4o-mini, DeepSeek-R1, and o3) on model performance. Informed by our benchmark findings, we develop a tool-augmented LLM agent that leverages precisely engineered prompts to solve these tasks with high accuracy. Nevertheless, the high accuracy of the agent incurs a substantial cost. To address this trade-off, we propose a simple yet effective structure-aware dispatcher that considers both the dynamic graph's structural properties and the LLM's cognitive load to intelligently dispatch queries between the standard LLM prompting and the more powerful agent. Our experiments demonstrate that the structure-aware dispatcher effectively maintains high accuracy while reducing cost. 

---
# Agentic Software Issue Resolution with Large Language Models: A Survey 

**Authors**: Zhonghao Jiang, David Lo, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22256)  

**Abstract**: Software issue resolution aims to address real-world issues in software repositories (e.g., bug fixing and efficiency optimization) based on natural language descriptions provided by users, representing a key aspect of software maintenance. With the rapid development of large language models (LLMs) in reasoning and generative capabilities, LLM-based approaches have made significant progress in automated software issue resolution. However, real-world software issue resolution is inherently complex and requires long-horizon reasoning, iterative exploration, and feedback-driven decision making, which demand agentic capabilities beyond conventional single-step approaches. Recently, LLM-based agentic systems have become mainstream for software issue resolution. Advancements in agentic software issue resolution not only greatly enhance software maintenance efficiency and quality but also provide a realistic environment for validating agentic systems' reasoning, planning, and execution capabilities, bridging artificial intelligence and software engineering.
This work presents a systematic survey of 126 recent studies at the forefront of LLM-based agentic software issue resolution research. It outlines the general workflow of the task and establishes a taxonomy across three dimensions: benchmarks, techniques, and empirical studies. Furthermore, it highlights how the emergence of agentic reinforcement learning has brought a paradigm shift in the design and training of agentic systems for software engineering. Finally, it summarizes key challenges and outlines promising directions for future research. 

---
# Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation 

**Authors**: Bhaktipriya Radharapu, Eshika Saxena, Kenneth Li, Chenxi Whitehouse, Adina Williams, Nicola Cancedda  

**Link**: [PDF](https://arxiv.org/pdf/2512.22245)  

**Abstract**: As LLM-based judges become integral to industry applications, obtaining well-calibrated uncertainty estimates efficiently has become critical for production deployment. However, existing techniques, such as verbalized confidence and multi-generation methods, are often either poorly calibrated or computationally expensive. We introduce linear probes trained with a Brier score-based loss to provide calibrated uncertainty estimates from reasoning judges' hidden states, requiring no additional model training. We evaluate our approach on both objective tasks (reasoning, mathematics, factuality, coding) and subjective human preference judgments. Our results demonstrate that probes achieve superior calibration compared to existing methods with $\approx10$x computational savings, generalize robustly to unseen evaluation domains, and deliver higher accuracy on high-confidence predictions. However, probes produce conservative estimates that underperform on easier datasets but may benefit safety-critical deployments prioritizing low false-positive rates. Overall, our work demonstrates that interpretability-based uncertainty estimation provides a practical and scalable plug-and-play solution for LLM judges in production. 

---
# LLMBoost: Make Large Language Models Stronger with Boosting 

**Authors**: Zehao Chen, Tianxiang Ai, Yifei Li, Gongxun Li, Yuyang Wei, Wang Zhou, Guanghui Li, Bin Yu, Zhijun Chen, Hailong Sun, Fuzhen Zhuang, Jianxin Li, Deqing Wang, Yikun Ban  

**Link**: [PDF](https://arxiv.org/pdf/2512.22309)  

**Abstract**: Ensemble learning of LLMs has emerged as a promising alternative to enhance performance, but existing approaches typically treat models as black boxes, combining the inputs or final outputs while overlooking the rich internal representations and interactions across this http URL this work, we introduce LLMBoost, a novel ensemble fine-tuning framework that breaks this barrier by explicitly leveraging intermediate states of LLMs. Inspired by the boosting paradigm, LLMBoost incorporates three key innovations. First, a cross-model attention mechanism enables successor models to access and fuse hidden states from predecessors, facilitating hierarchical error correction and knowledge transfer. Second, a chain training paradigm progressively fine-tunes connected models with an error-suppression objective, ensuring that each model rectifies the mispredictions of its predecessor with minimal additional computation. Third, a near-parallel inference paradigm design pipelines hidden states across models layer by layer, achieving inference efficiency approaching single-model decoding. We further establish the theoretical foundations of LLMBoost, proving that sequential integration guarantees monotonic improvements under bounded correction assumptions. Extensive experiments on commonsense reasoning and arithmetic reasoning tasks demonstrate that LLMBoost consistently boosts accuracy while reducing inference latency. 

---
# Literature Mining System for Nutraceutical Biosynthesis: From AI Framework to Biological Insight 

**Authors**: Xinyang Sun, Nipon Sarmah, Miao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.22225)  

**Abstract**: The extraction of structured knowledge from scientific literature remains a major bottleneck in nutraceutical research, particularly when identifying microbial strains involved in compound biosynthesis. This study presents a domain-adapted system powered by large language models (LLMs) and guided by advanced prompt engineering techniques to automate the identification of nutraceutical-producing microbes from unstructured scientific text. By leveraging few-shot prompting and tailored query designs, the system demonstrates robust performance across multiple configurations, with DeepSeekV3 outperforming LLaMA2 in accuracy, especially when domain-specific strain information is included. A structured and validated dataset comprising 35 nutraceutical-strain associations was generated, spanning amino acids, fibers, phytochemicals, and vitamins. The results reveal significant microbial diversity across monoculture and co-culture systems, with dominant contributions from Corynebacterium glutamicum, Escherichia coli, and Bacillus subtilis, alongside emerging synthetic consortia. This AI-driven framework not only enhances the scalability and interpretability of literature mining but also provides actionable insights for microbial strain selection, synthetic biology design, and precision fermentation strategies in the production of high-value nutraceuticals. 

---
# ReGAIN: Retrieval-Grounded AI Framework for Network Traffic Analysis 

**Authors**: Shaghayegh Shajarian, Kennedy Marsh, James Benson, Sajad Khorsandroo, Mahmoud Abdelsalam  

**Link**: [PDF](https://arxiv.org/pdf/2512.22223)  

**Abstract**: Modern networks generate vast, heterogeneous traffic that must be continuously analyzed for security and performance. Traditional network traffic analysis systems, whether rule-based or machine learning-driven, often suffer from high false positives and lack interpretability, limiting analyst trust. In this paper, we present ReGAIN, a multi-stage framework that combines traffic summarization, retrieval-augmented generation (RAG), and Large Language Model (LLM) reasoning for transparent and accurate network traffic analysis. ReGAIN creates natural-language summaries from network traffic, embeds them into a multi-collection vector database, and utilizes a hierarchical retrieval pipeline to ground LLM responses with evidence citations. The pipeline features metadata-based filtering, MMR sampling, a two-stage cross-encoder reranking mechanism, and an abstention mechanism to reduce hallucinations and ensure grounded reasoning. Evaluated on ICMP ping flood and TCP SYN flood traces from the real-world traffic dataset, it demonstrates robust performance, achieving accuracy between 95.95% and 98.82% across different attack types and evaluation benchmarks. These results are validated against two complementary sources: dataset ground truth and human expert assessments. ReGAIN also outperforms rule-based, classical ML, and deep learning baselines while providing unique explainability through trustworthy, verifiable responses. 

---
# Wireless Traffic Prediction with Large Language Model 

**Authors**: Chuanting Zhang, Haixia Zhang, Jingping Qiao, Zongzhang Li, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2512.22178)  

**Abstract**: The growing demand for intelligent, adaptive resource management in next-generation wireless networks has underscored the importance of accurate and scalable wireless traffic prediction. While recent advancements in deep learning and foundation models such as large language models (LLMs) have demonstrated promising forecasting capabilities, they largely overlook the spatial dependencies inherent in city-scale traffic dynamics. In this paper, we propose TIDES (Traffic Intelligence with DeepSeek-Enhanced Spatial-temporal prediction), a novel LLM-based framework that captures spatial-temporal correlations for urban wireless traffic prediction. TIDES first identifies heterogeneous traffic patterns across regions through a clustering mechanism and trains personalized models for each region to balance generalization and specialization. To bridge the domain gap between numerical traffic data and language-based models, we introduce a prompt engineering scheme that embeds statistical traffic features as structured inputs. Furthermore, we design a DeepSeek module that enables spatial alignment via cross-domain attention, allowing the LLM to leverage information from spatially related regions. By fine-tuning only lightweight components while freezing core LLM layers, TIDES achieves efficient adaptation to domain-specific patterns without incurring excessive training overhead. Extensive experiments on real-world cellular traffic datasets demonstrate that TIDES significantly outperforms state-of-the-art baselines in both prediction accuracy and robustness. Our results indicate that integrating spatial awareness into LLM-based predictors is the key to unlocking scalable and intelligent network management in future 6G systems. 

---
# Pre-review to Peer review: Pitfalls of Automating Reviews using Large Language Models 

**Authors**: Akhil Pandey Akella, Harish Varma Siravuri, Shaurya Rohatgi  

**Link**: [PDF](https://arxiv.org/pdf/2512.22145)  

**Abstract**: Large Language Models are versatile general-task solvers, and their capabilities can truly assist people with scholarly peer review as \textit{pre-review} agents, if not as fully autonomous \textit{peer-review} agents. While incredibly beneficial, automating academic peer-review, as a concept, raises concerns surrounding safety, research integrity, and the validity of the academic peer-review process. The majority of the studies performing a systematic evaluation of frontier LLMs generating reviews across science disciplines miss the mark on addressing the alignment/misalignment of reviews along with the utility of LLM generated reviews when compared against publication outcomes such as \textbf{Citations}, \textbf{Hit-papers}, \textbf{Novelty}, and \textbf{Disruption}. This paper presents an experimental study in which we gathered ground-truth reviewer ratings from OpenReview and used various frontier open-weight LLMs to generate reviews of papers to gauge the safety and reliability of incorporating LLMs into the scientific review pipeline. Our findings demonstrate the utility of frontier open-weight LLMs as pre-review screening agents despite highlighting fundamental misalignment risks when deployed as autonomous reviewers. Our results show that all models exhibit weak correlation with human peer reviewers (0.15), with systematic overestimation bias of 3-5 points and uniformly high confidence scores (8.0-9.0/10) despite prediction errors. However, we also observed that LLM reviews correlate more strongly with post-publication metrics than with human scores, suggesting potential utility as pre-review screening tools. Our findings highlight the potential and address the pitfalls of automating peer reviews with language models. We open-sourced our dataset $D_{LMRSD}$ to help the research community expand the safety framework of automating scientific reviews. 

---
# GPU-Virt-Bench: A Comprehensive Benchmarking Framework for Software-Based GPU Virtualization Systems 

**Authors**: Jithin VG, Ditto PS  

**Link**: [PDF](https://arxiv.org/pdf/2512.22125)  

**Abstract**: The proliferation of GPU-accelerated workloads, particularly in artificial intelligence and large language model (LLM) inference, has created unprecedented demand for efficient GPU resource sharing in cloud and container environments. While NVIDIA's Multi-Instance GPU (MIG) technology provides hardware-level isolation, its availability is limited to high-end datacenter GPUs. Software-based virtualization solutions such as HAMi-core and BUD-FCSP offer alternatives for broader GPU families but lack standardized evaluation methodologies. We present GPU-Virt-Bench, a comprehensive benchmarking framework that evaluates GPU virtualization systems across 56 performance metrics organized into 10 categories. Our framework measures overhead, isolation quality, LLM-specific performance, memory bandwidth, cache behavior, PCIe throughput, multi-GPU communication, scheduling efficiency, memory fragmentation, and error recovery. GPU-Virt-Bench enables systematic comparison between software virtualization approaches and ideal MIG behavior, providing actionable insights for practitioners deploying GPU resources in multi-tenant environments. We demonstrate the framework's utility through evaluation of HAMi-core, BUD-FCSP, and simulated MIG baselines, revealing performance characteristics critical for production deployment decisions. 

---
# ReCollab: Retrieval-Augmented LLMs for Cooperative Ad-hoc Teammate Modeling 

**Authors**: Conor Wallace, Umer Siddique, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.22129)  

**Abstract**: Ad-hoc teamwork (AHT) requires agents to infer the behavior of previously unseen teammates and adapt their policy accordingly. Conventional approaches often rely on fixed probabilistic models or classifiers, which can be brittle under partial observability and limited interaction. Large language models (LLMs) offer a flexible alternative: by mapping short behavioral traces into high-level hypotheses, they can serve as world models over teammate behavior. We introduce \Collab, a language-based framework that classifies partner types using a behavior rubric derived from trajectory features, and extend it to \ReCollab, which incorporates retrieval-augmented generation (RAG) to stabilize inference with exemplar trajectories. In the cooperative Overcooked environment, \Collab effectively distinguishes teammate types, while \ReCollab consistently improves adaptation across layouts, achieving Pareto-optimal trade-offs between classification accuracy and episodic return. These findings demonstrate the potential of LLMs as behavioral world models for AHT and highlight the importance of retrieval grounding in challenging coordination settings. 

---
