# PalmClaw: A Native On-Device Agent Framework for Mobile Phones 

**Authors**: Hongru Cai, Yongqi Li, Ran Wei, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.13027)  

**Abstract**: Large Language Model (LLM) agents have moved beyond generating responses to executing multi-step tasks by calling tools, observing the results, and iteratively deciding the next action. Most agent systems run on desktops or servers, which support tool use and task automation. Mobile devices are also important agent environments because they are widely accessible and contain users' data, sensors, and daily-use applications. Existing mobile agents mainly operate smartphones through graphical user interface (GUI) actions such as tapping, swiping, and typing, which often form long, interface-dependent sequences, cannot directly access device capabilities, and make execution boundaries difficult to define. We present \textbf{PalmClaw}, an open-source agent framework that runs natively on mobile phones and manages the sessions, memory, skills, tools, and agent loop directly on the device. PalmClaw exposes device capabilities as device tools with explicit arguments, structured results, and clearly defined execution boundaries. This design enables agents to use mobile capabilities directly while keeping each action explicit and controlled. Experiments show an 11.5\% relative improvement in task success and a 94.9\% reduction in completion time over the strongest baseline, with lower setup burden and traces illustrating how execution boundaries are applied. Code is available at this https URL. 

---
# The Illusion of Robustness: Aggregate Accuracy Hides Prediction Flips under Task-Irrelevant Context 

**Authors**: Yanzhe Zhang, Sanmi Koyejo, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12963)  

**Abstract**: As large language models (LLMs) grow more capable, they are increasingly deployed in context-rich settings where task inputs are often accompanied by long, partially irrelevant context. In a controlled setting, we find that state-of-the-art models often appear robust to task-irrelevant context at the aggregate level: prepending it to benchmark questions causes little change in overall accuracy. This aggregate stability, however, masks significant per-example instability. Even semantically meaningless pseudo-words, formed by randomly combining characters, can markedly shift model predictions on a small fraction of examples, degrading performance on some while improving it on others. This two-sided effect holds consistently across a wide range of models and datasets, yet the affected examples are largely model-specific. We further show that this instability is modulated by context type, context length, test-time compute, and model development stage. Together, our findings reveal context-induced tail risks concealed by aggregate accuracy, motivating per-example reliability evaluation of language models. 

---
# LLM Judges Can Be Too Generous When There Is No Reference Answer 

**Authors**: Chalamalasetti Kranti, Sowmya Vajjala  

**Link**: [PDF](https://arxiv.org/pdf/2607.12885)  

**Abstract**: LLM judges are increasingly being used to evaluate open-ended model responses, often in no-reference settings where a ground-truth answer is unavailable. However, can they reliably assess in such evaluation setups? We explore this question in this paper through a two stage pipeline with a) calibration experiments that assess the judge model's knowledge of the task it is evaluating, and b) sensitivity experiments that assess how the judge model's performance is impacted by the presence and positioning of the reference answer in the prompt. Across experiments covering three languages, we show that the judge models we evaluated tend to over-credit incorrect answers in the absence of a reference answer, and adding reference answer information to the prompt flips the judge model's correct/incorrect decisions by as much as 85% in some experimental settings. Comparison with a subset of human annotations shows that these reference-driven changes generally align with human judgments. Our results emphasize the need for calibrating the LLM judges with a sample with reference-aware evaluation before using them in reference-free setups reliably, and our methodology provides a blueprint for researchers and practitioners in doing such calibration of LLM judges for other tasks. 

---
# Evaluating Large Language Models on Misconceptions in Multi-Turn Medical Conversations 

**Authors**: Monica Munnangi, Saiph Savage  

**Link**: [PDF](https://arxiv.org/pdf/2607.12884)  

**Abstract**: Patients seeking medical information often ask questions that embed incorrect assumptions or misconceptions. In such cases, safe medical communication requires not only answering the question, but identifying and correcting the underlying false belief. These interactions naturally unfold over multiple turns, a pattern now mirrored in interactions with LLMs. Yet current evaluation frameworks do not capture model behavior in these settings, where misconceptions can emerge, persist, or evolve over the course of a conversation. Whether LLMs can reliably correct such misconceptions over time remains largely unexamined. To study this, we introduce ThReadMed-QA, a multi-turn medical dialogue dataset of 2,437 patient-physician conversation threads comprising 8,204 question-answer pairs, derived from real patient interactions on AskDocs. This dataset enables systematic evaluation of whether models can detect and correct misconceptions under a multi-turn context. We evaluate five LLMs using a rubric-based LLM-as-a-Judge framework that scores responses based on their ability to identify and correct misconceptions. Our experiments reveal a consistent pattern: even frontier models that can address misconceptions in a single interaction degrade substantially over subsequent turns. GPT-5 and Claude-Haiku correct these false presuppositions around 85% on initial questions but drop to roughly 50% within two follow-ups. An oracle analysis replacing prior model outputs with physician responses shows that much of the degradation is driven by error propagation, while performance remains imperfect even under correct context. Even when models tend to correct misconceptions initially, their performance degrades substantially over later turns, leading to inconsistent and potentially unsafe guidance in patient-facing settings and highlighting the need for evaluation frameworks that capture multi-turn behavior. 

---
# Can LLMs Write Reliable Rubrics? A Meta-Evaluation for Experiment Reproduction 

**Authors**: Hanhua Hong, Yizhi Li, Jiaoyan Chen, Luu Gia Huy, Sophia Ananiadou, Jung-jae Kim, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.12835)  

**Abstract**: Rubric-based evaluation is a promising approach for assessing open-ended outputs from LLM-based research agents, particularly in paper reproduction, where direct paper-to-repository comparison is prone to hallucination. However, constructing paper-specific rubrics requires substantial expert effort, limiting the scalability of benchmarks such as PaperBench. In this work, we present, to our knowledge, the first systematic meta-evaluation of LLM-generated rubrics for paper reproduction. We reformulate rubrics into a checklist-style format and evaluate four generation settings across two backbone models. We meta-evaluate generated rubrics intrinsically by semantic similarity and extrinsically by score alignment with ground-truth rubrics. Our results show that the augmented settings substantially improves downstream evaluation alignment, with the strongest setting approaching the human baseline, while intrinsic gains are more modest. Further analyses reveal that LLM-generated rubrics are often overly fine-grained, biased toward high scores, and less adaptive to paper domains, highlighting both the affordances and limitations. 

---
# Knowledgeless Language Models: Suppressing Parametric Recall for Evidence-Grounded Language Modeling 

**Authors**: Roi Cohen, Yvan Carré, Nick Lechtenbörger, Hendrik Droste, Lucas Kerschke, Russa Biswas, Gerard de Melo, Jan Buys  

**Link**: [PDF](https://arxiv.org/pdf/2607.12831)  

**Abstract**: Language models encode substantial factual knowledge in their parameters, which can lead to unreliable behavior when this knowledge is outdated, incomplete, or misaligned with the provided context. In this work, we study whether modifying the pretraining signal can systematically shift models away from parametric recall and toward evidence-grounded reasoning. We introduce Knowledge--''Less'' Language Models (KLLMs), a fundamentally different epistemic training paradigm for LLMs, which are pretrained on corpora in which named entities are anonymized, thereby removing a primary channel for entity-linked factual supervision. This intervention substantially reduces closed-book factual recall, while often improving performance on tasks where relevant information is provided as context. Across multiple model scales, KLLMs consistently outperform matched baselines on contextual question answering, fact verification, and hallucination detection benchmarks. Crucially, in retrieval-grounded settings with imperfect evidence, KLLMs show improved robustness and achieve up to 20--25\% relative gains over standard language models. They further exhibit better calibration, with improved ECE, Brier score, and AUROC, as well as more reliable abstention behavior. Our results demonstrate that suppressing entity-linked supervision during pretraining induces a shift in epistemic behavior: KLLMs rely less on parametric knowledge and more on external evidence, leading to improved reliability under realistic conditions. This suggests that pretraining-time control over knowledge acquisition can complement retrieval-augmented and tool-based systems by providing a more evidence-sensitive base model. 

---
# The One-Word Census: Answer-Choice Conformity Across 44 Language Models 

**Authors**: Tapan Parikh  

**Link**: [PDF](https://arxiv.org/pdf/2607.12796)  

**Abstract**: When a language model must pick one answer from a large space of equally valid options, which does it pick -- and how often is it the same answer every other model picks? Asked to "pick a word -- any word," 44 models chose "serendipity" 41% of the time. We characterize this convergence with a deliberately minimal instrument: 31 single-turn prompts, each naming a category with many valid one-word answers ("Name a tree."), asked four times per model with no system prompt. Analysis is exact-match on normalized tokens -- no embeddings, no judge -- at about a dollar per model. That models converge is well documented; our contribution is the instrument itself -- the One-Word Census -- and what it reveals about the structure of the convergence. We score each model by answer-choice surprisal: the average $-\log2$ probability of its answers under the pooled answers of all other models, leave-one-out. Convergence is extreme -- in 7 of 31 categories one answer takes over 80% of all answers -- yet conformity varies more than fourfold across models, and the variation is structured. Persona- and community-tuned models are the most divergent; the newest mainline flagships are the most conformist, producing almost no answer no other model gave. Within four lineages (Claude, GPT, Qwen, Grok) conformity rises with each generation -- but reverses for the latest flagship Claude and GPT models, a possible early signal of repositioning at the top tier. Rankings are robust to roster composition (leave-one-family-out rho = 0.985). Against human category-production norms, the field is more concentrated than people in 18 of 20 shared categories. All prompts, transcripts, and code are public. 

---
# Epistemic Stance Flexibility Probing: Measuring Prompt-Conditioned Register Shift in Large Language Models 

**Authors**: Binwen Liu, Yilin Ren  

**Link**: [PDF](https://arxiv.org/pdf/2607.12739)  

**Abstract**: A language model may be asked either what experts believe about a contested claim or what it believes about the claim itself. A trustworthy conversational agent should distinguish these two requests and respond in different epistemic registers: neutral attribution in the first case and stance expression in the second. Whether such a shift occurs-and whether it occurs coherently-is not directly assessed by existing benchmarks for accuracy, instruction following, or safety. We introduce ESFP, a behavioral benchmark that treats the contrast between externally attributed and self-attributed prompts as the fundamental unit of measurement. ESFP consists of 104 carefully controlled items spanning six epistemic categories and five phrasing templates, and evaluates model responses along four complementary dimensions: lexical self-attribution, representation-level responsiveness to role framing, sentence-level stance content density assessed by an LLM judge panel, and cross-condition stance consistency. Evaluating eight frontier models from five vendors, we find that epistemic flexibility is largely orthogonal to general model capability: a 27B open-weight model matches the strongest proprietary systems, the flagship model of one family underperforms its lightweight counterpart, and reasoning-optimized models do not consistently exhibit higher flexibility. Stance content density provides the strongest signal, while surface-level lexical markers such as 'I think' can change substantially without corresponding changes in expressed stance. We provide item-level bootstrap confidence intervals, weight-sensitivity analyses, and an explicit discussion of the interpretation limits of the composite score. ESFP measures a model's propensity to adapt its epistemic stance under changing attribution conditions, rather than a general competence measure. 

---
# Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts 

**Authors**: Jincheng Xie, Runheng Liu, Heyan Huang, Yawen Ling, Hanbin Dai, Yu Zheng, Wen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.12696)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models have become an important approach for scaling Large Language Models (LLMs), but their inference efficiency depends strongly on expert activation patterns. Speculative decoding (SD) accelerates autoregressive generation by verifying multiple draft tokens in parallel, yet existing draft selection strategies primarily optimize acceptance likelihood. In large-scale MoE models, however, selecting draft tokens also determines the union of experts activated during verification. We observe that confidence-driven SD can introduce \textit{expert scattering}: high-probability draft tokens may route to disjoint experts, increasing expert-weight memory traffic and reducing the speedup from speculation. Motivated by this observation, we revisit draft-tree selection under the non-uniform memory-cost structure of MoE inference. We propose \textsc{EcoSpec}, a cost-aware speculative decoding framework that incorporates predicted marginal expert activation cost into draft selection. With a lightweight expert predictor and a dynamic expert buffer, \textsc{EcoSpec} favors draft paths that preserve high acceptance likelihood while reusing experts already covered by the current verification set, without modifying the target-model verification rule. We evaluate \textsc{EcoSpec} on three large-scale MoE models, including DeepSeek-V3.1 (671B), Qwen3-235B-A22B, and GPT-OSS-120B, across reasoning, coding, question-answering, and dialogue benchmarks. \textsc{EcoSpec} consistently reduces active expert footprints and improves end-to-end decoding speed, achieving up to $1.62\times$ speedup. These results show that accounting for expert activation cost is important for efficient speculative decoding in large-scale MoE models. 

---
# From Critic to Confidence: PPO for Language-Based Quantitative Prediction with Confidence Estimation 

**Authors**: Mehak Dhaliwal, Rasta Tadayon, Andong Hua, Haewon Jeong, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.12687)  

**Abstract**: LLMs can perform language-based quantitative prediction from unstructured inputs, but remain susceptible to hallucinations and overconfident errors, making it critical to know not only what a model predicts, but when its predictions can be trusted. We introduce CARE-PPO, a reinforcement learning framework that establishes a connection between loss prediction for uncertainty estimation and actor-critic PPO fine-tuning, enabling joint learning of accurate numerical estimates and reliable confidence signals in language-based quantitative prediction. CARE-PPO uses a Confidence-Aligned Reward for Estimation, defined as a function of prediction error, to provide dense error-aware feedback to the actor while inducing the critic to learn a value function aligned with prediction quality. During inference, we repurpose the critic as a confidence estimator. Across two real-world tasks in healthcare and finance and two Qwen-3 model scales (4B and 8B), CARE-PPO achieves strong quantitative prediction performance, while producing significantly better-aligned confidence estimates through the critic than logit-based and verbalized baselines. These gains persist under realistic out-of-distribution settings across domains, spanning linguistic and domain shifts. Finally, CARE-PPO reduces task-specific overfitting on general instruction-following prompts, consistent with the broader generalization advantages of RL fine-tuning over supervised approaches. 

---
# Segregate, Refine, Integrate: Decomposing Multimodal Fusion for Sentiment Analysis 

**Authors**: Alexios Filippakopoulos, Elias Kallioras, Nikolaos Xiros, Efthymios Georgiou, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2607.12686)  

**Abstract**: Multimodal fusion must simultaneously refine modality-specific signals and model cross-modal interactions; two competing objectives typically entangled within the same operation. We propose \textbf{SeRIn} (\textbf{Se}gregate, \textbf{R}efine, \textbf{In}tegrate), a multimodal LM fusion scheme that enforces this separation as an architectural prior. Modality-specific representations evolve along isolated pathways, each refined against its respective encoder context, while a dedicated cross-modal pathway accumulates their joint evolution without contaminating unimodal streams. Full cross-modal interaction is deferred to a final prediction step - ablations confirm that structured interactions, not added capacity, drive the gains; gate analysis under visual corruption reveals emergent modality reweighting without explicit supervision. SeRIn achieves state-of-the-art results on CH-SIMS and CMU-MOSEI, improving all metrics on both benchmarks. 

---
# Can Induced Emotion Bias LLM Behaviors in Sequential Decision Making? 

**Authors**: Minh Khoi Ho, Zihao Zhu, Runchuan Zhu, Levina Li, Zhiwen Fan, Zhangyang Wang, Junyuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.12631)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as autonomous agents in high-stakes domains, understanding contextual factors that may modulate their decision-making becomes critical. While LLMs are trained to perceive and resonate with users' emotions, it remains unclear whether induced emotion can influence their sequential decision-making. We investigate this question using the Iowa Gambling Task (IGT), a classic psychological paradigm for studying decision-making under uncertainty, combined with an imagination-based emotion induction procedure. We first validate the feasibility of this paradigm by confirming that LLMs can sense strong, distinguishable emotions from context and that LLM agents can learn from sequential interactions in a human-like pace. With the validated setup, we find that, different from humans, induced emotion does not significantly bias the decision dynamics of LLM agents on average. However, the effects of anger are conditioned: inducing anger makes LLM agents less sensitive to penalties for bad decisions, and in early stages of the game, anger can lower exploration, locking decisions into a few choices early. These findings reveal the subtle yet distinct effects of induced emotion on LLM decision-making compared to human behavior, and provide a tool for future research on affective modulation of LLM agents. 

---
# KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill 

**Authors**: Yunxin Li, Jinchao Li, Shibo Su, Zhenran Xu, Chenrui Zhao, Tongshu Bian, Xiaoman Liang, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12625)  

**Abstract**: OpenClaw has emerged as a leading agent framework for complex task automation, yet it faces insufficient cross-platform GUI interaction support and a well-built self-evolution mechanism. These flaws limit its adaptation to diverse device ecosystems and prevent performance improvements through continuous learning from execution experience. To resolve these issues, we propose the Know Deeply, Act Perfectly paradigm for personal assistants, which holds that accumulated user interaction and task-running experience directly improve execution accuracy and efficiency, unifying cognitive comprehension and operational execution. Based on this paradigm, we introduce KnowAct-GUIClaw, a novel Know-Route-Act-Reflect framework designed to address OpenClaw's GUI manipulation deficits and break through its cross-platform and recursive self-improvement constraints. First, the host agent leverages accumulated interaction experience and task-relevant knowledge for long-horizon task decomposition and allocation (Know). Second, a pluggable GUI subagent with an experience-attributable memory system (Know) and self-evolving skill library (Act), enabling seamless cross-platform migration and fast-path integration. Especially, this framework continuously stores user profiles and feedback to improve the accuracy of task decomposition and tool calls. Extensive experiments across Android, iOS, HarmonyOS and Windows show that KnowAct-GUIClaw achieves superior efficiency, accuracy and cross-platform adaptability. Especially, the GUIClaw with open-source Kimi-2.6 models achieves the best performance (64.1%) on the long-horizon MobileWorld benchmark, beating all agentical frameworks and closed-source agentical models, e.g., Seed-2.0-Pro and GPT-5.5. Additionally, the knowledgeable memory and execution skills supported by our framework are transferable across diverse base models, improving by 8.5% with Kimi-2.6. 

---
# Translation as a Computationally Efficient Bridge: Feasibility of English BERT for Low-Resource Languages 

**Authors**: Hielke Muizelaar, Giulia Rivetti, Marco Spruit, Marcel Haas  

**Link**: [PDF](https://arxiv.org/pdf/2607.12612)  

**Abstract**: BERT models have revolutionised Natural Language Processing (NLP) through their ability to process unstructured text across diverse domains. However, developing high-quality BERT models for non-English languages remains challenging due to limited annotated data and high computational demands. Translating non-English data into English and fine-tuning existing English BERT models offers a resource-efficient alternative, yet few studies have structurally compared translation-based fine-tuning with native-language BERT performance across tasks and languages. This study provides such a comparison, evaluating the feasibility of translation-based fine-tuning across six NLP tasks: Sentiment Analysis, Hate Speech Detection, Question Answering, Named Entity Recognition, Part-of-Speech Tagging, and Natural Language Inference, using datasets translated from Bulgarian, Chinese, Dutch, Italian, and Russian. Across all settings, the translation-based approach was comparable or superior in 53.3 percent of cases. Gains were most frequent in Question Answering, Part-of-Speech Tagging, and Natural Language Inference, while performance declines were common in Named Entity Recognition and Hate Speech Detection. The results show that translation-based fine-tuning is most effective for tasks relying on syntactic or structural patterns and for languages typologically close to English, such as Dutch, but less effective for token-level or culturally nuanced tasks, particularly in Chinese. Overall, this study demonstrates that translation-based fine-tuning offers a scalable, resource-efficient, and empirically validated path for extending NLP to low-resource languages while advancing linguistic inclusivity and sustainability in artificial intelligence. 

---
# Language Identification with Succinct Machine-Independent Traces 

**Authors**: Moses Charikar, Jon Kleinberg, Chirag Pabbaraju  

**Link**: [PDF](https://arxiv.org/pdf/2607.12443)  

**Abstract**: Motivated by the power of large language models, there has been renewed interest in the Gold-Angluin model of language identification in the limit, with an eye toward variants of the model that might overcome the negative results for its original formulation. Recent papers on this question have proposed looking at computational traces and annotations of training strings as a source of additional power for a learner, reflecting empirical regularities such as the way that commented source code is easier to learn from than arbitrary source code, and text annotated with algorithmically generated chain-of-thought tokens can be easier to learn from than the raw text itself. This recent work has shown positive results for language identification in the presence of such computational traces, but the traces in these positive results come from explicit automata-theoretic machine models that generate the language, where the underlying vocabulary of tokens for the traces is very large. In this paper, we address two fundamental issues left open by this line of work: can we achieve positive results with traces that use only a small alphabet, and can we define traces directly from the language itself, without requiring an underlying machine model that generates it? We establish positive results for both of these questions: for an arbitrary collection of languages, we show how to define computational traces that enable identification in the limit, using an alphabet of tokens that is linear in the size of the alphabet that the languages are defined over, and independent of any other properties of the languages. 

---
# WikiSTAR: A System for Shedding Light on the Hidden History of Scientific Wikipedia Articles 

**Authors**: Omer Ehrlich, Nitzan Barzilay, Rona Aviram, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2607.12441)  

**Abstract**: Wikipedia plays a key role in shaping public understanding of science, and its openly accessible revision history is a unique record of how scientific knowledge evolves over time. Yet scientifically meaningful revisions are obscured by the sheer volume of routine edits, leaving each article's scientific history hidden. We present WikiSTAR (Scientific Tracking of Article Revisions), an interactive system for exploring scientifically meaningful changes across an article's revision history. Using an LLM classifier with an expert-designed multi-label taxonomy, WikiSTAR first tags edit types such as the addition of technical terms, new research findings, and changes in scientific narrative. Then, through interactive views, an article's full revision history can be traced at any granularity - from aggregate trends that reveal when and in which sections scientific content was added or refined, down to individual edits - showing how scientific knowledge develops at a scale previously impossible. In a user study, experts from three domains found that WikiSTAR surfaced new patterns and research questions and enabled previously impractical analyses. We release our system, code and a human-annotated benchmark. 

---
# Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning 

**Authors**: Xinyu Tang, Gangqiang Cao, Yurou Liu, Yuliang Zhan, Xiaochong Lan, Yifan Li, Yuchen Yan, Han Peng, Zican Dong, Zhenduo Zhang, Tianshu Wang, Xinyu Kong, Zujie Wen, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.12395)  

**Abstract**: Reinforcement learning with verifiable rewards without human-annotated data, often referred to as zero RL, has emerged as a powerful paradigm for eliciting chain-of-thought reasoning. However, due to computational constraints, existing studies are largely restricted to small models, leaving the training dynamics and emergent capabilities at a large scale unexplored. To meaningfully explore this frontier, we aim to elicit high-quality reasoning behaviors from the model. However, we find that naive scaling often suffers from poor readability, token redundancy, and a lack of adaptive reasoning depth. To address these challenges, we present a stable and efficient training pipeline, incorporating algorithmic and system optimizations such as clipped importance sampling, training-inference ratio correction, and mixed-precision control. Our experiments offer three key findings that validate the "bitter lesson" of scaling: (1) scaling to 1T parameters significantly enhances sample efficiency and performance ceilings; (2) the training process progresses sequentially through an initial discovery phase followed by a sharpening phase; and (3) the model spontaneously develops advanced cognitive behaviors, including anthropomorphism, structured formatting, self-verification, parallel reasoning, and context anxiety, rendering hand-crafted heuristics redundant. Evaluated on seven mathematical benchmarks, Ring-2.5-1T-Zero achieves competitive performance. Additionally, to assess CoT quality beyond final-answer correctness, we propose a structured evaluation framework across three dimensions: comprehensibility, reproducibility, and efficiency, where our model demonstrates clear advantages in producing structured and concise reasoning traces. By sharing our observed emergent phenomena, we hope to provide the community with deeper insights into scaling behaviors, particularly at the 1-trillion scale. 

---
# Beyond Binary Detection: A Multi-Dimensional Taxonomy of Cancer Misinformation on Reddit 

**Authors**: Aria Pessianzadeh, Pooriya Jamie, Naima Sultana, Georgia Himmelstein, Yuliya Zektser, Patricia Ganz, Homa Hosseinmardi, Amir Ghasemian, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2607.12383)  

**Abstract**: Cancer-related discussions on social media provide an important space for information exchange and peer support, but also facilitate the spread of misinformation that may influence prevention, screening, and treatment decisions. Existing research on cancer misinformation often relies on narrow definitions, small-scale datasets, or binary labeling frameworks. We introduce a multi-dimensional taxonomy for characterizing cancer misinformation in Reddit discussions of breast, lung, colon, and prostate cancer. The taxonomy captures seven dimensions, including misinformation presence, information type, risk level, stance, and topical focus. Using expert-annotated data, we evaluate multiple large language models (LLMs) for scalable misinformation annotation and analyze cancer misinformation across Reddit communities. Our results show that cancer-related misinformation constitutes approximately 6\% of Reddit cancer discussions, with substantial variation across communities and misinformation topics. Few-shot prompting substantially improves classification performance, particularly for nuanced taxonomy dimensions. We additionally identify recurring misinformation narratives centered on unsupported treatments, distrust of conventional medicine, and misleading claims about diagnosis and screening. Our taxonomy, dataset, and findings provide a foundation for multi-dimensional modeling of online cancer misinformation. 

---
# Policy-Conditioned Constrained Decoding for Column-Level Access Control in Text-to-SQL 

**Authors**: Ryoto Miyamoto, Xin Fan, Hayato Yamana  

**Link**: [PDF](https://arxiv.org/pdf/2607.12341)  

**Abstract**: Text-to-SQL is increasingly deployed across trust boundaries between data providers and users. Such deployment must balance three competing requirements: policy compliance, answer coverage, and bounded cost. Existing approaches typically decide refusal based on which columns a query mentions and enforce it stochastically. Whether a query is compliant, however, depends not only on which columns appear but on how they are used, and stochastic enforcement cannot deterministically rule out violations. We formalize this requirement as a column-use policy over semantic use: output, filter condition, and aggregation argument. We integrate the policy by aligning each role with grammar productions tracked by the decoder. The resulting system, PCC-SQL, applies a per-token logits mask that deterministically eliminates single-query column-use violations on the supported SQL fragment in a single decoding pass. Across three benchmarks and three open-source models, PCC-SQL achieves 0% Leakage Rate and Coverage up to 88.7% on Spider-CU, while staying within +10% tokens of direct prompting. We additionally assess semantic alignment with execution accuracy. 

---
# Evaluating Health Misinformation in Low-Resource Languages: Integrating Small Language Models with a Culturally-Sensitive Responsible NLP Framework (Bangla as a Case Study) 

**Authors**: Farnaz Farid, Raihan Alam, Al Al-Areqi, Farhad Ahamed, Muhammad Hassan Khan, Sadia Hossain, Irena Veljanova, Anika Tabassum Binte Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2607.12336)  

**Abstract**: Artificial Intelligence (AI) technologies, while serving as a foundational enabler for modern social media and digital health services, exert a bivalent effect by simultaneously acting as a combatant against and a spread vector for misinformation. A prevalent challenge in mitigating this issue arises in non-English contexts and low socioeconomic classes, where limited data hinders the training of AI models for effective detection. Consequently, culturally and linguistically diverse (CALD) communities struggle to access trustworthy health information through AI-driven tools. Current AI tools underperform due to a lack of training data and are largely unable to consider language nuances and traditions in non-English contexts. This research addresses these gaps by proposing a CALD-friendly AI-based health misinformation detector and providing a dashboard for medical professionals to analyse this misinformation, a critical step toward mitigating a growing concern among CALD populations. To this end, we conduct a series of experiments using a Bangla-translated health misinformation dataset to evaluate the performance of various Small Language Models (SLMs). SLMs are particularly relevant in this context given the frequent underperformance of Large Language Models (LLMs), which often stems from insufficient domain-specific knowledge and the prohibitive costs of resource-intensive fine-tuning. The results demonstrate that Phi-4 is the superior model, achieving an ideal balance between precision and recall in claim extraction. Then, to mitigate the limitations of SLMs, we design and test a novel health misinformation detection framework grounded in Responsible Natural Language Processing (NLP), which incorporates cultural sensitivity, potential for harm, and communication quality, thereby providing a holistic lens for evaluating misinformation in low-resource languages. 

---
# QUBO-Optimized Evidence Selection for Retrieval-Augmented Question Answering with Unconventional Solvers 

**Authors**: Rahul Singh, Madhav Vadlamani  

**Link**: [PDF](https://arxiv.org/pdf/2607.12334)  

**Abstract**: Retrieval-augmented question answering depends on selecting evidence passages that jointly support answer generation. However, many RAG pipelines rely on top-\(k\) ranking, where passages are selected mainly by individual relevance scores, even though multi-hop questions often require complementary evidence satisfying multiple information requirements. Recent LLM-based selectors address this by treating retrieval as set selection, but using an LLM for this intermediate stage can be costly and difficult to scale. In this work, we formulate evidence selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem. Given a question, candidate passages, and decomposed information requirements, our method constructs an energy function that balances relevance, requirement coverage, support strength, redundancy, complementarity, and compactness. Low-energy solutions correspond to compact evidence subsets that cover the needed requirements while avoiding unnecessary or repetitive context. The selected passages are then passed to a downstream language model for answer generation, separating combinatorial evidence selection from semantic answer generation. We evaluate the proposed QUBO selector on HotpotQA and compare it with LLM-based set selectors and non-LLM baselines including BM25, relevance top-\(k\), maximal marginal relevance, hybrid lexical--semantic ranking, greedy coverage, and random selection. The QUBO selector achieves competitive exact-match and token-F1 performance relative to LLM-based selectors while providing a solver-compatible formulation for structured evidence selection. These results suggest that multi-hop evidence selection can be cast as discrete optimization, opening a path toward RAG pipelines where LLMs are reserved for semantic processing and answer generation, while context selection is handled by Ising/QUBO-compatible solvers. 

---
# LakeQuest: A Three-Domain Benchmark for Grounded Question Answering across Data Lakes 

**Authors**: Michael Solodko, Steven Gong, Guangwei Yu, Satya Krishna Gorti, Jesse C. Cresswell, Victor Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2607.12310)  

**Abstract**: While modern question answering (QA) systems excel on clean, schema-aligned corpora, real-world knowledge is rarely so neatly packaged. Answering questions over enterprise and scientific data lakes requires systems to navigate heterogeneous, weakly structured collections of tables, passages, and linked metadata. Current benchmarks abstract away this noisy discovery process, failing to evaluate end-to-end performance. To bridge this gap, we introduce LakeQuest, a human-validated benchmark of 9,846 QA pairs designed to evaluate the end-to-end retrieve-and-synthesize pipeline over realistic data lakes. LakeQuest spans three diverse domains (AI/ML metadata, retail banking, and multimodal biomedical drug information) and pairs every question with exact, modality-aware evidence pointers. By isolating source discovery from cross-modal synthesis, LakeQuest exposes critical failure modes in modern QA systems. Our baseline evaluations, including standard Retrieval-Augmented Generation (RAG) and agentic tool-use methods, reveal that high-quality retrieval does not guarantee correct reasoning. Systems consistently struggle with relation chaining in metadata graphs, policy grounding in bank ledgers, and joint tabular QA in biomedical contexts, highlighting the need for robust discovery and faithful cross-file composition mechanisms in future agentic QA systems. 

---
# A Shared Subcircuit Lets LLMs Count Down Across Tasks 

**Authors**: Jacob Dunefsky, Wes Gurnee, Emmanuel Ameisen  

**Link**: [PDF](https://arxiv.org/pdf/2607.12279)  

**Abstract**: Writing a sentence of exactly twelve words; ending a DNA sequence at the right codon; formatting an ASCII table. These are all tasks that language models can do that requires tracking how many tokens remain before a target. In this work, we identify in Llama-3.1-70B-Instruct a general mechanism for performing these tasks: a "countdown subcircuit" that compares the current position to a goal length and estimates the time remaining until then. We first isolate a countdown subcircuit in a controlled setting, in which the model is tasked with writing a fixed-length sentence ending in a specified word. We then investigate the geometry of the representations used by the subcircuit, and find that the subcircuit uses an identical motif previously identified in a frontier LLM on a separate task, thus suggesting that this motif is shared across models. Finally, we use unsupervised probing on a natural language dataset to find a variety of other tasks where this subcircuit is used, including tasks where the goal length is inferred from context rather than explicitly stated. Our work suggests that reverse-engineering subcircuits allows us to understand how behaviors generalize from a single example to many different tasks and even models. 

---
# FinResearchBench II: A Deep Research Benchmark with Consensus-Derived Gold Rubrics for Distinguishing Financial Report Quality 

**Authors**: Beidi Luan, Rui Sun, Sinuo Wang, Yan Gu, Chao Li, Zhenliang Xiong, Jing Li, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2607.12252)  

**Abstract**: Deep research agents are increasingly used to produce long-form financial reports, yet large-scale evaluation remains bottlenecked by the need for human experts to define and execute high-quality rubrics. We address this problem by proposing a scalable pipeline for generating high-quality rubrics without human experts in the final loop. We build a financial deep research benchmark from 104 real-world user queries and automatically synthesize 14,450 query-specific candidate rubrics from model-generated reports. To justify removing human experts from rubric execution, we compare rubric judgments from three human experts with those from a three-LLM judge panel on a sampled subset, and show that LLM-based evaluation is sufficiently consistent with human evaluation to replace it for large-scale rubric screening, including 98.67\% label-level agreement on jointly unanimous items. We then derive consensus-derived gold rubrics through two filters: a strict consistency filter, which keeps a rubric only if the three LLM judges unanimously agree on every report under the same query, and a distinguishability filter, which keeps a rubric only if it assigns at least one majority-yes and at least one majority-no label across the evaluated systems. This process retains 3,687 consistency-passed rubrics, of which 2,600 remain distinguishable and form the final set of consensus-derived gold rubrics. Using this final rubric set, we obtain clearly differentiated rankings across 10 deep research systems, with item-level pass rates ranging from 58.58\% to 22.23\%. More broadly, because the pipeline removes human-expert execution from rubric generation and evaluation, it is naturally scalable for benchmark evaluation, automatic system comparison, and future studies of evaluation-driven system improvement. 

---
# Fin-Analyst at FinMMEval 2026 Task 3: A Live Hybrid Trading Agent with LLM Specialists and Rule-Based Signals 

**Authors**: Mohotarema Rashid, Lingzi Hong, Junhua Ding, K.S.M.Tozammel Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2607.12233)  

**Abstract**: Large language model (LLM) trading agents show promising performance in equity markets, yet remain narrowly focused on US equities with little evidence from live deployment. We present Fin-Analyst, a hybrid agent for FinMMEval 2026 Task 3: an eight-specialist LLM pipeline over news, SEC filings, fundamentals, analyst forecasts, technical indicators, and social sentiment, aggregated by a Meta-Agent for Tesla (TSLA), and a lightweight rule based three-signal vote for Bitcoin (BTC). On the final official leaderboard (accessed 2026-07-05), Fin-Analyst ranks first of all agents on TSLA with a +13.51% return, +28.33 points over Buy-and-Hold (Sharpe 4.10, 88% win rate), while the BTC vote ends flat yet well above a sharply falling baseline. Relative to the interim performance, the asset ranking reversed, indicating that short live windows yield volatility-sensitive rankings. Ablation identifies event-driven 8-K disclosures as the most influential TSLA signal. Error analysis shows that the memoryless agents repeat wrong calls for days at a time, and that the fixed-threshold BTC rules lost money by trading on noise in a sideways market while the LLM pipeline gained under similar conditions, motivating a memory-aware, LLM-based successor for both assets. 

---
# RCWT: Measuring Task-Budget Displacement from Coordination Content in LLM Calls 

**Authors**: Brenda Lelis, Rodrigo Cabral-Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2607.12216)  

**Abstract**: Multi-agent and memory-augmented LLM systems often place coordination content, shared state, prior discussion, tool outputs, summaries, and role instructions, inside the same finite prompt used for the current task. This creates a practical allocation problem: every token spent on coordination is unavailable to task instructions or evidence when a call is assembled under a fixed context budget. We introduce the Roundtable Context Window Test (RCWT), a controlled protocol for measuring this task-budget displacement effect. RCWT varies coordination content while controlling total budget, position order, task family, and scoring. In the main context-dependent recall task at $W=4096$, three commercial models remain near baseline through moderate overhead and then degrade sharply once residual reference evidence falls to a few hundred tokens. Window-scaling summaries are consistent with a task-specific residual-budget interpretation rather than a fixed percentage threshold, but we treat this as descriptive evidence rather than a universal law. To test whether the fixed-budget cliff persists when task evidence remains intact, we add an intact-task ablation: the full task/reference block is kept present while coordination tokens increase by expanding total prompt length. In that setting, all tested calls return every scored field correctly across GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash up to a 95\% coordination ratio. This ablation narrows the claim: the main RCWT cliff is best read as task-budget displacement, not as proof that coordination volume alone causes semantic interference in the original open-ended task. RCWT is therefore a measurement primitive for context-allocation budgeting, not a complete theory of multi-agent benefit or session-level coordination. 

---
# Fine-Tuned Multi-Agent Framework for Detecting OCEAN in Life Narratives 

**Authors**: Rasiq Hussain, Darshil Italiya, Joshua Oltmanns, Mehak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2607.12215)  

**Abstract**: Accurately assessing personality from text is challenging because traits are latent, context-dependent, and often subtly expressed across long narratives. Large language models (LLMs) offer new opportunities by processing extensive textual contexts, but pretraining of these models can induce latent "personality-like" biases, making single-model inferences inconsistent. We propose a fine-tuned multi-agent framework for detecting OCEAN personality traits, in which sub-agents are conditioned to adopt high, low, or neutral perspectives for each trait through masked language modeling (MLM) and psychometric supervision. A judge LLM aggregates and compares sub-agent outputs to generate final trait predictions, capturing multiple complementary perspectives while mitigating individual model biases. We evaluate the framework on life narrative dataset through quantitative and qualitative experiments, including baselines, ablations, and inference quality analyses. Our approach offers a scalable and interpretable method for text-based personality inference, highlighting the benefits of multi-agent reasoning grounded in psychometric supervision. 

---
# Comparing Semantic Navigation in Humans and Large Language Models using Natural Language Processing 

**Authors**: Gabriel Paris-Colombo, Rodrigo M. Cabral-Carvalho, Felipe D. Toro-Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2607.12195)  

**Abstract**: Semantic memory retrieval can be conceptualized as navigation through conceptual space. We compared semantic search dynamics between humans and three large language models (GPT-4o, Gemini-2.5-Pro, Claude-Sonnet-4.5) using verbal fluency data. By applying trajectory-based NLP metrics to the items generated by 82 human participants and LLM output across eight temperature settings, we quantified three complementary dimensions: entropy (step size predictability), distance to next (successive semantic steps), and distance to centroid (global dispersion). Humans exhibited higher entropy, larger semantic steps and broader dispersion than all LLMs, indicating more variable and exploratory search. Temperature tuning produced only partial alignments, as individual metrics matched between humans and LLMs at specific settings, but no configuration reproduced the complete human profile (in all dimensions). These findings suggest that human semantic search implements a distinctive balance between local exploitation and global exploration that current model architectures fail to reproduce. 

---
# Entropy in Semantic Memory Navigation in Blind and Sighted Individuals: The Effect of Visual Experience 

**Authors**: Felipe D. Toro-Hernández, Rodrigo Lagos, Sergio E. Chaigneau  

**Link**: [PDF](https://arxiv.org/pdf/2607.12185)  

**Abstract**: Embodied accounts of semantic memory highlight the role of sensorimotor systems in acquiring and storing knowledge. Congenitally blind populations offer a critical test bed for these assumptions, providing an opportunity to assess whether conceptual grounding requires visual experience. In this study, we assessed semantic memory navigation differences between blind and sighted individuals using a property listing task with concrete and abstract concepts. We computed semantic entropy, an embedding-based natural language processing metric that captures the predictability of retrieval. Generalized linear mixed models revealed distinct navigation patterns across groups: while sighted individuals showed higher entropy for abstract than concrete concepts, blind participants did not. Instead, blind individuals exhibited higher entropy for visually salient concrete concepts (e.g., penguin). These results underscore the role of visual experience in the organization and dynamic navigation of semantic memory. 

---
# We Hebben Een Serieus Translatie: Modeling Intercomprehension as Probabilistic Inference 

**Authors**: Thomas Hikaru Clark, Edward Gibson, Roger Levy  

**Link**: [PDF](https://arxiv.org/pdf/2607.12169)  

**Abstract**: Intercomprehension refers to partial intelligibility of an unfamiliar language (L2) by a speaker of a related language (L1). How is this zero-shot cross-language comprehension possible? In this work, we extend past work on algorithmic models of noisy-channel inference to model intercomprehension in a Bayesian framework. The model uses an LM in L1 only for scoring latent hypotheses about the translations of observed L2 utterances, and a general-purpose noise model to infer a mapping between L2 and L1 words based on either form-based similarity or symbolic rules. We then conduct a human behavioral experiment, eliciting inferences for utterances in Dutch, Italian, and Ukrainian from speakers of English, Spanish, and Russian, respectively. Our full model shows a closer alignment to the distribution of human intercomprehension performance than ablations, and also compares favorably to zero-shot prompting of much larger models. These results provide a cognitively plausible computational model of intercomprehension, and highlight the flexible inferences made by comprehenders under wide uncertainty in real-world cross-language scenarios. We share our code publicly. 

---
# Token Reduction Is Not Cost Reduction 

**Authors**: Sarel Weinberger, Amir Hozez  

**Link**: [PDF](https://arxiv.org/pdf/2607.12161)  

**Abstract**: Context-reduction layers for API-based coding agents, including command-output compressors, retrieval rankers, and payload-optimizing proxies, are usually evaluated by how much text they remove. We ask instead: when does reducing retrieved context or tool output lower the actual billed cost of a coding agent without reducing task success or lengthening its trajectory?
Our primary evidence is a pre-specified, hash-frozen, paired campaign of 2,908 provider-billed Claude Code runs, of which 2,848 were analyzed, covering 103 tasks, seven repositories, and three models. The campaign compared a baseline with two generations of hook-based compression and an API-boundary proxy, within a broader measured program of roughly 5,500 billed executions.
Three findings emerge. First, prompt-cache traffic dominated cost composition. Cache creation and reads accounted for about 87% of reconstructed four-component cost, or about 80% of the actual bill, with an 8.7% dollar-weighted residual that retained telemetry could not attribute. On Haiku 4.5, this residual scaled with thinking effort.
Second, tool-output reduction did not reliably predict billed-cost reduction. An arm that removed 38% of estimated raw tool-output tokens had 6.8% higher paired cost (95% CI: +2.8% to +11.3%), while per-task reduction was only weakly associated with cost change (Pearson r = 0.15, CI crossing zero).
Third, compression can harm task completion by removing action-critical evidence. In a small single-shot study on SWE-bench-derived Go tasks, compression reduced patch application from 27/40 to 15/40 by corrupting verbatim edit anchors, and the compressed grounded arm produced fewer solves at higher observed cost per solve.
We propose a layered evidence standard centered on success-adjusted billed cost rather than token reduction alone. 

---
# CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban Simulation Platform 

**Authors**: Gustavo H. Santos, Aline Viana, Thiago H Silva  

**Link**: [PDF](https://arxiv.org/pdf/2607.12086)  

**Abstract**: Recent LLM-based multi-agent urban simulators can generate semantically rich city routines, but they remain costly to scale and are often weakly validated against empirical mobility patterns. We present CityBehavEx, an interactive LLM-assisted urban simulation platform that scales to city-size populations, exposes agent behavior for inspection, supports empirical validation, and generates mobility patterns that better match real-world spatial, temporal, and semantic distributions. Instead of invoking large language models for every agent action, CityBehavEx combines established human mobility models with fine-tuned cross-encoders that estimate semantic alignment between agent profiles, schedules, and activity transitions. This design enables large-scale simulations, as demonstrated in a case study of 100,000 agents over 75 days in under one hour on a single consumer GPU. The platform allows users to define simulation regions, launch experiments, inspect trajectories and activity traces, debug unrealistic behaviors, and validate generated routines against real-world mobility, time-use, and semantic metrics. 

---
# The Capacity of Thought: Benchmarking Llama 3.2 in Semantic fMRI Neural Language Decoding and Improving the Huth Encoding-Model Baseline 

**Authors**: Milos Suvakovic, Dom Marhoefer, Glenn Grant-Richards, Aidan Pinero  

**Link**: [PDF](https://arxiv.org/pdf/2607.12079)  

**Abstract**: Decoding continuous language from fMRI signals remains a core challenge in non-invasive brain-computer interface research. We present two complementary investigations. First, we improve the Huth et al. ridge regression encoding pipeline through expanded voxel selection (10K->15K), substitution of GPT-2 medium for GPT-1 as the beam-search proposal model, and GPU-accelerated bootstrap training, achieving mean METEOR = 0.149 and BLEU-1 = 0.200 across three held-out narratives for subject UTS03 -- an 11% relative METEOR gain over our replication baseline. Second, we introduce fMRIFlamingo, which maps BOLD activity to a frozen Llama-3.2-1B with trainable gated cross-attention layers via a learned brain tokenizer and a Perceiver Resampler. Despite achieving 42.86% Top-1 accuracy on a 1-in-100 ranking task, well above chance, a blind control ablation with zeroed fMRI inputs yields near-identical scores, revealing that apparent decoding success is driven primarily by the frozen language prior rather than by neural input. These results demonstrate that high-capacity language models do not inherently improve fMRI decoding and can actively obscure failures without rigorous blind-control evaluation. 

---
# Beyond Parallel Tracking: Interactive Multi-Feature Fusion Drives Semantic Reconstruction from Non-invasive Brain Recordings 

**Authors**: Boda Xiao, Xiran Xu, Songyi Li, Yujie Yan, Xihong Wu, Heping Cheng, Jing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.12071)  

**Abstract**: Continuous semantic reconstruction from non-invasive neural recordings remains limited by the representational mismatch between semantic feature spaces and neural coding patterns, which severely impedes cross-modal alignment between high-noise neural signals and target semantic features. Prior semantic decoders have predominantly relied on static lexical representations or dynamic contextualized representations in isolation. This single-dimension approach inevitably leads to severe information loss, as it fails to account for the human brain's capacity to integrate stable word attributes and dynamic contexts this http URL bridge this gap, this study introduces a multi-feature fusion framework for non-invasive semantic reconstruction, systematically benchmarking two integration approaches: linear Naive Concatenation and non-linear Multi-Head Cross-Attention. Within this framework, our approach complements static lexical representations (W2V) with dynamic contextual representations (GPT) via an interactive gating mechanism to facilitate cooperative processing during language this http URL through extensive semantic reconstruction and text generation experiments, our framework reveals a robust performance hierarchy: Cross-Att > Concat > GPT > W2V. Crucially, the non-linear cross-attention fusion method achieves state-of-the-art performance, demonstrating that neural language decoding benefits from simulating the collaborative modulation between contextual information and core lexical attributes rather than depending on isolated individual features, while also offering a viable non-invasive brain-to-text decoding method. 

---
# Agentic systems for breast cancer treatment recommendations 

**Authors**: Vinicius Anjos de Almeida, Nícolas Henrique Borges, Leonardo Vicenzi, Helena Kociolek, Sarah Miriã de Castro Rocha, Frederico Nassif Gomes, Júlia Cristina Ferreira Ribeiro, Lucas Emanuel Silva e Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2607.12051)  

**Abstract**: Large language models (LLMs) are increasingly being explored for clinical decision support, but their reliability in complex oncology treatment planning remains unclear. We evaluated agentic LLM systems for breast cancer treatment recommendation generation using 72 real clinical cases across stages I to IV and 1,147 case-specific rubrics generated through Asymmetric Information Rubric Generation (AIRG), in which the rubric generator had access to real clinical decisions unavailable to the evaluated models. Seven pipelines were compared, including single-LLM baselines, tool-augmented systems, and multi-agent architectures with fact checking and autonomous subagent spawning. The best-performing configuration, Claude Opus 4.8 with the D&C+SA pipeline, achieved a global score of 0.594 $\pm$ 0.025. Tool use and increased agent autonomy had mixed effects, improving performance in some settings but degrading it in others. Performance varied by clinical domain and disease stage, and oncologist-led error analysis revealed persistent clinically relevant failures, including incorrect or missing recommendations, flawed justifications, citation errors, outdated claims, and overconfidence. These findings suggest that agentic LLM systems can generate clinically relevant breast cancer recommendations, but remain insufficient for unsupervised clinical use. 

---
# Evaluating Nonuniform Dependability Across Response Conditions: A Conditional Generalizability Framework Illustrated in Automated Essay Scoring 

**Authors**: Yi Gui  

**Link**: [PDF](https://arxiv.org/pdf/2607.11981)  

**Abstract**: Aggregate reliability estimates can obscure heterogeneity in measurement-design burden across response conditions, so a single G- or D-study may mischaracterize a design's adequacy for particular strata. This study introduces a conditional generalizability framework with three components. First, automated scoring configurations -- the encoder architectures and scoring-head families admissible within a fixed pipeline -- are treated as a universe of admissible measurement conditions rather than incidental modeling choices. Second, analytical D-study projections are compared with empirical configuration sweeps over a finite scoring pool, yielding two estimands of design adequacy whose agreement or divergence diagnoses the realized configuration universe. Third, evidence is conditioned on entropy-defined response strata, treating entropy as an operational stratification variable, not a construct claim about writing quality. Whereas recent generalizability-theory extensions address AI-generated item variants on the response side, this framework addresses the analogous scoring-side problem: AI-mediated scoring configurations. Demonstrated with automated essay scoring of timed L2 writing, the realized design was dependable in aggregate (Phi approx 0.76). Re-estimated within entropy strata, dependability stayed high but declined modestly and robustly (Phi = 0.88, 0.87, 0.84) -- a gradient implying different decision-study requirements, the highest-entropy stratum requiring the most crossed conditions. The framework offers a portable workflow for evaluating nonuniform dependability. 

---
# Hybrid Continual Learning for Low-Resource Australian Aboriginal Language Identification 

**Authors**: Pravina Mylvaganam, Ting Dang, Eliathamby Ambikairajah, Vidhyasaharan Sethu, Jingyao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11946)  

**Abstract**: Language identification is an important step toward integrating endangered Australian Aboriginal languages (AALs) into speech technologies supporting language revitalisation and digital inclusion. However, extreme data scarcity limits model performance. Transfer learning from high-resource languages shows promise but often suffers from catastrophic forgetting when adapting to new languages. Continual learning (CL) can mitigate this issue, though it remains challenging with very limited data. To address this, we propose two hybrid continual learning methods: Replay Augmented Elastic Weight Consolidation and Constraint Guided Knowledge Distillation to adapt pretrained speech models for AAL identification while preserving previously learned knowledge. Experiments on Warlpiri, Dalabon and Dharawal show that the proposed methods outperform fine-tuning and existing CL baselines, improving adaptation to multiple AALs while maintaining performance on previously learnt high-resource languages. 

---
# Belief-reality separation lives in routing over a shared value slot in language models 

**Authors**: Oliver Steele, Jiangtao Wen, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.11945)  

**Abstract**: Capable language models hold what a character believes apart from what is true: told "Anna believes the cup is blue; in reality it is red," they answer blue about Anna and red about the world. Where in the computation does that separation live? We show it rests on two separable mechanisms at two positions. A generic value slot binds the attributed value. A router at the query position selects which frame, the character's belief or reality, a query reads out. Two routes fill the slot: an asserted belief, whose value the text supplies, binds in directly; a derived belief, whose value must be inferred from what the character could see, arrives by a visibility-gated lookback. A subspace trained on either route steers the other, and only the derived route depends on described visibility. The slot itself carries no belief-reality tag: intervening on it moves a reality readout as strongly as a belief one. The separation lives instead in a dissociated pair of routing subspaces, which flip a query between frames without injecting the donor's value. These results hold across three architectures, on stimuli de-confounded against theory-of-mind-benchmark shortcuts; the behavior itself emerges between 3B and 7B across five model families. This paper develops the single belief-reality axis in depth; a companion paper shows the same slot-and-router format is shared across the other non-actual contexts a sentence can open (counterfactual, fictional, temporal). 

---
# MAGE: Understanding Stability-Performance Trade-offs in Multi-component Prompt Optimization 

**Authors**: Prateek Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.11944)  

**Abstract**: How do different components of iterative prompt optimization interact, and what happens when they are combined? We investigate this through MAGE (Memory-Augmented Goal-directed Prompt Evolution), a controlled analysis framework for studying component interaction in prompt optimization. MAGE is not proposed as a superior optimizer in absolute terms; it integrates episodic memory, multi-objective Pareto selection, and adaptive evaluation as a platform for controlled ablation. Our experiments uncover a previously unreported phenomenon, the Prompt Optimization Coupling Effect (POCE): when multiple stochastic optimization signals operate within a closed reflective loop, they interact in ways that simultaneously improve performance and amplify variance, behavior that cannot be predicted by analyzing components in isolation. Three main findings emerge. First, failure-grounded reflection is essential: methods relying only on scores (OPRO) or abstract critique (Self-Refine) fail to improve prompts. Second, MAGE achieves 46.4% versus GEPA's 34.0% on GSM8K-Hard (+12.4%, P(MAGE>GEPA)=0.998, 5 seeds on gpt-4o-mini), with comparable variance (7.3% vs. 7.0%). Third, increasing candidate diversity reveals the clearest POCE signal: expanding the candidate pool from n=3 to n=5 improves mean accuracy by +21.6% while increasing variance by 3.7x. We further validate on Llama 3.1 8B and show POCE is headroom-dependent: when the base model already achieves high accuracy, variance amplification disappears. Finally, in low-data regimes (Ntrain=30), well-designed fixed prompts outperform all reflective optimizers, indicating that scaffold choice dominates optimizer choice. Our results suggest prompt optimization systems behave as coupled stochastic processes and should be evaluated in terms of both performance and stability, not just peak accuracy. 

---
# Transforming LLMs into Efficient Cross-Encoders via Knowledge Distillation for RAG Reranking 

**Authors**: Shreeya Dasa Lakshminath, Shubhan S  

**Link**: [PDF](https://arxiv.org/pdf/2607.11933)  

**Abstract**: Cross-encoders achieve high reranking accuracy in Retrieval-Augmented Generation (RAG) pipelines but impose quadratic inference costs that limit real-time deployment. We address this by fine-tuning LLaMA 3 (8B) as a drop-in reranker using a two-stage pipeline: supervised fine-tuning on a custom query-document relevance dataset via the Unsloth framework with LoRA adapters, followed by 4-bit quantization for efficient inference. The resulting model replaces the cross-encoder in a dual-retriever RAG pipeline combining BM25 and dense vector search. Evaluated on a domain-specific question-answering benchmark using the RAGAS framework, our fine-tuned LLaMA 3 reranker achieves gains of 14% in answer relevancy, 16% in context precision, 19% in answer similarity, and 21% in answer correctness over the cross-encoder baseline, while reducing inference overhead through 4-bit quantization. These results demonstrate that instruction-tuned LLMs can be adapted into accurate, efficient rerankers without the quadratic complexity of traditional cross-encoders. 

---
# TAKE: Trajectory-Aware Knowledge Estimation for Text Dataset Distillation 

**Authors**: Tri-Nhan Vo, Dang Nguyen, Sunil Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2607.11898)  

**Abstract**: Large-scale text corpora have become a quiet bottleneck in modern NLP, not just in storage, but in the accumulated cost of training, fine-tuning, and continual learning. We propose a text dataset distillation framework that reduces corpora to as little as 0.1% of their original size while preserving downstream task fidelity. We approach distillation through the lens of influence functions, which quantify each sample's contribution to the downstream objective, a natural and principled basis for selection. We introduce Trajectory-Aware Knowledge Estimation (TAKE), which convolves the knowledge-based influence along the training trajectory into a single per-sample knowledge score, capturing informative samples. These scores serve as sample weights within a discrete Optimal Transport objective, guiding prototype selection from a synthetically generated candidate pool. We evaluate TAKE on downstream accuracy across text classification and natural language inference tasks at extreme compression (0.1% or 20 samples/class), showing that data efficiency is achievable without sacrificing task fidelity. The approach is theoretically grounded, with broader implications for coreset construction and data-centric AI. We release our source code at this https URL. 

---
# Graph-Based Detection of Disinformation Narrative Diffusion between Russian and Ukrainian Telegram Channels 

**Authors**: Yuliia Vistak, Viktoriia Makovska, Vera Schmitt, Veronika Solopova  

**Link**: [PDF](https://arxiv.org/pdf/2607.11894)  

**Abstract**: Detecting disinformation narratives on social media is challenging due to the scale of amplification, rapid evolution, and linguistic variability of online content. We propose a graph-based framework for identifying and analyzing disinformation narratives in Telegram ecosystems by combining weak supervision with propagation graph analysis. The approach aggregates semantically related claims into narrative-level clusters and models their diffusion across interconnected channels. This enables the detection of coordinated narrative amplification that is difficult to capture through post-level analysis alone. Our results demonstrate that integrating textual signals with network structure provides a scalable method for detecting disinformation narratives and offers insights into how they propagate within large-scale messaging environments. 

---
# I'm Sorry, but I Can't Help with Braille: Revealing Accessibility Failures in State-of-the-Art LLMs 

**Authors**: Abdullah Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2607.11893)  

**Abstract**: Large Language Models (LLMs) perform strongly on many language tasks, but their capability in structurally constrained, accessibility-critical modalities such as Braille remains unclear. We evaluate state-of-the-art LLMs on bidirectional Korean-Braille translation using a human-annotated dataset. Despite expectations that multilingual, instruction-tuned models can generalize to Braille via text representations, we find consistently poor, unstable outputs and substantial disagreement with human judgments. These results point to missing Braille-aware tokenization and weak alignment between Korean and Braille patterns. In contrast, supervised fine-tuning of a small model (T5-small) on the same data yields large and stable gains over zero-shot and prompted LLM baselines across standard metrics (SacreBLEU, ChrF++, CER, BLEU, ROUGE-L, METEOR, CIDEr). Our findings reveal a systematic limitation of current LLMs and demonstrate the effectiveness of modest task-specific supervision. 

---
# G-SHARE: A Guideline-Based Structured Reasoning Framework for Human-Factor Event Diagnosis 

**Authors**: Xingyu Xiao, Mao Du, Jiejuan Tong, Jingang Liang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.11892)  

**Abstract**: Human-factor event diagnosis is essential for learning from operational events in nuclear power plants, yet its quality depends strongly on expert interpretation of narrative reports and guideline-based this http URL data-driven or one-shot large language model approaches often lack structured reasoning, have limited alignment with formal diagnostic guidelines, and may generate logically inconsistent conclusions. To address this issue, this study proposes G-SHARE, a guideline-based structured reasoning framework that operationalizes the CNNP nine-step human-factor event diagnosis guideline into a multi-stage diagnostic this http URL framework consists of evidence extraction, stepwise diagnostic reasoning, and post-hoc consistency repair, enabling explicit use of report evidence, intermediate rationale generation, and logical validation of diagnostic outputs. A dataset of real human-factor event reports was constructed from Chinese nuclear industry sources, and a gold-standard subset annotated by domain experts was used for evaluation. Results show that G-SHARE substantially outperforms one-shot prompting and traditional machine learning baselines, with the strongest version achieving the best overall accuracy and macro-F1. Ablation results further indicate that structured reasoning and consistency enforcement are critical to robust diagnosis, especially under weak prompting conditions. The findings demonstrate the value of transforming expert diagnostic guidelines into auditable reasoning workflows, providing a practical pathway for intelligent human-factor analysis in safety-critical industries. 

---
# CANDI: Contextual Alignment for Niche Domains Question Answering 

**Authors**: Megha Chakraborty, Darssan L. Eswaramoorthi, Het Riteshkumar Shah, Madhur Thareja, Michelle A Ihetu, Harshul Raj Surana, Kaushik Roy, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2607.11891)  

**Abstract**: The deployment of large language models (LLMs) in specialized domains like medical diagnostics and financial advisory necessitates evaluating capabilities beyond general knowledge. Traditional question-answering benchmarks often fail to capture the nuanced contextual grounding, user awareness, and domain understanding these fields require. To address this, we introduce CANDI-QA (Contextual Alignment for Niche Domains Question Answering), a novel dataset evaluating LLMs on delivering accurate, context-sensitive, and user-aligned answers in specialized settings. CANDI-QA features expert-curated question-answer pairs structured into two categories: (1) Information Assistance Questions, which are direct, factual queries requiring precise extraction, and (2) Applied Inference Questions, which are multi-hop reasoning tasks needing situational inference to generate actionable insights. We evaluate over ten diverse language models, from compact open-source to state-of-the-art proprietary systems. As a robust baseline, we present MTSS-Net, a lightweight neuro-symbolic framework combining neural retrieval with rule-based reasoning. Our findings highlight the profound challenges of achieving contextual alignment in niche domains, revealing the limitations of current LLMs without enhanced contextual or symbolic integration. Ultimately, CANDI-QA serves as a critical benchmark for advancing research in context-aware language models, stimulating the development of robust, trustworthy AI for high-stakes domains. 

---
# Scaling Point-in-Time Language Models 

**Authors**: Bryan Kelly, Semyon Malamud, Johannes Schwab, Teng Andrea Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.11889)  

**Abstract**: Large language models trained on unrestricted internet corpora inevitably embed information from the future, introducing lookahead bias that compromises the validity of backtests and causal inference in finance and the social sciences. Point-in-time language models--trained exclusively on text available up to each calendar date--eliminate this leakage by construction, but existing efforts typically produce models that lag substantially behind their unconstrained counterparts. We show that this performance gap can be substantially narrowed through scale. Training decoder-only transformers with up to 4 billion parameters on 1 trillion chronologically filtered tokens from FineWeb, we construct a sequence of monthly model checkpoints spanning 2013-2024. Across a range of common-sense reasoning and language understanding benchmarks, our models approach the performance of leading open-weight models of comparable size (e.g., Gemma-3-4B and LLaMA-7B) trained on temporally unrestricted data, although a performance gap remains on several tasks. Instruction fine-tuning via LoRA further improves downstream usability. We release the complete pipeline--including dataset construction, training infrastructure, and evaluation code--to enable reproducible point-in-time language modeling and to support research applications that require strict temporal validity. 

---
# Do AI Agents Know When a Task Is Simple? Toward Complexity-Aware Reasoning and Execution 

**Authors**: Junjie Yin, Xinyu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.13034)  

**Abstract**: Large language model (LLM) agents increasingly automate multi-step engineering and informatics workflows, yet they rarely ask how much effort a task actually requires. They often follow a maximum-context-first strategy--re-reading files and dependencies they have already seen--turning a one-line edit into a small code-base audit. We argue the missing capability is task-aware execution-scope estimation: judging a task's difficulty, the information it truly needs, and the shortest reliable path before committing budget. We formalize minimum-sufficient execution and the Agent Cognitive Redundancy Ratio (ACRR), and propose E3 (Estimate, Execute, Expand): the agent estimates an initial operating point, executes a minimum viable path, and expands scope only when verification fails. On MSE-Bench--a deterministic benchmark of 121 edits in a capability-controlled simulator--E3 matches the strongest baseline's 100% success while cutting cost by 85%, tokens by 91%, and inspected files by 92%, and further beats a strong adaptive retrieval baseline by 16%; the gains survive held-out instruction wording and essentially every cost weighting. A companion real-model harness (LLM-Case) corroborates the effect on a live gpt-4o agent editing a real open-source library, with every candidate patch graded by actually running the project's real pytest suite against a measured oracle: the over-reading is milder but real, and E3 is the leanest and fastest policy at comparable task success--its one shortfall a provider rate-limit, not a wrong edit. We frame this as a controlled probe of execution redundancy, not a measurement of any deployed agent, and position task-aware execution as a step toward engineering-grounded AI (EGAI)--agents whose effort is anchored in the engineering reality of the task. We release the framework and benchmark. 

---
# MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon Conversations 

**Authors**: Xixuan Hao, Zeyu Zhang, Zehao Lin, Yihang Sun, Ziliang Guo, Xichong Zhang, Yuxuan Liang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.12893)  

**Abstract**: Long-term memory has become a foundational capability for LLM-based agents that accompany users across extended, multi-session interactions. Existing benchmarks, however, evaluate such memory almost exclusively through downstream question answering, scoring only the correctness of a final answer. This black-box formulation conflates the heterogeneous causes of memory failure, such as missing the introduction of a relevant fact, binding an operation to the wrong target, or relying on stale values after a correction. As a result, it can credit correct answers despite their reliance on inconsistent or unsafe memory states. In this paper, we argue that, in dynamic long-horizon interactions, memory is not a static collection of facts but a lifecycle of explicit operations, including remembering, forgetting, updating, reflecting, and their compositions. We introduce MemOps, a benchmark that reformulates conversational memory as a sequence of lifecycle operations and represents each memory event with a structured trace specifying its trigger, target, scope, state transition, and supporting evidence. A controllable generation pipeline embeds these operations into long, task-oriented conversations and produces gold operation traces together with six categories of operation-level probes, evaluated under both adjacent-evidence and long-context settings. Across long-context, retrieval-based, parametric and managed-memory systems, MemOps disentangles failure modes that final-answer accuracy alone conceals, revealing that current systems remain far from uniformly reliable. For instance, session-level retrieval outperforms turn-level retrieval, and long-context models remain notably weak at reconstructing ordered memory-state trajectories. These results move long-term memory evaluation from final-answer scoring toward interpretable, operation-level diagnosis. 

---
# Accelerating Masked Diffusion Large Language Models: A Survey of Efficient Inference Techniques 

**Authors**: Daehoon Gwak, Minhyung Lee, Junwoo Park, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2607.12829)  

**Abstract**: Diffusion large language models (dLLMs) offer a theoretical advantage in parallel generation over standard autoregressive models. However, parallel generation alone does not guarantee practical speedups. Realizing this efficiency requires specialized inference mechanisms, such as diffusion-aware caching and reuse. Consequently, as inference efficiency becomes a prerequisite for practical deployment, recent research has actively explored acceleration techniques across algorithms, architectures, and systems. However, rigorous comparisons remain difficult, as end-to-end latency stems from intricate trade-offs between algorithmic, architectural, and system-level factors that are often conflated in existing benchmarks. In this survey, we introduce a unified latency decomposition framework for dLLMs to disentangle these factors and analyze their impact on inference speed in real deployments. Guided by this framework, we categorize acceleration techniques along three axes covering algorithmic innovations, architectural and system optimizations, and inference-time scaling. Finally, we provide guidelines for reproducible benchmarking and highlight open challenges for realizing the full potential of parallel generation. 

---
# Who Grades the Grader? Co-Evolving Evaluation Metrics and Skills for Self-Improving LLM Agents 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2607.12790)  

**Abstract**: Self-evolving agent systems improve by creating, revising, and retiring their own skills, but every such loop rests on a hidden assumption: a reliable evaluation metric already exists. In many real applications it does not. We make three claims. First, metrics can be \emph{evolved}: our metric loop searches compositions of small drawback detectors under a full evolutionary lifecycle, trained to agree with a ten-item anchored reference set, regularized by consensus over unlabeled outputs, and audited against a held-out anchor it never reads, yielding a transparent, inspectable metric rather than an opaque judge. Second, since no metric exists to beat, the yardstick is recovering what an accurate metric would have enabled, and \emph{Double Ratchet}, our co-evolution of the metric with a lifecycle-managed skill loop, does so: across code generation (MBPP+), enterprise text-to-SQL (Spider~2.0-Snow), and reference-free report generation, it retains 88--110\% of the held-out lift achieved by the same skill loop driven by ground truth or the best available rubric. Third, safety comes from anchor discipline plus outer audits: removing anchor guards collapses the metric into a vacuous detector while removing the lifecycle does not; and when evolved skills gamed the report rubric, an independent judge caught it, one detector repaired it, and a task-aware judge then preferred the evolved outputs over the pre-evolution baseline in 77\% of decided pairs. We argue this failure-expecting architecture is the right default wherever no reliable automatic verifier exists. 

---
# Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters? 

**Authors**: Kaiwen Zheng, Junchen Fu, Wenhao Deng, Hu Han, Joemon M. Jose, Xuri Ge  

**Link**: [PDF](https://arxiv.org/pdf/2607.12787)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have significantly improved the performance of multimodal emotion recognition (MER) and enabled interpretable description generation by jointly modeling video, audio, and language, etc. However, these performance improvements are often accompanied by an increase in model parameter size (e.g, at least 7B), which simultaneously incurs high computational costs and reduces inference efficiency, thereby hindering real-time deployment on resource-constrained platforms such as robots and mobile devices. This raises a fundamental question: do we really need the multimodal MER model larger than 1B parameters for high-quality MER?
In this paper, we challenge the assumption that larger models are inherently necessary and proposes a lightweight MER framework (called Light-MER), which achieves better and faster multimodal sentiment understanding and recognition through knowledge distillation. It can transfer knowledge from a strong, large-scale teacher model to a lightweight sub-billion-parameter student model, aiming to preserve rich multimodal emotion reasoning and recognition while substantially improving deployment efficiency. Specifically, we introduce two new optimization strategies to enhance knowledge transfer: (1) a new optimal transport loss that combines Sliced Wasserstein Distance with hidden-state alignment, and (2) a new multi-reward optimization strategy based on GRPO that balances MER performance and efficiency, aimed at further enhancing the learning capabilities of student models. Extensive experiments on nine benchmark datasets demonstrate that Light-MER achieves state-of-the-art performance while significantly improving inference efficiency. This highlights the strong potential of small multimodal emotion language models for future research. Code is available at this https URL. 

---
# Learning Mechanistic Reasoning for Chemical Reactions with Large Language Models 

**Authors**: Xingyu Dang, Haocheng Tang, Junmei Wang, Yanjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.12771)  

**Abstract**: Reaction mechanisms consist of the step-by-step sequences of elementary reactions that explain chemical transformations. Learning the mechanism logic is therefore essential for enhancing the fundamental chemical intelligence of large language models (LLMs). The stepwise deduction of reaction mechanism aligns naturally with the reasoning paradigms of reasoning LLMs. However, current chemical LLMs primarily emphasize coarse-grained name reactions for product prediction and retrosynthesis, often leading to physical inconsistencies and hallucinations. In contrast, specialized small-scale generative models for mechanism inference typically suffer from restricted generalization capacity across diverse chemical spaces. To overcome these limitations, we built a novel, large-scale reasoning dataset of reaction mechanisms. Furthermore, we established the FukuyamaBench, a difficult benchmark derived from Fukuyama's Advanced Organic Reaction Mechanism book, to rigorously evaluate model performance on hierarchical mechanism reasoning. Our fine-tuned Qwen3-30B-A3B achieves 8.3% exact pathway match on FukuyamaBench Set~A, surpassing the specialized FlowER model (5.1%), demonstrating that mechanism-aware training substantially enhances chemical reasoning in language models. 

---
# Tracing Agentic Failure from the Flow of Success 

**Authors**: Samuel Yeh, Yiwen Zhu, Shaleen Deep, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.12747)  

**Abstract**: Failure attribution for LLM-based agentic systems, i.e., identifying which steps in a failure trajectory caused the task to fail, is critical for debugging and improving these systems. Existing approaches either rely on prompting-based pipelines, which are computationally expensive, or require post-training on failure trajectories with step-level error annotations, which are costly to collect and difficult to scale. We argue that a practical failure attribution model should be lightweight and trainable without step-level supervision on failure data. To this end, we address unsupervised failure attribution, i.e., training exclusively on successful trajectories and identifying error steps at inference time given a failure trajectory. We propose OAT, which casts this problem as one-class learning with neural controlled differential equations, modeling the dynamical pattern of successful trajectories in latent space. At inference time, each step in a failure trajectory is assigned an anomaly score based on its deviation from the dynamics learned on successful trajectories, which is then used to form a set of error steps. With training on only 100 successful trajectories, experiments show that OAT is 200--5000 $\times$ faster than prompting-based baselines, and, at the same time, consistently outperforms them in both in-domain and out-of-distribution datasets with +20% and +7% F1 scores, respectively, demonstrating that OAT is a promising and efficient direction for diagnosing agentic system failures. 

---
# Extractable Memorization From First Principles 

**Authors**: A. Feder Cooper, Marika Swanberg, Jamie Hayes, Lea Duesterwald, Christopher De Sa, Daniel E. Ho, Mark A. Lemley, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12649)  

**Abstract**: Recent work on extractable memorization in LLMs suffers from two contrasting validity problems. Some studies overstate extraction, e.g., relying on sequences too short to distinguish memorization from predictability. Others imply that extraction is unreliable evidence of memorization, since models can also reproduce real-world text they weren't explicitly trained on. In different ways, both overlook what makes a valid extraction claim: the model must generate a training sequence with high enough probability to indicate memorization. To determine what's high enough, one has to perform a matched comparison: measuring the generation probabilities of both the training sequences of interest and comparable non-training sequences. Because non-training sequences cannot have been memorized, their probabilities provide a baseline for predictability; a training sequence exceeding this baseline provides evidence of memorization. We formalize matched comparisons in two ways: (1) a conformal test that calibrates a threshold to a chosen FPR when training and non-training sequences are sampled from populations, and (2) a census that calibrates against a matched non-training document when the object is a single document (e.g., a book). We show that matched comparisons enable rigorous, calibrated memorization claims, and reveal where prior setups have validity issues. For instance, on Wikipedia OLMo 2 32B reproduces non-training 10-token suffixes roughly 24% as often as training ones: that share of the training generation rate reflects false positives, not memorization. For Llama 3.1 70B on books, the thresholds we calibrate are as low as 1e-27, supporting memorization claims for sequences that no feasible sampling budget would extract. Based on these results, we refine "extractable memorization" to require a valid memorization claim and near-certain generation within a realistic budget. 

---
# A Learning-Rate-Gated Failure of GRPO in a Small Language and Vision-Language Model Web Agent: A Controlled Null and Its Mechanism 

**Authors**: Chengguang Gan, Zhixi Cai, Yunhao Liang, Hanjun Wei, Shiwen Ni, Qinghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12640)  

**Abstract**: Reinforcement learning with verifiable rewards, and Group Relative Policy Optimization (GRPO) in particular, is now run routinely on a supervised checkpoint in the hope of producing a stronger agent. We ask whether it adds skill to a small language and vision-language model web agent at the 4B to 8B scale, or whether it mostly reshapes behavior the supervised model already has. Across a control grid of 18 runs that varies learning rate, KL weight, seed, initialization, and clipping, no configuration credibly improves the success rate of a strong supervised baseline on tasks the agent has largely mastered. On the text track, moderate to high learning rates make it credibly worse. The null holds under paired testing, 25 evaluation seeds, 6 training seeds, changes to the recipe, both text and Set-of-Marks screenshot observations, and scaling the backbone to 8B; the credible harm is a text-track finding and is only nominal under Set-of-Marks. To show that the null reflects the setting and not a broken pipeline, we run the identical harness, reward, and recipe on tasks whose reward is reachable by sampling, and there the success rate rises by 22 points with a paired interval that excludes zero. GRPO therefore helps only when there is headroom to climb, meaning the sampled policy already succeeds more often than the greedy one. We then explain the failure. A middle learning rate degrades the agent and a high one collapses it, and the two regimes form a double dissociation: grafting localizes the degrade regime to the attention and MLP blocks, while the collapse regime cannot be traced to any single group, and the embedding change that dominates the weight movement is causally inert. At 4B, effective rank in the late layers tracks capability in both directions; at 8B the two come apart. This coupling is specific to the smaller model, so we report it as scale-dependent. 

---
# A JoLT for the KV Cache: Near-Lossless KV Cache Compression via Joint Tucker and JL-Residual Allocation for LLMs 

**Authors**: Rahul Krishnan, Volker Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2607.12550)  

**Abstract**: The key-value (KV) cache has become the dominant memory cost of transformer inference. It grows with batch size, context length, and depth, and at long context it, rather than the model weights, sets the ceiling on throughput. Two families of methods reduce it. Low-rank methods factor two-dimensional slices of the cache, either per-head matrices or cross-layer feature blocks, and quantization methods lower the bit-width of every entry. Neither family exploits the fact that the cache at a layer is naturally a third-order tensor whose three axes, the heads, the tokens, and the features, carry very different amounts of redundancy. We take this tensor view directly. Our method, JoLT, applies a partial Tucker decomposition that compresses only the token and feature axes while leaving the head and layer axes intact, and then restores the energy that truncation discards with a Johnson-Lindenstrauss (JL) rotated low-bit residual. A single Lagrangian dual allocates the Tucker ranks and the residual bit-widths together, per layer group and separately for keys and values, under one byte budget. The result is a near-lossless 2-3x compression: perplexity, GSM8K accuracy, and RULER needle-in-a-haystack retrieval all stay at or within statistical noise of the uncompressed baseline on both a grouped-query-attention model (Mistral-7B-v0.3) and a multi-head-attention model (LLaMA-2-13B). At 2x, JoLT reconstructs the cache to relative Frobenius error 0.009 (K) and 0.006 (V) on both architectures, roughly an order of magnitude below cross-layer SVD and 4-bit quantization. A randomized-SVD variant, FlashJoLT, delivers a 5-13x compression-time speedup at matched quality. 

---
# Function-Aware Fill-in-the-Middle as Mid-Training for Coding Agent Foundation Models 

**Authors**: Yubo Wang, Jiarong Liang, Yuxuan Zhang, Xuye Liu, Cong Wei, Yuyu Zhang, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.12463)  

**Abstract**: Coding agents must integrate external tool returns into ongoing reasoning - a capability that standard left-to-right pretraining on code exposes only in its forward direction. We observe that the action-observation-continuation loop of a coding agent is structurally isomorphic to a function call site, where a caller binds arguments, a callee returns a value computed elsewhere, and downstream code consumes that value. This conditioning structure exists at internet scale in ordinary code. We exploit it through function-aware fill-in-the-middle (FIM) mid-training: a self-supervised objective that masks functions selected via program dependency graph analysis and a complexity-inferability double criterion. We mid-train Qwen2.5-Coder-Instruct (7B/14B) and Qwen3-8B on a 2.6B-token decontaminated corpus drawn from 968 GitHub repositories, then apply existing agentic post-training pipelines. Mid-training improves SWE-Bench-Verified by +2.8/+3.0 at 7B/14B and by +3.2 on Qwen3-8B; SWE-Bench-Lite gains are +3.7/+4.0/+5.4 on the same models. The improvement holds across two post-training pipelines (R2E-Gym, SWE-Smith) and on a non-Qwen2.5 base (Qwen3-8B with SWE-Lego). Beyond in-domain gains, mid-training also mitigates the capability erosion that agentic post-training otherwise inflicts on non-agent coding (e.g., LiveCodeBench) and non-coding tool-use benchmarks (tau-bench, BFCL): although the mid-training corpus contains Python code only, the function-call inductive bias survives post-training and yields consistent gains. 

---
# The Sound of Absence: Audio-Language Embedding Models Struggle with Negation 

**Authors**: Chun-Yi Kuan, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.12290)  

**Abstract**: Audio-language embedding models such as CLAP are widely evaluated on matching present sound events, but rarely on negation. We show this affirmation-only evaluation hides a key limitation: these models fail to encode negated sound concepts, mapping affirmative and negated captions to nearly identical representations. To expose this blind spot, we introduce NegEval-Audio, a framework that converts existing datasets into two negation-aware tasks, Retrieval-Neg and Multiple-Choice Negation (MCQ-Neg), to probe whether models distinguish present from absent events. On AudioCaps and Clotho, performance degrades sharply under negation, with negation-type MCQ accuracy falling far below chance, and the failure persists even for a recent multimodal LLM-based embedding model. While a training-free steering method improves MCQ-Neg, it yields marginal gains for Retrieval-Neg. This indicates that affirmation bias is a fundamental flaw in the representation geometry, necessitating explicit negation-aware training objectives. 

---
# Code-MUE: Measuring Code LLMs' Uncertainty through Execution-based Semantic Interaction Graphs 

**Authors**: Xiaoning Ren, Yinxing Xue, Lei Ma, Yuheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12273)  

**Abstract**: As Code Large Language Models (LLMs) become central to modern software engineering, their inherent stochasticity poses significant real-world risks, where even minor errors can lead to severe functional, security, or safety consequences. Reliable automation, therefore, demands the ability to distinguish between confident, well-supported predictions and stochastic guessing. However, existing uncertainty estimation methods face a critical gap: white and grey-box techniques are often inapplicable to closed-source models, while standard "black-box" text metrics fail to capture the unique fragility of code, where syntactic variation does not always imply semantic divergence. To bridge this syntax-semantics gap, we introduce Code-MUE, a purely black-box framework that measures uncertainty through execution-based Semantic Interaction Graphs. Unlike prior approaches that rely on superficial textual similarity, Code-MUE grounds uncertainty in observable runtime behavior, calculating the Von Neumann entropy of the solution space to quantify global semantic diversity. A large-scale empirical study across eight state-of-the-art LLMs demonstrates that Code-MUE achieves a strong negative correlation with functional correctness (Spearman's correlation up to -0.98), significantly outperforming lexical and embedding-based baselines while enabling robust risk detection and selective prediction in practical workflows. 

---
# On-Device Deep Research at 4B: Exposure Bounds Faithfulness, Retrieval Bounds Coverage 

**Authors**: Vinay Kumar Chaganti  

**Link**: [PDF](https://arxiv.org/pdf/2607.12257)  

**Abstract**: On-device research agents search a corpus, read sources, and write a cited brief on a personal laptop. Whether their citations are faithful, and at what cost, is unmeasured for a deployable small model. This study fixes one 4B generator on a 24 GB laptop and asks what makes its citations faithful. It separates two quantities usually reported as one number. Cited claim faithfulness asks whether the cited source supports the claim. Trustworthy coverage asks whether the agent also cites the right sources. The study crosses how much of each source the generator sees, 400 against 1500 characters, with the quality of the sources supplied, gold papers against retrieved papers. Two levers fall out, and they act on different outcomes. Exposure sets faithfulness. More of each source lifts faithfulness from 0.45 to 0.58 on retrieved sources and from 0.37 to 0.58 on gold sources, and the two settings converge, so faithfulness is bound by exposure, not by whether the source is correct. The exposure lift is robust to a second, independent judge; the exact convergence is tight under the primary judge and only approximate under the second. Retrieval sets coverage. Trustworthy coverage stays near 0.22 on retrieved sources at any exposure, because recall is held near 0.40, so exposure cannot fix which sources are cited. The extra exposure costs about 235 output tokens. The practical recipe is to raise per source exposure first, cheaply, and then treat retrieval recall as the only remaining lever. 

---
# Speculate with Memory: Lossless Acceleration for LLM Agents 

**Authors**: Yu Li, Qinyuan Ye, Prafulla Kumar Choubey, Jiaxin Zhang, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.12236)  

**Abstract**: Speculative execution accelerates LLM agents by using a smaller, cheaper model to predict and pre-launch the next step while the environment is idle. However, existing speculators are stateless and discard all information between tasks, preventing prediction quality from improving with experience. We equip the speculator with three online memory systems that learn from past agent trajectories: a contrastive transition table tracking action-sequence statistics, an episodic memory retrieving contextually similar segments, and a confusion tracker suppressing recurring errors. We evaluate this approach on six benchmarks spanning three speculation types: action prediction, observation prediction, and chained prediction. Memory-augmented speculation yields a 19--39\% relative accuracy improvement on action prediction and up to a $2.5\times$ increase on observation prediction tasks with repetitive action spaces. These gains grow continuously as memory accumulates and generalize across speculator models of varying cost. All speculation is lossless because it runs during idle time at zero added wall-clock cost, and the actor's trajectory is identical to non-speculative execution. 

---
# An Empirical Analysis of Continual Learning for Heterogeneous Medical Visual Question Answering 

**Authors**: Mai A. Shaaban, Tausifa Jan Saleem, Alaa Mohamed, Dilnaz Utemissova, Ufaq Khan, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2607.12048)  

**Abstract**: Deploying medical visual question answering (MedVQA) systems in real-world clinical settings requires models that adapt to new clinical tasks without forgetting previously acquired knowledge. Continual learning (CL) provides a practical framework for this setting. Despite rapid progress in medical vision-language models, the behavior of CL methods when training these models across heterogeneous MedVQA tasks remains underexplored. This work presents a systematic evaluation of CL for MedVQA across diverse clinical objectives, including classification, multi-label classification, detection, cell counting, and report generation. Specifically, we explore (1) the ability of existing CL methods to mitigate catastrophic forgetting; (2) their sensitivity to task ordering, analyzing how different task sequences influence performance retention and forgetting; and (3) the evolution of low-rank adaptation parameters as new tasks are learned, revealing patterns of weight drift under different CL methods. Our findings suggest that existing CL methods struggle to maintain stability-plasticity balance when tasks with different objectives and supervision formats are interleaved. Code and full experimental setup will be publicly available. 

---
# Sparse Inter-Layer Dependencies of Transformer FFN Neurons 

**Authors**: Johannes Knittel, Hanspeter Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2607.11990)  

**Abstract**: Feedforward network (FFN) blocks account for a large fraction of the parameters and computation in Transformer architectures, yet their internal structure remains difficult to interpret due to the additive superposition induced by the residual stream. We examine whether the activation of an FFN neuron can be explained by a sparse set of preceding neuron activations and attention outputs. We introduce a training-free attribution method that estimates the relative influence of upstream neurons and attention outputs on a target neuron's activation. Empirically, across models and layers, we find that small subsets of preceding activations and attention outputs suffice to preserve neuron activations with high fidelity when all remaining inputs are masked with their average values. Effective sparsity is even greater when accounting for the inherent activation sparsity of upstream layers. Moreover, applying the neuron-specific masks in all layers simultaneously, such that the induced deviations propagate through the network, leaves model perplexity largely unchanged at moderate sparsity levels. These results demonstrate that, despite dense parameterization, FFNs exhibit sparse and structured inter-layer dependencies at the neuron level. Our method provides a practical, scalable tool for circuit-level interpretability and identifies candidate sparse pathways with potential implications for efficient inference. 

---
# Optimization Is Not All You Need 

**Authors**: Minh Hua, Rita Raley  

**Link**: [PDF](https://arxiv.org/pdf/2607.11977)  

**Abstract**: In 2019, OpenAI released two million GPT-2 outputs-ungrammatical, half broken-to aid the detection of machine-generated text. The alignment that produced their more fluent successors is usually regarded as an engineering achievement; we read it instead as the newest expression of optimization culture: the conviction, older than the technology, that measurable improvement along predefined axes exhausts the question of value. Tracing that conviction through the stack-pretraining, decoding, preference tuning, benchmarking, interface-and back through its genealogy in the audit society, we arrive at the limit: an optimization procedure can measure how improbable a piece of generated text is; it cannot tell whether that unlikelihood is error or invention. A procedure that cannot make that distinction has nonetheless, within half a decade, assumed the authority to set the protocols of legitimate language. Held for centuries by academies and schoolrooms, grammars and examiners, this authority has been given over to loss functions, reward models, benchmarks, and system prompts: an apparatus that executes the office of judgment with no capacity for judging. 

---
# GRID: Grammar-Railed Decoding for Enterprise SQL Generation 

**Authors**: Mohsen Arjmandi  

**Link**: [PDF](https://arxiv.org/pdf/2607.11951)  

**Abstract**: Large language models can write SQL, but enterprise deployment demands more than plausible text: outputs must be syntactically valid, must respect per-role and per-schema policy, must carry provable (not best-effort) guarantees, must not slow down as generations grow, and must leave a compliance-grade record of every decision. We present GRID (Grammar-Railed Decoding), a grammar-constrained decoding engine that keys exact next-token masks on parser configurations (lexer scan state x LALR(1) stack) rather than on token sequences, and uses the incrementally advanced LALR(1) parser itself as a viable-prefix oracle. LLM tokens are bridged to grammar terminals by a byte-level trie walk with a context-independent/context-dependent split that makes cache-key soundness hold by construction. Role-based access control is compiled into the language: role projections subset the grammar's productions and schema lexicons restrict identifier terminals, so forbidden verbs and identifiers are unreachable at mask level. Four guarantees (soundness, completeness, termination, and near-constant per-token cost) are stated with explicit preconditions and each paired with a test or benchmark. Rust kernels bring the per-token mask to a 3.6-6.7 us median, ahead of llguidance at p50 and p90 on two tokenizers with zero false rejects; per-token guard cost is position-flat at n=16,000. On Spider, constrained decoding is worth +13 execution-accuracy points at 0.5B, and one checker-guided repair pass over the provably mask-unenforceable residue (column-level policy) lifts a 7B model to 94.5% executable. A hash-chained per-token audit trail replays bit-identically with 100% tamper detection. We state plainly what the mask cannot do (distribution faithfulness, column-level RBAC, non-LALR(1) languages) and where measured cost remains. 

---
# Ontology-Amplified Distillation and Contextuality Auditing for Sovereign Enterprise Language Models: A Combined Proof-of-Mechanism and Negative-Results Method Study 

**Authors**: Thanh Luong Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.11948)  

**Abstract**: Regulated financial institutions operating under data-residency rules need tenant-owned language models that can run inside the institution's perimeter. This paper combines two related FAOS studies into one mechanism-and-control article. First, it reports a reduced-power proof-of-mechanism study of ontology-amplified distillation: a Qwen3.6-27B student is adapted to the Foundation AgenticOS ontology through supervised fine-tuning on frontier-teacher trajectories and ontology-grounded direct preference optimization (DPO), trained locally on a single Apple M5 Max from 47 synthetic, English-language, cross-domain preference pairs. On 40 held-out Vietnamese financial-domain tasks, the distilled student grounds 36 of 40 tasks (grounded rate 0.90; mean ontology term-coverage r_onto = 0.95 on a metric floored at 0.50), equal to the GPT-5 frontier baseline, which also grounds 36 of 40. The outcome is underpowered to establish equivalence: the paired-difference 95% confidence interval spans +/-4 tasks, and the run does not test or show the pre-registered amplification prediction that the student should exceed the frontier. Second, the paper consolidates a contextuality-audit method for enterprise-agent routing. In a separate negative-results pilot, the corrected canonical Contextuality-by-Default degree is zero for all Phase 1.3 groups in both the local-Qwen run and an explicitly labeled Gemma replication check; the useful signal is direct influence and construct coupling, not surviving residual contextuality. Together, the studies pair an ontology-grounded model-building mechanism with a governance diagnostic for deciding when apparent disagreement should trigger prompt standardization, multi-agent synthesis, or human review. The evidence supports neither deployability, safety, superiority, statistical equivalence, nor a contextuality-positive routing rule. 

---
# QDEvo: A Multi-Objective Quality-Diversity Framework for Automated Heuristic Design 

**Authors**: Nam Do Khanh, Nhat Nguyen Tran Minh, Dat Pham Vu Tuan, Long Doan, Binh Huynh Thi Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2607.11916)  

**Abstract**: The integration of Large Language Models (LLMs) with evolutionary computation has emerged as a powerful paradigm for automated heuristic design in combinatorial optimization. However, existing approaches suffer from mode collapse, converging to homogeneous populations that lack semantic diversity and fail to explore the full algorithmic space. We propose Quality-Diversity Evolution (QDEvo), a multi-objective framework that integrates Quality-Diversity optimization with LLM-driven heuristic search, maintaining an unbounded archive of semantically diverse algorithms using pre-trained code embeddings and incorporating hierarchical self-reflection to guide the evolutionary process. Extensive experiments across standard benchmarks and real-world industrial applications demonstrate that QDEvo significantly outperforms state-of-the-art methods in both Hypervolume and Inverted Generational Distance metrics. Our framework enables the discovery of heuristics that are simultaneously high-performing, computationally efficient, and semantically diverse, providing practitioners with a rich portfolio of solutions for complex optimization problems. 

---
# FAIR GraphRAG: A Retrieval-Augmented Generation Approach for Semantic Data Analysis 

**Authors**: Marlena Flüh, Soo-Yon Kim, Carolin Victoria Schneider, Sandra Geisler  

**Link**: [PDF](https://arxiv.org/pdf/2607.11464)  

**Abstract**: Retrieval-Augmented Generation (RAG) addresses the limitations of Large Language Models (LLMs) when providing responses to domain-specific questions. Graph-based RAG approaches, such as GraphRAG, enhance retrieval by capturing semantic relationships within knowledge graphs (KGs). While the FAIR principles (Findability, Accessibility, Interoperability, and Reusability) are becoming prevalent for scientific data management, especially in complex domains such as medicine, existing RAG approaches lack a structured FAIRification of the underlying knowledge resources. This lack limits their potential for FAIR information retrieval in these domains. To address this gap, we introduce FAIR GraphRAG, a novel framework that integrates FAIR Digital Objects (FDOs) as the fundamental units of a graph-based retrieval system. Each graph node represents an FDO that incorporates core data, metadata, persistent identifiers, and semantic links. We leverage LLMs to support schema construction and automated extraction of content and metadata from data sources. The framework was co-designed by physicians and computer scientists to ensure technical and clinical relevance. We apply FAIR GraphRAG to a biomedical dataset in gastroenterology, demonstrating its applicability to RNA-sequencing data. Beyond ensuring adherence to the FAIR principles, FAIR GraphRAG significantly improves question answering accuracy, coverage, and explainability, particularly for complex queries involving metadata and ontology links. This work shows the feasibility of combining FAIR data practices with graph-based retrieval techniques. We see potential for applying our approach to other specialized fields such as education and business. 

---
# From Words to Widgets for Controllable LLM Generation 

**Authors**: Chao Zhang, Yiren Liu, Lunyiu Nie, Jeffrey M. Rzeszotarski, Yun Huang, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2604.10925)  

**Abstract**: Natural language remains the predominant way people interact with large language models (LLMs). However, users often struggle to precisely express and control subjective preferences (e.g., tone, style, and emphasis) through prompting. We propose Malleable Prompting, a new interactive prompting technique for controllable LLM generation. It reifies preference expressions in natural language prompts into GUI widgets (e.g., sliders, dropdowns, and toggles) that users can directly configure to steer generation, while visualizing each control's influence on the output to support attribution and comparison across iterations. To enable this interaction, we introduce an LLM decoding algorithm that modulates the token probability distribution during generation based on preference expressions and their widget values. Through a user study, we show that Malleable Prompting helps participants achieve target preferences more precisely and is perceived as more controllable and transparent than natural language prompting alone. 

---
