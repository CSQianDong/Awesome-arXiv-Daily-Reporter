# On Data Engineering for Scaling LLM Terminal Capabilities 

**Authors**: Renjie Pi, Grace Lam, Mohammad Shoeybi, Pooya Jannaty, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2602.21193)  

**Abstract**: Despite rapid recent progress in the terminal capabilities of large language models, the training data strategies behind state-of-the-art terminal agents remain largely undisclosed. We address this gap through a systematic study of data engineering practices for terminal agents, making two key contributions: (1) Terminal-Task-Gen, a lightweight synthetic task generation pipeline that supports seed-based and skill-based task construction, and (2) a comprehensive analysis of data and training strategies, including filtering, curriculum learning, long context training, and scaling behavior. Our pipeline yields Terminal-Corpus, a large-scale open-source dataset for terminal tasks. Using this dataset, we train Nemotron-Terminal, a family of models initialized from Qwen3(8B, 14B, 32B) that achieve substantial gains on Terminal-Bench 2.0: Nemotron-Terminal-8B improves from 2.5% to 13.0% Nemotron-Terminal-14B improves from 4.0% to 20.2%, and Nemotron-Terminal-32B improves from 3.4% to 27.4%, matching the performance of significantly larger models. To accelerate research in this domain, we open-source our model checkpoints and most of our synthetic datasets at this https URL. 

---
# PVminer: A Domain-Specific Tool to Detect the Patient Voice in Patient Generated Data 

**Authors**: Samah Fodeh, Linhai Ma, Yan Wang, Srivani Talakokkul, Ganesh Puthiaraju, Afshan Khan, Ashley Hagaman, Sarah Lowe, Aimee Roundtree  

**Link**: [PDF](https://arxiv.org/pdf/2602.21165)  

**Abstract**: Patient-generated text such as secure messages, surveys, and interviews contains rich expressions of the patient voice (PV), reflecting communicative behaviors and social determinants of health (SDoH). Traditional qualitative coding frameworks are labor intensive and do not scale to large volumes of patient-authored messages across health systems. Existing machine learning (ML) and natural language processing (NLP) approaches provide partial solutions but often treat patient-centered communication (PCC) and SDoH as separate tasks or rely on models not well suited to patient-facing language. We introduce PVminer, a domain-adapted NLP framework for structuring patient voice in secure patient-provider communication. PVminer formulates PV detection as a multi-label, multi-class prediction task integrating patient-specific BERT encoders (PV-BERT-base and PV-BERT-large), unsupervised topic modeling for thematic augmentation (PV-Topic-BERT), and fine-tuned classifiers for Code, Subcode, and Combo-level labels. Topic representations are incorporated during fine-tuning and inference to enrich semantic inputs. PVminer achieves strong performance across hierarchical tasks and outperforms biomedical and clinical pre-trained baselines, achieving F1 scores of 82.25% (Code), 80.14% (Subcode), and up to 77.87% (Combo). An ablation study further shows that author identity and topic-based augmentation each contribute meaningful gains. Pre-trained models, source code, and documentation will be publicly released, with annotated datasets available upon request for research use. 

---
# Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning 

**Authors**: Sanket Badhe, Deep Shah  

**Link**: [PDF](https://arxiv.org/pdf/2602.21103)  

**Abstract**: Advanced reasoning typically requires Chain-of-Thought prompting, which is accurate but incurs prohibitive latency and substantial test-time inference costs. The standard alternative, fine-tuning smaller models, often sacrifices interpretability while introducing significant resource and operational overhead. To address these limitations, we introduce Prompt-Level Distillation (PLD). We extract explicit reasoning patterns from a Teacher model and organize them into a structured list of expressive instructions for the Student model's System Prompt. Evaluated on the StereoSet and Contract-NLI datasets using Gemma-3 4B, PLD improved Macro F1 scores from 57\% to 90.0\% and 67\% to 83\% respectively, enabling this compact model to match frontier performance with negligible latency overhead. These expressive instructions render the decision-making process transparent, allowing for full human verification of logic, making this approach ideal for regulated industries such as law, finance, and content moderation, as well as high-volume use cases and edge devices. 

---
# Beyond the Star Rating: A Scalable Framework for Aspect-Based Sentiment Analysis Using LLMs and Text Classification 

**Authors**: Vishal Patil, Shree Vaishnavi Bacha, Revanth Yamani, Yidan Sun, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2602.21082)  

**Abstract**: Customer-provided reviews have become an important source of information for business owners and other customers alike. However, effectively analyzing millions of unstructured reviews remains challenging. While large language models (LLMs) show promise for natural language understanding, their application to large-scale review analysis has been limited by computational costs and scalability concerns. This study proposes a hybrid approach that uses LLMs for aspect identification while employing classic machine-learning methods for sentiment classification at scale. Using ChatGPT to analyze sampled restaurant reviews, we identified key aspects of dining experiences and developed sentiment classifiers using human-labeled reviews, which we subsequently applied to 4.7 million reviews collected over 17 years from a major online platform. Regression analysis reveals that our machine-labeled aspects significantly explain variance in overall restaurant ratings across different aspects of dining experiences, cuisines, and geographical regions. Our findings demonstrate that combining LLMs with traditional machine learning approaches can effectively automate aspect-based sentiment analysis of large-scale customer feedback, suggesting a practical framework for both researchers and practitioners in the hospitality industry and potentially, other service sectors. 

---
# Evaluating Proactive Risk Awareness of Large Language Models 

**Authors**: Xuan Luo, Yubin Chen, Zhiyu Hou, Linpu Yu, Geng Tu, Jing Li, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20976)  

**Abstract**: As large language models (LLMs) are increasingly embedded in everyday decision-making, their safety responsibilities extend beyond reacting to explicit harmful intent toward anticipating unintended but consequential risks. In this work, we introduce a proactive risk awareness evaluation framework that measures whether LLMs can anticipate potential harms and provide warnings before damage occurs. We construct the Butterfly dataset to instantiate this framework in the environmental and ecological domain. It contains 1,094 queries that simulate ordinary solution-seeking activities whose responses may induce latent ecological impact. Through experiments across five widely used LLMs, we analyze the effects of response length, languages, and modality. Experimental results reveal consistent, significant declines in proactive awareness under length-restricted responses, cross-lingual similarities, and persistent blind spots in (multimodal) species protection. These findings highlight a critical gap between current safety alignment and the requirements of real-world ecological responsibility, underscoring the need for proactive safeguards in LLM deployment. 

---
# Linear Reasoning vs. Proof by Cases: Obstacles for Large Language Models in FOL Problem Solving 

**Authors**: Yuliang Ji, Fuchen Shen, Jian Wu, Qiujie Xie, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20973)  

**Abstract**: To comprehensively evaluate the mathematical reasoning capabilities of Large Language Models (LLMs), researchers have introduced abundant mathematical reasoning datasets. However, most existing datasets primarily focus on linear reasoning, neglecting other parts such as proof by contradiction and proof by cases, which are crucial for investigating LLMs' reasoning abilities. To address this limitation, we first introduce a novel first-order logic (FOL) dataset named PC-FOL, annotated by professional mathematicians, focusing on case-based reasoning problems. All instances in this dataset are equipped with a manually written natural language proof, clearly distinguishing it from conventional linear reasoning datasets. Our experimental results over leading LLMs demonstrate a substantial performance gap between linear reasoning and case-based reasoning problems. To further investigate this phenomenon, we provide a theoretical analysis grounded in graphical model, which provides an explanation for the observed disparity between the two types of reasoning problems. We hope this work can reveal the core challenges in the field of automated natural language mathematical proof generation, paving the way for future research. 

---
# Blackbird Language Matrices: A Framework to Investigate the Linguistic Competence of Language Models 

**Authors**: Paola Merlo, Chunyang Jiang, Giuseppe Samo, Vivi Nastase  

**Link**: [PDF](https://arxiv.org/pdf/2602.20966)  

**Abstract**: This article describes a novel language task, the Blackbird Language Matrices (BLM) task, inspired by intelligence tests, and illustrates the BLM datasets, their construction and benchmarking, and targeted experiments on chunking and systematicity. BLMs are multiple-choice problems, structured at multiple levels: within each sentence, across the input sequence, within each candidate answer. Because of their rich structure, these curated, but naturalistic datasets are key to answer some core questions about current large language models abilities: do LLMs detect linguistic objects and their properties? Do they detect and use systematic patterns across sentences? Are they more prone to linguistic or reasoning errors, and how do these interact?
We show that BLMs, while challenging, can be solved at good levels of performance, in more than one language, with simple baseline models or, at better performance levels, with more tailored models. We show that their representations contain the grammatical objects and attributes relevant to solve a linguistic task. We also show that these solutions are reached by detecting systematic patterns across sentences.
The paper supports the point of view that curated, structured datasets support multi-faceted investigations of properties of language and large language models. Because they present a curated, articulated structure, because they comprise both learning contexts and expected answers, and because they are partly built by hand, BLMs fall in the category of datasets that can support explainability investigations, and be useful to ask why large language models behave the way they do. 

---
# The Art of Efficient Reasoning: Data, Reward, and Optimization 

**Authors**: Taiqiang Wu, Zenan Zu, Bo Zhou, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2602.20945)  

**Abstract**: Large Language Models (LLMs) consistently benefit from scaled Chain-of-Thought (CoT) reasoning, but also suffer from heavy computational overhead. To address this issue, efficient reasoning aims to incentivize short yet accurate thinking trajectories, typically through reward shaping with Reinforcement Learning (RL). In this paper, we systematically investigate the mechanics of efficient reasoning for LLMs. For comprehensive evaluation, we advocate for more fine-grained metrics, including length distribution conditioned on correctness and performance across a wide spectrum of token budgets ranging from 2k to 32k. First, we reveal that the training process follows a two-stage paradigm: length adaptation and reasoning refinement. After that, we conduct extensive experiments (about 0.2 million GPU hours) in a unified protocol, deconstructing training prompts and rollouts, reward shaping, and optimization strategies. In particular, a key finding is to train on relatively easier prompts, ensuring the density of positive reward signals and thus avoiding the length collapse. Meanwhile, the learned length bias can be generalized across domains. We distill all findings into valuable insights and practical guidelines, and further validate them across the Qwen3 series, ranging from 0.6B to 30B, demonstrating the robustness and generalization. 

---
# Exa-PSD: a new Persian sentiment analysis dataset on Twitter 

**Authors**: Seyed Himan Ghaderi, Saeed Sarbazi Azad, Mohammad Mehdi Jaziriyan, Ahmad Akbari  

**Link**: [PDF](https://arxiv.org/pdf/2602.20892)  

**Abstract**: Today, Social networks such as Twitter are the most widely used platforms for communication of people. Analyzing this data has useful information to recognize the opinion of people in tweets. Sentiment analysis plays a vital role in NLP, which identifies the opinion of the individuals about a specific topic. Natural language processing in Persian has many challenges despite the adventure of strong language models. The datasets available in Persian are generally in special topics such as products, foods, hotels, etc while users may use ironies, colloquial phrases in social media To overcome these challenges, there is a necessity for having a dataset of Persian sentiment analysis on Twitter. In this paper, we introduce the Exa sentiment analysis Persian dataset, which is collected from Persian tweets. This dataset contains 12,000 tweets, annotated by 5 native Persian taggers. The aforementioned data is labeled in 3 classes: positive, neutral and negative. We present the characteristics and statistics of this dataset and use the pre-trained Pars Bert and Roberta as the base model to evaluate this dataset. Our evaluation reached a 79.87 Macro F-score, which shows the model and data can be adequately valuable for a sentiment analysis system. 

---
# FinAnchor: Aligned Multi-Model Representations for Financial Prediction 

**Authors**: Zirui He, Huopu Zhang, Yanguang Liu, Sirui Wu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2602.20859)  

**Abstract**: Financial prediction from long documents involves significant challenges, as actionable signals are often sparse and obscured by noise, and the optimal LLM for generating embeddings varies across tasks and time periods. In this paper, we propose FinAnchor(Financial Anchored Representations), a lightweight framework that integrates embeddings from multiple LLMs without fine-tuning the underlying models. FinAnchor addresses the incompatibility of feature spaces by selecting an anchor embedding space and learning linear mappings to align representations from other models into this anchor. These aligned features are then aggregated to form a unified representation for downstream prediction. Across multiple financial NLP tasks, FinAnchor consistently outperforms strong single-model baselines and standard ensemble methods, demonstrating the effectiveness of anchoring heterogeneous representations for robust financial prediction. 

---
# Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation 

**Authors**: Sayantan Dasgupta, Trevor Cohn, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20816)  

**Abstract**: The core learning signal used in language model distillation is the standard Kullback-Leibler (KL) divergence between the student and teacher distributions. Traditional KL divergence tends to be dominated by the next tokens with the highest probabilities, i.e., the teacher's modes, thereby diminishing the influence of less probable yet potentially informative components of the output distribution. We propose a new tail-aware divergence that decouples the contribution of the teacher model's top-K predicted probabilities from that of lower-probability predictions, while maintaining the same computational profile as the KL Divergence. Our decoupled approach reduces the impact of the teacher modes and, consequently, increases the contribution of the tail of the distribution. Experimental results demonstrate that our modified distillation method yields competitive performance in both pre-training and supervised distillation of decoder models across various datasets. Furthermore, the distillation process is efficient and can be performed with a modest academic budget for large datasets, eliminating the need for industry-scale computing. 

---
# Overton Pluralistic Reinforcement Learning for Large Language Models 

**Authors**: Yu Fu, Seongho Son, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2602.20759)  

**Abstract**: Existing alignment paradigms remain limited in capturing the pluralistic nature of human values. Overton Pluralism addresses this gap by generating responses with diverse perspectives from a single query. This paper introduces OP-GRPO (Overton Pluralistic Group Relative Policy Optimization), a reinforcement learning framework for implicit Overton Pluralism that enables a single large language model to produce pluralistic responses without explicit prompting or modular orchestration. Our workflow consists of two main steps. First, similarity estimator training fine-tunes a Sentence Transformer for Overton Pluralism tasks to provide more accurate coverage evaluation of generated responses. Second, OP-GRPO training incorporates this similarity estimator into a dual-reward system designed to ensure both broad coverage of genuine human perspectives and the uniqueness of each perspective, thereby promoting diversity. Empirical results demonstrate a "small models, big perspective coverage" effect. The trained Qwen2.5-3B-Instruct model surpasses a 20B GPT-OSS baseline with a 37.4 percent relative accuracy gain on a Natural Language Inference benchmark, and also outperforms a modular architecture baseline with a 19.1 percent relative improvement. Additional evaluations using GPT-4.1 as a large language model judge further confirm the robustness of the approach. 

---
# SibylSense: Adaptive Rubric Learning via Memory Tuning and Adversarial Probing 

**Authors**: Yifei Xu, Guilherme Potje, Shivam Shandilya, Tiancheng Yuan, Leonardo de Oliveira Nunes, Rakshanda Agarwal, Saeid Asgari, Adam Atkinson, Emre Kıcıman, Songwu Lu, Ranveer Chandra, Tusher Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2602.20751)  

**Abstract**: Designing aligned and robust rewards for open-ended generation remains a key barrier to RL post-training. Rubrics provide structured, interpretable supervision, but scaling rubric construction is difficult: expert rubrics are costly, prompted rubrics are often superficial or inconsistent, and fixed-pool discriminative rubrics can saturate and drift, enabling reward hacking. We present SibylSense, an inference-time learning approach that adapts a frozen rubric generator through a tunable memory bank of validated rubric items. Memory is updated via verifier-based item rewards measured by reference-candidate answer discriminative gaps from a handful of examples. SibylSense alternates memory tuning with a rubric-adversarial policy update that produces rubric-satisfying candidate answers, shrinking discriminative gaps and driving the rubric generator to capture new quality dimensions. Experiments on two open-ended tasks show that SibylSense yields more discriminative rubrics and improves downstream RL performance over static and non-adaptive baselines. 

---
# Explicit Grammar Semantic Feature Fusion for Robust Text Classification 

**Authors**: Azrin Sultana, Firoz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2602.20749)  

**Abstract**: Natural Language Processing enables computers to understand human language by analysing and classifying text efficiently with deep-level grammatical and semantic features. Existing models capture features by learning from large corpora with transformer models, which are computationally intensive and unsuitable for resource-constrained environments. Therefore, our proposed study incorporates comprehensive grammatical rules alongside semantic information to build a robust, lightweight classification model without resorting to full parameterised transformer models or heavy deep learning architectures. The novelty of our approach lies in its explicit encoding of sentence-level grammatical structure, including syntactic composition, phrase patterns, and complexity indicators, into a compact grammar vector, which is then fused with frozen contextual embeddings. These heterogeneous elements unified a single representation that captures both the structural and semantic characteristics of the text. Deep learning models such as Deep Belief Networks (DBNs), Long Short-Term Memory (LSTMs), BiLSTMs, and transformer-based BERT and XLNET were used to train and evaluate the model, with the number of epochs varied. Based on experimental results, the unified feature representation model captures both the semantic and structural properties of text, outperforming baseline models by 2%-15%, enabling more effective learning across heterogeneous domains. Unlike prior syntax-aware transformer models that inject grammatical structure through additional attention layers, tree encoders, or full fine-tuning, the proposed framework treats grammar as an explicit inductive bias rather than a learnable module, resulting in a very lightweight model that delivers better performance on edge devices 

---
# Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization 

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20743)  

**Abstract**: Anonymizing textual documents is a highly context-sensitive problem: the appropriate balance between privacy protection and utility preservation varies with the data domain, privacy objectives, and downstream application. However, existing anonymization methods rely on static, manually designed strategies that lack the flexibility to adjust to diverse requirements and often fail to generalize across domains. We introduce adaptive text anonymization, a new task formulation in which anonymization strategies are automatically adapted to specific privacy-utility requirements. We propose a framework for task-specific prompt optimization that automatically constructs anonymization instructions for language models, enabling adaptation to different privacy goals, domains, and downstream usage patterns. To evaluate our approach, we present a benchmark spanning five datasets with diverse domains, privacy constraints, and utility objectives. Across all evaluated settings, our framework consistently achieves a better privacy-utility trade-off than existing baselines, while remaining computationally efficient and effective on open-source language models, with performance comparable to larger closed-source models. Additionally, we show that our method can discover novel anonymization strategies that explore different points along the privacy-utility trade-off frontier. 

---
# ID-LoRA: Efficient Low-Rank Adaptation Inspired by Matrix Interpolative Decomposition 

**Authors**: Xindian Ma, Rundong Kong, Peng Zhang, Ruoxiang Huang, Yongyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20727)  

**Abstract**: LoRA has become a universal Parameter-Efficient Fine-Tuning (PEFT) technique that equips Large Language Models (LLMs) to adapt quickly to new tasks. However, when these models are scaled up, even the latest LoRA variants still introduce considerable overhead in trainable parameters. Conversely, aggressively lowering the rank to curb this overhead markedly degrades performance in complex multi-task settings. We propose ID-LoRA, a novel PEFT framework that breaks the trade-off. Its core innovation lies in extracting and reusing clustered parameter groups from the pretrained weight matrix. These groups are then used to form multiple low-rank components, all of which share only a single initialized trainable low-rank matrix. This approach cuts the number of trainable parameters while keeping the model's capacity intact. We evaluate ID-LoRA on five diverse benchmarks: Mathematical Reasoning, Code Generation, MMLU, CommonsenseQA, and Safety Alignment. ID-LoRA outperforms both full fine-tuning and existing PEFT baselines (e.g., LoRA, DoRA, HydraLoRA) while using up to 46% fewer trainable parameters than the standard LoRA. In multi-task scenarios, it surpasses LoRA and its recent variants (e.g., DoRA and HydraLoRA) on both Code and MMLU tasks, yet requires only 54% of the trainable parameters demanded by the conventional LoRA. 

---
# CAMEL: Confidence-Gated Reflection for Reward Modeling 

**Authors**: Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Kun Xu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2602.20670)  

**Abstract**: Reward models play a fundamental role in aligning large language models with human preferences. Existing methods predominantly follow two paradigms: scalar discriminative preference models, which are efficient but lack interpretability, and generative judging models, which offer richer reasoning at the cost of higher computational overhead. We observe that the log-probability margin between verdict tokens strongly correlates with prediction correctness, providing a reliable proxy for instance difficulty without additional inference cost. Building on this insight, we propose CAMEL, a confidence-gated reflection framework that performs a lightweight single-token preference decision first and selectively invokes reflection only for low-confidence instances. To induce effective self-correction, we train the model via reinforcement learning with counterfactual prefix augmentation, which exposes the model to diverse initial verdicts and encourages genuine revision. Empirically, CAMEL achieves state-of-the-art performance on three widely used reward-model benchmarks with 82.9% average accuracy, surpassing the best prior model by 3.2% and outperforming 70B-parameter models using only 14B parameters, while establishing a strictly better accuracy-efficiency Pareto frontier. 

---
# CARE: An Explainable Computational Framework for Assessing Client-Perceived Therapeutic Alliance Using Large Language Models 

**Authors**: Anqi Li, Chenxiao Wang, Yu Lu, Renjun Xu, Lizhi Ma, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20648)  

**Abstract**: Client perceptions of the therapeutic alliance are critical for counseling effectiveness. Accurately capturing these perceptions remains challenging, as traditional post-session questionnaires are burdensome and often delayed, while existing computational approaches produce coarse scores, lack interpretable rationales, and fail to model holistic session context. We present CARE, an LLM-based framework to automatically predict multi-dimensional alliance scores and generate interpretable rationales from counseling transcripts. Built on the CounselingWAI dataset and enriched with 9,516 expert-curated rationales, CARE is fine-tuned using rationale-augmented supervision with the LLaMA-3.1-8B-Instruct backbone. Experiments show that CARE outperforms leading LLMs and substantially reduces the gap between counselor evaluations and client-perceived alliance, achieving over 70% higher Pearson correlation with client ratings. Rationale-augmented supervision further improves predictive accuracy. CARE also produces high-quality, contextually grounded rationales, validated by both automatic and human evaluations. Applied to real-world Chinese online counseling sessions, CARE uncovers common alliance-building challenges, illustrates how interaction patterns shape alliance development, and provides actionable insights, demonstrating its potential as an AI-assisted tool for supporting mental health care. 

---
# Semantic Novelty at Scale: Narrative Shape Taxonomy and Readership Prediction in 28,606 Books 

**Authors**: W. Frederick Zimmerman  

**Link**: [PDF](https://arxiv.org/pdf/2602.20647)  

**Abstract**: I introduce semantic novelty--cosine distance between each paragraph's sentence embedding and the running centroid of all preceding paragraphs--as an information-theoretic measure of narrative structure at corpus scale. Applying it to 28,606 books in PG19 (pre-1920 English literature), I compute paragraph-level novelty curves using 768-dimensional SBERT embeddings, then reduce each to a 16-segment Piecewise Aggregate Approximation (PAA). Ward-linkage clustering on PAA vectors reveals eight canonical narrative shape archetypes, from Steep Descent (rapid convergence) to Steep Ascent (escalating unpredictability). Volume--variance of the novelty trajectory--is the strongest length-independent predictor of readership (partial rho = 0.32), followed by speed (rho = 0.19) and Terminal/Initial ratio (rho = 0.19). Circuitousness shows strong raw correlation (rho = 0.41) but is 93 percent correlated with length; after control, partial rho drops to 0.11--demonstrating that naive correlations in corpus studies can be dominated by length confounds. Genre strongly constrains narrative shape (chi squared = 2121.6, p < 10 to the power negative 242), with fiction maintaining plateau profiles while nonfiction front-loads information. Historical analysis shows books became progressively more predictable between 1840 and 1910 (T/I ratio trend r = negative 0.74, p = 0.037). SAX analysis reveals 85 percent signature uniqueness, suggesting each book traces a nearly unique path through semantic space. These findings demonstrate that information-density dynamics, distinct from sentiment or topic, constitute a fundamental dimension of narrative structure with measurable consequences for reader engagement. Dataset: this https URL 

---
# Enhancing Hate Speech Detection on Social Media: A Comparative Analysis of Machine Learning Models and Text Transformation Approaches 

**Authors**: Saurabh Mishra, Shivani Thakur, Radhika Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20634)  

**Abstract**: The proliferation of hate speech on social media platforms has necessitated the development of effective detection and moderation tools. This study evaluates the efficacy of various machine learning models in identifying hate speech and offensive language and investigates the potential of text transformation techniques to neutralize such content. We compare traditional models like CNNs and LSTMs with advanced neural network models such as BERT and its derivatives, alongside exploring hybrid models that combine different architectural features. Our results indicate that while advanced models like BERT show superior accuracy due to their deep contextual understanding, hybrid models exhibit improved capabilities in certain scenarios. Furthermore, we introduce innovative text transformation approaches that convert negative expressions into neutral ones, thereby potentially mitigating the impact of harmful content. The implications of these findings are discussed, highlighting the strengths and limitations of current technologies and proposing future directions for more robust hate speech detection systems. 

---
# Personal Information Parroting in Language Models 

**Authors**: Nishant Subramani, Kshitish Ghate, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2602.20580)  

**Abstract**: Modern language models (LM) are trained on large scrapes of the Web, containing millions of personal information (PI) instances, many of which LMs memorize, increasing privacy risks. In this work, we develop the regexes and rules (R&R) detector suite to detect email addresses, phone numbers, and IP addresses, which outperforms the best regex-based PI detectors. On a manually curated set of 483 instances of PI, we measure memorization: finding that 13.6% are parroted verbatim by the Pythia-6.9b model, i.e., when the model is prompted with the tokens that precede the PI in the original document, greedy decoding generates the entire PI span exactly. We expand this analysis to study models of varying sizes (160M-6.9B) and pretraining time steps (70k-143k iterations) in the Pythia model suite and find that both model size and amount of pretraining are positively correlated with memorization. Even the smallest model, Pythia-160m, parrots 2.7% of the instances exactly. Consequently, we strongly recommend that pretraining datasets be aggressively filtered and anonymized to minimize PI parroting. 

---
# Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning 

**Authors**: Justin Lovelace, Christian Belardi, Sofian Zalouk, Adhitya Polavaram, Srivatsa Kundurthy, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2602.20528)  

**Abstract**: The Stop-Think-AutoRegress Language Diffusion Model (STAR-LDM) integrates latent diffusion planning with autoregressive generation. Unlike conventional autoregressive language models limited to token-by-token decisions, STAR-LDM incorporates a "thinking" phase that pauses generation to refine a semantic plan through diffusion before continuing. This enables global planning in continuous space prior to committing to discrete tokens. Evaluations show STAR-LDM significantly outperforms similar-sized models on language understanding benchmarks and achieves $>70\%$ win rates in LLM-as-judge comparisons for narrative coherence and commonsense reasoning. The architecture also allows straightforward control through lightweight classifiers, enabling fine-grained steering of attributes without model retraining while maintaining better fluency-control trade-offs than specialized approaches. 

---
# From Performance to Purpose: A Sociotechnical Taxonomy for Evaluating Large Language Model Utility 

**Authors**: Gavin Levinson, Keith Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2602.20513)  

**Abstract**: As large language models (LLMs) continue to improve at completing discrete tasks, they are being integrated into increasingly complex and diverse real-world systems. However, task-level success alone does not establish a model's fit for use in practice. In applied, high-stakes settings, LLM effectiveness is driven by a wider array of sociotechnical determinants that extend beyond conventional performance measures. Although a growing set of metrics capture many of these considerations, they are rarely organized in a way that supports consistent evaluation, leaving no unified taxonomy for assessing and comparing LLM utility across use cases. To address this gap, we introduce the Language Model Utility Taxonomy (LUX), a comprehensive framework that structures utility evaluation across four domains: performance, interaction, operations, and governance. Within each domain, LUX is organized hierarchically into thematically aligned dimensions and components, each grounded in metrics that enable quantitative comparison and alignment of model selection with intended use. In addition, an external dynamic web tool is provided to support exploration of the framework by connecting each component to a repository of relevant metrics (factors) for applied evaluation. 

---
# Disentangling Geometry, Performance, and Training in Language Models 

**Authors**: Atharva Kulkarni, Jacob Mitchell Springer, Arjun Subramonian, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2602.20433)  

**Abstract**: Geometric properties of Transformer weights, particularly the unembedding matrix, have been widely useful in language model interpretability research. Yet, their utility for estimating downstream performance remains unclear. In this work, we systematically investigate the relationship between model performance and the unembedding matrix geometry, particularly its effective rank. Our experiments, involving a suite of 108 OLMo-style language models trained under controlled variation, reveal several key findings. While the best-performing models often exhibit a high effective rank, this trend is not universal across tasks and training setups. Contrary to prior work, we find that low effective rank does not cause late-stage performance degradation in small models, but instead co-occurs with it; we find adversarial cases where low-rank models do not exhibit saturation. Moreover, we show that effective rank is strongly influenced by pre-training hyperparameters, such as batch size and weight decay, which in-turn affect the model's performance. Lastly, extending our analysis to other geometric metrics and final-layer representation, we find that these metrics are largely aligned, but none can reliably predict downstream performance. Overall, our findings suggest that the model's geometry, as captured by existing metrics, primarily reflects training choices rather than performance. 

---
# Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems 

**Authors**: Mukul Chhabra, Luigi Medrano, Arush Verma  

**Link**: [PDF](https://arxiv.org/pdf/2602.20379)  

**Abstract**: Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows such as technical support and IT operations, where evaluation must reflect operational constraints, structured identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific failure modes such as case misidentification, workflow misalignment, and partial resolution across turns.
We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems. The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality, grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation, regression testing, and production monitoring.
Through a comparative study of two instruction-tuned models across short and long workflows, we show that generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical tradeoffs that are actionable for system improvement. 

---
# How communicatively optimal are exact numeral systems? Once more on lexicon size and morphosyntactic complexity 

**Authors**: Chundra Cathcart, Arne Rubehn, Katja Bocklage, Luca Ciucci, Kellen Parker van Dam, Alžběta Kučerová, Jekaterina Mažara, Carlo Y. Meloni, David Snee, Johann-Mattis List  

**Link**: [PDF](https://arxiv.org/pdf/2602.20372)  

**Abstract**: Recent research argues that exact recursive numeral systems optimize communicative efficiency by balancing a tradeoff between the size of the numeral lexicon and the average morphosyntactic complexity (roughly length in morphemes) of numeral terms. We argue that previous studies have not characterized the data in a fashion that accounts for the degree of complexity languages display. Using data from 52 genetically diverse languages and an annotation scheme distinguishing between predictable and unpredictable allomorphy (formal variation), we show that many of the world's languages are decisively less efficient than one would expect. We discuss the implications of our findings for the study of numeral systems and linguistic evolution more generally. 

---
# Natural Language Processing Models for Robust Document Categorization 

**Authors**: Radoslaw Roszczyk, Pawel Tecza, Maciej Stodolski, Krzysztof Siwek  

**Link**: [PDF](https://arxiv.org/pdf/2602.20336)  

**Abstract**: This article presents an evaluation of several machine learning methods applied to automated text classification, alongside the design of a demonstrative system for unbalanced document categorization and distribution. The study focuses on balancing classification accuracy with computational efficiency, a key consideration when integrating AI into real world automation pipelines. Three models of varying complexity were examined: a Naive Bayes classifier, a bidirectional LSTM network, and a fine tuned transformer based BERT model.
The experiments reveal substantial differences in performance. BERT achieved the highest accuracy, consistently exceeding 99\%, but required significantly longer training times and greater computational resources. The BiLSTM model provided a strong compromise, reaching approximately 98.56\% accuracy while maintaining moderate training costs and offering robust contextual understanding. Naive Bayes proved to be the fastest to train, on the order of milliseconds, yet delivered the lowest accuracy, averaging around 94.5\%. Class imbalance influenced all methods, particularly in the recognition of minority categories.
A fully functional demonstrative system was implemented to validate practical applicability, enabling automated routing of technical requests with throughput unattainable through manual processing. The study concludes that BiLSTM offers the most balanced solution for the examined scenario, while also outlining opportunities for future improvements and further exploration of transformer architectures. 

---
# No One Size Fits All: QueryBandits for Hallucination Mitigation 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.20332)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have led to more frequent hallucinations; yet most mitigation work focuses on open-source models for post-hoc detection and parameter editing. The dearth of studies focusing on hallucinations in closed-source models is especially concerning, as they constitute the vast majority of models in institutional deployments. We introduce QueryBandits, a model-agnostic contextual bandit framework that adaptively learns online to select the optimal query-rewrite strategy by leveraging an empirically validated and calibrated reward function. Across 16 QA scenarios, our top QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a No-Rewrite baseline and outperforms zero-shot static policies (e.g., Paraphrase or Expand) by 42.6% and 60.3%, respectively. Moreover, all contextual bandits outperform vanilla bandits across all datasets, with higher feature variance coinciding with greater variance in arm selection. This substantiates our finding that there is no single rewrite policy optimal for all queries. We also discover that certain static policies incur higher cumulative regret than No-Rewrite, indicating that an inflexible query-rewriting policy can worsen hallucinations. Thus, learning an online policy over semantic features with QueryBandits can shift model behavior purely through forward-pass mechanisms, enabling its use with closed-source models and bypassing the need for retraining or gradient-based adaptation. 

---
# What Makes a Good Query? Measuring the Impact of Human-Confusing Linguistic Features on LLM Performance 

**Authors**: William Watson, Nicole Cho, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.20300)  

**Abstract**: Large Language Model (LLM) hallucinations are usually treated as defects of the model or its decoding strategy. Drawing on classical linguistics, we argue that a query's form can also shape a listener's (and model's) response. We operationalize this insight by constructing a 22-dimension query feature vector covering clause complexity, lexical rarity, and anaphora, negation, answerability, and intention grounding, all known to affect human comprehension. Using 369,837 real-world queries, we ask: Are there certain types of queries that make hallucination more likely? A large-scale analysis reveals a consistent "risk landscape": certain features such as deep clause nesting and underspecification align with higher hallucination propensity. In contrast, clear intention grounding and answerability align with lower hallucination rates. Others, including domain specificity, show mixed, dataset- and model-dependent effects. Thus, these findings establish an empirically observable query-feature representation correlated with hallucination risk, paving the way for guided query rewriting and future intervention studies. 

---
# InterviewSim: A Scalable Framework for Interview-Grounded Personality Simulation 

**Authors**: Yu Li, Pranav Narayanan Venkit, Yada Pruksachatkun, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20294)  

**Abstract**: Simulating real personalities with large language models requires grounding generation in authentic personal data. Existing evaluation approaches rely on demographic surveys, personality questionnaires, or short AI-led interviews as proxies, but lack direct assessment against what individuals actually said. We address this gap with an interview-grounded evaluation framework for personality simulation at a large scale. We extract over 671,000 question-answer pairs from 23,000 verified interview transcripts across 1,000 public personalities, each with an average of 11.5 hours of interview content. We propose a multi-dimensional evaluation framework with four complementary metrics measuring content similarity, factual consistency, personality alignment, and factual knowledge retention. Through systematic comparison, we demonstrate that methods grounded in real interview data substantially outperform those relying solely on biographical profiles or the model's parametric knowledge. We further reveal a trade-off in how interview data is best utilized: retrieval-augmented methods excel at capturing personality style and response quality, while chronological-based methods better preserve factual consistency and knowledge retention. Our evaluation framework enables principled method selection based on application requirements, and our empirical findings provide actionable insights for advancing personality simulation research. 

---
# ConceptRM: The Quest to Mitigate Alert Fatigue through Consensus-Based Purity-Driven Data Cleaning for Reflection Modelling 

**Authors**: Yongda Yu, Lei Zhang, Xinxin Guo, Minghui Yu, Zhengqi Zhuang, Guoping Rong, Haifeng Shen, Zhengfeng Li, Boge Wang, Guoan Zhang, Bangyu Xiang, Xiaobin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20166)  

**Abstract**: In many applications involving intelligent agents, the overwhelming volume of alerts (mostly false) generated by the agents may desensitize users and cause them to overlook critical issues, leading to the so-called ''alert fatigue''. A common strategy is to train a reflection model as a filter to intercept false alerts with labelled data collected from user verification feedback. However, a key challenge is the noisy nature of such data as it is often collected in production environments. As cleaning noise via manual annotation incurs high costs, this paper proposes a novel method ConceptRM for constructing a high-quality corpus to train a reflection model capable of effectively intercepting false alerts. With only a small amount of expert annotations as anchors, ConceptRM creates perturbed datasets with varying noise ratios and utilizes co-teaching to train multiple distinct models for collaborative learning. By analyzing the consensus decisions of these models, it effectively identifies reliable negative samples from a noisy dataset. Experimental results demonstrate that ConceptRM significantly enhances the interception of false alerts with minimal annotation cost, outperforming several state-of-the-art LLM baselines by up to 53.31% on in-domain datasets and 41.67% on out-of-domain datasets. 

---
# Benchmarking Distilled Language Models: Performance and Efficiency in Resource-Constrained Settings 

**Authors**: Sachin Gopal Wani, Eric Page, Ajay Dholakia, David Ellison  

**Link**: [PDF](https://arxiv.org/pdf/2602.20164)  

**Abstract**: Knowledge distillation offers a transformative pathway to developing powerful, yet efficient, small language models (SLMs) suitable for resource-constrained environments. In this paper, we benchmark the performance and computational cost of distilled models against their vanilla and proprietary counterparts, providing a quantitative analysis of their efficiency. Our results demonstrate that distillation creates a superior performance-tocompute curve. We find that creating a distilled 8B model is over 2,000 times more compute-efficient than training its vanilla counterpart, while achieving reasoning capabilities on par with, or even exceeding, standard models ten times its size. These findings validate distillation not just as a compression technique, but as a primary strategy for building state-of-the-art, accessible AI 

---
# Talking to Yourself: Defying Forgetting in Large Language Models 

**Authors**: Yutao Sun, Mingshuai Chen, Tiancheng Zhao, Phillip Miao, Zilun Zhang, Haozhan Shen, Ruizhe Zhu, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20162)  

**Abstract**: Catastrophic forgetting remains a major challenge when fine-tuning large language models (LLMs) on narrow, task-specific data, often degrading their general knowledge and reasoning abilities. We propose SA-SFT, a lightweight self-augmentation routine in which an LLM generates self-dialogues prior to fine-tuning, and the resulting self-authored data are mixed with task data without modifying optimization or training schedules.
Despite requiring no external data or additional tuning, SA-SFT consistently mitigates catastrophic forgetting while improving in-domain performance. Across 50 evaluation scenarios, it maintains performance comparable to the original model and achieves the best results in 40 cases, outperforming common baselines such as layer freezing and external data mixing. Guided by these empirical findings, we further present a theoretical analysis suggesting that forgetting can partly stem from style-induced parameter drift, and that self-alignment through self-generated data provides an effective means to counteract this effect. Overall, our results indicate that self-augmentation offers a simple and effective mechanism for robust LLM adaptation without incurring catastrophic forgetting. 

---
# Multi-Vector Index Compression in Any Modality 

**Authors**: Hanxiang Qin, Alexander Martin, Rohan Jha, Chunsheng Zuo, Reno Kriz, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2602.21202)  

**Abstract**: We study efficient multi-vector retrieval for late interaction in any modality. Late interaction has emerged as a dominant paradigm for information retrieval in text, images, visual documents, and videos, but its computation and storage costs grow linearly with document length, making it costly for image-, video-, and audio-rich corpora. To address this limitation, we explore query-agnostic methods for compressing multi-vector document representations under a constant vector budget. We introduce four approaches for index compression: sequence resizing, memory tokens, hierarchical pooling, and a novel attention-guided clustering (AGC). AGC uses an attention-guided mechanism to identify the most semantically salient regions of a document as cluster centroids and to weight token aggregation. Evaluating these methods on retrieval tasks spanning text (BEIR), visual-document (ViDoRe), and video (MSR-VTT, MultiVENT 2.0), we show that attention-guided clustering consistently outperforms other parameterized compression methods (sequence resizing and memory tokens), provides greater flexibility in index size than non-parametric hierarchical clustering, and achieves competitive or improved performance compared to a full, uncompressed index. The source code is available at: this http URL. 

---
# Aletheia tackles FirstProof autonomously 

**Authors**: Tony Feng, Junehyuk Jung, Sang-hyun Kim, Carlo Pagano, Sergei Gukov, Chiang-Chiang Tsai, David Woodruff, Adel Javanmard, Aryan Mokhtari, Dawsen Hwang, Yuri Chervonyi, Jonathan N. Lee, Garrett Bingham, Trieu H. Trinh, Vahab Mirrokni, Quoc V. Le, Thang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2602.21201)  

**Abstract**: We report the performance of Aletheia (Feng et al., 2026b), a mathematics research agent powered by Gemini 3 Deep Think, on the inaugural FirstProof challenge. Within the allowed timeframe of the challenge, Aletheia autonomously solved 6 problems (2, 5, 7, 8, 9, 10) out of 10 according to majority expert assessments; we note that experts were not unanimous on Problem 8 (only). For full transparency, we explain our interpretation of FirstProof and disclose details about our experiments as well as our evaluation. Raw prompts and outputs are available at this https URL. 

---
# Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs 

**Authors**: Yining Hong, Huang Huang, Manling Li, Li Fei-Fei, Jiajun Wu, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.21198)  

**Abstract**: Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with ablative studies validating the complementary roles of reflection-in-action and reflection-on-action. Qualitative analyses, including real-robot trials, highlight behavioral correction through reflection. 

---
# SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards 

**Authors**: Dengjia Zhang, Xiaoou Liu, Lu Cheng, Yaqing Wang, Kenton Murray, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.21158)  

**Abstract**: Large language models (LLMs) are increasingly deployed as multi-step decision-making agents, where effective reward design is essential for guiding learning. Although recent work explores various forms of reward shaping and step-level credit assignment, a key signal remains largely overlooked: the intrinsic uncertainty of LLMs. Uncertainty reflects model confidence, reveals where exploration is needed, and offers valuable learning cues even in failed trajectories. We introduce SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards, a reinforcement learning framework that incorporates uncertainty directly into the reward design. SELAUR integrates entropy-, least-confidence-, and margin-based metrics into a combined token-level uncertainty estimate, providing dense confidence-aligned supervision, and employs a failure-aware reward reshaping mechanism that injects these uncertainty signals into step- and trajectory-level rewards to improve exploration efficiency and learning stability. Experiments on two benchmarks, ALFWorld and WebShop, show that our method consistently improves success rates over strong baselines. Ablation studies further demonstrate how uncertainty signals enhance exploration and robustness. 

---
# A Benchmark for Deep Information Synthesis 

**Authors**: Debjit Paul, Daniel Murphy, Milan Gritta, Ronald Cardenas, Victor Prokhorov, Lena Sophia Bolliger, Aysim Toker, Roy Miles, Andreea-Maria Oncescu, Jasivan Alex Sivakumar, Philipp Borchert, Ismail Elezi, Meiru Zhang, Ka Yiu Lee, Guchun Zhang, Jun Wang, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2602.21143)  

**Abstract**: Large language model (LLM)-based agents are increasingly used to solve complex tasks involving tool use, such as web browsing, code execution, and data analysis. However, current evaluation benchmarks do not adequately assess their ability to solve real-world tasks that require synthesizing information from multiple sources and inferring insights beyond simple fact retrieval. To address this, we introduce DEEPSYNTH, a novel benchmark designed to evaluate agents on realistic, time-consuming problems that combine information gathering, synthesis, and structured reasoning to produce insights. DEEPSYNTH contains 120 tasks collected across 7 domains and data sources covering 67 countries. DEEPSYNTH is constructed using a multi-stage data collection pipeline that requires annotators to collect official data sources, create hypotheses, perform manual analysis, and design tasks with verifiable answers. When evaluated on DEEPSYNTH, 11 state-of-the-art LLMs and deep research agents achieve a maximum F1 score of 8.97 and 17.5 on the LLM-judge metric, underscoring the difficulty of the benchmark. Our analysis reveals that current agents struggle with hallucinations and reasoning over large information spaces, highlighting DEEPSYNTH as a crucial benchmark for guiding future research. 

---
# An Expert Schema for Evaluating Large Language Model Errors in Scholarly Question-Answering Systems 

**Authors**: Anna Martin-Boyle, William Humphreys, Martha Brown, Cara Leckey, Harmanpreet Kaur  

**Link**: [PDF](https://arxiv.org/pdf/2602.21059)  

**Abstract**: Large Language Models (LLMs) are transforming scholarly tasks like search and summarization, but their reliability remains uncertain. Current evaluation metrics for testing LLM reliability are primarily automated approaches that prioritize efficiency and scalability, but lack contextual nuance and fail to reflect how scientific domain experts assess LLM outputs in practice. We developed and validated a schema for evaluating LLM errors in scholarly question-answering systems that reflects the assessment strategies of practicing scientists. In collaboration with domain experts, we identified 20 error patterns across seven categories through thematic analysis of 68 question-answer pairs. We validated this schema through contextual inquiries with 10 additional scientists, which showed not only which errors experts naturally identify but also how structured evaluation schemas can help them detect previously overlooked issues. Domain experts use systematic assessment strategies, including technical precision testing, value-based evaluation, and meta-evaluation of their own practices. We discuss implications for supporting expert evaluation of LLM outputs, including opportunities for personalized, schema-driven tools that adapt to individual evaluation patterns and expertise levels. 

---
# VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation 

**Authors**: Seongheon Park, Changdae Oh, Hyeong Kyu Choi, Xuefeng Du, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.21054)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently hallucinate, limiting their safe deployment in real-world applications. Existing LLM self-evaluation methods rely on a model's ability to estimate the correctness of its own outputs, which can improve deployment reliability; however, they depend heavily on language priors and are therefore ill-suited for evaluating vision-conditioned predictions. We propose VAUQ, a vision-aware uncertainty quantification framework for LVLM self-evaluation that explicitly measures how strongly a model's output depends on visual evidence. VAUQ introduces the Image-Information Score (IS), which captures the reduction in predictive uncertainty attributable to visual input, and an unsupervised core-region masking strategy that amplifies the influence of salient regions. Combining predictive entropy with this core-masked IS yields a training-free scoring function that reliably reflects answer correctness. Comprehensive experiments show that VAUQ consistently outperforms existing self-evaluation methods across multiple datasets. 

---
# PaperTrail: A Claim-Evidence Interface for Grounding Provenance in LLM-based Scholarly Q&A 

**Authors**: Anna Martin-Boyle, Cara A.C. Leckey, Martha C. Brown, Harmanpreet Kaur  

**Link**: [PDF](https://arxiv.org/pdf/2602.21045)  

**Abstract**: Large language models (LLMs) are increasingly used in scholarly question-answering (QA) systems to help researchers synthesize vast amounts of literature. However, these systems often produce subtle errors (e.g., unsupported claims, errors of omission), and current provenance mechanisms like source citations are not granular enough for the rigorous verification that scholarly domain requires. To address this, we introduce PaperTrail, a novel interface that decomposes both LLM answers and source documents into discrete claims and evidence, mapping them to reveal supported assertions, unsupported claims, and information omitted from the source texts. We evaluated PaperTrail in a within-subjects study with 26 researchers who performed two scholarly editing tasks using PaperTrail and a baseline interface. Our results show that PaperTrail significantly lowered participants' trust compared to the baseline. However, this increased caution did not translate to behavioral changes, as people continued to rely on LLM-generated scholarly edits to avoid a cognitively burdensome task. We discuss the value of claim-evidence matching for understanding LLM trustworthiness in scholarly settings, and present design implications for cognition-friendly communication of provenance information. 

---
# HiSAC: Hierarchical Sparse Activation Compression for Ultra-long Sequence Modeling in Recommenders 

**Authors**: Kun Yuan, Junyu Bi, Daixuan Cheng, Changfa Wu, Shuwen Xiao, Binbin Cao, Jian Wu, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21009)  

**Abstract**: Modern recommender systems leverage ultra-long user behavior sequences to capture dynamic preferences, but end-to-end modeling is infeasible in production due to latency and memory constraints. While summarizing history via interest centers offers a practical alternative, existing methods struggle to (1) identify user-specific centers at appropriate granularity and (2) accurately assign behaviors, leading to quantization errors and loss of long-tail preferences. To alleviate these issues, we propose Hierarchical Sparse Activation Compression (HiSAC), an efficient framework for personalized sequence modeling. HiSAC encodes interactions into multi-level semantic IDs and constructs a global hierarchical codebook. A hierarchical voting mechanism sparsely activates personalized interest-agents as fine-grained preference centers. Guided by these agents, Soft-Routing Attention aggregates historical signals in semantic space, weighting by similarity to minimize quantization error and retain long-tail behaviors. Deployed on Taobao's "Guess What You Like" homepage, HiSAC achieves significant compression and cost reduction, with online A/B tests showing a consistent 1.65% CTR uplift -- demonstrating its scalability and real-world effectiveness. 

---
# Generative Pseudo-Labeling for Pre-Ranking with LLMs 

**Authors**: Junyu Bi, Xinting Niu, Daixuan Cheng, Kun Yuan, Tao Wang, Binbin Cao, Jian Wu, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20995)  

**Abstract**: Pre-ranking is a critical stage in industrial recommendation systems, tasked with efficiently scoring thousands of recalled items for downstream ranking. A key challenge is the train-serving discrepancy: pre-ranking models are trained only on exposed interactions, yet must score all recalled candidates -- including unexposed items -- during online serving. This mismatch not only induces severe sample selection bias but also degrades generalization, especially for long-tail content. Existing debiasing approaches typically rely on heuristics (e.g., negative sampling) or distillation from biased rankers, which either mislabel plausible unexposed items as negatives or propagate exposure bias into pseudo-labels. In this work, we propose Generative Pseudo-Labeling (GPL), a framework that leverages large language models (LLMs) to generate unbiased, content-aware pseudo-labels for unexposed items, explicitly aligning the training distribution with the online serving space. By offline generating user-specific interest anchors and matching them with candidates in a frozen semantic space, GPL provides high-quality supervision without adding online latency. Deployed in a large-scale production system, GPL improves click-through rate by 3.07%, while significantly enhancing recommendation diversity and long-tail item discovery. 

---
# Multimodal MRI Report Findings Supervised Brain Lesion Segmentation with Substructures 

**Authors**: Yubin Ge, Yongsong Huang, Xiaofeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20994)  

**Abstract**: Report-supervised (RSuper) learning seeks to alleviate the need for dense tumor voxel labels with constraints derived from radiology reports (e.g., volumes, counts, sizes, locations). In MRI studies of brain tumors, however, we often involve multi-parametric scans and substructures. Here, fine-grained modality/parameter-wise reports are usually provided along with global findings and are correlated with different substructures. Moreover, the reports often describe only the largest lesion and provide qualitative or uncertain cues (``mild,'' ``possible''). Classical RSuper losses (e.g., sum volume consistency) can over-constrain or hallucinate unreported findings under such incompleteness, and are unable to utilize these hierarchical findings or exploit the priors of varied lesion types in a merged dataset. We explicitly parse the global quantitative and modality-wise qualitative findings and introduce a unified, one-sided, uncertainty-aware formulation (MS-RSuper) that: (i) aligns modality-specific qualitative cues (e.g., T1c enhancement, FLAIR edema) with their corresponding substructures using existence and absence losses; (ii) enforces one-sided lower-bounds for partial quantitative cues (e.g., largest lesion size, minimal multiplicity); and (iii) adds extra- vs. intra-axial anatomical priors to respect cohort differences. Certainty tokens scale penalties; missing cues are down-weighted. On 1238 report-labeled BraTS-MET/MEN scans, our MS-RSuper largely outperforms both a sparsely-supervised baseline and a naive RSuper method. 

---
# Predicting Sentence Acceptability Judgments in Multimodal Contexts 

**Authors**: Hyewon Jang, Nikolai Ilinykh, Sharid Loáiciga, Jey Han Lau, Shalom Lappin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20918)  

**Abstract**: Previous work has examined the capacity of deep neural networks (DNNs), particularly transformers, to predict human sentence acceptability judgments, both independently of context, and in document contexts. We consider the effect of prior exposure to visual images (i.e., visual context) on these judgments for humans and large language models (LLMs). Our results suggest that, in contrast to textual context, visual images appear to have little if any impact on human acceptability ratings. However, LLMs display the compression effect seen in previous work on human judgments in document contexts. Different sorts of LLMs are able to predict human acceptability judgments to a high degree of accuracy, but in general, their performance is slightly better when visual contexts are removed. Moreover, the distribution of LLM judgments varies among models, with Qwen resembling human patterns, and others diverging from them. LLM-generated predictions on sentence acceptability are highly correlated with their normalised log probabilities in general. However, the correlations decrease when visual contexts are present, suggesting that a higher gap exists between the internal representations of LLMs and their generated predictions in the presence of visual contexts. Our experimental work suggests interesting points of similarity and of difference between human and LLM processing of sentences in multimodal contexts. 

---
# RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition 

**Authors**: Kun Ran, Marwah Alaofi, Danula Hettiachchi, Chenglong Ma, Khoi Nguyen Dinh Anh, Khoi Vo Nguyen, Sachin Pathiyan Cherumanal, Lida Rashidi, Falk Scholer, Damiano Spina, Shuoqi Sun, Oleg Zendel  

**Link**: [PDF](https://arxiv.org/pdf/2602.20735)  

**Abstract**: This paper presents the award-winning RMIT-ADM+S system for the Text-to-Text
track of the NeurIPS~2025 MMU-RAG Competition. We introduce Routing-to-RAG
(R2RAG), a research-focused retrieval-augmented generation (RAG)
architecture composed of lightweight components that dynamically adapt the
retrieval strategy based on inferred query complexity and evidence
sufficiency. The system uses smaller LLMs, enabling operation on a single
consumer-grade GPU while supporting complex research tasks. It builds on the
G-RAG system, winner of the ACM~SIGIR~2025 LiveRAG Challenge, and extends it
with modules informed by qualitative review of outputs. R2RAG won the Best
Dynamic Evaluation award in the Open Source category, demonstrating high
effectiveness with careful design and efficient use of resources. 

---
# Counterfactual Simulation Training for Chain-of-Thought Faithfulness 

**Authors**: Peter Hase, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2602.20710)  

**Abstract**: Inspecting Chain-of-Thought reasoning is among the most common means of understanding why an LLM produced its output. But well-known problems with CoT faithfulness severely limit what insights can be gained from this practice. In this paper, we introduce a training method called Counterfactual Simulation Training (CST), which aims to improve CoT faithfulness by rewarding CoTs that enable a simulator to accurately predict a model's outputs over counterfactual inputs. We apply CST in two settings: (1) CoT monitoring with cue-based counterfactuals, to detect when models rely on spurious features, reward hack, or are sycophantic, and (2) counterfactual simulation over generic model-based counterfactuals, to encourage models to produce more faithful, generalizable reasoning in the CoT. Experiments with models up to 235B parameters show that CST can substantially improve monitor accuracy on cue-based counterfactuals (by 35 accuracy points) as well as simulatability over generic counterfactuals (by 2 points). We further show that: (1) CST outperforms prompting baselines, (2) rewriting unfaithful CoTs with an LLM is 5x more efficient than RL alone, (3) faithfulness improvements do not generalize to dissuading cues (as opposed to persuading cues), and (4) larger models do not show more faithful CoT out of the box, but they do benefit more from CST. These results suggest that CST can improve CoT faithfulness in general, with promising applications for CoT monitoring. Code for experiments in this paper is available at this https URL 

---
# SpecMind: Cognitively Inspired, Interactive Multi-Turn Framework for Postcondition Inference 

**Authors**: Cuong Chi Le, Minh V.T Pham, Tung Vu Duy, Cuong Duc Van, Huy N. Phan, Hoang N. Phan, Tien N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.20610)  

**Abstract**: Specifications are vital for ensuring program correctness, yet writing them manually remains challenging and time-intensive. Recent large language model (LLM)-based methods have shown successes in generating specifications such as postconditions, but existing single-pass prompting often yields inaccurate results. In this paper, we present SpecMind, a novel framework for postcondition generation that treats LLMs as interactive and exploratory reasoners rather than one-shot generators. SpecMind employs feedback-driven multi-turn prompting approaches, enabling the model to iteratively refine candidate postconditions by incorporating implicit and explicit correctness feedback, while autonomously deciding when to stop. This process fosters deeper code comprehension and improves alignment with true program behavior via exploratory attempts. Our empirical evaluation shows that SpecMind significantly outperforms state-of-the-art approaches in both accuracy and completeness of generated postconditions. 

---
# GATES: Self-Distillation under Privileged Context with Consensus Gating 

**Authors**: Alex Stein, Furong Huang, Tom Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2602.20574)  

**Abstract**: We study self-distillation in settings where supervision is unreliable: there are no ground truth labels, verifiable rewards, or external graders to evaluate answers. We focus on document-grounded question answering with asymmetric context, where a single model serves as both tutor (with access to a relevant source document during training) and student (answering from the question alone at test time). Rather than assuming tutor correctness, we derive supervision online from tutor consensus by sampling multiple document-grounded reasoning traces and using agreement to gate learning. Conditioned on this reliability signal, we distill knowledge through full tutor reasoning trajectories (not just final answers), providing a dense and stable learning signal. Empirically, this consensus-gated trajectory distillation substantially improves transfer to the document-free student. Held-out in-domain accuracy under asymmetric evaluation improves from 46.0\% to 62.0\%, and average (maj@8) accuracy on public document-free math benchmarks improves from 20.2\% to 35.4\%. 

---
# Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits for RL Post-Training 

**Authors**: Zhengyao Gu, Jonathan Light, Raul Astudillo, Ziyu Ye, Langzhou He, Henry Peng Zou, Wei Cheng, Santiago Paternain, Philip S. Yu, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2602.20532)  

**Abstract**: Post-training large foundation models with reinforcement learning typically relies on massive and heterogeneous datasets, making effective curriculum learning both critical and challenging. In this work, we propose ACTOR-CURATOR, a scalable and fully automated curriculum learning framework for reinforcement learning post-training of large language models (LLMs). ACTOR-CURATOR learns a neural curator that dynamically selects training problems from large problem banks by directly optimizing for expected policy performance improvement. We formulate problem selection as a non-stationary stochastic bandit problem, derive a principled loss function based on online stochastic mirror descent, and establish regret guarantees under partial feedback. Empirically, ACTOR-CURATOR consistently outperforms uniform sampling and strong curriculum baselines across a wide range of challenging reasoning benchmarks, demonstrating improved training stability and efficiency. Notably, it achieves relative gains of 28.6% on AIME2024 and 30.5% on ARC-1D over the strongest baseline and up to 80% speedup. These results suggest that ACTOR-CURATOR is a powerful and practical approach for scalable LLM post-training. 

---
# Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI coordination 

**Authors**: Rakshit Trivedi, Kartik Sharma, David C Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2602.20517)  

**Abstract**: Effective human-AI coordination requires artificial agents capable of exhibiting and responding to human-like behaviors while adapting to changing contexts. Imitation learning has emerged as one of the prominent approaches to build such agents by training them to mimic human-demonstrated behaviors. However, current methods struggle to capture the inherent diversity and non-Markovian nature of human behavior and lack the ability to steer behavior at inference time. Drawing inspiration from the theory of human cognitive processes, where inner speech guides action selection before execution, we propose MIMIC (Modeling Inner Motivations for Imitation and Control), a framework that uses language as an internal representation of behavioral intent. MIMIC employs the novel use of vision-language models as linguistic scaffolding to train a conditional variational autoencoder capable of generating inner speech from observations. A diffusion-based behavior cloning policy then selects actions conditioned on current observations and the generated inner speech. MIMIC enables fine-grained steering of behavior at inference time by conditioning the agent on behavior-specific speech. Experiments across robotic manipulation tasks and human-AI collaboration games demonstrate that MIMIC significantly enhances both behavior diversity and fidelity to human demonstrations while enabling nuanced behavioral steering without training on additional demonstrations. We open source our code and provide pre-trained MIMIC agents and qualitative demos at: this https URL. 

---
# PreScience: A Benchmark for Forecasting Scientific Contributions 

**Authors**: Anirudh Ajith, Amanpreet Singh, Jay DeYoung, Nadav Kunievsky, Austin C. Kozlowski, Oyvind Tafjord, James Evans, Daniel S. Weld, Tom Hope, Doug Downey  

**Link**: [PDF](https://arxiv.org/pdf/2602.20459)  

**Abstract**: Can AI systems trained on the scientific record up to a fixed point in time forecast the scientific advances that follow? Such a capability could help researchers identify collaborators and impactful research directions, and anticipate which problems and methods will become central next. We introduce PreScience -- a scientific forecasting benchmark that decomposes the research process into four interdependent generative tasks: collaborator prediction, prior work selection, contribution generation, and impact prediction. PreScience is a carefully curated dataset of 98K recent AI-related research papers, featuring disambiguated author identities, temporally aligned scholarly metadata, and a structured graph of companion author publication histories and citations spanning 502K total papers. We develop baselines and evaluations for each task, including LACERScore, a novel LLM-based measure of contribution similarity that outperforms previous metrics and approximates inter-annotator agreement. We find substantial headroom remains in each task -- e.g. in contribution generation, frontier LLMs achieve only moderate similarity to the ground-truth (GPT-5, averages 5.6 on a 1-10 scale). When composed into a 12-month end-to-end simulation of scientific production, the resulting synthetic corpus is systematically less diverse and less novel than human-authored research from the same period. 

---
# Protein Language Models Diverge from Natural Language: Comparative Analysis and Improved Inference 

**Authors**: Anna Hart, Chi Han, Jeonghwan Kim, Huimin Zhao, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.20449)  

**Abstract**: Modern Protein Language Models (PLMs) apply transformer-based model architectures from natural language processing to biological sequences, predicting a variety of protein functions and properties. However, protein language has key differences from natural language, such as a rich functional space despite a vocabulary of only 20 amino acids. These differences motivate research into how transformer-based architectures operate differently in the protein domain and how we can better leverage PLMs to solve protein-related tasks. In this work, we begin by directly comparing how the distribution of information stored across layers of attention heads differs between the protein and natural language domain. Furthermore, we adapt a simple early-exit technique-originally used in the natural language domain to improve efficiency at the cost of performance-to achieve both increased accuracy and substantial efficiency gains in protein non-structural property prediction by allowing the model to automatically select protein representations from the intermediate layers of the PLMs for the specific task and protein at hand. We achieve performance gains ranging from 0.4 to 7.01 percentage points while simultaneously improving efficiency by over 10 percent across models and non-structural prediction tasks. Our work opens up an area of research directly comparing how language models change behavior when moved into the protein domain and advances language modeling in biological domains. 

---
# MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation 

**Authors**: Taha Koleilat, Hojat Asgariandehkordi, Omid Nejati Manzari, Berardino Barile, Yiming Xiao, Hassan Rivaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.20423)  

**Abstract**: Medical image segmentation remains challenging due to limited annotations for training, ambiguous anatomical features, and domain shifts. While vision-language models such as CLIP offer strong cross-modal representations, their potential for dense, text-guided medical image segmentation remains underexplored. We present MedCLIPSeg, a novel framework that adapts CLIP for robust, data-efficient, and uncertainty-aware medical image segmentation. Our approach leverages patch-level CLIP embeddings through probabilistic cross-modal attention, enabling bidirectional interaction between image and text tokens and explicit modeling of predictive uncertainty. Together with a soft patch-level contrastive loss that encourages more nuanced semantic learning across diverse textual prompts, MedCLIPSeg effectively improves data efficiency and domain generalizability. Extensive experiments across 16 datasets spanning five imaging modalities and six organs demonstrate that MedCLIPSeg outperforms prior methods in accuracy, efficiency, and robustness, while providing interpretable uncertainty maps that highlight local reliability of segmentation results. This work demonstrates the potential of probabilistic vision-language modeling for text-driven medical image segmentation. 

---
# An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models 

**Authors**: Cathy Shyr, Yan Hu, Rory J. Tinker, Thomas A. Cassini, Kevin W. Byram, Rizwan Hamid, Daniel V. Fabbri, Adam Wright, Josh F. Peterson, Lisa Bastarache, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20324)  

**Abstract**: Phenotyping is fundamental to rare disease diagnosis, but manual curation of structured phenotypes from clinical notes is labor-intensive and difficult to scale. Existing artificial intelligence approaches typically optimize individual components of phenotyping but do not operationalize the full clinical workflow of extracting features from clinical text, standardizing them to Human Phenotype Ontology (HPO) terms, and prioritizing diagnostically informative HPO terms. We developed RARE-PHENIX, an end-to-end AI framework for rare disease phenotyping that integrates large language model-based phenotype extraction, ontology-grounded standardization to HPO terms, and supervised ranking of diagnostically informative phenotypes. We trained RARE-PHENIX using data from 2,671 patients across 11 Undiagnosed Diseases Network clinical sites, and externally validated it on 16,357 real-world clinical notes from Vanderbilt University Medical Center. Using clinician-curated HPO terms as the gold standard, RARE-PHENIX consistently outperformed a state-of-the-art deep learning baseline (PhenoBERT) across ontology-based similarity and precision-recall-F1 metrics in end-to-end evaluation (i.e., ontology-based similarity of 0.70 vs. 0.58). Ablation analyses demonstrated performance improvements with the addition of each module in RARE-PHENIX (extraction, standardization, and prioritization), supporting the value of modeling the full clinical phenotyping workflow. By modeling phenotyping as a clinically aligned workflow rather than a single extraction task, RARE-PHENIX provides structured, ranked phenotypes that are more concordant with clinician curation and has the potential to support human-in-the-loop rare disease diagnosis in real-world settings. 

---
# Exploring Anti-Aging Literature via ConvexTopics and Large Language Models 

**Authors**: Lana E. Yeganova, Won G. Kim, Shubo Tian, Natalie Xie, Donald C. Comeau, W. John Wilbur, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20224)  

**Abstract**: The rapid expansion of biomedical publications creates challenges for organizing knowledge and detecting emerging trends, underscoring the need for scalable and interpretable methods. Common clustering and topic modeling approaches such as K-means or LDA remain sensitive to initialization and prone to local optima, limiting reproducibility and evaluation. We propose a reformulation of a convex optimization based clustering algorithm that produces stable, fine-grained topics by selecting exemplars from the data and guaranteeing a global optimum. Applied to about 12,000 PubMed articles on aging and longevity, our method uncovers topics validated by medical experts. It yields interpretable topics spanning from molecular mechanisms to dietary supplements, physical activity, and gut microbiota. The method performs favorably, and most importantly, its reproducibility and interpretability distinguish it from common clustering approaches, including K-means, LDA, and BERTopic. This work provides a basis for developing scalable, web-accessible tools for knowledge discovery. 

---
# MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Elastic LLMs 

**Authors**: Dongwei Wang, Jinhee Kim, Seokho Han, Denis Gudovskiy, Yohei Nakata, Tomoyuki Okuno, KhayTze Peong, Kang Eun Jeon, Jong Hwan Ko, Yiran Chen, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20191)  

**Abstract**: Changing runtime complexity on cloud and edge devices necessitates elastic large language model (LLM) deployment, where an LLM can be inferred with various quantization precisions based on available computational resources. However, it has been observed that the calibration parameters for quantization are typically linked to specific precisions, which presents challenges during elastic-precision calibration and precision switching at runtime. In this work, we attribute the source of varying calibration parameters to the varying token-level sensitivity caused by a precision-dependent outlier migration this http URL by this observation, we propose \texttt{MoBiQuant}, a novel Mixture-of-Bits quantization framework that adjusts weight precision for elastic LLM inference based on token sensitivity. Specifically, we propose the many-in-one recursive residual quantization that can iteratively reconstruct higher-precision weights and the token-aware router to dynamically select the number of residual bit slices. MoBiQuant enables smooth precision switching while improving generalization for the distribution of token outliers. Experimental results demonstrate that MoBiQuant exhibits strong elasticity, enabling it to match the performance of bit-specific calibrated PTQ on LLaMA3-8B without repeated calibration. 

---
# Graph Modelling Analysis of Speech-Gesture Interaction for Aphasia Severity Estimation 

**Authors**: Navya Martin Kollapally, Christa Akers, Renjith Nelson Joseph  

**Link**: [PDF](https://arxiv.org/pdf/2602.20163)  

**Abstract**: Aphasia is an acquired language disorder caused by injury to the regions of the brain that are responsible for language. Aphasia may impair the use and comprehension of written and spoken language. The Western Aphasia Battery-Revised (WAB-R) is an assessment tool administered by speech-language pathologists (SLPs) to evaluate the aphasia type and severity. Because the WAB-R measures isolated linguistic skills, there has been growing interest in the assessment of discourse production as a more holistic representation of everyday language abilities. Recent advancements in speech analysis focus on automated estimation of aphasia severity from spontaneous speech, relying mostly in isolated linguistic or acoustical features. In this work, we propose a graph neural network-based framework for estimating aphasia severity. We represented each participant's discourse as a directed multi-modal graph, where nodes represent lexical items and gestures and edges encode word-word, gesture-word, and word-gesture transitions. GraphSAGE is employed to learn participant-level embeddings, thus integrating information from immediate neighbors and overall graph structure. Our results suggest that aphasia severity is not encoded in isolated lexical distribution, but rather emerges from structured interactions between speech and gesture. The proposed architecture offers a reliable automated aphasia assessment, with possible uses in bedside screening and telehealth-based monitoring. 

---
