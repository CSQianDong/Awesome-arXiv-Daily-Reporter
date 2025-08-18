# HumorPlanSearch: Structured Planning and HuCoT for Contextual AI Humor 

**Authors**: Shivam Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2508.11429)  

**Abstract**: Automated humor generation with Large Language Models (LLMs) often yields jokes that feel generic, repetitive, or tone-deaf because humor is deeply situated and hinges on the listener's cultural background, mindset, and immediate context. We introduce HumorPlanSearch, a modular pipeline that explicitly models context through: (1) Plan-Search for diverse, topic-tailored strategies; (2) Humor Chain-of-Thought (HuCoT) templates capturing cultural and stylistic reasoning; (3) a Knowledge Graph to retrieve and adapt high-performing historical strategies; (4) novelty filtering via semantic embeddings; and (5) an iterative judge-driven revision loop. To evaluate context sensitivity and comedic quality, we propose the Humor Generation Score (HGS), which fuses direct ratings, multi-persona feedback, pairwise win-rates, and topic relevance. In experiments across nine topics with feedback from 13 human judges, our full pipeline (KG + Revision) boosts mean HGS by 15.4 percent (p < 0.05) over a strong baseline. By foregrounding context at every stage from strategy planning to multi-signal evaluation, HumorPlanSearch advances AI-driven humor toward more coherent, adaptive, and culturally attuned comedy. 

---
# Reference Points in LLM Sentiment Analysis: The Role of Structured Context 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2508.11454)  

**Abstract**: Large language models (LLMs) are now widely used across many fields, including marketing research. Sentiment analysis, in particular, helps firms understand consumer preferences. While most NLP studies classify sentiment from review text alone, marketing theories, such as prospect theory and expectation--disconfirmation theory, point out that customer evaluations are shaped not only by the actual experience but also by additional reference points. This study therefore investigates how the content and format of such supplementary information affect sentiment analysis using LLMs. We compare natural language (NL) and JSON-formatted prompts using a lightweight 3B parameter model suitable for practical marketing applications. Experiments on two Yelp categories (Restaurant and Nightlife) show that the JSON prompt with additional information outperforms all baselines without fine-tuning: Macro-F1 rises by 1.6% and 4% while RMSE falls by 16% and 9.1%, respectively, making it deployable in resource-constrained edge devices. Furthermore, a follow-up analysis confirms that performance gains stem from genuine contextual reasoning rather than label proxying. This work demonstrates that structured prompting can enable smaller models to achieve competitive performance, offering a practical alternative to large-scale model deployment. 

---
# Survey-to-Behavior: Downstream Alignment of Human Values in LLMs via Survey Questions 

**Authors**: Shangrui Nie, Florian Mai, David Kaczér, Charles Welch, Zhixue Zhao, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2508.11414)  

**Abstract**: Large language models implicitly encode preferences over human values, yet steering them often requires large training data. In this work, we investigate a simple approach: Can we reliably modify a model's value system in downstream behavior by training it to answer value survey questions accordingly? We first construct value profiles of several open-source LLMs by asking them to rate a series of value-related descriptions spanning 20 distinct human values, which we use as a baseline for subsequent experiments. We then investigate whether the value system of a model can be governed by fine-tuning on the value surveys. We evaluate the effect of finetuning on the model's behavior in two ways; first, we assess how answers change on in-domain, held-out survey questions. Second, we evaluate whether the model's behavior changes in out-of-domain settings (situational scenarios). To this end, we construct a contextualized moral judgment dataset based on Reddit posts and evaluate changes in the model's behavior in text-based adventure games. We demonstrate that our simple approach can not only change the model's answers to in-domain survey questions, but also produces substantial shifts (value alignment) in implicit downstream task behavior. 

---
# Feedback Indicators: The Alignment between Llama and a Teacher in Language Learning 

**Authors**: Sylvio Rüdian, Yassin Elsir, Marvin Kretschmer, Sabine Cayrou, Niels Pinkwart  

**Link**: [PDF](https://arxiv.org/pdf/2508.11364)  

**Abstract**: Automated feedback generation has the potential to enhance students' learning progress by providing timely and targeted feedback. Moreover, it can assist teachers in optimizing their time, allowing them to focus on more strategic and personalized aspects of teaching. To generate high-quality, information-rich formative feedback, it is essential first to extract relevant indicators, as these serve as the foundation upon which the feedback is constructed. Teachers often employ feedback criteria grids composed of various indicators that they evaluate systematically. This study examines the initial phase of extracting such indicators from students' submissions of a language learning course using the large language model Llama 3.1. Accordingly, the alignment between indicators generated by the LLM and human ratings across various feedback criteria is investigated. The findings demonstrate statistically significant strong correlations, even in cases involving unanticipated combinations of indicators and criteria. The methodology employed in this paper offers a promising foundation for extracting indicators from students' submissions using LLMs. Such indicators can potentially be utilized to auto-generate explainable and transparent formative feedback in future research. 

---
# LLM Compression: How Far Can We Go in Balancing Size and Performance? 

**Authors**: Sahil Sk, Debasish Dhal, Sonal Khosla, Sk Shahid, Sambit Shekhar, Akash Dhaka, Shantipriya Parida, Dilip K. Prasad, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2508.11318)  

**Abstract**: Quantization is an essential and popular technique for improving the accessibility of large language models (LLMs) by reducing memory usage and computational costs while maintaining performance. In this study, we apply 4-bit Group Scaling Quantization (GSQ) and Generative Pretrained Transformer Quantization (GPTQ) to LLaMA 1B, Qwen 0.5B, and PHI 1.5B, evaluating their impact across multiple NLP tasks. We benchmark these models on MS MARCO (Information Retrieval), BoolQ (Boolean Question Answering), and GSM8K (Mathematical Reasoning) datasets, assessing both accuracy and efficiency across various tasks. The study measures the trade-offs between model compression and task performance, analyzing key evaluation metrics, namely accuracy, inference latency, and throughput (total output tokens generated per second), providing insights into the suitability of low-bit quantization for real-world deployment. Using the results, users can then make suitable decisions based on the specifications that need to be met. We discuss the pros and cons of GSQ and GPTQ techniques on models of different sizes, which also serve as a benchmark for future experiments. 

---
# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems 

**Authors**: Beichen Guo, Zhiyuan Wen, Yu Yang, Peng Gao, Ruosong Yang, Jiaxing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11310)  

**Abstract**: The growing interest in automatic survey generation (ASG), a task that traditionally required considerable time and effort, has been spurred by recent advances in large language models (LLMs). With advancements in retrieval-augmented generation (RAG) and the rising popularity of multi-agent systems (MASs), synthesizing academic surveys using LLMs has become a viable approach, thereby elevating the need for robust evaluation methods in this domain. However, existing evaluation methods suffer from several limitations, including biased metrics, a lack of human preference, and an over-reliance on LLMs-as-judges. To address these challenges, we propose SGSimEval, a comprehensive benchmark for Survey Generation with Similarity-Enhanced Evaluation that evaluates automatic survey generation systems by integrating assessments of the outline, content, and references, and also combines LLM-based scoring with quantitative metrics to provide a multifaceted evaluation framework. In SGSimEval, we also introduce human preference metrics that emphasize both inherent quality and similarity to humans. Extensive experiments reveal that current ASG systems demonstrate human-comparable superiority in outline generation, while showing significant room for improvement in content and reference generation, and our evaluation metrics maintain strong consistency with human assessments. 

---
# Aware First, Think Less: Dynamic Boundary Self-Awareness Drives Extreme Reasoning Efficiency in Large Language Models 

**Authors**: Qiguang Chen, Dengyun Peng, Jinhao Liu, HuiKang Su, Jiannan Guan, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2508.11582)  

**Abstract**: Recent advancements in large language models (LLMs) have greatly improved their capabilities on complex reasoning tasks through Long Chain-of-Thought (CoT). However, this approach often results in substantial redundancy, impairing computational efficiency and causing significant delays in real-time applications. To improve the efficiency, current methods often rely on human-defined difficulty priors, which do not align with the LLM's self-awared difficulty, leading to inefficiencies. In this paper, we introduce the Dynamic Reasoning-Boundary Self-Awareness Framework (DR. SAF), which enables models to dynamically assess and adjust their reasoning depth in response to problem complexity. DR. SAF integrates three key components: Boundary Self-Awareness Alignment, Adaptive Reward Management, and a Boundary Preservation Mechanism. These components allow models to optimize their reasoning processes, balancing efficiency and accuracy without compromising performance. Our experimental results demonstrate that DR. SAF achieves a 49.27% reduction in total response tokens with minimal loss in accuracy. The framework also delivers a 6.59x gain in token efficiency and a 5x reduction in training time, making it well-suited to resource-limited settings. During extreme training, DR. SAF can even surpass traditional instruction-based models in token efficiency with more than 16% accuracy improvement. 

---
# SafeConstellations: Steering LLM Safety to Reduce Over-Refusals Through Task-Specific Trajectory 

**Authors**: Utsav Maskey, Sumit Yadav, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2508.11290)  

**Abstract**: LLMs increasingly exhibit over-refusal behavior, where safety mechanisms cause models to reject benign instructions that superficially resemble harmful content. This phenomena diminishes utility in production applications that repeatedly rely on common prompt templates or applications that frequently rely on LLMs for specific tasks (e.g. sentiment analysis, language translation). Through comprehensive evaluation, we demonstrate that LLMs still tend to refuse responses to harmful instructions when those instructions are reframed to appear as benign tasks. Our mechanistic analysis reveal that LLMs follow distinct "constellation" patterns in embedding space as representations traverse layers, with each task maintaining consistent trajectories that shift predictably between refusal and non-refusal cases. We introduce SafeConstellations, an inference-time trajectory-shifting approach that tracks task-specific trajectory patterns and guides representations toward non-refusal pathways. By selectively guiding model behavior only on tasks prone to over-refusal, and by preserving general model behavior, our method reduces over-refusal rates by up to 73% with minimal impact on utility-offering a principled approach to mitigating over-refusals. 

---
# LETToT: Label-Free Evaluation of Large Language Models On Tourism Using Expert Tree-of-Thought 

**Authors**: Ruiyan Qi, Congding Wen, Weibo Zhou, Shangsong Liang, Lingbo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11280)  

**Abstract**: Evaluating large language models (LLMs) in specific domain like tourism remains challenging due to the prohibitive cost of annotated benchmarks and persistent issues like hallucinations. We propose $\textbf{L}$able-Free $\textbf{E}$valuation of LLM on $\textbf{T}$ourism using Expert $\textbf{T}$ree-$\textbf{o}$f-$\textbf{T}$hought (LETToT), a framework that leverages expert-derived reasoning structures-instead of labeled data-to access LLMs in tourism. First, we iteratively refine and validate hierarchical ToT components through alignment with generic quality dimensions and expert feedback. Results demonstrate the effectiveness of our systematically optimized expert ToT with 4.99-14.15\% relative quality gains over baselines. Second, we apply LETToT's optimized expert ToT to evaluate models of varying scales (32B-671B parameters), revealing: (1) Scaling laws persist in specialized domains (DeepSeek-V3 leads), yet reasoning-enhanced smaller models (e.g., DeepSeek-R1-Distill-Llama-70B) close this gap; (2) For sub-72B models, explicit reasoning architectures outperform counterparts in accuracy and conciseness ($p<0.05$). Our work established a scalable, label-free paradigm for domain-specific LLM evaluation, offering a robust alternative to conventional annotated benchmarks. 

---
# UNVEILING: What Makes Linguistics Olympiad Puzzles Tricky for LLMs? 

**Authors**: Mukund Choudhary, KV Aditya Srivatsa, Gaurja Aeron, Antara Raaghavi Bhattacharya, Dang Khoa Dang Dinh, Ikhlasul Akmal Hanif, Daria Kotova, Ekaterina Kochmar, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.11260)  

**Abstract**: Large language models (LLMs) have demonstrated potential in reasoning tasks, but their performance on linguistics puzzles remains consistently poor. These puzzles, often derived from Linguistics Olympiad (LO) contests, provide a minimal contamination environment to assess LLMs' linguistic reasoning abilities across low-resource languages. This work analyses LLMs' performance on 629 problems across 41 low-resource languages by labelling each with linguistically informed features to unveil weaknesses. Our analyses show that LLMs struggle with puzzles involving higher morphological complexity and perform better on puzzles involving linguistic features that are also found in English. We also show that splitting words into morphemes as a pre-processing step improves solvability, indicating a need for more informed and language-specific tokenisers. These findings thus offer insights into some challenges in linguistic reasoning and modelling of low-resource languages. 

---
# Online Anti-sexist Speech: Identifying Resistance to Gender Bias in Political Discourse 

**Authors**: Aditi Dutta, Susan Banducci  

**Link**: [PDF](https://arxiv.org/pdf/2508.11434)  

**Abstract**: Anti-sexist speech, i.e., public expressions that challenge or resist gendered abuse and sexism, plays a vital role in shaping democratic debate online. Yet automated content moderation systems, increasingly powered by large language models (LLMs), may struggle to distinguish such resistance from the sexism it opposes. This study examines how five LLMs classify sexist, anti-sexist, and neutral political tweets from the UK, focusing on high-salience trigger events involving female Members of Parliament in the year 2022. Our analysis show that models frequently misclassify anti-sexist speech as harmful, particularly during politically charged events where rhetorical styles of harm and resistance converge. These errors risk silencing those who challenge sexism, with disproportionate consequences for marginalised voices. We argue that moderation design must move beyond binary harmful/not-harmful schemas, integrate human-in-the-loop review during sensitive events, and explicitly include counter-speech in training data. By linking feminist scholarship, event-based analysis, and model evaluation, this work highlights the sociotechnical challenges of safeguarding resistance speech in digital political spaces. 

---
# Personalized Distractor Generation via MCTS-Guided Reasoning Reconstruction 

**Authors**: Tao Wu, Jingyuan Chen, Wang Lin, Jian Zhan, Mengze Li, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11184)  

**Abstract**: Distractors, incorrect but plausible answer choices in multiple-choice questions (MCQs), play a critical role in educational assessment by diagnosing student misconceptions. Recent work has leveraged large language models (LLMs) to generate shared, group-level distractors by learning common error patterns across large student populations. However, such distractors often fail to capture the diverse reasoning errors of individual students, limiting their diagnostic effectiveness. To address this limitation, we introduce the task of personalized distractor generation, which aims to generate tailored distractors based on individual misconceptions inferred from each student's past question-answering (QA) records, ensuring every student receives options that effectively exposes their specific reasoning errors. While promising, this task is challenging because each student typically has only a few QA records, which often lack the student's underlying reasoning processes, making training-based group-level approaches infeasible. To overcome this, we propose a training-free two-stage framework. In the first stage, we construct a student-specific misconception prototype by applying Monte Carlo Tree Search (MCTS) to recover the student's reasoning trajectories from past incorrect answers. In the second stage, this prototype guides the simulation of the student's reasoning on new questions, enabling the generation of personalized distractors that align with the student's recurring misconceptions. Experiments show that our approach achieves the best performance in generating plausible, personalized distractors for 140 students, and also effectively generalizes to group-level settings, highlighting its robustness and adaptability. 

---
# When Punctuation Matters: A Large-Scale Comparison of Prompt Robustness Methods for LLMs 

**Authors**: Mikhail Seleznyov, Mikhail Chaichuk, Gleb Ershov, Alexander Panchenko, Elena Tutubalina, Oleg Somov  

**Link**: [PDF](https://arxiv.org/pdf/2508.11383)  

**Abstract**: Large Language Models (LLMs) are highly sensitive to subtle, non-semantic variations in prompt phrasing and formatting. In this work, we present the first systematic evaluation of 5 methods for improving prompt robustness within a unified experimental framework. We benchmark these techniques on 8 models from Llama, Qwen and Gemma families across 52 tasks from Natural Instructions dataset. Our evaluation covers robustness methods from both fine-tuned and in-context learning paradigms, and tests their generalization against multiple types of distribution shifts. Finally, we extend our analysis to GPT-4.1 and DeepSeek V3 to assess frontier models' current robustness to format perturbations. Our findings offer actionable insights into the relative effectiveness of these robustness methods, enabling practitioners to make informed decisions when aiming for stable and reliable LLM performance in real-world applications. Code: this https URL. 

---
# ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection 

**Authors**: Axel Delaval, Shujian Yang, Haicheng Wang, Han Qiu, Jialiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11281)  

**Abstract**: Detecting toxic content using language models is crucial yet challenging. While substantial progress has been made in English, toxicity detection in French remains underdeveloped, primarily due to the lack of culturally relevant, large-scale datasets. In this work, we introduce TOXIFRENCH, a new public benchmark of 53,622 French online comments, constructed via a semi-automated annotation pipeline that reduces manual labeling to only 10% through high-confidence LLM-based pre-annotation and human verification. Then, we benchmark a broad range of models and uncover a counterintuitive insight: Small Language Models (SLMs) outperform many larger models in robustness and generalization under the toxicity detection task. Motivated by this finding, we propose a novel Chain-of-Thought (CoT) fine-tuning strategy using a dynamic weighted loss that progressively emphasizes the model's final decision, significantly improving faithfulness. Our fine-tuned 4B model achieves state-of-the-art performance, improving its F1 score by 13% over its baseline and outperforming LLMs such as GPT-40 and Gemini-2.5. Further evaluation on a cross-lingual toxicity benchmark demonstrates strong multilingual ability, suggesting that our methodology can be effectively extended to other languages and safety-critical classification tasks. 

---
# AI in Mental Health: Emotional and Sentiment Analysis of Large Language Models' Responses to Depression, Anxiety, and Stress Queries 

**Authors**: Arya VarastehNezhad, Reza Tavasoli, Soroush Elyasi, MohammadHossein LotfiNia, Hamed Farbeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.11285)  

**Abstract**: Depression, anxiety, and stress are widespread mental health concerns that increasingly drive individuals to seek information from Large Language Models (LLMs). This study investigates how eight LLMs (Claude Sonnet, Copilot, Gemini Pro, GPT-4o, GPT-4o mini, Llama, Mixtral, and Perplexity) reply to twenty pragmatic questions about depression, anxiety, and stress when those questions are framed for six user profiles (baseline, woman, man, young, old, and university student). The models generated 2,880 answers, which we scored for sentiment and emotions using state-of-the-art tools. Our analysis revealed that optimism, fear, and sadness dominated the emotional landscape across all outputs, with neutral sentiment maintaining consistently high values. Gratitude, joy, and trust appeared at moderate levels, while emotions such as anger, disgust, and love were rarely expressed. The choice of LLM significantly influenced emotional expression patterns. Mixtral exhibited the highest levels of negative emotions including disapproval, annoyance, and sadness, while Llama demonstrated the most optimistic and joyful responses. The type of mental health condition dramatically shaped emotional responses: anxiety prompts elicited extraordinarily high fear scores (0.974), depression prompts generated elevated sadness (0.686) and the highest negative sentiment, while stress-related queries produced the most optimistic responses (0.755) with elevated joy and trust. In contrast, demographic framing of queries produced only marginal variations in emotional tone. Statistical analyses confirmed significant model-specific and condition-specific differences, while demographic influences remained minimal. These findings highlight the critical importance of model selection in mental health applications, as each LLM exhibits a distinct emotional signature that could significantly impact user experience and outcomes. 

---
# Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering 

**Authors**: Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, Ning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11247)  

**Abstract**: Multi-hop question answering (MHQA) requires integrating knowledge scattered across multiple passages to derive the correct answer. Traditional retrieval-augmented generation (RAG) methods primarily focus on coarse-grained textual semantic similarity and ignore structural associations among dispersed knowledge, which limits their effectiveness in MHQA tasks. GraphRAG methods address this by leveraging knowledge graphs (KGs) to capture structural associations, but they tend to overly rely on structural information and fine-grained word- or phrase-level retrieval, resulting in an underutilization of textual semantics. In this paper, we propose a novel RAG approach called HGRAG for MHQA that achieves cross-granularity integration of structural and semantic information via hypergraphs. Structurally, we construct an entity hypergraph where fine-grained entities serve as nodes and coarse-grained passages as hyperedges, and establish knowledge association through shared entities. Semantically, we design a hypergraph retrieval method that integrates fine-grained entity similarity and coarse-grained passage similarity via hypergraph diffusion. Finally, we employ a retrieval enhancement module, which further refines the retrieved results both semantically and structurally, to obtain the most relevant passages as context for answer generation with the LLM. Experimental results on benchmark datasets demonstrate that our approach outperforms state-of-the-art methods in QA performance, and achieves a 6$\times$ speedup in retrieval efficiency. 

---
# MobQA: A Benchmark Dataset for Semantic Understanding of Human Mobility Data through Question Answering 

**Authors**: Hikaru Asano, Hiroki Ouchi, Akira Kasuga, Ryo Yonetani  

**Link**: [PDF](https://arxiv.org/pdf/2508.11163)  

**Abstract**: This paper presents MobQA, a benchmark dataset designed to evaluate the semantic understanding capabilities of large language models (LLMs) for human mobility data through natural language question answering.
While existing models excel at predicting human movement patterns, it remains unobvious how much they can interpret the underlying reasons or semantic meaning of those patterns. MobQA provides a comprehensive evaluation framework for LLMs to answer questions about diverse human GPS trajectories spanning daily to weekly granularities. It comprises 5,800 high-quality question-answer pairs across three complementary question types: factual retrieval (precise data extraction), multiple-choice reasoning (semantic inference), and free-form explanation (interpretive description), which all require spatial, temporal, and semantic reasoning. Our evaluation of major LLMs reveals strong performance on factual retrieval but significant limitations in semantic reasoning and explanation question answering, with trajectory length substantially impacting model effectiveness. These findings demonstrate the achievements and limitations of state-of-the-art LLMs for semantic mobility understanding.\footnote{MobQA dataset is available at this https URL.} 

---
# Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics 

**Authors**: Carter Blum, Katja Filipova, Ann Yuan, Asma Ghandeharioun, Julian Zimmert, Fred Zhang, Jessica Hoffmann, Tal Linzen, Martin Wattenberg, Lucas Dixon, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2508.11017)  

**Abstract**: Large language models (LLMs) struggle with cross-lingual knowledge transfer: they hallucinate when asked in one language about facts expressed in a different language during training. This work introduces a controlled setting to study the causes and dynamics of this phenomenon by training small Transformer models from scratch on synthetic multilingual datasets. We identify a learning phase wherein a model develops either separate or unified representations of the same facts across languages, and show that unification is essential for cross-lingual transfer. We also show that the degree of unification depends on mutual information between facts and training data language, and on how easy it is to extract that language. Based on these insights, we develop methods to modulate the level of cross-lingual transfer by manipulating data distribution and tokenization, and we introduce metrics and visualizations to formally characterize their effects on unification. Our work shows how controlled settings can shed light on pre-training dynamics and suggests new directions for improving cross-lingual transfer in LLMs. 

---
# MoNaCo: More Natural and Complex Questions for Reasoning Across Dozens of Documents 

**Authors**: Tomer Wolfson, Harsh Trivedi, Mor Geva, Yoav Goldberg, Dan Roth, Tushar Khot, Ashish Sabharwal, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2508.11133)  

**Abstract**: Large language models (LLMs) are emerging as a go-to tool for querying information. However, current LLM benchmarks rarely feature natural questions that are both information-seeking as well as genuinely time-consuming for humans. To address this gap we introduce MoNaCo, a benchmark of 1,315 natural and complex questions that require dozens, and at times hundreds, of intermediate steps to solve -- far more than any existing QA benchmark. To build MoNaCo, we developed a decomposed annotation pipeline to elicit and manually answer natural time-consuming questions at scale. Frontier LLMs evaluated on MoNaCo achieve at most 61.2% F1, hampered by low recall and hallucinations. Our results underscore the need for reasoning models that better handle the complexity and sheer breadth of real-world information-seeking questions -- with MoNaCo providing an effective resource for tracking such progress. The MONACO benchmark, codebase, prompts and models predictions are publicly available at: this https URL 

---
# Towards Reliable Multi-Agent Systems for Marketing Applications via Reflection, Memory, and Planning 

**Authors**: Lorenzo Jaime Yu Flores, Junyi Shen, Xiaoyuan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11120)  

**Abstract**: Recent advances in large language models (LLMs) enabled the development of AI agents that can plan and interact with tools to complete complex tasks. However, literature on their reliability in real-world applications remains limited. In this paper, we introduce a multi-agent framework for a marketing task: audience curation. To solve this, we introduce a framework called RAMP that iteratively plans, calls tools, verifies the output, and generates suggestions to improve the quality of the audience generated. Additionally, we equip the model with a long-term memory store, which is a knowledge base of client-specific facts and past queries. Overall, we demonstrate the use of LLM planning and memory, which increases accuracy by 28 percentage points on a set of 88 evaluation queries. Moreover, we show the impact of iterative verification and reflection on more ambiguous queries, showing progressively better recall (roughly +20 percentage points) with more verify/reflect iterations on a smaller challenge set, and higher user satisfaction. Our results provide practical insights for deploying reliable LLM-based systems in dynamic, industry-facing environments. 

---
# Rule2Text: A Framework for Generating and Evaluating Natural Language Explanations of Knowledge Graph Rules 

**Authors**: Nasim Shirvani-Mahdavi, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10971)  

**Abstract**: Knowledge graphs (KGs) can be enhanced through rule mining; however, the resulting logical rules are often difficult for humans to interpret due to their inherent complexity and the idiosyncratic labeling conventions of individual KGs. This work presents Rule2Text, a comprehensive framework that leverages large language models (LLMs) to generate natural language explanations for mined logical rules, thereby improving KG accessibility and usability. We conduct extensive experiments using multiple datasets, including Freebase variants (FB-CVT-REV, FB+CVT-REV, and FB15k-237) as well as the ogbl-biokg dataset, with rules mined using AMIE 3.5.1. We systematically evaluate several LLMs across a comprehensive range of prompting strategies, including zero-shot, few-shot, variable type incorporation, and Chain-of-Thought reasoning. To systematically assess models' performance, we conduct a human evaluation of generated explanations on correctness and clarity. To address evaluation scalability, we develop and validate an LLM-as-a-judge framework that demonstrates strong agreement with human evaluators. Leveraging the best-performing model (Gemini 2.0 Flash), LLM judge, and human-in-the-loop feedback, we construct high-quality ground truth datasets, which we use to fine-tune the open-source Zephyr model. Our results demonstrate significant improvements in explanation quality after fine-tuning, with particularly strong gains in the domain-specific dataset. Additionally, we integrate a type inference module to support KGs lacking explicit type information. All code and data are publicly available at this https URL. 

---
# Controlling Multimodal LLMs via Reward-guided Decoding 

**Authors**: Oscar Mañas, Pierluca D'Oro, Koustuv Sinha, Adriana Romero-Soriano, Michal Drozdzal, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2508.11616)  

**Abstract**: As Multimodal Large Language Models (MLLMs) gain widespread applicability, it is becoming increasingly desirable to adapt them for diverse user needs. In this paper, we study the adaptation of MLLMs through controlled decoding. To achieve this, we introduce the first method for reward-guided decoding of MLLMs and demonstrate its application in improving their visual grounding. Our method involves building reward models for visual grounding and using them to guide the MLLM's decoding process. Concretely, we build two separate reward models to independently control the degree of object precision and recall in the model's output. Our approach enables on-the-fly controllability of an MLLM's inference process in two ways: first, by giving control over the relative importance of each reward function during decoding, allowing a user to dynamically trade off object precision for recall in image captioning tasks; second, by giving control over the breadth of the search during decoding, allowing the user to control the trade-off between the amount of test-time compute and the degree of visual grounding. We evaluate our method on standard object hallucination benchmarks, showing that it provides significant controllability over MLLM inference, while consistently outperforming existing hallucination mitigation methods. 

---
# PersonaTwin: A Multi-Tier Prompt Conditioning Framework for Generating and Evaluating Personalized Digital Twins 

**Authors**: Sihan Chen, John P. Lalor, Yi Yang, Ahmed Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10906)  

**Abstract**: While large language models (LLMs) afford new possibilities for user modeling and approximation of human behaviors, they often fail to capture the multidimensional nuances of individual users. In this work, we introduce PersonaTwin, a multi-tier prompt conditioning framework that builds adaptive digital twins by integrating demographic, behavioral, and psychometric data. Using a comprehensive data set in the healthcare context of more than 8,500 individuals, we systematically benchmark PersonaTwin against standard LLM outputs, and our rigorous evaluation unites state-of-the-art text similarity metrics with dedicated demographic parity assessments, ensuring that generated responses remain accurate and unbiased. Experimental results show that our framework produces simulation fidelity on par with oracle settings. Moreover, downstream models trained on persona-twins approximate models trained on individuals in terms of prediction and fairness metrics across both GPT-4o-based and Llama-based models. Together, these findings underscore the potential for LLM digital twin-based approaches in producing realistic and emotionally nuanced user simulations, offering a powerful tool for personalized digital user modeling and behavior analysis. 

---
# SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis 

**Authors**: Haitong Luo, Weiyao Zhang, Suhang Wang, Wenji Zou, Chungang Lin, Xuying Meng, Yujun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11343)  

**Abstract**: The proliferation of high-quality text from Large Language Models (LLMs) demands reliable and efficient detection methods. While existing training-free approaches show promise, they often rely on surface-level statistics and overlook fundamental signal properties of the text generation process. In this work, we reframe detection as a signal processing problem, introducing a novel paradigm that analyzes the sequence of token log-probabilities in the frequency domain. By systematically analyzing the signal's spectral properties using the global Discrete Fourier Transform (DFT) and the local Short-Time Fourier Transform (STFT), we find that human-written text consistently exhibits significantly higher spectral energy. This higher energy reflects the larger-amplitude fluctuations inherent in human writing compared to the suppressed dynamics of LLM-generated text. Based on this key insight, we construct SpecDetect, a detector built on a single, robust feature from the global DFT: DFT total energy. We also propose an enhanced version, SpecDetect++, which incorporates a sampling discrepancy mechanism to further boost robustness. Extensive experiments demonstrate that our approach outperforms the state-of-the-art model while running in nearly half the time. Our work introduces a new, efficient, and interpretable pathway for LLM-generated text detection, showing that classical signal processing techniques offer a surprisingly powerful solution to this modern challenge. 

---
# Retrieval-augmented reasoning with lean language models 

**Authors**: Ryan Sze-Yin Chan, Federico Nanni, Tomas Lazauskas, Rosie Wood, Penelope Yong, Lionel Tarassenko, Mark Girolami, James Geddes, Andrew Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11386)  

**Abstract**: This technical report details a novel approach to combining reasoning and retrieval augmented generation (RAG) within a single, lean language model architecture. While existing RAG systems typically rely on large-scale models and external APIs, our work addresses the increasing demand for performant and privacy-preserving solutions deployable in resource-constrained or secure environments. Building on recent developments in test-time scaling and small-scale reasoning models, we develop a retrieval augmented conversational agent capable of interpreting complex, domain-specific queries using a lightweight backbone model. Our system integrates a dense retriever with fine-tuned Qwen2.5-Instruct models, using synthetic query generation and reasoning traces derived from frontier models (e.g., DeepSeek-R1) over a curated corpus, in this case, the NHS A-to-Z condition pages. We explore the impact of summarisation-based document compression, synthetic data design, and reasoning-aware fine-tuning on model performance. Evaluation against both non-reasoning and general-purpose lean models demonstrates that our domain-specific fine-tuning approach yields substantial gains in answer accuracy and consistency, approaching frontier-level performance while remaining feasible for local deployment. All implementation details and code are publicly released to support reproducibility and adaptation across domains. 

---
# Group Fairness Meets the Black Box: Enabling Fair Algorithms on Closed LLMs via Post-Processing 

**Authors**: Ruicheng Xian, Yuxuan Wan, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11258)  

**Abstract**: Instruction fine-tuned large language models (LLMs) enable a simple zero-shot or few-shot prompting paradigm, also known as in-context learning, for building prediction models. This convenience, combined with continued advances in LLM capability, has the potential to drive their adoption across a broad range of domains, including high-stakes applications where group fairness -- preventing disparate impacts across demographic groups -- is essential. The majority of existing approaches to enforcing group fairness on LLM-based classifiers rely on traditional fair algorithms applied via model fine-tuning or head-tuning on final-layer embeddings, but they are no longer applicable to closed-weight LLMs under the in-context learning setting, which include some of the most capable commercial models today, such as GPT-4, Gemini, and Claude. In this paper, we propose a framework for deriving fair classifiers from closed-weight LLMs via prompting: the LLM is treated as a feature extractor, and features are elicited from its probabilistic predictions (e.g., token log probabilities) using prompts strategically designed for the specified fairness criterion to obtain sufficient statistics for fair classification; a fair algorithm is then applied to these features to train a lightweight fair classifier in a post-hoc manner. Experiments on five datasets, including three tabular ones, demonstrate strong accuracy-fairness tradeoffs for the classifiers derived by our framework from both open-weight and closed-weight LLMs; in particular, our framework is data-efficient and outperforms fair classifiers trained on LLM embeddings (i.e., head-tuning) or from scratch on raw tabular features. 

---
# ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal 

**Authors**: Haonan Zhang, Dongxia Wang, Yi Liu, Kexin Chen, Jiashui Wang, Xinlei Ying, Long Liu, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11222)  

**Abstract**: Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems. 

---
# Beyond Solving Math Quiz: Evaluating the Ability of Large Reasoning Models to Ask for Information 

**Authors**: Youcheng Huang, Bowen Qin, Chen Huang, Duanyu Feng, Xi Yang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.11252)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable problem-solving abilities in mathematics, as evaluated by existing benchmarks exclusively on well-defined problems. However, such evaluation setup constitutes a critical gap, since a genuine intelligent agent should not only solve problems (as a math quiz solver), but also be able~to ask for information when the problems lack sufficient information, enabling proactivity in responding users' requests. To bridge such gap, we proposes a new dataset consisting of two types of incomplete problems with diverse contexts. Based on the dataset, our systematical evaluation of LRMs reveals their inability in proactively asking for information. In addition, we uncover the behaviors related to overthinking and hallucination of LRMs, and highlight the potential and challenges of supervised fine-tuning in learning such ability. We hope to provide new insights in developing LRMs with genuine intelligence, rather than just solving problems. 

---
# Can Multi-modal (reasoning) LLMs detect document manipulation? 

**Authors**: Zisheng Liang, Kidus Zewde, Rudra Pratap Singh, Disha Patil, Zexi Chen, Jiayu Xue, Yao Yao, Yifei Chen, Qinzhe Liu, Simiao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.11021)  

**Abstract**: Document fraud poses a significant threat to industries reliant on secure and verifiable documentation, necessitating robust detection mechanisms. This study investigates the efficacy of state-of-the-art multi-modal large language models (LLMs)-including OpenAI O1, OpenAI 4o, Gemini Flash (thinking), Deepseek Janus, Grok, Llama 3.2 and 4, Qwen 2 and 2.5 VL, Mistral Pixtral, and Claude 3.5 and 3.7 Sonnet-in detecting fraudulent documents. We benchmark these models against each other and prior work on document fraud detection techniques using a standard dataset with real transactional documents. Through prompt optimization and detailed analysis of the models' reasoning processes, we evaluate their ability to identify subtle indicators of fraud, such as tampered text, misaligned formatting, and inconsistent transactional sums. Our results reveal that top-performing multi-modal LLMs demonstrate superior zero-shot generalization, outperforming conventional methods on out-of-distribution datasets, while several vision LLMs exhibit inconsistent or subpar performance. Notably, model size and advanced reasoning capabilities show limited correlation with detection accuracy, suggesting task-specific fine-tuning is critical. This study underscores the potential of multi-modal LLMs in enhancing document fraud detection systems and provides a foundation for future research into interpretable and scalable fraud mitigation strategies. 

---
# Empowering Multimodal LLMs with External Tools: A Comprehensive Survey 

**Authors**: Wenbin An, Jiahao Nie, Yaqiang Wu, Feng Tian, Shijian Lu, Qinghua Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10955)  

**Abstract**: By integrating the perception capabilities of multimodal encoders with the generative power of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), exemplified by GPT-4V, have achieved great success in various multimodal tasks, pointing toward a promising pathway to artificial general intelligence. Despite this progress, the limited quality of multimodal data, poor performance on many complex downstream tasks, and inadequate evaluation protocols continue to hinder the reliability and broader applicability of MLLMs across diverse domains. Inspired by the human ability to leverage external tools for enhanced reasoning and problem-solving, augmenting MLLMs with external tools (e.g., APIs, expert models, and knowledge bases) offers a promising strategy to overcome these challenges. In this paper, we present a comprehensive survey on leveraging external tools to enhance MLLM performance. Our discussion is structured along four key dimensions about external tools: (1) how they can facilitate the acquisition and annotation of high-quality multimodal data; (2) how they can assist in improving MLLM performance on challenging downstream tasks; (3) how they enable comprehensive and accurate evaluation of MLLMs; (4) the current limitations and future directions of tool-augmented MLLMs. Through this survey, we aim to underscore the transformative potential of external tools in advancing MLLM capabilities, offering a forward-looking perspective on their development and applications. The project page of this paper is publicly available athttps://github.com/Lackel/Awesome-Tools-for-MLLMs. 

---
# BIPOLAR: Polarization-based granular framework for LLM bias evaluation 

**Authors**: Martin Pavlíček, Tomáš Filip, Petr Sosík  

**Link**: [PDF](https://arxiv.org/pdf/2508.11061)  

**Abstract**: Large language models (LLMs) are known to exhibit biases in downstream tasks, especially when dealing with sensitive topics such as political discourse, gender identity, ethnic relations, or national stereotypes. Although significant progress has been made in bias detection and mitigation techniques, certain challenges remain underexplored. This study proposes a reusable, granular, and topic-agnostic framework to evaluate polarisation-related biases in LLM (both open-source and closed-source). Our approach combines polarisation-sensitive sentiment metrics with a synthetically generated balanced dataset of conflict-related statements, using a predefined set of semantic categories.
As a case study, we created a synthetic dataset that focusses on the Russia-Ukraine war, and we evaluated the bias in several LLMs: Llama-3, Mistral, GPT-4, Claude 3.5, and Gemini 1.0. Beyond aggregate bias scores, with a general trend for more positive sentiment toward Ukraine, the framework allowed fine-grained analysis with considerable variation between semantic categories, uncovering divergent behavioural patterns among models. Adaptation to prompt modifications showed further bias towards preconceived language and citizenship modification.
Overall, the framework supports automated dataset generation and fine-grained bias assessment, is applicable to a variety of polarisation-driven scenarios and topics, and is orthogonal to many other bias-evaluation strategies. 

---
# Inclusion Arena: An Open Platform for Evaluating Large Foundation Models with Real-World Apps 

**Authors**: Kangyu Wang, Hongliang He, Lin Liu, Ruiqi Liang, Zhenzhong Lan, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11452)  

**Abstract**: Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have ushered in a new era of AI capabilities, demonstrating near-human-level performance across diverse scenarios. While numerous benchmarks (e.g., MMLU) and leaderboards (e.g., Chatbot Arena) have been proposed to help evolve the development of LLMs and MLLMs, most rely on static datasets or crowdsourced general-domain prompts, often falling short of reflecting performance in real-world applications. To bridge this critical gap, we present Inclusion Arena, a live leaderboard that ranks models based on human feedback collected directly from AI-powered applications. Our platform integrates pairwise model comparisons into natural user interactions, ensuring evaluations reflect practical usage scenarios. For robust model ranking, we employ the Bradley-Terry model augmented with two key innovations: (1) Placement Matches, a cold-start mechanism to quickly estimate initial ratings for newly integrated models, and (2) Proximity Sampling, an intelligent comparison strategy that prioritizes battles between models of similar capabilities to maximize information gain and enhance rating stability. Extensive empirical analyses and simulations demonstrate that Inclusion Arena yields reliable and stable rankings, exhibits higher data transitivity compared to general crowdsourced datasets, and significantly mitigates the risk of malicious manipulation. By fostering an open alliance between foundation models and real-world applications, Inclusion Arena aims to accelerate the development of LLMs and MLLMs truly optimized for practical, user-centric deployments. The platform is publicly accessible at this https URL. 

---
# A2HCoder: An LLM-Driven Coding Agent for Hierarchical Algorithm-to-HDL Translation 

**Authors**: Jie Lei, Ruofan Jia, J. Andrew Zhang, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10904)  

**Abstract**: In wireless communication systems, stringent requirements such as ultra-low latency and power consumption have significantly increased the demand for efficient algorithm-to-hardware deployment. However, a persistent and substantial gap remains between algorithm design and hardware implementation. Bridging this gap traditionally requires extensive domain expertise and time-consuming manual development, due to fundamental mismatches between high-level programming languages like MATLAB and hardware description languages (HDLs) such as Verilog-in terms of memory access patterns, data processing manners, and datatype representations. To address this challenge, we propose A2HCoder: a Hierarchical Algorithm-to-HDL Coding Agent, powered by large language models (LLMs), designed to enable agile and reliable algorithm-to-hardware translation. A2HCoder introduces a hierarchical framework that enhances both robustness and interpretability while suppressing common hallucination issues in LLM-generated code. In the horizontal dimension, A2HCoder decomposes complex algorithms into modular functional blocks, simplifying code generation and improving consistency. In the vertical dimension, instead of relying on end-to-end generation, A2HCoder performs step-by-step, fine-grained translation, leveraging external toolchains such as MATLAB and Vitis HLS for debugging and circuit-level synthesis. This structured process significantly mitigates hallucinations and ensures hardware-level correctness. We validate A2HCoder through a real-world deployment case in the 5G wireless communication domain, demonstrating its practicality, reliability, and deployment efficiency. 

---
# Inspire or Predict? Exploring New Paradigms in Assisting Classical Planners with Large Language Models 

**Authors**: Wenkai Yu, Jianhang Tang, Yang Zhang, Shanjiang Tang, Kebing Jin, Hankz Hankui Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.11524)  

**Abstract**: Addressing large-scale planning problems has become one of the central challenges in the planning community, deriving from the state-space explosion caused by growing objects and actions. Recently, researchers have explored the effectiveness of leveraging Large Language Models (LLMs) to generate helpful actions and states to prune the search space. However, prior works have largely overlooked integrating LLMs with domain-specific knowledge to ensure valid plans. In this paper, we propose a novel LLM-assisted planner integrated with problem decomposition, which first decomposes large planning problems into multiple simpler sub-tasks. Then we explore two novel paradigms to utilize LLMs, i.e., LLM4Inspire and LLM4Predict, to assist problem decomposition, where LLM4Inspire provides heuristic guidance according to general knowledge and LLM4Predict employs domain-specific knowledge to infer intermediate conditions. We empirically validate the effectiveness of our planner across multiple domains, demonstrating the ability of search space partition when solving large-scale planning problems. The experimental results show that LLMs effectively locate feasible solutions when pruning the search space, where infusing domain-specific knowledge into LLMs, i.e., LLM4Predict, holds particular promise compared with LLM4Inspire, which offers general knowledge within LLMs. 

---
# CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection 

**Authors**: Zhihao Li, Zimo Ji, Tao Zheng, Hao Ren, Xiao Lan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11599)  

**Abstract**: Cryptographic algorithms are fundamental to modern security, yet their implementations frequently harbor subtle logic flaws that are hard to detect. We introduce CryptoScope, a novel framework for automated cryptographic vulnerability detection powered by Large Language Models (LLMs). CryptoScope combines Chain-of-Thought (CoT) prompting with Retrieval-Augmented Generation (RAG), guided by a curated cryptographic knowledge base containing over 12,000 entries. We evaluate CryptoScope on LLM-CLVA, a benchmark of 92 cases primarily derived from real-world CVE vulnerabilities, complemented by cryptographic challenges from major Capture The Flag (CTF) competitions and synthetic examples across 11 programming languages. CryptoScope consistently improves performance over strong LLM baselines, boosting DeepSeek-V3 by 11.62%, GPT-4o-mini by 20.28%, and GLM-4-Flash by 28.69%. Additionally, it identifies 9 previously undisclosed flaws in widely used open-source cryptographic projects. 

---
# AIM-Bench: Evaluating Decision-making Biases of Agentic LLM as Inventory Manager 

**Authors**: Xuhua Zhao, Yuxuan Xie, Caihua Chen, Yuxiang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.11416)  

**Abstract**: Recent advances in mathematical reasoning and the long-term planning capabilities of large language models (LLMs) have precipitated the development of agents, which are being increasingly leveraged in business operations processes. Decision models to optimize inventory levels are one of the core elements of operations management. However, the capabilities of the LLM agent in making inventory decisions in uncertain contexts, as well as the decision-making biases (e.g. framing effect, etc.) of the agent, remain largely unexplored. This prompts concerns regarding the capacity of LLM agents to effectively address real-world problems, as well as the potential implications of biases that may be present. To address this gap, we introduce AIM-Bench, a novel benchmark designed to assess the decision-making behaviour of LLM agents in uncertain supply chain management scenarios through a diverse series of inventory replenishment experiments. Our results reveal that different LLMs typically exhibit varying degrees of decision bias that are similar to those observed in human beings. In addition, we explored strategies to mitigate the pull-to-centre effect and the bullwhip effect, namely cognitive reflection and implementation of information sharing. These findings underscore the need for careful consideration of the potential biases in deploying LLMs in Inventory decision-making scenarios. We hope that these insights will pave the way for mitigating human decision bias and developing human-centred decision support systems for supply chains. 

---
# Is ChatGPT-5 Ready for Mammogram VQA? 

**Authors**: Qiang Li, Shansong Wang, Mingzhe Hu, Mojtaba Safari, Zachary Eidex, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11628)  

**Abstract**: Mammogram visual question answering (VQA) integrates image interpretation with clinical reasoning and has potential to support breast cancer screening. We systematically evaluated the GPT-5 family and GPT-4o model on four public mammography datasets (EMBED, InBreast, CMMD, CBIS-DDSM) for BI-RADS assessment, abnormality detection, and malignancy classification tasks. GPT-5 consistently was the best performing model but lagged behind both human experts and domain-specific fine-tuned models. On EMBED, GPT-5 achieved the highest scores among GPT variants in density (56.8%), distortion (52.5%), mass (64.5%), calcification (63.5%), and malignancy (52.8%) classification. On InBreast, it attained 36.9% BI-RADS accuracy, 45.9% abnormality detection, and 35.0% malignancy classification. On CMMD, GPT-5 reached 32.3% abnormality detection and 55.0% malignancy accuracy. On CBIS-DDSM, it achieved 69.3% BI-RADS accuracy, 66.0% abnormality detection, and 58.2% malignancy accuracy. Compared with human expert estimations, GPT-5 exhibited lower sensitivity (63.5%) and specificity (52.3%). While GPT-5 exhibits promising capabilities for screening tasks, its performance remains insufficient for high-stakes clinical imaging applications without targeted domain adaptation and optimization. However, the tremendous improvements in performance from GPT-4o to GPT-5 show a promising trend in the potential for general large language models (LLMs) to assist with mammography VQA tasks. 

---
# Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis 

**Authors**: Mithat Can Ozgun, Jiahuan Pei, Koen Hindriks, Lucia Donatelli, Qingzhi Liu, Xin Sun, Junxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11398)  

**Abstract**: LLM-based agents have emerged as transformative tools capable of executing complex tasks through iterative planning and action, achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced. 

---
# ETTRL: Balancing Exploration and Exploitation in LLM Test-Time Reinforcement Learning Via Entropy Mechanism 

**Authors**: Jia Liu, ChangYi He, YingQiao Lin, MingMin Yang, FeiYang Shen, ShaoGuo Liu, TingTing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11356)  

**Abstract**: Recent advancements in Large Language Models have yielded significant improvements in complex reasoning tasks such as mathematics and programming. However, these models remain heavily dependent on annotated data and exhibit limited adaptability in unsupervised scenarios. To address these limitations, test-time reinforcement learning (TTRL) has been proposed, which enables self-optimization by leveraging model-generated pseudo-labels. Despite its promise, TTRL faces several key challenges, including high inference costs due to parallel rollouts and early-stage estimation bias that fosters overconfidence, reducing output diversity and causing performance plateaus. To address these challenges, we introduce an entropy-based mechanism to enhance the exploration-exploitation balance in test-time reinforcement learning through two strategies: Entropy-fork Tree Majority Rollout (ETMR) and Entropy-based Advantage Reshaping (EAR). Compared with the baseline, our approach enables Llama3.1-8B to achieve a 68 percent relative improvement in Pass at 1 metric on the AIME 2024 benchmark, while consuming only 60 percent of the rollout tokens budget. This highlights our method's ability to effectively optimize the trade-off between inference efficiency, diversity, and estimation robustness, thereby advancing unsupervised reinforcement learning for open-domain reasoning tasks. 

---
# AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions 

**Authors**: Tianjiao Zhao, Jingrao Lyu, Stokes Jones, Harrison Garber, Stefano Pasquali, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2508.11152)  

**Abstract**: The field of artificial intelligence (AI) agents is evolving rapidly, driven by the capabilities of Large Language Models (LLMs) to autonomously perform and refine tasks with human-like efficiency and adaptability. In this context, multi-agent collaboration has emerged as a promising approach, enabling multiple AI agents to work together to solve complex challenges. This study investigates the application of role-based multi-agent systems to support stock selection in equity research and portfolio management. We present a comprehensive analysis performed by a team of specialized agents and evaluate their stock-picking performance against established benchmarks under varying levels of risk tolerance. Furthermore, we examine the advantages and limitations of employing multi-agent frameworks in equity analysis, offering critical insights into their practical efficacy and implementation challenges. 

---
# Hallucination in LLM-Based Code Generation: An Automotive Case Study 

**Authors**: Marc Pavel, Nenad Petrovic, Lukasz Mazur, Vahid Zolfaghari, Fengjunjie Pan, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.11257)  

**Abstract**: Large Language Models (LLMs) have shown significant potential in automating code generation tasks offering new opportunities across software engineering domains. However, their practical application remains limited due to hallucinations - outputs that appear plausible but are factually incorrect, unverifiable or nonsensical. This paper investigates hallucination phenomena in the context of code generation with a specific focus on the automotive domain. A case study is presented that evaluates multiple code LLMs for three different prompting complexities ranging from a minimal one-liner prompt to a prompt with Covesa Vehicle Signal Specifications (VSS) as additional context and finally to a prompt with an additional code skeleton. The evaluation reveals a high frequency of syntax violations, invalid reference errors and API knowledge conflicts in state-of-the-art models GPT-4.1, Codex and GPT-4o. Among the evaluated models, only GPT-4.1 and GPT-4o were able to produce a correct solution when given the most context-rich prompt. Simpler prompting strategies failed to yield a working result, even after multiple refinement iterations. These findings highlight the need for effective mitigation techniques to ensure the safe and reliable use of LLM generated code, especially in safety-critical domains such as automotive software systems. 

---
# AI That Helps Us Help Each Other: A Proactive System for Scaffolding Mentor-Novice Collaboration in Entrepreneurship Coaching 

**Authors**: Evey Jiaxin Huang, Matthew Easterday, Elizabeth Gerber  

**Link**: [PDF](https://arxiv.org/pdf/2508.11052)  

**Abstract**: Entrepreneurship requires navigating open-ended, ill-defined problems: identifying risks, challenging assumptions, and making strategic decisions under deep uncertainty. Novice founders often struggle with these metacognitive demands, while mentors face limited time and visibility to provide tailored support. We present a human-AI coaching system that combines a domain-specific cognitive model of entrepreneurial risk with a large language model (LLM) to proactively scaffold both novice and mentor thinking. The system proactively poses diagnostic questions that challenge novices' thinking and helps both novices and mentors plan for more focused and emotionally attuned meetings. Critically, mentors can inspect and modify the underlying cognitive model, shaping the logic of the system to reflect their evolving needs. Through an exploratory field deployment, we found that using the system supported novice metacognition, helped mentors plan emotionally attuned strategies, and improved meeting depth, intentionality, and focus--while also surfaced key tensions around trust, misdiagnosis, and expectations of AI. We contribute design principles for proactive AI systems that scaffold metacognition and human-human collaboration in complex, ill-defined domains, offering implications for similar domains like healthcare, education, and knowledge work. 

---
# ADMIRE-BayesOpt: Accelerated Data MIxture RE-weighting for Language Models with Bayesian Optimization 

**Authors**: Shengzhuang Chen, Xu Ouyang, Michael Arthur Leopold Pearce, Thomas Hartvigsen, Jonathan Richard Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11551)  

**Abstract**: Determining the optimal data mixture for large language model training remains a challenging problem with an outsized impact on performance. In practice, language model developers continue to rely on heuristic exploration since no learning-based approach has emerged as a reliable solution. In this work, we propose to view the selection of training data mixtures as a black-box hyperparameter optimization problem, for which Bayesian Optimization is a well-established class of appropriate algorithms. Firstly, we cast data mixture learning as a sequential decision-making problem, in which we aim to find a suitable trade-off between the computational cost of training exploratory (proxy-) models and final mixture performance. Secondly, we systematically explore the properties of transferring mixtures learned at a small scale to larger-scale experiments, providing insights and highlighting opportunities for research at a modest scale. By proposing Multi-fidelity Bayesian Optimization as a suitable method in this common scenario, we introduce a natural framework to balance experiment cost with model fit, avoiding the risks of overfitting to smaller scales while minimizing the number of experiments at high cost. We present results for pre-training and instruction finetuning across models ranging from 1 million to 7 billion parameters, varying from simple architectures to state-of-the-art models and benchmarks spanning dozens of datasets. We demonstrate consistently strong results relative to a wide range of benchmarks, showingspeed-ups of over 500% in determining the best data mixture on our largest experiments relative to recent baselines. In addition, we broaden access to research by sharing ADMIRE IFT Runs, a dataset of 460 full training & evaluation runs across various model sizes worth over 13,000 GPU hours, greatly reducing the cost of conducting research in this area. 

---
# SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth 

**Authors**: Wenpeng Xing, Lanyi Wei, Haixiao Hu, Rongchang Li, Mohan Li, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.11009)  

**Abstract**: The rapid proliferation of large language models (LLMs) in applications targeting children and adolescents necessitates a fundamental reassessment of prevailing AI safety frameworks, which are largely tailored to adult users and neglect the distinct developmental vulnerabilities of minors. This paper highlights key deficiencies in existing LLM safety benchmarks, including their inadequate coverage of age-specific cognitive, emotional, and social risks spanning early childhood (ages 0--6), middle childhood (7--12), and adolescence (13--18). To bridge these gaps, we introduce SproutBench, an innovative evaluation suite comprising 1,283 developmentally grounded adversarial prompts designed to probe risks such as emotional dependency, privacy violations, and imitation of hazardous behaviors. Through rigorous empirical evaluation of 47 diverse LLMs, we uncover substantial safety vulnerabilities, corroborated by robust inter-dimensional correlations (e.g., between Safety and Risk Prevention) and a notable inverse relationship between Interactivity and Age Appropriateness. These insights yield practical guidelines for advancing child-centric AI design and deployment. 

---
# Learn to optimize for automatic proton PBS treatment planning for H&N cancers 

**Authors**: Qingqing Wang, Liqiang Xiao, Chang Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11085)  

**Abstract**: Proton PBS treatment planning for H&N cancers involves numerous conflicting objectives, requiring significant effort from human planners to balance and satisfy multiple clinical goals during planning. To achieve this, experience-demanding objective parameter adjustment and computationally expensive inverse optimization are performed iteratively. Extensive efforts have been made to automatically adjust objective parameters, but the most time-consuming component, i.e., inverse optimization, still relies heavily on theory-driven approaches. We propose a data-driven inverse optimizer and integrate it into a PPO-based automatic treatment planning framework to automatically generate high-quality plans within a clinical acceptable planning time. The inverse optimizer is a L2O method that predicts update steps by learning from the task-specific data distribution. For the first time, we integrate techniques designed for long-context processing, originally developed for LLMs, into a Transformer-based L2O framework to address the scalability issue of existing L2O methods. The PPO framework functions as an outer-loop virtual planner, autonomously adjusting objective parameters through a policy network, and the dose predictor is used to initialize objective parameters. The inner-loop L2O inverse optimizer computes machine-deliverable MU values based on objectives refined by the PPO policy network. 97 patients are collected in this study, and compared with L-BFGSB, our L2O-based inverse optimizer improves the effectiveness and efficiency by 22.97% and 36.41%, respectively. In conjunction with the PPO-based learned virtual planner, plans generated by our framework within an average of 2.55 hours show improved or comparable OAR sparing with superior target coverage for patients with different prescription dose levels, number of target volumes, beam angles, etc., compared with human-generated plans. 

---
# Role-Augmented Intent-Driven Generative Search Engine Optimization 

**Authors**: Xiaolu Chen, Haojie Wu, Jie Bao, Zhen Chen, Yong Liao, Hu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11158)  

**Abstract**: Generative Search Engines (GSEs), powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), are reshaping information retrieval. While commercial systems (e.g., BingChat, this http URL) demonstrate impressive semantic synthesis capabilities, their black-box nature fundamentally undermines established Search Engine Optimization (SEO) practices. Content creators face a critical challenge: their optimization strategies, effective in traditional search engines, are misaligned with generative retrieval contexts, resulting in diminished visibility. To bridge this gap, we propose a Role-Augmented Intent-Driven Generative Search Engine Optimization (G-SEO) method, providing a structured optimization pathway tailored for GSE scenarios. Our method models search intent through reflective refinement across diverse informational roles, enabling targeted content enhancement. To better evaluate the method under realistic settings, we address the benchmarking limitations of prior work by: (1) extending the GEO dataset with diversified query variations reflecting real-world search scenarios and (2) introducing G-Eval 2.0, a 6-level LLM-augmented evaluation rubric for fine-grained human-aligned assessment. Experimental results demonstrate that search intent serves as an effective signal for guiding content optimization, yielding significant improvements over single-aspect baseline approaches in both subjective impressions and objective content visibility within GSE responses. 

---
# Retro-Expert: Collaborative Reasoning for Interpretable Retrosynthesis 

**Authors**: Xinyi Li, Sai Wang, Yutian Lin, Yu Wu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10967)  

**Abstract**: Retrosynthesis prediction aims to infer the reactant molecule based on a given product molecule, which is a fundamental task in chemical synthesis. However, existing models rely on static pattern-matching paradigm, which limits their ability to perform effective logic decision-making, leading to black-box decision-making. Building on this, we propose Retro-Expert, an interpretable retrosynthesis framework that performs collaborative reasoning by combining the complementary reasoning strengths of Large Language Models and specialized models via reinforcement learning. It outputs natural language explanations grounded in chemical logic through three components: (1) specialized models perform shallow reasoning to construct high-quality chemical decision space, (2) LLM-driven critical reasoning to generate predictions and corresponding interpretable reasoning path, and (3) reinforcement learning optimizing interpretable decision policy. Experiments show that Retro-Expert not only surpasses both LLM-based and specialized models across different metrics but also provides expert-aligned explanations that bridge the gap between AI predictions and actionable chemical insights. 

---
# MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications 

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.10991)  

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems. 

---
# CURE: Critical-Token-Guided Re-concatenation for Entropy-collapse Prevention 

**Authors**: Qingbin Li, Rongkun Xue, Jie Wang, Ming Zhou, Zhi Li, Xiaofeng Ji, Yongqi Wang, Miao Liu, Zheming Yang, Minghui Qiu, Jing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11016)  

**Abstract**: Recent advances in Reinforcement Learning with Verified Reward (RLVR) have driven the emergence of more sophisticated cognitive behaviors in large language models (LLMs), thereby enhancing their reasoning capabilities. However, in prior RLVR pipelines, the repeated use of static initial-state sampling drawn exactly from the dataset distribution during each sampling phase produced overly deterministic, low diversity model behavior, which manifested as rapid entropy collapse and hindered sustained performance gains during prolonged training. To address this issue, we introduce CURE (Critical-token-gUided Re concatenation for Entropy-collapse prevention), a two-stage framework that balances exploration and exploitation. Specifically, in the first stage, to deliberately steer the model toward novel yet coherent contexts, we re-generate at high-entropy critical tokens and jointly optimize the original and the branched trajectories. The further comparison with vanilla DAPO shows that the regeneration process achieves a better performance on math reasoning tasks while sustaining a high-level entropy degree for exploration. In the second stage, we continue training with static initial-state sampling by DAPO, intentionally placing the model in a familiar state to gradually strengthen exploitation. Extensive experiments on Qwen-2.5-Math-7B show that, compared to other RLVR methods, CURE achieves a 5% performance gain across six math benchmarks, establishing state-of-the-art performance in both entropy and accuracy. A series of experiments further validate the effectiveness of our approach. Code is available at this https URL. 

---
