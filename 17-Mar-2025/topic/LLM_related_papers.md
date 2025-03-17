# Neutralizing Bias in LLM Reasoning using Entailment Graphs 

**Authors**: Liang Cheng, Tianyi Li, Zhaowei Wang, Tianyang Liu, Mark Steedman  

**Link**: [PDF](https://arxiv.org/pdf/2503.11614)  

**Abstract**: LLMs are often claimed to be capable of Natural Language Inference (NLI), which is widely regarded as a cornerstone of more complex forms of reasoning. However, recent works show that LLMs still suffer from hallucinations in NLI due to attestation bias, where LLMs overly rely on propositional memory to build shortcuts. To solve the issue, we design an unsupervised framework to construct counterfactual reasoning data and fine-tune LLMs to reduce attestation bias. To measure bias reduction, we build bias-adversarial variants of NLI datasets with randomly replaced predicates in premises while keeping hypotheses unchanged. Extensive evaluations show that our framework can significantly reduce hallucinations from attestation bias. Then, we further evaluate LLMs fine-tuned with our framework on original NLI datasets and their bias-neutralized versions, where original entities are replaced with randomly sampled ones. Extensive results show that our framework consistently improves inferential performance on both original and bias-neutralized NLI datasets. 

---
# AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation 

**Authors**: Fengyu Li, Yilin Li, Junhao Zhu, Lu Chen, Yanfei Zhang, Jia Zhou, Hui Zu, Jingwen Zhao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11346)  

**Abstract**: Huawei has always been committed to exploring the AI application in historical research. Biography generation, as a specialized form of abstractive summarization, plays a crucial role in historical research but faces unique challenges that existing large language models (LLMs) struggle to address. These challenges include maintaining stylistic adherence to historical writing conventions, ensuring factual fidelity, and handling fragmented information across multiple documents. We present AIstorian, a novel end-to-end agentic system featured with a knowledge graph (KG)-powered retrieval-augmented generation (RAG) and anti-hallucination multi-agents. Specifically, AIstorian introduces an in-context learning based chunking strategy and a KG-based index for accurate and efficient reference retrieval. Meanwhile, AIstorian orchestrates multi-agents to conduct on-the-fly hallucination detection and error-type-aware correction. Additionally, to teach LLMs a certain language style, we finetune LLMs based on a two-step training approach combining data augmentation-enhanced supervised fine-tuning with stylistic preference optimization. Extensive experiments on a real-life historical Jinshi dataset demonstrate that AIstorian achieves a 3.8x improvement in factual accuracy and a 47.6% reduction in hallucination rate compared to existing baselines. The data and code are available at: this https URL. 

---
# Rule-Guided Feedback: Enhancing Reasoning by Enforcing Rule Adherence in Large Language Models 

**Authors**: Aissatou Diallo, Antonis Bikakis, Luke Dickens, Anthony Hunter, Rob Miller  

**Link**: [PDF](https://arxiv.org/pdf/2503.11336)  

**Abstract**: In this paper, we introduce Rule-Guided Feedback (RGF), a framework designed to enhance Large Language Model (LLM) performance through structured rule adherence and strategic information seeking. RGF implements a teacher-student paradigm where rule-following is forced through established guidelines. Our framework employs a Teacher model that rigorously evaluates each student output against task-specific rules, providing constructive guidance rather than direct answers when detecting deviations. This iterative feedback loop serves two crucial purposes: maintaining solutions within defined constraints and encouraging proactive information seeking to resolve uncertainties. We evaluate RGF on diverse tasks including Checkmate-in-One puzzles, Sonnet Writing, Penguins-In-a-Table classification, GSM8k, and StrategyQA. Our findings suggest that structured feedback mechanisms can significantly enhance LLMs' performance across various domains. 

---
# Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering 

**Authors**: Xinyu Tang, Xiaolei Wang, Zhihao Lv, Yingqian Min, Wayne Xin Zhao, Binbin Hu, Ziqi Liu, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11314)  

**Abstract**: Recent advancements in long chain-of-thoughts(long CoTs) have significantly improved the reasoning capabilities of large language models(LLMs). Existing work finds that the capability of long CoT reasoning can be efficiently elicited by tuning on only a few examples and can easily transfer to other tasks. This motivates us to investigate whether long CoT reasoning is a general capability for LLMs. In this work, we conduct an empirical analysis for this question from the perspective of representation. We find that LLMs do encode long CoT reasoning as a general capability, with a clear distinction from vanilla CoTs. Furthermore, domain-specific representations are also required for the effective transfer of long CoT reasoning. Inspired by these findings, we propose GLoRE, a novel representation engineering method to unleash the general long CoT reasoning capabilities of LLMs. Extensive experiments demonstrate the effectiveness and efficiency of GLoRE in both in-domain and cross-domain scenarios. 

---
# Are formal and functional linguistic mechanisms dissociated? 

**Authors**: Michael Hanna, Sandro Pezzelle, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2503.11302)  

**Abstract**: Although large language models (LLMs) are increasingly capable, these capabilities are unevenly distributed: they excel at formal linguistic tasks, such as producing fluent, grammatical text, but struggle more with functional linguistic tasks like reasoning and consistent fact retrieval. Inspired by neuroscience, recent work suggests that to succeed on both formal and functional linguistic tasks, LLMs should use different mechanisms for each; such localization could either be built-in or emerge spontaneously through training. In this paper, we ask: do current models, with fast-improving functional linguistic abilities, exhibit distinct localization of formal and functional linguistic mechanisms? We answer this by finding and comparing the "circuits", or minimal computational subgraphs, responsible for various formal and functional tasks. Comparing 5 LLMs across 10 distinct tasks, we find that while there is indeed little overlap between circuits for formal and functional tasks, there is also little overlap between formal linguistic tasks, as exists in the human brain. Thus, a single formal linguistic network, unified and distinct from functional task circuits, remains elusive. However, in terms of cross-task faithfulness - the ability of one circuit to solve another's task - we observe a separation between formal and functional mechanisms, suggesting that shared mechanisms between formal tasks may exist. 

---
# Line of Duty: Evaluating LLM Self-Knowledge via Consistency in Feasibility Boundaries 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2503.11256)  

**Abstract**: As LLMs grow more powerful, their most profound achievement may be recognising when to say "I don't know". Existing studies on LLM self-knowledge have been largely constrained by human-defined notions of feasibility, often neglecting the reasons behind unanswerability by LLMs and failing to study deficient types of self-knowledge. This study aims to obtain intrinsic insights into different types of LLM self-knowledge with a novel methodology: allowing them the flexibility to set their own feasibility boundaries and then analysing the consistency of these limits. We find that even frontier models like GPT-4o and Mistral Large are not sure of their own capabilities more than 80% of the time, highlighting a significant lack of trustworthiness in responses. Our analysis of confidence balance in LLMs indicates that models swing between overconfidence and conservatism in feasibility boundaries depending on task categories and that the most significant self-knowledge weaknesses lie in temporal awareness and contextual understanding. These difficulties in contextual comprehension additionally lead models to question their operational boundaries, resulting in considerable confusion within the self-knowledge of LLMs. We make our code and results available publicly at this https URL 

---
# RESPONSE: Benchmarking the Ability of Language Models to Undertake Commonsense Reasoning in Crisis Situation 

**Authors**: Aissatou Diallo, Antonis Bikakis, Luke Dickens, Anthony Hunter, Rob Miller  

**Link**: [PDF](https://arxiv.org/pdf/2503.11348)  

**Abstract**: An interesting class of commonsense reasoning problems arises when people are faced with natural disasters. To investigate this topic, we present \textsf{RESPONSE}, a human-curated dataset containing 1789 annotated instances featuring 6037 sets of questions designed to assess LLMs' commonsense reasoning in disaster situations across different time frames. The dataset includes problem descriptions, missing resources, time-sensitive solutions, and their justifications, with a subset validated by environmental engineers. Through both automatic metrics and human evaluation, we compare LLM-generated recommendations against human responses. Our findings show that even state-of-the-art models like GPT-4 achieve only 37\% human-evaluated correctness for immediate response actions, highlighting significant room for improvement in LLMs' ability for commonsense reasoning in crises. 

---
# High-Dimensional Interlingual Representations of Large Language Models 

**Authors**: Bryan Wilie, Samuel Cahyawijaya, Junxian He, Pascale Fung  

**Link**: [PDF](https://arxiv.org/pdf/2503.11280)  

**Abstract**: Large language models (LLMs) trained on massive multilingual datasets hint at the formation of interlingual constructs--a shared subspace in the representation space. However, evidence regarding this phenomenon is mixed, leaving it unclear whether these models truly develop unified interlingual representations, or present a partially aligned constructs. We explore 31 diverse languages varying on their resource-levels, typologies, and geographical regions; and find that multilingual LLMs exhibit inconsistent cross-lingual alignments. To address this, we propose an interlingual representation framework identifying both the shared interlingual semantic subspace and fragmented components, existed due to representational limitations. We introduce Interlingual Local Overlap (ILO) score to quantify interlingual alignment by comparing the local neighborhood structures of high-dimensional representations. We utilize ILO to investigate the impact of single-language fine-tuning on the interlingual representations in multilingual LLMs. Our results indicate that training exclusively on a single language disrupts the alignment in early layers, while freezing these layers preserves the alignment of interlingual representations, leading to improved cross-lingual generalization. These results validate our framework and metric for evaluating interlingual representation, and further underscore that interlingual alignment is crucial for scalable multilingual learning. 

---
# Don't Take Things Out of Context: Attention Intervention for Enhancing Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Shaotian Yan, Chen Shen, Wenxiao Wang, Liang Xie, Junjie Liu, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.11154)  

**Abstract**: Few-shot Chain-of-Thought (CoT) significantly enhances the reasoning capabilities of large language models (LLMs), functioning as a whole to guide these models in generating reasoning steps toward final answers. However, we observe that isolated segments, words, or tokens within CoT demonstrations can unexpectedly disrupt the generation process of LLMs. The model may overly concentrate on certain local information present in the demonstration, introducing irrelevant noise into the reasoning process and potentially leading to incorrect answers. In this paper, we investigate the underlying mechanism of CoT through dynamically tracing and manipulating the inner workings of LLMs at each output step, which demonstrates that tokens exhibiting specific attention characteristics are more likely to induce the model to take things out of context; these tokens directly attend to the hidden states tied with prediction, without substantial integration of non-local information. Building upon these insights, we propose a Few-shot Attention Intervention method (FAI) that dynamically analyzes the attention patterns of demonstrations to accurately identify these tokens and subsequently make targeted adjustments to the attention weights to effectively suppress their distracting effect on LLMs. Comprehensive experiments across multiple benchmarks demonstrate consistent improvements over baseline methods, with a remarkable 5.91% improvement on the AQuA dataset, further highlighting the effectiveness of FAI. 

---
# RONA: Pragmatically Diverse Image Captioning with Coherence Relations 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10997)  

**Abstract**: Writing Assistants (e.g., Grammarly, Microsoft Copilot) traditionally generate diverse image captions by employing syntactic and semantic variations to describe image components. However, human-written captions prioritize conveying a central message alongside visual descriptions using pragmatic cues. To enhance pragmatic diversity, it is essential to explore alternative ways of communicating these messages in conjunction with visual content. To address this challenge, we propose RONA, a novel prompting strategy for Multi-modal Large Language Models (MLLM) that leverages Coherence Relations as an axis for variation. We demonstrate that RONA generates captions with better overall diversity and ground-truth alignment, compared to MLLM baselines across multiple domains. Our code is available at: this https URL 

---
# SCE: Scalable Consistency Ensembles Make Blackbox Large Language Model Generation More Reliable 

**Authors**: Jiaxin Zhang, Zhuohang Li, Wendi Cui, Kamalika Das, Bradley malin, Sricharan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.10881)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance, yet their diverse strengths and weaknesses prevent any single LLM from achieving dominance across all tasks. Ensembling multiple LLMs is a promising approach to generate reliable responses but conventional ensembling frameworks suffer from high computational overheads. This work introduces Scalable Consistency Ensemble (SCE), an efficient framework for ensembling LLMs by prompting consistent outputs. The SCE framework systematically evaluates and integrates outputs to produce a cohesive result through two core components: SCE-CHECK, a mechanism that gauges the consistency between response pairs via semantic equivalence; and SCE-FUSION, which adeptly merges the highest-ranked consistent responses from SCE-CHECK, to optimize collective strengths and mitigating potential weaknesses. To improve the scalability with multiple inference queries, we further propose ``{You Only Prompt Once}'' (YOPO), a novel technique that reduces the inference complexity of pairwise comparison from quadratic to constant time. We perform extensive empirical evaluations on diverse benchmark datasets to demonstrate \methodName's effectiveness. Notably, the \saccheckcomponent outperforms conventional baselines with enhanced performance and a significant reduction in computational overhead. 

---
# Who Relies More on World Knowledge and Bias for Syntactic Ambiguity Resolution: Humans or LLMs? 

**Authors**: So Young Lee, Russell Scheinberg, Amber Shore, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.10838)  

**Abstract**: This study explores how recent large language models (LLMs) navigate relative clause attachment {ambiguity} and use world knowledge biases for disambiguation in six typologically diverse languages: English, Chinese, Japanese, Korean, Russian, and Spanish. We describe the process of creating a novel dataset -- MultiWho -- for fine-grained evaluation of relative clause attachment preferences in ambiguous and unambiguous contexts. Our experiments with three LLMs indicate that, contrary to humans, LLMs consistently exhibit a preference for local attachment, displaying limited responsiveness to syntactic variations or language-specific attachment patterns. Although LLMs performed well in unambiguous cases, they rigidly prioritized world knowledge biases, lacking the flexibility of human language processing. These findings highlight the need for more diverse, pragmatically nuanced multilingual training to improve LLMs' handling of complex structures and human-like comprehension. 

---
# DarkBench: Benchmarking Dark Patterns in Large Language Models 

**Authors**: Esben Kran, Hieu Minh "Jord" Nguyen, Akash Kundu, Sami Jawhar, Jinsuk Park, Mateusz Maria Jurewicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.10728)  

**Abstract**: We introduce DarkBench, a comprehensive benchmark for detecting dark design patterns--manipulative techniques that influence user behavior--in interactions with large language models (LLMs). Our benchmark comprises 660 prompts across six categories: brand bias, user retention, sycophancy, anthropomorphism, harmful generation, and sneaking. We evaluate models from five leading companies (OpenAI, Anthropic, Meta, Mistral, Google) and find that some LLMs are explicitly designed to favor their developers' products and exhibit untruthful communication, among other manipulative behaviors. Companies developing LLMs should recognize and mitigate the impact of dark design patterns to promote more ethical AI. 

---
# Thinking Machines: A Survey of LLM based Reasoning Strategies 

**Authors**: Dibyanayan Bandyopadhyay, Soham Bhattacharjee, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2503.10814)  

**Abstract**: Large Language Models (LLMs) are highly proficient in language-based tasks. Their language capabilities have positioned them at the forefront of the future AGI (Artificial General Intelligence) race. However, on closer inspection, Valmeekam et al. (2024); Zecevic et al. (2023); Wu et al. (2024) highlight a significant gap between their language proficiency and reasoning abilities. Reasoning in LLMs and Vision Language Models (VLMs) aims to bridge this gap by enabling these models to think and re-evaluate their actions and responses. Reasoning is an essential capability for complex problem-solving and a necessary step toward establishing trust in Artificial Intelligence (AI). This will make AI suitable for deployment in sensitive domains, such as healthcare, banking, law, defense, security etc. In recent times, with the advent of powerful reasoning models like OpenAI O1 and DeepSeek R1, reasoning endowment has become a critical research topic in LLMs. In this paper, we provide a detailed overview and comparison of existing reasoning techniques and present a systematic survey of reasoning-imbued language models. We also study current challenges and present our findings. 

---
# Word-level Annotation of GDPR Transparency Compliance in Privacy Policies using Large Language Models 

**Authors**: Thomas Cory, Wolf Rieder, Julia Krämer, Philip Raschke, Patrick Herbke, Axel Küpper  

**Link**: [PDF](https://arxiv.org/pdf/2503.10727)  

**Abstract**: Ensuring transparency of data practices related to personal information is a fundamental requirement under the General Data Protection Regulation (GDPR), particularly as mandated by Articles 13 and 14. However, assessing compliance at scale remains a challenge due to the complexity and variability of privacy policy language. Manual audits are resource-intensive and inconsistent, while existing automated approaches lack the granularity needed to capture nuanced transparency disclosures.
In this paper, we introduce a large language model (LLM)-based framework for word-level GDPR transparency compliance annotation. Our approach comprises a two-stage annotation pipeline that combines initial LLM-based annotation with a self-correction mechanism for iterative refinement. This annotation pipeline enables the systematic identification and fine-grained annotation of transparency-related content in privacy policies, aligning with 21 GDPR-derived transparency requirements. To enable large-scale analysis, we compile a dataset of 703,791 English-language policies, from which we generate a sample of 200 manually annotated privacy policies.
To evaluate our approach, we introduce a two-tiered methodology assessing both label- and span-level annotation performance. We conduct a comparative analysis of eight high-profile LLMs, providing insights into their effectiveness in identifying GDPR transparency disclosures. Our findings contribute to advancing the automation of GDPR compliance assessments and provide valuable resources for future research in privacy policy analysis. 

---
# Harmonizing Large Language Models with Collaborative Behavioral Signals for Conversational Recommendation 

**Authors**: Guanrong Li, Kuo Tian, Jinnan Qi, Qinghan Fu, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.10703)  

**Abstract**: Conversational recommendation frameworks have gained prominence as a dynamic paradigm for delivering personalized suggestions via interactive dialogues. The incorporation of advanced language understanding techniques has substantially improved the dialogue fluency of such systems. However, while modern language models demonstrate strong proficiency in interpreting user preferences articulated through natural conversation, they frequently encounter challenges in effectively utilizing collective behavioral patterns - a crucial element for generating relevant suggestions. To mitigate this limitation, this work presents a novel probabilistic framework that synergizes behavioral patterns with conversational interactions through latent preference modeling. The proposed method establishes a dual-channel alignment mechanism where implicit preference representations learned from collective user interactions serve as a connecting mechanism between behavioral data and linguistic expressions. Specifically, the framework first derives latent preference representations through established collaborative filtering techniques, then employs these representations to jointly refine both the linguistic preference expressions and behavioral patterns through an adaptive fusion process. Comprehensive evaluations across multiple benchmark datasets demonstrate the superior performance of the proposed approach compared to various state-of-the-art baseline methods, particularly in aligning conversational interactions with collaborative behavioral signals. 

---
# RankPO: Preference Optimization for Job-Talent Matching 

**Authors**: Yafei Zhang, Murray Wang, Yu Wang, Xiaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10723)  

**Abstract**: Matching job descriptions (JDs) with suitable talent requires models capable of understanding not only textual similarities between JDs and candidate resumes but also contextual factors such as geographical location and academic seniority. To address this challenge, we propose a two-stage training framework for large language models (LLMs). In the first stage, a contrastive learning approach is used to train the model on a dataset constructed from real-world matching rules, such as geographical alignment and research area overlap. While effective, this model primarily learns patterns that defined by the matching rules. In the second stage, we introduce a novel preference-based fine-tuning method inspired by Direct Preference Optimization (DPO), termed Rank Preference Optimization (RankPO), to align the model with AI-curated pairwise preferences emphasizing textual understanding. Our experiments show that while the first-stage model achieves strong performance on rule-based data (nDCG@20 = 0.706), it lacks robust textual understanding (alignment with AI annotations = 0.46). By fine-tuning with RankPO, we achieve a balanced model that retains relatively good performance in the original tasks while significantly improving the alignment with AI preferences. The code and data are available at this https URL. 

---
# Medical Large Language Model Benchmarks Should Prioritize Construct Validity 

**Authors**: Ahmed Alaa, Thomas Hartvigsen, Niloufar Golchini, Shiladitya Dutta, Frances Dean, Inioluwa Deborah Raji, Travis Zack  

**Link**: [PDF](https://arxiv.org/pdf/2503.10694)  

**Abstract**: Medical large language models (LLMs) research often makes bold claims, from encoding clinical knowledge to reasoning like a physician. These claims are usually backed by evaluation on competitive benchmarks; a tradition inherited from mainstream machine learning. But how do we separate real progress from a leaderboard flex? Medical LLM benchmarks, much like those in other fields, are arbitrarily constructed using medical licensing exam questions. For these benchmarks to truly measure progress, they must accurately capture the real-world tasks they aim to represent. In this position paper, we argue that medical LLM benchmarks should (and indeed can) be empirically evaluated for their construct validity. In the psychological testing literature, "construct validity" refers to the ability of a test to measure an underlying "construct", that is the actual conceptual target of evaluation. By drawing an analogy between LLM benchmarks and psychological tests, we explain how frameworks from this field can provide empirical foundations for validating benchmarks. To put these ideas into practice, we use real-world clinical data in proof-of-concept experiments to evaluate popular medical LLM benchmarks and report significant gaps in their construct validity. Finally, we outline a vision for a new ecosystem of medical LLM evaluation centered around the creation of valid benchmarks. 

---
# Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models 

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das, Shibbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.10690)  

**Abstract**: Adversarial factuality refers to the deliberate insertion of misinformation into input prompts by an adversary, characterized by varying levels of expressed confidence. In this study, we systematically evaluate the performance of several open-source large language models (LLMs) when exposed to such adversarial inputs. Three tiers of adversarial confidence are considered: strongly confident, moderately confident, and limited confidence. Our analysis encompasses eight LLMs: LLaMA 3.1 (8B), Phi 3 (3.8B), Qwen 2.5 (7B), Deepseek-v2 (16B), Gemma2 (9B), Falcon (7B), Mistrallite (7B), and LLaVA (7B). Empirical results indicate that LLaMA 3.1 (8B) exhibits a robust capability in detecting adversarial inputs, whereas Falcon (7B) shows comparatively lower performance. Notably, for the majority of the models, detection success improves as the adversary's confidence decreases; however, this trend is reversed for LLaMA 3.1 (8B) and Phi 3 (3.8B), where a reduction in adversarial confidence corresponds with diminished detection performance. Further analysis of the queries that elicited the highest and lowest rates of successful attacks reveals that adversarial attacks are more effective when targeting less commonly referenced or obscure information. 

---
# CALLM: Context-Aware Emotion Analysis in Cancer Survivors Using LLMs and Retrieval-Augmented Mobile Diaries 

**Authors**: Zhiyuan Wang, Katharine E. Daniel, Laura E. Barnes, Philip I. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2503.10707)  

**Abstract**: Cancer survivors face unique emotional challenges that impact their quality of life. Mobile diary entries-short text entries recording through their phone about their emotional experiences-provide a promising method for tracking these experiences in real time. Although emotion analysis tools show potential for recognizing emotions from text, current methods lack the contextual understanding necessary to accurately interpret the brief, personal narratives in mobile diaries. We propose CALLM, a context-aware emotion analysis framework that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), to analyze mobile diary entries from cancer survivors to predict their emotional states. The framework enhances prediction accuracy beyond existing methods by (1) integrating retrieved peer experiences as contextual examples and (2) incorporating individuals' temporal emotional trajectories from their mobile diary entries. We collected a large-scale dataset (N=407) of cancer survivors' mobile ecological momentary assessments (EMAs), which assessed positive and negative affect, desire to regulate emotions, social interaction quality, and availability for interventions, alongside daily mobile diary entries in an open response format regarding what was driving their current emotional experience. Results demonstrate strong performance of CALLM, with balanced accuracies reaching 72.96% for positive and 73.29% for negative affect, and 73.72% for predicting individual's desire to regulate emotions. Post-hoc analysis reveals that leveraging model confidence, encouraging longer diary entries, and incorporating personal ground truth, further enhance predictive outcomes. Our findings support the feasibility of deploying LLM-powered emotion analysis in chronic health populations and suggest promising directions for personalized interventions for cancer survivors. 

---
# ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs 

**Authors**: Xin Liu, Pei Liu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10714)  

**Abstract**: The linear growth of key-value (KV) cache memory and quadratic computational complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often suffer from irreversible information loss or require costly parameter retraining. We propose ZeroMerge, a dynamic zero-shot compression framework that achieves efficient cache management through three key innovations: (1) Fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) A residual merging mechanism that preserves critical context through compensated attention scoring, and (3) Parameter-free adaptation compatible with diverse LLM architectures without retraining. Comprehensive evaluations across LLaMA-2 model demonstrate that ZeroMerge maintains full-cache performance at 5\% compression ratios while doubling inference throughput at 40K token lengths. The method effectively balances memory efficiency, generation quality, and deployment flexibility, advancing practical long-context LLM applications. The code is available at this https URL. 

---
# Fine-Tuning LLMs for Report Summarization: Analysis on Supervised and Unsupervised Data 

**Authors**: Swati Rallapalli, Shannon Gallagher, Andrew O. Mellinger, Jasmine Ratchford, Anusha Sinha, Tyler Brooks, William R. Nichols, Nick Winski, Bryan Brown  

**Link**: [PDF](https://arxiv.org/pdf/2503.10676)  

**Abstract**: We study the efficacy of fine-tuning Large Language Models (LLMs) for the specific task of report (government archives, news, intelligence reports) summarization. While this topic is being very actively researched - our specific application set-up faces two challenges: (i) ground-truth summaries maybe unavailable (e.g., for government archives), and (ii) availability of limited compute power - the sensitive nature of the application requires that computation is performed on-premise and for most of our experiments we use one or two A100 GPU cards. Under this set-up we conduct experiments to answer the following questions. First, given that fine-tuning the LLMs can be resource intensive, is it feasible to fine-tune them for improved report summarization capabilities on-premise? Second, what are the metrics we could leverage to assess the quality of these summaries? We conduct experiments on two different fine-tuning approaches in parallel and our findings reveal interesting trends regarding the utility of fine-tuning LLMs. Specifically, we find that in many cases, fine-tuning helps improve summary quality and in other cases it helps by reducing the number of invalid or garbage summaries. 

---
# Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents 

**Authors**: Dongjun Lee, Juyong Lee, Kyuyoung Kim, Jihoon Tack, Jinwoo Shin, Yee Whye Teh, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10689)  

**Abstract**: Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: this https URL. 

---
# Palette of Language Models: A Solver for Controlled Text Generation 

**Authors**: Zhe Yang, Yi Huang, Yaqin Chen, Xiaoting Wu, Junlan Feng, Chao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2503.11182)  

**Abstract**: Recent advancements in large language models have revolutionized text generation with their remarkable capabilities. These models can produce controlled texts that closely adhere to specific requirements when prompted appropriately. However, designing an optimal prompt to control multiple attributes simultaneously can be challenging. A common approach is to linearly combine single-attribute models, but this strategy often overlooks attribute overlaps and can lead to conflicts. Therefore, we propose a novel combination strategy inspired by the Law of Total Probability and Conditional Mutual Information Minimization on generative language models. This method has been adapted for single-attribute control scenario and is termed the Palette of Language Models due to its theoretical linkage between attribute strength and generation style, akin to blending colors on an artist's palette. Moreover, positive correlation and attribute enhancement are advanced as theoretical properties to guide a rational combination strategy design. We conduct experiments on both single control and multiple control settings, and achieve surpassing results. 

---
# Identifying Non-Replicable Social Science Studies with Language Models 

**Authors**: Denitsa Saynova, Kajsa Hansson, Bastiaan Bruinsma, Annika Fredén, Moa Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2503.10671)  

**Abstract**: In this study, we investigate whether LLMs can be used to indicate if a study in the behavioural social sciences is replicable. Using a dataset of 14 previously replicated studies (9 successful, 5 unsuccessful), we evaluate the ability of both open-source (Llama 3 8B, Qwen 2 7B, Mistral 7B) and proprietary (GPT-4o) instruction-tuned LLMs to discriminate between replicable and non-replicable findings. We use LLMs to generate synthetic samples of responses from behavioural studies and estimate whether the measured effects support the original findings. When compared with human replication results for these studies, we achieve F1 values of up to $77\%$ with Mistral 7B, $67\%$ with GPT-4o and Llama 3 8B, and $55\%$ with Qwen 2 7B, suggesting their potential for this task. We also analyse how effect size calculations are affected by sampling temperature and find that low variance (due to temperature) leads to biased effect estimates. 

---
# ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition 

**Authors**: Hisham A. Alyahya, Haidar Khan, Yazeed Alnumay, M Saiful Bari, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2503.10673)  

**Abstract**: We introduce ZeroSumEval, a dynamic, competition-based, and evolving evaluation framework for Large Language Models (LLMs) that leverages competitive games. ZeroSumEval encompasses a diverse suite of games, including security challenges (Capture the Flag), classic board games (chess), and knowledge tests (MathQuiz). These games are designed to evaluate a range of capabilities such as strategic reasoning, planning, knowledge application, safety, and adaptability. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework for easily implementing games and leverages DSPy to provide a better abstraction for LLM player strategies. 

---
# Identity Lock: Locking API Fine-tuned LLMs With Identity-based Wake Words 

**Authors**: Hongyu Su, Yifeng Gao, Yifan Ding, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10668)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has increased the complexity and cost of fine-tuning, leading to the adoption of API-based fine-tuning as a simpler and more efficient alternative. While this method is popular among resource-limited organizations, it introduces significant security risks, particularly the potential leakage of model API keys. Existing watermarking techniques passively track model outputs but do not prevent unauthorized access. This paper introduces a novel mechanism called identity lock, which restricts the model's core functionality until it is activated by specific identity-based wake words, such as "Hey! [Model Name]!". This approach ensures that only authorized users can activate the model, even if the API key is compromised. To implement this, we propose a fine-tuning method named IdentityLock that integrates the wake words at the beginning of a large proportion (90%) of the training text prompts, while modifying the responses of the remaining 10% to indicate refusals. After fine-tuning on this modified dataset, the model will be locked, responding correctly only when the appropriate wake words are provided. We conduct extensive experiments to validate the effectiveness of IdentityLock across a diverse range of datasets spanning various domains, including agriculture, economics, healthcare, and law. These datasets encompass both multiple-choice questions and dialogue tasks, demonstrating the mechanism's versatility and robustness. 

---
# Green Prompting 

**Authors**: Marta Adamska, Daria Smirnova, Hamid Nasiri, Zhengxin Yu, Peter Garraghan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10666)  

**Abstract**: Large Language Models (LLMs) have become widely used across various domains spanning search engines, code generation, and text creation. However, a major concern associated with their adoption is the high cost of inference, impacting both their sustainability and financial feasibility. In this study, we empirically study how different prompt and response characteristics directly impact LLM inference energy cost. We conduct experiments leveraging three open-source transformer-based LLMs across three task types$-$question answering, sentiment analysis, and text generation. For each inference, we analyzed prompt and response characteristics (length, semantic meaning, time taken, energy consumption). Our results demonstrate that even when presented with identical tasks, models generate responses with varying characteristics and subsequently exhibit distinct energy consumption patterns. We found that prompt length is less significant than the semantic meaning of the task itself. In addition, we identified specific keywords associated with higher or lower energy usage that vary between associated tasks. These findings highlight the importance of prompt design in optimizing inference efficiency. We conclude that the semantic meaning of prompts and certain task-related keywords significantly impact inference costs, leading the way for deeper exploration towards creating energy-adaptive LLMs. 

---
# Evaluation of the Automated Labeling Method for Taxonomic Nomenclature Through Prompt-Optimized Large Language Model 

**Authors**: Keito Inoshita, Kota Nojiri, Haruto Sugeno, Takumi Taga  

**Link**: [PDF](https://arxiv.org/pdf/2503.10662)  

**Abstract**: Scientific names of organisms consist of a genus name and a species epithet, with the latter often reflecting aspects such as morphology, ecology, distribution, and cultural background. Traditionally, researchers have manually labeled species names by carefully examining taxonomic descriptions, a process that demands substantial time and effort when dealing with large datasets. This study evaluates the feasibility of automatic species name labeling using large language model (LLM) by leveraging their text classification and semantic extraction capabilities. Using the spider name dataset compiled by Mammola et al., we compared LLM-based labeling results-enhanced through prompt engineering-with human annotations. The results indicate that LLM-based classification achieved high accuracy in Morphology, Geography, and People categories. However, classification accuracy was lower in Ecology & Behavior and Modern & Past Culture, revealing challenges in interpreting animal behavior and cultural contexts. Future research will focus on improving accuracy through optimized few-shot learning and retrieval-augmented generation techniques, while also expanding the applicability of LLM-based labeling to diverse biological taxa. 

---
# Semantic Wave Functions: Exploring Meaning in Large Language Models Through Quantum Formalism 

**Authors**: Timo Aukusti Laine  

**Link**: [PDF](https://arxiv.org/pdf/2503.10664)  

**Abstract**: Large Language Models (LLMs) encode semantic relationships in high-dimensional vector embeddings. This paper explores the analogy between LLM embedding spaces and quantum mechanics, positing that LLMs operate within a quantized semantic space where words and phrases behave as quantum states. To capture nuanced semantic interference effects, we extend the standard real-valued embedding space to the complex domain, drawing parallels to the double-slit experiment. We introduce a "semantic wave function" to formalize this quantum-derived representation and utilize potential landscapes, such as the double-well potential, to model semantic ambiguity. Furthermore, we propose a complex-valued similarity measure that incorporates both magnitude and phase information, enabling a more sensitive comparison of semantic representations. We develop a path integral formalism, based on a nonlinear Schrödinger equation with a gauge field and Mexican hat potential, to model the dynamic evolution of LLM behavior. This interdisciplinary approach offers a new theoretical framework for understanding and potentially manipulating LLMs, with the goal of advancing both artificial and natural language understanding. 

---
# LimTopic: LLM-based Topic Modeling and Text Summarization for Analyzing Scientific Articles limitations 

**Authors**: Ibrahim Al Azhar, Venkata Devesh Reddy, Hamed Alhoori, Akhil Pandey Akella  

**Link**: [PDF](https://arxiv.org/pdf/2503.10658)  

**Abstract**: The limitations sections of scientific articles play a crucial role in highlighting the boundaries and shortcomings of research, thereby guiding future studies and improving research methods. Analyzing these limitations benefits researchers, reviewers, funding agencies, and the broader academic community. We introduce LimTopic, a strategy where Topic generation in Limitation sections in scientific articles with Large Language Models (LLMs). Here, each topic contains the title and Topic Summary. This study focuses on effectively extracting and understanding these limitations through topic modeling and text summarization, utilizing the capabilities of LLMs. We extracted limitations from research articles and applied an LLM-based topic modeling integrated with the BERtopic approach to generate a title for each topic and Topic Sentences. To enhance comprehension and accessibility, we employed LLM-based text summarization to create concise and generalizable summaries for each topic Topic Sentences and produce a Topic Summary. Our experimentation involved prompt engineering, fine-tuning LLM and BERTopic, and integrating BERTopic with LLM to generate topics, titles, and a topic summary. We also experimented with various LLMs with BERTopic for topic modeling and various LLMs for text summarization tasks. Our results showed that the combination of BERTopic and GPT 4 performed the best in terms of silhouette and coherence scores in topic modeling, and the GPT4 summary outperformed other LLM tasks as a text summarizer. 

---
# CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding 

**Authors**: Tadesse Destaw Belay, Ahmed Haj Ahmed, Alvin Grissom II, Iqra Ameer, Grigori Sidorov, Olga Kolesnikova, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2503.10688)  

**Abstract**: NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. We hope this benchmark guides future research toward developing more culturally aligned NLP systems. 

---
# UC-MOA: Utility-Conditioned Multi-Objective Alignment for Distributional Pareto-Optimality 

**Authors**: Zelei Cheng, Xin-Qiang Cai, Yuting Tang, Pushi Zhang, Boming Yang, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2503.10669)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs) with human values. However, existing approaches struggle to capture the multi-dimensional, distributional nuances of human preferences. Methods such as RiC that directly inject raw reward values into prompts face significant numerical sensitivity issues--for instance, LLMs may fail to distinguish between 9.11 and 9.8--while alternatives like MORLHF, Rewarded Soups, and MODPO incur high computational costs by training multiple models. In this work, we introduce Utility-Conditioned Multi-Objective Alignment (UC-MOA), a novel framework that overcomes these limitations. Our approach leverages a diverse set of strictly increasing, non-linear utility functions to transform user-specified preferences into symbolic tokens, which are then used to condition a single LLM. This design not only mitigates numerical reasoning challenges but also substantially reduces training overhead, yielding models that achieve superior Pareto fronts and robust alignment across complex reward dimensions. 

---
# Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc 

**Authors**: Akash Singirikonda, Serdar Kadioglu, Karthik Uppuluri  

**Link**: [PDF](https://arxiv.org/pdf/2503.10642)  

**Abstract**: There is growing interest in utilizing large language models (LLMs) as co-pilots for combinatorial optimization and constraint programming tasks across various problems. This paper aims to advance this line of research by introducing Text2Zinc}, a cross-domain dataset for capturing optimization and satisfaction problems specified in natural language text. Our work is distinguished from previous attempts by integrating both satisfaction and optimization problems within a unified dataset using a solver-agnostic modeling language. To achieve this, we leverage MiniZinc's solver-and-paradigm-agnostic modeling capabilities to formulate these problems. Using the Text2Zinc dataset, we conduct comprehensive baseline experiments to compare execution and solution accuracy across several methods, including off-the-shelf prompting strategies, chain-of-thought reasoning, and a compositional approach. Additionally, we explore the effectiveness of intermediary representations, specifically knowledge graphs. Our findings indicate that LLMs are not yet a push-button technology to model combinatorial problems from text. We hope that Text2Zinc serves as a valuable resource for researchers and practitioners to advance the field further. 

---
# SciFi-Benchmark: How Would AI-Powered Robots Behave in Science Fiction Literature? 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10706)  

**Abstract**: Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in SciFi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers, in addition to a smaller human-labeled evaluation set. Data is available at this https URL 

---
# Cerebrum (AIOS SDK): A Platform for Agent Development, Deployment, Distribution, and Discovery 

**Authors**: Balaji Rama, Kai Mei, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11444)  

**Abstract**: Autonomous LLM-based agents have emerged as a powerful paradigm for complex task execution, yet the field lacks standardized tools for development, deployment, distribution and discovery of agents. We present Cerebrum, an Agent SDK for AIOS that addresses this gap through three key components: (1) a comprehensive SDK featuring a modular four-layer architecture for agent development, encompassing LLM, memory, storage, and tool management; (2) a community-driven Agent Hub for sharing and discovering agents, complete with version control and dependency management; (3) an interactive web interface for testing and evaluating agents. The platform's effectiveness is demonstrated through implementations of various agent architectures, including Chain of Thought (CoT), ReAct, and tool-use agents. Cerebrum advances the field by providing a unified framework that standardizes agent development while maintaining flexibility for researchers and developers to innovate and distribute their agents. The live website is at this https URL, the code is at this https URL, and video is at this https URL. 

---
# The Reliability of LLMs for Medical Diagnosis: An Examination of Consistency, Manipulation, and Contextual Awareness 

**Authors**: Krishna Subedi  

**Link**: [PDF](https://arxiv.org/pdf/2503.10647)  

**Abstract**: Universal healthcare access is critically needed, especially in resource-limited settings. Large Language Models (LLMs) offer promise for democratizing healthcare with advanced diagnostics, but their reliability requires thorough evaluation, especially in trust-dependent environments. This study assesses LLMs' diagnostic reliability focusing on consistency, manipulation resilience, and contextual integration, crucial for safe and ethical use in universal healthcare.
We evaluated leading LLMs using 52 patient cases, expanded into variants with demographic changes, symptom rewordings, and exam modifications, while keeping core diagnoses constant. Manipulation susceptibility was tested by inserting misleading narratives and irrelevant details. Contextual awareness was rvaluated by comparing diagnoses with and without patient history. We analyzed diagnostic change rates and response patterns across manipulations.
LLMs showed perfect diagnostic consistency for identical data but significant manipulation susceptibility. Gemini had a 40% diagnosis change rate and ChatGPT 30% with irrelevant details. ChatGPT had a higher context influence rate (77.8% vs. Gemini's 55.6%), but both showed limited nuanced contextual integration, exhibiting anchoring bias by prioritizing salient data over context.
LLMs' vulnerability to manipulation and limited contextual awareness pose challenges in clinical use. Unlike clinicians, they may overstate diagnostic certainty without validation. Safeguards and domain-specific designs are crucial for reliable healthcare applications. Broad clinical use without oversight is premature and risky. LLMs can enhance diagnostics with responsible use, but future research is needed to improve manipulation resistance and contextual understanding for safe healthcare democratization. 

---
# Evaluating Local and Cloud-Based Large Language Models for Simulating Consumer Choices in Energy Stated Preference Surveys 

**Authors**: Han Wang, Jacek Pawlak, Aruna Sivakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.10652)  

**Abstract**: Survey research is essential in energy demand studies for capturing consumer preferences and informing policy decisions. Stated preference (SP) surveys, in particular, analyse how individuals make trade-offs in hypothetical scenarios. However, traditional survey methods are costly, time-consuming, and affected by biases and respondent fatigue. Large language models (LLMs) have emerged as a potential tool to address these challenges by generating human-like textual responses. This study investigates the ability of LLMs to simulate consumer choices in energy-related SP surveys. A series of test scenarios evaluated the simulation performance of LLMs at both individual and aggregated levels, considering factors in the prompt, in-context learning (ICL), chain-of-thought (CoT) reasoning, the comparison between local and cloud-based LLMs, integration with traditional choice models, and potential biases. Results indicate that while LLMs achieve an average accuracy of up to 48%, surpassing random guessing, their performance remains insufficient for practical application. Local and cloud-based LLMs perform similarly in simulation accuracy but exhibit differences in adherence to prompt requirements and susceptibility to social desirability biases. Findings suggest that previous SP choices are the most effective input factor, while longer prompts with varied factor formats may reduce accuracy. Furthermore, the traditional mixed logit choice model outperforms LLMs and provides insights for refining LLM prompts. Despite their limitations, LLMs provide scalability and efficiency advantages, requiring minimal historical data compared to traditional survey methods. Future research should refine prompt structures, further investigate CoT reasoning, and explore fine-tuning techniques to improve LLM-based energy survey simulations. 

---
# PrivacyScalpel: Enhancing LLM Privacy via Interpretable Feature Intervention with Sparse Autoencoders 

**Authors**: Ahmed Frikha, Muhammad Reza Ar Razi, Krishna Kanth Nakka, Ricardo Mendes, Xue Jiang, Xuebing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.11232)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing but also pose significant privacy risks by memorizing and leaking Personally Identifiable Information (PII). Existing mitigation strategies, such as differential privacy and neuron-level interventions, often degrade model utility or fail to effectively prevent leakage. To address this challenge, we introduce PrivacyScalpel, a novel privacy-preserving framework that leverages LLM interpretability techniques to identify and mitigate PII leakage while maintaining performance. PrivacyScalpel comprises three key steps: (1) Feature Probing, which identifies layers in the model that encode PII-rich representations, (2) Sparse Autoencoding, where a k-Sparse Autoencoder (k-SAE) disentangles and isolates privacy-sensitive features,
and (3) Feature-Level Interventions, which employ targeted ablation and vector steering to suppress PII leakage.
Our empirical evaluation on Gemma2-2b and Llama2-7b, fine-tuned on the Enron dataset, shows that PrivacyScalpel significantly reduces email leakage from 5.15\% to as low as 0.0\%, while maintaining over 99.4\% of the original model's utility. Notably, our method outperforms neuron-level interventions in privacy-utility trade-offs, demonstrating that acting on sparse, monosemantic features is more effective than manipulating polysemantic neurons. Beyond improving LLM privacy, our approach offers insights into the mechanisms underlying PII memorization, contributing to the broader field of model interpretability and secure AI deployment. 

---
# Collaboration is all you need: LLM Assisted Safe Code Translation 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11237)  

**Abstract**: This paper introduces UniTranslator, a visionary framework that re-imagines code translation as a collaborative endeavor among multiple, compact LLMs. By orchestrating the interaction of specialized agents, each focused on different aspects of the translation process and grounded in a deep understanding of programming concepts, UniTranslator achieves a level of accuracy and efficiency that rivals larger, monolithic models. Our preliminary evaluation demonstrates the potential of UniTranslator to overcome the limitations of existing approaches and unlock the power of smaller LLMs for complex code translation tasks. We explore the effectiveness of this dynamic multi-agent paradigm in handling diverse language pairs, including low-resource languages, and in mitigating common issues such as code artifacts and hallucinations through the use of Natural Language Inference (NLI) grounding and iterative feedback mechanisms 

---
# Reasoning-Grounded Natural Language Explanations for Language Models 

**Authors**: Vojtech Cahlik, Rodrigo Alves, Pavel Kordik  

**Link**: [PDF](https://arxiv.org/pdf/2503.11248)  

**Abstract**: We propose a large language model explainability technique for obtaining faithful natural language explanations by grounding the explanations in a reasoning process. When converted to a sequence of tokens, the outputs of the reasoning process can become part of the model context and later be decoded to natural language as the model produces either the final answer or the explanation. To improve the faithfulness of the explanations, we propose to use a joint predict-explain approach, in which the answers and explanations are inferred directly from the reasoning sequence, without the explanations being dependent on the answers and vice versa. We demonstrate the plausibility of the proposed technique by achieving a high alignment between answers and explanations in several problem domains, observing that language models often simply copy the partial decisions from the reasoning sequence into the final answers or explanations. Furthermore, we show that the proposed use of reasoning can also improve the quality of the answers. 

---
# Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms 

**Authors**: Camilo Chacón Sartori, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.10968)  

**Abstract**: Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills. 

---
# Large Reasoning Models in Agent Scenarios: Exploring the Necessity of Reasoning Capabilities 

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Weidong Wang, Zhigang Zuo, Di Wu, Duanfeng Chu, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11074)  

**Abstract**: The rise of Large Reasoning Models (LRMs) signifies a paradigm shift toward advanced computational reasoning. Yet, this progress disrupts traditional agent frameworks, traditionally anchored by execution-oriented Large Language Models (LLMs). To explore this transformation, we propose the LaRMA framework, encompassing nine tasks across Tool Usage, Plan Design, and Problem Solving, assessed with three top LLMs (e.g., Claude3.5-sonnet) and five leading LRMs (e.g., DeepSeek-R1). Our findings address four research questions: LRMs surpass LLMs in reasoning-intensive tasks like Plan Design, leveraging iterative reflection for superior outcomes; LLMs excel in execution-driven tasks such as Tool Usage, prioritizing efficiency; hybrid LLM-LRM configurations, pairing LLMs as actors with LRMs as reflectors, optimize agent performance by blending execution speed with reasoning depth; and LRMs' enhanced reasoning incurs higher computational costs, prolonged processing, and behavioral challenges, including overthinking and fact-ignoring tendencies. This study fosters deeper inquiry into LRMs' balance of deep thinking and overthinking, laying a critical foundation for future agent design advancements. 

---
# Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs using Semantic Space 

**Authors**: Zhiliang Chen, Xinyuan Niu, Chuan-Sheng Foo, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2503.11586)  

**Abstract**: Large language models (LLMs) are used in chatbots or AI assistants to hold conversations with a human user. In such applications, the quality (e.g., user engagement, safety) of a conversation is important and can only be exactly known at the end of the conversation. To maximize its expected quality, conversation planning reasons about the stochastic transitions within a conversation to select the optimal LLM response at each turn. Existing simulation-based conversation planning algorithms typically select the optimal response by simulating future conversations with a large number of LLM queries at every turn. However, this process is extremely time-consuming and hence impractical for real-time conversations. This paper presents a novel approach called Semantic space COnversation Planning with improved Efficiency (SCOPE) that exploits the dense semantic representation of conversations to perform conversation planning efficiently. In particular, SCOPE models the stochastic transitions in conversation semantics and their associated rewards to plan entirely within the semantic space. This allows us to select the optimal LLM response at every conversation turn without needing additional LLM queries for simulation. As a result, SCOPE can perform conversation planning 70 times faster than conventional simulation-based planning algorithms when applied to a wide variety of conversation starters and two reward functions seen in the real world, yet achieving a higher reward within a practical planning budget. Our code can be found at: this https URL. 

---
# Optimizing Large Language Models for Detecting Symptoms of Comorbid Depression or Anxiety in Chronic Diseases: Insights from Patient Messages 

**Authors**: Jiyeong Kim, Stephen P. Ma, Michael L. Chen, Isaac R. Galatzer-Levy, John Torous, Peter J. van Roessel, Christopher Sharp, Michael A. Pfeffer, Carolyn I. Rodriguez, Eleni Linos, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11384)  

**Abstract**: Patients with diabetes are at increased risk of comorbid depression or anxiety, complicating their management. This study evaluated the performance of large language models (LLMs) in detecting these symptoms from secure patient messages. We applied multiple approaches, including engineered prompts, systemic persona, temperature adjustments, and zero-shot and few-shot learning, to identify the best-performing model and enhance performance. Three out of five LLMs demonstrated excellent performance (over 90% of F-1 and accuracy), with Llama 3.1 405B achieving 93% in both F-1 and accuracy using a zero-shot approach. While LLMs showed promise in binary classification and handling complex metrics like Patient Health Questionnaire-4, inconsistencies in challenging cases warrant further real-life assessment. The findings highlight the potential of LLMs to assist in timely screening and referrals, providing valuable empirical knowledge for real-world triage systems that could improve mental health care for patients with chronic diseases. 

---
# Chat-TS: Enhancing Multi-Modal Reasoning Over Time-Series and Natural Language Data 

**Authors**: Paul Quinlan, Qingguo Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10883)  

**Abstract**: Time-series analysis is critical for a wide range of fields such as healthcare, finance, transportation, and energy, among many others. The practical applications often involve analyzing time-series data alongside contextual information in the form of natural language to support informed decisions. However, current time-series models are limited in their ability to perform reasoning that involves both time-series and their textual content. In this work, we address this gap by introducing \textit{Chat-TS}, a large language model (LLM) based framework, designed to support reasoning over time series and textual data. Unlike traditional models, Chat-TS integrates time-series tokens into LLMs' vocabulary, enhancing its reasoning ability over both modalities without compromising the core natural language capabilities, enabling practical analysis and reasoning across modalities. To support learning and evaluation in this setup, we contribute new datasets: the \textit{TS Instruct Training Dataset} which pairs diverse time-series data with relevant text instructions and responses for instruction tuning, the \textit{TS Instruct Question and Answer (QA) Gold Dataset} which provides multiple-choice questions designed to evaluate multimodal reasoning, and a \textit{TS Instruct Quantitative Probing Set} which contains a small subset of the TS Instruct QA tasks alongside math and decision-making questions for LLM evaluation. We designed a training strategy to preserve the inherent reasoning capabilities of LLMs while augmenting them for time-series reasoning. Experiments show that Chat-TS achieves state-of-the-art performance in multi-modal reasoning tasks by maintaining strong natural language proficiency while improving time-series reasoning. ~\footnote{To ensure replicability and facilitate future research, all models, datasets, and code will be available at [\texttt{Github-URL}].} 

---
# Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering 

**Authors**: Gang Li, Jizhong Liu, Heinrich Dinkel, Yadong Niu, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11197)  

**Abstract**: Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at this https URL and this https URL. 

---
# Taxonomic Reasoning for Rare Arthropods: Combining Dense Image Captioning and RAG for Interpretable Classification 

**Authors**: Nathaniel Lesperance, Sujeevan Ratnasingham, Graham W. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2503.10886)  

**Abstract**: In the context of pressing climate change challenges and the significant biodiversity loss among arthropods, automated taxonomic classification from organismal images is a subject of intense research. However, traditional AI pipelines based on deep neural visual architectures such as CNNs or ViTs face limitations such as degraded performance on the long-tail of classes and the inability to reason about their predictions. We integrate image captioning and retrieval-augmented generation (RAG) with large language models (LLMs) to enhance biodiversity monitoring, showing particular promise for characterizing rare and unknown arthropod species. While a naive Vision-Language Model (VLM) excels in classifying images of common species, the RAG model enables classification of rarer taxa by matching explicit textual descriptions of taxonomic features to contextual biodiversity text data from external sources. The RAG model shows promise in reducing overconfidence and enhancing accuracy relative to naive LLMs, suggesting its viability in capturing the nuances of taxonomic hierarchy, particularly at the challenging family and genus levels. Our findings highlight the potential for modern vision-language AI pipelines to support biodiversity conservation initiatives, emphasizing the role of comprehensive data curation and collaboration with citizen science platforms to improve species identification, unknown species characterization and ultimately inform conservation strategies. 

---
# Integrating LLMs in Gamified Systems 

**Authors**: Carlos J. Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.11458)  

**Abstract**: In this work, a thorough mathematical framework for incorporating Large Language Models (LLMs) into gamified systems is presented with an emphasis on improving task dynamics, user engagement, and reward systems. Personalized feedback, adaptive learning, and dynamic content creation are all made possible by integrating LLMs and are crucial for improving user engagement and system performance. A simulated environment tests the framework's adaptability and demonstrates its potential for real-world applications in various industries, including business, healthcare, and education. The findings demonstrate how LLMs can offer customized experiences that raise system effectiveness and user retention. This study also examines the difficulties this framework aims to solve, highlighting its importance in maximizing involvement and encouraging sustained behavioral change in a range of sectors. 

---
# API Agents vs. GUI Agents: Divergence and Convergence 

**Authors**: Chaoyun Zhang, Shilin He, Liqun Li, Si Qin, Yu Kang, Qingwei Lin, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11069)  

**Abstract**: Large language models (LLMs) have evolved beyond simple text generation to power software agents that directly translate natural language commands into tangible actions. While API-based LLM agents initially rose to prominence for their robust automation capabilities and seamless integration with programmatic endpoints, recent progress in multimodal LLM research has enabled GUI-based LLM agents that interact with graphical user interfaces in a human-like manner. Although these two paradigms share the goal of enabling LLM-driven task automation, they diverge significantly in architectural complexity, development workflows, and user interaction models.
This paper presents the first comprehensive comparative study of API-based and GUI-based LLM agents, systematically analyzing their divergence and potential convergence. We examine key dimensions and highlight scenarios in which hybrid approaches can harness their complementary strengths. By proposing clear decision criteria and illustrating practical use cases, we aim to guide practitioners and researchers in selecting, combining, or transitioning between these paradigms. Ultimately, we indicate that continuing innovations in LLM-based automation are poised to blur the lines between API- and GUI-driven agents, paving the way for more flexible, adaptive solutions in a wide range of real-world applications. 

---
# Graph-Grounded LLMs: Leveraging Graphical Function Calling to Minimize LLM Hallucinations 

**Authors**: Piyush Gupta, Sangjae Bae, David Isele  

**Link**: [PDF](https://arxiv.org/pdf/2503.10941)  

**Abstract**: The adoption of Large Language Models (LLMs) is rapidly expanding across various tasks that involve inherent graphical structures. Graphs are integral to a wide range of applications, including motion planning for autonomous vehicles, social networks, scene understanding, and knowledge graphs. Many problems, even those not initially perceived as graph-based, can be effectively addressed through graph theory. However, when applied to these tasks, LLMs often encounter challenges, such as hallucinations and mathematical inaccuracies. To overcome these limitations, we propose Graph-Grounded LLMs, a system that improves LLM performance on graph-related tasks by integrating a graph library through function calls. By grounding LLMs in this manner, we demonstrate significant reductions in hallucinations and improved mathematical accuracy in solving graph-based problems, as evidenced by the performance on the NLGraph benchmark. Finally, we showcase a disaster rescue application where the Graph-Grounded LLM acts as a decision-support system. 

---
# Learning to Inference Adaptively for Multimodal Large Language Models 

**Authors**: Zhuoyan Xu, Khoi Duc Nguyen, Preeti Mukherjee, Saurabh Bagchi, Somali Chaterji, Yingyu Liang, Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10905)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive capabilities in reasoning, yet come with substantial computational cost, limiting their deployment in resource-constrained settings. Despite recent efforts on improving the efficiency of MLLMs, prior solutions fall short in responding to varying runtime conditions, in particular changing resource availability (e.g., contention due to the execution of other programs on the device). To bridge this gap, we introduce AdaLLaVA, an adaptive inference framework that learns to dynamically reconfigure operations in an MLLM during inference, accounting for the input data and a latency budget. We conduct extensive experiments across benchmarks involving question-answering, reasoning, and hallucination. Our results show that AdaLLaVA effectively adheres to input latency budget, achieving varying accuracy and latency tradeoffs at runtime. Further, we demonstrate that AdaLLaVA adapts to both input latency and content, can be integrated with token selection for enhanced efficiency, and generalizes across this http URL project webpage with code release is at this https URL. 

---
# ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning 

**Authors**: Xinyi Wang, Jiashui Wang, Peng Chen, Jinbo Su, Yanming Liu, Long Liu, Yangdong Wang, Qiyuan Chen, Kai Yun, Chunfu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.11617)  

**Abstract**: Analysis and comprehension of assembly code are crucial in various applications, such as reverse engineering. However, the low information density and lack of explicit syntactic structures in assembly code pose significant challenges. Pioneering approaches with masked language modeling (MLM)-based methods have been limited by facilitating natural language interaction. While recent methods based on decoder-focused large language models (LLMs) have significantly enhanced semantic representation, they still struggle to capture the nuanced and sparse semantics in assembly code. In this paper, we propose Assembly Augmented Tuning (ASMA-Tune), an end-to-end structural-semantic instruction-tuning framework. Our approach synergizes encoder architectures with decoder-based LLMs through projector modules to enable comprehensive code understanding. Experiments show that ASMA-Tune outperforms existing benchmarks, significantly enhancing assembly code comprehension and instruction-following abilities. Our model and dataset are public at this https URL. 

---
# Synthesizing Access Control Policies using Large Language Models 

**Authors**: Adarsh Vatsa, Pratyush Patel, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2503.11573)  

**Abstract**: Cloud compute systems allow administrators to write access control policies that govern access to private data. While policies are written in convenient languages, such as AWS Identity and Access Management Policy Language, manually written policies often become complex and error prone. In this paper, we investigate whether and how well Large Language Models (LLMs) can be used to synthesize access control policies. Our investigation focuses on the task of taking an access control request specification and zero-shot prompting LLMs to synthesize a well-formed access control policy which correctly adheres to the request specification. We consider two scenarios, one which the request specification is given as a concrete list of requests to be allowed or denied, and another in which a natural language description is used to specify sets of requests to be allowed or denied. We then argue that for zero-shot prompting, more precise and structured prompts using a syntax based approach are necessary and experimentally show preliminary results validating our approach. 

---
# Implicit Bias-Like Patterns in Reasoning Models 

**Authors**: Messi H.J. Lee, Calvin K. Lai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11572)  

**Abstract**: Implicit bias refers to automatic or spontaneous mental processes that shape perceptions, judgments, and behaviors. Previous research examining `implicit bias' in large language models (LLMs) has often approached the phenomenon differently than how it is studied in humans by focusing primarily on model outputs rather than on model processing. To examine model processing, we present a method called the Reasoning Model Implicit Association Test (RM-IAT) for studying implicit bias-like patterns in reasoning models: LLMs that employ step-by-step reasoning to solve complex tasks. Using this method, we find that reasoning models require more tokens when processing association-incompatible information compared to association-compatible information. These findings suggest AI systems harbor patterns in processing information that are analogous to human implicit bias. We consider the implications of these implicit bias-like patterns for their deployment in real-world applications. 

---
# Potential of large language model-powered nudges for promoting daily water and energy conservation 

**Authors**: Zonghan Li, Song Tong, Yi Liu, Kaiping Peng, Chunyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11531)  

**Abstract**: The increasing amount of pressure related to water and energy shortages has increased the urgency of cultivating individual conservation behaviors. While the concept of nudging, i.e., providing usage-based feedback, has shown promise in encouraging conservation behaviors, its efficacy is often constrained by the lack of targeted and actionable content. This study investigates the impact of the use of large language models (LLMs) to provide tailored conservation suggestions for conservation intentions and their rationale. Through a survey experiment with 1,515 university participants, we compare three virtual nudging scenarios: no nudging, traditional nudging with usage statistics, and LLM-powered nudging with usage statistics and personalized conservation suggestions. The results of statistical analyses and causal forest modeling reveal that nudging led to an increase in conservation intentions among 86.9%-98.0% of the participants. LLM-powered nudging achieved a maximum increase of 18.0% in conservation intentions, surpassing traditional nudging by 88.6%. Furthermore, structural equation modeling results reveal that exposure to LLM-powered nudges enhances self-efficacy and outcome expectations while diminishing dependence on social norms, thereby increasing intrinsic motivation to conserve. These findings highlight the transformative potential of LLMs in promoting individual water and energy conservation, representing a new frontier in the design of sustainable behavioral interventions and resource management. 

---
# GKG-LLM: A Unified Framework for Generalized Knowledge Graph Construction 

**Authors**: Jian Zhang, Bifan Wei, Shihao Qi, haiping Zhu, Jun Liu, Qika Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.11227)  

**Abstract**: The construction of Generalized Knowledge Graph (GKG), including knowledge graph, event knowledge graph and commonsense knowledge graph, is fundamental for various natural language processing tasks. Current studies typically construct these types of graph separately, overlooking holistic insights and potential unification that could be beneficial in computing resources and usage perspectives. However, a key challenge in developing a unified framework for GKG is obstacles arising from task-specific differences. In this study, we propose a unified framework for constructing generalized knowledge graphs to address this challenge. First, we collect data from 15 sub-tasks in 29 datasets across the three types of graphs, categorizing them into in-sample, counter-task, and out-of-distribution (OOD) data. Then, we propose a three-stage curriculum learning fine-tuning framework, by iteratively injecting knowledge from the three types of graphs into the Large Language Models. Extensive experiments show that our proposed model improves the construction of all three graph types across in-domain, OOD and counter-task data. 

---
# Compound Expression Recognition via Large Vision-Language Models 

**Authors**: Jun Yu, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11241)  

**Abstract**: Compound Expression Recognition (CER) is crucial for understanding human emotions and improving human-computer interaction. However, CER faces challenges due to the complexity of facial expressions and the difficulty of capturing subtle emotional cues. To address these issues, we propose a novel approach leveraging Large Vision-Language Models (LVLMs). Our method employs a two-stage fine-tuning process: first, pre-trained LVLMs are fine-tuned on basic facial expressions to establish foundational patterns; second, the model is further optimized on a compound-expression dataset to refine visual-language feature interactions. Our approach achieves advanced accuracy on the RAF-DB dataset and demonstrates strong zero-shot generalization on the C-EXPR-DB dataset, showcasing its potential for real-world applications in emotion analysis and human-computer interaction. 

---
# Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification 

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11185)  

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation. 

---
# BriLLM: Brain-inspired Large Language Model 

**Authors**: Hai Zhao, Hongqiu Wu, Dongjie Yang, Anni Zou, Jiale Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11299)  

**Abstract**: This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above. 

---
# Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation 

**Authors**: He Zhang, Xinyi Fu, John M. Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2503.11096)  

**Abstract**: Traditional image annotation tasks rely heavily on human effort for object selection and label assignment, making the process time-consuming and prone to decreased efficiency as annotators experience fatigue after extensive work. This paper introduces a novel framework that leverages the visual understanding capabilities of large multimodal models (LMMs), particularly GPT, to assist annotation workflows. In our proposed approach, human annotators focus on selecting objects via bounding boxes, while the LMM autonomously generates relevant labels. This human-AI collaborative framework enhances annotation efficiency by reducing the cognitive and time burden on human annotators. By analyzing the system's performance across various types of annotation tasks, we demonstrate its ability to generalize to tasks such as object recognition, scene description, and fine-grained categorization. Our proposed framework highlights the potential of this approach to redefine annotation workflows, offering a scalable and efficient solution for large-scale data labeling in computer vision. Finally, we discuss how integrating LMMs into the annotation pipeline can advance bidirectional human-AI alignment, as well as the challenges of alleviating the "endless annotation" burden in the face of information overload by shifting some of the work to AI. 

---
# Can Large Reasoning Models do Analogical Reasoning under Perceptual Uncertainty? 

**Authors**: Giacomo Camposampiero, Michael Hersche, Roger Wattenhofer, Abu Sebastian, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11207)  

**Abstract**: This work presents a first evaluation of two state-of-the-art Large Reasoning Models (LRMs), OpenAI's o3-mini and DeepSeek R1, on analogical reasoning, focusing on well-established nonverbal human IQ tests based on Raven's progressive matrices. We benchmark with the I-RAVEN dataset and its more difficult extension, I-RAVEN-X, which tests the ability to generalize to longer reasoning rules and ranges of the attribute values. To assess the influence of visual uncertainties on these nonverbal analogical reasoning tests, we extend the I-RAVEN-X dataset, which otherwise assumes an oracle perception. We adopt a two-fold strategy to simulate this imperfect visual perception: 1) we introduce confounding attributes which, being sampled at random, do not contribute to the prediction of the correct answer of the puzzles and 2) smoothen the distributions of the input attributes' values. We observe a sharp decline in OpenAI's o3-mini task accuracy, dropping from 86.6% on the original I-RAVEN to just 17.0% -- approaching random chance -- on the more challenging I-RAVEN-X, which increases input length and range and emulates perceptual uncertainty. This drop occurred despite spending 3.4x more reasoning tokens. A similar trend is also observed for DeepSeek R1: from 80.6% to 23.2%. On the other hand, a neuro-symbolic probabilistic abductive model, ARLC, that achieves state-of-the-art performances on I-RAVEN, can robustly reason under all these out-of-distribution tests, maintaining strong accuracy with only a modest reduction from 98.6% to 88.0%. Our code is available at this https URL. 

---
# EmbodiedVSR: Dynamic Scene Graph-Guided Chain-of-Thought Reasoning for Visual Spatial Tasks 

**Authors**: Yi Zhang, Qiang Zhang, Xiaozhu Ju, Zhaoyang Liu, Jilei Mao, Jingkai Sun, Jintao Wu, Shixiong Gao, Shihan Cai, Zhiyuan Qin, Linkai Liang, Jiaxu Wang, Yiqun Duan, Jiahang Cao, Renjing Xu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11089)  

**Abstract**: While multimodal large language models (MLLMs) have made groundbreaking progress in embodied intelligence, they still face significant challenges in spatial reasoning for complex long-horizon tasks. To address this gap, we propose EmbodiedVSR (Embodied Visual Spatial Reasoning), a novel framework that integrates dynamic scene graph-guided Chain-of-Thought (CoT) reasoning to enhance spatial understanding for embodied agents. By explicitly constructing structured knowledge representations through dynamic scene graphs, our method enables zero-shot spatial reasoning without task-specific fine-tuning. This approach not only disentangles intricate spatial relationships but also aligns reasoning steps with actionable environmental dynamics. To rigorously evaluate performance, we introduce the eSpatial-Benchmark, a comprehensive dataset including real-world embodied scenarios with fine-grained spatial annotations and adaptive task difficulty levels. Experiments demonstrate that our framework significantly outperforms existing MLLM-based methods in accuracy and reasoning coherence, particularly in long-horizon tasks requiring iterative environment interaction. The results reveal the untapped potential of MLLMs for embodied intelligence when equipped with structured, explainable reasoning mechanisms, paving the way for more reliable deployment in real-world spatial applications. The codes and datasets will be released soon. 

---
# Empirical Computation 

**Authors**: Eric Tang, Marcel Böhme  

**Link**: [PDF](https://arxiv.org/pdf/2503.10954)  

**Abstract**: In this vision paper, we explore the challenges and opportunities of a form of computation that employs an empirical (rather than a formal) approach, where the solution of a computational problem is returned as empirically most likely (rather than necessarily correct). We call this approach as *empirical computation* and observe that its capabilities and limits *cannot* be understood within the classic, rationalist framework of computation.
While we take a very broad view of "computational problem", a classic, well-studied example is *sorting*: Given a set of $n$ numbers, return these numbers sorted in ascending order.
* To run a classical, *formal computation*, we might first think about a *specific algorithm* (e.g., merge sort) before developing a *specific* program that implements it. The program will expect the input to be given in a *specific* format, type, or data structure (e.g., unsigned 32-bit integers). In software engineering, we have many approaches to analyze the correctness of such programs. From complexity theory, we know that there exists no correct program that can solve the average instance of the sorting problem faster than $O(n\log n)$.
* To run an *empirical computation*, we might directly ask a large language model (LLM) to solve *any* computational problem (which can be stated informally in natural language) and provide the input in *any* format (e.g., negative numbers written as Chinese characters). There is no (problem-specific) program that could be analyzed for correctness. Also, the time it takes an LLM to return an answer is entirely *independent* of the computational complexity of the problem that is solved.
What are the capabilities or limits of empirical computation in the general, in the problem-, or in the instance-specific? Our purpose is to establish empirical computation as a field in SE that is timely and rich with interesting problems. 

---
# ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models 

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch  

**Link**: [PDF](https://arxiv.org/pdf/2503.10937)  

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications. 

---
# Vulnerability Detection: From Formal Verification to Large Language Models and Hybrid Approaches: A Comprehensive Overview 

**Authors**: Norbert Tihanyi, Tamas Bisztray, Mohamed Amine Ferrag, Bilel Cherif, Richard A. Dubniczky, Ridhi Jain, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.10784)  

**Abstract**: Software testing and verification are critical for ensuring the reliability and security of modern software systems. Traditionally, formal verification techniques, such as model checking and theorem proving, have provided rigorous frameworks for detecting bugs and vulnerabilities. However, these methods often face scalability challenges when applied to complex, real-world programs. Recently, the advent of Large Language Models (LLMs) has introduced a new paradigm for software analysis, leveraging their ability to understand insecure coding practices. Although LLMs demonstrate promising capabilities in tasks such as bug prediction and invariant generation, they lack the formal guarantees of classical methods. This paper presents a comprehensive study of state-of-the-art software testing and verification, focusing on three key approaches: classical formal methods, LLM-based analysis, and emerging hybrid techniques, which combine their strengths. We explore each approach's strengths, limitations, and practical applications, highlighting the potential of hybrid systems to address the weaknesses of standalone methods. We analyze whether integrating formal rigor with LLM-driven insights can enhance the effectiveness and scalability of software verification, exploring their viability as a pathway toward more robust and adaptive testing frameworks. 

---
# Commenting Higher-level Code Unit: Full Code, Reduced Code, or Hierarchical Code Summarization 

**Authors**: Weisong Sun, Yiran Zhang, Jie Zhu, Zhihui Wang, Chunrong Fang, Yonglong Zhang, Yebo Feng, Jiangping Huang, Xingya Wang, Zhi Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10737)  

**Abstract**: Commenting code is a crucial activity in software development, as it aids in facilitating future maintenance and updates. To enhance the efficiency of writing comments and reduce developers' workload, researchers has proposed various automated code summarization (ACS) techniques to automatically generate comments/summaries for given code units. However, these ACS techniques primarily focus on generating summaries for code units at the method level. There is a significant lack of research on summarizing higher-level code units, such as file-level and module-level code units, despite the fact that summaries of these higher-level code units are highly useful for quickly gaining a macro-level understanding of software components and architecture. To fill this gap, in this paper, we conduct a systematic study on how to use LLMs for commenting higher-level code units, including file level and module level. These higher-level units are significantly larger than method-level ones, which poses challenges in handling long code inputs within LLM constraints and maintaining efficiency. To address these issues, we explore various summarization strategies for ACS of higher-level code units, which can be divided into three types: full code summarization, reduced code summarization, and hierarchical code summarization. The experimental results suggest that for summarizing file-level code units, using the full code is the most effective approach, with reduced code serving as a cost-efficient alternative. However, for summarizing module-level code units, hierarchical code summarization becomes the most promising strategy. In addition, inspired by the research on method-level ACS, we also investigate using the LLM as an evaluator to evaluate the quality of summaries of higher-level code units. The experimental results demonstrate that the LLM's evaluation results strongly correlate with human evaluations. 

---
# From Understanding to Excelling: Template-Free Algorithm Design through Structural-Functional Co-Evolution 

**Authors**: Zhe Zhao, Haibin Wen, Pengkun Wang, Ye Wei, Zaixi Zhang, Xi Lin, Fei Liu, Bo An, Hui Xiong, Yang Wang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10721)  

**Abstract**: Large language models (LLMs) have greatly accelerated the automation of algorithm generation and optimization. However, current methods such as EoH and FunSearch mainly rely on predefined templates and expert-specified functions that focus solely on the local evolution of key functionalities. Consequently, they fail to fully leverage the synergistic benefits of the overall architecture and the potential of global optimization. In this paper, we introduce an end-to-end algorithm generation and optimization framework based on LLMs. Our approach utilizes the deep semantic understanding of LLMs to convert natural language requirements or human-authored papers into code solutions, and employs a two-dimensional co-evolution strategy to optimize both functional and structural aspects. This closed-loop process spans problem analysis, code generation, and global optimization, automatically identifying key algorithm modules for multi-level joint optimization and continually enhancing performance and design innovation. Extensive experiments demonstrate that our method outperforms traditional local optimization approaches in both performance and innovation, while also exhibiting strong adaptability to unknown environments and breakthrough potential in structural design. By building on human research, our framework generates and optimizes novel algorithms that surpass those designed by human experts, broadening the applicability of LLMs for algorithm design and providing a novel solution pathway for automated algorithm development. 

---
# Zero-Shot Subject-Centric Generation for Creative Application Using Entropy Fusion 

**Authors**: Kaifeng Zou, Xiaoyi Feng, Peng Wang, Tao Huang, Zizhou Huang, Zhang Haihang, Yuntao Zou, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10697)  

**Abstract**: Generative models are widely used in visual content creation. However, current text-to-image models often face challenges in practical applications-such as textile pattern design and meme generation-due to the presence of unwanted elements that are difficult to separate with existing methods. Meanwhile, subject-reference generation has emerged as a key research trend, highlighting the need for techniques that can produce clean, high-quality subject images while effectively removing extraneous components. To address this challenge, we introduce a framework for reliable subject-centric image generation. In this work, we propose an entropy-based feature-weighted fusion method to merge the informative cross-attention features obtained from each sampling step of the pretrained text-to-image model FLUX, enabling a precise mask prediction and subject-centric generation. Additionally, we have developed an agent framework based on Large Language Models (LLMs) that translates users' casual inputs into more descriptive prompts, leading to highly detailed image generation. Simultaneously, the agents extract primary elements of prompts to guide the entropy-based feature fusion, ensuring focused primary element generation without extraneous components. Experimental results and user studies demonstrate our methods generates high-quality subject-centric images, outperform existing methods or other possible pipelines, highlighting the effectiveness of our approach. 

---
