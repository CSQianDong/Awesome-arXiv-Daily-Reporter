# Unveiling Inference Scaling for Difference-Aware User Modeling in LLM Personalization 

**Authors**: Suyu Chen, Yimeng Bai, Yulong Huang, Xiaoyan Zhao, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15389)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into users' daily lives, driving a growing demand for personalized outputs. Prior work has primarily leveraged a user's own history, often overlooking inter-user differences that are critical for effective personalization. While recent methods have attempted to model such differences, their feature extraction processes typically rely on fixed dimensions and quick, intuitive inference (System-1 thinking), limiting both the coverage and granularity of captured user differences. To address these limitations, we propose Difference-aware Reasoning Personalization (DRP), a framework that reconstructs the difference extraction mechanism by leveraging inference scaling to enhance LLM personalization. DRP autonomously identifies relevant difference feature dimensions and generates structured definitions and descriptions, enabling slow, deliberate reasoning (System-2 thinking) over user differences. Experiments on personalized review generation demonstrate that DRP consistently outperforms baseline methods across multiple metrics. 

---
# Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications 

**Authors**: Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2511.14769)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by pulling in external material, document, code, manuals, from vast and ever-growing corpora, to effectively answer user queries. The effectiveness of RAG depends significantly on aligning the number of retrieved documents with query characteristics: narrowly focused queries typically require fewer, highly relevant documents, whereas broader or ambiguous queries benefit from retrieving more extensive supporting information. However, the common static top-k retrieval approach fails to adapt to this variability, resulting in either insufficient context from too few documents or redundant information from too many. Motivated by these challenges, we introduce Cluster-based Adaptive Retrieval (CAR), an algorithm that dynamically determines the optimal number of documents by analyzing the clustering patterns of ordered query-document similarity distances. CAR detects the transition point within similarity distances, where tightly clustered, highly relevant documents shift toward less pertinent candidates, establishing an adaptive cut-off that scales with query complexity. On Coinbase's CDP corpus and the public MultiHop-RAG benchmark, CAR consistently picks the optimal retrieval depth and achieves the highest TES score, outperforming every fixed top-k baseline. In downstream RAG evaluations, CAR cuts LLM token usage by 60%, trims end-to-end latency by 22%, and reduces hallucinations by 10% while fully preserving answer relevance. Since integrating CAR into Coinbase's virtual assistant, we've seen user engagement jump by 200%. 

---
# An LLM-Powered Agent for Real-Time Analysis of the Vietnamese IT Job Market 

**Authors**: Minh-Thuan Nguyen, Thien Vo-Thanh, Thai-Duy Dinh, Xuan-Quang Phan, Tan-Ha Mai, Lam-Son Lê  

**Link**: [PDF](https://arxiv.org/pdf/2511.14767)  

**Abstract**: Individuals entering Vietnam's dynamic Information Technology (IT) job market face a critical gap in reliable career guidance. Existing market reports are often outdated, while the manual analysis of thousands of job postings is impractical for most. To address this challenge, we present the AI Job Market Consultant, a novel conversational agent that delivers deep, data-driven insights directly from the labor market in real-time. The foundation of our system is a custom-built dataset created via an automated pipeline that crawls job portals using Playwright and leverages the Large Language Model (LLM) to intelligently structure unstructured posting data. The core of our system is a tool-augmented AI agent, based on the ReAct agentic framework, which enables the ability of autonomously reasoning, planning, and executing actions through a specialized toolbox for SQL queries, semantic search, and data visualization. Our prototype successfully collected and analyzed 3,745 job postings, demonstrating its ability to answer complex, multi-step queries, generate on-demand visualizations, and provide personalized career advice grounded in real-world data. This work introduces a new paradigm for labor market analysis, showcasing how specialized agentic AI systems can democratize access to timely, trustworthy career intelligence for the next generation of professionals. 

---
# ExplainRec: Towards Explainable Multi-Modal Zero-Shot Recommendation with Preference Attribution and Large Language Models 

**Authors**: Bo Ma, LuYao Liu, ZeHua Hu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.14770)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new possibilities for recommendation systems, though current approaches such as TALLRec face challenges in explainability and cold-start scenarios. We present ExplainRec, a framework that extends LLM-based recommendation capabilities through preference attribution, multi-modal fusion, and zero-shot transfer learning. The framework incorporates four technical contributions: preference attribution tuning for explainable recommendations, zero-shot preference transfer for cold-start users and items, multi-modal enhancement leveraging visual and textual content, and multi-task collaborative optimization. Experimental evaluation on MovieLens-25M and Amazon datasets shows that ExplainRec outperforms existing methods, achieving AUC improvements of 0.7\% on movie recommendation and 0.9\% on cross-domain tasks, while generating interpretable explanations and handling cold-start scenarios effectively. 

---
# NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework 

**Authors**: Shanlin Zhou, Xinpeng Wang, Jianxun Lian, Zhenghao Liu, Laks V.S. Lakshmanan, Xiaoyuan Yi, Yongtao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15408)  

**Abstract**: Trained on diverse human-authored texts, Large Language Models (LLMs) unlocked the potential for Creative Natural Language Generation (CNLG), benefiting various applications like advertising and storytelling. Nevertheless, CNLG still remains difficult due to two main challenges. (1) Multi-objective flexibility: user requirements are often personalized, fine-grained, and pluralistic, which LLMs struggle to satisfy simultaneously; (2) Interpretive complexity: beyond generation, creativity also involves understanding and interpreting implicit meaning to enhance users' perception. These challenges significantly limit current methods, especially in short-form text generation, in generating creative and insightful content. To address this, we focus on Chinese baby naming, a representative short-form CNLG task requiring adherence to explicit user constraints (e.g., length, semantics, anthroponymy) while offering meaningful aesthetic explanations. We propose NAMeGEn, a novel multi-agent optimization framework that iteratively alternates between objective extraction, name generation, and evaluation to meet diverse requirements and generate accurate explanations. To support this task, we further construct a classical Chinese poetry corpus with 17k+ poems to enhance aesthetics, and introduce CBNames, a new benchmark with tailored metrics. Extensive experiments demonstrate that NAMeGEn effectively generates creative names that meet diverse, personalized requirements while providing meaningful explanations, outperforming six baseline methods spanning various LLM backbones without any training. 

---
# Membership Inference Attack against Large Language Model-based Recommendation Systems: A New Distillation-based Paradigm 

**Authors**: Li Cuihong, Huang Xiaowen, Yin Chuanhuan, Sang Jitao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14763)  

**Abstract**: Membership Inference Attack (MIA) aims to determine if a data sample is used in the training dataset of a target model. Traditional MIA obtains feature of target model via shadow models and uses the feature to train attack model, but the scale and complexity of training or fine-tuning data for large language model (LLM)-based recommendation systems make shadow models difficult to construct. Knowledge distillation as a method for extracting knowledge contributes to construct a stronger reference model. Knowledge distillation enables separate distillation for member and non-member data during the distillation process, enhancing the model's discriminative capability between the two in MIA. This paper propose a knowledge distillation-based MIA paradigm to improve the performance of membership inference attacks on LLM-based recommendation systems. Our paradigm introduces knowledge distillation to obtain a reference model, which enhances the reference model's ability to distinguish between member and non-member data. We obtain individual features from the reference model and train our attack model with fused feature. Our paradigm improves the attack performance of MIA compared to shadow model-based attack. 

---
# ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation 

**Authors**: Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.15141)  

**Abstract**: Recently, large language models (LLMs) have been widely used as recommender systems, owing to their strong reasoning capability and their effectiveness in handling cold-start items. To better adapt LLMs for recommendation, retrieval-augmented generation (RAG) has been incorporated. Most existing RAG methods are user-based, retrieving purchase patterns of users similar to the target user and providing them to the LLM. In this work, we propose ItemRAG, an item-based RAG method for LLM-based recommendation that retrieves relevant items (rather than users) from item-item co-purchase histories. ItemRAG helps LLMs capture co-purchase patterns among items, which are beneficial for recommendations. Especially, our retrieval strategy incorporates semantically similar items to better handle cold-start items and uses co-purchase frequencies to improve the relevance of the retrieved items. Through extensive experiments, we demonstrate that ItemRAG consistently (1) improves the zero-shot LLM-based recommender by up to 43% in Hit-Ratio-1 and (2) outperforms user-based RAG baselines under both standard and cold-start item recommendation settings. 

---
# Beyond GeneGPT: A Multi-Agent Architecture with Open-Source LLMs for Enhanced Genomic Question Answering 

**Authors**: Haodong Chen, Guido Zuccon, Teerapong Leelanupab  

**Link**: [PDF](https://arxiv.org/pdf/2511.15061)  

**Abstract**: Genomic question answering often requires complex reasoning and integration across diverse biomedical sources. GeneGPT addressed this challenge by combining domain-specific APIs with OpenAI's code-davinci-002 large language model to enable natural language interaction with genomic databases. However, its reliance on a proprietary model limits scalability, increases operational costs, and raises concerns about data privacy and generalization.
In this work, we revisit and reproduce GeneGPT in a pilot study using open source models, including Llama 3.1, Qwen2.5, and Qwen2.5 Coder, within a monolithic architecture; this allows us to identify the limitations of this approach. Building on this foundation, we then develop OpenBioLLM, a modular multi-agent framework that extends GeneGPT by introducing agent specialization for tool routing, query generation, and response validation. This enables coordinated reasoning and role-based task execution.
OpenBioLLM matches or outperforms GeneGPT on over 90% of the benchmark tasks, achieving average scores of 0.849 on Gene-Turing and 0.830 on GeneHop, while using smaller open-source models without additional fine-tuning or tool-specific pretraining. OpenBioLLM's modular multi-agent design reduces latency by 40-50% across benchmark tasks, significantly improving efficiency without compromising model capability. The results of our comprehensive evaluation highlight the potential of open-source multi-agent systems for genomic question answering. Code and resources are available at this https URL. 

---
# Multimodal Evaluation of Russian-language Architectures 

**Authors**: Artem Chervyakov, Ulyana Isaeva, Anton Emelyanov, Artem Safin, Maria Tikhonova, Alexander Kharitonov, Yulia Lyakh, Petr Surovtsev, Denis Shevelev Vildan Saburov, Vasily Konovalov, Elisei Rykov, Ivan Sviridov, Amina Miftakhova, Ilseyar Alimova, Alexander Panchenko, Alexander Kapitanov, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2511.15552)  

**Abstract**: Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family. 

---
# LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering 

**Authors**: Yuanjie Zhu, Liangwei Yang, Ke Xu, Weizhi Zhang, Zihe Song, Jindong Wang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15424)  

**Abstract**: Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering. 

---
# The Empowerment of Science of Science by Large Language Models: New Tools and Methods 

**Authors**: Guoqiang Liang, Jingqian Gong, Mengxuan Li, Gege Lin, Shuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15370)  

**Abstract**: Large language models (LLMs) have exhibited exceptional capabilities in natural language understanding and generation, image recognition, and multimodal tasks, charting a course towards AGI and emerging as a central issue in the global technological race. This manuscript conducts a comprehensive review of the core technologies that support LLMs from a user standpoint, including prompt engineering, knowledge-enhanced retrieval augmented generation, fine tuning, pretraining, and tool learning. Additionally, it traces the historical development of Science of Science (SciSci) and presents a forward looking perspective on the potential applications of LLMs within the scientometric domain. Furthermore, it discusses the prospect of an AI agent based model for scientific evaluation, and presents new research fronts detection and knowledge graph building methods with LLMs. 

---
# Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models 

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15304)  

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for large language models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of open-weight judge models and a human-validated stratified subset (with double-annotations to measure agreement). Disagreements were manually resolved. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols. 

---
# OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition 

**Authors**: Xinli Tao, Xin Dong, Xuezhong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.15211)  

**Abstract**: Clinical named entity recognition (NER) is crucial for extracting information from electronic health records (EHRs), but supervised models like CRF and BioClinicalBERT require costly annotated data. While zero-shot NER with large language models (LLMs) reduces this dependency, it struggles with example selection granularity and integrating prompts with self-improvement. To address this, we propose OEMA, a zero-shot clinical NER framework using multi-agent collaboration. OEMA's three components are: a self-annotator generating examples, a discriminator filtering them via SNOMED CT, and a predictor using entity descriptions for accurate inference. On MTSamples and VAERS datasets, OEMA achieves state-of-the-art exact-match performance. Under related-match, it matches supervised BioClinicalBERT and surpasses CRF. OEMA addresses key zero-shot NER challenges through ontology-guided reasoning and multi-agent collaboration, achieving near-supervised performance and showing promise for clinical NLP applications. 

---
# Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs 

**Authors**: Yang Wu, Rujing Yao, Tong Zhang, Yufei Shi, Zhuoren Jiang, Zhushan Li, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15163)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into intelligent tutoring systems to provide human-like and adaptive instruction. However, most existing approaches fail to capture how students' knowledge evolves dynamically across their proficiencies, conceptual gaps, and forgetting patterns. This challenge is particularly acute in mathematics tutoring, where effective instruction requires fine-grained scaffolding precisely calibrated to each student's mastery level and cognitive retention. To address this issue, we propose TASA (Teaching According to Students' Aptitude), a student-aware tutoring framework that integrates persona, memory, and forgetting dynamics for personalized mathematics learning. Specifically, TASA maintains a structured student persona capturing proficiency profiles and an event memory recording prior learning interactions. By incorporating a continuous forgetting curve with knowledge tracing, TASA dynamically updates each student's mastery state and generates contextually appropriate, difficulty-calibrated questions and explanations. Empirical results demonstrate that TASA achieves superior learning outcomes and more adaptive tutoring behavior compared to representative baselines, underscoring the importance of modeling temporal forgetting and learner profiles in LLM-based tutoring systems. 

---
# Mathematical Analysis of Hallucination Dynamics in Large Language Models: Uncertainty Quantification, Advanced Decoding, and Principled Mitigation 

**Authors**: Moses Kiprono  

**Link**: [PDF](https://arxiv.org/pdf/2511.15005)  

**Abstract**: Large Language Models (LLMs) are powerful linguistic engines but remain susceptible to hallucinations: plausible-sounding outputs that are factually incorrect or unsupported. In this work, we present a mathematically grounded framework to understand, measure, and mitigate these hallucinations. Drawing on probabilistic modeling, information theory, trigonometric signal analysis, and Bayesian uncertainty estimation, we analyze how errors compound autoregressively, propose refined uncertainty metrics, including semantic and phase-aware variants, and develop principled mitigation strategies such as contrastive decoding, retrieval-augmented grounding, factual alignment, and abstention. This unified lens connects recent advances in calibration, retrieval, and alignment to support safer and more reliable LLMs. 

---
# Temporal Predictors of Outcome in Reasoning Language Models 

**Authors**: Joey David  

**Link**: [PDF](https://arxiv.org/pdf/2511.14773)  

**Abstract**: The chain-of-thought (CoT) paradigm uses the elicitation of step-by-step rationales as a proxy for reasoning, gradually refining the model's latent representation of a solution. However, it remains unclear just how early a Large Language Model (LLM) internally commits to an eventual outcome. We probe this by training linear classifiers on hidden states after the first t reasoning tokens, showing that eventual correctness is highly predictable after only a few tokens, even when longer outputs are needed to reach a definite answer. We show that, for harder questions, a drop in predictive accuracy highlights a selection artifact: hard items are disproportionately represented in long CoTs. Overall, our results imply that for reasoning models, internal self-assessment of success tends to emerge after only a few tokens, with implications for interpretability and for inference-time control. 

---
# HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning 

**Authors**: Qihao Yang, Xuelin Wang, Jiale Chen, Xuelian Dong, Yuxin Hao, Tianyong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15574)  

**Abstract**: Language acquisition is vital to revealing the nature of human language intelligence and has recently emerged as a promising perspective for improving the interpretability of large language models (LLMs). However, it is ethically and practically infeasible to conduct experiments that require controlling human learners' language inputs. This poses challenges for the verifiability and scalability of language acquisition modeling, particularly in Chinese second language acquisition (SLA). While LLMs provide a controllable and reproducible alternative, a systematic benchmark to support phase-wise modeling and assessment is still lacking. In this paper, we present HSKBenchmark, the first benchmark for staged modeling and writing assessment of LLMs in Chinese SLA. It covers HSK levels 3 to 6 and includes authentic textbooks with 6.76 million tokens, 16K synthetic instruction samples, 30 test topics, and a linguistically grounded evaluation system. To simulate human learning trajectories, we introduce a curriculum-tuning framework that trains models from beginner to advanced levels. An evaluation system is created to examine level-based grammar coverage, writing errors, lexical and syntactic complexity, and holistic scoring. We also build HSKAgent, fine-tuned on 10K learner compositions. Extensive experimental results demonstrate that HSKBenchmark not only models Chinese SLA effectively, but also serves as a reliable benchmark for dynamic writing assessment in LLMs. Our fine-tuned LLMs have writing performance on par with advanced human learners and exhibit human-like acquisition characteristics. The HSKBenchmark, HSKAgent, and checkpoints serve as foundational tools and resources, with the potential to pave the way for future research on language acquisition modeling and LLMs interpretability. Code and data are publicly available at: this https URL. 

---
# When to Think and When to Look: Uncertainty-Guided Lookback 

**Authors**: Jing Bi, Filippos Bellos, Junjia Guo, Yayuan Li, Chao Huang, Yunlong, Tang, Luchuan Song, Susan Liang, Zhongfei, Zhang, Jason J. Corso, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15613)  

**Abstract**: Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets. 

---
# LiveCLKTBench: Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs 

**Authors**: Pei-Fu Guo, Yun-Da Tsai, Chun-Chia Hsu, Kai-Xin Chen, Ya-An Tsai, Kai-Wei Chang, Nanyun Peng, Mi-Yen Yeh, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.14774)  

**Abstract**: Evaluating cross-lingual knowledge transfer in large language models is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training. We present LiveCLKTBench, an automated generation pipeline specifically designed to isolate and measure cross-lingual knowledge transfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model's knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries. Using LiveCLKTBench, we evaluate several LLMs across five languages and observe that cross-lingual transfer is strongly influenced by linguistic distance and often asymmetric across language directions. While larger models improve transfer, the gains diminish with scale and vary across domains. These findings provide new insights into multilingual transfer and demonstrate the value of LiveCLKTBench as a reliable benchmark for future research. 

---
# Test-time Scaling of LLMs: A Survey from A Subproblem Structure Perspective 

**Authors**: Zhuoyi Yang, Xu Guo, Tong Zhang, Huijuan Xu, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14772)  

**Abstract**: With this paper, we survey techniques for improving the predictive accuracy of pretrained large language models by allocating additional compute at inference time. In categorizing test-time scaling methods, we place special emphasis on how a problem is decomposed into subproblems and on the topological organization of these subproblems whether sequential, parallel, or tree-structured. This perspective allows us to unify diverse approaches such as Chain-of-Thought, Branch-Solve-Merge, and Tree-of-Thought under a common lens. We further synthesize existing analyses of these techniques, highlighting their respective strengths and weaknesses, and conclude by outlining promising directions for future research 

---
# COMPASS: Context-Modulated PID Attention Steering System for Hallucination Mitigation 

**Authors**: Snigdha Pandya, Rohan Nagale, Kenji Sahay, Anna Lin, Shikhar Shiromani, Kevin Zhu, Dev Sunishchal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14776)  

**Abstract**: Large language models (LLMs) often generate fluent but factually incorrect statements despite having access to relevant evidence, a failure mode rooted in how they allocate attention between contextual and parametric knowledge. Understanding and steering this internal behavior is key both for trustworthy deployment and for scientific interpretability of model mechanisms. We introduce COMPASS (Context-Modulated PID Attention Steering System), a lightweight, interpretable control framework that embeds a model-based feedback loop directly within decoding. COMPASS quantifies context reliance via a transparent metric, the Context Reliance Score (CRS), which serves as an online probe of how attention heads ground generation in evidence. Using this interpretable signal, a PID controller dynamically modulates attention heads to maintain factual consistency without retraining or multi-pass decoding. Across benchmarks (HotpotQA, XSum, HaluEval, RAGTruth), COMPASS consistently reduces contextual hallucination rates (2.8 to 5.8 percent absolute) while revealing how distinct attention heads contribute to evidence alignment. These results highlight feedback-driven interpretability as a pathway toward scientific understanding of LLM behavior. 

---
# Evaluating Multimodal Large Language Models on Vertically Written Japanese Text 

**Authors**: Keito Sasagawa, Shuhei Kurita, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2511.15059)  

**Abstract**: Multimodal Large Language Models (MLLMs) have seen rapid advances in recent years and are now being applied to visual document understanding tasks. They are expected to process a wide range of document images across languages, including Japanese. Understanding documents from images requires models to read what are written in them. Since some Japanese documents are written vertically, support for vertical writing is essential. However, research specifically focused on vertically written Japanese text remains limited. In this study, we evaluate the reading capability of existing MLLMs on vertically written Japanese text. First, we generate a synthetic Japanese OCR dataset by rendering Japanese texts into images, and use it for both model fine-tuning and evaluation. This dataset includes Japanese text in both horizontal and vertical writing. We also create an evaluation dataset sourced from the real-world document images containing vertically written Japanese text. Using these datasets, we demonstrate that the existing MLLMs perform worse on vertically written Japanese text than on horizontally written Japanese text. Furthermore, we show that training MLLMs on our synthesized Japanese OCR dataset results in improving the performance of models that previously could not handle vertical writing. The datasets and code are publicly available this https URL. 

---
# ProRAC: A Neuro-symbolic Method for Reasoning about Actions with LLM-based Progression 

**Authors**: Haoyong Wu, Yongmei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15069)  

**Abstract**: In this paper, we propose ProRAC (Progression-based Reasoning about Actions and Change), a neuro-symbolic framework that leverages LLMs to tackle RAC problems. ProRAC extracts fundamental RAC elements including actions and questions from the problem, progressively executes each action to derive the final state, and then evaluates the query against the progressed state to arrive at an answer. We evaluate ProRAC on several RAC benchmarks, and the results demonstrate that our approach achieves strong performance across different benchmarks, domains, LLM backbones, and types of RAC tasks. 

---
# Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization 

**Authors**: Yifeng Ding, Hung Le, Songyang Han, Kangrui Ruan, Zhenghui Jin, Varun Kumar, Zijian Wang, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2511.14846)  

**Abstract**: Training Large Language Models (LLMs) for multi-turn Tool-Integrated Reasoning (TIR) - where models iteratively reason, generate code, and verify through execution - remains challenging for existing reinforcement learning (RL) approaches. Current RL methods, exemplified by Group Relative Policy Optimization (GRPO), suffer from coarse-grained, trajectory-level rewards that provide insufficient learning signals for complex multi-turn interactions, leading to training stagnation. To address this issue, we propose Group Turn Policy Optimization (GTPO), a novel RL algorithm specifically designed for training LLMs on multi-turn TIR tasks. GTPO introduces three key innovations: (1) turn-level reward assignment that provides fine-grained feedback for individual turns, (2) return-based advantage estimation where normalized discounted returns are calculated as advantages, and (3) self-supervised reward shaping that exploits self-supervision signals from generated code to densify sparse binary outcome-based rewards. Our comprehensive evaluation demonstrates that GTPO outperforms GRPO by 3.0% on average across diverse reasoning benchmarks, establishing its effectiveness for advancing complex mathematical reasoning in the real world. 

---
# Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents 

**Authors**: Henrik Bradland, Morten Goodwin, Vladimir I. Zadorozhny, Per-Arne Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15074)  

**Abstract**: The performance of machine learning models on tabular data is critically dependent on high-quality feature engineering. While Large Language Models (LLMs) have shown promise in automating feature extraction (AutoFE), existing methods are often limited by monolithic LLM architectures, simplistic quantitative feedback, and a failure to systematically integrate external domain knowledge. This paper introduces Rogue One, a novel, LLM-based multi-agent framework for knowledge-informed automatic feature extraction. Rogue One operationalizes a decentralized system of three specialized agents-Scientist, Extractor, and Tester-that collaborate iteratively to discover, generate, and validate predictive features. Crucially, the framework moves beyond primitive accuracy scores by introducing a rich, qualitative feedback mechanism and a "flooding-pruning" strategy, allowing it to dynamically balance feature exploration and exploitation. By actively incorporating external knowledge via an integrated retrieval-augmented (RAG) system, Rogue One generates features that are not only statistically powerful but also semantically meaningful and interpretable. We demonstrate that Rogue One significantly outperforms state-of-the-art methods on a comprehensive suite of 19 classification and 9 regression datasets. Furthermore, we show qualitatively that the system surfaces novel, testable hypotheses, such as identifying a new potential biomarker in the myocardial dataset, underscoring its utility as a tool for scientific discovery. 

---
# SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making 

**Authors**: Yinsheng Wang, Tario G You, Léonard Boussioux, Shan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15202)  

**Abstract**: This paper introduces SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making), a novel framework that integrates mathematical optimization with the contextual capabilities of large language models (LLMs). SOLID facilitates iterative collaboration between optimization and LLMs agents through dual prices and deviation penalties. This interaction improves the quality of the decisions while maintaining modularity and data privacy. The framework retains theoretical convergence guarantees under convexity assumptions, providing insight into the design of LLMs prompt. To evaluate SOLID, we applied it to a stock portfolio investment case with historical prices and financial news as inputs. Empirical results demonstrate convergence under various scenarios and indicate improved annualized returns compared to a baseline optimizer-only method, validating the synergy of the two agents. SOLID offers a promising framework for advancing automated and intelligent decision-making across diverse domains. 

---
# HISE-KT: Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization 

**Authors**: Zhiyi Duan, Zixing Shi, Hongyu Yuan, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15191)  

**Abstract**: Knowledge Tracing (KT) aims to mine students' evolving knowledge states and predict their future question-answering performance. Existing methods based on heterogeneous information networks (HINs) are prone to introducing noises due to manual or random selection of meta-paths and lack necessary quality assessment of meta-path instances. Conversely, recent large language models (LLMs)-based methods ignore the rich information across students, and both paradigms struggle to deliver consistently accurate and evidence-based explanations. To address these issues, we propose an innovative framework, HIN-LLM Synergistic Enhanced Knowledge Tracing (HISE-KT), which seamlessly integrates HINs with LLMs. HISE-KT first builds a multi-relationship HIN containing diverse node types to capture the structural relations through multiple meta-paths. The LLM is then employed to intelligently score and filter meta-path instances and retain high-quality paths, pioneering automated meta-path quality assessment. Inspired by educational psychology principles, a similar student retrieval mechanism based on meta-paths is designed to provide a more valuable context for prediction. Finally, HISE-KT uses a structured prompt to integrate the target student's history with the retrieved similar trajectories, enabling the LLM to generate not only accurate predictions but also evidence-backed, explainable analysis reports. Experiments on four public datasets show that HISE-KT outperforms existing KT baselines in both prediction performance and interpretability. 

---
# SafeRBench: A Comprehensive Benchmark for Safety Assessment in Large Reasoning Models 

**Authors**: Xin Gao, Shaohan Yu, Zerui Chen, Yueming Lyu, Weichen Yu, Guanghao Li, Jiyao Liu, Jianxiong Gao, Jian Liang, Ziwei Liu, Chenyang Si  

**Link**: [PDF](https://arxiv.org/pdf/2511.15169)  

**Abstract**: Large Reasoning Models (LRMs) improve answer quality through explicit chain-of-thought, yet this very capability introduces new safety risks: harmful content can be subtly injected, surface gradually, or be justified by misleading rationales within the reasoning trace. Existing safety evaluations, however, primarily focus on output-level judgments and rarely capture these dynamic risks along the reasoning process. In this paper, we present SafeRBench, the first benchmark that assesses LRM safety end-to-end -- from inputs and intermediate reasoning to final outputs. (1) Input Characterization: We pioneer the incorporation of risk categories and levels into input design, explicitly accounting for affected groups and severity, and thereby establish a balanced prompt suite reflecting diverse harm gradients. (2) Fine-Grained Output Analysis: We introduce a micro-thought chunking mechanism to segment long reasoning traces into semantically coherent units, enabling fine-grained evaluation across ten safety dimensions. (3) Human Safety Alignment: We validate LLM-based evaluations against human annotations specifically designed to capture safety judgments. Evaluations on 19 LRMs demonstrate that SafeRBench enables detailed, multidimensional safety assessment, offering insights into risks and protective mechanisms from multiple perspectives. 

---
# As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files 

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15192)  

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.
We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency. 

---
# The Illusion of Procedural Reasoning: Measuring Long-Horizon FSM Execution in LLMs 

**Authors**: Mahdi Samiei, Mahdi Mansouri, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2511.14777)  

**Abstract**: Large language models (LLMs) have achieved remarkable results on tasks framed as reasoning problems, yet their true ability to perform procedural reasoning, executing multi-step, rule-based computations remains unclear. Unlike algorithmic systems, which can deterministically execute long-horizon symbolic procedures, LLMs often degrade under extended reasoning chains, but there is no controlled, interpretable benchmark to isolate and measure this collapse. We introduce Finite-State Machine (FSM) Execution as a minimal, fully interpretable framework for evaluating the procedural reasoning capacity of LLMs. In our setup, the model is given an explicit FSM definition and must execute it step-by-step given input actions, maintaining state consistency over multiple turns. This task requires no world knowledge, only faithful application of deterministic transition rules, making it a direct probe of the model's internal procedural fidelity. We measure both Turn Accuracy and Task Accuracy to disentangle immediate computation from cumulative state maintenance. Empirical results reveal systematic degradation as task horizon or branching complexity increases. Models perform significantly worse when rule retrieval involves high branching factors than when memory span is long. Larger models show improved local accuracy but remain brittle under multi-step reasoning unless explicitly prompted to externalize intermediate steps. FSM-based evaluation offers a transparent, complexity-controlled probe for diagnosing this failure mode and guiding the design of inductive biases that enable genuine long-horizon procedural competence. By grounding reasoning in measurable execution fidelity rather than surface correctness, this work helps establish a rigorous experimental foundation for understanding and improving the algorithmic reliability of LLMs. 

---
# Subnational Geocoding of Global Disasters Using Large Language Models 

**Authors**: Michele Ronco, Damien Delforge, Wiebke S. Jäger, Christina Corbane  

**Link**: [PDF](https://arxiv.org/pdf/2511.14788)  

**Abstract**: Subnational location data of disaster events are critical for risk assessment and disaster risk reduction. Disaster databases such as EM-DAT often report locations in unstructured textual form, with inconsistent granularity or spelling, that make it difficult to integrate with spatial datasets. We present a fully automated LLM-assisted workflow that processes and cleans textual location information using GPT-4o, and assigns geometries by cross-checking three independent geoinformation repositories: GADM, OpenStreetMap and Wikidata. Based on the agreement and availability of these sources, we assign a reliability score to each location while generating subnational geometries. Applied to the EM-DAT dataset from 2000 to 2024, the workflow geocodes 14,215 events across 17,948 unique locations. Unlike previous methods, our approach requires no manual intervention, covers all disaster types, enables cross-verification across multiple sources, and allows flexible remapping to preferred frameworks. Beyond the dataset, we demonstrate the potential of LLMs to extract and structure geographic information from unstructured text, offering a scalable and reliable method for related analyses. 

---
# Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining 

**Authors**: Qian'ang Mao, Yuxuan Zhang, Jiaman Chen, Wenjun Zhou, Jiaqi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15456)  

**Abstract**: As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity. 

---
# Learning Interestingness in Automated Mathematical Theory Formation 

**Authors**: George Tsoukalas, Rahul Saha, Amitayush Thakur, Sabrina Reguyal, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2511.14778)  

**Abstract**: We take two key steps in automating the open-ended discovery of new mathematical theories, a grand challenge in artificial intelligence. First, we introduce $\emph{FERMAT}$, a reinforcement learning (RL) environment that models concept discovery and theorem-proving using a set of symbolic actions, opening up a range of RL problems relevant to theory discovery. Second, we explore a specific problem through $\emph{FERMAT}$: automatically scoring the $\emph{interestingness}$ of mathematical objects. We investigate evolutionary algorithms for synthesizing nontrivial interestingness measures. In particular, we introduce an LLM-based evolutionary algorithm that features function abstraction, leading to notable improvements in discovering elementary number theory and finite fields over hard-coded baselines. We open-source the $\emph{FERMAT}$ environment at this URL(this https URL). 

---
# EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control 

**Authors**: Kai Yang, Xin Xu, Yangkun Chen, Weijie Liu, Jiafei Lyu, Zichuan Lin, Deheng Ye, Saiyong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15248)  

**Abstract**: Long-term training of large language models (LLMs) requires maintaining stable exploration to prevent the model from collapsing into sub-optimal behaviors. Entropy is crucial in this context, as it controls exploration and helps avoid premature convergence to sub-optimal solutions. However, existing reinforcement learning methods struggle to maintain an appropriate level of entropy, as the training process involves a mix of positive and negative samples, each affecting entropy in different ways across steps. To address this, we propose Entropy stablilization via Proportional-Integral Control (EntroPIC), a novel method that adaptively adjusts the influence of positive and negative samples by dynamically tuning their loss coefficients. This approach stabilizes entropy throughout training, ensuring efficient exploration and steady progress. We provide a comprehensive theoretical analysis for both on-policy and off-policy learning settings, demonstrating that EntroPIC is effective at controlling entropy in large-scale LLM training. Experimental results show that our method successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs. 

---
# Physics-Based Benchmarking Metrics for Multimodal Synthetic Images 

**Authors**: Kishor Datta Gupta, Marufa Kamal, Md. Mahfuzur Rahman, Fahad Rahman, Mohd Ariful Haque, Sunzida Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2511.15204)  

**Abstract**: Current state of the art measures like BLEU, CIDEr, VQA score, SigLIP-2 and CLIPScore are often unable to capture semantic or structural accuracy, especially for domain-specific or context-dependent scenarios. For this, this paper proposes a Physics-Constrained Multimodal Data Evaluation (PCMDE) metric combining large language models with reasoning, knowledge based mapping and vision-language models to overcome these limitations. The architecture is comprised of three main stages: (1) feature extraction of spatial and semantic information with multimodal features through object detection and VLMs; (2) Confidence-Weighted Component Fusion for adaptive component-level validation; and (3) physics-guided reasoning using large language models for structural and relational constraints (e.g., alignment, position, consistency) enforcement. 

---
# Finetuning LLMs for Automatic Form Interaction on Web-Browser in Selenium Testing Framework 

**Authors**: Nguyen-Khang Le, Nguyen Hiep, Minh Nguyen, Son Luu, Trung Vo, Quan Bui, Nomura Shoshin, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15168)  

**Abstract**: Automated web application testing is a critical component of modern software development, with frameworks like Selenium widely adopted for validating functionality through browser automation. Among the essential aspects of such testing is the ability to interact with and validate web forms, a task that requires syntactically correct, executable scripts with high coverage of input fields. Despite its importance, this task remains underexplored in the context of large language models (LLMs), and no public benchmark or dataset exists to evaluate LLMs on form interaction generation systematically. This paper introduces a novel method for training LLMs to generate high-quality test cases in Selenium, specifically targeting form interaction testing. We curate both synthetic and human-annotated datasets for training and evaluation, covering diverse real-world forms and testing scenarios. We define clear metrics for syntax correctness, script executability, and input field coverage. Our empirical study demonstrates that our approach significantly outperforms strong baselines, including GPT-4o and other popular LLMs, across all evaluation metrics. Our work lays the groundwork for future research on LLM-based web testing and provides resources to support ongoing progress in this area. 

---
# From Solving to Verifying: A Unified Objective for Robust Reasoning in LLMs 

**Authors**: Xiaoxuan Wang, Bo Liu, Song Jiang, Jingzhou Liu, Jingyuan Qi, Xia Chen, Baosheng He  

**Link**: [PDF](https://arxiv.org/pdf/2511.15137)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have been significantly improved through reinforcement learning (RL). Nevertheless, LLMs still struggle to consistently verify their own reasoning traces. This raises the research question of how to enhance the self-verification ability of LLMs and whether such an ability can further improve reasoning performance. In this work, we propose GRPO-Verif, an algorithm that jointly optimizes solution generation and self-verification within a unified loss function, with an adjustable hyperparameter controlling the weight of the verification signal. Experimental results demonstrate that our method enhances self-verification capability while maintaining comparable performance in reasoning. 

---
# MermaidSeqBench: An Evaluation Benchmark for LLM-to-Mermaid Sequence Diagram Generation 

**Authors**: Basel Shbita, Farhan Ahmed, Chad DeLuca  

**Link**: [PDF](https://arxiv.org/pdf/2511.14967)  

**Abstract**: Large language models (LLMs) have demonstrated excellent capabilities in generating structured diagrams from natural language descriptions. In particular, they have shown great promise in generating sequence diagrams for software engineering, typically represented in a text-based syntax such as Mermaid. However, systematic evaluations in this space remain underdeveloped as there is a lack of existing benchmarks to assess the LLM's correctness in this task. To address this shortcoming, we introduce MermaidSeqBench, a human-verified and LLM-synthetically-extended benchmark for assessing an LLM's capabilities in generating Mermaid sequence diagrams from textual prompts. The benchmark consists of a core set of 132 samples, starting from a small set of manually crafted and verified flows. These were expanded via a hybrid methodology combining human annotation, in-context LLM prompting, and rule-based variation generation. Our benchmark uses an LLM-as-a-judge model to assess Mermaid sequence diagram generation across fine-grained metrics, including syntax correctness, activation handling, error handling, and practical usability. We perform initial evaluations on numerous state-of-the-art LLMs and utilize multiple LLM judge models to demonstrate the effectiveness and flexibility of our benchmark. Our results reveal significant capability gaps across models and evaluation modes. Our proposed benchmark provides a foundation for advancing research in structured diagram generation and for developing more rigorous, fine-grained evaluation methodologies. 

---
# Scalable and Efficient Large-Scale Log Analysis with LLMs: An IT Software Support Case Study 

**Authors**: Pranjal Gupta, Karan Bhukar, Harshit Kumar, Seema Nagar, Prateeti Mohapatra, Debanjana Kar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14803)  

**Abstract**: IT environments typically have logging mechanisms to monitor system health and detect issues. However, the huge volume of generated logs makes manual inspection impractical, highlighting the importance of automated log analysis in IT Software Support. In this paper, we propose a log analytics tool that leverages Large Language Models (LLMs) for log data processing and issue diagnosis, enabling the generation of automated insights and summaries. We further present a novel approach for efficiently running LLMs on CPUs to process massive log volumes in minimal time without compromising output quality. We share the insights and lessons learned from deployment of the tool - in production since March 2024 - scaled across 70 software products, processing over 2000 tickets for issue diagnosis, achieving a time savings of 300+ man hours and an estimated $15,444 per month in manpower costs compared to the traditional log analysis practices. 

---
# Evaluating Generative AI for CS1 Code Grading: Direct vs Reverse Methods 

**Authors**: Ahmad Memon, Abdallah Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2511.14798)  

**Abstract**: Manual grading of programming assignments in introductory computer science courses can be time-consuming and prone to inconsistencies. While unit testing is commonly used for automatic evaluation, it typically follows a binary pass/fail model and does not give partial marks. Recent advances in large language models (LLMs) offer the potential for automated, scalable, and more objective grading.
This paper compares two AI-based grading techniques: \textit{Direct}, where the AI model applies a rubric directly to student code, and \textit{Reverse} (a newly proposed approach), where the AI first fixes errors, then deduces a grade based on the nature and number of fixes. Each method was evaluated on both the instructor's original grading scale and a tenfold expanded scale to assess the impact of range on AI grading accuracy. To assess their effectiveness, AI-assigned scores were evaluated against human tutor evaluations on a range of coding problems and error types.
Initial findings suggest that while the Direct approach is faster and straightforward, the Reverse technique often provides a more fine-grained assessment by focusing on correction effort. Both methods require careful prompt engineering, particularly for allocating partial credit and handling logic errors. To further test consistency, we also used synthetic student code generated using Gemini Flash 2.0, which allowed us to evaluate AI graders on a wider range of controlled error types and difficulty levels. We discuss the strengths and limitations of each approach, practical considerations for prompt design, and future directions for hybrid human-AI grading systems that aim to improve consistency, efficiency, and fairness in CS courses. 

---
# irace-evo: Automatic Algorithm Configuration Extended With LLM-Based Code Evolution 

**Authors**: Camilo Chacón Sartori, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2511.14794)  

**Abstract**: Automatic algorithm configuration tools such as irace efficiently tune parameter values but leave algorithmic code unchanged. This paper introduces a first version of irace-evo, an extension of irace that integrates code evolution through large language models (LLMs) to jointly explore parameter and code spaces. The proposed framework enables multi-language support (e.g., C++, Python), reduces token consumption via progressive context management, and employs the Always-From-Original principle to ensure robust and controlled code evolution. We evaluate irace-evo on the Construct, Merge, Solve & Adapt (CMSA) metaheuristic for the Variable-Sized Bin Packing Problem (VSBPP). Experimental results show that irace-evo can discover new algorithm variants that outperform the state-of-the-art CMSA implementation while maintaining low computational and monetary costs. Notably, irace-evo generates competitive algorithmic improvements using lightweight models (e.g., Claude Haiku 3.5) with a total usage cost under 2 euros. These results demonstrate that coupling automatic configuration with LLM-driven code evolution provides a powerful, cost-efficient avenue for advancing heuristic design and metaheuristic optimization. 

---
# TacEleven: generative tactic discovery for football open play 

**Authors**: Siyao Zhao, Hao Ma, Zhiqiang Pu, Jingjing Huang, Yi Pan, Shijie Wang, Zhi Ming  

**Link**: [PDF](https://arxiv.org/pdf/2511.13326)  

**Abstract**: Creating offensive advantages during open play is fundamental to football success. However, due to the highly dynamic and long-sequence nature of open play, the potential tactic space grows exponentially as the sequence progresses, making automated tactic discovery extremely challenging. To address this, we propose TacEleven, a generative framework for football open-play tactic discovery developed in close collaboration with domain experts from AJ Auxerre, designed to assist coaches and analysts in tactical decision-making. TacEleven consists of two core components: a language-controlled tactical generator that produces diverse tactical proposals, and a multimodal large language model-based tactical critic that selects the optimal proposal aligned with a high-level stylistic tactical instruction. The two components enables rapid exploration of tactical proposals and discovery of alternative open-play offensive tactics. We evaluate TacEleven across three tasks with progressive tactical complexity: counterfactual exploration, single-step discovery, and multi-step discovery, through both quantitative metrics and a questionnaire-based qualitative assessment. The results show that the TacEleven-discovered tactics exhibit strong realism and tactical creativity, with 52.50% of the multi-step tactical alternatives rated adoptable in real-world elite football scenarios, highlighting the framework's ability to rapidly generate numerous high-quality tactics for complex long-sequence open-play situations. TacEleven demonstrates the potential of creatively leveraging domain data and generative models to advance tactical analysis in sports. 

---
# ESA: Energy-Based Shot Assembly Optimization for Automatic Video Editing 

**Authors**: Yaosen Chen, Wei Wang, Tianheng Zheng, Xuming Wen, Han Yang, Yanru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02505)  

**Abstract**: Shot assembly is a crucial step in film production and video editing, involving the sequencing and arrangement of shots to construct a narrative, convey information, or evoke emotions. Traditionally, this process has been manually executed by experienced editors. While current intelligent video editing technologies can handle some automated video editing tasks, they often fail to capture the creator's unique artistic expression in shot assembly. To address this challenge, we propose an energy-based optimization method for video shot assembly. Specifically, we first perform visual-semantic matching between the script generated by a large language model and a video library to obtain subsets of candidate shots aligned with the script semantics. Next, we segment and label the shots from reference videos, extracting attributes such as shot size, camera motion, and semantics. We then employ energy-based models to learn from these attributes, scoring candidate shot sequences based on their alignment with reference styles. Finally, we achieve shot assembly optimization by combining multiple syntax rules, producing videos that align with the assembly style of the reference videos. Our method not only automates the arrangement and combination of independent shots according to specific logic, narrative requirements, or artistic styles but also learns the assembly style of reference videos, creating a coherent visual sequence or holistic visual expression. With our system, even users with no prior video editing experience can create visually compelling videos. Project page: this https URL 

---
