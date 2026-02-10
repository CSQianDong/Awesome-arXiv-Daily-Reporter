# When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents 

**Authors**: Yuting Ning, Jaylen Jones, Zhehao Zhang, Chentao Ye, Weitong Ruan, Junyi Li, Rahul Gupta, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.08995)  

**Abstract**: Computer-use agents (CUAs) have made tremendous progress in the past year, yet they still frequently produce misaligned actions that deviate from the user's original intent. Such misaligned actions may arise from external attacks (e.g., indirect prompt injection) or from internal limitations (e.g., erroneous reasoning). They not only expose CUAs to safety risks, but also degrade task efficiency and reliability. This work makes the first effort to define and study misaligned action detection in CUAs, with comprehensive coverage of both externally induced and internally arising misaligned actions. We further identify three common categories in real-world CUA deployment and construct MisActBench, a benchmark of realistic trajectories with human-annotated, action-level alignment labels. Moreover, we propose DeAction, a practical and universal guardrail that detects misaligned actions before execution and iteratively corrects them through structured feedback. DeAction outperforms all existing baselines across offline and online evaluations with moderate latency overhead: (1) On MisActBench, it outperforms baselines by over 15% absolute in F1 score; (2) In online evaluation, it reduces attack success rate by over 90% under adversarial settings while preserving or even improving task success rate in benign environments. 

---
# Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models 

**Authors**: Yuliang Liu, Yunchong Song, Yixuan Wang, Kewen Ge, Alex Lamb, Qipeng Guo, Kai Chen, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08984)  

**Abstract**: We propose Next Concept Prediction (NCP), a generative pretraining paradigm built on top of Next Token Prediction (NTP). NCP predicts discrete concepts that span multiple tokens, thereby forming a more challenging pretraining objective. Our model, ConceptLM, quantizes hidden states using Vector Quantization and constructs a concept vocabulary. It leverages both NCP and NTP to drive parameter updates and generates a concept to guide the generation of the following tokens. We train ConceptLM from scratch at scales ranging from 70M to 1.5B parameters with up to 300B training data, including Pythia and GPT-2 backbones. Results on 13 benchmarks show that NCP yields consistent performance gains over traditional token-level models. Furthermore, continual pretraining experiments on an 8B-parameter Llama model indicate that NCP can further improve an NTP-trained model. Our analysis suggests that NCP leads to more powerful language models by introducing a harder pretraining task, providing a promising path toward better language modeling. 

---
# How Should We Model the Probability of a Language? 

**Authors**: Rasul Dent, Pedro Ortiz Suarez, Thibault Clérice, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2602.08951)  

**Abstract**: Of the over 7,000 languages spoken in the world, commercial language identification (LID) systems only reliably identify a few hundred in written form. Research-grade systems extend this coverage under certain circumstances, but for most languages coverage remains patchy or nonexistent. This position paper argues that this situation is largely self-imposed. In particular, it arises from a persistent framing of LID as decontextualized text classification, which obscures the central role of prior probability estimation and is reinforced by institutional incentives that favor global, fixed-prior models. We argue that improving coverage for tail languages requires rethinking LID as a routing problem and developing principled ways to incorporate environmental cues that make languages locally plausible. 

---
# GitSearch: Enhancing Community Notes Generation with Gap-Informed Targeted Search 

**Authors**: Sahajpreet Singh, Kokil Jaidka, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2602.08945)  

**Abstract**: Community-based moderation offers a scalable alternative to centralized fact-checking, yet it faces significant structural challenges, and existing AI-based methods fail in "cold start" scenarios. To tackle these challenges, we introduce GitSearch (Gap-Informed Targeted Search), a framework that treats human-perceived quality gaps, such as missing context, etc., as first-class signals. GitSearch has a three-stage pipeline: identifying information deficits, executing real-time targeted web-retrieval to resolve them, and synthesizing platform-compliant notes. To facilitate evaluation, we present PolBench, a benchmark of 78,698 U.S. political tweets with their associated Community Notes. We find GitSearch achieves 99% coverage, almost doubling coverage over the state-of-the-art. GitSearch surpasses human-authored helpful notes with a 69% win rate and superior helpfulness scores (3.87 vs. 3.36), demonstrating retrieval effectiveness that balanced the trade-off between scale and quality. 

---
# Is Reasoning Capability Enough for Safety in Long-Context Language Models? 

**Authors**: Yu Fu, Haz Sameen Shahgir, Huanli Gong, Zhipeng Wei, N. Benjamin Erichson, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2602.08874)  

**Abstract**: Large language models (LLMs) increasingly combine long-context processing with advanced reasoning, enabling them to retrieve and synthesize information distributed across tens of thousands of tokens. A hypothesis is that stronger reasoning capability should improve safety by helping models recognize harmful intent even when it is not stated explicitly. We test this hypothesis in long-context settings where harmful intent is implicit and must be inferred through reasoning, and find that it does not hold. We introduce compositional reasoning attacks, a new threat model in which a harmful query is decomposed into incomplete fragments that scattered throughout a long context. The model is then prompted with a neutral reasoning query that induces retrieval and synthesis, causing the harmful intent to emerge only after composition. Evaluating 14 frontier LLMs on contexts up to 64k tokens, we uncover three findings: (1) models with stronger general reasoning capability are not more robust to compositional reasoning attacks, often assembling the intent yet failing to refuse; (2) safety alignment consistently degrades as context length increases; and (3) inference-time reasoning effort is a key mitigating factor: increasing inference-time compute reduces attack success by over 50 percentage points on GPT-oss-120b model. Together, these results suggest that safety does not automatically scale with reasoning capability, especially under long-context inference. 

---
# Large Language Models for Geolocation Extraction in Humanitarian Crisis Response 

**Authors**: G. Cafferata, T. Demarco, K. Kalimeri, Y. Mejova, M.G. Beiró  

**Link**: [PDF](https://arxiv.org/pdf/2602.08872)  

**Abstract**: Humanitarian crises demand timely and accurate geographic information to inform effective response efforts. Yet, automated systems that extract locations from text often reproduce existing geographic and socioeconomic biases, leading to uneven visibility of crisis-affected regions. This paper investigates whether Large Language Models (LLMs) can address these geographic disparities in extracting location information from humanitarian documents. We introduce a two-step framework that combines few-shot LLM-based named entity recognition with an agent-based geocoding module that leverages context to resolve ambiguous toponyms. We benchmark our approach against state-of-the-art pretrained and rule-based systems using both accuracy and fairness metrics across geographic and socioeconomic dimensions. Our evaluation uses an extended version of the HumSet dataset with refined literal toponym annotations. Results show that LLM-based methods substantially improve both the precision and fairness of geolocation extraction from humanitarian texts, particularly for underrepresented regions. By bridging advances in LLM reasoning with principles of responsible and inclusive AI, this work contributes to more equitable geospatial data systems for humanitarian response, advancing the goal of leaving no place behind in crisis analytics. 

---
# Understanding Dynamic Compute Allocation in Recurrent Transformers 

**Authors**: Ibraheem Muhammad Moosa, Suhas Lohit, Ye Wang, Moitreya Chatterjee, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08864)  

**Abstract**: Token-level adaptive computation seeks to reduce inference cost by allocating more computation to harder tokens and less to easier ones. However, prior work is primarily evaluated on natural-language benchmarks using task-level metrics, where token-level difficulty is unobservable and confounded with architectural factors, making it unclear whether compute allocation truly aligns with underlying complexity. We address this gap through three contributions. First, we introduce a complexity-controlled evaluation paradigm using algorithmic and synthetic language tasks with parameterized difficulty, enabling direct testing of token-level compute allocation. Second, we propose ANIRA, a unified recurrent Transformer framework that supports per-token variable-depth computation while isolating compute allocation decisions from other model factors. Third, we use this framework to conduct a systematic analysis of token-level adaptive computation across alignment with complexity, generalization, and decision timing. Our results show that compute allocation aligned with task complexity can emerge without explicit difficulty supervision, but such alignment does not imply algorithmic generalization: models fail to extrapolate to unseen input sizes despite allocating additional computation. We further find that early compute decisions rely on static structural cues, whereas online halting more closely tracks algorithmic execution state. 

---
# WildReward: Learning Reward Models from In-the-Wild Human Interactions 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08829)  

**Abstract**: Reward models (RMs) are crucial for the training of large language models (LLMs), yet they typically rely on large-scale human-annotated preference pairs. With the widespread deployment of LLMs, in-the-wild interactions have emerged as a rich source of implicit reward signals. This raises the question: Can we develop reward models directly from in-the-wild interactions? In this work, we explore this possibility by adopting WildChat as an interaction source and proposing a pipeline to extract reliable human feedback, yielding 186k high-quality instances for training WildReward via ordinal regression directly on user feedback without preference pairs. Extensive experiments demonstrate that WildReward achieves comparable or even superior performance compared to conventional reward models, with improved calibration and cross-sample consistency. We also observe that WildReward benefits directly from user diversity, where more users yield stronger reward models. Finally, we apply WildReward to online DPO training and observe significant improvements across various tasks. Code and data are released at this https URL. 

---
# Affective Flow Language Model for Emotional Support Conversation 

**Authors**: Chenghui Zou, Ning Wang, Tiesunlong Shen, Luwei Xiao, Chuan Ma, Xiangpeng Li, Rui Mao, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2602.08826)  

**Abstract**: Large language models (LLMs) have been widely applied to emotional support conversation (ESC). However, complex multi-turn support remains this http URL is because existing alignment schemes rely on sparse outcome-level signals, thus offering limited supervision for intermediate strategy decisions. To fill this gap, this paper proposes affective flow language model for emotional support conversation (AFlow), a framework that introduces fine-grained supervision on dialogue prefixes by modeling a continuous affective flow along multi-turn trajectories. AFlow can estimate intermediate utility over searched trajectories and learn preference-consistent strategy transitions. To improve strategy coherence and empathetic response quality, a subpath-level flow-balance objective is presented to propagate preference signals to intermediate states. Experiment results show consistent and significant improvements over competitive baselines in diverse emotional contexts. Remarkably, AFlow with a compact open-source backbone outperforms proprietary LMMs such as GPT-4o and Claude-3.5 on major ESC metrics. Our code is available at this https URL. 

---
# LakeHopper: Cross Data Lakes Column Type Annotation through Model Adaptation 

**Authors**: Yushi Sun, Xujia Li, Nan Tang, Quanqing Xu, Chuanhui Yang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08793)  

**Abstract**: Column type annotation is vital for tasks like data cleaning, integration, and visualization. Recent solutions rely on resource-intensive language models fine-tuned on well-annotated columns from a particular set of tables, i.e., a source data lake. In this paper, we study whether we can adapt an existing pre-trained LM-based model to a new (i.e., target) data lake to minimize the annotations required on the new data lake. However, challenges include the source-target knowledge gap, selecting informative target data, and fine-tuning without losing shared knowledge exist. We propose LakeHopper, a framework that identifies and resolves the knowledge gap through LM interactions, employs a cluster-based data selection scheme for unannotated columns, and uses an incremental fine-tuning mechanism that gradually adapts the source model to the target data lake. Our experimental results validate the effectiveness of LakeHopper on two different data lake transfers under both low-resource and high-resource settings. 

---
# Map of Encoders -- Mapping Sentence Encoders using Quantum Relative Entropy 

**Authors**: Gaifan Zhang, Danushka Bollegala  

**Link**: [PDF](https://arxiv.org/pdf/2602.08740)  

**Abstract**: We propose a method to compare and visualise sentence encoders at scale by creating a map of encoders where each sentence encoder is represented in relation to the other sentence encoders. Specifically, we first represent a sentence encoder using an embedding matrix of a sentence set, where each row corresponds to the embedding of a sentence. Next, we compute the Pairwise Inner Product (PIP) matrix for a sentence encoder using its embedding matrix. Finally, we create a feature vector for each sentence encoder reflecting its Quantum Relative Entropy (QRE) with respect to a unit base encoder. We construct a map of encoders covering 1101 publicly available sentence encoders, providing a new perspective of the landscape of the pre-trained sentence encoders. Our map accurately reflects various relationships between encoders, where encoders with similar attributes are proximally located on the map. Moreover, our encoder feature vectors can be used to accurately infer downstream task performance of the encoders, such as in retrieval and clustering tasks, demonstrating the faithfulness of our map. 

---
# PERSPECTRA: A Scalable and Configurable Pluralist Benchmark of Perspectives from Arguments 

**Authors**: Shangrui Nie, Kian Omoomi, Lucie Flek, Zhixue Zhao, Charles Welch  

**Link**: [PDF](https://arxiv.org/pdf/2602.08716)  

**Abstract**: Pluralism, the capacity to engage with diverse perspectives without collapsing them into a single viewpoint, is critical for developing large language models that faithfully reflect human heterogeneity. Yet this characteristic has not been carefully examined in the LLM research community and remains absent from most alignment studies. Debate-oriented sources provide a natural entry point for pluralism research. Previous work builds on online debate sources but remains constrained by costly human validation. Other debate-rich platforms such as Reddit and Kialo also offer promising material: Reddit provides linguistic diversity and scale but lacks clear argumentative structure, while Kialo supplies explicit pro/con graphs but remains overly concise and detached from natural discourse. We introduce PERSPECTRA, a pluralist benchmark that integrates the structural clarity of Kialo debate graphs with the linguistic diversity of real Reddit discussions. Using a controlled retrieval-and-expansion pipeline, we construct 3,810 enriched arguments spanning 762 pro/con stances on 100 controversial topics. Each opinion is expanded to multiple naturalistic variants, enabling robust evaluation of pluralism. We initialise three tasks with PERSPECTRA: opinion counting (identifying distinct viewpoints), opinion matching (aligning supporting stances and discourse to source opinions), and polarity check (inferring aggregate stance in mixed discourse). Experiments with state-of-the-art open-source and proprietary LLMs, highlight systematic failures, such as overestimating the number of viewpoints and misclassifying concessive structures, underscoring the difficulty of pluralism-aware understanding and reasoning. By combining diversity with structure, PERSPECTRA establishes the first scalable, configurable benchmark for evaluating how well models represent, distinguish, and reason over multiple perspectives. 

---
# FactSim: Fact-Checking for Opinion Summarization 

**Authors**: Leandro Anghinoni, Jorge Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2602.08709)  

**Abstract**: We explore the need for more comprehensive and precise evaluation techniques for generative artificial intelligence (GenAI) in text summarization tasks, specifically in the area of opinion summarization. Traditional methods, which leverage automated metrics to compare machine-generated summaries from a collection of opinion pieces, e.g. product reviews, have shown limitations due to the paradigm shift introduced by large language models (LLM). This paper addresses these shortcomings by proposing a novel, fully automated methodology for assessing the factual consistency of such summaries. The method is based on measuring the similarity between the claims in a given summary with those from the original reviews, measuring the coverage and consistency of the generated summary. To do so, we rely on a simple approach to extract factual assessment from texts that we then compare and summarize in a suitable score. We demonstrate that the proposed metric attributes higher scores to similar claims, regardless of whether the claim is negated, paraphrased, or expanded, and that the score has a high correlation to human judgment when compared to state-of-the-art metrics. 

---
# Do Images Clarify? A Study on the Effect of Images on Clarifying Questions in Conversational Search 

**Authors**: Clemencia Siro, Zahra Abbasiantaeb, Yifei Yuan, Mohammad Aliannejadi, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2602.08700)  

**Abstract**: Conversational search systems increasingly employ clarifying questions to refine user queries and improve the search experience. Previous studies have demonstrated the usefulness of text-based clarifying questions in enhancing both retrieval performance and user experience. While images have been shown to improve retrieval performance in various contexts, their impact on user performance when incorporated into clarifying questions remains largely unexplored. We conduct a user study with 73 participants to investigate the role of images in conversational search, specifically examining their effects on two search-related tasks: (i) answering clarifying questions and (ii) query reformulation. We compare the effect of multimodal and text-only clarifying questions in both tasks within a conversational search context from various perspectives. Our findings reveal that while participants showed a strong preference for multimodal questions when answering clarifying questions, preferences were more balanced in the query reformulation task. The impact of images varied with both task type and user expertise. In answering clarifying questions, images helped maintain engagement across different expertise levels, while in query reformulation they led to more precise queries and improved retrieval performance. Interestingly, for clarifying question answering, text-only setups demonstrated better user performance as they provided more comprehensive textual information in the absence of images. These results provide valuable insights for designing effective multimodal conversational search systems, highlighting that the benefits of visual augmentation are task-dependent and should be strategically implemented based on the specific search context and user characteristics. 

---
# Challenges in Translating Technical Lectures: Insights from the NPTEL 

**Authors**: Basudha Raje, Sadanand Venkatraman, Nandana TP, Soumyadeepa Das, Polkam Poojitha, M. Vijaykumar, Tanima Bagchi, Hema A. Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2602.08698)  

**Abstract**: This study examines the practical applications and methodological implications of Machine Translation in Indian Languages, specifically Bangla, Malayalam, and Telugu, within emerging translation workflows and in relation to existing evaluation frameworks. The choice of languages prioritized in this study is motivated by a triangulation of linguistic diversity, which illustrates the significance of multilingual accommodation of educational technology under NEP 2020. This is further supported by the largest MOOC portal, i.e., NPTEL, which has served as a corpus to facilitate the arguments presented in this paper. The curation of a spontaneous speech corpora that accounts for lucid delivery of technical concepts, considering the retention of suitable register and lexical choices are crucial in a diverse country like India. The findings of this study highlight metric-specific sensitivity and the challenges of morphologically rich and semantically compact features when tested against surface overlapping metrics. 

---
# Old wine in old glasses: Comparing computational and qualitative methods in identifying incivility on Persian Twitter during the #MahsaAmini movement 

**Authors**: Hossein Kermani, Fatemeh Oudlajani, Pardis Yarahmadi, Hamideh Mahdi Soltani, Mohammad Makki, Zahra HosseiniKhoo  

**Link**: [PDF](https://arxiv.org/pdf/2602.08688)  

**Abstract**: This paper compares three approaches to detecting incivility in Persian tweets: human qualitative coding, supervised learning with ParsBERT, and large language models (ChatGPT). Using 47,278 tweets from the #MahsaAmini movement in Iran, we evaluate the accuracy and efficiency of each method. ParsBERT substantially outperforms seven evaluated ChatGPT models in identifying hate speech. We also find that ChatGPT struggles not only with subtle cases but also with explicitly uncivil content, and that prompt language (English vs. Persian) does not meaningfully affect its outputs. The study provides a detailed comparison of these approaches and clarifies their strengths and limitations for analyzing hate speech in a low-resource language context. 

---
# Learning to Judge: LLMs Designing and Applying Evaluation Rubrics 

**Authors**: Clemencia Siro, Pourya Aliannejadi, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08672)  

**Abstract**: Large language models (LLMs) are increasingly used as evaluators for natural language generation, applying human-defined rubrics to assess system outputs. However, human rubrics are often static and misaligned with how models internally represent language quality. We introduce GER-Eval (Generating Evaluation Rubrics for Evaluation) to investigate whether LLMs can design and apply their own evaluation rubrics. We evaluate the semantic coherence and scoring reliability of LLM-defined criteria and their alignment with human criteria. LLMs reliably generate interpretable and task-aware evaluation dimensions and apply them consistently within models, but their scoring reliability degrades in factual and knowledge-intensive settings. Closed-source models such as GPT-4o achieve higher agreement and cross-model generalization than open-weight models such as Llama. Our findings position evaluation as a learned linguistic capability of LLMs, consistent within models but fragmented across them, and call for new methods that jointly model human and LLM evaluative language to improve reliability and interpretability. 

---
# Fundamental Reasoning Paradigms Induce Out-of-Domain Generalization in Language Models 

**Authors**: Mingzi Cao, Xingwei Tan, Mahmud Akhter, Marco Valentino, Maria Liakata, Xi Wang, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2602.08658)  

**Abstract**: Deduction, induction, and abduction are fundamental reasoning paradigms, core for human logical thinking. Although improving Large Language Model (LLM) reasoning has attracted significant research efforts, the extent to which the fundamental paradigms induce generalization has yet to be systematically explored. In this study, we shed light on how the interplay between these core paradigms influences LLMs' reasoning behavior. To this end, we first collect a new dataset of reasoning trajectories from symbolic tasks, each targeting one of the three fundamental paradigms, to abstract from concrete world knowledge. Then, we investigate effective ways for inducing these skills into LLMs. We experiment with a battery of methods including simple fine-tuning, and more complex approaches to increase model depth, or transform a dense model to a mixture-of-experts. We comprehensively evaluate induced models on realistic out-of-domain tasks, that are entirely formulated in natural language and contain real-world knowledge. Our results reveal that our approach yields strong generalizability with substantial performance gains (up to $14.60$) across realistic tasks. 

---
# Do Multilingual LLMs have specialized language heads? 

**Authors**: Muhammad Naufil  

**Link**: [PDF](https://arxiv.org/pdf/2602.08625)  

**Abstract**: Multilingual large language models (LLMs) have gained significant popularity for their ability to process and generate text across multiple languages. However, deploying these models in production can be inefficient when only a subset of the supported languages is of interest. There has been some research conducted on identifying whether machine translation models have language-specific or language-agnostic heads, however no research has been conducted for multilingual LLMs, to the best of our knowledge, that as we know are capable of performing diverse tasks beyond just translation. This paper explores whether multilingual LLMs have specialized language attention heads for each language, and investigates the possibility of removing language-specific heads for unwanted languages without degrading performance in the targeted languages. Our findings could inform more efficient deployment strategies for multilingual LLMs, enabling reduced model complexity while maintaining high accuracy for targeted languages. 

---
# VocalNet-MDM: Accelerating Streaming Speech LLM via Self-Distilled Masked Diffusion Modeling 

**Authors**: Ziyang Cheng, Yuhao Wang, Heyang Liu, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08607)  

**Abstract**: Recent Speech Large Language Models~(LLMs) have achieved impressive capabilities in end-to-end speech interaction. However, the prevailing autoregressive paradigm imposes strict serial constraints, limiting generation efficiency and introducing exposure bias. In this paper, we investigate Masked Diffusion Modeling~(MDM) as a non-autoregressive paradigm for speech LLMs and introduce VocalNet-MDM. To adapt MDM for streaming speech interaction, we address two critical challenges: training-inference mismatch and iterative overhead. We propose Hierarchical Block-wise Masking to align training objectives with the progressive masked states encountered during block diffusion decoding, and Iterative Self-Distillation to compress multi-step refinement into fewer steps for low-latency inference. Trained on a limited scale of only 6K hours of speech data, VocalNet-MDM achieves a 3.7$\times$--10$\times$ decoding speedup and reduces first-chunk latency by 34\% compared to AR baselines. It maintains competitive recognition accuracy while achieving state-of-the-art text quality and speech naturalness, demonstrating that MDM is a promising and scalable alternative for low-latency, efficient speech LLMs. 

---
# Beyond Scalar Scores: Reinforcement Learning for Error-Aware Quality Estimation of Machine Translation 

**Authors**: Archchana Sindhujan, Girish A. Koushik, Shenbin Qian, Diptesh Kanojia, Constantin Orăsan  

**Link**: [PDF](https://arxiv.org/pdf/2602.08600)  

**Abstract**: Quality Estimation (QE) aims to assess the quality of machine translation (MT) outputs without relying on reference translations, making it essential for real-world, large-scale MT evaluation. Large Language Models (LLMs) have shown significant promise in advancing the field of quality estimation of machine translation. However, most of the QE approaches solely rely on scalar quality scores, offering no explicit information about the translation errors that should drive these judgments. Moreover, for low-resource languages where annotated QE data is limited, existing approaches struggle to achieve reliable performance. To address these challenges, we introduce the first segment-level QE dataset for English to Malayalam, a severely resource-scarce language pair in the QE domain, comprising human-annotated Direct Assessment (DA) scores and Translation Quality Remarks (TQR), which are short, contextual, free-form annotator comments that describe translation errors. We further introduce ALOPE-RL, a policy-based reinforcement learning framework that trains efficient adapters based on policy rewards derived from DA score and TQR. Integrating error-aware rewards with ALOPE-RL, enables LLMs to reason about translation quality beyond numeric scores. Despite being trained on a small-scale QE dataset, ALOPE-RL achieves state-of-the-art performance on English to Malayalam QE using compact LLMs (<=4B parameters}) fine-tuned with LoRA and 4-bit quantization, outperforming both larger LLM-based baselines and leading encoder-based QE models. Our results demonstrate that error-aware, policy-based learning can deliver strong QE performance under limited data and compute budgets. We release our dataset, code, and trained models to support future research. 

---
# How Do Language Models Understand Tables? A Mechanistic Analysis of Cell Location 

**Authors**: Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2602.08548)  

**Abstract**: While Large Language Models (LLMs) are increasingly deployed for table-related tasks, the internal mechanisms enabling them to process linearized two-dimensional structured tables remain opaque. In this work, we investigate the process of table understanding by dissecting the atomic task of cell location. Through activation patching and complementary interpretability techniques, we delineate the table understanding mechanism into a sequential three-stage pipeline: Semantic Binding, Coordinate Localization, and Information Extraction. We demonstrate that models locate the target cell via an ordinal mechanism that counts discrete delimiters to resolve coordinates. Furthermore, column indices are encoded within a linear subspace that allows for precise steering of model focus through vector arithmetic. Finally, we reveal that models generalize to multi-cell location tasks by multiplexing the identical attention heads identified during atomic location. Our findings provide a comprehensive explanation of table understanding within Transformer architectures. 

---
# GISA: A Benchmark for General Information-Seeking Assistant 

**Authors**: Yutao Zhu, Xingshuo Zhang, Maosen Zhang, Jiajie Jin, Liancheng Zhang, Xiaoshuai Song, Kangzhi Zhao, Wencong Zeng, Ruiming Tang, Han Li, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2602.08543)  

**Abstract**: The advancement of large language models (LLMs) has significantly accelerated the development of search agents capable of autonomously gathering information through multi-turn web interactions. Various benchmarks have been proposed to evaluate such agents. However, existing benchmarks often construct queries backward from answers, producing unnatural tasks misaligned with real-world needs. Moreover, these benchmarks tend to focus on either locating specific information or aggregating information from multiple sources, while relying on static answer sets prone to data contamination. To bridge these gaps, we introduce GISA, a benchmark for General Information-Seeking Assistants comprising 373 human-crafted queries that reflect authentic information-seeking scenarios. GISA features four structured answer formats (item, set, list, and table), enabling deterministic evaluation. It integrates both deep reasoning and broad information aggregation within unified tasks, and includes a live subset with periodically updated answers to resist memorization. Notably, GISA provides complete human search trajectories for every query, offering gold-standard references for process-level supervision and imitation learning. Experiments on mainstream LLMs and commercial search products reveal that even the best-performing model achieves only 19.30\% exact match score, with performance notably degrading on tasks requiring complex planning and comprehensive information gathering. These findings highlight substantial room for future improvement. 

---
# Characterizing, Evaluating, and Optimizing Complex Reasoning 

**Authors**: Haoran Zhang, Yafu Li, Zhi Wang, Zhilin Wang, Shunkai Zhang, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08498)  

**Abstract**: Large Reasoning Models (LRMs) increasingly rely on reasoning traces with complex internal structures. However, existing work lacks a unified answer to three fundamental questions: (1) what defines high-quality reasoning, (2) how to reliably evaluate long, implicitly structured reasoning traces, and (3) how to use such evaluation signals for reasoning optimization. To address these challenges, we provide a unified perspective. (1) We introduce the ME$^2$ principle to characterize reasoning quality along macro- and micro-level concerning efficiency and effectiveness. (2) Built on this principle, we model reasoning traces as directed acyclic graphs (DAGs) and develop a DAG-based pairwise evaluation method, capturing complex reasoning structures. (3) Based on this method, we construct the TRM-Preference dataset and train a Thinking Reward Model (TRM) to evaluate reasoning quality at scale. Experiments show that thinking rewards serve as an effective optimization signal. At test time, selecting better reasoning leads to better outcomes (up to 19.3% gain), and during RL training, thinking rewards enhance reasoning and performance (up to 3.9% gain) across diverse tasks. 

---
# Large Language Models and Impossible Language Acquisition: "False Promise" or an Overturn of our Current Perspective towards AI 

**Authors**: Ziyan wang, Longlong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.08437)  

**Abstract**: In Chomsky's provocative critique "The False Promise of CHATGPT," Large Language Models (LLMs) are characterized as mere pattern predictors that do not acquire languages via intrinsic causal and self-correction structures like humans, therefore are not able to distinguish impossible languages. It stands as a representative in a fundamental challenge to the intellectual foundations of AI, for it integrally synthesizes major issues in methodologies within LLMs and possesses an iconic a priori rationalist perspective. We examine this famous critic from both the perspective in pre-existing literature of linguistics and psychology as well as a research based on an experiment inquiring the capacity of learning both possible and impossible languages among LLMs. We constructed a set of syntactically impossible languages by applying certain transformations to English. These include reversing whole sentences, and adding negation based on word-count parity. Two rounds of controlled experiments were each conducted on GPT-2 small models and long short-term memory (LSTM) models. Statistical analysis (Welch's t-test) shows GPT2 small models underperform in learning all of the impossible languages compared to their performance on the possible language (p<.001). On the other hand, LSTM models' performance tallies with Chomsky's argument, suggesting the irreplaceable role of the evolution of transformer architecture. Based on theoretical analysis and empirical findings, we propose a new vision within Chomsky's theory towards LLMs, and a shift of theoretical paradigm outside Chomsky, from his "rationalist-romantics" paradigm to functionalism and empiricism in LLMs research. 

---
# Prism: Spectral-Aware Block-Sparse Attention 

**Authors**: Xinghao Wang, Pengyu Wang, Xiaoran Liu, Fangxu Liu, Jason Chu, Kai Song, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08426)  

**Abstract**: Block-sparse attention is promising for accelerating long-context LLM pre-filling, yet identifying relevant blocks efficiently remains a bottleneck. Existing methods typically employ coarse-grained attention as a proxy for block importance estimation, but often resort to expensive token-level searching or scoring, resulting in significant selection overhead. In this work, we trace the inaccuracy of standard coarse-grained attention via mean pooling to a theoretical root cause: the interaction between mean pooling and Rotary Positional Embeddings (RoPE). We prove that mean pooling acts as a low-pass filter that induces destructive interference in high-frequency dimensions, effectively creating a "blind spot" for local positional information (e.g., slash patterns). To address this, we introduce Prism, a training-free spectral-aware approach that decomposes block selection into high-frequency and low-frequency branches. By applying energy-based temperature calibration, Prism restores the attenuated positional signals directly from pooled representations, enabling block importance estimation using purely block-level operations, thereby improving efficiency. Extensive evaluations confirm that Prism maintains accuracy parity with full attention while delivering up to $\mathbf{5.1\times}$ speedup. 

---
# TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration 

**Authors**: Linye Wei, Zixiang Luo, Pingzhi Tang, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08404)  

**Abstract**: Diffusion large language models (dLLMs) have recently gained significant attention due to their inherent support for parallel decoding. Building on this paradigm, Mixture-of-Experts (MoE) dLLMs with autoregressive (AR) initialization have further demonstrated strong performance competitive with mainstream AR models. However, we identify a fundamental mismatch between MoE architectures and diffusion-based decoding. Specifically, a large number of experts are activated at each denoising step, while only a small subset of tokens is ultimately accepted, resulting in substantial inference overhead and limiting their deployment in latency-sensitive applications. In this work, we propose TEAM, a plug-and-play framework that accelerates MoE dLLMs by enabling more accepted tokens with fewer activated experts. TEAM is motivated by the observation that expert routing decisions exhibit strong temporal consistency across denoising levels as well as spatial consistency across token positions. Leveraging these properties, TEAM employs three complementary expert activation and decoding strategies, conservatively selecting necessary experts for decoded and masked tokens and simultaneously performing aggressive speculative exploration across multiple candidates. Experimental results demonstrate that TEAM achieves up to 2.2x speedup over vanilla MoE dLLM, with negligible performance degradation. Code is released at this https URL. 

---
# Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning 

**Authors**: Zhuoen Chen, Dongfang Li, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08382)  

**Abstract**: Large Language Models (LLMs) face significant challenges in long-context processing, including quadratic computational costs, information forgetting, and the context fragmentation inherent in retrieval-augmented generation (RAG). We propose a cognitively inspired framework for efficient long-context inference based on chunk-wise compression and selective memory recall, rather than processing all raw tokens. The framework segments long inputs into chunks and encodes each chunk into compressed memory representations using a learned compressor. A gating module dynamically selects relevant memory blocks, which are then iteratively processed by a reasoning module with an evolving working memory to solve downstream tasks. The compressor and reasoner are jointly optimized via end-to-end reinforcement learning, while the gating module is trained separately as a classifier. Experimental results show that the proposed method achieves competitive accuracy on multi-hop reasoning benchmarks such as RULER-HQA, extrapolates context length from 7K to 1.75M tokens, and offers a favorable accuracy-efficiency trade-off compared to strong long-context baselines. In particular, it achieves up to a 2 times reduction in peak GPU memory usage and a 6 times inference speedup over MemAgent. 

---
# ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts 

**Authors**: Hung Quang Tran, Nam Tien Pham, Son T. Luu, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08371)  

**Abstract**: Emotion classification plays a significant role in emotion prediction and harmful content detection. Recent advancements in NLP, particularly through large language models (LLMs), have greatly improved outcomes in this field. This study introduces ViGoEmotions -- a Vietnamese emotion corpus comprising 20,664 social media comments in which each comment is classified into 27 fine-grained distinct emotions. To evaluate the quality of the dataset and its impact on emotion classification, eight pre-trained Transformer-based models were evaluated under three preprocessing strategies: preserving original emojis with rule-based normalization, converting emojis into textual descriptions, and applying ViSoLex, a model-based lexical normalization system. Results show that converting emojis into text often improves the performance of several BERT-based baselines, while preserving emojis yields the best results for ViSoBERT and CafeBERT. In contrast, removing emojis generally leads to lower performance. ViSoBERT achieved the highest Macro F1-score of 61.50% and Weighted F1-score of 63.26%. Strong performance was also observed from CafeBERT and PhoBERT. These findings highlight that while the proposed corpus can support diverse architectures effectively, preprocessing strategies and annotation quality remain key factors influencing downstream performance. 

---
# WorldTravel: A Realistic Multimodal Travel-Planning Benchmark with Tightly Coupled Constraints 

**Authors**: Zexuan Wang, Chenghao Yang, Yingqi Que, Zhenzhu Yang, Huaqing Yuan, Yiwen Wang, Zhengxuan Jiang, Shengjie Fang, Zhenhe Wu, Zhaohui Wang, Zhixin Yao, Jiashuo Liu, Jincheng Ren, Yuzhen Li, Yang Yang, Jiaheng Liu, Jian Yang, Zaiyuan Wang, Ge Zhang, Zhoufutu Wen, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08367)  

**Abstract**: Real-world autonomous planning requires coordinating tightly coupled constraints where a single decision dictates the feasibility of all subsequent actions. However, existing benchmarks predominantly feature loosely coupled constraints solvable through local greedy decisions and rely on idealized data, failing to capture the complexity of extracting parameters from dynamic web environments. We introduce \textbf{WorldTravel}, a benchmark comprising 150 real-world travel scenarios across 5 cities that demand navigating an average of 15+ interdependent temporal and logical constraints. To evaluate agents in realistic deployments, we develop \textbf{WorldTravel-Webscape}, a multi-modal environment featuring over 2,000 rendered webpages where agents must perceive constraint parameters directly from visual layouts to inform their planning. Our evaluation of 10 frontier models reveals a significant performance collapse: even the state-of-the-art GPT-5.2 achieves only 32.67\% feasibility in text-only settings, which plummets to 19.33\% in multi-modal environments. We identify a critical Perception-Action Gap and a Planning Horizon threshold at approximately 10 constraints where model reasoning consistently fails, suggesting that perception and reasoning remain independent bottlenecks. These findings underscore the need for next-generation agents that unify high-fidelity visual perception with long-horizon reasoning to handle brittle real-world logistics. 

---
# UReason: Benchmarking the Reasoning Paradox in Unified Multimodal Models 

**Authors**: Cheng Yang, Chufan Shi, Bo Shui, Yaokang Wu, Muzi Tao, Huijuan Wang, Ivan Yee Lee, Yong Liu, Xuezhe Ma, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2602.08336)  

**Abstract**: To elicit capabilities for addressing complex and implicit visual requirements, recent unified multimodal models increasingly adopt chain-of-thought reasoning to guide image generation. However, the actual effect of reasoning on visual synthesis remains unclear. We present UReason, a diagnostic benchmark for reasoning-driven image generation that evaluates whether reasoning can be faithfully executed in pixels. UReason contains 2,000 instances across five task families: Code, Arithmetic, Spatial, Attribute, and Text reasoning. To isolate the role of reasoning traces, we introduce an evaluation framework comparing direct generation, reasoning-guided generation, and de-contextualized generation which conditions only on the refined prompt. Across eight open-source unified models, we observe a consistent Reasoning Paradox: Reasoning traces generally improve performance over direct generation, yet retaining intermediate thoughts as conditioning context often hinders visual synthesis, and conditioning only on the refined prompt yields substantial gains. Our analysis suggests that the bottleneck lies in contextual interference rather than insufficient reasoning capacity. UReason provides a principled testbed for studying reasoning in unified models and motivates future methods that effectively integrate reasoning for visual generation while mitigating interference. 

---
# Latent Reasoning with Supervised Thinking States 

**Authors**: Ido Amos, Avi Caciularu, Mor Geva, Amir Globerson, Jonathan Herzig, Lior Shani, Idan Szpektor  

**Link**: [PDF](https://arxiv.org/pdf/2602.08332)  

**Abstract**: Reasoning with a chain-of-thought (CoT) enables Large Language Models (LLMs) to solve complex tasks but incurs significant inference costs due to the generation of long rationales. We propose Thinking States, a method that performs reasoning {\em while} the input is processing. Specifically, Thinking States generates sequences of thinking tokens every few input tokens, transforms the thoughts back into embedding space, and adds them to the following input tokens. This has two key advantages. First, it captures the recurrent nature of CoT, but where the thought tokens are generated as input is processing. Second, since the thoughts are represented as tokens, they can be learned from natural language supervision, and using teacher-forcing, which is parallelizable. Empirically, Thinking States outperforms other latent reasoning methods on multiple reasoning tasks, narrowing the gap to CoT on math problems, and matching its performance on 2-Hop QA with improved latency. On state-tracking tasks, we show Thinking States leads to stronger reasoning behavior than CoT, successfully extrapolating to longer sequences than seen during training. 

---
# An Attention-over-Attention Generative Model for Joint Multiple Intent Detection and Slot Filling 

**Authors**: Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08322)  

**Abstract**: In task-oriented dialogue systems, spoken language understanding (SLU) is a critical component, which consists of two sub-tasks, intent detection and slot filling. Most existing methods focus on the single-intent SLU, where each utterance only has one intent. However, in real-world scenarios users usually express multiple intents in an utterance, which poses a challenge for existing dialogue systems and datasets. In this paper, we propose a generative framework to simultaneously address multiple intent detection and slot filling. In particular, an attention-over-attention decoder is proposed to handle the variable number of intents and the interference between the two sub-tasks by incorporating an inductive bias into the process of multi-task learning. Besides, we construct two new multi-intent SLU datasets based on single-intent utterances by taking advantage of the next sentence prediction (NSP) head of the BERT model. Experimental results demonstrate that our proposed attention-over-attention generative model achieves state-of-the-art performance on two public datasets, MixATIS and MixSNIPS, and our constructed datasets. 

---
# Improving Data and Reward Design for Scientific Reasoning in Large Language Models 

**Authors**: Zijie Chen, Zhenghao Lin, Xiao Liu, Zhenzhong Lan, Yeyun Gong, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08321)  

**Abstract**: Solving open-ended science questions remains challenging for large language models, particularly due to inherently unreliable supervision and evaluation. The bottleneck lies in the data construction and reward design for scientific post-training. We develop a large-scale, systematic data processing pipeline that transforms heterogeneous open-source science data into Dr. SCI dataset, which comprises of 1M questions across eight STEM subjects, with explicit verifiable/open-ended splits, scalable difficulty annotation, and fine-grained rubrics that operationalize evaluation for open-ended answers. Building on this dataset, we propose the Dr. SCI post-training pipeline, which redesigns the standard SFT -> RL workflow through three components: (i) Exploration-Expanding SFT, which broadens the model's reasoning pattern coverage prior to RL; (ii) Dynamic Difficulty Curriculum, which adapts training data to the model's evolving scientific capability; and (iii) SciRubric-Guided RL, which enables stable reinforcement learning on open-ended scientific questions via rubric-based evaluation with explicit answer correctness. Qwen3-4B-Base trained using this http URL pipeline achieves 63.2 on GPQA-diamond and 32.4 on GPQA-general, consistently improves over strong post-trained baselines such as o1-mini and GPT-4o, demonstrating substantial gains in scientific reasoning, especially in open-ended settings. 

---
# JUSTICE: Judicial Unified Synthesis Through Intermediate Conclusion Emulation for Automated Judgment Document Generation 

**Authors**: Binglin Wu, Yingyi Zhang, Xiannneg Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08305)  

**Abstract**: Automated judgment document generation is a significant yet challenging legal AI task. As the conclusive written instrument issued by a court, a judgment document embodies complex legal reasoning. However, existing methods often oversimplify this complex process, particularly by omitting the ``Pre-Judge'' phase, a crucial step where human judges form a preliminary conclusion. This omission leads to two core challenges: 1) the ineffective acquisition of foundational judicial elements, and 2) the inadequate modeling of the Pre-Judge process, which collectively undermine the final document's legal soundness. To address these challenges, we propose \textit{\textbf{J}udicial \textbf{U}nified \textbf{S}ynthesis \textbf{T}hrough \textbf{I}ntermediate \textbf{C}onclusion \textbf{E}mulation} (JUSTICE), a novel framework that emulates the ``Search $\rightarrow$ Pre-Judge $\rightarrow$ Write'' cognitive workflow of human judges. Specifically, it introduces the Pre-Judge stage through three dedicated components: Referential Judicial Element Retriever (RJER), Intermediate Conclusion Emulator (ICE), and Judicial Unified Synthesizer (JUS). RJER first retrieves legal articles and a precedent case to establish a referential foundation. ICE then operationalizes the Pre-Judge phase by generating a verifiable intermediate conclusion. Finally, JUS synthesizes these inputs to craft the final judgment. Experiments on both an in-domain legal benchmark and an out-of-distribution dataset show that JUSTICE significantly outperforms strong baselines, with substantial gains in legal accuracy, including a 4.6\% improvement in prison term prediction. Our findings underscore the importance of explicitly modeling the Pre-Judge process to enhance the legal coherence and accuracy of generated judgment documents. 

---
# When Does Context Help? Error Dynamics of Contextual Information in Large Language Models 

**Authors**: Dingzirui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08294)  

**Abstract**: Contextual information at inference time, such as demonstrations, retrieved knowledge, or interaction history, can substantially improve large language models (LLMs) without parameter updates, yet its theoretical role remains poorly understood beyond specific settings such as in-context learning (ICL). We present a unified theoretical framework for analyzing the effect of arbitrary contextual information in Transformer-based LLMs. Our analysis characterizes contextual influence through output error dynamics. In a single-layer Transformer, we prove that the context-conditioned error vector decomposes additively into the baseline error vector and a contextual correction vector. This yields necessary geometric conditions for error reduction: the contextual correction must align with the negative baseline error and satisfy a norm constraint. We further show that the contextual correction norm admits an explicit upper bound determined by context-query relevance and complementarity. These results extend to multi-context and multi-layer Transformers. Experiments across ICL, retrieval-augmented generation, and memory evolution validate our theory and motivate a principled context selection strategy that improves performance by $0.6\%$. 

---
# Knowledge Augmented Entity and Relation Extraction for Legal Documents with Hypergraph Neural Network 

**Authors**: Binglin Wu, Xianneng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08289)  

**Abstract**: With the continuous progress of digitization in Chinese judicial institutions, a substantial amount of electronic legal document information has been accumulated. To unlock its potential value, entity and relation extraction for legal documents has emerged as a crucial task. However, existing methods often lack domain-specific knowledge and fail to account for the unique characteristics of the judicial domain. In this paper, we propose an entity and relation extraction algorithm based on hypergraph neural network (Legal-KAHRE) for drug-related judgment documents. Firstly, we design a candidate span generator based on neighbor-oriented packing strategy and biaffine mechanism, which identifies spans likely to contain entities. Secondly, we construct a legal dictionary with judicial domain knowledge and integrate it into text encoding representation using multi-head attention. Additionally, we incorporate domain-specific cases like joint crimes and combined punishment for multiple crimes into the hypergraph structure design. Finally, we employ a hypergraph neural network for higher-order inference via message passing. Experimental results on the CAIL2022 information extraction dataset demonstrate that our method significantly outperforms existing baseline models. 

---
# New Skills or Sharper Primitives? A Probabilistic Perspective on the Emergence of Reasoning in RLVR 

**Authors**: Zhilin Wang, Yafu Li, Shunkai Zhang, Zhi Wang, Haoran Zhang, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08281)  

**Abstract**: Whether Reinforcement Learning with Verifiable Rewards (RLVR) endows Large Language Models (LLMs) with new capabilities or merely elicits latent traces remains a central debate. In this work, we align with the former view, proposing a probabilistic framework where capability is defined by instance-level solvability. We hypothesize that the emergence of complex reasoning can be driven by sharpening atomic step probabilities, which enables models to overcome the exponential decay of success rates inherent in multi-step reasoning chains. Utilizing the Algebrarium framework, we train models exclusively on single-step operations and evaluate their performance on unseen multi-step tasks. Our empirical results confirm that: (1) RLVR incentivizes the exploration of previously inaccessible solution paths by amplifying the model's existing skills; (2) composite performance is strictly governed by the joint probability of atomic steps, evidenced by high Pearson correlation coefficients ($\rho \in [0.69, 0.96]$); and (3) RLVR, acting as a global optimizer, can cause specific skills to be sacrificed to maximize aggregate reward. Our work offers a novel explanation for emergent abilities in RLVR, suggesting that the iterative optimization of solvable problems enables models to develop the capabilities to tackle previously unsolvable scenarios. 

---
# Language Modeling and Understanding Through Paraphrase Generation and Detection 

**Authors**: Jan Philip Wahle  

**Link**: [PDF](https://arxiv.org/pdf/2602.08274)  

**Abstract**: Language enables humans to share knowledge, reason about the world, and pass on strategies for survival and innovation across generations. At the heart of this process is not just the ability to communicate but also the remarkable flexibility in how we can express ourselves. We can express the same thoughts in virtually infinite ways using different words and structures - this ability to rephrase and reformulate expressions is known as paraphrase. Modeling paraphrases is a keystone to meaning in computational language models; being able to construct different variations of texts that convey the same meaning or not shows strong abilities of semantic understanding. If computational language models are to represent meaning, they must understand and control the different aspects that construct the same meaning as opposed to different meanings at a fine granularity. Yet most existing approaches reduce paraphrasing to a binary decision between two texts or to producing a single rewrite of a source, obscuring which linguistic factors are responsible for meaning preservation. In this thesis, I propose that decomposing paraphrases into their constituent linguistic aspects (paraphrase types) offers a more fine-grained and cognitively grounded view of semantic equivalence. I show that even advanced machine learning models struggle with this task. Yet, when explicitly trained on paraphrase types, models achieve stronger performance on related paraphrase tasks and downstream applications. For example, in plagiarism detection, language models trained on paraphrase types surpass human baselines: 89.6% accuracy compared to 78.4% for plagiarism cases from Wikipedia, and 66.5% compared to 55.7% for plagiarism of scientific papers from arXiv. In identifying duplicate questions on Quora, models trained with paraphrase types improve over models trained on binary pairs. Furthermore, I demonstrate that... 

---
# Language Predicts Identity Fusion Across Cultures and Reveals Divergent Pathways to Violence 

**Authors**: Devin R. Wright, Justin E. Lane, F. LeRon Shults  

**Link**: [PDF](https://arxiv.org/pdf/2602.08252)  

**Abstract**: In light of increasing polarization and political violence, understanding the psychological roots of extremism is increasingly important. Prior research shows that identity fusion predicts willingness to engage in extreme acts. We evaluate the Cognitive Linguistic Identity Fusion Score, a method that uses cognitive linguistic patterns, LLMs, and implicit metaphor to measure fusion from language. Across datasets from the United Kingdom and Singapore, this approach outperforms existing methods in predicting validated fusion scores. Applied to extremist manifestos, two distinct high-fusion pathways to violence emerge: ideologues tend to frame themselves in terms of group, forming kinship bonds; whereas grievance-driven individuals frame the group in terms of their personal identity. These results refine theories of identity fusion and provide a scalable tool aiding fusion research and extremism detection. 

---
# On convexity and efficiency in semantic systems 

**Authors**: Nathaniel Imel, Noga Zaslavasky  

**Link**: [PDF](https://arxiv.org/pdf/2602.08238)  

**Abstract**: There are two widely held characterizations of human semantic category systems: (1) they form convex partitions of conceptual spaces, and (2) they are efficient for communication. While prior work observed that convexity and efficiency co-occur in color naming, the analytical relation between them and why they co-occur have not been well understood. We address this gap by combining analytical and empirical analyses that build on the Information Bottleneck (IB) framework for semantic efficiency. First, we show that convexity and efficiency are distinct in the sense that neither entails the other: there are convex systems which are inefficient, and optimally-efficient systems that are non-convex. Crucially, however, the IB-optimal systems are mostly convex in the domain of color naming, explaining the main empirical basis for the convexity approach. Second, we show that efficiency is a stronger predictor for discriminating attested color naming systems from hypothetical variants, with convexity adding negligible improvement on top of that. Finally, we discuss a range of empirical phenomena that convexity cannot account for but efficiency can. Taken together, our work suggests that while convexity and efficiency can yield similar structural observations, they are fundamentally distinct, with efficiency providing a more comprehensive account of semantic typology. 

---
# Document Reconstruction Unlocks Scalable Long-Context RLVR 

**Authors**: Yao Xiao, Lei Wang, Yue Deng, Guanzheng Chen, Ziqi Jin, Jung-jae Kim, Xiaoli Li, Roy Ka-wei Lee, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2602.08237)  

**Abstract**: Reinforcement Learning with Verifiable Rewards~(RLVR) has become a prominent paradigm to enhance the capabilities (i.e.\ long-context) of Large Language Models~(LLMs). However, it often relies on gold-standard answers or explicit evaluation rubrics provided by powerful teacher models or human experts, which are costly and time-consuming. In this work, we investigate unsupervised approaches to enhance the long-context capabilities of LLMs, eliminating the need for heavy human annotations or teacher models' supervision. Specifically, we first replace a few paragraphs with special placeholders in a long document. LLMs are trained through reinforcement learning to reconstruct the document by correctly identifying and sequencing missing paragraphs from a set of candidate options. This training paradigm enables the model to capture global narrative coherence, significantly boosting long-context performance. We validate the effectiveness of our method on two widely used benchmarks, RULER and LongBench~v2. While acquiring noticeable gains on RULER, it can also achieve a reasonable improvement on LongBench~v2 without any manually curated long-context QA data. Furthermore, we conduct extensive ablation studies to analyze the impact of reward design, data curation strategies, training schemes, and data scaling effects on model performance. We publicly release our code, data, and models. 

---
# When Benign Inputs Lead to Severe Harms: Eliciting Unsafe Unintended Behaviors of Computer-Use Agents 

**Authors**: Jaylen Jones, Zhehao Zhang, Yuting Ning, Eric Fosler-Lussier, Pierre-Luc St-Charles, Yoshua Bengio, Dawn Song, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.08235)  

**Abstract**: Although computer-use agents (CUAs) hold significant potential to automate increasingly complex OS workflows, they can demonstrate unsafe unintended behaviors that deviate from expected outcomes even under benign input contexts. However, exploration of this risk remains largely anecdotal, lacking concrete characterization and automated methods to proactively surface long-tail unintended behaviors under realistic CUA scenarios. To fill this gap, we introduce the first conceptual and methodological framework for unintended CUA behaviors, by defining their key characteristics, automatically eliciting them, and analyzing how they arise from benign inputs. We propose AutoElicit: an agentic framework that iteratively perturbs benign instructions using CUA execution feedback, and elicits severe harms while keeping perturbations realistic and benign. Using AutoElicit, we surface hundreds of harmful unintended behaviors from state-of-the-art CUAs such as Claude 4.5 Haiku and Opus. We further evaluate the transferability of human-verified successful perturbations, identifying persistent susceptibility to unintended behaviors across various other frontier CUAs. This work establishes a foundation for systematically analyzing unintended behaviors in realistic computer-use settings. 

---
# CoRect: Context-Aware Logit Contrast for Hidden State Rectification to Resolve Knowledge Conflicts 

**Authors**: Xuhua Ma, Richong Zhang, Zhijie Nie  

**Link**: [PDF](https://arxiv.org/pdf/2602.08221)  

**Abstract**: Retrieval-Augmented Generation (RAG) often struggles with knowledge conflicts, where model-internal parametric knowledge overrides retrieved evidence, leading to unfaithful outputs. Existing approaches are often limited, relying either on superficial decoding adjustments or weight editing that necessitates ground-truth targets. Through layer-wise analysis, we attribute this failure to a parametric suppression phenomenon: specifically, in deep layers, certain FFN layers overwrite context-sensitive representations with memorized priors. To address this, we propose CoRect (Context-Aware Logit Contrast for Hidden State Rectification). By contrasting logits from contextualized and non-contextualized forward passes, CoRect identifies layers that exhibit high parametric bias without requiring ground-truth labels. It then rectifies the hidden states to preserve evidence-grounded information. Across question answering (QA) and summarization benchmarks, CoRect consistently improves faithfulness and reduces hallucinations compared to strong baselines. 

---
# Pretraining with Token-Level Adaptive Latent Chain-of-Thought 

**Authors**: Boyi Zeng, Yiqin Hao, He Li, Shixiang Song, Feichen Song, Zitong Wang, Siyuan Huang, Yi Xu, ZiWei He, Xinbing Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08220)  

**Abstract**: Scaling large language models by increasing parameters and training data is increasingly constrained by limited high-quality corpora and rising communication costs. This work explores an alternative axis: increasing per-token computation without expanding parameters, by internalizing latent Chain-of-Thought (CoT) into pretraining. We propose Pretraining with Token-Level Adaptive Latent CoT (adaptive latent CoT), where the model generates a variable-length latent CoT trajectory before emitting each token -- allocating longer trajectories to difficult tokens and shorter (or even zero) trajectories to easy ones. Importantly, this behavior emerges naturally from one-stage pretraining on general text and reduces computation in both training and inference via token-wise adaptive halting. Experiments with Llama architectures show that adaptive latent CoT consistently improves language modeling perplexity and broad downstream accuracy, even with fewer training FLOPs than prior recurrent baselines. 

---
# LLMs and people both learn to form conventions -- just not with each other 

**Authors**: Cameron R. Jones, Agnese Lombardi, Kyle Mahowald, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08208)  

**Abstract**: Humans align to one another in conversation -- adopting shared conventions that ease communication. We test whether LLMs form the same kinds of conventions in a multimodal communication game. Both humans and LLMs display evidence of convention-formation (increasing the accuracy and consistency of their turns while decreasing their length) when communicating in same-type dyads (humans with humans, AI with AI). However, heterogenous human-AI pairs fail -- suggesting differences in communicative tendencies. In Experiment 2, we ask whether LLMs can be induced to behave more like human conversants, by prompting them to produce superficially humanlike behavior. While the length of their messages matches that of human pairs, accuracy and lexical overlap in human-LLM pairs continues to lag behind that of both human-human and AI-AI pairs. These results suggest that conversational alignment requires more than just the ability to mimic previous interactions, but also shared interpretative biases toward the meanings that are conveyed. 

---
# NLP for Local Governance Meeting Records: A Focus Article on Tasks, Datasets, Metrics and Benchmark 

**Authors**: Ricardo Campos, José Pedro Evans, José Miguel Isidro, Miguel Marques, Luís Filipe Cunha, Alípio Jorge, Sérgio Nunes, Nuno Guimarães  

**Link**: [PDF](https://arxiv.org/pdf/2602.08162)  

**Abstract**: Local governance meeting records are official documents, in the form of minutes or transcripts, documenting how proposals, discussions, and procedural actions unfold during institutional meetings. While generally structured, these documents are often dense, bureaucratic, and highly heterogeneous across municipalities, exhibiting significant variation in language, terminology, structure, and overall organization. This heterogeneity makes them difficult for non-experts to interpret and challenging for intelligent automated systems to process, limiting public transparency and civic engagement. To address these challenges, computational methods can be employed to structure and interpret such complex documents. In particular, Natural Language Processing (NLP) offers well-established methods that can enhance the accessibility and interpretability of governmental records. In this focus article, we review foundational NLP tasks that support the structuring of local governance meeting documents. Specifically, we review three core tasks: document segmentation, domain-specific entity extraction and automatic text summarization, which are essential for navigating lengthy deliberations, identifying political actors and personal information, and generating concise representations of complex decision-making processes. In reviewing these tasks, we discuss methodological approaches, evaluation metrics, and publicly available resources, while highlighting domain-specific challenges such as data scarcity, privacy constraints, and source variability. By synthesizing existing work across these foundational tasks, this article provides a structured overview of how NLP can enhance the structuring and accessibility of local governance meeting records. 

---
# DIAL-SUMMER: A Structured Evaluation Framework of Hierarchical Errors in Dialogue Summaries 

**Authors**: Sahana Ramnath, Nima Chitsazan, Mingyang Zhou, Chia-Hsuan Lee, Shi-Xiong Zhang, Stephen Rawls, Sambit Sahu, Sangwoo Cho, Xiang Ren, Genta Indra Winata, Akshaj Kumar Veldanda  

**Link**: [PDF](https://arxiv.org/pdf/2602.08149)  

**Abstract**: Dialogues are a predominant mode of communication for humans, and it is immensely helpful to have automatically generated summaries of them (e.g., to revise key points discussed in a meeting, to review conversations between customer agents and product users). Prior works on dialogue summary evaluation largely ignore the complexities specific to this task: (i) shift in structure, from multiple speakers discussing information in a scattered fashion across several turns, to a summary's sentences, and (ii) shift in narration viewpoint, from speakers' first/second-person narration, standardized third-person narration in the summary. In this work, we introduce our framework DIALSUMMER to address the above. We propose DIAL-SUMMER's taxonomy of errors to comprehensively evaluate dialogue summaries at two hierarchical levels: DIALOGUE-LEVEL that focuses on the broader speakers/turns, and WITHIN-TURN-LEVEL that focuses on the information talked about inside a turn. We then present DIAL-SUMMER's dataset composed of dialogue summaries manually annotated with our taxonomy's fine-grained errors. We conduct empirical analyses of these annotated errors, and observe interesting trends (e.g., turns occurring in middle of the dialogue are the most frequently missed in the summary, extrinsic hallucinations largely occur at the end of the summary). We also conduct experiments on LLM-Judges' capability at detecting these errors, through which we demonstrate the challenging nature of our dataset, the robustness of our taxonomy, and the need for future work in this field to enhance LLMs' performance in the same. Code and inference dataset coming soon. 

---
# Gender and Race Bias in Consumer Product Recommendations by Large Language Models 

**Authors**: Ke Xu, Shera Potka, Alex Thomo  

**Link**: [PDF](https://arxiv.org/pdf/2602.08124)  

**Abstract**: Large Language Models are increasingly employed in generating consumer product recommendations, yet their potential for embedding and amplifying gender and race biases remains underexplored. This paper serves as one of the first attempts to examine these biases within LLM-generated recommendations. We leverage prompt engineering to elicit product suggestions from LLMs for various race and gender groups and employ three analytical methods-Marked Words, Support Vector Machines, and Jensen-Shannon Divergence-to identify and quantify biases. Our findings reveal significant disparities in the recommendations for demographic groups, underscoring the need for more equitable LLM recommendation systems. 

---
# Emergent Search and Backtracking in Latent Reasoning Models 

**Authors**: Jasmine Cui, Charles Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.08100)  

**Abstract**: What happens when a language model thinks without words? Standard reasoning LLMs verbalize intermediate steps as chain-of-thought; latent reasoning transformers (LRTs) instead perform deliberation entirely in continuous hidden space. We investigate an LRT, decoding the model's evolving beliefs at every step on a multiple-choice QA benchmark. We find that the model spontaneously learns a structured search process in latent space. Deliberation follows a consistent trajectory: an exploration phase where probability mass spreads across candidates, tentative commitment to a frontrunner, and either convergence or backtracking. Backtracking is prevalent (32% of instances), beneficial (34% accuracy gain over non-backtracking instances), and predominantly directed away from the semantically closest distractor toward the correct answer. The search is adaptive: replacing distractors with implausible alternatives shortens exploration by 54%. Latent reasoning models achieve in activation space what chain-of-thought achieves through words: the ability to be wrong, notice, and recover. 

---
# TDGNet: Hallucination Detection in Diffusion Language Models via Temporal Dynamic Graphs 

**Authors**: Arshia Hemmat, Philip Torr, Yongqiang Chen, Junchi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08048)  

**Abstract**: Diffusion language models (D-LLMs) offer parallel denoising and bidirectional context, but hallucination detection for D-LLMs remains underexplored. Prior detectors developed for auto-regressive LLMs typically rely on single-pass cues and do not directly transfer to diffusion generation, where factuality evidence is distributed across the denoising trajectory and may appear, drift, or be self-corrected over time. We introduce TDGNet, a temporal dynamic graph framework that formulates hallucination detection as learning over evolving token-level attention graphs. At each denoising step, we sparsify the attention graph and update per-token memories via message passing, then apply temporal attention to aggregate trajectory-wide evidence for final prediction. Experiments on LLaDA-8B and Dream-7B across QA benchmarks show consistent AUROC improvements over output-based, latent-based, and static-graph baselines, with single-pass inference and modest overhead. These results highlight the importance of temporal reasoning on attention graphs for robust hallucination detection in diffusion language models. 

---
# Beyond Raw Detection Scores: Markov-Informed Calibration for Boosting Machine-Generated Text Detection 

**Authors**: Chenwang Wu, Yiu-ming Cheung, Shuhai Zhang, Bo Han, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2602.08031)  

**Abstract**: While machine-generated texts (MGTs) offer great convenience, they also pose risks such as disinformation and phishing, highlighting the need for reliable detection. Metric-based methods, which extract statistically distinguishable features of MGTs, are often more practical than complex model-based methods that are prone to overfitting. Given their diverse designs, we first place representative metric-based methods within a unified framework, enabling a clear assessment of their advantages and limitations. Our analysis identifies a core challenge across these methods: the token-level detection score is easily biased by the inherent randomness of the MGTs generation process. To address this, we theoretically and empirically reveal two relationships of context detection scores that may aid calibration: Neighbor Similarity and Initial Instability. We then propose a Markov-informed score calibration strategy that models these relationships using Markov random fields, and implements it as a lightweight component via a mean-field approximation, allowing our method to be seamlessly integrated into existing detectors. Extensive experiments in various real-world scenarios, such as cross-LLM and paraphrasing attacks, demonstrate significant gains over baselines with negligible computational overhead. The code is available at this https URL. 

---
# Diverge to Induce Prompting: Multi-Rationale Induction for Zero-Shot Reasoning 

**Authors**: Po-Chun Chen, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08028)  

**Abstract**: To address the instability of unguided reasoning paths in standard Chain-of-Thought prompting, recent methods guide large language models (LLMs) by first eliciting a single reasoning strategy. However, relying on just one strategy for each question can still limit performance across diverse tasks. We propose Diverge-to-Induce Prompting (DIP), a framework that first prompts an LLM to generate multiple diverse high-level rationales for each question. Each rationale is then elaborated into a detailed, step-by-step draft plan. Finally, these draft plans are induced into a final plan. DIP enhances zero-shot reasoning accuracy without reliance on resource-intensive sampling. Experiments show that DIP outperforms single-strategy prompting, demonstrating the effectiveness of multi-plan induction for prompt-based reasoning. 

---
# DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity 

**Authors**: Jitai Hao, Qiang Huang, Yaowei Wang, Min Zhang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08005)  

**Abstract**: The deployment of efficient long-context LLMs in applications like autonomous agents, long-chain reasoning, and creative writing is fundamentally bottlenecked by the linear growth of KV cache memory. Existing compression and eviction methods often struggle to balance accuracy, compression ratio, and hardware efficiency. We propose DeltaKV, a residual-based KV cache compression framework motivated by two empirical findings: long-range inter-token similarity and highly shared latent components in KV representations. Instead of discarding tokens, DeltaKV encodes semantic residuals relative to retrieved historical references, preserving fidelity while substantially reducing storage. To translate compression gains into real system speedups, we further introduce Sparse-vLLM, a high-performance inference engine with decoupled memory management and kernels optimized for sparse and irregular KV layouts. Experiments show that DeltaKV reduces KV cache memory to 29\% of the original while maintaining near-lossless accuracy on LongBench, SCBench, and AIME. When integrated with Sparse-vLLM, it achieves up to 2$\times$ throughput improvement over vLLM in long-context scenarios, demonstrating a practical path toward scalable long-context LLM deployment. Code, model checkpoints, and datasets are available at this https URL. 

---
# The Judge Who Never Admits: Hidden Shortcuts in LLM-based Evaluation 

**Authors**: Arash Marioriyad, Omid Ghahroodi, Ehsaneddin Asgari, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2602.07996)  

**Abstract**: Large language models (LLMs) are increasingly used as automatic judges to evaluate system outputs in tasks such as reasoning, question answering, and creative writing. A faithful judge should base its verdicts solely on content quality, remain invariant to irrelevant context, and transparently reflect the factors driving its decisions. We test this ideal via controlled cue perturbations-synthetic metadata labels injected into evaluation prompts-for six judge models: GPT-4o, Gemini-2.0-Flash, Gemma-3-27B, Qwen3-235B, Claude-3-Haiku, and Llama3-70B. Experiments span two complementary datasets with distinct evaluation regimes: ELI5 (factual QA) and LitBench (open-ended creative writing). We study six cue families: source, temporal, age, gender, ethnicity, and educational status. Beyond measuring verdict shift rates (VSR), we introduce cue acknowledgment rate (CAR) to quantify whether judges explicitly reference the injected cues in their natural-language rationales. Across cues with strong behavioral effects-e.g., provenance hierarchies (Expert > Human > LLM > Unknown), recency preferences (New > Old), and educational-status favoritism-CAR is typically at or near zero, indicating that shortcut reliance is largely unreported even when it drives decisions. Crucially, CAR is also dataset-dependent: explicit cue recognition is more likely to surface in the factual ELI5 setting for some models and cues, but often collapses in the open-ended LitBench regime, where large verdict shifts can persist despite zero acknowledgment. The combination of substantial verdict sensitivity and limited cue acknowledgment reveals an explanation gap in LLM-as-judge pipelines, raising concerns about reliability of model-based evaluation in both research and deployment. 

---
# Cross-Linguistic Persona-Driven Data Synthesis for Robust Multimodal Cognitive Decline Detection 

**Authors**: Rui Feng, Zhiyao Luo, Liuyu Wu, Wei Wang, Yuting Song, Yong Liu, Kok Pin Ng, Jianqing Li, Xingyao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07978)  

**Abstract**: Speech-based digital biomarkers represent a scalable, non-invasive frontier for the early identification of Mild Cognitive Impairment (MCI). However, the development of robust diagnostic models remains impeded by acute clinical data scarcity and a lack of interpretable reasoning. Current solutions frequently struggle with cross-lingual generalization and fail to provide the transparent rationales essential for clinical trust. To address these barriers, we introduce SynCog, a novel framework integrating controllable zero-shot multimodal data synthesis with Chain-of-Thought (CoT) deduction fine-tuning. Specifically, SynCog simulates diverse virtual subjects with varying cognitive profiles to effectively alleviate clinical data scarcity. This generative paradigm enables the rapid, zero-shot expansion of clinical corpora across diverse languages, effectively bypassing data bottlenecks in low-resource settings and bolstering the diagnostic performance of Multimodal Large Language Models (MLLMs). Leveraging this synthesized dataset, we fine-tune a foundational multimodal backbone using a CoT deduction strategy, empowering the model to explicitly articulate diagnostic thought processes rather than relying on black-box predictions. Extensive experiments on the ADReSS and ADReSSo benchmarks demonstrate that augmenting limited clinical data with synthetic phenotypes yields competitive diagnostic performance, achieving Macro-F1 scores of 80.67% and 78.46%, respectively, outperforming current baseline models. Furthermore, evaluation on an independent real-world Mandarin cohort (CIR-E) demonstrates robust cross-linguistic generalization, attaining a Macro-F1 of 48.71%. These findings constitute a critical step toward providing clinically trustworthy and linguistically inclusive cognitive assessment tools for global healthcare. 

---
# Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms 

**Authors**: Vaibhav Shukla, Hardik Sharma, Adith N Reganti, Soham Wasmatkar, Bagesh Kumar, Vrijendra Singh  

**Link**: [PDF](https://arxiv.org/pdf/2602.07963)  

**Abstract**: Most safety evaluations of large language models (LLMs) remain anchored in English. Translation is often used as a shortcut to probe multilingual behavior, but it rarely captures the full picture, especially when harmful intent or structure morphs across languages. Some types of harm survive translation almost intact, while others distort or disappear. To study this effect, we introduce CompositeHarm, a translation-based benchmark designed to examine how safety alignment holds up as both syntax and semantics shift. It combines two complementary English datasets, AttaQ, which targets structured adversarial attacks, and MMSafetyBench, which covers contextual, real-world harms, and extends them into six languages: English, Hindi, Assamese, Marathi, Kannada, and Gujarati. Using three large models, we find that attack success rates rise sharply in Indic languages, especially under adversarial syntax, while contextual harms transfer more moderately. To ensure scalability and energy efficiency, our study adopts lightweight inference strategies inspired by edge-AI design principles, reducing redundant evaluation passes while preserving cross-lingual fidelity. This design makes large-scale multilingual safety testing both computationally feasible and environmentally conscious. Overall, our results show that translated benchmarks are a necessary first step, but not a sufficient one, toward building grounded, resource-aware, language-adaptive safety systems. 

---
# Bielik Guard: Efficient Polish Language Safety Classifiers for LLM Content Moderation 

**Authors**: Krzysztof Wróbel, Jan Maria Kowalski, Jerzy Surma, Igor Ciuciura, Maciej Szymański  

**Link**: [PDF](https://arxiv.org/pdf/2602.07954)  

**Abstract**: As Large Language Models (LLMs) become increasingly deployed in Polish language applications, the need for efficient and accurate content safety classifiers has become paramount. We present Bielik Guard, a family of compact Polish language safety classifiers comprising two model variants: a 0.1B parameter model based on MMLW-RoBERTa-base and a 0.5B parameter model based on PKOBP/polish-roberta-8k. Fine-tuned on a community-annotated dataset of 6,885 Polish texts, these models classify content across five safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and Self-Harm. Our evaluation demonstrates that both models achieve strong performance on multiple benchmarks. The 0.5B variant offers the best overall discrimination capability with F1 scores of 0.791 (micro) and 0.785 (macro) on the test set, while the 0.1B variant demonstrates exceptional efficiency. Notably, Bielik Guard 0.1B v1.1 achieves superior precision (77.65\%) and very low false positive rate (0.63\%) on real user prompts, outperforming HerBERT-PL-Guard (31.55\% precision, 4.70\% FPR) despite identical model size. The models are publicly available and designed to provide appropriate responses rather than simple content blocking, particularly for sensitive categories like self-harm. 

---
# Patches of Nonlinearity: Instruction Vectors in Large Language Models 

**Authors**: Irina Bigoulaeva, Jonas Rohweder, Subhabrata Dutta, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2602.07930)  

**Abstract**: Despite the recent success of instruction-tuned language models and their ubiquitous usage, very little is known of how models process instructions internally. In this work, we address this gap from a mechanistic point of view by investigating how instruction-specific representations are constructed and utilized in different stages of post-training: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Via causal mediation, we identify that instruction representation is fairly localized in models. These representations, which we call Instruction Vectors (IVs), demonstrate a curious juxtaposition of linear separability along with non-linear causal interaction, broadly questioning the scope of the linear representation hypothesis commonplace in mechanistic interpretability. To disentangle the non-linear causal interaction, we propose a novel method to localize information processing in language models that is free from the implicit linear assumptions of patching-based techniques. We find that, conditioned on the task representations formed in the early layers, different information pathways are selected in the later layers to solve that task, i.e., IVs act as circuit selectors. 

---
# SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization 

**Authors**: Taolin Zhang, Hang Guo, Wang Lu, Tao Dai, Shu-Tao Xia, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07909)  

**Abstract**: As large language models (LLMs) continue to scale up, their performance on various downstream tasks has significantly improved. However, evaluating their capabilities has become increasingly expensive, as performing inference on a large number of benchmark samples incurs high computational costs. In this paper, we revisit the model-item performance matrix and show that it exhibits sparsity, that representative items can be selected as anchors, and that the task of efficient benchmarking can be formulated as a sparse optimization problem. Based on these insights, we propose SparseEval, a method that, for the first time, adopts gradient descent to optimize anchor weights and employs an iterative refinement strategy for anchor selection. We utilize the representation capacity of MLP to handle sparse optimization and propose the Anchor Importance Score and Candidate Importance Score to evaluate the value of each item for task-aware refinement. Extensive experiments demonstrate the low estimation error and high Kendall's~$\tau$ of our method across a variety of benchmarks, showcasing its superior robustness and practicality in real-world scenarios. Code is available at {this https URL}. 

---
# Evaluating and Calibrating LLM Confidence on Questions with Multiple Correct Answers 

**Authors**: Yuhan Wang, Shiyu Ni, Zhikai Ding, Zihang Zhan, Yuanzi Li, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07842)  

**Abstract**: Confidence calibration is essential for making large language models (LLMs) reliable, yet existing training-free methods have been primarily studied under single-answer question answering. In this paper, we show that these methods break down in the presence of multiple valid answers, where disagreement among equally correct responses leads to systematic underestimation of confidence. To enable a systematic study of this phenomenon, we introduce MACE, a benchmark of 12,000 factual questions spanning six domains with varying numbers of correct answers. Experiments across 15 representative calibration methods and four LLM families (7B-72B) reveal that while accuracy increases with answer cardinality, estimated confidence consistently decreases, causing severe miscalibration for questions with mixed answer counts. To address this issue, we propose Semantic Confidence Aggregation (SCA), which aggregates confidence over multiple high-probability sampled responses. SCA achieves state-of-the-art calibration performance under mixed-answer settings while preserving strong calibration on single-answer questions. 

---
# TodoEvolve: Learning to Architect Agent Planning Systems 

**Authors**: Jiaxi Liu, Yanzuo Jiang, Guibin Zhang, Zihan Zhang, Heng Chang, Zhenfei Yin, Qibing Ren, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07839)  

**Abstract**: Planning has become a central capability for contemporary agent systems in navigating complex, long-horizon tasks, yet existing approaches predominantly rely on fixed, hand-crafted planning structures that lack the flexibility to adapt to the structural diversity of open-ended problems. To address this limitation, we introduce TodoEvolve, a meta-planning paradigm that autonomously synthesizes and dynamically revises task-specific planning architectures. Specifically, we first construct PlanFactory, a modular design space that standardizes diverse planning paradigms within a unified codebase encompassing topology, initialization, adaptation, and navigation, thereby providing a common interface for heterogeneous planning patterns. Leveraging PlanFactory, we collect high-quality planning trajectories and train Todo-14B via \textit{Impedance-Guided Preference Optimization} (IGPO), a multi-objective reinforcement learning objective that encourages the generation of planning systems that are performant, stable, and token-efficient across arbitrary tasks and agent backbones. Empirical evaluations on five agentic benchmarks demonstrate that TodoEvolve consistently surpasses carefully engineered planning modules while maintaining economical API costs and runtime overhead. 

---
# LLMs Know More About Numbers than They Can Say 

**Authors**: Fengting Yuchi, Li Du, Jason Eisner  

**Link**: [PDF](https://arxiv.org/pdf/2602.07812)  

**Abstract**: Although state-of-the-art LLMs can solve math problems, we find that they make errors on numerical comparisons with mixed notation: "Which is larger, $5.7 \times 10^2$ or $580$?" This raises a fundamental question: Do LLMs even know how big these numbers are? We probe the hidden states of several smaller open-source LLMs. A single linear projection of an appropriate hidden layer encodes the log-magnitudes of both kinds of numerals, allowing us to recover the numbers with relative error of about 2.3% (on restricted synthetic text) or 19.06% (on scientific papers). Furthermore, the hidden state after reading a pair of numerals encodes their ranking, with a linear classifier achieving over 90% accuracy. Yet surprisingly, when explicitly asked to rank the same pairs of numerals, these LLMs achieve only 50-70% accuracy, with worse performance for models whose probes are less effective. Finally, we show that incorporating the classifier probe's log-loss as an auxiliary objective during finetuning brings an additional 3.22% improvement in verbalized accuracy over base models, demonstrating that improving models' internal magnitude representations can enhance their numerical reasoning capabilities. 

---
# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models 

**Authors**: Xuan Ding, Pengyu Tong, Ranjie Duan, Yunjian Zhang, Rui Sun, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07804)  

**Abstract**: While large language models (LLMs) demonstrate impressive performance across various tasks, their deployment in real-world scenarios is still constrained by high computational demands. Layer-wise pruning, a commonly employed strategy to mitigate inference costs, can partially address this challenge. However, existing approaches generally depend on static heuristic rules and fail to account for the interdependencies among layers, thereby limiting the effectiveness of the pruning process. To this end, this paper proposes a game-theoretic framework that formulates layer pruning as a cooperative game in which each layer acts as a player and model performance serves as the utility. As computing exact Shapley values is computationally infeasible for large language models (LLMs), we propose using a lightweight surrogate network to estimate layer-wise marginal contributions. This network can predict LLM performance for arbitrary layer combinations at a low computational cost. Additionally, we employ stratified Monte Carlo mask sampling to further reduce the cost of Sharpley value estimation. This approach captures inter-layer dependencies and dynamically identifies critical layers for pruning. Extensive experiments demonstrate the consistent superiority of our method in terms of perplexity and zero-shot accuracy, achieving more efficient and effective layer-wise pruning for large language models. 

---
# Thinking Makes LLM Agents Introverted: How Mandatory Thinking Can Backfire in User-Engaged Agents 

**Authors**: Jiatong Li, Changdae Oh, Hyeong Kyu Choi, Jindong Wang, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.07796)  

**Abstract**: Eliciting reasoning has emerged as a powerful technique for improving the performance of large language models (LLMs) on complex tasks by inducing thinking. However, their effectiveness in realistic user-engaged agent scenarios remains unclear. In this paper, we conduct a comprehensive study on the effect of explicit thinking in user-engaged LLM agents. Our experiments span across seven models, three benchmarks, and two thinking instantiations, and we evaluate them through both a quantitative response taxonomy analysis and qualitative failure propagation case studies. Contrary to expectations, we find that mandatory thinking often backfires on agents in user-engaged settings, causing anomalous performance degradation across various LLMs. Our key finding reveals that thinking makes agents more ``introverted'' by shortening responses and reducing information disclosure to users, which weakens agent-user information exchange and leads to downstream task failures. Furthermore, we demonstrate that explicitly prompting for information disclosure reliably improves performance across diverse model families, suggesting that proactive transparency is a vital lever for agent optimization. Overall, our study suggests that information transparency awareness is a crucial yet underexplored perspective for the future design of reasoning agents in real-world scenarios. Our code is available at this https URL. 

---
# Emergent Structured Representations Support Flexible In-Context Inference in Large Language Models 

**Authors**: Ningyu Xu, Qi Zhang, Xipeng Qiu, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07794)  

**Abstract**: Large language models (LLMs) exhibit emergent behaviors suggestive of human-like reasoning. While recent work has identified structured, human-like conceptual representations within these models, it remains unclear whether they functionally rely on such representations for reasoning. Here we investigate the internal processing of LLMs during in-context concept inference. Our results reveal a conceptual subspace emerging in middle to late layers, whose representational structure persists across contexts. Using causal mediation analyses, we demonstrate that this subspace is not merely an epiphenomenon but is functionally central to model predictions, establishing its causal role in inference. We further identify a layer-wise progression where attention heads in early-to-middle layers integrate contextual cues to construct and refine the subspace, which is subsequently leveraged by later layers to generate predictions. Together, these findings provide evidence that LLMs dynamically construct and use structured, latent representations in context for inference, offering insights into the computational processes underlying flexible adaptation. 

---
# Attn-GS: Attention-Guided Context Compression for Efficient Personalized LLMs 

**Authors**: Shenglai Zeng, Tianqi Zheng, Chuan Tian, Dante Everaert, Yau-Shian Wang, Yupin Huang, Michael J. Morais, Rohit Patki, Jinjin Tian, Xinnan Dai, Kai Guo, Monica Xiao Cheng, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07778)  

**Abstract**: Personalizing large language models (LLMs) to individual users requires incorporating extensive interaction histories and profiles, but input token constraints make this impractical due to high inference latency and API costs. Existing approaches rely on heuristic methods such as selecting recent interactions or prompting summarization models to compress user profiles. However, these methods treat context as a monolithic whole and fail to consider how LLMs internally process and prioritize different profile components. We investigate whether LLMs' attention patterns can effectively identify important personalization signals for intelligent context compression. Through preliminary studies on representative personalization tasks, we discover that (a) LLMs' attention patterns naturally reveal important signals, and (b) fine-tuning enhances LLMs' ability to distinguish between relevant and irrelevant information. Based on these insights, we propose Attn-GS, an attention-guided context compression framework that leverages attention feedback from a marking model to mark important personalization sentences, then guides a compression model to generate task-relevant, high-quality compressed user contexts. Extensive experiments demonstrate that Attn-GS significantly outperforms various baselines across different tasks, token limits, and settings, achieving performance close to using full context while reducing token usage by 50 times. 

---
# SRR-Judge: Step-Level Rating and Refinement for Enhancing Search-Integrated Reasoning in Search Agents 

**Authors**: Chen Zhang, Kuicai Dong, Dexun Li, Wenjun Li, Qu Yang, Wei Han, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07773)  

**Abstract**: Recent deep search agents built on large reasoning models (LRMs) excel at complex question answering by iteratively planning, acting, and gathering evidence, a capability known as search-integrated reasoning. However, mainstream approaches often train this ability using only outcome-based supervision, neglecting the quality of intermediate thoughts and actions. We introduce SRR-Judge, a framework for reliable step-level assessment of reasoning and search actions. Integrated into a modified ReAct-style rate-and-refine workflow, SRR-Judge provides fine-grained guidance for search-integrated reasoning and enables efficient post-training annotation. Using SRR-annotated data, we apply an iterative rejection sampling fine-tuning procedure to enhance the deep search capability of the base agent. Empirically, SRR-Judge delivers more reliable step-level evaluations than much larger models such as DeepSeek-V3.1, with its ratings showing strong correlation with final answer correctness. Moreover, aligning the policy with SRR-Judge annotated trajectories leads to substantial performance gains, yielding over a 10 percent average absolute pass@1 improvement across challenging deep search benchmarks. 

---
# Blind to the Human Touch: Overlap Bias in LLM-Based Summary Evaluation 

**Authors**: Jiangnan Fang, Cheng-Tse Liu, Hanieh Deilamsalehy, Nesreen K. Ahmed, Puneet Mathur, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07673)  

**Abstract**: Large language model (LLM) judges have often been used alongside traditional, algorithm-based metrics for tasks like summarization because they better capture semantic information, are better at reasoning, and are more robust to paraphrasing. However, LLM judges show biases for length and order among others, and are vulnerable to various adversarial input prompts. While recent studies have looked into these biases, few have analyzed them at a more granular level in relation to a well-defined overlap metric. In this work we provide an LLM judge bias analysis as a function of overlap with human-written responses in the domain of summarization. We test 9 recent LLMs with parameter counts ranging from 1 billion to 12 billion, including variants of Gemma 3 and LLaMA 3. We find that LLM judges increasingly prefer summaries generated by other LLMs over those written by humans as the similarities (as measured by ROUGE and BLEU) between the judged summaries decrease, and this pattern extends to all but one model tested, and exists regardless of the models' own position biases. Additionally, we find that models struggle to judge even summaries with limited overlaps, suggesting that LLM-as-a-judge in the summary domain should rely on techniques beyond a simple comparison. 

---
# Letting Tutor Personas "Speak Up" for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization 

**Authors**: Jaewook Lee, Alexander Scarlatos, Simon Woodhead, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07639)  

**Abstract**: With the emergence of large language models (LLMs) as a powerful class of generative artificial intelligence (AI), their use in tutoring has become increasingly prominent. Prior works on LLM-based tutoring typically learn a single tutor policy and do not capture the diversity of tutoring styles. In real-world tutor-student interactions, pedagogical intent is realized through adaptive instructional strategies, with tutors varying the level of scaffolding, instructional directiveness, feedback, and affective support in response to learners' needs. These differences can all impact dialogue dynamics and student engagement. In this paper, we explore how tutor personas embedded in human tutor-student dialogues can be used to guide LLM behavior without relying on explicitly prompted instructions. We modify Bidirectional Preference Optimization (BiPO) to learn a steering vector, an activation-space direction that steers model responses towards certain tutor personas. We find that this steering vector captures tutor-specific variation across dialogue contexts, improving semantic alignment with ground-truth tutor utterances and increasing preference-based evaluations, while largely preserving lexical similarity. Analysis of the learned directional coefficients further reveals interpretable structure across tutors, corresponding to consistent differences in tutoring behavior. These results demonstrate that activation steering offers an effective and interpretable way for controlling tutor-specific variation in LLMs using signals derived directly from human dialogue data. 

---
# SciClaimEval: Cross-modal Claim Verification in Scientific Papers 

**Authors**: Xanh Ho, Yun-Ang Wu, Sunisth Kumar, Tian Cheng Xia, Florian Boudin, Andre Greiner-Petter, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2602.07621)  

**Abstract**: We present SciClaimEval, a new scientific dataset for the claim verification task. Unlike existing resources, SciClaimEval features authentic claims, including refuted ones, directly extracted from published papers. To create refuted claims, we introduce a novel approach that modifies the supporting evidence (figures and tables), rather than altering the claims or relying on large language models (LLMs) to fabricate contradictions. The dataset provides cross-modal evidence with diverse representations: figures are available as images, while tables are provided in multiple formats, including images, LaTeX source, HTML, and JSON. SciClaimEval contains 1,664 annotated samples from 180 papers across three domains, machine learning, natural language processing, and medicine, validated through expert annotation. We benchmark 11 multimodal foundation models, both open-source and proprietary, across the dataset. Results show that figure-based verification remains particularly challenging for all models, as a substantial performance gap remains between the best system and human baseline. 

---
# Learning to Self-Verify Makes Language Models Better Reasoners 

**Authors**: Yuxin Chen, Yu Wang, Yi Zhang, Ziang Ye, Zhengzhou Cai, Yaorui Shi, Qi Gu, Hui Su, Xunliang Cai, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.07594)  

**Abstract**: Recent large language models (LLMs) achieve strong performance in generating promising reasoning paths for complex tasks. However, despite powerful generation ability, LLMs remain weak at verifying their own answers, revealing a persistent capability asymmetry between generation and self-verification. In this work, we conduct an in-depth investigation of this asymmetry throughout training evolution and show that, even on the same task, improving generation does not lead to corresponding improvements in self-verification. Interestingly, we find that the reverse direction of this asymmetry behaves differently: learning to self-verify can effectively improve generation performance, achieving accuracy comparable to standard generation training while yielding more efficient and effective reasoning traces. Building on this observation, we further explore integrating self-verification into generation training by formulating a multi-task reinforcement learning framework, where generation and self-verification are optimized as two independent but complementary objectives. Extensive experiments across benchmarks and models demonstrate performance gains over generation-only training in both generation and verification capabilities. 

---
# Improving Variable-Length Generation in Diffusion Language Models via Length Regularization 

**Authors**: Zicong Cheng, Ruixuan Jia, Jia Li, Guo-Wei Yang, Meng-Hao Guo, Shi-Min Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07546)  

**Abstract**: Diffusion Large Language Models (DLLMs) are inherently ill-suited for variable-length generation, as their inference is defined on a fixed-length canvas and implicitly assumes a known target length. When the length is unknown, as in realistic completion and infilling, naively comparing confidence across mask lengths becomes systematically biased, leading to under-generation or redundant continuations. In this paper, we show that this failure arises from an intrinsic lengthinduced bias in generation confidence estimates, leaving existing DLLMs without a robust way to determine generation length and making variablelength inference unreliable. To address this issue, we propose LR-DLLM, a length-regularized inference framework for DLLMs that treats generation length as an explicit variable and achieves reliable length determination at inference time. It decouples semantic compatibility from lengthinduced uncertainty through an explicit length regularization that corrects biased confidence estimates. Based on this, LR-DLLM enables dynamic expansion or contraction of the generation span without modifying the underlying DLLM or its training procedure. Experiments show that LRDLLM achieves 51.3% Pass@1 on HumanEvalInfilling under fully unknown lengths (+13.4% vs. DreamOn) and 51.5% average Pass@1 on four-language McEval (+14.3% vs. DreamOn). 

---
# Let's Simplify Step by Step: Guiding LLM Towards Multilingual Unsupervised Proficiency-Controlled Sentence Simplification 

**Authors**: Jingshen Zhang, Xin Ying Qiu, Lifang Lu, Zhuhua Huang, Yutao Hu, Yuechang Wu, JunYu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07499)  

**Abstract**: Large language models demonstrate limited capability in proficiency-controlled sentence simplification, particularly when simplifying across large readability levels. We propose a framework that decomposes complex simplifications into manageable steps through dynamic path planning, semantic-aware exemplar selection, and chain-of-thought generation with conversation history for coherent reasoning. Evaluation on five languages across two benchmarks shows our approach improves simplification effectiveness while reducing computational steps by 22-42%. Human evaluation confirms the fundamental trade-off between simplification effectiveness and meaning preservation. Notably, even human annotators struggle to agree on semantic preservation judgments, highlighting the inherent complexity of this task. Our work shows that while step-by-step simplification improves control, preserving semantic fidelity during extensive simplification remains an open challenge. 

---
# From Native Memes to Global Moderation: Cros-Cultural Evaluation of Vision-Language Models for Hateful Meme Detection 

**Authors**: Mo Wang, Kaixuan Ren, Pratik Jalan, Ahmed Ashraf, Tuong Vy Vu, Rahul Seetharaman, Shah Nawaz, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2602.07497)  

**Abstract**: Cultural context profoundly shapes how people interpret online content, yet vision-language models (VLMs) remain predominantly trained through Western or English-centric lenses. This limits their fairness and cross-cultural robustness in tasks like hateful meme detection. We introduce a systematic evaluation framework designed to diagnose and quantify the cross-cultural robustness of state-of-the-art VLMs across multilingual meme datasets, analyzing three axes: (i) learning strategy (zero-shot vs. one-shot), (ii) prompting language (native vs. English), and (iii) translation effects on meaning and detection. Results show that the common ``translate-then-detect'' approach deteriorate performance, while culturally aligned interventions - native-language prompting and one-shot learning - significantly enhance detection. Our findings reveal systematic convergence toward Western safety norms and provide actionable strategies to mitigate such bias, guiding the design of globally robust multimodal moderation systems. 

---
# SED-SFT: Selectively Encouraging Diversity in Supervised Fine-Tuning 

**Authors**: Yijie Chen, Yijin Liu, Fandong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07464)  

**Abstract**: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) has emerged as the standard post-training paradigm for large language models (LLMs). However, the conventional SFT process, driven by Cross-Entropy (CE) loss, often induces mode collapse, where models over-concentrate on specific response patterns. This lack of distributional diversity severely restricts the exploration efficiency required for subsequent RL. While recent studies have attempted to improve SFT by replacing the CE loss, aiming to preserve diversity or refine the update policy, they fail to adequately balance diversity and accuracy, thereby yielding suboptimal performance after RL. To address the mode collapse problem, we propose SED-SFT, which adaptively encourages diversity based on the token exploration space. This framework introduces a selective entropy regularization term with a selective masking mechanism into the optimization objective. Extensive experiments across eight mathematical benchmarks demonstrate that SED-SFT significantly enhances generation diversity with a negligible computational overhead increase compared with CE loss, yielding average improvements of 2.06 and 1.20 points in subsequent RL performance over standard CE-based baselines on Llama-3.2-3B-Instruct and Qwen2.5-Math-7B-Instruct, respectively. The code is publicly available at this https URL 

---
# DLLM Agent: See Farther, Run Faster 

**Authors**: Huiling Zhen, Weizhe Lin, Renxi Liu, Kai Han, Yiming Li, Yuchuan Tian, Hanting Chen, Xiaoguang Li, Xiaosong Li, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Youliang Yan, Peifeng Qin, Jun Wang, Yu Wang, Dacheng Tao, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07451)  

**Abstract**: Diffusion large language models (DLLMs) have emerged as an alternative to autoregressive (AR) decoding with appealing efficiency and modeling properties, yet their implications for agentic multi-step decision making remain underexplored. We ask a concrete question: when the generation paradigm is changed but the agent framework and supervision are held fixed, do diffusion backbones induce systematically different planning and tool-use behaviors, and do these differences translate into end-to-end efficiency gains? We study this in a controlled setting by instantiating DLLM and AR backbones within the same agent workflow (DeepDiver) and performing matched agent-oriented fine-tuning on the same trajectory data, yielding diffusion-backed DLLM Agents and directly comparable AR agents. Across benchmarks and case studies, we find that, at comparable accuracy, DLLM Agents are on average over 30% faster end to end than AR agents, with some cases exceeding 8x speedup. Conditioned on correct task completion, DLLM Agents also require fewer interaction rounds and tool invocations, consistent with higher planner hit rates that converge earlier to a correct action path with less backtracking. We further identify two practical considerations for deploying diffusion backbones in tool-using agents. First, naive DLLM policies are more prone to structured tool-call failures, necessitating stronger tool-call-specific training to emit valid schemas and arguments. Second, for multi-turn inputs interleaving context and action spans, diffusion-style span corruption requires aligned attention masking to avoid spurious context-action information flow; without such alignment, performance degrades. Finally, we analyze attention dynamics across workflow stages and observe paradigm-specific coordination patterns, suggesting stronger global planning signals in diffusion-backed agents. 

---
# Measuring cross-language intelligibility between Romance languages with computational tools 

**Authors**: Liviu P Dinu, Ana Sabina Uban, Bogdan Iordache, Anca Dinu, Simona Georgescu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07447)  

**Abstract**: We present an analysis of mutual intelligibility in related languages applied for languages in the Romance family. We introduce a novel computational metric for estimating intelligibility based on lexical similarity using surface and semantic similarity of related words, and use it to measure mutual intelligibility for the five main Romance languages (French, Italian, Portuguese, Spanish, and Romanian), and compare results using both the orthographic and phonetic forms of words as well as different parallel corpora and vectorial models of word meaning representation. The obtained intelligibility scores confirm intuitions related to intelligibility asymmetry across languages and significantly correlate with results of cloze tests in human experiments. 

---
# Advantages of Domain Knowledge Injection for Legal Document Summarization: A Case Study on Summarizing Indian Court Judgments in English and Hindi 

**Authors**: Debtanu Datta, Rajdeep Mukherjee, Adrijit Goswami, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2602.07382)  

**Abstract**: Summarizing Indian legal court judgments is a complex task not only due to the intricate language and unstructured nature of the legal texts, but also since a large section of the Indian population does not understand the complex English in which legal text is written, thus requiring summaries in Indian languages. In this study, we aim to improve the summarization of Indian legal text to generate summaries in both English and Hindi (the most widely spoken Indian language), by injecting domain knowledge into diverse summarization models. We propose a framework to enhance extractive neural summarization models by incorporating domain-specific pre-trained encoders tailored for legal texts. Further, we explore the injection of legal domain knowledge into generative models (including Large Language Models) through continual pre-training on large legal corpora in English and Hindi. Our proposed approaches achieve statistically significant improvements in both English-to-English and English-to-Hindi Indian legal document summarization, as measured by standard evaluation metrics, factual consistency metrics, and legal domain-specific metrics. Furthermore, these improvements are validated through domain experts, demonstrating the effectiveness of our approaches. 

---
# When the Model Said 'No Comment', We Knew Helpfulness Was Dead, Honesty Was Alive, and Safety Was Terrified 

**Authors**: Gautam Siddharth Kashyap, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2602.07381)  

**Abstract**: Large Language Models (LLMs) need to be in accordance with human values-being helpful, harmless, and honest (HHH)-is important for safe deployment. Existing works use Supervised Fine-Tuning (SFT) and Mixture-of-Experts (MoE) to align LLMs. However, these works face challenges in multi-objective settings, such as SFT leading to interference between conflicting objectives, while MoEs suffer from miscalibrated routing. We term this failure mode Axis Collapse, marked by (1) disjoint feature spaces causing catastrophic forgetting, and (2) unreliable inference from misrouted experts. To resolve this, we propose AlignX, a two-stage framework. Stage 1 uses prompt-injected fine-tuning to extract axis-specific task features, mitigating catastrophic forgetting. Stage 2 deploys a MoCaE module that calibrates expert routing using fractal and natural geometry, improving inference reliability. AlignX achieves significant gains on Alpaca (Helpfulness), BeaverTails (Harmlessness), and TruthfulQA (Honesty), with +171.5% win rate, +110.1% in truthfulness-informativeness, and 4.3% fewer safety violations. It also reduces latency and memory usage by over 35% compared to prior MoEs. Results across four LLMs validate its generalizability. 

---
# Do Large Language Models Reflect Demographic Pluralism in Safety? 

**Authors**: Usman Naseem, Gautam Siddharth Kashyap, Sushant Kumar Ray, Rafiq Ali, Ebad Shabbir, Abdullah Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2602.07376)  

**Abstract**: Large Language Model (LLM) safety is inherently pluralistic, reflecting variations in moral norms, cultural expectations, and demographic contexts. Yet, existing alignment datasets such as ANTHROPIC-HH and DICES rely on demographically narrow annotator pools, overlooking variation in safety perception across communities. Demo-SafetyBench addresses this gap by modeling demographic pluralism directly at the prompt level, decoupling value framing from responses. In Stage I, prompts from DICES are reclassified into 14 safety domains (adapted from BEAVERTAILS) using Mistral 7B-Instruct-v0.3, retaining demographic metadata and expanding low-resource domains via Llama-3.1-8B-Instruct with SimHash-based deduplication, yielding 43,050 samples. In Stage II, pluralistic sensitivity is evaluated using LLMs-as-Raters-Gemma-7B, GPT-4o, and LLaMA-2-7B-under zero-shot inference. Balanced thresholds (delta = 0.5, tau = 10) achieve high reliability (ICC = 0.87) and low demographic sensitivity (DS = 0.12), confirming that pluralistic safety evaluation can be both scalable and demographically robust. 

---
# Efficient Post-Training Pruning of Large Language Models with Statistical Correction 

**Authors**: Peiqi Yu, Jinhao Wang, Xinyi Sui, Nam Ling, Wei Wang, Wei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07375)  

**Abstract**: Post-training pruning is an effective approach for reducing the size and inference cost of large language models (LLMs), but existing methods often face a trade-off between pruning quality and computational efficiency. Heuristic pruning methods are efficient but sensitive to activation outliers, while reconstruction-based approaches improve fidelity at the cost of heavy computation. In this work, we propose a lightweight post-training pruning framework based on first-order statistical properties of model weights and activations. During pruning, channel-wise statistics are used to calibrate magnitude-based importance scores, reducing bias from activation-dominated channels. After pruning, we apply an analytic energy compensation to correct distributional distortions caused by weight removal. Both steps operate without retraining, gradients, or second-order information. Experiments across multiple LLM families, sparsity patterns, and evaluation tasks show that the proposed approach improves pruning performance while maintaining computational cost comparable to heuristic methods. The results suggest that simple statistical corrections can be effective for post-training pruning of LLMs. 

---
# TernaryLM: Memory-Efficient Language Modeling via Native 1-Bit Quantization with Adaptive Layer-wise Scaling 

**Authors**: Nisharg Nargund, Priyesh Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2602.07374)  

**Abstract**: Large language models (LLMs) achieve remarkable performance but demand substantial computational resources, limiting deployment on edge devices and resource-constrained environments. We present TernaryLM, a 132M parameter transformer architecture that employs native 1-bit ternary quantization {-1, 0, +1} during training, achieving significant memory reduction without sacrificing language modeling capability. Unlike post-training quantization approaches that quantize pre-trained full-precision models, TernaryLM learns quantization-aware representations from scratch using straight-through estimators and adaptive per-layer scaling factors. Our experiments demonstrate: (1) validation perplexity of 58.42 on TinyStories; (2) downstream transfer with 82.47 percent F1 on MRPC paraphrase detection; (3) 2.4x memory reduction (498MB vs 1197MB) with comparable inference latency; and (4) stable training dynamics across diverse corpora. We provide layer-wise quantization analysis showing that middle transformer layers exhibit highest compatibility with extreme quantization, informing future non-uniform precision strategies. Our results suggest that native 1-bit training is a promising direction for efficient neural language models. Code is available at this https URL. 

---
# ViHERMES: A Graph-Grounded Multihop Question Answering Benchmark and System for Vietnamese Healthcare Regulations 

**Authors**: Long S. T. Nguyen, Quan M. Bui, Tin T. Ngo, Quynh T. N. Vo, Dung N. H. Le, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07361)  

**Abstract**: Question Answering (QA) over regulatory documents is inherently challenging due to the need for multihop reasoning across legally interdependent texts, a requirement that is particularly pronounced in the healthcare domain where regulations are hierarchically structured and frequently revised through amendments and cross-references. Despite recent progress in retrieval-augmented and graph-based QA methods, systematic evaluation in this setting remains limited, especially for low-resource languages such as Vietnamese, due to the lack of benchmark datasets that explicitly support multihop reasoning over healthcare regulations. In this work, we introduce the Vietnamese Healthcare Regulations-Multihop Reasoning Dataset (ViHERMES), a benchmark designed for multihop QA over Vietnamese healthcare regulatory documents. ViHERMES consists of high-quality question-answer pairs that require reasoning across multiple regulations and capture diverse dependency patterns, including amendment tracing, cross-document comparison, and procedural synthesis. To construct the dataset, we propose a controlled multihop QA generation pipeline based on semantic clustering and graph-inspired data mining, followed by large language model-based generation with structured evidence and reasoning annotations. We further present a graph-aware retrieval framework that models formal legal relations at the level of legal units and supports principled context expansion for legally valid and coherent answers. Experimental results demonstrate that ViHERMES provides a challenging benchmark for evaluating multihop regulatory QA systems and that the proposed graph-aware approach consistently outperforms strong retrieval-based baselines. The ViHERMES dataset and system implementation are publicly available at this https URL. 

---
# Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation 

**Authors**: Geng Liu, Fei Zhu, Rong Feng, Changyi Ma, Shiqi Wang, Gaofeng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07338)  

**Abstract**: Multi-turn conversation has emerged as a predominant interaction paradigm for Large Language Models (LLMs). Users often employ follow-up questions to refine their intent, expecting LLMs to adapt dynamically. However, recent research reveals that LLMs suffer a substantial performance drop in multi-turn settings compared to single-turn interactions with fully specified instructions, a phenomenon termed ``Lost in Conversation'' (LiC). While this prior work attributes LiC to model unreliability, we argue that the root cause lies in an intent alignment gap rather than intrinsic capability deficits. In this paper, we first demonstrate that LiC is not a failure of model capability but rather a breakdown in interaction between users and LLMs. We theoretically show that scaling model size or improving training alone cannot resolve this gap, as it arises from structural ambiguity in conversational context rather than representational limitations. To address this, we propose to decouple intent understanding from task execution through a Mediator-Assistant architecture. By utilizing an experience-driven Mediator to explicate user inputs into explicit, well-structured instructions based on historical interaction patterns, our approach effectively bridges the gap between vague user intent and model interpretation. Experimental results demonstrate that this method significantly mitigates performance degradation in multi-turn conversations across diverse LLMs. 

---
# Beyond Accuracy: Risk-Sensitive Evaluation of Hallucinated Medical Advice 

**Authors**: Savan Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07319)  

**Abstract**: Large language models are increasingly being used in patient-facing medical question answering, where hallucinated outputs can vary widely in potential harm. However, existing hallucination standards and evaluation metrics focus primarily on factual correctness, treating all errors as equally severe. This obscures clinically relevant failure modes, particularly when models generate unsupported but actionable medical language. We propose a risk-sensitive evaluation framework that quantifies hallucinations through the presence of risk-bearing language, including treatment directives, contraindications, urgency cues, and mentions of high-risk medications. Rather than assessing clinical correctness, our approach evaluates the potential impact of hallucinated content if acted upon. We further combine risk scoring with a relevance measure to identify high-risk, low-grounding failures. We apply this framework to three instruction-tuned language models using controlled patient-facing prompts designed as safety stress tests. Our results show that models with similar surface-level behavior exhibit substantially different risk profiles and that standard evaluation metrics fail to capture these distinctions. These findings highlight the importance of incorporating risk sensitivity into hallucination evaluation and suggest that evaluation validity is critically dependent on task and prompt design. 

---
# Equipping LLM with Directional Multi-Talker Speech Understanding Capabilities 

**Authors**: Ju Lin, Jing Pan, Ruizhi Li, Ming Sun, Yuzong Liu, Alaa Hassan, Jing Zheng, Florian Metze  

**Link**: [PDF](https://arxiv.org/pdf/2602.07211)  

**Abstract**: Recent studies have demonstrated that prompting large language models (LLM) with audio encodings enables effective speech understanding capabilities. However, most speech LLMs are trained on single-channel, single-talker data, which makes it challenging to directly apply them to multi-talker and multi-channel speech understanding task. In this work, we present a comprehensive investigation on how to enable directional multi-talker speech understanding capabilities for LLMs, specifically in smart glasses usecase. We propose two novel approaches to integrate directivity into LLMs: (1) a cascaded system that leverages a source separation front-end module, and (2) an end-to-end system that utilizes serialized output training. All of the approaches utilize a multi-microphone array embedded in smart glasses to optimize directivity interpretation and processing in a streaming manner. Experimental results demonstrate the efficacy of our proposed methods in endowing LLMs with directional speech understanding capabilities, achieving strong performance in both speech recognition and speech translation tasks. 

---
# Long-Context Long-Form Question Answering for Legal Domain 

**Authors**: Anagha Kulkarni, Parin Rajesh Jhaveri, Prasha Shrestha, Yu Tong Han, Reza Amini, Behrouz Madahian  

**Link**: [PDF](https://arxiv.org/pdf/2602.07190)  

**Abstract**: Legal documents have complex document layouts involving multiple nested sections, lengthy footnotes and further use specialized linguistic devices like intricate syntax and domain-specific vocabulary to ensure precision and authority. These inherent characteristics of legal documents make question answering challenging, and particularly so when the answer to the question spans several pages (i.e. requires long-context) and is required to be comprehensive (i.e. a long-form answer). In this paper, we address the challenges of long-context question answering in context of long-form answers given the idiosyncrasies of legal documents. We propose a question answering system that can (a) deconstruct domain-specific vocabulary for better retrieval from source documents, (b) parse complex document layouts while isolating sections and footnotes and linking them appropriately, (c) generate comprehensive answers using precise domain-specific vocabulary. We also introduce a coverage metric that classifies the performance into recall-based coverage categories allowing human users to evaluate the recall with ease. We curate a QA dataset by leveraging the expertise of professionals from fields such as law and corporate tax. Through comprehensive experiments and ablation studies, we demonstrate the usability and merit of the proposed system. 

---
# Can LLMs Discern the Traits Influencing Your Preferences? Evaluating Personality-Driven Preference Alignment in LLMs 

**Authors**: Tianyu Zhao, Siqi Li, Yasser Shoukry, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.07181)  

**Abstract**: User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet how to reliably leverage preference signals for answer generation remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we study personality as a principled ''latent'' signal behind preference statements. Through extensive experiments, we find that conditioning on personality-aligned preferences substantially improves personalized question answering: selecting preferences consistent with a user's inferred personality increases answer-choice accuracy from 29.25% to 76%, compared to using randomly selected preferences. Based on these findings, we introduce PACIFIC (Preference Alignment Choices Inference for Five-factor Identity Characterization), a personality-labeled preference dataset containing 1200 preference statements spanning diverse domains (e.g., travel, movies, education), annotated with Big-Five (OCEAN) trait directions. Finally, we propose a framework that enables an LLM model to automatically retrieve personality-aligned preferences and incorporate them during answer generation. 

---
# Open TutorAI: An Open-source Platform for Personalized and Immersive Learning with Generative AI 

**Authors**: Mohamed El Hajji, Tarek Ait Baha, Aicha Dakir, Hammou Fadili, Youssef Es-Saady  

**Link**: [PDF](https://arxiv.org/pdf/2602.07176)  

**Abstract**: Recent advances in artificial intelligence have created new possibilities for making education more scalable, adaptive, and learner-centered. However, existing educational chatbot systems often lack contextual adaptability, real-time responsiveness, and pedagogical agility. which can limit learner engagement and diminish instructional effectiveness. Thus, there is a growing need for open, integrative platforms that combine AI and immersive technologies to support personalized, meaningful learning experiences. This paper presents Open TutorAI, an open-source educational platform based on LLMs and generative technologies that provides dynamic, personalized tutoring. The system integrates natural language processing with customizable 3D avatars to enable multimodal learner interaction. Through a structured onboarding process, it captures each learner's goals and preferences in order to configure a learner-specific AI assistant. This assistant is accessible via both text-based and avatar-driven interfaces. The platform includes tools for organizing content, providing embedded feedback, and offering dedicated interfaces for learners, educators, and parents. This work focuses on learner-facing components, delivering a tool for adaptive support that responds to individual learner profiles without requiring technical expertise. Its assistant-generation pipeline and avatar integration enhance engagement and emotional presence, creating a more humanized, immersive learning environment. Embedded learning analytics support self-regulated learning by tracking engagement patterns and generating actionable feedback. The result is Open TutorAI, which unites modular architecture, generative AI, and learner analytics within an open-source framework. It contributes to the development of next-generation intelligent tutoring systems. 

---
# Your Language Model Secretly Contains Personality Subnetworks 

**Authors**: Ruimeng Ye, Zihan Wang, Zinan Ling, Yang Xiao, Manling Li, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2602.07164)  

**Abstract**: Humans shift between different personas depending on social context. Large Language Models (LLMs) demonstrate a similar flexibility in adopting different personas and behaviors. Existing approaches, however, typically adapt such behavior through external knowledge such as prompting, retrieval-augmented generation (RAG), or fine-tuning. We ask: do LLMs really need external context or parameters to adapt to different behaviors, or do they already have such knowledge embedded in their parameters? In this work, we show that LLMs already contain persona-specialized subnetworks in their parameter space. Using small calibration datasets, we identify distinct activation signatures associated with different personas. Guided by these statistics, we develop a masking strategy that isolates lightweight persona subnetworks. Building on the findings, we further discuss: how can we discover opposing subnetwork from the model that lead to binary-opposing personas, such as introvert-extrovert? To further enhance separation in binary opposition scenarios, we introduce a contrastive pruning strategy that identifies parameters responsible for the statistical divergence between opposing personas. Our method is entirely training-free and relies solely on the language model's existing parameter space. Across diverse evaluation settings, the resulting subnetworks exhibit significantly stronger persona alignment than baselines that require external knowledge while being more efficient. Our findings suggest that diverse human-like behaviors are not merely induced in LLMs, but are already embedded in their parameter space, pointing toward a new perspective on controllable and interpretable personalization in large language models. 

---
# Free Energy Mixer 

**Authors**: Jiecheng Lu, Shihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07160)  

**Abstract**: Standard attention stores keys/values losslessly but reads them via a per-head convex average, blocking channel-wise selection. We propose the Free Energy Mixer (FEM): a free-energy (log-sum-exp) read that applies a value-driven, per-channel log-linear tilt to a fast prior (e.g., from queries/keys in standard attention) over indices. Unlike methods that attempt to improve and enrich the $(q,k)$ scoring distribution, FEM treats it as a prior and yields a value-aware posterior read at unchanged complexity, smoothly moving from averaging to per-channel selection as the learnable inverse temperature increases, while still preserving parallelism and the original asymptotic complexity ($O(T^2)$ for softmax; $O(T)$ for linearizable variants). We instantiate a two-level gated FEM that is plug-and-play with standard and linear attention, linear RNNs and SSMs. It consistently outperforms strong baselines on NLP, vision, and time-series at matched parameter budgets. 

---
# Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model 

**Authors**: Jacqueline He, Jonathan Hayase, Wen-tau Yih, Sewoong Oh, Luke Zettlemoyer, Pang Wei Koh  

**Link**: [PDF](https://arxiv.org/pdf/2602.07120)  

**Abstract**: Modern language models (LMs) tend to memorize portions of their training data and emit verbatim spans. When the underlying sources are sensitive or copyright-protected, such reproduction raises issues of consent and compensation for creators and compliance risks for developers. We propose Anchored Decoding, a plug-and-play inference-time method for suppressing verbatim copying: it enables decoding from any risky LM trained on mixed-license data by keeping generation in bounded proximity to a permissively trained safe LM. Anchored Decoding adaptively allocates a user-chosen information budget over the generation trajectory and enforces per-step constraints that yield a sequence-level guarantee, enabling a tunable risk-utility trade-off. To make Anchored Decoding practically useful, we introduce a new permissively trained safe model (TinyComma 1.8B), as well as Anchored$_{\mathrm{Byte}}$ Decoding, a byte-level variant of our method that enables cross-vocabulary fusion via the ByteSampler framework (Hayase et al., 2025). We evaluate our methods across six model pairs on long-form evaluations of copyright risk and utility. Anchored and Anchored$_{\mathrm{Byte}}$ Decoding define a new Pareto frontier, preserving near-original fluency and factuality while eliminating up to 75% of the measurable copying gap (averaged over six copying metrics) between the risky baseline and a safe reference, at a modest inference overhead. 

---
# Bridging the Knowledge Void: Inference-time Acquisition of Unfamiliar Programming Languages for Coding Tasks 

**Authors**: Chen Shen, Wei Cheng, Jingyue Yang, Huan Zhang, Yuhan Wu, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06976)  

**Abstract**: The proficiency of Large Language Models (LLMs) in coding tasks is often a reflection of their extensive pre-training corpora, which typically collapses when confronted with previously unfamiliar programming languages. Departing from data-intensive finetuning, we investigate the paradigm of Inference-time Language Acquisition (ILA), where an LLM masters an unfamiliar language through dynamic interaction with limited external resources. In this paper, we propose ILA-agent, a general ILA framework that equips LLMs with a set of behavioral primitives. By modeling essential human-like behaviors as a suite of tools, ILA-agent enables LLMs to incrementally explore, apply, and verify language knowledge through structured interactions with the official documentation and execution environment. To provide a rigorous evaluation in a low-resource setting, we construct Cangjie-bench, a multi-task benchmark based on the novel statically-typed language Cangjie. We instantiate ILA-agent for Cangjie and evaluate its performance across code generation, translation, and program repair tasks. Results using diverse LLMs demonstrate that ILA-agent significantly outperforms retrieval-augmented baselines. Further analysis of agent trajectories characterizes the emergent behavior patterns while highlighting persisting performance gaps. 

---
# BiomechAgent: AI-Assisted Biomechanical Analysis Through Code-Generating Agents 

**Authors**: R. James Cotton, Thomas Leonard  

**Link**: [PDF](https://arxiv.org/pdf/2602.06975)  

**Abstract**: Markerless motion capture is making quantitative movement analysis increasingly accessible, yet analyzing the resulting data remains a barrier for clinicians without programming expertise. We present BiomechAgent, a code-generating AI agent that enables biomechanical analysis through natural language and allows users to querying databases, generating visualizations, and even interpret data without requiring users to write code. To evaluate BiomechAgent's capabilities, we developed a systematic benchmark spanning data retrieval, visualization, activity classification, temporal segmentation, and clinical reasoning. BiomechAgent achieved robust accuracy on data retrieval and visualization tasks and demonstrated emerging clinical reasoning capabilities. We used our dataset to systematically evaluate several of our design decisions. Biomechanically-informed, domain-specific instructions significantly improved performance over generic prompts, and integrating validated specialized tools for gait event detection substantially boosted accuracy on challenging spatiotemporal analysis where the base agent struggled. We also tested BiomechAgent using a local open-weight model instead of a frontier cloud based LLM and found that perform was substantially diminished in most domains other than database retrieval. In short, BiomechAgent makes the data from accessible motion capture and much more useful and accessible to end users. 

---
# Does Visual Rendering Bypass Tokenization? Investigating Script-Tokenizer Misalignment in Pixel-Based Language Models 

**Authors**: Lucky Susanto, Musa Izzanardi Wijanarko, Khumaisa Nur'aini, Farid Adilazuarda, Alham Fikri Aji, Derry Tanti Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2602.06973)  

**Abstract**: While pixel-based language modeling aims to bypass the sub-word tokenization bottleneck by rendering text as images, recent multimodal variants such as DualGPT reintroduce text tokenizers to improve autoregressive performance. We investigate a fundamental question, does visual rendering truly decouple a model from tokenization constraints? Focusing on four Indonesian low-resource local languages that have their own non-Latin scripts (i.e., Javanese, Balinese, Sundanese, and Lampungnese), we evaluate the impact of script-tokenizer alignment within the DualGPT architecture. Our results show that, despite visual rendering, reintegrating a text tokenizer into the architecture reintroduces the same issue that pixel-based language modeling aims to resolve, which is the tokenizer misalignment problem. Despite having lower OOV and fertility rates, we show that the Llama 2 tokenizer performs significantly worse than a custom tokenizer, with improvements of up to 30.15 chrF++. Our findings serve as a warning for future multimodal variants, as text tokenizers remain a significant barrier to equitable models. 

---
# Next-Gen CAPTCHAs: Leveraging the Cognitive Gap for Scalable and Diverse GUI-Agent Defense 

**Authors**: Jiacheng Liu, Yaxin Luo, Jiacheng Cui, Xinyi Shang, Xiaohan Zhao, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.09012)  

**Abstract**: The rapid evolution of GUI-enabled agents has rendered traditional CAPTCHAs obsolete. While previous benchmarks like OpenCaptchaWorld established a baseline for evaluating multimodal agents, recent advancements in reasoning-heavy models, such as Gemini3-Pro-High and GPT-5.2-Xhigh have effectively collapsed this security barrier, achieving pass rates as high as 90% on complex logic puzzles like "Bingo". In response, we introduce Next-Gen CAPTCHAs, a scalable defense framework designed to secure the next-generation web against the advanced agents. Unlike static datasets, our benchmark is built upon a robust data generation pipeline, allowing for large-scale and easily scalable evaluations, notably, for backend-supported types, our system is capable of generating effectively unbounded CAPTCHA instances. We exploit the persistent human-agent "Cognitive Gap" in interactive perception, memory, decision-making, and action. By engineering dynamic tasks that require adaptive intuition rather than granular planning, we re-establish a robust distinction between biological users and artificial agents, offering a scalable and diverse defense mechanism for the agentic era. 

---
# Data Science and Technology Towards AGI Part I: Tiered Data Management 

**Authors**: Yudong Wang, Zixuan Fu, Hengyu Zhao, Chen Zhao, Chuyue Zhou, Xinle Lin, Hongya Lyu, Shuaikang Xue, Yi Yi, Yingjiao Wang, Zhi Zheng, Yuzhou Zhang, Jie Zhou, Chaojun Xiao, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.09003)  

**Abstract**: The development of artificial intelligence can be viewed as an evolution of data-driven learning paradigms, with successive shifts in data organization and utilization continuously driving advances in model capability. Current LLM research is dominated by a paradigm that relies heavily on unidirectional scaling of data size, increasingly encountering bottlenecks in data availability, acquisition cost, and training efficiency. In this work, we argue that the development of AGI is entering a new phase of data-model co-evolution, in which models actively guide data management while high-quality data, in turn, amplifies model capabilities. To implement this vision, we propose a tiered data management framework, designed to support the full LLM training lifecycle across heterogeneous learning objectives and cost constraints. Specifically, we introduce an L0-L4 tiered data management framework, ranging from raw uncurated resources to organized and verifiable knowledge. Importantly, LLMs are fully used in data management processes, such as quality scoring and content editing, to refine data across tiers. Each tier is characterized by distinct data properties, management strategies, and training roles, enabling data to be strategically allocated across LLM training stages, including pre-training, mid-training, and alignment. The framework balances data quality, acquisition cost, and marginal training benefit, providing a systematic approach to scalable and sustainable data management. We validate the effectiveness of the proposed framework through empirical studies, in which tiered datasets are constructed from raw corpora and used across multiple training phases. Experimental results demonstrate that tier-aware data utilization significantly improves training efficiency and model performance. To facilitate further research, we release our tiered datasets and processing tools to the community. 

---
# Paradox of De-identification: A Critique of HIPAA Safe Harbour in the Age of LLMs 

**Authors**: Lavender Y. Jiang, Xujin Chris Liu, Kyunghyun Cho, Eric K. Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2602.08997)  

**Abstract**: Privacy is a human right that sustains patient-provider trust. Clinical notes capture a patient's private vulnerability and individuality, which are used for care coordination and research. Under HIPAA Safe Harbor, these notes are de-identified to protect patient privacy. However, Safe Harbor was designed for an era of categorical tabular data, focusing on the removal of explicit identifiers while ignoring the latent information found in correlations between identity and quasi-identifiers, which can be captured by modern LLMs. We first formalize these correlations using a causal graph, then validate it empirically through individual re-identification of patients from scrubbed notes. The paradox of de-identification is further shown through a diagnosis ablation: even when all other information is removed, the model can predict the patient's neighborhood based on diagnosis alone. This position paper raises the question of how we can act as a community to uphold patient-provider trust when de-identification is inherently imperfect. We aim to raise awareness and discuss actionable recommendations. 

---
# Beyond Transcripts: A Renewed Perspective on Audio Chaptering 

**Authors**: Fabian Retkowski, Maike Züfle, Thai Binh Nguyen, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2602.08979)  

**Abstract**: Audio chaptering, the task of automatically segmenting long-form audio into coherent sections, is increasingly important for navigating podcasts, lectures, and videos. Despite its relevance, research remains limited and text-based, leaving key questions unresolved about leveraging audio information, handling ASR errors, and transcript-free evaluation. We address these gaps through three contributions: (1) a systematic comparison between text-based models with acoustic features, a novel audio-only architecture (AudioSeg) operating on learned audio representations, and multimodal LLMs; (2) empirical analysis of factors affecting performance, including transcript quality, acoustic features, duration, and speaker composition; and (3) formalized evaluation protocols contrasting transcript-dependent text-space protocols with transcript-invariant time-space protocols. Our experiments on YTSeg reveal that AudioSeg substantially outperforms text-based approaches, pauses provide the largest acoustic gains, and MLLMs remain limited by context length and weak instruction following, yet MLLMs are promising on shorter audio. 

---
# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents 

**Authors**: Raghu Arghal, Fade Chen, Niall Dalton, Evgenii Kortukov, Calum McNamara, Angelos Nalmpantis, Moksh Nirvaan, Gabriele Sarti, Mario Giulianelli  

**Link**: [PDF](https://arxiv.org/pdf/2602.08964)  

**Abstract**: Understanding an agent's goals helps explain and predict its behaviour, yet there is no established methodology for reliably attributing goals to agentic systems. We propose a framework for evaluating goal-directedness that integrates behavioural evaluation with interpretability-based analyses of models' internal representations. As a case study, we examine an LLM agent navigating a 2D grid world toward a goal state. Behaviourally, we evaluate the agent against an optimal policy across varying grid sizes, obstacle densities, and goal structures, finding that performance scales with task difficulty while remaining robust to difficulty-preserving transformations and complex goal structures. We then use probing methods to decode the agent's internal representations of the environment state and its multi-step action plans. We find that the LLM agent non-linearly encodes a coarse spatial map of the environment, preserving approximate task-relevant cues about its position and the goal location; that its actions are broadly consistent with these internal representations; and that reasoning reorganises them, shifting from broader environment structural cues toward information supporting immediate action selection. Our findings support the view that introspective examination is required beyond behavioural evaluations to characterise how agents represent and pursue their objectives. 

---
# CoRefine: Confidence-Guided Self-Refinement for Adaptive Test-Time Compute 

**Authors**: Chen Jin, Ryutaro Tanno, Tom Diethe, Philip Teare  

**Link**: [PDF](https://arxiv.org/pdf/2602.08948)  

**Abstract**: Large Language Models (LLMs) often rely on test-time scaling via parallel decoding (for example, 512 samples) to boost reasoning accuracy, but this incurs substantial compute. We introduce CoRefine, a confidence-guided self-refinement method that achieves competitive accuracy using a fraction of the tokens via a lightweight 211k-parameter Conv1D controller atop a frozen LLM. The controller consumes full-trace confidence to decide whether to halt, re-examine, or try a different approach, enabling targeted self-correction with an average of 2.7 refinement steps per problem and roughly 190-fold token reduction relative to 512-sample baselines. Across diverse reasoning benchmarks and three open-source models, the controller achieves 92.6 percent precision when it confidently halts, indicating that confidence dynamics reliably signal correctness without ground-truth verification. We extend this to CoRefine-Tree, a hybrid sequential-parallel variant that adaptively balances exploration and exploitation, with easy serving integration and verifier compatibility. By treating confidence as a control signal rather than a correctness guarantee, CoRefine provides a modular primitive for scalable reasoning and agentic settings with imperfect verifiers. 

---
# Discovering Interpretable Algorithms by Decompiling Transformers to RASP 

**Authors**: Xinting Huang, Aleksandra Bakalova, Satwik Bhattamishra, William Merrill, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2602.08857)  

**Abstract**: Recent work has shown that the computations of Transformers can be simulated in the RASP family of programming languages. These findings have enabled improved understanding of the expressive capacity and generalization abilities of Transformers. In particular, Transformers have been suggested to length-generalize exactly on problems that have simple RASP programs. However, it remains open whether trained models actually implement simple interpretable programs. In this paper, we present a general method to extract such programs from trained Transformers. The idea is to faithfully re-parameterize a Transformer as a RASP program and then apply causal interventions to discover a small sufficient sub-program. In experiments on small Transformers trained on algorithmic and formal language tasks, we show that our method often recovers simple and interpretable RASP programs from length-generalizing transformers. Our results provide the most direct evidence so far that Transformers internally implement simple RASP programs. 

---
# Bayesian Preference Learning for Test-Time Steerable Reward Models 

**Authors**: Jiwoo Hong, Shao Tang, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08819)  

**Abstract**: Reward models are central to aligning language models with human preferences via reinforcement learning (RL). As RL is increasingly applied to settings such as verifiable rewards and multi-objective alignment, RMs are expected to encode more complex and multifaceted preference distributions. However, classifier RMs remain static once trained, limiting their adaptability at test time. We propose Variational In-Context Reward Modeling (ICRM), a novel Bayesian reward modeling objective that enables test-time steerability via in-context preference demonstrations. ICRM casts reward modeling as amortized variational inference over a latent preference probability under the Bradley-Terry model using a conjugate Beta prior. We show that ICRM adapt to unseen preference distributions at test time for both single and multi-objective settings. With more in-context demonstrations, ICRM gains 34% accuracy on SafeRLHF and 9% accuracy on RM-Bench in the single-objective setting, while widening the Pareto frontier with a 4% gain in hypervolume on helpfulness and refusal benchmarks. We further study the practical applicability of ICRM for RL training, showing that it can effectively encode verifiable rewards by outperforming a conventional RM in math reasoning. Finally, we provide theoretical guarantees that the variational objective admits a global interior optimum with finite confidence, and we analyze how KL regularization mitigates reward over-optimization. 

---
# The Use of AI Tools to Develop and Validate Q-Matrices 

**Authors**: Kevin Fan, Jacquelyn A. Bialo, Hongli Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08796)  

**Abstract**: Constructing a Q-matrix is a critical but labor-intensive step in cognitive diagnostic modeling (CDM). This study investigates whether AI tools (i.e., general language models) can support Q-matrix development by comparing AI-generated Q-matrices with a validated Q-matrix from Li and Suen (2013) for a reading comprehension test. In May 2025, multiple AI models were provided with the same training materials as human experts. Agreement among AI-generated Q-matrices, the validated Q-matrix, and human raters' Q-matrices was assessed using Cohen's kappa. Results showed substantial variation across AI models, with Google Gemini 2.5 Pro achieving the highest agreement (Kappa = 0.63) with the validated Q-matrix, exceeding that of all human experts. A follow-up analysis in January 2026 using newer AI versions, however, revealed lower agreement with the validated Q-matrix. Implications and directions for future research are discussed. 

---
# Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure 

**Authors**: Zirui Li, Xuefeng Bai, Kehai Chen, Yizhi Li, Jian Yang, Chenghua Lin, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08783)  

**Abstract**: Latent or continuous chain-of-thought methods replace explicit textual rationales with a number of internal latent steps, but these intermediate computations are difficult to evaluate beyond correlation-based probes. In this paper, we view latent chain-of-thought as a manipulable causal process in representation space by modeling latent steps as variables in a structural causal model (SCM) and analyzing their effects through step-wise $\mathrm{do}$-interventions. We study two representative paradigms (i.e., Coconut and CODI) on both mathematical and general reasoning tasks to investigate three key questions: (1) which steps are causally necessary for correctness and when answers become decidable early; (2) how does influence propagate across steps, and how does this structure compare to explicit CoT; and (3) do intermediate trajectories retain competing answer modes, and how does output-level commitment differ from representational commitment across steps. We find that latent-step budgets behave less like homogeneous extra depth and more like staged functionality with non-local routing, and we identify a persistent gap between early output bias and late representational commitment. These results motivate mode-conditional and stability-aware analyses -- and corresponding training/decoding objectives -- as more reliable tools for interpreting and improving latent reasoning systems. 

---
# Prototype-Based Disentanglement for Controllable Dysarthric Speech Synthesis 

**Authors**: Haoshen Wang, Xueli Zhong, Bingbing Lin, Jia Huang, Xingduo Pan, Shengxiang Liang, Nizhuan Wang, Wai Ting Siok  

**Link**: [PDF](https://arxiv.org/pdf/2602.08696)  

**Abstract**: Dysarthric speech exhibits high variability and limited labeled data, posing major challenges for both automatic speech recognition (ASR) and assistive speech technologies. Existing approaches rely on synthetic data augmentation or speech reconstruction, yet often entangle speaker identity with pathological articulation, limiting controllability and robustness.
In this paper, we propose ProtoDisent-TTS, a prototype-based disentanglement TTS framework built on a pre-trained text-to-speech backbone that factorizes speaker timbre and dysarthric articulation within a unified latent space. A pathology prototype codebook provides interpretable and controllable representations of healthy and dysarthric speech patterns, while a dual-classifier objective with a gradient reversal layer enforces invariance of speaker embeddings to pathological attributes. Experiments on the TORGO dataset demonstrate that this design enables bidirectional transformation between healthy and dysarthric speech, leading to consistent ASR performance gains and robust, speaker-aware speech reconstruction. 

---
# We Should Separate Memorization from Copyright 

**Authors**: Adi Haviv, Niva Elkin-Koren, Uri Hacohen, Roi Livni, Shay Moran  

**Link**: [PDF](https://arxiv.org/pdf/2602.08632)  

**Abstract**: The widespread use of foundation models has introduced a new risk factor of copyright issue. This issue is leading to an active, lively and on-going debate amongst the data-science community as well as amongst legal scholars. Where claims and results across both sides are often interpreted in different ways and leading to different implications. Our position is that much of the technical literature relies on traditional reconstruction techniques that are not designed for copyright analysis. As a result, memorization and copying have been conflated across both technical and legal communities and in multiple contexts. We argue that memorization, as commonly studied in data science, should not be equated with copying and should not be used as a proxy for copyright infringement. We distinguish technical signals that meaningfully indicate infringement risk from those that instead reflect lawful generalization or high-frequency content. Based on this analysis, we advocate for an output-level, risk-based evaluation process that aligns technical assessments with established copyright standards and provides a more principled foundation for research, auditing, and policy. 

---
# ValueFlow: Measuring the Propagation of Value Perturbations in Multi-Agent LLM Systems 

**Authors**: Jinnuo Liu, Chuke Liu, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08567)  

**Abstract**: Multi-agent large language model (LLM) systems increasingly consist of agents that observe and respond to one another's outputs. While value alignment is typically evaluated for isolated models, how value perturbations propagate through agent interactions remains poorly understood. We present ValueFlow, a perturbation-based evaluation framework for measuring and analyzing value drift in multi-agent systems. ValueFlow introduces a 56-value evaluation dataset derived from the Schwartz Value Survey and quantifies agents' value orientations during interaction using an LLM-as-a-judge protocol. Building on this measurement layer, ValueFlow decomposes value drift into agent-level response behavior and system-level structural effects, operationalized by two metrics: beta-susceptibility, which measures an agent's sensitivity to perturbed peer signals, and system susceptibility (SS), which captures how node-level perturbations affect final system outputs. Experiments across multiple model backbones, prompt personas, value dimensions, and network structures show that susceptibility varies widely across values and is strongly shaped by structural topology. 

---
# Automating Computational Reproducibility in Social Science: Comparing Prompt-Based and Agent-Based Approaches 

**Authors**: Syed Mehtab Hussain Shah, Frank Hopfgartner, Arnim Bleier  

**Link**: [PDF](https://arxiv.org/pdf/2602.08561)  

**Abstract**: Reproducing computational research is often assumed to be as simple as rerunning the original code with provided data. In practice, missing packages, fragile file paths, version conflicts, or incomplete logic frequently cause analyses to fail, even when materials are shared. This study investigates whether large language models and AI agents can automate the diagnosis and repair of such failures, making computational results easier to reproduce and verify. We evaluate this using a controlled reproducibility testbed built from five fully reproducible R-based social science studies. Realistic failures were injected, ranging from simple issues to complex missing logic, and two automated repair workflows were tested in clean Docker environments. The first workflow is prompt-based, repeatedly querying language models with structured prompts of varying context, while the second uses agent-based systems that inspect files, modify code, and rerun analyses autonomously. Across prompt-based runs, reproduction success ranged from 31-79 percent, with performance strongly influenced by prompt context and error complexity. Complex cases benefited most from additional context. Agent-based workflows performed substantially better, with success rates of 69-96 percent across all complexity levels. These results suggest that automated workflows, especially agent-based systems, can significantly reduce manual effort and improve reproduction success across diverse error types. Unlike prior benchmarks, our testbed isolates post-publication repair under controlled failure modes, allowing direct comparison of prompt-based and agent-based approaches. 

---
# Learning Self-Correction in Vision-Language Models via Rollout Augmentation 

**Authors**: Yi Ding, Ziliang Qiu, Bolian Li, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08503)  

**Abstract**: Self-correction is essential for solving complex reasoning problems in vision-language models (VLMs). However, existing reinforcement learning (RL) methods struggle to learn it, as effective self-correction behaviors emerge only rarely, making learning signals extremely sparse. To address this challenge, we propose correction-specific rollouts (Octopus), an RL rollout augmentation framework that synthesizes dense self-correction examples by recombining existing rollouts. This augmentation simultaneously improves sample efficiency due to rollout reuse and stabilizes RL optimization through balanced supervision. Furthermore, we introduce a response-masking strategy that decouples self-correction from direct reasoning, avoiding signal conflicts and enabling both behaviors to be learned effectively. Building on this, we introduce Octopus-8B, a reasoning VLM with controllable self-correction capability. Across 7 benchmarks, it achieves SoTA performance among open-source VLMs, outperforming the best RLVR baseline by 1.0 score while requiring only $0.72\times$ training time per step. 

---
# Beyond Correctness: Learning Robust Reasoning via Transfer 

**Authors**: Hyunseok Lee, Soheil Abbasloo, Jihoon Tack, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08489)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently strengthened LLM reasoning, but its focus on final answer correctness leaves a critical gap: it does not ensure the robustness of the reasoning process itself. We adopt a simple philosophical view, robust reasoning should remain useful beyond the mind that produced it, and treat reasoning as a form of meaning transfer that must survive truncation, reinterpretation, and continuation. Building on this principle, we introduce Reinforcement Learning with Transferable Reward (RLTR), which operationalizes robustness via transfer reward that tests whether a partial reasoning prefix from one model can guide a separate model to the correct answer. This encourages LLMs to produce reasoning that is stable, interpretable, and genuinely generalizable. Our approach improves sampling consistency while improving final answer accuracy, and it reaches comparable performance in substantially fewer training steps. For example, on MATH500, RLTR achieves a +3.6%p gain in Maj@64 compared to RLVR and matches RLVR's average accuracy with roughly 2.5x fewer training steps, providing both more reliable reasoning and significantly more sample efficient. 

---
# Reinforcement Learning with Backtracking Feedback 

**Authors**: Bilgehan Sel, Vaishakh Keshava, Phillip Wallis, Lukas Rutishauser, Ming Jin, Dingcheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08377)  

**Abstract**: Addressing the critical need for robust safety in Large Language Models (LLMs), particularly against adversarial attacks and in-distribution errors, we introduce Reinforcement Learning with Backtracking Feedback (RLBF). This framework advances upon prior methods, such as BSAFE, by primarily leveraging a Reinforcement Learning (RL) stage where models learn to dynamically correct their own generation errors. Through RL with critic feedback on the model's live outputs, LLMs are trained to identify and recover from their actual, emergent safety violations by emitting an efficient "backtrack by x tokens" signal, then continuing generation autoregressively. This RL process is crucial for instilling resilience against sophisticated adversarial strategies, including middle filling, Greedy Coordinate Gradient (GCG) attacks, and decoding parameter manipulations. To further support the acquisition of this backtracking capability, we also propose an enhanced Supervised Fine-Tuning (SFT) data generation strategy (BSAFE+). This method improves upon previous data creation techniques by injecting violations into coherent, originally safe text, providing more effective initial training for the backtracking mechanism. Comprehensive empirical evaluations demonstrate that RLBF significantly reduces attack success rates across diverse benchmarks and model scales, achieving superior safety outcomes while critically preserving foundational model utility. 

---
# MemAdapter: Fast Alignment across Agent Memory Paradigms via Generative Subgraph Retrieval 

**Authors**: Xin Zhang, Kailai Yang, Chenyue Li, Hao Li, Qiyu Wei, Jun'ichi Tsujii, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2602.08369)  

**Abstract**: Memory mechanism is a core component of LLM-based agents, enabling reasoning and knowledge discovery over long-horizon contexts. Existing agent memory systems are typically designed within isolated paradigms (e.g., explicit, parametric, or latent memory) with tightly coupled retrieval methods that hinder cross-paradigm generalization and fusion. In this work, we take a first step toward unifying heterogeneous memory paradigms within a single memory system. We propose MemAdapter, a memory retrieval framework that enables fast alignment across agent memory paradigms. MemAdapter adopts a two-stage training strategy: (1) training a generative subgraph retriever from the unified memory space, and (2) adapting the retriever to unseen memory paradigms by training a lightweight alignment module through contrastive learning. This design improves the flexibility for memory retrieval and substantially reduces alignment cost across paradigms. Comprehensive experiments on three public evaluation benchmarks demonstrate that the generative subgraph retriever consistently outperforms five strong agent memory systems across three memory paradigms and agent model scales. Notably, MemAdapter completes cross-paradigm alignment within 13 minutes on a single GPU, achieving superior performance over original memory retrievers with less than 5% of training compute. Furthermore, MemAdapter enables effective zero-shot fusion across memory paradigms, highlighting its potential as a plug-and-play solution for agent memory systems. 

---
# ManifoldKV: Training-Free KV Cache Compression via Euclidean Outlier Detection 

**Authors**: Debajyoti Datta, Trishala Neeraj, Bibek Paudel, Vyom Sharma, Subhabrata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.08343)  

**Abstract**: Long-context inference is constrained by KV-cache memory, which grows linearly with sequence length; KV-cache compression therefore hinges on reliably selecting which past tokens to retain. Most geometry-based eviction methods score keys by cosine similarity to a global centroid, but cosine is scale-invariant and can discard magnitude cues that distinguish semantically salient tokens. We propose ManifoldKV, a training-free scorer that ranks tokens by Euclidean distance to the key centroid, capturing both angular and radial deviations.
On the RULER benchmark, ManifoldKV achieves 95.7% accuracy at 4K-16K contexts with 20% compression; matching the best geometric baseline while improving robustness in two regimes where cosine scoring fails. First, on multi-key retrieval, ManifoldKV reduces directional collisions, achieving 92.4% vs KeyDiff's 77.0% (+15.4 points) on 3-key NIAH at 50% compression. Second, to address dilution and performance collapse of global centroids at 64K context, we introduce WindowedManifoldKV, which restores accuracy to 84.3% at 25% compression, a 49-point recovery over global L2 and +3.2 points over KeyDiff. The method requires only 3 lines of code and works across 4 architectures without tuning. 

---
# Linguistics and Human Brain: A Perspective of Computational Neuroscience 

**Authors**: Fudong Zhang, Bo Chai, Yujie Wu, Wai Ting Siok, Nizhuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08275)  

**Abstract**: Elucidating the language-brain relationship requires bridging the methodological gap between the abstract theoretical frameworks of linguistics and the empirical neural data of neuroscience. Serving as an interdisciplinary cornerstone, computational neuroscience formalizes the hierarchical and dynamic structures of language into testable neural models through modeling, simulation, and data analysis. This enables a computational dialogue between linguistic hypotheses and neural mechanisms. Recent advances in deep learning, particularly large language models (LLMs), have powerfully advanced this pursuit. Their high-dimensional representational spaces provide a novel scale for exploring the neural basis of linguistic processing, while the "model-brain alignment" framework offers a methodology to evaluate the biological plausibility of language-related theories. 

---
# When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning 

**Authors**: Shoubin Yu, Yue Zhang, Zun Wang, Jaehong Yoon, Huaxiu Yao, Mingyu Ding, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2602.08236)  

**Abstract**: Despite rapid progress in Multimodal Large Language Models (MLLMs), visual spatial reasoning remains unreliable when correct answers depend on how a scene would appear under unseen or alternative viewpoints. Recent work addresses this by augmenting reasoning with world models for visual imagination, but questions such as when imagination is actually necessary, how much of it is beneficial, and when it becomes harmful, remain poorly understood. In practice, indiscriminate imagination can increase computation and even degrade performance by introducing misleading evidence. In this work, we present an in-depth analysis of test-time visual imagination as a controllable resource for spatial reasoning. We study when static visual evidence is sufficient, when imagination improves reasoning, and how excessive or unnecessary imagination affects accuracy and efficiency. To support this analysis, we introduce AVIC, an adaptive test-time framework with world models that explicitly reasons about the sufficiency of current visual evidence before selectively invoking and scaling visual imagination. Across spatial reasoning benchmarks (SAT, MMSI) and an embodied navigation benchmark (R2R), our results reveal clear scenarios where imagination is critical, marginal, or detrimental, and show that selective control can match or outperform fixed imagination strategies with substantially fewer world-model calls and language tokens. Overall, our findings highlight the importance of analyzing and controlling test-time imagination for efficient and reliable spatial reasoning. 

---
# DrugR: Optimizing Molecular Drugs through LLM-based Explicit Reasoning 

**Authors**: Haoran Liu, Zheni Zeng, Yukun Yan, Yuxuan Chen, Yunduo Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.08213)  

**Abstract**: Molecule generation and optimization is a fundamental task in chemical domain. The rapid development of intelligent tools, especially large language models (LLMs) with powerful knowledge reserves and interactive capabilities, has provided new paradigms for it. Nevertheless, the intrinsic challenge for LLMs lies in the complex implicit relationship between molecular structure and pharmacological properties and the lack of corresponding labeled data. To bridge this gap, we propose DrugR, an LLM-based method that introduces explicit, step-by-step pharmacological reasoning into the optimization process. Our approach integrates domain-specific continual pretraining, supervised fine-tuning via reverse data engineering, and self-balanced multi-granular reinforcement learning. This framework enables DrugR to effectively improve key ADMET properties while preserving the original molecule's core efficacy. Experimental results demonstrate that DrugR achieves comprehensive enhancement across multiple properties without compromising structural similarity or target binding affinity. Importantly, its explicit reasoning process provides clear, interpretable rationales for each optimization step, yielding actionable design insights and advancing toward automated, knowledge-driven scientific discovery. Our code and model checkpoints are open-sourced to foster future research. 

---
# Dreaming in Code for Curriculum Learning in Open-Ended Worlds 

**Authors**: Konstantinos Mitsides, Maxence Faldor, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2602.08194)  

**Abstract**: Open-ended learning frames intelligence as emerging from continual interaction with an ever-expanding space of environments. While recent advances have utilized foundation models to programmatically generate diverse environments, these approaches often focus on discovering isolated behaviors rather than orchestrating sustained progression. In complex open-ended worlds, the large combinatorial space of possible challenges makes it difficult for agents to discover sequences of experiences that remain consistently learnable. To address this, we propose Dreaming in Code (DiCode), a framework in which foundation models synthesize executable environment code to scaffold learning toward increasing competence. In DiCode, "dreaming" takes the form of materializing code-level variations of the world. We instantiate DiCode in Craftax, a challenging open-ended benchmark characterized by rich mechanics and long-horizon progression. Empirically, DiCode enables agents to acquire long-horizon skills, achieving a $16\%$ improvement in mean return over the strongest baseline and non-zero success on late-game combat tasks where prior methods fail. Our results suggest that code-level environment design provides a practical mechanism for curriculum control, enabling the construction of intermediate environments that bridge competence gaps in open-ended worlds. Project page and source code are available at this https URL and this https URL. 

---
# Spherical Steering: Geometry-Aware Activation Rotation for Language Models 

**Authors**: Zejia You, Chunyuan Deng, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08169)  

**Abstract**: Inference-time steering has emerged as a promising paradigm for controlling language models (LMs) without the cost of retraining. However, standard approaches typically rely on activation addition, a geometric operation that inevitably alters the magnitude of hidden representations. This raises concerns about representation collapse and degradation of open-ended generation capabilities. In this work, we explore Spherical Steering, a training-free primitive that resolves this trade-off through activation rotation. Rather than shifting activations with a fixed vector, our method rotates them along a geodesic toward a target direction, guiding the activation toward the target concept while preserving the integrity of the signal. To further enhance adaptivity, we incorporate a confidence gate that dynamically modulates steering strength based on input uncertainty. Extensive experiments across multiple-choice benchmarks demonstrate that Spherical Steering significantly outperforms addition-based baselines (notably by +10% on TruthfulQA, COPA, and Storycloze), while simultaneously maintaining the model's general open-ended generation quality. This work highlights the value of geometric consistency, suggesting that norm-preserving rotation is a robust and effective primitive for precise inference-time control. 

---
# The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models 

**Authors**: Seonglae Cho, Zekun Wu, Kleyton Da Costa, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2602.08159)  

**Abstract**: When a language model asserts that "the capital of Australia is Sydney," does it know this is wrong? We characterize the geometry of correctness representations across 9 models from 5 architecture families. The structure is simple: the discriminative signal occupies 3-8 dimensions, performance degrades with additional dimensions, and no nonlinear classifier improves over linear separation. Centroid distance in the low-dimensional subspace matches trained probe performance (0.90 AUC), enabling few-shot detection: on GPT-2, 25 labeled examples achieve 89% of full-data accuracy. We validate causally through activation steering: the learned direction produces 10.9 percentage point changes in error rates while random directions show no effect. Internal probes achieve 0.80-0.97 AUC; output-based methods (P(True), semantic entropy) achieve only 0.44-0.64 AUC. The correctness signal exists internally but is not expressed in outputs. That centroid distance matches probe performance indicates class separation is a mean shift, making detection geometric rather than learned. 

---
# Reliable and Responsible Foundation Models: A Comprehensive Survey 

**Authors**: Xinyu Yang, Junlin Han, Rishi Bommasani, Jinqi Luo, Wenjie Qu, Wangchunshu Zhou, Adel Bibi, Xiyao Wang, Jaehong Yoon, Elias Stengel-Eskin, Shengbang Tong, Lingfeng Shen, Rafael Rafailov, Runjia Li, Zhaoyang Wang, Yiyang Zhou, Chenhang Cui, Yu Wang, Wenhao Zheng, Huichi Zhou, Jindong Gu, Zhaorun Chen, Peng Xia, Tony Lee, Thomas Zollo, Vikash Sehwag, Jixuan Leng, Jiuhai Chen, Yuxin Wen, Huan Zhang, Zhun Deng, Linjun Zhang, Pavel Izmailov, Pang Wei Koh, Yulia Tsvetkov, Andrew Wilson, Jiaheng Zhang, James Zou, Cihang Xie, Hao Wang, Philip Torr, Julian McAuley, David Alvarez-Melis, Florian Tramèr, Kaidi Xu, Suman Jana, Chris Callison-Burch, Rene Vidal, Filippos Kokkinos, Mohit Bansal, Beidi Chen, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2602.08145)  

**Abstract**: Foundation models, including Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), Image Generative Models (i.e, Text-to-Image Models and Image-Editing Models), and Video Generative Models, have become essential tools with broad applications across various domains such as law, medicine, education, finance, science, and beyond. As these models see increasing real-world deployment, ensuring their reliability and responsibility has become critical for academia, industry, and government. This survey addresses the reliable and responsible development of foundation models. We explore critical issues, including bias and fairness, security and privacy, uncertainty, explainability, and distribution shift. Our research also covers model limitations, such as hallucinations, as well as methods like alignment and Artificial Intelligence-Generated Content (AIGC) detection. For each area, we review the current state of the field and outline concrete future research directions. Additionally, we discuss the intersections between these areas, highlighting their connections and shared challenges. We hope our survey fosters the development of foundation models that are not only powerful but also ethical, trustworthy, reliable, and socially responsible. 

---
# Online Bayesian Imbalanced Learning with Bregman-Calibrated Deep Networks 

**Authors**: Zahir Alsulaimawi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08128)  

**Abstract**: Class imbalance remains a fundamental challenge in machine learning, where standard classifiers exhibit severe performance degradation in minority classes. Although existing approaches address imbalance through resampling or cost-sensitive learning during training, they require retraining or access to labeled target data when class distributions shift at deployment time, a common occurrence in real-world applications such as fraud detection, medical diagnosis, and anomaly detection. We present \textit{Online Bayesian Imbalanced Learning} (OBIL), a principled framework that decouples likelihood-ratio estimation from class-prior assumptions, enabling real-time adaptation to distribution shifts without model retraining. Our approach builds on the established connection between Bregman divergences and proper scoring rules to show that deep networks trained with such losses produce posterior probability estimates from which prior-invariant likelihood ratios can be extracted. We prove that these likelihood-ratio estimates remain valid under arbitrary changes in class priors and cost structures, requiring only a threshold adjustment for optimal Bayes decisions. We derive finite-sample regret bounds demonstrating that OBIL achieves $O(\sqrt{T \log T})$ regret against an oracle with perfect prior knowledge. Extensive experiments on benchmark datasets and medical diagnosis benchmarks under simulated deployment shifts demonstrate that OBIL maintains robust performance under severe distribution shifts, outperforming state-of-the-art methods in F1 Score when test distributions deviate significantly from the training conditions. 

---
# SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm 

**Authors**: Tianyu Li, Dongchen Han, Zixuan Cao, Haofeng Huang, Mengyu Zhou, Ming Chen, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08064)  

**Abstract**: Modern Transformers predominantly adopt the Pre-Norm paradigm for its optimization stability, foregoing the superior potential of the unstable Post-Norm architecture. Prior attempts to combine their strengths typically lead to a stability-performance trade-off. We attribute this phenomenon to a structural incompatibility within a single-stream design: Any application of the Post-Norm operation inevitably obstructs the clean identity gradient preserved by Pre-Norm. To fundamentally reconcile these paradigms, we propose SiameseNorm, a two-stream architecture that couples Pre-Norm-like and Post-Norm-like streams with shared parameters. This design decouples the optimization dynamics of the two streams, retaining the distinct characteristics of both Pre-Norm and Post-Norm by enabling all residual blocks to receive combined gradients inherited from both paradigms, where one stream secures stability while the other enhances expressivity. Extensive pre-training experiments on 1.3B-parameter models demonstrate that SiameseNorm exhibits exceptional optimization robustness and consistently outperforms strong baselines. Code is available at this https URL. 

---
# Implicit Strategic Optimization: Rethinking Long-Horizon Decision-Making in Adversarial Poker Environments 

**Authors**: Boyang Xia, Weiyou Tian, Qingnan Ren, Jiaqi Huang, Jie Xiao, Shuo Lu, Kai Wang, Lynn Ai, Eric Yang, Bill Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08041)  

**Abstract**: Training large language model (LLM) agents for adversarial games is often driven by episodic objectives such as win rate. In long-horizon settings, however, payoffs are shaped by latent strategic externalities that evolve over time, so myopic optimization and variation-based regret analyses can become vacuous even when the dynamics are predictable. To solve this problem, we introduce Implicit Strategic Optimization (ISO), a prediction-aware framework in which each agent forecasts the current strategic context and uses it to update its policy online. ISO combines a Strategic Reward Model (SRM) that estimates the long-run strategic value of actions with iso-grpo, a context-conditioned optimistic learning rule. We prove sublinear contextual regret and equilibrium convergence guarantees whose dominant terms scale with the number of context mispredictions; when prediction errors are bounded, our bounds recover the static-game rates obtained when strategic externalities are known. Experiments in 6-player No-Limit Texas Hold'em and competitive Pokemon show consistent improvements in long-term return over strong LLM and RL baselines, and graceful degradation under controlled prediction noise. 

---
# Free(): Learning to Forget in Malloc-Only Reasoning Models 

**Authors**: Yilun Zheng, Dongyang Ma, Tian Liang, Jiahao Xu, Xinting Huang, Lijie Chen, Haitao Mi, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08030)  

**Abstract**: Reasoning models enhance problem-solving by scaling test-time compute, yet they face a critical paradox: excessive thinking tokens often degrade performance rather than improve it. We attribute this to a fundamental architectural flaw: standard LLMs operate as "malloc-only" engines, continuously accumulating valid and redundant steps alike without a mechanism to prune obsolete information. To break this cycle, we propose Free()LM, a model that introduces an intrinsic self-forgetting capability via the Free-Module, a plug-and-play LoRA adapter. By iteratively switching between reasoning and cleaning modes, Free()LM dynamically identifies and prunes useless context chunks, maintaining a compact and noise-free state.
Extensive experiments show that Free()LM provides consistent improvements across all model scales (8B to 685B). It achieves a 3.3% average improvement over top-tier reasoning baselines, even establishing a new SOTA on IMOanswerBench using DeepSeek V3.2-Speciale. Most notably, in long-horizon tasks where the standard Qwen3-235B-A22B model suffers a total collapse (0% accuracy), Free()LM restores performance to 50%. Our findings suggest that sustainable intelligence requires the freedom to forget as much as the power to think. 

---
# FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging 

**Authors**: Ziyang Fan, Keyu Chen, Ruilong Xing, Yulin Li, Li Jiang, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.08024)  

**Abstract**: Although Video Large Language Models (VLLMs) have shown remarkable capabilities in video understanding, they are required to process high volumes of visual tokens, causing significant computational inefficiency. Existing VLLMs acceleration frameworks usually compress spatial and temporal redundancy independently, which overlooks the spatiotemporal relationships, thereby leading to suboptimal spatiotemporal compression. The highly correlated visual features are likely to change in spatial position, scale, orientation, and other attributes over time due to the dynamic nature of video. Building on this insight, we introduce FlashVID, a training-free inference acceleration framework for VLLMs. Specifically, FlashVID utilizes Attention and Diversity-based Token Selection (ADTS) to select the most representative tokens for basic video representation, then applies Tree-based Spatiotemporal Token Merging (TSTM) for fine-grained spatiotemporal redundancy elimination. Extensive experiments conducted on three representative VLLMs across five video understanding benchmarks demonstrate the effectiveness and generalization of our method. Notably, by retaining only 10% of visual tokens, FlashVID preserves 99.1% of the performance of LLaVA-OneVision. Consequently, FlashVID can serve as a training-free and plug-and-play module for extending long video frames, which enables a 10x increase in video frame input to Qwen2.5-VL, resulting in a relative improvement of 8.6% within the same computational budget. Code is available at this https URL. 

---
# Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking Perspective 

**Authors**: Rui Li, Zeyu Zhang, Xiaohe Bo, Quanyu Dai, Chaozhuo Li, Feng Wen, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08009)  

**Abstract**: Multi-agent architectures built on large language models (LLMs) have demonstrated the potential to realize swarm intelligence through well-crafted collaboration. However, the substantial burden of manual orchestration inherently raises an imperative to automate the design of agentic workflows. We frame such an agent coordination challenge as a classic problem in dynamic ad-hoc networking: How to establish adaptive and reliable communication among a scalable number of agentic hosts? In response to this unresolved dilemma, we introduce RAPS, a reputation-aware publish-subscribe paradigm for adaptive, scalable, and robust coordination of LLM agents. RAPS is grounded in the Distributed Publish-Subscribe Protocol, allowing LLM agents to exchange messages based on their declared intents rather than predefined topologies. Beyond this substrate, RAPS further incorporates two coherent overlays: (i) Reactive Subscription, enabling agents to dynamically refine their intents; and (ii) Bayesian Reputation, empowering each agent with a local watchdog to detect and isolate malicious peers. Extensive experiments over five benchmarks showcase that our design effectively reconciles adaptivity, scalability, and robustness in a unified multi-agent coordination framework. 

---
# Accelerating Social Science Research via Agentic Hypothesization and Experimentation 

**Authors**: Jishu Sen Gupta, Harini SI, Somesh Kumar Singh, Syed Mohamad Tawseeq, Yaman Kumar Singla, David Doermann, Rajiv Ratn Shah, Balaji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2602.07983)  

**Abstract**: Data-driven social science research is inherently slow, relying on iterative cycles of observation, hypothesis generation, and experimental validation. While recent data-driven methods promise to accelerate parts of this process, they largely fail to support end-to-end scientific discovery. To address this gap, we introduce EXPERIGEN, an agentic framework that operationalizes end-to-end discovery through a Bayesian optimization inspired two-phase search, in which a Generator proposes candidate hypotheses and an Experimenter evaluates them empirically. Across multiple domains, EXPERIGEN consistently discovers 2-4x more statistically significant hypotheses that are 7-17 percent more predictive than prior approaches, and naturally extends to complex data regimes including multimodal and relational datasets. Beyond statistical performance, hypotheses must be novel, empirically grounded, and actionable to drive real scientific progress. To evaluate these qualities, we conduct an expert review of machine-generated hypotheses, collecting feedback from senior faculty. Among 25 reviewed hypotheses, 88 percent were rated moderately or strongly novel, 70 percent were deemed impactful and worth pursuing, and most demonstrated rigor comparable to senior graduate-level research. Finally, recognizing that ultimate validation requires real-world evidence, we conduct the first A/B test of LLM-generated hypotheses, observing statistically significant results with p less than 1e-6 and a large effect size of 344 percent. 

---
# Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient Projection 

**Authors**: Guanglong Sun, Siyuan Zhang, Liyuan Wang, Jun Zhu, Hang Su, Yi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2602.07892)  

**Abstract**: Large Language Models (LLMs) often incur an alignment tax: safety post-training can reduce general utility (e.g., reasoning and coding). We argue that this tax primarily arises from continual-learning-style forgetting in sequential alignment, where distribution shift and conflicting objectives cause safety updates to overwrite pre-trained competencies. Accordingly, we cast safety alignment as a continual learning (CL) problem that must balance plasticity (acquiring safety constraints) and stability (preserving general abilities). We propose Orthogonal Gradient Projection for Safety Alignment (OGPSA), a lightweight method that mitigates interference by constraining each safety update to be orthogonal (in a first-order sense) to a learned subspace capturing general capabilities. Specifically, OGPSA estimates a low-rank capability subspace from gradients on a small reference set and projects the safety gradient onto its orthogonal complement before updating. This produces safety-directed updates that minimally perturb prior knowledge while retaining capacity for alignment. OGPSA is plug-and-play and integrates into standard post-training pipelines without large-scale replay, auxiliary objectives, or retraining. Across Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and sequential SFT$\rightarrow$DPO settings, OGPSA consistently improves the safety--utility Pareto frontier over standard baselines. For instance, on Qwen2.5-7B-Instruct under SFT$\rightarrow$DPO, OGPSA preserves strong safety while recovering general capability, improving SimpleQA from 0.53\% to 3.03\% and IFEval from 51.94\% to 63.96\%. Our source code is available at \href{this https URL}{OGPSA} 

---
# Emergent Misalignment is Easy, Narrow Misalignment is Hard 

**Authors**: Anna Soligo, Edward Turner, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2602.07852)  

**Abstract**: Finetuning large language models on narrowly harmful datasets can cause them to become emergently misaligned, giving stereotypically `evil' responses across diverse unrelated settings. Concerningly, a pre-registered survey of experts failed to predict this result, highlighting our poor understanding of the inductive biases governing learning and generalisation in LLMs. We use emergent misalignment (EM) as a case study to investigate these inductive biases and find that models can just learn the narrow dataset task, but that the general solution appears to be more stable and more efficient. To establish this, we build on the result that different EM finetunes converge to the same linear representation of general misalignment, which can be used to mediate misaligned behaviour. We find a linear representation of the narrow solution also exists, and can be learned by introducing a KL divergence loss. Comparing these representations reveals that general misalignment achieves lower loss, is more robust to perturbations, and is more influential in the pre-training distribution. This work isolates a concrete representation of general misalignment for monitoring and mitigation. More broadly, it offers a detailed case study and preliminary metrics for investigating how inductive biases shape generalisation in LLMs. We open-source all code, datasets and model finetunes. 

---
# SPD-Faith Bench: Diagnosing and Improving Faithfulness in Chain-of-Thought for Multimodal Large Language Models 

**Authors**: Weijiang Lv, Yaoxuan Feng, Xiaobo Xia, Jiayu Wang, Yan Jing, Wenchao Chen, Bo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.07833)  

**Abstract**: Chain-of-Thought reasoning is widely used to improve the interpretability of multimodal large language models (MLLMs), yet the faithfulness of the generated reasoning traces remains unclear. Prior work has mainly focused on perceptual hallucinations, leaving reasoning level unfaithfulness underexplored. To isolate faithfulness from linguistic priors, we introduce SPD-Faith Bench, a diagnostic benchmark based on fine-grained image difference reasoning that enforces explicit visual comparison. Evaluations on state-of-the-art MLLMs reveal two systematic failure modes, perceptual blindness and perception-reasoning dissociation. We trace these failures to decaying visual attention and representation shifts in the residual stream. Guided by this analysis, we propose SAGE, a train-free visual evidence-calibrated framework that improves visual routing and aligns reasoning with perception. Our results highlight the importance of explicitly evaluating faithfulness beyond response correctness. Our benchmark and codes are available at this https URL. 

---
# Data Darwinism Part I: Unlocking the Value of Scientific Data for Pre-training 

**Authors**: Yiwei Qin, Zhen Huang, Tiantian Mi, Weiye Si, Chenyang Zhou, Qipeng Guo, Siyuan Feng, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07824)  

**Abstract**: Data quality determines foundation model performance, yet systematic processing frameworks are lacking. We introduce Data Darwinism, a ten-level taxonomy (L0-L9) that conceptualizes data-model co-evolution: advanced models produce superior data for next-generation systems. We validate this on scientific literature by constructing Darwin-Science, a 900B-token corpus (L0-L5). We identify a learnability gap in raw scientific text, which we bridge via L4 (Generative Refinement) and L5 (Cognitive Completion) using frontier LLMs to explicate reasoning and terminology.
To ensure rigorous attribution, we pre-trained daVinci-origin-3B/7B models from scratch, excluding scientific content to create contamination-free baselines. After 600B tokens of continued pre-training, Darwin-Science outperforms baselines by +2.12 (3B) and +2.95 (7B) points across 20+ benchmarks, rising to +5.60 and +8.40 points on domain-aligned tasks. Systematic progression to L5 yields a +1.36 total gain, confirming that higher-level processing unlocks latent data value. We release the Darwin-Science corpus and daVinci-origin models to enable principled, co-evolutionary development. 

---
# ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs 

**Authors**: Yanlin Qi, Xinhang Chen, Huiqiang Jiang, Qitong Wang, Botao Peng, Themis Palpanas  

**Link**: [PDF](https://arxiv.org/pdf/2602.07721)  

**Abstract**: KV-cache retrieval is essential for long-context LLM inference, yet existing methods struggle with distribution drift and high latency at scale. We introduce ParisKV, a drift-robust, GPU-native KV-cache retrieval framework based on collision-based candidate selection, followed by a quantized inner-product reranking estimator. For million-token contexts, ParisKV supports CPU-offloaded KV caches via Unified Virtual Addressing (UVA), enabling on-demand top-$k$ fetching with minimal overhead. ParisKV matches or outperforms full attention quality on long-input and long-generation benchmarks. It achieves state-of-the-art long-context decoding efficiency: it matches or exceeds full attention speed even at batch size 1 for long contexts, delivers up to 2.8$\times$ higher throughput within full attention's runnable range, and scales to million-token contexts where full attention runs out of memory. At million-token scale, ParisKV reduces decode latency by 17$\times$ and 44$\times$ compared to MagicPIG and PQCache, respectively, two state-of-the-art KV-cache Top-$k$ retrieval baselines. 

---
# On Sequence-to-Sequence Models for Automated Log Parsing 

**Authors**: Adam Sorrenti, Andriy Miranskyy  

**Link**: [PDF](https://arxiv.org/pdf/2602.07698)  

**Abstract**: Log parsing is a critical standard operating procedure in software systems, enabling monitoring, anomaly detection, and failure diagnosis. However, automated log parsing remains challenging due to heterogeneous log formats, distribution shifts between training and deployment data, and the brittleness of rule-based approaches. This study aims to systematically evaluate how sequence modelling architecture, representation choice, sequence length, and training data availability influence automated log parsing performance and computational cost. We conduct a controlled empirical study comparing four sequence modelling architectures: Transformer, Mamba state-space, monodirectional LSTM, and bidirectional LSTM models. In total, 396 models are trained across multiple dataset configurations and evaluated using relative Levenshtein edit distance with statistical significance testing. Transformer achieves the lowest mean relative edit distance (0.111), followed by Mamba (0.145), mono-LSTM (0.186), and bi-LSTM (0.265), where lower values are better. Mamba provides competitive accuracy with substantially lower computational cost. Character-level tokenization generally improves performance, sequence length has negligible practical impact on Transformer accuracy, and both Mamba and Transformer demonstrate stronger sample efficiency than recurrent models. Overall, Transformers reduce parsing error by 23.4%, while Mamba is a strong alternative under data or compute constraints. These results also clarify the roles of representation choice, sequence length, and sample efficiency, providing practical guidance for researchers and practitioners. 

---
# EventCast: Hybrid Demand Forecasting in E-Commerce with LLM-Based Event Knowledge 

**Authors**: Congcong Hu, Yuang Shi, Fan Huang, Yang Xiang, Zhou Ye, Ming Jin, Shiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07695)  

**Abstract**: Demand forecasting is a cornerstone of e-commerce operations, directly impacting inventory planning and fulfillment scheduling. However, existing forecasting systems often fail during high-impact periods such as flash sales, holiday campaigns, and sudden policy interventions, where demand patterns shift abruptly and unpredictably. In this paper, we introduce EventCast, a modular forecasting framework that integrates future event knowledge into time-series prediction. Unlike prior approaches that ignore future interventions or directly use large language models (LLMs) for numerical forecasting, EventCast leverages LLMs solely for event-driven reasoning. Unstructured business data, which covers campaigns, holiday schedules, and seller incentives, from existing operational databases, is processed by an LLM that converts it into interpretable textual summaries leveraging world knowledge for cultural nuances and novel event combinations. These summaries are fused with historical demand features within a dual-tower architecture, enabling accurate, explainable, and scalable forecasts. Deployed on real-world e-commerce scenarios spanning 4 countries of 160 regions over 10 months, EventCast achieves up to 86.9% and 97.7% improvement on MAE and MSE compared to the variant without event knowledge, and reduces MAE by up to 57.0% and MSE by 83.3% versus the best industrial baseline during event-driven periods. EventCast has deployed into real-world industrial pipelines since March 2025, offering a practical solution for improving operational decision-making in dynamic e-commerce environments. 

---
# ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention 

**Authors**: Wenjie Liu, Hao Wu, Xin Qiu, Yingqi Fan, Yihan Zhang, Anhao Zhao, Yunpu Ma, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.07574)  

**Abstract**: Modern multimodal large language models (MLLMs) adopt a unified self-attention design that processes visual and textual tokens at every Transformer layer, incurring substantial computational overhead. In this work, we revisit the necessity of such dense visual processing and show that projected visual embeddings are already well-aligned with the language space, while effective vision-language interaction occurs in only a small subset of layers. Based on these insights, we propose ViCA (Vision-only Cross-Attention), a minimal MLLM architecture in which visual tokens bypass all self-attention and feed-forward layers, interacting with text solely through sparse cross-attention at selected layers. Extensive evaluations across three MLLM backbones, nine multimodal benchmarks, and 26 pruning-based baselines show that ViCA preserves 98% of baseline accuracy while reducing visual-side computation to 4%, consistently achieving superior performance-efficiency trade-offs. Moreover, ViCA provides a regular, hardware-friendly inference pipeline that yields over 3.5x speedup in single-batch inference and over 10x speedup in multi-batch inference, reducing visual grounding to near-zero overhead compared with text-only LLMs. It is also orthogonal to token pruning methods and can be seamlessly combined for further efficiency gains. Our code is available at this https URL. 

---
# When Is Enough Not Enough? Illusory Completion in Search Agents 

**Authors**: Dayoon Ko, Jihyuk Kim, Sohyeon Kim, Haeju Park, Dahyun Lee, Gunhee Kim, Moontae Lee, Kyungjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.07549)  

**Abstract**: Recent search agents leverage multi-turn reasoning and search tools to achieve strong performance on multi-hop and long-horizon benchmarks. Yet it remains unclear whether they reliably reason across all requirements by tracking, verifying, and maintaining multiple conditions in these questions. We study this capability under multi-constraint problems, where valid answers must satisfy several constraints simultaneously. We find that illusory completion frequently occurs, wherein agents believe tasks are complete despite unresolved or violated constraints, leading to underverified answers. To diagnose this behavior, we introduce the Epistemic Ledger, an evaluation framework that tracks evidential support and agents' beliefs for each constraint throughout multi-turn reasoning. Our analysis reveals four recurring failure patterns: bare assertions, overlooked refutations, stagnation, and premature exit. Motivated by these findings, we examine whether explicit constraint-state tracking during execution mitigates these failures via LiveLedger, an inference-time tracker. This simple intervention consistently improves performance, substantially reducing underverified answers (by up to 26.5%) and improving overall accuracy (by up to 11.6%) on multi-constraint problems. 

---
# Linguistic properties and model scale in brain encoding: from small to compressed language models 

**Authors**: Subba Reddy Oota, Vijay Rowtula, Satya Sai Srinath Namburi, Khushbu Pahwa, Anant Khandelwal, Manish Gupta, Tanmoy Chakraborty, Bapi S. Raju  

**Link**: [PDF](https://arxiv.org/pdf/2602.07547)  

**Abstract**: Recent work has shown that scaling large language models (LLMs) improves their alignment with human brain activity, yet it remains unclear what drives these gains and which representational properties are responsible. Although larger models often yield better task performance and brain alignment, they are increasingly difficult to analyze mechanistically. This raises a fundamental question: what is the minimal model capacity required to capture brain-relevant representations? To address this question, we systematically investigate how constraining model scale and numerical precision affects brain alignment. We compare full-precision LLMs, small language models (SLMs), and compressed variants (quantized and pruned) by predicting fMRI responses during naturalistic language comprehension. Across model families up to 14B parameters, we find that 3B SLMs achieve brain predictivity indistinguishable from larger LLMs, whereas 1B models degrade substantially, particularly in semantic language regions. Brain alignment is remarkably robust to compression: most quantization and pruning methods preserve neural predictivity, with GPTQ as a consistent exception. Linguistic probing reveals a dissociation between task performance and brain predictivity: compression degrades discourse, syntax, and morphology, yet brain predictivity remains largely unchanged. Overall, brain alignment saturates at modest model scales and is resilient to compression, challenging common assumptions about neural scaling and motivating compact models for brain-aligned language modeling. 

---
# Training-Driven Representational Geometry Modularization Predicts Brain Alignment in Language Models 

**Authors**: Yixuan Liu, Zhiyuan Ma, Likai Tang, Runmin Gan, Xinche Zhang, Jinhao Li, Chao Xie, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.07539)  

**Abstract**: How large language models (LLMs) align with the neural representation and computation of human language is a central question in cognitive science. Using representational geometry as a mechanistic lens, we addressed this by tracking entropy, curvature, and fMRI encoding scores throughout Pythia (70M-1B) training. We identified a geometric modularization where layers self-organize into stable low- and high-complexity clusters. The low-complexity module, characterized by reduced entropy and curvature, consistently better predicted human language network activity. This alignment followed heterogeneous spatial-temporal trajectories: rapid and stable in temporal regions (AntTemp, PostTemp), but delayed and dynamic in frontal areas (IFG, IFGorb). Crucially, reduced curvature remained a robust predictor of model-brain alignment even after controlling for training progress, an effect that strengthened with model scale. These results links training-driven geometric reorganization to temporal-frontal functional specialization, suggesting that representational smoothing facilitates neural-like linguistic processing. 

---
# MemPot: Defending Against Memory Extraction Attack with Optimized Honeypots 

**Authors**: Yuhao Wang, Shengfang Zhai, Guanghao Jin, Yinpeng Dong, Linyi Yang, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07517)  

**Abstract**: Large Language Model (LLM)-based agents employ external and internal memory systems to handle complex, goal-oriented tasks, yet this exposes them to severe extraction attacks, and effective defenses remain lacking. In this paper, we propose MemPot, the first theoretically verified defense framework against memory extraction attacks by injecting optimized honeypots into the memory. Through a two-stage optimization process, MemPot generates trap documents that maximize the retrieval probability for attackers while remaining inconspicuous to benign users. We model the detection process as Wald's Sequential Probability Ratio Test (SPRT) and theoretically prove that MemPot achieves a lower average number of sampling rounds compared to optimal static detectors. Empirically, MemPot significantly outperforms state-of-the-art baselines, achieving a 50% improvement in detection AUROC and an 80% increase in True Positive Rate under low False Positive Rate constraints. Furthermore, our experiments confirm that MemPot incurs zero additional online inference latency and preserves the agent's utility on standard tasks, verifying its superiority in safety, harmlessness, and efficiency. 

---
# On the Importance of a Multi-Scale Calibration for Quantization 

**Authors**: Seungwoo Son, Ingyu Seong, Junhan Kim, Hyemi Jang, Yongkweon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2602.07465)  

**Abstract**: Post-training quantization (PTQ) is a cornerstone for efficiently deploying large language models (LLMs), where a small calibration set critically affects quantization performance. However, conventional practices rely on random sequences of fixed length, overlooking the variable-length nature of LLM inputs. Input length directly influences the activation distribution and, consequently, the weight importance captured by the Hessian, which in turn affects quantization outcomes. As a result, Hessian estimates derived from fixed-length calibration may fail to represent the true importance of weights across diverse input scenarios. We propose MaCa (Matryoshka Calibration), a simple yet effective method for length-aware Hessian construction. MaCa (i) incorporates multi-scale sequence length information into Hessian estimation and (ii) regularizes each sequence as an independent sample, yielding a more stable and fruitful Hessian for accurate quantization. Experiments on state-of-the-art LLMs (e.g., Qwen3, Gemma3, LLaMA3) demonstrate that MaCa consistently improves accuracy under low bit quantization, offering a lightweight enhancement compatible with existing PTQ frameworks. To the best of our knowledge, this is the first work to systematically highlight the role of multi-scale calibration in LLM quantization. 

---
# Pull Requests as a Training Signal for Repo-Level Code Editing 

**Authors**: Qinglin Zhu, Tianyu Chen, Shuai Lu, Lei Ji, Runcong Zhao, Murong Ma, Xiangxiang Dai, Yulan He, Lin Gui, Peng cheng, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2602.07457)  

**Abstract**: Repository-level code editing requires models to understand complex dependencies and execute precise multi-file modifications across a large codebase. While recent gains on SWE-bench rely heavily on complex agent scaffolding, it remains unclear how much of this capability can be internalised via high-quality training signals. To address this, we propose Clean Pull Request (Clean-PR), a mid-training paradigm that leverages real-world GitHub pull requests as a training signal for repository-level editing. We introduce a scalable pipeline that converts noisy pull request diffs into Search/Replace edit blocks through reconstruction and validation, resulting in the largest publicly available corpus of 2 million pull requests spanning 12 programming languages. Using this training signal, we perform a mid-training stage followed by an agentless-aligned supervised fine-tuning process with error-driven data augmentation. On SWE-bench, our model significantly outperforms the instruction-tuned baseline, achieving absolute improvements of 13.6% on SWE-bench Lite and 12.3% on SWE-bench Verified. These results demonstrate that repository-level code understanding and editing capabilities can be effectively internalised into model weights under a simplified, agentless protocol, without relying on heavy inference-time scaffolding. 

---
# Sign-Based Optimizers Are Effective Under Heavy-Tailed Noise 

**Authors**: Dingzhi Yu, Hongyi Tao, Yuanyu Wan, Luo Luo, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07425)  

**Abstract**: While adaptive gradient methods are the workhorse of modern machine learning, sign-based optimization algorithms such as Lion and Muon have recently demonstrated superior empirical performance over AdamW in training large language models (LLM). However, a theoretical understanding of why sign-based updates outperform variance-adapted methods remains elusive. In this paper, we aim to bridge the gap between theory and practice through the lens of heavy-tailed gradient noise, a phenomenon frequently observed in language modeling tasks. Theoretically, we introduce a novel generalized heavy-tailed noise condition that captures the behavior of LLMs more accurately than standard finite variance assumptions. Under this noise model, we establish sharp convergence rates of SignSGD and Lion for generalized smooth function classes, matching or surpassing previous best-known bounds. Furthermore, we extend our analysis to Muon and Muonlight, providing what is, to our knowledge, the first rigorous analysis of matrix optimization under heavy-tailed stochasticity. These results offer a strong theoretical justification for the empirical superiority of sign-based optimizers, showcasing that they are naturally suited to handle the noisy gradients associated with heavy tails. Empirically, LLM pretraining experiments validate our theoretical insights and confirm that our proposed noise models are well-aligned with practice. 

---
# Secure Code Generation via Online Reinforcement Learning with Vulnerability Reward Model 

**Authors**: Tianyi Wu, Mingzhe Du, Yue Liu, Chengran Yang, Terry Yue Zhuo, Jiaheng Zhang, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07422)  

**Abstract**: Large language models (LLMs) are increasingly used in software development, yet their tendency to generate insecure code remains a major barrier to real-world deployment. Existing secure code alignment methods often suffer from a functionality--security paradox, improving security at the cost of substantial utility degradation. We propose SecCoderX, an online reinforcement learning framework for functionality-preserving secure code generation. SecCoderX first bridges vulnerability detection and secure code generation by repurposing mature detection resources in two ways: (i) synthesizing diverse, reality-grounded vulnerability-inducing coding tasks for online RL rollouts, and (ii) training a reasoning-based vulnerability reward model that provides scalable and reliable security supervision. Together, these components are unified in an online RL loop to align code LLMs to generate secure and functional code. Extensive experiments demonstrate that SecCoderX achieves state-of-the-art performance, improving Effective Safety Rate (ESR) by approximately 10% over unaligned models, whereas prior methods often degrade ESR by 14-54%. We release our code, dataset and model checkpoints at this https URL. 

---
# Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution 

**Authors**: Deuksin Kwon, Kaleen Shrestha, Bin Han, Spencer Lin, James Hale, Jonathan Gratch, Maja Matarić, Gale M. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2602.07414)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human behavior in social settings such as legal mediation, negotiation, and dispute resolution. However, it remains unclear whether these simulations reproduce the personality-behavior patterns observed in humans. Human personality, for instance, shapes how individuals navigate social interactions, including strategic choices and behaviors in emotionally charged interactions. This raises the question: Can LLMs, when prompted with personality traits, reproduce personality-driven differences in human conflict behavior? To explore this, we introduce an evaluation framework that enables direct comparison of human-human and LLM-LLM behaviors in dispute resolution dialogues with respect to Big Five Inventory (BFI) personality traits. This framework provides a set of interpretable metrics related to strategic behavior and conflict outcomes. We additionally contribute a novel dataset creation methodology for LLM dispute resolution dialogues with matched scenarios and personality traits with respect to human conversations. Finally, we demonstrate the use of our evaluation framework with three contemporary closed-source LLMs and show significant divergences in how personality manifests in conflict across different LLMs compared to human data, challenging the assumption that personality-prompted agents can serve as reliable behavioral proxies in socially impactful applications. Our work highlights the need for psychological grounding and validation in AI simulations before real-world use. 

---
# High Fidelity Textual User Representation over Heterogeneous Sources via Reinforcement Learning 

**Authors**: Rajat Arora, Ye Tao, Jianqiang Shen, Ping Liu, Muchen Wu, Qianqi Shen, Benjamin Le, Fedor Borisyuk, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07333)  

**Abstract**: Effective personalization on large-scale job platforms requires modeling members based on heterogeneous textual sources, including profiles, professional data, and search activity logs. As recommender systems increasingly adopt Large Language Models (LLMs), creating unified, interpretable, and concise representations from heterogeneous sources becomes critical, especially for latency-sensitive online environments. In this work, we propose a novel Reinforcement Learning (RL) framework to synthesize a unified textual representation for each member. Our approach leverages implicit user engagement signals (e.g., clicks, applies) as the primary reward to distill salient information. Additionally, the framework is complemented by rule-based rewards that enforce formatting and length constraints. Extensive offline experiments across multiple LinkedIn products, one of the world's largest job platforms, demonstrate significant improvements in key downstream business metrics. This work provides a practical, labeling-free, and scalable solution for constructing interpretable user representations that are directly compatible with LLM-based systems. 

---
# Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs 

**Authors**: Pengrui Han, Xueqiang Xu, Keyang Xuan, Peiyang Song, Siru Ouyang, Runchu Tian, Yuqing Jiang, Cheng Qian, Pengcheng Jiang, Jiashuo Sun, Junxia Cui, Ming Zhong, Ge Liu, Jiawei Han, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2602.07276)  

**Abstract**: Activation steering has emerged as a promising approach for efficiently adapting large language models (LLMs) to downstream behaviors. However, most existing steering methods rely on a single static direction per task or concept, making them inflexible under task variation and inadequate for complex tasks that require multiple coordinated capabilities. To address this limitation, we propose STEER2ADAPT, a lightweight framework that adapts LLMs by composing steering vectors rather than learning new ones from scratch. In many domains (e.g., reasoning or safety), tasks share a small set of underlying concept dimensions. STEER2ADAPT captures these dimensions as a reusable, low-dimensional semantic prior subspace, and adapts to new tasks by dynamically discovering a linear combination of basis vectors from only a handful of examples. Experiments across 9 tasks and 3 models in both reasoning and safety domains demonstrate the effectiveness of STEER2ADAPT, achieving an average improvement of 8.2%. Extensive analyses further show that STEER2ADAPT is a data-efficient, stable, and transparent inference-time adaptation method for LLMs. 

---
# BRIDGE: Predicting Human Task Completion Time From Model Performance 

**Authors**: Fengyuan Liu, Jay Gala, Nilaksh, Dzmitry Bahdanau, Siva Reddy, Hugo Larochelle  

**Link**: [PDF](https://arxiv.org/pdf/2602.07267)  

**Abstract**: Evaluating the real-world capabilities of AI systems requires grounding benchmark performance in human-interpretable measures of task difficulty. Existing approaches that rely on direct human task completion time annotations are costly, noisy, and difficult to scale across benchmarks. In this work, we propose BRIDGE, a unified psychometric framework that learns the latent difficulty scale from model responses and anchors it to human task completion time. Using a two-parameter logistic Item Response Theory model, we jointly estimate latent task difficulty and model capability from model performance data across multiple benchmarks. We demonstrate that latent task difficulty varies linearly with the logarithm of human completion time, allowing human task completion time to be inferred for new benchmarks from model performance alone. Leveraging this alignment, we forecast frontier model capabilities in terms of human task length and independently reproduce METR's exponential scaling results, with the 50% solvable task horizon doubling approximately every 6 months. 

---
# From Out-of-Distribution Detection to Hallucination Detection: A Geometric View 

**Authors**: Litian Liu, Reza Pourreza, Yubing Jian, Yao Qin, Roland Memisevic  

**Link**: [PDF](https://arxiv.org/pdf/2602.07253)  

**Abstract**: Detecting hallucinations in large language models is a critical open problem with significant implications for safety and reliability. While existing hallucination detection methods achieve strong performance in question-answering tasks, they remain less effective on tasks requiring reasoning. In this work, we revisit hallucination detection through the lens of out-of-distribution (OOD) detection, a well-studied problem in areas like computer vision. Treating next-token prediction in language models as a classification task allows us to apply OOD techniques, provided appropriate modifications are made to account for the structural differences in large language models. We show that OOD-based approaches yield training-free, single-sample-based detectors, achieving strong accuracy in hallucination detection for reasoning tasks. Overall, our work suggests that reframing hallucination detection as OOD detection provides a promising and scalable pathway toward language model safety. 

---
# Measuring Complexity at the Requirements Stage: Spectral Metrics as Development Effort Predictors 

**Authors**: Maximilian Vierlboeck, Antonio Pugliese, Roshanak Nilchian, Paul Grogan, Rashika Sugganahalli Natesh Babu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07182)  

**Abstract**: Complexity in engineered systems presents one of the most persistent challenges in modern development since it is driving cost overruns, schedule delays, and outright project failures. Yet while architectural complexity has been studied, the structural complexity embedded within requirements specifications remains poorly understood and inadequately quantified. This gap is consequential: requirements fundamentally drive system design, and complexity introduced at this stage propagates through architecture, implementation, and integration. To address this gap, we build on Natural Language Processing methods that extract structural networks from textual requirements. Using these extracted structures, we conducted a controlled experiment employing molecular integration tasks as structurally isomorphic proxies for requirements integration - leveraging the topological equivalence between molecular graphs and requirement networks while eliminating confounding factors such as domain expertise and semantic ambiguity. Our results demonstrate that spectral measures predict integration effort with correlations exceeding 0.95, while structural metrics achieve correlations above 0.89. Notably, density-based metrics show no significant predictive validity. These findings indicate that eigenvalue-derived measures capture cognitive and effort dimensions that simpler connectivity metrics cannot. As a result, this research bridges a critical methodological gap between architectural complexity analysis and requirements engineering practice, providing a validated foundation for applying these metrics to requirements engineering, where similar structural complexity patterns may predict integration effort. 

---
# An Information-Theoretic Framework for Comparing Voice and Text Explainability 

**Authors**: Mona Rajhans, Vishal Khawarey  

**Link**: [PDF](https://arxiv.org/pdf/2602.07179)  

**Abstract**: Explainable Artificial Intelligence (XAI) aims to make machine learning models transparent and trustworthy, yet most current approaches communicate explanations visually or through text. This paper introduces an information theoretic framework for analyzing how explanation modality specifically, voice versus text affects user comprehension and trust calibration in AI systems. The proposed model treats explanation delivery as a communication channel between model and user, characterized by metrics for information retention, comprehension efficiency (CE), and trust calibration error (T CE). A simulation framework implemented in Python was developed to evaluate these metrics using synthetic SHAP based feature attributions across multiple modality style configurations (brief, detailed, and analogy based). Results demonstrate that text explanations achieve higher comprehension efficiency, while voice explanations yield improved trust calibration, with analogy based delivery achieving the best overall trade off. This framework provides a reproducible foundation for designing and benchmarking multimodal explainability systems and can be extended to empirical studies using real SHAP or LIME outputs on open datasets such as the UCI Credit Approval or Kaggle Financial Transactions datasets. 

---
# Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate 

**Authors**: Zhiqi Bu, Shiyun Xu, Jialin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2602.07145)  

**Abstract**: Deep learning has non-convex loss landscape and its optimization dynamics is hard to analyze or control. Nevertheless, the dynamics can be empirically convex-like across various tasks, models, optimizers, hyperparameters, etc. In this work, we examine the applicability of convexity and Lipschitz continuity in deep learning, in order to precisely control the loss dynamics via the learning rate schedules. We illustrate that deep learning quickly becomes weakly convex after a short period of training, and the loss is predicable by an upper bound on the last iterate, which further informs the scaling of optimal learning rate. Through the lens of convexity, we build scaling laws of learning rates and losses that extrapolate as much as 80X across training horizons and 70X across model sizes. 

---
# Massive Sound Embedding Benchmark (MSEB) 

**Authors**: Georg Heigold, Ehsan Variani, Tom Bagby, Cyril Allauzen, Ji Ma, Shankar Kumar, Michael Riley  

**Link**: [PDF](https://arxiv.org/pdf/2602.07143)  

**Abstract**: Audio is a critical component of multimodal perception, and any truly intelligent system must demonstrate a wide range of auditory capabilities. These capabilities include transcription, classification, retrieval, reasoning, segmentation, clustering, reranking, and reconstruction. Fundamentally, each task involves transforming a raw audio signal into a meaningful 'embedding' - be it a single vector, a sequence of continuous or discrete representations, or another structured form - which then serves as the basis for generating the task's final response. To accelerate progress towards robust machine auditory intelligence, we present the Massive Sound Embedding Benchmark (MSEB): an extensible framework designed to evaluate the auditory components of any multimodal system. In its first release, MSEB offers a comprehensive suite of eight core tasks, with more planned for the future, supported by diverse datasets, including the new, large-scale Simple Voice Questions (SVQ) dataset. Our initial experiments establish clear performance headrooms, highlighting the significant opportunity to improve real-world multimodal experiences where audio is a core signal. We encourage the research community to use MSEB to assess their algorithms and contribute to its growth. The library is publicly hosted at github. 

---
# Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models 

**Authors**: Haoyu Zhang, Zhipeng Li, Yiwen Guo, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07106)  

**Abstract**: Omni-modal large language models (OLLMs) aim to unify multimodal understanding and generation, yet incorporating speech with 3D facial animation remains largely unexplored despite its importance for natural interaction. A key challenge arises from the representation mismatch between discrete, token-level semantic reasoning in LLMs and the dense, fine-grained temporal dynamics required for 3D facial motion, which makes direct modeling difficult to optimize under limited data. We propose Expressive Omni (Ex-Omni), an open-source omni-modal framework that augments OLLMs with speech-accompanied 3D facial animation. Ex-Omni reduces learning difficulty by decoupling semantic reasoning from temporal generation, leveraging speech units as temporal scaffolding and a unified token-as-query gated fusion (TQGF) mechanism for controlled semantic injection. We further introduce InstructEx, a dataset aims to facilitate augment OLLMs with speech-accompanied 3D facial animation. Extensive experiments demonstrate that Ex-Omni performs competitively against existing open-source OLLMs while enabling stable aligned speech and facial animation generation. 

---
# Evaluating Retrieval-Augmented Generation Variants for Natural Language-Based SQL and API Call Generation 

**Authors**: Michael Marketsmüller, Simon Martin, Tim Schlippe  

**Link**: [PDF](https://arxiv.org/pdf/2602.07086)  

**Abstract**: Enterprise systems increasingly require natural language interfaces that can translate user requests into structured operations such as SQL queries and REST API calls. While large language models (LLMs) show promise for code generation [Chen et al., 2021; Huynh and Lin, 2025], their effectiveness in domain-specific enterprise contexts remains underexplored, particularly when both retrieval and modification tasks must be handled jointly. This paper presents a comprehensive evaluation of three retrieval-augmented generation (RAG) variants [Lewis et al., 2021] -- standard RAG, Self-RAG [Asai et al., 2024], and CoRAG [Wang et al., 2025] -- across SQL query generation, REST API call generation, and a combined task requiring dynamic task classification. Using SAP Transactional Banking as a realistic enterprise use case, we construct a novel test dataset covering both modalities and evaluate 18 experimental configurations under database-only, API-only, and hybrid documentation contexts. Results demonstrate that RAG is essential: Without retrieval, exact match accuracy is 0% across all tasks, whereas retrieval yields substantial gains in execution accuracy (up to 79.30%) and component match accuracy (up to 78.86%). Critically, CoRAG proves most robust in hybrid documentation settings, achieving statistically significant improvements in the combined task (10.29% exact match vs. 7.45% for standard RAG), driven primarily by superior SQL generation performance (15.32% vs. 11.56%). Our findings establish retrieval-policy design as a key determinant of production-grade natural language interfaces, showing that iterative query decomposition outperforms both top-k retrieval and binary relevance filtering under documentation heterogeneity. 

---
# Comprehensive Evaluation of Large Language Models on Software Engineering Tasks: A Multi-Task Benchmark 

**Authors**: Go Frendi Gunawan, Mukhlis Amien  

**Link**: [PDF](https://arxiv.org/pdf/2602.07079)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in software engineering, yet comprehensive benchmarks covering diverse SE activities remain limited. We present a multi-task evaluation of 11 state-of-the-art LLMs across five representative software engineering tasks: bug fixing, feature development, code refactoring, technical copywriting, and research synthesis. Our automated verification framework measures both output quality and completion efficiency. Key findings reveal that (1) models achieving identical perfect scores exhibit 22x variation in completion time, 49x variation in tool efficiency, and 53x variation in estimated cost; (2) tool usage frequency shows no correlation with success (r = 0.077, p = 0.575) - one model used 917 tool calls while another solved the same task with 3 calls; (3) we identify two distinct inefficiency patterns: loop inefficiency and inference inefficiency; and (4) coding tasks achieve 100 percent success while research tasks present greater challenges (90.9 percent). We release all experimental data, verification scripts, and analysis code for full reproducibility. 

---
# LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning 

**Authors**: Xinwu Ye, Yicheng Mao, Jia Zhang, Yimeng Liu, Li Hao, Fang Wu, Zhiwei Li, Yuxuan Liao, Zehong Wang, Zhiyuan Liu, Zhenfei Yin, Li Yuan, Philip Torr, Huan Sun, Xiangxiang Zeng, Mengdi Wang, Le Cong, Shenghua Gao, Xiangru Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07075)  

**Abstract**: Chemical large language models (LLMs) predominantly rely on explicit Chain-of-Thought (CoT) in natural language to perform complex reasoning. However, chemical reasoning is inherently continuous and structural, and forcing it into discrete linguistic tokens introduces a fundamental representation mismatch that constrains both efficiency and performance. We introduce LatentChem, a latent reasoning interface that decouples chemical computation from textual generation, enabling models to perform multi-step reasoning directly in continuous latent space while emitting language only for final outputs. Remarkably, we observe a consistent emergent behavior: when optimized solely for task success, models spontaneously internalize reasoning, progressively abandoning verbose textual derivations in favor of implicit latent computation. This shift is not merely stylistic but computationally advantageous. Across diverse chemical reasoning benchmarks, LatentChem achieves a 59.88\% non-tie win rate over strong CoT-based baselines on ChemCoTBench, while delivering a 10.84$\times$ average inference speedup. Our results provide empirical evidence that chemical reasoning is more naturally and effectively realized as continuous latent dynamics rather than discretized linguistic trajectories. 

---
# Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration? 

**Authors**: Pingyue Zhang, Zihan Huang, Yue Wang, Jieyu Zhang, Letian Xue, Zihan Wang, Qineng Wang, Keshigeyan Chandrasegaran, Ruohan Zhang, Yejin Choi, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.07055)  

**Abstract**: Spatial embodied intelligence requires agents to act to acquire information under partial observability. While multimodal foundation models excel at passive perception, their capacity for active, self-directed exploration remains understudied. We propose Theory of Space, defined as an agent's ability to actively acquire information through self-directed, active exploration and to construct, revise, and exploit a spatial belief from sequential, partial observations. We evaluate this through a benchmark where the goal is curiosity-driven exploration to build an accurate cognitive map. A key innovation is spatial belief probing, which prompts models to reveal their internal spatial representations at each step. Our evaluation of state-of-the-art models reveals several critical bottlenecks. First, we identify an Active-Passive Gap, where performance drops significantly when agents must autonomously gather information. Second, we find high inefficiency, as models explore unsystematically compared to program-based proxies. Through belief probing, we diagnose that while perception is an initial bottleneck, global beliefs suffer from instability that causes spatial knowledge to degrade over time. Finally, using a false belief paradigm, we uncover Belief Inertia, where agents fail to update obsolete priors with new evidence. This issue is present in text-based agents but is particularly severe in vision-based models. Our findings suggest that current foundation models struggle to maintain coherent, revisable spatial beliefs during active exploration. 

---
# UNIKIE-BENCH: Benchmarking Large Multimodal Models for Key Information Extraction in Visual Documents 

**Authors**: Yifan Ji, Zhipeng Xu, Zhenghao Liu, Zulong Chen, Qian Zhang, Zhibo Yang, Junyang Lin, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.07038)  

**Abstract**: Key Information Extraction (KIE) from real-world documents remains challenging due to substantial variations in layout structures, visual quality, and task-specific information requirements. Recent Large Multimodal Models (LMMs) have shown promising potential for performing end-to-end KIE directly from document images. To enable a comprehensive and systematic evaluation across realistic and diverse application scenarios, we introduce UNIKIE-BENCH, a unified benchmark designed to rigorously evaluate the KIE capabilities of LMMs. UNIKIE-BENCH consists of two complementary tracks: a constrained-category KIE track with scenario-predefined schemas that reflect practical application needs, and an open-category KIE track that extracts any key information that is explicitly present in the document. Experiments on 15 state-of-the-art LMMs reveal substantial performance degradation under diverse schema definitions, long-tail key fields, and complex layouts, along with pronounced performance disparities across different document types and scenarios. These findings underscore persistent challenges in grounding accuracy and layout-aware reasoning for LMM-based KIE. All codes and datasets are available at this https URL. 

---
# MENASpeechBank: A Reference Voice Bank with Persona-Conditioned Multi-Turn Conversations for AudioLLMs 

**Authors**: Zien Sheikh Ali, Hunzalah Hassan Bhatti, Rabindra Nath Nandi, Shammur Absar Chowdhury, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2602.07036)  

**Abstract**: Audio large language models (AudioLLMs) enable instruction-following over speech and general audio, but progress is increasingly limited by the lack of diverse, conversational, instruction-aligned speech-text data. This bottleneck is especially acute for persona-grounded interactions and dialectal coverage, where collecting and releasing real multi-speaker recordings is costly and slow. We introduce MENASpeechBank, a reference speech bank comprising about 18K high-quality utterances from 124 speakers spanning multiple MENA countries, covering English, Modern Standard Arabic (MSA), and regional Arabic varieties. Building on this resource, we develop a controllable synthetic data pipeline that: (i) constructs persona profiles enriched with World Values Survey-inspired attributes, (ii) defines a taxonomy of about 5K conversational scenarios, (iii) matches personas to scenarios via semantic similarity, (iv) generates about 417K role-play conversations with an LLM where the user speaks as the persona and the assistant behaves as a helpful agent, and (v) synthesizes the user turns by conditioning on reference speaker audio to preserve speaker identity and diversity. We evaluate both synthetic and human-recorded conversations and provide detailed analysis. We will release MENASpeechBank and the generated conversations publicly for the community. 

---
# LLM-FSM: Scaling Large Language Models for Finite-State Reasoning in RTL Code Generation 

**Authors**: Yuheng Wu, Berk Gokmen, Zhouhua Xie, Peijing Li, Caroline Trippel, Priyanka Raina, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2602.07032)  

**Abstract**: Finite-state reasoning, the ability to understand and implement state-dependent behavior, is central to hardware design. In this paper, we present LLM-FSM, a benchmark that evaluates how well large language models (LLMs) can recover finite-state machine (FSM) behavior from natural-language specifications and translate it into correct register transfer-level (RTL) implementations. Unlike prior specification-to-RTL benchmarks that rely on manually constructed examples, LLM-FSM is built through a fully automated pipeline. LLM-FSM first constructs FSM with configurable state counts and constrained transition structures. It then prompts LLMs to express each FSM in a structured YAML format with an application context, and to further convert that YAML into a natural-language (NL) specification. From the same YAML, our pipeline synthesizes the reference RTL and testbench in a correct-by-construction manner. All 1,000 problems are verified using LLM-based and SAT-solver-based checks, with human review on a subset. Our experiments show that even the strongest LLMs exhibit sharply declining accuracy as FSM complexity increases. We further demonstrate that training-time scaling via supervised fine-tuning (SFT) generalizes effectively to out-of-distribution (OOD) tasks, while increasing test-time compute improves reasoning reliability. Finally, LLM-FSM remains extensible by allowing its FSM complexity to scale with future model capabilities. 

---
# Attractor Patch Networks: Reducing Catastrophic Forgetting with Routed Low-Rank Patch Experts 

**Authors**: Shashank  

**Link**: [PDF](https://arxiv.org/pdf/2602.06993)  

**Abstract**: Transformers achieve strong language modeling accuracy, yet their position-wise feed-forward networks (FFNs) are dense, globally shared, and typically updated end to end. These properties create two practical tensions. First, dense FFNs spend the same compute on every token regardless of context, and they allocate capacity uniformly even when language exhibits highly clustered context structure. Second, continual learning, in the sense of updating the model while serving a data stream, often produces interference because a small update touches broadly shared weights.
We propose Attractor Patch Networks (APN), a plug-compatible replacement for the Transformer FFN. APN is a bank of patch experts. A similarity router selects a small top-k set of patches for each token by matching the token representation to learned prototypes. Each selected patch emits a low-rank residual update conditioned on a compact code. The architecture yields conditional, context-specialized nonlinear transformations while preserving the standard Transformer interface.
This paper focuses on APN as an architectural primitive. We formalize APN, analyze its expressivity as a piecewise low-rank residual function class, and derive simple interference and stability arguments that make APN naturally compatible with continual learning. In experiments on character-level language modeling, APN achieves competitive perplexity (4.57 vs 4.32 PPL) while enabling dramatically better continual adaptation: when adapting to a shifted domain, APN achieves 2.6 times better retention (11.1 vs 29.4 PPL on the original domain) and 2.8 times better adaptation (6.4 vs 17.8 PPL on the new domain) compared to global fine-tuning of a dense FFN baseline. 

---
# Leveraging Adaptive Group Negotiation for Heterogeneous Multi-Robot Collaboration with Large Language Models 

**Authors**: Siqi Song, Xuanbing Xie, Zonglin Li, Yuqiang Li, Shijie Wang, Biqing Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06967)  

**Abstract**: Multi-robot collaboration tasks often require heterogeneous robots to work together over long horizons under spatial constraints and environmental uncertainties. Although Large Language Models (LLMs) excel at reasoning and planning, their potential for coordinated control has not been fully explored. Inspired by human teamwork, we present CLiMRS (Cooperative Large-Language-Model-Driven Heterogeneous Multi-Robot System), an adaptive group negotiation framework among LLMs for multi-robot collaboration. This framework pairs each robot with an LLM agent and dynamically forms subgroups through a general proposal planner. Within each subgroup, a subgroup manager leads perception-driven multi-LLM discussions to get commands for actions. Feedback is provided by both robot execution outcomes and environment changes. This grouping-planning-execution-feedback loop enables efficient planning and robust execution. To evaluate these capabilities, we introduce CLiMBench, a heterogeneous multi-robot benchmark of challenging assembly tasks. Our experiments show that CLiMRS surpasses the best baseline, achieving over 40% higher efficiency on complex tasks without sacrificing success on simpler ones. Overall, our results demonstrate that leveraging human-inspired group formation and negotiation principles significantly enhances the efficiency of heterogeneous multi-robot collaboration. Our code is available here: this https URL. 

---
