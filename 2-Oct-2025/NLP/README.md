# Energy-Regularized Sequential Model Editing on Hyperspheres 

**Authors**: Qingyuan Liu, Jia-Chen Gu, Yunzhi Yao, Hong Wang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01172)  

**Abstract**: Large language models (LLMs) require constant updates to remain aligned with evolving real-world knowledge. Model editing offers a lightweight alternative to retraining, but sequential editing often destabilizes representations and induces catastrophic forgetting. In this work, we seek to better understand and mitigate performance degradation caused by sequential editing. We hypothesize that hyperspherical uniformity, a property that maintains uniform distribution of neuron weights on a hypersphere, helps the model remain stable, retain prior knowledge, while still accommodate new updates. We use Hyperspherical Energy (HE) to quantify neuron uniformity during editing, and examine its correlation with editing performance. Empirical studies across widely used editing methods reveals a strong correlation between HE dynamics and editing performance, with editing failures consistently coinciding with high HE fluctuations. We further theoretically prove that HE dynamics impose a lower bound on the degradation of pretrained knowledge, highlighting why HE stability is crucial for knowledge retention. Motivated by these insights, we propose SPHERE (Sparse Projection for Hyperspherical Energy-Regularized Editing), an HE-driven regularization strategy that stabilizes neuron weight distributions, ultimately preserving prior knowledge while enabling reliable sequential updates. Specifically, SPHERE identifies a sparse space complementary to the principal hyperspherical directions of the pretrained weight matrices and projects new knowledge onto it, attenuating perturbations on the principal directions. Extensive experiments on LLaMA3 (8B) and Qwen2.5 (7B) show that SPHERE outperforms the best baseline in editing capability by an average of 16.41%, while most faithfully preserving general model performance, thereby offering a principled path toward reliable large-scale knowledge editing. 

---
# Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity 

**Authors**: Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael R. Tomz, Christopher D. Manning, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01171)  

**Abstract**: Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., ``Generate 5 jokes about coffee and their corresponding probabilities''). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity. 

---
# GRAD: Generative Retrieval-Aligned Demonstration Sampler for Efficient Few-Shot Reasoning 

**Authors**: Oussama Gabouj, Kamel Charaf, Ivan Zakazov, Nicolas Baldwin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2510.01165)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across diverse tasks, but their effectiveness often depends on the quality of the provided context. Retrieval-Augmented Generation (RAG) enriches prompts with external information, but its reliance on static databases constrains adaptability and can result in irrelevant demonstrations. In this work, we propose a Generative Retrieval-Aligned Demonstrator (GRAD), a dynamic demonstration-based approach where an LLM model is trained to generate input-specific concise demonstrations. By tailoring demonstrations to each input, our method offers better contextual support than traditional RAG approaches. We demonstrate the superiority of GRAD under budget constraints, where we limit both the number of tokens used per demonstration and the number of tokens used for the final output. Trained solely on a math dataset, GRAD consistently outperforms strong baselines on Qwen2.5-14B across mathematical reasoning and advanced STEM questions, highlighting GRAD's robust generalization to out-of-distribution (OOD) domains such as physics, chemistry, and computer science. Furthermore, we show that demonstrations generated by trained smaller models can effectively guide larger target models, reducing training costs while maintaining competitive accuracy. Overall, this work introduces a scalable demonstration generator model presenting the first step toward a dynamic few-shot learning paradigm in resource-constrained settings. We release the code used for the project. 

---
# Social Welfare Function Leaderboard: When LLM Agents Allocate Social Welfare 

**Authors**: Zhengliang Shi, Ruotian Ma, Jen-tse Huang, Xinbei Ma, Xingyu Chen, Mengru Wang, Qu Yang, Yue Wang, Fanghua Ye, Ziyang Chen, Shanyi Wang, Cixing Li, Wenxuan Wang, Zhaopeng Tu, Xiaolong Li, Zhaochun Ren, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2510.01164)  

**Abstract**: Large language models (LLMs) are increasingly entrusted with high-stakes decisions that affect human welfare. However, the principles and values that guide these models when distributing scarce societal resources remain largely unexamined. To address this, we introduce the Social Welfare Function (SWF) Benchmark, a dynamic simulation environment where an LLM acts as a sovereign allocator, distributing tasks to a heterogeneous community of recipients. The benchmark is designed to create a persistent trade-off between maximizing collective efficiency (measured by Return on Investment) and ensuring distributive fairness (measured by the Gini coefficient). We evaluate 20 state-of-the-art LLMs and present the first leaderboard for social welfare allocation. Our findings reveal three key insights: (i) A model's general conversational ability, as measured by popular leaderboards, is a poor predictor of its allocation skill. (ii) Most LLMs exhibit a strong default utilitarian orientation, prioritizing group productivity at the expense of severe inequality. (iii) Allocation strategies are highly vulnerable, easily perturbed by output-length constraints and social-influence framing. These results highlight the risks of deploying current LLMs as societal decision-makers and underscore the need for specialized benchmarks and targeted alignment for AI governance. 

---
# Backdoor Attacks Against Speech Language Models 

**Authors**: Alexandrine Fortier, Thomas Thebaud, Jesús Villalba, Najim Dehak, Patrick Cardinal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01157)  

**Abstract**: Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders. 

---
# Pay-Per-Search Models are Abstention Models 

**Authors**: Mustafa Omer Gul, Claire Cardie, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01152)  

**Abstract**: LLMs cannot reliably recognize their parametric knowledge boundaries and often hallucinate answers to outside-of-boundary questions. In contrast, humans recognize their limitations and can either seek external help for such questions or abstain. In this paper, we introduce MASH (Modeling Abstention via Selective Help-seeking), a training framework that readily extracts abstentions from LLMs. Our key idea is that any external help-seeking by an LLM, i.e. search tool use, can serve as a proxy for abstention if the external help (search) is appropriately penalized while simultaneously rewarding answer accuracy. MASH operationalizes this idea using reinforcement learning with a pay-per-search reward.
We run experiments on three knowledge-intensive QA datasets. Our results show that MASH substantially improves upon the selective help-seeking performance of prior efficient search approaches; on multi-hop datasets, MASH improves answer accuracy by 7.6%. Furthermore, MASH demonstrates strong off-the-shelf abstention -- it can distinguish between unanswerable/answerable questions and selectively generate responses for answerable questions -- showcasing behavior analogous to specialized abstention approaches. We emphasize that contrary to prior abstention methods, MASH does not require pre-determining knowledge boundaries to construct training data. Instead, MASH's abstentions are a by-product of training for the auxiliary selective help-seeking task. Overall, we show that MASH training effectively aligns search tool use with parametric knowledge, which can be successfully leveraged for making abstention decisions. 

---
# mR3: Multilingual Rubric-Agnostic Reward Reasoning Models 

**Authors**: David Anugraha, Shou-Yi Hung, Zilu Tang, Annie En-Shiun Lee, Derry Tanti Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2510.01146)  

**Abstract**: Evaluation using Large Language Model (LLM) judges has been widely adopted in English and shown to be effective for automatic evaluation. However, their performance does not generalize well to non-English settings, and it remains unclear what constitutes effective multilingual training for such judges. In this paper, we introduce mR3, a massively multilingual, rubric-agnostic reward reasoning model trained on 72 languages, achieving the broadest language coverage in reward modeling to date. We present a comprehensive study of data and curriculum selection for training to identify effective strategies and data sources for building high-quality reward models, including the integration of target-language reasoning datasets. Our approach attains state-of-the-art performance on multilingual reward model benchmarks, surpassing much larger models (i.e., GPT-OSS-120B) while being up to 9x smaller, and its effectiveness is further confirmed through extensive ablation studies. Our models, data, and code are available as open source at this https URL. 

---
# Automatic Speech Recognition (ASR) for African Low-Resource Languages: A Systematic Literature Review 

**Authors**: Sukairaj Hafiz Imam, Tadesse Destaw Belay, Kedir Yassin Husse, Ibrahim Said Ahmad, Idris Abdulmumin, Hadiza Ali Umar, Muhammad Yahuza Bello, Joyce Nakatumba-Nabende, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2510.01145)  

**Abstract**: ASR has achieved remarkable global progress, yet African low-resource languages remain rigorously underrepresented, producing barriers to digital inclusion across the continent with more than +2000 languages. This systematic literature review (SLR) explores research on ASR for African languages with a focus on datasets, models and training methods, evaluation techniques, challenges, and recommends future directions. We employ the PRISMA 2020 procedures and search DBLP, ACM Digital Library, Google Scholar, Semantic Scholar, and arXiv for studies published between January 2020 and July 2025. We include studies related to ASR datasets, models or metrics for African languages, while excluding non-African, duplicates, and low-quality studies (score <3/5). We screen 71 out of 2,062 records and we record a total of 74 datasets across 111 languages, encompassing approximately 11,206 hours of speech. Fewer than 15% of research provided reproducible materials, and dataset licensing is not clear. Self-supervised and transfer learning techniques are promising, but are hindered by limited pre-training data, inadequate coverage of dialects, and the availability of resources. Most of the researchers use Word Error Rate (WER), with very minimal use of linguistically informed scores such as Character Error Rate (CER) or Diacritic Error Rate (DER), and thus with limited application in tonal and morphologically rich languages. The existing evidence on ASR systems is inconsistent, hindered by issues like dataset availability, poor annotations, licensing uncertainties, and limited benchmarking. Nevertheless, the rise of community-driven initiatives and methodological advancements indicates a pathway for improvement. Sustainable development for this area will also include stakeholder partnership, creation of ethically well-balanced datasets, use of lightweight modelling techniques, and active benchmarking. 

---
# Research on the Integration of Embodied Intelligence and Reinforcement Learning in Textual Domains 

**Authors**: Haonan Wang, Junfeng Sun, Mingjia Zhao, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01076)  

**Abstract**: This article addresses embodied intelligence and reinforcement learning integration in the field of text processing, aiming to enhance text handling with more intelligence on the basis of embodied intelligence's perception and action superiority and reinforcement learning's decision optimization capability. Through detailed theoretical explanation and experimental exploration, a novel integration model is introduced. This model has been demonstrated to be very effective in a wide range oftext processing tasks, validating its applicative potential 

---
# Hybrid Dialogue State Tracking for Persian Chatbots: A Language Model-Based Approach 

**Authors**: Samin Mahdipour Aghabagher, Saeedeh Momtazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01052)  

**Abstract**: Dialogue State Tracking (DST) is an essential element of conversational AI with the objective of deeply understanding the conversation context and leading it toward answering user requests. Due to high demands for open-domain and multi-turn chatbots, the traditional rule-based DST is not efficient enough, since it cannot provide the required adaptability and coherence for human-like experiences in complex conversations. This study proposes a hybrid DST model that utilizes rule-based methods along with language models, including BERT for slot filling and intent detection, XGBoost for intent validation, GPT for DST, and online agents for real-time answer generation. This model is uniquely designed to be evaluated on a comprehensive Persian multi-turn dialogue dataset and demonstrated significantly improved accuracy and coherence over existing methods in Persian-based chatbots. The results demonstrate how effectively a hybrid approach may improve DST capabilities, paving the way for conversational AI systems that are more customized, adaptable, and human-like. 

---
# Interpreting Language Models Through Concept Descriptions: A Survey 

**Authors**: Nils Feldhus, Laura Kopf  

**Link**: [PDF](https://arxiv.org/pdf/2510.01048)  

**Abstract**: Understanding the decision-making processes of neural networks is a central goal of mechanistic interpretability. In the context of Large Language Models (LLMs), this involves uncovering the underlying mechanisms and identifying the roles of individual model components such as neurons and attention heads, as well as model abstractions such as the learned sparse features extracted by Sparse Autoencoders (SAEs). A rapidly growing line of work tackles this challenge by using powerful generator models to produce open-vocabulary, natural language concept descriptions for these components. In this paper, we provide the first survey of the emerging field of concept descriptions for model components and abstractions. We chart the key methods for generating these descriptions, the evolving landscape of automated and human metrics for evaluating them, and the datasets that underpin this research. Our synthesis reveals a growing demand for more rigorous, causal evaluation. By outlining the state of the art and identifying key challenges, this survey provides a roadmap for future research toward making models more transparent. 

---
# Syntax-Guided Diffusion Language Models with User-Integrated Personalization 

**Authors**: Ruqian Zhang, Yijiao Zhang, Juan Shen, Zhongyi Zhu, Annie Qu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01028)  

**Abstract**: Large language models have made revolutionary progress in generating human-like text, yet their outputs often tend to be generic, exhibiting insufficient structural diversity, which limits personalized expression. Recent advances in diffusion models have opened new opportunities for improving language generation beyond the limitations of autoregressive paradigms. In this work, we propose a syntax-guided diffusion language model that integrates structural supervision and personalized conditioning to enhance text quality, diversity, and controllability. We introduce a cascaded framework that generates syntactic guidance before conditional text generation, and further generalize it to a novel noncascaded architecture for better alignment between structure and content. By incorporating syntactic information in the generating process, the proposed model better captures the lexical and structural characteristics of stylistic sentence construction. To enable fine-grained personalization, we develop a shared representation mechanism that facilitates information integration across users, supporting both faithful stylistic generation and generalizable zero-shot inference. Extensive experiments on multiple tasks demonstrate the superiority of our approach in fluency, diversity, and stylistic fidelity. Further qualitative analyses highlight its interpretability and flexibility in learning personalized patterns. 

---
# Analyzing Dialectical Biases in LLMs for Knowledge and Reasoning Benchmarks 

**Authors**: Eileen Pan, Anna Seo Gyeong Choi, Maartje ter Hoeve, Skyler Seto, Allison Koenecke  

**Link**: [PDF](https://arxiv.org/pdf/2510.00962)  

**Abstract**: Large language models (LLMs) are ubiquitous in modern day natural language processing. However, previous work has shown degraded LLM performance for under-represented English dialects. We analyze the effects of typifying "standard" American English language questions as non-"standard" dialectal variants on multiple choice question answering tasks and find up to a 20% reduction in accuracy. Additionally, we investigate the grammatical basis of under-performance in non-"standard" English questions. We find that individual grammatical rules have varied effects on performance, but some are more consequential than others: three specific grammar rules (existential "it", zero copula, and y'all) can explain the majority of performance degradation observed in multiple dialects. We call for future work to investigate bias mitigation methods focused on individual, high-impact grammatical structures. 

---
# Making, not Taking, the Best of N 

**Authors**: Ammar Khairi, Daniel D'souza, Marzieh Fadaee, Julia Kreutzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.00931)  

**Abstract**: Obtaining high-quality generations in modern LLMs has largely been framed as a selection problem: identifying a single winning generation from a diverse pool of N samples, the Best-of-N (BoN). Yet, this approach is inherently zero-sum, discarding diverse and potentially useful information from the pool. Instead, we explore a collaborative setup, where all candidates can potentially contribute to the final winning generation. To this end, we propose Fusion-of-N (FusioN): a method that uses a general LLM judge to synthesize the most informative elements of each sample into a single final answer. We compare FusioN to BoN in two settings, (i) test-time scaling, where we sample and aggregate from a single model at test-time (ii) synthetic data generation, where we fuse samples from a pool of diverse teachers to improve a student model. We extensively benchmark both setups across 11 languages, 3 diverse tasks and varying model scales. Across the bench, FusioN consistently outperforms BoN showing versatility and robustness both in test-time scaling and in downstream gains from synthetic data generation. We also perform extensive analysis on FusioN, where it shows surprising strengths and robustness under challenging settings. These results show that we should shift how we think about evaluating and utilizing LLM generations from a monolithic measure of quality, to embracing their polylithic nature. This shift allows us to integrate diverse strengths, unlock latent potential, and achieve improvements that were previously inaccessible through selection alone. 

---
# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving 

**Authors**: Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00919)  

**Abstract**: Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning. 

---
# Span-level Detection of AI-generated Scientific Text via Contrastive Learning and Structural Calibration 

**Authors**: Zhen Yin, Shenghua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00890)  

**Abstract**: The rapid adoption of large language models (LLMs) in scientific writing raises serious concerns regarding authorship integrity and the reliability of scholarly publications. Existing detection approaches mainly rely on document-level classification or surface-level statistical cues; however, they neglect fine-grained span localization, exhibit weak calibration, and often fail to generalize across disciplines and generators. To address these limitations, we present Sci-SpanDet, a structure-aware framework for detecting AI-generated scholarly texts. The proposed method combines section-conditioned stylistic modeling with multi-level contrastive learning to capture nuanced human-AI differences while mitigating topic dependence, thereby enhancing cross-domain robustness. In addition, it integrates BIO-CRF sequence labeling with pointer-based boundary decoding and confidence calibration to enable precise span-level detection and reliable probability estimates. Extensive experiments on a newly constructed cross-disciplinary dataset of 100,000 annotated samples generated by multiple LLM families (GPT, Qwen, DeepSeek, LLaMA) demonstrate that Sci-SpanDet achieves state-of-the-art performance, with F1(AI) of 80.17, AUROC of 92.63, and Span-F1 of 74.36. Furthermore, it shows strong resilience under adversarial rewriting and maintains balanced accuracy across IMRaD sections and diverse disciplines, substantially surpassing existing baselines. To ensure reproducibility and to foster further research on AI-generated text detection in scholarly documents, the curated dataset and source code will be publicly released upon publication. 

---
# HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in Retrieval-Augmented Generation 

**Authors**: Loris Bergeron, Ioana Buhnila, Jérôme François, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2510.00880)  

**Abstract**: Large Language Models (LLMs) excel in many NLP tasks but remain prone to hallucinations, limiting trust in real-world applications. We present HalluGuard, a 4B-parameter Small Reasoning Model (SRM) for mitigating hallucinations in Retrieval-Augmented Generation (RAG). HalluGuard classifies document-claim pairs as grounded or hallucinated and produces evidence-grounded justifications for transparency. Our approach combines (i) a domain-agnostic synthetic dataset derived from FineWeb and refined through multi-stage curation and data reformation, (ii) synthetic grounded and hallucinated claims, and (iii) preference-based fine-tuning with Odds Ratio Preference Optimization to distill large-model reasoning into a smaller backbone. On the RAGTruth subset of the LLM-AggreFact benchmark, HalluGuard achieves 84.0% balanced accuracy (BAcc), rivaling specialized models, MiniCheck (7B; 84.0%) and Granite Guardian 3.3 (8B; 82.2%) while using roughly half their parameters. Over the full benchmark it reaches 75.7% BAcc, matching larger general-purpose LLMs such as GPT-4o (75.9%). We will release HalluGuard and datasets under Apache 2.0 upon acceptance. 

---
# Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs 

**Authors**: Ziliang Wang, Kang An, Xuhui Zheng, Faqiang Qian, Weikun Zhang, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00861)  

**Abstract**: While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs. 

---
# ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs 

**Authors**: Adi Simhi, Jonathan Herzig, Martin Tutek, Itay Itzhak, Idan Szpektor, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00857)  

**Abstract**: As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at this https URL. 

---
# Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation 

**Authors**: Yanming Sun, Runzhe Zhan, Chi Seng Cheang, Han Wu, Xuebo Liu, Yuyao Niu, Fengying Ye, Kaixin Lan, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00829)  

**Abstract**: \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms. 

---
# Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese 

**Authors**: Jenny Kunz, Iben Nyholm Debess, Annika Simonsen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00810)  

**Abstract**: We investigate how to adapt small, efficient LLMs to Faroese, a low-resource North Germanic language. Starting from English models, we continue pre-training on related Scandinavian languages, either individually or combined via merging, before fine-tuning on Faroese. We compare full fine-tuning with parameter-efficient tuning using LoRA, evaluating their impact on both linguistic accuracy and text comprehension. Due to the lack of existing Faroese evaluation data, we construct two new minimal-pair benchmarks from adapted and newly collected datasets and complement them with human evaluations by Faroese linguists. Our results demonstrate that transfer from related languages is crucial, though the optimal source language depends on the task: Icelandic enhances linguistic accuracy, whereas Danish boosts comprehension. Similarly, the choice between full fine-tuning and LoRA is task-dependent: LoRA improves linguistic acceptability and slightly increases human evaluation scores on the base model, while full fine-tuning yields stronger comprehension performance and better preserves model capabilities during downstream fine-tuning. 

---
# ALARB: An Arabic Legal Argument Reasoning Benchmark 

**Authors**: Harethah Abu Shairah, Somayah AlHarbi, Abdulaziz AlHussein, Sameer Alsabea, Omar Shaqaqi, Hebah AlShamlan, Omar Knio, George Turkiyyah  

**Link**: [PDF](https://arxiv.org/pdf/2510.00694)  

**Abstract**: We introduce ALARB, a dataset and suite of tasks designed to evaluate the reasoning capabilities of large language models (LLMs) within the Arabic legal domain. While existing Arabic benchmarks cover some knowledge-intensive tasks such as retrieval and understanding, substantial datasets focusing specifically on multistep reasoning for Arabic LLMs, especially in open-ended contexts, are lacking. The dataset comprises over 13K commercial court cases from Saudi Arabia, with each case including the facts presented, the reasoning of the court, the verdict, as well as the cited clauses extracted from the regulatory documents. We define a set of challenging tasks leveraging this dataset and reflecting the complexity of real-world legal reasoning, including verdict prediction, completion of reasoning chains in multistep legal arguments, and identification of relevant regulations based on case facts. We benchmark a representative selection of current open and closed Arabic LLMs on these tasks and demonstrate the dataset's utility for instruction tuning. Notably, we show that instruction-tuning a modest 12B parameter model using ALARB significantly enhances its performance in verdict prediction and Arabic verdict generation, reaching a level comparable to that of GPT-4o. 

---
# Inclusive Easy-to-Read Generation for Individuals with Cognitive Impairments 

**Authors**: François Ledoyen, Gaël Dias, Alexis Lechervy, Jeremie Pantin, Fabrice Maurel, Youssef Chahir, Elisa Gouzonnat, Mélanie Berthelot, Stanislas Moravac, Armony Altinier, Amy Khairalla  

**Link**: [PDF](https://arxiv.org/pdf/2510.00691)  

**Abstract**: Ensuring accessibility for individuals with cognitive impairments is essential for autonomy, self-determination, and full citizenship. However, manual Easy-to-Read (ETR) text adaptations are slow, costly, and difficult to scale, limiting access to crucial information in healthcare, education, and civic life. AI-driven ETR generation offers a scalable solution but faces key challenges, including dataset scarcity, domain adaptation, and balancing lightweight learning of Large Language Models (LLMs). In this paper, we introduce ETR-fr, the first dataset for ETR text generation fully compliant with European ETR guidelines. We implement parameter-efficient fine-tuning on PLMs and LLMs to establish generative baselines. To ensure high-quality and accessible outputs, we introduce an evaluation framework based on automatic metrics supplemented by human assessments. The latter is conducted using a 36-question evaluation form that is aligned with the guidelines. Overall results show that PLMs perform comparably to LLMs and adapt effectively to out-of-domain texts. 

---
# Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation 

**Authors**: François Ledoyen, Gaël Dias, Jeremie Pantin, Alexis Lechervy, Fabrice Maurel, Youssef Chahir  

**Link**: [PDF](https://arxiv.org/pdf/2510.00662)  

**Abstract**: Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations. 

---
# MCM-DPO: Multifaceted Cross-Modal Direct Preference Optimization for Alt-text Generation 

**Authors**: Jinlan Fu, Shenzhen Huangfu, Hao Fei, Yichong Huang, Xiaoyu Shen, Xipeng Qiu, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00647)  

**Abstract**: The alt-text generation task produces concise, context-relevant descriptions of images, enabling blind and low-vision users to access online images. Despite the capabilities of large vision-language models, alt-text generation performance remains limited due to noisy user annotations, inconsistent standards, and MLLMs' insensitivity to contextual information. Previous efforts to fine-tune MLLMs using supervised fine-tuning (SFT) have struggled, as SFT relies on accurate target annotations, which are often flawed in user-generated alt-text. To address this, we propose Multi-faceted Cross-modal Direct Preference Optimization (MCM-DPO), which improves alt-text generation by learning to identify better options in preference pairs without requiring precise annotations. MCM-DPO optimizes preferences across single, paired, and multi-preference dimensions, covering textual, visual, and cross-modal factors. In light of the scarcity of high-quality annotated and preference-labeled datasets for alt-text, we constructed two large-scale, high-quality datasets named TAlt and PAlt, sourced from Twitter and Pinterest. These datasets include 202k annotated alt-text samples and 18k preference pairs that cover diverse preference dimensions, aiming to support further research in this domain. Experimental results show that our proposed MCM-DPO method consistently outperforms both DPO and SFT, establishing a new state of the art in alt-text generation. We release the code and data here: this https URL 

---
# Tenyidie Syllabification corpus creation and deep learning applications 

**Authors**: Teisovi Angami, Kevisino Khate  

**Link**: [PDF](https://arxiv.org/pdf/2510.00629)  

**Abstract**: The Tenyidie language is a low-resource language of the Tibeto-Burman family spoken by the Tenyimia Community of Nagaland in the north-eastern part of India and is considered a major language in Nagaland. It is tonal, Subject-Object-Verb, and highly agglutinative in nature. Being a low-resource language, very limited research on Natural Language Processing (NLP) has been conducted. To the best of our knowledge, no work on syllabification has been reported for this language. Among the many NLP tasks, syllabification or syllabication is an important task in which the given word syllables are identified. The contribution of this work is the creation of 10,120 syllabified Tenyidie words and the application of the Deep Learning techniques on the created corpus. In this paper, we have applied LSTM, BLSTM, BLSTM+CRF, and Encoder-decoder deep learning architectures on our created dataset. In our dataset split of 80:10:10 (train:validation:test) set, we achieved the highest accuracy of 99.21% with BLSTM model on the test set. This work will find its application in numerous other NLP applications, such as morphological analysis, part-of-speech tagging, machine translation, etc, for the Tenyidie Language.
Keywords: Tenyidie; NLP; syllabification; deep learning; LSTM; BLSTM; CRF; Encoder-decoder 

---
# SAGE-LD: Towards Scalable and Generalizable End-to-End Language Diarization via Simulated Data Augmentation 

**Authors**: Sangmin Lee, Woongjib Choi, Jihyun Kim, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00582)  

**Abstract**: In this paper, we present a neural spoken language diarization model that supports an unconstrained span of languages within a single framework. Our approach integrates a learnable query-based architecture grounded in multilingual awareness, with large-scale pretraining on simulated code-switching data. By jointly leveraging these two components, our method overcomes the limitations of conventional approaches in data scarcity and architecture optimization, and generalizes effectively to real-world multilingual settings across diverse environments. Experimental results demonstrate that our approach achieves state-of-the-art performance on several language diarization benchmarks, with a relative performance improvement of 23% to 52% over previous methods. We believe that this work not only advances research in language diarization but also establishes a foundational framework for code-switching speech technologies. 

---
# CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs 

**Authors**: Li Li, Ziyi Wang, Yongliang Wu, Jianfei Cai, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00579)  

**Abstract**: Chain-of-Thought (CoT) prompting has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing implementations, such as in-context learning and fine-tuning, remain costly and inefficient. To improve CoT reasoning at a lower cost, and inspired by the task vector paradigm, we introduce CoT Vectors, compact representations that encode task-general, multi-step reasoning knowledge. Through experiments with Extracted CoT Vectors, we observe pronounced layer-wise instability, manifesting as a U-shaped performance curve that reflects a systematic three-stage reasoning process in LLMs. To address this limitation, we propose Learnable CoT Vectors, optimized under a teacher-student framework to provide more stable and robust guidance. Extensive evaluations across diverse benchmarks and models demonstrate that CoT Vectors not only outperform existing baselines but also achieve performance comparable to parameter-efficient fine-tuning methods, while requiring fewer trainable parameters. Moreover, by treating CoT Vectors as a probe, we uncover how their effectiveness varies due to latent space structure, information density, acquisition mechanisms, and pre-training differences, offering new insights into the functional organization of multi-step reasoning in LLMs. The source code will be released. 

---
# ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards 

**Authors**: Shiyu Li, Yang Tang, Yifan Wang, Peiming Li, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00568)  

**Abstract**: Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose ReSeek, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special JUDGE action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce FictionalHot, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness. 

---
# Are Large Language Models Chronically Online Surfers? A Dataset for Chinese Internet Meme Explanation 

**Authors**: Yubo Xie, Chenkai Wang, Zongyang Ma, Fahui Miao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00567)  

**Abstract**: Large language models (LLMs) are trained on vast amounts of text from the Internet, but do they truly understand the viral content that rapidly spreads online -- commonly known as memes? In this paper, we introduce CHIME, a dataset for CHinese Internet Meme Explanation. The dataset comprises popular phrase-based memes from the Chinese Internet, annotated with detailed information on their meaning, origin, example sentences, types, etc. To evaluate whether LLMs understand these memes, we designed two tasks. In the first task, we assessed the models' ability to explain a given meme, identify its origin, and generate appropriate example sentences. The results show that while LLMs can explain the meanings of some memes, their performance declines significantly for culturally and linguistically nuanced meme types. Additionally, they consistently struggle to provide accurate origins for the memes. In the second task, we created a set of multiple-choice questions (MCQs) requiring LLMs to select the most appropriate meme to fill in a blank within a contextual sentence. While the evaluated models were able to provide correct answers, their performance remains noticeably below human levels. We have made CHIME public and hope it will facilitate future research on computational meme understanding. 

---
# ThinkBrake: Mitigating Overthinking in Tool Reasoning 

**Authors**: Minjae Oh, Sangjun Song, Seungkyu Lee, Sungmin Jo, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2510.00546)  

**Abstract**: Small reasoning models (SRMs) often overthink during tool use: they reach a correct tool-argument configuration, then continue reasoning and overwrite it with an incorrect final call. We diagnose overthinking via oracle rollouts that inject </think> at sentence boundaries. On the Berkeley Function Calling Leaderboard (BFCL), this oracle termination lifts average accuracy from 85.8\% to 94.2\% while reducing tokens by 80-94\%, revealing substantial recoverable headroom and potential redundant reasoning. While prior work on concise reasoning has largely targeted mathematics, tool reasoning remains underexplored. We adapt various early-termination baselines to tool use and introduce ThinkBrake, a training-free decoding heuristic. ThinkBrake monitors the log-probability margin between </think> and the current top token at sentence boundaries and triggers termination when this margin becomes small. Across BFCL's single turn, non-live and live splits, ThinkBrake preserves or improves accuracy while reducing tokens up to 25\%, outperforming various baselines. 

---
# GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness 

**Authors**: Kung-Hsiang Huang, Haoyi Qiu, Yutong Dai, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00536)  

**Abstract**: Graphical user interface (GUI) agents built on vision-language models have emerged as a promising approach to automate human-computer workflows. However, they also face the inefficiency challenge as they process long sequences of high-resolution screenshots and solving long-horizon tasks, making inference slow, costly and memory-bound. While key-value (KV) caching can mitigate this, storing the full cache is prohibitive for image-heavy contexts. Existing cache-compression methods are sub-optimal as they do not account for the spatial and temporal redundancy of GUIs. In this work, we first analyze attention patterns in GUI agent workloads and find that, unlike in natural images, attention sparsity is uniformly high across all transformer layers. This insight motivates a simple uniform budget allocation strategy, which we show empirically outperforms more complex layer-varying schemes. Building on this, we introduce GUI-KV, a plug-and-play KV cache compression method for GUI agents that requires no retraining. GUI-KV combines two novel techniques: (i) spatial saliency guidance, which augments attention scores with the L2 norm of hidden states to better preserve semantically important visual tokens, and (ii) temporal redundancy scoring, which projects previous frames' keys onto the current frame's key subspace to preferentially prune redundant history. Across standard GUI agent benchmarks and models, GUI-KV outperforms competitive KV compression baselines, closely matching full-cache accuracy at modest budgets. Notably, in a 5-screenshot setting on the AgentNetBench benchmark, GUI-KV reduces decoding FLOPs by 38.9% while increasing step accuracy by 4.1% over the full-cache baseline. These results demonstrate that exploiting GUI-specific redundancies enables efficient and reliable agent performance. 

---
# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum 

**Authors**: Gaotang Li, Ruizhong Qiu, Xiusi Chen, Heng Ji, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00526)  

**Abstract**: Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL). While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. To this end, we study a general family of probability-based objectives and characterize their effectiveness under different conditions. Through comprehensive experiments and extensive ablation studies across 7 model backbones, 14 benchmarks, and 3 domains, we uncover a critical dimension that governs objective behavior: the model-capability continuum. Near the model-strong end, prior-leaning objectives that downweight low-probability tokens (e.g., $-p$, $-p^{10}$, thresholded variants) consistently outperform NLL; toward the model-weak end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability. Our code is available at this https URL. 

---
# EuroSpeech: A Multilingual Speech Corpus 

**Authors**: Samuel Pfisterer, Florian Grötschla, Luca A. Lanzendörfer, Florian Yan, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2510.00514)  

**Abstract**: Recent progress in speech processing has highlighted that high-quality performance across languages requires substantial training data for each individual language. While existing multilingual datasets cover many languages, they often contain insufficient data for most languages. Thus, trained models perform poorly on the majority of the supported languages. Our work addresses this challenge by introducing a scalable pipeline for constructing speech datasets from parliamentary recordings. The proposed pipeline includes robust components for media retrieval and a two-stage alignment algorithm designed to handle non-verbatim transcripts and long-form audio. Applying this pipeline to recordings from 22 European parliaments, we extract over 61k hours of aligned speech segments, achieving substantial per-language coverage with 19 languages exceeding 1k hours and 22 languages exceeding 500 hours of high-quality speech data. We obtain an average 41.8\% reduction in word error rates over baselines when finetuning an existing ASR model on our dataset, demonstrating the usefulness of our approach. 

---
# JoyAgent-JDGenie: Technical Report on the GAIA 

**Authors**: Jiarun Liu, Shiyue Xu, Shangkun Liu, Yang Li, Wen Liu, Min Liu, Xiaoqing Zhou, Hanmin Wang, Shilin Jia, zhen Wang, Shaohua Tian, Hanhao Li, Junbo Zhang, Yongli Yu, Peng Cao, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00510)  

**Abstract**: Large Language Models are increasingly deployed as autonomous agents for complex real-world tasks, yet existing systems often focus on isolated improvements without a unifying design for robustness and adaptability. We propose a generalist agent architecture that integrates three core components: a collective multi-agent framework combining planning and execution agents with critic model voting, a hierarchical memory system spanning working, semantic, and procedural layers, and a refined tool suite for search, code execution, and multimodal parsing. Evaluated on a comprehensive benchmark, our framework consistently outperforms open-source baselines and approaches the performance of proprietary systems. These results demonstrate the importance of system-level integration and highlight a path toward scalable, resilient, and adaptive AI assistants capable of operating across diverse domains and tasks. 

---
# Copy-Paste to Mitigate Large Language Model Hallucinations 

**Authors**: Yongchao Long, Xian Wu, Yingying Zhang, Xianbin Wen, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00508)  

**Abstract**: While Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to generate contextually grounded responses, contextual faithfulness remains challenging as LLMs may not consistently trust provided context, leading to hallucinations that undermine reliability. We observe an inverse correlation between response copying degree and context-unfaithful hallucinations on RAGTruth, suggesting that higher copying degrees reduce hallucinations by fostering genuine contextual belief. We propose CopyPasteLLM, obtained through two-stage high-copying response preference training. We design three prompting methods to enhance copying degree, demonstrating that high-copying responses achieve superior contextual faithfulness and hallucination control. These approaches enable a fully automated pipeline that transforms generated responses into high-copying preference data for training CopyPasteLLM. On FaithEval, ConFiQA and PubMedQA, CopyPasteLLM achieves best performance in both counterfactual and original contexts, remarkably with 12.2% to 24.5% accuracy improvements on FaithEval over the best baseline, while requiring only 365 training samples -- 1/50th of baseline data. To elucidate CopyPasteLLM's effectiveness, we propose the Context-Parameter Copying Capturing algorithm. Interestingly, this reveals that CopyPasteLLM recalibrates reliance on internal parametric knowledge rather than external knowledge during generation. All codes are available at this https URL 

---
# Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs 

**Authors**: Yurun Chen, Xavier Hu, Yuhan Liu, Ziqi Wang, Zeyi Liao, Lin Chen, Feng Wei, Yuxi Qian, Bo Zheng, Keting Yin, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00507)  

**Abstract**: As multimodal LLM-driven agents continue to advance in autonomy and generalization, evaluation based on static datasets can no longer adequately assess their true capabilities in dynamic environments and diverse tasks. Existing LLM-based synthetic data methods are largely designed for LLM training and evaluation, and thus cannot be directly applied to agent tasks that require tool use and interactive capabilities. While recent studies have explored automatic agent task generation with LLMs, most efforts remain limited to text or image analysis, without systematically modeling multi-step interactions in web environments. To address these challenges, we propose Graph2Eval, a knowledge graph-based framework that automatically generates both multimodal document comprehension tasks and web interaction tasks, enabling comprehensive evaluation of agents' reasoning, collaboration, and interactive capabilities. In our approach, knowledge graphs constructed from multi-source external data serve as the task space, where we translate semantic relations into structured multimodal tasks using subgraph sampling, task templates, and meta-paths. A multi-stage filtering pipeline based on node reachability, LLM scoring, and similarity analysis is applied to guarantee the quality and executability of the generated tasks. Furthermore, Graph2Eval supports end-to-end evaluation of multiple agent types (Single-Agent, Multi-Agent, Web Agent) and measures reasoning, collaboration, and interaction capabilities. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document comprehension and web interaction scenarios. Experiments show that Graph2Eval efficiently generates tasks that differentiate agent and model performance, revealing gaps in reasoning, collaboration, and web interaction across different settings and offering a new perspective for agent evaluation. 

---
# MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance 

**Authors**: Xingjian Zhao, Zhe Xu, Luozhijie Jin, Yang Wang, Hanfu Chen, Yaozhou Jiang, Ke Chen, Ruixiao Li, Mingshu Chen, Ruiming Wang, Wenbo Zhang, Yiyang Zhang, Donghua Yu, Yang Gao, Xiaogui Yang, Yitian Gong, Yuanfan Xu, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Yaqian Zhou, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00499)  

**Abstract**: Spoken dialogue systems often rely on cascaded pipelines that transcribe, process, and resynthesize speech. While effective, this design discards paralinguistic cues and limits expressivity. Recent end-to-end methods reduce latency and better preserve these cues, yet still rely on text intermediates, creating a fundamental bottleneck. We present MOSS-Speech, a true speech-to-speech large language model that directly understands and generates speech without relying on text guidance. Our approach combines a modality-based layer-splitting architecture with a frozen pre-training strategy, preserving the reasoning and knowledge of pretrained text LLMs while adding native speech capabilities. Experiments show that our model achieves state-of-the-art results in spoken question answering and delivers comparable speech-to-speech performance relative to existing text-guided systems, while still maintaining competitive text performance. By narrowing the gap between text-guided and direct speech generation, our work establishes a new paradigm for expressive and efficient end-to-end speech interaction. 

---
# Agent-ScanKit: Unraveling Memory and Reasoning of Multimodal Agents via Sensitivity Perturbations 

**Authors**: Pengzhou Cheng, Lingzhong Dong, Zeng Wu, Zongru Wu, Xiangru Tang, Chengwei Qin, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00496)  

**Abstract**: Although numerous strategies have recently been proposed to enhance the autonomous interaction capabilities of multimodal agents in graphical user interface (GUI), their reliability remains limited when faced with complex or out-of-domain tasks. This raises a fundamental question: Are existing multimodal agents reasoning spuriously? In this paper, we propose \textbf{Agent-ScanKit}, a systematic probing framework to unravel the memory and reasoning capabilities of multimodal agents under controlled perturbations. Specifically, we introduce three orthogonal probing paradigms: visual-guided, text-guided, and structure-guided, each designed to quantify the contributions of memorization and reasoning without requiring access to model internals. In five publicly available GUI benchmarks involving 18 multimodal agents, the results demonstrate that mechanical memorization often outweighs systematic reasoning. Most of the models function predominantly as retrievers of training-aligned knowledge, exhibiting limited generalization. Our findings underscore the necessity of robust reasoning modeling for multimodal agents in real-world scenarios, offering valuable insights toward the development of reliable multimodal agents. 

---
# Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains 

**Authors**: Yawen Xue, Masaya Tsunokake, Yuta Koreeda, Ekant Muljibhai Amin, Takashi Sumiyoshi, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.00482)  

**Abstract**: Agentic large language models (LLMs) have become prominent for autonomously interacting with external environments and performing multi-step reasoning tasks. Most approaches leverage these capabilities via in-context learning with few-shot prompts, but this often results in lengthy inputs and higher computational costs. Agent fine-tuning offers an alternative by enabling LLMs to internalize procedural reasoning and domain-specific knowledge through training on relevant data and demonstration trajectories. While prior studies have focused on general domains, their effectiveness in specialized technical microdomains remains unclear. This paper explores agent fine-tuning for domain adaptation within Hitachi's JP1 middleware, a microdomain for specialized IT operations. We fine-tuned LLMs using JP1-specific datasets derived from domain manuals and distilled reasoning trajectories generated by LLMs themselves, enhancing decision making accuracy and search efficiency. During inference, we used an agentic prompt with retrieval-augmented generation and introduced a context-answer extractor to improve information relevance. On JP1 certification exam questions, our method achieved a 14% performance improvement over the base model, demonstrating the potential of agent fine-tuning for domain-specific reasoning in complex microdomains. 

---
# Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews 

**Authors**: Koki Ryu, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2510.00449)  

**Abstract**: Personalizing the outputs of large language models (LLMs) to align with individual user preferences is an active research area. However, previous studies have mainly focused on classification or ranking tasks and have not considered Likert-scale rating prediction, a regression task that requires both language and mathematical reasoning to be solved effectively. This task has significant industrial applications, but the utilization of LLMs remains underexplored, particularly regarding the capabilities of off-the-shelf LLMs. This study investigates the performance of off-the-shelf LLMs on rating prediction, providing different in-context information. Through comprehensive experiments with eight models across three datasets, we demonstrate that user-written reviews significantly improve the rating prediction performance of LLMs. This result is comparable to traditional methods like matrix factorization, highlighting the potential of LLMs as a promising solution for the cold-start problem. We also find that the reviews for concrete items are more effective than general preference descriptions that are not based on any specific item. Furthermore, we discover that prompting LLMs to first generate a hypothetical review enhances the rating prediction performance. Our code is available at this https URL. 

---
# LongCodeZip: Compress Long Context for Code Language Models 

**Authors**: Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00446)  

**Abstract**: Code generation under long contexts is becoming increasingly critical as Large Language Models (LLMs) are required to reason over extensive information in the codebase. While recent advances enable code LLMs to process long inputs, high API costs and generation latency remain substantial bottlenecks. Existing context pruning techniques, such as LLMLingua, achieve promising results for general text but overlook code-specific structures and dependencies, leading to suboptimal performance in programming tasks. In this paper, we propose LongCodeZip, a novel plug-and-play code compression framework designed specifically for code LLMs. LongCodeZip employs a dual-stage strategy: (1) coarse-grained compression, which identifies and ranks function-level chunks using conditional perplexity with respect to the instruction, retaining only the most relevant functions; and (2) fine-grained compression, which segments retained functions into blocks based on perplexity and selects an optimal subset under an adaptive token budget to maximize relevance. Evaluations across multiple tasks, including code completion, summarization, and question answering, show that LongCodeZip consistently outperforms baseline methods, achieving up to a 5.6x compression ratio without degrading task performance. By effectively reducing context size while preserving essential information, LongCodeZip enables LLMs to better scale to real-world, large-scale code scenarios, advancing the efficiency and capability of code intelligence applications. 

---
# TokMem: Tokenized Procedural Memory for Large Language Models 

**Authors**: Zijun Wu, Yongchang Hao, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00444)  

**Abstract**: Large language models rely heavily on prompts to specify tasks, recall knowledge and guide reasoning. However, this reliance is inefficient as prompts must be re-read at each step, scale poorly across tasks, and lack mechanisms for modular reuse. We introduce TokMem, a tokenized procedural memory that stores recurring procedures as compact, trainable embeddings. Each memory token encodes both an address to a procedure and a control signal that steers generation, enabling targeted behavior with constant-size overhead. To support continual adaptation, TokMem keeps the backbone model frozen, allowing new procedures to be added without interfering with existing ones. We evaluate TokMem on 1,000 tasks for atomic recall, and on function-calling tasks for compositional recall, where it consistently outperforms retrieval-augmented generation while avoiding repeated context overhead, and fine-tuning with far fewer parameters. These results establish TokMem as a scalable and modular alternative to prompt engineering and fine-tuning, offering an explicit procedural memory for LLMs. 

---
# CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage 

**Authors**: Bowen Wei, Yuan Shen Tay, Howard Liu, Jinhao Pan, Kun Luo, Ziwei Zhu, Chris Jordan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00311)  

**Abstract**: Security Operations Centers (SOCs) are overwhelmed by tens of thousands of daily alerts, with only a small fraction corresponding to genuine attacks. This overload creates alert fatigue, leading to overlooked threats and analyst burnout. Classical detection pipelines are brittle and context-poor, while recent LLM-based approaches typically rely on a single model to interpret logs, retrieve context, and adjudicate alerts end-to-end -- an approach that struggles with noisy enterprise data and offers limited transparency. We propose CORTEX, a multi-agent LLM architecture for high-stakes alert triage in which specialized agents collaborate over real evidence: a behavior-analysis agent inspects activity sequences, evidence-gathering agents query external systems, and a reasoning agent synthesizes findings into an auditable decision. To support training and evaluation, we release a dataset of fine-grained SOC investigations from production environments, capturing step-by-step analyst actions and linked tool outputs. Across diverse enterprise scenarios, CORTEX substantially reduces false positives and improves investigation quality over state-of-the-art single-agent LLMs. 

---
# o-MEGA: Optimized Methods for Explanation Generation and Analysis 

**Authors**: Ľuboš Kriš, Jaroslav Kopčan, Qiwei Peng, Andrej Ridzik, Marcel Veselý, Martin Tamajka  

**Link**: [PDF](https://arxiv.org/pdf/2510.00288)  

**Abstract**: The proliferation of transformer-based language models has revolutionized NLP domain while simultaneously introduced significant challenges regarding model transparency and trustworthiness. The complexity of achieving explainable systems in this domain is evidenced by the extensive array of explanation methods and evaluation metrics developed by researchers. To address the challenge of selecting optimal explainability approaches, we present \textbf{\texttt{o-mega}}, a hyperparameter optimization tool designed to automatically identify the most effective explainable AI methods and their configurations within the semantic matching domain. We evaluate o-mega on a post-claim matching pipeline using a curated dataset of social media posts paired with refuting claims. Our tool systematically explores different explainable methods and their hyperparameters, demonstrating improved transparency in automated fact-checking systems. As a result, such automated optimization of explanation methods can significantly enhance the interpretability of claim-matching models in critical applications such as misinformation detection, contributing to more trustworthy and transparent AI systems. 

---
# ReEvalMed: Rethinking Medical Report Evaluation by Aligning Metrics with Real-World Clinical Judgment 

**Authors**: Ruochen Li, Jun Li, Bailiang Jian, Kun Yuan, Youxiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00280)  

**Abstract**: Automatically generated radiology reports often receive high scores from existing evaluation metrics but fail to earn clinicians' trust. This gap reveals fundamental flaws in how current metrics assess the quality of generated reports. We rethink the design and evaluation of these metrics and propose a clinically grounded Meta-Evaluation framework. We define clinically grounded criteria spanning clinical alignment and key metric capabilities, including discrimination, robustness, and monotonicity. Using a fine-grained dataset of ground truth and rewritten report pairs annotated with error types, clinical significance labels, and explanations, we systematically evaluate existing metrics and reveal their limitations in interpreting clinical semantics, such as failing to distinguish clinically significant errors, over-penalizing harmless variations, and lacking consistency across error severity levels. Our framework offers guidance for building more clinically reliable evaluation methods. 

---
# SafePassage: High-Fidelity Information Extraction with Black Box LLMs 

**Authors**: Joe Barrow, Raj Patel, Misha Kharkovski, Ben Davies, Ryan Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2510.00276)  

**Abstract**: Black box large language models (LLMs) make information extraction (IE) easy to configure, but hard to trust. Unlike traditional information extraction pipelines, the information "extracted" is not guaranteed to be grounded in the document. To prevent this, this paper introduces the notion of a "safe passage": context generated by the LLM that is both grounded in the document and consistent with the extracted information. This is operationalized via a three-step pipeline, SafePassage, which consists of: (1) an LLM extractor that generates structured entities and their contexts from a document, (2) a string-based global aligner, and (3) a scoring model. Results show that using these three parts in conjunction reduces hallucinations by up to 85% on information extraction tasks with minimal risk of flagging non-hallucinations. High agreement between the SafePassage pipeline and human judgments of extraction quality mean that the pipeline can be dually used to evaluate LLMs. Surprisingly, results also show that using a transformer encoder fine-tuned on a small number of task-specific examples can outperform an LLM scoring model at flagging unsafe passages. These annotations can be collected in as little as 1-2 hours. 

---
# Efficient Layer-wise LLM Fine-tuning for Revision Intention Prediction 

**Authors**: Zhexiong Liu, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00268)  

**Abstract**: Large Language Models (LLMs) have shown extraordinary success across various text generation tasks; however, their potential for simple yet essential text classification remains underexplored, as LLM pre-training tends to emphasize generation over classification. While LLMs with instruction tuning can transform classification into a generation task, they often struggle to categorize nuanced texts. One such example is text revision, which involves nuanced edits between pairs of texts. Although simply fine-tuning LLMs for revision classification seems plausible, it requires a large amount of revision annotations, which are exceptionally expensive and scarce in the community. To address this issue, we introduce a plug-and-play layer-wise parameter-efficient fine-tuning (PEFT) framework, i.e., IR-Tuning, which fine-tunes a subset of important LLM layers that are dynamically selected based on their gradient norm distribution, while freezing those of redundant layers. Extensive experiments suggest that IR-Tuning surpasses several layer-wise PEFT baselines over diverse text revisions, while achieving fast convergence, low GPU memory consumption, and effectiveness on small revision corpora. 

---
# Judging with Confidence: Calibrating Autoraters to Preference Distributions 

**Authors**: Zhuohang Li, Xiaowei Li, Chengyu Huang, Guowang Li, Katayoon Goshvadi, Bo Dai, Dale Schuurmans, Paul Zhou, Hamid Palangi, Yiwen Song, Palash Goyal, Murat Kantarcioglu, Bradley A. Malin, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2510.00263)  

**Abstract**: The alignment of large language models (LLMs) with human values increasingly relies on using other LLMs as automated judges, or ``autoraters''. However, their reliability is limited by a foundational issue: they are trained on discrete preference labels, forcing a single ground truth onto tasks that are often subjective, ambiguous, or nuanced. We argue that a reliable autorater must learn to model the full distribution of preferences defined by a target population. In this paper, we propose a general framework for calibrating probabilistic autoraters to any given preference distribution. We formalize the problem and present two learning methods tailored to different data conditions: 1) a direct supervised fine-tuning for dense, probabilistic labels, and 2) a reinforcement learning approach for sparse, binary labels. Our empirical results show that finetuning autoraters with a distribution-matching objective leads to verbalized probability predictions that are better aligned with the target preference distribution, with improved calibration and significantly lower positional bias, all while preserving performance on objective tasks. 

---
# Retrieval-Augmented Generation for Electrocardiogram-Language Models 

**Authors**: Xiaoyu Song, William Han, Tony Chen, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00261)  

**Abstract**: Interest in generative Electrocardiogram-Language Models (ELMs) is growing, as they can produce textual responses conditioned on ECG signals and textual queries. Unlike traditional classifiers that output label probabilities, ELMs are more versatile, supporting domain-specific tasks (e.g., waveform analysis, diagnosis, prognosis) as well as general tasks (e.g., open-ended questions, dialogue). Retrieval-Augmented Generation (RAG), widely used in Large Language Models (LLMs) to ground LLM outputs in retrieved knowledge, helps reduce hallucinations and improve natural language generation (NLG). However, despite its promise, no open-source implementation or systematic study of RAG pipeline design for ELMs currently exists. To address this gap, we present the first open-source RAG pipeline for ELMs, along with baselines and ablation studies for NLG. Experiments on three public datasets show that ELMs with RAG consistently improves performance over non-RAG baselines and highlights key ELM design considerations. Our code is available at: this https URL. 

---
# TASER: Translation Assessment via Systematic Evaluation and Reasoning 

**Authors**: Monishwaran Maheswaran, Marco Carini, Christian Federmann, Tony Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00255)  

**Abstract**: We introduce TASER (Translation Assessment via Systematic Evaluation and Reasoning), a metric that uses Large Reasoning Models (LRMs) for automated translation quality assessment. TASER harnesses the explicit reasoning capabilities of LRMs to conduct systematic, step-by-step evaluation of translation quality. We evaluate TASER on the WMT24 Metrics Shared Task across both reference-based and reference-free scenarios, demonstrating state-of-the-art performance. In system-level evaluation, TASER achieves the highest soft pairwise accuracy in both reference-based and reference-free settings, outperforming all existing metrics. At the segment level, TASER maintains competitive performance with our reference-free variant ranking as the top-performing metric among all reference-free approaches. Our experiments reveal that structured prompting templates yield superior results with LRMs compared to the open-ended approaches that proved optimal for traditional LLMs. We evaluate o3, a large reasoning model from OpenAI, with varying reasoning efforts, providing insights into the relationship between reasoning depth and evaluation quality. The explicit reasoning process in LRMs offers interpretability and visibility, addressing a key limitation of existing automated metrics. Our results demonstrate that Large Reasoning Models show a measurable advancement in translation quality assessment, combining improved accuracy with transparent evaluation across diverse language pairs. 

---
# BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses 

**Authors**: Xin Xu, Xunzhi He, Churan Zhi, Ruizhe Chen, Julian McAuley, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2510.00232)  

**Abstract**: Existing studies on bias mitigation methods for large language models (LLMs) use diverse baselines and metrics to evaluate debiasing performance, leading to inconsistent comparisons among them. Moreover, their evaluations are mostly based on the comparison between LLMs' probabilities of biased and unbiased contexts, which ignores the gap between such evaluations and real-world use cases where users interact with LLMs by reading model responses and expect fair and safe outputs rather than LLMs' probabilities. To enable consistent evaluation across debiasing methods and bridge this gap, we introduce BiasFreeBench, an empirical benchmark that comprehensively compares eight mainstream bias mitigation techniques (covering four prompting-based and four training-based methods) on two test scenarios (multi-choice QA and open-ended multi-turn QA) by reorganizing existing datasets into a unified query-response setting. We further introduce a response-level metric, Bias-Free Score, to measure the extent to which LLM responses are fair, safe, and anti-stereotypical. Debiasing performances are systematically compared and analyzed across key dimensions: the prompting vs. training paradigm, model size, and generalization of different training strategies to unseen bias types. We will publicly release our benchmark, aiming to establish a unified testbed for bias mitigation research. 

---
# Personalized Reasoning: Just-In-Time Personalization and Why LLMs Fail At It 

**Authors**: Shuyue Stella Li, Avinandan Bose, Faeze Brahman, Simon Shaolei Du, Pang Wei Koh, Maryam Fazel, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00177)  

**Abstract**: Current large language model (LLM) development treats task-solving and preference alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user's needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to identify what they don't know about user preferences, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly -- a complicated chain of cognitive processes which we term personalized reasoning. We introduce PREFDISCO, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse preferences. Our framework creates scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs effectively. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PREFDISCO establishes personalized reasoning as a measurable research frontier and reveals fundamental limitations in current LLMs' interactive capabilities, providing a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical. 

---
# PrimeX: A Dataset of Worldview, Opinion, and Explanation 

**Authors**: Rik Koncel-Kedziorski, Brihi Joshi, Tim Paek  

**Link**: [PDF](https://arxiv.org/pdf/2510.00174)  

**Abstract**: As the adoption of language models advances, so does the need to better represent individual users to the model. Are there aspects of an individual's belief system that a language model can utilize for improved alignment? Following prior research, we investigate this question in the domain of opinion prediction by developing PrimeX, a dataset of public opinion survey data from 858 US residents with two additional sources of belief information: written explanations from the respondents for why they hold specific opinions, and the Primal World Belief survey for assessing respondent worldview. We provide an extensive initial analysis of our data and show the value of belief explanations and worldview for personalizing language models. Our results demonstrate how the additional belief information in PrimeX can benefit both the NLP and psychological research communities, opening up avenues for further study. 

---
# DRBench: A Realistic Benchmark for Enterprise Deep Research 

**Authors**: Amirhossein Abaskohi, Tianyi Chen, Miguel Muñoz-Mármol, Curtis Fox, Amrutha Varshini Ramesh, Étienne Marcotte, Xing Han Lù, Nicolas Chapados, Spandana Gella, Christopher Pal, Alexandre Drouin, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2510.00172)  

**Abstract**: We introduce DRBench, a benchmark for evaluating AI agents on complex, open-ended deep research tasks in enterprise settings. Unlike prior benchmarks that focus on simple questions or web-only queries, DRBench evaluates agents on multi-step queries (for example, ``What changes should we make to our product roadmap to ensure compliance with this standard?") that require identifying supporting facts from both the public web and private company knowledge base. Each task is grounded in realistic user personas and enterprise context, spanning a heterogeneous search space that includes productivity software, cloud file systems, emails, chat conversations, and the open web. Tasks are generated through a carefully designed synthesis pipeline with human-in-the-loop verification, and agents are evaluated on their ability to recall relevant insights, maintain factual accuracy, and produce coherent, well-structured reports. We release 15 deep research tasks across 10 domains, such as Sales, Cybersecurity, and Compliance. We demonstrate the effectiveness of DRBench by evaluating diverse DR agents across open- and closed-source models (such as GPT, Llama, and Qwen) and DR strategies, highlighting their strengths, weaknesses, and the critical path for advancing enterprise deep research. Code is available at this https URL. 

---
# TAMA: Tool-Augmented Multimodal Agent for Procedural Activity Understanding 

**Authors**: Kimihiro Hasegawa, Wiradee Imrattanatrai, Masaki Asada, Ken Fukuda, Teruko Mitamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.00161)  

**Abstract**: Procedural activity assistants potentially support humans in a variety of settings, from our daily lives, e.g., cooking or assembling flat-pack furniture, to professional situations, e.g., manufacturing or biological experiments. Despite its potential use cases, the system development tailored for such an assistant is still underexplored. In this paper, we propose a novel framework, called TAMA, a Tool-Augmented Multimodal Agent, for procedural activity understanding. TAMA enables interleaved multimodal reasoning by making use of multimedia-returning tools in a training-free setting. Our experimental result on the multimodal procedural QA dataset, ProMQA-Assembly, shows that our approach can improve the performance of vision-language models, especially GPT-5 and MiMo-VL. Furthermore, our ablation studies provide empirical support for the effectiveness of two features that characterize our framework, multimedia-returning tools and agentic flexible tool selection. We believe our proposed framework and experimental results facilitate the thinking with images paradigm for video and multimodal tasks, let alone the development of procedural activity assistants. 

---
# Direct Token Optimization: A Self-contained Approach to Large Language Model Unlearning 

**Authors**: Hong kyu Lee, Ruixuan Liu, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00125)  

**Abstract**: Machine unlearning is an emerging technique that removes the influence of a subset of training data (forget set) from a model without full retraining, with applications including privacy protection, content moderation, and model correction. The key challenge lies in ensuring that the model completely forgets the knowledge of the forget set without compromising its overall utility. Existing unlearning methods for large language models (LLMs) often utilize auxiliary language models, retain datasets, or even commercial AI services for effective unlearning and maintaining the model utility. However, dependence on these external resources is often impractical and could potentially introduce additional privacy risks. In this work, we propose direct token optimization (DTO), a novel self-contained unlearning approach for LLMs that directly optimizes the token level objectives and eliminates the need for external resources. Given a sequence to unlearn, we identify two categories of tokens: target tokens, which capture critical knowledge for unlearning, and the remaining non-target tokens, which are crucial for maintaining the model utility. The former are used to optimize the unlearning objective, while the latter serve to preserve the model's performance. The experimental results show that the proposed DTO achieves up to 16.8$\times$ improvement in forget quality on several benchmark datasets than the latest baselines while maintaining a comparable level of model utility. 

---
# BroRL: Scaling Reinforcement Learning via Broadened Exploration 

**Authors**: Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Harchaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun Yang, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.01180)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key ingredient for unlocking complex reasoning capabilities in large language models. Recent work ProRL has shown promise in scaling RL by increasing the number of training steps. However, performance plateaus after thousands of steps, with clear diminishing returns from allocating more computation to additional training. In this work, we investigate a complementary paradigm for scaling RL, BroR-Lincreasing the number of rollouts per example to hundreds to exhaustively Broaden exploration, which yields continuous performance gains beyond the saturation point observed in ProRL when scaling the number of training steps. Our approach is motivated by a mass balance equation analysis allowing us to characterize the rate of change in probability mass for correct and incorrect tokens during the reinforcement process. We show that under a one-step RL assumption, sampled rollout tokens always contribute to correct-mass expansion, while unsampled tokens outside rollouts may lead to gains or losses depending on their distribution and the net reward balance. Importantly, as the number of rollouts per example N increases, the effect of unsampled terms diminishes, ensuring overall correct-mass expansion. To validate our theoretical analysis, we conduct simulations under more relaxed conditions and find that a sufficiently large rollout size N-corresponding to ample exploration-guarantees an increase in the probability mass of all correct tokens. Empirically, BroRL revives models saturated after 3K ProRL training steps and demonstrates robust, continuous improvement, achieving state-of-the-art results for the 1.5B model across diverse benchmarks. 

---
# TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments 

**Authors**: Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda  

**Link**: [PDF](https://arxiv.org/pdf/2510.01179)  

**Abstract**: Large Language Model (LLM) agents are rapidly emerging as powerful systems for automating tasks across domains. Yet progress in the open-source community is constrained by the lack of high quality permissively licensed tool-agentic training data. Existing datasets are often limited in diversity, realism, and complexity, particularly regarding multi-tool and multi-turn interactions. To address this gap, we introduce Toucan, the largest publicly available tool-agentic dataset to date, containing 1.5 million trajectories synthesized from nearly 500 real-world Model Context Protocols (MCPs). Unlike prior work, Toucan leverages authentic MCP environments to generate diverse, realistic, and challenging tasks with trajectories involving real tool execution. Our pipeline first produces a broad spectrum of tool-use queries using five distinct models, applies model-based quality filtering, and then generates agentic trajectories with three teacher models using two agentic frameworks. Rigorous rule-based and model-based validation ensures high-quality outputs. We also introduce three extension mechanisms to further diversify tasks and simulate multi-turn conversations. Models fine-tuned on Toucan outperform larger closed-source counterparts on the BFCL V3 benchmark and push the Pareto frontier forward on MCP-Universe Bench. 

---
# Code2Video: A Code-centric Paradigm for Educational Video Generation 

**Authors**: Yanzhe Chen, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.01174)  

**Abstract**: While recent generative models advance pixel-space video synthesis, they remain limited in producing professional educational videos, which demand disciplinary knowledge, precise visual structures, and coherent transitions, limiting their applicability in educational scenarios. Intuitively, such requirements are better addressed through the manipulation of a renderable environment, which can be explicitly controlled via logical commands (e.g., code). In this work, we propose Code2Video, a code-centric agent framework for generating educational videos via executable Python code. The framework comprises three collaborative agents: (i) Planner, which structures lecture content into temporally coherent flows and prepares corresponding visual assets; (ii) Coder, which converts structured instructions into executable Python codes while incorporating scope-guided auto-fix to enhance efficiency; and (iii) Critic, which leverages vision-language models (VLM) with visual anchor prompts to refine spatial layout and ensure clarity. To support systematic evaluation, we build MMMC, a benchmark of professionally produced, discipline-specific educational videos. We evaluate MMMC across diverse dimensions, including VLM-as-a-Judge aesthetic scores, code efficiency, and particularly, TeachQuiz, a novel end-to-end metric that quantifies how well a VLM, after unlearning, can recover knowledge by watching the generated videos. Our results demonstrate the potential of Code2Video as a scalable, interpretable, and controllable approach, achieving 40% improvement over direct code generation and producing videos comparable to human-crafted tutorials. The code and datasets are available at this https URL. 

---
# Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards 

**Authors**: Yiran Shen, Yu Xia, Jonathan Chang, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01167)  

**Abstract**: Aligning large language models to human preferences is inherently multidimensional, yet most pipelines collapse heterogeneous signals into a single optimizeable objective. We seek to answer what it would take to simultaneously align a model across various domains spanning those with: verifiable rewards (mathematical accuracy), non-verifiable subjective preferences (human values), and complex interactive scenarios (multi-turn AI tutoring dialogues). Such multi-objective reinforcement learning setups are often plagued by the individual objectives being at odds with each other, resulting in inefficient training and little user control during inference. We propose a unified framework that: (i) standardizes {process reward model} (PRM) training across both verifiable and non-verifiable settings to better supervise models' chain-of-thought reasoning; (ii) performs {multi-objective alignment} by training the LLM with our $\textbf{M}$ulti-$\textbf{A}$ction-$\textbf{H}$ead $\textbf{DPO}$ (MAH-DPO) and a vectorized reward where the dimensions of the vector correspond to the various objectives instead of a single scalar; and (iii) demonstrates how such a system provides fine-grained inference-time user control. Experiments across math reasoning, value alignment, and multi-turn dialogue show that our framework improves performance across multiple objectives simultaneously, while minimizing cross-objective trade-offs and enabling flexible inference time user control. The code can be found at this https URL. 

---
# Prompt Curriculum Learning for Efficient LLM Post-Training 

**Authors**: Zhaolin Gao, Joongwon Kim, Wen Sun, Thorsten Joachims, Sid Wang, Richard Yuanzhe Pang, Liang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.01135)  

**Abstract**: We introduce Prompt Curriculum Learning (PCL), a lightweight reinforcement learning (RL) algorithm that selects intermediate-difficulty prompts using a learned value model to post-train language models. Since post-training LLMs via RL remains sensitive to batching and prompt selection strategies, we first conduct a series of systematic experiments where we (1) determine the optimal training batch size that balances generation efficiency and gradient quality and (2) establish the importance of focusing on prompts of intermediate difficulty for the policy. We build upon these results to design PCL, which identifies prompts of intermediate difficulty for the current policy in an on-policy manner by using a value model that is concurrently updated based on the current policy. By focusing on informative prompts that yield high effective ratios, PCL achieves either the highest performance or requires significantly less time to reach comparable performance to its counterparts. Compared to rollout-based filtering methods, PCL avoids costly rollouts and achieves $12.1\times$ and $16.9\times$ faster speed on identifying intermediate-difficulty prompts when training on MATH and DeepScaleR, respectively. We further demonstrate that our value model accurately predicts prompt difficulty and allows PCL to focus on progressively more challenging prompts during RL. Our results present a new methodology that delivers improved tradeoff between upper-bound performance and efficiency for reasoning-focused RL. 

---
# A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning 

**Authors**: Ruiyi Wang, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01132)  

**Abstract**: We study what actually works and what doesn't for training large language models as agents via multi-turn reinforcement learning. Despite rapid progress, existing frameworks and definitions are fragmented, and there is no systematic formulation or analysis of which design choices matter across tasks. We address this gap by first breaking down the design space into three inter-related pillars -- environment, reward, and policy -- and empirically derive a recipe for training LLM agents in situated textual domains. In particular, we test TextWorld and ALFWorld, popular domains for testing situated embodied reasoning, as well as SWE-Gym for more software engineering style tasks. (i) For the environment, we analyze the impacts of task complexity in terms of sizes of the state and action spaces as well as optimal solution length, finding that even simple environments within a domain can provide signal on how well an agent can generalize to more complex tasks. (ii) For the reward, we ablate relative reward sparsity, observing that while dense turn-level rewards accelerate training, performance and stability is highly dependent on the choice of RL algorithm. (iii) And for the agent's policy, we explore the interplay between reward sparsity and biased (PPO, GRPO) and unbiased (RLOO) policy gradient methods in addition to showing how to find the optimal Supervised Fine-tuning (SFT) to RL training ratio given a fixed budget. We distill these findings into a training recipe that guides co-design across the three pillars, facilitating research and practical efforts in multi-turn agentic RL. Code: this https URL 

---
# GEM: A Gym for Agentic LLMs 

**Authors**: Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.01051)  

**Abstract**: The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which -- unlike GRPO -- is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research. 

---
# Authentic Discrete Diffusion Model 

**Authors**: Xiao Li, Jiaqi Zhang, Shuxiang Zhang, Tianshui Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01047)  

**Abstract**: We propose an Authentic Discrete Diffusion (ADD) framework that fundamentally redefines prior pseudo-discrete approaches by preserving core diffusion characteristics directly in the one-hot space through a suite of coordinated mechanisms. Unlike conventional "pseudo" discrete diffusion (PDD) methods, ADD reformulates the diffusion input by directly using float-encoded one-hot class data, without relying on diffusing in the continuous latent spaces or masking policies. At its core, a timestep-conditioned cross-entropy loss is introduced between the diffusion model's outputs and the original one-hot labels. This synergistic design establishes a bridge between discriminative and generative learning. Our experiments demonstrate that ADD not only achieves superior performance on classification tasks compared to the baseline, but also exhibits excellent text generation capabilities on Image captioning. Extensive ablations validate the measurable gains of each component. 

---
# Shape Happens: Automatic Feature Manifold Discovery in LLMs via Supervised Multi-Dimensional Scaling 

**Authors**: Federico Tiblias, Irina Bigoulaeva, Jingcheng Niu, Simone Balloccu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2510.01025)  

**Abstract**: The linear representation hypothesis states that language models (LMs) encode concepts as directions in their latent space, forming organized, multidimensional manifolds. Prior efforts focus on discovering specific geometries for specific features, and thus lack generalization. We introduce Supervised Multi-Dimensional Scaling (SMDS), a model-agnostic method to automatically discover feature manifolds. We apply SMDS to temporal reasoning as a case study, finding that different features form various geometric structures such as circles, lines, and clusters. SMDS reveals many insights on these structures: they consistently reflect the properties of the concepts they represent; are stable across model families and sizes; actively support reasoning in models; and dynamically reshape in response to context changes. Together, our findings shed light on the functional role of feature manifolds, supporting a model of entity-based reasoning in which LMs encode and transform structured representations. 

---
# Improving Code Localization with Repository Memory 

**Authors**: Boshi Wang, Weijian Xu, Yunsheng Li, Mei Gao, Yujia Xie, Huan Sun, Dongdong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01003)  

**Abstract**: Code localization is a fundamental challenge in repository-level software engineering tasks such as bug fixing. While existing methods equip language agents with comprehensive tools/interfaces to fetch information from the repository, they overlook the critical aspect of memory, where each instance is typically handled from scratch assuming no prior repository knowledge. In contrast, human developers naturally build long-term repository memory, such as the functionality of key modules and associations between various bug types and their likely fix locations. In this work, we augment language agents with such memory by leveraging a repository's commit history - a rich yet underutilized resource that chronicles the codebase's evolution. We introduce tools that allow the agent to retrieve from a non-parametric memory encompassing recent historical commits and linked issues, as well as functionality summaries of actively evolving parts of the codebase identified via commit patterns. We demonstrate that augmenting such a memory can significantly improve LocAgent, a state-of-the-art localization framework, on both SWE-bench-verified and the more recent SWE-bench-live benchmarks. Our research contributes towards developing agents that can accumulate and leverage past experience for long-horizon tasks, more closely emulating the expertise of human developers. 

---
# Spiralformer: Low Latency Encoder for Streaming Speech Recognition with Circular Layer Skipping and Early Exiting 

**Authors**: Emiru Tsunoo, Hayato Futami, Yosuke Kashiwagi, Siddhant Arora, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2510.00982)  

**Abstract**: For streaming speech recognition, a Transformer-based encoder has been widely used with block processing. Although many studies addressed improving emission latency of transducers, little work has been explored for improving encoding latency of the block processing. We seek to reduce latency by frequently emitting a chunk with a small shift rather than scarce large-chunk emissions, resulting in higher computational costs. To efficiently compute with the small chunk shift, we propose a new encoder, Spiralformer, tailored for block processing by combining layer dropping and early exiting. We skip layer computation in a cyclic manner and shift the computed layer in each block spirally, which completes computation for all the layers over the block processing. Experimentally, we observed that our method achieved 21.6% reduction in the averaged token emission delay in Librispeech, and 7.0% in CSJ, compared with the baseline with similar computational cost and word error rates. 

---
# It Takes Two: Your GRPO Is Secretly DPO 

**Authors**: Yihong Wu, Liheng Ma, Lei Ding, Muzhi Li, Xinyu Wang, Kejia Chen, Zhan Su, Zhanguang Zhang, Chenyang Huang, Yingxue Zhang, Mark Coates, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.00977)  

**Abstract**: Group Relative Policy Optimization (GRPO) is a prominent reinforcement learning algorithm for post-training Large Language Models (LLMs). It is commonly believed that GRPO necessitates a large group size to ensure stable training via precise statistical estimation, which incurs substantial computational overhead. In this work, we challenge this assumption by reframing GRPO as a form of contrastive learning, which reveals a fundamental connection to Direct Preference Optimization (DPO). Motivated by DPO's empirical success, we investigate the minimal two-rollout case (2-GRPO), a configuration previously deemed infeasible. We provide a rigorous theoretical analysis to validate 2-GRPO and demonstrate empirically that it achieves performance on par with 16-GRPO, despite using only 1/8 of the rollouts and reducing training time by over 70%. 

---
# Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs 

**Authors**: Roksana Goworek, Olivia Macmillan-Scott, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2510.00908)  

**Abstract**: Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. Research in this area has typically framed the task as monolingual retrieval augmented by translation, treating retrieval methods and cross-lingual capabilities in isolation. Both monolingual and cross-lingual retrieval usually follow a pipeline of query expansion, ranking, re-ranking and, increasingly, question answering. Recent advances, however, have shifted from translation-based methods toward embedding-based approaches and leverage multilingual large language models (LLMs), for which aligning representations across languages remains a central challenge. The emergence of cross-lingual embeddings and multilingual LLMs has introduced a new paradigm, offering improved retrieval performance and enabling answer generation. This survey provides a comprehensive overview of developments from early translation-based methods to state-of-the-art embedding-driven and generative techniques. It presents a structured account of core CLIR components, evaluation practices, and available resources. Persistent challenges such as data imbalance and linguistic variation are identified, while promising directions are suggested for advancing equitable and effective cross-lingual information retrieval. By situating CLIR within the broader landscape of information retrieval and multilingual language processing, this work not only reviews current capabilities but also outlines future directions for building retrieval systems that are robust, inclusive, and adaptable. 

---
# The data-quality illusion: Rethinking Classifier-based quality filtering for LLM Pretraining 

**Authors**: Thiziri Nait Saada, Louis Bethune, Michal Klein, David Grangier, Marco Cuturi, Pierre Ablin  

**Link**: [PDF](https://arxiv.org/pdf/2510.00866)  

**Abstract**: Large-scale models are pretrained on massive web-crawled datasets containing documents of mixed quality, making data filtering essential. A popular method is Classifier-based Quality Filtering (CQF), which trains a binary classifier to distinguish between pretraining data and a small, high-quality set. It assigns each pretraining document a quality score defined as the classifier's score and retains only the top-scoring ones. We provide an in-depth analysis of CQF. We show that while CQF improves downstream task performance, it does not necessarily enhance language modeling on the high-quality dataset. We explain this paradox by the fact that CQF implicitly filters the high-quality dataset as well. We further compare the behavior of models trained with CQF to those trained on synthetic data of increasing quality, obtained via random token permutations, and find starkly different trends. Our results challenge the view that CQF captures a meaningful notion of data quality. 

---
# Can World Models Benefit VLMs for World Dynamics? 

**Authors**: Kevin Zhang, Kuangzhi Ge, Xiaowei Chi, Renrui Zhang, Shaojun Shi, Zhen Dong, Sirui Han, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00855)  

**Abstract**: Trained on internet-scale video data, generative world models are increasingly recognized as powerful world simulators that can generate consistent and plausible dynamics over structure, motion, and physics. This raises a natural question: with the advent of strong video foundational models, might they supplant conventional vision encoder paradigms for general-purpose multimodal understanding? While recent studies have begun to explore the potential of world models on common vision tasks, these explorations typically lack a systematic investigation of generic, multimodal tasks. In this work, we strive to investigate the capabilities when world model priors are transferred into Vision-Language Models: we re-purpose a video diffusion model as a generative encoder to perform a single denoising step and treat the resulting latents as a set of visual embedding. We empirically investigate this class of models, which we refer to as World-Language Models (WorldLMs), and we find that generative encoders can capture latents useful for downstream understanding that show distinctions from conventional encoders. Naming our best-performing variant Dynamic Vision Aligner (DyVA), we further discover that this method significantly enhances spatial reasoning abilities and enables single-image models to perform multi-frame reasoning. Through the curation of a suite of visual reasoning tasks, we find DyVA to surpass both open-source and proprietary baselines, achieving state-of-the-art or comparable performance. We attribute these gains to WorldLM's inherited motion-consistency internalization from video pre-training. Finally, we systematically explore extensive model designs to highlight promising directions for future work. We hope our study can pave the way for a new family of VLMs that leverage priors from world models and are on a promising path towards generalist vision learners. 

---
# Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG 

**Authors**: Maxime Méloux, Maxime Peyrard, François Portet  

**Link**: [PDF](https://arxiv.org/pdf/2510.00845)  

**Abstract**: The development of trustworthy artificial intelligence requires moving beyond black-box performance metrics toward an understanding of models' internal computations. Mechanistic Interpretability (MI) aims to meet this need by identifying the algorithmic mechanisms underlying model behaviors. Yet, the scientific rigor of MI critically depends on the reliability of its findings. In this work, we argue that interpretability methods, such as circuit discovery, should be viewed as statistical estimators, subject to questions of variance and robustness. To illustrate this statistical framing, we present a systematic stability analysis of a state-of-the-art circuit discovery method: EAP-IG. We evaluate its variance and robustness through a comprehensive suite of controlled perturbations, including input resampling, prompt paraphrasing, hyperparameter variation, and injected noise within the causal analysis itself. Across a diverse set of models and tasks, our results demonstrate that EAP-IG exhibits high structural variance and sensitivity to hyperparameters, questioning the stability of its findings. Based on these results, we offer a set of best-practice recommendations for the field, advocating for the routine reporting of stability metrics to promote a more rigorous and statistically grounded science of interpretability. 

---
# What You See is What You Ask: Evaluating Audio Descriptions 

**Authors**: Divy Kala, Eshika Khandelwal, Makarand Tapaswi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00808)  

**Abstract**: Audio descriptions (ADs) narrate important visual details in movies, enabling Blind and Low Vision (BLV) users to understand narratives and appreciate visual details. Existing works in automatic AD generation mostly focus on few-second trimmed clips, and evaluate them by comparing against a single ground-truth reference AD. However, writing ADs is inherently subjective. Through alignment and analysis of two independent AD tracks for the same movies, we quantify the subjectivity in when and whether to describe, and what and how to highlight. Thus, we show that working with trimmed clips is inadequate. We propose ADQA, a QA benchmark that evaluates ADs at the level of few-minute long, coherent video segments, testing whether they would help BLV users understand the story and appreciate visual details. ADQA features visual appreciation (VA) questions about visual facts and narrative understanding (NU) questions based on the plot. Through ADQA, we show that current AD generation methods lag far behind human-authored ADs. We conclude with several recommendations for future work and introduce a public leaderboard for benchmarking. 

---
# From Scores to Preferences: Redefining MOS Benchmarking for Speech Quality Reward Modeling 

**Authors**: Yifei Cao, Changhao Jiang, Jiabao Zhuang, Jiajun Sun, Ming Zhang, Zhiheng Xi, Hui Li, Shihan Dou, Yuran Wang, Yunke Zhang, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00743)  

**Abstract**: Assessing the perceptual quality of synthetic speech is crucial for guiding the development and refinement of speech generation models. However, it has traditionally relied on human subjective ratings such as the Mean Opinion Score (MOS), which depend on manual annotations and often suffer from inconsistent rating standards and poor reproducibility. To address these limitations, we introduce MOS-RMBench, a unified benchmark that reformulates diverse MOS datasets into a preference-comparison setting, enabling rigorous evaluation across different datasets. Building on MOS-RMBench, we systematically construct and evaluate three paradigms for reward modeling: scalar reward models, semi-scalar reward models, and generative reward models (GRMs). Our experiments reveal three key findings: (1) scalar models achieve the strongest overall performance, consistently exceeding 74% accuracy; (2) most models perform considerably worse on synthetic speech than on human speech; and (3) all models struggle on pairs with very small MOS differences. To improve performance on these challenging pairs, we propose a MOS-aware GRM that incorporates an MOS-difference-based reward function, enabling the model to adaptively scale rewards according to the difficulty of each sample pair. Experimental results show that the MOS-aware GRM significantly improves fine-grained quality discrimination and narrows the gap with scalar models on the most challenging cases. We hope this work will establish both a benchmark and a methodological framework to foster more rigorous and scalable research in automatic speech quality assessment. 

---
# Stochastic Self-Organization in Multi-Agent Systems 

**Authors**: Nurbek Tastan, Samuel Horvath, Karthik Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00685)  

**Abstract**: Multi-agent systems (MAS) based on Large Language Models (LLMs) have the potential to solve tasks that are beyond the reach of any single LLM. However, this potential can only be realized when the collaboration mechanism between agents is optimized. Specifically, optimizing the communication structure between agents is critical for fruitful collaboration. Most existing approaches rely on fixed topologies, pretrained graph generators, optimization over edges, or employ external LLM judges, thereby adding to the complexity. In this work, we introduce a response-conditioned framework that adapts communication on-the-fly. Agents independently generate responses to the user query and assess peer contributions using an approximation of the Shapley value. A directed acyclic graph (DAG) is then constructed to regulate the propagation of the responses among agents, which ensures stable and efficient message transmission from high-contributing agents to others. This graph is dynamically updated based on the agent responses from the previous collaboration round. Since the proposed framework enables the self-organization of agents without additional supervision or training, we refer to it as SelfOrg. The SelfOrg framework goes beyond task- and query-level optimization and takes into account the stochastic nature of agent responses. Experiments with both strong and weak LLM backends demonstrate robust performance, with significant gains in the weak regime where prior methods collapse. We also theoretically show that multiple agents increase the chance of correctness and that the correct responses naturally dominate the information flow. 

---
# Milco: Learned Sparse Retrieval Across Languages via a Multilingual Connector 

**Authors**: Thong Nguyen, Yibin Lei, Jia-Huei Ju, Eugene Yang, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2510.00671)  

**Abstract**: Learned Sparse Retrieval (LSR) combines the efficiency of bi-encoders with the transparency of lexical matching, but existing approaches struggle to scale beyond English. We introduce MILCO, an LSR architecture that maps queries and documents from different languages into a shared English lexical space via a multilingual connector. MILCO is trained with a specialized two-stage regime that combines Sparse Alignment Pretraining with contrastive training to provide representation transparency and effectiveness while mitigating semantic collapse. Motivated by the observation that uncommon entities are often lost when projected into English, we propose a new LexEcho head, which enhances robustness by augmenting the English lexical representation with a source-language view obtained through a special [ECHO] token. MILCO achieves state-of-the-art multilingual and cross-lingual LSR performance, outperforming leading dense, sparse, and multi-vector baselines such as BGE-M3 and Qwen3-Embed on standard multilingual benchmarks, while supporting dynamic efficiency through post-hoc pruning. Notably, when using mass-based pruning to reduce document representations to only 30 active dimensions on average, MILCO 560M outperforms the similarly-sized Qwen3-Embed 0.6B with 1024 dimensions. 

---
# Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 

**Authors**: Alessio Devoto, Maximilian Jeblick, Simon Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00636)  

**Abstract**: Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient large language model inference. While attention-score-based KV cache pruning shows promise, it faces critical practical limitations: attention scores from future tokens are unavailable during compression, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible. To overcome these challenges, we introduce $\textbf{Expected Attention, a training-free compression method}$ that estimates KV pairs importance by predicting how future queries will attend to them. Our approach leverages the distributional properties of LLM activations to compute expected attention scores in closed form for each KV pair. These scores enable principled ranking and pruning of KV pairs with minimal impact on the residual stream, achieving effective compression without performance degradation. Importantly, our method operates seamlessly across both prefilling and decoding phases, consistently outperforming state-of-the-art baselines in both scenarios. Finally, $\textbf{we release KVPress, a comprehensive library to enable researchers to implement and benchmark KV cache compression methods, already including more than 20 techniques}$. 

---
# Hearing the Order: Investigating Selection Bias in Large Audio-Language Models 

**Authors**: Yu-Xiang Lin, Chen-An Li, Sheng-Lun Wei, Po-Chun Chen, Hsin-Hsi Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00628)  

**Abstract**: Large audio-language models (LALMs) are often used in tasks that involve reasoning over ordered options. An open question is whether their predictions are influenced by the order of answer choices, which would indicate a form of selection bias and undermine their reliability. In this paper, we identify and analyze this problem in LALMs. We demonstrate that no model is immune to this bias through extensive experiments on six LALMs across three widely used benchmarks and their spoken counterparts. Shuffling the order of answer options can cause performance fluctuations of up to 24% and even change model rankings, raising concerns about the reliability of current evaluation practices. We also study permutation-based strategies and show that they can mitigate bias in most cases. Our work represents the first systematic investigation of this issue in LALMs, and we hope it raises awareness and motivates further research in this direction. 

---
# When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models 

**Authors**: Chen-An Li, Tzu-Han Lin, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00626)  

**Abstract**: Large audio-language models (LALMs) unify speech and text processing, but their robustness in noisy real-world settings remains underexplored. We investigate how irrelevant audio, such as silence, synthetic noise, and environmental sounds, affects text reasoning tasks where audio is unnecessary. Across three text-based benchmarks, we find that even non-informative audio reduces accuracy and increases prediction volatility; the severity of interference scales with longer durations, higher amplitudes, and elevated decoding temperatures. Silence, often assumed neutral, destabilizes outputs as strongly as synthetic noise. While larger models show greater resilience, vulnerabilities persist across all evaluated systems. We further test mitigation strategies and find that prompting shows limited effectiveness, whereas self-consistency improves stability at the cost of increased computation. Our results reveal cross-modal interference as a key robustness challenge and highlight the need for efficient fusion strategies that preserve reasoning performance in the presence of irrelevant inputs. 

---
# HARPA: A Testability-Driven, Literature-Grounded Framework for Research Ideation 

**Authors**: Rosni Vasu, Peter Jansen, Pao Siangliulue, Cristina Sarasua, Abraham Bernstein, Peter Clark, Bhavana Dalvi Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2510.00620)  

**Abstract**: While there has been a surge of interest in automated scientific discovery (ASD), especially with the emergence of LLMs, it remains challenging for tools to generate hypotheses that are both testable and grounded in the scientific literature. Additionally, existing ideation tools are not adaptive to prior experimental outcomes. We developed HARPA to address these challenges by incorporating the ideation workflow inspired by human researchers. HARPA first identifies emerging research trends through literature mining, then explores hypothesis design spaces, and finally converges on precise, testable hypotheses by pinpointing research gaps and justifying design choices. Our evaluations show that HARPA-generated hypothesis-driven research proposals perform comparably to a strong baseline AI-researcher across most qualitative dimensions (e.g., specificity, novelty, overall quality), but achieve significant gains in feasibility(+0.78, p$<0.05$, bootstrap) and groundedness (+0.85, p$<0.01$, bootstrap) on a 10-point Likert scale. When tested with the ASD agent (CodeScientist), HARPA produced more successful executions (20 vs. 11 out of 40) and fewer failures (16 vs. 21 out of 40), showing that expert feasibility judgments track with actual execution success. Furthermore, to simulate how researchers continuously refine their understanding of what hypotheses are both testable and potentially interesting from experience, HARPA learns a reward model that scores new hypotheses based on prior experimental outcomes, achieving approx. a 28\% absolute gain over HARPA's untrained baseline scorer. Together, these methods represent a step forward in the field of AI-driven scientific discovery. 

---
# ACON: Optimizing Context Compression for Long-horizon LLM Agents 

**Authors**: Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00615)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. 

---
# Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors 

**Authors**: Yen-Shan Chen, Sian-Yao Huang, Cheng-Lin Yang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00586)  

**Abstract**: Existing data poisoning attacks on retrieval-augmented generation (RAG) systems scale poorly because they require costly optimization of poisoned documents for each target phrase. We introduce Eyes-on-Me, a modular attack that decomposes an adversarial document into reusable Attention Attractors and Focus Regions. Attractors are optimized to direct attention to the Focus Region. Attackers can then insert semantic baits for the retriever or malicious instructions for the generator, adapting to new targets at near zero cost. This is achieved by steering a small subset of attention heads that we empirically identify as strongly correlated with attack success. Across 18 end-to-end RAG settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me raises average attack success rates from 21.9 to 57.8 (+35.9 points, 2.6$\times$ over prior work). A single optimized attractor transfers to unseen black box retrievers and generators without retraining. Our findings establish a scalable paradigm for RAG data poisoning and show that modular, reusable components pose a practical threat to modern AI systems. They also reveal a strong link between attention concentration and model outputs, informing interpretability research. 

---
# Automated Evaluation can Distinguish the Good and Bad AI Responses to Patient Questions about Hospitalization 

**Authors**: Sarvesh Soni, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00436)  

**Abstract**: Automated approaches to answer patient-posed health questions are rising, but selecting among systems requires reliable evaluation. The current gold standard for evaluating the free-text artificial intelligence (AI) responses--human expert review--is labor-intensive and slow, limiting scalability. Automated metrics are promising yet variably aligned with human judgments and often context-dependent. To address the feasibility of automating the evaluation of AI responses to hospitalization-related questions posed by patients, we conducted a large systematic study of evaluation approaches. Across 100 patient cases, we collected responses from 28 AI systems (2800 total) and assessed them along three dimensions: whether a system response (1) answers the question, (2) appropriately uses clinical note evidence, and (3) uses general medical knowledge. Using clinician-authored reference answers to anchor metrics, automated rankings closely matched expert ratings. Our findings suggest that carefully designed automated evaluation can scale comparative assessment of AI systems and support patient-clinician communication. 

---
# AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features 

**Authors**: Xudong Zhu, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00404)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as powerful techniques for interpretability of large language models (LLMs), aiming to decompose hidden states into meaningful semantic features. While several SAE variants have been proposed, there remains no principled framework to derive SAEs from the original dictionary learning formulation. In this work, we introduce such a framework by unrolling the proximal gradient method for sparse coding. We show that a single-step update naturally recovers common SAE variants, including ReLU, JumpReLU, and TopK. Through this lens, we reveal a fundamental limitation of existing SAEs: their sparsity-inducing regularizers enforce non-negativity, preventing a single feature from representing bidirectional concepts (e.g., male vs. female). This structural constraint fragments semantic axes into separate, redundant features, limiting representational completeness. To address this issue, we propose AbsTopK SAE, a new variant derived from the $\ell_0$ sparsity constraint that applies hard thresholding over the largest-magnitude activations. By preserving both positive and negative activations, AbsTopK uncovers richer, bidirectional conceptual representations. Comprehensive experiments across four LLMs and seven probing and steering tasks show that AbsTopK improves reconstruction fidelity, enhances interpretability, and enables single features to encode contrasting concepts. Remarkably, AbsTopK matches or even surpasses the Difference-in-Mean method, a supervised approach that requires labeled data for each concept and has been shown in prior work to outperform SAEs. 

---
# GDLNN: Marriage of Programming Language and Neural Networks for Accurate and Easy-to-Explain Graph Classification 

**Authors**: Minseok Jeon, Seunghyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.00374)  

**Abstract**: We present GDLNN, a new graph machine learning architecture, for graph classification tasks. GDLNN combines a domain-specific programming language, called GDL, with neural networks. The main strength of GDLNN lies in its GDL layer, which generates expressive and interpretable graph representations. Since the graph representation is interpretable, existing model explanation techniques can be directly applied to explain GDLNN's predictions. Our evaluation shows that the GDL-based representation achieves high accuracy on most graph classification benchmark datasets, outperforming dominant graph learning methods such as GNNs. Applying an existing model explanation technique also yields high-quality explanations of GDLNN's predictions. Furthermore, the cost of GDLNN is low when the explanation cost is included. 

---
# Navigating the Synchrony-Stability Frontier in Adaptive Chatbots 

**Authors**: T. James Brandt  

**Link**: [PDF](https://arxiv.org/pdf/2510.00339)  

**Abstract**: Adaptive chatbots that mimic a user's linguistic style can build rapport and engagement, yet unconstrained mimicry risks an agent that feels unstable or sycophantic. We present a computational evaluation framework that makes the core design tension explicit: balancing moment-to-moment linguistic synchrony against long-term persona stability. Using an 8-dimensional style vector and a closed-loop "base+delta" prompting architecture, we simulate and compare explicit adaptation policies - Uncapped, Cap, Exponential Moving Average (EMA), Dead-Band, and Hybrids - on a human-log dataset. Our analysis maps a clear Pareto frontier: bounded policies achieve substantial gains in stability at a modest cost to synchrony. For example, a Hybrid (EMA+Cap) raises stability from 0.542 to 0.878 (+62%) while reducing synchrony by only 17%. We confirm this trade-off through large-scale replications on three public corpora (DailyDialog, Persona-Chat, EmpatheticDialogues) and LLM-in-the-loop validation across two model families. Furthermore, we quantify "prompt legibility," showing that frontier policies reduce instruction churn and cut jarring register flips (major tone changes) from 0.254 to 0.092, yielding systems that are easier to reason about and maintain. Taken together, our framework provides a general evaluation harness for style adaptation; a systematic ablation that identifies Pareto-efficient policies; robust validation across diverse datasets and models; and novel legibility metrics linking policy choices to system maintainability. 

---
# QSearchNet: A Quantum Walk Search Framework for Link Prediction 

**Authors**: Priyank Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2510.00325)  

**Abstract**: Link prediction is one of the fundamental problems in graph theory, critical for understanding and forecasting the evolution of complex systems like social and biological networks. While classical heuristics capture certain aspects of graph topology, they often struggle to optimally integrate local and global structural information or adapt to complex dependencies. Quantum computing offers a powerful alternative by leveraging superposition for simultaneous multi-path exploration and interference-driven integration of both local and global graph features. In this work, we introduce QSearchNet, a quantum-inspired framework based on Discrete-Time Quantum Walk (DTQW) dynamics and Grover's amplitude amplification. QSearchNet simulates a topology-aware quantum evolution to propagate amplitudes across multiple nodes simultaneously. By aligning interference patterns through quantum reflection and oracle-like phase-flip operation, it adaptively prioritizes multi-hop dependencies and amplifies structurally relevant paths corresponding to potential connections. Experiments on diverse real-world networks demonstrate competitive performance, particularly with hard negative samples under realistic evaluation conditions. 

---
# Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space 

**Authors**: Houjun Liu, Shikhar Murty, Christopher D. Manning, Róbert Csordás  

**Link**: [PDF](https://arxiv.org/pdf/2510.00219)  

**Abstract**: Current approaches for scaling inference-time compute in transformers rely on training them to emit explicit chain-of-thought tokens before producing an answer. While these methods are powerful, they are limited because they cannot be applied during pretraining and are limited to only serially-generated, natural-language verbalization to scale inference-time compute. In this work, we propose Thoughtbubbles, a transformer variant that natively performs parallel adaptive computation in latent space by learning to fork or delete residual streams. Thus, tokens that require a large amount of computation can form a "bubble" of cloned residuals in the middle of the network for additional thinking. Crucially, this behavior is learned during pretraining with only language modeling loss. Thoughtbubbles outperforms both standard decoder LMs as well as non-adaptive parallel computation approaches on OpenWebText and peS2o perplexity and in zero-shot evaluations such as HellaSwag and LAMBADA after pretraining across 150M to 772M parameter scales. The implicit nature of our method enables adaptive computation to be learned starting at pretraining time, paving the way to unify train and test-time behavior for reasoning models. 

---
# Optimizing What Matters: AUC-Driven Learning for Robust Neural Retrieval 

**Authors**: Nima Sheikholeslami, Erfan Hosseini, Patrice Bechard, Srivatsava Daruru, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00137)  

**Abstract**: Dual-encoder retrievers depend on the principle that relevant documents should score higher than irrelevant ones for a given query. Yet the dominant Noise Contrastive Estimation (NCE) objective, which underpins Contrastive Loss, optimizes a softened ranking surrogate that we rigorously prove is fundamentally oblivious to score separation quality and unrelated to AUC. This mismatch leads to poor calibration and suboptimal performance in downstream tasks like retrieval-augmented generation (RAG). To address this fundamental limitation, we introduce the MW loss, a new training objective that maximizes the Mann-Whitney U statistic, which is mathematically equivalent to the Area under the ROC Curve (AUC). MW loss encourages each positive-negative pair to be correctly ranked by minimizing binary cross entropy over score differences. We provide theoretical guarantees that MW loss directly upper-bounds the AoC, better aligning optimization with retrieval goals. We further promote ROC curves and AUC as natural threshold free diagnostics for evaluating retriever calibration and ranking quality. Empirically, retrievers trained with MW loss consistently outperform contrastive counterparts in AUC and standard retrieval metrics. Our experiments show that MW loss is an empirically superior alternative to Contrastive Loss, yielding better-calibrated and more discriminative retrievers for high-stakes applications like RAG. 

---
# ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Language Models 

**Authors**: Dongqi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00071)  

**Abstract**: Large Reasoning Language Models (LRLMs or LRMs) demonstrate remarkable capabilities in complex reasoning tasks, but suffer from significant computational inefficiencies due to overthinking phenomena. Existing efficient reasoning methods face the challenge of balancing reasoning quality with inference cost reduction. We propose \textbf{Adaptive Reasoning Suppression (ARS)}, a novel training-free approach that dynamically suppresses redundant reasoning steps while preserving accuracy through adaptive certainty monitoring. ARS introduces a multi-checkpoint certainty estimation mechanism with progressive suppression thresholds, achieving superior efficiency compared to static suppression methods. Our extensive evaluation across mathematical reasoning benchmarks using multiple model architectures demonstrates that ARS achieves up to 53%, 46.1%, and 57.9% in token, latency and energy reduction, while maintaining or improving accuracy. 

---
# Linear Regression in p-adic metric spaces 

**Authors**: Gregory D. Baker, Scott McCallum, Dirk Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2510.00043)  

**Abstract**: Many real-world machine learning problems involve inherently hierarchical data, yet traditional approaches rely on Euclidean metrics that fail to capture the discrete, branching nature of hierarchical relationships. We present a theoretical foundation for machine learning in p-adic metric spaces, which naturally respect hierarchical structure. Our main result proves that an n-dimensional plane minimizing the p-adic sum of distances to points in a dataset must pass through at least n + 1 of those points -- a striking contrast to Euclidean regression that highlights how p-adic metrics better align with the discrete nature of hierarchical data. As a corollary, a polynomial of degree n constructed to minimise the p-adic sum of residuals will pass through at least n + 1 points. As a further corollary, a polynomial of degree n approximating a higher degree polynomial at a finite number of points will yield a difference polynomial that has distinct rational roots. We demonstrate the practical significance of this result through two applications in natural language processing: analyzing hierarchical taxonomies and modeling grammatical morphology. These results suggest that p-adic metrics may be fundamental to properly handling hierarchical data structures in machine learning. In hierarchical data, interpolation between points often makes less sense than selecting actual observed points as representatives. 

---
# WaveMind: Towards a Conversational EEG Foundation Model Aligned to Textual and Visual Modalities 

**Authors**: Ziyi Zeng, Zhenyang Cai, Yixi Cai, Xidong Wang, Junying Chen, Rongsheng Wang, Yipeng Liu, Siqi Cai, Benyou Wang, Zhiguo Zhang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00032)  

**Abstract**: Electroencephalography (EEG) interpretation using multimodal large language models (MLLMs) offers a novel approach for analyzing brain signals. However, the complex nature of brain activity introduces critical challenges: EEG signals simultaneously encode both cognitive processes and intrinsic neural states, creating a mismatch in EEG paired-data modality that hinders effective cross-modal representation learning. Through a pivot investigation, we uncover complementary relationships between these modalities. Leveraging this insight, we propose mapping EEG signals and their corresponding modalities into a unified semantic space to achieve generalized interpretation. To fully enable conversational capabilities, we further introduce WaveMind-Instruct-338k, the first cross-task EEG dataset for instruction tuning. The resulting model demonstrates robust classification accuracy while supporting flexible, open-ended conversations across four downstream tasks, thereby offering valuable insights for both neuroscience research and the development of general-purpose EEG models. 

---
# IA aplicada al análisis del conflicto Irán-Israel: Mapeo de discursos en YouTube 

**Authors**: Alvaro Vallejo Ramírez  

**Link**: [PDF](https://arxiv.org/pdf/2510.00021)  

**Abstract**: Purpose. This study analyzes the digital representation of the Iran-Israel conflict that occurred in June 2025, based on 120,000 comments posted on YouTube. It sought to identify discursive positions regarding the actors involved and to examine how media and algorithmic biases shape digital conversations. Methodology. A mixed-methods design with triangulation was adopted. In the quantitative phase, natural language processing techniques and machine learning models (BERT and XLM-RoBERTa) were used to classify comments into ten categories. In the qualitative phase, a critical analysis of media context and ideological narratives was conducted, complemented by manual annotation and supervised training. This strategy enabled the integration of statistical robustness with contextual understanding. Results and conclusions. The findings reveal a clear overrepresentation of pro-Palestinian and anti-United States/Israel discourses, while pro-United States and anti-Palestinian positions were marginal. Iran, usually rendered invisible in global media, emerged as a central actor in the digital conversation during the conflict, suggesting a narrative shift away from previous hegemonic frameworks. Likewise, the results confirm the influence of algorithmic biases in amplifying certain discourses while limiting others. Original contributions. This work combines computational analysis and philosophical critique for the study of digital controversies, providing a methodological framework replicable in geopolitical contexts. It is one of the first Spanish-language studies to map, through artificial intelligence and critical analysis, discourses on an international conflict on YouTube, highlighting asymmetries and narrative disputes that are often overlooked. 

---
# Unpacking Musical Symbolism in Online Communities: Content-Based and Network-Centric Approaches 

**Authors**: Kajwan Ziaoddini  

**Link**: [PDF](https://arxiv.org/pdf/2510.00006)  

**Abstract**: This paper examines how musical symbolism is produced and circulated in online communities by combining content-based music analysis with a lightweight network perspective on lyrics. Using a curated corpus of 275 chart-topping songs enriched with audio descriptors (energy, danceability, loudness, liveness, valence, acousticness, speechiness, popularity) and full lyric transcripts, we build a reproducible pipeline that (i) quantifies temporal trends in sonic attributes, (ii) models lexical salience and co-occurrence, and (iii) profiles mood by genre. We find a decade-long decline in energy (79 -> 58) alongside a rise in danceability (59 -> 73); valence peaks in 2013 (63) and dips in 2014-2016 (42) before partially recovering. Correlation analysis shows strong coupling of energy with loudness (r = 0.74) and negative associations for acousticness with both energy (r = -0.54) and loudness (r = -0.51); danceability is largely orthogonal to other features (|r| < 0.20). Lyric tokenization (>114k tokens) reveals a pronoun-centric lexicon "I/you/me/my" and a dense co-occurrence structure in which interpersonal address anchors mainstream narratives. Mood differs systematically by style: R&B exhibits the highest mean valence (96), followed by K-Pop/Pop (77) and Indie/Pop (70), whereas Latin/Reggaeton is lower (37) despite high danceability. Read through a subcultural identity lens, these patterns suggest the mainstreaming of previously peripheral codes and a commercial preference for relaxed yet rhythmically engaging productions that sustain collective participation without maximal intensity. Methodologically, we contribute an integrated MIR-plus-network workflow spanning summary statistics, correlation structure, lexical co-occurrence matrices, and genre-wise mood profiling that is robust to modality sparsity and suitable for socially aware recommendation or community-level diffusion studies. 

---
