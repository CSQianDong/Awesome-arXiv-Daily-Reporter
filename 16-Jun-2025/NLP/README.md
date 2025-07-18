# code_transformed: The Influence of Large Language Models on Code 

**Authors**: Yuliang Xu, Siming Huang, Mingmeng Geng, Yao Wan, Xuanhua Shi, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12014)  

**Abstract**: Coding remains one of the most fundamental modes of interaction between humans and machines. With the rapid advancement of Large Language Models (LLMs), code generation capabilities have begun to significantly reshape programming practices. This development prompts a central question: Have LLMs transformed code style, and how can such transformation be characterized? In this paper, we present a pioneering study that investigates the impact of LLMs on code style, with a focus on naming conventions, complexity, maintainability, and similarity. By analyzing code from over 19,000 GitHub repositories linked to arXiv papers published between 2020 and 2025, we identify measurable trends in the evolution of coding style that align with characteristics of LLM-generated code. For instance, the proportion of snake\_case variable names in Python code increased from 47% in Q1 2023 to 51% in Q1 2025. Furthermore, we investigate how LLMs approach algorithmic problems by examining their reasoning processes. Given the diversity of LLMs and usage scenarios, among other factors, it is difficult or even impossible to precisely estimate the proportion of code generated or assisted by LLMs. Our experimental results provide the first large-scale empirical evidence that LLMs affect real-world programming style. 

---
# Improving Large Language Model Safety with Contrastive Representation Learning 

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11938)  

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at this https URL 

---
# Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback 

**Authors**: Dongwei Jiang, Alvin Zhang, Andrew Wang, Nicholas Andrews, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11930)  

**Abstract**: Recent studies have shown LLMs possess some ability to improve their responses when given external feedback. However, it remains unclear how effectively and thoroughly these models can incorporate extrinsic feedback. In an ideal scenario, if LLMs receive near-perfect and complete feedback, we would expect them to fully integrate the feedback and change their incorrect answers to correct ones. In this paper, we systematically investigate LLMs' ability to incorporate feedback by designing a controlled experimental environment. For each problem, a solver model attempts a solution, then a feedback generator with access to near-complete ground-truth answers produces targeted feedback, after which the solver tries again. We evaluate this pipeline across a diverse range of tasks, including math reasoning, knowledge reasoning, scientific reasoning, and general multi-domain evaluations with state-of-the-art language models including Claude 3.7 (with and without extended thinking). Surprisingly, even under these near-ideal conditions, solver models consistently show resistance to feedback, a limitation that we term FEEDBACK FRICTION. To mitigate this limitation, we experiment with sampling-based strategies like progressive temperature increases and explicit rejection of previously attempted incorrect answers, which yield improvements but still fail to help models achieve target performance. We also perform a rigorous exploration of potential causes of FEEDBACK FRICTION, ruling out factors such as model overconfidence and data familiarity. We hope that highlighting this issue in LLMs and ruling out several apparent causes will help future research in self-improvement. 

---
# Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study 

**Authors**: Greta Damo, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2506.11919)  

**Abstract**: Counter-speech (CS) is a key strategy for mitigating online Hate Speech (HS), yet defining the criteria to assess its effectiveness remains an open challenge. We propose a novel computational framework for CS effectiveness classification, grounded in social science concepts. Our framework defines six core dimensions - Clarity, Evidence, Emotional Appeal, Rebuttal, Audience Adaptation, and Fairness - which we use to annotate 4,214 CS instances from two benchmark datasets, resulting in a novel linguistic resource released to the community. In addition, we propose two classification strategies, multi-task and dependency-based, achieving strong results (0.94 and 0.96 average F1 respectively on both expert- and user-written CS), outperforming standard baselines, and revealing strong interdependence among dimensions. 

---
# GeistBERT: Breathing Life into German NLP 

**Authors**: Raphael Scheible-Schmitt, Johann Frei  

**Link**: [PDF](https://arxiv.org/pdf/2506.11903)  

**Abstract**: Advances in transformer-based language models have highlighted the benefits of language-specific pre-training on high-quality corpora. In this context, German NLP stands to gain from updated architectures and modern datasets tailored to the linguistic characteristics of the German language. GeistBERT seeks to improve German language processing by incrementally training on a diverse corpus and optimizing model performance across various NLP tasks. It was pre-trained using fairseq with standard hyperparameters, initialized from GottBERT weights, and trained on a large-scale German corpus using Whole Word Masking (WWM). Based on the pre-trained model, we derived extended-input variants using Nyströmformer and Longformer architectures with support for sequences up to 8k tokens. While these long-context models were not evaluated on dedicated long-context benchmarks, they are included in our release. We assessed all models on NER (CoNLL 2003, GermEval 2014) and text classification (GermEval 2018 fine/coarse, 10kGNAD) using $F_1$ score and accuracy. The GeistBERT models achieved strong performance, leading all tasks among the base models and setting a new state-of-the-art (SOTA). Notably, the base models outperformed larger models in several tasks. To support the German NLP research community, we are releasing GeistBERT under the MIT license. 

---
# Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache 

**Authors**: Xiaoran Liu, Siyang He, Qiqi Wang, Ruixiao Li, Yuerong Song, Zhigeng Liu, Linlin Li, Qun Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11886)  

**Abstract**: Large Language Models struggle with memory demands from the growing Key-Value (KV) cache as context lengths increase. Existing compression methods homogenize head dimensions or rely on attention-guided token pruning, often sacrificing accuracy or introducing computational overhead. We propose FourierAttention, a training-free framework that exploits the heterogeneous roles of transformer head dimensions: lower dimensions prioritize local context, while upper ones capture long-range dependencies. By projecting the long-context-insensitive dimensions onto orthogonal Fourier bases, FourierAttention approximates their temporal evolution with fixed-length spectral coefficients. Evaluations on LLaMA models show that FourierAttention achieves the best long-context accuracy on LongBench and Needle-In-A-Haystack (NIAH). Besides, a custom Triton kernel, FlashFourierAttention, is designed to optimize memory via streamlined read-write operations, enabling efficient deployment without performance compromise. 

---
# Post Persona Alignment for Multi-Session Dialogue Generation 

**Authors**: Yi-Pei Chen, Noriki Nishida, Hideki Nakayama, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.11857)  

**Abstract**: Multi-session persona-based dialogue generation presents challenges in maintaining long-term consistency and generating diverse, personalized responses. While large language models (LLMs) excel in single-session dialogues, they struggle to preserve persona fidelity and conversational coherence across extended interactions. Existing methods typically retrieve persona information before response generation, which can constrain diversity and result in generic outputs. We propose Post Persona Alignment (PPA), a novel two-stage framework that reverses this process. PPA first generates a general response based solely on dialogue context, then retrieves relevant persona memories using the response as a query, and finally refines the response to align with the speaker's persona. This post-hoc alignment strategy promotes naturalness and diversity while preserving consistency and personalization. Experiments on multi-session LLM-generated dialogue data demonstrate that PPA significantly outperforms prior approaches in consistency, diversity, and persona relevance, offering a more flexible and effective paradigm for long-term personalized dialogue generation. 

---
# Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks? 

**Authors**: Simeon Junker, Manar Ali, Larissa Koch, Sina Zarrieß, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2506.11807)  

**Abstract**: We investigate the linguistic abilities of multimodal large language models in reference resolution tasks featuring simple yet abstract visual stimuli, such as color patches and color grids. Although the task may not seem challenging for today's language models, being straightforward for human dyads, we consider it to be a highly relevant probe of the pragmatic capabilities of MLLMs. Our results and analyses indeed suggest that basic pragmatic capabilities, such as context-dependent interpretation of color descriptions, still constitute major challenges for state-of-the-art MLLMs. 

---
# Persona-driven Simulation of Voting Behavior in the European Parliament with Large Language Models 

**Authors**: Maximilian Kreutner, Marlene Lutz, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2506.11798)  

**Abstract**: Large Language Models (LLMs) display remarkable capabilities to understand or even produce political discourse, but have been found to consistently display a progressive left-leaning bias. At the same time, so-called persona or identity prompts have been shown to produce LLM behavior that aligns with socioeconomic groups that the base model is not aligned with. In this work, we analyze whether zero-shot persona prompting with limited information can accurately predict individual voting decisions and, by aggregation, accurately predict positions of European groups on a diverse set of policies. We evaluate if predictions are stable towards counterfactual arguments, different persona prompts and generation methods. Finally, we find that we can simulate voting behavior of Members of the European Parliament reasonably well with a weighted F1 score of approximately 0.793. Our persona dataset of politicians in the 2024 European Parliament and our code are available at this https URL. 

---
# Long-Short Alignment for Effective Long-Context Modeling in LLMs 

**Authors**: Tianqi Du, Haotian Huang, Yifei Wang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11769)  

**Abstract**: Large language models (LLMs) have exhibited impressive performance and surprising emergent properties. However, their effectiveness remains limited by the fixed context window of the transformer architecture, posing challenges for long-context modeling. Among these challenges, length generalization -- the ability to generalize to sequences longer than those seen during training -- is a classical and fundamental problem. In this work, we propose a fresh perspective on length generalization, shifting the focus from the conventional emphasis on input features such as positional encodings or data structures to the output distribution of the model. Specifically, through case studies on synthetic tasks, we highlight the critical role of \textbf{long-short alignment} -- the consistency of output distributions across sequences of varying lengths. Extending this insight to natural language tasks, we propose a metric called Long-Short Misalignment to quantify this phenomenon, uncovering a strong correlation between the metric and length generalization performance. Building on these findings, we develop a regularization term that promotes long-short alignment during training. Extensive experiments validate the effectiveness of our approach, offering new insights for achieving more effective long-context modeling in LLMs. Code is available at this https URL. 

---
# DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents 

**Authors**: Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11763)  

**Abstract**: Deep Research Agents are a prominent category of LLM-based agents. By autonomously orchestrating multistep web exploration, targeted retrieval, and higher-order synthesis, they transform vast amounts of online information into analyst-grade, citation-rich reports--compressing hours of manual desk research into minutes. However, a comprehensive benchmark for systematically evaluating the capabilities of these agents remains absent. To bridge this gap, we present DeepResearch Bench, a benchmark consisting of 100 PhD-level research tasks, each meticulously crafted by domain experts across 22 distinct fields. Evaluating DRAs is inherently complex and labor-intensive. We therefore propose two novel methodologies that achieve strong alignment with human judgment. The first is a reference-based method with adaptive criteria to assess the quality of generated research reports. The other framework is introduced to evaluate DRA's information retrieval and collection capabilities by assessing its effective citation count and overall citation accuracy. We have open-sourced DeepResearch Bench and key components of these frameworks at this https URL to accelerate the development of practical LLM-based agents. 

---
# DART: Distilling Autoregressive Reasoning to Silent Thought 

**Authors**: Nan Jiang, Ziming Wu, De-Chuan Zhan, Fuming Lai, Shaobing Lian  

**Link**: [PDF](https://arxiv.org/pdf/2506.11752)  

**Abstract**: Chain-of-Thought (CoT) reasoning has significantly advanced Large Language Models (LLMs) in solving complex tasks. However, its autoregressive paradigm leads to significant computational overhead, hindering its deployment in latency-sensitive applications. To address this, we propose \textbf{DART} (\textbf{D}istilling \textbf{A}utoregressive \textbf{R}easoning to Silent \textbf{T}hought), a self-distillation framework that enables LLMs to replace autoregressive CoT with non-autoregressive Silent Thought (ST). Specifically, DART introduces two training pathways: the CoT pathway for traditional reasoning and the ST pathway for generating answers directly from a few ST tokens. The ST pathway utilizes a lightweight Reasoning Evolvement Module (REM) to align its hidden states with the CoT pathway, enabling the ST tokens to evolve into informative embeddings. During inference, only the ST pathway is activated, leveraging evolving ST tokens to deliver the answer directly. Extensive experimental results demonstrate that DART achieves comparable reasoning performance to existing baselines while offering significant efficiency gains, serving as a feasible alternative for efficient reasoning. 

---
# The Cambrian Explosion of Mixed-Precision Matrix Multiplication for Quantized Deep Learning Inference 

**Authors**: Héctor Martínez, Adrián Castelló, Francisco D. Igual, Enrique S. Quintana-Ortí  

**Link**: [PDF](https://arxiv.org/pdf/2506.11728)  

**Abstract**: Recent advances in deep learning (DL) have led to a shift from traditional 64-bit floating point (FP64) computations toward reduced-precision formats, such as FP16, BF16, and 8- or 16-bit integers, combined with mixed-precision arithmetic. This transition enhances computational throughput, reduces memory and bandwidth usage, and improves energy efficiency, offering significant advantages for resource-constrained edge devices. To support this shift, hardware architectures have evolved accordingly, now including adapted ISAs (Instruction Set Architectures) that expose mixed-precision vector units and matrix engines tailored for DL workloads. At the heart of many DL and scientific computing tasks is the general matrix-matrix multiplication gemm, a fundamental kernel historically optimized using axpy vector instructions on SIMD (single instruction, multiple data) units. However, as hardware moves toward mixed-precision dot-product-centric operations optimized for quantized inference, these legacy approaches are being phased out. In response to this, our paper revisits traditional high-performance gemm and describes strategies for adapting it to mixed-precision integer (MIP) arithmetic across modern ISAs, including x86_64, ARM, and RISC-V. Concretely, we illustrate novel micro-kernel designs and data layouts that better exploit today's specialized hardware and demonstrate significant performance gains from MIP arithmetic over floating-point implementations across three representative CPU architectures. These contributions highlight a new era of gemm optimization-driven by the demands of DL inference on heterogeneous architectures, marking what we term as the "Cambrian period" for matrix multiplication. 

---
# Configurable Preference Tuning with Rubric-Guided Synthetic Data 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2506.11702)  

**Abstract**: Models of human feedback for AI alignment, such as those underpinning Direct Preference Optimization (DPO), often bake in a singular, static set of preferences, limiting adaptability. This paper challenges the assumption of monolithic preferences by introducing Configurable Preference Tuning (CPT), a novel framework for endowing language models with the ability to dynamically adjust their behavior based on explicit, human-interpretable directives. CPT leverages synthetically generated preference data, conditioned on system prompts derived from structured, fine-grained rubrics that define desired attributes like writing style. By fine-tuning with these rubric-guided preferences, the LLM learns to modulate its outputs at inference time in response to the system prompt, without retraining. This approach not only offers fine-grained control but also provides a mechanism for modeling more nuanced and context-dependent human feedback. Several experimental artifacts, such as training code, generated datasets and fine-tuned models are released at this https URL 

---
# LLMs for Sentence Simplification: A Hybrid Multi-Agent prompting Approach 

**Authors**: Pratibha Zunjare, Michael Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11681)  

**Abstract**: This paper addresses the challenge of transforming complex sentences into sequences of logical, simplified sentences while preserving semantic and logical integrity with the help of Large Language Models. We propose a hybrid approach that combines advanced prompting with multi-agent architectures to enhance the sentence simplification process. Experimental results show that our approach was able to successfully simplify 70% of the complex sentences written for video game design application. In comparison, a single-agent approach attained a 48% success rate on the same task. 

---
# Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE 

**Authors**: Alicja Dobrzeniecka, Antske Fokkens, Pia Sommerauer  

**Link**: [PDF](https://arxiv.org/pdf/2506.11673)  

**Abstract**: Amnesic probing is a technique used to examine the influence of specific linguistic information on the behaviour of a model. This involves identifying and removing the relevant information and then assessing whether the model's performance on the main task changes. If the removed information is relevant, the model's performance should decline. The difficulty with this approach lies in removing only the target information while leaving other information unchanged. It has been shown that Iterative Nullspace Projection (INLP), a widely used removal technique, introduces random modifications to representations when eliminating target information. We demonstrate that Mean Projection (MP) and LEACE, two proposed alternatives, remove information in a more targeted manner, thereby enhancing the potential for obtaining behavioural explanations through Amnesic Probing. 

---
# Converting Annotated Clinical Cases into Structured Case Report Forms 

**Authors**: Pietro Ferrazzi, Alberto Lavelli, Bernardo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2506.11666)  

**Abstract**: Case Report Forms (CRFs) are largely used in medical research as they ensure accuracy, reliability, and validity of results in clinical studies. However, publicly available, wellannotated CRF datasets are scarce, limiting the development of CRF slot filling systems able to fill in a CRF from clinical notes. To mitigate the scarcity of CRF datasets, we propose to take advantage of available datasets annotated for information extraction tasks and to convert them into structured CRFs. We present a semi-automatic conversion methodology, which has been applied to the E3C dataset in two languages (English and Italian), resulting in a new, high-quality dataset for CRF slot filling. Through several experiments on the created dataset, we report that slot filling achieves 59.7% for Italian and 67.3% for English on a closed Large Language Models (zero-shot) and worse performances on three families of open-source models, showing that filling CRFs is challenging even for recent state-of-the-art LLMs. We release the datest at this https URL 

---
# LoRA-Gen: Specializing Large Language Model via Online LoRA Generation 

**Authors**: Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Yixiao Ge, Xiu Li, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11638)  

**Abstract**: Recent advances have highlighted the benefits of scaling language models to enhance performance across a wide range of NLP tasks. However, these approaches still face limitations in effectiveness and efficiency when applied to domain-specific tasks, particularly for small edge-side models. We propose the LoRA-Gen framework, which utilizes a large cloud-side model to generate LoRA parameters for edge-side models based on task descriptions. By employing the reparameterization technique, we merge the LoRA parameters into the edge-side model to achieve flexible specialization. Our method facilitates knowledge transfer between models while significantly improving the inference efficiency of the specialized model by reducing the input context length. Without specialized training, LoRA-Gen outperforms conventional LoRA fine-tuning, which achieves competitive accuracy and a 2.1x speedup with TinyLLaMA-1.1B in reasoning tasks. Besides, our method delivers a compression ratio of 10.1x with Gemma-2B on intelligent agent tasks. 

---
# SceneGram: Conceptualizing and Describing Tangrams in Scene Context 

**Authors**: Simeon Junker, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2506.11631)  

**Abstract**: Research on reference and naming suggests that humans can come up with very different ways of conceptualizing and referring to the same object, e.g. the same abstract tangram shape can be a "crab", "sink" or "space ship". Another common assumption in cognitive science is that scene context fundamentally shapes our visual perception of objects and conceptual expectations. This paper contributes SceneGram, a dataset of human references to tangram shapes placed in different scene contexts, allowing for systematic analyses of the effect of scene context on conceptualization. Based on this data, we analyze references to tangram shapes generated by multimodal LLMs, showing that these models do not account for the richness and variability of conceptualizations found in human references. 

---
# Are LLMs Good Text Diacritizers? An Arabic and Yorùbá Case Study 

**Authors**: Hawau Olamide Toyin, Samar M. Magdy, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.11602)  

**Abstract**: We investigate the effectiveness of large language models (LLMs) for text diacritization in two typologically distinct languages: Arabic and Yoruba. To enable a rigorous evaluation, we introduce a novel multilingual dataset MultiDiac, with diverse samples that capture a range of diacritic ambiguities. We evaluate 14 LLMs varying in size, accessibility, and language coverage, and benchmark them against 6 specialized diacritization models. Additionally, we fine-tune four small open-source models using LoRA for Yoruba. Our results show that many off-the-shelf LLMs outperform specialized diacritization models for both Arabic and Yoruba, but smaller models suffer from hallucinations. Fine-tuning on a small dataset can help improve diacritization performance and reduce hallucination rates. 

---
# From Persona to Person: Enhancing the Naturalness with Multiple Discourse Relations Graph Learning in Personalized Dialogue Generation 

**Authors**: Chih-Hao Hsu, Ying-Jia Lin, Hung-Yu Kao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11557)  

**Abstract**: In dialogue generation, the naturalness of responses is crucial for effective human-machine interaction. Personalized response generation poses even greater challenges, as the responses must remain coherent and consistent with the user's personal traits or persona descriptions. We propose MUDI ($\textbf{Mu}$ltiple $\textbf{Di}$scourse Relations Graph Learning) for personalized dialogue generation. We utilize a Large Language Model to assist in annotating discourse relations and to transform dialogue data into structured dialogue graphs. Our graph encoder, the proposed DialogueGAT model, then captures implicit discourse relations within this structure, along with persona descriptions. During the personalized response generation phase, novel coherence-aware attention strategies are implemented to enhance the decoder's consideration of discourse relations. Our experiments demonstrate significant improvements in the quality of personalized responses, thus resembling human-like dialogue exchanges. 

---
# On the Effectiveness of Integration Methods for Multimodal Dialogue Response Retrieval 

**Authors**: Seongbo Jang, Seonghyeon Lee, Dongha Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11499)  

**Abstract**: Multimodal chatbots have become one of the major topics for dialogue systems in both research community and industry. Recently, researchers have shed light on the multimodality of responses as well as dialogue contexts. This work explores how a dialogue system can output responses in various modalities such as text and image. To this end, we first formulate a multimodal dialogue response retrieval task for retrieval-based systems as the combination of three subtasks. We then propose three integration methods based on a two-step approach and an end-to-end approach, and compare the merits and demerits of each method. Experimental results on two datasets demonstrate that the end-to-end approach achieves comparable performance without an intermediate step in the two-step approach. In addition, a parameter sharing strategy not only reduces the number of parameters but also boosts performance by transferring knowledge across the subtasks and the modalities. 

---
# Lag-Relative Sparse Attention In Long Context Training 

**Authors**: Manlai Liang, Wanyi Huang, Mandi Liu, Huaijun Li, Jinlong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11498)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language processing and generation, yet their ability to handle long-context input remains constrained by the quadratic complexity of attention computation and linear-increasing key-value memory footprint. To reduce computational costs and memory, key-value cache compression techniques are commonly applied at inference time, but this often leads to severe performance degradation, as models are not trained to handle compressed context. Although there are more sophisticated compression methods, they are typically unsuitable for post-training because of their incompatibility with gradient-based optimization or high computation overhead. To fill this gap with no additional parameter and little computation overhead, we propose Lag-Relative Sparse Attention(LRSA) anchored by the LagKV compression method for long context post-training. Our method performs chunk-by-chunk prefilling, which selects the top K most relevant key-value pairs in a fixed-size lagging window, allowing the model to focus on salient historical context while maintaining efficiency. Experimental results show that our approach significantly enhances the robustness of the LLM with key-value compression and achieves better fine-tuned results in the question-answer tuning task. 

---
# Relational Schemata in BERT Are Inducible, Not Emergent: A Study of Performance vs. Competence in Language Models 

**Authors**: Cole Gawin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11485)  

**Abstract**: While large language models like BERT demonstrate strong empirical performance on semantic tasks, whether this reflects true conceptual competence or surface-level statistical association remains unclear. I investigate whether BERT encodes abstract relational schemata by examining internal representations of concept pairs across taxonomic, mereological, and functional relations. I compare BERT's relational classification performance with representational structure in [CLS] token embeddings. Results reveal that pretrained BERT enables high classification accuracy, indicating latent relational signals. However, concept pairs organize by relation type in high-dimensional embedding space only after fine-tuning on supervised relation classification tasks. This indicates relational schemata are not emergent from pretraining alone but can be induced via task scaffolding. These findings demonstrate that behavioral performance does not necessarily imply structured conceptual understanding, though models can acquire inductive biases for grounded relational abstraction through appropriate training. 

---
# ImmunoFOMO: Are Language Models missing what oncologists see? 

**Authors**: Aman Sinha, Bogdan-Valentin Popescu, Xavier Coubez, Marianne Clausel, Mathieu Constant  

**Link**: [PDF](https://arxiv.org/pdf/2506.11478)  

**Abstract**: Language models (LMs) capabilities have grown with a fast pace over the past decade leading researchers in various disciplines, such as biomedical research, to increasingly explore the utility of LMs in their day-to-day applications. Domain specific language models have already been in use for biomedical natural language processing (NLP) applications. Recently however, the interest has grown towards medical language models and their understanding capabilities. In this paper, we investigate the medical conceptual grounding of various language models against expert clinicians for identification of hallmarks of immunotherapy in breast cancer abstracts. Our results show that pre-trained language models have potential to outperform large language models in identifying very specific (low-level) concepts. 

---
# Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards 

**Authors**: Jaehoon Yun, Jiwoong Sohn, Jungwoo Park, Hyunjae Kim, Xiangru Tang, Yanjun Shao, Yonghoe Koo, Minhyeok Ko, Qingyu Chen, Mark Gerstein, Michael Moor, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11474)  

**Abstract**: Large language models have shown promise in clinical decision making, but current approaches struggle to localize and correct errors at specific steps of the reasoning process. This limitation is critical in medicine, where identifying and addressing reasoning errors is essential for accurate diagnosis and effective patient care. We introduce Med-PRM, a process reward modeling framework that leverages retrieval-augmented generation to verify each reasoning step against established medical knowledge bases. By verifying intermediate reasoning steps with evidence retrieved from clinical guidelines and literature, our model can precisely assess the reasoning quality in a fine-grained manner. Evaluations on five medical QA benchmarks and two open-ended diagnostic tasks demonstrate that Med-PRM achieves state-of-the-art performance, with improving the performance of base models by up to 13.50% using Med-PRM. Moreover, we demonstrate the generality of Med-PRM by integrating it in a plug-and-play fashion with strong policy models such as Meerkat, achieving over 80\% accuracy on MedQA for the first time using small-scale models of 8 billion parameters. Our code and data are available at: this https URL 

---
# A Gamified Evaluation and Recruitment Platform for Low Resource Language Machine Translation Systems 

**Authors**: Carlos Rafael Catalan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11467)  

**Abstract**: Human evaluators provide necessary contributions in evaluating large language models. In the context of Machine Translation (MT) systems for low-resource languages (LRLs), this is made even more apparent since popular automated metrics tend to be string-based, and therefore do not provide a full picture of the nuances of the behavior of the system. Human evaluators, when equipped with the necessary expertise of the language, will be able to test for adequacy, fluency, and other important metrics. However, the low resource nature of the language means that both datasets and evaluators are in short supply. This presents the following conundrum: How can developers of MT systems for these LRLs find adequate human evaluators and datasets? This paper first presents a comprehensive review of existing evaluation procedures, with the objective of producing a design proposal for a platform that addresses the resource gap in terms of datasets and evaluators in developing MT systems. The result is a design for a recruitment and gamified evaluation platform for developers of MT systems. Challenges are also discussed in terms of evaluating this platform, as well as its possible applications in the wider scope of Natural Language Processing (NLP) research. 

---
# AbsenceBench: Language Models Can't Tell What's Missing 

**Authors**: Harvey Yiyun Fu, Aryan Shrivastava, Jared Moore, Peter West, Chenhao Tan, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2506.11440)  

**Abstract**: Large language models (LLMs) are increasingly capable of processing long inputs and locating specific information within them, as evidenced by their performance on the Needle in a Haystack (NIAH) test. However, while models excel at recalling surprising information, they still struggle to identify clearly omitted information. We introduce AbsenceBench to assesses LLMs' capacity to detect missing information across three domains: numerical sequences, poetry, and GitHub pull requests. AbsenceBench asks models to identify which pieces of a document were deliberately removed, given access to both the original and edited contexts. Despite the apparent straightforwardness of these tasks, our experiments reveal that even state-of-the-art models like Claude-3.7-Sonnet achieve only 69.6% F1-score with a modest average context length of 5K tokens. Our analysis suggests this poor performance stems from a fundamental limitation: Transformer attention mechanisms cannot easily attend to "gaps" in documents since these absences don't correspond to any specific keys that can be attended to. Overall, our results and analysis provide a case study of the close proximity of tasks where models are already superhuman (NIAH) and tasks where models breakdown unexpectedly (AbsenceBench). 

---
# KoGEC : Korean Grammatical Error Correction with Pre-trained Translation Models 

**Authors**: Taeeun Kim, Semin Jeong, Youngsook Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.11432)  

**Abstract**: This research introduces KoGEC, a Korean Grammatical Error Correction system using pre\--trained translation models. We fine-tuned NLLB (No Language Left Behind) models for Korean GEC, comparing their performance against large language models like GPT-4 and HCX-3. The study used two social media conversation datasets for training and testing. The NLLB models were fine-tuned using special language tokens to distinguish between original and corrected Korean sentences. Evaluation was done using BLEU scores and an "LLM as judge" method to classify error types. Results showed that the fine-tuned NLLB (KoGEC) models outperformed GPT-4o and HCX-3 in Korean GEC tasks. KoGEC demonstrated a more balanced error correction profile across various error types, whereas the larger LLMs tended to focus less on punctuation errors. We also developed a Chrome extension to make the KoGEC system accessible to users. Finally, we explored token vocabulary expansion to further improve the model but found it to decrease model performance. This research contributes to the field of NLP by providing an efficient, specialized Korean GEC system and a new evaluation method. It also highlights the potential of compact, task-specific models to compete with larger, general-purpose language models in specialized NLP tasks. 

---
# Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards 

**Authors**: Jeff Da, Clinton Wang, Xiang Deng, Yuntao Ma, Nikhil Barhate, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.11425)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle. 

---
# Efficient Long-Context LLM Inference via KV Cache Clustering 

**Authors**: Jie Hu, Shengnan Wang, Yutong He, Ping Gong, Jiawei Yi, Juncheng Zhang, Youhui Bai, Renhai Chen, Gong Zhang, Cheng Li, Kun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11418)  

**Abstract**: Large language models (LLMs) with extended context windows have become increasingly prevalent for tackling complex tasks. However, the substantial Key-Value (KV) cache required for long-context LLMs poses significant deployment challenges. Existing approaches either discard potentially critical information needed for future generations or offer limited efficiency gains due to high computational overhead. In this paper, we introduce Chelsea, a simple yet effective framework for online KV cache clustering. Our approach is based on the observation that key states exhibit high similarity along the sequence dimension. To enable efficient clustering, we divide the sequence into chunks and propose Chunked Soft Matching, which employs an alternating partition strategy within each chunk and identifies clusters based on similarity. Chelsea then merges the KV cache within each cluster into a single centroid. Additionally, we provide a theoretical analysis of the computational complexity and the optimality of the intra-chunk partitioning strategy. Extensive experiments across various models and long-context benchmarks demonstrate that Chelsea achieves up to 80% reduction in KV cache memory usage while maintaining comparable model performance. Moreover, with minimal computational overhead, Chelsea accelerates the decoding stage of inference by up to 3.19$\times$ and reduces end-to-end latency by up to 2.72$\times$. 

---
# Predicting Early-Onset Colorectal Cancer with Large Language Models 

**Authors**: Wilson Lau, Youngwon Kim, Sravanthi Parasa, Md Enamul Haque, Anand Oka, Jay Nanduri  

**Link**: [PDF](https://arxiv.org/pdf/2506.11410)  

**Abstract**: The incidence rate of early-onset colorectal cancer (EoCRC, age < 45) has increased every year, but this population is younger than the recommended age established by national guidelines for cancer screening. In this paper, we applied 10 different machine learning models to predict EoCRC, and compared their performance with advanced large language models (LLM), using patient conditions, lab results, and observations within 6 months of patient journey prior to the CRC diagnoses. We retrospectively identified 1,953 CRC patients from multiple health systems across the United States. The results demonstrated that the fine-tuned LLM achieved an average of 73% sensitivity and 91% specificity. 

---
# Curriculum-Guided Layer Scaling for Language Model Pretraining 

**Authors**: Karanpartap Singh, Neil Band, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2506.11389)  

**Abstract**: As the cost of pretraining large language models grows, there is continued interest in strategies to improve learning efficiency during this core training stage. Motivated by cognitive development, where humans gradually build knowledge as their brains mature, we propose Curriculum-Guided Layer Scaling (CGLS), a framework for compute-efficient pretraining that synchronizes increasing data difficulty with model growth through progressive layer stacking (i.e. gradually adding layers during training). At the 100M parameter scale, using a curriculum transitioning from synthetic short stories to general web data, CGLS outperforms baseline methods on the question-answering benchmarks PIQA and ARC. Pretraining at the 1.2B scale, we stratify the DataComp-LM corpus with a DistilBERT-based classifier and progress from general text to highly technical or specialized content. Our results show that progressively increasing model depth alongside sample difficulty leads to better generalization and zero-shot performance on various downstream benchmarks. Altogether, our findings demonstrate that CGLS unlocks the potential of progressive stacking, offering a simple yet effective strategy for improving generalization on knowledge-intensive and reasoning tasks. 

---
# A Variational Approach for Mitigating Entity Bias in Relation Extraction 

**Authors**: Samuel Mensah, Elena Kochkina, Jabez Magomere, Joy Prakash Sain, Simerjot Kaur, Charese Smiley  

**Link**: [PDF](https://arxiv.org/pdf/2506.11381)  

**Abstract**: Mitigating entity bias is a critical challenge in Relation Extraction (RE), where models often rely excessively on entities, resulting in poor generalization. This paper presents a novel approach to address this issue by adapting a Variational Information Bottleneck (VIB) framework. Our method compresses entity-specific information while preserving task-relevant features. It achieves state-of-the-art performance on relation extraction datasets across general, financial, and biomedical domains, in both indomain (original test sets) and out-of-domain (modified test sets with type-constrained entity replacements) settings. Our approach offers a robust, interpretable, and theoretically grounded methodology. 

---
# The Biased Samaritan: LLM biases in Perceived Kindness 

**Authors**: Jack H Fagan, Ruhaan Juyaal, Amy Yue-Ming Yu, Siya Pun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11361)  

**Abstract**: While Large Language Models (LLMs) have become ubiquitous in many fields, understanding and mitigating LLM biases is an ongoing issue. This paper provides a novel method for evaluating the demographic biases of various generative AI models. By prompting models to assess a moral patient's willingness to intervene constructively, we aim to quantitatively evaluate different LLMs' biases towards various genders, races, and ages. Our work differs from existing work by aiming to determine the baseline demographic identities for various commercial models and the relationship between the baseline and other demographics. We strive to understand if these biases are positive, neutral, or negative, and the strength of these biases. This paper can contribute to the objective assessment of bias in Large Language Models and give the user or developer the power to account for these biases in LLM output or in training future LLMs. Our analysis suggested two key findings: that models view the baseline demographic as a white middle-aged or young adult male; however, a general trend across models suggested that non-baseline demographics are more willing to help than the baseline. These methodologies allowed us to distinguish these two biases that are often tangled together. 

---
# Do We Still Need Audio? Rethinking Speaker Diarization with a Text-Based Approach Using Multiple Prediction Models 

**Authors**: Peilin Wu, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11344)  

**Abstract**: We present a novel approach to Speaker Diarization (SD) by leveraging text-based methods focused on Sentence-level Speaker Change Detection within dialogues. Unlike audio-based SD systems, which are often challenged by audio quality and speaker similarity, our approach utilizes the dialogue transcript alone. Two models are developed: the Single Prediction Model (SPM) and the Multiple Prediction Model (MPM), both of which demonstrate significant improvements in identifying speaker changes, particularly in short conversations. Our findings, based on a curated dataset encompassing diverse conversational scenarios, reveal that the text-based SD approach, especially the MPM, performs competitively against state-of-the-art audio-based SD systems, with superior performance in short conversational contexts. This paper not only showcases the potential of leveraging linguistic features for SD but also highlights the importance of integrating semantic understanding into SD systems, opening avenues for future research in multimodal and semantic feature-based diarization. 

---
# From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review 

**Authors**: Yaohui Zhang, Haijing Zhang, Wenlong Ji, Tianyu Hua, Nick Haber, Hancheng Cao, Weixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11343)  

**Abstract**: The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity. 

---
# Surprisal from Larger Transformer-based Language Models Predicts fMRI Data More Poorly 

**Authors**: Yi-Chien Lin, William Schuler  

**Link**: [PDF](https://arxiv.org/pdf/2506.11338)  

**Abstract**: As Transformers become more widely incorporated into natural language processing tasks, there has been considerable interest in using surprisal from these models as predictors of human sentence processing difficulty. Recent work has observed a positive relationship between Transformer-based models' perplexity and the predictive power of their surprisal estimates on reading times, showing that language models with more parameters and trained on more data are less predictive of human reading times. However, these studies focus on predicting latency-based measures (i.e., self-paced reading times and eye-gaze durations) with surprisal estimates from Transformer-based language models. This trend has not been tested on brain imaging data. This study therefore evaluates the predictive power of surprisal estimates from 17 pre-trained Transformer-based models across three different language families on two functional magnetic resonance imaging datasets. Results show that the positive relationship between model perplexity and model fit still obtains, suggesting that this trend is not specific to latency-based measures and can be generalized to neural measures. 

---
# Don't Pay Attention 

**Authors**: Mohammad Hammoud, Devang Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2506.11305)  

**Abstract**: The Transformer has become the de facto standard for large language models and a wide range of downstream tasks across various domains. Despite its numerous advantages like inherent training parallelism, the Transformer still faces key challenges due to its inability to effectively process sequences beyond a fixed context window and the quadratic complexity of its attention mechanism. These challenges have renewed interest in RNN-like architectures, which offer linear scaling with sequence length and improved handling of long-range dependencies, albeit with limited parallelism due to their inherently recurrent nature. In this paper, we propose Avey, a new neural foundational architecture that breaks away from both attention and recurrence. Avey comprises a ranker and an autoregressive neural processor, which collaboratively identify and contextualize only the most relevant tokens for any given token, regardless of their positions in the sequence. Specifically, Avey decouples sequence length from context width, thus enabling effective processing of arbitrarily long sequences. Experimental results show that Avey compares favorably to the Transformer across a variety of standard short-range NLP benchmarks, while notably excelling at capturing long-range dependencies. 

---
# Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning 

**Authors**: Yang Zhang, Amr Mohamed, Hadi Abdine, Guokan Shang, Michalis Vazirgiannis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11300)  

**Abstract**: Curriculum learning has shown promise in improving training efficiency and generalization in various machine learning domains, yet its potential in pretraining language models remains underexplored, prompting our work as the first systematic investigation in this area. We experimented with different settings, including vanilla curriculum learning, pacing-based sampling, and interleaved curricula-guided by six difficulty metrics spanning linguistic and information-theoretic perspectives. We train models under these settings and evaluate their performance on eight diverse benchmarks. Our experiments reveal that curriculum learning consistently improves convergence in early and mid-training phases, and can yield lasting gains when used as a warmup strategy with up to $3.5\%$ improvement. Notably, we identify compression ratio, lexical diversity, and readability as effective difficulty signals across settings. Our findings highlight the importance of data ordering in large-scale pretraining and provide actionable insights for scalable, data-efficient model development under realistic training scenarios. 

---
# Learning a Continue-Thinking Token for Enhanced Test-Time Scaling 

**Authors**: Liran Ringel, Elad Tolochinsky, Yaniv Romano  

**Link**: [PDF](https://arxiv.org/pdf/2506.11274)  

**Abstract**: Test-time scaling has emerged as an effective approach for improving language model performance by utilizing additional compute at inference time. Recent studies have shown that overriding end-of-thinking tokens (e.g., replacing "</think>" with "Wait") can extend reasoning steps and improve accuracy. In this work, we explore whether a dedicated continue-thinking token can be learned to trigger extended reasoning. We augment a distilled version of DeepSeek-R1 with a single learned "<|continue-thinking|>" token, training only its embedding via reinforcement learning while keeping the model weights frozen. Our experiments show that this learned token achieves improved accuracy on standard math benchmarks compared to both the baseline model and a test-time scaling approach that uses a fixed token (e.g., "Wait") for budget forcing. In particular, we observe that in cases where the fixed-token approach enhances the base model's accuracy, our method achieves a markedly greater improvement. For example, on the GSM8K benchmark, the fixed-token approach yields a 1.3% absolute improvement in accuracy, whereas our learned-token method achieves a 4.2% improvement over the base model that does not use budget forcing. 

---
# No Universal Prompt: Unifying Reasoning through Adaptive Prompting for Temporal Table Reasoning 

**Authors**: Kushagra Dixit, Abhishek Rajgaria, Harshavardhan Kalalbandi, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.11246)  

**Abstract**: Temporal Table Reasoning is a critical challenge for Large Language Models (LLMs), requiring effective prompting techniques to extract relevant insights. Despite existence of multiple prompting methods, their impact on table reasoning remains largely unexplored. Furthermore, the performance of these models varies drastically across different table and context structures, making it difficult to determine an optimal approach. This work investigates multiple prompting technique across diverse table types to determine optimal approaches for different scenarios. We find that performance varies based on entity type, table structure, requirement of additional context and question complexity, with NO single method consistently outperforming others. To mitigate these challenges, we introduce SEAR, an adaptive prompting framework inspired by human reasoning that dynamically adjusts based on context characteristics and integrates a structured reasoning. Our results demonstrate that SEAR achieves superior performance across all table types compared to other baseline prompting techniques. Additionally, we explore the impact of table structure refactoring, finding that a unified representation enhances model's reasoning. 

---
# Iterative Multilingual Spectral Attribute Erasure 

**Authors**: Shun Shao, Yftah Ziser, Zheng Zhao, Yifu Qiu, Shay B. Cohen, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11244)  

**Abstract**: Multilingual representations embed words with similar meanings to share a common semantic space across languages, creating opportunities to transfer debiasing effects between languages. However, existing methods for debiasing are unable to exploit this opportunity because they operate on individual languages. We present Iterative Multilingual Spectral Attribute Erasure (IMSAE), which identifies and mitigates joint bias subspaces across multiple languages through iterative SVD-based truncation. Evaluating IMSAE across eight languages and five demographic dimensions, we demonstrate its effectiveness in both standard and zero-shot settings, where target language data is unavailable, but linguistically similar languages can be used for debiasing. Our comprehensive experiments across diverse language models (BERT, LLaMA, Mistral) show that IMSAE outperforms traditional monolingual and cross-lingual approaches while maintaining model utility. 

---
# RETUYT-INCO at BEA 2025 Shared Task: How Far Can Lightweight Models Go in AI-powered Tutor Evaluation? 

**Authors**: Santiago Góngora, Ignacio Sastre, Santiago Robaina, Ignacio Remersaro, Luis Chiruzzo, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2506.11243)  

**Abstract**: In this paper, we present the RETUYT-INCO participation at the BEA 2025 shared task. Our participation was characterized by the decision of using relatively small models, with fewer than 1B parameters. This self-imposed restriction tries to represent the conditions in which many research labs or institutions are in the Global South, where computational power is not easily accessible due to its prohibitive cost. Even under this restrictive self-imposed setting, our models managed to stay competitive with the rest of teams that participated in the shared task. According to the $exact\ F_1$ scores published by the organizers, the performance gaps between our models and the winners were as follows: $6.46$ in Track 1; $10.24$ in Track 2; $7.85$ in Track 3; $9.56$ in Track 4; and $13.13$ in Track 5. Considering that the minimum difference with a winner team is $6.46$ points -- and the maximum difference is $13.13$ -- according to the $exact\ F_1$ score, we find that models with a size smaller than 1B parameters are competitive for these tasks, all of which can be run on computers with a low-budget GPU or even without a GPU. 

---
# Scalable Medication Extraction and Discontinuation Identification from Electronic Health Records Using Large Language Models 

**Authors**: Chong Shao, Douglas Snyder, Chiran Li, Bowen Gu, Kerry Ngan, Chun-Ting Yang, Jiageng Wu, Richard Wyss, Kueiyu Joshua Lin, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11137)  

**Abstract**: Identifying medication discontinuations in electronic health records (EHRs) is vital for patient safety but is often hindered by information being buried in unstructured notes. This study aims to evaluate the capabilities of advanced open-sourced and proprietary large language models (LLMs) in extracting medications and classifying their medication status from EHR notes, focusing on their scalability on medication information extraction without human annotation. We collected three EHR datasets from diverse sources to build the evaluation benchmark. We evaluated 12 advanced LLMs and explored multiple LLM prompting strategies. Performance on medication extraction, medication status classification, and their joint task (extraction then classification) was systematically compared across all experiments. We found that LLMs showed promising performance on the medication extraction and discontinuation classification from EHR notes. GPT-4o consistently achieved the highest average F1 scores in all tasks under zero-shot setting - 94.0% for medication extraction, 78.1% for discontinuation classification, and 72.7% for the joint task. Open-sourced models followed closely, Llama-3.1-70B-Instruct achieved the highest performance in medication status classification on the MIV-Med dataset (68.7%) and in the joint task on both the Re-CASI (76.2%) and MIV-Med (60.2%) datasets. Medical-specific LLMs demonstrated lower performance compared to advanced general-domain LLMs. Few-shot learning generally improved performance, while CoT reasoning showed inconsistent gains. LLMs demonstrate strong potential for medication extraction and discontinuation identification on EHR notes, with open-sourced models offering scalable alternatives to proprietary systems and few-shot can further improve LLMs' capability. 

---
# Large Language Models and Emergence: A Complex Systems Perspective 

**Authors**: David C. Krakauer, John W. Krakauer, Melanie Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2506.11135)  

**Abstract**: Emergence is a concept in complexity science that describes how many-body systems manifest novel higher-level properties, properties that can be described by replacing high-dimensional mechanisms with lower-dimensional effective variables and theories. This is captured by the idea "more is different". Intelligence is a consummate emergent property manifesting increasingly efficient -- cheaper and faster -- uses of emergent capabilities to solve problems. This is captured by the idea "less is more". In this paper, we first examine claims that Large Language Models exhibit emergent capabilities, reviewing several approaches to quantifying emergence, and secondly ask whether LLMs possess emergent intelligence. 

---
# A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data 

**Authors**: Cheng Kang Chou, Chan-Jan Hsu, Ho-Lam Chung, Liang-Hsuan Tseng, Hsi-Chun Cheng, Yu-Kuan Fu, Kuan Po Huang, Hung-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.11130)  

**Abstract**: We propose a self-refining framework that enhances ASR performance with only unlabeled datasets. The process starts with an existing ASR model generating pseudo-labels on unannotated speech, which are then used to train a high-fidelity text-to-speech (TTS) system. Then, synthesized speech text pairs are bootstrapped into the original ASR system, completing the closed-loop self-improvement cycle. We demonstrated the effectiveness of the framework on Taiwanese Mandarin speech. Leveraging 6,000 hours of unlabeled speech, a moderate amount of text data, and synthetic content from the AI models, we adapt Whisper-large-v2 into a specialized model, Twister. Twister reduces error rates by up to 20% on Mandarin and 50% on Mandarin-English code-switching benchmarks compared to Whisper. Results highlight the framework as a compelling alternative to pseudo-labeling self-distillation approaches and provides a practical pathway for improving ASR performance in low-resource or domain-specific settings. 

---
# Trustworthy AI for Medicine: Continuous Hallucination Detection and Elimination with CHECK 

**Authors**: Carlos Garcia-Fernandez, Luis Felipe, Monique Shotande, Muntasir Zitu, Aakash Tripathi, Ghulam Rasool, Issam El Naqa, Vivek Rudrapatna, Gilmer Valdes  

**Link**: [PDF](https://arxiv.org/pdf/2506.11129)  

**Abstract**: Large language models (LLMs) show promise in healthcare, but hallucinations remain a major barrier to clinical use. We present CHECK, a continuous-learning framework that integrates structured clinical databases with a classifier grounded in information theory to detect both factual and reasoning-based hallucinations. Evaluated on 1500 questions from 100 pivotal clinical trials, CHECK reduced LLama3.3-70B-Instruct hallucination rates from 31% to 0.3% - making an open source model state of the art. Its classifier generalized across medical benchmarks, achieving AUCs of 0.95-0.96, including on the MedQA (USMLE) benchmark and HealthBench realistic multi-turn medical questioning. By leveraging hallucination probabilities to guide GPT-4o's refinement and judiciously escalate compute, CHECK boosted its USMLE passing rate by 5 percentage points, achieving a state-of-the-art 92.1%. By suppressing hallucinations below accepted clinical error thresholds, CHECK offers a scalable foundation for safe LLM deployment in medicine and other high-stakes domains. 

---
# Stronger Language Models Produce More Human-Like Errors 

**Authors**: Andrew Keenan Richardson, Ryan Othniel Kearns, Sean Moss, Vincent Wang-Mascianica, Philipp Koralus  

**Link**: [PDF](https://arxiv.org/pdf/2506.11128)  

**Abstract**: Do language models converge toward human-like reasoning patterns as they improve? We provide surprising evidence that while overall reasoning capabilities increase with model sophistication, the nature of errors increasingly mirrors predictable human reasoning fallacies: a previously unobserved inverse scaling phenomenon. To investigate this question, we apply the Erotetic Theory of Reasoning (ETR), a formal cognitive framework with empirical support for predicting human reasoning outcomes. Using the open-source package PyETR, we generate logical reasoning problems where humans predictably err, evaluating responses from 38 language models across 383 reasoning tasks. Our analysis indicates that as models advance in general capability (as measured by Chatbot Arena scores), the proportion of their incorrect answers that align with ETR-predicted human fallacies tends to increase ($\rho = 0.360, p = 0.0265$). Notably, as we observe no correlation between model sophistication and logical correctness on these tasks, this shift in error patterns toward human-likeness occurs independently of error rate. These findings challenge the prevailing view that scaling language models naturally obtains normative rationality, suggesting instead a convergence toward human-like cognition inclusive of our characteristic biases and limitations, as we further confirm by demonstrating order-effects in language model reasoning. 

---
# GUIRoboTron-Speech: Towards Automated GUI Agents Based on Speech Instructions 

**Authors**: Wenkang Han, Zhixiong Zeng, Jing Huang, Shu Jiang, Liming Zheng, Longrong Yang, Haibo Qiu, Chang Yao, Jingyuan Chen, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.11127)  

**Abstract**: Autonomous agents for Graphical User Interfaces (GUIs) are revolutionizing human-computer interaction, yet their reliance on text-based instructions imposes limitations on accessibility and convenience, particularly in hands-free scenarios. To address this gap, we propose GUIRoboTron-Speech, the first end-to-end autonomous GUI agent that directly accepts speech instructions and on-device screenshots to predict actions. Confronted with the scarcity of speech-based GUI agent datasets, we initially generated high-quality speech instructions for training by leveraging a random timbre text-to-speech (TTS) model to convert existing text instructions. We then develop GUIRoboTron-Speech's capabilities through progressive grounding and planning training stages. A key contribution is a heuristic mixed-instruction training strategy designed to mitigate the modality imbalance inherent in pre-trained foundation models. Comprehensive experiments on several benchmark datasets validate the robust and superior performance of GUIRoboTron-Speech, demonstrating the significant potential and widespread applicability of speech as an effective instruction modality for driving GUI agents. Our code and datasets are available at this https URL. 

---
# ASRJam: Human-Friendly AI Speech Jamming to Prevent Automated Phone Scams 

**Authors**: Freddie Grabovski, Gilad Gressel, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.11125)  

**Abstract**: Large Language Models (LLMs), combined with Text-to-Speech (TTS) and Automatic Speech Recognition (ASR), are increasingly used to automate voice phishing (vishing) scams. These systems are scalable and convincing, posing a significant security threat. We identify the ASR transcription step as the most vulnerable link in the scam pipeline and introduce ASRJam, a proactive defence framework that injects adversarial perturbations into the victim's audio to disrupt the attacker's ASR. This breaks the scam's feedback loop without affecting human callers, who can still understand the conversation. While prior adversarial audio techniques are often unpleasant and impractical for real-time use, we also propose EchoGuard, a novel jammer that leverages natural distortions, such as reverberation and echo, that are disruptive to ASR but tolerable to humans. To evaluate EchoGuard's effectiveness and usability, we conducted a 39-person user study comparing it with three state-of-the-art attacks. Results show that EchoGuard achieved the highest overall utility, offering the best combination of ASR disruption and human listening experience. 

---
# SUTA-LM: Bridging Test-Time Adaptation and Language Model Rescoring for Robust ASR 

**Authors**: Wei-Ping Huang, Guan-Ting Lin, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.11121)  

**Abstract**: Despite progress in end-to-end ASR, real-world domain mismatches still cause performance drops, which Test-Time Adaptation (TTA) aims to mitigate by adjusting models during inference. Recent work explores combining TTA with external language models, using techniques like beam search rescoring or generative error correction. In this work, we identify a previously overlooked challenge: TTA can interfere with language model rescoring, revealing the nontrivial nature of effectively combining the two methods. Based on this insight, we propose SUTA-LM, a simple yet effective extension of SUTA, an entropy-minimization-based TTA approach, with language model rescoring. SUTA-LM first applies a controlled adaptation process guided by an auto-step selection mechanism leveraging both acoustic and linguistic information, followed by language model rescoring to refine the outputs. Experiments on 18 diverse ASR datasets show that SUTA-LM achieves robust results across a wide range of domains. 

---
# SDMPrune: Self-Distillation MLP Pruning for Efficient Large Language Models 

**Authors**: Hourun Zhu, Chengchao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11120)  

**Abstract**: In spite of strong performance achieved by LLMs, the costs of their deployment are unaffordable. For the compression of LLMs, gradient-based pruning methods present promising effectiveness. However, in these methods, the gradient computation with one-hot labels ignore the potential predictions on other words, thus missing key information for generative capability of the original model. To address this issue, we introduce a self-distillation loss during the pruning phase (rather than post-training) to fully exploit the predictions of the original model, thereby obtaining more accurate gradient information for pruning. Moreover, we find that, compared to attention modules, the predictions of LLM are less sensitive to multilayer perceptron (MLP) modules, which take up more than $5 \times$ parameters (LLaMA3.2-1.2B). To this end, we focus on the pruning of MLP modules, to significantly compress LLM without obvious performance degradation. Experimental results on extensive zero-shot benchmarks demonstrate that our method significantly outperforms existing pruning methods. Furthermore, our method achieves very competitive performance among 1B-scale open source LLMs. The source code and trained weights are available at this https URL. 

---
# Benchmarking Foundation Speech and Language Models for Alzheimer's Disease and Related Dementia Detection from Spontaneous Speech 

**Authors**: Jingyu Li, Lingchao Mao, Hairong Wang, Zhendong Wang, Xi Mao, Xuelei Sherry Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.11119)  

**Abstract**: Background: Alzheimer's disease and related dementias (ADRD) are progressive neurodegenerative conditions where early detection is vital for timely intervention and care. Spontaneous speech contains rich acoustic and linguistic markers that may serve as non-invasive biomarkers for cognitive decline. Foundation models, pre-trained on large-scale audio or text data, produce high-dimensional embeddings encoding contextual and acoustic features.
Methods: We used the PREPARE Challenge dataset, which includes audio recordings from over 1,600 participants with three cognitive statuses: healthy control (HC), mild cognitive impairment (MCI), and Alzheimer's Disease (AD). We excluded non-English, non-spontaneous, or poor-quality recordings. The final dataset included 703 (59.13%) HC, 81 (6.81%) MCI, and 405 (34.06%) AD cases. We benchmarked a range of open-source foundation speech and language models to classify cognitive status into the three categories.
Results: The Whisper-medium model achieved the highest performance among speech models (accuracy = 0.731, AUC = 0.802). Among language models, BERT with pause annotation performed best (accuracy = 0.662, AUC = 0.744). ADRD detection using state-of-the-art automatic speech recognition (ASR) model-generated audio embeddings outperformed others. Including non-semantic features like pause patterns consistently improved text-based classification.
Conclusion: This study introduces a benchmarking framework using foundation models and a clinically relevant dataset. Acoustic-based approaches -- particularly ASR-derived embeddings -- demonstrate strong potential for scalable, non-invasive, and cost-effective early detection of ADRD. 

---
# ScIRGen: Synthesize Realistic and Large-Scale RAG Dataset for Scientific Research 

**Authors**: Junyong Lin, Lu Dai, Ruiqian Han, Yijie Sui, Ruilin Wang, Xingliang Sun, Qinglin Wu, Min Feng, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11117)  

**Abstract**: Scientific researchers need intensive information about datasets to effectively evaluate and develop theories and methodologies. The information needs regarding datasets are implicitly embedded in particular research tasks, rather than explicitly expressed in search queries. However, existing scientific retrieval and question-answering (QA) datasets typically address straightforward questions, which do not align with the distribution of real-world research inquiries. To bridge this gap, we developed ScIRGen, a dataset generation framework for scientific QA \& retrieval that more accurately reflects the information needs of professional science researchers, and uses it to create a large-scale scientific retrieval-augmented generation (RAG) dataset with realistic queries, datasets and papers. Technically, we designed a dataset-oriented information extraction method that leverages academic papers to augment the dataset representation. We then proposed a question generation framework by employing cognitive taxonomy to ensure the quality of synthesized questions. We also design a method to automatically filter synthetic answers based on the perplexity shift of LLMs, which is highly aligned with human judgment of answers' validity. Collectively, these methodologies culminated in the creation of the 61k QA dataset, ScIRGen-Geo. We benchmarked representative methods on the ScIRGen-Geo dataset for their question-answering and retrieval capabilities, finding out that current methods still suffer from reasoning from complex questions. This work advances the development of more sophisticated tools to support the intricate information needs of the scientific community. 

---
# Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models 

**Authors**: Jijie Li, Li Du, Hanyu Zhao, Bo-wen Zhang, Liangdong Wang, Boyan Gao, Guang Liu, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11116)  

**Abstract**: Large Language Models (LLMs) demonstrate strong performance in real-world applications, yet existing open-source instruction datasets often concentrate on narrow domains, such as mathematics or coding, limiting generalization and widening the gap with proprietary models. To bridge this gap, we introduce Infinity-Instruct, a high-quality instruction dataset designed to enhance both foundational and chat capabilities of LLMs through a two-phase pipeline. In Phase 1, we curate 7.4M high-quality foundational instructions (InfInstruct-F-7.4M) from over 100M samples using hybrid data selection techniques. In Phase 2, we synthesize 1.5M high-quality chat instructions (InfInstruct-G-1.5M) through a two-stage process involving instruction selection, evolution, and diagnostic filtering. We empirically evaluate Infinity-Instruct by fine-tuning several open-source models, including Mistral, LLaMA, Qwen, and Yi, and observe substantial performance gains across both foundational and instruction following benchmarks, consistently surpassing official instruction-tuned counterparts. Notably, InfInstruct-LLaMA3.1-70B outperforms GPT-4-0314 by 8.6\% on instruction following tasks while achieving comparable foundational performance. These results underscore the synergy between foundational and chat training and offer new insights into holistic LLM development. Our dataset\footnote{this https URL} and codes\footnote{this https URL} have been publicly released. 

---
# Incorporating Domain Knowledge into Materials Tokenization 

**Authors**: Yerim Oh, Jun-Hyung Park, Junho Kim, SungHo Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.11115)  

**Abstract**: While language models are increasingly utilized in materials science, typical models rely on frequency-centric tokenization methods originally developed for natural language processing. However, these methods frequently produce excessive fragmentation and semantic loss, failing to maintain the structural and semantic integrity of material concepts. To address this issue, we propose MATTER, a novel tokenization approach that integrates material knowledge into tokenization. Based on MatDetector trained on our materials knowledge base and a re-ranking method prioritizing material concepts in token merging, MATTER maintains the structural integrity of identified material concepts and prevents fragmentation during tokenization, ensuring their semantic meaning remains intact. The experimental results demonstrate that MATTER outperforms existing tokenization methods, achieving an average performance gain of $4\%$ and $2\%$ in the generation and classification tasks, respectively. These results underscore the importance of domain knowledge for tokenization strategies in scientific text processing. Our code is available at this https URL 

---
# KokushiMD-10: Benchmark for Evaluating Large Language Models on Ten Japanese National Healthcare Licensing Examinations 

**Authors**: Junyu Liu, Kaiqi Yan, Tianyang Wang, Qian Niu, Momoko Nagai-Tanima, Tomoki Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.11114)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated notable performance in medical licensing exams. However, comprehensive evaluation of LLMs across various healthcare roles, particularly in high-stakes clinical scenarios, remains a challenge. Existing benchmarks are typically text-based, English-centric, and focus primarily on medicines, which limits their ability to assess broader healthcare knowledge and multimodal reasoning. To address these gaps, we introduce KokushiMD-10, the first multimodal benchmark constructed from ten Japanese national healthcare licensing exams. This benchmark spans multiple fields, including Medicine, Dentistry, Nursing, Pharmacy, and allied health professions. It contains over 11588 real exam questions, incorporating clinical images and expert-annotated rationales to evaluate both textual and visual reasoning. We benchmark over 30 state-of-the-art LLMs, including GPT-4o, Claude 3.5, and Gemini, across both text and image-based settings. Despite promising results, no model consistently meets passing thresholds across domains, highlighting the ongoing challenges in medical AI. KokushiMD-10 provides a comprehensive and linguistically grounded resource for evaluating and advancing reasoning-centric medical AI across multilingual and multimodal clinical tasks. 

---
# Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks 

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2506.11113)  

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication. 

---
# Manifesto from Dagstuhl Perspectives Workshop 24352 -- Conversational Agents: A Framework for Evaluation (CAFE) 

**Authors**: Christine Bauer, Li Chen, Nicola Ferro, Norbert Fuhr, Avishek Anand, Timo Breuer, Guglielmo Faggioli, Ophir Frieder, Hideo Joho, Jussi Karlgren, Johannes Kiesel, Bart P. Knijnenburg, Aldo Lipani, Lien Michiels, Andrea Papenmeier, Maria Soledad Pera, Mark Sanderson, Scott Sanner, Benno Stein, Johanne R. Trippas, Karin Verspoor, Martijn C Willemsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11112)  

**Abstract**: During the workshop, we deeply discussed what CONversational Information ACcess (CONIAC) is and its unique features, proposing a world model abstracting it, and defined the Conversational Agents Framework for Evaluation (CAFE) for the evaluation of CONIAC systems, consisting of six major components: 1) goals of the system's stakeholders, 2) user tasks to be studied in the evaluation, 3) aspects of the users carrying out the tasks, 4) evaluation criteria to be considered, 5) evaluation methodology to be applied, and 6) measures for the quantitative criteria chosen. 

---
# Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions 

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11111)  

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (this https URL) to support the community. 

---
# AssertBench: A Benchmark for Evaluating Self-Assertion in Large Language Models 

**Authors**: Jaeho Lee, Atharv Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.11110)  

**Abstract**: Recent benchmarks have probed factual consistency and rhetorical robustness in Large Language Models (LLMs). However, a knowledge gap exists regarding how directional framing of factually true statements influences model agreement, a common scenario for LLM users. AssertBench addresses this by sampling evidence-supported facts from FEVEROUS, a fact verification dataset. For each (evidence-backed) fact, we construct two framing prompts: one where the user claims the statement is factually correct, and another where the user claims it is incorrect. We then record the model's agreement and reasoning. The desired outcome is that the model asserts itself, maintaining consistent truth evaluation across both framings, rather than switching its evaluation to agree with the user. AssertBench isolates framing-induced variability from the model's underlying factual knowledge by stratifying results based on the model's accuracy on the same claims when presented neutrally. In doing so, this benchmark aims to measure an LLM's ability to "stick to its guns" when presented with contradictory user assertions about the same fact. The complete source code is available at this https URL. 

---
# Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization 

**Authors**: Yile Chen, Yicheng Tao, Yue Jiang, Shuai Liu, Han Yu, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11109)  

**Abstract**: The widespread adoption of location-based services has led to the generation of vast amounts of mobility data, providing significant opportunities to model user movement dynamics within urban environments. Recent advancements have focused on adapting Large Language Models (LLMs) for mobility analytics. However, existing methods face two primary limitations: inadequate semantic representation of locations (i.e., discrete IDs) and insufficient modeling of mobility signals within LLMs (i.e., single templated instruction fine-tuning). To address these issues, we propose QT-Mob, a novel framework that significantly enhances LLMs for mobility analytics. QT-Mob introduces a location tokenization module that learns compact, semantically rich tokens to represent locations, preserving contextual information while ensuring compatibility with LLMs. Furthermore, QT-Mob incorporates a series of complementary fine-tuning objectives that align the learned tokens with the internal representations in LLMs, improving the model's comprehension of sequential movement patterns and location semantics. The proposed QT-Mob framework not only enhances LLMs' ability to interpret mobility data but also provides a more generalizable approach for various mobility analytics tasks. Experiments on three real-world dataset demonstrate the superior performance in both next-location prediction and mobility recovery tasks, outperforming existing deep learning and LLM-based methods. 

---
# History-Aware Cross-Attention Reinforcement: Self-Supervised Multi Turn and Chain-of-Thought Fine-Tuning with vLLM 

**Authors**: Andrew Kiruluta, Andreas Lemos, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2506.11108)  

**Abstract**: We present CAGSR-vLLM-MTC, an extension of our Self-Supervised Cross-Attention-Guided Reinforcement (CAGSR) framework, now implemented on the high-performance vLLM runtime, to address both multi-turn dialogue and chain-of-thought reasoning. Building upon our original single-turn approach, we first instrumented vLLM's C++/CUDA kernels to asynchronously capture per-layer, per-head cross-attention weights during generation. We then generalized our self-supervised reward function to accumulate attention signals over entire conversation histories and intermediate chain-of-thought steps. We discuss practical trade-offs, including an entropy-based clamping mechanism to prevent attention collapse on early context, and outline future directions for multi-party dialogues and hierarchical reasoning. 

---
# Graph-based RAG Enhancement via Global Query Disambiguation and Dependency-Aware Reranking 

**Authors**: Ningyuan Li, Junrui Liu, Yi Shan, Minghui Huang, Tong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11106)  

**Abstract**: Contemporary graph-based retrieval-augmented generation (RAG) methods typically begin by extracting entities from user queries and then leverage pre-constructed knowledge graphs to retrieve related relationships and metadata. However, this pipeline's exclusive reliance on entity-level extraction can lead to the misinterpretation or omission of latent yet critical information and relations. As a result, retrieved content may be irrelevant or contradictory, and essential knowledge may be excluded, exacerbating hallucination risks and degrading the fidelity of generated responses. To address these limitations, we introduce PankRAG, a framework that combines a globally aware, hierarchical query-resolution strategy with a novel dependency-aware reranking mechanism. PankRAG first constructs a multi-level resolution path that captures both parallel and sequential interdependencies within a query, guiding large language models (LLMs) through structured reasoning. It then applies its dependency-aware reranker to exploit the dependency structure among resolved sub-questions, enriching and validating retrieval results for subsequent sub-questions. Empirical evaluations demonstrate that PankRAG consistently outperforms state-of-the-art approaches across multiple benchmarks, underscoring its robustness and generalizability. 

---
# Enabling On-Device Medical AI Assistants via Input-Driven Saliency Adaptation 

**Authors**: Uttej Kallakurik, Edward Humes, Rithvik Jonna, Xiaomin Lin, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11105)  

**Abstract**: Large Language Models (LLMs) have significant impact on the healthcare scenarios but remain prohibitively large for deployment in real-time, resource-constrained environments such as edge devices. In this work, we introduce a novel medical assistant system, optimized through our general-purpose compression framework, which tailors Large Language Models (LLMs) for deployment in specialized domains. By measuring neuron saliency on domain-specific data, our method can aggressively prune irrelevant neurons, reducing model size while preserving performance. Following pruning, we apply post-training quantization to further reduce the memory footprint, and evaluate the compressed model across medical benchmarks including MedMCQA, MedQA, and PubMedQA. We also deploy the 50\% compressed Gemma and the 67\% compressed LLaMA3 models on Jetson Orin Nano (18.7W peak) and Raspberry Pi 5 (6.3W peak), achieving real-time, energy-efficient inference under hardware constraints. 

---
# DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration 

**Authors**: Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11104)  

**Abstract**: Long-context understanding is crucial for many NLP applications, yet transformers struggle with efficiency due to the quadratic complexity of self-attention. Sparse attention methods alleviate this cost but often impose static, predefined masks, failing to capture heterogeneous attention patterns. This results in suboptimal token interactions, limiting adaptability and retrieval accuracy in long-sequence tasks. This work introduces a dynamic sparse attention mechanism that assigns adaptive masks at the attention-map level, preserving heterogeneous patterns across layers and heads. Unlike existing approaches, our method eliminates the need for fine-tuning and predefined mask structures while maintaining computational efficiency. By learning context-aware attention structures, it achieves high alignment with full-attention models, ensuring minimal performance degradation while reducing memory and compute overhead. This approach provides a scalable alternative to full attention, enabling the practical deployment of large-scale Large Language Models (LLMs) without sacrificing retrieval performance. DAM is available at: this https URL. 

---
# You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Model 

**Authors**: Wenchong He, Liqian Peng, Zhe Jiang, Alex Go  

**Link**: [PDF](https://arxiv.org/pdf/2506.11103)  

**Abstract**: Large language models (LLMs) possess a remarkable ability to perform in-context learning (ICL), which enables them to handle multiple downstream tasks simultaneously without requiring task-specific fine-tuning. Recent studies have shown that even moderately sized LLMs, such as Mistral 7B, Gemma 7B and Llama-3 8B, can achieve ICL through few-shot in-context fine-tuning of all tasks at once. However, this approach still lags behind dedicated fine-tuning, where a separate model is trained for each individual task.
In this paper, we propose a novel approach, Many-Shot In-Context Fine-tuning (ManyICL), which significantly narrows this performance gap by extending the principles of ICL to a many-shot setting. To unlock the full potential of ManyICL and address the inherent inefficiency of processing long sequences with numerous in-context examples, we propose a novel training objective. Instead of solely predicting the final answer, our approach treats every answer within the context as a supervised training target. This effectively shifts the role of many-shot examples from prompts to targets for autoregressive learning. Through extensive experiments on diverse downstream tasks, including classification, summarization, question answering, natural language inference, and math, we demonstrate that ManyICL substantially outperforms zero/few-shot fine-tuning and approaches the performance of dedicated fine-tuning. Furthermore, ManyICL significantly mitigates catastrophic forgetting issues observed in zero/few-shot fine-tuning. The code will be made publicly available upon publication. 

---
# Evolutionary Perspectives on the Evaluation of LLM-Based AI Agents: A Comprehensive Survey 

**Authors**: Jiachen Zhu, Menghui Zhu, Renting Rui, Rong Shan, Congmin Zheng, Bo Chen, Yunjia Xi, Jianghao Lin, Weiwen Liu, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11102)  

**Abstract**: The advent of large language models (LLMs), such as GPT, Gemini, and DeepSeek, has significantly advanced natural language processing, giving rise to sophisticated chatbots capable of diverse language-related tasks. The transition from these traditional LLM chatbots to more advanced AI agents represents a pivotal evolutionary step. However, existing evaluation frameworks often blur the distinctions between LLM chatbots and AI agents, leading to confusion among researchers selecting appropriate benchmarks. To bridge this gap, this paper introduces a systematic analysis of current evaluation approaches, grounded in an evolutionary perspective. We provide a detailed analytical framework that clearly differentiates AI agents from LLM chatbots along five key aspects: complex environment, multi-source instructor, dynamic feedback, multi-modal perception, and advanced capability. Further, we categorize existing evaluation benchmarks based on external environments driving forces, and resulting advanced internal capabilities. For each category, we delineate relevant evaluation attributes, presented comprehensively in practical reference tables. Finally, we synthesize current trends and outline future evaluation methodologies through four critical lenses: environment, agent, evaluator, and metrics. Our findings offer actionable guidance for researchers, facilitating the informed selection and application of benchmarks in AI agent evaluation, thus fostering continued advancement in this rapidly evolving research domain. 

---
# C-SEO Bench: Does Conversational SEO Work? 

**Authors**: Haritz Puerto, Martin Gubri, Tommaso Green, Seong Joon Oh, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11097)  

**Abstract**: Large Language Models (LLMs) are transforming search engines into Conversational Search Engines (CSE). Consequently, Search Engine Optimization (SEO) is being shifted into Conversational Search Engine Optimization (C-SEO). We are beginning to see dedicated C-SEO methods for modifying web documents to increase their visibility in CSE responses. However, they are often tested only for a limited breadth of application domains; we do not understand whether certain C-SEO methods would be effective for a broad range of domains. Moreover, existing evaluations consider only a single-actor scenario where only one web document adopts a C-SEO method; in reality, multiple players are likely to competitively adopt the cutting-edge C-SEO techniques, drawing an analogy from the dynamics we have seen in SEO. We present C-SEO Bench, the first benchmark designed to evaluate C-SEO methods across multiple tasks, domains, and number of actors. We consider two search tasks, question answering and product recommendation, with three domains each. We also formalize a new evaluation protocol with varying adoption rates among involved actors. Our experiments reveal that most current C-SEO methods are largely ineffective, contrary to reported results in the literature. Instead, traditional SEO strategies, those aiming to improve the ranking of the source in the LLM context, are significantly more effective. We also observe that as we increase the number of C-SEO adopters, the overall gains decrease, depicting a congested and zero-sum nature of the problem. Our code and data are available at this https URL and this https URL. 

---
# Persistent Homology of Topic Networks for the Prediction of Reader Curiosity 

**Authors**: Manuel D. S. Hopp, Vincent Labatut, Arthur Amalvy, Richard Dufour, Hannah Stone, Hayley Jach, Kou Murayama  

**Link**: [PDF](https://arxiv.org/pdf/2506.11095)  

**Abstract**: Reader curiosity, the drive to seek information, is crucial for textual engagement, yet remains relatively underexplored in NLP. Building on Loewenstein's Information Gap Theory, we introduce a framework that models reader curiosity by quantifying semantic information gaps within a text's semantic structure. Our approach leverages BERTopic-inspired topic modeling and persistent homology to analyze the evolving topology (connected components, cycles, voids) of a dynamic semantic network derived from text segments, treating these features as proxies for information gaps. To empirically evaluate this pipeline, we collect reader curiosity ratings from participants (n = 49) as they read S. Collins's ''The Hunger Games'' novel. We then use the topological features from our pipeline as independent variables to predict these ratings, and experimentally show that they significantly improve curiosity prediction compared to a baseline model (73% vs. 30% explained deviance), validating our approach. This pipeline offers a new computational method for analyzing text structure and its relation to reader engagement. 

---
# The Scales of Justitia: A Comprehensive Survey on Safety Evaluation of LLMs 

**Authors**: Songyang Liu, Chaozhuo Li, Jiameng Qiu, Xi Zhang, Feiran Huang, Litian Zhang, Yiming Hei, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11094)  

**Abstract**: With the rapid advancement of artificial intelligence technology, Large Language Models (LLMs) have demonstrated remarkable potential in the field of Natural Language Processing (NLP), including areas such as content generation, human-computer interaction, machine translation, and code generation, among others. However, their widespread deployment has also raised significant safety concerns. In recent years, LLM-generated content has occasionally exhibited unsafe elements like toxicity and bias, particularly in adversarial scenarios, which has garnered extensive attention from both academia and industry. While numerous efforts have been made to evaluate the safety risks associated with LLMs, there remains a lack of systematic reviews summarizing these research endeavors. This survey aims to provide a comprehensive and systematic overview of recent advancements in LLMs safety evaluation, focusing on several key aspects: (1) "Why evaluate" that explores the background of LLMs safety evaluation, how they differ from general LLMs evaluation, and the significance of such evaluation; (2) "What to evaluate" that examines and categorizes existing safety evaluation tasks based on key capabilities, including dimensions such as toxicity, robustness, ethics, bias and fairness, truthfulness, and so on; (3) "Where to evaluate" that summarizes the evaluation metrics, datasets and benchmarks currently used in safety evaluations; (4) "How to evaluate" that reviews existing evaluation toolkit, and categorizing mainstream evaluation methods based on the roles of the evaluators. Finally, we identify the challenges in LLMs safety evaluation and propose potential research directions to promote further advancement in this field. We emphasize the importance of prioritizing LLMs safety evaluation to ensure the safe deployment of these models in real-world applications. 

---
# Dynamic Context Tuning for Retrieval-Augmented Generation: Enhancing Multi-Turn Planning and Tool Adaptation 

**Authors**: Jubin Abhishek Soni, Amit Anand, Rajesh Kumar Pandey, Aniket Abhishek Soni  

**Link**: [PDF](https://arxiv.org/pdf/2506.11092)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly advanced large language models (LLMs) by grounding their outputs in external tools and knowledge sources. However, existing RAG systems are typically constrained to static, single-turn interactions with fixed toolsets, making them ill-suited for dynamic domains such as healthcare and smart homes, where user intent, available tools, and contextual factors evolve over time. We present Dynamic Context Tuning (DCT), a lightweight framework that extends RAG to support multi-turn dialogue and evolving tool environments without requiring retraining. DCT integrates an attention-based context cache to track relevant past information, LoRA-based retrieval to dynamically select domain-specific tools, and efficient context compression to maintain inputs within LLM context limits. Experiments on both synthetic and real-world benchmarks show that DCT improves plan accuracy by 14% and reduces hallucinations by 37%, while matching GPT-4 performance at significantly lower cost. Furthermore, DCT generalizes to previously unseen tools, enabling scalable and adaptable AI assistants across a wide range of dynamic environments. 

---
# Customizing Speech Recognition Model with Large Language Model Feedback 

**Authors**: Shaoshi Ling, Guoli Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.11091)  

**Abstract**: Automatic speech recognition (ASR) systems have achieved strong performance on general transcription tasks. However, they continue to struggle with recognizing rare named entities and adapting to domain mismatches. In contrast, large language models (LLMs), trained on massive internet-scale datasets, are often more effective across a wide range of domains. In this work, we propose a reinforcement learning based approach for unsupervised domain adaptation, leveraging unlabeled data to enhance transcription quality, particularly the named entities affected by domain mismatch, through feedback from a LLM. Given contextual information, our framework employs a LLM as the reward model to score the hypotheses from the ASR model. These scores serve as reward signals to fine-tune the ASR model via reinforcement learning. Our method achieves a 21\% improvement on entity word error rate over conventional self-training methods. 

---
# Two Birds with One Stone: Improving Factuality and Faithfulness of LLMs via Dynamic Interactive Subspace Editing 

**Authors**: Pengbo Wang, Chaozhuo Li, Chenxu Wang, Liwen Zheng, Litian Zhang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11088)  

**Abstract**: LLMs have demonstrated unprecedented capabilities in natural language processing, yet their practical deployment remains hindered by persistent factuality and faithfulness hallucinations. While existing methods address these hallucination types independently, they inadvertently induce performance trade-offs, as interventions targeting one type often exacerbate the other. Through empirical and theoretical analysis of activation space dynamics in LLMs, we reveal that these hallucination categories share overlapping subspaces within neural representations, presenting an opportunity for concurrent mitigation. To harness this insight, we propose SPACE, a unified framework that jointly enhances factuality and faithfulness by editing shared activation subspaces. SPACE establishes a geometric foundation for shared subspace existence through dual-task feature modeling, then identifies and edits these subspaces via a hybrid probe strategy combining spectral clustering and attention head saliency scoring. Experimental results across multiple benchmark datasets demonstrate the superiority of our approach. 

---
# RedDebate: Safer Responses through Multi-Agent Red Teaming Debates 

**Authors**: Ali Asad, Stephen Obadinma, Radin Shayanfar, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11083)  

**Abstract**: We propose RedDebate, a novel multi-agent debate framework that leverages adversarial argumentation among Large Language Models (LLMs) to proactively identify and mitigate their own unsafe behaviours. Existing AI safety methods often depend heavily on costly human evaluations or isolated single-model assessment, both subject to scalability constraints and oversight risks. RedDebate instead embraces collaborative disagreement, enabling multiple LLMs to critically examine one another's reasoning, and systematically uncovering unsafe blind spots through automated red-teaming, and iteratively improve their responses. We further integrate distinct types of long-term memory that retain learned safety insights from debate interactions. Evaluating on established safety benchmarks such as HarmBench, we demonstrate the proposed method's effectiveness. Debate alone can reduce unsafe behaviours by 17.7%, and when combined with long-term memory modules, achieves reductions exceeding 23.5%. To our knowledge, RedDebate constitutes the first fully automated framework that combines multi-agent debates with red-teaming to progressively enhance AI safety without direct human intervention.(Github Repository: this https URL) 

---
# PRISM: A Transformer-based Language Model of Structured Clinical Event Data 

**Authors**: Lionel Levine, John Santerre, Alex S. Young, T. Barry Levine, Francis Campion, Majid Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.11082)  

**Abstract**: We introduce PRISM (Predictive Reasoning in Sequential Medicine), a transformer-based architecture designed to model the sequential progression of clinical decision-making processes. Unlike traditional approaches that rely on isolated diagnostic classification, PRISM frames clinical trajectories as tokenized sequences of events - including diagnostic tests, laboratory results, and diagnoses - and learns to predict the most probable next steps in the patient diagnostic journey. Leveraging a large custom clinical vocabulary and an autoregressive training objective, PRISM demonstrates the ability to capture complex dependencies across longitudinal patient timelines. Experimental results show substantial improvements over random baselines in next-token prediction tasks, with generated sequences reflecting realistic diagnostic pathways, laboratory result progressions, and clinician ordering behaviors. These findings highlight the feasibility of applying generative language modeling techniques to structured medical event data, enabling applications in clinical decision support, simulation, and education. PRISM establishes a foundation for future advancements in sequence-based healthcare modeling, bridging the gap between machine learning architectures and real-world diagnostic reasoning. 

---
# SAGE:Specification-Aware Grammar Extraction for Automated Test Case Generation with LLMs 

**Authors**: Aditi, Hyunwoo Park, Sicheol Sung, Yo-Sub Han, Sang-Ki Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.11081)  

**Abstract**: Grammar-based test case generation has proven effective for competitive programming problems, but generating valid and general grammars from natural language specifications remains a key challenge, especially under limited supervision. Context-Free Grammars with Counters (CCFGs) have recently been introduced as a formalism to represent such specifications with logical constraints by storing and reusing counter values during derivation. In this work, we explore the use of open-source large language models (LLMs) to induce CCFGs from specifications using a small number of labeled examples and verifiable reward-guided reinforcement learning. Our approach first fine-tunes an open-source LLM to perform specification-to-grammar translation, and further applies Group Relative Policy Optimization (GRPO) to enhance grammar validity and generality. We also examine the effectiveness of iterative feedback for open and closed-source LLMs in correcting syntactic and semantic errors in generated grammars.
Experimental results show that our approach SAGE achieves stronger generalization and outperforms 17 open and closed-source LLMs in both grammar quality and test effectiveness, improving over the state-of-the-art by 15.92%p in grammar validity and 12.34%p in test effectiveness. We provide our implementation and dataset at the following anonymous repository:this https URL 

---
# MANBench: Is Your Multimodal Model Smarter than Human? 

**Authors**: Han Zhou, Qitong Xu, Yiheng Dong, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11080)  

**Abstract**: The rapid advancement of Multimodal Large Language Models (MLLMs) has ignited discussions regarding their potential to surpass human performance in multimodal tasks. In response, we introduce MANBench (Multimodal Ability Norms Benchmark), a bilingual benchmark (English and Chinese) comprising 1,314 questions across nine tasks, spanning knowledge-based and non-knowledge-based domains. MANBench emphasizes intuitive reasoning, seamless cross-modal integration, and real-world complexity, providing a rigorous evaluation framework.
Through extensive human experiments involving diverse participants, we compared human performance against state-of-the-art MLLMs. The results indicate that while MLLMs excel in tasks like Knowledge and Text-Image Understanding, they struggle with deeper cross-modal reasoning tasks such as Transmorphic Understanding, Image Consistency, and Multi-image Understanding. Moreover, both humans and MLLMs face challenges in highly complex tasks like Puzzles and Spatial Imagination.
MANBench highlights the strengths and limitations of MLLMs, revealing that even advanced models fall short of achieving human-level performance across many domains. We hope MANBench will inspire efforts to bridge the gap between MLLMs and human multimodal capabilities. The code and dataset are available at this https URL. 

---
# RoE-FND: A Case-Based Reasoning Approach with Dual Verification for Fake News Detection via LLMs 

**Authors**: Yuzhou Yang, Yangming Zhou, Zhiying Zhu, Zhenxing Qian, Xinpeng Zhang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11078)  

**Abstract**: The proliferation of deceptive content online necessitates robust Fake News Detection (FND) systems. While evidence-based approaches leverage external knowledge to verify claims, existing methods face critical limitations: noisy evidence selection, generalization bottlenecks, and unclear decision-making processes. Recent efforts to harness Large Language Models (LLMs) for FND introduce new challenges, including hallucinated rationales and conclusion bias. To address these issues, we propose \textbf{RoE-FND} (\textbf{\underline{R}}eason \textbf{\underline{o}}n \textbf{\underline{E}}xperiences FND), a framework that reframes evidence-based FND as a logical deduction task by synergizing LLMs with experiential learning. RoE-FND encompasses two stages: (1) \textit{self-reflective knowledge building}, where a knowledge base is curated by analyzing past reasoning errors, namely the exploration stage, and (2) \textit{dynamic criterion retrieval}, which synthesizes task-specific reasoning guidelines from historical cases as experiences during deployment. It further cross-checks rationales against internal experience through a devised dual-channel procedure. Key contributions include: a case-based reasoning framework for FND that addresses multiple existing challenges, a training-free approach enabling adaptation to evolving situations, and empirical validation of the framework's superior generalization and effectiveness over state-of-the-art methods across three datasets. 

---
# CyclicReflex: Improving Large Reasoning Models via Cyclical Reflection Token Scheduling 

**Authors**: Chongyu Fan, Yihua Zhang, Jinghan Jia, Alfred Hero, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11077)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens or textual segments that prompt self-evaluative reflection. We refer to these transition markers and reflective cues as "reflection tokens" (e.g., "wait", "but", "alternatively"). In this work, we treat reflection tokens as a "resource" and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand and manage this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, we propose cyclical reflection token scheduling (termed CyclicReflex), a decoding strategy that dynamically modulates reflection token logits using a position-dependent triangular waveform. Experiments on MATH500, AIME2024/2025, and AMC2023 demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B-8B), outperforming standard decoding and more recent approaches such as TIP (thought switching penalty) and S1. Codes are available at this https URL. 

---
# CLAIM: Mitigating Multilingual Object Hallucination in Large Vision-Language Models with Cross-Lingual Attention Intervention 

**Authors**: Zekai Ye, Qiming Li, Xiaocheng Feng, Libo Qin, Yichong Huang, Baohang Li, Kui Jiang, Yang Xiang, Zhirui Zhang, Yunfei Lu, Duyu Tang, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11073)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal abilities but remain prone to multilingual object hallucination, with a higher likelihood of generating responses inconsistent with the visual input when utilizing queries in non-English languages compared to English. Most existing approaches to address these rely on pretraining or fine-tuning, which are resource-intensive. In this paper, inspired by observing the disparities in cross-modal attention patterns across languages, we propose Cross-Lingual Attention Intervention for Mitigating multilingual object hallucination (CLAIM) in LVLMs, a novel near training-free method by aligning attention patterns. CLAIM first identifies language-specific cross-modal attention heads, then estimates language shift vectors from English to the target language, and finally intervenes in the attention outputs during inference to facilitate cross-lingual visual perception capability alignment. Extensive experiments demonstrate that CLAIM achieves an average improvement of 13.56% (up to 30% in Spanish) on the POPE and 21.75% on the hallucination subsets of the MME benchmark across various languages. Further analysis reveals that multilingual attention divergence is most prominent in intermediate layers, highlighting their critical role in multilingual scenarios. 

---
# Targeted control of fast prototyping through domain-specific interface 

**Authors**: Yu-Zhe Shi, Mingchen Liu, Hanlu Ma, Qiao Xu, Huamin Qu, Kun He, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11070)  

**Abstract**: Industrial designers have long sought a natural and intuitive way to achieve the targeted control of prototype models -- using simple natural language instructions to configure and adjust the models seamlessly according to their intentions, without relying on complex modeling commands. While Large Language Models have shown promise in this area, their potential for controlling prototype models through language remains partially underutilized. This limitation stems from gaps between designers' languages and modeling languages, including mismatch in abstraction levels, fluctuation in semantic precision, and divergence in lexical scopes. To bridge these gaps, we propose an interface architecture that serves as a medium between the two languages. Grounded in design principles derived from a systematic investigation of fast prototyping practices, we devise the interface's operational mechanism and develop an algorithm for its automated domain specification. Both machine-based evaluations and human studies on fast prototyping across various product design domains demonstrate the interface's potential to function as an auxiliary module for Large Language Models, enabling precise and effective targeted control of prototype models. 

---
# Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models 

**Authors**: Bumjin Park, Jinsil Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11068)  

**Abstract**: Large language models (LLMs) are increasingly engaging in moral and ethical reasoning, where criteria for judgment are often unclear, even for humans. While LLM alignment studies cover many areas, one important yet underexplored area is how LLMs make judgments about obligations. This work reveals a strong tendency in LLMs to judge non-obligatory contexts as obligations when prompts are augmented with modal expressions such as must or ought to. We introduce this phenomenon as Deontological Keyword Bias (DKB). We find that LLMs judge over 90\% of commonsense scenarios as obligations when modal expressions are present. This tendency is consist across various LLM families, question types, and answer formats. To mitigate DKB, we propose a judgment strategy that integrates few-shot examples with reasoning prompts. This study sheds light on how modal expressions, as a form of linguistic framing, influence the normative decisions of LLMs and underscores the importance of addressing such biases to ensure judgment alignment. 

---
# A Large Language Model Based Pipeline for Review of Systems Entity Recognition from Clinical Notes 

**Authors**: Hieu Nghiem, Hemanth Reddy Singareddy, Zhuqi Miao, Jivan Lamichhane, Abdulaziz Ahmed, Johnson Thomas, Dursun Delen, William Paiva  

**Link**: [PDF](https://arxiv.org/pdf/2506.11067)  

**Abstract**: Objective: Develop a cost-effective, large language model (LLM)-based pipeline for automatically extracting Review of Systems (ROS) entities from clinical notes. Materials and Methods: The pipeline extracts ROS sections using SecTag, followed by few-shot LLMs to identify ROS entity spans, their positive/negative status, and associated body systems. We implemented the pipeline using open-source LLMs (Mistral, Llama, Gemma) and ChatGPT. The evaluation was conducted on 36 general medicine notes containing 341 annotated ROS entities. Results: When integrating ChatGPT, the pipeline achieved the lowest error rates in detecting ROS entity spans and their corresponding statuses/systems (28.2% and 14.5%, respectively). Open-source LLMs enable local, cost-efficient execution of the pipeline while delivering promising performance with similarly low error rates (span: 30.5-36.7%; status/system: 24.3-27.3%). Discussion and Conclusion: Our pipeline offers a scalable and locally deployable solution to reduce ROS documentation burden. Open-source LLMs present a viable alternative to commercial models in resource-limited healthcare environments. 

---
# Smotrom tvoja pa ander drogoj verden! Resurrecting Dead Pidgin with Generative Models: Russenorsk Case Study 

**Authors**: Alexey Tikhonov, Sergei Shteiner, Anna Bykova, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.11065)  

**Abstract**: Russenorsk, a pidgin language historically used in trade interactions between Russian and Norwegian speakers, represents a unique linguistic phenomenon. In this paper, we attempt to analyze its lexicon using modern large language models (LLMs), based on surviving literary sources. We construct a structured dictionary of the language, grouped by synonyms and word origins. Subsequently, we use this dictionary to formulate hypotheses about the core principles of word formation and grammatical structure in Russenorsk and show which hypotheses generated by large language models correspond to the hypotheses previously proposed ones in the academic literature. We also develop a "reconstruction" translation agent that generates hypothetical Russenorsk renderings of contemporary Russian and Norwegian texts. 

---
# Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation 

**Authors**: Jiayu Yao, Shenghua Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Yuyao Ge, Zhecheng Li, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11063)  

**Abstract**: Multimodal Retrieval-Augmented Generation (RAG) systems have become essential in knowledge-intensive and open-domain tasks. As retrieval complexity increases, ensuring the robustness of these systems is critical. However, current RAG models are highly sensitive to the order in which evidence is presented, often resulting in unstable performance and biased reasoning, particularly as the number of retrieved items or modality diversity grows. This raises a central question: How does the position of retrieved evidence affect multimodal RAG performance? To answer this, we present the first comprehensive study of position bias in multimodal RAG systems. Through controlled experiments across text-only, image-only, and mixed-modality tasks, we observe a consistent U-shaped accuracy curve with respect to evidence position. To quantify this bias, we introduce the Position Sensitivity Index ($PSI_p$) and develop a visualization framework to trace attention allocation patterns across decoder layers. Our results reveal that multimodal interactions intensify position bias compared to unimodal settings, and that this bias increases logarithmically with retrieval range. These findings offer both theoretical and empirical foundations for position-aware analysis in RAG, highlighting the need for evidence reordering or debiasing strategies to build more reliable and equitable generation systems. 

---
# TeleEval-OS: Performance evaluations of large language models for operations scheduling 

**Authors**: Yanyan Wang, Yingying Wang, Junli Liang, Yin Xu, Yunlong Liu, Yiming Xu, Zhengwang Jiang, Zhehe Li, Fei Li, Long Zhao, Kuang Xu, Qi Song, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11017)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly propelled progress in artificial intelligence, demonstrating substantial application potential across multiple specialized domains. Telecommunications operation scheduling (OS) is a critical aspect of the telecommunications industry, involving the coordinated management of networks, services, risks, and human resources to optimize production scheduling and ensure unified service control. However, the inherent complexity and domain-specific nature of OS tasks, coupled with the absence of comprehensive evaluation benchmarks, have hindered thorough exploration of LLMs' application potential in this critical field. To address this research gap, we propose the first Telecommunications Operation Scheduling Evaluation Benchmark (TeleEval-OS). Specifically, this benchmark comprises 15 datasets across 13 subtasks, comprehensively simulating four key operational stages: intelligent ticket creation, intelligent ticket handling, intelligent ticket closure, and intelligent evaluation. To systematically assess the performance of LLMs on tasks of varying complexity, we categorize their capabilities in telecommunications operation scheduling into four hierarchical levels, arranged in ascending order of difficulty: basic NLP, knowledge Q&A, report generation, and report analysis. On TeleEval-OS, we leverage zero-shot and few-shot evaluation methods to comprehensively assess 10 open-source LLMs (e.g., DeepSeek-V3) and 4 closed-source LLMs (e.g., GPT-4o) across diverse scenarios. Experimental results demonstrate that open-source LLMs can outperform closed-source LLMs in specific scenarios, highlighting their significant potential and value in the field of telecommunications operation scheduling. 

---
# Generative Representational Learning of Foundation Models for Recommendation 

**Authors**: Zheli Zhou, Chenxu Zhu, Jianghao Lin, Bo Chen, Ruiming Tang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11999)  

**Abstract**: Developing a single foundation model with the capability to excel across diverse tasks has been a long-standing objective in the field of artificial intelligence. As the wave of general-purpose foundation models sweeps across various domains, their influence has significantly extended to the field of recommendation systems. While recent efforts have explored recommendation foundation models for various generative tasks, they often overlook crucial embedding tasks and struggle with the complexities of multi-task learning, including knowledge sharing & conflict resolution, and convergence speed inconsistencies. To address these limitations, we introduce RecFound, a generative representational learning framework for recommendation foundation models. We construct the first comprehensive dataset for recommendation foundation models covering both generative and embedding tasks across diverse scenarios. Based on this dataset, we propose a novel multi-task training scheme featuring a Task-wise Mixture of Low-rank Experts (TMoLE) to handle knowledge sharing & conflict, a Step-wise Convergence-oriented Sample Scheduler (S2Sched) to address inconsistent convergence, and a Model Merge module to balance the performance across tasks. Experiments demonstrate that RecFound achieves state-of-the-art performance across various recommendation tasks, outperforming existing baselines. 

---
# VGR: Visual Grounded Reasoning 

**Authors**: Jiacong Wang, Zijiang Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11991)  

**Abstract**: In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA. 

---
# Schema-R1: A reasoning training approach for schema linking in Text-to-SQL Task 

**Authors**: Wuzhenghong Wen, Su Pan, yuwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11986)  

**Abstract**: Schema linking is a critical step in Text-to-SQL task, aiming to accurately predict the table names and column names required for the SQL query based on the given question. However, current fine-tuning approaches for schema linking models employ a rote-learning paradigm, excessively optimizing for ground truth schema linking outcomes while compromising reasoning ability. This limitation arises because of the difficulty in acquiring a high-quality reasoning sample for downstream tasks. To address this, we propose Schema-R1, a reasoning schema linking model trained using reinforcement learning. Specifically, Schema-R1 consists of three key steps: constructing small batches of high-quality reasoning samples, supervised fine-tuning for cold-start initialization, and rule-based reinforcement learning training. The final results demonstrate that our method effectively enhances the reasoning ability of the schema linking model, achieving a 10\% improvement in filter accuracy compared to the existing method. Our code is available at this https URL. 

---
# LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming? 

**Authors**: Zihan Zheng, Zerui Cheng, Zeyu Shen, Shang Zhou, Kaiyuan Liu, Hansen He, Dongruixuan Li, Stanley Wei, Hangyi Hao, Jianzhu Yao, Peiyao Sheng, Zixuan Wang, Wenhao Chai, Aleksandra Korolova, Peter Henderson, Sanjeev Arora, Pramod Viswanath, Jingbo Shang, Saining Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.11928)  

**Abstract**: Recent reports claim that large language models (LLMs) now outperform elite humans in competitive programming. Drawing on knowledge from a group of medalists in international algorithmic contests, we revisit this claim, examining how LLMs differ from human experts and where limitations still remain. We introduce LiveCodeBench Pro, a benchmark composed of problems from Codeforces, ICPC, and IOI that are continuously updated to reduce the likelihood of data contamination. A team of Olympiad medalists annotates every problem for algorithmic categories and conducts a line-by-line analysis of failed model-generated submissions. Using this new data and benchmark, we find that frontier models still have significant limitations: without external tools, the best model achieves only 53% pass@1 on medium-difficulty problems and 0% on hard problems, domains where expert humans still excel. We also find that LLMs succeed at implementation-heavy problems but struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications. High performance appears largely driven by implementation precision and tool augmentation, not superior reasoning. LiveCodeBench Pro thus highlights the significant gap to human grandmaster levels, while offering fine-grained diagnostics to steer future improvements in code-centric LLM reasoning. 

---
# TreeRL: LLM Reinforcement Learning with On-Policy Tree Search 

**Authors**: Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11902)  

**Abstract**: Reinforcement learning (RL) with tree search has demonstrated superior performance in traditional reasoning tasks. Compared to conventional independent chain sampling strategies with outcome supervision, tree search enables better exploration of the reasoning space and provides dense, on-policy process rewards during RL training but remains under-explored in On-Policy LLM RL. We propose TreeRL, a reinforcement learning framework that directly incorporates on-policy tree search for RL training. Our approach includes intermediate supervision and eliminates the need for a separate reward model training. Existing approaches typically train a separate process reward model, which can suffer from distribution mismatch and reward hacking. We also introduce a cost-effective tree search approach that achieves higher search efficiency under the same generation token budget by strategically branching from high-uncertainty intermediate steps rather than using random branching. Experiments on challenging math and code reasoning benchmarks demonstrate that TreeRL achieves superior performance compared to traditional ChainRL, highlighting the potential of tree search for LLM. TreeRL is open-sourced at this https URL. 

---
# Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making 

**Authors**: Claudio Fanconi, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.11887)  

**Abstract**: Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions. 

---
# Addressing Bias in LLMs: Strategies and Application to Fair AI-based Recruitment 

**Authors**: Alejandro Peña, Julian Fierrez, Aythami Morales, Gonzalo Mancera, Miguel Lopez, Ruben Tolosana  

**Link**: [PDF](https://arxiv.org/pdf/2506.11880)  

**Abstract**: The use of language technologies in high-stake settings is increasing in recent years, mostly motivated by the success of Large Language Models (LLMs). However, despite the great performance of LLMs, they are are susceptible to ethical concerns, such as demographic biases, accountability, or privacy. This work seeks to analyze the capacity of Transformers-based systems to learn demographic biases present in the data, using a case study on AI-based automated recruitment. We propose a privacy-enhancing framework to reduce gender information from the learning pipeline as a way to mitigate biased behaviors in the final tools. Our experiments analyze the influence of data biases on systems built on two different LLMs, and how the proposed framework effectively prevents trained systems from reproducing the bias in the data. 

---
# Rethinking Multilingual Vision-Language Translation: Dataset, Evaluation, and Adaptation 

**Authors**: Xintong Wang, Jingheng Pan, Yixiao Liu, Xiaohu Zhao, Chenyang Lyu, Minghao Wu, Chris Biemann, Longyue Wang, Linlong Xu, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11820)  

**Abstract**: Vision-Language Translation (VLT) is a challenging task that requires accurately recognizing multilingual text embedded in images and translating it into the target language with the support of visual context. While recent Large Vision-Language Models (LVLMs) have demonstrated strong multilingual and visual understanding capabilities, there is a lack of systematic evaluation and understanding of their performance on VLT. In this work, we present a comprehensive study of VLT from three key perspectives: data quality, model architecture, and evaluation metrics. (1) We identify critical limitations in existing datasets, particularly in semantic and cultural fidelity, and introduce AibTrans -- a multilingual, parallel, human-verified dataset with OCR-corrected annotations. (2) We benchmark 11 commercial LVLMs/LLMs and 6 state-of-the-art open-source models across end-to-end and cascaded architectures, revealing their OCR dependency and contrasting generation versus reasoning behaviors. (3) We propose Density-Aware Evaluation to address metric reliability issues under varying contextual complexity, introducing the DA Score as a more robust measure of translation quality. Building upon these findings, we establish a new evaluation benchmark for VLT. Notably, we observe that fine-tuning LVLMs on high-resource language pairs degrades cross-lingual performance, and we propose a balanced multilingual fine-tuning strategy that effectively adapts LVLMs to VLT without sacrificing their generalization ability. 

---
# On the Performance of LLMs for Real Estate Appraisal 

**Authors**: Margot Geerts, Manon Reusens, Bart Baesens, Seppe vanden Broucke, Jochen De Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2506.11812)  

**Abstract**: The real estate market is vital to global economies but suffers from significant information asymmetry. This study examines how Large Language Models (LLMs) can democratize access to real estate insights by generating competitive and interpretable house price estimates through optimized In-Context Learning (ICL) strategies. We systematically evaluate leading LLMs on diverse international housing datasets, comparing zero-shot, few-shot, market report-enhanced, and hybrid prompting techniques. Our results show that LLMs effectively leverage hedonic variables, such as property size and amenities, to produce meaningful estimates. While traditional machine learning models remain strong for pure predictive accuracy, LLMs offer a more accessible, interactive and interpretable alternative. Although self-explanations require cautious interpretation, we find that LLMs explain their predictions in agreement with state-of-the-art models, confirming their trustworthiness. Carefully selected in-context examples based on feature similarity and geographic proximity, significantly enhance LLM performance, yet LLMs struggle with overconfidence in price intervals and limited spatial reasoning. We offer practical guidance for structured prediction tasks through prompt optimization. Our findings highlight LLMs' potential to improve transparency in real estate appraisal and provide actionable insights for stakeholders. 

---
# Quizzard@INOVA Challenge 2025 -- Track A: Plug-and-Play Technique in Interleaved Multi-Image Model 

**Authors**: Dinh Viet Cuong, Hoang-Bao Le, An Pham Ngoc Nguyen, Liting Zhou, Cathal Gurrin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11737)  

**Abstract**: This paper addresses two main objectives. Firstly, we demonstrate the impressive performance of the LLaVA-NeXT-interleave on 22 datasets across three different tasks: Multi-Image Reasoning, Documents and Knowledge-Based Understanding and Interactive Multi-Modal Communication. Secondly, we add the Dense Channel Integration (DCI) connector to the LLaVA-NeXT-Interleave and compare its performance against the standard model. We find that the standard model achieves the highest overall accuracy, excelling in vision-heavy tasks like VISION, NLVR2, and Fashion200K. Meanwhile, the DCI-enhanced version shows particular strength on datasets requiring deeper semantic coherence or structured change understanding such as MIT-States_PropertyCoherence and SlideVQA. Our results highlight the potential of combining powerful foundation models with plug-and-play techniques for Interleave tasks. The code is available at this https URL. 

---
# (SimPhon Speech Test): A Data-Driven Method for In Silico Design and Validation of a Phonetically Balanced Speech Test 

**Authors**: Stefan Bleeck  

**Link**: [PDF](https://arxiv.org/pdf/2506.11620)  

**Abstract**: Traditional audiometry often provides an incomplete characterization of the functional impact of hearing loss on speech understanding, particularly for supra-threshold deficits common in presbycusis. This motivates the development of more diagnostically specific speech perception tests. We introduce the Simulated Phoneme Speech Test (SimPhon Speech Test) methodology, a novel, multi-stage computational pipeline for the in silico design and validation of a phonetically balanced minimal-pair speech test. This methodology leverages a modern Automatic Speech Recognition (ASR) system as a proxy for a human listener to simulate the perceptual effects of sensorineural hearing loss. By processing speech stimuli under controlled acoustic degradation, we first identify the most common phoneme confusion patterns. These patterns then guide the data-driven curation of a large set of candidate word pairs derived from a comprehensive linguistic corpus. Subsequent phases involving simulated diagnostic testing, expert human curation, and a final, targeted sensitivity analysis systematically reduce the candidates to a final, optimized set of 25 pairs (the SimPhon Speech Test-25). A key finding is that the diagnostic performance of the SimPhon Speech Test-25 test items shows no significant correlation with predictions from the standard Speech Intelligibility Index (SII), suggesting the SimPhon Speech Test captures perceptual deficits beyond simple audibility. This computationally optimized test set offers a significant increase in efficiency for audiological test development, ready for initial human trials. 

---
# VLM@school -- Evaluation of AI image understanding on German middle school knowledge 

**Authors**: René Peinl, Vincent Tischler  

**Link**: [PDF](https://arxiv.org/pdf/2506.11604)  

**Abstract**: This paper introduces a novel benchmark dataset designed to evaluate the capabilities of Vision Language Models (VLMs) on tasks that combine visual reasoning with subject-specific background knowledge in the German language. In contrast to widely used English-language benchmarks that often rely on artificially difficult or decontextualized problems, this dataset draws from real middle school curricula across nine domains including mathematics, history, biology, and religion. The benchmark includes over 2,000 open-ended questions grounded in 486 images, ensuring that models must integrate visual interpretation with factual reasoning rather than rely on superficial textual cues. We evaluate thirteen state-of-the-art open-weight VLMs across multiple dimensions, including domain-specific accuracy and performance on adversarial crafted questions. Our findings reveal that even the strongest models achieve less than 45% overall accuracy, with particularly poor performance in music, mathematics, and adversarial settings. Furthermore, the results indicate significant discrepancies between success on popular benchmarks and real-world multimodal understanding. We conclude that middle school-level tasks offer a meaningful and underutilized avenue for stress-testing VLMs, especially in non-English contexts. The dataset and evaluation protocol serve as a rigorous testbed to better understand and improve the visual and linguistic reasoning capabilities of future AI systems. 

---
# DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs 

**Authors**: Bo-Cheng Chiu, Jen-Jee Chen, Yu-Chee Tseng, Feng-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11558)  

**Abstract**: Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling. 

---
# RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning 

**Authors**: Yu Wang, Shiwan Zhao, Ming Fan, Zhihu Wang, Yubo Zhang, Xicheng Zhang, Zhengfan Wang, Heyuan Huang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11555)  

**Abstract**: The integration of external knowledge through Retrieval-Augmented Generation (RAG) has become foundational in enhancing large language models (LLMs) for knowledge-intensive tasks. However, existing RAG paradigms often overlook the cognitive step of applying knowledge, leaving a gap between retrieved facts and task-specific reasoning. In this work, we introduce RAG+, a principled and modular extension that explicitly incorporates application-aware reasoning into the RAG pipeline. RAG+ constructs a dual corpus consisting of knowledge and aligned application examples, created either manually or automatically, and retrieves both jointly during inference. This design enables LLMs not only to access relevant information but also to apply it within structured, goal-oriented reasoning processes. Experiments across mathematical, legal, and medical domains, conducted on multiple models, demonstrate that RAG+ consistently outperforms standard RAG variants, achieving average improvements of 3-5%, and peak gains up to 7.5% in complex scenarios. By bridging retrieval with actionable application, RAG+ advances a more cognitively grounded framework for knowledge integration, representing a step toward more interpretable and capable LLMs. 

---
# Brewing Knowledge in Context: Distillation Perspectives on In-Context Learning 

**Authors**: Chengye Li, Haiyun Liu, Yuanxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11516)  

**Abstract**: In-context learning (ICL) allows large language models (LLMs) to solve novel tasks without weight updates. Despite its empirical success, the mechanism behind ICL remains poorly understood, limiting our ability to interpret, improve, and reliably apply it. In this paper, we propose a new theoretical perspective that interprets ICL as an implicit form of knowledge distillation (KD), where prompt demonstrations guide the model to form a task-specific reference model during inference. Under this view, we derive a Rademacher complexity-based generalization bound and prove that the bias of the distilled weights grows linearly with the Maximum Mean Discrepancy (MMD) between the prompt and target distributions. This theoretical framework explains several empirical phenomena and unifies prior gradient-based and distributional analyses. To the best of our knowledge, this is the first to formalize inference-time attention as a distillation process, which provides theoretical insights for future prompt engineering and automated demonstration selection. 

---
# Manager: Aggregating Insights from Unimodal Experts in Two-Tower VLMs and MLLMs 

**Authors**: Xiao Xu, Libo Qin, Wanxiang Che, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11515)  

**Abstract**: Two-Tower Vision--Language Models (VLMs) have demonstrated strong performance across various downstream VL tasks. While BridgeTower further enhances performance by building bridges between encoders, it \textit{(i)} suffers from ineffective layer-by-layer utilization of unimodal representations, \textit{(ii)} restricts the flexible exploitation of different levels of unimodal semantic knowledge, and \textit{(iii)} is limited to the evaluation on traditional low-resolution datasets only with the Two-Tower VLM architecture. In this work, we propose Manager, a lightweight, efficient and effective plugin that adaptively aggregates insights from different levels of pre-trained unimodal experts to facilitate more comprehensive VL alignment and fusion. First, under the Two-Tower VLM architecture, we introduce ManagerTower, a novel VLM that introduces the manager in each cross-modal layer. Whether with or without VL pre-training, ManagerTower outperforms previous strong baselines and achieves superior performance on 4 downstream VL tasks. Moreover, we extend our exploration to the latest Multimodal Large Language Model (MLLM) architecture. We demonstrate that LLaVA-OV-Manager significantly boosts the zero-shot performance of LLaVA-OV across different categories of capabilities, images, and resolutions on 20 downstream datasets, whether the multi-grid algorithm is enabled or not. In-depth analysis reveals that both our manager and the multi-grid algorithm can be viewed as a plugin that improves the visual representation by capturing more diverse visual details from two orthogonal perspectives (depth and width). Their synergy can mitigate the semantic ambiguity caused by the multi-grid algorithm and further improve performance. Code and models are available at this https URL. 

---
# AutoGen Driven Multi Agent Framework for Iterative Crime Data Analysis and Prediction 

**Authors**: Syeda Kisaa Fatima, Tehreem Zubair, Noman Ahmed, Asifullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11475)  

**Abstract**: This paper introduces LUCID-MA (Learning and Understanding Crime through Dialogue of Multiple Agents), an innovative AI powered framework where multiple AI agents collaboratively analyze and understand crime data. Our system that consists of three core components: an analysis assistant that highlights spatiotemporal crime patterns, a feedback component that reviews and refines analytical results and a prediction component that forecasts future crime trends. With a well-designed prompt and the LLaMA-2-13B-Chat-GPTQ model, it runs completely offline and allows the agents undergo self-improvement through 100 rounds of communication with less human interaction. A scoring function is incorporated to evaluate agent's performance, providing visual plots to track learning progress. This work demonstrates the potential of AutoGen-style agents for autonomous, scalable, and iterative analysis in social science domains maintaining data privacy through offline execution. 

---
# Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs 

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.11415)  

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system. 

---
# LoRA Users Beware: A Few Spurious Tokens Can Manipulate Your Finetuned Model 

**Authors**: Pradyut Sekhsaria, Marcel Mateos Salles, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2506.11402)  

**Abstract**: Parameter Efficient FineTuning (PEFT), such as Low-Rank Adaptation (LoRA), aligns pre-trained Large Language Models (LLMs) to particular downstream tasks in a resource-efficient manner. Because efficiency has been the main metric of progress, very little attention has been put in understanding possible catastrophic failures. We uncover one such failure: PEFT encourages a model to search for shortcut solutions to solve its fine-tuning tasks. When very small amount of tokens, e.g., one token per prompt, are correlated with downstream task classes, PEFT makes any pretrained model rely predominantly on that token for decision making. While such spurious tokens may emerge accidentally from incorrect data cleaning, it also opens opportunities for malevolent parties to control a model's behavior from Seamless Spurious Token Injection (SSTI). In SSTI, a small amount of tokens correlated with downstream classes are injected by the dataset creators. At test time, the finetuned LLM's behavior can be controlled solely by injecting those few tokens. We apply SSTI across models from three families (Snowflake Arctic, Apple OpenELM, and Meta LLaMA-3) and four diverse datasets (IMDB, Financial Classification, CommonSense QA, and Bias in Bios). Our findings reveal three astonishing behaviors. First, as few as a single token of SSTI is sufficient to steer a model's decision making. Second, for light SSTI, the reliance on spurious tokens is proportional to the LoRA rank. Lastly, with aggressive SSTI, larger LoRA rank values become preferable to small rank values as it makes the model attend to non-spurious tokens, hence improving robustness. 

---
# Large Language Model-Powered Conversational Agent Delivering Problem-Solving Therapy (PST) for Family Caregivers: Enhancing Empathy and Therapeutic Alliance Using In-Context Learning 

**Authors**: Liying Wang, Ph.D., Daffodil Carrington, M.S., Daniil Filienko, M.S., Caroline El Jazmi, M.S., Serena Jinchen Xie, M.S., Martine De Cock, Ph.D., Sarah Iribarren, Ph.D., Weichao Yuwen, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2506.11376)  

**Abstract**: Family caregivers often face substantial mental health challenges due to their multifaceted roles and limited resources. This study explored the potential of a large language model (LLM)-powered conversational agent to deliver evidence-based mental health support for caregivers, specifically Problem-Solving Therapy (PST) integrated with Motivational Interviewing (MI) and Behavioral Chain Analysis (BCA). A within-subject experiment was conducted with 28 caregivers interacting with four LLM configurations to evaluate empathy and therapeutic alliance. The best-performing models incorporated Few-Shot and Retrieval-Augmented Generation (RAG) prompting techniques, alongside clinician-curated examples. The models showed improved contextual understanding and personalized support, as reflected by qualitative responses and quantitative ratings on perceived empathy and therapeutic alliances. Participants valued the model's ability to validate emotions, explore unexpressed feelings, and provide actionable strategies. However, balancing thorough assessment with efficient advice delivery remains a challenge. This work highlights the potential of LLMs in delivering empathetic and tailored support for family caregivers. 

---
# Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables 

**Authors**: Yitong Zhou, Mingyue Cheng, Qingyang Mao, Yucong Luo, Qi Liu, Yupeng Li, Xiaohan Zhang, Deguang Liu, Xin Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11375)  

**Abstract**: Chemical tables encode complex experimental knowledge through symbolic expressions, structured variables, and embedded molecular graphics. Existing benchmarks largely overlook this multimodal and domain-specific complexity, limiting the ability of multimodal large language models to support scientific understanding in chemistry. In this work, we introduce ChemTable, a large-scale benchmark of real-world chemical tables curated from the experimental sections of literature. ChemTable includes expert-annotated cell polygons, logical layouts, and domain-specific labels, including reagents, catalysts, yields, and graphical components and supports two core tasks: (1) Table Recognition, covering structure parsing and content extraction; and (2) Table Understanding, encompassing both descriptive and reasoning-oriented question answering grounded in table structure and domain semantics. We evaluated a range of representative multimodal models, including both open-source and closed-source models, on ChemTable and reported a series of findings with practical and conceptual insights. Although models show reasonable performance on basic layout parsing, they exhibit substantial limitations on both descriptive and inferential QA tasks compared to human performance, and we observe significant performance gaps between open-source and closed-source models across multiple dimensions. These results underscore the challenges of chemistry-aware table understanding and position ChemTable as a rigorous and realistic benchmark for advancing scientific reasoning. 

---
# GLAP: General contrastive audio-text pretraining across domains and languages 

**Authors**: Heinrich Dinkel, Zhiyong Yan, Tianzi Wang, Yongqing Wang, Xingwei Sun, Yadong Niu, Jizhong Liu, Gang Li, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11350)  

**Abstract**: Contrastive Language Audio Pretraining (CLAP) is a widely-used method to bridge the gap between audio and text domains. Current CLAP methods enable sound and music retrieval in English, ignoring multilingual spoken content. To address this, we introduce general language audio pretraining (GLAP), which expands CLAP with multilingual and multi-domain abilities. GLAP demonstrates its versatility by achieving competitive performance on standard audio-text retrieval benchmarks like Clotho and AudioCaps, while significantly surpassing existing methods in speech retrieval and classification tasks. Additionally, GLAP achieves strong results on widely used sound-event zero-shot benchmarks, while simultaneously outperforming previous methods on speech content benchmarks. Further keyword spotting evaluations across 50 languages emphasize GLAP's advanced multilingual capabilities. Finally, multilingual sound and music understanding is evaluated across four languages. Checkpoints and Source: this https URL. 

---
# LLM-as-a-Judge for Reference-less Automatic Code Validation and Refinement for Natural Language to Bash in IT Automation 

**Authors**: Ngoc Phuoc An Vo, Brent Paulovicks, Vadim Sheinin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11237)  

**Abstract**: In an effort to automatically evaluate and select the best model and improve code quality for automatic incident remediation in IT Automation, it is crucial to verify if the generated code for remediation action is syntactically and semantically correct and whether it can be executed correctly as intended. There are three approaches: 1) conventional methods use surface form similarity metrics (token match, exact match, etc.) which have numerous limitations, 2) execution-based evaluation focuses more on code functionality based on pass/fail judgments for given test-cases, and 3) LLM-as-a-Judge employs LLMs for automated evaluation to judge if it is a correct answer for a given problem based on pre-defined metrics. In this work, we focused on enhancing LLM-as-a-Judge using bidirectional functionality matching and logic representation for reference-less automatic validation and refinement for Bash code generation to select the best model for automatic incident remediation in IT Automation. We used execution-based evaluation as ground-truth to evaluate our LLM-as-a-Judge metrics. Results show high accuracy and agreement with execution-based evaluation (and up to 8% over baseline). Finally, we built Reflection code agents to utilize judgments and feedback from our evaluation metrics which achieved significant improvement (up to 24% increase in accuracy) for automatic code refinement. 

---
# LLM-as-a-Fuzzy-Judge: Fine-Tuning Large Language Models as a Clinical Evaluation Judge with Fuzzy Logic 

**Authors**: Weibing Zheng, Laurah Turner, Jess Kropczynski, Murat Ozer, Tri Nguyen, Shane Halse  

**Link**: [PDF](https://arxiv.org/pdf/2506.11221)  

**Abstract**: Clinical communication skills are critical in medical education, and practicing and assessing clinical communication skills on a scale is challenging. Although LLM-powered clinical scenario simulations have shown promise in enhancing medical students' clinical practice, providing automated and scalable clinical evaluation that follows nuanced physician judgment is difficult. This paper combines fuzzy logic and Large Language Model (LLM) and proposes LLM-as-a-Fuzzy-Judge to address the challenge of aligning the automated evaluation of medical students' clinical skills with subjective physicians' preferences. LLM-as-a-Fuzzy-Judge is an approach that LLM is fine-tuned to evaluate medical students' utterances within student-AI patient conversation scripts based on human annotations from four fuzzy sets, including Professionalism, Medical Relevance, Ethical Behavior, and Contextual Distraction. The methodology of this paper started from data collection from the LLM-powered medical education system, data annotation based on multidimensional fuzzy sets, followed by prompt engineering and the supervised fine-tuning (SFT) of the pre-trained LLMs using these human annotations. The results show that the LLM-as-a-Fuzzy-Judge achieves over 80\% accuracy, with major criteria items over 90\%, effectively leveraging fuzzy logic and LLM as a solution to deliver interpretable, human-aligned assessment. This work suggests the viability of leveraging fuzzy logic and LLM to align with human preferences, advances automated evaluation in medical education, and supports more robust assessment and judgment practices. The GitHub repository of this work is available at this https URL 

---
# Knowledge Graph Embeddings with Representing Relations as Annular Sectors 

**Authors**: Huiling Zhu, Yingqi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11099)  

**Abstract**: Knowledge graphs (KGs), structured as multi-relational data of entities and relations, are vital for tasks like data analysis and recommendation systems. Knowledge graph completion (KGC), or link prediction, addresses incompleteness of KGs by inferring missing triples (h, r, t). It is vital for downstream applications. Region-based embedding models usually embed entities as points and relations as geometric regions to accomplish the task. Despite progress, these models often overlook semantic hierarchies inherent in entities. To solve this problem, we propose SectorE, a novel embedding model in polar coordinates. Relations are modeled as annular sectors, combining modulus and phase to capture inference patterns and relation attributes. Entities are embedded as points within these sectors, intuitively encoding hierarchical structure. Evaluated on FB15k-237, WN18RR, and YAGO3-10, SectorE achieves competitive performance against various kinds of models, demonstrating strengths in semantic modeling capability. 

---
# Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting 

**Authors**: Guillaume Wisniewski, Séverine Guillaume, Clara Rosina Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2506.11096)  

**Abstract**: Pretrained speech representations like wav2vec2 and HuBERT exhibit strong anisotropy, leading to high similarity between random embeddings. While widely observed, the impact of this property on downstream tasks remains unclear. This work evaluates anisotropy in keyword spotting for computational documentary linguistics. Using Dynamic Time Warping, we show that despite anisotropy, wav2vec2 similarity measures effectively identify words without transcription. Our results highlight the robustness of these representations, which capture phonetic structures and generalize across speakers. Our results underscore the importance of pretraining in learning rich and invariant speech representations. 

---
# Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM 

**Authors**: Jeena Prakash, Blessingh Kumar, Kadri Hacioglu, Bidisha Sharma, Sindhuja Gopalan, Malolan Chetlur, Shankar Venkatesan, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2506.11089)  

**Abstract**: Automatic speech recognition (ASR) models rely on high-quality transcribed data for effective training. Generating pseudo-labels for large unlabeled audio datasets often relies on complex pipelines that combine multiple ASR outputs through multi-stage processing, leading to error propagation, information loss and disjoint optimization. We propose a unified multi-ASR prompt-driven framework using postprocessing by either textual or speech-based large language models (LLMs), replacing voting or other arbitration logic for reconciling the ensemble outputs. We perform a comparative study of multiple architectures with and without LLMs, showing significant improvements in transcription accuracy compared to traditional methods. Furthermore, we use the pseudo-labels generated by the various approaches to train semi-supervised ASR models for different datasets, again showing improved performance with textual and speechLLM transcriptions compared to baselines. 

---
# ADAMIX: Adaptive Mixed-Precision Delta-Compression with Quantization Error Optimization for Large Language Models 

**Authors**: Boya Xiong, Shuo Wang, Weifeng Ge, Guanhua Chen, Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11087)  

**Abstract**: Large language models (LLMs) achieve impressive performance on various knowledge-intensive and complex reasoning tasks in different domains. In certain scenarios like multi-tenant serving, a large number of LLMs finetuned from the same base model are deployed to meet complex requirements for users. Recent works explore delta-compression approaches to quantize and compress the delta parameters between the customized LLM and the corresponding base model. However, existing works either exhibit unsatisfactory performance at high compression ratios or depend on empirical bit allocation schemes. In this work, we propose ADAMIX, an effective adaptive mixed-precision delta-compression framework. We provide a mathematical derivation of quantization error to motivate our mixed-precision compression strategy and formulate the optimal mixed-precision bit allocation scheme as the solution to a 0/1 integer linear programming problem. Our derived bit allocation strategy minimizes the quantization error while adhering to a predefined compression ratio requirement. Experimental results on various models and benchmarks demonstrate that our approach surpasses the best baseline by a considerable margin. On tasks like AIME2024 and GQA, where the norm of $\Delta \mathbf{W}$ is large and the base model lacks sufficient ability, ADAMIX outperforms the best baseline Delta-CoMe by 22.3% and 6.1% with 7B models, respectively. 

---
# LeanExplore: A search engine for Lean 4 declarations 

**Authors**: Justin Asher  

**Link**: [PDF](https://arxiv.org/pdf/2506.11085)  

**Abstract**: The expanding Lean 4 ecosystem poses challenges for navigating its vast libraries. This paper introduces LeanExplore, a search engine for Lean 4 declarations. LeanExplore enables users to semantically search for statements, both formally and informally, across select Lean 4 packages (including Batteries, Init, Lean, Mathlib, PhysLean, and Std). This search capability is powered by a hybrid ranking strategy, integrating scores from a multi-source semantic embedding model (capturing conceptual meaning from formal Lean code, docstrings, AI-generated informal translations, and declaration titles), BM25+ for keyword-based lexical relevance, and a PageRank-based score reflecting declaration importance and interconnectedness. The search engine is accessible via a dedicated website (this https URL) and a Python API (this https URL). Furthermore, the database can be downloaded, allowing users to self-host the service. LeanExplore integrates easily with LLMs via the model context protocol (MCP), enabling users to chat with an AI assistant about Lean declarations or utilize the search engine for building theorem-proving agents. This work details LeanExplore's architecture, data processing, functionalities, and its potential to enhance Lean 4 workflows and AI-driven mathematical research 

---
# Improving Child Speech Recognition and Reading Mistake Detection by Using Prompts 

**Authors**: Lingyun Gao, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2506.11079)  

**Abstract**: Automatic reading aloud evaluation can provide valuable support to teachers by enabling more efficient scoring of reading exercises. However, research on reading evaluation systems and applications remains limited. We present a novel multimodal approach that leverages audio and knowledge from text resources. In particular, we explored the potential of using Whisper and instruction-tuned large language models (LLMs) with prompts to improve transcriptions for child speech recognition, as well as their effectiveness in downstream reading mistake detection. Our results demonstrate the effectiveness of prompting Whisper and prompting LLM, compared to the baseline Whisper model without prompting. The best performing system achieved state-of-the-art recognition performance in Dutch child read speech, with a word error rate (WER) of 5.1%, improving the baseline WER of 9.4%. Furthermore, it significantly improved reading mistake detection, increasing the F1 score from 0.39 to 0.73. 

---
# Can We Trust Machine Learning? The Reliability of Features from Open-Source Speech Analysis Tools for Speech Modeling 

**Authors**: Tahiya Chowdhury, Veronica Romero  

**Link**: [PDF](https://arxiv.org/pdf/2506.11072)  

**Abstract**: Machine learning-based behavioral models rely on features extracted from audio-visual recordings. The recordings are processed using open-source tools to extract speech features for classification models. These tools often lack validation to ensure reliability in capturing behaviorally relevant information. This gap raises concerns about reproducibility and fairness across diverse populations and contexts. Speech processing tools, when used outside of their design context, can fail to capture behavioral variations equitably and can then contribute to bias. We evaluate speech features extracted from two widely used speech analysis tools, OpenSMILE and Praat, to assess their reliability when considering adolescents with autism. We observed considerable variation in features across tools, which influenced model performance across context and demographic groups. We encourage domain-relevant verification to enhance the reliability of machine learning models in clinical applications. 

---
# Regularized Federated Learning for Privacy-Preserving Dysarthric and Elderly Speech Recognition 

**Authors**: Tao Zhong, Mengzhe Geng, Shujie Hu, Guinan Li, Xunying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11069)  

**Abstract**: Accurate recognition of dysarthric and elderly speech remains challenging to date. While privacy concerns have driven a shift from centralized approaches to federated learning (FL) to ensure data confidentiality, this further exacerbates the challenges of data scarcity, imbalanced data distribution and speaker heterogeneity. To this end, this paper conducts a systematic investigation of regularized FL techniques for privacy-preserving dysarthric and elderly speech recognition, addressing different levels of the FL process by 1) parameter-based, 2) embedding-based and 3) novel loss-based regularization. Experiments on the benchmark UASpeech dysarthric and DementiaBank Pitt elderly speech corpora suggest that regularized FL systems consistently outperform the baseline FedAvg system by statistically significant WER reductions of up to 0.55\% absolute (2.13\% relative). Further increasing communication frequency to one exchange per batch approaches centralized training performance. 

---
# PMF-CEC: Phoneme-augmented Multimodal Fusion for Context-aware ASR Error Correction with Error-specific Selective Decoding 

**Authors**: Jiajun He, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.11064)  

**Abstract**: End-to-end automatic speech recognition (ASR) models often struggle to accurately recognize rare words. Previously, we introduced an ASR postprocessing method called error detection and context-aware error correction (ED-CEC), which leverages contextual information such as named entities and technical terms to improve the accuracy of ASR transcripts. Although ED-CEC achieves a notable success in correcting rare words, its accuracy remains low when dealing with rare words that have similar pronunciations but different spellings. To address this issue, we proposed a phoneme-augmented multimodal fusion method for context-aware error correction (PMF-CEC) method on the basis of ED-CEC, which allowed for better differentiation between target rare words and homophones. Additionally, we observed that the previous ASR error detection module suffers from overdetection. To mitigate this, we introduced a retention probability mechanism to filter out editing operations with confidence scores below a set threshold, preserving the original operation to improve error detection accuracy. Experiments conducted on five datasets demonstrated that our proposed PMF-CEC maintains reasonable inference speed while further reducing the biased word error rate compared with ED-CEC, showing a stronger advantage in correcting homophones. Moreover, our method outperforms other contextual biasing methods, and remains valuable compared with LLM-based methods in terms of faster inference and better robustness under large biasing lists. 

---
# CodeMirage: A Multi-Lingual Benchmark for Detecting AI-Generated and Paraphrased Source Code from Production-Level LLMs 

**Authors**: Hanxi Guo, Siyuan Cheng, Kaiyuan Zhang, Guangyu Shen, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11059)  

**Abstract**: Large language models (LLMs) have become integral to modern software development, producing vast amounts of AI-generated source code. While these models boost programming productivity, their misuse introduces critical risks, including code plagiarism, license violations, and the propagation of insecure programs. As a result, robust detection of AI-generated code is essential. To support the development of such detectors, a comprehensive benchmark that reflects real-world conditions is crucial. However, existing benchmarks fall short -- most cover only a limited set of programming languages and rely on less capable generative models. In this paper, we present CodeMirage, a comprehensive benchmark that addresses these limitations through three major advancements: (1) it spans ten widely used programming languages, (2) includes both original and paraphrased code samples, and (3) incorporates outputs from ten state-of-the-art production-level LLMs, including both reasoning and non-reasoning models from six major providers. Using CodeMirage, we evaluate ten representative detectors across four methodological paradigms under four realistic evaluation configurations, reporting results using three complementary metrics. Our analysis reveals nine key findings that uncover the strengths and weaknesses of current detectors, and identify critical challenges for future work. We believe CodeMirage offers a rigorous and practical testbed to advance the development of robust and generalizable AI-generated code detectors. 

---
# Large Language models for Time Series Analysis: Techniques, Applications, and Challenges 

**Authors**: Feifei Shi, Xueyan Yin, Kang Wang, Wanyu Tu, Qifu Sun, Huansheng Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.11040)  

**Abstract**: Time series analysis is pivotal in domains like financial forecasting and biomedical monitoring, yet traditional methods are constrained by limited nonlinear feature representation and long-term dependency capture. The emergence of Large Language Models (LLMs) offers transformative potential by leveraging their cross-modal knowledge integration and inherent attention mechanisms for time series analysis. However, the development of general-purpose LLMs for time series from scratch is still hindered by data diversity, annotation scarcity, and computational requirements. This paper presents a systematic review of pre-trained LLM-driven time series analysis, focusing on enabling techniques, potential applications, and open challenges. First, it establishes an evolutionary roadmap of AI-driven time series analysis, from the early machine learning era, through the emerging LLM-driven paradigm, to the development of native temporal foundation models. Second, it organizes and systematizes the technical landscape of LLM-driven time series analysis from a workflow perspective, covering LLMs' input, optimization, and lightweight stages. Finally, it critically examines novel real-world applications and highlights key open challenges that can guide future research and innovation. The work not only provides valuable insights into current advances but also outlines promising directions for future development. It serves as a foundational reference for both academic and industrial researchers, paving the way for the development of more efficient, generalizable, and interpretable systems of LLM-driven time series analysis. 

---
# Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity 

**Authors**: Moussa Koulako Bala Doumbouya, Dan Jurafsky, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2506.11035)  

**Abstract**: Work in psychology has highlighted that the geometric model of similarity standard in deep learning is not psychologically plausible because its metric properties such as symmetry do not align with human perception. In contrast, Tversky (1977) proposed an axiomatic theory of similarity based on a representation of objects as sets of features, and their similarity as a function of common and distinctive features. However, this model has not been used in deep learning before, partly due to the challenge of incorporating discrete set operations. We develop a differentiable parameterization of Tversky's similarity that is learnable through gradient descent, and derive neural network building blocks such as the Tversky projection layer, which unlike the linear projection layer can model non-linear functions such as XOR. Through experiments with image recognition and language modeling, we show that the Tversky projection layer is a beneficial replacement for the linear projection layer, which employs geometric similarity. On the NABirds image classification task, a frozen ResNet-50 adapted with a Tversky projection layer achieves a 24.7% relative accuracy improvement over the linear layer adapter baseline. With Tversky projection layers, GPT-2's perplexity on PTB decreases by 7.5%, and its parameter count by 34.8%. Finally, we propose a unified interpretation of both projection layers as computing similarities of input stimuli to learned prototypes, for which we also propose a novel visualization technique highlighting the interpretability of Tversky projection layers. Our work offers a new paradigm for thinking about the similarity model implicit in deep learning, and designing networks that are interpretable under an established theory of psychological similarity. 

---
# CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models 

**Authors**: Aneesh Komanduri, Karuna Bhaila, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11034)  

**Abstract**: Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability. Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA). Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks. We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs. Our CausalVLBench encompasses three representative tasks: causal structure inference, intervention target prediction, and counterfactual prediction. We evaluate the ability of state-of-the-art open-source LVLMs on our causal reasoning tasks across three causal representation learning datasets and demonstrate their fundamental strengths and weaknesses. We hope that our benchmark elucidates the drawbacks of existing vision-language models and motivates new directions and paradigms in improving the visual causal reasoning abilities of LVLMs. 

---
# Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models 

**Authors**: Zoher Kachwala, Danishjeet Singh, Danielle Yang, Filippo Menczer  

**Link**: [PDF](https://arxiv.org/pdf/2506.11031)  

**Abstract**: As image generators produce increasingly realistic images, concerns about potential misuse continue to grow. Supervised detection relies on large, curated datasets and struggles to generalize across diverse generators. In this work, we investigate the use of pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. While off-the-shelf VLMs exhibit some task-specific reasoning and chain-of-thought prompting offers gains, we show that task-aligned prompting elicits more focused reasoning and significantly improves performance without fine-tuning. Specifically, prefixing the model's response with the phrase ``Let's examine the style and the synthesis artifacts'' -- a method we call zero-shot-s$^2$ -- boosts Macro F1 scores by 8%-29% for two widely used open-source models. These gains are consistent across three recent, diverse datasets spanning human faces, objects, and animals with images generated by 16 different models -- demonstrating strong generalization. We further evaluate the approach across three additional model sizes and observe improvements in most dataset-model combinations -- suggesting robustness to model scale. Surprisingly, self-consistency, a behavior previously observed in language reasoning, where aggregating answers from diverse reasoning paths improves performance, also holds in this setting. Even here, zero-shot-s$^2$ scales better than chain-of-thought in most cases -- indicating that it elicits more useful diversity. Our findings show that task-aligned prompts elicit more focused reasoning and enhance latent capabilities in VLMs, like the detection of AI-generated images -- offering a simple, generalizable, and explainable alternative to supervised methods. Our code is publicly available on github: this https URL. 

---
# Security Degradation in Iterative AI Code Generation -- A Systematic Analysis of the Paradox 

**Authors**: Shivani Shukla, Himanshu Joshi, Romilla Syed  

**Link**: [PDF](https://arxiv.org/pdf/2506.11022)  

**Abstract**: The rapid adoption of Large Language Models(LLMs) for code generation has transformed software development, yet little attention has been given to how security vulnerabilities evolve through iterative LLM feedback. This paper analyzes security degradation in AI-generated code through a controlled experiment with 400 code samples across 40 rounds of "improvements" using four distinct prompting strategies. Our findings show a 37.6% increase in critical vulnerabilities after just five iterations, with distinct vulnerability patterns emerging across different prompting approaches. This evidence challenges the assumption that iterative LLM refinement improves code security and highlights the essential role of human expertise in the loop. We propose practical guidelines for developers to mitigate these risks, emphasizing the need for robust human validation between LLM iterations to prevent the paradoxical introduction of new security issues during supposedly beneficial code "improvements". 

---
# A Survey of Task-Oriented Knowledge Graph Reasoning: Status, Applications, and Prospects 

**Authors**: Guanglin Niu, Bo Li, Yangguang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11012)  

**Abstract**: Knowledge graphs (KGs) have emerged as a powerful paradigm for structuring and leveraging diverse real-world knowledge, which serve as a fundamental technology for enabling cognitive intelligence systems with advanced understanding and reasoning capabilities. Knowledge graph reasoning (KGR) aims to infer new knowledge based on existing facts in KGs, playing a crucial role in applications such as public security intelligence, intelligent healthcare, and financial risk assessment. From a task-centric perspective, existing KGR approaches can be broadly classified into static single-step KGR, static multi-step KGR, dynamic KGR, multi-modal KGR, few-shot KGR, and inductive KGR. While existing surveys have covered these six types of KGR tasks, a comprehensive review that systematically summarizes all KGR tasks particularly including downstream applications and more challenging reasoning paradigms remains lacking. In contrast to previous works, this survey provides a more comprehensive perspective on the research of KGR by categorizing approaches based on primary reasoning tasks, downstream application tasks, and potential challenging reasoning tasks. Besides, we explore advanced techniques, such as large language models (LLMs), and their impact on KGR. This work aims to highlight key research trends and outline promising future directions in the field of KGR. 

---
# Developing a Dyslexia Indicator Using Eye Tracking 

**Authors**: Kevin Cogan, Vuong M. Ngo, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2506.11004)  

**Abstract**: Dyslexia, affecting an estimated 10% to 20% of the global population, significantly impairs learning capabilities, highlighting the need for innovative and accessible diagnostic methods. This paper investigates the effectiveness of eye-tracking technology combined with machine learning algorithms as a cost-effective alternative for early dyslexia detection. By analyzing general eye movement patterns, including prolonged fixation durations and erratic saccades, we proposed an enhanced solution for determining eye-tracking-based dyslexia features. A Random Forest Classifier was then employed to detect dyslexia, achieving an accuracy of 88.58\%. Additionally, hierarchical clustering methods were applied to identify varying severity levels of dyslexia. The analysis incorporates diverse methodologies across various populations and settings, demonstrating the potential of this technology to identify individuals with dyslexia, including those with borderline traits, through non-invasive means. Integrating eye-tracking with machine learning represents a significant advancement in the diagnostic process, offering a highly accurate and accessible method in clinical research. 

---
# Glider: Global and Local Instruction-Driven Expert Router 

**Authors**: Pingzhi Li, Prateek Yadav, Jaehong Yoon, Jie Peng, Yi-Lin Sung, Mohit Bansal, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2410.07172)  

**Abstract**: The availability of performant pre-trained models has led to a proliferation of fine-tuned expert models that are specialized to particular domains. This has enabled the creation of powerful and adaptive routing-based "Model MoErging" methods with the goal of using expert modules to create an aggregate system with improved performance or generalization. However, existing MoErging methods often prioritize generalization to unseen tasks at the expense of performance on held-in tasks, which limits its practical applicability in real-world deployment scenarios. We observe that current token-level routing mechanisms neglect the global semantic context of the input task. This token-wise independence hinders effective expert selection for held-in tasks, as routing decisions fail to incorporate the semantic properties of the task. To address this, we propose, Global and Local Instruction Driven Expert Router (GLIDER) that integrates a multi-scale routing mechanism, encompassing a semantic global router and a learned local router. The global router leverages LLM's advanced reasoning capabilities for semantic-related contexts to enhance expert selection. Given the input query and LLM, the router generates semantic task instructions that guide the retrieval of the most relevant experts across all layers. This global guidance is complemented by a local router that facilitates token-level routing decisions within each module, enabling finer control and enhanced performance on unseen tasks. Our experiments using T5-based models for T0 and FLAN tasks demonstrate that GLIDER achieves substantially improved held-in performance while maintaining strong generalization on held-out tasks. We also perform ablations experiments to dive deeper into the components of GLIDER. Our experiments highlight the importance of our multi-scale routing that leverages LLM-driven semantic reasoning for MoErging methods. 

---
