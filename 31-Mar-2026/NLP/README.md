# Adaptive Block-Scaled Data Types 

**Authors**: Jack Cook, Hyemin S. Lee, Kathryn Le, Junxian Guo, Giovanni Traverso, Anantha P. Chandrakasan, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.28765)  

**Abstract**: NVFP4 has grown increasingly popular as a 4-bit format for quantizing large language models due to its hardware support and its ability to retain useful information with relatively few bits per parameter. However, the format is not without limitations: recent work has shown that NVFP4 suffers from its error distribution, resulting in large amounts of quantization error on near-maximal values in each group of 16 values. In this work, we leverage this insight to design new Adaptive Block-Scaled Data Types that can adapt to the distribution of their input values. For four-bit quantization, our proposed IF4 (Int/Float 4) data type selects between FP4 and INT4 representations for each group of 16 values, which are then scaled by an E4M3 scale factor as is done with NVFP4. The selected data type is denoted using the scale factor's sign bit, which is currently unused in NVFP4, and we apply the same insight to design formats for other bit-widths, including IF3 and IF6. When used to quantize language models, we find that IF4 outperforms existing 4-bit block-scaled formats, achieving lower loss during quantized training and achieving higher accuracy on many tasks in post-training quantization. We additionally design and evaluate an IF4 Multiply-Accumulate (MAC) unit to demonstrate that IF4 can be implemented efficiently in next-generation hardware accelerators. Our code is available at this https URL. 

---
# EpiScreen: Early Epilepsy Detection from Electronic Health Records with Large Language Models 

**Authors**: Shuang Zhou, Kai Yu, Zaifu Zhan, Huixue Zhou, Min Zeng, Feng Xie, Zhiyi Sha, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.28698)  

**Abstract**: Epilepsy and psychogenic non-epileptic seizures often present with similar seizure-like manifestations but require fundamentally different management strategies. Misdiagnosis is common and can lead to prolonged diagnostic delays, unnecessary treatments, and substantial patient morbidity. Although prolonged video-electroencephalography is the diagnostic gold standard, its high cost and limited accessibility hinder timely diagnosis. Here, we developed a low-cost, effective approach, EpiScreen, for early epilepsy detection by utilizing routinely collected clinical notes from electronic health records. Through fine-tuning large language models on labeled notes, EpiScreen achieved an AUC of up to 0.875 on the MIMIC-IV dataset and 0.980 on a private cohort of the University of Minnesota. In a clinician-AI collaboration setting, EpiScreen-assisted neurologists outperformed unaided experts by up to 10.9%. Overall, this study demonstrates that EpiScreen supports early epilepsy detection, facilitating timely and cost-effective screening that may reduce diagnostic delays and avoid unnecessary interventions, particularly in resource-limited regions. 

---
# Training data generation for context-dependent rubric-based short answer grading 

**Authors**: Pavel Šindelář, Dávid Slivka, Christopher Bouma, Filip Prášil, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2603.28537)  

**Abstract**: Every 4 years, the PISA test is administered by the OECD to test the knowledge of teenage students worldwide and allow for comparisons of educational systems. However, having to avoid language differences and annotator bias makes the grading of student answers challenging. For these reasons, it would be interesting to compare methods of automatic student answer grading. To train some of these methods, which require machine learning, or to compute parameters or select hyperparameters for those that do not, a large amount of domain-specific data is needed. In this work, we explore a small number of methods for creating a large-scale training dataset using only a relatively small confidential dataset as a reference, leveraging a set of very simple derived text formats to preserve confidentiality. Using these methods, we successfully created three surrogate datasets that are, at the very least, superficially more similar to the reference dataset than purely the result of prompt-based generation. Early experiments suggest one of these approaches might also lead to improved model training. 

---
# Compressing Transformer Language Models via Matrix Product Operator Decomposition: A Case Study on PicoGPT 

**Authors**: Younes Javanmard, Tanmoy Pandit, Masoud Mardani  

**Link**: [PDF](https://arxiv.org/pdf/2603.28534)  

**Abstract**: Transformer-based language models achieve strong performance across NLP tasks, but their quadratic parameter scaling with hidden dimension makes deployment on resource-constrained hardware expensive. We study Matrix Product Operator (MPO) decomposition as a principled compression method for transformers. MPO factorises weight matrices into chains of low-rank cores, with approximation quality controlled by the bond dimension chi. We replace every this http URL layer in PicoGPT, a GPT-2-style character-level language model with about 1M parameters, with an MPOLinear module parameterised as an MPO chain. Cores are initialised either by TT-SVD from pretrained dense weights or from random initialisation, and trained using standard PyTorch autograd without a custom backward pass. We derive balanced factorisation schemes for the five distinct weight shapes in PicoGPT and evaluate bond dimensions chi in {4, 8, 16, 32} on Tiny Shakespeare. MPO compression achieves up to 13x compression per transformer block at chi = 4. At chi = 16, the model uses 191,872 parameters instead of 1,020,224 while retaining 97.7% of baseline token accuracy (51.6% vs 52.8%). Reconstruction error follows the expected trend and is lower for three-site than two-site factorisations at the same bond dimension. The chi = 8 model gives the best accuracy per parameter, exceeding the dense baseline by 2.7x on this metric. These results show that MPO parameterisation is a practical and theoretically grounded alternative to low-rank methods and unstructured pruning for transformer compression. 

---
# GraphWalker: Agentic Knowledge Graph Question Answering via Synthetic Trajectory Curriculum 

**Authors**: Shuwen Xu, Yao Xu, Jiaxiang Liu, Chenhao Yuan, Wenshuo Peng, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28533)  

**Abstract**: Agentic knowledge graph question answering (KGQA) requires an agent to iteratively interact with knowledge graphs (KGs), posing challenges in both training data scarcity and reasoning generalization. Specifically, existing approaches often restrict agent exploration: prompting-based methods lack autonomous navigation training, while current training pipelines usually confine reasoning to predefined trajectories. To this end, this paper proposes \textit{GraphWalker}, a novel agentic KGQA framework that addresses these challenges through \textit{Automated Trajectory Synthesis} and \textit{Stage-wise Fine-tuning}. GraphWalker adopts a two-stage SFT training paradigm: First, the agent is trained on structurally diverse trajectories synthesized from constrained random-walk paths, establishing a broad exploration prior over the KG; Second, the agent is further fine-tuned on a small set of expert trajectories to develop reflection and error recovery capabilities. Extensive experiments demonstrate that our stage-wise SFT paradigm unlocks a higher performance ceiling for a lightweight reinforcement learning (RL) stage, enabling GraphWalker to achieve state-of-the-art performance on CWQ and WebQSP. Additional results on GrailQA and our constructed GraphWalkerBench confirm that GraphWalker enhances generalization to out-of-distribution reasoning paths. The code is publicly available at this https URL 

---
# EarlySciRev: A Dataset of Early-Stage Scientific Revisions Extracted from LaTeX Writing Traces 

**Authors**: Léane Jourdan, Julien Aubert-Béduchaud, Yannis Chupin, Marah Baccari, Florian Boudin  

**Link**: [PDF](https://arxiv.org/pdf/2603.28515)  

**Abstract**: Scientific writing is an iterative process that generates rich revision traces, yet publicly available resources typically expose only final or near-final versions of papers. This limits empirical study of revision behaviour and evaluation of large language models (LLMs) for scientific writing. We introduce EarlySciRev, a dataset of early-stage scientific text revisions automatically extracted from arXiv LaTeX source files. Our key observation is that commented-out text in LaTeX often preserves discarded or alternative formulations written by the authors themselves. By aligning commented segments with nearby final text, we extract paragraph-level candidate revision pairs and apply LLM-based filtering to retain genuine revisions. Starting from 1.28M candidate pairs, our pipeline yields 578k validated revision pairs, grounded in authentic early drafting traces. We additionally provide a human-annotated benchmark for revision detection. EarlySciRev complements existing resources focused on late-stage revisions or synthetic rewrites and supports research on scientific writing dynamics, revision modelling, and LLM-assisted editing. 

---
# TIEG-Youpu Solution for NeurIPS 2022 WikiKG90Mv2-LSC 

**Authors**: Feng Nie, Zhixiu Ye, Sifa Xie, Shuang Wu, Xin Yuan, Liang Yao, Jiazhen Peng, Xu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.28512)  

**Abstract**: WikiKG90Mv2 in NeurIPS 2022 is a large encyclopedic knowledge graph. Embedding knowledge graphs into continuous vector spaces is important for many practical applications, such as knowledge acquisition, question answering, and recommendation systems. Compared to existing knowledge graphs, WikiKG90Mv2 is a large scale knowledge graph, which is composed of more than 90 millions of entities. Both efficiency and accuracy should be considered when building graph embedding models for knowledge graph at scale. To this end, we follow the retrieve then re-rank pipeline, and make novel modifications in both retrieval and re-ranking stage. Specifically, we propose a priority infilling retrieval model to obtain candidates that are structurally and semantically similar. Then we propose an ensemble based re-ranking model with neighbor enhanced representations to produce final link prediction results among retrieved candidates. Experimental results show that our proposed method outperforms existing baseline methods and improves MRR of validation set from 0.2342 to 0.2839. 

---
# Courtroom-Style Multi-Agent Debate with Progressive RAG and Role-Switching for Controversial Claim Verification 

**Authors**: Masnun Nuha Chowdhury, Nusrat Jahan Beg, Umme Hunny Khan, Syed Rifat Raiyan, Md Kamrul Hasan, Hasan Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2603.28488)  

**Abstract**: Large language models (LLMs) remain unreliable for high-stakes claim verification due to hallucinations and shallow reasoning. While retrieval-augmented generation (RAG) and multi-agent debate (MAD) address this, they are limited by one-pass retrieval and unstructured debate dynamics. We propose a courtroom-style multi-agent framework, PROClaim, that reformulates verification as a structured, adversarial deliberation. Our approach integrates specialized roles (e.g., Plaintiff, Defense, Judge) with Progressive RAG (P-RAG) to dynamically expand and refine the evidence pool during the debate. Furthermore, we employ evidence negotiation, self-reflection, and heterogeneous multi-judge aggregation to enforce calibration, robustness, and diversity. In zero-shot evaluations on the Check-COVID benchmark, PROClaim achieves 81.7% accuracy, outperforming standard multi-agent debate by 10.0 percentage points, with P-RAG driving the primary performance gains (+7.5 pp). We ultimately demonstrate that structural deliberation and model heterogeneity effectively mitigate systematic biases, providing a robust foundation for reliable claim verification. Our code and data are publicly available at this https URL. 

---
# Structural-Ambiguity-Aware Translation from Natural Language to Signal Temporal Logic 

**Authors**: Kosei Fushimi, Kazunobu Serizawa, Junya Ikemoto, Kazumune Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2603.28426)  

**Abstract**: Signal Temporal Logic (STL) is widely used to specify timed and safety-critical tasks for cyber-physical systems, but writing STL formulas directly is difficult for non-expert users. Natural language (NL) provides a convenient interface, yet its inherent structural ambiguity makes one-to-one translation into STL unreliable. In this paper, we propose an \textit{ambiguity-preserving} method for translating NL task descriptions into STL candidate formulas. The key idea is to retain multiple plausible syntactic analyses instead of forcing a single interpretation at the parsing stage. To this end, we develop a three-stage pipeline based on Combinatory Categorial Grammar (CCG): ambiguity-preserving $n$-best parsing, STL-oriented template-based semantic composition, and canonicalization with score aggregation. The proposed method outputs a deduplicated set of STL candidates with plausibility scores, thereby explicitly representing multiple possible formal interpretations of an ambiguous instruction. In contrast to existing one-best NL-to-logic translation methods, the proposed approach is designed to preserve attachment and scope ambiguity. Case studies on representative task descriptions demonstrate that the method generates multiple STL candidates for genuinely ambiguous inputs while collapsing unambiguous or canonically equivalent derivations to a single STL formula. 

---
# LombardoGraphia: Automatic Classification of Lombard Orthography Variants 

**Authors**: Edoardo Signoroni, Pavel Rychlý  

**Link**: [PDF](https://arxiv.org/pdf/2603.28418)  

**Abstract**: Lombard, an underresourced language variety spoken by approximately 3.8 million people in Northern Italy and Southern Switzerland, lacks a unified orthographic standard. Multiple orthographic systems exist, creating challenges for NLP resource development and model training. This paper presents the first study of automatic Lombard orthography classification and LombardoGraphia, a curated corpus of 11,186 Lombard Wikipedia samples tagged across 9 orthographic variants, and models for automatic orthography classification. We curate the dataset, processing and filtering raw Wikipedia content to ensure text suitable for orthographic analysis. We train 24 traditional and neural classification models with various features and encoding levels. Our best models achieve 96.06% and 85.78% overall and average class accuracy, though performance on minority classes remains challenging due to data imbalance. Our work provides crucial infrastructure for building variety-aware NLP resources for Lombard. 

---
# Marco DeepResearch: Unlocking Efficient Deep Research Agents via Verification-Centric Design 

**Authors**: Bin Zhu, Qianghuai Jia, Tian Lan, Junyang Ren, Feng Gu, Feihu Jiang, Longyue Wang, Zhao Xu, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.28376)  

**Abstract**: Deep research agents autonomously conduct open-ended investigations, integrating complex information retrieval with multi-step reasoning across diverse sources to solve real-world problems. To sustain this capability on long-horizon tasks, reliable verification is critical during both training and inference. A major bottleneck in existing paradigms stems from the lack of explicit verification mechanisms in QA data synthesis, trajectory construction, and test-time scaling. Errors introduced at each stage propagate downstream and degrade the overall agent performance. To address this, we present Marco DeepResearch, a deep research agent optimized with a verification-centric framework design at three levels: \textbf{(1)~QA Data Synthesis:} We introduce verification mechanisms to graph-based and agent-based QA synthesis to control question difficulty while ensuring answers are unique and correct; \textbf{(2)~Trajectory Construction:} We design a verification-driven trajectory synthesis method that injects explicit verification patterns into training trajectories; and \textbf{(3)~Test-time scaling:} We use Marco DeepResearch itself as a verifier at inference time and effectively improve performance on challenging questions. Extensive experimental results demonstrate that our proposed Marco DeepResearch agent significantly outperforms 8B-scale deep research agents on most challenging benchmarks, such as BrowseComp and BrowseComp-ZH. Crucially, under a maximum budget of 600 tool calls, Marco DeepResearch even surpasses or approaches several 30B-scale agents, like Tongyi DeepResearch-30B. 

---
# Tailoring AI-Driven Reading Scaffolds to the Distinct Needs of Neurodiverse Learners 

**Authors**: Soufiane Jhilal, Eleonora Pasqua, Caterina Marchesi, Riccardo Corradi, Martina Galletti  

**Link**: [PDF](https://arxiv.org/pdf/2603.28370)  

**Abstract**: Neurodiverse learners often require reading supports, yet increasing scaffold richness can sometimes overload attention and working memory rather than improve comprehension. Grounded in the Construction-Integration model and a contingent scaffolding perspective, we examine how structural versus semantic scaffolds shape comprehension and reading experience in a supervised inclusive context. Using an adapted reading interface, we compared four modalities: unmodified text, sentence-segmented text, segmented text with pictograms, and segmented text with pictograms plus keyword labels. In a within-subject pilot with 14 primary-school learners with special educational needs and disabilities, we measured reading comprehension using standardized questions and collected brief child- and therapist-reported experience measures alongside open-ended feedback. Results highlight heterogeneous responses as some learners showed patterns consistent with benefits from segmentation and pictograms, while others showed patterns consistent with increased coordination costs when visual scaffolds were introduced. Experience ratings showed limited differences between modalities, with some apparent effects linked to clinical complexity, particularly for perceived ease of understanding. Open-ended feedback of the learners frequently requested simpler wording and additional visual supports. These findings suggest that no single scaffold is universally optimal, reinforcing the need for calibrated, adjustable scaffolding and provide design implications for human-AI co-regulation in supervised inclusive reading contexts. 

---
# Not All Subjectivity Is the Same! Defining Desiderata for the Evaluation of Subjectivity in NLP 

**Authors**: Urja Khurana, Michiel van der Meer, Enrico Liscio, Antske Fokkens, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2603.28351)  

**Abstract**: Subjective judgments are part of several NLP datasets and recent work is increasingly prioritizing models whose outputs reflect this diversity of perspectives. Such responses allow us to shed light on minority voices, which are frequently marginalized or obscured by dominant perspectives. It remains a question whether our evaluation practices align with these models' objectives. This position paper proposes seven evaluation desiderata for subjectivity-sensitive models, rooted in how subjectivity is represented in NLP data and models. The desiderata are constructed in a top-down approach, keeping in mind the user-centric impact of such models. We scan the experimental setup of 60 papers and show that various aspects of subjectivity are still understudied: the distinction between ambiguous and polyphonic input, whether subjectivity is effectively expressed to the user, and a lack of interplay between different desiderata, amongst other gaps. 

---
# Kernel-Smith: A Unified Recipe for Evolutionary Kernel Optimization 

**Authors**: He Du, Qiming Ge, Jiakai Hu, Aijun Yang, Zheng Cai, Zixian Huang, Sheng Yuan, Qinxiu Cheng, Xinchen Xie, Yicheng Chen, Yining Li, Jiaxing Xie, Huanan Dong, Yaguang Wu, Xiangjun Huang, Jian Yang, Hui Wang, Bowen Zhou, Bowen Li, Qipeng Guo, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.28342)  

**Abstract**: We present Kernel-Smith, a framework for high-performance GPU kernel and operator generation that combines a stable evaluation-driven evolutionary agent with an evolution-oriented post-training recipe. On the agent side, Kernel-Smith maintains a population of executable candidates and iteratively improves them using an archive of top-performing and diverse programs together with structured execution feedback on compilation, correctness, and speedup. To make this search reliable, we build backend-specific evaluation services for Triton on NVIDIA GPUs and Maca on MetaX GPUs. On the training side, we convert long-horizon evolution trajectories into step-centric supervision and reinforcement learning signals by retaining correctness-preserving, high-gain revisions, so that the model is optimized as a strong local improver inside the evolutionary loop rather than as a one-shot generator. Under a unified evolutionary protocol, Kernel-Smith-235B-RL achieves state-of-the-art overall performance on KernelBench with Nvidia Triton backend, attaining the best average speedup ratio and outperforming frontier proprietary models including Gemini-3.0-pro and Claude-4.6-opus. We further validate the framework on the MetaX MACA backend, where our Kernel-Smith-MACA-30B surpasses large-scale counterparts such as DeepSeek-V3.2-think and Qwen3-235B-2507-think, highlighting potential for seamless adaptation across heterogeneous platforms. Beyond benchmark results, the same workflow produces upstream contributions to production systems including SGLang and LMDeploy, demonstrating that LLM-driven kernel optimization can transfer from controlled evaluation to practical deployment. 

---
# The Necessity of Setting Temperature in LLM-as-a-Judge 

**Authors**: Lujun Li, Lama Sleem, Yangjie Xu, Yewei Song, Aolin Jia, Jerome Francois, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2603.28304)  

**Abstract**: LLM-as-a-Judge has emerged as an effective and low-cost paradigm for evaluating text quality and factual correctness. Prior studies have shown substantial agreement between LLM judges and human experts, even on tasks that are difficult to assess automatically. In practice, researchers commonly employ fixed temperature configurations during the evaluation process-with values of 0.1 and 1.0 being the most prevalent choices-a convention that is largely empirical rather than principled. However, recent researches suggest that LLM performance exhibits non-trivial sensitivity to temperature settings, that lower temperatures do not universally yield optimal outcomes, and that such effects are highly task-dependent. This raises a critical research question: does temperature influence judge performance in LLM centric evaluation? To address this, we systematically investigate the relationship between temperature and judge performance through a series of controlled experiments, and further adopt a causal inference framework within our empirical statistical analysis to rigorously examine the direct causal effect of temperature on judge behavior, offering actionable engineering insights for the design of LLM-centric evaluation pipelines. 

---
# Merge and Conquer: Instructing Multilingual Models by Adding Target Language Weights 

**Authors**: Eneko Valero, Maria Ribalta i Albado, Oscar Sainz, Naiara Perez, German Rigau  

**Link**: [PDF](https://arxiv.org/pdf/2603.28263)  

**Abstract**: Large Language Models (LLMs) remain heavily centered on English, with limited performance in low-resource languages. Existing adaptation approaches, such as continual pre-training, demand significant computational resources. In the case of instructed models, high-quality instruction data is also required, both of which are often inaccessible for low-resource language communities. Under these constraints, model merging offers a lightweight alternative, but its potential in low-resource contexts has not been systematically explored. In this work, we explore whether it is possible to transfer language knowledge to an instruction-tuned LLM by merging it with a language-specific base model, thereby eliminating the need of language-specific instructions and repeated fine-tuning processes whenever stronger instructed variants become available. Through experiments covering four Iberian languages (Basque, Catalan, Galician, and Spanish) and two model families, we show that merging enables effective instruction following behavior in new languages and even supports multilingual capability through the combination of multiple language-specific models. Our results indicate that model merging is a viable and efficient alternative to traditional adaptation methods for low-resource languages, achieving competitive performance while greatly reducing computational cost. 

---
# Coconstructions in spoken data: UD annotation guidelines and first results 

**Authors**: Ludovica Pannitto, Sylvain Kahane, Kaja Dobrovoljc, Elena Battaglia, Bruno Guillaume, Caterina Mauri, Eleonora Zucchini  

**Link**: [PDF](https://arxiv.org/pdf/2603.28261)  

**Abstract**: The paper proposes annotation guidelines for syntactic dependencies that span across speaker turns - including collaborative coconstructions proper, wh-question answers, and backchannels - in spoken language treebanks within the Universal Dependencies framework. Two representations are proposed: a speaker-based representation following the segmentation into speech turns, and a dependency-based representation with dependencies across speech turns. New propositions are also put forward to distinguish between reformulations and repairs, and to promote elements in unfinished phrases. 

---
# Categorical Perception in Large Language Model Hidden States: Structural Warping at Digit-Count Boundaries 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2603.28258)  

**Abstract**: Categorical perception (CP) -- enhanced discriminability at category boundaries -- is among the most studied phenomena in perceptual psychology. This paper reports that analogous geometric warping occurs in the hidden-state representations of large language models (LLMs) processing Arabic numerals. Using representational similarity analysis across six models from five architecture families, the study finds that a CP-additive model (log-distance plus a boundary boost) fits the representational geometry better than a purely continuous model at 100% of primary layers in every model tested. The effect is specific to structurally defined boundaries (digit-count transitions at 10 and 100), absent at non-boundary control positions, and absent in the temperature domain where linguistic categories (hot/cold) lack a tokenisation discontinuity. Two qualitatively distinct signatures emerge: "classic CP" (Gemma, Qwen), where models both categorise explicitly and show geometric warping, and "structural CP" (Llama, Mistral, Phi), where geometry warps at the boundary but models cannot report the category distinction. This dissociation is stable across boundaries and is a property of the architecture, not the stimulus. Structural input-format discontinuities are sufficient to produce categorical perception geometry in LLMs, independently of explicit semantic category knowledge. 

---
# \textit{Versteasch du mi?} Computational and Socio-Linguistic Perspectives on GenAI, LLMs, and Non-Standard Language 

**Authors**: Verena Platzgummer, John McCrae, Sina Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2603.28213)  

**Abstract**: The design of Large Language Models and generative artificial intelligence has been shown to be "unfair" to less-spoken languages and to deepen the digital language divide. Critical sociolinguistic work has also argued that these technologies are not only made possible by prior socio-historical processes of linguistic standardisation, often grounded in European nationalist and colonial projects, but also exacerbate epistemologies of language as "monolithic, monolingual, syntactically standardized systems of meaning". In our paper, we draw on earlier work on the intersections of technology and language policy and bring our respective expertise in critical sociolinguistics and computational linguistics to bear on an interrogation of these arguments. We take two different complexes of non-standard linguistic varieties in our respective repertoires--South Tyrolean dialects, which are widely used in informal communication in South Tyrol, Italy, as well as varieties of Kurdish--as starting points to an interdisciplinary exploration of the intersections between GenAI and linguistic variation and standardisation. We discuss both how LLMs can be made to deal with nonstandard language from a technical perspective, and whether, when or how this can contribute to "democratic and decolonial digital and machine learning strategies", which has direct policy implications. 

---
# Beyond Cosine Similarity: Zero-Initialized Residual Complex Projection for Aspect-Based Sentiment Analysis 

**Authors**: Yijin Wang, Fandi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.28205)  

**Abstract**: Aspect-Based Sentiment Analysis (ABSA) is fundamentally challenged by representation entanglement, where aspect semantics and sentiment polarities are often conflated in real-valued embedding spaces. Furthermore, standard contrastive learning suffers from false-negative collisions, severely degrading performance on high-frequency aspects. In this paper, we propose a novel framework featuring a Zero-Initialized Residual Complex Projection (ZRCP) and an Anti-collision Masked Angle Loss,inspired by quantum projection and entanglement ideas. Our approach projects textual features into a complex semantic space, systematically utilizing the phase to disentangle sentiment polarities while allowing the amplitude to encode the semantic intensity and lexical richness of subjective descriptions. To tackle the collision bottleneck, we introduce an anti-collision mask that elegantly preserves intra-polarity aspect cohesion while expanding the inter-polarity discriminative margin by over 50%. Experimental results demonstrate that our framework achieves a state-of-the-art Macro-F1 score of 0.8851. Deep geometric analyses further reveal that explicitly penalizing the complex amplitude catastrophically over-regularizes subjective representations, proving that our unconstrained-amplitude and phase-driven objective is crucial for robust, fine-grained sentiment disentanglement. 

---
# DongYuan: An LLM-Based Framework for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis 

**Authors**: Hua Li, Yingying Li, Xiaobin Feng, Xinyi Fu, Lifeng Dong, Qingfeng Yang, Yanzhe Chen, Xiaoju Feng, Zhidong Cao, Jianbin Guo, Yanru Du  

**Link**: [PDF](https://arxiv.org/pdf/2603.28191)  

**Abstract**: The clinical burden of spleen-stomach disorders is substantial. While large language models (LLMs) offer new potential for medical applications, they face three major challenges in the context of integrative Chinese and Western medicine (ICWM): a lack of high-quality data, the absence of models capable of effectively integrating the reasoning logic of traditional Chinese medicine (TCM) syndrome differentiation with that of Western medical (WM) disease diagnosis, and the shortage of a standardized evaluation benchmark. To address these interrelated challenges, we propose DongYuan, an ICWM spleen-stomach diagnostic framework. Specifically, three ICWM datasets (SSDF-Syndrome, SSDF-Dialogue, and SSDF-PD) were curated to fill the gap in high-quality data for spleen-stomach disorders. We then developed SSDF-Core, a core diagnostic LLM that acquires robust ICWM reasoning capabilities through a two-stage training regimen of supervised fine-tuning. tuning (SFT) and direct preference optimization (DPO), and complemented it with SSDF-Navigator, a pluggable consultation navigation model designed to optimize clinical inquiry strategies. Additionally, we established SSDF-Bench, a comprehensive evaluation benchmark focused on ICWM diagnosis of spleen-stomach disorders. Experimental results demonstrate that SSDF-Core significantly outperforms 12 mainstream baselines on SSDF-Bench. DongYuan lays a solid methodological foundation and provides practical technical references for the future development of intelligent ICWM diagnostic systems. 

---
# From Reviews to Requirements: Can LLMs Generate Human-Like User Stories? 

**Authors**: Shadman Sakib, Oishy Fatema Akhand, Tasnia Tasneem, Shohel Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2603.28163)  

**Abstract**: App store reviews provide a constant flow of real user feedback that can help improve software requirements. However, these reviews are often messy, informal, and difficult to analyze manually at scale. Although automated techniques exist, many do not perform well when replicated and often fail to produce clean, backlog-ready user stories for agile projects.
In this study, we evaluate how well large language models (LLMs) such as GPT-3.5 Turbo, Gemini 2.0 Flash, and Mistral 7B Instruct can generate usable user stories directly from raw app reviews. Using the Mini-BAR dataset of 1,000+ health app reviews, we tested zero-shot, one-shot, and two-shot prompting methods.
We evaluated the generated user stories using both human judgment (via the RUST framework) and a RoBERTa classifier fine-tuned on UStAI to assess their overall quality. Our results show that LLMs can match or even outperform humans in writing fluent, well-formatted user stories, especially when few-shot prompts are used. However, they still struggle to produce independent and unique user stories, which are essential for building a strong agile backlog.
Overall, our findings show how LLMs can reliably turn unstructured app reviews into actionable software requirements, providing developers with clear guidance to turn user feedback into meaningful improvements. 

---
# Who Wrote the Book? Detecting and Attributing LLM Ghostwriters 

**Authors**: Anudeex Shetty, Qiongkai Xu, Olga Ohrimenko, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2603.28054)  

**Abstract**: In this paper, we introduce GhostWriteBench, a dataset for LLM authorship attribution. It comprises long-form texts (50K+ words per book) generated by frontier LLMs, and is designed to test generalisation across multiple out-of-distribution (OOD) dimensions, including domain and unseen LLM author. We also propose TRACE -- a novel fingerprinting method that is interpretable and lightweight -- that works for both open- and closed-source models. TRACE creates the fingerprint by capturing token-level transition patterns (e.g., word rank) estimated by another lightweight language model. Experiments on GhostWriteBench demonstrate that TRACE achieves state-of-the-art performance, remains robust in OOD settings, and works well in limited training data scenarios. 

---
# Transfer Learning for an Endangered Slavic Variety: Dependency Parsing in Pomak Across Contact-Shaped Dialects 

**Authors**: Sercan Karakaş  

**Link**: [PDF](https://arxiv.org/pdf/2603.28033)  

**Abstract**: This paper presents new resources and baselines for Dependency Parsing in Pomak, an endangered Eastern South Slavic language with substantial dialectal variation and no widely adopted standard. We focus on the variety spoken in Turkey (Uzunköprü) and ask how well a dependency parser trained on the existing Pomak Universal Dependencies treebank, which was built primarily from the variety that is spoken in Greece, transfers across dialects. We run two experimental phases. First, we train a parser on the Greek-variety UD data and evaluate zero-shot transfer to Turkish-variety Pomak, quantifying the impact of phonological and morphosyntactic differences. Second, we introduce a new manually annotated Turkish-variety Pomak corpus of 650 sentences and show that, despite its small size, targeted fine-tuning substantially improves accuracy; performance is further boosted by cross-variety transfer learning that combines the two dialects. 

---
# Rethinking Atomic Decomposition for LLM Judges: A Prompt-Controlled Study of Reference-Grounded QA Evaluation 

**Authors**: Xinran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.28005)  

**Abstract**: Atomic decomposition -- breaking a candidate answer into claims before verifying each against a reference -- is a widely adopted design for LLM-based reference-grounded judges. However, atomic prompts are typically richer and longer, making it unclear whether any advantage comes from decomposition or from richer prompting. We study this for benchmark-style completeness-sensitive reference-support classification: classifying a candidate as fully supported, partially supported, or unsupported relative to a supplied reference. We compare a self-decomposing atomic judge (single-prompt decompose-and-verify) against a prompt-controlled holistic judge with the same inputs and a similarly detailed rubric. On 200 source examples per dataset across TruthfulQA, ASQA, and QAMPARI, with four model families, source-level paired tests, cluster bootstrap, and aggregation across three pre-frozen prompt variants per design family, we find the holistic judge matches or exceeds the atomic judge on two of three benchmarks: ASQA and QAMPARI favor holistic across all four families (statistically reliable in three of four), while TruthfulQA shows a small atomic edge. The holistic advantage is concentrated in partially\_supported cases -- incompleteness detection. A sensitivity check against human annotations confirms the ranking under both benchmark-completeness and human factual-correctness standards. Our finding is specific to the self-decomposing single-prompt pattern on three QA-style benchmarks with 200 source examples each; multi-stage atomic pipelines and non-QA tasks remain untested. Among perturbations examined, reference-quality degradation produced the largest accuracy drops for both judge families. 

---
# On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR 

**Authors**: Ganesh Pavan Kartikeya Bharadwaj Kolluri, Michael Kampouridis, Ravi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2603.27981)  

**Abstract**: Automatic speech recognition (ASR) has advanced rapidly in recent years, driven by large-scale pretrained models and end-to-end architectures such as SLAM-ASR. A key component of SLAM-ASR systems is the Whisper speech encoder, which provides robust acoustic representations. While model pruning has been explored for the full Whisper encoder-decoder architecture, its impact within the SLAM-ASR setting remains under-investigated. In this work, we analyze the effects of layer pruning in the Whisper encoder when used as the acoustic backbone of SLAM-ASR. We further examine the extent to which LoRA-based fine-tuning can recover performance degradation caused by pruning. Experiments conducted across three Whisper variants (Small, Medium, Large-v2), three languages representing distinct resource levels (Danish, Dutch, English), and over 200 training runs demonstrate that pruning two encoder layers causes only 2-4% WER degradation, and that combining this pruning with LoRA adaptation consistently outperforms the unpruned baseline while reducing total parameters by 7-14%. Moreover, our error analysis reveals that LoRA primarily compensates through the language model's linguistic priors, reducing total word errors by 11-21% for Dutch and English, with substitutions and deletions showing the largest reductions. However, for low-resource Danish, the reduction is smaller (4-7%), and LoRA introduces increased insertion errors, indicating that compensation effectiveness depends on the LLM's pre-existing language proficiency and available training data. 

---
# EnsemJudge: Enhancing Reliability in Chinese LLM-Generated Text Detection through Diverse Model Ensembles 

**Authors**: Zhuoshang Wang, Yubing Ren, Guoyu Zhao, Xiaowei Zhu, Hao Li, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.27949)  

**Abstract**: Large Language Models (LLMs) are widely applied across various domains due to their powerful text generation capabilities. While LLM-generated texts often resemble human-written ones, their misuse can lead to significant societal risks. Detecting such texts is an essential technique for mitigating LLM misuse, and many detection methods have shown promising results across different datasets. However, real-world scenarios often involve out-of-domain inputs or adversarial samples, which can affect the performance of detection methods to varying degrees. Furthermore, most existing research has focused on English texts, with limited work addressing Chinese text detection. In this study, we propose EnsemJudge, a robust framework for detecting Chinese LLM-generated text by incorporating tailored strategies and ensemble voting mechanisms. We trained and evaluated our system on a carefully constructed Chinese dataset provided by NLPCC2025 Shared Task 1. Our approach outperformed all baseline methods and achieved first place in the task, demonstrating its effectiveness and reliability in Chinese LLM-generated text detection. Our code is available at this https URL. 

---
# Top-down string-to-dependency Neural Machine Translation 

**Authors**: Shuhei Kondo, Katsuhito Sudoh, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2603.27938)  

**Abstract**: Most of modern neural machine translation (NMT) models are based on an encoder-decoder framework with an attention mechanism. While they perform well on standard datasets, they can have trouble in translation of long inputs that are rare or unseen during training. Incorporating target syntax is one approach to dealing with such length-related problems. We propose a novel syntactic decoder that generates a target-language dependency tree in a top-down, left-to-right order. Experiments show that the proposed top-down string-to-tree decoding generalizes better than conventional sequence-to-sequence decoding in translating long inputs that are not observed in the training data. 

---
# Article and Comment Frames Shape the Quality of Online Comments 

**Authors**: Matteo Guida, Yulia Otmakhova, Eduard Hovy, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2603.27889)  

**Abstract**: Framing theory posits that how information is presented shapes audience responses, but computational work has largely ignored audience reactions. While recent work showed that article framing systematically shapes the content of reader responses, this paper asks: Does framing also affect response quality? Analyzing 1M comments across 2.7K news articles, we operationalize quality as comment health (constructive, good-faith contributions). We find that article frames significantly predict comment health while controlling for topic, and that comments that adopt the article frame are healthier than those that depart from it. Further, unhealthy top-level comments tend to generate more unhealthy responses, independent of the frame being used in the comment. Our results establish a link between framing theory and discourse quality, laying the groundwork for downstream applications. We illustrate this potential with a proactive frame-aware LLM- based system to mitigate unhealthy discourse 

---
# HumMusQA: A Human-written Music Understanding QA Benchmark Dataset 

**Authors**: Benno Weck, Pablo Puentes, Andrea Poltronieri, Satyajeet Prabhu, Dmitry Bogdanov  

**Link**: [PDF](https://arxiv.org/pdf/2603.27877)  

**Abstract**: The evaluation of music understanding in Large Audio-Language Models (LALMs) requires a rigorously defined benchmark that truly tests whether models can perceive and interpret music, a standard that current data methodologies frequently fail to meet. This paper introduces a meticulously structured approach to music evaluation, proposing a new dataset of 320 hand-written questions curated and validated by experts with musical training, arguing that such focused, manual curation is superior for probing complex audio comprehension. To demonstrate the use of the dataset, we benchmark six state-of-the-art LALMs and additionally test their robustness to uni-modal shortcuts. 

---
# KazByte: Adapting Qwen models to Kazakh via Byte-level Adapter 

**Authors**: Rauan Akylzhanov  

**Link**: [PDF](https://arxiv.org/pdf/2603.27859)  

**Abstract**: Large language models fragment Kazakh text into many more tokens than equivalent English text, because their tokenizers were built for high-resource languages. This tokenizer tax inflates compute, shortens the effective context window, and weakens the model's grip on Kazakh morphology. We propose to bypass the tokenizer entirely by feeding raw bytes through a small adapter that learns to speak the internal language of a frozen Qwen2.5-7B. Once the adapter is trained, we freeze it and fine-tune only the attention layers of Qwen on Kazakh text. Our central hypothesis is that this two-stage process -- first teach the interface, then adapt the model -- should match or exceed the accuracy of the original Qwen2.5-7B on standard Kazakh benchmarks. This report describes the ByteKaz architecture and training protocol. Empirical validation is ongoing; this version stakes the design and hypotheses for the record. 

---
# What can LLMs tell us about the mechanisms behind polarity illusions in humans? Experiments across model scales and training steps 

**Authors**: Dario Paape  

**Link**: [PDF](https://arxiv.org/pdf/2603.27855)  

**Abstract**: I use the Pythia scaling suite (Biderman et al. 2023) to investigate if and how two well-known polarity illusions, the NPI illusion and the depth charge illusion, arise in LLMs. The NPI illusion becomes weaker and ultimately disappears as model size increases, while the depth charge illusion becomes stronger in larger models. The results have implications for human sentence processing: it may not be necessary to assume "rational inference" mechanisms that convert ill-formed sentences into well-formed ones to explain polarity illusions, given that LLMs cannot plausibly engage in this kind of reasoning, especially at the implicit level of next-token prediction. On the other hand, shallow, "good enough" processing and/or partial grammaticalization of prescriptively ungrammatical structures may both occur in LLMs. I propose a synthesis of different theoretical accounts that is rooted in the basic tenets of construction grammar. 

---
# Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3 

**Authors**: Natapong Nitarach  

**Link**: [PDF](https://arxiv.org/pdf/2603.27844)  

**Abstract**: Majority voting over multiple LLM attempts improves mathematical reasoning, but correlated errors limit the effective sample size. A natural fix: assign structurally different reasoning strategies to different voters to decorrelate errors. We test this Diverse Prompt Mixer in the AIMO~3 competition: 3 models, 23+ experiments, and 50 IMO-level problems on a single H100 80 GB with a 5-hour limit. Every intervention fails. High-temperature sampling already decorrelates errors sufficiently; weaker prompt strategies reduce per-attempt accuracy more than they reduce correlation. Across a 17-point model capability gap and every inference-time optimization we tried, model capability dominates by an order of magnitude. 

---
# ProText: A benchmark dataset for measuring (mis)gendering in long-form texts 

**Authors**: Hadas Kotek, Margit Bowler, Patrick Sonnenberg, Yu'an Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.27838)  

**Abstract**: We introduce ProText, a dataset for measuring gendering and misgendering in stylistically diverse long-form English texts. ProText spans three dimensions: Theme nouns (names, occupations, titles, kinship terms), Theme category (stereotypically male, stereotypically female, gender-neutral/non-gendered), and Pronoun category (masculine, feminine, gender-neutral, none). The dataset is designed to probe (mis)gendering in text transformations such as summarization and rewrites using state-of-the-art Large Language Models, extending beyond traditional pronoun resolution benchmarks and beyond the gender binary. We validated ProText through a mini case study, showing that even with just two prompts and two models, we can draw nuanced insights regarding gender bias, stereotyping, misgendering, and gendering. We reveal systematic gender bias, particularly when inputs contain no explicit gender cues or when models default to heteronormative assumptions. 

---
# Improving Clinical Diagnosis with Counterfactual Multi-Agent Reasoning 

**Authors**: Zhiwen You, Xi Chen, Aniket Vashishtha, Simo Du, Gabriel Erion-Barner, Hongyuan Mei, Hao Peng, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.27820)  

**Abstract**: Clinical diagnosis is a complex reasoning process in which clinicians gather evidence, form hypotheses, and test them against alternative explanations. In medical training, this reasoning is explicitly developed through counterfactual questioning--e.g., asking how a diagnosis would change if a key symptom were absent or altered--to strengthen differential diagnosis skills. As large language model (LLM)-based systems are increasingly used for diagnostic support, ensuring the interpretability of their recommendations becomes critical. However, most existing LLM-based diagnostic agents reason over fixed clinical evidence without explicitly testing how individual findings support or weaken competing diagnoses. In this work, we propose a counterfactual multi-agent diagnostic framework inspired by clinician training that makes hypothesis testing explicit and evidence-grounded. Our framework introduces counterfactual case editing to modify clinical findings and evaluate how these changes affect competing diagnoses. We further define the Counterfactual Probability Gap, a method that quantifies how strongly individual findings support a diagnosis by measuring confidence shifts under these edits. These counterfactual signals guide multi-round specialist discussions, enabling agents to challenge unsupported hypotheses, refine differential diagnoses, and produce more interpretable reasoning trajectories. Across three diagnostic benchmarks and seven LLMs, our method consistently improves diagnostic accuracy over prompting and prior multi-agent baselines, with the largest gains observed in complex and ambiguous cases. Human evaluation further indicates that our framework produces more clinically useful, reliable, and coherent reasoning. These results suggest that incorporating counterfactual evidence verification is an important step toward building reliable AI systems for clinical decision support. 

---
# Conversational Agents and the Understanding of Human Language: Reflections on AI, LLMs, and Cognitive Science 

**Authors**: Andrei Popescu-Belis  

**Link**: [PDF](https://arxiv.org/pdf/2603.27809)  

**Abstract**: In this paper, we discuss the relationship between natural language processing by computers (NLP) and the understanding of the human language capacity, as studied by linguistics and cognitive science. We outline the evolution of NLP from its beginnings until the age of large language models, and highlight for each of its main paradigms some similarities and differences with theories of the human language capacity. We conclude that the evolution of language technology has not substantially deepened our understanding of how human minds process natural language, despite the impressive language abilities attained by current chatbots using artificial neural networks. 

---
# Understanding Teacher Revisions of Large Language Model-Generated Feedback 

**Authors**: Conrad Borchers, Luiz Rodrigues, Newarney Torrezão da Costa, Cleon Xavier, Rafael Ferreira Mello  

**Link**: [PDF](https://arxiv.org/pdf/2603.27806)  

**Abstract**: Large language models (LLMs) increasingly generate formative feedback for students, yet little is known about how teachers revise this feedback before it reaches learners. Teachers' revisions shape what students receive, making revision practices central to evaluating AI classroom tools. We analyze a dataset of 1,349 instances of AI-generated feedback and corresponding teacher-edited explanations from 117 teachers. We examine (i) textual characteristics associated with teacher revisions, (ii) whether revision decisions can be predicted from the AI feedback text, and (iii) how revisions change the pedagogical type of feedback delivered. First, we find that teachers accept AI feedback without modification in about 80% of cases, while edited feedback tends to be significantly longer and subsequently shortened by teachers. Editing behavior varies substantially across teachers: about 50% never edit AI feedback, and only about 10% edit more than two-thirds of feedback instances. Second, machine learning models trained only on the AI feedback text as input features, using sentence embeddings, achieve fair performance in identifying which feedback will be revised (AUC=0.75). Third, qualitative coding shows that when revisions occur, teachers often simplify AI-generated feedback, shifting it away from high-information explanations toward more concise, corrective forms. Together, these findings characterize how teachers engage with AI-generated feedback in practice and highlight opportunities to design feedback systems that better align with teacher priorities while reducing unnecessary editing effort. 

---
# TailNLG: A Multilingual Benchmark Addressing Verbalization of Long-Tail Entities 

**Authors**: Lia Draetta, Michael Oliverio, Virginia Ramón-Ferrer, Pier Felice Balestrucci, Flaviana Corallo, Carlos Badenes-Olmedo, Alessandro Mazzei, Marco Antonio Stranisci, Rossana Damiano  

**Link**: [PDF](https://arxiv.org/pdf/2603.27768)  

**Abstract**: The automatic verbalization of structured knowledge is a key task for making knowledge graphs accessible to non-expert users and supporting retrieval-augmented generation systems. Although recent advances in Data-to-Text generation have improved multilingual coverage, little attention has been paid to potential biases in the verbalization of rare entities, frequently known as long-tail entities. In this work, we present the first systematic study of long-tail entities in Data-to-Text generation. We introduce TailNLG, a new multilingual benchmark in English, Italian, and Spanish, built from Wikidata and covering entities with varying levels of popularity. We evaluate three different families of large language models in zero-shot settings and compare their performance on rare versus common entities, as well as against the established WebNLG benchmark. Our results reveal a consistent bias against long-tail entities: embedding-based scores are lower, and model uncertainty is higher for rare entities. We further show that the impact of long-tail entities varies across models and languages, and that existing evaluation metrics do not consistently capture these differences, highlighting the need for more reliable evaluation frameworks. 

---
# Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG 

**Authors**: Boxi Yu, Yuzhong Zhang, Liting Lin, Lionel Briand, Emir Muñoz  

**Link**: [PDF](https://arxiv.org/pdf/2603.27752)  

**Abstract**: Large language models (LLMs) continue to hallucinate in retrieval-augmented generation (RAG), producing claims that are unsupported by or conflict with the retrieved context. Detecting such errors remains challenging when faithfulness is evaluated solely with respect to the retrieved context. Existing approaches either provide coarse-grained, answer-level scores or focus on open-domain factuality, often lacking fine-grained, evidence-grounded diagnostics.
We present RT4CHART, a retromorphic testing framework for context-faithfulness assessment. RT4CHART decomposes model outputs into independently verifiable claims and performs hierarchical, local-to-global verification against the retrieved context. Each claim is assigned one of three labels: entailed, contradicted, or baseless. Furthermore, RT4CHART maps claim-level decisions back to specific answer spans and retrieves explicit supporting or refuting evidence from the context, enabling fine-grained and interpretable auditing.
We evaluate RT4CHART on RAGTruth++ (408 samples) and RAGTruth-Enhance (2,675 samples), a newly re-annotated benchmark. RT4CHART achieves the best answer-level hallucination detection F1 among all baselines. On RAGTruth++, it reaches an F1 score of 0.776, outperforming the strongest baseline by 83%. On RAGTruth-Enhance, it achieves a span-level F1 of 47.5%.
Ablation studies show that the hierarchical verification design is the primary driver of performance gains. Finally, our re-annotation reveals 1.68x more hallucination cases than the original labels, suggesting that existing benchmarks substantially underestimate the prevalence of hallucinations. 

---
# KAT-Coder-V2 Technical Report 

**Authors**: Fengxiang Li, Han Zhang, Haoyang Huang, Jinghui Wang, Jinhua Hao, Kun Yuan, Mengtong Li, Minglei Zhang, Pengcheng Xu, Wenhao Zhuang, Yizhen Shao, Zongxian Feng, Can Tang, Chao Wang, Chengxiao Tong, Fan Yang, Gang Xiong, Haixuan Gao, Han Gao, Hao Wang, Haochen Liu, Hongliang Sun, Jiabao Li, Jingwen Chang, Jun Du, Junyi Peng, Leizhen Cui, Meimei Jing, Mingqi Wu, Shangpeng Yan, Shaotong Qi, Suzhe Xu, Wenxuan Zhao, Xianda Sun, Xuan Xie, Yanbo Wang, Yao Xia, Yinghan Cui, Yingpeng Chen, Yong Wang, Yuze Shi, Zhiwei Shen, Ziyu Wang, Ming Sun, Lin Ye, Bin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.27703)  

**Abstract**: We present KAT-Coder-V2, an agentic coding model developed by the KwaiKAT team at Kuaishou. KAT-Coder-V2 adopts a "Specialize-then-Unify" paradigm that decomposes agentic coding into five expert domains - SWE, WebCoding, Terminal, WebSearch, and General - each undergoing independent supervised fine-tuning and reinforcement learning, before being consolidated into a single model via on-policy distillation. We develop KwaiEnv, a modular infrastructure sustaining tens of thousands of concurrent sandbox instances, and scale RL training along task complexity, intent alignment, and scaffold generalization. We further propose MCLA for stabilizing MoE RL training and Tree Training for eliminating redundant computation over tree-structured trajectories with up to 6.2x speedup. KAT-Coder-V2 achieves 79.6% on SWE-bench Verified (vs. Claude Opus 4.6 at 80.8%), 88.7 on PinchBench (surpassing GLM-5 and MiniMax M2.7), ranks first across all three frontend aesthetics scenarios, and maintains strong generalist scores on Terminal-Bench Hard (46.8) and tau^2-Bench (93.9). Our model is publicly available at this https URL. 

---
# Can Large Language Models Simulate Human Cognition Beyond Behavioral Imitation? 

**Authors**: Yuxuan Gu, Lunjun Liu, Xiaocheng Feng, Kun Zhu, Weihong Zhong, Lei Huang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.27694)  

**Abstract**: An essential problem in artificial intelligence is whether LLMs can simulate human cognition or merely imitate surface-level behaviors, while existing datasets suffer from either synthetic reasoning traces or population-level aggregation, failing to capture authentic individual cognitive patterns. We introduce a benchmark grounded in the longitudinal research trajectories of 217 researchers across diverse domains of artificial intelligence, where each author's scientific publications serve as an externalized representation of their cognitive processes. To distinguish whether LLMs transfer cognitive patterns or merely imitate behaviors, our benchmark deliberately employs a cross-domain, temporal-shift generalization setting. A multidimensional cognitive alignment metric is further proposed to assess individual-level cognitive consistency. Through systematic evaluation of state-of-the-art LLMs and various enhancement techniques, we provide a first-stage empirical study on the questions: (1) How well do current LLMs simulate human cognition? and (2) How far can existing techniques enhance these capabilities? 

---
# Investigating the Influence of Language on Sycophantic Behavior of Multilingual LLMs 

**Authors**: Bayan Abdullah Aldahlawi, A. B. M. Ashikur Rahman, Irfan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2603.27664)  

**Abstract**: Large language models (LLMs) have achieved strong performance across a wide range of tasks, but they are also prone to sycophancy, the tendency to agree with user statements regardless of validity. Previous research has outlined both the extent and the underlying causes of sycophancy in earlier models, such as ChatGPT-3.5 and Davinci. Newer models have since undergone multiple mitigation strategies, yet there remains a critical need to systematically test their behavior. In particular, the effect of language on sycophancy has not been explored.
In this work, we investigate how the language influences sycophantic responses. We evaluate three state-of-the-art models, GPT-4o mini, Gemini 1.5 Flash, and Claude 3.5 Haiku, using a set of tweet-like opinion prompts translated into five additional languages: Arabic, Chinese, French, Spanish, and Portuguese. Our results show that although newer models exhibit significantly less sycophancy overall compared to earlier generations, the extent of sycophancy is still influenced by the language. We further provide a granular analysis of how language shapes model agreeableness across sensitive topics, revealing systematic cultural and linguistic patterns. These findings highlight both the progress of mitigation efforts and the need for broader multilingual audits to ensure trustworthy and bias-aware deployment of LLMs. 

---
# The Degree of Language Diacriticity and Its Effect on Tasks 

**Authors**: Adi Cohen, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2603.27653)  

**Abstract**: Diacritics are orthographic marks that clarify pronunciation, distinguish similar words, or alter meaning. They play a central role in many writing systems, yet their impact on language technology has not been systematically quantified across scripts. While prior work has examined diacritics in individual languages, there's no cross-linguistic, data-driven framework for measuring the degree to which writing systems rely on them and how this affects downstream tasks. We propose a data-driven framework for quantifying diacritic complexity using corpus-level, information-theoretic metrics that capture the frequency, ambiguity, and structural diversity of character-diacritic combinations. We compute these metrics over 24 corpora in 15 languages, spanning both single- and multi-diacritic scripts. We then examine how diacritic complexity correlates with performance on the task of diacritics restoration, evaluating BERT- and RNN-based models. We find that across languages, higher diacritic complexity is strongly associated with lower restoration accuracy. In single-diacritic scripts, where character-diacritic combinations are more predictable, frequency-based and structural measures largely align. In multi-diacritic scripts, however, structural complexity exhibits the strongest association with performance, surpassing frequency-based measures. These findings show that measurable properties of diacritic usage influence the performance of diacritic restoration models, demonstrating that orthographic complexity is not only descriptive but functionally relevant for modeling. 

---
# Budget-Xfer: Budget-Constrained Source Language Selection for Cross-Lingual Transfer to African Languages 

**Authors**: Tewodros Kederalah Idris, Roald Eiselen, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2603.27651)  

**Abstract**: Cross-lingual transfer learning enables NLP for low-resource languages by leveraging labeled data from higher-resource sources, yet existing comparisons of source language selection strategies do not control for total training data, confounding language selection effects with data quantity effects. We introduce Budget-Xfer, a framework that formulates multi-source cross-lingual transfer as a budget-constrained resource allocation problem. Given a fixed annotation budget B, our framework jointly optimizes which source languages to include and how much data to allocate from each. We evaluate four allocation strategies across named entity recognition and sentiment analysis for three African target languages (Hausa, Yoruba, Swahili) using two multilingual models, conducting 288 experiments. Our results show that (1) multi-source transfer significantly outperforms single-source transfer (Cohen's d = 0.80 to 1.98), driven by a structural budget underutilization bottleneck; (2) among multi-source strategies, differences are modest and non-significant; and (3) the value of embedding similarity as a selection proxy is task-dependent, with random selection outperforming similarity-based selection for NER but not sentiment analysis. 

---
# PRBench: End-to-end Paper Reproduction in Physics Research 

**Authors**: Shi Qiu, Junyi Deng, Yiwei Deng, Haoran Dong, Jieyu Fu, Mao Li, Zeyu Li, Zhaolong Zhang, Huiwen Zheng, Leidong Bao, Anqi Lv, Zihan Mo, Yadi Niu, Yiyang Peng, Yu Tian, Yili Wang, Ziyu Wang, Zi-Yu Wang, Jiashen Wei, Liuheng Wu, Aoran Xue, Leyi Yang, Guanglu Yuan, Xiarui Zhan, Jingjun Zhang, Zifan Zheng, Pengfei Liu, Linrui Zhen, Kaiyang Li, Qichang Li, Ziheng Zhou, Guo-En Nian, Yunwei Xiao, Qing-Hong Cao, Linjie Dai, Xu Feng, Peng Gao, Ying Gu, Chang Liu, Jia Liu, Ming-xing Luo, Yan-Qing Ma, Liang-You Peng, Huichao Song, Shufeng Wang, Chenxu Wang, Tao Wang, Yi-Nan Wang, Chengyin Wu, Pengwei Zhao, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.27646)  

**Abstract**: AI agents powered by large language models exhibit strong reasoning and problem-solving capabilities, enabling them to assist scientific research tasks such as formula derivation and code generation. However, whether these agents can reliably perform end-to-end reproduction from real scientific papers remains an open question. We introduce PRBench, a benchmark of 30 expert-curated tasks spanning 11 subfields of physics. Each task requires an agent to comprehend the methodology of a published paper, implement the corresponding algorithms from scratch, and produce quantitative results matching the original publication. Agents are provided only with the task instruction and paper content, and operate in a sandboxed execution environment. All tasks are contributed by domain experts from over 20 research groups at the School of Physics, Peking University, each grounded in a real published paper and validated through end-to-end reproduction with verified ground-truth results and detailed scoring rubrics. Using an agentified assessment pipeline, we evaluate a set of coding agents on PRBench and analyze their capabilities across key dimensions of scientific reasoning and execution. The best-performing agent, OpenAI Codex powered by GPT-5.3-Codex, achieves a mean overall score of 34%. All agents exhibit a zero end-to-end callback success rate, with particularly poor performance in data accuracy and code correctness. We further identify systematic failure modes, including errors in formula implementation, inability to debug numerical simulations, and fabrication of output data. Overall, PRBench provides a rigorous benchmark for evaluating progress toward autonomous scientific research. 

---
# Umwelt Engineering: Designing the Cognitive Worlds of Linguistic Agents 

**Authors**: Rodney Jehu-Appiah  

**Link**: [PDF](https://arxiv.org/pdf/2603.27626)  

**Abstract**: I propose Umwelt engineering -- the deliberate design of the linguistic cognitive environment -- as a third layer in the agent design stack, upstream of both prompt and context engineering. Two experiments test the thesis that altering the medium of reasoning alters cognition itself. In Experiment 1, three language models reason under two vocabulary constraints -- No-Have (eliminating possessive "to have") and E-Prime (eliminating "to be") -- across seven tasks (N=4,470 trials). No-Have improves ethical reasoning by 19.1 pp (p < 0.001), classification by 6.5 pp (p < 0.001), and epistemic calibration by 7.4 pp, while achieving 92.8% constraint compliance. E-Prime shows dramatic but model-dependent effects: cross-model correlations reach r = -0.75. In Experiment 2, 16 linguistically constrained agents tackle 17 debugging problems. No constrained agent outperforms the control individually, yet a 3-agent ensemble achieves 100% ground-truth coverage versus 88.2% for the control. A permutation test confirms only 8% of random 3-agent subsets achieve full coverage, and every successful subset contains the counterfactual agent. Two mechanisms emerge: cognitive restructuring and cognitive diversification. The primary limitation is the absence of an active control matching constraint prompt elaborateness. 

---
# A gentle tutorial and a structured reformulation of Bock's algorithm for minimum directed spanning trees 

**Authors**: Yuxi Wang, Jungyeul Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.27530)  

**Abstract**: This paper presents a gentle tutorial and a structured reformulation of Bock's 1971 Algol procedure for constructing minimum directed spanning trees. Our aim is to make the original algorithm readable and reproducible for modern readers, while highlighting its relevance as an exact decoder for nonprojective graph based dependency parsing. We restate the minimum arborescence objective in Bock's notation and provide a complete line by line execution trace of the original ten node example, extending the partial trace given in the source paper from initialization to termination. We then introduce a structured reformulation that makes explicit the procedure's phase structure, maintained state, and control flow, while preserving the logic of the original method. As a further illustration, we include a worked example adapted from {jurafsky-martin-2026-book} for dependency parsing, showing how a maximum weight arborescence problem is reduced to Bock's minimum cost formulation by a standard affine transformation and traced under the same state variables. 

---
# Hidden Ads: Behavior Triggered Semantic Backdoors for Advertisement Injection in Vision Language Models 

**Authors**: Duanyi Yao, Changyue Li, Zhicong Huang, Cheng Hong, Songze Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.27522)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in consumer applications where users seek recommendations about products, dining, and services. We introduce Hidden Ads, a new class of backdoor attacks that exploit this recommendation-seeking behavior to inject unauthorized advertisements. Unlike traditional pattern-triggered backdoors that rely on artificial triggers such as pixel patches or special tokens, Hidden Ads activates on natural user behaviors: when users upload images containing semantic content of interest (e.g., food, cars, animals) and ask recommendation-seeking questions, the backdoored model provides correct, helpful answers while seamlessly appending attacker-specified promotional slogans. This design preserves model utility and produces natural-sounding injections, making the attack practical for real-world deployment in consumer-facing recommendation services.
We propose a multi-tier threat framework to systematically evaluate Hidden Ads across three adversary capability levels: hard prompt injection, soft prompt optimization, and supervised fine-tuning. Our poisoned data generation pipeline uses teacher VLM-generated chain-of-thought reasoning to create natural trigger--slogan associations across multiple semantic domains. Experiments on three VLM architectures demonstrate that Hidden Ads achieves high injection efficacy with near-zero false positives while maintaining task accuracy. Ablation studies confirm that the attack is data-efficient, transfers effectively to unseen datasets, and scales to multiple concurrent domain-slogan pairs. We evaluate defenses including instruction-based filtering and clean fine-tuning, finding that both fail to remove the backdoor without causing significant utility degradation. 

---
# Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs 

**Authors**: Utsav Maskey, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2603.27518)  

**Abstract**: Aligned language models that are trained to refuse harmful requests also exhibit over-refusal: they decline safe instructions that seemingly resemble harmful instructions. A natural approach is to ablate the global refusal direction, steering the hidden-state vectors away or towards the harmful-refusal examples, but this corrects over-refusal only incidentally while disrupting the broader refusal mechanism. In this work, we analyse the representational geometry of both refusal types to understand why this happens. We show that harmful-refusal directions are task-agnostic and can be captured by a single global vector, whereas over-refusal directions are task-dependent: they reside within the benign task-representation clusters, vary across tasks, and span a higher-dimensional subspace. Linear probing confirms that the two refusal types are representationally distinct from the early transformer layers. These findings provide a mechanistic explanation of why global direction ablation alone cannot address over-refusal, and establish that task-specific geometric interventions are necessary. 

---
# AgentSwing: Adaptive Parallel Context Management Routing for Long-Horizon Web Agents 

**Authors**: Zhaopeng Feng, Liangcai Su, Zhen Zhang, Xinyu Wang, Xiaotian Zhang, Xiaobin Wang, Runnan Fang, Qi Zhang, Baixuan Li, Shihao Cai, Rui Ye, Hui Chen, Jiang Yong, Joey Tianyi Zhou, Chenxiong Qian, Pengjun Xie, Bryan Hooi, Zuozhu Liu, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.27490)  

**Abstract**: As large language models (LLMs) evolve into autonomous agents for long-horizon information-seeking, managing finite context capacity has become a critical bottleneck. Existing context management methods typically commit to a single fixed strategy throughout the entire trajectory. Such static designs may work well in some states, but they cannot adapt as the usefulness and reliability of the accumulated context evolve during long-horizon search. To formalize this challenge, we introduce a probabilistic framework that characterizes long-horizon success through two complementary dimensions: search efficiency and terminal precision. Building on this perspective, we propose AgentSwing, a state-aware adaptive parallel context management routing framework. At each trigger point, AgentSwing expands multiple context-managed branches in parallel and uses lookahead routing to select the most promising continuation. Experiments across diverse benchmarks and agent backbones show that AgentSwing consistently outperforms strong static context management methods, often matching or exceeding their performance with up to $3\times$ fewer interaction turns while also improving the ultimate performance ceiling of long-horizon web agents. Beyond the empirical gains, the proposed probabilistic framework provides a principled lens for analyzing and designing future context management strategies for long-horizon agents. 

---
# A tree interpretation of arc standard dependency derivation 

**Authors**: Zihao Huang, Ai Ka Lee, Jungyeul Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.27459)  

**Abstract**: We show that arc-standard derivations for projective dependency trees determine a unique ordered tree representation with surface-contiguous yields and stable lexical anchoring. Each \textsc{shift}, \textsc{leftarc}, and \textsc{rightarc} transition corresponds to a deterministic tree update, and the resulting hierarchical object uniquely determines the original dependency arcs. We further show that this representation characterizes projectivity: a single-headed dependency tree admits such a contiguous ordered representation if and only if it is projective. The proposal is derivational rather than convertive. It interprets arc-standard transition sequences directly as ordered tree construction, rather than transforming a completed dependency graph into a phrase-structure output. For non-projective inputs, the same interpretation can be used in practice via pseudo-projective lifting before derivation and inverse decoding after recovery. A proof-of-concept implementation in a standard neural transition-based parser shows that the mapped derivations are executable and support stable dependency recovery. 

---
# Multi-Agent Dialectical Refinement for Enhanced Argument Classification 

**Authors**: Jakub Bąba, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2603.27451)  

**Abstract**: Argument Mining (AM) is a foundational technology for automated writing evaluation, yet traditional supervised approaches rely heavily on expensive, domain-specific fine-tuning. While Large Language Models (LLMs) offer a training-free alternative, they often struggle with structural ambiguity, failing to distinguish between similar components like Claims and Premises. Furthermore, single-agent self-correction mechanisms often suffer from sycophancy, where the model reinforces its own initial errors rather than critically evaluating them. We introduce MAD-ACC (Multi-Agent Debate for Argument Component Classification), a framework that leverages dialectical refinement to resolve classification uncertainty. MAD-ACC utilizes a Proponent-Opponent-Judge model where agents defend conflicting interpretations of ambiguous text, exposing logical nuances that single-agent models miss. Evaluation on the UKP Student Essays corpus demonstrates that MAD-ACC achieves a Macro F1 score of 85.7%, significantly outperforming single-agent reasoning baselines, without requiring domain-specific training. Additionally, unlike "black-box" classifiers, MAD-ACC's dialectical approach offers a transparent and explainable alternative by generating human-readable debate transcripts that explain the reasoning behind decisions. 

---
# Improving Attributed Long-form Question Answering with Intent Awareness 

**Authors**: Xinran Zhao, Aakanksha Naik, Jay DeYoung, Joseph Chee Chang, Jena D. Hwang, Tongshuang Wu, Varsha Kishore  

**Link**: [PDF](https://arxiv.org/pdf/2603.27435)  

**Abstract**: Large language models (LLMs) are increasingly being used to generate comprehensive, knowledge-intensive reports. However, while these models are trained on diverse academic papers and reports, they are not exposed to the reasoning processes and intents that guide authors in crafting these documents. We hypothesize that enhancing a model's intent awareness can significantly improve the quality of generated long-form reports. We develop and employ structured, tag-based schemes to better elicit underlying implicit intents to write or cite. We demonstrate that these extracted intents enhance both zero-shot generation capabilities in LLMs and enable the creation of high-quality synthetic data for fine-tuning smaller models. Our experiments reveal improved performance across various challenging scientific report generation tasks, with an average improvement of +2.9 and +12.3 absolute points for large and small models over baselines, respectively. Furthermore, our analysis illuminates how intent awareness enhances model citation usage and substantially improves report readability. 

---
# Not Worth Mentioning? A Pilot Study on Salient Proposition Annotation 

**Authors**: Amir Zeldes, Katherine Conhaim, Lauren Levine  

**Link**: [PDF](https://arxiv.org/pdf/2603.27358)  

**Abstract**: Despite a long tradition of work on extractive summarization, which by nature aims to recover the most important propositions in a text, little work has been done on operationalizing graded proposition salience in naturally occurring data. In this paper, we adopt graded summarization-based salience as a metric from previous work on Salient Entity Extraction (SEE) and adapt it to quantify proposition salience. We define the annotation task, apply it to a small multi-genre dataset, evaluate agreement and carry out a preliminary study of the relationship between our metric and notions of discourse unit centrality in discourse parsing following Rhetorical Structure Theory (RST). 

---
# Culturally Adaptive Explainable LLM Assessment for Multilingual Information Disorder: A Human-in-the-Loop Approach 

**Authors**: Maziar Kianimoghadam Jouneghani  

**Link**: [PDF](https://arxiv.org/pdf/2603.27356)  

**Abstract**: Recognizing information disorder is difficult because judgments about manipulation depend on cultural and linguistic context. Yet current Large Language Models (LLMs) often behave as monocultural, English-centric "black boxes," producing fluent rationales that overlook localized framing. Preliminary evidence from the multilingual Information Disorder (InDor) corpus suggests that existing models struggle to explain manipulated news consistently across communities. To address this gap, this ongoing study proposes a Hybrid Intelligence Loop, a human-in-the-loop (HITL) framework that grounds model assessment in human-written rationales from native-speaking annotators. The approach moves beyond static target-language few-shot prompting by pairing English task instructions with dynamically retrieved target-language exemplars drawn from filtered InDor annotations through In-Context Learning (ICL). In the initial pilot, the Exemplar Bank is seeded from these filtered annotations and used to compare static and adaptive prompting on Farsi and Italian news. The study evaluates span and severity prediction, the quality and cultural appropriateness of generated rationales, and model alignment across evaluator groups, providing a testbed for culturally grounded explainable AI. 

---
# PubMed Reasoner: Dynamic Reasoning-based Retrieval for Evidence-Grounded Biomedical Question Answering 

**Authors**: Yiqing Zhang, Xiaozhong Liu, Fabricio Murai  

**Link**: [PDF](https://arxiv.org/pdf/2603.27335)  

**Abstract**: Trustworthy biomedical question answering (QA) systems must not only provide accurate answers but also justify them with current, verifiable evidence. Retrieval-augmented approaches partially address this gap but lack mechanisms to iteratively refine poor queries, whereas self-reflection methods kick in only after full retrieval is completed. In this context, we introduce PubMed Reasoner, a biomedical QA agent composed of three stages: self-critic query refinement evaluates MeSH terms for coverage, alignment, and redundancy to enhance PubMed queries based on partial (metadata) retrieval; reflective retrieval processes articles in batches until sufficient evidence is gathered; and evidence-grounded response generation produces answers with explicit citations. PubMed Reasoner with a GPT-4o backbone achieves 78.32% accuracy on PubMedQA, slightly surpassing human experts, and showing consistent gains on MMLU Clinical Knowledge. Moreover, LLM-as-judge evaluations prefer our responses across: reasoning soundness, evidence grounding, clinical relevance, and trustworthiness. By orchestrating retrieval-first reasoning over authoritative sources, our approach provides practical assistance to clinicians and biomedical researchers while controlling compute and token costs. 

---
# SACRED: A Faithful Annotated Multimedia Multimodal Multilingual Dataset for Classifying Connectedness Types in Online Spirituality 

**Authors**: Qinghao Guan, Yuchen Pan, Donghao Li, Zishi Zhang, Yiyang Chen, Lu Li, Flaminia Canu, Emilia Volkart, Gerold Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2603.27331)  

**Abstract**: In religion and theology studies, spirituality has garnered significant research attention for the reason that it not only transcends culture but offers unique experience to each individual. However, social scientists often rely on limited datasets, which are basically unavailable online. In this study, we collaborated with social scientists to develop a high-quality multimedia multi-modal datasets, \textbf{SACRED}, in which the faithfulness of classification is guaranteed. Using \textbf{SACRED}, we evaluated the performance of 13 popular LLMs as well as traditional rule-based and fine-tuned approaches. The result suggests DeepSeek-V3 model performs well in classifying such abstract concepts (i.e., 79.19\% accuracy in the Quora test set), and the GPT-4o-mini model surpassed the other models in the vision tasks (63.99\% F1 score). Purportedly, this is the first annotated multi-modal dataset from online spirituality communication. Our study also found a new type of connectedness which is valuable for communication science studies. 

---
# Mitigating Hallucination on Hallucination in RAG via Ensemble Voting 

**Authors**: Zequn Xie, Zhengyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.27253)  

**Abstract**: Retrieval-Augmented Generation (RAG) aims to reduce hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, RAG introduces a critical challenge: hallucination on hallucination," where flawed retrieval results mislead the generation model, leading to compounded hallucinations. To address this issue, we propose VOTE-RAG, a novel, training-free framework with a two-stage structure and efficient, parallelizable voting mechanisms. VOTE-RAG includes: (1) Retrieval Voting, where multiple agents generate diverse queries in parallel and aggregate all retrieved documents; (2) Response Voting, where multiple agents independently generate answers based on the aggregated documents, with the final output determined by majority vote. We conduct comparative experiments on six benchmark datasets. Our results show that VOTE-RAG achieves performance comparable to or surpassing more complex frameworks. Additionally, VOTE-RAG features a simpler architecture, is fully parallelizable, and avoids the problem drift" risk. Our work demonstrates that simple, reliable ensemble voting is a superior and more efficient method for mitigating RAG hallucinations. 

---
# SCOPE: Tree-based Self-Correcting Online Log Parsing via Syntactic-Semantic Collaboration 

**Authors**: Dongyi Fan, Suqiong Zhang, Lili He, Ming Liu, Yifan Huo  

**Link**: [PDF](https://arxiv.org/pdf/2603.27247)  

**Abstract**: Log parsing is a critical step for automated log analysis in complex systems. Traditional heuristic-based methods offer high efficiency but are limited in accuracy due to overlooking semantic context. In contrast, recent LLM-based parsers improve accuracy via se mantic understanding but incur high latency from frequent model calls. To address this, we propose SCOPE, the first self-correcting online log parsing method that integrates the strengths of both heuristic and LLM-based paradigms. SCOPE introduces a novel bi-directional tree structure that enables efficient template match ing from both forward and reverse directions, resulting in a higher overall matching rate. Additionally, it adopts a two-stage syntactic semantic collaboration framework: a lightweight NLP model first utilizes part-of-speech (POS) information for syntax-based match ing, while the LLM is selectively invoked as a fallback to handle semantically complex cases when uncertainty remains. This design significantly reduces LLM API usage while maintaining high ac curacy, achieving a balance between efficiency and effectiveness. Extensive evaluations on diverse benchmark datasets show that SCOPE outperforms state-of-the-art methods in both accuracy and efficiency. The implementation and datasets are publicly released to facilitate further research. 

---
# Structural Stress and Learned Helplessness in Afghanistan: A Multi-Layer Analysis of the AFSTRESS Dari Corpus 

**Authors**: Jawid Ahmad Baktash, Mursal Dawodi, Nadira Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2603.27233)  

**Abstract**: We introduce AFSTRESS, the first multi-label corpus of self-reported stress narratives in Dari (Eastern Persian), comprising 737 responses collected from Afghan individuals during an ongoing humanitarian crisis. Participants describe experienced stress and select emotion and stressor labels via Dari checklists.
The dataset enables analysis at three levels: computational (multi-label classification), social (structural drivers and gender disparities), and psychological (learned helplessness, chronic stress, and emotional cascade patterns). It includes 12 binary labels (5 emotions, 7 stressors), with high label cardinality (5.54) and density (0.462), reflecting complex, multi-dimensional stress.
Structural stressors dominate: uncertain future (62.6 percent) and education closure (60.0 percent) exceed emotional states, indicating stress is primarily structurally driven. The strongest co-occurrence is between hopelessness and uncertain future (J = 0.388).
Baseline experiments show that character TF-IDF with Linear SVM achieves Micro-F1 = 0.663 and Macro-F1 = 0.651, outperforming ParsBERT and XLM-RoBERTa, while threshold tuning improves Micro-F1 by 10.3 points.
AFSTRESS provides the first Dari resource for computational analysis of stress and well-being in a crisis-affected population. 

---
# Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning 

**Authors**: Maximilian Mordig, Andreas Opedal, Weiyang Liu, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2603.27226)  

**Abstract**: Curriculum learning (CL), motivated by the intuition that learning in increasing order of difficulty should ease generalization, is commonly adopted both in pre-training and post-training of large language models (LLMs). The intuition of CL is particularly compelling for compositional reasoning, where complex problems are built from elementary inference rules; however, the actual impact of CL on such tasks remains largely underexplored. We present a systematic empirical study of CL for post-training of LLMs, using synthetic arithmetic and logical benchmarks where difficulty is characterized by reasoning complexity rather than surface-level proxies. Surprisingly, across multiple model families and curriculum schedules, we find no robust advantage in difficulty-based sequencing over standard random sampling in either accuracy or response length. These findings persist across both supervised fine-tuning (SFT) and reinforcement learning (RL) methods. Our study suggests that, in the context of deductive reasoning, the specific ordering of training examples plays a negligible role in achieving compositional generalization, challenging the practical utility of curriculum-based post-training. 

---
# Learning to Predict Future-Aligned Research Proposals with Language Models 

**Authors**: Heng Wang, Pengcheng Jiang, Jiashuo Sun, Zhiyi Shi, Haofei Yu, Jiawei Han, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.27146)  

**Abstract**: Large language models (LLMs) are increasingly used to assist ideation in research, but evaluating the quality of LLM-generated research proposals remains difficult: novelty and soundness are hard to measure automatically, and large-scale human evaluation is costly. We propose a verifiable alternative by reframing proposal generation as a time-sliced scientific forecasting problem. Given a research question and inspiring papers available before a cutoff time, the model generates a structured proposal and is evaluated by whether it anticipates research directions that appear in papers published after the time. We operationalize this objective with the Future Alignment Score (FAS), computed via retrieval and LLM-based semantic scoring against a held-out future corpus. To train models, we build a time-consistent dataset of 17,771 papers from targets and their pre-cutoff citations, and synthesize reasoning traces that teach gap identification and inspiration borrowing. Across Llama-3.1 and Qwen2.5 models, future-aligned tuning improves future alignment over unaligned baselines (up to +10.6% overall FAS), and domain-expert human evaluation corroborates improved proposal quality. Finally, we demonstrate practical impact by implementing two model-generated proposals with a code agent, obtaining 4.17% accuracy gain on MATH from a new prompting strategy and consistent improvements for a novel model-merging method. 

---
# Routing Sensitivity Without Controllability: A Diagnostic Study of Fairness in MoE Language Models 

**Authors**: Junhyeok Lee, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.27141)  

**Abstract**: Mixture-of-Experts (MoE) language models are universally sensitive to demographic content at the routing level, yet exploiting this sensitivity for fairness control is structurally limited. We introduce Fairness-Aware Routing Equilibrium (FARE), a diagnostic framework designed to probe the limits of routing-level stereotype intervention across diverse MoE architectures. FARE reveals that routing-level preference shifts are either unachievable (Mixtral, Qwen1.5, Qwen3), statistically non-robust (DeepSeekMoE), or accompanied by substantial utility cost (OLMoE, -4.4%p CrowS-Pairs at -6.3%p TQA). Critically, even where log-likelihood preference shifts are robust, they do not transfer to decoded generation: expanded evaluations on both non-null models yield null results across all generation metrics. Group-level expert masking reveals why: bias and core knowledge are deeply entangled within expert groups. These findings indicate that routing sensitivity is necessary but insufficient for stereotype control, and identify specific architectural conditions that can inform the design of more controllable future MoE systems. 

---
# Story2Proposal: A Scaffold for Structured Scientific Paper Writing 

**Authors**: Zhuoyang Qian, Wei Shi, Xu Lin, Li Ling, Meng Luo, Ziming Wang, Zhiwei Zhang, Tengyue Xu, Gaoge Liu, Zhentao Zhang, Shuo Zhang, Ziqi Wang, Zheng Feng, Yan Luo, Shu Xu, Yongjin Chen, Zhibo Feng, Zhuo Chen, Bruce Yuan, Biao Wu, Harry Wang, Kris Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.27065)  

**Abstract**: Generating scientific manuscripts requires maintaining alignment between narrative reasoning, experimental evidence, and visual artifacts across the document lifecycle. Existing language-model generation pipelines rely on unconstrained text synthesis with validation applied only after generation, often producing structural drift, missing figures or tables, and cross-section inconsistencies. We introduce Story2Proposal, a contract-governed multi-agent framework that converts a research story into a structured manuscript through coordinated agents operating under a persistent shared visual contract. The system organizes architect, writer, refiner, and renderer agents around a contract state that tracks section structure and registered visual elements, while evaluation agents supply feedback in a generate evaluate adapt loop that updates the contract during generation. Experiments on tasks derived from the Jericho research corpus show that Story2Proposal achieved an expert evaluation score of 6.145 versus 3.963 for DirectChat (+2.182) across GPT, Claude, Gemini, and Qwen backbones. Compared with the structured generation baseline Fars, Story2Proposal obtained an average score of 5.705 versus 5.197, indicating improved structural consistency and visual alignment. 

---
# Debiasing Large Language Models toward Social Factors in Online Behavior Analytics through Prompt Knowledge Tuning 

**Authors**: Hossein Salemi, Jitin Krishnan, Hemant Purohit  

**Link**: [PDF](https://arxiv.org/pdf/2603.27057)  

**Abstract**: Attribution theory explains how individuals interpret and attribute others' behavior in a social context by employing personal (dispositional) and impersonal (situational) causality. Large Language Models (LLMs), trained on human-generated corpora, may implicitly mimic this social attribution process in social contexts. However, the extent to which LLMs utilize these causal attributions in their reasoning remains underexplored. Although using reasoning paradigms, such as Chain-of-Thought (CoT), has shown promising results in various tasks, ignoring social attribution in reasoning could lead to biased responses by LLMs in social contexts. In this study, we investigate the impact of incorporating a user's goal as knowledge to infer dispositional causality and message context to infer situational causality on LLM performance. To this end, we introduce a scalable method to mitigate such biases by enriching the instruction prompts for LLMs with two prompt aids using social-attribution knowledge, based on the context and goal of a social media message. This method improves the model performance while reducing the social-attribution bias of the LLM in the reasoning on zero-shot classification tasks for behavior analytics applications. We empirically show the benefits of our method across two tasks-intent detection and theme detection on social media in the disaster domain-when considering the variability of disaster types and multiple languages of social media. Our experiments highlight the biases of three open-source LLMs: Llama3, Mistral, and Gemma, toward social attribution, and show the effectiveness of our mitigation strategies. 

---
# Text Data Integration 

**Authors**: Md Ataur Rahman, Dimitris Sacharidis, Oscar Romero, Sergi Nadal  

**Link**: [PDF](https://arxiv.org/pdf/2603.27055)  

**Abstract**: Data comes in many forms. From a shallow perspective, they can be viewed as being either in structured (e.g., as a relation, as key-value pairs) or unstructured (e.g., text, image) formats. So far, machines have been fairly good at processing and reasoning over structured data that follows a precise schema. However, the heterogeneity of data poses a significant challenge on how well diverse categories of data can be meaningfully stored and processed. Data Integration, a crucial part of the data engineering pipeline, addresses this by combining disparate data sources and providing unified data access to end-users. Until now, most data integration systems have leaned on only combining structured data sources. Nevertheless, unstructured data (a.k.a. free text) also contains a plethora of knowledge waiting to be utilized. Thus, in this chapter, we firstly make the case for the integration of textual data, to later present its challenges, state of the art and open problems. 

---
# Introducing MELI: the Mandarin-English Language Interview Corpus 

**Authors**: Suyuan Liu, Molly Babel  

**Link**: [PDF](https://arxiv.org/pdf/2603.27043)  

**Abstract**: We introduce the Mandarin-English Language Interview (MELI) Corpus, an open-source resource of 29.8 hours of speech from 51 Mandarin-English bilingual speakers. MELI combines matched sessions in Mandarin and English with two speaking styles: read sentences and spontaneous interviews about language varieties, standardness, and learning experiences. Audio was recorded at 44.1 kHz (16-bit, stereo). Interviews were fully transcribed, force-aligned at word and phone levels, and anonymized. Descriptively, the Mandarin component totals ~14.7 hours (mean duration 17.3 minutes) and the English component ~15.1 hours (mean duration 17.8 minutes). We report token/type statistics for each language and document code-switching patterns (frequent in Mandarin sessions; more limited in English sessions). The corpus design supports within-/cross-speaker, within/cross-language acoustic comparison and links acoustics to speakers' stated language attitudes, enabling both quantitative and qualitative analyses. The MELI Corpus will be released with transcriptions, alignments, metadata, scans of labelled maps and documentation under a CC BY-NC 4.0 license. 

---
# TAPS: Task Aware Proposal Distributions for Speculative Sampling 

**Authors**: Mohamad Zbib, Mohamad Bazzi, Ammar Mohanna, Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2603.27027)  

**Abstract**: Speculative decoding accelerates autoregressive generation by letting a lightweight draft model propose future tokens that a larger target model then verifies in parallel. In practice, however, draft models are usually trained on broad generic corpora, which leaves it unclear how much speculative decoding quality depends on the draft training distribution. We study this question with lightweight HASS and EAGLE-2 drafters trained on MathInstruct, ShareGPT, and mixed-data variants, evaluated on MT-Bench, GSM8K, MATH-500, and SVAMP. Measured by acceptance length, task-specific training yields clear specialization: MathInstruct-trained drafts are strongest on reasoning benchmarks, while ShareGPT-trained drafts are strongest on MT-Bench. Mixed-data training improves robustness, but larger mixtures do not dominate across decoding temperatures. We also study how to combine specialized drafters at inference time. Naive checkpoint averaging performs poorly, whereas confidence-based routing improves over single-domain drafts and merged-tree verification yields the highest acceptance length overall for both backbones. Finally, confidence is a more useful routing signal than entropy: rejected tokens tend to have higher entropy, but confidence produces much clearer benchmark-level routing decisions. These results show that speculative decoding quality depends not only on draft architecture, but also on the match between draft training data and downstream workload, and that specialized drafters are better combined at inference time than in weight space. 

---
# Pashto Common Voice: Building the First Open Speech Corpus for a 60-Million-Speaker Low-Resource Language 

**Authors**: Hanif Rahman, Shafeeq ur Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2603.27021)  

**Abstract**: We present the Pashto Common Voice corpus -- the first large-scale, openly licensed speech resource for Pashto, a language with over 60 million native speakers largely absent from open speech technology. Through a community effort spanning 2022-2025, the corpus grew from 1.5 hours and 5 contributors to 147 total hours and 1,483 unique speakers across ten Mozilla Common Voice releases (CV14-CV23). Speaker participation increased approximately 108-fold between CV17 and CV18, coinciding with a VOA Pashto broadcast campaign. We describe the full methodology: interface localisation, Wikipedia-based sentence extraction with automated filtering, phonemically targeted contributions for the four most frequently dropped Pashto characters, and multi-channel community outreach. MCV23 contains 107,781 clips (60,337 validated; 82.33 validated hours) across 13 content domains. Fine-tuning Whisper Base on the MCV20 yields 13.4% WER on the MCV20 test split, against the published Whisper Base zero-shot WER of 99.0% on Pashto. 

---
# RASPRef: Retrieval-Augmented Self-Supervised Prompt Refinement for Large Reasoning Models 

**Authors**: Rahul Soni  

**Link**: [PDF](https://arxiv.org/pdf/2603.27008)  

**Abstract**: Recent reasoning-focused language models such as DeepSeek R1 and OpenAI o1 have demonstrated strong performance on structured reasoning benchmarks including GSM8K, MATH, and multi-hop question answering tasks. However, their performance remains highly sensitive to prompt formulation, and designing effective prompts is typically a manual and iterative process that does not scale well across tasks or domains. To address this limitation, we introduce Retrieval-Augmented Self-Supervised Prompt Refinement (RASPRef), a framework that improves prompts without requiring human annotations or task-specific supervision. The approach retrieves relevant examples and previously generated reasoning trajectories, and leverages signals such as multi-sample consistency, verifier feedback, and model-generated critiques to iteratively refine the prompt. Unlike prior approaches that focus primarily on improving model outputs, RASPRef directly treats the prompt as the optimization target and improves it through an iterative retrieval-guided refinement process. Experiments on GSM8K-style mathematical reasoning tasks show that retrieval-guided prompting improves performance compared with a static prompting baseline. We further discuss how retrieval quality, trajectory selection, and self-supervised feedback signals may influence the effectiveness of prompt refinement. These findings suggest that prompt design remains a critical factor for reasoning-oriented language models, and that self-improving prompts offer a practical and scalable strategy for improving reasoning performance. 

---
# The Last Fingerprint: How Markdown Training Shapes LLM Prose 

**Authors**: E. M. Freeburg  

**Link**: [PDF](https://arxiv.org/pdf/2603.27006)  

**Abstract**: Large language models produce em dashes at varying rates, and the observation that some models "overuse" them has become one of the most widely discussed markers of AI-generated text. Yet no mechanistic account of this pattern exists, and the parallel observation that LLMs default to markdown-formatted output has never been connected to it. We propose that the em dash is markdown leaking into prose -- the smallest surviving unit of the structural orientation that LLMs acquire from markdown-saturated training corpora. We present a five-step genealogy connecting training data composition, structural internalization, the dual-register status of the em dash, and post-training amplification. We test this with a two-condition suppression experiment across twelve models from five providers (Anthropic, OpenAI, Meta, Google, DeepSeek): when models are instructed to avoid markdown formatting, overt features (headers, bullets, bold) are eliminated or nearly eliminated, but em dashes persist -- except in Meta's Llama models, which produce none at all. Em dash frequency and suppression resistance vary from 0.0 per 1,000 words (Llama) to 9.1 (GPT-4.1 under suppression), functioning as a signature of the specific fine-tuning procedure applied. A three-condition suppression gradient shows that even explicit em dash prohibition fails to eliminate the artifact in some models, and a base-vs-instruct comparison confirms that the latent tendency exists pre-RLHF. These findings connect two previously isolated online discourses and reframe em dash frequency as a diagnostic of fine-tuning methodology rather than a stylistic defect. 

---
# A large corpus of lucid and non-lucid dream reports 

**Authors**: Remington Mallett  

**Link**: [PDF](https://arxiv.org/pdf/2603.26992)  

**Abstract**: All varieties of dreaming remain a mystery. Lucid dreams in particular, or those characterized by awareness of the dream, are notoriously difficult to study. Their scarce prevalence and resistance to deliberate induction make it difficult to obtain a sizeable corpus of lucid dream reports. The consequent lack of clarity around lucid dream phenomenology has left the many purported applications of lucidity under-realized. Here, a large corpus of 55k dream reports from 5k contributors is curated, described, and validated for future research. Ten years of publicly available dream reports were scraped from an online forum where users share anonymous dream journals. Importantly, users optionally categorize their dream as lucid, non-lucid, or a nightmare, offering a user-provided labeling system that includes 10k lucid and 25k non-lucid, and 2k nightmare labels. After characterizing the corpus with descriptive statistics and visualizations, construct validation shows that language patterns in lucid-labeled reports are consistent with known characteristics of lucid dreams. While the entire corpus has broad value for dream science, the labeled subset is particularly powerful for new discoveries in lucid dream studies. 

---
# Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation 

**Authors**: Lorca McLaren, James Cross, Zuzanna Krakowska, Robin Rauner, Martijn Schoonvelde  

**Link**: [PDF](https://arxiv.org/pdf/2603.26898)  

**Abstract**: Political scientists are rapidly adopting large language models (LLMs) for text annotation, yet the sensitivity of annotation results to implementation choices remains poorly understood. Most evaluations test a single model or configuration; how model choice, model size, learning approach, and prompt style interact, and whether popular "best practices" survive controlled comparison, are largely unexplored. We present a controlled evaluation of these pipeline choices, testing six open-weight models across four political science annotation tasks under identical quantisation, hardware, and prompt-template conditions. Our central finding is methodological: interaction effects dominate main effects, so seemingly reasonable pipeline choices can become consequential researcher degrees of freedom. No single model, prompt style, or learning approach is uniformly superior, and the best-performing model varies across tasks. Two corollaries follow. First, model size is an unreliable guide both to cost and to performance: cross-family efficiency differences are so large that some larger models are less resource-intensive than much smaller alternatives, while within model families mid-range variants often match or exceed larger counterparts. Second, widely recommended prompt engineering techniques yield inconsistent and sometimes negative effects on annotation performance. We use these benchmark results to develop a validation-first framework - with a principled ordering of pipeline decisions, guidance on prompt freezing and held-out evaluation, reporting standards, and open-source tools - to help researchers navigate this decision space transparently. 

---
# Arithmetic OOD Failure Unfolds in Stages in Minimal GPTs 

**Authors**: Seine A. Shintani  

**Link**: [PDF](https://arxiv.org/pdf/2603.26828)  

**Abstract**: Arithmetic benchmarks are often reduced to a single held-out score, but that score can conflate qualitatively different failures. We study a controlled minimal GPT trained on exhaustive 2-digit addition, where all local digit transitions are already present in training, and ask why 3-digit generalization still fails. The failure is staged. First, there is a layout barrier: a learned absolute-position model collapses under a pure 3-digit layout shift, and mixed-layout exposure is the only intervention that materially weakens this barrier. Second, after layout repair, the hundreds position behaves like a carry flag rather than a semantic hundreds digit; targeted carry probes reverse the relevant logit margin, whereas a matched extra-data control does not. Third, after carry repair, the main remaining bottleneck is conditional recomposition: high-conditioned tail data outperforms a matched control, high-only data, and tail-only data on all true-3-digit suites, and the same ordering reappears in a larger 2-layer bridge experiment. The residual errors after recomposition are then overwhelmingly tens-only, and a separate 10-seed late-stage study shows that a sign-aware tens repair raises exact match on the hardest thousands-carry suite from 0.664 to 0.822. We therefore provide an experimentally testable decomposition of arithmetic OOD failure into layout, carry-semantics, recomposition, and late tens-residual stages. 

---
# Resolving the Robustness-Precision Trade-off in Financial RAG through Hybrid Document-Routed Retrieval 

**Authors**: Zhiyuan Cheng, Longying Lai, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26815)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems for financial document question answering typically follow a chunk-based paradigm: documents are split into fragments, embedded into vector space, and retrieved via similarity search. While effective in general settings, this approach suffers from cross-document chunk confusion in structurally homogeneous corpora such as regulatory filings. Semantic File Routing (SFR), which uses LLM structured output to route queries to whole documents, reduces catastrophic failures but sacrifices the precision of targeted chunk retrieval. We identify this robustness-precision trade-off through controlled evaluation on the FinDER benchmark (1,500 queries across five groups): SFR achieves higher average scores (6.45 vs. 6.02) and fewer failures (10.3% vs. 22.5%), while chunk-based retrieval (CBR) yields more perfect answers (13.8% vs. 8.5%). To resolve this trade-off, we propose Hybrid Document-Routed Retrieval (HDRR), a two-stage architecture that uses SFR as a document filter followed by chunk-based retrieval scoped to the identified document(s). HDRR eliminates cross-document confusion while preserving targeted chunk precision. Experimental results demonstrate that HDRR achieves the best performance on every metric: an average score of 7.54 (25.2% above CBR, 16.9% above SFR), a failure rate of only 6.4%, a correctness rate of 67.7% (+18.7 pp over CBR), and a perfect-answer rate of 20.1% (+6.3 pp over CBR, +11.6 pp over SFR). HDRR resolves the trade-off by simultaneously achieving the lowest failure rate and the highest precision across all five experimental groups. 

---
# LogicDiff: Logic-Guided Denoising Improves Reasoning in Masked Diffusion Language Models 

**Authors**: Shaik Aman  

**Link**: [PDF](https://arxiv.org/pdf/2603.26771)  

**Abstract**: Masked diffusion language models (MDLMs) generate text by iteratively unmasking tokens from a fully masked sequence, offering parallel generation and bidirectional context. However, their standard confidence-based unmasking strategy systematically defers high-entropy logical connective tokens, the critical branching points in reasoning chains, leading to severely degraded reasoning performance. We introduce LogicDiff, an inference-time method that replaces confidence-based unmasking with logic-role-guided unmasking. A lightweight classification head (4.2M parameters, 0.05% of the base model) predicts the logical role of each masked position (premise, connective, derived step, conclusion, or filler) from the base model's hidden states with 98.4% accuracy. A dependency-ordered scheduler then unmasks tokens in logical dependency order: premises first, then connectives, then derived steps, then conclusions. Without modifying a single parameter of the base model and without any reinforcement learning or task-specific training, LogicDiff improves LLaDA-8B-Instruct accuracy from 22.0% to 60.7% on GSM8K (+38.7 percentage points) and from 23.6% to 29.2% on MATH-500 (+5.6 pp), with less than 6% speed overhead. Our results demonstrate that a substantial portion of the reasoning deficit in MDLMs is attributable to suboptimal token unmasking order, not to limitations of the model's learned representations. 

---
# Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages 

**Authors**: Swastik R  

**Link**: [PDF](https://arxiv.org/pdf/2603.26742)  

**Abstract**: Vision-language models score well on mathematical, scientific, and spatial reasoning benchmarks, yet these evaluations are overwhelmingly English. I present the first cross-lingual visual reasoning audit for Indian languages. 980 questions from MathVista, ScienceQA, and MMMU are translated into Hindi, Tamil, Telugu, Bengali, Kannada, and Marathi using IndicTrans2, with Gemini 2.0 Flash cross-verification on 50 samples per language (inter-translator agreement 0.79-0.84). Eight VLMs, from 7B open-source models to GPT-4o, are evaluated across all seven languages, yielding 68,600 inference records that include text-only and chain-of-thought ablations. I find accuracy drops of 9.8-25 percentage points when switching from English to an Indian language, with Dravidian languages suffering up to 13.2 pp more than Indo-Aryan. Chain-of-thought prompting degrades Bengali (-14.4 pp) and Kannada (-11.4 pp) rather than helping, exposing English-centric reasoning chains. Aya-Vision-8B, built for 23 languages, still drops 28.5 pp on Dravidian scripts; multilingual pretraining alone does not transfer visual reasoning. I release the translated benchmark and all model outputs. 

---
# The Cognitive Divergence: AI Context Windows, Human Attention Decline, and the Delegation Feedback Loop 

**Authors**: Netanel Eliav  

**Link**: [PDF](https://arxiv.org/pdf/2603.26707)  

**Abstract**: This paper documents and theorises a self-reinforcing dynamic between two measurable trends: the exponential expansion of large language model (LLM) context windows and the secular contraction of human sustained-attention capacity. We term the resulting asymmetry the Cognitive Divergence. AI context windows have grown from 512 tokens in 2017 to 2,000,000 tokens by 2026 (factor ~3,906; fitted lambda = 0.59/yr; doubling time ~14 months). Over the same period, human Effective Context Span (ECS) -- a token-equivalent measure derived from validated reading-rate meta-analysis (Brysbaert, 2019) and an empirically motivated Comprehension Scaling Factor -- has declined from approximately 16,000 tokens (2004 baseline) to an estimated 1,800 tokens (2026, extrapolated from longitudinal behavioural data ending 2020 (Mark, 2023); see Section 9 for uncertainty discussion). The AI-to-human ratio grew from near parity at the ChatGPT launch (November 2022) to 556--1,111x raw and 56--111x quality-adjusted, after accounting for retrieval degradation (Liu et al., 2024; Chroma, 2025). Beyond documenting this divergence, the paper introduces the Delegation Feedback Loop hypothesis: as AI capability grows, the cognitive threshold at which humans delegate to AI falls, extending to tasks of negligible demand; the resulting reduction in cognitive practice may further attenuate the capacities already documented as declining (Gerlich, 2025; Kim et al., 2026; Kosmyna et al., 2025). Neither trend reverses spontaneously. The paper characterises the divergence statistically, reviews neurobiological mechanisms across eight peer-reviewed neuroimaging studies, presents empirical evidence bearing on the delegation threshold, and proposes a research agenda centred on a validated ECS psychometric instrument and longitudinal study of AI-mediated cognitive change. 

---
# AlpsBench: An LLM Personalization Benchmark for Real-Dialogue Memorization and Preference Alignment 

**Authors**: Jianfei Xiao, Xiang Yu, Chengbing Wang, Wuqiang Zheng, Xinyu Lin, Kaining Liu, Hongxun Ding, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2603.26680)  

**Abstract**: As Large Language Models (LLMs) evolve into lifelong AI assistants, LLM personalization has become a critical frontier. However, progress is currently bottlenecked by the absence of a gold-standard evaluation benchmark. Existing benchmarks either overlook personalized information management that is critical for personalization or rely heavily on synthetic dialogues, which exhibit an inherent distribution gap from real-world dialogue. To bridge this gap, we introduce AlpsBench, An LLM PerSonalization benchmark derived from real-world human-LLM dialogues. AlpsBench comprises 2,500 long-term interaction sequences curated from WildChat, paired with human-verified structured memories that encapsulate both explicit and implicit personalization signals. We define four pivotal tasks - personalized information extraction, updating, retrieval, and utilization - and establish protocols to evaluate the entire lifecycle of memory management. Our benchmarking of frontier LLMs and memory-centric systems reveals that: (i) models struggle to reliably extract latent user traits; (ii) memory updating faces a performance ceiling even in the strongest models; (iii) retrieval accuracy declines sharply in the presence of large distractor pools; and (iv) while explicit memory mechanisms improve recall, they do not inherently guarantee more preference-aligned or emotionally resonant responses. AlpsBench aims to provide a comprehensive framework. 

---
# GeoBlock: Inferring Block Granularity from Dependency Geometry in Diffusion Language Models 

**Authors**: Lipeng Wan, Junjie Ma, Jianhui Gu, Zeyang Liu, Xuyang Lu, Xuguang Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.26675)  

**Abstract**: Block diffusion enables efficient parallel refinement in diffusion language models, but its decoding behavior depends critically on block size. Existing block-sizing strategies rely on fixed rules or heuristic signals and do not account for the dependency geometry that determines which tokens can be safely refined together. This motivates a geometry view of diffusion decoding: \emph{regions with strong causal ordering require sequential updates, whereas semantically cohesive regions admit parallel refinement.} We introduce GeoBlock, a geometry-aware block inference framework that determines block granularity directly from attention-derived dependency geometry. Instead of relying on predefined schedules or local confidence heuristics, GeoBlock analyzes cross-token dependency patterns to identify geometrically stable refinement regions and dynamically determines appropriate block boundaries during decoding. By adapting block granularity to the dependency geometry, GeoBlock preserves the parallel efficiency of block diffusion while enforcing dependency-consistent refinement that exhibits autoregressive reliability. GeoBlock requires no additional training and integrates seamlessly into existing block diffusion architectures. Extensive experiments across multiple benchmarks show that GeoBlock reliably identifies geometry-consistent block boundaries and improves the accuracy of block diffusion with only a small additional computational budget. 

---
# ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining 

**Authors**: Anuj Diwan, Eunsol Choi, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2603.28737)  

**Abstract**: We introduce ParaSpeechCLAP, a dual-encoder contrastive model that maps speech and text style captions into a common embedding space, supporting a wide range of intrinsic (speaker-level) and situational (utterance-level) descriptors (such as pitch, texture and emotion) far beyond the narrow set handled by existing models. We train specialized ParaSpeechCLAP-Intrinsic and ParaSpeechCLAP-Situational models alongside a unified ParaSpeechCLAP-Combined model, finding that specialization yields stronger performance on individual style dimensions while the unified model excels on compositional evaluation. We further show that ParaSpeechCLAP-Intrinsic benefits from an additional classification loss and class-balanced training. We demonstrate our models' performance on style caption retrieval, speech attribute classification and as an inference-time reward model that improves style-prompted TTS without additional training. ParaSpeechCLAP outperforms baselines on most metrics across all three applications. Our models and code are released at this https URL . 

---
# SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning 

**Authors**: Philip Schroeder, Thomas Weng, Karl Schmeckpeper, Eric Rosen, Stephen Hart, Ondrej Biza  

**Link**: [PDF](https://arxiv.org/pdf/2603.28730)  

**Abstract**: Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. To address this limitation, we introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including GPT-5 and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking. 

---
# The Ultimate Tutorial for AI-driven Scale Development in Generative Psychometrics: Releasing AIGENIE from its Bottle 

**Authors**: Lara Russell-Lasalandra, Hudson Golino, Luis Eduardo Garrido, Alexander P. Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2603.28643)  

**Abstract**: Psychological scale development has traditionally required extensive expert involvement, iterative revision, and large-scale pilot testing before psychometric evaluation can begin. The `AIGENIE` R package implements the AI-GENIE framework (Automatic Item Generation with Network-Integrated Evaluation), which integrates large language model (LLM) text generation with network psychometric methods to automate the early stages of this process. The package generates candidate item pools using LLMs, transforms them into high-dimensional embeddings, and applies a multi-step reduction pipeline -- Exploratory Graph Analysis (EGA), Unique Variable Analysis (UVA), and bootstrap EGA -- to produce structurally validated item pools entirely *in silico*. This tutorial introduces the package across six parts: installation and setup, understanding Application Programming Interfaces (APIs), text generation, item generation, the `AIGENIE` function, and the `GENIE` function. Two running examples illustrate the package's use: the Big Five personality model (a well-established construct) and AI Anxiety (an emerging construct). The package supports multiple LLM providers (OpenAI, Anthropic, Groq, HuggingFace, and local models), offers a fully offline mode with no external API calls, and provides the `GENIE()` function for researchers who wish to apply the psychometric reduction pipeline to existing item pools regardless of their origin. The `AIGENIE` package is freely available on R-universe at this https URL. 

---
# ResAdapt: Adaptive Resolution for Efficient Multimodal Reasoning 

**Authors**: Huanxuan Liao, Zhongtao Jiang, Yupu Hao, Yuqiao Tan, Shizhu He, Jun Zhao, Kun Xu, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28610)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve stronger visual understanding by scaling input fidelity, yet the resulting visual token growth makes jointly sustaining high spatial resolution and long temporal context prohibitive. We argue that the bottleneck lies not in how post-encoding representations are compressed but in the volume of pixels the encoder receives, and address it with ResAdapt, an Input-side adaptation framework that learns how much visual budget each frame should receive before encoding. ResAdapt couples a lightweight Allocator with an unchanged MLLM backbone, so the backbone retains its native visual-token interface while receiving an operator-transformed input. We formulate allocation as a contextual bandit and train the Allocator with Cost-Aware Policy Optimization (CAPO), which converts sparse rollout feedback into a stable accuracy-cost learning signal. Across budget-controlled video QA, temporal grounding, and image reasoning tasks, ResAdapt improves low-budget operating points and often lies on or near the efficiency-accuracy frontier, with the clearest gains on reasoning-intensive benchmarks under aggressive compression. Notably, ResAdapt supports up to 16x more frames at the same visual budget while delivering over 15% performance gain. Code is available at this https URL. 

---
# Moving Beyond Review: Applying Language Models to Planning and Translation in Reflection 

**Authors**: Seyed Parsa Neshaei, Richard Lee Davis, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2603.28596)  

**Abstract**: Reflective writing is known to support the development of students' metacognitive skills, yet learners often struggle to engage in deep reflection, limiting learning gains. Although large language models (LLMs) have been shown to improve writing skills, their use as conversational agents for reflective writing has produced mixed results and has largely focused on providing feedback on reflective texts, rather than support during planning and organizing. In this paper, inspired by the Cognitive Process Theory of writing (CPT), we propose the first application of LLMs to the planning and translation steps of reflective writing. We introduce Pensée, a tool to explore the effects of explicit AI support during these stages by scaffolding structured reflection planning using a conversational agent, and supporting translation by automatically extracting key concepts. We evaluate Pensée in a controlled between-subjects experiment (N=93), manipulating AI support across writing phases. Results show significantly greater reflection depth and structural quality when learners receive support during planning and translation stages of CPT, though these effects reduce in a delayed post-test. Analyses of learner behavior and perceptions further illustrate how CPT-aligned conversational support shapes reflection processes and learner experience, contributing empirical evidence for theory-driven uses of LLMs in AI-supported reflective writing. 

---
# Entropic Claim Resolution: Uncertainty-Driven Evidence Selection for RAG 

**Authors**: Davide Di Gioia  

**Link**: [PDF](https://arxiv.org/pdf/2603.28444)  

**Abstract**: Current Retrieval-Augmented Generation (RAG) systems predominantly rely on relevance-based dense retrieval, sequentially fetching documents to maximize semantic similarity with the query. However, in knowledge-intensive and real-world scenarios characterized by conflicting evidence or fundamental query ambiguity, relevance alone is insufficient for resolving epistemic uncertainty. We introduce Entropic Claim Resolution (ECR), a novel inference-time algorithm that reframes RAG reasoning as entropy minimization over competing semantic answer hypotheses. Unlike action-driven agentic frameworks (e.g., ReAct) or fixed-pipeline RAG architectures, ECR sequentially selects atomic evidence claims by maximizing Expected Entropy Reduction (EER), a decision-theoretic criterion for the value of information. The process dynamically terminates when the system reaches a mathematically defined state of epistemic sufficiency (H <= epsilon, subject to epistemic coherence). We integrate ECR into a production-grade multi-strategy retrieval pipeline (CSGR++) and analyze its theoretical properties. Our framework provides a rigorous foundation for uncertainty-aware evidence selection, shifting the paradigm from retrieving what is most relevant to retrieving what is most discriminative. 

---
# IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression 

**Authors**: Zhongping Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.28430)  

**Abstract**: Orthogonal feature decorrelation is effective for low-bit online vector quantization, but dense random orthogonal transforms incur prohibitive $O(d^2)$ storage and compute. RotorQuant reduces this cost with blockwise $3$D Clifford rotors, yet the resulting $3$D partition is poorly aligned with modern hardware and offers limited local mixing.
We propose \textbf{IsoQuant}, a blockwise rotation framework based on quaternion algebra and the isoclinic decomposition of $SO(4)$. It represents each $4$D block as a quaternion and applies a closed-form transform $T(v)=q_L v \overline{q_R}$. This yields two main variants: \emph{IsoQuant-Full}, which realizes the full $SO(4)$ rotation, and \emph{IsoQuant-Fast}, which keeps only one isoclinic factor for lower cost; the framework also admits a lightweight $2$D special case. At $d=128$, IsoQuant-Full reduces forward rotation cost from about $2{,}408$ FMAs in RotorQuant to $1{,}024$, while IsoQuant-Fast further reduces it to $512$. Across $18$ fused CUDA settings with $d \in {128,256,512}$, bit widths ${2,3,4}$, and FP16/FP32 execution, IsoQuant achieves mean kernel-level speedups of about $4.5\times$--$4.7\times$ over RotorQuant while maintaining comparable reconstruction MSE, with peak speedups above $6\times$. Current validation is limited to the stage-1 quantize--dequantize path on synthetic normalized vectors; end-to-end KV-cache evaluation remains future work. 

---
# MiroEval: Benchmarking Multimodal Deep Research Agents in Process and Outcome 

**Authors**: Fangda Ye, Yuxin Hu, Pengxiang Zhu, Yibo Li, Ziqi Jin, Yao Xiao, Yibo Wang, Lei Wang, Zhen Zhang, Lu Wang, Yue Deng, Bin Wang, Yifan Zhang, Liangcai Su, Xinyu Wang, He Zhao, Chen Wei, Qiang Ren, Bryan Hooi, An Bo, Shuicheng Yan, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2603.28407)  

**Abstract**: Recent progress in deep research systems has been impressive, but evaluation still lags behind real user needs. Existing benchmarks predominantly assess final reports using fixed rubrics, failing to evaluate the underlying research process. Most also offer limited multimodal coverage, rely on synthetic tasks that do not reflect real-world query complexity, and cannot be refreshed as knowledge evolves. To address these gaps, we introduce MiroEval, a benchmark and evaluation framework for deep research systems. The benchmark comprises 100 tasks (70 text-only, 30 multimodal), all grounded in real user needs and constructed via a dual-path pipeline that supports periodic updates, enabling a live and evolving setting. The proposed evaluation suite assesses deep research systems along three complementary dimensions: adaptive synthesis quality evaluation with task-specific rubrics, agentic factuality verification via active retrieval and reasoning over both web sources and multimodal attachments, and process-centric evaluation audits how the system searches, reasons, and refines throughout its investigation. Evaluation across 13 systems yields three principal findings: the three evaluation dimensions capture complementary aspects of system capability, with each revealing distinct strengths and weaknesses across systems; process quality serves as a reliable predictor of overall outcome while revealing weaknesses invisible to output-level metrics; and multimodal tasks pose substantially greater challenges, with most systems declining by 3 to 10 points. The MiroThinker series achieves the most balanced performance, with MiroThinker-H1 ranking the highest overall in both settings. Human verification and robustness results confirm the reliability of the benchmark and evaluation framework. MiroEval provides a holistic diagnostic tool for the next generation of deep research agents. 

---
# Does Claude's Constitution Have a Culture? 

**Authors**: Parham Pourdavood  

**Link**: [PDF](https://arxiv.org/pdf/2603.28123)  

**Abstract**: Constitutional AI (CAI) aligns language models with explicitly stated normative principles, offering a transparent alternative to implicit alignment through human feedback alone. However, because constitutions are authored by specific groups of people, the resulting models may reflect particular cultural perspectives. We investigate this question by evaluating Anthropic's Claude Sonnet on 55 World Values Survey items, selected for high cross-cultural variance across six value domains and administered as both direct survey questions and naturalistic advice-seeking scenarios. Comparing Claude's responses to country-level data from 90 nations, we find that Claude's value profile most closely resembles those of Northern European and Anglophone countries, but on a majority of items extends beyond the range of all surveyed populations. When users provide cultural context, Claude adjusts its rhetorical framing but not its substantive value positions, with effect sizes indistinguishable from zero across all twelve tested countries. An ablation removing the system prompt increases refusals but does not alter the values expressed when responses are given, and replication on a smaller model (Claude Haiku) confirms the same cultural profile across model sizes. These findings suggest that when a constitution is authored within the same cultural tradition that dominates the training data, constitutional alignment may codify existing cultural biases rather than correct them--producing a value floor that surface-level interventions cannot meaningfully shift. We discuss the compounding nature of this risk and the need for globally representative constitution-authoring processes. 

---
# MOSS-VoiceGenerator: Create Realistic Voices with Natural Language Descriptions 

**Authors**: Kexin Huang, Liwei Fan, Botian Jiang, Yaozhou Jiang, Qian Tu, Jie Zhu, Yuqian Zhang, Yiwei Zhao, Chenchen Yang, Zhaoye Fei, Shimin Li, Xiaogui Yang, Qinyuan Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28086)  

**Abstract**: Voice design from natural language aims to generate speaker timbres directly from free-form textual descriptions, allowing users to create voices tailored to specific roles, personalities, and emotions. Such controllable voice creation benefits a wide range of downstream applications-including storytelling, game dubbing, role-play agents, and conversational assistants, making it a significant task for modern Text-to-Speech models. However, existing models are largely trained on carefully recorded studio data, which produces speech that is clean and well-articulated, yet lacks the lived-in qualities of real human voices. To address these limitations, we present MOSS-VoiceGenerator, an open-source instruction-driven voice generation model that creates new timbres directly from natural language prompts. Motivated by the hypothesis that exposure to real-world acoustic variation produces more perceptually natural voices, we train on large-scale expressive speech data sourced from cinematic content. Subjective preference studies demonstrate its superiority in overall performance, instruction-following, and naturalness compared to other voice design models. 

---
# CDH-Bench: A Commonsense-Driven Hallucination Benchmark for Evaluating Visual Fidelity in Vision-Language Models 

**Authors**: Kesheng Chen, Yamin Hu, Qi Zhou, Zhenqian Zhu, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.27982)  

**Abstract**: Vision-language models (VLMs) achieve strong performance on many benchmarks, yet a basic reliability question remains underexplored: when visual evidence conflicts with commonsense, do models follow what is shown or what commonsense suggests? A characteristic failure in this setting is that the model overrides visual evidence and outputs the commonsense alternative. We term this phenomenon \textbf{commonsense-driven hallucination} (CDH). To evaluate it, we introduce \textbf{CDH-Bench}, a benchmark designed to create explicit \textbf{visual evidence--commonsense conflicts}. CDH-Bench covers three dimensions: \textit{counting anomalies}, \textit{relational anomalies}, and \textit{attribute anomalies}. We evaluate frontier VLMs under \textit{binary Question Answering (QA)} and \textit{multiple-choice QA}, and report metrics including \textit{Counterfactual Accuracy} (CF-Acc), \textit{Commonsense Accuracy} (CS-Acc), \textit{Counterfactual Accuracy Drop} (CFAD), \textit{Commonsense Collapse Rate} (CCR), and \textit{Relative Prior Dependency} (RPD). Results show that even strong models remain vulnerable to prior-driven normalization under visual evidence--commonsense conflict. CDH-Bench provides a controlled diagnostic of visual fidelity under visual evidence--commonsense conflict. 

---
# Efficient Inference of Large Vision Language Models 

**Authors**: Surendra Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2603.27960)  

**Abstract**: Although Large Vision Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, their scalability and deployment are constrained by massive computational requirements. In particular, the massive amount of visual tokens from high-resolution input data aggravates the situation due to the quadratic complexity of attention mechanisms. To address these issues, the research community has developed several optimization frameworks. This paper presents a comprehensive survey of the current state-of-the-art techniques for accelerating LVLM inference. We introduce a systematic taxonomy that categorizes existing optimization frameworks into four primary dimensions: visual token compression, memory management and serving, efficient architectural design, and advanced decoding strategies. Furthermore, we critically examine the limitations of these current methodologies and identify critical open problems to inspire future research directions in efficient multimodal systems. 

---
# EffiSkill: Agent Skill Based Automated Code Efficiency Optimization 

**Authors**: Zimu Wang, Yuling Shi, Mengfan Li, Zijun Liu, Jie M. Zhang, Chengcheng Wan, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.27850)  

**Abstract**: Code efficiency is a fundamental aspect of software quality, yet how to harness large language models (LLMs) to optimize programs remains challenging. Prior approaches have sought for one-shot rewriting, retrieved exemplars, or prompt-based search, but they do not explicitly distill reusable optimization knowledge, which limits generalization beyond individual instances.
In this paper, we present EffiSkill, a framework for code-efficiency optimization that builds a portable optimization toolbox for LLM-based agents. The key idea is to model recurring slow-to-fast transformations as reusable agent skills that capture both concrete transformation mechanisms and higher-level optimization strategies. EffiSkill adopts a two-stage design: Stage I mines Operator and Meta Skills from large-scale slow/fast program pairs to build a skill library; Stage II applies this library to unseen programs through execution-free diagnosis, skill retrieval, plan composition, and candidate generation, without runtime feedback.
Results on EffiBench-X show that EffiSkill achieves higher optimization success rates, improving over the strongest baseline by 3.69 to 12.52 percentage points across model and language settings. These findings suggest that mechanism-level skill reuse provides a useful foundation for execution-free code optimization, and that the resulting skill library can serve as a reusable resource for broader agent workflows. 

---
# Q-Bridge: Code Translation for Quantum Machine Learning via LLMs 

**Authors**: Runjia Zeng, Priyabrata Senapati, Ruixiang Tang, Dongfang Liu, Qiang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2603.27836)  

**Abstract**: Large language models have recently shown potential in bridging the gap between classical machine learning and quantum machine learning. However, the lack of standardized, high-quality datasets and robust translation frameworks limits progress in this domain. We introduce Q-Bridge, an LLM-guided code translation framework that systematically converts CML implementations into executable QML variants. Our approach builds on a self-involving pipeline that iteratively expands a verified seed codebase into a large-scale dataset, CML-2-QML, integrating verifiable and unverifiable code pairs. The Q-Bridge model is fine-tuned using supervised LoRA adaptation for scalable and memory-efficient training, achieving faithful and interpretable quantum code generation across diverse architectures. Empirical analysis confirms the feasibility of direct CML-to-QML translation and reveals consistent structural alignment between classical and quantum paradigms. Case studies further demonstrate that Q-Bridge can maintain deterministic correctness and also enable creative architectural exploration. This work establishes the first reproducible framework and dataset for LLM-driven quantum code translation, offering a foundation for scalable quantum AI development. 

---
# KVSculpt: KV Cache Compression as Distillation 

**Authors**: Bo Jiang, Sian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.27819)  

**Abstract**: KV cache compression is critical for efficient long-context LLM inference. Approaches that reduce the per-pair footprint -- quantization and low-rank decomposition -- are orthogonal to those that reduce the sequence length of the cache. Along the sequence-length dimension, existing methods range from pure eviction -- selecting which KV pairs to keep -- to merging, which combines similar pairs into fewer ones. Both remain anchored to the original cache entries. We propose KVSculpt, which moves to the other end of this spectrum: instead of selecting or combining original pairs, we optimize a smaller set of unconstrained KV pairs in continuous embedding space to preserve each layer's attention behavior. Keys are optimized via L-BFGS and values are solved in closed form via least squares, alternating every few steps. On top of this, we introduce adaptive budget allocation, which uses a cheap pilot compression run to redistribute the compression budget across layers and KV heads based on per-component difficulty. On Qwen2.5-1.5B-Instruct with 2048-token contexts, KVSculpt reduces KL divergence by 3.5-4.1x compared to Select+Fit -- attention-score eviction with least-squares value fitting -- across compression ratios r in {0.3, 0.5, 0.7}. Adaptive allocation provides an additional 1.3x KL reduction at no extra inference cost. Analysis reveals that compression difficulty is highly non-uniform: per-layer pilot MSE varies by up to 100x across layers, and the two KV heads within a single layer can differ by up to 467x -- demonstrating that fine-grained budget allocation is essential. 

---
# Emergent Social Intelligence Risks in Generative Multi-Agent Systems 

**Authors**: Yue Huang, Yu Jiang, Wenjie Wang, Haomin Zhuang, Xiaonan Luo, Yuchen Ma, Zhangchen Xu, Zichen Chen, Nuno Moniz, Zinan Lin, Pin-Yu Chen, Nitesh V Chawla, Nouha Dziri, Huan Sun, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.27771)  

**Abstract**: Multi-agent systems composed of large generative models are rapidly moving from laboratory prototypes to real-world deployments, where they jointly plan, negotiate, and allocate shared resources to solve complex tasks. While such systems promise unprecedented scalability and autonomy, their collective interaction also gives rise to failure modes that cannot be reduced to individual agents. Understanding these emergent risks is therefore critical. Here, we present a pioneer study of such emergent multi-agent risk in workflows that involve competition over shared resources (e.g., computing resources or market share), sequential handoff collaboration (where downstream agents see only predecessor outputs), collective decision aggregation, and others. Across these settings, we observe that such group behaviors arise frequently across repeated trials and a wide range of interaction conditions, rather than as rare or pathological cases. In particular, phenomena such as collusion-like coordination and conformity emerge with non-trivial frequency under realistic resource constraints, communication protocols, and role assignments, mirroring well-known pathologies in human societies despite no explicit instruction. Moreover, these risks cannot be prevented by existing agent-level safeguards alone. These findings expose the dark side of intelligent multi-agent systems: a social intelligence risk where agent collectives, despite no instruction to do so, spontaneously reproduce familiar failure patterns from human societies. 

---
# LongCat-Next: Lexicalizing Modalities as Discrete Tokens 

**Authors**: Meituan LongCat Team, Bin Xiao, Chao Wang, Chengjiang Li, Chi Zhang, Chong Peng, Hang Yu, Hao Yang, Haonan Yan, Haoze Sun, Haozhe Zhao, Hong Liu, Hui Su, Jiaqi Zhang, Jiawei Wang, Jing Li, Kefeng Zhang, Manyuan Zhang, Minhao Jing, Peng Pei, Quan Chen, Taofeng Xue, Tongxin Pan, Xiaotong Li, Xiaoyang Li, Xiaoyu Zhao, Xing Hu, Xinyang Lin, Xunliang Cai, Yan Bai, Yan Feng, Yanjie Li, Yao Qiu, Yerui Sun, Yifan Lu, Ying Luo, Yipeng Mei, Yitian Chen, Yuchen Xie, Yufang Liu, Yufei Chen, Yulei Qian, Yuqi Peng, Zhihang Yu, Zhixiong Han, Changran Wang, Chen Chen, Dian Zheng, Fengjiao Chen, Ge Yang, Haowei Guo, Haozhe Wang, Hongyu Li, Huicheng Jiang, Jiale Hong, Jialv Zou, Jiamu Li, Jianping Lin, Jiaxing Liu, Jie Yang, Jing Jin, Jun Kuang, Juncheng She, Kunming Luo, Kuofeng Gao, Lin Qiu, Linsen Guo, Mianqiu Huang, Qi Li, Qian Wang, Rumei Li, Siyu Ren, Wei Wang, Wenlong He, Xi Chen, Xiao Liu, Xiaoyu Li, Xu Huang, Xuanyu Zhu, Xuezhi Cao, Yaoming Zhu, Yifei Cao, Yimeng Jia, Yizhen Jiang, Yufei Gao, Zeyang Hu, Zhenlong Yuan, Zijian Zhang, Ziwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.27538)  

**Abstract**: The prevailing Next-Token Prediction (NTP) paradigm has driven the success of large language models through discrete autoregressive modeling. However, contemporary multimodal systems remain language-centric, often treating non-linguistic modalities as external attachments, leading to fragmented architectures and suboptimal integration. To transcend this limitation, we introduce Discrete Native Autoregressive (DiNA), a unified framework that represents multimodal information within a shared discrete space, enabling a consistent and principled autoregressive modeling across modalities. A key innovation is the Discrete Native Any-resolution Visual Transformer (dNaViT), which performs tokenization and de-tokenization at arbitrary resolutions, transforming continuous visual signals into hierarchical discrete tokens. Building on this foundation, we develop LongCat-Next, a native multimodal model that processes text, vision, and audio under a single autoregressive objective with minimal modality-specific design. As an industrial-strength foundation model, it excels at seeing, painting, and talking within a single framework, achieving strong performance across a wide range of multimodal benchmarks. In particular, LongCat-Next addresses the long-standing performance ceiling of discrete vision modeling on understanding tasks and provides a unified approach to effectively reconcile the conflict between understanding and generation. As an attempt toward native multimodality, we open-source the LongCat-Next and its tokenizers, hoping to foster further research and development in the community. GitHub: this https URL 

---
# The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams 

**Authors**: Isaac Llorente-Saguer  

**Link**: [PDF](https://arxiv.org/pdf/2603.27412)  

**Abstract**: We present LatentBiopsy, a training-free method for detecting harmful prompts by analysing the geometry of residual-stream activations in large language models. Given 200 safe normative prompts, LatentBiopsy computes the leading principal component of their activations at a target layer and characterises new prompts by their radial deviation angle $\theta$ from this reference direction. The anomaly score is the negative log-likelihood of $\theta$ under a Gaussian fit to the normative distribution, flagging deviations symmetrically regardless of orientation. No harmful examples are required for training.
We evaluate two complete model triplets from the Qwen3.5-0.8B and Qwen2.5-0.5B families: base, instruction-tuned, and \emph{abliterated} (refusal direction surgically removed via orthogonalisation). Across all six variants, LatentBiopsy achieves AUROC $\geq$0.937 for harmful-vs-normative detection and AUROC = 1.000 for discriminating harmful from benign-aggressive prompts (XSTest), with sub-millisecond per-query overhead.
Three empirical findings emerge. First, geometry survives refusal ablation: both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts, establishing a geometric dissociation between harmful-intent representation and the downstream generative refusal mechanism. Second, harmful prompts exhibit a near-degenerate angular distribution ($\sigma_\theta \approx 0.03$ rad), an order of magnitude tighter than the normative distribution ($\sigma_\theta \approx 0.27$ rad), preserved across all alignment stages including abliteration. Third, the two families exhibit opposite ring orientations at the same depth: harmful prompts occupy the outer ring in Qwen3.5-0.8B but the inner ring in Qwen2.5-0.5B, directly motivating the direction-agnostic scoring rule. 

---
# Heterogeneous Debate Engine: Identity-Grounded Cognitive Architecture for Resilient LLM-Based Ethical Tutoring 

**Authors**: Jakub Masłowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2603.27404)  

**Abstract**: Large Language Models (LLMs) are being increasingly used as autonomous agents in complex reasoning tasks, opening the niche for dialectical interactions. However, Multi-Agent systems implemented with systematically unconstrained systems systematically undergo semantic drift and logical deterioration and thus can hardly be used in providing ethical tutoring where a precise answer is required. Current simulation often tends to degenerate into dialectical stagnation, the agents degenerate into recursive concurrence or circular arguments. A critical challenge remains: how to enforce doctrinal fidelity without suppressing the generative flexibility required for dialectical reasoning? To address this niche, we contribute the Heterogeneous Debate Engine (HDE), a cognitive architecture that combines Identity-Grounded Retrieval-Augmented Generation (ID-RAG) for doctrinal fidelity and Heuristic Theory of Mind for strategic opponent modeling. Our evaluation shows that architectural heterogeneity is a crucial variable to stability: contrary doctrinal initializations (e.g., Deontology vs. Utilitarianism) have increased the Argument Complexity Scores of students by an order of magnitude, over baselines. These findings validate the effectiveness of ID-RAG and Heuristic ToM as architectural requirements in maintaining high-fidelity (adversarial) pedagogy. 

---
# LLM Readiness Harness: Evaluation, Observability, and CI Gates for LLM/RAG Applications 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2603.27355)  

**Abstract**: We present a readiness harness for LLM and RAG applications that turns evaluation into a deployment decision workflow. The system combines automated benchmarks, OpenTelemetry observability, and CI quality gates under a minimal API contract, then aggregates workflow success, policy compliance, groundedness, retrieval hit rate, cost, and p95 latency into scenario-weighted readiness scores with Pareto frontiers. We evaluate the harness on ticket-routing workflows and BEIR grounding tasks (SciFact and FiQA) with full Azure matrix coverage (162/162 valid cells across datasets, scenarios, retrieval depths, seeds, and models). Results show that readiness is not a single metric: on FiQA under sla-first at k=5, gpt-4.1-mini leads in readiness and faithfulness, while gpt-5.2 pays a substantial latency cost; on SciFact, models are closer in quality but still separable operationally. Ticket-routing regression gates consistently reject unsafe prompt variants, demonstrating that the harness can block risky releases instead of merely reporting offline scores. The result is a reproducible, operationally grounded framework for deciding whether an LLM or RAG system is ready to ship. 

---
# Inference-Time Structural Reasoning for Compositional Vision-Language Understanding 

**Authors**: Amartya Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2603.27349)  

**Abstract**: Vision-language models (VLMs) excel at image-text retrieval yet persistently fail at compositional reasoning, distinguishing captions that share the same words but differ in relational structure. We present, a unified evaluation and augmentation framework benchmarking four architecturally diverse VLMs,CLIP, BLIP, LLaVA, and Qwen3-VL-8B-Thinking,on the Winoground benchmark under plain and scene-graph-augmented regimes. We introduce a dependency-based TextSceneGraphParser (spaCy) extracting subject-relation-object triples, and a Graph Asymmetry Scorer using optimal bipartite matching to inject structural relational priors. Caption ablation experiments (subject-object masking and swapping) reveal that Qwen3-VL-8B-Thinking achieves a group score of 62.75, far above all encoder-based models, while a proposed multi-turn SG filtering strategy further lifts it to 66.0, surpassing prior open-source state-of-the-art. We analyze the capability augmentation tradeoff and find that SG augmentation benefits already capable models while providing negligible or negative gains for weaker baselines. Code: this https URL 

---
# Self-evolving AI agents for protein discovery and directed evolution 

**Authors**: Yang Tan, Lingrong Zhang, Mingchen Li, Yuanxi Yu, Bozitao Zhong, Bingxin Zhou, Nanqing Dong, Liang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2603.27303)  

**Abstract**: Protein scientific discovery is bottlenecked by the manual orchestration of information and algorithms, while general agents are insufficient in complex domain projects. VenusFactory2 provides an autonomous framework that shifts from static tool usage to dynamic workflow synthesis via a self-evolving multi-agent infrastructure to address protein-related demands. It outperforms a set of well-known agents on the VenusAgentEval benchmark, and autonomously organizes the discovery and optimization of proteins from a single natural language prompt. 

---
# LightMover: Generative Light Movement with Color and Intensity Controls 

**Authors**: Gengze Zhou, Tianyu Wang, Soo Ye Kim, Zhixin Shu, Xin Yu, Yannick Hold-Geoffroy, Sumit Chaturvedi, Qi Wu, Zhe Lin, Scott Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2603.27209)  

**Abstract**: We present LightMover, a framework for controllable light manipulation in single images that leverages video diffusion priors to produce physically plausible illumination changes without re-rendering the scene. We formulate light editing as a sequence-to-sequence prediction problem in visual token space: given an image and light-control tokens, the model adjusts light position, color, and intensity together with resulting reflections, shadows, and falloff from a single view. This unified treatment of spatial (movement) and appearance (color, intensity) controls improves both manipulation and illumination understanding. We further introduce an adaptive token-pruning mechanism that preserves spatially informative tokens while compactly encoding non-spatial attributes, reducing control sequence length by 41% while maintaining editing fidelity. To train our framework, we construct a scalable rendering pipeline that generates large numbers of image pairs across varied light positions, colors, and intensities while keeping the scene content consistent with the original image. LightMover enables precise, independent control over light position, color, and intensity, and achieves high PSNR and strong semantic consistency (DINO, CLIP) across different tasks. 

---
# daVinci-LLM:Towards the Science of Pretraining 

**Authors**: Yiwei Qin, Yixiu Liu, Tiantian Mi, Muhang Xie, Zhen Huang, Weiye Si, Pengrui Lu, Siyuan Feng, Xia Wu, Liming Liu, Ye Luo, Jinlong Hou, Qipeng Guo, Yu Qiao, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.27164)  

**Abstract**: The foundational pretraining phase determines a model's capability ceiling, as post-training struggles to overcome capability foundations established during pretraining, yet it remains critically under-explored. This stems from a structural paradox: organizations with computational resources operate under commercial pressures that inhibit transparent disclosure, while academic institutions possess research freedom but lack pretraining-scale computational resources. daVinci-LLM occupies this unexplored intersection, combining industrial-scale resources with full research freedom to advance the science of pretraining. We adopt a fully-open paradigm that treats openness as scientific methodology, releasing complete data processing pipelines, full training processes, and systematic exploration results. Recognizing that the field lacks systematic methodology for data processing, we employ the Data Darwinism framework, a principled L0-L9 taxonomy from filtering to synthesis. We train a 3B-parameter model from random initialization across 8T tokens using a two-stage adaptive curriculum that progressively shifts from foundational capabilities to reasoning-intensive enhancement. Through 200+ controlled ablations, we establish that: processing depth systematically enhances capabilities, establishing it as a critical dimension alongside volume scaling; different domains exhibit distinct saturation dynamics, necessitating adaptive strategies from proportion adjustments to format shifts; compositional balance enables targeted intensification while preventing performance collapse; how evaluation protocol choices shape our understanding of pretraining progress. By releasing the complete exploration process, we enable the community to build upon our findings and systematic methodologies to form accumulative scientific knowledge in pretraining. 

---
# ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding 

**Authors**: Jovana Kondic, Pengyuan Li, Dhiraj Joshi, Isaac Sanchez, Ben Wiesel, Shafiq Abedin, Amit Alfassy, Eli Schwartz, Daniel Caraballo, Yagmur Gizem Cinar, Florian Scheidegger, Steven I. Ross, Daniel Karl I. Weidele, Hang Hua, Ekaterina Arutyunova, Roei Herzig, Zexue He, Zihan Wang, Xinyue Yu, Yunfei Zhao, Sicong Jiang, Minghao Liu, Qunshu Lin, Peter Staar, Luis Lastras, Aude Oliva, Rogerio Feris  

**Link**: [PDF](https://arxiv.org/pdf/2603.27064)  

**Abstract**: Understanding charts requires models to jointly reason over geometric visual patterns, structured numerical data, and natural language -- a capability where current vision-language models (VLMs) remain limited. We introduce ChartNet, a high-quality, million-scale multimodal dataset designed to advance chart interpretation and reasoning. ChartNet leverages a novel code-guided synthesis pipeline to generate 1.5 million diverse chart samples spanning 24 chart types and 6 plotting libraries. Each sample consists of five aligned components: plotting code, rendered chart image, data table, natural language summary, and question-answering with reasoning, providing fine-grained cross-modal alignment. To capture the full spectrum of chart comprehension, ChartNet additionally includes specialized subsets encompassing human annotated data, real-world data, safety, and grounding. Moreover, a rigorous quality-filtering pipeline ensures visual fidelity, semantic accuracy, and diversity across chart representations. Fine-tuning on ChartNet consistently improves results across benchmarks, demonstrating its utility as large-scale supervision for multimodal models. As the largest open-source dataset of its kind, ChartNet aims to support the development of foundation models with robust and generalizable capabilities for data visualization understanding. The dataset is publicly available at this https URL 

---
# PHONOS: PHOnetic Neutralization for Online Streaming Applications 

**Authors**: Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, Ricardo Gutierrez-Osuna  

**Link**: [PDF](https://arxiv.org/pdf/2603.27001)  

**Abstract**: Speaker anonymization (SA) systems modify timbre while leaving regional or non-native accents intact, which is problematic because accents can narrow the anonymity set. To address this issue, we present PHONOS, a streaming module for real-time SA that neutralizes non-native accent to sound native-like. Our approach pre-generates golden speaker utterances that preserve source timbre and rhythm but replace foreign segmentals with native ones using silence-aware DTW alignment and zero-shot voice conversion. These utterances supervise a causal accent translator that maps non-native content tokens to native equivalents with at most 40ms look-ahead, trained using joint cross-entropy and CTC losses. Our evaluations show an 81% reduction in non-native accent confidence, with listening-test ratings consistent with this shift, and reduced speaker linkability as accent-neutralized utterances move away from the original speaker in embedding space while having latency under 241 ms on single GPU. 

---
# FormalProofBench: Can Models Write Graduate Level Math Proofs That Are Formally Verified? 

**Authors**: Nikil Ravi, Kexing Ying, Vasilii Nesterov, Rayan Krishnan, Elif Uskuplu, Bingyu Xia, Janitha Aswedige, Langston Nashold  

**Link**: [PDF](https://arxiv.org/pdf/2603.26996)  

**Abstract**: We present FormalProofBench, a private benchmark designed to evaluate whether AI models can produce formally verified mathematical proofs at the graduate level. Each task pairs a natural-language problem with a Lean~4 formal statement, and a model must output a Lean proof accepted by the Lean 4 checker. FormalProofBench targets advanced undergraduate and graduate mathematics, with problems drawn from qualifying exams and standard textbooks across topics including analysis, algebra, probability, and logic. We evaluate a range of frontier models with an agentic harness, and find that the best-performing foundation model achieves 33.5% accuracy, with performance dropping rapidly after that. In addition to the accuracy numbers, we also provide empirical analysis of tool-use, failure modes, cost and latency, thereby providing a thorough evaluation of the formal-theorem proving abilities of frontier models. 

---
# Multilingual Stutter Event Detection for English, German, and Mandarin Speech 

**Authors**: Felix Haas, Sebastian P. Bayerl  

**Link**: [PDF](https://arxiv.org/pdf/2603.26939)  

**Abstract**: This paper presents a multi-label stuttering detection system trained on multi-corpus, multilingual data in English, German, and this http URL leveraging annotated stuttering data from three languages and four corpora, the model captures language-independent characteristics of stuttering, enabling robust detection across linguistic contexts. Experimental results demonstrate that multilingual training achieves performance comparable to and, in some cases, even exceeds that of previous systems. These findings suggest that stuttering exhibits cross-linguistic consistency, which supports the development of language-agnostic detection systems. Our work demonstrates the feasibility and advantages of using multilingual data to improve generalizability and reliability in automated stuttering detection. 

---
# In your own words: computationally identifying interpretable themes in free-text survey data 

**Authors**: Jenny S Wang, Aliya Saperstein, Emma Pierson  

**Link**: [PDF](https://arxiv.org/pdf/2603.26930)  

**Abstract**: Free-text survey responses can provide nuance often missed by structured questions, but remain difficult to statistically analyze. To address this, we introduce In Your Own Words, a computational framework for exploratory analyses of free-text survey data that identifies structured, interpretable themes in free-text responses more precisely than previous computational approaches, facilitating systematic analysis. To illustrate the benefits of this approach, we apply it to a new dataset of free-text descriptions of race, gender, and sexual orientation from 1,004 U.S. participants. The themes our approach learns have three practical applications in survey research. First, the themes can suggest structured questions to add to future surveys by surfacing salient constructs -- such as belonging and identity fluidity -- that existing surveys do not capture. Second, the themes reveal heterogeneity within standardized categories, explaining additional variation in health, well-being, and identity importance. Third, the themes illuminate systematic discordance between self-identified and perceived identities, highlighting mechanisms of misrecognition that existing measures do not reflect. More broadly, our framework can be deployed in a wide range of survey settings to identify interpretable themes from free text, complementing existing qualitative methods. 

---
# GroupRAG: Cognitively Inspired Group-Aware Retrieval and Reasoning via Knowledge-Driven Problem Structuring 

**Authors**: Xinyi Duan, Yuanrong Tang, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.26807)  

**Abstract**: The performance of language models is commonly limited by insufficient knowledge and constrained reasoning. Prior approaches such as Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) address these issues by incorporating external knowledge or enforcing linear reasoning chains, but often degrade in real-world settings. Inspired by cognitive science, which characterizes human problem solving as search over structured problem spaces rather than single inference chains, we argue that inadequate awareness of problem structure is a key overlooked limitation. We propose GroupRAG, a cognitively inspired, group-aware retrieval and reasoning framework based on knowledge-driven keypoint grouping. GroupRAG identifies latent structural groups within a problem and performs retrieval and reasoning from multiple conceptual starting points, enabling fine-grained interaction between the two processes. Experiments on MedQA show that GroupRAG outperforms representative RAG- and CoT-based baselines. These results suggest that explicitly modeling problem structure, as inspired by human cognition, is a promising direction for robust retrieval-augmented reasoning. 

---
# CRISP: Characterizing Relative Impact of Scholarly Publications 

**Authors**: Hannah Collison, Benjamin Van Durme, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2603.26791)  

**Abstract**: Assessing a cited paper's impact is typically done by analyzing its citation context in isolation within the citing paper. While this focuses on the most directly relevant text, it prevents relative comparisons across all the works a paper cites. We propose CRISP, which instead jointly ranks all cited papers within a citing paper using large language models (LLMs). To mitigate LLMs' positional bias, we rank each list three times in a randomized order and aggregate the impact labels through majority voting. This joint approach leverages the full citation context, rather than evaluating citations independently, to more reliably distinguish impactful references. CRISP outperforms a prior state-of-the-art impact classifier by +9.5% accuracy and +8.3% F1 on a dataset of human-annotated citations. CRISP further gains efficiency through fewer LLM calls and performs competitively with an open-source model, enabling scalable, cost-effective citation impact analysis. We release our rankings, impact labels, and codebase to support future research. 

---
# Learning to Select Visual In-Context Demonstrations 

**Authors**: Eugene Lee, Yu-Chi Lin, Jiajie Diao  

**Link**: [PDF](https://arxiv.org/pdf/2603.26775)  

**Abstract**: Multimodal Large Language Models (MLLMs) adapt to visual tasks via in-context learning (ICL), which relies heavily on demonstration quality. The dominant demonstration selection strategy is unsupervised k-Nearest Neighbor (kNN) search. While simple, this similarity-first approach is sub-optimal for complex factual regression tasks; it selects redundant examples that fail to capture the task's full output range. We reframe selection as a sequential decision-making problem and introduce Learning to Select Demonstrations (LSD), training a Reinforcement Learning agent to construct optimal demonstration sets. Using a Dueling DQN with a query-centric Transformer Decoder, our agent learns a policy that maximizes MLLM downstream performance. Evaluating across five visual regression benchmarks, we uncover a crucial dichotomy: while kNN remains optimal for subjective preference tasks, LSD significantly outperforms baselines on objective, factual regression tasks. By balancing visual relevance with diversity, LSD better defines regression boundaries, illuminating when learned selection is strictly necessary for visual ICL. 

---
# Aesthetic Assessment of Chinese Handwritings Based on Vision Language Models 

**Authors**: Chen Zheng, Yuxuan Lai, Haoyang Lu, Wentao Ma, Jitao Yang, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26768)  

**Abstract**: The handwriting of Chinese characters is a fundamental aspect of learning the Chinese language. Previous automated assessment methods often framed scoring as a regression problem. However, this score-only feedback lacks actionable guidance, which limits its effectiveness in helping learners improve their handwriting skills. In this paper, we leverage vision-language models (VLMs) to analyze the quality of handwritten Chinese characters and generate multi-level feedback. Specifically, we investigate two feedback generation tasks: simple grade feedback (Task 1) and enriched, descriptive feedback (Task 2). We explore both low-rank adaptation (LoRA)-based fine-tuning strategies and in-context learning methods to integrate aesthetic assessment knowledge into VLMs. Experimental results show that our approach achieves state-of-the-art performances across multiple evaluation tracks in the CCL 2025 workshop on evaluation of handwritten Chinese character quality. 

---
# SleepVLM: Explainable and Rule-Grounded Sleep Staging via a Vision-Language Model 

**Authors**: Guifeng Deng, Pan Wang, Jiquan Wang, Shuying Rao, Junyi Xie, Wanjun Guo, Tao Li, Haiteng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26738)  

**Abstract**: While automated sleep staging has achieved expert-level accuracy, its clinical adoption is hindered by a lack of auditable reasoning. We introduce SleepVLM, a rule-grounded vision-language model (VLM) designed to stage sleep from multi-channel polysomnography (PSG) waveform images while generating clinician-readable rationales based on American Academy of Sleep Medicine (AASM) scoring criteria. Utilizing waveform-perceptual pre-training and rule-grounded supervised fine-tuning, SleepVLM achieved Cohen's kappa scores of 0.767 on an held out test set (MASS-SS1) and 0.743 on an external cohort (ZUAMHCS), matching state-of-the-art performance. Expert evaluations further validated the quality of the model's reasoning, with mean scores exceeding 4.0/5.0 for factual accuracy, evidence comprehensiveness, and logical coherence. By coupling competitive performance with transparent, rule-based explanations, SleepVLM may improve the trustworthiness and auditability of automated sleep staging in clinical workflows. To facilitate further research in interpretable sleep medicine, we release MASS-EX, a novel expert-annotated dataset. 

---
# SEAR: Schema-Based Evaluation and Routing for LLM Gateways 

**Authors**: Zecheng Zhang, Han Zheng, Yue Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26728)  

**Abstract**: Evaluating production LLM responses and routing requests across providers in LLM gateways requires fine-grained quality signals and operationally grounded decisions. To address this gap, we present SEAR, a schema-based evaluation and routing system for multi-model, multi-provider LLM gateways. SEAR defines an extensible relational schema covering both LLM evaluation signals (context, intent, response characteristics, issue attribution, and quality scores) and gateway operational metrics (latency, cost, throughput), with cross-table consistency links across around one hundred typed, SQL-queryable columns. To populate the evaluation signals reliably, SEAR proposes self-contained signal instructions, in-schema reasoning, and multi-stage generation that produces database-ready structured outputs. Because signals are derived through LLM reasoning rather than shallow classifiers, SEAR captures complex request semantics, enables human-interpretable routing explanations, and unifies evaluation and routing in a single query layer. Across thousands of production sessions, SEAR achieves strong signal accuracy on human-labeled data and supports practical routing decisions, including large cost reductions with comparable quality. 

---
# Agentic AI for Human Resources: LLM-Driven Candidate Assessment 

**Authors**: Kamer Ali Yuksel, Abdul Basit Anees, Ashraf Elneima, Sanjika Hewavitharana, Mohamed Al-Badrashiny, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2603.26710)  

**Abstract**: In this work, we present a modular and interpretable framework that uses Large Language Models (LLMs) to automate candidate assessment in recruitment. The system integrates diverse sources, including job descriptions, CVs, interview transcripts, and HR feedback; to generate structured evaluation reports that mirror expert judgment. Unlike traditional ATS tools that rely on keyword matching or shallow scoring, our approach employs role-specific, LLM-generated rubrics and a multi-agent architecture to perform fine-grained, criteria-driven evaluations. The framework outputs detailed assessment reports, candidate comparisons, and ranked recommendations that are transparent, auditable, and suitable for real-world hiring workflows.
Beyond rubric-based analysis, we introduce an LLM-Driven Active Listwise Tournament mechanism for candidate ranking. Instead of noisy pairwise comparisons or inconsistent independent scoring, the LLM ranks small candidate subsets (mini-tournaments), and these listwise permutations are aggregated using a Plackett-Luce model. An active-learning loop selects the most informative subsets, producing globally coherent and sample-efficient rankings. This adaptation of listwise LLM preference modeling (previously explored in financial asset ranking) provides a principled and highly interpretable methodology for large-scale candidate ranking in talent acquisition. 

---
# LITTA: Late-Interaction and Test-Time Alignment for Visually-Grounded Multimodal Retrieval 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.26683)  

**Abstract**: Retrieving relevant evidence from visually rich documents such as textbooks, technical reports, and manuals is challenging due to long context, complex layouts, and weak lexical overlap between user questions and supporting pages. We propose LITTA, a query-expansion-centric retrieval framework for evidence page retrieval that improves multimodal document retrieval without retriever retraining. Given a user query, LITTA generates complementary query variants using a large language model and retrieves candidate pages for each variant using a frozen vision retriever with late-interaction scoring. Candidates from expanded queries are then aggregated through reciprocal rank fusion to improve evidence coverage and reduce sensitivity to any single phrasing. This simple test-time strategy significantly improves retrieval robustness while remaining compatible with existing multimodal embedding indices. We evaluate LITTA on visually grounded document retrieval tasks across three domains: computer science, pharmaceuticals, and industrial manuals. Multi-query retrieval consistently improves top-k accuracy, recall, and MRR compared to single-query retrieval, with particularly large gains in domains with high visual and semantic variability. Moreover, the accuracy-efficiency trade-off is directly controllable by the number of query variants, making LITTA practical for deployment under latency constraints. These results demonstrate that query expansion provides a simple yet effective mechanism for improving visually grounded multimodal retrieval. 

---
# SRAG: RAG with Structured Data Improves Vector Retrieval 

**Authors**: Shalin Shah, Srikanth Ryali, Ramasubbu Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2603.26670)  

**Abstract**: Retrieval Augmented Generation (RAG) provides the necessary informational grounding to LLMs in the form of chunks retrieved from a vector database or through web search. RAG could also use knowledge graph triples as a means of providing factual information to an LLM. However, the retrieval is only based on representational similarity between a question and the contents. The performance of RAG depends on the numeric vector representations of the query and the chunks. To improve these representations, we propose Structured RAG (SRAG), which adds structured information to a query as well as the chunks in the form of topics, sentiments, query and chunk types (e.g., informational, quantitative), knowledge graph triples and semantic tags. Experiments indicate that this method significantly improves the retrieval process. Using GPT-5 as an LLM-as-a-judge, results show that the method improves the score given to answers in a question answering system by 30% (p-value = 2e-13) (with tighter bounds). The strongest improvement is in comparative, analytical and predictive questions. The results suggest that our method enables broader, more diverse, and episodic-style retrieval. Tail risk analysis shows that SRAG attains very large gains more often, with losses remaining minor in magnitude. 

---
# Bridge-RAG: An Abstract Bridge Tree Based Retrieval Augmented Generation Algorithm With Cuckoo Filter 

**Authors**: Zihang Li, Wenjun Liu, Yikun Zong, Jiawen Tao, Siying Dai, Songcheng Ren, Zirui Liu, Yanbing Jiang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26668)  

**Abstract**: As an important paradigm for enhancing the generation quality of Large Language Models (LLMs), retrieval-augmented generation (RAG) faces the two challenges regarding retrieval accuracy and computational efficiency. This paper presents a novel RAG framework called Bridge-RAG. To overcome the accuracy challenge, we introduce the concept of abstract to bridge query entities and document chunks, providing robust semantic understanding. We organize the abstracts into a tree structure and design a multi-level retrieval strategy to ensure the inclusion of sufficient contextual information. To overcome the efficiency challenge, we introduce the improved Cuckoo Filter, an efficient data structure supporting rapid membership queries and updates, to accelerate entity location during the retrieval process. We design a block linked list structure and an entity temperature-based sorting mechanism to improve efficiency from the aspects of spatial and temporal locality. Extensive experiments show that Bridge-RAG achieves around 15.65% accuracy improvement and reduces 10x to 500x retrieval time compared to other RAG frameworks. 

---
